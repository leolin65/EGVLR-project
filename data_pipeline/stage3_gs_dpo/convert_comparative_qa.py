# [Stage III: GS-DPO] Comparative-inspection QA converter (train subset).
# Originally: Stage3_Dataset/read_comp.py
# Same conversion as stage2_kg_it/convert_comparative_qa.py, plus
# --max_samples/--sample_mode stratified sampling to build a smaller,
# class-balanced GRPO training subset.

import json
import argparse
import re
import random
from pathlib import Path
from PIL import Image

"""
Example:

python convert_comparative_qa.py \
  --input /home/psmc2/project/dataset/Stage3_Dataset/comparative_inspection_stage2_mix_15pct_normal_flat.jsonl \
  --output comp_stage3_random600.json \
  --stage2_root /home/psmc2/project/dataset/Stage3_Dataset \
  --mmad_root /home/psmc2/project/dataset/MMAD \
  --long_side 256 \
  --max_samples 600 \
  --sample_mode stratified \
  --seed 42
"""


def load_json_or_jsonl(path):
    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        return []

    # JSON array
    if text[0] == "[":
        return json.loads(text)

    # JSONL
    records = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def save_json(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def resize_long_side_256(src_path, dst_path, long_side=256):
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(src_path) as img:
        img = img.convert("RGB")

        w, h = img.size
        scale = long_side / max(w, h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        suffix = dst_path.suffix.lower()
        if suffix in [".jpg", ".jpeg"]:
            img.save(dst_path, quality=95)
        else:
            img.save(dst_path)


def ensure_image_exists(rel_path, stage2_root, mmad_root, long_side=256):
    """
    If Stage2_Dataset/rel_path does not exist,
    copy from MMAD/rel_path and resize long side to 256.
    """
    if rel_path is None:
        return False, "empty image path"

    rel_path = str(rel_path).replace("\\", "/").lstrip("/")
    dst_path = Path(stage2_root) / rel_path

    if dst_path.exists():
        return True, "exists"

    src_path = Path(mmad_root) / rel_path

    if not src_path.exists():
        return False, f"missing source: {src_path}"

    resize_long_side_256(src_path, dst_path, long_side=long_side)
    return True, "copied_resized"


def get_image_pair(record):
    """
    Target format wants:
    first image  = query/test image
    second image = normal reference image
    """
    normal_ref = record.get("normal_reference_image")
    query_img = record.get("query_image")

    if normal_ref and query_img:
        return [query_img, normal_ref]

    images = record.get("image", [])

    if not isinstance(images, list):
        images = [images]

    if len(images) < 2:
        return images

    image_order = record.get("image_order", "").lower()

    # Old format example:
    # first=normal reference image; second=query image
    if "first=normal" in image_order and "second=query" in image_order:
        return [images[1], images[0]]

    if "first=query" in image_order or "first=test" in image_order:
        return [images[0], images[1]]

    # default for your current source format
    # old image usually = [normal_reference, query]
    return [images[1], images[0]]


def format_options(options):
    """
    options:
    {
      "A": "Yes.",
      "B": "No."
    }
    ->
    A. Yes.
    B. No.
    """
    if not isinstance(options, dict):
        return ""

    lines = []
    for key in sorted(options.keys()):
        lines.append(f"{key}. {options[key]}")
    return "\n".join(lines)


def get_answer_choices(options):
    if isinstance(options, dict) and options:
        return "/".join(sorted(options.keys()))

    return "A/B/C/D"


def extract_tag(text, tag):
    if not isinstance(text, str):
        return None

    pattern = re.compile(
        rf"<{tag}>\s*(.*?)\s*</{tag}>",
        re.IGNORECASE | re.DOTALL
    )
    m = pattern.search(text)
    return m.group(1).strip() if m else None


def is_normal_sample(record):
    sample_type = str(record.get("sample_type", "")).lower()
    condition = str(record.get("condition", "")).lower()
    answer_text = str(record.get("answer_text", "")).lower()

    if "normal" in sample_type:
        return True

    if condition == "good":
        return True

    if "no clear localized defect" in answer_text:
        return True

    return False


def build_human_prompt(record):
    task_type = record.get("task_type", "Anomaly Detection")
    question = record.get("question", "")
    options = record.get("options", {})
    options_text = format_options(options)
    answer_choices = get_answer_choices(options)

    prompt = (
        "<image>\n"
        "<image>\n"
        f"{task_type}:\n\n"
        "The first image is the test image.\n"
        "The second image is a normal reference image.\n\n"
        "Use the normal reference image as comparison context.\n\n"
        "Question:\n"
        f"{question}\n\n"
        f"{options_text}\n\n"
        "Answer using exactly:\n"
        "<think>\n"
        "<evidence>...</evidence>\n"
        "<logic>...</logic>\n"
        "</think>\n"
        f"<answer>{answer_choices}</answer><location>[[x1,y1,x2,y2],...]</location>\n\n"
        "For normal images, use exactly:\n"
        "<think>\n"
        "<evidence>...</evidence>\n"
        "<logic>...</logic>\n"
        "</think>\n"
        f"<answer>{answer_choices}</answer><location>[]</location>"
    )

    return prompt


def normalize_location(location):
    """
    Make location safe for output.

    Accepts:
    - list: [[x1,y1,x2,y2], ...]
    - string: "[[x1,y1,x2,y2], ...]"
    - empty / None

    Returns string.
    """
    if location is None:
        return "[]"

    if isinstance(location, list):
        return json.dumps(location, ensure_ascii=False)

    location = str(location).strip()

    if location == "":
        return "[]"

    return location


def build_gpt_answer(record):
    answer = str(record.get("answer", "")).strip().upper()
    answer_text = record.get("answer_text", "")

    old_gpt = ""
    conversations = record.get("conversations", [])
    if isinstance(conversations, list) and len(conversations) >= 2:
        old_gpt = conversations[1].get("value", "")

    # fallback: if answer field is missing, try to extract from old gpt
    if not answer:
        old_answer = extract_tag(old_gpt, "answer")
        if old_answer:
            answer = old_answer.strip().upper()

    evidence = extract_tag(old_gpt, "evidence")
    logic = extract_tag(old_gpt, "logic")

    if not evidence:
        evidence = (
            f"The query image is compared with the normal reference image. "
            f"The visual evidence supports answer {answer}: {answer_text}."
        )

    if not logic:
        logic = (
            f"Based on the comparison with the normal reference image, "
            f"option {answer} is the best answer."
        )

    if is_normal_sample(record):
        location = "[]"
    else:
        location = normalize_location(record.get("location", "[]"))

    gpt_value = (
        "<think>\n"
        f"<evidence>{evidence}</evidence>\n"
        f"<logic>{logic}</logic>\n"
        "</think>\n"
        f"<answer>{answer}</answer><location>{location}</location>"
    )

    return gpt_value


def get_stratify_key(record):
    """
    Used for diversity sampling.

    This tries to avoid selecting too many samples from the same:
    dataset / object category / condition / answer / task type.
    """
    dataset = str(record.get("dataset", "unknown"))
    object_category = str(record.get("object_category", "unknown"))
    condition = str(record.get("condition", "unknown"))
    sample_type = str(record.get("sample_type", "unknown"))
    answer = str(record.get("answer", "unknown"))
    task_type = str(record.get("task_type", "unknown"))

    return (
        dataset,
        object_category,
        condition,
        sample_type,
        answer,
        task_type,
    )


def sample_records(records, max_samples=0, sample_mode="first", seed=42):
    """
    sample_mode:
    - first      : take first max_samples records
    - random     : random sample max_samples records
    - stratified : balanced random sample by dataset/category/condition/answer/task_type

    max_samples <= 0 means use all records.
    """
    if max_samples is None or max_samples <= 0:
        return records

    if max_samples >= len(records):
        return records

    rng = random.Random(seed)

    if sample_mode == "first":
        return records[:max_samples]

    if sample_mode == "random":
        return rng.sample(records, max_samples)

    if sample_mode == "stratified":
        groups = {}

        for record in records:
            key = get_stratify_key(record)
            groups.setdefault(key, []).append(record)

        # shuffle each group
        for key in groups:
            rng.shuffle(groups[key])

        # shuffle group order
        group_keys = list(groups.keys())
        rng.shuffle(group_keys)

        selected = []
        group_index = 0

        # round-robin sampling across groups
        while len(selected) < max_samples and group_keys:
            key = group_keys[group_index]
            group = groups[key]

            if group:
                selected.append(group.pop())

            # remove empty group
            if not group:
                group_keys.pop(group_index)
                if not group_keys:
                    break
                group_index = group_index % len(group_keys)
            else:
                group_index = (group_index + 1) % len(group_keys)

        rng.shuffle(selected)
        return selected

    raise ValueError(f"Unknown sample_mode: {sample_mode}")


def convert_record(record, stage2_root, mmad_root, long_side=256):
    images = get_image_pair(record)

    for img_path in images:
        ok, msg = ensure_image_exists(
            img_path,
            stage2_root=stage2_root,
            mmad_root=mmad_root,
            long_side=long_side
        )

        if not ok:
            print(f"[WARNING] image not found: {img_path} | {msg}")

    new_record = {
        "id": record.get("id", record.get("original_id", "")),
        "image": images,
        "conversations": [
            {
                "from": "human",
                "value": build_human_prompt(record)
            },
            {
                "from": "gpt",
                "value": build_gpt_answer(record)
            }
        ]
    }

    return new_record


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Input JSONL or JSON file"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON file"
    )

    parser.add_argument(
        "--stage2_root",
        default="/home/psmc2/project/dataset/Stage2_Dataset",
        help="Target Stage2_Dataset root"
    )

    parser.add_argument(
        "--mmad_root",
        default="/home/psmc2/project/dataset/MMAD",
        help="Source MMAD root"
    )

    parser.add_argument(
        "--long_side",
        type=int,
        default=256,
        help="Resize image long side to this value"
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Maximum number of samples to output. 0 means no limit."
    )

    parser.add_argument(
        "--sample_mode",
        type=str,
        default="first",
        choices=["first", "random", "stratified"],
        help="Sampling mode: first, random, or stratified."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for random or stratified sampling."
    )

    args = parser.parse_args()

    records = load_json_or_jsonl(args.input)
    total_input = len(records)

    records = sample_records(
        records,
        max_samples=args.max_samples,
        sample_mode=args.sample_mode,
        seed=args.seed
    )

    converted = []

    for i, record in enumerate(records):
        new_record = convert_record(
            record,
            stage2_root=args.stage2_root,
            mmad_root=args.mmad_root,
            long_side=args.long_side
        )
        converted.append(new_record)

        if (i + 1) % 1000 == 0:
            print(f"Converted {i + 1} samples")

    save_json(converted, args.output)

    print("=" * 60)
    print(f"Original input samples : {total_input}")
    print(f"Used input samples     : {len(records)}")
    print(f"Output samples         : {len(converted)}")
    print(f"Sample mode            : {args.sample_mode}")
    print(f"Seed                   : {args.seed}")
    print(f"Saved to               : {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()