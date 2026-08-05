# [Stage III: GS-DPO] Domain-knowledge MCQ converter (train subset).
# Originally: Stage3_Dataset/read_judo.py
# Same conversion as stage2_kg_it/convert_domain_knowledge_qa.py, plus
# --max_samples/--sample_mode stratified sampling.

import os
import json
import argparse
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple
from PIL import Image, ImageOps

"""
Example:

python convert_domain_knowledge_qa.py \
  --input_json /home/psmc2/project/dataset/Stage3_Dataset/judo_domain_qa_mcq_dk_ultra_strict_flat.jsonl \
  --src_root /home/psmc2/project/dataset/MMAD \
  --dst_root /home/psmc2/project/dataset/Stage3_Dataset \
  --output_json /home/psmc2/project/dataset/Stage3_Dataset/train_stage3_judo_300.json \
  --long_edge 256 \
  --skip_missing \
  --max_samples 300 \
  --sample_mode stratified \
  --seed 42
"""


def read_json_any(path: str) -> List[Dict[str, Any]]:
    """
    支援：
    1. JSON array: [ {...}, {...} ]
    2. JSONL: 每行一個 JSON
    3. concatenated JSON objects: {...} {...} {...}
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        return []

    # Case 1: normal JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            return [obj]
    except json.JSONDecodeError:
        pass

    # Case 2: JSONL
    records = []
    ok_jsonl = True
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            ok_jsonl = False
            break

    if ok_jsonl and records:
        return records

    # Case 3: concatenated JSON objects
    decoder = json.JSONDecoder()
    records = []
    idx = 0
    n = len(text)

    while idx < n:
        while idx < n and text[idx].isspace():
            idx += 1
        if idx >= n:
            break

        obj, end = decoder.raw_decode(text, idx)
        if not isinstance(obj, dict):
            raise ValueError(f"Expected dict JSON object at position {idx}, got {type(obj)}")
        records.append(obj)
        idx = end

    return records


def resize_long_edge_256(src_path: Path, dst_path: Path, long_edge: int = 256) -> Tuple[int, int]:
    """
    將圖片縮放成長邊 long_edge，保持比例。
    回傳輸出圖片大小: (width, height)
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(src_path) as img:
        img = ImageOps.exif_transpose(img)

        w, h = img.size
        if w <= 0 or h <= 0:
            raise ValueError(f"Invalid image size: {src_path}, size={img.size}")

        scale = long_edge / max(w, h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        suffix = dst_path.suffix.lower()
        if suffix in [".jpg", ".jpeg"] and img.mode in ["RGBA", "LA", "P"]:
            img = img.convert("RGB")

        img.save(dst_path)

    return new_w, new_h


def build_conversations_if_missing(item: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    如果原資料沒有 conversations，就用 question/options/answer_text 建一個。
    正常情況你的資料已經有 conversations，會直接保留原本 QA。
    """
    question = item.get("question", "")
    options = item.get("options", {})
    answer = item.get("answer", "")
    answer_text = item.get("answer_text", "")

    choices_text = ""
    for k in ["A", "B", "C", "D"]:
        if k in options:
            choices_text += f"{k}. {options[k]}\n"

    human_value = (
        "<image>\n"
        "Domain Knowledge MCQ:\n\n"
        "This image is a normal reference for the object category. "
        "Use it as visual context only; do not infer that this normal reference image contains a defect.\n\n"
        "Question:\n"
        f"{question}\n\n"
        "Choices:\n"
        f"{choices_text}\n"
        "Answer using exactly:\n"
        "<think>\n"
        "<evidence>...</evidence>\n"
        "<logic>...</logic>\n"
        "</think>\n"
        "<answer>A/B/C/D</answer>"
    )

    gpt_value = (
        "<think>\n"
        "<evidence>The image provides a normal visual reference for the object category.</evidence>\n"
        f"<logic>Based on the domain question and choices, option {answer} is correct because it states: {answer_text}</logic>\n"
        "</think>\n"
        f"<answer>{answer}</answer>"
    )

    return [
        {"from": "human", "value": human_value},
        {"from": "gpt", "value": gpt_value},
    ]


def validate_no_location_and_think(item_id: str, conversations: List[Dict[str, str]]) -> List[str]:
    warnings = []

    if not conversations or len(conversations) < 2:
        warnings.append(f"{item_id}: conversations length < 2")
        return warnings

    gpt_value = conversations[-1].get("value", "")

    if "<location>" in gpt_value or "</location>" in gpt_value:
        warnings.append(f"{item_id}: GPT answer contains <location>, please check.")

    required_tags = [
        "<think>",
        "<evidence>",
        "</evidence>",
        "<logic>",
        "</logic>",
        "</think>",
        "<answer>",
        "</answer>",
    ]

    for tag in required_tags:
        if tag not in gpt_value:
            warnings.append(f"{item_id}: missing tag {tag}")

    return warnings


def get_stratify_key(item: Dict[str, Any]) -> Tuple[str, ...]:
    """
    給 stratified sampling 用，避免抽樣太集中在同一類。

    DK 資料通常比較適合用：
    dataset / object_category / category / answer / task_type / question_type
    """
    dataset = str(item.get("dataset", "unknown"))
    object_category = str(item.get("object_category", item.get("category", "unknown")))
    category = str(item.get("category", "unknown"))
    answer = str(item.get("answer", "unknown"))
    task_type = str(item.get("task_type", "domain_knowledge"))
    question_type = str(item.get("question_type", item.get("sample_type", "unknown")))

    return (
        dataset,
        object_category,
        category,
        answer,
        task_type,
        question_type,
    )


def sample_records(
    records: List[Dict[str, Any]],
    max_samples: int = 0,
    sample_mode: str = "first",
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    sample_mode:
    - first      : 取前 max_samples 筆
    - random     : 隨機抽 max_samples 筆
    - stratified : 依照 dataset/category/answer/task_type 做分散抽樣

    max_samples <= 0 表示不限制，使用全部資料。
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

        for item in records:
            key = get_stratify_key(item)
            groups.setdefault(key, []).append(item)

        # 每組內部先洗牌
        for key in groups:
            rng.shuffle(groups[key])

        # group 順序也洗牌
        group_keys = list(groups.keys())
        rng.shuffle(group_keys)

        selected = []
        group_index = 0

        # round-robin 從不同 group 抽，避免集中在同類
        while len(selected) < max_samples and group_keys:
            key = group_keys[group_index]
            group = groups[key]

            if group:
                selected.append(group.pop())

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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_json",
        required=True,
        help="原始 Domain Knowledge MCQ JSON / JSONL，例如 judo_domain_qa_mcq.jsonl",
    )
    parser.add_argument(
        "--src_root",
        default="/home/psmc2/project/dataset/MMAD",
        help="原圖資料集 root",
    )
    parser.add_argument(
        "--dst_root",
        default="/home/psmc2/project/dataset/Stage2_Dataset",
        help="輸出圖片 root",
    )
    parser.add_argument(
        "--output_json",
        default="/home/psmc2/project/dataset/Stage2_Dataset/judo_domain_knowledge_mcq_stage2.json",
        help="輸出的 QA JSON",
    )
    parser.add_argument(
        "--long_edge",
        type=int,
        default=256,
        help="輸出圖片長邊大小",
    )
    parser.add_argument(
        "--skip_missing",
        action="store_true",
        help="遇到缺圖就跳過，不中斷",
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="最多輸出幾筆。0 表示不限制。",
    )

    parser.add_argument(
        "--sample_mode",
        type=str,
        default="first",
        choices=["first", "random", "stratified"],
        help="抽樣模式：first / random / stratified",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random / stratified 抽樣用的 seed，可重現。",
    )

    args = parser.parse_args()

    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)
    output_json = Path(args.output_json)

    records = read_json_any(args.input_json)
    total_input_records = len(records)

    records = sample_records(
        records,
        max_samples=args.max_samples,
        sample_mode=args.sample_mode,
        seed=args.seed,
    )

    output_items = []
    missing_images = []
    warnings = []

    for idx, item in enumerate(records):
        item_id = item.get("id", f"sample_{idx:06d}")

        image_rel = item.get("image", None)

        # 你的 DK 資料應該只有一張 image
        if isinstance(image_rel, list):
            if len(image_rel) != 1:
                warnings.append(f"{item_id}: image is list with length {len(image_rel)}, skip.")
                continue
            image_rel = image_rel[0]

        if not isinstance(image_rel, str) or not image_rel:
            warnings.append(f"{item_id}: invalid image field, skip.")
            continue

        image_rel = image_rel.replace("\\", "/").lstrip("/")

        src_img = src_root / image_rel
        dst_img = dst_root / image_rel

        if not src_img.exists():
            msg = f"{item_id}: missing image: {src_img}"
            missing_images.append(msg)
            if args.skip_missing:
                continue
            raise FileNotFoundError(msg)

        try:
            resize_long_edge_256(src_img, dst_img, args.long_edge)
        except Exception as e:
            msg = f"{item_id}: failed to resize {src_img}: {repr(e)}"
            warnings.append(msg)
            if args.skip_missing:
                continue
            raise

        conversations = item.get("conversations", None)
        if not conversations:
            conversations = build_conversations_if_missing(item)

        warnings.extend(validate_no_location_and_think(item_id, conversations))

        # 這裡就是你要的 QA 格式：一張 image，不是 list
        output_items.append(
            {
                "id": item_id,
                "image": image_rel,
                "conversations": conversations,
            }
        )

        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1} selected samples")

    output_json.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output_items, f, ensure_ascii=False, indent=2)

    log_dir = output_json.parent

    if missing_images:
        with open(log_dir / "missing_images.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(missing_images) + "\n")

    if warnings:
        with open(log_dir / "convert_warnings.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(warnings) + "\n")

    print("=" * 60)
    print("Done.")
    print(f"Original input records : {total_input_records}")
    print(f"Selected records       : {len(records)}")
    print(f"Output records         : {len(output_items)}")
    print(f"Max samples            : {args.max_samples}")
    print(f"Sample mode            : {args.sample_mode}")
    print(f"Seed                   : {args.seed}")
    print(f"Output JSON            : {output_json}")
    print(f"Output image root      : {dst_root}")

    if missing_images:
        print(f"Missing images         : {len(missing_images)}")
        print(f"See                    : {log_dir / 'missing_images.txt'}")

    if warnings:
        print(f"Warnings               : {len(warnings)}")
        print(f"See                    : {log_dir / 'convert_warnings.txt'}")

    print("=" * 60)


if __name__ == "__main__":
    main()