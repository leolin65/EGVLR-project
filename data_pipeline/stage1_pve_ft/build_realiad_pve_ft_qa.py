# [Stage I: PVE-FT] Real-IAD QA converter.
# Originally: Stage1_Dataset/read_realiad.py
# Converts real Real-IAD defect samples (not synthetic) into the same
# detection-QA format as build_synthetic_pve_ft_qa.py, to mix real defects
# in alongside synthetic ones for Stage I.

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

try:
    import cv2
except Exception:
    cv2 = None

"""
Run command:

python build_realiad_pve_ft_qa.py \
  --input /home/psmc2/project/dataset/Stage1_Dataset/realiad_samples_gt.jsonl \
  --src_dataset_root /home/psmc2/project/dataset \
  --stage_root /home/psmc2/project/dataset/Stage1_Dataset \
  --output /home/psmc2/project/dataset/Stage1_Dataset/realiad_stage1.json
"""

QUESTION_TEMPLATE = """<image>\n<image>
Anomaly Detection:

The first image is the test image.
The second image is a normal reference image.

Compared with the normal reference image, is there any defect in the test image?

{options}

Answer using exactly:
<think>
<evidence>...</evidence>
<logic>...</logic>
</think>
<answer>A/B</answer><location>[[x1,y1,x2,y2],...]</location>

For normal images, use exactly:
<think>
<evidence>...</evidence>
<logic>...</logic>
</think>
<answer>A/B</answer><location>[]</location>"""


# Same programmatic, visual-evidence-only rationale style as
# build_synthetic_pve_ft_qa.py's PVE-FT think block (real Real-IAD defects,
# no domain-knowledge claims).
def build_detection_think(is_anomaly: bool) -> Dict[str, str]:
    if not is_anomaly:
        return {
            "evidence": (
                "Compared with the normal reference image, the test image "
                "shows no visible deviation in texture, shape, or structure."
            ),
            "logic": (
                "Since no localized abnormal region is observed relative to "
                "the normal reference, the test image appears normal."
            ),
        }

    return {
        "evidence": (
            "Compared with the normal reference image, a localized region in "
            "the test image visibly differs from the reference, consistent "
            "with a physical surface or structural defect."
        ),
        "logic": (
            "This localized visual deviation from the normal reference "
            "indicates the presence of a defect in the test image."
        ),
    }


def load_records(path: str) -> List[Dict[str, Any]]:
    """
    支援：
    1. JSON array:
       [{...}, {...}]

    2. JSONL:
       {...}
       {...}

    3. 多個 JSON object 直接接在一起：
       {...}
       {...}
    """
    text = Path(path).read_text(encoding="utf-8").strip()

    if not text:
        return []

    # Case 1: standard JSON
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
    jsonl_ok = True

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            jsonl_ok = False
            break

    if jsonl_ok and records:
        return records

    # Case 3: concatenated JSON objects
    decoder = json.JSONDecoder()
    idx = 0
    records = []

    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1

        if idx >= len(text):
            break

        obj, end = decoder.raw_decode(text, idx)

        if isinstance(obj, dict):
            records.append(obj)
        elif isinstance(obj, list):
            records.extend(obj)
        else:
            raise ValueError(f"Unsupported JSON object type: {type(obj)}")

        idx = end

        while idx < len(text) and text[idx].isspace():
            idx += 1

        if idx < len(text) and text[idx] == ",":
            idx += 1

    return records


def normalize_realiad_relative_path(
    path_str: Optional[str],
    src_dataset_root: Path,
    realiad_name: str = "REALIAD",
) -> Optional[str]:
    """
    統一轉成：
    REALIAD/xxx/xxx.jpg

    支援：
    REALIAD/xxx.jpg
    xxx/xxx.jpg
    /home/psmc2/project/dataset/REALIAD/xxx.jpg
    """
    if not path_str:
        return None

    s = str(path_str).replace("\\", "/").strip()
    p = Path(s)

    if p.is_absolute():
        try:
            rel = p.relative_to(src_dataset_root).as_posix()
        except ValueError:
            try:
                rel_inside = p.relative_to(src_dataset_root / realiad_name).as_posix()
                rel = f"{realiad_name}/{rel_inside}"
            except ValueError:
                raise ValueError(f"Path is outside source dataset root: {path_str}")
    else:
        rel = s.lstrip("./")

        if not rel.startswith(f"{realiad_name}/"):
            rel = f"{realiad_name}/{rel}"

    return rel


def copy_one_file(
    rel_path: Optional[str],
    src_dataset_root: Path,
    stage_root: Path,
    overwrite: bool = False,
) -> bool:
    if not rel_path:
        return False

    src = src_dataset_root / rel_path
    dst = stage_root / rel_path

    if not src.exists():
        print(f"[WARN] Missing source file: {src}")
        return False

    dst.parent.mkdir(parents=True, exist_ok=True)

    if overwrite or not dst.exists():
        shutil.copy2(src, dst)

    return True


def mask_to_bboxes(
    mask_path: Path,
    threshold: int = 0,
    min_area: int = 1,
) -> List[List[int]]:
    """
    從 mask 算異常區域 bbox。

    回傳：
    [[x1, y1, x2, y2], ...]

    座標是針對第一張 test image。
    x2, y2 使用 inclusive 座標。
    """
    if not mask_path.exists():
        return []

    mask = Image.open(mask_path).convert("L")
    arr = np.array(mask)

    binary = (arr > threshold).astype(np.uint8)

    if binary.sum() == 0:
        return []

    # 有 cv2：找 connected components，多個異常區域會有多個 bbox
    if cv2 is not None:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )

        bboxes = []

        for label_id in range(1, num_labels):
            x, y, w, h, area = stats[label_id]

            if int(area) < min_area:
                continue

            x1 = int(x)
            y1 = int(y)
            x2 = int(x + w - 1)
            y2 = int(y + h - 1)

            bboxes.append([x1, y1, x2, y2])

        bboxes.sort(key=lambda box: (box[1], box[0]))
        return bboxes

    # 沒有 cv2：fallback 成一個大 bbox
    ys, xs = np.where(binary > 0)

    if len(xs) == 0 or len(ys) == 0:
        return []

    return [[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]]


def has_path_part(rel_path: str, part: str) -> bool:
    return part in rel_path.replace("\\", "/").split("/")


def infer_is_anomaly(
    image_rel: str,
    bboxes: List[List[int]],
) -> bool:
    """
    判斷第一張 test image 是否異常。

    優先順序：
    1. mask 有 bbox -> 異常
    2. 路徑包含 /NG/ -> 異常
    3. 路徑包含 /OK/ -> 正常
    4. 其他情況預設正常
    """
    if len(bboxes) > 0:
        return True

    if has_path_part(image_rel, "NG"):
        return True

    if has_path_part(image_rel, "OK"):
        return False

    return False


def make_id(
    item_type: str,
    image_rel: str,
    normal_rel: str,
    counter: int,
) -> str:
    raw = f"{item_type}|{image_rel}|{normal_rel}"
    h = hashlib.md5(raw.encode("utf-8")).hexdigest()[:8]

    clean_item = item_type.replace("/", "_").replace(" ", "_")

    return f"realiad_{clean_item}_{counter:06d}_{h}"


def build_question_answer(
    is_anomaly: bool,
    bboxes: List[List[int]],
    rng: random.Random,
) -> Dict[str, str]:
    """
    A/B Yes/No 順序隨機。
    Answer 根據第一張 test image 是否異常決定。
    location 也是第一張 test image 的 mask bbox。
    """
    choices = [
        ("No", False),
        ("Yes", True),
    ]

    rng.shuffle(choices)

    option_lines = []
    answer_letter = None

    for idx, (choice_text, choice_is_anomaly) in enumerate(choices):
        letter = chr(ord("A") + idx)
        option_lines.append(f"{letter}. {choice_text}.")

        if choice_is_anomaly == is_anomaly:
            answer_letter = letter

    if answer_letter is None:
        raise RuntimeError("Failed to build answer letter.")

    options_text = "\n".join(option_lines)
    question = QUESTION_TEMPLATE.format(options=options_text)

    location = bboxes if is_anomaly else []

    think = build_detection_think(is_anomaly=is_anomaly)
    think_block = (
        "<think>\n"
        f"<evidence>{think['evidence']}</evidence>\n"
        f"<logic>{think['logic']}</logic>\n"
        "</think>\n"
    )

    answer = (
        think_block
        + f"<answer>{answer_letter}</answer>"
        f"<location>{json.dumps(location, ensure_ascii=False, separators=(',', ':'))}</location>"
    )

    return {
        "question": question,
        "answer": answer,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Input annotation json/jsonl file.",
    )

    parser.add_argument(
        "--src_dataset_root",
        default="/home/psmc2/project/dataset",
        help="Source dataset root. REALIAD should be under this folder.",
    )

    parser.add_argument(
        "--stage_root",
        default="/home/psmc2/project/dataset/Stage1_Dataset",
        help="Output Stage1 dataset root.",
    )

    parser.add_argument(
        "--output",
        default="/home/psmc2/project/dataset/Stage1_Dataset/realiad_stage1_qa_two_images.json",
        help="Output QA json path.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for Yes/No option order.",
    )

    parser.add_argument(
        "--mask_threshold",
        type=int,
        default=0,
        help="Mask pixel > threshold will be treated as anomaly.",
    )

    parser.add_argument(
        "--min_area",
        type=int,
        default=1,
        help="Minimum connected component area for bbox.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite copied files if already exist.",
    )

    parser.add_argument(
        "--image_key",
        default="image",
        choices=["image", "images"],
        help="Output key for image list. Default is image.",
    )

    args = parser.parse_args()

    src_dataset_root = Path(args.src_dataset_root)
    stage_root = Path(args.stage_root)
    output_path = Path(args.output)

    rng = random.Random(args.seed)

    records = load_records(args.input)

    qa_items = []

    copied_files = 0
    skipped_mmad = 0
    skipped_invalid = 0
    skipped_missing_pair = 0

    sample_counter = 0

    for rec_idx, rec in enumerate(records):
        item_type = str(rec.get("item_type", "")).strip()

        # 跳過 item_type == "mmad"
        if item_type.lower() == "mmad":
            skipped_mmad += 1
            continue

        try:
            image_rel = normalize_realiad_relative_path(
                rec.get("image_path"),
                src_dataset_root,
            )

            mask_rel = normalize_realiad_relative_path(
                rec.get("mask_path"),
                src_dataset_root,
            )

            normal_rel = normalize_realiad_relative_path(
                rec.get("normal_image_path"),
                src_dataset_root,
            )

        except Exception as e:
            skipped_invalid += 1
            print(f"[WARN] Skip record {rec_idx}: {e}")
            continue

        # 兩張圖都是必要條件
        # image_path = 第一張待測試圖
        # normal_image_path = 第二張正常參考圖
        if not image_rel or not normal_rel:
            skipped_missing_pair += 1
            print(
                f"[WARN] Skip record {rec_idx}: missing image_path or normal_image_path."
            )
            continue

        # 複製 image_path、mask_path、normal_image_path
        for rel in [image_rel, mask_rel, normal_rel]:
            if rel:
                ok = copy_one_file(
                    rel_path=rel,
                    src_dataset_root=src_dataset_root,
                    stage_root=stage_root,
                    overwrite=args.overwrite,
                )

                if ok:
                    copied_files += 1

        # mask 是對應第一張 test image 的異常標註
        bboxes = []

        if mask_rel:
            mask_abs = src_dataset_root / mask_rel

            bboxes = mask_to_bboxes(
                mask_abs,
                threshold=args.mask_threshold,
                min_area=args.min_area,
            )

        # Answer 看第一張 test image 是否異常
        is_anomaly = infer_is_anomaly(
            image_rel=image_rel,
            bboxes=bboxes,
        )

        qa = build_question_answer(
            is_anomaly=is_anomaly,
            bboxes=bboxes,
            rng=rng,
        )

        # 圖片順序：
        # 第一張：test image
        # 第二張：normal reference image
        item = {
            "id": make_id(
                item_type=item_type,
                image_rel=image_rel,
                normal_rel=normal_rel,
                counter=sample_counter,
            ),
            args.image_key: [
                image_rel,
                normal_rel,
            ],
            "conversations": [
                {
                    "from": "human",
                    "value": qa["question"],
                },
                {
                    "from": "gpt",
                    "value": qa["answer"],
                },
            ],
        }

        qa_items.append(item)
        sample_counter += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(qa_items, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Done.")
    print(f"Total input records: {len(records)}")
    print(f"Skipped item_type == mmol/mmad: {skipped_mmad}")
    print(f"Skipped invalid records: {skipped_invalid}")
    print(f"Skipped missing image pair: {skipped_missing_pair}")
    print(f"Copied files count: {copied_files}")
    print(f"Generated QA samples: {len(qa_items)}")
    print(f"Output json: {output_path}")


if __name__ == "__main__":
    main()