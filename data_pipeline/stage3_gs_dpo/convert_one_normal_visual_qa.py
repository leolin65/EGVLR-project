# [Stage III: GS-DPO] One-normal visual QA converter.
# Originally: Stage3_Dataset/read_mmad.py
# Same as stage2_kg_it/convert_one_normal_visual_qa.py; its output was
# named 'test_stage3.json' upstream (same non-eval, training-source caveat
# as Stage II — see that file's header note).

import os
import re
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from PIL import Image, ImageOps

"""
python convert_one_normal_visual_qa.py \
  --input_json /home/psmc2/project/dataset/Stage3_Dataset/mmad_train5_with_think_dk_verified_flat.jsonl \
  --src_root /home/psmc2/project/dataset/MMAD \
  --dst_root /home/psmc2/project/dataset/Stage3_Dataset \
  --output_json /home/psmc2/project/dataset/Stage3_Dataset/test_stage3.json \
  --long_edge 256 \
  --skip_missing
"""

IMAGE_EXTS = [
    ".png", ".jpg", ".jpeg", ".bmp",
    ".PNG", ".JPG", ".JPEG", ".BMP"
]


def read_json_any(path: str) -> List[Dict[str, Any]]:
    """
    支援：
    1. JSON array
    2. JSONL
    3. 連續 JSON object: {...}{...}{...}
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        return []

    # JSON array or single object
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            return [obj]
    except json.JSONDecodeError:
        pass

    # JSONL
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

    # concatenated JSON objects
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
            raise ValueError(f"Expected JSON object at position {idx}")

        records.append(obj)
        idx = end

    return records


def is_good_sample(item: Dict[str, Any]) -> bool:
    condition = str(item.get("condition", "")).lower()
    image = str(item.get("image", "")).replace("\\", "/").lower()

    if condition in ["good", "ok", "normal"]:
        return True

    good_tokens = [
        "/good/",
        "/ok/",
        "/normal/",
        "/train/good/",
        "/test/good/",
        "/image/good/",
    ]

    return any(token in image for token in good_tokens)


def resize_long_edge(
    src_path: Path,
    dst_path: Path,
    long_edge: int = 256,
) -> Tuple[int, int, int, int]:
    """
    將圖片縮成長邊 long_edge。
    回傳 old_w, old_h, new_w, new_h
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(src_path) as img:
        img = ImageOps.exif_transpose(img)

        old_w, old_h = img.size
        scale = long_edge / max(old_w, old_h)

        new_w = max(1, int(round(old_w * scale)))
        new_h = max(1, int(round(old_h * scale)))

        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        if dst_path.suffix.lower() in [".jpg", ".jpeg"] and img.mode in ["RGBA", "LA", "P"]:
            img = img.convert("RGB")

        img.save(dst_path)

    return old_w, old_h, new_w, new_h


def find_first_good_reference(src_root: Path, image_rel: str) -> Optional[str]:
    """
    Reference normal image 規則：

    只有 DS-MVTec 特殊處理：
    DS-MVTec/zipper/image/good/015.png
    -> MVTec-AD/zipper/train/good/第一張

    其他 dataset 一般處理：
    VisA/cashew/test/bad/076.JPG
    -> VisA/cashew/train/good/第一張

    MVTec-AD/tile/test/glue_strip/010.png
    -> MVTec-AD/tile/train/good/第一張
    """
    rel = image_rel.replace("\\", "/")
    parts = rel.split("/")

    if len(parts) < 2:
        return None

    dataset = parts[0]
    category = parts[1]

    # 只有 DS-MVTec 特殊：reference 從 MVTec-AD 拿
    if dataset == "DS-MVTec":
        ref_dataset = "MVTec-AD"
    else:
        ref_dataset = dataset

    ref_dir = src_root / ref_dataset / category / "train" / "good"

    if not ref_dir.exists():
        return None

    candidates = []
    for p in sorted(ref_dir.iterdir()):
        if p.is_file() and p.suffix in IMAGE_EXTS:
            candidates.append(p)

    if not candidates:
        return None

    return str(candidates[0].relative_to(src_root)).replace("\\", "/")


def get_mask_candidates(src_root: Path, image_rel: str) -> List[Path]:
    """
    Mask 規則：

    1. DS-MVTec 特殊：
       DS-MVTec/tile/image/glue_strip/010.png
       -> MVTec-AD/tile/ground_truth/glue_strip/010.png
       -> MVTec-AD/tile/ground_truth/glue_strip/010_mask.png

    2. MVTec-LOCO 特殊：
       MVTec-LOCO/breakfast_box/test/structural_anomalies/007.png
       -> MVTec-LOCO/breakfast_box/ground_truth/structural_anomalies/007/000.png

    3. 其他 dataset 一般：
       VisA/cashew/test/bad/076.JPG
       -> VisA/cashew/ground_truth/bad/076.JPG
       -> VisA/cashew/ground_truth/bad/076.png
       -> VisA/cashew/ground_truth/bad/076_mask.png

       MVTec-AD/tile/test/glue_strip/010.png
       -> MVTec-AD/tile/ground_truth/glue_strip/010.png
       -> MVTec-AD/tile/ground_truth/glue_strip/010_mask.png
    """
    rel = image_rel.replace("\\", "/")
    parts = rel.split("/")

    candidates = []

    if len(parts) < 5:
        return candidates

    dataset = parts[0]
    category = parts[1]
    filename = parts[-1]
    stem = Path(filename).stem
    defect_type = parts[-2]

    if defect_type.lower() in ["good", "ok", "normal"]:
        return candidates

    # ------------------------------------------------------------
    # MVTec-LOCO 特殊處理
    # example:
    # MVTec-LOCO/breakfast_box/test/structural_anomalies/007.png
    # -> MVTec-LOCO/breakfast_box/ground_truth/structural_anomalies/007/000.png
    # ------------------------------------------------------------
    if dataset == "MVTec-LOCO":
        gt_folder = src_root / "MVTec-LOCO" / category / "ground_truth" / defect_type / stem

        # 最常見指定 mask
        candidates.append(gt_folder / "000.png")
        candidates.append(gt_folder / "000.PNG")

        # 保險：如果該資料夾裡還有其他 mask，也加入候選
        if gt_folder.exists():
            for p in sorted(gt_folder.iterdir()):
                if p.is_file() and p.suffix in IMAGE_EXTS:
                    candidates.append(p)

        # 去重保序
        unique = []
        seen = set()
        for p in candidates:
            s = str(p)
            if s not in seen:
                unique.append(p)
                seen.add(s)

        return unique

    # ------------------------------------------------------------
    # DS-MVTec 特殊處理
    # DS-MVTec 的 image 來自 DS-MVTec，
    # 但 mask 要去 MVTec-AD 找。
    # ------------------------------------------------------------
    if dataset == "DS-MVTec":
        mask_dataset = "MVTec-AD"
    else:
        mask_dataset = dataset

    gt_dir = src_root / mask_dataset / category / "ground_truth" / defect_type

    # 原副檔名優先
    candidates.append(gt_dir / filename)

    for ext in IMAGE_EXTS:
        candidates.append(gt_dir / f"{stem}{ext}")
        candidates.append(gt_dir / f"{stem}_mask{ext}")

    # 去重保序
    unique = []
    seen = set()
    for p in candidates:
        s = str(p)
        if s not in seen:
            unique.append(p)
            seen.add(s)

    return unique


def find_existing_path(candidates: List[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists():
            return p
    return None


def mask_to_bboxes(
    mask_path: Path,
    output_w: int,
    output_h: int,
    min_area: int = 3,
) -> List[List[int]]:
    """
    將 mask 轉 bbox。
    bbox 座標會對應縮放後的 256 圖片座標。
    """
    with Image.open(mask_path) as m:
        m = ImageOps.exif_transpose(m)
        mask_w, mask_h = m.size
        arr = np.array(m)

    if arr.ndim == 3:
        arr = arr[..., 0]

    binary = arr > 0

    if not binary.any():
        return []

    scale_x = output_w / mask_w
    scale_y = output_h / mask_h

    # 優先用 cv2 拆多個 connected components
    try:
        import cv2

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary.astype(np.uint8),
            connectivity=8,
        )

        boxes = []

        for label_id in range(1, num_labels):
            x, y, w, h, area = stats[label_id]

            if area < min_area:
                continue

            x1 = int(round(x * scale_x))
            y1 = int(round(y * scale_y))
            x2 = int(round((x + w - 1) * scale_x))
            y2 = int(round((y + h - 1) * scale_y))

            x1 = max(0, min(output_w - 1, x1))
            y1 = max(0, min(output_h - 1, y1))
            x2 = max(0, min(output_w - 1, x2))
            y2 = max(0, min(output_h - 1, y2))

            if x2 >= x1 and y2 >= y1:
                boxes.append([x1, y1, x2, y2])

        return boxes

    except Exception:
        ys, xs = np.where(binary)

        x1 = int(round(xs.min() * scale_x))
        y1 = int(round(ys.min() * scale_y))
        x2 = int(round(xs.max() * scale_x))
        y2 = int(round(ys.max() * scale_y))

        x1 = max(0, min(output_w - 1, x1))
        y1 = max(0, min(output_h - 1, y1))
        x2 = max(0, min(output_w - 1, x2))
        y2 = max(0, min(output_h - 1, y2))

        return [[x1, y1, x2, y2]]


def strip_existing_location(text: str) -> str:
    text = re.sub(r"<location>.*?</location>", "", text, flags=re.DOTALL)
    return text.strip()


def ensure_gpt_value_with_location(
    item: Dict[str, Any],
    location: List[List[int]],
) -> str:
    """
    保留原本 gpt 的 think/evidence/logic/answer，
    只補上 <location>...</location>。
    """
    conversations = item.get("conversations", [])
    old_gpt = ""

    if conversations and len(conversations) >= 2:
        old_gpt = conversations[-1].get("value", "")

    old_gpt = strip_existing_location(old_gpt)

    location_text = json.dumps(location, ensure_ascii=False)

    if old_gpt and "</answer>" in old_gpt:
        return old_gpt + f"<location>{location_text}</location>"

    # fallback
    answer = str(item.get("answer", "")).strip()
    answer_text = str(item.get("answer_text", "")).strip()
    category = str(item.get("object_category", "")).strip()

    return (
        "<think>\n"
        f"<evidence>The image pair provides a test image and a normal reference image for {category}.</evidence>\n"
        f"<logic>Based on the question and choices, option {answer} is correct. {answer_text}</logic>\n"
        "</think>\n"
        f"<answer>{answer}</answer>"
        f"<location>{location_text}</location>"
    )


def choices_to_text(options: Dict[str, Any]) -> str:
    lines = []
    for key in sorted(options.keys()):
        lines.append(f"{key}. {options[key]}")
    return "\n".join(lines)


def answer_spec(options: Dict[str, Any]) -> str:
    keys = sorted(options.keys())
    if keys:
        return "/".join(keys)
    return "A/B/C/D"


def build_two_image_prompt(item: Dict[str, Any]) -> str:
    """
    兩張圖 prompt：
    - 第一張 test
    - 第二張 normal reference
    - answer 格式包含 think/evidence/logic + answer + location
    """
    task_type = str(item.get("task_type", "Anomaly Detection"))
    question = str(item.get("question", ""))
    options = item.get("options", {})

    if not isinstance(options, dict):
        options = {}

    choices = choices_to_text(options)
    spec = answer_spec(options)

    return (
        "<image>\n"
        "<image>\n"
        f"{task_type}:\n\n"
        "The first image is the test image.\n"
        "The second image is a normal reference image.\n\n"
        "Use the normal reference image as comparison context.\n\n"
        f"Question:\n{question}\n\n"
        f"{choices}\n\n"
        "Answer using exactly:\n"
        "<think>\n"
        "<evidence>...</evidence>\n"
        "<logic>...</logic>\n"
        "</think>\n"
        f"<answer>{spec}</answer><location>[[x1,y1,x2,y2],...]</location>\n\n"
        "For normal images, use exactly:\n"
        "<think>\n"
        "<evidence>...</evidence>\n"
        "<logic>...</logic>\n"
        "</think>\n"
        f"<answer>{spec}</answer><location>[]</location>"
    )


def build_defect_localization_prompt() -> str:
    """
    如果你要全部強制變成 Defect Localization prompt，
    執行時加 --force_defect_localization_prompt。
    """
    return (
        "<image>\n"
        "<image>\n"
        "Defect Localization:\n\n"
        "The first image is the test image.\n"
        "The second image is a normal reference image.\n\n"
        "The test image contains a defect.\n"
        "Where is the anomaly mainly located?\n\n"
        "A. middle.\n"
        "B. top.\n"
        "C. right.\n"
        "D. bottom.\n\n"
        "Answer using exactly:\n"
        "<think>\n"
        "<evidence>...</evidence>\n"
        "<logic>...</logic>\n"
        "</think>\n"
        "<answer>A/B/C/D</answer><location>[[x1,y1,x2,y2],...]</location>\n\n"
        "For normal images, use exactly:\n"
        "<think>\n"
        "<evidence>...</evidence>\n"
        "<logic>...</logic>\n"
        "</think>\n"
        "<answer>A/B/C/D</answer><location>[]</location>"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_json",
        required=True,
        help="原始 JSON / JSONL / concatenated JSON objects",
    )
    parser.add_argument(
        "--src_root",
        default="/home/psmc2/project/dataset/MMAD",
        help="原始圖片 root",
    )
    parser.add_argument(
        "--dst_root",
        default="/home/psmc2/project/dataset/Stage2_Dataset",
        help="輸出圖片 root",
    )
    parser.add_argument(
        "--output_json",
        default="/home/psmc2/project/dataset/Stage2_Dataset/stage2_two_image_bbox_qa.json",
        help="輸出的 QA JSON",
    )
    parser.add_argument(
        "--long_edge",
        type=int,
        default=256,
        help="圖片長邊大小",
    )
    parser.add_argument(
        "--skip_missing",
        action="store_true",
        help="缺圖或缺 reference 時跳過，不中斷",
    )
    parser.add_argument(
        "--force_defect_localization_prompt",
        action="store_true",
        help="全部樣本強制使用 Defect Localization prompt",
    )
    parser.add_argument(
        "--min_mask_area",
        type=int,
        default=3,
        help="過小 mask component 過濾門檻",
    )

    args = parser.parse_args()

    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)
    output_json = Path(args.output_json)

    records = read_json_any(args.input_json)

    output_items = []
    missing_logs = []
    warning_logs = []

    for idx, item in enumerate(records):
        item_id = str(item.get("id", f"sample_{idx:06d}"))

        image_rel = item.get("image", None)

        # 如果原本是 list，只取第一張當 test image
        if isinstance(image_rel, list):
            if len(image_rel) == 0:
                warning_logs.append(f"{item_id}: empty image list, skipped.")
                continue
            image_rel = image_rel[0]

        if not isinstance(image_rel, str) or not image_rel:
            warning_logs.append(f"{item_id}: invalid image field, skipped.")
            continue

        image_rel = image_rel.replace("\\", "/")
        item["image"] = image_rel

        src_test = src_root / image_rel
        dst_test = dst_root / image_rel

        if not src_test.exists():
            msg = f"{item_id}: missing test image: {src_test}"
            missing_logs.append(msg)
            if args.skip_missing:
                continue
            raise FileNotFoundError(msg)

        old_w, old_h, new_w, new_h = resize_long_edge(
            src_path=src_test,
            dst_path=dst_test,
            long_edge=args.long_edge,
        )

        ref_rel = find_first_good_reference(src_root, image_rel)

        if ref_rel is None:
            msg = f"{item_id}: cannot find train/good reference for {image_rel}"
            missing_logs.append(msg)
            if args.skip_missing:
                continue
            raise FileNotFoundError(msg)

        src_ref = src_root / ref_rel
        dst_ref = dst_root / ref_rel

        if not src_ref.exists():
            msg = f"{item_id}: missing reference image: {src_ref}"
            missing_logs.append(msg)
            if args.skip_missing:
                continue
            raise FileNotFoundError(msg)

        resize_long_edge(
            src_path=src_ref,
            dst_path=dst_ref,
            long_edge=args.long_edge,
        )

        # location
        if is_good_sample(item):
            location = []
        else:
            mask_candidates = get_mask_candidates(src_root, image_rel)
            mask_path = find_existing_path(mask_candidates)

            if mask_path is None:
                warning_logs.append(
                    f"{item_id}: mask not found for bad image {image_rel}; location set to []."
                )
                location = []
            else:
                location = mask_to_bboxes(
                    mask_path=mask_path,
                    output_w=new_w,
                    output_h=new_h,
                    min_area=args.min_mask_area,
                )

        if args.force_defect_localization_prompt:
            human_value = build_defect_localization_prompt()
        else:
            human_value = build_two_image_prompt(item)

        gpt_value = ensure_gpt_value_with_location(item, location)

        output_items.append(
            {
                "id": item_id,
                "image": [
                    image_rel,
                    ref_rel,
                ],
                "conversations": [
                    {
                        "from": "human",
                        "value": human_value,
                    },
                    {
                        "from": "gpt",
                        "value": gpt_value,
                    },
                ],
            }
        )

    output_json.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output_items, f, ensure_ascii=False, indent=2)

    if missing_logs:
        missing_path = output_json.parent / "missing_stage2_two_image.txt"
        with open(missing_path, "w", encoding="utf-8") as f:
            f.write("\n".join(missing_logs) + "\n")

    if warning_logs:
        warning_path = output_json.parent / "warnings_stage2_two_image.txt"
        with open(warning_path, "w", encoding="utf-8") as f:
            f.write("\n".join(warning_logs) + "\n")

    print("Done.")
    print(f"Input records: {len(records)}")
    print(f"Output records: {len(output_items)}")
    print(f"Output JSON: {output_json}")
    print(f"Output image root: {dst_root}")

    if missing_logs:
        print(f"Missing logs: {len(missing_logs)}")
        print(f"See: {output_json.parent / 'missing_stage2_two_image.txt'}")

    if warning_logs:
        print(f"Warnings: {len(warning_logs)}")
        print(f"See: {output_json.parent / 'warnings_stage2_two_image.txt'}")


if __name__ == "__main__":
    main()