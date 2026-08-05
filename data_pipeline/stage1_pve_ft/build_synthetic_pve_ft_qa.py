# [Stage I: PVE-FT] Synthetic anomaly QA generator.
# Originally: Stage1_Dataset/read_mmad.py
# Inserts DTD-texture and copy-paste synthetic defects into MMAD normal
# (train/good) images, producing detection + region-localization QA under
# the EDDP schema (no <think> block yet — PVE-FT is vision-only pre-alignment).

import argparse
import json
import random
import re
import shutil
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
from tqdm import tqdm

try:
    from perlin_noise import PerlinNoise
except Exception:
    PerlinNoise = None

"""
python build_synthetic_pve_ft_qa.py \
  --mmad_root /home/psmc2/project/dataset/MMAD \
  --dtd_root /home/psmc2/project/dataset/dtd/images \
  --stage_root /home/psmc2/project/dataset/Stage1_Dataset \
  --output /home/psmc2/project/dataset/Stage1_Dataset/mmad_stage1.json
"""

IMG_EXTS = {
    ".jpg", ".jpeg", ".png",
    ".bmp", ".tif", ".tiff"
}


# Unified 3x3 grid (9 cells, all at 1/3 granularity). Replaces the previous
# five_region (5 cells at 1/3) + quadrant (4 corners at coarse 1/2) split,
# which under-resolved the 4 corners and mislabeled near-center boxes that
# straddled the quadrant's half-split as a full corner (see EXPERIMENT_LOG.md,
# 2026-07-23 "3x3 vs 2x2" analysis).
GRID3X3_LABELS = [
    "top-left", "top-center", "top-right",
    "middle-left", "center", "middle-right",
    "bottom-left", "bottom-center", "bottom-right",
]


DETECTION_QUESTION_TEMPLATE = """<image>\n<image>
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
<answer>A/B</answer><location>[[x1,y1,x2,y2],...]</location>"""


REGION_QUESTION_TEMPLATE = """<image>\n<image>
Defect Localization:

The first image is the test image.
The second image is a normal reference image.

The test image contains a defect.
Where is the anomaly mainly located?

{options}

Answer using exactly:
<think>
<evidence>...</evidence>
<logic>...</logic>
</think>
<answer>A/B/C/D</answer><location>[[x1,y1,x2,y2],...]</location>

For normal images, use exactly:
<think>
<evidence>...</evidence>
<logic>...</logic>
</think>
<answer>A/B/C/D</answer><location>[]</location>"""


# Programmatic <think> generation for PVE-FT. Per the paper: "The target
# rationale is ... generated programmatically from this controlled metadata.
# It only verbalizes observable visual evidence and does not introduce
# external domain knowledge." -- so these templates stay generic/visual only,
# no object-category-specific domain claims.
def build_detection_think(is_anomaly: bool, anomaly_kind: Optional[str]) -> Tuple[str, str]:
    if not is_anomaly:
        evidence = (
            "Compared with the normal reference image, the test image shows "
            "no visible deviation in texture, shape, or structure."
        )
        logic = (
            "Since no localized abnormal region is observed relative to the "
            "normal reference, the test image appears normal."
        )
        return evidence, logic

    if anomaly_kind == "texture":
        evidence = (
            "Compared with the normal reference image, a localized region in "
            "the test image shows an abnormal texture or discoloration that "
            "is not present in the reference."
        )
    elif anomaly_kind == "copy_paste":
        evidence = (
            "Compared with the normal reference image, a localized region in "
            "the test image shows a duplicated or inconsistent patch that "
            "disrupts the otherwise uniform surface seen in the reference."
        )
    else:
        evidence = (
            "Compared with the normal reference image, a localized region in "
            "the test image visibly differs from the reference."
        )

    logic = (
        "This localized visual deviation from the normal reference indicates "
        "the presence of a defect in the test image."
    )
    return evidence, logic


def build_region_think(region_label: str) -> Tuple[str, str]:
    evidence = (
        f"The abnormal region identified when comparing against the normal "
        f"reference image is concentrated in the {region_label} part of the "
        f"test image."
    )
    logic = (
        f"Based on the position of the abnormal area relative to the whole "
        f"image, the most consistent location is {region_label}."
    )
    return evidence, logic


def format_think_block(evidence: str, logic: str) -> str:
    return (
        "<think>\n"
        f"<evidence>{evidence}</evidence>\n"
        f"<logic>{logic}</logic>\n"
        "</think>\n"
    )


def natural_key(path: Path):
    text = path.name
    return [
        int(s) if s.isdigit() else s.lower()
        for s in re.split(r"(\d+)", text)
    ]


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTS


def collect_images(folder: Path) -> List[Path]:
    imgs = [
        p for p in folder.iterdir()
        if p.is_file() and is_image_file(p)
    ]
    imgs.sort(key=natural_key)
    return imgs


def collect_good_dirs(mmad_root: Path) -> List[Path]:
    """
    Collect paths like:

    MMAD/MVTec-AD/bottle/train/good
    MMAD/VisA/candle/train/good
    MMAD/DS-MVTec/xxx/train/good
    """
    good_dirs = []

    for p in safe_rglob(mmad_root, "good"):
        if not p.is_dir():
            continue

        if p.parent.name != "train":
            continue

        good_dirs.append(p)

    good_dirs.sort(key=lambda x: x.as_posix())
    return good_dirs


def safe_rglob(root: Path, pattern: str):
    """
    Separate helper only to keep rglob failure messages clear.
    """
    if not root.exists():
        raise FileNotFoundError(f"MMAD root does not exist: {root}")

    return root.rglob(pattern)


def resize_long_edge(
    img: np.ndarray,
    long_edge: int = 256,
    is_mask: bool = False,
) -> np.ndarray:
    """
    Resize image so that its longest side equals long_edge.
    Aspect ratio is preserved.

    Example:
        512 x 384 -> 256 x 192
        300 x 600 -> 128 x 256

    BBox coordinates in JSON are computed after this resize.
    """
    if img is None:
        raise ValueError("resize_long_edge received None image.")

    if long_edge is None or long_edge <= 0:
        return img

    h, w = img.shape[:2]

    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid image shape: {img.shape}")

    scale = float(long_edge) / float(max(h, w))

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    if new_w == w and new_h == h:
        return img

    if is_mask:
        interp = cv2.INTER_NEAREST
    else:
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR

    return cv2.resize(img, (new_w, new_h), interpolation=interp)


def read_image_color(path: Path) -> Optional[np.ndarray]:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if img is None:
        return None

    return img


def stage_image_needs_write(
    dst_path: Path,
    long_edge: int,
) -> bool:
    """
    If stage image exists but is not long-edge resized, rewrite it.
    """
    if not dst_path.exists():
        return True

    img = cv2.imread(str(dst_path), cv2.IMREAD_UNCHANGED)

    if img is None:
        return True

    h, w = img.shape[:2]

    return max(h, w) != long_edge


def save_resized_to_stage(
    src_path: Path,
    source_root: Path,
    stage_root: Path,
    long_edge: int = 256,
    overwrite: bool = False,
) -> Tuple[Path, np.ndarray]:
    """
    Example:

    From:
    /home/psmc2/project/dataset/MMAD/MVTec-AD/bottle/train/good/000.png

    To:
    /home/psmc2/project/dataset/Stage1_Dataset/MVTec-AD/bottle/train/good/000.png

    The saved image is resized to long-edge 256 by default.
    """
    rel_path = src_path.relative_to(source_root)
    dst_path = stage_root / rel_path
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    img = read_image_color(src_path)

    if img is None:
        raise RuntimeError(f"Cannot read image: {src_path}")

    resized = resize_long_edge(
        img=img,
        long_edge=long_edge,
        is_mask=False,
    )

    if overwrite or stage_image_needs_write(dst_path, long_edge):
        cv2.imwrite(str(dst_path), resized)

    return dst_path, resized


def load_dtd_images(dtd_root: Path) -> List[Path]:
    dtd_images = []

    for ext in IMG_EXTS:
        dtd_images.extend(dtd_root.rglob(f"*{ext}"))
        dtd_images.extend(dtd_root.rglob(f"*{ext.upper()}"))

    dtd_images = sorted(set(dtd_images), key=lambda x: x.as_posix())
    return dtd_images


def generate_perlin_mask(h: int, w: int) -> np.ndarray:
    """
    Generate binary anomaly mask.

    Return:
        mask with values 0 or 1
    """
    if PerlinNoise is not None:
        noise = PerlinNoise(
            octaves=random.randint(2, 6),
            seed=random.randint(0, 100000)
        )

        mask = np.zeros((h, w), dtype=np.float32)

        for y in range(h):
            for x in range(w):
                mask[y, x] = noise([
                    y / max(h, 1),
                    x / max(w, 1)
                ])

        mask -= mask.min()
        mask /= mask.max() + 1e-8

    else:
        mask = np.random.rand(h, w).astype(np.float32)

        k = random.choice([9, 15, 21, 31])

        if k >= min(h, w):
            k = max(3, min(h, w) // 2 * 2 - 1)

        if k % 2 == 0:
            k += 1

        mask = cv2.GaussianBlur(mask, (k, k), 0)
        mask -= mask.min()
        mask /= mask.max() + 1e-8

    threshold = random.uniform(0.55, 0.75)
    binary = (mask > threshold).astype(np.uint8)

    return binary


def region_bounds(
    h: int,
    w: int,
    mode: str,
    label: str,
) -> Tuple[int, int, int, int]:
    """
    Return:
        x1, y1, x2, y2

    Note:
        x2, y2 are exclusive bounds.
    """
    if mode == "grid3x3":
        x_third = w // 3
        y_third = h // 3

        col = {"left": (0, x_third), "center": (x_third, 2 * x_third), "right": (2 * x_third, w)}
        row = {"top": (0, y_third), "middle": (y_third, 2 * y_third), "bottom": (2 * y_third, h)}

        if label == "center":
            row_key, col_key = "middle", "center"
        else:
            row_key, col_key = label.split("-")

        y1, y2 = row[row_key]
        x1, x2 = col[col_key]

        return x1, y1, x2, y2

    raise ValueError(f"Unknown mode/label: {mode}, {label}")


def sample_bbox_in_region(
    h: int,
    w: int,
    mode: str,
    label: str,
) -> Tuple[int, int, int, int]:
    """
    Sample bbox inside selected region.

    Return:
        inclusive bbox:
        x1, y1, x2, y2
    """
    rx1, ry1, rx2, ry2 = region_bounds(
        h=h,
        w=w,
        mode=mode,
        label=label,
    )

    rw = max(rx2 - rx1, 1)
    rh = max(ry2 - ry1, 1)

    min_bw = max(8, w // 12)
    max_bw = max(min_bw, w // 4)

    min_bh = max(8, h // 12)
    max_bh = max(min_bh, h // 4)

    bw = random.randint(min_bw, max_bw)
    bh = random.randint(min_bh, max_bh)

    bw = min(bw, rw)
    bh = min(bh, rh)

    max_x1 = max(rx1, rx2 - bw)
    max_y1 = max(ry1, ry2 - bh)

    if max_x1 <= rx1:
        x1 = rx1
    else:
        x1 = random.randint(rx1, max_x1)

    if max_y1 <= ry1:
        y1 = ry1
    else:
        y1 = random.randint(ry1, max_y1)

    x2 = min(x1 + bw - 1, w - 1)
    y2 = min(y1 + bh - 1, h - 1)

    return int(x1), int(y1), int(x2), int(y2)


def choose_region_mode_and_label() -> Tuple[str, str]:
    """
    Single unified localization mode: a true 3x3 grid, 9 cells at 1/3
    granularity (see GRID3X3_LABELS above).
    """
    mode = "grid3x3"
    label = random.choice(GRID3X3_LABELS)

    return mode, label


def mask_to_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return None

    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def texture_anomaly(
    img: np.ndarray,
    dtd_images: List[Path],
    mode: str,
    label: str,
) -> Tuple[np.ndarray, Tuple[int, int, int, int], np.ndarray]:
    """
    Generate DTD texture anomaly.

    Input img is already long-edge resized.
    Returned bbox is therefore also in the resized coordinate system.
    """
    h, w = img.shape[:2]

    x1, y1, x2, y2 = sample_bbox_in_region(
        h=h,
        w=w,
        mode=mode,
        label=label,
    )

    bw = x2 - x1 + 1
    bh = y2 - y1 + 1

    texture_path = random.choice(dtd_images)
    texture = cv2.imread(str(texture_path), cv2.IMREAD_COLOR)

    if texture is None:
        return texture_anomaly(
            img=img,
            dtd_images=dtd_images,
            mode=mode,
            label=label,
        )

    texture = cv2.resize(texture, (bw, bh), interpolation=cv2.INTER_AREA)

    roi = img[y1:y2 + 1, x1:x2 + 1].copy()

    local_mask = generate_perlin_mask(bh, bw)

    retry = 0

    while local_mask.sum() < max(20, int(0.02 * bh * bw)) and retry < 10:
        local_mask = generate_perlin_mask(bh, bw)
        retry += 1

    if local_mask.sum() == 0:
        local_mask[:, :] = 1

    alpha = random.uniform(0.45, 0.85)

    anomaly_roi = (
        alpha * texture.astype(np.float32)
        + (1.0 - alpha) * roi.astype(np.float32)
    ).clip(0, 255).astype(np.uint8)

    output = img.copy()
    out_roi = roi.copy()

    out_roi[local_mask == 1] = anomaly_roi[local_mask == 1]
    output[y1:y2 + 1, x1:x2 + 1] = out_roi

    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[y1:y2 + 1, x1:x2 + 1] = local_mask * 255

    real_bbox = mask_to_bbox(full_mask)

    if real_bbox is None:
        real_bbox = (x1, y1, x2, y2)

    return output, real_bbox, full_mask


def copy_paste_anomaly(
    img: np.ndarray,
    mode: str,
    label: str,
) -> Tuple[np.ndarray, Tuple[int, int, int, int], np.ndarray]:
    """
    Generate copy-paste anomaly.

    Input img is already long-edge resized.
    Returned bbox is therefore also in the resized coordinate system.
    """
    h, w = img.shape[:2]

    tx1, ty1, tx2, ty2 = sample_bbox_in_region(
        h=h,
        w=w,
        mode=mode,
        label=label,
    )

    bw = tx2 - tx1 + 1
    bh = ty2 - ty1 + 1

    sx1 = 0
    sy1 = 0

    for _ in range(100):
        sx1 = random.randint(0, max(0, w - bw))
        sy1 = random.randint(0, max(0, h - bh))

        sx2 = sx1 + bw - 1
        sy2 = sy1 + bh - 1

        center_dist = (
            abs((sx1 + sx2) / 2 - (tx1 + tx2) / 2)
            + abs((sy1 + sy2) / 2 - (ty1 + ty2) / 2)
        )

        if center_dist > max(bw, bh):
            break

    sx2 = sx1 + bw - 1
    sy2 = sy1 + bh - 1

    patch = img[sy1:sy2 + 1, sx1:sx2 + 1].copy()

    if random.random() < 0.5:
        patch = cv2.flip(patch, 1)

    if random.random() < 0.5:
        patch = cv2.flip(patch, 0)

    alpha = random.uniform(0.85, 1.20)
    beta = random.randint(-25, 25)

    patch = cv2.convertScaleAbs(
        patch,
        alpha=alpha,
        beta=beta,
    )

    output = img.copy()
    output[ty1:ty2 + 1, tx1:tx2 + 1] = patch

    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[ty1:ty2 + 1, tx1:tx2 + 1] = 255

    return output, (tx1, ty1, tx2, ty2), full_mask


def build_detection_qa(
    is_anomaly: bool,
    bbox: Optional[Tuple[int, int, int, int]],
    anomaly_kind: Optional[str] = None,
) -> Tuple[str, str]:
    """
    第一題：
    問有沒有異常。
    answer 包含：
    <think><evidence>...</evidence><logic>...</logic></think>
    <answer>A/B</answer><location>...</location>

    anomaly_kind: "texture" | "copy_paste" | None (only used when is_anomaly=True,
    to vary the programmatic evidence sentence; see build_detection_think()).
    """
    if random.random() < 0.5:
        choices = [
            ("Yes", True),
            ("No", False),
        ]
    else:
        choices = [
            ("No", False),
            ("Yes", True),
        ]

    option_lines = []
    answer_letter = None

    for idx, (text, value) in enumerate(choices):
        letter = chr(ord("A") + idx)
        option_lines.append(f"{letter}. {text}.")

        if value == is_anomaly:
            answer_letter = letter

    if answer_letter is None:
        raise RuntimeError("Failed to build detection answer.")

    question = DETECTION_QUESTION_TEMPLATE.format(
        options="\n".join(option_lines)
    )

    if is_anomaly and bbox is not None:
        location = [
            [
                int(bbox[0]),
                int(bbox[1]),
                int(bbox[2]),
                int(bbox[3]),
            ]
        ]
    else:
        location = []

    location_str = json.dumps(
        location,
        ensure_ascii=False,
        separators=(",", ":")
    )

    evidence, logic = build_detection_think(is_anomaly=is_anomaly, anomaly_kind=anomaly_kind)

    answer = (
        format_think_block(evidence, logic)
        + f"<answer>{answer_letter}</answer>"
        f"<location>{location_str}</location>"
    )

    return question, answer


def build_region_qa(
    region_mode: str,
    region_label: str,
    bbox: Tuple[int, int, int, int],
) -> Tuple[str, str]:
    """
    第二題：
    只問異常位置。
    只給異常圖片使用。
    answer 也保留 bbox location。
    """
    if region_mode != "grid3x3":
        raise ValueError(f"Unknown region mode: {region_mode}")

    distractors = [
        x for x in GRID3X3_LABELS
        if x != region_label
    ]

    sampled = random.sample(distractors, 3)

    labels = [region_label] + sampled
    random.shuffle(labels)

    option_lines = []
    answer_letter = None

    for idx, label in enumerate(labels):
        letter = chr(ord("A") + idx)
        option_lines.append(f"{letter}. {label}.")

        if label == region_label:
            answer_letter = letter

    if answer_letter is None:
        raise RuntimeError("Failed to build region answer.")

    question = REGION_QUESTION_TEMPLATE.format(
        options="\n".join(option_lines)
    )

    location = [
        [
            int(bbox[0]),
            int(bbox[1]),
            int(bbox[2]),
            int(bbox[3]),
        ]
    ]

    location_str = json.dumps(
        location,
        ensure_ascii=False,
        separators=(",", ":")
    )

    evidence, logic = build_region_think(region_label=region_label)

    answer = (
        format_think_block(evidence, logic)
        + f"<answer>{answer_letter}</answer>"
        f"<location>{location_str}</location>"
    )

    return question, answer


def save_mask(mask_path: Path, mask: np.ndarray):
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(mask_path), mask)


def make_id(
    dataset_name: str,
    object_name: str,
    stem: str,
    kind: str,
    sample_id: int,
) -> str:
    dataset_name = dataset_name.replace("/", "_")
    object_name = object_name.replace("/", "_")
    stem = stem.replace("/", "_")

    return f"mmad_{dataset_name}_{object_name}_{stem}_{kind}_{sample_id:06d}"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mmad_root",
        default="/home/psmc2/project/dataset/MMAD",
        help="MMAD dataset root.",
    )

    parser.add_argument(
        "--dtd_root",
        default="/home/psmc2/project/dataset/dtd/images",
        help="DTD image root.",
    )

    parser.add_argument(
        "--stage_root",
        default="/home/psmc2/project/dataset/Stage1_Dataset",
        help="Output Stage1 dataset root.",
    )

    parser.add_argument(
        "--output",
        default="/home/psmc2/project/dataset/Stage1_Dataset/mmad_stage1.json",
        help="Output QA json path.",
    )

    parser.add_argument(
        "--num_train",
        type=int,
        default=5,
        help="Number of train images per object after the reference image.",
    )

    parser.add_argument(
        "--long_edge",
        type=int,
        default=256,
        help="Resize every saved image so its longest side equals this value.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing original/reference stage files.",
    )

    parser.add_argument(
        "--save_masks",
        type=int,
        default=1,
        help="1: save synthetic masks. 0: do not save masks.",
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    mmad_root = Path(args.mmad_root)
    dtd_root = Path(args.dtd_root)
    stage_root = Path(args.stage_root)
    output_json = Path(args.output)

    dtd_images = load_dtd_images(dtd_root)

    print("MMAD root:", mmad_root)
    print("Stage root:", stage_root)
    print("DTD root:", dtd_root)
    print("DTD images:", len(dtd_images))
    print("Long edge:", args.long_edge)

    if len(dtd_images) == 0:
        raise RuntimeError(f"No DTD images found under: {dtd_root}")

    good_dirs = collect_good_dirs(mmad_root)

    print("Object good dirs:", len(good_dirs))

    qa_data = []
    sample_id = 0

    for good_dir in tqdm(good_dirs, desc="Objects"):
        images = collect_images(good_dir)

        if len(images) < 2:
            print(f"[WARN] Skip {good_dir}, images < 2")
            continue

        # Example:
        # good_dir:
        # MMAD/MVTec-AD/bottle/train/good
        #
        # object_dir:
        # MMAD/MVTec-AD/bottle
        object_dir = good_dir.parent.parent
        object_name = object_dir.name
        dataset_name = object_dir.parent.name

        # 檔名排序後：
        # 第 1 張：reference image
        # 後面 5 張：training/test images
        ref_img_path = images[0]
        train_img_paths = images[1:1 + args.num_train]

        if len(train_img_paths) < args.num_train:
            print(
                f"[WARN] {good_dir}: only {len(train_img_paths)} train images after reference."
            )

        # Save resized reference image
        ref_stage_path, _ = save_resized_to_stage(
            src_path=ref_img_path,
            source_root=mmad_root,
            stage_root=stage_root,
            long_edge=args.long_edge,
            overwrite=args.overwrite,
        )

        ref_rel = ref_stage_path.relative_to(stage_root).as_posix()

        for img_path in train_img_paths:
            # Save resized original normal test image.
            # The returned img is already long-edge resized.
            try:
                test_stage_path, img = save_resized_to_stage(
                    src_path=img_path,
                    source_root=mmad_root,
                    stage_root=stage_root,
                    long_edge=args.long_edge,
                    overwrite=args.overwrite,
                )
            except RuntimeError as e:
                print(f"[WARN] {e}")
                continue

            test_rel = test_stage_path.relative_to(stage_root).as_posix()
            stem = img_path.stem

            # =====================================================
            # 1. Normal image detection QA
            # =====================================================

            question, answer = build_detection_qa(
                is_anomaly=False,
                bbox=None,
            )

            qa_data.append(
                {
                    "id": make_id(
                        dataset_name=dataset_name,
                        object_name=object_name,
                        stem=stem,
                        kind="normal_detection",
                        sample_id=sample_id,
                    ),
                    "image": [
                        test_rel,
                        ref_rel,
                    ],
                    "conversations": [
                        {
                            "from": "human",
                            "value": question,
                        },
                        {
                            "from": "gpt",
                            "value": answer,
                        },
                    ],
                }
            )

            sample_id += 1

            # =====================================================
            # 2. DTD texture anomaly image
            # =====================================================

            region_mode, region_label = choose_region_mode_and_label()

            dtd_img, dtd_bbox, dtd_mask = texture_anomaly(
                img=img,
                dtd_images=dtd_images,
                mode=region_mode,
                label=region_label,
            )

            dtd_dir = test_stage_path.parent.parent / "dtd_texture"
            dtd_dir.mkdir(parents=True, exist_ok=True)

            dtd_save_path = dtd_dir / f"{stem}.png"

            # Always write synthetic image to keep image, mask, bbox, and JSON consistent.
            cv2.imwrite(str(dtd_save_path), dtd_img)

            if args.save_masks:
                dtd_mask_dir = test_stage_path.parent.parent / "dtd_texture_mask"
                dtd_mask_path = dtd_mask_dir / f"{stem}.png"
                save_mask(dtd_mask_path, dtd_mask)

            dtd_rel = dtd_save_path.relative_to(stage_root).as_posix()

            # 2-1. DTD detection QA
            question, answer = build_detection_qa(
                is_anomaly=True,
                bbox=dtd_bbox,
                anomaly_kind="texture",
            )

            qa_data.append(
                {
                    "id": make_id(
                        dataset_name=dataset_name,
                        object_name=object_name,
                        stem=stem,
                        kind="dtd_texture_detection",
                        sample_id=sample_id,
                    ),
                    "image": [
                        dtd_rel,
                        ref_rel,
                    ],
                    "conversations": [
                        {
                            "from": "human",
                            "value": question,
                        },
                        {
                            "from": "gpt",
                            "value": answer,
                        },
                    ],
                }
            )

            sample_id += 1

            # 2-2. DTD region QA with bbox
            question, answer = build_region_qa(
                region_mode=region_mode,
                region_label=region_label,
                bbox=dtd_bbox,
            )

            qa_data.append(
                {
                    "id": make_id(
                        dataset_name=dataset_name,
                        object_name=object_name,
                        stem=stem,
                        kind="dtd_texture_region",
                        sample_id=sample_id,
                    ),
                    "image": [
                        dtd_rel,
                        ref_rel,
                    ],
                    "conversations": [
                        {
                            "from": "human",
                            "value": question,
                        },
                        {
                            "from": "gpt",
                            "value": answer,
                        },
                    ],
                }
            )

            sample_id += 1

            # =====================================================
            # 3. Copy-paste anomaly image
            # =====================================================

            region_mode, region_label = choose_region_mode_and_label()

            cp_img, cp_bbox, cp_mask = copy_paste_anomaly(
                img=img,
                mode=region_mode,
                label=region_label,
            )

            cp_dir = test_stage_path.parent.parent / "copy_paste"
            cp_dir.mkdir(parents=True, exist_ok=True)

            cp_save_path = cp_dir / f"{stem}.png"

            # Always write synthetic image to keep image, mask, bbox, and JSON consistent.
            cv2.imwrite(str(cp_save_path), cp_img)

            if args.save_masks:
                cp_mask_dir = test_stage_path.parent.parent / "copy_paste_mask"
                cp_mask_path = cp_mask_dir / f"{stem}.png"
                save_mask(cp_mask_path, cp_mask)

            cp_rel = cp_save_path.relative_to(stage_root).as_posix()

            # 3-1. Copy-paste detection QA
            question, answer = build_detection_qa(
                is_anomaly=True,
                bbox=cp_bbox,
                anomaly_kind="copy_paste",
            )

            qa_data.append(
                {
                    "id": make_id(
                        dataset_name=dataset_name,
                        object_name=object_name,
                        stem=stem,
                        kind="copy_paste_detection",
                        sample_id=sample_id,
                    ),
                    "image": [
                        cp_rel,
                        ref_rel,
                    ],
                    "conversations": [
                        {
                            "from": "human",
                            "value": question,
                        },
                        {
                            "from": "gpt",
                            "value": answer,
                        },
                    ],
                }
            )

            sample_id += 1

            # 3-2. Copy-paste region QA with bbox
            question, answer = build_region_qa(
                region_mode=region_mode,
                region_label=region_label,
                bbox=cp_bbox,
            )

            qa_data.append(
                {
                    "id": make_id(
                        dataset_name=dataset_name,
                        object_name=object_name,
                        stem=stem,
                        kind="copy_paste_region",
                        sample_id=sample_id,
                    ),
                    "image": [
                        cp_rel,
                        ref_rel,
                    ],
                    "conversations": [
                        {
                            "from": "human",
                            "value": question,
                        },
                        {
                            "from": "gpt",
                            "value": answer,
                        },
                    ],
                }
            )

            sample_id += 1

    output_json.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(
            qa_data,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("Finished.")
    print("Generated QA:", len(qa_data))
    print("Saved JSON:", output_json)


if __name__ == "__main__":
    main()
