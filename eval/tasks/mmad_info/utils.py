import json
import os
import re
from PIL import Image

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

IMAGE_ROOT = os.environ.get("MMAD_IMAGE_ROOT", "./eval_data/MMAD256")


# ==========================================================
# Image
# ==========================================================

def mmad_doc_to_visual(doc):
    image = Image.open(
        os.path.join(
            IMAGE_ROOT,
            doc["image"]
        )
    ).convert("RGB")

    return [image]


# ==========================================================
# Prompt
# ==========================================================

def mmad_doc_to_text(doc, lmms_eval_specific_kwargs=None):

    question = doc["question"]

    if lmms_eval_specific_kwargs:
        question += lmms_eval_specific_kwargs.get(
            "post_prompt",
            ""
        )

    return question


# ==========================================================
# Utils
# ==========================================================

def normalize(x):
    if x is None:
        return ""
    return str(x).strip().lower()


def is_good(anomaly):
    return normalize(anomaly) == "good"


def extract_json(text):

    if text is None:
        return None

    text = text.strip()

    m = re.search(r"\{.*\}", text, re.S)

    if m is None:
        return None

    try:
        return json.loads(m.group())
    except Exception:
        return None


# ==========================================================
# IoU
# ==========================================================

def bbox_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)

    inter = inter_w * inter_h

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = area1 + area2 - inter

    if union <= 0:
        return 0.0

    return inter / union


def location_score(pred_boxes, gt_boxes):

    if len(pred_boxes) == 0 and len(gt_boxes) == 0:
        return 1.0

    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return 0.0

    n = len(gt_boxes)
    m = len(pred_boxes)

    # Hungarian matching
    if HAS_SCIPY:

        import numpy as np

        cost = np.ones((n, m), dtype=float)

        for i, gt in enumerate(gt_boxes):
            for j, pred in enumerate(pred_boxes):
                cost[i, j] = 1.0 - bbox_iou(pred, gt)

        rows, cols = linear_sum_assignment(cost)

        total = 0.0

        for r, c in zip(rows, cols):
            total += 1.0 - cost[r, c]

        return total / max(n, m)

    # Greedy fallback
    used = set()

    total = 0.0

    for gt in gt_boxes:

        best = 0.0
        best_idx = -1

        for i, pred in enumerate(pred_boxes):

            if i in used:
                continue

            iou = bbox_iou(pred, gt)

            if iou > best:
                best = iou
                best_idx = i

        if best_idx >= 0:
            used.add(best_idx)

        total += best

    return total / max(n, m)


# ==========================================================
# Evaluation
# ==========================================================

def mmad_process_results(doc, results):

    pred = extract_json(results[0])

    gt = doc["answer"]

    if isinstance(gt, str):
        gt = json.loads(gt)

    if pred is None:
        return {
            "object_acc": 0.0,
            "detection_acc": 0.0,
            "location_iou": 0.0,
            "exact_match": 0.0,
        }

    pred_object = normalize(pred.get("object"))
    pred_location = pred.get("location", [])

    gt_object = normalize(gt.get("object"))
    gt_location = gt.get("location", [])

    object_acc = float(pred_object == gt_object)

    # bbox 是否存在 = 是否有 defect
    pred_has_defect = len(pred_location) > 0
    gt_has_defect = len(gt_location) > 0

    detection_acc = float(
        pred_has_defect == gt_has_defect
    )

    location_iou = location_score(
        pred_location,
        gt_location,
    )

    exact_match = float(
        object_acc == 1.0
        and detection_acc == 1.0
        and len(pred_location) == len(gt_location)
        and location_iou >= 0.999999
    )

    return {
        "object_acc": object_acc,
        "detection_acc": detection_acc,
        "location_iou": location_iou,
        "exact_match": exact_match,
    }


# ==========================================================
# Aggregate
# ==========================================================

def object_acc(results):
    return 100.0 * sum(results) / len(results)


def detection_acc(results):
    return 100.0 * sum(results) / len(results)


def location_iou(results):
    return 100.0 * sum(results) / len(results)


def exact_match(results):
    return 100.0 * sum(results) / len(results)