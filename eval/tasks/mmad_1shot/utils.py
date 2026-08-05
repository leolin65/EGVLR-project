import os
import re
from PIL import Image

IMAGE_ROOT1 = os.environ.get("MMAD_IMAGE_ROOT1", "./eval_data/images")
IMAGE_ROOT2 = os.environ.get("MMAD_IMAGE_ROOT2", "./eval_data/images_t")


def resolve_image_path(rel_path):
    """
    Try to find image from IMAGE_ROOT1 first, then IMAGE_ROOT2.

    This supports:
    - test image in IAD256
    - reference image in IAD256
    - fallback reference image in IAD256t
    """
    if os.path.isabs(rel_path):
        if os.path.exists(rel_path):
            return rel_path
        raise FileNotFoundError(f"Image not found: {rel_path}")

    candidates = [
        os.path.join(IMAGE_ROOT1, rel_path),
        os.path.join(IMAGE_ROOT2, rel_path),
    ]

    for p in candidates:
        if os.path.exists(p):
            return p

    raise FileNotFoundError(
        "Image not found. Tried:\n" + "\n".join(candidates)
    )


def get_test_and_reference_paths(doc):
    """
    Supported JSON formats:

    Format 1, recommended:
    {
      "image": [
        "test_image_path",
        "reference_image_path"
      ]
    }

    Format 2, old fallback:
    {
      "image": "test_image_path",
      "reference_image": "reference_image_path"
    }
    """
    image_field = doc["image"]

    # New format: image is [test, reference]
    if isinstance(image_field, list):
        if len(image_field) < 2:
            raise ValueError(
                f"doc['image'] should contain [test_image, reference_image], "
                f"but got: {image_field}"
            )

        test_rel = image_field[0]
        ref_rel = image_field[1]
        return test_rel, ref_rel

    # Old fallback format
    if "reference_image" in doc:
        test_rel = image_field
        ref_rel = doc["reference_image"]
        return test_rel, ref_rel

    raise ValueError(
        "Invalid image format. Expected either:\n"
        "1. doc['image'] = [test_image, reference_image]\n"
        "or\n"
        "2. doc['image'] = test_image and doc['reference_image'] = reference_image"
    )


def mmad_doc_to_visual(doc):
    """
    Image order must match the prompt.

    Current prompt:
      The first image is the test image.
      The second image is a normal reference image.

    Therefore return:
      [test_img, ref_img]
    """
    test_rel, ref_rel = get_test_and_reference_paths(doc)

    test_path = resolve_image_path(test_rel)
    ref_path = resolve_image_path(ref_rel)

    test_img = Image.open(test_path).convert("RGB")
    ref_img = Image.open(ref_path).convert("RGB")

    return [
        test_img,
        ref_img,
    ]

def mmad_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]

    # LMMs-Eval already receives images from doc_to_visual.
    # Remove textual image tokens to avoid token/image mismatch.
    question = question.replace("<image>\n", "")
    question = question.replace("<image>", "")

    if lmms_eval_specific_kwargs:
        question += lmms_eval_specific_kwargs.get("post_prompt", "")

    return question


def extract_answer(text):
    """
    Supports:
    - B
    - <answer>B</answer>
    - model verbose response containing <answer>B</answer>
    """
    text = str(text).strip()

    # Prefer explicit <answer>...</answer>
    match = re.search(r"<answer>\s*([A-D])\s*</answer>", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()

    text_upper = text.upper()

    # Then find standalone A/B/C/D
    match = re.search(r"\b([A-D])\b", text_upper)
    if match:
        return match.group(1)

    return text_upper[:1]


def mmad_process_results(doc, results):
    pred = extract_answer(results[0])
    gt = extract_answer(doc["answer"])

    return {
        "acc": float(pred == gt)
    }


def mmad_aggregate(results):
    return sum(results) / len(results) * 100