import re
from PIL import Image
import os

IMAGE_ROOT = os.environ.get("MMAD_IMAGE_ROOT", "./eval_data/images")


def mmad_doc_to_visual(doc):

    image_path = os.path.join(
        IMAGE_ROOT,
        doc["image"]
    )

    image = Image.open(image_path).convert("RGB")

    return [image]


def mmad_doc_to_text(doc, lmms_eval_specific_kwargs=None):

    question = doc["question"]

    question = question.replace("<image>\n", "")
    question = question.replace("<image>", "")

    if lmms_eval_specific_kwargs:
        question += lmms_eval_specific_kwargs.get(
            "post_prompt", ""
        )

    return question


def extract_answer(text):

    text = text.strip().upper()

    match = re.search(r"\b([A-D])\b", text)

    if match:
        return match.group(1)

    return text[:1]


def mmad_process_results(doc, results):

    pred = extract_answer(results[0])

    gt = doc["answer"].strip().upper()

    return {
        "acc": float(pred == gt)
    }


def mmad_aggregate(results):
    return sum(results) / len(results) * 100