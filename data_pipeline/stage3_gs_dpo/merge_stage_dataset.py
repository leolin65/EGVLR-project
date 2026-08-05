# [Stage III: GS-DPO] Merges the GRPO training sources into one file.
# Originally: Stage3_Dataset/merge_json.py

import json
import argparse
import random
from pathlib import Path
from typing import Any, Dict, List

"""
python merge_stage_dataset.py \
  --json1 /home/psmc2/project/dataset/Stage3_Dataset/test_stage3.json \
  --json2 /home/psmc2/project/dataset/Stage3_Dataset/stage3.json \
  --output /home/psmc2/project/dataset/Stage3_Dataset/stage3.json \
  --repeat_json1 1 \
  --unique_repeat_id \
  --shuffle \
  --seed 42
"""

def read_json_any(path: str) -> List[Dict[str, Any]]:
    """
    支援：
    1. JSON array: [ {...}, {...} ]
    2. JSONL: 每行一個 JSON
    3. concatenated JSON objects: {...}{...}{...}
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        return []

    # Case 1: normal JSON array / single object
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
            raise ValueError(f"Expected JSON object at position {idx}, got {type(obj)}")

        records.append(obj)
        idx = end

    return records


def write_json_array(records: List[Dict[str, Any]], output_path: str):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def write_jsonl(records: List[Dict[str, Any]], output_path: str):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--json1",
        required=True,
        help="第一份 JSON，會重複 N 次，例如 stage2_two_image_bbox_qa.json",
    )
    parser.add_argument(
        "--json2",
        required=True,
        help="第二份 JSON，原樣加入，例如 judo_domain_knowledge_mcq_stage2.json",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="合併後輸出路徑",
    )
    parser.add_argument(
        "--repeat_json1",
        type=int,
        default=4,
        help="第一份 JSON 重複幾次，預設 4",
    )
    parser.add_argument(
        "--unique_repeat_id",
        action="store_true",
        help="重複第一份 JSON 時，替 id 加 __rep0/__rep1 避免 id 重複",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="合併後是否打亂順序",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="shuffle random seed",
    )
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="輸出成 JSONL；預設輸出 JSON array",
    )

    args = parser.parse_args()

    data1 = read_json_any(args.json1)
    data2 = read_json_any(args.json2)

    merged = []

    # 第一份 JSON 重複 repeat_json1 次
    for rep_idx in range(args.repeat_json1):
        for item in data1:
            new_item = json.loads(json.dumps(item, ensure_ascii=False))

            if args.unique_repeat_id:
                old_id = str(new_item.get("id", ""))
                new_item["id"] = f"{old_id}__rep{rep_idx}"

            merged.append(new_item)

    # 第二份 JSON 原樣加入
    for item in data2:
        new_item = json.loads(json.dumps(item, ensure_ascii=False))
        merged.append(new_item)

    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(merged)

    if args.jsonl:
        write_jsonl(merged, args.output)
    else:
        write_json_array(merged, args.output)

    print("Done.")
    print(f"JSON1 records: {len(data1)}")
    print(f"JSON1 repeated: {args.repeat_json1} times")
    print(f"JSON1 total after repeat: {len(data1) * args.repeat_json1}")
    print(f"JSON2 records: {len(data2)}")
    print(f"Merged total: {len(merged)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()