"""
Stage2 data-mix v2: shrink Domain Knowledge MCQ's share from 50.3% down to
~25-30%, so it stops crowding out the weak eval categories (Anomaly Detection,
Defect Localization, Defect Classification) in Stage2's single training epoch.

merge_stage_dataset.py only supports REPEATING a source up, not shrinking one
down, and Domain Knowledge MCQ is the one source that gets added as-is (not
repeated) — so the only way to reduce its share is to randomly downsample the
already-built pool post-hoc, keeping every other category's count untouched.

Does NOT touch the source stage2.json in place — writes a new versioned file per the "never
overwrite, always version" rule.
"""
import json
import os
import random

SRC = os.environ.get("STAGE2_DATA_PATH", "./data/Stage2_Dataset/stage2.json")
OUT = os.environ.get("STAGE2_V2_OUT_PATH", "./data/Stage2_Dataset_v2/stage2_v2_dkmcq_downsampled.json")
TARGET_DK_COUNT = 5000  # -> ~27.8% of the new total, mid-way in the 25-30% target
SEED = 42


def type_of(item):
    q = item["conversations"][0]["value"]
    lines = q.split("\n")
    i = 0
    while i < len(lines) and lines[i].strip() in ("<image>", ""):
        i += 1
    return lines[i].rstrip(":").strip() if i < len(lines) else "UNKNOWN"


if __name__ == "__main__":
    import os

    data = json.load(open(SRC))
    dk_items = [x for x in data if type_of(x) == "Domain Knowledge MCQ"]
    other_items = [x for x in data if type_of(x) != "Domain Knowledge MCQ"]

    random.seed(SEED)
    dk_sampled = random.sample(dk_items, min(TARGET_DK_COUNT, len(dk_items)))

    merged = dk_sampled + other_items
    random.shuffle(merged)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    new_total = len(merged)
    print(f"Original: {len(data)} total, {len(dk_items)} Domain Knowledge MCQ ({len(dk_items)/len(data)*100:.1f}%)")
    print(f"New:      {new_total} total, {len(dk_sampled)} Domain Knowledge MCQ ({len(dk_sampled)/new_total*100:.1f}%)")
    print(f"Written to {OUT}")
