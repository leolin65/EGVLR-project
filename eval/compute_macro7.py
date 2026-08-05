"""
Recompute MMAD-style scores from lmms-eval's mmad_1shot per-sample jsonl logs,
matching the ORIGINAL MMAD paper's (arXiv:2410.09453) "Average" metric exactly:
a simple unweighted mean across the 7 official subtasks (Anomaly Discrimination,
Defect Classification, Defect Localization, Defect Description, Defect Analysis,
Object Classification, Object Analysis) — verified by reproducing the paper's own
Table 2 numbers (Human-expert 86.65, GPT-4o 74.92, InternVL2-76B 70.75) from the
per-subtask values listed there.

Our eval data additionally splits "Object Analysis" into 3 finer buckets (Object
Analysis / Object Structure / Object Details) that don't exist in the official
7-task taxonomy. To match the paper exactly, those 3 are POOLED into one "Object
Analysis" bucket (samples combined, not sub-averages averaged) before taking the
7-way mean, since the paper only ever had one Object Analysis category — MMAD's
own 7-task split just doesn't exist at that granularity in the original benchmark.
"""
import argparse
import json
import collections
import glob

MMAD7_MERGE = {
    "Anomaly Detection": "Anomaly Discrimination",
    "Defect Localization": "Defect Localization",
    "Defect Analysis": "Defect Analysis",
    "Defect Description": "Defect Description",
    "Defect Classification": "Defect Classification",
    "Object Classification": "Object Classification",
    "Object Analysis": "Object Analysis",
    "Object Structure": "Object Analysis",
    "Object Details": "Object Analysis",
}


def type_of(row):
    q = row["input"] if isinstance(row["input"], str) else json.dumps(row["input"])
    lines = q.split("\n")
    i = 0
    while i < len(lines) and lines[i].strip() in ("<image>", ""):
        i += 1
    return lines[i].rstrip(":").strip() if i < len(lines) else "UNKNOWN"


def score_file(path):
    rows = [json.loads(l) for l in open(path)]
    by_7 = collections.defaultdict(list)
    for r in rows:
        raw_type = type_of(r)
        mmad7_type = MMAD7_MERGE.get(raw_type)
        if mmad7_type is None:
            raise ValueError(f"Unrecognized question type: {raw_type!r} in {path}")
        by_7[mmad7_type].append(r["acc"])

    per_type = {k: sum(v) / len(v) * 100 for k, v in by_7.items()}
    macro7 = sum(per_type.values()) / len(per_type)
    flat = sum(r["acc"] for r in rows) / len(rows) * 100
    return {"n": len(rows), "flat": flat, "macro7": macro7, "per_type": per_type}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Recompute the 7-task MMAD macro-average from one or more lmms-eval "
        "mmad_1shot result directories."
    )
    ap.add_argument(
        "targets",
        nargs="+",
        metavar="LABEL=DIR",
        help="One or more label=result_dir pairs, e.g. "
        "'GS-DPO (full)=./results/mmad_1shot/egvlr__stage3_gs_dpo_merged'",
    )
    args = ap.parse_args()

    for pair in args.targets:
        label, _, d = pair.partition("=")
        matches = sorted(glob.glob(f"{d}/*_samples_mmad_1shot.jsonl"))
        if not matches:
            print(f"{label:30s} NO FILE FOUND in {d}")
            continue
        result = score_file(matches[-1])
        print(f"{label:30s} n={result['n']:5d}  flat={result['flat']:6.2f}  macro7={result['macro7']:6.2f}")
