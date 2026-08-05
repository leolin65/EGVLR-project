"""CLI wrapper around reference_banks.build_reference_banks() for the Stage2/Stage3 data used by
the live pipeline. CPU-only, no GPU job needed, does not touch any running training job.

Usage:
    python build_banks.py --out /path/to/ref_banks_v1.pkl
"""

import argparse
import os

from reference_banks import build_reference_banks

DEFAULT_DATA_PATHS = [
    os.environ.get("STAGE2_DATA_PATH", "./data/Stage2_Dataset/stage2.json"),
    os.environ.get("STAGE3_DATA_PATH", "./data/Stage3_Dataset/stage3.json"),
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=DEFAULT_DATA_PATHS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-per-bank", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    counts = build_reference_banks(
        args.data,
        output_path=args.out,
        max_per_bank=args.max_per_bank,
        seed=args.seed,
    )
    print(f"Wrote {args.out}: {counts}")
