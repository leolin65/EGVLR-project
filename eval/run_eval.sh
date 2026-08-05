#!/bin/bash
# Evaluate a merged EGVLR checkpoint on the MMAD benchmark via lmms-eval.
#
# Prereq: copy/symlink this repo's eval/tasks/ directory into your lmms-eval
# checkout's lmms_eval/tasks/, and place the MMAD eval json files referenced
# in tasks/mmad*/*.yaml under ./eval_data/ (or update those paths).
#
# Usage:
#   MODEL_PATH=./output/stage3_gs_dpo_merged TASK=mmad_1shot bash eval/run_eval.sh

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a merged EGVLR checkpoint dir}"
TASK="${TASK:-mmad_1shot}"
# LIMIT=5000 = full mmad_1shot test set (the only number that may be reported
# as a "zero-shot" result). For a smoke test before a full run, use
# eval/smoke_test_eval.sh (LIMIT=20) instead of overriding this by hand.
LIMIT="${LIMIT:-5000}"

python -m lmms_eval \
    --model qwen3_vl \
    --model_args pretrained="$MODEL_PATH" \
    --tasks "$TASK" \
    --batch_size 32 \
    --output_path "results/$TASK" \
    --log_samples \
    --limit "$LIMIT"
