#!/bin/bash
# Cheap pipeline check BEFORE a full zero-shot run: confirms checkpoint path,
# lmms-eval task registration, and output parsing all work, on a tiny slice.
# Never treat this number as a reported result — see run_eval.sh for the
# full-100%-test-set run that is.
#
# Usage:
#   MODEL_PATH=./output/stage3_gs_dpo_merged TASK=mmad_1shot bash eval/smoke_test_eval.sh

LIMIT=20 TASK="${TASK:-mmad_1shot}" MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH}" \
    bash "$(dirname "$0")/run_eval.sh"
