#!/bin/bash
# Merge a LoRA checkpoint produced by any of the three EGVLR stages
# (finetune_stage1_pve_ft.sh / finetune_stage2_kg_it.sh / finetune_stage3_gs_dpo.sh)
# back into the base backbone, producing a standalone merged model that the
# next stage (or eval) can load directly.
#
# Usage:
#   MODEL_BASE="Qwen/Qwen3-VL-8B-Instruct" \
#   MODEL_PATH=output/stage1_pve_ft_lora/checkpoint-XXXX \
#   SAVE_PATH=output/stage1_pve_ft_merged \
#   bash training/merge_lora.sh

MODEL_BASE="${MODEL_BASE:-Qwen/Qwen3-VL-8B-Instruct}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the LoRA checkpoint dir (must contain adapter_config.json + adapter_model.safetensors)}"
SAVE_PATH="${SAVE_PATH:?Set SAVE_PATH to the output dir for the merged model}"

export PYTHONPATH=src:$PYTHONPATH

python src/merge_lora_weights.py \
    --model-path "$MODEL_PATH" \
    --model-base "$MODEL_BASE" \
    --save-model-path "$SAVE_PATH" \
    --safe-serialization
