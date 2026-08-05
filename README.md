# EGVLR: Evidence-Grounded Vision–Language Reinforcement for Anomaly Reasoning

Official code for **EGVLR** (ECCV 2026), a four-stage vision-language reinforcement framework for
industrial anomaly reasoning on [MMAD](https://arxiv.org/abs/2410.09453) (Jiang et al., ICLR 2025).

[Project page](https://leolin65.github.io/EGVLR-project/) · [Paper (PDF)](https://leolin65.github.io/EGVLR-project/paper.pdf) · [Supplementary (PDF)](https://leolin65.github.io/EGVLR-project/supplementary.pdf)

## Overview

EGVLR trains a vision-language model to answer industrial inspection questions (defect presence,
location, type, and consequence) while grounding every answer in explicit visual/domain evidence.
All trainable stages share one output schema, the **Evidence-Driven Diagnostic Protocol (EDDP)**:

```
<think><evidence> e </evidence><logic> l </logic></think><location> B </location><answer> a </answer>
```

where `e` is visual/domain evidence, `l` is diagnostic logic, `B` is a list of normalized
`[x1,y1,x2,y2]` bounding boxes (empty for normal/negative-region/non-spatial samples), and `a` is
the final answer letter. For paired inputs, image order is fixed: first image = query/test image,
second = normal reference image.

### The four stages

| Stage | Name | What it does |
|---|---|---|
| I | **PVE-FT** (Progressive Visual-Evidential Fine-Tuning) | Teaches localized visual grounding using synthetic anomalies (CutPaste/DTD texture insertion), grid localization, local decoy verification, and normal-reference null-hypothesis calibration — before any domain knowledge is introduced. Plain autoregressive CE loss. |
| II | **KG-IT** (Knowledge-Grounded Instruction Tuning) | Injects task semantics and domain knowledge via domain QA, one-normal visual QA, and comparative inspection QA, under the same EDDP schema. Plain autoregressive CE loss. |
| III | **GS-DPO** (Geometry-Semantic Decoupled Preference Optimization) | GRPO-style RL refinement with a decoupled reward: `R = λ_f R_fmt + λ_a R_ans + λ_b R_box + λ_r R_sem + λ_c R_cpl` — format validity, answer correctness, IoU-based box matching (with a redundant-box penalty), sentence-embedding-based rationale-semantics margin (against three fixed reference banks: normal-supporting, abnormal-supporting, domain-oriented), and an answer–location–rationale coupling term. See `main.tex` Eq. 1–9 for the full derivation. |
| IV | **BGSR** (Box-Guided Segmentation Rendering) | Non-trainable: converts predicted boxes into dense masks via an off-the-shelf SAM-family segmentation backend. Does not modify the MLLM's answer or boxes. |

## Repository structure

```
data_pipeline/          Data-conversion scripts for Stage I/II/III (synthetic + real anomaly QA,
                         domain-knowledge QA, comparative QA, dataset merging)
src/                     Core training code — reward decomposition (reward_funcs.py), GRPO
                         entrypoint (train_grpo.py), SFT entrypoint (train_sft.py), the modified
                         GRPO trainer (grpo_trainer.py), and a reward-function smoke test
scripts/                 SBATCH job templates for Stage I/II/III training, LoRA merging, and eval
                         (adapt --account / module / conda lines to your own cluster)
eval/                    lmms-eval task definitions for MMAD (mmad, mmad_1shot, mmad_info,
                         mmad_train) plus a macro-7 scoring script matching the original MMAD
                         paper's metric exactly
paper_fidelity/          Two known paper-vs-code gaps, with fixes written and tested but NOT
                         applied to the reported results — see paper_fidelity/README.md for
                         exactly what differs and why
```

## Setup

This code is a set of training-entrypoint overrides and data/eval scripts meant to be dropped into
a checkout of [2U1/Qwen-VL-Series-Finetune](https://github.com/2U1/Qwen-VL-Series-Finetune) (the
underlying Qwen-VL fine-tuning framework this project builds on):

```bash
git clone https://github.com/2U1/Qwen-VL-Series-Finetune
cd Qwen-VL-Series-Finetune
pip install -r requirements.txt
pip install -r /path/to/this/repo/requirements.txt

# Drop in the EGVLR-specific training code
cp /path/to/this/repo/src/reward_funcs.py src/train/reward_funcs.py
cp /path/to/this/repo/src/train_grpo.py    src/train/train_grpo.py
cp /path/to/this/repo/src/train_sft.py     src/train/train_sft.py
cp /path/to/this/repo/src/grpo_trainer.py  src/trainer/grpo_trainer.py
```

For evaluation, this project uses [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval); copy
`eval/tasks/*` into your `lmms-eval` checkout's `lmms_eval/tasks/` directory.

MMAD's underlying image data (MVTec-AD, MVTec-LOCO, VisA, GoodsAD, DS-MVTec, RealIAD, etc.) is
**not redistributed here** — obtain it from the original dataset providers and the
[MMAD benchmark](https://github.com/jam-cc/MMAD) release, then point the data-pipeline scripts and
`eval/tasks/*/*.yaml` at your local copies.

## Usage

1. **Build Stage I/II/III training data** — see `data_pipeline/stage1_pve_ft/`,
   `data_pipeline/stage2_kg_it/`, `data_pipeline/stage3_gs_dpo/`. Each stage's `merge_stage_dataset.py`
   combines its QA sources into the final training JSON.
2. **Train Stage I → II → III** in sequence — `scripts/slurm/train_stage1_pve_ft.slurm`,
   `train_stage2_kg_it.slurm`, `train_stage3_gs_dpo.slurm`. Each stage initializes from the
   previous stage's merged checkpoint (`scripts/merge_lora.sh` / `scripts/slurm/merge_lora.slurm`
   merges a LoRA adapter back into the base backbone between stages).
3. **Evaluate** with `scripts/slurm/eval_mmad.slurm` (wraps `eval/run_eval.sh`, an `lmms-eval`
   invocation against the `mmad`/`mmad_1shot`/`mmad_info`/`mmad_train` tasks). Use
   `eval/compute_macro7.py` to recompute the original MMAD paper's unweighted 7-task macro-average
   from the resulting per-sample logs.
4. **Stage IV (BGSR)** is a post-hoc rendering step over a trained checkpoint's predicted boxes —
   not part of the trainable pipeline, no separate training script.



## Citation

```bibtex
@inproceedings{lin2026egvlr,
  title     = {EGVLR: Evidence-Grounded Vision--Language Reinforcement for Anomaly Reasoning},
  author    = {Lin, Shih-Chih and Lu, Ying-Heng and Ye, Dong You and Lai, Shang-Hong},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
