# Paper-fidelity fixes

Everything in this folder implements a mechanism the EGVLR paper **itself specifies with a
citable equation number**, that the released training code did not actually match. These are
gap-fixes, not new ideas.

Status: code written and tested, **not wired into the live `reward_funcs.py`/training pipeline
for the reported results** (a deliberate, reversible "later" decision, not an oversight). The
reference banks have been built for real (see below); wiring the reward function into the live
trainer is the one remaining step.

## 1. `reference_banks.py` + `build_banks.py` — R_sem v2 (paper Eq. 8-9)

Replaces `reward_funcs.py`'s current `think_similarity_reward` (direct cosine similarity between
a completion's `<think>` and THIS SAMPLE's own GT think) with the paper's actual design: three
fixed reference banks (B_N normal-supporting, B_A abnormal-supporting, B_D domain-oriented — i.e.
"Domain Knowledge MCQ" samples specifically) and a three-branch margin/tanh reward (Eq. 9):

```
R_sem = tanh(γ(s_A−s_N))              abnormal visual samples
      = tanh(γ(s_N−s_A))              normal/negative-region samples
      = tanh(γ(s_D−max(s_N,s_A)))     domain-only QA
```

where s_N/s_A/s_D are MAXIMUM cosine similarity (Eq. 8) to any single member of the corresponding
bank — not centroid distance — so a bank's phrasing diversity doesn't get averaged away.

**Build a bank** with `build_banks.py` against your own Stage2/Stage3 data (set
`STAGE2_DATA_PATH`/`STAGE3_DATA_PATH`, or pass `--data`), capped at 500 examples/bank by default
(B_N=500, B_A=500, B_D=500). This is a CPU-only embedding job (BAAI/bge-small-en-v1.5), takes under
a minute:

```
python build_banks.py --out ./ref_banks_v1.pkl
```

Verified (real, non-mocked SentenceTransformer, on both a synthetic test and a real bank):
normal-style text scores higher graded as "normal" than as "abnormal" (0.824 vs -0.824),
abnormal-style text is the mirror (0.787 vs -0.787), domain-style text scores positive on the
domain_only branch while abnormal-style text scores negative on it (0.756 vs -0.912) — all three
of Eq. 9's branches discriminate in the correct direction.

### Wiring this into the live trainer

Traced `_calculate_rewards` in TRL's `GRPOTrainer` (the live trainer's parent class): it already
calls every reward function as
`reward_func(prompts=prompts, completions=completions, completion_ids=..., **reward_kwargs)` —
`prompts` (the raw question text, needed to detect "Domain Knowledge MCQ" via `sample_branch()`)
is passed automatically to every reward function today. **No trainer change is needed.**

`load_reward_funcs()` (`src/utils.py`) auto-discovers every `*_reward`-named callable that is a
direct member of the `train.reward_funcs` module, in source-line order. To activate:

1. In `reward_funcs.py`, add near the top:
   ```python
   import sys
   sys.path.insert(0, "/path/to/paper_fidelity")
   from reference_banks import think_reference_bank_reward
   ```
2. Either rename `think_similarity_reward` to something not ending in `_reward` (so
   `load_reward_funcs` stops picking it up) and let `think_reference_bank_reward` take its place
   in the discovered list, or keep both live side by side for an A/B ablation (in which case halve
   `sem_weight`/`THINK_WEIGHT` so their combined contribution doesn't double-count against the
   paper's single λ_r term).
3. Set `export REFERENCE_BANK_PATH=/path/to/ref_banks_v1.pkl` in the training script (or pass
   `bank_path=` directly) before launching training.
4. `think_reference_bank_reward` fails open to `[0.0]*len(completions)` if the bank file is
   missing — safe to land this edit even before the bank exists, and safe against any path typo.

## 2. `eddp_v2_schema.py` — EDDP tag-order fix (paper Eq. 1: think → location → answer)

The paper's order vs. the current codebase's think → answer → location. `format_reward_eddp_v2()`
is a full duplicate of `reward_funcs.py`'s `format_reward()` with only the order-check condition
flipped. Verified: correct new-order completions score full marks (1.0 on a fmt_weight=1.0 test),
and old-order completions lose exactly the 0.20 order-bonus fraction and nothing else (0.80/1.0).

**Not done for the reported results**: no Stage1/2/3 data has been regenerated in the new order —
this is a *grader*, useful once new-order data exists, not a data migration. Actually switching
over means editing the `ANSWER_TEMPLATE`/`QUESTION_TEMPLATE`-equivalent constants in
`data_pipeline/*/convert_*.py` to the new order (see `ANSWER_TEMPLATE_V2`/`QUESTION_INSTRUCTION_V2`
at the top of this file) and rerunning the builders against the same raw sources — then retraining
Stage1→2→3 so the model is actually shown the new order, not just graded on it. That is a full
retrain cycle, scoped as future work.

## Running the tests

```
cd paper_fidelity
python test_paper_fidelity.py
```
5/5 passing.
