# R_sem v2: reference-bank semantic reward, rewritten to match the paper's
# ACTUAL Eq. 8-9 exactly (previous version used a generic "correct bank vs
# incorrect bank" formula and guessed B_D's contents wrong — see git history
# / EXPERIMENT_LOG.md 2026-07-24 entry for the correction).
#
# Paper definition (EGVLR, ECCV 2026):
#   Three fixed banks: B_N (normal-supporting), B_A (abnormal-supporting),
#   B_D (domain-oriented — i.e. for domain-only QA samples specifically,
#   which in our data is the "Domain Knowledge MCQ" question type).
#   s_N = max_{u in B_N} cos(z,u), s_A = max_{u in B_A} cos(z,u),
#   s_D = max_{u in B_D} cos(z,u)   (Eq. 8: maximum cosine similarity)
#
#   R_sem = tanh(gamma*(s_A - s_N))            abnormal visual samples
#         = tanh(gamma*(s_N - s_A))            normal / negative-region samples
#         = tanh(gamma*(s_D - max(s_N,s_A)))   domain-only QA           (Eq. 9)
#
# STATUS: written, CPU-testable (BAAI/bge-small-en-v1.5 is small enough to run
# on CPU — same as reward_funcs.py's existing default). See build_banks.py to
# build a bank file, and test_paper_fidelity.py / README.md for wiring
# instructions into reward_funcs.py.

import json
import os
import pickle
from typing import Optional

import numpy as np

DOMAIN_ONLY_QUESTION_TYPE = "Domain Knowledge MCQ"


def _question_type_of(question_text: str) -> str:
    lines = question_text.split("\n")
    i = 0
    while i < len(lines) and lines[i].strip() in ("<image>", ""):
        i += 1
    return lines[i].rstrip(":").strip() if i < len(lines) else "UNKNOWN"


def _is_normal_gt(gt_answer_text: str) -> Optional[bool]:
    """Best-effort: an example is "normal" if its GT <location> is an empty
    list `[]` (no defect boxes) — matches this project's existing convention
    (is_location_valid() in reward_funcs.py uses the same signal). Returns
    None if it can't tell (malformed data), so callers can skip that example
    rather than mis-bucket it."""
    import re

    loc_match = re.search(r"<location>\s*(\[.*?\])\s*</location>", gt_answer_text, re.DOTALL)
    if loc_match is None:
        return None
    try:
        import ast

        boxes = ast.literal_eval(loc_match.group(1))
    except (ValueError, SyntaxError):
        return None
    if not isinstance(boxes, list):
        return None
    return len(boxes) == 0


def extract_think_text_standalone(text: str) -> Optional[str]:
    """Standalone copy of reward_funcs.py's extract_think_text's core logic,
    duplicated here so this module has no import-time dependency on the live
    reward_funcs.py (which needs the full training env with scipy etc.) —
    keeps this buildable/testable from a plain CPU Python env."""
    import re

    m = re.search(r"<think>\s*(.*?)\s*</think>", text, re.DOTALL | re.IGNORECASE)
    if m is None:
        return None
    inner = m.group(1)
    ev = re.search(r"<evidence>\s*(.*?)\s*</evidence>", inner, re.DOTALL | re.IGNORECASE)
    lo = re.search(r"<logic>\s*(.*?)\s*</logic>", inner, re.DOTALL | re.IGNORECASE)
    parts = []
    if ev:
        parts.append(ev.group(1).strip())
    if lo:
        parts.append(lo.group(1).strip())
    return " ".join(parts) if parts else None


def sample_branch(question_text: str, gt_text: str) -> Optional[str]:
    """Which of the paper's three Eq.9 branches this sample belongs to:
    "domain_only", "abnormal", or "normal". Returns None if undecidable
    (malformed GT), so callers can skip it."""
    qtype = _question_type_of(question_text)
    if qtype == DOMAIN_ONLY_QUESTION_TYPE:
        return "domain_only"
    is_normal = _is_normal_gt(gt_text)
    if is_normal is None:
        return None
    return "normal" if is_normal else "abnormal"


def build_reference_banks(
    data_json_paths: list,
    output_path: str,
    model_name: str = "BAAI/bge-small-en-v1.5",
    max_per_bank: int = 500,
    seed: int = 0,
):
    """Walk one or more Stage2/Stage3-style JSON files, bucket GT think-text
    into exactly the paper's three banks (B_N, B_A, B_D — B_D is samples
    whose question type is "Domain Knowledge MCQ", i.e. domain-only QA, NOT
    per-defect-type as an earlier version of this file guessed), embed them,
    and save centroid + a capped sample of raw embeddings to `output_path`.

    max_per_bank caps how many examples get embedded per bank, purely to keep
    this fast to build/test — 500 is already generous; raise it for a "real"
    bank once this is actually being used to train, not just dev-tested.
    """
    import random

    from sentence_transformers import SentenceTransformer

    rng = random.Random(seed)
    buckets: dict[str, list[str]] = {"B_N": [], "B_A": [], "B_D": []}
    branch_to_bank = {"normal": "B_N", "abnormal": "B_A", "domain_only": "B_D"}

    for path in data_json_paths:
        data = json.load(open(path))
        for ex in data:
            conv = ex.get("conversations")
            if not conv or len(conv) < 2:
                continue
            question_text = conv[0]["value"]
            gt_text = conv[1]["value"]

            think = extract_think_text_standalone(gt_text)
            if not think:
                continue

            branch = sample_branch(question_text, gt_text)
            if branch is None:
                continue

            buckets[branch_to_bank[branch]].append(think)

    for key in list(buckets.keys()):
        if len(buckets[key]) > max_per_bank:
            buckets[key] = rng.sample(buckets[key], max_per_bank)

    model = SentenceTransformer(model_name, device="cpu")

    banks = {}
    for key, texts in buckets.items():
        if not texts:
            continue
        embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        banks[key] = {
            "centroid": embeddings.mean(axis=0),
            "embeddings": embeddings,
            "n_examples": len(texts),
        }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump({"model_name": model_name, "banks": banks}, f)

    return {k: v["n_examples"] for k, v in banks.items()}


class ReferenceBankStore:
    def __init__(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model_name = data["model_name"]
        self.banks = data["banks"]  # "B_N"/"B_A"/"B_D" -> {"centroid", "embeddings", "n_examples"}

    def max_similarity(self, embedding: np.ndarray, bank_key: str) -> float:
        """s_N / s_A / s_D per Eq. 8: max cosine similarity to any member of
        the bank (embeddings are pre-normalized, so dot product == cosine)."""
        if bank_key not in self.banks:
            return -1.0  # bank doesn't exist (e.g. never built) -> definitionally not similar
        refs = self.banks[bank_key]["embeddings"]
        sims = refs @ embedding
        return float(sims.max())


def reference_bank_reward_eq9(
    embedding: np.ndarray,
    branch: str,
    store: ReferenceBankStore,
    gamma: float = 4.0,
) -> float:
    """Direct implementation of the paper's Eq. 9, three separate branches —
    NOT a single unified formula, per the paper's own definition. `branch` is
    one of "normal", "abnormal", "domain_only" (see sample_branch()).
    `embedding` must already be L2-normalized."""
    s_N = store.max_similarity(embedding, "B_N")
    s_A = store.max_similarity(embedding, "B_A")

    if branch == "abnormal":
        margin = s_A - s_N
    elif branch == "normal":
        margin = s_N - s_A
    elif branch == "domain_only":
        s_D = store.max_similarity(embedding, "B_D")
        margin = s_D - max(s_N, s_A)
    else:
        raise ValueError(f"Unknown branch: {branch!r}, expected normal/abnormal/domain_only")

    return float(np.tanh(gamma * margin))


# =====================================================
# reward_funcs.py-style entry point (NOT wired in yet — see README.md)
# =====================================================

_STORE_CACHE: dict[str, ReferenceBankStore] = {}


def think_reference_bank_reward(
    completions,
    assistant,
    prompts=None,
    bank_path=None,
    sem_weight=None,
    **kwargs,
):
    """Drop-in replacement candidate for reward_funcs.py's
    `think_similarity_reward` — same `(completions, assistant, ..., **kwargs)
    -> list[float]` signature, so `load_reward_funcs()` can pick it up the
    same way once actually wired in (see this module's docstring / README).

    Needs `prompts` (the question text, to detect domain-only QA via
    sample_branch()) threaded in via kwargs or the explicit argument — not
    yet wired through the real training loop's reward-function call, which
    currently only passes `completions`/`assistant`.

    Requires a prebuilt bank file (see build_reference_banks()) at
    `bank_path` (or env var REFERENCE_BANK_PATH).
    """
    from sentence_transformers import SentenceTransformer

    if bank_path is None:
        bank_path = os.environ.get("REFERENCE_BANK_PATH")
    if not bank_path or not os.path.exists(bank_path):
        # Fail open to zero reward (not a crash) if the bank isn't built yet.
        return [0.0] * len(completions)

    if bank_path not in _STORE_CACHE:
        _STORE_CACHE[bank_path] = ReferenceBankStore(bank_path)
    store = _STORE_CACHE[bank_path]

    if sem_weight is None:
        sem_weight = 0.10  # matches THINK_WEIGHT's current default in reward_funcs.py

    model = SentenceTransformer(store.model_name, device="cpu")

    if prompts is None:
        prompts = kwargs.get("prompts", [None] * len(completions))

    rewards = []
    for completion, gt, prompt in zip(completions, assistant, prompts):
        think = extract_think_text_standalone(to_text_local(completion))
        gt_text = to_text_local(gt)
        question_text = to_text_local(prompt) if prompt is not None else ""

        branch = sample_branch(question_text, gt_text) if think else None
        if think is None or branch is None:
            rewards.append(0.0)
            continue

        embedding = model.encode([think], convert_to_numpy=True, normalize_embeddings=True)[0]
        r = reference_bank_reward_eq9(embedding, branch, store)
        rewards.append(sem_weight * r)

    return rewards


def to_text_local(x):
    if isinstance(x, str):
        return x
    if isinstance(x, list) and x and isinstance(x[0], dict) and "content" in x[0]:
        return x[0]["content"]
    return str(x)
