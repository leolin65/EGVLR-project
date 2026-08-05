# [Stage III: GS-DPO] Reward decomposition for GRPO preference optimization.
# Implements the paper's Eq. 7 R = lambda_f*R_fmt + lambda_a*R_ans + lambda_b*R_box
#                                + lambda_r*R_sem + lambda_c*R_cpl :
#   format_reward            -> R_fmt  (EDDP tag presence/order)
#   accuracy_reward          -> R_ans  (predicted vs GT <answer>)
#   location_reward          -> R_box  (Hungarian-matched IoU, bbox_score())
#   think_similarity_reward  -> R_sem  (SentenceTransformer cosine vs reference banks)
#   gated_think_answer_reward -> R_cpl (answer correct AND think-similarity passes threshold)
# Weights: ACC_WEIGHT/LOC_WEIGHT/FMT_WEIGHT/THINK_WEIGHT/GATED_THINK_ANSWER_WEIGHT below.
#
# Depends on the surrounding Qwen-VL-Series-Finetune framework (src/train/, src/model/,
# src/dataset.py, src/params.py) — drop this file into that repo's src/train/ directory.

import os
import re
import sys
import ast
import hashlib
import importlib
from contextlib import contextmanager

import numpy as np
from scipy.optimize import linear_sum_assignment

# =====================================================
# Optional: SentenceTransformer
# pip install sentence-transformers
# =====================================================
# This file is designed for TRL/GRPO + DeepSpeed ZeRO-3.
# Main fix:
#   Do NOT let SentenceTransformer/BGE be initialized under HF ZeRO-3 init.
# Otherwise the BGE embedding table can be partitioned and crash with:
#   RuntimeError: 'weight' must be 2-D
# =====================================================

# =====================================================
# Global
# =====================================================

# Default off. Massive printing inside reward functions slows GRPO a lot.
DEBUG = os.environ.get("REWARD_DEBUG", "0").lower() in {"1", "true", "yes", "y"}

# -----------------------------
# Reward weights
# -----------------------------
ACC_WEIGHT = 0.40
LOC_WEIGHT = 0.25
FMT_WEIGHT = 0.15
THINK_WEIGHT = 0.10
GATED_THINK_ANSWER_WEIGHT = 0.10
# Total = 1.00 by default

# -----------------------------
# Think similarity config
# -----------------------------
THINK_MODEL_NAME = os.environ.get("THINK_SIM_MODEL", "BAAI/bge-small-en-v1.5")

# IMPORTANT:
# Default to CPU for reward BGE because it is safest with ZeRO-3 and avoids
# occupying training GPU memory. If you really want GPU, export:
#   export THINK_SIM_DEVICE=cuda
THINK_DEVICE = os.environ.get("THINK_SIM_DEVICE", os.environ.get("THINK_DEVICE", "cpu"))

THINK_BATCH_SIZE = int(os.environ.get("THINK_SIM_BATCH_SIZE", "32"))

# Recommended:
# 0.65 = loose
# 0.70 = balanced
# 0.75 = strict
THINK_SIM_THRESHOLD = float(os.environ.get("THINK_SIM_THRESHOLD", "0.70"))

# If True: gated reward = weight * similarity
# If False: gated reward = full weight once threshold is passed
GATED_USE_SIM_SCORE = os.environ.get("GATED_USE_SIM_SCORE", "0").lower() in {"1", "true", "yes", "y"}

# If BGE fails for any reason, lexical fallback prevents the whole GRPO job
# from crashing. Set to 0 to make failure return 0.0 instead.
THINK_LEXICAL_FALLBACK = os.environ.get("THINK_LEXICAL_FALLBACK", "1").lower() in {"1", "true", "yes", "y"}

# Cache SentenceTransformer instances by (model_name, device).
_THINK_MODELS = {}

# Cache latest similarities because TRL may call think_similarity_reward and
# gated_think_answer_reward separately on the same completions.
_LAST_SIM_CACHE = {
    "key": None,
    "sims": None,
}


def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


# =====================================================
# Basic text helpers
# =====================================================

def to_text(x):
    """
    Make reward functions robust to either:
    - plain string
    - dict with content
    - chat-style list of dicts
    """

    if isinstance(x, str):
        return x

    if isinstance(x, dict):
        return x.get("content", str(x))

    if isinstance(x, list):
        if len(x) > 0 and isinstance(x[-1], dict) and "content" in x[-1]:
            return x[-1]["content"]
        return "\n".join(to_text(v) for v in x)

    return str(x)


def normalize_space(text):
    text = to_text(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def find_single_tag(text, tag):
    """
    Extract content from exactly one XML-like tag.

    Example:
        find_single_tag(text, "answer")
    """

    text = to_text(text)

    pattern = re.compile(
        rf"<{tag}>\s*(.*?)\s*</{tag}>",
        flags=re.DOTALL | re.IGNORECASE,
    )

    matches = pattern.findall(text)

    if len(matches) != 1:
        return None

    return matches[0].strip()


def find_tag_span(text, tag):
    text = to_text(text)

    pattern = re.compile(
        rf"<{tag}>\s*.*?\s*</{tag}>",
        flags=re.DOTALL | re.IGNORECASE,
    )

    match = pattern.search(text)

    if match is None:
        return None

    return match.span()


def count_tag(text, tag):
    text = to_text(text)

    open_tags = re.findall(
        rf"<{tag}>",
        text,
        flags=re.IGNORECASE,
    )

    close_tags = re.findall(
        rf"</{tag}>",
        text,
        flags=re.IGNORECASE,
    )

    return len(open_tags), len(close_tags)


# =====================================================
# Answer Helper
# =====================================================

def extract_answer(text):
    """
    Extract exactly one answer from:

        <answer>A</answer>

    Returns:
        A/B/C/D or None
    """

    text = to_text(text)

    matches = re.findall(
        r"<answer>\s*([A-D])\s*</answer>",
        text,
        flags=re.IGNORECASE,
    )

    if len(matches) != 1:
        return None

    return matches[0].upper()


# =====================================================
# Think Helper
# =====================================================

def extract_think_text(text):
    """
    Extract semantic text from:

        <think>
        <evidence>...</evidence>
        <logic>...</logic>
        </think>

    Returns normalized evidence + logic string.
    """

    text = to_text(text)

    think_content = find_single_tag(text, "think")

    if think_content is None:
        return None

    evidence = find_single_tag(think_content, "evidence")
    logic = find_single_tag(think_content, "logic")

    if evidence is None or logic is None:
        return None

    evidence = normalize_space(evidence)
    logic = normalize_space(logic)

    if len(evidence) == 0 or len(logic) == 0:
        return None

    return normalize_space(f"evidence: {evidence}\nlogic: {logic}")


def _resolve_think_device(device=None):
    """
    Resolve SentenceTransformer device.

    In GRPO + ZeRO-3, CPU is the safest default.
    """

    if device is None:
        device = os.environ.get("THINK_SIM_DEVICE", os.environ.get("THINK_DEVICE", THINK_DEVICE))

    if device is None:
        return "cpu"

    device = str(device).strip()

    if device == "" or device.lower() == "none":
        return "cpu"

    if device.lower() == "auto":
        # Auto inside a ZeRO-3 process may choose CUDA and cause memory pressure.
        # Keep it safe unless user explicitly requests CUDA.
        return "cpu"

    return device


@contextmanager
def _temporarily_disable_hf_deepspeed_zero3_init():
    """
    Temporarily disable HF DeepSpeed global config while loading BGE.

    Why:
      When transformers Trainer is launched with DeepSpeed ZeRO-3,
      from_pretrained() can be wrapped by deepspeed.zero.Init. If BGE is lazy-loaded
      inside a reward function, its BERT embedding matrix may be partitioned too.
      Then torch.nn.functional.embedding() sees a non-2D weight and crashes:
          RuntimeError: 'weight' must be 2-D

    This context only affects the SentenceTransformer construction window and
    restores the original HF DeepSpeed state immediately afterwards.
    """

    saved = []
    module_names = [
        "transformers.integrations.deepspeed",
        "transformers.deepspeed",  # older transformers fallback
    ]

    for module_name in module_names:
        try:
            mod = importlib.import_module(module_name)
        except Exception:
            continue

        if hasattr(mod, "_hf_deepspeed_config_weak_ref"):
            try:
                saved.append((mod, getattr(mod, "_hf_deepspeed_config_weak_ref")))
                setattr(mod, "_hf_deepspeed_config_weak_ref", None)
            except Exception:
                pass

    try:
        yield
    finally:
        for mod, old_value in saved:
            try:
                setattr(mod, "_hf_deepspeed_config_weak_ref", old_value)
            except Exception:
                pass


def _check_sentence_transformer_embedding_weight(model):
    """
    Sanity check for the BGE word embedding weight.
    A normal BERT embedding table must be 2-D: [vocab_size, hidden_dim].
    """

    try:
        weight = model[0].auto_model.embeddings.word_embeddings.weight
    except Exception:
        return

    try:
        dim = weight.dim()
        shape = tuple(weight.shape)
    except Exception:
        return

    if dim != 2:
        raise RuntimeError(
            "SentenceTransformer embedding weight is not 2-D. "
            f"dim={dim}, shape={shape}. "
            "This usually means the reward BGE model was loaded under "
            "DeepSpeed ZeRO-3 parameter partitioning."
        )


def _load_sentence_transformer_safely(model_name, device):
    """
    Load SentenceTransformer without allowing HF ZeRO-3 to partition it.
    """

    from sentence_transformers import SentenceTransformer

    resolved_device = _resolve_think_device(device)

    with _temporarily_disable_hf_deepspeed_zero3_init():
        model = SentenceTransformer(model_name, device=resolved_device)

    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    _check_sentence_transformer_embedding_weight(model)

    return model


def get_think_model(model_name=None, device=None):
    """
    Lazy-load SentenceTransformer only once per (model_name, device).

    Safe for DeepSpeed ZeRO-3:
      - temporarily disables HF ZeRO-3 init during BGE construction
      - defaults to CPU
      - verifies BGE embedding weight remains 2-D
    """

    if model_name is None:
        model_name = os.environ.get("THINK_SIM_MODEL", THINK_MODEL_NAME)

    device = _resolve_think_device(device)

    key = (model_name, device)

    if key not in _THINK_MODELS:
        debug_print(f"Loading think similarity model: {model_name} on {device}")
        _THINK_MODELS[key] = _load_sentence_transformer_safely(model_name, device)

    return _THINK_MODELS[key]


def _texts_cache_key(completions, assistant, model_name, device, batch_size):
    h = hashlib.sha1()
    h.update(str(model_name).encode("utf-8"))
    h.update(str(device).encode("utf-8"))
    h.update(str(batch_size).encode("utf-8"))

    for x in completions:
        h.update(to_text(x).encode("utf-8", errors="ignore"))
        h.update(b"\0")

    h.update(b"assistant")

    for x in assistant:
        h.update(to_text(x).encode("utf-8", errors="ignore"))
        h.update(b"\0")

    return h.hexdigest()


def _tokenize_for_fallback(text):
    text = normalize_space(text).lower()
    # Keep numbers because locations/regions may appear in generated think.
    return set(re.findall(r"[a-z0-9_]+", text))


def _lexical_similarity(a, b):
    """
    Cheap fallback when SentenceTransformer fails.
    This is not as good as BGE but prevents long Slurm jobs from crashing.
    """

    sa = _tokenize_for_fallback(a)
    sb = _tokenize_for_fallback(b)

    if len(sa) == 0 or len(sb) == 0:
        return 0.0

    return float(len(sa & sb) / len(sa | sb))


def _compute_lexical_think_similarities(completions, assistant):
    sims = [0.0 for _ in completions]

    for i, (completion, gt) in enumerate(zip(completions, assistant)):
        pred_think = extract_think_text(completion)
        gt_think = extract_think_text(gt)

        if pred_think is None or gt_think is None:
            sims[i] = 0.0
        else:
            sims[i] = max(0.0, min(1.0, _lexical_similarity(pred_think, gt_think)))

    return sims


def compute_think_similarities(
    completions,
    assistant,
    model_name=None,
    device=None,
    batch_size=None,
):
    """
    Return cosine similarities between predicted think and GT think.

    Missing / malformed think -> similarity = 0.0

    ZeRO-3 safety:
      - BGE is loaded through get_think_model()
      - if BGE encode still fails, fallback prevents training crash
    """

    if batch_size is None:
        batch_size = THINK_BATCH_SIZE

    if model_name is None:
        model_name = os.environ.get("THINK_SIM_MODEL", THINK_MODEL_NAME)

    device = _resolve_think_device(device)

    # Reuse latest sims when both think rewards are called on the same batch.
    cache_key = _texts_cache_key(completions, assistant, model_name, device, batch_size)
    if _LAST_SIM_CACHE["key"] == cache_key and _LAST_SIM_CACHE["sims"] is not None:
        return list(_LAST_SIM_CACHE["sims"])

    pred_thinks = []
    gt_thinks = []
    valid_indices = []

    sims = [0.0 for _ in completions]

    for i, (completion, gt) in enumerate(zip(completions, assistant)):
        pred_think = extract_think_text(completion)
        gt_think = extract_think_text(gt)

        if pred_think is None or gt_think is None:
            sims[i] = 0.0
            continue

        pred_thinks.append(pred_think)
        gt_thinks.append(gt_think)
        valid_indices.append(i)

    if len(valid_indices) == 0:
        _LAST_SIM_CACHE["key"] = cache_key
        _LAST_SIM_CACHE["sims"] = list(sims)
        return sims

    all_texts = []
    for p, g in zip(pred_thinks, gt_thinks):
        all_texts.append(p)
        all_texts.append(g)

    try:
        model = get_think_model(model_name=model_name, device=device)

        embeddings = model.encode(
            all_texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        for k, original_idx in enumerate(valid_indices):
            pred_emb = embeddings[2 * k]
            gt_emb = embeddings[2 * k + 1]

            sim = float(np.dot(pred_emb, gt_emb))
            sim = max(0.0, min(1.0, sim))
            sims[original_idx] = sim

    except RuntimeError as e:
        # Recover specifically from ZeRO-3 partitioned BGE or CUDA OOM.
        err = str(e)
        debug_print("BGE think similarity failed with RuntimeError:", err)

        # Clear possibly bad cached model.
        _THINK_MODELS.pop((model_name, device), None)

        # If user tried CUDA, retry once on CPU.
        if device != "cpu":
            try:
                debug_print("Retrying think similarity model on CPU")
                model = get_think_model(model_name=model_name, device="cpu")
                embeddings = model.encode(
                    all_texts,
                    batch_size=batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )

                for k, original_idx in enumerate(valid_indices):
                    pred_emb = embeddings[2 * k]
                    gt_emb = embeddings[2 * k + 1]
                    sim = float(np.dot(pred_emb, gt_emb))
                    sims[original_idx] = max(0.0, min(1.0, sim))

            except Exception as e2:
                debug_print("CPU retry failed:", repr(e2))
                if THINK_LEXICAL_FALLBACK:
                    sims = _compute_lexical_think_similarities(completions, assistant)
                else:
                    sims = [0.0 for _ in completions]
        else:
            if THINK_LEXICAL_FALLBACK:
                sims = _compute_lexical_think_similarities(completions, assistant)
            else:
                sims = [0.0 for _ in completions]

    except Exception as e:
        debug_print("BGE think similarity failed:", repr(e))
        _THINK_MODELS.pop((model_name, device), None)
        if THINK_LEXICAL_FALLBACK:
            sims = _compute_lexical_think_similarities(completions, assistant)
        else:
            sims = [0.0 for _ in completions]

    _LAST_SIM_CACHE["key"] = cache_key
    _LAST_SIM_CACHE["sims"] = list(sims)

    return sims


# =====================================================
# Accuracy Reward
# =====================================================

def accuracy_reward(
    completions,
    assistant,
    acc_weight=None,
    **kwargs,
):
    """
    Compare predicted answer with GT answer.

    completion:
        <think>...</think>
        <answer>A</answer><location>...</location>

    assistant:
        <think>...</think>
        <answer>A</answer><location>...</location>
    """

    if acc_weight is None:
        acc_weight = ACC_WEIGHT

    rewards = []

    for completion, gt in zip(completions, assistant):
        pred_answer = extract_answer(completion)
        gt_answer = extract_answer(gt)

        correct = (
            pred_answer is not None
            and gt_answer is not None
            and pred_answer == gt_answer
        )

        rewards.append(acc_weight * float(correct))

    return rewards


# =====================================================
# Location Helpers
# =====================================================

def extract_location(text):
    """
    Extract bbox list from:

        <location>[[x1,y1,x2,y2], ...]</location>

    [] is considered valid prediction for no anomaly.
    """

    text = to_text(text)

    matches = re.findall(
        r"<location>\s*(.*?)\s*</location>",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if len(matches) != 1:
        debug_print("Location tag count != 1")
        return None

    content = matches[0].strip()

    try:
        boxes = ast.literal_eval(content)

        if not isinstance(boxes, list):
            debug_print("Location is not a list:", boxes)
            return None

        # [] is valid.
        normalized_boxes = []

        for box in boxes:
            # Be slightly permissive: accept tuple but convert to list.
            if isinstance(box, tuple):
                box = list(box)

            if not isinstance(box, list):
                debug_print("Box is not list:", box)
                return None

            if len(box) != 4:
                debug_print("Box length != 4:", box)
                return None

            if not all(isinstance(v, (int, float)) for v in box):
                debug_print("Box contains non-number:", box)
                return None

            x1, y1, x2, y2 = [float(v) for v in box]

            if x2 < x1 or y2 < y1:
                debug_print("Invalid bbox order:", box)
                return None

            normalized_boxes.append([x1, y1, x2, y2])

        return normalized_boxes

    except Exception as e:
        debug_print("literal_eval error:", e)
        return None


def bbox_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union = area1 + area2 - inter

    if union <= 0:
        return 0.0

    return float(inter / union)


def bbox_score(gt_boxes, pred_boxes, iou_thresh=0.5):
    debug_print("\n========== bbox_score ==========")
    debug_print("GT boxes:", gt_boxes)
    debug_print("Pred boxes:", pred_boxes)

    # both empty
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return 1.0

    # false positive: GT says normal/empty but the model hallucinated a box.
    # Paper Eq.7: Rbox = -1 here (an explicit penalty, not just "no reward"),
    # since this is exactly the failure mode FPRimg/FPBox measure.
    if len(gt_boxes) == 0:
        return -1.0

    # false negative
    if len(pred_boxes) == 0:
        return 0.0

    n_gt = len(gt_boxes)
    n_pred = len(pred_boxes)

    iou_matrix = np.zeros((n_gt, n_pred), dtype=np.float32)

    for i, gt in enumerate(gt_boxes):
        for j, pred in enumerate(pred_boxes):
            iou_matrix[i, j] = bbox_iou(gt, pred)

    cost = 1.0 - iou_matrix
    gt_idx, pred_idx = linear_sum_assignment(cost)

    matched_iou = iou_matrix[gt_idx, pred_idx]
    matched_iou = np.where(matched_iou >= iou_thresh, matched_iou, 0.0)

    score = float(matched_iou.sum() / max(n_gt, n_pred))

    debug_print("IoU Matrix:", iou_matrix)
    debug_print("Matched IoU:", matched_iou)
    debug_print("Final score:", score)

    return score


# =====================================================
# Format Reward
# =====================================================

def format_reward(
    completions,
    fmt_weight=None,
    **kwargs,
):
    """
    Progressive format reward.

    Required format:

        <think>
        <evidence>...</evidence>
        <logic>...</logic>
        </think>
        <answer>A</answer><location>[[x1,y1,x2,y2],...]</location>

    For normal images:

        <think>
        <evidence>...</evidence>
        <logic>...</logic>
        </think>
        <answer>B</answer><location>[]</location>
    """

    if fmt_weight is None:
        fmt_weight = FMT_WEIGHT

    rewards = []

    for completion in completions:
        completion = to_text(completion)
        reward = 0.0

        # 1. exactly one think block
        think_open, think_close = count_tag(completion, "think")
        has_one_think = think_open == 1 and think_close == 1
        if has_one_think:
            reward += fmt_weight * 0.15

        # 2. evidence / logic inside think
        think_content = find_single_tag(completion, "think")
        has_evidence = False
        has_logic = False

        if think_content is not None:
            ev_open, ev_close = count_tag(think_content, "evidence")
            logic_open, logic_close = count_tag(think_content, "logic")
            has_evidence = ev_open == 1 and ev_close == 1
            has_logic = logic_open == 1 and logic_close == 1

        if has_evidence:
            reward += fmt_weight * 0.15

        if has_logic:
            reward += fmt_weight * 0.15

        # 3. answer
        pred_answer = extract_answer(completion)
        if pred_answer is not None:
            reward += fmt_weight * 0.15

        # 4. location
        pred_location = extract_location(completion)
        if pred_location is not None:
            reward += fmt_weight * 0.20

        # 5. correct order: think -> answer -> location
        think_span = find_tag_span(completion, "think")
        answer_span = find_tag_span(completion, "answer")
        location_span = find_tag_span(completion, "location")

        order_ok = (
            think_span is not None
            and answer_span is not None
            and location_span is not None
            and think_span[1] <= answer_span[0]
            and answer_span[1] <= location_span[0]
        )

        if order_ok:
            reward += fmt_weight * 0.20

        rewards.append(float(reward))

    return rewards


# =====================================================
# Location Reward
# =====================================================

def location_reward(
    completions,
    assistant,
    loc_weight=None,
    **kwargs,
):
    """
    Compare predicted bbox with GT bbox.

    reward:
        loc_weight * Hungarian matching IoU score
    """

    if loc_weight is None:
        loc_weight = LOC_WEIGHT

    rewards = []

    for completion, gt in zip(completions, assistant):
        pred_boxes = extract_location(completion)
        gt_boxes = extract_location(gt)

        if pred_boxes is None or gt_boxes is None:
            rewards.append(0.0)
            continue

        score = bbox_score(gt_boxes, pred_boxes)
        rewards.append(float(loc_weight * score))

    return rewards


# =====================================================
# Think Similarity Reward
# =====================================================

_REFERENCE_BANK_STORE_CACHE = {}


def _paper_fidelity_dir():
    d = os.path.join(os.path.dirname(os.path.realpath(__file__)), "paper_fidelity")
    if d not in sys.path:
        sys.path.insert(0, d)
    return d


def compute_reference_bank_think_rewards(
    completions, assistant, prompts, bank_path, model_name=None, device=None
):
    """
    Paper-faithful R_sem (Eq. 8-9): tanh(gamma * margin) between the max
    cosine similarity of the predicted <think> text to the closest example
    in each of the three FIXED reference banks (B_N/B_A/B_D built from real
    training data, see paper_fidelity/build_banks.py) -- NOT a direct cosine
    to this sample's own GT <think> text (that's compute_think_similarities,
    used when REFERENCE_BANK_PATH is unset). Values are in [-1, 1], unlike
    the direct-cosine path's [0, 1].
    """
    _paper_fidelity_dir()
    from reference_banks import sample_branch, reference_bank_reward_eq9

    if bank_path not in _REFERENCE_BANK_STORE_CACHE:
        from reference_banks import ReferenceBankStore

        _REFERENCE_BANK_STORE_CACHE[bank_path] = ReferenceBankStore(bank_path)
    store = _REFERENCE_BANK_STORE_CACHE[bank_path]

    model = get_think_model(model_name=model_name, device=_resolve_think_device(device))

    rewards = [0.0] * len(completions)
    pending_idx = []
    pending_think = []

    for i, (completion, gt, prompt) in enumerate(zip(completions, assistant, prompts)):
        pred_think = extract_think_text(completion)
        if pred_think is None:
            continue
        branch = sample_branch(to_text(prompt), to_text(gt))
        if branch is None:
            continue
        pending_idx.append((i, branch))
        pending_think.append(pred_think)

    if not pending_think:
        return rewards

    try:
        embeddings = model.encode(
            pending_think, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
    except Exception as e:
        debug_print("Reference-bank think embedding failed:", repr(e))
        return rewards

    for (i, branch), emb in zip(pending_idx, embeddings):
        rewards[i] = reference_bank_reward_eq9(emb, branch, store)

    return rewards


def think_similarity_reward(
    completions,
    assistant,
    think_weight=None,
    think_model_name=None,
    think_device=None,
    prompts=None,
    **kwargs,
):
    """
    Semantic reward between generated <think> and GT <think>.

    It compares:
        <evidence>...</evidence>
        <logic>...</logic>

    using SentenceTransformer cosine similarity.

    reward:
        think_weight * cosine_similarity

    If REFERENCE_BANK_PATH is set, uses the paper's actual Eq. 8-9 reference-
    bank formula instead (compute_reference_bank_think_rewards) -- an
    ablation switch, not a default, so existing runs are unaffected unless
    the env var is explicitly set.
    """

    if think_weight is None:
        think_weight = THINK_WEIGHT

    if prompts is None:
        prompts = kwargs.get("prompts")

    bank_path = os.environ.get("REFERENCE_BANK_PATH")
    if bank_path:
        rewards = compute_reference_bank_think_rewards(
            completions=completions,
            assistant=assistant,
            prompts=prompts if prompts is not None else [None] * len(completions),
            bank_path=bank_path,
            model_name=think_model_name,
            device=think_device,
        )
        return [float(think_weight * r) for r in rewards]

    sims = compute_think_similarities(
        completions=completions,
        assistant=assistant,
        model_name=think_model_name,
        device=think_device,
    )

    return [float(think_weight * sim) for sim in sims]


def is_location_valid(pred_boxes, gt_boxes, iou_thresh=0.5):
    """
    LocValid, per paper Eq.10's coupling-reward definition:
      - GT has no <location> tag at all (gt_boxes is None) -> non-spatial QA;
        location validity is trivially satisfied regardless of prediction.
      - GT location is empty (normal / negative-region) -> valid iff the
        prediction is also empty (no hallucinated box).
      - GT location is non-empty (abnormal, spatial supervision) -> valid iff
        the prediction is parseable, non-empty, and matches at least one GT
        box with IoU >= iou_thresh (bbox_score > 0).
    Malformed/unparseable predictions (pred_boxes is None) are never valid
    once GT expects a specific (empty or non-empty) location.
    """
    if gt_boxes is None:
        return True

    if pred_boxes is None:
        return False

    if len(gt_boxes) == 0:
        return len(pred_boxes) == 0

    if len(pred_boxes) == 0:
        return False

    return bbox_score(gt_boxes, pred_boxes, iou_thresh=iou_thresh) > 0.0


# =====================================================
# Gated Think + Answer Reward
# =====================================================

def gated_think_answer_reward(
    completions,
    assistant,
    gated_weight=None,
    sim_threshold=None,
    use_sim_score=None,
    think_model_name=None,
    think_device=None,
    **kwargs,
):
    """
    Extra reward only when (paper Eq.10, R_cpl = I(answer correct) x I(LocValid) x I(SemValid)):
        1. answer is correct
        2. predicted location is valid for this sample type (is_location_valid())
        3. think cosine similarity >= threshold (our proxy for SemValid)

    Default:
        reward = gated_weight

    If use_sim_score=True:
        reward = gated_weight * cosine_similarity
    """

    if gated_weight is None:
        gated_weight = GATED_THINK_ANSWER_WEIGHT

    if sim_threshold is None:
        sim_threshold = THINK_SIM_THRESHOLD

    if use_sim_score is None:
        use_sim_score = GATED_USE_SIM_SCORE

    sims = compute_think_similarities(
        completions=completions,
        assistant=assistant,
        model_name=think_model_name,
        device=think_device,
    )

    rewards = []

    for completion, gt, sim in zip(completions, assistant, sims):
        pred_answer = extract_answer(completion)
        gt_answer = extract_answer(gt)

        answer_correct = (
            pred_answer is not None
            and gt_answer is not None
            and pred_answer == gt_answer
        )

        think_pass = sim >= sim_threshold

        pred_boxes = extract_location(completion)
        gt_boxes = extract_location(gt)
        loc_valid = is_location_valid(pred_boxes, gt_boxes)

        if answer_correct and loc_valid and think_pass:
            reward = gated_weight * sim if use_sim_score else gated_weight
        else:
            reward = 0.0

        rewards.append(float(reward))

    return rewards
