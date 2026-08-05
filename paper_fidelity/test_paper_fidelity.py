# Dev-scale (CPU-only) tests for mechanisms the EGVLR PAPER ITSELF describes
# (cite-able equation numbers) but the released reward_funcs.py did not
# actually implement — these are gap-fixes, not new ideas.
#
#   1. reference_banks.py  (R_sem v2, paper Eq. 8-9)
#   2. eddp_v2_schema.py    (EDDP tag order, paper Eq. 1)
#
# Run: python test_paper_fidelity.py   (from within this directory, `train`
# conda env activated — needs sentence-transformers + numpy for the
# reference-bank test; everything else is pure stdlib).

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eddp_v2_schema import format_reward_eddp_v2, extract_answer_v2, extract_location_v2


# =====================================================
# EDDP v2 (tag order, Eq. 1) tests
# =====================================================

def test_eddp_v2_correct_order_scores_full_order_bonus():
    completion = (
        "<think>\n<evidence>e</evidence>\n<logic>l</logic>\n</think>\n"
        "<location>[[1,2,3,4]]</location><answer>B</answer>"
    )
    rewards = format_reward_eddp_v2([completion], fmt_weight=1.0)
    assert rewards[0] == 1.0, rewards  # all 6 sub-checks should pass -> full weight


def test_eddp_v2_old_order_loses_order_bonus_only():
    # think -> answer -> location (the CURRENT codebase's order) should fail
    # ONLY the order check under the v2 grader, not the other 5 sub-checks.
    completion = (
        "<think>\n<evidence>e</evidence>\n<logic>l</logic>\n</think>\n"
        "<answer>B</answer><location>[[1,2,3,4]]</location>"
    )
    rewards = format_reward_eddp_v2([completion], fmt_weight=1.0)
    # 0.15+0.15+0.15+0.15+0.20 = 0.80 (all but the 0.20 order bonus)
    assert abs(rewards[0] - 0.80) < 1e-9, rewards[0]


def test_eddp_v2_extract_answer_and_location_unaffected_by_order():
    completion = "<location>[]</location><answer>A</answer>"
    assert extract_answer_v2(completion) == "A"
    assert extract_location_v2(completion) == []


def test_eddp_v2_malformed_location_fails_open_to_none():
    completion = "<location>not a list</location><answer>A</answer>"
    assert extract_location_v2(completion) is None


# =====================================================
# Reference banks tests (Eq. 8-9; needs sentence-transformers, CPU, real model)
# =====================================================

def test_reference_banks_build_and_score(tmp_dir=None):
    import json

    from reference_banks import (
        build_reference_banks,
        ReferenceBankStore,
        reference_bank_reward_eq9,
        sample_branch,
    )
    from sentence_transformers import SentenceTransformer

    tmp_dir = tmp_dir or tempfile.mkdtemp(prefix="egvlr_bank_test_")
    data_path = os.path.join(tmp_dir, "fake_stage2.json")
    bank_path = os.path.join(tmp_dir, "banks.pkl")

    # Hand-built tiny dataset covering all three of the paper's Eq.9
    # branches: normal, abnormal, and domain-only QA (B_D).
    fake_data = []
    normal_think = "<think><evidence>The surface is smooth and uniform, matching the normal reference.</evidence><logic>No anomaly is present.</logic></think>"
    abnormal_think = "<think><evidence>There is a visible crack near the edge, unlike the reference sample.</evidence><logic>This is a crack-type defect.</logic></think>"
    domain_think = "<think><evidence>Bottles in this category are typically inspected for cap seal integrity and label alignment.</evidence><logic>This is general domain knowledge about the object category.</logic></think>"

    for i in range(5):
        fake_data.append({
            "conversations": [
                {"from": "human", "value": "<image>\nAnomaly Detection:\n\nIs there a defect?"},
                {"from": "gpt", "value": f"{normal_think}<answer>A</answer><location>[]</location>"},
            ]
        })
    for i in range(5):
        fake_data.append({
            "conversations": [
                {"from": "human", "value": "<image>\nDefect Classification:\n\nWhat type?"},
                {"from": "gpt", "value": f"{abnormal_think}<answer>B</answer><location>[[1,2,3,4]]</location>"},
            ]
        })
    for i in range(5):
        fake_data.append({
            "conversations": [
                {"from": "human", "value": "<image>\nDomain Knowledge MCQ:\n\nWhat should inspectors check?"},
                {"from": "gpt", "value": f"{domain_think}<answer>C</answer><location>[]</location>"},
            ]
        })

    json.dump(fake_data, open(data_path, "w"))

    bank_sizes = build_reference_banks([data_path], bank_path, max_per_bank=100)
    assert bank_sizes.get("B_N", 0) == 5, bank_sizes
    assert bank_sizes.get("B_A", 0) == 5, bank_sizes
    assert bank_sizes.get("B_D", 0) == 5, bank_sizes  # domain-only QA, NOT per-defect-type

    # Sanity-check branch classification itself (Domain Knowledge MCQ ->
    # domain_only regardless of its own <location>, which happens to be [] here).
    assert sample_branch("<image>\nDomain Knowledge MCQ:\n\nWhat?", "<location>[]</location>") == "domain_only"
    assert sample_branch("<image>\nAnomaly Detection:\n\nIs there a defect?", "<location>[]</location>") == "normal"
    assert sample_branch("<image>\nDefect Classification:\n\nWhat?", "<location>[[1,2,3,4]]</location>") == "abnormal"

    store = ReferenceBankStore(bank_path)
    model = SentenceTransformer(store.model_name, device="cpu")

    normal_like_text = "The surface looks smooth and consistent with the normal reference, no anomaly visible."
    abnormal_like_text = "There is a crack near the edge that does not match the reference sample."
    domain_like_text = "Inspectors should check the cap seal and label placement for this object category."

    emb_normal = model.encode([normal_like_text], convert_to_numpy=True, normalize_embeddings=True)[0]
    emb_abnormal = model.encode([abnormal_like_text], convert_to_numpy=True, normalize_embeddings=True)[0]
    emb_domain = model.encode([domain_like_text], convert_to_numpy=True, normalize_embeddings=True)[0]

    # Eq.9 branch 1 & 2: normal-style text scores higher graded as "normal"
    # than graded as "abnormal", and vice versa for abnormal-style text.
    r_normal_as_normal = reference_bank_reward_eq9(emb_normal, "normal", store)
    r_normal_as_abnormal = reference_bank_reward_eq9(emb_normal, "abnormal", store)
    assert r_normal_as_normal > r_normal_as_abnormal, (r_normal_as_normal, r_normal_as_abnormal)

    r_abnormal_as_abnormal = reference_bank_reward_eq9(emb_abnormal, "abnormal", store)
    r_abnormal_as_normal = reference_bank_reward_eq9(emb_abnormal, "normal", store)
    assert r_abnormal_as_abnormal > r_abnormal_as_normal, (r_abnormal_as_abnormal, r_abnormal_as_normal)

    # Eq.9 branch 3 (domain_only): domain-style text should score positively
    # (s_D > max(s_N, s_A)) when graded as domain_only, and a non-domain text
    # (e.g. abnormal-style) should score lower on this branch since it's
    # closer to B_A than to B_D.
    r_domain_as_domain = reference_bank_reward_eq9(emb_domain, "domain_only", store)
    r_abnormal_as_domain = reference_bank_reward_eq9(emb_abnormal, "domain_only", store)
    assert r_domain_as_domain > r_abnormal_as_domain, (r_domain_as_domain, r_abnormal_as_domain)
    assert r_domain_as_domain > 0, r_domain_as_domain  # s_D should exceed max(s_N,s_A) for domain-style text

    print(f"  (normal-text: as normal={r_normal_as_normal:.3f} vs as abnormal={r_normal_as_abnormal:.3f})")
    print(f"  (abnormal-text: as abnormal={r_abnormal_as_abnormal:.3f} vs as normal={r_abnormal_as_normal:.3f})")
    print(f"  (domain-text: as domain_only={r_domain_as_domain:.3f}; abnormal-text as domain_only={r_abnormal_as_domain:.3f})")


if __name__ == "__main__":
    tests = [
        test_eddp_v2_correct_order_scores_full_order_bonus,
        test_eddp_v2_old_order_loses_order_bonus_only,
        test_eddp_v2_extract_answer_and_location_unaffected_by_order,
        test_eddp_v2_malformed_location_fails_open_to_none,
        test_reference_banks_build_and_score,
    ]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
