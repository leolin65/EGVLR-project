# Smoke test for the 3 fixes made 2026-07-23 (Rbox -1 penalty, R_cpl LocValid,
# Stage1 <think>), run against the REAL reward_funcs.py in the actual training
# env (numpy/scipy/sentence-transformers available), not the pure-python mocks
# used earlier for logic verification. Meant to catch import/runtime bugs
# before committing to the long batched Stage1(v3)->Stage2->Stage3 retrain.
#
# Usage (from Qwen-VL-Series-Finetune root, so `src` resolves on PYTHONPATH):
#   PYTHONPATH=src python smoke_test_rewards.py

import sys

sys.path.insert(0, "src")

from train.reward_funcs import (
    format_reward,
    accuracy_reward,
    location_reward,
    think_similarity_reward,
    gated_think_answer_reward,
    is_location_valid,
    extract_location,
    bbox_score,
)


def think(evidence, logic):
    return f"<think>\n<evidence>{evidence}</evidence>\n<logic>{logic}</logic>\n</think>\n"


CASES = []

# Case 1: normal sample, model correctly says no defect, empty location, good think.
gt = think("The reference and test images match.", "No deviation observed.") + "<answer>B</answer><location>[]</location>"
CASES.append(("normal_correct", gt, gt))

# Case 2: normal sample, model HALLUCINATES a box -> should trigger Rbox=-1 and LocValid=False.
pred = think("The reference and test images match.", "No deviation observed.") + "<answer>B</answer><location>[[10,10,50,50]]</location>"
CASES.append(("normal_hallucinated_box", pred, gt))

# Case 3: abnormal sample, correct answer + well-matched box + good think -> should pass the gate.
gt_ab = think("A scratch is visible on the surface.", "This scratch indicates a defect.") + "<answer>A</answer><location>[[100,100,140,140]]</location>"
pred_ab = think("A scratch is visible on the surface.", "This scratch indicates a defect.") + "<answer>A</answer><location>[[102,101,139,141]]</location>"
CASES.append(("abnormal_correct_matched", pred_ab, gt_ab))

# Case 4: abnormal sample, correct answer but WRONG location -> LocValid should be False, gate should fail.
pred_wrong_loc = think("A scratch is visible on the surface.", "This scratch indicates a defect.") + "<answer>A</answer><location>[[5,5,15,15]]</location>"
CASES.append(("abnormal_correct_wrong_loc", pred_wrong_loc, gt_ab))

# Case 5: malformed completion (missing </think>, unparsable) -> format_reward should be low, location None.
malformed = "<answer>A</answer><location>[[not,valid]]</location>"
CASES.append(("malformed", malformed, gt_ab))


def main():
    completions = [c[1] for c in CASES]
    assistants = [c[2] for c in CASES]

    print("Running 5 reward functions on", len(CASES), "hand-built cases...\n")

    fmt = format_reward(completions)
    acc = accuracy_reward(completions, assistants)
    loc = location_reward(completions, assistants)
    sem = think_similarity_reward(completions, assistants)
    gated = gated_think_answer_reward(completions, assistants)

    ok = True
    for i, (name, pred, gt_text) in enumerate(CASES):
        print(f"--- {name} ---")
        print(f"  format={fmt[i]:.3f}  acc={acc[i]:.3f}  loc={loc[i]:.3f}  sem={sem[i]:.3f}  gated={gated[i]:.3f}")

        pred_boxes = extract_location(pred)
        gt_boxes = extract_location(gt_text)
        lv = is_location_valid(pred_boxes, gt_boxes)
        print(f"  is_location_valid={lv}")

    print()
    # Targeted assertions for the 2 reward fixes:
    assert loc[1] < 0, f"Case 2 (hallucinated box on normal) should get NEGATIVE loc reward, got {loc[1]}"
    print("PASS: normal-sample hallucinated box gets a negative location_reward (Rbox=-1 fix works).")

    assert gated[3] == 0.0, f"Case 4 (correct answer, wrong location) should be gated to 0, got {gated[3]}"
    print("PASS: coupling reward correctly zeroes out when LocValid=False despite correct answer (LocValid fix works).")

    assert gated[2] > 0.0, f"Case 3 (correct answer, matched location, good think) should pass the gate, got {gated[2]}"
    print("PASS: coupling reward fires when answer+location+think are all consistent.")

    print("\nAll smoke-test assertions passed.")


if __name__ == "__main__":
    main()
