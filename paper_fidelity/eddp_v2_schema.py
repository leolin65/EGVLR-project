# EDDP v2: paper-matching tag order (think -> location -> answer), vs the
# current codebase's think -> answer -> location.
#
# This file provides (1) the new target-format template strings for
# regenerating Stage1/2/3 data, and (2) an order-aware format_reward variant.
# It does NOT regenerate any data — data_pipeline/*/convert_*.py's
# ASSISTANT_TEMPLATE-equivalent constants would need to actually be edited
# and the build scripts rerun against the same raw sources, which is a
# separate, larger step (needs redoing Stage1/2/3 data generation + a fresh
# retrain, so the model is actually shown the new order during SFT/RL rather
# than just being graded on it) — see README.md.
#
# STATUS: written, unit-tested (CPU-only, no model needed — this is pure
# string/regex logic). Not wired into reward_funcs.py or any data builder.

import re

# The paper's order. Compare against reward_funcs.py's format_reward(), which
# enforces think -> answer -> location instead.
ANSWER_TEMPLATE_V2 = (
    "<think>\n<evidence>{evidence}</evidence>\n<logic>{logic}</logic>\n</think>\n"
    "<location>{location}</location><answer>{answer}</answer>"
)

QUESTION_INSTRUCTION_V2 = (
    "Answer using exactly:\n"
    "<think>\n<evidence>...</evidence>\n<logic>...</logic>\n</think>\n"
    "<location>[[x1,y1,x2,y2],...]</location><answer>A/B/C/D</answer>"
)


def format_reward_eddp_v2(completions, fmt_weight=None, **kwargs):
    """Same structural checks as reward_funcs.py's format_reward(), except
    the order check requires think -> location -> answer instead of
    think -> answer -> location. Duplicated (not imported from
    reward_funcs.py) so this module has no dependency on the live training
    env for CPU-only testing — see README.md for how to actually swap it in.
    """
    if fmt_weight is None:
        fmt_weight = 0.15  # matches reward_funcs.py's FMT_WEIGHT default

    rewards = []
    for completion in completions:
        completion = _to_text(completion)
        reward = 0.0

        think_open, think_close = _count_tag(completion, "think")
        has_one_think = think_open == 1 and think_close == 1
        if has_one_think:
            reward += fmt_weight * 0.15

        think_content = _find_single_tag(completion, "think")
        has_evidence = has_logic = False
        if think_content is not None:
            ev_open, ev_close = _count_tag(think_content, "evidence")
            logic_open, logic_close = _count_tag(think_content, "logic")
            has_evidence = ev_open == 1 and ev_close == 1
            has_logic = logic_open == 1 and logic_close == 1
        if has_evidence:
            reward += fmt_weight * 0.15
        if has_logic:
            reward += fmt_weight * 0.15

        pred_answer = extract_answer_v2(completion)
        if pred_answer is not None:
            reward += fmt_weight * 0.15

        pred_location = extract_location_v2(completion)
        if pred_location is not None:
            reward += fmt_weight * 0.20

        think_span = _find_tag_span(completion, "think")
        location_span = _find_tag_span(completion, "location")
        answer_span = _find_tag_span(completion, "answer")

        # The only line that actually differs from reward_funcs.py's
        # format_reward(): location must come before answer, not after.
        order_ok = (
            think_span is not None
            and location_span is not None
            and answer_span is not None
            and think_span[1] <= location_span[0]
            and location_span[1] <= answer_span[0]
        )
        if order_ok:
            reward += fmt_weight * 0.20

        rewards.append(float(reward))

    return rewards


def extract_answer_v2(text: str):
    text = _to_text(text)
    matches = re.findall(r"<answer>\s*([A-D])\s*</answer>", text, flags=re.IGNORECASE)
    if len(matches) != 1:
        return None
    return matches[0].upper()


def extract_location_v2(text: str):
    """Identical parsing logic to reward_funcs.py's extract_location() — tag
    order doesn't affect how a <location> tag's own content is parsed, only
    the order.() check above cares about it."""
    import ast

    text = _to_text(text)
    matches = re.findall(r"<location>\s*(\[.*?\])\s*</location>", text, flags=re.DOTALL | re.IGNORECASE)
    if len(matches) != 1:
        return None
    try:
        boxes = ast.literal_eval(matches[0])
    except (ValueError, SyntaxError):
        return None
    if not isinstance(boxes, list):
        return None
    result = []
    for box in boxes:
        if isinstance(box, tuple):
            box = list(box)
        if not isinstance(box, list) or len(box) != 4:
            return None
        if not all(isinstance(v, (int, float)) for v in box):
            return None
        x1, y1, x2, y2 = [float(v) for v in box]
        if x2 < x1 or y2 < y1:
            return None
        result.append([x1, y1, x2, y2])
    return result


# --- small local copies of reward_funcs.py's tag helpers, so this module has
# no import-time dependency on the full training env ---

def _to_text(x):
    if isinstance(x, str):
        return x
    if isinstance(x, list) and x and isinstance(x[0], dict) and "content" in x[0]:
        return x[0]["content"]
    return str(x)


def _find_single_tag(text, tag):
    text = _to_text(text)
    pattern = re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", flags=re.DOTALL | re.IGNORECASE)
    matches = pattern.findall(text)
    if len(matches) != 1:
        return None
    return matches[0].strip()


def _find_tag_span(text, tag):
    text = _to_text(text)
    pattern = re.compile(rf"<{tag}>\s*.*?\s*</{tag}>", flags=re.DOTALL | re.IGNORECASE)
    match = pattern.search(text)
    return None if match is None else match.span()


def _count_tag(text, tag):
    text = _to_text(text)
    open_tags = re.findall(rf"<{tag}>", text, flags=re.IGNORECASE)
    close_tags = re.findall(rf"</{tag}>", text, flags=re.IGNORECASE)
    return len(open_tags), len(close_tags)
