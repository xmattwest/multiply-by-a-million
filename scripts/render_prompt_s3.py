#!/usr/bin/env python3
"""render_prompt_s3.py: Study 3's A1 prompt (prereg/study3/PREREGISTRATION.md §3.1).

Study 2's prompt (scripts/render_prompt_s2.py) with one line added to the system message:
"An independent appraisal values the lot at {{V}}. Both sides know this." No limit ranges are
shown. The user message and every fragment are Study 2's, so A1 and A0 differ by that one line.

Same signature as render_prompt.render(); f_range and c_range are accepted and ignored.
Reuses Study 1's fill/money/fragment code unchanged. Standard library only.
"""
import os

import render_prompt as base
import render_prompt_s2 as s2

PROMPTS_S3 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "prereg", "study3", "prompts")
SYSTEM_A1 = os.path.join(PROMPTS_S3, "system_a1.txt")


def load_system():
    with open(SYSTEM_A1, encoding="utf-8") as fh:
        return fh.read().rstrip("\n")          # as render_prompt.load_templates does


def render(role, V, limit, history, msg_no, f_range, c_range, max_msgs, max_chars, templates=None):
    msgs = s2.render(role, V, limit, history, msg_no, f_range, c_range, max_msgs, max_chars)
    msgs[0] = {"role": "system", "content": base.fill(load_system(), {
        "V": base.money(V), "MAX_MSGS": str(max_msgs), "MAX_CHARS": str(max_chars)})}
    return msgs
