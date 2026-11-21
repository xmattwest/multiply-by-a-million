#!/usr/bin/env python3
"""render_prompt_s2.py: the Study 2 prompt (prereg/study2/PREREGISTRATION.md).

Identical to scripts/render_prompt.py (Study 1, frozen) except the system
message: no public appraisal and no public ranges. Each side knows only its
own limit and that a deal is possible. The scale reaches the model only
through its own limit and the prices in the history.

Same signature as render_prompt.render(), so game_engine and replay_audit
can use either; V, f_range and c_range are accepted and ignored here.
Reuses Study 1's fill/money/fragment code unchanged.
"""
import os

import render_prompt as base

PROMPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "prereg", "study2", "prompts")


def render(role, V, limit, history, msg_no, f_range, c_range, max_msgs, max_chars, templates=None):
    templates = templates or base.load_templates(PROMPTS)
    system_t, user_t, fr = templates
    # Render with Study 1's code on a system template that has its public
    # placeholders, then swap in the Study 2 system message. Everything in
    # the user message (role, limit, history, pending offer) is unchanged.
    msgs = base.render(role, V, limit, history, msg_no, f_range, c_range, max_msgs, max_chars,
                       templates=(base.load_templates(base.PROMPTS)[0], user_t, fr))
    msgs[0] = {"role": "system", "content": base.fill(system_t, {
        "MAX_MSGS": str(max_msgs), "MAX_CHARS": str(max_chars)})}
    return msgs
