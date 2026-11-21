#!/usr/bin/env python3
"""render_prompt.py: build the exact messages one negotiator sees on one turn.

WHAT IT DOES
  Fills the pre-registered templates in prereg/prompts/ for one side on one
  turn: a system message (the rules, the same for both sides) and a user
  message (your role, your private limit, the public history, whose turn).
  Each call is built from scratch from the game's state; nothing carries
  over between calls.

  Money is always written the same way, as "<whole number> dollars", with
  no commas, so the only thing that changes between scales is the digits.
  tests/test_render_prompt.py checks that: delete every number, and the
  prompts at $100, $100,000 and $100,000,000 are identical.

  The other side's private limit is never an input: render() for the seller
  takes the floor F but not the ceiling C, and vice versa.

USAGE (for inspection)
  python3 scripts/render_prompt.py GAME_ID ROLE      e.g. F12-x1e5-S seller
  Prints the first-turn messages for that game and role as JSON.

Standard library only.
"""

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
PROMPTS = os.path.join(ROOT, "prereg", "prompts")
PLACEHOLDER = re.compile(r"\{\{([A-Z_]+)\}\}")


def money(x):
    if isinstance(x, bool) or not isinstance(x, int) or x < 1:
        raise ValueError("money must be a positive whole number")
    return f"{x} dollars"


def fill(template, values):
    """Replace every {{NAME}} in one pass. Unknown or unused names are errors,
    and inserted text is never scanned again (so {{...}} inside a model's
    message stays literal)."""
    used = set()

    def sub(m):
        name = m.group(1)
        if name not in values:
            raise KeyError(f"no value for placeholder {name}")
        used.add(name)
        return values[name]

    out = PLACEHOLDER.sub(sub, template)
    unused = set(values) - used
    if unused:
        raise KeyError(f"values given but not used: {sorted(unused)}")
    return out


def load_fragments(path):
    frags, name, buf = {}, None, []
    with open(path, encoding="utf-8") as fh:
        for line in fh.read().split("\n"):
            if line.startswith("#"):
                continue
            m = re.fullmatch(r"\[([a-z_]+)\]", line)
            if m:
                if name is not None:
                    frags[name] = "\n".join(buf).strip("\n")
                name, buf = m.group(1), []
            elif name is not None:
                buf.append(line)
    if name is not None:
        frags[name] = "\n".join(buf).strip("\n")
    return frags


def load_templates(prompts_dir=PROMPTS):
    with open(os.path.join(prompts_dir, "system.txt"), encoding="utf-8") as fh:
        system = fh.read().rstrip("\n")
    with open(os.path.join(prompts_dir, "user.txt"), encoding="utf-8") as fh:
        user = fh.read().rstrip("\n")
    return system, user, load_fragments(os.path.join(prompts_dir, "fragments.txt"))


def render(role, V, limit, history, msg_no, f_range, c_range, max_msgs, max_chars,
           templates=None):
    """Messages for one turn.

    role      "seller" or "buyer"
    V         public reference value (dollars)
    limit     this side's private value: F for the seller, C for the buyer
    history   earlier moves, oldest first: list of (role, price, text); every
              earlier move is an OFFER (ACCEPT and WALKAWAY end the game)
    msg_no    number of the message being written now (1-based)
    f_range, c_range   (lo_pct, hi_pct) public ranges, percent of V
    """
    if role not in ("seller", "buyer"):
        raise ValueError("role must be seller or buyer")
    if not 1 <= msg_no <= max_msgs or len(history) != msg_no - 1:
        raise ValueError("msg_no must be len(history) + 1 and within the cap")
    other = "buyer" if role == "seller" else "seller"
    for i, (who, price, text) in enumerate(history):
        # Turns alternate, so the side that moves now made the messages
        # that are an even number of steps back.
        expect = role if (len(history) - i) % 2 == 0 else other
        if who != expect:
            raise ValueError("history does not alternate between the two sides")
        if not isinstance(text, str):
            raise ValueError("history message must be text")
        money(price)
    if history and history[-1][0] != other:
        raise ValueError("the last move must be the other side's")

    system_t, user_t, fr = templates or load_templates()
    pct = lambda p: money(p * V // 100)
    for p in (*f_range, *c_range):
        if (p * V) % 100:
            raise ValueError("range bounds must be whole dollars")

    system = fill(system_t, {
        "V": money(V),
        "F_LO": pct(f_range[0]), "F_HI": pct(f_range[1]),
        "C_LO": pct(c_range[0]), "C_HI": pct(c_range[1]),
        "MAX_MSGS": str(max_msgs), "MAX_CHARS": str(max_chars),
    })

    if history:
        lines = [fr["history_head"]]
        for i, (who, price, text) in enumerate(history, start=1):
            lines.append(fill(fr["history_line"], {
                "N": str(i),
                "WHO": "you" if who == role else f"the {who}",
                "PRICE": money(price),
                # JSON-quoted: newlines and quotes in the other model's text
                # can't break the layout or pose as a new line of the prompt.
                "TEXT": json.dumps(text, ensure_ascii=False),
            }))
        hist = "\n".join(lines)
        pending = fill(fr["pending_offer"], {"OTHER": other, "PRICE": money(history[-1][1])})
    else:
        hist = fr["history_empty"]
        pending = fill(fr["pending_none"], {"OTHER": other})

    user = fill(user_t, {
        "ROLE": role,
        "LIMIT_NAME": "floor" if role == "seller" else "ceiling",
        "LIMIT": money(limit),
        "OTHER": other,
        "GOAL": fr["goal_seller"] if role == "seller" else fr["goal_buyer"],
        "HISTORY": hist,
        "MSG_NO": str(msg_no),
        "MAX_MSGS": str(max_msgs),
        "PENDING": pending,
    })
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def main(argv):
    if len(argv) != 2 or argv[1] not in ("seller", "buyer"):
        print(__doc__.split("USAGE (for inspection)")[1].split("Standard")[0].strip(), file=sys.stderr)
        return 2
    with open(os.path.join(ROOT, "config", "fixtures.json"), encoding="utf-8") as fh:
        fx = json.load(fh)
    with open(os.path.join(ROOT, "prereg", "settings.json"), encoding="utf-8") as fh:
        st = json.load(fh)
    games = {g["game_id"]: g for g in fx["games"]}
    if argv[0] not in games:
        print(f"unknown game ID (examples: {', '.join(list(games)[:3])})", file=sys.stderr)
        return 2
    g = games[argv[0]]
    role = argv[1]
    if role != g["opener"]:
        print(f"note: in {g['game_id']} the {g['opener']} opens; showing the {role}'s view "
              "of message 1 anyway", file=sys.stderr)
    msgs = render(role, g["V"], g["F"] if role == "seller" else g["C"], [], 1,
                  fx["f_pct_range"], fx["c_pct_range"],
                  st["max_messages"]["value"], st["max_message_chars"]["value"])
    print(json.dumps(msgs, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
