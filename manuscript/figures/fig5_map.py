#!/usr/bin/env python3
"""Figure 5 (exploratory): the map of what changes the x10^6 effect, as one forest plot.

Every row is one exploratory cell (fresh fixtures, n = 30 unless marked), every point the mean
x10^6 - x1 change with its 95% bootstrap CI as stored in derived/explore/X-*/summary.json.
  (a) the model as seller vs the scale-proof bot: change in the MODEL'S OWN share
  (b) the model as buyer vs the bot: change in the MODEL'S OWN share
  (c) self-play: change in the seller's share (the Study 2 outcome)
Sign conventions differ between analyzers, so they are normalised here:
  analyze_geom_probe / analyze_game_probe store the SELLER's share for "BOT model=buyer"
  (a rise = the model-buyer pays more) -> negated; analyze_cell_probe (prompt, info) already
  stores the model's own share. The redesigned game (X17) uses its slipped-games-excluded
  contrasts, as reported in STORY.

Stdlib only; writes fig5_map.svg next to this script.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from figstyle import BAND, GREY, LIGHT, MODEL, ONYX, ROOT, line, svg_doc, text, write  # noqa: E402

EX = os.path.join(ROOT, "derived", "explore")
RUN = {
    ("geom", "gemma-4-31b-it"): "X-geom-gemma-4-31b-it-20261004T154019Z",
    ("geom", "qwen3.8-27b-fp8"): "X-geom-qwen3.8-27b-fp8-20261004T181743Z",
    ("prompt", "gemma-4-31b-it"): "X-prompt-gemma-4-31b-it-20261004T181723Z",
    ("prompt", "qwen3.8-27b-fp8"): "X-prompt-qwen3.8-27b-fp8-20261004T181752Z",
    ("info", "gemma-4-31b-it"): "X-info-gemma-4-31b-it-20261004T190958Z",
    ("info", "qwen3.8-27b-fp8"): "X-info-qwen3.8-27b-fp8-20261004T191018Z",
    ("newgame", "gemma-4-31b-it"): "X-newgame-gemma-4-31b-it-20261004T142428Z",
    ("cross", None): "X-cross-gemma-4-31b-it-vs-qwen3.8-27b-fp8-20261004T181800Z",
}
MODELS = ("gemma-4-31b-it", "qwen3.8-27b-fp8")
_cache = {}


def contrast(probe, model, key, flip=False):
    run = RUN.get((probe, model))
    if run is None:
        return None
    if run not in _cache:
        _cache[run] = json.load(open(os.path.join(EX, run, "summary.json")))
    d = _cache[run]
    c = (d["contrasts_excluding_slipped_games"] if probe == "newgame" else d["contrasts"]).get(key)
    if c is None:
        raise KeyError((run, key))
    m, (a, b) = c["mean"], c["ci95"]
    return (-m, (-b, -a), c["n"]) if flip else (m, (a, b), c["n"])


def geom(zone, nota):
    k = f"{zone} {nota} "
    return lambda m: (contrast("geom", m, k + "BOT model=seller share"),
                      contrast("geom", m, k + "BOT model=buyer share", flip=True),
                      contrast("geom", m, k + "SPN d"))


def cell(probe, s):
    k = "round raw | "
    return lambda m: (contrast(probe, m, f"{k}BOT {s} model=seller | share"),
                      contrast(probe, m, f"{k}BOT {s} model=buyer | share"),
                      contrast(probe, m, f"{k}SP {s}/{s} | share"))


def newgame(m):
    if m != "gemma-4-31b-it":
        return (None, None, None)
    return (contrast("newgame", m, "BOT model=seller share"), contrast("newgame", m, "BOT model=buyer share", flip=True),
            contrast("newgame", m, "SPN d"))


GROUPS = [
    ("Zone shape and notation (X18, X22)", [
        ("Study 2 zone, plain digits", geom("round", "raw")),
        ("Study 2 zone, business notation", geom("round", "biz")),
        ("narrow zone", geom("narrow", "raw")),
        ("wide zone", geom("wide", "raw")),
    ]),
    ("Instructions to both sides (X20, X21)", [
        ("none", cell("prompt", "N")),
        ("\u201cnegotiate hard\u201d", cell("prompt", "A")),
        ("\u201ca deal matters more\u201d", cell("prompt", "C")),
        ("\u201cthe size of the numbers doesn't matter\u201d", cell("prompt", "D")),
    ]),
    ("Information (X23, X24)", [
        ("none", cell("info", "N")),
        ("a private appraisal", cell("info", "A")),
        ("the other side's range", cell("info", "P")),
    ]),
    ("Redesigned game (X17; Gemma only)", [
        ("wide, off-centre zones", newgame),
    ]),
]
CROSS = [("Gemma sells to Gemma", "G/G", "gemma-4-31b-it"), ("Qwen sells to Qwen", "Q/Q", "qwen3.8-27b-fp8"),
         ("Gemma sells to Qwen", "G/Q", "gemma-4-31b-it"), ("Qwen sells to Gemma", "Q/G", "qwen3.8-27b-fp8")]

LO, HI = -0.35, 0.20
COLS = [("a", "model sells to the bot", "model's own share"),
        ("b", "model buys from the bot", "model's own share"),
        ("c", "self-play", "seller's share")]


def main():
    labw, pw, pg = 300, 215, 15
    x_of = lambda col, v: labw + col * (pw + pg) + (v - LO) / (HI - LO) * pw
    out = []
    top = 96
    for i, (tag, t1, t2) in enumerate(COLS):
        cx = labw + i * (pw + pg) + pw / 2
        out.append(text(labw + i * (pw + pg), 30, tag, 22, weight="bold"))
        out.append(text(labw + i * (pw + pg) + 24, 30, t1, 16, ONYX, "bold"))
        out.append(text(cx, 52, f"change in the {t2}", 15, GREY, anchor="middle"))
    # Legend.
    for j, m in enumerate(MODELS):
        out.append(f"<circle cx='{30 + j * 150}' cy='{72}' r='5.5' fill='{MODEL[m]['col']}'/>")
        out.append(text(42 + j * 150, 77, MODEL[m]["label"], 15, MODEL[m]["col"], "bold"))
    y = top
    rows_drawn = 0
    allrows = [(g, r) for g, rs in GROUPS for r in rs]
    for gname, rows in GROUPS:
        y += 8
        out.append(text(20, y + 14, gname, 15, ONYX, "bold"))
        y += 24
        for label, fn in rows:
            cy = y + 14
            out.append(text(labw - 12, cy + 5, label, 15, ONYX, anchor="end"))
            for col in range(3):
                for j, m in enumerate(MODELS):
                    v = fn(m)[col]
                    if v is None:
                        continue
                    mean, (a, b), n = v
                    yy = cy - 5 + 10 * j
                    c = MODEL[m]["col"]
                    out.append(line(x_of(col, max(a, LO)), yy, x_of(col, min(b, HI)), yy, c, 2.2))
                    out.append(f"<circle cx='{x_of(col, mean):.1f}' cy='{yy:.1f}' r='4.2' fill='{c}'/>")
                    if n != 30:
                        out.append(text(x_of(col, min(b, HI)) + 6, yy + 5, f"n={n}", 15, GREY))
            y += 30
            rows_drawn += 1
    # Cross-play rows: self-play column only, coloured by the seller's model.
    y += 8
    out.append(text(20, y + 14, "Mixed pairs, Study 2 zone (X19)", 15, ONYX, "bold"))
    y += 24
    for label, key, seller in CROSS:
        cy = y + 14
        out.append(text(labw - 12, cy + 5, label, 15, ONYX, anchor="end"))
        mean, (a, b), n = contrast("cross", None, f"round raw | {key} | share")
        c = MODEL[seller]["col"]
        out.append(line(x_of(2, a), cy, x_of(2, b), cy, c, 2.2))
        out.append(f"<rect x='{x_of(2, mean) - 4.5:.1f}' y='{cy - 4.5:.1f}' width='9' height='9' fill='{c}'/>")
        y += 30
    bot = y + 4
    # Panels: zero lines, Study 2 reference, axes.
    for col in range(3):
        x0 = labw + col * (pw + pg)
        out.insert(0, f"<rect x='{x0}' y='{top}' width='{pw}' height='{bot - top}' fill='#FAFAF7'/>")
        out.append(line(x_of(col, 0), top, x_of(col, 0), bot, GREY, 1.2, dash="4 3"))
        out.append(line(x0, bot, x0 + pw, bot, ONYX, 1.2))
        for v in (-0.3, -0.2, -0.1, 0.0, 0.1, 0.2):
            out.append(line(x_of(col, v), bot, x_of(col, v), bot + 6, ONYX, 1.2))
            if v in (-0.2, 0.0, 0.1):
                out.append(text(x_of(col, v), bot + 24, f"{v:+.1f}" if v else "0", 15, ONYX, anchor="middle"))
    out.append(text(labw, bot + 48, "\u2190 worse for the model at \u00d710\u2076 (a, b) / for the seller (c)", 15, GREY))
    notes = ["Points: mean \u00d710\u2076 \u2212 \u00d71 change per fixture, bars: 95% CI; fresh fixtures, n = 30 unless marked. "
             "All exploratory.",
             "The bot's moves are ratios of its own limit, so at \u00d710\u2076 they are its \u00d71 moves with six zeros appended.",
             "Squares in (c): mixed pairs, coloured by the seller's model.",
             "X17 (redesigned game) excludes the games with a comma slip in a price (n = 28)."]
    for k, n in enumerate(notes):
        out.append(text(20, bot + 76 + 20 * k, n, 15, GREY))
    H = bot + 76 + 20 * len(notes)
    doc = svg_doc(1000, int(H), "Exploratory map: what changes the effect of multiplying every number by a million",
                  "Forest plot. Rows are exploratory cells grouped by question (zone shape and notation, instructions, "
                  "information, a redesigned game, mixed pairs); columns are the model as seller and as buyer against a "
                  "scale-proof bot (change in its own share) and self-play (change in the seller's share); Gemma 4 31B "
                  "and Qwen points with 95% intervals.", out)
    path = write(os.path.join(HERE, "fig5_map.svg"), doc)
    # Print the plotted values so the caption and text can be checked against STORY rows.
    for gname, rows in GROUPS:
        for label, fn in rows:
            for m in MODELS:
                vals = fn(m)
                print(f"{gname[:22]:22} | {label[:34]:34} | {m[:6]} | " +
                      " | ".join("   --   " if v is None else f"{v[0]:+.3f}" for v in vals))
    for label, key, _ in CROSS:
        print("cross", label, f"{contrast('cross', None, f'round raw | {key} | share')[0]:+.3f}")
    print(path)


if __name__ == "__main__":
    sys.exit(main())
