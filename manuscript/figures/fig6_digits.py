#!/usr/bin/env python3
"""Figure 6 (exploratory): digits or stakes? Gemma 4 31B first moves under 13 wordings of the limit.

Rows are wordings grouped by the stake they describe ($81, $81 thousand, $81 million); columns:
(a) the seller's opening markup over its floor, (b) the buyer's opening discount below its
ceiling (both in % of the limit, mean over fixtures with a 95% fixture-bootstrap CI), and
(c) the share of the seller's first offers that are round tens. Values are read from
derived/explore/digits_combined/gemma-4-31b-it/combined.json (X15, X16; fresh fixtures Y01-Y30).

Stdlib only; writes fig6_digits.svg next to this script.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from figstyle import GREY, MODEL, ONYX, ROOT, line, svg_doc, text, write  # noqa: E402

SRC = os.path.join(ROOT, "derived", "explore", "digits_combined", "gemma-4-31b-it", "combined.json")
GROUPS = [
    ("Stake $81", ["D2", "S2", "X2", "K2"]),
    ("Stake $81 thousand", ["D5", "U5"]),
    ("Stake $81 million", ["D8", "SN", "C8", "S8", "W8", "SM", "U8"]),
]
ANCHORS = ("D2", "D8")  # the confirmatory Study 2 wordings


def main():
    d = json.load(open(SRC))
    by = d["by_condition"]
    col = MODEL[d["model"]]["col"]
    labw = 330
    cols = [("a", "seller opens: markup over floor", "seller", "mean_gap_pct", 10, 50, (10, 20, 30, 40, 50), "%"),
            ("b", "buyer opens: discount below ceiling", "buyer", "mean_gap_pct", 30, 50, (30, 40, 50), "%"),
            ("c", "round-ten openings: share of seller first offers", "seller", "round_ten_share", 0, 1, (0, 0.5, 1), "share")]
    widths = (220, 160, 160)
    gapx = 40
    xs = []
    x = labw
    for w in widths:
        xs.append(x)
        x += w + gapx
    out = []
    for i, (tag, title, *_rest) in enumerate(cols):
        out.append(text(xs[i], 30, tag, 22, weight="bold"))
        words = title.split(": ")
        out.append(text(xs[i] + 22, 30, words[0] + (":" if len(words) > 1 else ""), 15, ONYX, "bold"))
        if len(words) > 1:
            out.append(text(xs[i] + 22, 50, words[1], 15, ONYX, "bold"))
    top = 70
    y = top
    for gname, conds in GROUPS:
        y += 6
        out.append(text(20, y + 16, gname, 15, ONYX, "bold"))
        y += 26
        for c in conds:
            v = by[c]
            cy = y + 12
            anchor = c in ANCHORS
            out.append(text(labw - 14, cy + 5, v["label"] + ("  (Study 2)" if anchor else ""), 15, ONYX,
                            "bold" if anchor else "normal", "end"))
            for i, (_, _, role, key, lo, hi, _, unit) in enumerate(cols):
                X = lambda val: xs[i] + (val - lo) / (hi - lo) * widths[i]
                r = v[role]
                m = r[key]
                if key == "round_ten_share":
                    out.append(f"<rect x='{xs[i]}' y='{cy - 6}' width='{X(m) - xs[i]:.1f}' height='12' fill='{col}' "
                               f"fill-opacity='{0.9 if anchor else 0.55}'/>")
                    out.append(text(X(m) + 6 if m < 0.8 else X(m) - 6, cy + 5, f"{m * 100:.0f}%", 15,
                                    ONYX, anchor="start" if m < 0.8 else "end"))
                else:
                    a, b = r["ci95"]
                    out.append(line(X(a), cy, X(b), cy, col, 2.4))
                    out.append(f"<circle cx='{X(m):.1f}' cy='{cy:.1f}' r='5' fill='{col if anchor else 'white'}' "
                               f"stroke='{col}' stroke-width='2'/>")
            y += 28
    bot = y + 6
    for i, (_, _, _, key, lo, hi, ticks, unit) in enumerate(cols):
        X = lambda val: xs[i] + (val - lo) / (hi - lo) * widths[i]
        out.insert(0, f"<rect x='{xs[i]}' y='{top}' width='{widths[i]}' height='{bot - top}' fill='#FAFAF7'/>")
        out.append(line(xs[i], bot, xs[i] + widths[i], bot, ONYX, 1.2))
        for t in ticks:
            out.append(line(X(t), top, X(t), bot, "#E6E6E6", 1))
            out.append(line(X(t), bot, X(t), bot + 6, ONYX, 1.2))
            out.append(text(X(t), bot + 24, f"{t:g}%" if unit == "%" else f"{t * 100:.0f}%", 15, ONYX, anchor="middle"))
    n_off = sorted({by[c][r]["n_offers"] for c in by for r in ("seller", "buyer")})
    notes = ["Points: mean over 30 fresh fixtures, bars: 95% fixture-bootstrap CI; filled = the two Study 2 wordings.",
             "\u201creply in \u2026\u201d: the model is asked to write its price in that unit. "
             f"First offers per cell: {', '.join(map(str, n_off))}. Exploratory (X15, X16)."]
    for k, n in enumerate(notes):
        out.append(text(20, bot + 54 + 20 * k, n, 15, GREY))
    H = bot + 54 + 20 * len(notes)
    doc = svg_doc(1000, int(H), "Digits or stakes: Gemma 4 31B's first offers under 13 wordings of the limit",
                  "Dot plot of the seller's opening markup and the buyer's opening discount, and a bar chart of the "
                  "share of round-ten seller openings, for 13 wordings grouped by the stake they describe.", out)
    path = write(os.path.join(HERE, "fig6_digits.svg"), doc)
    for g, conds in GROUPS:
        for c in conds:
            print(c, by[c]["label"], round(by[c]["seller"]["mean_gap_pct"], 1), round(by[c]["buyer"]["mean_gap_pct"], 1),
                  round(by[c]["seller"]["round_ten_share"], 3))
    print(path)


if __name__ == "__main__":
    sys.exit(main())
