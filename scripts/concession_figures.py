#!/usr/bin/env python3
"""Concession-path figure (WP1), exploratory.

CC1_concession_decomposition.svg
    For self-play cells, splits the x1 -> x10^6 change in the seller's share
    into two channels:
      * opening channel  = where the midpoint of the two FIRST offers moves
                           (zone units); this is the opener-modesty (HA/HG)
                           part.
      * concession channel = the residual the bargaining adds after the
                           openings, i.e. (deal - opening-midpoint) moving;
                           measured by the change in midpoint_dev.
    Their sum reconstructs the total share change. The point of the figure is
    the P/P cell: when both openings are pinned by the range anchor, the
    opening channel is ~0 but the split still falls, so the shift lives in
    the concessions (E7 / HD).

Reads the combined JSON written by analyze_concessions.py (--json). Stdlib
only; Okabe-Ito colours, never colour-only (hatching + labels).

Usage:
  python3 scripts/analyze_concessions.py <runs...> --json derived/explore/concessions_wp1.json
  python3 scripts/concession_figures.py derived/explore/concessions_wp1.json OUTDIR
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_figures import BLUE, ORANGE, GREEN, GREY, INK, Svg  # noqa: E402

VERM = "#D55E00"

# cells to show, in order, with a short human label. Keyed by substring match
# against the analyzer's cell string, within a chosen run.
GEMMA_INFO = "X-info-gemma-4-31b-it"
QWEN_INFO = "X-info-qwen3.8-27b-fp8"

CELL_ROWS = [
    # (run substring, cell substring, label)
    (GEMMA_INFO, "info=b:N/s:N", "Gemma  N/N  (no info)"),
    (GEMMA_INFO, "info=b:A/s:A", "Gemma  A/A  (both appraised)"),
    (GEMMA_INFO, "info=b:P/s:P", "Gemma  P/P  (both know the range)"),
    (QWEN_INFO,  "info=b:N/s:N", "Qwen   N/N  (no info)"),
    (QWEN_INFO,  "info=b:A/s:A", "Qwen   A/A  (both appraised)"),
    (QWEN_INFO,  "info=b:P/s:P", "Qwen   P/P  (both know the range)"),
]


def find_cell(data, run_sub, cell_sub):
    for r in data:
        if run_sub in r["run_dir"]:
            for c in r["cells"]:
                if cell_sub in c["cell"] and c["n_paired"] > 0:
                    return c
    return None


def opening_mid_delta(c):
    """Change in the opening-midpoint's zone position = opening channel.

    opening_mid_share__hi_mean - opening_mid_share__lo_mean is not stored, so
    reconstruct it from the ambitions: the opening midpoint share moves by
    0.5*(seller_open_ambition_delta - buyer_open_ambition_delta) in share
    units (seller ambition raises the midpoint, buyer ambition lowers it).
    Both ambitions are 'toward the far end'; a drop in seller ambition lowers
    the seller share, a drop in buyer ambition raises it."""
    s = c.get("seller_open_ambition__delta_mean")
    b = c.get("buyer_open_ambition__delta_mean")
    if s is None or b is None:
        return None
    return 0.5 * (s - b)


def fig_decomposition(data, out):
    rows = []
    for run_sub, cell_sub, label in CELL_ROWS:
        c = find_cell(data, run_sub, cell_sub)
        if not c:
            continue
        total = c.get("seller_share__delta_mean")
        opening = opening_mid_delta(c)
        conc = c.get("midpoint_dev__delta_mean")
        rows.append((label, total, opening, conc, c["n_paired"]))

    W, H = 820, 150 + len(rows) * 58 + 90
    L, R = 250, W - 40
    top = 150
    # x range symmetric around 0, covering all values
    allv = []
    for _, t, o, cc, _ in rows:
        allv += [v for v in (t, o, cc) if v is not None]
    span = max(0.25, max(abs(v) for v in allv) + 0.03)
    span = round(span * 20 + 0.4999) / 20
    lo, hi = -span, span
    X = lambda v: L + (v - lo) / (hi - lo) * (R - L)  # noqa: E731

    s = Svg(W, H,
            "Where the big-number split shift lives: opening vs concessions",
            "For each self-play cell, the x1->x10^6 change in the seller's "
            "share (square) is split into an opening channel (how far the "
            "midpoint of the two first offers moves) and a concession channel "
            "(how far the deal then drifts off that midpoint). In the P/P "
            "cell the openings are pinned by the range anchor, so the opening "
            "channel is near zero yet the split still falls: that shift lives "
            "in the concessions. Exploratory, not pre-registered.")
    s.text(20, 32, "Where does the \u00d710\u2076 split shift live?", 18, weight="bold")
    s.text(20, 54, "Change in the seller's share of the zone from \u00d71 to \u00d710\u2076, "
           "split into the opening and the concessions", 12.5, color="#444")
    s.text(20, 72, "Self-play, Study-2 zone, no reference price unless stated. "
           "60 fixture\u00d7opener pairs per cell.", 12, color="#666")

    # legend
    lx = L
    s.add(f"<rect x='{lx}' y='96' width='12' height='12' fill='white' stroke='{INK}' stroke-width='2'/>")
    s.text(lx + 18, 106, "total share change", 11.5)
    s.add(f"<circle cx='{lx + 170}' cy='102' r='6' fill='{BLUE}'/>")
    s.text(lx + 180, 106, "opening channel (midpoint of first offers)", 11.5)
    s.add(f"<path d='M{lx + 170},{132 - 6} l6,6 l-6,6 l-6,-6 Z' fill='{ORANGE}'/>")
    s.text(lx + 180, 132 - 2 + 6, "concession channel (drift off that midpoint)", 11.5)
    s.text(lx, 132 + 6, "", 11.5)

    # zero line + equivalence hint
    s.line(X(0), top - 10, X(0), top + len(rows) * 58 - 10, GREY, 1, "4 3")

    for i, (label, total, opening, conc, n) in enumerate(rows):
        y = top + i * 58 + 14
        s.line(L, y + 16, R, y + 16, "#f3f3f3", 1)
        s.text(L - 12, y + 4, label, 12, "end", "#333", weight="bold")
        s.text(L - 12, y + 20, f"n={n}", 10.5, "end", "#888")
        # opening channel bar from 0
        if opening is not None:
            s.add(f"<rect x='{min(X(0), X(opening)):.1f}' y='{y - 6}' "
                  f"width='{abs(X(opening) - X(0)):.1f}' height='9' "
                  f"fill='{BLUE}' fill-opacity='0.75'/>")
        # concession channel bar, stacked from the opening end
        if conc is not None and opening is not None:
            x0 = X(opening)
            x1 = X(opening + conc)
            s.add(f"<rect x='{min(x0, x1):.1f}' y='{y + 5}' "
                  f"width='{abs(x1 - x0):.1f}' height='9' "
                  f"fill='{ORANGE}' fill-opacity='0.8'/>")
        # total marker (square)
        if total is not None:
            s.add(f"<rect x='{X(total) - 6:.1f}' y='{y + 1}' width='12' height='12' "
                  f"fill='white' stroke='{INK}' stroke-width='2'/>")
            s.text(X(total) + (10 if total >= 0 else -10), y + 11,
                   f"{total:+.3f}", 11, "start" if total >= 0 else "end", INK)
        # markers for the two channels
        if opening is not None:
            s.add(f"<circle cx='{X(opening):.1f}' cy='{y - 1.5:.1f}' r='4' fill='{BLUE}'/>")
        if conc is not None and opening is not None:
            cx = X(opening + conc)
            s.add(f"<path d='M{cx:.1f},{y + 9.5 - 5:.1f} l5,5 l-5,5 l-5,-5 Z' fill='{ORANGE}'/>")

    # axis
    ya = top + len(rows) * 58 - 2
    s.line(L, ya, R, ya, INK)
    step = 0.05 if span <= 0.3 else 0.1
    t = -span
    k = -int(round(span / step))
    while k <= int(round(span / step)):
        tv = round(k * step, 3)
        s.line(X(tv), ya, X(tv), ya + 5, INK)
        s.text(X(tv), ya + 19, f"{tv:+.2f}" if tv else "0", 10.5, "middle")
        k += 1
    s.text(X(-span * 0.55), ya + 40, "\u25c0 seller does worse at \u00d710\u2076", 12, "middle", "#444")
    s.text(X(span * 0.55), ya + 40, "seller does better \u25b6", 12, "middle", "#444")
    s.text(20, H - 12, "Exploratory \u00b7 derived from games.jsonl via analyze_concessions.py \u00b7 "
           "opening + concession channels sum to the total share change by construction.",
           10.5, color="#777")
    s.save(os.path.join(out, "CC1_concession_decomposition.svg"))


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    data = json.load(open(argv[1]))
    out = argv[2]
    os.makedirs(out, exist_ok=True)
    fig_decomposition(data, out)
    print(f"wrote {os.path.join(out, 'CC1_concession_decomposition.svg')}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
