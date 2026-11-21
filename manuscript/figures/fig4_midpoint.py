#!/usr/bin/env python3
"""Figure 4 (exploratory): why the split moves in one model and not the other.

(a) Every Study 2 deal's price against the midpoint of the two first offers, both in units of
    the lot's value V, by scale. Deals land on that midpoint (X14), so the split follows the
    openings.
(b) The x10^6 - x1 shift in seller share per game pair, split into an opening channel (how far
    the midpoint of the first offers moves) and a concession channel (how far the deal then
    drifts off it), as in WP1 (X25-X27). Values come from derived/explore/concessions_wp1.json
    via the committed helpers in scripts/concession_figures.py (imported, not modified).

Exploratory: post hoc. Stdlib only; writes fig4_midpoint.svg next to this script.
"""
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from figstyle import GREY, LIGHT, MODEL, ONYX, ROOT, line, svg_doc, text, write  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "scripts"))
import analyze_concessions as ac  # noqa: E402
import concession_figures as cf  # noqa: E402

RUNS = ["s2-gemma-4-31b-it-20261004T012932Z", "s2-qwen3.8-27b-fp8-20261003T223042Z"]
SCALES = (("x1e2", "$100"), ("x1e5", "$100K"), ("x1e8", "$100M"))
ROWS = [  # (run substring, cell substring, model, label)
    ("s2-gemma", "S2-selfplay", "gemma-4-31b-it", "Study 2 game"),
    ("X-info-gemma", "info=b:N/s:N", "gemma-4-31b-it", "info probe: no info (N/N)"),
    ("X-info-gemma", "info=b:A/s:A", "gemma-4-31b-it", "both told an appraisal (A/A)"),
    ("X-info-gemma", "info=b:P/s:P", "gemma-4-31b-it", "both told the other's range (P/P)"),
    ("s2-qwen", "S2-selfplay", "qwen3.8-27b-fp8", "Study 2 game"),
    ("X-info-qwen", "info=b:N/s:N", "qwen3.8-27b-fp8", "info probe: no info (N/N)"),
    ("X-info-qwen", "info=b:A/s:A", "qwen3.8-27b-fp8", "both told an appraisal (A/A)"),
    ("X-info-qwen", "info=b:P/s:P", "qwen3.8-27b-fp8", "both told the other's range (P/P)"),
]


def load_points(run):
    games = [json.loads(l) for l in open(os.path.join(ROOT, "data", "runs", run, "games.jsonl"))]
    pts = []
    for g in games:
        if g["terminal"] == "AGREEMENT" and len(g["offers"]) >= 2:
            mid = (g["offers"][0][2] + g["offers"][1][2]) / 2 / g["V"]
            pts.append((g["scale"], mid, g["price"] / g["V"]))
    return games[0]["model"], pts


def panel_a(y0):
    out = [text(20, y0, "a", 22, weight="bold"),
           text(46, y0, "Deals land on the midpoint of the two first offers (Study 2, every deal)", 19, weight="bold")]
    size, gap, left = 330, 70, 95
    top = y0 + 44
    lo, hi = 0.75, 1.25
    stats = []
    for k, run in enumerate(RUNS):
        model, pts = load_points(run)
        col = MODEL[model]["col"]
        px = left + k * (size + gap)
        P = lambda v: (v - lo) / (hi - lo) * size
        X = lambda v: px + P(v)
        Y = lambda v: top + size - P(v)
        out.append(text(px + size / 2, top - 12, MODEL[model]["label"], 16, col, "bold", "middle"))
        out.append(f"<rect x='{px}' y='{top}' width='{size}' height='{size}' fill='white' stroke='{GREY}' stroke-width='1'/>")
        for v in (0.8, 0.9, 1.0, 1.1, 1.2):
            out.append(line(X(v), top, X(v), top + size, LIGHT, 1))
            out.append(line(px, Y(v), px + size, Y(v), LIGHT, 1))
            out.append(text(X(v), top + size + 22, f"{v:.1f}", 15, ONYX, anchor="middle"))
            out.append(text(px - 8, Y(v) + 5, f"{v:.1f}", 15, ONYX, anchor="end"))
        out.append(line(X(lo), Y(lo), X(hi), Y(hi), GREY, 1.5, dash="5 4"))
        out.append(text(X(1.24), Y(0.765), "dashed: deal = midpoint", 15, GREY, anchor="end", italic=True))
        for sc, mid, price in pts:
            if not (lo <= mid <= hi and lo <= price <= hi):
                continue
            if sc == "x1e2":
                out.append(f"<circle cx='{X(mid):.1f}' cy='{Y(price):.1f}' r='4' fill='white' stroke='{col}' stroke-width='1.4'/>")
            elif sc == "x1e5":
                out.append(f"<rect x='{X(mid) - 3.2:.1f}' y='{Y(price) - 3.2:.1f}' width='6.4' height='6.4' fill='{GREY}' "
                           f"fill-opacity='0.55'/>")
            else:
                out.append(f"<circle cx='{X(mid):.1f}' cy='{Y(price):.1f}' r='4' fill='{col}' fill-opacity='0.8'/>")
        inside = sum(1 for _, m, p in pts if lo <= m <= hi and lo <= p <= hi)
        dev = statistics.median(p - m for _, m, p in pts)
        stats.append((model, len(pts), inside, dev))
        out.append(text(px + 10, top + 24, f"median (deal \u2212 midpoint) = {dev:+.3f} V", 15, ONYX))
        out.append(text(px + 10, top + 44, f"n = {len(pts)} deals" + ("" if inside == len(pts) else
                                                                    f" ({len(pts) - inside} off the axes)"), 15, GREY))
    out.append(text(left + size + gap / 2, top + size + 48, "midpoint of the two first offers (in units of V)", 15, ONYX,
                    anchor="middle"))
    out.append(text(30, top + size / 2, "deal price (in units of V)", 15, ONYX, anchor="middle", rotate=-90))
    # Legend.
    lx, ly = left + 2 * size + gap + 22, top + 30
    out.append(text(lx, ly, "scale", 15, ONYX, "bold"))
    for i, (sc, lab) in enumerate(SCALES):
        y = ly + 26 + 24 * i
        if sc == "x1e2":
            out.append(f"<circle cx='{lx + 6}' cy='{y - 5}' r='4.5' fill='white' stroke='{ONYX}' stroke-width='1.4'/>")
        elif sc == "x1e5":
            out.append(f"<rect x='{lx + 2}' y='{y - 9}' width='7' height='7' fill='{GREY}' fill-opacity='0.55'/>")
        else:
            out.append(f"<circle cx='{lx + 6}' cy='{y - 5}' r='4.5' fill='{ONYX}'/>")
        out.append(text(lx + 20, y, lab, 15, ONYX))
    return out, top + size + 64, stats


def same_pairs(data, sub, cell):
    """Total, opening and concession channel on ONE set of game pairs (fixture x opener).

    The set: pairs whose x1 and x10^6 games both have a seller share (a deal) and a first offer from
    each side. On it, s = (P - F)/W = (mid - F)/W + (P - mid)/W holds game by game, so
    total = opening (change in the first offers' midpoint, zone units) + concessions (change in the
    deal's deviation from that midpoint), exactly. The 10-08 version took each term from
    concessions_wp1.json, where each is averaged over its own pairs, so they need not sum.
    Per-game features come from scripts/analyze_concessions.py (imported, not modified); the
    each-on-its-own-pairs means are asserted against concessions_wp1.json first.
    """
    run = next(r["run_dir"] for r in data if sub in r["run_dir"])
    ref = cf.find_cell(data, sub, cell)
    games = {"lo": {}, "hi": {}}
    for l in open(os.path.join(ROOT, run, "games.jsonl")):
        g = json.loads(l)
        ck = ac.cell_key(g)
        if ck != ref["cell"]:
            continue
        sb, f = ac.scale_bucket(g), ac.game_features(g)
        if sb is None or f is None:
            continue
        games[sb][(f["fixture"], f["opener"])] = f
    lo, hi = games["lo"], games["hi"]
    shared = set(lo) & set(hi)
    assert len(shared) == ref["n_paired"], (sub, cell)

    def delta(field, keys):
        v = [hi[k][field] - lo[k][field] for k in keys if hi[k].get(field) is not None and lo[k].get(field) is not None]
        return sum(v) / len(v), len(v)

    for field in ("seller_share", "midpoint_dev"):  # same inputs as the WP1 analysis
        m, n = delta(field, shared)
        assert abs(m - ref[f"{field}__delta_mean"]) < 1e-9 and n == ref[f"{field}__delta_n"], (sub, cell, field)
    keys = [k for k in shared if all(x[k].get(f) is not None for x in (lo, hi)
                                     for f in ("seller_share", "opening_mid_share", "midpoint_dev"))]
    total, n = delta("seller_share", keys)
    opening, _ = delta("opening_mid_share", keys)
    conc, _ = delta("midpoint_dev", keys)
    resid = total - opening - conc
    assert abs(resid) < 1e-12, (sub, cell, resid)
    all_total, all_n = delta("seller_share", shared)
    return total, opening, conc, n, resid, all_total, all_n


def panel_b(y0):
    out = [text(20, y0, "b", 22, weight="bold"),
           text(46, y0, "Where the \u00d710\u2076 shift in seller share comes from: opening vs concessions", 19,
                weight="bold")]
    data = json.load(open(os.path.join(ROOT, "derived", "explore", "concessions_wp1.json")))
    x0, x1, lo, hi = 400, 760, -0.30, 0.10
    X = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)
    top, rh = y0 + 56, 34
    # Legend.
    out.append(f"<rect x='{x0}' y='{y0 + 18}' width='22' height='12' fill='{ONYX}' fill-opacity='0.75'/>")
    out.append(text(x0 + 30, y0 + 29, "opening channel", 15, ONYX))
    out.append(f"<rect x='{x0 + 170}' y='{y0 + 18}' width='22' height='12' fill='white' stroke='{ONYX}' "
               f"stroke-width='1.5' stroke-dasharray='3 2'/>")
    out.append(text(x0 + 200, y0 + 29, "concession channel", 15, ONYX))
    out.append(f"<rect x='{x0 + 362}' y='{y0 + 19}' width='10' height='10' fill='{ONYX}'/>")
    out.append(text(x0 + 380, y0 + 29, "total", 15, ONYX))
    rows = [(model, label) + same_pairs(data, sub, cell) for sub, cell, model, label in ROWS]
    bot = top + rh * len(rows)
    full = max(r[5] for r in rows)
    out.append(line(X(0), top - 6, X(0), bot, GREY, 1.2, dash="4 3"))
    for i, (model, label, total, opening, conc, n, resid, all_total, all_n) in enumerate(rows):
        cy = top + rh * i + rh / 2
        col = MODEL[model]["col"]
        if i == 0 or rows[i - 1][0] != model:
            out.append(text(30, cy - 1, MODEL[model]["label"], 16, col, "bold"))
        out.append(text(385, cy + 5, label, 15, ONYX, anchor="end"))
        out.append(f"<rect x='{min(X(0), X(opening)):.1f}' y='{cy - 7}' width='{abs(X(opening) - X(0)):.1f}' height='14' "
                   f"fill='{col}' fill-opacity='0.8'/>")
        a, b = sorted((X(opening), X(opening + conc)))
        out.append(f"<rect x='{a:.1f}' y='{cy - 7}' width='{b - a:.1f}' height='14' fill='white' stroke='{col}' "
                   f"stroke-width='1.5' stroke-dasharray='3 2'/>")
        out.append(f"<rect x='{X(total) - 5:.1f}' y='{cy - 5}' width='10' height='10' fill='{ONYX}'/>")
        out.append(text(775, cy + 5, f"{total:+.3f} = {opening:+.3f} {'+' if conc >= 0 else '\u2212'} {abs(conc):.3f}"
                        + ("" if n == full else f"  (n {n})"), 15, ONYX))
        if i > 0 and rows[i - 1][0] != model:
            out.append(line(30, cy - rh / 2, 960, cy - rh / 2, LIGHT, 1.5))
    out.append(line(x0, bot + 4, x1, bot + 4, ONYX, 1.2))
    for v in (-0.3, -0.2, -0.1, 0.0, 0.1):
        out.append(line(X(v), bot + 4, X(v), bot + 10, ONYX, 1.2))
        out.append(text(X(v), bot + 28, f"{v:+.1f}" if v else "0", 15, ONYX, anchor="middle"))
    out.append(text(X(0), bot + 50, "change in seller share at \u00d710\u2076, per game pair", 15, ONYX, anchor="middle"))
    out.append(text(30, bot + 72, f"Each row: the same game pairs (fixture \u00d7 opener) for all three terms, n = {full} unless "
                    "marked; pairs where both games reached a deal", 15, GREY))
    out.append(text(30, bot + 92, "after an offer from each side. Not the pre-registered fixture-level d. Exploratory "
                    "decomposition (WP1); total = opening + concessions exactly.", 15, GREY))
    return out, bot + 108


def main():
    a, ya, stats = panel_a(34)
    b, yb = panel_b(ya + 22)
    doc = svg_doc(1000, int(yb), "Deals land on the midpoint of the first offers; the shift is in the opening",
                  "Scatter of every Study 2 deal against the midpoint of the two first offers, for Gemma 4 31B and Qwen; "
                  "below, the x10^6 shift in seller share split into an opening and a concession channel for the "
                  "Study 2 game and the information probe. Exploratory.", a + b)
    path = write(os.path.join(HERE, "fig4_midpoint.svg"), doc)
    for s in stats:
        print("midpoint", s)
    data = json.load(open(os.path.join(ROOT, "derived", "explore", "concessions_wp1.json")))
    for sub, cell, model, label in ROWS:
        t, o, c, n, r, at, an = same_pairs(data, sub, cell)
        print(f"{model:16s} {label:34s} {t:+.4f} = {o:+.4f} + {c:+.4f} (resid {r:+.1e}; n {n}; all-deal pairs {at:+.4f}, n {an})")
    print(path)


if __name__ == "__main__":
    sys.exit(main())
