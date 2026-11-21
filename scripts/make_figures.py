#!/usr/bin/env python3
"""make_figures.py: SVG figures for one model's primary result.

    python3 scripts/make_figures.py data/runs/<run_id>

Reads derived/<run_id>/primary.json (from scripts/analyze.py) and the run's
games.jsonl. Writes three SVGs to derived/<run_id>/figures/:
  1_shift_per_fixture.svg   d_i per fixture, the mean with its 95% and 90%
                            CIs, and the ±0.10 equivalence margin (§5.1-5.2)
  2_paired_shares.svg       seller share at $100 vs at $100M for every
                            fixture × opener pair where both games agreed
  3_shares_by_scale.svg     every agreed seller share at $100, $100K, $100M
                            (descriptive; the S1 tests come later)
Accessible: each SVG has a <title> and <desc>; colors are Okabe-Ito and
never the only cue (shapes and labels too). Standard library only.
"""
import json
import os
import statistics
import sys
from xml.sax.saxutils import escape

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLUE, ORANGE, GREEN, GREY, INK = "#0072B2", "#E69F00", "#009E73", "#999999", "#222222"
FONT = "font-family='-apple-system, Helvetica, Arial, sans-serif'"
SCALES = (("x1e2", "$100"), ("x1e5", "$100K"), ("x1e8", "$100M"))


class Svg:
    def __init__(self, w, h, title, desc):
        self.w, self.h, self.parts = w, h, []
        self.head = (f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}' viewBox='0 0 {w} {h}'"
                     f" role='img' aria-labelledby='t d' {FONT}>"
                     f"<title id='t'>{escape(title)}</title><desc id='d'>{escape(desc)}</desc>"
                     f"<rect width='{w}' height='{h}' fill='white'/>")

    def add(self, s):
        self.parts.append(s)

    def text(self, x, y, s, size=13, anchor="start", color=INK, weight="normal", italic=False):
        style = " font-style='italic'" if italic else ""
        self.add(f"<text x='{x:.1f}' y='{y:.1f}' font-size='{size}' text-anchor='{anchor}' fill='{color}'"
                 f" font-weight='{weight}'{style}>{escape(s)}</text>")

    def line(self, x1, y1, x2, y2, color=INK, width=1, dash=None):
        d = f" stroke-dasharray='{dash}'" if dash else ""
        self.add(f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' stroke='{color}'"
                 f" stroke-width='{width}'{d}/>")

    def save(self, path):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.head + "".join(self.parts) + "</svg>\n")


def marker(svg, x, y, opener, r=5):
    if opener == "seller":      # circle
        svg.add(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{r}' fill='{BLUE}' fill-opacity='0.8' stroke='white'/>")
    else:                       # triangle
        svg.add(f"<path d='M{x:.1f},{y - r - 1:.1f} L{x + r + 1:.1f},{y + r:.1f} L{x - r - 1:.1f},{y + r:.1f} Z'"
                f" fill='{ORANGE}' fill-opacity='0.85' stroke='white'/>")


def fig_shift(res, out, model):
    d = sorted(((v["float"], f) for f, v in res["d_i"].items()), reverse=True)
    n = len(d)
    W, top, row = 800, 120, 15
    H = top + n * row + 95
    L, R = 175, W - 60
    span = max(0.5, max(abs(v) for v, _ in d) + 0.05)
    span = round(span * 10 + 0.4999) / 10                 # widen to fit, in steps of 0.1
    lo, hi = -span, span
    X = lambda v: L + (v - lo) / (hi - lo) * (R - L)      # noqa: E731
    m, c95, c90 = res["mean_d"], res["bootstrap"]["ci95_mean"], res["bootstrap"]["ci90_mean"]
    p = res["H1"]["p_unadjusted"]
    s = Svg(W, H, f"{model}: change in seller share from $100 to $100M, per fixture",
            f"Dot plot of {n} fixtures. Each dot is the seller's share at $100M minus at $100. Mean {m:+.3f}, "
            f"95% CI {c95[0]:+.3f} to {c95[1]:+.3f}, 90% CI {c90[0]:+.3f} to {c90[1]:+.3f}, inside the ±0.10 "
            f"equivalence margin. Wilcoxon p = {p:.2f} unadjusted.")
    s.text(20, 30, f"Does a 1,000,000× bigger deal change the split?  {model}", 17, weight="bold")
    s.text(20, 52, "Seller's share of the bargaining range at $100M minus at $100, one dot per fixture"
           " (average of its two opener games)", 12.5, color="#444")
    # equivalence band and zero line
    s.add(f"<rect x='{X(-0.1):.1f}' y='{top - 46}' width='{X(0.1) - X(-0.1):.1f}' height='{n * row + 52}'"
          f" fill='{GREEN}' fill-opacity='0.10'/>")
    s.text(X(0.1) - 4, top - 34, "equivalence margin ±0.10", 11, "end", GREEN)
    s.line(X(0), top - 46, X(0), top + n * row + 6, GREY, 1, "4 3")
    # summary row: mean, 95% CI (thin), 90% CI (thick)
    y0 = top - 20
    s.line(X(c95[0]), y0, X(c95[1]), y0, INK, 2)
    s.line(X(c90[0]), y0, X(c90[1]), y0, INK, 6)
    s.add(f"<rect x='{X(m) - 5:.1f}' y='{y0 - 5}' width='10' height='10' fill='white' stroke='{INK}' stroke-width='2'/>")
    s.text(L - 10, y0 + 4, "mean, 90% / 95% CI", 11.5, "end", weight="bold")
    s.text(X(c95[1]) + 8, y0 + 4, f"{m:+.3f}", 11.5)
    # fixtures
    for k, (v, f) in enumerate(d):
        y = top + 8 + k * row
        s.line(L, y, R, y, "#f0f0f0", 1)
        s.add(f"<circle cx='{X(v):.1f}' cy='{y:.1f}' r='4.5' fill='{BLUE}'/>")
        s.text(L - 10, y + 4, f, 10.5, "end", "#555")
    # axis
    ya = top + n * row + 14
    s.line(L, ya, R, ya, INK)
    step = 0.1 if span <= 0.5 else 0.2
    for t in [round(k * step, 1) for k in range(-int(span / step + 1e-9), int(span / step + 1e-9) + 1)]:
        s.line(X(t), ya, X(t), ya + 5, INK)
        s.text(X(t), ya + 19, f"{t:+.1f}" if t else "0", 11, "middle")
    s.text(X(-span * 0.6), ya + 40, "◀ seller gets less at $100M", 12, "middle", "#444")
    s.text(X(span * 0.6), ya + 40, "seller gets more at $100M ▶", 12, "middle", "#444")
    s.text(20, H - 14, f"n = {n} fixtures, {res['H1']['n_nonzero']} nonzero. Wilcoxon signed-rank p = {p:.2f}"
           f" (unadjusted). 90% CI inside ±0.10 → invariant within ±0.10 (pre-registered reading).", 11, color="#444")
    s.save(os.path.join(out, "1_shift_per_fixture.svg"))


def fig_pairs(games, out, model):
    by = {(g["fixture"], g["opener"], g["scale"]): g for g in games}
    pts = []
    for (f, o, sc), g in by.items():
        if sc == "x1e2":
            hi = by.get((f, o, "x1e8"))
            if hi and g["terminal"] == hi["terminal"] == "AGREEMENT":
                pts.append((g["seller_share"]["float"], hi["seller_share"]["float"], o))
    W = H = 560
    L, T, S = 80, 70, 420
    X = lambda v: L + v * S          # noqa: E731
    Y = lambda v: T + (1 - v) * S    # noqa: E731
    s = Svg(W, H + 40, f"{model}: seller share at $100 versus $100M, paired",
            f"Scatter of {len(pts)} fixture-opener pairs. Points near the diagonal mean the split was the "
            f"same at both scales. Circles: seller opened. Triangles: buyer opened.")
    s.text(20, 30, f"Same negotiation, $100 vs $100M  ·  {model}", 17, weight="bold")
    s.text(20, 52, "Each point: one fixture and opener, both games agreed. On the line = same split.", 12.5,
           color="#444")
    s.add(f"<rect x='{L}' y='{T}' width='{S}' height='{S}' fill='none' stroke='#ddd'/>")
    for t in (0, 0.25, 0.5, 0.75, 1):
        s.line(X(t), T + S, X(t), T + S + 5, INK); s.text(X(t), T + S + 20, f"{t:g}", 11, "middle")
        s.line(L - 5, Y(t), L, Y(t), INK); s.text(L - 9, Y(t) + 4, f"{t:g}", 11, "end")
        s.line(X(t), T, X(t), T + S, "#f0f0f0"); s.line(L, Y(t), L + S, Y(t), "#f0f0f0")
    s.line(X(0), Y(0), X(1), Y(1), GREY, 1.5, "5 4")
    s.text(X(0.97), Y(0.97) + 16, "same split", 11, "end", GREY, italic=True)
    for a, b, o in pts:
        marker(s, X(a), Y(b), o)
    s.text(L + S / 2, T + S + 44, "seller's share at $100", 13, "middle")
    s.add(f"<text x='{L - 50}' y='{T + S / 2}' font-size='13' text-anchor='middle' fill='{INK}'"
          f" transform='rotate(-90 {L - 50} {T + S / 2})'>seller's share at $100M</text>")
    marker(s, L + 10, H + 18, "seller"); s.text(L + 22, H + 22, "seller opened", 12)
    marker(s, L + 140, H + 18, "buyer"); s.text(L + 152, H + 22, "buyer opened", 12)
    s.text(L + 270, H + 22, f"{len(pts)} pairs", 12, color="#444")
    s.save(os.path.join(out, "2_paired_shares.svg"))


def fig_scales(games, out, model):
    W, H = 640, 505
    L, T, S = 80, 75, 330
    allv = [g["seller_share"]["float"] for g in games if g["terminal"] == "AGREEMENT"]
    lo, hi = min(0.0, min(allv) - 0.05), max(1.0, max(allv) + 0.05)
    Y = lambda v: T + (hi - v) / (hi - lo) * S    # noqa: E731
    s = Svg(W, H, f"{model}: seller share at each scale",
            "Strip plot of every agreed game's seller share at $100, $100K and $100M, with medians. "
            "Descriptive; the pre-registered secondary tests come later.")
    s.text(20, 30, f"Seller's share at each scale  ·  {model}", 17, weight="bold")
    s.text(20, 52, "Every game that ended in a deal; bar = median. Below 0 or above 1: a deal past a side's"
           " limit. Descriptive only.", 12.5, color="#444")
    if lo < 0 or hi > 1:
        s.add(f"<rect x='{L}' y='{Y(1):.1f}' width='{W - 30 - L}' height='{Y(0) - Y(1):.1f}' fill='{GREEN}' fill-opacity='0.06'/>")
    for t in (0, 0.25, 0.5, 0.75, 1):
        s.line(L, Y(t), W - 30, Y(t), "#f0f0f0"); s.text(L - 9, Y(t) + 4, f"{t:g}", 11, "end")
    s.add(f"<text x='{L - 50}' y='{T + S / 2}' font-size='13' text-anchor='middle' fill='{INK}'"
          f" transform='rotate(-90 {L - 50} {T + S / 2})'>seller's share of the range</text>")
    colw = (W - 30 - L) / 3
    for k, (sc, label) in enumerate(SCALES):
        cx = L + colw * (k + 0.5)
        vals = sorted((g["seller_share"]["float"], g["opener"], g["game_id"]) for g in games
                      if g["scale"] == sc and g["terminal"] == "AGREEMENT")
        for j, (v, o, gid) in enumerate(vals):
            jit = ((sum(map(ord, gid)) * 37) % 100 / 100 - 0.5) * colw * 0.5   # deterministic jitter
            marker(s, cx + jit, Y(v), o, 4)
        med = statistics.median(v for v, _, _ in vals)
        s.line(cx - colw * 0.32, Y(med), cx + colw * 0.32, Y(med), INK, 3)
        s.text(cx + colw * 0.34, Y(med) + 4, f"{med:.2f}", 11.5, weight="bold")
        s.text(cx, T + S + 24, label, 14, "middle", weight="bold")
        n_all = sum(1 for g in games if g["scale"] == sc)
        s.text(cx, T + S + 42, f"{len(vals)} of {n_all} agreed", 11.5, "middle", "#444")
    marker(s, L + 10, H - 16, "seller", 4); s.text(L + 20, H - 12, "seller opened", 12)
    marker(s, L + 130, H - 16, "buyer", 4); s.text(L + 140, H - 12, "buyer opened", 12)
    s.save(os.path.join(out, "3_shares_by_scale.svg"))


def main(argv):
    run_dir = argv[0]
    rid = os.path.basename(os.path.normpath(run_dir))
    res = json.load(open(os.path.join(ROOT, "derived", rid, "primary.json")))
    games = [json.loads(l) for l in open(os.path.join(run_dir, "games.jsonl")) if l.strip()]
    games = [g for g in games if g["terminal"] != "INFRASTRUCTURE_VOID"]
    man = json.load(open(os.path.join(run_dir, "run_manifest.json")))
    study = man.get("study", 1)
    label = f"Study {study} ({'appraisal shown' if study == 1 else 'no appraisal'}) · {res['model']}"
    out = os.path.join(ROOT, "derived", rid, "figures")
    os.makedirs(out, exist_ok=True)
    fig_shift(res, out, label)
    fig_pairs(games, out, label)
    fig_scales(games, out, label)
    for f in sorted(os.listdir(out)):
        print(os.path.relpath(os.path.join(out, f), ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
