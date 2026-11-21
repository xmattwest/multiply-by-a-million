"""probe_figures.py: figures for the Q1 + Q3 probe (exploratory, not pre-registered).

    python3 scripts/probe_figures.py data/explore/X-q1q3-<model>-<stamp>

Writes to derived/explore/<run_id>/figures/:
  P1_split_by_condition.svg   seller share at $100 and $100M in each condition (A, N, LS, LB)
  P2_leak_as_anchor.svg       the informed side's opening offer vs the known limit, against no leak
  P3_q1_replication.svg       opening gap $100 -> $100M: Study 1/2 (earlier fixtures) vs probe A/N (fresh)
Colors are Okabe-Ito and never the only cue (shapes, dashes, labels). Stdlib only.
"""
import json
import os
import random
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_figures import BLUE, GREY, INK, ROOT, Svg   # noqa: E402
from explore_figures import mark, VERM                # noqa: E402

GREEN, PURPLE = "#009E73", "#CC79A7"
SC = (("x1e2", "$100"), ("x1e8", "$100M"))
NOTE = "Exploratory, not pre-registered · fresh fixtures X01–X12"
COND = {  # label, color, shape, dashed
    "A": ("A · appraisal shown", BLUE, "circle", False),
    "N": ("N · no appraisal", VERM, "square", False),
    "LS": ("LS · seller knows buyer's ceiling", GREEN, "up", False),
    "LB": ("LB · buyer knows seller's floor", PURPLE, "down", False),
}


def shape(s, x, y, color, kind, hollow=False, r=5.5):
    fill = "white" if hollow else color
    if kind in ("circle", "square"):
        mark(s, x, y, color, kind, hollow, r)
    else:
        d = (f"M{x:.1f},{y - r - 1:.1f} L{x + r + 1:.1f},{y + r:.1f} L{x - r - 1:.1f},{y + r:.1f} Z" if kind == "up" else
             f"M{x:.1f},{y + r + 1:.1f} L{x + r + 1:.1f},{y - r:.1f} L{x - r - 1:.1f},{y - r:.1f} Z")
        s.add(f"<path d='{d}' fill='{fill}' stroke='{color}' stroke-width='2'/>")


def ci(vals, rng, n=4000):
    m = [st.fmean(rng.choice(vals) for _ in vals) for _ in range(n)]
    m.sort()
    return m[int(0.025 * n)], m[int(0.975 * n) - 1]


def load(run):
    return [json.loads(l) for l in open(os.path.join(ROOT, run, "games.jsonl")) if l.strip()]


def share(g):
    return g["seller_share"]["float"] if g.get("seller_share") else None


def axes(s, x0, y0, w, h, lo, hi, ticks, fmt, title):
    Y = lambda v: y0 + h - (v - lo) / (hi - lo) * h       # noqa: E731
    s.text(x0, y0 - 14, title, 13.5, weight="bold")
    for t in ticks:
        s.line(x0, Y(t), x0 + w, Y(t), "#e6e6e6")
        s.text(x0 - 6, Y(t) + 4, fmt.format(t), 11.5, "end", "#444")
    s.line(x0, y0 + h, x0 + w, y0 + h, INK)
    return Y


def halo(s, x, y, txt, color, size=12, anchor="start"):
    s.add(f"<text x='{x:.1f}' y='{y:.1f}' font-size='{size}' fill='{color}' font-weight='bold' text-anchor='{anchor}'"
          f" stroke='white' stroke-width='4' paint-order='stroke'>{txt}</text>")


def fig_split(G, out):
    rng = random.Random(1)
    s = Svg(900, 540, "Seller share at $100 and $100M by condition",
            "Four lines. With the appraisal (A) and with no appraisal (N) the seller share barely moves with stakes."
            " When the seller knows the buyer's ceiling (LS) its share rises from 0.45 to 0.56 at $100M; when the buyer"
            " knows the seller's floor (LB) the seller share falls from 0.41 to 0.23. Bars are 95% bootstrap intervals.")
    s.text(30, 32, "With inside information, the stakes start to matter", 19, weight="bold")
    s.text(30, 54, "Mean seller share of the surplus over deals (bars: 95% bootstrap interval). 24 games per point. " + NOTE,
           12, color="#444")
    x0, y0, w, h = 110, 100, 420, 340
    Y = axes(s, x0, y0, w, h, 0, 0.8, [0, 0.2, 0.4, 0.6, 0.8], "{:.1f}", "Seller's share of the surplus")
    X = [x0 + w * 0.2, x0 + w * 0.8]
    for x, (_, lab) in zip(X, SC):
        s.text(x, y0 + h + 20, lab, 13, "middle", weight="bold")
    labels = []
    for k, (lab, color, kind, _) in COND.items():
        pts = []
        for i, (sc, _) in enumerate(SC):
            v = [share(g) for g in G if g["condition"] == k and g["scale"] == sc and share(g) is not None]
            m, (a, b) = st.fmean(v), ci(v, rng)
            off = {"A": -9, "N": -3, "LS": 3, "LB": 9}[k]
            pts.append((X[i] + off, m))
            s.line(X[i] + off, Y(a), X[i] + off, Y(b), color, 1.6)
        s.add(f"<polyline points='{' '.join(f'{x:.1f},{Y(m):.1f}' for x, m in pts)}' fill='none' stroke='{color}'"
              f" stroke-width='2.4'/>")
        for x, m in pts:
            shape(s, x, Y(m), color, kind)
        labels.append([Y(pts[1][1]), f"{lab}: {pts[0][1]:.2f} → {pts[1][1]:.2f}", color])
    labels.sort()
    for i in range(1, len(labels)):
        labels[i][0] = max(labels[i][0], labels[i - 1][0] + 17)
    for y, t, color in labels:
        halo(s, X[1] + 24, y + 4, t, color)
    s.text(30, 490, "With no leak (A, N) the split barely moves with stakes. With a leak, the informed side gains more at $100M:",
           12.5, color="#444")
    s.text(30, 508, "Informed side's share vs the same game with no leak: seller (LS) +0.13 at $100, +0.28 at $100M;"
           " buyer (LB) −0.09 at $100 (it loses), +0.05 at $100M.", 12.5, color="#444")
    s.save(os.path.join(out, "P1_split_by_condition.svg"))


def fig_anchor(G, out):
    rng = random.Random(2)
    s = Svg(980, 580, "The leaked limit becomes the informed side's opening offer",
            "Two panels of dots, one per game. Buyer opens: bid divided by the seller's floor; with the leak almost every"
            " bid is exactly 1.0, the floor, while without it bids at $100 sit well below the floor. Seller opens: ask"
            " divided by the buyer's ceiling; with the leak asks move up toward the ceiling, most at $100M.")
    s.text(30, 32, "The leaked number becomes the opening offer", 19, weight="bold")
    s.text(30, 54, "Opener's first offer relative to the OTHER side's limit. Hollow: no leak (N). Filled: the opener knows that limit. "
           + NOTE, 12, color="#444")
    for j, (side, cond, lim, title, lo, hi, ticks) in enumerate((
            ("buyer", "LB", "F", "Buyer opens: bid ÷ seller's floor", 0.6, 1.3, [0.6, 0.8, 1.0, 1.2]),
            ("seller", "LS", "C", "Seller opens: ask ÷ buyer's ceiling", 0.6, 1.3, [0.6, 0.8, 1.0, 1.2]))):
        x0, y0, w, h = 90 + j * 470, 100, 360, 330
        Y = axes(s, x0, y0, w, h, lo, hi, ticks, "{:.1f}×", title)
        s.line(x0, Y(1.0), x0 + w, Y(1.0), INK, 1.4, "6 4")
        s.text(x0 - 6, Y(1.0) + 17, "= limit", 10.5, "end", "#444", italic=True)
        for i, (sc, lab) in enumerate(SC):
            xc = x0 + w * (0.25 + 0.5 * i)
            s.text(xc, y0 + h + 20, lab, 13, "middle", weight="bold")
            for k, dx, hollow, color in (("N", -38, True, VERM), (cond, 38, False, COND[cond][1])):
                v = [g["offers"][0][2] / g[lim] for g in G
                     if g["condition"] == k and g["scale"] == sc and g["opener"] == side and g["offers"]]
                for x in v:
                    shape(s, xc + dx + rng.uniform(-14, 14), Y(min(max(x, lo), hi)), color,
                          "square" if k == "N" else COND[k][2], hollow, 4.5)
                m = st.median(v)
                s.line(xc + dx - 22, Y(m), xc + dx + 22, Y(m), INK, 2.6)
                halo(s, xc + dx, y0 + h - 8, f"{m:.2f}", INK, 11, "middle")
            s.text(xc - 38, y0 + 14, "no leak", 11, "middle", "#444")
            s.text(xc + 38, y0 + 14, "knows", 11, "middle", "#444")
        exact = {sc: sum(g["offers"][0][2] == g[lim] for g in G if g["condition"] == cond and g["scale"] == sc
                         and g["opener"] == side) for sc, _ in SC}
        if side == "buyer":
            s.text(x0, y0 + h + 66, "Below 1.0: a bid under the seller's floor, where no deal is possible.", 11.5,
                   color="#444", italic=True)
        s.text(x0, y0 + h + 48, f"Opens at exactly that limit: $100 {exact['x1e2']}/12, $100M {exact['x1e8']}/12"
               + (" — the leak replaces the lowball" if side == "buyer" else " — asks rise but stay below it"), 12, color="#444")
    s.text(30, 560, "Black bars: medians. Reading: the informed side treats the leaked number as where to start, not as leverage."
           " For the buyer that start is higher than its usual lowball.", 12, color="#444")
    s.save(os.path.join(out, "P2_leak_as_anchor.svg"))


def fig_q1(G, out):
    def gap(g):
        if not g["offers"] or g["offers"][0][1] != g["opener"]:
            return None
        p = g["offers"][0][2]
        return 100 * (p / g["F"] - 1) if g["opener"] == "seller" else 100 * (1 - p / g["C"])

    def means(games, side, cond=None):
        return [st.fmean(x for g in games if (cond is None or g.get("condition") == cond) and g["opener"] == side
                         and g["scale"] == sc and (x := gap(g)) is not None) for sc, _ in SC]
    S1 = load("data/runs/qwen3.8-27b-fp8-20261003T180835Z")
    S2 = load("data/runs/s2-qwen3.8-27b-fp8-20261003T223042Z")
    series = (("Study 1 (appraisal), earlier fixtures", S1, None, BLUE, "circle", True),
              ("Study 2 (no appraisal), earlier fixtures", S2, None, VERM, "square", True),
              ("Probe A (appraisal), fresh fixtures", G, "A", BLUE, "circle", False),
              ("Probe N (no appraisal), fresh fixtures", G, "N", VERM, "square", False))
    s = Svg(980, 560, "Q1: opener modesty at $100M replicates on fresh fixtures, only without the appraisal",
            "Two panels: seller's markup over its floor and buyer's discount under its ceiling, at $100 and $100M. Without"
            " the appraisal (Study 2 and probe N) both shrink at $100M; with it (Study 1 and probe A) they don't. The"
            " probe's shrink is smaller than Study 2's.")
    s.text(30, 32, "Q1: big numbers make the opener modest, again only without the appraisal", 19, weight="bold")
    s.text(30, 54, "Opener's first offer, distance from its own limit (mean). Dashed + hollow: earlier runs (30 fixtures)."
           " Solid: this probe (12 fresh fixtures, randomized in one session).", 12, color="#444")
    for j, (side, title) in enumerate((("seller", "Seller opens: ask above its floor"),
                                       ("buyer", "Buyer opens: bid below its ceiling"))):
        x0, y0, w, h = 80 + j * 470, 100, 240, 300
        Y = axes(s, x0, y0, w, h, 0, 40, [0, 10, 20, 30, 40], "{:.0f}%", title)
        X = [x0 + w * 0.2, x0 + w * 0.8]
        for x, (_, lab) in zip(X, SC):
            s.text(x, y0 + h + 20, lab, 13, "middle", weight="bold")
        labels = []
        for lab, games, cond, color, kind, hollow in series:
            v = means(games, side, cond)
            dash = " stroke-dasharray='5 4'" if hollow else ""
            s.add(f"<polyline points='{X[0]:.1f},{Y(v[0]):.1f} {X[1]:.1f},{Y(v[1]):.1f}' fill='none' stroke='{color}'"
                  f" stroke-width='{1.6 if hollow else 2.6}'{dash}/>")
            for x, m in zip(X, v):
                shape(s, x, Y(m), color, kind, hollow)
            labels.append([Y(v[1]), f"{lab.split(',')[0]} {v[0]:.0f}→{v[1]:.0f}%", color])
        labels.sort()
        for i in range(1, len(labels)):
            labels[i][0] = max(labels[i][0], labels[i - 1][0] + 16)
        for y, t, color in labels:
            halo(s, X[1] + 14, y + 4, t, color, 11.5)
    s.text(30, 470, "Probe N: smaller at $100M in 8 of 12 fixtures for each side (p 0.02, 0.01). Probe A: 4 and 3 of 12.",
           12.5, color="#444")
    s.text(30, 490, "Not replicated: the opener's-advantage flip (HB). In probe N the opener × stakes interaction is −0.04"
           " (p 0.68), against −0.33 in Study 2.", 12.5, color="#444", weight="bold")
    s.text(30, 510, NOTE + " · qwen3.8-27b-fp8", 12, color="#444")
    s.save(os.path.join(out, "P3_q1_replication.svg"))


def main(argv):
    run = argv[0]
    G = [g for g in load(run) if g["terminal"] != "INFRASTRUCTURE_VOID"]
    out = os.path.join(ROOT, "derived", "explore", os.path.basename(os.path.normpath(run)), "figures")
    os.makedirs(out, exist_ok=True)
    fig_split(G, out)
    fig_anchor(G, out)
    fig_q1(G, out)
    for f in sorted(os.listdir(out)):
        print(os.path.relpath(os.path.join(out, f), ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
