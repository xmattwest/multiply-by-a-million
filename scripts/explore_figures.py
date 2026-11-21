"""explore_figures.py: figures for the exploratory findings (not pre-registered).

    python3 scripts/explore_figures.py

Reads the four Qwen datasets (Study 1 and 2, pilot and real; the same set as
scripts/explore.py). Writes SVGs to derived/explore/figures/:
  E1_opening_gap.svg         opener's first offer, distance from its own limit, by scale
  E2_opener_edge.svg         seller share when the seller opens minus when the buyer opens
  E3_opening_per_fixture.svg $100 -> $100M opening gap for every fixture (real runs)
  E4_process_vs_outcome.svg  the split stays flat while the bargaining process changes
Colors are Okabe-Ito and never the only cue: studies differ by marker shape,
pilots are dashed with hollow markers, and lines are labelled. Stdlib only.
"""
import json
import os
import re
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_figures import BLUE, GREY, INK, ROOT, SCALES, Svg  # noqa: E402

VERM = "#D55E00"
DATASETS = (  # key, label, run dir, color, pilot?, marker shape
    ("S1", "Study 1 (appraisal shown)", "data/runs/qwen3.8-27b-fp8-20261003T180835Z", BLUE, False, "circle"),
    ("S1p", "Study 1 pilot", "data/pilot/pilot-qwen3.8-27b-fp8-20261003T173737Z", BLUE, True, "circle"),
    ("S2", "Study 2 (no appraisal)", "data/runs/s2-qwen3.8-27b-fp8-20261003T223042Z", VERM, False, "square"),
    ("S2p", "Study 2 pilot", "data/pilot/s2-pilot-qwen3.8-27b-fp8-20261003T220356Z", VERM, True, "square"),
)
SC = [s for s, _ in SCALES]
FINAL = re.compile(r"\bfinal\b|best and final|\bbest offer\b", re.I)
NOTE = "Exploratory, not pre-registered · qwen3.8-27b-fp8"


def load(run):
    games = [json.loads(l) for l in open(os.path.join(ROOT, run, "games.jsonl")) if l.strip()]
    games = [g for g in games if g["terminal"] != "INFRASTRUCTURE_VOID"]
    text = {}
    for l in open(os.path.join(ROOT, run, "calls.jsonl")):
        c = json.loads(l)
        if c["schema"] == "call-result/1":
            try:
                text[c["call_id"]] = json.loads(json.loads(c["response_text"])["choices"][0]["message"]["content"]).get("message") or ""
            except (ValueError, KeyError, TypeError):
                text[c["call_id"]] = ""
    for g in games:
        g["texts"] = [text.get(cid, "") for cid in g["call_ids"]]
    return games


def gap(g):
    """Opener's first offer: % markup over own floor (seller) or % discount under own ceiling (buyer)."""
    p = g["offers"][0][2]
    return 100 * (p / g["F"] - 1) if g["opener"] == "seller" else 100 * (1 - p / g["C"])


def share(g):
    return g["seller_share"]["float"] if g["seller_share"] else None


def mean(v):
    return st.fmean(v) if v else None


def stats(games):
    out = {"gap": {}, "edge": [], "share": [], "step": [], "offers": [], "words": [], "final": []}
    for side in ("seller", "buyer"):
        out["gap"][side] = [mean([gap(g) for g in games if g["opener"] == side and g["scale"] == s]) for s in SC]
    for s in SC:
        gs = [g for g in games if g["scale"] == s]
        ag = {o: [share(g) for g in gs if g["opener"] == o and share(g) is not None] for o in ("seller", "buyer")}
        out["edge"].append(mean(ag["seller"]) - mean(ag["buyer"]))
        out["share"].append(mean(ag["seller"] + ag["buyer"]))
        steps = []
        for g in gs:
            for side in ("seller", "buyer"):
                o = [p for _, r, p in g["offers"] if r == side]
                steps += [abs(b - a) / g["V"] * 100 for a, b in zip(o, o[1:])]
        out["step"].append(st.median(steps))
        out["offers"].append(mean([len(g["offers"]) for g in gs]))
        t = [x for g in gs for x in g["texts"]]
        out["words"].append(mean([len(x.split()) for x in t]))
        out["final"].append(100 * mean([bool(FINAL.search(x)) for x in t]))
    return out


def per_fixture(games, side):
    by = {(g["fixture"], g["opener"], g["scale"]): g for g in games}
    pairs = []
    for f in sorted({g["fixture"] for g in games}):
        lo, hi = by.get((f, side, "x1e2")), by.get((f, side, "x1e8"))
        if lo and hi:
            pairs.append((gap(lo), gap(hi)))
    return pairs


def mark(s, x, y, color, shape, hollow, r=5):
    fill = "white" if hollow else color
    if shape == "circle":
        s.add(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{r}' fill='{fill}' stroke='{color}' stroke-width='2'/>")
    else:
        s.add(f"<rect x='{x - r:.1f}' y='{y - r:.1f}' width='{2 * r}' height='{2 * r}' fill='{fill}'"
              f" stroke='{color}' stroke-width='2'/>")


def panel(s, x0, y0, w, h, lo, hi, ticks, series, title, fmt="{:g}", end_labels=True, zero=None):
    """Line chart over the three scales. series: (label, values, color, pilot, shape)."""
    X = [x0 + w * (i + 0.5) / 3 for i in range(3)]
    Y = lambda v: y0 + h - (v - lo) / (hi - lo) * h          # noqa: E731
    s.text(x0, y0 - 12, title, 13.5, weight="bold")
    for t in ticks:
        s.line(x0, Y(t), x0 + w, Y(t), "#e6e6e6")
        s.text(x0 - 6, Y(t) + 4, fmt.format(t), 11.5, "end", "#444")
    if zero is not None:
        s.line(x0, Y(zero), x0 + w, Y(zero), INK, 1.2)
    s.line(x0, y0 + h, x0 + w, y0 + h, INK)
    for x, (_, lab) in zip(X, SCALES):
        s.text(x, y0 + h + 18, lab, 12.5, "middle", weight="bold")
    labels = []
    for lab, vals, color, pilot, shape in series:
        pts = " ".join(f"{x:.1f},{Y(v):.1f}" for x, v in zip(X, vals))
        dash = " stroke-dasharray='5 4'" if pilot else ""
        s.add(f"<polyline points='{pts}' fill='none' stroke='{color}' stroke-width='{1.6 if pilot else 2.4}'{dash}/>")
        for x, v in zip(X, vals):
            mark(s, x, Y(v), color, shape, pilot)
        labels.append([Y(vals[-1]), lab, color, vals[-1]])
    if end_labels:                          # spread labels so they don't overlap
        labels.sort()
        for i in range(1, len(labels)):
            labels[i][0] = max(labels[i][0], labels[i - 1][0] + 15)
        for y, lab, color, v in labels:
            val = fmt.format(v) if "%" in fmt else f"{v:+.2f}" if zero is not None else f"{v:.2f}"
            s.add(f"<text x='{X[-1] + 12:.1f}' y='{y + 4:.1f}' font-size='11.5' fill='{color}' font-weight='bold'"
                  f" stroke='white' stroke-width='4' paint-order='stroke'>{lab}  {val}</text>")


def legend(s, x, y):
    for i, (_, lab, _, color, pilot, shape) in enumerate(DATASETS):
        xx = x + (i % 2) * 300
        yy = y + (i // 2) * 20
        dash = " stroke-dasharray='5 4'" if pilot else ""
        s.add(f"<line x1='{xx}' y1='{yy}' x2='{xx + 30}' y2='{yy}' stroke='{color}' stroke-width='2'{dash}/>")
        mark(s, xx + 15, yy, color, shape, pilot, 4)
        s.text(xx + 38, yy + 4, lab + (" (separate fixtures)" if pilot else ""), 12)


def fig_gap(D, out, counts):
    s = Svg(980, 520, "Opening offers move toward the opener's own limit at higher stakes, only without the appraisal",
            "Two panels. Seller opens: markup of the first ask over the seller's floor. Buyer opens: discount of the first"
            " bid under the buyer's ceiling. Study 2 lines fall from $100 to $100M in both the real run and its pilot;"
            " Study 1 lines stay flat.")
    s.text(30, 32, "Big numbers make the opener modest, but only without the appraisal", 19, weight="bold")
    s.text(30, 54, "Opener's first offer: how far it sits from the opener's own limit (mean over games). " + NOTE, 12.5, color="#444")
    for j, (side, title, lab) in enumerate((("seller", "Seller opens: ask above its floor", "markup"),
                                            ("buyer", "Buyer opens: bid below its ceiling", "discount"))):
        x0 = 75 + j * 470
        series = [(k, D[k]["gap"][side], c, p, sh) for k, _, _, c, p, sh in DATASETS]
        panel(s, x0, 100, 250, 290, 0, 40, range(0, 41, 10), series, title, "{:.0f}%")
        s.text(x0, 440, f"Smaller at $100M than at $100, same fixture:", 11.5, color="#444")
        s.text(x0, 456, f"Study 2 {counts['S2'][side]} · Study 1 {counts['S1'][side]}", 11.5, color="#444", weight="bold")
    legend(s, 75, 488)
    s.save(os.path.join(out, "E1_opening_gap.svg"))


def fig_edge(D, out):
    s = Svg(820, 500, "The opener's advantage reverses at high stakes without the appraisal",
            "Seller share when the seller opens minus seller share when the buyer opens, at $100, $100K and $100M."
            " Study 2 and its pilot fall from about +0.1 to between -0.24 and -0.34. Study 1 and its pilot stay near zero.")
    s.text(30, 32, "Without the appraisal, going first stops paying at $100M", 19, weight="bold")
    s.text(30, 54, "Seller's share when the seller opens minus when the buyer opens. Above 0: the opener does better. " , 12.5, color="#444")
    s.text(30, 72, NOTE, 12.5, color="#444")
    series = [(k, D[k]["edge"], c, p, sh) for k, _, _, c, p, sh in DATASETS]
    panel(s, 90, 110, 480, 280, -0.4, 0.2, [-0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2], series,
          "Opener's advantage", "{:+.1f}", zero=0)
    s.text(96, 126, "▲ opener does better", 11.5, color="#444", italic=True)
    s.text(96, 382, "▼ responder does better", 11.5, color="#444", italic=True)
    legend(s, 90, 450)
    s.save(os.path.join(out, "E2_opener_edge.svg"))


def fig_fixtures(G, out):
    s = Svg(1000, 640, "Opening gap at $100 and at $100M for every fixture",
            "Slopegraphs, one line per fixture, for Study 1 and Study 2 real runs, seller-opened and buyer-opened."
            " Most Study 2 lines slope down; Study 1 lines go both ways.")
    s.text(30, 32, "Fixture by fixture: Study 2 openers shrink their gap at $100M", 19, weight="bold")
    s.text(30, 54, "One line per fixture: opener's first-offer gap from its own limit at $100 (left) and $100M (right). "
           "Real runs. " + NOTE, 12, color="#444")
    lo, hi = 0, 60
    for i, (key, slab) in enumerate((("S1", "Study 1 (appraisal)"), ("S2", "Study 2 (no appraisal)"))):
        for j, (side, tl) in enumerate((("seller", "seller opens: ask over floor"), ("buyer", "buyer opens: bid under ceiling"))):
            x0, y0, w, h = 110 + j * 470, 100 + i * 270, 220, 190
            Y = lambda v: y0 + h - (min(max(v, lo), hi) - lo) / (hi - lo) * h      # noqa: E731
            pairs = per_fixture(G[key], side)
            down = sum(b < a for a, b in pairs)
            s.text(x0 - 60, y0 - 14, f"{slab} · {tl}", 13, weight="bold")
            for t in range(0, 61, 20):
                s.line(x0, Y(t), x0 + w, Y(t), "#e6e6e6")
                s.text(x0 - 8, Y(t) + 4, f"{t}%", 11, "end", "#444")
            for a, b in pairs:
                c, dash = (VERM, "") if b < a else (BLUE, " stroke-dasharray='4 3'")
                s.add(f"<line x1='{x0:.1f}' y1='{Y(a):.1f}' x2='{x0 + w:.1f}' y2='{Y(b):.1f}' stroke='{c}'"
                      f" stroke-opacity='0.55' stroke-width='1.6'{dash}/>")
            ma, mb = st.fmean(a for a, _ in pairs), st.fmean(b for _, b in pairs)
            s.line(x0, Y(ma), x0 + w, Y(mb), INK, 3.2)
            s.text(x0, y0 + h + 18, "$100", 12, "middle", weight="bold")
            s.text(x0 + w, y0 + h + 18, "$100M", 12, "middle", weight="bold")
            s.text(x0 + w + 10, Y(mb) + 4, f"mean {ma:.0f}% → {mb:.0f}%", 11.5, weight="bold")
            s.text(x0 + w + 10, Y(mb) + 20, f"{down} of {len(pairs)} down", 11.5, color=VERM)
    s.add(f"<line x1='110' y1='618' x2='140' y2='618' stroke='{VERM}' stroke-width='2'/>")
    s.text(146, 622, "smaller at $100M (solid)", 12)
    s.add(f"<line x1='330' y1='618' x2='360' y2='618' stroke='{BLUE}' stroke-width='2' stroke-dasharray='4 3'/>")
    s.text(366, 622, "same or larger (dashed)", 12)
    s.line(550, 618, 580, 618, INK, 3.2)
    s.text(586, 622, "mean", 12)
    s.save(os.path.join(out, "E3_opening_per_fixture.svg"))


def fig_process(D, out):
    s = Svg(1100, 470, "The split stays flat while the bargaining process changes with scale",
            "Five small panels across $100, $100K, $100M for all four datasets: mean seller share is flat within each study;"
            " median concession step falls; offers per game, words per message and the share of messages saying final or"
            " best offer rise.")
    s.text(30, 32, "Bigger numbers change how they bargain, not how they split", 19, weight="bold")
    s.text(30, 54, "All four Qwen datasets. Step size is partly built in: $100 prices are whole dollars (a 1% grid). " + NOTE,
           12.5, color="#444")
    specs = (("share", "Seller share (outcome)", 0, 0.7, [0, 0.2, 0.4, 0.6], "{:.1f}"),
             ("step", "Median step, % of V", 0, 6, [0, 2, 4, 6], "{:.0f}%"),
             ("offers", "Offers per game", 4, 7, [4, 5, 6, 7], "{:.0f}"),
             ("words", "Words per message", 35, 50, [35, 40, 45, 50], "{:.0f}"),
             ("final", "Says 'final'/'best offer'", 0, 35, [0, 10, 20, 30], "{:.0f}%"))
    for i, (k, title, lo, hi, ticks, fmt) in enumerate(specs):
        series = [(key, D[key][k], c, p, sh) for key, _, _, c, p, sh in DATASETS]
        panel(s, 70 + i * 208, 100, 160, 250, lo, hi, ticks, series, title, fmt, end_labels=False)
    legend(s, 70, 420)
    s.save(os.path.join(out, "E4_process_vs_outcome.svg"))


def main():
    G = {k: load(run) for k, _, run, *_ in DATASETS}
    D = {k: stats(g) for k, g in G.items()}
    counts = {k: {side: (lambda p: f"{sum(b < a for a, b in p)} of {len(p)}")(per_fixture(G[k], side))
                  for side in ("seller", "buyer")} for k in ("S1", "S2")}
    out = os.path.join(ROOT, "derived", "explore", "figures")
    os.makedirs(out, exist_ok=True)
    fig_gap(D, out, counts)
    fig_edge(D, out)
    fig_fixtures(G, out)
    fig_process(D, out)
    for f in sorted(os.listdir(out)):
        print(os.path.relpath(os.path.join(out, f), ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
