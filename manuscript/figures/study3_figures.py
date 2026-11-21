#!/usr/bin/env python3
"""Five print figures for the Study 3-centred manuscript, offline, standard library only.

    python3 manuscript/figures/study3_figures.py            write s3_*.svg and s3_*.pdf (PDF via rsvg-convert)
    python3 manuscript/figures/study3_figures.py --check    verify sources and print sizes without writing

  s3_openings  opening gap from own limit at $100 / $100K / $100M, A0 vs A1, per model and opener
  s3_shift     closing shift d = s($100M) - s($100) per model and condition (registered P2)
  s3_clamp     opening-clamp decomposition in A0: d, omega, delta_m and kappa (registered P2/P3)
  s3_e5        Gemma E5 appraisal swap share by layer, sellers and buyers
  s3_e6        Gemma E6: notation transfer by layer, MLP neuron census, negotiation direction in other tasks

Sources (read only): derived/study3/study3.json (registered analysis); the six Study 3 session runs
data/runs/s3-*-s[12]-*/games.jsonl, loaded and scored with scripts/analyze_addendum_c.py (load,
opening_gap, share), as scripts/analyze_study3.py does; and the E5 and E6 run summary.json files. Opening gaps are
averaged per (fixture, opener, cell) over that cell's games, then across the 40 fixtures; 95% CIs by a
seeded fixture bootstrap (4,000 resamples). Registered values are re-derived from the games where the
figure needs them and checked against study3.json. Final width 6.75 in; labels >= 17 units (8.3 pt).
"""
import argparse
import glob
import json
import os
import random
import statistics
import subprocess
import sys
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from figstyle import BAND, GREY, LIGHT, MODEL, ONYX, ORANGE, ORANGE_DARK, ROOT, SLATE, line, svg_doc, write  # noqa: E402
import addendum_c as C          # noqa: E402  (signed, close)
import fig8_models as O         # noqa: E402  (marker)
import main_figures as M        # noqa: E402  (label, point, cross, axis, font_check)

sys.path.insert(1, os.path.join(ROOT, "scripts"))
import analyze as A             # noqa: E402  (bootstrap_ci, seed)
import analyze_addendum_c as AC  # noqa: E402  (load, opening_gap, share)

label, point, axis = M.label, M.point, M.axis
ORDER = ("gemma-4-31b-it", "qwen3.8-27b-fp8", "gpt-6-luna", "claude-haiku-5.5", "gemini-3.5-flash-lite")
SCALES = (("x1e2", "$100"), ("x1e5", "$100K"), ("x1e8", "$100M"))
OPENERS = ("seller", "buyer")
BOOTS = 4000
K_CELL = 4                       # games per (model, fixture, opener, cell), sessions pooled
A1_COL = GREY                    # A0: model colour, solid, filled marker; A1: grey, dashed, open marker
DASH = "7 5"
E5_RUN = "interp-e5-gemma-4-31b-it-20261015T232511Z"
E6_RUN = "interp-e6-gemma-4-31b-it-20261016T013146Z"
STUDY3 = os.path.join(ROOT, "derived", "study3", "study3.json")


def fx(v):
    return f"{v:+.2f}".replace("-", "−")


def polyline(pts, col, w=2.5, dash=None):
    d = f" stroke-dasharray='{dash}'" if dash else ""
    return ("<polyline points='" + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts) +
            f"' fill='none' stroke='{col}' stroke-width='{w}' stroke-linejoin='round'{d}/>")


def rect(x0, y0, x1, y1, fill):
    return f"<rect x='{x0:.1f}' y='{y0:.1f}' width='{x1-x0:.1f}' height='{y1-y0:.1f}' fill='{fill}'/>"


def mark(arm, cond, x, y, r=5.5):
    """A0: the model's filled marker; A1: the same shape, open, grey."""
    if cond == "A0":
        return point(arm, x, y, r)
    return O.marker(MODEL[arm].get("marker", "circle"), x, y, r, A1_COL, "white", 2)


# ---------------------------------------------------------------- data

def load():
    with open(STUDY3) as fh:
        s3 = json.load(fh)
    assert s3["registered"] and s3["schema"] == "study3/1"
    assert set(s3["per_model"]) == set(ORDER)
    runs = sorted(os.path.basename(d) for d in glob.glob(os.path.join(ROOT, "data", "runs", "s3-*-s[12]-*")))
    assert runs == sorted(s3["runs"]) and len(runs) == 6, runs
    games = {}
    for run in runs:
        man, scored, _, missing = AC.load(os.path.join(ROOT, "data", "runs", run))
        assert man["study"] == 3 and man["kind"] == "study3_session" and not missing, run
        assert not set(scored) & set(games)
        games.update(scored)
    fixtures = sorted(s3["per_model"][ORDER[0]]["P1"]["I_f"])
    assert len(fixtures) == 40
    cells = {}
    for r in games.values():
        if r["condition"] in ("A0", "A1"):
            assert r["cell"] == f"{r['condition']}-{r['scale']}"
            cells.setdefault((r["arm"], r["condition"], r["opener"], r["scale"], r["fixture"]), []).append(r)
    for arm in ORDER:
        assert s3["per_model"][arm]["games_scored"] == sum(r["arm"] == arm for r in games.values()) == 3200
    assert {len(v) for v in cells.values()} == {K_CELL} and len(cells) == 5*2*2*3*40
    return s3, cells, fixtures


def cell_mean(rows, value):
    v = [x for x in map(value, rows) if x is not None]
    return sum(v, Fraction(0)) / len(v) if v else None


def openings(s3, cells, fixtures):
    """{(arm, cond, opener, scale): (mean, ci95, n)} of the fixture-mean opening gap; checks h and I_f."""
    out, undefined = {}, 0
    for arm in ORDER:
        per = {}
        for cond in ("A0", "A1"):
            for op in OPENERS:
                for scale, _ in SCALES:
                    vals = {}
                    for f in fixtures:
                        rows = cells.get((arm, cond, op, scale, f), [])
                        undefined += sum(AC.opening_gap(r) is None for r in rows)
                        m = cell_mean(rows, AC.opening_gap)
                        if m is not None:
                            vals[f] = m
                    per[cond, op, scale] = vals
                    v = [float(x) for x in vals.values()]
                    ci = A.bootstrap_ci(v, random.Random(A.seed(f"study3-figure-opening-{cond}-{op}-{scale}", arm)),
                                        levels=(0.95,), boots=BOOTS)["95"]
                    out[arm, cond, op, scale] = (statistics.fmean(v), ci, len(v))
        # Check against the registered h (S1) and I_f (P1): h_f = mean over openers of [gap($100M) - gap($100)].
        h = {}
        for cond in ("A0", "A1"):
            h[cond] = {f: sum(per[cond, op, "x1e8"][f] - per[cond, op, "x1e2"][f] for op in OPENERS) / 2
                       for f in fixtures}
        sec = s3["per_model"][arm]["secondary"]
        C.close(statistics.fmean(float(x) for x in h["A0"].values()), sec["S1_HA_A0"]["mean"], arm + " h(A0)")
        C.close(statistics.fmean(float(x) for x in h["A1"].values()), sec["S1_h_A1_equivalence"]["mean"], arm + " h(A1)")
        for f, i in s3["per_model"][arm]["P1"]["I_f"].items():
            C.close(float(h["A1"][f] - h["A0"][f]), i, f"{arm} I_{f}")
    return out, undefined


def shifts(s3, cells, fixtures):
    """Registered P2 per model and condition, d_i re-derived from the games (agreements' seller share)."""
    out = {}
    for arm in ORDER:
        for cond in ("A0", "A1"):
            p = s3["per_model"][arm]["P2"][cond]
            assert p["n_fixtures"] == 40 and not p["fixtures_left_out"]
            for f in fixtures:
                parts = [cell_mean(cells[arm, cond, op, "x1e8", f], AC.share) -
                         cell_mean(cells[arm, cond, op, "x1e2", f], AC.share) for op in OPENERS]
                C.close(float(sum(parts) / 2), p["d_i"][f], f"{arm} {cond} d_{f}")
            C.close(statistics.fmean(p["d_i"].values()), p["mean"], f"{arm} {cond} mean d")
            out[arm, cond] = p
    return out


def e5():
    with open(os.path.join(ROOT, "data", "interp", E5_RUN, "summary.json")) as fh:
        s = json.load(fh)
    series = {}
    for role in ("seller", "buyer"):
        for key, direction in (("N6", "add"), ("P6", "remove")):
            cells = s["roles"][role]["swap"][key]
            pts = sorted((int(k.split("@")[1]), v["share"]) for k, v in cells.items() if k.startswith("cure@"))
            assert [l for l, _ in pts] == s["layers"]["cure"]
            series[role, direction] = pts
    printed = {   # the run's printed summary, checked to the percent
        ("seller", "add"): {30: -13, 36: -30, 38: -35, 39: -27, 40: -23, 41: 75, 42: 83, 44: 88, 48: 99, 52: 104, 58: 111},
        ("seller", "remove"): {30: -15, 36: -59, 38: -61, 39: -40, 40: -27, 41: 78, 42: 92, 44: 89, 48: 102, 52: 113, 58: 113},
        ("buyer", "add"): {30: 18, 36: 38, 38: 40, 39: 40, 40: 37, 41: 89, 42: 89, 44: 87, 48: 90, 52: 98, 58: 99},
        ("buyer", "remove"): {30: 13, 36: 42, 38: 49, 39: 65, 40: 63, 41: 79, 42: 82, 44: 82, 48: 84, 52: 95, 58: 100},
    }
    for k, ref in printed.items():
        got = dict(series[k])
        for layer, pct in ref.items():
            assert round(got[layer] * 100) == pct, (k, layer, got[layer], pct)
    add = s["roles"]["seller"]["swap"]["N6"]
    split = {"all": add["cure@44"]["share"], "reply": add["cure reply@44"]["share"], "S": add["cure S@44"]["share"]}
    assert (round(split["reply"] * 100), round(split["S"] * 100)) == (57, 22)
    n = {role: s["roles"][role]["swap"]["N6"]["cure@44"]["n"] for role in ("seller", "buyer")}
    return series, split, n, s["layers"]["n_layers"]


# ---------------------------------------------------------------- figures

def y_axis(out, x, Y, ticks, x1, fmt=lambda v: f"{v:g}", labels=True, ref=()):
    for v in ticks:
        out.append(line(x, Y(v), x1, Y(v), GREY if v in ref else LIGHT, 1.2 if v in ref else 1))
        if labels:
            out.append(label(x-8, Y(v)+6, fmt(v).replace("-", "−"), anchor="end"))


def openings_figure(op):
    hi = 0.9
    assert all(0 <= v[1][0] and v[1][1] <= hi for v in op.values())
    out = [label(20, 28, "Opening gap from own limit, by scale", 20, weight="bold"),
           line(300, 47, 336, 47, ORANGE_DARK, 2.5), point("gemma-4-31b-it", 318, 47, 5),
           label(346, 53, "A0: no appraisal", 17),
           line(510, 47, 546, 47, A1_COL, 2.5, DASH), mark("gemma-4-31b-it", "A1", 528, 47, 5),
           label(556, 53, "A1: appraisal shown", 17),
           label(980, 53, "mean of 40 fixtures, 95% CI", 17, fill=GREY, anchor="end")]
    panels = ((290, 610, "seller", "Seller opens"), (660, 980, "buyer", "Buyer opens"))
    for x0, x1, _, head in panels:
        out.append(label((x0+x1)/2, 88, head, 18, weight="bold", anchor="middle"))
    top0, ph, step = 104, 95, 124
    for i, arm in enumerate(ORDER):
        top = top0 + i*step
        bot = top + ph
        Y = lambda v, t=top: t + ph - v/hi*ph
        out.append(label(20, top+ph/2+6, MODEL[arm]["label"], 18, fill=MODEL[arm]["col"], weight="bold"))
        for j, (x0, x1, opener, _) in enumerate(panels):
            y_axis(out, x0, Y, (0, 0.4, 0.8), x1, labels=j == 0, ref=(0,))
            out.append(line(x0, top, x0, bot, ONYX, 1.2))
            X = [x0+45, (x0+x1)/2, x1-45]
            for cond, dx in (("A0", -5), ("A1", 5)):
                col = MODEL[arm]["col"] if cond == "A0" else A1_COL
                pts = [(X[k]+dx, Y(op[arm, cond, opener, s][0])) for k, (s, _) in enumerate(SCALES)]
                for k, (s, _) in enumerate(SCALES):
                    lo, up = op[arm, cond, opener, s][1]
                    x = pts[k][0]
                    out += [line(x, Y(lo), x, Y(up), col, 1.6), line(x-4, Y(lo), x+4, Y(lo), col, 1.6),
                            line(x-4, Y(up), x+4, Y(up), col, 1.6)]
                out.append(polyline(pts, col, 2.5, None if cond == "A0" else DASH))
                out += [mark(arm, cond, x, y, 4.5) for x, y in pts]
            if i == len(ORDER) - 1:
                for k, (_, name) in enumerate(SCALES):
                    out += [line(X[k], bot, X[k], bot+5, ONYX, 1), label(X[k], bot+25, name, anchor="middle")]
    last = top0 + (len(ORDER)-1)*step + ph
    hw = [(v[1][1]-v[1][0])/2 for v in op.values()]
    out += [label(20, last+60, "Gap = first offer / own limit − 1 (seller opens) or 1 − first offer / own limit "
                  "(buyer opens), shared y-axis.", 17, fill=GREY),
            label(20, last+84, f"Each point: mean over 40 fixtures of the {K_CELL} games per fixture and cell, both "
                  "sessions pooled. 95% CIs are mostly", 17, fill=GREY),
            label(20, last+108, "narrower than the markers (half-width median "
                  f"{statistics.median(hw):.3f}, max {max(hw):.3f}); A0 and A1 offset horizontally.", 17, fill=GREY)]
    return svg_doc(1000, last+124, "Study 3 opening gaps by scale, appraisal and opener",
                   "Source: the six Study 3 session runs (derived/study3/study3.json runs), games.jsonl, A0 and A1 cells; "
                   "per-game gap from scripts/analyze_addendum_c.py opening_gap, averaged per fixture and cell then across "
                   "40 fixtures; 95% CI by seeded fixture bootstrap, 4,000 resamples. Per-fixture h(A0), h(A1) and P1 I_f "
                   "re-derived and checked against study3.json.", out)


def verdict(p):
    reading = {"invariant to within +-0.10": "equivalent", "a real but small shift (below 0.10)": "small shift",
               "scale shifts this model's split": "shift"}[p["reading"]]
    assert (reading != "shift") == p["H2_equivalent_within_0.10"]
    assert (reading == "equivalent") == (p["p_holm_across_models"] >= 0.05)
    return reading


def shift_figure(sh):
    lo, hi = -0.75, 0.35
    vals = [v for p in sh.values() for v in p["d_i"].values()]
    assert lo < min(vals) and max(vals) < hi
    left, right = 270, 820
    top, rows, gap = 92, 36, 22
    ys = {}
    for i, arm in enumerate(ORDER):
        for j, cond in enumerate(("A0", "A1")):
            ys[arm, cond] = top + 18 + i*(2*rows+gap) + j*rows
    bot = max(ys.values()) + 20
    X = lambda v: left + (v-lo)/(hi-lo)*(right-left)
    out = [rect(X(-0.1), top, X(0.1), bot, BAND),
           label(20, 28, "Closing shift: seller share at $100M − $100", 20, weight="bold"),
           label(20, 54, "Dots: 40 fixtures. Thick: 90% CI; thin: 95% CI. Band: ±0.10 equivalence margin (H2).", 17, fill=GREY),
           label(852, 82, "mean", 17, fill=GREY), label(990, 82, "H1 / H2", 17, fill=GREY, anchor="end")]
    axis(out, left, right, bot, lo, hi, (-0.6, -0.4, -0.2, 0, 0.2), top)
    out.append(line(X(0), top, X(0), bot, GREY, 1.1, "4 3"))
    for i, arm in enumerate(ORDER):
        if i:
            yb = ys[arm, "A0"] - 18 - gap/2
            out.append(line(20, yb, 990, yb, LIGHT, 1))
        out.append(label(20, ys[arm, "A0"]+24, MODEL[arm]["label"], 18, fill=MODEL[arm]["col"], weight="bold"))
        for cond in ("A0", "A1"):
            p, y = sh[arm, cond], ys[arm, cond]
            col = MODEL[arm]["col"] if cond == "A0" else A1_COL
            rng = random.Random(A.seed(f"study3-figure-jitter-{cond}", arm))
            out.append(label(left-12, y+6, cond, 17, anchor="end"))
            for f in sorted(p["d_i"]):
                out.append(O.marker("circle", X(p["d_i"][f]), y + rng.uniform(-10, 10), 2.8,
                                    MODEL[arm]["col"], MODEL[arm].get("fill", MODEL[arm]["col"]), 1, 0.3))
            out += [line(X(p["ci95"][0]), y, X(p["ci95"][1]), y, col, 2),
                    line(X(p["ci90"][0]), y, X(p["ci90"][1]), y, col, 6), mark(arm, cond, X(p["mean"]), y, 6),
                    label(852, y+6, C.signed(p["mean"]), 17, fill=col, weight="bold"),
                    label(990, y+6, verdict(p), 17, anchor="end")]
    out += [label((left+right)/2, bot+52, "d = mean over openers of [seller share at $100M − at $100] (surplus units)",
                  17, anchor="middle"),
            label(20, bot+84, "A0: no appraisal (model colour, filled); A1: appraisal shown (grey, open). Equivalent: 90% CI "
                  "inside ±0.10;", 17, fill=GREY),
            label(20, bot+108, "small shift: equivalent but H1 rejects zero (Holm). Agreements only; 40 fixtures per row.",
                  17, fill=GREY)]
    return svg_doc(1000, bot+124, "Study 3 registered closing shift (P2) by model and appraisal condition",
                   "Source: derived/study3/study3.json per_model.P2.A0 and .A1 (mean, ci90, ci95, d_i, reading); every "
                   "d_i re-derived from the six session runs' games and checked. Fixture dots jittered vertically.", out)


def clamp_figure(s3, sh):
    lo, hi = -0.4, 0.1
    klo, khi = -0.1, 0.1
    left, right, kl, kr = 270, 600, 680, 920
    top, sub, gap = 96, 24, 26
    out = [label(20, 28, "Where the A0 shift enters: the opening clamp", 20, weight="bold"),
           label(left, 62, "a  Natural shift and opening channel", 18, weight="bold"),
           label(kl-20, 62, "b  Concession channel κ", 18, weight="bold")]
    g = [top + 10 + i*(3*sub+gap) for i in range(len(ORDER))]
    bot = g[-1] + 2*sub + 18
    X = lambda v: left + (v-lo)/(hi-lo)*(right-left)
    K = lambda v: kl + (v-klo)/(khi-klo)*(kr-kl)
    out.insert(0, rect(K(-0.05), top, K(0.05), bot, BAND))
    axis(out, left, right, bot, lo, hi, (-0.4, -0.3, -0.2, -0.1, 0, 0.1), top)
    axis(out, kl, kr, bot, klo, khi, (-0.1, -0.05, 0, 0.05, 0.1), top)
    for Z in (X, K):
        out.append(line(Z(0), top, Z(0), bot, GREY, 1.1, "4 3"))
    for i, arm in enumerate(ORDER):
        p3, d = s3["per_model"][arm]["P3"], sh[arm, "A0"]
        k = p3["P3a_kappa"]
        assert p3["P3b_omega_minus_delta_m"]["equivalent_within_0.05"] and k["n_fixtures"] == 40
        col, y = MODEL[arm]["col"], g[i]
        if i:
            out.append(line(20, y-gap/2-6, 990, y-gap/2-6, LIGHT, 1))
        out.append(label(20, y+sub+6, MODEL[arm]["label"], 18, fill=col, weight="bold"))
        for r, sym in enumerate(("d", "ω", "Δm")):
            out.append(label(left-12, y+r*sub+6, sym, 17, anchor="end", italic=r < 2))
        out += [line(X(d["ci90"][0]), y, X(d["ci90"][1]), y, col, 5), point(arm, X(d["mean"]), y),
                point(arm, X(p3["omega_mean"]), y+sub, hollow=True), M.cross(X(p3["delta_m_mean"]), y+2*sub, GREY, 5),
                line(K(k["ci90"][0]), y+sub, K(k["ci90"][1]), y+sub, col, 5), point(arm, K(k["mean"]), y+sub),
                label(990, y+sub+6, C.signed(k["mean"]), 17, fill=col, weight="bold", anchor="end")]
    not_eq = [MODEL[a]["label"] for a in ORDER if not s3["per_model"][a]["P3"]["P3a_kappa"]["equivalent_within_0.05"]]
    out += [label((left+right)/2, bot+52, "change from $100 to $100M (surplus units)", 17, anchor="middle"),
            label((kl+kr)/2, bot+52, "band: ±0.05 (P3a)", 17, anchor="middle"),
            label(20, bot+84, "d: closing shift in the A0 games (P2), 90% CI.  ω (open marker): opening channel, both "
                  "openings clamped,", 17, fill=GREY),
            label(20, bot+108, "opening scale moved.  +: scripted midpoint shift Δm.  κ: concession channel, opening fixed, "
                  "game scale moved;", 17, fill=GREY),
            label(20, bot+132, "90% CI; κ's CI leaves ±0.05 only for " + ", ".join(not_eq) +
                  ".  ω − Δm within ±0.05 on all five (P3b).  40 fixtures per model.", 17, fill=GREY)]
    return svg_doc(1000, bot+148, "Study 3 opening-clamp decomposition in A0 (registered P2 and P3)",
                   "Source: derived/study3/study3.json per_model.P2.A0 (mean, ci90), P3.omega_mean, P3.delta_m_mean, "
                   "P3.P3a_kappa (mean, ci90) and P3.P3b_omega_minus_delta_m.equivalent_within_0.05. omega and delta_m "
                   "have no separate CI in the analysis; their difference is P3b.", out)


def e5_figure(series, split, n, n_layers):
    lo, hi = -0.75, 1.25
    vals = [v for pts in series.values() for _, v in pts]
    assert lo < min(vals) and max(vals) < hi
    top, ph = 96, 300
    bot = top + ph
    Y = lambda v: bot - (v-lo)/(hi-lo)*ph
    out = [label(20, 28, "Gemma: where the appraisal enters the opening", 20, weight="bold"),
           line(520, 47, 556, 47, SLATE, 2.5), label(566, 53, "adding (A1 into A0)", 17),
           line(760, 47, 796, 47, ORANGE_DARK, 2.5, DASH), label(806, 53, "removing (A0 into A1)", 17)]
    for left, right, role, head in ((95, 465, "seller", "a  Sellers"), (605, 975, "buyer", "b  Buyers")):
        X = lambda layer, l=left, r=right: l + layer/60*(r-l)
        out += [rect(X(40.5), top, X(42.5), bot, BAND),
                label(left, 84, f"{head} (n = {n[role]} fixtures)", 18, weight="bold"),
                label(X(41.5), top+20, "L41–42", 17, fill=GREY, anchor="middle")]
        y_axis(out, left, Y, (-0.5, 0, 0.5, 1.0), right, lambda v: f"{v:.0%}", ref=(0, 1.0))
        axis(out, left, right, bot, 0, 60, (0, 10, 20, 30, 40, 50, 60))
        out.append(line(left, top, left, bot, ONYX, 1.2))
        for direction, col, dash, fill in (("add", SLATE, None, SLATE), ("remove", ORANGE_DARK, DASH, "white")):
            pts = [(X(l), Y(v)) for l, v in series[role, direction]]
            assert max(l for l, _ in series[role, direction]) < n_layers
            out.append(polyline(pts, col, 2.5, dash))
            out += [O.marker("circle", x, y, 2.6, col, fill, 1.4) for x, y in pts]
        out.append(label((left+right)/2, bot+52, f"layer (of {n_layers})", 17, anchor="middle"))
    out.append(label(20, top+ph/2, "share of the log-odds distance", 17, anchor="middle", rotate=-90))
    # L44 token split, sellers adding.
    y0, bx, bw = bot+130, 290, 300
    out.append(label(20, y0-30, "c  Which tokens carry it: sellers, adding, layer 44", 18, weight="bold"))
    for k, (key, name) in enumerate((("all", "prompt S + reply"), ("reply", "reply tokens only"), ("S", "prompt S tokens only"))):
        y = y0 + k*26
        out += [label(bx-12, y+6, name, 17, anchor="end"), rect(bx, y-8, bx+split[key]*bw, y+8, SLATE if k else GREY),
                label(bx+split[key]*bw+8, y+6, f"{split[key]:.0%}", 17, weight="bold" if k else "normal")]
    out += [label(20, y0+100, "Swap: the other run's states on the prompt's S tokens and the reply's opening tokens, one "
                  "layer at a time, at ×10⁶;", 17, fill=GREY),
            label(20, y0+124, "share of the log-odds distance between the A0 and A1 runs. Adding: A1 states into A0; "
                  "removing: A0 into A1.", 17, fill=GREY)]
    return svg_doc(1000, y0+140, "Gemma E5 appraisal swap by layer, sellers and buyers",
                   f"Source: data/interp/{E5_RUN}/summary.json roles.<seller|buyer>.swap.N6 (adding: A1 x1e6 states into "
                   "the A0 x1e6 run) and .P6 (removing: A0 into A1), keys cure@<layer>.share; token split from "
                   "roles.seller.swap.N6 'cure reply@44' and 'cure S@44'. Printed run summary values checked to the percent.",
                   out)


def e6():
    with open(os.path.join(ROOT, "data", "interp", E6_RUN, "summary.json")) as fh:
        s = json.load(fh)
    assert s["build"]["usable"] == 19 and s["limit"] == 0 and s["checks"]["hooks_fired_once"]
    tr = s["fit"]["transfer_from_digits"]
    assert all(len(v) == s["layers"]["n_layers"] for v in tr.values())
    # the run's printed summary, checked to two decimals
    for k, ref in {"words": {1: 0.43, 2: 0.45, 5: 0.96, 20: 0.97}, "short": {1: -0.04, 2: -0.00, 5: 0.89, 20: 0.96},
                   "count": {0: 0.94}}.items():
        for layer, r in ref.items():
            assert abs(round(tr[k][layer], 2) - r) < 1e-9, (k, layer, tr[k][layer])
    census = {int(L): v for L, v in s["census"]["layers"].items()}
    assert census[20]["n_dial"] == 1640 and census[4]["n_digit_only"] == 1766
    fam = s["families"]
    steer = {f: {v: fam[f]["share"]["0"][f"{v}@42x+2"] for v in ("vR", "u", "rand")} for f in ("nego", "risk", "percent")}
    pred = {p["id"]: p for p in s["predictions"]}
    assert (round(steer["nego"]["vR"], 3), round(steer["nego"]["rand"], 3)) == tuple(pred["E6-1"]["value"].values())
    assert round(steer["risk"]["vR"], 3) == pred["E6-6"]["value"]["vR"]
    assert round(steer["percent"]["vR"], 3) == pred["E6-7"]["value"]["vR"]
    fermi = {v: fam["fermi"]["delta"]["None"][f"{v}@42x+2"] for v in ("vR", "u")}
    gaps = {f: fam[f]["natural_gap"] for f in steer}
    return tr, census, steer, fermi, gaps, s["fit"]["cos_digits_vR"]["42"]


def e6_figure(tr, census, steer, fermi, gaps, cos42):
    out = [label(20, 28, "Gemma: magnitude as a number, and the negotiation direction elsewhere", 20, weight="bold")]
    # (a) cross-notation transfer of the digits-fitted magnitude direction, by layer
    top, ph, left, right = 96, 250, 95, 465
    bot = top + ph
    lo, hi = -0.2, 1.0
    Y = lambda v: bot - (v-lo)/(hi-lo)*ph
    X = lambda layer: left + layer/60*(right-left)
    out += [rect(X(0.5), top, X(2.5), bot, BAND), label(X(1.5)+6, bot-10, "L1–2", 17, fill=GREY),
            label(left, 84, "a  Digits-fitted direction read in other notations", 18, weight="bold")]
    y_axis(out, left, Y, (0, 0.5, 1.0), right, ref=(0,))
    axis(out, left, right, bot, 0, 60, (0, 10, 20, 30, 40, 50, 60))
    out.append(line(left, top, left, bot, ONYX, 1.2))
    series = (("words", SLATE, None, "“3.7 thousand dollars”"), ("short", ORANGE_DARK, DASH, "“$3.7K”"),
              ("count", GREY, "2 3", "“3700 people”"))
    for k, (key, col, dash, name) in enumerate(series):
        out.append(polyline([(X(i), Y(r)) for i, r in enumerate(tr[key])], col, 2.2, dash))
        out += [line(right-150, top+150+k*24, right-118, top+150+k*24, col, 2.2, dash),
                label(right-110, top+156+k*24, name, 17)]
    out += [label((left+right)/2, bot+52, "layer (of 60)", 17, anchor="middle"),
            label(20, top+ph/2, "r with log10 amount", 17, anchor="middle", rotate=-90)]
    # (b) MLP neuron census
    bl, br, ymax = 640, 975, 2600
    B = lambda v: bot - v/ymax*ph
    out += [label(bl-35, 84, "b  MLP neurons per layer (of 21,504)", 18, weight="bold")]
    y_axis(out, bl, B, (0, 500, 1000, 1500, 2000, 2500), br, fmt=lambda v: f"{v:,.0f}", ref=(0,))
    out.append(line(bl, bot, br, bot, ONYX, 1.2))
    layers = sorted(census)
    step = (br-bl)/len(layers)
    for i, L in enumerate(layers):
        c, x = census[L], bl + step*i + step/2
        out += [rect(x-26, B(c["n_dial"]), x-2, bot, SLATE),
                rect(x-26, B(c["n_dial_count_too"]), x-2, bot, ORANGE),
                rect(x+2, B(c["n_digit_only"]), x+26, bot, LIGHT),
                f"<rect x='{x+2:.1f}' y='{B(c['n_digit_only']):.1f}' width='24' height='{bot-B(c['n_digit_only']):.1f}' "
                f"fill='none' stroke='{GREY}' stroke-width='1.2'/>",
                line(x, bot, x, bot+5, ONYX, 1), label(x, bot+25, f"L{L}", anchor="middle")]
    out += [rect(bl+10, top+2, bl+28, top+16, SLATE), label(bl+34, top+15, "dial in all 4 money notations", 17),
            rect(bl+10, top+26, bl+28, top+40, ORANGE), label(bl+34, top+39, "… and in people counts", 17),
            rect(bl+10, top+50, bl+28, top+64, LIGHT), label(bl+34, top+63, "digits only (not words)", 17)]
    # (c) the negotiation scale direction (vR, L42, alpha +2) steered into other tasks on $100 prompts
    ct, rh = bot + 130, 34
    cl, cr, clo, chi = 330, 900, -0.2, 1.0
    C_ = lambda v: cl + (v-clo)/(chi-clo)*(cr-cl)
    rows = (("nego", "Negotiation openings", f"{abs(gaps['nego']):.2f} (opening ÷ floor)"),
            ("risk", "Risk choices", f"{gaps['risk']:.1f} nats"), ("percent", "Percentage answers", f"{gaps['percent']:.1f} points"))
    cb = ct + len(rows)*(2*rh+14)
    out += [label(20, ct-40, "c  Share of the natural $100 → $100M gap moved by the negotiation direction (L42, α = +2)",
                  18, weight="bold"), rect(C_(-0.1), ct-8, C_(0.1), cb-14, BAND)]
    axis(out, cl, cr, cb-14, clo, chi, (-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0), ct-8)
    out.append(line(C_(0), ct-8, C_(0), cb-14, GREY, 1.1, "4 3"))
    for i, (f, name, gap) in enumerate(rows):
        y = ct + i*(2*rh+14)
        out += [label(cl-14, y+18, name, 17, anchor="end", weight="bold"), label(cl-14, y+40, f"gap {gap}", 17,
                                                                                 fill=GREY, anchor="end")]
        for j, (v, col, nm) in enumerate((("vR", SLATE, "negotiation direction"), ("u", GREY, "non-money control"))):
            yy = y + j*rh
            x0, x1 = sorted((C_(0), C_(steer[f][v])))
            out += [rect(x0, yy+4, x1, yy+24, col), label(max(C_(0), C_(steer[f][v]))+8, yy+20,
                                                          f"{steer[f][v]:+.0%}".replace("-", "−"), 17)]
    ly = ct + 2*(2*rh+14) + 4
    out += [rect(C_(0.45), ly+4, C_(0.45)+24, ly+20, SLATE), label(C_(0.45)+32, ly+18, "negotiation direction (vR)", 17),
            rect(C_(0.45), ly+38, C_(0.45)+24, ly+54, GREY), label(C_(0.45)+32, ly+52, "non-money control (u)", 17)]
    out += [label((cl+cr)/2, cb+38, "share of the gap (band ±10%)", 17, anchor="middle"),
            label(20, cb+76, f"Random vector of the same norm: {steer['nego']['rand']:.0%} on negotiation. It breaks the "
                  "short prompts (risk and comparison: mean option", 17, fill=GREY),
            label(20, cb+100, f"mass 0.006; percentage answers fail to parse), so it is not shown.  Fermi estimates: "
                  f"+{fermi['vR']:.2f} log10 (vR), +{fermi['u']:.2f} (u).", 17, fill=GREY),
            label(20, cb+124, f"cos(digits magnitude direction, negotiation direction) at L42: {cos42:.2f} (chance ≈ "
                  "0.014).  Exploratory.", 17, fill=GREY)]
    return svg_doc(1000, cb+140, "Gemma E6: magnitude as a number, and the negotiation scale direction in other tasks",
                   f"Source: data/interp/{E6_RUN}/summary.json fit.transfer_from_digits (panel a), census.layers "
                   "(n_dial, n_dial_count_too, n_digit_only; panel b), families.<nego|risk|percent>.share['0']"
                   "['<vR|u|rand>@42x+2'] and natural_gap (panel c), families.fermi.delta, fit.cos_digits_vR. "
                   "Values checked against the printed run summary and the predictions block.", out)


def build():
    s3, cells, fixtures = load()
    op, undefined = openings(s3, cells, fixtures)
    assert undefined == 0
    sh = shifts(s3, cells, fixtures)
    series, split, n, n_layers = e5()
    docs = {"s3_openings": openings_figure(op), "s3_shift": shift_figure(sh),
            "s3_clamp": clamp_figure(s3, sh), "s3_e5": e5_figure(series, split, n, n_layers),
            "s3_e6": e6_figure(*e6())}
    for stem, doc in docs.items():
        print(f"{stem}: minimum label {M.font_check(doc, 6.75, 7)} pt at 6.75 in")
    return docs, s3, op, sh, split


def spot(s3, op, sh, split):
    for arm in ORDER:
        p3 = s3["per_model"][arm]["P3"]
        print(f"{MODEL[arm]['label']:22s} P2 A0 {fx(sh[arm, 'A0']['mean'])} A1 {fx(sh[arm, 'A1']['mean'])}  "
              f"kappa {C.signed(p3['P3a_kappa']['mean'])} omega {C.signed(p3['omega_mean'])} "
              f"delta_m {C.signed(p3['delta_m_mean'])}")
        for cond in ("A0", "A1"):
            print("    seller", cond, " → ".join(f"{op[arm, cond, 'seller', s][0]:.2f}" for s, _ in SCALES),
                  "| buyer", " → ".join(f"{op[arm, cond, 'buyer', s][0]:.2f}" for s, _ in SCALES))
    print("E5 L44 sellers adding:", {k: round(v, 3) for k, v in split.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="verify sources and print sizes without writing figures")
    args = ap.parse_args()
    docs, s3, op, sh, split = build()
    spot(s3, op, sh, split)
    if not args.check:
        for stem, doc in docs.items():
            svg = write(os.path.join(HERE, stem+".svg"), doc)
            subprocess.run(["rsvg-convert", "-f", "pdf", "-o", os.path.join(HERE, stem+".pdf"), svg], check=True)
            print(svg)


if __name__ == "__main__":
    main()
