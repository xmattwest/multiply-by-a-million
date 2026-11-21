#!/usr/bin/env python3
"""Build the two Addendum C manuscript figures, offline.

    python3 manuscript/figures/addendum_c.py

Writes fig9_addendum_c.svg (HA and H1/H2) and fig10_addendum_c_midpoint.svg
(HM1 and HM2). Uses the registered analysis without changing it. Reconstructs
the plotted fixture quantities and CIs from the frozen helpers and raw games,
and checks them against derived/addC-20261011T182548Z/addendum_C.json.
The midpoint scatter uses the same eligible games for both axes; the primary
split uses all agreements. No extra test or significance correction is added.
PDFs: rsvg-convert -f pdf -o <stem>.pdf <stem>.svg
"""
import hashlib
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from figstyle import BAND, GREY, LIGHT, MODEL, ONYX, ROOT, line, svg_doc, text, write  # noqa: E402
from fig8_models import marker  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "scripts"))
import analyze_addendum_c as AC  # noqa: E402

RUN = "addC-20261011T182548Z"
ARMS = ("gpt-6-luna", "claude-haiku-5.5", "gemini-3.5-flash-lite")
FREEZE = "e54f00e0f6f8dd1dff5418270fe53d905ddb515e"


def close(a, b, label):
    if not math.isclose(a, b, rel_tol=0, abs_tol=1e-9):
        raise AssertionError(f"{label}: raw {a!r} != analysis {b!r}")


def check_summary(values, arm, what, reference, levels=(0.95, 0.90)):
    actual = AC.summary(values, arm, what, levels=levels)
    assert actual["n_fixtures"] == reference["n_fixtures"]
    assert actual["below_0"] == reference["below_0"]
    close(actual["mean"], reference["mean"], f"{arm} {what} mean")
    for key in ("ci90", "ci95"):
        if actual[key] is None:
            assert reference[key] is None
        else:
            for x, y in zip(actual[key], reference[key]):
                close(x, y, f"{arm} {what} {key}")


def load():
    run_dir = os.path.join(ROOT, "data", "runs", RUN)
    with open(os.path.join(ROOT, "derived", RUN, "addendum_C.json")) as fh:
        result = json.load(fh)
    manifest, scored, voids, missing = AC.load(run_dir)
    assert manifest["registered"] and not manifest["preview"]
    assert manifest["git"]["head"] == FREEZE
    assert result["registered"] and result["run_id"] == RUN
    assert set(result["holm_family"]) == set(ARMS)
    assert len(scored) == 2160 and not voids and not missing
    assert result["h2_margin"] == 0.10 and result["hm_margin"] == 0.05
    for name, expected in result["inputs_sha256"].items():
        with open(os.path.join(run_dir, name), "rb") as fh:
            assert hashlib.file_digest(fh, "sha256").hexdigest() == expected, name
    parts = {}
    for arm in ARMS:
        p = result["per_model"][arm]
        games = [g for g in scored.values() if g["arm"] == arm]
        assert len(games) == p["games_scored"] == 720
        fixtures = sorted({g["fixture"] for g in games})
        cells = AC.cells_of(games)
        d = AC.paired(cells, fixtures, AC.share)
        h = AC.paired(cells, fixtures, AC.opening_gap)
        parts[arm] = AC.hm_parts(cells, fixtures)
        e = {f: ds - ms for f, (ds, ms) in parts[arm].items()}
        check_summary(d.values(), arm, "H1", p["H1"])
        check_summary(h.values(), arm, "HA", p["HA"], levels=(0.95,))
        check_summary(e.values(), arm, "HM1", p["HM"]["HM1"])
        assert set(d) == set(p["H1"]["d_i"])
        assert set(e) == set(p["HM"]["HM1"]["e_i"])
        for f in d:
            close(float(d[f]), p["H1"]["d_i"][f], f"{arm} d {f}")
        for f in e:
            close(float(e[f]), p["HM"]["HM1"]["e_i"][f], f"{arm} e {f}")
        xs = [float(ms) for ds, ms in parts[arm].values()]
        ys = [float(ds) for ds, ms in parts[arm].values()]
        close(AC.spearman(ys, xs), p["HM"]["HM2"]["spearman"], f"{arm} Spearman")
        close(sum(xs) / len(xs), p["HM"]["midpoint_shift_mean"], f"{arm} mean midpoint shift")
        close(sum(ys) / len(ys), p["HM"]["share_shift_mean_same_games"], f"{arm} matched split")
        omitted = sum(g["terminal"] == "AGREEMENT" and AC.midpoint(g) is None for g in games)
        assert omitted == p["HM"]["agreements_without_both_first_offers"]
        ci = p["HM"]["HM1"]["ci90"]
        assert p["HM"]["HM1"]["equivalent"] == (-0.05 < ci[0] and ci[1] < 0.05)
        assert p["HM"]["supported"] == (p["HM"]["HM1"]["equivalent"] and p["HM"]["HM2"]["p_holm"] < 0.05)
    return result, parts


def signed(x):
    return f"{x:+.3f}".replace("-", "−")


def model_mark(arm, x, y, radius=6, opacity=None):
    s = MODEL[arm]
    return marker(s["marker"], x, y, radius, s["col"], s["fill"], 2, opacity)


def forest(result, y, letter, title, subtitle, key, bounds, ticks, ci_key, margin=None):
    out = [text(24, y, letter, 22, weight="bold"), text(50, y, title, 19, weight="bold"),
           text(50, y + 24, subtitle, 15, GREY)]
    x0, x1 = 305, 635
    lo, hi = bounds
    X = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)
    top, row_h = y + 58, 65
    bot = top + row_h * len(ARMS)
    if margin is not None:
        out.append(f"<rect x='{X(-margin):.1f}' y='{top}' width='{X(margin) - X(-margin):.1f}' "
                   f"height='{bot - top}' fill='{BAND}'/>")
    for tick in ticks:
        out.append(line(X(tick), top, X(tick), bot, LIGHT, 1))
        out.append(text(X(tick), bot + 27, f"{tick:+.2f}".replace("-", "−") if tick else "0", anchor="middle"))
    out.append(line(X(0), top, X(0), bot, GREY, 1.2, "4 3"))
    for i, arm in enumerate(ARMS):
        p = result["per_model"][arm]
        s = p["HM"]["HM1"] if key == "HM1" else p[key]
        cy = top + row_h * (i + 0.5)
        col = MODEL[arm]["col"]
        out += [model_mark(arm, 37, cy), text(54, cy + 5, arm, 16, weight="bold")]
        a, b = s[ci_key]
        out.append(line(X(a), cy, X(b), cy, col, 3))
        if key == "H1":
            out.append(line(X(s["ci90"][0]), cy, X(s["ci90"][1]), cy, col, 7))
        out.append(model_mark(arm, X(s["mean"]), cy))
        out.append(text(663, cy - 3, f"{signed(s['mean'])}  [{signed(a)}, {signed(b)}]", 16, col, "bold"))
        if key == "H1":
            verdict = "invariant within ±0.10" if p["H2_equivalent_within_0.10"] else "split shifts" if s["p_holm"] < 0.05 else "inconclusive"
        elif key == "HA":
            verdict = ("HA supported" if s["supported"] else "HA not supported") + f" · {s['below_0']}/{s['n_fixtures']} below 0"
        else:
            verdict = "HM1 passes ±0.05" if s["equivalent"] else "HM1 equivalence not established"
        out.append(text(663, cy + 20, verdict, 15))
    out.append(line(x0, bot + 4, x1, bot + 4, ONYX, 1.2))
    return out, bot + 55


def opening_and_split(result):
    a, bottom = forest(result, 32, "a", "The opening gets more modest in all three models",
                       "HA (registered secondary): change in the opener's gap from its own limit · 95% CI",
                       "HA", (-0.28, 0.02), (-0.25, -0.20, -0.15, -0.10, -0.05, 0), "ci95")
    a.append(text(50, bottom, "Negative h: the opener asks for less at $100M. This is not a seller-share measure.", 15, GREY))
    b, bottom = forest(result, bottom + 53, "b", "The final split shifts on Luna and Gemini, not Haiku",
                       "H1/H2 (primary): seller-share change · thin bar / brackets: 95% CI; thick: 90%; band: ±0.10",
                       "H1", (-0.40, 0.15), (-0.4, -0.3, -0.2, -0.1, 0, 0.1), "ci95", 0.10)
    b += [text(50, bottom, "Negative d: the seller keeps less of the bargaining surplus at $100M.", 15, GREY),
          text(50, bottom + 27, "Each row: 30 fixtures, 720 games; $100M − $100. H1 and HA: Holm across the three models.", 15, GREY),
          text(50, bottom + 49, f"Source: {RUN}; frozen Addendum C analysis. H1 includes all agreements.", 15, GREY)]
    return svg_doc(1000, int(bottom + 68), "Addendum C: opening modesty and final splits",
                   "Registered full games on three API models, 720 games each. Opening modesty (HA) holds on all three. "
                   "Seller share falls by 0.116 on Luna and 0.304 on Gemini; Haiku's shift of 0.006 meets equivalence "
                   "within plus or minus 0.10. The two panels use different measures; all changes are $100M minus $100.", a + b)


def midpoint_figure(result, parts):
    out, bottom = forest(result, 32, "a", "Calibration within ±0.05 is established only on Luna",
                         "HM1: closing-share shift − opening-midpoint shift, on the same games · 90% CI; band: ±0.05",
                         "HM1", (-0.10, 0.10), (-0.10, -0.05, 0, 0.05, 0.10), "ci90", 0.05)
    y0 = bottom + 35
    out += [text(24, y0, "b", 22, weight="bold"),
            text(50, y0, "The opening and closing shifts track across fixtures on all three models", 19, weight="bold"),
            text(50, y0 + 24, "HM2: one point per fixture; the same eligible deals on both axes. Dashed line: equality.", 15, GREY)]
    values = [float(v) for pairs in parts.values() for pair in pairs.values() for v in pair]
    lo = math.floor((min(values) - 0.04) * 5) / 5
    hi = math.ceil((max(values) + 0.04) * 5) / 5
    ticks = [k / 5 for k in range(round(lo * 5), round(hi * 5) + 1)]
    size, top = 250, y0 + 95
    for i, arm in enumerate(ARMS):
        left = 85 + 305 * i
        X = lambda v: left + (v - lo) / (hi - lo) * size
        Y = lambda v: top + size - (v - lo) / (hi - lo) * size
        hm = result["per_model"][arm]["HM"]
        out += [text(left + size / 2, top - 42, arm, 16, weight="bold", anchor="middle"),
                text(left + size / 2, top - 19, f"Spearman {hm['HM2']['spearman']:.3f}; Holm p = {hm['HM2']['p_holm']:.5f}",
                     15, GREY, anchor="middle")]
        for v in ticks:
            out += [line(X(v), top, X(v), top + size, LIGHT, 1), line(left, Y(v), left + size, Y(v), LIGHT, 1),
                    text(X(v), top + size + 24, f"{v:.1f}".replace("-", "−"), anchor="middle")]
            if i == 0:
                out.append(text(left - 10, Y(v) + 5, f"{v:.1f}".replace("-", "−"), anchor="end"))
        out.append(line(X(lo), Y(lo), X(hi), Y(hi), GREY, 1.5, "5 4"))
        for ds, ms in parts[arm].values():
            out.append(model_mark(arm, X(float(ms)), Y(float(ds)), 3.7, 0.8))
        out += [line(left, top + size, left + size, top + size, ONYX, 1.2),
                line(left, top, left, top + size, ONYX, 1.2),
                text(left + size / 2, top + size + 49, f"mean opening {signed(hm['midpoint_shift_mean'])}", 15, anchor="middle"),
                text(left + size / 2, top + size + 70, f"mean closing {signed(hm['share_shift_mean_same_games'])}", 15, anchor="middle")]
    out += [text(27, top + size / 2, "closing shift (surplus share)", 15, anchor="middle", rotate=-90),
            text(500, top + size + 99, "opening-midpoint shift (surplus share)", 15, anchor="middle")]
    g = result["per_model"][ARMS[2]]
    foot = top + size + 134
    out += [text(50, foot, "Both tests are required for HM support. Positive tracking alone does not establish calibration or causation.", 15, GREY),
            text(50, foot + 23, f"Gemini: {g['HM']['agreements_without_both_first_offers']} deals lack both first offers and are excluded from HM; "
                 f"matched closing {signed(g['HM']['share_shift_mean_same_games'])}, primary {signed(g['H1']['mean'])}.", 15, GREY),
            text(50, foot + 46, f"30 fixtures per model; {RUN}. All changes: $100M − $100.", 15, GREY)]
    return svg_doc(1000, int(foot + 66), "Addendum C: the registered midpoint tests",
                   "HM1's 90% confidence interval fits inside plus or minus 0.05 only on Luna. HM2 tracking is positive "
                   "and significant on all three models. The scatter compares the same eligible games for opening and "
                   "closing shifts, excluding Gemini's 76 deals without a first offer from both sides. Accounting, not causation.", out)


def main():
    result, parts = load()
    for stem, doc in (("fig9_addendum_c", opening_and_split(result)),
                      ("fig10_addendum_c_midpoint", midpoint_figure(result, parts))):
        print(write(os.path.join(HERE, stem + ".svg"), doc))
    print("Plotted fixture values, means, CIs and correlations match the registered analysis (tolerance 1e-9).")


if __name__ == "__main__":
    main()
