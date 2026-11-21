"""cross_model_figures.py: the across-model view, plus the registered Holm corrections.

    python3 scripts/cross_model_figures.py

Finds every real run (data/runs/*/run_manifest.json, kind "real") that has
derived/<run_id>/primary.json (from scripts/analyze.py), groups them by study,
and writes:
  derived/cross_model/summary.json
      Holm across the models that ran, per family:
        Study 1 H1 (pre-registration §5), Study 2 H1 (Study 2 §4-5),
        Addendum A HA and HB (Gemma models only; ADDENDUM_A.md §3).
  derived/cross_model/figures/C1_primary_by_model.svg
      the headline: mean d with 90% and 95% CIs per study x model, the ±0.10 band
  derived/cross_model/figures/C2_opener_modesty_by_model.svg
      Addendum A, HA: mean change in the opener's gap ($100M − $100) per model;
      Qwen drawn hollow as the exploratory source, Gemma as the confirmatory test
  derived/cross_model/figures/C3_fixture_slopes_by_model.svg
      Study 2: every fixture's seller share at $100, $100K, $100M, per model (descriptive)
  derived/cross_model/figures/C4_price_levels_by_model.svg
      Study 2: median opening offers and deal price as a multiple of V, by scale (exploratory)
Models that haven't run show as "not run yet", "not run: pilot failed the format gate"
(read from the pilot's gate5 event), or "not run (scope change)" for Study 1 Gemma.
Re-run after every new analysis. Standard library only.
"""
import glob
import json
import os
import statistics as st
import sys
from xml.sax.saxutils import escape

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from analyze import holm                                   # noqa: E402
from make_figures import BLUE, GREEN, GREY, INK, ORANGE, ROOT, SCALES, Svg   # noqa: E402
from explore_figures import VERM, mark                      # noqa: E402

MODELS = [m["name"] for m in json.load(open(os.path.join(ROOT, "config", "pins.json")))["models"]]
SHORT = {"qwen3.8-27b-fp8": "Qwen3.8 27B", "gemma-4-12b-it": "Gemma 4 12B", "gemma-4-31b-it": "Gemma 4 31B"}
STUDY_LABEL = {1: "Study 1 · appraisal shown", 2: "Study 2 · no appraisal"}
GEMMA = [m for m in MODELS if m.startswith("gemma")]


def runs():
    """{(study, model): primary.json dict (+ run_id)}, latest run per pair."""
    out = {}
    for mpath in sorted(glob.glob(os.path.join(ROOT, "data", "runs", "*", "run_manifest.json"))):
        man = json.load(open(mpath))
        if man.get("kind") != "real" or man.get("preview"):
            continue
        prim = os.path.join(ROOT, "derived", man["run_id"], "primary.json")
        if not os.path.exists(prim):
            continue
        key = (man.get("study", 1), man["model"])
        if key not in out or man["run_id"] > out[key]["run_id"]:
            out[key] = json.load(open(prim))
    return out


def families(R):
    fam = {}
    for study in (1, 2):
        p = {m: R[(study, m)]["H1"]["p_unadjusted"] for m in MODELS if (study, m) in R}
        fam[f"study{study}_H1"] = {"models": sorted(p), "p_unadjusted": p, "p_holm": holm(p) if p else {}}
    for h, key in (("HA", "p_one_sided_unadjusted"), ("HB", "p_two_sided_unadjusted")):
        p = {m: R[(2, m)]["addendum_A"][h][key] for m in GEMMA if (2, m) in R and "addendum_A" in R[(2, m)]}
        fam[f"addendumA_{h}"] = {"models": sorted(p), "p_unadjusted": p, "p_holm": holm(p) if p else {},
                                 "note": "Gemma models only; Qwen is the hypothesis source and is excluded"}
    return fam


def pfmt(p):
    return "< 0.0001" if p < 0.0001 else f"{p:.4f}"


def halo(s, x, y, txt, color=INK, size=12, anchor="start", weight="normal"):
    s.add(f"<text x='{x:.1f}' y='{y:.1f}' font-size='{size}' fill='{color}' font-weight='{weight}'"
          f" text-anchor='{anchor}' stroke='white' stroke-width='4' paint-order='stroke'>{escape(txt)}</text>")


def not_run_label(study, m):
    """Why a (study, model) row has no result: a failed format gate if its pilot says so."""
    if study == 1 and m != "qwen3.8-27b-fp8":
        return "not run (scope change)"
    prefix = "s2-pilot-" if study == 2 else "pilot-"
    for ev in sorted(glob.glob(os.path.join(ROOT, "data", "pilot", f"{prefix}{m}-*", "events.jsonl"))):
        for line in open(ev):
            e = json.loads(line)
            if e.get("kind") == "gate5" and not e.get("passed"):
                return f"not run: pilot failed the format gate ({100 * e['valid_fraction']:.1f}% valid)"
    return "not run yet"


def fig_primary(R, fam, out):
    rows = [(1, m) for m in MODELS] + [(2, m) for m in MODELS]
    vals = [x for k in rows if k in R for x in R[k]["bootstrap"]["ci95_mean"]]
    lim = max(0.3, max((abs(v) for v in vals), default=0) + 0.05)
    W, H = 1000, 140 + 46 * len(rows) + 120
    s = Svg(W, H, "Shift in seller share from $100 to $100M, by study and model",
            "One row per study and model: the mean per-fixture shift d with its 90% (thick) and 95% (thin) intervals."
            " The green band is the pre-registered ±0.10 equivalence margin. Rows not yet run are labelled.")
    s.text(30, 34, "Does multiplying every number by a million move the split?", 19, weight="bold")
    s.text(30, 56, "Mean shift in the seller's share, $100M minus $100 (per fixture, averaged over openers)."
           " Thick bar: 90% CI; thin: 95% CI. Green: ±0.10.", 12.5, color="#444")
    L, Rr, T = 300, 640, 100
    X = lambda v: L + (v + lim) / (2 * lim) * (Rr - L)      # noqa: E731
    bottom = T + 46 * len(rows)
    s.add(f"<rect x='{X(-0.1):.1f}' y='{T - 10}' width='{X(0.1) - X(-0.1):.1f}' height='{bottom - T + 10}'"
          f" fill='{GREEN}' fill-opacity='0.12'/>")
    for t in [round(-lim + i * 0.1, 1) for i in range(int(2 * lim / 0.1) + 1)]:
        s.line(X(t), T - 10, X(t), bottom, "#e6e6e6")
        s.text(X(t), bottom + 18, f"{t:+.1f}" if t else "0", 11.5, "middle", "#444")
    s.line(X(0), T - 10, X(0), bottom, INK, 1.2)
    s.text(X(-lim), bottom + 38, "← seller gets less at $100M", 11.5, color="#444", italic=True)
    s.text(X(lim), bottom + 38, "seller gets more at $100M →", 11.5, "end", "#444", italic=True)
    s.text(Rr + 20, T - 18, "Wilcoxon p (Holm across models that ran) · reading", 11.5, color="#444", weight="bold")
    for i, (study, m) in enumerate(rows):
        y = T + 46 * i + 14
        if i == len(MODELS):
            s.line(30, y - 24, W - 30, y - 24, "#cccccc")
        s.text(30, y + 4, f"{STUDY_LABEL[study]} · {SHORT.get(m, m)}", 12.5, weight="bold")
        color = BLUE if study == 1 else VERM
        if (study, m) not in R:
            halo(s, X(0), y + 4, not_run_label(study, m), GREY, 12, "middle")
            continue
        p = R[(study, m)]
        lo95, hi95 = p["bootstrap"]["ci95_mean"]
        lo90, hi90 = p["bootstrap"]["ci90_mean"]
        s.line(X(lo95), y, X(hi95), y, color, 1.6)
        s.line(X(lo90), y, X(hi90), y, color, 5)
        mark(s, X(p["mean_d"]), y, color, "circle" if study == 1 else "square", False, 6)
        ph = fam[f"study{study}_H1"]["p_holm"].get(m)
        eq = "invariant within ±0.10" if p.get("H2_equivalent_within_0.10") else "not shown invariant"
        p2 = lambda v: "< 0.0001" if v < 0.0001 else f"{v:.2f}"   # noqa: E731
        halo(s, Rr + 20, y + 4, f"p {p2(p['H1']['p_unadjusted'])} (Holm {p2(ph)}) · {eq}", INK, 12)
        halo(s, X(p["mean_d"]), y - 10, f"{p['mean_d']:+.3f}", color, 11, "middle", "bold")
        s.text(Rr + 20, y + 20, f"{p['fixtures_in_test']} fixtures, {p['games_scored']} games", 11, color="#666")
    s.text(30, H - 40, "Confirmatory: each study's pre-registered primary test (Study 1: prereg/PREREGISTRATION.md §5;"
           " Study 2: prereg/study2/PREREGISTRATION.md §4–5).", 11.5, color="#444")
    s.text(30, H - 22, "Holm is applied within each study across the models that ran. Source: derived/<run>/primary.json.",
           11.5, color="#444")
    s.save(os.path.join(out, "C1_primary_by_model.svg"))


def fig_ha(R, fam, out):
    rows = list(MODELS)
    vals = [100 * x for m in rows if (2, m) in R and "addendum_A" in R[(2, m)]
            for x in (R[(2, m)]["addendum_A"]["HA"]["ci95_mean_h"] or [])]
    lo_lim = min(-20, min(vals, default=0) - 3)
    hi_lim = max(10, max(vals, default=0) + 3)
    W, H = 1000, 140 + 50 * len(rows) + 110
    s = Svg(W, H, "Opener modesty at $100M, by model (Study 2 Addendum A, HA)",
            "One row per model: the mean per-fixture change in the opener's gap from its own limit, $100M minus $100,"
            " in percentage points, with a 95% interval. Below zero means the opener sits closer to its own limit at"
            " $100M. Qwen, the source of the hypothesis, is hollow; Gemma rows are the confirmatory test.")
    s.text(30, 34, "Do big numbers make the opener modest? (Addendum A, HA)", 19, weight="bold")
    s.text(30, 56, "Change in the opener's first-offer gap from its own limit, $100M minus $100, percentage points"
           " (mean over fixtures; bar: 95% CI). Study 2 prompt.", 12.5, color="#444")
    L, Rr, T = 300, 640, 100
    X = lambda v: L + (v - lo_lim) / (hi_lim - lo_lim) * (Rr - L)   # noqa: E731
    bottom = T + 50 * len(rows)
    for t in range(int(lo_lim // 5 * 5), int(hi_lim) + 1, 5):
        if lo_lim <= t <= hi_lim:
            s.line(X(t), T - 10, X(t), bottom, "#e6e6e6")
            s.text(X(t), bottom + 18, f"{t:+d}" if t else "0", 11.5, "middle", "#444")
    s.line(X(0), T - 10, X(0), bottom, INK, 1.2)
    s.text(X(lo_lim), bottom + 38, "← opener closer to its own limit at $100M (HA)", 11.5, color="#444", italic=True)
    s.text(Rr + 20, T - 18, "one-sided p (Holm across Gemma) · fixtures below 0", 11.5, color="#444", weight="bold")
    for i, m in enumerate(rows):
        y = T + 50 * i + 14
        source = not m.startswith("gemma")
        s.text(30, y + 4, f"{SHORT.get(m, m)}", 12.5, weight="bold")
        s.text(30, y + 20, "hypothesis source (exploratory)" if source else "confirmatory test", 11,
               color="#666", italic=True)
        a = R.get((2, m), {}).get("addendum_A")
        if not a:
            halo(s, X(0), y + 4, not_run_label(2, m), GREY, 12, "middle")
            continue
        ha = a["HA"]
        lo, hi = (100 * v for v in ha["ci95_mean_h"])
        s.line(X(lo), y, X(hi), y, VERM, 2.4)
        mark(s, X(100 * ha["mean_h"]), y, VERM, "square", source, 6.5)
        halo(s, X(100 * ha["mean_h"]), y - 11, f"{100 * ha['mean_h']:+.1f}", VERM, 11, "middle", "bold")
        ph = fam["addendumA_HA"]["p_holm"].get(m)
        ptxt = f"p {pfmt(ha['p_one_sided_unadjusted'])}" + (f" (Holm {pfmt(ph)})" if ph is not None else " (not in the test)")
        halo(s, Rr + 20, y + 4, f"{ptxt} · {ha['fixtures_h_below_0']} of {ha['n_fixtures']}", INK, 12)
    s.text(30, H - 40, "Registered in prereg/study2/ADDENDUM_A.md before any Gemma game; tested on Gemma only."
           " Qwen is shown for reference and is not evidence.", 11.5, color="#444")
    s.text(30, H - 22, "Source: derived/<run>/primary.json (analyze.py --addendum).", 11.5, color="#444")
    s.save(os.path.join(out, "C2_opener_modesty_by_model.svg"))


def study2_games(R):
    """[(model, primary, games)] for every Study 2 model that ran, in pins.json order."""
    out = []
    for m in MODELS:
        if (2, m) in R:
            rid = R[(2, m)]["run_id"]
            games = [json.loads(l) for l in open(os.path.join(ROOT, "data", "runs", rid, "games.jsonl")) if l.strip()]
            out.append((m, R[(2, m)], [g for g in games if g["terminal"] != "INFRASTRUCTURE_VOID"]))
    return out


def axes(s, x0, y0, w, h, lo, hi, ticks, fmt):
    """Y gridlines + labels and the three scale labels; returns (X positions, Y function)."""
    X = [x0 + w * (i + 0.5) / 3 for i in range(3)]
    Y = lambda v: y0 + h - (v - lo) / (hi - lo) * h       # noqa: E731
    for t in ticks:
        s.line(x0, Y(t), x0 + w, Y(t), "#e6e6e6")
        s.text(x0 - 8, Y(t) + 4, fmt.format(t), 11.5, "end", "#444")
    for x, (_, lab) in zip(X, SCALES):
        s.text(x, y0 + h + 22, lab, 13, "middle", weight="bold")
    return X, Y


def fig_slopes(R, out):
    """C3: every fixture's seller share at the three scales, one panel per Study 2 model."""
    data = study2_games(R)
    if not data:
        return
    PW, PH, T = 400, 330, 120
    W, H = 90 + len(data) * (PW + 70), T + PH + 130
    s = Svg(W, H, "Seller share at $100, $100K and $100M for every fixture, by model (Study 2)",
            "One panel per Study 2 model. Each grey line is one fixture: its seller share (mean of the agreed games,"
            " both openers) at the three scales. The thick line is the mean over fixtures. Descriptive; the"
            " pre-registered test of $100M against $100 is in C1.")
    s.text(30, 34, "Same deals, bigger numbers: where does each fixture's split go?", 19, weight="bold")
    s.text(30, 56, "Seller's share of the bargaining range at each scale. Grey: one line per fixture (mean of its"
           " agreed games). Thick: mean over fixtures. Study 2 prompt (no appraisal).", 12.5, color="#444")
    for k, (m, prim, games) in enumerate(data):
        cell = {}
        for g in games:
            if g["seller_share"] is not None:
                cell.setdefault((g["fixture"], g["scale"]), []).append(g["seller_share"]["float"])
        fixtures = sorted({f for f, _ in cell})
        rows = [[st.mean(cell[(f, sc)]) if (f, sc) in cell else None for sc, _ in SCALES] for f in fixtures]
        vals = [v for r in rows for v in r if v is not None]
        lo, hi = min(-0.05, min(vals)), max(1.0, max(vals))
        x0 = 90 + k * (PW + 70)
        X, Y = axes(s, x0, T, PW, PH, lo, hi, [0, 0.25, 0.5, 0.75, 1.0], "{:.2f}")
        s.text(x0, T - 30, SHORT.get(m, m), 15, weight="bold")
        eq = "invariant within ±0.10" if prim.get("H2_equivalent_within_0.10") else "shifts"
        s.text(x0, T - 12, f"mean d {prim['mean_d']:+.3f} ($100M − $100) · pre-registered reading: {eq}", 12,
               color="#444")
        for r in rows:
            pts = [(X[i], Y(v)) for i, v in enumerate(r) if v is not None]
            if len(pts) > 1:
                s.add("<polyline points='" + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts) +
                      "' fill='none' stroke='#9a9a9a' stroke-opacity='0.55' stroke-width='1.2'/>")
        means = [st.mean([r[i] for r in rows if r[i] is not None]) for i in range(3)]
        s.add("<polyline points='" + " ".join(f"{X[i]:.1f},{Y(v):.1f}" for i, v in enumerate(means)) +
              f"' fill='none' stroke='{VERM}' stroke-width='4'/>")
        for i, v in enumerate(means):
            mark(s, X[i], Y(v), VERM, "square", False, 6)
            halo(s, X[i] + 12, Y(v) - 8, f"{v:.2f}", VERM, 12.5, "start", "bold")
        down = sum(1 for r in rows if r[0] is not None and r[2] is not None and r[2] < r[0])
        both = sum(1 for r in rows if r[0] is not None and r[2] is not None)
        s.text(x0, T + PH + 48, f"{down} of {both} fixtures lower at $100M than at $100 · {len(games)} games",
               11.5, color="#444")
    s.text(30, H - 22, "Descriptive. The registered test ($100M vs $100, Wilcoxon, Holm) is C1. Source:"
           " data/runs/<run>/games.jsonl.", 11.5, color="#444")
    s.save(os.path.join(out, "C3_fixture_slopes_by_model.svg"))


def sym(s, x, y, color, shape, r=5.5):
    """Marker: triangle, or explore_figures.mark's circle/square (filled)."""
    if shape == "triangle":
        s.add(f"<path d='M{x:.1f},{y - r - 1:.1f} L{x + r + 1:.1f},{y + r:.1f} L{x - r - 1:.1f},{y + r:.1f} Z'"
              f" fill='{color}' stroke='white'/>")
    else:
        mark(s, x, y, color, shape, False, r)


def fig_prices(R, out):
    """C4: where the prices sit, as a multiple of V (never shown to the models), by scale and side."""
    data = study2_games(R)
    if not data:
        return
    PW, PH, T = 400, 330, 120
    W, H = 90 + len(data) * (PW + 130) + 90, T + PH + 150
    s = Svg(W, H, "Where the prices sit, as a multiple of the lot's value, by scale and model (Study 2)",
            "One panel per Study 2 model. Median opening offer when the seller opens (circles), when the buyer opens"
            " (triangles), and the median deal price (squares), each divided by V, the value the fixture was built"
            " around (never shown to the models). Dashed lines: the median seller floor and buyer ceiling, the same"
            " at every scale by design. Exploratory.")
    s.text(30, 34, "Where do the prices sit? Openings and deals as a multiple of the lot's value", 19, weight="bold")
    s.text(30, 56, "Medians over 30 games per point. V sets each fixture's limits but is never shown to the models."
           " Dashed: median floor and ceiling (identical at every scale).", 12.5, color="#444")
    for k, (m, prim, games) in enumerate(data):
        def med(f):
            return [st.median([f(g) for g in games if g["scale"] == sc and f(g) is not None]) for sc, _ in SCALES]
        first = lambda side: lambda g: g["offers"][0][2] / g["V"] if g["opener"] == side and g["offers"] else None  # noqa: E731,E501
        series = [("seller opens: first offer", med(first("seller")), BLUE, "circle"),
                  ("buyer opens: first offer", med(first("buyer")), ORANGE, "triangle"),
                  ("deal price", med(lambda g: g["price"] / g["V"] if g["price"] is not None else None), INK, "square")]
        fl = st.median([g["F"] / g["V"] for g in games])
        ce = st.median([g["C"] / g["V"] for g in games])
        vals = [v for _, vs, _, _ in series for v in vs] + [fl, ce]
        lo, hi = min(vals) - 0.05, max(vals) + 0.05
        ticks = [t / 10 for t in range(int(lo * 10) + 1, int(hi * 10) + 1)]
        x0 = 90 + k * (PW + 130)
        X, Y = axes(s, x0, T, PW, PH, lo, hi, ticks, "{:.1f} V")
        s.text(x0, T - 30, SHORT.get(m, m), 15, weight="bold")
        s.text(x0, T - 12, f"seller share falls by {-prim['mean_d']:.2f} from $100 to $100M" if prim["mean_d"] < -0.1
               else f"seller share: mean d {prim['mean_d']:+.3f} (invariant within ±0.10)" if
               prim.get("H2_equivalent_within_0.10") else f"seller share: mean d {prim['mean_d']:+.3f}", 12,
               color="#444")
        for v, lab in ((fl, "seller's floor"), (ce, "buyer's ceiling")):
            s.line(x0, Y(v), x0 + PW, Y(v), GREY, 1.3, "6 4")
            halo(s, x0 + PW + 6, Y(v) + 4, lab, "#666", 11)
        for lab, vs, color, shape in series:
            dy = 20 if shape == "triangle" else -8           # buyer labels below the point, others above
            s.add("<polyline points='" + " ".join(f"{X[i]:.1f},{Y(v):.1f}" for i, v in enumerate(vs)) +
                  f"' fill='none' stroke='{color}' stroke-width='2.5'/>")
            for i, v in enumerate(vs):
                sym(s, X[i], Y(v), color, shape)
                halo(s, X[i] + 10, Y(v) + dy, f"{v:.2f}", color, 11.5, "start", "bold")
        for j, (lab, _, color, shape) in enumerate(series):
            yy = T + PH + 50 + 18 * j
            sym(s, x0 + 6, yy - 4, color, shape, 4.5)
            s.text(x0 + 18, yy, lab, 12, color=INK)
    s.text(30, H - 22, "Exploratory, not pre-registered (HA, the registered opening test, is C2). Source:"
           " data/runs/<run>/games.jsonl.", 11.5, color="#444")
    s.save(os.path.join(out, "C4_price_levels_by_model.svg"))


def main():
    R = runs()
    fam = families(R)
    base = os.path.join(ROOT, "derived", "cross_model")
    out = os.path.join(base, "figures")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(base, "summary.json"), "w") as fh:
        json.dump({"runs": {f"study{k[0]}|{k[1]}": v["run_id"] for k, v in sorted(R.items())}, "families": fam},
                  fh, indent=1)
    fig_primary(R, fam, out)
    fig_ha(R, fam, out)
    fig_slopes(R, out)
    fig_prices(R, out)
    print("runs found: " + ", ".join(f"Study {k[0]} {k[1]}" for k in sorted(R)))
    for name, f in fam.items():
        if f["p_unadjusted"]:
            print(f"  {name}: " + ", ".join(f"{m} p {f['p_unadjusted'][m]:.4f} → Holm {f['p_holm'][m]:.4f}"
                                           for m in f["models"]))
    for f in sorted(os.listdir(out)):
        print(os.path.relpath(os.path.join(out, f), ROOT))
    print(os.path.relpath(os.path.join(base, "summary.json"), ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
