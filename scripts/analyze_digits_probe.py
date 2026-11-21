#!/usr/bin/env python3
"""analyze_digits_probe.py: summary of the digits x stakes probe (STORY §12, Q2). Exploratory.

    python3 scripts/analyze_digits_probe.py data/explore/X-digits-<model>-<stamp>

Outcome: the opener's first-offer gap from its own limit (HA's g, in %):
seller P/F - 1, buyer 1 - P/C, in the reply's own unit. Per fixture, role and
condition, the mean over samples (OFFERs only). A contrast X - Y per fixture is
the mean over roles of (gap_X - gap_Y); tests are the sign-flip Wilcoxon and
bootstrap from analyze.py (two-sided, exploratory signals, not confirmatory).

Writes derived/explore/<run_id>/summary.json and
derived/explore/<run_id>/figures/D1_gap_by_condition.svg. Standard library only.
"""
import collections
import json
import os
import random
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import analyze as A                                        # noqa: E402
from make_figures import BLUE, GREY, INK, ORANGE, Svg      # noqa: E402

FLIPS = 50_000                 # exploratory: Monte Carlo error about ±0.001 near small p
ORDER = ["D2", "D5", "D8", "C8", "W8", "U5", "U8", "K2", "S2", "SN", "S8", "SM", "X2"]
LABEL = {"D2": '"81 dollars"', "D5": '"81000 dollars"', "D8": '"81000000 dollars"',
         "C8": '"81,000,000 dollars"', "W8": '"81 million dollars"', "U5": '"81 thousand", reply in thousands',
         "U8": '"81 million", reply in millions', "K2": '"8100 cents", reply in cents',
         "S2": '"$81"', "SN": '"$81000000"', "S8": '"$81,000,000"', "SM": '"$81 million"', "X2": '"81.00 dollars"'}
CONTRASTS = [  # (X, Y, question)
    ("D8", "D2", "digits + stakes together (the Study 2 contrast, on fresh fixtures)"),
    ("U8", "D2", "stakes only: same digits read and written, unit word says millions"),
    ("W8", "D2", "stakes, short digits in the limit, long digits in the reply"),
    ("D8", "U8", "digits only, at fixed $81M stakes (long vs short digit strings)"),
    ("W8", "D8", "limit in words vs digits, same stakes, same reply unit"),
    ("C8", "D8", "commas vs none, same stakes"),
    ("K2", "D2", "more digits, same small stakes (cents)"),
    ("D5", "D2", "digits + stakes, $100K step"),
    ("U5", "D2", "stakes only, $100K step"),
    ("S2", "D2", "dollar sign vs the word, small stakes"),
    ("SN", "D8", "dollar sign vs the word, $81M in plain digits"),
    ("S8", "D8", "'$81,000,000' vs '81000000 dollars'"),
    ("SM", "W8", "'$81 million' vs '81 million dollars'"),
    ("S8", "S2", "the scale effect when both are written with a dollar sign and commas"),
    ("X2", "D2", "cents shown ('81.00') vs not, small stakes"),
]


def load(run_dir):
    """Final attempt per item (an INFRASTRUCTURE record is replaced by its retry)."""
    recs = {}
    for line in open(os.path.join(run_dir, "games.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        if r.get("schema") != "firstmove/1":
            continue
        prev = recs.get(r["item_id"])
        if prev is None or r.get("attempt", 1) > prev.get("attempt", 1):
            recs[r["item_id"]] = r
    return [r for r in recs.values() if r["errors"] != ["INFRASTRUCTURE"]]


def cell_means(recs):
    """{(fixture, role, condition): mean gap in % over the samples that OFFERed}."""
    acc = collections.defaultdict(list)
    for r in recs:
        if r["gap"] is not None:
            acc[(r["fixture"], r["role"], r["condition"])].append(100 * r["gap"])
    return {k: st.mean(v) for k, v in acc.items()}


def contrast(cm, x, y, roles=("seller", "buyer")):
    """Per-fixture values: mean over roles (where both defined) of gap_x - gap_y."""
    out = []
    for f in sorted({k[0] for k in cm}):
        parts = [cm[(f, r, x)] - cm[(f, r, y)] for r in roles if (f, r, x) in cm and (f, r, y) in cm]
        if parts:
            out.append(st.mean(parts))
    return out


def test(values, what, model):
    if len(values) < 2:
        return {"n": len(values)}
    p, wplus, nz = A.wilcoxon_signflip(values, random.Random(A.seed(f"digits-{what}", model)), flips=FLIPS)
    ci = A.bootstrap_ci(values, random.Random(A.seed(f"digits-boot-{what}", model)), levels=(0.95,))["95"]
    return {"n": len(values), "mean": st.mean(values), "ci95": ci, "p_two_sided": p, "W_plus": wplus,
            "n_nonzero": nz, "fixtures_below_0": sum(v < 0 for v in values)}


def figure(summary, out_path):
    conds = [c for c in ORDER if c in summary["by_condition"]]
    roles = ("seller", "buyer")
    vals = [summary["by_condition"][c][r]["mean_gap_pct"] for c in conds for r in roles
            if summary["by_condition"][c][r]["mean_gap_pct"] is not None]
    hi = max(10, max(vals, default=0) * 1.15)
    lo = min(0, min(vals, default=0) * 1.15)
    W, rowh, T, L, PW = 1060, 34, 110, 330, 270
    H = T + rowh * len(conds) + 110
    s = Svg(W, H, f"Digits x stakes probe: opener's first-offer gap by condition · {summary['model']}",
            "One row per condition, two panels (seller opens, buyer opens): the mean gap between the opener's"
            " first offer and its own limit, in percent of the limit, over fresh fixtures. Lower means the opener"
            " sits closer to its own limit (more modest). Exploratory, not pre-registered.")
    s.text(30, 34, "What makes the opener modest: the digits or the stakes?", 19, weight="bold")
    s.text(30, 56, f"Mean first-offer gap from own limit (%), {summary['n_fixtures']} fresh fixtures ×"
           f" {summary['samples']} samples · {summary['model']} · exploratory", 12.5, color="#444")
    for k, (role, color, x0) in enumerate((("seller", BLUE, L), ("buyer", ORANGE, L + PW + 110))):
        X = lambda v: x0 + (v - lo) / (hi - lo) * PW                         # noqa: E731
        s.text(x0, T - 22, "seller opens: markup over floor" if role == "seller"
               else "buyer opens: discount under ceiling", 13, weight="bold")
        for t in range(int(lo // 10 * 10), int(hi) + 1, 10):
            s.line(X(t), T - 10, X(t), T + rowh * len(conds), "#e6e6e6")
            s.text(X(t), T + rowh * len(conds) + 16, f"{t}%", 11, "middle", "#444")
        base = summary["by_condition"].get("D2", {}).get(role, {}).get("mean_gap_pct")
        if base is not None:
            s.line(X(base), T - 10, X(base), T + rowh * len(conds), GREY, 1.2, "5 4")
        for i, c in enumerate(conds):
            y = T + rowh * i + rowh / 2
            d = summary["by_condition"][c][role]
            if k == 0:
                s.text(30, y + 4, f"{c}  {LABEL[c]}", 12, weight="bold" if c in ("D2", "D8", "U8") else "normal")
            if d["mean_gap_pct"] is None:
                continue
            x_a, x_b = sorted((X(0), X(d["mean_gap_pct"])))
            s.add(f"<rect x='{x_a:.1f}' y='{y - 9:.1f}' width='{x_b - x_a:.1f}' height='18'"
                  f" fill='{color}' fill-opacity='0.8'/>")
            s.text(x_b + 6, y + 4, f"{d['mean_gap_pct']:.1f}", 11.5, color=INK)
    s.text(30, H - 40, "Dashed line: D2 (the $100 wording). D2/D5/D8 = digits and stakes together, as in"
           " Studies 1–2; D2/U5/U8 = stakes only (same digits).", 11.5, color="#444")
    s.text(30, H - 22, "Source: data/explore/<run>/games.jsonl via scripts/analyze_digits_probe.py.", 11.5,
           color="#444")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    s.save(out_path)


def main(argv):
    if len(argv) != 1 or not os.path.isdir(argv[0]):
        print(__doc__.strip().splitlines()[2].strip())
        return 2
    run_dir = argv[0].rstrip("/")
    man = json.load(open(os.path.join(run_dir, "run_manifest.json")))
    model, run_id = man["model"], man["run_id"]
    recs = load(run_dir)
    cm = cell_means(recs)
    summary = {"run_id": run_id, "model": model, "exploratory": True, "samples": man.get("samples"),
               "n_fixtures": len({r["fixture"] for r in recs}), "records": len(recs),
               "quality": {}, "by_condition": {}, "contrasts": {}, "by_role": {}}
    for c in [c for c in ORDER if any(r["condition"] == c for r in recs)]:
        rc = [r for r in recs if r["condition"] == c]
        summary["quality"][c] = {"n": len(rc), "offers": sum(r["action"] == "OFFER" for r in rc),
                                 "walkaways": sum(r["action"] == "WALKAWAY" for r in rc),
                                 "invalid": dict(collections.Counter(",".join(r["errors"]) for r in rc if r["errors"]))}
        summary["by_condition"][c] = {}
        for role in ("seller", "buyer"):
            g = [100 * r["gap"] for r in rc if r["role"] == role and r["gap"] is not None]
            summary["by_condition"][c][role] = {
                "n_offers": len(g), "mean_gap_pct": st.mean(g) if g else None,
                "median_gap_pct": st.median(g) if g else None,
                "below_own_limit": sum(v < 0 for v in g)}       # an offer worse than its own limit
    for x, y, q in CONTRASTS:
        if x in summary["by_condition"] and y in summary["by_condition"]:
            summary["contrasts"][f"{x}-{y}"] = {"question": q, **test(contrast(cm, x, y), f"{x}-{y}", model)}
            summary["by_role"][f"{x}-{y}"] = {r: test(contrast(cm, x, y, (r,)), f"{x}-{y}-{r}", model)
                                             for r in ("seller", "buyer")}
    dd, du = summary["contrasts"].get("D8-D2", {}), summary["contrasts"].get("U8-D2", {})
    if dd.get("mean") and "mean" in du:
        summary["stakes_share_of_D8_effect"] = du["mean"] / dd["mean"]

    out = os.path.join(ROOT, "derived", "explore", run_id)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    figure(summary, os.path.join(out, "figures", "D1_gap_by_condition.svg"))

    print(f"Digits x stakes probe · {model} · {summary['n_fixtures']} fixtures · {len(recs)} first moves"
          " · EXPLORATORY")
    print("\nQuality (per condition): offers / walk-aways / invalid")
    for c, qd in summary["quality"].items():
        print(f"  {c:3} {qd['offers']:4} / {qd['walkaways']:3} / {sum(qd['invalid'].values()):3}"
              + (f"  {qd['invalid']}" if qd["invalid"] else ""))
    print("\nMean first-offer gap from own limit, % (seller markup | buyer discount)")
    for c, d in summary["by_condition"].items():
        f = lambda v: "   —" if v is None else f"{v:5.1f}"            # noqa: E731
        print(f"  {c:3} {f(d['seller']['mean_gap_pct'])} | {f(d['buyer']['mean_gap_pct'])}   {LABEL[c]}")
    print("\nContrasts per fixture (percentage points; both roles averaged; then seller | buyer means)")
    for k, t in summary["contrasts"].items():
        if "mean" not in t:
            continue
        br = summary["by_role"][k]
        rm = " | ".join(f"{br[r]['mean']:+.1f}" if "mean" in br[r] else "—" for r in ("seller", "buyer"))
        print(f"  {k:6} {t['mean']:+6.1f}  [{t['ci95'][0]:+.1f}, {t['ci95'][1]:+.1f}]  p {t['p_two_sided']:.4f}"
              f"  {t['fixtures_below_0']}/{t['n']} below 0   ({rm})  {t['question']}")
    if "stakes_share_of_D8_effect" in summary:
        print(f"\nStakes-only (U8) reproduces {100 * summary['stakes_share_of_D8_effect']:.0f}% of the D8 effect.")
    print(f"\nWrote {os.path.relpath(out, ROOT)}/summary.json and figures/D1_gap_by_condition.svg")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
