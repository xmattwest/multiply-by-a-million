#!/usr/bin/env python3
"""analyze.py: the pre-registered primary analysis for one model's real run.

    python3 scripts/analyze.py data/runs/<run_id> [--pilot data/pilot/<run_id>]

Implements pre-registration §4 and §5.1-5.2 exactly:
  - per fixture i and opener o: d_io = s(i, $100M, o) - s(i, $100, o), defined
    only when both games agreed; s = (P - F)/(C - F), exact fractions;
  - d_i = mean of the defined d_io; fixtures with none are left out (counted);
  - H1: two-sided Wilcoxon signed-rank on the d_i, zeros dropped, average
    ranks for ties, p from 200,000 random sign flips;
  - mean and median of d_i, 95% percentile bootstrap CI of the mean (10,000
    resamples of fixtures);
  - H2: "shift smaller than 0.10" only if the 90% bootstrap CI lies inside
    (-0.10, +0.10).
Holm across models (G4) needs every model's p, so it's applied by
--holm once all runs are in.

With --pilot, also reports the pilot's spread sigma of d_i, as amendment A1
asks (context for the §7 power table; it changes nothing).

Seeds ("fixed seed" in §5; the value wasn't registered): each is the first 8
hex digits of SHA256("scale-invariance/analysis/v1|<what>|<model>"), set
before any result was computed (2026-10-03) and never changed.

Writes derived/<run_id>/primary.json with the sha256 of every input file.
Standard library only.
"""
import argparse
import hashlib
import json
import os
import random
import statistics
import sys
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FLIPS = 200_000
BOOT = 10_000
LOW, HIGH = "x1e2", "x1e8"            # $100 and $100M


def seed(what, model):
    return int(hashlib.sha256(f"scale-invariance/analysis/v1|{what}|{model}".encode()).hexdigest()[:8], 16)


def sha_file(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def load_games(run_dir):
    """The one scored attempt per game (void/interrupted attempts never count, §4)."""
    out, missing = {}, []
    with open(os.path.join(run_dir, "games.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["terminal"] != "INFRASTRUCTURE_VOID":
                assert r["game_id"] not in out, f"two scored attempts for {r['game_id']}"
                out[r["game_id"]] = r
    with open(os.path.join(run_dir, "run_manifest.json"), encoding="utf-8") as fh:
        planned = json.load(fh)["games_planned"]
    missing = [g for g in planned if g not in out]
    return out, missing


def share(r):
    if r["terminal"] != "AGREEMENT":
        return None
    s = r["seller_share"]
    return Fraction(s["num"], s["den"])


def fixture_diffs(games):
    """d_i per fixture (exact), and how many d_io each used."""
    by = {}
    for r in games.values():
        by[(r["fixture"], r["opener"], r["scale"])] = r
    fixtures = sorted({r["fixture"] for r in games.values()})
    d, used = {}, {}
    for f in fixtures:
        parts = []
        for o in ("seller", "buyer"):
            hi, lo = by.get((f, o, HIGH)), by.get((f, o, LOW))
            if hi and lo and share(hi) is not None and share(lo) is not None:
                parts.append(share(hi) - share(lo))
        if parts:
            d[f] = sum(parts, Fraction(0)) / len(parts)
            used[f] = len(parts)
    return d, used, fixtures


def wilcoxon_signflip(values, rng, flips=FLIPS):
    """Two-sided Wilcoxon signed-rank p by random sign flips (zeros dropped,
    average ranks for tied magnitudes). Returns (p, W+, n_nonzero)."""
    v = [x for x in values if x != 0]
    n = len(v)
    if n == 0:
        return 1.0, 0, 0
    order = sorted(range(n), key=lambda i: abs(v[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(v[order[j + 1]]) == abs(v[order[i]]):
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    total = sum(ranks)
    wplus = sum(r for r, x in zip(ranks, v) if x > 0)
    t_obs = abs(2 * wplus - total)           # = |sum of signed ranks|
    hits = 0
    for _ in range(flips):
        t = sum(r if rng.random() < 0.5 else -r for r in ranks)
        if abs(t) >= t_obs - 1e-9:
            hits += 1
    return hits / flips, wplus, n


def wilcoxon_signflip_less(values, rng, flips=FLIPS):
    """One-sided (alternative: median < 0) Wilcoxon signed-rank p by random
    sign flips; same ranking rules as wilcoxon_signflip. Returns (p, W+, n)."""
    v = [x for x in values if x != 0]
    n = len(v)
    if n == 0:
        return 1.0, 0, 0
    order = sorted(range(n), key=lambda i: abs(v[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(v[order[j + 1]]) == abs(v[order[i]]):
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    wplus = sum(r for r, x in zip(ranks, v) if x > 0)
    t_obs = 2 * wplus - sum(ranks)           # signed-rank sum
    hits = sum(1 for _ in range(flips) if sum(r if rng.random() < 0.5 else -r for r in ranks) <= t_obs + 1e-9)
    return hits / flips, wplus, n


def addendum_a(run_dir):
    """prereg/study2/ADDENDUM_A.md: HA (one-sided) and HB (two-sided) for one model.
    Holm across models is applied when the per-model results are combined."""
    with open(os.path.join(run_dir, "run_manifest.json"), encoding="utf-8") as fh:
        model = json.load(fh)["model"]
    games, _ = load_games(run_dir)
    by = {(r["fixture"], r["opener"], r["scale"]): r for r in games.values()}
    fixtures = sorted({r["fixture"] for r in games.values()})

    def gap(r):
        if not r or not r["offers"] or r["offers"][0][1] != r["opener"]:
            return None
        p1 = Fraction(r["offers"][0][2])
        return p1 / r["F"] - 1 if r["opener"] == "seller" else 1 - p1 / r["C"]

    h, per_side = [], {"seller": [], "buyer": []}
    for f in fixtures:
        parts = []
        for o in ("seller", "buyer"):
            lo, hi = gap(by.get((f, o, LOW))), gap(by.get((f, o, HIGH)))
            if lo is not None and hi is not None:
                parts.append(hi - lo)
                per_side[o].append((float(lo), float(hi)))
        if parts:
            h.append(sum(parts, Fraction(0)) / len(parts))
    p_ha, wp_ha, n_ha = wilcoxon_signflip_less(h, random.Random(seed("wilcoxon-HA", model)))
    hv = [float(x) for x in h]
    ci = bootstrap_ci(hv, random.Random(seed("bootstrap-HA", model)), levels=(0.95,)) if hv else {}
    inter = []
    for f in fixtures:
        sh = [share(by.get((f, o, sc))) if by.get((f, o, sc)) else None
              for o in ("seller", "buyer") for sc in (LOW, HIGH)]
        if all(x is not None for x in sh):
            inter.append((sh[1] - sh[0]) - (sh[3] - sh[2]))
    p_hb, wp_hb, n_hb = wilcoxon_signflip(inter, random.Random(seed("wilcoxon-HB", model)))
    return {
        "model": model,
        "HA": {"n_fixtures": len(h), "n_nonzero": n_ha, "W_plus": wp_ha, "p_one_sided_unadjusted": p_ha,
               "mean_h": statistics.fmean(hv) if hv else None, "ci95_mean_h": ci.get("95"),
               "fixtures_h_below_0": sum(x < 0 for x in hv),
               "by_opener_mean_gap": {o: {"x1e2": statistics.fmean(a for a, _ in v) if v else None,
                                          "x1e8": statistics.fmean(b for _, b in v) if v else None, "n": len(v)}
                                      for o, v in per_side.items()}},
        "HB": {"n_fixtures": len(inter), "n_nonzero": n_hb, "W_plus": wp_hb, "p_two_sided_unadjusted": p_hb,
               "mean_I": statistics.fmean(float(x) for x in inter) if inter else None},
        "note": "Holm across the Gemma models is applied over both models' results (ADDENDUM_A.md §3).",
    }


def bootstrap_ci(values, rng, levels=(0.95, 0.90), boots=BOOT):
    n = len(values)
    means = sorted(sum(rng.choice(values) for _ in range(n)) / n for _ in range(boots))
    out = {}
    for lv in levels:
        lo = means[int((1 - lv) / 2 * boots)]
        hi = means[min(boots - 1, int((1 + lv) / 2 * boots) - 1)]
        out[f"{int(lv * 100)}"] = [lo, hi]
    return out


def holm(pvals):
    """Holm-adjusted p-values, same keys."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    out, run = {}, 0.0
    for k, (name, p) in enumerate(items):
        run = max(run, min(1.0, (len(items) - k) * p))
        out[name] = run
    return out


def secondary(run_dir):
    """Pre-registration §5.3, S1-S6, for one model. Holm across the tests
    with a p-value (the per-model family)."""
    with open(os.path.join(run_dir, "run_manifest.json"), encoding="utf-8") as fh:
        model = json.load(fh)["model"]
    games, _ = load_games(run_dir)
    by = {(r["fixture"], r["opener"], r["scale"]): r for r in games.values()}
    fixtures = sorted({r["fixture"] for r in games.values()})
    F = Fraction
    rng = lambda what: random.Random(seed(what, model))            # noqa: E731
    out, pv = {}, {}

    def wil(name, d):
        p, wp, n = wilcoxon_signflip(d, rng("wilcoxon-" + name))
        v = [float(x) for x in d]
        out[name] = {"n": len(d), "n_nonzero": n, "mean": statistics.fmean(v) if v else None,
                     "median": statistics.median(v) if v else None, "p_unadjusted": p, "W_plus": wp}
        pv[name] = p

    def share_diff(hi_sc, lo_sc, share_fn):
        d = []
        for f in fixtures:
            parts = []
            for o in ("seller", "buyer"):
                hi, lo = by.get((f, o, hi_sc)), by.get((f, o, lo_sc))
                if hi and lo:
                    a, b = share_fn(hi), share_fn(lo)
                    if a is not None and b is not None:
                        parts.append(a - b)
            if parts:
                d.append(sum(parts, F(0)) / len(parts))
        return d

    # S1: other scale pairs
    wil("S1a_100K_vs_100", share_diff("x1e5", "x1e2", share))
    wil("S1b_100M_vs_100K", share_diff("x1e8", "x1e5", share))
    # S2: agreements per fixture (out of 2), $100M vs $100
    wil("S2_agreements", [sum(by[(f, o, HIGH)]["terminal"] == "AGREEMENT" for o in ("seller", "buyer"))
                          - sum(by[(f, o, LOW)]["terminal"] == "AGREEMENT" for o in ("seller", "buyer"))
                          for f in fixtures])
    # S3: within-fixture disagreement |s_S - s_B|, $100M vs $100; plus across-fixture SD by bootstrap
    disp = {}
    for f in fixtures:
        for sc in (LOW, HIGH):
            a, b = share(by[(f, "seller", sc)]), share(by[(f, "buyer", sc)])
            disp[(f, sc)] = abs(a - b) if a is not None and b is not None else None
    wil("S3_opener_disagreement", [disp[(f, HIGH)] - disp[(f, LOW)] for f in fixtures
                                   if disp[(f, HIGH)] is not None and disp[(f, LOW)] is not None])
    fmean = {}
    for f in fixtures:
        for sc in (LOW, HIGH):
            v = [float(share(by[(f, o, sc)])) for o in ("seller", "buyer") if share(by[(f, o, sc)]) is not None]
            fmean[(f, sc)] = statistics.fmean(v) if v else None
    both = [f for f in fixtures if fmean[(f, LOW)] is not None and fmean[(f, HIGH)] is not None]
    r = rng("bootstrap-S3sd")
    sd = lambda fs, sc: statistics.stdev([fmean[(f, sc)] for f in fs])      # noqa: E731
    diffs = sorted(sd(x, HIGH) - sd(x, LOW) for x in ([r.choice(both) for _ in both] for _ in range(BOOT)))
    out["S3_sd_across_fixtures"] = {"sd_100": sd(both, LOW), "sd_100M": sd(both, HIGH),
                                    "diff_ci95": [diffs[int(0.025 * BOOT)], diffs[int(0.975 * BOOT) - 1]]}
    # S4: opener's first offer as a share of the range, fixture means, $100M vs $100
    def opening(rec):
        p0 = rec["offers"][0][2]
        return F(p0 - rec["F"], rec["C"] - rec["F"])
    wil("S4_opening_offer", [sum(opening(by[(f, o, HIGH)]) for o in ("seller", "buyer")) / 2
                             - sum(opening(by[(f, o, LOW)]) for o in ("seller", "buyer")) / 2 for f in fixtures])
    # S5: $100M prices rounded to the $100 grid (round-half-even, Python's round on Fractions)
    def coarse(rec):
        if rec["terminal"] != "AGREEMENT":
            return None
        if rec["scale"] != HIGH:
            return share(rec)
        P2 = round(F(rec["price"], 1_000_000))
        return F(P2 - F(rec["F"], 1_000_000), F(rec["C"] - rec["F"], 1_000_000))
    wil("S5_coarse_price", share_diff(HIGH, LOW, coarse))
    # S6: ZOPA seller utility, every game counts (no-deal 0, forfeit -1 for the forfeiter)
    def util(rec):
        u = rec["u_seller"]
        return F(u["num"], u["den"])
    wil("S6_seller_utility", share_diff(HIGH, LOW, util))
    adj = holm(pv)
    for k in pv:
        out[k]["p_holm"] = adj[k]
    out["holm_family"] = sorted(pv)
    return out


def analyze(run_dir, pilot_dir=None):
    with open(os.path.join(run_dir, "run_manifest.json"), encoding="utf-8") as fh:
        man = json.load(fh)
    model = man["model"]
    games, missing = load_games(run_dir)
    d, used, fixtures = fixture_diffs(games)
    vals = [float(d[f]) for f in sorted(d)]
    p, wplus, n_nz = wilcoxon_signflip([d[f] for f in sorted(d)], random.Random(seed("wilcoxon-H1", model)))
    ci = bootstrap_ci(vals, random.Random(seed("bootstrap-H1", model)))
    terminals = {}
    for r in games.values():
        terminals.setdefault(r["scale"], {}).setdefault(r["terminal"], 0)
        terminals[r["scale"]][r["terminal"]] += 1
    res = {
        "schema": "primary/1", "model": model, "run_id": man["run_id"],
        "games_scored": len(games), "games_missing": missing,
        "fixtures": len(fixtures), "fixtures_in_test": len(d), "fixtures_left_out": len(fixtures) - len(d),
        "fixtures_using_both_openers": sum(1 for x in used.values() if x == 2),
        "d_i": {f: {"num": d[f].numerator, "den": d[f].denominator, "float": float(d[f])} for f in sorted(d)},
        "mean_d": statistics.fmean(vals) if vals else None,
        "median_d": statistics.median(vals) if vals else None,
        "sd_d": statistics.stdev(vals) if len(vals) > 1 else None,
        "H1": {"p_unadjusted": p, "W_plus": wplus, "n_nonzero": n_nz, "flips": FLIPS,
               "seed": seed("wilcoxon-H1", model), "holm": "pending: applied across models with --holm"},
        "bootstrap": {"ci95_mean": ci["95"], "ci90_mean": ci["90"], "resamples": BOOT,
                      "seed": seed("bootstrap-H1", model)},
        "H2_equivalent_within_0.10": -0.10 < ci["90"][0] and ci["90"][1] < 0.10,
        "terminals_by_scale": terminals,
        "inputs_sha256": {os.path.relpath(os.path.join(run_dir, f), ROOT): sha_file(os.path.join(run_dir, f))
                          for f in ("run_manifest.json", "games.jsonl", "run_summary.json")},
    }
    if pilot_dir:
        pg, _ = load_games(pilot_dir)
        pd, pused, pfix = fixture_diffs(pg)
        pv = [float(x) for x in pd.values()]
        res["pilot_sigma"] = {"run_id": os.path.basename(pilot_dir), "fixtures_with_d": len(pv),
                              "sd_d": statistics.stdev(pv) if len(pv) > 1 else None,
                              "note": "amendment A1: reported only; the design is fixed"}
        res["inputs_sha256"][os.path.relpath(os.path.join(pilot_dir, "games.jsonl"), ROOT)] = \
            sha_file(os.path.join(pilot_dir, "games.jsonl"))
    return res


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--pilot")
    ap.add_argument("--secondary", action="store_true", help="also S1-S6 (§5.3)")
    ap.add_argument("--addendum", action="store_true", help="also Study 2 Addendum A (HA, HB)")
    a = ap.parse_args(argv)
    res = analyze(a.run_dir, a.pilot)
    if a.secondary:
        res["secondary"] = secondary(a.run_dir)
    if a.addendum:
        res["addendum_A"] = addendum_a(a.run_dir)
    out = os.path.join(ROOT, "derived", res["run_id"])
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "primary.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    b = res["bootstrap"]
    print(f"{res['model']}: {res['games_scored']} games scored, {len(res['games_missing'])} missing")
    print(f"  fixtures in the primary test: {res['fixtures_in_test']} of {res['fixtures']}"
          f" ({res['fixtures_left_out']} left out: no opener pair with both games agreed);"
          f" {res['fixtures_using_both_openers']} use both openers")
    print(f"  d = share at $100M minus share at $100:  mean {res['mean_d']:+.4f}, median {res['median_d']:+.4f},"
          f" SD {res['sd_d']:.4f}")
    print(f"  95% CI of the mean {b['ci95_mean'][0]:+.4f} to {b['ci95_mean'][1]:+.4f};"
          f" 90% CI {b['ci90_mean'][0]:+.4f} to {b['ci90_mean'][1]:+.4f}")
    print(f"  H1 Wilcoxon: p = {res['H1']['p_unadjusted']:.4f} (unadjusted; Holm across models comes later),"
          f" W+ = {res['H1']['W_plus']}, n = {res['H1']['n_nonzero']}")
    print(f"  H2 equivalence within ±0.10: {'YES' if res['H2_equivalent_within_0.10'] else 'no'}")
    for sc in ("x1e2", "x1e5", "x1e8"):
        print(f"  endings at {sc}: {res['terminals_by_scale'].get(sc)}")
    if "pilot_sigma" in res:
        ps = res["pilot_sigma"]
        print(f"  pilot spread (A1, reported only): SD of d = {ps['sd_d']:.4f} over {ps['fixtures_with_d']} pilot fixtures")
    if "secondary" in res:
        print("  secondary (§5.3; Holm across the 7 tests with a p-value):")
        for k, v in res["secondary"].items():
            if isinstance(v, dict) and "p_holm" in v:
                print(f"    {k:<26} mean {v['mean']:+.4f}  median {v['median']:+.4f}  n {v['n']:>2}"
                      f"  p {v['p_unadjusted']:.4f}  Holm {v['p_holm']:.4f}")
        sdv = res["secondary"]["S3_sd_across_fixtures"]
        print(f"    S3 across-fixture SD: $100 {sdv['sd_100']:.3f}, $100M {sdv['sd_100M']:.3f},"
              f" difference 95% CI {sdv['diff_ci95'][0]:+.3f} to {sdv['diff_ci95'][1]:+.3f}")
    if "addendum_A" in res:
        ha, hb = res["addendum_A"]["HA"], res["addendum_A"]["HB"]
        print(f"  Addendum A (unadjusted; Holm across the Gemma models comes from combining both runs):")
        print(f"    HA  mean h {ha['mean_h']:+.4f} (95% CI {ha['ci95_mean_h'][0]:+.4f} to {ha['ci95_mean_h'][1]:+.4f}),"
              f" h<0 in {ha['fixtures_h_below_0']}/{ha['n_fixtures']} fixtures, one-sided p {ha['p_one_sided_unadjusted']:.4f}")
        print(f"    HB  mean I {hb['mean_I']:+.4f}, n {hb['n_fixtures']}, two-sided p {hb['p_two_sided_unadjusted']:.4f}")
    print(f"  written: {os.path.relpath(os.path.join(out, 'primary.json'), ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
