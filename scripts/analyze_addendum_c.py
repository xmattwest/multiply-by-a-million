#!/usr/bin/env python3
"""analyze_addendum_c.py: the registered tests of Addendum C (prereg/addendum_C/ADDENDUM_C.md).

    python3 scripts/analyze_addendum_c.py data/runs/addC-<stamp>   writes derived/addC-<stamp>/addendum_C.json

PER MODEL (one record per scored game: the last attempt that was not void)
  Cell = (fixture, opener, scale), up to k = 4 games. Cell means are over the
  cell's games that qualify for the quantity in question.

  H1/H2 (primary, Study 2 §4-5 with cell means):
    d_io = mean seller share at $100M - mean seller share at $100 (agreements),
    d_i  = mean of the d_io that are defined.
    H1: two-sided Wilcoxon signed-rank on the d_i, 200,000 sign flips, Holm
        across the three models. H2: the 90% bootstrap CI of the mean d_i lies
        inside (-0.10, +0.10). Reading: Study 2 §5.4.
  HM (confirmatory, new): the split moves with the midpoint of the two sides'
    first offers. Per game m = (first seller price + first buyer price)/2 as a
    share of the zone (agreements in which both sides offered).
    e_i = mean over openers of [(mean s at $100M - mean s at $100)
                                - (mean m at $100M - mean m at $100)], same games.
    HM1: the 90% bootstrap CI of the mean e_i lies inside (-0.05, +0.05).
    HM2: Spearman correlation of the per-fixture share shift and midpoint
         shift on those games > 0; one-sided permutation test (20,000).
    HM is supported on a model if HM1 and HM2 (HM2 Holm across three) hold.
  Secondary: HA in full games (opening gap, as Addendum B, one-sided, Holm
    across three); HB (opener advantage, as Addendum A, two-sided, Holm across
    three); S1 ($100K vs $100, $100M vs $100K), S2 (agreement rate), S5
    (coarse-price check), S6 (seller utility, every game), Holm within each
    model as Study 2.
  Not tested, reported: the Addendum B-based expectation for d (opener-only
    midpoint, frozen first-move data) next to the observed d; quality counts;
    a sensitivity analysis without games in which any call used reasoning tokens.

Standard library only. Deterministic (seeds from scripts/analyze.py's rule).
"""
import argparse
import json
import os
import random
import statistics
import sys
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import analyze as A                 # noqa: E402

LOW, MID, HIGH = "x1e2", "x1e5", "x1e8"
OPENERS = ("seller", "buyer")
H2_MARGIN = 0.10
HM_MARGIN = 0.05
PERMS = 20_000
ADD_B_RUN = "addB-20261010T191121Z"


def load(run_dir):
    with open(os.path.join(run_dir, "run_manifest.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)
    scored, voids = {}, {}
    with open(os.path.join(run_dir, "games.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["terminal"] == "INFRASTRUCTURE_VOID":
                voids[r["game_id"]] = voids.get(r["game_id"], 0) + 1
                continue
            if r["game_id"] in scored:
                raise ValueError(f"two scored attempts for {r['game_id']}")
            scored[r["game_id"]] = r
    missing = [g for g in manifest["games_planned"] if g not in scored]
    return manifest, scored, voids, missing


def share(r):
    if r["terminal"] != "AGREEMENT":
        return None
    return Fraction(r["seller_share"]["num"], r["seller_share"]["den"])


def midpoint(r):
    if r["terminal"] != "AGREEMENT":
        return None
    s = [p for _, who, p in r["offers"] if who == "seller"]
    b = [p for _, who, p in r["offers"] if who == "buyer"]
    if not s or not b:
        return None
    return (Fraction(s[0] + b[0], 2) - r["F"]) / (r["C"] - r["F"])


def opening_gap(r):
    if not r["offers"] or r["offers"][0][1] != r["opener"]:
        return None
    p = Fraction(r["offers"][0][2])
    return p / r["F"] - 1 if r["opener"] == "seller" else 1 - p / r["C"]


def utility(r):
    u = r.get("u_seller")
    return None if u is None else Fraction(u["num"], u["den"])


def mean(xs):
    xs = list(xs)
    return sum(xs, Fraction(0)) / len(xs) if xs else None


def cells_of(games):
    c = {}
    for r in games:
        c.setdefault((r["fixture"], r["opener"], r["scale"]), []).append(r)
    return c


def paired(cells, fixtures, value, hi=HIGH, lo=LOW, openers=OPENERS):
    """{fixture: mean over openers of [cell mean at hi - cell mean at lo]} for a per-game value."""
    out = {}
    for f in fixtures:
        parts = []
        for o in openers:
            a = mean(v for v in (value(r) for r in cells.get((f, o, hi), [])) if v is not None)
            b = mean(v for v in (value(r) for r in cells.get((f, o, lo), [])) if v is not None)
            if a is not None and b is not None:
                parts.append(a - b)
        if parts:
            out[f] = sum(parts, Fraction(0)) / len(parts)
    return out


def hm_parts(cells, fixtures):
    """Per fixture: (share shift, midpoint shift) on agreements where both sides offered."""
    out = {}
    for f in fixtures:
        ds, ms = [], []
        for o in OPENERS:
            hi = [r for r in cells.get((f, o, HIGH), []) if midpoint(r) is not None]
            lo = [r for r in cells.get((f, o, LOW), []) if midpoint(r) is not None]
            if hi and lo:
                ds.append(mean(share(r) for r in hi) - mean(share(r) for r in lo))
                ms.append(mean(midpoint(r) for r in hi) - mean(midpoint(r) for r in lo))
        if ds:
            out[f] = (sum(ds, Fraction(0)) / len(ds), sum(ms, Fraction(0)) / len(ms))
    return out


def ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    if len(set(rx)) < 2 or len(set(ry)) < 2:
        return None
    return statistics.correlation(rx, ry)


def spearman_perm_greater(x, y, rng, perms=PERMS):
    rho = spearman(x, y)
    if rho is None:
        return None, 1.0
    y = list(y)
    hits = 0
    for _ in range(perms):
        rng.shuffle(y)
        r = spearman(x, y)
        hits += r is not None and r >= rho - 1e-12
    return rho, (hits + 1) / (perms + 1)


def summary(vals, arm, what, levels=(0.95, 0.90)):
    v = [float(x) for x in vals]
    ci = A.bootstrap_ci(v, random.Random(A.seed(f"addendumC-bootstrap-{what}", arm)), levels=levels) if v else {}
    return {"n_fixtures": len(v), "mean": statistics.fmean(v) if v else None,
            "median": statistics.median(v) if v else None, "sd": statistics.stdev(v) if len(v) > 1 else None,
            "ci95": ci.get("95"), "ci90": ci.get("90"), "below_0": sum(x < 0 for x in v),
            "above_0": sum(x > 0 for x in v)}


def reading(sig, equiv):
    return {(True, False): "scale shifts this model's split",
            (False, True): "invariant to within +-0.10",
            (False, False): "inconclusive",
            (True, True): "a real but small shift (below 0.10)"}[(sig, equiv)]


def b_expectation(arm):
    """Opener-only midpoint expectation for d from Addendum B's frozen first moves (not a test)."""
    path = os.path.join(ROOT, "data", "runs", ADD_B_RUN, "games.jsonl")
    fx_path = os.path.join(ROOT, "config", "fixtures.json")
    if not os.path.exists(path):
        return None
    with open(fx_path, encoding="utf-8") as fh:
        fx = {f["id"]: f for f in json.load(fh)["fixtures"]}
    cell = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["arm"] == arm and not r["errors"] and r["action"] == "OFFER":
                cell.setdefault((r["fixture"], r["role"], r["scale"]), []).append(r["gap"])
    out = {}
    for f, F in fx.items():
        fp, cp = F["f_pct"] / 100, F["c_pct"] / 100
        s = {}
        for sc in (LOW, HIGH):
            gs, gb = cell.get((f, "seller", sc)), cell.get((f, "buyer", sc))
            if not gs or not gb:
                break
            mid = (fp * (1 + statistics.fmean(gs)) + cp * (1 - statistics.fmean(gb))) / 2
            s[sc] = (mid - fp) / (cp - fp)
        if len(s) == 2:
            out[f] = s[HIGH] - s[LOW]
    return out


def quality(games, voids):
    out = {}
    for sc in (LOW, MID, HIGH):
        rs = [r for r in games if r["scale"] == sc]
        errs = {}
        for r in rs:
            for d in r["diagnostics"]:
                if d.startswith("INVALID:"):
                    for e in d.split(":", 2)[2].split(","):
                        errs[e] = errs.get(e, 0) + 1
        term = {}
        for r in rs:
            term[r["terminal"]] = term.get(r["terminal"], 0) + 1
        calls = [c for r in rs for c in r["api"]]
        out[sc] = {"games": len(rs), "terminals": term,
                   "forfeit_share": term.get("PROTOCOL_FORFEIT", 0) / len(rs) if rs else None,
                   "invalid_by_error": errs, "refusals": sum(r["refusals"] for r in rs),
                   "calls": len(calls), "calls_with_reasoning_tokens": sum(r["reasoning_token_calls"] for r in rs),
                   "mean_messages": statistics.fmean(r["n_messages"] for r in rs) if rs else None,
                   "violations": sum(1 for r in rs if r.get("violation")),
                   "outside_zopa": sum(1 for r in rs if r.get("outside_zopa"))}
    out["void_attempts"] = sum(voids.values())
    return out


def per_model(games, arm, voids, flips, boots):
    fixtures = sorted({r["fixture"] for r in games})
    cells = cells_of(games)
    res = {"status": "run", "games_scored": len(games)}
    d = paired(cells, fixtures, share)
    res["H1"] = {**summary(d.values(), arm, "H1"), "d_i": {f: float(v) for f, v in d.items()},
                 "fixtures_left_out": [f for f in fixtures if f not in d]}
    p, wp, n = A.wilcoxon_signflip(list(d.values()), random.Random(A.seed("addendumC-wilcoxon-H1", arm)), flips)
    res["H1"].update({"p_two_sided_unadjusted": p, "W_plus": wp, "n_nonzero": n})
    ci90 = res["H1"]["ci90"]
    res["H2_equivalent_within_0.10"] = bool(ci90 and -H2_MARGIN < ci90[0] and ci90[1] < H2_MARGIN)

    parts = hm_parts(cells, fixtures)
    e = {f: ds - ms for f, (ds, ms) in parts.items()}
    s_e = summary(e.values(), arm, "HM1")
    rho, p_rho = spearman_perm_greater([float(ds) for ds, _ in parts.values()],
                                       [float(ms) for _, ms in parts.values()],
                                       random.Random(A.seed("addendumC-perm-HM2", arm)))
    left = sum(1 for r in games if r["terminal"] == "AGREEMENT" and midpoint(r) is None)
    res["HM"] = {"HM1": {**s_e, "margin": HM_MARGIN,
                         "equivalent": bool(s_e["ci90"] and -HM_MARGIN < s_e["ci90"][0] and s_e["ci90"][1] < HM_MARGIN),
                         "equivalent_within_0.10_descriptive":
                             bool(s_e["ci90"] and -0.10 < s_e["ci90"][0] and s_e["ci90"][1] < 0.10),
                         "e_i": {f: float(v) for f, v in e.items()}},
                 "HM2": {"spearman": rho, "p_one_sided_unadjusted": p_rho, "permutations": PERMS},
                 "agreements_without_both_first_offers": left,
                 "midpoint_shift_mean": statistics.fmean(float(ms) for _, ms in parts.values()) if parts else None,
                 "share_shift_mean_same_games": statistics.fmean(float(ds) for ds, _ in parts.values()) if parts else None}

    h = paired(cells, fixtures, opening_gap)
    res["HA"] = summary(h.values(), arm, "HA", levels=(0.95,))
    p, wp, n = A.wilcoxon_signflip_less(list(h.values()), random.Random(A.seed("addendumC-wilcoxon-HA", arm)), flips)
    by_opener = {}
    for o in OPENERS:
        by_opener[o] = {}
        for sc in (LOW, HIGH):
            g = [opening_gap(r) for f in fixtures for r in cells.get((f, o, sc), [])]
            g = [x for x in g if x is not None]
            by_opener[o][sc] = float(mean(g)) if g else None
    res["HA"].update({"p_one_sided_unadjusted": p, "W_plus": wp, "n_nonzero": n, "by_opener_mean_gap": by_opener})

    inter = []
    for f in fixtures:
        m = {}
        for o in OPENERS:
            for sc in (LOW, HIGH):
                m[(o, sc)] = mean(v for v in (share(r) for r in cells.get((f, o, sc), [])) if v is not None)
        if all(v is not None for v in m.values()):
            inter.append((m[("seller", HIGH)] - m[("seller", LOW)]) - (m[("buyer", HIGH)] - m[("buyer", LOW)]))
    p, wp, n = A.wilcoxon_signflip(inter, random.Random(A.seed("addendumC-wilcoxon-HB", arm)), flips)
    res["HB"] = {"n_fixtures": len(inter), "n_nonzero": n, "W_plus": wp, "p_two_sided_unadjusted": p,
                 "mean_I": statistics.fmean(float(x) for x in inter) if inter else None}

    sec = {}
    for name, hi, lo in (("S1a_100K_vs_100", MID, LOW), ("S1b_100M_vs_100K", HIGH, MID)):
        v = paired(cells, fixtures, share, hi=hi, lo=lo)
        p, _, n = A.wilcoxon_signflip(list(v.values()), random.Random(A.seed(f"addendumC-{name}", arm)), flips)
        sec[name] = {**summary(v.values(), arm, name, levels=(0.95,)), "p_unadjusted": p, "n_nonzero": n}
    v = paired(cells, fixtures, lambda r: Fraction(int(r["terminal"] == "AGREEMENT")))
    p, _, n = A.wilcoxon_signflip(list(v.values()), random.Random(A.seed("addendumC-S2", arm)), flips)
    sec["S2_agreement_rate"] = {**summary(v.values(), arm, "S2", levels=(0.95,)), "p_unadjusted": p, "n_nonzero": n}

    def coarse(r):
        s = share(r)
        if s is None or r["scale"] != HIGH:
            return s
        P = round(Fraction(r["price"], 1_000_000))           # round half to even
        F, C = Fraction(r["F"], 1_000_000), Fraction(r["C"], 1_000_000)
        return (P - F) / (C - F)
    v = paired(cells, fixtures, coarse)
    p, _, n = A.wilcoxon_signflip(list(v.values()), random.Random(A.seed("addendumC-S5", arm)), flips)
    sec["S5_coarse_price"] = {**summary(v.values(), arm, "S5", levels=(0.95,)), "p_unadjusted": p, "n_nonzero": n}
    v = paired(cells, fixtures, utility)
    p, _, n = A.wilcoxon_signflip(list(v.values()), random.Random(A.seed("addendumC-S6", arm)), flips)
    sec["S6_seller_utility"] = {**summary(v.values(), arm, "S6", levels=(0.95,)), "p_unadjusted": p, "n_nonzero": n}
    adj = A.holm({k: x["p_unadjusted"] for k, x in sec.items()})
    for k in sec:
        sec[k]["p_holm_within_model"] = adj[k]
    res["secondary"] = sec

    exp = b_expectation(arm)
    if exp:
        common = [f for f in sorted(exp) if f in d]
        res["B_expectation_not_a_test"] = {
            "mean_expected_d": statistics.fmean(exp.values()), "fixtures_expected_below_0": sum(v < 0 for v in exp.values()),
            "mean_observed_d": res["H1"]["mean"],
            "fixture_spearman_expected_vs_observed": spearman([exp[f] for f in common], [float(d[f]) for f in common])
            if len(common) > 2 else None}

    clean = [r for r in games if r["reasoning_token_calls"] == 0]
    if len(clean) < len(games):
        dc = paired(cells_of(clean), fixtures, share)
        res["sensitivity_without_reasoning_token_games"] = {"games_dropped": len(games) - len(clean),
                                                           **summary(dc.values(), arm, "H1-noreason", levels=(0.90,))}
    q = quality(games, voids)
    res["quality_by_scale"] = q
    flags = []
    total_forfeits = sum(q[sc]["terminals"].get("PROTOCOL_FORFEIT", 0) for sc in (LOW, MID, HIGH))
    if games and total_forfeits / len(games) > 0.25:
        flags.append("format-limited")
    fl, fh = q[LOW]["forfeit_share"], q[HIGH]["forfeit_share"]
    if fl is not None and fh is not None and abs(fh - fl) > 0.10:
        flags.append("scale-dependent forfeits")
    res["flags"] = flags
    calls = [c for r in games for c in r["api"] if c["kind"] == "model_output"]
    res["models_returned"] = sorted({str(c.get("model_returned")) for c in calls})
    res["model_versions"] = sorted({str(c.get("model_version")) for c in calls})
    res["providers_returned"] = sorted({str(c.get("provider_returned")) for c in calls})
    res["system_fingerprints"] = sorted({str(c.get("system_fingerprint")) for c in calls})
    res["cost_reported"] = sum(r["cost"] for r in games)
    return res


def analyze(run_dir, flips=A.FLIPS):
    manifest, scored, voids, missing = load(run_dir)
    if "pilot" in manifest["kind"] and not manifest.get("preview"):
        raise ValueError("a real format pilot is never analysed (ADDENDUM_C.md §9)")
    per = {}
    for arm in manifest["arms"]:
        gs = [r for r in scored.values() if r["arm"] == arm]
        if not gs:
            per[arm] = {"status": "not run"}
            continue
        per[arm] = per_model(gs, arm, {k: v for k, v in voids.items() if k.startswith(arm + "|")}, flips, A.BOOT)
        per[arm]["games_planned"] = sum(1 for g in manifest["games_planned"] if g.startswith(arm + "|"))
        per[arm]["games_missing"] = [g for g in missing if g.startswith(arm + "|")]
    run = [a for a, v in per.items() if v["status"] == "run"]
    for key, path in (("H1", ("H1", "p_two_sided_unadjusted")), ("HM2", ("HM", "HM2", "p_one_sided_unadjusted")),
                      ("HA", ("HA", "p_one_sided_unadjusted")), ("HB", ("HB", "p_two_sided_unadjusted"))):
        def get(v, p=path):
            for k in p:
                v = v[k]
            return v
        adj = A.holm({a: get(per[a]) for a in run})
        for a in run:
            node = per[a]
            for k in path[:-1]:
                node = node[k]
            node["p_holm"] = adj[a]
    for a in run:
        v = per[a]
        sig = v["H1"]["p_holm"] < 0.05
        v["reading"] = reading(sig, v["H2_equivalent_within_0.10"])
        v["HM"]["supported"] = v["HM"]["HM1"]["equivalent"] and v["HM"]["HM2"]["p_holm"] < 0.05
        v["HA"]["supported"] = v["HA"]["p_holm"] < 0.05
    sums = {name: A.sha_file(os.path.join(run_dir, name))
            for name in ("run_manifest.json", "games.jsonl", "calls.jsonl", "events.jsonl", "run_summary.json")
            if os.path.exists(os.path.join(run_dir, name))}
    label = ("PREVIEW: synthetic data from the fake server; not a result" if manifest.get("preview") else
             "registered test (ADDENDUM_C.md)")
    return {"schema": "addendum_C/1", "run_id": manifest["run_id"], "label": label,
            "registered": bool(manifest.get("registered")), "holm_family": run, "alpha": 0.05,
            "h2_margin": H2_MARGIN, "hm_margin": HM_MARGIN, "flips": flips, "bootstrap": A.BOOT,
            "per_model": per, "inputs_sha256": sums}


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--flips", type=int, default=A.FLIPS, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    res = analyze(a.run_dir, flips=a.flips)
    out = a.out or os.path.join(ROOT, "derived", os.path.basename(os.path.normpath(a.run_dir)), "addendum_C.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(res["label"])
    for arm, v in res["per_model"].items():
        if v["status"] != "run":
            print(f"  {arm}: not run")
            continue
        h1, hm = v["H1"], v["HM"]
        print(f"  {arm:22s} d {h1['mean']:+.3f} 95% {[round(x, 3) for x in h1['ci95']]} p_holm {h1['p_holm']:.4f}"
              f" -> {v['reading']}; HM e {hm['HM1']['mean']:+.3f} 90% {[round(x, 3) for x in hm['HM1']['ci90']]},"
              f" rho {hm['HM2']['spearman']:+.2f} -> {'supported' if hm['supported'] else 'not supported'};"
              f" HA h {v['HA']['mean']:+.3f}; flags {v['flags'] or 'none'}")
    print(f"  written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
