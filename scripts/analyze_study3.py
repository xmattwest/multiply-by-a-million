#!/usr/bin/env python3
"""analyze_study3.py: Study 3's registered tests (prereg/study3/PREREGISTRATION.md §5).

    python3 scripts/analyze_study3.py data/runs/s3-<...> [more run dirs]    writes derived/study3/study3.json

Pass every session run (both backends, both sessions). Games are merged by arm; each game counts once
(its last attempt that was not void). Cell = (fixture, opener, cell code), k <= 4 games; cell means are
over the cell's games that qualify for the quantity. Fixture-level statistics average over openers whose
cells are defined. Wilcoxon p from 200,000 sign flips; bootstrap CIs resample fixtures 10,000 times;
seeds from scripts/analyze.py's rule with what = "study3-<test>". Holm across the five models within each
test unless stated.

PRIMARY
  P1  I_f = h_f(A1) - h_f(A0); h_f(A) = mean over openers of [mean opener gap at A-$100M - at A-$100],
      gap = P1/F - 1 (seller) or 1 - P1/C (buyer). One-sided Wilcoxon, I > 0.
  P2  per condition A0, A1: d_i = mean over openers of [mean seller share at $100M - at $100], agreements.
      H1 two-sided Wilcoxon (Holm across models within condition); H2 90% CI inside +-0.10.
  P3  clamp cells s(G, O) (agreements). Per opener with all four cells:
        kappa = mean over O of [s($100M, O) - s($100, O)]   (concession channel)
        omega = mean over G of [s(G, O=$100M) - s(G, O=$100)] (opening channel)
      P3a: kappa two-sided Wilcoxon (Holm) and equivalence (90% CI inside +-0.05).
      P3b: omega - delta_m, delta_m = scripted midpoint shift (O=$100M minus O=$100, zone units);
           equivalence of the mean (90% CI inside +-0.05).
SECONDARY (labelled secondary): S1 HA in A0 (one-sided, Holm) and h(A1) equivalence +-0.03; S2 adjacent
  scales and S3 agreement rate, all-game seller utility, coarse-grid prices, both conditions (Holm within
  model); S4 HB per condition (two-sided, Holm); S5 HM on A0 (as Addendum C); S6 scripting artefact
  (diagonal clamp cells vs A0 games at the same scale); S7 session consistency; S8 consistency with
  Studies 1-2 and Addendum C; S9 appraisal position (pre-declared moderator).

Standard library only. Deterministic.
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
import analyze_addendum_c as AC     # noqa: E402
import study3 as S                  # noqa: E402

LOW, MID, HIGH = S.LOW, S.MID, S.HIGH
H2_MARGIN, P3_MARGIN, HA_MARGIN = 0.10, 0.05, 0.03
PERMS = 20_000
EARLIER = {   # S8: (run dir, arm filter or None, prompt)
    "qwen3.8-27b-fp8": {"A0": ("data/runs/s2-qwen3.8-27b-fp8-20261003T223042Z", None),
                        "A1_vs_study1_appraisal_and_ranges": ("data/runs/qwen3.8-27b-fp8-20261003T180835Z", None)},
    "gemma-4-31b-it": {"A0": ("data/runs/s2-gemma-4-31b-it-20261004T012932Z", None)},
    "gpt-6-luna": {"A0": ("data/runs/addC-20261011T182548Z", "gpt-6-luna")},
    "claude-haiku-5.5": {"A0": ("data/runs/addC-20261011T182548Z", "claude-haiku-5.5")},
    "gemini-3.5-flash-lite": {"A0": ("data/runs/addC-20261011T182548Z", "gemini-3.5-flash-lite")},
}

share, opening_gap, utility, midpoint = AC.share, AC.opening_gap, AC.utility, AC.midpoint


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs, Fraction(0)) / len(xs) if xs else None


def rng(what, arm):
    return random.Random(A.seed(f"study3-{what}", arm))


def summary(vals, arm, what, levels=(0.95, 0.90)):
    v = [float(x) for x in vals]
    ci = A.bootstrap_ci(v, rng(f"bootstrap-{what}", arm), levels=levels) if v else {}
    return {"n_fixtures": len(v), "mean": statistics.fmean(v) if v else None,
            "median": statistics.median(v) if v else None, "sd": statistics.stdev(v) if len(v) > 1 else None,
            "ci95": ci.get("95"), "ci90": ci.get("90"), "below_0": sum(x < 0 for x in v),
            "above_0": sum(x > 0 for x in v)}


def inside(ci, margin):
    return bool(ci and -margin < ci[0] and ci[1] < margin)


def load(run_dirs):
    games, missing, manifests = {}, [], []
    for d in run_dirs:
        man, scored, voids, miss = AC.load(d)
        if man.get("study") != 3:
            raise ValueError(f"{d}: not a Study 3 run")
        if man["kind"] == "study3_pilot" and not man.get("preview"):
            raise ValueError(f"{d}: a real format pilot is never analysed (§6)")
        manifests.append(man)
        for gid, r in scored.items():
            if gid in games:
                raise ValueError(f"{gid} appears in two runs")
            r["_session"] = man.get("session")
            games[gid] = r
        missing += miss
    return games, missing, manifests


def cells_of(games):
    c = {}
    for r in games:
        c.setdefault((r["fixture"], r["opener"], r["cell"]), []).append(r)
    return c


def cell_mean(cells, f, o, cell, value):
    return mean(value(r) for r in cells.get((f, o, cell), []))


def contrast(cells, fixtures, value, hi, lo):
    """{fixture: mean over openers of [cell mean at hi - at lo]}."""
    out = {}
    for f in fixtures:
        parts = []
        for o in S.OPENERS:
            a, b = cell_mean(cells, f, o, hi, value), cell_mean(cells, f, o, lo, value)
            if a is not None and b is not None:
                parts.append(a - b)
        if parts:
            out[f] = sum(parts, Fraction(0)) / len(parts)
    return out


def wil2(vals, arm, what, flips):
    p, wp, n = A.wilcoxon_signflip(list(vals), rng(f"wilcoxon-{what}", arm), flips)
    return {"p_two_sided_unadjusted": p, "W_plus": wp, "n_nonzero": n}


def wil_less(vals, arm, what, flips):
    p, wp, n = A.wilcoxon_signflip_less(list(vals), rng(f"wilcoxon-{what}", arm), flips)
    return {"p_one_sided_unadjusted": p, "W_plus": wp, "n_nonzero": n}


def scripted_midpoint(game_like, values):
    p1, p2 = S.clamp_prices(game_like, values)
    ps, pb = (p1, p2) if game_like["opener"] == "seller" else (p2, p1)
    return (Fraction(ps + pb, 2) - game_like["F"]) / (game_like["C"] - game_like["F"])


def clamp_parts(cells, fixtures, values, arm, fx_meta):
    """Per fixture: (kappa, omega, delta_m), mean over openers that have all four clamp cells."""
    out = {}
    for f in fixtures:
        ks, ws, dms = [], [], []
        for o in S.OPENERS:
            s = {(g, src): cell_mean(cells, f, o, f"K-{g}-o{src[1:]}", share) for g in (LOW, HIGH) for src in (LOW, HIGH)}
            if any(v is None for v in s.values()):
                continue
            ks.append(((s[(HIGH, LOW)] - s[(LOW, LOW)]) + (s[(HIGH, HIGH)] - s[(LOW, HIGH)])) / 2)
            ws.append(((s[(LOW, HIGH)] - s[(LOW, LOW)]) + (s[(HIGH, HIGH)] - s[(HIGH, LOW)])) / 2)
            dm = []
            for g in (LOW, HIGH):
                m = {src: scripted_midpoint(S.make_game(arm, fx_meta[f], f"K-{g}-o{src[1:]}", o, 1), values)
                     for src in (LOW, HIGH)}
                dm.append(m[HIGH] - m[LOW])
            dms.append(sum(dm, Fraction(0)) / 2)
        if ks:
            out[f] = (sum(ks, Fraction(0)) / len(ks), sum(ws, Fraction(0)) / len(ws), sum(dms, Fraction(0)) / len(dms))
    return out


def hb(cells, fixtures, cond):
    inter = []
    for f in fixtures:
        m = {(o, sc): cell_mean(cells, f, o, f"{cond}-{sc}", share) for o in S.OPENERS for sc in (LOW, HIGH)}
        if all(v is not None for v in m.values()):
            inter.append((m[("seller", HIGH)] - m[("seller", LOW)]) - (m[("buyer", HIGH)] - m[("buyer", LOW)]))
    return inter


def coarse(r):
    s = share(r)
    if s is None or r["scale"] != HIGH:
        return s
    P = round(Fraction(r["price"], 1_000_000))
    F, C = Fraction(r["F"], 1_000_000), Fraction(r["C"], 1_000_000)
    return (P - F) / (C - F)


def agree(r):
    return Fraction(int(r["terminal"] == "AGREEMENT"))


def earlier_d(run, arm_filter):
    _, scored, _, _ = AC.load(os.path.join(ROOT, run))
    gs = [r for r in scored.values() if arm_filter is None or r.get("arm") == arm_filter]
    fx = sorted({r["fixture"] for r in gs})
    return [float(x) for x in AC.paired(AC.cells_of(gs), fx, AC.share).values()]


def boot_diff(a, b, arm, what):
    """Unpaired fixture bootstrap of mean(a) - mean(b), 95%."""
    r = rng(f"bootstrap-{what}", arm)
    ds = sorted(statistics.fmean(r.choice(a) for _ in a) - statistics.fmean(r.choice(b) for _ in b)
                for _ in range(A.BOOT))
    return {"diff": statistics.fmean(a) - statistics.fmean(b),
            "ci95": [ds[int(0.025 * A.BOOT)], ds[int(0.975 * A.BOOT) - 1]], "n_new": len(a), "n_earlier": len(b)}


def per_fixture_primaries(cells, fixtures, values, arm, fx_meta):
    ha = {c: contrast(cells, fixtures, opening_gap, f"{c}-{HIGH}", f"{c}-{LOW}") for c in ("A0", "A1")}
    inter = {f: ha["A1"][f] - ha["A0"][f] for f in fixtures if f in ha["A1"] and f in ha["A0"]}
    d = {c: contrast(cells, fixtures, share, f"{c}-{HIGH}", f"{c}-{LOW}") for c in ("A0", "A1")}
    cp = clamp_parts(cells, fixtures, values, arm, fx_meta)
    return {"P1_I": inter, "P2_d_A0": d["A0"], "P2_d_A1": d["A1"], "h_A0": ha["A0"], "h_A1": ha["A1"],
            "P3a_kappa": {f: v[0] for f, v in cp.items()}, "P3b_omega_minus_dm": {f: v[1] - v[2] for f, v in cp.items()},
            "omega": {f: v[1] for f, v in cp.items()}, "delta_m": {f: v[2] for f, v in cp.items()}}


def per_model(games, arm, values, fx_meta, flips):
    fixtures = sorted({r["fixture"] for r in games})
    cells = cells_of(games)
    pf = per_fixture_primaries(cells, fixtures, values, arm, fx_meta)
    res = {"status": "run", "games_scored": len(games)}

    res["P1"] = {**summary(pf["P1_I"].values(), arm, "P1"), "I_f": {f: float(v) for f, v in pf["P1_I"].items()},
                 **wil_less([-x for x in pf["P1_I"].values()], arm, "P1", flips),
                 "note": "one-sided test of I > 0 (computed as the 'less' test on -I)",
                 "h_A0_mean": statistics.fmean(float(x) for x in pf["h_A0"].values()) if pf["h_A0"] else None,
                 "h_A1_mean": statistics.fmean(float(x) for x in pf["h_A1"].values()) if pf["h_A1"] else None}
    res["P2"] = {}
    for c in ("A0", "A1"):
        d = pf[f"P2_d_{c}"]
        node = {**summary(d.values(), arm, f"P2-{c}"), "d_i": {f: float(v) for f, v in d.items()},
                "fixtures_left_out": [f for f in fixtures if f not in d], **wil2(d.values(), arm, f"P2-{c}", flips)}
        node["H2_equivalent_within_0.10"] = inside(node["ci90"], H2_MARGIN)
        res["P2"][c] = node
    k, wd = pf["P3a_kappa"], pf["P3b_omega_minus_dm"]
    res["P3"] = {"P3a_kappa": {**summary(k.values(), arm, "P3a"), **wil2(k.values(), arm, "P3a", flips),
                               "kappa_f": {f: float(v) for f, v in k.items()}},
                 "P3b_omega_minus_delta_m": {**summary(wd.values(), arm, "P3b"),
                                             "values_f": {f: float(v) for f, v in wd.items()}},
                 "omega_mean": statistics.fmean(float(x) for x in pf["omega"].values()) if pf["omega"] else None,
                 "delta_m_mean": statistics.fmean(float(x) for x in pf["delta_m"].values()) if pf["delta_m"] else None,
                 "margin": P3_MARGIN, "fixtures_left_out": [f for f in fixtures if f not in k]}
    res["P3"]["P3a_kappa"]["equivalent_within_0.05"] = inside(res["P3"]["P3a_kappa"]["ci90"], P3_MARGIN)
    res["P3"]["P3b_omega_minus_delta_m"]["equivalent_within_0.05"] = inside(
        res["P3"]["P3b_omega_minus_delta_m"]["ci90"], P3_MARGIN)

    sec = {}
    sec["S1_HA_A0"] = {**summary(pf["h_A0"].values(), arm, "S1-A0", levels=(0.95,)),
                       **wil_less(pf["h_A0"].values(), arm, "S1-A0", flips)}
    s1 = summary(pf["h_A1"].values(), arm, "S1-A1")
    sec["S1_h_A1_equivalence"] = {**s1, "margin": HA_MARGIN, "equivalent_within_0.03": inside(s1["ci90"], HA_MARGIN)}
    fam = {}
    for c in ("A0", "A1"):
        for name, hi, lo, val in ((f"S2a_100K_vs_100_{c}", MID, LOW, share), (f"S2b_100M_vs_100K_{c}", HIGH, MID, share),
                                  (f"S3_agreement_rate_{c}", HIGH, LOW, agree), (f"S3_seller_utility_{c}", HIGH, LOW, utility),
                                  (f"S3_coarse_price_{c}", HIGH, LOW, coarse)):
            vals = contrast(cells, fixtures, val, f"{c}-{hi}", f"{c}-{lo}")
            fam[name] = {**summary(vals.values(), arm, name, levels=(0.95,)), **wil2(vals.values(), arm, name, flips)}
    adj = A.holm({k_: v["p_two_sided_unadjusted"] for k_, v in fam.items()})
    for k_ in fam:
        fam[k_]["p_holm_within_model"] = adj[k_]
    sec["S2_S3_within_model_family"] = fam
    for c in ("A0", "A1"):
        inter = hb(cells, fixtures, c)
        sec[f"S4_HB_{c}"] = {"n_fixtures": len(inter), "mean_I": statistics.fmean(float(x) for x in inter) if inter else None,
                             **wil2(inter, arm, f"S4-{c}", flips)}
    parts = {}
    for f in fixtures:
        ds, ms = [], []
        for o in S.OPENERS:
            hi = [r for r in cells.get((f, o, f"A0-{HIGH}"), []) if midpoint(r) is not None]
            lo = [r for r in cells.get((f, o, f"A0-{LOW}"), []) if midpoint(r) is not None]
            if hi and lo:
                ds.append(mean(share(r) for r in hi) - mean(share(r) for r in lo))
                ms.append(mean(midpoint(r) for r in hi) - mean(midpoint(r) for r in lo))
        if ds:
            parts[f] = (sum(ds, Fraction(0)) / len(ds), sum(ms, Fraction(0)) / len(ms))
    e = [ds - ms for ds, ms in parts.values()]
    s_e = summary(e, arm, "S5-HM1")
    rho, p_rho = AC.spearman_perm_greater([float(ds) for ds, _ in parts.values()], [float(ms) for _, ms in parts.values()],
                                          rng("perm-S5-HM2", arm), PERMS)
    sec["S5_HM_A0"] = {"HM1": {**s_e, "equivalent_within_0.05": inside(s_e["ci90"], 0.05)},
                       "HM2": {"spearman": rho, "p_one_sided_unadjusted": p_rho, "permutations": PERMS}}
    for g in (LOW, HIGH):
        vals = contrast(cells, fixtures, share, f"K-{g}-o{g[1:]}", f"A0-{g}")
        sec[f"S6_scripting_artefact_{g}"] = {**summary(vals.values(), arm, f"S6-{g}"), **wil2(vals.values(), arm, f"S6-{g}", flips)}

    s7 = {}
    for sess in (1, 2):
        gs = [r for r in games if r["_session"] == sess]
        if gs:
            s7[sess] = per_fixture_primaries(cells_of(gs), sorted({r["fixture"] for r in gs}), values, arm, fx_meta)
    if len(s7) == 2:
        out7 = {}
        for key in ("P1_I", "P2_d_A0", "P2_d_A1", "P3a_kappa", "P3b_omega_minus_dm"):
            common = [f for f in s7[1][key] if f in s7[2][key]]
            diff = [s7[2][key][f] - s7[1][key][f] for f in common]
            out7[key] = {"session1_mean": statistics.fmean(float(x) for x in s7[1][key].values()) if s7[1][key] else None,
                         "session2_mean": statistics.fmean(float(x) for x in s7[2][key].values()) if s7[2][key] else None,
                         "paired_difference": summary(diff, arm, f"S7-{key}", levels=(0.95,))}
        sec["S7_session_consistency"] = out7
    else:
        sec["S7_session_consistency"] = {"status": f"sessions present: {sorted(s7)}"}

    s8 = {}
    for label, (run, filt) in EARLIER.get(arm, {}).items():
        if os.path.exists(os.path.join(ROOT, run, "games.jsonl")):
            old = earlier_d(run, filt)
            new = [float(x) for x in pf["P2_d_A1" if label.startswith("A1") else "P2_d_A0"].values()]
            if old and new:
                s8[label] = {"earlier_run": run, "earlier_mean_d": statistics.fmean(old), **boot_diff(new, old, arm, f"S8-{label}")}
    sec["S8_consistency_with_earlier_descriptive"] = s8

    pos = {f: float(S.appraisal_position(fx_meta[f])) for f in fixtures}
    s9 = {}
    for sc in (LOW, MID, HIGH):
        lv = {f: mean(cell_mean(cells, f, o, f"A1-{sc}", share) for o in S.OPENERS) for f in fixtures}
        fs = [f for f in fixtures if lv[f] is not None]
        if len(fs) > 2:
            rho, p = AC.spearman_perm_greater([pos[f] for f in fs], [float(lv[f]) for f in fs], rng(f"perm-S9-{sc}", arm), PERMS)
            s9[f"position_vs_share_A1_{sc}"] = {"spearman": rho, "p_one_sided_unadjusted": p, "n": len(fs)}
    d1 = pf["P2_d_A1"]
    if len(d1) > 2:
        fs = sorted(d1)
        s9["position_vs_d_A1_spearman"] = AC.spearman([pos[f] for f in fs], [float(d1[f]) for f in fs])
    sec["S9_appraisal_position_moderator"] = s9
    res["secondary"] = sec

    q = {}
    for cell in S.CELLS:
        rs = [r for r in games if r["cell"] == cell]
        term = {}
        for r in rs:
            term[r["terminal"]] = term.get(r["terminal"], 0) + 1
        q[cell] = {"games": len(rs), "terminals": term,
                   "forfeit_share": term.get("PROTOCOL_FORFEIT", 0) / len(rs) if rs else None,
                   "outside_zopa": sum(1 for r in rs if r.get("outside_zopa")),
                   "mean_messages": statistics.fmean(r["n_messages"] for r in rs) if rs else None,
                   "reasoning_token_calls": sum(r["reasoning_token_calls"] for r in rs),
                   "reasoning_text_calls": sum(r["reasoning_text_calls"] for r in rs),
                   "trailing_text_calls": sum(r.get("trailing_text_calls", 0) for r in rs),
                   "refusals": sum(r["refusals"] for r in rs)}
    res["quality_by_cell"] = q
    flags = []
    fz = sum(1 for r in games if r["terminal"] == "PROTOCOL_FORFEIT")
    if games and fz / len(games) > 0.25:
        flags.append("format-limited")
    for c in ("A0", "A1"):
        lo, hi = q[f"{c}-{LOW}"]["forfeit_share"], q[f"{c}-{HIGH}"]["forfeit_share"]
        if lo is not None and hi is not None and abs(hi - lo) > 0.10:
            flags.append(f"scale-dependent forfeits ({c})")
    res["flags"] = flags
    res["cost_reported"] = sum(r.get("cost") or 0 for r in games)
    if games and games[0]["backend"] == "api":
        calls = [c for r in games for c in r["calls"] if c["kind"] == "model_output"]
        res["model_versions"] = sorted({str(c.get("model_version")) for c in calls})
        res["providers_returned"] = sorted({str(c.get("provider_returned")) for c in calls})
    return res


def analyze(run_dirs, flips=A.FLIPS):
    games, missing, mans = load(run_dirs)
    values = mans[0]["clamp_values"]
    if any(m["clamp_values"] != values for m in mans):
        raise ValueError("runs differ in their clamp values")
    fx_meta = {}
    for m in mans:
        for f in (m.get("fixture_meta") or {}).get("fixtures", []):
            fx_meta[f["id"]] = f
    for r in games.values():
        fx_meta.setdefault(r["fixture"], {"id": r["fixture"], "f_pct": r["f_pct"], "c_pct": r["c_pct"]})
    per = {}
    for arm in S.ARMS:
        gs = [r for r in games.values() if r["arm"] == arm]
        if not gs:
            per[arm] = {"status": "not run"}
            continue
        per[arm] = per_model(gs, arm, values, fx_meta, flips)
        per[arm]["games_missing"] = [g for g in missing if g.startswith(arm + "|")]
    run = [a for a, v in per.items() if v["status"] == "run"]
    paths = {"P1": ("P1", "p_one_sided_unadjusted"),
             "P2_A0": ("P2", "A0", "p_two_sided_unadjusted"), "P2_A1": ("P2", "A1", "p_two_sided_unadjusted"),
             "P3a": ("P3", "P3a_kappa", "p_two_sided_unadjusted"),
             "S1_HA_A0": ("secondary", "S1_HA_A0", "p_one_sided_unadjusted"),
             "S4_HB_A0": ("secondary", "S4_HB_A0", "p_two_sided_unadjusted"),
             "S4_HB_A1": ("secondary", "S4_HB_A1", "p_two_sided_unadjusted"),
             "S5_HM2": ("secondary", "S5_HM_A0", "HM2", "p_one_sided_unadjusted")}
    for key, path in paths.items():
        def node(v, p=path):
            for k in p[:-1]:
                v = v[k]
            return v
        pv = {a: node(per[a])[path[-1]] for a in run if node(per[a])[path[-1]] is not None}
        adj = A.holm(pv)
        for a in run:
            node(per[a])["p_holm_across_models"] = adj.get(a)
    for a in run:
        v = per[a]
        sig = lambda node: node.get("p_holm_across_models") is not None and node["p_holm_across_models"] < 0.05  # noqa: E731
        v["P1"]["supported"] = sig(v["P1"])
        for c in ("A0", "A1"):
            n = v["P2"][c]
            n["reading"] = AC.reading(sig(n), n["H2_equivalent_within_0.10"])
        k = v["P3"]["P3a_kappa"]
        k["reading"] = ("concession channel non-zero" if sig(k) and not k["equivalent_within_0.05"]
                        else "concession channel within +-0.05" if k["equivalent_within_0.05"] and not sig(k)
                        else "small but detectable concession channel" if k["equivalent_within_0.05"] else "inconclusive")
        hm = v["secondary"]["S5_HM_A0"]
        hm["supported"] = hm["HM1"]["equivalent_within_0.05"] and sig(hm["HM2"])
    sums = {}
    for d in run_dirs:
        for name in ("run_manifest.json", "games.jsonl", "calls.jsonl", "events.jsonl", "run_summary.json"):
            p = os.path.join(d, name)
            if os.path.exists(p):
                sums[f"{os.path.basename(os.path.normpath(d))}/{name}"] = A.sha_file(p)
    preview = any(m.get("preview") for m in mans)
    registered = all(m.get("registered") for m in mans)
    label = ("PREVIEW: synthetic data from fake servers; not a result" if preview else
             "registered test (prereg/study3/PREREGISTRATION.md)" if registered else "NOT a registered run")
    return {"schema": "study3/1", "runs": [m["run_id"] for m in mans], "label": label, "registered": registered,
            "holm_family": run, "alpha": 0.05, "margins": {"H2": H2_MARGIN, "P3": P3_MARGIN, "S1_A1": HA_MARGIN},
            "flips": flips, "bootstrap": A.BOOT, "per_model": per, "inputs_sha256": sums}


def fmt(x, nd=3):
    return "n/a" if x is None else f"{x:+.{nd}f}"


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--flips", type=int, default=A.FLIPS, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    res = analyze(a.run_dirs, flips=a.flips)
    out = a.out or os.path.join(ROOT, "derived", "study3", "study3.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(res["label"])
    for arm, v in res["per_model"].items():
        if v["status"] != "run":
            print(f"  {arm}: not run")
            continue
        p1, p2, p3 = v["P1"], v["P2"], v["P3"]
        print(f"  {arm:22s} P1 I {fmt(p1['mean'])} (h A0 {fmt(p1['h_A0_mean'])}, A1 {fmt(p1['h_A1_mean'])})"
              f" {'supported' if p1['supported'] else 'not supported'}")
        for c in ("A0", "A1"):
            print(f"  {'':22s} P2 {c} d {fmt(p2[c]['mean'])} 90% {[round(x, 3) for x in p2[c]['ci90'] or []]} -> {p2[c]['reading']}")
        print(f"  {'':22s} P3 kappa {fmt(p3['P3a_kappa']['mean'])} -> {p3['P3a_kappa']['reading']};"
              f" omega {fmt(p3['omega_mean'])} vs delta_m {fmt(p3['delta_m_mean'])},"
              f" within 0.05: {p3['P3b_omega_minus_delta_m']['equivalent_within_0.05']}; flags {v['flags'] or 'none'}")
    print(f"  written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
