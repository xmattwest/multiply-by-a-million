#!/usr/bin/env python3
"""analyze_addendum_b.py: the registered analysis of Addendum B (prereg/addendum_B/ADDENDUM_B.md §4).

    python3 scripts/analyze_addendum_b.py data/runs/addB-<stamp>          writes derived/addB-<stamp>/addendum_B.json
    python3 scripts/analyze_addendum_b.py <run_dir> --out <file>          write elsewhere (preview / tests)

HA per model, mirroring Addendum A with k first moves per cell:
  - gap of one valid OFFER: seller P/F - 1, buyer 1 - P/C (F, C = the role's own limit);
  - cell (fixture, role, scale) mean over its valid OFFERs;
  - h_f = mean over the roles where both scales have a cell mean of
    [mean gap at $100M - mean gap at $100]; fixtures with no such role are left out;
  - one-sided Wilcoxon signed-rank on the h_f (alternative: median < 0), 200,000
    random sign flips, zeros dropped, average ranks for ties (scripts/analyze.py);
  - Holm across the models that ran, alpha 0.05;
  - mean h_f with a 95% percentile bootstrap CI (10,000 resamples of fixtures).
  Seeds: scripts/analyze.py's rule, what = "addendumB-wilcoxon-HA" / "addendumB-bootstrap-HA",
  model = the arm name (e.g. "gpt-6-luna").

Also reported (registered as descriptive): per-role mean gaps by scale; per-scale
counts of replies, valid OFFERs, WALKAWAYs, invalid replies by error code,
refusals and missing items; replies with reasoning tokens; the sensitivity HA
without replies that used reasoning tokens; the returned model IDs, providers and
system fingerprints; spend. A model whose completed replies are more than 25%
invalid or refused gets the flag "format-limited"; a gap of more than 10 points
in the invalid-or-refused share between the two scales gets "scale-dependent
format failures".

Standard library only. Free, offline.
"""
import argparse
import json
import os
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import analyze as A               # noqa: E402

LOW, HIGH = "x1e2", "x1e8"
ROLES = ("seller", "buyer")


def load(run_dir):
    with open(os.path.join(run_dir, "run_manifest.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)
    recs = []
    with open(os.path.join(run_dir, "games.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            recs.append(json.loads(line))
    ids = [r["item_id"] for r in recs]
    if len(ids) != len(set(ids)):
        raise ValueError("games.jsonl has more than one record for an item")
    return manifest, recs


def h_values(recs, keep=lambda r: True):
    """{fixture: h_f} and per-role (low mean, high mean) pairs."""
    cells = {}
    for r in recs:
        if not r["errors"] and r["action"] == "OFFER" and r["gap"] is not None and keep(r):
            cells.setdefault((r["fixture"], r["role"], r["scale"]), []).append(r["gap"])
    h, per_role = {}, {o: [] for o in ROLES}
    for f in sorted({r["fixture"] for r in recs}):
        parts = []
        for o in ROLES:
            lo, hi = cells.get((f, o, LOW)), cells.get((f, o, HIGH))
            if lo and hi:
                lo_m, hi_m = statistics.fmean(lo), statistics.fmean(hi)
                parts.append(hi_m - lo_m)
                per_role[o].append((lo_m, hi_m))
        if parts:
            h[f] = statistics.fmean(parts)
    return h, per_role


def ha_test(h, arm, flips=A.FLIPS, boots=A.BOOT):
    vals = [h[f] for f in sorted(h)]
    p, wplus, n = A.wilcoxon_signflip_less(vals, random.Random(A.seed("addendumB-wilcoxon-HA", arm)), flips)
    ci = A.bootstrap_ci(vals, random.Random(A.seed("addendumB-bootstrap-HA", arm)), levels=(0.95,), boots=boots) \
        if vals else {}
    return {"n_fixtures": len(vals), "n_nonzero": n, "W_plus": wplus, "p_one_sided_unadjusted": p,
            "mean_h": statistics.fmean(vals) if vals else None, "ci95_mean_h": ci.get("95"),
            "fixtures_h_below_0": sum(v < 0 for v in vals), "h_f": {f: h[f] for f in sorted(h)}}


def quality(recs):
    out = {}
    for sc in (LOW, HIGH):
        rs = [r for r in recs if r["scale"] == sc]
        done = [r for r in rs if r["errors"] != ["INFRASTRUCTURE"]]
        bad = [r for r in done if r["errors"]]
        errs = {}
        for r in bad:
            for e in r["errors"]:
                errs[e] = errs.get(e, 0) + 1
        out[sc] = {"items": len(rs), "replies": len(done), "missing_infrastructure": len(rs) - len(done),
                   "valid_offers": sum(1 for r in done if not r["errors"] and r["action"] == "OFFER"),
                   "walkaways": sum(1 for r in done if not r["errors"] and r["action"] == "WALKAWAY"),
                   "refusals": errs.get("REFUSAL", 0), "invalid_by_error": errs,
                   "invalid_or_refused_share": len(bad) / len(done) if done else None,
                   "replies_with_reasoning_tokens": sum(1 for r in done if (r.get("reasoning_tokens") or 0) > 0),
                   "offers_below_own_limit": sum(1 for r in done if not r["errors"] and r["action"] == "OFFER"
                                                 and r["gap"] is not None and r["gap"] < 0)}
    return out


def analyze(run_dir, flips=A.FLIPS, boots=A.BOOT):
    manifest, recs = load(run_dir)
    if "pilot" in manifest["kind"] and not manifest.get("preview"):
        raise ValueError("a real format pilot is never analysed for HA (ADDENDUM_B.md §9)")
    arms = list(manifest["arms"])
    per = {}
    for arm in arms:
        rs = [r for r in recs if r["arm"] == arm]
        if not rs:
            per[arm] = {"status": "not run"}
            continue
        h, per_role = h_values(rs)
        q = quality(rs)
        done = [r for r in rs if r["errors"] != ["INFRASTRUCTURE"]]
        bad_share = sum(1 for r in done if r["errors"]) / len(done) if done else None
        shares = [q[sc]["invalid_or_refused_share"] for sc in (LOW, HIGH)]
        flags = []
        if bad_share is not None and bad_share > 0.25:
            flags.append("format-limited")
        if None not in shares and abs(shares[1] - shares[0]) > 0.10:
            flags.append("scale-dependent format failures")
        h_nr, _ = h_values(rs, keep=lambda r: not (r.get("reasoning_tokens") or 0) > 0)
        per[arm] = {
            "status": "run", "items_planned": sum(1 for i in manifest["items_planned"] if i.startswith(arm + "|")),
            "records": len(rs), "HA": ha_test(h, arm, flips, boots),
            "by_role_mean_gap": {o: {LOW: statistics.fmean(a for a, _ in v) if v else None,
                                     HIGH: statistics.fmean(b for _, b in v) if v else None, "n_fixtures": len(v)}
                                 for o, v in per_role.items()},
            "quality_by_scale": q, "flags": flags,
            "sensitivity_without_reasoning_token_replies": {
                k: v for k, v in ha_test(h_nr, arm, flips, boots).items() if k != "h_f"},
            "models_returned": sorted({str(r.get("model_returned")) for r in done}),
            "model_versions": sorted({str(r.get("model_version")) for r in done}),
            "providers_returned": sorted({str(r.get("provider_returned")) for r in done}),
            "system_fingerprints": sorted({str(r.get("system_fingerprint")) for r in done}),
            "cost_reported": sum(r.get("cost") or 0.0 for r in rs),
        }
    pv = {a: v["HA"]["p_one_sided_unadjusted"] for a, v in per.items() if v["status"] == "run"
          and v["HA"]["n_fixtures"] > 0}
    adj = A.holm(pv) if pv else {}
    for a, v in per.items():
        if v["status"] == "run":
            v["HA"]["p_holm"] = adj.get(a)
            v["HA"]["supported"] = adj.get(a) is not None and adj[a] < 0.05
    sums = {}
    for name in ("run_manifest.json", "games.jsonl", "calls.jsonl", "events.jsonl", "run_summary.json"):
        path = os.path.join(run_dir, name)
        if os.path.exists(path):
            sums[name] = A.sha_file(path)
    label = ("PREVIEW: synthetic data from the fake server; not a result" if manifest.get("preview") else
             "PILOT: format check only; excluded from the test (ADDENDUM_B.md §2)" if "pilot" in manifest["kind"]
             else "registered test (ADDENDUM_B.md)")
    return {"schema": "addendum_B/1", "run_id": manifest["run_id"], "label": label,
            "registered": bool(manifest.get("registered")), "holm_family": sorted(pv), "alpha": 0.05,
            "flips": flips, "bootstrap": boots, "per_model": per, "inputs_sha256": sums}


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    res = analyze(a.run_dir)
    out = a.out or os.path.join(ROOT, "derived", os.path.basename(os.path.normpath(a.run_dir)), "addendum_B.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(res["label"])
    for arm, v in res["per_model"].items():
        if v["status"] != "run":
            print(f"  {arm}: not run")
            continue
        ha = v["HA"]
        ci = ha["ci95_mean_h"] or [float("nan")] * 2
        print(f"  {arm:22s} mean h {ha['mean_h']:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]  "
              f"{ha['fixtures_h_below_0']}/{ha['n_fixtures']} below 0  p {ha['p_one_sided_unadjusted']:.5f}"
              f"  Holm {ha['p_holm']:.5f}  {'SUPPORTED' if ha['supported'] else 'not supported'}"
              + (f"  flags {v['flags']}" if v["flags"] else ""))
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
