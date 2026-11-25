#!/usr/bin/env python3
"""run_e6b_pod.py (LAPTOP-SIDE): Drive E6b BOS/covariance-matched random control on Gemma 4 31B interp pod.

Connects to an already-launched `--mode interp --slot 3` pod on 127.0.0.1:<port>, runs:
  1. e6_build (19 Study 3 seller fixtures + non-money pairs)
  2. e6_dirs (L42: vR, vS, u, rand)
  3. e6b_dirs (L42: rand_cov, rand_orth)
  4. e6_steer across all 6 E6 Part B families (nego, risk, compare, percent, sum, fermi)
     comparing vR, u, rand (isotropic), rand_cov, and rand_orth at alpha=+1 and +2 (and -2 on fermi).
Writes derived/addendum_d/e6b_results.json and prints summary metrics.
"""
import argparse
import json
import math
import os
import statistics as st
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import interp_e6_laptop as X6     # noqa: E402
import launch_interp as li        # noqa: E402


def http_json(port, method, path, body=None, timeout=600):
    url = f"http://127.0.0.1:{port}{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_interp(port, timeout_s=900):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            r = http_json(port, "GET", "/health", timeout=5)
            if r.get("ok") and r.get("info", {}).get("usable"):
                print(f"[interp:{port}] ready and usable!", flush=True)
                return r["info"]
        except Exception:
            pass
        time.sleep(5)
    raise RuntimeError(f"interp pod on port {port} not ready within {timeout_s}s")


def run_task(port, task, items, settings, **kw):
    body = {"task": task, "items": items, "settings": settings, **kw}
    try:
        return http_json(port, "POST", "/run", body, timeout=600)["results"]
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} on {task}: {err_body[:1000]}") from None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18003)
    ap.add_argument("--out", default=os.path.join(ROOT, "derived", "addendum_d", "e6b_results.json"))
    args = ap.parse_args()

    info = wait_interp(args.port)
    settings = json.load(open(os.path.join(ROOT, "prereg", "settings.json"), encoding="utf-8"))
    import interp_e4_laptop as X4
    import interp_e5_laptop as X5
    X6._LI[:] = [li]
    fx = X5.build_items("seller")
    nonmoney = [v for v in li.build_e3_items(settings, 1)[0] if v["kind"] == "nonmoney"]
    fams = X6.family_items(fx, None)
    b_kw = X6.kws(X6.layer_plan(X4.CANON))["build"]

    print("[e6b] running e6_build...", flush=True)
    b_items = [dict(x, e6_step="e6 build") for x in nonmoney + fx]
    b_res = []
    for i in range(0, len(b_items), 5):
        b_res.extend(run_task(args.port, "e6_build", b_items[i : i + 5], settings, **b_kw))
    print(f"[e6b] e6_build done: {len(b_res)} items", flush=True)

    print("[e6b] running e6_dirs + e6b_dirs at L42...", flush=True)
    d_res = run_task(args.port, "e6_dirs", [{"item_id": "dirs"}], settings, layers=[42], role="seller")
    db_res = run_task(args.port, "e6b_dirs", [{"item_id": "e6b-dirs"}], settings, layers=[42], role="seller", k_fmt=8)
    print("[e6b] dirs:", d_res, db_res, flush=True)

    plan = [
        ["vR", 42, 1.0], ["vR", 42, 2.0],
        ["u", 42, 2.0],
        ["rand", 42, 1.0], ["rand", 42, 2.0],
        ["rand_cov", 42, 1.0], ["rand_cov", 42, 2.0],
        ["rand_orth", 42, 1.0], ["rand_orth", 42, 2.0],
        ["vR", 42, -2.0], ["rand_cov", 42, -2.0], ["rand_orth", 42, -2.0],
    ]

    all_recs = {}
    summary = {"dirs": d_res, "e6b_dirs": db_res, "families": {}}
    for fam in ("nego", "risk", "compare", "percent", "sum", "fermi"):
        fam_items = fams[fam]
        print(f"[e6b] steering family={fam} ({len(fam_items)} items)...", flush=True)
        recs = []
        for i in range(0, len(fam_items), 8):
            chunk = fam_items[i : i + 8]
            recs.extend(run_task(args.port, "e6_steer", chunk, settings, plan=plan, n_none=4, batch=16, role="seller"))
        all_recs[fam] = recs

        # Summarize valid-option mass, parse rate, and gap share per row key
        meta = {it["item_id"]: it for it in fam_items}
        by_key = {}
        plain_vals = {0: [], 6: []}
        for r in recs:
            it = meta[r["item_id"]]
            r2 = dict(r, want=it.get("safe") or it.get("correct"), answer=it.get("answer"),
                      base_limit=it.get("base_limit"), base=it.get("base"), scale=it.get("scale"))
            p_vals = [X6.metric(r2, x, li) for x in r["rows"] if x["vector"] == "none"]
            p_vals = [v for v in p_vals if v is not None]
            if not p_vals:
                continue
            p_mean = st.mean(p_vals)
            plain_vals[it["scale"]].append(p_mean)
            for x in r["rows"]:
                if x["vector"] == "none":
                    continue
                k = f"{x['vector']}@{x['layer']}x{x['alpha']:+g}"
                entry = by_key.setdefault(k, {"deltas_x1": [], "masses_x1": [], "valid_x1": 0, "total_x1": 0})
                if it["scale"] == 0:
                    entry["total_x1"] += 1
                    if "mass" in x and x["mass"] is not None:
                        entry["masses_x1"].append(x["mass"])
                    v = X6.metric(r2, x, li)
                    if v is not None:
                        entry["valid_x1"] += 1
                        entry["deltas_x1"].append(v - p_mean)
        nat_gap = (st.mean(plain_vals[6]) - st.mean(plain_vals[0])) if (plain_vals[0] and plain_vals[6]) else None
        fam_sum = {"n_items": len(fam_items), "natural_gap": nat_gap, "rows": {}}
        for k, e in sorted(by_key.items()):
            md = st.mean(e["deltas_x1"]) if e["deltas_x1"] else None
            fam_sum["rows"][k] = {
                "valid_rate_x1": e["valid_x1"] / max(e["total_x1"], 1),
                "mean_mass_x1": st.mean(e["masses_x1"]) if e["masses_x1"] else None,
                "mean_delta_x1": md,
                "share_of_gap_x1": (md / nat_gap) if (md is not None and nat_gap and abs(nat_gap) > 1e-9) else None,
            }
        summary["families"][fam] = fam_sum
        print(f"  [{fam}] nat_gap={nat_gap}", flush=True)
        for k in ("vR@42x+2", "u@42x+2", "rand@42x+2", "rand_cov@42x+2", "rand_orth@42x+2", "rand_cov@42x+1"):
            if k in fam_sum["rows"]:
                rr = fam_sum["rows"][k]
                print(f"    {k:<16} valid={rr['valid_rate_x1']:.2f} mass={rr['mean_mass_x1']} delta={rr['mean_delta_x1']} share={rr['share_of_gap_x1']}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
