#!/usr/bin/env python3
"""analyze_addendum_d.py: Paired within-fixture analysis for Addendum D (D1 Thinking-On, D3 $1,000 Anchor).

Links directly to derived/study3/study3.json (read-only) on the exact same 40 fixtures G01–G40.
Uses Study 3's exact statistical primitives: 200,000 Wilcoxon sign flips, 10,000 fixture bootstrap resamples,
90% CI for equivalence (±0.10 for closing shifts, ±0.05 for paired deltas), and 95% CI for estimation.
"""
import argparse
import json
import os
import random
import statistics as st
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
S3_JSON = os.path.join(ROOT, "derived", "study3", "study3.json")


def wilcoxon_sign_flip(vals, n_flips=200_000, seed=20261122, one_sided=None):
    nz = [v for v in vals if v != 0]
    n = len(nz)
    if n == 0:
        return 1.0, 0.0, 0
    order = sorted(range(n), key=lambda i: abs(nz[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(nz[order[j + 1]]) == abs(nz[order[i]]):
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    w_plus = sum(ranks[i] for i in range(n) if nz[i] > 0)
    total = sum(ranks)
    rng = random.Random(seed)
    extreme = 0
    for _ in range(n_flips):
        wp = sum(r for r in ranks if rng.getrandbits(1))
        if one_sided == "greater":
            if wp >= w_plus - 1e-12:
                extreme += 1
        elif one_sided == "less":
            if wp <= w_plus + 1e-12:
                extreme += 1
        else:
            if abs(wp - total / 2) >= abs(w_plus - total / 2) - 1e-12:
                extreme += 1
    return extreme / n_flips, w_plus, n


def bootstrap_ci(vals, n_boot=10_000, seed=20261122):
    n = len(vals)
    rng = random.Random(seed)
    means = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return {
        "ci90": [means[int(0.05 * n_boot)], means[int(0.95 * n_boot)]],
        "ci95": [means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]],
    }


def summarize_paired(val_by_fx, margin=0.10, one_sided=None):
    fxs = sorted(val_by_fx)
    vals = [val_by_fx[f] for f in fxs]
    p, wp, nz = wilcoxon_sign_flip(vals, one_sided=one_sided)
    ci = bootstrap_ci(vals)
    lo90, hi90 = ci["ci90"]
    return {
        "n_fixtures": len(vals),
        "mean": st.mean(vals),
        "median": st.median(vals),
        "sd": st.stdev(vals) if len(vals) > 1 else 0.0,
        "ci90": ci["ci90"],
        "ci95": ci["ci95"],
        "below_0": sum(v < 0 for v in vals),
        "above_0": sum(v > 0 for v in vals),
        "p_unadjusted": p,
        "W_plus": wp,
        "n_nonzero": nz,
        f"equivalent_within_{margin:.2f}": (lo90 >= -margin and hi90 <= margin),
        "by_fixture": val_by_fx,
    }


def load_games(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def cell_means_by_fixture(games, cond, scale, key):
    # key: "seller_share" (agreements only) or "opener_gap" (valid first offers)
    by_fx_op = defaultdict(lambda: defaultdict(list))
    for g in games:
        if g["condition"] != cond or g["scale"] != scale:
            continue
        v = g.get(key)
        if v is not None:
            by_fx_op[g["fixture"]][g["opener"]].append(v)
    out = {}
    for fx, ops in by_fx_op.items():
        if "seller" in ops and "buyer" in ops:
            out[fx] = 0.5 * (st.mean(ops["seller"]) + st.mean(ops["buyer"]))
    return out


def analyze_d1(games, s3_model):
    out = {"games_total": len(games), "conditions": {}}
    # Quality & thinking tokens
    terminals = defaultdict(int)
    outside = 0
    thought_toks = []
    stakes_mentions = 0
    total_thoughts = 0
    for g in games:
        terminals[g["terminal"]] += 1
        if g["outside_zopa"]:
            outside += 1
        for c in g.get("calls", []):
            thought_toks.append(c.get("thoughts_tokens", 0))
            tp = (c.get("thought_preview") or "").lower()
            if tp:
                total_thoughts += 1
                if any(w in tp for w in ("million", "high stakes", "large amount", "cautious", "careful", "conservative")):
                    stakes_mentions += 1
    out["quality"] = {
        "terminals": dict(terminals),
        "outside_zopa": outside,
        "mean_thoughts_tokens_per_call": st.mean(thought_toks) if thought_toks else 0.0,
        "median_thoughts_tokens_per_call": st.median(thought_toks) if thought_toks else 0.0,
        "thought_previews_with_scale_caution_terms": stakes_mentions,
        "thought_previews_total": total_thoughts,
        "model_versions": sorted(set(v for g in games for v in g.get("model_versions", []))),
    }

    for cond in sorted(set(g["condition"] for g in games)):
        s_lo = cell_means_by_fixture(games, cond, "x1e2", "seller_share")
        s_hi = cell_means_by_fixture(games, cond, "x1e8", "seller_share")
        d_i = {fx: s_hi[fx] - s_lo[fx] for fx in sorted(set(s_lo) & set(s_hi))}

        g_lo = cell_means_by_fixture(games, cond, "x1e2", "opener_gap")
        g_hi = cell_means_by_fixture(games, cond, "x1e8", "opener_gap")
        h_f = {fx: g_hi[fx] - g_lo[fx] for fx in sorted(set(g_lo) & set(g_hi))}

        # Paired delta vs Study 3 Thinking-Off
        s3_d = s3_model["P2"][cond]["d_i"]
        delta_d = {fx: d_i[fx] - s3_d[fx] for fx in sorted(set(d_i) & set(s3_d))}

        out["conditions"][cond] = {
            "closing_shift_d": summarize_paired(d_i, margin=0.10),
            "opening_modesty_h": summarize_paired(h_f, margin=0.03),
            "delta_d_vs_study3": summarize_paired(delta_d, margin=0.05),
            "study3_d_mean": s3_model["P2"][cond]["mean"],
        }

    # P1 interaction I_f = h_f(A1) - h_f(A0) if both A0 and A1 present
    if "A0" in out["conditions"] and "A1" in out["conditions"]:
        h0 = out["conditions"]["A0"]["opening_modesty_h"]["by_fixture"]
        h1 = out["conditions"]["A1"]["opening_modesty_h"]["by_fixture"]
        I_f = {fx: h1[fx] - h0[fx] for fx in sorted(set(h0) & set(h1))}
        out["P1_interaction_I"] = summarize_paired(I_f, margin=0.03, one_sided="greater")

    return out


def analyze_d3(games, s3_model):
    # Compare $1,000 (x1e3) vs Study 3's $100 (x1e2) and $100M (x1e8)
    s_1k = cell_means_by_fixture(games, "A0", "x1e3", "seller_share")
    g_1k = cell_means_by_fixture(games, "A0", "x1e3", "opener_gap")

    # Reconstruct Study 3 per-fixture A0 x1e2 and x1e8 share? Study 3 stores d_i = s(100M) - s(100),
    # and if we also run x1e2/x1e8 or load Study 3 raw, wait: check if x1e3 alone needs x1e8 baseline.
    # For standalone precision, we report mean share & opening gap at $1,000, AND if D3 run also includes x1e8,
    # compute paired d(100M - 1K) directly!
    terminals = defaultdict(int)
    outside = 0
    for g in games:
        terminals[g["terminal"]] += 1
        if g["outside_zopa"]:
            outside += 1
    out = {
        "games_total": len(games),
        "quality": {"terminals": dict(terminals), "outside_zopa": outside},
        "mean_seller_share_1k": st.mean(s_1k.values()) if s_1k else None,
        "mean_opener_gap_1k": st.mean(g_1k.values()) if g_1k else None,
        "n_fixtures_1k": len(s_1k),
    }
    s_hi = cell_means_by_fixture(games, "A0", "x1e8", "seller_share")
    if s_hi:
        d_100m_1k = {fx: s_hi[fx] - s_1k[fx] for fx in sorted(set(s_hi) & set(s_1k))}
        s3_d = s3_model["P2"]["A0"]["d_i"]
        diff = {fx: d_100m_1k[fx] - s3_d[fx] for fx in sorted(set(d_100m_1k) & set(s3_d))}
        out["d_100M_minus_1K"] = summarize_paired(d_100m_1k, margin=0.10)
        out["delta_vs_study3_100M_minus_100"] = summarize_paired(diff, margin=0.05)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d1-games", default=None)
    ap.add_argument("--d3-games", default=None)
    ap.add_argument("--model", default="gemini-3.5-flash-lite")
    ap.add_argument("--out", default=os.path.join(ROOT, "derived", "addendum_d", "addendum_d.json"))
    args = ap.parse_args()

    s3_doc = json.load(open(S3_JSON, encoding="utf-8"))
    s3_model = s3_doc["per_model"][args.model]
    result = {"model": args.model, "blocks": {}}

    if args.d1_games:
        result["blocks"]["D1_thinking_on"] = analyze_d1(load_games(args.d1_games), s3_model)
    if args.d3_games:
        result["blocks"]["D3_1k_anchor"] = analyze_d3(load_games(args.d3_games), s3_model)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print("wrote", args.out)
    # Print compact summary
    for bname, b in result["blocks"].items():
        print(f"=== {bname} ===")
        if bname == "D1_thinking_on":
            q = b["quality"]
            print(f"  games={b['games_total']} terminals={q['terminals']} outside={q['outside_zopa']} mean_thought_toks={q['mean_thoughts_tokens_per_call']:.1f}")
            for cond, c in b["conditions"].items():
                d = c["closing_shift_d"]
                h = c["opening_modesty_h"]
                dd = c["delta_d_vs_study3"]
                print(f"  {cond}: d={d['mean']:+.3f} 90%CI=[{d['ci90'][0]:+.3f}, {d['ci90'][1]:+.3f}] p={d['p_unadjusted']:.4f} "
                      f"(Study3={c['study3_d_mean']:+.3f}, delta={dd['mean']:+.3f} p={dd['p_unadjusted']:.4f}); "
                      f"h={h['mean']:+.3f} p={h['p_unadjusted']:.4f}")
            if "P1_interaction_I" in b:
                p1 = b["P1_interaction_I"]
                print(f"  P1 I={p1['mean']:+.3f} 90%CI=[{p1['ci90'][0]:+.3f}, {p1['ci90'][1]:+.3f}] p={p1['p_unadjusted']:.4f}")
        elif bname == "D3_1k_anchor":
            print(f"  games={b['games_total']} terminals={b['quality']['terminals']} outside={b['quality']['outside_zopa']} "
                  f"mean_share_1k={b['mean_seller_share_1k']:.3f} mean_gap_1k={b['mean_opener_gap_1k']:.3f}")
            if "d_100M_minus_1K" in b:
                d = b["d_100M_minus_1K"]
                dd = b["delta_vs_study3_100M_minus_100"]
                print(f"  d(100M-1K)={d['mean']:+.3f} 90%CI=[{d['ci90'][0]:+.3f}, {d['ci90'][1]:+.3f}] p={d['p_unadjusted']:.4f}; "
                      f"delta_vs_S3={dd['mean']:+.3f} 90%CI=[{dd['ci90'][0]:+.3f}, {dd['ci90'][1]:+.3f}]")


if __name__ == "__main__":
    main()
