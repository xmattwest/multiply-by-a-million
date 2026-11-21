#!/usr/bin/env python3
"""interp_buyers_laptop.py (LAPTOP-SIDE, MI Session 2; loaded through scripts/interp/exp_buyers.py): does the buyer's smaller timidity use the seller's mechanism?

    python3 scripts/launch_interp.py --preview --experiment buyers               (free, canned numbers)
    python3 scripts/launch_interp.py --preview --experiment buyers --buyers-preview-fail   (canned E0 fails the gate)
    python3 scripts/launch_interp.py --experiment buyers --slot 7 --lifetime-min 120 --budget-min 90   (paid)

One session, Gemma 4 31B, buyers (a.roles = "buyer"), fixtures Y01-Y30, D2 "N dollars" vs D8 "N000000 dollars":
  1. E0 gate: the vLLM digits run's prompts and seeds; 240 sampled + 60 greedy + 60 R2 (core tasks).
     Pass (plan §2, applied to the buyer's discount below the ceiling): mean within 3 points of vLLM at
     both scales and the shift within 25% of vLLM's. Checked right after the 240 sampled openings; a
     fail stops the session there. (Not enforced in the local rehearsal: random weights.)
  2. Coarse E2 (core e2_patch): every second layer, positions "post" (all text after the limit) and
     "final", both directions, R2 price.
  3. Layers, chosen automatically after E2: A at --steer-layer (44) and at the buyer's E2 peak (mean of
     both directions, "post") if that peak lies outside --buyers-same-window (default 40-48, the
     seller's plateau). B at 44, unless a second layer exists, A's shared direction at 44 moves < 50%
     and the second layer moves more: then B there.
  4. Directions (tasks_buyers.buyers_build): both roles' per-token x1e6 - x1 differences at each A layer
     (prompt after the limit + the reply's 11 opening tokens), cosines with split-half references.
  5. Part A (buyers_bridge, E2's setup) and Part B (buyers_steer, real sampled openings, 4 per row):
     the buyer's own direction, the seller's direction on buyer prompts (as is, norm-matched, and its
     part orthogonal to the buyer's), non-money; fractions of the gap AND points moved.
Buyer gap = 1 - price / ceiling. Off-scale prices are excluded, never averaged. For buyers the
on-scale window is 0.2-3x the ceiling (the core 0.5-3x is a seller window: 8 of 120 vLLM x1 buyer
openings are below 0.5x). Standard library only. Exploratory.

Why this file is not in scripts/interp: the launcher packs all of scripts/interp into the pod's create
request (capped at 100 KB). This code never runs on the pod, so it lives here and the plug-in file
scripts/interp/exp_buyers.py only re-exports it.
"""
import contextlib
import json
import math
import os
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LI = None                                # launch_interp, set by setup() (report() has no other handle on it)
BUYER_ON_SCALE = (0.2, 3.0)
OFF_SCALE_MAX_SHARE = 0.1                 # an E2 layer with more off-scale readings than this can't be the peak
BRIDGE_ROWS = (("none", "all", 0.0),
               ("own", "all", 1.0), ("own", "prompt", 1.0), ("own", "reply", 1.0),
               ("v_pos", "all", 0.5), ("v_pos", "all", 1.0), ("v_pos", "all", 1.5), ("v_pos", "all", 2.0),
               ("v_pos", "prompt", 1.0), ("v_pos", "reply", 1.0), ("v_mean", "all", 1.0), ("random", "all", 1.0),
               ("nonmoney", "all", 1.0), ("nonmoney", "all", 2.0),
               ("seller", "all", 0.5), ("seller", "all", 1.0), ("seller", "all", 2.0),
               ("seller", "prompt", 1.0), ("seller", "reply", 1.0),
               ("seller_norm", "all", 0.5), ("seller_norm", "all", 1.0), ("seller_norm", "all", 2.0),
               ("seller_norm", "prompt", 1.0), ("seller_norm", "reply", 1.0),
               ("seller_perp", "all", 1.0), ("seller_perp", "all", 2.0))
STEER_ROWS = (("v_pos", "prompt+reply", (0.5, 1.0, 2.0), 4), ("seller", "prompt+reply", (1.0,), 4),
              ("seller_norm", "prompt+reply", (1.0,), 4), ("nonmoney", "prompt+reply", (1.0,), 4))
NAME = {"own": "own swap", "v_pos": "buyer v", "v_mean": "buyer v single", "random": "random",
        "nonmoney": "non-money", "seller": "seller v", "seller_norm": "seller v, buyer size",
        "seller_perp": "seller v ⟂ buyer v"}
A_SHOW = (("own", "all", 1.0), ("own", "prompt", 1.0), ("own", "reply", 1.0), ("v_pos", "all", 0.5),
          ("v_pos", "all", 1.0), ("v_pos", "all", 2.0), ("v_pos", "prompt", 1.0), ("v_pos", "reply", 1.0),
          ("v_mean", "all", 1.0), ("random", "all", 1.0), ("nonmoney", "all", 1.0), ("nonmoney", "all", 2.0),
          ("seller", "all", 0.5), ("seller", "all", 1.0), ("seller", "all", 2.0), ("seller", "prompt", 1.0),
          ("seller", "reply", 1.0), ("seller_norm", "all", 0.5), ("seller_norm", "all", 1.0), ("seller_norm", "all", 2.0),
          ("seller_norm", "prompt", 1.0), ("seller_norm", "reply", 1.0), ("seller_perp", "all", 1.0),
          ("seller_perp", "all", 2.0))
A_LIVE = (("own", "all", 1.0), ("v_pos", "all", 1.0), ("seller", "all", 1.0), ("seller_norm", "all", 1.0),
          ("nonmoney", "all", 1.0))
B_SHOW = (("v_pos|prompt+reply", (0.5, 1.0, 2.0)), ("seller|prompt+reply", (1.0,)),
          ("seller_norm|prompt+reply", (1.0,)), ("nonmoney|prompt+reply", (1.0,)))
GUESS_S = {"E2": 9.0, "dirs 1st": 0.6, "A 1st": 4.0, "dirs 2nd": 0.6, "A 2nd": 4.0, "B": 40.0}


def add_args(ap):
    ap.add_argument("--buyers-layer-step", type=int, default=2, help="buyers: coarse E2 every Nth layer (default 2)")
    ap.add_argument("--buyers-same-window", default="40-48",
                    help="buyers: an E2 peak inside this layer range counts as 'the same as --steer-layer' (default 40-48)")
    ap.add_argument("--buyers-second-layer", type=int, default=None,
                    help="buyers: force A's second layer (rehearsals; the paid run leaves the choice to E2)")
    ap.add_argument("--buyers-preview-fail", action="store_true",
                    help="buyers, --preview only: canned E0 numbers that fail the gate")


def configure(a):
    a.roles = "buyer"


@contextlib.contextmanager
def buyer_scale(li, window=BUYER_ON_SCALE):
    """The core scale_free reads li.ON_SCALE at call time; use the buyer window while summarizing.
    (window= lets the figures reuse the tables for the sellers' runs with the core window.)"""
    old = li.ON_SCALE
    li.ON_SCALE = window
    try:
        yield
    finally:
        li.ON_SCALE = old


def _window(text):
    lo, hi = (int(x) for x in text.split("-"))
    return lo, hi


# --- setup -------------------------------------------------------------------------------------
def setup(a, settings, schema, li):
    if a.buyers_preview_fail and not a.preview:
        raise SystemExit("--buyers-preview-fail is for --preview only")
    _window(a.buyers_same_window)
    li.STEP_GUESS_S.update(GUESS_S)          # panel ETA before a step has its own timing (this process only)
    e2_items = li.build_e2_items(settings, a.fixtures, role="buyer")
    s_vec, _ = li.build_e3_items(settings, a.fixtures, role="seller")
    b_vec, steer_items = li.build_e3_items(settings, a.fixtures, role="buyer")
    vec_items = [v for v in s_vec if v["kind"] == "pair"] + b_vec           # seller pairs, buyer pairs, non-money
    e2_layers = li.layer_list(60, a.buyers_layer_step)
    L1 = a.steer_layer
    kw = {"E2": {"batch": a.batch, "layers": e2_layers, "positions": ["post", "final"]},
          "dirs 1st": {"layer": L1}, "A 1st": {"layer": L1, "rows": [list(r) for r in BRIDGE_ROWS], "batch": a.batch},
          "dirs 2nd": {"layer": None}, "A 2nd": {"layer": None, "rows": [list(r) for r in BRIDGE_ROWS], "batch": a.batch},
          "B": {"layer": L1, "plan": [[v, sc, list(al), n] for v, sc, al, n in STEER_ROWS], "batch": a.batch,
                "schema": schema}}
    lists = {"dirs 2nd": list(vec_items), "A 2nd": list(steer_items)}
    holder = {"li": li, "a": a, "kw": kw, "lists": lists, "e2_layers": e2_layers, "layers": {"A": [L1], "B": L1},
              "choice_log": [], "done": set()}

    def steps(ctx):
        holder["ctx"] = ctx
        return li.e0_steps(ctx) + [
            ("E2", "e2_patch", e2_items, kw["E2"], 1),
            ("dirs 1st", "buyers_build", vec_items, kw["dirs 1st"], len(vec_items)),
            ("A 1st", "buyers_bridge", steer_items, kw["A 1st"], 4),
            ("dirs 2nd", "buyers_build", lists["dirs 2nd"], kw["dirs 2nd"], max(1, len(vec_items))),
            ("A 2nd", "buyers_bridge", lists["A 2nd"], kw["A 2nd"], 4),
            ("B", "buyers_steer", steer_items, kw["B"], 1)]

    n_e0 = a.fixtures * a.samples * 2
    plan = (f"Buyers on {a.model}: E0 gate ({n_e0} sampled + {a.fixtures * 2} greedy + {a.fixtures * 2} R2; stops if it "
            f"fails); coarse E2 every {a.buyers_layer_step}nd layer x post/final ({len(e2_items)} items); directions of "
            f"both roles; A at layer {L1} (+ the buyer's E2 peak if outside {a.buyers_same_window}), "
            f"{len(BRIDGE_ROWS)} rows x {len(steer_items)} items; B at one layer, "
            f"{sum(len(al) * n for _, _, al, n in STEER_ROWS) + 8} sampled openings x {len(steer_items)} items")
    inputs = ([{"step": "e2", **it} for it in e2_items] + [{"step": "vectors", **it} for it in vec_items]
              + [{"step": "steer", **it} for it in steer_items])
    _, e0_items = li.build_items(settings, a.fixtures, a.samples, list(li.CONDITIONS), ("buyer",))
    inputs = [{"step": "e0", **it} for it in e0_items] + inputs
    global _LI
    _LI = li
    return {"exp": "Buyers", "plan": plan, "inputs": inputs, "steps": steps,
            "ctx": {"buyers": holder}, "live": lambda got: live(got, holder), "after_step": after_step}


def on_pod_info(ctx, pod_info):
    h = ctx["buyers"]
    n = int(pod_info.get("num_hidden_layers") or 60)
    h["e2_layers"][:] = h["li"].layer_list(n, h["a"].buyers_layer_step)
    h["rehearsal"] = "REHEARSAL" in str(pod_info.get("attention_choice"))
    if not 0 <= h["a"].steer_layer < n:
        raise ValueError(f"--steer-layer {h['a'].steer_layer} outside 0..{n - 1}")


def manifest(ctx):
    h = ctx["buyers"]
    return {"name": "buyers", "role": "buyer", "bridge_rows": BRIDGE_ROWS, "steer_rows": STEER_ROWS,
            "buyer_on_scale": BUYER_ON_SCALE, "same_window": h["a"].buyers_same_window,
            "layer_step": h["a"].buyers_layer_step, "primary_layer": h["a"].steer_layer,
            "gate": "plan §2 on the buyer discount: |pod - vLLM| <= 3 points at x1 and x1e6, shift within 25%",
            "layer_rule": "A: primary + buyer E2 'post' peak (mean of directions) if outside same_window; "
                          "B: primary unless A(v_pos|all|1) at primary < 50% and the 2nd layer moves more"}


# --- statistics ------------------------------------------------------------------------------------
def buyer_fidelity(li, sampled, vllm_recs):
    """Plan §2 on the buyer discount, plus per-fixture agreement."""
    summ = li.summarize(sampled, [], [], vllm_recs)
    pod, vl = summ["pod_sampled"], summ["vllm"]
    out = {"n_sampled": len(sampled), "n_vllm": len(vllm_recs)}
    try:
        p2, p8 = pod["buyer_D2"]["mean_gap"], pod["buyer_D8"]["mean_gap"]
        v2, v8 = vl["buyer_D2"]["mean_gap"], vl["buyer_D8"]["mean_gap"]
    except (KeyError, TypeError):
        out.update({"pass": False, "why": "missing buyer cells"})
        return out
    if None in (p2, p8, v2, v8):
        out.update({"pass": False, "why": "no valid buyer openings in a cell"})
        return out
    sp, sv = p8 - p2, v8 - v2
    out.update({"pod_D2": p2, "pod_D8": p8, "vllm_D2": v2, "vllm_D8": v8, "shift_pod": sp, "shift_vllm": sv,
                "D2_diff": abs(p2 - v2), "D8_diff": abs(p8 - v8), "gaps_within_3": abs(p2 - v2) <= 3 and abs(p8 - v8) <= 3,
                "shift_within_25pct": abs(sp - sv) <= 0.25 * abs(sv), "shift_tolerance_pts": 0.25 * abs(sv),
                "invalid": sum(c.get("invalid", 0) for c in pod.values()),
                "slips": sum(c.get("slips", 0) for c in pod.values())})
    out["pass"] = out["gaps_within_3"] and out["shift_within_25pct"]
    fx = {}
    for src, recs in (("pod", sampled), ("vllm", vllm_recs)):
        for r in recs:
            if r.get("gap") is not None and r["role"] == "buyer":
                fx.setdefault((r["item_id"][:3], r["condition"], src), []).append(100 * r["gap"])
    fids = sorted({k[0] for k in fx})
    for c in ("D2", "D8"):
        xs = [(st.mean(fx[(f, c, "pod")]), st.mean(fx[(f, c, "vllm")])) for f in fids
              if (f, c, "pod") in fx and (f, c, "vllm") in fx]
        out[f"fixture_r_{c}"] = li.pearson([x for x, _ in xs], [y for _, y in xs])
    shifts = [st.mean(fx[(f, "D8", "pod")]) - st.mean(fx[(f, "D2", "pod")]) for f in fids
              if (f, "D8", "pod") in fx and (f, "D2", "pod") in fx]
    out["fixtures"] = len(shifts)
    out["fixtures_negative"] = sum(1 for s in shifts if s < 0)
    lo, hi = li.boot_ci(shifts)
    out["shift_pod_ci"] = None if lo is None else [lo, hi]
    return out


def bridge_table(li, recs):
    """Part A per (direction, vector|scope|alpha): fraction of the gap moved, points moved, both with
    CIs over fixtures. g_t = this item's unsteered reading, g_s = the same fixture's other direction's."""
    with buyer_scale(li):
        clean = {}
        for r in recs:
            c = next((x for x in r["rows"] if x["vector"] == "none"), None)
            clean[(r["fixture"], r["target_condition"])] = li.scale_free(r["role"], c and c["price"], r["base_limit"])[0]
        acc = {}
        for r in recs:
            g_t = clean.get((r["fixture"], r["target_condition"]))
            g_s = clean.get((r["fixture"], "D8" if r["target_condition"] == "D2" else "D2"))
            if g_t is None or g_s is None:
                continue
            for x in r["rows"]:
                if x["vector"] == "none":
                    continue
                c = acc.setdefault((r["direction"], f"{x['vector']}|{x['scope']}|{x['alpha']}"),
                                   {"num": [], "den": [], "off_scale": 0, "n": 0})
                c["n"] += 1
                g, _m, off = li.scale_free(r["role"], x["price"], r["base_limit"])
                if off or g is None:
                    c["off_scale"] += 1
                    continue
                c["num"].append(g - g_t)
                c["den"].append(g_s - g_t)
    out = {"n_fixtures": len({r["fixture"] for r in recs}), "n_items": len(recs),
           "clean_consistent": sum(1 for r in recs if r.get("clean_consistent")), "cells": {}}
    for (dname, key), c in sorted(acc.items()):
        cell = {"n": c["n"], "off_scale": c["off_scale"], "fraction": None, "ci": None, "points": None, "points_ci": None}
        if c["num"]:
            den = st.mean(c["den"])
            cell["points"] = 100 * st.mean(c["num"])
            lo, hi = li.boot_ci(c["num"], n=1000)
            if lo is not None:
                cell["points_ci"] = [100 * lo, 100 * hi]
            if den:
                cell["fraction"] = st.mean(c["num"]) / den
                cell["ci"] = sorted([lo / den, hi / den]) if lo is not None else None
            cell["denominator_pts"] = 100 * den
        out["cells"].setdefault(dname, {})[key] = cell
    return out


def steer_table(li, recs, aligned_only=False, window=BUYER_ON_SCALE):
    """Part B per (direction, vector|scope, alpha): mean gap (own scale; off-scale excluded), points moved
    against the same fixture's unsteered rows (paired, CI over fixtures), share of the x1/x1e6 gap."""
    per, counts = {}, {}
    with buyer_scale(li, window):
        for r in recs:
            own = 0 if r["target_condition"] == "D2" else 6
            for x in r["rows"]:
                vec = "none" if x["vector"] == "none" else f"{x['vector']}|{x['scope']}"
                key = (r["direction"], vec, float(x["alpha"]))
                c = counts.setdefault(key, {"n": 0, "valid": 0, "offers": 0, "off_scale": 0, "other_scale": 0,
                                            "aligned": 0, "length": 0})
                c["n"] += 1
                c["aligned"] += bool(x.get("aligned"))
                c["length"] += x.get("finish") == "length"
                if aligned_only and vec != "none" and not x.get("aligned"):
                    continue
                if x.get("errors"):
                    continue
                c["valid"] += 1
                if x.get("action") != "OFFER" or not x.get("price"):
                    continue
                c["offers"] += 1
                g, m, off = li.scale_free(r["role"], x["price"], r["base_limit"])
                if off:
                    c["off_scale"] += 1
                    continue
                if m != own:
                    c["other_scale"] += 1
                    continue
                per.setdefault(key, {}).setdefault(r["fixture"], []).append(g)
    base = {}
    for (dname, vec, _a), fx in per.items():
        if vec == "none":
            base[dname] = {f: st.mean(v) for f, v in fx.items()}
    base_mean = {d: st.mean(v.values()) for d, v in base.items() if v}
    out = {"n_fixtures": len({r["fixture"] for r in recs}), "unsteered_gap": {d: 100 * v for d, v in base_mean.items()},
           "directions": {}}
    for key in sorted(counts):
        dname, vec, a = key
        c = counts[key]
        fx = per.get(key, {})
        means = {f: st.mean(v) for f, v in fx.items()}
        cell = {**c, "aligned_share": c["aligned"] / c["n"] if c["n"] else None, "n_fixtures": len(means),
                "mean_gap": 100 * st.mean(means.values()) if means else None}
        diffs = [means[f] - base[dname][f] for f in means if f in base.get(dname, {})]
        if vec != "none" and diffs:
            cell["points"] = 100 * st.mean(diffs)
            lo, hi = li.boot_ci(diffs, n=1000)
            cell["points_ci"] = None if lo is None else [100 * lo, 100 * hi]
            other = [d for d in base_mean if d != dname]
            if other and dname in base_mean and base_mean[other[0]] != base_mean[dname]:
                span = base_mean[other[0]] - base_mean[dname]
                cell["share_moved"] = st.mean(diffs) / span
                cell["share_ci"] = None if lo is None else sorted([lo / span, hi / span])
        out["directions"].setdefault(dname, {}).setdefault(vec, {})[str(a)] = cell
    return out


def e2_profile(li, recs, layers):
    with buyer_scale(li):
        s2 = li.summarize_e2(recs)
    prof = {}
    for dname, d in s2["directions"].items():
        for pos, cells in d["cells"].items():
            prof.setdefault(pos, {})[dname] = [cells.get(str(L), {}).get("fraction") for L in layers]
    return s2, prof


def e2_peak(li, recs, layers):
    """The layer where swapping the text after the limit restores most (mean of both directions);
    layers with more than 10% off-scale readings in either direction are not eligible."""
    s2, _ = e2_profile(li, recs, layers)
    best = None
    dirs = s2["directions"]
    if len(dirs) < 2:
        return None, s2
    for L in layers:
        vals = []
        for d in dirs.values():
            c = d["cells"].get("post", {}).get(str(L))
            if not c or c.get("fraction") is None or c["n"] == 0 or c["slips"] / c["n"] > OFF_SCALE_MAX_SHARE:
                vals = None
                break
            vals.append(c["fraction"])
        if vals:
            m = st.mean(vals)
            if best is None or m > best[0]:
                best = (m, L)
    return (None if best is None else {"layer": best[1], "mean_fraction": best[0]}), s2


def a_moved(table, key="v_pos|all|1.0"):
    vals = [c[key]["fraction"] for c in table["cells"].values() if c.get(key, {}).get("fraction") is not None]
    return st.mean(vals) if len(vals) == 2 else None


# --- the gate and the adaptive choices ---------------------------------------------------------------
def after_step(label, got, ctx):
    h = ctx["buyers"]
    li = h["li"]
    h["done"].add(label)
    if label == "sampled":
        v = buyer_fidelity(li, got.get("sampled") or [], ctx.get("vllm_recs") or [])
        h["e0_verdict"] = v
        if not v.get("pass"):
            why = (f"buyer E0 FAILED the gate: x1 {v.get('pod_D2', float('nan')):.1f}% vs vLLM "
                   f"{v.get('vllm_D2', float('nan')):.1f}, x1e6 {v.get('pod_D8', float('nan')):.1f}% vs "
                   f"{v.get('vllm_D8', float('nan')):.1f}, shift {v.get('shift_pod', float('nan')):+.1f} vs "
                   f"{v.get('shift_vllm', float('nan')):+.1f}" if "pod_D2" in v else "buyer E0 FAILED: " + v.get("why", "?"))
            if h.get("rehearsal"):
                ctx["panel"].note("REHEARSAL (random weights): " + why + " — not enforced")
                return None
            return why + "; stopping before E2"
        ctx["panel"].note(f"buyer E0 PASS: x1 {v['pod_D2']:.1f}% (vLLM {v['vllm_D2']:.1f}), x1e6 {v['pod_D8']:.1f}% "
                          f"(vLLM {v['vllm_D8']:.1f}), shift {v['shift_pod']:+.1f} (vLLM {v['shift_vllm']:+.1f})")
    elif label == "teacher":
        s = li.summarize([], got.get("greedy") or [], got.get("teacher") or [], [])
        h["r2"] = {"r": s["r1_greedy_vs_r2"], "n": s["r2_n"]}
        if s["r1_greedy_vs_r2"] is not None and s["r1_greedy_vs_r2"] < 0.8:
            ctx["panel"].note(f"note: R2 vs greedy R1 r {s['r1_greedy_vs_r2']:.2f} < 0.8 (E2 and A use R2; B uses R1)")
    elif label == "E2":
        peak, _s2 = e2_peak(li, got.get("E2") or [], h["e2_layers"])
        lo, hi = _window(h["a"].buyers_same_window)
        L1 = h["a"].steer_layer
        h["e2_peak"] = peak
        forced = h["a"].buyers_second_layer
        if forced is not None or (peak and not lo <= peak["layer"] <= hi and peak["layer"] != L1):
            L2 = forced if forced is not None else peak["layer"]
            h["kw"]["dirs 2nd"]["layer"] = h["kw"]["A 2nd"]["layer"] = L2
            h["layers"]["A"] = [L1, L2]
            msg = (f"--buyers-second-layer {L2}: A also at {L2}" if forced is not None else
                   f"buyer E2 peak at layer {L2} ({100 * peak['mean_fraction']:.0f}%), outside {lo}-{hi}: A also at {L2}")
        else:
            for lab in ("dirs 2nd", "A 2nd"):
                h["lists"][lab][:] = []
            panel = ctx.get("panel")
            if panel is not None:
                panel.steps = [(lab, 0 if lab in ("dirs 2nd", "A 2nd") else n) for lab, n in panel.steps]
            msg = (f"buyer E2 peak at layer {peak['layer']} ({100 * peak['mean_fraction']:.0f}%)" if peak else
                   "no usable buyer E2 peak") + f": A and B at layer {L1} only"
        h["choice_log"].append(msg)
        ctx["panel"].note(msg)
    elif label == "A 2nd":
        L1 = h["a"].steer_layer
        if len(h["layers"]["A"]) == 2:
            L2 = h["layers"]["A"][1]
            m1 = a_moved(bridge_table(li, got.get("A 1st") or []))
            m2 = a_moved(bridge_table(li, got.get("A 2nd") or []))
            use2 = m1 is not None and m2 is not None and m1 < 0.5 and m2 > m1
            h["layers"]["B"] = L2 if use2 else L1
            h["kw"]["B"]["layer"] = h["layers"]["B"]
            f = lambda v: "—" if v is None else f"{100 * v:.0f}%"        # noqa: E731
            msg = f"A buyer v ×1 moved {f(m1)} at layer {L1}, {f(m2)} at {L2}: B at layer {h['layers']['B']}"
            h["choice_log"].append(msg)
            ctx["panel"].note(msg)
    return None


# --- live panel ----------------------------------------------------------------------------------------
def _p(v):
    return "  —" if v is None else f"{100 * v:+.0f}%"


def live(got, h):
    li, out = h["li"], []
    ctx = h.get("ctx") or {}
    if got.get("sampled"):
        vidx = {r["item_id"]: r for r in ctx.get("vllm_recs", [])}
        out += li.live_lines(got["sampled"], vidx, ("buyer",))
    if got.get("E2"):
        layers = h["e2_layers"]
        _s2, prof = e2_profile(li, got["E2"], layers)
        out.append(f"E2 buyers, share restored, layers {layers[0]}→{layers[-1]} every {h['a'].buyers_layer_step} "
                   f"(▁ 0, █ 100%), {len({r['fixture'] for r in got['E2']})} fixtures")
        for pos in ("post", "final"):
            for dname, fr in sorted(prof.get(pos, {}).items()):
                v, L = li.peak(layers, fr)
                out.append(f"  {dname:<8} {li.E2_POS_LABEL[pos]:<12} {li.spark(fr)}"
                           + (f"  peak L{L} {100 * v:.0f}%" if v is not None else ""))
    for lab in ("dirs 1st", "dirs 2nd"):
        info = (got.get(lab) or [None])[0]
        if info and info.get("cos_seller_buyer"):
            c, sh = info["cos_seller_buyer"], info["cos_split_half"]
            out.append(f"L{info['layer']} cos(seller v, buyer v): reply {c['reply']:.2f}, prompt {c['prompt']:.2f} "
                       f"(split-half buyer {sh['buyer']['reply']:.2f}/{sh['buyer']['prompt']:.2f}, seller "
                       f"{sh['seller']['reply']:.2f}/{sh['seller']['prompt']:.2f}); |v| reply seller "
                       f"{info['norm_v']['seller']['reply']:.0f} buyer {info['norm_v']['buyer']['reply']:.0f}")
    for lab in ("A 1st", "A 2nd"):
        recs = got.get(lab) or []
        if recs:
            t = bridge_table(li, recs)
            parts = []
            for vec, sc, a in A_LIVE:
                fr = [t["cells"].get(d, {}).get(f"{vec}|{sc}|{a}", {}).get("fraction") for d in ("x1+v", "x1e6-v")]
                parts.append(f"{NAME[vec]} {_p(fr[0])}/{_p(fr[1])}")
            out.append(f"A L{recs[0]['layer']} ({t['n_fixtures']} fx; ×1+v/×10⁶−v, α 1): " + " · ".join(parts))
    recs = got.get("B") or []
    if recs:
        t = steer_table(li, recs)
        parts = []
        for vec, alphas in B_SHOW:
            for a in alphas:
                cs = [t["directions"].get(d, {}).get(vec, {}).get(str(a), {}) for d in ("x1+v", "x1e6-v")]
                parts.append(f"{NAME[vec.split('|')[0]]}×{a:g} {_p(cs[0].get('share_moved'))}/{_p(cs[1].get('share_moved'))}")
        ug = t["unsteered_gap"]
        out.append(f"B L{recs[0]['layer']} ({t['n_fixtures']} fx; unsteered ×1 {_f(ug.get('x1+v')).strip()}%, ×10⁶ "
                   f"{_f(ug.get('x1e6-v')).strip()}%; moved ×1+v/×10⁶−v):")
        for i in range(0, len(parts), 3):
            out.append("   " + " · ".join(parts[i:i + 3]))
    return out


# --- preview stubs (canned shapes, not a model) ---------------------------------------------------------
def stub(task, items, ctx):
    h = ctx["buyers"]
    li = h["li"]
    if task in ("render", "generate", "teacher"):
        res = li.stub(task, items)
        if h["a"].buyers_preview_fail and task == "generate":
            for r in res:
                if r["condition"] == "D8":
                    r["gap"] = 0.44
                    r["price"] = round((1 - 0.44) * r["limit_units"])
        return res
    if task == "e2_patch":
        out = []
        for it in items:
            lo, hi = (0.37, 0.46) if it["direction"] == "x1->x1e6" else (0.46, 0.37)
            rows = []
            for p in ("post", "final"):
                for L in h["e2_layers"]:
                    f = 1 / (1 + math.exp(-(L - 41) / 2)) * (1.0 if p == "post" else 0.3)
                    g = lo + f * (hi - lo)
                    rows.append({"layer": L, "pos": p, "price": round((1 - g) * it["target"]["limit_units"]), "cont": ""})
            rows.append({"layer": "clean", "pos": None, "price": round((1 - lo) * it["target"]["limit_units"])})
            out.append({"item_id": it["item_id"], "fixture": it["fixture"], "role": it["role"],
                        "direction": it["direction"], "base_limit": it["base_limit"],
                        "source_condition": it["source"]["condition"], "target_condition": it["target"]["condition"],
                        "clean_consistent": True, "rows": rows, "secs": 0.1})
        return out
    if task == "buyers_build":
        L = h["kw"]["dirs 2nd"]["layer"] if "A 1st" in h["done"] else h["a"].steer_layer
        parts = lambda r, p: {"all": (r + p) / 2, "prompt": p, "reply": r, "token_mean": 0.6}   # noqa: E731
        return [{"layer": L, "fixtures": [], "n_prompt": 106, "n_reply": 11, "n_post": 117, "cross_aligned": True,
                 "same_token_positions": 96, "norm_v": {"seller": {"prompt": 27.0, "reply": 60.0},
                                                        "buyer": {"prompt": 15.0, "reply": 35.0}},
                 "cos_seller_buyer": parts(0.7, 0.4), "cos_seller_buyer_per_reply_token": [0.7] * 11,
                 "cos_split_half": {"seller": parts(0.9, 0.7), "buyer": parts(0.8, 0.5), "cross": parts(0.65, 0.35)},
                 "buyer_along_seller_share": {"prompt": 0.4, "reply": 0.7}, "cos_v_u": {"seller": 0.3, "buyer": 0.25},
                 "cos_fixture_diffs_mean": {"seller": 0.8, "buyer": 0.7}}]
    eff = {"own": 0.9, "v_pos": 0.8, "v_mean": 0.4, "random": 0.0, "nonmoney": 0.05, "seller": 1.4,
           "seller_norm": 0.7, "seller_perp": 0.1, "none": 0.0}
    out = []
    for it in items:
        g0, g1 = (0.46, 0.37) if it["direction"] == "x1+v" else (0.37, 0.46)
        unit = it["target"]["limit_units"]
        price = lambda v, a: round((1 - (g0 + eff[v] * min(1.0, a) * (g1 - g0))) * unit)   # noqa: E731
        base = {"item_id": it["item_id"], "fixture": it["fixture"], "role": it["role"], "direction": it["direction"],
                "base_limit": it["base_limit"], "target_condition": it["target"]["condition"], "secs": 0.1}
        if task == "buyers_bridge":
            L = h["kw"]["A 2nd"]["layer"] if "A 1st" in h["done"] else h["a"].steer_layer
            rows = [{"vector": v, "scope": sc, "alpha": a, "price": price(v, a)} for v, sc, a in BRIDGE_ROWS]
            out.append({**base, "layer": L, "clean_consistent": True, "rows": rows})
        else:
            rows = [{"vector": "none", "scope": "prompt+reply", "alpha": 0.0, "errors": [], "action": "OFFER",
                     "finish": "stop", "aligned": True, "price": price("none", 0.0)} for _ in range(8)]
            for v, sc, al, n in STEER_ROWS:
                for a in al:
                    rows += [{"vector": v, "scope": sc, "alpha": a, "errors": [], "action": "OFFER", "finish": "stop",
                              "aligned": True, "price": price(v, a)} for _ in range(n)]
            out.append({**base, "layer": h["kw"]["B"]["layer"], "rows": rows})
    return out


# --- summary and report ---------------------------------------------------------------------------------
def summarize(got, ctx):
    h = ctx["buyers"]
    li = h["li"]
    vllm = ctx.get("vllm_recs") or []
    e0 = li.summarize(got.get("sampled") or [], got.get("greedy") or [], got.get("teacher") or [], vllm)
    e0.pop("fidelity", None)                       # the core verdict is for sellers only
    s = {"e0": e0, "e0_buyer_fidelity": h.get("e0_verdict") or (buyer_fidelity(li, got["sampled"], vllm)
                                                                 if got.get("sampled") else None),
         "e2_layers": list(h["e2_layers"]), "layers": h["layers"], "e2_peak": h.get("e2_peak"),
         "choices": h["choice_log"], "buyer_on_scale": BUYER_ON_SCALE, "rehearsal": bool(h.get("rehearsal"))}
    if got.get("E2"):
        s["e2"], _ = e2_profile(li, got["E2"], h["e2_layers"])
        r2 = {r["item_id"][:3] + r["condition"]: r.get("price_r2") for r in got.get("teacher") or []}
        same = n = 0
        for r in got["E2"]:
            c = next((x for x in r["rows"] if x["layer"] == "clean"), None)
            k = r["fixture"] + r["target_condition"]
            if c is not None and k in r2:
                n += 1
                same += c["price"] == r2[k]
        s["e2_clean_vs_teacher"] = {"same_price": same, "of": n}
    s["vectors"] = {str(x["layer"]): x for lab in ("dirs 1st", "dirs 2nd") for x in (got.get(lab) or [])}
    s["bridge"] = {str(recs[0]["layer"]): bridge_table(li, recs) for recs in (got.get("A 1st"), got.get("A 2nd")) if recs}
    if got.get("B"):
        s["steer"] = steer_table(li, got["B"])
        s["steer"]["layer"] = got["B"][0]["layer"]
        s["steer_aligned_only"] = steer_table(li, got["B"], aligned_only=True)
        with buyer_scale(li):
            s["steer_core_summary"] = li.summarize_e3b_steer(got["B"])
    s["secs_per_item"] = {lab: st.mean(r["secs"] for r in v) for lab, v in got.items() if v and "secs" in v[0]}
    return s


def _f(v, w=5):
    return " " * (w - 1) + "—" if v is None else f"{v:+{w}.1f}"


def _seller_reference():
    try:
        D = json.load(open(os.path.join(ROOT, "config", "interp", "mi_figures.json")))
        return D.get("e3b_bridge"), D.get("e3b_steer")
    except (OSError, ValueError):
        return None, None


def report(s, preview):
    print("Buyers (MI Session 2; exploratory" + (", PREVIEW: canned numbers" if preview else "")
          + (", REHEARSAL: random weights" if s.get("rehearsal") else "") + ")")
    v = s.get("e0_buyer_fidelity") or {}
    if "pod_D2" in v:
        print(f"  E0 buyer discount below the ceiling: ×1 {v['pod_D2']:.1f}% (vLLM {v['vllm_D2']:.1f}), ×10⁶ "
              f"{v['pod_D8']:.1f}% (vLLM {v['vllm_D8']:.1f}); shift {v['shift_pod']:+.1f}"
              + (f" [{v['shift_pod_ci'][0]:+.1f}, {v['shift_pod_ci'][1]:+.1f}]" if v.get("shift_pod_ci") else "")
              + f" vs {v['shift_vllm']:+.1f} (tolerance ±{v['shift_tolerance_pts']:.1f}); negative in "
              f"{v['fixtures_negative']}/{v['fixtures']} fixtures; per-fixture r "
              + "/".join("—" if v.get(f"fixture_r_{c}") is None else f"{v[f'fixture_r_{c}']:.2f}" for c in ("D2", "D8"))
              + f"; invalid {v['invalid']}, slips {v['slips']}")
        print(f"  gate: diffs {v['D2_diff']:.1f} / {v['D8_diff']:.1f} pts (max 3), shift within 25% "
              f"{v['shift_within_25pct']}  ->  {'PASS' if v['pass'] else 'FAIL'}")
    elif v:
        print(f"  E0 gate: FAIL ({v.get('why')})")
    e0 = s.get("e0") or {}
    if e0.get("r2_n"):
        print(f"  R2 vs greedy R1: r {e0['r1_greedy_vs_r2']} (n {e0['r2_n']})")
    if s.get("e2"):
        layers, li = s["e2_layers"], _LI
        e2 = s["e2"]
        c = s.get("e2_clean_vs_teacher") or {}
        print(f"  E2 (coarse, every {layers[1] - layers[0]} layers), {e2['n_fixtures']} fixtures; unpatched = this "
              f"session's R2 in {c.get('same_price')}/{c.get('of')}; share restored by layer:")
        for dname, d in e2["directions"].items():
            print(f"    {dname}: clean gap difference {d['mean_gap_difference_pts']:+.1f} pts")
            for pos in ("post", "final"):
                cells = d["cells"].get(pos, {})
                fr = [cells.get(str(L), {}).get("fraction") for L in layers]
                pv, pL = li.peak(layers, fr)
                cc = cells.get(str(pL), {}) if pL is not None else {}
                off = max((cells.get(str(L), {}).get("slips", 0) for L in layers), default=0)
                print(f"      {li.E2_POS_LABEL[pos]:<12} {li.spark(fr)}  "
                      + (f"peak L{pL} {100 * pv:.0f}%" + (f" [{100 * cc['ci'][0]:.0f}, {100 * cc['ci'][1]:.0f}]"
                                                          if cc.get("ci") else "") if pv is not None else "—")
                      + (f", off-scale up to {off}/{cc.get('n')}" if off else ""))
    for msg in s.get("choices") or []:
        print("  choice: " + msg)
    for L, x in (s.get("vectors") or {}).items():
        print(f"  directions at layer {L}: |v| prompt/reply seller {x['norm_v']['seller']['prompt']:.0f}/"
              f"{x['norm_v']['seller']['reply']:.0f}, buyer {x['norm_v']['buyer']['prompt']:.0f}/"
              f"{x['norm_v']['buyer']['reply']:.0f}; fixtures agree cos seller {x['cos_fixture_diffs_mean']['seller']:.2f}, "
              f"buyer {x['cos_fixture_diffs_mean']['buyer']:.2f}")
        if x.get("cos_seller_buyer"):
            c, sh = x["cos_seller_buyer"], x["cos_split_half"]
            print(f"    cos(seller v, buyer v): reply {c['reply']:.2f}, prompt {c['prompt']:.2f}, all {c['all']:.2f}, "
                  f"token-mean {c['token_mean']:.2f} ({x['same_token_positions']}/{x['n_post']} positions same token)")
            for k in ("buyer", "seller", "cross"):
                print(f"    split halves, {k:<6}: reply {sh[k]['reply']:.2f}, prompt {sh[k]['prompt']:.2f}, all {sh[k]['all']:.2f}")
        else:
            print("    cross-role rows NOT run: the two roles' token positions do not align")
    ref_a, ref_b = _seller_reference()
    for L, t in (s.get("bridge") or {}).items():
        print(f"  A at layer {L} (forced opening, greedy price), {t['n_fixtures']} fixtures; moved: share [CI] / points:")
        for vec, sc, a in A_SHOW:
            key = f"{vec}|{sc}|{a}"
            row = []
            for d in ("x1+v", "x1e6-v"):
                c = t["cells"].get(d, {}).get(key)
                if not c:
                    row.append(" " * 26)
                    continue
                ci = f"[{100 * c['ci'][0]:+.0f},{100 * c['ci'][1]:+.0f}]" if c.get("ci") else ""
                row.append(f"{_p(c['fraction']):>5} {ci:<11} {_f(c['points'])} pt"
                           + (f" off {c['off_scale']}" if c["off_scale"] else ""))
            if any(r.strip() for r in row):
                sref = ""
                if ref_a and vec in ("own", "v_pos", "nonmoney", "v_mean", "random"):
                    sv = [ref_a.get(d, {}).get(key, {}).get("fraction") for d in ("x1+v", "x1e6-v")]
                    if None not in sv:
                        sref = f"   (sellers {_p(sv[0]).strip()}/{_p(sv[1]).strip()})"
                print(f"    {NAME[vec]:<20} {sc:<6} ×{a:<4g} " + " | ".join(row) + sref)
    for key, title in (("steer", "B"), ("steer_aligned_only", "B, aligned openings only")):
        t = s.get(key)
        if not t:
            continue
        ug = t["unsteered_gap"]
        print(f"  {title} at layer {s['steer']['layer']} (real sampled openings), {t['n_fixtures']} fixtures; unsteered "
              f"×1 {_f(ug.get('x1+v')).strip()}%, ×10⁶ {_f(ug.get('x1e6-v')).strip()}%:")
        for vec, alphas in B_SHOW:
            for a in alphas:
                for d in ("x1+v", "x1e6-v"):
                    c = t["directions"].get(d, {}).get(vec, {}).get(str(a))
                    if not c:
                        continue
                    ci = f" [{_p(c['share_ci'][0]).strip()}, {_p(c['share_ci'][1]).strip()}]" if c.get("share_ci") else ""
                    bad = c["off_scale"] + c["other_scale"]
                    print(f"    {NAME[vec.split('|')[0]]:<20} ×{a:<4g} {d:<7} gap {_f(c['mean_gap'])}%  moved "
                          f"{_p(c.get('share_moved')).strip()}{ci} = {_f(c.get('points')).strip()} pt  valid "
                          f"{c['valid']}/{c['n']}  aligned {100 * (c['aligned_share'] or 0):.0f}%"
                          + (f"  off-scale {bad}" if bad else ""))
        if key == "steer" and ref_b:
            sv = [ref_b.get(d, {}).get("v_pos|prompt+reply", {}).get("1.0", {}).get("share_moved") for d in ("x1+v", "x1e6-v")]
            if None not in sv:
                print(f"    (sellers, X54: v ×1 moved {_p(sv[0]).strip()} / {_p(sv[1]).strip()})")
    if s.get("secs_per_item"):
        print("  seconds per item: " + ", ".join(f"{k} {v:.1f}" for k, v in s["secs_per_item"].items()))
