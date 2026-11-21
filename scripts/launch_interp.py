#!/usr/bin/env python3
"""launch_interp.py (LAPTOP-SIDE): one-terminal interp session for E0 fidelity (plan_mechinterp §2).

THIS SPENDS MONEY (except with --preview). Standard library only; never imports torch.

    python3 scripts/launch_interp.py --preview                          (free: fake machine, canned numbers)
    python3 scripts/launch_interp.py --model gemma-4-31b-it --roles seller   (paid: one pod, then deleted)
    python3 scripts/launch_interp.py --attach --slot 8 --model ...      (a session you launched yourself)

Paid flow, like the probes: a free dry run prints the plan and the worst case, you type that number,
then this program starts scripts/launch_pod.sh --mode interp itself, shows one live panel (stages,
progress with ETA, speed, GPU memory, what the pod loaded, reply validity, live E0 numbers next to the
vLLM run's on the same items, time left, cost so far), runs E0, and deletes the machine. Ctrl-C stops
after the current chunk and deletes the machine; a second Ctrl-C skips the wait.

E0 on the digits probe's prompts (D2 "81 dollars", D8 "81000000 dollars", fixtures Y01.., same seeds):
render 4 prompts; R1 sampled (frozen settings, vLLM's JSON schema via xgrammar on the pod); R1 greedy
(one per fixture x role x condition); R2 teacher-forced price digits on the greedy items. Chunks of 16,
each written to data/interp/<run_id>/results.jsonl on return (plus progress.json). Pass (plan §2): seller
mean gap within 3 points at x1 and x1e6, shift within 25%; R2 kept if r >= 0.8 vs greedy R1. "Slips"
(seller price < 0.5 x floor, buyer < 0.2 x ceiling: a digit lost) are counted separately. Exploratory.
"""
import argparse
import ast
import glob
import hashlib
import importlib.util
import http.client
import json
import math
import os
import re
import shutil
import signal
import statistics as st
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import game_engine as ge          # noqa: E402  (read-only imports)
import make_fixtures as mf        # noqa: E402
import render_prompt as rp        # noqa: E402
import render_prompt_s2 as rp2    # noqa: E402
import run_job as J               # noqa: E402
import run_records as rr          # noqa: E402

PHRASE = "scale-invariance/fixtures/explore-v2"      # the digits probe's fixtures (Y01..)
CONDITIONS = {"D2": (lambda b: f"{b} dollars", 1), "D8": (lambda b: f"{b * 10**6} dollars", 10**6)}
CHUNK = 16
STAGES = ("machine", "model files", "interp code", "model load", "E0", "teardown")
LIFETIME_GUARD_S = 360            # no new chunk when the pod has less than 6 min left


# --- prompts and items (identical to run_digits_probe for D2/D8) ------------------
def item_seed(fixture, role, sample):
    label = f"scale-invariance/digits-probe/v1|{fixture}|{role}|{sample}"
    return int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:8], 16)


def render_item(role, base_limit, condition, settings, templates=None):
    text_of = CONDITIONS[condition][0]
    msgs = rp2.render(role, 100, base_limit, [], 1, (75, 95), (105, 125),
                      settings["max_messages"]["value"], settings["max_message_chars"]["value"],
                      templates=templates)
    sysm, user = msgs[0]["content"], msgs[1]["content"]
    name = "floor" if role == "seller" else "ceiling"
    anchor = f"Your private {name} is {rp.money(base_limit)}."
    assert user.count(anchor) == 1, "limit anchor not found"
    user = user.replace(anchor, f"Your private {name} is {text_of(base_limit)}.", 1)
    return [{"role": "system", "content": sysm}, {"role": "user", "content": user}]


def build_items(settings, n_fix, samples, conds, roles=("seller", "buyer")):
    doc = mf.build(PHRASE)
    items = []
    for f in doc["fixtures"][:n_fix]:
        fid = f"Y{f['id'][1:]}"
        for role, b in (("seller", f["f_pct"]), ("buyer", f["c_pct"])):
            if role not in roles:
                continue
            for s in range(1, samples + 1):
                for c in conds:
                    items.append({
                        "item_id": f"{fid}-{role[0].upper()}-{c}-s{s}", "fixture": fid, "role": role,
                        "condition": c, "sample": s, "base_limit": b, "limit_units": b * CONDITIONS[c][1],
                        "seed": item_seed(fid, role, s), "messages": render_item(role, b, c, settings)})
    return {"seed_phrase": PHRASE, "fixtures": [f"Y{f['id'][1:]}" for f in doc["fixtures"][:n_fix]]}, items


# --- statistics (the pod's results and the vLLM run alike) ----------------------
def is_slip(role, price, limit_units):
    q = price / limit_units
    return q < 0.5 if role == "seller" else q < 0.2


def cell_stats(recs):
    out = {}
    for key in sorted({(r["role"], r["condition"]) for r in recs}):
        rs = [r for r in recs if (r["role"], r["condition"]) == key]
        offers = [r for r in rs if r.get("gap") is not None]
        g = [100 * r["gap"] for r in offers]
        clean = [100 * r["gap"] for r in offers if not is_slip(r["role"], r["price"], r["limit_units"])]
        out[f"{key[0]}_{key[1]}"] = {
            "n": len(rs), "offers": len(offers), "invalid": sum(1 for r in rs if r.get("errors")),
            "slips": len(g) - len(clean), "mean_gap": st.mean(g) if g else None,
            "median_gap": st.median(g) if g else None, "mean_gap_no_slips": st.mean(clean) if clean else None}
    return out


def shifts(cells, field="mean_gap"):
    out = {}
    for role in ("seller", "buyer"):
        a, b = cells.get(f"{role}_D2", {}).get(field), cells.get(f"{role}_D8", {}).get(field)
        out[role] = None if a is None or b is None else b - a
    return out


def load_vllm(run_dir, conds):
    recs = {}
    for line in open(os.path.join(run_dir, "games.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        if r.get("schema") == "firstmove/1" and r["condition"] in conds:
            prev = recs.get(r["item_id"])
            if prev is None or r.get("attempt", 1) >= prev.get("attempt", 1):
                recs[r["item_id"]] = r
    return [r for r in recs.values() if r["errors"] != ["INFRASTRUCTURE"]]


def default_vllm_run(model):
    for d in reversed(sorted(glob.glob(os.path.join(ROOT, "data", "explore", f"X-digits-{model}-*")))):
        try:
            if {"D2", "D8"} <= set(json.load(open(os.path.join(d, "run_manifest.json"))).get("conditions", {})):
                return d
        except (OSError, ValueError):
            continue
    return None


def pearson(xs, ys):
    try:
        return st.correlation(xs, ys) if len(xs) >= 3 else None
    except (st.StatisticsError, ValueError):
        return None


def summarize(sampled, greedy, teacher, vllm_recs):
    pod, vl = cell_stats(sampled), cell_stats(vllm_recs) if vllm_recs else {}
    g1 = {r["item_id"]: r["gap"] for r in greedy if r.get("gap") is not None}
    g2 = {r["item_id"]: r["gap_r2"] for r in teacher if r.get("gap_r2") is not None}
    common = sorted(set(g1) & set(g2))
    summ = {"pod_sampled": pod, "pod_greedy": cell_stats(greedy), "vllm": vl,
            "shift_pod": shifts(pod), "shift_vllm": shifts(vl) if vl else None,
            "shift_pod_no_slips": shifts(pod, "mean_gap_no_slips"),
            "shift_vllm_no_slips": shifts(vl, "mean_gap_no_slips") if vl else None,
            "r1_greedy_vs_r2": pearson([g1[i] for i in common], [g2[i] for i in common]), "r2_n": len(common),
            "constrained_share": (sum(1 for r in sampled if r.get("constrained")) / len(sampled)) if sampled else None}
    if vl and pod.get("seller_D2") and pod.get("seller_D8") and pod["seller_D2"]["mean_gap"] is not None \
            and pod["seller_D8"]["mean_gap"] is not None:
        d2 = abs(pod["seller_D2"]["mean_gap"] - vl["seller_D2"]["mean_gap"])
        d8 = abs(pod["seller_D8"]["mean_gap"] - vl["seller_D8"]["mean_gap"])
        sp, sv = summ["shift_pod"]["seller"], summ["shift_vllm"]["seller"]
        fd = {"seller_D2_diff": d2, "seller_D8_diff": d8, "gaps_within_3": d2 <= 3 and d8 <= 3,
              "shift_within_25pct": abs(sp - sv) <= 0.25 * abs(sv),
              "r2_usable": summ["r1_greedy_vs_r2"] is not None and summ["r1_greedy_vs_r2"] >= 0.8}
        fd["pass"] = fd["gaps_within_3"] and fd["shift_within_25pct"]
        summ["fidelity"] = fd
    return summ


# --- E1 and E2 (plan_mechinterp §4): items and analysis ------------------------------
E1_SCALES = tuple(range(0, 7))                 # negotiation limits b x 10^k, k = 0..6 (k=0 is D2, k=6 is D8)
E1_PLAIN_TEMPLATES = (
    "The house sold last spring for {A} dollars.", "Her annual salary is {A} dollars.",
    "The city spent {A} dollars on new roads.", "A used car like this costs about {A} dollars.",
    "The company reported revenue of {A} dollars this quarter.", "He donated {A} dollars to the hospital.",
    "The painting was insured for {A} dollars.", "Their monthly grocery bill came to {A} dollars.",
    "The bridge repair is expected to cost {A} dollars.", "She found a wallet with {A} dollars inside.",
    "The startup raised {A} dollars from investors.", "The museum's ticket revenue was {A} dollars last year.")
E2_DIRECTIONS = (("x1->x1e6", "D2", "D8"), ("x1e6->x1", "D8", "D2"))


def render_with_limit(role, base_limit, limit_text, settings):
    """The digits-probe prompt with the limit written as `limit_text` dollars (D2/D8 are special cases)."""
    msgs = render_item(role, base_limit, "D2", settings)
    name = "floor" if role == "seller" else "ceiling"
    old = f"Your private {name} is {base_limit} dollars."
    assert msgs[1]["content"].count(old) == 1
    msgs[1]["content"] = msgs[1]["content"].replace(old, f"Your private {name} is {limit_text} dollars.", 1)
    return msgs


def build_e1_items(settings, n_fix, role="seller", n_plain_per_template=20):
    import random
    doc = mf.build(PHRASE)
    items = []
    for f in doc["fixtures"][:n_fix]:
        fid, b = f"Y{f['id'][1:]}", f["f_pct"] if role == "seller" else f["c_pct"]
        for k in E1_SCALES:
            amount = b * 10 ** k
            items.append({"item_id": f"{fid}-{role[0].upper()}-k{k}", "kind": "negotiation", "fixture": fid,
                          "role": role, "scale_exp": k, "base_limit": b, "amount": amount,
                          "limit_text": str(amount), "messages": render_with_limit(role, b, str(amount), settings)})
    rng = random.Random("scale-invariance/interp/e1-plain/v1")
    for t, tmpl in enumerate(E1_PLAIN_TEMPLATES):
        for j in range(n_plain_per_template):
            e = rng.uniform(1.0, 9.0)
            q = 10 ** (math.floor(e) - 1)
            amount = int(round(10 ** e / q)) * q
            text = tmpl.format(A=amount)
            items.append({"item_id": f"P{t:02d}-{j:02d}", "kind": "plain", "fixture": None, "amount": amount,
                          "amount_text": str(amount), "text": text})
    return items


def build_e2_items(settings, n_fix, role="seller"):
    doc = mf.build(PHRASE)
    items = []
    for f in doc["fixtures"][:n_fix]:
        fid, b = f"Y{f['id'][1:]}", f["f_pct"] if role == "seller" else f["c_pct"]
        for name, src, tgt in E2_DIRECTIONS:
            side = {}
            for c in (src, tgt):
                units = b * CONDITIONS[c][1]
                side[c] = {"condition": c, "limit_units": units, "limit_text": str(units),
                           "messages": render_item(role, b, c, settings)}
            items.append({"item_id": f"{fid}-{role[0].upper()}-{name}", "fixture": fid, "role": role,
                          "direction": name, "base_limit": b, "source": side[src], "target": side[tgt]})
    return items


ON_SCALE = (0.5, 3.0)                          # a reading is "on scale" if price / limit is in this range


def scale_free(role, price, base_limit):
    """(gap, units exponent 0 or 6, slip) for a price read against whichever scale it is in.

    Prices within ON_SCALE of the x1 or x1e6 limit get that scale's gap. Anything else (e.g.
    1080 or 110000 for a limit of 93: the patch broke the number's size) is "off scale":
    returned as (None, None, True) and counted with the slips, never averaged in.
    """
    if not price:
        return None, None, None
    for m in (0, 6):
        q = price / (base_limit * 10 ** m)
        if ON_SCALE[0] <= q <= ON_SCALE[1]:
            return (q - 1 if role == "seller" else 1 - q), m, False
    return None, None, True


def boot_ci(xs, n=2000, seed=1):
    import random
    rng = random.Random(seed)
    if len(xs) < 3:
        return None, None
    b = sorted(st.mean(rng.choices(xs, k=len(xs))) for _ in range(n))
    return b[int(0.025 * n)], b[int(0.975 * n) - 1]


def summarize_e2(recs, ci=True):
    """Fraction of the x1/x1e6 gap difference restored by each (direction, position, layer) patch.

    Per fixture: g_t = clean target gap, g_s = clean gap of the same fixture's other direction
    (= the source prompt's own reading), g_p = patched. Fraction = mean(g_p - g_t) / mean(g_s - g_t)
    over fixtures with both directions; slips are excluded from g_p. Units: share of patched
    readings that came out in the SOURCE's scale (the patch carried the magnitude itself).
    """
    clean = {}
    for r in recs:
        c = next((x for x in r["rows"] if x["layer"] == "clean"), None)
        clean[(r["fixture"], r["target_condition"])] = scale_free(r["role"], c and c["price"], r["base_limit"])[0]
    out = {"directions": {}, "n_fixtures": len({r["fixture"] for r in recs}),
           "clean_consistent": sum(1 for r in recs if r.get("clean_consistent")), "n_items": len(recs)}
    for r in recs:
        g_t = clean.get((r["fixture"], r["target_condition"]))
        g_s = clean.get((r["fixture"], r["source_condition"]))
        if g_t is None or g_s is None:
            continue
        src_m = 0 if r["source_condition"] == "D2" else 6
        d = out["directions"].setdefault(r["direction"], {"denominator": [], "cells": {}})
        d["denominator"].append(g_s - g_t)
        for x in r["rows"]:
            if x["layer"] == "clean":
                continue
            g, m, slip = scale_free(r["role"], x["price"], r["base_limit"])
            if slip:
                g = None
            cell = d["cells"].setdefault(x["pos"], {}).setdefault(str(x["layer"]), {"num": [], "units": 0,
                                                                                    "slips": 0, "none": 0, "n": 0})
            cell["n"] += 1
            if slip:
                cell["slips"] += 1                 # off scale: the number's size broke
            elif g is None:
                cell["none"] += 1
            else:
                cell["num"].append(g - g_t)
                cell["units"] += m == src_m
    for d in out["directions"].values():
        den = st.mean(d["denominator"]) if d["denominator"] else None
        d["mean_gap_difference_pts"] = None if den is None else 100 * den
        for pos, layers in d["cells"].items():
            for L, c in layers.items():
                num = c.pop("num")
                if den and num:
                    lo, hi = boot_ci(num, n=1000) if ci else (None, None)
                    c.update(fraction=st.mean(num) / den, ci=[lo / den, hi / den] if lo is not None else None)
                    if den < 0:
                        c["ci"] = c["ci"][::-1] if c["ci"] else None
                else:
                    c.update(fraction=None, ci=None)
                c["units_share"] = c.pop("units") / c["n"] if c["n"] else None
        d.pop("denominator")
    return out


def summarize_e1(fit, e0_dir=None):
    """Best layers for the magnitude probe, and (if E0 data exist) does the final-token readout
    predict each fixture's opening gap beyond the scale itself?"""
    if not fit:
        return {}
    s = {"within": {}, "transfer": {}}
    for pos, v in fit["within"].items():
        r2 = v["r2"]
        best = max(range(len(r2)), key=lambda i: r2[i] if r2[i] is not None else -9)
        s["within"][pos] = {"best_layer": best, "best_r2": r2[best], "r2": r2}
    for key, v in fit.get("transfer", {}).items():
        r2 = v["r2_negotiation"]
        best = max(range(len(r2)), key=lambda i: r2[i] if r2[i] is not None else -9)
        s["transfer"][key] = {"best_layer": best, "best_r2": r2[best], "r2": r2}
    if e0_dir and os.path.exists(os.path.join(e0_dir, "results.jsonl")) and fit.get("pred_within", {}).get("final"):
        gaps = {}
        for line in open(os.path.join(e0_dir, "results.jsonl")):
            r = json.loads(line)
            if r.get("step") == "sampled" and r.get("gap") is not None and r["role"] == "seller":
                gaps.setdefault((r["item_id"][:3], r["condition"]), []).append(100 * r["gap"])
        L = s["transfer"].get("final->final", s["within"]["final"])["best_layer"]   # within is at ceiling
        preds = dict(zip(fit["items"], fit["pred_within"]["final"][L]))
        beyond = {}
        for k, cond in ((0, "D2"), (6, "D8")):
            xs, ys, zs = [], [], []
            for item, p in preds.items():
                fid, kk = item.split("-")[0], int(item.split("-k")[1])
                g = gaps.get((fid, cond))
                amount = fit.get("amounts", {}).get(item)
                if kk == k and g and amount:
                    xs.append(p - math.log10(amount))        # what the readout says beyond the true scale
                    ys.append(st.mean(g))
                    zs.append(math.log10(amount))
            beyond[cond] = {"r_residual_vs_gap": pearson(xs, ys), "r_partial_limit": partial_r(xs, ys, zs),
                            "n": len(xs)}
        s["beyond_scale"] = {"layer": L, "e0_run": os.path.relpath(e0_dir, ROOT), **beyond}
    return s


# --- E3 (plan_mechinterp §4): steering --------------------------------------------
E3_ALPHAS = (0.25, 0.5, 1.0, 1.5, 2.0)
E3_VECTORS = ("v_pos", "v_mean", "random", "nonmoney")
E3_DIRECTIONS = (("x1+v", "D2"), ("x1e6-v", "D8"))
E3_NONMONEY = (
    "There are {A} stars visible in this photograph.", "The library holds {A} books on its shelves.",
    "The forest has about {A} trees.", "The survey counted {A} birds this spring.",
    "The stadium recorded {A} visitors last season.", "The colony contains {A} ants.",
    "The archive stores {A} letters.", "The farm grows {A} apples each year.",
    "The lake is home to {A} fish.", "The museum owns {A} paintings.")


def build_e3_items(settings, n_fix, role="seller", n_bases=4):
    """(vector items, steer items). Steer seeds follow the E0 seed rule (sample 1)."""
    doc = mf.build(PHRASE)
    vec, steer = [], []
    for f in doc["fixtures"][:n_fix]:
        fid, b = f"Y{f['id'][1:]}", f["f_pct"] if role == "seller" else f["c_pct"]
        side = {c: {"condition": c, "limit_units": b * CONDITIONS[c][1], "limit_text": str(b * CONDITIONS[c][1]),
                    "messages": render_item(role, b, c, settings)} for c in ("D2", "D8")}
        vec.append({"item_id": f"{fid}-{role[0].upper()}-pair", "kind": "pair", "fixture": fid, "role": role,
                    "x1": side["D2"], "x1e6": side["D8"]})
        for name, c in E3_DIRECTIONS:
            steer.append({"item_id": f"{fid}-{role[0].upper()}-{name}", "fixture": fid, "role": role,
                          "direction": name, "base_limit": b, "seed": item_seed(fid, role, 1), "target": side[c]})
    bases = (76, 81, 88, 93)[:n_bases]
    for t, tmpl in enumerate(E3_NONMONEY):
        for b in bases:
            vec.append({"item_id": f"N{t:02d}-{b}", "kind": "nonmoney", "small": tmpl.format(A=b),
                        "big": tmpl.format(A=b * 10 ** 6), "small_num": str(b), "big_num": str(b * 10 ** 6)})
    return vec, steer


def summarize_e3(recs, ci=True):
    """Per direction x vector x alpha: mean opening gap (on its own scale), validity, and the
    share of the x1/x1e6 gap difference moved, relative to the unsteered rows."""
    out = {"directions": {}, "n_fixtures": len({r["fixture"] for r in recs})}
    per = {}                        # (direction, vector, alpha) -> fixture -> [gaps]; plus counts
    counts = {}
    for r in recs:
        own = 0 if r["target_condition"] == "D2" else 6
        for x in r["rows"]:
            key = (r["direction"], x["vector"], x["alpha"])
            c = counts.setdefault(key, {"n": 0, "valid": 0, "offers": 0, "off_scale": 0, "other_scale": 0,
                                        "length": 0})
            c["n"] += 1
            c["length"] += x.get("finish") == "length"
            if x.get("errors"):
                continue
            c["valid"] += 1
            if x.get("action") != "OFFER" or not x.get("price"):
                continue
            c["offers"] += 1
            g, m, off = scale_free(r["role"], x["price"], r["base_limit"])
            if off:
                c["off_scale"] += 1
                continue
            c["other_scale"] += m != own
            if m == own:
                per.setdefault(key, {}).setdefault(r["fixture"], []).append(g)
    base = {}
    for (dname, vec, a), fx in per.items():
        if vec == "none":
            base[dname] = st.mean(st.mean(v) for v in fx.values())
    for key, c in sorted(counts.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        dname, vec, a = key
        fx = per.get(key, {})
        means = [st.mean(v) for v in fx.values()]
        m = st.mean(means) if means else None
        cell = {**c, "mean_gap": None if m is None else 100 * m, "n_fixtures": len(means)}
        if ci and len(means) >= 3:
            lo, hi = boot_ci(means, n=1000)
            cell["ci"] = [100 * lo, 100 * hi]
        other = [d for d in base if d != dname]
        if m is not None and dname in base and other and base[other[0]] != base[dname]:
            cell["share_moved"] = (m - base[dname]) / (base[other[0]] - base[dname])
        out["directions"].setdefault(dname, {}).setdefault(vec, {})[str(a)] = cell
    out["unsteered_gap"] = {d: 100 * v for d, v in base.items()}
    return out


def e3_live(got):
    out = []
    info = (got.get("e3 vectors") or [None])[0]
    if info:
        out.append(f"E3 layer {info['layer']}: |v| per position {info['norm_v_pos_mean']:.0f} vs activations "
                   f"{info['mean_resid_norm_post']:.0f}; fixtures' directions agree cos {info['cos_fixture_dirs_mean']:.2f}; "
                   f"non-money cos {info['cos_v_mean_u']:.2f}")
    recs = got.get("e3 steer") or []
    if recs:
        s3 = summarize_e3(recs, ci=False)
        for dname, vecs in s3["directions"].items():
            base = vecs.get("none", {}).get("0.0", {})
            head = f"E3 {dname:<7} unsteered {base.get('mean_gap') or 0:+.0f}%"
            out.append(head + f"  ({s3['n_fixtures']} fixtures; gap by α " + " ".join(str(a) for a in E3_ALPHAS) + ")")
            for vec in E3_VECTORS:
                cells = vecs.get(vec, {})
                if not cells:
                    continue
                vals = []
                for a in E3_ALPHAS:
                    c = cells.get(str(a), {})
                    vals.append("  —  " if c.get("mean_gap") is None else f"{c['mean_gap']:+5.0f}%")
                n = sum(c["n"] for c in cells.values())
                valid = sum(c["valid"] for c in cells.values())
                off = sum(c["off_scale"] + c["other_scale"] for c in cells.values())
                out.append(f"   {vec:<9} " + " ".join(vals) + f"   valid {100 * valid / max(1, n):.0f}%"
                           + (f" · off-scale {off}" if off else ""))
    return out


def stub_e3(task, items):
    """Preview only: canned shapes, not a model."""
    if task == "e3_build":
        return [{"layer": 44, "fixtures": [], "n_post": 108, "n_nonmoney": 40, "norm_v_pos_mean": 40.0,
                 "norm_v_mean": 25.0, "norm_u": 12.0, "mean_resid_norm_post": 180.0, "cos_v_mean_u": 0.2,
                 "cos_fixture_dirs_mean": 0.95, "cos_fixture_dirs_min": 0.9}]
    out = []
    for it in items:
        g0, g1 = (0.41, 0.21) if it["direction"] == "x1+v" else (0.21, 0.41)
        rows = []
        for vec, a in [("none", 0.0)] * 8 + [(v, a) for v in E3_VECTORS for a in E3_ALPHAS for _ in range(2)]:
            k = {"v_pos": 1.0, "v_mean": 0.8, "random": 0.0, "nonmoney": 0.1, "none": 0.0}[vec] * min(1.0, a)
            g = g0 + k * (g1 - g0)
            rows.append({"vector": vec, "alpha": a, "errors": [], "action": "OFFER", "finish": "stop",
                         "price": round((1 + g) * it["target"]["limit_units"]), "content": "(preview)"})
        out.append({"item_id": it["item_id"], "fixture": it["fixture"], "role": it["role"],
                    "direction": it["direction"], "base_limit": it["base_limit"],
                    "target_condition": it["target"]["condition"], "rows": rows, "secs": 0.2})
    return out


# --- E3b: steering that covers the reply's own opening tokens -------------------------
E3B_SCOPE_LABEL = {"all": "prompt+reply", "prompt": "prompt only", "reply": "reply only",
                   "prompt+reply": "prompt+reply", "all_gen": "prompt+every reply token"}


def summarize_e3b_bridge(recs, ci=True):
    """Part A (R2, greedy): share of the x1/x1e6 gap moved per (vector, scope, alpha), like E2.
    g_t = this item's unsteered reading, g_s = the same fixture's other direction unsteered."""
    clean = {}
    for r in recs:
        c = next((x for x in r["rows"] if x["vector"] == "none"), None)
        clean[(r["fixture"], r["target_condition"])] = scale_free(r["role"], c and c["price"], r["base_limit"])[0]
    out = {"cells": {}, "n_fixtures": len({r["fixture"] for r in recs}),
           "clean_consistent": sum(1 for r in recs if r.get("clean_consistent")), "n_items": len(recs)}
    acc = {}
    for r in recs:
        g_t = clean.get((r["fixture"], r["target_condition"]))
        g_s = clean.get((r["fixture"], "D8" if r["target_condition"] == "D2" else "D2"))
        if g_t is None or g_s is None:
            continue
        for x in r["rows"]:
            if x["vector"] == "none":
                continue
            key = (r["direction"], x["vector"], x["scope"], x["alpha"])
            c = acc.setdefault(key, {"num": [], "den": [], "off_scale": 0, "n": 0})
            c["n"] += 1
            g, m, off = scale_free(r["role"], x["price"], r["base_limit"])
            if off or g is None:
                c["off_scale"] += 1
                continue
            c["num"].append(g - g_t)
            c["den"].append(g_s - g_t)
    for (dname, vec, sc, a), c in acc.items():
        den = st.mean(c["den"]) if c["den"] else None
        cell = {"n": c["n"], "off_scale": c["off_scale"], "fraction": None, "ci": None}
        if den and c["num"]:
            cell["fraction"] = st.mean(c["num"]) / den
            if ci and len(c["num"]) >= 3:
                lo, hi = boot_ci(c["num"], n=1000)
                cell["ci"] = sorted([lo / den, hi / den])
        out["cells"].setdefault(dname, {})[f"{vec}|{sc}|{a}"] = cell
    return out


def summarize_e3b_steer(recs, ci=True, aligned_only=False):
    """Part B: summarize_e3 with vector = 'vector|scope', plus the share of rows whose generated
    opening matched the expected tokens (the reply steering is by token index)."""
    flat = []
    for r in recs:
        rows = [dict(x, vector=x["vector"] if x["vector"] == "none" else f"{x['vector']}|{x['scope']}")
                for x in r["rows"] if not aligned_only or x.get("aligned") or x["vector"] == "none"]
        flat.append(dict(r, rows=rows))
    s3 = summarize_e3(flat, ci=ci)
    al = {}
    for r in recs:
        for x in r["rows"]:
            k = (r["direction"], x["vector"] if x["vector"] == "none" else f"{x['vector']}|{x['scope']}", x["alpha"])
            a = al.setdefault(k, [0, 0])
            a[0] += bool(x.get("aligned")); a[1] += 1
    for dname, vecs in s3["directions"].items():
        for vec, cells in vecs.items():
            for a, c in cells.items():
                n_al = al.get((dname, vec, float(a)))
                c["aligned_share"] = n_al[0] / n_al[1] if n_al else None
    return s3


E3B_BRIDGE_SHOW = (("own", "all", 1.0), ("own", "prompt", 1.0), ("own", "reply", 1.0), ("v_pos", "all", 0.5),
                   ("v_pos", "all", 1.0), ("v_pos", "all", 2.0), ("v_pos", "prompt", 1.0), ("v_pos", "reply", 1.0),
                   ("v_mean", "all", 1.0), ("v_mean", "all", 2.0), ("random", "all", 1.0), ("random", "all", 2.0),
                   ("nonmoney", "all", 1.0), ("nonmoney", "all", 2.0))
E3B_STEER_SHOW = (("v_pos|prompt+reply", (0.5, 1.0, 1.5, 2.0)), ("own|prompt+reply", (1.0,)),
                  ("v_pos|reply", (1.0, 2.0)), ("v_mean|all_gen", (1.0, 2.0)),
                  ("random|prompt+reply", (1.0, 2.0)), ("nonmoney|prompt+reply", (1.0, 2.0)))
E3B_NAME = {"own": "own swap", "v_pos": "v per token", "v_mean": "v single", "random": "random", "nonmoney": "non-money"}


def _pct(v):
    return "  —  " if v is None else f"{100 * v:+4.0f}%"


def e3b_live(got):
    out = []
    info = (got.get("e3b vectors") or [None])[0]
    if info:
        out.append(f"E3b layer {info['layer']}: {info['n_prompt']} prompt + {info['n_reply']} reply tokens; |v| prompt "
                   f"{info['norm_v_prompt']:.0f}, reply {info['norm_v_reply']:.0f}; fixtures agree cos "
                   f"{info['cos_fixture_diffs_mean']:.2f}")
    br = got.get("e3b bridge") or []
    if br:
        sb = summarize_e3b_bridge(br, ci=False)
        out.append(f"E3b A (forced reply opening, greedy; share of gap moved), {sb['n_fixtures']} fixtures:")
        for dname, cells in sb["cells"].items():
            parts = []
            for vec, sc, a in E3B_BRIDGE_SHOW:
                c = cells.get(f"{vec}|{sc}|{a}")
                if c:
                    parts.append(f"{E3B_NAME[vec]}{'' if sc == 'all' else '/' + sc}×{a:g} {_pct(c['fraction']).strip()}")
            for i in range(0, len(parts), 5):
                out.append(f"  {dname if i == 0 else '':<7} " + " · ".join(parts[i:i + 5]))
    sr = got.get("e3b steer") or []
    if sr:
        ss = summarize_e3b_steer(sr, ci=False)
        out.append(f"E3b B (real sampled openings; share of gap moved, aligned openings %), {ss['n_fixtures']} fixtures:")
        for dname, vecs in ss["directions"].items():
            base = vecs.get("none", {}).get("0.0", {})
            parts = [f"unsteered {base.get('mean_gap') or 0:+.0f}%"]
            for vec, alphas in E3B_STEER_SHOW:
                for a in alphas:
                    c = vecs.get(vec, {}).get(str(a))
                    if c:
                        v, sc = vec.split("|")
                        parts.append(f"{E3B_NAME[v]}{'' if sc == 'prompt+reply' else '/' + sc}×{a:g} "
                                     f"{_pct(c.get('share_moved')).strip()}")
            for i in range(0, len(parts), 5):
                out.append(f"  {dname if i == 0 else '':<7} " + " · ".join(parts[i:i + 5]))
    return out


def stub_e3b(task, items):
    """Preview only: canned shapes, not a model."""
    if task == "e3b_build":
        return [{"layer": 44, "n_post": 117, "n_prompt": 106, "n_reply": 11, "reply_tokens": [], "n_nonmoney": 40,
                 "norm_v_prompt": 27.0, "norm_v_reply": 60.0, "cos_v_mean_u": 0.4, "cos_fixture_diffs_mean": 0.9,
                 "cos_fixture_diffs_min": 0.8}]
    out = []
    eff = {"own": 0.9, "v_pos": 0.7, "v_mean": 0.5, "random": 0.0, "nonmoney": 0.05, "none": 0.0}
    for it in items:
        g0, g1 = (0.41, 0.21) if it["direction"] == "x1+v" else (0.21, 0.41)
        unit = it["target"]["limit_units"]
        if task == "e3b_bridge":
            rows = [{"vector": v, "scope": sc, "alpha": a, "price": round((1 + g0 + eff[v] * min(1, a) * (g1 - g0)) * unit)}
                    for v, sc, a in (("none", "all", 0.0),) + E3B_BRIDGE_SHOW]
            out.append({"item_id": it["item_id"], "fixture": it["fixture"], "role": it["role"],
                        "direction": it["direction"], "base_limit": it["base_limit"],
                        "target_condition": it["target"]["condition"], "clean_consistent": True, "rows": rows,
                        "secs": 0.1})
        else:
            rows = [{"vector": "none", "scope": "prompt+reply", "alpha": 0.0, "errors": [], "action": "OFFER",
                     "finish": "stop", "aligned": True, "price": round((1 + g0) * unit)} for _ in range(16)]
            for vec, alphas in E3B_STEER_SHOW:
                v, sc = vec.split("|")
                for a in alphas:
                    rows.append({"vector": v, "scope": sc, "alpha": a, "errors": [], "action": "OFFER", "finish": "stop",
                                 "aligned": True, "price": round((1 + g0 + eff[v] * min(1, a) * (g1 - g0)) * unit)})
            out.append({"item_id": it["item_id"], "fixture": it["fixture"], "role": it["role"],
                        "direction": it["direction"], "base_limit": it["base_limit"],
                        "target_condition": it["target"]["condition"], "rows": rows, "secs": 0.2})
    return out


def report_e3b(s, preview):
    print("E3b steering incl. the reply's opening tokens (exploratory" + (", PREVIEW: canned shapes" if preview else "") + ")")
    for line in e3b_live({"e3b vectors": [s["vectors"]] if s.get("vectors") else []}):
        print("  " + line)
    a = s.get("bridge") or {}
    if a.get("cells"):
        print(f"  A. forced reply opening, greedy price (E2's setup), {a['n_fixtures']} fixtures; share of the gap moved:")
        for dname, cells in a["cells"].items():
            print(f"    {dname}")
            for vec, sc, al in E3B_BRIDGE_SHOW:
                c = cells.get(f"{vec}|{sc}|{al}")
                if c:
                    ci = f" [{100 * c['ci'][0]:+.0f}, {100 * c['ci'][1]:+.0f}]" if c.get("ci") else ""
                    print(f"      {E3B_NAME[vec]:<12} {E3B_SCOPE_LABEL[sc]:<13} ×{al:<4g} {_pct(c['fraction'])}{ci}"
                          + (f"  off-scale {c['off_scale']}/{c['n']}" if c["off_scale"] else ""))
    b = s.get("steer") or {}
    if b.get("directions"):
        print(f"  B. real sampled openings, steering the prompt and the reply as generated, {b['n_fixtures']} fixtures:")
        for dname, vecs in b["directions"].items():
            base = vecs.get("none", {}).get("0.0", {})
            print(f"    {dname}: unsteered gap {base.get('mean_gap') or 0:+.1f}% (n {base.get('n')})")
            for vec, alphas in E3B_STEER_SHOW:
                v, sc = vec.split("|")
                for al in alphas:
                    c = vecs.get(vec, {}).get(str(al))
                    if not c:
                        continue
                    g = "—" if c["mean_gap"] is None else f"{c['mean_gap']:+.1f}%"
                    ci = f" [{c['ci'][0]:+.1f}, {c['ci'][1]:+.1f}]" if c.get("ci") else ""
                    sh = f"  moved {_pct(c.get('share_moved')).strip()}" if c.get("share_moved") is not None else ""
                    print(f"      {E3B_NAME[v]:<12} {E3B_SCOPE_LABEL[sc]:<24} ×{al:<4g} gap {g}{ci}{sh}  valid "
                          f"{c['valid']}/{c['n']}  aligned {100 * (c.get('aligned_share') or 0):.0f}%"
                          + (f"  off-scale {c['off_scale'] + c['other_scale']}" if c["off_scale"] + c["other_scale"] else "")
                          + (f"  hit length {c['length']}" if c["length"] else ""))
        ba = s.get("steer_aligned_only") or {}
        if ba.get("directions"):
            print("    (aligned openings only:)")
            for dname, vecs in ba["directions"].items():
                parts = []
                for vec, alphas in E3B_STEER_SHOW:
                    for al in alphas:
                        c = vecs.get(vec, {}).get(str(al))
                        if c and c.get("share_moved") is not None:
                            v, sc = vec.split("|")
                            parts.append(f"{E3B_NAME[v]}{'' if sc == 'prompt+reply' else '/' + sc}×{al:g} "
                                         f"{_pct(c['share_moved']).strip()}")
                print(f"      {dname}: " + " · ".join(parts))


def clean_vs_e0(recs, e0_dir):
    """Do E2's unpatched readings (prefix cache, batched) reproduce E0's R2 prices on the same prompts?"""
    if not e0_dir or not recs:
        return None
    r2 = {}
    for line in open(os.path.join(e0_dir, "results.jsonl")):
        r = json.loads(line)
        if r.get("step") == "teacher":
            r2[(r["item_id"][:3], r["condition"])] = r.get("price_r2")
    same = n = 0
    for r in recs:
        c = next((x for x in r["rows"] if x["layer"] == "clean"), None)
        key = (r["fixture"], r["target_condition"])
        if c is not None and key in r2:
            n += 1
            same += c["price"] == r2[key]
    return {"same_price": same, "of": n, "e0_run": os.path.relpath(e0_dir, ROOT)}


def partial_r(xs, ys, zs):
    """corr(x, y) with z (the limit's size) partialled out of both."""
    rxy, rxz, ryz = pearson(xs, ys), pearson(xs, zs), pearson(ys, zs)
    if None in (rxy, rxz, ryz) or abs(rxz) >= 1 or abs(ryz) >= 1:
        return None
    return (rxy - rxz * ryz) / math.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))


def default_e0_run(model):
    for d in reversed(sorted(glob.glob(os.path.join(ROOT, "data", "interp", f"interp-e0-{model}-*")))):
        try:
            if json.load(open(os.path.join(d, "summary.json"))).get("fidelity", {}).get("pass"):
                return d
        except (OSError, ValueError):
            continue
    return None


# --- pod control API over the tunnel ---------------------------------------------
def control(port, method, path, obj=None, timeout=1800.0):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        body = None if obj is None else json.dumps(obj).encode("utf-8")
        conn.request(method, path, body=body, headers={"Content-Type": "application/json",
                                                       "Accept": "application/json", "Connection": "close"})
        resp = conn.getresponse()
        data = resp.read()
        try:
            return resp.status, json.loads(data)
        except ValueError:
            return resp.status, {"error": "non-json reply"}
    except (OSError, http.client.HTTPException) as e:
        return None, {"error": f"{type(e).__name__}: {e}"}
    finally:
        conn.close()


def stub(task, items):
    """Preview only: canned numbers near the 31B vLLM run. Not a model."""
    out = []
    for it in items:
        if task == "render":
            out.append({"item_id": it["item_id"], "prompt_text": "(preview)", "n_tokens": 571})
            continue
        gap = {("seller", "D2"): 0.41, ("seller", "D8"): 0.21, ("buyer", "D2"): 0.46,
               ("buyer", "D8"): 0.37}[(it["role"], it["condition"])]
        price = round((1 + gap if it["role"] == "seller" else 1 - gap) * it["limit_units"])
        base = {"item_id": it["item_id"], "role": it["role"], "condition": it["condition"],
                "limit_units": it["limit_units"]}
        if task == "teacher":
            out.append({**base, "price_r2": price, "gap_r2": gap})
        else:
            out.append({**base, "errors": [], "action": "OFFER", "price": price, "gap": gap, "constrained": True})
    return out


SPARK = "▁▂▃▄▅▆▇█"
E2_POS = ("final", "post", "span")
E2_POS_LABEL = {"final": "last token", "post": "after limit", "span": "limit digits"}


def spark(vals, lo=0.0, hi=1.0):
    """One character per layer: ▁ = lo or less, █ = hi or more, · = no reading yet."""
    out = []
    for v in vals:
        if v is None:
            out.append("·")
            continue
        x = (min(max(v, lo), hi) - lo) / (hi - lo)
        out.append(SPARK[min(7, int(x * 8))])
    return "".join(out)


def peak(layers, vals):
    pairs = [(v, L) for L, v in zip(layers, vals) if v is not None]
    return max(pairs) if pairs else (None, None)


def e1e2_live(got, layers):
    out = []
    fit = got.get("e1 fit") or []
    if fit:
        s1 = summarize_e1(fit[0])
        for pos in ("digit", "final"):
            w = s1["within"][pos]
            out.append(f"E1 {pos:<6} probe R² by layer  {spark(w['r2'])}  best L{w['best_layer']} {w['best_r2']:.2f}")
    recs = got.get("e2 patch") or []
    if recs:
        s2 = summarize_e2(recs, ci=False)
        out.append(f"E2 share of the ×1/×10⁶ gap restored, layer {layers[0]}→{layers[-1]} (▁ 0, █ 100%), "
                   f"{s2['n_fixtures']} fixtures")
        for dname, d in s2["directions"].items():
            for pos in E2_POS:
                cells = d["cells"].get(pos, {})
                fr = [cells.get(str(L), {}).get("fraction") for L in layers]
                v, L = peak(layers, fr)
                tag = f"peak L{L} {100 * v:.0f}%" if v is not None else ""
                out.append(f"E2 {dname:<8} {E2_POS_LABEL[pos]:<12} {spark(fr)}  {tag}")
    return out


def stub_e1e2(task, items, layers=None):
    """Preview only: canned shapes, not a model."""
    if task == "e1_capture":
        return [{"item_id": it["item_id"], "kind": it["kind"], "n_tokens": 578, "positions": {}, "secs": 0.3}
                for it in items]
    if task == "e1_fit":
        r2 = [round(min(0.99, 0.2 + L / 20), 3) for L in range(60)]
        return [{"within": {p: {"alpha": 0.1, "r2": r2} for p in ("digit", "prompt_end", "final")},
                 "transfer": {}, "pred_within": {}, "items": [], "amounts": {}}]   # no predictions: no beyond-scale
    out = []
    for it in items:
        lo, hi = (0.21, 0.41) if it["direction"] == "x1->x1e6" else (0.41, 0.21)
        rows = []
        for p in E2_POS:
            for L in layers or range(60):
                f = 1 / (1 + math.exp(-(L - (30 if p != "span" else 12)) / 3)) if p != "span" else \
                    1 / (1 + math.exp((L - 20) / 3))
                g = lo + f * (hi - lo)
                rows.append({"layer": L, "pos": p, "price": round((1 + g) * it["target"]["limit_units"]), "cont": ""})
        rows.append({"layer": "clean", "pos": None, "price": round((1 + lo) * it["target"]["limit_units"])})
        out.append({"item_id": it["item_id"], "fixture": it["fixture"], "role": it["role"],
                    "direction": it["direction"], "base_limit": it["base_limit"],
                    "source_condition": it["source"]["condition"], "target_condition": it["target"]["condition"],
                    "clean_consistent": True, "rows": rows, "secs": 0.2})
    return out


# --- the live panel ---------------------------------------------------------------
class InterpPanel(J.Panel):
    """The probes' panel, with interp lines instead of game lines."""

    def __init__(self, title, tty, preview, stages=STAGES, exp="E0"):
        super().__init__(title, stages, tty, preview)
        self.exp = exp
        self.steps, self.step_done, self.step_secs = [], {}, {}
        self.pod, self.gpu, self.live = {}, None, []
        self.valid = self.replies_n = self.slips = self.constrained = 0
        self.eta_s = None

    def lines(self):
        w = max(60, min(110, shutil.get_terminal_size((110, 30)).columns - 1))
        icon = {"todo": "·", "run": "▶", "ok": "✓", "fail": "✗"}
        rule = " " + "─" * (w - 2)
        out = [f" scale-invariance · {self.model}" + (" · PREVIEW (nothing rented)" if self.preview else ""), rule,
               " Stages   " + "  ".join(f"{icon[self.stage[s]]} {s}" for s in self.stages),
               f" Now      {self.now}"]
        if self.steps:
            out.append(" Steps    " + "  ·  ".join(
                f"{lab} {min(self.step_done.get(lab, 0), tot)}/{tot}" + (" ✓" if self.step_done.get(lab, 0) >= tot else "")
                for lab, tot in self.steps))
            tot = sum(t for _, t in self.steps)
            n = sum(min(self.step_done.get(lab, 0), t) for lab, t in self.steps)
            fill = int(30 * n / tot) if tot else 0
            eta = f"   ~{math.ceil(self.eta_s / 60)} min left" if self.eta_s else ""
            out.append(f" Progress [{'█' * fill}{'░' * (30 - fill)}]  {n} / {tot} items   {100 * n // max(1, tot)}%{eta}")
        else:
            out.append(f" Progress ({self.exp} not started)")
        speed = []
        for lab, secs in self.step_secs.items():
            if secs and lab != "render":
                speed.append(f"{lab} {st.mean(secs[-CHUNK:]):.1f} s/item")
        if self.gpu:
            speed.append(f"GPU {self.gpu.get('used_gib')}/{self.gpu.get('total_gib')} GiB")
        if speed:
            out.append(" Speed    " + " · ".join(speed))
        if self.pod:
            p = self.pod
            out.append(f" Pod      transformers {p.get('transformers_version', '?')} · {p.get('model_class', '?')} · "
                       f"{p.get('num_hidden_layers', '?')} layers · {p.get('weights_check', '?')}")
            ref = p.get("vllm_ref") or {}
            diag = p.get("diagnosis") or {}
            if diag or ref:
                rn = diag.get("vllm_plain", ref.get("plain_nll"))
                vr = diag.get("vllm_reply") or []
                parts = []
                for k, v in diag.items():
                    if isinstance(v, dict) and "plain" in v:
                        parts.append(f"{k} {v['reply_diff']:.3f}" if "reply_diff" in v else f"{k} {v['plain']:.2f}")
                head = ("vLLM loss " + (f"{rn:.2f}" if isinstance(rn, float) else str(ref.get("error", "—"))[:40])
                        + (f", reply {sum(vr) / len(vr):.3f}" if vr else ""))
                out.append(f" Check    → {p.get('attention_choice') or 'NONE USABLE'} · {head} · gap to vLLM: "
                           + (" · ".join(parts) or "—"))
            for g in p.get("greedy_vs_vllm") or []:
                out.append(f" Greedy   {g['item_id']}: {g.get('agree_tokens', '?')} of {g.get('of', '?')} reply tokens "
                           f"identical to vLLM · same prompt ids {g.get('same_prompt_ids', '?')}")
            out.append(f" Grammar  {p.get('grammar', '—')}")
        if self.replies_n:
            out.append(f" Replies  {self.valid} of {self.replies_n} valid ({100 * self.valid / self.replies_n:.1f}%)"
                       f" · schema-constrained {100 * self.constrained // self.replies_n}% · digit slips {self.slips}")
        out += [" " + line for line in self.live]
        if self.t_created:
            up = time.time() - self.t_created
            left = (self.lifetime_s - up) if self.lifetime_s else 0
            cost = self.cost_final if self.cost_final is not None else (self.price or 0) * up / 3600
            out.append(f" Machine  up {J.mmss(up)} · auto-delete in {max(0, int(left // 60))} min"
                       f" · cost so far ≈ ${cost:.2f} (worst case ${self.worst})")
        out.append(rule)
        out += [" " + line for line in self.log] or [" "]
        out.append(f" Exploratory (plan_mechinterp {'§2' if self.exp == 'E0' else '§4'}). "
                   "Ctrl-C stops after this chunk and deletes the machine.")
        return [line[:w] for line in out]

    def draw(self):
        if self.tty:
            return super().draw()
        with self.lock:
            key = (tuple(self.stage.values()), tuple(sorted(self.step_done.items())))
            if key != self.last_plain:
                self.last_plain = key
                done = "/".join(s for s in self.stages if self.stage[s] == "ok")
                steps = " ".join(f"{lab} {self.step_done.get(lab, 0)}/{t}" for lab, t in self.steps)
                print(f"  [{done}] {steps}".rstrip(), flush=True)


class _InterpParse:
    """Launcher lines specific to --mode interp; the rest go to run_job.Machine.parse."""

    def parse(self, line):
        p, s = self.panel, line.strip()
        if s.startswith("Creating pod "):
            self.pod_name = s.split()[2]
            super().parse(line)
        elif "WEIGHTS_OK" in s:
            self.weights_line = s
            p.set_stage("model files", "ok"); p.set_stage("interp code", "run")
            n = re.search(r"files=(\d+) bytes=(\d+)", s)
            p.note(f"model files OK: {n.group(1)} files, {int(n.group(2)) / 1e9:.1f} GB, all match the manifest"
                   if n else s)
            p.set_now("unpacking and checking the interp code")
        elif s.startswith("INTERP_CODE_OK"):
            p.set_stage("interp code", "ok"); p.set_stage("model load", "run")
            p.note("interp code unpacked; its hash matches")
            p.set_now("vLLM reference run, then transformers load and checks (several minutes)")
        elif s.startswith("Tunnel open"):
            p.note("SSH tunnel open")
        elif s.startswith("INTERP_READY"):
            p.note(s.replace("INTERP_READY", "model ready:"))
        elif s.startswith("SESSION_READY"):
            for stg in ("model files", "interp code", "model load"):
                p.set_stage(stg, "ok")
            p.note("control API reachable through the tunnel")
            self.session_ready = True
            self.ready.set()
        else:
            super().parse(line)


class InterpMachine(_InterpParse, J.Machine):
    pass


class InterpFakeMachine(_InterpParse, J.FakeMachine):
    pass


# --- E0 ----------------------------------------------------------------------------
def live_lines(sampled, vidx, roles):
    """Running seller/buyer means next to the vLLM run's on exactly the items done so far."""
    out = []
    for role in roles:
        cells = {}
        for c in ("D2", "D8"):
            rs = [r for r in sampled if r["role"] == role and r["condition"] == c and r.get("gap") is not None]
            vs = [vidx[r["item_id"]] for r in rs if r["item_id"] in vidx and vidx[r["item_id"]].get("gap") is not None]
            cells[c] = (st.mean(100 * r["gap"] for r in rs) if rs else None,
                        st.mean(100 * r["gap"] for r in vs) if vs else None, len(rs))
        f = lambda v: "  —  " if v is None else f"{v:+.1f}%"     # noqa: E731
        (p2, v2, n2), (p8, v8, n8) = cells["D2"], cells["D8"]
        sh = (p8 - p2) if p2 is not None and p8 is not None else None
        vsh = (v8 - v2) if v2 is not None and v8 is not None else None
        out.append(f"E0 {role:<6} ×1 {f(p2)} (vLLM {f(v2)}, n {n2})   ×10⁶ {f(p8)} (vLLM {f(v8)}, n {n8})   "
                   f"shift {f(sh)} (vLLM {f(vsh)})")
    return out


def e0_steps(ctx):
    return [("render", "render", ctx["sampled_items"][:4], {}, CHUNK),
            ("sampled", "generate", ctx["sampled_items"], {"greedy": False, "schema": ctx["schema"]}, CHUNK),
            ("greedy", "generate", ctx["greedy_items"], {"greedy": True, "schema": ctx["schema"]}, CHUNK),
            ("teacher", "teacher", ctx["greedy_items"], {}, CHUNK)]


def layer_list(n_layers, step):
    out = list(range(0, n_layers, step))
    return out if out[-1] == n_layers - 1 else out + [n_layers - 1]


def e1e2_steps(ctx):
    a = ctx["args"]
    return [("e1 capture", "e1_capture", ctx["e1_items"], {}, CHUNK),
            ("e1 fit", "e1_fit", [{"item_id": "fit"}], {}, 1),
            ("e2 patch", "e2_patch", ctx["e2_items"], {"batch": a.batch, "layers": ctx["e2_layers"]}, 1)]


def e3_steps(ctx):
    a = ctx["args"]
    return [("e3 vectors", "e3_build", ctx["e3_vec_items"], {"layer": a.steer_layer}, len(ctx["e3_vec_items"])),
            ("e3 steer", "e3_steer", ctx["e3_items"], {"batch": a.batch, "schema": ctx["schema"],
                                                       "samples": a.samples_e3, "alphas": list(E3_ALPHAS),
                                                       "vectors": list(E3_VECTORS)}, 1)]


def e3b_steps(ctx):
    a = ctx["args"]
    return [("e3b vectors", "e3b_build", ctx["e3_vec_items"], {"layer": a.steer_layer}, len(ctx["e3_vec_items"])),
            ("e3b bridge", "e3b_bridge", ctx["e3_items"], {"batch": a.batch}, 4),
            ("e3b steer", "e3b_steer", ctx["e3_items"], {"batch": a.batch, "schema": ctx["schema"]}, 1)]


STEP_GUESS_S = {"e3b vectors": 60.0, "e3b bridge": 3.0, "e3b steer": 25.0, "e1 capture": 0.5, "e1 fit": 60.0, "e2 patch": 25.0, "e3 vectors": 40.0, "e3 steer": 25.0}      # before a step has its own timing


def run_e0(ctx):
    """Runs the experiment's steps in chunks; updates the panel; writes results as they arrive."""
    panel, a = ctx["panel"], ctx["args"]
    steps = ctx.get("steps") or e0_steps(ctx)
    got = {lab: [] for lab, *_ in steps}
    panel.steps = [(lab, len(items)) for lab, _, items, _, _ in steps]
    vidx = {r["item_id"]: r for r in ctx.get("vllm_recs", [])}
    roles = ctx["roles"]
    t0 = time.time()
    for label, task, items, kw, chunk_n in steps:
        for i in range(0, len(items), chunk_n):
            if ctx["interrupts"][0]:
                return got, "stopped by you"
            if panel.t_created and panel.lifetime_s and \
                    panel.t_created + panel.lifetime_s - time.time() < LIFETIME_GUARD_S:
                return got, "machine time nearly up; no new chunk started"
            if (time.time() - t0) / 60 > a.budget_min:
                return got, f"--budget-min {a.budget_min:g} reached"
            if ctx["machine"] is not None and not ctx["machine"].alive():
                return got, "the machine session ended"
            chunk = items[i:i + chunk_n]
            panel.set_now(f"{panel.exp} {label}: " + (f"item {i + 1} of {len(items)} ({chunk[0]['item_id']})"
                                                       if len(chunk) == 1 and len(items) > 1 else
                                                       f"items {i + 1}-{i + len(chunk)} of {len(items)}"))
            tc = time.time()
            if a.preview:
                time.sleep(0.15)
                res = ctx["plugin"].stub(task, chunk, ctx) if ctx.get("plugin") else \
                    stub(task, chunk) if task in ("render", "generate", "teacher") else \
                    stub_e3b(task, chunk) if task.startswith("e3b") else \
                    stub_e3(task, chunk) if task.startswith("e3") else stub_e1e2(task, chunk, ctx.get("e2_layers"))
                gpu = {"used_gib": 60.9, "total_gib": 95.6}
            else:
                code, r = control(ctx["port"], "POST", "/run", {"task": task, "items": chunk,
                                                                 "settings": ctx["settings"],
                                                                 "model_name": a.model, **kw})
                if code != 200:
                    why = f"{label}: HTTP {code}: {r.get('error')}"
                    if r.get("traceback"):
                        with open(os.path.join(ctx["out_dir"], "pod_error.txt"), "w") as fh:
                            fh.write(r["traceback"])
                        why += " (traceback in pod_error.txt)"
                    return got, why
                res, gpu = r["results"], r.get("gpu")
            got[label] += res
            with open(os.path.join(ctx["out_dir"], "results.jsonl"), "a") as fh:
                for x in res:
                    fh.write(json.dumps({"step": label, **x}) + "\n")
            with panel.lock:
                panel.step_done[label] = i + len(chunk)          # items processed (a step may return one record)
                panel.step_secs.setdefault(label, []).append((time.time() - tc) / max(1, len(chunk)))
                panel.gpu = gpu or panel.gpu
                if task == "generate":
                    panel.replies_n += len(res)
                    panel.valid += sum(1 for x in res if not x.get("errors"))
                    panel.constrained += sum(1 for x in res if x.get("constrained"))
                    panel.slips += sum(1 for x in res if x.get("gap") is not None
                                       and is_slip(x["role"], x["price"], x["limit_units"]))
                panel.live = ctx["live"](got) if ctx.get("live") else \
                    [] if ctx.get("plugin") else live_lines(got["sampled"], vidx, roles)
                panel.eta_s = eta(panel)
            if ctx.get("save_partial"):
                ctx["save_partial"](got)
            write_json(os.path.join(ctx["out_dir"], "progress.json"),
                       {"t": rr.utc_now(), "steps": dict(panel.step_done), "of": dict(panel.steps),
                        "valid": panel.valid, "replies": panel.replies_n, "slips": panel.slips,
                        "live": panel.live, "gpu": panel.gpu})
            panel.draw()
        if ctx.get("after_step"):              # plug-in gate: a reason string stops the session here;
            why = ctx["after_step"](label, got, ctx)     # it may also edit later steps' kw dicts in place
            if why:
                panel.note(f"{label}: {why}")
                return got, why
    return got, None


def eta(panel):
    """Seconds left: remaining items x the recent per-item time of each step (teacher ~1/3 of generate)."""
    gen = panel.step_secs.get("sampled") or panel.step_secs.get("greedy")
    left = 0.0
    for lab, tot in panel.steps:
        rem = tot - panel.step_done.get(lab, 0)
        if rem <= 0:
            continue
        secs = panel.step_secs.get(lab)
        if secs:
            per = st.mean(secs[-CHUNK:])
        elif lab in STEP_GUESS_S:
            per = STEP_GUESS_S[lab]
        elif gen:
            per = st.mean(gen[-CHUNK:]) * (0.35 if lab == "teacher" else 1.0)
        else:
            return None
        left += rem * per
    return left


def load_plugins():
    """Experiments in scripts/interp/exp_<name>.py, selected with --experiment <name>.

    A plug-in module provides (see config/interp/README.md, "Plug-ins"):
      add_args(ap)                       optional: its own flags (prefix them with its name)
      configure(a)                       optional: right after parsing (e.g. a.roles = "buyer")
      setup(a, settings, schema, li)  -> {"exp", "plan", "inputs", "ctx", "steps", "live"?}
      on_pod_info(ctx, pod_info)         optional: once the pod has reported (e.g. its layer count)
      stub(task, items, ctx)             preview: canned results, not a model
      summarize(got, ctx) -> dict;  report(summary, preview);  manifest(ctx) -> dict (optional)
    """
    out = {}
    for path in sorted(glob.glob(os.path.join(HERE, "interp", "exp_*.py"))):
        name = os.path.basename(path)[4:-3]
        spec = importlib.util.spec_from_file_location(f"interp_exp_{name}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        out[name] = mod
    return out


def pod_plugins(experiment, extra=()):
    """The pod plug-ins a session ships (launch_pod.sh --interp-plugins): scripts/interp/tasks_<experiment>.py, the
    laptop plug-in's POD_PLUGINS (extra: pod code it reuses under another name), and the tasks_* modules they import,
    transitively; "none" for the built-in experiments (their tasks are in tasks.py)."""
    here, need = os.path.join(HERE, "interp"), set()
    todo = [experiment] if os.path.exists(os.path.join(HERE, "interp", f"tasks_{experiment}.py")) else []
    todo += list(extra)
    while todo:
        n = todo.pop()
        if n in need:
            continue
        need.add(n)
        for node in ast.walk(ast.parse(open(os.path.join(here, f"tasks_{n}.py")).read())):
            mods = [x.name for x in node.names] if isinstance(node, ast.Import) else \
                [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
            todo += [m[len("tasks_"):] for m in mods if m.startswith("tasks_")]
    return ",".join(sorted(need)) or "none"


def shipped_tasks(plugs):
    """Task names the pod will have: TASKS = {...} / TASKS.update({...}) keys in tasks.py and the shipped plug-ins."""
    files = ["tasks"] + ([] if plugs == "none" else [f"tasks_{n}" for n in plugs.split(",")])
    out = set()
    for f in files:
        for node in ast.walk(ast.parse(open(os.path.join(HERE, "interp", f + ".py")).read())):
            d = node.value if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "TASKS" for t in node.targets) else \
                node.args[0] if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "update" \
                and getattr(node.func.value, "id", None) == "TASKS" and node.args else None
            if isinstance(d, ast.Dict):
                out |= {k.value for k in d.keys if isinstance(k, ast.Constant)}
    return out


def main(argv):
    plugins = load_plugins()
    ap = argparse.ArgumentParser(description="One-terminal interp E0 session (see the top of this file).")
    ap.add_argument("--model", default="gemma-4-31b-it")
    ap.add_argument("--fixtures", type=int, default=30)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--roles", default="seller,buyer")
    ap.add_argument("--lifetime-min", type=int, default=90)
    ap.add_argument("--slot", type=int, default=8, help="default 8 (the MI track's slot; slot 0 is shared)")
    ap.add_argument("--attach", action="store_true", help="use a session launched separately (no launch here)")
    ap.add_argument("--preview", action="store_true", help="free: fake machine, canned numbers")
    ap.add_argument("--budget-min", type=float, default=60.0, help="no new chunk after this many E0 minutes")
    ap.add_argument("--vllm-run", help="digits-probe run to compare against (default: newest for --model)")
    ap.add_argument("--experiment", choices=("e0", "e1e2", "e3", "e3b") + tuple(plugins), default="e0",
                    help="e0: fidelity (default); e1e2: probe + patching; e3: steering; "
                         "e3b: steering incl. the reply's opening tokens (sellers)")
    ap.add_argument("--steer-layer", type=int, default=44, help="e3: layer (default 44, mid E2 plateau 41-47; X52)")
    ap.add_argument("--samples-e3", type=int, default=2, help="e3: openings per (vector, alpha)")
    ap.add_argument("--batch", type=int, default=16, help="e1e2: patched runs per forward pass")
    ap.add_argument("--layer-step", type=int, default=1, help="e1e2: patch every Nth layer (1 = all)")
    ap.add_argument("--print-pod-plugins", action="store_true", help="print the plug-ins this experiment ships, then exit")
    for mod in plugins.values():
        if hasattr(mod, "add_args"):
            mod.add_args(ap)
    a = ap.parse_args(argv)
    P = plugins.get(a.experiment)
    plugs = pod_plugins(a.experiment, getattr(P, "POD_PLUGINS", ()))
    if a.print_pod_plugins:
        print(plugs)
        return 0
    if a.experiment in ("e1e2", "e3", "e3b"):
        a.roles = "seller"
    if P is not None and hasattr(P, "configure"):
        P.configure(a)
    roles = tuple(a.roles.split(","))
    if not 1 <= a.fixtures <= 30 or not 1 <= a.samples <= 8 or not set(roles) <= {"seller", "buyer"} \
            or not 0 <= a.slot <= 9:
        print("bad --fixtures (1-30), --samples (1-8), --roles or --slot")
        return 2
    J.use_slot(a.slot)
    pins = json.load(open(os.path.join(ROOT, "config", "pins.json")))
    pin = next((m for m in pins["models"] if m["name"] == a.model), None)
    if pin is None:
        print(f"--model must be one of: {', '.join(m['name'] for m in pins['models'])}")
        return 2

    settings_path = os.path.join(ROOT, "prereg", "settings.json")
    settings = json.load(open(settings_path))
    schema = ge.response_schema(settings["max_message_chars"]["value"])
    meta, sampled_items = build_items(settings, a.fixtures, a.samples, list(CONDITIONS), roles)
    greedy_items = [it for it in sampled_items if it["sample"] == 1]
    vllm_dir = a.vllm_run or default_vllm_run(a.model)
    vllm_recs = load_vllm(vllm_dir, CONDITIONS) if vllm_dir else []
    want = {it["item_id"] for it in sampled_items}
    vllm_recs = [r for r in vllm_recs if r["item_id"] in want]
    n_total = 4 + len(sampled_items) + 2 * len(greedy_items)
    plan = (f"E0 on {a.model}: {a.fixtures} fixtures x {'/'.join(roles)} x D2/D8; {len(sampled_items)} sampled"
            f" + {len(greedy_items)} greedy openings + {len(greedy_items)} teacher-forced = {n_total} items;"
            f" compared with {os.path.relpath(vllm_dir, ROOT) if vllm_dir else 'no vLLM run'}"
            f" ({len(vllm_recs)} matching first moves)")
    exp, stages = "E0", STAGES
    e1_items = e2_items = e2_layers = None
    if a.experiment == "e1e2":
        e2_layers = layer_list(60, a.layer_step)             # replaced by the pod's own count once it reports
        e1_items, e2_items = build_e1_items(settings, a.fixtures), build_e2_items(settings, a.fixtures)
        n_e1 = sum(1 for x in e1_items if x["kind"] == "negotiation")
        exp, stages = "E1+E2", STAGES[:4] + ("E1+E2",) + STAGES[5:]
        plan = (f"E1+E2 on {a.model} (sellers): E1 = {n_e1} negotiation prompts (limit x10^0..10^6) + "
                f"{len(e1_items) - n_e1} plain sentences, probes fitted on the pod; E2 = {len(e2_items)} "
                f"fixture x direction items, each {len(e2_layers)} layers x 3 positions patched "
                f"({len(e2_items) * len(e2_layers) * 3} patched runs, batch {a.batch})")
    e3_vec_items = e3_items = None
    if a.experiment == "e3b":
        e3_vec_items, e3_items = build_e3_items(settings, a.fixtures)
        exp, stages = "E3b", STAGES[:4] + ("E3b",) + STAGES[5:]
        plan = (f"E3b on {a.model} (sellers): layer {a.steer_layer}, vectors over the prompt's tokens after the limit "
                f"AND the reply's opening tokens, from {a.fixtures} fixture pairs (leave-one-fixture-out) + non-money pairs; "
                f"A = {len(e3_items)} items x {len(E3B_BRIDGE_SHOW) + 1} rows in E2's forced-opening setup (greedy); "
                f"B = {len(e3_items)} items x 48 real sampled openings steered on the prompt and the reply; batch {a.batch}")
    if P is not None:
        plug = P.setup(a, settings, schema, sys.modules[__name__])
        exp, plan = plug["exp"], plug["plan"]
        stages = STAGES[:4] + (exp,) + STAGES[5:]
    if a.experiment == "e3":
        e3_vec_items, e3_items = build_e3_items(settings, a.fixtures)
        n_rows = len(E3_VECTORS) * len(E3_ALPHAS) * a.samples_e3
        exp, stages = "E3", STAGES[:4] + ("E3",) + STAGES[5:]
        plan = (f"E3 on {a.model} (sellers): steering at layer {a.steer_layer} on the text after the limit; "
                f"vectors from {a.fixtures} fixture pairs (leave-one-fixture-out) + "
                f"{sum(1 for x in e3_vec_items if x['kind'] == 'nonmoney')} non-money pairs; {len(e3_items)} "
                f"fixture x direction items, each {n_rows} steered + unsteered sampled openings "
                f"({'/'.join(E3_VECTORS)} x alpha {'/'.join(str(x) for x in E3_ALPHAS)}), batch {a.batch}")

    worst, price = "0.00", 0.0
    if a.preview:
        print(f"PREVIEW: fake machine on this laptop, canned numbers. Nothing is rented.\n  {plan}\n")
    elif not a.attach:
        if J.uncommitted():
            print("Note: uncommitted changes exist; the run records the commit and the pod reports its code hashes.")
        r = subprocess.run(["bash", J.LAUNCHER, "--mode", "interp", "--model", a.model,
                            "--lifetime-min", str(a.lifetime_min), "--slot", str(a.slot), "--interp-plugins", plugs],
                           capture_output=True, text=True, timeout=180)
        out = r.stdout + r.stderr
        print("\n".join(l for l in out.rstrip().splitlines() if not l.startswith("Dry run:")).rstrip())
        m = re.search(r"--confirm-cost ([0-9.]+)", out)
        if r.returncode != 0 or not m:
            print("\nNot launching (the launcher's checks above say why).")
            return 3
        worst, price = m.group(1), float(re.search(r"\$([0-9.]+)/hr", out).group(1))
        print(f"\nThis session: {plan}.\nThe machine deletes itself after {a.lifetime_min} min at the latest.")
        try:
            typed = input(f"Type the worst-case cost ({worst}) to launch, or anything else to cancel: ").strip()
        except (EOFError, KeyboardInterrupt):
            typed = ""
        if typed.lstrip("$") != worst:
            print("Cancelled. Nothing was created.")
            return 0
        print()
        if shutil.which("caffeinate"):
            subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_id = f"interp-{a.experiment}-{a.model}-{stamp}" + ("-preview" if a.preview else "")
    out_dir = os.path.join(ROOT, "data", "interp", run_id)
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "results.jsonl"), "w").close()
    if a.experiment == "e0":
        write_jsonl(os.path.join(out_dir, "inputs.jsonl"),
                    [{k: it[k] for k in ("item_id", "fixture", "role", "condition", "sample", "base_limit",
                                         "limit_units", "seed", "messages")} for it in sampled_items])
    elif P is not None:
        write_jsonl(os.path.join(out_dir, "inputs.jsonl"), plug["inputs"])
    elif a.experiment == "e1e2":
        write_jsonl(os.path.join(out_dir, "inputs.jsonl"),
                    [{"step": "e1", **it} for it in e1_items] + [{"step": "e2", **it} for it in e2_items])
    else:
        write_jsonl(os.path.join(out_dir, "inputs.jsonl"),
                    [{"step": "vectors", **it} for it in e3_vec_items] + [{"step": "steer", **it} for it in e3_items])

    panel = InterpPanel(f"interp {exp} · {a.model} · slot {a.slot}", sys.stdout.isatty(), a.preview,
                        stages=stages, exp=exp)
    panel.lifetime_s, panel.price, panel.worst = a.lifetime_min * 60, price, worst
    machine = None
    if not a.attach:
        os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
        cls = InterpFakeMachine if a.preview else InterpMachine
        machine = cls(panel, os.path.join(ROOT, "logs", f"interp_{stamp}.launcher.out"), slot=a.slot,
                      manifest_tsv=os.path.join(ROOT, "config", "manifests", a.model + ".tsv"))
    interrupts = [0]

    def on_int(signum, frame):
        interrupts[0] += 1
        panel.note("stop requested: finishing this chunk, then deleting the machine" if interrupts[0] == 1
                   else "stop requested again: not waiting for the chunk")
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, on_int)
    threading.Thread(target=panel.ticker, daemon=True).start()
    panel.draw()

    ctx = {"panel": panel, "args": a, "machine": machine, "port": 18000 + a.slot, "settings": settings,
           "schema": schema, "sampled_items": sampled_items, "greedy_items": greedy_items,
           "vllm_recs": vllm_recs, "roles": roles, "out_dir": out_dir, "interrupts": interrupts}
    if a.experiment == "e1e2":
        ctx.update(e1_items=e1_items, e2_items=e2_items, e2_layers=e2_layers,
                   live=lambda got: e1e2_live(got, e2_layers))
        ctx["steps"] = e1e2_steps(ctx)
    if a.experiment == "e3":
        ctx.update(e3_vec_items=e3_vec_items, e3_items=e3_items, live=e3_live)
        ctx["steps"] = e3_steps(ctx)
    if a.experiment == "e3b":
        ctx.update(e3_vec_items=e3_vec_items, e3_items=e3_items, live=e3b_live)
        ctx["steps"] = e3b_steps(ctx)
    if P is not None:
        ctx.update(plug.get("ctx") or {}, plugin=P)
        if plug.get("live"):
            ctx["live"] = plug["live"]
        if plug.get("after_step"):
            ctx["after_step"] = plug["after_step"]
        ctx["steps"] = plug["steps"](ctx)
    missing = sorted({task for _lab, task, *_ in (ctx.get("steps") or e0_steps(ctx))} - shipped_tasks(plugs))
    if missing:                                    # before any machine exists
        print(f"Not launching: the pod code (plug-ins {plugs}) has no task {', '.join(missing)}.")
        shutil.rmtree(out_dir, ignore_errors=True)
        return 2
    got, stopped, pod_info = {lab: [] for lab, *_ in (ctx.get("steps") or e0_steps(ctx))}, None, {"preview": a.preview}
    try:
        if machine is not None:
            machine.start(["--mode", "interp", "--model", a.model, "--lifetime-min", str(a.lifetime_min),
                           "--slot", str(a.slot), "--interp-plugins", plugs, "--confirm-cost", worst])
            while not machine.ready.wait(1.0):
                if interrupts[0]:
                    raise J.JobStop("stopped by you before E0 began")
            if not machine.alive() or machine.failed:
                raise J.JobStop(machine.failed or "the launcher ended before the model was ready")
        if not a.preview:
            code, h = control(ctx["port"], "GET", "/health", timeout=15)
            pod_info = (h or {}).get("info") or {}
            if code != 200 or h.get("ok") is not True or "usable" not in pod_info:
                raise J.JobStop(f"the control API on slot {a.slot} is not the interp pod ({h.get('error', 'wrong reply')})")
            write_json(os.path.join(out_dir, "pod_info.json"), pod_info)
            panel.pod = pod_info
            if not pod_info.get("usable"):
                panel.set_stage("model load", "fail")
                raise J.JobStop("the pod failed its checks: " + str(pod_info.get("unusable_why")))
        else:
            pod_info = {"transformers_version": "(preview)", "model_class": "(preview)", "num_hidden_layers": 60,
                        "weights_check": "(preview)", "usable": True, "attention_choice": "default",
                        "vllm_ref": {"plain_nll": 2.71},
                        "diagnosis": {"vllm_plain": 2.71, "vllm_reply": [0.21, 0.24],
                                      "default": {"plain": 2.72, "reply": [0.21, 0.24], "reply_diff": 0.004},
                                      "eager": {"plain": 2.72, "reply": [0.21, 0.24], "reply_diff": 0.006}},
                        "greedy_vs_vllm": [{"item_id": "Y01-S-D2-s1", "agree_tokens": 41, "of": 41,
                                            "same_prompt_ids": True}],
                        "grammar": "(preview)"}
        panel.pod = pod_info
        if a.experiment == "e1e2" and pod_info.get("num_hidden_layers"):
            e2_layers[:] = layer_list(int(pod_info["num_hidden_layers"]), a.layer_step)
            ctx["steps"] = e1e2_steps(ctx)
        if P is not None and hasattr(P, "on_pod_info"):
            P.on_pod_info(ctx, pod_info)
            ctx["steps"] = plug["steps"](ctx)
        write_json(os.path.join(out_dir, "run_manifest.json"), {
            "schema": "interp_run/1", "run_id": run_id, "experiment": exp, "not_preregistered": True,
            "preview": a.preview, "model": a.model, "repo": pin["repo"], "revision": pin["revision"],
            "image": "vllm/vllm-openai@" + pins["vllm_image"]["index_digest"], "pod": pod_info,
            "git_commit": git_commit(), "settings_sha256": sha256_file(settings_path), "response_schema": schema,
            "conditions": list(CONDITIONS), "roles": list(roles), "fixtures": meta, "samples": a.samples,
            "e2_layers": e2_layers, "batch": a.batch if a.experiment != "e0" else None,
            "steer_layer": a.steer_layer if a.experiment in ("e3", "e3b") else None,
            "e3": {"alphas": E3_ALPHAS, "vectors": E3_VECTORS, "samples": a.samples_e3}
            if a.experiment == "e3" else None,
            "compare_vllm_run": vllm_dir and os.path.relpath(vllm_dir, ROOT), "slot": a.slot, "pod_plugins": plugs,
            "plan": "docs/plan_mechinterp.md §2", "started_utc": rr.utc_now(),
            "plugin": (P.manifest(ctx) if hasattr(P, "manifest") else {"name": a.experiment}) if P is not None else None})
        panel.set_stage(exp, "run")
        panel.note(f"{exp} started: " + (plan.split(";")[1].strip() if exp == "E0" else plan.split(": ", 1)[-1][:90]))
        box = {}
        def work():
            try:
                box.update(zip(("got", "stopped"), run_e0(ctx)))
            except Exception as e:             # noqa: BLE001  report the real reason, keep partial results
                import traceback
                with open(os.path.join(out_dir, "laptop_error.txt"), "w") as fh:
                    fh.write(traceback.format_exc())
                box["stopped"] = f"{type(e).__name__}: {e} (traceback in laptop_error.txt)"
        worker = threading.Thread(target=work, daemon=True)
        worker.start()
        while worker.is_alive() and interrupts[0] < 2:
            worker.join(0.5)
        got, stopped = box.get("got", got), box.get("stopped", "stopped by you (chunk abandoned)")
        panel.set_stage(exp, "fail" if stopped else "ok")
    except J.JobStop as e:
        stopped = str(e)
        for stg in stages[:-1]:
            if panel.stage.get(stg) == "run":
                panel.set_stage(stg, "fail")
        panel.note("stopped: " + stopped)
    except Exception as e:                     # noqa: BLE001  never skip the teardown
        stopped = f"{type(e).__name__}: {e}"
        panel.note("error: " + stopped + " (traceback in laptop_error.txt)")
        import traceback
        with open(os.path.join(out_dir, "laptop_error.txt"), "w") as fh:
            fh.write(traceback.format_exc())
    finally:
        try:                                   # a bug in the analysis must never skip the teardown below
            if P is not None:
                summary = P.summarize(got, ctx)
            elif a.experiment == "e0":
                summary = summarize(got["sampled"], got["greedy"], got["teacher"], vllm_recs)
            elif a.experiment == "e3b":
                sr = got.get("e3b steer") or []
                summary = {"vectors": (got.get("e3b vectors") or [None])[0],
                           "bridge": summarize_e3b_bridge(got.get("e3b bridge") or []),
                           "steer": summarize_e3b_steer(sr), "steer_aligned_only": summarize_e3b_steer(sr, aligned_only=True),
                           "steer_secs_per_item": st.mean(r["secs"] for r in sr) if sr else None}
            elif a.experiment == "e3":
                summary = {"vectors": (got.get("e3 vectors") or [None])[0],
                           "e3": summarize_e3(got.get("e3 steer") or []),
                           "e3_secs_per_item": st.mean(r["secs"] for r in got["e3 steer"])
                           if got.get("e3 steer") else None}
            else:
                e0_dir = default_e0_run(a.model)
                summary = {"e1": summarize_e1((got.get("e1 fit") or [None])[0], e0_dir),
                           "e2": summarize_e2(got.get("e2 patch") or []), "e2_layers": e2_layers,
                           "e2_clean_vs_e0": clean_vs_e0(got.get("e2 patch") or [], e0_dir),
                           "e2_secs_per_item": st.mean(r["secs"] for r in got["e2 patch"])
                           if got.get("e2 patch") else None}
        except Exception as e:                 # noqa: BLE001  raw results are already in results.jsonl
            summary = {"analysis_error": f"{type(e).__name__}: {e}"}
            panel.note("analysis error (raw results kept): " + summary["analysis_error"][:80])
        summary["stopped"] = stopped
        write_json(os.path.join(out_dir, "summary.json"), summary)
        if not os.path.exists(os.path.join(out_dir, "run_manifest.json")):
            write_json(os.path.join(out_dir, "run_manifest.json"),
                       {"schema": "interp_run/1", "run_id": run_id, "experiment": exp, "preview": a.preview,
                        "model": a.model, "revision": pin["revision"], "not_started": stopped})
        write_json(os.path.join(out_dir, "hashes.json"),
                   {f: sha256_file(os.path.join(out_dir, f)) for f in sorted(os.listdir(out_dir))
                    if f != "hashes.json"})
        if machine is not None:
            panel.set_now("deleting the machine")
            machine.finish()
        panel.set_now("done" if not stopped else "stopped")
        panel.stop.set()
        panel.draw()

    print()
    if "analysis_error" in summary:
        print("Analysis failed (raw results are in results.jsonl): " + summary["analysis_error"])
    elif P is not None:
        P.report(summary, a.preview)
    elif a.experiment == "e0":
        report(summary, a.preview)
    elif a.experiment == "e3":
        report_e3(summary, a.preview)
    elif a.experiment == "e3b":
        report_e3b(summary, a.preview)
    else:
        report_e1e2(summary, a.preview)
    print(f"\nData: {os.path.relpath(out_dir, ROOT)}" + (f"   (stopped: {stopped})" if stopped else ""))
    name = getattr(machine, "pod_name", None)
    if stopped and name:
        why = pod_reason(os.path.join(ROOT, "logs", f"pod_{name}", "interp.log"))
        if why:
            print("Pod's own account (logs/pod_" + name + "/interp.log):")
            for line in why:
                print("  " + line)
    if machine is None:
        print("This session belongs to the launcher you started; end it with:  touch logs/"
              + ("pod_session.stop" if a.slot == 0 else f"pod_session.{a.slot}.stop"))
        return 0 if not stopped else 1
    if machine.deleted is True:
        cost = f"${panel.cost_final:.2f}" if panel.cost_final is not None else "see logs/budget.tsv"
        print(f"Machine deleted; 0 of our pods left. Estimated cost {cost}.")
        return 0 if not stopped else 1
    print("!!! COULD NOT CONFIRM THE MACHINE IS GONE. Check the RunPod console now. !!!")
    return 9


def report(s, preview):
    f = lambda v: "     —" if v is None else f"{v:6.1f}"              # noqa: E731
    print("E0 (exploratory" + (", PREVIEW: canned numbers" if preview else "") + ")")
    print("  cell          pod mean  vLLM mean | pod slips  vLLM slips | pod no-slip  vLLM no-slip")
    for k in sorted(set(s["pod_sampled"]) | set(s["vllm"])):
        p, v = s["pod_sampled"].get(k, {}), s["vllm"].get(k, {})
        print(f"  {k:12} {f(p.get('mean_gap'))}%   {f(v.get('mean_gap'))}% | {p.get('slips', '—'):>9}  "
              f"{v.get('slips', '—'):>10} | {f(p.get('mean_gap_no_slips'))}%      {f(v.get('mean_gap_no_slips'))}%")
    print(f"  seller shift D8-D2: pod {f(s['shift_pod']['seller'])}  vLLM {f((s['shift_vllm'] or {}).get('seller'))}")
    print(f"  R2 vs greedy R1: r = {s['r1_greedy_vs_r2']} (n {s['r2_n']}); schema-constrained share "
          f"{s['constrained_share']}")
    if "fidelity" in s:
        fd = s["fidelity"]
        print(f"  fidelity: gaps within 3 pts {fd['gaps_within_3']} (x1 diff {fd['seller_D2_diff']:.1f}, "
              f"x1e6 diff {fd['seller_D8_diff']:.1f}); shift within 25% {fd['shift_within_25pct']}; "
              f"R2 usable {fd['r2_usable']}  ->  {'PASS' if fd['pass'] else 'FAIL'}")


def report_e1e2(s, preview):
    print("E1+E2 (exploratory" + (", PREVIEW: canned shapes" if preview else "") + ")")
    e1 = s.get("e1") or {}
    for pos, w in (e1.get("within") or {}).items():
        print(f"  E1 probe (negotiation, held-out fixtures) at {pos:<10}: best layer {w['best_layer']}, R² {w['best_r2']}")
    for key, w in (e1.get("transfer") or {}).items():
        print(f"  E1 probe trained on plain sentences, {key:<14}: best layer {w['best_layer']}, R² {w['best_r2']}")
    if e1.get("beyond_scale"):
        b = e1["beyond_scale"]
        f = lambda v: "—" if v is None else f"{v:+.2f}"       # noqa: E731
        print(f"  E1 beyond scale (layer {b['layer']}): readout residual vs E0 gap, limit size partialled out: "
              f"r {f(b['D2']['r_partial_limit'])} (x1, n {b['D2']['n']}), {f(b['D8']['r_partial_limit'])} (x1e6)")
    e2, layers = s.get("e2") or {}, s.get("e2_layers") or []
    if s.get("e2_clean_vs_e0"):
        c = s["e2_clean_vs_e0"]
        print(f"  E2 unpatched readings equal E0's R2 price in {c['same_price']} of {c['of']} (same prompts)")
    if e2.get("directions"):
        print(f"  E2: {e2['n_fixtures']} fixtures; clean rows consistent in {e2['clean_consistent']} of {e2['n_items']}")
        for dname, d in e2["directions"].items():
            print(f"  E2 {dname}: clean gap difference {d['mean_gap_difference_pts']:+.1f} points; share restored by layer")
            for pos in E2_POS:
                cells = d["cells"].get(pos, {})
                fr = [cells.get(str(L), {}).get("fraction") for L in layers]
                v, L = peak(layers, fr)
                c = cells.get(str(L), {}) if L is not None else {}
                ci = c.get("ci")
                off = max((cells.get(str(x), {}).get("slips", 0) for x in layers), default=0)
                extra = (f"peak L{L} {100 * v:.0f}%" + (f" [{100 * ci[0]:.0f}, {100 * ci[1]:.0f}]" if ci else "")
                         + f", source-scale units {100 * (c.get('units_share') or 0):.0f}%"
                         + (f", off-scale up to {off}/{c.get('n')}" if off else "")) if v is not None else "—"
                print(f"    {E2_POS_LABEL[pos]:<12} {spark(fr)}  {extra}")


def report_e3(s, preview):
    print("E3 steering (exploratory" + (", PREVIEW: canned shapes" if preview else "") + ")")
    if s.get("vectors"):
        for line in e3_live({"e3 vectors": [s["vectors"]]}):
            print("  " + line)
    e3 = s.get("e3") or {}
    for dname, vecs in e3.get("directions", {}).items():
        base = vecs.get("none", {}).get("0.0", {})
        print(f"  {dname}: unsteered gap {base.get('mean_gap') or 0:+.1f}% (n {base.get('n')}, valid {base.get('valid')})")
        for vec in E3_VECTORS:
            for a in E3_ALPHAS:
                c = vecs.get(vec, {}).get(str(a))
                if not c:
                    continue
                g = "—" if c["mean_gap"] is None else f"{c['mean_gap']:+.1f}%"
                ci = f" [{c['ci'][0]:+.1f}, {c['ci'][1]:+.1f}]" if c.get("ci") else ""
                sh = f"  moved {100 * c['share_moved']:+.0f}%" if c.get("share_moved") is not None else ""
                print(f"    {vec:<9} α {a:<4} gap {g}{ci}{sh}  valid {c['valid']}/{c['n']}"
                      + (f"  off-scale {c['off_scale'] + c['other_scale']}" if c["off_scale"] + c["other_scale"] else "")
                      + (f"  hit length {c['length']}" if c["length"] else ""))


def pod_reason(path, keep=("VLLM_REF", "LOADED", "DIAGNOSIS", "GREEDY", "INTERP_LOAD_FAILED", "INTERP_UNUSABLE",
                           "Error", "error:")):
    """The pod's key lines (screen-safe: no addresses), cut to the terminal width."""
    try:
        text = open(path, encoding="utf-8", errors="replace").read().replace("\r", "\n")
    except OSError:
        return []
    lines = [l.strip() for l in text.splitlines() if any(k in l for k in keep) and "LIBARCHIVE" not in l]
    lines = [re.sub(r"\b\d{1,3}(\.\d{1,3}){3}\b", "<ip>", l) for l in lines]
    return [l[:200] for l in lines[-8:]]


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def write_json(path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1)


def write_jsonl(path, rows):
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
