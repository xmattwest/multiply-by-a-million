"""interp_e5_laptop.py (LAPTOP-SIDE, standard library only): E5, the appraisal inside Gemma 4 31B. Study 3 found that one
public line, "An independent appraisal values the lot at {V}. Both sides know this.", removes Gemma's closing shift and
most of its opening modesty. E5 asks, with E4's tools (X61), where that line takes over inside the model. The plug-in
scripts/interp/exp_e5.py is a shim importing from here (keeps the pod's create request small). Exploratory.

    python3 scripts/launch_interp.py --preview --experiment e5                              (free: canned shapes)
    python3 scripts/launch_interp.py --model gemma-4-31b-it --experiment e5 --slot 8 --lifetime-min 100   (paid)

Predictions written before the run: config/interp/e5_predictions.md (must be committed for a paid launch; its sha256 is
in the run manifest; summarize() scores each one).

PROMPTS. Study 3's own message-1 prompts, rendered exactly as scripts/study3.py renders them (checked against the
recorded Study 3 requests in tests/test_interp_e5.py): N1 / N6 = A0 (Study 2's prompt) at $100 / $100M, P1 / P6 = A1.
A role's first-move prompt depends only on its own limit, so one fixture per DISTINCT limit among G01-G40 (sellers:
19 floors; buyers: 18 ceilings), so no prompt repeats and discovery / held-out fixtures never share a prompt (X62).

READOUT. As E4: ld = log p(b) - log p(a) at the deciding token of the forced opening, a and b = the A0 prompts' greedy
prices (N1's, N6's). Shares of the plain gap G_N; "appraisal swap" rows (E4's "cure") divide by the ld distance
between the two runs swapped.

GATE. Real sampled openings (4 per prompt) on all four prompts, against Study 3's vLLM first moves on the same prompts
(data/runs/s3-<model>-s1/s2, opener = the role, both sessions): each cell's mean gap within 3 points of vLLM's, each
condition's x1e6 - x1 shift within 3 points of vLLM's, >= 95% valid.

STEPS. Sellers (full, E4's order): gate, build, stats, attention, patch, steer P (A1 openings steered with the A0
direction at 42/44), steer N (the same on A0, discovery floors), appraisal swap by layer, swap heads (discovery),
swap circuit (held-out). Buyers (brief): gate, build, steer P, steer N, appraisal swap by layer.
"""
import glob
import hashlib
import json
import math
import os
import random
import statistics as st
import subprocess

import interp_e4_laptop as X4

PROMPTS, SCALE, OTHER = X4.PROMPTS, X4.SCALE, X4.OTHER
ROLES = ("seller", "buyer")
TAG = {"seller": "S", "buyer": "B"}
GATE = {"valid": 0.95, "cell": 0.03, "shift": 0.03}
PRED_FILE = os.path.join("config", "interp", "e5_predictions.md")
SCALE_KEY = {0: "x1e2", 6: "x1e8"}
COND = {"N": "A0", "P": "A1"}
# X57's (scale) and X61's (range cure) top-8 heads, for prediction E5-5.
PRIOR_HEADS = sorted(set(X4.TOP8) | {(39, 2), (41, 29), (41, 11), (41, 8), (41, 14), (41, 30), (40, 26), (43, 5)})
STEPS_S = ("gate", "build", "stats", "attention", "patch", "steer P", "steer N", "swap", "swap heads", "swap circuit")
STEPS_B = ("gate", "build", "steer P", "steer N", "swap")
_LI = []


def _li():
    if _LI:
        return _LI[0]
    import sys
    m = sys.modules.get("__main__")
    return m if hasattr(m, "boot_ci") else sys.modules.get("launch_interp")


def _study3():
    import study3 as S
    import render_prompt_s2 as rp2
    import render_prompt_s3 as rp3
    return S, rp2, rp3


# --- flags ---------------------------------------------------------------------------------------------------------
def add_args(ap):
    ap.add_argument("--e5-discovery", type=int, default=7, help="e5: discovery limits for the swap head sweep (default 7)")
    ap.add_argument("--e5-gate-samples", type=int, default=4, help="e5: sampled openings per prompt in the gate (default 4)")
    ap.add_argument("--e5-roles", default="seller,buyer", help="e5: roles to run (default seller,buyer; buyers are brief)")


def configure(a):
    a.roles = "seller,buyer"


# --- prompts and items -----------------------------------------------------------------------------------------------
def distinct_fixtures(role):
    """[(fixture id, limit pct, [every G fixture with that limit])] in G order, one per distinct limit."""
    S, _, _ = _study3()
    key = "f_pct" if role == "seller" else "c_pct"
    seen = {}
    for f in S.fixtures():
        seen.setdefault(f[key], []).append(f["id"])
    return [(ids[0], pct, ids) for pct, ids in sorted(seen.items(), key=lambda x: x[1][0])]


def render(role, V, limit, a1):
    S, rp2, rp3 = _study3()
    return (rp3 if a1 else rp2).render(role, V, limit, [], 1, S.F_RANGE, S.C_RANGE, S.MAX_MSGS, S.MAX_CHARS)


def build_items(role):
    S, _, _ = _study3()
    out = []
    for fid, pct, group in distinct_fixtures(role):
        pr = {}
        for k in PROMPTS:
            m, a1 = SCALE[k], k[0] == "P"
            V, limit = S.SCALES[SCALE_KEY[m]], pct * 10 ** m
            pr[k] = {"condition": f"{COND[k[0]]}-{SCALE_KEY[m]}", "limit_units": limit, "limit_text": str(limit),
                     "messages": render(role, V, limit, a1), "appraisal": str(V) if a1 else None}
        out.append({"item_id": f"{fid}-{TAG[role]}-e5", "fixture": fid, "role": role, "base_limit": pct, "group": group,
                    "seed": S.call_seed(fid, role, 1, 1, 1), "prompts": pr})
    return out


def steer_items(fx, targets, tag, label):
    return [dict(it, item_id=f"{it['fixture']}-{TAG[it['role']]}-{t}-{tag}", target=t, sign=1.0 if SCALE[t] == 0 else -1.0,
                 e5_step=label) for it in fx for t in targets]


def tagged(items, label):
    return [dict(it, e5_step=label) for it in items]


# --- Study 3's vLLM openings (the gate's reference and the shares' denominator) ---------------------------------------
def vllm_reference(model, role, root=None):
    """{(cond, m, pct): [gap]} from Study 3's core games of this model, opener = role, both sessions."""
    import analyze_addendum_c as AC
    root = root or _li().ROOT
    out = {}
    runs = sorted(glob.glob(os.path.join(root, "data", "runs", f"s3-{model}-s[12]-*")))
    for d in runs:
        for line in open(os.path.join(d, "games.jsonl"), encoding="utf-8"):
            g = json.loads(line)
            if g["condition"] not in ("A0", "A1") or g["scale"] not in ("x1e2", "x1e8") or g["opener"] != role:
                continue
            v = AC.opening_gap(g)
            if v is None:
                continue
            pct = g["f_pct"] if role == "seller" else g["c_pct"]
            out.setdefault((g["condition"], 0 if g["scale"] == "x1e2" else 6, pct), []).append(float(v))
    return out, [os.path.relpath(d, root) for d in runs]


def ref_plain(vref, fx):
    """{fixture: {0: A0 gap at x1, 6: at x1e6}} (vLLM means for the prompt), the steering shares' denominator."""
    out = {}
    for it in fx:
        a, b = vref.get(("A0", 0, it["base_limit"])), vref.get(("A0", 6, it["base_limit"]))
        if a and b:
            out[it["fixture"]] = {0: st.mean(a), 6: st.mean(b)}
    return out


# --- row specs and timing --------------------------------------------------------------------------------------------
def kws(role, Ls, a, schema, top, rand):
    full = role == "seller"
    plan = [list(x) for x in X4.B_PLAN]
    return {"gate": {"plan": [], "n_none": a.e5_gate_samples, "batch": a.e5_gate_samples, "schema": schema, "role": role},
            "build": {"keep": Ls["keep"], "head_layers": Ls["heads"], "role": role},
            "stats": {"role": role}, "attention": {"role": role},
            "patch": {"rows": X4.patch_rows(Ls, top, rand) if full else {}, "batch": a.batch, "role": role},
            "steer P": {"plan": plan, "n_none": 6, "batch": 16, "schema": schema, "role": role},
            "steer N": {"plan": plan, "n_none": 6, "batch": 16, "schema": schema, "role": role},
            "swap": {"rows": X4.cure_rows(Ls), "batch": a.batch, "role": role},
            "swap heads": {"rows": {}, "batch": a.batch, "role": role},
            "swap circuit": {"rows": {}, "batch": a.batch, "role": role}}


def estimate_min(n, batch, gate_samples):
    """{role: n_fixtures} -> (typical, slow, per-step seconds by label)."""
    Ls = X4.layer_sets(X4.CANON)
    P = X4._nrows(X4.patch_rows(Ls, X4.TOP8, X4.RAND8))
    C = X4._nrows(X4.cure_rows(Ls))
    H = X4._nrows(X4.cure_head_rows(list(range(38, 45)), 32))
    K = X4._nrows(X4.circuit_rows(list(X4.TOP8), list(X4.RAND8), list(range(38, 45))))
    b = lambda R: sum(X4._batches(v, batch) for v in R.values()) * X4.BATCH_S    # noqa: E731
    per_item = {"gate": X4.GEN4_S * (1 + gate_samples / 16), "build": 1.6 * X4.BUILD_S, "stats": 5.0,
                "attention": X4.ATTN_S, "patch": b(P) + 4 * X4.SRC_S, "steer P": X4.GEN16_S, "steer N": X4.GEN16_S,
                "swap": b(C) + 4 * X4.SRC_S, "swap heads": b(H) + 4 * X4.SRC_S, "swap circuit": b(K) + 4 * X4.SRC_S}
    per, t = {}, 40 * 0.1
    for role, nf in n.items():
        nd = min(7, max(1, nf // 2))
        cnt = {"gate": 4 * nf, "build": nf, "stats": 1, "attention": nf, "patch": nf, "steer P": 2 * nf, "steer N": 2 * nd,
               "swap": nf, "swap heads": nd, "swap circuit": nf - nd}
        for s in (STEPS_S if role == "seller" else STEPS_B):
            per[f"e5 {role} {s}"] = per_item[s]
            t += per_item[s] * cnt[s]
    return t / 60, t * X4.SLOW / 60, per


# --- setup -----------------------------------------------------------------------------------------------------------
def _sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def predictions_committed(root):
    """None if config/interp/e5_predictions.md is committed and unchanged, else the reason."""
    p = os.path.join(root, PRED_FILE)
    if not os.path.exists(p):
        return f"{PRED_FILE} does not exist"
    r = subprocess.run(["git", "-C", root, "ls-files", "--error-unmatch", PRED_FILE], capture_output=True, text=True)
    if r.returncode != 0:
        return f"{PRED_FILE} is not committed"
    r = subprocess.run(["git", "-C", root, "status", "--porcelain", "--", PRED_FILE], capture_output=True, text=True)
    return f"{PRED_FILE} has uncommitted changes" if r.stdout.strip() else None


def setup(a, settings, schema, li):
    _LI[:] = [li]
    S, _, _ = _study3()
    if not a.preview and not a.attach:
        why = predictions_committed(li.ROOT)
        if why:
            raise SystemExit(f"Not launching E5: {why}. Commit the predictions before any paid run.")
    roles = [r for r in ROLES if r in a.e5_roles.split(",")]
    schema = S.response_schema()                      # Study 3's schema (no minimum / maxLength), as the vLLM runs
    Ls = X4.layer_sets(X4.CANON)
    nonmoney = [v for v in li.build_e3_items(settings, 1)[0] if v["kind"] == "nonmoney"]
    R, vref, vruns = {}, {}, {}
    for role in roles:
        fx = build_items(role)
        fix = [it["fixture"] for it in fx]
        nd = min(a.e5_discovery, max(1, len(fix) // 2))       # 19 floors -> 7 discovery, 12 held-out (predictions file)
        vref[role], vruns[role] = vllm_reference(a.model, role, li.ROOT)
        R[role] = {"fx": fx, "disc": fix[:nd], "held": fix[nd:], "kw": kws(role, Ls, a, schema, list(X4.TOP8), list(X4.RAND8)),
                   "window": None, "pick": None, "top": list(X4.TOP8), "rand": list(X4.RAND8)}
    typ, slow, per = estimate_min({r: len(R[r]["fx"]) for r in roles}, a.batch, a.e5_gate_samples)
    li.STEP_GUESS_S.update(per)
    ctx = {"li": li, "R": R, "e5_roles": roles, "nonmoney": nonmoney, "layers": Ls, "n_heads": 32, "model": a.model,
           "vref": vref, "vruns": vruns, "estimate_min": [round(typ, 1), round(slow, 1)],
           "pred_sha256": _sha256(os.path.join(li.ROOT, PRED_FILE)) if os.path.exists(os.path.join(li.ROOT, PRED_FILE)) else None}
    n_vllm = {r: sum(len(v) for v in vref[r].values()) for r in roles}
    plan = (f"E5 (the appraisal) on {a.model}: Study 3's message-1 prompts A0/A1 at $100/$100M, one fixture per distinct "
            f"limit ({', '.join(r + 's ' + str(len(R[r]['fx'])) for r in roles)}). Gate against Study 3's vLLM openings "
            f"({', '.join(f'{r} {n_vllm[r]}' for r in roles)} games); sellers: build, stats, attention, patch, steering "
            f"(A1 and A0, layers 42/44), appraisal swap by layer, swap heads ({len(R['seller']['disc']) if 'seller' in R else 0} "
            f"discovery limits) and circuit; buyers: build, steering, swap by layer. About {typ:.0f} min of experiment time "
            f"({slow:.0f} if forwards run 1.5x slower; --budget-min {a.budget_min:g} stops new items)")
    inputs = [{"step": "fixture", **it} for r in roles for it in R[r]["fx"]] + [{"step": "nonmoney", **v} for v in nonmoney]
    return {"exp": "E5", "plan": plan, "inputs": inputs, "ctx": ctx, "steps": steps, "live": live, "after_step": after_step}


def steps(ctx):
    out = []
    for role in ctx["e5_roles"]:
        R, kw = ctx["R"][role], ctx["R"][role]["kw"]
        fx = R["fx"]
        dfx = [it for it in fx if it["fixture"] in R["disc"]]
        hfx = [it for it in fx if it["fixture"] in R["held"]]
        lab = lambda s: f"e5 {role} {s}"                                     # noqa: E731
        spec = {"gate": ("e5_steer", steer_items(fx, PROMPTS, "gate", lab("gate")), 4),
                "build": ("e5_build", tagged(ctx["nonmoney"] + fx, lab("build")), 5),
                "stats": ("e5_stats", [{"item_id": f"stats-{role}", "e5_step": lab("stats")}], 1),
                "attention": ("e5_attn", tagged(fx, lab("attention")), 5),
                "patch": ("e5_patch", tagged(fx, lab("patch")), 2),
                "steer P": ("e5_steer", steer_items(fx, ("P1", "P6"), "steer", lab("steer P")), 2),
                "steer N": ("e5_steer", steer_items(dfx, ("N1", "N6"), "steer", lab("steer N")), 2),
                "swap": ("e5_patch", tagged(fx, lab("swap")), 3),
                "swap heads": ("e5_patch", tagged(dfx, lab("swap heads")), 1),
                "swap circuit": ("e5_patch", tagged(hfx, lab("swap circuit")), 5)}
        for s in (STEPS_S if role == "seller" else STEPS_B):
            task, items, chunk = spec[s]
            out.append((lab(s), task, items, kw[s], chunk))
    return out


def _relayer(ctx, n, nH):
    Ls = X4.layer_sets(n)
    ctx["layers"], ctx["n_heads"] = Ls, nH
    for role, R in ctx["R"].items():
        R["top"], R["rand"] = X4.map_heads(X4.TOP8, n, nH), X4.map_heads(X4.RAND8, n, nH)
        kw = R["kw"]
        kw["build"].update(keep=Ls["keep"], head_layers=Ls["heads"])
        kw["patch"]["rows"] = X4.patch_rows(Ls, R["top"], R["rand"]) if role == "seller" else {}
        kw["swap"]["rows"] = X4.cure_rows(Ls)
        for s in ("steer P", "steer N"):
            kw[s]["plan"][:] = [[v, X4.map_layer(L, n), al, k] for v, L, al, k in X4.B_PLAN]


def on_pod_info(ctx, pod_info):
    n = pod_info.get("num_hidden_layers")
    if n and int(n) != ctx["layers"]["n_layers"]:
        _relayer(ctx, int(n), ctx["n_heads"])


def _split(label):
    _e5, role, step = label.split(" ", 2)
    return role, step


def after_step(label, got, ctx):
    li = ctx.get("li") or _li()
    note = (lambda s: ctx["panel"].note(s)) if ctx.get("panel") is not None else (lambda s: None)
    role, step = _split(label)
    R = ctx["R"][role]
    if step == "gate":
        G = summarize_gate(got.get(label) or [], li, role, ctx["vref"][role], R["fx"])
        if G.get("rehearsal"):
            note(f"{role} gate: REHEARSAL (random weights), not applied: " + (G.get("why") or "pass"))
            return None
        note(f"{role} gate: " + ("PASS" if G["pass"] else "FAIL") + f" ({G['line']})")
        return None if G["pass"] else f"{role} gate failed: " + G["why"]
    if step == "build":
        recs = [r for r in got.get(label) or [] if r.get("kind") == "fixture"]
        use = [r for r in recs if not r.get("skipped")]
        if len(use) < min(10, len(recs)):
            return f"{role}: only {len(use)} of {len(recs)} fixtures have two distinct A0 greedy prices"
        if len({(r["n_H"], r["n_S"], r["n_reply"]) for r in use}) != 1:
            return f"{role}: the aligned segments differ in length between fixtures"
        n, nH = use[0]["n_layers"], use[0]["n_heads"]
        if n != ctx["layers"]["n_layers"] or nH != ctx["n_heads"]:
            _relayer(ctx, n, nH)
        note(f"{role} build: {len(use)}/{len(recs)} usable; aligned {use[0]['n_A']} = {use[0]['n_H']} + {use[0]['n_S']} + "
             f"{use[0]['n_reply']} tokens; appraisal line {use[0]['n_R']['P1']}/{use[0]['n_R']['P6']} tokens")
        return None
    if step == "swap" and role == "seller":
        C = X4.cure_summary(got.get(label) or [])
        band, info = X4.pick_window(C, ctx["layers"]["n_layers"])
        if band is None:
            return None
        R["window"] = dict(info, band=band)
        R["kw"]["swap heads"]["rows"].update(X4.cure_head_rows(band, ctx["n_heads"]))
        note(f"swap heads: window L{band[0]}-L{band[-1]} ({info['rule']})")
        return None
    if step == "swap heads":
        S = X4.ratio_table(got.get(label) or [], "cure", ci=False)
        top, rand, info = X4.pick_cure_heads(S, R["window"]["band"] if R.get("window") else [], ctx["n_heads"],
                                             seed="scale-invariance/interp/e5/random/v1")
        if top:
            R["pick"] = dict(info, top=[list(x) for x in top], random=[list(x) for x in rand])
            R["kw"]["swap circuit"]["rows"].update(X4.circuit_rows(top, rand, R["window"]["band"]))
            note("swap heads: " + ", ".join(f"L{L}.{h}" for L, h in top))
    return None


# --- analysis --------------------------------------------------------------------------------------------------------
def _openings(recs, li, role):
    """{(target, fixture): {row key: [gap]}} for valid OFFERs on the target's own scale; counts per (target, key)."""
    per, cnt = {}, {}
    for r in recs:
        own = SCALE[r["target"]]
        for x in r["rows"]:
            key = "none" if x["vector"] == "none" else f"{x['vector']}@{x['layer']}x{x['alpha']:g}"
            c = cnt.setdefault((r["target"], key), {"n": 0, "valid": 0, "off_scale": 0, "other_scale": 0, "aligned": 0, "prices": {}})
            c["n"] += 1
            c["aligned"] += bool(x.get("aligned"))
            if x.get("errors"):
                continue
            c["valid"] += 1
            if x.get("action") != "OFFER" or not x.get("price"):
                continue
            g, m, off = li.scale_free(role, x["price"], r["base_limit"])
            if off:
                c["off_scale"] += 1
                continue
            if m != own:
                c["other_scale"] += 1
                continue
            p = X4.units(x["price"], own)
            c["prices"][p] = c["prices"].get(p, 0) + 1
            per.setdefault((r["target"], r["fixture"]), {}).setdefault(key, []).append(g)
    return per, cnt


def summarize_gate(recs, li, role, vref, fx):
    per, cnt = _openings(recs, li, role)
    lim = {it["fixture"]: it["base_limit"] for it in fx}
    out = {"rehearsal": any(r.get("rehearsal") for r in recs), "cells": {}, "role": role}
    for t in PROMPTS:
        c = cnt.get((t, "none"), {"n": 0, "valid": 0, "off_scale": 0, "other_scale": 0, "prices": {}})
        fm = {f: st.mean(v["none"]) for (tt, f), v in per.items() if tt == t and v.get("none")}
        vm = {f: st.mean(vref[(COND[t[0]], SCALE[t], lim[f])]) for f in fm if vref.get((COND[t[0]], SCALE[t], lim[f]))}
        both = sorted(set(fm) & set(vm))
        out["cells"][t] = {"n": c["n"], "valid": c["valid"], "valid_share": c["valid"] / c["n"] if c["n"] else None,
                           "off_scale": c["off_scale"], "other_scale": c["other_scale"], "n_prompts": len(both),
                           "mean_gap": st.mean(fm[f] for f in both) if both else None,
                           "vllm_gap": st.mean(vm[f] for f in both) if both else None,
                           "r": _pearson([fm[f] for f in both], [vm[f] for f in both])}
    why = []
    for t, c in out["cells"].items():
        if c["valid_share"] is None or c["valid_share"] < GATE["valid"]:
            why.append(f"{t} valid {c['valid']}/{c['n']}")
        if c["mean_gap"] is None or c["vllm_gap"] is None:
            why.append(f"{t} no gap")
        elif abs(c["mean_gap"] - c["vllm_gap"]) > GATE["cell"]:
            why.append(f"{t} {100 * c['mean_gap']:.1f}% vs vLLM {100 * c['vllm_gap']:.1f}%")
    out["shift"] = {}
    for k, (a, b) in (("A0", ("N1", "N6")), ("A1", ("P1", "P6"))):
        A, B = out["cells"][a], out["cells"][b]
        if None in (A["mean_gap"], B["mean_gap"], A["vllm_gap"], B["vllm_gap"]):
            out["shift"][k] = None
            why.append(f"{k} shift n/a")
            continue
        s, sv = B["mean_gap"] - A["mean_gap"], B["vllm_gap"] - A["vllm_gap"]
        out["shift"][k] = {"pod": s, "vllm": sv}
        if abs(s - sv) > GATE["shift"]:
            why.append(f"{k} shift {100 * s:+.1f} vs vLLM {100 * sv:+.1f} points")
    out["pass"], out["why"] = not why, "; ".join(why)
    f = lambda v: "n/a" if v is None else f"{100 * v:.1f}%"                        # noqa: E731
    out["line"] = " · ".join(f"{t} {f(c['mean_gap'])} (vLLM {f(c['vllm_gap'])})" for t, c in out["cells"].items()) + " · shift " + \
        ", ".join(f"{k} {100 * v['pod']:+.1f} (vLLM {100 * v['vllm']:+.1f})" for k, v in out["shift"].items() if v)
    return out


def _pearson(x, y):
    if len(x) < 3:
        return None
    mx, my = st.mean(x), st.mean(y)
    sx, sy = math.sqrt(sum((a - mx) ** 2 for a in x)), math.sqrt(sum((b - my) ** 2 for b in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy) if sx and sy else None


def summarize_steer(recs, li, role, ref):
    """As E4's: per target and row, counts, mean gap, points moved vs the item's unsteered rows, and the share of the A0
    gap (ref: per-fixture vLLM A0 gaps); per-fixture moves kept for the A1-vs-A0 comparison (E5-4)."""
    per, cnt = _openings(recs, li, role)
    out = {}
    for (t, key), c in cnt.items():
        cell = {k: c[k] for k in ("n", "valid", "off_scale", "other_scale", "aligned")}
        g = [x for (tt, _f), v in per.items() if tt == t for x in v.get(key, [])]
        cell["mean_gap"] = st.mean(g) if g else None
        if key != "none":
            num, den, fxs = [], [], []
            for (tt, f), v in per.items():
                if tt != t or key not in v or "none" not in v:
                    continue
                r_ = ref.get(f)
                if not r_:
                    continue
                num.append(st.mean(v[key]) - st.mean(v["none"]))
                den.append(r_[6 - SCALE[t]] - r_[SCALE[t]])
                fxs.append(f)
            cell["moved_pts"] = 100 * st.mean(num) if num else None
            cell["n_fixtures"] = len(num)
            cell["per_fixture"] = {f: [a, b] for f, a, b in zip(fxs, num, den)}
            if num and st.mean(den):
                cell["share"] = st.mean(num) / st.mean(den)
                cell["ci"] = X4.ratio_ci(num, den)
        out.setdefault(t, {})[key] = cell
    return out


def summarize(got, ctx):
    li = ctx.get("li") or _li()
    out = {"layers": ctx["layers"], "n_heads": ctx["n_heads"], "estimate_min": ctx.get("estimate_min"),
           "vllm_runs": ctx.get("vruns"), "predictions_sha256": ctx.get("pred_sha256"), "roles": {}}
    for role in ctx["e5_roles"]:
        R = ctx["R"][role]
        g = lambda s: got.get(f"e5 {role} {s}") or []                     # noqa: E731
        ref = ref_plain(ctx["vref"][role], R["fx"])
        out["roles"][role] = {
            "discovery": R["disc"], "held_out": R["held"], "top": R["top"], "rand": R["rand"],
            "gate": summarize_gate(g("gate"), li, role, ctx["vref"][role], R["fx"]),
            "build": X4.build_summary(g("build")), "stats": (g("stats") or [None])[0],
            "attention": X4.attention_summary(g("attention")) if g("attention") else None,
            "patch": X4.ratio_table(g("patch"), "plain"), "patch_clean": X4.clean_table(g("patch")),
            "steer_P": summarize_steer(g("steer P"), li, role, ref), "steer_N": summarize_steer(g("steer N"), li, role, ref),
            "swap": X4.ratio_table(g("swap"), "cure"), "swap_clean": X4.clean_table(g("swap")), "window": R.get("window"),
            "swap_heads": X4.ratio_table(g("swap heads"), "cure", ci=False), "pick": R.get("pick"),
            "circuit": X4.ratio_table(g("swap circuit"), "cure")}
    pr = [r for lab, recs in got.items() if lab.endswith(("patch", "swap", "swap heads", "swap circuit")) for r in recs]
    out["checks"] = {"hooks_ok": all(T.get("hooks_ok", True) for r in pr for T in (r.get("targets") or {}).values()),
                     "clean_spread_max": max((T.get("clean_spread") or 0 for r in pr for T in (r.get("targets") or {}).values()), default=None),
                     "steer_aligned": {lab: [sum(1 for r in recs for x in r["rows"] if x["vector"] != "none" and x.get("aligned")),
                                             sum(1 for r in recs for x in r["rows"] if x["vector"] != "none")]
                                       for lab, recs in got.items() if "steer" in lab},
                     "loo": sorted({str(r.get("loo")) for lab, recs in got.items() if lab.endswith(("patch", "steer P", "steer N"))
                                    for r in recs})}
    out["secs_per_item"] = {lab: st.mean(r["secs"] for r in got[lab] if "secs" in r) if got.get(lab) and
                            any("secs" in r for r in got[lab]) else None for lab, *_ in steps(ctx)}
    out["predictions"] = evaluate(out)
    return out


# --- the predictions (config/interp/e5_predictions.md) ---------------------------------------------------------------
def _swap_at(S, t, L):
    c = (S.get(t) or {}).get(f"cure@{L}")
    return None if c is None else c.get("share")


def _onset(sw, n_layers):
    """(max share at measured layers <= 39, max of layers 41 and 42) for P6 into N6 (rows on N6), 31B layer numbers."""
    rows = {int(n.split("@")[1]): c["share"] for n, c in (sw.get("N6") or {}).items()
            if n.startswith("cure@") and c.get("share") is not None}
    early = [v for L, v in rows.items() if L <= X4.map_layer(39, n_layers)]
    late = [rows[L] for L in {X4.map_layer(41, n_layers), X4.map_layer(42, n_layers)} if L in rows]
    return (max(early) if early else None), (max(late) if late else None)


def evaluate(s):
    """[{id, claim, result: HIT / MISS / n.a., value}] for the six predictions."""
    n = (s.get("layers") or {}).get("n_layers", X4.CANON)
    out = []

    def add(pid, claim, cond, value):
        out.append({"id": pid, "claim": claim, "result": "n.a." if cond is None else ("HIT" if cond else "MISS"), "value": value})
    se = (s.get("roles") or {}).get("seller") or {}
    bu = (s.get("roles") or {}).get("buyer") or {}
    B = se.get("build") or {}
    ratio = B["G_P"] / B["G_N"] if B.get("G_N") else None
    add("E5-1", "sellers: under A1 the forced opening keeps 5-50% of the A0 log-odds gap (G_P / G_N)",
        None if ratio is None else 0.05 <= ratio <= 0.50, None if ratio is None else round(ratio, 3))
    e, l = _onset(se.get("swap") or {}, n)
    add("E5-2", "sellers: adding the appraisal (A1 x1e6 states on S + reply into A0 x1e6) moves <= 15% at every layer <= 39 "
        "and >= 50% at layer 41 or 42", None if e is None or l is None else (e <= 0.15 and l >= 0.50),
        None if e is None else {"max_le_39": round(e, 3), "max_41_42": None if l is None else round(l, 3)})
    sw = (se.get("swap") or {}).get("N6") or {}
    L44 = X4.map_layer(44, n)
    rep, sg = (sw.get(f"cure reply@{L44}") or {}).get("share"), (sw.get(f"cure S@{L44}") or {}).get("share")
    add("E5-3", "sellers: at layer 44 the swap on the reply's opening tokens alone moves more than on the prompt's S tokens alone",
        None if rep is None or sg is None else rep > sg, None if rep is None else {"reply": round(rep, 3), "S": None if sg is None else round(sg, 3)})
    key = f"vN@{X4.map_layer(42, n)}x1"
    res, val = [], {}
    for p, q in (("P1", "N1"), ("P6", "N6")):
        a = ((se.get("steer_P") or {}).get(p) or {}).get(key) or {}
        b = ((se.get("steer_N") or {}).get(q) or {}).get(key) or {}
        fa, fb = a.get("per_fixture") or {}, b.get("per_fixture") or {}
        both = sorted(set(fa) & set(fb))
        if not both or not st.mean(fb[f][1] for f in both):
            res.append(None)
            continue
        sa = st.mean(fa[f][0] for f in both) / st.mean(fa[f][1] for f in both)
        sb = st.mean(fb[f][0] for f in both) / st.mean(fb[f][1] for f in both)
        val[p] = {"A1": round(sa, 3), "A0": round(sb, 3), "n": len(both)}
        res.append(sa < 0.5 * sb)
    add("E5-4", "sellers: the A0 direction at layer 42 (alpha 1) moves real A1 openings less than half as much as A0 openings, "
        "both directions (discovery floors)", None if None in res else all(res), val or None)
    pk, K = se.get("pick") or {}, (se.get("circuit") or {}).get("N6") or {}
    top = [tuple(x) for x in pk.get("top") or []]
    shared = sorted(set(top) & set(PRIOR_HEADS)) if top else None
    joint = (K.get(f"top{len(top)}") or {}).get("share") if top else None
    add("E5-5", "sellers: >= 2 of the appraisal's top 8 heads are among X57's / X61's top 8, and the 8 carry >= 50% of the "
        "swap on held-out floors (P6 into N6)", None if shared is None or joint is None else (len(shared) >= 2 and joint >= 0.5),
        None if shared is None else {"shared": [list(x) for x in shared], "held_out_share": None if joint is None else round(joint, 3)})
    e, l = _onset(bu.get("swap") or {}, n)
    add("E5-6", "buyers: the same onset as E5-2 (<= 15% at every layer <= 39, >= 50% at 41 or 42)",
        None if e is None or l is None else (e <= 0.15 and l >= 0.50),
        None if e is None else {"max_le_39": round(e, 3), "max_41_42": None if l is None else round(l, 3)})
    return out


def manifest(ctx):
    return {"name": "e5", "plan": "config/interp/e5_predictions.md (E5, the appraisal)", "layers": ctx["layers"],
            "predictions_file": PRED_FILE, "predictions_sha256": ctx.get("pred_sha256"), "gate": GATE,
            "vllm_runs": ctx.get("vruns"), "prior_heads": [list(x) for x in PRIOR_HEADS],
            "roles": {r: {"fixtures": [it["fixture"] for it in R["fx"]], "limits": [it["base_limit"] for it in R["fx"]],
                          "discovery": R["disc"], "held_out": R["held"]} for r, R in ctx["R"].items()},
            "estimate_min": ctx.get("estimate_min")}


# --- panel, preview, report ------------------------------------------------------------------------------------------
_p, _ci = X4._p, X4._ci


def live(got):
    li, out = _li(), []
    for lab, recs in got.items():
        if not recs:
            continue
        role, step = _split(lab)
        if step == "build":
            B = X4.build_summary(recs)
            if B.get("ld"):
                out.append(f"{role} build: {B['usable']}/{B['n']} usable; G_N {B['G_N']:+.1f} nats, G_P {B['G_P']:+.1f}")
        if step == "swap":
            C = X4.cure_summary(recs)
            out.append(f"{role} swap (A0 x1e6 <- A1 x1e6) by layer: " + " ".join(
                f"{n[5:]}:{_p(v).strip()}" for n, v in sorted(C.get("N6", {}).items(), key=lambda x: int(x[0][5:]))[::4]))
    return out[-4:]


def stub(task, items, ctx):
    """Preview only: canned shapes (gate openings drawn from Study 3's vLLM means), not a model."""
    if not items:
        return []
    label = items[0]["e5_step"]
    role, step = _split(label)
    R, kw = ctx["R"][role], ctx["R"][role]["kw"][step]
    Ls, n, nH = ctx["layers"], ctx["layers"]["n_layers"], ctx["n_heads"]
    vref = ctx["vref"][role]
    sig = lambda x: 1 / (1 + math.exp(-x))                                  # noqa: E731
    sgn = 1 if role == "seller" else -1

    def gap0(t, pct):
        v = vref.get((COND[t[0]], SCALE[t], pct))
        return st.mean(v) if v else (0.4 if SCALE[t] == 0 else 0.25)
    if task == "e5_steer":
        out = []
        for it in items:
            t, s = it["target"], SCALE[it["target"]]
            rng = random.Random(it["item_id"])
            rows = [{"vector": "none", "layer": None, "alpha": 0.0, "sample": i} for i in range(kw["n_none"])]
            rows += [{"vector": v, "layer": L, "alpha": a, "sample": j} for v, L, a, k in kw["plan"] for j in range(k)]
            plain = gap0("N" + ("6" if s == 0 else "1"), it["base_limit"]) - gap0("N" + str(s or 1), it["base_limit"])
            for x in rows:
                g = gap0(t, it["base_limit"]) + rng.uniform(-0.01, 0.01)
                if x["vector"] == "vN":
                    g += (0.15 if t[0] == "P" else 0.85) * x["alpha"] * plain
                p = it["base_limit"] * (1 + sgn * g)
                x.update(price=int(round(p)) * 10 ** s, errors=[], action="OFFER", aligned=True)
            out.append({"item_id": it["item_id"], "fixture": it["fixture"], "role": role, "target": t, "sign": it["sign"],
                        "base_limit": it["base_limit"], "rehearsal": False, "rows": rows, "secs": 0.1})
        return out
    if task in ("e5_build", "e5_stats", "e5_attn"):
        base = X4.stub({"e5_build": "e4_build", "e5_stats": "e4_stats", "e5_attn": "e4_attn"}[task],
                       [dict(x) for x in items], {"layers": Ls, "n_heads": nH, "fx": R["fx"], "nonmoney": ctx["nonmoney"]})
        for r in base:
            if task == "e5_build" and r.get("kind") == "fixture":
                b = r["base_limit"]
                r["price"] = {"N1": round(b * (1 + sgn * 0.42)), "N6": round(b * (1 + sgn * 0.22)) * 10 ** 6,
                              "P1": round(b * (1 + sgn * 0.35)), "P6": round(b * (1 + sgn * 0.33)) * 10 ** 6}
                r["n_R"] = {"N1": 0, "N6": 0, "P1": 16, "P6": 24}
            if task == "e5_stats":
                r["role"] = role
        return base
    ld0 = {"N1": -12.0, "N6": 5.0, "P1": -13.0, "P6": -11.0}

    def share(t, name):
        L = int(name.split("@")[1]) if "@" in name else 42
        if name.startswith("cure"):
            return sig((L - 40.5) / 0.8) * (0.4 if name.startswith("cure S") else 0.9 if name.startswith("cure reply") else 1.0)
        if name.startswith("h"):
            return 0.2 if (name.endswith(".29") or name.endswith(".11")) and L == 41 else 0.004
        if name.startswith(("top", "window", "random")):
            return {"top1": 0.15, "top4": 0.4, "top8": 0.6}.get(name, 0.03)
        k = 0.2 if t[0] == "P" else 1.0
        return k * (sig((L - 40.5) / 1.2) if name.startswith(("own", "plain", "vN")) else 0.05)
    out = []
    for it in items:
        targets = {}
        for t, specs in kw["rows"].items():
            other = ld0[t] + (17.0 if SCALE[t] == 0 else -17.0) if step == "patch" else ld0["P6" if t == "N6" else "N6"]
            rows = [{"n": x["n"], "ld": round(ld0[t] + share(t, x["n"]) * (other - ld0[t]), 4), "lpa": -0.3, "lpb": -2.0,
                     "p1": [0.0] * 10} for x in specs]
            targets[t] = {"ld_clean": ld0[t], "clean_spread": 0.0, "hooks_ok": True,
                          "rows": rows + [{"n": "clean", "ld": ld0[t], "lpa": -0.3, "lpb": -2.0, "p1": [0.0] * 10}]}
        out.append({"item_id": it["item_id"], "fixture": it["fixture"], "base_limit": it["base_limit"], "role": role,
                    "ld_build": ld0, "targets": targets, "secs": 0.1})
    return out


def _row_line(li, rows):
    return f"{X4._spark(li, [v for _L, v in rows])}  " + " ".join(f"L{L} {_p(v).strip()}" for L, v in rows if L in
                                                                 (30, 36, 38, 39, 40, 41, 42, 44, 48, 52, 58))


def report(s, preview):
    li = _li()
    print("E5, the appraisal inside Gemma (exploratory; predictions in " + PRED_FILE
          + (", PREVIEW: canned shapes" if preview else "") + ")")
    Ls = s.get("layers") or {}
    if Ls.get("mapped"):
        print(f"  NOTE: a {Ls['n_layers']}-layer model; the 31B's layer sets were mapped onto it")
    for role, r in (s.get("roles") or {}).items():
        print(f"\n  --- {role}s ({len(r['discovery']) + len(r['held_out'])} distinct limits; discovery {len(r['discovery'])}) ---")
        G = r.get("gate") or {}
        if G.get("cells"):
            print(f"  gate vs Study 3's vLLM openings: {'PASS' if G['pass'] else 'FAIL: ' + G['why']}"
                  + (" [REHEARSAL]" if G.get("rehearsal") else ""))
            print("    " + G["line"])
            print("    per-prompt r: " + ", ".join(f"{t} {c['r']:.2f}" if c.get("r") is not None else f"{t} —"
                                                  for t, c in G["cells"].items()))
        B = r.get("build") or {}
        if B.get("ld"):
            print(f"  build: {B['usable']}/{B['n']} usable" + (" (STAND-IN prices: rehearsal)" if B.get("stand_in") else "")
                  + f"; aligned {B['n_A']} tokens = H {B['n_H']} + S {B['n_S']} + reply {B['n_reply']}; appraisal line "
                  f"{B['n_R']['P1']}/{B['n_R']['P6']} tokens")
            print("    ld: " + ", ".join(f"{k} {B['ld'][k]:+.2f}" for k in PROMPTS) + f"; G_N {B['G_N']:+.2f}, G_P {B['G_P']:+.2f} nats "
                  f"({_p(B['G_P'] / B['G_N'] if B['G_N'] else None).strip()} of G_N); greedy A1 openings P1 {B.get('greedy_P1')}, "
                  f"P6 {B.get('greedy_P6')}")
            lay = B["layers"]["reply"]
            pick = [L for L in (38, 40, 41, 42, 44, 48) if L < len(lay["ratio"])] or list(range(len(lay["ratio"])))
            print("    reply tokens, |dP|/|dN| and projection of dP on dN: " + "  ".join(
                f"L{L} {lay['ratio'][L]:.2f}/{lay['proj'][L]:+.2f}" for L in pick if lay["ratio"][L] is not None))
        A = r.get("attention") or {}
        for k, v in (A.get("top_last_to_hi") or {}).items():
            print(f"  attention, last token -> appraisal digits ({k}): " + ", ".join(f"L{L}.{h} {m:.2f}" for L, h, m in v[:6]))
        P = r.get("patch") or {}
        for n_ in ("own@42", "plain@42", "vN@42", "top8 add", "rand8 add"):
            parts = [f"{t} {_p(P[t][n_]['share']).strip()}{_ci(P[t][n_])}" for t in PROMPTS if n_ in P.get(t, {})]
            if parts:
                print(f"  patch {n_:<10} " + " | ".join(parts))
        for lab, key in (("steer A1 (P)", "steer_P"), ("steer A0 (N)", "steer_N")):
            for t, cells in (r.get(key) or {}).items():
                for k, c in sorted(cells.items()):
                    if k == "none":
                        continue
                    print(f"  {lab} {t} {k:<10} moved {c.get('moved_pts') or 0:+.1f} pts, share {_p(c.get('share')).strip()}"
                          f"{_ci(c)}; valid {c['valid']}/{c['n']}, off-scale {c['off_scale']}")
        C = r.get("swap") or {}
        for t, lab in (("N6", "A1 into A0 (adding the appraisal)"), ("P6", "A0 into A1 (removing it)")):
            rows = sorted((int(n.split("@")[1]), c["share"]) for n, c in C.get(t, {}).items() if n.startswith("cure@"))
            if rows:
                print(f"  swap {lab}: " + _row_line(li, rows))
                seg = sorted((n, c["share"]) for n, c in C.get(t, {}).items() if not n.startswith("cure@"))
                print("      " + ", ".join(f"{n[5:]} {_p(v).strip()}" for n, v in seg))
        if r.get("window"):
            print(f"  swap window L{r['window']['band'][0]}-L{r['window']['band'][-1]} ({r['window']['rule']})")
        if r.get("pick"):
            print("  swap heads (discovery): " + ", ".join(f"L{L}.{h}" for L, h in r["pick"]["top"]))
        K = r.get("circuit") or {}
        for n_ in sorted({n_ for t in K for n_ in K[t]}):
            print(f"  circuit {n_:<12} " + " | ".join(f"{t} {_p(K.get(t, {}).get(n_, {}).get('share')).strip()}{_ci(K.get(t, {}).get(n_))}"
                                                    for t in ("N6", "P6")))
    print("\n  predictions (" + PRED_FILE + (f", sha256 {s['predictions_sha256'][:12]}" if s.get("predictions_sha256") else "") + "):")
    for p in s.get("predictions") or []:
        print(f"    {p['id']} {p['result']:<5} {p['claim']}\n          value: {p['value']}")
    ch = s.get("checks") or {}
    print(f"  checks: hooks fired as planned {ch.get('hooks_ok')}; clean rows agree within {ch.get('clean_spread_max')} nats; "
          f"steered openings aligned {ch.get('steer_aligned')}; leave-one-out {ch.get('loo')}")
    sec = s.get("secs_per_item") or {}
    print("  seconds per item: " + ", ".join(f"{k[3:]} {v:.1f}" for k, v in sec.items() if v)
          + (f" (estimated {s['estimate_min'][0]} min typical)" if s.get("estimate_min") else ""))
