"""interp_e3c_laptop.py (LAPTOP-SIDE, standard library only): E3c, the seller layer sweep. The plug-in
scripts/interp/exp_e3c.py is a shim importing from here (keeps the pod's create request small).

    python3 scripts/launch_interp.py --preview --experiment e3c              (free: canned shapes)
    python3 scripts/launch_interp.py --model gemma-4-31b-it --experiment e3c --slot 8     (paid)

config/interp/plan_next.md, Session 1. At which layers does E3b's shared x1e6 - x1 direction steer Gemma 4
31B's seller opening, and does the prompt-vs-reply split hold at every layer? Exploratory.

  vectors  per-fixture differences at every layer on the 117 tokens after the limit (106 prompt + the reply's
           11 forced opening tokens); the swept layers stay on the pod.
  A        E2's setup (opening forced, greedy price), every 2nd layer 0-58 plus 59. Rows per layer, all at
           alpha 1: the shared direction (leave-one-fixture-out, per token) on prompt+reply, on the reply only,
           on the prompt only, and the non-money contrast on prompt+reply. Rows in one batch steer different
           layers. (The fixture's own difference at every layer is E2 itself, X52; not repeated.)
  B        real sampled openings at 3 layers picked from A (after_step): the earliest layer moving >= 50%
           both ways, the peak, and the last layer moving >= 50% both ways. alpha 1 and 2, 3 samples each,
           plus unsteered rows. Fixture-major, so a budget stop leaves complete fixtures.
Readouts: share of the gap moved by layer with CIs over fixtures (both directions), validity, off-scale prices
(excluded, never averaged), whether the generated opening matched the steered tokens, |v| per layer.
"""
import math
import statistics as st

A_ROWS = (("v_pos", "all", 1.0), ("v_pos", "reply", 1.0), ("v_pos", "prompt", 1.0), ("nonmoney", "all", 1.0))
A_KEYS = tuple(f"{v}|{s}|{a}" for v, s, a in A_ROWS)
A_LABEL = {"v_pos|all|1.0": "prompt+reply", "v_pos|reply|1.0": "reply only", "v_pos|prompt|1.0": "prompt only",
           "nonmoney|all|1.0": "non-money"}
MAIN = "v_pos|all|1.0"
DIRS = ("x1+v", "x1e6-v")
OFF_SHARE_MAX = 0.10            # a layer is eligible for B only if <= 10% of its A readings are off-scale, both ways
# Timing model (E3b on the pod, X54): A 1.9 s per batch of 16 (greedy, <= 12 tokens);
# B 0.22 s per decode step at batch 16, a normal batch ~65 steps, a batch that runs to the cap 192.
A_BATCH_S, B_STEP_S, B_STEPS_TYPICAL, B_STEPS_MAX, VEC_S = 1.9, 0.22, 65, 192, 0.6
_LI = []                        # launch_interp, handed over by setup() (the plug-in never imports it itself)


def _li():
    if _LI:
        return _LI[0]
    import sys
    m = sys.modules.get("__main__")
    return m if hasattr(m, "summarize_e3b_steer") else sys.modules.get("launch_interp")


# --- flags -----------------------------------------------------------------------------------------------
def add_args(ap):
    ap.add_argument("--e3c-layer-step", type=int, default=2, help="e3c: sweep every Nth layer, plus the last (default 2)")
    ap.add_argument("--e3c-b-fixtures", type=int, default=30, help="e3c: fixtures in Part B (default 30; fixture-major)")
    ap.add_argument("--e3c-b-alphas", default="1,2", help="e3c: Part B strengths (default 1,2)")
    ap.add_argument("--e3c-b-samples", type=int, default=3, help="e3c: Part B openings per (layer, alpha) (default 3)")
    ap.add_argument("--e3c-b-layers", default="", help="e3c: force Part B's layers (e.g. 40,44,52) instead of picking from A")
    ap.add_argument("--e3c-threshold", type=float, default=0.5, help="e3c: share moved (both ways) for B's earliest/last")


def configure(a):
    a.roles = "seller"


def _int_list(s):
    return [int(x) for x in str(s).split(",") if x.strip()]


# --- setup -------------------------------------------------------------------------------------------------
def a_specs_per_item(n_layers_swept, batch):
    n = 1 + len(A_ROWS) * n_layers_swept
    return int(math.ceil(n / batch) * batch)


def b_rows_per_item(n_layers, n_alphas, samples, batch, min_none=8):
    steered = n_layers * n_alphas * samples
    return int(math.ceil((steered + min_none) / batch) * batch)


def estimate_min(n_vec, n_a, n_b, n_swept, batch, b_layers=3, n_alphas=2, samples=3):
    """(minutes typical, minutes if every Part B batch runs to the token cap)."""
    a = n_a * a_specs_per_item(n_swept, batch) / batch * A_BATCH_S
    nb = b_rows_per_item(b_layers, n_alphas, samples, batch) / batch
    b_typ = n_b * nb * B_STEPS_TYPICAL * B_STEP_S
    b_max = n_b * nb * B_STEPS_MAX * B_STEP_S
    v = n_vec * VEC_S
    return (v + a + b_typ) / 60, (v + a + b_max) / 60


def setup(a, settings, schema, li):
    _LI[:] = [li]
    vec_items, steer_items = li.build_e3_items(settings, a.fixtures)
    nb_fix = max(1, min(a.e3c_b_fixtures, a.fixtures))
    keep_fix = sorted({it["fixture"] for it in steer_items})[:nb_fix]
    b_items = [it for it in steer_items if it["fixture"] in keep_fix]
    layers = li.layer_list(60, a.e3c_layer_step)          # replaced by the pod's own count once it reports
    alphas = [float(x) for x in a.e3c_b_alphas.split(",") if x.strip()]
    forced = _int_list(a.e3c_b_layers)
    li.STEP_GUESS_S.update({"e3c vectors": VEC_S, "e3c A": a_specs_per_item(len(layers), a.batch) / a.batch * A_BATCH_S,
                            "e3c B": b_rows_per_item(3, len(alphas), a.e3c_b_samples, a.batch) / a.batch
                            * B_STEPS_TYPICAL * B_STEP_S})
    typ, worst = estimate_min(len(vec_items), len(steer_items), len(b_items), len(layers), a.batch,
                              n_alphas=len(alphas), samples=a.e3c_b_samples)
    ctx = {"li": li, "e3c_layers": layers, "vec_items": vec_items, "a_items": steer_items, "b_items": b_items,
           "a_kw": {"layers": layers, "batch": a.batch},
           "b_kw": {"layers": [], "alphas": alphas, "samples": a.e3c_b_samples, "batch": a.batch, "schema": schema},
           "forced_b_layers": forced, "threshold": a.e3c_threshold, "estimate_min": [round(typ, 1), round(worst, 1)]}
    n_np = sum(1 for x in vec_items if x["kind"] == "nonmoney")
    plan = (f"E3c layer sweep on {a.model} (sellers): directions at every layer from {a.fixtures} fixture pairs "
            f"(leave-one-fixture-out) + {n_np} non-money pairs; A = {len(steer_items)} items x {len(layers)} layers "
            f"(step {a.e3c_layer_step}, plus the last) x {len(A_ROWS)} rows ({', '.join(A_LABEL[k] for k in A_KEYS)}; "
            f"alpha 1) in E2's forced-opening setup; B = {len(b_items)} items x "
            f"{b_rows_per_item(3, len(alphas), a.e3c_b_samples, a.batch)} real sampled openings at 3 layers "
            + ("forced " + ",".join(map(str, forced)) if forced else "picked from A") +
            f" (alpha {'/'.join(f'{x:g}' for x in alphas)}, {a.e3c_b_samples} samples, rest unsteered); batch {a.batch}; "
            f"about {typ:.0f} min of experiment time ({worst:.0f} if every B batch ran to the token cap; "
            f"--budget-min {a.budget_min:g} stops new items)")
    inputs = ([{"step": "vectors", **it} for it in vec_items] + [{"step": "A", **it} for it in steer_items]
              + [{"step": "B", "item_id": it["item_id"]} for it in b_items])
    return {"exp": "E3c", "plan": plan, "inputs": inputs, "ctx": ctx, "steps": steps, "live": live,
            "after_step": after_step}


def steps(ctx):
    return [("e3c vectors", "e3c_build", ctx["vec_items"], {"layers": ctx["e3c_layers"]}, len(ctx["vec_items"])),
            ("e3c A", "e3c_bridge", ctx["a_items"], ctx["a_kw"], 4),
            ("e3c B", "e3c_steer", ctx["b_items"], ctx["b_kw"], 1)]


def on_pod_info(ctx, pod_info):
    n = pod_info.get("num_hidden_layers")
    if n:
        ctx["e3c_layers"][:] = ctx["li"].layer_list(int(n), ctx["args"].e3c_layer_step)


def after_step(label, got, ctx):
    """After A: set B's layers in place (a gate if none qualifies). After the vectors: check what was stored."""
    if label == "e3c vectors":
        info = (got.get("e3c vectors") or [None])[0]
        if not info or info.get("kept") != list(ctx["e3c_layers"]):
            return "the pod did not store the swept layers"
        return None
    if label != "e3c A":
        return None
    layers = list(ctx["e3c_layers"])
    if ctx["forced_b_layers"]:
        bad = [L for L in ctx["forced_b_layers"] if L not in layers]
        if bad:
            return f"--e3c-b-layers {bad} are not swept layers"
        picked, info = sorted(ctx["forced_b_layers"]), {"rule": "forced by --e3c-b-layers"}
    else:
        A = summarize_A(got.get("e3c A") or [], ctx["li"], ci=False)
        picked, info = pick_layers(A, layers, ctx["threshold"])
    ctx["b_pick"] = dict(info, layers=picked)
    if not picked:
        return "no swept layer is on scale in both directions; Part B skipped"
    ctx["b_kw"]["layers"][:] = picked
    if ctx.get("panel") is not None:
        ctx["panel"].note("Part B layers: " + ", ".join(f"L{L}" for L in picked) + f" ({info.get('rule', '')[:70]})")
    return None


# --- Part A ----------------------------------------------------------------------------------------------
def summarize_A(recs, li, ci=True):
    """E3b's bridge summary with a layer axis: per direction, readout 'vector|scope|alpha' and layer, the share
    of the x1/x1e6 gap moved = mean(g_steered - g_t) / mean(g_s - g_t) over fixtures with both clean readings
    (g_t: this item's clean reading; g_s: the same fixture's other direction, clean). Off-scale or missing
    prices are counted, never averaged. CIs: bootstrap over fixtures."""
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
            key = (r["direction"], f"{x['vector']}|{x['scope']}|{x['alpha']}", str(x["layer"]))
            c = acc.setdefault(key, {"num": [], "den": [], "off_scale": 0, "n": 0})
            c["n"] += 1
            g, _m, off = li.scale_free(r["role"], x["price"], r["base_limit"])
            if off or g is None:
                c["off_scale"] += 1
                continue
            c["num"].append(g - g_t)
            c["den"].append(g_s - g_t)
    out = {"cells": {}, "n_fixtures": len({r["fixture"] for r in recs}), "n_items": len(recs),
           "clean_consistent": sum(1 for r in recs if r.get("clean_consistent")),
           "layers": sorted({int(x["layer"]) for r in recs for x in r["rows"] if x["vector"] != "none"})}
    for (dname, key, L), c in acc.items():
        den = st.mean(c["den"]) if c["den"] else None
        cell = {"n": c["n"], "off_scale": c["off_scale"], "fraction": None, "ci": None}
        if den and c["num"]:
            cell["fraction"] = st.mean(c["num"]) / den
            if ci and len(c["num"]) >= 3:
                lo, hi = li.boot_ci(c["num"], n=1000)
                cell["ci"] = sorted([lo / den, hi / den])
        out["cells"].setdefault(dname, {}).setdefault(key, {})[L] = cell
    return out


def _eligible(c):
    return c is not None and c.get("fraction") is not None and c["off_scale"] <= OFF_SHARE_MAX * c["n"]


def pick_layers(A, layers, thr=0.5):
    """Part B's layers from A's main readout (shared direction, prompt+reply, alpha 1). A layer's score is the
    smaller of its two directions' shares (it 'moves >= 50%' only if it does so both ways); layers with > 10%
    off-scale readings in either direction are not eligible. earliest = first layer with score >= thr,
    peak = highest score (ties: the earlier layer), last = last layer with score >= thr. If fewer than 3
    distinct layers result, the next-highest-scoring eligible layers are added."""
    score = {}
    for L in layers:
        cs = [A.get("cells", {}).get(d, {}).get(MAIN, {}).get(str(L)) for d in DIRS]
        if all(_eligible(c) for c in cs):
            score[L] = min(c["fraction"] for c in cs)
    if not score:
        return [], {"rule": "no eligible layer", "score": {}}
    peak = max(score, key=lambda L: (score[L], -L))
    above = [L for L in sorted(score) if score[L] >= thr]
    info = {"score": {str(L): round(v, 4) for L, v in sorted(score.items())}, "peak": peak, "threshold": thr,
            "earliest": above[0] if above else None, "last": above[-1] if above else None, "filled": []}
    picked = []
    for L in (info["earliest"], peak, info["last"]):
        if L is not None and L not in picked:
            picked.append(L)
    for L in sorted(score, key=lambda L: (-score[L], L)):
        if len(picked) >= 3:
            break
        if L not in picked:
            picked.append(L)
            info["filled"].append(L)
    info["rule"] = (f"earliest >= {thr:.0%} both ways L{info['earliest']}, peak L{peak}, last >= {thr:.0%} L{info['last']}"
                    if above else f"none reached {thr:.0%} both ways; peak L{peak} + next best")
    return sorted(picked), info


# --- Part B ----------------------------------------------------------------------------------------------
def _flat_b(recs):
    return [dict(r, rows=[dict(x, vector=x["vector"] if x["vector"] == "none" else f"{x['vector']}@{x['layer']}")
                          for x in r["rows"]]) for r in recs]


def share_b(recs, li, ci=True, aligned_only=False):
    """Share of the gap moved per (direction, layer, alpha) with a CI over fixtures: per fixture,
    num_f = mean steered gap - mean unsteered gap (same item); den_f = the other direction's unsteered mean
    minus this one's. share = mean(num_f) / mean(den_f) over fixtures having both. Only valid OFFERs on the
    target's own scale count (off-scale and other-scale prices are excluded, as in summarize_e3)."""
    per = {}
    for r in recs:
        own = 0 if r["target_condition"] == "D2" else 6
        for x in r["rows"]:
            if aligned_only and x["vector"] != "none" and not x.get("aligned"):
                continue
            if x.get("errors") or x.get("action") != "OFFER" or not x.get("price"):
                continue
            g, m, off = li.scale_free(r["role"], x["price"], r["base_limit"])
            if off or m != own:
                continue
            k = "none" if x["vector"] == "none" else f"{x['layer']}|{x['alpha']}"
            per.setdefault((r["direction"], r["fixture"]), {}).setdefault(k, []).append(g)
    base = {k: st.mean(v["none"]) for k, v in per.items() if v.get("none")}
    out = {}
    for dname in DIRS:
        other = DIRS[1 - DIRS.index(dname)]
        keys = sorted({k for (d, _f), v in per.items() if d == dname for k in v if k != "none"},
                      key=lambda s: (int(s.split("|")[0]), float(s.split("|")[1])))
        for k in keys:
            num, den = [], []
            for (d, f), v in per.items():
                if d != dname or k not in v or (d, f) not in base or (other, f) not in base:
                    continue
                num.append(st.mean(v[k]) - base[(d, f)])
                den.append(base[(other, f)] - base[(d, f)])
            cell = {"n_fixtures": len(num), "share": None, "ci": None}
            if num and st.mean(den):
                D = st.mean(den)
                cell["share"] = st.mean(num) / D
                cell["moved_pts"] = 100 * st.mean(num)
                if ci and len(num) >= 3:
                    lo, hi = li.boot_ci(num, n=1000)
                    cell["ci"] = sorted([lo / D, hi / D])
            out.setdefault(dname, {})[k] = cell
    return out


def summarize_B(recs, li, ci=True, aligned_only=False):
    cells = li.summarize_e3b_steer(_flat_b(recs), ci=ci, aligned_only=aligned_only)
    return {"cells": cells, "share": share_b(recs, li, ci=ci, aligned_only=aligned_only),
            "layers": sorted({int(x["layer"]) for r in recs for x in r["rows"] if x["vector"] != "none"}),
            "n_fixtures": len({r["fixture"] for r in recs}), "n_items": len(recs)}


def summarize(got, ctx):
    li = ctx["li"]
    ra, rb = got.get("e3c A") or [], got.get("e3c B") or []
    return {"vectors": (got.get("e3c vectors") or [None])[0], "layers": list(ctx["e3c_layers"]),
            "A": summarize_A(ra, li), "b_pick": ctx.get("b_pick"),
            "B": summarize_B(rb, li), "B_aligned_only": summarize_B(rb, li, aligned_only=True),
            "secs_per_item": {"A": st.mean(r["secs"] for r in ra) if ra else None,
                              "B": st.mean(r["secs"] for r in rb) if rb else None},
            "estimate_min": ctx.get("estimate_min")}


def manifest(ctx):
    a = ctx["args"]
    return {"name": "e3c", "plan": "config/interp/plan_next.md, Session 1", "layers": list(ctx["e3c_layers"]),
            "a_rows": [list(r) for r in A_ROWS], "b": {k: v for k, v in ctx["b_kw"].items() if k != "schema"},
            "b_fixtures": sorted({it["fixture"] for it in ctx["b_items"]}), "threshold": ctx["threshold"],
            "forced_b_layers": ctx["forced_b_layers"], "layer_step": a.e3c_layer_step,
            "estimate_min": ctx.get("estimate_min")}


# --- panel and report --------------------------------------------------------------------------------------
def _pct(v):
    return "  —  " if v is None else f"{100 * v:+4.0f}%"


def _ci(c):
    return f" [{100 * c['ci'][0]:+.0f}, {100 * c['ci'][1]:+.0f}]" if c and c.get("ci") else ""


def profile_lines(A, layers, li, indent="  "):
    out = []
    for dname in DIRS:
        for j, key in enumerate(A_KEYS):
            cells = A["cells"].get(dname, {}).get(key, {})
            fr = [cells.get(str(L), {}).get("fraction") for L in layers]
            v, L = li.peak(layers, fr)
            off = sum(c["off_scale"] for c in cells.values())
            tag = (f"peak L{L} {100 * v:.0f}%" if v is not None else "") + (f" · off-scale {off}" if off else "")
            out.append(f"{indent}{dname if j == 0 else '':<7} {A_LABEL[key]:<13} {li.spark(fr)}  {tag}")
    return out


def live(got):
    li = _li()
    out = []
    info = (got.get("e3c vectors") or [None])[0]
    if info:
        p = info["per_layer"]
        mid = min(44, info["layers"] - 1)
        out.append(f"E3c vectors: {info['n_pairs']} pairs × {info['layers']} layers, {info['n_prompt']} prompt + "
                   f"{info['n_reply']} reply tokens; stored {len(info['kept'])} layers ({info['stored_gib']} GiB); "
                   f"L{mid} |v| {p['norm_v_prompt'][mid]:.0f}/{p['norm_v_reply'][mid]:.0f}, fixtures agree "
                   f"cos {p['cos_fixture_diffs_mean'][mid]:.2f}")
    ra = got.get("e3c A") or []
    if ra and li is not None:
        A = summarize_A(ra, li, ci=False)
        layers = A["layers"]
        out.append(f"E3c A share moved by layer L{layers[0]}→L{layers[-1]} (▁ 0, █ 100%), α 1, {A['n_fixtures']} fixtures:")
        out += profile_lines(A, layers, li)
    rb = got.get("e3c B") or []
    if rb and li is not None:
        sh = share_b(rb, li, ci=False)
        out.append(f"E3c B real openings, share moved ({len({r['fixture'] for r in rb})} fixtures):")
        for dname in DIRS:
            parts = [f"L{k.split('|')[0]} α{float(k.split('|')[1]):g} {_pct(c['share']).strip()}"
                     for k, c in sh.get(dname, {}).items()]
            out.append(f"  {dname:<7} " + " · ".join(parts))
    return out


def stub(task, items, ctx):
    """Preview only: canned shapes, not a model."""
    layers = list(ctx["e3c_layers"])
    nL = (ctx.get("panel").pod or {}).get("num_hidden_layers", 60) if ctx.get("panel") else 60
    sig = lambda x: 1 / (1 + math.exp(-x))                                             # noqa: E731
    prof = {"v_pos|all|1.0": lambda L: 0.95 * sig((L - 39) / 1.5) * (1 - 0.45 * sig((L - 54) / 2)),
            "v_pos|reply|1.0": lambda L: 0.8 * sig((L - 41) / 2),
            "v_pos|prompt|1.0": lambda L: 0.5 * sig((L - 30) / 3) * (1 - sig((L - 42) / 2)),
            "nonmoney|all|1.0": lambda L: 0.05}
    if task == "e3c_build":
        return [{"layers": nL, "kept": layers, "n_pairs": 30, "n_post": 117, "n_prompt": 106, "n_reply": 11,
                 "reply_tokens": [], "n_nonmoney": 40, "stored_gib": 1.17, "secs": 50.0,
                 "per_layer": {"norm_v_prompt": [round(2 + 0.6 * L, 2) for L in range(nL)],
                               "norm_v_reply": [round(1 + 0.6 * L, 2) for L in range(nL)],
                               "resid_norm_prompt": [round(20 + 3 * L, 2) for L in range(nL)],
                               "resid_norm_reply": [round(20 + 3 * L, 2) for L in range(nL)],
                               "cos_fixture_diffs_mean": [0.9] * nL, "cos_v_mean_u": [0.3] * nL,
                               "norm_u": [round(1 + 0.3 * L, 2) for L in range(nL)]}}]
    out = []
    for it in items:
        g0, g1 = (0.41, 0.21) if it["direction"] == "x1+v" else (0.21, 0.41)
        unit = it["target"]["limit_units"]
        jit = (sum(map(ord, it["fixture"])) % 7 - 3) / 100
        price = lambda f: round((1 + g0 + f * (g1 - g0) + jit / 3) * unit)               # noqa: E731,B023
        base = {"item_id": it["item_id"], "fixture": it["fixture"], "role": it["role"], "direction": it["direction"],
                "base_limit": it["base_limit"], "target_condition": it["target"]["condition"], "loo": True}
        if task == "e3c_bridge":
            rows = [{"layer": L, "vector": v, "scope": s, "alpha": a, "price": price(prof[f"{v}|{s}|{a}"](L)), "cont": ""}
                    for L in layers for v, s, a in A_ROWS]
            rows.append({"layer": None, "vector": "none", "scope": "all", "alpha": 0.0, "price": price(0), "cont": ""})
            out.append(dict(base, rows=rows, clean_consistent=True, secs=0.1))
        else:
            kw = ctx["b_kw"]
            rows = [{"layer": None, "vector": "none", "scope": "prompt+reply", "alpha": 0.0, "sample": s,
                     "errors": [], "action": "OFFER", "finish": "stop", "aligned": True, "price": price(0)}
                    for s in range(14)]
            for L in kw["layers"]:
                for a in kw["alphas"]:
                    for s in range(kw["samples"]):
                        rows.append({"layer": L, "vector": "v_pos", "scope": "prompt+reply", "alpha": a, "sample": s,
                                     "errors": [], "action": "OFFER", "finish": "stop", "aligned": True,
                                     "price": price(min(a, 1.6) * prof[MAIN](L))})
            out.append(dict(base, rows=rows, secs=0.2))
    return out


def report(s, preview):
    li = _li()
    print("E3c layer sweep (exploratory" + (", PREVIEW: canned shapes" if preview else "") + ")")
    v = s.get("vectors")
    if v:
        p = v["per_layer"]
        print(f"  vectors: {v['n_pairs']} fixture pairs x {v['layers']} layers, {v['n_prompt']} prompt + {v['n_reply']} reply "
              f"tokens, {v['n_nonmoney']} non-money pairs; stored {len(v['kept'])} layers ({v['stored_gib']} GiB) on the pod")
        print("  |v| per token prompt/reply (activation norm prompt/reply; fixtures agree cos; cos with non-money):")
        for L in range(0, v["layers"], max(1, v["layers"] // 12)):
            print(f"    L{L:<3} {p['norm_v_prompt'][L]:6.1f} / {p['norm_v_reply'][L]:<6.1f} ({p['resid_norm_prompt'][L]:.0f} / "
                  f"{p['resid_norm_reply'][L]:.0f}; cos {p['cos_fixture_diffs_mean'][L]:.2f}; {p['cos_v_mean_u'][L]:+.2f})")
    A = s.get("A") or {}
    layers = A.get("layers") or []
    if layers:
        print(f"  A. forced opening, greedy price (E2's setup), α 1, {A['n_fixtures']} fixtures; share of the gap moved "
              f"by layer (▁ 0, █ 100%), clean rows consistent {A['clean_consistent']}/{A['n_items']}:")
        for line in profile_lines(A, layers, li, "    "):
            print(line)
        print("    layer  prompt+reply ×1+v [CI]  ×10⁶−v [CI]       reply only     prompt only    non-money      off-scale")
        for L in layers:
            c = {(d, k): A["cells"].get(d, {}).get(k, {}).get(str(L), {}) for d in DIRS for k in A_KEYS}
            off = sum(x.get("off_scale", 0) for x in c.values())
            main = "  ".join(f"{_pct(c[(d, MAIN)].get('fraction'))}{_ci(c[(d, MAIN)]):<14}" for d in DIRS)
            rest = "  ".join(f"{_pct(c[('x1+v', k)].get('fraction')).strip()} / {_pct(c[('x1e6-v', k)].get('fraction')).strip():<7}"
                             for k in A_KEYS[1:])
            print(f"    L{L:<4} {main}  {rest}  {off or ''}")
    pk = s.get("b_pick")
    if pk:
        print(f"  Part B layers: {', '.join(f'L{L}' for L in pk.get('layers') or [])} ({pk.get('rule')})")
    for name, B in (("B", s.get("B")), ("B, aligned openings only", s.get("B_aligned_only"))):
        if not B or not B.get("n_items"):
            continue
        print(f"  {name}. real sampled openings, steered on the prompt and the reply's opening; {B['n_fixtures']} fixtures:")
        for dname in DIRS:
            cells = B["cells"]["directions"].get(dname, {})
            base = cells.get("none", {}).get("0.0", {})
            print(f"    {dname}: unsteered gap {base.get('mean_gap') or 0:+.1f}% (n {base.get('n')}, valid {base.get('valid')})")
            for k, sc in B["share"].get(dname, {}).items():
                L, a = k.split("|")
                c = cells.get(f"v_pos@{L}|prompt+reply", {}).get(a, {})
                g = "—" if c.get("mean_gap") is None else f"{c['mean_gap']:+.1f}%"
                print(f"      L{L:<3} α {float(a):<3g} gap {g:>7}  moved {_pct(sc['share']).strip():>5}{_ci(sc):<13}"
                      f" valid {c.get('valid')}/{c.get('n')}  aligned {100 * (c.get('aligned_share') or 0):.0f}%"
                      + (f"  off-scale {c['off_scale'] + c['other_scale']}" if c.get("off_scale", 0) + c.get("other_scale", 0) else "")
                      + (f"  hit length {c['length']}" if c.get("length") else ""))
    sec = s.get("secs_per_item") or {}
    if sec.get("A") or sec.get("B"):
        print(f"  speed: A {sec.get('A') or 0:.1f} s/item, B {sec.get('B') or 0:.1f} s/item"
              + (f" (estimated {s['estimate_min'][0]} min typical)" if s.get("estimate_min") else ""))
