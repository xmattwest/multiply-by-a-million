#!/usr/bin/env python3
"""test_interp_e3c.py: checks for the E3c layer sweep plug-in (scripts/interp/tasks_e3c.py on the pod;
scripts/interp_e3c_laptop.py behind the shim scripts/interp/exp_e3c.py; scripts/interp_e3c_figures.py).

    python3 tests/test_interp_e3c.py            laptop checks (stdlib), then the tiny-model checks in the
                                                rehearsal venv if it exists (else they are skipped, loudly)
    /tmp/si-interp-venv/bin/python tests/test_interp_e3c.py --torch     the tiny-model checks only

Laptop: Part B's layer pick, Part A/B summaries against E3b's own summaries on E3b's data (if present),
the after_step hand-over to Part B, the time estimate, and a full --preview session.
Tiny model (random-weight Gemma 4 of the 31B's class, fp32 on the CPU):
  - the stored differences equal E3b's at the same layer (and the leave-one-out vectors);
  - steering a layer with the fixture's own full difference equals E2's patch at that layer
    (task level: same greedy prices and digit probabilities; and the logits);
  - a batch whose rows steer different layers equals running each layer separately (Part A and Part B,
    bit for bit on the same batch shape; and at task level);
  - Part B's batch layout matches the laptop's count.
"""
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "interp"))
VENV_PY = "/tmp/si-interp-venv/bin/python"
TINY = "/tmp/si-tiny-31b"
E3B_RUN = os.path.join(ROOT, "data", "interp", "interp-e3b-gemma-4-31b-it-20261005T233449Z")
N = {"pass": 0, "fail": 0}


def ok(cond, name):
    N["pass" if cond else "fail"] += 1
    print(("ok    " if cond else "FAIL  ") + name)


def close(a, b, tol=1e-9):
    if a is None or b is None:
        return a is b
    return abs(a - b) <= tol


# ======================================================================================================
# laptop (stdlib)
# ======================================================================================================
def laptop():
    import launch_interp as li
    import interp_e3c_laptop as X
    X._LI[:] = [li]

    # --- pick_layers ---
    def cellA(fr, off=0, n=30):
        return {"fraction": fr, "off_scale": off, "n": n, "ci": None}

    def mkA(prof1, prof6, off=None):
        off = off or {}
        return {"cells": {"x1+v": {X.MAIN: {str(L): cellA(v, off.get(L, 0)) for L, v in prof1.items()}},
                          "x1e6-v": {X.MAIN: {str(L): cellA(v) for L, v in prof6.items()}}}}
    layers = [0, 2, 38, 40, 42, 44, 46, 58, 59]
    p1 = {0: 0.0, 2: 0.02, 38: 0.3, 40: 0.55, 42: 0.8, 44: 0.95, 46: 0.9, 58: 0.6, 59: 0.4}
    p6 = {0: 0.0, 2: 0.01, 38: 0.6, 40: 0.7, 42: 0.85, 44: 0.92, 46: 0.97, 58: 0.52, 59: 0.7}
    picked, info = X.pick_layers(mkA(p1, p6), layers)
    ok(picked == [40, 44, 58] and info["peak"] == 44 and info["earliest"] == 40 and info["last"] == 58,
       f"pick: earliest/peak/last by the smaller direction ({picked})")
    picked, info = X.pick_layers(mkA(p1, p6, off={40: 4}), layers)
    ok(picked == [42, 44, 58], "pick: a layer with > 10% off-scale readings is not eligible")
    q1 = {L: 0.0 for L in layers}; q1[44] = 0.6
    picked, info = X.pick_layers(mkA(q1, p6), layers)
    ok(picked == [2, 44, 46] or (44 in picked and len(picked) == 3 and info["filled"]),
       f"pick: fewer than 3 distinct -> filled with the next best ({picked}, filled {info['filled']})")
    picked, info = X.pick_layers(mkA({L: 0.1 + L / 1000 for L in layers}, {L: 0.3 for L in layers}), layers)
    ok(len(picked) == 3 and info["earliest"] is None and info["peak"] == 59, "pick: none >= 50% -> peak + next best")
    t1 = {L: 0.0 for L in layers}; t1[42] = t1[44] = 0.9
    picked, info = X.pick_layers(mkA(t1, {L: 0.9 for L in layers}), layers)
    ok(info["peak"] == 42, "pick: ties go to the earlier layer")
    picked, info = X.pick_layers({"cells": {}}, layers)
    ok(picked == [] and info["rule"] == "no eligible layer", "pick: nothing eligible -> no layers")

    # --- summaries against E3b's (layer 44) on E3b's own data ---
    if os.path.isdir(E3B_RUN):
        rows = [json.loads(l) for l in open(os.path.join(E3B_RUN, "results.jsonl"))]
        s_old = json.load(open(os.path.join(E3B_RUN, "summary.json")))
        br = [r for r in rows if r["step"] == "e3b bridge"]
        br44 = [dict(r, rows=[dict(x, layer=None if x["vector"] == "none" else 44) for x in r["rows"]]) for r in br]
        A = X.summarize_A(br44, li)
        old = li.summarize_e3b_bridge(br)
        same = all(close(A["cells"][d][k]["44"]["fraction"], c["fraction"]) and A["cells"][d][k]["44"]["off_scale"] == c["off_scale"]
                   and A["cells"][d][k]["44"]["ci"] == c["ci"] for d, cells in old["cells"].items() for k, c in cells.items())
        ok(same and len(old["cells"]["x1+v"]) == 14, "A summary with a layer axis = E3b's bridge summary at layer 44 (all 14 cells, CIs)")
        ok(close(A["cells"]["x1+v"][X.MAIN]["44"]["fraction"], s_old["bridge"]["cells"]["x1+v"][X.MAIN]["fraction"]),
           f"A: v per token, prompt+reply, x1+v = E3b's stored {100 * s_old['bridge']['cells']['x1+v'][X.MAIN]['fraction']:.1f}%")
        sr = [r for r in rows if r["step"] == "e3b steer"]
        keep = [dict(r, rows=[dict(x, layer=None if x["vector"] == "none" else 44) for x in r["rows"]
                              if x["vector"] == "none" or (x["vector"] == "v_pos" and x["scope"] == "prompt+reply"
                                                           and x["alpha"] in (1.0, 2.0))]) for r in sr]
        B = X.summarize_B(keep, li)
        for d in X.DIRS:
            want = s_old["steer"]["directions"][d]["v_pos|prompt+reply"]
            for a in ("1.0", "2.0"):
                got = B["share"][d][f"44|{a}"]
                cell = B["cells"]["directions"][d]["v_pos@44|prompt+reply"][a]
                ok(close(got["share"], want[a]["share_moved"], 1e-9) and close(cell["mean_gap"], want[a]["mean_gap"])
                   and cell["valid"] == want[a]["valid"] and got["ci"] and got["ci"][0] <= got["share"] <= got["ci"][1],
                   f"B share {d} α{a} = E3b's {100 * want[a]['share_moved']:.1f}% (CI {100 * got['ci'][0]:.0f}–{100 * got['ci'][1]:.0f}%)")
        Ba = X.summarize_B(keep, li, aligned_only=True)
        ok(Ba["share"]["x1+v"]["44|1.0"]["share"] is not None, "B aligned-only summary runs")
    else:
        print("skip  E3b data not present (summaries vs E3b)")

    # --- share_b: off-scale and other-scale prices are excluded ---
    def rec(direction, cond, prices, steered):
        rows = [{"layer": None, "vector": "none", "alpha": 0.0, "errors": [], "action": "OFFER", "price": p, "aligned": True}
                for p in prices] + [{"layer": 40, "vector": "v_pos", "alpha": 1.0, "errors": [], "action": "OFFER",
                                     "price": p, "aligned": True} for p in steered]
        return {"fixture": "Y01", "role": "seller", "direction": direction, "base_limit": 100, "target_condition": cond,
                "rows": rows}
    recs = [rec("x1+v", "D2", [140, 140], [120, 120, 99999]),          # 99999: off scale
            rec("x1e6-v", "D8", [120_000_000], [140_000_000, 130])]    # 130: other scale
    sh = X.share_b(recs, li, ci=False)
    ok(close(sh["x1+v"]["40|1.0"]["share"], 1.0) and close(sh["x1e6-v"]["40|1.0"]["share"], 1.0),
       "B share: off-scale and other-scale prices excluded, never averaged")

    # --- setup, after_step, estimate ---
    settings = json.load(open(os.path.join(ROOT, "prereg", "settings.json")))

    class A_:
        model, fixtures, batch, budget_min = "gemma-4-31b-it", 30, 16, 60.0
        e3c_layer_step, e3c_b_fixtures, e3c_b_alphas, e3c_b_samples, e3c_b_layers, e3c_threshold = 2, 30, "1,2", 3, "", 0.5
    plug = X.setup(A_(), settings, None, li)
    ctx = dict(plug["ctx"], args=A_(), panel=None)
    ok(ctx["e3c_layers"] == list(range(0, 60, 2)) + [59] and len(ctx["e3c_layers"]) == 31, "31 swept layers: 0, 2, ..., 58, 59")
    ok(len(ctx["a_items"]) == 60 and len(ctx["b_items"]) == 60 and len(ctx["vec_items"]) == 70, "items: 60 A, 60 B, 30+40 vectors")
    ok([it["fixture"] for it in ctx["b_items"]][:4] == ["Y01", "Y01", "Y02", "Y02"], "B items are fixture-major")
    ok(X.a_specs_per_item(31, 16) == 128 and X.b_rows_per_item(3, 2, 3, 16) == 32, "rows per item: A 128 (8 batches), B 32 (2 batches)")
    typ, worst = ctx["estimate_min"]
    ok(40 <= typ <= 50 and worst > 60, f"estimate {typ} min typical (fits 60), {worst} if every B batch hit the cap")
    S = steps(plug, ctx)
    ok([s[0] for s in S] == ["e3c vectors", "e3c A", "e3c B"] and S[2][3] is ctx["b_kw"], "steps; B's kw is the shared dict")
    X.on_pod_info(ctx, {"num_hidden_layers": 6})
    ok(ctx["e3c_layers"] == [0, 2, 4, 5] and ctx["a_kw"]["layers"] is ctx["e3c_layers"], "on_pod_info: layers follow the pod (6 -> 0,2,4,5), in place")
    X.on_pod_info(ctx, {"num_hidden_layers": 60})
    got = {"e3c vectors": [{"kept": list(ctx["e3c_layers"])}]}
    ok(X.after_step("e3c vectors", got, ctx) is None, "after vectors: stored layers match")
    ok(X.after_step("e3c vectors", {"e3c vectors": [{"kept": [0]}]}, ctx), "after vectors: a mismatch stops the session")
    stub_ctx = dict(ctx, panel=None)
    got["e3c A"] = X.stub("e3c_bridge", ctx["a_items"], stub_ctx)
    why = X.after_step("e3c A", got, ctx)
    ok(why is None and ctx["b_kw"]["layers"] == ctx["b_pick"]["layers"] and len(ctx["b_kw"]["layers"]) == 3,
       f"after A: B's layers set in place ({ctx['b_kw']['layers']}; {ctx['b_pick']['rule']})")
    ctx["forced_b_layers"] = [40, 44]
    ok(X.after_step("e3c A", got, ctx) is None and ctx["b_kw"]["layers"] == [40, 44], "--e3c-b-layers overrides the pick")
    ctx["forced_b_layers"] = [41]
    ok("not swept" in (X.after_step("e3c A", got, ctx) or ""), "--e3c-b-layers must be swept layers")
    ctx["forced_b_layers"] = []
    bad = [dict(r, rows=[dict(x, price=None) for x in r["rows"]]) for r in got["e3c A"]]
    ok("skipped" in (X.after_step("e3c A", {"e3c A": bad}, ctx) or ""), "after A: no eligible layer stops before B")

    # --- the full preview session ---
    import glob
    pat = os.path.join(ROOT, "data", "interp", "interp-e3c-*-preview")      # only ours: another chat may preview at once
    before = set(glob.glob(pat))
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = li.main(["--preview", "--experiment", "e3c", "--fixtures", "4"])
    ok("Part B layers: L" in buf.getvalue() and "moved" in buf.getvalue(), "preview report prints B's layers and shares")
    new = sorted(set(glob.glob(pat)) - before)
    ok(rc == 0 and len(new) == 1 and new[0].endswith("-preview"), "preview session runs and writes one run folder")
    if new:
        s = json.load(open(os.path.join(new[0], "summary.json")))
        m = json.load(open(os.path.join(new[0], "run_manifest.json")))
        res = [json.loads(l) for l in open(os.path.join(new[0], "results.jsonl"))]
        ok(not s.get("analysis_error") and s["b_pick"]["layers"] and s["B"]["layers"] == s["b_pick"]["layers"],
           f"preview: B ran at the layers picked from A ({s['b_pick']['layers']})")
        ok(m["plugin"]["name"] == "e3c" and m["roles"] == ["seller"] and len(m["plugin"]["layers"]) == 31, "preview manifest")
        ok({r["step"] for r in res} == {"e3c vectors", "e3c A", "e3c B"}, "preview: results for all three steps")
        import interp_e3c_figures as F
        import tempfile
        tmp = os.path.join(tempfile.mkdtemp(), "fig.json")
        real, F.JSON_PATH = F.JSON_PATH, tmp
        try:
            ok(F.cards(tmp + ".none") == [], "figure: no JSON yet -> no card")
            D = F.refresh(new[0])
            c = F.cards(tmp)
            ok(len(c) == 1 and c[0].count("<svg") == 1 and "<circle" in c[0] and len(D["B"]["x1+v"]) == 6
               and len(D["A"]["x1e6-v"]["v_pos|reply|1.0"]) == 31, "figure: refresh -> one card, 31-layer lines, B's 6 points")
        finally:
            F.JSON_PATH = real
            shutil.rmtree(os.path.dirname(tmp))
        import mi_figures
        ok(any("E3c · layer sweep" in x for x in mi_figures.extra_cards()) == os.path.exists(real),
           "the site's plug-in loader finds the shim (a card only once the tracked JSON exists)")
        for d in new:
            shutil.rmtree(d)


def steps(plug, ctx):
    return plug["steps"](ctx)


# ======================================================================================================
# tiny model (torch; rehearsal venv)
# ======================================================================================================
def tiny():
    os.environ["SI_INTERP_DEVICE"] = "cpu"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    import launch_interp as li
    import model_io
    import tasks
    import hooks
    import tasks_e3c as E                # its own copy of the plug-in (tasks loads another for TASKS); used throughout
    import interp_e3c_laptop as X
    torch.manual_seed(0)
    model, tok, info = model_io.load_text_only(TINY, dtype=torch.float32)
    ok(info["num_hidden_layers"] == 6 and str(next(model.parameters()).dtype) == "torch.float32",
       "tiny model loaded in fp32 on the CPU (6 layers)")
    ok(tasks.TASKS.get("e3c_bridge") is not None, "tasks loads the e3c plug-in (task names e3c_*)")
    settings = json.load(open(os.path.join(ROOT, "prereg", "settings.json")))
    vec, steer = li.build_e3_items(settings, 3)
    vec = [v for v in vec if v["kind"] == "pair"] + [v for v in vec if v["kind"] == "nonmoney"][:4]
    blocks = model_io.decoder_layers(model)
    kept = li.layer_list(6, 2)
    name = "gemma-4-31b-it"

    # 1. stored differences = E3b's at the same layer; leave-one-out vectors too
    info3 = E.e3c_build(model, tok, vec, settings, name, model_io, layers=kept)[0]
    st3 = E._STORE["e3c"]
    ok(info3["kept"] == kept and info3["n_post"] == 117 and info3["n_prompt"] == 106 and len(info3["per_layer"]["norm_v_prompt"]) == 6,
       f"build: {info3['n_pairs']} pairs, 117 post-limit tokens (106 + 11), per-layer readouts at all 6 layers")
    worst_d = worst_v = worst_u = 0.0
    for L in kept:
        tasks.e3b_build(model, tok, vec, settings, name, model_io, layer=L)
        sb = tasks._STORE["e3b"]
        for f in sb["d"]:
            worst_d = max(worst_d, float((st3["d"][f][st3["index"][L]].float() - sb["d"][f]).abs().max()))
            V3, _ = E._vectors(st3, f, L)
            Vb, _ = tasks._e3b_vectors(sb, f)
            worst_v = max(worst_v, float((V3["v_pos"] - Vb["v_pos"]).abs().max()),
                          float((V3["nonmoney"] - Vb["nonmoney"]).abs().max()))
        worst_u = max(worst_u, float((st3["u"][L] - sb["u"]).abs().max()))
    ok(worst_d < 1e-5 and worst_v < 1e-5 and worst_u < 1e-5,
       f"build: differences, LOO vectors and non-money = E3b's at every swept layer (max |Δ| {worst_d:.1e}, {worst_v:.1e}, {worst_u:.1e})")
    ok(abs(info3["per_layer"]["norm_v_prompt"][kept[-1]] - float(sum(st3["d"][f][-1].float() for f in st3["d"])[:106].div(len(st3["d"])).norm(dim=-1).mean())) < 1e-3,
       "build: |v| per layer = the norm of the stored mean")

    # 2. own full difference at layer L = E2's patch at L (task level, every swept layer)
    e2_items = li.build_e2_items(settings, 3)
    e2 = tasks.e2_patch(model, tok, e2_items, settings, name, model_io, layers=kept, positions=("post",), batch=4)
    br = E.e3c_bridge(model, tok, steer, settings, name, model_io, layers=kept, rows=(("own", "all", 1.0),), batch=4)
    to_e3c = {"x1->x1e6": "x1e6-v", "x1e6->x1": "x1+v"}
    e3 = {(r["fixture"], r["direction"], x["layer"]): x for r in br for x in r["rows"] if x["vector"] == "own"}
    n_cmp = same = 0
    worst_p = 0.0
    for r in e2:
        for x in [x for x in r["rows"] if x["layer"] != "clean"]:
            y = e3[(r["fixture"], to_e3c[r["direction"]], x["layer"])]
            n_cmp += 1
            same += x["price"] == y["price"] and x["cont"] == y["cont"]
            worst_p = max(worst_p, max(abs(a - b) for a, b in zip(x["p1"], y["p1"])))
    ok(n_cmp == 6 * len(kept) and same == n_cmp and worst_p <= 2e-4,
       f"own difference x1 at layer L = E2's swap at L: {same}/{n_cmp} same greedy continuation, digit probs within {worst_p:.0e}")
    # ... and the logits (batch 1 vs E2's patch, low level)
    worst_rel = 0.0
    for it_e2 in e2_items[:2]:
        it3 = next(s for s in steer if s["fixture"] == it_e2["fixture"] and s["direction"] == to_e3c[it_e2["direction"]])
        S = tasks._r2_encode(tok, model_io, it_e2["source"]["messages"], it_e2["source"]["limit_text"], "seller")
        Tg = tasks._r2_encode(tok, model_io, it_e2["target"]["messages"], it_e2["target"]["limit_text"], "seller")
        s0 = Tg["span"][0]
        src = tasks._capture_all_layers(model, model_io, S["ids"], list(range(S["span"][0], len(S["ids"]))), device_keep=True)
        src_post = src[:, len(S["span"]):, :]
        suf = Tg["ids"][s0:]
        for L in kept:
            base, _ = tasks._batched_prefix_cache(model, model_io, Tg["ids"][:s0], 1)
            p = hooks.BatchPatcher(blocks).add(L, 0, torch.arange(len(Tg["span"]), len(suf)), src_post[L])
            p.arm()
            try:
                lp = tasks._forward(model, model_io, input_ids=torch.tensor([suf]), past_key_values=base, use_cache=True).logits[0, -1].float()
            finally:
                p.disarm()
            E.e3c_bridge(model, tok, [it3], settings, name, model_io, layers=[L], rows=(("own", "all", 1.0),), batch=2, debug=True)
            ls = E._STORE["debug_logits"][0][1]
            worst_rel = max(worst_rel, float((lp - ls).abs().max() / lp.abs().max()))
    ok(worst_rel < 1e-5, f"... and the first-price-token logits equal E2's patch (max relative |Δ| {worst_rel:.1e})")

    # 3a. Part A: rows of one batch at different layers = each layer separately
    rows = X.A_ROWS + (("own", "reply", 1.0),)
    it = steer[1]
    multi = E.e3c_bridge(model, tok, [it], settings, name, model_io, layers=kept, rows=rows, batch=8, debug=True)[0]
    lm = torch.cat(E._STORE["debug_logits"])
    spec_m = [None] + [(L,) + tuple(r) for L in kept for r in rows]
    worst_sep, same, n_cmp = 0.0, 0, 0
    for L in kept:
        sep = E.e3c_bridge(model, tok, [it], settings, name, model_io, layers=[L], rows=rows, batch=8, debug=True)[0]
        ls = torch.cat(E._STORE["debug_logits"])
        spec_s = [None] + [(L,) + tuple(r) for r in rows]
        for j, sp in enumerate(spec_s[1:], 1):
            worst_sep = max(worst_sep, float((lm[spec_m.index(sp)] - ls[j]).abs().max() / ls[j].abs().max()))
        a = {(x["layer"], x["vector"], x["scope"]): (x["price"], x["cont"], x["p1"]) for x in multi["rows"] if x["layer"] == L}
        b = {(x["layer"], x["vector"], x["scope"]): (x["price"], x["cont"], x["p1"]) for x in sep["rows"] if x["vector"] != "none"}
        n_cmp += len(b)
        same += sum(a[k] == b[k] for k in b)
    ok(same == n_cmp == len(kept) * len(rows) and worst_sep < 1e-5,
       f"Part A: mixed-layer batches = one layer at a time: {same}/{n_cmp} rows identical, logits max relative |Δ| {worst_sep:.1e}")
    # bit for bit on the same batch shape: one forward with 2 layers = two forwards with one layer each
    E3 = tasks._r2_encode(tok, model_io, it["target"]["messages"], it["target"]["limit_text"], "seller")
    s0 = E3["span"][0]
    suf = E3["ids"][s0:]
    rel = torch.tensor([p - s0 for p in range(E3["span"][-1] + 1, len(E3["ids"]))])
    V2, _ = E._vectors(st3, it["fixture"], 2)
    V4, _ = E._vectors(st3, it["fixture"], 4)
    lay = [2, 2, 4, None]
    dl = [V2["v_pos"], -V2["v_pos"], V4["v_pos"], None]

    def fwd(row_layers, row_deltas):
        base, _ = tasks._batched_prefix_cache(model, model_io, E3["ids"][:s0], 4)
        ads = E.multi_prefill(blocks, row_layers, rel, row_deltas, len(suf))
        try:
            return tasks._forward(model, model_io, input_ids=torch.tensor([suf]).expand(4, -1).contiguous(),
                                  past_key_values=base, use_cache=True).logits[:, -1].float(), ads
        finally:
            E._remove(ads)
    lmix, ads = fwd(lay, dl)
    l2, _ = fwd([2, 2, None, None], dl[:2] + [None, None])
    l4, _ = fwd([None, None, 4, None], [None, None, dl[2], None])
    l0, _ = fwd([None] * 4, [None] * 4)
    bit = [float((lmix[0] - l2[0]).abs().max()), float((lmix[1] - l2[1]).abs().max()),
           float((lmix[2] - l4[2]).abs().max()), float((lmix[3] - l0[3]).abs().max())]
    ok(max(bit) == 0.0 and len(ads) == 2 and all(len(a.fired) == 1 for a in ads),
       f"Part A, same batch shape: 2-layer forward = 1-layer forwards bit for bit (max |Δ| {max(bit):.1e}); 1 adder per layer, each fired once")
    ok(float((lmix[0] - lmix[1]).abs().max()) > 0 and float((l0[0] - lmix[0]).abs().max()) > 0, "... and the steering does change the logits")

    # 3b. Part B: the same for generation (GenAdder per layer), greedy
    Ec = tasks._chat_encode(tok, model_io, it["target"]["messages"], it["target"]["limit_text"], "seller")
    s0c = Ec["span"][0]
    relc = torch.tensor([p - s0c for p in Ec["post"]])
    mask = tasks._scope_mask("prompt+reply", 106, 117, model.device)
    full = torch.tensor([Ec["ids"]]).expand(4, -1).contiguous()

    def gen(row_layers, row_full, new=24):
        base, _ = tasks._batched_prefix_cache(model, model_io, Ec["ids"][:s0c], 4)
        ads = E.multi_gen(blocks, row_layers, relc, row_full, 106, len(Ec["ids"]) - s0c)
        try:
            g = model.generate(full, attention_mask=torch.ones_like(full), past_key_values=base, max_new_tokens=new,
                               do_sample=False, temperature=None, top_p=None, top_k=None, output_scores=True,
                               return_dict_in_generate=True)
        finally:
            E._remove(ads)
        return g, ads
    rf = [V2["v_pos"] * mask, -2 * V2["v_pos"] * mask, V4["v_pos"] * mask, None]
    gm, ads = gen(lay, rf)
    g2, _ = gen([2, 2, None, None], rf[:2] + [None, None])
    g4, _ = gen([None, None, 4, None], [None, None, rf[2], None])
    g0, _ = gen([None] * 4, [None] * 4)
    sc = lambda g, r: torch.stack([s[r] for s in g.scores])                         # noqa: E731
    n_s = min(len(gm.scores), len(g2.scores), len(g4.scores), len(g0.scores))
    bitg = max(float((sc(gm, 0)[:n_s] - sc(g2, 0)[:n_s]).abs().max()), float((sc(gm, 1)[:n_s] - sc(g2, 1)[:n_s]).abs().max()),
               float((sc(gm, 2)[:n_s] - sc(g4, 2)[:n_s]).abs().max()), float((sc(gm, 3)[:n_s] - sc(g0, 3)[:n_s]).abs().max()))
    same_tok = all(torch.equal(gm.sequences[r], g.sequences[r]) for r, g in ((0, g2), (1, g2), (2, g4), (3, g0)))
    ok(bitg == 0.0 and same_tok and ads[0].calls[0] == len(Ec["ids"]) - s0c and len(ads[0].calls) == len(gm.scores),
       f"Part B, same batch shape: 2-layer generation = 1-layer generations bit for bit over {n_s} tokens; hook saw the prefill then each token")
    ok(float((sc(gm, 0)[:11] - sc(g0, 0)[:11]).abs().max()) > 0, "... and the reply-token steering does change the scores")
    # task level: mixed layers vs one layer at a time (greedy)
    bl = kept[1:]
    rm = E.e3c_steer(model, tok, [it], settings, name, model_io, layers=bl, alphas=(1.0, 2.0), samples=1, batch=8,
                     min_none=2, max_new=40, greedy=True)[0]
    same = n_cmp = 0
    for L in bl:
        rs = E.e3c_steer(model, tok, [it], settings, name, model_io, layers=[L], alphas=(1.0, 2.0), samples=1, batch=8,
                         min_none=2, max_new=40, greedy=True)[0]
        a = {(x["layer"], x["alpha"]): x["content"] for x in rm["rows"] if x["layer"] == L}
        b = {(x["layer"], x["alpha"]): x["content"] for x in rs["rows"] if x["vector"] != "none"}
        n_cmp += len(b)
        same += sum(a[k] == b[k] for k in b)
    ok(same == n_cmp == 2 * len(bl), f"Part B task: mixed-layer batches = one layer at a time ({same}/{n_cmp} greedy replies identical)")
    ok(all(c[0] == len(Ec["ids"]) - s0c and c[2] == len(bl) for c in rm["hook_calls"]), f"Part B task: hook calls {rm['hook_calls']}")

    # 4. Part B batch layout = the laptop's count
    good = True
    for nl, na, ns, b in ((3, 2, 3, 16), (3, 2, 4, 16), (1, 1, 1, 8), (3, 2, 3, 8), (2, 3, 2, 16)):
        plan = E.b_batches(list(range(nl)), [float(x + 1) for x in range(na)], ns, b)
        flat = [x for c in plan for x in c]
        st_ = [x for x in flat if x is not None]
        good &= len(flat) == X.b_rows_per_item(nl, na, ns, b) and len(st_) == nl * na * ns and len(flat) - len(st_) >= 8 \
            and all(len(c) == b for c in plan)
    p = E.b_batches([10, 20, 30], [1.0, 2.0], 3, 16)
    good &= [sorted({x[1] for x in c if x}) for c in p] == [[1.0], [2.0]] and [c.count(None) for c in p] == [7, 7]
    ok(good, "Part B layout: rows = the laptop's count, >= 8 unsteered, α 1 and α 2 in separate batches of 9 + 7 unsteered")


def main(argv):
    if "--torch" in argv:
        tiny()
    else:
        laptop()
        if os.path.exists(VENV_PY) and os.path.exists(os.path.join(TINY, "config.json")):
            print("--- tiny-model checks (rehearsal venv, fp32, CPU) ---")
            r = subprocess.run([VENV_PY, os.path.abspath(__file__), "--torch"], capture_output=True, text=True, timeout=1800)
            lines = [l for l in r.stdout.splitlines() if l.startswith(("ok", "FAIL", "skip"))]
            for l in lines:
                print(l)
                N["pass" if l.startswith("ok") else "fail"] += l.startswith(("ok", "FAIL"))
            if r.returncode != 0:
                N["fail"] += 1
                print("FAIL  tiny-model checks crashed:\n" + r.stderr[-3000:])
        else:
            N["fail"] += 1
            print("FAIL  tiny-model checks skipped: run bash scripts/interp/make_rehearsal_model.sh")
    print(f"\n{N['pass']} passed, {N['fail']} failed")
    return 1 if N["fail"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
