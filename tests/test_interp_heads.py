#!/usr/bin/env python3
"""test_interp_heads.py: checks for the heads plug-in (scripts/interp/tasks_heads.py on the pod;
scripts/interp_heads_laptop.py behind the shim scripts/interp/exp_heads.py).

    python3 tests/test_interp_heads.py           laptop checks (stdlib), then the tiny-model checks in the
                                                 rehearsal venv if it exists (else they fail, loudly)
    /tmp/si-interp-venv/bin/python tests/test_interp_heads.py --torch     the tiny-model checks only

Laptop: layer sets and row counts, the head pick, shares and interactions on synthetic records, the after_step
hand-overs (build gate, circuit rows in place), the time estimate, and a full --preview session.
Tiny model (random-weight Gemma 4 of the 31B's class, fp32 on the CPU):
  - the logit lens at the last layer equals the model's own ld; the steering difference equals E3b's;
  - a residual patch row equals E2's 'post' patch at that layer (digit probabilities);
  - patching every head of a layer equals patching the layer's attention output;
  - patching a run with its own values changes nothing; mixed batches equal rows run alone;
  - the knockout is exact: with attention to the digits blocked at every layer, two prompts that differ only in
    the limit's digits give identical readouts (sdpa and eager), and per-head = all-heads knockout;
  - the steer row equals E3b's bridge (v per token, all post-limit tokens, alpha 1) at that layer;
  - the attention step returns masses in [0, 1] and restores the attention setting.
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
N = {"pass": 0, "fail": 0}


def ok(cond, name):
    N["pass" if cond else "fail"] += 1
    print(("ok    " if cond else "FAIL  ") + name)


# ======================================================================================================
# laptop (stdlib)
# ======================================================================================================
def laptop():
    import launch_interp as li
    import interp_heads_laptop as X
    X._LI[:] = [li]
    opts = {"heads": list(range(38, 45)), "steer": 20, "ko_from": 21, "top": 8, "discovery": 10}
    L = X.layer_sets(60, opts)
    rows = X.sweep_rows(L, 32)
    ok(len(L["resid"]) == 36 and len(L["comp"]) == 22 and L["heads"] == list(range(38, 45)) and not L["mapped"],
       "31B layer sets: 36 residual, 22 attention/MLP, heads at 38-44")
    ok(len(rows) == 36 + 44 + 224 and X._batches(len(rows), 16) == 20, "sweep: 304 rows + clean = 20 batches of 16")
    ok(len({r["n"] for r in rows}) == len(rows), "sweep row names are unique")
    T6 = X.layer_sets(6, opts)
    ok(T6["mapped"] and T6["resid"] == list(range(6)) and all(0 <= x < 6 for x in T6["heads"] + T6["comp"])
       and T6["ko_from"] > T6["steer"], f"6-layer mapping stays in range, knockout after the steer ({T6})")
    top = [(41, 7), (41, 19), (42, 3), (44, 11), (38, 0), (39, 1), (40, 2), (43, 4)]
    rand = [(38, 5), (38, 6), (39, 7), (40, 8), (41, 9), (42, 10), (43, 11), (44, 12)]
    C = X.circuit_rows(top, rand, L, 32)
    names = [r["n"] for r in C]
    ok(len(C) == 15 and X._batches(len(C), 16) == 1, "circuit: 15 rows + clean = one batch of 16")
    ok(names[:3] == ["top1", "top4", "top8"] and "random8" in names and "steer+ko top" in names, f"circuit rows {names}")
    other = next(r for r in C if r["n"] == "other heads")
    ok(len(other["p"]) == 224 - 8 and not ({(p[1], p[2]) for p in other["p"]} & set(top)), "'other heads' = the band minus the top 8")
    ko_all = next(r for r in C if r["n"] == "ko all")
    ok([k[0] for k in ko_all["k"]] == list(range(21, 60)) and all(k[1] == -1 for k in ko_all["k"]), "'ko all' blocks every head at 21-59")
    ok(next(r for r in C if r["n"] == "steer+ko top")["k"] == [[a, b] for a, b in top], "'ko top' blocks only the top heads")

    # --- shares / pick / interactions on synthetic records ---
    def rec(f, d, clean, src, rows):
        return {"fixture": f, "direction": d, "ld_clean": clean, "ld_source": src,
                "rows": [{"n": n, "ld": v} for n, v in rows.items()] + [{"n": "clean", "ld": clean}]}
    recs = []
    for i, f in enumerate(["Y01", "Y02", "Y03", "Y04"]):
        g = 1 + i / 10
        recs.append(rec(f, "x1->x1e6", 4 * g, -4 * g, {"h41.7": 4 * g - 0.5 * 8 * g, "h41.8": 4 * g - 0.1 * 8 * g,
                                                       "h42.0": 4 * g - 0.4 * 8 * g}))
        recs.append(rec(f, "x1e6->x1", -4 * g, 4 * g, {"h41.7": -4 * g + 0.3 * 8 * g, "h41.8": -4 * g + 0.6 * 8 * g}))
    recs.append({"fixture": "Y05", "direction": "x1->x1e6", "skipped": "no usable prices"})
    S = X.shares(recs)
    ok(abs(S["x1->x1e6"]["h41.7"]["share"] - 0.5) < 1e-9 and abs(S["x1e6->x1"]["h41.8"]["share"] - 0.6) < 1e-9,
       "share = mean(ld_row - ld_clean) / mean(ld_source - ld_clean); skipped items ignored")
    c = S["x1->x1e6"]["h41.7"]
    ok(c["n"] == 4 and c["ci"] and c["ci"][0] <= 0.5 <= c["ci"][1], f"paired bootstrap CI brackets the share ({c['ci']})")
    top_, rand_, info = X.pick_heads(S, [41, 42], 9, 2)
    ok(top_ == [(41, 7), (41, 8)] and rand_ == [] and info["scores"] == {"h41.7": 0.4, "h41.8": 0.35},
       f"pick: by the mean of both directions (0.40 > 0.35); a head read in one direction only is left out ({top_}, {rand_})")
    circ = []
    for i, f in enumerate(["Y06", "Y07", "Y08"]):
        circ.append(rec(f, "x1->x1e6", 4, -4, {"steer": 4 - 0.8, "ko all": 4 - 3.2, "steer+ko all": 4 - 7.2}))
    I = X.interactions(circ, keys=("all",))
    ok(abs(I["x1->x1e6"]["all"]["share"] - 0.4) < 1e-9, "interaction = (steer+ko) - ko - steer, as a share of the gap")

    # --- setup, after_step, estimate ---
    settings = json.load(open(os.path.join(ROOT, "prereg", "settings.json")))

    class A_:
        model, fixtures, batch, budget_min = "gemma-4-31b-it", 30, 16, 60.0
        heads_layers, heads_discovery, heads_top, heads_steer_layer, heads_ko_from = "38-44", 10, 8, 20, 21
    plug = X.setup(A_(), settings, None, li)
    ctx = dict(plug["ctx"], args=A_(), panel=None)
    ok(len(ctx["pairs"]) == 30 and ctx["disc"] == [f"Y{i:02d}" for i in range(1, 11)] and len(ctx["held"]) == 20
       and len(ctx["e2_disc"]) == 20 and len(ctx["e2_held"]) == 40, "items: 30 pairs; discovery Y01-Y10, held-out 20 fixtures")
    ok(all(p["x1"]["condition"] == "D2" and p["x1e6"]["condition"] == "D8" for p in ctx["pairs"]), "pairs: x1 = D2, x1e6 = D8")
    typ, slow = ctx["estimate_min"]
    ok(30 <= typ <= 50 and slow <= 75, f"estimate {typ} min typical, {slow} if 1.5x slower")
    S_ = plug["steps"](ctx)
    ok([s[0] for s in S_] == ["heads build", "heads attention", "heads sweep 1", "heads circuit", "heads sweep 2"]
       and S_[3][3] is ctx["circuit_kw"] and S_[2][3] is S_[4][3], "steps; the circuit's kw is the shared dict")
    build = [{"fixture": f"Y{i:02d}", "skipped": "no greedy price"} for i in range(1, 29)] + \
            [{"fixture": "Y29", "n_layers": 60, "n_heads": 32, "steer_layer": 20}, {"fixture": "Y30", "n_layers": 60, "n_heads": 32, "steer_layer": 20}]
    ok("only 2" in (X.after_step("heads build", {"heads build": build}, ctx) or ""), "build gate: fewer than 3 usable fixtures stops")
    rows_obj = ctx["sweep_kw"]["rows"]
    stub_ctx = dict(ctx)
    got = {"heads build": X.stub("heads_build", ctx["pairs"], stub_ctx)}
    ok(X.after_step("heads build", got, ctx) is None and ctx["sweep_kw"]["rows"] is rows_obj and len(rows_obj) == 304,
       "after build: sweep rows set in place from the pod's geometry")
    got["heads sweep 1"] = X.stub("heads_patch", ctx["e2_disc"], ctx)
    ok(X.after_step("heads sweep 1", got, ctx) is None and len(ctx["circuit_kw"]["rows"]) == 15
       and ctx["pick"]["top"][:2] == [[41, 7], [41, 19]], f"after sweep 1: circuit rows set in place, top heads {ctx['pick']['top'][:4]}")
    ctx2 = dict(ctx)
    X.on_pod_info(ctx2, {"num_hidden_layers": 6})
    ok(ctx2["layers"]["mapped"] and ctx2["build_kw"]["steer_layer"] == ctx2["layers"]["steer"], "on_pod_info maps the layers")

    # --- full preview session ---
    import glob
    import contextlib
    import io
    pat = os.path.join(ROOT, "data", "interp", "interp-heads-*-preview")       # only ours
    before = set(glob.glob(pat))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = li.main(["--preview", "--experiment", "heads", "--fixtures", "9"])
    out = buf.getvalue()
    new = sorted(set(glob.glob(pat)) - before)
    ok(rc == 0 and len(new) == 1, "preview session runs and writes one run folder")
    ok("latch test" in out and "picked on 3 discovery fixtures" in out and "logit lens" in out, "preview report prints all parts")
    if new:
        s = json.load(open(os.path.join(new[0], "summary.json")))
        m = json.load(open(os.path.join(new[0], "run_manifest.json")))
        res = [json.loads(l) for l in open(os.path.join(new[0], "results.jsonl"))]
        ok(not s.get("analysis_error") and s["pick"]["top"] and s["interactions"] and s["lens"]["stays_above_half_from"] is not None,
           "preview summary: pick, interactions, lens")
        ok(m["plugin"]["name"] == "heads" and m["roles"] == ["seller"], "preview manifest")
        ok({r["step"] for r in res} == {"heads build", "heads attention", "heads sweep 1", "heads circuit", "heads sweep 2"},
           "preview: results for all five steps")
        import interp_heads_figures as F
        import tempfile
        tmp = os.path.join(tempfile.mkdtemp(), "fig.json")
        real, F.JSON_PATH = F.JSON_PATH, tmp
        try:
            ok(F.cards(tmp + ".none") == [], "figure: no JSON yet -> no card")
            D = F.refresh(new[0])
            c = F.cards(tmp)
            ok(len(c) == 1 and c[0].count("<svg") == 1 and c[0].count("<rect") >= 7 * 32 and len(D["heads"]) == 7 * 32
               and D["circuit"]["x1->x1e6"]["top8"] is not None, "figure: refresh -> one card with the 7 x 32 head map and circuit bars")
        finally:
            F.JSON_PATH = real
            shutil.rmtree(os.path.dirname(tmp))
        import mi_figures
        ok(any("heads · X57" in x for x in mi_figures.extra_cards()) == os.path.exists(real),
           "the site's plug-in loader finds the figure shim (a card only once the tracked JSON exists)")
        for d in new:
            shutil.rmtree(d)
    ok("heads" in li.load_plugins(), "launch_interp finds the shim")


# ======================================================================================================
# tiny model (torch; rehearsal venv)
# ======================================================================================================
def tiny():
    os.environ["SI_INTERP_DEVICE"] = "cpu"
    os.environ["SI_INTERP_REHEARSAL"] = "1"            # random weights write no prices: stand-ins
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    import launch_interp as li
    import model_io
    import tasks
    import tasks_heads as H                           # its own copy (tasks loads another for TASKS)
    import interp_heads_laptop as X
    torch.manual_seed(0)
    model, tok, info = model_io.load_text_only(TINY, dtype=torch.float32)
    name = "gemma-4-31b-it"
    ok(tasks.TASKS.get("heads_patch") is not None, "tasks loads the heads plug-in")
    settings = json.load(open(os.path.join(ROOT, "prereg", "settings.json")))
    pairs, e2 = X.build_items(li, settings, 3)
    blocks = model_io.decoder_layers(model)
    nL, nH = len(blocks), int(blocks[0].self_attn.config.num_attention_heads)
    SL = 2

    # 1. build: lens at the last layer = the model's ld; the steering difference = E3b's
    B = H.heads_build(model, tok, pairs, settings, name, model_io, steer_layer=SL)
    ok(all(not r.get("skipped") for r in B) and B[0]["n_layers"] == nL and B[0]["n_heads"] == nH and B[0]["n_post"] == 117,
       f"build: {len(B)} pairs usable (stand-in prices: {B[0]['stand_in']}), 117 post-limit tokens")
    worst = max(abs(r["lens"][c][-1] - r["ld"][c]) for r in B for c in ("x1", "x1e6"))
    ok(worst < 1e-3, f"logit lens at the last layer = the model's own ld (max |Δ| {worst:.1e})")
    vec, steer = li.build_e3_items(settings, 3)
    vec = [v for v in vec if v["kind"] == "pair"] + [v for v in vec if v["kind"] == "nonmoney"][:2]
    tasks.e3b_build(model, tok, vec, settings, name, model_io, layer=SL)
    sb = tasks._STORE["e3b"]
    wd = max(float((H._STORE["d"][f] - sb["d"][f]).abs().max()) for f in sb["d"])
    ok(wd < 1e-5, f"steering difference at layer {SL} = E3b's (max |Δ| {wd:.1e})")

    # 2. residual patch row = E2's 'post' patch (digit probabilities)
    resid_rows = [{"n": f"resid@{L}", "p": [["resid", L, -1, "post"]]} for L in range(nL)]
    mine = H.heads_patch(model, tok, e2, settings, name, model_io, rows=resid_rows, batch=4)
    ref = tasks.e2_patch(model, tok, e2, settings, name, model_io, layers=list(range(nL)), positions=("post",), batch=4)
    w2 = 0.0
    for a, b in zip(mine, ref):
        rp = {x["layer"]: x["p1"] for x in b["rows"] if x["layer"] != "clean"}
        for x in a["rows"]:
            if x["n"].startswith("resid@"):
                w2 = max(w2, max(abs(p - q) for p, q in zip(x["p1"], rp[int(x["n"][6:])])))
    ok(w2 <= 2e-4 and all(r["hooks_ok"] for r in mine), f"residual patch = E2's post patch at every layer (digit probs within {w2:.0e})")

    # 3. all heads of a layer = that layer's attention output; and it moves something
    it = e2[0]
    rows3 = [{"n": f"attn@{L}", "p": [["attn", L, -1, "post"]]} for L in range(nL)] + \
            [{"n": f"heads@{L}", "p": [["head", L, h, "post"] for h in range(nH)]} for L in range(nL)]
    r3 = {x["n"]: x for x in H.heads_patch(model, tok, [it], settings, name, model_io, rows=rows3, batch=8)[0]["rows"]}
    w3 = max(abs(r3[f"attn@{L}"]["ld"] - r3[f"heads@{L}"]["ld"]) for L in range(nL))
    moved = max(abs(r3[f"attn@{L}"]["ld"] - r3["clean"]["ld"]) for L in range(nL))
    ok(w3 < 1e-4 and moved > 1e-3, f"every head of a layer = its attention output (max |Δ ld| {w3:.1e}; largest move {moved:.3f})")

    # 4. a run patched with its own values is unchanged; mixed batches = rows alone
    selfit = dict(it, source=it["target"], item_id="self")
    rows4 = rows3[:3] + [{"n": "h", "p": [["head", 3, 1, "reply"]]}, {"n": "m", "p": [["mlp", 4, -1, "prompt"]]},
                         {"n": "r", "p": [["resid", 1, -1, "post"]]}]
    r4 = H.heads_patch(model, tok, [selfit], settings, name, model_io, rows=rows4, batch=4)[0]
    w4 = max(abs(x["ld"] - r4["ld_clean"]) for x in r4["rows"])
    ok(w4 < 1e-5, f"patching a run with its own values changes nothing (max |Δ ld| {w4:.1e})")
    mixed = {x["n"]: x["ld"] for x in H.heads_patch(model, tok, [it], settings, name, model_io, rows=rows4, batch=4)[0]["rows"]}
    alone = {}
    for spec in rows4:
        alone.update({x["n"]: x["ld"] for x in H.heads_patch(model, tok, [it], settings, name, model_io, rows=[spec], batch=4)[0]["rows"]})
    w5 = max(abs(mixed[n] - alone[n]) for n in mixed)
    ok(w5 < 1e-6, f"mixed batches = each row alone (max |Δ ld| {w5:.1e})")

    # 5. knockout exactness: two prompts differing only in the limit's digits (same token count)
    b = it["base_limit"]
    other = str(b - 11 if b > 20 else b + 11)
    sideA = it["source"] if it["source"]["condition"] == "D2" else it["target"]
    sideB = dict(sideA, limit_text=other, limit_units=int(other), messages=li.render_with_limit("seller", b, other, settings))
    ko_all = [[L, -1] for L in range(nL)]
    ko_each = [[L, h] for L in range(nL) for h in range(nH)]
    rows5 = [{"n": "ko", "k": ko_all}, {"n": "ko each", "k": ko_each}, {"n": "ko late", "k": [[L, -1] for L in range(3, nL)]}]
    impl0 = model.config._attn_implementation
    for impl in ("sdpa", "eager"):
        model.set_attn_implementation(impl)
        ra, rb = (H.heads_patch(model, tok, [dict(it, item_id=f"ko{s}", source=sd, target=sd, direction="x1e6->x1")],
                                settings, name, model_io, rows=rows5, batch=4)[0] for s, sd in (("A", sideA), ("B", sideB)))
        ga, gb = ({x["n"]: x for x in r["rows"]} for r in (ra, rb))
        same = max(abs(ga["ko"][k] - gb["ko"][k]) for k in ("ld", "lpa", "lpb", "pd"))
        diff = abs(ga["clean"]["ld"] - gb["clean"]["ld"]) + abs(ga["ko late"]["ld"] - gb["ko late"]["ld"])
        each = abs(ga["ko"]["ld"] - ga["ko each"]["ld"])
        ok(same < 1e-5 and diff > 1e-4 and each < 1e-6,
           f"{impl}: knockout at every layer -> prompts differing only in the digits read the same (|Δ| {same:.1e}); "
           f"without it they differ ({diff:.3f}); per-head = all heads ({each:.1e})")
    model.set_attn_implementation(impl0)

    # 6. steer row = E3b's bridge (v per token on all post-limit tokens, alpha 1) at the steering layer
    to3 = {"x1e6->x1": "x1+v", "x1->x1e6": "x1e6-v"}
    rs = H.heads_patch(model, tok, e2, settings, name, model_io, rows=[{"n": "steer", "s": 1.0}], batch=4)
    br = tasks.e3b_bridge(model, tok, steer, settings, name, model_io, rows=(("v_pos", "all", 1.0),), batch=4)
    bp = {(r["fixture"], r["direction"]): next(x["p1"] for x in r["rows"] if x["vector"] == "v_pos") for r in br}
    w6 = max(max(abs(p - q) for p, q in zip(next(x["p1"] for x in r["rows"] if x["n"] == "steer"), bp[(r["fixture"], to3[r["direction"]])]))
             for r in rs)
    ok(w6 <= 2e-4 and all(r["loo"] for r in rs), f"steer row = E3b's bridge at layer {SL}, leave-one-out (digit probs within {w6:.0e})")

    # 7. attention step
    impl0 = model.config._attn_implementation
    A = H.heads_attn(model, tok, pairs[:2], settings, name, model_io)
    vals = [v for r in A for c in ("x1", "x1e6") for L in r["mass"][c] for q in L for v in q]
    ok(model.config._attn_implementation == impl0 and len(A[0]["mass"]["x1"]) == nL and len(A[0]["mass"]["x1"][0]) == 3
       and all(0 <= v <= 1.0001 for v in vals) and max(vals) > 0 and A[0]["n_digits"] == {"x1": 2, "x1e6": 8},
       f"attention: masses in [0, 1] for {nL} layers x 3 query sets x {nH} heads; attention setting restored ({impl0})")


def main(argv):
    if "--torch" in argv:
        tiny()
    else:
        laptop()
        if os.path.exists(VENV_PY) and os.path.exists(os.path.join(TINY, "config.json")):
            print("--- tiny-model checks (rehearsal venv, fp32, CPU) ---")
            r = subprocess.run([VENV_PY, os.path.abspath(__file__), "--torch"], capture_output=True, text=True, timeout=1800)
            for l in [l for l in r.stdout.splitlines() if l.startswith(("ok", "FAIL", "skip"))]:
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
