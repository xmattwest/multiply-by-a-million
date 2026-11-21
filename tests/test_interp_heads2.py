#!/usr/bin/env python3
"""test_interp_heads2.py: checks for the heads2 plug-in (scripts/interp/tasks_heads2.py on the pod;
scripts/interp_heads2_laptop.py behind the shim scripts/interp/exp_heads2.py).

    python3 tests/test_interp_heads2.py          laptop checks (stdlib), then the tiny-model checks in the
                                                 rehearsal venv if it exists (else they fail, loudly)
    /tmp/si-interp-venv/bin/python tests/test_interp_heads2.py --torch    the tiny-model checks only

Laptop: geometry and row counts, region offsets, position picks and their random control, the summaries on synthetic
records, the after_step hand-overs (rows set in place), the time estimate, and a full --preview session.
Tiny model (random-weight Gemma 4 of the 31B's class, fp32 on the CPU):
  - sources: each head's output is rebuilt exactly from its attention weights and values; the split of its
    x1e6 - x1 output difference by position sums to the whole; query masses sum to 1; the all-heads scan agrees with
    the per-head difference size; the attention setting is restored;
  - path: 'T feed none' = clean; 'S feed none' = 'direct'; 'direct' = X57's joint head patch; 'resid' = X57's
    residual patch; with all receivers in one layer, 'back' at the layer below = 'feed'; mixed batches = rows
    alone; a missing per-fixture position list leaves the row out; a back layer at or above the last receiver
    layer is refused;
  - roles: building buyers does not disturb the sellers' store, and the buyer patch runs on its own store.
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


class A_:
    model, fixtures, batch, budget_min = "gemma-4-31b-it", 30, 16, 60.0
    heads2_recv, heads2_contrast = "41.29,39.28,42.9,41.27,41.10,41.11,43.5,39.22", "43.5,41.26,54.23,42.9"
    heads2_back, heads2_after, heads2_top, heads2_discovery = "20,24,28,32,36,38", 2, 8, 10
    heads2_parts, heads2_x57_run = "sources,path,buyers", "data/interp/none"


# ======================================================================================================
# laptop (stdlib)
# ======================================================================================================
def synth_sources(fixture, n_post=8, n_prompt=4, heads=("41.29", "39.28")):
    """A heads2_sources record with known numbers (n_post 8, n_prompt 4: after = 0-1, rest = 2-3, reply = 4-7)."""
    hs = {}
    for i, k in enumerate(heads):
        post = [0.0] * n_post
        post[3 if i == 0 else 4] = 0.3
        d = [0.0] * n_post
        d[1] = 0.5
        d[5] = 0.1
        per = {c: {"mass_pre": {"bos": 0.4, "sum": 0.5, "last": [0.05, 0.05], "top": [[0, 0.4]]},
                   "mass_digits": [0.1, 0.1], "mass_post": post, "vw_pre": {"bos": 0.1, "sum": 0.2, "last": [0.0, 0.0], "top": []},
                   "vw_digits": [0.2, 0.2], "vw_post": post, "q_regions": [[0.4, 0.1, 0.2, 0.3, 0.0]] * 2, "out_norm": 1.0,
                   "recon_err": 0.0} for c in ("x1", "x1e6")}
        per["diff"] = {"den": 4.0 if i == 0 else 1.0, "pre": {"bos": 0.0, "sum": 0.1, "last": [0.0, 0.1], "top": []},
                       "digits": 0.3, "post": d, "sum_err": 0.0, "dnorm": 2.0}
        hs[k] = per
    return {"item_id": f"{fixture}-S-pair", "fixture": fixture, "n_post": n_post, "n_prompt": n_prompt,
            "post_tokens": [f"t{o}" for o in range(n_post)], "heads": hs,
            "scan": {"41": {"d_reply": [1.0, 2.0, 3.0], "d_prompt": [0.5, 0.2, 0.1], "n_x1": [1.0] * 3, "n_x1e6": [1.0] * 3}}}


def laptop():
    import launch_interp as li
    import interp_heads2_laptop as X2
    X2._LI[:] = [li]
    opts = {"recv": X2.parse_heads(A_.heads2_recv), "contrast": X2.parse_heads(A_.heads2_contrast),
            "back": [20, 24, 28, 32, 36, 38], "after": 2, "top": 8}
    G = X2.geometry(60, 32, opts)
    ok(G["recv"][:2] == [(41, 29), (39, 28)] and len(G["recv"]) == 8 and G["feed"] == [38, 40, 41, 42]
       and G["back"] == [20, 24, 28, 32, 36, 38] and not G["mapped"], f"31B geometry: 8 receivers, feeding layers {G['feed']}")
    ok(len(G["sources"]) == 10 and (41, 26) in G["sources"] and (54, 23) in G["sources"], "sources: top 8 + 2 new contrast heads (43.5, 42.9 are in the top 8)")
    T6 = X2.geometry(6, 2, opts)
    ok(T6["mapped"] and all(0 <= L < 6 and 0 <= h < 2 for L, h in T6["recv"] + T6["sources"]) and T6["back"]
       and all(L < max(x for x, _ in T6["recv"]) for L in T6["back"]), f"6-layer, 2-head mapping stays in range ({T6['recv']}, back {T6['back']})")
    R = X2.regions(117, 106, 2)
    ok(R["after"] == [0, 1] and R["rest"] == list(range(2, 106)) and R["reply"] == list(range(106, 117)) and len(R["post"]) == 117,
       "regions: after = 2 tokens, rest = the other 104 prompt tokens, reply = the 11 opening tokens")
    picks = {"Y01": {"attn": [1, 2], "diff": [3, 110], "rand_a": [4, 111], "rand_b": [5, 112]}}
    rows = X2.path_rows(G, 117, 106, 2, picks)
    names = [r["n"] for r in rows]
    ok(len(names) == len(set(names)) and len(rows) == 10 + 6 + 24 + 5 + 8, f"path rows: {len(rows)} (10 + 6 feed, 24 back, 5 direct, 8 resid), unique")
    rr = {r["n"]: r for r in rows}
    ok(rr["T feed reply"]["P"] == list(range(106, 117)) and rr["S feed post"]["seq"] == "S" and rr["T back20 post"]["L"] == 20
       and rr["T feed diff"]["P"] == {"Y01": [3, 110]} and rr["direct L41"]["use"] == [[41, 29], [41, 27], [41, 10], [41, 11]],
       "path rows: regions, per-fixture picks, layer-41 receivers")
    ok(len(X2.path_rows(G, 117, 106, 2)) == len(rows) - 5, "without picks: the 5 pick rows are left out")
    seller8 = G["recv"]
    bs = X2.buyer_sweep_rows(G, seller8)
    ok(len(bs) == 11 + 14 + 224 + 1 and len({r["n"] for r in bs}) == len(bs) and bs[-1]["n"] == "seller8"
       and len(bs[-1]["p"]) == 8, "buyer sweep: 11 residual + 14 attention/MLP + 224 heads + the sellers' 8 jointly")
    top = [(41, 3), (41, 29), (42, 1), (40, 5), (38, 0), (39, 1), (43, 2), (44, 4)]
    bc = {r["n"]: r for r in X2.buyer_circuit_rows(top, [(38, 9)] * 8, seller8, G)}
    ok({"top1", "top4", "top8", "top8 reply", "seller8", "random8", "shared1", "seller-only7", "other heads"} <= set(bc)
       and len(bc["other heads"]["p"]) == 224 - 8, "buyer circuit: own top, seller 8, overlap split, controls")
    ok("shared1" not in {r["n"] for r in X2.buyer_circuit_rows([(38, 0)], [], seller8, G)}, "no overlap -> no shared row")

    # --- picks and summaries on synthetic records ---
    recs = [synth_sources(f) for f in ("Y01", "Y02", "Y03")]
    pk = X2.pick_positions(recs, [(41, 29), (39, 28)], 8, 4, 2, k=2)
    p1 = pk["Y01"]
    ok(p1["attn"] == [3, 4] and p1["diff"] == [1, 5], f"picks: most attended {p1['attn']}, most different {p1['diff']}")
    reg = lambda o: "after" if o < 2 else ("rest" if o < 4 else "reply")     # noqa: E731
    ok(all(sorted(map(reg, p1[k])) == sorted(map(reg, p1["diff"])) and not set(p1[k]) & (set(p1["attn"]) | set(p1["diff"]))
           for k in ("rand_a", "rand_b")) and pk == X2.pick_positions(recs, [(41, 29), (39, 28)], 8, 4, 2, k=2),
       f"random control: matched by region, disjoint from the picks, reproducible ({p1['rand_a']}, {p1['rand_b']})")
    S = X2.sources_summary(recs, 2, {"h41.0": 0.1, "h41.1": 0.3, "h41.2": 0.5})
    c = S["heads"]["41.29"]
    ok(abs(c["mass_x1"]["bos"] - 0.4) < 1e-9 and abs(c["mass_x1"]["before"] - 0.1) < 1e-9 and abs(c["mass_x1"]["digits"] - 0.2) < 1e-9
       and abs(c["mass_x1"]["rest"] - 0.3) < 1e-9 and c["mass_x1"]["reply"] == 0 and c["top_mass"][0][:3] == [3, "t3", "rest"], "sources summary: region sums")
    ok(abs(c["diff"]["after"]["share"] - 0.5) < 1e-9 and abs(c["diff"]["reply"]["share"] - 0.1) < 1e-9
       and abs(c["diff"]["digits"]["share"] - 0.3) < 1e-9 and c["top_diff"][0][:3] == [1, "t1", "after"],
       "sources summary: difference split by region, top position labelled")
    ok(abs(S["scan_vs_x57_spearman"]["d_reply"] - 1) < 1e-9 and abs(S["scan_vs_x57_spearman"]["d_prompt"] + 1) < 1e-9,
       "scan vs X57 importance: Spearman on the heads both have")

    def prec(f, d, clean, src, rows):
        return {"fixture": f, "direction": d, "ld_clean": clean, "ld_source": src,
                "rows": [{"n": n, "ld": v, "lpa": -0.5, "lpb": -1.0} for n, v in rows.items()] + [{"n": "clean", "ld": clean}]}
    pr = [prec(f, "x1->x1e6", 4, -4, {"direct": 4 - 0.4 * 8, "T feed post": 4 - 0.1 * 8}) for f in ("Y01", "Y02", "Y03")]
    od = X2.of_direct(pr)
    ok(abs(od["x1->x1e6"]["T feed post"]["share"] - 0.25) < 1e-9 and abs(od["x1->x1e6"]["direct"]["share"] - 1) < 1e-9,
       "of_direct: a row's movement over the direct row's")
    cp = X2.cand_prob(pr)["x1->x1e6"]["direct"]
    ok(abs(cp - (0.6065306597 + 0.3678794412)) < 1e-6, f"candidate probability = p(a) + p(b) ({cp:.3f})")

    # --- setup, after_step, estimate ---
    settings = json.load(open(os.path.join(ROOT, "prereg", "settings.json")))
    plug = X2.setup(A_(), settings, None, li)
    ctx = dict(plug["ctx"], args=A_(), panel=None)
    ok(len(ctx["sell_pairs"]) == 30 and len(ctx["path_items"]) == 60 and len(ctx["buy_pairs"]) == 30
       and len(ctx["buy_disc"]) == 20 and len(ctx["buy_held"]) == 40 and ctx["buy_pairs"][0]["role"] == "buyer"
       and ctx["buy_pairs"][0]["x1"]["condition"] == "D2" and ctx["buy_pairs"][0]["item_id"] == "Y01-B-pair",
       "items: 30 seller pairs, 60 path items; buyers 30 pairs, 20 discovery + 40 held-out items")
    typ, slow = ctx["estimate_min"]
    ok(12 <= typ <= 35 and slow <= 50, f"estimate {typ} min typical, {slow} if 1.5x slower")
    St = plug["steps"](ctx)
    ok([s[0] for s in St] == ["heads2 seller build", "heads2 sources", "heads2 path", "heads2 buyer build", "heads2 buyer sweep 1",
                              "heads2 buyer circuit", "heads2 buyer sweep 2"] and St[2][3] is ctx["path_kw"] and St[4][3] is St[6][3],
       "steps in order; kw dicts shared")
    rows_obj, circ_obj = ctx["path_kw"]["rows"], ctx["bcirc_kw"]["rows"]
    got = {"heads2 seller build": X2.stub("heads2_build", ctx["sell_pairs"], ctx)}
    ok(X2.after_step("heads2 seller build", got, ctx) is None and ctx["path_kw"]["rows"] is rows_obj and len(rows_obj) == 48,
       f"after seller build: path rows (no picks yet) set in place ({len(rows_obj)})")
    got["heads2 sources"] = X2.stub("heads2_sources", ctx["src_pairs"], ctx)
    ok(X2.after_step("heads2 sources", got, ctx) is None and len(rows_obj) == 53 and len(ctx["src_pick"]) == 30,
       "after sources: picks for 30 fixtures, 5 pick rows added in place")
    got["heads2 buyer sweep 1"] = X2.stub("heads2_patch", ctx["buy_disc"], ctx)
    ok(X2.after_step("heads2 buyer sweep 1", got, ctx) is None and ctx["bcirc_kw"]["rows"] is circ_obj and len(circ_obj) >= 11
       and ctx["pick"]["top"][0] == [41, 3], f"after buyer sweep 1: circuit rows in place, top {ctx['pick']['top'][:3]}")
    bad = [{"fixture": f"Y{i:02d}", "skipped": "the two prices are equal after rescaling"} for i in range(1, 29)] + \
          [{"fixture": "Y29", "n_layers": 60, "n_heads": 32, "n_post": 117, "n_prompt": 106}] * 2
    ctx2 = dict(ctx, buy_disc=list(ctx["buy_disc"]), buy_held=list(ctx["buy_held"]))
    X2.after_step("heads2 buyer build", {"heads2 buyer build": bad}, ctx2)
    ok(ctx2["buy_disc"] == [] and ctx2["buy_held"] == [], "buyer build gate: fewer than 3 usable fixtures skips the buyer sweeps")
    a2 = A_()
    a2.heads2_parts = "buyers"
    c3 = X2.setup(a2, settings, None, li)["ctx"]
    ok(c3["sell_pairs"] == [] and c3["src_pairs"] == [] and c3["path_items"] == [] and len(c3["buy_pairs"]) == 30,
       "--heads2-parts buyers: seller steps empty")

    # --- full preview session ---
    import glob
    import contextlib
    import io
    pat = os.path.join(ROOT, "data", "interp", "interp-heads2-*-preview")       # only ours
    before = set(glob.glob(pat))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = li.main(["--preview", "--experiment", "heads2", "--fixtures", "9"])
    out = buf.getvalue()
    new = sorted(set(glob.glob(pat)) - before)
    ok(rc == 0 and len(new) == 1, "preview session runs and writes one run folder")
    ok("SOURCES" in out and "PATH PATCHING" in out and "BUYERS" in out and "seller-only" in out, "preview report prints all parts")
    if new:
        s = json.load(open(os.path.join(new[0], "summary.json")))
        m = json.load(open(os.path.join(new[0], "run_manifest.json")))
        res = [json.loads(l) for l in open(os.path.join(new[0], "results.jsonl"))]
        ok(not s.get("analysis_error") and s["path"]["shares"] and s["buyers"]["circuit"] and s["sources"]["heads"],
           "preview summary: sources, path, buyers")
        ok(m["plugin"]["name"] == "heads2" and m["roles"] == ["seller", "buyer"], "preview manifest")
        ok({r["step"] for r in res} == {s_[0] for s_ in St}, "preview: results for all seven steps")
        import interp_heads2_figures as F
        import tempfile
        tmp = os.path.join(tempfile.mkdtemp(), "fig.json")
        real, F.JSON_PATH = F.JSON_PATH, tmp
        try:
            ok(F.cards(tmp + ".none") == [], "figure: no JSON yet -> no card")
            D = F.refresh(new[0])
            c = F.cards(tmp)
            ok(len(c) == 1 and c[0].count("<svg") == 1 and len(D["sources"]) == 10 and D["path"]["x1->x1e6"]["direct"]
               and D["buyers"]["x1e6->x1"]["seller8"], "figure: refresh -> one card (sources, path, buyers)")
        finally:
            F.JSON_PATH = real
            shutil.rmtree(os.path.dirname(tmp))
        import mi_figures
        ok(any("heads2 · X60" in x for x in mi_figures.extra_cards()) == os.path.exists(real),
           "the site's plug-in loader finds the figure shim (a card only once the tracked JSON exists)")
        for d in new:
            shutil.rmtree(d)
    ok("heads2" in li.load_plugins(), "launch_interp finds the shim")


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
    import tasks_heads2 as H2                         # its own copy (tasks loads another for TASKS)
    import interp_heads2_laptop as X2
    H = H2.H
    torch.manual_seed(0)
    model, tok, info = model_io.load_text_only(TINY, dtype=torch.float32)
    name = "gemma-4-31b-it"
    ok(all(tasks.TASKS.get(t) is not None for t in ("heads2_build", "heads2_patch", "heads2_sources", "heads2_path"))
       and tasks.TASKS.get("heads_patch") is not None, "tasks loads the heads2 plug-in next to heads")
    settings = json.load(open(os.path.join(ROOT, "prereg", "settings.json")))
    sp, se2 = X2.build_items(li, settings, 3, "seller")
    bp, be2 = X2.build_items(li, settings, 3, "buyer")
    blocks = model_io.decoder_layers(model)
    nL, nH = len(blocks), int(blocks[0].self_attn.config.num_attention_heads)
    B = H2.heads2_build(model, tok, sp, settings, name, model_io, role="seller", steer_layer=2)
    ok(all(not r.get("skipped") for r in B) and set(H2._STORES["seller"]["pairs"]) == {"Y01", "Y02", "Y03"},
       "seller build into the seller store")

    # 1. sources
    heads = [[3, 1], [3, 0], [5, 1]]                  # two sliding-layer heads and one global (k = v) head
    impl0 = model.config._attn_implementation
    Sx = H2.heads2_sources(model, tok, sp, settings, name, model_io, heads=heads, scan=[3, 5])
    hs = [h for r in Sx for h in r["heads"].values()]
    rec = max(h[c]["recon_err"] for h in hs for c in ("x1", "x1e6"))
    se = max(h["diff"]["sum_err"] for h in hs)
    ok(rec < 1e-5 and se < 1e-5, f"sources: head output rebuilt from weights x values (rel err {rec:.1e}); the split sums to the whole ({se:.1e})")
    qs = [sum(q) for h in hs for c in ("x1", "x1e6") for q in h[c]["q_regions"]]
    tot = [h[c]["mass_pre"]["sum"] + sum(h[c]["mass_digits"]) + sum(h[c]["mass_post"]) for h in hs for c in ("x1", "x1e6")]
    ok(all(abs(x - 1) < 1e-3 for x in qs + tot) and len(Sx[0]["post_tokens"]) == Sx[0]["n_post"] == 117
       and len(hs[0]["x1e6"]["mass_digits"]) == Sx[0]["n_digits"]["x1e6"], "sources: each query's mass sums to 1 over the regions")
    sc = max(abs(r["scan"]["3"]["d_reply"][1] - r["heads"]["3.1"]["diff"]["dnorm"]) / max(1e-9, r["heads"]["3.1"]["diff"]["dnorm"])
             for r in Sx)
    ok(sc < 1e-3 and model.config._attn_implementation == impl0 and len(Sx[0]["scan"]["5"]["d_reply"]) == nH,
       f"sources: the all-heads scan matches the per-head difference size ({sc:.1e}); attention setting restored")

    # 2. path
    G = X2.geometry(nL, nH, {"recv": X2.parse_heads(A_.heads2_recv), "contrast": [], "back": [20, 24, 28, 32, 36, 38], "after": 2})
    recv = [list(x) for x in G["recv"]]
    picks = X2.pick_positions(Sx, G["recv"], 117, 106, 2)
    rows = X2.path_rows(G, 117, 106, 2, picks)
    P1 = H2.heads2_path(model, tok, se2, settings, name, model_io, recv=recv, rows=rows, batch=8)
    ok(all(r["hooks_ok"] for r in P1) and len(P1) == 6, f"path: {len(P1)} items, hooks fired as planned ({len(rows)} rows)")
    g = [{x["n"]: x["ld"] for x in r["rows"]} for r in P1]
    w1 = max(abs(d["T feed none"] - d["clean"]) for d in g)
    w2 = max(abs(d["S feed none"] - d["direct"]) for d in g)
    mv = max(abs(d["direct"] - d["clean"]) for d in g)
    ok(w1 < 1e-5 and w2 < 1e-5 and mv > 1e-3, f"'T feed none' = clean ({w1:.1e}); 'S feed none' = direct ({w2:.1e}); direct moves ({mv:.3f})")
    x57 = [{"n": "topk", "p": [["head", L, h, "post"] for L, h in recv]}] + \
          [{"n": f"resid@{L}", "p": [["resid", L, -1, "post"]]} for L in G["resid_ref"]]
    H2._use("seller")
    ref = H.heads_patch(model, tok, se2, settings, name, model_io, rows=x57, batch=8)
    w3 = max(abs(d["direct"] - {x["n"]: x["ld"] for x in r["rows"]}["topk"]) for d, r in zip(g, ref))
    w4 = max(abs(d[n] - x["ld"]) for d, r in zip(g, ref) for x in r["rows"] if (n := x["n"]).startswith("resid@"))
    ok(w3 < 1e-5 and w4 < 1e-5, f"'direct' = X57's joint head patch ({w3:.1e}); 'resid' = X57's residual patch ({w4:.1e})")
    one = [[4, 0], [4, 1]]
    rows1 = [{"n": "feed", "kind": "path", "seq": "T", "mode": "feed", "P": list(range(117))},
             {"n": "back", "kind": "path", "seq": "T", "mode": "back", "L": 3, "P": list(range(117))},
             {"n": "back reply", "kind": "path", "seq": "T", "mode": "back", "L": 3, "P": list(range(106, 117))},
             {"n": "feed reply", "kind": "path", "seq": "T", "mode": "feed", "P": list(range(106, 117))},
             {"n": "S feed", "kind": "path", "seq": "S", "mode": "feed", "P": list(range(50))},
             {"n": "S back", "kind": "path", "seq": "S", "mode": "back", "L": 3, "P": list(range(50))}]
    P2 = H2.heads2_path(model, tok, se2, settings, name, model_io, recv=one, rows=rows1, batch=4)
    g2 = [{x["n"]: x["ld"] for x in r["rows"]} for r in P2]
    w5 = max(max(abs(d["feed"] - d["back"]), abs(d["feed reply"] - d["back reply"]), abs(d["S feed"] - d["S back"])) for d in g2)
    ok(w5 < 1e-5 and max(abs(d["feed"] - d["clean"]) for d in g2) > 1e-4,
       f"receivers in one layer: 'back' at the layer below = 'feed' (max |Δ ld| {w5:.1e})")
    alone = {}
    for spec in rows1:
        r1 = H2.heads2_path(model, tok, se2[:1], settings, name, model_io, recv=one, rows=[spec], batch=4)[0]
        alone[spec["n"]] = {x["n"]: x["ld"] for x in r1["rows"]}[spec["n"]]
    w6 = max(abs(alone[n] - g2[0][n]) for n in alone)
    ok(w6 < 1e-6, f"mixed batches = each row alone (max |Δ ld| {w6:.1e})")
    rmiss = H2.heads2_path(model, tok, se2[:1], settings, name, model_io, recv=one, batch=4,
                           rows=[{"n": "x", "kind": "path", "seq": "T", "mode": "feed", "P": {"Y99": [1]}}, {"n": "direct", "kind": "direct"}])[0]
    ok([x["n"] for x in rmiss["rows"]] == ["direct", "clean"], "a per-fixture position list without this fixture leaves the row out")
    try:
        H2.heads2_path(model, tok, se2[:1], settings, name, model_io, recv=one, rows=[dict(rows1[1], L=4)], batch=4)
        refused = False
    except ValueError:
        refused = True
    ok(refused, "a back layer at or above the last receiver layer is refused")
    er = tasks.TASKS["heads2_path"](model, tok, se2[:2], settings, name, model_io, recv=one, rows=[dict(rows1[1], L=4)], batch=4)
    ok(len(er) == 2 and all("back layer" in r.get("error", "") and r["fixture"] == "Y01" for r in er),
       "the pod entry turns a failing item into an error record (the session goes on)")

    # 3. roles
    BB = H2.heads2_build(model, tok, bp, settings, name, model_io, role="buyer", steer_layer=2)
    P3 = H2.heads2_path(model, tok, se2, settings, name, model_io, recv=one, rows=rows1, batch=4)
    w7 = max(abs(a["ld"] - b["ld"]) for r1, r2 in zip(P2, P3) for a, b in zip(r1["rows"], r2["rows"]))
    ok(w7 == 0 and H2._STORES["buyer"]["pairs"] and H2._STORES["seller"]["pairs"]["Y01"] is not H2._STORES["buyer"]["pairs"]["Y01"],
       f"buyer build ({sum(1 for r in BB if not r.get('skipped'))} usable) leaves the seller store alone (path rows identical)")
    G6 = X2.geometry(nL, nH, {"recv": X2.parse_heads(A_.heads2_recv), "contrast": [], "back": [20], "after": 2})
    bsw = X2.buyer_sweep_rows(G6, G6["recv"])
    BP = H2.heads2_patch(model, tok, be2, settings, name, model_io, role="buyer", rows=bsw, batch=8)
    ok(all(r["role"] == "buyer" and r["hooks_ok"] and not r.get("skipped") for r in BP) and len(BP[0]["rows"]) == len(bsw) + 1,
       f"buyer patch runs on the buyer store ({len(BP)} items x {len(bsw)} rows)")


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
