#!/usr/bin/env python3
"""test_interp_e4.py: checks for the E4 plug-in (scripts/interp/tasks_e4.py on the pod; scripts/interp_e4_laptop.py
behind the shim scripts/interp/exp_e4.py).

    python3 tests/test_interp_e4.py            laptop checks (stdlib), then the tiny-model checks in the rehearsal
                                               venv if it exists (else they fail, loudly)
    /tmp/si-interp-venv/bin/python tests/test_interp_e4.py --torch      the tiny-model checks only

Laptop: the range prompts (Q17's committed sentence, range and placement; N = the digits probe's D2/D8 text; the
fixtures' ceilings inside the range), layer sets and row specs, the gate on synthetic openings (pass, each failure,
the rehearsal bypass), shares on synthetic records, the cure window and head pick, after_step hand-overs, the time
estimate, and a full --preview session.
Tiny model (random-weight Gemma 4 of the 31B's class, fp32 on the CPU):
  - the four prompts align on A = H + S + reply (117 tokens), H's states are the same with and without the range;
  - the stored plain differences equal E3b's/E3c's at the same layer; ld from the build = the patch step's clean rows;
  - 'set' of the other plain run's residual = E2's post patch; 'add' of the plain difference = 'set' on the plain run;
  - a run patched with its own values is unchanged (every kind and segment); all heads of a layer = its attention;
  - the cure at the last layer on S + reply gives the swapped-in run's first-digit probabilities, both ways;
  - the 'vec' row = E3b's bridge (v per token, alpha 1); mixed batches = rows alone;
  - real-opening steering (greedy) = E3c's on the plain prompts; unsteered range openings = tasks.generate;
  - the attention step's masses lie in [0, 1] and the attention setting is restored.
"""
import json
import os
import re
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
    import interp_e4_laptop as X
    X._LI[:] = [li]
    settings = json.load(open(os.path.join(ROOT, "prereg", "settings.json")))

    # --- prompts ---
    probe = subprocess.run(["git", "show", "HEAD:scripts/run_game_probe.py"], cwd=ROOT, capture_output=True, text=True).stdout
    m = re.search(r'INFO_RANGES = \{"F": \((\d+), (\d+)\), "C": \((\d+), (\d+)\)\}', probe)
    ok(m and (int(m.group(3)) // 100, int(m.group(4)) // 100) == X.RANGE_C, f"range = Q17's committed ceiling range / 100 {X.RANGE_C}")
    ng = subprocess.run(["git", "show", "HEAD:scripts/newgame.py"], cwd=ROOT, capture_output=True, text=True).stdout
    ok('"GOAL": goal + ("\\n" + extra if extra else "")' in ng and 'return f"{d}" + (f".{c:02d}" if c else "") + " dollars"' in ng,
       "committed newgame: the information line follows the goal on its own line; raw amounts are '<digits> dollars'")
    committed = subprocess.run(["git", "show", "HEAD:config/explore/info/info.txt"], cwd=ROOT, capture_output=True, text=True).stdout
    ok(X.EXPECT_SENTENCE in committed, "the sentence is the committed [info_p_seller] fragment")
    fx = X.build_items(li, settings, 30)
    ok(len(fx) == 30 and all(X.RANGE_C[0] <= it["ceiling"] <= X.RANGE_C[1] for it in fx),
       f"30 fixtures; every ceiling inside {X.RANGE_C} ({min(it['ceiling'] for it in fx)}-{max(it['ceiling'] for it in fx)})")
    it = fx[2]
    b = it["base_limit"]
    P1, P6, N1, N6 = (it["prompts"][k] for k in ("P1", "P6", "N1", "N6"))
    ok(N1["messages"] == li.render_item("seller", b, "D2", settings) and N6["messages"] == li.render_item("seller", b, "D8", settings),
       "N1/N6 = the digits probe's D2/D8 prompts")
    s1 = "You also know that the buyer's ceiling was drawn at random from the range 105 dollars to 125 dollars. The buyer does not know that you know this."
    s6 = s1.replace("105 dollars", "105000000 dollars").replace("125 dollars", "125000000 dollars")
    u1, u6 = P1["messages"][1]["content"], P6["messages"][1]["content"]
    ok(u1.count(s1) == 1 and u6.count(s6) == 1 and u1.replace(s1 + "\n", "") == N1["messages"][1]["content"]
       and u6.replace(s6 + "\n", "") == N6["messages"][1]["content"] and P1["messages"][0] == N1["messages"][0],
       "P = N plus the sentence (and nothing else); same system message")
    ok("no deal.\n" + s1 + "\n\nNo messages" in u1, "the sentence is on the line right after the goal")
    ok(P1["range"] == {"lo": "105", "hi": "125"} and P6["range"] == {"lo": "105000000", "hi": "125000000"} and N1["range"] is None
       and P6["limit_text"] == str(b * 10 ** 6), "range digits and limits per prompt")

    # --- layer sets and rows ---
    L = X.layer_sets(60)
    ok(L["band"] == list(range(38, 49)) and L["steer"] == [42, 43, 44] and len(L["cure"]) == 36 and L["keep"] == [42, 43, 44]
       and not L["mapped"], "31B layer sets: band 38-48, steering 42-44, 36 cure layers")
    T6 = X.layer_sets(6)
    ok(T6["mapped"] and all(0 <= x < 6 for k in ("band", "steer", "cure", "heads", "keep") for x in T6[k]), f"6-layer mapping in range")
    R = X.patch_rows(L, X.TOP8, X.RAND8)
    nr = {t: len(v) for t, v in R.items()}
    ok(nr == {"N1": 25, "N6": 25, "P1": 37, "P6": 37} and all(len({r["n"] for r in v}) == len(v) for v in R.values()),
       f"patch rows per target {nr}, names unique")
    ok(all(op[5] == X.OTHER[t] for t in R for r in R[t] for op in r["ops"] if op[0] == "set")
       and all(op[6:] == (["N6", "N1"] if X.SCALE[t] == 0 else ["N1", "N6"]) for t in R for r in R[t] for op in r["ops"] if op[0] == "add")
       and all((op[3] > 0) == (X.SCALE[t] == 0) for t in R for r in R[t] for op in r["ops"] if op[0] == "vec"),
       "'set' takes the other scale's run of the same kind; 'add' is the plain difference toward the other scale; signs")
    top = next(r for r in R["P1"] if r["n"] == "top8 add")
    ok([(op[2], op[3]) for op in top["ops"]] == list(X.TOP8), "top8 = X57's heads")
    C = X.cure_rows(L)
    ok({t: len(v) for t, v in C.items()} == {"N6": 44, "P6": 44} and all(r["ops"][0][5] == ("P6" if t == "N6" else "N6") for t in C for r in C[t]),
       "cure rows: 36 layers + 8 segment rows, P6 into N6 and N6 into P6")
    H = X.cure_head_rows(list(range(30, 37)), 32)
    ok(len(H["N6"]) == 231 and H["N6"][0]["ops"][0][:5] == ["set", "head", 30, 0, "Srep"], "cure head rows: 224 heads + 7 attention rows")

    # --- gate on synthetic openings ---
    def gate_recs(p1, p6, n=4, fixtures=5, bad=0, reh=False):
        out = []
        for i in range(fixtures):
            for t, p in (("P1", p1), ("P6", p6)):
                rows = [{"vector": "none", "layer": None, "alpha": 0.0, "price": (p[j % len(p)]) * 10 ** X.SCALE[t],
                         "errors": ["BAD"] if j < bad else [], "action": "OFFER", "aligned": True} for j in range(n)]
                out.append({"target": t, "fixture": f"Y{i:02d}", "base_limit": 90, "rows": rows, "rehearsal": reh})
        return out
    g = X.summarize_gate(gate_recs([120], [120]), li)
    ok(g["pass"] and g["cells"]["P1"]["modal"] == 120 and g["cells"]["P6"]["at_120"] == 1.0 and abs(g["shift"]) < 1e-12,
       "gate passes when both scales open at 120")
    g = X.summarize_gate(gate_recs([120], [100]), li)
    ok(not g["pass"] and "shift" in g["why"] and "differ" in g["why"], f"gate fails on a shift / different openings ({g['why']})")
    g = X.summarize_gate(gate_recs([120], [120], bad=1), li)
    ok(not g["pass"] and "valid" in g["why"], "gate fails below 95% valid")
    g = X.summarize_gate(gate_recs([120, 120, 125, 115], [120, 125, 115, 120]), li)
    ok(not g["pass"] and "most common opening only 50%" in g["why"], f"gate fails when the common opening is < 70% ({g['why']})")
    ctx0 = {"li": li, "panel": None}
    ok("gate failed" in (X.after_step("e4 gate", {"e4 gate": gate_recs([120], [95])}, ctx0) or ""), "after_step stops on a failed gate")
    ok(X.after_step("e4 gate", {"e4 gate": gate_recs([120], [95], reh=True)}, ctx0) is None, "rehearsal: the gate is reported, not applied")

    # --- shares ---
    def prec(f, tl, rows):
        return {"fixture": f, "targets": {t: {"ld_clean": c, "rows": [{"n": n, "ld": v, "lpa": -1.0, "lpb": -2.0} for n, v in rows.get(t, {}).items()]
                                              + [{"n": "clean", "ld": c, "lpa": -1.0, "lpb": -2.0}]} for t, c in tl.items()}}
    recs = [prec(f"Y{i}", {"N1": -10.0 * g_, "N6": 6.0 * g_, "P1": -12.0 * g_, "P6": -8.0 * g_},
                 {"N1": {"own@42": -10.0 * g_ + 0.8 * 16 * g_}, "P1": {"plain@42": -12.0 * g_ + 0.2 * 16 * g_},
                  "N6": {"own@42": 6.0 * g_ - 0.9 * 16 * g_}, "P6": {"plain@42": -8.0 * g_ - 0.1 * 16 * g_}})
            for i, g_ in enumerate((1.0, 1.1, 0.9, 1.2))]
    S = X.ratio_table(recs, "plain")
    ok(abs(S["N1"]["own@42"]["share"] - 0.8) < 1e-9 and abs(S["P1"]["plain@42"]["share"] - 0.2) < 1e-9
       and abs(S["N6"]["own@42"]["share"] - 0.9) < 1e-9 and abs(S["P6"]["plain@42"]["share"] - 0.1) < 1e-9,
       "plain shares: moved / (G_N signed by the target's scale)")
    c = S["N1"]["own@42"]
    ok(c["ci"] and c["ci"][0] <= 0.8 <= c["ci"][1] and abs(c["pk"] - (2.718281828 ** -1 + 2.718281828 ** -2)) < 1e-6, "CI and p(a)+p(b)")
    cl = X.clean_table(recs)
    ok(abs(cl["G_N"] - 16 * 1.05) < 1e-9 and abs(cl["G_P_share"] - 0.25) < 1e-9, "clean table: G_N and G_P / G_N")
    crec = [prec(f"Y{i}", {"N6": 6.0, "P6": -8.0}, {"N6": {"cure@40": 6.0 - 0.7 * 14}, "P6": {"cure@40": -8.0 + 0.3 * 14}}) for i in range(3)]
    Cc = X.ratio_table(crec, "cure")
    ok(abs(Cc["N6"]["cure@40"]["share"] - 0.7) < 1e-9 and abs(Cc["P6"]["cure@40"]["share"] - 0.3) < 1e-9,
       "cure shares: moved / (ld of the swapped-in run - the target's)")
    win, info = X.pick_window({"N6": {f"cure@{L}": (0.0 if L < 30 else 1.0) for L in range(0, 60, 2)},
                               "P6": {f"cure@{L}": (0.0 if L < 30 else 1.0) for L in range(0, 60, 2)}}, 60)
    ok(win == list(range(24, 31)), f"cure window = where the share rises ({win}; {info['rule']})")
    Sh = {"N6": {f"h30.{h}": {"share": 0.1 * (h == 3) + 0.01} for h in range(4)}, "P6": {f"h30.{h}": {"share": 0.05 * (h == 3)} for h in range(4)}}
    top, rand, _ = X.pick_cure_heads(Sh, [30], 4, k=2)
    ok(top[0] == (30, 3) and len(top) == 2 and len(rand) == 2 and not set(top) & set(rand), f"cure head pick {top} / {rand}")

    # --- setup, steps, after_step, estimate ---
    class A_:
        model, fixtures, batch, budget_min, e4_discovery, e4_gate_samples = "gemma-4-31b-it", 30, 16, 60.0, 10, 4
    plug = X.setup(A_(), settings, {"type": "object"}, li)
    ctx = dict(plug["ctx"], args=A_(), panel=None)
    S_ = plug["steps"](ctx)
    ok([s[0] for s in S_] == ["e4 gate", "e4 build", "e4 stats", "e4 attention", "e4 patch", "e4 steer P", "e4 cure",
                              "e4 cure heads", "e4 cure circuit", "e4 steer N"], "ten steps in priority order")
    ok(len(S_[0][2]) == 60 and len(S_[1][2]) == 70 and len(S_[5][2]) == 60 and len(S_[7][2]) == 10 and len(S_[8][2]) == 20
       and len(S_[9][2]) == 20 and {x["target"] for x in S_[9][2]} == {"N1", "N6"}, "item counts per step")
    ok(S_[0][3]["plan"] == [] and S_[0][3]["n_none"] == 4 and S_[0][3]["schema"] == {"type": "object"}, "gate: unsteered, 4 per prompt, schema")
    typ, slow = ctx["estimate_min"]
    ok(18 <= typ <= 45 and slow <= 65, f"estimate {typ} min typical, {slow} if 1.5x slower")
    kw_rows = ctx["heads_kw"]["rows"]
    got = {"e4 cure": X.stub("e4_patch", ctx["fx"], dict(ctx, _stub_seen={x["item_id"]: 1 for x in ctx["fx"]}))}
    ok(X.after_step("e4 cure", got, ctx) is None and ctx["heads_kw"]["rows"] is kw_rows and len(kw_rows["N6"]) == 231
       and ctx["window"]["band"], f"after cure: head rows set in place, window {ctx['window']['band']}")
    stub_ctx = dict(ctx, _stub_seen={x["item_id"]: 2 for x in ctx["fx"]})
    got["e4 cure heads"] = X.stub("e4_patch", [x for x in ctx["fx"] if x["fixture"] in ctx["disc"]], stub_ctx)
    ok(X.after_step("e4 cure heads", got, ctx) is None and len(ctx["circuit_kw"]["rows"]["N6"]) == 7 and len(ctx["pick"]["top"]) == 8,
       f"after cure heads: circuit rows in place, top {ctx['pick']['top'][:3]}")
    build = [{"kind": "fixture", "fixture": f"Y{i:02d}", "skipped": "x"} for i in range(25)] + \
        X.stub("e4_build", ctx["fx"][:5], ctx)
    ok("only 5 of 30" in (X.after_step("e4 build", {"e4 build": build}, ctx) or ""), "build gate: < 10 usable fixtures stops")
    ctx2 = dict(ctx, patch_kw=dict(ctx["patch_kw"]), cure_kw=dict(ctx["cure_kw"]), build_kw=dict(ctx["build_kw"]),
                steerP_kw=dict(ctx["steerP_kw"], plan=[list(x) for x in ctx["steerP_kw"]["plan"]]))
    X._relayer(ctx2, 6, 2)
    ok(ctx2["layers"]["mapped"] and ctx2["top"] and all(h < 2 for _L, h in ctx2["top"]) and ctx2["build_kw"]["keep"] == [4]
       and all(x[1] < 6 for x in ctx2["steerP_kw"]["plan"]), f"relayer to 6 layers x 2 heads (top {ctx2['top']})")

    # --- full preview session ---
    import contextlib
    import glob
    import io
    pat = os.path.join(ROOT, "data", "interp", "interp-e4-*-preview")       # only ours
    before = set(glob.glob(pat))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = li.main(["--preview", "--experiment", "e4", "--fixtures", "9"])
    out = buf.getvalue()
    new = sorted(set(glob.glob(pat)) - before)
    ok(rc == 0 and len(new) == 1, "preview session runs and writes one run folder")
    ok("gate (sampled openings" in out and "cure window" in out and "steer N" in out and "cure circuit" in out, "preview report prints all parts")
    if new:
        s = json.load(open(os.path.join(new[0], "summary.json")))
        mf_ = json.load(open(os.path.join(new[0], "run_manifest.json")))
        res = [json.loads(l) for l in open(os.path.join(new[0], "results.jsonl"))]
        ok(not s.get("analysis_error") and s["gate"]["pass"] and s["patch"] and s["cure"] and s["circuit"] and s["steer_P"],
           "preview summary: gate, patch, cure, circuit, steering")
        ok(mf_["plugin"]["name"] == "e4" and mf_["roles"] == ["seller"], "preview manifest")
        ok({r["step"] for r in res} == {lab for lab, *_ in plug["steps"](ctx)}, "preview: results for all ten steps")
        for d in new:
            shutil.rmtree(d)
    ok("e4" in li.load_plugins(), "launch_interp finds the shim")


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
    import tasks_e4 as E                               # its own copy (tasks loads another for TASKS)
    import tasks_e3c as E3C
    import interp_e4_laptop as X
    torch.manual_seed(0)
    model, tok, info = model_io.load_text_only(TINY, dtype=torch.float32)
    name = "gemma-4-31b-it"
    ok(tasks.TASKS.get("e4_patch") is not None, "tasks loads the e4 plug-in")
    settings = json.load(open(os.path.join(ROOT, "prereg", "settings.json")))
    X._LI[:] = [li]
    fx = X.build_items(li, settings, 3)
    blocks = model_io.decoder_layers(model)
    nL, nH = len(blocks), int(blocks[0].self_attn.config.num_attention_heads)
    Ls = X.layer_sets(nL)
    SL = Ls["keep"][0]

    # 1. alignment
    Ef = E.encode_fixture(tok, model_io, fx[0])
    e1, e6 = Ef["P1"], Ef["P6"]
    ok(len(Ef["N1"]["A"]) == 117 and (Ef["N1"]["n_H"], Ef["N1"]["n_S"], Ef["N1"]["n_reply"]) == (38, 68, 11)
       and Ef["N1"]["A"] == list(range(Ef["N1"]["span"][-1] + 1, len(Ef["N1"]["ids"]))),
       "N: A = every token after the limit (117 = 38 + 68 + 11), as in E2")
    ok(len(e1["R"]) == 39 and len(e6["R"]) == 51 and [tok.decode([e1["ids"][i]]) for i in e1["hi"]] == ["1", "2", "5"]
       and len(e6["lo"]) == 9 and tok.decode([e1["ids"][i] for i in e1["R"]]).startswith("\nYou also know")
       and tok.decode([e1["ids"][i] for i in e1["R"]]).endswith("know this."), "P: range line 39 / 51 tokens; LO / HI digit spans")

    # 2. build
    vec, steer = li.build_e3_items(settings, 3)
    nonmoney = [v for v in vec if v["kind"] == "nonmoney"][:4]
    B = E.e4_build(model, tok, nonmoney + fx, settings, name, model_io, keep=Ls["keep"], head_layers=Ls["heads"])
    Bf = [r for r in B if r["kind"] == "fixture"]
    ok(all(not r.get("skipped") for r in Bf) and Bf[0]["n_A"] == 117 and all(r["stand_in"] for r in Bf),
       "build: 3 fixtures usable (stand-in prices on random weights)")
    ok(max(r["h_identical"] for r in Bf) < 1e-5, f"H's states are identical with and without the range ({max(r['h_identical'] for r in Bf):.1e})")
    pairs = [v for v in vec if v["kind"] == "pair"]
    E3C.e3c_build(model, tok, pairs + nonmoney, settings, name, model_io, layers=[SL])
    st3 = E3C._STORE["e3c"]
    wd = max(float((E._STORE["d"][f][0] - st3["d"][f][0]).abs().max() / st3["d"][f][0].abs().max()) for f in st3["d"])
    ok(wd < 1e-5, f"plain differences at layer {SL} = E3c's (max |Δ| {wd:.1e} of the largest entry)")
    V, used = E.vectors("Y01", SL)
    v3 = E3C._vectors(st3, "Y01", SL)[0]
    ok(used and float((V["vN"] - v3["v_pos"]).abs().max()) < 1e-5 and float((V["u"] - v3["nonmoney"]).abs().max()) < 1e-5,
       "leave-one-out direction and non-money vector = E3c's")
    S = E.e4_stats(model, tok, [{"item_id": "stats"}], settings, name, model_io)[0]
    h = S["per_segment"]["H"]
    ok(all(abs(x - 1) < 1e-4 for x in h["proj_vP_on_vN"][1:]) and all(abs(x) < 1e-6 for x in h["norm_c6"]),
       "stats on H: v_P = v_N and c6 = 0 (the range comes after H)")

    # 3. patch exactness
    it = fx[0]
    L0 = Ls["band"][0]
    rows = {"N1": [{"n": f"own@{L}", "ops": [["set", "resid", L, -1, "A", "N6"]]} for L in range(nL)]
            + [{"n": f"add@{L}", "ops": [["add", "resid", L, -1, "A", 1.0, "N6", "N1"]]} for L in range(nL)]
            + [{"n": "vN", "ops": [["vec", SL, "A", 1.0, "vN"]]}]}
    Pt = E.e4_patch(model, tok, [it], settings, name, model_io, rows=rows, batch=4)[0]
    got = {x["n"]: x for x in Pt["targets"]["N1"]["rows"]}
    e2 = [x for x in li.build_e2_items(settings, 1) if x["direction"] == "x1e6->x1"]
    ref = tasks.e2_patch(model, tok, e2, settings, name, model_io, layers=list(range(nL)), positions=("post",), batch=4)[0]
    rp = {x["layer"]: x["p1"] for x in ref["rows"] if x["layer"] != "clean"}
    w = max(abs(p - q) for L in range(nL) for p, q in zip(got[f"own@{L}"]["p1"], rp[L]))
    ok(w <= 2e-4 and Pt["targets"]["N1"]["hooks_ok"], f"'set' N6 into N1 = E2's post patch at every layer (digit probs within {w:.0e})")
    w = max(abs(got[f"own@{L}"]["ld"] - got[f"add@{L}"]["ld"]) for L in range(nL))
    ok(w < 1e-4, f"'add' of the plain difference to N1 = 'set' of N6 (max |Δ ld| {w:.1e})")
    ok(abs(got["clean"]["ld"] - Bf[0]["ld"]["N1"]) < 1e-3, f"clean row ld = the build's ({got['clean']['ld']:.4f} vs {Bf[0]['ld']['N1']:.4f})")
    br = tasks.e3b_build(model, tok, pairs + nonmoney, settings, name, model_io, layer=SL)
    bb = tasks.e3b_bridge(model, tok, [s for s in steer if s["fixture"] == "Y01" and s["direction"] == "x1+v"], settings, name,
                          model_io, rows=(("v_pos", "all", 1.0),), batch=4)[0]
    w = max(abs(p - q) for p, q in zip(got["vN"]["p1"], next(x["p1"] for x in bb["rows"] if x["vector"] == "v_pos")))
    ok(w <= 2e-4 and Pt["loo"], f"'vec' vN on N1 = E3b's bridge at layer {SL} (digit probs within {w:.0e})")

    selfrows = {}
    for t in X.PROMPTS:
        selfrows[t] = [{"n": f"{k}@{L}.{seg}", "ops": [["set", k, L, 0 if k == "head" else -1, seg, t]]}
                       for k in ("resid", "attn", "head") for L in (L0, nL - 1) for seg in ("A", "S", "reply", "Srep", "H")] + \
                      [{"n": "addself", "ops": [["add", "resid", L0, -1, "A", 1.0, t, t]]}]
    Ps = E.e4_patch(model, tok, [it], settings, name, model_io, rows=selfrows, batch=8)[0]
    w = max(abs(x["ld"] - T["ld_clean"]) for T in Ps["targets"].values() for x in T["rows"])
    ok(w < 1e-5, f"every prompt patched with its own values is unchanged (all kinds, segments; max |Δ ld| {w:.1e})")

    hrows = {t: [{"n": "attn", "ops": [["set", "attn", L0, -1, "Srep", src]]},
                 {"n": "heads", "ops": [["set", "head", L0, h, "Srep", src] for h in range(nH)]},
                 {"n": "last", "ops": [["set", "resid", nL - 1, -1, "Srep", src]]}] for t, src in (("N6", "P6"), ("P6", "N6"))}
    Ph = E.e4_patch(model, tok, [it], settings, name, model_io, rows=hrows, batch=4)[0]
    g6, gp = ({x["n"]: x for x in Ph["targets"][t]["rows"]} for t in ("N6", "P6"))
    ok(max(abs(g6["attn"]["ld"] - g6["heads"]["ld"]), abs(gp["attn"]["ld"] - gp["heads"]["ld"])) < 1e-4
       and abs(g6["attn"]["ld"] - g6["clean"]["ld"]) > 1e-4, "all heads of a layer = its attention output (and it moves)")
    w = max(max(abs(p - q) for p, q in zip(g6["last"]["p1"], gp["clean"]["p1"])), max(abs(p - q) for p, q in zip(gp["last"]["p1"], g6["clean"]["p1"])))
    ok(w < 1e-5, f"the cure at the last layer on S + reply gives the other run's first-digit probabilities, both ways ({w:.1e})")

    mix = {"P1": [{"n": "a", "ops": [["add", "resid", L0, -1, "A", 1.0, "N6", "N1"]]}, {"n": "b", "ops": [["set", "head", L0, 1, "reply", "P6"]]},
                  {"n": "c", "ops": [["vec", SL, "A", 2.0, "vN"]]}, {"n": "d", "ops": [["set", "resid", L0, -1, "A", "P6"],
                                                                              ["add", "head", SL, 0, "S", 1.0, "N6", "N1"]]}]}
    mixed = {x["n"]: x["ld"] for x in E.e4_patch(model, tok, [it], settings, name, model_io, rows=mix, batch=4)[0]["targets"]["P1"]["rows"]}
    alone = {}
    for r in mix["P1"]:
        alone.update({x["n"]: x["ld"] for x in E.e4_patch(model, tok, [it], settings, name, model_io, rows={"P1": [r]}, batch=4)[0]["targets"]["P1"]["rows"]})
    w = max(abs(mixed[n] - alone[n]) for n in "abcd")
    ok(w < 1e-6, f"mixed batches = each row alone (max |Δ ld| {w:.1e})")

    # 4. real-opening steering
    sit = X.steer_items([it], ("N1",), "t")
    mine = E.e4_steer(model, tok, sit, settings, name, model_io, plan=[["vN", SL, 1.0, 1]], n_none=8, batch=16, greedy=True, max_new=20)[0]
    E3C.e3c_build(model, tok, pairs + nonmoney, settings, name, model_io, layers=[SL])
    s3 = [s for s in steer if s["fixture"] == "Y01" and s["direction"] == "x1+v"]
    theirs = E3C.e3c_steer(model, tok, s3, settings, name, model_io, layers=[SL], alphas=(1.0,), samples=1, batch=16, min_none=8,
                           greedy=True, max_new=20)[0]
    a = [x["content"] for x in mine["rows"] if x["vector"] == "vN"]
    b_ = [x["content"] for x in theirs["rows"] if x["vector"] == "v_pos"]
    ok(a == b_ and mine["hook_calls"][0][0] == len(Ef["N1"]["ids"]) - Ef["N1"]["n_reply"] - Ef["N1"]["s0"] and mine["loo"],
       "steered real opening (greedy) on N1 = E3c's at the same layer")
    pit = X.steer_items([it], ("P6",), "t")
    mine = E.e4_steer(model, tok, pit, settings, name, model_io, plan=[], n_none=2, batch=2, greedy=True, max_new=20)[0]
    ref_g = tasks.generate(model, tok, [{"item_id": "g", "messages": it["prompts"]["P6"]["messages"], "role": "seller", "condition": "D8",
                                         "limit_units": it["prompts"]["P6"]["limit_units"], "seed": 1}], settings, name, model_io,
                           greedy=True)[0]
    ok(all(x["content"][:40] == ref_g["content"][:40] for x in mine["rows"]) and mine["rehearsal"],
       "unsteered range openings (greedy, batched with a cached prefix) = tasks.generate")

    # 5. attention
    impl0 = model.config._attn_implementation
    A = E.e4_attn(model, tok, fx[:1], settings, name, model_io)[0]
    vals = [v for k in A["mass"] for L in A["mass"][k] for q in L for sp in q for v in sp]
    ok(model.config._attn_implementation == impl0 and A["keys"]["P1"] == ["limit", "lo", "hi", "range"] and A["keys"]["N1"] == ["limit"]
       and len(A["mass"]["P6"]) == nL and all(0 <= v <= 1.0001 for v in vals) and max(vals) > 0,
       f"attention: masses in [0, 1], keys limit/lo/hi/range on P; setting restored ({impl0})")


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
