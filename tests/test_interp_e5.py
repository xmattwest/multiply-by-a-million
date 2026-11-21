#!/usr/bin/env python3
"""test_interp_e5.py: checks for the E5 plug-in (scripts/interp/tasks_e5.py on the pod; scripts/interp_e5_laptop.py
behind the shim scripts/interp/exp_e5.py).

    python3 tests/test_interp_e5.py            laptop checks (stdlib), then the tiny-model checks in the rehearsal
                                               venv if it exists (else they fail, loudly)
    /tmp/si-interp-venv/bin/python tests/test_interp_e5.py --torch      the tiny-model checks only

Laptop: E5's prompts are byte-identical to the requests Study 3 recorded for message 1 (both roles, A0/A1, both
scales); A0 is also the earlier interp runs' D2/D8 prompt; A1 differs only by the appraisal line; one fixture per
distinct limit, discovery and held-out disjoint; the vLLM reference; the gate (pass and each failure); the six
predictions scored on synthetic summaries (hit and miss); after_step hand-overs; the predictions-committed check; a full
--preview session.
Tiny model (random-weight Gemma 4 of the 31B's class, fp32, CPU): the four prompts align on A after the limit while
differing before it; appraisal digits and line located; each role has its own store; the per-target prefix cache gives
the build's ld on every target; own-value patches are no-ops; the last-layer swap gives the other run's digit
probabilities both ways; unsteered A1 openings = tasks.generate; E4's own encoder and store are restored afterwards.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "interp"))
VENV_PY = "/tmp/si-interp-venv/bin/python"
TINY = "/tmp/si-tiny-31b"
N = {"pass": 0, "fail": 0}
MODEL = "gemma-4-31b-it"


def ok(cond, name):
    N["pass" if cond else "fail"] += 1
    print(("ok    " if cond else "FAIL  ") + name)


# ======================================================================================================
# laptop (stdlib)
# ======================================================================================================
def recorded_first_moves():
    """{(fixture, cond, scale, role): messages} of Study 3's recorded message-1 requests for Gemma (attempt 1, try 1)."""
    out = {}
    for d in sorted(glob.glob(os.path.join(ROOT, "data", "runs", f"s3-{MODEL}-s1-*"))):
        for line in open(os.path.join(d, "calls.jsonl"), encoding="utf-8"):
            if '"request"' not in line or "/a1/m1/t1" not in line:
                continue
            c = json.loads(line)
            gid = c["call_id"].split("|", 1)[1].split("/")[0]            # G01-A1-x1e8-S-s1
            fx, cond, scale, who, _s = gid.split("-")
            if cond in ("A0", "A1"):
                out.setdefault((fx, cond, scale, "seller" if who == "S" else "buyer"), c["request"]["messages"])
    return out


def laptop():
    import launch_interp as li
    import interp_e5_laptop as X
    X._LI[:] = [li]
    settings = json.load(open(os.path.join(ROOT, "prereg", "settings.json")))

    # --- prompts: Study 3's own, one per distinct limit ---
    rec = recorded_first_moves()
    for role in X.ROLES:
        fx = X.build_items(role)
        lim = [it["base_limit"] for it in fx]
        ok(len(fx) == (19 if role == "seller" else 18) and len(set(lim)) == len(lim),
           f"{role}s: {len(fx)} fixtures, one per distinct limit")
        n, bad = 0, []
        for it in fx:
            for k, p in it["prompts"].items():
                key = (it["fixture"], X.COND[k[0]], X.SCALE_KEY[X.SCALE[k]], role)
                if key in rec:
                    n += 1
                    if rec[key] != p["messages"]:
                        bad.append(key)
        ok(n >= 3 * len(fx) and not bad, f"{role}s: {n} prompts byte-identical to Study 3's recorded message-1 requests"
           + (f" (differ: {bad[:3]})" if bad else ""))
        it = fx[0]
        b = it["base_limit"]
        ok(it["prompts"]["N1"]["messages"] == li.render_item(role, b, "D2", settings)
           and it["prompts"]["N6"]["messages"] == li.render_item(role, b, "D8", settings),
           f"{role}s: A0 = the earlier interp runs' D2/D8 prompt (X51-X64 used Study 2's prompt)")
        for m, (a, p) in ((0, ("N1", "P1")), (6, ("N6", "P6"))):
            A, P = it["prompts"][a]["messages"], it["prompts"][p]["messages"]
            V = "100" if m == 0 else "100000000"
            line = f"- An independent appraisal values the lot at {V} dollars. Both sides know this."
            ok(A[1] == P[1] and P[0]["content"].count(line) == 1 and P[0]["content"].replace(line + "\n", "") == A[0]["content"]
               and it["prompts"][p]["appraisal"] == V and it["prompts"][a]["appraisal"] is None,
               f"{role}s x10^{m}: A1 = A0 plus the appraisal line in the system message, nothing else")

    # --- vLLM reference ---
    for role in X.ROLES:
        v, runs = X.vllm_reference(MODEL, role, ROOT)
        fx = X.build_items(role)
        ok(sum(len(x) for x in v.values()) == 640 and len(runs) == 2
           and all(v.get((c, m, it["base_limit"])) for it in fx for c in ("A0", "A1") for m in (0, 6)),
           f"{role}s: 640 Study 3 openings from both sessions; every prompt has vLLM openings in all four cells")

    # --- gate ---
    fx = X.build_items("seller")
    vref, _ = X.vllm_reference(MODEL, "seller", ROOT)

    def recs(shift=0.0, bad=0, reh=False):
        out = []
        for it in fx:
            for t in X.PROMPTS:
                g = X.st.mean(vref[(X.COND[t[0]], X.SCALE[t], it["base_limit"])]) + (shift if t == "P6" else 0.0)
                p = round(it["base_limit"] * (1 + g)) * 10 ** X.SCALE[t]
                out.append({"target": t, "fixture": it["fixture"], "base_limit": it["base_limit"], "rehearsal": reh,
                            "rows": [{"vector": "none", "layer": None, "alpha": 0.0, "price": p, "errors": ["X"] if j < bad else [],
                                      "action": "OFFER", "aligned": True} for j in range(4)]})
        return out
    g = X.summarize_gate(recs(), li, "seller", vref, fx)
    ok(g["pass"] and all(abs(c["mean_gap"] - c["vllm_gap"]) < 0.01 for c in g["cells"].values()), f"gate passes on vLLM-like openings ({g['line'][:70]})")
    g = X.summarize_gate(recs(shift=0.05), li, "seller", vref, fx)
    ok(not g["pass"] and "P6" in g["why"] and "A1 shift" in g["why"], f"gate fails on a 5-point A1 difference ({g['why']})")
    g = X.summarize_gate(recs(bad=1), li, "seller", vref, fx)
    ok(not g["pass"] and "valid" in g["why"], "gate fails below 95% valid")

    # --- setup, steps, hand-overs ---
    class A_:
        model, batch, budget_min, e5_discovery, e5_gate_samples, e5_roles, preview, attach = MODEL, 16, 60.0, 7, 4, "seller,buyer", True, False
    plug = X.setup(A_(), settings, {"type": "object"}, li)
    ctx = dict(plug["ctx"], args=A_(), panel=None)
    S_ = plug["steps"](ctx)
    labs = [s[0] for s in S_]
    ok(labs == [f"e5 seller {s}" for s in X.STEPS_S] + [f"e5 buyer {s}" for s in X.STEPS_B], "15 steps: sellers' 10, then buyers' 5")
    Rs = ctx["R"]["seller"]
    ok(len(Rs["disc"]) == 7 and len(Rs["held"]) == 12 and not set(Rs["disc"]) & set(Rs["held"]),
       "sellers: 7 discovery and 12 held-out limits, disjoint (as the predictions file says)")
    ok(all(kw.get("role") == lab.split()[1] for lab, _t, _i, kw, _c in S_) and all(it.get("e5_step") == lab for lab, _t, items, _k, _c in S_ for it in items),
       "every step's kw names its role; every item names its step")
    ok(S_[0][3]["schema"] == X._study3()[0].response_schema() and S_[0][3]["n_none"] == 4 and len(S_[0][2]) == 76,
       "gate: Study 3's schema, 4 openings per prompt, 19 x 4 prompts")
    ok(not S_[-1][3]["rows"] == {} and ctx["R"]["buyer"]["kw"]["patch"]["rows"] == {}, "buyers: no patch rows (brief), swap rows present")
    typ, slow = ctx["estimate_min"]
    ok(20 <= typ <= 60 and slow <= 90, f"estimate {typ} min typical, {slow} if 1.5x slower")
    got = {"e5 seller swap": X.stub("e5_patch", plug["steps"](ctx)[7][2], ctx)}
    ok(X.after_step("e5 seller swap", got, ctx) is None and len(Rs["kw"]["swap heads"]["rows"].get("N6", [])) == 231 and Rs["window"]["band"],
       f"after the swap: head rows in place, window {Rs['window']['band']}")
    got["e5 seller swap heads"] = X.stub("e5_patch", plug["steps"](ctx)[8][2], ctx)
    ok(X.after_step("e5 seller swap heads", got, ctx) is None and len(Rs["pick"]["top"]) == 8 and Rs["kw"]["swap circuit"]["rows"]["N6"],
       "after swap heads: top 8 picked, circuit rows in place")
    stop = X.after_step("e5 seller gate", {"e5 seller gate": recs(shift=0.05)}, ctx)
    ok(stop and "seller gate failed" in stop and X.after_step("e5 seller gate", {"e5 seller gate": recs(shift=0.05, reh=True)}, ctx) is None,
       "a failed gate stops the session; a rehearsal's gate is reported, not applied")

    # --- predictions ---
    def summ(gp=0.2, early=0.05, late=0.8, rep=0.7, sg=0.2, a1=0.1, a0=0.9, shared=((41, 29), (41, 11)), joint=0.7, b_early=0.05):
        def swap(e, l):
            return {"N6": {**{f"cure@{L}": {"share": e} for L in (30, 36, 38, 39)}, "cure@40": {"share": 0.3},
                           "cure@41": {"share": l}, "cure@42": {"share": l}, "cure reply@44": {"share": rep}, "cure S@44": {"share": sg}}}
        pf = lambda s: {f"G{i:02d}": [s * 0.2, 0.2] for i in range(5)}                  # noqa: E731
        top = list(shared) + [(38, h) for h in range(8 - len(shared))]
        se = {"build": {"G_N": 10.0, "G_P": 10.0 * gp}, "swap": swap(early, late),
              "steer_P": {t: {"vN@42x1": {"per_fixture": pf(a1)}} for t in ("P1", "P6")},
              "steer_N": {t: {"vN@42x1": {"per_fixture": pf(a0)}} for t in ("N1", "N6")},
              "pick": {"top": [list(x) for x in top]}, "circuit": {"N6": {"top8": {"share": joint}}}}
        return {"layers": {"n_layers": 60}, "roles": {"seller": se, "buyer": {"swap": swap(b_early, late)}}}
    P = {p["id"]: p for p in X.evaluate(summ())}
    ok([P[k]["result"] for k in sorted(P)] == ["HIT"] * 6, "all six predictions score HIT on a summary that meets them")
    P = {p["id"]: p for p in X.evaluate(summ(gp=0.7, early=0.3, rep=0.1, a1=0.6, shared=((41, 29),), b_early=0.4))}
    ok([P[k]["result"] for k in sorted(P)] == ["MISS"] * 6, "and MISS on one that breaks each")
    P = {p["id"]: p for p in X.evaluate({"layers": {"n_layers": 60}, "roles": {}})}
    ok(all(p["result"] == "n.a." for p in P.values()), "n.a. when a step produced nothing")
    ok(len(X.PRIOR_HEADS) == 13, "13 heads in X57's and X61's top 8 together (3 shared)")

    # --- predictions-committed check ---
    tmp = tempfile.mkdtemp()
    try:
        subprocess.run(["git", "init", "-q", tmp], check=True)
        ok("does not exist" in (X.predictions_committed(tmp) or ""), "paid launch refused without the predictions file")
        os.makedirs(os.path.join(tmp, "config", "interp"))
        open(os.path.join(tmp, X.PRED_FILE), "w").write("x\n")
        ok("not committed" in (X.predictions_committed(tmp) or ""), "... or with it uncommitted")
        env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
        subprocess.run(["git", "-C", tmp, "add", X.PRED_FILE], check=True)
        subprocess.run(["git", "-C", tmp, "commit", "-q", "-m", "p"], check=True, env=env)
        ok(X.predictions_committed(tmp) is None, "accepted once committed")
        open(os.path.join(tmp, X.PRED_FILE), "a").write("y\n")
        ok("uncommitted changes" in (X.predictions_committed(tmp) or ""), "refused again once edited")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # --- full preview session ---
    import contextlib
    import io
    pat = os.path.join(ROOT, "data", "interp", "interp-e5-*-preview")
    before = set(glob.glob(pat))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = li.main(["--preview", "--experiment", "e5"])
    out = buf.getvalue()
    new = sorted(set(glob.glob(pat)) - before)
    ok(rc == 0 and len(new) == 1, "preview session runs and writes one run folder")
    ok("gate vs Study 3's vLLM openings: PASS" in out and "swap A1 into A0" in out and "predictions (" in out and "E5-6" in out,
       "preview report prints gate, swap and predictions")
    if new:
        s = json.load(open(os.path.join(new[0], "summary.json")))
        mf_ = json.load(open(os.path.join(new[0], "run_manifest.json")))
        res = [json.loads(l) for l in open(os.path.join(new[0], "results.jsonl"))]
        ok(not s.get("analysis_error") and set(s["roles"]) == {"seller", "buyer"} and len(s["predictions"]) == 6,
           "preview summary: both roles, six predictions")
        ok(mf_["plugin"]["name"] == "e5" and mf_["plugin"]["predictions_sha256"] and mf_["roles"] == ["seller", "buyer"],
           "preview manifest records the predictions' sha256")
        ok({r["step"] for r in res} == set(labs), "preview: results for all 15 steps")
        for d in new:
            shutil.rmtree(d)
    ok("e5" in li.load_plugins() and li.pod_plugins("e5") == "e4,e5,heads", "launch_interp finds the shim; ships e4, e5, heads")


# ======================================================================================================
# tiny model (torch; rehearsal venv)
# ======================================================================================================
def tiny():
    os.environ["SI_INTERP_DEVICE"] = "cpu"
    os.environ["SI_INTERP_REHEARSAL"] = "1"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    import launch_interp as li
    import model_io
    import tasks
    import tasks_e5 as E5
    import interp_e5_laptop as X
    torch.manual_seed(0)
    model, tok, _info = model_io.load_text_only(TINY, dtype=torch.float32)
    ok(tasks.TASKS.get("e5_patch") is not None, "tasks loads the e5 plug-in")
    settings = json.load(open(os.path.join(ROOT, "prereg", "settings.json")))
    X._LI[:] = [li]
    blocks = model_io.decoder_layers(model)
    nL, nH = len(blocks), int(blocks[0].self_attn.config.num_attention_heads)
    Ls = X4L = X.X4.layer_sets(nL)
    fs, fb = X.build_items("seller")[:3], X.build_items("buyer")[:3]
    orig = (E5.E4.encode, E5.E4.encode_fixture, E5.E4._STORE)

    # 1. alignment and spans
    for fx in (fs, fb):
        E = E5.encode_fixture(tok, model_io, fx[0])
        r = E["N1"]
        ok(all([e["ids"][p] for p in e["A"]] == [r["ids"][p] for p in r["A"]] for e in E.values())
           and E["P1"]["s0"] > E["N1"]["s0"] and E["P1"]["ids"][:E["P1"]["s0"]] != E["N1"]["ids"][:E["N1"]["s0"]],
           f"{fx[0]['role']}: the four prompts share A ({len(r['A'])} tokens) and differ before the limit")
        ok(tok.decode([E["P1"]["ids"][i] for i in E["P1"]["hi"]]) == "100" and tok.decode([E["P6"]["ids"][i] for i in E["P6"]["hi"]]) == "100000000"
           and "appraisal values the lot at" in tok.decode([E["P1"]["ids"][i] for i in E["P1"]["R"]])
           and E["P1"]["R"][-1] < E["P1"]["span"][0] and E["N1"]["R"] == [], f"{fx[0]['role']}: appraisal digits and line located, before the limit")

    # 2. builds, one store per role
    nonmoney = [v for v in li.build_e3_items(settings, 1)[0] if v["kind"] == "nonmoney"][:3]
    Bs = E5.e5_build(model, tok, nonmoney + fs, settings, MODEL, model_io, keep=Ls["keep"], head_layers=Ls["heads"], role="seller")
    Bb = E5.e5_build(model, tok, nonmoney + fb, settings, MODEL, model_io, keep=Ls["keep"], head_layers=Ls["heads"], role="buyer")
    ok(all(not r.get("skipped") for r in Bs + Bb if r.get("kind") == "fixture"), "builds: every fixture usable (stand-in prices)")
    shared = set(E5._STORES["seller"]["d"]) & set(E5._STORES["buyer"]["d"])
    ok(E5._STORES["seller"]["n"] == 3 and E5._STORES["buyer"]["n"] == 3 and E5._STORES["seller"]["u_n"] == 3
       and all(float((E5._STORES["seller"]["d"][f] - E5._STORES["buyer"]["d"][f]).abs().max()) > 0 for f in shared),
       f"separate stores per role ({len(shared)} fixture ids in both, different directions)")
    ok((E5.E4.encode, E5.E4.encode_fixture, E5.E4._STORE) == orig, "E4's encoder and store are restored after every call")

    # 3. per-target prefix caches: clean rows = the build's ld on all four targets
    it, b0 = fs[0], next(r for r in Bs if r.get("fixture") == fs[0]["fixture"])
    L0 = Ls["band"][0]
    rows = {t: [{"n": f"{k}@{L}.{seg}", "ops": [["set", k, L, 0 if k == "head" else -1, seg, t]]}
                for k in ("resid", "attn", "head") for L in (L0, nL - 1) for seg in ("A", "S", "reply", "Srep", "H")]
            for t in X.PROMPTS}
    Pt = E5.e5_patch(model, tok, [it], settings, MODEL, model_io, rows=rows, batch=8, role="seller")[0]
    w = max(abs(Pt["targets"][t]["ld_clean"] - b0["ld"][t]) for t in X.PROMPTS)
    ok(w < 1e-3 and all(T["hooks_ok"] for T in Pt["targets"].values()), f"clean ld on each target's own prefix cache = the build's (max |Δ| {w:.1e})")
    w = max(abs(x["ld"] - T["ld_clean"]) for T in Pt["targets"].values() for x in T["rows"])
    ok(w <= 2e-5, f"every prompt patched with its own values is unchanged (all kinds, segments; max |Δ ld| {w:.1e}, ld stored to 1e-5)")
    sw = {t: [{"n": "last", "ops": [["set", "resid", nL - 1, -1, "Srep", src]]}] for t, src in (("N6", "P6"), ("P6", "N6"))}
    Ps = E5.e5_patch(model, tok, [it], settings, MODEL, model_io, rows=sw, batch=2, role="seller")[0]
    g6, gp = ({x["n"]: x for x in Ps["targets"][t]["rows"]} for t in ("N6", "P6"))
    w = max(max(abs(p - q) for p, q in zip(g6["last"]["p1"], gp["clean"]["p1"])), max(abs(p - q) for p, q in zip(gp["last"]["p1"], g6["clean"]["p1"])))
    ok(w < 1e-5, f"the swap at the last layer on S + reply gives the other run's first-digit probabilities, both ways ({w:.1e})")
    vr = {"P1": [{"n": "v", "ops": [["vec", Ls["keep"][0], "A", 1.0, "vN"]]}]}
    Pv = E5.e5_patch(model, tok, [it], settings, MODEL, model_io, rows=vr, batch=2, role="seller")[0]
    ok(Pv["loo"] and abs(Pv["targets"]["P1"]["rows"][0]["ld"] - Pv["targets"]["P1"]["ld_clean"]) > 1e-6, "the seller's A0 direction applies to A1 (and moves it)")

    # 4. unsteered A1 openings = tasks.generate; attention keys
    pit = X.steer_items([it], ("P6",), "t", "e5 seller steer P")
    mine = E5.e5_steer(model, tok, pit, settings, MODEL, model_io, plan=[], n_none=2, batch=2, greedy=True, max_new=20, role="seller")[0]
    ref = tasks.generate(model, tok, [{"item_id": "g", "messages": it["prompts"]["P6"]["messages"], "role": "seller", "condition": "A1-x1e8",
                                       "limit_units": it["prompts"]["P6"]["limit_units"], "seed": 1}], settings, MODEL, model_io, greedy=True)[0]
    ok(all(x["content"][:40] == ref["content"][:40] for x in mine["rows"]), "unsteered A1 openings (greedy, own prefix cache) = tasks.generate")
    A = E5.e5_attn(model, tok, fs[:1], settings, MODEL, model_io, role="seller")[0]
    vals = [v for k in A["mass"] for L in A["mass"][k] for q in L for sp in q for v in sp]
    ok(A["keys"]["P1"] == ["limit", "lo", "hi", "range"] and A["keys"]["N1"] == ["limit"] and all(0 <= v <= 1.0001 for v in vals),
       "attention: appraisal digits and line as keys on A1; masses in [0, 1]")


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
