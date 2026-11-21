#!/usr/bin/env python3
"""Tests for E6 (magnitude as a number vs as stakes): laptop logic, then the pod tasks on the tiny rehearsal model.

    python3 tests/test_interp_e6.py            laptop checks (stdlib), then the tiny-model checks in the rehearsal venv
    /tmp/si-interp-venv/bin/python tests/test_interp_e6.py --torch      the tiny-model checks only
"""
import contextlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "interp"))
TINY, VENV = "/tmp/si-tiny-31b", "/tmp/si-interp-venv/bin/python"
MODEL = "gemma-4-31b-it"
N = {"ok": 0, "fail": 0}


def ok(cond, name):
    N["ok" if cond else "fail"] += 1
    print(("ok    " if cond else "FAIL  ") + name, flush=True)


def laptop():
    import launch_interp as li
    import interp_e6_laptop as X
    X._LI[:] = [li]

    # notations and items
    ok([X.amount(81000000, k) for k in X.NOTATIONS] == ["81000000 dollars", "81 million dollars", "$81M", "$81,000,000",
                                                        "81000000 people"], "notations of 81000000")
    ok(X.amount(1200, "words") == "1.2 thousand dollars" and X.amount(58, "short") == "$58" and X.amount(640000, "short") == "$640K",
       "notations below and between the scale words")
    P = X.probe_items()
    ok(len(P) == 5 * 9 * 8 and len({p["item_id"] for p in P}) == len(P) and all(p["text"].endswith(" in total.") for p in P),
       "probe: 360 distinct sentences, one shared ending")
    ok(all(abs(p["log10"] - math.log10(p["mantissa"] * 10 ** p["exp"])) < 1e-6 for p in P), "probe: log10 matches mantissa x 10^e")
    fx = X.X5.build_items("seller")
    F = X.family_items(fx)
    ok({k: len(v) for k, v in F.items()} == {"nego": 14, "risk": 60, "compare": 40, "percent": 30, "sum": 24, "fermi": 10},
       "families: item counts")
    ok(all(it["sign"] == (1.0 if it["scale"] in (0, None) else -1.0) for v in F.values() for it in v),
       "families: sign +1 on $100 (and Fermi), -1 on $100M")
    r = [it for it in F["risk"] if it["base"] == "70-2.4-safeB"]
    ok(len(r) == 2 and "Receive 70000000 dollars for sure." in r[1]["messages"][0]["content"] and r[0]["safe"] == "B"
       and r[0]["messages"][0]["content"].index("B) Receive") > 0, "risk: scaled amounts and option order")
    c = [it for it in F["compare"] if it["scale"] == 6]
    ok(all(it["correct"] == ("A" if int(it["messages"][0]["content"].split("A) ")[1].split(" ")[0])
                                    > int(it["messages"][0]["content"].split("B) ")[1].split(" ")[0]) else "B") for it in c),
       "compare: the correct letter is the larger amount")
    s6 = [it for it in F["sum"] if it["scale"] == 6][0]
    ok(s6["answer"] == 68 * 10 ** 6 and "23000000 dollars plus 45000000 dollars" in s6["messages"][0]["content"], "sum: scaled answer")
    ng = F["nego"]
    ok(ng[0]["messages"] == fx[0]["prompts"]["N1"]["messages"] and ng[1]["messages"] == fx[0]["prompts"]["N6"]["messages"]
       and ng[0]["prefix"].startswith('{"action": "OFFER"'), "nego: Study 3's own A0 seller prompts with the JSON opening")
    lim = X.family_items(fx[:3], 2)
    ok(all(len(v) == 2 for v in lim.values()) and len(X.probe_items(2)) == 5 * 3 * 4, "rehearsal limit caps every family")

    # layers, plan, steps
    L60, L6 = X.layer_plan(60), X.layer_plan(6)
    ok(L60["keep"] == [42, 43, 44] and L60["stake"] == [42, 44] and L60["mlp"] == [4, 10, 20, 30, 41]
       and len(L60["plan"]) == 12 and X.N_NONE + len(L60["plan"]) == X.BATCH, "layers and the 16-row plan (31B)")
    ok(L6["mapped"] and max(L6["keep"] + L6["mlp"]) <= 5, "layers mapped onto a 6-layer model")

    class A:
        preview, attach, e6_limit, model, budget_min = True, False, 0, MODEL, 60.0
    settings = json.load(open(os.path.join(ROOT, "prereg", "settings.json")))
    plug = X.setup(A(), settings, None, li)
    ctx = dict(plug["ctx"])
    st = plug["steps"](ctx)
    ok([s[0] for s in st] == ["e6 build", "e6 dirs", "e6 probe", "e6 fit", "e6 census"] + [f"e6 {f}" for f in X.FAMILIES]
       and {s[1] for s in st} <= li.shipped_tasks(li.pod_plugins("e6")), "steps, all tasks shipped")
    ok(li.pod_plugins("e6") == "e4,e5,e6,heads", "pod plug-ins e4,e5,e6,heads")
    ok(ctx["estimate_min"][0] < 30, f"estimate {ctx['estimate_min']} min")
    kw_before = st[5][3]
    X.on_pod_info(ctx, {"num_hidden_layers": 6})
    ok(kw_before["plan"][0][1] == X.X4.map_layer(42, 6) and ctx["kw"]["probe"]["mlp_layers"] == L6["mlp"],
       "on_pod_info remaps the plan in place")

    class Paid(A):
        preview = False
    for why, prep in (("is not committed", None), ("rehearsals only", "limit")):
        tmp = tempfile.mkdtemp()
        try:
            os.makedirs(os.path.join(tmp, "config", "interp"))
            subprocess.run(["git", "init", "-q", tmp], check=True)
            p = os.path.join(tmp, X.PRED_FILE)
            open(p, "w").write("x\n")
            if prep:
                subprocess.run(["git", "-C", tmp, "add", "."], check=True)
                subprocess.run(["git", "-C", tmp, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "p"], check=True)
            ok(X.predictions_committed(tmp) == (None if prep else f"{X.PRED_FILE} is not committed"),
               f"predictions_committed: {'committed' if prep else 'not committed'}")

            class FakeLi:
                ROOT = tmp
                STEP_GUESS_S = {}
                build_e3_items = staticmethod(li.build_e3_items)

            class P2(Paid):
                e6_limit = 2 if prep else 0
            try:
                X.setup(P2(), settings, None, FakeLi)
                ok(False, f"paid setup refuses ({why})")
            except SystemExit as e:
                ok(why in str(e), f"paid setup refuses ({why})")
        finally:
            shutil.rmtree(tmp)

    # analysis on synthetic records
    ctx6 = dict(plug["ctx"])
    ctx6["kw"] = X.kws(X.layer_plan(60))
    ctx6["layers"] = X.layer_plan(60)
    got = {}
    for lab, task, items, kw, _n in plug["steps"](ctx6):
        got[lab] = X.stub(task, items, ctx6)
    s = X.summarize(got, ctx6)
    res = {p["id"]: p["result"] for p in s["predictions"]}
    ok(len(res) == 10 and all(v == "HIT" for v in res.values()), "evaluate: the canned shapes hit all ten")
    fam = s["families"]["risk"]
    ok(abs(fam["natural_gap"] - 0.8) < 0.05 and fam["share"]["0"]["vR@42x+2"] > 0.5 and abs(fam["share"]["0"]["rand@42x+2"]) < 0.05,
       "risk: natural gap and shares from paired items")
    ok(s["families"]["nego"]["natural_gap"] < 0 and s["families"]["nego"]["share"]["0"]["vR@42x+2"] > 0.5,
       "nego: a negative gap (modesty) and a positive share toward $100M")
    bad = json.loads(json.dumps(got))
    for r in bad["e6 risk"]:
        vr = {(x["layer"], x["alpha"]): x["lp"] for x in r["rows"] if x["vector"] == "vR"}
        for x in r["rows"]:
            if x["vector"] == "rand":
                x["lp"] = list(vr[(x["layer"], x["alpha"])])          # the random vector as strong as vR
    s2 = X.summarize(bad, ctx6)
    ok({p["id"]: p["result"] for p in s2["predictions"]}["E6-6"] == "MISS", "evaluate: a large random-vector effect makes E6-6 miss")
    for r in bad["e6 fermi"]:
        for x in r["rows"]:
            if x["vector"] == "vR":
                x["value"] *= 10
    s3 = X.summarize(bad, ctx6)
    ok({p["id"]: p["result"] for p in s3["predictions"]}["E6-9"] == "MISS", "evaluate: a 10x Fermi change makes E6-9 miss")
    empty = X.evaluate({"layers": X.layer_plan(60)})
    ok(all(p["result"] == "n.a." for p in empty), "evaluate: n.a. without data")
    ok(X.metric({"family": "nego", "base_limit": 93, "scale": 0}, {"value": 120.0}, li) == 120 / 93 - 1
       and X.metric({"family": "nego", "base_limit": 93, "scale": 0}, {"value": 120e6}, li) is None,
       "metric: nego gap on its own scale only")
    ok(X.metric({"family": "percent"}, {"value": 150.0}, li) is None and X.metric({"family": "fermi"}, {"value": 1000.0}, li) == 3.0,
       "metric: percent bounds, Fermi log10")

    # full preview through the launcher
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = li.main(["--preview", "--experiment", "e6"])
    out = buf.getvalue()
    run = next((l.split("Data: ")[1].strip() for l in out.splitlines() if l.startswith("Data: ")), None)
    ok(rc in (0, None) and run and "E6-10" in out, "preview: launcher runs E6 end to end")
    if run:
        d = os.path.join(ROOT, run)
        summ = json.load(open(os.path.join(d, "summary.json")))
        man = json.load(open(os.path.join(d, "run_manifest.json")))
        ok(len(summ["predictions"]) == 10 and man["plugin"]["name"] == "e6" and man["pod_plugins"] == "e4,e5,e6,heads",
           "preview: summary and manifest")
        shutil.rmtree(d)


def tiny():
    os.environ["SI_INTERP_DEVICE"] = "cpu"
    os.environ["SI_INTERP_REHEARSAL"] = "1"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    import launch_interp as li
    import model_io
    import tasks
    import tasks_e6 as E6
    import interp_e6_laptop as X
    X._LI[:] = [li]
    torch.manual_seed(0)
    model, tok, _info = model_io.load_text_only(TINY, dtype=torch.float32)
    ok(tasks.TASKS.get("e6_steer") is not None, "tasks loads the e6 plug-in")
    settings = json.load(open(os.path.join(ROOT, "prereg", "settings.json")))
    blocks = model_io.decoder_layers(model)
    nL = len(blocks)
    Ls = X.layer_plan(nL)
    ok(hasattr(blocks[0].mlp, "down_proj"), "Gemma 4 blocks have mlp.down_proj")
    fx = X.X5.build_items("seller")[:3]
    nonmoney = [v for v in li.build_e3_items(settings, 1)[0] if v["kind"] == "nonmoney"][:2]
    B = E6.e6_build(model, tok, nonmoney + fx, settings, MODEL, model_io, keep=Ls["keep"], head_layers=[], role="seller")
    ok(sum(1 for r in B if r.get("kind") == "fixture" and not r.get("skipped")) == 3, "build: three usable fixtures (stand-in prices)")
    D = E6.e6_dirs(model, tok, [{}], settings, MODEL, model_io, layers=Ls["stake"])
    v = E6._DIR[Ls["stake"][0]]
    ok(len(D) == len(Ls["stake"]) and set(v) == {"vR", "vS", "u", "rand"}
       and all(abs(float(v[k].norm()) - D[0]["norm"]) < 1e-3 * max(1, D[0]["norm"]) for k in v),
       "dirs: four vectors per layer at one norm")
    # probe, fit, census
    P = X.probe_items(2)
    R = E6.e6_probe(model, tok, P, settings, MODEL, model_io, mlp_layers=Ls["mlp"])
    ok(len(R) == len(P) and {r["last_token"] for r in R} == {"."} and len(E6._PROBE["resid"]) == len(P)
       and E6._PROBE["resid"][0].shape[0] == nL, "probe: one state per layer at the shared final '.'")
    ok(set(E6._PROBE["mlp"]) == set(Ls["mlp"]) and E6._PROBE["mlp"][Ls["mlp"][0]][0].shape[0] == model_io.decoder_layers(model)[0].mlp.down_proj.in_features,
       "probe: MLP activations = down_proj inputs")
    Fi = E6.e6_fit(model, tok, [{}], settings, MODEL, model_io, stakes_layers=Ls["stake"])[0]
    ok(len(Fi["layers"]) == nL and all(-1.0001 <= Fi["layers"][L]["transfer"]["digits"]["words"] <= 1.0001 for L in range(nL))
       and Fi["layers"][Ls["stake"][0]].get("cos_digits_vR") is not None, "fit: transfer and cosines per layer")
    C = E6.e6_census(model, tok, [{}], settings, MODEL, model_io)[0]
    c0 = C["layers"][str(Ls["mlp"][0])]
    ok(c0["d_mlp"] > 0 and len(c0["top"]) > 0 and set(c0["top"][0]["rho"]) == set(X.NOTATIONS), "census: counts and top neurons")
    # steering: choice and gen readouts, hooks fire once on the prefill only
    Fm = X.family_items(fx, 2)
    for fam in ("risk", "percent", "nego"):
        out = E6.e6_steer(model, tok, Fm[fam][:1], settings, MODEL, model_io, plan=Ls["plan"], n_none=X.N_NONE, batch=X.BATCH)
        r = out[0]
        ok(len(r["rows"]) == X.BATCH and all(n == 1 for n in r["hooks_fired"]) and r["n_steer"] > len(tok(Fm[fam][0]["prefix"],
           add_special_tokens=False)["input_ids"]), f"{fam}: 16 rows; each steered layer's hook fired once; header + prefix steered")
        none = [x for x in r["rows"] if x["vector"] == "none"]
        key = "lp" if fam == "risk" else "text"
        ok(all(x[key] == none[0][key] for x in none), f"{fam}: plain rows identical")
        if fam == "risk":
            ok(any(x["lp"] != none[0]["lp"] for x in r["rows"] if x["vector"] == "vR"), "risk: steering changes the log-probs")
        if fam == "nego":
            ok(r["loo"] == ["True"], "nego: leave-one-fixture-out vectors")
    # parse_number
    ok(E6.parse_number(" 12}") == 12 and E6.parse_number(' 1,250,000, "x"') == 1250000 and E6.parse_number("none") is None,
       "parse_number")
    # encode_chat: the steered span is the model-turn header + prefix
    ids, n = E6.encode_chat(tok, model_io, [{"role": "user", "content": "Hi"}], '{"choice": "')
    tail = tok.decode(ids[-n:])
    ok("model" in tail and tail.endswith('{"choice": "'), f"encode_chat: steered span {tail!r}")


def main():
    if "--torch" in sys.argv:
        tiny()
    else:
        laptop()
        if os.path.exists(VENV) and os.path.exists(TINY):
            p = subprocess.run([VENV, os.path.abspath(__file__), "--torch"], capture_output=True, text=True)
            sys.stdout.write(p.stdout)
            sys.stdout.write(p.stderr[-3000:] if p.returncode else "")
            for l in p.stdout.splitlines():
                N["ok" if l.startswith("ok") else "fail"] += l.startswith(("ok", "FAIL"))
            if p.returncode:
                N["fail"] += 1
        else:
            ok(False, f"rehearsal venv / tiny model missing ({VENV}, {TINY})")
    print(f"\n{N['ok']} passed, {N['fail']} failed")
    return 1 if N["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
