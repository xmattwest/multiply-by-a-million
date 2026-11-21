#!/usr/bin/env python3
"""test_interp_concepts.py: checks for the concepts plug-in (scripts/interp/tasks_concepts.py on the pod;
scripts/interp_concepts_laptop.py behind the shim scripts/interp/exp_concepts.py).

    python3 tests/test_interp_concepts.py           laptop checks (stdlib), then the tiny-model checks in the rehearsal venv
    /tmp/si-interp-venv/bin/python tests/test_interp_concepts.py --torch      the tiny-model checks only

Laptop: one fixture per distinct floor, discovery / held-out floors disjoint; the concept pairs (counts, the 'large' side,
shared suffixes); row specs; the SAE gate and its hand-overs; pins; the time estimate; a full --preview session.
Tiny model (random-weight Gemma 4 of the 31B's class, fp32 on the CPU):
  - every concept pair has a shared token suffix; the 'count' direction = E4's non-money vector u on the same pairs;
  - the stored N6 - N1 differences and ld = E4's; the span share >= every single concept's cos^2;
  - forced-opening patch: vN row = E4's 'vec' row; alpha 0 = clean; real-opening steering (greedy) = E4's;
  - an exact SAE (f = relu([x, -x]), decode = f [I; -I]) splices in losslessly; 'all latents' = 'all + error', and
    'all + error' on the aligned tokens = E4's residual 'set'; no latents = clean; the attribution's gradient matches a
    finite difference; the two file loaders read the published key names; fetch reuses a checked file and refuses
    other hosts.
"""
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


def ok(cond, name):
    N["pass" if cond else "fail"] += 1
    print(("ok    " if cond else "FAIL  ") + name)


class A_:
    fixtures, batch, budget_min, model = 30, 16, 60.0, "gemma-4-31b-it"
    concepts_discovery, concepts_steer_samples, concepts_no_adam = 8, 3, False


def laptop():
    import launch_interp as li
    import interp_concepts_laptop as X
    X._LI[:] = [li]
    settings = json.load(open(os.path.join(ROOT, "prereg", "settings.json")))

    fx = X.distinct_fixtures(li, settings, 30)
    floors = [it["base_limit"] for it in fx]
    ok(len(fx) == 17 and len(set(floors)) == 17, f"17 distinct floors among Y01-Y30 ({sorted(floors)})")
    it = fx[3]
    ok(it["prompts"]["N1"]["messages"] == li.render_item("seller", it["base_limit"], "D2", settings)
       and it["prompts"]["N6"]["messages"] == li.render_item("seller", it["base_limit"], "D8", settings)
       and it["prompts"]["N6"]["limit_text"] == str(it["base_limit"] * 10 ** 6), "N1/N6 = the digits probe's D2/D8 seller prompts")
    plug = X.setup(A_(), settings, {"type": "object"}, li)
    ctx = plug["ctx"]
    fl = {it["fixture"]: it["base_limit"] for it in ctx["fx"]}
    ok(len(ctx["disc"]) == 8 and len(ctx["held"]) == 9 and not {fl[f] for f in ctx["disc"]} & {fl[f] for f in ctx["held"]},
       f"discovery 8 / held-out 9 floors, disjoint ({ctx['disc']} | {ctx['held']})")

    P = X.concept_pairs(li.E3_NONMONEY)
    cnt = {c: sum(1 for x in P if x[0] == c) for c in X.CONCEPTS}
    ok(cnt == {"count": 40, "money": 40, "million": 40, "digits": 40, "stakes": 20, "modesty": 20}, f"pairs per concept {cnt}")

    def suffix(a, b):
        k = 0
        while k < min(len(a), len(b)) and a[-1 - k] == b[-1 - k]:
            k += 1
        return k
    ok(all(a != b and suffix(a, b) >= 6 for _c, a, b in P), "every pair differs and shares >= 6 characters at the end")
    big = all(any(str(v * 10 ** 6) in b and str(v * 10 ** 6) not in a and str(v) in a for v in X.BASES)
              for c, a, b in P if c in ("count", "money", "digits"))
    ok(big and all(" million dollars" in b and " million" not in a for c, a, b in P if c == "million")
       and all(any(m in b for _bold, m in X.MODEST_P) for c, a, b in P if c == "modesty")
       and all(any(h in b for _lo, h in X.STAKES_P) for c, a, b in P if c == "stakes"), "b is the 'large' / high-stakes / modest side")
    ok(all(" dollars" in a for c, a, b in P if c in ("money", "million")) and not any("$" in a + b or "," in a.split("dollars")[0][-12:]
                                                                                  for c, a, b in P if c in ("money", "million")),
       "money is written '<digits> dollars' (no '$', no commas), as in the game")
    cx = ctx["cx"]
    ok(len(cx) == 200 and len({x["item_id"] for x in cx}) == 200 and all(sum(1 for x in cx if x["concept"] == c and x["half"] == h) >= 10
                                                                         for c in X.CONCEPTS for h in (0, 1)), "200 concept items, both halves filled")

    Ls = X.layer_sets(60)
    ok(Ls["concept"] == [30, 36, 40, 41, 42, 44, 48] and Ls["keep"] == [30, 36, 40, 41, 42, 44, 48] and Ls["decode"] == 30
       and Ls["adam"] == {"adam41": 41, "adam44": 44}, "31B layers")
    T6 = X.layer_sets(6)
    ok(T6["mapped"] and all(0 <= L < 6 for L in T6["concept"]) and T6["steer"] in T6["keep"], f"6-layer mapping {T6['concept']}")
    R = X.patch_rows(Ls)
    ok({t: len(v) for t, v in R.items()} == {"N1": 56, "N6": 56} and all(len({r["n"] for r in v}) == len(v) for v in R.values())
       and all(r["ops"][0][3] == 1.0 for r in R["N1"]) and all(r["ops"][0][3] == -1.0 for r in R["N6"]), "patch rows: 8 vectors x 7 layers, signs")
    top = [[1000 + j, 1.0 / (1 + j), 1.0] for j in range(64)]
    S = X.sae_rows(top)
    names = [r["n"] for r in S]
    ok(len(names) == len(set(names)) == 10 and S[3]["latents"] == list(range(1000, 1064)) and S[-1]["latents"]["exclude"] == list(range(1000, 1064)),
       f"SAE rows {names}")
    ok(X.url(X.DECODE) == "https://huggingface.co/decoderesearch/gemma-4-saes/resolve/bb8e6b333def5cf2667f2cd4b9e67258dae7029b/"
       "gemma-4-31b/btk-mat-layer-30-k-100/sae_weights.safetensors" and X.url(X.ADAM[0]).startswith(
           "https://huggingface.co/datasets/Adam1010/gemma-4-31b-sae-features/resolve/b690a06a87cff055dc997f60a71ce62b7fdd0164/weights/sae_layer_41.pt"),
       "pinned SAE URLs (revision commits, not branches)")
    ok(ctx["estimate_min"][1] < 50, f"time estimate {ctx['estimate_min']} min (90-min lifetime, ~25 min boot)")

    def qrec(kept, reh=False):
        P_ = {k: {"ld": v, "ld_spliced": v * kept, "pk": 0.8, "pk_spliced": 0.7, "stats": {"reply": {"rel_err": 0.1, "fvu_local": 0.3, "cos": 0.9, "l0": 90}}}
              for k, v in (("N1", -10.0), ("N6", 10.0))}
        return [{"item_id": "x", "fixture": "Y01", "prompts": P_, "rehearsal": reh}]
    for kept, want in ((0.9, True), (0.5, False), (1.4, False)):
        c2 = X.setup(A_(), settings, {"type": "object"}, li)["ctx"]
        X.after_step("sae30 quality", {"sae30 quality": qrec(kept)}, c2)
        skipped = all(c2["kw"][k]["skip"] for k in ("attr", "pick", "sae_patch", "profile"))
        ok(c2["gate"]["pass"] == want and skipped == (not want), f"SAE gate at kept {kept:.0%}: {'pass' if want else 'fail, SAE steps skipped'}")
    c2 = X.setup(A_(), settings, {"type": "object"}, li)["ctx"]
    X.after_step("sae30 quality", {"sae30 quality": qrec(0.1, reh=True)}, c2)
    ok(not c2["kw"]["attr"]["skip"], "rehearsal: the gate is reported, not applied")
    pr = c2["kw"]["sae_patch"]["rows"]
    X.after_step("sae30 pick", {"sae30 pick": [{"item_id": "pick", "top": top}]}, c2)
    ok(c2["kw"]["sae_patch"]["rows"] is pr and len(pr) == 10 and c2["kw"]["profile"]["latents"] == list(range(1000, 1064)),
       "the pick fills the patch rows and the profile latents in place")
    st_ = plug["steps"](ctx)
    ok([s[0] for s in st_] == ["concepts build", "concepts stats", "concepts patch", "concepts steer", "sae30 load", "sae30 quality",
                               "sae30 attr", "sae30 pick", "sae30 patch", "sae30 profile", "adam load", "adam41 quality", "adam44 quality"]
       and st_[8][3] is ctx["kw"]["sae_patch"], "step order; later steps hold the same kw dicts after_step edits")
    import interp_concepts2_laptop as X2
    ok(li.pod_plugins("concepts2", X2.POD_PLUGINS) == "concepts,e4,heads", "concepts2 ships tasks_concepts (POD_PLUGINS) and its imports")
    R2 = X2.patch_rows(X2.layer_sets(60))
    ok(len(R2["N1"]) == 24 and {x["ops"][0][4] for x in R2["N6"]} >= {"abl6_N6", "ablv_N6", "ablr_N6", "perp", "par"}
       and all(x["ops"][0][3] == (1.0 if x["n"].split("@")[0] in X2.ABLS else -1.0) for x in R2["N6"]),
       "concepts2 rows: 5 vectors + 3 ablations x 3 layers; ablations use the target's own state, unsigned")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "launch_interp.py"), "--preview", "--experiment", "concepts2"],
                       capture_output=True, text=True, timeout=300, cwd=ROOT)
    run = next((l.split()[1] for l in r.stdout.splitlines() if l.startswith("Data: ")), None)
    s2 = json.load(open(os.path.join(ROOT, run, "summary.json"))) if run else {}
    ok(r.returncode == 0 and abs(s2["ablation"]["ablr@44"]["kept"] - 1.0) < 1e-9 and abs(s2["additivity"]["N1"]["44"]) < 1e-9
       and s2["patch_flips"]["N1"]["perp@44"]["flips"] == 17, "concepts2 --preview: ablation, additivity, flips, report")
    if run:
        shutil.rmtree(os.path.join(ROOT, run))
    plugs = li.pod_plugins("concepts")
    need = {s[1] for s in st_}
    ok(plugs == "concepts,e4,heads" and need <= li.shipped_tasks(plugs) and not need <= li.shipped_tasks("none"),
       f"the pod code ships tasks_concepts + what it imports ({plugs}) and has every task the steps call")

    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "launch_interp.py"), "--preview", "--experiment", "concepts"],
                       capture_output=True, text=True, timeout=300, cwd=ROOT)
    run = next((l.split()[1] for l in r.stdout.splitlines() if l.startswith("Data: ")), None)
    s = json.load(open(os.path.join(ROOT, run, "summary.json"))) if run else {}
    ok(r.returncode == 0 and not s.get("stopped") and "analysis_error" not in s and s["gate"]["pass"] and s["patch"]["N1"]["vN@44"]["share"] > 0.9
       and s["sae_patch"]["N6"]["all + error"]["share"] > 0.99 and len(s["profile"]) == 64, "--preview session: all steps, summary, report")
    if run:
        shutil.rmtree(os.path.join(ROOT, run))


# ======================================================================================================
# tiny model
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
    import tasks_concepts as C
    import tasks_e4 as E
    import interp_concepts_laptop as X
    import interp_e4_laptop as X4
    torch.manual_seed(0)
    model, tok, info = model_io.load_text_only(TINY, dtype=torch.float32)
    name = "gemma-4-31b-it"
    ok(tasks.TASKS.get("concepts_build") is not None and tasks.TASKS.get("sae_patch") is not None, "tasks loads the concepts plug-in")
    settings = json.load(open(os.path.join(ROOT, "prereg", "settings.json")))
    X._LI[:] = [li]
    X4._LI[:] = [li]
    blocks = model_io.decoder_layers(model)
    nL, H = len(blocks), int(blocks[0].self_attn.config.hidden_size)
    Ls = X.layer_sets(nL)
    SL = Ls["steer"]
    fx = X.distinct_fixtures(li, settings, 3)
    cx = X.concept_items(li)

    # 1. pairs
    bad = []
    for x in cx:
        try:
            ia, ib, pa, pb = C.pair_encode(tok, x["a"], x["b"])
            if [ia[p] for p in pa] != [ib[p] for p in pb]:
                bad.append(x["item_id"])
        except ValueError as e:
            bad.append(f"{x['item_id']}: {e}")
    ok(not bad, f"all {len(cx)} concept pairs have a shared token suffix ({bad[:3]})")
    ia, ib, pa, pb = C.pair_encode(tok, "There are 76 stars visible in this photograph.", "There are 76000000 stars visible in this photograph.")
    ok(tok.decode([ia[p] for p in pa]) == " stars visible in this photograph.", "the count pair's suffix is the text after the number")

    # 2. build vs E4
    sub = [x for x in cx if x["concept"] == "count"][:4] + [x for x in cx if x["concept"] != "count"][::9]
    B = C.concepts_build(model, tok, fx + sub, settings, name, model_io, keep=Ls["keep"])
    Bf = [r for r in B if r["kind"] == "fixture"]
    ok(len(Bf) == 3 and all(not r.get("skipped") and r["stand_in"] for r in Bf) and Bf[0]["n_A"] == 117, "build: 3 floors (stand-in prices)")
    e4fx = X4.build_items(li, settings, 3)
    vec, _steer = li.build_e3_items(settings, 3)
    nonmoney = [v for v in vec if v["kind"] == "nonmoney"][:4]
    E4L = X4.layer_sets(nL)
    EB = E.e4_build(model, tok, nonmoney + e4fx, settings, name, model_io, keep=E4L["keep"], head_layers=E4L["heads"])
    Lc = E4L["keep"][0]
    w = max(float((C._S["dN"][f][Ls["keep"].index(Lc)] - E._STORE["d"][f][E4L["keep"].index(Lc)]).abs().max()) for f in C._S["dN"])
    eb = {r["fixture"]: r for r in EB if r.get("kind") == "fixture"}
    wl = max(abs(r["ld"][k] - eb[r["fixture"]]["ld"][k]) for r in Bf for k in ("N1", "N6"))
    ok(w < 1e-5 and wl < 1e-4, f"N6 - N1 at layer {Lc} and ld = E4's build (max |Δ| {w:.1e}, ld {wl:.1e})")
    u = E._STORE["u"][Lc] / E._STORE["u_n"]
    cnt = C._S["c"]["count"]
    cdir = cnt["sum"][Lc] / cnt["n"]
    ok(cnt["n"] == 4 and float((u - cdir).abs().max()) < 1e-5, "the 'count' direction on E3's first 4 pairs = E4's non-money vector u")

    # 3. stats
    S = C.concepts_stats(model, tok, [{"item_id": "stats"}], settings, name, model_io)[0]
    rep = S["per_segment"]["reply"]
    lo = all(rep["span_share"][L] + 1e-5 >= max(rep["cos"][c][L] ** 2 for c in S["concepts"]) for L in range(nL))
    ok(lo and all(0 <= x <= 1 + 1e-6 for x in rep["span_share"] + rep["span_share_random"]) and set(S["concepts"]) == set(X.CONCEPTS),
       "span share in [0, 1] and >= every single concept's cos^2")

    # 4. patch and steering vs E4
    it = fx[0]
    rows = {"N1": [{"n": "vN", "ops": [["vec", SL, "A", 1.0, "vN"]]}, {"n": "zero", "ops": [["vec", SL, "A", 0.0, "count"]]},
                   {"n": "cnt", "ops": [["vec", SL, "A", 1.0, "count"]]}], "N6": [{"n": "vN", "ops": [["vec", SL, "A", -1.0, "vN"]]}]}
    Pc = C.concepts_patch(model, tok, [it], settings, name, model_io, rows=rows, batch=4)[0]
    g = {x["n"]: x for x in Pc["targets"]["N1"]["rows"]}
    e4it = next(x for x in e4fx if x["fixture"] == it["fixture"])
    Pe = E.e4_patch(model, tok, [e4it], settings, name, model_io, rows={"N1": [{"n": "vN", "ops": [["vec", SL, "A", 1.0, "vN"]]}],
                                                                         "N6": [{"n": "vN", "ops": [["vec", SL, "A", -1.0, "vN"]]}]}, batch=4)[0]
    ge = {t: {x["n"]: x["ld"] for x in Pe["targets"][t]["rows"]} for t in ("N1", "N6")}
    w = max(abs(g["vN"]["ld"] - ge["N1"]["vN"]), abs(next(x["ld"] for x in Pc["targets"]["N6"]["rows"] if x["n"] == "vN") - ge["N6"]["vN"]))
    ok(w < 1e-4 and Pc["loo"] and Pc["targets"]["N1"]["hooks_ok"], f"vN rows = E4's 'vec' rows, both targets (max |Δ ld| {w:.1e})")
    ok(abs(g["zero"]["ld"] - g["clean"]["ld"]) < 1e-6 and abs(g["cnt"]["ld"] - g["clean"]["ld"]) > 1e-6, "alpha 0 = clean; a concept moves it")
    sit = X.steer_items([it])[:1]
    mine = C.concepts_steer(model, tok, sit, settings, name, model_io, plan=[["vN", SL, 1.0, 1]], n_none=8, batch=16, greedy=True, max_new=20)[0]
    theirs = E.e4_steer(model, tok, X4.steer_items([e4it], ("N1",), "t"), settings, name, model_io, plan=[["vN", SL, 1.0, 1]], n_none=8,
                        batch=16, greedy=True, max_new=20)[0]
    ok([x["content"] for x in mine["rows"]] == [x["content"] for x in theirs["rows"]] and mine["loo"], "steered real openings (greedy) = E4's")

    # 4b. concepts2's vectors: split of vN and the ablations
    V, _use = C._vectors(it["fixture"], SL)
    Cd = C._dirs()
    Qs = torch.linalg.qr(torch.stack([Cd[n][SL].float() for n in sorted(Cd)], 1))[0]
    nv = V["vN"].norm(dim=-1)
    ok(float((V["par"] + V["perp"] - V["vN"]).abs().max()) < 1e-4 * float(nv.max()) and float((V["perp"] @ Qs).abs().max()) < 1e-4 * float(nv.max())
       and float((V["par_s"].norm(dim=-1) - nv).abs().max()) < 1e-4 * float(nv.max()) and float((V["perp_s"].norm(dim=-1) - nv).abs().max()) < 1e-4 * float(nv.max()),
       "par + perp = vN; perp is orthogonal to the concepts' span; par_s / perp_s have vN's per-token size")
    i_ = Ls["keep"].index(SL)
    xs = {k: C._S["x"][it["fixture"]][k][i_].float() for k in ("N1", "N6")}
    u_ = V["vN"] / V["vN"].norm(dim=-1, keepdim=True)
    sc = max(float(x.norm(dim=-1).max()) for x in xs.values())
    ok(all(float(((xs[k] + V[f"abl6_{k}"]) @ Qs).abs().max()) < 1e-4 * sc and float(((xs[k] + V[f"ablv_{k}"]) * u_).sum(-1).abs().max()) < 1e-4 * sc
           and 0 < float(V[f"ablr_{k}"].norm()) < float(xs[k].norm()) for k in xs),
       "abl6 / ablv remove the run's own projection onto the concepts' span / onto vN's direction; ablr removes a little")
    x6 = C._capture(model, model_io, blocks, E.encode_fixture(tok, model_io, it)["N6"]["ids"], E.encode_fixture(tok, model_io, it)["N6"]["A"],
                    [("resid", SL)])[0][("resid", SL)].float()
    ok(float((x6 - xs["N6"]).abs().max()) < 1e-4 * sc, "the stored states are the runs' own states on A at that layer")
    import interp_concepts2_laptop as X2
    R2 = X2.patch_rows({"patch": [SL], "steer": SL})
    P2 = C.concepts_patch(model, tok, [it], settings, name, model_io, rows=R2, batch=8)[0]
    g2 = {t: {x["n"]: x for x in P2["targets"][t]["rows"]} for t in ("N1", "N6")}
    A2 = X2.ablation_table([P2], X4)
    ok(P2["targets"]["N1"]["hooks_ok"] and P2["targets"]["N6"]["hooks_ok"] and set(A2) == {f"{ab}@{SL}" for ab in X2.ABLS}
       and abs(g2["N1"][f"vN@{SL}"]["ld"] - g["vN"]["ld"]) < 1e-4,
       f"concepts2 rows run: {len(R2['N1'])} per target; ablation table has abl6 / ablv / ablr; the vN row repeats")

    # 5. SAEs
    L30 = Ls["decode"]
    I = torch.eye(H)
    exact = C.JumpReLUSAE(torch.cat([I, -I], 1), torch.zeros(2 * H), torch.cat([I, -I], 0), torch.zeros(H), torch.zeros(2 * H))
    C._S["sae"]["exact"] = exact
    Q = C.sae_quality(model, tok, [it], settings, name, model_io, sae="exact", layer=L30)[0]
    ok(all(abs(p["ld_spliced"] - p["ld"]) < 1e-4 and p["stats"]["reply"]["rel_err"] < 1e-10 for p in Q["prompts"].values()),
       "an exact SAE splices in losslessly")
    srows = [{"n": "allerr aligned", "latents": "all+err", "where": "aligned"}, {"n": "all", "latents": "all"},
             {"n": "allerr", "latents": "all+err"}, {"n": "none", "latents": []}, {"n": "r", "latents": {"random": 3, "seed": "t", "exclude": []}}]
    Sp = C.sae_patch(model, tok, [it], settings, name, model_io, sae="exact", layer=L30, rows=srows, batch=8)[0]
    gs = {t: {x["n"]: x["ld"] for x in Sp["targets"][t]["rows"]} for t in ("N1", "N6")}
    Pset = E.e4_patch(model, tok, [e4it], settings, name, model_io, rows={"N1": [{"n": "set", "ops": [["set", "resid", L30, -1, "A", "N6"]]}]}, batch=4)[0]
    rset = next(x["ld"] for x in Pset["targets"]["N1"]["rows"] if x["n"] == "set")
    ok(abs(gs["N1"]["allerr aligned"] - rset) < 1e-4, f"'all + error' on the aligned tokens = E4's residual 'set' at layer {L30}")
    ok(all(abs(gs[t]["all"] - gs[t]["allerr"]) < 1e-4 and abs(gs[t]["none"] - gs[t]["clean"]) < 1e-6 for t in gs)
       and Sp["targets"]["N1"]["n_latents"]["r"] == 3, "exact SAE: 'all latents' = 'all + error'; no latents = clean; random picks 3")
    for p in model.parameters():
        p.requires_grad_(False)
    E_ = E.encode_fixture(tok, model_io, it)
    e = E_["N6"]
    pre, ta, tb = C._S["neg"][it["fixture"]]["cand"]["x1e6"]
    ld, f, gf, err, gerr = C._attr_forward(model, model_io, blocks, e["ids"], pre, L30, e["s0"], exact, ta, tb)
    p, i = divmod(int(gf.abs().argmax()), gf.shape[1])
    x = C._capture(model, model_io, blocks, e["ids"], list(range(e["s0"], len(e["ids"]))), [("resid", L30)])[0][("resid", L30)].float()
    eps = 0.02 / max(1e-6, abs(float(gf[p, i])))          # ld moves ~0.02 nats: well above the 1e-5 rounding of ld
    xp = x.clone()
    xp[p] += eps * exact.W_dec[i]
    ldp = C._splice_ld(model, model_io, blocks, e["ids"], pre, L30, e["s0"], xp, ta, tb)[0]
    ldm = C._splice_ld(model, model_io, blocks, e["ids"], pre, L30, e["s0"], x - (xp - x), ta, tb)[0]
    fd = (ldp - ldm) / (2 * eps)
    ok(abs(gf[p, i]) > 1e-4 and abs(fd - float(gf[p, i])) < 0.05 * abs(fd) and float(err.abs().max()) < 1e-5,
       f"attribution gradient = finite difference ({float(gf[p, i]):+.5f} vs {fd:+.5f}); error term ~0 for the exact SAE")
    A1 = C.sae_attr(model, tok, [it], settings, name, model_io, sae="exact", layer=L30, top=5)[0]
    pk = C.sae_attr_pick(model, tok, [{"item_id": "pick"}], settings, name, model_io, sae="exact", k=4)[0]
    ok(abs(A1["err_aligned"]) < 1e-4 and len(A1["top"]) == 5 and len(pk["top"]) == 4 and len(pk["lens"]) == 4, "attribution and pick run")
    Pr = C.sae_profile(model, tok, cx[:2], settings, name, model_io, sae="exact", layer=L30, latents=[0, 1])
    ok(len(Pr) == 2 and len(Pr[0]["a"][0]) == 2, "profile runs")

    d = tempfile.mkdtemp()
    try:
        from safetensors.torch import save_file
        m = 8
        w = {"W_enc": torch.randn(H, m), "b_enc": torch.randn(m), "W_dec": torch.randn(m, H), "b_dec": torch.randn(H), "threshold": torch.rand(m)}
        save_file(w, os.path.join(d, "s.safetensors"))
        S1 = C.load_sae({"kind": "jumprelu"}, os.path.join(d, "s.safetensors"), "cpu")
        xx = torch.randn(5, H)
        ref = torch.relu((xx - w["b_dec"]) @ w["W_enc"] + w["b_enc"])
        ref = ref * (ref > w["threshold"])
        ok(float((S1.encode(xx) - ref).abs().max()) < 1e-5, "JumpReLU loader reads Decode's key names; encode = relu(pre) * (pre > threshold)")
        import collections
        wa = collections.OrderedDict(W_enc=torch.randn(H, m), b_enc=torch.randn(m), W_dec=torch.randn(m, H), b_dec=torch.randn(H),
                                     norm_mean=torch.randn(H), norm_std=torch.rand(H) + 0.5)
        torch.save(wa, os.path.join(d, "sae_final.pt"))
        os.rename(os.path.join(d, "sae_final.pt"), os.path.join(d, "sae_layer_44.pt"))
        S2 = C.load_sae({"kind": "topk", "k": 3}, os.path.join(d, "sae_layer_44.pt"), "cpu")
        f2 = S2.encode(xx)
        ok(int((f2 > 0).sum(-1).max()) <= 3 and S2.decode(f2).shape == xx.shape, "TopK loader reads Adam1010's key names; at most k active")
        path = os.path.join(d, "blob.bin")
        open(path, "wb").write(b"abc" * 1000)
        got = C.fetch("https://huggingface.co/x/y/resolve/0/blob.bin", path, 3000, C.file_sha256(path))
        try:
            C.fetch("https://example.com/blob.bin", path, 3000, "0" * 64)
            refused = False
        except ValueError:
            refused = True
        ok(got["downloaded"] is False and refused, "fetch reuses a file that checks out (no network) and refuses other hosts")
    finally:
        shutil.rmtree(d)
    L = C.sae_load(model, tok, [{"item_id": "s", "name": "syn", "kind": "jumprelu"}, {"item_id": "t", "name": "syn2", "kind": "topk", "k": 64}],
                   settings, name, model_io)
    ok(all(r.get("synthetic") and r["d_in"] == H for r in L), "rehearsal: synthetic SAEs of the model's width (no download)")


def main(argv):
    if "--torch" in argv:
        tiny()
    else:
        laptop()
        if os.path.exists(VENV_PY) and os.path.exists(os.path.join(TINY, "config.json")):
            r = subprocess.run([VENV_PY, os.path.abspath(__file__), "--torch"], capture_output=True, text=True, timeout=1800)
            print(r.stdout, end="")
            if r.returncode or "Traceback" in r.stderr:
                print(r.stderr[-3000:])
            N["pass"] += r.stdout.count("\nok    ") + r.stdout.startswith("ok    ")
            N["fail"] += r.stdout.count("FAIL  ") + (1 if r.returncode else 0)
        else:
            ok(False, "rehearsal venv / tiny model missing: bash scripts/interp/make_rehearsal_model.sh")
    print(f"\n{N['pass']} passed, {N['fail']} failed")
    return 1 if N["fail"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
