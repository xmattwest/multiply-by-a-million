#!/usr/bin/env python3
"""test_interp_buyers.py: checks of the buyers plug-in (MI Session 2). Laptop part: standard library only.

  - buyer items are the vLLM digits run's buyer items (same ids, prompts differ only in the limit),
  - the buyer gate (plan §2 on the buyer discount) passes and fails as it should,
  - the buyer on-scale window, the A and B tables (fractions AND points; off-scale excluded),
  - the E2 peak rule and the layer choices made in after_step,
  - the row tables sent by the laptop equal the pod plug-in's defaults; no task-name clash,
  - --preview runs end to end, and stops at the gate on a failing canned E0.
  - If the rehearsal venv and tiny model exist (bash scripts/interp/make_rehearsal_model.sh):
    fp32 checks on the tiny Gemma 4 with the real tokenizer: buyer prompts align token for token;
    seller and buyer have the same post-limit length and reply tokens; steering the buyer x1 prompt
    with its own full difference equals E2's patch (logits, and the tasks' prices); seller_norm and
    seller_perp are what they say.

Run: python3 tests/test_interp_buyers.py
"""
import argparse
import ast
import glob
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "interp"))
import launch_interp as li          # noqa: E402
import interp_buyers_laptop as eb    # noqa: E402  (scripts/interp/exp_buyers.py re-exports it)

N = {"pass": 0, "fail": 0}
VLLM = os.path.join(ROOT, "data", "explore", "X-digits-gemma-4-31b-it-20261004T042802Z")
VENV_PY, TINY = "/tmp/si-interp-venv/bin/python", "/tmp/si-tiny-31b"


def ok(cond, name):
    N["pass" if cond else "fail"] += 1
    print(("ok    " if cond else "FAIL  ") + name)


def recs_e0(d2, d8, n_fix=4, per=2):
    return [{"item_id": f"Y{f:02d}-B-{c}-s{s}", "role": "buyer", "condition": c, "limit_units": 100, "errors": [],
             "action": "OFFER", "price": round(100 * (1 - g)), "gap": g, "constrained": True}
            for f in range(1, n_fix + 1) for s in range(1, per + 1) for c, g in (("D2", d2 + 0.01 * f), ("D8", d8 + 0.01 * f))]


class Panel:
    def __init__(self):
        self.notes, self.steps = [], [("E2", 4), ("dirs 2nd", 10), ("A 2nd", 4), ("B", 4)]

    def note(self, s):
        self.notes.append(s)


def fake_ctx(window="40-48", layer=44):
    a = argparse.Namespace(steer_layer=layer, buyers_same_window=window, buyers_layer_step=2, batch=16, fixtures=3,
                           buyers_second_layer=None,
                           samples=1, model="gemma-4-31b-it", buyers_preview_fail=False, preview=True)
    settings = json.load(open(os.path.join(ROOT, "prereg", "settings.json")))
    plug = eb.setup(a, settings, {}, li)
    _, e0 = li.build_items(settings, a.fixtures, a.samples, list(li.CONDITIONS), ("buyer",))
    ctx = {**plug["ctx"], "panel": Panel(), "vllm_recs": [], "sampled_items": e0, "greedy_items": e0, "schema": {}}
    return plug, ctx


def e2rec(fid, direction, layers, frac, slip_layer=None):
    src, tgt = ("D2", "D8") if direction == "x1->x1e6" else ("D8", "D2")
    lo, hi = (0.37, 0.46) if direction == "x1->x1e6" else (0.46, 0.37)
    unit = 10 ** 6 if tgt == "D8" else 1
    rows = []
    for L in layers:
        g = lo + frac(L) * (hi - lo)
        price = round((1 - g) * 100 * unit)
        if L == slip_layer:
            price = round(0.05 * 100 * unit)                  # a lost digit: off scale
        rows.append({"layer": L, "pos": "post", "price": price})
    rows.append({"layer": "clean", "pos": None, "price": round((1 - lo) * 100 * unit)})
    return {"fixture": fid, "role": "buyer", "direction": direction, "base_limit": 100, "source_condition": src,
            "target_condition": tgt, "clean_consistent": True, "rows": rows}


def main():
    settings = json.load(open(os.path.join(ROOT, "prereg", "settings.json")))
    # --- items -------------------------------------------------------------------------------------
    _, items = li.build_items(settings, 30, 4, list(li.CONDITIONS), ("buyer",))
    ok(len(items) == 240 and all(it["role"] == "buyer" for it in items), "E0: 240 buyer items (30 x 4 x D2/D8)")
    if os.path.isdir(VLLM):
        vl = {r["item_id"] for r in li.load_vllm(VLLM, li.CONDITIONS) if r["role"] == "buyer"}
        ok({it["item_id"] for it in items} == vl, "E0: the same 240 buyer item ids as the vLLM digits run")
    else:
        print("skip  vLLM digits run not present")
    by = {}
    for it in items:
        by.setdefault((it["fixture"], it["sample"]), {})[it["condition"]] = it
    same = all(p["D2"]["messages"][0] == p["D8"]["messages"][0] and p["D2"]["seed"] == p["D8"]["seed"]
               and p["D2"]["messages"][1]["content"].replace(f"ceiling is {p['D2']['limit_units']} dollars.", "#", 1)
               == p["D8"]["messages"][1]["content"].replace(f"ceiling is {p['D8']['limit_units']} dollars.", "#", 1)
               for p in by.values())
    ok(same, "E0: buyer D2/D8 prompts differ only in the limit; seeds shared")
    e2 = li.build_e2_items(settings, 30, role="buyer")
    _, steer = li.build_e3_items(settings, 30, role="buyer")
    ok(all(x["role"] == "buyer" and "ceiling" in x["target"]["messages"][1]["content"] for x in e2 + steer)
       and [x["fixture"] for x in steer][:4] == ["Y01", "Y01", "Y02", "Y02"], "E2/E3 items: buyers, fixture-major")

    # --- the gate --------------------------------------------------------------------------------
    good = eb.buyer_fidelity(li, recs_e0(0.46, 0.37), recs_e0(0.46, 0.37))
    ok(good["pass"] and abs(good["shift_pod"] - (-9.0)) < 1e-9 and good["fixtures_negative"] == 4,
       "gate: passes when the pod equals vLLM")
    ok(not eb.buyer_fidelity(li, recs_e0(0.46, 0.42), recs_e0(0.46, 0.37))["pass"], "gate: fails when the shift halves")
    near = eb.buyer_fidelity(li, recs_e0(0.48, 0.39), recs_e0(0.46, 0.37))
    ok(near["pass"] and near["gaps_within_3"], "gate: 2-point offset at both scales, same shift -> pass")
    off = eb.buyer_fidelity(li, recs_e0(0.50, 0.41), recs_e0(0.46, 0.37))
    ok(not off["pass"] and off["shift_within_25pct"] and not off["gaps_within_3"], "gate: 4-point offset -> fail")
    ok(not eb.buyer_fidelity(li, [], recs_e0(0.46, 0.37))["pass"], "gate: no pod data -> fail")
    pc = Panel()
    why = eb.after_step("sampled", {"sampled": recs_e0(0.46, 0.44)}, {"buyers": {"li": li, "done": set()},
                                                                       "vllm_recs": recs_e0(0.46, 0.37), "panel": pc})
    ok(isinstance(why, str) and "FAILED" in why, "gate: after_step stops the session on a fail")
    why = eb.after_step("sampled", {"sampled": recs_e0(0.46, 0.44)}, {"buyers": {"li": li, "done": set(), "rehearsal": True},
                                                                       "vllm_recs": recs_e0(0.46, 0.37), "panel": pc})
    ok(why is None and "not enforced" in pc.notes[-1], "gate: not enforced on a REHEARSAL pod (random weights)")

    # --- scale window and tables -----------------------------------------------------------------------
    old = li.ON_SCALE
    with eb.buyer_scale(li):
        g, m, offs = li.scale_free("buyer", 45, 100)
        g6, m6, _ = li.scale_free("buyer", 45_000_000, 100)
        slip = li.scale_free("buyer", 5_000_000, 100)[2]
    ok(abs(g - 0.55) < 1e-9 and m == 0 and not offs and m6 == 6 and slip and li.ON_SCALE == old,
       "buyer window: 0.45x ceiling is on scale (gap 55%), 0.05x is off scale; core window restored")
    ok(li.scale_free("buyer", 45, 100)[2] is True, "(the core seller window would have dropped that buyer opening)")

    def brec(fid, d, g0, rows):
        own = "D2" if d == "x1+v" else "D8"
        unit = 1 if own == "D2" else 10 ** 6
        mk = lambda v, sc, a, g: {"vector": v, "scope": sc, "alpha": a,          # noqa: E731
                                  "price": None if g is None else round((1 - g) * 1000 * unit)}
        return {"fixture": fid, "role": "buyer", "direction": d, "base_limit": 1000, "target_condition": own,
                "clean_consistent": True, "rows": [mk("none", "all", 0.0, g0)] + [mk(*r) for r in rows]}
    fx = ("Y01", "Y02", "Y03")
    A = eb.bridge_table(li, [brec(f, "x1+v", 0.46, [("seller", "all", 1.0, 0.415), ("v_pos", "all", 1.0, 0.95)])
                             for f in fx] + [brec(f, "x1e6-v", 0.37, [("seller", "all", 1.0, 0.37)]) for f in fx])
    c = A["cells"]["x1+v"]["seller|all|1.0"]
    ok(abs(c["fraction"] - 0.5) < 0.02 and abs(c["points"] - (-4.5)) < 0.2 and abs(c["denominator_pts"] + 9) < 0.2,
       "A table: half the gap = 0.5 and -4.5 points")
    ok(A["cells"]["x1+v"]["v_pos|all|1.0"]["off_scale"] == 3 and A["cells"]["x1+v"]["v_pos|all|1.0"]["fraction"] is None,
       "A table: off-scale readings excluded, not averaged")
    ok(abs(A["cells"]["x1e6-v"]["seller|all|1.0"]["fraction"]) < 0.02, "A table: no move = 0")

    def srec(fid, d, g0, g1, aligned=True, n=4):
        own = "D2" if d == "x1+v" else "D8"
        unit = 1 if own == "D2" else 10 ** 6
        row = lambda v, a, g, al: {"vector": v, "scope": "prompt+reply", "alpha": a, "errors": [], "action": "OFFER",  # noqa: E731
                                   "finish": "stop", "aligned": al, "price": round((1 - g) * 1000 * unit)}
        return {"fixture": fid, "role": "buyer", "direction": d, "base_limit": 1000, "target_condition": own,
                "rows": [row("none", 0.0, g0, True)] * 8 + [row("v_pos", 1.0, g1, aligned)] * n
                + [row("seller", 1.0, 0.95, True)]}
    B = eb.steer_table(li, [srec(f, "x1+v", 0.46, 0.415, f != "Y03") for f in fx] + [srec(f, "x1e6-v", 0.37, 0.37) for f in fx])
    c = B["directions"]["x1+v"]["v_pos|prompt+reply"]["1.0"]
    ok(abs(c["share_moved"] - 0.5) < 0.02 and abs(c["points"] + 4.5) < 0.2 and abs(c["aligned_share"] - 2 / 3) < 1e-9
       and abs(B["unsteered_gap"]["x1+v"] - 46) < 0.1, "B table: share 0.5, -4.5 points, aligned share")
    s = B["directions"]["x1+v"]["seller|prompt+reply"]["1.0"]
    ok(s["off_scale"] == 3 and s["mean_gap"] is None, "B table: off-scale openings counted, not averaged")
    Ba = eb.steer_table(li, [srec(f, "x1+v", 0.46, 0.415, f != "Y03") for f in fx], aligned_only=True)
    ok(Ba["directions"]["x1+v"]["v_pos|prompt+reply"]["1.0"]["n_fixtures"] == 2, "B table: aligned-only filter")

    # --- E2 peak and the layer choices ------------------------------------------------------------------
    layers = li.layer_list(60, 2)
    plateau = lambda L: 0.9 if 42 <= L <= 46 else (0.95 if L == 30 else 0.1)            # noqa: E731
    recs = [e2rec(f, d, layers, plateau, slip_layer=30) for f in fx for d in ("x1->x1e6", "x1e6->x1")]
    pk, _ = eb.e2_peak(li, recs, layers)
    ok(pk["layer"] == 42 and abs(pk["mean_fraction"] - 0.9) < 0.02, "E2 peak: first layer of the plateau; off-scale layer 30 not eligible")
    plug, ctx = fake_ctx()
    eb.after_step("E2", {"E2": recs}, ctx)
    h = ctx["buyers"]
    ok(h["layers"]["A"] == [44] and h["lists"]["dirs 2nd"] == [] and h["lists"]["A 2nd"] == []
       and dict(ctx["panel"].steps)["A 2nd"] == 0, "choice: peak inside 40-48 -> one layer, 2nd-layer steps emptied")
    late = lambda L: 0.95 if L == 30 else 0.1                                             # noqa: E731
    plug, ctx = fake_ctx()
    eb.after_step("E2", {"E2": [e2rec(f, d, layers, late) for f in fx for d in ("x1->x1e6", "x1e6->x1")]}, ctx)
    h = ctx["buyers"]
    ok(h["layers"]["A"] == [44, 30] and h["kw"]["A 2nd"]["layer"] == 30 and h["kw"]["dirs 2nd"]["layer"] == 30
       and len(h["lists"]["A 2nd"]) == 6, "choice: peak at 30 -> A also at layer 30")
    steps = plug["steps"](ctx)
    ok([s[0] for s in steps] == ["render", "sampled", "greedy", "teacher", "E2", "dirs 1st", "A 1st", "dirs 2nd",
                                 "A 2nd", "B"] and steps[8][3] is h["kw"]["A 2nd"] and steps[8][2] is h["lists"]["A 2nd"],
       "steps: fixed order; later steps share the kw dicts / item lists after_step edits")
    moved = lambda share: [brec(f, d, g0, [("v_pos", "all", 1.0, g0 + share * (g1 - g0))])      # noqa: E731
                           for f in fx for d, g0, g1 in (("x1+v", 0.46, 0.37), ("x1e6-v", 0.37, 0.46))]
    weak, strong = moved(0.2), moved(0.9)
    ok(abs(eb.a_moved(eb.bridge_table(li, weak)) - 0.2) < 0.02, "A summary used by the B-layer rule: 20% moved")
    for r in strong:
        r["layer"] = 30
    eb.after_step("A 2nd", {"A 1st": weak, "A 2nd": strong}, ctx)
    ok(h["layers"]["B"] == 30 and h["kw"]["B"]["layer"] == 30, "choice: A at 44 moves < 50% and 30 moves more -> B at 30")
    eb.after_step("A 2nd", {"A 1st": strong, "A 2nd": weak}, ctx)
    ok(h["layers"]["B"] == 44, "choice: A at 44 moves >= 50% -> B stays at 44")

    # --- pod plug-in: same row tables, no task-name clash (parsed, no torch here) ---------------------------
    src = open(os.path.join(ROOT, "scripts", "interp", "tasks_buyers.py")).read()
    tree = ast.parse(src)
    consts = {t.targets[0].id: ast.literal_eval(t.value) for t in tree.body
              if isinstance(t, ast.Assign) and isinstance(t.targets[0], ast.Name) and t.targets[0].id in ("BRIDGE_ROWS", "STEER_ROWS")}
    ok(consts.get("BRIDGE_ROWS") == eb.BRIDGE_ROWS and consts.get("STEER_ROWS") == eb.STEER_ROWS,
       "row tables: laptop == pod defaults")
    core = open(os.path.join(ROOT, "scripts", "interp", "tasks.py")).read()
    ok(all(f'"{t}"' not in core for t in ("buyers_build", "buyers_bridge", "buyers_steer")), "task names not in core tasks.py")
    ok("buyers" in li.load_plugins(), "plug-in discovered by launch_interp")

    # --- preview end to end: pass, and the failing gate ------------------------------------------------------
    for flags, name in (([], "pass"), (["--buyers-preview-fail"], "fail")):
        before = set(glob.glob(os.path.join(ROOT, "data", "interp", "interp-buyers-*-preview")))
        rc = li.main(["--preview", "--experiment", "buyers", "--fixtures", "3", "--samples", "2"] + flags)
        # Only this test's own preview folders: never touch another chat's runs (paid or rehearsal).
        new = sorted(d for d in set(glob.glob(os.path.join(ROOT, "data", "interp", "interp-buyers-*-preview"))) - before)
        try:
            summ = json.load(open(os.path.join(new[0], "summary.json"))) if new else {}
            man = json.load(open(os.path.join(new[0], "run_manifest.json"))) if new else {}
            inputs = [json.loads(l) for l in open(os.path.join(new[0], "inputs.jsonl"))] if new else []
            if name == "pass":
                ok(rc == 0 and len(new) == 1 and "interp-buyers-" in new[0] and summ["e0_buyer_fidelity"]["pass"]
                   and summ.get("steer", {}).get("n_fixtures") == 3 and summ.get("bridge") and summ["stopped"] is None
                   and man["roles"] == ["buyer"] and man["plugin"]["name"] == "buyers"
                   and sum(1 for x in inputs if x["step"] == "e0") == 12, "preview: runs end to end (buyers only)")
            else:
                ok(len(new) == 1 and "FAILED the gate" in (summ.get("stopped") or "") and not summ.get("e2")
                   and not summ.get("bridge") and not summ.get("steer"), "preview: a failing canned E0 stops at the gate")
        finally:
            for d in new:
                shutil.rmtree(d, ignore_errors=True)

    # --- website figures (tracked JSON only) ------------------------------------------------------------------
    import interp_buyers_figures as fb
    cards = fb.cards()
    if os.path.exists(fb.JSON_PATH):
        ok(len(cards) == 4 and all(c.count("<svg") == 1 and "nan" not in c.lower().replace("nance", "") for c in cards),
           "figures: 4 cards from config/interp/mi_figures_buyers.json, one SVG each, no NaN")
        sys.path.insert(0, os.path.join(ROOT, "scripts", "interp"))
        import mi_figures
        ok(sum("Buyers ·" in c for c in mi_figures.extra_cards()) == 4, "figures: picked up by mi_figures.extra_cards()")
    ok(fb.cards(os.path.join(ROOT, "no-such.json")) == [], "figures: no JSON -> no cards")

    # --- tiny model, fp32 ------------------------------------------------------------------------------------
    if os.path.exists(VENV_PY) and os.path.exists(os.path.join(TINY, "config.json")):
        r = subprocess.run(["perl", "-e", "alarm 900; exec @ARGV", VENV_PY, "-c", TINY_CHECK, ROOT, TINY],
                           capture_output=True, text=True, env={**os.environ, "SI_INTERP_DEVICE": "cpu",
                                                                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        lines = [l for l in r.stdout.splitlines() if l.startswith(("ok ", "FAIL "))]
        for l in lines:
            ok(l.startswith("ok "), "tiny fp32: " + l.split(" ", 1)[1])
        ok(r.returncode == 0 and len(lines) >= 8, f"tiny fp32 checks ran (exit {r.returncode})")
        if r.returncode != 0:
            print(r.stdout[-2000:], r.stderr[-3000:])
    else:
        print("skip  tiny-model checks (run: bash scripts/interp/make_rehearsal_model.sh)")

    print(f"\npassed: {N['pass']}  failed: {N['fail']}")
    return 1 if N["fail"] else 0


TINY_CHECK = r'''
import json, os, sys
root, tiny = sys.argv[1], sys.argv[2]
sys.path[:0] = [os.path.join(root, "scripts", "interp"), os.path.join(root, "scripts")]
import torch
import model_io, tasks as T, hooks
import launch_interp as li
torch.manual_seed(0)
def ok(c, name): print(("ok " if c else "FAIL ") + name)
model, tok, info = model_io.load_text_only(tiny, dtype=torch.float32)
settings = json.load(open(os.path.join(root, "prereg", "settings.json")))
L = 3

# 1. buyer prompts align token for token, all 30 fixtures; seller/buyer post-limit lengths and reply tokens.
e2b = li.build_e2_items(settings, 30, role="buyer")
e2s = li.build_e2_items(settings, 30, role="seller")
good, npost, reply = True, set(), set()
for it in e2b[::2]:
    S = T._r2_encode(tok, model_io, it["source"]["messages"], it["source"]["limit_text"], "buyer")
    G = T._r2_encode(tok, model_io, it["target"]["messages"], it["target"]["limit_text"], "buyer")
    s0 = G["span"][0]
    good &= S["span"][0] == s0 and S["ids"][:s0] == G["ids"][:s0] and S["ids"][S["span"][-1] + 1:] == G["ids"][G["span"][-1] + 1:]
    good &= len(S["span"]) + 6 == len(G["span"])
    post = G["ids"][G["span"][-1] + 1:]
    npost.add(len(post)); reply.add(tuple(post[-11:]))
ok(good, "buyer D2/D8 R2 inputs identical outside the limit digits (30 fixtures; D8 = 6 more digit tokens)")
spost = set()
for it in e2s[::2]:
    G = T._r2_encode(tok, model_io, it["target"]["messages"], it["target"]["limit_text"], "seller")
    p = G["ids"][G["span"][-1] + 1:]
    spost.add(len(p)); reply.add(tuple(p[-11:]))
ok(npost == spost == {117} and len(reply) == 1, f"seller and buyer: 117 tokens after the limit, one shared 11-token reply opening")

# 2. build both roles on 3 fixtures
sv, _ = li.build_e3_items(settings, 3, role="seller")
bv, steer = li.build_e3_items(settings, 3, role="buyer")
vec = [v for v in sv if v["kind"] == "pair"] + [v for v in bv if v["kind"] == "pair"] + [v for v in bv if v["kind"] == "nonmoney"][:4]
info = T.TASKS["buyers_build"](model, tok, vec, settings, "gemma-4-31b-it", model_io, layer=L)[0]
ok(info["cross_aligned"] and info["n_prompt"] == 106 and info["n_reply"] == 11 and T._STORE["e3b"] is T._STORE["buyers"]["buyer"][L]
   and T._STORE["buyers"]["seller"][L] is not T._STORE["buyers"]["buyer"][L], "build: both roles stored; core e3b slot = buyer set")
import tasks_buyers as TB
V, used = TB._vectors(L, "Y01")
s = T._STORE["buyers"]["seller"][L]
loo = (s["d"]["Y02"] + s["d"]["Y03"]) / 2
nb = V["v_pos"].norm(dim=-1)
ok(used and torch.allclose(V["seller"], loo, atol=1e-5), "seller direction is leave-one-fixture-out")
ok(torch.allclose(V["seller_norm"].norm(dim=-1), nb, rtol=1e-4)
   and float(torch.nn.functional.cosine_similarity(V["seller_norm"], V["seller"], dim=-1).min()) > 0.9999,
   "seller_norm: the seller direction at the buyer v's per-token size")
dots = (V["seller_perp"] * V["v_pos"]).sum(-1).abs() / (nb * V["seller_perp"].norm(dim=-1)).clamp_min(1e-9)
ok(float(dots.max()) < 1e-4, "seller_perp is orthogonal to the buyer v, per token")

# 3. steering the buyer x1 prompt with its own full difference == E2's x1e6 -> x1 patch (logits)
it = next(x for x in steer if x["fixture"] == "Y01" and x["direction"] == "x1+v")
pair = next(x for x in e2b if x["fixture"] == "Y01" and x["direction"] == "x1e6->x1")
Tg = T._r2_encode(tok, model_io, it["target"]["messages"], it["target"]["limit_text"], "buyer")
Sr = T._r2_encode(tok, model_io, pair["source"]["messages"], pair["source"]["limit_text"], "buyer")
blocks = model_io.decoder_layers(model)
s0 = Tg["span"][0]; suf = Tg["ids"][s0:]
post_rel = torch.tensor([p - s0 for p in range(Tg["span"][-1] + 1, len(Tg["ids"]))])
def run(hook_setup):
    base, _ = T._batched_prefix_cache(model, model_io, Tg["ids"][:s0], 1)
    undo = hook_setup()
    try:
        o = T._forward(model, model_io, input_ids=torch.tensor([suf]), past_key_values=base, use_cache=True)
    finally:
        undo()
    return o.logits[0, -1].float()
ad = {}
def steer_setup():
    ad["a"] = hooks.PrefillAdder(blocks[L], post_rel, V["own"][None], len(suf)); return ad["a"].remove
src = T._capture_all_layers(model, model_io, Sr["ids"], list(range(Sr["span"][0], len(Sr["ids"]))), device_keep=True)
def patch_setup():
    p = hooks.BatchPatcher(blocks).add(L, 0, post_rel, src[L, len(Sr["span"]):, :]); p.arm(); return p.disarm
a, b, c = run(steer_setup), run(patch_setup), run(lambda: (lambda: None))
ok(float((a - b).abs().max()) < 1e-4 and float((a - c).abs().max()) > 1e-3,
   f"own full difference == E2 patch at layer {L}: max |logit diff| {float((a - b).abs().max()):.1e} (unsteered differs by {float((a - c).abs().max()):.1e})")

# 4. the tasks themselves: bridge own|all|1 price == e2_patch post price at L, for every item
br = T.TASKS["buyers_bridge"](model, tok, steer, settings, "gemma-4-31b-it", model_io, layer=L, rows=TB.BRIDGE_ROWS)
e2 = T.e2_patch(model, tok, [x for x in e2b if x["fixture"] in ("Y01", "Y02", "Y03")], settings, "gemma-4-31b-it",
                model_io, layers=[L], positions=["post"], batch=4)
e2p = {(r["fixture"], r["target_condition"]): next(x for x in r["rows"] if x["layer"] == L)["cont"] for r in e2}
same = all(next(x for x in r["rows"] if (x["vector"], x["scope"]) == ("own", "all"))["cont"] == e2p[(r["fixture"], r["target_condition"])]
           for r in br)
ok(same and all(not r["skipped_vectors"] for r in br) and len(br[0]["rows"]) == 32,
   "tasks: bridge own|all|1 continuation == e2_patch post continuation (6 items); seller rows present")

# 5. Part B on one item (short replies): every planned row runs, cross-role rows included, hook fired per token
import game_engine as ge
sb = T.TASKS["buyers_steer"](model, tok, steer[:1], settings, "gemma-4-31b-it", model_io, layer=L, plan=TB.STEER_ROWS,
                             schema=ge.response_schema(settings["max_message_chars"]["value"]), max_new=16)[0]
vecs = {}
for x in sb["rows"]:
    vecs[x["vector"]] = vecs.get(x["vector"], 0) + 1
ok(len(sb["rows"]) == 32 and vecs == {"none": 8, "v_pos": 12, "seller": 4, "seller_norm": 4, "nonmoney": 4}
   and not sb["skipped_vectors"] and all(c[0] == len(T._r2_encode(tok, model_io, steer[0]["target"]["messages"],
   steer[0]["target"]["limit_text"], "buyer")["ids"]) - 11 - T._r2_encode(tok, model_io, steer[0]["target"]["messages"],
   steer[0]["target"]["limit_text"], "buyer")["span"][0] for c in sb["hook_calls"]),
   f"B: 32 rows (8 unsteered, 12 buyer v, 4 seller, 4 seller-size, 4 non-money); prefill hook on the prompt suffix")
ok(T._e3b_vectors is TB._CORE_VECTORS and T._STORE["e3b"] is T._STORE["buyers"]["buyer"][L],
   "core tasks._e3b_vectors and _STORE['e3b'] restored after the plug-in's runs")
'''

if __name__ == "__main__":
    sys.exit(main())
