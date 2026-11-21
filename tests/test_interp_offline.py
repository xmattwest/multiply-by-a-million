#!/usr/bin/env python3
"""test_interp_offline.py: offline checks of the laptop-side interp driver (stdlib, no pod).

  - D2/D8 prompts are identical to run_digits_probe's for the same cells,
  - seeds are shared across conditions,
  - the slip rule and cell statistics reproduce the X50 numbers from the vLLM run,
  - the fidelity verdict is computed as plan §2 says,
  - --preview writes a complete data/interp/<run_id>/ (then removed).

Run: python3 tests/test_interp_offline.py
"""
import glob
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import launch_interp as li          # noqa: E402
import run_digits_probe as dp       # noqa: E402

N = {"pass": 0, "fail": 0}
X50 = "data/explore/X-digits-gemma-4-12b-it-20261005T000526Z"


def ok(cond, name):
    N["pass" if cond else "fail"] += 1
    print(("ok    " if cond else "FAIL  ") + name)


def main():
    settings = json.load(open(os.path.join(ROOT, "prereg", "settings.json")))
    for cond in ("D2", "D8"):
        for role, b in (("seller", 81), ("buyer", 124)):
            ok(li.render_item(role, b, cond, settings) ==
               dp.render_item({"condition": cond, "role": role, "base_limit": b}, settings),
               f"prompt identical to the digits probe: {cond} {role}")
    ok(li.item_seed("Y03", "buyer", 2) == dp.item_seed("Y03", "buyer", 2), "seed rule identical to the digits probe")

    meta, items = li.build_items(settings, 3, 2, ["D2", "D8"])
    ok(len(items) == 3 * 2 * 2 * 2, "items = fixtures x roles x samples x conditions")
    seeds = {}
    for it in items:
        seeds.setdefault((it["fixture"], it["role"], it["sample"]), set()).add(it["seed"])
    ok(all(len(s) == 1 for s in seeds.values()), "seed shared across conditions")
    _, sel = li.build_items(settings, 3, 2, ["D2", "D8"], roles=("seller",))
    ok(len(sel) == 12 and all(it["role"] == "seller" for it in sel), "--roles seller keeps sellers only")

    ok(li.is_slip("seller", 15_000_000, 95_000_000) and not li.is_slip("seller", 95_000_000, 81_000_000),
       "slip rule, seller (price < 0.5 x floor)")
    ok(li.is_slip("buyer", 12_000_000, 124_000_000) and not li.is_slip("buyer", 60_000_000, 124_000_000),
       "slip rule, buyer (price < 0.2 x ceiling)")

    vl_dir = os.path.join(ROOT, X50)
    if os.path.isdir(vl_dir):
        vl = li.cell_stats(li.load_vllm(vl_dir, li.CONDITIONS))
        ok(abs(vl["seller_D2"]["mean_gap"] - 44.3) < 0.05 and abs(vl["seller_D8"]["mean_gap"] - (-19.2)) < 0.05,
           "X50 seller means reproduced (44.3, -19.2)")
        ok(vl["seller_D8"]["slips"] == 88 and vl["buyer_D8"]["slips"] == 36 and vl["seller_D2"]["slips"] == 0,
           "X50 slips counted (seller D8 88/240, buyer D8 36/240, D2 none)")
        ok(abs(vl["seller_D8"]["mean_gap_no_slips"] - 19.6) < 0.05, "X50 seller D8 without slips = 19.6")
        ok(li.default_vllm_run("gemma-4-12b-it").endswith(X50.split("/")[-1]), "default vLLM run found for 12B")
    else:
        print("skip  X50 data not present")

    # Fidelity verdict on hand-made records: pod equals vLLM -> pass; shift halved -> fail.
    def recs(d2, d8):
        return [{"item_id": f"{c}{i}", "role": "seller", "condition": c, "limit_units": 100, "errors": [],
                 "price": round(100 * (1 + g)), "gap": g, "constrained": True}
                for i in range(4) for c, g in (("D2", d2), ("D8", d8))]
    same = li.summarize(recs(0.44, 0.20), [], [], recs(0.44, 0.20))
    ok(same["fidelity"]["pass"] and same["shift_pod"]["seller"] == same["shift_vllm"]["seller"],
       "fidelity passes when the pod matches vLLM")
    half = li.summarize(recs(0.44, 0.32), [], [], recs(0.44, 0.20))
    ok(not half["fidelity"]["pass"] and not half["fidelity"]["gaps_within_3"], "fidelity fails when the shift halves")

    before = set(glob.glob(os.path.join(ROOT, "data", "interp", "*")))
    rc = li.main(["--preview", "--fixtures", "2", "--samples", "1"])
    new = sorted(set(glob.glob(os.path.join(ROOT, "data", "interp", "*"))) - before)
    try:
        ok(rc == 0 and len(new) == 1 and new[0].endswith("-preview"), "preview exits 0 and makes one -preview run dir")
        files = set(os.listdir(new[0])) if new else set()
        ok({"run_manifest.json", "inputs.jsonl", "results.jsonl", "summary.json", "hashes.json"} <= files,
           "preview writes manifest, inputs, results, summary, hashes")
        man = json.load(open(os.path.join(new[0], "run_manifest.json"))) if new else {}
        ok(man.get("revision") and man.get("response_schema", {}).get("required") == ["action", "price", "message"],
           "manifest records the pinned revision and the vLLM response schema")
    finally:
        for d in new:
            shutil.rmtree(d, ignore_errors=True)

    # E1/E2 (laptop side). Prompts differ only in the limit's digits.
    e2 = li.build_e2_items(settings, 3)
    ok(len(e2) == 6 and {x["direction"] for x in e2} == {"x1->x1e6", "x1e6->x1"}, "E2: 2 directions per fixture")
    x = e2[0]
    a_txt, b_txt = x["source"]["messages"][1]["content"], x["target"]["messages"][1]["content"]
    ok(a_txt.replace(x["source"]["limit_text"] + " dollars.", "#", 1) ==
       b_txt.replace(x["target"]["limit_text"] + " dollars.", "#", 1)
       and x["source"]["messages"][0] == x["target"]["messages"][0], "E2: source and target differ only in the limit")
    ok(x["target"]["messages"] == li.render_item("seller", x["base_limit"], "D8", settings),
       "E2: the x1e6 prompt is exactly the digits probe's D8 prompt")
    e1 = li.build_e1_items(settings, 2)
    neg = [i for i in e1 if i["kind"] == "negotiation"]
    plain = [i for i in e1 if i["kind"] == "plain"]
    ok(len(neg) == 14 and neg[0]["messages"] == li.render_item("seller", neg[0]["base_limit"], "D2", settings)
       and neg[6]["messages"] == li.render_item("seller", neg[6]["base_limit"], "D8", settings),
       "E1: k=0 and k=6 prompts are the D2/D8 prompts")
    ok(len(plain) == 240 and all(p["text"].count(p["amount_text"]) == 1 and 10 <= p["amount"] <= 10 ** 9 for p in plain)
       and plain == [i for i in li.build_e1_items(settings, 2) if i["kind"] == "plain"],
       "E1: 240 plain sentences, amounts $10-$1e9, deterministic")
    ok(li.scale_free("seller", 120, 85)[:2] == (120 / 85 - 1, 0) and li.scale_free("seller", 110_000_000, 85)[1] == 6
       and li.scale_free("seller", 11_000_000, 85)[2] is True, "scale-free gap: units and slips")

    def e2rec(fid, direction, clean_gap, patched_gap):
        src, tgt = ("D2", "D8") if direction == "x1->x1e6" else ("D8", "D2")
        unit = 10 ** 6 if tgt == "D8" else 1
        price = lambda g: round((1 + g) * 80 * unit)                       # noqa: E731
        return {"fixture": fid, "role": "seller", "direction": direction, "base_limit": 80,
                "source_condition": src, "target_condition": tgt, "clean_consistent": True,
                "rows": [{"layer": 5, "pos": "final", "price": price(patched_gap)},
                         {"layer": "clean", "pos": None, "price": price(clean_gap)}]}
    recs = [e2rec(f, "x1->x1e6", 0.20, 0.30) for f in ("Y01", "Y02", "Y03")] + \
           [e2rec(f, "x1e6->x1", 0.40, 0.40) for f in ("Y01", "Y02", "Y03")]
    s2 = li.summarize_e2(recs)
    fr = s2["directions"]["x1->x1e6"]["cells"]["final"]["5"]["fraction"]
    ok(abs(fr - 0.5) < 0.02 and abs(s2["directions"]["x1e6->x1"]["cells"]["final"]["5"]["fraction"]) < 0.02,
       "E2: half the gap moved -> 0.5 restored; no move -> 0")

    # E3 (laptop side)
    vec, steer = li.build_e3_items(settings, 3)
    pairs = [v for v in vec if v["kind"] == "pair"]
    ok(len(pairs) == 3 and len(steer) == 6 and {x["direction"] for x in steer} == {"x1+v", "x1e6-v"}
       and steer[0]["target"]["messages"] == li.render_item("seller", steer[0]["base_limit"], "D2", settings)
       and steer[1]["target"]["messages"] == li.render_item("seller", steer[1]["base_limit"], "D8", settings)
       and steer[0]["seed"] == li.item_seed(steer[0]["fixture"], "seller", 1), "E3: items are the D2/D8 prompts, E0 seeds")
    nm = [v for v in vec if v["kind"] == "nonmoney"]
    ok(len(nm) == 40 and all(v["small"].replace(v["small_num"], "#") == v["big"].replace(v["big_num"], "#") for v in nm)
       and not any("dollar" in v["big"] for v in nm), "E3: 40 non-money pairs differ only in the number")

    def e3rec(fid, direction, g0, g1):
        own = "D2" if direction == "x1+v" else "D8"
        unit = 1 if own == "D2" else 10 ** 6
        row = lambda vec, a, g: {"vector": vec, "alpha": a, "errors": [], "action": "OFFER", "finish": "stop",  # noqa: E731
                                 "price": round((1 + g) * 80 * unit)}
        return {"fixture": fid, "role": "seller", "direction": direction, "base_limit": 80, "target_condition": own,
                "rows": [row("none", 0.0, g0), row("v_pos", 1.0, g1)]}
    s3 = li.summarize_e3([e3rec(f, "x1+v", 0.40, 0.30) for f in ("Y01", "Y02", "Y03")] +
                         [e3rec(f, "x1e6-v", 0.20, 0.20) for f in ("Y01", "Y02", "Y03")])
    ok(abs(s3["directions"]["x1+v"]["v_pos"]["1.0"]["share_moved"] - 0.5) < 0.02
       and abs(s3["directions"]["x1e6-v"]["v_pos"]["1.0"]["share_moved"]) < 0.02, "E3: share moved 0.5 / 0")

    # E3b (laptop side): bridge fractions and the steer summary on hand-made records.
    def bre(fid, direction, g0, rows):
        own = "D2" if direction == "x1+v" else "D8"
        unit = 1 if own == "D2" else 10 ** 6
        mk = lambda vec, sc, a, g: {"vector": vec, "scope": sc, "alpha": a, "price": round((1 + g) * 80 * unit)}  # noqa: E731
        return {"fixture": fid, "role": "seller", "direction": direction, "base_limit": 80, "target_condition": own,
                "clean_consistent": True, "rows": [mk("none", "all", 0.0, g0)] + [mk(*r) for r in rows]}
    sb = li.summarize_e3b_bridge([bre(f, "x1+v", 0.40, [("own", "reply", 1.0, 0.30)]) for f in ("Y01", "Y02", "Y03")] +
                                 [bre(f, "x1e6-v", 0.20, [("own", "reply", 1.0, 0.20)]) for f in ("Y01", "Y02", "Y03")])
    ok(abs(sb["cells"]["x1+v"]["own|reply|1.0"]["fraction"] - 0.5) < 0.02
       and abs(sb["cells"]["x1e6-v"]["own|reply|1.0"]["fraction"]) < 0.02, "E3b bridge: 0.5 / 0 moved")
    srec = lambda fid, d, g0, g1, al: {"fixture": fid, "role": "seller", "direction": d, "base_limit": 80,  # noqa: E731
                                       "target_condition": "D2" if d == "x1+v" else "D8",
                                       "rows": [{"vector": v, "scope": "prompt+reply", "alpha": a, "errors": [],
                                                 "action": "OFFER", "finish": "stop", "aligned": al,
                                                 "price": round((1 + g) * 80 * (1 if d == "x1+v" else 10 ** 6))}
                                                for v, a, g in (("none", 0.0, g0), ("v_pos", 1.0, g1))]}
    recs3b = [srec(f, "x1+v", 0.40, 0.30, f != "Y03") for f in ("Y01", "Y02", "Y03")] + \
             [srec(f, "x1e6-v", 0.20, 0.20, True) for f in ("Y01", "Y02", "Y03")]
    ss = li.summarize_e3b_steer(recs3b)
    c = ss["directions"]["x1+v"]["v_pos|prompt+reply"]["1.0"]
    ok(abs(c["share_moved"] - 0.5) < 0.02 and abs(c["aligned_share"] - 2 / 3) < 1e-9
       and li.summarize_e3b_steer(recs3b, aligned_only=True)["directions"]["x1+v"]["v_pos|prompt+reply"]["1.0"]["n_fixtures"] == 2,
       "E3b steer: share moved, aligned share, aligned-only filter")
    before = set(glob.glob(os.path.join(ROOT, "data", "interp", "*")))
    rc = li.main(["--preview", "--experiment", "e3b", "--fixtures", "2"])
    new = sorted(set(glob.glob(os.path.join(ROOT, "data", "interp", "*"))) - before)
    try:
        summ = json.load(open(os.path.join(new[0], "summary.json"))) if new else {}
        ok(rc == 0 and len(new) == 1 and "interp-e3b-" in new[0] and summ.get("bridge", {}).get("n_fixtures") == 2
           and summ.get("steer", {}).get("n_fixtures") == 2, "E3b preview runs end to end")
    finally:
        for d in new:
            shutil.rmtree(d, ignore_errors=True)

    before = set(glob.glob(os.path.join(ROOT, "data", "interp", "*")))
    rc = li.main(["--preview", "--experiment", "e3", "--fixtures", "2"])
    new = sorted(set(glob.glob(os.path.join(ROOT, "data", "interp", "*"))) - before)
    try:
        summ = json.load(open(os.path.join(new[0], "summary.json"))) if new else {}
        ok(rc == 0 and len(new) == 1 and "interp-e3-" in new[0] and summ.get("e3", {}).get("n_fixtures") == 2,
           "E3 preview runs end to end")
    finally:
        for d in new:
            shutil.rmtree(d, ignore_errors=True)

    before = set(glob.glob(os.path.join(ROOT, "data", "interp", "*")))
    rc = li.main(["--preview", "--experiment", "e1e2", "--fixtures", "2", "--layer-step", "10"])
    new = sorted(set(glob.glob(os.path.join(ROOT, "data", "interp", "*"))) - before)
    try:
        summ = json.load(open(os.path.join(new[0], "summary.json"))) if new else {}
        ok(rc == 0 and len(new) == 1 and "interp-e1e2-" in new[0] and summ.get("e2", {}).get("n_fixtures") == 2
           and summ.get("e2_layers") == [0, 10, 20, 30, 40, 50, 59], "E1+E2 preview runs end to end")
    finally:
        for d in new:
            shutil.rmtree(d, ignore_errors=True)

    # Plug-ins: a throwaway exp_/tasks_ pair is discovered, gated and summarized (preview only).
    pdir = os.path.join(ROOT, "scripts", "interp")
    exp_p, task_p = os.path.join(pdir, "exp_zzofflinetest.py"), os.path.join(pdir, "tasks_zzofflinetest.py")
    with open(exp_p, "w") as fh:
        fh.write(
            "def add_args(ap):\n    ap.add_argument('--zzo-n', type=int, default=3)\n"
            "def configure(a):\n    a.roles = 'buyer'\n"
            "def setup(a, settings, schema, li):\n"
            "    items = [{'item_id': f'i{k}'} for k in range(a.zzo_n)]\n    kw = {'x': None}\n"
            "    def after(label, got, ctx):\n"
            "        if label == 'one':\n            kw['x'] = 7\n            return None\n"
            "        return 'gate' if label == 'two' else None\n"
            "    return {'exp': 'ZZO', 'plan': 'zz', 'inputs': items, 'ctx': {'kw': kw},\n"
            "            'steps': lambda ctx: [('one', 'zzo', items, {}, 2), ('two', 'zzo', items, kw, 2), ('three', 'zzo', items, {}, 2)],\n"
            "            'after_step': after}\n"
            "def stub(task, items, ctx):\n    return [{'item_id': it['item_id'], 'x': ctx['kw']['x']} for it in items]\n"
            "def summarize(got, ctx):\n    return {k: [r['x'] for r in v] for k, v in got.items()}\n"
            "def report(s, preview):\n    pass\n")
    with open(task_p, "w") as fh:
        fh.write("TASKS = {'zzo_task': lambda *a, **k: []}\n")
    before = set(glob.glob(os.path.join(ROOT, "data", "interp", "*")))
    try:
        ok("zzofflinetest" in li.load_plugins(), "plug-in discovered")
        rc = li.main(["--preview", "--experiment", "zzofflinetest", "--zzo-n", "4"])
        new = sorted(set(glob.glob(os.path.join(ROOT, "data", "interp", "*"))) - before)
        summ = json.load(open(os.path.join(new[0], "summary.json"))) if new else {}
        ok(summ.get("one") == [None] * 4 and summ.get("two") == [7] * 4 and summ.get("three") == []
           and summ.get("stopped") == "gate", "plug-in: after_step adjusts the next step and gates the session")
        src = open(os.path.join(pdir, "tasks.py")).read()
        ok("_load_plugins()" in src and "tasks_*.py" in src, "pod tasks.py loads tasks_*.py plug-ins")
    finally:
        for f in (exp_p, task_p):
            os.remove(f)
        for d in sorted(set(glob.glob(os.path.join(ROOT, "data", "interp", "*"))) - before):
            shutil.rmtree(d, ignore_errors=True)

    print(f"\npassed: {N['pass']}  failed: {N['fail']}")
    return 1 if N["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
