"""interp_e4_laptop.py (LAPTOP-SIDE, standard library only): E4, the cure. Does Q17's range sentence override the
seller's x1e6 - x1 signal, or is the signal not computed? The plug-in scripts/interp/exp_e4.py is a shim importing
from here (keeps the pod's create request small).

    python3 scripts/launch_interp.py --preview --experiment e4                  (free: canned shapes)
    python3 scripts/launch_interp.py --model gemma-4-31b-it --experiment e4 --slot 8          (paid)

config/interp/plan_next.md, session 1 (STORY X23, X57). Gemma 4 31B, sellers, Y01-Y30. Exploratory.

Prompts per fixture: N1 / N6 = the digits probe's D2 "81 dollars" / D8 "81000000 dollars" seller prompts; P1 / P6 = the
same with Q17's sentence (config/explore/info/info.txt, [info_p_seller]) on its own line after the goal, with Study 1's
ceiling range at the prompt's scale (105 to 125 dollars, or 105000000 to 125000000 dollars; Q17's INFO_RANGES).
All four share the same tokens on A = H + S + reply (the pod checks this per fixture; see tasks_e4.py).

Readout (E2's forced opening, as X57): ld = log p(b) - log p(a) at the deciding token, a and b = the PLAIN prompts'
greedy prices (N1's, N6's), in the target's units. With the range, both scales open at the same price, so the range
prompts' own prices can't define a pair; the plain pair stays defined and measures how far a run leans toward the
plain x1e6 price. Shares divide by the plain gap G = ld(N6) - ld(N1) of the same fixture (cure rows: by the
ld distance between the two runs swapped).

Steps (a stop for time costs only the later ones):
  gate          real sampled openings on P1 and P6 (4 per prompt). Stops the session unless >= 95% are valid, the
                x1e6 - x1 shift is within 5 points, and both scales share the same most common opening (>= 70% each).
  build         greedy prices of all four prompts, ld, per-layer differences (dN = N6 - N1, dP = P6 - P1, c6 = P6 - N6),
                per-head output differences at 38-44, non-money contrast; stops unless >= 10 fixtures are usable.
  stats         fixture-mean directions per layer: |v_N|, |v_P|, cos, projections.
  attention     attention from the reply's opening tokens and the last token to the limit, LO, HI and the range sentence.
  patch         on N1, N6, P1, P6: the plain difference added by layer (38-48), the range runs' own difference, the plain
                direction (leave-one-fixture-out) at 42-44, non-money and random controls, X57's top 8 heads (set from the
                other scale's run, or the plain difference added) and X57's 8 random heads.
  steer P       real sampled openings on P1 + v and P6 - v at layers 42 and 44, alpha 1 and 2; non-money at 42.
  cure          P6's states on S + reply put into N6 by layer (and N6's into P6): where the range's effect reaches the
                tokens that write the price.
  cure heads    discovery fixtures: each head of the 7-layer window where the cure rises fastest, swapped alone.
  cure circuit  held-out fixtures: the top cure heads jointly, random heads, the window's attention.
  steer N       the plain prompts' own steering on the discovery fixtures, the same rows (reference for steer P).
"""
import math
import os
import random
import statistics as st

CANON = 60
RESID_60 = sorted(set(range(0, 59, 2)) | {59} | set(range(36, 47)))
TOP8 = ((41, 29), (39, 28), (42, 9), (41, 27), (41, 10), (41, 11), (43, 5), (39, 22))     # X57, picked on Y01-Y10
RAND8 = ((39, 4), (40, 21), (40, 27), (40, 29), (42, 6), (42, 27), (43, 0), (43, 17))     # X57's random control
RANGE_C = (105, 125)                         # Study 1's ceiling range in dollars at x1 (Q17 INFO_RANGES["C"] / 100)
EXPECT_SENTENCE = ("You also know that the buyer's ceiling was drawn at random from the range {{LO}} to {{HI}}. "
                   "The buyer does not know that you know this.")
PROMPTS = ("N1", "N6", "P1", "P6")
SCALE = {"N1": 0, "N6": 6, "P1": 0, "P6": 6}
OTHER = {"N1": "N6", "N6": "N1", "P1": "P6", "P6": "P1"}
GATE = {"valid": 0.95, "shift": 0.05, "modal": 0.70}
B_PLAN = (("vN", 42, 1.0, 2), ("vN", 44, 1.0, 2), ("vN", 42, 2.0, 2), ("vN", 44, 2.0, 2), ("u", 42, 1.0, 2))
# Timing model (31B pod): X57 ran a batch of 16 forced openings in ~0.65 s; E0 sampled one opening in 3.2 s; E3c B one
# batch of 16 openings in ~7 s. Slow factor for the "if forwards are 1.5x slower" line.
BATCH_S, SRC_S, GEN4_S, GEN16_S, BUILD_S, ATTN_S, SLOW = 0.7, 0.35, 3.6, 7.5, 5.0, 1.6, 1.5
_LI = []


def _li():
    if _LI:
        return _LI[0]
    import sys
    m = sys.modules.get("__main__")
    return m if hasattr(m, "boot_ci") else sys.modules.get("launch_interp")


# --- flags ---------------------------------------------------------------------------------------------------------
def add_args(ap):
    ap.add_argument("--e4-discovery", type=int, default=10, help="e4: fixtures for the cure head sweep and steer N (default 10)")
    ap.add_argument("--e4-gate-samples", type=int, default=4, help="e4: sampled openings per range prompt in the gate (default 4)")


def configure(a):
    a.roles = "seller"


def map_layer(L, n):
    return L if n == CANON else min(n - 1, int(round(L * (n - 1) / (CANON - 1))))


def layer_sets(n):
    """Gemma 4 31B's layer sets (n = 60); other depths (the rehearsal's tiny model) are mapped."""
    m = lambda Ls: sorted({map_layer(L, n) for L in Ls})                # noqa: E731
    out = {"band": m(range(38, 49)), "steer": m((42, 43, 44)), "steer2": m((42, 44)), "cure": RESID_60 if n == CANON
           else list(range(n)), "cure_seg": m((41, 44, 48, 52)), "heads": m(range(38, 45)), "n_layers": n, "mapped": n != CANON}
    out["keep"] = sorted(set(out["steer"]) | {map_layer(L, n) for _v, L, _a, _s in B_PLAN})
    return out


def map_heads(hs, n, nH):
    out = []
    for L, h in hs:
        x = (map_layer(L, n), h % nH)
        if x not in out:
            out.append(x)
    return out


# --- prompts and items -----------------------------------------------------------------------------------------------
def range_sentence(li, mult):
    path = os.path.join(li.ROOT, "config", "explore", "info", "info.txt")
    frag = li.rp.load_fragments(path)["info_p_seller"]
    if frag != EXPECT_SENTENCE:
        raise ValueError("config/explore/info/info.txt [info_p_seller] is not Q17's sentence")
    lo, hi = RANGE_C[0] * mult, RANGE_C[1] * mult
    return li.rp.fill(frag, {"LO": f"{lo} dollars", "HI": f"{hi} dollars"}), {"lo": str(lo), "hi": str(hi)}


def render_side(li, settings, b, cond, with_range):
    """One seller prompt: the digits probe's D2/D8 text, plus (with_range) Q17's sentence on the line after the goal,
    as newgame.render places it (GOAL + newline + the information line)."""
    mult = li.CONDITIONS[cond][1]
    msgs = li.render_item("seller", b, cond, settings)
    rg = None
    if with_range:
        sent, rg = range_sentence(li, mult)
        lines = msgs[1]["content"].split("\n")
        at = [i for i, l in enumerate(lines) if l.startswith("Your goal is")]
        if len(at) != 1:
            raise ValueError("goal line not found exactly once")
        lines.insert(at[0] + 1, sent)
        msgs[1]["content"] = "\n".join(lines)
    return {"condition": cond, "limit_units": b * mult, "limit_text": str(b * mult), "messages": msgs, "range": rg}


def build_items(li, settings, n_fix):
    doc = li.mf.build(li.PHRASE)
    out = []
    for f in doc["fixtures"][:n_fix]:
        fid, b = f"Y{f['id'][1:]}", f["f_pct"]
        if not RANGE_C[0] <= f["c_pct"] <= RANGE_C[1]:
            raise ValueError(f"{fid}: the ceiling {f['c_pct']} is outside the range {RANGE_C}")
        pr = {"N1": render_side(li, settings, b, "D2", False), "N6": render_side(li, settings, b, "D8", False),
              "P1": render_side(li, settings, b, "D2", True), "P6": render_side(li, settings, b, "D8", True)}
        out.append({"item_id": f"{fid}-S-e4", "fixture": fid, "role": "seller", "base_limit": b, "ceiling": f["c_pct"],
                    "seed": li.item_seed(fid, "seller", 1), "prompts": pr})
    return out


def steer_items(fx, targets, tag):
    return [dict(it, item_id=f"{it['fixture']}-S-{t}-{tag}", target=t, sign=1.0 if SCALE[t] == 0 else -1.0)
            for it in fx for t in targets]


# --- row specs ---------------------------------------------------------------------------------------------------------
def patch_rows(Ls, top, rand):
    """{target: [row spec]}; see tasks_e4.e4_patch for the ops."""
    hs = lambda op, heads, seg, *rest: [[op, "head", L, h, seg, *rest] for L, h in heads]     # noqa: E731
    out = {}
    for t in PROMPTS:
        o, sg = OTHER[t], 1.0 if SCALE[t] == 0 else -1.0
        n_o, n_t = ("N6", "N1") if SCALE[t] == 0 else ("N1", "N6")
        rows = []
        if t[0] == "N":
            rows += [{"n": f"own@{L}", "ops": [["set", "resid", L, -1, "A", o]]} for L in Ls["band"]]
            rows += [{"n": f"own reply@{L}", "ops": [["set", "resid", L, -1, "reply", o]]} for L in Ls["steer2"]]
        else:
            rows += [{"n": f"plain@{L}", "ops": [["add", "resid", L, -1, "A", 1.0, n_o, n_t]]} for L in Ls["band"]]
            rows += [{"n": f"plain reply@{L}", "ops": [["add", "resid", L, -1, "reply", 1.0, n_o, n_t]]} for L in Ls["steer2"]]
            rows += [{"n": f"own@{L}", "ops": [["set", "resid", L, -1, "A", o]]} for L in Ls["band"]]
        rows += [{"n": f"vN@{L}", "ops": [["vec", L, "A", sg, "vN"]]} for L in Ls["steer"]]
        rows += [{"n": f"vNx2@{L}", "ops": [["vec", L, "A", 2 * sg, "vN"]]} for L in Ls["steer2"]]
        rows += [{"n": f"u@{L}", "ops": [["vec", L, "A", sg, "u"]]} for L in Ls["steer2"]]
        rows += [{"n": f"rand@{Ls['steer'][0]}", "ops": [["vec", Ls["steer"][0], "A", sg, "rand"]]}]
        rows += [{"n": "top8 set", "ops": hs("set", top, "A", o)},
                 {"n": "top8 add", "ops": hs("add", top, "A", 1.0, n_o, n_t)},
                 {"n": "rand8 add", "ops": hs("add", rand, "A", 1.0, n_o, n_t)},
                 {"n": "top8 add reply", "ops": hs("add", top, "reply", 1.0, n_o, n_t)}]
        if t[0] == "P":
            rows.append({"n": "rand8 set", "ops": hs("set", rand, "A", o)})
        out[t] = rows
    return out


def cure_rows(Ls):
    out = {}
    for t, src in (("N6", "P6"), ("P6", "N6")):
        rows = [{"n": f"cure@{L}", "ops": [["set", "resid", L, -1, "Srep", src]]} for L in Ls["cure"]]
        for L in Ls["cure_seg"]:
            rows += [{"n": f"cure S@{L}", "ops": [["set", "resid", L, -1, "S", src]]},
                     {"n": f"cure reply@{L}", "ops": [["set", "resid", L, -1, "reply", src]]}]
        out[t] = rows
    return out


def cure_head_rows(band, nH):
    out = {}
    for t, src in (("N6", "P6"), ("P6", "N6")):
        out[t] = [{"n": f"h{L}.{h}", "ops": [["set", "head", L, h, "Srep", src]]} for L in band for h in range(nH)] + \
                 [{"n": f"attn@{L}", "ops": [["set", "attn", L, -1, "Srep", src]]} for L in band]
    return out


def circuit_rows(top, rand, band):
    out = {}
    for t, src in (("N6", "P6"), ("P6", "N6")):
        hs = lambda heads, seg="Srep": [["set", "head", L, h, seg, src] for L, h in heads]       # noqa: E731
        rows = [{"n": f"top{k}", "ops": hs(top[:k])} for k in sorted({1, max(1, len(top) // 2), len(top)}) if top]
        rows += [{"n": f"top{len(top)} reply", "ops": hs(top, "reply")}, {"n": f"top{len(top)} S", "ops": hs(top, "S")},
                 {"n": f"random{len(rand)}", "ops": hs(rand)},
                 {"n": "window attn", "ops": [["set", "attn", L, -1, "Srep", src] for L in band]}]
        out[t] = rows
    return out


def _nrows(rows):
    return {t: len(v) for t, v in rows.items()}


def _batches(n, batch):
    return int(math.ceil((n + 1) / batch))


def estimate_min(n_fix, n_disc, n_held, batch, gate_samples):
    Ls = layer_sets(CANON)
    P = _nrows(patch_rows(Ls, TOP8, RAND8))
    C = _nrows(cure_rows(Ls))
    H = _nrows(cure_head_rows(list(range(38, 45)), 32))
    K = _nrows(circuit_rows(list(TOP8), list(RAND8), list(range(38, 45))))
    per = {"e4 gate": GEN4_S * (1 + gate_samples / 16), "e4 build": BUILD_S, "e4 stats": 5.0, "e4 attention": ATTN_S,
           "e4 patch": sum(_batches(v, batch) for v in P.values()) * BATCH_S + 4 * SRC_S,
           "e4 steer P": GEN16_S, "e4 cure": sum(_batches(v, batch) for v in C.values()) * BATCH_S + 2 * SRC_S,
           "e4 cure heads": sum(_batches(v, batch) for v in H.values()) * BATCH_S + 2 * SRC_S,
           "e4 cure circuit": sum(_batches(v, batch) for v in K.values()) * BATCH_S + 2 * SRC_S, "e4 steer N": GEN16_S}
    n = {"e4 gate": 2 * n_fix, "e4 build": n_fix, "e4 stats": 1, "e4 attention": n_fix, "e4 patch": n_fix,
         "e4 steer P": 2 * n_fix, "e4 cure": n_fix, "e4 cure heads": n_disc, "e4 cure circuit": n_held, "e4 steer N": 2 * n_disc}
    t = sum(per[k] * n[k] for k in per) + 40 * 0.1
    return t / 60, t * SLOW / 60, per


# --- setup ---------------------------------------------------------------------------------------------------------------
def setup(a, settings, schema, li):
    _LI[:] = [li]
    fx = build_items(li, settings, a.fixtures)
    fix = [it["fixture"] for it in fx]
    n_disc = min(a.e4_discovery, max(1, len(fix) // 3))
    disc, held = fix[:n_disc], fix[n_disc:]
    Ls = layer_sets(CANON)
    nonmoney = [v for v in li.build_e3_items(settings, 1)[0] if v["kind"] == "nonmoney"]
    typ, slow, per = estimate_min(len(fx), len(disc), len(held), a.batch, a.e4_gate_samples)
    li.STEP_GUESS_S.update(per)
    ctx = {"li": li, "fx": fx, "disc": disc, "held": held, "nonmoney": nonmoney, "layers": Ls, "n_heads": 32,
           "top": list(TOP8), "rand": list(RAND8), "window": None, "pick": None, "model": a.model,
           "gate_kw": {"plan": [], "n_none": a.e4_gate_samples, "batch": a.e4_gate_samples, "schema": schema},
           "build_kw": {"keep": Ls["keep"], "head_layers": Ls["heads"]}, "stats_kw": {}, "attn_kw": {},
           "patch_kw": {"rows": patch_rows(Ls, TOP8, RAND8), "batch": a.batch},
           "steerP_kw": {"plan": [list(x) for x in B_PLAN], "n_none": 6, "batch": 16, "schema": schema},
           "cure_kw": {"rows": cure_rows(Ls), "batch": a.batch},
           "heads_kw": {"rows": {}, "batch": a.batch}, "circuit_kw": {"rows": {}, "batch": a.batch},
           "estimate_min": [round(typ, 1), round(slow, 1)]}
    ctx["steerN_kw"] = ctx["steerP_kw"]
    np_ = _nrows(ctx["patch_kw"]["rows"])
    plan = (f"E4 (the cure) on {a.model} (sellers): {len(fx)} fixtures x 4 prompts (plain D2/D8 and the same with Q17's "
            f"range sentence, {RANGE_C[0]}-{RANGE_C[1]} dollars at the prompt's scale). Gate = {a.e4_gate_samples} sampled "
            f"openings per range prompt (stops unless the cure holds); build + stats (|v| and cosines at every layer); "
            f"attention to the range's digits; patch rows per prompt N1/N6/P1/P6 {np_['N1']}/{np_['N6']}/{np_['P1']}/{np_['P6']} "
            f"(plain difference at 38-48, steering at 42-44, controls, X57's top 8 and random 8 heads); steer P (real "
            f"openings, layers 42/44); cure by layer ({len(Ls['cure'])} layers); cure heads on {len(disc)} fixtures, "
            f"circuit on {len(held)}; steer N on {len(disc)}; batch {a.batch}; about {typ:.0f} min of experiment time "
            f"({slow:.0f} if forwards run 1.5x slower; --budget-min {a.budget_min:g} stops new items)")
    inputs = [{"step": "fixture", **it} for it in fx] + [{"step": "nonmoney", **v} for v in nonmoney]
    return {"exp": "E4", "plan": plan, "inputs": inputs, "ctx": ctx, "steps": steps, "live": live, "after_step": after_step}


def steps(ctx):
    fx = ctx["fx"]
    dfx = [it for it in fx if it["fixture"] in ctx["disc"]]
    hfx = [it for it in fx if it["fixture"] in ctx["held"]]
    return [("e4 gate", "e4_steer", steer_items(fx, ("P1", "P6"), "gate"), ctx["gate_kw"], 4),
            ("e4 build", "e4_build", ctx["nonmoney"] + fx, ctx["build_kw"], 5),
            ("e4 stats", "e4_stats", [{"item_id": "stats"}], ctx["stats_kw"], 1),
            ("e4 attention", "e4_attn", fx, ctx["attn_kw"], 5),
            ("e4 patch", "e4_patch", fx, ctx["patch_kw"], 2),
            ("e4 steer P", "e4_steer", steer_items(fx, ("P1", "P6"), "steer"), ctx["steerP_kw"], 2),
            ("e4 cure", "e4_patch", fx, ctx["cure_kw"], 3),
            ("e4 cure heads", "e4_patch", dfx, ctx["heads_kw"], 1),
            ("e4 cure circuit", "e4_patch", hfx, ctx["circuit_kw"], 5),
            ("e4 steer N", "e4_steer", steer_items(dfx, ("N1", "N6"), "steer"), ctx["steerN_kw"], 2)]


def _relayer(ctx, n, nH):
    Ls = layer_sets(n)
    ctx["layers"], ctx["n_heads"] = Ls, nH
    ctx["top"], ctx["rand"] = map_heads(TOP8, n, nH), map_heads(RAND8, n, nH)
    ctx["build_kw"].update(keep=Ls["keep"], head_layers=Ls["heads"])
    ctx["patch_kw"]["rows"] = patch_rows(Ls, ctx["top"], ctx["rand"])
    ctx["cure_kw"]["rows"] = cure_rows(Ls)
    ctx["steerP_kw"]["plan"][:] = [[v, map_layer(L, n), a, s] for v, L, a, s in B_PLAN]


def on_pod_info(ctx, pod_info):
    n = pod_info.get("num_hidden_layers")
    if n and int(n) != ctx["layers"]["n_layers"]:
        _relayer(ctx, int(n), ctx["n_heads"])


def after_step(label, got, ctx):
    li = ctx.get("li") or _li()
    note = (lambda s: ctx["panel"].note(s)) if ctx.get("panel") is not None else (lambda s: None)
    if label == "e4 gate":
        G = summarize_gate(got.get("e4 gate") or [], li)
        if G.get("rehearsal"):
            note("gate: REHEARSAL (random weights), not applied: " + (G.get("why") or "pass"))
            return None
        note("gate: " + ("PASS" if G["pass"] else "FAIL") + f" ({G['line']})")
        return None if G["pass"] else "gate failed: " + G["why"]
    if label == "e4 build":
        recs = [r for r in got.get("e4 build") or [] if r.get("kind") == "fixture"]
        use = [r for r in recs if not r.get("skipped")]
        if len(use) < min(10, len(recs)):
            return f"only {len(use)} of {len(recs)} fixtures have two distinct plain greedy prices"
        if len({(r["n_H"], r["n_S"], r["n_reply"]) for r in use}) != 1:
            return "the aligned segments differ in length between fixtures"
        n, nH = use[0]["n_layers"], use[0]["n_heads"]
        if n != ctx["layers"]["n_layers"] or nH != ctx["n_heads"]:
            _relayer(ctx, n, nH)
        note(f"build: {len(use)}/{len(recs)} fixtures usable; aligned tokens {use[0]['n_A']} = {use[0]['n_H']} + "
             f"{use[0]['n_S']} + {use[0]['n_reply']}; range sentence {use[0]['n_R']['P1']}/{use[0]['n_R']['P6']} tokens")
        return None
    if label == "e4 cure":
        C = cure_summary(got.get("e4 cure") or [])
        band, info = pick_window(C, ctx["layers"]["n_layers"])
        if band is None:
            return None                                      # no cure readings: the later cure steps have no rows
        ctx["window"] = dict(info, band=band)
        ctx["heads_kw"]["rows"].update(cure_head_rows(band, ctx["n_heads"]))
        note(f"cure heads: window L{band[0]}-L{band[-1]} ({info['rule']})")
        return None
    if label == "e4 cure heads":
        S = ratio_table(got.get("e4 cure heads") or [], "cure", ci=False)
        top, rand, info = pick_cure_heads(S, ctx["window"]["band"] if ctx.get("window") else [], ctx["n_heads"])
        if top:
            ctx["pick"] = dict(info, top=[list(x) for x in top], random=[list(x) for x in rand])
            ctx["circuit_kw"]["rows"].update(circuit_rows(top, rand, ctx["window"]["band"]))
            note("cure heads: " + ", ".join(f"L{L}.{h}" for L, h in top))
    return None


# --- analysis ------------------------------------------------------------------------------------------------------------
def ratio_ci(num, den, n=1000, seed=1):
    k = len(num)
    if k < 3:
        return None
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        idx = [rng.randrange(k) for _ in range(k)]
        d = sum(den[i] for i in idx)
        if d:
            vals.append(sum(num[i] for i in idx) / d)
    if len(vals) < 20:
        return None
    vals.sort()
    return [vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]]


def units(price, m):
    """A price in the x1 unit: whole numbers stay whole (120000000 at x1e6 -> 120)."""
    return price // 10 ** m if price % 10 ** m == 0 else price / 10 ** m


def _openings(recs, li):
    """{(target, fixture): {row key: [gap]}} for valid OFFERs on the target's own scale; counts per (target, key)."""
    per, cnt = {}, {}
    for r in recs:
        own = SCALE[r["target"]]
        for x in r["rows"]:
            key = "none" if x["vector"] == "none" else f"{x['vector']}@{x['layer']}x{x['alpha']:g}"
            c = cnt.setdefault((r["target"], key), {"n": 0, "valid": 0, "off_scale": 0, "other_scale": 0, "aligned": 0,
                                                     "prices": {}})
            c["n"] += 1
            c["aligned"] += bool(x.get("aligned"))
            if x.get("errors"):
                continue
            c["valid"] += 1
            if x.get("action") != "OFFER" or not x.get("price"):
                continue
            g, m, off = li.scale_free("seller", x["price"], r["base_limit"])
            if off:
                c["off_scale"] += 1
                continue
            if m != own:
                c["other_scale"] += 1
                continue
            p = units(x["price"], own)
            c["prices"][p] = c["prices"].get(p, 0) + 1
            per.setdefault((r["target"], r["fixture"]), {}).setdefault(key, []).append(g)
    return per, cnt


def summarize_gate(recs, li):
    per, cnt = _openings(recs, li)
    out = {"rehearsal": any(r.get("rehearsal") for r in recs), "cells": {}}
    for t in ("P1", "P6"):
        c = cnt.get((t, "none"), {"n": 0, "valid": 0, "prices": {}, "off_scale": 0, "other_scale": 0})
        g = [x for (tt, _f), v in per.items() if tt == t for x in v.get("none", [])]
        k = sum(c["prices"].values())
        modal = max(c["prices"], key=lambda p: (c["prices"][p], -p)) if c["prices"] else None
        out["cells"][t] = {"n": c["n"], "valid": c["valid"], "valid_share": c["valid"] / c["n"] if c["n"] else None,
                           "on_scale": k, "off_scale": c["off_scale"], "other_scale": c["other_scale"],
                           "mean_gap": st.mean(g) if g else None, "modal": modal,
                           "modal_share": c["prices"][modal] / k if modal is not None else None,
                           "at_120": c["prices"].get(120, 0) / k if k else None, "at_125": c["prices"].get(125, 0) / k if k else None,
                           "prices": {str(p): n for p, n in sorted(c["prices"].items())}}
    a, b = out["cells"]["P1"], out["cells"]["P6"]
    diffs = [st.mean(per[("P6", f)]["none"]) - st.mean(per[("P1", f)]["none"])
             for (t, f) in per if t == "P1" and ("P6", f) in per and per[("P1", f)].get("none") and per[("P6", f)].get("none")]
    out["shift"] = b["mean_gap"] - a["mean_gap"] if a["mean_gap"] is not None and b["mean_gap"] is not None else None
    out["shift_ci"] = list(li.boot_ci(diffs)) if len(diffs) >= 3 else None
    out["n_fixtures_paired"] = len(diffs)
    why = []
    for t, c in (("P1", a), ("P6", b)):
        if c["valid_share"] is None or c["valid_share"] < GATE["valid"]:
            why.append(f"{t} valid {c['valid']}/{c['n']}")
    if out["shift"] is None or abs(out["shift"]) > GATE["shift"]:
        why.append("shift " + ("n/a" if out["shift"] is None else f"{100 * out['shift']:+.1f} points"))
    if a["modal"] is None or a["modal"] != b["modal"]:
        why.append(f"most common openings differ ({a['modal']} vs {b['modal']} after rescaling)")
    for t, c in (("P1", a), ("P6", b)):
        if c["modal_share"] is None or c["modal_share"] < GATE["modal"]:
            why.append(f"{t} most common opening only {0 if c['modal_share'] is None else 100 * c['modal_share']:.0f}%")
    out["pass"], out["why"] = not why, "; ".join(why)
    f = lambda v: "n/a" if v is None else f"{100 * v:.1f}%"                       # noqa: E731
    out["line"] = (f"P1 {f(a['mean_gap'])} / P6 {f(b['mean_gap'])}, shift "
                   + ("n/a" if out["shift"] is None else f"{100 * out['shift']:+.1f} pts")
                   + f"; most common {a['modal']} ({f(a['modal_share'])}) / {b['modal']} ({f(b['modal_share'])}); "
                   f"valid {a['valid']}/{a['n']}, {b['valid']}/{b['n']}")
    return out


def e0_plain(li, model):
    """Per-fixture plain gaps {fixture: {0: mean, 6: mean}} from the E0 run on the same prompts (X51), or {}."""
    d = li.default_e0_run(model) if model else None
    out = {}
    if not d or not os.path.exists(os.path.join(d, "results.jsonl")):
        return out, None
    import json
    for line in open(os.path.join(d, "results.jsonl")):
        r = json.loads(line)
        if r.get("step") == "sampled" and r.get("role") == "seller" and r.get("gap") is not None and not r.get("errors"):
            out.setdefault(r["item_id"][:3], {}).setdefault(0 if r["condition"] == "D2" else 6, []).append(r["gap"])
    return {f: {m: st.mean(v) for m, v in d_.items()} for f, d_ in out.items()}, os.path.relpath(d, li.ROOT)


def summarize_steer(recs, li, ref):
    """Per target and row: counts, mean gap, points moved vs the same item's unsteered rows, and the share of the PLAIN
    gap (ref: per-fixture plain gaps, {fixture: {0, 6}}): den_f = plain gap at the other scale - at the target's."""
    per, cnt = _openings(recs, li)
    out = {}
    for (t, key), c in cnt.items():
        cell = {k: c[k] for k in ("n", "valid", "off_scale", "other_scale", "aligned")}
        g = [x for (tt, _f), v in per.items() if tt == t for x in v.get(key, [])]
        cell["mean_gap"] = st.mean(g) if g else None
        cell["round10"] = (sum(n for p, n in c["prices"].items() if p % 10 == 0) / sum(c["prices"].values())) if c["prices"] else None
        if key != "none":
            num, den = [], []
            for (tt, f), v in per.items():
                if tt != t or key not in v or "none" not in v:
                    continue
                num.append(st.mean(v[key]) - st.mean(v["none"]))
                r_ = ref.get(f)
                den.append(r_[6 - SCALE[t]] - r_[SCALE[t]] if r_ and 0 in r_ and 6 in r_ else None)
            cell["moved_pts"] = 100 * st.mean(num) if num else None
            pairs = [(a, b) for a, b in zip(num, den) if b is not None]
            cell["n_fixtures"] = len(num)
            if pairs and st.mean(b for _a, b in pairs):
                D = st.mean(b for _a, b in pairs)
                cell["share"] = st.mean(a for a, _b in pairs) / D
                cell["ci"] = ratio_ci([a for a, _ in pairs], [b for _, b in pairs])
        out.setdefault(t, {})[key] = cell
    return out


def _target_ld(r, t):
    T = (r.get("targets") or {}).get(t)
    return None if T is None else T["ld_clean"]


def ratio_table(recs, mode, ci=True):
    """{target: {row: {share, ci, n, moved, pk}}}. mode 'plain': den = s * (ld_clean(N6) - ld_clean(N1)) of the same item
    (s = +1 for x1 targets, -1 for x1e6); mode 'cure': den = ld_clean(the swapped-in run) - ld_clean(the target)."""
    acc = {}
    for r in recs:
        if r.get("skipped"):
            continue
        for t, T in (r.get("targets") or {}).items():
            if mode == "plain":
                a, b = _target_ld(r, "N1"), _target_ld(r, "N6")
                den = None if a is None or b is None else (b - a) * (1 if SCALE[t] == 0 else -1)
            else:
                o = _target_ld(r, "P6" if t == "N6" else "N6")
                den = None if o is None else o - T["ld_clean"]
            if den is None:
                continue
            for x in T["rows"]:
                if x["n"] == "clean":
                    continue
                c = acc.setdefault((t, x["n"]), {"num": [], "den": [], "pk": []})
                c["num"].append(x["ld"] - T["ld_clean"])
                c["den"].append(den)
                c["pk"].append(math.exp(x["lpa"]) + math.exp(x["lpb"]))
    out = {}
    for (t, n), c in acc.items():
        D = st.mean(c["den"])
        out.setdefault(t, {})[n] = {"n": len(c["num"]), "share": st.mean(c["num"]) / D if D else None,
                                    "ci": ratio_ci(c["num"], c["den"]) if ci and D else None, "moved": st.mean(c["num"]),
                                    "gap": D, "pk": st.mean(c["pk"])}
    return out


def clean_table(recs):
    """Per target: mean clean ld and the two prices' probability; the gaps G_N, G_P, and the cure gaps (in nats)."""
    ld, pk = {}, {}
    for r in recs:
        for t, T in (r.get("targets") or {}).items():
            ld.setdefault(t, []).append(T["ld_clean"])
            c = next((x for x in T["rows"] if x["n"] == "clean"), None)
            if c:
                pk.setdefault(t, []).append(math.exp(c["lpa"]) + math.exp(c["lpb"]))
    out = {"ld": {t: st.mean(v) for t, v in ld.items()}, "pk": {t: st.mean(v) for t, v in pk.items()}}
    m = out["ld"]
    if {"N1", "N6"} <= set(m):
        out["G_N"] = m["N6"] - m["N1"]
        if {"P1", "P6"} <= set(m):
            out["G_P"] = m["P6"] - m["P1"]
            out["G_P_share"] = out["G_P"] / out["G_N"] if out["G_N"] else None
    return out


def cure_summary(recs):
    T = ratio_table(recs, "cure", ci=False)
    return {t: {n: c["share"] for n, c in T.get(t, {}).items() if n.startswith("cure@")} for t in ("N6", "P6")}


def pick_window(C, n_layers, width=7):
    """The 7-layer window where the cure rises most: score(L) = mean of both directions' 'cure@L' shares (linearly
    interpolated between measured layers); rise = score(end) - score(start - 1)."""
    pts = {}
    for t in ("N6", "P6"):
        for n, v in (C.get(t) or {}).items():
            if v is not None:
                pts.setdefault(int(n.split("@")[1]), []).append(v)
    pts = {L: st.mean(v) for L, v in pts.items() if len(v) == 2}
    if len(pts) < 2:
        return None, {"rule": "no cure readings in both directions"}
    xs = sorted(pts)

    def at(L):
        if L <= xs[0]:
            return pts[xs[0]] if L == xs[0] else 0.0
        if L >= xs[-1]:
            return pts[xs[-1]]
        j = next(i for i, x in enumerate(xs) if x >= L)
        x0, x1 = xs[j - 1], xs[j]
        return pts[x0] + (pts[x1] - pts[x0]) * (L - x0) / (x1 - x0)
    w = min(width, n_layers)
    best = max(range(0, n_layers - w + 1), key=lambda s: (at(s + w - 1) - at(s - 1), -s))
    return list(range(best, best + w)), {"rule": f"largest rise of the mean cure share over {w} layers "
                                         f"({at(best - 1):+.2f} -> {at(best + w - 1):+.2f})",
                                         "rise": at(best + w - 1) - at(best - 1)}


def pick_cure_heads(S, band, nH, k=8, seed="scale-invariance/interp/e4/random/v1"):
    scored = []
    for L in band:
        for h in range(nH):
            v = [S.get(t, {}).get(f"h{L}.{h}", {}).get("share") for t in ("N6", "P6")]
            if None not in v:
                scored.append(((L, h), sum(v) / 2))
    scored.sort(key=lambda x: (-x[1], x[0]))
    k = min(k, len(scored) // 2) if len(scored) >= 2 else 0
    top = [x[0] for x in scored[:k]]
    pool = [x[0] for x in scored if x[0] not in top]
    rand = sorted(random.Random(seed).sample(pool, min(k, len(pool))))
    return top, rand, {"rule": f"top {k} by the mean cure share of both directions",
                       "scores": {f"h{L}.{h}": round(v, 4) for (L, h), v in scored[:3 * max(k, 1)]}}


def build_summary(recs):
    fx = [r for r in recs if r.get("kind") == "fixture"]
    use = [r for r in fx if not r.get("skipped")]
    out = {"n": len(fx), "usable": len(use), "skipped": {r["fixture"]: r["skipped"] for r in fx if r.get("skipped")},
           "stand_in": any(r.get("stand_in") for r in fx), "prices": {r["fixture"]: r.get("price") for r in fx},
           "n_nonmoney": sum(1 for r in recs if r.get("kind") == "nonmoney")}
    if not use:
        return out
    u0 = use[0]
    out.update(n_layers=u0["n_layers"], n_heads=u0["n_heads"], n_A=u0["n_A"], n_H=u0["n_H"], n_S=u0["n_S"],
               n_reply=u0["n_reply"], n_R=u0["n_R"], n_tokens=u0["n_tokens"],
               h_identical_max=max(r.get("h_identical", 0) for r in use))
    out["ld"] = {k: st.mean(r["ld"][k] for r in use) for k in PROMPTS}
    out["G_N"] = out["ld"]["N6"] - out["ld"]["N1"]
    out["G_P"] = out["ld"]["P6"] - out["ld"]["P1"]
    for k in ("P1", "P6"):
        ps = [units(r["price"][k], SCALE[k]) for r in use if r["price"].get(k)]
        out[f"greedy_{k}"] = {str(p): ps.count(p) for p in sorted(set(ps))}
    nL = u0["n_layers"]
    lay = {}
    for s in ("H", "S", "reply"):
        tot = [[sum(r["layers"][s][j][L] for r in use) for L in range(nL)] for j in range(5)]
        aa, bb, ab, cc, ac = tot
        cosf = [st.mean(r["layers"][s][2][L] / math.sqrt(max(1e-30, r["layers"][s][0][L] * r["layers"][s][1][L])) for r in use)
                for L in range(nL)]
        lay[s] = {"ratio": [math.sqrt(bb[L] / aa[L]) if aa[L] else None for L in range(nL)],
                  "proj": [ab[L] / aa[L] if aa[L] else None for L in range(nL)], "cos_fixture": cosf,
                  "cure_proj": [ac[L] / aa[L] if aa[L] else None for L in range(nL)],
                  "cure_ratio": [math.sqrt(cc[L] / aa[L]) if aa[L] else None for L in range(nL)]}
    out["layers"] = lay
    heads = {}
    for L in u0["head_layers"]:
        for s in ("Srep", "reply"):
            sN = [sum(r["heads"][str(L)][s][0][h] for r in use) for h in range(u0["n_heads"])]
            sP = [sum(r["heads"][str(L)][s][1][h] for r in use) for h in range(u0["n_heads"])]
            sX = [sum(r["heads"][str(L)][s][2][h] for r in use) for h in range(u0["n_heads"])]
            for h in range(u0["n_heads"]):
                heads.setdefault(f"h{L}.{h}", {})[s] = {"ratio": math.sqrt(sP[h] / sN[h]) if sN[h] else None,
                                                        "proj": sX[h] / sN[h] if sN[h] else None, "normN": math.sqrt(sN[h] / len(use))}
    out["heads"] = heads
    return out


def attention_summary(recs):
    ok = [r for r in recs if not r.get("error") and r.get("mass")]
    if not ok:
        return {"error": next((r["error"] for r in recs if r.get("error")), "no records")}
    out = {"n_fixtures": len(ok), "keys": ok[0]["keys"], "mean": {}}
    for k in ok[0]["mass"]:
        M = [r["mass"][k] for r in ok if r["mass"].get(k) and r["mass"][k][0] is not None]
        nL, nq, nk, nH = len(M[0]), len(M[0][0]), len(M[0][0][0]), len(M[0][0][0][0])
        out["mean"][k] = [[[[st.mean(m[L][q][j][h] for m in M) for h in range(nH)] for j in range(nk)] for q in range(nq)]
                          for L in range(nL)]
    out["n_layers"], out["n_heads"] = nL, nH
    top = {}
    for k in ("P1", "P6"):
        if k in out["mean"] and "hi" in out["keys"][k]:
            j = out["keys"][k].index("hi")
            sc = sorted(((L, h, out["mean"][k][L][1][j][h]) for L in range(nL) for h in range(nH)), key=lambda x: -x[2])
            top[k] = [[L, h, round(v, 4)] for L, h, v in sc[:12]]
    out["top_last_to_hi"] = top
    return out


def summarize(got, ctx):
    li = ctx.get("li") or _li()
    ref, ref_from = e0_plain(li, ctx.get("model"))
    b = got.get("e4 build") or []
    out = {"layers": ctx["layers"], "n_heads": ctx["n_heads"], "top": ctx["top"], "rand": ctx["rand"],
           "discovery": ctx["disc"], "held_out": ctx["held"], "range": list(RANGE_C),
           "gate": summarize_gate(got.get("e4 gate") or [], li), "build": build_summary(b),
           "stats": (got.get("e4 stats") or [None])[0], "attention": attention_summary(got.get("e4 attention") or []),
           "patch": ratio_table(got.get("e4 patch") or [], "plain"), "patch_clean": clean_table(got.get("e4 patch") or []),
           "steer_P": summarize_steer(got.get("e4 steer P") or [], li, ref),
           "steer_N": summarize_steer(got.get("e4 steer N") or [], li, ref), "steer_ref": ref_from,
           "cure": ratio_table(got.get("e4 cure") or [], "cure"), "cure_clean": clean_table(got.get("e4 cure") or []),
           "window": ctx.get("window"), "cure_heads": ratio_table(got.get("e4 cure heads") or [], "cure", ci=False),
           "pick": ctx.get("pick"), "circuit": ratio_table(got.get("e4 cure circuit") or [], "cure"),
           "estimate_min": ctx.get("estimate_min")}
    pr = [r for lab in ("e4 patch", "e4 cure", "e4 cure heads", "e4 cure circuit") for r in got.get(lab) or []]
    out["checks"] = {"hooks_ok": all(T.get("hooks_ok", True) for r in pr for T in (r.get("targets") or {}).values()),
                     "clean_spread_max": max((T.get("clean_spread") or 0 for r in pr for T in (r.get("targets") or {}).values()),
                                             default=None),
                     "steer_aligned": {lab: [sum(1 for r in got.get(lab) or [] for x in r["rows"] if x["vector"] != "none" and x.get("aligned")),
                                             sum(1 for r in got.get(lab) or [] for x in r["rows"] if x["vector"] != "none")]
                                       for lab in ("e4 steer P", "e4 steer N")},
                     "loo": sorted({str(r.get("loo")) for lab in ("e4 patch", "e4 steer P", "e4 steer N") for r in got.get(lab) or []})}
    out["secs_per_item"] = {lab: st.mean(r["secs"] for r in got[lab] if "secs" in r) if got.get(lab) and
                            any("secs" in r for r in got[lab]) else None for lab, *_ in steps(ctx)}
    return out


def manifest(ctx):
    return {"name": "e4", "plan": "config/interp/plan_next.md, session 1 (E4, the cure)", "layers": ctx["layers"],
            "range": list(RANGE_C), "top8": ctx["top"], "rand8": ctx["rand"], "discovery": ctx["disc"],
            "held_out": ctx["held"], "gate": GATE, "rows": _nrows(ctx["patch_kw"]["rows"]), "estimate_min": ctx.get("estimate_min")}


# --- panel, preview, report ----------------------------------------------------------------------------------------------
def _p(v, d=0):
    return "  —  " if v is None else f"{100 * v:+.{d}f}%"


def _ci(c):
    return f" [{100 * c['ci'][0]:+.0f}, {100 * c['ci'][1]:+.0f}]" if c and c.get("ci") else ""


def live(got):
    li = _li()
    out = []
    g = got.get("e4 gate") or []
    if g and li is not None:
        out.append("gate: " + summarize_gate(g, li)["line"])
    b = [r for r in got.get("e4 build") or [] if r.get("kind") == "fixture"]
    if b:
        B = build_summary(b)
        if B.get("ld"):
            out.append(f"build: {B['usable']}/{B['n']} usable; plain gap G_N {B['G_N']:+.1f} nats, range gap G_P {B['G_P']:+.1f}")
    p = got.get("e4 patch") or []
    if p:
        S = ratio_table(p, "plain", ci=False)
        L = 42 if any(f"plain@42" in S.get(t, {}) for t in S) else None
        if L:
            out.append("patch L42: " + " · ".join(f"{t} {n.split('@')[0]} {_p(S[t].get(n, {}).get('share')).strip()}"
                                                  for t, n in (("N1", "own@42"), ("P1", "plain@42"), ("N6", "own@42"), ("P6", "plain@42"))
                                                  if t in S))
    c = got.get("e4 cure") or []
    if c:
        C = cure_summary(c)
        out.append("cure (N6 <- P6) by layer: " + " ".join(f"{n[5:]}:{_p(v).strip()}" for n, v in
                                                        sorted(C.get("N6", {}).items(), key=lambda x: int(x[0][5:]))[::4]))
    return out


def stub(task, items, ctx):
    """Preview only: canned shapes, not a model."""
    Ls, n, nH = ctx["layers"], ctx["layers"]["n_layers"], ctx["n_heads"]
    sig = lambda x: 1 / (1 + math.exp(-x))                                  # noqa: E731
    if task == "e4_steer":
        out = []
        for it in items:
            t, s = it["target"], SCALE[it["target"]]
            base = (120 if it["target"][0] == "P" else (120 if s == 0 else 0.97 * it["base_limit"])) * 10 ** s
            plan = ctx["gate_kw"]["plan"] if it["item_id"].endswith("gate") else ctx["steerP_kw"]["plan"]
            n_none = ctx["gate_kw"]["n_none"] if it["item_id"].endswith("gate") else ctx["steerP_kw"]["n_none"]
            rows = [{"vector": "none", "layer": None, "alpha": 0.0, "sample": i} for i in range(n_none)]
            rows += [{"vector": v, "layer": L, "alpha": a, "sample": j} for v, L, a, k in plan for j in range(k)]
            for x in rows:
                k = 0.15 if it["target"][0] == "P" else 0.8
                mv = 0 if x["vector"] != "vN" else -it["sign"] * k * x["alpha"] * 0.2 * it["base_limit"]
                x.update(price=int(round((base / 10 ** s + mv) / 5) * 5) * 10 ** s, errors=[], action="OFFER", aligned=True)
            out.append({"item_id": it["item_id"], "fixture": it["fixture"], "role": "seller", "target": t, "sign": it["sign"],
                        "base_limit": it["base_limit"], "rehearsal": False, "rows": rows, "secs": 0.1})
        return out
    if task == "e4_build":
        out = []
        for it in items:
            if it.get("kind") == "nonmoney":
                out.append({"item_id": it["item_id"], "kind": "nonmoney", "secs": 0.01})
                continue
            b = it["base_limit"]
            lay = {}
            for s in ("H", "S", "reply"):
                aa = [1.0 + 20 * sig((L - 40) / 2) for L in range(n)]
                k = 1.0 if s == "H" else [1.0 - 0.6 * sig((L - 44) / 2) for L in range(n)]
                kk = [k] * n if isinstance(k, float) else k
                lay[s] = [aa, [a * q * q for a, q in zip(aa, kk)], [a * q for a, q in zip(aa, kk)],
                          [0.0 if s == "H" else a * (1 - q) ** 2 for a, q in zip(aa, kk)],
                          [0.0 if s == "H" else -a * (1 - q) for a, q in zip(aa, kk)]]
            heads = {str(L): {s: [[1.0] * nH, [0.6] * nH, [0.5] * nH] for s in ("Srep", "reply")} for L in Ls["heads"]}
            out.append({"item_id": it["item_id"], "kind": "fixture", "fixture": it["fixture"], "base_limit": b, "n_layers": n,
                        "n_heads": nH, "head_layers": Ls["heads"], "n_A": 117, "n_H": 38, "n_S": 68, "n_reply": 11,
                        "n_R": {"N1": 0, "N6": 0, "P1": 39, "P6": 51}, "n_tokens": {"N1": 567, "N6": 573, "P1": 606, "P6": 624},
                        "price": {"N1": 120, "N6": 95 * 10 ** 6, "P1": 120, "P6": 120 * 10 ** 6}, "stand_in": False,
                        "ld": {"N1": -12.0, "N6": 5.0, "P1": -13.0, "P6": -9.0}, "layers": lay, "heads": heads,
                        "h_identical": 0.0, "secs": 0.1})
        return out
    if task == "e4_stats":
        per = {s: {"norm_vN": [1.0] * n, "norm_vP": [0.6] * n, "cos_vP_vN": [0.8] * n, "proj_vP_on_vN": [0.5] * n,
                   "norm_c6": [0.5] * n, "proj_c6_on_vN": [-0.4] * n, "cos_c6_vN": [-0.6] * n, "proj_c1_on_vN": [0.05] * n}
               for s in ("A", "H", "S", "reply")}
        return [{"n_fixtures": len(ctx["fx"]), "n_layers": n, "per_segment": per, "n_nonmoney": len(ctx["nonmoney"])}]
    if task == "e4_attn":
        out = []
        for it in items:
            mass, keys = {}, {}
            for k in PROMPTS:
                ks = ["limit"] + (["lo", "hi", "range"] if k[0] == "P" else [])
                mass[k] = [[[[round(0.3 if (L, h % 3) == (40, 1) and kk == "hi" else 0.02, 3) for h in range(nH)] for kk in ks]
                            for _q in range(2)] for L in range(n)]
                keys[k] = ks
            out.append({"item_id": it["item_id"], "fixture": it["fixture"], "mass": mass, "keys": keys, "secs": 0.1})
        return out
    # e4_patch: the k-th call for an item is patch (1), cure (2), then cure heads (discovery) or the circuit (held-out)
    ld0 = {"N1": -12.0, "N6": 5.0, "P1": -13.0, "P6": -9.0}
    seen = ctx.setdefault("_stub_seen", {})

    def share(t, name):
        L = int(name.split("@")[1]) if "@" in name else 42
        if name.startswith("cure"):
            return sig((L - 30) / 3) * (0.5 if " " in name.split("@")[0] else 1.0)
        if name.startswith("h"):
            return 0.2 if name.endswith(".1") and L == 42 else 0.004
        if name.startswith(("top", "window", "random")):
            return {"top1": 0.1, "top4": 0.3, "top8": 0.45}.get(name, 0.05)
        k = 0.25 if t[0] == "P" else 1.0
        return k * (sig((L - 40.5) / 1.2) if name.startswith(("own", "plain", "vN")) else 0.05)
    out = []
    for it in items:
        k = seen[it["item_id"]] = seen.get(it["item_id"], 0) + 1
        kw = {1: ctx["patch_kw"], 2: ctx["cure_kw"]}.get(k) or (ctx["heads_kw"] if it["fixture"] in ctx["disc"] else ctx["circuit_kw"])
        targets = {}
        for t in PROMPTS:
            specs = kw["rows"].get(t, [])
            if kw is ctx["patch_kw"]:
                other = ld0[t] + (17.0 if SCALE[t] == 0 else -17.0)
            else:
                other = ld0["P6" if t == "N6" else "N6"]
            rows = [{"n": x["n"], "ld": round(ld0[t] + share(t, x["n"]) * (other - ld0[t]), 4), "lpa": -0.3, "lpb": -2.0,
                     "p1": [0.0] * 10} for x in specs]
            targets[t] = {"ld_clean": ld0[t], "clean_spread": 0.0, "hooks_ok": True,
                          "rows": rows + [{"n": "clean", "ld": ld0[t], "lpa": -0.3, "lpb": -2.0, "p1": [0.0] * 10}]}
        out.append({"item_id": it["item_id"], "fixture": it["fixture"], "base_limit": it["base_limit"], "ld_build": ld0,
                    "targets": targets, "secs": 0.1})
    return out


def _spark(li, vals):
    return li.spark([None if v is None else max(-0.2, min(1.2, v)) for v in vals])


def report(s, preview):
    li = _li()
    print("E4, the cure (exploratory" + (", PREVIEW: canned shapes" if preview else "") + ")")
    Ls = s.get("layers") or {}
    if Ls.get("mapped"):
        print(f"  NOTE: a {Ls['n_layers']}-layer model; the 31B's layer sets were mapped onto it")
    G = s.get("gate") or {}
    if G.get("cells"):
        print(f"  gate (sampled openings with the range): {G['line']}" + (" [REHEARSAL]" if G.get("rehearsal") else ""))
        print(f"    {'PASS' if G['pass'] else 'FAIL: ' + G['why']}; shift CI {G.get('shift_ci')}; at 120: "
              + ", ".join(f"{t} {_p(G['cells'][t]['at_120']).strip()}" for t in ("P1", "P6"))
              + "; at 125: " + ", ".join(f"{t} {_p(G['cells'][t]['at_125']).strip()}" for t in ("P1", "P6")))
    B = s.get("build") or {}
    if B.get("ld"):
        print(f"  build: {B['usable']}/{B['n']} fixtures usable" + (" (STAND-IN prices: rehearsal)" if B.get("stand_in") else "")
              + f"; aligned {B['n_A']} tokens = H {B['n_H']} + S {B['n_S']} + reply {B['n_reply']}; range sentence "
              f"{B['n_R']['P1']}/{B['n_R']['P6']} tokens; H identical in N and P within {B['h_identical_max']}")
        print("    ld (log p plain-x1e6 price - log p plain-x1 price): " + ", ".join(f"{k} {B['ld'][k]:+.2f}" for k in PROMPTS)
              + f"; G_N {B['G_N']:+.2f}, G_P {B['G_P']:+.2f} nats ({_p(B['G_P'] / B['G_N'] if B['G_N'] else None).strip()} of G_N)")
        print(f"    greedy openings with the range: P1 {B.get('greedy_P1')}, P6 {B.get('greedy_P6')} (x1 units)")
        for seg in ("reply", "S"):
            lay = B["layers"][seg]
            print(f"    {seg:<5} by layer, |dP|/|dN| {_spark(li, lay['ratio'])}  projection of dP on dN {_spark(li, lay['proj'])}")
            pick = [L for L in (36, 38, 40, 41, 42, 43, 44, 46, 48, 52, 56, 59) if L < len(lay["ratio"])] or list(range(len(lay["ratio"])))
            print("      " + "  ".join(f"L{L} {lay['ratio'][L]:.2f}/{lay['proj'][L]:+.2f}/c{lay['cure_proj'][L]:+.2f}"
                                       for L in pick if lay["ratio"][L] is not None) + "   (ratio/proj/cure c6 on dN)")
        H = B.get("heads") or {}
        if H:
            print("    X57's top 8 heads, range vs plain on S + reply (|dP|/|dN|, projection): "
                  + ", ".join(f"L{L}.{h} {H.get(f'h{L}.{h}', {}).get('Srep', {}).get('ratio') or 0:.2f}/"
                              f"{H.get(f'h{L}.{h}', {}).get('Srep', {}).get('proj') or 0:+.2f}" for L, h in s.get("top") or []))
    S = s.get("stats")
    if S and S.get("per_segment"):
        for seg in ("reply", "S"):
            p = S["per_segment"].get(seg)
            if p:
                pick = [L for L in (38, 40, 41, 42, 43, 44, 46, 48) if L < len(p["norm_vN"])] or list(range(len(p["norm_vN"])))
                print(f"  fixture-mean directions ({seg}): " + "  ".join(
                    f"L{L} |vP|/|vN| {p['norm_vP'][L] / p['norm_vN'][L] if p['norm_vN'][L] else 0:.2f} cos {p['cos_vP_vN'][L]:+.2f} "
                    f"c6 {p['proj_c6_on_vN'][L]:+.2f}" for L in pick))
    A = s.get("attention") or {}
    if A.get("top_last_to_hi"):
        for k, v in A["top_last_to_hi"].items():
            print(f"  attention, last token -> HI digits ({k}): " + ", ".join(f"L{L}.{h} {m:.2f}" for L, h, m in v[:8]))
    P = s.get("patch") or {}
    if P:
        cl = s.get("patch_clean") or {}
        print(f"  patch (forced opening; share of the plain gap G_N, 95% CI; p(a)+p(b)); clean ld "
              + ", ".join(f"{k} {v:+.2f}" for k, v in (cl.get("ld") or {}).items()))
        names = []
        for t in PROMPTS:
            names += [n for n in P.get(t, {}) if n not in names]
        for n in names:
            parts = []
            for t in PROMPTS:
                c = P.get(t, {}).get(n)
                parts.append(f"{t} {_p(c['share']).strip():>6}{_ci(c):<12} {c['pk']:.2f}" if c else f"{t} {'—':>6}{'':<17}")
            print(f"    {n:<16} " + " | ".join(parts))
    for lab, key in (("steer P", "steer_P"), ("steer N", "steer_N")):
        Sx = s.get(key) or {}
        if Sx:
            print(f"  {lab} (real openings; points moved and share of the plain gap from {s.get('steer_ref') or 'no E0 run'}):")
            for t, cells in Sx.items():
                for k, c in sorted(cells.items()):
                    extra = "" if k == "none" else (f"  moved {c.get('moved_pts') or 0:+.1f} pts, share {_p(c.get('share')).strip()}{_ci(c)}")
                    print(f"    {t} {k:<14} gap {_p(c['mean_gap'], 1).strip():>7}  valid {c['valid']}/{c['n']}  off-scale "
                          f"{c['off_scale']}  round-ten {_p(c.get('round10')).strip()}{extra}")
    C = s.get("cure") or {}
    if C:
        cc = s.get("cure_clean") or {}
        print("  cure by layer (share of the ld distance between N6 and P6 moved by swapping S + reply at one layer):")
        for t, lab in (("N6", "P6 into N6"), ("P6", "N6 into P6")):
            rows = sorted(((int(n.split("@")[1]), c["share"]) for n, c in C.get(t, {}).items() if n.startswith("cure@")))
            if rows:
                print(f"    {lab}: {_spark(li, [v for _L, v in rows])}  " + " ".join(f"L{L} {_p(v).strip()}" for L, v in rows[::3]))
            seg = sorted((n, c["share"]) for n, c in C.get(t, {}).items() if not n.startswith("cure@"))
            if seg:
                print("      " + ", ".join(f"{n} {_p(v).strip()}" for n, v in seg))
        if cc.get("ld"):
            print("    clean ld " + ", ".join(f"{k} {v:+.2f}" for k, v in cc["ld"].items()))
    W = s.get("window")
    if W:
        print(f"  cure window: L{W['band'][0]}-L{W['band'][-1]} ({W['rule']})")
    pk = s.get("pick")
    if pk:
        print("  cure heads (discovery): " + ", ".join(f"L{L}.{h}" for L, h in pk["top"]) + "; random " +
              ", ".join(f"L{L}.{h}" for L, h in pk["random"]))
    K = s.get("circuit") or {}
    if K:
        print("  cure circuit (held-out; share, 95% CI) P6 into N6 | N6 into P6:")
        for n in sorted({n for t in K for n in K[t]}):
            print(f"    {n:<14} " + " | ".join(f"{_p(K.get(t, {}).get(n, {}).get('share')).strip()}{_ci(K.get(t, {}).get(n))}"
                                             for t in ("N6", "P6")))
    ch = s.get("checks") or {}
    print(f"  checks: hooks fired as planned {ch.get('hooks_ok')}; clean rows agree within {ch.get('clean_spread_max')} nats; "
          f"steered openings aligned {ch.get('steer_aligned')}; leave-one-out {ch.get('loo')}")
    sec = s.get("secs_per_item") or {}
    print("  seconds per item: " + ", ".join(f"{k[3:]} {v:.1f}" for k, v in sec.items() if v)
          + (f" (estimated {s['estimate_min'][0]} min typical)" if s.get("estimate_min") else ""))
