"""interp_heads_laptop.py (LAPTOP-SIDE, standard library only): which attention heads carry the seller's limit into
the reply? The plug-in scripts/interp/exp_heads.py is a shim importing from here (keeps the pod's create request small).

    python3 scripts/launch_interp.py --preview --experiment heads             (free: canned shapes)
    python3 scripts/launch_interp.py --model gemma-4-31b-it --experiment heads --slot 8      (paid)

config/interp/plan_next.md, session 2 (STORY X56). Gemma 4 31B, sellers, D2 "81 dollars" vs D8 "81000000 dollars".
Exploratory.

Readout: E2's setup (the reply's opening forced). Per fixture, a = the x1 run's greedy price and b = the x1e6 run's,
each written in the run's own units; ld = log p(b's next token) - log p(a's) at the first token where they differ.
A row's share of the gap = mean(ld_row - ld_clean) / mean(ld_source - ld_clean) over fixtures (paired bootstrap
CIs over fixtures). Continuous, where greedy prices move in round steps.

Steps:
  build      both greedy prices, ld in each run, the logit lens of ld at every layer, and the x1e6 - x1
             difference at the steering layer (layer 20; kept on the pod).
  attention  attention mass on the limit's digits per layer and head, from the reply's opening tokens, the
             prompt's post-limit tokens and the last token.
  sweep 1    discovery fixtures (Y01-Y10): every head at layers 38-44 patched alone (source run's values, all
             post-limit tokens), plus residual / attention / MLP patches by layer.
  circuit    held-out fixtures (Y11-Y30), heads picked on the discovery fixtures: the top heads jointly (all
             post-limit tokens, reply only, prompt only), the whole band's attention, the other heads, random
             heads; and the latch test: steer at layer 20 (leave-one-fixture-out), block attention from every
             token after the limit to its digits (layers 21-59, the band, or the top heads), and both.
  sweep 2    the held-out fixtures' head map (last, so a time stop only costs map resolution).
"""
import math
import random
import statistics as st

DIRS = ("x1->x1e6", "x1e6->x1")             # E2's names: source -> target
CANON = 60                                  # layer sets below are for Gemma 4 31B; mapped for other depths
RESID_60 = sorted(set(range(0, 59, 2)) | {59} | set(range(36, 47)))
COMP_60 = sorted(set(range(14, 35, 2)) | set(range(36, 47)))
# Timing model (E2 on the pod, X52: ~2.1 s per batch of 16 including ~0.4 s of decoding; here no decoding).
BATCH_S, SRC_S, BUILD_S, ATTN_S, SLOW = 1.8, 0.4, 2.8, 1.4, 1.5
_LI = []


def _li():
    if _LI:
        return _LI[0]
    import sys
    m = sys.modules.get("__main__")
    return m if hasattr(m, "boot_ci") else sys.modules.get("launch_interp")


# --- flags ---------------------------------------------------------------------------------------------------
def add_args(ap):
    ap.add_argument("--heads-layers", default="38-44", help="heads: layers whose heads are patched one by one (default 38-44)")
    ap.add_argument("--heads-discovery", type=int, default=10, help="heads: fixtures used to pick the top heads (default 10)")
    ap.add_argument("--heads-top", type=int, default=8, help="heads: heads tested jointly on the held-out fixtures (default 8)")
    ap.add_argument("--heads-steer-layer", type=int, default=20, help="heads: early steering layer for the latch test (default 20)")
    ap.add_argument("--heads-ko-from", type=int, default=21, help="heads: first layer of the 'all layers' knockout (default 21)")


def configure(a):
    a.roles = "seller"


def _range(s):
    out = set()
    for part in str(s).split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            out |= set(range(int(lo), int(hi) + 1))
        elif part:
            out.add(int(part))
    return sorted(out)


def map_layer(L, n):
    return L if n == CANON else min(n - 1, int(round(L * (n - 1) / (CANON - 1))))


def layer_sets(n, opts):
    """The layer sets for a model with n layers (Gemma 4 31B: n = 60; other depths are mapped, e.g. the rehearsal)."""
    m = lambda Ls: sorted({map_layer(L, n) for L in Ls})              # noqa: E731
    steer = map_layer(opts["steer"], n)
    return {"resid": m(RESID_60) if n == CANON else list(range(n)), "comp": m(COMP_60), "heads": m(opts["heads"]),
            "steer": steer, "ko_from": max(map_layer(opts["ko_from"], n), steer + 1), "n_layers": n,
            "mapped": n != CANON}


# --- row specs (the pod executes them as given) --------------------------------------------------------------
def sweep_rows(layers, nH):
    rows = [{"n": f"resid@{L}", "p": [["resid", L, -1, "post"]]} for L in layers["resid"]]
    for L in layers["comp"]:
        rows += [{"n": f"attn@{L}", "p": [["attn", L, -1, "post"]]}, {"n": f"mlp@{L}", "p": [["mlp", L, -1, "post"]]}]
    rows += [{"n": f"h{L}.{h}", "p": [["head", L, h, "post"]]} for L in layers["heads"] for h in range(nH)]
    return rows


def top_sizes(k):
    return sorted({1, max(1, k // 2), k}) if k else []


def circuit_rows(top, rand, layers, nH):
    hp = lambda hs, sc="post": [["head", L, h, sc] for L, h in hs]    # noqa: E731
    band = layers["heads"]
    comp = [(L, h) for L in band for h in range(nH) if (L, h) not in set(map(tuple, top))]
    ko = {"all": [[L, -1] for L in range(layers["ko_from"], layers["n_layers"])], "band": [[L, -1] for L in band],
          "top": [[L, h] for L, h in top]}
    k = len(top)
    rows = [{"n": f"top{n}", "p": hp(top[:n])} for n in top_sizes(k)]
    rows += [{"n": f"top{k} reply", "p": hp(top, "reply")}, {"n": f"top{k} prompt", "p": hp(top, "prompt")},
             {"n": "band attn", "p": [["attn", L, -1, "post"] for L in band]},
             {"n": "other heads", "p": hp(comp)}, {"n": f"random{len(rand)}", "p": hp(rand)}, {"n": "steer", "s": 1.0}]
    for key in ("all", "band", "top"):
        rows += [{"n": f"ko {key}", "k": ko[key]}, {"n": f"steer+ko {key}", "s": 1.0, "k": ko[key]}]
    return rows


def _batches(n_rows, batch):
    return int(math.ceil((n_rows + 1) / batch))


def estimate_min(n_pairs, n_disc, n_held, n_sweep, n_circ, batch):
    """(minutes typical, minutes if every forward is 1.5x slower)."""
    t = n_pairs * (BUILD_S + ATTN_S) + (n_disc + n_held) * (_batches(n_sweep, batch) * BATCH_S + SRC_S) \
        + n_held * (_batches(n_circ, batch) * BATCH_S + SRC_S)
    return t / 60, t * SLOW / 60


# --- setup ---------------------------------------------------------------------------------------------------
def build_items(li, settings, n_fix):
    e2 = li.build_e2_items(settings, n_fix)                         # fixture-major; both directions per fixture
    pairs = [{"item_id": f"{it['fixture']}-S-pair", "fixture": it["fixture"], "role": it["role"],
              "base_limit": it["base_limit"], "x1": it["source"], "x1e6": it["target"]}
             for it in e2 if it["direction"] == "x1->x1e6"]
    return pairs, e2


def setup(a, settings, schema, li):
    _LI[:] = [li]
    opts = {"heads": _range(a.heads_layers), "steer": a.heads_steer_layer, "ko_from": a.heads_ko_from,
            "top": a.heads_top, "discovery": a.heads_discovery}
    pairs, e2 = build_items(li, settings, a.fixtures)
    fix = [p["fixture"] for p in pairs]
    n_disc = min(opts["discovery"], max(1, len(fix) // 3))
    disc, held = fix[:n_disc], fix[n_disc:]
    layers = layer_sets(CANON, opts)
    nH = 32
    k = min(opts["top"], len(layers["heads"]) * nH // 2)
    sweep = sweep_rows(layers, nH)
    circ_guess = circuit_rows([(layers["heads"][0], h) for h in range(k)], [(layers["heads"][-1], h) for h in range(k)], layers, nH)
    typ, slow = estimate_min(len(pairs), 2 * len(disc), 2 * len(held), len(sweep), len(circ_guess), a.batch)
    li.STEP_GUESS_S.update({"heads build": BUILD_S, "heads attention": ATTN_S,
                            "heads sweep 1": _batches(len(sweep), a.batch) * BATCH_S + SRC_S,
                            "heads sweep 2": _batches(len(sweep), a.batch) * BATCH_S + SRC_S,
                            "heads circuit": _batches(len(circ_guess), a.batch) * BATCH_S + SRC_S})
    ctx = {"li": li, "opts": opts, "pairs": pairs, "disc": disc, "held": held,
           "e2_disc": [it for it in e2 if it["fixture"] in disc], "e2_held": [it for it in e2 if it["fixture"] in held],
           "layers": layers, "n_heads": nH, "k_top": k, "pick": None,
           "build_kw": {"steer_layer": layers["steer"]}, "attn_kw": {},
           "sweep_kw": {"rows": sweep, "batch": a.batch}, "circuit_kw": {"rows": [], "batch": a.batch},
           "estimate_min": [round(typ, 1), round(slow, 1)]}
    plan = (f"Heads on {a.model} (sellers): {len(pairs)} fixture pairs; ld readout (log-prob difference of the two "
            f"runs' greedy prices at the deciding token) in E2's forced-opening setup. Build (prices, logit lens, "
            f"layer-{layers['steer']} difference) and attention to the limit's digits; sweep = every head at layers "
            f"{layers['heads'][0]}-{layers['heads'][-1]} alone + residual/attention/MLP by layer ({len(sweep)} rows per "
            f"item) on {len(disc)} discovery fixtures, then {len(circ_guess)} circuit rows (top {k} heads picked on "
            f"them; latch test: steer layer {layers['steer']} x knockout of attention to the digits) and the sweep on "
            f"{len(held)} held-out fixtures; batch {a.batch}; about {typ:.0f} min of experiment time ({slow:.0f} if "
            f"forwards run 1.5x slower; --budget-min {a.budget_min:g} stops new items)")
    inputs = [{"step": "pairs", **p} for p in pairs] + [{"step": "patch", "item_id": it["item_id"]} for it in e2]
    return {"exp": "Heads", "plan": plan, "inputs": inputs, "ctx": ctx, "steps": steps, "live": live,
            "after_step": after_step}


def steps(ctx):
    return [("heads build", "heads_build", ctx["pairs"], ctx["build_kw"], 5),
            ("heads attention", "heads_attn", ctx["pairs"], ctx["attn_kw"], 5),
            ("heads sweep 1", "heads_patch", ctx["e2_disc"], ctx["sweep_kw"], 2),
            ("heads circuit", "heads_patch", ctx["e2_held"], ctx["circuit_kw"], 2),
            ("heads sweep 2", "heads_patch", ctx["e2_held"], ctx["sweep_kw"], 2)]


def on_pod_info(ctx, pod_info):
    n = pod_info.get("num_hidden_layers")
    if n and int(n) != ctx["layers"]["n_layers"]:
        ctx["layers"] = layer_sets(int(n), ctx["opts"])
        ctx["build_kw"]["steer_layer"] = ctx["layers"]["steer"]
        ctx["sweep_kw"]["rows"][:] = sweep_rows(ctx["layers"], ctx["n_heads"])


def after_step(label, got, ctx):
    if label == "heads build":
        recs = got.get("heads build") or []
        use = [r for r in recs if not r.get("skipped")]
        if len(use) < 3:
            return f"only {len(use)} fixtures have usable greedy prices (need 3)"
        n, nH = use[0]["n_layers"], use[0]["n_heads"]
        if n != ctx["layers"]["n_layers"]:
            ctx["layers"] = layer_sets(n, ctx["opts"])
        if use[0]["steer_layer"] != ctx["layers"]["steer"]:
            return "the pod built the steering difference at another layer"
        ctx["n_heads"] = nH
        ctx["k_top"] = min(ctx["opts"]["top"], len(ctx["layers"]["heads"]) * nH // 2)
        ctx["sweep_kw"]["rows"][:] = sweep_rows(ctx["layers"], nH)
        if ctx.get("panel") is not None:
            ctx["panel"].note(f"build: {len(use)}/{len(recs)} fixtures usable; {nH} heads per layer; "
                              f"{len(ctx['sweep_kw']['rows'])} sweep rows per item")
        return None
    if label == "heads sweep 1":
        S = shares(got.get("heads sweep 1") or [], ci=False)
        top, rand, info = pick_heads(S, ctx["layers"]["heads"], ctx["n_heads"], ctx["k_top"])
        if not top:
            return "no head has readings in both directions on the discovery fixtures"
        ctx["pick"] = dict(info, top=[list(x) for x in top], random=[list(x) for x in rand])
        ctx["circuit_kw"]["rows"][:] = circuit_rows(top, rand, ctx["layers"], ctx["n_heads"])
        if ctx.get("panel") is not None:
            ctx["panel"].note("top heads: " + ", ".join(f"L{L}.{h}" for L, h in top))
    return None


# --- analysis ------------------------------------------------------------------------------------------------
def ratio_ci(num, den, n=1000, seed=1):
    """Paired bootstrap over fixtures of mean(num) / mean(den)."""
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


def _per_fixture(recs):
    """{direction: {fixture: (ld_clean, ld_source, {row name: ld})}} for usable items."""
    out = {}
    for r in recs:
        if r.get("skipped") or r.get("ld_clean") is None or r.get("ld_source") is None:
            continue
        out.setdefault(r["direction"], {})[r["fixture"]] = (r["ld_clean"], r["ld_source"],
                                                            {x["n"]: x["ld"] for x in r["rows"] if x["n"] != "clean"})
    return out


def shares(recs, ci=True, n_boot=400):
    """{direction: {row name: {share, ci, n, moved (nats), gap (nats)}}}."""
    out = {}
    for d, per in _per_fixture(recs).items():
        names = sorted({n for _c, _s, rows in per.values() for n in rows})
        for name in names:
            num = [rows[name] - c for c, s, rows in per.values() if name in rows]
            den = [s - c for c, s, rows in per.values() if name in rows]
            D = st.mean(den) if den else 0
            cell = {"n": len(num), "share": st.mean(num) / D if D else None, "ci": None,
                    "moved": st.mean(num) if num else None, "gap": D or None}
            if ci and D:
                cell["ci"] = ratio_ci(num, den, n_boot)
            out.setdefault(d, {})[name] = cell
    return out


def interactions(recs, keys=("all", "band", "top"), n_boot=1000):
    """Share of the gap for (steer + knockout) - knockout - steer, per direction and knockout set."""
    out = {}
    for d, per in _per_fixture(recs).items():
        for key in keys:
            a, b, s = f"steer+ko {key}", f"ko {key}", "steer"
            num, den = [], []
            for c, src, rows in per.values():
                if a in rows and b in rows and s in rows:
                    num.append((rows[a] - c) - (rows[b] - c) - (rows[s] - c))
                    den.append(src - c)
            if num and st.mean(den):
                out.setdefault(d, {})[key] = {"n": len(num), "share": st.mean(num) / st.mean(den),
                                              "ci": ratio_ci(num, den, n_boot)}
    return out


def head_score(S, name):
    v = [S.get(d, {}).get(name, {}).get("share") for d in DIRS]
    return None if None in v else sum(v) / 2


def pick_heads(S, head_layers, nH, k, seed="scale-invariance/interp/heads/random/v1"):
    """Top k heads by the mean of the two directions' shares (discovery fixtures); k random others as a control."""
    scored = [((L, h), head_score(S, f"h{L}.{h}")) for L in head_layers for h in range(nH)]
    scored = [x for x in scored if x[1] is not None]
    scored.sort(key=lambda x: (-x[1], x[0]))
    top = [x[0] for x in scored[:k]]
    pool = [x[0] for x in scored if x[0] not in top]
    rand = sorted(random.Random(seed).sample(pool, min(k, len(pool))))
    return top, rand, {"rule": f"top {k} by the mean share of both directions", "scores":
                       {f"h{L}.{h}": round(v, 4) for (L, h), v in scored[:3 * k]}}


def lens_summary(build):
    use = [r for r in build if not r.get("skipped")]
    if not use:
        return None
    nL = use[0]["n_layers"]
    final = st.mean(r["ld"]["x1e6"] - r["ld"]["x1"] for r in use)
    diff = [st.mean(r["lens"]["x1e6"][L] - r["lens"]["x1"][L] for r in use) for L in range(nL)]
    frac = [x / final if final else None for x in diff]
    stays = next((L for L in range(nL) if all(f is not None and f >= 0.5 for f in frac[L:])), None)
    return {"n_fixtures": len(use), "final_gap": final, "gap_by_layer": diff, "fraction": frac, "stays_above_half_from": stays,
            "x1": [st.mean(r["lens"]["x1"][L] for r in use) for L in range(nL)],
            "x1e6": [st.mean(r["lens"]["x1e6"][L] for r in use) for L in range(nL)]}


def attention_summary(recs):
    ok = [r for r in recs if not r.get("error") and r.get("mass", {}).get("x1") and r["mass"]["x1"][0]]
    if not ok:
        return {"error": next((r["error"] for r in recs if r.get("error")), "no records")}
    nL, nH = len(ok[0]["mass"]["x1"]), len(ok[0]["mass"]["x1"][0][0])
    mean = {c: [[[st.mean(r["mass"][c][L][q][h] for r in ok) for h in range(nH)] for q in range(3)] for L in range(nL)]
            for c in ("x1", "x1e6")}
    both = [((L, h), (mean["x1"][L][0][h] + mean["x1e6"][L][0][h]) / 2) for L in range(nL) for h in range(nH)]
    both.sort(key=lambda x: -x[1])
    return {"n_fixtures": len(ok), "n_layers": nL, "n_heads": nH, "queries": ["reply", "prompt", "last"], "mean": mean,
            "top_reply": [[L, h, round(v, 4), round(mean["x1"][L][0][h], 4), round(mean["x1e6"][L][0][h], 4)] for (L, h), v in both[:16]]}


def spearman(xs, ys):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            for t in range(i, j + 1):
                r[o[t]] = (i + j) / 2
            i = j + 1
        return r
    if len(xs) < 3:
        return None
    rx, ry = rank(xs), rank(ys)
    mx, my = st.mean(rx), st.mean(ry)
    sx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    sy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / (sx * sy) if sx and sy else None


def summarize(got, ctx):
    build = got.get("heads build") or []
    s1, s2, circ = got.get("heads sweep 1") or [], got.get("heads sweep 2") or [], got.get("heads circuit") or []
    A = attention_summary(got.get("heads attention") or [])
    sweep_all = shares(s1 + s2)
    out = {"layers": ctx["layers"], "n_heads": ctx["n_heads"], "discovery": ctx["disc"], "held_out": ctx["held"],
           "pick": ctx.get("pick"),
           "build": [{k: r.get(k) for k in ("fixture", "base_limit", "price_x1", "price_x1e6", "stand_in", "skipped",
                                            "ld", "decision", "norm_d")} for r in build],
           "lens": lens_summary(build), "attention": A,
           "sweep_discovery": shares(s1), "sweep_all": sweep_all,
           "circuit": shares(circ, n_boot=1000), "interactions": interactions(circ),
           "checks": {"hooks_ok": all(r.get("hooks_ok", True) for r in s1 + s2 + circ),
                      "clean_spread_max": max((r.get("clean_spread") or 0 for r in s1 + s2 + circ), default=None),
                      "skipped": sorted({r["fixture"] for r in build if r.get("skipped")}),
                      "stand_in": any(r.get("stand_in") for r in build)},
           "secs_per_item": {lab: st.mean(r["secs"] for r in got.get(lab) or []) if got.get(lab) else None
                             for lab in ("heads build", "heads attention", "heads sweep 1", "heads circuit", "heads sweep 2")},
           "estimate_min": ctx.get("estimate_min")}
    if "mean" in A:
        hs = [(L, h) for L in ctx["layers"]["heads"] for h in range(ctx["n_heads"]) if L < A["n_layers"]]
        xs = [(A["mean"]["x1"][L][0][h] + A["mean"]["x1e6"][L][0][h]) / 2 for L, h in hs]
        ys = [head_score(sweep_all, f"h{L}.{h}") for L, h in hs]
        keep = [(x, y) for x, y in zip(xs, ys) if y is not None]
        out["attention_vs_patch_spearman"] = spearman([x for x, _ in keep], [y for _, y in keep])
    return out


def manifest(ctx):
    return {"name": "heads", "plan": "config/interp/plan_next.md, session 2", "layers": ctx["layers"], "opts": ctx["opts"],
            "discovery": ctx["disc"], "held_out": ctx["held"], "sweep_rows": len(ctx["sweep_kw"]["rows"]),
            "estimate_min": ctx.get("estimate_min")}


# --- panel, preview, report ----------------------------------------------------------------------------------
def _pct(v):
    return "  —  " if v is None else f"{100 * v:+.0f}%"


def _ci(c):
    return f" [{100 * c['ci'][0]:+.0f}, {100 * c['ci'][1]:+.0f}]" if c and c.get("ci") else ""


def live(got):
    li = _li()
    out = []
    b = got.get("heads build") or []
    if b:
        L = lens_summary(b)
        out.append(f"build: {sum(1 for r in b if not r.get('skipped'))}/{len(b)} fixtures usable"
                   + (f"; logit lens holds >= 50% of the final gap from layer {L['stays_above_half_from']}" if L else ""))
    a = got.get("heads attention") or []
    if a:
        A = attention_summary(a)
        if "top_reply" in A:
            out.append("attention, reply -> digits: " + ", ".join(f"L{x[0]}.{x[1]} {x[2]:.2f}" for x in A["top_reply"][:5]))
    for lab in ("heads sweep 1", "heads sweep 2"):
        r = got.get(lab) or []
        if r:
            S = shares(r, ci=False)
            sc = sorted(((n, head_score(S, n)) for n in S.get(DIRS[0], {}) if n.startswith("h")),
                        key=lambda x: -(x[1] if x[1] is not None else -9))[:5]
            out.append(f"{lab} ({len({x['fixture'] for x in r})} fixtures) top heads: "
                       + ", ".join(f"{n} {_pct(v).strip()}" for n, v in sc if v is not None))
    c = got.get("heads circuit") or []
    if c and li is not None:
        S = shares(c, ci=False)
        for d in DIRS:
            parts = [f"{n} {_pct(S.get(d, {}).get(n, {}).get('share')).strip()}" for n in ("steer", "ko all", "steer+ko all")
                     if n in S.get(d, {})]
            top = sorted((n for n in S.get(d, {}) if n.startswith("top") and " " not in n), key=lambda n: int(n[3:]))
            parts = [f"{n} {_pct(S[d][n]['share']).strip()}" for n in top] + parts
            out.append(f"circuit {d}: " + " · ".join(parts))
    return out


def stub(task, items, ctx):
    """Preview only: canned shapes, not a model."""
    n, nH = ctx["layers"]["n_layers"], ctx["n_heads"]
    sig = lambda x: 1 / (1 + math.exp(-x))                                  # noqa: E731
    if task == "heads_build":
        out = []
        for it in items:
            b = it["base_limit"]
            lens = {c: [round(s * (3.0 * sig((L - 41) / 2) + 1.0 * sig((L - 22) / 3)) / 4 * 4, 4) for L in range(n)]
                    for c, s in (("x1", -1), ("x1e6", 1))}
            out.append({"item_id": it["item_id"], "fixture": it["fixture"], "base_limit": b, "n_layers": n, "n_heads": nH,
                        "head_dim": [256] * n, "steer_layer": ctx["build_kw"]["steer_layer"], "price_x1": round(1.3 * b),
                        "price_x1e6": round(1.2 * b) * 10 ** 6, "stand_in": False, "ld": {"x1": -4.0, "x1e6": 4.0},
                        "lens": lens, "n_post": 117, "n_prompt": 106, "decision": {}, "norm_d": 4.9, "secs": 0.1})
        return out
    if task == "heads_attn":
        def m(c, L, q, h):
            hot = (L, h) in ((41, 7), (41, 19), (42, 3), (44, 11))
            return round((0.4 if hot else 0.02) * (1.3 if c == "x1e6" else 1.0) * (1.0 if q == 0 else 0.5), 4)
        return [{"item_id": it["item_id"], "fixture": it["fixture"], "n_digits": {"x1": 2, "x1e6": 8},
                 "mass": {c: [[[m(c, L, q, h) for h in range(nH)] for q in range(3)] for L in range(n)] for c in ("x1", "x1e6")},
                 "secs": 0.1} for it in items]
    hot = {(41, 7): 0.25, (41, 19): 0.15, (42, 3): 0.10, (44, 11): 0.06}

    def share(spec, d):
        late = d == "x1->x1e6"
        s = 0.0
        for kind, L, h, sc in spec.get("p", ()):
            f = 0.3 if sc == "prompt" else (0.8 if sc == "reply" else 1.0)
            if kind == "resid":
                s += sig((L - 40.5) / 1.2) if late else sig((L - 22) / 3)
            elif kind == "attn":
                s += 0.55 * math.exp(-((L - 41) / 1.5) ** 2)
            elif kind == "mlp":
                s += 0.0 if late else 0.15 * math.exp(-((L - 22) / 4) ** 2)
            else:
                s += f * hot.get((L, h), 0.003)
        if spec.get("s"):
            s += 0.12 if late else 0.5
        if spec.get("k"):
            s += (0.55 if late else 0.3) * min(1.0, len(spec["k"]) / 30)
            if spec.get("s") and late:
                s += 0.3
        return min(s, 1.1)
    out = []
    seen = ctx.setdefault("_stub_seen", {})
    for it in items:
        k = seen.get(it["item_id"], 0)
        seen[it["item_id"]] = k + 1
        specs = ctx["circuit_kw"]["rows"] if it["fixture"] not in ctx["disc"] and k == 0 else ctx["sweep_kw"]["rows"]
        jit = (sum(map(ord, it["fixture"])) % 7 - 3) / 20
        c, src = (4.0 + jit, -4.0) if it["target"]["condition"] == "D8" else (-4.0 + jit, 4.0)
        rows = [{"n": spec["n"], "ld": round(c + share(spec, it["direction"]) * (src - c), 4), "lpa": -1.0, "lpb": -2.0,
                 "pd": 0.99, "p1": [0.0] * 10} for spec in specs]
        out.append({"item_id": it["item_id"], "fixture": it["fixture"], "role": "seller", "direction": it["direction"],
                    "base_limit": it["base_limit"], "target_condition": it["target"]["condition"], "ld_source": src,
                    "ld_clean": c, "clean_spread": 0.0, "hooks_ok": True,
                    "rows": rows + [{"n": "clean", "ld": c, "lpa": -1, "lpb": -2, "pd": 0.99, "p1": [0.0] * 10}], "secs": 0.1})
    return out


def _spark(li, vals):
    return li.spark([None if v is None else max(-0.2, min(1.2, v)) for v in vals])


def report(s, preview):
    li = _li()
    print("Heads (exploratory" + (", PREVIEW: canned shapes" if preview else "") + ")")
    Ls = s.get("layers") or {}
    if Ls.get("mapped"):
        print(f"  NOTE: a {Ls['n_layers']}-layer model; the 31B's layer sets were mapped onto it")
    ch = s.get("checks") or {}
    if ch.get("stand_in"):
        print("  NOTE: stand-in prices (REHEARSAL: random weights); every number below is meaningless")
    use = [b for b in s.get("build") or [] if not b.get("skipped")]
    print(f"  build: {len(use)}/{len(s.get('build') or [])} fixtures usable" + (f" (skipped {', '.join(ch['skipped'])})" if ch.get("skipped") else "")
          + f"; hooks fired as planned: {ch.get('hooks_ok')}; clean rows agree within {ch.get('clean_spread_max')} nats")
    if use:
        print(f"  ld (log p x1e6 price - log p x1 price at the deciding token): x1 runs {st.mean(b['ld']['x1'] for b in use):+.2f}, "
              f"x1e6 runs {st.mean(b['ld']['x1e6'] for b in use):+.2f} nats")
    Lz = s.get("lens")
    if Lz:
        f = Lz["fraction"]
        print(f"  logit lens at the deciding token: share of the final x1e6 - x1 gap ({Lz['final_gap']:.2f} nats) by layer "
              f"(▁ 0, █ 100%): {_spark(li, f)}")
        print(f"    stays >= 50% from layer {Lz['stays_above_half_from']}; "
              + ", ".join(f"L{L} {_pct(f[L]).strip()}" for L in range(0, len(f), max(1, len(f) // 12))))
    A = s.get("attention") or {}
    if A.get("top_reply"):
        print(f"  attention from the reply's opening tokens to the limit's digits ({A['n_fixtures']} fixtures), top heads (mean of x1/x1e6; x1, x1e6):")
        print("    " + "  ".join(f"L{L}.{h} {v:.2f} ({a:.2f}, {b:.2f})" for L, h, v, a, b in A["top_reply"][:8]))
        per = [max(max(A["mean"]["x1"][L][0]), max(A["mean"]["x1e6"][L][0])) for L in range(A["n_layers"])]
        print(f"    strongest head per layer: {li.spark(per, 0, max(per) or 1)} (L0..L{A['n_layers'] - 1}; max {max(per):.2f} at L{per.index(max(per))})")
    elif A.get("error"):
        print(f"  attention: {A['error']}")
    S = s.get("sweep_all") or {}
    if S:
        print("  sweep, share of the gap restored (all fixtures):")
        for d in DIRS:
            for kind, Lset in (("resid", Ls.get("resid", [])), ("attn", Ls.get("comp", [])), ("mlp", Ls.get("comp", []))):
                vals = [S.get(d, {}).get(f"{kind}@{L}", {}).get("share") for L in Lset]
                if any(v is not None for v in vals):
                    v, L = li.peak(Lset, vals)
                    print(f"    {d:<9} {kind:<5} L{Lset[0]}..L{Lset[-1]}  {_spark(li, vals)}  peak L{L} {_pct(v).strip()}")
        heads = sorted({n for d in DIRS for n in S.get(d, {}) if n.startswith("h")},
                       key=lambda n: -(head_score(S, n) if head_score(S, n) is not None else -9))
        print(f"    top heads (mean of both directions; {DIRS[0]} / {DIRS[1]}, 95% CI):")
        for n in heads[:12]:
            c0, c1 = S.get(DIRS[0], {}).get(n, {}), S.get(DIRS[1], {}).get(n, {})
            print(f"      {n:<7} {_pct(head_score(S, n)).strip():>5}   {_pct(c0.get('share')).strip()}{_ci(c0)} / {_pct(c1.get('share')).strip()}{_ci(c1)}")
        for L in Ls.get("heads", []):
            hsum = [sum(S.get(d, {}).get(f"h{L}.{h}", {}).get("share") or 0 for h in range(s.get("n_heads") or 0)) for d in DIRS]
            at = [S.get(d, {}).get(f"attn@{L}", {}).get("share") for d in DIRS]
            print(f"      L{L}: sum of single heads {_pct(hsum[0]).strip()} / {_pct(hsum[1]).strip()}; whole attention "
                  f"{_pct(at[0]).strip()} / {_pct(at[1]).strip()}")
        if s.get("attention_vs_patch_spearman") is not None:
            print(f"    heads at the band: attention to the digits vs patch share, Spearman {s['attention_vs_patch_spearman']:+.2f}")
    pk = s.get("pick")
    if pk:
        print(f"  picked on {len(s.get('discovery') or [])} discovery fixtures ({pk['rule']}): "
              + ", ".join(f"L{L}.{h}" for L, h in pk["top"]) + "; random control " + ", ".join(f"L{L}.{h}" for L, h in pk["random"]))
    C = s.get("circuit") or {}
    if C:
        print(f"  circuit, held-out fixtures ({len(s.get('held_out') or [])}); share of the gap (95% CI) and nats moved, {DIRS[0]} | {DIRS[1]}:")
        names = []
        for d in DIRS:
            names += [n for n in C.get(d, {}) if n not in names]
        order = ("top", "band", "other", "random", "steer", "ko all", "steer+ko all", "ko band", "steer+ko band",
                 "ko top", "steer+ko top")

        def key(n):
            g = next((i for i, p in enumerate(order) if n == p or (p in ("top", "other", "random") and n.startswith(p))), 99)
            num = int("".join(ch for ch in n.split(" ")[0] if ch.isdigit()) or 0)
            return (g, num, n)
        for n in sorted(names, key=key):
            c0, c1 = C.get(DIRS[0], {}).get(n, {}), C.get(DIRS[1], {}).get(n, {})
            f = lambda c: f"{_pct(c.get('share')).strip():>5}{_ci(c):<12} {c.get('moved') or 0:+5.2f}"     # noqa: E731
            print(f"    {n:<16} {f(c0)} | {f(c1)}")
        I = s.get("interactions") or {}
        if I:
            print("  latch test, interaction = (steer + knockout) - knockout - steer, share of the gap:")
            for key in ("all", "band", "top"):
                c0, c1 = I.get(DIRS[0], {}).get(key, {}), I.get(DIRS[1], {}).get(key, {})
                print(f"    knockout {key:<5} {_pct(c0.get('share')).strip()}{_ci(c0)} | {_pct(c1.get('share')).strip()}{_ci(c1)}")
    sec = s.get("secs_per_item") or {}
    print("  seconds per item: " + ", ".join(f"{k.split(' ', 1)[1]} {v:.1f}" for k, v in sec.items() if v)
          + (f" (estimated {s['estimate_min'][0]} min typical)" if s.get("estimate_min") else ""))
