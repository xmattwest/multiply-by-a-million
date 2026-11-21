"""interp_heads2_laptop.py (LAPTOP-SIDE, standard library only): follow X57's key heads. The plug-in
scripts/interp/exp_heads2.py is a shim importing from here (keeps the pod's create request small).

    python3 scripts/launch_interp.py --preview --experiment heads2            (free: canned shapes)
    python3 scripts/launch_interp.py --model gemma-4-31b-it --experiment heads2 --slot 8     (paid)

config/interp/plan_next.md, session 2 "Follow the key heads" (STORY X57). Gemma 4 31B, D2 "81 dollars" vs D8
"81000000 dollars". Exploratory. Readout as X57: E2's setup (the reply's opening forced); ld = log p(x1e6 run's greedy
price) - log p(x1 run's) at the first token where they differ, both in the run's units; share of the gap =
mean(ld_row - ld_clean) / mean(ld_source - ld_clean) over fixtures (paired bootstrap CIs over fixtures).

Steps (ordered so that a time stop loses the least):
  seller build    X57's build for sellers (greedy prices, decision tokens) into this plug-in's seller store.
  sources         X57's top 8 heads + the attention-heavy contrast heads: attention from the reply's 11 opening tokens
                  over every earlier position (x1 and x1e6 runs), value-weighted attention, and an exact split of each
                  head's x1e6 - x1 output difference by source position; for every head at 38-44, the size of that
                  difference.
  path            path patching into the top 8 heads (receivers), all 60 seller items: positions from the source run
                  at the layers feeding the receivers (by region; the heads' most-read and most-different positions; a
                  matched random control), the source run's digits alone, and the post-limit tokens patched at
                  earlier layers (20-38); X57's direct row (receivers <- source outputs) and plain residual patches.
  buyer build     X57's build for buyers (fixtures whose prices coincide after rescaling are skipped, and said so).
  buyer sweep 1   discovery fixtures (Y01-Y10): every head at 38-44 alone, attention/MLP at 38-44, residual at a few
                  layers, and the sellers' top 8 jointly.
  buyer circuit   held-out fixtures: the buyers' own top heads jointly (picked on discovery), the sellers' top 8, the
                  overlap, random heads, the band, the rest of the band.
  buyer sweep 2   the held-out fixtures' head map (last: a time stop only costs the replication check).
"""
import json
import math
import os
import random
import statistics as st

import interp_heads_laptop as X                # X57's laptop code (reused, not edited)

DIRS = X.DIRS
CANON = 60
X57_TOP = "41.29,39.28,42.9,41.27,41.10,41.11,43.5,39.22"
CONTRAST = "43.5,41.26,54.23,42.9"
BACK_60 = (20, 24, 28, 32, 36, 38)
RESID_REF_60 = BACK_60 + (40, 41)
BRESID_60 = (20, 28, 36, 38, 39, 40, 41, 42, 43, 44, 48)
X57_RUN = "data/interp/interp-heads-gemma-4-31b-it-20261007T013856Z"
REGIONS = ("bos", "before", "digits", "after", "rest", "reply")
# Timing (X57 on the pod: 0.66 s per batch of 16, 2.2 s per build pair, 0.4 s per attention pair).
BATCH_S, PASSA_S, SRC_S, BUILD_S, SOURCES_S, SLOW = 0.7, 0.55, 0.3, 2.3, 1.5, 1.5
_LI = []


def _li():
    return _LI[0] if _LI else X._li()


# --- flags ---------------------------------------------------------------------------------------------------
def add_args(ap):
    ap.add_argument("--heads2-recv", default=X57_TOP, help="heads2: receiver heads L.h,... (default X57's top 8)")
    ap.add_argument("--heads2-contrast", default=CONTRAST, help="heads2: extra heads for the sources step")
    ap.add_argument("--heads2-back", default=",".join(map(str, BACK_60)), help="heads2: earlier layers for path patching")
    ap.add_argument("--heads2-after", type=int, default=2, help="heads2: post-limit tokens counted as 'right after the digits'")
    ap.add_argument("--heads2-top", type=int, default=8, help="heads2: buyer heads tested jointly on held-out fixtures")
    ap.add_argument("--heads2-discovery", type=int, default=10, help="heads2: buyer fixtures used to pick the top heads")
    ap.add_argument("--heads2-parts", default="sources,path,buyers", help="heads2: which parts to run")
    ap.add_argument("--heads2-x57-run", default=X57_RUN, help="heads2: X57's run folder (its seller head map, for comparison)")


def configure(a):
    a.roles = "seller,buyer"


def parse_heads(s):
    out = []
    for part in str(s).split(","):
        part = part.strip()
        if part:
            L, h = part.split(".")
            if (int(L), int(h)) not in out:
                out.append((int(L), int(h)))
    return out


def map_heads(hs, nL, nH):
    """Map 31B heads onto another geometry (the rehearsal's tiny model); identity for 60 layers x 32 heads."""
    out = []
    for L, h in hs:
        x = (X.map_layer(L, nL), h % nH)
        if x not in out:
            out.append(x)
    return out


def geometry(nL, nH, opts):
    m = lambda Ls: sorted({X.map_layer(L, nL) for L in Ls})            # noqa: E731
    recv = map_heads(opts["recv"], nL, nH)
    lmax = max(L for L, _ in recv)
    back = [L for L in m(opts["back"]) if L < lmax]
    feed = sorted({L - 1 for L, _ in recv if L >= 1})
    return {"n_layers": nL, "n_heads": nH, "mapped": (nL, nH) != (CANON, 32), "recv": recv,
            "sources": map_heads(opts["recv"] + opts["contrast"], nL, nH), "back": back, "feed": feed,
            "resid_ref": [L for L in m(RESID_REF_60)], "band": m(range(38, 45)), "bresid": m(BRESID_60)}


# --- row specs -----------------------------------------------------------------------------------------------
def regions(n_post, n_prompt, n_after):
    return {"post": list(range(n_post)), "prompt": list(range(n_prompt)), "after": list(range(min(n_after, n_prompt))),
            "rest": list(range(min(n_after, n_prompt), n_prompt)), "reply": list(range(n_prompt, n_post))}


def path_rows(G, n_post, n_prompt, n_after, picks=None):
    R = regions(n_post, n_prompt, n_after)
    rows = []

    def path(seq, mode, P, name, L=None):
        r = {"n": name, "kind": "path", "seq": seq, "mode": mode, "P": P}
        if L is not None:
            r["L"] = L
        rows.append(r)
    path("T", "feed", [], "T feed none")
    for k in ("post", "prompt", "after", "rest", "reply"):
        path("T", "feed", R[k], f"T feed {k}")
    if picks:
        for k in ("attn", "diff", "rand_a", "rand_b"):
            path("T", "feed", {f: p[k] for f, p in picks.items()}, f"T feed {k}")
    path("S", "feed", [], "S feed none")
    for k in ("post", "prompt", "reply", "after"):
        path("S", "feed", R[k], f"S feed {k}")
    if picks:
        path("S", "feed", {f: p["diff"] for f, p in picks.items()}, "S feed diff")
    for L in G["back"]:
        for k in ("post", "prompt", "reply"):
            path("T", "back", R[k], f"T back{L} {k}", L)
        path("S", "back", R["post"], f"S back{L} post", L)
    rows.append({"n": "direct", "kind": "direct"})
    for L in sorted({L for L, _ in G["recv"]}):
        rows.append({"n": f"direct L{L}", "kind": "direct", "use": [[a, b] for a, b in G["recv"] if a == L]})
    rows += [{"n": f"resid@{L}", "kind": "resid", "L": L} for L in G["resid_ref"]]
    return rows


def buyer_sweep_rows(G, seller8):
    rows = [{"n": f"resid@{L}", "p": [["resid", L, -1, "post"]]} for L in G["bresid"]]
    for L in G["band"]:
        rows += [{"n": f"attn@{L}", "p": [["attn", L, -1, "post"]]}, {"n": f"mlp@{L}", "p": [["mlp", L, -1, "post"]]}]
    rows += [{"n": f"h{L}.{h}", "p": [["head", L, h, "post"]]} for L in G["band"] for h in range(G["n_heads"])]
    rows.append({"n": "seller8", "p": [["head", L, h, "post"] for L, h in seller8]})
    return rows


def buyer_circuit_rows(top, rand, seller8, G):
    hp = lambda hs, sc="post": [["head", L, h, sc] for L, h in hs]    # noqa: E731
    k = len(top)
    rows = [{"n": f"top{n}", "p": hp(top[:n])} for n in X.top_sizes(k)]
    rows += [{"n": f"top{k} reply", "p": hp(top, "reply")}, {"n": f"top{k} prompt", "p": hp(top, "prompt")},
             {"n": "seller8", "p": hp(seller8)}, {"n": "seller8 reply", "p": hp(seller8, "reply")},
             {"n": "seller8 prompt", "p": hp(seller8, "prompt")},
             {"n": "band attn", "p": [["attn", L, -1, "post"] for L in G["band"]]},
             {"n": "other heads", "p": hp([(L, h) for L in G["band"] for h in range(G["n_heads"]) if (L, h) not in set(top)])},
             {"n": f"random{len(rand)}", "p": hp(rand)}]
    shared = [x for x in seller8 if x in set(top)]
    if shared:
        rows += [{"n": f"shared{len(shared)}", "p": hp(shared)}]
        if len(shared) < len(seller8):
            rows += [{"n": f"seller-only{len(seller8) - len(shared)}", "p": hp([x for x in seller8 if x not in shared])}]
    return rows


def _batches(n_rows, batch):
    return int(math.ceil((n_rows + 1) / batch))


def estimate_min(n_pairs, n_path_items, n_path_rows, n_passa, n_buy_disc, n_buy_held, n_bsweep, n_bcirc, batch):
    t = 2 * n_pairs * BUILD_S + n_pairs * SOURCES_S \
        + n_path_items * (2 * SRC_S + (math.ceil(n_passa / batch) + 1) * PASSA_S + _batches(n_path_rows, batch) * BATCH_S) \
        + (n_buy_disc + n_buy_held) * (_batches(n_bsweep, batch) * BATCH_S + SRC_S) + n_buy_held * (_batches(n_bcirc, batch) * BATCH_S + SRC_S)
    return t / 60, t * SLOW / 60


# --- setup ---------------------------------------------------------------------------------------------------
def build_items(li, settings, n_fix, role):
    e2 = li.build_e2_items(settings, n_fix, role=role)
    tag = role[0].upper()
    pairs = [{"item_id": f"{it['fixture']}-{tag}-pair", "fixture": it["fixture"], "role": role,
              "base_limit": it["base_limit"], "x1": it["source"], "x1e6": it["target"]}
             for it in e2 if it["direction"] == "x1->x1e6"]
    return pairs, e2


def setup(a, settings, schema, li):
    _LI[:] = [li]
    X._LI[:] = [li]
    parts = {p.strip() for p in a.heads2_parts.split(",") if p.strip()}
    opts = {"recv": parse_heads(a.heads2_recv), "contrast": parse_heads(a.heads2_contrast),
            "back": [int(x) for x in a.heads2_back.split(",") if x.strip()], "after": a.heads2_after,
            "top": a.heads2_top, "discovery": a.heads2_discovery, "parts": sorted(parts), "x57_run": a.heads2_x57_run}
    sp, se2 = build_items(li, settings, a.fixtures, "seller")
    bp, be2 = build_items(li, settings, a.fixtures, "buyer")
    fix = [p["fixture"] for p in bp]
    n_disc = min(opts["discovery"], max(1, len(fix) // 3))
    disc, held = fix[:n_disc], fix[n_disc:]
    G = geometry(CANON, 32, opts)
    k = min(opts["top"], len(G["band"]) * 32 // 2)
    prow = path_rows(G, 117, 106, opts["after"], {"Y01": {"attn": [], "diff": [], "rand_a": [], "rand_b": []}})
    n_passa = sum(1 for r in prow if r["kind"] == "path")
    bsweep = buyer_sweep_rows(G, G["recv"])
    bcirc = buyer_circuit_rows(G["recv"], G["recv"], G["recv"], G)
    on = lambda p, xs: list(xs) if p in parts else []                          # noqa: E731
    ctx = {"li": li, "opts": opts, "G": G, "k_top": k, "pick": None, "src_pick": None,
           "sell_pairs": list(sp), "src_pairs": on("sources", sp), "path_items": on("path", se2),
           "buy_pairs": on("buyers", bp), "buy_disc": on("buyers", [it for it in be2 if it["fixture"] in disc]),
           "buy_held": on("buyers", [it for it in be2 if it["fixture"] in held]), "disc": disc, "held": held,
           "sbuild_kw": {"role": "seller", "steer_layer": X.map_layer(20, CANON)},
           "src_kw": {"heads": [list(x) for x in G["sources"]], "scan": G["band"]},
           "path_kw": {"role": "seller", "recv": [list(x) for x in G["recv"]], "rows": [], "batch": a.batch},
           "bbuild_kw": {"role": "buyer", "steer_layer": X.map_layer(20, CANON)},
           "bsweep_kw": {"role": "buyer", "rows": bsweep, "batch": a.batch},
           "bcirc_kw": {"role": "buyer", "rows": [], "batch": a.batch}}
    if not ctx["src_pairs"] and not ctx["path_items"]:
        ctx["sell_pairs"] = []
    typ, slow = estimate_min(len(sp), len(ctx["path_items"]), len(prow), n_passa, len(ctx["buy_disc"]), len(ctx["buy_held"]),
                             len(bsweep), len(bcirc), a.batch)
    if "sources" not in parts:
        typ -= len(sp) * SOURCES_S / 60
    ctx["estimate_min"] = [round(typ, 1), round(slow, 1)]
    li.STEP_GUESS_S.update({"heads2 seller build": BUILD_S, "heads2 sources": SOURCES_S, "heads2 buyer build": BUILD_S,
                            "heads2 path": 2 * SRC_S + (math.ceil(n_passa / a.batch) + 1) * PASSA_S + _batches(len(prow), a.batch) * BATCH_S,
                            "heads2 buyer sweep 1": _batches(len(bsweep), a.batch) * BATCH_S + SRC_S,
                            "heads2 buyer sweep 2": _batches(len(bsweep), a.batch) * BATCH_S + SRC_S,
                            "heads2 buyer circuit": _batches(len(bcirc), a.batch) * BATCH_S + SRC_S})
    plan = (f"Heads 2 on {a.model}: X57's ld readout in E2's forced-opening setup. Sellers ({len(sp)} fixtures): build; "
            f"sources of {len(G['sources'])} heads (attention from the reply's 11 opening tokens over every position, "
            f"value-weighted, and the x1e6 - x1 output difference split by position; difference size for all heads at "
            f"38-44); path patching into the top {len(G['recv'])} heads ({len(prow)} rows per item, {len(ctx['path_items'])} "
            f"items; feeding layers {G['feed']}, earlier layers {G['back']}). Buyers: build; sweep of every head at 38-44 + "
            f"the sellers' top 8 ({len(bsweep)} rows) on {len(disc)} discovery fixtures; circuit (top {k} buyer heads, "
            f"seller 8, controls) and the sweep on {len(held)} held-out; batch {a.batch}; about {typ:.0f} min of experiment "
            f"time ({slow:.0f} if forwards run 1.5x slower; --budget-min {a.budget_min:g} stops new items); parts: "
            f"{', '.join(sorted(parts))}")
    inputs = [{"step": "pairs", **p} for p in sp + bp] + [{"step": "patch", "item_id": it["item_id"]} for it in se2 + be2]
    return {"exp": "Heads2", "plan": plan, "inputs": inputs, "ctx": ctx, "steps": steps, "live": live,
            "after_step": after_step}


def steps(ctx):
    return [("heads2 seller build", "heads2_build", ctx["sell_pairs"], ctx["sbuild_kw"], 5),
            ("heads2 sources", "heads2_sources", ctx["src_pairs"], ctx["src_kw"], 3),
            ("heads2 path", "heads2_path", ctx["path_items"], ctx["path_kw"], 2),
            ("heads2 buyer build", "heads2_build", ctx["buy_pairs"], ctx["bbuild_kw"], 5),
            ("heads2 buyer sweep 1", "heads2_patch", ctx["buy_disc"], ctx["bsweep_kw"], 2),
            ("heads2 buyer circuit", "heads2_patch", ctx["buy_held"], ctx["bcirc_kw"], 2),
            ("heads2 buyer sweep 2", "heads2_patch", ctx["buy_held"], ctx["bsweep_kw"], 2)]


def _regeom(ctx, nL, nH):
    if (nL, nH) != (ctx["G"]["n_layers"], ctx["G"]["n_heads"]):
        ctx["G"] = geometry(nL, nH, ctx["opts"])
    G = ctx["G"]
    ctx["src_kw"].update(heads=[list(x) for x in G["sources"]], scan=G["band"])
    ctx["path_kw"]["recv"] = [list(x) for x in G["recv"]]
    ctx["sbuild_kw"]["steer_layer"] = ctx["bbuild_kw"]["steer_layer"] = X.map_layer(20, nL)
    ctx["bsweep_kw"]["rows"][:] = buyer_sweep_rows(G, G["recv"])
    ctx["k_top"] = min(ctx["opts"]["top"], len(G["band"]) * nH // 2)


def on_pod_info(ctx, pod_info):
    n = pod_info.get("num_hidden_layers")
    if n and int(n) != ctx["G"]["n_layers"]:
        _regeom(ctx, int(n), ctx["G"]["n_heads"])


def _note(ctx, msg):
    if ctx.get("panel") is not None:
        ctx["panel"].note(msg)


def pick_positions(recs, recv, n_post, n_prompt, n_after, k=5, seed="scale-invariance/interp/heads2/random/v1"):
    """Per fixture: the k post-limit positions the receivers attend to most (mean of both runs), the k that carry the
    most of their x1e6 - x1 output difference, and two random draws matched by region to the latter."""
    R = regions(n_post, n_prompt, n_after)
    reg_of = {o: name for name in ("after", "rest", "reply") for o in R[name]}
    keys = [f"{L}.{h}" for L, h in recv]
    out = {}
    for r in recs:
        if r.get("error") or not r.get("heads"):
            continue
        hs = [r["heads"][x] for x in keys if x in r["heads"]]
        if not hs:
            continue
        att = [sum((h["x1"]["mass_post"][o] + h["x1e6"]["mass_post"][o]) / 2 for h in hs) for o in range(n_post)]
        dif = [sum(h["diff"]["post"][o] * h["diff"]["den"] for h in hs) for o in range(n_post)]
        top_a = sorted(range(n_post), key=lambda o: (-att[o], o))[:k]
        top_d = sorted(range(n_post), key=lambda o: (-dif[o], o))[:k]
        rng = random.Random(f"{seed}/{r['fixture']}")
        draws = []
        for _ in range(2):
            used, pick = set(top_a) | set(top_d), []
            for o in top_d:
                pool = [x for x in R[reg_of[o]] if x not in used] or [x for x in range(n_post) if x not in used]
                c = rng.choice(pool)
                used.add(c)
                pick.append(c)
            draws.append(sorted(pick))
        out[r["fixture"]] = {"attn": sorted(top_a), "diff": sorted(top_d), "rand_a": draws[0], "rand_b": draws[1]}
    return out


def after_step(label, got, ctx):
    G = ctx["G"]
    if label == "heads2 seller build":
        recs = got.get(label) or []
        use = [r for r in recs if not r.get("skipped")]
        if not recs:
            return None
        if len(use) < 3:
            ctx["src_pairs"][:], ctx["path_items"][:] = [], []
            _note(ctx, f"seller build: only {len(use)} usable fixtures; sources and path skipped")
            return None
        _regeom(ctx, use[0]["n_layers"], use[0]["n_heads"])
        ctx["geom_post"] = (use[0]["n_post"], use[0]["n_prompt"])
        ctx["path_kw"]["rows"][:] = path_rows(ctx["G"], use[0]["n_post"], use[0]["n_prompt"], ctx["opts"]["after"])
        _note(ctx, f"seller build: {len(use)}/{len(recs)} usable; receivers " + ", ".join(f"L{L}.{h}" for L, h in ctx["G"]["recv"]))
        return None
    if label == "heads2 sources":
        recs = got.get(label) or []
        n_post, n_prompt = ctx.get("geom_post") or (117, 106)
        picks = pick_positions(recs, G["recv"], n_post, n_prompt, ctx["opts"]["after"])
        ctx["src_pick"] = picks
        ctx["path_kw"]["rows"][:] = path_rows(G, n_post, n_prompt, ctx["opts"]["after"], picks or None)
        err = next((r["error"] for r in recs if r.get("error")), None)
        _note(ctx, f"sources: {len(picks)} fixtures with position picks" + (f" ({err})" if err else ""))
        return None
    if label == "heads2 buyer build":
        recs = got.get(label) or []
        use = [r for r in recs if not r.get("skipped")]
        if recs and len(use) < 3:
            ctx["buy_disc"][:], ctx["buy_held"][:] = [], []
            _note(ctx, f"buyer build: only {len(use)} usable fixtures; buyer sweeps skipped")
            return None
        if use:
            _regeom(ctx, use[0]["n_layers"], use[0]["n_heads"])
            sk = sorted(r["fixture"] for r in recs if r.get("skipped"))
            _note(ctx, f"buyer build: {len(use)}/{len(recs)} usable" + (f"; skipped {', '.join(sk)}" if sk else ""))
        return None
    if label == "heads2 buyer sweep 1":
        S = X.shares(got.get(label) or [], ci=False)
        top, rand, info = X.pick_heads(S, G["band"], G["n_heads"], ctx["k_top"], seed="scale-invariance/interp/heads2/buyer-random/v1")
        if not top:
            ctx["buy_held"][:] = []
            _note(ctx, "buyer sweep 1: no head read in both directions; circuit and sweep 2 skipped")
            return None
        ctx["pick"] = dict(info, top=[list(x) for x in top], random=[list(x) for x in rand])
        ctx["bcirc_kw"]["rows"][:] = buyer_circuit_rows(top, rand, G["recv"], G)
        _note(ctx, "buyer top heads: " + ", ".join(f"L{L}.{h}" for L, h in top))
    return None


# --- analysis ------------------------------------------------------------------------------------------------
def _m(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else None


def region_sums(per, n_prompt, n_after, kind="mass"):
    """{region: value} for one head and run from heads2_sources' compact record."""
    pre, dig, post = per[f"{kind}_pre"], per[f"{kind}_digits"], per[f"{kind}_post"]
    return {"bos": pre["bos"], "before": pre["sum"] - pre["bos"], "lead-in": sum(pre["last"]), "digits": sum(dig),
            "after": sum(post[:n_after]), "rest": sum(post[n_after:n_prompt]), "reply": sum(post[n_prompt:])}


def diff_sums(d, n_prompt, n_after):
    return {"bos": d["pre"]["bos"], "before": d["pre"]["sum"] - d["pre"]["bos"], "lead-in": sum(d["pre"]["last"]),
            "digits": d["digits"], "after": sum(d["post"][:n_after]), "rest": sum(d["post"][n_after:n_prompt]),
            "reply": sum(d["post"][n_prompt:])}


def _x57_map(run_dir, li=None):
    """X57's seller head scores {"h41.29": mean share of both directions} from its summary.json, or None."""
    root = getattr(li, "ROOT", None) or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = run_dir if os.path.isabs(run_dir) else os.path.join(root, run_dir)
    try:
        S = json.load(open(os.path.join(p, "summary.json")))["sweep_all"]
    except (OSError, ValueError, KeyError):
        return None
    names = {n for d in DIRS for n in S.get(d, {}) if n.startswith("h")}
    return {n: X.head_score(S, n) for n in names if X.head_score(S, n) is not None}


def sources_summary(recs, n_after, x57=None):
    ok = [r for r in recs if not r.get("error") and r.get("heads")]
    if not ok:
        return {"error": next((r["error"] for r in recs if r.get("error")), "no records")}
    n_post, n_prompt = ok[0]["n_post"], ok[0]["n_prompt"]
    toks = ok[0]["post_tokens"]
    reg_of = lambda o: "after" if o < n_after else ("rest" if o < n_prompt else "reply")    # noqa: E731
    heads = {}
    for key in ok[0]["heads"]:
        rs = [r["heads"][key] for r in ok if key in r["heads"]]
        cell = {"n": len(rs)}
        for kind in ("mass", "vw"):
            for c in ("x1", "x1e6"):
                sums = [region_sums(h[c], n_prompt, n_after, kind) for h in rs]
                cell[f"{kind}_{c}"] = {g: _m(s[g] for s in sums) for g in sums[0]}
            avg = [_m((h["x1"][f"{kind}_post"][o] + h["x1e6"][f"{kind}_post"][o]) / 2 for h in rs) for o in range(n_post)]
            cell[f"top_{kind}"] = [[o, toks[o], reg_of(o), round(avg[o], 4)] for o in sorted(range(n_post), key=lambda o: -avg[o])[:6]]
        ds = [diff_sums(h["diff"], n_prompt, n_after) for h in rs]
        dens = [h["diff"]["den"] for h in rs]
        cell["diff"] = {}
        for g in ds[0]:
            num = [s[g] * d for s, d in zip(ds, dens)]
            cell["diff"][g] = {"share": sum(num) / sum(dens) if sum(dens) else None, "ci": X.ratio_ci(num, dens, 400)}
        pos = [sum(h["diff"]["post"][o] * h["diff"]["den"] for h in rs) / sum(dens) for o in range(n_post)]
        cell["top_diff"] = [[o, toks[o], reg_of(o), round(pos[o], 4)] for o in sorted(range(n_post), key=lambda o: -pos[o])[:6]]
        cell["q_digits"] = {c: [_m(h[c]["q_regions"][q][2] for h in rs) for q in range(len(rs[0][c]["q_regions"]))]
                            for c in ("x1", "x1e6")}
        cell["dnorm"] = _m(h["diff"]["dnorm"] for h in rs)
        cell["out_norm"] = {c: _m(h[c]["out_norm"] for h in rs) for c in ("x1", "x1e6")}
        cell["x57_score"] = (x57 or {}).get(f"h{key}")
        heads[key] = cell
    checks = {"recon_err_max": max(h[c]["recon_err"] for r in ok for h in r["heads"].values() for c in ("x1", "x1e6")),
              "sum_err_max": max((h["diff"]["sum_err"] or 0) for r in ok for h in r["heads"].values())}
    scan = {}
    for Ls in ok[0].get("scan") or {}:
        nH = len(ok[0]["scan"][Ls]["d_reply"])
        for h in range(nH):
            scan[f"h{Ls}.{h}"] = {k: _m(r["scan"][Ls][k][h] for r in ok if Ls in r.get("scan", {}))
                                  for k in ("d_reply", "d_prompt", "n_x1", "n_x1e6")}
    rank = {}
    if x57 and scan:
        names = [n for n in scan if n in x57]
        y = [x57[n] for n in names]
        for k in ("d_reply", "d_prompt"):
            rank[k] = X.spearman([scan[n][k] for n in names], y)
        rank["d_reply_relative"] = X.spearman([scan[n]["d_reply"] / ((scan[n]["n_x1"] + scan[n]["n_x1e6"]) / 2 or 1)
                                               for n in names], y)
        rank["n_heads"] = len(names)
    return {"n_fixtures": len(ok), "n_post": n_post, "n_prompt": n_prompt, "n_after": n_after, "post_tokens": toks,
            "heads": heads, "checks": checks, "scan_vs_x57_spearman": rank,
            "scan_top": sorted(scan, key=lambda n: -(scan[n]["d_reply"] or 0))[:12], "scan": scan}


def cand_prob(recs):
    """{direction: {row: mean p(a) + p(b) at the decision token}} (is the readout still about these two prices?)."""
    out = {}
    for r in recs:
        if r.get("skipped") or not r.get("rows"):
            continue
        for x in r["rows"]:
            if "lpa" in x:
                out.setdefault(r["direction"], {}).setdefault(x["n"], []).append(math.exp(x["lpa"]) + math.exp(x["lpb"]))
    return {d: {n: st.mean(v) for n, v in rows.items()} for d, rows in out.items()}


def of_direct(recs, n_boot=1000):
    """{direction: {row: share of the 'direct' row's movement, with CI}}."""
    out = {}
    for d, per in X._per_fixture(recs).items():
        names = sorted({n for _c, _s, rows in per.values() for n in rows})
        for name in names:
            num, den = [], []
            for c, _s, rows in per.values():
                if name in rows and "direct" in rows:
                    num.append(rows[name] - c)
                    den.append(rows["direct"] - c)
            if num and st.mean(den):
                out.setdefault(d, {})[name] = {"share": st.mean(num) / st.mean(den), "ci": X.ratio_ci(num, den, n_boot)}
    return out


def _scores(S, G):
    return {f"h{L}.{h}": X.head_score(S, f"h{L}.{h}") for L in G["band"] for h in range(G["n_heads"])}


def buyers_summary(s1, s2, circ, G, pick, x57):
    out = {"sweep_discovery": X.shares(s1), "sweep_held": X.shares(s2, ci=False), "sweep_all": X.shares(s1 + s2),
           "circuit": X.shares(circ, n_boot=1000), "cand": cand_prob(circ), "cand_sweep": None, "pick": pick}
    cs = cand_prob(s1 + s2)
    out["cand_sweep"] = {d: {n: v for n, v in rows.items() if not n.startswith("h")} for d, rows in cs.items()}
    a, b = _scores(out["sweep_discovery"], G), _scores(out["sweep_held"], G)
    names = [n for n in a if a[n] is not None and b.get(n) is not None]
    out["replication_spearman"] = X.spearman([a[n] for n in names], [b[n] for n in names]) if names else None
    if names:
        tb = sorted(names, key=lambda n: -b[n])[:len(pick["top"]) if pick else 8]
        out["held_top"] = tb
        out["held_top_overlap"] = len(set(tb) & {f"h{L}.{h}" for L, h in (pick or {}).get("top", [])})
    allb = _scores(out["sweep_all"], G)
    if x57:
        nm = [n for n in allb if allb[n] is not None and n in x57]
        out["vs_seller_spearman"] = X.spearman([allb[n] for n in nm], [x57[n] for n in nm]) if nm else None
    seller8 = {f"h{L}.{h}" for L, h in G["recv"]}
    ranked = sorted((n for n in allb if allb[n] is not None), key=lambda n: -allb[n])
    out["seller8_ranks_in_buyers"] = {n: ranked.index(n) + 1 for n in seller8 if n in ranked}
    out["top_all"] = [[n, round(allb[n], 4)] for n in ranked[:12]]
    return out


def summarize(got, ctx):
    G, o = ctx["G"], ctx["opts"]
    li = _li()
    x57 = _x57_map(o["x57_run"], li)
    sb, bb = got.get("heads2 seller build") or [], got.get("heads2 buyer build") or []
    path = got.get("heads2 path") or []
    s1, s2, circ = (got.get(k) or [] for k in ("heads2 buyer sweep 1", "heads2 buyer sweep 2", "heads2 buyer circuit"))
    pk = ("fixture", "base_limit", "price_x1", "price_x1e6", "stand_in", "skipped", "ld", "decision")
    out = {"geometry": G, "opts": o, "x57_map_found": x57 is not None, "discovery": ctx["disc"], "held_out": ctx["held"],
           "build": {"seller": [{k: r.get(k) for k in pk} for r in sb], "buyer": [{k: r.get(k) for k in pk} for r in bb]},
           "sources": sources_summary(got.get("heads2 sources") or [], o["after"], x57),
           "src_pick": ctx.get("src_pick"),
           "path": {"shares": X.shares(path, n_boot=1000), "of_direct": of_direct(path), "cand": cand_prob(path),
                    "order": [r["n"] for r in ctx["path_kw"]["rows"]]},
           "buyers": buyers_summary(s1, s2, circ, G, ctx.get("pick"), x57),
           "checks": {"hooks_ok": all(r.get("hooks_ok", True) for r in path + s1 + s2 + circ),
                      "clean_spread_max": max((r.get("clean_spread") or 0 for r in path + s1 + s2 + circ), default=None),
                      "clean_vs_unbatched_mean": _m(abs(r["ld_clean"] - r["ld_clean_unbatched"]) for r in path
                                                    if r.get("ld_clean_unbatched") is not None),
                      "skipped": {"seller": sorted(r["fixture"] for r in sb if r.get("skipped")),
                                  "buyer": sorted(r["fixture"] for r in bb if r.get("skipped"))},
                      "stand_in": any(r.get("stand_in") for r in sb + bb),
                      "errors": {lab: [sum(1 for r in got.get(lab) or [] if r.get("error")),
                                       next((r["error"] for r in got.get(lab) or [] if r.get("error")), None)]
                                 for lab, *_ in steps(ctx) if any(r.get("error") for r in got.get(lab) or [])}},
           "secs_per_item": {lab: _m(r.get("secs") for r in got.get(lab) or []) for lab, *_ in steps(ctx)},
           "estimate_min": ctx.get("estimate_min")}
    return out


def manifest(ctx):
    return {"name": "heads2", "plan": "config/interp/plan_next.md, session 2 (Follow the key heads)", "geometry": ctx["G"],
            "opts": ctx["opts"], "discovery": ctx["disc"], "held_out": ctx["held"], "estimate_min": ctx.get("estimate_min"),
            "path_rows": len(ctx["path_kw"]["rows"]), "buyer_sweep_rows": len(ctx["bsweep_kw"]["rows"])}


# --- panel, preview, report ----------------------------------------------------------------------------------
_pct, _ci = X._pct, X._ci


def live(got):
    out = []
    for lab in ("heads2 seller build", "heads2 buyer build"):
        b = got.get(lab) or []
        if b:
            out.append(f"{lab.split()[1]} build: {sum(1 for r in b if not r.get('skipped'))}/{len(b)} fixtures usable")
    src = [r for r in got.get("heads2 sources") or [] if r.get("heads")]
    if src:
        key = next(iter(src[0]["heads"]))
        h = src[0]["heads"][key]
        out.append(f"sources ({len(src)} fixtures): L{key} reply -> digits {sum(h['x1']['mass_digits']):.2f} (x1), "
                   f"{sum(h['x1e6']['mass_digits']):.2f} (x1e6); difference from the digits {h['diff']['digits']:+.2f}")
    p = got.get("heads2 path") or []
    if p:
        S = X.shares(p, ci=False)
        for d in DIRS:
            out.append(f"path {d} ({len({r['fixture'] for r in p})} fixtures): " + " · ".join(
                f"{n} {_pct(S.get(d, {}).get(n, {}).get('share')).strip()}" for n in ("direct", "T feed post", "S feed post")))
    for lab in ("heads2 buyer sweep 1", "heads2 buyer sweep 2"):
        r = got.get(lab) or []
        if r:
            S = X.shares(r, ci=False)
            sc = sorted(((n, X.head_score(S, n)) for n in S.get(DIRS[0], {}) if n.startswith("h")),
                        key=lambda x: -(x[1] if x[1] is not None else -9))[:4]
            out.append(f"{lab[7:]}: seller8 {_pct(X.head_score(S, 'seller8')).strip()}; top " +
                       ", ".join(f"{n} {_pct(v).strip()}" for n, v in sc if v is not None))
    c = got.get("heads2 buyer circuit") or []
    if c:
        S = X.shares(c, ci=False)
        out.append("buyer circuit: " + " · ".join(f"{n} {_pct(X.head_score(S, n)).strip()}" for n in ("top8", "seller8", "random8")))
    return out


def stub(task, items, ctx):
    """Preview only: canned shapes, not a model."""
    G = ctx["G"]
    nL, nH = G["n_layers"], G["n_heads"]
    sig = lambda x: 1 / (1 + math.exp(-x))                                  # noqa: E731
    if task == "heads2_build":
        return [{"item_id": it["item_id"], "fixture": it["fixture"], "role": it["role"], "base_limit": it["base_limit"],
                 "n_layers": nL, "n_heads": nH, "head_dim": [256] * nL, "steer_layer": 20, "price_x1": round(1.3 * it["base_limit"]),
                 "price_x1e6": round(1.2 * it["base_limit"]) * 10 ** 6, "stand_in": False, "ld": {"x1": -4.0, "x1e6": 4.0},
                 "n_post": 117, "n_prompt": 106, "decision": {}, "secs": 0.1} for it in items]
    if task == "heads2_sources":
        out = []
        for it in items:
            heads = {}
            for i, (L, h) in enumerate(ctx["src_kw"]["heads"]):
                post = [0.0] * 117
                post[1], post[104], post[115] = 0.05 + 0.01 * i, 0.08, 0.10
                d = [0.0] * 117
                d[1], d[115] = 0.3, 0.2
                per = {c: {"mass_pre": {"bos": 0.5, "sum": 0.6, "last": [0.01] * 8, "top": [[0, 0.5]]},
                           "mass_digits": [0.1 * (1 + i % 3)] * (2 if c == "x1" else 8),
                           "mass_post": post, "vw_pre": {"bos": 0.1, "sum": 0.3, "last": [0.02] * 8, "top": [[0, 0.1]]},
                           "vw_digits": [0.1] * (2 if c == "x1" else 8), "vw_post": post,
                           "q_regions": [[0.5, 0.1, 0.2, 0.1, 0.1]] * 11, "out_norm": 1.0, "recon_err": 1e-7} for c in ("x1", "x1e6")}
                per["diff"] = {"den": 2.0, "pre": {"bos": 0.02, "sum": 0.1, "last": [0.0] * 8, "top": [[0, 0.02]]},
                               "digits": 0.4 - 0.03 * i, "post": d, "sum_err": 1e-7, "dnorm": 1.4}
                heads[f"{L}.{h}"] = per
            scan = {str(L): {k: [round(0.1 + (0.5 if (L, h) in set(map(tuple, G["recv"])) else 0) + 0.01 * (h % 5), 4)
                                 for h in range(nH)] for k in ("d_reply", "d_prompt", "n_x1", "n_x1e6")} for L in ctx["src_kw"]["scan"]}
            out.append({"item_id": it["item_id"], "fixture": it["fixture"], "role": "seller", "s0": 459,
                        "n_digits": {"x1": 2, "x1e6": 8}, "n_post": 117, "n_prompt": 106,
                        "post_tokens": [f"t{o}" for o in range(117)], "heads": heads, "scan": scan, "secs": 0.1})
        return out
    hot = {tuple(x): 0.05 for x in G["recv"]}
    bhot = {(41, 3): 0.12, (42, 17): 0.08, (41, 29): 0.05, (40, 1): 0.05}

    def share(spec, d, role):
        late = d == "x1->x1e6"
        if spec.get("kind") == "path":
            base = {"post": 0.6, "prompt": 0.2, "after": 0.1, "rest": 0.1, "reply": 0.4}
            nm = spec["n"].split()
            s = base.get(nm[-1], 0.15 if nm[-1] in ("attn", "diff") else 0.02)
            if nm[0] == "S":
                s = 0.4 if nm[-1] == "post" else (1.0 - s if nm[-1] != "none" else 1.0)
            if nm[1].startswith("back"):
                s *= sig((int(nm[1][4:]) - (36 if late else 24)) / 3)
            return s * (0.4 if late else 0.68)
        if spec.get("kind") == "direct":
            return (0.4 if late else 0.68) * (1.0 if not spec.get("use") else 0.5)
        if spec.get("kind") == "resid":
            return sig((spec["L"] - 40.5) / 1.2) if late else sig((spec["L"] - 22) / 3)
        s = 0.0
        for kind, L, h, sc in spec.get("p", ()):
            f = 0.3 if sc == "prompt" else (0.8 if sc == "reply" else 1.0)
            if kind == "resid":
                s += sig((L - 41) / 1.2)
            elif kind == "attn":
                s += 0.5 * math.exp(-((L - 41.5) / 1.5) ** 2)
            elif kind == "mlp":
                s += 0.02
            else:
                s += f * (bhot if role == "buyer" else hot).get((L, h), 0.003)
        return min(s, 1.1)
    out = []
    seen = ctx.setdefault("_stub_seen", {})
    for it in items:
        k = seen.get((task, it["item_id"]), 0)
        seen[(task, it["item_id"])] = k + 1
        if task == "heads2_path":
            specs = ctx["path_kw"]["rows"]
        elif it["fixture"] in ctx["held"] and k == 0:
            specs = ctx["bcirc_kw"]["rows"]
        else:
            specs = ctx["bsweep_kw"]["rows"]
        jit = (sum(map(ord, it["fixture"])) % 7 - 3) / 20
        c, src = (4.0 + jit, -4.0) if it["target"]["condition"] == "D8" else (-4.0 + jit, 4.0)
        rows = [{"n": sp["n"], "ld": round(c + share(sp, it["direction"], it["role"]) * (src - c), 4), "lpa": -0.7, "lpb": -1.5}
                for sp in specs]
        out.append({"item_id": it["item_id"], "fixture": it["fixture"], "role": it["role"], "direction": it["direction"],
                    "base_limit": it["base_limit"], "target_condition": it["target"]["condition"], "ld_source": src,
                    "ld_clean": c, "ld_clean_unbatched": c + 0.01, "clean_spread": 0.0, "hooks_ok": True,
                    "rows": rows + [{"n": "clean", "ld": c, "lpa": -0.7, "lpb": -1.5}], "secs": 0.1})
    return out


def _sg(v):
    return "  —  " if v is None else f"{v:+.2f}"


def _cell(S, d, n):
    c = S.get(d, {}).get(n) or {}
    return f"{_pct(c.get('share')).strip():>5}{_ci(c):<12}"


def report(s, preview):
    print("Heads 2: sources, path patching, buyers (exploratory" + (", PREVIEW: canned shapes" if preview else "") + ")")
    G, ch = s.get("geometry") or {}, s.get("checks") or {}
    if G.get("mapped"):
        print(f"  NOTE: a {G['n_layers']}-layer, {G['n_heads']}-head model; the 31B's heads and layers were mapped onto it")
    if ch.get("stand_in"):
        print("  NOTE: stand-in prices (REHEARSAL: random weights); every number below is meaningless")
    for role in ("seller", "buyer"):
        b = (s.get("build") or {}).get(role) or []
        if b:
            sk = ch.get("skipped", {}).get(role) or []
            print(f"  {role} build: {sum(1 for r in b if not r.get('skipped'))}/{len(b)} usable" + (f" (skipped {', '.join(sk)}: "
                  + "; ".join(sorted({r['skipped'] for r in b if r.get('skipped')})) + ")" if sk else ""))
    print(f"  checks: hooks fired as planned {ch.get('hooks_ok')}; clean rows agree within {ch.get('clean_spread_max')} nats; "
          f"batched vs unbatched clean ld {ch.get('clean_vs_unbatched_mean')}")
    for lab, (n, msg) in (ch.get("errors") or {}).items():
        print(f"  ERRORS in {lab}: {n} items failed on the pod; first: {msg}")
    Sx = s.get("sources") or {}
    if Sx.get("heads"):
        print(f"  SOURCES ({Sx['n_fixtures']} fixtures; queries = the reply's {Sx['n_post'] - Sx['n_prompt']} opening tokens; "
              f"recon err {Sx['checks']['recon_err_max']:.1e}, split sums err {Sx['checks']['sum_err_max']:.1e})")
        print("    attention mass by region, x1 | x1e6 (bos / before / digits / after / rest / reply); then the share of the "
              "head's x1e6 - x1 output difference from each region")
        for key, c in Sx["heads"].items():
            f = lambda d: " ".join(f"{d[g]:.2f}" for g in REGIONS)                    # noqa: E731
            sc = c.get("x57_score")
            print(f"    L{key:<6}{'' if sc is None else f' (X57 {100 * sc:+.0f}%)':<12} {f(c['mass_x1'])} | {f(c['mass_x1e6'])}"
                  f"   diff: " + " ".join(f"{g} {_pct(c['diff'][g]['share']).strip()}" for g in REGIONS))
            print(f"      value-weighted x1e6: {f(c['vw_x1e6'])};  top positions by diff: "
                  + ", ".join(f"{o}:{t!r} {100 * v:+.0f}%" for o, t, _r, v in c["top_diff"][:4])
                  + ";  by attention: " + ", ".join(f"{o}:{t!r} {v:.2f}" for o, t, _r, v in c["top_mass"][:3]))
        r = Sx.get("scan_vs_x57_spearman") or {}
        if r:
            print(f"    all {r.get('n_heads')} heads at 38-44: output-difference size vs X57 importance, Spearman: reply "
                  f"{_sg(r.get('d_reply'))}, prompt {_sg(r.get('d_prompt'))}, reply relative to output size {_sg(r.get('d_reply_relative'))}")
    elif Sx.get("error"):
        print(f"  sources: {Sx['error']}")
    P = (s.get("path") or {}).get("shares") or {}
    if P:
        OD, CP = s["path"]["of_direct"], s["path"]["cand"]
        print(f"  PATH PATCHING into {', '.join(f'L{L}.{h}' for L, h in G.get('recv', []))}; share of the gap (95% CI), "
              f"share of 'direct', p(two prices); {DIRS[0]} | {DIRS[1]}")
        names = [n for n in s["path"].get("order") or [] if any(n in P.get(d, {}) for d in DIRS)]
        for d in DIRS:
            names += [n for n in P.get(d, {}) if n not in names]
        for n in names:
            f = lambda d: f"{_cell(P, d, n)} {_pct((OD.get(d, {}).get(n) or {}).get('share')).strip():>5} {CP.get(d, {}).get(n, 0):.2f}"   # noqa: E731
            print(f"    {n:<20} {f(DIRS[0])} | {f(DIRS[1])}")
    B = s.get("buyers") or {}
    if B.get("sweep_all"):
        print("  BUYERS")
        S = B["sweep_all"]
        print(f"    sweep (all fixtures): seller8 jointly {_cell(S, DIRS[0], 'seller8')} | {_cell(S, DIRS[1], 'seller8')}")
        for kind, Ls in (("resid", G.get("bresid", [])), ("attn", G.get("band", []))):
            print(f"    {kind}: " + ", ".join(f"L{L} {_pct(S.get(DIRS[0], {}).get(f'{kind}@{L}', {}).get('share')).strip()}/"
                                            f"{_pct(S.get(DIRS[1], {}).get(f'{kind}@{L}', {}).get('share')).strip()}" for L in Ls))
        print("    top heads (all fixtures, mean of both directions): " + ", ".join(f"{n} {100 * v:+.0f}%" for n, v in B["top_all"][:10]))
        print(f"    the sellers' top 8 ranked among buyer heads: {B.get('seller8_ranks_in_buyers')}")
        print(f"    replication (discovery vs held-out) Spearman {_sg(B.get('replication_spearman'))}; held-out top overlap "
              f"{B.get('held_top_overlap')}; buyer vs seller (X57) head map Spearman {_sg(B.get('vs_seller_spearman'))}")
        if B.get("pick"):
            print("    picked on discovery: " + ", ".join(f"L{L}.{h}" for L, h in B["pick"]["top"]))
        C = B.get("circuit") or {}
        if C:
            print(f"    circuit, held-out fixtures; share (95% CI), p(two prices); {DIRS[0]} | {DIRS[1]}")
            for n in sorted({n for d in DIRS for n in C.get(d, {})}):
                print(f"      {n:<16} {_cell(C, DIRS[0], n)} {B['cand'].get(DIRS[0], {}).get(n, 0):.2f} | "
                      f"{_cell(C, DIRS[1], n)} {B['cand'].get(DIRS[1], {}).get(n, 0):.2f}")
    sec = s.get("secs_per_item") or {}
    print("  seconds per item: " + ", ".join(f"{k[7:]} {v:.1f}" for k, v in sec.items() if v)
          + (f" (estimated {s['estimate_min'][0]} min typical)" if s.get("estimate_min") else ""))
