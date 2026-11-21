"""interp_concepts2_laptop.py: the concepts follow-up (X63) for launch_interp --experiment concepts2. Laptop side only.

X63 found that v (the seller's x1e6 - x1 direction, layers 41-44) leans toward big-number / money concepts, yet those
concept directions, added one at a time, barely move the price. This session tests that directly (exploratory):

  Sufficiency. v split per token into its projection onto the six concepts' span (par) and the rest (perp; par + perp
  = v), added on the 117 aligned tokens at 41 / 42 / 44 on the forced opening, at their natural sizes and scaled to
  v's size (par_s, perp_s); real openings at 44 with v, par, perp, par_s.
  Necessity. Each run's own state at one layer with its projection onto the six-concept span removed (abl6), against
  removing a random 6-d span (ablr: removes almost nothing, a check that the code path is inert) and removing v's own
  per-token direction (ablv: the positive control). Measured as the share of the plain log-odds gap ld(N6) - ld(N1)
  that survives, with p(a)+p(b) to tell a lost signal from a broken readout.

Pod code: tasks_concepts.py (its _vectors provides par / perp / abl*; POD_PLUGINS below ships it). Fixtures, concept
pairs and the build are the concepts session's, so v, the concept directions and the prompts are the same as X63's.
"""
import math
import statistics as st

import interp_concepts_laptop as C

POD_PLUGINS = ("concepts",)          # this experiment's tasks live in tasks_concepts.py
LAYERS = (41, 42, 44)
STEER_LAYER = 44
VECS = ("vN", "par", "perp", "par_s", "perp_s")
ABLS = ("abl6", "ablv", "ablr")
STEER_VECS = ("vN", "par", "perp", "par_s")
PER_S = {"concepts build": 0.6, "decomp patch": 4.0, "decomp steer": 30.0}
_LI = C._LI


def add_args(ap):
    ap.add_argument("--concepts2-samples", type=int, default=6, help="concepts2: real openings per (vector, target, floor) (default 6)")


def configure(a):
    a.roles = "seller"


def layer_sets(n):
    out = {"patch": sorted({C.map_layer(L, n) for L in LAYERS}), "steer": C.map_layer(STEER_LAYER, n), "n_layers": n,
           "mapped": n != C.CANON}
    out["keep"] = sorted(set(out["patch"]) | {out["steer"]})
    return out


def patch_rows(Ls):
    out = {}
    for t, sg in (("N1", 1.0), ("N6", -1.0)):
        out[t] = [{"n": f"{v}@{L}", "ops": [["vec", L, "A", sg, v]]} for L in Ls["patch"] for v in VECS] + \
                 [{"n": f"{ab}@{L}", "ops": [["vec", L, "A", 1.0, f"{ab}_{t}"]]} for L in Ls["patch"] for ab in ABLS]
    return out


def _kw(ctx, n):
    Ls = layer_sets(n)
    a = ctx["a"]
    ctx["layers"] = Ls
    ctx["kw"] = {"build": {"keep": Ls["keep"]}, "patch": {"rows": patch_rows(Ls), "batch": a.batch},
                 "steer": {"plan": [[v, Ls["steer"], 1.0, int(a.concepts2_samples)] for v in STEER_VECS], "n_none": 8,
                           "batch": 16, "schema": ctx["schema"]}}


def setup(a, settings, schema, li):
    _LI[:] = [li]
    fx, cx = C.distinct_fixtures(li, settings, a.fixtures), C.concept_items(li)
    ctx = {"li": li, "a": a, "schema": schema, "fx": fx, "cx": cx, "model": a.model}
    _kw(ctx, C.CANON)
    n = {"concepts build": len(fx) * 7 + len(cx), "decomp patch": len(fx), "decomp steer": 2 * len(fx)}
    t = sum(PER_S[k] * n[k] for k in PER_S) / 60
    ctx["estimate_min"] = [round(t, 1), round(t * C.SLOW, 1)]
    li.STEP_GUESS_S.update(PER_S)
    plan = (f"Concepts follow-up on {a.model} (sellers): the same {len(fx)} floors and {len(cx)} concept pairs as X63. Forced "
            f"opening at layers {'/'.join(map(str, ctx['layers']['patch']))}: v split into its six-concept part and the rest "
            f"({', '.join(VECS)}), and each run's own state with the concepts' span / a random 6-d span / v's direction removed "
            f"({', '.join(ABLS)}); real openings at layer {STEER_LAYER} ({', '.join(STEER_VECS)}; {a.concepts2_samples} per vector, "
            f"target and floor); about {t:.0f} min of experiment time ({t * C.SLOW:.0f} if 1.5x slower)")
    inputs = [{"step": "fixture", **it} for it in fx] + [{"step": "concept", **c} for c in cx]
    return {"exp": "Concepts2", "plan": plan, "inputs": inputs, "ctx": ctx, "steps": steps, "live": live, "after_step": after_step}


def steps(ctx):
    fx, kw = ctx["fx"], ctx["kw"]
    return [("concepts build", "concepts_build", fx + ctx["cx"], kw["build"], 10),
            ("decomp patch", "concepts_patch", fx, kw["patch"], 2),
            ("decomp steer", "concepts_steer", C.steer_items(fx), kw["steer"], 2)]


def on_pod_info(ctx, pod_info):
    n = pod_info.get("num_hidden_layers")
    if n and int(n) != ctx["layers"]["n_layers"]:
        _kw(ctx, int(n))


def after_step(label, got, ctx):
    if label != "concepts build":
        return None
    recs = [r for r in got.get(label) or [] if r.get("kind") == "fixture"]
    use = [r for r in recs if not r.get("skipped")]
    if len(use) < min(12, len(recs)):
        return f"only {len(use)} of {len(recs)} floors have two distinct plain greedy prices"
    if ctx.get("panel") is not None:
        ctx["panel"].note(f"build: {len(use)}/{len(recs)} floors usable")
    return None


# --- analysis -----------------------------------------------------------------------------------------------------------------
def ablation_table(recs, X4):
    """{row: {kept, ci, n, sign_kept, pk}}: share of the clean gap ld(N6) - ld(N1) left when both runs get the same
    ablation (pooled over floors: sum of ablated gaps / sum of clean gaps); sign_kept: floors whose gap stays > 0."""
    acc = {}
    for r in recs:
        T = r.get("targets") or {}
        if "N1" not in T or "N6" not in T:
            continue
        g0 = T["N6"]["ld_clean"] - T["N1"]["ld_clean"]
        rows = {t: {x["n"]: x for x in T[t]["rows"]} for t in ("N1", "N6")}
        for n in rows["N1"]:
            if n.split("@")[0] not in ABLS or n not in rows["N6"]:
                continue
            a, b = rows["N1"][n], rows["N6"][n]
            c = acc.setdefault(n, {"num": [], "den": [], "pk": []})
            c["num"].append(b["ld"] - a["ld"])
            c["den"].append(g0)
            c["pk"] += [math.exp(x["lpa"]) + math.exp(x["lpb"]) for x in (a, b)]
    return {n: {"n": len(c["num"]), "kept": sum(c["num"]) / sum(c["den"]) if sum(c["den"]) else None,
                "ci": X4.ratio_ci(c["num"], c["den"]), "sign_kept": sum(1 for g in c["num"] if g > 0), "pk": st.mean(c["pk"])}
            for n, c in acc.items()}


def additivity(P):
    """{target: {layer: share(par) + share(perp) - share(vN)}}: near 0 if the effect is additive in this split."""
    out = {}
    for t, rows in P.items():
        for n in rows:
            v, L = n.split("@")
            if v == "vN" and f"par@{L}" in rows and f"perp@{L}" in rows and None not in (rows[n]["share"], rows[f"par@{L}"]["share"], rows[f"perp@{L}"]["share"]):
                out.setdefault(t, {})[L] = rows[f"par@{L}"]["share"] + rows[f"perp@{L}"]["share"] - rows[n]["share"]
    return out


def summarize(got, ctx):
    li = ctx.get("li") or C._li()
    X4 = C._x4()
    ref, ref_from = X4.e0_plain(li, ctx.get("model"))
    b = [r for r in got.get("concepts build") or [] if r.get("kind") == "fixture"]
    use = [r for r in b if not r.get("skipped")]
    recs = [r for r in got.get("decomp patch") or [] if not r.get("skipped")]
    P = X4.ratio_table([dict(r, targets={t: dict(T, rows=[x for x in T["rows"] if x["n"].split("@")[0] in VECS + ("clean",)])
                                          for t, T in r["targets"].items()}) for r in recs], "plain")
    out = {"layers": ctx["layers"], "build": {"n": len(b), "usable": len(use), "skipped": {r["fixture"]: r["skipped"] for r in b if r.get("skipped")},
                                              "stand_in": any(r.get("stand_in") for r in use),
                                              "gap_mean": st.mean(r["ld"]["N6"] - r["ld"]["N1"] for r in use) if use else None,
                                              "prices": {r["fixture"]: r.get("price") for r in b}},
           "patch": P, "patch_flips": C.flip_table(recs), "additivity": additivity(P), "ablation": ablation_table(recs, X4),
           "steer": X4.summarize_steer(got.get("decomp steer") or [], li, ref), "steer_ref": ref_from,
           "estimate_min": ctx.get("estimate_min")}
    out["checks"] = {"hooks_ok": all(T.get("hooks_ok", True) for r in recs for T in r["targets"].values()),
                     "clean_spread_max": max((T.get("clean_spread") or 0 for r in recs for T in r["targets"].values()), default=None),
                     "loo": sorted({str(r.get("loo")) for lab in ("decomp patch", "decomp steer") for r in got.get(lab) or []}),
                     "steer_aligned": [sum(1 for r in got.get("decomp steer") or [] for x in r["rows"] if x["vector"] != "none" and x.get("aligned")),
                                       sum(1 for r in got.get("decomp steer") or [] for x in r["rows"] if x["vector"] != "none")]}
    out["secs_per_item"] = {lab: st.mean(r["secs"] for r in got[lab] if "secs" in r) if got.get(lab) and any("secs" in r for r in got[lab])
                            else None for lab, *_ in steps(ctx)}
    return out


def manifest(ctx):
    return {"name": "concepts2", "plan": "config/interp/plan_next.md (X63 follow-up)", "layers": ctx["layers"],
            "fixtures": [it["fixture"] for it in ctx["fx"]], "concept_pairs": len(ctx["cx"]), "vectors": list(VECS),
            "ablations": list(ABLS), "steer": ctx["kw"]["steer"]["plan"], "estimate_min": ctx.get("estimate_min")}


# --- panel, preview, report ---------------------------------------------------------------------------------------------------
def live(got):
    p = [r for r in got.get("decomp patch") or [] if not r.get("skipped")]
    if not p:
        return []
    A = ablation_table(p, C._x4())
    return ["ablation, gap kept: " + " ".join(f"{n} {C._p(c['kept']).strip()}" for n, c in sorted(A.items()))]


def stub(task, items, ctx):
    """Preview only: canned shapes, not a model."""
    if task == "concepts_build":
        return C.stub(task, items, ctx)
    ld0 = {"N1": -12.0, "N6": 5.0}
    sh = {"vN": 0.9, "par": 0.15, "perp": 0.75, "par_s": 0.3, "perp_s": 0.85}
    kept = {"abl6": 0.85, "ablv": 0.3, "ablr": 1.0}
    if task == "concepts_patch":
        out = []
        for it in items:
            T_ = {}
            for t in ("N1", "N6"):
                o = ld0["N6" if t == "N1" else "N1"]
                rows = []
                for x in ctx["kw"]["patch"]["rows"][t]:
                    v = x["n"].split("@")[0]
                    ld = ld0[t] + sh[v] * (o - ld0[t]) if v in sh else ld0[t] * kept[v]
                    rows.append({"n": x["n"], "ld": round(ld, 4), "lpa": -0.3, "lpb": -2.0})
                T_[t] = {"ld_clean": ld0[t], "clean_spread": 0.0, "hooks_ok": True, "rows": rows + [{"n": "clean", "ld": ld0[t], "lpa": -0.3, "lpb": -2.0}]}
            out.append({"item_id": it["item_id"], "fixture": it["fixture"], "base_limit": it["base_limit"], "loo": True, "targets": T_, "secs": 0.1})
        return out
    if task == "concepts_steer":
        out = []
        for it in items:
            s = 0 if it["target"] == "N1" else 6
            base = (1.41 if s == 0 else 1.21) * it["base_limit"]
            rows = [{"vector": "none", "layer": None, "alpha": 0.0, "sample": i} for i in range(ctx["kw"]["steer"]["n_none"])]
            rows += [{"vector": v, "layer": L, "alpha": a_, "sample": j} for v, L, a_, k in ctx["kw"]["steer"]["plan"] for j in range(k)]
            for x in rows:
                mv = 0 if x["vector"] == "none" else -it["sign"] * sh[x["vector"]] * 0.2 * it["base_limit"]
                x.update(price=int(round((base + mv) / 5) * 5) * 10 ** s, errors=[], action="OFFER", aligned=True)
            out.append({"item_id": it["item_id"], "fixture": it["fixture"], "role": "seller", "target": it["target"], "sign": it["sign"],
                        "base_limit": it["base_limit"], "rehearsal": False, "rows": rows, "secs": 0.1})
        return out
    raise ValueError(task)


def report(s, preview):
    print("Concepts follow-up: is the six-concept part of v what moves the price? (exploratory"
          + (", PREVIEW: canned shapes" if preview else "") + ")")
    Ls = s.get("layers") or {}
    if Ls.get("mapped"):
        print(f"  NOTE: a {Ls['n_layers']}-layer model; the 31B's layers were mapped onto it")
    B = s.get("build") or {}
    print(f"  floors: {B.get('usable')}/{B.get('n')} usable" + (" (STAND-IN prices: rehearsal)" if B.get("stand_in") else "")
          + (f"; mean plain gap {B['gap_mean']:+.1f} nats" if B.get("gap_mean") is not None else ""))
    P = s.get("patch") or {}
    if P:
        print("  sufficiency: par = v's projection onto the six concepts' span, perp = the rest (par + perp = v); _s = scaled to v's size")
        C.print_patch(P, s.get("patch_flips") or {}, VECS, Ls.get("steer"))
        ad = s.get("additivity") or {}
        if ad:
            print("    additivity, share(par) + share(perp) - share(v): " + "; ".join(
                f"{t} " + " ".join(f"L{L} {100 * v:+.0f}" for L, v in sorted(d.items(), key=lambda x: int(x[0]))) for t, d in ad.items()))
    A = s.get("ablation") or {}
    if A:
        print("  necessity: the plain gap ld(N6) - ld(N1) left when both runs lose a part of their own state at one layer")
        print("    (abl6: the six concepts' span; ablv: v's own direction, positive control; ablr: a random 6-d span, inert check)")
        for n in sorted(A, key=lambda n: (int(n.split("@")[1]), ABLS.index(n.split("@")[0]))):
            c = A[n]
            print(f"    {n:<9} kept {C._p(c['kept']).strip():>6}{C._ci(c):<14} sign kept {c['sign_kept']}/{c['n']}  p(a)+p(b) {c['pk']:.2f}")
    Sx = s.get("steer") or {}
    if Sx:
        print(f"  real openings at layer {Ls.get('steer')} (points moved; share of the plain gap from {s.get('steer_ref') or 'no E0 run'}):")
        for t, cells in Sx.items():
            for k, c in sorted(cells.items()):
                extra = "" if k == "none" else f"  moved {c.get('moved_pts') or 0:+.1f} pts, share {C._p(c.get('share')).strip()}{C._ci(c)}"
                print(f"    {t} {k:<14} gap {C._p(c['mean_gap'], 1).strip():>7}  valid {c['valid']}/{c['n']}  off-scale {c['off_scale']}{extra}")
    ch = s.get("checks") or {}
    print(f"  checks: hooks fired {ch.get('hooks_ok')}; clean rows agree within {ch.get('clean_spread_max')} nats; steered openings "
          f"aligned {ch.get('steer_aligned')}; leave-one-out {ch.get('loo')}")
    sec = s.get("secs_per_item") or {}
    print("  seconds per item: " + ", ".join(f"{k} {v:.1f}" for k, v in sec.items() if v)
          + (f" (estimated {s['estimate_min'][0]} min typical)" if s.get("estimate_min") else ""))
