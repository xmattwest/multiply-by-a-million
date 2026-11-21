"""interp_concepts_laptop.py: the concepts plug-in (what the seller's x1e6 - x1 direction is made of) for
launch_interp --experiment concepts. The shim scripts/interp/exp_concepts.py imports from here, so this file is not packed
into the pod's create request. Pod side: scripts/interp/tasks_concepts.py. Exploratory; manuscript/PLAN.md D4.

    python3 scripts/launch_interp.py --preview --experiment concepts            (free: canned shapes)
    python3 scripts/launch_interp.py --model gemma-4-31b-it --experiment concepts --slot 8          (paid)

Gemma 4 31B, sellers, N1 / N6 = the digits probe's D2 "81 dollars" / D8 "81000000 dollars" prompts. The seller prompt
depends only on the floor, so one fixture per DISTINCT floor (17 of Y01-Y30); discovery = the first 8 distinct floors,
held-out = the other 9 (no held-out prompt repeats a discovery prompt; the X62 lesson).

Part 1, named concepts (the main tool). Six concepts, each a set of sentence pairs (a, b) that share a token suffix;
direction = mean over pairs of mean_b - mean_a on that suffix, at every layer:
  count     a large count, not money ("There are 81 / 81000000 stars ...")             (E3's non-money control)
  money     a large amount of money outside a negotiation ("sold for 81 / 81000000 dollars ...")
  million   the same digits with the word "million" ("81 dollars" / "81 million dollars")
  digits    a long digit string that is not a quantity ("order number 81 / 81000000 ...")
  stakes    high stakes in words, no numbers ("a small, everyday purchase" / "a huge, high-stakes deal")
  modesty   the behaviour itself ("open far above my floor" / "open just above my floor")
  geometry  cosine of the negotiation direction v (fixture-mean N6 - N1 on H / S / reply) with each concept, by layer;
            the share of |v|^2 inside the six concepts' span (and inside 8 random directions: chance)
  patch     forced opening: add each concept's direction (scaled to v's per-token size) on A at layers 30-48, toward
            the other scale; share of the plain gap moved (vN = v itself, leave-one-fixture-out; rand = control)
  steer     real sampled openings, the same vectors at layer 44
Part 2, SAEs (supporting evidence):
  Decode Research's layer-30 SAE (131k latents; trained on the BASE model) at a pinned revision, sha256-checked.
  quality   reconstruction on our prompts, and how much of the gap survives splicing the reconstruction in
            (gate: kept share in [0.8, 1.25]; the SAE steps below are skipped if it fails)
  attr      discovery floors: first-order attribution of the gap to latents (aligned tokens + the limit's digits)
  patch     held-out floors: move the top 1/4/16/64 latents toward the other scale; all latents; all + error;
            random latents (controls)
  profile   the top latents' activations on the concept sentences, and their unembedding top tokens
  Adam1010's TopK SAEs at layers 41 and 44 (side check only): the same quality step.
"""
import math
import os
import statistics as st

CANON = 60
CONCEPT_LAYERS = (30, 36, 40, 41, 42, 44, 48)
STEER_LAYER = 44
VECS = ("vN", "count", "money", "million", "digits", "stakes", "modesty", "rand")
CONCEPTS = ("count", "money", "million", "digits", "stakes", "modesty")
BASES = (76, 81, 88, 93)
HF = "https://huggingface.co/"
DECODE = {"name": "decode30", "kind": "jumprelu", "layer": 30, "repo": "decoderesearch/gemma-4-saes",
          "rev": "bb8e6b333def5cf2667f2cd4b9e67258dae7029b", "path": "gemma-4-31b/btk-mat-layer-30-k-100/sae_weights.safetensors",
          "size": 5638215072, "sha256": "578dd8f60e9c67fd450135c5a9e5a334aff291ad94a22f7b8e24468a3b6f451d",
          "trained_on": "google/gemma-4-31B (base), the Pile (uncopyrighted), 1.5B tokens; Matryoshka BatchTopK k=100"}
ADAM = tuple({"name": f"adam{L}", "kind": "topk", "k": 64, "layer": L, "repo": "datasets/Adam1010/gemma-4-31b-sae-features",
              "rev": "b690a06a87cff055dc997f60a71ce62b7fdd0164", "path": f"weights/sae_layer_{L}.pt", "size": size, "sha256": sha,
              "trained_on": "google/gemma-4-31b-it, ~5M tokens per layer; TopK-64, 43k latents"}
             for L, size, sha in ((41, 1849916448, "5b9cb99900819e34fff5f15882102357ccfbc079831fb98c92ec644f736f3ff3"),
                                  (44, 1849927068, "c9b448453a22fb0cfbd76064d6e40fbfcccd8584b1a13060ba15771416a336c6")))
GATE = {"kept_lo": 0.8, "kept_hi": 1.25}
TOPS = (1, 4, 16, 64)
# Timing model (31B pod, seconds per item): from X57/X61 (forced openings ~0.7 s per batch of 16; one sampled batch of
# 16 ~7.5 s; a teacher-forced price ~1.5 s). Downloads assume ~40 MB/s (worst case well inside the lifetime).
PER_S = {"concepts build": 0.6, "concepts stats": 10.0, "concepts patch": 6.0, "concepts steer": 15.0, "sae30 load": 150.0,
         "sae30 quality": 1.5, "sae30 attr": 5.0, "sae30 pick": 5.0, "sae30 patch": 2.5, "sae30 profile": 0.25,
         "adam load": 50.0, "adam41 quality": 1.5, "adam44 quality": 1.5}
SLOW = 1.5
_LI = []

COUNT_T = None          # E3's non-money templates, read from launch_interp at setup
MONEY_T = ("The painting sold at auction for {A} last week.", "Her monthly budget for groceries is {A} in total.",
           "The repair bill came to {A}, including parts.", "The charity raised {A} at its spring dinner.",
           "He paid {A} for the used car in cash.", "The fine for the violation was {A} per day.",
           "The company reported a profit of {A} this quarter.", "Tickets to the show cost {A} for each seat.",
           "The contract is worth {A} over two years.", "They spent {A} on the new roof last summer.")
DIGITS_T = ("Please call extension {A} if you need any help.", "Your order number is {A} and it ships today.",
            "The serial code {A} is printed on the back.", "Enter the access code {A} at the front gate.",
            "The package with tracking number {A} has arrived.", "Ticket {A} won the second prize in the draw.",
            "The file was saved as report {A} on the server.", "Patient {A} was seen by the doctor this morning.",
            "Badge {A} was issued to the new guard.", "Locker {A} is assigned to you for the season.")
STAKES_F = ("This is {X}, so think it through before you answer.", "We are negotiating {X} today, and both sides want a fair result.",
            "The buyer sees this as {X} for the company.", "Treat this as {X} and make your offer carefully.")
STAKES_P = (("a small, everyday purchase", "a huge, high-stakes deal"), ("a minor matter", "a major matter"),
            ("a cheap item", "an expensive asset"), ("a trivial sale", "a momentous sale"), ("a low-stakes trade", "a high-stakes trade"))
MODEST_F = ("As the seller, I will open {X} my floor and then wait for the reply.", "My first offer will be {X} my limit, and I will explain why.",
            "I plan to ask for a price {X} the minimum I would accept.", "The opening price I name will sit {X} what I need.")
MODEST_P = (("far above", "just above"), ("well above", "slightly above"), ("much higher than", "a little higher than"),
            ("a lot more than", "a bit more than"), ("way over", "barely over"))


def _li():
    if _LI:
        return _LI[0]
    import sys
    m = sys.modules.get("__main__")
    return m if hasattr(m, "boot_ci") else sys.modules.get("launch_interp")


def _x4():
    import interp_e4_laptop as X4
    if _li() is not None:
        X4._LI[:] = [_li()]
    return X4


def url(spec):
    return f"{HF}{spec['repo']}/resolve/{spec['rev']}/{spec['path']}"


# --- flags -----------------------------------------------------------------------------------------------------------------
def add_args(ap):
    ap.add_argument("--concepts-discovery", type=int, default=8, help="concepts: distinct floors for the SAE latent pick (default 8)")
    ap.add_argument("--concepts-steer-samples", type=int, default=3, help="concepts: real openings per (vector, target) (default 3)")
    ap.add_argument("--concepts-no-adam", action="store_true", help="concepts: skip the Adam1010 side check")


def configure(a):
    a.roles = "seller"


# --- layers ------------------------------------------------------------------------------------------------------------------
def map_layer(L, n):
    return L if n == CANON else min(n - 1, int(round(L * (n - 1) / (CANON - 1))))


def layer_sets(n):
    m = lambda Ls: sorted({map_layer(L, n) for L in Ls})                      # noqa: E731
    out = {"concept": m(CONCEPT_LAYERS), "steer": map_layer(STEER_LAYER, n), "decode": map_layer(DECODE["layer"], n),
           "adam": {s["name"]: map_layer(s["layer"], n) for s in ADAM}, "n_layers": n, "mapped": n != CANON}
    out["keep"] = sorted(set(out["concept"]) | {out["steer"]})
    return out


# --- items --------------------------------------------------------------------------------------------------------------------
def distinct_fixtures(li, settings, n_fix):
    """One fixture per distinct floor among the first n_fix (the seller prompt depends only on the floor)."""
    X4 = _x4()
    doc, seen, out = li.mf.build(li.PHRASE), set(), []
    for f in doc["fixtures"][:n_fix]:
        if f["f_pct"] in seen:
            continue
        seen.add(f["f_pct"])
        fid, b = f"Y{f['id'][1:]}", f["f_pct"]
        out.append({"item_id": f"{fid}-S-cx", "kind": "fixture", "fixture": fid, "role": "seller", "base_limit": b, "seed": li.item_seed(fid, "seller", 1),
                    "prompts": {"N1": X4.render_side(li, settings, b, "D2", False), "N6": X4.render_side(li, settings, b, "D8", False)}})
    return out


def concept_pairs(count_templates):
    """[(concept, a, b)]: b is the 'large' side (big number, money, 'million', long digits, high stakes, modest)."""
    out = []
    for t in count_templates:
        out += [("count", t.format(A=b), t.format(A=b * 10 ** 6)) for b in BASES]
    for t in MONEY_T:
        out += [("money", t.format(A=f"{b} dollars"), t.format(A=f"{b * 10 ** 6} dollars")) for b in BASES]
        out += [("million", t.format(A=f"{b} dollars"), t.format(A=f"{b} million dollars")) for b in BASES]
    for t in DIGITS_T:
        out += [("digits", t.format(A=b), t.format(A=b * 10 ** 6)) for b in BASES]
    for fr in STAKES_F:
        out += [("stakes", fr.format(X=lo), fr.format(X=hi)) for lo, hi in STAKES_P]
    for fr in MODEST_F:
        out += [("modesty", fr.format(X=bold), fr.format(X=modest)) for bold, modest in MODEST_P]
    return out


def concept_items(li):
    out, k = [], {}
    for c, a, b in concept_pairs(li.E3_NONMONEY):
        i = k[c] = k.get(c, -1) + 1
        out.append({"item_id": f"C-{c}-{i:02d}", "kind": "concept", "concept": c, "half": i % 2, "a": a, "b": b})
    return out


def steer_items(fx):
    return [dict(it, item_id=f"{it['fixture']}-S-{t}-cs", target=t, sign=1.0 if t == "N1" else -1.0) for it in fx for t in ("N1", "N6")]


# --- row specs ----------------------------------------------------------------------------------------------------------------
def patch_rows(Ls):
    out = {}
    for t, sg in (("N1", 1.0), ("N6", -1.0)):
        out[t] = [{"n": f"{v}@{L}", "ops": [["vec", L, "A", sg, v]]} for L in Ls["concept"] for v in VECS]
    return out


def steer_plan(Ls, samples):
    return [[v, Ls["steer"], 1.0, int(samples)] for v in VECS]


def sae_rows(top):
    """top: latent ids ordered by |mean attribution|."""
    ids = [int(x[0]) for x in top]
    rows = [{"n": f"top{k}", "latents": ids[:k]} for k in TOPS if len(ids) >= k or k == TOPS[0]]
    k16 = ids[:16]
    rows += [{"n": "top16 digits", "latents": k16, "where": "span"}, {"n": "top16 after", "latents": k16, "where": "aligned"},
             {"n": "all latents", "latents": "all"}, {"n": "all + error", "latents": "all+err"},
             {"n": "random16", "latents": {"random": 16, "seed": "concepts/random16/v1", "exclude": ids}},
             {"n": "random64", "latents": {"random": 64, "seed": "concepts/random64/v1", "exclude": ids}}]
    return rows


def _nrows(rows):
    return {t: len(v) for t, v in rows.items()}


def estimate(n):
    per = PER_S
    t = sum(per[k] * n.get(k, 0) for k in per)
    return t / 60, t * SLOW / 60


# --- setup --------------------------------------------------------------------------------------------------------------------
def _kw(ctx, n_layers):
    Ls = layer_sets(n_layers)
    ctx["layers"] = Ls
    schema, a = ctx["schema"], ctx["a"]
    ctx["kw"] = {
        "build": {"keep": Ls["keep"]}, "stats": {},
        "patch": {"rows": patch_rows(Ls), "batch": a.batch},
        "steer": {"plan": steer_plan(Ls, a.concepts_steer_samples), "n_none": 8, "batch": 16, "schema": schema},
        "sae_load": {"drop": []},
        "quality": {"sae": DECODE["name"], "layer": Ls["decode"]},
        "attr": {"sae": DECODE["name"], "layer": Ls["decode"], "skip": None},
        "pick": {"sae": DECODE["name"], "k": max(TOPS), "skip": None},
        "sae_patch": {"sae": DECODE["name"], "layer": Ls["decode"], "rows": [], "batch": a.batch, "skip": None},
        "profile": {"sae": DECODE["name"], "layer": Ls["decode"], "latents": [], "skip": None},
        "adam_load": {"drop": [DECODE["name"]]},
        **{f"{s['name']} quality": {"sae": s["name"], "layer": Ls["adam"][s["name"]]} for s in ADAM}}


def setup(a, settings, schema, li):
    _LI[:] = [li]
    fx = distinct_fixtures(li, settings, a.fixtures)
    n_disc = min(a.concepts_discovery, max(1, len(fx) // 2))
    disc, held = [it["fixture"] for it in fx[:n_disc]], [it["fixture"] for it in fx[n_disc:]]
    cx = concept_items(li)
    sae_specs = [dict(DECODE, item_id=DECODE["name"], url=url(DECODE))]
    adam_specs = [] if a.concepts_no_adam else [dict(s, item_id=s["name"], url=url(s)) for s in ADAM]
    ctx = {"li": li, "a": a, "schema": schema, "fx": fx, "cx": cx, "disc": disc, "held": held, "sae_specs": sae_specs,
           "adam_specs": adam_specs, "model": a.model, "gate": None, "pick": None}
    _kw(ctx, CANON)
    n = {"concepts build": len(fx) * 7 + len(cx), "concepts stats": 1, "concepts patch": len(fx), "concepts steer": 2 * len(fx),
         "sae30 load": 1, "sae30 quality": len(fx), "sae30 attr": len(disc), "sae30 pick": 1, "sae30 patch": len(held),
         "sae30 profile": len(cx), "adam load": len(adam_specs), "adam41 quality": len(fx) if adam_specs else 0,
         "adam44 quality": len(fx) if adam_specs else 0}
    typ, slow = estimate(n)
    ctx["estimate_min"] = [round(typ, 1), round(slow, 1)]
    li.STEP_GUESS_S.update({k: v for k, v in PER_S.items()})
    floors = ", ".join(f"{it['fixture']}:{it['base_limit']}" for it in fx)
    plan = (f"Concepts on {a.model} (sellers): {len(fx)} distinct floors ({floors}); discovery {len(disc)}, held-out {len(held)}. "
            f"Part 1: {len(cx)} concept sentence pairs ({', '.join(CONCEPTS)}); geometry at every layer; forced-opening patch "
            f"{len(VECS)} vectors x layers {'/'.join(map(str, ctx['layers']['concept']))}; real openings at layer {STEER_LAYER} "
            f"({a.concepts_steer_samples} per vector and target). Part 2: Decode's layer-30 SAE (downloads 5.6 GB at a pinned "
            f"revision, sha256-checked): quality gate, attribution on discovery, latent patching on held-out, profile"
            + ("" if a.concepts_no_adam else "; Adam1010's SAEs at 41/44 (3.7 GB): quality only")
            + f"; about {typ:.0f} min of experiment time ({slow:.0f} if 1.5x slower; --budget-min {a.budget_min:g} stops new items)")
    inputs = [{"step": "fixture", **it} for it in fx] + [{"step": "concept", **c} for c in cx] + \
             [{"step": "sae", **s} for s in sae_specs + adam_specs]
    return {"exp": "Concepts", "plan": plan, "inputs": inputs, "ctx": ctx, "steps": steps, "live": live, "after_step": after_step}


def steps(ctx):
    fx, kw = ctx["fx"], ctx["kw"]
    dfx = [it for it in fx if it["fixture"] in ctx["disc"]]
    hfx = [it for it in fx if it["fixture"] in ctx["held"]]
    out = [("concepts build", "concepts_build", fx + ctx["cx"], kw["build"], 10),
           ("concepts stats", "concepts_stats", [{"item_id": "stats"}], kw["stats"], 1),
           ("concepts patch", "concepts_patch", fx, kw["patch"], 2),
           ("concepts steer", "concepts_steer", steer_items(fx), kw["steer"], 2),
           ("sae30 load", "sae_load", ctx["sae_specs"], kw["sae_load"], 1),
           ("sae30 quality", "sae_quality", fx, kw["quality"], 3),
           ("sae30 attr", "sae_attr", dfx, kw["attr"], 1),
           ("sae30 pick", "sae_attr_pick", [{"item_id": "pick"}], kw["pick"], 1),
           ("sae30 patch", "sae_patch", hfx, kw["sae_patch"], 2),
           ("sae30 profile", "sae_profile", ctx["cx"], kw["profile"], 25)]
    if ctx["adam_specs"]:
        out.append(("adam load", "sae_load", ctx["adam_specs"], kw["adam_load"], 1))
        out += [(f"{s['name']} quality", "sae_quality", fx, kw[f"{s['name']} quality"], 3) for s in ctx["adam_specs"]]
    return out


def on_pod_info(ctx, pod_info):
    n = pod_info.get("num_hidden_layers")
    if n and int(n) != ctx["layers"]["n_layers"]:
        _kw(ctx, int(n))


# --- gates and hand-overs -----------------------------------------------------------------------------------------------------
def quality_summary(recs):
    use = [r for r in recs if not r.get("skipped") and r.get("prompts")]
    out = {"n": len(recs), "usable": len(use), "rehearsal": any(r.get("rehearsal") for r in recs)}
    if not use:
        return out
    segs_ = sorted({s for r in use for p in r["prompts"].values() for s in p["stats"]})
    out["stats"] = {s: {m: st.mean(p["stats"][s][m] for r in use for p in r["prompts"].values() if s in p["stats"])
                        for m in ("rel_err", "fvu_local", "cos", "l0")} for s in segs_}
    kept = []
    for r in use:
        P = r["prompts"]
        g = P["N6"]["ld"] - P["N1"]["ld"]
        if abs(g) > 1e-6:
            kept.append(((P["N6"]["ld_spliced"] - P["N1"]["ld_spliced"]) / g, g, P["N6"]["ld_spliced"] - P["N1"]["ld_spliced"]))
    if kept:
        out["kept_mean"] = st.mean(k for k, _, _ in kept)
        out["kept_median"] = st.median(k for k, _, _ in kept)
        out["kept_pooled"] = sum(s for _, _, s in kept) / sum(g for _, g, _ in kept)
        out["gap_mean"] = st.mean(g for _, g, _ in kept)
    out["ld_shift"] = {k: st.mean(r["prompts"][k]["ld_spliced"] - r["prompts"][k]["ld"] for r in use) for k in ("N1", "N6")}
    out["pk"] = {k: [st.mean(r["prompts"][k]["pk"] for r in use), st.mean(r["prompts"][k]["pk_spliced"] for r in use)] for k in ("N1", "N6")}
    var = [p.get("variant") for r in use for p in r["prompts"].values() if p.get("variant")]
    if var:
        out["variant"] = {str(v): var.count(v) for v in sorted(set(var))}
    tops = {}
    for r in use:
        for j, d, *_ in r.get("top_delta_A") or []:
            tops.setdefault(j, []).append(d)
    out["top_delta_A"] = sorted(([j, st.mean(v), len(v)] for j, v in tops.items()), key=lambda x: -abs(x[1]) * x[2])[:10]
    return out


def gate(Q):
    if Q.get("kept_mean") is None:
        return False, "no usable splice readings"
    k = Q["kept_mean"]
    if not GATE["kept_lo"] <= k <= GATE["kept_hi"]:
        return False, f"the spliced SAE keeps {100 * k:.0f}% of the gap (needs {100 * GATE['kept_lo']:.0f}-{100 * GATE['kept_hi']:.0f}%)"
    return True, f"keeps {100 * k:.0f}% of the gap (median {100 * Q['kept_median']:.0f}%)"


def after_step(label, got, ctx):
    note = (lambda s: ctx["panel"].note(s)) if ctx.get("panel") is not None else (lambda s: None)
    kw = ctx["kw"]
    if label == "concepts build":
        recs = [r for r in got.get(label) or [] if r.get("kind") == "fixture"]
        use = [r for r in recs if not r.get("skipped")]
        if len(use) < min(12, len(recs)):
            return f"only {len(use)} of {len(recs)} floors have two distinct plain greedy prices"
        if len({(r["n_H"], r["n_S"], r["n_reply"]) for r in use}) != 1:
            return "the aligned segments differ in length between floors"
        note(f"build: {len(use)}/{len(recs)} floors usable; A = {use[0]['n_H']} + {use[0]['n_S']} + {use[0]['n_reply']} tokens")
        return None
    if label == "sae30 quality":
        Q = quality_summary(got.get(label) or [])
        ok, why = gate(Q)
        ctx["gate"] = {"pass": ok, "why": why, "rehearsal": Q.get("rehearsal")}
        if Q.get("rehearsal"):
            note(f"SAE gate: REHEARSAL (synthetic SAE), not applied: {why}")
            return None
        note(f"SAE gate: {'PASS' if ok else 'FAIL'} ({why})")
        if not ok:
            for k in ("attr", "pick", "sae_patch", "profile"):
                kw[k]["skip"] = "SAE gate failed: " + why
        return None
    if label == "sae30 pick":
        r = (got.get(label) or [{}])[0]
        if r.get("skipped") or not r.get("top"):
            return None
        ctx["pick"] = r
        kw["sae_patch"]["rows"][:] = sae_rows(r["top"])
        kw["profile"]["latents"][:] = [int(x[0]) for x in r["top"]]
        note("SAE pick: top latents " + ", ".join(str(x[0]) for x in r["top"][:6]) + " ...")
    return None


# --- analysis -----------------------------------------------------------------------------------------------------------------
def attr_summary(recs):
    use = [r for r in recs if not r.get("skipped") and "gap" in r]
    if not use:
        return {"usable": 0, "skipped": next((r["skipped"] for r in recs if r.get("skipped")), None)}
    err = [r["err_aligned"] + r["err_span"] for r in use]
    feat = [r["feat_total"] for r in use]
    return {"usable": len(use), "gap": st.mean(r["gap"] for r in use), "feat": st.mean(feat), "err": st.mean(err),
            "feat_share": st.mean(feat) / (st.mean(feat) + st.mean(err)) if st.mean(feat) + st.mean(err) else None,
            "linear_vs_gap": (st.mean(feat) + st.mean(err)) / st.mean(r["gap"] for r in use) if st.mean(r["gap"] for r in use) else None,
            "feat_span_share": st.mean(r["feat_span"] for r in use) / st.mean(feat) if st.mean(feat) else None}


def profile_summary(recs, pick):
    use = [r for r in recs if not r.get("skipped") and r.get("latents")]
    if not use or not pick:
        return {}
    lat = use[0]["latents"]
    out = {}
    for j, x in enumerate(lat):
        per = {}
        for c in CONCEPTS:
            rs = [r for r in use if r["concept"] == c]
            if rs:
                per[c] = [st.mean(r["b"][0][j] - r["a"][0][j] for r in rs), st.mean(r["b"][1][j] - r["a"][1][j] for r in rs),
                          st.mean(max(r["a"][1][j], r["b"][1][j]) > 0 for r in rs)]
        out[str(x)] = per
    return out


def flip_table(recs):
    """{target: {row: {"flips", "n"}}}: fixtures whose forced pair now prefers the other scale's price (N1: ld > 0;
    N6: ld < 0). With p(a)+p(b) (ratio_table's pk) this separates a real flip from a broken readout, which the share
    alone can't (a random direction pulls ld toward 0 from both sides; X61, X63)."""
    out = {}
    for r in recs:
        for t, T in (r.get("targets") or {}).items():
            for x in T["rows"]:
                if x["n"] != "clean":
                    c = out.setdefault(t, {}).setdefault(x["n"], {"flips": 0, "n": 0})
                    c["n"] += 1
                    c["flips"] += (x["ld"] > 0) if t == "N1" else (x["ld"] < 0)
    return out


def summarize(got, ctx):
    li = ctx.get("li") or _li()
    X4 = _x4()
    ref, ref_from = X4.e0_plain(li, ctx.get("model"))
    b = [r for r in got.get("concepts build") or [] if r.get("kind") == "fixture"]
    use = [r for r in b if not r.get("skipped")]
    out = {"layers": ctx["layers"], "discovery": ctx["disc"], "held_out": ctx["held"], "concepts": list(CONCEPTS),
           "build": {"n": len(b), "usable": len(use), "skipped": {r["fixture"]: r["skipped"] for r in b if r.get("skipped")},
                     "stand_in": any(r.get("stand_in") for r in use),
                     "gap_mean": st.mean(r["ld"]["N6"] - r["ld"]["N1"] for r in use) if use else None,
                     "prices": {r["fixture"]: r.get("price") for r in b}},
           "stats": (got.get("concepts stats") or [None])[0],
           "patch": X4.ratio_table(got.get("concepts patch") or [], "plain"),
           "patch_flips": flip_table([r for r in got.get("concepts patch") or [] if not r.get("skipped")]),
           "steer": X4.summarize_steer(got.get("concepts steer") or [], li, ref), "steer_ref": ref_from,
           "sae_load": got.get("sae30 load") or [], "sae_quality": quality_summary(got.get("sae30 quality") or []),
           "gate": ctx.get("gate"), "attr": attr_summary(got.get("sae30 attr") or []), "pick": ctx.get("pick"),
           "sae_patch": X4.ratio_table([r for r in got.get("sae30 patch") or [] if not r.get("skipped")], "plain"),
           "profile": profile_summary(got.get("sae30 profile") or [], ctx.get("pick")),
           "adam": {s["name"]: quality_summary(got.get(f"{s['name']} quality") or []) for s in ctx["adam_specs"]},
           "adam_load": got.get("adam load") or [], "estimate_min": ctx.get("estimate_min"),
           "pins": {"decode": {k: DECODE[k] for k in ("repo", "rev", "path", "sha256", "trained_on")},
                    "adam": [{k: s[k] for k in ("repo", "rev", "path", "sha256", "trained_on")} for s in ADAM]}}
    recs = [r for lab in ("concepts patch", "sae30 patch") for r in got.get(lab) or [] if not r.get("skipped")]
    out["checks"] = {"hooks_ok": all(T.get("hooks_ok", True) for r in recs for T in (r.get("targets") or {}).values()),
                     "clean_spread_max": max((T.get("clean_spread") or 0 for r in recs for T in (r.get("targets") or {}).values()), default=None),
                     "steer_aligned": [sum(1 for r in got.get("concepts steer") or [] for x in r["rows"] if x["vector"] != "none" and x.get("aligned")),
                                       sum(1 for r in got.get("concepts steer") or [] for x in r["rows"] if x["vector"] != "none")],
                     "loo": sorted({str(r.get("loo")) for lab in ("concepts patch", "concepts steer") for r in got.get(lab) or []})}
    out["secs_per_item"] = {lab: st.mean(r["secs"] for r in got[lab] if "secs" in r) if got.get(lab) and any("secs" in r for r in got[lab])
                            else None for lab, *_ in steps(ctx)}
    return out


def manifest(ctx):
    return {"name": "concepts", "plan": "manuscript/PLAN.md D4; config/interp/plan_next.md", "layers": ctx["layers"],
            "fixtures": [it["fixture"] for it in ctx["fx"]], "discovery": ctx["disc"], "held_out": ctx["held"],
            "concept_pairs": {c: sum(1 for x in ctx["cx"] if x["concept"] == c) for c in CONCEPTS},
            "saes": [{k: s[k] for k in ("name", "kind", "layer", "repo", "rev", "path", "size", "sha256")} for s in ctx["sae_specs"] + ctx["adam_specs"]],
            "gate": GATE, "rows": _nrows(ctx["kw"]["patch"]["rows"]), "estimate_min": ctx.get("estimate_min")}


# --- panel, preview, report ---------------------------------------------------------------------------------------------------
def _p(v, d=0):
    return "  —  " if v is None else f"{100 * v:+.{d}f}%"


def _ci(c):
    return f" [{100 * c['ci'][0]:+.0f}, {100 * c['ci'][1]:+.0f}]" if c and c.get("ci") else ""


def live(got):
    out = []
    S = (got.get("concepts stats") or [None])[0]
    if S:
        rep = S["per_segment"].get("reply") or {}
        L = min(41, S["n_layers"] - 1)
        out.append(f"geometry L{L} (reply): " + " ".join(f"{c} {rep['cos'][c][L]:+.2f}" for c in S["concepts"])
                   + f"; span {rep['span_share'][L]:.2f} (random {rep['span_share_random'][L]:.3f})")
    p = got.get("concepts patch") or []
    if p:
        T_ = _x4().ratio_table(p, "plain", ci=False)
        L = next((n.split("@")[1] for n in T_.get("N1", {}) if n.startswith("vN@44")), None) or \
            (sorted({n.split("@")[1] for n in T_.get("N1", {})}, key=int) or [None])[-1]
        if L:
            out.append(f"patch L{L} N1: " + " ".join(f"{v} {_p(T_.get('N1', {}).get(f'{v}@{L}', {}).get('share')).strip()}" for v in VECS))
    q = got.get("sae30 quality") or []
    if q:
        Q = quality_summary(q)
        if Q.get("kept_mean") is not None:
            out.append(f"SAE L30: kept {100 * Q['kept_mean']:.0f}% of the gap; reply rel. error "
                       f"{Q['stats'].get('reply', {}).get('rel_err', float('nan')):.3f}, L0 {Q['stats'].get('reply', {}).get('l0', 0):.0f}")
    return out


def stub(task, items, ctx):
    """Preview only: canned shapes, not a model."""
    Ls, n = ctx["layers"], ctx["layers"]["n_layers"]
    sig = lambda x: 1 / (1 + math.exp(-x))                                      # noqa: E731
    shares = {"vN": 1.0, "count": 0.1, "money": 0.3, "million": 0.45, "digits": 0.15, "stakes": 0.05, "modesty": 0.35, "rand": 0.0}
    ld0 = {"N1": -12.0, "N6": 5.0}

    def targets(rows_by_t, share):
        T_ = {}
        for t in ("N1", "N6"):
            other = ld0["N6" if t == "N1" else "N1"]
            rows = [{"n": x["n"], "ld": round(ld0[t] + share(x) * (other - ld0[t]), 4), "lpa": -0.3, "lpb": -2.0} for x in rows_by_t(t)]
            T_[t] = {"ld_clean": ld0[t], "clean_spread": 0.0, "hooks_ok": True,
                     "rows": rows + [{"n": "clean", "ld": ld0[t], "lpa": -0.3, "lpb": -2.0}]}
        return T_
    if task == "concepts_build":
        out = []
        for it in items:
            if it.get("kind") == "concept":
                out.append({"item_id": it["item_id"], "kind": "concept", "concept": it["concept"], "n_suffix": 6, "secs": 0.01})
            else:
                out.append({"item_id": it["item_id"], "kind": "fixture", "fixture": it["fixture"], "base_limit": it["base_limit"],
                            "n_layers": n, "n_A": 117, "n_H": 38, "n_S": 68, "n_reply": 11, "n_span": {"N1": 2, "N6": 8},
                            "price": {"N1": 120, "N6": 95 * 10 ** 6}, "stand_in": False, "ld": dict(ld0), "secs": 0.1})
        return out
    if task == "concepts_stats":
        base = {"count": 0.1, "money": 0.25, "million": 0.4, "digits": 0.2, "stakes": 0.05, "modesty": 0.3}
        per = {s: {"cos": {c: [round(v * sig((L - 38) / 3), 3) for L in range(n)] for c, v in base.items()},
                   "cos_token": {c: [round(0.8 * v * sig((L - 38) / 3), 3) for L in range(n)] for c, v in base.items()},
                   "span_share": [round(0.35 * sig((L - 38) / 3), 3) for L in range(n)],
                   "span_share_token": [round(0.25 * sig((L - 38) / 3), 3) for L in range(n)],
                   "span_share_random": [0.0015] * n, "norm_v": [20.0] * n} for s in ("H", "S", "reply", "A")}
        return [{"n_fixtures": len(ctx["fx"]), "n_layers": n, "concepts": sorted(base), "per_segment": per,
                 "split_half_cos": {c: [0.9] * n for c in base}, "n_pairs": {c: 20 for c in base}}]
    if task == "concepts_patch":
        def sh(x):
            v, L = x["n"].split("@")
            return shares[v] * sig((int(L) - 40.5) / 1.2)
        return [{"item_id": it["item_id"], "fixture": it["fixture"], "base_limit": it["base_limit"], "loo": True,
                 "targets": targets(lambda t: ctx["kw"]["patch"]["rows"][t], sh), "secs": 0.1} for it in items]
    if task == "concepts_steer":
        out = []
        for it in items:
            s = 0 if it["target"] == "N1" else 6
            base = (1.41 if s == 0 else 1.21) * it["base_limit"]
            rows = [{"vector": "none", "layer": None, "alpha": 0.0, "sample": i} for i in range(ctx["kw"]["steer"]["n_none"])]
            rows += [{"vector": v, "layer": L, "alpha": a_, "sample": j} for v, L, a_, k in ctx["kw"]["steer"]["plan"] for j in range(k)]
            for x in rows:
                mv = 0 if x["vector"] == "none" else -it["sign"] * shares[x["vector"]] * 0.2 * it["base_limit"]
                x.update(price=int(round((base + mv) / 5) * 5) * 10 ** s, errors=[], action="OFFER", aligned=True)
            out.append({"item_id": it["item_id"], "fixture": it["fixture"], "role": "seller", "target": it["target"], "sign": it["sign"],
                        "base_limit": it["base_limit"], "rehearsal": False, "rows": rows, "secs": 0.1})
        return out
    if task == "sae_load":
        return [{"item_id": it["item_id"], "name": it["name"], "kind": it["kind"], "d_in": 5376,
                 "d_sae": 131072 if it["kind"] == "jumprelu" else 43008, "downloaded": True, "secs": 0.1} for it in items]
    if task == "sae_quality":
        out = []
        for it in items:
            P = {}
            for k in ("N1", "N6"):
                stats = {s: {"rel_err": 0.08, "fvu_local": 0.3, "cos": 0.95, "l0": 95.0, "n": 10} for s in ("ctx", "span", "H", "S", "reply")}
                P[k] = {"ld": ld0[k], "pk": 0.8, "ld_spliced": ld0[k] * 0.9, "pk_spliced": 0.7, "stats": stats,
                        "variant": None}
            out.append({"item_id": it["item_id"], "fixture": it["fixture"], "prompts": P, "rehearsal": False,
                        "top_delta_A": [[101, 2.5, 0.0, 2.5], [7, -1.2, 1.3, 0.1]], "secs": 0.1})
        return out
    if task == "sae_attr":
        return [{"item_id": it["item_id"], "fixture": it["fixture"], "gap": 17.0, "feat_total": 12.0, "feat_aligned": 5.0,
                 "feat_span": 7.0, "err_aligned": 2.0, "err_span": 2.5, "top": [], "secs": 0.1} for it in items]
    if task == "sae_attr_pick":
        top = [[1000 + j, round(2.0 / (1 + j), 4), round(2.2 / (1 + j), 4)] for j in range(64)]
        return [{"item_id": "pick", "sae": "decode30", "n_fixtures": len(ctx["disc"]), "top": top, "total_mean": 12.0,
                 "lens": {str(x[0]): [["million", " millions"], [" cents", " cheap"]] for x in top[:8]}}]
    if task == "sae_patch":
        sh = {"top1": 0.1, "top4": 0.25, "top16": 0.4, "top64": 0.55, "top16 digits": 0.3, "top16 after": 0.1, "all latents": 0.8,
              "all + error": 1.0, "random16": 0.0, "random64": 0.02}
        return [{"item_id": it["item_id"], "fixture": it["fixture"], "base_limit": it["base_limit"],
                 "targets": targets(lambda t: ctx["kw"]["sae_patch"]["rows"], lambda x: sh.get(x["n"], 0.0)), "secs": 0.1}
                for it in items]
    if task == "sae_profile":
        lat = ctx["kw"]["profile"]["latents"]
        return [{"item_id": it["item_id"], "concept": it["concept"], "latents": lat,
                 "a": [[0.0] * len(lat), [0.1] * len(lat)],
                 "b": [[(1.0 if it["concept"] in ("million", "money") and i < 4 else 0.05) for i in range(len(lat))], [1.5] * len(lat)]}
                for it in items]
    raise ValueError(task)


def print_patch(P, F, vecs, ci_layer):
    """Per target: share of the plain gap by layer (CI at ci_layer), then fixtures flipped of n and p(a)+p(b) by layer."""
    Lz = sorted({int(n.split("@")[1]) for t in P for n in P[t] if "@" in n})

    def fp(t, n):
        f, k = F.get(t, {}).get(n), P.get(t, {}).get(n, {}).get("pk")
        return "—" if not f or k is None else f"{f['flips']}/{f['n']} {k:.2f}"
    for t, lab in (("N1", "N1 pushed toward x1e6"), ("N6", "N6 pushed toward x1")):
        print(f"  forced-opening patch, {lab}: share of the plain gap by layer (CI at L{ci_layer}) | fixtures flipped, p(a)+p(b):")
        print("    " + " " * 9 + "".join(f"{'L' + str(L):>7}" for L in Lz) + " " * 14 + "".join(f"{'L' + str(L):>11}" for L in Lz))
        for v in vecs:
            if not any(f"{v}@{L}" in P.get(t, {}) for L in Lz):
                continue
            print(f"    {v + ('*' if v == 'rand' else ''):<9}"
                  + "".join(f"{_p(P.get(t, {}).get(f'{v}@{L}', {}).get('share')).strip():>7}" for L in Lz)
                  + f"{_ci(P.get(t, {}).get(f'{v}@{ci_layer}')):<14}" + "".join(f"{fp(t, f'{v}@{L}'):>11}" for L in Lz))
    if "rand" in vecs:
        print("    * random direction at vN's size: it breaks the readout (low p(a)+p(b)), so its share and flips are not a chance level")


def _spark(li, vals):
    return li.spark([None if v is None else max(-0.2, min(1.2, v)) for v in vals])


def report(s, preview):
    li = _li()
    print("Concepts: what the x1e6 - x1 direction is made of (exploratory" + (", PREVIEW: canned shapes" if preview else "") + ")")
    Ls = s.get("layers") or {}
    if Ls.get("mapped"):
        print(f"  NOTE: a {Ls['n_layers']}-layer model; the 31B's layers were mapped onto it")
    B = s.get("build") or {}
    print(f"  floors: {B.get('usable')}/{B.get('n')} usable" + (" (STAND-IN prices: rehearsal)" if B.get("stand_in") else "")
          + (f"; mean plain gap {B['gap_mean']:+.1f} nats" if B.get("gap_mean") is not None else "")
          + f"; discovery {', '.join(s.get('discovery') or [])}; held-out {', '.join(s.get('held_out') or [])}")
    S = s.get("stats")
    if S:
        for seg in ("reply", "S", "H"):
            p = S["per_segment"].get(seg)
            if not p:
                continue
            pick = [L for L in (20, 30, 36, 40, 41, 42, 44, 48, 52, 59) if L < S["n_layers"]] or list(range(S["n_layers"]))
            print(f"  geometry ({seg}; cosine of v with each concept; span share of |v|^2, random in brackets):")
            for c in S["concepts"]:
                print(f"    {c:<8} {_spark(li, p['cos'][c])}  " + " ".join(f"L{L} {p['cos'][c][L]:+.2f}" for L in pick)
                      + f"   split-half L{pick[len(pick) // 2]} {S['split_half_cos'][c][pick[len(pick) // 2]]:+.2f}")
            print(f"    span     {_spark(li, p['span_share'])}  " + " ".join(f"L{L} {p['span_share'][L]:.2f}({p['span_share_random'][L]:.3f})" for L in pick))
    P, F = s.get("patch") or {}, s.get("patch_flips") or {}
    if P:
        print_patch(P, F, VECS, Ls.get("steer"))
    Sx = s.get("steer") or {}
    if Sx:
        print(f"  real openings at layer {Ls.get('steer')} (points moved; share of the plain gap from {s.get('steer_ref') or 'no E0 run'}):")
        for t, cells in Sx.items():
            for k, c in sorted(cells.items()):
                extra = "" if k == "none" else f"  moved {c.get('moved_pts') or 0:+.1f} pts, share {_p(c.get('share')).strip()}{_ci(c)}"
                print(f"    {t} {k:<14} gap {_p(c['mean_gap'], 1).strip():>7}  valid {c['valid']}/{c['n']}  off-scale {c['off_scale']}{extra}")
    for lab, Q in [("Decode L30", s.get("sae_quality") or {})] + [(f"Adam {k}", v) for k, v in (s.get("adam") or {}).items()]:
        if not Q.get("usable"):
            continue
        st_ = Q.get("stats") or {}
        print(f"  {lab} SAE on our prompts: " + "; ".join(f"{seg} rel.err {v['rel_err']:.3f} cos {v['cos']:.3f} L0 {v['l0']:.0f}"
                                                         for seg, v in st_.items() if not seg.startswith("variant")))
        if Q.get("kept_mean") is not None:
            print(f"    spliced in: keeps {_p(Q['kept_mean']).strip()} of the gap (median {_p(Q['kept_median']).strip()}, pooled "
                  f"{_p(Q['kept_pooled']).strip()}); ld shift N1 {Q['ld_shift']['N1']:+.2f} N6 {Q['ld_shift']['N6']:+.2f}; p(a)+p(b) "
                  + ", ".join(f"{k} {v[0]:.2f}->{v[1]:.2f}" for k, v in Q["pk"].items()) + (f"; encoder variant {Q['variant']}" if Q.get("variant") else ""))
    G = s.get("gate")
    if G:
        print(f"  SAE gate: {'PASS' if G['pass'] else 'FAIL'} ({G['why']})" + (" [REHEARSAL: not applied]" if G.get("rehearsal") else ""))
    A = s.get("attr") or {}
    if A.get("usable"):
        print(f"  attribution (discovery, {A['usable']} floors): gap {A['gap']:+.2f}; latents {A['feat']:+.2f}, error {A['err']:+.2f} "
              f"(latent share {_p(A['feat_share']).strip()}; digits' share of the latents' part {_p(A['feat_span_share']).strip()}); "
              f"first-order total / gap {A['linear_vs_gap']:.2f}")
    pk = s.get("pick")
    if pk:
        print("  top latents (mean attribution; unembedding top / bottom tokens):")
        for j, m, _a in pk["top"][:10]:
            lens = pk.get("lens", {}).get(str(j))
            pf = (s.get("profile") or {}).get(str(j)) or {}
            print(f"    #{j:<7} {m:+.3f}  " + (f"up {lens[0][:5]} down {lens[1][:5]}" if lens else "")
                  + ("  concepts (b - a, suffix mean): " + " ".join(f"{c} {v[0]:+.2f}" for c, v in pf.items()) if pf else ""))
    K = s.get("sae_patch") or {}
    if K:
        print("  latent patching (held-out; share of the plain gap, 95% CI; N1 toward x1e6 | N6 toward x1):")
        for nm in sorted({n for t in K for n in K[t]}, key=lambda n: (n.startswith("random"), n.startswith("all"), n)):
            print(f"    {nm:<14} " + " | ".join(f"{t} {_p(K.get(t, {}).get(nm, {}).get('share')).strip():>6}{_ci(K.get(t, {}).get(nm))}"
                                               for t in ("N1", "N6")))
    ch = s.get("checks") or {}
    print(f"  checks: hooks fired {ch.get('hooks_ok')}; clean rows agree within {ch.get('clean_spread_max')} nats; steered openings "
          f"aligned {ch.get('steer_aligned')}; leave-one-out {ch.get('loo')}")
    sec = s.get("secs_per_item") or {}
    print("  seconds per item: " + ", ".join(f"{k} {v:.1f}" for k, v in sec.items() if v)
          + (f" (estimated {s['estimate_min'][0]} min typical)" if s.get("estimate_min") else ""))
