"""interp_e6_laptop.py (LAPTOP): E6, magnitude as a number vs magnitude as stakes, in Gemma 4 31B.
launch_interp --experiment e6 (shim scripts/interp/exp_e6.py; pod code scripts/interp/tasks_e6.py). Exploratory;
predictions in config/interp/e6_predictions.md, committed before any paid run (setup refuses otherwise).

Part A, the number: 9 mantissas x 10^0..10^7 in five notations (digits "81000000 dollars", words "81 million
dollars", short "$81M", commas "$81,000,000", and a non-money count "81000000 people"), one short sentence each;
residual states at the sentence's last token, all layers; magnitude directions per notation and layer, their
held-out and cross-notation transfer; an MLP neuron census at layers 4, 10, 20, 30 and 41.
Part B, the stakes: the negotiation scale direction (E5's seller build on Study 3's prompts), collapsed to one
vector per layer from its reply-token rows (vR), added on the model-turn header and a forced answer prefix of
non-negotiation prompts: risk choices, amount comparisons, scale-free percentages, sums, and Fermi estimates, with
negotiation openings as the positive control. Controls: the S-row vector (vS), the non-money contrast (u) and a
random vector, at vR's norm.
"""
import hashlib
import math
import os
import random
import statistics as st
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import interp_e4_laptop as X4      # noqa: E402  layer mapping, helpers
import interp_e5_laptop as X5      # noqa: E402  Study 3's seller items

PRED_FILE = "config/interp/e6_predictions.md"
_LI = []
MANT = (12, 23, 37, 45, 58, 64, 73, 81, 96)
EXPS = tuple(range(8))
NOTATIONS = ("digits", "words", "short", "commas", "count")
MLP_60 = (4, 10, 20, 30, 41)
STAKE_60 = (42, 44)
PLAN_60 = (("vR", 42, 1.0), ("vR", 42, -1.0), ("vR", 42, 2.0), ("vR", 42, -2.0), ("vR", 44, 2.0), ("vR", 44, -2.0),
           ("vS", 42, 2.0), ("vS", 42, -2.0), ("u", 42, 2.0), ("u", 42, -2.0), ("rand", 42, 2.0), ("rand", 42, -2.0))
N_NONE, BATCH = 4, 16
FAMILIES = ("nego", "risk", "compare", "percent", "sum", "fermi")
JSON_CHOICE = 'Reply with JSON only, in the form {"choice": "A"} or {"choice": "B"}.'


def _li():
    if not _LI:
        import launch_interp
        _LI.append(launch_interp)
    return _LI[0]


# --- flags -----------------------------------------------------------------------------------------------------------
def add_args(ap):
    ap.add_argument("--e6-limit", type=int, default=0, help="e6: REHEARSAL ONLY: cap items per family (refused on a paid launch)")


def configure(a):
    a.roles = "seller"


# --- Part A items ----------------------------------------------------------------------------------------------------
def _scaled(v):
    """(number text, unit index): 81000000 -> ('81', 2) for million."""
    for k, unit in ((3, 10 ** 9), (2, 10 ** 6), (1, 10 ** 3)):
        if v >= unit:
            return f"{v / unit:g}", k
    return str(v), 0


def amount(v, notation):
    q, k = _scaled(v)
    if notation == "digits":
        return f"{v} dollars"
    if notation == "commas":
        return f"${v:,}"
    if notation == "words":
        return f"{q} {('', 'thousand ', 'million ', 'billion ')[k]}dollars".replace("  ", " ")
    if notation == "short":
        return f"${q}{('', 'K', 'M', 'B')[k]}"
    return f"{v} people"


def probe_items(limit=0):
    mant, exps = (MANT[:3], EXPS[:4]) if limit else (MANT, EXPS)
    out = []
    for nt in NOTATIONS:
        for m in mant:
            for e in exps:
                v = m * 10 ** e
                text = (f"The event was attended by {amount(v, nt)} in total." if nt == "count"
                        else f"The lot was sold for {amount(v, nt)} in total.")
                out.append({"item_id": f"A-{nt}-{m}e{e}", "text": text, "notation": nt, "mantissa": m, "exp": e,
                            "log10": round(math.log10(v), 6), "e6_step": "e6 probe"})
    return out


# --- Part B items ----------------------------------------------------------------------------------------------------
def _chat(user):
    return [{"role": "user", "content": user}]


def _choice(it, prefix='{"choice": "'):
    return dict(it, prefix=prefix, readout={"type": "choice", "options": ["A", "B"]})


def risk_items():
    out = []
    for X in (40, 55, 70, 85, 100):
        for r in (2.0, 2.4, 3.0):
            for m in (0, 6):
                x, y = X * 10 ** m, int(round(X * r)) * 10 ** m
                safe, risky = f"Receive {x} dollars for sure.", f"A 50% chance to receive {y} dollars, and otherwise nothing."
                for order in ("safeA", "safeB"):
                    a, b = (safe, risky) if order == "safeA" else (risky, safe)
                    user = f"You are offered a choice between two options.\nA) {a}\nB) {b}\nWhich option do you choose? {JSON_CHOICE}"
                    out.append(_choice({"item_id": f"risk-{X}-{r:g}-{order}-x1e{m}", "family": "risk", "base": f"{X}-{r:g}-{order}",
                                        "scale": m, "sign": 1.0 if m == 0 else -1.0, "safe": "A" if order == "safeA" else "B",
                                        "messages": _chat(user)}))
    return out


PAIRS = ((89, 91), (47, 74), (305, 350), (612, 598), (1200, 980), (77, 707), (4500, 4050), (19, 91), (250, 205), (830, 838))


def compare_items():
    out = []
    for a0, b0 in PAIRS:
        for m in (0, 6):
            for flip in (False, True):
                a, b = (b0, a0) if flip else (a0, b0)
                a, b = a * 10 ** m, b * 10 ** m
                user = f"Which amount is larger?\nA) {a} dollars\nB) {b} dollars\n{JSON_CHOICE}"
                out.append(_choice({"item_id": f"cmp-{a0}-{b0}-{int(flip)}-x1e{m}", "family": "compare", "base": f"{a0}-{b0}-{int(flip)}",
                                    "scale": m, "sign": 1.0 if m == 0 else -1.0, "correct": "A" if a > b else "B",
                                    "messages": _chat(user)}))
    return out


PERCENT_T = (
    ("reward", "You lost a wallet containing {x} dollars, and a stranger returned it to you. What percentage of the money "
               "would you give them as a reward?"),
    ("reserve", "You have a budget of {x} dollars for a project. What percentage of it would you keep aside as a "
                "contingency reserve?"),
    ("donate", "You won {x} dollars in a lottery. What percentage of it would you donate to charity?"))


def percent_items():
    out = []
    for name, t in PERCENT_T:
        for X in (60, 120, 250, 400, 800):
            for m in (0, 6):
                user = t.format(x=X * 10 ** m) + ' Reply with JSON only, in the form {"percent": N}.'
                out.append({"item_id": f"pct-{name}-{X}-x1e{m}", "family": "percent", "base": f"{name}-{X}", "scale": m,
                            "sign": 1.0 if m == 0 else -1.0, "messages": _chat(user), "prefix": '{"percent": ',
                            "readout": {"type": "gen", "max_new": 6}})
    return out


SUMS = ((23, 45), (67, 18), (125, 240), (310, 95), (48, 52), (199, 301), (76, 87), (405, 370), (88, 64), (152, 49),
        (230, 415), (91, 9))


def sum_items():
    out = []
    for a, b in SUMS:
        for m in (0, 6):
            user = (f"What is {a * 10 ** m} dollars plus {b * 10 ** m} dollars? Reply with JSON only, in the form "
                    '{"answer": N}, with N in dollars.')
            out.append({"item_id": f"sum-{a}-{b}-x1e{m}", "family": "sum", "base": f"{a}-{b}", "scale": m,
                        "sign": 1.0 if m == 0 else -1.0, "answer": (a + b) * 10 ** m, "messages": _chat(user),
                        "prefix": '{"answer": ', "readout": {"type": "gen", "max_new": 14}})
    return out


FERMI = ("piano tuners work in Chicago", "people live in Denmark", "grains of rice are in a one-kilogram bag",
         "words are in a typical novel", "trees are in New York's Central Park", "golf balls fit inside a school bus",
         "hairs are on an adult human head", "books are in a large public library", "cars are registered in Los Angeles County",
         "litres of water fill an Olympic swimming pool")


def fermi_items():
    return [{"item_id": f"fermi-{i:02d}", "family": "fermi", "base": f"{i:02d}", "scale": None, "sign": 1.0,
             "messages": _chat(f"Estimate how many {q}. Reply with JSON only, in the form " + '{"estimate": N}.'),
             "prefix": '{"estimate": ', "readout": {"type": "gen", "max_new": 16}} for i, q in enumerate(FERMI)]


def nego_items(fx, n=7):
    """Positive control: Study 3's A0 seller prompts at $100 (N1, +v) and $100M (N6, -v), forced JSON opening."""
    out = []
    for it in fx[:n]:
        for k, m in (("N1", 0), ("N6", 6)):
            out.append({"item_id": f"nego-{it['fixture']}-{k}", "family": "nego", "base": it["fixture"], "scale": m,
                        "sign": 1.0 if m == 0 else -1.0, "fixture": it["fixture"], "base_limit": it["base_limit"],
                        "messages": it["prompts"][k]["messages"], "prefix": '{"action": "OFFER", "price": ',
                        "readout": {"type": "gen", "max_new": 12}})
    return out


def family_items(fx, limit=0):
    fams = {"nego": nego_items(fx), "risk": risk_items(), "compare": compare_items(), "percent": percent_items(),
            "sum": sum_items(), "fermi": fermi_items()}
    if limit:
        fams = {k: v[:max(2, limit)] for k, v in fams.items()}
    return {k: [dict(x, e6_step=f"e6 {k}") for x in v] for k, v in fams.items()}


# --- layers ----------------------------------------------------------------------------------------------------------
def layer_plan(n):
    m = lambda L: X4.map_layer(L, n)                                  # noqa: E731
    return {"n_layers": n, "keep": sorted({m(L) for L in (42, 43, 44)}), "stake": sorted({m(L) for L in STAKE_60}),
            "mlp": sorted({m(L) for L in MLP_60}), "plan": [[v, m(L), a] for v, L, a in PLAN_60], "mapped": n != X4.CANON}


def kws(Ls):
    k = {"build": {"keep": Ls["keep"], "head_layers": [], "role": "seller"},
         "dirs": {"layers": Ls["stake"], "role": "seller"},
         "probe": {"mlp_layers": Ls["mlp"]}, "fit": {"stakes_layers": Ls["stake"]}, "census": {}}
    for f in FAMILIES:
        k[f] = {"plan": [list(x) for x in Ls["plan"]], "n_none": N_NONE, "batch": BATCH, "role": "seller"}
    return k


def estimate_min(n):
    """{step: items} -> (typical, slow) minutes. Choice: one 16-row forward; gen: 16 rows x max_new greedy tokens."""
    per = {"build": 1.6 * X4.BUILD_S / 2, "dirs": 2.0, "probe": 0.25, "fit": 30.0, "census": 20.0, "nego": 3.0, "risk": 0.8,
           "compare": 0.8, "percent": 1.6, "sum": 3.0, "fermi": 3.5}
    t = sum(per[s] * c for s, c in n.items())
    return t / 60, t * X4.SLOW / 60, {f"e6 {s}": v for s, v in per.items()}


# --- setup -----------------------------------------------------------------------------------------------------------
def _sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def predictions_committed(root):
    p = os.path.join(root, PRED_FILE)
    if not os.path.exists(p):
        return f"{PRED_FILE} does not exist"
    r = subprocess.run(["git", "-C", root, "ls-files", "--error-unmatch", PRED_FILE], capture_output=True, text=True)
    if r.returncode != 0:
        return f"{PRED_FILE} is not committed"
    r = subprocess.run(["git", "-C", root, "status", "--porcelain", "--", PRED_FILE], capture_output=True, text=True)
    return f"{PRED_FILE} has uncommitted changes" if r.stdout.strip() else None


def setup(a, settings, schema, li):
    _LI[:] = [li]
    paid = not a.preview and not a.attach
    if paid:
        why = predictions_committed(li.ROOT)
        if why:
            raise SystemExit(f"Not launching E6: {why}. Commit the predictions before any paid run.")
        if a.e6_limit:
            raise SystemExit("Not launching E6: --e6-limit is for local rehearsals only.")
    fx = X5.build_items("seller")
    if a.e6_limit:
        fx = fx[:max(3, a.e6_limit)]
    nonmoney = [v for v in li.build_e3_items(settings, 1)[0] if v["kind"] == "nonmoney"]
    Ls = layer_plan(X4.CANON)
    probes = probe_items(a.e6_limit)
    fams = family_items(fx, a.e6_limit)
    counts = {"build": len(fx) + len(nonmoney), "dirs": 1, "probe": len(probes), "fit": 1, "census": 1,
              **{f: len(v) for f, v in fams.items()}}
    typ, slow, per = estimate_min(counts)
    li.STEP_GUESS_S.update(per)
    ctx = {"li": li, "fx": fx, "nonmoney": nonmoney, "probes": probes, "fams": fams, "layers": Ls, "kw": kws(Ls),
           "estimate_min": [round(typ, 1), round(slow, 1)], "limit": a.e6_limit, "model": a.model,
           "pred_sha256": _sha256(os.path.join(li.ROOT, PRED_FILE)) if os.path.exists(os.path.join(li.ROOT, PRED_FILE)) else None}
    plan = (f"E6 (magnitude as number vs stakes) on {a.model}: A) {len(probes)} amount sentences in {len(NOTATIONS)} notations, "
            f"all-layer probes + MLP census at {Ls['mlp']}; B) the seller scale direction from E5's build ({len(fx)} fixtures) "
            f"collapsed at {Ls['stake']}, steered into " + ", ".join(f"{k} {len(v)}" for k, v in fams.items())
            + f" prompts ({len(Ls['plan'])} steered + {N_NONE} plain rows each). About {typ:.0f} min of experiment time "
            f"({slow:.0f} if slow)" + (f"; REHEARSAL limit {a.e6_limit}" if a.e6_limit else ""))
    inputs = ([{"step": "fixture", **it} for it in fx] + [{"step": "nonmoney", **v} for v in nonmoney]
              + [{"step": "probe", **p} for p in probes] + [{"step": f, **x} for f, v in fams.items() for x in v])
    return {"exp": "E6", "plan": plan, "inputs": inputs, "ctx": ctx, "steps": steps, "live": live, "after_step": after_step}


def steps(ctx):
    kw = ctx["kw"]
    out = [("e6 build", "e6_build", [dict(x, e6_step="e6 build") for x in ctx["nonmoney"] + ctx["fx"]], kw["build"], 5),
           ("e6 dirs", "e6_dirs", [{"item_id": "dirs", "e6_step": "e6 dirs"}], kw["dirs"], 1),
           ("e6 probe", "e6_probe", ctx["probes"], kw["probe"], 40),
           ("e6 fit", "e6_fit", [{"item_id": "fit", "e6_step": "e6 fit"}], kw["fit"], 1),
           ("e6 census", "e6_census", [{"item_id": "census", "e6_step": "e6 census"}], kw["census"], 1)]
    for f in FAMILIES:
        out.append((f"e6 {f}", "e6_steer", ctx["fams"][f], kw[f], 8 if f in ("risk", "compare") else 4))
    return out


def _relayer(ctx, n):
    Ls = layer_plan(n)
    ctx["layers"] = Ls
    new = kws(Ls)
    for k, v in new.items():                           # in place: run_e0 holds references to these dicts
        ctx["kw"][k].clear()
        ctx["kw"][k].update(v)


def on_pod_info(ctx, pod_info):
    n = pod_info.get("num_hidden_layers")
    if n and int(n) != ctx["layers"]["n_layers"]:
        _relayer(ctx, int(n))


def after_step(label, got, ctx):
    note = (lambda s: ctx["panel"].note(s)) if ctx.get("panel") is not None else (lambda s: None)
    if label == "e6 build":
        recs = [r for r in got.get(label) or [] if r.get("kind") == "fixture"]
        use = [r for r in recs if not r.get("skipped")]
        if len(use) < min(10, len(recs)):
            return f"only {len(use)} of {len(recs)} fixtures have two distinct A0 greedy prices"
        n = use[0]["n_layers"]
        if n != ctx["layers"]["n_layers"]:
            _relayer(ctx, n)
        note(f"build: {len(use)}/{len(recs)} usable" + (" (stand-in prices: rehearsal)" if any(r.get("stand_in") for r in use) else ""))
    if label == "e6 dirs":
        d = got.get(label) or []
        note("dirs: " + "; ".join(f"L{r['layer']} |v| {r['norm']:.1f}, reply coherence {r['coherence_reply']:.2f}, "
                                  f"cos(vR, u) {r['cos_vR_u']}" for r in d))
    return None


# --- analysis --------------------------------------------------------------------------------------------------------
def rkey(x):
    return "none" if x["vector"] == "none" else f"{x['vector']}@{x['layer']}x{x['alpha']:+g}"


def metric(rec, x, li):
    """The family's scalar per row: risk safe log-odds, compare correct-answer margin, percent value, sum exact match,
    fermi log10 estimate, nego seller opening gap."""
    f = rec["family"]
    if f in ("risk", "compare"):
        a, b = x["lp"]
        want = rec["want"]
        return (a - b) if want == "A" else (b - a)
    v = x.get("value")
    if f == "percent":
        return v if v is not None and 0 <= v <= 100 else None
    if f == "sum":
        return None if v is None else float(v == rec["answer"])
    if f == "fermi":
        return math.log10(v) if v and v > 0 else None
    if f == "nego":
        g, m, off = li.scale_free("seller", int(v) if v else None, rec["base_limit"])
        return g if not off and m == rec["scale"] else None
    return None


def family_summary(recs, items, li):
    """Natural gap (x1e6 - x1, plain rows), per-row-key mean change from plain on x1 and on x1e6 items, shares."""
    meta = {it["item_id"]: it for it in items}
    vals = {}
    for r in recs:
        it = meta.get(r["item_id"], {})
        r2 = dict(r, want=it.get("safe") or it.get("correct"), answer=it.get("answer"), base_limit=it.get("base_limit"),
                  base=it.get("base"), scale=it.get("scale"))
        per = {}
        for x in r["rows"]:
            v = metric(r2, x, li)
            if v is not None:
                per.setdefault(rkey(x), []).append(v)
        vals[r["item_id"]] = {"base": r2["base"], "scale": r2["scale"], "m": {k: st.mean(v) for k, v in per.items()}}
    scales = sorted({v["scale"] for v in vals.values()}, key=lambda s: -1 if s is None else s)
    out = {"n_items": len(vals), "plain": {}, "delta": {}, "share": {}}
    for s in scales:
        p = [v["m"]["none"] for v in vals.values() if v["scale"] == s and "none" in v["m"]]
        out["plain"][str(s)] = st.mean(p) if p else None
    g = None
    if "0" in out["plain"] and "6" in out["plain"]:
        pairs = {}
        for v in vals.values():
            if "none" in v["m"]:
                pairs.setdefault(v["base"], {})[v["scale"]] = v["m"]["none"]
        d = [x[6] - x[0] for x in pairs.values() if 0 in x and 6 in x]
        g = st.mean(d) if d else None
    out["natural_gap"] = g
    keys = sorted({k for v in vals.values() for k in v["m"] if k != "none"})
    for s in scales:
        for k in keys:
            d = [v["m"][k] - v["m"]["none"] for v in vals.values() if v["scale"] == s and k in v["m"] and "none" in v["m"]]
            if d:
                out["delta"].setdefault(str(s), {})[k] = st.mean(d)
                if g:
                    out["share"].setdefault(str(s), {})[k] = st.mean(d) / (g if s == 0 else -g)
    return out


def fit_summary(fit):
    if not fit:
        return None
    L = fit["layers"]
    tr = lambda a, b: [r["transfer"][a][b] for r in L]                    # noqa: E731
    out = {"notations": fit["notations"], "n": fit["n"], "within": {k: [r["within"][k] for r in L] for k in fit["notations"]}}
    out["transfer_from_digits"] = {b: tr("digits", b) for b in fit["notations"] if b != "digits"}
    out["cos_digits_vR"] = {r["layer"]: r.get("cos_digits_vR") for r in L if "cos_digits_vR" in r}
    out["vR_proj_r"] = {r["layer"]: r.get("vR_proj_r") for r in L if "vR_proj_r" in r}
    return out


def summarize(got, ctx):
    li = ctx.get("li") or _li()
    g = lambda s: got.get(f"e6 {s}") or []                               # noqa: E731
    b = [r for r in g("build") if r.get("kind") == "fixture"]
    out = {"layers": ctx["layers"], "estimate_min": ctx.get("estimate_min"), "predictions_sha256": ctx.get("pred_sha256"),
           "limit": ctx.get("limit"),
           "build": {"n": len(b), "usable": sum(1 for r in b if not r.get("skipped")), "stand_in": any(r.get("stand_in") for r in b)},
           "dirs": g("dirs"), "probe": {"n": len(g("probe")), "last_tokens": sorted({r["last_token"] for r in g("probe")}),
                                        "n_tokens": [min((r["n_tokens"] for r in g("probe")), default=None),
                                                     max((r["n_tokens"] for r in g("probe")), default=None)]},
           "fit": fit_summary((g("fit") or [None])[0]), "census": (g("census") or [None])[0], "families": {}}
    for f in FAMILIES:
        out["families"][f] = family_summary(g(f), ctx["fams"][f], li)
    st_recs = [r for f in FAMILIES for r in g(f)]
    out["checks"] = {"hooks_fired_once": all(all(n == 1 for n in r["hooks_fired"]) for r in st_recs),
                     "loo": sorted({x for r in g("nego") for x in r["loo"]}),
                     "choice_mass_min": min((x["mass"] for f in ("risk", "compare") for r in g(f) for x in r["rows"]), default=None),
                     "n_steer_tokens": sorted({r["n_steer"] for r in st_recs})}
    out["secs_per_item"] = {lab: st.mean(r["secs"] for r in got[lab] if "secs" in r) if got.get(lab) and
                            any("secs" in r for r in got[lab]) else None for lab, *_ in steps(ctx)}
    out["predictions"] = evaluate(out)
    return out


# --- the predictions (config/interp/e6_predictions.md) ---------------------------------------------------------------
def _band(n, lo, hi):
    return {X4.map_layer(L, n) for L in range(lo, hi + 1)}


def evaluate(s):
    n = (s.get("layers") or {}).get("n_layers", X4.CANON)
    L42, L44 = X4.map_layer(42, n), X4.map_layer(44, n)
    k2, k2n = f"vR@{L42}x+2", f"vR@{L42}x-2"
    out = []

    def add(pid, claim, cond, value):
        out.append({"id": pid, "claim": claim, "result": "n.a." if cond is None else ("HIT" if cond else "MISS"), "value": value})
    F = s.get("families") or {}

    def sh(f, k, scale="0"):
        return ((F.get(f) or {}).get("share") or {}).get(scale, {}).get(k)

    def de(f, k, scale="0"):
        return ((F.get(f) or {}).get("delta") or {}).get(scale, {}).get(k)
    a, r_ = sh("nego", k2), sh("nego", f"rand@{L42}x+2")
    add("E6-1", "positive control: vR at L42 (alpha +2) moves $100 seller openings >= 40% of the natural $100 -> $100M gap, "
        "and the random vector <= 10%", None if a is None or r_ is None else (a >= 0.40 and abs(r_) <= 0.10),
        None if a is None else {"vR": round(a, 3), "rand": None if r_ is None else round(r_, 3)})
    fit = s.get("fit") or {}
    T = fit.get("transfer_from_digits") or {}
    mid = sorted(L for L in _band(n, 10, 40) if L < len(T.get("words") or []))
    if T.get("words") and T.get("short") and mid:
        best = max(min(T["words"][L], T["short"][L]) for L in mid)
        early = max(T["words"][L] for L in sorted(_band(n, 0, 2)))
        peak = max(T["words"][L] for L in mid)
    else:
        best = early = peak = None
    add("E6-2", "number: at some layer in 10-40, the digits-derived magnitude direction transfers to words and to short "
        "notation with r >= 0.8 (both)", None if best is None else best >= 0.8, None if best is None else round(best, 3))
    add("E6-3", "number: transfer to words at layers 0-2 is at least 0.2 below its peak in 10-40 (computed, not lexical)",
        None if peak is None else peak - early >= 0.2, None if peak is None else {"early": round(early, 3), "peak": round(peak, 3)})
    c = T.get("count")
    cb = max(c[L] for L in mid) if c and mid else None
    add("E6-4", "number: at some layer in 10-40, the money (digits) direction transfers to non-money counts with r >= 0.8",
        None if cb is None else cb >= 0.8, None if cb is None else round(cb, 3))
    g = (F.get("risk") or {}).get("natural_gap")
    add("E6-5", "stakes, natural: plain risk choices are safer at $100M than at $100 by >= 0.5 nats (safe-option log-odds)",
        None if g is None else g >= 0.5, None if g is None else round(g, 3))

    def transfer(fam, minimum):
        gap = (F.get(fam) or {}).get("natural_gap")
        if gap is None or abs(gap) < minimum:
            return None, {"natural_gap": gap}
        v, cu, cr = sh(fam, k2), sh(fam, f"u@{L42}x+2"), sh(fam, f"rand@{L42}x+2")
        if v is None:
            return None, {"natural_gap": gap}
        ctl = max(abs(cu or 0), abs(cr or 0))
        return v >= 0.25 and abs(v) > 3 * ctl, {"natural_gap": round(gap, 3), "vR": round(v, 3), "u": cu and round(cu, 3),
                                               "rand": cr and round(cr, 3)}
    ok, val = transfer("risk", 0.5)
    add("E6-6", "stakes transfer to risk: vR at L42 (alpha +2) on $100 choices moves >= 25% of the natural gap toward $100M, "
        "> 3x the u and random controls (n.a. if the natural gap < 0.5 nats)", ok, val)
    ok, val = transfer("percent", 1.0)
    add("E6-7", "stakes transfer to scale-free percentages: the same, on $100 percentage answers (n.a. if the natural gap "
        "< 1 point)", ok, val)
    acc = []
    for fam in ("compare", "sum"):
        Fm = F.get(fam) or {}
        for sc in ("0", "6"):
            d = (Fm.get("delta") or {}).get(sc, {}).get(k2)
            if fam == "compare":
                base = (Fm.get("plain") or {}).get(sc)
                acc.append(None if d is None or not base else abs(d) <= 0.10 * abs(base))
            else:
                acc.append(None if d is None else abs(d) <= 0.05)
    add("E6-8", "number intact: under vR at L42 (alpha +2), comparison margins move <= 10% of their plain value and sum "
        "exact-match moves <= 5 points, at both scales", None if None in acc else all(acc), acc)
    fp, fn = de("fermi", k2, "None"), de("fermi", k2n, "None")
    add("E6-9", "no number-size leak: Fermi estimates move <= 0.15 in log10 under vR at L42, alpha +2 and -2",
        None if fp is None or fn is None else max(abs(fp), abs(fn)) <= 0.15,
        None if fp is None else {"+2": round(fp, 3), "-2": None if fn is None else round(fn, 3)})
    cd = (fit.get("cos_digits_vR") or {}).get(L42)
    add("E6-10", "geometry: |cos| between the digits magnitude direction and vR at L42 <= 0.2",
        None if cd is None else abs(cd) <= 0.2, cd)
    return out


def manifest(ctx):
    return {"name": "e6", "plan": "config/interp/e6_predictions.md (E6, number vs stakes)", "layers": ctx["layers"],
            "predictions_file": PRED_FILE, "predictions_sha256": ctx.get("pred_sha256"), "limit": ctx.get("limit"),
            "fixtures": [it["fixture"] for it in ctx["fx"]], "families": {k: len(v) for k, v in ctx["fams"].items()},
            "notations": list(NOTATIONS), "mantissas": list(MANT), "exponents": list(EXPS), "estimate_min": ctx.get("estimate_min")}


# --- panel, preview, report ------------------------------------------------------------------------------------------
def live(got):
    out = []
    fit = (got.get("e6 fit") or [None])[0]
    if fit:
        T = fit_summary(fit)["transfer_from_digits"]
        n = len(T["words"])
        out.append("fit digits->words r by layer: " + " ".join(f"L{L}:{T['words'][L]:.2f}" for L in range(0, n, max(1, n // 8))))
    for f in FAMILIES:
        recs = got.get(f"e6 {f}") or []
        if recs:
            out.append(f"{f}: {len(recs)} items")
    return out[-4:]


def stub(task, items, ctx):
    """Preview only: canned shapes, not a model."""
    if not items:
        return []
    Ls, n = ctx["layers"], ctx["layers"]["n_layers"]
    if task == "e6_build":
        b = X4.stub("e4_build", [dict(x) for x in items], {"layers": X4.layer_sets(n), "n_heads": 32, "fx": ctx["fx"],
                                                            "nonmoney": ctx["nonmoney"]})
        return b
    if task == "e6_dirs":
        return [{"item_id": f"dirs-L{L}", "layer": L, "norm": 40.0, "n_fixtures": len(ctx["fx"]), "coherence_reply": 0.8,
                 "coherence_S": 0.6, "cos_vR_vS": 0.5, "cos_vR_u": 0.1, "cos_vR_rand": 0.0, "segs": {}} for L in Ls["stake"]]
    if task == "e6_probe":
        return [{"item_id": it["item_id"], "n_tokens": 12, "last_token": ".", "secs": 0.01} for it in items]
    if task == "e6_fit":
        nots = list(NOTATIONS)
        lay = []
        for L in range(n):
            q = min(1.0, 0.3 + L / 15)
            tr = {a: {b: round(0.99 if a == b else q * (0.95 if b != "count" else 0.85), 3) for b in nots} for a in nots}
            rec = {"layer": L, "within": {k: 0.97 for k in nots}, "transfer": tr, "cos": tr}
            if L in Ls["stake"]:
                rec.update(cos_digits_vR=0.05, vR_proj_r={k: 0.1 for k in nots})
            lay.append(rec)
        return [{"item_id": "fit", "notations": nots, "n": {k: 72 for k in nots}, "layers": lay}]
    if task == "e6_census":
        return [{"item_id": "census", "notations": list(NOTATIONS), "exps": list(EXPS), "dial": 0.8, "weak": 0.3,
                 "layers": {str(L): {"d_mlp": 21504, "n_dial": 12, "n_dial_count_too": 7, "n_digit_only": 40, "n_tuned": 25,
                                     "tuned_by_exp": {str(e): 3 for e in EXPS}, "top": []} for L in Ls["mlp"]}}]
    out = []
    for it in items:
        rng, f, s = random.Random(it["item_id"]), it["family"], it.get("scale")
        rows = []
        spec = [None] * N_NONE + [tuple(x) for x in ctx["kw"][f]["plan"]]
        for x in spec:
            push = (x[2] if x and x[0] in ("vR", "vS") else 0.0) * (0.4 if x and x[0] == "vR" else 0.15)
            if f in ("risk", "compare"):
                base = (1.0 + (0.8 if s == 6 else 0.0)) if f == "risk" else 6.0
                if f == "risk":
                    base += push * 0.8 * (1 if s == 0 else 1)
                sv = base + rng.uniform(-0.05, 0.05)
                want = it.get("safe") or it.get("correct")
                lp = [-0.2, -0.2 - sv] if want == "A" else [-0.2 - sv, -0.2]
                rows.append({"lp": lp, "mass": 0.99})
            else:
                if f == "percent":
                    v = 10.0 - (3.0 if s == 6 else 0.0) - 3.0 * push * 0.8
                elif f == "sum":
                    v = it["answer"]
                elif f == "fermi":
                    v = 10 ** (4 + int(it["base"]) % 3)
                else:
                    g = (0.40 if s == 0 else 0.22) - 0.18 * push
                    v = round(it["base_limit"] * (1 + g)) * 10 ** s
                rows.append({"text": str(v), "value": float(v)})
        rows = [dict(r, vector=x[0] if x else "none", layer=x[1] if x else None, alpha=float(x[2]) if x else 0.0)
                for x, r in zip(spec, rows)]
        out.append({"item_id": it["item_id"], "family": f, "scale": s, "sign": it["sign"], "fixture": it.get("fixture"),
                    "n_steer": 14, "n_tokens": 90, "hooks_fired": [1, 1], "loo": ["True"] if f == "nego" else ["None"],
                    "rows": rows, "secs": 0.1})
    return out


def _f(v, d=3):
    return "—" if v is None else (f"{v:+.{d}f}" if isinstance(v, (int, float)) else str(v))


def report(s, preview):
    print("E6, magnitude as a number vs as stakes (exploratory; predictions in " + PRED_FILE
          + (", PREVIEW: canned shapes" if preview else "") + ")")
    Ls = s.get("layers") or {}
    if Ls.get("mapped"):
        print(f"  NOTE: a {Ls['n_layers']}-layer model; the 31B's layers were mapped onto it")
    B = s.get("build") or {}
    print(f"  build: {B.get('usable')}/{B.get('n')} usable" + (" (STAND-IN prices: rehearsal)" if B.get("stand_in") else ""))
    for d in s.get("dirs") or []:
        print(f"  direction L{d['layer']}: |vR| {d['norm']:.2f}; reply coherence {d['coherence_reply']:.2f}; cos(vR, vS) "
              f"{d['cos_vR_vS']}, cos(vR, u) {d['cos_vR_u']}")
    P = s.get("probe") or {}
    print(f"  probe: {P.get('n')} sentences; last tokens {P.get('last_tokens')}; lengths {P.get('n_tokens')}")
    fit = s.get("fit") or {}
    if fit:
        n = len(next(iter(fit["within"].values())))
        pick = sorted({X4.map_layer(L, n) for L in (0, 1, 2, 5, 10, 15, 20, 30, 41, 50, 59)} & set(range(n)))
        print("  within-notation held-out r: " + "; ".join(f"{k} " + " ".join(f"L{L}:{fit['within'][k][L]:.2f}" for L in pick[::2])
                                                          for k in fit["notations"]))
        for b, vals in fit["transfer_from_digits"].items():
            print(f"  digits -> {b:<6} r: " + " ".join(f"L{L}:{vals[L]:.2f}" for L in pick))
        print("  cos(digits direction, vR): " + ", ".join(f"L{L} {v}" for L, v in fit["cos_digits_vR"].items()))
    C = (s.get("census") or {}).get("layers") or {}
    for L, c in C.items():
        print(f"  census L{L}: d_mlp {c['d_mlp']}; dials (all money notations) {c['n_dial']} (+count {c['n_dial_count_too']}); "
              f"digit-only {c['n_digit_only']}; tuned {c['n_tuned']} by exponent {c['tuned_by_exp']}")
        for t in c["top"][:3]:
            print(f"    neuron {t['neuron']}: rho " + ", ".join(f"{k} {v:+.2f}" for k, v in t["rho"].items()))
    for f, Fm in (s.get("families") or {}).items():
        print(f"  {f}: {Fm['n_items']} items; plain " + ", ".join(f"x1e{k}: {_f(v)}" for k, v in Fm["plain"].items())
              + f"; natural gap {_f(Fm['natural_gap'])}")
        for sc, d in (Fm.get("delta") or {}).items():
            sh = (Fm.get("share") or {}).get(sc, {})
            print(f"    on x1e{sc}: " + "  ".join(f"{k} {_f(v, 2)}" + (f" ({sh[k]:+.0%})" if k in sh else "") for k, v in sorted(d.items())))
    print("\n  predictions (" + PRED_FILE + (f", sha256 {s['predictions_sha256'][:12]}" if s.get("predictions_sha256") else "") + "):")
    for p in s.get("predictions") or []:
        print(f"    {p['id']:<6} {p['result']:<5} {p['claim']}\n           value: {p['value']}")
    ch = s.get("checks") or {}
    print(f"  checks: hooks fired once per steered layer {ch.get('hooks_fired_once')}; nego leave-one-out {ch.get('loo')}; "
          f"min option mass {ch.get('choice_mass_min')}; steered tokens per prompt {ch.get('n_steer_tokens')}")
    print("  seconds per item: " + ", ".join(f"{k[3:]} {v:.1f}" for k, v in (s.get("secs_per_item") or {}).items() if v))
