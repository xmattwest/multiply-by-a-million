"""tasks_e4.py (POD): E4, the cure (STORY X23, X57). Details: scripts/interp_e4_laptop.py. Exploratory.
Four prompts per fixture: N1/N6 (D2/D8) and P1/P6 (the same + Q17's range sentence after the goal). Aligned positions
A = H (after the limit, before the range line) + S (from "\\n\\nNo messages") + reply (the forced opening); the same
tokens in all four (checked). Ops of e4_patch rows, on a segment of A: ["set", kind, L, h, seg, src],
["add", kind, L, h, seg, alpha, b, a] (+= alpha * (b - a)), ["vec", L, seg, alpha, vN|u|rand]; kind resid/attn/head.
"""
import copy
import math
import os
import time

import torch                                   # pod-only

import hooks
import tasks as T
from tasks_heads import _decision, _fwd, _geometry    # stateless helpers (that module's own state is not used)

PROMPTS = ("N1", "N6", "P1", "P6")
SCALE = {"N1": "x1", "N6": "x1e6", "P1": "x1", "P6": "x1e6"}
_STORE = {"pairs": {}, "d": {}, "acc": {}, "n": 0, "u": None, "u_n": 0, "keep": None, "segs": None}


def _span(offs, c0, c1):
    sp = [k for k, (s, e) in enumerate(offs) if s < c1 and e > c0]
    if not sp or not T.tokens_cover(offs, sp, c0, c1):
        raise ValueError("digits do not map onto whole tokens")
    return sp


def _start(offs, c):
    k = next((i for i, (s, e) in enumerate(offs) if s == c and e > s), None)
    if not k or offs[k - 1][1] != c:
        raise ValueError("a segment boundary falls inside a token")
    return k


def encode(tok, model_io, side, role):
    text, _ = model_io.render_messages(tok, side["messages"])
    ch = T._chat_encode(tok, model_io, side["messages"], side["limit_text"], role)
    full = text + T._PREFIX
    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = list(enc["input_ids"]), [tuple(o) for o in enc["offset_mapping"]]
    if ids[:len(ch["ids"])] != ch["ids"] or full.count("\n\nNo messages") != 1:
        raise ValueError("unexpected prompt layout")
    span = T._locate(full, offs, side["limit_text"], role)
    k_s = _start(offs, full.index("\n\nNo messages"))
    e = {"ids": ids, "span": span, "s0": span[0], "n_chat": len(ch["ids"]), "lo": [], "hi": [], "R": []}
    k_g, rg = k_s, side.get("range")
    if rg:
        sent = f"from the range {rg['lo']} dollars to {rg['hi']} dollars."
        if full.count(sent) != 1:
            raise ValueError("range sentence not found exactly once")
        lo0 = full.index(sent) + len("from the range ")
        hi0 = lo0 + len(rg["lo"]) + len(" dollars to ")
        e["lo"], e["hi"] = _span(offs, lo0, lo0 + len(rg["lo"])), _span(offs, hi0, hi0 + len(rg["hi"]))
        k_g = _start(offs, full.rindex("\n", 0, lo0))
        if not span[-1] < k_g < k_s:
            raise ValueError("the range line is not between the limit and 'No messages'")
        e["R"] = list(range(k_g, k_s))
    H = list(range(span[-1] + 1, k_g))
    e.update(A=H + list(range(k_s, len(ids))), n_H=len(H), n_S=len(ch["ids"]) - k_s, n_reply=len(ids) - len(ch["ids"]))
    if not H or e["n_S"] <= 0:
        raise ValueError("empty segment")
    return e


def segs(e):
    h, s, n = e["n_H"], e["n_S"], len(e["A"])
    return {"A": (0, n), "H": (0, h), "S": (h, h + s), "reply": (h + s, n), "Srep": (h, n), "prompt": (0, h + s)}


def encode_fixture(tok, model_io, it):
    E = {k: encode(tok, model_io, it["prompts"][k], it["role"]) for k in it["prompts"]}
    r = next(iter(E.values()))
    key = lambda e: (e["s0"], e["ids"][:e["s0"]], [e["ids"][p] for p in e["A"]], e["n_H"], e["n_S"], e["n_reply"])  # noqa: E731
    if any(key(e) != key(r) for e in E.values()):
        raise ValueError(f"{it['fixture']}: prompts differ outside the limit and the range sentence")
    return E


def _capture(model, model_io, blocks, ids, pos, need, extra=(), k=1):
    """One unbatched forward of ids + extra; values at `pos` for each (kind, layer) in need; and the kept logits."""
    vals, P, hs = {}, torch.tensor(pos, device=model.device), []
    put = lambda key, x: vals.__setitem__(key, x[0, P].detach())                  # noqa: E731
    for kind, L in need:
        if kind == "resid":
            hs.append(blocks[L].register_forward_hook(lambda _m, _i, o, L=L: put(("resid", L), hooks._block_output_residual(o))))
        elif kind == "attn":
            hs.append(blocks[L].self_attn.register_forward_hook(lambda _m, _i, o, L=L: put(("attn", L), o[0])))
        else:
            hs.append(blocks[L].self_attn.o_proj.register_forward_pre_hook(lambda _m, a, L=L: put(("head", L), a[0])))
    try:
        o = _fwd(model, model_io, k, input_ids=torch.tensor([list(ids) + list(extra)], device=model.device), use_cache=False)
    finally:
        for h in hs:
            h.remove()
    return vals, o.logits[0].float()


def vectors(fixture, L, loo=True):
    """Plain direction at L over A, leave-one-fixture-out (vN); non-money at its mean size (u); random (rand)."""
    D, i = _STORE["d"], _STORE["keep"].index(L)
    tot = sum(d[i].float() for d in D.values())
    use = loo and len(D) >= 3 and fixture in D
    v = (tot - D[fixture][i].float()) / (len(D) - 1) if use else tot / len(D)
    size = v.norm(dim=-1).mean()
    out = {"vN": v}
    if _STORE["u_n"]:
        u = _STORE["u"][L]
        out["u"] = (u / u.norm() * size).expand_as(v)
    r = torch.randn(tuple(v.shape), generator=torch.Generator(device="cpu").manual_seed(12345 + L)).to(v.device)
    out["rand"] = r / r.norm(dim=-1, keepdim=True) * v.norm(dim=-1, keepdim=True)
    return out, use


def layout(plan, n_none, batch):
    """Steered rows (vector, layer, alpha, sample) sorted by alpha, layer, vector over the fewest batches that also
    hold >= n_none unsteered rows (None); each batch topped up with unsteered rows (E3c's layout)."""
    st_ = sorted(((str(v), int(L), float(a), s) for v, L, a, n in plan for s in range(int(n))), key=lambda x: (x[2], x[1], x[0], x[3]))
    nb = max(1, math.ceil((len(st_) + n_none) / batch))
    q, x = divmod(len(st_), nb)
    out, i = [], 0
    for k in range(nb):
        part = st_[i:i + q + (k < x)]
        i += len(part)
        if len(part) > batch:
            raise ValueError("too many steered rows for the batch size")
        out.append([None] * (batch - len(part)) + part)
    return out


def e4_steer(model, tok, items, settings, model_name, model_io, plan=(), n_none=8, batch=16, schema=None, max_new=192,
             loo=True, greedy=False, **kw):
    """Real sampled openings on it["target"]; steered rows add sign * alpha * vector on A's prompt tokens (prefill) and
    on the reply's opening tokens as generated (hooks.GenAdder). Seed = the item's seed + the batch's first row."""
    import game_engine as ge
    blocks = model_io.decoder_layers(model)
    samp, max_chars, eos = settings["sampling"][model_name], settings["max_message_chars"]["value"], T._eos_ids(model, tok)
    vocab = model.get_output_embeddings().weight.shape[0]
    batches, batch = layout(plan, int(n_none), int(batch)), int(batch)
    out = []
    for it in items:
        t0 = time.time()
        e = encode(tok, model_io, it["prompts"][it["target"]], it["role"])
        chat, s0, nP = e["ids"][:e["n_chat"]], e["s0"], e["n_H"] + e["n_S"]
        rel, reply = torch.tensor([p - s0 for p in e["A"][:nP]]), e["ids"][e["n_chat"]:]
        V = {L: vectors(it["fixture"], L, loo) for L in sorted({x[1] for b in batches for x in b if x})}
        base, how = T._batched_prefix_cache(model, model_io, chat[:s0], batch)
        full = torch.tensor([chat], device=model.device).expand(batch, -1).contiguous()
        rows, calls, gram, n0 = [], [], None, 0
        for k, chunk in enumerate(batches):
            ads = []
            for L in sorted({x[1] for x in chunk if x}):
                d = torch.zeros((batch,) + tuple(V[L][0]["vN"].shape), device=model.device)
                for r, x in enumerate(chunk):
                    if x and x[1] == L:
                        d[r] = float(it.get("sign", 1.0)) * x[2] * V[L][0][x[0]]
                ads.append(hooks.GenAdder(blocks[L], rel, d[:, :nP], len(chat) - s0, d[:, nP:], torch.zeros_like(d[:, 0])))
            lp, gram = (None, "no schema") if schema is None else T._grammar_processor(tok, schema, vocab)
            seed = int(it["seed"]) + k * batch
            torch.manual_seed(seed)
            kwa = {"max_new_tokens": int(max_new), "attention_mask": torch.ones_like(full), "past_key_values": copy.deepcopy(base),
                   **({"do_sample": False, "temperature": None, "top_p": None, "top_k": None} if greedy else
                      {"do_sample": True, "temperature": samp["temperature"], "top_p": samp["top_p"], "top_k": samp["top_k"]})}
            if lp is not None:
                kwa["logits_processor"] = [lp]
            try:
                with torch.no_grad(), model_io.attn_ctx():
                    gen = model.generate(full, **kwa)
            finally:
                for ad in ads:
                    ad.remove()
            calls.append([ads[0].calls[0] if ads and ads[0].calls else None, len(ads[0].calls) if ads else 0, len(ads)])
            for r, x in enumerate(chunk):
                new = gen[r, full.shape[1]:].tolist()
                cut = next((i for i, t in enumerate(new) if t in eos), None)
                new = new[:cut] if cut is not None else new
                content = tok.decode(new, skip_special_tokens=True)
                move, errors = ge.validate_reply(content, "stop" if cut is not None else "length", False, max_chars)
                if x is None:
                    x, n0 = ("none", None, 0.0, n0), n0 + 1
                rows.append({"vector": x[0], "layer": x[1], "alpha": x[2], "sample": x[3], "seed": seed, "aligned": new[:len(reply)] == reply,
                             "finish": "stop" if cut is not None else "length", "completion_tokens": len(new), "errors": errors,
                             "action": move and move["action"], "price": move and move.get("price"), "content": content[:300]})
            del gen
        del base
        out.append({"item_id": it["item_id"], "fixture": it["fixture"], "role": it["role"], "target": it["target"],
                    "sign": it.get("sign", 1.0), "base_limit": it["base_limit"], "loo": all(v[1] for v in V.values()) if V else None,
                    "n_R": len(e["R"]), "prefix_cache": how, "hook_calls": calls, "grammar": gram, "greedy": bool(greedy),
                    "rehearsal": os.environ.get("SI_INTERP_REHEARSAL") == "1", "rows": rows, "secs": round(time.time() - t0, 1)})
    return out


def e4_build(model, tok, items, settings, model_name, model_io, keep=(42, 43, 44), head_layers=tuple(range(38, 45)), **kw):
    """Greedy R2 prices of all four prompts; ld with the plain prices; per-layer dN = N6 - N1, dP = P6 - P1, c6 = P6 - N6,
    c1 = P1 - N1 on A (sums kept for e4_stats; dN per fixture at `keep`); per-head differences; non-money contrast."""
    blocks = model_io.decoder_layers(model)
    nL, (nH, hd) = len(blocks), _geometry(blocks)
    keep, hl = sorted(int(x) for x in keep), sorted(int(x) for x in head_layers)
    if min(keep + hl) < 0 or max(keep + hl) >= nL or _STORE["keep"] not in (None, keep):
        raise ValueError("bad layers")
    _STORE["keep"] = keep
    digits = T._digit_ids(tok)
    r4 = lambda t: [round(float(x), 4) for x in t]                       # noqa: E731
    out = []
    for it in items:
        t0 = time.time()
        if it.get("kind") == "nonmoney":
            ea, eb = T._plain_encode(tok, it["small"], it["small_num"]), T._plain_encode(tok, it["big"], it["big_num"])
            pa, pb = list(range(ea["digit"] + 1, len(ea["ids"]))), list(range(eb["digit"] + 1, len(eb["ids"])))
            if len(pa) != len(pb) or not pa:
                raise ValueError(f"{it['item_id']}: text after the number differs")
            u = T._capture_all_layers(model, model_io, eb["ids"], pb, device_keep=True).float().mean(1) \
                - T._capture_all_layers(model, model_io, ea["ids"], pa, device_keep=True).float().mean(1)
            _STORE["u"] = u if _STORE["u"] is None else _STORE["u"] + u
            _STORE["u_n"] += 1
            out.append({"item_id": it["item_id"], "kind": "nonmoney", "secs": round(time.time() - t0, 2)})
            continue
        f, E = it["fixture"], encode_fixture(tok, model_io, it)
        e = E["N1"]
        rec = {"item_id": it["item_id"], "kind": "fixture", "fixture": f, "base_limit": it["base_limit"], "n_layers": nL,
               "n_heads": nH, "keep": keep, "head_layers": hl, "n_A": len(e["A"]), "n_H": e["n_H"], "n_S": e["n_S"],
               "n_reply": e["n_reply"], "n_R": {k: len(E[k]["R"]) for k in PROMPTS}, "n_tokens": {k: len(E[k]["ids"]) for k in PROMPTS}}
        pr = {k: T.teacher(model, tok, [{"item_id": k, "messages": it["prompts"][k]["messages"], "role": it["role"],
                                          "condition": it["prompts"][k]["condition"], "limit_units": it["prompts"][k]["limit_units"]}],
                           settings, model_name, model_io)[0] for k in PROMPTS}
        rec.update(price={k: pr[k]["price_r2"] for k in PROMPTS}, cont={k: pr[k]["continuation"] for k in PROMPTS}, stand_in=False)
        a, b = rec["price"]["N1"], rec["price"]["N6"]
        why = None if a and b else "no greedy price on a plain prompt"
        why = why or ("the N6 price is not whole millions" if b % 10 ** 6 else "the plain prices are equal" if a * 10 ** 6 == b else None)
        if why and os.environ.get("SI_INTERP_REHEARSAL") == "1":       # local rehearsal: random weights write no prices
            a, b, why, rec["stand_in"] = round(1.3 * it["base_limit"]), round(1.2 * it["base_limit"]) * 10 ** 6, None, True
        if why:
            out.append(dict(rec, skipped=why, secs=round(time.time() - t0, 1)))
            continue
        cand = {"x1": _decision(tok, a, b // 10 ** 6), "x1e6": _decision(tok, a * 10 ** 6, b)}
        need = [("resid", L) for L in range(nL)] + [("head", L) for L in hl]
        X, Hd, ld, lpa, lpb, p1 = {}, {}, {}, {}, {}, {}
        for k in PROMPTS:
            pre, ta, tb = cand[SCALE[k]]
            vals, lg = _capture(model, model_io, blocks, E[k]["ids"], E[k]["A"], need, pre, len(pre) + 1)
            lp, p0 = torch.log_softmax(lg[-1], -1), torch.softmax(lg[0], -1)
            ld[k], lpa[k], lpb[k] = round(float(lp[tb] - lp[ta]), 5), round(float(lp[ta]), 4), round(float(lp[tb]), 4)
            p1[k] = r4(p0[digits]) if None not in digits else None
            X[k] = torch.stack([vals[("resid", L)] for L in range(nL)]).float()          # (L, A, H)
            Hd[k] = {L: vals[("head", L)].float().view(len(e["A"]), nH, hd[L]) for L in hl}
        D = {"dN": X["N6"] - X["N1"], "dP": X["P6"] - X["P1"], "c6": X["P6"] - X["N6"], "c1": X["P1"] - X["N1"]}
        for name, t in D.items():
            _STORE["acc"][name] = t if name not in _STORE["acc"] else _STORE["acc"][name] + t
        _STORE["n"] += 1
        _STORE["d"][f] = D["dN"][keep].to(next(model.parameters()).dtype)
        sg = _STORE["segs"] = segs(e)
        dot = lambda x, y: (x * y).sum((1, 2))                                   # noqa: E731
        lay = {}
        for s in ("H", "S", "reply"):
            Q = {k: v[:, sg[s][0]:sg[s][1]] for k, v in D.items()}
            lay[s] = [r4(dot(Q[x], Q[y])) for x, y in (("dN", "dN"), ("dP", "dP"), ("dP", "dN"), ("c6", "c6"), ("c6", "dN"))]
        heads = {}
        for L in hl:
            hN, hP = Hd["N6"][L] - Hd["N1"][L], Hd["P6"][L] - Hd["P1"][L]
            heads[str(L)] = {s: [r4((x[i0:i1] * y[i0:i1]).sum((0, 2))) for x, y in ((hN, hN), (hP, hP), (hN, hP))]
                             for s, (i0, i1) in (("Srep", sg["Srep"]), ("reply", sg["reply"]))}
        _STORE["pairs"][f] = {"cand": cand, "ld": ld}
        rec.update(ld=ld, lpa=lpa, lpb=lpb, p1=p1, used_prices=[a, b], layers=lay, heads=heads,
                   decision={c: [tok.decode(cand[c][0]), tok.decode([cand[c][1]]), tok.decode([cand[c][2]])] for c in cand},
                   resid_norm_N1={s: r4(X["N1"][:, sg[s][0]:sg[s][1]].norm(dim=-1).mean(1)) for s in ("S", "reply")},
                   h_identical=round(float((D["dP"][:, :e["n_H"]] - D["dN"][:, :e["n_H"]]).abs().max()), 6), secs=round(time.time() - t0, 1))
        del X, Hd, D
        out.append(rec)
    return out


def e4_stats(model, tok, items, settings, model_name, model_io, **kw):
    """Fixture-mean directions per layer and segment: |v_N|, |v_P|, cos, projections of v_P, c6, c1 on v_N."""
    n = _STORE["n"]
    if not n:
        raise ValueError("run e4_build first")
    v = {k: x / n for k, x in _STORE["acc"].items()}
    r5 = lambda t: [round(float(x), 5) for x in t]                       # noqa: E731
    per = {}
    for s in ("A", "H", "S", "reply"):
        i0, i1 = _STORE["segs"][s]
        a, b, c, c1 = (v[k][:, i0:i1] for k in ("dN", "dP", "c6", "c1"))
        dot = lambda x, y: (x * y).sum((1, 2))                           # noqa: E731
        aa = dot(a, a).clamp_min(1e-30)
        per[s] = {"norm_vN": r5(a.norm(dim=-1).mean(1)), "norm_vP": r5(b.norm(dim=-1).mean(1)), "norm_c6": r5(c.norm(dim=-1).mean(1)),
                  "cos_vP_vN": r5(dot(a, b) / (aa * dot(b, b).clamp_min(1e-30)).sqrt()), "proj_vP_on_vN": r5(dot(a, b) / aa),
                  "cos_c6_vN": r5(dot(a, c) / (aa * dot(c, c).clamp_min(1e-30)).sqrt()), "proj_c6_on_vN": r5(dot(a, c) / aa),
                  "proj_c1_on_vN": r5(dot(a, c1) / aa)}
    res = {"n_fixtures": n, "n_layers": int(v["dN"].shape[0]), "per_segment": per, "n_nonmoney": _STORE["u_n"]}
    if _STORE["u_n"]:
        u, m = _STORE["u"], v["dN"].mean(1)
        res["cos_u_vN_mean"] = r5((u * m).sum(-1) / (u.norm(dim=-1) * m.norm(dim=-1)).clamp_min(1e-30))
    return [res]


def e4_attn(model, tok, items, settings, model_name, model_io, **kw):
    """Per prompt and layer: [query: reply opening tokens, last token][key: limit digits (+ LO, HI, range line)] x heads."""
    prev = getattr(model.config, "_attn_implementation", None)
    if prev != "eager":
        model.set_attn_implementation("eager")
    blocks = model_io.decoder_layers(model)
    out = []
    try:
        for it in items:
            t0 = time.time()
            rec = {"item_id": it["item_id"], "fixture": it["fixture"], "mass": {}, "keys": {}}
            for k, e in encode_fixture(tok, model_io, it).items():
                qs = (e["A"][e["n_H"] + e["n_S"]:], [len(e["ids"]) - 1])
                keys = [("limit", e["span"])] + ([("lo", e["lo"]), ("hi", e["hi"]), ("range", e["R"])] if e["R"] else [])
                m = [None] * len(blocks)

                def mk(i, qs=qs, keys=keys, m=m):
                    def h(_m, _i, o):
                        if len(o) > 1 and o[1] is not None:
                            w = o[1][0].float()
                            m[i] = [[[round(float(x), 4) for x in w[:, q][:, :, sp[0]:sp[-1] + 1].sum(-1).mean(-1)] for _n, sp in keys] for q in qs]
                    return h
                hs = [blk.self_attn.register_forward_hook(mk(i)) for i, blk in enumerate(blocks)]
                try:
                    _fwd(model, model_io, 1, input_ids=torch.tensor([e["ids"]], device=model.device), use_cache=False)
                finally:
                    for h in hs:
                        h.remove()
                rec["mass"][k], rec["keys"][k] = m, [n for n, _ in keys]
            out.append(dict(rec, secs=round(time.time() - t0, 1)))
    finally:
        if prev != "eager":
            model.set_attn_implementation(prev)
    return out


def _arm(blocks, hd, chunk, vals, V, apos, sg):
    tab = {}
    for r, spec in enumerate(chunk):
        for op in spec.get("ops", ()):
            if op[0] == "vec":
                kind, L, h, (i0, i1) = "resid", int(op[1]), -1, sg[op[2]]
                mode, v = "add", float(op[3]) * V[L][op[4]][i0:i1]
            else:
                kind, L, h, (i0, i1) = op[1], int(op[2]), int(op[3]), sg[op[4]]
                if op[0] == "set":
                    mode, v = "set", vals[op[5]][(kind, L)][i0:i1]
                else:
                    mode, v = "add", float(op[5]) * (vals[op[6]][(kind, L)][i0:i1].float() - vals[op[7]][(kind, L)][i0:i1].float())
                if kind == "head":
                    v = v[:, h * hd[L]:(h + 1) * hd[L]]
            tab.setdefault((kind, L), []).append((r, apos[i0:i1], h, mode, v))
    fired, hs = [0], []

    def apply(x, ents, w=None):
        for want in ("set", "add"):
            for r, p, h, mode, v in ents:
                if mode == want:
                    c = slice(None) if w is None or h < 0 else slice(h * w, (h + 1) * w)
                    x[r, p, c] = v.to(x.dtype) if mode == "set" else x[r, p, c] + v.to(x.dtype)
        return x

    def hook(kind, ents, w):
        def f(_m, a, o=None):
            fired[0] += 1
            if kind == "resid":
                return hooks._replace_block_output(o, apply(hooks._block_output_residual(o).clone(), ents))
            if kind == "attn":
                return (apply(o[0].clone(), ents),) + tuple(o[1:])
            return (apply(a[0].clone(), ents, w),) + tuple(a[1:])
        return f
    for (kind, L), ents in tab.items():
        f = hook(kind, ents, hd[L])
        if kind == "resid":
            hs.append(blocks[L].register_forward_hook(f))
        elif kind == "attn":
            hs.append(blocks[L].self_attn.register_forward_hook(f))
        else:
            hs.append(blocks[L].self_attn.o_proj.register_forward_pre_hook(f))
    return hs, fired


def e4_patch(model, tok, items, settings, model_name, model_io, rows=None, batch=16, loo=True, **kw):
    """Item: one fixture. rows: {target: [{"n", "ops"}]}, run in batches (padded with clean rows) on E2's forced opening
    with the shared prefix's KV cache. Per row: ld (a, b = the plain greedy prices in the target's units), the two
    log-probs, and the first price digit's probabilities."""
    rows, blocks = rows or {}, model_io.decoder_layers(model)
    hd, digits = _geometry(blocks)[1], T._digit_ids(tok)
    out = []
    for it in items:
        t0, f = time.time(), it["fixture"]
        Pq = _STORE["pairs"].get(f)
        rec = {"item_id": it["item_id"], "fixture": f, "base_limit": it["base_limit"]}
        if Pq is None:
            out.append(dict(rec, skipped="no usable prices from e4_build"))
            continue
        E = encode_fixture(tok, model_io, it)
        ref = E["N1"]
        sg, s0, need, vl = segs(ref), ref["s0"], {}, set()
        for op in (op for specs in rows.values() for spec in specs for op in spec.get("ops", ())):
            if op[0] == "vec":
                vl.add(int(op[1]))
            for p in (op[5:6] if op[0] == "set" else op[6:8] if op[0] == "add" else ()):
                need.setdefault(p, set()).add((op[1], int(op[2])))
        vals = {p: _capture(model, model_io, blocks, E[p]["ids"], E[p]["A"], sorted(need[p]))[0] for p in need}
        V = {L: vectors(f, L, loo) for L in vl}
        base, how = T._batched_prefix_cache(model, model_io, ref["ids"][:s0], batch)
        res = {}
        for tgt, specs in rows.items():
            e = E[tgt]
            pre, ta, tb = Pq["cand"][SCALE[tgt]]
            apos = torch.tensor([p - s0 for p in e["A"]], device=model.device)
            allspecs = list(specs) + [{"n": "clean"}] * (batch - len(specs) % batch)
            x = torch.tensor([e["ids"][s0:] + pre], device=model.device).expand(batch, -1).contiguous()
            got, ok = [], True
            for b0 in range(0, len(allspecs), batch):
                chunk = allspecs[b0:b0 + batch]
                hs, fired = _arm(blocks, hd, chunk, vals, {L: V[L][0] for L in V}, apos, sg)
                cache = copy.deepcopy(base)
                try:
                    o = _fwd(model, model_io, len(pre) + 1, input_ids=x, past_key_values=cache, use_cache=True)
                finally:
                    for h in hs:
                        h.remove()
                ok &= fired[0] == len(hs)
                lg = o.logits.float()
                p0, lp = torch.softmax(lg[:, 0], -1), torch.log_softmax(lg[:, -1], -1)
                got += [{"n": s["n"], "ld": round(float(lp[r, tb] - lp[r, ta]), 5), "lpa": round(float(lp[r, ta]), 4),
                         "lpb": round(float(lp[r, tb]), 4), "p1": [round(float(p0[r, i]), 4) for i in digits] if None not in digits else None}
                        for r, s in enumerate(chunk)]
                del cache, o, lg
            clean = [g["ld"] for g in got if g["n"] == "clean"]
            res[tgt] = {"ld_clean": clean[0], "clean_spread": round(max(clean) - min(clean), 5), "hooks_ok": ok,
                        "rows": [g for g in got if g["n"] != "clean"] + [g for g in got if g["n"] == "clean"][:1]}
        used = all(v[1] for v in V.values()) if V else None
        del base, vals, V
        out.append(dict(rec, ld_build=Pq["ld"], batch=batch, prefix_cache=how, loo=used,
                        targets=res, secs=round(time.time() - t0, 1)))
    return out


TASKS = {"e4_steer": e4_steer, "e4_build": e4_build, "e4_stats": e4_stats, "e4_attn": e4_attn, "e4_patch": e4_patch}
