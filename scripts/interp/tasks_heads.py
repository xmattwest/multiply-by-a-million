"""tasks_heads.py (POD-SIDE plug-in): which attention heads carry the seller's limit into the reply?
(config/interp/plan_next.md, session 2; STORY X56.) Exploratory.

Readout everywhere: a logit difference at the decision token. Per fixture, a = the x1 run's greedy price and
b = the x1e6 run's (R2: the opening forced), each written in the run's own units (x1e6 prices are whole millions).
After the forced opening plus the two prices' shared leading digits, ld = log p(b's next token) - log p(a's).
It moves continuously where greedy prices move in round steps.

  heads_build  per fixture pair: both greedy prices; ld in each run; the logit lens of ld at every layer (final
               norm, tied unembedding, softcap; at the decision position); and the x1e6 - x1 difference at the
               steering layer on the post-limit tokens (kept here, for the latch test).
  heads_attn   attention mass on the limit's digit tokens per layer and head, from the reply's opening tokens,
               the prompt's post-limit tokens and the last token (eager attention for this step only).
  heads_patch  E2's setup, with row specs from the laptop. A row may patch the residual stream, an attention
               output, an MLP output or single heads (o_proj input) with the source run's values, on the
               post-limit tokens or part of them ("p"); steer at the steering layer, leave-one-fixture-out
               ("s"); and block attention from every token after the limit to the limit's digits, for all or
               chosen heads at chosen layers ("k").
"""
import copy
import os
import time

import torch                                   # pod-only

import hooks
import tasks as T

_STORE = {"pairs": {}, "d": {}, "steer_layer": None}


def _fwd(model, model_io, k, **kw):
    with torch.no_grad(), model_io.attn_ctx():
        try:
            return model(logits_to_keep=int(k), **kw)
        except TypeError:
            return model(**kw)


def _decision(tok, a, b):
    """(shared prefix ids, a's next token, b's next token) for the prices a and b followed by ','."""
    ia, ib = (tok(f"{int(x)},", add_special_tokens=False)["input_ids"] for x in (a, b))
    k = 0
    while k < min(len(ia), len(ib)) and ia[k] == ib[k]:
        k += 1
    if k == min(len(ia), len(ib)):
        raise ValueError("the two prices are the same")
    return ia[:k], ia[k], ib[k]


def _lens_fn(model, model_io):
    norm = model_io.text_decoder(model).norm
    W = model.get_output_embeddings().weight
    cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    cap = getattr(cfg, "final_logit_softcapping", None)

    def f(h):
        x = torch.nn.functional.linear(norm(h.to(W.dtype)), W).float()
        return torch.tanh(x / cap) * cap if cap else x
    return f


def _geometry(blocks):
    nH = int(blocks[0].self_attn.config.num_attention_heads)
    return nH, [int(b.self_attn.o_proj.in_features) // nH for b in blocks]


def heads_build(model, tok, items, settings, model_name, model_io, steer_layer=20, **kw):
    blocks = model_io.decoder_layers(model)
    nL, L_s = len(blocks), int(steer_layer)
    if not 0 <= L_s < nL:
        raise ValueError(f"steer_layer must lie in 0..{nL - 1}")
    nH, hd = _geometry(blocks)
    lens = _lens_fn(model, model_io)
    stand_in_ok = os.environ.get("SI_INTERP_REHEARSAL") == "1"
    _STORE["steer_layer"] = L_s
    out = []
    for it in items:
        t0 = time.time()
        b = it["base_limit"]
        rec = {"item_id": it["item_id"], "fixture": it["fixture"], "base_limit": b, "n_layers": nL, "n_heads": nH,
               "head_dim": hd, "steer_layer": L_s}
        pr = {c: T.teacher(model, tok, [{"item_id": f"{it['fixture']}-{c}", "messages": it[c]["messages"],
                                          "role": it["role"], "condition": it[c]["condition"],
                                          "limit_units": it[c]["limit_units"]}], settings, model_name, model_io)[0]
              for c in ("x1", "x1e6")}
        P1, P6 = pr["x1"]["price_r2"], pr["x1e6"]["price_r2"]
        rec.update(price_x1=P1, price_x1e6=P6, cont={c: pr[c]["continuation"] for c in pr}, stand_in=False)
        why = None if P1 and P6 else "no greedy price"
        if why is None and P6 % 10 ** 6:
            why = "the x1e6 price is not a whole number of millions"
        if why is None and P1 * 10 ** 6 == P6:
            why = "the two prices are equal after rescaling"
        if why and stand_in_ok:                 # local rehearsal only: random weights write no prices
            P1, P6, why = round(1.3 * b), round(1.2 * b) * 10 ** 6, None
            rec.update(price_x1=P1, price_x1e6=P6, stand_in=True)
        if why:
            out.append(dict(rec, skipped=why, secs=round(time.time() - t0, 1)))
            continue
        cand = {"x1": _decision(tok, P1, P6 // 10 ** 6), "x1e6": _decision(tok, P1 * 10 ** 6, P6)}
        ld, lens_ld, acts, shape = {}, {}, {}, {}
        for c in ("x1", "x1e6"):
            side = it[c]
            E = T._r2_encode(tok, model_io, side["messages"], side["limit_text"], it["role"])
            ch = T._chat_encode(tok, model_io, side["messages"], side["limit_text"], it["role"])
            if E["ids"][:len(ch["ids"])] != ch["ids"]:
                raise ValueError("the chat prompt is not a prefix of the R2 input")
            pre, ta, tb = cand[c]
            ids = E["ids"] + pre
            post = list(range(E["span"][-1] + 1, len(E["ids"])))
            got = [None] * nL

            def mk(i, post=post, c=c, dec=len(ids) - 1):
                def h(_m, _i, o):
                    r = hooks._block_output_residual(o)
                    got[i] = r[0, dec].detach()
                    if i == L_s:
                        acts[c] = r[0, post].detach().float()
                return h
            hs = [blk.register_forward_hook(mk(i)) for i, blk in enumerate(blocks)]
            try:
                o = _fwd(model, model_io, 1, input_ids=torch.tensor([ids], device=model.device), use_cache=False)
            finally:
                for h in hs:
                    h.remove()
            lp = torch.log_softmax(o.logits[0, -1].float(), -1)
            ld[c] = float(lp[tb] - lp[ta])
            ll = torch.log_softmax(lens(torch.stack(got)), -1)
            lens_ld[c] = [round(float(x), 4) for x in ll[:, tb] - ll[:, ta]]
            shape[c] = (len(post), len(ch["post"]), E["ids"][post[0]:])
        if shape["x1"] != shape["x1e6"]:
            raise ValueError(f"{it['fixture']}: text after the limit differs")
        n_post, n_prompt, _ = shape["x1"]
        _STORE["pairs"][it["fixture"]] = {"cand": cand, "n_post": n_post, "n_prompt": n_prompt}
        _STORE["d"][it["fixture"]] = acts["x1e6"] - acts["x1"]
        rec.update(ld=ld, lens=lens_ld, n_post=n_post, n_prompt=n_prompt,
                   decision={c: {"prefix": tok.decode(cand[c][0]), "a": tok.decode([cand[c][1]]),
                                 "b": tok.decode([cand[c][2]])} for c in cand},
                   norm_d=round(float(_STORE["d"][it["fixture"]].norm(dim=-1).mean()), 4),
                   secs=round(time.time() - t0, 1))
        out.append(rec)
    return out


def heads_attn(model, tok, items, settings, model_name, model_io, **kw):
    """Mass on the digit tokens: per run, a list over layers of [reply, prompt, last] x heads."""
    prev = getattr(model.config, "_attn_implementation", None)
    switched = prev != "eager"
    if switched:
        try:
            model.set_attn_implementation("eager")
        except Exception as e:                # noqa: BLE001
            return [{"item_id": it["item_id"], "fixture": it["fixture"], "error": f"no eager attention: {e}"} for it in items]
    blocks = model_io.decoder_layers(model)
    out = []
    try:
        for it in items:
            t0 = time.time()
            rec = {"item_id": it["item_id"], "fixture": it["fixture"], "mass": {}, "n_digits": {}}
            for c in ("x1", "x1e6"):
                side = it[c]
                E = T._r2_encode(tok, model_io, side["messages"], side["limit_text"], it["role"])
                n_prompt = len(T._chat_encode(tok, model_io, side["messages"], side["limit_text"], it["role"])["post"])
                sp = E["span"]
                post = list(range(sp[-1] + 1, len(E["ids"])))
                qs = (post[n_prompt:], post[:n_prompt], [len(E["ids"]) - 1])
                m = [None] * len(blocks)

                def mk(i):
                    def h(_m, _i, o):
                        if o[1] is not None:
                            w = o[1][0, :, :, sp[0]:sp[-1] + 1].float().sum(-1)        # (heads, queries)
                            m[i] = [[round(float(x), 4) for x in w[:, q].mean(-1)] for q in qs]
                    return h
                hs = [blk.self_attn.register_forward_hook(mk(i)) for i, blk in enumerate(blocks)]
                try:
                    _fwd(model, model_io, 1, input_ids=torch.tensor([E["ids"]], device=model.device), use_cache=False)
                finally:
                    for h in hs:
                        h.remove()
                rec["mass"][c], rec["n_digits"][c] = m, len(sp)
            rec["secs"] = round(time.time() - t0, 1)
            out.append(rec)
    finally:
        if switched:
            model.set_attn_implementation(prev)
    return out


def _source(model, model_io, blocks, ids, base, sl, need, ta, tb, k):
    """The source run's values on the post-limit tokens for every (kind, layer) needed, and its own ld."""
    vals = {kind: {} for kind in need}
    hs = []
    for L in need["resid"]:
        hs.append(blocks[L].register_forward_hook(
            lambda _m, _i, o, L=L: vals["resid"].__setitem__(L, hooks._block_output_residual(o)[0, sl].detach())))
    for L in need["attn"]:
        hs.append(blocks[L].self_attn.register_forward_hook(
            lambda _m, _i, o, L=L: vals["attn"].__setitem__(L, o[0][0, sl].detach())))
    for L in need["mlp"]:
        hs.append(blocks[L].mlp.register_forward_hook(lambda _m, _i, o, L=L: vals["mlp"].__setitem__(L, o[0, sl].detach())))
    for L in need["head"]:
        hs.append(blocks[L].self_attn.o_proj.register_forward_pre_hook(
            lambda _m, a, L=L: vals["head"].__setitem__(L, a[0][0, sl].detach())))
    try:
        o = _fwd(model, model_io, k, input_ids=torch.tensor([ids], device=model.device), past_key_values=base, use_cache=True)
    finally:
        for h in hs:
            h.remove()
    lp = torch.log_softmax(o.logits[0, -1].float(), -1)
    return vals, float(lp[tb] - lp[ta])


def _arm(blocks, hd, chunk, src, sl, vsl, add, ko_q0, ko_k):
    """Register one batch's hooks; returns (handles, fired counter, expected count)."""
    tab = {x: {} for x in ("resid", "attn", "mlp", "head", "add", "ko")}
    for r, spec in enumerate(chunk):
        for kind, L, h, sc in spec.get("p", ()):
            v = src[kind][L][vsl[sc]]
            if kind == "head":
                v = v[:, h * hd[L]:(h + 1) * hd[L]]
            tab[kind].setdefault(L, []).append((r, sl[sc], h, v))
        if spec.get("s") and add is not None:
            tab["add"].setdefault(add[0], []).append((r, sl["post"], -1, float(spec["s"]) * add[1]))
        for L, h in spec.get("k", ()):
            cur = tab["ko"].setdefault(L, {})
            cur[r] = None if h < 0 or cur.get(r, ()) is None else sorted(set(cur.get(r, ())) | {h})
    fired, hs = [0], []

    def layer_hook(pat, ad):
        def f(_m, _i, o):
            fired[0] += 1
            x = hooks._block_output_residual(o).clone()
            for r, s, _h, v in ad:
                x[r, s] = x[r, s] + v.to(x.dtype)
            for r, s, _h, v in pat:
                x[r, s] = v
            return hooks._replace_block_output(o, x)
        return f

    def attn_hook(pat):
        def f(_m, _i, o):
            fired[0] += 1
            x = o[0].clone()
            for r, s, _h, v in pat:
                x[r, s] = v
            return (x,) + tuple(o[1:])
        return f

    def mlp_hook(pat):
        def f(_m, _i, o):
            fired[0] += 1
            x = o.clone()
            for r, s, _h, v in pat:
                x[r, s] = v
            return x
        return f

    def head_hook(pat, d):
        def f(_m, a):
            fired[0] += 1
            x = a[0].clone()
            for r, s, h, v in pat:
                x[r, s, h * d:(h + 1) * d] = v
            return (x,) + tuple(a[1:])
        return f

    def ko_hook(rows, nH):
        def f(_m, a, kw):
            fired[0] += 1
            hs_ = a[0]
            B, q = hs_.shape[0], hs_.shape[1]
            m = kw.get("attention_mask")
            if m is None:                    # no explicit mask: build the causal one (window > sequence)
                kv = ko_k[0] + q
                m = torch.ones(q, kv, dtype=torch.bool, device=hs_.device).tril(kv - q)[None, None]
            per_head = any(v is not None for v in rows.values())
            m = m.expand(B, nH if per_head else m.shape[1], q, m.shape[-1]).clone()
            block = False if m.dtype == torch.bool else torch.finfo(m.dtype).min
            for r, heads in rows.items():
                if heads is None:
                    m[r, :, ko_q0:, ko_k[0]:ko_k[1]] = block
                else:
                    for h in heads:
                        m[r, h, ko_q0:, ko_k[0]:ko_k[1]] = block
            kw["attention_mask"] = m
            return a, kw
        return f

    nH = int(blocks[0].self_attn.config.num_attention_heads)
    for L in sorted(set(tab["resid"]) | set(tab["add"])):
        hs.append(blocks[L].register_forward_hook(layer_hook(tab["resid"].get(L, []), tab["add"].get(L, []))))
    for L, pat in tab["attn"].items():
        hs.append(blocks[L].self_attn.register_forward_hook(attn_hook(pat)))
    for L, pat in tab["mlp"].items():
        hs.append(blocks[L].mlp.register_forward_hook(mlp_hook(pat)))
    for L, pat in tab["head"].items():
        hs.append(blocks[L].self_attn.o_proj.register_forward_pre_hook(head_hook(pat, hd[L])))
    for L, rows in tab["ko"].items():
        hs.append(blocks[L].register_forward_pre_hook(ko_hook(rows, nH), with_kwargs=True))
    return hs, fired, len(hs)


def heads_patch(model, tok, items, settings, model_name, model_io, rows=(), batch=16, loo=True, **kw):
    """Item: one fixture and direction (E2's source -> target). Rows: the laptop's specs; padded with clean rows
    to a multiple of `batch`. Per row: ld, the decision tokens' log-probs, p(first price token is a digit) and the
    first-digit probabilities (E2's p1)."""
    blocks = model_io.decoder_layers(model)
    _nH, hd = _geometry(blocks)
    digit_ids = T._digit_ids(tok)
    if None in digit_ids:
        raise ValueError("digits are not single tokens")
    need = {x: set() for x in ("resid", "attn", "mlp", "head")}
    for spec in rows:
        for kind, L, _h, _s in spec.get("p", ()):
            need[kind].add(int(L))
    dev = model.device
    out = []
    for it in items:
        t0 = time.time()
        f = it["fixture"]
        P = _STORE["pairs"].get(f)
        base_rec = {"item_id": it["item_id"], "fixture": f, "role": it["role"], "direction": it["direction"],
                    "base_limit": it["base_limit"], "target_condition": it["target"]["condition"]}
        if P is None:
            out.append(dict(base_rec, skipped="no usable prices from heads_build"))
            continue
        S = T._r2_encode(tok, model_io, it["source"]["messages"], it["source"]["limit_text"], it["role"])
        Tg = T._r2_encode(tok, model_io, it["target"]["messages"], it["target"]["limit_text"], it["role"])
        s0 = Tg["span"][0]
        ns, nt = len(S["span"]), len(Tg["span"])
        n_post, n_prompt = P["n_post"], P["n_prompt"]
        if S["span"][0] != s0 or S["ids"][:s0] != Tg["ids"][:s0] or S["ids"][S["span"][-1] + 1:] != Tg["ids"][Tg["span"][-1] + 1:] \
                or len(Tg["ids"]) - Tg["span"][-1] - 1 != n_post:
            raise ValueError(f"{it['item_id']}: source and target differ outside the limit digits")
        cu_t = "x1" if it["target"]["condition"] == "D2" else "x1e6"
        cu_s = "x1e6" if cu_t == "x1" else "x1"
        pre_t, ta, tb = P["cand"][cu_t]
        pre_s, sa, sb = P["cand"][cu_s]
        base1, _ = T._batched_prefix_cache(model, model_io, Tg["ids"][:s0], 1)
        src, ld_s = _source(model, model_io, blocks, S["ids"][s0:] + pre_s, base1, slice(ns, ns + n_post), need,
                            sa, sb, len(pre_s) + 1)
        del base1
        add = None
        if any(spec.get("s") for spec in rows):
            D = _STORE["d"]
            tot = sum(D.values())
            use_loo = loo and len(D) >= 3
            v = (tot - D[f]) / (len(D) - 1) if use_loo else tot / len(D)
            add = (_STORE["steer_layer"], (1.0 if cu_t == "x1" else -1.0) * v)
        sl = {"post": slice(nt, nt + n_post), "prompt": slice(nt, nt + n_prompt), "reply": slice(nt + n_prompt, nt + n_post)}
        vsl = {"post": slice(None), "prompt": slice(0, n_prompt), "reply": slice(n_prompt, None)}
        suf = Tg["ids"][s0:] + pre_t
        k = len(pre_t) + 1
        specs = list(rows) + [{"n": "clean"}]
        while len(specs) % batch:
            specs.append({"n": "clean"})
        base, how = T._batched_prefix_cache(model, model_io, Tg["ids"][:s0], batch)
        x = torch.tensor([suf], device=dev).expand(batch, -1).contiguous()
        res, hooks_ok = [], True
        for b0 in range(0, len(specs), batch):
            chunk = specs[b0:b0 + batch]
            hs, fired, n_hooks = _arm(blocks, hd, chunk, src, sl, vsl, add, nt, (s0, s0 + nt))
            cache = copy.deepcopy(base)
            try:
                o = _fwd(model, model_io, k, input_ids=x, past_key_values=cache, use_cache=True)
            finally:
                for h in hs:
                    h.remove()
            hooks_ok &= fired[0] == n_hooks
            lg = o.logits.float()
            p0 = torch.softmax(lg[:, 0], -1)
            lp = torch.log_softmax(lg[:, -1], -1)
            for r, spec in enumerate(chunk):
                res.append({"n": spec["n"], "ld": round(float(lp[r, tb] - lp[r, ta]), 5), "lpa": round(float(lp[r, ta]), 4),
                            "lpb": round(float(lp[r, tb]), 4), "pd": round(float(p0[r, digit_ids].sum()), 4),
                            "p1": [round(float(p0[r, i]), 4) for i in digit_ids]})
            del cache, o, lg
        del base, src
        clean = [r["ld"] for r in res if r["n"] == "clean"]
        out.append(dict(base_rec, ld_source=round(ld_s, 5), ld_clean=clean[0], clean_spread=round(max(clean) - min(clean), 5),
                        n_rows=len(rows), batch=batch, prefix_cache=how, hooks_ok=hooks_ok, loo=add is not None and loo,
                        rows=[r for r in res if r["n"] != "clean"] + [r for r in res if r["n"] == "clean"][:1],
                        secs=round(time.time() - t0, 1)))
    return out


TASKS = {"heads_build": heads_build, "heads_attn": heads_attn, "heads_patch": heads_patch}
