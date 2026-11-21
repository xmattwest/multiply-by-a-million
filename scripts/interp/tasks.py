#!/usr/bin/env python3
"""tasks.py (POD-SIDE): the interp experiments. E0 (fidelity) now; E1-E3 after E0 passes.

Every task has the same signature:
    fn(model, tok, items, settings, model_name, model_io, **kw) -> list of dicts
and returns only scalars / short lists (full caches never leave the pod).

E0 (plan_mechinterp §2):
  render      prompt text + token count per item (template check).
  generate    R1: the opening, parsed with game_engine.validate_reply, as a gap.
              Constrained by the same JSON schema vLLM enforced (xgrammar, which
              ships in the vLLM image). If xgrammar can't be used, falls back to
              unconstrained generation and ALSO reports a lenient parse
              (code fences stripped), flagged constrained=False.
              kw: greedy (bool), schema (dict, from the laptop).
  teacher     R2: the text up to '{"action": "OFFER", "price": ' is forced; the
              greedy continuation's leading digits give a price, and the
              first-digit distribution is returned for cheap sweeps later.
"""
import json
import re
import time

import torch                                   # pod-only

_GRAMMAR_CACHE = {}
_PREFIX = '{"action": "OFFER", "price": '
_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.S)


def _gap(role, price, limit_units):
    return price / limit_units - 1 if role == "seller" else 1 - price / limit_units


def _eos_ids(model, tok):
    ids = getattr(model.generation_config, "eos_token_id", None)
    if ids is None:
        ids = tok.eos_token_id
    return set(ids if isinstance(ids, (list, tuple)) else [ids])


def _grammar_processor(tok, schema, vocab_size):
    """A fresh xgrammar HF logits processor for this schema, or (None, reason)."""
    try:
        import xgrammar as xgr
        from xgrammar.contrib.hf import LogitsProcessor
    except Exception as e:                    # noqa: BLE001
        return None, f"xgrammar unavailable: {type(e).__name__}: {e}"
    key = json.dumps(schema, sort_keys=True)
    try:
        if key not in _GRAMMAR_CACHE:
            info = xgr.TokenizerInfo.from_huggingface(tok, vocab_size=vocab_size)
            _GRAMMAR_CACHE[key] = xgr.GrammarCompiler(info).compile_json_schema(json.dumps(schema))
        try:
            from importlib.metadata import version
            ver = version("xgrammar")
        except Exception:                     # noqa: BLE001
            ver = getattr(xgr, "__version__", "?")
        return LogitsProcessor(_GRAMMAR_CACHE[key]), "xgrammar " + ver
    except Exception as e:                    # noqa: BLE001
        return None, f"xgrammar failed: {type(e).__name__}: {e}"


def render(model, tok, items, settings, model_name, model_io, **kw):
    out = []
    for it in items:
        text, ids = model_io.render_messages(tok, it["messages"])
        out.append({"item_id": it["item_id"], "prompt_text": text, "n_tokens": int(ids.shape[1])})
    return out


def generate(model, tok, items, settings, model_name, model_io, greedy=False, schema=None, **kw):
    """R1 openings. Sampled with the frozen settings and the item's seed unless greedy."""
    import game_engine as ge
    max_chars = settings["max_message_chars"]["value"]
    max_new = settings["max_tokens"]["value"]
    samp = settings["sampling"][model_name]
    eos = _eos_ids(model, tok)
    try:
        vocab = model.get_output_embeddings().weight.shape[0]
    except AttributeError:
        vocab = getattr(model.config, "text_config", model.config).vocab_size
    out = []
    for it in items:
        t0 = time.time()
        _text, ids = model_io.render_messages(tok, it["messages"])
        ids = ids.to(model.device)
        lp, how = (None, "no schema sent") if schema is None else _grammar_processor(tok, schema, vocab)
        kwargs = {"max_new_tokens": max_new, "attention_mask": torch.ones_like(ids)}
        if lp is not None:
            kwargs["logits_processor"] = [lp]
        if greedy:
            kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        else:
            torch.manual_seed(int(it["seed"]) & 0x7FFFFFFF)
            kwargs.update(do_sample=True, temperature=samp["temperature"], top_p=samp["top_p"],
                          top_k=samp["top_k"])
        with model_io.attn_ctx():
            gen = model.generate(ids, **kwargs)
        new = gen[0, ids.shape[1]:].tolist()
        stopped = bool(new) and new[-1] in eos
        finish = "stop" if stopped else ("length" if len(new) >= max_new else "stop")
        content = tok.decode(new, skip_special_tokens=True)
        move, errors = ge.validate_reply(content, finish, False, max_chars)
        rec = {"item_id": it["item_id"], "role": it["role"], "condition": it["condition"],
               "limit_units": it["limit_units"], "greedy": bool(greedy), "constrained": lp is not None,
               "grammar": how, "finish": finish, "completion_tokens": len(new), "content": content[:800],
               "errors": errors, "action": None, "price": None, "gap": None,
               "secs": round(time.time() - t0, 2)}
        if move is None and lp is None:       # unconstrained fallback: try without code fences
            m = _FENCE.match(content)
            if m:
                move, lerr = ge.validate_reply(m.group(1), finish, False, max_chars)
                rec["lenient_errors"] = lerr
                rec["lenient"] = move is not None
        if move:
            rec["action"] = move["action"]
            if move["action"] == "OFFER":
                rec["price"] = move["price"]
                rec["gap"] = _gap(it["role"], move["price"], it["limit_units"])
        out.append(rec)
    return out


def _digit_ids(tok):
    ids = []
    for d in "0123456789":
        enc = tok(d, add_special_tokens=False)["input_ids"]
        ids.append(enc[0] if len(enc) == 1 else None)
    return ids


def teacher(model, tok, items, settings, model_name, model_io, max_new=14, **kw):
    """R2: force the JSON up to the price; greedy digits + first-digit distribution."""
    digit_ids = _digit_ids(tok)
    out = []
    for it in items:
        text, _ = model_io.render_messages(tok, it["messages"])
        ids = tok(text + _PREFIX, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
        with model_io.attn_ctx():
            res = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new, do_sample=False,
                                 temperature=None, top_p=None, top_k=None,
                                 output_scores=True, return_dict_in_generate=True)
        cont = tok.decode(res.sequences[0, ids.shape[1]:], skip_special_tokens=True)
        m = re.match(r"\s*(\d+)", cont)
        price = int(m.group(1)) if m else None
        p_first = None
        if all(i is not None for i in digit_ids):
            probs = torch.softmax(res.scores[0][0].float(), dim=-1)
            p_first = [round(float(probs[i]), 5) for i in digit_ids]
        rec = {"item_id": it["item_id"], "role": it["role"], "condition": it["condition"],
               "limit_units": it["limit_units"], "continuation": cont[:40], "price_r2": price,
               "p_first_digit": p_first, "gap_r2": None}
        if price:
            rec["gap_r2"] = _gap(it["role"], price, it["limit_units"])
        out.append(rec)
    return out


TASKS = {"render": render, "generate": generate, "teacher": teacher}


# ======================================================================================
# E1 and E2 (plan_mechinterp §4). Heavy tensors stay on the pod; only scalars come back.
# ======================================================================================
import copy                                    # noqa: E402
import math                                    # noqa: E402

_STORE = {"e1": {}}                            # item_id -> {"meta":..., "acts": {pos: (layers, hidden)}}
POS_TYPES = ("final", "post", "span")          # E2 (a), (b), (c)


def _forward(model, model_io, **kw):
    """One forward under the chosen attention setting; only the last position's logits."""
    with torch.no_grad(), model_io.attn_ctx():
        try:
            return model(logits_to_keep=1, **kw)
        except TypeError:                      # older signature
            return model(**kw)


def _batched_prefix_cache(model, model_io, prefix_ids, batch):
    """KV cache of the shared prefix, repeated to `batch` rows; (cache, how)."""
    one = torch.tensor([prefix_ids], device=model.device)
    cache = _forward(model, model_io, input_ids=one, use_cache=True).past_key_values
    if batch == 1:
        return cache, "batch 1"
    try:
        cache.batch_repeat_interleave(batch)
        return cache, "computed once, repeated"
    except (AttributeError, NotImplementedError, TypeError):
        many = one.expand(batch, -1).contiguous()
        return _forward(model, model_io, input_ids=many, use_cache=True).past_key_values, "computed at batch size"


def _locate(full, offs, limit_text, role):
    """Token indices of the limit's digits in `full` (exactly one 'Your private ... is N dollars.')."""
    name = "floor" if role == "seller" else "ceiling"
    anchor = f"Your private {name} is {limit_text} dollars."
    if full.count(anchor) != 1:
        raise ValueError(f"limit anchor found {full.count(anchor)} times")
    c0 = full.index(anchor) + len(f"Your private {name} is ")
    c1 = c0 + len(limit_text)
    span = [k for k, (s, e) in enumerate(offs) if s < c1 and e > c0]
    if not span or not tokens_cover(offs, span, c0, c1):
        raise ValueError("limit digits do not map onto whole tokens")
    return span


def tokens_cover(offs, span, c0, c1):
    return offs[span[0]][0] >= c0 and offs[span[-1]][1] <= c1


def _r2_encode(tok, model_io, messages, limit_text, role):
    """The R2 input (prompt + forced '{"action": "OFFER", "price": ') with the positions E1/E2 use."""
    text, _ = model_io.render_messages(tok, messages)
    full = text + _PREFIX
    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = list(enc["input_ids"]), [tuple(o) for o in enc["offset_mapping"]]
    span = _locate(full, offs, limit_text, role)
    prompt_end = max(k for k, (s, e) in enumerate(offs) if e <= len(text) and e > 0)
    return {"ids": ids, "span": span, "digit": span[-1], "prompt_end": prompt_end, "final": len(ids) - 1}


def _plain_encode(tok, text, amount_text):
    """A plain sentence with <bos>, and the position of the amount's last digit."""
    bos = tok.bos_token_id if tok.bos_token_id is not None else 2
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    offs = [tuple(o) for o in enc["offset_mapping"]]
    if text.count(amount_text) != 1:
        raise ValueError("amount must occur once")
    c0 = text.index(amount_text); c1 = c0 + len(amount_text)
    span = [k for k, (s, e) in enumerate(offs) if s < c1 and e > c0]
    if not span or not tokens_cover(offs, span, c0, c1):
        raise ValueError("amount digits do not map onto whole tokens")
    ids = [bos] + list(enc["input_ids"])
    return {"ids": ids, "digit": span[-1] + 1, "final": len(ids) - 1}


def _capture_all_layers(model, model_io, ids, positions, device_keep=False):
    """resid_post at every decoder layer, at `positions`; returns (layers, len(positions), hidden)."""
    layers = model_io.decoder_layers(model)
    pos = torch.tensor(positions, device=model.device)
    got = [None] * len(layers)

    def mk(i):
        def hook(_m, _inp, out):
            r = out[0] if isinstance(out, tuple) else out
            got[i] = r[0, pos, :].detach() if device_keep else r[0, pos, :].detach().to("cpu", torch.float32)
        return hook
    hs = [layer.register_forward_hook(mk(i)) for i, layer in enumerate(layers)]
    try:
        _forward(model, model_io, input_ids=torch.tensor([ids], device=model.device), use_cache=False)
    finally:
        for h in hs:
            h.remove()
    return torch.stack(got)


def e1_capture(model, tok, items, settings, model_name, model_io, **kw):
    """E1 step 1: store resid_post (all layers) at the digit / prompt-end / final tokens, on the pod."""
    out = []
    for it in items:
        t0 = time.time()
        if it["kind"] == "negotiation":
            e = _r2_encode(tok, model_io, it["messages"], it["limit_text"], it["role"])
            names = ("digit", "prompt_end", "final")
        else:
            e = _plain_encode(tok, it["text"], it["amount_text"])
            names = ("digit", "final")
        acts = _capture_all_layers(model, model_io, e["ids"], [e[n] for n in names])
        _STORE["e1"][it["item_id"]] = {"meta": {k: it[k] for k in it if k not in ("messages", "text")},
                                        "acts": {n: acts[:, j, :] for j, n in enumerate(names)}}
        out.append({"item_id": it["item_id"], "kind": it["kind"], "n_tokens": len(e["ids"]),
                    "positions": {n: e[n] for n in names}, "secs": round(time.time() - t0, 2)})
    return out


def _ridge_fit(X, y, lam):
    """Dual ridge on centred X (n x h, float64). Returns (mean, w, y_mean)."""
    mu, ym = X.mean(0), y.mean()
    Xc = X - mu
    K = Xc @ Xc.T
    a = torch.linalg.solve(K + lam * torch.eye(K.shape[0], dtype=K.dtype), y - ym)
    return mu, Xc.T @ a, ym


def _r2(y, p):
    ss = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - p) ** 2).sum() / ss) if ss > 0 else None


def e1_fit(model, tok, items, settings, model_name, model_io, alphas=(1e-3, 1e-2, 1e-1, 1.0), folds=5, **kw):
    """E1 step 2: linear probes of log10(amount), per layer, fitted on the pod.

    within: negotiation prompts only, folds by fixture (no fixture in both train and test);
            at digit, prompt_end and final.
    transfer: fit on the plain sentences, test on the negotiation prompts (digit->digit, final->final).
    The ridge strength is alpha x trace(K)/n, alpha picked per (probe, position) by the mean score
    over layers (one choice per curve, not per layer). Returns curves plus the held-out predictions.
    """
    st_ = _STORE["e1"]
    neg = sorted((k for k, v in st_.items() if v["meta"]["kind"] == "negotiation"))
    gen = sorted((k for k, v in st_.items() if v["meta"]["kind"] == "plain"))
    if len(neg) < 10:
        raise ValueError(f"only {len(neg)} negotiation items stored")
    y_n = torch.tensor([math.log10(st_[k]["meta"]["amount"]) for k in neg], dtype=torch.float64)
    y_g = torch.tensor([math.log10(st_[k]["meta"]["amount"]) for k in gen], dtype=torch.float64)
    fixtures = sorted({st_[k]["meta"]["fixture"] for k in neg})
    fold_of = {f: i % folds for i, f in enumerate(fixtures)}
    fold = torch.tensor([fold_of[st_[k]["meta"]["fixture"]] for k in neg])
    n_layers = next(iter(st_.values()))["acts"]["digit"].shape[0]

    def layer_X(keys, pos, layer):
        return torch.stack([st_[k]["acts"][pos][layer] for k in keys]).to(torch.float64)

    res = {"n_negotiation": len(neg), "n_plain": len(gen), "layers": n_layers, "within": {}, "transfer": {},
           "pred_within": {}, "items": neg, "amounts": {k: st_[k]["meta"]["amount"] for k in neg}}
    for pos in ("digit", "prompt_end", "final"):
        best = None
        for alpha in alphas:
            scores, preds = [], []
            for layer in range(n_layers):
                X = layer_X(neg, pos, layer)
                p = torch.empty_like(y_n)
                for f in range(folds):
                    tr, te = fold != f, fold == f
                    Xt = X[tr]
                    lam = alpha * float((Xt - Xt.mean(0)).pow(2).sum()) / max(1, int(tr.sum()))
                    mu, w, ym = _ridge_fit(Xt, y_n[tr], lam)
                    p[te] = (X[te] - mu) @ w + ym
                scores.append(_r2(y_n, p)); preds.append(p)
            mean = sum(s for s in scores if s is not None) / max(1, len(scores))
            if best is None or mean > best[0]:
                best = (mean, alpha, scores, preds)
        res["within"][pos] = {"alpha": best[1], "r2": [round(s, 4) if s is not None else None for s in best[2]]}
        res["pred_within"][pos] = [[round(float(v), 4) for v in p] for p in best[3]]
    if len(gen) >= 10:
        for gpos, npos in (("digit", "digit"), ("final", "final")):
            best = None
            for alpha in alphas:
                scores, plain_cv = [], []
                for layer in range(n_layers):
                    Xg, Xn = layer_X(gen, gpos, layer), layer_X(neg, npos, layer)
                    lam = alpha * float((Xg - Xg.mean(0)).pow(2).sum()) / len(gen)
                    mu, w, ym = _ridge_fit(Xg, y_g, lam)
                    scores.append(_r2(y_n, (Xn - mu) @ w + ym))
                    half = torch.arange(len(gen)) % 2 == 0           # plain-sentence fit quality (split half)
                    mu2, w2, ym2 = _ridge_fit(Xg[half], y_g[half], lam)
                    plain_cv.append(_r2(y_g[~half], (Xg[~half] - mu2) @ w2 + ym2))
                mean = sum(s for s in plain_cv if s is not None) / max(1, len(plain_cv))   # chosen on plain only
                if best is None or mean > best[0]:
                    best = (mean, alpha, scores, plain_cv)
            res["transfer"][f"{gpos}->{npos}"] = {"alpha": best[1],
                                                  "r2_negotiation": [round(s, 4) if s is not None else None for s in best[2]],
                                                  "r2_plain_split": [round(s, 4) if s is not None else None for s in best[3]]}
    return [res]


def e2_patch(model, tok, items, settings, model_name, model_io, layers=None, positions=POS_TYPES, batch=16,
             max_new=12, **kw):
    """E2: patch source activations into the target run, one (layer, position type) per batch row.

    Each item: one fixture and direction (source -> target, e.g. x1 -> x1e6). The tokens before
    the limit are identical in both prompts, so their KV cache is computed once (at the batch
    size) and copied for each batch; each batch then runs only the limit and what follows it,
    with row r patched at its layer, and greedy-decodes the price (R2). Rows that are not
    patched ("clean") pad the last batch and check determinism.
      final: the last input token gets the source's last-token resid
      post:  every token after the limit (identical text), end-aligned
      span:  every limit-digit token gets the mean of the source's limit-digit resids
    """
    import hooks
    blocks = model_io.decoder_layers(model)
    layers = list(range(len(blocks))) if layers is None else [int(x) for x in layers]
    digit_ids = _digit_ids(tok)
    dev = model.device
    out = []
    for it in items:
        t0 = time.time()
        S = _r2_encode(tok, model_io, it["source"]["messages"], it["source"]["limit_text"], it["role"])
        T = _r2_encode(tok, model_io, it["target"]["messages"], it["target"]["limit_text"], it["role"])
        s0 = T["span"][0]
        ns_after, nt_after = len(S["ids"]) - S["span"][-1] - 1, len(T["ids"]) - T["span"][-1] - 1
        if S["span"][0] != s0 or S["ids"][:s0] != T["ids"][:s0] or ns_after != nt_after \
                or S["ids"][S["span"][-1] + 1:] != T["ids"][T["span"][-1] + 1:]:
            raise ValueError(f"{it['item_id']}: source and target differ outside the limit digits")
        # Source resid_post at every layer from the limit to the end (kept on the GPU, bf16).
        src = _capture_all_layers(model, model_io, S["ids"], list(range(S["span"][0], len(S["ids"]))),
                                  device_keep=True)                      # (L, n_src_suffix, H)
        n_span_s = len(S["span"])
        src_final, src_post, src_span_mean = src[:, -1, :], src[:, n_span_s:, :], src[:, :n_span_s, :].mean(1)
        t_suffix = T["ids"][s0:]
        n_span_t = len(T["span"])
        tpos = {"final": torch.tensor([len(t_suffix) - 1]),
                "post": torch.arange(n_span_t, len(t_suffix)),
                "span": torch.arange(0, n_span_t)}
        specs = [("clean", None)] + [(L, p) for p in positions for L in layers]
        while len(specs) % batch:
            specs.append(("clean", None))
        base, cache_how = _batched_prefix_cache(model, model_io, T["ids"][:s0], batch)
        rows = []
        for b0 in range(0, len(specs), batch):
            chunk = specs[b0:b0 + batch]
            patcher = hooks.BatchPatcher(blocks)
            for r, (L, p) in enumerate(chunk):
                if L == "clean":
                    continue
                val = {"final": src_final[L][None, :], "post": src_post[L], "span": src_span_mean[L][None, :]}[p]
                patcher.add(L, r, tpos[p], val)
            cache = copy.deepcopy(base)
            suf = torch.tensor([t_suffix], device=dev).expand(batch, -1).contiguous()
            patcher.arm()
            try:
                o = _forward(model, model_io, input_ids=suf, past_key_values=cache, use_cache=True)
            finally:
                patcher.disarm()
            logits = o.logits[:, -1, :].float()
            probs = torch.softmax(logits, -1)
            p1 = [[round(float(probs[r, i]), 4) for i in digit_ids] if None not in digit_ids else None
                  for r in range(len(chunk))]
            seqs = [[] for _ in chunk]
            done = [False] * len(chunk)
            nxt = logits.argmax(-1)
            for step in range(max_new):
                for r in range(len(chunk)):
                    if done[r]:
                        continue
                    piece = tok.decode([int(nxt[r])])
                    seqs[r].append(piece)
                    is_digit = piece.strip().isdigit()
                    if not is_digit and (step > 0 or piece.strip()):
                        done[r] = True
                if all(done) or step == max_new - 1:
                    break
                o = _forward(model, model_io, input_ids=nxt[:, None], past_key_values=cache, use_cache=True)
                nxt = o.logits[:, -1, :].argmax(-1)
            for r, (L, p) in enumerate(chunk):
                cont = "".join(seqs[r])
                m = re.match(r"\s*(\d+)", cont)
                rows.append({"layer": L, "pos": p, "price": int(m.group(1)) if m else None, "cont": cont[:24],
                             "p1": p1[r]})
            del cache, o
        del base, src
        clean = [r["price"] for r in rows if r["layer"] == "clean"]
        out.append({"item_id": it["item_id"], "fixture": it["fixture"], "role": it["role"],
                    "direction": it["direction"], "base_limit": it["base_limit"],
                    "source_condition": it["source"]["condition"], "target_condition": it["target"]["condition"],
                    "target_limit_units": it["target"]["limit_units"],
                    "n_tokens": {"source": len(S["ids"]), "target": len(T["ids"]), "shared_prefix": s0},
                    "span_tokens": {"source": n_span_s, "target": n_span_t}, "batch": batch, "prefix_cache": cache_how,
                    "clean_prices": clean, "clean_consistent": len(set(clean)) == 1,
                    "rows": [r for r in rows if r["layer"] != "clean"] + [r for r in rows if r["layer"] == "clean"][:1],
                    "secs": round(time.time() - t0, 1)})
    return out


TASKS.update({"e1_capture": e1_capture, "e1_fit": e1_fit, "e2_patch": e2_patch})


# ======================================================================================
# E3 (plan_mechinterp §4): steering along the x1e6 - x1 direction found by E2.
# ======================================================================================
class _StopForward(Exception):
    pass


def _chat_encode(tok, model_io, messages, limit_text, role):
    """The chat prompt (no forced prefix) and its post-limit positions."""
    text, ids_t = model_io.render_messages(tok, messages)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(enc["input_ids"])
    if ids != ids_t[0].tolist():
        raise ValueError("offset tokenization differs from render_messages")
    span = _locate(text, [tuple(o) for o in enc["offset_mapping"]], limit_text, role)
    return {"ids": ids, "span": span, "post": list(range(span[-1] + 1, len(ids)))}


def _layer_acts(model, model_io, ids, layer, positions):
    """resid_post at one layer, at `positions` (float32, on the GPU); stops the forward there."""
    blk = model_io.decoder_layers(model)[layer]
    got = {}
    pos = torch.tensor(positions, device=model.device)

    def hook(_m, _i, out):
        r = out[0] if isinstance(out, tuple) else out
        got["x"] = r[0, pos, :].detach().float()
        raise _StopForward
    h = blk.register_forward_hook(hook)
    try:
        _forward(model, model_io, input_ids=torch.tensor([ids], device=model.device), use_cache=False)
    except _StopForward:
        pass
    finally:
        h.remove()
    return got["x"]


def e3_build(model, tok, items, settings, model_name, model_io, layer=44, **kw):
    """Per-fixture differences d_f = resid(x1e6) - resid(x1) at `layer`, every post-limit position
    (end-aligned), and the non-money contrast u = mean over sentence pairs of the post-number mean
    difference. Kept on the pod; returns norms and cosines only."""
    layer = int(layer)
    d, u_list, resid_norms = {}, [], []
    for it in items:
        if it["kind"] == "pair":
            a = _chat_encode(tok, model_io, it["x1"]["messages"], it["x1"]["limit_text"], it["role"])
            b = _chat_encode(tok, model_io, it["x1e6"]["messages"], it["x1e6"]["limit_text"], it["role"])
            if len(a["post"]) != len(b["post"]) or a["ids"][a["post"][0]:] != b["ids"][b["post"][0]:]:
                raise ValueError(f"{it['fixture']}: text after the limit differs")
            xa = _layer_acts(model, model_io, a["ids"], layer, a["post"])
            xb = _layer_acts(model, model_io, b["ids"], layer, b["post"])
            d[it["fixture"]] = xb - xa
            resid_norms.append(float(xa.norm(dim=-1).mean()))
        else:
            ea, eb = _plain_encode(tok, it["small"], it["small_num"]), _plain_encode(tok, it["big"], it["big_num"])
            pa, pb = list(range(ea["digit"] + 1, len(ea["ids"]))), list(range(eb["digit"] + 1, len(eb["ids"])))
            if len(pa) != len(pb) or not pa:
                raise ValueError(f"{it['item_id']}: text after the number differs")
            u_list.append(_layer_acts(model, model_io, eb["ids"], layer, pb).mean(0)
                          - _layer_acts(model, model_io, ea["ids"], layer, pa).mean(0))
    if len(d) < 2 or not u_list:
        raise ValueError("need at least 2 fixtures and 1 non-money pair")
    D = torch.stack([d[f] for f in sorted(d)])                    # (F, P, H)
    u = torch.stack(u_list).mean(0)
    v_pos = D.mean(0)
    v_mean = v_pos.mean(0)
    _STORE["e3"] = {"layer": layer, "d": d, "sum": D.sum(0), "u": u}
    cos = lambda x, y: float(torch.nn.functional.cosine_similarity(x, y, dim=0))   # noqa: E731
    split = [cos(D[i].mean(0), D[j].mean(0)) for i in range(len(d)) for j in range(i + 1, len(d))]
    return [{"layer": layer, "fixtures": sorted(d), "n_post": int(D.shape[1]), "n_nonmoney": len(u_list),
             "norm_v_pos_mean": float(v_pos.norm(dim=-1).mean()), "norm_v_mean": float(v_mean.norm()),
             "norm_u": float(u.norm()), "mean_resid_norm_post": sum(resid_norms) / len(resid_norms),
             "cos_v_mean_u": cos(v_mean, u), "cos_fixture_dirs_mean": sum(split) / len(split),
             "cos_fixture_dirs_min": min(split)}]


def e3_steer(model, tok, items, settings, model_name, model_io, alphas=(0.25, 0.5, 1.0, 1.5, 2.0),
             vectors=("v_pos", "v_mean", "random", "nonmoney"), samples=2, batch=16, schema=None,
             max_new=192, loo=True, **kw):
    """Steered openings. Item: one fixture and direction ("x1+v": the x1 prompt plus alpha*v;
    "x1e6-v": the x1e6 prompt minus alpha*v). v is built without the item's own fixture (loo).
      v_pos    per-position mean difference (P, H)
      v_mean   one direction (its mean over positions), at every post-limit position
      random   Gaussian per position (fixed seed)
      nonmoney one direction: the non-money magnitude contrast, at every post-limit position
    All four have the same size: v_pos's per-position norms (random) or their mean (the single
    directions), so a weaker vector cannot be explained by its length.
    Rows (vector, alpha, sample) share the prompt; the prefix before the limit is cached once.
    Sampling: the frozen settings, the JSON schema via xgrammar, one seed per batch."""
    import game_engine as ge
    import hooks
    st_ = _STORE.get("e3")
    if not st_:
        raise ValueError("run e3_build first")
    L = st_["layer"]
    blk = model_io.decoder_layers(model)[L]
    samp = settings["sampling"][model_name]
    max_chars = settings["max_message_chars"]["value"]
    eos = _eos_ids(model, tok)
    try:
        vocab = model.get_output_embeddings().weight.shape[0]
    except AttributeError:
        vocab = getattr(model.config, "text_config", model.config).vocab_size
    dev = model.device
    out = []
    for it in items:
        t0 = time.time()
        sign = 1.0 if it["direction"] == "x1+v" else -1.0
        E = _chat_encode(tok, model_io, it["target"]["messages"], it["target"]["limit_text"], it["role"])
        f = it["fixture"]
        n_f = len(st_["d"])
        use_loo = loo and f in st_["d"] and n_f >= 3
        v_pos = (st_["sum"] - st_["d"][f]) / (n_f - 1) if use_loo else st_["sum"] / n_f
        if v_pos.shape[0] != len(E["post"]):
            raise ValueError("post-limit length differs from the vectors'")
        size = v_pos.norm(dim=-1).mean()
        v_mean = v_pos.mean(0)
        v_mean = v_mean / v_mean.norm() * size
        g = torch.Generator(device="cpu").manual_seed(12345)
        rnd = torch.randn(v_pos.shape, generator=g).to(v_pos.device)
        rnd = rnd / rnd.norm(dim=-1, keepdim=True) * v_pos.norm(dim=-1, keepdim=True)
        u = st_["u"] / st_["u"].norm() * size
        V = {"v_pos": v_pos, "v_mean": v_mean.expand_as(v_pos), "random": rnd, "nonmoney": u.expand_as(v_pos)}
        specs = [(vec, float(a), s) for vec in vectors for a in alphas for s in range(samples)]
        n0 = 0
        while n0 < 2 * samples or (len(specs) + n0) % batch:
            n0 += 1
        # Unsteered first, then by strength: high-alpha rows (most likely to run long) share batches.
        specs = [("none", 0.0, s) for s in range(n0)] + sorted(specs, key=lambda x: (x[1], vectors.index(x[0]), x[2]))
        s0 = E["span"][0]
        post_rel = torch.tensor([p - s0 for p in E["post"]])
        base, how = _batched_prefix_cache(model, model_io, E["ids"][:s0], batch)
        full = torch.tensor([E["ids"]], device=dev).expand(batch, -1).contiguous()
        rows, fired = [], []
        for b0 in range(0, len(specs), batch):
            chunk = specs[b0:b0 + batch]
            delta = torch.stack([torch.zeros_like(v_pos) if vec == "none" else sign * a * V[vec]
                                 for vec, a, _s in chunk])
            adder = hooks.PrefillAdder(blk, post_rel, delta, len(E["ids"]) - s0)
            lp, gram = (None, "no schema") if schema is None else _grammar_processor(tok, schema, vocab)
            seed = int(it["seed"]) + b0
            torch.manual_seed(seed)
            kwargs = {"max_new_tokens": int(max_new), "attention_mask": torch.ones_like(full),
                      "past_key_values": copy.deepcopy(base), "do_sample": True,
                      "temperature": samp["temperature"], "top_p": samp["top_p"], "top_k": samp["top_k"]}
            if lp is not None:
                kwargs["logits_processor"] = [lp]
            try:
                with torch.no_grad(), model_io.attn_ctx():
                    gen = model.generate(full, **kwargs)
            finally:
                adder.remove()
            fired.append(adder.fired)
            for r, (vec, a, s) in enumerate(chunk):
                new = gen[r, full.shape[1]:].tolist()
                cut = next((i for i, t in enumerate(new) if t in eos), None)
                finish = "stop" if cut is not None else "length"
                new = new[:cut] if cut is not None else new
                content = tok.decode(new, skip_special_tokens=True)
                move, errors = ge.validate_reply(content, finish, False, max_chars)
                rows.append({"vector": vec, "alpha": a, "sample": s, "seed": seed, "finish": finish,
                             "completion_tokens": len(new), "errors": errors,
                             "action": move and move["action"], "price": move and move.get("price"),
                             "content": content[:600], "constrained": lp is not None})
            del gen
        del base
        out.append({"item_id": it["item_id"], "fixture": f, "role": it["role"], "direction": it["direction"],
                    "base_limit": it["base_limit"], "target_condition": it["target"]["condition"],
                    "target_limit_units": it["target"]["limit_units"], "layer": L, "loo": use_loo,
                    "n_post": len(E["post"]), "prefix_cache": how, "prefill_fired": fired, "grammar": gram,
                    "norm_v_pos_mean": float(v_pos.norm(dim=-1).mean()), "norm_v_mean": float(v_mean.norm()),
                    "rows": rows, "secs": round(time.time() - t0, 1)})
    return out


TASKS.update({"e3_build": e3_build, "e3_steer": e3_steer})


# ======================================================================================
# E3b: steering that covers the reply's own opening tokens (the gap E3 left; X53).
# ======================================================================================
def e3b_build(model, tok, items, settings, model_name, model_io, layer=44, **kw):
    """d_f = resid(x1e6) - resid(x1) at `layer` on every token after the limit in the R2 input:
    the prompt's tokens (n_prompt) and then the reply's forced opening '{"action": "OFFER",
    "price": ' (n_reply). Plus the non-money contrast. Kept on the pod; returns norms/cosines."""
    layer = int(layer)
    d, u_list, n_prompt, reply_ids = {}, [], None, None
    for it in items:
        if it["kind"] == "pair":
            enc = {}
            for c in ("x1", "x1e6"):
                r = _r2_encode(tok, model_io, it[c]["messages"], it[c]["limit_text"], it["role"])
                ch = _chat_encode(tok, model_io, it[c]["messages"], it[c]["limit_text"], it["role"])
                if r["ids"][:len(ch["ids"])] != ch["ids"]:
                    raise ValueError("the chat prompt is not a prefix of the R2 input")
                post = list(range(r["span"][-1] + 1, len(r["ids"])))
                enc[c] = (r, post, len(ch["post"]), r["ids"][len(ch["ids"]):])
            (ra, pa, npa, rep_a), (rb, pb, npb, rep_b) = enc["x1"], enc["x1e6"]
            if len(pa) != len(pb) or npa != npb or rep_a != rep_b or ra["ids"][pa[0]:] != rb["ids"][pb[0]:]:
                raise ValueError(f"{it['fixture']}: text after the limit differs")
            n_prompt, reply_ids = npa, rep_a
            d[it["fixture"]] = _layer_acts(model, model_io, rb["ids"], layer, pb) - \
                _layer_acts(model, model_io, ra["ids"], layer, pa)
        else:
            ea, eb = _plain_encode(tok, it["small"], it["small_num"]), _plain_encode(tok, it["big"], it["big_num"])
            pa, pb = list(range(ea["digit"] + 1, len(ea["ids"]))), list(range(eb["digit"] + 1, len(eb["ids"])))
            if len(pa) != len(pb) or not pa:
                raise ValueError(f"{it['item_id']}: text after the number differs")
            u_list.append(_layer_acts(model, model_io, eb["ids"], layer, pb).mean(0)
                          - _layer_acts(model, model_io, ea["ids"], layer, pa).mean(0))
    if len(d) < 2 or not u_list:
        raise ValueError("need at least 2 fixtures and 1 non-money pair")
    D = torch.stack([d[f] for f in sorted(d)])
    u = torch.stack(u_list).mean(0)
    _STORE["e3b"] = {"layer": layer, "d": d, "sum": D.sum(0), "u": u, "n_prompt": n_prompt,
                     "reply_ids": reply_ids}
    v = D.mean(0)
    cos = lambda x, y: float(torch.nn.functional.cosine_similarity(x, y, dim=0))   # noqa: E731
    split = [cos(D[i].flatten(), D[j].flatten()) for i in range(len(d)) for j in range(i + 1, len(d))]
    return [{"layer": layer, "n_post": int(D.shape[1]), "n_prompt": n_prompt, "n_reply": len(reply_ids),
             "reply_tokens": [tok.decode([t]) for t in reply_ids], "n_nonmoney": len(u_list),
             "norm_v_prompt": float(v[:n_prompt].norm(dim=-1).mean()), "norm_v_reply": float(v[n_prompt:].norm(dim=-1).mean()),
             "cos_v_mean_u": cos(v.mean(0), u), "cos_fixture_diffs_mean": sum(split) / len(split),
             "cos_fixture_diffs_min": min(split)}]


def _e3b_vectors(st_, fixture, loo=True):
    """The steering vectors for one fixture, all (P, H) over the post-limit tokens (prompt+reply)."""
    n_f = len(st_["d"])
    own = st_["d"][fixture]
    v_pos = (st_["sum"] - own) / (n_f - 1) if loo and n_f >= 3 else st_["sum"] / n_f
    size = v_pos.norm(dim=-1).mean()
    v_mean = v_pos.mean(0)
    v_mean = v_mean / v_mean.norm() * size
    g = torch.Generator(device="cpu").manual_seed(12345)
    rnd = torch.randn(v_pos.shape, generator=g).to(v_pos.device)
    rnd = rnd / rnd.norm(dim=-1, keepdim=True) * v_pos.norm(dim=-1, keepdim=True)
    u = st_["u"] / st_["u"].norm() * size
    return {"own": own, "v_pos": v_pos, "v_mean": v_mean.expand_as(v_pos), "random": rnd,
            "nonmoney": u.expand_as(v_pos)}, loo and n_f >= 3


def _scope_mask(scope, n_prompt, n_post, device):
    m = torch.zeros(n_post, 1, device=device)
    if scope in ("all", "prompt", "prompt+reply", "all_gen"):
        m[:n_prompt] = 1
    if scope in ("all", "reply", "prompt+reply", "all_gen"):
        m[n_prompt:] = 1
    return m


E3B_BRIDGE_ROWS = (("none", "all", 0.0), ("own", "all", 1.0), ("own", "prompt", 1.0), ("own", "reply", 1.0),
                   ("v_pos", "all", 0.5), ("v_pos", "all", 1.0), ("v_pos", "all", 2.0), ("v_pos", "prompt", 1.0),
                   ("v_pos", "reply", 1.0), ("v_mean", "all", 1.0), ("v_mean", "all", 2.0),
                   ("random", "all", 1.0), ("random", "all", 2.0), ("nonmoney", "all", 1.0), ("nonmoney", "all", 2.0))


def e3b_bridge(model, tok, items, settings, model_name, model_io, rows=E3B_BRIDGE_ROWS, batch=16, max_new=12,
               loo=True, **kw):
    """Part A, E2's exact setup (R2: the reply's opening is forced; greedy price): add sign*alpha*V
    at layer L on the prompt tokens, the reply tokens, or both. own x 1 on 'all' reproduces E2's
    swap at this layer; the rest asks whether an average direction does the same, and where."""
    import hooks
    st_ = _STORE.get("e3b")
    if not st_:
        raise ValueError("run e3b_build first")
    blk = model_io.decoder_layers(model)[st_["layer"]]
    digit_ids = _digit_ids(tok)
    dev = model.device
    out = []
    for it in items:
        t0 = time.time()
        sign = 1.0 if it["direction"] == "x1+v" else -1.0
        T = _r2_encode(tok, model_io, it["target"]["messages"], it["target"]["limit_text"], it["role"])
        post = list(range(T["span"][-1] + 1, len(T["ids"])))
        V, used_loo = _e3b_vectors(st_, it["fixture"], loo)
        if V["v_pos"].shape[0] != len(post):
            raise ValueError("post-limit length differs from the vectors'")
        specs = [tuple(r) for r in rows]
        while len(specs) % batch:
            specs.append(("none", "all", 0.0))
        s0 = T["span"][0]
        rel = torch.tensor([p - s0 for p in post])
        suf = T["ids"][s0:]
        res = []
        for b0 in range(0, len(specs), batch):
            chunk = specs[b0:b0 + batch]
            base, how = _batched_prefix_cache(model, model_io, T["ids"][:s0], len(chunk))
            delta = torch.stack([torch.zeros_like(V["v_pos"]) if vec == "none" else
                                 sign * float(a) * V[vec] * _scope_mask(sc, st_["n_prompt"], len(post), dev)
                                 for vec, sc, a in chunk])
            adder = hooks.PrefillAdder(blk, rel, delta, len(suf))
            try:
                o = _forward(model, model_io, input_ids=torch.tensor([suf], device=dev).expand(len(chunk), -1).contiguous(),
                             past_key_values=base, use_cache=True)
            finally:
                adder.remove()
            logits = o.logits[:, -1, :].float()
            probs = torch.softmax(logits, -1)
            seqs, done = [[] for _ in chunk], [False] * len(chunk)
            nxt = logits.argmax(-1)
            for step in range(max_new):
                for r in range(len(chunk)):
                    if done[r]:
                        continue
                    piece = tok.decode([int(nxt[r])])
                    seqs[r].append(piece)
                    if not piece.strip().isdigit() and (step > 0 or piece.strip()):
                        done[r] = True
                if all(done) or step == max_new - 1:
                    break
                o = _forward(model, model_io, input_ids=nxt[:, None], past_key_values=base, use_cache=True)
                nxt = o.logits[:, -1, :].argmax(-1)
            for r, (vec, sc, a) in enumerate(chunk):
                cont = "".join(seqs[r])
                m = re.match(r"\s*(\d+)", cont)
                res.append({"vector": vec, "scope": sc, "alpha": float(a), "price": int(m.group(1)) if m else None,
                            "cont": cont[:24],
                            "p1": [round(float(probs[r, i]), 4) for i in digit_ids] if None not in digit_ids else None})
            del base, o
        clean = [x["price"] for x in res if x["vector"] == "none"]
        out.append({"item_id": it["item_id"], "fixture": it["fixture"], "role": it["role"], "direction": it["direction"],
                    "base_limit": it["base_limit"], "target_condition": it["target"]["condition"],
                    "layer": st_["layer"], "loo": used_loo, "n_post": len(post), "n_prompt": st_["n_prompt"],
                    "clean_consistent": len(set(clean)) == 1, "rows": res, "secs": round(time.time() - t0, 1)})
    return out


E3B_STEER_ROWS = (("v_pos", "prompt+reply", (0.5, 1.0, 1.5, 2.0), 3), ("own", "prompt+reply", (1.0,), 4),
                  ("v_pos", "reply", (1.0, 2.0), 2), ("v_mean", "all_gen", (1.0, 2.0), 2),
                  ("random", "prompt+reply", (1.0, 2.0), 2), ("nonmoney", "prompt+reply", (1.0, 2.0), 2))


def e3b_steer(model, tok, items, settings, model_name, model_io, plan=E3B_STEER_ROWS, batch=16, schema=None,
              max_new=192, loo=True, **kw):
    """Part B, real sampled openings (frozen settings, xgrammar), steered on the prompt's tokens
    after the limit AND on the reply as it is generated: generated token t gets V[n_prompt + t]
    for the reply's 11 opening tokens; after them, nothing ('prompt+reply', 'reply') or the
    single direction on every further token ('all_gen', v_mean only). Each row records whether
    its generated opening matched the expected tokens (steering is by token index)."""
    import game_engine as ge
    import hooks
    st_ = _STORE.get("e3b")
    if not st_:
        raise ValueError("run e3b_build first")
    blk = model_io.decoder_layers(model)[st_["layer"]]
    n_prompt, reply_ids = st_["n_prompt"], st_["reply_ids"]
    samp = settings["sampling"][model_name]
    max_chars = settings["max_message_chars"]["value"]
    eos = _eos_ids(model, tok)
    try:
        vocab = model.get_output_embeddings().weight.shape[0]
    except AttributeError:
        vocab = getattr(model.config, "text_config", model.config).vocab_size
    dev = model.device
    out = []
    for it in items:
        t0 = time.time()
        sign = 1.0 if it["direction"] == "x1+v" else -1.0
        E = _chat_encode(tok, model_io, it["target"]["messages"], it["target"]["limit_text"], it["role"])
        if len(E["post"]) != n_prompt:
            raise ValueError("prompt length after the limit differs from the vectors'")
        V, used_loo = _e3b_vectors(st_, it["fixture"], loo)
        specs = [(vec, sc, float(a), s) for vec, sc, alphas, n in plan for a in alphas for s in range(n)]
        n0 = 0
        while n0 < 8 or (len(specs) + n0) % batch:
            n0 += 1
        specs = [("none", "prompt+reply", 0.0, s) for s in range(n0)] + sorted(specs, key=lambda x: (x[2], x[0], x[1], x[3]))
        s0 = E["span"][0]
        rel = torch.tensor([p - s0 for p in E["post"]])
        base, how = _batched_prefix_cache(model, model_io, E["ids"][:s0], batch)
        full = torch.tensor([E["ids"]], device=dev).expand(batch, -1).contiguous()
        rows, calls = [], []
        for b0 in range(0, len(specs), batch):
            chunk = specs[b0:b0 + batch]
            pd, gd, tl = [], [], []
            for vec, sc, a, _s in chunk:
                if vec == "none":
                    z = torch.zeros_like(V["v_pos"])
                    pd.append(z[:n_prompt]); gd.append(z[n_prompt:]); tl.append(z[0])
                    continue
                full_v = sign * a * V[vec] * _scope_mask(sc, n_prompt, V["v_pos"].shape[0], dev)
                pd.append(full_v[:n_prompt]); gd.append(full_v[n_prompt:])
                tl.append(full_v[0] if sc == "all_gen" else torch.zeros_like(full_v[0]))
            adder = hooks.GenAdder(blk, rel, torch.stack(pd), len(E["ids"]) - s0, torch.stack(gd), torch.stack(tl))
            lp, gram = (None, "no schema") if schema is None else _grammar_processor(tok, schema, vocab)
            seed = int(it["seed"]) + b0
            torch.manual_seed(seed)
            kwargs = {"max_new_tokens": int(max_new), "attention_mask": torch.ones_like(full),
                      "past_key_values": copy.deepcopy(base), "do_sample": True,
                      "temperature": samp["temperature"], "top_p": samp["top_p"], "top_k": samp["top_k"]}
            if lp is not None:
                kwargs["logits_processor"] = [lp]
            try:
                with torch.no_grad(), model_io.attn_ctx():
                    gen = model.generate(full, **kwargs)
            finally:
                adder.remove()
            calls.append([adder.calls[0] if adder.calls else None, len(adder.calls)])
            for r, (vec, sc, a, s) in enumerate(chunk):
                new = gen[r, full.shape[1]:].tolist()
                cut = next((i for i, t in enumerate(new) if t in eos), None)
                finish = "stop" if cut is not None else "length"
                new = new[:cut] if cut is not None else new
                content = tok.decode(new, skip_special_tokens=True)
                move, errors = ge.validate_reply(content, finish, False, max_chars)
                rows.append({"vector": vec, "scope": sc, "alpha": a, "sample": s, "seed": seed, "finish": finish,
                             "aligned": new[:len(reply_ids)] == reply_ids, "completion_tokens": len(new),
                             "errors": errors, "action": move and move["action"],
                             "price": move and move.get("price"), "content": content[:600],
                             "constrained": lp is not None})
            del gen
        del base
        out.append({"item_id": it["item_id"], "fixture": it["fixture"], "role": it["role"], "direction": it["direction"],
                    "base_limit": it["base_limit"], "target_condition": it["target"]["condition"],
                    "layer": st_["layer"], "loo": used_loo, "n_prompt": n_prompt, "prefix_cache": how,
                    "hook_calls": calls, "grammar": gram, "rows": rows, "secs": round(time.time() - t0, 1)})
    return out


TASKS.update({"e3b_build": e3b_build, "e3b_bridge": e3b_bridge, "e3b_steer": e3b_steer})


# --- plug-ins: any tasks_<name>.py next to this file adds its TASKS (so parallel work needs no edits here).
#     A plug-in may `import tasks` to reuse the helpers above. Name clashes are an error.
def _load_plugins():
    import glob
    import importlib.util
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    for path in sorted(glob.glob(os.path.join(here, "tasks_*.py"))):
        name = os.path.basename(path)[:-3]
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        clash = set(getattr(mod, "TASKS", {})) & set(TASKS)
        if clash:
            raise ValueError(f"{name}: task names already taken: {sorted(clash)}")
        TASKS.update(getattr(mod, "TASKS", {}))


_load_plugins()
