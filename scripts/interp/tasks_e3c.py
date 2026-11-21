#!/usr/bin/env python3
"""tasks_e3c.py (POD-SIDE plug-in): E3c, the seller layer sweep of E3b's steering (config/interp/plan_next.md,
Session 1). Exploratory.

Question: at which layers does the shared x1e6 - x1 direction steer the opening, and does E3b's
prompt-vs-reply split hold at every layer?

  e3c_build   d_f = resid(x1e6) - resid(x1) at EVERY layer on the 117 tokens after the limit in the R2 input
              (106 prompt tokens + the reply's forced opening '{"action": "OFFER", "price": '), for each
              fixture pair, plus the non-money contrast at every layer. Kept on the pod (the swept layers in
              the model's dtype); returns per-layer norms and cosines only.
  e3c_bridge  Part A, E2's setup (opening forced, greedy price), the E3b bridge at many layers at once: rows
              of one batch steer DIFFERENT layers (one PrefillAdder per layer, zero delta for the other rows).
  e3c_steer   Part B, real sampled openings (frozen settings, xgrammar), steered on the prompt as it is read
              and on the reply's 11 opening tokens as they are generated, at a few layers chosen from Part A
              (one GenAdder per layer, zero delta for the other rows).

Vectors are E3b's: leave-one-fixture-out per-token mean ('v_pos'), the non-money contrast at the same
mean size ('nonmoney'), and the fixture's own difference ('own', used only by the exactness checks:
own x 1 on every post-limit token equals E2's swap at that layer, which X52 already measured).
"""
import copy
import math
import re
import time

import torch                                   # pod-only

import hooks
import tasks as T

_STORE = {}


# --- multi-layer adders --------------------------------------------------------------------------------
def multi_prefill(blocks, row_layers, positions, row_deltas, seq_len):
    """Rows of one batch steer different layers. row_layers[r]: a layer index or None (unsteered);
    row_deltas[r]: (P, H) or None. One hooks.PrefillAdder per layer used, with zeros for the other rows
    (adding an exact zero leaves a row's activations bit-identical). Returns the adders; remove them all."""
    ref = next(d for d in row_deltas if d is not None) if any(d is not None for d in row_deltas) else None
    adders = []
    for L in sorted({x for x in row_layers if x is not None}):
        delta = torch.zeros((len(row_layers),) + tuple(ref.shape), dtype=ref.dtype, device=ref.device)
        for r, (lr, d) in enumerate(zip(row_layers, row_deltas)):
            if lr == L:
                delta[r] = d
        adders.append(hooks.PrefillAdder(blocks[L], positions, delta, seq_len))
    return adders


def multi_gen(blocks, row_layers, positions, row_full, n_prompt, seq_len):
    """Part B's version: row_full[r] is the (P, H) delta over prompt + reply-opening tokens (or None).
    One hooks.GenAdder per layer: prompt part on the prefill, reply part on generated tokens 0..n_reply-1,
    then nothing (tail zeros)."""
    ref = next((d for d in row_full if d is not None), None)
    adders = []
    for L in sorted({x for x in row_layers if x is not None}):
        full = torch.zeros((len(row_layers),) + tuple(ref.shape), dtype=ref.dtype, device=ref.device)
        for r, (lr, d) in enumerate(zip(row_layers, row_full)):
            if lr == L:
                full[r] = d
        adders.append(hooks.GenAdder(blocks[L], positions, full[:, :n_prompt], seq_len, full[:, n_prompt:],
                                     torch.zeros_like(full[:, 0])))
    return adders


def _remove(adders):
    for ad in adders:
        ad.remove()


# --- directions at every layer --------------------------------------------------------------------------
def e3c_build(model, tok, items, settings, model_name, model_io, layers=None, **kw):
    """Per-fixture differences at every layer; the swept `layers` are stored (model dtype), the rest
    only enter the per-layer readouts (|v|, activation norms, fixture agreement, non-money cosine)."""
    blocks = model_io.decoder_layers(model)
    n_layers = len(blocks)
    keep = sorted({int(x) for x in (range(n_layers) if layers is None else layers)})
    if not keep or keep[0] < 0 or keep[-1] >= n_layers:
        raise ValueError(f"layers must lie in 0..{n_layers - 1}")
    kidx = torch.tensor(keep)
    dt = next(model.parameters()).dtype
    d, u_list = {}, []
    n_prompt = reply_ids = None
    s_keep = v_all = unit = rn_p = rn_r = None
    t0 = time.time()
    for it in items:
        if it["kind"] == "pair":
            enc = {}
            for c in ("x1", "x1e6"):
                r = T._r2_encode(tok, model_io, it[c]["messages"], it[c]["limit_text"], it["role"])
                ch = T._chat_encode(tok, model_io, it[c]["messages"], it[c]["limit_text"], it["role"])
                if r["ids"][:len(ch["ids"])] != ch["ids"]:
                    raise ValueError("the chat prompt is not a prefix of the R2 input")
                post = list(range(r["span"][-1] + 1, len(r["ids"])))
                enc[c] = (r, post, len(ch["post"]), r["ids"][len(ch["ids"]):])
            (ra, pa, npa, rep_a), (rb, pb, npb, rep_b) = enc["x1"], enc["x1e6"]
            if len(pa) != len(pb) or npa != npb or rep_a != rep_b or ra["ids"][pa[0]:] != rb["ids"][pb[0]:]:
                raise ValueError(f"{it['fixture']}: text after the limit differs")
            if n_prompt is not None and (npa != n_prompt or rep_a != reply_ids):
                raise ValueError(f"{it['fixture']}: post-limit length differs between fixtures")
            n_prompt, reply_ids = npa, rep_a
            xa = T._capture_all_layers(model, model_io, ra["ids"], pa, device_keep=True).float()   # (L, P, H)
            xb = T._capture_all_layers(model, model_io, rb["ids"], pb, device_keep=True).float()
            df = xb - xa
            stored = df[kidx.to(df.device)].to(dt)
            d[it["fixture"]] = stored
            flat = df.flatten(1)
            flat = flat / flat.norm(dim=1, keepdim=True).clamp_min(1e-30)
            if v_all is None:
                s_keep, v_all, unit = torch.zeros_like(stored, dtype=torch.float32), torch.zeros_like(df), torch.zeros_like(flat)
                rn_p = torch.zeros(n_layers, device=df.device)
                rn_r = torch.zeros(n_layers, device=df.device)
            s_keep += stored.float()             # the sum of what is stored, so leave-one-out is exact
            v_all += df
            unit += flat
            rn_p += xa[:, :npa].norm(dim=-1).mean(1)
            rn_r += xa[:, npa:].norm(dim=-1).mean(1)
            del xa, xb, df, flat
        else:
            ea, eb = T._plain_encode(tok, it["small"], it["small_num"]), T._plain_encode(tok, it["big"], it["big_num"])
            pa, pb = list(range(ea["digit"] + 1, len(ea["ids"]))), list(range(eb["digit"] + 1, len(eb["ids"])))
            if len(pa) != len(pb) or not pa:
                raise ValueError(f"{it['item_id']}: text after the number differs")
            u_list.append(T._capture_all_layers(model, model_io, eb["ids"], pb, device_keep=True).float().mean(1)
                          - T._capture_all_layers(model, model_io, ea["ids"], pa, device_keep=True).float().mean(1))
    n = len(d)
    if n < 3 or not u_list:
        raise ValueError("need at least 3 fixtures (leave-one-out) and 1 non-money pair")
    u = torch.stack(u_list).mean(0)                                        # (L, H)
    _STORE["e3c"] = {"kept": keep, "index": {L: i for i, L in enumerate(keep)}, "d": d, "sum": s_keep, "u": u,
                     "n_prompt": n_prompt, "reply_ids": reply_ids, "n_post": int(s_keep.shape[1])}
    v = v_all / n
    cos = torch.nn.functional.cosine_similarity
    r4 = lambda t: [round(float(x), 4) for x in t]                         # noqa: E731
    per = {"norm_v_prompt": r4(v[:, :n_prompt].norm(dim=-1).mean(1)), "norm_v_reply": r4(v[:, n_prompt:].norm(dim=-1).mean(1)),
           "resid_norm_prompt": r4(rn_p / n), "resid_norm_reply": r4(rn_r / n),
           "cos_fixture_diffs_mean": r4((unit.pow(2).sum(1) - n) / (n * (n - 1))),
           "cos_v_mean_u": r4(cos(v.mean(1), u, dim=-1)), "norm_u": r4(u.norm(dim=-1))}
    stored_bytes = sum(x.numel() * x.element_size() for x in d.values()) + s_keep.numel() * 4
    return [{"layers": n_layers, "kept": keep, "n_pairs": n, "n_post": int(s_keep.shape[1]), "n_prompt": n_prompt,
             "n_reply": len(reply_ids), "reply_tokens": [tok.decode([t]) for t in reply_ids], "n_nonmoney": len(u_list),
             "per_layer": per, "stored_gib": round(stored_bytes / 2 ** 30, 3), "secs": round(time.time() - t0, 1)}]


def _vectors(st_, fixture, L, loo=True):
    """E3b's vectors at layer L, all (P, H): own difference, leave-one-out per-token mean, non-money."""
    i = st_["index"][L]
    n_f = len(st_["d"])
    own = st_["d"][fixture][i].float()
    use_loo = loo and n_f >= 3
    v = (st_["sum"][i] - own) / (n_f - 1) if use_loo else st_["sum"][i] / n_f
    size = v.norm(dim=-1).mean()
    u = st_["u"][L]
    u = u / u.norm() * size
    return {"own": own, "v_pos": v, "nonmoney": u.expand_as(v)}, use_loo


# --- Part A: forced opening, greedy price, many layers per batch ---------------------------------------
E3C_A_ROWS = (("v_pos", "all", 1.0), ("v_pos", "reply", 1.0), ("v_pos", "prompt", 1.0), ("nonmoney", "all", 1.0))


def e3c_bridge(model, tok, items, settings, model_name, model_io, layers=None, rows=E3C_A_ROWS, batch=16,
               max_new=12, loo=True, debug=False, **kw):
    """Item: one fixture and direction. Specs: one clean row, then (layer, vector, scope, alpha) layer-major,
    padded with clean rows to a multiple of `batch`. The shared prefix's KV cache is computed once and
    copied per batch (as in e2_patch). debug: keep each batch's first-step logits in _STORE (tests)."""
    st_ = _STORE.get("e3c")
    if not st_:
        raise ValueError("run e3c_build first")
    layers = st_["kept"] if layers is None else [int(x) for x in layers]
    missing = [L for L in layers if L not in st_["index"]]
    if missing:
        raise ValueError(f"layers not stored by e3c_build: {missing}")
    blocks = model_io.decoder_layers(model)
    digit_ids = T._digit_ids(tok)
    dev = model.device
    if debug:
        _STORE["debug_logits"] = []
    out = []
    for it in items:
        t0 = time.time()
        sign = 1.0 if it["direction"] == "x1+v" else -1.0
        E = T._r2_encode(tok, model_io, it["target"]["messages"], it["target"]["limit_text"], it["role"])
        post = list(range(E["span"][-1] + 1, len(E["ids"])))
        if len(post) != st_["n_post"]:
            raise ValueError("post-limit length differs from the vectors'")
        specs = [(None, "none", "all", 0.0)] + [(L, vec, sc, float(a)) for L in layers for vec, sc, a in rows]
        while len(specs) % batch:
            specs.append((None, "none", "all", 0.0))
        s0 = E["span"][0]
        rel = torch.tensor([p - s0 for p in post])
        suf = E["ids"][s0:]
        base, how = T._batched_prefix_cache(model, model_io, E["ids"][:s0], batch)
        vcache, used_loo, res = {}, None, []
        for b0 in range(0, len(specs), batch):
            chunk = specs[b0:b0 + batch]
            row_layers, row_deltas = [], []
            for L, vec, sc, a in chunk:
                if vec == "none":
                    row_layers.append(None); row_deltas.append(None)
                    continue
                if L not in vcache:
                    vcache[L] = _vectors(st_, it["fixture"], L, loo)
                V, used_loo = vcache[L]
                row_layers.append(L)
                row_deltas.append(sign * a * V[vec] * T._scope_mask(sc, st_["n_prompt"], len(post), dev))
            adders = multi_prefill(blocks, row_layers, rel, row_deltas, len(suf))
            cache = copy.deepcopy(base)
            try:
                o = T._forward(model, model_io, input_ids=torch.tensor([suf], device=dev).expand(len(chunk), -1).contiguous(),
                               past_key_values=cache, use_cache=True)
            finally:
                _remove(adders)
            if any(len(ad.fired) != 1 for ad in adders):
                raise RuntimeError("a layer's steering did not fire exactly once on the prefill")
            logits = o.logits[:, -1, :].float()
            if debug:
                _STORE["debug_logits"].append(logits.detach().cpu())
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
                o = T._forward(model, model_io, input_ids=nxt[:, None], past_key_values=cache, use_cache=True)
                nxt = o.logits[:, -1, :].argmax(-1)
            for r, (L, vec, sc, a) in enumerate(chunk):
                cont = "".join(seqs[r])
                m = re.match(r"\s*(\d+)", cont)
                res.append({"layer": L, "vector": vec, "scope": sc, "alpha": a, "price": int(m.group(1)) if m else None,
                            "cont": cont[:24],
                            "p1": [round(float(probs[r, i]), 4) for i in digit_ids] if None not in digit_ids else None})
            del cache, o
        del base, vcache
        clean = [x["price"] for x in res if x["vector"] == "none"]
        out.append({"item_id": it["item_id"], "fixture": it["fixture"], "role": it["role"], "direction": it["direction"],
                    "base_limit": it["base_limit"], "target_condition": it["target"]["condition"], "layers": layers,
                    "loo": used_loo, "n_post": len(post), "n_prompt": st_["n_prompt"], "prefix_cache": how,
                    "clean_consistent": len(set(clean)) == 1,
                    "rows": [x for x in res if x["vector"] != "none"] + [x for x in res if x["vector"] == "none"][:1],
                    "n_clean": len(clean), "secs": round(time.time() - t0, 1)})
    return out


# --- Part B: real sampled openings at a few layers ------------------------------------------------------
def b_batches(layers, alphas, samples, batch, min_none=8):
    """Batches of (layer, alpha, sample) rows for one item; None = unsteered. Steered rows sorted by alpha,
    then layer, are split evenly over the fewest batches that also hold >= min_none unsteered rows in total;
    every batch is topped up with unsteered rows (so each batch has a baseline under its own seed, and the
    strongest rows, most likely to run long, share a batch)."""
    steered = sorted(((int(L), float(a), s) for a in alphas for L in layers for s in range(samples)),
                     key=lambda x: (x[1], x[0], x[2]))
    nb = max(1, math.ceil((len(steered) + min_none) / batch))
    q, extra = divmod(len(steered), nb)
    out, i = [], 0
    for k in range(nb):
        part = steered[i:i + q + (k < extra)]
        i += len(part)
        if len(part) > batch:
            raise ValueError("too many steered rows for the batch size")
        out.append([None] * (batch - len(part)) + part)
    return out


def e3c_steer(model, tok, items, settings, model_name, model_io, layers=(), alphas=(1.0, 2.0), samples=3, batch=16,
              min_none=8, schema=None, max_new=192, loo=True, greedy=False, **kw):
    """Rows: the shared direction (v_pos, leave-one-out) on prompt + reply opening at each layer in `layers`
    x alpha x sample, plus unsteered rows. Each batch is seeded with the item's seed + its first row index,
    as in E3b. greedy=True is for the exactness checks only."""
    import game_engine as ge
    st_ = _STORE.get("e3c")
    if not st_:
        raise ValueError("run e3c_build first")
    layers = [int(x) for x in layers]
    if not layers:
        raise ValueError("no layers chosen for Part B")
    missing = [L for L in layers if L not in st_["index"]]
    if missing:
        raise ValueError(f"layers not stored by e3c_build: {missing}")
    blocks = model_io.decoder_layers(model)
    n_prompt, reply_ids = st_["n_prompt"], st_["reply_ids"]
    samp = settings["sampling"][model_name]
    max_chars = settings["max_message_chars"]["value"]
    eos = T._eos_ids(model, tok)
    try:
        vocab = model.get_output_embeddings().weight.shape[0]
    except AttributeError:
        vocab = getattr(model.config, "text_config", model.config).vocab_size
    dev = model.device
    plan = b_batches(layers, alphas, samples, batch, min_none)
    out = []
    for it in items:
        t0 = time.time()
        sign = 1.0 if it["direction"] == "x1+v" else -1.0
        E = T._chat_encode(tok, model_io, it["target"]["messages"], it["target"]["limit_text"], it["role"])
        if len(E["post"]) != n_prompt:
            raise ValueError("prompt length after the limit differs from the vectors'")
        vcache = {L: _vectors(st_, it["fixture"], L, loo) for L in layers}
        used_loo = next(iter(vcache.values()))[1]
        mask = T._scope_mask("prompt+reply", n_prompt, st_["n_post"], dev)
        s0 = E["span"][0]
        rel = torch.tensor([p - s0 for p in E["post"]])
        base, how = T._batched_prefix_cache(model, model_io, E["ids"][:s0], batch)
        full = torch.tensor([E["ids"]], device=dev).expand(batch, -1).contiguous()
        rows, calls, gram, n_none = [], [], None, 0
        for k, chunk in enumerate(plan):
            b0 = k * batch
            row_layers = [None if x is None else x[0] for x in chunk]
            row_full = [None if x is None else sign * x[1] * vcache[x[0]][0]["v_pos"] * mask for x in chunk]
            adders = multi_gen(blocks, row_layers, rel, row_full, n_prompt, len(E["ids"]) - s0)
            lp, gram = (None, "no schema") if schema is None else T._grammar_processor(tok, schema, vocab)
            seed = int(it["seed"]) + b0
            torch.manual_seed(seed)
            kwargs = {"max_new_tokens": int(max_new), "attention_mask": torch.ones_like(full),
                      "past_key_values": copy.deepcopy(base)}
            if greedy:
                kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
            else:
                kwargs.update(do_sample=True, temperature=samp["temperature"], top_p=samp["top_p"], top_k=samp["top_k"])
            if lp is not None:
                kwargs["logits_processor"] = [lp]
            try:
                with torch.no_grad(), model_io.attn_ctx():
                    gen = model.generate(full, **kwargs)
            finally:
                _remove(adders)
            calls.append([adders[0].calls[0] if adders and adders[0].calls else None,
                          len(adders[0].calls) if adders else 0, len(adders)])
            for r, x in enumerate(chunk):
                new = gen[r, full.shape[1]:].tolist()
                cut = next((i for i, t in enumerate(new) if t in eos), None)
                finish = "stop" if cut is not None else "length"
                new = new[:cut] if cut is not None else new
                content = tok.decode(new, skip_special_tokens=True)
                move, errors = ge.validate_reply(content, finish, False, max_chars)
                if x is None:
                    L, vec, a, s = None, "none", 0.0, n_none
                    n_none += 1
                else:
                    (L, a, s), vec = x, "v_pos"
                rows.append({"layer": L, "vector": vec, "scope": "prompt+reply", "alpha": a, "sample": s, "seed": seed,
                             "finish": finish, "aligned": new[:len(reply_ids)] == reply_ids,
                             "completion_tokens": len(new), "errors": errors, "action": move and move["action"],
                             "price": move and move.get("price"), "content": content[:600], "constrained": lp is not None})
            del gen
        del base, vcache
        out.append({"item_id": it["item_id"], "fixture": it["fixture"], "role": it["role"], "direction": it["direction"],
                    "base_limit": it["base_limit"], "target_condition": it["target"]["condition"], "layers": layers,
                    "loo": used_loo, "n_prompt": n_prompt, "prefix_cache": how, "hook_calls": calls, "grammar": gram,
                    "greedy": bool(greedy), "rows": rows, "secs": round(time.time() - t0, 1)})
    return out


TASKS = {"e3c_build": e3c_build, "e3c_bridge": e3c_bridge, "e3c_steer": e3c_steer}
