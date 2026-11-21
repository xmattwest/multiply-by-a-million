"""tasks_e6.py (POD): E6, magnitude as a number vs magnitude as stakes, in Gemma 4 31B. Details and predictions:
scripts/interp_e6_laptop.py, config/interp/e6_predictions.md. Exploratory.

Part A (number). Short sentences that state an amount in five notations (digits, words, "$81M", "$81,000,000", and a
non-money count), mantissa x 10^e. Residual states at the sentence's last token at every layer, and MLP neuron
activations (the input of mlp.down_proj) at a few layers, are kept on the pod; e6_fit then fits a magnitude direction
per notation and layer and reports how it transfers across notations, and e6_census classifies neurons.

Part B (stakes). E5's seller build (Study 3's A0/A1 prompts) gives the negotiation scale direction vN (x1e6 - x1, per
aligned token). e6_dirs collapses its reply-token rows into one vector per layer (vR; vS from the S rows), at vN's
mean per-token norm, with a non-money contrast (u) and a random vector (rand) of the same norm. e6_steer adds
sign * alpha * vector on the model-turn header and a forced answer prefix of any chat prompt (risk choices,
comparisons, percentages, sums, estimates, and negotiation openings as the positive control), then reads either
the next-token log-probs of two options or a short greedy continuation.
"""
import re
import time

import torch                                   # pod-only

import hooks
import tasks as T
import tasks_e5 as E5                          # its tasks_e4 instance (E5.E4) holds the role stores used below
from tasks_heads import _fwd

E4 = E5.E4
_DIR = {}                                      # layer -> {"vR", "vS", "u", "rand"}; float32 on the model's device
_PROBE = {"meta": [], "resid": [], "mlp": {}, "mlp_layers": None}


# ---------------------------------------------------------------- Part B: the stakes direction
def e6_build(model, tok, items, settings, model_name, model_io, role="seller", **kw):
    """E5's build (Study 3's four prompts per fixture, plus the non-money pairs), into E5's role store."""
    return E5.e5_build(model, tok, items, settings, model_name, model_io, role=role, **kw)


def _unit(v, size):
    return v / v.norm().clamp_min(1e-12) * size


def collapse(vN, sg, seg="reply"):
    """One vector from vN's rows of a segment: the row mean, rescaled to the rows' mean norm (alpha 1 = one token's
    natural x1e6 - x1 difference)."""
    i0, i1 = sg[seg]
    rows = vN[i0:i1].float()
    return _unit(rows.mean(0), rows.norm(dim=-1).mean()), float(rows.mean(0).norm() / rows.norm(dim=-1).mean().clamp_min(1e-12))


def e6_dirs(model, tok, items, settings, model_name, model_io, layers=(42, 44), role="seller", **kw):
    """vR, vS, u, rand per layer from the role's store (all fixtures; leave-one-out vectors for fixtures are made in
    e6_steer). Returns their norms and cosines."""
    out = []
    with E5._mode(role):
        st_, sg = E4._STORE, E4._STORE["segs"]
        if not st_["d"] or sg is None:
            raise ValueError("run e6_build first")
        for L in sorted(int(x) for x in layers):
            V, _ = E4.vectors(None, L, loo=False)
            vR, kR = collapse(V["vN"], sg, "reply")
            vS, kS = collapse(V["vN"], sg, "S")
            size = vR.norm()
            vS = _unit(vS, size)                                       # every vector at vR's norm: alpha means the same
            u = _unit(V["u"][0].float(), size) if "u" in V else None
            g = torch.Generator(device="cpu").manual_seed(67890 + L)
            rand = _unit(torch.randn(vR.shape, generator=g).to(vR.device), size)
            _DIR[L] = {"vR": vR, "vS": vS, **({"u": u} if u is not None else {}), "rand": rand}
            cos = lambda a, b: round(float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-12)), 4)   # noqa: E731
            out.append({"item_id": f"dirs-L{L}", "layer": L, "norm": round(float(size), 4), "n_fixtures": len(st_["d"]),
                        "coherence_reply": round(kR, 4), "coherence_S": round(kS, 4),
                        "cos_vR_vS": cos(vR, vS), "cos_vR_u": cos(vR, u) if u is not None else None,
                        "cos_vR_rand": cos(vR, rand), "segs": sg})
    return out


def _vector(name, L, fixture, role):
    """A stored vector, or for a negotiation fixture the leave-one-fixture-out vR / vS."""
    if fixture is not None and name in ("vR", "vS"):
        with E5._mode(role):
            V, used = E4.vectors(fixture, L, loo=True)
            v, _ = collapse(V["vN"], E4._STORE["segs"], "reply" if name == "vR" else "S")
        return _unit(v, _DIR[L]["vR"].norm()), used
    return _DIR[L][name], None


def encode_chat(tok, model_io, messages, prefix):
    """(ids, n_steer): the rendered chat + forced prefix; n_steer = the model-turn header + prefix tokens."""
    text, _ = model_io.render_messages(tok, messages)
    user = tok.apply_chat_template(messages, add_generation_prompt=False, tokenize=False, enable_thinking=False)
    if not text.startswith(user):
        raise ValueError("chat template: the generation prompt does not extend the conversation")
    ids = tok(text + prefix, add_special_tokens=False)["input_ids"]
    head = tok(user, add_special_tokens=False)["input_ids"]
    if ids[:len(head)] != head:
        raise ValueError("the header/prefix boundary falls inside a token")
    return ids, len(ids) - len(head)


_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def parse_number(s):
    m = _NUM.search(s or "")
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def e6_steer(model, tok, items, settings, model_name, model_io, plan=(), n_none=2, batch=16, role="seller", **kw):
    """Item: {"messages", "prefix", "readout": {"type": "choice", "options": [..2]} | {"type": "gen", "max_new": n},
    "sign" (+1 on x1 prompts, -1 on x1e6), "fixture" (negotiation only)}. Rows: n_none unsteered + plan
    [[vector, layer, alpha], ...], one batch; each steered row adds sign * alpha * vector at every header + prefix
    position in the prompt's forward (hooks.PrefillAdder); generated tokens are not steered."""
    blocks = model_io.decoder_layers(model)
    rows_spec = [None] * int(n_none) + [tuple(x) for x in plan]
    if len(rows_spec) > int(batch):
        raise ValueError("plan larger than the batch")
    rows_spec += [None] * (int(batch) - len(rows_spec))
    out = []
    for it in items:
        t0 = time.time()
        ids, n_steer = encode_chat(tok, model_io, it["messages"], it["prefix"])
        Tn = len(ids)
        pos = torch.arange(Tn - n_steer, Tn)
        B = len(rows_spec)
        sign, fixture = float(it.get("sign", 1.0)), it.get("fixture")
        ads, loo = [], set()
        H = next(iter(_DIR.values()))["vR"].shape[0] if _DIR else None
        for L in sorted({int(x[1]) for x in rows_spec if x}):
            if L not in _DIR:
                raise ValueError(f"no vectors at layer {L}: run e6_dirs with it")
            d = torch.zeros((B, n_steer, H), device=model.device)
            for r, x in enumerate(rows_spec):
                if x and int(x[1]) == L:
                    v, used = _vector(x[0], L, fixture, role)
                    if used is not None:
                        loo.add(used)
                    d[r] = sign * float(x[2]) * v.to(d.device)
            ads.append(hooks.PrefillAdder(blocks[L], pos, d, Tn))
        x_ids = torch.tensor([ids], device=model.device).expand(B, -1).contiguous()
        ro = it["readout"]
        try:
            with torch.no_grad(), model_io.attn_ctx():
                if ro["type"] == "choice":
                    o = _fwd(model, model_io, 1, input_ids=x_ids, use_cache=False)
                    lp = torch.log_softmax(o.logits[:, -1].float(), -1)
                    opt = [tok(s, add_special_tokens=False)["input_ids"] for s in ro["options"]]
                    if any(len(t) != 1 for t in opt):
                        raise ValueError(f"options are not single tokens: {ro['options']}")
                    res = [{"lp": [round(float(lp[r, t[0]]), 4) for t in opt],
                            "mass": round(float(sum(lp[r, t[0]].exp() for t in opt)), 4)} for r in range(B)]
                else:
                    gen = model.generate(x_ids, attention_mask=torch.ones_like(x_ids), max_new_tokens=int(ro["max_new"]),
                                         do_sample=False, temperature=None, top_p=None, top_k=None)
                    res = []
                    for r in range(B):
                        text = tok.decode(gen[r, Tn:].tolist(), skip_special_tokens=True)
                        res.append({"text": text[:80], "value": parse_number(text)})
        finally:
            fired = [len(a.fired) for a in ads]
            for a in ads:
                a.remove()
        rows = [dict(r, vector=x[0] if x else "none", layer=int(x[1]) if x else None, alpha=float(x[2]) if x else 0.0)
                for x, r in zip(rows_spec, res)]
        out.append({"item_id": it["item_id"], "family": it["family"], "scale": it.get("scale"), "sign": sign,
                    "fixture": fixture, "n_steer": n_steer, "n_tokens": Tn, "hooks_fired": fired,
                    "loo": sorted(str(x) for x in loo), "rows": rows, "secs": round(time.time() - t0, 2)})
    return out


# ---------------------------------------------------------------- Part A: the number
def _mlp_inputs(blocks, layers, store):
    """Forward pre-hooks on mlp.down_proj: its input is the neuron activation act(gate) * up, (B, T, d_mlp)."""
    hs = []
    for L in layers:
        mod = blocks[L].mlp.down_proj

        def f(_m, a, L=L):
            store[L] = a[0][0, -1].detach().float().cpu()
        hs.append(mod.register_forward_pre_hook(f))
    return hs


def e6_probe(model, tok, items, settings, model_name, model_io, mlp_layers=(), **kw):
    """Item: {"text", "notation", "log10", "mantissa", "exp"}. Keeps the last token's residual at every layer
    (float16, CPU) and the MLP activations at mlp_layers; returns token counts only."""
    blocks = model_io.decoder_layers(model)
    nL, ml = len(blocks), sorted(int(x) for x in mlp_layers)
    if _PROBE["mlp_layers"] not in (None, ml):
        raise ValueError("mlp_layers changed between calls")
    _PROBE["mlp_layers"] = ml
    out = []
    for it in items:
        t0 = time.time()
        ids = tok(it["text"], add_special_tokens=True)["input_ids"]
        if tok.bos_token_id is not None and ids[0] != tok.bos_token_id:
            ids = [tok.bos_token_id] + ids
        mstore = {}
        hs = _mlp_inputs(blocks, ml, mstore)
        try:
            vals, _ = E4._capture(model, model_io, blocks, ids, [len(ids) - 1], [("resid", L) for L in range(nL)])
        finally:
            for h in hs:
                h.remove()
        _PROBE["resid"].append(torch.stack([vals[("resid", L)][0] for L in range(nL)]).to("cpu", torch.float16))
        for L in ml:
            _PROBE["mlp"].setdefault(L, []).append(mstore[L].to(torch.float16))
        _PROBE["meta"].append({k: it[k] for k in ("item_id", "notation", "log10", "mantissa", "exp")})
        out.append({"item_id": it["item_id"], "n_tokens": len(ids), "last_token": tok.decode([ids[-1]]),
                    "secs": round(time.time() - t0, 3)})
    return out


def _pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-12))


def _direction(X, y):
    """Covariance direction sum_i (x_i - mean)(y_i - mean): the least-squares slope direction up to scale."""
    return ((X - X.mean(0)) * (y - y.mean())[:, None]).sum(0)


def e6_fit(model, tok, items, settings, model_name, model_io, stakes_layers=(42, 44), **kw):
    """Per layer: within-notation held-out-mantissa r; cross-notation transfer r (direction from one notation,
    projections of another, Pearson with log10); cosines between notation directions; cosine of the digits
    direction with vR (stakes) where it exists."""
    M = _PROBE["meta"]
    if not M:
        raise ValueError("run e6_probe first")
    R = torch.stack(_PROBE["resid"]).float()                         # (N, L, H)
    y = torch.tensor([m["log10"] for m in M], dtype=torch.float32)
    nots = sorted({m["notation"] for m in M})
    idx = {k: torch.tensor([i for i, m in enumerate(M) if m["notation"] == k]) for k in nots}
    mant = torch.tensor([m["mantissa"] for m in M])
    out = {"notations": nots, "n": {k: int(len(v)) for k, v in idx.items()}, "layers": []}
    for L in range(R.shape[1]):
        X = R[:, L]
        W = {k: _direction(X[idx[k]], y[idx[k]]) for k in nots}
        within = {}
        for k in nots:
            I = idx[k]
            proj, ys = [], []
            for mv in sorted(set(mant[I].tolist())):
                te, tr = I[mant[I] == mv], I[mant[I] != mv]
                if len(tr) < 3 or not len(te):
                    continue
                w = _direction(X[tr], y[tr])
                proj.append(X[te] @ w)
                ys.append(y[te])
            within[k] = round(_pearson(torch.cat(proj), torch.cat(ys)), 4) if proj else None
        transfer = {a: {b: round(_pearson(X[idx[b]] @ W[a], y[idx[b]]), 4) for b in nots} for a in nots}
        cos = {a: {b: round(float((W[a] @ W[b]) / (W[a].norm() * W[b].norm()).clamp_min(1e-12)), 4) for b in nots} for a in nots}
        rec = {"layer": L, "within": within, "transfer": transfer, "cos": cos}
        if L in _DIR:
            v = _DIR[L]["vR"].float().cpu()
            rec["cos_digits_vR"] = round(float((W["digits"] @ v) / (W["digits"].norm() * v.norm()).clamp_min(1e-12)), 4) \
                if "digits" in W else None
            rec["vR_proj_r"] = {k: round(_pearson(X[idx[k]] @ v, y[idx[k]]), 4) for k in nots}
        out["layers"].append(rec)
    return [dict(out, item_id="fit")]


def _ranks(x):
    """Ranks along dim 0 (ties broken by order; activations are continuous)."""
    return torch.argsort(torch.argsort(x, 0), 0).float()


def e6_census(model, tok, items, settings, model_name, model_io, dial=0.8, weak=0.3, top=12, **kw):
    """Per kept MLP layer, every neuron's Spearman correlation with log10 within each notation. Dials: |rho| >= dial
    with one sign in every money notation; digit-only: |rho| >= dial in digits and < weak in words; tuned: a peak
    exponent shared by >= 3 money notations with a z-score >= 3. Top neurons with their tuning curves."""
    M = _PROBE["meta"]
    if not M or not _PROBE["mlp"]:
        raise ValueError("run e6_probe with mlp_layers first")
    nots = sorted({m["notation"] for m in M})
    money = [k for k in nots if k != "count"]
    exps = sorted({m["exp"] for m in M})
    out = {"layers": {}, "notations": nots, "exps": exps, "dial": dial, "weak": weak}
    for L, rows in _PROBE["mlp"].items():
        A = torch.stack(rows).float()                                 # (N, d_mlp)
        rho, curves = {}, {}
        for k in nots:
            I = [i for i, m in enumerate(M) if m["notation"] == k]
            a, yv = A[I], torch.tensor([M[i]["log10"] for i in I])
            ra, ry = _ranks(a), _ranks(yv[:, None])[:, 0]
            ra, ry = ra - ra.mean(0), ry - ry.mean()
            rho[k] = (ra * ry[:, None]).sum(0) / (ra.norm(dim=0) * ry.norm()).clamp_min(1e-12)
            curves[k] = torch.stack([a[[j for j, i in enumerate(I) if M[i]["exp"] == e]].mean(0) for e in exps])   # (E, d)
        Rm = torch.stack([rho[k] for k in money])                    # (k, d)
        dials = ((Rm.abs() >= dial).all(0) & ((Rm > 0).all(0) | (Rm < 0).all(0)))
        digit_only = (rho["digits"].abs() >= dial) & (rho["words"].abs() < weak) if "words" in rho else torch.zeros_like(dials)
        C = torch.stack([curves[k] for k in money])                  # (k, E, d)
        z = (C - C.mean(1, keepdim=True)) / C.std(1, keepdim=True).clamp_min(1e-6)
        peak, zmax = z.argmax(1), z.max(1).values                    # (k, d)
        mode = torch.mode(peak, 0).values
        agree = (peak == mode).sum(0)
        tuned = (agree >= 3) & ((zmax * (peak == mode)).max(0).values >= 3.0) & ~dials
        score = Rm.abs().min(0).values
        pick = torch.argsort(score, descending=True)[:int(top)]
        out["layers"][str(L)] = {
            "d_mlp": int(A.shape[1]), "n_dial": int(dials.sum()), "n_dial_count_too": int((dials & (rho["count"].abs() >= dial)).sum())
            if "count" in rho else None, "n_digit_only": int(digit_only.sum()), "n_tuned": int(tuned.sum()),
            "tuned_by_exp": {str(exps[int(e)]): int(((mode == e) & tuned).sum()) for e in range(len(exps))},
            "top": [{"neuron": int(n), "rho": {k: round(float(rho[k][n]), 3) for k in nots},
                     "curve": {k: [round(float(x), 4) for x in curves[k][:, n]] for k in nots}} for n in pick.tolist()]}
    return [dict(out, item_id="census")]


TASKS = {"e6_build": e6_build, "e6_dirs": e6_dirs, "e6_steer": e6_steer, "e6_probe": e6_probe, "e6_fit": e6_fit,
         "e6_census": e6_census}
