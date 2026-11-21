"""tasks_concepts.py (POD): what the seller's x1e6 - x1 direction is made of. Laptop side, design and details:
scripts/interp_concepts_laptop.py. Exploratory. N1 / N6 = D2 / D8 seller prompts; A = H + S + reply (tasks_e4.encode).
Concepts: sentence pairs (a, b) sharing a token suffix; direction = mean over pairs of (mean_b - mean_a) on it.
Patch and real-opening steering are tasks_e4's own code with this module's vectors swapped in.
SAEs (pinned, size + sha256 checked; synthetic in the rehearsal): jumprelu (Decode, L30), topk (Adam1010, 41/44)."""
import contextlib
import copy
import hashlib
import os
import random
import shutil
import time
import urllib.request

import torch                                   # pod-only

import hooks
import tasks as T
import tasks_e4 as E4
from tasks_e4 import _capture, encode_fixture, segs
from tasks_heads import _decision, _fwd

PAIR = ("N1", "N6")
SCALE = {"N1": "x1", "N6": "x1e6"}
_S = {"neg": {}, "dN": {}, "x": {}, "keep": None, "acc": None, "n": 0, "segs": None, "c": {}, "sae": {}, "attr": {}}


def _rehearsal():
    return os.environ.get("SI_INTERP_REHEARSAL") == "1"


def _r(t, d=5):
    return [round(float(x), d) for x in t]


def pair_encode(tok, a, b, min_suffix=2):
    """<bos> + each sentence; positions of their longest common token suffix."""
    bos = tok.bos_token_id if tok.bos_token_id is not None else 2
    ia, ib = ([bos] + list(tok(s, add_special_tokens=False)["input_ids"]) for s in (a, b))
    k = 0
    while k < min(len(ia), len(ib)) - 1 and ia[-1 - k] == ib[-1 - k]:
        k += 1
    if k < min_suffix or ia == ib:
        raise ValueError(f"common suffix of {k} tokens: {a!r} / {b!r}")
    return ia, ib, list(range(len(ia) - k, len(ia))), list(range(len(ib) - k, len(ib)))


def concepts_build(model, tok, items, settings, model_name, model_io, keep=(30, 42, 44), **kw):
    """Fixtures: greedy prices, ld, dN = N6 - N1 at every layer on A (sum; per fixture at `keep`).
    Concept pairs: the suffix-mean difference at every layer, summed per concept and half."""
    blocks = model_io.decoder_layers(model)
    nL, keep = len(blocks), sorted(int(x) for x in keep)
    if min(keep) < 0 or max(keep) >= nL or _S["keep"] not in (None, keep):
        raise ValueError("bad keep layers")
    _S["keep"] = keep
    out = []
    for it in items:
        t0 = time.time()
        if it.get("kind") == "concept":
            ia, ib, pa, pb = pair_encode(tok, it["a"], it["b"])
            xa, xb = (T._capture_all_layers(model, model_io, i, p, device_keep=True).float().mean(1) for i, p in ((ia, pa), (ib, pb)))
            z = torch.zeros_like(xa)
            c = _S["c"].setdefault(it["concept"], {"sum": z.clone(), "half": [z.clone(), z.clone()], "n": 0, "nh": [0, 0]})
            c["sum"] += xb - xa
            c["half"][int(it["half"])] += xb - xa
            c["n"] += 1
            c["nh"][int(it["half"])] += 1
            out.append({"item_id": it["item_id"], "kind": "concept", "concept": it["concept"], "n_suffix": len(pa), "secs": round(time.time() - t0, 2)})
            continue
        E = encode_fixture(tok, model_io, it)
        e = E["N1"]
        rec = {"item_id": it["item_id"], "kind": "fixture", "fixture": it["fixture"], "base_limit": it["base_limit"], "n_layers": nL,
               "n_A": len(e["A"]), "n_H": e["n_H"], "n_S": e["n_S"], "n_reply": e["n_reply"], "n_span": {k: len(E[k]["span"]) for k in PAIR}}
        pr = {k: T.teacher(model, tok, [{"item_id": k, "messages": it["prompts"][k]["messages"], "role": it["role"],
                                          "condition": it["prompts"][k]["condition"], "limit_units": it["prompts"][k]["limit_units"]}],
                           settings, model_name, model_io)[0]["price_r2"] for k in PAIR}
        a, b = pr["N1"], pr["N6"]
        why = None if a and b else "no greedy price"
        why = why or ("the N6 price is not whole millions" if b % 10 ** 6 else "the two prices are equal" if a * 10 ** 6 == b else None)
        if why and _rehearsal():                   # random weights write no prices
            a, b, why = round(1.3 * it["base_limit"]), round(1.2 * it["base_limit"]) * 10 ** 6, None
            rec["stand_in"] = True
        rec["price"] = {"N1": a, "N6": b}
        if why:
            out.append(dict(rec, skipped=why, secs=round(time.time() - t0, 1)))
            continue
        cand = {"x1": _decision(tok, a, b // 10 ** 6), "x1e6": _decision(tok, a * 10 ** 6, b)}
        X, ld = {}, {}
        for k in PAIR:
            pre, ta, tb = cand[SCALE[k]]
            vals, lg = _capture(model, model_io, blocks, E[k]["ids"], E[k]["A"], [("resid", L) for L in range(nL)], pre, len(pre) + 1)
            lp = torch.log_softmax(lg[-1], -1)
            ld[k] = round(float(lp[tb] - lp[ta]), 5)
            X[k] = torch.stack([vals[("resid", L)] for L in range(nL)]).float()
        D = X["N6"] - X["N1"]
        _S["acc"] = D if _S["acc"] is None else _S["acc"] + D
        _S["n"] += 1
        dt = next(model.parameters()).dtype
        _S["dN"][it["fixture"]] = D[keep].to(dt)
        _S["x"][it["fixture"]] = {k: X[k][keep].to(dt) for k in PAIR}     # the runs' own states (ablation rows)
        _S["neg"][it["fixture"]] = E4._STORE["pairs"][it["fixture"]] = {"cand": cand, "ld": ld}
        sg = _S["segs"] = segs(e)
        rec.update(ld=ld, norm={s: _r(D[:, sg[s][0]:sg[s][1]].norm(dim=-1).mean(1), 3) for s in ("H", "S", "reply")}, secs=round(time.time() - t0, 1))
        out.append(rec)
    return out


def _dirs(half=None):
    return {n: (c["sum"] / max(1, c["n"])) if half is None else c["half"][half] / max(1, c["nh"][half]) for n, c in sorted(_S["c"].items())}


def concepts_stats(model, tok, items, settings, model_name, model_io, n_random=8, **kw):
    """Per layer and segment of v = fixture-mean dN: cosine with each concept (segment mean; per token), share of |v|^2 in
    the concepts' span (and in n_random random directions' span: chance), split-half and pairwise concept cosines."""
    if not _S["n"] or not _S["c"]:
        raise ValueError("run concepts_build first")
    v, C = _S["acc"] / _S["n"], _dirs()
    names, nL, H = sorted(C), v.shape[0], v.shape[-1]
    M = torch.stack([C[n] for n in names], 1)
    R = torch.randn((n_random, H), generator=torch.Generator(device="cpu").manual_seed(4242)).to(v.device)
    cs = torch.nn.functional.cosine_similarity

    def share(x, B):
        Q = torch.linalg.qr(B.T.double())[0]
        x = x.double()
        return ((x @ Q).pow(2).sum(-1) / x.pow(2).sum(-1).clamp_min(1e-30)).float()
    per = {}
    for s in ("H", "S", "reply", "A"):
        i0, i1 = _S["segs"][s]
        vs = v[:, i0:i1]
        m = vs.mean(1)
        per[s] = {"cos": {n: [float(cs(m[L], C[n][L], dim=0)) for L in range(nL)] for n in names},
                  "cos_token": {n: [float(cs(vs[L], C[n][L][None], dim=-1).mean()) for L in range(nL)] for n in names},
                  "span_share": [float(share(m[L], M[L])) for L in range(nL)], "span_share_token": [float(share(vs[L], M[L]).mean()) for L in range(nL)],
                  "span_share_random": [float(share(m[L], R)) for L in range(nL)], "norm_v": _r(m.norm(dim=-1), 3)}
    h0, h1 = _dirs(0), _dirs(1)
    pick = sorted(set(_S["keep"]) | {nL // 2, nL - 1})
    return [{"n_fixtures": _S["n"], "n_layers": nL, "concepts": names, "n_pairs": {n: _S["c"][n]["n"] for n in names},
             "norm_c": {n: _r(C[n].norm(dim=-1), 3) for n in names}, "per_segment": per, "n_random": int(n_random),
             "split_half_cos": {n: [float(cs(h0[n][L], h1[n][L], dim=0)) for L in range(nL)] for n in names},
             "pairwise_cos": {str(L): {f"{a}|{b}": round(float(cs(C[a][L], C[b][L], dim=0)), 4) for i, a in enumerate(names)
                                       for b in names[i + 1:]} for L in pick}}]


def _vectors(fixture, L, loo=True):
    """{vN: leave-one-fixture-out mean dN at L (A, H); each concept at L scaled to vN's per-token norm; rand;
    par / perp: vN's per-token projection onto the six concepts' span and the rest (par + perp = vN), par_s / perp_s
    the same scaled to vN's per-token norm; abl6_<k> / ablr_<k> / ablv_<k>: minus run k's own state projected onto the
    concepts' span / a random 6-d span / vN's own per-token direction (adding it ablates that part)}, used."""
    D, i = _S["dN"], _S["keep"].index(L)
    tot = sum(d[i].float() for d in D.values())
    use = loo and len(D) >= 3 and fixture in D
    v = (tot - D[fixture][i].float()) / (len(D) - 1) if use else tot / len(D)
    size = v.norm(dim=-1, keepdim=True)
    out = {"vN": v}
    for n, c in _dirs().items():
        c = c[L].to(v.device)
        out[n] = (c / c.norm().clamp_min(1e-30)).expand_as(v) * size
    r = torch.randn(tuple(v.shape), generator=torch.Generator(device="cpu").manual_seed(777 + L)).to(v.device)
    out["rand"] = r / r.norm(dim=-1, keepdim=True) * size
    C = _dirs()
    Q = torch.linalg.qr(torch.stack([C[n][L].to(v.device).float() for n in sorted(C)], 1))[0]          # [H, 6]
    unit = lambda x: x / x.norm(dim=-1, keepdim=True).clamp_min(1e-30)                                # noqa: E731
    par = (v @ Q) @ Q.T
    out.update(par=par, perp=v - par, par_s=unit(par) * size, perp_s=unit(v - par) * size)
    X = _S["x"].get(fixture)
    if X is not None:
        R = torch.linalg.qr(torch.randn((v.shape[-1], Q.shape[1]), generator=torch.Generator(device="cpu").manual_seed(991 + L)))[0]
        R, u = R.to(v.device), unit(v)
        for k in PAIR:
            x = X[k][i].float()
            out[f"abl6_{k}"], out[f"ablr_{k}"] = -(x @ Q) @ Q.T, -(x @ R) @ R.T
            out[f"ablv_{k}"] = -(x * u).sum(-1, keepdim=True) * u
    return out, use


@contextlib.contextmanager
def _our_vectors():
    old, E4.vectors = E4.vectors, _vectors
    try:
        yield
    finally:
        E4.vectors = old


def concepts_patch(model, tok, items, settings, model_name, model_io, **kw):
    """tasks_e4.e4_patch (forced opening; rows {target: [{"n", "ops": [["vec", L, seg, alpha, name]]}]}) with these vectors."""
    with _our_vectors():
        return E4.e4_patch(model, tok, items, settings, model_name, model_io, **kw)


def concepts_steer(model, tok, items, settings, model_name, model_io, **kw):
    """tasks_e4.e4_steer (real openings; plan [[name, L, alpha, n]]) with these vectors."""
    with _our_vectors():
        return E4.e4_steer(model, tok, items, settings, model_name, model_io, **kw)


# --- SAEs --------------------------------------------------------------------------------------------------------------
class JumpReLUSAE:
    """SAELens JumpReLU: pre = (x - b_dec) W_enc + b_enc; f = relu(pre) * (pre > threshold); x^ = f W_dec + b_dec."""
    kind = "jumprelu"

    def __init__(self, W_enc, b_enc, W_dec, b_dec, threshold):
        self.W_enc, self.b_enc, self.W_dec, self.b_dec, self.thr = W_enc, b_enc, W_dec, b_dec, threshold
        self.d_in, self.d_sae = W_enc.shape

    def encode(self, x):
        pre = (x.float() - self.b_dec) @ self.W_enc + self.b_enc
        return torch.relu(pre) * (pre > self.thr)

    def decode(self, f):
        return f @ self.W_dec + self.b_dec

    def dec_dirs(self, idx):
        return self.W_dec[idx]


class TopKSAE(JumpReLUSAE):
    """TopK on xn = (x - norm_mean) / norm_std (variant 2 also subtracts b_dec first); decoder output un-normalised."""
    kind = "topk"

    def __init__(self, W_enc, b_enc, W_dec, b_dec, mean, std, k):
        self.W_enc, self.b_enc, self.W_dec, self.b_dec = W_enc, b_enc, W_dec, b_dec
        self.mean, self.std, self.k, self.variant = mean, std.clamp_min(1e-6), int(k), 1
        self.d_in, self.d_sae = W_enc.shape

    def encode(self, x):
        xn = (x.float() - self.mean) / self.std - (self.b_dec if self.variant == 2 else 0)
        v, i = torch.relu(xn @ self.W_enc + self.b_enc).topk(self.k, dim=-1)
        return torch.zeros((*x.shape[:-1], self.d_sae), device=x.device).scatter_(-1, i, v)

    def decode(self, f):
        return (f @ self.W_dec + self.b_dec) * self.std + self.mean

    def dec_dirs(self, idx):
        return self.W_dec[idx] * self.std


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url, path, size, sha256, tries=3):
    """https://huggingface.co/ only; size and sha256 checked; a file that already checks out is reused."""
    if not url.startswith("https://huggingface.co/"):
        raise ValueError("only https://huggingface.co/ URLs")
    if os.path.exists(path) and os.path.getsize(path) == size and file_sha256(path) == sha256:
        return {"downloaded": False, "secs": 0.0}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if shutil.disk_usage(os.path.dirname(path)).free < size + 2 * 10 ** 9:
        raise RuntimeError("not enough disk for " + os.path.basename(path))
    t0, why = time.time(), None
    for _ in range(int(tries)):
        h, n = hashlib.sha256(), 0
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(path + ".part", "wb") as fh:
                for chunk in iter(lambda: r.read(1 << 22), b""):
                    fh.write(chunk)
                    h.update(chunk)
                    n += len(chunk)
        except Exception as e:                     # noqa: BLE001
            why = f"{type(e).__name__}: {e}"
            continue
        if n == size and h.hexdigest() == sha256:
            os.replace(path + ".part", path)
            return {"downloaded": True, "secs": round(time.time() - t0, 1)}
        why = f"got {n} bytes, sha256 {h.hexdigest()[:12]}"
    if os.path.exists(path + ".part"):
        os.remove(path + ".part")
    raise RuntimeError(f"{os.path.basename(path)}: download failed ({why})")


def load_sae(spec, path, device):
    if spec["kind"] == "jumprelu":
        from safetensors.torch import load_file
        w, keys = load_file(path, device=str(device)), ("W_enc", "b_enc", "W_dec", "b_dec", "threshold")
    else:
        w, keys = torch.load(path, map_location=device, weights_only=True), ("W_enc", "b_enc", "W_dec", "b_dec", "norm_mean", "norm_std")
    if not set(keys) <= set(w):
        raise ValueError(f"unexpected tensors {sorted(w)}")
    t = [w[k].float() for k in keys]
    return JumpReLUSAE(*t) if spec["kind"] == "jumprelu" else TopKSAE(*t, spec["k"])


def sae_load(model, tok, items, settings, model_name, model_io, dir="/tmp/si-sae", drop=(), **kw):
    """items: SAE specs {"item_id", "name", "kind", "url", "size", "sha256", "k"?}; drop: names to free first."""
    for n in drop:
        _S["sae"].pop(n, None)
    if drop and torch.cuda.is_available():
        torch.cuda.empty_cache()
    H, dev = model_io.decoder_layers(model)[0].self_attn.config.hidden_size, next(model.parameters()).device
    out = []
    for it in items:
        t0 = time.time()
        if _rehearsal():                           # stand-in: random weights of the model's width
            W = torch.randn((256, H), generator=torch.Generator(device="cpu").manual_seed(31)).to(dev) / H ** 0.5
            z = torch.zeros(H, device=dev)
            sae = JumpReLUSAE(W.T.contiguous(), torch.zeros(256, device=dev), W, z, torch.full((256,), 0.05, device=dev)) \
                if it["kind"] == "jumprelu" else TopKSAE(W.T.contiguous(), torch.zeros(256, device=dev), W, z, z, z + 1, 16)
            got = {"downloaded": False, "synthetic": True}
        else:
            path = os.path.join(dir, it["name"] + (".safetensors" if it["kind"] == "jumprelu" else ".pt"))
            got = fetch(it["url"], path, int(it["size"]), it["sha256"])
            sae = load_sae(it, path, dev)
        if sae.d_in != H:
            raise ValueError(f"{it['name']}: d_in {sae.d_in}, model hidden {H}")
        _S["sae"][it["name"]] = sae
        out.append({"item_id": it["item_id"], "name": it["name"], "kind": sae.kind, "d_in": sae.d_in, "d_sae": sae.d_sae, **got,
                    "secs": round(time.time() - t0, 1)})
    return out


def _seg_index(e):
    """Indices into the suffix (positions s0.. of e["ids"]): span, H, S, reply, A."""
    s0, sg = e["s0"], segs(e)
    A = [p - s0 for p in e["A"]]
    return {"span": [p - s0 for p in e["span"]], "A": A, **{s: A[sg[s][0]:sg[s][1]] for s in ("H", "S", "reply")}}


def _stats(x, xh, f):
    err = (x - xh).pow(2).sum(-1)
    return {"rel_err": round(float(err.sum() / x.pow(2).sum().clamp_min(1e-30)), 5),
            "fvu_local": round(float(err.sum() / (x - x.mean(0)).pow(2).sum().clamp_min(1e-30)), 5),
            "cos": round(float(torch.nn.functional.cosine_similarity(x, xh, dim=-1).mean()), 5),
            "l0": round(float((f > 0).float().sum(-1).mean()), 2), "n": int(x.shape[0])}


def _splice_ld(model, model_io, blocks, ids, pre, L, s0, xh, ta, tb):
    """ld with layer L's output on positions s0..len(ids) replaced by xh."""
    pos = torch.arange(s0, len(ids), device=model.device)

    def hk(_m, _i, o):
        z = hooks._block_output_residual(o).clone()
        z[0, pos] = xh.to(z.dtype)
        return hooks._replace_block_output(o, z)
    h = blocks[L].register_forward_hook(hk)
    try:
        o = _fwd(model, model_io, len(pre) + 1, input_ids=torch.tensor([list(ids) + list(pre)], device=model.device), use_cache=False)
    finally:
        h.remove()
    lp = torch.log_softmax(o.logits[0, -1].float(), -1)
    return round(float(lp[tb] - lp[ta]), 5), round(float(lp[ta].exp() + lp[tb].exp()), 4)


def _pick_ctx(tok, model_io, it):
    P = _S["neg"].get(it["fixture"])
    return (None, None) if P is None else (P, encode_fixture(tok, model_io, it))


def sae_quality(model, tok, items, settings, model_name, model_io, sae="decode30", layer=30, top=20, **kw):
    """Per fixture and prompt: reconstruction on the 40 tokens before the limit (ctx) and on span / H / S / reply (topk: the
    better of the two encoder variants); ld with the reconstruction spliced in from the limit on; the top latents by
    |mean over A of f(N6) - f(N1)|."""
    S_, blocks, L = _S["sae"][sae], model_io.decoder_layers(model), int(layer)
    out = []
    for it in items:
        t0 = time.time()
        rec = {"item_id": it["item_id"], "fixture": it["fixture"], "sae": sae, "layer": L, "rehearsal": _rehearsal()}
        P, E = _pick_ctx(tok, model_io, it)
        if P is None:
            out.append(dict(rec, skipped="no usable prices from concepts_build"))
            continue
        s0 = E["N1"]["s0"]
        c0 = max(1, s0 - 40)
        per, F = {}, {}
        for k in PAIR:
            e, (pre, ta, tb) = E[k], P["cand"][SCALE[k]]
            vals, lg = _capture(model, model_io, blocks, e["ids"], list(range(c0, len(e["ids"]))), [("resid", L)], pre, len(pre) + 1)
            x, lp = vals[("resid", L)].float(), torch.log_softmax(lg[-1], -1)
            errs = {}
            for v in ((1, 2) if S_.kind == "topk" else (0,)):
                S_.variant = v
                fx = S_.encode(x)
                errs[v] = (float((x[s0 - c0:] - S_.decode(fx[s0 - c0:])).pow(2).sum() / x[s0 - c0:].pow(2).sum()), fx)
            S_.variant = best = min(errs, key=lambda v: errs[v][0])
            fx = errs[best][1]
            xh = S_.decode(fx)
            ix = {n: [j + s0 - c0 for j in v] for n, v in _seg_index(e).items()}
            ix["ctx"] = list(range(s0 - c0))
            ld_s, pk_s = _splice_ld(model, model_io, blocks, e["ids"], pre, L, s0, xh[s0 - c0:], ta, tb)
            per[k] = {"ld": round(float(lp[tb] - lp[ta]), 5), "pk": round(float(lp[ta].exp() + lp[tb].exp()), 4), "ld_spliced": ld_s,
                      "pk_spliced": pk_s, "stats": {n: _stats(x[v], xh[v], fx[v]) for n, v in ix.items() if v and n != "A"},
                      "variant": best or None, "variant_rel_err": {str(v): round(r, 5) for v, (r, _f) in errs.items()}}
            F[k] = fx[ix["A"]].mean(0)
        dA = F["N6"] - F["N1"]
        i = dA.abs().topk(min(int(top), dA.shape[0])).indices.tolist()
        rec["top_delta_A"] = [[j, round(float(dA[j]), 4), round(float(F["N1"][j]), 4), round(float(F["N6"][j]), 4)] for j in i]
        out.append(dict(rec, prompts=per, secs=round(time.time() - t0, 1)))
    return out


def _attr_forward(model, model_io, blocks, ids, pre, L, s0, S_, ta, tb):
    """Forward + backward with layer L's output on s0.. = decode(f) + err (f, err leaves): (ld, f, df, err, derr)."""
    pos, box = torch.arange(s0, len(ids), device=model.device), {}

    def hk(_m, _i, o):
        z = hooks._block_output_residual(o)
        x = z[0, pos].detach().float()
        f = S_.encode(x).detach().requires_grad_(True)
        err = (x - S_.decode(f.detach())).detach().requires_grad_(True)
        box.update(f=f, err=err)
        z = z.clone()
        z[0, pos] = (S_.decode(f) + err).to(z.dtype)
        return hooks._replace_block_output(o, z)
    h = blocks[L].register_forward_hook(hk)
    try:
        with torch.enable_grad(), model_io.attn_ctx():
            o = model(input_ids=torch.tensor([list(ids) + list(pre)], device=model.device), use_cache=False, logits_to_keep=1)
            lp = torch.log_softmax(o.logits[0, -1].float(), -1)
            ld = lp[tb] - lp[ta]
            gf, ge = torch.autograd.grad(ld, [box["f"], box["err"]])
    finally:
        h.remove()
    return float(ld), box["f"].detach(), gf.float(), box["err"].detach(), ge.float()


def sae_attr(model, tok, items, settings, model_name, model_io, sae="decode30", layer=30, top=300, skip=None, **kw):
    """Discovery fixtures: first-order attribution of ld(N6) - ld(N1) to latents at `layer`: aligned tokens
    (f6 - f1) (g6 + g1) / 2, the limit's digits sum f6 g6 - sum f1 g1; the same for the error term. Totals kept for the pick."""
    if skip:
        return [{"item_id": it["item_id"], "fixture": it.get("fixture"), "skipped": skip} for it in items]
    for p in model.parameters():
        p.requires_grad_(False)
    S_, blocks, L = _S["sae"][sae], model_io.decoder_layers(model), int(layer)
    acc = _S["attr"].setdefault(sae, {"sum": 0, "abs": 0, "n": 0})
    out = []
    for it in items:
        t0 = time.time()
        rec = {"item_id": it["item_id"], "fixture": it["fixture"], "sae": sae, "layer": L}
        P, E = _pick_ctx(tok, model_io, it)
        if P is None:
            out.append(dict(rec, skipped="no usable prices from concepts_build"))
            continue
        R = {}
        for k in PAIR:
            pre, ta, tb = P["cand"][SCALE[k]]
            R[k] = _attr_forward(model, model_io, blocks, E[k]["ids"], pre, L, E["N1"]["s0"], S_, ta, tb) + (_seg_index(E[k]),)
        ld1, f1, g1, e1, h1, I1 = R["N1"]
        ld6, f6, g6, e6, h6, I6 = R["N6"]
        A1, A6, s1, s6 = I1["A"], I6["A"], I1["span"], I6["span"]
        al = ((f6[A6] - f1[A1]) * (g6[A6] + g1[A1]) / 2).sum(0)
        sp = (f6[s6] * g6[s6]).sum(0) - (f1[s1] * g1[s1]).sum(0)
        tot = al + sp
        acc["sum"], acc["abs"], acc["n"] = acc["sum"] + tot, acc["abs"] + tot.abs(), acc["n"] + 1
        mf = {k: {n: F[I[n]].mean(0) for n in ("span", "H", "S", "reply")} for k, F, I in (("N1", f1, I1), ("N6", f6, I6))}
        rec.update(ld={"N1": round(ld1, 5), "N6": round(ld6, 5)}, gap=round(ld6 - ld1, 5), feat_total=round(float(tot.sum()), 5),
                   feat_aligned=round(float(al.sum()), 5), feat_span=round(float(sp.sum()), 5),
                   err_aligned=round(float(((e6[A6] - e1[A1]) * (h6[A6] + h1[A1]) / 2).sum()), 5),
                   err_span=round(float((e6[s6] * h6[s6]).sum() - (e1[s1] * h1[s1]).sum()), 5),
                   top=[[j, round(float(tot[j]), 5), round(float(al[j]), 5), round(float(sp[j]), 5),
                         {k: {n: round(float(mf[k][n][j]), 3) for n in mf[k]} for k in PAIR}]
                        for j in tot.abs().topk(min(int(top), tot.shape[0])).indices.tolist()], secs=round(time.time() - t0, 1))
        out.append(rec)
    return out


def sae_attr_pick(model, tok, items, settings, model_name, model_io, sae="decode30", k=64, skip=None, **kw):
    """The k latents with the largest |mean attribution| (sign kept) and their decoder directions' top / bottom tokens."""
    acc = _S["attr"].get(sae)
    if skip or not acc or not acc["n"]:
        return [{"item_id": "pick", "sae": sae, "skipped": skip or "no attribution records"}]
    m = acc["sum"] / acc["n"]
    idx = m.abs().topk(min(int(k), m.shape[0])).indices
    W = model.get_output_embeddings().weight
    lg = _S["sae"][sae].dec_dirs(idx).to(W.device, W.dtype) @ W.T
    hi, lo = lg.topk(8, dim=-1).indices.tolist(), (-lg).topk(8, dim=-1).indices.tolist()
    return [{"item_id": "pick", "sae": sae, "n_fixtures": acc["n"], "total_mean": round(float(m.sum()), 5),
             "top": [[j, round(float(m[j]), 5), round(float(acc["abs"][j] / acc["n"]), 5)] for j in idx.tolist()],
             "lens": {str(j): [[tok.decode([t]) for t in hi[n]], [tok.decode([t]) for t in lo[n]]] for n, j in enumerate(idx.tolist())}}]


def sae_patch(model, tok, items, settings, model_name, model_io, sae="decode30", layer=30, rows=None, batch=16, skip=None, **kw):
    """Held-out fixtures, forced opening: move the target's latents toward the other scale's run at `layer`. rows: [{"n",
    "latents": [ids] | "all" | "all+err" | {"random", "seed", "exclude"}, "where": both | span | aligned}]. Aligned tokens
    take the other run's value at the same token; the limit's digits take the other run's mean over its digits."""
    if skip or not rows:
        return [{"item_id": it["item_id"], "fixture": it.get("fixture"), "skipped": skip or "no rows"} for it in items]
    S_, blocks, L = _S["sae"][sae], model_io.decoder_layers(model), int(layer)
    out = []
    for it in items:
        t0, f = time.time(), it["fixture"]
        rec = {"item_id": it["item_id"], "fixture": f, "sae": sae, "layer": L}
        P, E = _pick_ctx(tok, model_io, it)
        if P is None:
            out.append(dict(rec, skipped="no usable prices from concepts_build"))
            continue
        s0 = E["N1"]["s0"]
        X = {k: _capture(model, model_io, blocks, E[k]["ids"], list(range(s0, len(E[k]["ids"]))), [("resid", L)])[0][("resid", L)].float()
             for k in PAIR}
        F, I = {k: S_.encode(X[k]) for k in PAIR}, {k: _seg_index(E[k]) for k in PAIR}
        active = torch.nonzero((F["N1"] > 0).any(0) | (F["N6"] > 0).any(0)).flatten().tolist()
        base, how = T._batched_prefix_cache(model, model_io, E["N1"]["ids"][:s0], batch)
        res = {}
        for t in PAIR:
            s, (pre, ta, tb) = "N1" if t == "N6" else "N6", P["cand"][SCALE[t]]
            nt = len(E[t]["ids"]) - s0
            specs = list(rows) + [{"n": "clean"}] * (batch - len(rows) % batch)
            x_in = torch.tensor([E[t]["ids"][s0:] + pre], device=model.device).expand(batch, -1).contiguous()
            got, nsel = [], {}
            for b0 in range(0, len(specs), batch):
                chunk = specs[b0:b0 + batch]
                D = torch.zeros((batch, nt, X[t].shape[1]), device=model.device)
                for r_, sp in enumerate(chunk):
                    lat = sp.get("latents")
                    if lat is None:
                        continue
                    if lat == "all+err":
                        src, tgt, W = X[s], X[t], None
                    else:
                        if isinstance(lat, dict):
                            pool = [j for j in active if j not in set(lat["exclude"])]
                            sel = sorted(random.Random(f"{lat['seed']}/{f}").sample(pool, min(int(lat["random"]), len(pool))))
                        else:
                            sel = slice(None) if lat == "all" else [int(j) for j in lat]
                        nsel[sp["n"]] = len(active) if lat == "all" else len(sel)
                        if not nsel[sp["n"]]:
                            continue
                        src, tgt, W = F[s][:, sel], F[t][:, sel], S_.dec_dirs(sel)
                    dd = torch.zeros_like(tgt)
                    if sp.get("where", "both") != "span":
                        dd[I[t]["A"]] = src[I[s]["A"]] - tgt[I[t]["A"]]
                    if sp.get("where", "both") != "aligned":
                        dd[I[t]["span"]] = src[I[s]["span"]].mean(0, keepdim=True) - tgt[I[t]["span"]]
                    D[r_] = dd if W is None else dd @ W

                def hk(_m, _i, o, D=D):
                    z = hooks._block_output_residual(o).clone()
                    z[:, :nt] = z[:, :nt] + D.to(z.dtype)
                    return hooks._replace_block_output(o, z)
                h = blocks[L].register_forward_hook(hk)
                try:
                    o = _fwd(model, model_io, len(pre) + 1, input_ids=x_in, past_key_values=copy.deepcopy(base), use_cache=True)
                finally:
                    h.remove()
                lp = torch.log_softmax(o.logits[:, -1].float(), -1)
                got += [{"n": sp["n"], "ld": round(float(lp[r_, tb] - lp[r_, ta]), 5), "lpa": round(float(lp[r_, ta]), 4),
                         "lpb": round(float(lp[r_, tb]), 4)} for r_, sp in enumerate(chunk)]
            clean = [g["ld"] for g in got if g["n"] == "clean"]
            res[t] = {"ld_clean": clean[0], "clean_spread": round(max(clean) - min(clean), 5), "hooks_ok": True, "n_latents": nsel,
                      "rows": [g for g in got if g["n"] != "clean"] + [g for g in got if g["n"] == "clean"][:1]}
        out.append(dict(rec, ld_build=P["ld"], n_active=len(active), prefix_cache=how, targets=res, secs=round(time.time() - t0, 1)))
    return out


def sae_profile(model, tok, items, settings, model_name, model_io, sae="decode30", layer=30, latents=(), skip=None, **kw):
    """Concept pairs: the chosen latents at `layer` on each sentence: [mean over the shared suffix, max over all tokens]."""
    sel = [int(j) for j in latents]
    if skip or not sel:
        return [{"item_id": it["item_id"], "skipped": skip or "no latents"} for it in items]
    S_, L, blocks = _S["sae"][sae], int(layer), model_io.decoder_layers(model)
    out = []
    for it in items:
        ia, ib, pa, pb = pair_encode(tok, it["a"], it["b"])
        r = {}
        for side, ids, p in (("a", ia, pa), ("b", ib, pb)):
            fx = S_.encode(_capture(model, model_io, blocks, ids, list(range(1, len(ids))), [("resid", L)])[0][("resid", L)].float())[:, sel]
            r[side] = [_r(fx[[j - 1 for j in p]].mean(0), 3), _r(fx.max(0).values, 3)]
        out.append({"item_id": it["item_id"], "concept": it["concept"], "latents": sel, **r})
    return out


TASKS = {"concepts_build": concepts_build, "concepts_stats": concepts_stats, "concepts_patch": concepts_patch,
         "concepts_steer": concepts_steer, "sae_load": sae_load, "sae_quality": sae_quality, "sae_attr": sae_attr,
         "sae_attr_pick": sae_attr_pick, "sae_patch": sae_patch, "sae_profile": sae_profile}
