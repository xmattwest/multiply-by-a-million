#!/usr/bin/env python3
"""tasks_buyers.py (POD-SIDE plug-in, MI Session 2): buyers, and the cross-role test.

E0 and E2 for buyers use the core tasks unchanged (generate, teacher, e2_patch with buyer items).
This file adds E3b-style steering for buyers with BOTH roles' directions kept side by side:

  buyers_build   at one layer: the seller's per-token x1e6 - x1 differences (core e3b_build on the
                 seller pairs), then the buyer's (core e3b_build on the buyer pairs). Both sets are
                 kept in _STORE["buyers"][role][layer]; _STORE["e3b"] is left holding the buyer set.
                 Returns norms and cosines only (seller vs buyer, with split-half references).
  buyers_bridge  Part A, E2's setup (reply opening forced, greedy price): core e3b_bridge, run with
                 the buyer's own vectors plus the seller's direction on buyer prompts:
                   seller       the seller's leave-one-fixture-out per-token mean difference, as is
                   seller_norm  the same, rescaled per token to the buyer v's per-token norm
                   seller_perp  the seller direction minus its per-token projection on the buyer v
  buyers_steer   Part B, real sampled openings (frozen settings, xgrammar): core e3b_steer, same vectors.
                 Both run the core code unchanged (see _with_vectors), so they match the sellers' X54 path.

Index alignment: the seller and buyer prompts have the same number of tokens after the limit and
the same 11 reply-opening tokens (checked here; the cross-role rows are refused otherwise). The
prompt words differ (buy/sell, lowest/highest, below/above, ceiling/floor), so on the prompt tokens
the cross-role vectors are aligned by position, not by token. Exploratory.
"""
import torch                                   # pod-only

import tasks as T                              # core helpers (read-only reuse)

_CORE_VECTORS = T._e3b_vectors                 # kept: _with_vectors swaps the module attribute while it runs

_ROLES = ("seller", "buyer")


def _cos(x, y):
    return float(torch.nn.functional.cosine_similarity(x.flatten(), y.flatten(), dim=0))


def _rowcos(x, y):
    """Per-row cosine of two (P, H) tensors -> list of P floats."""
    return [float(c) for c in torch.nn.functional.cosine_similarity(x, y, dim=-1)]


def _post_ids(tok, model_io, it):
    """Token ids after the limit in the R2 input (prompt + forced reply opening), for one pair side."""
    r = T._r2_encode(tok, model_io, it["x1"]["messages"], it["x1"]["limit_text"], it["role"])
    return r["ids"][r["span"][-1] + 1:]


def buyers_build(model, tok, items, settings, model_name, model_io, layer=44, **kw):
    """items: seller pairs + buyer pairs (kind 'pair', with role) + non-money pairs (kind 'nonmoney')."""
    layer = int(layer)
    nonmoney = [it for it in items if it["kind"] == "nonmoney"]
    store = T._STORE.setdefault("buyers", {r: {} for r in _ROLES})
    post = {}
    for role in _ROLES:
        pairs = [it for it in items if it["kind"] == "pair" and it["role"] == role]
        if len(pairs) < 3:
            raise ValueError(f"need at least 3 {role} pairs, got {len(pairs)}")
        T.e3b_build(model, tok, pairs + nonmoney, settings, model_name, model_io, layer=layer)
        store[role][layer] = T._STORE["e3b"]           # a fresh dict per call: safe to keep
        post[role] = _post_ids(tok, model_io, pairs[0])
    s, b = store["seller"][layer], store["buyer"][layer]
    common = sorted(set(s["d"]) & set(b["d"]))
    aligned = (s["n_prompt"] == b["n_prompt"] and s["reply_ids"] == b["reply_ids"]
               and len(post["seller"]) == len(post["buyer"]))
    b["cross_ok"] = aligned
    n_p = b["n_prompt"]
    Ds = torch.stack([s["d"][f] for f in common])            # (F, P, H)
    Db = torch.stack([b["d"][f] for f in common])
    vs, vb = Ds.mean(0), Db.mean(0)
    half = torch.arange(len(common)) % 2 == 0
    res = {"layer": layer, "fixtures": common, "n_prompt": n_p, "n_reply": len(b["reply_ids"]),
           "cross_aligned": aligned,
           "same_token_positions": sum(1 for x, y in zip(post["seller"], post["buyer"]) if x == y),
           "n_post": len(post["buyer"]),
           "norm_v": {r: {"prompt": float(v[:n_p].norm(dim=-1).mean()), "reply": float(v[n_p:].norm(dim=-1).mean())}
                      for r, v in (("seller", vs), ("buyer", vb))}}
    if aligned:
        def parts(x, y):
            return {"all": _cos(x, y), "prompt": _cos(x[:n_p], y[:n_p]), "reply": _cos(x[n_p:], y[n_p:]),
                    "token_mean": _cos(x.mean(0), y.mean(0))}
        res["cos_seller_buyer"] = parts(vs, vb)
        res["cos_seller_buyer_per_reply_token"] = _rowcos(vs[n_p:], vb[n_p:])
        # Noise references: the same comparison between halves of the fixtures.
        res["cos_split_half"] = {"seller": parts(Ds[half].mean(0), Ds[~half].mean(0)),
                                 "buyer": parts(Db[half].mean(0), Db[~half].mean(0)),
                                 "cross": parts(Ds[half].mean(0), Db[~half].mean(0))}
        # How much of the buyer v lies along the seller v (per token, then averaged): projection share.
        proj = (vb * vs).sum(-1) / vs.norm(dim=-1).clamp_min(1e-12)
        res["buyer_along_seller_share"] = {"prompt": float((proj[:n_p] / vb[:n_p].norm(dim=-1)).mean()),
                                           "reply": float((proj[n_p:] / vb[n_p:].norm(dim=-1)).mean())}
    res["cos_v_u"] = {r: _cos(v.mean(0), st_["u"]) for r, v, st_ in (("seller", vs, s), ("buyer", vb, b))}
    res["cos_fixture_diffs_mean"] = {}
    for r, D in (("seller", Ds), ("buyer", Db)):
        c = [_cos(D[i], D[j]) for i in range(len(common)) for j in range(i + 1, len(common))]
        res["cos_fixture_diffs_mean"][r] = sum(c) / len(c)
    return [res]


def _vectors(layer, fixture, loo=True):
    """Buyer vectors (core _e3b_vectors) plus the seller direction in three forms, all (P, H)."""
    st_ = T._STORE.get("buyers", {})
    if layer not in st_.get("buyer", {}):
        raise ValueError(f"run buyers_build at layer {layer} first")
    b, s = st_["buyer"][layer], st_["seller"][layer]
    V, used = _CORE_VECTORS(b, fixture, loo)
    if b.get("cross_ok") and fixture in s["d"]:
        sv = _CORE_VECTORS(s, fixture, loo)[0]["v_pos"]
        vb = V["v_pos"]
        nb = vb.norm(dim=-1, keepdim=True)
        V["seller"] = sv
        V["seller_norm"] = sv / sv.norm(dim=-1, keepdim=True).clamp_min(1e-12) * nb
        unit_b = vb / nb.clamp_min(1e-12)
        V["seller_perp"] = sv - (sv * unit_b).sum(-1, keepdim=True) * unit_b
    return V, used


BRIDGE_ROWS = (("none", "all", 0.0),
               ("own", "all", 1.0), ("own", "prompt", 1.0), ("own", "reply", 1.0),
               ("v_pos", "all", 0.5), ("v_pos", "all", 1.0), ("v_pos", "all", 1.5), ("v_pos", "all", 2.0),
               ("v_pos", "prompt", 1.0), ("v_pos", "reply", 1.0), ("v_mean", "all", 1.0), ("random", "all", 1.0),
               ("nonmoney", "all", 1.0), ("nonmoney", "all", 2.0),
               ("seller", "all", 0.5), ("seller", "all", 1.0), ("seller", "all", 2.0),
               ("seller", "prompt", 1.0), ("seller", "reply", 1.0),
               ("seller_norm", "all", 0.5), ("seller_norm", "all", 1.0), ("seller_norm", "all", 2.0),
               ("seller_norm", "prompt", 1.0), ("seller_norm", "reply", 1.0),
               ("seller_perp", "all", 1.0), ("seller_perp", "all", 2.0))
STEER_ROWS = (("v_pos", "prompt+reply", (0.5, 1.0, 2.0), 4), ("seller", "prompt+reply", (1.0,), 4),
              ("seller_norm", "prompt+reply", (1.0,), 4), ("nonmoney", "prompt+reply", (1.0,), 4))


def _with_vectors(layer, fn, rows_key, rows, **kw):
    """Run a core E3b task (e3b_bridge / e3b_steer) at `layer` with the cross-role vectors: the core
    code path is used unchanged, with _STORE["e3b"] pointed at the buyer set for this layer and
    tasks._e3b_vectors swapped for _vectors while it runs (restored afterwards; one request at a time)."""
    layer = int(layer)
    st_ = T._STORE.get("buyers", {}).get("buyer", {}).get(layer)
    if st_ is None:
        raise ValueError(f"run buyers_build at layer {layer} first")
    have = set(_vectors(layer, next(iter(st_["d"])), True)[0]) | {"none"}
    use = [r for r in rows if r[0] in have]
    skipped = sorted({r[0] for r in rows if r[0] not in have})
    saved = T._STORE.get("e3b")
    T._STORE["e3b"] = st_
    T._e3b_vectors = lambda _st, fixture, loo=True: _vectors(layer, fixture, loo)
    try:
        out = fn(**{rows_key: use}, **kw)
    finally:
        T._e3b_vectors, T._STORE["e3b"] = _CORE_VECTORS, saved
    for r in out:
        r["skipped_vectors"] = skipped
    return out


def buyers_bridge(model, tok, items, settings, model_name, model_io, layer=44, rows=BRIDGE_ROWS, **kw):
    """Part A at `layer`: core e3b_bridge (E2's setup: reply opening forced, greedy price)."""
    return _with_vectors(layer, lambda **k: T.e3b_bridge(model, tok, items, settings, model_name, model_io, **k),
                         "rows", [tuple(r) for r in rows], **kw)


def buyers_steer(model, tok, items, settings, model_name, model_io, layer=44, plan=STEER_ROWS, **kw):
    """Part B at `layer`: core e3b_steer (real sampled openings, frozen settings, xgrammar)."""
    return _with_vectors(layer, lambda **k: T.e3b_steer(model, tok, items, settings, model_name, model_io, **k),
                         "plan", [(v, sc, tuple(al), n) for v, sc, al, n in plan], **kw)


TASKS = {"buyers_build": buyers_build, "buyers_bridge": buyers_bridge, "buyers_steer": buyers_steer}
