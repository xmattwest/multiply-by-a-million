"""tasks_heads2.py (POD-SIDE plug-in): where do X57's key heads read the seller's limit from, and which heads carry
it for buyers? (config/interp/plan_next.md, session 2 "Follow the key heads"; STORY X57.) Exploratory.

X57's code is reused, not edited: `import tasks_heads` gives this plug-in its own copy of that module (tasks.py loads
plug-ins without registering them), and its build/patch run against stores kept here, one per role.

  heads2_build    tasks_heads.heads_build for one role, into that role's store.
  heads2_patch    tasks_heads.heads_patch (X57's row specs) against one role's store.
  heads2_sources  chosen heads: attention from the reply's opening tokens over every earlier position, x1 and x1e6
                  runs (eager attention for this step only); the same weighted by each position's value size through
                  the head's output projection; and an exact split of the head's x1e6 - x1 output difference at those
                  queries by source position. Every head at `scan` layers: the size of that output difference.
  heads2_path     path patching into receiver heads, two passes. Pass A runs the target sequence ("T") or the source
                  sequence ("S") with chosen post-limit positions taken from the other run: at the layers feeding the
                  receivers, with the rest of the post-limit tokens held at the run's own values there ("feed"), or
                  once at an earlier layer ("back"). It records the receivers' outputs on the post-limit tokens. Pass B
                  is the clean target run with only those outputs replaced; everything after them recomputes, as in
                  X57's head patches. Reference rows: receivers set to the source run's own outputs ("direct", X57's
                  top-k row) and plain residual patches ("resid").
"""
import copy
import time

import torch                                   # pod-only

import hooks
import tasks as T
import tasks_heads as H                        # a separate copy of X57's module (see above)

_STORES = {}


class _Stop(Exception):
    pass


def _use(role):
    """Point X57's functions at this role's store (they read the module global at call time)."""
    H._STORE = _STORES.setdefault(role, {"pairs": {}, "d": {}, "steer_layer": None})
    return H._STORE


def heads2_build(model, tok, items, settings, model_name, model_io, role="seller", **kw):
    _use(role)
    return H.heads_build(model, tok, items, settings, model_name, model_io, **kw)


def heads2_patch(model, tok, items, settings, model_name, model_io, role="seller", **kw):
    _use(role)
    return H.heads_patch(model, tok, items, settings, model_name, model_io, **kw)


# --- sources ------------------------------------------------------------------------------------------------
def _r(x, n=5):
    return [round(float(v), n) for v in x]


def _pre(vec, s0, k=8):
    """Positions before the limit, compactly: <bos>, the sum, the last k (the lead-in) and the top k."""
    v = vec[:s0]
    top = torch.topk(v, min(k, s0)).indices.tolist()
    return {"bos": round(float(v[0]), 5), "sum": round(float(v.sum()), 5), "last": _r(v[max(0, s0 - k):]),
            "top": [[int(i), round(float(v[i]), 5)] for i in top]}


def heads2_sources(model, tok, items, settings, model_name, model_io, heads=(), scan=(), **kw):
    heads = [tuple(int(v) for v in x) for x in heads]
    prev = getattr(model.config, "_attn_implementation", None)
    switched = prev != "eager"
    if switched:
        try:
            model.set_attn_implementation("eager")
        except Exception as e:                # noqa: BLE001
            return [{"item_id": it["item_id"], "fixture": it["fixture"], "error": f"no eager attention: {e}"} for it in items]
    blocks = model_io.decoder_layers(model)
    nH, hd = H._geometry(blocks)
    layers = sorted({L for L, _ in heads})
    scan = sorted({int(L) for L in scan})
    out = []
    try:
        for it in items:
            t0 = time.time()
            run = {}
            for c in ("x1", "x1e6"):
                side = it[c]
                E = T._r2_encode(tok, model_io, side["messages"], side["limit_text"], it["role"])
                n_prompt = len(T._chat_encode(tok, model_io, side["messages"], side["limit_text"], it["role"])["post"])
                sp, n = E["span"], len(E["ids"])
                p0 = sp[-1] + 1
                q = torch.arange(p0 + n_prompt, n, device=model.device)        # the reply's opening tokens
                g = {"w": {}, "v": {}, "z": {}}
                hs = []
                for L in layers:
                    at = blocks[L].self_attn
                    if getattr(at, "v_norm", None) is None:
                        raise ValueError(f"layer {L} shares its keys and values; not supported")
                    hs.append(at.register_forward_hook(lambda _m, _i, o, L=L: g["w"].__setitem__(L, o[1][0][:, q].float())))
                    hs.append(at.v_norm.register_forward_hook(lambda _m, _i, o, L=L: g["v"].__setitem__(L, o[0].float())))
                for L in sorted(set(layers) | set(scan)):
                    hs.append(blocks[L].self_attn.o_proj.register_forward_pre_hook(
                        lambda _m, a, L=L: g["z"].__setitem__(L, a[0][0, p0:].float())))
                try:
                    H._fwd(model, model_io, 1, input_ids=torch.tensor([E["ids"]], device=model.device), use_cache=False)
                finally:
                    for h in hs:
                        h.remove()
                if any(g["w"].get(L) is None for L in layers):
                    raise ValueError("no attention weights (eager attention did not return them)")
                run[c] = {"s0": sp[0], "nd": len(sp), "n": n, "p0": p0, "n_prompt": n_prompt, "g": g,
                          "toks": [tok.decode([i]) for i in E["ids"][p0:]] if c == "x1" else None}
            a, b = run["x1"], run["x1e6"]
            if a["s0"] != b["s0"] or a["n"] - a["p0"] != b["n"] - b["p0"] or a["n_prompt"] != b["n_prompt"]:
                raise ValueError(f"{it['fixture']}: the runs differ outside the limit digits")
            s0, n_post, n_prompt = a["s0"], a["n"] - a["p0"], a["n_prompt"]
            rec = {"item_id": it["item_id"], "fixture": it["fixture"], "role": it["role"], "s0": s0,
                   "n_digits": {"x1": a["nd"], "x1e6": b["nd"]}, "n_post": n_post, "n_prompt": n_prompt,
                   "post_tokens": a["toks"], "heads": {}, "scan": {}}
            for L, h in heads:
                W = blocks[L].self_attn.o_proj.weight[:, h * hd[L]:(h + 1) * hd[L]].float()       # (hidden, hd)
                per, o = {}, {}
                for c, R in (("x1", a), ("x1e6", b)):
                    w, v = R["g"]["w"][L][h], R["g"]["v"][L]                    # (Q, n), (n, kv, hd)
                    v = v[:, h // (nH // v.shape[1])]
                    u = v @ W.T                                                 # (n, hidden): value through W_O
                    z = R["g"]["z"][L][n_prompt:, h * hd[L]:(h + 1) * hd[L]]   # this head's output, reply queries
                    recon = float((w @ v - z).abs().max() / (z.abs().max() + 1e-12))
                    wn = w * u.norm(dim=-1)
                    vw = (wn / wn.sum(-1, keepdim=True)).mean(0)
                    m = w.mean(0)
                    sp_ = slice(R["s0"], R["s0"] + R["nd"])
                    reg = torch.stack([w[:, :1].sum(-1), w[:, 1:R["s0"]].sum(-1), w[:, sp_].sum(-1),
                                       w[:, R["p0"]:R["p0"] + n_prompt].sum(-1), w[:, R["p0"] + n_prompt:].sum(-1)], -1)
                    per[c] = {"mass_pre": _pre(m, R["s0"]), "mass_digits": _r(m[sp_]), "mass_post": _r(m[R["p0"]:]),
                              "vw_pre": _pre(vw, R["s0"]), "vw_digits": _r(vw[sp_]), "vw_post": _r(vw[R["p0"]:]),
                              "q_regions": [_r(x, 4) for x in reg], "out_norm": round(float((w @ u).norm(dim=-1).mean()), 4),
                              "recon_err": recon}
                    o[c] = (w, u, w @ u, sp_, R["p0"])
                D = o["x1e6"][2] - o["x1"][2]                                   # (Q, hidden)
                den = float((D * D).sum())
                cc = {c: o[c][0] * (D @ o[c][1].T) for c in o}                  # (Q, n): position k's part along D
                pre = (cc["x1e6"][:, :s0] - cc["x1"][:, :s0]).sum(0)
                dig = float(cc["x1e6"][:, o["x1e6"][3]].sum() - cc["x1"][:, o["x1"][3]].sum())
                post = (cc["x1e6"][:, o["x1e6"][4]:] - cc["x1"][:, o["x1"][4]:]).sum(0)
                tot = float(pre.sum()) + dig + float(post.sum())
                sc = 1.0 / den if den else 0.0
                per["diff"] = {"den": den, "pre": _pre(pre * sc, s0), "digits": round(dig * sc, 5), "post": _r(post * sc),
                               "sum_err": abs(tot - den) / den if den else None,
                               "dnorm": round(float(D.norm(dim=-1).mean()), 4)}
                rec["heads"][f"{L}.{h}"] = per
            for L in scan:                     # every head: |W_O,h (z_x1e6 - z_x1)| on the reply / prompt queries
                Wr = blocks[L].self_attn.o_proj.weight.float().view(-1, nH, hd[L])
                za, zb = a["g"]["z"][L].view(n_post, nH, hd[L]), b["g"]["z"][L].view(n_post, nH, hd[L])
                dn = torch.einsum("qhd,ohd->qho", zb - za, Wr).norm(dim=-1)        # (n_post, heads)
                n1 = torch.einsum("qhd,ohd->qho", za[n_prompt:], Wr).norm(dim=-1)
                n6 = torch.einsum("qhd,ohd->qho", zb[n_prompt:], Wr).norm(dim=-1)
                rec["scan"][str(L)] = {"d_reply": _r(dn[n_prompt:].mean(0), 4), "d_prompt": _r(dn[:n_prompt].mean(0), 4),
                                       "n_x1": _r(n1.mean(0), 4), "n_x1e6": _r(n6.mean(0), 4)}
            rec["secs"] = round(time.time() - t0, 1)
            out.append(rec)
    finally:
        if switched:
            model.set_attn_implementation(prev)
    return out


# --- path patching ------------------------------------------------------------------------------------------
def _positions(spec, fixture):
    P = spec.get("P", [])
    if isinstance(P, dict):
        P = P.get(fixture)
    return None if P is None else sorted({int(x) for x in P})


def heads2_path(model, tok, items, settings, model_name, model_io, recv=(), rows=(), batch=16, role="seller", **kw):
    """Item: one fixture and direction (E2's source -> target). Rows (laptop specs):
      {"n", "kind": "path", "seq": "T"|"S", "mode": "feed"|"back", "L": layer (back), "P": offsets or {fixture: offsets},
       "use": [[L, h], ...] (optional subset of the receivers to patch in pass B)}
      {"n", "kind": "direct", "use": ...}       receivers <- the source run's own outputs
      {"n", "kind": "resid", "L": layer}        the source run's residual on every post-limit token (not via the heads)
    Offsets count post-limit tokens (0 = the token after the last digit). A path row whose P is a dict without the
    item's fixture is left out."""
    _use(role)
    blocks = model_io.decoder_layers(model)
    nH, hd = H._geometry(blocks)
    recv = sorted({tuple(int(v) for v in x) for x in recv})
    if not recv:
        raise ValueError("no receiver heads")
    rl = sorted({L for L, _ in recv})
    feed = sorted({L - 1 for L in rl if L >= 1})
    lmax = rl[-1]
    for s in rows:
        if s.get("kind") == "path" and s.get("mode") == "back" and not 0 <= int(s["L"]) < lmax:
            raise ValueError(f"{s['n']}: a back layer must lie below the last receiver layer {lmax}")
    need = {x: set() for x in ("resid", "attn", "mlp", "head")}
    need["resid"] = set(feed) | {int(s["L"]) for s in rows if s.get("kind") in ("path", "resid") and "L" in s}
    need["head"] = set(rl)
    dev = model.device
    out = []
    for it in items:
        t0 = time.time()
        f = it["fixture"]
        PR = H._STORE["pairs"].get(f)
        base_rec = {"item_id": it["item_id"], "fixture": f, "role": it["role"], "direction": it["direction"],
                    "base_limit": it["base_limit"], "target_condition": it["target"]["condition"]}
        if PR is None:
            out.append(dict(base_rec, skipped="no usable prices from heads2_build"))
            continue
        S = T._r2_encode(tok, model_io, it["source"]["messages"], it["source"]["limit_text"], it["role"])
        Tg = T._r2_encode(tok, model_io, it["target"]["messages"], it["target"]["limit_text"], it["role"])
        s0, ns, nt, n_post = Tg["span"][0], len(S["span"]), len(Tg["span"]), PR["n_post"]
        if S["span"][0] != s0 or S["ids"][:s0] != Tg["ids"][:s0] or S["ids"][S["span"][-1] + 1:] != Tg["ids"][Tg["span"][-1] + 1:] \
                or len(Tg["ids"]) - Tg["span"][-1] - 1 != n_post:
            raise ValueError(f"{it['item_id']}: source and target differ outside the limit digits")
        cu_t = "x1" if it["target"]["condition"] == "D2" else "x1e6"
        cu_s = "x1e6" if cu_t == "x1" else "x1"
        pre_t, ta, tb = PR["cand"][cu_t]
        pre_s, sa, sb = PR["cand"][cu_s]
        vals, ld = {}, {}
        for X, E, pre, a_, b_ in (("S", S, pre_s, sa, sb), ("T", Tg, pre_t, ta, tb)):
            base1, _ = T._batched_prefix_cache(model, model_io, Tg["ids"][:s0], 1)
            n_d = ns if X == "S" else nt
            vals[X], ld[X] = H._source(model, model_io, blocks, E["ids"][s0:] + pre, base1, slice(n_d, n_d + n_post),
                                       need, a_, b_, len(pre) + 1)
            del base1
        base, how = T._batched_prefix_cache(model, model_io, Tg["ids"][:s0], batch)
        other = {"T": "S", "S": "T"}
        hooks_ok = True
        caps = {}
        prow = []
        for s in rows:
            if s.get("kind") == "path":
                P = _positions(s, f)
                if P is not None:
                    prow.append((s, P))
        # pass A
        for X, ids_X, nX in (("T", Tg["ids"][s0:], nt), ("S", S["ids"][s0:], ns)):
            group = [(s, P) for s, P in prow if s["seq"] == X]
            if not group:
                continue
            post = slice(nX, nX + n_post)
            x = torch.tensor([ids_X], device=dev).expand(batch, -1).contiguous()
            own, oth = vals[X]["resid"], vals[other[X]]["resid"]
            for b0 in range(0, len(group), batch):
                chunk = group[b0:b0 + batch]
                got = [dict() for _ in chunk]
                fired, hs = [0], []

                def lay(F):
                    def fn(_m, _i, o):
                        fired[0] += 1
                        y = hooks._block_output_residual(o).clone()
                        for r, (s, P) in enumerate(chunk):
                            idx = torch.tensor(P, device=y.device, dtype=torch.long)
                            if s["mode"] == "feed" and F in feed:
                                y[r, post] = own[F].to(y.dtype)
                                if P:
                                    y[r, nX + idx] = oth[F][idx].to(y.dtype)
                            elif s["mode"] == "back" and int(s["L"]) == F and P:
                                y[r, nX + idx] = oth[F][idx].to(y.dtype)
                        return hooks._replace_block_output(o, y)
                    return fn

                def cap(L):
                    def fn(_m, a):
                        fired[0] += 1
                        for r in range(len(chunk)):
                            for L_, h in recv:
                                if L_ == L:
                                    got[r][(L, h)] = a[0][r, post, h * hd[L]:(h + 1) * hd[L]].detach().clone()
                    return fn

                def stop(_m, _i, _o):
                    raise _Stop
                pl = sorted(set(feed) | {int(s["L"]) for s, _ in chunk if s["mode"] == "back"})
                for F in pl:
                    hs.append(blocks[F].register_forward_hook(lay(F)))
                for L in rl:
                    hs.append(blocks[L].self_attn.o_proj.register_forward_pre_hook(cap(L)))
                hs.append(blocks[lmax].register_forward_hook(stop))
                cache = copy.deepcopy(base)
                try:
                    H._fwd(model, model_io, 1, input_ids=x, past_key_values=cache, use_cache=True)
                    hooks_ok = False                  # the stop hook should have fired
                except _Stop:
                    pass
                finally:
                    for h in hs:
                        h.remove()
                hooks_ok &= fired[0] == len(pl) + len(rl)
                for r, (s, _P) in enumerate(chunk):
                    caps[s["n"]] = got[r]
                del cache
        # pass B
        specs = [s for s in rows if s.get("kind") != "path" or s["n"] in caps] + [{"n": "clean", "kind": "clean"}]
        while len(specs) % batch:
            specs.append({"n": "clean", "kind": "clean"})
        postT = slice(nt, nt + n_post)
        suf = Tg["ids"][s0:] + pre_t
        k = len(pre_t) + 1
        x = torch.tensor([suf], device=dev).expand(batch, -1).contiguous()
        res = []
        for b0 in range(0, len(specs), batch):
            chunk = specs[b0:b0 + batch]
            tab = {L: [] for L in rl}
            rtab = {}
            for r, s in enumerate(chunk):
                use = {tuple(int(v) for v in u) for u in s["use"]} if s.get("use") else set(recv)
                if s["kind"] == "path":
                    for (L, h), v in caps[s["n"]].items():
                        if (L, h) in use:
                            tab[L].append((r, h, v))
                elif s["kind"] == "direct":
                    for L, h in sorted(use):
                        tab[L].append((r, h, vals["S"]["head"][L][:, h * hd[L]:(h + 1) * hd[L]]))
                elif s["kind"] == "resid":
                    rtab.setdefault(int(s["L"]), []).append(r)
            fired, hs = [0], []

            def hpatch(L, pat):
                def fn(_m, a):
                    fired[0] += 1
                    y = a[0].clone()
                    for r, h, v in pat:
                        y[r, postT, h * hd[L]:(h + 1) * hd[L]] = v.to(y.dtype)
                    return (y,) + tuple(a[1:])
                return fn

            def rpatch(L, rs):
                def fn(_m, _i, o):
                    fired[0] += 1
                    y = hooks._block_output_residual(o).clone()
                    for r in rs:
                        y[r, postT] = vals["S"]["resid"][L].to(y.dtype)
                    return hooks._replace_block_output(o, y)
                return fn
            for L, pat in tab.items():
                if pat:
                    hs.append(blocks[L].self_attn.o_proj.register_forward_pre_hook(hpatch(L, pat)))
            for L, rs in rtab.items():
                hs.append(blocks[L].register_forward_hook(rpatch(L, rs)))
            cache = copy.deepcopy(base)
            try:
                o = H._fwd(model, model_io, k, input_ids=x, past_key_values=cache, use_cache=True)
            finally:
                for h in hs:
                    h.remove()
            hooks_ok &= fired[0] == len(hs)
            lp = torch.log_softmax(o.logits[:, -1].float(), -1)
            for r, s in enumerate(chunk):
                res.append({"n": s["n"], "ld": round(float(lp[r, tb] - lp[r, ta]), 5), "lpa": round(float(lp[r, ta]), 4),
                            "lpb": round(float(lp[r, tb]), 4)})
            del cache, o
        del base, vals
        clean = [r["ld"] for r in res if r["n"] == "clean"]
        out.append(dict(base_rec, ld_source=round(ld["S"], 5), ld_clean=clean[0], ld_clean_unbatched=round(ld["T"], 5),
                        clean_spread=round(max(clean) - min(clean), 5), n_rows=len(specs), batch=batch, prefix_cache=how,
                        hooks_ok=hooks_ok, recv=[list(x) for x in recv],
                        rows=[r for r in res if r["n"] != "clean"] + [r for r in res if r["n"] == "clean"][:1],
                        secs=round(time.time() - t0, 1)))
    return out


def _each(fn):
    """The pod entry for a new task: item by item, and an item that fails becomes an error record, so one bug in a
    step does not end the paid session (the later steps still run)."""
    def run(model, tok, items, *a, **kw):
        out = []
        for it in items:
            try:
                out += fn(model, tok, [it], *a, **kw)
            except Exception as e:            # noqa: BLE001
                out.append(dict({k: it.get(k) for k in ("item_id", "fixture", "role", "direction")},
                                error=f"{type(e).__name__}: {e}"[:400]))
        return out
    return run


TASKS = {"heads2_build": heads2_build, "heads2_patch": heads2_patch, "heads2_sources": _each(heads2_sources),
         "heads2_path": _each(heads2_path)}
