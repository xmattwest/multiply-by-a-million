"""tasks_e5.py (POD): E5, the appraisal inside Gemma 4 31B (Study 3's A1 line). Details: scripts/interp_e5_laptop.py.
Exploratory. Four prompts per fixture and role, Study 3's own message-1 prompts: N1 / N6 = A0 (Study 2's prompt) at
$100 / $100M, P1 / P6 = A1 (the same + "An independent appraisal values the lot at {V}. Both sides know this." in the
system message).

E4's tasks are reused unchanged, with two differences that this module supplies:
  - the appraisal line comes BEFORE the limit (system message), so the four prompts differ there too. The aligned
    positions A = H (after the limit up to "No messages") + S (from "\\n\\nNo messages") + reply are still the same
    tokens in all four (checked per fixture); everything before the limit may differ.
  - so every forced-opening pass needs its target's OWN prefix cache (e5_patch; E4 shared one, since its prompts were
    identical before the limit). e4_steer, e4_build, e4_attn already encode and run each prompt on its own.
Each role has its own store (directions, sums), so sellers and buyers never mix.
Appraisal digits are reported as "lo" = "hi" and the appraisal line as "R", so E4's attention step reads them.
"""
import contextlib
import copy
import time

import torch                                   # pod-only

import tasks as T
import tasks_e4 as E4
from tasks_heads import _fwd, _geometry         # stateless helpers

APPRAISAL = "An independent appraisal values the lot at {v} dollars."
_STORES = {}


def _fresh():
    return {"pairs": {}, "d": {}, "acc": {}, "n": 0, "u": None, "u_n": 0, "keep": None, "segs": None}


def encode(tok, model_io, side, role):
    text, _ = model_io.render_messages(tok, side["messages"])
    ch = T._chat_encode(tok, model_io, side["messages"], side["limit_text"], role)
    full = text + T._PREFIX
    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = list(enc["input_ids"]), [tuple(o) for o in enc["offset_mapping"]]
    if ids[:len(ch["ids"])] != ch["ids"] or full.count("\n\nNo messages") != 1:
        raise ValueError("unexpected prompt layout")
    span = T._locate(full, offs, side["limit_text"], role)
    k_s = E4._start(offs, full.index("\n\nNo messages"))
    e = {"ids": ids, "span": span, "s0": span[0], "n_chat": len(ch["ids"]), "lo": [], "hi": [], "R": []}
    v = side.get("appraisal")
    if v:
        sent = APPRAISAL.format(v=v)
        if full.count(sent) != 1:
            raise ValueError("appraisal sentence not found exactly once")
        c0 = full.index(sent)
        v0 = c0 + len(APPRAISAL.split("{v}")[0])
        e["lo"] = e["hi"] = E4._span(offs, v0, v0 + len(v))
        c1 = full.index("\n", c0)                                     # the line ends with "Both sides know this."
        e["R"] = [k for k, (s, t) in enumerate(offs) if s < c1 and t > c0]
        if not e["R"] or not e["R"][-1] < span[0]:
            raise ValueError("the appraisal line is not before the limit")
    H = list(range(span[-1] + 1, k_s))
    e.update(A=H + list(range(k_s, len(ids))), n_H=len(H), n_S=len(ch["ids"]) - k_s, n_reply=len(ids) - len(ch["ids"]))
    if not H or e["n_S"] <= 0:
        raise ValueError("empty segment")
    return e


def encode_fixture(tok, model_io, it):
    E = {k: encode(tok, model_io, it["prompts"][k], it["role"]) for k in it["prompts"]}
    r = next(iter(E.values()))
    key = lambda e: ([e["ids"][p] for p in e["A"]], e["n_H"], e["n_S"], e["n_reply"])   # noqa: E731
    if any(key(e) != key(r) for e in E.values()):
        raise ValueError(f"{it['fixture']}: prompts differ after the limit")
    return E


@contextlib.contextmanager
def _mode(role):
    """Run E4's code with this module's encoder and the role's own store."""
    saved = (E4.encode, E4.encode_fixture, E4._STORE)
    E4.encode, E4.encode_fixture = encode, encode_fixture
    E4._STORE = _STORES.setdefault(role, _fresh())
    try:
        yield
    finally:
        _STORES[role] = E4._STORE
        E4.encode, E4.encode_fixture, E4._STORE = saved


def e5_build(model, tok, items, settings, model_name, model_io, role="seller", **kw):
    with _mode(role):
        return E4.e4_build(model, tok, items, settings, model_name, model_io, **kw)


def e5_stats(model, tok, items, settings, model_name, model_io, role="seller", **kw):
    with _mode(role):
        return [dict(r, role=role) for r in E4.e4_stats(model, tok, items, settings, model_name, model_io, **kw)]


def e5_attn(model, tok, items, settings, model_name, model_io, role="seller", **kw):
    with _mode(role):
        return E4.e4_attn(model, tok, items, settings, model_name, model_io, **kw)


def e5_steer(model, tok, items, settings, model_name, model_io, role="seller", **kw):
    with _mode(role):
        return E4.e4_steer(model, tok, items, settings, model_name, model_io, **kw)


def e5_patch(model, tok, items, settings, model_name, model_io, rows=None, batch=16, loo=True, role="seller", **kw):
    """E4's e4_patch with one change: each target runs on its OWN prefix cache (its tokens before the limit), since the
    appraisal line makes the four prompts differ there. Rows, readout and records as e4_patch."""
    with _mode(role):
        rows, blocks = rows or {}, model_io.decoder_layers(model)
        hd, digits = _geometry(blocks)[1], T._digit_ids(tok)
        out = []
        for it in items:
            t0, f = time.time(), it["fixture"]
            Pq = E4._STORE["pairs"].get(f)
            rec = {"item_id": it["item_id"], "fixture": f, "base_limit": it["base_limit"], "role": role}
            if Pq is None:
                out.append(dict(rec, skipped="no usable prices from e5_build"))
                continue
            E = encode_fixture(tok, model_io, it)
            sg, need, vl = E4.segs(E["N1"]), {}, set()
            for op in (op for specs in rows.values() for spec in specs for op in spec.get("ops", ())):
                if op[0] == "vec":
                    vl.add(int(op[1]))
                for p in (op[5:6] if op[0] == "set" else op[6:8] if op[0] == "add" else ()):
                    need.setdefault(p, set()).add((op[1], int(op[2])))
            vals = {p: E4._capture(model, model_io, blocks, E[p]["ids"], E[p]["A"], sorted(need[p]))[0] for p in need}
            V = {L: E4.vectors(f, L, loo) for L in vl}
            res, hows = {}, {}
            for tgt, specs in rows.items():
                e = E[tgt]
                s0 = e["s0"]
                base, hows[tgt] = T._batched_prefix_cache(model, model_io, e["ids"][:s0], batch)
                pre, ta, tb = Pq["cand"][E4.SCALE[tgt]]
                apos = torch.tensor([p - s0 for p in e["A"]], device=model.device)
                allspecs = list(specs) + [{"n": "clean"}] * (batch - len(specs) % batch)
                x = torch.tensor([e["ids"][s0:] + pre], device=model.device).expand(batch, -1).contiguous()
                got, ok = [], True
                for b0 in range(0, len(allspecs), batch):
                    chunk = allspecs[b0:b0 + batch]
                    hs, fired = E4._arm(blocks, hd, chunk, vals, {L: V[L][0] for L in V}, apos, sg)
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
                             "lpb": round(float(lp[r, tb]), 4),
                             "p1": [round(float(p0[r, i]), 4) for i in digits] if None not in digits else None}
                            for r, s in enumerate(chunk)]
                    del cache, o, lg
                del base
                clean = [g["ld"] for g in got if g["n"] == "clean"]
                res[tgt] = {"ld_clean": clean[0], "clean_spread": round(max(clean) - min(clean), 5), "hooks_ok": ok,
                            "rows": [g for g in got if g["n"] != "clean"] + [g for g in got if g["n"] == "clean"][:1]}
            used = all(v[1] for v in V.values()) if V else None
            del vals, V
            out.append(dict(rec, ld_build=Pq["ld"], batch=batch, prefix_cache=sorted(set(hows.values())), loo=used,
                            targets=res, secs=round(time.time() - t0, 1)))
        return out


TASKS = {"e5_build": e5_build, "e5_stats": e5_stats, "e5_attn": e5_attn, "e5_steer": e5_steer, "e5_patch": e5_patch}
