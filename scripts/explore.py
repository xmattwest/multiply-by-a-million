"""Exploratory sweep over the Qwen runs (Study 1/2, pilot/real). Not pre-registered.
Run: python3 scripts/explore.py   (FOLLOW=1 adds sections H-K)
"""
import json, os, re, statistics as st
from collections import Counter, defaultdict
D = {"S1": "data/runs/qwen3.8-27b-fp8-20261003T180835Z", "S1p": "data/pilot/pilot-qwen3.8-27b-fp8-20261003T173737Z",
     "S2": "data/runs/s2-qwen3.8-27b-fp8-20261003T223042Z", "S2p": "data/pilot/s2-pilot-qwen3.8-27b-fp8-20261003T220356Z"}
SC = ("x1e2", "x1e5", "x1e8"); LAB = {"x1e2": "$100", "x1e5": "$100K", "x1e8": "$100M"}
def load(R):
    G = [json.loads(l) for l in open(f"{R}/games.jsonl") if l.strip()]
    G = [g for g in G if g["terminal"] != "INFRASTRUCTURE_VOID"]
    calls = {}
    for l in open(f"{R}/calls.jsonl"):
        c = json.loads(l); calls.setdefault(c["call_id"], {})[c["schema"]] = c
    for g in G:
        mv = []
        for cid in g["call_ids"]:
            req, res = calls[cid]["call-request/1"], calls[cid]["call-result/1"]
            try: m = json.loads(json.loads(res["response_text"])["choices"][0]["message"]["content"])
            except Exception: m = {}
            m["role"] = req["role"]; mv.append(m)
        g["moves"] = mv
    return G
f = lambda x: f"{x:+.3f}" if isinstance(x, float) else str(x)
m = lambda v: st.fmean(v) if v else float("nan")
DATA = {k: load(R) for k, R in D.items()}
def sh(g): return g["seller_share"]["float"] if g["seller_share"] else None
def pos(g, p): return (p - g["F"]) / (g["C"] - g["F"])

print("A. Mean seller share by opener (n deals), and opener advantage = seller-opened − buyer-opened")
for k, G in DATA.items():
    row = []
    for s in SC:
        a = [sh(g) for g in G if g["scale"] == s and g["opener"] == "seller" and sh(g) is not None]
        b = [sh(g) for g in G if g["scale"] == s and g["opener"] == "buyer" and sh(g) is not None]
        row.append(f"{LAB[s]}: S-open {m(a):.3f}({len(a)}) B-open {m(b):.3f}({len(b)}) adv {m(a)-m(b):+.3f}")
    print(f"  {k:<4} " + " | ".join(row))

print("\nB. Opener's first offer: position in range [0=floor,1=ceiling]; markup vs own limit")
for k, G in DATA.items():
    row = []
    for s in SC:
        so = [g for g in G if g["scale"] == s and g["opener"] == "seller"]
        bo = [g for g in G if g["scale"] == s and g["opener"] == "buyer"]
        row.append(f"{LAB[s]}: ask {m([pos(g, g['offers'][0][2]) for g in so]):.2f} ({m([g['offers'][0][2]/g['F'] for g in so]):.3f}×F)"
                   f" bid {m([pos(g, g['offers'][0][2]) for g in bo]):.2f} ({m([g['offers'][0][2]/g['C'] for g in bo]):.3f}×C)")
    print(f"  {k:<4} " + " | ".join(row))

print("\nC. Concessions (fractions of range): responder's first counter position; total concession by seller / buyer; offers per game")
for k, G in DATA.items():
    row = []
    for s in SC:
        gs = [g for g in G if g["scale"] == s]
        def tot(g, side):
            o = [p for _, r, p in g["offers"] if r == side]
            if len(o) < 1: return None
            last = g["price"] if g["terminal"] == "AGREEMENT" else o[-1]
            return abs(o[0] - last) / (g["C"] - g["F"])
        cs = [x for g in gs if (x := tot(g, "seller")) is not None]; cb = [x for g in gs if (x := tot(g, "buyer")) is not None]
        rs = [pos(g, next(p for _, r, p in g["offers"] if r == "seller")) for g in gs if g["opener"] == "buyer" and any(r == "seller" for _, r, _ in g["offers"])]
        rb = [pos(g, next(p for _, r, p in g["offers"] if r == "buyer")) for g in gs if g["opener"] == "seller" and any(r == "buyer" for _, r, _ in g["offers"])]
        row.append(f"{LAB[s]}: resp-ask {m(rs):.2f} resp-bid {m(rb):.2f} concede S {m(cs):.2f} B {m(cb):.2f} offers {m([len(g['offers']) for g in gs]):.1f}")
    print(f"  {k:<4} " + " | ".join(row))

print("\nD. Endings: terminal counts; who accepted; whose number became the price (opener's / responder's)")
for k, G in DATA.items():
    row = []
    for s in SC:
        gs = [g for g in G if g["scale"] == s]
        ag = [g for g in gs if g["terminal"] == "AGREEMENT"]
        acc = Counter(g["ended_by"] for g in ag)
        prop = Counter("opener" if g["offers"][-1][1] == g["opener"] else "responder" for g in ag)
        walk = Counter(f"{g['ended_by']}" for g in gs if g["terminal"] == "WALKAWAY")
        row.append(f"{LAB[s]}: {dict(Counter(g['terminal'] for g in gs))} acc-by {dict(acc)} price-from {dict(prop)} walk-by {dict(walk)}")
    print(f"  {k:<4} " + "\n       ".join(row))

print("\nE. Deals exactly at a limit / past a limit; final price = 'round' (multiple of 5% of V)")
for k, G in DATA.items():
    row = []
    for s in SC:
        ag = [g for g in G if g["scale"] == s and g["terminal"] == "AGREEMENT"]
        row.append(f"{LAB[s]}: @F {sum(g['price']==g['F'] for g in ag)} @C {sum(g['price']==g['C'] for g in ag)} past {sum(not g['F']<=g['price']<=g['C'] for g in ag)}"
                   f" round5% {sum(g['price'] % (g['V']//20) == 0 for g in ag)}/{len(ag)}")
    print(f"  {k:<4} " + " | ".join(row))

print("\nF. Offer granularity: median |step| between a side's consecutive offers, in % of V; share of offers on a 1%-of-V grid")
for k, G in DATA.items():
    row = []
    for s in SC:
        gs = [g for g in G if g["scale"] == s]
        steps = []
        for g in gs:
            for side in ("seller", "buyer"):
                o = [p for _, r, p in g["offers"] if r == side]
                steps += [abs(b - a) / g["V"] * 100 for a, b in zip(o, o[1:])]
        offs = [p for g in gs for _, _, p in g["offers"]]
        grid = m([p % (g["V"] // 100) == 0 for g in gs for _, _, p in g["offers"]])
        small = m([x < 1 for x in steps])
        row.append(f"{LAB[s]}: step med {st.median(steps):.1f}% <1%-steps {small:.2f} on1%grid {grid:.2f}")
    print(f"  {k:<4} " + " | ".join(row))

print("\nG. Language: words/message; % messages with term; own-limit number stated in text")
TERMS = {"budget": r"budget", "floor/limit": r"\bfloor\b|\blimit\b|\bminimum\b|\bbottom\b", "final/best": r"\bfinal\b|best and final|\bbest offer\b",
         "walk": r"walk away|\bwalk\b", "fair": r"\bfair\b", "risk": r"\brisk", "invest": r"invest", "significant": r"significant|substantial|considerable",
         "market": r"market", "value": r"\bvalue\b|\bworth\b", "close today": r"\btoday\b|close the deal|move forward"}
def num_in(text, x):
    t = text.replace(",", "")
    return str(x) in re.findall(r"\d+", t) or (x % 1_000_000 == 0 and re.search(rf"\b{x//1_000_000}(\.0)?\s*(million|m\b)", text, re.I) is not None)
for k, G in DATA.items():
    print(f"  {k}")
    for s in SC:
        msgs = [(g, mv) for g in G if g["scale"] == s for mv in g["moves"]]
        txt = [mv.get("message", "") or "" for _, mv in msgs]
        wc = m([len(t.split()) for t in txt])
        terms = " ".join(f"{n} {m([bool(re.search(p, t, re.I)) for t in txt])*100:.0f}" for n, p in TERMS.items())
        leakS = sum(num_in(mv.get("message", "") or "", g["F"]) for g, mv in msgs if mv["role"] == "seller")
        leakB = sum(num_in(mv.get("message", "") or "", g["C"]) for g, mv in msgs if mv["role"] == "buyer")
        print(f"    {LAB[s]:<5} words {wc:.0f} | {terms} | own-limit stated: seller {leakS}, buyer {leakB}")

if os.environ.get("FOLLOW"):
    import random, sys
    sys.path.insert(0, "scripts"); import analyze as A
    from fractions import Fraction as Fr
    def flip_test(G, label):
        by = {(g["fixture"], g["opener"], g["scale"]): g for g in G}
        fx = sorted({g["fixture"] for g in G})
        d = []
        for f_ in fx:
            try:
                ss = [Fr(by[(f_, o, "x1e8")]["seller_share"]["num"], by[(f_, o, "x1e8")]["seller_share"]["den"])
                      - Fr(by[(f_, o, "x1e2")]["seller_share"]["num"], by[(f_, o, "x1e2")]["seller_share"]["den"]) for o in ("seller", "buyer")]
            except (KeyError, TypeError): continue
            d.append(ss[0] - ss[1])
        p, _, n = A.wilcoxon_signflip(d, random.Random(1)) if d else (None, None, 0)
        print(f"  {label}: interaction (seller-opened shift − buyer-opened shift) mean {m([float(x) for x in d]):+.3f}, n {len(d)} fixtures, p {p}")
    print("H. Opener × stakes interaction on each dataset (paired by fixture)")
    for k, G in DATA.items(): flip_test(G, k)
    print("\nI. Opener's first offer, $100M vs $100, per fixture: seller markup over F, buyer discount under C")
    for k, G in DATA.items():
        by = {(g["fixture"], g["opener"], g["scale"]): g for g in G}
        for side, fn in (("seller", lambda g: g["offers"][0][2] / g["F"] - 1), ("buyer", lambda g: 1 - g["offers"][0][2] / g["C"])):
            d = [fn(by[(f_, side, "x1e8")]) - fn(by[(f_, side, "x1e2")]) for f_ in {g["fixture"] for g in G}
                 if (f_, side, "x1e8") in by and (f_, side, "x1e2") in by]
            fewer = sum(x < 0 for x in d)
            print(f"  {k:<4} {side} opens: gap from own limit at $100 {m([fn(g) for g in G if g['opener']==side and g['scale']=='x1e2'])*100:.1f}%"
                  f" -> $100M {m([fn(g) for g in G if g['opener']==side and g['scale']=='x1e8'])*100:.1f}%; smaller at $100M in {fewer}/{len(d)} fixtures")
    print("\nJ. Seller-opened games: corr(opening ask position, final share)")
    for k in ("S1", "S2"):
        r = [(pos(g, g["offers"][0][2]), sh(g)) for g in DATA[k] if g["opener"] == "seller" and sh(g) is not None]
        print(f"  {k}: r = {st.correlation(*zip(*r)):.2f} (n {len(r)})")
    print("\nK. S2 $100M: seller messages containing the seller's own floor number")
    G = DATA["S2"]; hits = []
    for g in G:
        if g["scale"] != "x1e8": continue
        for i, mv in enumerate(g["moves"]):
            if mv["role"] == "seller" and num_in(mv.get("message", "") or "", g["F"]):
                hits.append((g["game_id"], i + 1, mv.get("action"), mv.get("price"), g["price"], g["terminal"], (mv.get("message") or "")[:150]))
    print(f"  {len(hits)} messages in {len({h[0] for h in hits})} games; deal at F in {sum(1 for gid in {h[0] for h in hits} if next(g for g in G if g['game_id']==gid)['price']==next(g for g in G if g['game_id']==gid)['F'])} of them")
    for h in hits[:8]: print("   ", h[0], "move", h[1], h[2], h[3], "| end", h[5], h[4], "|", h[6])
    print("  deals exactly at F at $100M:", [(g["game_id"], g["opener"], g["ended_by"]) for g in G if g["scale"]=="x1e8" and g["terminal"]=="AGREEMENT" and g["price"]==g["F"]])
    print("  walkaways:", [(g["game_id"], g["n_messages"], g["offers"][-1][1], round(g["offers"][-1][2]/g["F"], 3)) for g in G if g["terminal"]=="WALKAWAY"])
