#!/usr/bin/env python3
"""analyze_probe.py: summary of one exploratory probe run (STORY §12). Not pre-registered.

    python3 scripts/analyze_probe.py data/explore/X-<name>-<model>-<stamp>

Q1 (conditions A vs N): does "openers turn modest at $100M" appear only
without the appraisal? Opening gaps, opener's advantage and the
opener x scale interaction, per condition.
Q3 (LS, LB vs N): how much of the possible gain the informed side captures
(exploitation rate), at $100 vs $100M; whether it says it knows; endings.
p-values are exploratory signals (sign-flip Wilcoxon from analyze.py).
Writes derived/explore/<run_id>/summary.json.
"""
import json
import os
import random
import re
import statistics as st
import sys
from fractions import Fraction as Fr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import analyze as A  # noqa: E402

LO, HI = "x1e2", "x1e8"
OUT = {}


def load(run):
    games = [json.loads(l) for l in open(os.path.join(run, "games.jsonl")) if l.strip()]
    games = [g for g in games if g["terminal"] != "INFRASTRUCTURE_VOID"]
    text = {}
    for l in open(os.path.join(run, "calls.jsonl")):
        c = json.loads(l)
        if c["schema"] == "call-result/1":
            try:
                text[c["call_id"]] = json.loads(json.loads(c["response_text"])["choices"][0]["message"]["content"])
            except (ValueError, KeyError, TypeError):
                text[c["call_id"]] = {}
    for g in games:
        g["moves"] = [text.get(cid, {}) for cid in g["call_ids"]]
    return games


def share(g):
    s = g.get("seller_share")
    return Fr(s["num"], s["den"]) if s else None


def mean(v):
    v = [float(x) for x in v if x is not None]
    return st.fmean(v) if v else float("nan")


def gap(g):
    if not g["offers"] or g["offers"][0][1] != g["opener"]:
        return None
    p = g["offers"][0][2]
    return 100 * (p / g["F"] - 1) if g["opener"] == "seller" else 100 * (1 - p / g["C"])


def test(d, label):
    if len(d) < 3:
        return {"n": len(d), "mean": mean(d), "p": None}
    p, _, n = A.wilcoxon_signflip(d, random.Random(A.seed("probe|" + label, "x")))
    return {"n": len(d), "mean": mean(d), "p": p}


def fmt(t):
    return f"mean {t['mean']:+.3f}, n {t['n']}" + (f", p {t['p']:.3f}" if t["p"] is not None else "")


def main(argv):
    run = argv[0]
    rid = os.path.basename(os.path.normpath(run))
    G = load(run)
    man = json.load(open(os.path.join(run, "run_manifest.json")))
    conds = list(man["conditions"])
    by = {(g["condition"], g["fixture"], g["opener"], g["scale"]): g for g in G}
    fixtures = sorted({g["fixture"] for g in G})
    print(f"Probe {man['probe']} · {man['model']} · {len(G)} scored games · conditions {', '.join(conds)}")
    print("EXPLORATORY: not pre-registered; p-values are signals only.\n")

    print("Overview (per condition and scale): agreements, mean seller share, walk-aways, past a limit")
    for c in conds:
        row = []
        for sc, lab in ((LO, "$100"), (HI, "$100M")):
            gs = [g for g in G if g["condition"] == c and g["scale"] == sc]
            ag = [g for g in gs if g["terminal"] == "AGREEMENT"]
            row.append(f"{lab}: {len(ag)}/{len(gs)} deals, share {mean(share(g) for g in ag):.3f},"
                       f" walk {sum(g['terminal'] == 'WALKAWAY' for g in gs)},"
                       f" past-limit {sum(not g['F'] <= g['price'] <= g['C'] for g in ag)}")
        print(f"  {c:<3} " + " | ".join(row))
        OUT.setdefault("overview", {})[c] = row

    # ---- Q1-style measures, for every condition
    print("\nOpening gap from own limit, $100 -> $100M (mean %; fixtures where it shrank)")
    for c in conds:
        parts = []
        for side in ("seller", "buyer"):
            pairs = [(gap(by[(c, f, side, LO)]), gap(by[(c, f, side, HI)])) for f in fixtures
                     if (c, f, side, LO) in by and (c, f, side, HI) in by]
            pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
            t = test([b - a for a, b in pairs], f"gap|{c}|{side}")
            parts.append(f"{side} opens {mean(a for a, _ in pairs):.1f}% -> {mean(b for _, b in pairs):.1f}%"
                         f" ({sum(b < a for a, b in pairs)}/{len(pairs)} shrank, p {t['p']:.3f})"
                         if t["p"] is not None else f"{side}: n {len(pairs)}")
            OUT.setdefault("opening_gap", {}).setdefault(c, {})[side] = {
                "lo": mean(a for a, _ in pairs), "hi": mean(b for _, b in pairs),
                "shrank": sum(b < a for a, b in pairs), "n": len(pairs), "p": t["p"]}
        print(f"  {c:<3} " + " | ".join(parts))

    print("\nOpener's advantage (seller share when seller opens minus when buyer opens) and opener x scale")
    for c in conds:
        adv = []
        for sc in (LO, HI):
            a_ = [share(by[k]) for k in by if k[0] == c and k[2] == "seller" and k[3] == sc]
            b_ = [share(by[k]) for k in by if k[0] == c and k[2] == "buyer" and k[3] == sc]
            adv.append(mean(a_) - mean(b_))
        d = []
        for f in fixtures:
            try:
                s_ = [share(by[(c, f, o, HI)]) - share(by[(c, f, o, LO)]) for o in ("seller", "buyer")]
            except (KeyError, TypeError):
                continue
            d.append(s_[0] - s_[1])
        t = test(d, f"interaction|{c}")
        print(f"  {c:<3} $100 {adv[0]:+.3f} -> $100M {adv[1]:+.3f}; interaction {fmt(t)}")
        OUT.setdefault("opener_advantage", {})[c] = {"lo": adv[0], "hi": adv[1], "interaction": t}

    print("\nPrimary-style shift d ($100M share − $100 share, per fixture, mean over openers)")
    for c in conds:
        d = []
        for f in fixtures:
            parts = [share(by[(c, f, o, HI)]) - share(by[(c, f, o, LO)]) for o in ("seller", "buyer")
                     if (c, f, o, HI) in by and (c, f, o, LO) in by
                     and share(by[(c, f, o, HI)]) is not None and share(by[(c, f, o, LO)]) is not None]
            if parts:
                d.append(sum(parts, Fr(0)) / len(parts))
        t = test(d, f"d|{c}")
        print(f"  {c:<3} {fmt(t)}")
        OUT.setdefault("shift_d", {})[c] = t

    # ---- Q3: hidden information
    if "N" in conds and any(c in conds for c in ("LS", "LB")):
        print("\nQ3. Exploitation rate: share of the possible gain the informed side takes vs the same game in N")
        print("    (LS: (s_LS − s_N)/(1 − s_N); LB: (s_N − s_LB)/s_N; pairs where both games made a deal)")
        for c, informed in (("LS", "seller"), ("LB", "buyer")):
            if c not in conds:
                continue
            res = {}
            for sc, lab in ((LO, "$100"), (HI, "$100M")):
                rates = {}
                for f in fixtures:
                    for o in ("seller", "buyer"):
                        x, n = by.get((c, f, o, sc)), by.get(("N", f, o, sc))
                        if not x or not n or share(x) is None or share(n) is None:
                            continue
                        sn, sx = share(n), share(x)
                        room = (1 - sn) if informed == "seller" else sn
                        if room > 0:
                            rates[(f, o)] = ((sx - sn) if informed == "seller" else (sn - sx)) / room
                res[sc] = rates
                ag = [g for g in G if g["condition"] == c and g["scale"] == sc]
                deals = [g for g in ag if g["terminal"] == "AGREEMENT"]
                at_lim = sum(g["price"] == (g["C"] if informed == "seller" else g["F"]) for g in deals)
                print(f"  {c} {lab:<5}: mean rate {mean(rates.values()):+.3f} (median {st.median([float(r) for r in rates.values()]) if rates else float('nan'):+.3f},"
                      f" n {len(rates)}); deals exactly at the known limit {at_lim}/{len(deals)};"
                      f" informed side's share {mean(share(g) if informed == 'seller' else 1 - share(g) for g in deals):.3f}")
                OUT.setdefault("exploitation", {}).setdefault(c, {})[sc] = {
                    "mean": mean(rates.values()), "n": len(rates), "at_known_limit": at_lim, "deals": len(deals)}
            both = [res[HI][k] - res[LO][k] for k in res[LO] if k in res[HI]]
            t = test(both, f"exploit-scale|{c}")
            print(f"  {c} stakes contrast ($100M − $100 rate, paired): {fmt(t)}")
            OUT["exploitation"][c]["scale_contrast"] = t

        print("\nQ3. Does the informed side say it knows? (its messages: the exact other limit, or 'your ceiling/floor/budget/limit')")
        pat = re.compile(r"your (private )?(ceiling|floor|budget|limit|maximum|minimum)|we know|i know|we are aware|informed that", re.I)
        for c, informed in (("LS", "seller"), ("LB", "buyer")):
            if c not in conds:
                continue
            for sc, lab in ((LO, "$100"), (HI, "$100M")):
                gs = [g for g in G if g["condition"] == c and g["scale"] == sc]
                said_num = said_phrase = 0
                for g in gs:
                    other = g["C"] if informed == "seller" else g["F"]
                    roles = [g["opener"] if i % 2 == 0 else ("buyer" if g["opener"] == "seller" else "seller")
                             for i in range(len(g["moves"]))]
                    msgs = [m.get("message") or "" for m, r in zip(g["moves"], roles) if r == informed]
                    offers = {p for _, r, p in g["offers"] if r == informed}
                    said_num += any(str(other) in m.replace(",", "") and other not in offers for m in msgs)
                    said_phrase += any(pat.search(m) for m in msgs)
                print(f"  {c} {lab:<5}: games where the informed side states the other's limit {said_num}/{len(gs)};"
                      f" uses 'your ceiling/floor/...' or 'we know' {said_phrase}/{len(gs)}")
                OUT.setdefault("reveals", {}).setdefault(c, {})[sc] = {"number": said_num, "phrase": said_phrase, "n": len(gs)}

    out = os.path.join(ROOT, "derived", "explore", rid)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(OUT, fh, indent=1, default=float)
    print(f"\nwritten: {os.path.relpath(os.path.join(out, 'summary.json'), ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
