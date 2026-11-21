#!/usr/bin/env python3
"""study3_clamp_values.py: the scripted first-offer gaps of Study 3's opening clamp
(prereg/study3/PREREGISTRATION.md §3.2 and Appendix T).

    python3 scripts/study3_clamp_values.py            write prereg/study3/clamp_values.json, print Appendix T
    python3 scripts/study3_clamp_values.py --check    recompute; exit 1 if the file on disk differs

Per arm, opener and source scale ($100 or $100M): the mean positions of the seller's and the buyer's first
offers in zone units, x = (P - F)/(C - F) (floor 0, ceiling 1), over that arm's earlier games (no-appraisal
prompt) in which both first offers exist (messages 1 and 2 are OFFERs), whatever the game's outcome.
Zone units, not own-limit gaps: own-limit gaps transplanted onto narrow zones made the buyer's scripted
counter exceed the seller's ask in 42 clamp games (checked 2026-10-13); zone positions keep every pair
ordered, and they are the units of the split and of the midpoint account. Sources (frozen, earlier
data; no Study 3 game):
  Qwen3.8 27B, Gemma 4 31B: their Study 2 runs (one game per cell);
  the three API arms: Addendum C's registered run (four games per cell).
Values are stored as decimal strings rounded to 4 places; the runner prices from those strings exactly.

Also checks every clamp game of Study 3 (G01-G40 and the pilot's Q01-Q05) for usable prices: both >= 1
and the buyer's price below the seller's. Standard library only. Reads files only.
"""
import argparse
import json
import os
import statistics
import sys
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import run_records as rr            # noqa: E402
import study3 as S                  # noqa: E402

SOURCES = {
    "qwen3.8-27b-fp8": ("data/runs/s2-qwen3.8-27b-fp8-20261003T223042Z", None),
    "gemma-4-31b-it": ("data/runs/s2-gemma-4-31b-it-20261004T012932Z", None),
    "gpt-6-luna": ("data/runs/addC-20261011T182548Z", "gpt-6-luna"),
    "claude-haiku-5.5": ("data/runs/addC-20261011T182548Z", "claude-haiku-5.5"),
    "gemini-3.5-flash-lite": ("data/runs/addC-20261011T182548Z", "gemini-3.5-flash-lite"),
}


def gap(role, price, game):
    return Fraction(price, game["F"]) - 1 if role == "seller" else 1 - Fraction(price, game["C"])


def scored_games(run_dir, arm_filter):
    out = {}
    with open(os.path.join(ROOT, run_dir, "games.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["terminal"] == "INFRASTRUCTURE_VOID":
                continue
            if arm_filter is not None and r.get("arm") != arm_filter:
                continue
            out[r["game_id"]] = r
    return list(out.values())


def compute():
    values, detail, srcs = {}, {}, {}
    for arm, (run_dir, filt) in SOURCES.items():
        path = os.path.join(ROOT, run_dir, "games.jsonl")
        with open(path, "rb") as fh:
            srcs[arm] = {"run": run_dir, "games_sha256": rr.sha256_hex(fh.read())}
        games = scored_games(run_dir, filt)
        values[arm], detail[arm] = {}, {}
        for opener in S.OPENERS:
            values[arm][opener], detail[arm][opener] = {}, {}
            for sc in (S.LOW, S.HIGH):
                xs, xb, gs, gb = [], [], [], []
                for r in games:
                    if r["opener"] != opener or r["scale"] != sc or len(r["offers"]) < 2:
                        continue
                    (_, w1, p1), (_, w2, p2) = r["offers"][0][:3], r["offers"][1][:3]
                    if w1 != opener or w2 != S.OTHER[opener]:
                        continue
                    first = {w1: p1, w2: p2}
                    w = r["C"] - r["F"]
                    xs.append(Fraction(first["seller"] - r["F"], w))
                    xb.append(Fraction(first["buyer"] - r["F"], w))
                    gs.append(gap("seller", first["seller"], r))
                    gb.append(gap("buyer", first["buyer"], r))
                if not xs:
                    raise SystemExit(f"{arm} {opener} {sc}: no game with both first offers")
                mx = lambda v: sum(v, Fraction(0)) / len(v)      # noqa: E731
                values[arm][opener][sc] = {"x_seller": f"{float(mx(xs)):.4f}", "x_buyer": f"{float(mx(xb)):.4f}"}
                detail[arm][opener][sc] = {"n_games": len(xs),
                                           "median_x_seller": round(float(statistics.median(xs)), 4),
                                           "median_x_buyer": round(float(statistics.median(xb)), 4),
                                           "mean_gap_seller_own_limit": round(float(mx(gs)), 4),
                                           "mean_gap_buyer_own_limit": round(float(mx(gb)), 4)}
    return {"schema": "study3-clamp-values/1",
            "rule": "mean first-offer positions in zone units, x = (P - F)/(C - F), of the seller and the buyer, over "
                    "each arm's earlier no-appraisal games containing both first offers, per opener and scale; "
                    "4 decimals. Scripted price = F + x (C - F), rounded half up to a multiple of V/100",
            "sources": srcs, "values": values, "detail_not_used_for_pricing": detail}


def feasibility(doc):
    """Every clamp game of the pilot and main fixtures: prices usable; also the scripted midpoints."""
    problems, mids = [], {}
    for pilot in (True, False):
        fxs = S.fixtures(S.PILOT_PHRASE, S.N_PILOT, S.PILOT_PREFIX) if pilot else S.fixtures()
        for arm in S.ARMS:
            for fx in fxs:
                for cell in S.CLAMP_CELLS:
                    for o in S.OPENERS:
                        g = S.make_game(arm, fx, cell, o, 1)
                        try:
                            p1, p2 = S.clamp_prices(g, doc)
                        except ValueError as e:
                            problems.append(str(e))
                            continue
                        if not pilot:
                            ps, pb = (p1, p2) if o == "seller" else (p2, p1)
                            m = (Fraction(ps + pb, 2) - g["F"]) / (g["C"] - g["F"])
                            mids.setdefault((arm, cell), []).append(float(m))
    return problems, {f"{a}|{c}": round(statistics.fmean(v), 4) for (a, c), v in mids.items()}


def table(doc):
    lines = ["| Model | Opener | Source scale | Seller's position x_S | Buyer's position x_B | Scripted midpoint | n games |",
             "|---|---|---|---|---|---|---|"]
    for arm, by in doc["values"].items():
        for o, sc in by.items():
            for s, v in sc.items():
                n = doc["detail_not_used_for_pricing"][arm][o][s]["n_games"]
                mid = (float(v["x_seller"]) + float(v["x_buyer"])) / 2
                lines.append(f"| {arm} | {o} | {'$100' if s == S.LOW else '$100M'} | {v['x_seller']} | {v['x_buyer']}"
                             f" | {mid:.4f} | {n} |")
    return "\n".join(lines)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", default=S.CLAMP_VALUES)
    a = ap.parse_args(argv)
    doc = compute()
    problems, mids = feasibility(doc)
    text = json.dumps(doc, indent=1) + "\n"
    if a.check:
        with open(a.out, encoding="utf-8") as fh:
            same = fh.read() == text
        print("clamp values match" if same else f"{a.out} differs from a fresh computation")
        return 0 if same and not problems else 1
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(table(doc))
    print("\nMean scripted opening midpoint in zone units (G01-G40, both openers):")
    for k, v in sorted(mids.items()):
        print(f"  {k}: {v}")
    if problems:
        print(f"\n{len(problems)} UNUSABLE clamp game(s); first: {problems[0]}")
        return 1
    print(f"\nAll clamp games usable. Written: {os.path.relpath(a.out, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
