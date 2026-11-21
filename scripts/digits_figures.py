#!/usr/bin/env python3
"""Combined figures for every closed digits x stakes probe run of one model (exploratory).

Usage: python3 scripts/digits_figures.py [--model gemma-4-31b-it]

Reads data/explore/X-digits-<model>-*/games.jsonl (real runs only, not previews) and writes
derived/explore/digits_combined/<model>/:
  combined.json                         per-condition means with 95% fixture-bootstrap intervals,
                                        round-ten shares, opening-vs-floor fit
  figures/DX1_all_wordings.svg          every wording, grouped by stakes, with intervals
  figures/DX2_round_tens.svg            share of openings that are round tens, by wording
  figures/DX3_opening_vs_floor.svg      seller openings against the floor at "81 dollars" vs "81000000 dollars"

Conditions that appear in more than one run (the D2/D8 anchors) are pooled: each fixture's cell
mean is taken over all its samples from all runs. The anchors replicate across sessions within
0.3 points (STORY §7, 2026-10-03), but sessions were not randomised against each other.
Exploratory, not pre-registered.
"""
import argparse
import collections
import glob
import json
import os
import random
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import analyze as A                                             # noqa: E402
import analyze_digits_probe as P                                # noqa: E402
from make_figures import BLUE, GREY, INK, ORANGE, Svg           # noqa: E402

# Rows grouped by what the deal is worth; within a group, the order of the text.
GROUPS = [("$81 at stake", ["D2", "S2", "X2", "K2"]),
          ("$81 thousand at stake", ["D5", "U5"]),
          ("$81 million at stake", ["D8", "SN", "C8", "S8", "W8", "SM", "U8"])]


def per100(r):
    """The offer re-expressed on the $100 scale (the same units as base_limit)."""
    return r["price"] * r["base_limit"] / r["limit_units"]


def round_ten(r):
    """True when the offer, on the $100 scale, is a whole multiple of 10 (e.g. 120 or 120,000,000)."""
    v = per100(r) / 10
    return abs(v - round(v)) < 1e-9


def load_runs(model):
    runs, recs = [], []
    for m in sorted(glob.glob(os.path.join(ROOT, "data", "explore", f"X-digits-{model}-*", "run_manifest.json"))):
        man = json.load(open(m))
        d = os.path.dirname(m)
        if man.get("preview") or not os.path.exists(os.path.join(d, "run_summary.json")):
            continue
        runs.append(man["run_id"])
        recs += [dict(r, run_id=man["run_id"]) for r in P.load(d)]
    return runs, recs


def summarise(recs, model):
    cm = P.cell_means(recs)                       # pools samples across runs for the same cell
    conds = [c for c in P.ORDER if any(r["condition"] == c for r in recs)]
    out = {}
    for c in conds:
        out[c] = {"label": P.LABEL[c], "runs": sorted({r["run_id"] for r in recs if r["condition"] == c})}
        for role in ("seller", "buyer"):
            fx = [v for (f, ro, cc), v in sorted(cm.items()) if ro == role and cc == c]
            offers = [r for r in recs if r["condition"] == c and r["role"] == role and r["gap"] is not None]
            ci = A.bootstrap_ci(fx, random.Random(A.seed(f"digits-combined-{c}-{role}", model)),
                                levels=(0.95,))["95"]
            modal = collections.Counter(round(per100(r), 6) for r in offers).most_common(1)[0]
            floors, opens = [r["base_limit"] for r in offers], [per100(r) for r in offers]
            out[c][role] = {"n_fixtures": len(fx), "n_offers": len(offers), "mean_gap_pct": st.mean(fx),
                            "ci95": ci, "round_ten_share": sum(map(round_ten, offers)) / len(offers),
                            "modal_offer_per100": modal[0], "modal_share": modal[1] / len(offers),
                            "slope_on_limit": st.covariance(floors, opens) / st.variance(floors),
                            "r_with_limit": st.correlation(floors, opens)}
    return out


def rows_layout():
    rows = []                                      # (kind, text or condition)
    for name, cs in GROUPS:
        rows.append(("head", name))
        rows += [("cond", c) for c in cs]
    return rows


def fig_wordings(summ, model, path):
    rows = [r for r in rows_layout() if r[0] == "head" or r[1] in summ]
    W, rowh, T, L, PW, lo, hi = 1100, 28, 112, 330, 300, 0, 55
    H = T + rowh * len(rows) + 96
    s = Svg(W, H, f"Opener's first-offer gap for every wording tested, grouped by stakes · {model}",
            "Two panels (seller opens, buyer opens). One row per wording of the opener's own limit, grouped by"
            " what the deal is worth ($81, $81 thousand, $81 million). Dot: mean over fresh fixtures of the gap"
            " between the first offer and the opener's own limit, in percent of the limit; bar: 95% bootstrap"
            " interval over fixtures. Lower = more modest. Exploratory, not pre-registered.")
    s.text(30, 34, "Small dollar amounts get bold openings, with or without \"$\"; big amounts get modest ones", 19,
           weight="bold")
    s.text(30, 56, f"Mean first-offer gap from own limit (%), 95% interval over 30 fresh fixtures · {model}"
           " · two sessions pooled · exploratory", 12.5, color="#444")
    for role, color, x0 in (("seller", BLUE, L), ("buyer", ORANGE, L + PW + 90)):
        X = lambda v: x0 + (v - lo) / (hi - lo) * PW                              # noqa: E731
        s.text(x0, T - 24, "seller opens: markup over floor" if role == "seller"
               else "buyer opens: discount under ceiling", 13, weight="bold")
        for t in range(lo, hi + 1, 10):
            s.line(X(t), T - 12, X(t), T + rowh * len(rows), "#e6e6e6")
            s.text(X(t), T + rowh * len(rows) + 16, f"{t}%", 11, "middle", "#444")
        ref = summ.get("D2", {}).get(role, {}).get("mean_gap_pct")
        if ref is not None:
            s.line(X(ref), T - 12, X(ref), T + rowh * len(rows), GREY, 1.2, "5 4")
        for i, (kind, v) in enumerate(rows):
            y = T + rowh * i + rowh / 2
            if kind == "head":
                if role == "seller":
                    s.text(30, y + 5, v, 12.5, weight="bold", color="#555")
                    s.line(30, y + 10, L + 2 * PW + 90, y + 10, "#cccccc")
                continue
            d = summ[v][role]
            if role == "seller":
                s.text(42, y + 4, f"{v}  {summ[v]['label']}", 12)
            a, b = d["ci95"]
            s.line(X(a), y, X(b), y, color, 3)
            s.add(f"<circle cx='{X(d['mean_gap_pct']):.1f}' cy='{y:.1f}' r='5.5' fill='{color}' stroke='white'/>")
            s.text(X(b) + 7, y + 4, f"{d['mean_gap_pct']:.1f}", 11, color=INK)
    s.text(30, H - 52, "Dashed line: \"81 dollars\" (D2). Replies are in plain whole dollars except U5/U8 (in"
           " thousands/millions) and K2 (in cents).", 11.5, color="#444")
    s.text(30, H - 34, "D2 and D8 ran in both sessions and are pooled (they agree within 0.3 points)."
           " Source: data/explore/X-digits-*/games.jsonl via scripts/digits_figures.py.", 11.5, color="#444")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    s.save(path)


def fig_round(summ, model, path):
    rows = [r for r in rows_layout() if r[0] == "head" or r[1] in summ]
    W, rowh, T, L, PW = 1000, 28, 112, 330, 260
    H = T + rowh * len(rows) + 110
    s = Svg(W, H, f"Share of first offers that are round tens, by wording · {model}",
            "Two panels (seller opens, buyer opens). One row per wording, grouped by stakes. Bar: share of first"
            " offers that are a whole multiple of 10 once put on the $100 scale (120, 120 thousand, 120 million"
            " all count). The text gives the single most common offer and its share. Exploratory.")
    s.text(30, 34, "At small dollar amounts the seller reaches for a round number", 19, weight="bold")
    s.text(30, 56, f"Share of first offers that are round tens on the $100 scale · {model} · exploratory",
           12.5, color="#444")
    for role, color, x0 in (("seller", BLUE, L), ("buyer", ORANGE, L + PW + 100)):
        X = lambda v: x0 + v * PW                                                 # noqa: E731
        s.text(x0, T - 24, f"{role} opens", 13, weight="bold")
        for t in range(0, 101, 25):
            s.line(X(t / 100), T - 12, X(t / 100), T + rowh * len(rows), "#e6e6e6")
            s.text(X(t / 100), T + rowh * len(rows) + 16, f"{t}%", 11, "middle", "#444")
        for i, (kind, v) in enumerate(rows):
            y = T + rowh * i + rowh / 2
            if kind == "head":
                if role == "seller":
                    s.text(30, y + 5, v, 12.5, weight="bold", color="#555")
                    s.line(30, y + 10, L + 2 * PW + 100, y + 10, "#cccccc")
                continue
            d = summ[v][role]
            if role == "seller":
                s.text(42, y + 4, f"{v}  {summ[v]['label']}", 12)
            s.add(f"<rect x='{X(0):.1f}' y='{y - 8:.1f}' width='{X(d['round_ten_share']) - X(0):.1f}'"
                  f" height='16' fill='{color}' fill-opacity='0.8'/>")
            s.text(X(d["round_ten_share"]) + 6, y + 4, f"{100 * d['round_ten_share']:.0f}%", 11, color=INK)
    s.text(30, H - 64, "Buyers' openings are round tens in almost every wording (95–100%), so the habit"
           " separates the wordings only for the seller.", 11.5, color="#444")
    s.text(30, H - 46, "Seller's most common opening: \"81 dollars\" → "
           f"{summ['D2']['seller']['modal_offer_per100']:g} ({100 * summ['D2']['seller']['modal_share']:.0f}%"
           " of openings, floors 75–95)" if "D2" in summ else "", 11.5, color="#444")
    s.text(30, H - 28, "Source: data/explore/X-digits-*/games.jsonl via scripts/digits_figures.py.", 11.5,
           color="#444")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    s.save(path)


def fig_scatter(recs, summ, model, path):
    panels = [c for c in ("D2", "D8") if c in summ]
    W, H, T, PW, PH = 1000, 560, 110, 380, 340
    lo_x, hi_x, lo_y, hi_y = 70, 100, 70, 160
    s = Svg(W, H, f"Seller's first offer against its floor: \"81 dollars\" vs \"81000000 dollars\" · {model}",
            "Two panels, one per wording. Each dot is one seller first offer (on the $100 scale) against the"
            " seller's floor; dots are jittered slightly so stacks are visible. Solid line: the floor itself;"
            " dashed line: a least-squares fit. At \"81 dollars\" offers stack on round prices (most on 120)"
            " whatever the floor; at \"81000000 dollars\" they sit closer to the floor and step up with it (stacks at 95 and"
            " 110 million). Exploratory.")
    s.text(30, 34, "At \"81 dollars\" most openings are 120 whatever the floor; at $81M they sit lower and track it", 19, weight="bold")
    s.text(30, 56, f"Seller's first offer vs its own floor, $100 scale, both sessions · {model} · exploratory",
           12.5, color="#444")
    for k, c in enumerate(panels):
        x0 = 90 + k * (PW + 110)
        X = lambda v: x0 + (v - lo_x) / (hi_x - lo_x) * PW                      # noqa: E731
        Y = lambda v: T + PH - (v - lo_y) / (hi_y - lo_y) * PH                  # noqa: E731
        s.text(x0, T - 18, f"{c}  {summ[c]['label']}", 13, weight="bold")
        for t in range(lo_y, hi_y + 1, 10):
            s.line(x0, Y(t), x0 + PW, Y(t), "#eeeeee")
            s.text(x0 - 6, Y(t) + 4, f"{t}", 10.5, "end", "#444")
        for t in range(lo_x, hi_x + 1, 5):
            s.text(X(t), T + PH + 16, f"{t}", 10.5, "middle", "#444")
        s.text(x0 + PW / 2, T + PH + 36, "seller's floor", 11.5, "middle", "#444")
        if k == 0:
            s.add(f"<text x='{x0 - 44:.1f}' y='{T + PH / 2:.1f}' font-size='11.5' fill='#444' text-anchor='middle'"
                  f" transform='rotate(-90 {x0 - 44:.1f} {T + PH / 2:.1f})'>seller's first offer</text>")
        s.line(X(lo_x), Y(lo_x), X(hi_x), Y(hi_x), INK, 1.2)
        rng = random.Random(A.seed("digits-scatter-jitter", c))
        pts = [(r["base_limit"], per100(r)) for r in recs
               if r["condition"] == c and r["role"] == "seller" and r["gap"] is not None]
        for fl, op in pts:
            if lo_y <= op <= hi_y:
                s.add(f"<circle cx='{X(fl) + rng.uniform(-3, 3):.1f}' cy='{Y(op) + rng.uniform(-2.5, 2.5):.1f}'"
                      f" r='2.6' fill='{BLUE}' fill-opacity='0.35'/>")
        d = summ[c]["seller"]
        mx, my = st.mean(p[0] for p in pts), st.mean(p[1] for p in pts)
        f = lambda v: my + d["slope_on_limit"] * (v - mx)                         # noqa: E731
        s.line(X(lo_x), Y(f(lo_x)), X(hi_x), Y(f(hi_x)), ORANGE, 2, "6 4")
        s.text(x0 + 8, T + 18, f"r with floor {d['r_with_limit']:.2f} · slope {d['slope_on_limit']:.2f}"
               f" · round tens {100 * d['round_ten_share']:.0f}%", 11.5, color=INK)
        off = sum(not (lo_y <= p[1] <= hi_y) for p in pts)
        if off:
            s.text(x0 + 8, T + 32, f"{off} offers outside the plotted range", 10.5, color="#666")
    s.text(30, H - 28, "Solid line: offer = floor. Dashed: least-squares fit. At $81M the stacks are 95 million (low floors) and 110 million (high floors). Source:"
           " data/explore/X-digits-*/games.jsonl via scripts/digits_figures.py.", 11.5, color="#444")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    s.save(path)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--model", default="gemma-4-31b-it")
    a = ap.parse_args(argv)
    runs, recs = load_runs(a.model)
    if not recs:
        print(f"no closed digits runs for {a.model}")
        return 2
    summ = summarise(recs, a.model)
    out = os.path.join(ROOT, "derived", "explore", "digits_combined", a.model)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "combined.json"), "w") as fh:
        json.dump({"model": a.model, "runs": runs, "exploratory": True, "records": len(recs),
                   "by_condition": summ}, fh, indent=1)
    fig_wordings(summ, a.model, os.path.join(out, "figures", "DX1_all_wordings.svg"))
    fig_round(summ, a.model, os.path.join(out, "figures", "DX2_round_tens.svg"))
    fig_scatter(recs, summ, a.model, os.path.join(out, "figures", "DX3_opening_vs_floor.svg"))
    print(f"Digits probe, combined · {a.model} · {len(runs)} runs · {len(recs)} first moves · EXPLORATORY")
    print("  cond  seller gap [95%]        round10  r_floor | buyer gap   round10")
    for c, d in summ.items():
        se, bu = d["seller"], d["buyer"]
        print(f"  {c:4} {se['mean_gap_pct']:5.1f} [{se['ci95'][0]:4.1f}, {se['ci95'][1]:4.1f}]"
              f"   {100 * se['round_ten_share']:4.0f}%   {se['r_with_limit']:.2f}  | {bu['mean_gap_pct']:5.1f}"
              f"   {100 * bu['round_ten_share']:4.0f}%")
    print(f"Wrote {os.path.relpath(out, ROOT)}/combined.json and figures/DX1–DX3")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
