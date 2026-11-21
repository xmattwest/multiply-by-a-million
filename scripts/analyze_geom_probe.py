#!/usr/bin/env python3
"""analyze_geom_probe.py: geometry x notation probe (STORY §12, Q13). Exploratory, not pre-registered.

Usage: python3 scripts/analyze_geom_probe.py data/explore/X-geom-<model>-<stamp>

Per geometry (round / narrow / wide) and notation (raw / biz), per fixture, x1e6 minus x1:
  SPN d          seller share, self-play (mean over the two openers)
  BOT seller     seller share with the model as seller vs the scale-proof bot
  BOT buyer      seller share with the model as buyer (a rise = the model-buyer pays more)
  gaps           opener first-offer gap from its own limit (percentage points)
Then the two questions the probe was built for:
  notation       biz minus raw, per fixture (paired: same fixtures, same seeds)
  geometry       round vs narrow vs wide (different fixtures: difference of means, bootstrap)
Writes derived/explore/<run_id>/summary.json and figures GX1 (shifts), GX2 (zone positions).
Standard library only.
"""
import collections
import json
import os
import random
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import analyze as A                                             # noqa: E402
import analyze_game_probe as G                                  # noqa: E402
from make_figures import BLUE, GREEN, GREY, INK, ORANGE, Svg    # noqa: E402

GEOMS = ("round", "narrow", "wide")
NOTS = ("raw", "biz")
GNAME = {"round": "Study 2 zone (narrow, around $100)", "narrow": "narrow zone, off-centre",
         "wide": "wide zone, off-centre"}
MEASURES = {
    "SPN d": (G.share, lambda r: r["cell"] == "SPN"),
    "BOT model=seller share": (G.share, lambda r: r["cell"] == "BOT" and r["model_role"] == "seller"),
    "BOT model=buyer share": (G.share, lambda r: r["cell"] == "BOT" and r["model_role"] == "buyer"),
    "SPN seller-opener gap": (lambda r: G.gap(*G.first_offer(r), r) if r["opener"] == "seller" else None,
                              lambda r: r["cell"] == "SPN"),
    "SPN buyer-opener gap": (lambda r: G.gap(*G.first_offer(r), r) if r["opener"] == "buyer" else None,
                             lambda r: r["cell"] == "SPN"),
}


def boot_diff(a, b, what, model):
    """Difference of means a - b over two independent fixture sets, 95% bootstrap."""
    rng = random.Random(A.seed(f"geom-diff-{what}", model))
    ds = sorted(st.mean(rng.choices(a, k=len(a))) - st.mean(rng.choices(b, k=len(b))) for _ in range(10000))
    return {"mean": st.mean(a) - st.mean(b), "ci95": [ds[250], ds[9749]], "n": [len(a), len(b)]}


def analyse(recs, model):
    out, per = {}, {}
    for g in GEOMS:
        for nt in NOTS:
            rs = [r for r in recs if r.get("geom") == g and r.get("notation") == nt]
            if not rs:
                continue
            for name, (val, key) in MEASURES.items():
                pf = G.per_fixture(rs, val, key)
                diffs = {f: b - s for f, (s, b) in pf.items()}
                per[(g, nt, name)] = diffs
                res = G.test(list(diffs.values()), f"{g}-{nt}-{name}", model)
                if pf:
                    res["level_small"] = st.mean(s for s, _ in pf.values())
                    res["level_big"] = st.mean(b for _, b in pf.values())
                out[f"{g} {nt} {name}"] = res
    # notation: biz - raw, paired by fixture
    for g in GEOMS:
        for name in MEASURES:
            a, b = per.get((g, "biz", name), {}), per.get((g, "raw", name), {})
            common = [f for f in a if f in b]
            if len(common) >= 2:
                out[f"{g} biz-raw {name}"] = G.test([a[f] - b[f] for f in common], f"{g}-notation-{name}", model)
    # geometry: unpaired differences of the scale shift, per notation
    for nt in NOTS:
        for name in ("SPN d", "BOT model=seller share"):
            for g1, g2 in (("round", "narrow"), ("narrow", "wide"), ("round", "wide")):
                a, b = list(per.get((g1, nt, name), {}).values()), list(per.get((g2, nt, name), {}).values())
                if len(a) >= 2 and len(b) >= 2:
                    out[f"{nt} {g1}-{g2} {name}"] = boot_diff(a, b, f"{nt}-{g1}-{g2}-{name}", model)
    return out


def zone_positions(recs):
    z = {}
    for g in GEOMS:
        for nt in NOTS:
            for role in ("seller", "buyer"):
                for sm in (True, False):
                    v = [(G.first_offer(r)[1] - r["F"]) / (r["C"] - r["F"]) for r in recs
                         if r["cell"] == "SPN" and r.get("geom") == g and r.get("notation") == nt
                         and r["opener"] == role and G.small(r["scale"]) == sm and r["offers"]]
                    if v:
                        z[f"{g} {nt} {role} {'small' if sm else 'big'}"] = st.mean(v)
    return z


def fig_shifts(C, model, path):
    panels = [("SPN d", "self-play: change in seller share"),
              ("BOT model=seller share", "model as seller vs the bot"),
              ("BOT model=buyer share", "model as buyer vs the bot")]
    rows = [(g, nt) for g in GEOMS for nt in NOTS if f"{g} {nt} SPN d" in C]
    # The value labels get their own column right of each panel's +0.3 gridline.
    W, rowh, T, L, PW, gapw, lo, hi = 1270, 34, 120, 350, 250, 60, -0.6, 0.3
    H = T + rowh * len(rows) + 100
    s = Svg(W, H, f"Geometry x notation: change in seller share from x1 to x10^6 · {model}",
            "Three panels (self-play; model as seller vs the bot; model as buyer vs the bot). One row per zone"
            " geometry and price notation: the mean per-fixture change in seller share when every amount gains six"
            " zeros, with a 95% bootstrap interval. Exploratory, not pre-registered.")
    s.text(30, 34, "Seller share at x1 vs x10^6, by zone geometry and price notation", 19, weight="bold")
    s.text(30, 56, f"Change in seller share, x10^6 minus x1, per fixture (95% interval) · {model} · exploratory",
           12.5, color="#444")
    for k, (name, title) in enumerate(panels):
        x0 = L + k * (PW + gapw)
        X = lambda v: x0 + (v - lo) / (hi - lo) * PW                              # noqa: E731
        s.text(x0, T - 34, title, 12.5, weight="bold")
        s.add(f"<rect x='{X(-0.1):.1f}' y='{T - 18}' width='{X(0.1) - X(-0.1):.1f}' height='{rowh * len(rows) + 8}'"
              f" fill='{GREEN}' fill-opacity='0.08'/>")
        for t in (-0.6, -0.3, 0, 0.3):
            s.line(X(t), T - 18, X(t), T + rowh * len(rows) - 10, INK if t == 0 else "#e6e6e6")
            s.text(X(t), T + rowh * len(rows) + 6, f"{t:+.1f}" if t else "0", 11, "middle", "#444")
        for i, (g, nt) in enumerate(rows):
            y = T + rowh * i
            if k == 0:
                s.text(30, y + 4, f"{GNAME[g]} · {'$1,234 (biz)' if nt == 'biz' else '1234 dollars (raw)'}", 12)
            c = C.get(f"{g} {nt} {name}", {})
            if "mean" not in c:
                continue
            col = BLUE if nt == "biz" else GREY
            a, b = c["ci95"]
            s.line(X(max(lo, a)), y, X(min(hi, b)), y, col, 3)
            s.add(f"<circle cx='{X(max(lo, min(hi, c['mean']))):.1f}' cy='{y:.1f}' r='5.5' fill='{col}' stroke='white'/>")
            s.text(X(hi) + 8, y + 4, f"{c['mean']:+.2f}", 10, "start", INK)
    s.text(30, H - 40, "Shaded band: ±0.10. Grey = raw digits (Study 2 style); blue = business notation. The bot plays"
           " identically at both scales. Buyer panel: a rise means the model-buyer pays more.", 11.5, color="#444")
    s.text(30, H - 22, "Source: data/explore/<run>/games.jsonl via scripts/analyze_geom_probe.py.", 11.5, color="#444")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    s.save(path)


def fig_zones(Z, model, path):
    rows = [(g, nt) for g in GEOMS for nt in NOTS if f"{g} {nt} seller small" in Z]
    W, T, L, PW, lo, hi, rowh = 1100, 120, 330, 600, -1.2, 1.6, 64
    H = T + rowh * len(rows) + 80
    s = Svg(W, H, f"Geometry x notation: openers' first offers as a position in the deal zone · {model}",
            "One row per zone geometry and notation. Axis: the deal zone, 0 = seller's floor, 1 = buyer's ceiling. Blue:"
            " seller openers; orange: buyer openers; hollow = x1, filled = x10^6; diamonds = the midpoint of the two"
            " openings (where deals land). Self-play. Exploratory.")
    s.text(30, 34, "Same modest openers; the zone decides whether the midpoint moves", 19, weight="bold")
    s.text(30, 56, f"Opener's first offer in units of the deal zone · self-play · {model} · exploratory", 12.5,
           color="#444")
    X = lambda v: L + (v - lo) / (hi - lo) * PW                                   # noqa: E731
    for t in (-1, -0.5, 0, 0.5, 1, 1.5):
        s.line(X(t), T - 20, X(t), T + rowh * len(rows) - 20, INK if t in (0, 1) else "#e6e6e6")
        s.text(X(t), T + rowh * len(rows) - 4, {0: "floor", 1: "ceiling"}.get(t, f"{t:g}"), 11, "middle", "#444")
    for i, (g, nt) in enumerate(rows):
        y0 = T + rowh * i
        s.text(30, y0 + 8, f"{GNAME[g]} · {nt}", 12)
        mids = []
        for j, sm in enumerate((True, False)):
            y = y0 + j * 18
            sp, bp = Z[f"{g} {nt} seller {'small' if sm else 'big'}"], Z[f"{g} {nt} buyer {'small' if sm else 'big'}"]
            s.line(X(min(sp, bp)), y, X(max(sp, bp)), y, "#cccccc", 2)
            for v_, col in ((sp, BLUE), (bp, ORANGE)):
                s.add(f"<circle cx='{X(v_):.1f}' cy='{y:.1f}' r='5' fill='{'white' if sm else col}' stroke='{col}'"
                      " stroke-width='2'/>")
            m = (sp + bp) / 2
            mids.append(m)
            s.add(f"<rect x='{X(m) - 4:.1f}' y='{y - 4:.1f}' width='8' height='8' fill='{INK}'"
                  f" transform='rotate(45 {X(m):.1f} {y:.1f})'/>")
        s.text(L + PW + 12, y0 + 12, f"midpoint {mids[0]:+.2f} → {mids[1]:+.2f}", 11, color=INK)
    s.text(30, H - 40, "Blue = seller opens, orange = buyer opens; hollow = x1, filled = x10^6; diamond = midpoint of the"
           " two openings. 0 = seller's floor, 1 = buyer's ceiling.", 11.5, color="#444")
    s.text(30, H - 22, "Source: data/explore/<run>/games.jsonl via scripts/analyze_geom_probe.py.", 11.5, color="#444")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    s.save(path)


def main(argv):
    if len(argv) != 1 or not os.path.isdir(argv[0]):
        print(__doc__.strip().splitlines()[2].strip())
        return 2
    run_dir = argv[0].rstrip("/")
    man = json.load(open(os.path.join(run_dir, "run_manifest.json")))
    model, run_id = man["model"], man["run_id"]
    recs = G.load(run_dir)
    slips = [r for r in recs if G.slipped(r)]
    C = analyse(recs, model)
    Z = zone_positions(recs)
    quality = {}
    for g in GEOMS:
        for nt in NOTS:
            rs = [r for r in recs if r.get("geom") == g and r.get("notation") == nt]
            if rs:
                quality[f"{g} {nt}"] = {
                    "games": len(rs), "terminals": dict(collections.Counter(r["terminal"] for r in rs)),
                    "invalid": dict(collections.Counter(x.split(":")[2] for r in rs for x in r["diagnostics"]
                                                        if x.startswith("INVALID:"))),
                    "slipped": sum(G.slipped(r) for r in rs), "violations": sum(bool(r.get("violation")) for r in rs)}
    summary = {"run_id": run_id, "model": model, "exploratory": True, "games": len(recs), "slipped_games": len(slips),
               "quality": quality, "contrasts": C, "zone_positions": Z}
    out = os.path.join(ROOT, "derived", "explore", run_id)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    fig_shifts(C, model, os.path.join(out, "figures", "GX1_shift_by_geometry_notation.svg"))
    fig_zones(Z, model, os.path.join(out, "figures", "GX2_zone_positions.svg"))

    print(f"Geometry x notation probe · {model} · {len(recs)} games · slipped {len(slips)} · EXPLORATORY")
    print("\nQuality: games, terminals, invalid replies, slipped, violations")
    for k, q in quality.items():
        print(f"  {k:12} {q['games']:4}  {q['terminals']}  invalid {q['invalid'] or 0}  slipped {q['slipped']}"
              f"  violations {q['violations']}")
    print("\nScale contrasts (x1e6 minus x1), per fixture: mean [95%]  p  below/above 0   small → big")
    for k, c in C.items():
        if "p_two_sided" in c:
            lv = f"   {c['level_small']:.3f} → {c['level_big']:.3f}" if "level_small" in c else ""
            print(f"  {k:44} {c['mean']:+8.3f}  [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]  p {c['p_two_sided']:.4f}"
                  f"  {c['fixtures_below_0']}/{c['fixtures_above_0']} of {c['n']}{lv}")
    print("\nGeometry differences in the scale shift (unpaired, 95% bootstrap)")
    for k, c in C.items():
        if "p_two_sided" not in c and "mean" in c:
            print(f"  {k:44} {c['mean']:+8.3f}  [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]  n {c['n']}")
    print("\nMidpoint of the two openings (zone units), x1 → x10^6, self-play")
    for g in GEOMS:
        for nt in NOTS:
            k = f"{g} {nt}"
            if f"{k} seller small" in Z:
                m = [(Z[f"{k} seller {sc}"] + Z[f"{k} buyer {sc}"]) / 2 for sc in ("small", "big")]
                print(f"  {k:12} {m[0]:+.2f} → {m[1]:+.2f}  (Δ {m[1] - m[0]:+.2f})")
    print(f"\nWrote {os.path.relpath(out, ROOT)}/summary.json and figures/GX1, GX2")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
