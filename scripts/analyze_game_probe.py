#!/usr/bin/env python3
"""analyze_game_probe.py: analyse one new-game probe run (exploratory, not pre-registered).

Usage: python3 scripts/analyze_game_probe.py data/explore/X-newgame-<model>-<stamp>

Writes derived/explore/<run_id>/summary.json and figures/G1_scale_shift_by_cell.svg,
figures/G2_share_levels_by_cell.svg. All contrasts are per fixture, x1e6 (or $100M) minus x1 (or $100),
over the games present at both scales; p is the two-sided sign-flip Wilcoxon (50,000 flips) and the
interval a 95% bootstrap over fixtures (scripts/analyze.py).

  SPN d        seller share, self-play, no reference (mean over the two openers)
  SPR d        the same with the public appraisal; SPR-SPN = does the appraisal protect?
  BOT seller   seller share when the MODEL is the seller and the bot the buyer (mean over who opens)
  BOT buyer    seller share when the model is the buyer (a rise = the model-buyer pays more)
  LEG d        the exact Study 2 game on fresh fixtures (bridge to Study 2's d = -0.256)
  gaps         the opener's first-offer gap from its own limit, in percentage points (HA's g)
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
import analyze as A                                        # noqa: E402
from make_figures import BLUE, GREEN, GREY, INK, ORANGE, Svg   # noqa: E402

FLIPS = 50_000
BIG = {"x1": "x1e6", "x1e2": "x1e8"}                       # small scale -> its big partner


def load(run_dir):
    """Final attempt per game; voided games dropped."""
    recs = {}
    for line in open(os.path.join(run_dir, "games.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        if r.get("schema") not in ("newgame/1", "game/1"):
            continue
        prev = recs.get(r["game_id"])
        if prev is None or r.get("attempt", 1) > prev.get("attempt", 1):
            recs[r["game_id"]] = r
    return [r for r in recs.values() if r["terminal"] != "INFRASTRUCTURE_VOID"]


def small(scale):
    return scale in BIG


def first_offer(r, by_role=None, by_model=False):
    """(role, price) of the first offer, optionally only by a role or only by the model."""
    for o in r["offers"]:
        msg_no, role, price = o[0], o[1], o[2]
        who = o[3] if len(o) > 3 else "model"
        if (by_role is None or role == by_role) and (not by_model or who == "model"):
            return role, price
    return None, None


def gap(role, price, r):
    if price is None:
        return None
    return 100 * (price / r["F"] - 1) if role == "seller" else 100 * (1 - price / r["C"])


def share(r):
    return r["seller_share"]["float"] if r.get("seller_share") else None


def per_fixture(recs, value, key):
    """{fixture: (small mean, big mean)} over games where value() is defined; key(r) groups the games."""
    acc = collections.defaultdict(lambda: {"s": [], "b": []})
    for r in recs:
        v = value(r)
        if v is None or not key(r):
            continue
        acc[r["fixture"]]["s" if small(r["scale"]) else "b"].append(v)
    return {f: (st.mean(d["s"]), st.mean(d["b"])) for f, d in acc.items() if d["s"] and d["b"]}


def test(values, what, model):
    if len(values) < 2:
        return {"n": len(values)}
    p, wplus, nz = A.wilcoxon_signflip(values, random.Random(A.seed(f"newgame-{what}", model)), flips=FLIPS)
    ci = A.bootstrap_ci(values, random.Random(A.seed(f"newgame-boot-{what}", model)), levels=(0.95,))["95"]
    return {"n": len(values), "mean": st.mean(values), "ci95": ci, "p_two_sided": p,
            "fixtures_below_0": sum(v < 0 for v in values), "fixtures_above_0": sum(v > 0 for v in values)}


def contrast(recs, value, key, what, model):
    pf = per_fixture(recs, value, key)
    out = test([b - s for s, b in pf.values()], what, model)
    if pf:
        out["level_small"] = st.mean(s for s, _ in pf.values())
        out["level_big"] = st.mean(b for _, b in pf.values())
    return out


def analyse(recs, model):
    cell = lambda c: (lambda r: r["cell"] == c)                                   # noqa: E731
    bot = lambda mr: (lambda r: r["cell"] == "BOT" and r["model_role"] == mr)     # noqa: E731
    opener_gap = lambda role: (lambda r: gap(*first_offer(r), r) if r["opener"] == role else None)  # noqa: E731
    model_gap = lambda r: gap(*first_offer(r, by_model=True), r)                  # noqa: E731
    C = {}
    for c in ("SPN", "SPR", "LEG"):
        C[f"{c} d"] = contrast(recs, share, cell(c), f"{c}-d", model)
        for role in ("seller", "buyer"):
            C[f"{c} {role}-opener gap"] = contrast(recs, opener_gap(role), cell(c), f"{c}-gap-{role}", model)
    for mr in ("seller", "buyer"):
        C[f"BOT model={mr} share"] = contrast(recs, share, bot(mr), f"BOT-{mr}", model)
        C[f"BOT model={mr} first-offer gap"] = contrast(recs, model_gap, bot(mr), f"BOT-gap-{mr}", model)
    # Does the appraisal protect? per fixture (SPR big-small) - (SPN big-small)
    a, b = per_fixture(recs, share, cell("SPR")), per_fixture(recs, share, cell("SPN"))
    C["SPR-SPN d"] = test([(a[f][1] - a[f][0]) - (b[f][1] - b[f][0]) for f in a if f in b], "SPR-SPN", model)
    return C


def slipped(r):
    """A model offer below 1/1000 of its own limit: the JSON number cut at the first comma
    ("price": 68, for $68,640,000). Seen only at x1e6 (STORY §7, 2026-10-04)."""
    for o in r["offers"]:
        if (o[3] if len(o) > 3 else "model") == "model" and o[2] < (r["F"] if o[1] == "seller" else r["C"]) / 1000:
            return True
    return False


def quality(recs):
    q = {}
    for c in ("SPN", "SPR", "BOT", "LEG"):
        for scale in ("small", "big"):
            rs = [r for r in recs if r["cell"] == c and small(r["scale"]) == (scale == "small")]
            if not rs:
                continue
            q[f"{c} {scale}"] = {
                "games": len(rs), "terminals": dict(collections.Counter(r["terminal"] for r in rs)),
                "invalid": dict(collections.Counter(x.split(":")[2] for r in rs for x in r["diagnostics"]
                                                    if x.startswith("INVALID:"))),
                "median_messages": st.median(r["n_messages"] for r in rs),
                "mean_seller_share": (st.mean(share(r) for r in rs if share(r) is not None)
                                      if any(share(r) is not None for r in rs) else None),
                "model_first_offer_with_cents": (sum(1 for r in rs if (first_offer(r, by_model=True)[1] or 0) % 100)
                                                 if c != "LEG" else None),
                "violations": sum(bool(r.get("violation")) for r in rs),
                "games_with_slipped_offer": sum(slipped(r) for r in rs)}
    return q


def descriptives(recs):
    """Where deals sit relative to the appraisal (SPR) and to the midpoint of the first offers."""
    out = {}
    for scale in ("small", "big"):
        spr = [r for r in recs if r["cell"] == "SPR" and r["terminal"] == "AGREEMENT"
               and small(r["scale"]) == (scale == "small")]
        if spr:
            out[f"SPR {scale}: deal minus appraisal, share of zone"] = st.mean(
                (r["price"] - r["R"]) / (r["C"] - r["F"]) for r in spr)
        mids = []
        for r in recs:
            if r["cell"] in ("SPN", "LEG") and r["terminal"] == "AGREEMENT" and small(r["scale"]) == (scale == "small"):
                s_role, s_p = first_offer(r, "seller")
                b_role, b_p = first_offer(r, "buyer")
                if s_p and b_p:
                    mids.append(((s_p + b_p) / 2 - r["F"]) / (r["C"] - r["F"]) - share(r))
        if mids:
            out[f"SPN+LEG {scale}: midpoint-of-first-offers share minus actual share"] = st.mean(mids)
    return out


def fig_shift(C, model, path, C2=None, n_slip=0):
    rows = [("SPN d", "self-play, no reference", GREEN, "share"),
            ("SPR d", "self-play, public appraisal", GREEN, "share"),
            ("BOT model=seller share", "model = seller vs bot buyer", BLUE, "share"),
            ("BOT model=buyer share", "model = buyer vs bot seller", ORANGE, "share"),
            ("LEG d", "bridge: exact Study 2 game", GREY, "share")]
    rows = [r for r in rows if "mean" in C.get(r[0], {})]
    W, rowh, T, L, PW, lo, hi = 1080, 48, 110, 300, 520, -1.2, 0.4
    H = T + rowh * len(rows) + 110
    s = Svg(W, H, f"New-game probe: change in seller share from x1 to x10^6, by cell · {model}",
            "One row per cell: the mean per-fixture change in the seller's share of the surplus when every amount"
            " gains six zeros, with a 95% bootstrap interval. Negative means the seller does worse at the large"
            " scale. Exploratory, not pre-registered.")
    s.text(30, 34, "Does the big-number squeeze survive the redesigned game?", 19, weight="bold")
    s.text(30, 56, f"Change in seller share, x10^6 minus x1, per fixture (95% interval) · {model} · exploratory",
           12.5, color="#444")
    X = lambda v: L + (v - lo) / (hi - lo) * PW                                   # noqa: E731
    for t in (-1.2, -0.8, -0.4, 0, 0.4):
        s.line(X(t), T - 10, X(t), T + rowh * len(rows), INK if t == 0 else "#e6e6e6", 1.2 if t == 0 else 1)
        s.text(X(t), T + rowh * len(rows) + 16, f"{t:+.1f}" if t else "0", 11, "middle", "#444")
    s.add(f"<rect x='{X(-0.1):.1f}' y='{T - 10}' width='{X(0.1) - X(-0.1):.1f}' height='{rowh * len(rows) + 10}'"
          f" fill='{GREEN}' fill-opacity='0.08'/>")
    for i, (k, label, color, _) in enumerate(rows):
        y, c = T + rowh * i + rowh / 2, C[k]
        s.text(30, y + 4, label, 12.5)
        a, b = c["ci95"]
        s.line(X(max(lo, a)), y, X(min(hi, b)), y, color, 3)
        s.add(f"<circle cx='{X(c['mean']):.1f}' cy='{y:.1f}' r='6' fill='{color}' stroke='white'/>")
        s.text(L + PW + 12, y + 4, f"{c['mean']:+.3f}  ({c['fixtures_below_0']}/{c['n']} below 0)", 11.5)
        c2 = (C2 or {}).get(k, {})
        if "mean" in c2:
            y2 = y + 12
            a2, b2 = c2["ci95"]
            s.line(X(max(lo, a2)), y2, X(min(hi, b2)), y2, color, 1.2, "4 3")
            s.add(f"<circle cx='{X(c2['mean']):.1f}' cy='{y2:.1f}' r='4.5' fill='white' stroke='{color}'"
                  " stroke-width='2'/>")
            s.text(L + PW + 12, y2 + 4, f"{c2['mean']:+.3f}  without slipped games", 10.5, color="#555")
    s.text(30, H - 58, "Shaded band: ±0.10, Study 2's equivalence margin. Bot games: the bot behaves identically at"
           " both scales, so any shift comes from the model's side.", 11.5, color="#444")
    if C2:
        s.text(30, H - 40, f"Hollow, dashed: the same without the {n_slip} games where a model price was cut at its first"
               " comma (\"price\": 68 for $68,640,000; x10^6 only).", 11.5, color="#444")
    s.text(30, H - 22, "Source: data/explore/<run>/games.jsonl via scripts/analyze_game_probe.py.", 11.5, color="#444")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    s.save(path)


def fig_gaps(C, model, path):
    rows = [(f"{c} {r}-opener gap", f"{c}: {r} opens", BLUE if r == "seller" else ORANGE)
            for c in ("SPN", "SPR", "LEG") for r in ("seller", "buyer")]
    rows += [("BOT model=seller first-offer gap", "BOT: model-seller's first offer", BLUE),
             ("BOT model=buyer first-offer gap", "BOT: model-buyer's first offer", ORANGE)]
    rows = [r for r in rows if "level_small" in C.get(r[0], {})]
    W, rowh, T, L, PW, lo, hi = 1000, 32, 110, 300, 480, 0, 90
    H = T + rowh * len(rows) + 90
    s = Svg(W, H, f"New-game probe: opener's first-offer gap at x1 and x10^6, by cell · {model}",
            "One row per cell and opening role: hollow dot = mean gap between the first offer and the opener's own"
            " limit at the small scale, filled dot = at the x10^6 scale, in percent of the limit; the line joins"
            " them. Exploratory.")
    s.text(30, 34, "Opening offers at small and large stakes, redesigned game", 19, weight="bold")
    s.text(30, 56, f"First-offer gap from own limit (%): hollow = x1, filled = x10^6 · {model} · exploratory",
           12.5, color="#444")
    X = lambda v: L + (v - lo) / (hi - lo) * PW                                   # noqa: E731
    for t in range(lo, hi + 1, 15):
        s.line(X(t), T - 10, X(t), T + rowh * len(rows), "#e6e6e6")
        s.text(X(t), T + rowh * len(rows) + 16, f"{t}%", 11, "middle", "#444")
    for i, (k, label, color) in enumerate(rows):
        y, c = T + rowh * i + rowh / 2, C[k]
        s.text(30, y + 4, label, 12)
        a, b = max(lo, min(hi, c["level_small"])), max(lo, min(hi, c["level_big"]))
        s.line(X(a), y, X(b), y, color, 2)
        s.add(f"<circle cx='{X(a):.1f}' cy='{y:.1f}' r='5.5' fill='white' stroke='{color}' stroke-width='2'/>")
        s.add(f"<circle cx='{X(b):.1f}' cy='{y:.1f}' r='5.5' fill='{color}'/>")
        s.text(L + PW + 12, y + 4, f"{c['level_small']:.1f} → {c['level_big']:.1f}  (Δ {c['mean']:+.1f})", 11.5)
    s.text(30, H - 22, "Source: data/explore/<run>/games.jsonl via scripts/analyze_game_probe.py.", 11.5, color="#444")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    s.save(path)


def zone_positions(recs):
    """Mean opener first offer as a position in the deal zone (floor 0, ceiling 1), by cell, role, scale."""
    out = {}
    for cell in ("LEG", "SPN", "SPR"):
        for role in ("seller", "buyer"):
            for sm in (True, False):
                v = [(first_offer(r)[1] - r["F"]) / (r["C"] - r["F"]) for r in recs
                     if r["cell"] == cell and r["opener"] == role and small(r["scale"]) == sm and r["offers"]]
                if v:
                    out[f"{cell} {role} {'small' if sm else 'big'}"] = st.mean(v)
    return out


def fig_zone(Z, model, path):
    """Where the openers start, in units of the deal zone: the geometry behind 'split moves' vs 'split holds'."""
    cells = [c for c in ("LEG", "SPN") if f"{c} seller small" in Z]
    W, H, T, L, PW, lo, hi = 1000, 470, 130, 230, 640, -1.2, 1.6
    s = Svg(W, H, f"Opening offers as a position in the deal zone, small vs x10^6 · {model}",
            "One row per game design. Horizontal axis: the deal zone, 0 = seller's floor, 1 = buyer's ceiling. Blue:"
            " where seller openers start; orange: buyer openers. Hollow = small scale, filled = x10^6. The diamond is the"
            " midpoint of the two openings. Games without a slipped offer. Exploratory.")
    s.text(30, 34, "Same modest openers, different geometry: why the split moves in one game and not the other", 18,
           weight="bold")
    s.text(30, 56, f"Opener's first offer in units of the deal zone (0 = floor, 1 = ceiling) · {model} · exploratory",
           12.5, color="#444")
    X = lambda v: L + (v - lo) / (hi - lo) * PW                                   # noqa: E731
    rowh = 130
    for t in (-1, -0.5, 0, 0.5, 1, 1.5):
        s.line(X(t), T - 20, X(t), T + rowh * len(cells) - 20, INK if t in (0, 1) else "#e6e6e6", 1.2 if t in (0, 1) else 1)
        s.text(X(t), T + rowh * len(cells) - 4, {0: "floor", 1: "ceiling"}.get(t, f"{t:g}"), 11, "middle", "#444")
    names = {"LEG": "Study 2 game (narrow zone around $100)", "SPN": "redesigned game (wide, off-centre zone)"}
    for i, c in enumerate(cells):
        y0 = T + rowh * i
        s.text(30, y0 - 4, names[c], 12.5, weight="bold")
        for j, (sm, lab) in enumerate(((True, "x1"), (False, "x10^6"))):
            y = y0 + 22 + j * 32
            s.text(L - 12, y + 4, lab, 11, "end", "#444")
            sp, bp = Z[f"{c} seller {'small' if sm else 'big'}"], Z[f"{c} buyer {'small' if sm else 'big'}"]
            s.line(X(min(sp, bp)), y, X(max(sp, bp)), y, "#bbbbbb", 2)
            for v, col in ((sp, BLUE), (bp, ORANGE)):
                fill = "white" if sm else col
                s.add(f"<circle cx='{X(v):.1f}' cy='{y:.1f}' r='6' fill='{fill}' stroke='{col}' stroke-width='2'/>")
            mid = (sp + bp) / 2
            s.add(f"<rect x='{X(mid) - 5:.1f}' y='{y - 5:.1f}' width='10' height='10' fill='{INK}'"
                  f" transform='rotate(45 {X(mid):.1f} {y:.1f})'/>")
            s.text(L + PW + 10, y + 4, f"midpoint {mid:+.2f}", 11, color=INK)
    s.text(30, H - 40, "Blue = the seller opened; orange = the buyer opened (separate games). Deals land near the midpoint"
           " of the two first offers (STORY X14), so the diamond tracks the split.", 11.5, color="#444")
    s.text(30, H - 22, "Source: data/explore/<run>/games.jsonl via scripts/analyze_game_probe.py.", 11.5, color="#444")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    s.save(path)


def fig_slips(recs, model, path):
    """How often a model price was cut at its first comma, by cell and scale."""
    cells = [c for c in ("SPN", "SPR", "BOT", "LEG") if any(r["cell"] == c for r in recs)]
    W, H, T, L, PW = 900, 190 + 56 * len(cells), 110, 300, 420
    s = Svg(W, H, f"Games with a price cut at its first comma, by cell and scale · {model}",
            "One row per cell: the share of games in which a model's JSON price was cut at the first comma (for example"
            " \"price\": 68 while the message says $68,640,000), at the small scale (hollow bar) and at x10^6"
            " (filled). A measurement artifact of the reply format; fixed in the next probe.")
    s.text(30, 34, "A format slip that only happens at big numbers", 19, weight="bold")
    s.text(30, 56, f"Share of games with a JSON price cut at its first comma · {model}", 12.5, color="#444")
    X = lambda v: L + v * PW                                                       # noqa: E731
    for t in (0, 0.25, 0.5):
        s.line(X(t), T - 10, X(t), T + 56 * len(cells) - 10, "#e6e6e6")
        s.text(X(t), T + 56 * len(cells) + 6, f"{100 * t:.0f}%", 11, "middle", "#444")
    for i, c in enumerate(cells):
        y = T + 56 * i
        s.text(30, y + 14, {"SPN": "new game, self-play", "SPR": "new game + appraisal", "BOT": "model vs bot",
                            "LEG": "Study 2 game (raw digits)"}[c], 12.5)
        for j, sm in enumerate((True, False)):
            rs = [r for r in recs if r["cell"] == c and small(r["scale"]) == sm]
            sh = sum(slipped(r) for r in rs) / len(rs) if rs else 0
            yy = y + 4 + j * 20
            fill = "white" if sm else ORANGE
            s.add(f"<rect x='{X(0):.1f}' y='{yy:.1f}' width='{max(1, X(sh) - X(0)):.1f}' height='14'"
                  f" fill='{fill}' stroke='{ORANGE}'/>")
            s.text(X(sh) + 6, yy + 11, f"{'x1' if sm else 'x10^6'}: {sum(slipped(r) for r in rs)} of {len(rs)}",
                   11, color=INK)
    s.text(30, H - 22, "Source: data/explore/<run>/games.jsonl via scripts/analyze_game_probe.py (slipped()).", 11.5,
           color="#444")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    s.save(path)


def main(argv):
    if len(argv) != 1 or not os.path.isdir(argv[0]):
        print(__doc__.strip().splitlines()[2].strip())
        return 2
    run_dir = argv[0].rstrip("/")
    man = json.load(open(os.path.join(run_dir, "run_manifest.json")))
    model, run_id = man["model"], man["run_id"]
    recs = load(run_dir)
    C = analyse(recs, model)
    clean = [r for r in recs if not slipped(r)]
    C_clean = analyse(clean, model + "-clean")
    summary = {"run_id": run_id, "model": model, "exploratory": True, "games": len(recs),
               "quality": quality(recs), "contrasts": C, "descriptives": descriptives(recs),
               "games_with_slipped_offer": len(recs) - len(clean),
               "contrasts_excluding_slipped_games": C_clean}
    out = os.path.join(ROOT, "derived", "explore", run_id)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    fig_shift(C, model, os.path.join(out, "figures", "G1_scale_shift_by_cell.svg"), C_clean,
              summary["games_with_slipped_offer"])
    fig_gaps(C, model, os.path.join(out, "figures", "G2_opening_gaps_by_cell.svg"))

    summary["zone_positions_excluding_slipped"] = zone_positions(clean)
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    if any(r["cell"] in ("LEG", "SPN") for r in clean):
        fig_zone(summary["zone_positions_excluding_slipped"], model,
                 os.path.join(out, "figures", "G3_zone_positions.svg"))
    fig_slips(recs, model, os.path.join(out, "figures", "G4_comma_slips.svg"))
    print(f"New-game probe · {model} · {len(recs)} games · EXPLORATORY")
    print("\nQuality: games, terminals, invalid replies, median messages, mean seller share")
    for k, q in summary["quality"].items():
        ms = "—" if q["mean_seller_share"] is None else f"{q['mean_seller_share']:.3f}"
        print(f"  {k:10} {q['games']:4}  {q['terminals']}  invalid {q['invalid'] or 0}  msgs {q['median_messages']}"
              f"  share {ms}" + (f"  violations {q['violations']}" if q["violations"] else ""))
    for title, CC in (("as recorded", C), (f"excluding the {summary['games_with_slipped_offer']} games with a"
                                               " slipped offer (sensitivity)", C_clean)):
        print(f"\nScale contrasts per fixture, {title} (big minus small): mean [95%]  p  below/above 0   small → big")
        show(CC)
    print("\nDescriptives")
    for k, val in summary["descriptives"].items():
        print(f"  {k}: {val:+.3f}")
    print(f"\nWrote {os.path.relpath(out, ROOT)}/summary.json and figures/G1-G4")
    return 0


def show(C):
    for k, c in C.items():
        if "mean" not in c:
            print(f"  {k:34} n={c['n']}")
            continue
        lv = f"   {c['level_small']:.3f} → {c['level_big']:.3f}" if "level_small" in c else ""
        print(f"  {k:34} {c['mean']:+8.3f}  [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]  p {c['p_two_sided']:.4f}"
              f"  {c['fixtures_below_0']}/{c['fixtures_above_0']} of {c['n']}{lv}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
