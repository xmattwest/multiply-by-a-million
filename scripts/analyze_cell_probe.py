#!/usr/bin/env python3
"""analyze_cell_probe.py: the prompt-style probe (Q14) and the cross-play probe (Q15). Exploratory.

Usage: python3 scripts/analyze_cell_probe.py data/explore/X-prompt-<model>-<stamp>
       python3 scripts/analyze_cell_probe.py data/explore/X-cross-<m1>-vs-<m2>-<stamp>

Every game carries a label: "SP A/N" (self-play, seller style A, buyer style N), "BOT C model=buyer"
(the model in style C vs the scale-proof bot), or "G/Q" (cross-play: Gemma sells, Qwen buys). Per label
(and, in cross-play, per zone + notation group), per fixture, at x1 and x1e6:
  share      self-play rows: the seller's share of the zone; BOT rows: the MODEL's share
             (so a drop always means the side we are watching gets less)
  gaps       the opener's first-offer gap from its own limit, percentage points (seller and buyer
             openers separately; BOT rows: the model's first offer)
Contrasts, paired by fixture: x1e6 - x1 for each label; and that shift minus the baseline label's
shift (SP N/N, BOT N with the same role, or G/G in cross-play).
Writes derived/explore/<run_id>/summary.json and two figures (PX1/PX2 for prompt, MX1/MX2 for cross).
Standard library only.
"""
import collections
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import analyze_game_probe as G                                  # noqa: E402
from make_figures import BLUE, GREY, INK, ORANGE, Svg           # noqa: E402

GREEN_BAND = "#e8f6f0"


def group_of(r):
    return f"{r.get('geom')} {r.get('notation')}"


def model_share(r):
    s = G.share(r)
    if s is None:
        return None
    return (1 - s) if (r["cell"] == "BOT" and r.get("model_role") == "buyer") else s


def opener_gap(role):
    def f(r):
        if r["cell"] == "BOT":
            return None
        return G.gap(*G.first_offer(r), r) if r["opener"] == role else None
    return f


def model_gap(r):
    return G.gap(*G.first_offer(r, by_model=True), r) if r["cell"] == "BOT" else None


MEASURES = {"share": model_share, "seller-opener gap": opener_gap("seller"),
            "buyer-opener gap": opener_gap("buyer"), "model first-offer gap": model_gap}


def baseline(label):
    if label.startswith("SP "):
        return "SP N/N"
    if label.startswith("BOT "):
        return "BOT N " + label.split(" ", 2)[2]
    return "G/G"


def label_order(labels):
    sp = [l for l in labels if l.startswith("SP ")]
    bot = [l for l in labels if l.startswith("BOT ")]
    other = [l for l in labels if l not in sp and l not in bot]
    key_sp = {"SP N/N": 0, "SP A/A": 1, "SP C/C": 2, "SP D/D": 3}
    sp.sort(key=lambda l: (key_sp.get(l, 9), l))
    bot.sort(key=lambda l: (l.split()[2], "NACD".index(l.split()[1]) if l.split()[1] in "NACD" else 9))
    pref = {"G/G": 0, "Q/Q": 1, "G/Q": 2, "Q/G": 3}
    other.sort(key=lambda l: (pref.get(l, 9), l))
    return other + sp + bot


def analyse(recs, model):
    out, per = {}, {}
    groups = sorted({group_of(r) for r in recs})
    for g in groups:
        rg = [r for r in recs if group_of(r) == g]
        for lab in label_order({r["label"] for r in rg}):
            rs = [r for r in rg if r["label"] == lab]
            for name, val in MEASURES.items():
                pf = G.per_fixture(rs, val, lambda r: True)
                if not pf:
                    continue
                diffs = {f: b - s for f, (s, b) in pf.items()}
                per[(g, lab, name)] = diffs
                res = G.test(list(diffs.values()), f"{g}-{lab}-{name}", model)
                res["level_small"] = st.mean(s for s, _ in pf.values())
                res["level_big"] = st.mean(b for _, b in pf.values())
                out[f"{g} | {lab} | {name}"] = res
        for (gg, lab, name), d in list(per.items()):
            if gg != g or lab == baseline(lab):
                continue
            b = per.get((g, baseline(lab), name), {})
            common = [f for f in d if f in b]
            if len(common) >= 2:
                out[f"{g} | {lab} minus {baseline(lab)} | {name}"] = G.test(
                    [d[f] - b[f] for f in common], f"{g}-{lab}-vs-base-{name}", model)
    return out, groups


def rows_for(C, groups):
    rows = []
    for g in groups:
        labs = [k.split(" | ")[1] for k in C if k.startswith(g + " | ") and k.endswith(" | share")
                and " minus " not in k]
        rows += [(g, lab) for lab in label_order(labs)]
    return rows


def row_name(g, lab, multi):
    if lab.startswith("SP "):
        s, b = lab[3:].split("/")
        txt = f"self-play: seller {s}, buyer {b}"
    elif lab.startswith("BOT "):
        _, sty, role = lab.split(" ", 2)
        txt = f"vs bot: model ({role.split('=')[1]}) in style {sty}"
    else:
        names = {"G": "Gemma", "Q": "Qwen"}
        s, b = lab.split("/")
        txt = f"{names.get(s, s)} sells, {names.get(b, b)} buys"
    return (g.replace("round", "Study 2 zone").replace("wide", "wide zone") + " · " + txt) if multi else txt


def fig_levels(C, groups, model, path, prefix, legend):
    rows = rows_for(C, groups)
    W, T, L, PW, rowh = 1200, 120, 400, 420, 26
    H = T + rowh * len(rows) + 90
    X = lambda v: L + max(-0.05, min(1.05, v)) * PW                                # noqa: E731
    s = Svg(W, H, f"{prefix}1: share of the deal zone at x1 and x10^6, per cell · {model}",
            "One row per cell. Hollow dot: mean share at x1; filled dot: at x10^6 (per-fixture means). Self-play"
            " rows show the seller's share; bot rows show the model's own share. Right: the change x10^6 minus"
            " x1 with its 95% interval. Exploratory.")
    s.text(30, 34, "Share of the deal zone at small and big stakes, per cell", 19, weight="bold")
    s.text(30, 56, f"Mean per fixture; hollow = x1, filled = x10^6 · {model} · exploratory", 12.5, color="#444")
    for t in (0, 0.25, 0.5, 0.75, 1):
        s.line(X(t), T - 18, X(t), T + rowh * len(rows) - 8, INK if t == 0.5 else "#e6e6e6")
        s.text(X(t), T + rowh * len(rows) + 8, f"{t:g}", 11, "middle", "#444")
    s.text(L + PW + 50, T - 22, "change x10^6 − x1 [95%]", 11.5, weight="bold")
    multi = len(groups) > 1
    for i, (g, lab) in enumerate(rows):
        y = T + rowh * i
        c = C[f"{g} | {lab} | share"]
        col = ORANGE if lab.startswith("BOT ") else BLUE
        s.text(30, y + 4, row_name(g, lab, multi), 12)
        a, b = c["level_small"], c["level_big"]
        s.line(X(min(a, b)), y, X(max(a, b)), y, "#cccccc", 2)
        s.add(f"<circle cx='{X(a):.1f}' cy='{y:.1f}' r='5' fill='white' stroke='{col}' stroke-width='2'/>")
        s.add(f"<circle cx='{X(b):.1f}' cy='{y:.1f}' r='5' fill='{col}' stroke='{col}' stroke-width='2'/>")
        if "ci95" in c:
            s.text(L + PW + 50, y + 4, f"{c['mean']:+.3f}  [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]", 11.5,
                   color=INK if c["ci95"][1] < 0 or c["ci95"][0] > 0 else "#777")
    s.text(30, H - 40, legend, 11.5, color="#444")
    s.text(30, H - 22, "Source: data/explore/<run>/games.jsonl via scripts/analyze_cell_probe.py.", 11.5,
           color="#444")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    s.save(path)


def fig_gaps(C, groups, model, path, prefix):
    rows = []
    for g, lab in rows_for(C, groups):
        names = ("model first-offer gap",) if lab.startswith("BOT ") else ("seller-opener gap", "buyer-opener gap")
        rows += [(g, lab, n) for n in names if f"{g} | {lab} | {n}" in C]
    W, T, L, PW, rowh, hi = 1200, 120, 470, 480, 22, 90
    H = T + rowh * len(rows) + 90
    X = lambda v: L + max(0, min(hi, v)) / hi * PW                                 # noqa: E731
    s = Svg(W, H, f"{prefix}2: opening ambition at x1 and x10^6, per cell · {model}",
            "One row per cell and opening side. The gap between the first offer and the side's own limit, in"
            " percentage points of that limit. Hollow = x1, filled = x10^6. Blue = seller opens; orange = buyer"
            " opens; green = the model's first offer against the bot. Exploratory.")
    s.text(30, 34, "How ambitious is the first offer, at small and big stakes?", 19, weight="bold")
    s.text(30, 56, f"First offer's distance from own limit, % of the limit · {model} · exploratory", 12.5,
           color="#444")
    for t in range(0, hi + 1, 15):
        s.line(X(t), T - 16, X(t), T + rowh * len(rows) - 8, "#e6e6e6")
        s.text(X(t), T + rowh * len(rows) + 8, f"{t}", 11, "middle", "#444")
    multi = len(groups) > 1
    for i, (g, lab, n) in enumerate(rows):
        y = T + rowh * i
        c = C[f"{g} | {lab} | {n}"]
        col = {"seller-opener gap": BLUE, "buyer-opener gap": ORANGE}.get(n, "#2a9d6f")
        who = {"seller-opener gap": "seller opens", "buyer-opener gap": "buyer opens"}.get(n, "model's offer")
        s.text(30, y + 4, f"{row_name(g, lab, multi)} · {who}", 11.5)
        a, b = c["level_small"], c["level_big"]
        s.line(X(min(a, b)), y, X(max(a, b)), y, "#cccccc", 2)
        s.add(f"<circle cx='{X(a):.1f}' cy='{y:.1f}' r='4.5' fill='white' stroke='{col}' stroke-width='2'/>")
        s.add(f"<circle cx='{X(b):.1f}' cy='{y:.1f}' r='4.5' fill='{col}' stroke='{col}' stroke-width='2'/>")
        s.text(L + PW + 16, y + 4, f"{a:.0f} → {b:.0f}", 11, color=INK)
    s.text(30, H - 22, "Source: data/explore/<run>/games.jsonl via scripts/analyze_cell_probe.py.", 11.5,
           color="#444")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    s.save(path)


def main(argv):
    if len(argv) != 1 or not os.path.isdir(argv[0]):
        print(__doc__.strip().splitlines()[2].strip())
        return 2
    run_dir = argv[0].rstrip("/")
    man = json.load(open(os.path.join(run_dir, "run_manifest.json")))
    model, run_id, design = man["model"], man["run_id"], man.get("design")
    recs = [r for r in G.load(run_dir) if r.get("label")]
    C, groups = analyse(recs, model)
    quality = {}
    for g in groups:
        rs = [r for r in recs if group_of(r) == g]
        quality[g] = {"games": len(rs), "terminals": dict(collections.Counter(r["terminal"] for r in rs)),
                      "invalid": dict(collections.Counter(x.split(":")[2] for r in rs for x in r["diagnostics"]
                                                          if x.startswith("INVALID:"))),
                      "slipped": sum(G.slipped(r) for r in rs),
                      "violations": sum(bool(r.get("violation")) for r in rs)}
    out = os.path.join(ROOT, "derived", "explore", run_id)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump({"run_id": run_id, "model": model, "design": design, "exploratory": True, "games": len(recs),
                   "quality": quality, "contrasts": C}, fh, indent=1)
    prefix = "MX" if design == "cross" else "PX"
    legend = ("Styles: N = no extra instruction; A = 'negotiate hard'; C = 'a deal matters more than the price';"
              " D = 'the size of the numbers doesn't change good negotiating'." if prefix == "PX" else
              "G = gemma-4-31b-it, Q = qwen3.8-27b-fp8 (seller/buyer). Each model on its own pod, same settings.")
    fig_levels(C, groups, model, os.path.join(out, "figures", f"{prefix}1_share_by_cell.svg"), prefix, legend)
    fig_gaps(C, groups, model, os.path.join(out, "figures", f"{prefix}2_opening_gaps_by_cell.svg"), prefix)

    print(f"{design} probe · {model} · {len(recs)} games · EXPLORATORY")
    print("\nQuality: games, terminals, invalid replies, slipped, violations")
    for k, q in quality.items():
        print(f"  {k:12} {q['games']:4}  {q['terminals']}  invalid {q['invalid'] or 0}  slipped {q['slipped']}"
              f"  violations {q['violations']}")
    print("\nPer cell, x1e6 minus x1, per fixture: mean [95%]  p  below/above 0   x1 → x1e6")
    print("(share: seller's share in self-play, the model's own share against the bot)")
    for k, c in C.items():
        if " minus " in k or "mean" not in c:
            continue
        print(f"  {k:52} {c['mean']:+8.3f}  [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]  p {c['p_two_sided']:.4f}"
              f"  {c['fixtures_below_0']}/{c['fixtures_above_0']} of {c['n']}   {c['level_small']:.3f} →"
              f" {c['level_big']:.3f}")
    print("\nShift minus the baseline cell's shift (paired by fixture)")
    for k, c in C.items():
        if " minus " in k and "mean" in c and k.endswith(" | share"):
            print(f"  {k:52} {c['mean']:+8.3f}  [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]  p {c['p_two_sided']:.4f}")
    print(f"\nWrote {os.path.relpath(out, ROOT)}/summary.json and figures/{prefix}1, {prefix}2")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
