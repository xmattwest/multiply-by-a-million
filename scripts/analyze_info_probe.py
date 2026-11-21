#!/usr/bin/env python3
"""analyze_info_probe.py: the asymmetric-information probe (STORY §12, Q17). Exploratory.

Usage: python3 scripts/analyze_info_probe.py data/explore/X-info-<model>-<stamp>

Labels: "SP A/N" = self-play, the seller has private information A, the buyer has none; "BOT P
model=buyer" = the model as buyer, knowing the seller's range, vs the scale-proof bot (which knows
nothing). Codes: N nothing, A a private off-centre appraisal, P the range the other side's limit was
drawn from. Same measures and contrasts as analyze_cell_probe.py (share: seller's share in self-play,
the model's own share against the bot; opening gaps; x1e6 - x1 per fixture; minus the N/N or BOT N
cell). Added here: the appraisal cells split by where the appraisal sits (seller-favourable, above
the zone's midpoint, vs buyer-favourable), because an informed side's gain depends on it.
Writes derived/explore/<run_id>/summary.json and figures IX1, IX2. Standard library only.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import analyze_cell_probe as CP                                 # noqa: E402
import analyze_game_probe as G                                  # noqa: E402

WHAT = {"N": "nothing", "A": "private appraisal", "P": "knows the other's range"}


def row_name(g, lab, multi):
    if lab.startswith("SP "):
        s, b = lab[3:].split("/")
        return f"self-play: seller {WHAT[s]}, buyer {WHAT[b]}"
    if lab.startswith("BOT "):
        _, code, role = lab.split(" ", 2)
        return f"vs bot: model ({role.split('=')[1]}), {WHAT[code]}"
    return lab


def label_order(labels):
    sp_key = {"SP N/N": 0, "SP A/N": 1, "SP N/A": 2, "SP A/A": 3, "SP P/N": 4, "SP N/P": 5, "SP P/P": 6}
    sp = sorted((l for l in labels if l.startswith("SP ")), key=lambda l: (sp_key.get(l, 9), l))
    bot = sorted((l for l in labels if l.startswith("BOT ")),
                 key=lambda l: (l.split()[2], "NAP".index(l.split()[1]) if l.split()[1] in "NAP" else 9))
    return sp + bot


def appraisal_side(r):
    pos = (r["R_private"] - r["F"]) / (r["C"] - r["F"])
    return "seller-favourable" if pos > 0.5 else "buyer-favourable"


def split_by_appraisal(recs, model):
    """Shift per appraisal cell, separately for fixtures whose appraisal sits above vs below the
    zone's midpoint (15 fixtures each), and the same fixtures' N/N or BOT N shift for comparison."""
    out = {}
    for side in ("seller-favourable", "buyer-favourable"):
        rs = [r for r in recs if r.get("R_private") and appraisal_side(r) == side]
        for lab in label_order({r["label"] for r in rs}):
            if "A" not in lab.split()[1] and lab != CP.baseline(lab):
                continue
            pf = G.per_fixture([r for r in rs if r["label"] == lab], CP.model_share, lambda r: True)
            if len(pf) >= 2:
                res = G.test([b - s for s, b in pf.values()], f"info-{side}-{lab}", model)
                res["level_small"] = sum(s for s, _ in pf.values()) / len(pf)
                res["level_big"] = sum(b for _, b in pf.values()) / len(pf)
                out[f"{side} | {lab} | share"] = res
    return out


def main(argv):
    if len(argv) != 1 or not os.path.isdir(argv[0]):
        print(__doc__.strip().splitlines()[2].strip())
        return 2
    run_dir = argv[0].rstrip("/")
    man = json.load(open(os.path.join(run_dir, "run_manifest.json")))
    model, run_id = man["model"], man["run_id"]
    CP.row_name, CP.label_order = row_name, label_order          # names and order for this design
    recs = [r for r in G.load(run_dir) if r.get("label")]
    C, groups = CP.analyse(recs, model)
    S = split_by_appraisal(recs, model)
    out = os.path.join(ROOT, "derived", "explore", run_id)
    os.makedirs(out, exist_ok=True)
    terminals = {}
    for r in recs:
        terminals[r["terminal"]] = terminals.get(r["terminal"], 0) + 1
    quality = {"games": len(recs), "terminals": terminals, "slipped": sum(G.slipped(r) for r in recs),
               "violations": sum(bool(r.get("violation")) for r in recs),
               "invalid": sum(r["terminal"] == "PROTOCOL_FORFEIT" for r in recs)}
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump({"run_id": run_id, "model": model, "design": "info", "exploratory": True, "quality": quality,
                   "contrasts": C, "by_appraisal_side": S}, fh, indent=1)
    legend = ("Private information for one side only: A = an independent appraisal the other side hasn't seen"
              " (off-centre); P = the range the other side's limit was drawn from. The bot never sees it.")
    CP.fig_levels(C, groups, model, os.path.join(out, "figures", "IX1_share_by_cell.svg"), "IX", legend)
    CP.fig_gaps(C, groups, model, os.path.join(out, "figures", "IX2_opening_gaps_by_cell.svg"), "IX")
    for name in ("IX1_share_by_cell.svg", "IX2_opening_gaps_by_cell.svg"):        # name this script as the source
        f = os.path.join(out, "figures", name)
        txt = open(f, encoding="utf-8").read().replace("analyze_cell_probe.py", "analyze_info_probe.py")
        open(f, "w", encoding="utf-8").write(txt)

    print(f"info probe · {model} · {len(recs)} games · EXPLORATORY")
    print(f"\nQuality: {quality}")
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
    print("\nAppraisal cells by where the appraisal sits (15 fixtures each)")
    for k, c in S.items():
        if "mean" in c:
            print(f"  {k:52} {c['mean']:+8.3f}  [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]"
                  f"   {c['level_small']:.3f} → {c['level_big']:.3f}")
    print(f"\nWrote {os.path.relpath(out, ROOT)}/summary.json and figures/IX1, IX2")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
