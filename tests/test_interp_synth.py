#!/usr/bin/env python3
"""test_interp_synth.py: checks for the X62 synthesis card (scripts/interp_synth_figures.py behind the shim
scripts/interp/mi_figures_synth.py). Standard library only; no pod, no torch.

    python3 tests/test_interp_synth.py

Synthetic: the new-prompt rule (held-out fixtures whose limit is not in discovery, one per limit, unreadable ones
dropped), the share rules for heads-style and cure records, the unreadable-readout rule, Spearman. If the three local
runs exist (data/interp/, gitignored): refresh() reproduces STORY X62's numbers. Always: no JSON -> no card; the
tracked JSON renders one card with one SVG; the site's plug-in loader finds the shim.
"""
import json
import math
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "interp"))
N = {"pass": 0, "fail": 0}


def ok(cond, name):
    N["pass" if cond else "fail"] += 1
    print(("ok    " if cond else "FAIL  ") + name)


def near(a, b, tol=0.005):
    return a is not None and abs(a - b) <= tol


def synthetic(F):
    lim = {"Y01": 80, "Y02": 81, "Y03": 80, "Y04": 82, "Y05": 82, "Y06": 83}
    ok(F.new_prompts(lim, {"Y01", "Y02"}, {"Y03", "Y04", "Y05", "Y06"}) == ["Y04", "Y06"],
       "new prompts: repeats of discovery limits dropped, one fixture per limit")
    ok(F.new_prompts(lim, {"Y01", "Y02"}, {"Y03", "Y04", "Y05", "Y06"}, drop=("Y04",)) == ["Y05", "Y06"],
       "new prompts: a dropped fixture is replaced by the next one with the same limit")
    recs = [{"fixture": "Y04", "direction": "x1->x1e6", "ld_clean": 0.0, "ld_source": 10.0, "rows": [{"n": "top8", "ld": 4.0}]},
            {"fixture": "Y06", "direction": "x1->x1e6", "ld_clean": 1.0, "ld_source": 21.0, "rows": [{"n": "top8", "ld": 3.0}]},
            {"fixture": "Y05", "direction": "x1->x1e6", "ld_clean": 0.0, "ld_source": 10.0, "rows": [{"n": "top8", "ld": 9.0}]}]
    c = F.shares(recs, {"Y04", "Y06"}, {"top8"})[("x1->x1e6", "top8")]
    ok(near(c["share"], (4 + 2) / 2 / ((10 + 20) / 2)) and c["n"] == 2, "shares: mean moved / mean gap over kept fixtures")
    T = lambda ld, rows: {"ld_clean": ld, "rows": rows}                                # noqa: E731
    cure = [{"fixture": "Y04", "targets": {"N6": T(5.0, [{"n": "top8", "ld": -5.0}]), "P6": T(-15.0, [{"n": "top8", "ld": -7.0}])}}]
    C = F.cure_shares(cure, {"Y04"}, {"top8"})
    ok(near(C[("N6", "top8")]["share"], 0.5) and near(C[("P6", "top8")]["share"], 0.4),
       "cure shares: by the ld distance between the two runs swapped, both directions")
    lp = lambda p: math.log(p)                                                        # noqa: E731
    fx = [{"fixture": "Y01", "lpa": {"P1": lp(0.6), "P6": lp(0.3)}, "lpb": {"P1": lp(0.1), "P6": lp(0.3)}},
          {"fixture": "Y02", "lpa": {"P1": lp(0.3), "P6": lp(0.9)}, "lpb": {"P1": lp(0.1), "P6": lp(0.05)}}]
    ok(F.unreadable(fx) == ["Y02"], "unreadable: p(a)+p(b) < 0.5 on either range prompt")
    ok(near(F.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0) and near(F.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0),
       "spearman: +1 / -1 on monotone data")
    hm = F.head_map({("a", "h41.1"): {"share": 0.2}, ("b", "h41.1"): {"share": 0.4}, ("a", "h41.2"): {"share": 0.1}}, ["41.1", "41.2"])
    ok(near(hm.get("41.1"), 0.3) and "41.2" not in hm, "head map: mean of both directions; a head missing one is left out")


def real(F):
    have = all(os.path.exists(os.path.join(ROOT, d, "results.jsonl")) for d in F.RUNS.values())
    if not have:
        print("skip  the three local runs are not here (data/interp is gitignored); X62's numbers not re-checked")
        return
    tmp = tempfile.mkdtemp()
    try:
        D = F.refresh(out_path=os.path.join(tmp, "s.json"))
        C = D["circuit"]
        ok(D["n_new"] == {"scale": 9, "cure": 7, "buyers": 6} and D["unreadable"] == ["Y18", "Y30"],
           "real runs: 9 / 7 / 6 new prompts; Y18 and Y30 unreadable with the range")
        ok(near(C["x1->x1e6"]["scale"]["share"], 0.404) and near(C["x1e6->x1"]["scale"]["share"], 0.681)
           and near(C["x1->x1e6"]["cure"]["share"], 0.690) and near(C["x1e6->x1"]["cure"]["share"], 0.813)
           and near(C["x1->x1e6"]["buyers"]["share"], 0.298) and near(C["x1e6->x1"]["buyers"]["share"], 0.580),
           "real runs: X62's new-prompt circuits (40/68, 69/81, 30/58)")
        ok(D["overlap8"] == ["41.11", "41.14", "41.29"] and near(D["spearman"], 0.13, 0.01) and len(D["heads"]) == 224,
           "real runs: top-8 overlap 3, Spearman +0.13 over 224 heads")
        L = D["layers"].index(41)
        ok(near(D["by_layer"]["scale_in"][L], 0.763) and near(D["by_layer"]["cure_in"][L], 0.861)
           and D["by_layer"]["cure_in"][L - 1] < 0.05, "real runs: the jump at 41 (scale 76%, cure 2% -> 86%)")
    finally:
        shutil.rmtree(tmp)


def main():
    import interp_synth_figures as F
    synthetic(F)
    real(F)
    ok(F.cards(os.path.join(tempfile.gettempdir(), "x62-no-such.json")) == [], "figure: no JSON -> no card")
    if os.path.exists(F.JSON_PATH):
        c = F.cards()
        ok(len(c) == 1 and c[0].count("<svg") == 1 and "synthesis · X62" in c[0] and "<desc" in c[0],
           "figure: the tracked JSON renders one card with one SVG (title and description)")
    import mi_figures
    ok(any("synthesis · X62" in x for x in mi_figures.extra_cards()) == os.path.exists(F.JSON_PATH),
       "the site's plug-in loader finds the shim (a card only once the tracked JSON exists)")
    print(f"\n{N['pass']} passed, {N['fail']} failed")
    return 1 if N["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
