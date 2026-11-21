#!/usr/bin/env python3
"""power_study3.py: projected precision of Study 3's tests at n = 40 fixtures
(prereg/study3/PREREGISTRATION.md §7). Reads earlier data only; no Study 3 game.

    python3 scripts/power_study3.py          prints the table; writes logs/power_study3.json

For each arm, the per-fixture SDs of the earlier analogues of each statistic:
  d     closing shift $100M - $100 (Study 2 for Qwen and Gemma, k = 1; Addendum C for the API arms, k = 4);
  h     opening-gap change (HA), the analogue of one arm of P1's interaction;
  e     closing shift minus opening-midpoint shift (Addendum C's HM; Study 2 games for the open models),
        the analogue of P3b's (omega - delta m).
Projected 90% CI half-width at n fixtures: 1.645 * SD / sqrt(n). P1's interaction is a difference of two
h's, so its SD is taken as sqrt(2) * SD(h) (independent conditions; conservative if they correlate).
P3a's kappa is a difference of two closing contrasts; its SD is taken as SD(d). Open-model SDs come
from k = 1 cells and overstate Study 3's (k = 4) spread. Standard library only.
"""
import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import analyze_addendum_c as AC     # noqa: E402

N = 40
Z90 = 1.6449
RUNS = {
    "qwen3.8-27b-fp8": ("data/runs/s2-qwen3.8-27b-fp8-20261003T223042Z", None),
    "gemma-4-31b-it": ("data/runs/s2-gemma-4-31b-it-20261004T012932Z", None),
    "gpt-6-luna": ("data/runs/addC-20261011T182548Z", "gpt-6-luna"),
    "claude-haiku-5.5": ("data/runs/addC-20261011T182548Z", "claude-haiku-5.5"),
    "gemini-3.5-flash-lite": ("data/runs/addC-20261011T182548Z", "gemini-3.5-flash-lite"),
}


def stats_for(run_dir, arm):
    _, scored, _, _ = AC.load(os.path.join(ROOT, run_dir))
    games = [r for r in scored.values() if arm is None or r.get("arm") == arm]
    fixtures = sorted({r["fixture"] for r in games})
    cells = AC.cells_of(games)
    d = [float(x) for x in AC.paired(cells, fixtures, AC.share).values()]
    h = [float(x) for x in AC.paired(cells, fixtures, AC.opening_gap).values()]
    e = [float(ds - ms) for ds, ms in AC.hm_parts(cells, fixtures).values()]
    return {k: {"n": len(v), "mean": statistics.fmean(v), "sd": statistics.stdev(v)} for k, v in
            (("d", d), ("h", h), ("e", e))}


def half(sd, n=N):
    return Z90 * sd / math.sqrt(n)


def main():
    out = {"n_fixtures": N, "per_arm": {}}
    print(f"Projected 90% CI half-widths at n = {N} fixtures (earlier per-fixture SDs)")
    print(f"{'arm':22s} {'SD d':>6s} {'P2 d':>6s} {'SD h':>6s} {'P1 I':>6s} {'P3a k':>6s} {'SD e':>6s} {'P3b':>6s}")
    for arm, (run, filt) in RUNS.items():
        s = stats_for(run, filt)
        row = {"earlier": s, "half_P2_d": half(s["d"]["sd"]), "half_P1_interaction": half(math.sqrt(2) * s["h"]["sd"]),
               "half_P3a_kappa": half(s["d"]["sd"]), "half_P3b_omega_minus_dm": half(s["e"]["sd"])}
        out["per_arm"][arm] = row
        print(f"{arm:22s} {s['d']['sd']:6.3f} {row['half_P2_d']:6.3f} {s['h']['sd']:6.3f}"
              f" {row['half_P1_interaction']:6.3f} {row['half_P3a_kappa']:6.3f} {s['e']['sd']:6.3f}"
              f" {row['half_P3b_omega_minus_dm']:6.3f}")
    print("Margins: P2 +-0.10 (H2); P3a and P3b +-0.05. A half-width above the margin cannot establish equivalence"
          " even at a true value of 0.")
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    with open(os.path.join(ROOT, "logs", "power_study3.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
