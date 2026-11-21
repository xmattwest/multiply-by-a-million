#!/usr/bin/env python3
"""power_addendum_b.py: how many samples per cell does Addendum B need?

WHAT IT DOES
  Addendum B tests HA on three API models: for each of the 30 fixtures,
  h_f = mean over the two openers of [mean gap at x10^6 - mean gap at x1],
  where each cell mean is over k independent first moves. One-sided
  Wilcoxon signed-rank (alternative: median < 0), Holm across 3 models.

  The spread of h_f has two parts: a fixture-level part that doesn't shrink
  with k (sd_true), and sampling noise in each first move (sd_within per
  sample), which enters h_f with variance sd_within^2 / k (four cell means,
  each over k samples, weighted 1/2 by the opener average:
  0.25 * 2 openers * 2 cells * sd_within^2 / k).

  The two parts are estimated from the data we have (all printed below):
    - Gemma 31B digits probe, D2 vs D8 on Y01..Y30, k = 4 (two runs):
      within-cell SD per sample and SD of h_f.
    - Gemma 31B and Qwen Study 2 (k = 1): SD of h_f.
  Qwen has no k > 1 first-move data, so its split into the two parts is an
  assumption (stated). A pessimistic scenario doubles both parts, in case
  the API models are noisier than either open model.

  The test is simulated with the exact no-ties Wilcoxon null (as
  scripts/power_sim.py), at alpha 0.05/3 (the strictest Holm step for three
  models). Normal effects; real data have ties, so read these as optimistic.

USAGE
  python3 scripts/power_addendum_b.py        prints the table; ~1 min
Deterministic (fixed seed). Standard library only. Free, offline.
"""
import json
import math
import os
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import analyze as A               # noqa: E402

N = 30
SIMS = 5000
SEED = 20261010
ALPHA = 0.05 / 3
EFFECTS = (0.02, 0.03, 0.04, 0.05, 0.08)
KS = (1, 2, 4, 8)
DIGITS_RUNS = ("X-digits-gemma-4-31b-it-20261004T034215Z", "X-digits-gemma-4-31b-it-20261004T042802Z")
S2_RUNS = ("s2-gemma-4-31b-it-20261004T012932Z", "s2-qwen3.8-27b-fp8-20261003T223042Z")


def wilcoxon_null(n):
    counts = [1]
    for k in range(1, n + 1):
        new = counts + [0] * k
        for s, c in enumerate(counts):
            new[s + k] += c
        counts = new
    total, acc, cdf = 2 ** n, 0, []
    for c in counts:
        acc += c
        cdf.append(acc / total)
    return cdf


CDF = wilcoxon_null(N)


def p_less(d):
    """One-sided p (median < 0): P(W+ <= observed W+) under the null."""
    order = sorted(range(N), key=lambda i: abs(d[i]))
    wplus = sum(r + 1 for r, i in enumerate(order) if d[i] > 0)
    return CDF[wplus]


def digits_spread():
    cells = {}
    for run in DIGITS_RUNS:
        with open(os.path.join(ROOT, "data", "explore", run, "games.jsonl"), encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("condition") in ("D2", "D8") and r.get("gap") is not None:
                    cells.setdefault((run, r["fixture"], r["role"], r["condition"]), []).append(r["gap"])
    within, hsd = [], []
    for run in DIGITS_RUNS:
        hf = []
        for f in sorted({k[1] for k in cells if k[0] == run}):
            parts = []
            for role in ("seller", "buyer"):
                lo, hi = cells[(run, f, role, "D2")], cells[(run, f, role, "D8")]
                within += [statistics.stdev(lo), statistics.stdev(hi)]
                parts.append(statistics.fmean(hi) - statistics.fmean(lo))
            hf.append(statistics.fmean(parts))
        hsd.append(statistics.stdev(hf))
    k = min(len(v) for v in cells.values())
    return statistics.fmean(within), statistics.fmean(hsd), k


def s2_spread(run):
    games, _ = A.load_games(os.path.join(ROOT, "data", "runs", run))
    by = {(r["fixture"], r["opener"], r["scale"]): r for r in games.values()}

    def gap(r):
        if not r or not r["offers"] or r["offers"][0][1] != r["opener"]:
            return None
        p = r["offers"][0][2]
        return p / r["F"] - 1 if r["opener"] == "seller" else 1 - p / r["C"]
    hf = []
    for f in sorted({r["fixture"] for r in games.values()}):
        parts = [gap(by.get((f, o, A.HIGH))) - gap(by.get((f, o, A.LOW))) for o in ("seller", "buyer")
                 if gap(by.get((f, o, A.HIGH))) is not None and gap(by.get((f, o, A.LOW))) is not None]
        if parts:
            hf.append(statistics.fmean(parts))
    return statistics.stdev(hf), statistics.fmean(hf)


def power(h, sd_true, sd_within, k, rng):
    noise = sd_within / math.sqrt(k)
    hits = 0
    for _ in range(SIMS):
        d = [rng.gauss(-h, sd_true) + rng.gauss(0.0, noise) for _ in range(N)]
        hits += p_less(d) < ALPHA
    return hits / SIMS


def main():
    rng = random.Random(SEED)
    print(f"power_addendum_b.py  n={N} fixtures  {SIMS} sims/point  seed={SEED}  alpha={ALPHA:.4f} (0.05/3)  "
          f"python {sys.version.split()[0]}")
    w, hsd4, k = digits_spread()
    true_g = math.sqrt(max(0.0, hsd4 ** 2 - w ** 2 / k))
    print("\nInputs (from the data):")
    print(f"  Gemma 31B digits D2/D8, k={k}: within-cell SD per sample {w:.4f}; SD of h_f {hsd4:.4f}"
          f" -> fixture-level SD {true_g:.4f}")
    s2 = {}
    for run in S2_RUNS:
        sd, mean = s2_spread(run)
        s2[run] = sd
        print(f"  {run} (k=1): SD of h_f {sd:.4f}, mean h {mean:+.4f}")
    g_s2 = s2[S2_RUNS[0]]
    w_g_s2 = math.sqrt(max(0.0, g_s2 ** 2 - true_g ** 2))
    q = s2[S2_RUNS[1]]
    # Qwen assumption: same fixture/noise ratio as Gemma Study 2.
    ratio = true_g / g_s2
    true_q, w_q = q * ratio, q * math.sqrt(1 - ratio ** 2)
    scen = [
        ("Gemma (digits probe)", true_g, w),
        ("Gemma (Study 2, k=1 split)", true_g, w_g_s2),
        ("Qwen (Study 2; split assumed as Gemma S2)", true_q, w_q),
        ("Pessimistic (2 x Qwen, both parts)", 2 * true_q, 2 * w_q),
    ]
    print("\nScenarios (fixture-level SD, within-cell SD per sample):")
    for name, t, s in scen:
        print(f"  {name:44s} {t:.4f}  {s:.4f}")
    print(f"\nPower, one-sided Wilcoxon at alpha {ALPHA:.4f}; true mean h = -effect "
          f"(Gemma's confirmed h was -0.161, Qwen's -0.131):")
    print("  " + f"{'scenario':44s} {'k':>2s}  " + "  ".join(f"{e:>5.2f}" for e in EFFECTS))
    for name, t, s in scen:
        for k_ in KS:
            row = [power(e, t, s, k_, rng) for e in EFFECTS]
            print("  " + f"{name:44s} {k_:>2d}  " + "  ".join(f"{p:>5.2f}" for p in row))
    rng0 = random.Random(SEED + 1)
    size = sum(p_less([rng0.gauss(0, 1) for _ in range(N)]) < 0.05 for _ in range(SIMS)) / SIMS
    print(f"\ncheck: false-positive rate with no effect at alpha 0.05 = {size:.3f} (should be <= 0.05)")


if __name__ == "__main__":
    main()
