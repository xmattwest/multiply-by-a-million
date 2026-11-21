#!/usr/bin/env python3
"""power_sim.py: how big a scale effect can 30 paired fixtures detect?

WHAT IT DOES
  Simulates the pre-registered primary test many times: 30 fixture-level
  differences d_i = (seller share at $100M) - (seller share at $100), drawn
  from a normal distribution with true mean mu and spread sigma, tested with
  a two-sided Wilcoxon signed-rank test (exact null distribution, n = 30).

  The Wilcoxon test only looks at ranks, so its power depends only on the
  ratio mu/sigma ("effect size"). The script finds the smallest ratio with
  at least 80% power, then converts it to seller-share points for a range
  of plausible spreads. It does the same for the equivalence check (can
  we conclude "no shift bigger than +/- margin" when there is no shift?).

  Assumptions (Assessment): normal differences, no ties, all 30 fixtures
  usable. Real data will have ties (round prices) and maybe fewer fixtures,
  so treat these numbers as optimistic.

USAGE
  python3 scripts/power_sim.py        prints a table; ~20 s
Deterministic (fixed seed). Standard library only.
"""

import math
import random
import statistics
import sys

N = 30
SIMS = 10000
SEED = 20261002
ALPHAS = [("alpha 0.05 (one model)", 0.05), ("alpha 0.0167 (Holm, strictest of 3)", 0.05 / 3)]
SIGMAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
MARGINS = [0.05, 0.10]


def wilcoxon_null(n):
    """Exact null distribution of W+ (sum of positive ranks), n with no ties."""
    counts = [1]
    for k in range(1, n + 1):
        new = counts + [0] * k
        for s, c in enumerate(counts):
            new[s + k] += c
        counts = new
    total = 2 ** n
    cdf, acc = [], 0
    for c in counts:
        acc += c
        cdf.append(acc / total)
    return cdf  # cdf[w] = P(W+ <= w)


CDF = wilcoxon_null(N)
WMAX = N * (N + 1) // 2


def p_two_sided(d):
    ranks = sorted(range(N), key=lambda i: abs(d[i]))
    wplus = sum(r + 1 for r, i in enumerate(ranks) if d[i] > 0)
    lo = CDF[wplus]
    hi = 1 - (CDF[wplus - 1] if wplus > 0 else 0.0)
    return min(1.0, 2 * min(lo, hi))


def power(effect, alpha, rng):
    hits = 0
    for _ in range(SIMS):
        d = [rng.gauss(effect, 1.0) for _ in range(N)]
        if p_two_sided(d) < alpha:
            hits += 1
    return hits / SIMS


def equivalence_power(sigma, margin, rng):
    """P(90% CI of the mean lies inside +/- margin) when the true shift is 0.
    Normal-approximation CI (Assessment); the real analysis uses a bootstrap."""
    hits = 0
    for _ in range(SIMS):
        d = [rng.gauss(0.0, sigma) for _ in range(N)]
        m, s = statistics.fmean(d), statistics.stdev(d)
        half = 1.645 * s / math.sqrt(N)
        if -margin < m - half and m + half < margin:
            hits += 1
    return hits / SIMS


def main():
    rng = random.Random(SEED)
    print(f"power_sim.py  n={N} fixtures  {SIMS} simulations per point  seed={SEED}  "
          f"python {sys.version.split()[0]}")
    # Sanity: with no effect, the test should reject about alpha of the time.
    size = power(0.0, 0.05, rng)
    print(f"check: false-positive rate with no effect at alpha 0.05 = {size:.3f} (should be <= 0.05)")
    print()
    print("Smallest detectable shift (80% power), in seller-share points:")
    header = "  spread of d (sigma)  " + "  ".join(f"{s:>5.2f}" for s in SIGMAS)
    for label, alpha in ALPHAS:
        effect = 0.30
        while effect < 1.5:
            p = power(effect, alpha, rng)
            if p >= 0.80:
                break
            effect = round(effect + 0.02, 2)
        print(f"  {label}: effect size {effect:.2f} sigma (power {p:.3f})")
        print(header)
        print("  smallest shift         " + "  ".join(f"{effect * s:>5.3f}" for s in SIGMAS))
    print()
    print("Equivalence (true shift 0): chance the 90% CI lands inside +/- margin:")
    print("  spread of d (sigma)  " + "  ".join(f"{s:>5.2f}" for s in SIGMAS))
    for margin in MARGINS:
        print(f"  margin {margin:.2f}            " + "  ".join(f"{equivalence_power(s, margin, rng):>5.2f}" for s in SIGMAS))


if __name__ == "__main__":
    main()
