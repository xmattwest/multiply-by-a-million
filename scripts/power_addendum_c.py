#!/usr/bin/env python3
"""power_addendum_c.py: how many games per cell does Addendum C need?

WHAT IT DOES
  Addendum C plays Study 2's full games on the three Addendum B API models.
  Its primary unit is Study 2's d_i: for each fixture, the mean over the two
  openers of [mean seller share at x10^6 - mean seller share at x1], where
  each cell mean is over the k games of that (fixture, opener, scale) cell
  that reached a deal.

  The spread of d_i has a fixture-level part that doesn't shrink with k and
  a game-level part that does (variance noise^2 / k). Study 2 ran k = 1, so
  the data give the total spread only:
    - Gemma 4 31B, Study 2: SD of d_i (k = 1)
    - Qwen3.8 27B, Study 2: SD of d_i (k = 1)
  An upper bound on the game-level part comes from the two openers' d_io in
  the same fixture: Var(d_iS - d_iB) / 2 >= the game-level variance of one
  opener's d_io (it also contains opener-by-fixture differences). Since
  d_i averages two openers, its game-level variance is at most a quarter of
  Var(d_iS - d_iB). Scenarios use that bound and two smaller shares, plus a
  pessimistic one (twice Qwen's spread), in case the API models (no seed on
  Anthropic) are noisier.

  Simulated, per scenario and k:
    - H1 power: two-sided Wilcoxon (exact no-ties null) at alpha 0.05/3, the
      strictest Holm step for three models, for true mean shifts of
      0.05-0.15 seller share.
    - H2 chance: the probability that the 90% CI of the mean d_i lies inside
      (-0.10, +0.10) when the true shift is 0 (normal-theory interval as a
      stand-in for the registered bootstrap).
    - HM1 chance (the midpoint-account equivalence, registered margin
      +-0.05): the same, for mean e_i = mean(d_i - m_i) with the spread of
      d_i - m_i observed in Study 2, when e is truly 0 and when it equals
      Qwen's Study 2 mean (-0.035).
  Normal effects; real data have ties, so read these as optimistic.

USAGE
  python3 scripts/power_addendum_c.py        prints the table; ~1 min
Deterministic (fixed seed). Standard library only. Free, offline.
"""
import math
import os
import random
import statistics
import sys
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import analyze as A               # noqa: E402

N = 30
SIMS = 4000
SEED = 20261011
ALPHA = 0.05 / 3
EFFECTS = (0.05, 0.08, 0.10, 0.15)
KS = (1, 2, 4)
Z90 = 1.6449
S2_RUNS = (("Gemma 4 31B", "s2-gemma-4-31b-it-20261004T012932Z"),
           ("Qwen3.8 27B", "s2-qwen3.8-27b-fp8-20261003T223042Z"))


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
TOTAL = N * (N + 1) // 2


def p_two_sided(d):
    order = sorted(range(N), key=lambda i: abs(d[i]))
    wplus = sum(r + 1 for r, i in enumerate(order) if d[i] > 0)
    lo = min(wplus, TOTAL - wplus)
    return min(1.0, 2 * CDF[lo])


def midpoint_share(r):
    s = [p for _, who, p in r["offers"] if who == "seller"]
    b = [p for _, who, p in r["offers"] if who == "buyer"]
    if not s or not b:
        return None
    return (Fraction(s[0] + b[0], 2) - r["F"]) / (r["C"] - r["F"])


def s2_inputs(run):
    """SD of d_i, SD of (d_iS - d_iB), SD of (d_i - m_i) from a Study 2 run (k = 1)."""
    games, _ = A.load_games(os.path.join(ROOT, "data", "runs", run))
    by = {(r["fixture"], r["opener"], r["scale"]): r for r in games.values()}
    d, opener_diff, dm = [], [], []
    for f in sorted({r["fixture"] for r in games.values()}):
        parts, mparts = {}, []
        for o in ("seller", "buyer"):
            hi, lo = by.get((f, o, A.HIGH)), by.get((f, o, A.LOW))
            if hi and lo and A.share(hi) is not None and A.share(lo) is not None:
                parts[o] = float(A.share(hi) - A.share(lo))
                mh, ml = midpoint_share(hi), midpoint_share(lo)
                if mh is not None and ml is not None:
                    mparts.append(parts[o] - float(mh - ml))
        if parts:
            d.append(statistics.fmean(parts.values()))
        if len(parts) == 2:
            opener_diff.append(parts["seller"] - parts["buyer"])
        if mparts:
            dm.append(statistics.fmean(mparts))
    ci = A.bootstrap_ci(dm, random.Random(A.seed("addendumC-power-e", run)), levels=(0.90,))["90"]
    return statistics.stdev(d), statistics.stdev(opener_diff), statistics.stdev(dm), statistics.fmean(dm), ci


def sim(mu, sd_fix, sd_game, k, rng, kind, margin=0.10):
    noise = sd_game / math.sqrt(k)
    hits = 0
    for _ in range(SIMS):
        d = [rng.gauss(mu, sd_fix) + rng.gauss(0.0, noise) for _ in range(N)]
        if kind == "H1":
            hits += p_two_sided(d) < ALPHA
        else:
            m, se = statistics.fmean(d), statistics.stdev(d) / math.sqrt(N)
            hits += (m - Z90 * se > -margin) and (m + Z90 * se < margin)
    return hits / SIMS


def main():
    rng = random.Random(SEED)
    print(f"power_addendum_c.py  n={N} fixtures  {SIMS} sims/point  seed={SEED}  alpha={ALPHA:.4f} (0.05/3)  "
          f"python {sys.version.split()[0]}")
    print("\nInputs (Study 2, k = 1):")
    inp = {}
    for name, run in S2_RUNS:
        sd_d, sd_od, sd_dm, mean_dm, ci_dm = s2_inputs(run)
        bound = min(sd_d, sd_od / 2)          # game-level SD of d_i is at most sd(d_iS - d_iB) / 2
        inp[name] = (sd_d, bound, sd_dm, mean_dm)
        print(f"  {name:12s} SD of d_i {sd_d:.4f}; SD of d_iS - d_iB {sd_od:.4f} -> game-level SD of d_i <= "
              f"{bound:.4f}; d_i - m_i: SD {sd_dm:.4f}, mean {mean_dm:+.4f}, 90% CI [{ci_dm[0]:+.3f}, {ci_dm[1]:+.3f}]")
    scen = []
    for name, (sd_d, bound, _, _) in inp.items():
        for share in (1.0, 0.5):
            g = bound * math.sqrt(share)
            scen.append((f"{name}, game part {'at bound' if share == 1.0 else 'half the bound (var)'}",
                         math.sqrt(max(0.0, sd_d ** 2 - g ** 2)), g))
    q_sd, q_b, _, q_e = inp["Qwen3.8 27B"]
    scen.append(("Pessimistic (2 x Qwen, game part at bound)", 2 * math.sqrt(max(0.0, q_sd ** 2 - q_b ** 2)),
                 2 * q_b))
    print("\nScenarios (fixture-level SD, game-level SD of d_i at k = 1):")
    for name, t, g in scen:
        print(f"  {name:48s} {t:.4f}  {g:.4f}")

    print(f"\nH1 power, two-sided Wilcoxon at alpha {ALPHA:.4f}; true |mean d| = effect "
          f"(Gemma's confirmed d was -0.256):")
    print("  " + f"{'scenario':48s} {'k':>2s}  " + "  ".join(f"{e:>5.2f}" for e in EFFECTS))
    for name, t, g in scen:
        for k in KS:
            row = [sim(-e, t, g, k, rng, "H1") for e in EFFECTS]
            print("  " + f"{name:48s} {k:>2d}  " + "  ".join(f"{p:>5.2f}" for p in row))

    print("\nH2 chance of showing equivalence within +-0.10 when the true shift is 0:")
    for name, t, g in scen:
        print("  " + f"{name:48s} " + "  ".join(f"k={k}: {sim(0.0, t, g, k, rng, 'H2'):.2f}" for k in KS))

    for margin in (0.05, 0.10):
        tag = "registered" if margin == 0.05 else "for comparison only"
        print(f"\nHM1 chance of showing mean e_i within +-{margin:.2f} ({tag}), with the Study 2 spread of")
        print("  d_i - m_i (k = 1; treated as all game-level, so k helps as much as it can; read k = 1 as the floor):")
        for true_e, elab in ((0.0, "true e 0"), (q_e, f"true e {q_e:+.3f} (Qwen)")):
            for name, (_, _, sd_dm, _) in inp.items():
                for mult, lab in ((1.0, ""), (2.0, " x 2")):
                    row = "  ".join(f"k={k}: {sim(true_e, 0.0, sd_dm * mult, k, rng, 'H2', margin):.2f}" for k in KS)
                    print("  " + f"{name + lab + ', ' + elab:48s} " + row)

    rng0 = random.Random(SEED + 1)
    size = sum(p_two_sided([rng0.gauss(0, 1) for _ in range(N)]) < 0.05 for _ in range(SIMS)) / SIMS
    print(f"\ncheck: false-positive rate with no effect at alpha 0.05 = {size:.3f} (should be <= 0.05)")


if __name__ == "__main__":
    main()
