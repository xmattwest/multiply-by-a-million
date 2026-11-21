#!/usr/bin/env python3
"""Concession-path analysis for the scale-invariance negotiation runs (WP1).

Question (plan WP1): after the opening, do the concessions scale with x10^6?
This probes E7 / HD -- how much of the x10^6 split shift lives in the
*process* (the back-and-forth) rather than in the first offer.

Everything here is EXPLORATORY. The script is standard-library only and
reads the committed run folders; its numbers are reproducible from
`games.jsonl` alone.

Per game, measured in units of the deal-zone width W = C - F (so the two
scales are directly comparable):

  * opening gap, each side: how far the side's *first* offer sits from its
    own limit, as a fraction of W. Seller gap = (first_seller_offer - C)/W
    is positive when the seller opens above the buyer's ceiling (ambitious);
    buyer gap = (F - first_buyer_offer)/W positive when ambitious. We report
    the signed "ambition" = offer beyond the own limit toward the far end.
  * concession per round, each side: mean step the side takes toward the
    other across its own consecutive offers, as a fraction of W.
  * total concession, each side: (first offer -> last offer) distance / W.
  * rounds: number of model offers (per side and total).
  * last_gap: |last seller offer - last buyer offer| / W just before the
    close.
  * midpoint_dev: (price - midpoint_of_the_two_first_offers) / W. ~0 means
    the deal landed on the midpoint of the openings.
  * who accepts (ended_by), terminal, seller_share.

Two record schemas are handled:
  * game/1    (Study 2): offers are [turn, side, price]; self-play only.
  * newgame/1 (X-probes): offers are [turn, side, price, source] with
    source in {model, bot}; carries cell/geom/notation/info/styles/models.

Cells are grouped so x1 vs x1e6 can be paired by fixture. For self-play
cells both sides are the model. For BOT cells only the model side's
concessions are summarised (the bot is scale-proof by construction), but
we keep the model's role.

Usage:
  python3 scripts/analyze_concessions.py RUN_DIR [RUN_DIR ...]
      [--json OUT.json] [--only-cell CELL] [--quiet]

With no --json, writes <run>/concessions.json next to each run's derived
summary is NOT touched; this is a standalone read-only analysis.
"""

import json
import math
import os
import sys


# ----------------------------------------------------------------------------
# small stdlib stats helpers (no numpy)
# ----------------------------------------------------------------------------

def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def median(xs):
    xs = sorted(x for x in xs if x is not None)
    n = len(xs)
    if n == 0:
        return None
    if n % 2:
        return xs[n // 2]
    return 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def stdev(xs):
    xs = [x for x in xs if x is not None]
    n = len(xs)
    if n < 2:
        return None
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def fmt(x, nd=3):
    if x is None:
        return "   n/a"
    return f"{x:+.{nd}f}" if x < 0 else f" {x:.{nd}f}"


# ----------------------------------------------------------------------------
# per-game feature extraction
# ----------------------------------------------------------------------------

def _offer_price(off):
    # offers: [turn, side, price] (game/1) or [turn, side, price, source]
    return off[2]


def _offer_side(off):
    return off[1]


def _offer_is_model(off):
    # game/1 has no source field -> both sides are the model (self-play)
    if len(off) < 4:
        return True
    return off[3] == "model"


def game_features(d):
    """Return a dict of per-game concession features, or None if unusable."""
    F = d.get("F")
    C = d.get("C")
    if F is None or C is None:
        return None
    W = C - F
    if W <= 0:
        return None
    offers = d.get("offers") or []
    if not offers:
        return None

    seller_offers = [(_offer_price(o), _offer_is_model(o))
                     for o in offers if _offer_side(o) == "seller"]
    buyer_offers = [(_offer_price(o), _offer_is_model(o))
                    for o in offers if _offer_side(o) == "buyer"]

    feat = {
        "game_id": d.get("game_id"),
        "fixture": d.get("fixture"),
        "scale": d.get("scale"),
        "mult": d.get("mult"),
        "opener": d.get("opener"),
        "terminal": d.get("terminal"),
        "ended_by": d.get("ended_by") or "none",   # None when the game hit the message cap
        "n_messages": d.get("n_messages"),
        "n_seller_offers": len(seller_offers),
        "n_buyer_offers": len(buyer_offers),
        "n_offers": len(offers),
    }
    ss = d.get("seller_share")
    feat["seller_share"] = ss.get("float") if isinstance(ss, dict) else ss

    # Opening ambition, in zone units (positive = ambitious, past own limit
    # toward the far end). Seller wants a high price (limit F = its floor);
    # ambition = (first ask - C)/W. Buyer wants a low price (limit C=ceiling,
    # F = its best); buyer ambition = (F - first bid)/W.
    first_seller = seller_offers[0][0] if seller_offers else None
    first_buyer = buyer_offers[0][0] if buyer_offers else None
    feat["seller_open_ambition"] = (first_seller - C) / W if first_seller is not None else None
    feat["buyer_open_ambition"] = (F - first_buyer) / W if first_buyer is not None else None

    # Concessions: steps toward the counterpart, per side, in zone units.
    # Seller concedes by lowering its ask; buyer concedes by raising its bid.
    def concessions(side_offers, is_seller):
        prices = [p for p, _ in side_offers]
        if len(prices) < 2:
            return None, None, None  # per-round mean, total, n_steps
        steps = []
        for a, b in zip(prices, prices[1:]):
            step = (a - b) if is_seller else (b - a)  # positive = concession
            steps.append(step / W)
        total = (prices[0] - prices[-1]) / W if is_seller else (prices[-1] - prices[0]) / W
        return mean(steps), total, len(steps)

    s_per, s_tot, s_steps = concessions(seller_offers, True)
    b_per, b_tot, b_steps = concessions(buyer_offers, False)
    feat["seller_concession_per_round"] = s_per
    feat["seller_concession_total"] = s_tot
    feat["seller_concession_steps"] = s_steps
    feat["buyer_concession_per_round"] = b_per
    feat["buyer_concession_total"] = b_tot
    feat["buyer_concession_steps"] = b_steps

    # Last gap just before the close: distance between the two sides' final
    # standing offers, in zone units.
    last_seller = seller_offers[-1][0] if seller_offers else None
    last_buyer = buyer_offers[-1][0] if buyer_offers else None
    if last_seller is not None and last_buyer is not None:
        feat["last_gap"] = abs(last_seller - last_buyer) / W
    else:
        feat["last_gap"] = None

    # Did the deal land on the midpoint of the two first offers?
    price = d.get("price")
    if (price is not None and first_seller is not None
            and first_buyer is not None):
        mid = 0.5 * (first_seller + first_buyer)
        feat["midpoint_dev"] = (price - mid) / W
        feat["opening_mid_share"] = (mid - F) / W  # where the opening-midpoint sits
    else:
        feat["midpoint_dev"] = None
        feat["opening_mid_share"] = None

    return feat


# ----------------------------------------------------------------------------
# cell labelling (so x1 and x1e6 pair up by fixture within a cell)
# ----------------------------------------------------------------------------

def cell_key(d):
    """A label that identifies a comparable cell across scales.

    Does NOT include scale/mult, so the x1 and x1e6 games of one fixture in
    one cell share a key (plus fixture + opener)."""
    schema = d.get("schema")
    if schema == "game/1":
        return "S2-selfplay"
    parts = []
    cell = d.get("cell")
    parts.append(cell or "?")
    geom = d.get("geom")
    if geom:
        parts.append(geom)
    notation = d.get("notation")
    if notation:
        parts.append(notation)
    info = d.get("info")
    if isinstance(info, dict):
        parts.append("info=" + "/".join(f"{k[0]}:{v}" for k, v in sorted(info.items())))
    styles = d.get("styles")
    if isinstance(styles, dict):
        parts.append("sty=" + "/".join(f"{k[0]}:{v}" for k, v in sorted(styles.items())))
    if cell == "BOT":
        parts.append("mrole=" + str(d.get("model_role")))
    models = d.get("models")
    if isinstance(models, dict):
        parts.append("pair=" + str(models.get('seller')) + ">" + str(models.get('buyer')))
    return " | ".join(parts)


def scale_bucket(d):
    """Map a record's scale to 'lo' (baseline) or 'hi' (x10^6)."""
    scale = d.get("scale")
    mult = d.get("mult")
    if mult == 1:
        return "lo"
    if mult == 1000000:
        return "hi"
    # Study 2: x1e2 is lo, x1e8 is hi, x1e5 is the mid point (dropped here).
    if scale == "x1e2":
        return "lo"
    if scale == "x1e8":
        return "hi"
    return None  # x1e5 or anything else -> excluded from the paired contrast


# ----------------------------------------------------------------------------
# aggregation
# ----------------------------------------------------------------------------

PAIRED_FIELDS = [
    "seller_share",
    "seller_open_ambition", "buyer_open_ambition",
    "seller_concession_per_round", "buyer_concession_per_round",
    "seller_concession_total", "buyer_concession_total",
    "seller_concession_steps", "buyer_concession_steps",
    "n_seller_offers", "n_buyer_offers",
    "last_gap", "midpoint_dev",
]


def analyze_run(run_dir, only_cell=None):
    games_path = os.path.join(run_dir, "games.jsonl")
    if not os.path.exists(games_path):
        raise FileNotFoundError(games_path)

    # cells[cellkey]['lo'|'hi'][(fixture,opener)] = features
    cells = {}
    n_read = 0
    for line in open(games_path):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        n_read += 1
        ck = cell_key(d)
        if only_cell and only_cell not in ck:
            continue
        sb = scale_bucket(d)
        if sb is None:
            continue
        feat = game_features(d)
        if feat is None:
            continue
        pairkey = (feat.get("fixture"), feat.get("opener"))
        cells.setdefault(ck, {}).setdefault(sb, {})[pairkey] = feat

    out_cells = []
    for ck in sorted(cells):
        lo = cells[ck].get("lo", {})
        hi = cells[ck].get("hi", {})
        shared = sorted(set(lo) & set(hi), key=lambda k: (str(k[0]), str(k[1])))
        row = {"cell": ck, "n_lo": len(lo), "n_hi": len(hi), "n_paired": len(shared)}
        # cell-level means at each scale
        for sb, bucket in (("lo", lo), ("hi", hi)):
            for f in PAIRED_FIELDS:
                row[f"{f}__{sb}_mean"] = mean([bucket[k].get(f) for k in bucket])
        # paired deltas (hi - lo) over shared fixtures
        for f in PAIRED_FIELDS:
            deltas = []
            for k in shared:
                a = hi[k].get(f)
                b = lo[k].get(f)
                if a is not None and b is not None:
                    deltas.append(a - b)
            row[f"{f}__delta_mean"] = mean(deltas)
            row[f"{f}__delta_sd"] = stdev(deltas)
            row[f"{f}__delta_n"] = len(deltas)
        # acceptance / terminal breakdown
        for sb, bucket in (("lo", lo), ("hi", hi)):
            term = {}
            acc = {}
            for k in bucket:
                term[bucket[k]["terminal"]] = term.get(bucket[k]["terminal"], 0) + 1
                acc[bucket[k]["ended_by"]] = acc.get(bucket[k]["ended_by"], 0) + 1
            row[f"terminal__{sb}"] = term
            row[f"ended_by__{sb}"] = acc
        out_cells.append(row)

    return {"run_dir": run_dir, "n_games_read": n_read, "cells": out_cells}


# ----------------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------------

def print_report(result):
    print(f"\n### {os.path.basename(result['run_dir'].rstrip('/'))}"
          f"  ({result['n_games_read']} games)")
    header = ("cell", "pair", "d(share)", "d(sAmb)", "d(bAmb)",
              "d(sConc/r)", "d(bConc/r)", "d(lastgap)", "d(midDev)")
    print("  " + " ".join(f"{h:>12}" if i else f"{h:<34}"
                          for i, h in enumerate(header)))
    for c in result["cells"]:
        if c["n_paired"] == 0:
            continue
        vals = [
            c["cell"][:34],
            str(c["n_paired"]),
            fmt(c["seller_share__delta_mean"]),
            fmt(c["seller_open_ambition__delta_mean"]),
            fmt(c["buyer_open_ambition__delta_mean"]),
            fmt(c["seller_concession_per_round__delta_mean"]),
            fmt(c["buyer_concession_per_round__delta_mean"]),
            fmt(c["last_gap__delta_mean"]),
            fmt(c["midpoint_dev__delta_mean"]),
        ]
        print("  " + " ".join(f"{v:>12}" if i else f"{v:<34}"
                              for i, v in enumerate(vals)))


def main(argv):
    args = argv[1:]
    out_json = None
    only_cell = None
    quiet = False
    runs = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--json":
            out_json = args[i + 1]; i += 2; continue
        if a == "--only-cell":
            only_cell = args[i + 1]; i += 2; continue
        if a == "--quiet":
            quiet = True; i += 1; continue
        runs.append(a); i += 1
    if not runs:
        print(__doc__)
        return 2

    all_results = []
    for run_dir in runs:
        res = analyze_run(run_dir, only_cell=only_cell)
        all_results.append(res)
        # write a per-run json next to the run unless a single --json is set
        if out_json is None:
            dest = os.path.join(run_dir, "concessions.json")
            with open(dest, "w") as fh:
                json.dump(res, fh, indent=1, sort_keys=True)
        if not quiet:
            print_report(res)

    if out_json is not None:
        with open(out_json, "w") as fh:
            json.dump(all_results, fh, indent=1, sort_keys=True)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
