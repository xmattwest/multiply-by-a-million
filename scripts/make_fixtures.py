#!/usr/bin/env python3
"""make_fixtures.py: generate the experiment's fixtures and game schedule.

WHAT IT DOES
  Writes config/fixtures.json: 30 fixtures (a seller floor factor f and a
  buyer ceiling factor c), the three scales, and the 180 games per model
  (30 fixtures x 3 scales x 2 opener orders) in a shuffled run order.

  At scale V, a fixture's floor is F = f/100 * V and its ceiling is
  C = c/100 * V, with f drawn from 75..95 and c from 105..125 (ZOPA spec
  section 2.2). The same f and c are reused at every scale: that is the
  pairing the whole experiment rests on.

WHERE THE RANDOMNESS COMES FROM
  No hidden seed. The seed is the SHA-256 of a public phrase (default
  "scale-invariance/fixtures/v1"), so anyone can regenerate the file, and
  changing the fixtures means visibly changing the phrase.

  Every random draw is derive(label, n): a whole number in 0..n-1 taken from
  SHA-256("<seed hex>|<label>|<k>"), first 12 hex digits (48 bits), with
  rejection sampling (try k = 0, 1, 2, ... until the value is below the
  largest multiple of n) so each outcome is exactly equally likely.
  Labels: "fixture/<i>/f", "fixture/<i>/c" (i = 1..30); "block/<j>" for
  shuffling the 60 blocks; "within/<b>/<j>" for the order inside block b.
  Shuffles are Fisher-Yates (j from the last index down to 1, swap with
  derive(label, j + 1)).

RUN ORDER
  Games come in 60 blocks, one per fixture and opener. A block is the same
  negotiation at all three scales, played back to back, so the three games
  that get compared run under near-identical server conditions, and the
  scales stay balanced across the whole run. Block order and the scale
  order inside each block are both random.
  tests/test_make_fixtures.sh re-derives all of it in bash with shasum.

USAGE
  python3 scripts/make_fixtures.py            write config/fixtures.json
  python3 scripts/make_fixtures.py --check    regenerate in memory; exit 1
                                              if the file on disk differs
  Options: --phrase TEXT, --out PATH

EXIT CODES  0 ok   1 --check found a difference   2 usage or local problem

Standard library only. Free, offline, deterministic.
"""

import argparse
import hashlib
import json
import os
import sys

GENERATOR_VERSION = "make_fixtures/2"
DEFAULT_PHRASE = "scale-invariance/fixtures/v1"
N_FIXTURES = 30
F_RANGE = (75, 95)    # seller floor, percent of V, inclusive
C_RANGE = (105, 125)  # buyer ceiling, percent of V, inclusive
SCALES = [            # (tag, V)
    ("x1e2", 100),
    ("x1e5", 100_000),
    ("x1e8", 100_000_000),
]
OPENERS = [("S", "seller"), ("B", "buyer")]


def seed_from_phrase(phrase):
    return hashlib.sha256(phrase.encode("utf-8")).hexdigest()


def derive(seed_hex, label, n):
    """Exactly uniform whole number in 0..n-1, from SHA-256 with rejection."""
    if n < 1 or n > 2 ** 48:
        raise ValueError("n must be between 1 and 2**48")
    limit = (2 ** 48 // n) * n
    k = 0
    while True:
        h = hashlib.sha256(f"{seed_hex}|{label}|{k}".encode("utf-8")).hexdigest()
        x = int(h[:12], 16)
        if x < limit:
            return x % n
        k += 1


def shuffle(items, seed_hex, label):
    """Fisher-Yates in place, driven by derive()."""
    for j in range(len(items) - 1, 0, -1):
        r = derive(seed_hex, f"{label}/{j}", j + 1)
        items[j], items[r] = items[r], items[j]


def build(phrase):
    seed = seed_from_phrase(phrase)
    fixtures = []
    for i in range(1, N_FIXTURES + 1):
        f = F_RANGE[0] + derive(seed, f"fixture/{i}/f", F_RANGE[1] - F_RANGE[0] + 1)
        c = C_RANGE[0] + derive(seed, f"fixture/{i}/c", C_RANGE[1] - C_RANGE[0] + 1)
        fixtures.append({"id": f"F{i:02d}", "f_pct": f, "c_pct": c})

    # One block per fixture x opener, holding that pairing's three scales.
    blocks = []
    for fx in fixtures:
        for otag, opener in OPENERS:
            block = []
            for tag, v in SCALES:
                floor = fx["f_pct"] * v // 100
                ceiling = fx["c_pct"] * v // 100
                # Integer arithmetic is exact here; check it anyway.
                assert floor * 100 == fx["f_pct"] * v and ceiling * 100 == fx["c_pct"] * v
                block.append({
                    "game_id": f"{fx['id']}-{tag}-{otag}",
                    "fixture": fx["id"],
                    "scale": tag,
                    "V": v,
                    "F": floor,
                    "C": ceiling,
                    "opener": opener,
                })
            blocks.append(block)

    # Same run order for every model: shuffle blocks, then within each block.
    shuffle(blocks, seed, "block")
    games = []
    for b, block in enumerate(blocks, start=1):
        shuffle(block, seed, f"within/{b}")
        for g in block:
            g["block"] = b
            games.append(g)
    for pos, g in enumerate(games, start=1):
        g["run_order"] = pos

    return {
        "about": ("Fixtures and run order for the scale-invariance experiment. "
                  "Generated by scripts/make_fixtures.py; regenerate with --check "
                  "to verify. Proposed in Phase 5; frozen by the pre-registration."),
        "generator_version": GENERATOR_VERSION,
        "seed_phrase": phrase,
        "seed_sha256": seed,
        "derivation": ("derive(label, n) = first 12 hex digits of "
                       "SHA256('<seed_sha256>|<label>|<k>') as an integer, k = 0,1,2,... "
                       "until below floor(2^48/n)*n; result mod n"),
        "f_pct_range": list(F_RANGE),
        "c_pct_range": list(C_RANGE),
        "scales": [{"tag": t, "V": v} for t, v in SCALES],
        "fixtures": fixtures,
        "games": games,
    }


def render(doc):
    # Fixed key order (insertion order), ASCII only, one trailing newline:
    # the same input always gives the same bytes, so the file can be hashed.
    return json.dumps(doc, indent=1, ensure_ascii=True) + "\n"


def main(argv):
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Generate config/fixtures.json")
    ap.add_argument("--phrase", default=DEFAULT_PHRASE)
    ap.add_argument("--out", default=os.path.join(here, "..", "config", "fixtures.json"))
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)

    if not args.phrase or not args.phrase.isprintable():
        print("ERROR: --phrase must be non-empty printable text", file=sys.stderr)
        return 2
    text = render(build(args.phrase))

    if args.check:
        try:
            with open(args.out, "r", encoding="utf-8", newline="") as fh:
                on_disk = fh.read()
        except OSError as e:
            print(f"ERROR: cannot read {args.out}: {e.strerror}", file=sys.stderr)
            return 2
        if on_disk == text:
            print(f"fixtures file matches a fresh regeneration (sha256 "
                  f"{hashlib.sha256(text.encode()).hexdigest()[:16]})")
            return 0
        print("DIFFERS: the fixtures file is not what the generator produces", file=sys.stderr)
        return 1

    tmp = args.out + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
        os.replace(tmp, args.out)
    except OSError as e:
        print(f"ERROR: cannot write {args.out}: {e.strerror}", file=sys.stderr)
        return 2
    print(f"wrote {N_FIXTURES} fixtures, {N_FIXTURES * len(SCALES) * len(OPENERS)} games "
          f"(sha256 {hashlib.sha256(text.encode()).hexdigest()[:16]})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
