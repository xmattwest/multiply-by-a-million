#!/usr/bin/env python3
"""show_game.py: print negotiations from a run as readable transcripts.

    python3 scripts/show_game.py data/runs/<run_id> F19-x1e8-B      one game
    python3 scripts/show_game.py data/runs/<run_id> F19             F19 at all scales, both openers
    python3 scripts/show_game.py data/runs/<run_id> F19 --prompt    also the full prompt of move 1

Reads only games.jsonl and calls.jsonl (the scored attempt of each game).
Shows each side's private limit, every move (action, price, message), how
the game ended and the seller's share. Read-only; standard library only.
"""
import json
import os
import sys


def money(x):
    return "—" if x is None else f"${x:,}"


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    run, key, show_prompt = argv[0], argv[1], "--prompt" in argv
    games = [json.loads(l) for l in open(os.path.join(run, "games.jsonl")) if l.strip()]
    games = [g for g in games if g["terminal"] != "INFRASTRUCTURE_VOID"]
    calls = {}
    for l in open(os.path.join(run, "calls.jsonl")):
        c = json.loads(l)
        calls.setdefault(c["call_id"], {})[c["schema"]] = c
    pick = [g for g in games if g["game_id"] == key or g["fixture"] == key]
    order = {"x1e2": 0, "x1e5": 1, "x1e8": 2}
    pick.sort(key=lambda g: (g["opener"], order.get(g["scale"], 9)))
    if not pick:
        print(f"no game or fixture {key!r} in {run}")
        return 1
    for g in pick:
        print("=" * 78)
        print(f"{g['game_id']}   scale V = {money(g['V'])}   opener: {g['opener']}")
        print(f"  seller's floor F = {money(g['F'])} (private)    buyer's ceiling C = {money(g['C'])} (private)")
        print(f"  bargaining range C − F = {money(g['C'] - g['F'])}")
        print("-" * 78)
        for n, cid in enumerate(g["call_ids"], 1):
            req, res = calls[cid]["call-request/1"], calls[cid]["call-result/1"]
            if show_prompt and n == 1:
                for m in req["request"]["messages"]:
                    print(f"  [{m['role']} prompt]\n" + "\n".join("    " + x for x in m["content"].splitlines()))
                print("-" * 78)
            content = json.loads(res["response_text"])["choices"][0]["message"]["content"]
            try:
                mv = json.loads(content)
                price = mv.get("price")
                rel = f"  ({price / g['V']:.3f} V)" if isinstance(price, int) else ""
                print(f"  {n:>2}. {req['role'].upper():<6} {mv.get('action'):<8} {money(price)}{rel}")
                print("      " + mv.get("message", ""))
            except ValueError:
                print(f"  {n:>2}. {req['role'].upper():<6} (not JSON) {content[:200]!r}")
        s = g.get("seller_share")
        print("-" * 78)
        print(f"  END: {g['terminal']} at {money(g['price'])} by {g['ended_by']}"
              + (f"   seller share {s['float']:.3f}" if s else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
