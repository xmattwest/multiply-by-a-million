#!/usr/bin/env python3
"""interp_heads2_figures.py (LAPTOP-SIDE): the heads2 session's card for the research website (X60). The plug-in
scripts/interp/mi_figures_heads2.py is a shim importing cards() from here (keeps the pod's create request small).

    python3 scripts/interp_heads2_figures.py --refresh data/interp/<heads2 run>   run -> config/interp/mi_figures_heads2.json
    python3 scripts/interp_heads2_figures.py --preview F                          standalone HTML with the card

cards() reads only the tracked JSON, so the site builds from the repository; no JSON yet -> no card.
Standard library only. Exploratory.
"""
import argparse
import html
import json
import os
import sys

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "interp")      # scripts/interp (mi_figures.py)
ROOT = os.path.dirname(os.path.dirname(HERE))
JSON_PATH = os.path.join(ROOT, "config", "interp", "mi_figures_heads2.json")
SLATE, ORANGE, SAGE, ONYX, IVORY, GREY = "#2C3E50", "#F39C12", "#87A96B", "#0F0F0F", "#FFFFF0", "#8A8F94"
DIRS = ("x1->x1e6", "x1e6->x1")
REGIONS = (("before", "text before the limit", GREY), ("digits", "the limit's digits", ORANGE),
           ("after", "“ dollars.”", SLATE), ("rest", "rest of the prompt", SAGE), ("reply", "reply tokens", ONYX))
PATH = (("direct", "the heads' whole effect"), ("S feed reply", "its digits + the text after"),
        ("S feed post", "its digits alone"), ("T feed after", "its “ dollars.” alone"),
        ("T feed post", "its text after the digits"), ("T feed attn", "its 5 most-read tokens"),
        ("T feed rand_a", "5 random tokens (matched)"), ("T feed reply", "the reply's tokens"))
BUYERS = (("top8", "buyers' own top 8"), ("seller8", "sellers' top 8"), ("shared2", "the 2 in both"),
          ("seller-only6", "other 6 seller heads"), ("random8", "8 random heads"))


def _cell(c):
    return None if not c or c.get("share") is None else {"share": c["share"], "ci": c.get("ci"), "n": c.get("n")}


def refresh(run_dir, story_row="X60"):
    s = json.load(open(os.path.join(run_dir, "summary.json")))
    Sx, P, B = s["sources"], s["path"]["shares"], s["buyers"]
    heads = [[k, {g: c["diff"][g]["share"] for g, _l, _c in REGIONS}, c.get("x57_score")] for k, c in Sx["heads"].items()]
    out = {"run": os.path.basename(run_dir.rstrip("/")), "story_row": story_row, "model": "gemma-4-31b-it",
           "recv": s["geometry"]["recv"], "sources": heads, "n_sources": Sx["n_fixtures"],
           "path": {d: {k: _cell(P.get(d, {}).get(k)) for k, _ in PATH} for d in DIRS},
           "n_path": len(s["build"]["seller"]),
           "buyers": {d: {k: _cell(B["circuit"].get(d, {}).get(k)) for k, _ in BUYERS} for d in DIRS},
           "buyer_top": B["pick"]["top"], "n_buyer_disc": len(set(s["discovery"]) - set(s["checks"]["skipped"]["buyer"])), "n_buyer_held": max((c.get("n") or 0) for d in DIRS for c in B["circuit"].get(d, {}).values()),
           "buyer_vs_seller_spearman": B.get("vs_seller_spearman"), "buyer_replication": B.get("replication_spearman"),
           "scan_spearman": (Sx.get("scan_vs_x57_spearman") or {}).get("d_reply")}
    with open(JSON_PATH, "w") as fh:
        json.dump(out, fh, indent=1)
    return out


def fig(D):
    sys.path.insert(0, HERE)
    from mi_figures import _svg, _txt, _line
    from interp_heads_figures import _bars
    W, H = 1000, 640
    body = [_txt(30, 26, "A. Where each head's ×10⁶ − ×1 output difference comes from", "start", "panel")]
    x0, y0, rw, sw = 130, 52, 24, 330
    for i, (k, sh, _sc) in enumerate(D["sources"]):
        y = y0 + i * rw
        lab = f"L{k}" + ("" if [int(v) for v in k.split(".")] in D["recv"] else " (contrast)")
        body.append(_txt(x0 - 10, y + rw * 0.5 + 4, lab, "end", "note"))
        x = x0
        for g, _l, col in REGIONS:
            w = max(0.0, sh.get(g) or 0) * sw
            if w > 0.5:
                body.append(f'<rect x="{x:.1f}" y="{y + 3:.1f}" width="{w:.1f}" height="{rw - 6:.1f}" fill="{col}" '
                            f'fill-opacity="0.85"><title>L{k} {html.escape(_l)}: {100 * (sh.get(g) or 0):.0f}%</title></rect>')
            x += w
    yl = y0 + len(D["sources"]) * rw + 18
    for j, (g, lab, col) in enumerate(REGIONS):
        xx = x0 - 60 + (j % 3) * 150
        yy = yl + (j // 3) * 18
        body.append(f'<rect x="{xx:.1f}" y="{yy - 10:.1f}" width="12" height="12" fill="{col}" fill-opacity=".85"/>')
        body.append(_txt(xx + 17, yy + 1, lab, "start", "note"))
    for t in (0, 0.5, 1.0):
        body.append(_line(x0 + t * sw, y0 - 6, x0 + t * sw, y0 + len(D["sources"]) * rw, SLATE, 1, op=.25))
        body.append(_txt(x0 + t * sw, y0 - 9, f"{t * 100:.0f}%", "middle", "axis"))
    # B: path patching, share of the gap
    W2 = {"circuit": D["path"]}
    _bars(body, W2, PATH, 720, 70, 30, 230, f"B. Path patching into the top 8 ({D['n_path']} fixtures): the other run's …", _txt, _line)
    # C: buyers, held-out fixtures
    W3 = {"circuit": D["buyers"]}
    cy = 70 + len(PATH) * 30 + 72
    _bars(body, W3, BUYERS, 720, cy, 30, 230, f"C. Buyers, held-out fixtures ({D['n_buyer_held']})", _txt, _line)
    kx, ky = 40, H - 70
    body += [_txt(kx, ky - 22, "Bars in B and C (share of the gap, 95% CI over fixtures):", "start", "note"),
             f'<rect x="{kx}" y="{ky - 11}" width="14" height="12" fill="{ONYX}" fill-opacity=".85"/>',
             _txt(kx + 20, ky, "×1 values in the ×10⁶ run (makes it more like ×1)", "start", "note"),
             f'<rect x="{kx}" y="{ky + 13}" width="14" height="12" fill="{ORANGE}" fill-opacity=".9"/>',
             _txt(kx + 20, ky + 24, "×10⁶ values in the ×1 run (makes it more like ×10⁶)", "start", "note")]
    return _svg(W, H, "Where the key heads read the limit from",
                "Panel A: for each of the seller's top 8 heads and two contrast heads, a stacked bar splitting the head's "
                "x1e6 minus x1 output difference at the reply's opening tokens by source: mostly the limit's digits, then "
                "the two tokens right after them. Panel B: path patching into the top 8 heads. Making the x1e6 run look "
                "like x1 through these heads needs both the x1 digits and the x1 ' dollars.' tokens; either alone does "
                "little. The other way, either alone carries about half. Random tokens and the reply's tokens do "
                "nothing. Panel C: for buyers on held-out fixtures, the buyers' own top 8 heads carry 31% and 56%; "
                "the sellers' top 8 carry 3% and 46%.", body)


def caption(D):
    sp, bs, br = D.get("scan_spearman"), D.get("buyer_vs_seller_spearman"), D.get("buyer_replication")
    f = lambda v: "n/a" if v is None else f"{v:+.2f}"          # noqa: E731
    return ("Gemma 4 31B; the opening is forced as in E2; readout = the log-probability difference between the ×1 and "
            "×10⁶ runs' greedy prices at the deciding token. A: an exact split of each head's output difference, "
            f"averaged over the reply's 11 opening tokens and {D['n_sources']} fixtures. Text before the limit is "
            "identical in both runs, so it enters only through shifts in attention. B: the heads' inputs are taken "
            "from the other run at the chosen positions (the digits by running the other run's prompt); only the "
            "heads' outputs are then patched. Reading the digits is common: two contrast heads read them as much "
            f"but matter little, and the size of a head's output change does not predict its importance (Spearman "
            f"{f(sp)}). C: buyers' top 8 picked on {D['n_buyer_disc']} discovery fixtures (ranking replicates, Spearman {f(br)}); "
            f"buyer vs seller head maps Spearman {f(bs)}. Exploratory; one model ({D['story_row']}).")


def cards(json_path=JSON_PATH):
    if not os.path.exists(json_path):
        return []
    D = json.load(open(json_path))
    title = "Where the key heads read the limit from"
    return [f'<figure class="chart-card mi-card mi-wide"><div class="chart-title"><h3>{title}</h3>'
            f'<span>heads2 · {D["story_row"]}</span></div><div class="chart-scroll" role="region" tabindex="0" '
            f'aria-label="{html.escape(title)} chart, scroll horizontally on small screens">{fig(D)}</div>'
            f"<figcaption>{caption(D)}</figcaption></figure>"]


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", metavar="RUN_DIR")
    ap.add_argument("--story-row", default="X60")
    ap.add_argument("--preview")
    a = ap.parse_args(argv)
    if a.refresh:
        refresh(a.refresh, a.story_row)
        print(f"wrote {os.path.relpath(JSON_PATH, ROOT)}")
    if a.preview:
        from mi_figures import CSS
        with open(a.preview, "w") as fh:
            fh.write('<!doctype html><meta charset="utf-8">'
                     f"<style>{CSS}body{{background:#FFFFF0;padding:30px}}</style>{''.join(cards())}")
        print(f"wrote {a.preview}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.exit(main(sys.argv[1:]))
