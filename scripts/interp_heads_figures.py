#!/usr/bin/env python3
"""interp_heads_figures.py (LAPTOP-SIDE): the heads session's card for the research website (X57). The plug-in
scripts/interp/mi_figures_heads.py is a shim importing cards() from here (keeps the pod's create request small).

    python3 scripts/interp_heads_figures.py --refresh data/interp/<heads run>   run -> config/interp/mi_figures_heads.json
    python3 scripts/interp_heads_figures.py --preview F                         standalone HTML with the card

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
JSON_PATH = os.path.join(ROOT, "config", "interp", "mi_figures_heads.json")
SLATE, ORANGE, SAGE, ONYX, IVORY, GREY = "#2C3E50", "#F39C12", "#87A96B", "#0F0F0F", "#FFFFF0", "#8A8F94"
DIRS = ("x1->x1e6", "x1e6->x1")              # x1 values into the x1e6 run; x1e6 values into the x1 run
CIRCUIT = (("top1", "top 1 head"), ("top4", "top 4 heads"), ("top8", "top 8 heads"), ("top8 reply", "top 8, reply only"),
           ("top8 prompt", "top 8, prompt only"), ("random8", "8 random heads"), ("other heads", "the other 216 heads"),
           ("band attn", "all attention, 38–44"))
LATCH = (("steer", "steer at layer 20"), ("ko band", "block digit reads, 38–44"), ("steer+ko band", "both"))
HEAT_MAX = 0.10                               # a head's share at which the colour saturates


def _cell(c):
    return None if not c or c.get("share") is None else {"share": c["share"], "ci": c.get("ci"), "moved": c.get("moved"), "n": c.get("n")}


def refresh(run_dir, story_row="X57"):
    s = json.load(open(os.path.join(run_dir, "summary.json")))
    L, nH = s["layers"]["heads"], s["n_heads"]
    S = s["sweep_all"]
    heads = {f"{l}.{h}": [None if S.get(d, {}).get(f"h{l}.{h}", {}).get("share") is None else round(S[d][f"h{l}.{h}"]["share"], 5)
                          for d in DIRS] for l in L for h in range(nH)}
    out = {"run": os.path.basename(run_dir.rstrip("/")), "story_row": story_row, "model": "gemma-4-31b-it", "role": "seller",
           "head_layers": L, "n_heads": nH, "heads": heads, "top": s["pick"]["top"], "random": s["pick"]["random"],
           "n_discovery": len(s["discovery"]), "n_held_out": len(s["held_out"]),
           "circuit": {d: {k: _cell(s["circuit"].get(d, {}).get(k)) for k, _ in CIRCUIT + LATCH} for d in DIRS},
           "interaction_band": {d: _cell(s["interactions"].get(d, {}).get("band")) for d in DIRS},
           "attention_vs_patch_spearman": s.get("attention_vs_patch_spearman")}
    with open(JSON_PATH, "w") as fh:
        json.dump(out, fh, indent=1)
    return out


def _bars(body, D, items, x0, y_top, row_h, scale_w, title, _txt, _line, show_dirs=DIRS):
    """Horizontal paired bars (one per direction) with CIs; x axis 0-100% of the gap."""
    sx = lambda v: x0 + max(-0.2, min(1.1, v)) / 1.0 * scale_w      # noqa: E731
    body.append(_txt(x0 - 210, y_top - 26, title, "start", "panel"))
    n = len(items)
    for t in (0, 0.5, 1.0):
        body.append(_line(sx(t), y_top - 8, sx(t), y_top + n * row_h, SLATE, 2 if t in (0, 1.0) else 1, op=.5 if t in (0, 1.0) else .15))
        body.append(_txt(sx(t), y_top - 12, f"{t * 100:.0f}%", "middle", "axis"))
    for i, (key, lab) in enumerate(items):
        y = y_top + i * row_h
        body.append(_txt(x0 - 10, y + row_h * 0.5 + 4, lab, "end"))
        for j, d in enumerate(show_dirs):
            c = D["circuit"][d].get(key)
            if c is None:
                continue
            k = len(show_dirs)
            yb, hb = y + 4 + j * (row_h - 8) / k, (row_h - 8) / k - (2 if k > 1 else 0)
            v = c["share"]
            xa, xb = sorted((sx(0), sx(v)))
            col = ONYX if j == 0 else ORANGE
            body.append(f'<rect x="{xa:.1f}" y="{yb:.1f}" width="{max(1.0, xb - xa):.1f}" height="{hb:.1f}" fill="{col}" fill-opacity="{0.85 if j == 0 else 0.9}"/>')
            if c.get("ci"):
                body.append(_line(sx(c["ci"][0]), yb + hb / 2, sx(c["ci"][1]), yb + hb / 2, SLATE if j else IVORY, 2, op=.9))
            body.append(_txt(max(xb, sx(c["ci"][1]) if c.get("ci") else xb) + 6, yb + hb - 1, f"{v * 100:.0f}%", "start", "note"))


def fig(D):
    sys.path.insert(0, HERE)
    from mi_figures import _svg, _txt, _line
    W, H = 1000, 560
    L, nH = D["head_layers"], D["n_heads"]
    top = {tuple(x) for x in D["top"]}
    body = [_txt(30, 26, "A. Every head at layers 38–44, patched alone: share of the gap restored", "start", "panel")]
    cw, ch, gx, gy = 13.0, 30.0, 92, 50
    for i, l in enumerate(L):
        y = gy + i * ch
        body.append(_txt(gx - 10, y + ch / 2 + 5, f"layer {l}", "end", "axis"))
        for h in range(nH):
            v = D["heads"][f"{l}.{h}"]
            m = None if None in v else sum(v) / 2
            x = gx + h * cw
            if m is None:
                fill, op = GREY, 0.2
            elif m >= 0:
                fill, op = ORANGE, max(0.06, min(1.0, m / HEAT_MAX))
            else:
                fill, op = SLATE, max(0.06, min(1.0, -m / HEAT_MAX))
            stroke = f' stroke="{ONYX}" stroke-width="2.5"' if (l, h) in top else ""
            lab = "" if m is None else f"{m * 100:.1f}%"
            body.append(f'<rect x="{x + 1:.1f}" y="{y + 1:.1f}" width="{cw - 2:.1f}" height="{ch - 2:.1f}" fill="{fill}" '
                        f'fill-opacity="{op:.2f}"{stroke}><title>L{l}.{h}: {lab}</title></rect>')
    for h in range(0, nH, 8):
        body.append(_txt(gx + h * cw + cw / 2, gy + len(L) * ch + 18, str(h), "middle", "axis"))
    body.append(_txt(gx + nH * cw / 2, gy + len(L) * ch + 38, "head (0–31)", "middle", "axis"))
    lx, ly = gx, gy + len(L) * ch + 62
    for k, v in enumerate((0.02, 0.05, 0.10)):
        body.append(f'<rect x="{lx + k * 92:.1f}" y="{ly - 11:.1f}" width="14" height="14" fill="{ORANGE}" fill-opacity="{v / HEAT_MAX:.2f}"/>')
        body.append(_txt(lx + k * 92 + 20, ly + 1, f"{v * 100:.0f}%" + (" or more" if v == HEAT_MAX else ""), "start", "note"))
    body.append(f'<rect x="{lx + 300:.1f}" y="{ly - 11:.1f}" width="14" height="14" fill="none" stroke="{ONYX}" stroke-width="2.5"/>')
    body.append(_txt(lx + 320, ly + 1, f"the top 8, picked on {D['n_discovery']} fixtures", "start", "note"))
    # B: circuit on held-out fixtures
    bx, by = 690, 70
    _bars(body, D, CIRCUIT, bx, by, 34, 230, f"B. Held-out fixtures ({D['n_held_out']})", _txt, _line)
    # C: the latch, x1e6 run only (the direction where early steering fails)
    cy = by + len(CIRCUIT) * 34 + 70
    _bars(body, D, LATCH, bx, cy, 34, 230, "C. The latch, in the ×10⁶ run", _txt, _line, show_dirs=(DIRS[0],))
    ib = D["interaction_band"][DIRS[0]]
    if ib:
        body.append(_txt(bx - 210, cy + len(LATCH) * 34 + 22, f"interaction {ib['share'] * 100:+.0f}% "
                     + (f"[{ib['ci'][0] * 100:+.0f}, {ib['ci'][1] * 100:+.0f}]" if ib.get("ci") else ""), "start", "note"))
    kx, ky = 40, gy + len(L) * ch + 130
    body += [_txt(kx, ky - 26, "Bars in B and C (95% CI over fixtures):", "start", "note"),
             f'<rect x="{kx}" y="{ky - 11}" width="14" height="12" fill="{ONYX}" fill-opacity=".85"/>',
             _txt(kx + 20, ky, "×1 values in the ×10⁶ run (makes it more like ×1)", "start", "note"),
             f'<rect x="{kx}" y="{ky + 13}" width="14" height="12" fill="{ORANGE}" fill-opacity=".9"/>',
             _txt(kx + 20, ky + 24, "×10⁶ values in the ×1 run (makes it more like ×10⁶)", "start", "note"),
             _txt(kx, ky + 56, "C is the ×10⁶ run only: there, steering at layer 20 alone fails.", "start", "note")]
    return _svg(W, H, "Which attention heads carry it",
                "Panel A: a grid of 7 layers by 32 heads coloured by the share of the seller's x1/x1e6 opening gap that "
                "patching that head alone restores; a few heads at layers 39-43, most at layer 41, stand out, and the "
                "top 8 are outlined. Panel B: on held-out fixtures, the top 8 heads together restore 40% and 68% of the "
                "gap in the two directions, while 8 random heads restore 1-2%. Panel C: in the x1e6 run, steering at "
                "layer 20 alone moves 5%, blocking the band's attention to the limit's digits 15%, and both 51%.", body)


def caption(D):
    sp = D.get("attention_vs_patch_spearman")
    return ("Gemma 4 31B, sellers. The opening is forced as in E2; the readout is the log-probability difference "
            "between the ×1 run's and the ×10⁶ run's greedy prices at the first token where they differ. A: each head's "
            "output, on every token after the limit, replaced by the other run's; colour = the mean share of the gap "
            "restored over both directions and all 30 fixtures; outlined = the 8 heads picked on 10 discovery fixtures. "
            f"B: those 8 heads tested on the other {D['n_held_out']} fixtures, with 95% CIs over fixtures; the ranking of "
            "all 224 heads replicates between the two sets (Spearman 0.85). C: the latch. Steering the ×10⁶ run at "
            "layer 20 barely works because heads at 38–44 re-read the limit's digits; blocking those reads lets it "
            "work, with the price itself intact. Attention to the digits does not predict a head's importance "
            + (f"(Spearman {sp:+.2f})." if sp is not None else ".") + " Exploratory; one model and role (X57).")


def cards(json_path=JSON_PATH):
    if not os.path.exists(json_path):
        return []
    D = json.load(open(json_path))
    title = "Which attention heads carry it"
    return [f'<figure class="chart-card mi-card mi-wide"><div class="chart-title"><h3>{title}</h3>'
            f'<span>heads · {D["story_row"]}</span></div><div class="chart-scroll" role="region" tabindex="0" '
            f'aria-label="{html.escape(title)} chart, scroll horizontally on small screens">{fig(D)}</div>'
            f"<figcaption>{caption(D)}</figcaption></figure>"]


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", metavar="RUN_DIR")
    ap.add_argument("--story-row", default="X57")
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
    sys.exit(main(sys.argv[1:]))
