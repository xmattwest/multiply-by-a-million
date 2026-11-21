#!/usr/bin/env python3
"""interp_synth_figures.py (LAPTOP-SIDE): the X62 synthesis card for the research website (heads2 X60 + E4 X61, with
X57's seller head map). The plug-in scripts/interp/mi_figures_synth.py is a shim importing cards() from here (keeps the
pod's create request small).

    python3 scripts/interp_synth_figures.py --refresh       the three local runs -> config/interp/mi_figures_synth.json
    python3 scripts/interp_synth_figures.py --preview F     standalone HTML with the card

Unlike the per-run cards, refresh() reads the runs' results.jsonl, not summary.json: X62's held-out numbers use only
fixtures whose prompt is new (a D2/D8 prompt depends only on the limit, so Y01-Y30 repeat prompts; see STORY X62),
one fixture per distinct prompt, and drop range-run fixtures whose readout pair has no probability on the range
prompts (p(a)+p(b) < 0.5). cards() reads only the tracked JSON, so the site builds from the repository; no JSON yet ->
no card. Standard library only. Exploratory.
"""
import argparse
import html
import json
import math
import os
import random
import statistics as st
import sys

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "interp")      # scripts/interp (mi_figures.py)
ROOT = os.path.dirname(os.path.dirname(HERE))
JSON_PATH = os.path.join(ROOT, "config", "interp", "mi_figures_synth.json")
RUNS = {"x57": "data/interp/interp-heads-gemma-4-31b-it-20261007T013856Z",
        "heads2": "data/interp/interp-heads2-gemma-4-31b-it-20261007T114921Z",
        "e4": "data/interp/interp-e4-gemma-4-31b-it-20261007T115115Z"}
SLATE, ORANGE, SAGE, ONYX, IVORY, GREY = "#2C3E50", "#F39C12", "#87A96B", "#0F0F0F", "#FFFFF0", "#8A8F94"
DIRS = ("x1->x1e6", "x1e6->x1")                  # bar order: into the plain x1e6 run first, then the reverse
CURE_DIRS = ("N6", "P6")                         # N6 <- P6 (the range run's states into plain x1e6), and back
LAYERS = list(range(36, 49))
PK_MIN = 0.5
CIRCUIT = (("scale", "sellers: X57's top 8 (the scale)"), ("cure", "sellers: the cure's top 8 (the range)"),
           ("buyers", "buyers: their own top 8"), ("random", "sellers: 8 random heads"))


# --- from records -------------------------------------------------------------------------------------------
def _load(d):
    p = d if os.path.isabs(d) else os.path.join(ROOT, d)
    with open(os.path.join(p, "results.jsonl")) as fh:
        return [json.loads(line) for line in fh]


def _pk(x):
    return math.exp(x["lpa"]) + math.exp(x["lpb"])


def ratio_ci(num, den, n=2000, seed=1):
    """Paired bootstrap over fixtures of mean(num) / mean(den) (the runs' own rule)."""
    k = len(num)
    if k < 3:
        return None
    rng, vals = random.Random(seed), []
    for _ in range(n):
        idx = [rng.randrange(k) for _ in range(k)]
        d = sum(den[i] for i in idx)
        if d:
            vals.append(sum(num[i] for i in idx) / d)
    vals.sort()
    return [vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]] if len(vals) >= 20 else None


def spearman(xs, ys):
    def rank(v):
        o, r, i = sorted(range(len(v)), key=lambda i: v[i]), [0.0] * len(v), 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            for t in range(i, j + 1):
                r[o[t]] = (i + j) / 2
            i = j + 1
        return r
    if len(xs) < 3:
        return None
    rx, ry = rank(xs), rank(ys)
    mx, my = st.mean(rx), st.mean(ry)
    sx, sy = math.sqrt(sum((a - mx) ** 2 for a in rx)), math.sqrt(sum((b - my) ** 2 for b in ry))
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / (sx * sy) if sx and sy else None


def new_prompts(limits, discovery, held, drop=()):
    """Held-out fixtures whose limit (= prompt) is not in discovery, one per limit (the first by name)."""
    seen, out = {limits[f] for f in discovery if f in limits}, []
    for f in sorted(held):
        if f in limits and f not in drop and limits[f] not in seen:
            seen.add(limits[f])
            out.append(f)
    return out


def shares(recs, keep, rows):
    """heads-style records ({direction, ld_clean, ld_source, rows}): {direction: {row: cell}} over fixtures in keep."""
    acc = {}
    for r in recs:
        if r.get("skipped") or r["fixture"] not in keep or r.get("ld_clean") is None:
            continue
        for x in r["rows"]:
            if x["n"] in rows:
                a = acc.setdefault((r["direction"], x["n"]), ([], []))
                a[0].append(x["ld"] - r["ld_clean"])
                a[1].append(r["ld_source"] - r["ld_clean"])
    return {k: _cell(*v) for k, v in acc.items()}


def cure_shares(recs, keep, rows):
    """E4 cure records ({targets: {N6, P6}}): share of the ld distance between the two runs swapped."""
    acc = {}
    for r in recs:
        if r["fixture"] not in keep:
            continue
        T = r["targets"]
        for t in CURE_DIRS:
            den = T["P6" if t == "N6" else "N6"]["ld_clean"] - T[t]["ld_clean"]
            for x in T[t]["rows"]:
                if x["n"] in rows:
                    a = acc.setdefault((t, x["n"]), ([], []))
                    a[0].append(x["ld"] - T[t]["ld_clean"])
                    a[1].append(den)
    return {k: _cell(*v) for k, v in acc.items()}


def _cell(num, den):
    D = st.mean(den)
    return {"share": st.mean(num) / D if D else None, "ci": ratio_ci(num, den), "n": len(num)}


def head_map(cells, nL_heads):
    """{"L.h": mean share of both directions} from {(dir, "hL.h"): cell}."""
    out = {}
    for name in nL_heads:
        v = [cells.get((d, "h" + name), {}).get("share") for d in sorted({d for d, _n in cells})]
        if v and None not in v:
            out[name] = sum(v) / len(v)
    return out


def unreadable(e4_fixtures):
    """Range-run fixtures whose plain readout pair has p(a)+p(b) < PK_MIN on either range prompt."""
    return sorted(r["fixture"] for r in e4_fixtures
                  if min(math.exp(r["lpa"][k]) + math.exp(r["lpb"][k]) for k in ("P1", "P6")) < PK_MIN)


def refresh(runs=RUNS, story_row="X62", out_path=None):
    R57, R2, R4 = _load(runs["x57"]), _load(runs["heads2"]), _load(runs["e4"])
    step = lambda R, s: [r for r in R if r["step"] == s]                              # noqa: E731
    heads = [f"{L}.{h}" for L in range(38, 45) for h in range(32)]
    # seller (scale) head map and residual by layer: X57, all fixtures
    sw = step(R57, "heads sweep 1") + step(R57, "heads sweep 2")
    allf = {r["fixture"] for r in sw}
    S = shares(sw, allf, {"h" + n for n in heads} | {f"resid@{L}" for L in LAYERS})
    scale = head_map(S, heads)
    # cure head map (discovery fixtures, as picked) and cure by layer (all fixtures)
    ch = step(R4, "e4 cure heads")
    C = cure_shares(ch, {r["fixture"] for r in ch}, {"h" + n for n in heads})
    cure = head_map(C, heads)
    cu = step(R4, "e4 cure")
    CL = cure_shares(cu, {r["fixture"] for r in cu}, {f"cure@{L}" for L in LAYERS})
    # held-out circuits on new prompts only
    lim57 = {r["fixture"]: r["base_limit"] for r in step(R57, "heads build")}
    disc57 = {r["fixture"] for r in step(R57, "heads sweep 1")}
    circ57 = step(R57, "heads circuit")
    keep57 = new_prompts(lim57, disc57, {r["fixture"] for r in circ57})
    K57 = shares(circ57, set(keep57), {"top8", "random8"})
    fx4 = [r for r in R4 if r.get("kind") == "fixture"]
    lim4, bad4 = {r["fixture"]: r["base_limit"] for r in fx4}, unreadable(fx4)
    cc = step(R4, "e4 cure circuit")
    keep4 = new_prompts(lim4, {r["fixture"] for r in ch}, {r["fixture"] for r in cc}, drop=bad4)
    K4 = cure_shares(cc, set(keep4), {"top8"})
    bb = step(R2, "heads2 buyer build")
    limB = {r["fixture"]: r["base_limit"] for r in bb if not r.get("skipped")}
    bc = step(R2, "heads2 buyer circuit")
    keepB = new_prompts(limB, {r["fixture"] for r in step(R2, "heads2 buyer sweep 1")}, {r["fixture"] for r in bc})
    KB = shares(bc, set(keepB), {"top8"})
    circuit = {DIRS[0]: {"scale": K57.get((DIRS[0], "top8")), "cure": K4.get(("N6", "top8")),
                         "buyers": KB.get((DIRS[0], "top8")), "random": K57.get((DIRS[0], "random8"))},
               DIRS[1]: {"scale": K57.get((DIRS[1], "top8")), "cure": K4.get(("P6", "top8")),
                         "buyers": KB.get((DIRS[1], "top8")), "random": K57.get((DIRS[1], "random8"))}}
    rs, rc = sorted(scale, key=lambda n: -scale[n]), sorted(cure, key=lambda n: -cure[n])
    common = [n for n in heads if n in scale and n in cure]
    out = {"story_row": story_row, "model": "gemma-4-31b-it", "runs": {k: os.path.basename(v.rstrip("/")) for k, v in runs.items()},
           "layers": LAYERS,
           "by_layer": {"scale_in": [_r(S.get((DIRS[0], f"resid@{L}"))) for L in LAYERS],
                        "scale_out": [_r(S.get((DIRS[1], f"resid@{L}"))) for L in LAYERS],
                        "cure_in": [_r(CL.get(("N6", f"cure@{L}"))) for L in LAYERS],
                        "cure_out": [_r(CL.get(("P6", f"cure@{L}"))) for L in LAYERS]},
           "heads": [[n, round(scale[n], 5), round(cure[n], 5)] for n in common],
           "scale_top8": rs[:8], "cure_top8": rc[:8], "overlap8": sorted(set(rs[:8]) & set(rc[:8])),
           "spearman": spearman([scale[n] for n in common], [cure[n] for n in common]),
           "circuit": circuit, "n_new": {"scale": len(keep57), "cure": len(keep4), "buyers": len(keepB)},
           "unreadable": bad4}
    with open(out_path or JSON_PATH, "w") as fh:
        json.dump(out, fh, indent=1)
    return out


def _r(c):
    return None if not c or c.get("share") is None else round(c["share"], 4)


# --- figure -------------------------------------------------------------------------------------------------
def fig(D):
    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.dirname(HERE))
    from mi_figures import _svg, _txt, _line, _path
    from interp_heads_figures import _bars
    W, H = 1000, 760
    body = [_txt(30, 26, "A. Both switch on at layer 41 (share moved, by layer patched)", "start", "panel")]
    x0, y0, w, h = 70, 50, 380, 210
    Ls = D["layers"]
    sx = lambda L: x0 + (L - Ls[0]) / (Ls[-1] - Ls[0]) * w                       # noqa: E731
    sy = lambda v: y0 + h - max(-0.1, min(1.1, v)) / 1.1 * h                      # noqa: E731
    for t in (0, 0.5, 1.0):
        body += [_line(x0, sy(t), x0 + w, sy(t), SLATE, 1, op=.15 if t else .5), _txt(x0 - 8, sy(t) + 4, f"{100 * t:.0f}%", "end", "axis")]
    for L in Ls:
        if L % 2 == 0 or L == 41:
            body.append(_txt(sx(L), y0 + h + 18, str(L), "middle", "axis", weight="600" if L == 41 else None))
    body.append(_line(sx(41), y0, sx(41), y0 + h, ORANGE, 1, dash="4 4", op=.6))
    series = (("scale_in", SLATE, None, "the ×1 run's states into the ×10⁶ run (X57)"),
              ("scale_out", SLATE, "6 5", "the reverse"),
              ("cure_in", ORANGE, None, "the range run's states into the plain ×10⁶ run (X61)"),
              ("cure_out", ORANGE, "6 5", "the reverse"))
    for i, (k, col, dash, lab) in enumerate(series):
        pts = [None if v is None else (sx(L), sy(v)) for L, v in zip(Ls, D["by_layer"][k])]
        body.append(_path(pts, col, 3, dash))
        yy = y0 + h + 42 + i * 18
        body += [_line(x0, yy - 4, x0 + 28, yy - 4, col, 3, dash), _txt(x0 + 36, yy, lab, "start", "note")]
    # B: head scatter
    bx, by, bw, bh = 600, 50, 340, 300
    body.append(_txt(bx - 70, 26, "B. Which heads: scale vs cure, 224 heads at 38–44", "start", "panel"))
    xs, ys = [s for _n, s, _c in D["heads"]], [c for _n, _s, c in D["heads"]]
    lo_x, hi_x, lo_y, hi_y = min(xs + [0]), max(xs), min(ys + [0]), max(ys)
    px = lambda v: bx + (v - lo_x) / ((hi_x - lo_x) or 1) * bw                    # noqa: E731
    py = lambda v: by + bh - (v - lo_y) / ((hi_y - lo_y) or 1) * bh               # noqa: E731
    body += [_line(px(0), by, px(0), by + bh, SLATE, 1, op=.4), _line(bx, py(0), bx + bw, py(0), SLATE, 1, op=.4),
             _txt(bx + bw / 2, by + bh + 32, "importance for the scale (X57, share of the gap)", "middle", "note"),
             f'<text x="{bx - 40:.1f}" y="{by + bh / 2:.1f}" text-anchor="middle" class="note" '
             f'transform="rotate(-90 {bx - 40:.1f} {by + bh / 2:.1f})">importance for the cure (X61)</text>']
    for t in (0, 0.05, 0.10):
        if lo_x <= t <= hi_x:
            body.append(_txt(px(t), by + bh + 14, f"{100 * t:.0f}%", "middle", "axis"))
        if lo_y <= t <= hi_y:
            body.append(_txt(bx - 6, py(t) + 4, f"{100 * t:.0f}%", "end", "axis"))
    st8, ct8 = set(D["scale_top8"]), set(D["cure_top8"])
    order = sorted(D["heads"], key=lambda r: (r[0] in st8) + (r[0] in ct8))
    placed = []
    for n, s, c in order:
        both, a, b = n in st8 and n in ct8, n in st8, n in ct8
        col, rad = (ONYX, 6) if both else (SLATE, 5) if a else (ORANGE, 5) if b else (GREY, 3)
        body.append(f'<circle cx="{px(s):.1f}" cy="{py(c):.1f}" r="{rad}" fill="{col}" fill-opacity="{0.9 if (a or b) else 0.45}">'
                    f'<title>L{n}: scale {100 * s:+.1f}%, cure {100 * c:+.1f}%</title></circle>')
        if a or b:
            lx, ly_ = px(s) + 8, py(c) + 4
            while any(abs(ly_ - q) < 13 and abs(lx - p) < 48 for p, q in placed):     # nudge labels apart
                ly_ += 13
            placed.append((lx, ly_))
            body.append(_txt(lx, ly_, f"L{n}", "start", "axis"))
    ly = by + bh + 56
    for i, (col, lab) in enumerate(((SLATE, "top 8 for the scale"), (ORANGE, "top 8 for the cure"), (ONYX, "in both"))):
        body += [f'<circle cx="{bx - 60 + i * 140:.1f}" cy="{ly - 4:.1f}" r="5" fill="{col}"/>', _txt(bx - 50 + i * 140, ly, lab, "start", "note")]
    sp = D.get("spearman")
    body.append(_txt(bx - 60, ly + 20, f"{len(D['overlap8'])} of 8 shared (chance 0.3); Spearman over all heads "
                     + ("n/a" if sp is None else f"{sp:+.2f}"), "start", "note"))
    # C: held-out circuits, new prompts only
    cy = 500
    nn = D["n_new"]
    _bars(body, {"circuit": D["circuit"]}, [(k, f"{lab} (n {nn.get(k, nn['scale'])})") for k, lab in CIRCUIT],
          470, cy, 34, 300,
          "C. Held-out fixtures with new prompts only: share carried by 8 heads", _txt, _line)
    ky = cy + len(CIRCUIT) * 34 + 30
    body += [f'<rect x="260" y="{ky - 11}" width="14" height="12" fill="{ONYX}" fill-opacity=".85"/>',
             _txt(280, ky, "into the plain ×10⁶ run (from the ×1 run, or from the range run)", "start", "note"),
             f'<rect x="260" y="{ky + 13}" width="14" height="12" fill="{ORANGE}" fill-opacity=".9"/>',
             _txt(280, ky + 24, "the reverse", "start", "note")]
    return _svg(W, H, "The scale and the cure: same layer, partly the same heads",
                "Panel A: share of the effect moved by patching one layer, for the scale (x1 vs x1e6, slate) and for the "
                "range sentence's cure (orange); both jump at layer 41. Panel B: each of the 224 heads at layers 38 to 44, "
                "its importance for the scale against its importance for the cure; three heads are in both top-8 lists, "
                "but overall the two rankings barely agree. Panel C: on held-out fixtures whose prompts were not used to "
                "pick the heads, the scale's top 8, the cure's top 8 and the buyers' top 8 each carry a large share; "
                "8 random heads carry almost nothing.", body)


def caption(D):
    bad = ", ".join(D.get("unreadable") or []) or "none"
    return ("Gemma 4 31B, sellers unless marked; the opening is forced as in E2; shares of the log-probability gap between "
            "the two runs' greedy prices at the deciding token. A: X57's residual patch (all post-limit tokens) and X61's "
            "cure (the 79 shared tokens after the range line), 30 fixtures. B: single-head patches, mean of both "
            "directions; scale on 30 fixtures, cure on its 10 discovery fixtures. C: only held-out fixtures whose limit "
            "(and so the whole prompt) does not occur among the discovery fixtures, one per prompt; range-run fixtures "
            f"without probability on the readout pair dropped ({bad}). 95% CIs over fixtures. Exploratory; one model "
            f"({D['story_row']}).")


def cards(json_path=None):
    p = json_path or JSON_PATH
    if not os.path.exists(p):
        return []
    with open(p) as fh:
        D = json.load(fh)
    title = "The scale and the cure: same layer, partly the same heads"
    return [f'<figure class="chart-card mi-card mi-wide"><div class="chart-title"><h3>{title}</h3>'
            f'<span>synthesis · {D["story_row"]}</span></div><div class="chart-scroll" role="region" tabindex="0" '
            f'aria-label="{html.escape(title)} chart, scroll horizontally on small screens">{fig(D)}</div>'
            f"<figcaption>{caption(D)}</figcaption></figure>"]


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--preview")
    a = ap.parse_args(argv)
    if a.refresh:
        refresh()
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
