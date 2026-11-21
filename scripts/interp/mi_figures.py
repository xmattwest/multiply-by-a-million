#!/usr/bin/env python3
"""mi_figures.py (LAPTOP-SIDE): figures for the mechanistic-interpretability track (X51-X54).

    python3 scripts/interp/mi_figures.py --refresh     re-read the local runs (data/interp/, gitignored)
                                                       into config/interp/mi_figures.json (tracked, numbers only)
    python3 scripts/interp/mi_figures.py --preview F   write a standalone HTML page with the four figures

The research-site builder calls section_html(); it reads only the tracked JSON, so the site builds from
the repository alone. Standard library only; SVG drawn by hand in the Dawn State palette. Exploratory.
"""
import argparse
import html
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
JSON_PATH = os.path.join(ROOT, "config", "interp", "mi_figures.json")
RUNS = {"e0": "interp-e0-gemma-4-31b-it-20261005T023324Z",
        "e1e2": "interp-e1e2-gemma-4-31b-it-20261005T105515Z",
        "e3": "interp-e3-gemma-4-31b-it-20261005T122056Z",
        "e3b": "interp-e3b-gemma-4-31b-it-20261005T233449Z"}
VLLM_RUN = "X-digits-gemma-4-31b-it-20261004T042802Z"
OFF_SCALE_MAX = 3                 # E2 cells with more off-scale readings than this (of 30) are not drawn
SLATE, ORANGE, SAGE, ONYX, IVORY = "#2C3E50", "#F39C12", "#87A96B", "#0F0F0F", "#FFFFF0"


# --- refresh: local runs -> small tracked JSON ------------------------------------------
def refresh():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import launch_interp as li
    d = lambda k: os.path.join(ROOT, "data", "interp", RUNS[k])        # noqa: E731
    out = {"model": "gemma-4-31b-it", "role": "seller", "runs": RUNS, "vllm_run": VLLM_RUN,
           "story_rows": {"e0": "X51", "e1e2": "X52", "e3": "X53", "e3b": "X54"}}
    # E0: per-fixture mean seller gap, pod vs vLLM, same prompts and seeds.
    rows = [json.loads(l) for l in open(os.path.join(d("e0"), "results.jsonl"))]
    pod = [r for r in rows if r["step"] == "sampled" and r.get("gap") is not None]
    vl = {r["item_id"]: r for r in li.load_vllm(os.path.join(ROOT, "data", "explore", VLLM_RUN), li.CONDITIONS)}
    fx = {}
    for r in pod:
        v = vl.get(r["item_id"])
        if v and v.get("gap") is not None:
            a = fx.setdefault((r["item_id"][:3], r["condition"]), ([], []))
            a[0].append(100 * r["gap"]); a[1].append(100 * v["gap"])
    out["e0"] = {"points": [{"fixture": f, "condition": c, "pod": round(st.mean(p), 2), "vllm": round(st.mean(q), 2)}
                            for (f, c), (p, q) in sorted(fx.items())],
                 "summary": json.load(open(os.path.join(d("e0"), "summary.json")))["fidelity"]}
    # E2: share of the gap restored per layer (off-scale prices excluded; summary_v2).
    e2 = json.load(open(os.path.join(d("e1e2"), "summary_v2.json")))["e2"]
    out["e2"] = {"n_fixtures": e2["n_fixtures"], "layers": 60, "off_scale_max": OFF_SCALE_MAX, "directions": {}}
    for name, v in e2["directions"].items():
        out["e2"]["directions"][name] = {pos: [{"layer": int(L), "fraction": c["fraction"], "ci": c.get("ci"),
                                                 "off_scale": c["slips"], "n": c["n"]}
                                                for L, c in sorted(cells.items(), key=lambda kv: int(kv[0]))]
                                         for pos, cells in v["cells"].items()}
    # E3 (prompt only) and E3b (prompt + reply opening): gap by alpha.
    e3 = json.load(open(os.path.join(d("e3"), "summary.json")))["e3"]["directions"]
    e3b = json.load(open(os.path.join(d("e3b"), "summary.json")))
    pick = lambda c: {"gap": c["mean_gap"], "ci": c.get("ci"), "valid": c["valid"], "n": c["n"],   # noqa: E731
                      "share_moved": c.get("share_moved")}
    out["e3"] = {dn: {"none": pick(v["none"]["0.0"]), "v_pos": {a: pick(c) for a, c in v["v_pos"].items()}}
                 for dn, v in e3.items()}
    out["e3b_steer"] = {dn: {vec: {a: pick(c) for a, c in cells.items()} for vec, cells in v.items()}
                        for dn, v in e3b["steer"]["directions"].items()}
    out["e3b_bridge"] = {dn: {k: {"fraction": c["fraction"], "ci": c["ci"], "off_scale": c["off_scale"], "n": c["n"]}
                              for k, c in cells.items()} for dn, cells in e3b["bridge"]["cells"].items()}
    out["e3b_vectors"] = e3b["vectors"]
    with open(JSON_PATH, "w") as fh:
        json.dump(out, fh, indent=1)
    return out


# --- SVG helpers -------------------------------------------------------------------------
def _svg(w, h, title, desc, body, cls="mi-chart"):
    t, dsc = html.escape(title), html.escape(desc)
    slug = "mi-" + "".join(ch for ch in title.lower() if ch.isalnum())[:24]
    return (f'<svg class="{cls}" viewBox="0 0 {w} {h}" role="img" aria-labelledby="{slug}-t {slug}-d">'
            f'<title id="{slug}-t">{t}</title><desc id="{slug}-d">{dsc}</desc>{"".join(body)}</svg>')


def _txt(x, y, s, anchor="start", cls="", size=None, fill=None, weight=None):
    extra = (f' font-size="{size}"' if size else "") + (f' fill="{fill}"' if fill else "") + \
        (f' font-weight="{weight}"' if weight else "")
    return f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" class="{cls}"{extra}>{html.escape(s)}</text>'


def _line(x1, y1, x2, y2, stroke=SLATE, w=2, dash=None, op=1.0):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{w}"'
            f' stroke-opacity="{op}"{d}/>')


def _path(points, stroke, w=3, dash=None, op=1.0):
    """Polyline with gaps at None."""
    segs, cur = [], []
    for p in points:
        if p is None:
            if len(cur) > 1:
                segs.append(cur)
            cur = []
        else:
            cur.append(p)
    if len(cur) > 1:
        segs.append(cur)
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return "".join(f'<polyline fill="none" stroke="{stroke}" stroke-width="{w}" stroke-opacity="{op}"'
                   f' stroke-linejoin="round" stroke-linecap="round"{d} points="'
                   + " ".join(f"{x:.1f},{y:.1f}" for x, y in s) + '"/>' for s in segs)


# --- figure 1: E0 fidelity -----------------------------------------------------------------
def fig_e0(D):
    W, H, L, R, T, B = 470, 420, 70, 20, 20, 60
    lo, hi = 0, 60
    sx = lambda v: L + (v - lo) / (hi - lo) * (W - L - R)          # noqa: E731
    sy = lambda v: H - B - (v - lo) / (hi - lo) * (H - T - B)      # noqa: E731
    body = []
    for t in range(0, 61, 10):
        body += [_line(sx(t), T, sx(t), H - B, SLATE, 1, op=.15), _line(L, sy(t), W - R, sy(t), SLATE, 1, op=.15),
                 _txt(sx(t), H - B + 22, f"{t}%", "middle"), _txt(L - 10, sy(t) + 5, f"{t}%", "end")]
    body.append(_line(sx(lo), sy(lo), sx(hi), sy(hi), SLATE, 2, dash="5 6", op=.6))
    for p in D["e0"]["points"]:
        col = SLATE if p["condition"] == "D2" else ORANGE
        body.append(f'<circle cx="{sx(p["vllm"]):.1f}" cy="{sy(p["pod"]):.1f}" r="5.5" fill="{col}" fill-opacity=".85"/>')
    body += [_txt((L + W - R) / 2, H - 12, "vLLM (the engine behind the data): seller's opening above floor", "middle", "axis"),
             f'<text transform="translate(18,{(T + H - B) / 2}) rotate(-90)" text-anchor="middle" class="axis">'
             'Transformers + hooks</text>',
             f'<circle cx="{L + 18}" cy="{T + 14}" r="6" fill="{SLATE}"/>', _txt(L + 30, T + 19, "×1 limit (e.g. 81 dollars)"),
             f'<circle cx="{L + 18}" cy="{T + 38}" r="6" fill="{ORANGE}"/>', _txt(L + 30, T + 43, "×10⁶ limit (81000000 dollars)")]
    return _svg(W, H, "E0: the hooked model reproduces the served model",
                "Scatter of per-fixture mean seller opening gaps: vLLM on the x axis, the Transformers copy on the y axis, "
                "for the x1 and x1e6 limits. Points fall on the diagonal.", body)


# --- figure 2: E2 layer profile --------------------------------------------------------------
def fig_e2(D):
    W, H = 940, 440
    panels = (("x1->x1e6", "×1 activations into the ×10⁶ prompt"), ("x1e6->x1", "×10⁶ activations into the ×1 prompt"))
    pw, gap, L, T, B = 395, 60, 84, 48, 100
    y0, y1 = -0.2, 1.3
    body = []
    series = (("post", SLATE, 3.5, None, "all text after the limit"), ("final", SAGE, 3, None, "last token only"),
              ("span", ORANGE, 2.5, "6 5", "the limit's own digits"))
    for i, (key, title) in enumerate(panels):
        x0 = L + i * (pw + gap)
        sx = lambda l, x0=x0: x0 + l / 59 * pw                      # noqa: E731
        sy = lambda v: T + (y1 - v) / (y1 - y0) * (H - T - B)       # noqa: E731
        body.append(_txt(x0, 24, title, "start", "panel"))
        for t in (0, .25, .5, .75, 1.0, 1.25):
            body.append(_line(x0, sy(t), x0 + pw, sy(t), SLATE, 2 if t in (0, 1.0) else 1, op=.5 if t in (0, 1.0) else .12))
            if i == 0:
                body.append(_txt(x0 - 10, sy(t) + 5, f"{t * 100:.0f}%", "end"))
        for l in range(0, 60, 10):
            body.append(_txt(sx(l), H - B + 22, str(l), "middle"))
        body.append(_line(sx(41), T - 6, sx(41), H - B, ORANGE, 2, dash="3 5", op=.8))
        body.append(_txt(sx(41) + 6, T + 6, "layer 41", "start", "note"))
        for pos, col, w, dash, _lab in series:
            cells = D["e2"]["directions"][key][pos]
            pts = [None if c["fraction"] is None or c["off_scale"] > OFF_SCALE_MAX or not y0 <= c["fraction"] <= y1
                   else (sx(c["layer"]), sy(c["fraction"])) for c in cells]
            body.append(_path(pts, col, w, dash))
        body.append(_txt(x0 + pw / 2, H - B + 46, "layer (0 = first, 59 = last)", "middle", "axis"))
    lx = L
    for pos, col, w, dash, lab in series:
        body += [_line(lx, H - 18, lx + 34, H - 18, col, w, dash), _txt(lx + 44, H - 13, lab)]
        lx += 60 + 8.2 * len(lab)
    body.append(f'<text transform="translate(16,{(T + H - B) / 2}) rotate(-90)" text-anchor="middle" class="axis">'
                'share of the ×1 / ×10⁶ gap restored</text>')
    return _svg(W, H, "E2: where the big-number timidity lives",
                "Two line charts over the 60 layers. Swapping the text after the limit restores little before about "
                "layer 40 and 82 to 95 percent from layer 41; the last token alone restores up to about 30 percent; the "
                "limit digits break the number before layer 39 and do nothing after layer 41.", body)


# --- figure 3: steering dose-response ---------------------------------------------------------
def fig_steer(D):
    W, H, L, R, T, B = 940, 430, 70, 260, 30, 62
    alphas = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)
    sx = lambda a: L + a / 2.0 * (W - L - R)                        # noqa: E731
    y0, y1 = 10, 62
    sy = lambda v: T + (y1 - v) / (y1 - y0) * (H - T - B)           # noqa: E731
    body = []
    for t in range(10, 61, 10):
        body += [_line(L, sy(t), W - R, sy(t), SLATE, 1, op=.12), _txt(L - 10, sy(t) + 5, f"{t}%", "end")]
    for a in alphas:
        body.append(_txt(sx(a), H - B + 22, f"{a:g}", "middle"))
    sb, e3 = D["e3b_steer"], D["e3"]
    base1, base6 = sb["x1+v"]["none"]["0.0"]["gap"], sb["x1e6-v"]["none"]["0.0"]["gap"]
    body += [_line(L, sy(base1), W - R, sy(base1), SLATE, 1.5, dash="2 5", op=.7),
             _line(L, sy(base6), W - R, sy(base6), ORANGE, 1.5, dash="2 5", op=.9)]
    for dn, col in (("x1+v", SLATE), ("x1e6-v", ORANGE)):
        cells = sb[dn]["v_pos|prompt+reply"]
        pts = [(sx(0), sy(sb[dn]["none"]["0.0"]["gap"]))] + [(sx(float(a)), sy(c["gap"])) for a, c in
                                                            sorted(cells.items(), key=lambda kv: float(kv[0]))]
        for a, c in cells.items():
            if c.get("ci"):
                body.append(_line(sx(float(a)), sy(c["ci"][0]), sx(float(a)), sy(c["ci"][1]), col, 3, op=.45))
        body.append(_path(pts, col, 4))
        body += [f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{col}"/>' for x, y in pts]
        old = e3[dn]
        opts = [(sx(0), sy(old["none"]["gap"]))] + [(sx(float(a)), sy(c["gap"])) for a, c in
                                                   sorted(old["v_pos"].items(), key=lambda kv: float(kv[0]))]
        body.append(_path(opts, col, 2.5, dash="6 6", op=.55))
        nm = sb[dn].get("nonmoney|prompt+reply", {})
        for a, c in nm.items():
            body.append(f'<rect x="{sx(float(a)) - 6:.1f}" y="{sy(c["gap"]) - 6:.1f}" width="12" height="12" '
                        f'fill="{IVORY}" stroke="{SAGE}" stroke-width="3" transform="rotate(45 {sx(float(a)):.1f} '
                        f'{sy(c["gap"]):.1f})"/>')
    lx = W - R + 26
    leg = [(SLATE, None, "circle", "×1 prompt + v"), (ORANGE, None, "circle", "×10⁶ prompt − v"),
           (SLATE, "6 6", "line", "prompt only (E3): no effect"), (SAGE, None, "diamond", "non-money control"),
           (SLATE, "2 5", "line", "unsteered ×1 and ×10⁶")]
    for j, (col, dash, kind, lab) in enumerate(leg):
        yy = T + 20 + j * 30
        if kind == "circle":
            body += [_line(lx, yy, lx + 30, yy, col, 4), f'<circle cx="{lx + 15}" cy="{yy}" r="6" fill="{col}"/>']
        elif kind == "diamond":
            body.append(f'<rect x="{lx + 9}" y="{yy - 6}" width="12" height="12" fill="{IVORY}" stroke="{SAGE}" '
                        f'stroke-width="3" transform="rotate(45 {lx + 15} {yy})"/>')
        else:
            body.append(_line(lx, yy, lx + 30, yy, col, 2.5, dash=dash, op=.7))
        body.append(_txt(lx + 40, yy + 5, lab))
    body += [_txt(W - R + 26, T + 190, "Random direction: breaks the", "start", "note"),
             _txt(W - R + 26, T + 210, "reply format (not drawn).", "start", "note"),
             _txt(sx(1.0), H - 14, "steering strength α (1 ≈ the full ×10⁶ − ×1 difference)", "middle", "axis"),
             f'<text transform="translate(16,{(T + H - B) / 2}) rotate(-90)" text-anchor="middle" class="axis">'
             "seller's opening above floor</text>"]
    return _svg(W, H, "E3b: steering the opening both ways",
                "Seller opening gap against steering strength. Adding the direction to x1 prompts lowers the gap from "
                "41 to 19 percent; subtracting it from x1e6 prompts raises it from 21 to 58 percent. Steering the prompt "
                "alone (E3) stays flat; the non-money control barely moves.", body)


# --- figure 4: prompt vs reply ----------------------------------------------------------------
def fig_scope(D):
    W, H, L, T = 940, 300, 300, 40
    br = D["e3b_bridge"]
    rows = (("own|prompt|1.0", "the fixture's own difference", "prompt only"),
            ("own|reply|1.0", "", "reply opening only"),
            ("own|all|1.0", "", "both"),
            ("v_pos|prompt|1.0", "shared direction (other 29)", "prompt only"),
            ("v_pos|reply|1.0", "", "reply opening only"),
            ("v_pos|all|1.0", "", "both"))
    pw = W - L - 40
    sx = lambda v: L + max(0, min(1.2, v)) / 1.2 * pw              # noqa: E731
    body = []
    for t in (0, .25, .5, .75, 1.0):
        body += [_line(sx(t), T - 10, sx(t), H - 40, SLATE, 2 if t in (0, 1.0) else 1, op=.5 if t in (0, 1.0) else .12),
                 _txt(sx(t), H - 18, f"{t * 100:.0f}%", "middle")]
    y = T
    for key, group, lab in rows:
        if group:
            y += 8
            body.append(_txt(0, y + 14, group, "start", "panel"))
        for k, (dn, col) in enumerate((("x1+v", SLATE), ("x1e6-v", ORANGE))):
            c = br[dn].get(key, {})
            f = c.get("fraction") or 0
            body.append(f'<rect x="{sx(0):.1f}" y="{y + k * 11:.1f}" width="{sx(f) - sx(0):.1f}" height="10" fill="{col}"/>')
        body.append(_txt(L - 12, y + 16, lab, "end"))
        body.append(_txt(sx(max(br["x1+v"].get(key, {}).get("fraction") or 0, br["x1e6-v"].get(key, {}).get("fraction") or 0)) + 8,
                         y + 16, f'{100 * (br["x1+v"].get(key, {}).get("fraction") or 0):.0f}% / '
                         f'{100 * (br["x1e6-v"].get(key, {}).get("fraction") or 0):.0f}%', "start", "note"))
        y += 30
    return _svg(W, H, "E3b: the timidity sits on the reply's first tokens",
                "Bars: share of the gap moved by steering at layer 44 in E2's forced-opening setup, for the prompt's "
                "tokens only, the reply's eleven opening tokens only, or both; for each fixture's own difference and "
                "the shared direction. Prompt only moves 8 to 15 percent; the reply opening carries most of it.", body)


# --- the website block -------------------------------------------------------------------------
def section_html(json_path=JSON_PATH):
    if not os.path.exists(json_path):
        return ""
    D = json.load(open(json_path))
    sb = D["e3b_steer"]
    g1 = sb["x1+v"]["v_pos|prompt+reply"]["1.0"]; g6 = sb["x1e6-v"]["v_pos|prompt+reply"]["1.0"]
    fd = D["e0"]["summary"]

    def card(title, tag, svg, caption, wide=True):
        return (f'<figure class="chart-card mi-card{" mi-wide" if wide else ""}"><div class="chart-title"><h3>{title}</h3>'
                f'<span>{tag}</span></div><div class="chart-scroll" role="region" tabindex="0" aria-label="{html.escape(title)} '
                f'chart, scroll horizontally on small screens">{svg}</div><figcaption>{caption}</figcaption></figure>')
    cards = [
        card("Swap one prompt’s activations into the other. Where does the gap move?", "E2 · activation patching",
             fig_e2(D), f"Gemma 4 31B, sellers, {D['e2']['n_fixtures']} fixtures; the ×1 and ×10⁶ prompts differ only in "
             "the limit’s digits. Swapping all the text after the limit restores 82–95% of the gap from layer 41 "
             "(a global-attention layer) and leaves the price on its own scale. Before about layer 40, later layers "
             "re-read the digits. Cells where the swap broke the number’s size in more than "
             f"{OFF_SCALE_MAX} of 30 fixtures are not drawn. Readout: greedy price after a forced reply opening."),
        card("Add the difference. Subtract it. The opening follows.", "E3b · steering", fig_steer(D),
             f"Layer 44. The direction is each token’s mean ×10⁶ − ×1 difference over the other 29 fixtures, added "
             f"to the prompt after the limit and to the reply’s first 11 tokens as they are generated. Real sampled "
             f"openings, 100% valid: at α = 1 the ×1 seller drops to {g1['gap']:.1f}% and the ×10⁶ seller rises to "
             f"{g6['gap']:.1f}%, 86% of the way in both directions. Bars: 95% CIs over fixtures. Steering the prompt "
             "alone (E3, dashed) does nothing."),
        card("It is computed as the model starts its reply.", "E3b · prompt vs reply", fig_scope(D),
             "Same steering in E2’s setup (opening forced, greedy price). Slate: ×1 + v; orange: ×10⁶ − v. "
             "The prompt’s 106 tokens after the limit carry little at layer 44; the reply’s opening "
             "<code>{\"action\": \"OFFER\", \"price\": </code> carries most of it."),
        card("First: is the hooked copy the same model?", "E0 · fidelity", fig_e0(D),
             f"Per-fixture mean seller opening, same prompts and seeds. Differences {fd['seller_D2_diff']:.1f} points "
             f"(×1) and {fd['seller_D8_diff']:.1f} (×10⁶); the shift reproduces in 30 of 30 fixtures.", wide=False),
    ]
    cards += extra_cards()
    return ('<div class="mi-results"><div class="evidence-label"><span class="badge exploratory">Exploratory</span>'
            "<span>Mechanistic interpretability · Gemma 4 31B · sellers · notebook rows "
            f"{D['story_rows']['e0']}–{D['story_rows']['e3b']}</span></div><div class=\"mi-grid\">"
            + "".join(cards) + f'</div><style>{CSS}</style><p class="fine-print">One model, one role, exploratory sessions on the pinned '
            "checkpoint and image; not pre-registered. Random directions of the same size break the reply format rather "
            "than leaving it intact. Full analyses: <a href=\"notes/story.html\">the research notebook</a>. "
            "Next: <a href=\"notes/interp-next.html\">the layer sweep and buyers</a>.</p></div>")


def extra_cards():
    """Figure cards from plug-ins: scripts/interp/mi_figures_<name>.py with cards() -> [html, ...].
    Each plug-in reads its own tracked JSON (config/interp/mi_figures_<name>.json). A broken one is skipped."""
    import glob
    import importlib.util
    out = []
    for path in sorted(glob.glob(os.path.join(HERE, "mi_figures_*.py"))):
        try:
            spec = importlib.util.spec_from_file_location(os.path.basename(path)[:-3], path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            out += list(mod.cards())
        except Exception as e:             # noqa: BLE001  never break the site build
            print(f"note: {os.path.basename(path)} skipped ({type(e).__name__}: {e})")
    return out


CSS = """
.mi-results{margin-top:40px}.mi-grid{display:grid;grid-template-columns:1fr;gap:24px;margin:16px 0}
.mi-card .mi-chart{width:100%;height:auto;display:block;margin:6px 0}
.mi-card:not(.mi-wide){max-width:560px}
.mi-chart text{fill:#0F0F0F;font-family:Inter,Arial,sans-serif;font-size:14px}
.mi-chart .axis{font-size:13px;fill:#2C3E50}.mi-chart .panel{font-weight:600;font-size:15px;fill:#2C3E50}
.mi-chart .note{font-size:13px;fill:#2C3E50}
@media(max-width:520px){.chart-scroll .mi-chart{min-width:650px}}
"""


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--preview")
    a = ap.parse_args(argv)
    if a.refresh:
        refresh()
        print(f"wrote {os.path.relpath(JSON_PATH, ROOT)}")
    if a.preview:
        with open(a.preview, "w") as fh:
            fh.write('<!doctype html><meta charset="utf-8"><link rel="stylesheet" href="research.css">'
                     f"<style>{CSS}body{{background:#FFFFF0;padding:30px}}</style>{section_html()}")
        print(f"wrote {a.preview}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
