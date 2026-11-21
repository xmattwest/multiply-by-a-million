#!/usr/bin/env python3
"""interp_buyers_figures.py (LAPTOP-SIDE): the buyers session's figures (X58-X59) for the research website.
The plug-in scripts/interp/mi_figures_buyers.py is a shim importing cards() from here (keeps the pod's create
request small).

    python3 scripts/interp_buyers_figures.py --refresh data/interp/<buyers run>   -> config/interp/mi_figures_buyers.json
    python3 scripts/interp_buyers_figures.py --preview F                         standalone HTML with the cards

--refresh also re-reads the sellers' E3b run (X54) from data/interp/ for the side-by-side bars, with the same
paired statistic. cards() reads only the tracked JSON, so the site builds from the repository; no JSON -> no
cards. Standard library only; SVG helpers and palette from scripts/interp/mi_figures.py. Exploratory.
"""
import argparse
import html
import json
import os
import statistics as st
import sys

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "interp")     # scripts/interp (mi_figures.py)
ROOT = os.path.dirname(os.path.dirname(HERE))
JSON_PATH = os.path.join(ROOT, "config", "interp", "mi_figures_buyers.json")
MAIN_JSON = os.path.join(ROOT, "config", "interp", "mi_figures.json")
VLLM_RUN = "X-digits-gemma-4-31b-it-20261004T042802Z"
SELLER_E3B = "interp-e3b-gemma-4-31b-it-20261005T233449Z"
SLATE, ORANGE, SAGE, ONYX, IVORY, GREY = "#2C3E50", "#F39C12", "#87A96B", "#0F0F0F", "#FFFFF0", "#8A8F94"
OFF_SCALE_MAX = 3                 # E2 cells with more off-scale readings than this (of 30) are not drawn
B_ROWS = (("seller", "v_pos|prompt+reply", "Sellers: the seller’s own direction (X54)"),
          ("seller", "nonmoney|prompt+reply", "Sellers: non-money control"),
          ("buyer", "v_pos|prompt+reply", "Buyers: the buyer’s own direction"),
          ("buyer", "seller|prompt+reply", "Buyers: the seller’s direction"),
          ("buyer", "seller_norm|prompt+reply", "Buyers: the seller’s, at buyer size"),
          ("buyer", "nonmoney|prompt+reply", "Buyers: non-money control"))


def _mods():
    sys.path[:0] = [os.path.join(ROOT, "scripts"), HERE]
    import launch_interp as li
    import interp_buyers_laptop as eb
    return li, eb


# --- refresh: local runs -> small tracked JSON ------------------------------------------
def refresh(run_dir, story_rows=("X58", "X59")):
    li, eb = _mods()
    rows = [json.loads(l) for l in open(os.path.join(run_dir, "results.jsonl"))]
    s = json.load(open(os.path.join(run_dir, "summary.json")))
    out = {"run": os.path.basename(run_dir.rstrip("/")), "story_rows": list(story_rows), "model": "gemma-4-31b-it",
           "role": "buyer", "vllm_run": VLLM_RUN, "seller_e3b_run": SELLER_E3B}
    # E0: per-fixture mean buyer discount, pod vs vLLM.
    vl = {r["item_id"]: r for r in li.load_vllm(os.path.join(ROOT, "data", "explore", VLLM_RUN), li.CONDITIONS)}
    fx = {}
    for r in rows:
        v = vl.get(r.get("item_id"))
        if r["step"] == "sampled" and r.get("gap") is not None and v and v.get("gap") is not None:
            a = fx.setdefault((r["item_id"][:3], r["condition"]), ([], []))
            a[0].append(100 * r["gap"]); a[1].append(100 * v["gap"])
    out["e0"] = {"points": [{"fixture": f, "condition": c, "pod": round(st.mean(p), 2), "vllm": round(st.mean(q), 2)}
                            for (f, c), (p, q) in sorted(fx.items())],
                 "fidelity": {k: v for k, v in s["e0_buyer_fidelity"].items()}}
    # E2 (coarse): share restored per layer, post and final.
    e2 = s["e2"]
    out["e2"] = {"n_fixtures": e2["n_fixtures"], "layers": s["e2_layers"], "directions": {}}
    for name, d in e2["directions"].items():
        out["e2"]["directions"][name] = {pos: [{"layer": int(L), "fraction": c["fraction"], "ci": c.get("ci"),
                                                "off_scale": c["slips"], "n": c["n"]}
                                               for L, c in sorted(cells.items(), key=lambda kv: int(kv[0]))]
                                         for pos, cells in d["cells"].items()}
    # B at alpha 1, buyers (this run) and sellers (X54), the same paired statistic.
    sel = [json.loads(l) for l in open(os.path.join(ROOT, "data", "interp", SELLER_E3B, "results.jsonl"))]
    tabs = {"buyer": eb.steer_table(li, [r for r in rows if r["step"] == "B"]),
            "seller": eb.steer_table(li, [r for r in sel if r["step"] == "e3b steer"], window=li.ON_SCALE)}
    out["B"] = {}
    for role, key, _lab in B_ROWS:
        for d in ("x1+v", "x1e6-v"):
            c = tabs[role]["directions"][d][key]["1.0"]
            out["B"].setdefault(role, {}).setdefault(key, {})[d] = {
                k: c.get(k) for k in ("share_moved", "share_ci", "points", "points_ci", "valid", "n", "n_fixtures")}
    out["B_unsteered"] = {role: t["unsteered_gap"] for role, t in tabs.items()}
    out["B_layer"] = s["steer"]["layer"]
    # Directions: seller vs buyer cosines, with split-half references.
    v = s["vectors"][str(s["layers"]["A"][0])]
    out["vectors"] = {k: v[k] for k in ("layer", "cos_seller_buyer", "cos_split_half", "cos_seller_buyer_per_reply_token",
                                        "norm_v", "cos_fixture_diffs_mean", "same_token_positions", "n_post")}
    a = s["bridge"][str(s["layers"]["A"][0])]["cells"]
    out["A"] = {d: {k: {f: c.get(f) for f in ("fraction", "ci", "points", "off_scale", "n")} for k, c in cells.items()}
                for d, cells in a.items()}
    with open(JSON_PATH, "w") as fh:
        json.dump(out, fh, indent=1)
    return out


# --- figures ----------------------------------------------------------------------------------
def _h():
    sys.path.insert(0, HERE)
    import mi_figures as M
    return M


def fig_e0(D):
    M = _h()
    W, H, L, R, T, B = 470, 420, 70, 20, 20, 60
    lo, hi = 25, 60
    sx = lambda v: L + (v - lo) / (hi - lo) * (W - L - R)          # noqa: E731
    sy = lambda v: H - B - (v - lo) / (hi - lo) * (H - T - B)      # noqa: E731
    body = []
    for t in range(25, 61, 5):
        body += [M._line(sx(t), T, sx(t), H - B, SLATE, 1, op=.15), M._line(L, sy(t), W - R, sy(t), SLATE, 1, op=.15)]
        if t % 10 == 0:
            body += [M._txt(sx(t), H - B + 22, f"{t}%", "middle"), M._txt(L - 10, sy(t) + 5, f"{t}%", "end")]
    body.append(M._line(sx(lo), sy(lo), sx(hi), sy(hi), SLATE, 2, dash="5 6", op=.6))
    for p in D["e0"]["points"]:
        col = SLATE if p["condition"] == "D2" else ORANGE
        body.append(f'<circle cx="{sx(p["vllm"]):.1f}" cy="{sy(p["pod"]):.1f}" r="5.5" fill="{col}" fill-opacity=".85"/>')
    fd = D["e0"]["fidelity"]
    for c, col, val in (("D2", SLATE, (fd["vllm_D2"], fd["pod_D2"])), ("D8", ORANGE, (fd["vllm_D8"], fd["pod_D8"]))):
        body.append(f'<rect x="{sx(val[0]) - 8:.1f}" y="{sy(val[1]) - 8:.1f}" width="16" height="16" fill="none" '
                    f'stroke="{col}" stroke-width="3"/>')
    body += [M._txt((L + W - R) / 2, H - 12, "vLLM: buyer's opening below ceiling", "middle", "axis"),
             f'<text transform="translate(18,{(T + H - B) / 2}) rotate(-90)" text-anchor="middle" class="axis">'
             'Transformers + hooks</text>',
             f'<circle cx="{L + 18}" cy="{T + 14}" r="6" fill="{SLATE}"/>', M._txt(L + 30, T + 19, "×1 ceiling (fixture mean)"),
             f'<circle cx="{L + 18}" cy="{T + 38}" r="6" fill="{ORANGE}"/>', M._txt(L + 30, T + 43, "×10⁶ ceiling"),
             f'<rect x="{L + 10}" y="{T + 54}" width="16" height="16" fill="none" stroke="{SLATE}" stroke-width="3"/>',
             M._txt(L + 30, T + 67, "overall means")]
    return M._svg(W, H, "Buyer E0: the hooked copy reproduces the buyer",
                  "Scatter of per-fixture mean buyer opening discounts: vLLM on the x axis, the Transformers copy on the y "
                  "axis, for the x1 and x1e6 ceilings. The overall means (squares) fall on the diagonal; single fixtures "
                  "scatter, as much as vLLM's own repeat samples do.", body)


def fig_e2(D, seller_e2=None):
    M = _h()
    W, H = 940, 440
    panels = (("x1->x1e6", "×1 activations into the ×10⁶ prompt"), ("x1e6->x1", "×10⁶ activations into the ×1 prompt"))
    pw, gap, L, T, B = 395, 60, 84, 48, 100
    y0, y1 = -0.2, 1.3
    body = []
    series = (("post", SLATE, 3.5, None, "buyers: all text after the limit"), ("final", SAGE, 3, None, "buyers: last token only"))
    for i, (key, title) in enumerate(panels):
        x0 = L + i * (pw + gap)
        sx = lambda l, x0=x0: x0 + l / 59 * pw                      # noqa: E731
        sy = lambda v: T + (y1 - max(y0, min(y1, v))) / (y1 - y0) * (H - T - B)   # noqa: E731
        body.append(M._txt(x0, 24, title, "start", "panel"))
        for t in (0, .25, .5, .75, 1.0, 1.25):
            body.append(M._line(x0, sy(t), x0 + pw, sy(t), SLATE, 2 if t in (0, 1.0) else 1, op=.5 if t in (0, 1.0) else .12))
            if i == 0:
                body.append(M._txt(x0 - 10, sy(t) + 5, f"{t * 100:.0f}%", "end"))
        for l in range(0, 60, 10):
            body.append(M._txt(sx(l), H - B + 22, str(l), "middle"))
        body.append(M._line(sx(41), T - 6, sx(41), H - B, ORANGE, 2, dash="3 5", op=.8))
        body.append(M._txt(sx(41) + 6, T + 6, "layer 41", "start", "note"))
        if seller_e2:
            cells = seller_e2["directions"][key]["post"]
            pts = [None if c["fraction"] is None or c["off_scale"] > OFF_SCALE_MAX else (sx(c["layer"]), sy(c["fraction"]))
                   for c in cells]
            body.append(M._path(pts, ONYX, 2, dash="6 5", op=.45))
        for pos, col, w, dash, _lab in series:
            cells = D["e2"]["directions"][key][pos]
            pts = [None if c["fraction"] is None or c["off_scale"] > OFF_SCALE_MAX else (sx(c["layer"]), sy(c["fraction"]))
                   for c in cells]
            body.append(M._path(pts, col, w, dash))
            body += [f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{col}"/>' for x, y in (p for p in pts if p)]
            for c in cells:
                if c["fraction"] is not None and c["off_scale"] > OFF_SCALE_MAX:
                    body.append(M._txt(sx(c["layer"]), sy(y0) - 4, "×", "middle", "note", fill=col))
        body.append(M._txt(x0 + pw / 2, H - B + 46, "layer patched (every second; 0 = first, 59 = last)", "middle", "axis"))
    lx = L
    leg = list(series) + ([("", ONYX, 2, "6 5", "sellers (X52): all text after the limit")] if seller_e2 else [])
    for _pos, col, w, dash, lab in leg:
        body += [M._line(lx, H - 18, lx + 34, H - 18, col, w, dash, op=.45 if lab.startswith("sellers") else 1),
                 M._txt(lx + 44, H - 13, lab)]
        lx += 60 + 7.4 * len(lab)
    body.append(f'<text transform="translate(16,{(T + H - B) / 2}) rotate(-90)" text-anchor="middle" class="axis">'
                'share of the ×1 / ×10⁶ gap restored</text>')
    return M._svg(W, H, "Buyer E2: the same layers, more on the last token",
                  "Two line charts over the layers. For buyers, swapping the text after the limit restores about all of "
                  "the gap from layer 42, like the sellers' 82 to 95 percent; the last token alone restores 55 to 70 "
                  "percent from layer 42, against 24 to 30 percent for sellers.", body)


def _bar_chart(title, desc, rows, W=940, L=330, scale=(-0.25, 1.25), ticks=(0, .25, .5, .75, 1.0, 1.25), note=None,
               tick_fmt=lambda t: f"{t * 100:.0f}%", R=120):
    """rows: [(group_label or '', label, [(value, ci, colour, text), ...])]"""
    M = _h()
    T, bh = 34, 11
    rh = max(34, max(len(b) for _g, _l, b in rows) * (bh + 6) + 10)
    H = T + sum(rh + (18 if g else 0) for g, _l, _b in rows) + 44 + (22 if note else 0)
    pw = W - L - R
    sx = lambda v: L + (max(scale[0], min(scale[1], v)) - scale[0]) / (scale[1] - scale[0]) * pw   # noqa: E731
    body = []
    for t in ticks:
        body += [M._line(sx(t), T - 14, sx(t), H - 44 - (22 if note else 0), SLATE, 2 if t in (0, 1.0) else 1,
                         op=.5 if t in (0, 1.0) else .12),
                 M._txt(sx(t), H - 24 - (22 if note else 0), tick_fmt(t), "middle")]
    y = T
    for group, label, bars in rows:
        if group:
            y += 10
            body.append(M._txt(0, y + 4, group, "start", "panel"))
            y += 8
        for k, (v, ci, col, txt) in enumerate(bars):
            yy = y + k * (bh + (6 if len(bars) > 2 else 2))
            if v is not None:
                x0, x1 = sorted((sx(0), sx(v)))
                body.append(f'<rect x="{x0:.1f}" y="{yy:.1f}" width="{x1 - x0:.1f}" height="{bh}" fill="{col}"/>')
                if ci:
                    body.append(M._line(sx(ci[0]), yy + bh / 2, sx(ci[1]), yy + bh / 2, ONYX, 1.5, op=.7))
            body.append(M._txt(sx(scale[1]) + 8, yy + bh - 1, txt, "start", "note"))
        body.append(M._txt(L - 12, y + 6 + len(bars) * (bh + (6 if len(bars) > 2 else 2)) / 2, label, "end"))
        y += rh
    if note:
        body.append(M._txt(L, H - 8, note, "start", "note"))
    return M._svg(W, H, title, desc, body)


def _pc(v):
    return "—" if v is None else f"{100 * v:.0f}%"


def fig_steer(D):
    rows, last = [], None
    for role, key, lab in B_ROWS:
        g = ("Sellers (X54, layer 44)" if role == "seller" else f"Buyers (layer {D['B_layer']})") if role != last else ""
        last = role
        c = D["B"][role][key]
        bars = [(c[d]["share_moved"], c[d]["share_ci"], col,
                 f"{_pc(c[d]['share_moved'])} ({c[d]['points']:+.1f} pt)".replace("-", "−"))
                for d, col in (("x1+v", SLATE), ("x1e6-v", ORANGE))]
        rows.append((g, lab.split(": ", 1)[1], bars))
    return _bar_chart("Steering buyers: their own direction works, the seller's does not",
                      "Horizontal bars: share of the x1/x1e6 opening gap moved by steering real sampled openings at "
                      "alpha 1, for sellers (their own direction, non-money) and buyers (their own direction, the "
                      "seller's direction as is and at the buyer's size, non-money). Buyers' own direction moves about "
                      "100 percent both ways; the seller's direction 10 to 22 percent, like the non-money control.",
                      rows, note="Slate: ×1 + v (toward ×10⁶); orange: ×10⁶ − v (toward ×1). Lines: 95% CIs over 30 fixtures.")


def fig_cos(D):
    v = D["vectors"]
    c, sh = v["cos_seller_buyer"], v["cos_split_half"]
    rows = []
    for k, lab in (("prompt", "the prompt's 106 tokens after the limit"), ("reply", "the reply's 11 opening tokens"),
                   ("all", "all 117 tokens"), ("token_mean", "averaged over the tokens (one vector)")):
        rows.append(("", lab, [(c[k], None, ORANGE, f"{c[k]:.2f}  seller vs buyer"),
                               (sh["buyer"][k], None, SLATE, f"{sh['buyer'][k]:.2f}  buyer, two halves"),
                               (sh["seller"][k], None, GREY, f"{sh['seller'][k]:.2f}  seller, two halves")]))
    return _bar_chart(f"Two directions, not one (layer {v['layer']})",
                      "Horizontal bars: cosine similarity between the seller's and the buyer's x1e6 - x1 directions "
                      "(orange), against the same comparison between two halves of the fixtures within one role (slate "
                      "buyers, grey sellers). Within a role the halves agree at 0.99; across roles 0.43 to 0.55 per "
                      "token, 0.82 once averaged over tokens.",
                      rows, W=940, L=330, scale=(0, 1.0), ticks=(0, .25, .5, .75, 1.0), tick_fmt=lambda t: f"{t:.2f}", R=200,
                      note="Cosine of the 30 fixtures' mean differences, concatenated over the tokens of each group.")


def cards(json_path=JSON_PATH):
    if not os.path.exists(json_path):
        return []
    D = json.load(open(json_path))
    seller_e2 = json.load(open(MAIN_JSON)).get("e2") if os.path.exists(MAIN_JSON) else None
    fd, rows = D["e0"]["fidelity"], "–".join(D["story_rows"])
    b = D["B"]["buyer"]

    def card(title, tag, svg, caption, wide=True):
        return (f'<figure class="chart-card mi-card{" mi-wide" if wide else ""}"><div class="chart-title"><h3>{title}</h3>'
                f'<span>{tag} · {rows}</span></div><div class="chart-scroll" role="region" tabindex="0" aria-label="'
                f'{html.escape(title)} chart, scroll horizontally on small screens">{svg}</div><figcaption>{caption}'
                "</figcaption></figure>")
    neg = f"{fd['shift_pod']:+.1f}".replace("-", "−")
    return [
        card("Buyers: the same model, the same smaller timidity", "Buyers · E0 fidelity", fig_e0(D),
             f"Gemma 4 31B as the buyer, same prompts and seeds as the vLLM run. Opening discount below the ceiling: "
             f"×1 {fd['pod_D2']:.1f}% (vLLM {fd['vllm_D2']:.1f}), ×10⁶ {fd['pod_D8']:.1f}% (vLLM {fd['vllm_D8']:.1f}); "
             f"shift {neg} points, negative in {fd['fixtures_negative']} of {fd['fixtures']} fixtures. Single fixtures "
             "scatter because buyers barely differ between fixtures (every ×1 opening is a round ten); vLLM’s own "
             "repeat samples agree no better.", wide=False),
        card("Buyers: the effect enters at the same layers", "Buyers · E2 patching", fig_e2(D, seller_e2),
             f"Every second layer, {D['e2']['n_fixtures']} fixtures, greedy price after a forced reply opening. Swapping "
             "all text after the limit restores about all of the buyer’s gap from layer 42, as for sellers (dashed). "
             "Unlike sellers, the last token alone restores 55–70% from layer 42. Before layer 40 it is less clean than "
             "for sellers: ×10⁶ into ×1 already restores most of the gap from layer 28, ×1 into ×10⁶ only briefly at "
             "26–28. Cells with more than 3 of 30 prices off scale are not drawn (×)."),
        card("Steering buyers: their own direction, not the seller’s", "Buyers · steering", fig_steer(D),
             f"Real sampled openings, the direction added to the prompt after the limit and to the reply’s 11 opening "
             f"tokens at layer {D['B_layer']}, α = 1, 4 openings per row, 100% valid. The buyer’s own direction (from the "
             f"other 29 fixtures) moves the buyer {_pc(b['v_pos|prompt+reply']['x1+v']['share_moved'])} and "
             f"{_pc(b['v_pos|prompt+reply']['x1e6-v']['share_moved'])} of the way. The seller’s direction, same layer "
             "and size, moves it about as little as a non-money magnitude direction. The buyer’s whole gap is about 8 "
             "points, the seller’s 21, so the points are shown too."),
        card("Two directions, partly shared", "Buyers · cross-role", fig_cos(D),
             "The seller’s and buyer’s ×10⁶ − ×1 directions at the same layer and token positions (111 of 117 are the "
             "same token). Each role’s direction is stable (two halves of the fixtures agree at 0.99), but the two "
             "roles agree only at 0.43–0.55 per token. Averaged over tokens they agree at 0.82, but that averaged "
             "vector steers buyers only 24–31%."),
    ]


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", help="a buyers run folder (data/interp/interp-buyers-...)")
    ap.add_argument("--preview")
    a = ap.parse_args(argv)
    if a.refresh:
        refresh(a.refresh)
        print(f"wrote {os.path.relpath(JSON_PATH, ROOT)}")
    if a.preview:
        M = _h()
        with open(a.preview, "w") as fh:
            fh.write('<!doctype html><meta charset="utf-8"><link rel="stylesheet" href="research.css">'
                     f"<style>{M.CSS}body{{background:#FFFFF0;padding:30px}}</style>" + "".join(cards()))
        print(f"wrote {a.preview}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
