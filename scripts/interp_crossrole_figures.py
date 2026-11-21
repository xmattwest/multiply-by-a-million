#!/usr/bin/env python3
"""interp_crossrole_figures.py (LAPTOP-SIDE): the cross-role (seller + buyer) synthesis card for the research
website (X56). The plug-in scripts/interp/mi_figures_crossrole.py is a shim importing cards() from here, so no
laptop-only code is packed into the pod's create request.

This card has no run of its own: it re-uses the numbers the seller sweep (X55) and the buyers session
(X58-X59) already tracked. --refresh copies the series it needs out of the three sibling tracked JSONs into
its own tracked config/interp/mi_figures_crossrole.json, so the site still builds from the repository alone.

    python3 scripts/interp_crossrole_figures.py --refresh      rebuild from the sibling tracked JSONs
    python3 scripts/interp_crossrole_figures.py --preview F     standalone HTML with the card

cards() reads only the tracked JSON (no JSON -> no card). Standard library only. Exploratory.
"""
import argparse
import html
import json
import os
import sys

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "interp")      # scripts/interp (mi_figures.py)
ROOT = os.path.dirname(os.path.dirname(HERE))
CFG = os.path.join(ROOT, "config", "interp")
JSON_PATH = os.path.join(CFG, "mi_figures_crossrole.json")
MAIN_JSON = os.path.join(CFG, "mi_figures.json")               # seller E2 (post) by layer
BUYERS_JSON = os.path.join(CFG, "mi_figures_buyers.json")      # buyer E2 (post) + buyer B at layer 44
E3C_JSON = os.path.join(CFG, "mi_figures_e3c.json")            # seller steering share by layer (A, prompt+reply)
SLATE, ORANGE, SAGE, ONYX, IVORY, GREY = "#2C3E50", "#F39C12", "#87A96B", "#0F0F0F", "#FFFFF0", "#8A8F94"
DIRS = ("x1->x1e6", "x1e6->x1")                                # E2 keys
STEER_DIRS = ("x1+v", "x1e6-v")                                # steering keys (same two directions)


def _post(e2, dk):
    """E2 'post' (all text after the limit) restore share by layer, for one direction, with off-scale counts."""
    return [{"layer": c["layer"], "fraction": c["fraction"], "ci": c.get("ci"),
             "off_scale": c.get("off_scale", 0), "n": c.get("n", 30)} for c in e2["directions"][dk]["post"]]


def refresh():
    """Copy the needed series out of the three sibling tracked JSONs. No run folder is read."""
    main = json.load(open(MAIN_JSON))
    buy = json.load(open(BUYERS_JSON))
    e3c = json.load(open(E3C_JSON))
    out = {
        "story_row": "X56",
        "model": "gemma-4-31b-it",
        "sources": {"seller_e2": os.path.basename(MAIN_JSON), "buyer": os.path.basename(BUYERS_JSON),
                    "seller_sweep": os.path.basename(E3C_JSON)},
        "seller_sweep_run": e3c.get("run"),
        "buyer_run": buy.get("run"),
        "steer_layer": buy.get("B_layer", 44),
        # Panel A: E2 'post' restore by layer, both roles, both directions (a->b = a's activations into b's run).
        "e2": {"seller": {dk: _post(main["e2"], dk) for dk in DIRS},
               "buyer": {dk: _post(buy["e2"], dk) for dk in DIRS}},
        # Panel B: real sampled openings at layer 44, both steering directions. seller_on_seller is the seller's
        # own E3b run (X54) as carried in the buyers JSON; the other three are the buyers session (X59).
        "lever": {
            "buyer_on_buyer": {sd: buy["B"]["buyer"]["v_pos|prompt+reply"][sd] for sd in STEER_DIRS},
            "seller_on_seller": {sd: buy["B"]["seller"]["v_pos|prompt+reply"][sd] for sd in STEER_DIRS},
            "seller_on_buyer": {sd: buy["B"]["buyer"]["seller|prompt+reply"][sd] for sd in STEER_DIRS},
            "nonmoney_on_buyer": {sd: buy["B"]["buyer"]["nonmoney|prompt+reply"][sd] for sd in STEER_DIRS},
        },
        "cos": buy["vectors"]["cos_seller_buyer"],
        "cos_split_within": {r: buy["vectors"]["cos_split_half"][r]["all"] for r in ("seller", "buyer")},
    }
    with open(JSON_PATH, "w") as fh:
        json.dump(out, fh, indent=1)
    return out


OFF_SHARE_MAX = 0.10     # a layer is not drawn if more than 10% of its prices were off scale (marked x)
A_LINES = (("seller", "x1e6->x1", SLATE, 4, None), ("buyer", "x1e6->x1", ORANGE, 3.5, None),
           ("seller", "x1->x1e6", SLATE, 3, "7 5"), ("buyer", "x1->x1e6", ORANGE, 3, "7 5"))


def _panelA(D, helpers):
    _txt, _line, _path = helpers
    x0, pw, T, Bm, H = 84, 400, 56, 150, 500
    nL = 60
    y0, y1 = -0.2, 1.3
    sx = lambda l: x0 + l / (nL - 1) * pw                                     # noqa: E731
    sy = lambda v: T + (y1 - max(y0, min(y1, v))) / (y1 - y0) * (H - T - Bm)  # noqa: E731
    body = [_txt(x0, 24, "A. where it enters: E2, swap all text after the limit", "start", "panel")]
    for t in (0, .5, 1.0):
        body.append(_line(x0, sy(t), x0 + pw, sy(t), SLATE, 2 if t in (0, 1.0) else 1,
                          op=.5 if t in (0, 1.0) else .12))
        body.append(_txt(x0 - 10, sy(t) + 5, f"{t * 100:.0f}%", "end"))
    for l in range(0, nL, 10):
        body.append(_txt(sx(l), H - Bm + 22, str(l), "middle"))
    body.append(_line(sx(41), sy(y0), sx(41), sy(y1), ONYX, 2, "3 4", op=.5))
    body.append(_txt(sx(41) + 5, sy(y1) + 2, "layer 41", "start", "note"))
    for role, dk, col, w, dash in A_LINES:
        cells = D["e2"][role][dk]
        bad = lambda c: c["fraction"] is None or c["off_scale"] > OFF_SHARE_MAX * c["n"]   # noqa: E731
        body.append(_path([None if bad(c) else (sx(c["layer"]), sy(c["fraction"])) for c in cells], col, w, dash))
        for c in cells:
            if c["fraction"] is not None and bad(c):
                body.append(_txt(sx(c["layer"]), sy(y0) - 4, "×", "middle", "note", fill=col))
    body.append(_txt(x0 + pw / 2, H - Bm + 46, "layer patched (0 = first, 59 = last)", "middle", "axis"))
    leg = [(SLATE, 4, None, "seller"), (ORANGE, 3.5, None, "buyer"),
           (ONYX, 3, None, "×10⁶ into the ×1 run"), (ONYX, 3, "7 5", "×1 into the ×10⁶ run")]
    lx, ly = x0, H - 62
    for j, (col, w, dash, lab) in enumerate(leg):
        if j == 2:
            lx, ly = x0, H - 36
        body += [_line(lx, ly, lx + 34, ly, col, w, dash), _txt(lx + 44, ly + 5, lab)]
        lx += 70 + 7.6 * len(lab)
    return body, (x0, pw, T, Bm, H)


def _panelB(D, helpers, geom):
    _txt, _line, _path = helpers
    x0p, pw, T, Bm, H = geom
    x0 = x0p + pw + 100
    bw = 380
    y0, y1 = -0.2, 1.3
    sy = lambda v: T + (y1 - max(y0, min(y1, v))) / (y1 - y0) * (H - T - Bm)   # noqa: E731
    body = [_txt(x0, 24, "B. steering at layer 44, real openings", "start", "panel")]
    for t in (0, .5, 1.0):
        body.append(_line(x0, sy(t), x0 + bw, sy(t), SLATE, 2 if t in (0, 1.0) else 1,
                          op=.5 if t in (0, 1.0) else .12))
        body.append(_txt(x0 - 10, sy(t) + 5, f"{t * 100:.0f}%", "end"))
    groups = (("seller_on_seller", "seller v", "on sellers", SLATE),
              ("buyer_on_buyer", "buyer v", "on buyers", ORANGE),
              ("seller_on_buyer", "seller v", "on buyers", SAGE),
              ("nonmoney_on_buyer", "non-money", "on buyers", GREY))
    gw = bw / len(groups)
    for i, (key, l1, l2, col) in enumerate(groups):
        for j, sd in enumerate(STEER_DIRS):
            c = D["lever"][key][sd]
            m = c["share_moved"]
            x = x0 + i * gw + 14 + j * (gw * 0.36 + 4)
            wdt = gw * 0.36
            top, bot = sy(max(m, 0)), sy(min(m, 0))
            body.append(f'<rect x="{x:.1f}" y="{top:.1f}" width="{wdt:.1f}" height="{abs(bot - top):.1f}" '
                        f'fill="{col}" fill-opacity="{0.9 if j == 0 else 0.45}"/>')
            if c.get("share_ci"):
                body.append(_line(x + wdt / 2, sy(c["share_ci"][0]), x + wdt / 2, sy(c["share_ci"][1]), ONYX, 2, op=.6))
            body.append(_txt(x + wdt / 2, sy(max(m, c["share_ci"][1] if c.get("share_ci") else m)) - 6,
                             f"{m * 100:.0f}", "middle", "note"))
        cx = x0 + i * gw + 14 + gw * 0.38
        body += [_txt(cx, H - Bm + 22, l1, "middle"), _txt(cx, H - Bm + 40, l2, "middle")]
    body += [f'<rect x="{x0}" y="{H - 70}" width="16" height="12" fill="{ONYX}" fill-opacity=".9"/>',
             _txt(x0 + 24, H - 60, "×1 + v (toward ×10⁶)"),
             f'<rect x="{x0 + 200}" y="{H - 70}" width="16" height="12" fill="{ONYX}" fill-opacity=".45"/>',
             _txt(x0 + 224, H - 60, "×10⁶ − v (toward ×1)")]
    cos = D["cos"]
    body.append(_txt(x0, H - 30, f"cos(seller v, buyer v) per token: {cos['prompt']:.2f} prompt, {cos['reply']:.2f} reply;"
                                 f" within a role {D['cos_split_within']['seller']:.2f}", "start", "note"))
    return body


def fig(D):
    sys.path.insert(0, HERE)
    from mi_figures import _svg, _txt, _line, _path
    helpers = (_txt, _line, _path)
    W, H = 1000, 500
    bodyA, geom = _panelA(D, helpers)
    body = bodyA + _panelB(D, helpers, geom)
    T, Bm = geom[2], geom[3]
    body.append(f'<text transform="translate(16,{(T + H - Bm) / 2}) rotate(-90)" text-anchor="middle" '
                'class="axis">share of the ×1 / ×10⁶ gap</text>')
    return _svg(W, H, "Sellers and buyers: same layers, role-dependent vectors",
                "Two panels. A: the share of each role's x1/x1e6 opening gap restored by swapping all text after the "
                "limit (E2), by layer, in both directions. For both roles, x1e6 activations into the x1 run work from "
                "about layer 18, and x1 activations into the x1e6 run only from layer 41-42. B: the share of the gap "
                "moved in real openings by steering at layer 44, both directions, for each role's own direction, the "
                "seller's direction on buyers, and the non-money control.", body)


def caption(D):
    L = D.get("steer_layer", 44)
    cos = D["cos"]
    return ("Gemma 4 31B. A: E2, the share of each role’s ×1/×10⁶ opening gap restored by swapping in the other "
            "scale’s activations over all text after the limit (sellers every layer, X52; buyers every second layer, "
            "X58). Both roles have the same shape. ×10⁶ activations in the ×1 run work from about layer 18, about "
            "100% from 28. ×1 activations in the ×10⁶ run work only from 41–42. Layers where more than 10% of prices "
            "were off scale are not drawn (×): at 28–36 the seller swap also carries the number’s size. "
            f"B: real sampled openings steered at layer {L} on the prompt and the reply’s opening; bars are both "
            "directions, with 95% CIs over 30 fixtures. Sellers come from E3b (X54), buyers from X59. The "
            "seller’s direction moves buyers about as little as the non-money control. Per token, the seller "
            f"and buyer directions agree at cos {cos['prompt']:.2f} (prompt) / {cos['reply']:.2f} (reply), against "
            f"{D['cos_split_within']['seller']:.2f} between two halves of one role’s fixtures. The buyer gap is "
            "8 points against the seller’s 21, so buyer shares are noisier. Exploratory; one model; built from "
            "the X52, X54 and X58–X59 runs.")


def cards(json_path=JSON_PATH):
    if not os.path.exists(json_path):
        return []
    D = json.load(open(json_path))
    title = "Sellers and buyers: same layers, role-dependent vectors"
    return [f'<figure class="chart-card mi-card mi-wide"><div class="chart-title"><h3>{title}</h3>'
            f'<span>synthesis · {D["story_row"]}</span></div><div class="chart-scroll" role="region" '
            f'tabindex="0" aria-label="{html.escape(title)} chart, scroll horizontally on small screens">'
            f"{fig(D)}</div><figcaption>{caption(D)}</figcaption></figure>"]


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--preview")
    a = ap.parse_args(argv)
    if a.refresh:
        refresh()
        print(f"wrote {os.path.relpath(JSON_PATH, ROOT)}")
    if a.preview:
        sys.path.insert(0, HERE)
        from mi_figures import CSS
        with open(a.preview, "w") as fh:
            fh.write('<!doctype html><meta charset="utf-8">'
                     f"<style>{CSS}body{{background:#FFFFF0;padding:30px}}</style>{''.join(cards())}")
        print(f"wrote {a.preview}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    sys.exit(main(sys.argv[1:]))
