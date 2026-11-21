#!/usr/bin/env python3
"""interp_e3c_figures.py (LAPTOP-SIDE): the E3c layer sweep's figure for the research website. The plug-in
scripts/interp/mi_figures_e3c.py is a shim importing cards() from here (keeps the pod's create request small).

    python3 scripts/interp_e3c_figures.py --refresh data/interp/<e3c run>   run -> config/interp/mi_figures_e3c.json
    python3 scripts/interp_e3c_figures.py --preview F                       standalone HTML with the card

cards() (called by mi_figures.extra_cards) reads only the tracked JSON, so the site builds from the repository;
no JSON yet -> no card. Standard library only. Exploratory.
"""
import argparse
import html
import json
import os
import sys

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "interp")     # scripts/interp (mi_figures.py)
ROOT = os.path.dirname(os.path.dirname(HERE))
JSON_PATH = os.path.join(ROOT, "config", "interp", "mi_figures_e3c.json")
MAIN_JSON = os.path.join(ROOT, "config", "interp", "mi_figures.json")
SLATE, ORANGE, SAGE, ONYX, IVORY, GREY = "#2C3E50", "#F39C12", "#87A96B", "#0F0F0F", "#FFFFF0", "#8A8F94"
KEYS = (("v_pos|all|1.0", "shared direction, prompt + reply", SLATE, 4, None),
        ("v_pos|reply|1.0", "reply opening only", ORANGE, 3, None),
        ("v_pos|prompt|1.0", "prompt only", SAGE, 3, "7 5"),
        ("nonmoney|all|1.0", "non-money control", GREY, 2.5, "2 5"))
PANELS = (("x1+v", "×1 prompt + v (toward ×10⁶)", "x1e6->x1"), ("x1e6-v", "×10⁶ prompt − v (toward ×1)", "x1->x1e6"))
OFF_SHARE_MAX = 0.10


def refresh(run_dir, story_row="X55"):
    s = json.load(open(os.path.join(run_dir, "summary.json")))
    A, B = s["A"], s["B"]
    out = {"run": os.path.basename(run_dir.rstrip("/")), "story_row": story_row, "model": "gemma-4-31b-it",
           "role": "seller", "n_fixtures_A": A["n_fixtures"], "n_fixtures_B": B["n_fixtures"], "layers": A["layers"],
           "b_pick": s.get("b_pick"), "A": {}, "B": {}}
    for d, cells in A["cells"].items():
        out["A"][d] = {k: [dict(layer=int(L), **{f: c.get(f) for f in ("fraction", "ci", "off_scale", "n")})
                           for L, c in sorted(v.items(), key=lambda kv: int(kv[0]))] for k, v in cells.items()}
    for d, cells in B["share"].items():
        out["B"][d] = []
        for k, c in cells.items():
            L, a = k.split("|")
            g = B["cells"]["directions"][d].get(f"v_pos@{L}|prompt+reply", {}).get(a, {})
            out["B"][d].append({"layer": int(L), "alpha": float(a), "share": c["share"], "ci": c["ci"],
                                "n_fixtures": c["n_fixtures"], "gap": g.get("mean_gap"), "valid": g.get("valid"),
                                "n": g.get("n"), "aligned_share": g.get("aligned_share")})
        base = B["cells"]["directions"][d].get("none", {}).get("0.0", {})
        out.setdefault("B_unsteered_gap", {})[d] = base.get("mean_gap")
    v = s.get("vectors") or {}
    if v:
        out["vectors"] = {k: v["per_layer"][k] for k in ("norm_v_prompt", "norm_v_reply", "resid_norm_prompt",
                                                         "resid_norm_reply", "cos_fixture_diffs_mean", "cos_v_mean_u")}
    with open(JSON_PATH, "w") as fh:
        json.dump(out, fh, indent=1)
    return out


def fig(D, e2=None):
    sys.path.insert(0, HERE)
    from mi_figures import _svg, _txt, _line, _path            # the main figures' SVG helpers and palette
    W, H = 940, 470
    pw, gap, L0, T, Bm = 395, 60, 84, 48, 128
    y0, y1 = -0.2, 1.4
    nL = 60
    body = []
    for i, (dn, title, e2key) in enumerate(PANELS):
        x0 = L0 + i * (pw + gap)
        sx = lambda l, x0=x0: x0 + l / (nL - 1) * pw                     # noqa: E731
        sy = lambda v: T + (y1 - max(y0, min(y1, v))) / (y1 - y0) * (H - T - Bm)   # noqa: E731
        body.append(_txt(x0, 24, title, "start", "panel"))
        for t in (0, .25, .5, .75, 1.0, 1.25):
            body.append(_line(x0, sy(t), x0 + pw, sy(t), SLATE, 2 if t in (0, 1.0) else 1, op=.5 if t in (0, 1.0) else .12))
            if i == 0:
                body.append(_txt(x0 - 10, sy(t) + 5, f"{t * 100:.0f}%", "end"))
        for l in range(0, nL, 10):
            body.append(_txt(sx(l), H - Bm + 22, str(l), "middle"))
        if e2:
            cells = e2["directions"][e2key]["post"]
            pts = [None if c["fraction"] is None or c["off_scale"] > 3 else (sx(c["layer"]), sy(c["fraction"])) for c in cells]
            body.append(_path(pts, ONYX, 2, op=.3))
        main = D["A"].get(dn, {}).get(KEYS[0][0], [])
        segs, cur = [], []                                     # the CI band, broken where a layer is not drawn
        for c in main:
            if c.get("ci") and c["fraction"] is not None and c["off_scale"] <= OFF_SHARE_MAX * c["n"]:
                cur.append(c)
            else:
                segs, cur = segs + [cur], []
        for band in segs + [cur]:
            if len(band) > 1:
                up = " ".join(f"{sx(c['layer']):.1f},{sy(c['ci'][1]):.1f}" for c in band)
                dn_ = " ".join(f"{sx(c['layer']):.1f},{sy(c['ci'][0]):.1f}" for c in reversed(band))
                body.append(f'<polygon points="{up} {dn_}" fill="{SLATE}" fill-opacity=".12"/>')
        for key, _lab, col, w, dash in reversed(KEYS):
            cells = D["A"].get(dn, {}).get(key, [])
            pts = [None if c["fraction"] is None or c["off_scale"] > OFF_SHARE_MAX * c["n"]
                   else (sx(c["layer"]), sy(c["fraction"])) for c in cells]
            body.append(_path(pts, col, w, dash))
            for c in cells:
                if c["fraction"] is not None and c["off_scale"] > OFF_SHARE_MAX * c["n"]:
                    body.append(_txt(sx(c["layer"]), sy(y0) - 4, "×", "middle", "note", fill=col))
        for p in D["B"].get(dn, []):
            if p["share"] is None:
                continue
            x, y = sx(p["layer"]), sy(p["share"])
            if p.get("ci"):
                body.append(_line(x, sy(p["ci"][0]), x, sy(p["ci"][1]), ONYX, 2, op=.6))
            fill = ONYX if p["alpha"] == 1.0 else IVORY
            body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{fill}" stroke="{ONYX}" stroke-width="2.5"/>')
        body.append(_txt(x0 + pw / 2, H - Bm + 46, "layer steered (0 = first, 59 = last)", "middle", "axis"))
    leg = [(col, w, dash, lab) for _k, lab, col, w, dash in KEYS] + ([(ONYX, 2, None, "E2: each fixture's own swap")] if e2 else [])
    lx, ly = L0, H - 52
    for j, (col, w, dash, lab) in enumerate(leg):
        if j == 3:
            lx, ly = L0, H - 26
        body += [_line(lx, ly, lx + 34, ly, col, w, dash, op=.3 if lab.startswith("E2") else 1.0), _txt(lx + 44, ly + 5, lab)]
        lx += 60 + 7.6 * len(lab)
    body += [f'<circle cx="{lx + 6}" cy="{ly}" r="6" fill="{ONYX}"/>', _txt(lx + 18, ly + 5, "real openings α 1"),
             f'<circle cx="{lx + 170}" cy="{ly}" r="6" fill="{IVORY}" stroke="{ONYX}" stroke-width="2.5"/>',
             _txt(lx + 182, ly + 5, "α 2"),
             f'<text transform="translate(16,{(T + H - Bm) / 2}) rotate(-90)" text-anchor="middle" class="axis">'
             "share of the ×1 / ×10⁶ gap moved</text>"]
    return _svg(W, H, "E3c: where the shared direction steers",
                "Two line charts over the layers, one per direction. Lines: share of the seller's x1/x1e6 opening gap moved by "
                "steering one layer with the shared direction (prompt and reply opening, reply only, prompt only) or the "
                "non-money control, in E2's forced-opening setup; dots: real sampled openings at three layers.", body)


def caption(D):
    pk = D.get("b_pick") or {}
    lays = ", ".join(str(x) for x in (pk.get("layers") or []))
    return (f"Gemma 4 31B, sellers. Lines: E2’s setup (reply opening forced, greedy price), {D['n_fixtures_A']} fixtures, "
            f"α = 1, one layer steered at a time (every second layer, plus 59); band: 95% CI over fixtures for the "
            f"prompt + reply line. Dots: real sampled openings, {D['n_fixtures_B']} fixtures, at layers {lays} "
            "(picked automatically: the earliest and last layers moving ≥ 50% both ways, and the peak); bars: 95% CIs. "
            "Off-scale prices are excluded; layers where more than 10% were off scale are not drawn (×): at 28–36, ×1 + v "
            "made about 1000× prices. The non-money line is clipped at −20%: at layers 32–38 it moves the opening the "
            "opposite way (to " + " and ".join(f"{100 * min(c['fraction'] for c in D['A'][d]['nonmoney|all|1.0'] if c['fraction'] is not None):.0f}%".replace("-", "−")
                                           for d in ("x1+v", "x1e6-v")) + "). Faint line: E2, each fixture’s own swap of all text after the limit.")


def cards(json_path=JSON_PATH):
    if not os.path.exists(json_path):
        return []
    D = json.load(open(json_path))
    e2 = json.load(open(MAIN_JSON)).get("e2") if os.path.exists(MAIN_JSON) else None
    title = "Which layers carry it? One layer at a time."
    return [f'<figure class="chart-card mi-card mi-wide"><div class="chart-title"><h3>{title}</h3>'
            f'<span>E3c · layer sweep · {D["story_row"]}</span></div><div class="chart-scroll" role="region" tabindex="0" '
            f'aria-label="{html.escape(title)} chart, scroll horizontally on small screens">{fig(D, e2)}</div>'
            f"<figcaption>{caption(D)}</figcaption></figure>"]


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", metavar="RUN_DIR")
    ap.add_argument("--story-row", default="X55")
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
