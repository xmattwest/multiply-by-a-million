#!/usr/bin/env python3
"""Recompose the five main-paper figures from existing local evidence, offline.

    python3 manuscript/figures/main_figures.py
    python3 manuscript/figures/main_figures.py --check

No model calls or new experiments. The original generators and figures remain
available. Registered opening and midpoint values are reconstructed with their
existing checked loaders; open-model closing CIs are checked with the frozen
bootstrap seeds. Mechanistic summaries are reconstructed by fig7_interp.load.
Exploratory bot contrasts are read, unchanged, through fig5_map.contrast.

Final widths: main_game 3.25 in; other figures 6.75 in. Main labels are at
least 8 pt at those widths; appendix panels retain the original >=7 pt labels.
"""
import argparse
import json
import math
import os
import random
import statistics
import subprocess
import sys
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from figstyle import (BAND, GREY, LIGHT, MODEL, ONYX, ORANGE_DARK, ROOT, SAGE_TEXT, SLATE,
                      line, svg_doc, text, write)
import addendum_c as C
import fig1_game as G
import fig5_map as P
import fig7_interp as I
import fig8_models as O

S2 = {
    "qwen3.8-27b-fp8": "s2-qwen3.8-27b-fp8-20261003T223042Z",
    "gemma-4-31b-it": "s2-gemma-4-31b-it-20261004T012932Z",
}
ORDER = tuple(S2) + C.ARMS
SHORT = dict(zip(C.ARMS, ("Luna", "Haiku", "Gemini")))


def label(x, y, s, size=17, **kw):
    return text(x, y, s, size, **kw)


def point(arm, x, y, r=5.5, hollow=False):
    sty = MODEL[arm]
    return O.marker(sty.get("marker", "circle"), x, y, r, sty["col"],
                    "white" if hollow else sty.get("fill", sty["col"]), 2)


def cross(x, y, col=GREY, r=4):
    return line(x-r, y, x+r, y, col, 2) + line(x, y-r, x, y+r, col, 2)


def axis(out, left, right, bottom, lo, hi, ticks, top=None):
    X = lambda v: left + (v-lo)/(hi-lo)*(right-left)
    for v in ticks:
        if top is not None:
            out.append(line(X(v), top, X(v), bottom, LIGHT, 1))
        out += [line(X(v), bottom, X(v), bottom+5, ONYX, 1),
                label(X(v), bottom+25, f"{v:g}".replace("-", "−"), anchor="middle")]
    out.append(line(left, bottom, right, bottom, ONYX, 1.2))
    return X


def primary():
    results = {}
    for arm, run in S2.items():
        with open(os.path.join(ROOT, "derived", run, "primary.json")) as fh:
            p = json.load(fh)
        values = [d["float"] for d in p["d_i"].values()]
        C.close(statistics.fmean(values), p["mean_d"], arm + " primary mean")
        assert len(values) == p["fixtures_in_test"]
        ci = O.A.bootstrap_ci(values, random.Random(p["bootstrap"]["seed"]), levels=(0.90, 0.95))
        for level in (90, 95):
            for actual, expected in zip(ci[str(level)], p["bootstrap"][f"ci{level}_mean"]):
                C.close(actual, expected, arm + f" CI{level}")
        results[arm] = dict(mean=p["mean_d"], ci90=p["bootstrap"]["ci90_mean"],
                            ci95=p["bootstrap"]["ci95_mean"], n=p["fixtures_in_test"],
                            equivalent=p["H2_equivalent_within_0.10"])
    assert results["qwen3.8-27b-fp8"]["equivalent"]
    assert round(results["qwen3.8-27b-fp8"]["ci90"][1], 3) == 0.099
    return results


def game_figure(games):
    small = games["F19-x1e2-S"]
    big = games["F19-x1e8-S"]
    assert (small["F"], small["C"], small["price"]) == (83, 118, 107)
    assert (big["F"], big["C"], big["price"]) == (83000000, 118000000, 87000000)
    out = [label(20, 27, "One bargain, rescaled", 22, weight="bold")]
    for left, role, frac, lim, col in ((20, "Seller", "F = 0.83 V", "private floor", SLATE),
                                     (270, "Buyer", "C = 1.18 V", "private ceiling", SAGE_TEXT)):
        out += [f"<rect x='{left}' y='44' width='210' height='81' rx='6' fill='white' stroke='{col}' stroke-width='2'/>",
                label(left+105, 68, role, 20, fill=col, weight="bold", anchor="middle"),
                label(left+105, 92, lim, 18, anchor="middle"),
                label(left+105, 115, frac, 18, anchor="middle")]
    out += [line(235, 83, 263, 83, GREY, 2),
            label(250, 152, "Same model; alternating JSON moves", 18, anchor="middle"),
            label(20, 189, "Scale", 18, weight="bold"), label(172, 189, "Opens", 18, weight="bold", anchor="middle"),
            label(294, 189, "Settles", 18, weight="bold", anchor="middle"),
            label(425, 189, "Seller share", 18, weight="bold", anchor="middle"), line(20, 198, 480, 198, GREY)]
    for g, y, scale, divisor, suffix in ((small, 229, "$100", 1, ""), (big, 269, "$100M", 1000000, "M")):
        first = g["offers"][0][2]
        share = (g["price"]-g["F"])/(g["C"]-g["F"])
        out += [label(20, y, scale, 19), label(172, y, f"${first//divisor}{suffix}", 19, anchor="middle"),
                label(294, y, f"${g['price']//divisor}{suffix}", 19, anchor="middle"),
                label(425, y, f"{share:.0%}", 19, weight="bold", anchor="middle")]
    out += [line(20, 284, 480, 284, GREY), label(250, 313, "Illustration: F19, Gemma, seller opens", 18, fill=GREY, anchor="middle")]
    return svg_doc(500, 330, "Fixed-geometry bargaining and an illustrative pair",
                   "Source: Study 2 Gemma run s2-gemma-4-31b-it-20261004T012932Z, F19 seller-opened games. "
                   "The complete design also includes $100K and buyer openers. Initial prompt identity checked by fig1_game.load.", out)


def models_figure(openings, closing, c):
    out = [label(20, 28, "Five tested models", 20, weight="bold"),
           label(245, 28, "a  Opening gap", 20, weight="bold"),
           label(670, 28, "b  Closing seller share", 20, weight="bold"),
           label(245, 52, "95% CI · own-limit units", 17, fill=GREY),
           label(670, 52, "90% / 95% CI · surplus units", 17, fill=GREY)]
    top, bot = 77, 364
    Xh = axis(out, 245, 520, bot, -0.28, 0.02, (-0.25, -0.15, -0.05, 0), top)
    Xd = lambda v: 670 + (v+0.4)/0.55*250
    out.insert(0, f"<rect x='{Xd(-0.1)}' y='{top}' width='{Xd(0.1)-Xd(-0.1)}' height='{bot-top}' fill='{BAND}'/>")
    axis(out, 670, 920, bot, -0.4, 0.15, (-0.4, -0.2, 0, 0.1), top)
    for X in (Xh, Xd):
        out.append(line(X(0), top, X(0), bot, GREY, 1.1, "4 3"))
    for i, arm in enumerate(ORDER):
        y = 105 + 57*i
        o, d = openings[arm], closing[arm]
        col = MODEL[arm]["col"]
        status = "hypothesis source (S2)" if i == 0 else "A: confirmation" if i == 1 else "B: confirmation; + C"
        out += [label(20, y-2, MODEL[arm]["label"], 18, weight="bold"), label(20, y+19, status, 17, fill=GREY)]
        oy = y if i < 2 else y-8
        out += [line(Xh(o["ci"][0]), oy, Xh(o["ci"][1]), oy, col, 3),
                point(arm, Xh(o["mean_h"]), oy, hollow=i == 0),
                label(538, oy+6, C.signed(o["mean_h"]), 17, fill=col)]
        if arm in C.ARMS:
            h = c["per_model"][arm]["HA"]
            out += [line(Xh(h["ci95"][0]), y+14, Xh(h["ci95"][1]), y+14, GREY, 2),
                    cross(Xh(h["mean"]), y+14), label(538, y+20, C.signed(h["mean"]), 17, fill=GREY)]
        out += [line(Xd(d["ci95"][0]), y, Xd(d["ci95"][1]), y, col, 2),
                line(Xd(d["ci90"][0]), y, Xd(d["ci90"][1]), y, col, 6), point(arm, Xd(d["mean"]), y),
                label(990, y-8, C.signed(d["mean"]), 17, fill=col, weight="bold", anchor="end"),
                label(990, y+17, "equivalent" if d["equivalent"] else "H1 shift", 17, anchor="end")]
    out += [label(245, 417, "+  C: registered-secondary replication", 17, fill=GREY),
            label(670, 417, "band: ±0.10; Study 2 / C", 17, fill=GREY),
            label(20, 447, "$100M − $100. Qwen: 29 closing fixtures; all other rows: 30. Cross-model comparison: descriptive.", 17, fill=GREY)]
    return svg_doc(1000, 463, "Shared opening modesty and model-dependent closing splits",
                   "Opening sources: A (Gemma), B (API), Qwen Study 2 hypothesis source; C HA is secondary and shown as a grey plus. "
                   "Closing: frozen Study 2 primary.json and Addendum C H1/H2, all agreements. Qwen upper 90% bound +0.099. "
                   "Sources and exact values verified by fig8_models.load, main_figures.primary and addendum_c.load.", out)


def midpoint_figure(c, parts):
    out = [label(20, 25, "HM1: calibration within ±0.05", 20, weight="bold"),
           label(20, 206, "HM2: matched fixture shifts", 20, weight="bold")]
    vals = [float(v) for pairs in parts.values() for pair in pairs.values() for v in pair]
    lo, hi = math.floor((min(vals)-0.025)*5)/5, math.ceil((max(vals)+0.025)*5)/5
    for i, arm in enumerate(C.ARMS):
        left, w = 75+320*i, 235
        hm = c["per_model"][arm]["HM"]
        cal = hm["HM1"]
        X = lambda v: left+(v+0.1)/0.2*w
        out += [label(left+w/2, 55, arm, 18, weight="bold", anchor="middle"),
                f"<rect x='{X(-0.05)}' y='67' width='{X(0.05)-X(-0.05)}' height='45' fill='{BAND}'/>",
                line(X(0), 67, X(0), 112, GREY, 1, "4 3"),
                line(X(cal["ci90"][0]), 89, X(cal["ci90"][1]), 89, SAGE_TEXT, 3), point(arm, X(cal["mean"]), 89)]
        axis(out, left, left+w, 115, -0.1, 0.1, (-0.1, 0, 0.1))
        out += [label(left+w/2, 162, "HM1 passes" if cal["equivalent"] else "equivalence not established", 17,
                      fill=SAGE_TEXT if cal["equivalent"] else ONYX, anchor="middle")]
        size, top = 205, 251
        px = left+15
        X = lambda v: px+(v-lo)/(hi-lo)*size
        Y = lambda v: top+size-(v-lo)/(hi-lo)*size
        out.append(label(left+w/2, 234, f"Spearman {hm['HM2']['spearman']:.3f}", 18, anchor="middle"))
        ticks = [v for v in (-0.8, -0.4, 0, 0.4) if lo <= v <= hi]
        for tick in ticks:
            out += [line(X(tick), top, X(tick), top+size, LIGHT), line(px, Y(tick), px+size, Y(tick), LIGHT),
                    label(X(tick), top+size+24, f"{tick:g}".replace("-", "−"), anchor="middle")]
            if i == 0:
                out.append(label(px-9, Y(tick)+5, f"{tick:g}".replace("-", "−"), anchor="end"))
        out += [line(X(lo), Y(lo), X(hi), Y(hi), GREY, 1.5, "4 3"),
                line(px, top, px, top+size, ONYX), line(px, top+size, px+size, top+size, ONYX)]
        for ds, ms in parts[arm].values():
            out.append(point(arm, X(float(ms)), Y(float(ds)), 3.6))
        out += [label(left+w/2, 509, f"means: {C.signed(hm['midpoint_shift_mean'])} / {C.signed(hm['share_shift_mean_same_games'])}", 17, anchor="middle")]
    out += [label(20, 353, "closing-share shift", 17, anchor="middle", rotate=-90),
            label(505, 538, "opening-midpoint shift (surplus units); means shown opening / closing", 17, anchor="middle")]
    return svg_doc(1000, 553, "Registered midpoint calibration and tracking in Addendum C",
                   "Source: derived/addC-20261011T182548Z/addendum_C.json, HM. All plotted fixture quantities, means, CIs and "
                   "correlations checked by addendum_c.load. HM1 passes only on Luna; HM2 Holm p 0.00015 on all three. "
                   "Gemini's 76 excluded agreements are not substituted into the matched closing mean of -0.338.", out)


def probe_figure():
    rows = [("No addition", "prompt", "N", 109), ("Scale-neutral instruction", "prompt", "D", 153),
            ("No information", "info", "N", 238), ("Private appraisal", "info", "A", 282),
            ("Counterpart's range", "info", "P", 326)]
    out = [label(20, 28, "Exploratory bot comparisons", 20, weight="bold"),
           label(20, 80, "Instructions · P01–P30", 17, fill=GREY),
           label(20, 209, "Information · I01–I30", 17, fill=GREY)]
    for j, arm in enumerate(S2):
        left, right = 345+325*j, 600+325*j
        out.append(label((left+right)/2, 45, MODEL[arm]["label"], 20, fill=MODEL[arm]["col"], weight="bold", anchor="middle"))
        X = axis(out, left, right, 356, -0.35, 0.1, (-0.3, -0.15, 0), 65)
        out.append(line(X(0), 65, X(0), 356, GREY, 1, "4 3"))
        for lab, probe, cond, y in rows:
            mean, ci, n = P.contrast(probe, arm, f"round raw | BOT {cond} model=seller | share")
            assert n == 30 and -0.35 <= ci[0] <= mean <= ci[1] <= 0.1
            if j == 0:
                out.append(label(20, y+5, lab, 18))
            out += [line(X(ci[0]), y, X(ci[1]), y, MODEL[arm]["col"], 3), point(arm, X(mean), y)]
    out += [label(635, 411, "×10⁶ − ×1 change in the model-seller's own surplus share", 17, anchor="middle")]
    return svg_doc(1000, 430, "Instruction versus information against a scale-invariant bot",
                   "Source: fig5_map.RUN prompt and info runs for Qwen and Gemma; exact BOT model=seller share contrasts, "
                   "95% CIs and n read through fig5_map.contrast. Separate baselines for the P and I fixture sets. All exploratory.", out)


def mechanism_figure(e):
    out = [label(20, 28, "a  Forced-opening layer sweep", 20, weight="bold"),
           label(530, 28, "b  Real sampled openings", 20, weight="bold")]
    top, height, left, right = 78, 250, 76, 465
    X = lambda layer: left+layer/59*(right-left)
    Y = lambda val: top+height-(val+0.25)/1.5*height
    out.append(f"<rect x='{X(42)}' y='{top}' width='{X(48)-X(42)}' height='{height}' fill='{BAND}'/>")
    for value in (0, 0.5, 1):
        out += [line(left, Y(value), right, Y(value), GREY if value == 0 else LIGHT),
                label(left-10, Y(value)+5, f"{value:.0%}", anchor="end")]
    for direction, col in (("x1+v", SLATE), ("x1e6-v", ORANGE_DARK)):
        cells = e["A"]["cells"][direction]["v_pos|all|1.0"]
        coords = []
        for layer in sorted(map(int, cells)):
            cell = cells[str(layer)]
            assert -0.25 <= cell["fraction"] <= 1.25
            coords.append((X(layer), Y(cell["fraction"])))
        out.append("<polyline points='" + " ".join(f"{x:.1f},{y:.1f}" for x,y in coords) +
                   f"' fill='none' stroke='{col}' stroke-width='3'/>")
        for layer, (x, y) in zip(sorted(map(int, cells)), coords):
            if cells[str(layer)]["off_scale"] > .1*cells[str(layer)]["n"]:
                out.append(O.marker("circle", x, y, 4, col, "white", 2))
    axis(out, left, right, top+height, 0, 59, (0, 20, 40, 59))
    out += [label(275, 379, "layer (of 60)", 17, anchor="middle"),
            label(20, 206, "fraction of price gap moved", 17, anchor="middle", rotate=-90)]
    cells = e["B"]["cells"]["directions"]
    base = {d: cells[d]["none"]["0.0"]["mean_gap"] for d in ("x1+v", "x1e6-v")}
    x0, x1 = 610, 980
    X = lambda gap: x0+(gap-10)/50*(x1-x0)
    for d in base:
        out += [line(X(base[d]), 85, X(base[d]), 328, GREY, 1.2, "4 3"),
                label(X(base[d]), 67, f"base {base[d]:.1f}%", 17, fill=GREY, anchor="middle")]
    valid, total, off = 0, 0, 0
    for i, layer in enumerate((42, 44, 48)):
        for j, (direction, col) in enumerate((("x1+v", SLATE), ("x1e6-v", ORANGE_DARK))):
            y = 111+76*i+29*j
            if j == 0:
                out.append(label(540, y+17, f"L{layer}", 18, weight="bold"))
            a, b = [cells[direction][f"v_pos@{layer}|prompt+reply"][alpha] for alpha in ("1.0", "2.0")]
            for q in (a, b):
                assert q["n"] == 90
                valid += q["valid"]
                total += q["n"]
                off += q["off_scale"] + q["other_scale"]
            out += [line(X(base[direction]), y, X(a["mean_gap"]), y, col, 2),
                    O.marker("circle", X(a["mean_gap"]), y, 5.5, col, col, 2),
                    O.marker("circle", X(b["mean_gap"]), y, 5.5, col, "white", 2)]
    assert valid == total == 1080
    assert off == 6
    axis(out, x0, x1, 328, 10, 60, (10, 30, 50, 60))
    out += [label(795, 379, "seller markup (%)", 17, anchor="middle"),
            label(20, 411, "Blue: ×1 + v; ochre: ×10⁶ − v", 17),
            label(530, 411, "Filled: α = 1; open: α = 2", 17),
            label(20, 437, "Prompt + reply; leave-one-fixture-out. 30 fixtures. Sampled checks: 90 openings per point.", 17, fill=GREY)]
    return svg_doc(1000, 453, "Localisation and real-opening steering in Gemma",
                   f"Source: {I.E3C_RUN}; raw results.jsonl reconstructed and checked by fig7_interp.load. "
                   "Forced layer sweep and sampled openings use different readouts. All 1080 steered replies format-valid; "
                   "six double-strength prices off-scale, excluded from gap means. Baselines are the E3c run's own rows.", out)


def font_check(doc, width_inches, minimum):
    root = ET.fromstring(doc)
    width = float(root.attrib["viewBox"].split()[2])
    smallest = min(float(node.attrib["font-size"]) for node in root.iter() if "font-size" in node.attrib)
    points = smallest/width*width_inches*72
    assert points >= minimum, (points, minimum)
    return round(points, 2)


def build():
    opening_rows, _, _ = O.load()
    openings = {m["model"]: m for m in opening_rows}
    c, parts = C.load()
    closing = primary()
    for arm in C.ARMS:
        p = c["per_model"][arm]
        closing[arm] = {**p["H1"], "equivalent": p["H2_equivalent_within_0.10"]}
    assert sum(p["equivalent"] for p in closing.values()) == 2
    assert [c["per_model"][a]["HM"]["supported"] for a in C.ARMS] == [True, False, False]
    games, _, _ = G.load()
    e, concepts, _, _ = I.load()
    a, ya = I.panel_a(e, 34)
    b, yb = I.panel_b(e, ya+22)
    cc, yc = I.panel_c(concepts, 34)
    docs = {
        "main_game": game_figure(games), "main_models": models_figure(openings, closing, c),
        "main_midpoint": midpoint_figure(c, parts), "main_probes": probe_figure(),
        "main_mechanism": mechanism_figure(e),
        "app_steering_detail": svg_doc(1000, int(yb+12), "Full exploratory steering comparisons",
                                       f"Panels a and b of fig7_interp, source {I.E3C_RUN}; original checked values.", a+b),
        "app_concepts_detail": svg_doc(1000, int(yc+12), "Concept decomposition and failed ablation control",
                                       f"Panel c of fig7_interp, source {I.C2_RUN}; original checked values.", cc),
    }
    for stem, doc in docs.items():
        minimum = font_check(doc, 3.25 if stem == "main_game" else 6.75, 8 if stem.startswith("main_") else 7)
        print(f"{stem}: minimum label {minimum} pt at final width")
    return docs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="verify sources and print size without writing figures")
    args = ap.parse_args()
    docs = build()
    if not args.check:
        for stem, doc in docs.items():
            svg = write(os.path.join(HERE, stem+".svg"), doc)
            subprocess.run(["rsvg-convert", "-f", "pdf", "-o", os.path.join(HERE, stem+".pdf"), svg], check=True)
            print(svg)
    print("Registered fixture values/CIs and mechanistic summaries verified against existing sources.")


if __name__ == "__main__":
    main()
