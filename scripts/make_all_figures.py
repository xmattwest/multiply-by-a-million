"""make_all_figures.py: regenerate every figure (and, optionally, every analysis) from the raw data.

    python3 scripts/make_all_figures.py              figures only (uses existing derived/*/primary.json)
    python3 scripts/make_all_figures.py --analyze    re-run every analysis first, then the figures

What it runs, in order (each step is skipped, with a note, if its data isn't there):
  1. --analyze only: scripts/analyze.py on every CLOSED real run (data/runs/*, run_summary.json
     present), with its pilot (from the manifest), --secondary, and --addendum for Study 2.
     Runs still in progress are never analysed.
  2. scripts/make_figures.py for every real run with a primary.json      (per-run figures 1-3)
  3. scripts/explore_figures.py                                          (E1-E4, exploratory)
  4. scripts/analyze_probe.py + scripts/probe_figures.py for each A/N/LS/LB probe (P1-P3)
  5. scripts/cross_model_figures.py                                      (C1-C2 + Holm summary)
  6. derived/figures_index.json: every figure, its group, status, title and description (read
     from the SVG itself) and the command that made it. scripts/build_site.py shows it as a gallery.
  7. scripts/build_site.py
Nothing here spends money or talks to the network. Standard library only.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
from xml.sax.saxutils import unescape

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable
QWEN_SETS = ["data/runs/qwen3.8-27b-fp8-20261003T180835Z", "data/pilot/pilot-qwen3.8-27b-fp8-20261003T173737Z",
             "data/runs/s2-qwen3.8-27b-fp8-20261003T223042Z", "data/pilot/s2-pilot-qwen3.8-27b-fp8-20261003T220356Z"]
problems = []


def rel(p):
    return os.path.relpath(p, ROOT)


def run(args, label):
    print(f"• {label}")
    r = subprocess.run([PY] + args, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        problems.append(f"{label}: exit {r.returncode}: {(r.stderr or r.stdout).strip().splitlines()[-1:]}")
        print(f"  ✗ failed (exit {r.returncode})")
    return r.returncode == 0


def svg_meta(path):
    s = open(path, encoding="utf-8").read(4000)
    t = re.search(r"<title[^>]*>(.*?)</title>", s, re.S)
    d = re.search(r"<desc[^>]*>(.*?)</desc>", s, re.S)
    return (unescape(t.group(1)) if t else os.path.basename(path)), (unescape(d.group(1)) if d else "")


def figures_in(folder):
    return sorted(glob.glob(os.path.join(ROOT, folder, "*.svg")))


def main(argv):
    ap = argparse.ArgumentParser(description="Regenerate all figures (see the top of this file).")
    ap.add_argument("--analyze", action="store_true", help="re-run every analysis first (slower: minutes per run)")
    ap.add_argument("--no-site", action="store_true")
    a = ap.parse_args(argv)
    index = []

    real = []
    for m in sorted(glob.glob(os.path.join(ROOT, "data", "runs", "*", "run_manifest.json"))):
        man = json.load(open(m))
        if man.get("kind") == "real" and not man.get("preview"):
            real.append((os.path.dirname(m), man))

    if a.analyze:
        for d, man in real:
            if not os.path.exists(os.path.join(d, "run_summary.json")):
                print(f"• skip analysis of {rel(d)}: not closed (still running, or stopped early)")
                continue
            args = ["scripts/analyze.py", rel(d), "--secondary"]
            pilot = man.get("pilot_run_id")
            if pilot and os.path.isdir(os.path.join(ROOT, "data", "pilot", pilot)):
                args += ["--pilot", f"data/pilot/{pilot}"]
            if man.get("study", 1) == 2:
                args.append("--addendum")
            run(args, f"analyze {man['run_id']}")

    for d, man in real:
        rid = man["run_id"]
        if not os.path.exists(os.path.join(ROOT, "derived", rid, "primary.json")):
            print(f"• skip figures for {rid}: no primary.json yet (run with --analyze once the run is closed)")
            continue
        if run(["scripts/make_figures.py", rel(d)], f"per-run figures {rid}"):
            study = man.get("study", 1)
            index.append({"group": f"Study {study} · {man['model']} · per-run figures",
                          "status": "1 = pre-registered primary; 2–3 descriptive",
                          "command": f"python3 scripts/make_figures.py {rel(d)}",
                          "figures": [rel(f) for f in figures_in(f"derived/{rid}/figures")]})

    if all(os.path.isdir(os.path.join(ROOT, p)) for p in QWEN_SETS):
        if run(["scripts/explore_figures.py"], "exploratory figures E1–E4"):
            index.append({"group": "Exploratory sweep · Qwen, Study 1 and 2 (real + pilot)",
                          "status": "exploratory, not pre-registered", "command": "python3 scripts/explore_figures.py",
                          "figures": [rel(f) for f in figures_in("derived/explore/figures")]})
    else:
        print("• skip exploratory figures: a Qwen dataset is missing")

    for m in sorted(glob.glob(os.path.join(ROOT, "data", "explore", "X-*", "run_manifest.json"))):
        man = json.load(open(m))
        d = os.path.dirname(m)
        if man.get("preview") or sorted(man.get("conditions", {})) != ["A", "LB", "LS", "N"]:
            continue
        rid = man["run_id"]
        if run(["scripts/analyze_probe.py", rel(d)], f"probe analysis {rid}") and \
                run(["scripts/probe_figures.py", rel(d)], f"probe figures {rid}"):
            index.append({"group": f"Probe {man['probe']} · {man['model']} (appraisal on/off; hidden information)",
                          "status": "exploratory, not pre-registered",
                          "command": f"python3 scripts/probe_figures.py {rel(d)}",
                          "figures": [rel(f) for f in figures_in(f"derived/explore/{rid}/figures")]})

    for m in sorted(glob.glob(os.path.join(ROOT, "data", "explore", "X-digits-*", "run_manifest.json"))):
        man = json.load(open(m))
        d = os.path.dirname(m)
        if man.get("preview") or not os.path.exists(os.path.join(d, "run_summary.json")):
            continue
        rid = man["run_id"]
        if run(["scripts/analyze_digits_probe.py", rel(d)], f"digits x stakes probe {rid}"):
            index.append({"group": f"Probe digits x stakes · {man['model']} (first moves, fresh fixtures)",
                          "status": "exploratory, not pre-registered",
                          "command": f"python3 scripts/analyze_digits_probe.py {rel(d)}",
                          "figures": [rel(f) for f in figures_in(f"derived/explore/{rid}/figures")]})

    for m in sorted(glob.glob(os.path.join(ROOT, "data", "explore", "X-newgame-*", "run_manifest.json"))):
        man = json.load(open(m))
        d = os.path.dirname(m)
        if man.get("preview") or not os.path.exists(os.path.join(d, "run_summary.json")):
            continue
        rid = man["run_id"]
        if run(["scripts/analyze_game_probe.py", rel(d)], f"new-game probe {rid}"):
            index.append({"group": f"Probe new game · {man['model']} (redesigned ranges, notation, bot, appraisal)",
                          "status": "exploratory, not pre-registered",
                          "command": f"python3 scripts/analyze_game_probe.py {rel(d)}",
                          "figures": [rel(f) for f in figures_in(f"derived/explore/{rid}/figures")]})

    for m in sorted(glob.glob(os.path.join(ROOT, "data", "explore", "X-geom-*", "run_manifest.json"))):
        man = json.load(open(m))
        d = os.path.dirname(m)
        if man.get("preview") or not os.path.exists(os.path.join(d, "run_summary.json")):
            continue
        rid = man["run_id"]
        if run(["scripts/analyze_geom_probe.py", rel(d)], f"geometry x notation probe {rid}"):
            index.append({"group": f"Probe geometry x notation · {man['model']} (zone shape, price notation, bot)",
                          "status": "exploratory, not pre-registered",
                          "command": f"python3 scripts/analyze_geom_probe.py {rel(d)}",
                          "figures": [rel(f) for f in figures_in(f"derived/explore/{rid}/figures")]})

    for kind, title in (("prompt", "Probe prompt style"), ("cross", "Probe cross-play")):
        for m in sorted(glob.glob(os.path.join(ROOT, "data", "explore", f"X-{kind}-*", "run_manifest.json"))):
            man = json.load(open(m))
            d = os.path.dirname(m)
            if man.get("preview") or not os.path.exists(os.path.join(d, "run_summary.json")):
                continue
            rid = man["run_id"]
            if run(["scripts/analyze_cell_probe.py", rel(d)], f"{kind} probe {rid}"):
                index.append({"group": f"{title} · {man['model']}",
                              "status": "exploratory, not pre-registered",
                              "command": f"python3 scripts/analyze_cell_probe.py {rel(d)}",
                              "figures": [rel(f) for f in figures_in(f"derived/explore/{rid}/figures")]})

    for m in sorted(glob.glob(os.path.join(ROOT, "data", "explore", "X-info-*", "run_manifest.json"))):
        man = json.load(open(m))
        d = os.path.dirname(m)
        if man.get("preview") or not os.path.exists(os.path.join(d, "run_summary.json")):
            continue
        rid = man["run_id"]
        if run(["scripts/analyze_info_probe.py", rel(d)], f"info probe {rid}"):
            index.append({"group": f"Probe asymmetric information · {man['model']}",
                          "status": "exploratory, not pre-registered",
                          "command": f"python3 scripts/analyze_info_probe.py {rel(d)}",
                          "figures": [rel(f) for f in figures_in(f"derived/explore/{rid}/figures")]})

    models = sorted({json.load(open(m))["model"] for m in glob.glob(os.path.join(ROOT, "data", "explore", "X-digits-*", "run_manifest.json"))
                     if not json.load(open(m)).get("preview")
                     and os.path.exists(os.path.join(os.path.dirname(m), "run_summary.json"))})
    for model in models:
        if run(["scripts/digits_figures.py", "--model", model], f"digits x stakes, all runs combined · {model}"):
            index.append({"group": f"Probe digits x stakes · {model} · all wordings combined",
                          "status": "exploratory, not pre-registered",
                          "command": f"python3 scripts/digits_figures.py --model {model}",
                          "figures": [rel(f) for f in figures_in(f"derived/explore/digits_combined/{model}/figures")]})

    # WP1: concession-path decomposition across the X-probes and Study 2.
    conc_runs = []
    for pat in ("data/explore/X-info-*", "data/explore/X-geom-*", "data/explore/X-prompt-*",
                "data/explore/X-cross-*", "data/runs/s2-*"):
        for m in sorted(glob.glob(os.path.join(ROOT, pat, "run_manifest.json"))):
            d = os.path.dirname(m)
            if json.load(open(m)).get("preview"):
                continue
            if os.path.exists(os.path.join(d, "games.jsonl")) and os.path.exists(os.path.join(d, "run_summary.json")):
                conc_runs.append(rel(d))
    conc_json = "derived/explore/concessions_wp1.json"
    conc_fig_dir = "derived/explore/concessions/figures"
    if conc_runs and run(["scripts/analyze_concessions.py", *conc_runs, "--quiet", "--json", conc_json],
                         "concession-path analysis (WP1)"):
        os.makedirs(os.path.join(ROOT, conc_fig_dir), exist_ok=True)
        if run(["scripts/concession_figures.py", conc_json, conc_fig_dir], "concession-path figure CC1"):
            index.append({"group": "Concession path · where the \u00d710\u2076 split shift lives (opening vs concessions)",
                          "status": "exploratory, not pre-registered",
                          "command": "python3 scripts/concession_figures.py " + conc_json + " " + conc_fig_dir,
                          "figures": [rel(f) for f in figures_in(conc_fig_dir)]})

    if run(["scripts/cross_model_figures.py"], "cross-model figures C1–C2 + Holm summary"):
        index.insert(0, {"group": "Across models · the headline views",
                         "status": "C1 pre-registered primary (Holm per study); C2 Addendum A (Gemma confirmatory, Qwen reference)",
                         "command": "python3 scripts/cross_model_figures.py",
                         "figures": [rel(f) for f in figures_in("derived/cross_model/figures")]})

    for g in index:
        g["figures"] = [{"path": f, "title": svg_meta(os.path.join(ROOT, f))[0],
                         "description": svg_meta(os.path.join(ROOT, f))[1]} for f in g["figures"]]
    os.makedirs(os.path.join(ROOT, "derived"), exist_ok=True)
    with open(os.path.join(ROOT, "derived", "figures_index.json"), "w") as fh:
        json.dump({"about": "Written by scripts/make_all_figures.py; read by scripts/build_site.py (figures.html).",
                   "groups": index}, fh, indent=1)
    n = sum(len(g["figures"]) for g in index)
    print(f"• index: {n} figures in {len(index)} groups → derived/figures_index.json")
    if not a.no_site:
        run(["scripts/build_site.py"], "site (open site/figures.html)")
    if problems:
        print("\nProblems:")
        for p in problems:
            print("  " + p)
        return 1
    print("\nDone: every figure regenerated.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
