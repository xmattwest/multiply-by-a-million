#!/usr/bin/env python3
"""mutation_check_addendum_c.py: proves tests/test_addendum_c.py can fail.

Same method as tests/mutation_check_addendum_b.py: for each mutant, copy
scripts/ and the test into a temp folder, plant one flaw (an exact text
replacement that must apply once), run the test there with a 120 s limit,
and require the named test to FAIL or ERROR. An unmodified copy runs first
and must pass. The real files are never modified. No network beyond
127.0.0.1.

    python3 tests/mutation_check_addendum_c.py
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TEST = "test_addendum_c.py"

# (name, file, old, new, test that must fail)
MUTANTS = [
    ("seed-by-scale", "run_addendum_c.py", 'seed = call_seed(game["fixture"], game["opener"]',
     'seed = call_seed(game["fixture"] + game["scale"], game["opener"]',
     "test_seed_sent_where_supported_and_common_across_scales"),
    ("seed-32-bit", "run_addendum_c.py", "hexdigest()[:8], 16) & 0x7FFFFFFF", "hexdigest()[:8], 16)", "test_seeds"),
    ("engine-seed-used", "run_addendum_c.py", 'body = build_body(game["arm"], vllm_body["messages"], seed, arms)',
     'body = build_body(game["arm"], vllm_body["messages"], vllm_body["seed"], arms)',
     "test_seed_sent_where_supported_and_common_across_scales"),
    ("engine-body-leaks", "run_addendum_c.py", 'body = build_body(game["arm"], vllm_body["messages"], seed, arms)',
     'body = {**build_body(game["arm"], vllm_body["messages"], seed, arms), "top_k": vllm_body["top_k"]}',
     "test_agreement_and_body_rules"),
    ("seed-to-anthropic", "run_addendum_c.py", '    if arm["send_seed"]:\n        body["seed"] = seed',
     '    if True:\n        body["seed"] = seed', "test_agreement_and_body_rules"),
    ("attempt-not-in-seed", "run_addendum_c.py", 'info["msg_no"], info["attempt"])', 'info["msg_no"], 1)',
     "test_void_games_are_replayed_with_a_new_attempt"),
    ("retry-invalid", "run_addendum_c.py", "            if fields is not None:\n                return",
     "            if fields is not None and len(fields['content'] or '') < 300:\n                return",
     "test_invalid_reply_forfeits_and_is_not_reasked"),
    ("no-cap", "run_addendum_c.py", "if not cap.reserve(wpc):", "if not cap.reserve(0.0):",
     "test_spend_cap_never_exceeded"),
    ("retry-rejected", "run_addendum_c.py",
     "if status is not None and 400 <= status < 500 and status not in (408, 429):", "if False:",
     "test_rejected_request_stops"),
    ("no-confirm-gate", "run_addendum_c.py", 'if a.confirm_cost.lstrip("$") != confirm_s:', "if False:",
     "test_wrong_confirm_cost"),
    ("no-freeze-check", "run_addendum_c.py", "        problem = check_frozen()\n", "        problem = None\n",
     "test_registered_run_needs_frozen_inputs"),
    ("no-credit-check", "run_addendum_c.py", "if a.pilot and remaining is not None and remaining < worst:",
      "if False:", "test_low_key_credit_stops_before_any_prompt"),
    ("main-client-cap-restored", "run_addendum_c.py", "rb.SpendCap(a.spend_cap if a.pilot else math.inf)",
     "rb.SpendCap(a.spend_cap if a.pilot else 15.0)", "test_main_uses_key_cap_and_stops_on_402"),
    ("no-main-key-cap-check", "run_addendum_c.py", "if not a.pilot and not all(isinstance(x, (int, float))",
     "if False and not all(isinstance(x, (int, float))", "test_main_requires_capped_key_with_credit"),
    ("no-100K", "run_addendum_c.py", 'SCALES = (("x1e2", 100), ("x1e5", 100_000), ("x1e8", 100_000_000))',
     'SCALES = (("x1e2", 100), ("x1e8", 100_000_000))', "test_main_games"),
    ("midpoint-last-offers", "analyze_addendum_c.py", "(Fraction(s[0] + b[0], 2)", "(Fraction(s[-1] + b[-1], 2)",
     "test_d_and_e_by_hand"),
    ("midpoint-without-reply", "analyze_addendum_c.py", "    if not s or not b:\n        return None",
     "    if not s or not b:\n        return share(r)", "test_accept_of_the_first_offer_has_no_midpoint"),
    ("first-sample-only", "analyze_addendum_c.py",
     "    return sum(xs, Fraction(0)) / len(xs) if xs else None", "    return xs[0] if xs else None",
     "test_cell_means_over_samples"),
    ("pilot-analysed", "analyze_addendum_c.py", 'if "pilot" in manifest["kind"] and not manifest.get("preview"):',
     "if False:", "test_keyed_pilot_end_to_end_key_never_written"),
    ("buyer-gap-sign", "analyze_addendum_c.py", 'else 1 - p / r["C"]', 'else p / r["C"] - 1',
     "test_opening_gap_both_roles"),
]


def layout(d):
    os.makedirs(os.path.join(d, "tests"))
    shutil.copytree(os.path.join(ROOT, "scripts"), os.path.join(d, "scripts"),
                    ignore=shutil.ignore_patterns("__pycache__", "interp", "*.sh"))
    shutil.copy(os.path.join(HERE, TEST), os.path.join(d, "tests"))
    for x in ("config", "prereg", "data"):
        os.symlink(os.path.join(ROOT, x), os.path.join(d, x))


def run(d):
    try:
        p = subprocess.run([sys.executable, "-W", "ignore", os.path.join(d, "tests", TEST)],
                           capture_output=True, text=True, timeout=120)
        return p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"


def main():
    tmp = tempfile.mkdtemp(prefix="si-addc-mut-")
    try:
        layout(os.path.join(tmp, "baseline"))
        rc, out = run(os.path.join(tmp, "baseline"))
        ran = re.search(r"^Ran .*$", out, re.M)
        if rc != 0:
            print("baseline FAILED; fix the tests first:\n" + "\n".join(re.findall(r"^(?:FAIL|ERROR):.*$", out, re.M)))
            return 1
        print(f"baseline: unmodified copy passes ({ran.group(0) if ran else '?'})\n")
        caught = 0
        for name, fname, old, new, expect in MUTANTS:
            d = os.path.join(tmp, name)
            layout(d)
            path = os.path.join(d, "scripts", fname)
            src = open(path).read()
            if src.count(old) != 1:
                print(f"BROKEN  {name:24s} planted text found {src.count(old)} times; update this mutant")
                continue
            os.remove(path)
            open(path, "w").write(src.replace(old, new))
            rc, out = run(d)
            hit = rc != 0 and (re.search(rf"^(FAIL|ERROR): {expect} ", out, re.M) or out == "TIMEOUT")
            caught += bool(hit)
            print(f"{'caught ' if hit else 'MISSED '} {name:24s} ({fname}; expected {expect})")
        print(f"\n{caught} of {len(MUTANTS)} planted flaws caught")
        return 0 if caught == len(MUTANTS) else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
