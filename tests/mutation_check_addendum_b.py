#!/usr/bin/env python3
"""mutation_check_addendum_b.py: proves tests/test_addendum_b.py can fail.

For each mutant: copy scripts/ and the test into a temp folder, plant one
flaw (an exact text replacement that must apply), run the test there with a
90 s limit, and require the named test to FAIL or ERROR. An unmodified copy
runs first and must pass. The real files are never modified. No network
beyond 127.0.0.1.

    python3 tests/mutation_check_addendum_b.py
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# (name, file, old, new, test that must fail)
MUTANTS = [
    ("fallbacks-on", "openrouter_client.py", '"allow_fallbacks": False', '"allow_fallbacks": True',
     "test_rejections"),
    ("any-provider", "openrouter_client.py", 'if body["provider"] != provider_block(arm):', 'if False:',
     "test_rejections"),
    ("extra-fields-ok", "openrouter_client.py", "    if unknown:\n        raise", "    if False:\n        raise",
     "test_rejections"),
    ("seed-to-anyone", "openrouter_client.py", 'if ("seed" in body) != arm["send_seed"]:', "if False:",
     "test_rejections"),
    ("key-in-repr", "openrouter_client.py", 'return f"Client(preview_port={self.preview_port!r})"',
     'return f"Client({self._key})"', "test_real_client_targets_openrouter_over_verified_tls"),
    ("no-tls-check", "openrouter_client.py", "context=ssl.create_default_context())",
     "context=ssl._create_unverified_context())", "test_real_client_targets_openrouter_over_verified_tls"),
    ("preview-reads-key", "openrouter_client.py", "            self._key = None\n",
     "            self._key = key_reader()\n", "test_preview_never_reads_keychain_or_sends_auth"),
    ("echo-not-scrubbed", "openrouter_client.py", "if self._key is not None and self._key.encode() in data:",
     "if False:", "test_response_echoing_the_key_is_discarded"),
    ("key-from-env", "openrouter_client.py", 'account = account or os.environ.get("USER") or ""',
     'account = account or os.environ.get("USER") or ""\n    if os.environ.get("OPENROUTER_API_KEY"):\n'
     '        return os.environ["OPENROUTER_API_KEY"]', "test_no_key_from_environment"),
    ("no-deadline", "openrouter_client.py",
     "                if remaining <= 0:\n                    return None, b\"\", f\"timeout: no complete reply "
     "within {self.total_timeout:g} s\"\n                sock.settimeout(remaining)\n",
     "                sock.settimeout(self.total_timeout)\n",
     "test_deadline"),
    ("key-in-log", "openrouter_client.py", "            return status, data, None\n",
     "            return status, data, ('auth ' + self._key) if self._key else None\n",
     "test_keyed_pilot_end_to_end_key_never_written"),
    ("retry-invalid", "run_addendum_b.py", "            if fields is not None:\n                return",
     "            if fields is not None and fields['content'] and fields['content'].startswith('{\"action'):\n"
     "                return", "test_invalid_reply_is_not_retried"),
    ("no-cap", "run_addendum_b.py", "if self.spent + self.reserved + amount > self.cap + 1e-12:", "if False:",
     "test_spend_cap_is_never_exceeded"),
    ("free-if-no-cost", "run_addendum_b.py", 'actual = fields["cost"] if fields["cost"] is not None else wpc',
     'actual = fields["cost"] or 0.0', "test_missing_cost_counts_at_worst"),
    ("seed-by-scale", "run_addendum_b.py", '"seed": item_seed(f["id"], role, s)}',
     '"seed": item_seed(f["id"] + tag, role, s)}', "test_seeds_common_across_scales_distinct_across_samples"),
    ("seed-32-bit", "run_addendum_b.py", "hexdigest()[:8], 16) & 0x7FFFFFFF", "hexdigest()[:8], 16)",
     "test_seeds_fit_int32"),
    ("retry-rejected", "run_addendum_b.py", "if status is not None and 400 <= status < 500 and status not in (408, 429):",
     "if False:", "test_rejected_request_stops_without_retry"),
    ("refusal-as-invalid", "run_addendum_b.py", 'if reply.get("refusal") or reply.get("finish_reason") == "content_filter":',
     "if False:", "test_invalid_refusal_walkaway"),
    ("buyer-gap-sign", "run_addendum_b.py", "else 1 - price / limit", "else price / limit - 1", "test_offer_gap"),
    ("no-confirm-gate", "run_addendum_b.py", 'if a.confirm_cost.lstrip("$") != worst_s:', "if False:",
     "test_wrong_confirm_cost"),
    ("no-freeze-check", "run_addendum_b.py", "        problem = check_frozen()\n", "        problem = None\n",
     "test_registered_run_needs_frozen_inputs"),
    ("no-credit-check", "run_addendum_b.py", "if remaining is not None and remaining < worst:", "if False:",
     "test_low_key_credit_stops_before_any_prompt"),
    ("sampling-set", "run_addendum_b.py", '"provider": oc.provider_block(arm), "reasoning": arm["reasoning"], '
     '"stream": False}', '"provider": oc.provider_block(arm), "reasoning": arm["reasoning"], "stream": False, '
     '"temperature": 0.7}', "test_every_arm_body_passes"),
    ("invalid-counted", "analyze_addendum_b.py",
     'if not r["errors"] and r["action"] == "OFFER" and r["gap"] is not None and keep(r):',
     'if (r["gap"] is not None or r["errors"]) and keep(r):', "test_h_by_hand"),
    ("pool-not-mean", "analyze_addendum_b.py", "lo_m, hi_m = statistics.fmean(lo), statistics.fmean(hi)",
     "lo_m, hi_m = lo[0], hi[0]", "test_h_by_hand"),
    ("wrong-seed", "analyze_addendum_b.py", 'A.seed("addendumB-wilcoxon-HA", arm)', 'A.seed("wilcoxon-HA", arm)',
     "test_test_matches_analyze_py_and_holm"),
]


def layout(d):
    os.makedirs(os.path.join(d, "tests"))
    shutil.copytree(os.path.join(ROOT, "scripts"), os.path.join(d, "scripts"),
                    ignore=shutil.ignore_patterns("__pycache__", "interp", "*.sh"))
    shutil.copy(os.path.join(HERE, "test_addendum_b.py"), os.path.join(d, "tests"))
    for x in ("config", "prereg", "data"):
        os.symlink(os.path.join(ROOT, x), os.path.join(d, x))


def run(d):
    try:
        p = subprocess.run([sys.executable, os.path.join(d, "tests", "test_addendum_b.py")],
                           capture_output=True, text=True, timeout=90)
        return p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"


def main():
    tmp = tempfile.mkdtemp(prefix="si-addb-mut-")
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
                print(f"BROKEN  {name:20s} planted text found {src.count(old)} times; update this mutant")
                continue
            os.remove(path)                      # copytree made a real file; replace it
            open(path, "w").write(src.replace(old, new))
            rc, out = run(d)
            hit = rc != 0 and (re.search(rf"^(FAIL|ERROR): {expect} ", out, re.M) or out == "TIMEOUT")
            caught += bool(hit)
            print(f"{'caught ' if hit else 'MISSED '} {name:20s} ({fname}; expected {expect})")
        print(f"\n{caught} of {len(MUTANTS)} planted flaws caught")
        return 0 if caught == len(MUTANTS) else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
