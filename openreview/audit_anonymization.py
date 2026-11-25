#!/usr/bin/env python3
"""Double-blind anonymization, LaTeX hygiene, and Study 3 freeze audit for openreview/."""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Patterns that must NEVER appear in double-blind submission text or compiled PDF
FORBIDDEN = [
    (r"\bmatt\b", "author first name (allowlist: Alexander Matt Turner, Matt Fredrikson in bib)"),
    (r"\bwest\b", "author last name (allowlist: West, Robert in bib)"),
    (r"shelltoho\.me", "personal domain"),
    (r"xmattwest", "GitHub handle"),
    (r"0009-0008-5557-9394", "author ORCID"),
    (r"zenodo\.2442", "Zenodo DOI"),
    (r"2442000[23]", "Zenodo record ID"),
]

ALLOWLIST_CONTEXT = [
    "Alexander Matt Turner",
    "Matt Fredrikson",
    "West, Robert",
    "Turner, Alexander Matt",
    "Fredrikson, Matt",
    "Robert West",
]


def check_text(label, text):
    leaks = []
    for lineno, line in enumerate(text.splitlines(), 1):
        # Strip allowlisted bibliography citations of unrelated authors
        scrubbed = line
        for ok in ALLOWLIST_CONTEXT:
            scrubbed = scrubbed.replace(ok, "")
        for pat, desc in FORBIDDEN:
            if re.search(pat, scrubbed, re.IGNORECASE):
                leaks.append((label, lineno, desc, line.strip()[:140]))
    return leaks


def main():
    leaks = []
    # 1. Scan all .tex files in openreview/
    for dirpath, _, filenames in os.walk(HERE):
        for fn in sorted(filenames):
            if fn.endswith(".tex"):
                p = os.path.join(dirpath, fn)
                rel = os.path.relpath(p, ROOT)
                leaks.extend(check_text(rel, open(p, encoding="utf-8").read()))

    # 2. Scan compiled PDF text
    pdf_path = os.path.join(HERE, "main.pdf")
    pdf_pages = None
    if os.path.exists(pdf_path):
        txt = subprocess.check_output(["pdftotext", pdf_path, "-"], text=True)
        leaks.extend(check_text("openreview/main.pdf", txt))
        info = subprocess.check_output(["pdfinfo", pdf_path], text=True)
        m = re.search(r"Pages:\s+(\d+)", info)
        pdf_pages = int(m.group(1)) if m else None
    else:
        print("WARN: openreview/main.pdf not found")

    # 3. Check main.log for undefined refs/citations and Overfull hbox > 3pt
    log_path = os.path.join(HERE, "main.log")
    undef_refs, undef_cites, overfull = [], [], []
    if os.path.exists(log_path):
        log = open(log_path, encoding="utf-8", errors="replace").read()
        undef_refs = re.findall(r"LaTeX Warning: Reference `([^']+)' on page \d+ undefined", log)
        undef_cites = re.findall(r"Citation `([^']+)' on page \d+ undefined", log)
        for m in re.finditer(r"Overfull \\hbox \((\d+\.\d+)pt too wide\)[^\n]*", log):
            if float(m.group(1)) > 3.0:
                overfull.append(m.group(0))

    # 4. Verify frozen Study 3 SHA-256 hashes
    sha_res = subprocess.run(
        ["shasum", "-a", "256", "-c", "prereg/study3/inputs.sha256"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    sha_ok = sha_res.returncode == 0
    sha_count = sum(1 for l in sha_res.stdout.splitlines() if l.endswith(": OK"))

    print("=== OPENREVIEW DOUBLE-BLIND & BUILD AUDIT ===")
    print(f"PDF compiled:       {'YES' if os.path.exists(pdf_path) else 'NO'} ({pdf_pages} pages)")
    print(f"Anonymization leaks: {len(leaks)}")
    for item in leaks:
        print(f"  LEAK {item[0]}:{item[1]} [{item[2]}]: {item[3]}")
    print(f"Undefined refs:     {len(undef_refs)} {undef_refs[:10]}")
    print(f"Undefined cites:    {len(undef_cites)} {undef_cites[:10]}")
    print(f"Overfull hbox >3pt: {len(overfull)}")
    for ov in overfull[:10]:
        print(f"  {ov}")
    print(f"Study 3 SHA-256:    {'PASS' if sha_ok else 'FAIL'} ({sha_count}/37 files OK)")

    if leaks or undef_refs or undef_cites or not sha_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
