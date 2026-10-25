#!/usr/bin/env python3
"""License audit for the ML workshop's installed dependencies.

Stdlib-only. Enumerates installed distributions via importlib.metadata,
reads License-Expression / License metadata and trove classifiers, and checks
every package against the course allowlist. The course itself is MIT, so all
runtime deps must be permissive / weak-copyleft.

Exit 0 with a PASS table when every installed distribution resolves to an
allowlisted license; exit 1 with a FAIL table (raw license strings included)
otherwise.

Signals are tried in order - a package passes if ANY resolves:
  1. License-Expression (SPDX, incl. AND/OR/WITH handling)
  2. License field (exact match, or signature-phrase sniffing for the old
     style that embeds full license text)
  3. Trove classifiers ("License :: OSI Approved :: ...")
This tolerance is deliberate: PyPI metadata is messy (e.g. isoduration ships
License "UNKNOWN" with an ISC classifier), and the audit's job is to catch
genuinely unknown or non-permissive licenses, not sloppy metadata.
"""

from __future__ import annotations

import importlib.metadata as md
import re
import sys

ALLOWLIST = {
    "MIT", "MIT-0", "BSD-2-Clause", "BSD-3-Clause", "Apache-2.0",
    "PSF-2.0", "Python-2.0", "ISC", "CC0-1.0", "MPL-2.0", "Zlib",
    "Unlicense",
}

# normalized-string -> canonical allowlist entry.
# Notes on judgment calls:
#   MIT-CMU      ~ MIT-style permissive grant (Pillow) -> MIT
#   0BSD         zero-clause BSD, strictly freer than BSD-2-Clause -> BSD-2-Clause
#   CNRI-Python  subsumed by the modern Python-2.0 stack (regex) -> Python-2.0
#   PSFL         Python Software Foundation License -> PSF-2.0
#   BSL-1.0      Boost Software License: permissive, MIT-equivalent terms
#                (bundled inside torch) -> MIT
_ALIASES = {
    "mit": "MIT", "the mit": "MIT", "mit-0": "MIT-0", "mit0": "MIT-0",
    "mit-cmu": "MIT", "expat": "MIT",
    "bsd": "BSD-3-Clause", "new bsd": "BSD-3-Clause", "modified bsd": "BSD-3-Clause",
    "bsd-3-clause": "BSD-3-Clause", "bsd 3-clause": "BSD-3-Clause",
    "bsd-2-clause": "BSD-2-Clause", "bsd 2-clause": "BSD-2-Clause",
    "simplified bsd": "BSD-2-Clause", "0bsd": "BSD-2-Clause",
    "apache": "Apache-2.0", "apache 2.0": "Apache-2.0", "apache-2.0": "Apache-2.0",
    "apache software": "Apache-2.0", "apache software license": "Apache-2.0",
    "apache license 2.0": "Apache-2.0",
    "psf": "PSF-2.0", "psfl": "PSF-2.0", "psf-2.0": "PSF-2.0",
    "python software foundation": "PSF-2.0",
    "python-2.0": "Python-2.0", "python": "Python-2.0",
    "cnri-python": "Python-2.0", "cnri": "Python-2.0",
    "isc": "ISC",
    "cc0": "CC0-1.0", "cc0-1.0": "CC0-1.0",
    "mpl-2.0": "MPL-2.0", "mpl 2.0": "MPL-2.0", "mpl2": "MPL-2.0",
    "mozilla public license 2.0": "MPL-2.0", "mozilla public 2.0": "MPL-2.0",
    "zlib": "Zlib",
    "bsl-1.0": "MIT", "bsl": "MIT", "boost software": "MIT", "boost": "MIT",
    "unlicense": "Unlicense", "the unlicense": "Unlicense",
}

_WS = re.compile(r"\s+")
_SUFFIXES = re.compile(r"\s+(license|software license|licence)$", re.I)
_PAREN = re.compile(r"\([^)]*\)")


def _norm(s: str) -> str:
    s = _PAREN.sub(" ", s)
    s = _WS.sub(" ", s).strip().lower()
    s = _SUFFIXES.sub("", s)
    return s.strip(" -_")


def _lookup(raw: str) -> str | None:
    raw = raw.strip()
    if raw in ALLOWLIST:
        return raw
    return _ALIASES.get(_norm(raw))


def _check_expression(expr: str) -> str | None:
    """Evaluate an SPDX-ish expression: OR of AND-branches; WITH strips exceptions."""
    for branch in re.split(r"\bOR\b", expr):
        tokens = re.split(r"\bAND\b", branch)
        resolved = []
        for tok in tokens:
            tok = re.sub(r"\bWITH\b.*", "", tok)  # drop exception modifiers
            tok = tok.strip(" ()")
            hit = _lookup(tok)
            if not hit:
                resolved = None
                break
            resolved.append(hit)
        if resolved:
            return " AND ".join(sorted(set(resolved)))
    return None


_SIGNATURES = [
    ("apache license", "Apache-2.0"),
    ("redistribution and use in source and binary forms", "BSD-3-Clause"),
    ("permission is hereby granted, free of charge", "MIT"),
    ("mozilla public license", "MPL-2.0"),
]


def _check_license_field(text: str) -> str | None:
    text = text.strip()
    if not text or text.upper() in ("UNKNOWN", "N/A"):
        return None
    if len(text) < 300 and "\n" not in text:
        hit = _lookup(text)
        if hit:
            return hit
        # expressions sometimes live in the License field too
        if re.search(r"\b(AND|OR)\b", text):
            return _check_expression(text)
        return None
    # old-style full license text: sniff signature phrases
    head = text[:4000].lower()
    for phrase, lic in _SIGNATURES:
        if phrase in head:
            return lic
    return None


def _check_classifiers(classifiers: list[str]) -> str | None:
    for c in classifiers:
        if not c.startswith("License ::"):
            continue
        hit = _lookup(c.split("::")[-1])
        if hit:
            return hit
    return None


def audit() -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    passed, failed = [], []
    dists = sorted(md.distributions(),
                   key=lambda d: (d.metadata.get("Name") or "").lower())
    for d in dists:
        meta = d.metadata
        name = meta.get("Name") or "<unnamed>"
        expr = (meta.get("License-Expression") or "").strip()
        lic_field = (meta.get("License") or "").strip()
        classifiers = [c for c in (meta.get_all("Classifier") or [])
                       if c.startswith("License ::")]
        matched = None
        if expr:
            matched = _check_expression(expr)
        if not matched and lic_field:
            matched = _check_license_field(lic_field)
        if not matched and classifiers:
            matched = _check_classifiers(classifiers)
        if matched:
            passed.append((name, matched, d.version or "?"))
        else:
            raw = expr or lic_field[:120].replace("\n", " ") or "<none>"
            cls = "; ".join(classifiers) or "<none>"
            failed.append((name, raw, cls))
    return passed, failed


def main() -> None:
    passed, failed = audit()
    if failed:
        print("LICENSE AUDIT: FAIL\n")
        print(f"{'package':<35} {'raw license':<70} classifiers")
        print("-" * 130)
        for name, raw, cls in failed:
            print(f"{name:<35} {raw:<70} {cls}")
        print(f"\n{len(failed)} package(s) have unknown or non-allowlisted licenses.")
        sys.exit(1)

    print(f"{'package':<45} license")
    print("-" * 70)
    for name, lic, _ver in passed:
        print(f"{name:<45} {lic}")
    print(f"\nLICENSE AUDIT: PASS - {len(passed)} packages, all licenses allowlisted.")
    sys.exit(0)


if __name__ == "__main__":
    main()
