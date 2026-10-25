"""Repo hygiene gate. Stdlib-only. Run: uv run python tools/check_repo.py [--regen-check]

Checks:
  1. Scrub greps — no personal paths, usernames, or non-shippable content references.
  2. Notebook cleanliness — every lessons/**/*.ipynb has empty outputs / null counts.
  3. Manifest validity — downloads/manifest.json parses and has required fields.
  4. --regen-check — build_lesson*.py generators reproduce the committed notebooks
     byte-identically (CI uses this; generators must stay stdlib-only).
"""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FORBIDDEN = [
    (re.compile(r"naruto", re.I), "copyrighted dataset reference"),
    (re.compile(r"a30wattbulb", re.I), "personal username"),
    (re.compile(r"C:\\Users\\", re.I), "personal Windows path"),
    (re.compile(r"C:\\Python", re.I), "hard-coded Python install path"),
    (re.compile(r"Z:\\", re.I), "absolute dev-machine path"),
    (re.compile(r"/Users/[A-Za-z0-9_.-]+", re.I), "personal macOS path"),
    (re.compile(r"/home/[A-Za-z0-9_.-]+", re.I), "personal Linux path"),
]
SKIP_DIRS = {".git", ".venv", "runs", "data", ".ipynb_checkpoints", "__pycache__", ".github"}
BINARY_EXT = {".png", ".jpg", ".jpeg", ".gif", ".pt", ".pth", ".bin", ".safetensors", ".ico", ".lock"}
SKIP_FILES = {"check_repo.py", "uv.lock"}  # this file contains the patterns by necessity

MANIFEST_FIELDS = {"id", "type", "ref", "license", "license_url", "size_hint",
                   "required_by", "optional", "fallback"}


def iter_text_files():
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if p.suffix.lower() in BINARY_EXT or p.name in SKIP_FILES:
            continue
        yield p, rel


def check_scrub():
    failures = []
    for p, rel in iter_text_files():
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for pattern, why in FORBIDDEN:
            for i, line in enumerate(text.splitlines(), 1):
                if pattern.search(line):
                    failures.append(f"{rel}:{i}: forbidden content ({why}): {line.strip()[:80]}")
    return failures


def check_notebooks():
    failures = []
    for nb in sorted((ROOT / "lessons").rglob("*.ipynb")):
        rel = nb.relative_to(ROOT)
        try:
            doc = json.loads(nb.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            failures.append(f"{rel}: unreadable notebook JSON: {e}")
            continue
        for idx, cell in enumerate(doc.get("cells", [])):
            if cell.get("cell_type") == "code":
                if cell.get("outputs"):
                    failures.append(f"{rel}: cell {idx} has outputs (notebooks ship clean)")
                if cell.get("execution_count") is not None:
                    failures.append(f"{rel}: cell {idx} has execution_count set")
    return failures


def check_manifest():
    failures = []
    manifest = ROOT / "downloads" / "manifest.json"
    try:
        items = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return [f"downloads/manifest.json: {e}"]
    items = items if isinstance(items, list) else items.get("items", [])
    for i, item in enumerate(items):
        missing = MANIFEST_FIELDS - set(item)
        if missing:
            failures.append(f"manifest item {i} ({item.get('id', '?')}): missing {sorted(missing)}")
    return failures


def check_regen():
    failures = []
    notebooks = {p: hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in (ROOT / "lessons").rglob("*.ipynb")}
    for gen in sorted((ROOT / "tools").glob("build_lesson*.py")):
        r = subprocess.run([sys.executable, str(gen)], cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            failures.append(f"{gen.name}: generator failed:\n{r.stderr[-500:]}")
    for p, before in notebooks.items():
        after = hashlib.sha256(p.read_bytes()).hexdigest()
        if before != after:
            failures.append(f"{p.relative_to(ROOT)}: out of sync with its generator "
                            f"(regenerated now — review and commit the regenerated file)")
    return failures


def check_pyproject():
    # bootstrap appends a machine-local torch-index block at setup time;
    # it must never be committed (each machine generates its own).
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    if "BEGIN machine-local PyTorch config" in text:
        return ["pyproject.toml: contains a generated machine-local PyTorch block — "
                "strip everything from '--- BEGIN machine-local' to '--- END machine-local' before committing"]
    return []


def main():
    regen = "--regen-check" in sys.argv
    checks = [("scrub greps", check_scrub), ("notebook cleanliness", check_notebooks),
              ("manifest validity", check_manifest)]
    if regen:
        # CI mode: the checked-out tree is the committed content, so a machine-local
        # torch block here means it was committed — hard fail. Locally it's expected
        # after bootstrap, so it stays a warning.
        checks.append(("generator sync", check_regen))
        checks.append(("pyproject neutrality", check_pyproject))
    failed = False
    for name, fn in checks:
        problems = fn()
        if problems:
            failed = True
            print(f"FAIL {name}:")
            for p in problems[:50]:
                print(f"  {p}")
        else:
            print(f"PASS {name}")
    if not regen:
        for p in check_pyproject():
            print(f"WARN {p}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
