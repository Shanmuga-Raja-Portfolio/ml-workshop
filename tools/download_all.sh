#!/usr/bin/env bash
# Manual fallback downloader for all course models/datasets.
#
# The normal path is automatic: tools/bootstrap.py (run by "START HERE.bat" /
# start-here.sh) can prefetch everything with --download. This script exists
# for people who prefer a plain shell entry point — it just forwards to the
# same code, which reads downloads/manifest.json, prints each item's license,
# retries with backoff, and resumes partial downloads.
set -euo pipefail
cd "$(dirname "$0")/.."

if command -v python3 >/dev/null 2>&1; then
    PY=python3
else
    PY=python
fi

exec "$PY" tools/bootstrap.py --download
