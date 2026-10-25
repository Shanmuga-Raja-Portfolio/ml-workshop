#!/usr/bin/env bash
# ML Workshop - lesson launcher for macOS / Linux.
# Equivalent of "START HERE.bat": bootstrap the environment, then open
# JupyterLab in your browser.
set -euo pipefail
cd "$(dirname "$0")"

# find a Python 3 interpreter
PYCMD=""
for cand in python3 python; do
    if command -v "$cand" >/dev/null 2>&1; then
        PYCMD="$cand"
        break
    fi
done
if [ -z "$PYCMD" ]; then
    echo "ERROR: no Python 3 found. Install Python 3.12+ and re-run this script."
    exit 1
fi

echo "============================================================"
echo "   ML WORKSHOP - starting the lesson notebook..."
echo
echo "   First run sets up everything automatically (Python"
echo "   packages, the right PyTorch build for your GPU) - this"
echo "   can take several minutes. Later runs start in seconds."
echo
echo "   1. Wait for your browser to open by itself"
echo "   2. In the browser left panel, open:"
echo "      lessons  >  01_foundations  >  lesson1_tiny_net..."
echo "   3. KEEP THIS TERMINAL OPEN while you work"
echo "   4. When done: Ctrl-C here to stop Jupyter"
echo "============================================================"
echo

# bootstrap finds/installs uv, detects the GPU, syncs dependencies
"$PYCMD" tools/bootstrap.py

# after bootstrap, uv is guaranteed: on PATH or in a well-known dir
UV="uv"
if ! command -v uv >/dev/null 2>&1; then
    if [ -x "$HOME/.local/bin/uv" ]; then UV="$HOME/.local/bin/uv"; fi
    if [ -x "$HOME/.cargo/bin/uv" ]; then UV="$HOME/.cargo/bin/uv"; fi
fi

"$UV" run jupyter lab
