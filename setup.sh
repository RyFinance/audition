#!/bin/sh
# One-time setup: a local Python environment with Audition's dependencies.
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
$PY -c 'import sys; assert sys.version_info >= (3, 10), "Audition needs Python 3.10+"'
[ -d .venv ] || $PY -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt
echo "ready. start the app with ./run.sh"
