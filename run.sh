#!/bin/sh
# Start the app at http://127.0.0.1:8767 (localhost only).
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || ./setup.sh
exec .venv/bin/python -m audition serve "$@"
