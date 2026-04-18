#!/usr/bin/env bash
# Launch the per-session visualization tool.
# Opens at http://127.0.0.1:8765/
set -euo pipefail

cd "$(dirname "$0")"
if [[ -d .venv ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

exec uvicorn viz.server:app --host 127.0.0.1 --port 8765 --reload
