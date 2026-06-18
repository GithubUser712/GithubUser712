#!/usr/bin/env bash
# One-command step 1 on Jetson (assumes jetson_setup.sh already ran).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pipeline.step1 --preset "${1:-sample}"
