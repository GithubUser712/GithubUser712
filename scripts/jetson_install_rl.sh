#!/usr/bin/env bash
# Install RL / PyTorch stack for step 2+ on Jetson.
# Run AFTER jetson_setup.sh succeeded.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

echo "=== Installing RL dependencies (step 2+) ==="

python -m pip install gymnasium pyserial stable-baselines3

# PyTorch on Jetson — try NVIDIA wheel index first, fall back to pip.
if [[ "$(uname -m)" == "aarch64" ]]; then
  echo "ARM64 detected — installing PyTorch for Jetson ..."
  python -m pip install torch torchvision --index-url https://developer.download.nvidia.com/compute/redist/jp/v60 2>/dev/null \
    || python -m pip install torch torchvision \
    || {
      echo ""
      echo "WARN: automatic torch install failed."
      echo "  Install manually from: https://forums.developer.nvidia.com/c/agx-autonomous-machines/jetson-embedded-systems/jetson-orin-nano/"
      exit 1
    }
else
  python -m pip install torch
fi

python -m pip install -e ".[dev]" 2>/dev/null || python -m pip install -e .

echo ""
echo "=== RL install complete ==="
echo "  python -m pipeline.step2"
