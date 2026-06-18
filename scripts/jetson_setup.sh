#!/usr/bin/env bash
# Step 0 — one-shot setup for Jetson Orin Nano (Ubuntu / aarch64).
# Usage:  bash scripts/jetson_setup.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== Raceline Step 0: Jetson setup ==="
echo "Project root: $ROOT"

# --- 0a. Verify we have the real code (not empty main branch) ---
if [[ ! -f pipeline/step1.py ]]; then
  echo ""
  echo "ERROR: pipeline/step1.py not found."
  echo "  The GitHub main branch is empty. Checkout the code branch:"
  echo "    git fetch origin"
  echo "    git checkout cursor/step5-6-vesc-3d48"
  exit 1
fi

# --- 0b. System packages (apt is reliable on Jetson) ---
if command -v apt-get &>/dev/null; then
  echo ""
  echo "Installing system packages (sudo) ..."
  sudo apt-get update -qq
  sudo apt-get install -y \
    python3 python3-pip python3-venv \
    python3-numpy python3-scipy python3-opencv python3-matplotlib python3-yaml \
    libgl1 libglib2.0-0 \
    git
fi

# --- 0c. Python venv ---
if [[ ! -d .venv ]]; then
  echo ""
  echo "Creating virtual environment .venv ..."
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo ""
echo "Python: $(python --version) on $(uname -m)"

# --- 0d. Step-1 pip packages only (no torch yet) ---
echo ""
echo "Installing step-1 Python packages ..."
python -m pip install --upgrade pip wheel
python -m pip install -r requirements-step1.txt
python -m pip install -e . --no-deps

# --- 0e. Verify ---
echo ""
echo "Verifying environment ..."
python -m raceline.setup --verify

echo ""
echo "=== Step 0 complete ==="
echo ""
echo "Next — run step 1 (one command, no prompts):"
echo "  source .venv/bin/activate"
echo "  python -m pipeline.step1 --preset sample"
echo ""
echo "When you reach step 2 (RL training), install ML packages:"
echo "  bash scripts/jetson_install_rl.sh"
