#!/usr/bin/env bash
# Run pipeline steps 0-4 with a shared training mode.
#
#   bash scripts/run_pipeline.sh bullet_learn sample
#   bash scripts/run_pipeline.sh quick_train sample
#   bash scripts/run_pipeline.sh deep_learn spa artifacts_spa
#
# Args:
#   $1  mode     bullet_learn | quick_train | deep_learn | deep_learn_xhigh
#   $2  preset   sample | monaco | spa ...  (default: sample)
#   $3  out dir  artifacts folder           (default: artifacts_<preset>)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODE="${1:-deep_learn}"
PRESET="${2:-sample}"
OUT="${3:-artifacts_${PRESET}}"
PARAMS_FILE="${PARAMS_FILE:-config/default_car.yaml}"

if [[ "$MODE" != "bullet_learn" && "$MODE" != "quick_train" && "$MODE" != "deep_learn" && "$MODE" != "deep_learn_xhigh" ]]; then
  echo "ERROR: mode must be bullet_learn, quick_train, deep_learn, or deep_learn_xhigh (got '$MODE')"
  exit 2
fi

echo "=== Raceline pipeline | mode=$MODE preset=$PRESET out=$OUT ==="

# --- Step 0: environment ---------------------------------------------------
if [[ ! -d .venv ]]; then
  if [[ -f /etc/nv_tegra_release ]]; then
    bash scripts/jetson_setup.sh
  else
    python3 -m venv .venv
    # shellcheck disable=SC1091
    source .venv/bin/activate
    python -m pip install --upgrade pip wheel
    python -m pip install -r requirements-step1.txt
    python -m pip install -e ".[dev]"
  fi
fi
# shellcheck disable=SC1091
source .venv/bin/activate

if ! python -c "import torch, stable_baselines3" 2>/dev/null; then
  if [[ -f /etc/nv_tegra_release ]]; then
    bash scripts/jetson_install_rl.sh
  else
    python -m pip install -e ".[dev]"
  fi
fi

python -m raceline.setup --verify

# --- Step 1: map ingestion -------------------------------------------------
python -m pipeline.step1 --preset "$PRESET" --out "$OUT"

# --- Step 2: RL racing line ------------------------------------------------
python -m pipeline.step2 --artifacts "$OUT" --mode "$MODE" --fresh

# --- Step 3: car physics + retune ------------------------------------------
STEP3_ARGS=(--artifacts "$OUT" --mode "$MODE")
if [[ -f "$PARAMS_FILE" ]]; then
  STEP3_ARGS+=(--params-file "$PARAMS_FILE")
fi
python -m pipeline.step3 "${STEP3_ARGS[@]}"

# --- Step 4: braking / acceleration zones ----------------------------------
python -m pipeline.step4 --artifacts "$OUT"

echo ""
echo "=== Pipeline complete (mode=$MODE) ==="
echo "  artifacts : $OUT/"
echo "  policy    : $OUT/rl_policy_tuned.zip"
echo "  zones     : $OUT/zones.yaml"
echo "  watch     : python -m pipeline.watch --artifacts $OUT --video run.mp4"
