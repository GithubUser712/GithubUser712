# Run pipeline steps 0-4 on Windows with a shared training mode.
#
#   .\scripts\run_pipeline.ps1 quick_train sample
#   .\scripts\run_pipeline.ps1 deep_learn spa artifacts_spa
param(
    [ValidateSet("bullet_learn", "quick_train", "deep_learn", "deep_learn_xhigh")]
    [string]$Mode = "deep_learn",
    [string]$Preset = "sample",
    [string]$Out = "",
    [string]$ParamsFile = "config/default_car.yaml"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

if (-not $Out) { $Out = "artifacts_$Preset" }

Write-Host "=== Raceline pipeline | mode=$Mode preset=$Preset out=$Out ==="

# --- Step 0 -----------------------------------------------------------------
if (-not (Test-Path .venv)) {
    python -m venv .venv
}
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip wheel
python -m pip install -r requirements-step1.txt
python -m pip install -e ".[dev]"
python -m raceline.setup --verify

# --- Step 1 -----------------------------------------------------------------
python -m pipeline.step1 --preset $Preset --out $Out

# --- Step 2 -----------------------------------------------------------------
python -m pipeline.step2 --artifacts $Out --mode $Mode --fresh

# --- Step 3 -----------------------------------------------------------------
$step3 = @("python", "-m", "pipeline.step3", "--artifacts", $Out, "--mode", $Mode)
if (Test-Path $ParamsFile) { $step3 += @("--params-file", $ParamsFile) }
& @step3

# --- Step 4 -----------------------------------------------------------------
python -m pipeline.step4 --artifacts $Out

Write-Host ""
Write-Host "=== Pipeline complete (mode=$Mode) ==="
Write-Host "  artifacts : $Out/"
Write-Host "  policy    : $Out/rl_policy_tuned.zip"
Write-Host "  zones     : $Out/zones.yaml"
