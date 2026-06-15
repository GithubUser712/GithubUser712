# AGENTS.md

## Cursor Cloud specific instructions

This is an **offline Python pipeline** (no servers, databases, or long-running
daemons). It turns a track image into a racing line for an F1TENTH-style RC car.
Stages run on demand as `python -m pipeline.stepN` and share an `artifacts/`
directory; each stage consumes the previous stage's outputs. See `README.md`
for the full per-stage command reference and map parameters.

### Environment
- Python 3.12 with a virtualenv at `.venv` (created by the update script).
  Activate it (`source .venv/bin/activate`) or call binaries directly
  (`.venv/bin/python`). Dependencies are not installed system-wide.
- Torch is installed CPU-only (from the PyTorch CPU index) on purpose; the
  default `torch` wheel pulls in large CUDA packages that aren't needed here.

### Running the pipeline (dev)
Run in order, reusing the same artifacts dir. Quick end-to-end smoke test:
```
python tools/make_sample_map.py
python -m pipeline.step1 --map maps/sample_track.png --resolution 0.05 --start 873,350 --heading -72
python -m pipeline.step2 --timesteps 20000   # tiny run to confirm RL works; real training default is 1,000,000
```
- `step2` is the only heavyweight stage (PPO training, CPU-intensive). The
  default 1M timesteps takes a long time; pass `--timesteps`/`--resume` while
  iterating. A short run will not complete a lap, so it won't emit
  `raceline.csv` — that is expected, not a failure.
- `pipeline.watch` opens an OpenCV GUI window; for headless use pass
  `--video out.mp4`.
- `car.bench` (`python -m car.bench`) needs physical VESC hardware on a serial
  port and cannot be exercised in the cloud VM.

### Lint / test / build
There is **no lint config, no automated test suite, and no build step** in this
repo. "Testing" means running the pipeline stages and inspecting
`artifacts/debug.png` (and later stage overlays). Do not assume a `pytest` or
lint command exists.

### Repo layout note
The `main` branch contains only a placeholder `README.md`; the actual pipeline
code lives on the `cursor/step*` feature branches (most complete:
`cursor/step5-6-vesc-3d48`). The update script guards on `requirements.txt`
existing, so it is a no-op on the empty `main` branch.
