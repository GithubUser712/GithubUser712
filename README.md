# Self-Driving RC Car — Raceline Pipeline

Offline pipeline that turns a hand-drawn (or SLAM-generated) track image into
an optimal racing line with braking/acceleration zones for an F1TENTH-style
RC car (Jetson Orin Nano + VESC 6 + RPLidar A2M12, ROS 2 Jazzy).

Pipeline stages:

1. **Map ingestion** (this repo, `pipeline/step1.py`) — implemented
2. RL racing line — planned
3. Vehicle parameters + constrained recompute — planned
4. Braking / acceleration zone dictation — planned
5. f1tenth_gym validation + real-car ROS 2 deployment — planned

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Step 1: map ingestion

Map format: **black (dark) pixels = 1 = walls, white (light) pixels = 0 = free
space**. The track must be a single closed loop (closed outer wall, closed
inner wall).

Generate a test map, then run step 1 (interactive — it prompts for anything
you don't pass as a flag):

```bash
python tools/make_sample_map.py
python -m pipeline.step1
```

Or fully scripted:

```bash
python -m pipeline.step1 --map maps/sample_track.png --resolution 0.05 \
    --start 873,350 --heading -69
```

Outputs in `artifacts/`:

| file             | contents                                              |
|------------------|-------------------------------------------------------|
| `map.pgm/.yaml`  | occupancy grid, ROS map_server / f1tenth_gym format   |
| `centerline.csv` | ordered waypoints: `x_m, y_m, clearance_m`            |
| `track_meta.yaml`| resolution, start pose, track length/width statistics |
| `debug.png`      | visual overlay — check this before trusting the rest  |
