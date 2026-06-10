# Self-Driving RC Car — Raceline Pipeline

Offline pipeline that turns a hand-drawn (or SLAM-generated) track image into
an optimal racing line with braking/acceleration zones for an F1TENTH-style
RC car (Jetson Orin Nano + VESC 6 + RPLidar A2M12, ROS 2 Jazzy).

Pipeline stages:

1. **Map ingestion** (`pipeline/step1.py`) — implemented
2. **RL racing line** (`pipeline/step2.py`) — implemented
3. **Vehicle parameters + recalculation** (`pipeline/step3.py`) — implemented
4. **Braking / acceleration zones** (`pipeline/step4.py`) — implemented
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

## Step 2: RL racing line

Trains a PPO agent (simulated lidar in, steering + throttle out) to lap the
track, printing one ping per finished run with its rewards and punishments.
The best deterministic lap is then smoothed into the racing line.

```bash
python -m pipeline.step2                      # train 1M steps on artifacts/
python -m pipeline.step2 --resume             # keep training a saved policy
python -m pipeline.step2 --eval-only          # just re-extract the raceline
```

Additional outputs in `artifacts/`:

| file                 | contents                                         |
|----------------------|--------------------------------------------------|
| `rl_policy.zip`      | trained PPO weights (resumable)                  |
| `raceline.csv`       | racing line waypoints: `x_m, y_m, curvature_1pm` |
| `raceline_debug.png` | racing line over the map, coloured by curvature  |

Note: pip's default `torch` wheel on x86 Linux bundles CUDA and is large; for
CPU-only training install it first with
`pip install torch --index-url https://download.pytorch.org/whl/cpu`.

## Step 3: car parameters + racing line recalculation

Prompts for the real car's measurements (weight, max/min speed, wheelbase
width/length, chassis width/length, turning radius, tire grip coefficient),
derives the physical limits (steering angle, lateral grip, traction-limited
acceleration, wall margin), bakes them into the simulator, fine-tunes the
step-2 policy under the new physics and re-extracts the racing line.

```bash
python -m pipeline.step3                                    # prompts, then trains
python -m pipeline.step3 --params-file artifacts/car_params.yaml   # no prompts
```

Outputs: `car_params.yaml`, `rl_policy_tuned.zip`, `raceline_tuned.csv`,
`raceline_tuned_debug.png`.

## Step 4: braking / acceleration zone dictation

Computes the fastest physically-possible speed at every waypoint
(curvature cap + friction-circle forward/backward passes), splits the lap
into ACCEL / BRAKE / HOLD zones, prints the dictation, and writes the final
trajectory the trackers will follow.

```bash
python -m pipeline.step4
```

Outputs: `raceline_final.csv` (`s, x, y, curvature, speed, zone` per
waypoint), `zones.yaml`, `zones_map.png`, `speed_profile.png`.
