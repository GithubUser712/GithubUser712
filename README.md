# Self-Driving RC Car — Raceline Pipeline v2

Offline pipeline that turns a hand-drawn (or SLAM-generated) track image into
an optimal racing line with braking/acceleration zones for an F1TENTH-style
RC car (Jetson Orin Nano + VESC 6 + RPLidar A2M12, ROS 2 Jazzy).

## Jetson Orin Nano — start here

**Step 0 (setup, one time):**
```bash
git clone https://github.com/GithubUser712/GithubUser712.git
cd GithubUser712
git checkout cursor/step5-6-vesc-3d48    # main branch has no code yet
bash scripts/jetson_setup.sh
```

**Step 1 (map ingestion, one command):**
```bash
source .venv/bin/activate
python -m pipeline.step1 --preset sample
```

Or use the wrapper script:
```bash
bash scripts/run_step1.sh
```

Verify setup anytime:
```bash
python -m raceline.setup --verify
```

**Before step 2 (RL training)** install the heavy ML packages:
```bash
bash scripts/jetson_install_rl.sh
```

---

## Project layout

```
raceline/              Core library (physics, validation, Gazebo export)
  core/                Preflight checks, typed errors
  physics/             Pacejka tires, VESC motor, RK4 vehicle model
  control/             Pure pursuit tracker
  gazebo/              SDF world export from artifacts
pipeline/              CLI entry points (steps 1–5) — unchanged interface
car/                   VESC bench tools (step 6)
ros2/raceline_gazebo/  3D Gazebo Harmonic simulation package
tools/                 Map generators, Gazebo export script
config/                Default vehicle parameters
tests/                 Unit tests (pytest)
maps/                  Track images + F1 circuit GeoJSON sources
```

## Pipeline stages

| Step | Module | Status |
|------|--------|--------|
| 1 | `pipeline/step1.py` | Map ingestion |
| 2 | `pipeline/step2.py` | RL racing line (PPO) |
| 3 | `pipeline/step3.py` | Car parameters + physics retune |
| 4 | `pipeline/step4.py` | Braking / acceleration zones |
| 5 | `pipeline/step5.py` | Pure pursuit validation |
| 6 | `car/bench.py` | VESC hardware bring-up |
| 7 | Lidar + ROS 2 autonomy | Planned |

Every step runs **preflight checks** before starting — missing artifacts produce
a clear error with the exact command to run next.

## Setup (non-Jetson / manual)

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-step1.txt
pip install -e . --no-deps
python -m raceline.setup --verify
```

For RL training (step 2+): `pip install -e ".[rl]"` or `pip install -e ".[dev]"`.

## Quick start

```bash
python -m pipeline.step1 --preset sample   # recommended — no prompts
python -m pipeline.step2
python -m pipeline.watch              # optional: visualize training
python -m pipeline.step3
python -m pipeline.step4
python -m pipeline.step5 --laps 3
```

## Physics (v2)

The simulator uses a **high-fidelity dynamic single-track model**:

- **Pacejka MF 6.1** lateral and longitudinal tire forces with combined-slip friction ellipse
- **Longitudinal + lateral load transfer** from CoG height and track width
- **VESC motor dynamics** — wheel speed, ERPM, gear ratio, torque limits
- **Servo actuator** — rate limit + first-order lag
- **RK4 integration** at 200 Hz (4 substeps × 50 Hz env step)
- Low-speed kinematic blend below 1 m/s (industry standard)

Physics lives in `raceline/physics/`; `pipeline/bicycle.py` re-exports for compatibility.

## 3D simulation (Gazebo)

Export your track to a Gazebo Harmonic world:

```bash
python -m pipeline.step1 --map maps/sample_track.png --resolution 0.05 \
    --start 873,350 --heading -72
python tools/export_gazebo_world.py --artifacts artifacts --vehicle-model
```

Then in a ROS 2 Jazzy workspace:

```bash
cd ros2 && colcon build --packages-select raceline_gazebo
source install/setup.bash
ros2 launch raceline_gazebo sim.launch.py
```

The exported `track.sdf` contains a ground plane and downsampled wall collision
boxes derived from your occupancy grid. The URDF in `ros2/raceline_gazebo/urdf/`
provides an F1TENTH-scale vehicle stub — attach `gz-sim` ackermann or
`ros2_control` plugins for full closed-loop driving.

## Testing

```bash
pytest tests/ -v
```

## Configuration

Default vehicle parameters: `config/default_car.yaml`. Override in step 3 or pass
`--params-file artifacts/car_params.yaml`.

## Hardware (Step 6)

```bash
python -m car.bench --port /dev/ttyACM0    # Windows: COM3
```

## Map reference (0.01 m/px — default)

All presets use **0.01 m/px** and **0.01 m waypoint spacing**.  
Regenerate maps after pulling: `python tools/make_sample_map.py && python tools/make_f1_tracks.py`

| Track | `--preset` | Start `x,y` | Heading (°) | Image size (px) | RC lap |
|-------|------------|-------------|-------------|-----------------|--------|
| Sample | `sample` | **4365, 1750** | **-72** | 5000 × 3500 | ~95 m |
| Monaco | `monaco` | **5083, 2180** | **108** | 7914 × 10292 | ~333 m |
| Spa | `spa` | **4136, 2483** | **125** | 13315 × 21214 | ~699 m |
| Nürburgring | `nurburgring` | **9878, 2720** | **-135** | 11849 × 15907 | ~515 m |

Coordinates are in `maps/*.meta.yaml` (auto-loaded by `--preset`).

```bash
python -m pipeline.step1 --preset sample      # -> artifacts/
python -m pipeline.step1 --preset monaco      # -> use --out artifacts_monaco
python -m pipeline.step1 --preset spa         --out artifacts_spa
python -m pipeline.step1 --preset nurburgring --out artifacts_nurburgring
```

### Regenerating maps at 0.01 m/px

```bash
python tools/make_sample_map.py --resolution 0.01
python tools/make_f1_tracks.py
```

F1 PNGs are large (~50–200 MB each). Generate on your Jetson rather than cloning them.
