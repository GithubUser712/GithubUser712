"""Step 3 entry point: enter the real car's parameters and recalculate the
racing line with proper physics.

    python -m pipeline.step3                      # prompts for every parameter
    python -m pipeline.step3 --params-file artifacts/car_params.yaml
    python -m pipeline.step3 --timesteps 800000 --device cuda

The prompts ask for exactly the quantities in the project spec: weight, max
speed, min speed, wheelbase width & length, chassis width & length, turning
radius and tire grip coefficient.  From those it derives the physical limits
(steering angle, lateral grip, acceleration caps, wall safety margin), bakes
them into the simulator, fine-tunes the step-2 policy under the new physics,
and extracts the recalculated racing line.
"""

from __future__ import annotations

import argparse
import math
import sys
from functools import partial
from pathlib import Path

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from .bicycle import CarParams
from .step2 import (AutosaveCallback, PingCallback, best_deterministic_lap,
                    save_raceline, smooth_raceline)
from .track_env import TrackData, TrackEnv, load_track_data

G = 9.81

# prompt text, dict key, unit, (sane minimum, sane maximum)
_PARAM_SPEC = [
    ("Car weight",                 "mass_kg",            "kg", 0.1,  50.0),
    ("Max speed",                  "max_speed_mps",      "m/s", 0.5, 40.0),
    ("Min speed",                  "min_speed_mps",      "m/s", 0.0, 10.0),
    ("Wheelbase width (track width, left-right wheel centres)",
                                   "wheelbase_width_m",  "m",  0.05, 1.0),
    ("Wheelbase length (front-rear axle distance)",
                                   "wheelbase_length_m", "m",  0.05, 1.5),
    ("Chassis width (widest point incl. tires)",
                                   "chassis_width_m",    "m",  0.05, 1.0),
    ("Chassis length (longest point incl. bumpers)",
                                   "chassis_length_m",   "m",  0.05, 1.5),
    ("Minimum turning radius",     "turning_radius_m",   "m",  0.1, 10.0),
    ("Tire grip coefficient (mu)", "tire_grip_mu",       "-",  0.1,  2.0),
]


def prompt_measurements() -> dict:
    print("Enter your car's parameters (SI units).\n")
    measured = {}
    for label, key, unit, lo, hi in _PARAM_SPEC:
        while True:
            raw = input(f"  {label} [{unit}]: ").strip()
            try:
                v = float(raw)
            except ValueError:
                print("    !! enter a number.")
                continue
            if not (lo <= v <= hi):
                print(f"    !! expected something between {lo} and {hi} {unit}.")
                continue
            measured[key] = v
            break
    if measured["min_speed_mps"] >= measured["max_speed_mps"]:
        print("\nERROR: min speed must be below max speed -- start over.")
        return prompt_measurements()
    return measured


def derive_params(m: dict) -> tuple[CarParams, dict]:
    """Turn tape-measure quantities into dynamic-model parameters.

    max steering angle : bicycle geometry, R = L / tan(delta)
    accel / brake caps : traction-limited (mu * g); the dynamic model then
                         further limits them through per-axle traction circles
    wall safety radius : half the chassis width + 5 cm margin, since the
                         distance field measures from the car's centre point
    yaw inertia        : box estimate from mass and chassis dimensions
    """
    max_steer = math.atan(m["wheelbase_length_m"] / m["turning_radius_m"])
    a_grip = m["tire_grip_mu"] * G
    params = CarParams(
        wheelbase_m=m["wheelbase_length_m"],
        max_speed_mps=m["max_speed_mps"],
        min_speed_mps=m["min_speed_mps"],
        max_steer_rad=max_steer,
        max_accel_mps2=a_grip,
        max_brake_mps2=a_grip,
        safety_radius_m=m["chassis_width_m"] / 2.0 + 0.05,
        mu=m["tire_grip_mu"],
        mass_kg=m["mass_kg"],
        chassis_l_m=m["chassis_length_m"],
        chassis_w_m=m["chassis_width_m"],
    )
    derived = {
        "max_steer_rad": round(max_steer, 4),
        "max_steer_deg": round(math.degrees(max_steer), 1),
        "max_accel_mps2": round(a_grip, 3),
        "max_brake_mps2": round(a_grip, 3),
        "safety_radius_m": round(params.safety_radius_m, 3),
        "yaw_inertia_kgm2": round(params.izz, 4),
        "mass_kg": m["mass_kg"],
        "cda_m2": params.cda_m2,
        "crr": params.crr,
    }
    return params, derived


def load_car_params(path: Path) -> tuple[CarParams, dict, dict]:
    with open(path) as f:
        data = yaml.safe_load(f)
    params, derived = derive_params(data["measured"])
    return params, derived, data["measured"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Step 3: real car parameters + racing line recalculation.")
    ap.add_argument("--artifacts", type=str, default="artifacts")
    ap.add_argument("--params-file", type=str,
                    help="load measurements from a saved car_params.yaml "
                         "instead of prompting")
    ap.add_argument("--timesteps", type=int, default=600_000,
                    help="fine-tuning budget under the new physics")
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--laps", type=int, default=1,
                    help="laps per run during training (default 1)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--fresh", action="store_true",
                    help="train from scratch instead of warm-starting from "
                         "the step-2 policy")
    ap.add_argument("--skip-training", action="store_true",
                    help="reuse the already-tuned policy, just re-extract")
    args = ap.parse_args(argv)

    out_dir = Path(args.artifacts)
    track = load_track_data(out_dir)

    # 1. parameters in ----------------------------------------------------
    if args.params_file:
        params, derived, measured = load_car_params(Path(args.params_file))
        print(f"Loaded car parameters from {args.params_file}")
    else:
        measured = prompt_measurements()
        params, derived = derive_params(measured)

    print("\nDerived physical limits (dynamic single-track model):")
    print(f"  max steering angle : {derived['max_steer_deg']:.1f} deg "
          f"(from {measured['turning_radius_m']} m turning radius)")
    print(f"  accel/brake cap    : {derived['max_accel_mps2']:.2f} m/s^2 "
          f"(traction-limited, mu = {measured['tire_grip_mu']})")
    print(f"  wall safety radius : {derived['safety_radius_m']:.3f} m "
          f"(chassis width {measured['chassis_width_m']} m + margin)")
    print(f"  yaw inertia        : {derived['yaw_inertia_kgm2']:.4f} kg m^2 "
          f"(from {measured['mass_kg']} kg + chassis dimensions)")
    print("  weight now matters: load transfer shifts grip between axles "
          "under braking/acceleration.\n")

    params_path = out_dir / "car_params.yaml"
    with open(params_path, "w") as f:
        yaml.safe_dump({"measured": measured, "derived": derived},
                       f, sort_keys=False)
    print(f"Saved {params_path}\n")

    # 2. recalculate the racing line under the new physics ----------------
    tuned_policy = out_dir / "rl_policy_tuned.zip"
    base_policy = out_dir / "rl_policy.zip"

    if args.skip_training:
        if not tuned_policy.is_file():
            print(f"ERROR: {tuned_policy} not found -- run without "
                  "--skip-training first.")
            return 2
        model = PPO.load(tuned_policy, device=args.device)
    else:
        factory = partial(TrackEnv, track, params, laps=args.laps)
        vec_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
        venv = vec_cls([factory for _ in range(args.n_envs)])

        model = None
        if not args.fresh:
            for warm in (tuned_policy, base_policy):
                if not warm.is_file():
                    continue
                try:
                    model = PPO.load(warm, env=venv, device=args.device)
                    print(f"Warm-starting from {warm} (same network, "
                          "new physics)")
                    break
                except Exception:
                    print(f"NOTE: {warm.name} is incompatible (saved before "
                          "a physics/observation upgrade) -- skipping it")
        if model is None:
            model = PPO("MlpPolicy", venv, seed=args.seed, verbose=0,
                        learning_rate=3e-4, n_steps=1024, batch_size=256,
                        gamma=0.995, gae_lambda=0.95, ent_coef=0.005,
                        policy_kwargs=dict(net_arch=[256, 256]),
                        device=args.device)
        print(f"Recalculating racing line: {args.timesteps:,} timesteps with "
              "your car's physics -- one ping per finished run:\n")
        from stable_baselines3.common.callbacks import CallbackList
        model.learn(total_timesteps=args.timesteps,
                    callback=CallbackList([PingCallback(),
                                           AutosaveCallback(tuned_policy)]))
        model.save(tuned_policy)
        venv.close()
        print(f"\nTuned policy saved to {tuned_policy}")

    print("\nExtracting recalculated racing line:")
    traj, lap_time = best_deterministic_lap(model, track, params=params)
    if traj is None:
        print("\nNo complete lap under the new physics yet -- fine-tune more:\n"
              f"  python -m pipeline.step3 --params-file {params_path} "
              f"--timesteps {args.timesteps}")
        return 1

    raceline = smooth_raceline(traj, track.spacing_m)
    written = save_raceline(track, raceline, lap_time, out_dir,
                            stem="raceline_tuned", note="your car's parameters")
    print(f"\nBest lap with your car's physics: {lap_time:.2f} s")
    print("Wrote:")
    for p in written:
        print(f"  {p}")
    print("\nStep 3 complete. Run step 4 to dictate the braking and "
          "acceleration zones:\n  python -m pipeline.step4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
