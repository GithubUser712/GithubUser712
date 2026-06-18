"""Step 5 entry point: validate the final raceline with a tracking controller.

    python -m pipeline.step5                      # follow raceline_final.csv
    python -m pipeline.step5 --laps 5 --video tracking.mp4

Drives the dynamic simulator with a classic PURE PURSUIT controller (not the
RL policy): steer at a lookahead point on the racing line, P-control the
speed to the step-4 velocity profile.  This is exactly the controller that
runs on the real car later -- the RL stage *designed* the line, the tracker
*follows* it, in sim and on hardware alike.

Reports lap times, cross-track error and speed-tracking error.  If this step
laps cleanly, the raceline + speed profile are trustworthy enough to take to
the car (step 6).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from raceline.control import PurePursuit
from raceline.core import PreflightError, check_step_prerequisites
from .track_env import TrackData, TrackEnv, load_track_data


def load_raceline(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """raceline_final.csv: s, x, y, curvature, speed, zone(text)."""
    data = np.genfromtxt(path, delimiter=",", skip_header=1,
                         usecols=(1, 2, 4))
    return data[:, :2], data[:, 2]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Step 5: track the final raceline with pure pursuit.")
    ap.add_argument("--artifacts", type=str, default="artifacts")
    ap.add_argument("--laps", type=int, default=3)
    ap.add_argument("--video", type=str, help="record the run to an MP4")
    ap.add_argument("--lookahead-gain", type=float, default=0.30,
                    help="lookahead distance = gain * speed (default 0.30; "
                         "too high cuts corners at the limit, too low "
                         "oscillates)")
    ap.add_argument("--speed-scale", type=float, default=1.0,
                    help="scale the target speeds (e.g. 0.5 for a careful "
                         "first hardware run)")
    args = ap.parse_args(argv)

    try:
        out_dir = check_step_prerequisites(5, args.artifacts)
    except PreflightError as exc:
        print(f"ERROR: {exc}")
        return 2
    track = load_track_data(out_dir)

    raceline_path = out_dir / "raceline_final.csv"
    if not raceline_path.is_file():
        print("ERROR: raceline_final.csv not found -- run step 4 first.")
        return 2
    xy, speeds = load_raceline(raceline_path)
    speeds = speeds * args.speed_scale

    params = None
    params_path = out_dir / "car_params.yaml"
    if params_path.is_file():
        from .step3 import load_car_params
        params, _, _ = load_car_params(params_path)
    env = TrackEnv(track, params=params, random_spawn=False, laps=args.laps)
    controller = PurePursuit(xy, speeds, env.p,
                             lookahead_gain=args.lookahead_gain)

    print(f"Tracking {raceline_path.name}: {len(xy)} waypoints, "
          f"target speeds {speeds.min():.2f}-{speeds.max():.2f} m/s, "
          f"{args.laps} lap(s)")

    writer = None
    viewer = None
    if args.video:
        import cv2
        from .watch import Viewer
        viewer = Viewer(track, env.p.chassis_l_m, env.p.chassis_w_m,
                        show_beams=False)
        writer = cv2.VideoWriter(args.video, cv2.VideoWriter_fourcc(*"mp4v"),
                                 int(round(1.0 / env.dt)), viewer.out_size)

    obs, _ = env.reset(seed=0)
    controller.reset(np.array([env.state.x, env.state.y]))
    errs, v_errs = [], []
    lap_marks: list[float] = []
    last_lap_progress = 0.0

    while True:
        action, err, v_target = controller.control(env.state)
        errs.append(err)
        v_errs.append(abs(env.state.vx - v_target))
        obs, _, terminated, truncated, info = env.step(action)

        # per-lap split times
        lap_no = int(env._total_progress / track.length_m)
        if lap_no > last_lap_progress:
            lap_marks.append(env._steps * env.dt)
            last_lap_progress = lap_no

        if writer is not None:
            hud = [(f"pure pursuit  lap {env._total_progress / track.length_m:5.2f}"
                    f"/{args.laps}  v {env.state.vx:4.1f} m/s "
                    f"(target {v_target:4.1f})", (120, 60, 20), 0),
                   (f"cross-track err {err:5.2f} m", (120, 60, 20), 1)]
            writer.write(viewer.frame([env], [obs], [action], hud, []))

        if terminated or truncated:
            rs = info["run_summary"]
            break

    if writer is not None:
        writer.release()
        print(f"wrote {args.video}")

    errs = np.array(errs)
    v_errs = np.array(v_errs)
    print(f"\nResult: progress {rs['progress_pct']:.1f}% of {args.laps} laps"
          + (", CRASHED" if rs["crashed"] else ""))
    if lap_marks:
        splits = np.diff([0.0] + lap_marks)
        for i, t in enumerate(splits, 1):
            print(f"  lap {i}: {t:6.2f} s" + ("  (standing start)" if i == 1 else ""))
    print(f"  cross-track error: mean {errs.mean():.3f} m / max {errs.max():.3f} m")
    print(f"  speed error:       mean {v_errs.mean():.2f} m/s")

    if rs["crashed"] or rs["lap_time_s"] is None:
        print("\nTracking failed -- tune --lookahead-gain (higher = smoother, "
              "cuts corners less aggressively) or test with --speed-scale 0.7.")
        return 1
    print("\nStep 5 complete: the raceline is followable by a geometric "
          "controller. Step 6 connects the same controller to the VESC.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
