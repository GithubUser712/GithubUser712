"""Watch the agent's attempts: live window or MP4 recording.

    python -m pipeline.watch                      # live window, trained policy
    python -m pipeline.watch --video runs.mp4     # record instead (headless OK)
    python -m pipeline.watch --stochastic --random-spawn   # training-style runs
    python -m pipeline.watch --random             # untrained baseline (chaos)

Draws the occupancy grid, the car (to scale), its 27 lidar beams, the path
driven so far, and a HUD with the same numbers as the training pings
(rewards, punishments, progress, lap times).

Works with step-2 artifacts alone.  If step 3 has been run, the tuned policy
and your real car parameters are picked up automatically.

While training is running (step 2 or 3), the policy file is autosaved every
50k steps -- so you can run this in a second terminal to check on what the
agent has learned so far.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .bicycle import CarParams
from .track_env import TrackData, TrackEnv, load_track_data

# colours (BGR)
_WALL = (40, 40, 40)
_FREE = (255, 255, 255)
_TRAIL = (60, 170, 60)
_BEAM = (230, 190, 150)
_CAR = (160, 80, 20)
_NOSE = (40, 40, 230)
_HUD = (120, 60, 20)
_GOOD = (60, 170, 60)
_BAD = (40, 40, 230)


class Viewer:
    def __init__(self, track: TrackData, chassis_l_m: float, chassis_w_m: float):
        self.track = track
        occ = track.grid.occupancy
        self.bg = np.where(occ[..., None] == 1, _WALL, _FREE).astype(np.uint8)
        self.chassis_l = chassis_l_m
        self.chassis_w = chassis_w_m

    def _to_px(self, xy: np.ndarray) -> np.ndarray:
        """world (x, y) -> opencv (col, row) int points."""
        rc = self.track.grid.pixel_from_world(xy)
        return np.rint(rc[..., ::-1]).astype(np.int32)

    def frame(self, env: TrackEnv, obs: np.ndarray, hud: list[str],
              banner: tuple[str, tuple] | None = None) -> np.ndarray:
        img = self.bg.copy()
        s = env.state

        # path driven so far
        if len(env.trajectory) > 1:
            trail = self._to_px(np.asarray(env.trajectory))
            cv2.polylines(img, [trail], False, _TRAIL, 2)

        # lidar beams (the observation is normalised distances)
        n = len(env.beam_angles)
        dists = obs[:n] * env.max_range
        angles = s.yaw + env.beam_angles
        origin = np.array([s.x, s.y])
        ends = origin + dists[:, None] * np.stack(
            [np.cos(angles), np.sin(angles)], axis=1)
        o_px = self._to_px(origin)
        for e_px in self._to_px(ends):
            cv2.line(img, tuple(o_px), tuple(e_px), _BEAM, 1)

        # the car, drawn to scale
        half_l, half_w = self.chassis_l / 2.0, self.chassis_w / 2.0
        corners_local = np.array([[+half_l, +half_w], [+half_l, -half_w],
                                  [-half_l, -half_w], [-half_l, +half_w]])
        rot = np.array([[np.cos(s.yaw), -np.sin(s.yaw)],
                        [np.sin(s.yaw), np.cos(s.yaw)]])
        corners = origin + corners_local @ rot.T
        cv2.fillPoly(img, [self._to_px(corners)], _CAR)
        nose = origin + rot @ np.array([half_l, 0.0])
        cv2.line(img, tuple(o_px), tuple(self._to_px(nose)), _NOSE, 2)

        # HUD
        for i, line in enumerate(hud):
            cv2.putText(img, line, (10, 22 + 20 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, _HUD, 1, cv2.LINE_AA)
        if banner is not None:
            text, colour = banner
            cv2.putText(img, text, (10, img.shape[0] - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, colour, 3, cv2.LINE_AA)
        return img


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Watch the agent's attempts.")
    ap.add_argument("--artifacts", type=str, default="artifacts")
    ap.add_argument("--policy", type=str,
                    help="policy zip (default: rl_policy_tuned.zip if it "
                         "exists, else rl_policy.zip)")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--video", type=str,
                    help="write an MP4 instead of opening a window")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="live playback speed multiplier (default 1.0)")
    ap.add_argument("--stochastic", action="store_true",
                    help="sample actions like during training (wobblier)")
    ap.add_argument("--random-spawn", action="store_true",
                    help="spawn anywhere on track, like training episodes")
    ap.add_argument("--random", action="store_true",
                    help="ignore the policy, take random actions")
    args = ap.parse_args(argv)

    out_dir = Path(args.artifacts)
    track = load_track_data(out_dir)

    # pick up step-3 results when present, otherwise generic step-2 setup
    params, note = None, "generic car (step 2)"
    chassis_l, chassis_w = 0.50, 0.30
    params_path = out_dir / "car_params.yaml"
    if params_path.is_file():
        from .step3 import load_car_params
        params, _, measured = load_car_params(params_path)
        chassis_l = measured["chassis_length_m"]
        chassis_w = measured["chassis_width_m"]
        note = "your car (step 3)"

    model = None
    if not args.random:
        policy_path = (Path(args.policy) if args.policy else None)
        if policy_path is None:
            tuned = out_dir / "rl_policy_tuned.zip"
            base = out_dir / "rl_policy.zip"
            policy_path = tuned if tuned.is_file() else base
        if not policy_path.is_file():
            print(f"ERROR: no policy found at {policy_path} -- train with "
                  "step 2 first, or pass --random to watch an untrained car.")
            return 2
        from stable_baselines3 import PPO
        model = PPO.load(policy_path, device="cpu")
        print(f"Watching {policy_path.name} | physics: {note}")
    else:
        print(f"Watching RANDOM actions | physics: {note}")

    env = TrackEnv(track, params=params, random_spawn=args.random_spawn)
    viewer = Viewer(track, chassis_l, chassis_w)

    writer = None
    if args.video:
        fps = int(round(1.0 / env.dt))
        h, w = viewer.bg.shape[:2]
        writer = cv2.VideoWriter(args.video,
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not writer.isOpened():
            print("ERROR: could not open video writer (codec missing?).")
            return 2
    else:
        try:
            cv2.namedWindow("attempt viewer", cv2.WINDOW_NORMAL)
        except cv2.error:
            print("ERROR: no display available -- use --video out.mp4 instead.")
            return 2

    best_lap = None
    frame_budget_s = env.dt / max(args.speed, 0.01)
    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset(seed=ep)
        banner = None
        while True:
            t0 = time.time()
            if model is None:
                action = env.action_space.sample()
            else:
                action, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, _, terminated, truncated, info = env.step(action)

            hud = [f"run {ep}/{args.episodes}   step {env._steps}   "
                   f"speed {env.state.v:4.1f} m/s",
                   f"progress {100 * env._total_progress / track.length_m:5.1f}%"
                   f"   reward +{env._reward_sum:.1f}"
                   f"   punishment -{env._punish_sum:.1f}",
                   f"best lap {best_lap:.2f} s" if best_lap else "best lap --"]

            done = terminated or truncated
            if done:
                rs = info["run_summary"]
                if rs["lap_time_s"] is not None:
                    if best_lap is None or rs["lap_time_s"] < best_lap:
                        best_lap = rs["lap_time_s"]
                    banner = (f"LAP {rs['lap_time_s']:.2f} s", _GOOD)
                elif rs["crashed"]:
                    banner = ("CRASH", _BAD)
                else:
                    banner = ("TIME LIMIT", _BAD)
                print(f"run {ep:3d} | reward +{rs['reward']:8.2f} | "
                      f"punishment -{rs['punishment']:7.2f} | "
                      f"progress {rs['progress_pct']:5.1f}%"
                      + (f" | lap {rs['lap_time_s']:.2f} s"
                         if rs["lap_time_s"] else ""))

            img = viewer.frame(env, obs, hud, banner)
            # hold the end-of-run banner for a moment
            repeats = int(round(1.0 / env.dt)) if done else 1
            for _ in range(repeats):
                if writer is not None:
                    writer.write(img)
                else:
                    cv2.imshow("attempt viewer", img)
                    wait_ms = max(1, int(1000 * (frame_budget_s
                                                 - (time.time() - t0))))
                    if cv2.waitKey(wait_ms) & 0xFF in (27, ord("q")):
                        print("quit.")
                        if writer is not None:
                            writer.release()
                        return 0
            if done:
                break

    if writer is not None:
        writer.release()
        print(f"wrote {args.video}")
    else:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
