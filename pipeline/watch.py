"""Watch the agent's attempts: live window or MP4 recording.

    python -m pipeline.watch                      # live window, trained policy
    python -m pipeline.watch --laps 3             # 3-lap runs
    python -m pipeline.watch --cars 4             # 4 cars running at once
    python -m pipeline.watch --video runs.mp4     # record instead (headless OK)
    python -m pipeline.watch --stochastic         # sampled actions (wobblier)
    python -m pipeline.watch --random             # untrained baseline (chaos)

Draws the occupancy grid, the car(s) to scale, lidar beams, the path driven
so far, throttle/brake and steering indicators, and a HUD with the same
numbers as the training pings (rewards, punishments, progress, lap times).

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

from .track_env import TrackData, TrackEnv, load_track_data

# colours (BGR)
_WALL = (40, 40, 40)
_FREE = (255, 255, 255)
_BEAM = (230, 190, 150)
_NOSE = (40, 40, 230)
_HUD = (120, 60, 20)
_GOOD = (60, 170, 60)
_BAD = (40, 40, 230)
_CAR_COLOURS = [(160, 80, 20), (20, 140, 200), (140, 20, 140), (20, 160, 60),
                (0, 90, 200), (90, 90, 90), (150, 150, 20), (60, 20, 120)]


class Viewer:
    def __init__(self, track: TrackData, chassis_l_m: float, chassis_w_m: float,
                 show_beams: bool):
        self.track = track
        occ = track.grid.occupancy
        self.bg = np.where(occ[..., None] == 1, _WALL, _FREE).astype(np.uint8)
        self.chassis_l = chassis_l_m
        self.chassis_w = chassis_w_m
        self.show_beams = show_beams

    def _to_px(self, xy: np.ndarray) -> np.ndarray:
        """world (x, y) -> opencv (col, row) int points."""
        rc = self.track.grid.pixel_from_world(xy)
        return np.rint(rc[..., ::-1]).astype(np.int32)

    def _draw_car(self, img, env: TrackEnv, obs: np.ndarray, colour) -> None:
        s = env.state
        origin = np.array([s.x, s.y])
        o_px = self._to_px(origin)

        if len(env.trajectory) > 1:
            trail = self._to_px(np.asarray(env.trajectory))
            cv2.polylines(img, [trail], False, colour, 1)

        if self.show_beams:
            n = len(env.beam_angles)
            dists = obs[:n] * env.max_range
            angles = s.yaw + env.beam_angles
            ends = origin + dists[:, None] * np.stack(
                [np.cos(angles), np.sin(angles)], axis=1)
            for e_px in self._to_px(ends):
                cv2.line(img, tuple(o_px), tuple(e_px), _BEAM, 1)

        half_l, half_w = self.chassis_l / 2.0, self.chassis_w / 2.0
        corners_local = np.array([[+half_l, +half_w], [+half_l, -half_w],
                                  [-half_l, -half_w], [-half_l, +half_w]])
        rot = np.array([[np.cos(s.yaw), -np.sin(s.yaw)],
                        [np.sin(s.yaw), np.cos(s.yaw)]])
        corners = origin + corners_local @ rot.T
        cv2.fillPoly(img, [self._to_px(corners)], colour)
        nose = origin + rot @ np.array([half_l, 0.0])
        cv2.line(img, tuple(o_px), tuple(self._to_px(nose)), _NOSE, 2)

    def _draw_pedals(self, img, action: np.ndarray, base_y: int) -> None:
        """Throttle/brake and steering bars (bottom-left)."""
        cx, w, h = 90, 70, 12
        for row, (label, val, pos_col, neg_col) in enumerate([
                ("throttle", float(action[1]), _GOOD, _BAD),
                ("steer", float(action[0]), _HUD, _HUD)]):
            y = base_y + row * (h + 10)
            cv2.putText(img, label, (8, y + h - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, _HUD, 1, cv2.LINE_AA)
            cv2.rectangle(img, (cx, y), (cx + 2 * w, y + h), (200, 200, 200), 1)
            cv2.line(img, (cx + w, y), (cx + w, y + h), (160, 160, 160), 1)
            v = max(-1.0, min(1.0, val))
            colour = pos_col if v >= 0 else neg_col
            cv2.rectangle(img, (cx + w, y + 2),
                          (cx + w + int(v * (w - 2)), y + h - 2), colour, -1)

    def frame(self, envs, obs_list, actions, hud: list, banners) -> np.ndarray:
        img = self.bg.copy()
        for i, (env, obs) in enumerate(zip(envs, obs_list)):
            if obs is not None:
                self._draw_car(img, env, obs, _CAR_COLOURS[i % len(_CAR_COLOURS)])
        for text, colour, count in hud:
            cv2.putText(img, text, (10, 22 + 20 * count),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)
        for b in banners:
            cv2.putText(img, b["text"], tuple(b["pos"]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, b["colour"], 2,
                        cv2.LINE_AA)
        if len(envs) == 1 and actions[0] is not None:
            self._draw_pedals(img, actions[0], img.shape[0] - 64)
        return img


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Watch the agent's attempts.")
    ap.add_argument("--artifacts", type=str, default="artifacts")
    ap.add_argument("--policy", type=str,
                    help="policy zip (default: rl_policy_tuned.zip if it "
                         "exists, else rl_policy.zip)")
    ap.add_argument("--episodes", type=int, default=5,
                    help="runs per car (default 5)")
    ap.add_argument("--laps", type=int, default=1,
                    help="laps the car must complete per run (default 1)")
    ap.add_argument("--cars", type=int, default=1,
                    help="cars running at the same time (default 1)")
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
    ap.add_argument("--show-beams", action="store_true",
                    help="draw lidar beams even with multiple cars")
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
        print(f"Watching {policy_path.name} | physics: {note} | "
              f"{args.cars} car(s), {args.laps} lap(s) per run")
    else:
        print(f"Watching RANDOM actions | physics: {note}")

    # with several cars from the same fixed start they'd all drive the exact
    # same line; spread them around the track instead
    random_spawn = args.random_spawn or args.cars > 1
    envs = [TrackEnv(track, params=params, random_spawn=random_spawn,
                     laps=args.laps) for _ in range(args.cars)]
    show_beams = args.show_beams or args.cars == 1
    viewer = Viewer(track, chassis_l, chassis_w, show_beams)

    writer = None
    fps = int(round(1.0 / envs[0].dt))
    if args.video:
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

    obs_list = [env.reset(seed=i)[0] for i, env in enumerate(envs)]
    actions = [None] * args.cars
    runs = [0] * args.cars            # completed runs per car
    active = [True] * args.cars
    banners: list[dict] = []          # floating LAP/CRASH banners with ttl
    best_lap = None
    total_runs = 0
    frame_budget_s = envs[0].dt / max(args.speed, 0.01)

    while any(active):
        t0 = time.time()
        for i, env in enumerate(envs):
            if not active[i]:
                continue
            if model is None:
                actions[i] = env.action_space.sample()
            else:
                actions[i], _ = model.predict(
                    obs_list[i], deterministic=not args.stochastic)
            obs_list[i], _, terminated, truncated, info = env.step(actions[i])

            if terminated or truncated:
                rs = info["run_summary"]
                runs[i] += 1
                total_runs += 1
                pos = viewer._to_px(np.array([env.state.x, env.state.y]))
                pos = (int(np.clip(pos[0], 10, viewer.bg.shape[1] - 200)),
                       int(np.clip(pos[1], 30, viewer.bg.shape[0] - 10)))
                if rs["lap_time_s"] is not None:
                    if best_lap is None or rs["lap_time_s"] < best_lap:
                        best_lap = rs["lap_time_s"]
                    banners.append({"text": f"{rs['laps_done']} lap(s) "
                                            f"{rs['lap_time_s']:.2f} s",
                                    "colour": _GOOD, "pos": pos, "ttl": fps})
                else:
                    cause = "CRASH" if rs["crashed"] else "TIME LIMIT"
                    banners.append({"text": cause, "colour": _BAD,
                                    "pos": pos, "ttl": fps})
                print(f"car {i + 1} run {runs[i]:3d} | "
                      f"reward +{rs['reward']:8.2f} | "
                      f"punishment -{rs['punishment']:7.2f} | "
                      f"progress {rs['progress_pct']:5.1f}% | "
                      f"laps {rs['laps_done']}/{args.laps}"
                      + (f" | {rs['lap_time_s']:.2f} s"
                         if rs["lap_time_s"] else ""))
                if runs[i] >= args.episodes:
                    active[i] = False
                    obs_list[i] = None
                else:
                    obs_list[i] = env.reset(seed=total_runs * args.cars + i)[0]

        # HUD: one line per car (first 8), then a global line
        hud = []
        for i, env in enumerate(envs[:8]):
            if not active[i]:
                continue
            colour = _CAR_COLOURS[i % len(_CAR_COLOURS)]
            lap_now = env._total_progress / track.length_m
            hud.append((f"car {i + 1}  run {runs[i] + 1}/{args.episodes}  "
                        f"lap {max(lap_now, 0):.2f}/{args.laps}  "
                        f"v {env.state.v:4.1f} m/s  "
                        f"R +{env._reward_sum:7.1f}  "
                        f"P -{env._punish_sum:6.1f}", colour, len(hud)))
        hud.append((f"best {f'{best_lap:.2f} s' if best_lap else '--'}   "
                    f"runs finished {total_runs}", _HUD, len(hud)))

        img = viewer.frame(envs, obs_list, actions, hud, banners)
        for b in banners:
            b["ttl"] -= 1
        banners = [b for b in banners if b["ttl"] > 0]

        if writer is not None:
            writer.write(img)
        else:
            cv2.imshow("attempt viewer", img)
            wait_ms = max(1, int(1000 * (frame_budget_s - (time.time() - t0))))
            if cv2.waitKey(wait_ms) & 0xFF in (27, ord("q")):
                print("quit.")
                break

    if writer is not None:
        writer.release()
        print(f"wrote {args.video}")
    else:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
