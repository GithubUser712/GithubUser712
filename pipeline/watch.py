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
                 show_beams: bool, display_px: int = 1100):
        self.track = track
        occ = track.grid.occupancy
        self.bg = np.where(occ[..., None] == 1, _WALL, _FREE).astype(np.uint8)
        self.chassis_l = chassis_l_m
        self.chassis_w = chassis_w_m
        self.show_beams = show_beams
        # big circuit maps (Spa is 2122 px tall) are downscaled for display;
        # even dimensions keep video codecs happy
        h, w = self.bg.shape[:2]
        self.scale = min(1.0, display_px / max(h, w))
        self.out_size = (int(w * self.scale) // 2 * 2,
                         int(h * self.scale) // 2 * 2)

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
        # world at full map resolution...
        img = self.bg.copy()
        for i, (env, obs) in enumerate(zip(envs, obs_list)):
            if obs is not None:
                self._draw_car(img, env, obs, _CAR_COLOURS[i % len(_CAR_COLOURS)])
        if img.shape[1] != self.out_size[0]:
            img = cv2.resize(img, self.out_size, interpolation=cv2.INTER_AREA)

        # ...text and pedals at display resolution so they stay readable
        for text, colour, count in hud:
            cv2.putText(img, text, (10, 22 + 20 * count),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)
        for b in banners:
            pos = (int(b["pos"][0] * self.scale), int(b["pos"][1] * self.scale))
            cv2.putText(img, b["text"], pos, cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, b["colour"], 2, cv2.LINE_AA)
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
    ap.add_argument("--display-width", type=int, default=1100,
                    help="max window/video dimension in px (default 1100)")
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

    # with several cars from the same fixed start they'd all drive the exact
    # same line; spread them around the track instead
    random_spawn = args.random_spawn or args.cars > 1
    envs = [TrackEnv(track, params=params, random_spawn=random_spawn,
                     laps=args.laps) for _ in range(args.cars)]
    obs_shape = envs[0].observation_space.shape

    def _try_load(path: Path, retries: int = 3):
        """Load a policy if it exists, is readable, and matches the env."""
        from stable_baselines3 import PPO
        for attempt in range(retries):
            if not path.is_file():
                return None
            try:
                candidate = PPO.load(path, device="cpu")
            except Exception:       # training may be mid-write
                if attempt + 1 < retries:
                    time.sleep(2.0)
                continue
            if candidate.observation_space.shape != obs_shape:
                print(f"NOTE: {path.name} expects observations "
                      f"{candidate.observation_space.shape}, the simulator "
                      f"produces {obs_shape} -- it predates an upgrade, "
                      "skipping it")
                return None
            return candidate
        return None

    model = None
    policy_path = None
    policy_mtime = 0.0
    if not args.random:
        candidates = ([Path(args.policy)] if args.policy else
                      [out_dir / "rl_policy_tuned.zip",
                       out_dir / "rl_policy.zip"])
        waiting_msg_shown = False
        while model is None:
            for cand in candidates:
                model = _try_load(cand)
                if model is not None:
                    policy_path = cand
                    policy_mtime = cand.stat().st_mtime
                    break
            if model is None:
                if not waiting_msg_shown:
                    print("No usable policy yet -- waiting for training's "
                          "next autosave (every 50k steps). If you are NOT "
                          "training right now, retrain with step 2 (stale "
                          "pre-upgrade policies cannot be watched). "
                          "Ctrl-C to give up.")
                    waiting_msg_shown = True
                time.sleep(5.0)
        print(f"Watching {policy_path.name} | physics: {note} | "
              f"{args.cars} car(s), {args.laps} lap(s) per run")
    else:
        print(f"Watching RANDOM actions | physics: {note}")
    show_beams = args.show_beams or args.cars == 1
    viewer = Viewer(track, chassis_l, chassis_w, show_beams,
                    display_px=args.display_width)

    writer = None
    fps = int(round(1.0 / envs[0].dt))
    if args.video:
        writer = cv2.VideoWriter(args.video,
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                 viewer.out_size)
        if not writer.isOpened():
            print("ERROR: could not open video writer (codec missing?).")
            return 2
    else:
        try:
            cv2.namedWindow("attempt viewer", cv2.WINDOW_NORMAL)
        except cv2.error:
            print("ERROR: no display available -- use --video out.mp4 instead.")
            return 2
        print("live window open. Tips: early-training policies may barely "
              "move (the picture only crawls) -- try --stochastic "
              "--random-spawn for livelier runs; if the window NEVER "
              "repaints on a Wayland desktop, retry with GDK_BACKEND=x11")

    obs_list = [env.reset(seed=i)[0] for i, env in enumerate(envs)]
    actions = [None] * args.cars
    runs = [0] * args.cars            # completed runs per car
    active = [True] * args.cars
    banners: list[dict] = []          # floating LAP/CRASH banners with ttl
    best_lap = None
    total_runs = 0
    frame_budget_s = envs[0].dt / max(args.speed, 0.01)
    frame_idx = 0
    reload_every = int(10.0 / envs[0].dt)      # check for a newer autosave ~10 s

    while any(active):
        t0 = time.time()
        frame_idx += 1

        # hot-reload training's newest autosave and restart the runs with it,
        # so a stalled early policy doesn't freeze the picture for minutes
        if model is not None and frame_idx % reload_every == 0:
            try:
                mtime = policy_path.stat().st_mtime
            except OSError:
                mtime = policy_mtime
            if mtime > policy_mtime:
                fresh = _try_load(policy_path, retries=1)
                if fresh is not None:
                    model = fresh
                    policy_mtime = mtime
                    print("        ...reloaded latest training autosave, "
                          "restarting runs")
                    for i, env in enumerate(envs):
                        if active[i]:
                            obs_list[i] = env.reset(
                                seed=total_runs * args.cars + i + frame_idx)[0]
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
            slide = "  SLIDE!" if getattr(env, "last_slid", False) else ""
            hud.append((f"car {i + 1}  run {runs[i] + 1}/{args.episodes}  "
                        f"lap {max(lap_now, 0):.2f}/{args.laps}  "
                        f"v {env.state.v:4.1f} m/s  "
                        f"R +{env._reward_sum:7.1f}  "
                        f"P -{env._punish_sum:6.1f}{slide}", colour, len(hud)))
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
