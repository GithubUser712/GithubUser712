"""Step 2 entry point: train an RL agent to find the racing line.

    python -m pipeline.step2                       # uses artifacts/, 1M steps
    python -m pipeline.step2 --timesteps 2000000   # train longer
    python -m pipeline.step2 --resume              # continue from saved policy
    python -m pipeline.step2 --eval-only           # just extract the raceline

Every finished run (episode) prints one ping line with its rewards and
punishments.  After training, the best deterministic lap is recorded,
smoothed, and written out as the racing line.
"""

from __future__ import annotations

import argparse
import sys
import time
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from scipy import interpolate

from .track_env import TrackData, TrackEnv, load_track_data


# ------------------------------------------------------------ terminal pings

class PingCallback(BaseCallback):
    """Print one line per finished run: rewards, punishments, progress."""

    def __init__(self):
        super().__init__()
        self.runs = 0
        self.best_lap: float | None = None
        self._t0 = time.time()

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            rs = info.get("run_summary")
            if rs is None:
                continue
            self.runs += 1
            lap = rs["lap_time_s"]
            if lap is not None and (self.best_lap is None or lap < self.best_lap):
                self.best_lap = lap
            lap_str = f"{lap:6.2f} s" if lap is not None else "  --   "
            best_str = f"{self.best_lap:6.2f} s" if self.best_lap else "  --   "
            print(f"run {self.runs:5d} | reward {rs['reward']:+9.2f} | "
                  f"punishment -{rs['punishment']:8.2f} | "
                  f"progress {rs['progress_pct']:5.1f}% | "
                  f"lap {lap_str} | best {best_str} | "
                  f"steps {self.num_timesteps:>9,} | "
                  f"{time.time() - self._t0:6.0f} s elapsed")
        return True


# -------------------------------------------------------- raceline extraction

def best_deterministic_lap(model: PPO, track: TrackData, params=None,
                           attempts: int = 5) -> tuple[np.ndarray | None, float]:
    """Roll out the trained policy without exploration noise from the start
    pose; return the trajectory of the fastest completed lap (or None)."""
    env = TrackEnv(track, params=params, random_spawn=False)
    best_traj, best_time = None, float("inf")
    for i in range(attempts):
        obs, _ = env.reset(seed=i)
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                rs = info["run_summary"]
                lap = rs["lap_time_s"]
                print(f"  eval attempt {i + 1}: progress {rs['progress_pct']:.1f}%"
                      + (f", lap {lap:.2f} s" if lap else ", no lap"))
                if lap is not None and lap < best_time:
                    best_time = lap
                    best_traj = np.array(env.trajectory)
                break
    return best_traj, best_time


def smooth_raceline(traj: np.ndarray, spacing_m: float) -> np.ndarray:
    """Periodic-spline smooth the raw lap trajectory, resample it at uniform
    arc length and attach the curvature of every waypoint.

    Returns (N, 3): x_m, y_m, curvature_1pm  (signed; + is a left turn).
    """
    # tolerate ~5 cm deviation from the raw driven path
    s = len(traj) * 0.05**2
    tck, _ = interpolate.splprep([traj[:, 0], traj[:, 1]], s=s, per=True)
    u = np.linspace(0.0, 1.0, 4000, endpoint=False)
    dense = np.stack(interpolate.splev(u, tck), axis=1)

    closed = np.vstack([dense, dense[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    length = float(arc[-1])
    n = max(int(round(length / spacing_m)), 50)
    s_t = np.linspace(0.0, length, n, endpoint=False)
    pts = np.stack([np.interp(s_t, arc, closed[:, 0]),
                    np.interp(s_t, arc, closed[:, 1])], axis=1)

    # curvature from central differences on the closed, uniformly spaced loop
    ds = length / n
    x, y = pts[:, 0], pts[:, 1]
    dx = (np.roll(x, -1) - np.roll(x, 1)) / (2 * ds)
    dy = (np.roll(y, -1) - np.roll(y, 1)) / (2 * ds)
    ddx = (np.roll(x, -1) - 2 * x + np.roll(x, 1)) / ds**2
    ddy = (np.roll(y, -1) - 2 * y + np.roll(y, 1)) / ds**2
    curvature = (dx * ddy - dy * ddx) / np.maximum((dx**2 + dy**2) ** 1.5, 1e-9)
    return np.column_stack([pts, curvature])


def save_raceline(track: TrackData, raceline: np.ndarray, lap_time: float,
                  out_dir: Path, stem: str = "raceline",
                  note: str = "generic car parameters") -> list[Path]:
    csv_path = out_dir / f"{stem}.csv"
    np.savetxt(csv_path, raceline, delimiter=",", fmt="%.4f",
               header="x_m,y_m,curvature_1pm", comments="")

    fig, ax = plt.subplots(figsize=(10, 10 * track.grid.shape[0] / track.grid.shape[1]))
    ax.imshow(track.grid.occupancy, cmap="gray_r", interpolation="nearest")
    cpx = track.grid.pixel_from_world(track.centerline)
    ax.plot(cpx[:, 1], cpx[:, 0], color="0.7", lw=1, label="centerline")
    rpx = track.grid.pixel_from_world(raceline[:, :2])
    sc = ax.scatter(rpx[:, 1], rpx[:, 0], c=np.abs(raceline[:, 2]),
                    cmap="plasma", s=4, label="RL racing line")
    fig.colorbar(sc, ax=ax, fraction=0.04, label="|curvature| (1/m)")
    ax.plot(rpx[0, 1], rpx[0, 0], "r*", markersize=14)
    ax.set_title(f"RL racing line (lap {lap_time:.2f} s with {note})")
    ax.legend(loc="upper right")
    fig.tight_layout()
    png_path = out_dir / f"{stem}_debug.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    return [csv_path, png_path]


# --------------------------------------------------------------------- main

def _make_env(track: TrackData) -> TrackEnv:
    return TrackEnv(track, random_spawn=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Step 2: RL training for the racing line.")
    ap.add_argument("--artifacts", type=str, default="artifacts",
                    help="directory written by step 1 (default: artifacts/)")
    ap.add_argument("--timesteps", type=int, default=1_000_000)
    ap.add_argument("--n-envs", type=int, default=4,
                    help="parallel simulation environments")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                    help="continue training from the saved policy")
    ap.add_argument("--eval-only", action="store_true",
                    help="skip training, just extract the raceline")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                    help="where the neural net runs (default: auto = cuda if "
                         "available); the simulator itself always runs on CPU")
    args = ap.parse_args(argv)

    import torch
    cuda = torch.cuda.is_available()
    print(f"torch {torch.__version__} | CUDA available: {cuda}"
          + (f" ({torch.cuda.get_device_name(0)})" if cuda else ""))

    out_dir = Path(args.artifacts)
    policy_path = out_dir / "rl_policy.zip"
    track = load_track_data(out_dir)
    print(f"Track loaded: {track.length_m:.1f} m loop, "
          f"{len(track.centerline)} waypoints, "
          f"width {2 * track.clearance.min():.2f}-{2 * track.clearance.max():.2f} m")

    factory = partial(_make_env, track)
    if not args.eval_only:
        vec_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
        venv = vec_cls([factory for _ in range(args.n_envs)])

        if args.resume and policy_path.is_file():
            print(f"Resuming training from {policy_path}\n")
            model = PPO.load(policy_path, env=venv, device=args.device)
        else:
            model = PPO(
                "MlpPolicy", venv, seed=args.seed, verbose=0,
                learning_rate=3e-4, n_steps=1024, batch_size=256,
                gamma=0.995, gae_lambda=0.95, ent_coef=0.005,
                policy_kwargs=dict(net_arch=[256, 256]),
                device=args.device,
            )
        print(f"Training PPO for {args.timesteps:,} timesteps on "
              f"{args.n_envs} parallel sims, net on '{model.device}' "
              "-- one ping per finished run:\n")
        model.learn(total_timesteps=args.timesteps, callback=PingCallback())
        model.save(policy_path)
        venv.close()
        print(f"\nPolicy saved to {policy_path}")
    else:
        if not policy_path.is_file():
            print(f"ERROR: {policy_path} not found -- train first.")
            return 2
        model = PPO.load(policy_path, device=args.device)

    print("\nExtracting racing line (deterministic rollouts from the start line):")
    traj, lap_time = best_deterministic_lap(model, track)
    if traj is None:
        print("\nNo complete lap yet -- the policy needs more training.\n"
              f"Run:  python -m pipeline.step2 --resume --timesteps {args.timesteps}")
        return 1

    raceline = smooth_raceline(traj, track.spacing_m)
    written = save_raceline(track, raceline, lap_time, out_dir)
    print(f"\nBest lap: {lap_time:.2f} s (generic car parameters)")
    print("Wrote:")
    for p in written:
        print(f"  {p}")
    print("\nStep 2 complete. Step 3 will ask for your real car's parameters "
          "and recompute this line with proper physics.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
