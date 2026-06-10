"""Step 2a: a Gymnasium environment that drives the bicycle model around the
occupancy grid produced by step 1.

Observation : n_beams simulated lidar ranges (normalised 0..1)
              + current speed (0..1) + current steering angle (-1..1)
Action      : [steering command, throttle/brake command], both in [-1, 1]
Reward      : + progress along the centerline (metres gained this step)
              + lap completion bonus
              - time penalty, steering thrash penalty, collision penalty

Positive and negative terms are accumulated separately so each finished run
can report "rewards" and "punishments" independently, as required.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces

from .bicycle import CarParams, CarState, kinematic_step
from .map_ingestion import TrackMap

# ----------------------------------------------------------- reward weights
PROGRESS_REWARD_PER_M = 1.0
LAP_BONUS = 100.0
COLLISION_PENALTY = 50.0
TIME_PENALTY_PER_STEP = 0.01
STEER_THRASH_PENALTY = 0.02       # * |change in steering command|
SLIDE_PENALTY_PER_STEP = 0.05     # exceeded lateral grip (understeering)


# ------------------------------------------------------------- track loading

@dataclass
class TrackData:
    grid: TrackMap                 # occupancy + distance field + px<->world
    centerline: np.ndarray         # (N, 2) ordered waypoints [x_m, y_m]
    clearance: np.ndarray          # (N,) metres to nearest wall
    length_m: float
    spacing_m: float
    start_pose: tuple[float, float, float]   # x, y, yaw


def load_track_data(artifacts_dir: str | Path) -> TrackData:
    """Load everything step 1 wrote.  Step 2 never re-reads the raw drawing."""
    d = Path(artifacts_dir)
    with open(d / "map.yaml") as f:
        map_cfg = yaml.safe_load(f)
    img = cv2.imread(str(d / map_cfg["image"]), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"missing {d / map_cfg['image']} -- run step 1 first")
    occupancy = np.where(img < 127, 1, 0).astype(np.uint8)
    grid = TrackMap(occupancy=occupancy, resolution=float(map_cfg["resolution"]),
                    source_path=d / map_cfg["image"])

    cl = np.loadtxt(d / "centerline.csv", delimiter=",", skiprows=1)
    with open(d / "track_meta.yaml") as f:
        meta = yaml.safe_load(f)
    sp = meta["start_pose"]
    return TrackData(
        grid=grid,
        centerline=cl[:, :2],
        clearance=cl[:, 2],
        length_m=float(meta["track_length_m"]),
        spacing_m=float(meta["waypoint_spacing_m"]),
        start_pose=(float(sp["x_m"]), float(sp["y_m"]), float(sp["yaw_rad"])),
    )


# -------------------------------------------------------------- environment

class TrackEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, track: TrackData, params: CarParams | None = None,
                 n_beams: int = 27, fov_deg: float = 270.0,
                 max_range_m: float = 12.0, dt: float = 0.05,
                 physics_substeps: int = 2, max_steps: int = 3000,
                 random_spawn: bool = True, laps: int = 1):
        super().__init__()
        self.track = track
        self.p = params or CarParams()
        self.dt = dt
        self.substeps = physics_substeps
        self.max_steps = max_steps
        self.random_spawn = random_spawn
        self.laps = laps               # episode ends after this many laps
        self.max_range = max_range_m
        self.beam_angles = np.deg2rad(
            np.linspace(-fov_deg / 2.0, fov_deg / 2.0, n_beams))

        low = np.concatenate([np.zeros(n_beams), [0.0], [-1.0]]).astype(np.float32)
        high = np.ones(n_beams + 2, dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        self.state = CarState()
        self.trajectory: list[tuple[float, float]] = []

    # ------------------------------------------------------------- helpers

    def _dist_at(self, x: float, y: float) -> float:
        """Distance to the nearest wall at a world position (0 if off-map)."""
        rc = self.track.grid.pixel_from_world((x, y))
        rows, cols = self.track.grid.shape
        r, c = int(round(rc[0])), int(round(rc[1]))
        if not (0 <= r < rows and 0 <= c < cols):
            return 0.0
        return float(self.track.grid.distance_m[r, c])

    def _raycast(self) -> np.ndarray:
        """Sphere-march all beams through the distance field at once.

        Each beam repeatedly jumps forward by the distance to the nearest
        wall -- it cannot skip through one, and it converges in a few dozen
        iterations instead of stepping pixel by pixel.
        """
        grid = self.track.grid
        rows, cols = grid.shape
        angles = self.state.yaw + self.beam_angles
        dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        pos = np.tile([self.state.x, self.state.y], (len(angles), 1))
        dist = np.zeros(len(angles))
        active = np.ones(len(angles), dtype=bool)

        for _ in range(64):
            if not active.any():
                break
            rc = grid.pixel_from_world(pos[active])
            r = np.clip(np.rint(rc[:, 0]).astype(int), 0, rows - 1)
            c = np.clip(np.rint(rc[:, 1]).astype(int), 0, cols - 1)
            d = grid.distance_m[r, c]
            hit = d < grid.resolution            # essentially touching a wall
            step = np.maximum(d, grid.resolution)
            pos[active] += dirs[active] * step[:, None]
            dist[active] += step
            done = hit | (dist[active] > self.max_range)
            idx = np.flatnonzero(active)
            active[idx[done]] = False

        return np.clip(dist, 0.0, self.max_range)

    def _nearest_idx(self, xy: np.ndarray, full_search: bool = False) -> int:
        """Index of the closest centerline waypoint (windowed for speed)."""
        cl = self.track.centerline
        if full_search:
            return int(np.argmin(np.linalg.norm(cl - xy, axis=1)))
        window = (self._last_idx + np.arange(-40, 41)) % len(cl)
        d = np.linalg.norm(cl[window] - xy, axis=1)
        return int(window[np.argmin(d)])

    def _progress_delta(self, idx_new: int) -> float:
        """Signed metres advanced along the loop since the last step."""
        n = len(self.track.centerline)
        di = (idx_new - self._last_idx + n // 2) % n - n // 2
        return di * self.track.spacing_m

    def _observation(self) -> np.ndarray:
        beams = self._raycast() / self.max_range
        return np.concatenate([
            beams,
            [self.state.v / self.p.max_speed_mps],
            [self.state.steer / self.p.max_steer_rad],
        ]).astype(np.float32)

    # ----------------------------------------------------------- gym API

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        cl = self.track.centerline
        if self.random_spawn:
            i = int(self.np_random.integers(len(cl)))
            tangent = cl[(i + 1) % len(cl)] - cl[i]
            yaw = float(np.arctan2(tangent[1], tangent[0]))
            yaw += float(self.np_random.uniform(-0.2, 0.2))
            x, y = cl[i]
            v0 = float(self.np_random.uniform(0.0, 1.0))
        else:
            x, y, yaw = self.track.start_pose
            v0 = 0.0
        self.state = CarState(x=x, y=y, yaw=yaw, v=v0, steer=0.0)

        self._last_idx = self._nearest_idx(np.array([x, y]), full_search=True)
        self._total_progress = 0.0
        self._steps = 0
        self._reward_sum = 0.0
        self._punish_sum = 0.0
        self._last_steer_cmd = 0.0
        self.trajectory = [(x, y)]
        self.progress_log = [0.0]      # cumulative progress per trajectory point
        return self._observation(), {}

    def step(self, action):
        steer_cmd = float(np.clip(action[0], -1.0, 1.0))
        throttle = float(np.clip(action[1], -1.0, 1.0))
        steer_target = steer_cmd * self.p.max_steer_rad
        accel = throttle * (self.p.max_accel_mps2 if throttle >= 0.0
                            else self.p.max_brake_mps2)

        collided = False
        slid = False
        sub_dt = self.dt / self.substeps
        for _ in range(self.substeps):
            slid |= kinematic_step(self.state, steer_target, accel, self.p, sub_dt)
            if self._dist_at(self.state.x, self.state.y) < self.p.safety_radius_m:
                collided = True
                break

        pos = np.array([self.state.x, self.state.y])
        idx = self._nearest_idx(pos)
        ds = self._progress_delta(idx)
        self._last_idx = idx
        self._total_progress += ds
        self.trajectory.append((self.state.x, self.state.y))
        self.progress_log.append(self._total_progress)
        self._steps += 1

        # --- separate reward / punishment bookkeeping --------------------
        reward_pos = max(ds, 0.0) * PROGRESS_REWARD_PER_M
        punish = max(-ds, 0.0) * PROGRESS_REWARD_PER_M
        punish += TIME_PENALTY_PER_STEP
        punish += STEER_THRASH_PENALTY * abs(steer_cmd - self._last_steer_cmd)
        if slid:
            punish += SLIDE_PENALTY_PER_STEP
        self._last_steer_cmd = steer_cmd

        lap_time = None
        terminated = False
        goal_m = self.laps * self.track.length_m
        if collided:
            punish += COLLISION_PENALTY
            terminated = True
        elif self._total_progress >= goal_m:
            reward_pos += LAP_BONUS
            lap_time = self._steps * self.dt
            terminated = True
        truncated = self._steps >= self.max_steps

        self._reward_sum += reward_pos
        self._punish_sum += punish

        info = {}
        if terminated or truncated:
            info["run_summary"] = {
                "reward": self._reward_sum,
                "punishment": self._punish_sum,
                "progress_pct": 100.0 * max(self._total_progress, 0.0) / goal_m,
                "lap_time_s": lap_time,
                "crashed": collided,
            }

        return (self._observation(), reward_pos - punish,
                terminated, truncated, info)
