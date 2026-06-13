"""Pure pursuit path tracker + proportional speed controller."""

from __future__ import annotations

import math

import numpy as np

from raceline.physics import CarParams


class PurePursuit:
    """Geometric path tracker with speed P-control."""

    def __init__(
        self,
        xy: np.ndarray,
        speeds: np.ndarray,
        params: CarParams,
        lookahead_gain: float = 0.30,
        lookahead_min: float = 0.4,
        lookahead_max: float = 1.5,
        speed_kp: float = 3.0,
    ):
        self.xy = xy
        self.speeds = speeds
        self.p = params
        self.k_ld = lookahead_gain
        self.ld_min = lookahead_min
        self.ld_max = lookahead_max
        self.speed_kp = speed_kp
        seg = np.linalg.norm(np.diff(np.vstack([xy, xy[:1]]), axis=0), axis=1)
        self.ds = float(seg.mean())
        self._last_idx = 0

    def reset(self, position: np.ndarray) -> None:
        self._last_idx = int(np.argmin(np.linalg.norm(self.xy - position, axis=1)))

    def nearest(self, position: np.ndarray) -> int:
        n = len(self.xy)
        window = (self._last_idx + np.arange(-30, 90)) % n
        d = np.linalg.norm(self.xy[window] - position, axis=1)
        self._last_idx = int(window[np.argmin(d)])
        return self._last_idx

    def control(self, state) -> tuple[np.ndarray, float, float]:
        pos = np.array([state.x, state.y])
        i = self.nearest(pos)
        err = float(np.linalg.norm(self.xy[i] - pos))

        ld = float(np.clip(self.k_ld * state.vx, self.ld_min, self.ld_max))
        j = (i + max(int(ld / self.ds), 1)) % len(self.xy)
        target = self.xy[j]

        dx, dy = target[0] - state.x, target[1] - state.y
        cos_y, sin_y = math.cos(state.yaw), math.sin(state.yaw)
        tx = cos_y * dx + sin_y * dy
        ty = -sin_y * dx + cos_y * dy
        alpha = math.atan2(ty, max(tx, 1e-3))
        steer = math.atan2(2.0 * self.p.wheelbase_m * math.sin(alpha), ld)

        v_target = float(self.speeds[j])
        accel = self.speed_kp * (v_target - state.vx)
        accel = float(np.clip(accel, -self.p.max_brake_mps2, self.p.max_accel_mps2))
        throttle = (
            accel / self.p.max_accel_mps2
            if accel >= 0.0
            else accel / self.p.max_brake_mps2 * -1.0
        )

        action = np.array(
            [
                np.clip(steer / self.p.max_steer_rad, -1.0, 1.0),
                np.clip(throttle if accel >= 0.0 else -abs(throttle), -1.0, 1.0),
            ],
            dtype=np.float32,
        )
        return action, err, v_target
