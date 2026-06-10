"""Kinematic bicycle model -- the physics used by the RL simulator.

Kept in its own module because step 3 (vehicle parameters) will construct a
`CarParams` from your real car's measurements and re-use exactly this code.
The defaults below are a generic F1TENTH-ish car for the first RL pass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class CarParams:
    wheelbase_m: float = 0.32        # axle-to-axle length
    max_speed_mps: float = 5.0
    min_speed_mps: float = 0.0
    max_steer_rad: float = 0.40      # ~23 degrees at the front wheels
    steer_rate_rps: float = 4.0      # how fast the steering can slew
    max_accel_mps2: float = 4.0
    max_brake_mps2: float = 6.0
    safety_radius_m: float = 0.15    # wall distance below which we call it a crash


@dataclass
class CarState:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0                 # radians, 0 = +x, CCW positive
    v: float = 0.0                   # m/s, forward only
    steer: float = 0.0               # current front wheel angle, radians


def kinematic_step(s: CarState, steer_target: float, accel: float,
                   p: CarParams, dt: float) -> None:
    """Advance the car state in place by `dt` seconds.

    steer_target : desired front wheel angle [rad]; the actual angle slews
                   toward it at steer_rate_rps (servos are not instant)
    accel        : longitudinal acceleration [m/s^2] (negative = braking)
    """
    max_delta = p.steer_rate_rps * dt
    s.steer += min(max(steer_target - s.steer, -max_delta), max_delta)
    s.steer = min(max(s.steer, -p.max_steer_rad), p.max_steer_rad)

    s.v = min(max(s.v + accel * dt, p.min_speed_mps), p.max_speed_mps)

    s.x += s.v * math.cos(s.yaw) * dt
    s.y += s.v * math.sin(s.yaw) * dt
    s.yaw += s.v / p.wheelbase_m * math.tan(s.steer) * dt
    s.yaw = (s.yaw + math.pi) % (2.0 * math.pi) - math.pi
