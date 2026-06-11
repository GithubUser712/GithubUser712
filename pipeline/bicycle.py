"""Dynamic single-track ("bicycle") vehicle model.

This is the same class of model the F1TENTH community uses for realistic
simulation.  Compared to the old kinematic model it adds:

  * tire slip: lateral forces come from slip angles through a simplified
    Pacejka magic-formula curve, so the car can understeer, oversteer and
    spin -- it is no longer geometrically glued to the steering angle
  * longitudinal load transfer: braking loads the front tires (more front
    grip, looser rear), accelerating does the opposite
  * a traction circle per axle: grip spent cornering is unavailable for
    drive/brake force, wheelspin and lock-up are capped at the limit
  * aerodynamic drag and rolling resistance
  * yaw inertia: the car takes time to rotate, estimated from mass and
    chassis dimensions if not given

Below ~1 m/s the slip-angle equations are singular, so the model blends
into the kinematic solution -- standard practice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

G = 9.81
RHO_AIR = 1.2          # kg/m^3
_BLEND_LO = 0.5        # m/s: fully kinematic below this
_BLEND_HI = 1.5        # m/s: fully dynamic above this


@dataclass
class CarParams:
    # geometry & actuator limits (step 3 fills these from your measurements)
    wheelbase_m: float = 0.32
    max_speed_mps: float = 8.0       # ERPM-limiter equivalent
    min_speed_mps: float = 0.0
    max_steer_rad: float = 0.40
    steer_rate_rps: float = 4.0
    max_accel_mps2: float = 4.0      # drive-force cap / mass (motor strength)
    max_brake_mps2: float = 6.0
    safety_radius_m: float = 0.15

    # tires & body
    mu: float = 0.7                  # tire grip coefficient
    mass_kg: float = 3.5
    chassis_l_m: float = 0.50        # used for the yaw inertia estimate
    chassis_w_m: float = 0.30
    cog_height_m: float = 0.04       # centre-of-gravity height (load transfer)
    yaw_inertia: float | None = None # kg m^2; None = box estimate from chassis
    pacejka_b: float = 8.0           # tire stiffness factor
    pacejka_c: float = 1.5           # tire shape factor
    cda_m2: float = 0.04             # drag area (0.5 * rho * CdA * v^2)
    crr: float = 0.02                # rolling resistance coefficient

    @property
    def lf(self) -> float:          # CoG assumed mid-wheelbase
        return self.wheelbase_m / 2.0

    @property
    def lr(self) -> float:
        return self.wheelbase_m / 2.0

    @property
    def izz(self) -> float:
        if self.yaw_inertia is not None:
            return self.yaw_inertia
        return self.mass_kg * (self.chassis_l_m**2 + self.chassis_w_m**2) / 12.0

    @property
    def alpha_peak(self) -> float:
        """Slip angle where the tire curve peaks; beyond this = sliding."""
        return math.tan(math.pi / (2.0 * self.pacejka_c)) / self.pacejka_b


@dataclass
class CarState:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0       # radians, 0 = +x, CCW positive
    vx: float = 0.0        # body-frame forward velocity
    vy: float = 0.0        # body-frame lateral velocity (+ = left)
    r: float = 0.0         # yaw rate, rad/s
    steer: float = 0.0     # current front wheel angle
    ax: float = 0.0        # last longitudinal accel (drives load transfer)

    @property
    def v(self) -> float:
        """Forward speed -- what a speedometer (or the VESC) would report."""
        return self.vx


def _pacejka(alpha: float, mu: float, fz: float, b: float, c: float) -> float:
    """Lateral tire force from slip angle (opposes the slip)."""
    return -mu * fz * math.sin(c * math.atan(b * alpha))


def kinematic_step(s: CarState, steer_target: float, accel_cmd: float,
                   p: CarParams, dt: float) -> bool:
    """Advance the car state in place by `dt` seconds.

    steer_target : desired front wheel angle [rad] (servo slews toward it)
    accel_cmd    : requested longitudinal acceleration [m/s^2]
                   (positive = drive force at the rear axle, negative = brakes)

    Returns True if any tire was saturated this step (slipping past the peak
    of its grip curve, wheelspin, or brake lock-up).
    """
    # steering actuator
    max_delta = p.steer_rate_rps * dt
    s.steer += min(max(steer_target - s.steer, -max_delta), max_delta)
    s.steer = min(max(s.steer, -p.max_steer_rad), p.max_steer_rad)
    delta = s.steer

    m, L = p.mass_kg, p.wheelbase_m
    vx = max(s.vx, 0.0)

    # vertical loads with longitudinal weight transfer (one step lagged)
    fz_total = m * G
    transfer = m * s.ax * p.cog_height_m / L
    fzf = min(max(fz_total * p.lr / L - transfer, 0.05 * fz_total), fz_total)
    fzr = fz_total - fzf

    # resistive forces
    drag = 0.5 * RHO_AIR * p.cda_m2 * vx * vx
    roll = p.crr * fz_total if vx > 0.05 else 0.0

    # requested drive / brake forces, then traction-circle capping
    sliding = False
    if accel_cmd >= 0.0:
        fx_r_req, fx_f_req = m * accel_cmd, 0.0
    else:   # brake split proportional to vertical load
        fx_f_req = m * accel_cmd * (fzf / fz_total)
        fx_r_req = m * accel_cmd * (fzr / fz_total)

    # kinematic reference (exact at low speed, blend target)
    r_kin = vx / L * math.tan(delta)
    vy_kin = r_kin * p.lr

    if vx < _BLEND_LO:
        # too slow for slip angles: pure kinematic motion
        fx = fx_r_req + fx_f_req - drag - roll
        ax = fx / m
        s.vx = vx + ax * dt
        s.r, s.vy = r_kin, vy_kin
    else:
        alpha_f = math.atan2(s.vy + p.lf * s.r, vx) - delta
        alpha_r = math.atan2(s.vy - p.lr * s.r, vx)
        fyf = _pacejka(alpha_f, p.mu, fzf, p.pacejka_b, p.pacejka_c)
        fyr = _pacejka(alpha_r, p.mu, fzr, p.pacejka_b, p.pacejka_c)
        if abs(alpha_f) > p.alpha_peak or abs(alpha_r) > p.alpha_peak:
            sliding = True

        # traction circle: longitudinal force is limited by unused grip
        cap_r = math.sqrt(max((p.mu * fzr) ** 2 - fyr**2, 0.0))
        cap_f = math.sqrt(max((p.mu * fzf) ** 2 - fyf**2, 0.0))
        fx_r = min(max(fx_r_req, -cap_r), cap_r)
        fx_f = min(max(fx_f_req, -cap_f), cap_f)
        if abs(fx_r_req - fx_r) > 1e-6 or abs(fx_f_req - fx_f) > 1e-6:
            sliding = True          # wheelspin or lock-up

        ax = (fx_r + fx_f * math.cos(delta) - fyf * math.sin(delta)
              - drag - roll) / m
        ay = (fyf * math.cos(delta) + fx_f * math.sin(delta) + fyr) / m
        r_dot = (p.lf * (fyf * math.cos(delta) + fx_f * math.sin(delta))
                 - p.lr * fyr) / p.izz

        s.vx = vx + (ax + s.vy * s.r) * dt
        s.vy = s.vy + (ay - vx * s.r) * dt
        s.r = s.r + r_dot * dt

        # blend toward the kinematic solution as speed drops
        w = min(max((vx - _BLEND_LO) / (_BLEND_HI - _BLEND_LO), 0.0), 1.0)
        s.r = w * s.r + (1.0 - w) * r_kin
        s.vy = w * s.vy + (1.0 - w) * vy_kin

    s.ax = ax
    s.vx = min(max(s.vx, p.min_speed_mps), p.max_speed_mps)
    s.vy = min(max(s.vy, -0.5 * p.max_speed_mps), 0.5 * p.max_speed_mps)

    # pose update (body velocities rotated into the world frame)
    cos_y, sin_y = math.cos(s.yaw), math.sin(s.yaw)
    s.x += (s.vx * cos_y - s.vy * sin_y) * dt
    s.y += (s.vx * sin_y + s.vy * cos_y) * dt
    s.yaw = (s.yaw + s.r * dt + math.pi) % (2.0 * math.pi) - math.pi
    return sliding
