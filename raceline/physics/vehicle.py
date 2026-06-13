"""High-fidelity dynamic single-track vehicle model.

Upgrades over the legacy bicycle model:
  • Pacejka MF 6.1 tires with combined-slip friction ellipse
  • Longitudinal + lateral load transfer
  • VESC motor / wheel-speed dynamics on the driven axle
  • Servo actuator with rate limit and first-order lag
  • RK4 integration for numerical stability at 50 Hz
  • Low-speed kinematic blend (standard practice below ~1 m/s)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .motor import MotorParams, MotorState, integrate_wheel_speed, motor_torque_from_accel
from .tires import (
    TireParams,
    combined_slip_forces,
    rear_slip_angle,
    slip_angle,
    slip_ratio,
)

G = 9.81
RHO_AIR = 1.2
_BLEND_LO = 0.5
_BLEND_HI = 1.5


@dataclass
class CarParams:
    wheelbase_m: float = 0.32
    track_width_m: float = 0.26
    max_speed_mps: float = 8.0
    min_speed_mps: float = 0.0
    max_steer_rad: float = 0.40
    steer_rate_rps: float = 4.0
    steer_tau_s: float = 0.05          # servo first-order time constant
    max_accel_mps2: float = 4.0
    max_brake_mps2: float = 6.0
    safety_radius_m: float = 0.15
    mass_kg: float = 3.5
    chassis_l_m: float = 0.50
    chassis_w_m: float = 0.30
    cog_height_m: float = 0.04
    cog_long_offset_m: float = 0.0     # + = toward front axle
    yaw_inertia: float | None = None
    cda_m2: float = 0.04
    crr: float = 0.02
    tire_front: TireParams = field(default_factory=TireParams)
    tire_rear: TireParams = field(default_factory=TireParams)
    motor: MotorParams = field(default_factory=MotorParams)
    use_motor_dynamics: bool = True

    @property
    def lf(self) -> float:
        return self.wheelbase_m / 2.0 + self.cog_long_offset_m

    @property
    def lr(self) -> float:
        return self.wheelbase_m - self.lf

    @property
    def izz(self) -> float:
        if self.yaw_inertia is not None:
            return self.yaw_inertia
        return self.mass_kg * (self.chassis_l_m ** 2 + self.chassis_w_m ** 2) / 12.0

    @property
    def mu(self) -> float:
        return self.tire_rear.mu


@dataclass
class CarState:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    r: float = 0.0
    steer: float = 0.0
    steer_cmd: float = 0.0
    ax: float = 0.0
    ay: float = 0.0
    motor: MotorState = field(default_factory=MotorState)

    @property
    def v(self) -> float:
        return self.vx


@dataclass
class StepDiagnostics:
    sliding: bool = False
    wheelspin: bool = False
    saturation_front: bool = False
    saturation_rear: bool = False


def _clone(s: CarState) -> CarState:
    return CarState(
        x=s.x, y=s.y, yaw=s.yaw, vx=s.vx, vy=s.vy, r=s.r,
        steer=s.steer, steer_cmd=s.steer_cmd, ax=s.ax, ay=s.ay,
        motor=MotorState(s.motor.wheel_rps, s.motor.erpm),
    )


def _vertical_loads(p: CarParams, ax: float, ay: float) -> tuple[float, float]:
    """Longitudinal + lateral weight transfer -> front/rear normal loads [N]."""
    m, h, L, t = p.mass_kg, p.cog_height_m, p.wheelbase_m, p.track_width_m
    fz = m * G
    d_long = m * ax * h / L
    d_lat = m * ay * h / max(t, 0.01)
    fzf = fz * p.lr / L - d_long + 0.5 * d_lat
    fzr = fz - fzf
    total = max(fzf + fzr, 1.0)
    fzf = max(0.05 * fz, min(fzf, 0.95 * fz))
    fzr = total - fzf
    return fzf, fzr


def _derivatives(s: CarState, steer_target: float, accel_cmd: float,
                 p: CarParams) -> tuple[CarState, StepDiagnostics]:
    """Compute state derivatives for RK4 (stored in a CarState-shaped delta)."""
    diag = StepDiagnostics()
    m, L = p.mass_kg, p.wheelbase_m
    vx = max(s.vx, 0.0)

    # Servo dynamics
    steer_err = steer_target - s.steer
    steer_rate = min(max(steer_err / max(p.steer_tau_s, 1e-3),
                         -p.steer_rate_rps), p.steer_rate_rps)
    steer_dot = steer_rate
    delta = min(max(s.steer, -p.max_steer_rad), p.max_steer_rad)

    fzf, fzr = _vertical_loads(p, s.ax, s.ay)
    drag = 0.5 * RHO_AIR * p.cda_m2 * vx * vx
    roll = p.crr * m * G if vx > 0.05 else 0.0

    r_kin = vx / L * math.tan(delta) if abs(delta) < 1.4 else 0.0
    vy_kin = r_kin * p.lr

    if vx < _BLEND_LO:
        fx = m * accel_cmd - drag - roll
        ax = fx / m
        d = _clone(s)
        d.vx, d.vy, d.r = ax, 0.0, r_kin
        d.steer = steer_dot
        d.yaw = s.r
        s.ax, s.ay = ax, 0.0
        return d, diag

    alpha_f = slip_angle(vx, s.vy, s.r, delta, p.lf)
    alpha_r = rear_slip_angle(vx, s.vy, s.r, p.lr)

    if p.use_motor_dynamics:
        kappa = slip_ratio(s.motor.wheel_rps, p.motor.wheel_radius_m, vx)
    else:
        kappa = accel_cmd / max(G * p.tire_rear.mu, 0.1) * 0.1

    fyf, fxf = combined_slip_forces(alpha_f, 0.0, fzf, p.tire_front)
    fyr, fxr = combined_slip_forces(alpha_r, kappa, fzr, p.tire_rear)

    if abs(alpha_f) > p.tire_front.alpha_peak_rad:
        diag.saturation_front = True
        diag.sliding = True
    if abs(alpha_r) > p.tire_rear.alpha_peak_rad:
        diag.saturation_rear = True
        diag.sliding = True

    # Requested longitudinal force, traction-limited
    if accel_cmd >= 0.0:
        fx_req_r, fx_req_f = m * accel_cmd, 0.0
    else:
        fx_req_f = m * accel_cmd * (fzf / (fzf + fzr))
        fx_req_r = m * accel_cmd * (fzr / (fzf + fzr))

    cap_r = math.sqrt(max((p.tire_rear.mu * fzr) ** 2 - fyr ** 2, 0.0))
    cap_f = math.sqrt(max((p.tire_front.mu * fzf) ** 2 - fyf ** 2, 0.0))
    fxr = min(max(fx_req_r, -cap_r), cap_r)
    fxf = min(max(fx_req_f, -cap_f), cap_f)
    if abs(fx_req_r - fxr) > 1e-3 or abs(fx_req_f - fxf) > 1e-3:
        diag.wheelspin = True
        diag.sliding = True

    ax = (fxr + fxf * math.cos(delta) - fyf * math.sin(delta) - drag - roll) / m
    ay = (fyf * math.cos(delta) + fxf * math.sin(delta) + fyr) / m
    r_dot = (p.lf * (fyf * math.cos(delta) + fxf * math.sin(delta))
             - p.lr * fyr) / p.izz

    w = min(max((vx - _BLEND_LO) / (_BLEND_HI - _BLEND_LO), 0.0), 1.0)
    vy_dot = w * (ay - vx * s.r) + (1.0 - w) * (vy_kin - s.vy) / 0.05
    r_eff = w * r_dot + (1.0 - w) * (r_kin - s.r) / 0.05

    d = _clone(s)
    d.vx = ax + s.vy * s.r
    d.vy = vy_dot
    d.r = r_eff
    d.steer = steer_dot
    d.yaw = s.r
    d.ax, d.ay = ax, ay
    return d, diag


def _apply_delta(s: CarState, d: CarState, dt: float) -> CarState:
    out = _clone(s)
    out.x += (s.vx * math.cos(s.yaw) - s.vy * math.sin(s.yaw)) * dt
    out.y += (s.vx * math.sin(s.yaw) + s.vy * math.cos(s.yaw)) * dt
    out.yaw = (s.yaw + s.r * dt + math.pi) % (2 * math.pi) - math.pi
    out.vx += d.vx * dt
    out.vy += d.vy * dt
    out.r += d.r * dt
    out.steer += d.steer * dt
    out.ax, out.ay = d.ax, d.ay
    return out


def dynamic_step(
    state: CarState,
    steer_target: float,
    accel_cmd: float,
    params: CarParams,
    dt: float,
) -> StepDiagnostics:
    """Advance `state` by `dt` using RK4 integration. Mutates in place."""
    k1, _ = _derivatives(state, steer_target, accel_cmd, params)
    s2 = _apply_delta(state, k1, dt * 0.5)
    k2, _ = _derivatives(s2, steer_target, accel_cmd, params)
    s3 = _apply_delta(state, k2, dt * 0.5)
    k3, _ = _derivatives(s3, steer_target, accel_cmd, params)
    s4 = _apply_delta(state, k3, dt)
    k4, diag = _derivatives(s4, steer_target, accel_cmd, params)

    state.vx += (k1.vx + 2 * k2.vx + 2 * k3.vx + k4.vx) * dt / 6.0
    state.vy += (k1.vy + 2 * k2.vy + 2 * k3.vy + k4.vy) * dt / 6.0
    state.r += (k1.r + 2 * k2.r + 2 * k3.r + k4.r) * dt / 6.0
    state.steer += (k1.steer + 2 * k2.steer + 2 * k3.steer + k4.steer) * dt / 6.0
    state.ax = k4.ax
    state.ay = k4.ay

    cos_y, sin_y = math.cos(state.yaw), math.sin(state.yaw)
    v_world_x = state.vx * cos_y - state.vy * sin_y
    v_world_y = state.vx * sin_y + state.vy * cos_y
    state.x += v_world_x * dt
    state.y += v_world_y * dt
    state.yaw = (state.yaw + state.r * dt + math.pi) % (2 * math.pi) - math.pi

    state.steer = min(max(state.steer, -params.max_steer_rad),
                      params.max_steer_rad)
    state.vx = min(max(state.vx, params.min_speed_mps), params.max_speed_mps)
    state.vy = min(max(state.vy, -0.5 * params.max_speed_mps),
                   0.5 * params.max_speed_mps)

    if params.use_motor_dynamics:
        drive_t = motor_torque_from_accel(
            max(accel_cmd, 0.0), params.mass_kg,
            params.motor.wheel_radius_m, params.motor)
        brake_t = motor_torque_from_accel(
            min(accel_cmd, 0.0), params.mass_kg,
            params.motor.wheel_radius_m, params.motor)
        state.motor.wheel_rps = integrate_wheel_speed(
            state.motor.wheel_rps, abs(drive_t), abs(brake_t),
            state.ax * params.mass_kg, params.motor.wheel_radius_m,
            params.motor, dt)
        state.motor.erpm = params.motor.wheel_rps_to_erpm(state.motor.wheel_rps)

    return diag


# Backward-compatible alias used by pipeline code
kinematic_step = dynamic_step
