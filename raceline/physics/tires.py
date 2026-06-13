"""Pacejka Magic Formula tire model (MF 6.1 simplified).

Reference: Pacejka, "Tire and Vehicle Dynamics", 3rd ed.
The lateral force uses the standard sine-atan form; longitudinal slip uses
the same curve with slip ratio kappa instead of slip angle alpha.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TireParams:
    """Pacejka coefficients for one axle (front or rear)."""
    b: float = 10.0          # stiffness factor
    c: float = 1.9           # shape factor
    d: float = 1.0           # peak factor (scaled by mu * Fz)
    e: float = 0.97          # curvature factor
    mu: float = 0.95         # friction coefficient at nominal load

    def lateral_force(self, alpha_rad: float, fz_n: float) -> float:
        """Lateral force [N]; positive alpha -> force opposes slip (negative fy)."""
        return -self._magic_formula(alpha_rad, fz_n)

    def longitudinal_force(self, kappa: float, fz_n: float) -> float:
        """Longitudinal force [N]; positive kappa (drive slip) -> forward force."""
        return self._magic_formula(kappa, fz_n)

    def _magic_formula(self, slip: float, fz_n: float) -> float:
        fz = max(fz_n, 1.0)
        bx = self.b * slip
        d_scaled = self.d * self.mu * fz
        inner = bx - self.e * (bx - math.atan(bx))
        return d_scaled * math.sin(self.c * math.atan(inner))

    @property
    def alpha_peak_rad(self) -> float:
        """Slip angle where the lateral curve peaks (beyond = sliding region)."""
        return math.tan(math.pi / (2.0 * self.c)) / self.b


def combined_slip_forces(
    alpha_rad: float,
    kappa: float,
    fz_n: float,
    tire: TireParams,
) -> tuple[float, float]:
    """Friction-ellipse combination of lateral and longitudinal tire forces.

    Returns (fy, fx) in the tire frame: fy lateral, fx longitudinal.
    """
    fy_pure = tire.lateral_force(alpha_rad, fz_n)
    fx_pure = tire.longitudinal_force(kappa, fz_n)
    fy_max = abs(tire.lateral_force(tire.alpha_peak_rad, fz_n))
    fx_max = abs(tire.longitudinal_force(0.15, fz_n))  # peak drive slip ~15 %
    denom = math.sqrt((fy_pure / max(fy_max, 1e-6)) ** 2
                      + (fx_pure / max(fx_max, 1e-6)) ** 2)
    if denom > 1.0:
        scale = 1.0 / denom
        return fy_pure * scale, fx_pure * scale
    return fy_pure, fx_pure


def slip_angle(vx: float, vy: float, yaw_rate: float,
               steer_rad: float, lf: float) -> float:
    """Front-axle slip angle [rad]."""
    v = max(math.hypot(vx, vy), 0.05)
    beta = math.atan2(vy, vx)
    return beta + yaw_rate * lf / v - steer_rad


def rear_slip_angle(vx: float, vy: float, yaw_rate: float, lr: float) -> float:
    v = max(math.hypot(vx, vy), 0.05)
    beta = math.atan2(vy, vx)
    return beta - yaw_rate * lr / v


def slip_ratio(wheel_rps: float, wheel_radius_m: float,
               vx_body: float) -> float:
    """Longitudinal slip ratio kappa at the driven axle."""
    v_wheel = wheel_rps * 2.0 * math.pi * wheel_radius_m
    v_ref = max(abs(vx_body), 0.1)
    return (v_wheel - vx_body) / v_ref
