"""VESC motor + drivetrain model for accurate longitudinal dynamics."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class MotorParams:
    """Flipsky / VESC-class brushless drive parameters."""
    pole_pairs: int = 7
    gear_ratio: float = 3.0          # motor : wheel
    wheel_radius_m: float = 0.05
    max_current_a: float = 40.0
    max_erpm: float = 100_000.0
    torque_constant_nm_per_a: float = 0.02   # Kt (motor torque per amp)
    efficiency: float = 0.85
    rolling_inertia_kgm2: float = 1e-4       # motor + wheel reflected inertia

    @property
    def max_wheel_torque_nm(self) -> float:
        return self.max_current_a * self.torque_constant_nm_per_a * self.gear_ratio

    def erpm_to_wheel_rps(self, erpm: float) -> float:
        motor_rps = erpm / (60.0 * self.pole_pairs)
        return motor_rps / self.gear_ratio

    def wheel_rps_to_erpm(self, wheel_rps: float) -> float:
        motor_rps = wheel_rps * self.gear_ratio
        return motor_rps * 60.0 * self.pole_pairs


@dataclass
class MotorState:
    wheel_rps: float = 0.0
    erpm: float = 0.0


def motor_torque_from_accel(
    accel_cmd_mps2: float,
    mass_kg: float,
    wheel_radius_m: float,
    motor: MotorParams,
) -> float:
    """Convert requested longitudinal acceleration to rear-wheel torque [Nm]."""
    force = mass_kg * accel_cmd_mps2
    torque = force * wheel_radius_m
    max_t = motor.max_wheel_torque_nm * motor.efficiency
    return max(-max_t, min(torque, max_t))


def integrate_wheel_speed(
    wheel_rps: float,
    drive_torque_nm: float,
    brake_torque_nm: float,
    fx_ground_n: float,
    wheel_radius_m: float,
    motor: MotorParams,
    dt: float,
) -> float:
    """Integrate driven wheel angular velocity with motor + ground reaction."""
    # Net torque on wheel: motor drive minus brake minus ground reaction
    t_ground = fx_ground_n * wheel_radius_m
    t_net = drive_torque_nm - brake_torque_nm - t_ground
    alpha = t_net / max(motor.rolling_inertia_kgm2, 1e-8)
    new_rps = wheel_rps + alpha * dt
    max_rps = motor.erpm_to_wheel_rps(motor.max_erpm)
    return max(-max_rps * 0.1, min(new_rps, max_rps))
