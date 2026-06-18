"""Tests for dynamic vehicle model."""

import math

import pytest

from raceline.physics import CarParams, CarState, dynamic_step, TireParams


@pytest.fixture
def params():
    tire = TireParams(mu=0.95, b=10.0, c=1.9)
    return CarParams(
        wheelbase_m=0.32,
        mass_kg=3.5,
        max_steer_rad=0.4,
        max_accel_mps2=4.0,
        max_brake_mps2=6.0,
        tire_front=tire,
        tire_rear=tire,
        use_motor_dynamics=True,
    )


def test_straight_line_acceleration(params):
    s = CarState(vx=2.0)
    for _ in range(100):
        dynamic_step(s, 0.0, 2.0, params, 0.01)
    assert s.vx > 2.0


def test_steering_changes_yaw(params):
    s = CarState(vx=3.0)
    yaw0 = s.yaw
    for _ in range(200):
        dynamic_step(s, 0.3, 0.0, params, 0.01)
    assert abs(s.yaw - yaw0) > 0.1


def test_speed_clamped_to_max(params):
    s = CarState(vx=params.max_speed_mps - 0.1)
    for _ in range(500):
        dynamic_step(s, 0.0, params.max_accel_mps2, params, 0.01)
    assert s.vx <= params.max_speed_mps + 1e-6


def test_motor_erpm_tracks_wheel(params):
    s = CarState(vx=1.0)
    for _ in range(50):
        dynamic_step(s, 0.0, 3.0, params, 0.02)
    assert s.motor.erpm != 0.0 or s.vx < 0.5


def test_low_speed_kinematic_blend(params):
    s = CarState(vx=0.1)
    dynamic_step(s, 0.2, 1.0, params, 0.05)
    assert math.isfinite(s.x) and math.isfinite(s.y)
