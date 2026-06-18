"""Tests for Pacejka tire model."""

import math

import pytest

from raceline.physics.tires import TireParams, combined_slip_forces, slip_angle


def test_lateral_force_opposes_slip():
    tire = TireParams(mu=1.0, b=10.0, c=1.9, d=1.0)
    alpha = math.radians(5.0)
    fy = tire.lateral_force(alpha, 100.0)
    assert fy < 0.0  # positive slip -> force pushes right (negative fy)


def test_zero_slip_zero_force():
    tire = TireParams()
    assert abs(tire.lateral_force(0.0, 200.0)) < 1e-6


def test_combined_slip_within_friction_ellipse():
    tire = TireParams(mu=1.0)
    fy, fx = combined_slip_forces(0.1, 0.1, 200.0, tire)
    fy_max = abs(tire.lateral_force(tire.alpha_peak_rad, 200.0))
    assert abs(fy) <= fy_max * 1.01
    assert abs(fx) <= abs(tire.longitudinal_force(0.15, 200.0)) * 1.01


def test_slip_angle_at_steady_state():
    alpha = slip_angle(vx=5.0, vy=0.2, yaw_rate=0.1, steer_rad=0.05, lf=0.16)
    assert abs(alpha) < 0.5
