"""Backward-compatible re-export of the upgraded vehicle dynamics model.

New code should import from ``raceline.physics`` directly.
"""

from raceline.physics.vehicle import (
    CarParams,
    CarState,
    StepDiagnostics,
    dynamic_step,
    kinematic_step,
)

__all__ = [
    "CarParams",
    "CarState",
    "StepDiagnostics",
    "dynamic_step",
    "kinematic_step",
]
