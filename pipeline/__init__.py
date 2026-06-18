"""Offline raceline pipeline for the self-driving RC car.

Package layout (v2)
-------------------
raceline/     Core library — physics, validation, Gazebo export, control
pipeline/     CLI entry points (python -m pipeline.stepN)
car/          VESC hardware tools

Steps 1–5 consume/write artifacts in a shared directory (default: artifacts/).
Every step runs preflight checks before starting.
"""
