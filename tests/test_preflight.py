"""Tests for pipeline preflight checks."""

import pytest

from raceline.core import PreflightError, check_step_prerequisites


def test_step2_missing_artifacts(tmp_path):
    with pytest.raises(PreflightError, match="map.yaml"):
        check_step_prerequisites(2, tmp_path)


def test_unknown_step(tmp_path):
    with pytest.raises(ValueError):
        check_step_prerequisites(99, tmp_path)
