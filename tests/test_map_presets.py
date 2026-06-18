"""Tests for map presets and project root resolution."""

from pathlib import Path

import pytest

from raceline.core.maps import ensure_sample_map, load_preset
from raceline.core.paths import find_project_root, resolve_path


def test_find_project_root():
    root = find_project_root()
    assert (root / "pipeline" / "step1.py").is_file()


def test_resolve_relative_path():
    root = find_project_root()
    p = resolve_path("config/map_presets.yaml", root)
    assert p.is_file()


def test_ensure_and_load_sample_preset():
    root = find_project_root()
    ensure_sample_map(root)
    preset = load_preset("sample", root)
    assert preset.map_path.is_file()
    assert preset.resolution > 0
    assert 0 <= preset.start_col < 2000
    assert 0 <= preset.start_row < 2000
