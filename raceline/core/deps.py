"""Dependency checks with Jetson-friendly install instructions."""

from __future__ import annotations

import importlib
import platform
import sys

_STEP1_MODULES = {
    "numpy": "numpy",
    "cv2": "opencv-python-headless",
    "scipy": "scipy",
    "skimage": "scikit-image",
    "matplotlib": "matplotlib",
    "yaml": "PyYAML",
}

_STEP1_APT = [
    "python3-pip",
    "python3-venv",
    "python3-numpy",
    "python3-scipy",
    "python3-opencv",
    "python3-matplotlib",
    "python3-yaml",
    "libgl1",
    "libglib2.0-0",
]


def _is_jetson() -> bool:
    try:
        with open("/etc/nv_tegra_release") as f:
            return True
    except OSError:
        return False


def _missing_step1() -> list[tuple[str, str]]:
    missing = []
    for mod, pip_name in _STEP1_MODULES.items():
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append((mod, pip_name))
    return missing


def check_step1_deps() -> list[tuple[str, str]]:
    """Return list of (module, pip_package) that failed to import."""
    return _missing_step1()


def step1_install_hint() -> str:
    """Human-readable fix for missing step-1 dependencies."""
    lines = [
        "Install step-1 dependencies (no PyTorch / RL packages needed yet):",
        "",
        "  pip install -r requirements-step1.txt",
        "",
    ]
    if _is_jetson() or platform.machine() == "aarch64":
        lines.extend([
            "Jetson / ARM64 — if pip fails, use apt first:",
            "",
            "  sudo apt update",
            f"  sudo apt install -y {' '.join(_STEP1_APT)}",
            "  pip install scikit-image PyYAML",
            "",
            "Or run the all-in-one setup script:",
            "",
            "  bash scripts/jetson_setup.sh",
            "",
        ])
    lines.append(f"Python: {sys.version.split()[0]} on {platform.machine()}")
    return "\n".join(lines)


def require_step1_deps() -> None:
    """Exit with code 2 and install instructions if step-1 deps are missing."""
    missing = check_step1_deps()
    if not missing:
        return
    print("ERROR: missing Python packages required for step 1:\n")
    for mod, pip_pkg in missing:
        print(f"  - {mod}  (pip install {pip_pkg})")
    print(f"\n{step1_install_hint()}")
    raise SystemExit(2)
