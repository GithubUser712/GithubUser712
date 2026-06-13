"""Step 0 entry point — verify and bootstrap the environment.

    python -m raceline.setup              # full check + instructions
    python -m raceline.setup --verify     # exit 0 only if step-1 ready
    python -m raceline.setup --install    # pip install step-1 deps
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path


def _is_jetson() -> bool:
    return Path("/etc/nv_tegra_release").is_file()


def _has_code(root: Path) -> bool:
    return (root / "pipeline" / "step1.py").is_file()


def verify(root: Path) -> int:
    """Print environment report; return 0 if step-1 can run."""
    print(f"Platform : {platform.system()} {platform.machine()}")
    print(f"Python   : {sys.version.split()[0]}")
    print(f"Root     : {root}")
    if _is_jetson():
        print("Device   : NVIDIA Jetson (Tegra)")
    print()

    ok = True

    if not _has_code(root):
        print("FAIL  pipeline/step1.py missing")
        print("      git checkout cursor/step5-6-vesc-3d48")
        ok = False
    else:
        print("OK    project code present")

    from raceline.core.deps import check_step1_deps, step1_install_hint
    missing = check_step1_deps()
    if missing:
        ok = False
        print("FAIL  missing Python packages:")
        for mod, pkg in missing:
            print(f"        {mod}  ->  pip install {pkg}")
        print()
        print(step1_install_hint())
    else:
        print("OK    step-1 Python packages")

    maps = root / "maps"
    if maps.is_dir() and any(maps.glob("*.png")):
        print("OK    map images found")
    else:
        print("NOTE  no maps/*.png yet — step 1 --preset sample will generate one")

    if ok:
        print()
        print("Ready for step 1:")
        print("  python -m pipeline.step1 --preset sample")
    return 0 if ok else 1


def install_step1(root: Path) -> int:
    req = root / "requirements-step1.txt"
    if not req.is_file():
        print(f"ERROR: {req} not found")
        return 1
    print(f"Installing {req} ...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(req)],
        cwd=root, check=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", str(root), "--no-deps"],
        check=True)
    return verify(root)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Step 0: environment setup & verify")
    ap.add_argument("--verify", action="store_true",
                    help="check only; exit 1 if not ready")
    ap.add_argument("--install", action="store_true",
                    help="pip install step-1 dependencies")
    args = ap.parse_args(argv)

    from raceline.core.paths import find_project_root
    try:
        root = find_project_root()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1

    if args.install:
        return install_step1(root)
    return verify(root)


if __name__ == "__main__":
    raise SystemExit(main())
