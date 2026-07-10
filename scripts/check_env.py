#!/usr/bin/env python3
"""Verify the Phase 0 Python and MuJoCo environment without opening a GUI."""

from __future__ import annotations

import importlib
import platform
import sys
from importlib import metadata


REQUIRED_MODULES = ("mujoco", "numpy", "matplotlib", "scipy", "pytest", "jupyter")


def package_version(module_name: str) -> str:
    """Return an installed distribution version when available."""
    distribution_name = "jupyter" if module_name == "jupyter" else module_name
    try:
        return metadata.version(distribution_name)
    except metadata.PackageNotFoundError:
        module = importlib.import_module(module_name)
        return str(getattr(module, "__version__", "unknown"))


def check_mujoco_model() -> tuple[int, int]:
    """Compile a tiny model to prove that the native MuJoCo library loads."""
    import mujoco

    xml = """
    <mujoco model="phase0_check">
      <worldbody>
        <body name="link">
          <joint name="hinge" type="hinge"/>
          <geom type="capsule" size="0.03 0.2"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    return model.nq, model.nv


def main() -> int:
    print("LinkJoin Robo SIM — Phase 0 environment check")
    print(f"Python: {platform.python_version()} ({sys.executable})")
    print(f"Platform: {platform.platform()}")

    if sys.version_info < (3, 10):
        print("FAIL: Python 3.10 or newer is required.", file=sys.stderr)
        return 1

    failed: list[str] = []
    for module_name in REQUIRED_MODULES:
        try:
            importlib.import_module(module_name)
            print(f"OK: {module_name} {package_version(module_name)}")
        except Exception as exc:  # Report every missing/broken dependency in one run.
            failed.append(module_name)
            print(f"FAIL: {module_name}: {exc}", file=sys.stderr)

    try:
        import robo_sim

        print(f"OK: robo_sim {robo_sim.__version__}")
    except Exception as exc:
        failed.append("robo_sim")
        print(f"FAIL: robo_sim: {exc}", file=sys.stderr)

    if "mujoco" not in failed:
        try:
            nq, nv = check_mujoco_model()
            print(f"OK: MuJoCo compiled and stepped a minimal model (nq={nq}, nv={nv})")
        except Exception as exc:
            failed.append("MuJoCo model check")
            print(f"FAIL: MuJoCo model check: {exc}", file=sys.stderr)

    if failed:
        print(f"Environment check failed: {', '.join(failed)}", file=sys.stderr)
        return 1

    print("All Phase 0 checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
