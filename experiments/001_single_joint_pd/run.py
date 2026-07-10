#!/usr/bin/env python3
"""Apply constant torque to one hinge and print the resulting joint state.

Phase 1 deliberately uses open-loop torque: the command does not depend on the
measured angle or velocity. Phase 2 will replace it with feedback (PD control).
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "single_joint" / "single_joint.xml"
VIEWER_WORKER_ENV = "ROBO_SIM_VIEWER_WORKER"


@dataclass(frozen=True)
class Sample:
    """One observation from the simulated joint."""

    time_s: float
    position_rad: float
    velocity_rad_s: float
    torque_nm: float


def simulate_constant_torque(
    torque_nm: float, duration_s: float, sample_count: int
) -> list[Sample]:
    """Run the headless experiment and return evenly spaced samples."""
    if duration_s <= 0:
        raise ValueError("duration must be greater than zero")
    if sample_count < 2:
        raise ValueError("samples must be at least 2 (initial and final state)")

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hinge")
    actuator_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, "joint_motor"
    )
    qpos_address = model.jnt_qposadr[joint_id]
    qvel_address = model.jnt_dofadr[joint_id]

    control_min, control_max = model.actuator_ctrlrange[actuator_id]
    if not control_min <= torque_nm <= control_max:
        raise ValueError(
            f"torque must be within [{control_min:g}, {control_max:g}] N·m"
        )

    step_count = max(1, round(duration_s / model.opt.timestep))
    sample_steps = {
        round(index * step_count / (sample_count - 1))
        for index in range(sample_count)
    }

    def observe() -> Sample:
        return Sample(
            time_s=float(data.time),
            position_rad=float(data.qpos[qpos_address]),
            velocity_rad_s=float(data.qvel[qvel_address]),
            torque_nm=float(data.ctrl[actuator_id]),
        )

    data.ctrl[actuator_id] = torque_nm
    samples = [observe()]
    for step in range(1, step_count + 1):
        mujoco.mj_step(model, data)
        if step in sample_steps:
            samples.append(observe())
    return samples


def print_viewer_explanation(torque_nm: float) -> None:
    """Explain the physical meaning before opening the GUI."""
    gravity_torque_scale = 1.0 * 9.81 * 0.25  # mass × gravity × COM distance
    expected_angle = math.asin(torque_nm / gravity_torque_scale)

    print("\nWhat the Viewer shows:")
    print("  blue box      = fixed base (it must not move)")
    print("  yellow hub    = hinge / simplified motor location")
    print("  orange rod    = 1 kg link, similar to a simplified lower leg")
    print(f"  motor input   = constant {torque_nm:.3f} N·m (not a target angle)")
    print(
        "  expected rest = "
        f"about {expected_angle:.3f} rad / {math.degrees(expected_angle):.1f} deg, "
        "where motor and gravity torque balance"
    )
    print("\nThe GUI stays open until you close its window.")
    print("You can also focus this terminal and press Ctrl+C.\n", flush=True)


def run_managed_viewer_worker(torque_nm: float) -> Sample:
    """Run MuJoCo's managed GUI inside the isolated worker process."""
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hinge")
    actuator_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, "joint_motor"
    )
    control_min, control_max = model.actuator_ctrlrange[actuator_id]
    if not control_min <= torque_nm <= control_max:
        raise ValueError(
            f"torque must be within [{control_min:g}, {control_max:g}] N·m"
        )

    data.ctrl[actuator_id] = torque_nm

    # Managed mode owns the GUI/physics lifecycle and is more stable on WSLg
    # than creating and automatically destroying a passive viewer after N seconds.
    from mujoco import viewer as mujoco_viewer

    mujoco_viewer.launch(model, data)
    print("Viewer window closed.")

    return Sample(
        time_s=float(data.time),
        position_rad=float(data.qpos[model.jnt_qposadr[joint_id]]),
        velocity_rad_s=float(data.qvel[model.jnt_dofadr[joint_id]]),
        torque_nm=float(data.ctrl[actuator_id]),
    )


def run_viewer_subprocess(torque_nm: float) -> int:
    """Keep terminal signal handling outside the native GUI process."""
    if not -2.0 <= torque_nm <= 2.0:
        raise ValueError("torque must be within [-2, 2] N·m")

    print_viewer_explanation(torque_nm)
    environment = os.environ.copy()
    environment[VIEWER_WORKER_ENV] = "1"
    process = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--view",
            "--torque",
            str(torque_nm),
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )

    try:
        return_code = process.wait()
    except KeyboardInterrupt:
        print("\nCtrl+C received; closing the Viewer...", flush=True)
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        print("Viewer stopped. Terminal control restored.")
        return 130

    if return_code != 0:
        print(
            f"Viewer process exited abnormally (code {return_code}). "
            "The terminal remains safe; run 'stty sane' only if its input looks wrong.",
            file=sys.stderr,
        )
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply a constant torque to the Phase 1 single-joint model."
    )
    parser.add_argument(
        "--torque",
        type=float,
        default=0.5,
        help="motor torque in N·m; valid range is [-2, 2] (default: 0.5)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
        help="headless simulation duration in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=6,
        help="headless states to print, including initial/final (default: 6)",
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="open the interactive Viewer until it is closed (requires WSLg/display)",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.view and os.environ.get(VIEWER_WORKER_ENV) == "1":
        try:
            final = run_managed_viewer_worker(args.torque)
        except ValueError as exc:
            parser.error(str(exc))
        print(
            "Final state: "
            f"time={final.time_s:.3f} s, "
            f"position={final.position_rad:.6f} rad, "
            f"velocity={final.velocity_rad_s:.6f} rad/s"
        )
        return 0

    print("Single-joint constant-torque experiment (Phase 1)")
    print(f"model: {MODEL_PATH.relative_to(PROJECT_ROOT)}")

    if args.view:
        try:
            return run_viewer_subprocess(args.torque)
        except ValueError as exc:
            parser.error(str(exc))

    try:
        samples = simulate_constant_torque(args.torque, args.duration, args.samples)
    except ValueError as exc:
        parser.error(str(exc))

    print("time_s  position_rad  velocity_rad_s  torque_nm")
    for sample in samples:
        print(
            f"{sample.time_s:6.3f}  {sample.position_rad:12.6f}  "
            f"{sample.velocity_rad_s:14.6f}  {sample.torque_nm:9.3f}"
        )

    final = samples[-1]
    print(
        "Final state: "
        f"position={final.position_rad:.6f} rad, "
        f"velocity={final.velocity_rad_s:.6f} rad/s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
