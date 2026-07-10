#!/usr/bin/env python3
"""Run the Phase 1 constant-torque or Phase 2 PD single-joint experiment."""

from __future__ import annotations

import argparse
import math
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import mujoco
from robo_sim.controllers.pd import PDController
from robo_sim.ui.learning_panel import LearningPanelServer


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


@dataclass(frozen=True)
class PDSample:
    """One closed-loop observation and the calculation that produced it."""

    time_s: float
    position_rad: float
    velocity_rad_s: float
    target_position_rad: float
    position_error_rad: float
    raw_torque_nm: float
    torque_nm: float
    saturated: bool


def simulate_pd(
    *,
    target_position_rad: float,
    kp: float,
    kd: float,
    duration_s: float,
) -> list[PDSample]:
    """Run a headless PD loop and record every physics step."""
    if duration_s <= 0:
        raise ValueError("duration must be greater than zero")

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hinge")
    actuator_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, "joint_motor"
    )
    qpos_address = model.jnt_qposadr[joint_id]
    qvel_address = model.jnt_dofadr[joint_id]
    joint_min, joint_max = model.jnt_range[joint_id]
    if not joint_min <= target_position_rad <= joint_max:
        raise ValueError(
            f"target must be within [{math.degrees(joint_min):.1f}, "
            f"{math.degrees(joint_max):.1f}] degrees"
        )
    control_min, control_max = model.actuator_ctrlrange[actuator_id]
    controller = PDController(
        kp=kp,
        kd=kd,
        target_position_rad=target_position_rad,
        torque_min_nm=float(control_min),
        torque_max_nm=float(control_max),
    )

    def control_and_observe() -> PDSample:
        output = controller.compute(
            position_rad=float(data.qpos[qpos_address]),
            velocity_rad_s=float(data.qvel[qvel_address]),
        )
        data.ctrl[actuator_id] = output.torque_nm
        return PDSample(
            time_s=float(data.time),
            position_rad=float(data.qpos[qpos_address]),
            velocity_rad_s=float(data.qvel[qvel_address]),
            target_position_rad=target_position_rad,
            position_error_rad=output.position_error_rad,
            raw_torque_nm=output.raw_torque_nm,
            torque_nm=output.torque_nm,
            saturated=output.saturated,
        )

    step_count = max(1, round(duration_s / model.opt.timestep))
    samples = [control_and_observe()]
    for _ in range(step_count):
        mujoco.mj_step(model, data)
        samples.append(control_and_observe())
    return samples


def evenly_spaced_samples(
    samples: list[PDSample], sample_count: int
) -> list[PDSample]:
    if sample_count < 2:
        raise ValueError("samples must be at least 2 (initial and final state)")
    last = len(samples) - 1
    return [samples[round(index * last / (sample_count - 1))] for index in range(sample_count)]


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


def print_pd_viewer_explanation(target_deg: float, kp: float, kd: float) -> None:
    """Explain what changes when the GUI is driven by feedback."""
    print("\nWhat the PD Viewer shows:")
    print("  blue box      = fixed base (it must not move)")
    print("  yellow hub    = hinge / simplified motor location")
    print("  orange rod    = the controlled 1 kg link")
    print(f"  target angle  = {target_deg:.1f} deg")
    print(f"  gains         = Kp={kp:g}, Kd={kd:g}")
    print("  motor torque  = recalculated from qpos/qvel every physics step")
    print("  torque limit  = [-2, 2] N·m")
    print(
        "\nUse the Chinese panel to change target/Kp/Kd exactly. "
        "The native purple Control is now an output and will be overwritten."
    )
    print("Close the GUI or focus this terminal and press Ctrl+C.\n", flush=True)


def run_managed_viewer_worker(
    *,
    mode: str,
    torque_nm: float,
    target_position_rad: float,
    kp: float,
    kd: float,
    control_port: int,
) -> Sample:
    """Run MuJoCo's managed GUI inside the isolated worker process."""
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hinge")
    actuator_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, "joint_motor"
    )
    control_min, control_max = model.actuator_ctrlrange[actuator_id]
    if mode == "torque" and not control_min <= torque_nm <= control_max:
        raise ValueError(
            f"torque must be within [{control_min:g}, {control_max:g}] N·m"
        )

    pd_controller: PDController | None = None
    if mode == "pd":
        joint_min, joint_max = model.jnt_range[joint_id]
        if not joint_min <= target_position_rad <= joint_max:
            raise ValueError(
                f"target must be within [{math.degrees(joint_min):.1f}, "
                f"{math.degrees(joint_max):.1f}] degrees"
            )
        pd_controller = PDController(
            kp=kp,
            kd=kd,
            target_position_rad=target_position_rad,
            torque_min_nm=float(control_min),
            torque_max_nm=float(control_max),
        )
        qpos_address = model.jnt_qposadr[joint_id]
        qvel_address = model.jnt_dofadr[joint_id]

        def pd_callback(
            callback_model: mujoco.MjModel, callback_data: mujoco.MjData
        ) -> None:
            del callback_model
            output = pd_controller.compute(
                position_rad=float(callback_data.qpos[qpos_address]),
                velocity_rad_s=float(callback_data.qvel[qvel_address]),
            )
            callback_data.ctrl[actuator_id] = output.torque_nm

        pd_callback(model, data)
        mujoco.set_mjcb_control(pd_callback)
    else:
        data.ctrl[actuator_id] = torque_nm
    panel = LearningPanelServer(
        model,
        data,
        joint_id,
        actuator_id,
        port=control_port,
        pd_controller=pd_controller,
    )
    panel.start()

    # Managed mode owns the GUI/physics lifecycle and is more stable on WSLg
    # than creating and automatically destroying a passive viewer after N seconds.
    from mujoco import viewer as mujoco_viewer

    try:
        mujoco_viewer.launch(model, data)
    finally:
        panel.close()
        if pd_controller is not None:
            mujoco.set_mjcb_control(None)
    print("Viewer window closed.")

    return Sample(
        time_s=float(data.time),
        position_rad=float(data.qpos[model.jnt_qposadr[joint_id]]),
        velocity_rad_s=float(data.qvel[model.jnt_dofadr[joint_id]]),
        torque_nm=float(data.ctrl[actuator_id]),
    )


def find_available_port() -> int:
    """Reserve an ephemeral localhost port for the learning panel."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def wait_for_panel(url: str, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url + "api/state", timeout=0.2) as response:
                response.read()
                return True
        except OSError:
            time.sleep(0.1)
    return False


def open_panel_in_browser(url: str) -> bool:
    """Open the localhost panel in the Windows browser when running in WSL."""
    explorer = shutil.which("explorer.exe")
    if explorer is None:
        return False
    try:
        subprocess.Popen(
            [explorer, url],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return True
    except OSError:
        return False


def run_viewer_subprocess(
    *,
    mode: str,
    torque_nm: float,
    target_position_rad: float,
    kp: float,
    kd: float,
    open_browser: bool = True,
) -> int:
    """Keep terminal signal handling outside the native GUI process."""
    if mode == "torque" and not -2.0 <= torque_nm <= 2.0:
        raise ValueError("torque must be within [-2, 2] N·m")

    if mode == "pd":
        # Validate before starting the isolated native GUI process.
        PDController(
            kp=kp,
            kd=kd,
            target_position_rad=target_position_rad,
            torque_min_nm=-2.0,
            torque_max_nm=2.0,
        )
        if not -2.094 <= target_position_rad <= 2.094:
            raise ValueError("target must be within [-120.0, 120.0] degrees")
        print_pd_viewer_explanation(math.degrees(target_position_rad), kp, kd)
    else:
        print_viewer_explanation(torque_nm)
    control_port = find_available_port()
    panel_url = f"http://127.0.0.1:{control_port}/"
    environment = os.environ.copy()
    environment[VIEWER_WORKER_ENV] = "1"
    process = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--view",
            "--mode",
            mode,
            "--torque",
            str(torque_nm),
            "--target-deg",
            str(math.degrees(target_position_rad)),
            "--kp",
            str(kp),
            "--kd",
            str(kd),
            "--control-port",
            str(control_port),
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )

    try:
        if wait_for_panel(panel_url):
            print(f"Chinese learning panel: {panel_url}", flush=True)
            if open_browser and not open_panel_in_browser(panel_url):
                print(
                    "Could not open the browser automatically; open the URL above."
                )
        else:
            print(
                "Learning panel did not become ready; the 3D Viewer may still work."
            )
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
        description="Run the single-joint constant-torque or PD experiment."
    )
    parser.add_argument(
        "--mode",
        choices=("torque", "pd"),
        default="torque",
        help="torque=Phase 1 open loop; pd=Phase 2 feedback (default: torque)",
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
        help="headless simulation duration in seconds (default: 1.0; try 3 for PD)",
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
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="do not open the Chinese learning panel in the browser automatically",
    )
    parser.add_argument(
        "--target-deg",
        type=float,
        default=30.0,
        help="PD target angle in degrees; joint range is about [-120, 120]",
    )
    parser.add_argument(
        "--kp",
        type=float,
        default=30.0,
        help="PD proportional gain in N·m/rad (default: 30)",
    )
    parser.add_argument(
        "--kd",
        type=float,
        default=3.0,
        help="PD derivative gain in N·m·s/rad (default: 3)",
    )
    parser.add_argument("--control-port", type=int, default=0, help=argparse.SUPPRESS)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    target_position_rad = math.radians(args.target_deg)

    if args.view and os.environ.get(VIEWER_WORKER_ENV) == "1":
        try:
            final = run_managed_viewer_worker(
                mode=args.mode,
                torque_nm=args.torque,
                target_position_rad=target_position_rad,
                kp=args.kp,
                kd=args.kd,
                control_port=args.control_port,
            )
        except ValueError as exc:
            parser.error(str(exc))
        print(
            "Final state: "
            f"time={final.time_s:.3f} s, "
            f"position={final.position_rad:.6f} rad, "
            f"velocity={final.velocity_rad_s:.6f} rad/s"
        )
        return 0

    if args.mode == "pd":
        print("Single-joint PD closed-loop experiment (Phase 2)")
    else:
        print("Single-joint constant-torque experiment (Phase 1)")
    print(f"model: {MODEL_PATH.relative_to(PROJECT_ROOT)}")

    if args.view:
        try:
            return run_viewer_subprocess(
                mode=args.mode,
                torque_nm=args.torque,
                target_position_rad=target_position_rad,
                kp=args.kp,
                kd=args.kd,
                open_browser=not args.no_browser,
            )
        except ValueError as exc:
            parser.error(str(exc))

    if args.mode == "pd":
        try:
            pd_samples = simulate_pd(
                target_position_rad=target_position_rad,
                kp=args.kp,
                kd=args.kd,
                duration_s=args.duration,
            )
            printed_samples = evenly_spaced_samples(pd_samples, args.samples)
        except ValueError as exc:
            parser.error(str(exc))

        print(
            f"target_deg={args.target_deg:.3f}, Kp={args.kp:g}, Kd={args.kd:g}, "
            "torque_limit=[-2, 2] N·m"
        )
        print(
            "time_s  target_deg  position_deg  velocity_deg_s  "
            "error_deg  torque_nm  limited"
        )
        for sample in printed_samples:
            print(
                f"{sample.time_s:6.3f}  "
                f"{math.degrees(sample.target_position_rad):10.3f}  "
                f"{math.degrees(sample.position_rad):12.3f}  "
                f"{math.degrees(sample.velocity_rad_s):14.3f}  "
                f"{math.degrees(sample.position_error_rad):9.3f}  "
                f"{sample.torque_nm:9.3f}  "
                f"{'yes' if sample.saturated else 'no'}"
            )
        final_pd = pd_samples[-1]
        print(
            "Final tracking error: "
            f"{final_pd.position_error_rad:.6f} rad / "
            f"{math.degrees(final_pd.position_error_rad):.3f} deg"
        )
        return 0

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
