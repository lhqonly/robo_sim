#!/usr/bin/env python3
"""Run the Phase 3 knee-like pendulum gravity lesson."""

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

from robo_sim.ui.learning_panel import LearningPanelServer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = (
    PROJECT_ROOT / "models" / "pendulum" / "knee_like_pendulum.xml"
)
VIEWER_WORKER_ENV = "ROBO_SIM_KNEE_VIEWER_WORKER"


@dataclass(frozen=True)
class KneeSample:
    """One full-rate observation from the knee lesson."""

    time_s: float
    angle_rad: float
    velocity_rad_s: float
    required_hold_torque_nm: float
    motor_torque_nm: float
    conceptual_human_torque_nm: float


def load_model(initial_angle_rad: float) -> tuple[
    mujoco.MjModel,
    mujoco.MjData,
    int,
    int,
    int,
    int,
]:
    """Load the model and make the requested angle its reset posture."""
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "knee_hinge"
    )
    actuator_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, "knee_motor"
    )
    qpos_address = int(model.jnt_qposadr[joint_id])
    qvel_address = int(model.jnt_dofadr[joint_id])
    joint_min, joint_max = model.jnt_range[joint_id]
    if not joint_min <= initial_angle_rad <= joint_max:
        raise ValueError(
            f"initial angle must be within "
            f"[{math.degrees(joint_min):.1f}, "
            f"{math.degrees(joint_max):.1f}] degrees"
        )
    data = mujoco.MjData(model)
    data.qpos[qpos_address] = initial_angle_rad
    mujoco.mj_forward(model, data)
    return (
        model,
        data,
        joint_id,
        actuator_id,
        qpos_address,
        qvel_address,
    )


def calculate_motor_torque(
    *,
    mode: str,
    required_hold_torque_nm: float,
    control_min_nm: float,
    control_max_nm: float,
) -> float:
    """Return zero for passive motion or the gravity-holding torque."""
    if mode == "passive":
        return 0.0
    if mode != "hold":
        raise ValueError("mode must be 'passive' or 'hold'")
    return min(
        max(required_hold_torque_nm, control_min_nm),
        control_max_nm,
    )


def observe(
    data: mujoco.MjData,
    *,
    qpos_address: int,
    qvel_address: int,
    actuator_id: int,
) -> KneeSample:
    required_torque = float(data.qfrc_bias[qvel_address])
    motor_torque = float(data.ctrl[actuator_id])
    return KneeSample(
        time_s=float(data.time),
        angle_rad=float(data.qpos[qpos_address]),
        velocity_rad_s=float(data.qvel[qvel_address]),
        required_hold_torque_nm=required_torque,
        motor_torque_nm=motor_torque,
        conceptual_human_torque_nm=required_torque - motor_torque,
    )


def simulate_knee(
    *,
    mode: str,
    initial_angle_rad: float,
    duration_s: float,
) -> list[KneeSample]:
    """Simulate passive falling or active gravity holding."""
    if duration_s <= 0.0:
        raise ValueError("duration must be greater than zero")
    (
        model,
        data,
        _,
        actuator_id,
        qpos_address,
        qvel_address,
    ) = load_model(initial_angle_rad)
    control_min, control_max = model.actuator_ctrlrange[actuator_id]

    def update_motor() -> None:
        data.ctrl[actuator_id] = calculate_motor_torque(
            mode=mode,
            required_hold_torque_nm=float(
                data.qfrc_bias[qvel_address]
            ),
            control_min_nm=float(control_min),
            control_max_nm=float(control_max),
        )

    update_motor()
    samples = [
        observe(
            data,
            qpos_address=qpos_address,
            qvel_address=qvel_address,
            actuator_id=actuator_id,
        )
    ]
    step_count = max(1, round(duration_s / model.opt.timestep))
    for _ in range(step_count):
        update_motor()
        mujoco.mj_step(model, data)
        samples.append(
            observe(
                data,
                qpos_address=qpos_address,
                qvel_address=qvel_address,
                actuator_id=actuator_id,
            )
        )
    return samples


def evenly_spaced_samples(
    samples: list[KneeSample], sample_count: int
) -> list[KneeSample]:
    if sample_count < 2:
        raise ValueError("samples must be at least 2")
    last = len(samples) - 1
    return [
        samples[round(index * last / (sample_count - 1))]
        for index in range(sample_count)
    ]


def print_result(
    *,
    mode: str,
    initial_angle_deg: float,
    samples: list[KneeSample],
    sample_count: int,
) -> None:
    print("Knee-like pendulum experiment (Phase 3)")
    print(f"model: {MODEL_PATH.relative_to(PROJECT_ROOT)}")
    print(
        f"mode={mode}, initial_angle={initial_angle_deg:.1f} deg, "
        "shank=3.0 kg, foot=0.8 kg"
    )
    print(
        "time_s  angle_deg  velocity_deg_s  gravity_need_nm  "
        "motor_nm  human_share_nm"
    )
    for sample in evenly_spaced_samples(samples, sample_count):
        print(
            f"{sample.time_s:6.3f}  "
            f"{math.degrees(sample.angle_rad):9.3f}  "
            f"{math.degrees(sample.velocity_rad_s):14.3f}  "
            f"{sample.required_hold_torque_nm:15.3f}  "
            f"{sample.motor_torque_nm:8.3f}  "
            f"{sample.conceptual_human_torque_nm:14.3f}"
        )
    first = samples[0]
    final = samples[-1]
    print(
        f"At {initial_angle_deg:.1f} deg, gravity requires about "
        f"{first.required_hold_torque_nm:.3f} N·m to hold the lower leg."
    )
    if mode == "passive":
        print(
            "The motor contributes 0 N·m, so gravity pulls the shank "
            "toward the natural hanging angle."
        )
    else:
        print(
            "The motor supplies the model's gravity torque, so the "
            "conceptual remaining human share is near 0 N·m."
        )
    print(
        f"Final state: angle={math.degrees(final.angle_rad):.3f} deg, "
        f"velocity={math.degrees(final.velocity_rad_s):.3f} deg/s"
    )


def run_managed_viewer_worker(
    *,
    mode: str,
    initial_angle_rad: float,
    control_port: int,
) -> None:
    (
        model,
        data,
        joint_id,
        actuator_id,
        qpos_address,
        qvel_address,
    ) = load_model(initial_angle_rad)
    control_min, control_max = model.actuator_ctrlrange[actuator_id]

    def knee_callback(
        callback_model: mujoco.MjModel,
        callback_data: mujoco.MjData,
    ) -> None:
        del callback_model
        callback_data.ctrl[actuator_id] = calculate_motor_torque(
            mode=mode,
            required_hold_torque_nm=float(
                callback_data.qfrc_bias[qvel_address]
            ),
            control_min_nm=float(control_min),
            control_max_nm=float(control_max),
        )

    knee_callback(model, data)
    mujoco.set_mjcb_control(knee_callback)
    panel = LearningPanelServer(
        model,
        data,
        joint_id,
        actuator_id,
        port=control_port,
        manual_torque_enabled=False,
        learning_context={
            "kind": "knee_pendulum",
            "experiment_mode": mode,
            "initial_angle_deg": math.degrees(initial_angle_rad),
        },
        reset_position_rad=initial_angle_rad,
    )
    panel.start()
    from mujoco import viewer as mujoco_viewer

    try:
        mujoco_viewer.launch(model, data)
    finally:
        panel.close()
        mujoco.set_mjcb_control(None)
    print(
        "Viewer window closed at "
        f"{math.degrees(data.qpos[qpos_address]):.2f} deg."
    )


def find_available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def wait_for_panel(url: str, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(
                url + "api/state", timeout=0.2
            ) as response:
                response.read()
                return True
        except OSError:
            time.sleep(0.1)
    return False


def open_panel_in_browser(url: str) -> bool:
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
    initial_angle_rad: float,
    open_browser: bool,
) -> int:
    # Validate before starting the isolated native GUI process.
    load_model(initial_angle_rad)
    initial_angle_deg = math.degrees(initial_angle_rad)
    print("\nPhase 3 Viewer:")
    print("  blue upper link = fixed thigh")
    print("  yellow hub      = knee joint / motor")
    print("  orange link     = simplified shank and foot")
    print(f"  initial angle   = {initial_angle_deg:.1f} deg")
    print(
        "  experiment      = "
        + (
            "motor off; observe gravity pulling the shank down"
            if mode == "passive"
            else "motor supplies gravity torque to hold the posture"
        )
    )
    print("Close the GUI or focus this terminal and press Ctrl+C.\n", flush=True)

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
            "--initial-deg",
            str(initial_angle_deg),
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
                    "Could not open the browser automatically; "
                    "open the URL above."
                )
        else:
            print(
                "Learning panel did not become ready; "
                "the 3D Viewer may still work."
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
            f"Viewer process exited abnormally (code {return_code}).",
            file=sys.stderr,
        )
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a passive lower-leg fall with gravity-torque holding."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("passive", "hold"),
        default="passive",
        help="passive=motor off; hold=motor compensates gravity",
    )
    parser.add_argument(
        "--initial-deg",
        type=float,
        default=45.0,
        help="initial knee angle in degrees (default: 45)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="headless simulation duration in seconds (default: 3)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=7,
        help="headless rows to print, including initial/final",
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="open the Viewer and Chinese Phase 3 panel until closed",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="do not open the Chinese learning panel automatically",
    )
    parser.add_argument("--control-port", type=int, default=0, help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    initial_angle_rad = math.radians(args.initial_deg)
    try:
        if args.view:
            if os.environ.get(VIEWER_WORKER_ENV) == "1":
                run_managed_viewer_worker(
                    mode=args.mode,
                    initial_angle_rad=initial_angle_rad,
                    control_port=args.control_port,
                )
                return 0
            return run_viewer_subprocess(
                mode=args.mode,
                initial_angle_rad=initial_angle_rad,
                open_browser=not args.no_browser,
            )
        samples = simulate_knee(
            mode=args.mode,
            initial_angle_rad=initial_angle_rad,
            duration_s=args.duration,
        )
        print_result(
            mode=args.mode,
            initial_angle_deg=args.initial_deg,
            samples=samples,
            sample_count=args.samples,
        )
        return 0
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
