from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
from pathlib import Path

import mujoco
import pytest

from robo_sim.ui.learning_panel import LearningPanelServer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = (
    PROJECT_ROOT / "models" / "pendulum" / "knee_like_pendulum.xml"
)
RUN_PATH = (
    PROJECT_ROOT / "experiments" / "002_pendulum_balance" / "run.py"
)


def load_experiment_module():
    spec = importlib.util.spec_from_file_location(
        "knee_pendulum_run", RUN_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_knee_model_has_explicit_thigh_shank_foot_and_motor() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    assert model.nq == 1
    assert model.nv == 1
    assert model.nu == 1
    for object_type, name in [
        (mujoco.mjtObj.mjOBJ_BODY, "fixed_thigh"),
        (mujoco.mjtObj.mjOBJ_BODY, "shank_and_foot"),
        (mujoco.mjtObj.mjOBJ_GEOM, "shank_geom"),
        (mujoco.mjtObj.mjOBJ_GEOM, "foot_geom"),
        (mujoco.mjtObj.mjOBJ_JOINT, "knee_hinge"),
        (mujoco.mjtObj.mjOBJ_ACTUATOR, "knee_motor"),
    ]:
        assert mujoco.mj_name2id(model, object_type, name) >= 0

    actuator_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, "knee_motor"
    )
    assert model.actuator_ctrlrange[actuator_id] == pytest.approx(
        [-10.0, 10.0]
    )


def test_passive_knee_falls_but_gravity_hold_keeps_initial_angle() -> None:
    experiment = load_experiment_module()
    initial_angle = math.radians(45.0)
    passive = experiment.simulate_knee(
        mode="passive",
        initial_angle_rad=initial_angle,
        duration_s=3.0,
    )
    held = experiment.simulate_knee(
        mode="hold",
        initial_angle_rad=initial_angle,
        duration_s=3.0,
    )

    assert passive[0].required_hold_torque_nm == pytest.approx(
        6.416, abs=0.01
    )
    assert abs(math.degrees(passive[-1].angle_rad)) < 2.0
    assert passive[0].motor_torque_nm == pytest.approx(0.0)
    assert passive[0].conceptual_human_torque_nm == pytest.approx(
        passive[0].required_hold_torque_nm
    )

    assert math.degrees(held[-1].angle_rad) == pytest.approx(45.0, abs=1e-6)
    assert held[0].motor_torque_nm == pytest.approx(
        held[0].required_hold_torque_nm
    )
    assert held[-1].conceptual_human_torque_nm == pytest.approx(
        0.0, abs=1e-9
    )


def test_knee_panel_exposes_gravity_motor_and_human_share() -> None:
    experiment = load_experiment_module()
    initial_angle = math.radians(45.0)
    (
        model,
        data,
        joint_id,
        actuator_id,
        qpos_address,
        _,
    ) = experiment.load_model(initial_angle)
    panel = LearningPanelServer(
        model,
        data,
        joint_id,
        actuator_id,
        manual_torque_enabled=False,
        learning_context={
            "kind": "knee_pendulum",
            "experiment_mode": "passive",
            "initial_angle_deg": 45.0,
        },
        reset_position_rad=initial_angle,
    )
    panel.start()
    try:
        state = panel.snapshot()
        assert state["learning_context"]["kind"] == "knee_pendulum"
        assert state["manual_torque_enabled"] is False
        assert state["required_hold_torque_nm"] == pytest.approx(
            6.416, abs=0.01
        )
        assert state["conceptual_human_torque_nm"] == pytest.approx(
            state["required_hold_torque_nm"]
        )

        data.qpos[qpos_address] = 0.0
        panel.reset()
        assert data.qpos[qpos_address] == pytest.approx(initial_angle)
        with pytest.raises(ValueError, match="实验模式自动控制"):
            panel.set_torque(1.0)
    finally:
        panel.close()


def test_knee_experiment_runs_headless() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(RUN_PATH),
            "--mode",
            "hold",
            "--initial-deg",
            "45",
            "--duration",
            "0.02",
            "--samples",
            "2",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Knee-like pendulum experiment (Phase 3)" in completed.stdout
    assert "gravity_need_nm" in completed.stdout
    assert "human_share_nm" in completed.stdout
