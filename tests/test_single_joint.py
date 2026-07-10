from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import mujoco


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "single_joint" / "single_joint.xml"
RUN_PATH = PROJECT_ROOT / "experiments" / "001_single_joint_pd" / "run.py"


def test_single_joint_model_moves_under_positive_torque() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    assert model.nq == 1
    assert model.nv == 1
    assert model.nu == 1

    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hinge")
    actuator_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, "joint_motor"
    )
    assert joint_id >= 0
    assert actuator_id >= 0
    assert model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE

    data.ctrl[actuator_id] = 0.5
    for _ in range(100):
        mujoco.mj_step(model, data)

    assert data.qpos[model.jnt_qposadr[joint_id]] > 0.0
    assert data.qvel[model.jnt_dofadr[joint_id]] > 0.0


def test_single_joint_experiment_runs_headless() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(RUN_PATH),
            "--duration",
            "0.02",
            "--torque",
            "0.5",
            "--samples",
            "2",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "position_rad" in completed.stdout
    assert "velocity_rad_s" in completed.stdout
    assert "torque_nm" in completed.stdout
    assert "Final state" in completed.stdout
