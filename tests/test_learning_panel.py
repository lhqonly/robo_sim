from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from pathlib import Path

import mujoco
import pytest

from robo_sim.analysis.step_response import StepResponseAnalyzer
from robo_sim.controllers.gravity import GravityCompensationSwitch
from robo_sim.controllers.pd import PDController
from robo_sim.ui.learning_panel import LearningPanelServer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "single_joint" / "single_joint.xml"


def post_json(url: str, payload: dict[str, object]) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=2) as response:
        return json.load(response)


def test_learning_panel_serves_chinese_watch_fields_and_exact_torque() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hinge")
    actuator_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, "joint_motor"
    )
    panel = LearningPanelServer(model, data, joint_id, actuator_id)
    panel.start()

    try:
        with urllib.request.urlopen(panel.url, timeout=2) as response:
            html = response.read().decode("utf-8")
        assert "Watch Field（观察变量）" in html
        assert "qpos — 关节角度" in html
        assert "rad / degree" in html
        assert "format(latest.qpos_deg, 2)" in html
        assert "degree/s" in html
        assert "joint_motor 精确扭矩输入" in html

        result = post_json(panel.url + "api/torque", {"value": 1.234})
        assert result["ctrl"] == pytest.approx(1.234)
        assert data.ctrl[actuator_id] == pytest.approx(1.234)

        with urllib.request.urlopen(panel.url + "api/state", timeout=2) as response:
            state = json.load(response)
        assert state["ctrl"] == pytest.approx(1.234)
        assert set(state) == {
            "time",
            "qpos",
            "qpos_deg",
            "qvel",
            "ctrl",
            "actuator_force",
            "mode",
        }
        assert state["mode"] == "torque"

        with pytest.raises(urllib.error.HTTPError) as error:
            post_json(panel.url + "api/torque", {"value": 3.0})
        assert error.value.code == 400

        post_json(panel.url + "api/reset", {})
        assert data.ctrl[actuator_id] == pytest.approx(0.0)
        assert data.qpos[model.jnt_qposadr[joint_id]] == pytest.approx(0.0)
    finally:
        panel.close()


def test_learning_panel_supports_exact_pd_tuning() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hinge")
    actuator_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, "joint_motor"
    )
    controller = PDController(
        kp=30.0,
        kd=3.0,
        target_position_rad=0.5,
        torque_min_nm=-2.0,
        torque_max_nm=2.0,
    )
    gravity_compensation = GravityCompensationSwitch(enabled=False)
    response_analyzer = StepResponseAnalyzer(
        tolerance_rad=math.radians(1.0),
        hold_time_s=0.5,
    )
    response_analyzer.start(
        time_s=0.0,
        position_rad=0.0,
        target_position_rad=0.5,
    )
    response_analyzer.observe(time_s=1.0, position_rad=0.2)
    measurement_id_before_panel_read = response_analyzer.snapshot()[
        "measurement_id"
    ]
    panel = LearningPanelServer(
        model,
        data,
        joint_id,
        actuator_id,
        pd_controller=controller,
        gravity_compensation=gravity_compensation,
        step_response_analyzer=response_analyzer,
    )
    panel.start()

    try:
        with urllib.request.urlopen(panel.url, timeout=2) as response:
            html = response.read().decode("utf-8")
        assert "PD 闭环参数" in html
        assert "目标角度" in html
        assert "比例增益 Kp" in html
        assert "实时响应曲线" in html
        assert 'id="angleChart"' in html
        assert "为什么停在这里，没有到目标" in html
        assert "PD 是 PID 里的 P + D" in html
        assert "重力补偿" in html
        assert 'id="toggleGravityCompensation"' in html
        assert "响应时间测量" in html
        assert 'id="settlingTime"' in html
        assert "连续保持" in html
        assert "应用参数并从 0° 开始测试" in html
        assert 'class="experiment-grid"' in html
        assert "请在 MuJoCo Viewer 左侧 Simulation 区域点击 Run" in html
        assert "[hidden] { display: none !important; }" in html
        assert '<section id="watchControl"' in html
        assert '<section id="torqueControl" class="card" hidden>' in html

        with urllib.request.urlopen(panel.url + "api/state", timeout=2) as response:
            initial_state = json.load(response)
        assert (
            initial_state["step_response"]["measurement_id"]
            == measurement_id_before_panel_read
        )
        assert initial_state["step_response"]["elapsed_time_s"] == pytest.approx(
            1.0
        )

        result = post_json(
            panel.url + "api/pd",
            {"target_deg": 25.0, "kp": 24.0, "kd": 2.5},
        )
        assert result["target_position_deg"] == pytest.approx(25.0)
        assert result["kp"] == pytest.approx(24.0)
        assert result["kd"] == pytest.approx(2.5)

        with urllib.request.urlopen(panel.url + "api/state", timeout=2) as response:
            state = json.load(response)
        assert state["mode"] == "pd"
        assert state["target_position_deg"] == pytest.approx(25.0)
        assert state["kp"] == pytest.approx(24.0)
        assert state["kd"] == pytest.approx(2.5)
        assert state["target_hold_torque_nm"] == pytest.approx(
            1.0 * 9.81 * 0.25 * 0.4226182617, rel=1e-5
        )
        assert state["proportional_torque_nm"] == pytest.approx(
            24.0 * state["position_error_rad"]
        )
        assert "derivative_torque_nm" in state
        assert "bias_torque_nm" in state
        assert state["gravity_compensation_enabled"] is False
        assert state["gravity_compensation_torque_nm"] == pytest.approx(0.0)
        response = state["step_response"]
        assert response["status"] == "tracking"
        assert response["tolerance_deg"] == pytest.approx(1.0)
        assert response["hold_time_s"] == pytest.approx(0.5)
        assert response["target_position_deg"] == pytest.approx(25.0)
        measurement_id_after_target_change = response["measurement_id"]

        post_json(
            panel.url + "api/pd",
            {"target_deg": 25.0, "kp": 30.0, "kd": 3.0},
        )
        with urllib.request.urlopen(panel.url + "api/state", timeout=2) as response:
            same_target_state = json.load(response)
        assert (
            same_target_state["step_response"]["measurement_id"]
            == measurement_id_after_target_change
        )

        criteria = post_json(
            panel.url + "api/response-criteria",
            {"tolerance_deg": 0.5, "hold_time_s": 0.25},
        )
        assert criteria["tolerance_deg"] == pytest.approx(0.5)
        assert criteria["hold_time_s"] == pytest.approx(0.25)

        result = post_json(
            panel.url + "api/gravity-compensation", {"enabled": True}
        )
        assert result["gravity_compensation_enabled"] is True
        assert gravity_compensation.enabled is True

        with urllib.request.urlopen(panel.url + "api/state", timeout=2) as response:
            compensated_state = json.load(response)
        assert compensated_state["gravity_compensation_enabled"] is True
        assert (
            compensated_state["step_response"]["measurement_id"]
            == measurement_id_after_target_change
        )
        assert compensated_state["gravity_compensation_torque_nm"] == pytest.approx(
            compensated_state["bias_torque_nm"]
        )
        assert compensated_state["raw_torque_nm"] == pytest.approx(
            compensated_state["pd_torque_nm"]
            + compensated_state["gravity_compensation_torque_nm"]
        )

        measurement_id = compensated_state["step_response"]["measurement_id"]
        post_json(panel.url + "api/reset", {})
        with urllib.request.urlopen(panel.url + "api/state", timeout=2) as response:
            reset_state = json.load(response)
        assert reset_state["step_response"]["measurement_id"] > measurement_id
        assert reset_state["step_response"]["initial_position_deg"] == pytest.approx(
            0.0
        )
        assert reset_state["step_response"]["target_position_deg"] == pytest.approx(
            25.0
        )

        response_test = post_json(
            panel.url + "api/run-response-test",
            {
                "target_deg": 45.0,
                "kp": 10.0,
                "kd": 3.0,
                "tolerance_deg": 0.75,
                "hold_time_s": 0.4,
            },
        )
        assert response_test["target_position_deg"] == pytest.approx(45.0)
        assert response_test["tolerance_deg"] == pytest.approx(0.75)
        assert response_test["hold_time_s"] == pytest.approx(0.4)
        assert data.qpos[model.jnt_qposadr[joint_id]] == pytest.approx(0.0)
        with urllib.request.urlopen(panel.url + "api/state", timeout=2) as response:
            restarted_state = json.load(response)
        assert restarted_state["step_response"]["initial_position_deg"] == pytest.approx(
            0.0
        )
        assert restarted_state["step_response"]["target_position_deg"] == pytest.approx(
            45.0
        )
        assert restarted_state["step_response"]["status"] == "tracking"

        with pytest.raises(urllib.error.HTTPError) as error:
            post_json(panel.url + "api/torque", {"value": 1.0})
        assert error.value.code == 400
    finally:
        panel.close()
