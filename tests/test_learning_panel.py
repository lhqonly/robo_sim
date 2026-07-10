from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

import mujoco
import pytest

from robo_sim.ui.learning_panel import LearningPanelServer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "single_joint" / "single_joint.xml"


def post_json(url: str, payload: dict[str, float]) -> dict[str, object]:
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
        }

        with pytest.raises(urllib.error.HTTPError) as error:
            post_json(panel.url + "api/torque", {"value": 3.0})
        assert error.value.code == 400

        post_json(panel.url + "api/reset", {})
        assert data.ctrl[actuator_id] == pytest.approx(0.0)
        assert data.qpos[model.jnt_qposadr[joint_id]] == pytest.approx(0.0)
    finally:
        panel.close()
