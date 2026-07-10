from __future__ import annotations

import mujoco

import robo_sim


def test_package_version_is_exposed() -> None:
    assert robo_sim.__version__ == "0.1.0"


def test_mujoco_can_compile_and_step_minimal_model() -> None:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco model="test">
          <worldbody>
            <body>
              <joint type="hinge"/>
              <geom type="sphere" size="0.05"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    assert model.nq == 1
    assert model.nv == 1
