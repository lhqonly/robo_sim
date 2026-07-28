from __future__ import annotations

import pytest

from robo_sim.controllers.pd import PDController


def test_pd_controller_combines_position_and_velocity_feedback() -> None:
    controller = PDController(
        kp=10.0,
        kd=2.0,
        target_position_rad=0.5,
        target_velocity_rad_s=0.0,
        torque_min_nm=-2.0,
        torque_max_nm=2.0,
    )

    output = controller.compute(position_rad=0.4, velocity_rad_s=0.1)

    assert output.position_error_rad == pytest.approx(0.1)
    assert output.velocity_error_rad_s == pytest.approx(-0.1)
    assert output.proportional_torque_nm == pytest.approx(1.0)
    assert output.derivative_torque_nm == pytest.approx(-0.2)
    assert output.raw_torque_nm == pytest.approx(0.8)
    assert output.torque_nm == pytest.approx(0.8)
    assert output.saturated is False


def test_pd_controller_clips_torque_and_allows_live_tuning() -> None:
    controller = PDController(
        kp=30.0,
        kd=3.0,
        target_position_rad=1.0,
        torque_min_nm=-2.0,
        torque_max_nm=2.0,
    )

    assert controller.compute(0.0, 0.0).torque_nm == pytest.approx(2.0)
    assert controller.compute(0.0, 0.0).saturated is True

    controller.update(kp=12.0, kd=1.5, target_position_rad=0.25)
    settings = controller.settings()
    assert settings["kp"] == pytest.approx(12.0)
    assert settings["kd"] == pytest.approx(1.5)
    assert settings["target_position_rad"] == pytest.approx(0.25)


def test_pd_controller_adds_feedforward_before_applying_torque_limit() -> None:
    controller = PDController(
        kp=10.0,
        kd=0.0,
        target_position_rad=0.08,
        torque_min_nm=-2.0,
        torque_max_nm=2.0,
    )

    output = controller.compute(
        position_rad=0.0,
        velocity_rad_s=0.0,
        feedforward_torque_nm=1.5,
    )

    assert output.pd_torque_nm == pytest.approx(0.8)
    assert output.feedforward_torque_nm == pytest.approx(1.5)
    assert output.raw_torque_nm == pytest.approx(2.3)
    assert output.torque_nm == pytest.approx(2.0)
    assert output.saturated is True


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"kp": -1.0}, "Kp"),
        ({"kd": -1.0}, "Kd"),
        ({"torque_min_nm": 1.0, "torque_max_nm": -1.0}, "torque"),
    ],
)
def test_pd_controller_rejects_invalid_settings(
    kwargs: dict[str, float], message: str
) -> None:
    defaults = {
        "kp": 10.0,
        "kd": 2.0,
        "target_position_rad": 0.5,
        "torque_min_nm": -2.0,
        "torque_max_nm": 2.0,
    }
    defaults.update(kwargs)

    with pytest.raises(ValueError, match=message):
        PDController(**defaults)
