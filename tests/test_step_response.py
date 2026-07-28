from __future__ import annotations

import math

import pytest

from robo_sim.analysis.step_response import StepResponseAnalyzer


def test_step_response_measures_rise_arrival_settling_and_overshoot() -> None:
    analyzer = StepResponseAnalyzer(
        tolerance_rad=0.05,
        hold_time_s=0.2,
    )
    analyzer.start(time_s=0.0, position_rad=0.0, target_position_rad=1.0)

    for time_s, position_rad in [
        (0.1, 0.10),
        (0.5, 0.90),
        (0.6, 1.10),
        (0.7, 0.99),
        (0.9, 0.99),
    ]:
        analyzer.observe(time_s=time_s, position_rad=position_rad)

    result = analyzer.snapshot()
    assert result["status"] == "settled"
    assert result["rise_time_s"] == pytest.approx(0.4)
    assert result["first_arrival_time_s"] == pytest.approx(0.7)
    assert result["settling_time_s"] == pytest.approx(0.7)
    assert result["settling_confirmed_time_s"] == pytest.approx(0.9)
    assert result["overshoot_rad"] == pytest.approx(0.1)
    assert result["overshoot_percent"] == pytest.approx(10.0)
    assert result["current_error_rad"] == pytest.approx(0.01)


def test_step_response_revokes_settled_state_if_joint_leaves_band() -> None:
    analyzer = StepResponseAnalyzer(tolerance_rad=0.05, hold_time_s=0.2)
    analyzer.start(time_s=0.0, position_rad=0.0, target_position_rad=1.0)

    analyzer.observe(time_s=0.5, position_rad=0.98)
    analyzer.observe(time_s=0.7, position_rad=1.00)
    assert analyzer.snapshot()["status"] == "settled"

    analyzer.observe(time_s=0.8, position_rad=0.90)
    result = analyzer.snapshot()
    assert result["status"] == "tracking"
    assert result["settling_time_s"] is None

    analyzer.observe(time_s=1.0, position_rad=0.99)
    analyzer.observe(time_s=1.2, position_rad=1.00)
    result = analyzer.snapshot()
    assert result["status"] == "settled"
    assert result["settling_time_s"] == pytest.approx(1.0)


def test_step_response_supports_negative_steps_and_replays_new_criteria() -> None:
    analyzer = StepResponseAnalyzer(tolerance_rad=0.05, hold_time_s=0.2)
    analyzer.start(time_s=0.0, position_rad=0.0, target_position_rad=-1.0)

    for time_s, position_rad in [
        (0.1, -0.10),
        (0.4, -0.90),
        (0.5, -1.20),
        (0.6, -0.98),
        (0.8, -0.98),
    ]:
        analyzer.observe(time_s=time_s, position_rad=position_rad)

    result = analyzer.snapshot()
    assert result["status"] == "settled"
    assert result["overshoot_rad"] == pytest.approx(0.2)
    assert result["overshoot_percent"] == pytest.approx(20.0)

    analyzer.configure(tolerance_rad=0.01, hold_time_s=0.2)
    result = analyzer.snapshot()
    assert result["status"] == "tracking"
    assert result["settling_time_s"] is None
    assert result["tolerance_rad"] == pytest.approx(0.01)


def test_step_response_reports_no_step_and_rejects_invalid_criteria() -> None:
    analyzer = StepResponseAnalyzer(
        tolerance_rad=math.radians(1.0),
        hold_time_s=0.5,
    )
    analyzer.start(time_s=2.0, position_rad=0.5, target_position_rad=0.5)

    result = analyzer.snapshot()
    assert result["status"] == "no_step"
    assert result["rise_time_s"] is None
    assert result["settling_time_s"] is None

    with pytest.raises(ValueError, match="tolerance"):
        analyzer.configure(tolerance_rad=0.0, hold_time_s=0.5)
    with pytest.raises(ValueError, match="hold"):
        analyzer.configure(tolerance_rad=0.1, hold_time_s=0.0)


def test_backward_time_sample_does_not_restart_explicit_measurement() -> None:
    analyzer = StepResponseAnalyzer(tolerance_rad=0.05, hold_time_s=0.2)
    analyzer.start(time_s=0.0, position_rad=0.0, target_position_rad=1.0)
    measurement_id = analyzer.snapshot()["measurement_id"]

    analyzer.observe(time_s=0.5, position_rad=0.5)
    # Reproduce an out-of-order managed Viewer sample arriving after a newer
    # one. Its position is already at the target, which used to overwrite the
    # valid 0 -> 1 experiment with an invalid 1 -> 1 experiment.
    analyzer.observe(time_s=0.4, position_rad=1.0)

    result = analyzer.snapshot()
    assert result["measurement_id"] == measurement_id
    assert result["initial_position_rad"] == pytest.approx(0.0)
    assert result["target_position_rad"] == pytest.approx(1.0)
    assert result["status"] == "tracking"
