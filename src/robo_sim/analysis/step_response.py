"""Online step-response timing for joint position control."""

from __future__ import annotations

import math
import threading


class StepResponseAnalyzer:
    """Measure a target step from every available physics sample.

    Settling is confirmed only after the position error remains inside the
    configured tolerance for the configured hold time. If the joint later
    leaves that band, the settled result is revoked and timing starts again.
    """

    _TIME_EPSILON_S = 1e-12
    _STEP_EPSILON_RAD = 1e-12

    def __init__(
        self,
        *,
        tolerance_rad: float,
        hold_time_s: float,
        evaluation_window_s: float = 3.0,
    ) -> None:
        self._lock = threading.Lock()
        self._measurement_id = 0
        self._samples: list[
            tuple[
                float,
                float,
                float | None,
                float | None,
                bool | None,
            ]
        ] = []
        self._validate_criteria(tolerance_rad, hold_time_s)
        if (
            not math.isfinite(evaluation_window_s)
            or evaluation_window_s <= 0.0
        ):
            raise ValueError(
                "evaluation window must be finite and greater than zero"
            )
        self._tolerance_rad = float(tolerance_rad)
        self._hold_time_s = float(hold_time_s)
        self._evaluation_window_s = float(evaluation_window_s)
        self._start_time_s = 0.0
        self._initial_position_rad = 0.0
        self._target_position_rad = 0.0
        self._reset_derived()

    def start(
        self,
        *,
        time_s: float,
        position_rad: float,
        target_position_rad: float,
        velocity_rad_s: float | None = None,
        torque_nm: float | None = None,
        saturated: bool | None = None,
    ) -> None:
        """Start a new measurement from the current position to a target."""
        required_values = (time_s, position_rad, target_position_rad)
        optional_values = (velocity_rad_s, torque_nm)
        if not all(math.isfinite(value) for value in required_values) or not all(
            value is None or math.isfinite(value) for value in optional_values
        ):
            raise ValueError("step response start values must be finite")
        with self._lock:
            self._start_locked(
                float(time_s),
                float(position_rad),
                float(target_position_rad),
                None if velocity_rad_s is None else float(velocity_rad_s),
                None if torque_nm is None else float(torque_nm),
                saturated,
            )

    def observe(
        self,
        *,
        time_s: float,
        position_rad: float,
        velocity_rad_s: float | None = None,
        torque_nm: float | None = None,
        saturated: bool | None = None,
    ) -> None:
        """Add one position sample, normally from a MuJoCo physics step."""
        values = (time_s, position_rad, velocity_rad_s, torque_nm)
        if not all(
            value is None or math.isfinite(value) for value in values
        ):
            raise ValueError("step response sample values must be finite")
        with self._lock:
            if not self._samples:
                raise RuntimeError("step response measurement has not started")
            if time_s + self._TIME_EPSILON_S < self._samples[-1][0]:
                # Managed Viewer samples can briefly appear out of order.
                # Never let one stale sample replace an explicit experiment
                # with an accidental "target -> target" measurement. New
                # experiments are started by the panel/controller commands.
                return
            if (
                math.isclose(
                    time_s,
                    self._samples[-1][0],
                    rel_tol=0.0,
                    abs_tol=self._TIME_EPSILON_S,
                )
                and math.isclose(
                    position_rad,
                    self._samples[-1][1],
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
            ):
                previous = self._samples[-1]
                self._samples[-1] = (
                    previous[0],
                    previous[1],
                    previous[2]
                    if velocity_rad_s is None
                    else float(velocity_rad_s),
                    previous[3] if torque_nm is None else float(torque_nm),
                    previous[4] if saturated is None else saturated,
                )
                return
            self._samples.append(
                (
                    float(time_s),
                    float(position_rad),
                    None if velocity_rad_s is None else float(velocity_rad_s),
                    None if torque_nm is None else float(torque_nm),
                    saturated,
                )
            )
            self._process_sample(float(time_s), float(position_rad))

    def configure(self, *, tolerance_rad: float, hold_time_s: float) -> None:
        """Change the settling rule and recalculate the current measurement."""
        self._validate_criteria(tolerance_rad, hold_time_s)
        with self._lock:
            self._tolerance_rad = float(tolerance_rad)
            self._hold_time_s = float(hold_time_s)
            samples = self._samples.copy()
            self._reset_derived()
            for time_s, position_rad, *_ in samples:
                self._process_sample(time_s, position_rad)

    def snapshot(self) -> dict[str, float | int | str | None]:
        """Return a stable copy suitable for CLI output or JSON."""
        with self._lock:
            step_amplitude = abs(
                self._target_position_rad - self._initial_position_rad
            )
            if step_amplitude <= self._STEP_EPSILON_RAD:
                status = "no_step"
            elif self._settling_time_s is not None:
                status = "settled"
            elif self._inside_since_s is not None:
                status = "stabilizing"
            else:
                status = "tracking"
            stable_for = (
                0.0
                if self._inside_since_s is None
                else max(0.0, self._current_time_s - self._inside_since_s)
            )
            rise_time = (
                None
                if self._ten_percent_time_s is None
                or self._ninety_percent_time_s is None
                else self._ninety_percent_time_s - self._ten_percent_time_s
            )
            overshoot_percent = (
                0.0
                if step_amplitude <= self._STEP_EPSILON_RAD
                else self._overshoot_rad / step_amplitude * 100.0
            )
            result: dict[str, float | int | str | None] = {
                "measurement_id": self._measurement_id,
                "status": status,
                "start_time_s": self._start_time_s,
                "elapsed_time_s": max(
                    0.0, self._current_time_s - self._start_time_s
                ),
                "initial_position_rad": self._initial_position_rad,
                "target_position_rad": self._target_position_rad,
                "step_amplitude_rad": step_amplitude,
                "tolerance_rad": self._tolerance_rad,
                "hold_time_s": self._hold_time_s,
                "rise_time_s": rise_time,
                "first_arrival_time_s": self._first_arrival_time_s,
                "settling_time_s": self._settling_time_s,
                "settling_confirmed_time_s": (
                    None
                    if self._settling_time_s is None
                    else self._settling_time_s + self._hold_time_s
                ),
                "stable_for_s": stable_for,
                "overshoot_rad": self._overshoot_rad,
                "overshoot_percent": overshoot_percent,
                "current_error_rad": self._current_error_rad,
            }
            result.update(self._performance_metrics_locked())
            return result

    def _reset_derived(self) -> None:
        self._current_time_s = self._start_time_s
        self._current_error_rad = (
            self._target_position_rad - self._initial_position_rad
        )
        self._ten_percent_time_s: float | None = None
        self._ninety_percent_time_s: float | None = None
        self._first_arrival_time_s: float | None = None
        self._inside_since_s: float | None = None
        self._settling_time_s: float | None = None
        self._overshoot_rad = 0.0

    def _start_locked(
        self,
        time_s: float,
        position_rad: float,
        target_position_rad: float,
        velocity_rad_s: float | None,
        torque_nm: float | None,
        saturated: bool | None,
    ) -> None:
        self._measurement_id += 1
        self._start_time_s = time_s
        self._initial_position_rad = position_rad
        self._target_position_rad = target_position_rad
        self._samples = [
            (
                time_s,
                position_rad,
                velocity_rad_s,
                torque_nm,
                saturated,
            )
        ]
        self._reset_derived()
        self._process_sample(time_s, position_rad)

    def _performance_metrics_locked(self) -> dict[str, float]:
        """Calculate motion intensity and actuator load for this experiment."""
        cutoff_time_s = min(
            self._current_time_s,
            self._start_time_s + self._evaluation_window_s,
        )
        samples = [
            sample
            for sample in self._samples
            if sample[0] <= cutoff_time_s + self._TIME_EPSILON_S
        ]
        if not samples:
            samples = self._samples[:1]

        peak_velocity = 0.0
        peak_acceleration = 0.0
        peak_jerk = 0.0
        peak_torque = 0.0
        integrated_absolute_error = 0.0
        integrated_squared_error = 0.0
        control_effort = 0.0
        absolute_mechanical_work = 0.0
        saturation_time = 0.0
        integrated_time = 0.0
        previous_acceleration: float | None = None
        previous_acceleration_time: float | None = None

        for sample in samples:
            velocity = sample[2]
            torque = sample[3]
            if velocity is not None:
                peak_velocity = max(peak_velocity, abs(velocity))
            if torque is not None:
                peak_torque = max(peak_torque, abs(torque))

        for previous, current in zip(samples, samples[1:]):
            dt = current[0] - previous[0]
            if dt <= self._TIME_EPSILON_S:
                continue
            integrated_time += dt
            previous_error = self._target_position_rad - previous[1]
            current_error = self._target_position_rad - current[1]
            integrated_absolute_error += (
                0.5 * (abs(previous_error) + abs(current_error)) * dt
            )
            integrated_squared_error += (
                0.5
                * (previous_error * previous_error + current_error * current_error)
                * dt
            )

            previous_velocity = previous[2]
            current_velocity = current[2]
            if previous_velocity is None:
                previous_velocity = (current[1] - previous[1]) / dt
            if current_velocity is None:
                current_velocity = (current[1] - previous[1]) / dt
            peak_velocity = max(
                peak_velocity,
                abs(previous_velocity),
                abs(current_velocity),
            )
            acceleration = (current_velocity - previous_velocity) / dt
            peak_acceleration = max(peak_acceleration, abs(acceleration))
            if (
                previous_acceleration is not None
                and previous_acceleration_time is not None
            ):
                acceleration_dt = current[0] - previous_acceleration_time
                if acceleration_dt > self._TIME_EPSILON_S:
                    jerk = (
                        acceleration - previous_acceleration
                    ) / acceleration_dt
                    peak_jerk = max(peak_jerk, abs(jerk))
            previous_acceleration = acceleration
            previous_acceleration_time = current[0]

            previous_torque = previous[3]
            current_torque = current[3]
            if previous_torque is not None and current_torque is not None:
                control_effort += (
                    0.5
                    * (
                        previous_torque * previous_torque
                        + current_torque * current_torque
                    )
                    * dt
                )
                absolute_mechanical_work += (
                    0.5
                    * (
                        abs(previous_torque * previous_velocity)
                        + abs(current_torque * current_velocity)
                    )
                    * dt
                )
            if current[4] is True:
                saturation_time += dt

        tracking_rmse = (
            0.0
            if integrated_time <= self._TIME_EPSILON_S
            else math.sqrt(integrated_squared_error / integrated_time)
        )
        saturation_percent = (
            0.0
            if integrated_time <= self._TIME_EPSILON_S
            else saturation_time / integrated_time * 100.0
        )
        return {
            "evaluation_window_s": integrated_time,
            "evaluation_window_target_s": self._evaluation_window_s,
            "tracking_rmse_rad": tracking_rmse,
            "integrated_absolute_error_rad_s": integrated_absolute_error,
            "peak_velocity_rad_s": peak_velocity,
            "peak_acceleration_rad_s2": peak_acceleration,
            "peak_jerk_rad_s3": peak_jerk,
            "peak_torque_nm": peak_torque,
            "saturation_time_s": saturation_time,
            "saturation_percent": saturation_percent,
            "control_effort_nm2_s": control_effort,
            "absolute_mechanical_work_j": absolute_mechanical_work,
        }

    def _process_sample(self, time_s: float, position_rad: float) -> None:
        self._current_time_s = time_s
        error = self._target_position_rad - position_rad
        self._current_error_rad = error
        signed_step = self._target_position_rad - self._initial_position_rad
        amplitude = abs(signed_step)
        if amplitude <= self._STEP_EPSILON_RAD:
            return

        direction = 1.0 if signed_step > 0.0 else -1.0
        progress = (
            direction * (position_rad - self._initial_position_rad) / amplitude
        )
        elapsed = time_s - self._start_time_s
        if self._ten_percent_time_s is None and progress >= 0.1:
            self._ten_percent_time_s = elapsed
        if self._ninety_percent_time_s is None and progress >= 0.9:
            self._ninety_percent_time_s = elapsed

        overshoot = max(
            0.0, direction * (position_rad - self._target_position_rad)
        )
        self._overshoot_rad = max(self._overshoot_rad, overshoot)

        if abs(error) <= self._tolerance_rad:
            if self._first_arrival_time_s is None:
                self._first_arrival_time_s = elapsed
            if self._inside_since_s is None:
                self._inside_since_s = time_s
            stable_for = time_s - self._inside_since_s
            if stable_for + self._TIME_EPSILON_S >= self._hold_time_s:
                self._settling_time_s = (
                    self._inside_since_s - self._start_time_s
                )
        else:
            self._inside_since_s = None
            self._settling_time_s = None

    @staticmethod
    def _validate_criteria(tolerance_rad: float, hold_time_s: float) -> None:
        if not math.isfinite(tolerance_rad) or tolerance_rad <= 0.0:
            raise ValueError(
                "settling tolerance must be finite and greater than zero"
            )
        if not math.isfinite(hold_time_s) or hold_time_s <= 0.0:
            raise ValueError(
                "settling hold time must be finite and greater than zero"
            )
