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

    def __init__(self, *, tolerance_rad: float, hold_time_s: float) -> None:
        self._lock = threading.Lock()
        self._measurement_id = 0
        self._samples: list[tuple[float, float]] = []
        self._validate_criteria(tolerance_rad, hold_time_s)
        self._tolerance_rad = float(tolerance_rad)
        self._hold_time_s = float(hold_time_s)
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
    ) -> None:
        """Start a new measurement from the current position to a target."""
        values = (time_s, position_rad, target_position_rad)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("step response start values must be finite")
        with self._lock:
            self._start_locked(
                float(time_s), float(position_rad), float(target_position_rad)
            )

    def observe(self, *, time_s: float, position_rad: float) -> None:
        """Add one position sample, normally from a MuJoCo physics step."""
        if not math.isfinite(time_s) or not math.isfinite(position_rad):
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
                return
            self._samples.append((float(time_s), float(position_rad)))
            self._process_sample(float(time_s), float(position_rad))

    def configure(self, *, tolerance_rad: float, hold_time_s: float) -> None:
        """Change the settling rule and recalculate the current measurement."""
        self._validate_criteria(tolerance_rad, hold_time_s)
        with self._lock:
            self._tolerance_rad = float(tolerance_rad)
            self._hold_time_s = float(hold_time_s)
            samples = self._samples.copy()
            self._reset_derived()
            for time_s, position_rad in samples:
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
            return {
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
        self, time_s: float, position_rad: float, target_position_rad: float
    ) -> None:
        self._measurement_id += 1
        self._start_time_s = time_s
        self._initial_position_rad = position_rad
        self._target_position_rad = target_position_rad
        self._samples = [(time_s, position_rad)]
        self._reset_derived()
        self._process_sample(time_s, position_rad)

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
