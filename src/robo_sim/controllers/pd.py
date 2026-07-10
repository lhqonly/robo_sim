"""Reusable proportional-derivative (PD) joint controller."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class PDOutput:
    """One PD calculation, including terms useful for learning and plotting."""

    position_error_rad: float
    velocity_error_rad_s: float
    proportional_torque_nm: float
    derivative_torque_nm: float
    raw_torque_nm: float
    torque_nm: float
    saturated: bool


class PDController:
    """Compute a bounded torque from joint position and velocity feedback.

    The parameters can be updated while a simulation is running. A small lock
    keeps the Viewer physics thread and the browser learning panel consistent.
    """

    def __init__(
        self,
        *,
        kp: float,
        kd: float,
        target_position_rad: float,
        target_velocity_rad_s: float = 0.0,
        torque_min_nm: float = -math.inf,
        torque_max_nm: float = math.inf,
    ) -> None:
        self._lock = threading.Lock()
        self._kp = 0.0
        self._kd = 0.0
        self._target_position_rad = 0.0
        self._target_velocity_rad_s = 0.0
        self._torque_min_nm = 0.0
        self._torque_max_nm = 0.0
        self._validate(
            kp,
            kd,
            target_position_rad,
            target_velocity_rad_s,
            torque_min_nm,
            torque_max_nm,
        )
        self._kp = float(kp)
        self._kd = float(kd)
        self._target_position_rad = float(target_position_rad)
        self._target_velocity_rad_s = float(target_velocity_rad_s)
        self._torque_min_nm = float(torque_min_nm)
        self._torque_max_nm = float(torque_max_nm)

    def compute(self, position_rad: float, velocity_rad_s: float) -> PDOutput:
        """Return the raw and actuator-limited torque for the current state."""
        with self._lock:
            kp = self._kp
            kd = self._kd
            target_position = self._target_position_rad
            target_velocity = self._target_velocity_rad_s
            torque_min = self._torque_min_nm
            torque_max = self._torque_max_nm

        position_error = target_position - float(position_rad)
        velocity_error = target_velocity - float(velocity_rad_s)
        proportional = kp * position_error
        derivative = kd * velocity_error
        raw_torque = proportional + derivative
        torque = min(max(raw_torque, torque_min), torque_max)
        return PDOutput(
            position_error_rad=position_error,
            velocity_error_rad_s=velocity_error,
            proportional_torque_nm=proportional,
            derivative_torque_nm=derivative,
            raw_torque_nm=raw_torque,
            torque_nm=torque,
            saturated=not math.isclose(torque, raw_torque, rel_tol=0.0, abs_tol=1e-12),
        )

    def update(
        self,
        *,
        kp: float | None = None,
        kd: float | None = None,
        target_position_rad: float | None = None,
        target_velocity_rad_s: float | None = None,
    ) -> None:
        """Update gains or targets without changing the actuator limits."""
        with self._lock:
            new_kp = self._kp if kp is None else float(kp)
            new_kd = self._kd if kd is None else float(kd)
            new_target_position = (
                self._target_position_rad
                if target_position_rad is None
                else float(target_position_rad)
            )
            new_target_velocity = (
                self._target_velocity_rad_s
                if target_velocity_rad_s is None
                else float(target_velocity_rad_s)
            )
            self._validate(
                new_kp,
                new_kd,
                new_target_position,
                new_target_velocity,
                self._torque_min_nm,
                self._torque_max_nm,
            )
            self._kp = new_kp
            self._kd = new_kd
            self._target_position_rad = new_target_position
            self._target_velocity_rad_s = new_target_velocity

    def settings(self) -> dict[str, float]:
        """Return a stable copy of the current settings."""
        with self._lock:
            return {
                "kp": self._kp,
                "kd": self._kd,
                "target_position_rad": self._target_position_rad,
                "target_velocity_rad_s": self._target_velocity_rad_s,
                "torque_min_nm": self._torque_min_nm,
                "torque_max_nm": self._torque_max_nm,
            }

    @staticmethod
    def _validate(
        kp: float,
        kd: float,
        target_position_rad: float,
        target_velocity_rad_s: float,
        torque_min_nm: float,
        torque_max_nm: float,
    ) -> None:
        finite_values = (kp, kd, target_position_rad, target_velocity_rad_s)
        if not all(math.isfinite(value) for value in finite_values):
            raise ValueError("PD settings must be finite")
        if kp < 0:
            raise ValueError("Kp must be greater than or equal to zero")
        if kd < 0:
            raise ValueError("Kd must be greater than or equal to zero")
        if torque_min_nm >= torque_max_nm:
            raise ValueError("torque minimum must be less than torque maximum")
