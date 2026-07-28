"""Small thread-safe switch for gravity feedforward compensation."""

from __future__ import annotations

import threading


class GravityCompensationSwitch:
    """Choose whether the model's gravity/bias torque is fed forward.

    MuJoCo calculates the required bias torque from the complete model. This
    class only owns the live on/off choice shared by the physics callback and
    the browser panel.
    """

    def __init__(self, *, enabled: bool = False) -> None:
        self._lock = threading.Lock()
        self._enabled = bool(enabled)

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    def set_enabled(self, enabled: bool) -> bool:
        if not isinstance(enabled, bool):
            raise ValueError("重力补偿开关必须是 true 或 false")
        with self._lock:
            self._enabled = enabled
            return self._enabled

    def torque(self, model_bias_torque_nm: float) -> float:
        """Return the compensation torque, or zero when disabled."""
        return float(model_bias_torque_nm) if self.enabled else 0.0
