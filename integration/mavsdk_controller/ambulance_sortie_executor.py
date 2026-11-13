"""Safety-gated Ambulance sortie executor foundation.

This module does not arm, take off, or start Offboard.
Execution is disabled by default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from mavsdk_grpc.offboard import VelocityNedYaw


MAX_RELATIVE_ALTITUDE_M = 4.0


@dataclass(frozen=True)
class ExecutorConfig:
    execution_enabled: bool = False
    max_horizontal_speed_m_s: float = 0.35
    max_relative_altitude_m: float = MAX_RELATIVE_ALTITUDE_M
    arrival_tolerance_m: float = 1.0
    descent_enabled: bool = False
    climb_enabled: bool = False

    def validate(self) -> None:
        values = (
            self.max_horizontal_speed_m_s,
            self.max_relative_altitude_m,
            self.arrival_tolerance_m,
        )
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("executor limits must be finite and positive")
        if self.max_relative_altitude_m > MAX_RELATIVE_ALTITUDE_M:
            raise ValueError("executor altitude limit exceeds the 4 m safety cap")


class AmbulanceSortieExecutor:
    """Convert validated adapter guidance into bounded MAVSDK setpoints."""

    def __init__(self, config: ExecutorConfig | None = None) -> None:
        self.config = config or ExecutorConfig()
        self.config.validate()

    def build_setpoint(self, guidance) -> VelocityNedYaw:
        """Build a setpoint only when execution is explicitly enabled."""
        if not self.config.execution_enabled:
            raise RuntimeError("ambulance_flight_execution_disabled")

        if guidance is None:
            raise ValueError("guidance is required")

        if guidance.state not in {"GOAL", "DETOUR", "HOLD", "DESCEND", "CLIMB"}:
            raise ValueError(f"unsupported guidance state: {guidance.state}")

        values = (
            guidance.north_m_s,
            guidance.east_m_s,
            guidance.down_m_s,
            guidance.yaw_deg,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("guidance setpoint values must be finite")

        if guidance.state == "CLIMB":
            if (not self.config.climb_enabled
                    or guidance.north_m_s != 0 or guidance.east_m_s != 0
                    or not -0.25 <= guidance.down_m_s < 0):
                raise ValueError("invalid_or_disabled_climb_setpoint")
            return VelocityNedYaw(0.0, 0.0, guidance.down_m_s, guidance.yaw_deg)

        if guidance.state == "DESCEND":
            if (not self.config.descent_enabled
                    or math.hypot(guidance.north_m_s, guidance.east_m_s) > 0.080001
                    or not 0 < guidance.down_m_s <= 0.15):
                raise ValueError("invalid_or_disabled_descent_setpoint")
            return VelocityNedYaw(
                guidance.north_m_s, guidance.east_m_s,
                guidance.down_m_s, guidance.yaw_deg,
            )

        if guidance.state == "HOLD":
            return VelocityNedYaw(
                0.0, 0.0, 0.0, guidance.yaw_deg
            )

        if abs(guidance.down_m_s) > 1e-9:
            raise ValueError("horizontal guidance cannot command altitude changes")

        north = guidance.north_m_s
        east = guidance.east_m_s
        speed = math.hypot(north, east)
        limit = self.config.max_horizontal_speed_m_s

        if speed > limit:
            scale = limit / speed
            north *= scale
            east *= scale

        return VelocityNedYaw(
            north,
            east,
            0.0,
            guidance.yaw_deg,
        )
