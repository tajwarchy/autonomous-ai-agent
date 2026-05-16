"""
Circuit breaker — per tool, per agent run.

States per tool:
  CLOSED  — normal operation (default)
  OPEN    — tool disabled after N consecutive failures

Transitions:
  CLOSED → OPEN   : consecutive_failures >= threshold
  OPEN   → CLOSED : does NOT auto-reset within a run
                    (reset only happens between runs via a new instance)

One CircuitBreaker instance is created per agent run.

Usage:
    cb = CircuitBreaker(config)
    allowed, msg = cb.check("wikipedia")
    if not allowed:
        return msg   # inject as Observation
    # ... call the tool
    if success:
        cb.record_success("wikipedia")
    else:
        cb.record_failure("wikipedia", reason="timeout")
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@dataclass
class ToolState:
    consecutive_failures: int = 0
    is_open: bool = False
    open_reason: Optional[str] = None
    total_failures: int = 0
    total_successes: int = 0


class CircuitBreaker:
    """
    Tracks tool health for one agent run.
    Not thread-safe — designed for single-threaded ReAct loop.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or load_config()
        self._threshold: int = cfg["tools"]["circuit_breaker"]["failure_threshold"]
        self._states: dict[str, ToolState] = {}

    def _state(self, tool_name: str) -> ToolState:
        if tool_name not in self._states:
            self._states[tool_name] = ToolState()
        return self._states[tool_name]

    def check(self, tool_name: str) -> tuple[bool, str]:
        """
        Returns (True, "") if the tool is usable.
        Returns (False, error_msg) if the circuit is open.
        """
        state = self._state(tool_name)
        if state.is_open:
            msg = (
                f"ERROR: tool '{tool_name}' is currently unavailable "
                f"(circuit breaker open after {self._threshold} consecutive failures"
                + (f": {state.open_reason}" if state.open_reason else "")
                + "). Try a different tool or finish with what you know."
            )
            return False, msg
        return True, ""

    def record_success(self, tool_name: str) -> None:
        """Reset consecutive failure count on success."""
        state = self._state(tool_name)
        state.consecutive_failures = 0
        state.total_successes += 1

    def record_failure(self, tool_name: str, reason: str = "") -> None:
        """
        Increment failure count. Open the circuit if threshold is reached.
        """
        state = self._state(tool_name)
        state.consecutive_failures += 1
        state.total_failures += 1

        logger.warning(
            "circuit_breaker_failure",
            extra={
                "tool": tool_name,
                "consecutive": state.consecutive_failures,
                "threshold": self._threshold,
                "reason": reason,
            },
        )

        if state.consecutive_failures >= self._threshold:
            state.is_open = True
            state.open_reason = reason
            logger.error(
                "circuit_breaker_open",
                extra={"tool": tool_name, "reason": reason},
            )

    def is_open(self, tool_name: str) -> bool:
        return self._state(tool_name).is_open

    def summary(self) -> dict[str, dict]:
        """Return state summary for run logging."""
        return {
            tool: {
                "is_open": s.is_open,
                "consecutive_failures": s.consecutive_failures,
                "total_failures": s.total_failures,
                "total_successes": s.total_successes,
                "open_reason": s.open_reason,
            }
            for tool, s in self._states.items()
        }