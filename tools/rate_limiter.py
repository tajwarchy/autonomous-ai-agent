"""
Per-tool rate limiter — scoped to a single agent run.

One RateLimiter instance is created per agent run and discarded afterward.
Limits are read from config.yaml (tools.rate_limits).

Usage:
    limiter = RateLimiter(config)
    allowed, msg = limiter.check("search")
    if not allowed:
        return msg   # inject as Observation
    # ... call the tool
    limiter.increment("search")
"""

import logging
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class RateLimiter:
    """
    Tracks call counts per tool for one agent run.
    Not thread-safe — designed for single-threaded ReAct loop.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or load_config()
        self._limits: dict[str, int] = cfg["tools"]["rate_limits"]
        self._counts: dict[str, int] = {tool: 0 for tool in self._limits}

    def check(self, tool_name: str) -> tuple[bool, str]:
        """
        Check whether a tool call is allowed.

        Returns:
            (True, "")           — call is allowed
            (False, error_msg)   — limit reached, inject as Observation
        """
        limit = self._limits.get(tool_name)
        if limit is None:
            # Tool has no configured limit — allow unconditionally
            return True, ""

        count = self._counts.get(tool_name, 0)
        if count >= limit:
            msg = (
                f"ERROR: tool '{tool_name}' rate limit reached "
                f"({count}/{limit} calls used this run). "
                "Try a different tool or finish with what you know."
            )
            logger.info(
                "rate_limit_hit",
                extra={"tool": tool_name, "count": count, "limit": limit},
            )
            return False, msg

        return True, ""

    def increment(self, tool_name: str) -> None:
        """Record a completed call. Call this after the tool returns."""
        if tool_name in self._counts:
            self._counts[tool_name] += 1
        else:
            self._counts[tool_name] = 1

    def usage(self) -> dict[str, dict]:
        """Return current usage stats — used for run logging."""
        return {
            tool: {
                "calls": self._counts.get(tool, 0),
                "limit": self._limits.get(tool, None),
            }
            for tool in self._limits
        }