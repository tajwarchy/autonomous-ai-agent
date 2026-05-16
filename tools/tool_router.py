"""
Tool router — dispatches agent actions to the correct tool.

Safety chain applied on every call (in order):
  1. Circuit breaker check  — skip if tool is OPEN
  2. Rate limiter check     — skip if call limit reached for this run
  3. Tool execution         — wrapped in a timeout thread
  4. Record outcome         — updates circuit breaker and rate limiter

The router returns a plain string in all cases.
The agent loop never needs to handle exceptions from here.
"""

import logging
import concurrent.futures
import time
from typing import Optional

import yaml

from tools.base import BaseTool
from tools.calculator import CalculatorTool
from tools.circuit_breaker import CircuitBreaker
from tools.file_reader import FileReaderTool
from tools.rate_limiter import RateLimiter
from tools.search import SearchTool
from tools.wikipedia import WikipediaTool

logger = logging.getLogger(__name__)


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class ToolRouter:
    """
    Created once per agent run.
    Holds the rate limiter and circuit breaker for that run.
    """

    def __init__(self, config: Optional[dict] = None):
        self._cfg = config or load_config()
        self._timeout = self._cfg["tools"]["timeout_seconds"]

        # Instantiate all tools
        self._tools: dict[str, BaseTool] = {
            t.name: t for t in [
                SearchTool(self._cfg),
                CalculatorTool(self._cfg),
                FileReaderTool(self._cfg),
                WikipediaTool(self._cfg),
            ]
        }

        # Per-run safety mechanisms
        self.rate_limiter = RateLimiter(self._cfg)
        self.circuit_breaker = CircuitBreaker(self._cfg)

    def dispatch(self, tool_name: str, input_str: str) -> tuple[str, float]:
        """
        Route an action to the appropriate tool.

        Returns:
            (observation, latency_seconds)
            observation is always a plain string — error messages included.
        """
        start = time.monotonic()

        # ── 1. Circuit breaker ─────────────────────────────────────────
        allowed, msg = self.circuit_breaker.check(tool_name)
        if not allowed:
            latency = time.monotonic() - start
            logger.info(
                "tool_blocked_circuit_breaker",
                extra={"tool": tool_name, "latency_s": round(latency, 3)},
            )
            return msg, latency

        # ── 2. Rate limiter ────────────────────────────────────────────
        allowed, msg = self.rate_limiter.check(tool_name)
        if not allowed:
            latency = time.monotonic() - start
            logger.info(
                "tool_blocked_rate_limit",
                extra={"tool": tool_name, "latency_s": round(latency, 3)},
            )
            return msg, latency

        # ── 3. Tool lookup ─────────────────────────────────────────────
        tool = self._tools.get(tool_name)
        if tool is None:
            latency = time.monotonic() - start
            return (
                f"ERROR: unknown tool '{tool_name}'. "
                f"Available: {sorted(self._tools.keys())}",
                latency,
            )

        # ── 4. Execute with timeout ────────────────────────────────────
        observation, success = self._run_with_timeout(tool, input_str)
        latency = time.monotonic() - start

        # ── 5. Update safety mechanisms ────────────────────────────────
        self.rate_limiter.increment(tool_name)

        if success:
            self.circuit_breaker.record_success(tool_name)
        else:
            self.circuit_breaker.record_failure(tool_name, reason=observation[:120])

        logger.info(
            "tool_call",
            extra={
                "tool": tool_name,
                "input_preview": input_str[:80],
                "success": success,
                "latency_s": round(latency, 3),
                "observation_chars": len(observation),
            },
        )

        return observation, latency

    def _run_with_timeout(self, tool: BaseTool, input_str: str) -> tuple[str, bool]:
        """
        Run tool.run() in a thread with a hard timeout.

        Returns:
            (result_string, is_success)
            is_success=False when the result starts with "ERROR:"
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(tool.run, input_str)
            try:
                result = future.result(timeout=self._timeout)
                is_success = not result.startswith("ERROR:")
                return result, is_success
            except concurrent.futures.TimeoutError:
                future.cancel()
                msg = (
                    f"ERROR: tool '{tool.name}' timed out after {self._timeout}s."
                )
                logger.warning(
                    "tool_timeout",
                    extra={"tool": tool.name, "timeout_s": self._timeout},
                )
                return msg, False
            except Exception as e:
                msg = f"ERROR: tool '{tool.name}' raised an unexpected error — {e}"
                logger.error(
                    "tool_unexpected_error",
                    extra={"tool": tool.name, "error": str(e)},
                )
                return msg, False

    def tool_names(self) -> list[str]:
        return list(self._tools.keys())