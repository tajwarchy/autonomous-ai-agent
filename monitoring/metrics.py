"""
Prometheus metrics for the agent system.

All metrics are module-level singletons — import and use anywhere.
The FastAPI app exposes them at GET /metrics via prometheus-client's
built-in WSGI app.

Metric types used:
  Counter   — monotonically increasing (calls, failures, trips)
  Histogram — measures distributions (latency)
  Gauge     — point-in-time value (active runs)

One real example of each (for README / interview):
  Metric  → tool_call_total{tool="calculator"} 42
  Log     → {"ts": "...", "level": "INFO", "msg": "tool_call", "tool": "calculator", "latency_s": 0.003}
  Trace   → run_id=abc, step 1: thought→action→observation, step 2: ...
"""

from prometheus_client import Counter, Gauge, Histogram

# ── Tool-level metrics ─────────────────────────────────────────────────────

TOOL_CALL_TOTAL = Counter(
    "tool_call_total",
    "Total number of tool calls dispatched",
    ["tool"],
)

TOOL_FAILURE_TOTAL = Counter(
    "tool_failure_total",
    "Total number of tool calls that returned an ERROR result",
    ["tool"],
)

TOOL_LATENCY_SECONDS = Histogram(
    "tool_latency_seconds",
    "Tool call latency in seconds",
    ["tool"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)

CIRCUIT_BREAKER_TRIPS_TOTAL = Counter(
    "circuit_breaker_trips_total",
    "Number of times the circuit breaker opened for a tool",
    ["tool"],
)

RATE_LIMIT_HITS_TOTAL = Counter(
    "rate_limit_hits_total",
    "Number of times a tool call was blocked by the rate limiter",
    ["tool"],
)

# ── Agent-level metrics ────────────────────────────────────────────────────

AGENT_RUN_TOTAL = Counter(
    "agent_run_total",
    "Total agent runs completed",
    ["outcome"],   # DONE | FAILED
)

AGENT_RUN_LATENCY_SECONDS = Histogram(
    "agent_run_latency_seconds",
    "End-to-end agent run latency in seconds",
    buckets=[1, 5, 10, 30, 60, 120, 300],
)

AGENT_STEPS_HISTOGRAM = Histogram(
    "agent_steps_total",
    "Number of ReAct iterations per agent run",
    buckets=[1, 2, 3, 5, 7, 10],
)

AGENT_ACTIVE_RUNS = Gauge(
    "agent_active_runs",
    "Number of agent runs currently in progress",
)