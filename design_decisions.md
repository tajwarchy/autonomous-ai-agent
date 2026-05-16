---

## Implementation Observations

### Mistral 7B prompt compliance
Mistral 7B does not reliably follow `finish` instructions in the ReAct format. Three mitigations were implemented iteratively:
1. One-shot example added to `prompts/react.txt`
2. Explicit finish nudge injected after successful observations in `prompt_builder.py`
3. Prompt continuation (`\nThought:` cue appended to every continuation prompt) — most effective

### Quote wrapping
Mistral consistently wraps Action Input values in surrounding quotes (`"2 + 2"`, `'Python (programming language)'`). Fixed by stripping quotes at the entry point of calculator, search, and wikipedia tools. Loop detection key also normalised by stripping quotes before hashing.

### DuckDuckGo rate limiting
DuckDuckGo's free API aggressively rate-limits repeated queries (HTTP 202). The circuit breaker correctly detects this and blocks further calls after 3 failures. A retry with exponential backoff (1s, 2s) was added for transient rate limit errors. For production, replace with a paid search API.

### ChromaDB telemetry noise
ChromaDB's PostHog telemetry client had a breaking API change. Silenced via `logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)` in `logger_setup.py`.

### NumPy 2.0 incompatibility
`sentence-transformers` and `chromadb` were incompatible with NumPy 2.0 at the time of development. Pinned to `numpy<2.0` in `environment.yml`.# Design Decisions

This document records architectural choices and failure mode handling decided before writing functional code, plus resolutions observed during implementation.

---

## Failure Modes & Responses

### 1. LLM returns malformed output
**Scenario**: The model omits the `Thought:` / `Action:` / `Action Input:` structure, or returns freeform prose.

**Response**:
- The `react_parser.py` module attempts regex extraction of each field.
- If any field is missing after parsing, the step is logged as a `PARSE_ERROR`.
- The agent loop records this as a failed iteration, increments a failure counter, and sends a corrective prompt asking the model to reformat.
- After 2 consecutive parse errors, the loop transitions to `FAILED` state and returns whatever partial answer exists.

**Why not crash?** Crashing on malformed LLM output is unacceptable in production. Graceful degradation preserves partial work.

---

### 2. A tool call times out
**Scenario**: DuckDuckGo or Wikipedia takes longer than `tools.timeout_seconds` (default: 10s).

**Response**:
- Every tool call is wrapped in `concurrent.futures.ThreadPoolExecutor` with a timeout.
- On timeout, the tool returns a structured error string: `"ERROR: tool 'search' timed out after 10s"`.
- This error string is injected as the `Observation` so the agent can reason around it.
- The timeout counts as a failure for the circuit breaker.

**Why return an error string instead of raising?** Raising breaks the agent loop. An error string lets the LLM adapt its plan.

---

### 3. The agent loop runs forever
**Scenario**: The LLM keeps generating new actions without reaching `finish`.

**Response**:
- Hard cap of `agent.max_iterations` (default: 10) iterations enforced in the loop.
- On hitting the cap, the loop transitions to `FAILED` state, returns partial answer, and logs the full trace.

---

### 4. The same action repeats
**Scenario**: The LLM calls `search("Paris weather")` three times in a row.

**Response**:
- Each `(action, action_input)` pair is tracked in a dict with a count per run.
- If the same pair reaches `agent.loop_detection_threshold` (default: 3), the loop aborts.
- The agent returns its last known observation as a partial answer with state `FAILED` and reason `LOOP_DETECTED`.

---

### 5. A tool fails repeatedly
**Scenario**: Wikipedia API is down; every call returns an exception.

**Response**:
- The circuit breaker in `circuit_breaker.py` tracks consecutive failures per tool.
- After `tools.circuit_breaker.failure_threshold` (default: 3) consecutive failures, the tool is marked `OPEN`.
- Subsequent calls to that tool immediately return: `"ERROR: tool 'wikipedia' is currently unavailable (circuit breaker open)"`.
- This prevents wasted LLM iterations and cascading timeouts.
- The circuit remains OPEN for the entire agent run (no auto-reset within a run).

---

### 6. Rate limit is hit
**Scenario**: The agent calls `search` more than `tools.rate_limits.search` (default: 5) times in one run.

**Response**:
- The rate limiter returns: `"ERROR: tool 'search' rate limit reached (5/5 calls used). Try a different tool or finish with what you know."`.
- This is injected as an Observation so the LLM knows to change approach.

---

### 7. ChromaDB is empty on first run
**Scenario**: No past traces exist yet.

**Response**:
- `retrieve_similar()` returns an empty list, which renders as an empty memory context in the prompt.
- The agent proceeds normally. No error, no special handling needed.

---

### 8. SQLite file doesn't exist yet
**Scenario**: First run, database file hasn't been created.

**Response**:
- `sqlite_logger.py` runs `CREATE TABLE IF NOT EXISTS` on startup.
- The file is created automatically at the configured path.

---

## Database Selection Rationale

| Concern | ChromaDB (vector) | SQLite (relational) |
|---|---|---|
| **What's stored** | Full reasoning traces as text embeddings | Structured run metadata: run_id, timestamps, outcome, steps |
| **Query type** | "Find past traces semantically similar to this new query" | "Show me all failed runs in the last hour" |
| **Why the other type is wrong** | SQL can't do semantic similarity search; you'd need exact match or keyword search | A vector DB has no concept of rows, foreign keys, or aggregate queries |
| **Access pattern** | Write once per run, read at start of next relevant run | Write once per run, read for dashboards and debugging |

---

## System Architecture

```
Client (HTTP)
    │
    ▼
FastAPI (/query)
    │
    ├── ChromaDB ◄── retrieve similar past traces (at run start)
    │
    ▼
Agent Loop (ReAct state machine)
    │  THINKING → ACTING → OBSERVING → (repeat) → DONE / FAILED
    │
    ▼
Tool Router
    │  checks: rate limiter → circuit breaker → tool timeout
    ├── search (DuckDuckGo)
    ├── calculator
    ├── file_reader
    └── wikipedia
    │
    ▼
Observations injected back into LLM context
    │
    ▼
Final Answer
    │
    ├── ChromaDB ◄── store trace (at run end)
    ├── SQLite   ◄── store structured log (at run end)
    └── Response to Client
```

---

## Agent State Machine

| State | Entered when | Transitions to |
|---|---|---|
| `THINKING` | Start of every iteration; LLM generates next Thought/Action | `ACTING` on valid parse; `FAILED` on 2 consecutive parse errors |
| `ACTING` | Valid action dispatched to tool router | `OBSERVING` always (tool returns result or error string) |
| `OBSERVING` | Tool result injected into context | `THINKING` if action ≠ finish and iterations remain; `DONE` if action = finish; `FAILED` if max_iterations or loop detected |
| `DONE` | Action = finish | — (terminal) |
| `FAILED` | Max iterations, loop detected, or parse error threshold | — (terminal) |