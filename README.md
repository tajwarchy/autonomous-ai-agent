# Autonomous AI Agent from Scratch

A production-grade ReAct agent built without LangChain — featuring circuit breakers, dual databases, Prometheus/Grafana monitoring, and a clean LLM abstraction layer that works with Ollama, OpenAI, or Anthropic by changing one config value.

---

## Quick Start

```bash
# 1. Clone and create environment
conda env create -f environment.yml
conda activate autonomous-agent

# 2. Pull the model
ollama pull mistral

# 3. Start Ollama
ollama serve

# 4. Start the full monitoring stack
docker compose up --build

# 5. Query the agent
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the speed of light? Use Wikipedia."}'
```

| Service | URL |
|---|---|
| Agent API + Swagger | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |
| Grafana (admin/admin) | http://localhost:3000 |

---

## Architecture

```
Client (HTTP POST /query)
        │
        ▼
FastAPI (api/main.py)
        │
        ├─► ChromaDB — retrieve top-K semantically similar past traces
        │
        ▼
Agent Loop — ReAct state machine (agent/agent_loop.py)
  THINKING → ACTING → OBSERVING → DONE / FAILED
        │
        ▼
Tool Router (tools/tool_router.py)
  1. Circuit breaker check  (skip if tool OPEN)
  2. Rate limiter check     (skip if call limit reached)
  3. Execute with timeout   (10s hard cap)
  4. Record outcome         (update breaker + limiter + metrics)
        │
        ├── search      (DuckDuckGo)
        ├── calculator  (safe eval)
        ├── file_reader (sandboxed directory)
        └── wikipedia   (wikipedia-api)
        │
        ▼
Observation injected back into LLM context
        │
        ▼
Final Answer
        │
        ├─► ChromaDB — store reasoning trace as embedding
        ├─► SQLite   — store structured run log
        └─► Response to client
```

---

## Agent State Machine

```
         ┌─────────────────────────────────────────┐
         │              THINKING                   │
         │   LLM generates Thought/Action/Input    │
         └──────┬──────────────────────┬───────────┘
                │ valid parse          │ 2× parse errors
                ▼                     ▼
         ┌────────────┐         ┌──────────┐
         │   ACTING   │         │  FAILED  │ (terminal)
         │ Tool router│         └──────────┘
         └──────┬─────┘
                │ tool returns (result or error string)
                ▼
         ┌─────────────────────────────────────────┐
         │              OBSERVING                  │
         │   Result injected into LLM context      │
         └──────┬──────────────┬───────────────────┘
                │              │
        action=finish    max_iter / loop
                │              │
                ▼              ▼
         ┌──────────┐   ┌──────────┐
         │   DONE   │   │  FAILED  │  (both terminal)
         └──────────┘   └──────────┘
```

**Transition conditions:**
- `THINKING → ACTING`: LLM output parsed into valid Thought/Action/Input
- `THINKING → FAILED`: 2 consecutive parse errors
- `ACTING → OBSERVING`: always (tool returns a string in all cases)
- `OBSERVING → THINKING`: action ≠ finish AND iterations remain AND no loop detected
- `OBSERVING → DONE`: action = finish
- `OBSERVING → FAILED`: max_iterations reached OR same (action, input) repeated 3×

---

## System Design Checkpoint

### Why two databases?

| | ChromaDB (vector) | SQLite (relational) |
|---|---|---|
| **What's stored** | Full reasoning traces as text embeddings | Structured run metadata: run_id, timestamps, steps, outcome |
| **Query type** | "Find past traces semantically similar to this new query" | "Show all failed runs", "Average steps per outcome" |
| **Access pattern** | Write once at run end, read at next relevant run start | Write once at run end, read for dashboards and debugging |
| **Why the other is wrong** | SQL has no semantic similarity — only exact or keyword match | A vector DB has no rows, joins, or aggregates |

SQLite answers: *what happened and when*. ChromaDB answers: *what have I seen before that's like this*.

---

### What is the circuit breaker pattern and why does it matter in agentic systems?

A circuit breaker tracks consecutive failures for a resource. After a threshold (here: 3), it moves to OPEN state and rejects all further calls immediately — returning a descriptive error string instead of attempting the call.

**Why it matters specifically for agents:**

A standard API has a human in the loop who notices a broken service. An agent loop has no such check — it will keep calling a failed tool on every iteration until max_iterations, burning LLM tokens and wall-clock time on guaranteed failures. In a multi-user system, one broken external API could saturate all agent runs.

The circuit breaker converts this from a timeout-per-call problem into a single-detection problem: detect failure once, block instantly for the rest of the run. Grafana's `circuit_breaker_trips_total` metric makes these events visible in real time.

---

### What is the difference between a metric, a log, and a trace?

**Metric** — a numeric measurement aggregated over time. Answers: *how many? how fast? how often?*
```
# HELP tool_call_total Total number of tool calls dispatched
# TYPE tool_call_total counter
tool_call_total{tool="wikipedia"} 12.0
tool_call_total{tool="calculator"} 8.0
```

**Log** — a structured record of a discrete event. Answers: *what happened, when, and why?*
```json
{
  "ts": "2026-05-16T15:12:14Z",
  "level": "INFO",
  "logger": "tools.tool_router",
  "msg": "tool_call",
  "tool": "wikipedia",
  "input_preview": "Python (programming language)",
  "success": true,
  "latency_s": 2.847,
  "run_id": "9f9254e0-..."
}
```

**Trace** — the causal sequence of steps across a request. Answers: *what path did this request take?*
```
run_id: 9f9254e0
  step 1: THINKING  → action=wikipedia, input="Python (programming language)"
  step 2: ACTING    → wikipedia returned 851 chars in 2.85s
  step 3: OBSERVING → finish called, answer extracted
  outcome: DONE in 76s
```

Together: metrics tell you *something is wrong*, logs tell you *what went wrong*, traces tell you *where in the request it went wrong*.

---

### If this agent served 500 concurrent users, what would break first?

**1. Ollama (breaks first)** — Mistral 7B runs single-threaded on one GPU/Metal device. Each request takes 30–120s. At 500 concurrent users, 499 requests queue behind the first. Fix: run multiple Ollama instances behind a load balancer, or switch to a hosted API (OpenAI/Anthropic) with real concurrency.

**2. ChromaDB** — running in-process, embedded in the FastAPI app. Concurrent writes from multiple requests will serialize on a file lock. Fix: run ChromaDB in server mode (`chromadb.HttpClient`) as a separate service.

**3. SQLite** — single-writer by design. 500 concurrent run completions all trying to `INSERT` will queue and some will timeout. Fix: replace with PostgreSQL.

**4. FastAPI workers** — configured at `workers=1` (required for M1/macOS multiprocessing). Fix: deploy on Linux with `workers=4`, or use async task queue (Celery + Redis) to decouple request acceptance from agent execution.

**5. DuckDuckGo rate limits** — already seen in development. At scale, every search call risks a 202 rate-limit. Fix: add a paid search API (SerpAPI, Brave Search API) as the primary, DuckDuckGo as fallback.

---

### How is the LLM abstracted so this agent works with any provider?

The entire codebase calls exactly one method: `LLMClient.generate(prompt: str) -> str`.

Everything else — HTTP client, auth, model parameters, response parsing — lives inside `agent/llm_service.py` behind private `_call_*` methods. To swap providers, change `provider:` in `config.yaml` and uncomment the corresponding method. The ReAct loop, parser, tool router, memory, and logging are all completely unchanged.

```python
# config.yaml — the only change needed to swap providers
llm:
  provider: openai          # was: ollama
  model: gpt-4o             # was: mistral
```

```python
# llm_service.py — the only code change needed
def generate(self, prompt: str) -> str:
    if self.provider == "ollama":
        return self._call_ollama(prompt)
    elif self.provider == "openai":
        return self._call_openai(prompt)      # uncomment this method
    elif self.provider == "anthropic":
        return self._call_anthropic(prompt)   # uncomment this method
```

Both `_call_openai` and `_call_anthropic` are fully implemented in `llm_service.py` as commented-out methods, ready to activate.

---

## Tech Stack

| Component | Technology | Why |
|---|---|---|
| LLM inference | Ollama + Mistral 7B | Free, local, M1 Metal acceleration |
| Agent framework | Custom ReAct loop | No LangChain — full control over every prompt |
| Vector memory | ChromaDB (persistent) | Semantic similarity search for past traces |
| Structured logs | SQLite | Exact queries on run metadata |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Local, no API key |
| API | FastAPI | Async, auto-docs, Pydantic validation |
| Monitoring | Prometheus + Grafana | Industry standard observability stack |
| Deployment | Docker Compose | One-command full stack startup |
| Environment | conda | Reproducible Python environment |

---

## Monitoring Dashboard

Grafana at `http://localhost:3000` (admin/admin) shows:

- **Active Runs** — agent runs currently in progress
- **Total Runs DONE / FAILED** — cumulative outcomes
- **Circuit Breaker Trips** — total safety interventions
- **Avg Run Latency** — end-to-end p50 response time
- **Tool Call Rate** — calls per minute per tool (time series)
- **Tool Latency p50** — median latency per tool
- **Tool Failure Rate** — failures per minute per tool
- **Agent Run Latency p50/p99** — full run duration distribution
- **Circuit Breaker Trips by Tool** — which tools are failing
- **Rate Limit Hits by Tool** — which tools are being throttled

---

## Key Design Patterns

### Rate Limiting
Each tool has a per-run call limit (`config.yaml: tools.rate_limits`). When hit, the agent receives a descriptive error string as an Observation and must reason around it — choosing a different tool or calling finish.

### Circuit Breaker
3 consecutive failures → tool marked OPEN for the remainder of the run. Subsequent calls return instantly with an error string. Prevents wasted LLM iterations on guaranteed failures.

### Loop Detection
The agent tracks `(action, action_input)` pairs per run. If the same pair appears 3 times, the run is aborted with `FAILED` and `LOOP_DETECTED`. Prevents infinite loops from model repetition.

### Semantic Memory
At run start, ChromaDB retrieves the 3 most similar past traces by cosine similarity. These are injected into the system prompt, giving the agent awareness of how similar questions were handled before.

### LLM Abstraction
`LLMClient.generate(prompt)` is the only interface. Swap Ollama for OpenAI or Anthropic by changing `provider:` in config and uncommenting one method. The entire agent stack is provider-agnostic.

---

## File Structure

```
autonomous-agent/
├── agent/          # LLM client, ReAct parser, prompt builder, agent loop
├── tools/          # 4 tools + rate limiter + circuit breaker + router
├── memory/         # ChromaDB semantic store
├── db/             # SQLite structured logger
├── api/            # FastAPI app
├── monitoring/     # Prometheus metrics + scrape config
├── grafana/        # Dashboard JSON + provisioning
├── config/         # config.yaml (all parameters)
├── prompts/        # ReAct prompt template
└── docker-compose.yml
```

---

## Swapping the LLM

```bash
# 1. Change config.yaml
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514

# 2. Set your API key
export ANTHROPIC_API_KEY=sk-ant-...

# 3. In agent/llm_service.py, uncomment _call_anthropic
# The ReAct loop, tools, memory, and monitoring are unchanged.
```

---

## Known Limitations (Mistral 7B)

- **Finish compliance** — Mistral 7B occasionally ignores `finish` instructions and repeats tool calls. Mitigated by prompt continuation (`\nThought:` cue) and loop detection.
- **Quote wrapping** — Mistral wraps Action Inputs in quotes (`"2 + 2"`). Mitigated by stripping quotes in the calculator, search, and Wikipedia tools.
- **Latency** — 15–120s per run on M1 Metal. Expected for a 7B model without GPU batching. A hosted API would reduce this to 2–5s.
- **DuckDuckGo rate limits** — aggressive rate limiting under repeated queries. The circuit breaker correctly handles this but limits search reliability in demo sessions.