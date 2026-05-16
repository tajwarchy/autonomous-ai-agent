"""
FastAPI application — exposes the agent via HTTP.

Endpoints:
  POST /query    — run the agent on a question
  GET  /health   — liveness check
  GET  /metrics  — Prometheus metrics exposition
  GET  /runs     — last N agent runs from SQLite
  GET  /runs/{run_id} — single run detail
"""

import yaml
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from agent.agent_loop import AgentLoop
from agent.logger_setup import setup_logging
from db.sqlite_logger import SQLiteLogger
from memory.chroma_store import ChromaStore


# ── Config ─────────────────────────────────────────────────────────────────

def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Shared singletons (created once at startup) ────────────────────────────

_config: dict = {}
_chroma: Optional[ChromaStore] = None
_db: Optional[SQLiteLogger] = None
_agent: Optional[AgentLoop] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all singletons on startup, clean up on shutdown."""
    global _config, _chroma, _db, _agent
    _config = load_config()
    setup_logging(_config)

    _chroma = ChromaStore(_config)
    _db = SQLiteLogger(_config)
    _agent = AgentLoop(config=_config, chroma_store=_chroma, sqlite_logger=_db)

    yield
    # Shutdown: nothing to explicitly close for SQLite/ChromaDB local clients


# ── App ────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Autonomous AI Agent",
    description="ReAct agent with tool use, semantic memory, and Prometheus monitoring.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Request / Response models ──────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str

    model_config = {"json_schema_extra": {"example": {"query": "What is the capital of France?"}}}


class QueryResponse(BaseModel):
    run_id: str
    query: str
    answer: str
    outcome: str
    fail_reason: Optional[str]
    steps: int
    tools_used: list[str]
    duration_s: float


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """
    Run the agent on a question and return the result.
    Runs synchronously — one request at a time (Ollama is single-threaded).
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    result = _agent.run(request.query.strip())

    return QueryResponse(
        run_id=result.run_id,
        query=result.query,
        answer=result.answer,
        outcome=result.outcome,
        fail_reason=result.fail_reason,
        steps=result.steps,
        tools_used=result.tools_used,
        duration_s=result.duration_s,
    )


@app.get("/health")
def health() -> dict:
    """Liveness check — returns 200 if the service is up."""
    return {
        "status": "ok",
        "chroma_traces": _chroma.count() if _chroma else 0,
    }


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> Response:
    """Prometheus metrics exposition endpoint."""
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


@app.get("/runs")
def recent_runs(n: int = 10) -> list[dict]:
    """Return the N most recent agent runs from SQLite."""
    if n < 1 or n > 100:
        raise HTTPException(status_code=400, detail="n must be between 1 and 100.")
    return _db.recent_runs(n=n)


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict:
    """Return a single agent run by ID."""
    row = _db.get_run(run_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    return row