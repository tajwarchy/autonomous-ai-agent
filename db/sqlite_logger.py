"""
SQLite logger — structured run logs, one row per completed agent run.

Schema:
    run_id      TEXT PRIMARY KEY   — uuid4
    query       TEXT               — original user question
    start_time  REAL               — unix timestamp
    end_time    REAL               — unix timestamp
    duration_s  REAL               — wall clock seconds
    steps       INTEGER            — number of ReAct iterations completed
    tools_used  TEXT               — JSON list of tool names called
    outcome     TEXT               — DONE | FAILED
    fail_reason TEXT               — null if DONE, reason string if FAILED
    trace_json  TEXT               — full step-by-step JSON trace

Why SQLite and not ChromaDB?
    ChromaDB stores embeddings for semantic similarity search.
    SQLite stores structured metadata for exact queries:
      "show all failed runs", "average steps per run", "runs using wikipedia".
    A vector DB cannot answer these questions efficiently.
"""

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS agent_runs (
    run_id      TEXT PRIMARY KEY,
    query       TEXT NOT NULL,
    start_time  REAL NOT NULL,
    end_time    REAL NOT NULL,
    duration_s  REAL NOT NULL,
    steps       INTEGER NOT NULL,
    tools_used  TEXT NOT NULL,
    outcome     TEXT NOT NULL,
    fail_reason TEXT,
    trace_json  TEXT NOT NULL
);
"""


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class SQLiteLogger:

    def __init__(self, config: Optional[dict] = None):
        cfg = config or load_config()
        db_path = Path(cfg["database"]["sqlite"]["path"])
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._path = str(db_path)
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE)
        logger.debug("sqlite_initialized", extra={"path": self._path})

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        return conn

    def log_run(
        self,
        run_id: str,
        query: str,
        start_time: float,
        steps: int,
        tools_used: list[str],
        outcome: str,
        trace: list[dict],
        fail_reason: Optional[str] = None,
    ) -> None:
        """
        Write one completed agent run to SQLite.
        Called at the end of every agent run regardless of outcome.
        """
        end_time = time.time()
        duration = round(end_time - start_time, 3)

        row = {
            "run_id": run_id,
            "query": query,
            "start_time": start_time,
            "end_time": end_time,
            "duration_s": duration,
            "steps": steps,
            "tools_used": json.dumps(tools_used),
            "outcome": outcome,
            "fail_reason": fail_reason,
            "trace_json": json.dumps(trace),
        }

        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO agent_runs
                        (run_id, query, start_time, end_time, duration_s,
                         steps, tools_used, outcome, fail_reason, trace_json)
                    VALUES
                        (:run_id, :query, :start_time, :end_time, :duration_s,
                         :steps, :tools_used, :outcome, :fail_reason, :trace_json)
                    """,
                    row,
                )
            logger.info(
                "run_logged",
                extra={
                    "run_id": run_id,
                    "outcome": outcome,
                    "steps": steps,
                    "duration_s": duration,
                },
            )
        except sqlite3.Error as e:
            logger.error("sqlite_log_failed", extra={"run_id": run_id, "error": str(e)})

    def get_run(self, run_id: str) -> Optional[dict]:
        """Fetch a single run by ID. Returns None if not found."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM agent_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["tools_used"] = json.loads(d["tools_used"])
        d["trace_json"] = json.loads(d["trace_json"])
        return d

    def recent_runs(self, n: int = 10) -> list[dict]:
        """Return the N most recent runs, newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM agent_runs ORDER BY start_time DESC LIMIT ?", (n,)
            ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["tools_used"] = json.loads(d["tools_used"])
            result.append(d)
        return result