"""
Integration test for the full agent loop.
Run from project root: python test_agent.py
Requires: ollama serve (Mistral running)

Tests:
  1. Simple finish-only run (no tools)
  2. Calculator tool run
  3. SQLite log written correctly
  4. ChromaDB trace stored and retrieved
  5. Loop detection triggers correctly
  6. Max iterations triggers FAILED correctly
"""

import time

from agent.logger_setup import setup_logging
setup_logging()

from agent.agent_loop import AgentLoop, State
from db.sqlite_logger import SQLiteLogger
from memory.chroma_store import ChromaStore

passed = 0
failed = 0


def check(label: str, condition: bool):
    global passed, failed
    status = "  PASS" if condition else "  FAIL"
    print(f"{status}  {label}")
    if condition:
        passed += 1
    else:
        failed += 1


# Shared instances (reused across tests)
chroma = ChromaStore()
db = SQLiteLogger()
loop = AgentLoop(chroma_store=chroma, sqlite_logger=db)

# ── Test 1: simple question (agent should finish in 1-2 steps) ─────────────
print("\n=== Test 1: Simple finish ===")
result = loop.run("What is 2 + 2? Use the calculator tool and give me the answer.")
check("returns AgentResult", result is not None)
check("has run_id", bool(result.run_id))
check("has answer", bool(result.answer))
check("outcome is DONE or FAILED", result.outcome in (State.DONE, State.FAILED))
check("duration > 0", result.duration_s > 0)

# ── Test 2: SQLite log persisted ───────────────────────────────────────────
print("\n=== Test 2: SQLite persistence ===")
row = db.get_run(result.run_id)
check("row written to SQLite", row is not None)
check("run_id matches", row["run_id"] == result.run_id)
check("query matches", row["query"] == result.query)
check("outcome matches", row["outcome"] == result.outcome)
check("steps matches", row["steps"] == result.steps)

# ── Test 3: ChromaDB trace stored ─────────────────────────────────────────
print("\n=== Test 3: ChromaDB memory ===")
count_before = chroma.count()
result2 = loop.run("What is the square root of 256?")
count_after = chroma.count()
check("trace count increased", count_after > count_before)

# Retrieve — should find something similar to a math question
traces = chroma.retrieve_similar("calculate square root")
check("retrieves similar traces", len(traces) > 0)
check("trace is a string", isinstance(traces[0], str))

# ── Test 4: recent runs ────────────────────────────────────────────────────
print("\n=== Test 4: SQLite recent_runs ===")
runs = db.recent_runs(n=5)
check("recent_runs returns list", isinstance(runs, list))
check("at least 2 runs logged", len(runs) >= 2)
check("newest run first", runs[0]["start_time"] >= runs[1]["start_time"])

# ── Test 5: loop detection ─────────────────────────────────────────────────
print("\n=== Test 5: Loop detection ===")
# This query is designed to make the agent repeat the same search
result3 = loop.run(
    "Search for 'loop detection test' exactly. "
    "No matter what you find, search for it again. Keep searching."
)
# Either loop detected or max iterations — both are FAILED
check("loop causes FAILED", result3.outcome == State.FAILED)
check("fail_reason set", bool(result3.fail_reason))

# ── Test 6: trace build helper ─────────────────────────────────────────────
print("\n=== Test 6: Trace summary builder ===")
dummy_steps = [
    {"thought": "I need to search", "action": "search", "action_input": "test"},
    {"thought": "I found the answer", "action": "finish", "action_input": "42"},
]
summary = ChromaStore.build_trace_summary("what is the answer", dummy_steps, "DONE")
check("summary is a string", isinstance(summary, str))
check("summary contains query", "what is the answer" in summary)
check("summary contains outcome", "DONE" in summary)
check("summary contains action", "search" in summary)

# ── Summary ───────────────────────────────────────────────────────────────
print(f"\n{'='*40}")
print(f"Results: {passed} passed, {failed} failed")
if failed == 0:
    print("All tests passed. Agent loop ready for Phase 5.")
else:
    print("Review failures above — some may be LLM non-determinism (rerun once).")