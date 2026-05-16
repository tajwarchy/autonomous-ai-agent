"""
ReAct agent loop — the core of the system.

State machine per run:
    THINKING   → LLM generates Thought/Action/Action Input
    ACTING     → Tool router dispatches the action
    OBSERVING  → Observation injected back into context
    DONE       → action == "finish", return answer
    FAILED     → max iterations / loop detected / parse errors exceeded

One AgentLoop instance handles one query. Create a new one per request.
"""

import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from agent.llm_service import LLMClient, LLMError
from agent.prompt_builder import PromptBuilder
from agent.react_parser import ReActParser, build_corrective_prompt
from db.sqlite_logger import SQLiteLogger
from memory.chroma_store import ChromaStore
from tools.tool_router import ToolRouter

logger = logging.getLogger(__name__)
console = Console()


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Agent states ───────────────────────────────────────────────────────────

class State:
    THINKING = "THINKING"
    ACTING = "ACTING"
    OBSERVING = "OBSERVING"
    DONE = "DONE"
    FAILED = "FAILED"


# ── Result dataclass ───────────────────────────────────────────────────────

@dataclass
class AgentResult:
    run_id: str
    query: str
    answer: str
    outcome: str                        # DONE | FAILED
    fail_reason: Optional[str]
    steps: int
    tools_used: list[str]
    duration_s: float
    trace: list[dict] = field(default_factory=list)


# ── Agent loop ─────────────────────────────────────────────────────────────

class AgentLoop:

    def __init__(
        self,
        config: Optional[dict] = None,
        llm_client: Optional[LLMClient] = None,
        chroma_store: Optional[ChromaStore] = None,
        sqlite_logger: Optional[SQLiteLogger] = None,
    ):
        self._cfg = config or load_config()
        agent_cfg = self._cfg["agent"]

        self._max_iter = agent_cfg["max_iterations"]
        self._loop_threshold = agent_cfg["loop_detection_threshold"]
        self._max_parse_errors = 2

        # Core components (injectable for testing)
        self._llm = llm_client or LLMClient(self._cfg)
        self._parser = ReActParser()
        self._prompt_builder = PromptBuilder(self._cfg)
        self._chroma = chroma_store or ChromaStore(self._cfg)
        self._db = sqlite_logger or SQLiteLogger(self._cfg)

    def run(self, query: str) -> AgentResult:
        """
        Execute one full agent run for the given query.
        Always returns an AgentResult — never raises.
        """
        run_id = str(uuid.uuid4())
        start_time = time.time()

        logger.info("run_start", extra={"run_id": run_id, "query": query})
        self._print_run_start(run_id, query)

        # Per-run state
        router = ToolRouter(self._cfg)
        history: list[dict] = []          # Thought/Action/Observation turns
        trace: list[dict] = []            # full structured log
        tools_used: list[str] = []
        action_counts: dict = defaultdict(int)   # loop detection
        parse_error_count = 0
        state = State.THINKING
        answer = ""
        fail_reason = None

        # ── Retrieve relevant past memory ──────────────────────────────
        memory_traces = self._chroma.retrieve_similar(query)
        initial_prompt = self._prompt_builder.build_initial_prompt(query, memory_traces)
        current_prompt = initial_prompt

        # ── Main loop ──────────────────────────────────────────────────
        for iteration in range(1, self._max_iter + 1):
            logger.info(
                "iteration_start",
                extra={"run_id": run_id, "iteration": iteration, "state": state},
            )

            # ── THINKING ──────────────────────────────────────────────
            state = State.THINKING
            try:
                raw = self._llm.generate(current_prompt)
            except LLMError as e:
                fail_reason = f"LLM error: {e}"
                state = State.FAILED
                break

            step = self._parser.parse(raw)

            if not step.is_valid:
                parse_error_count += 1
                self._print_parse_error(iteration, step.parse_error)
                logger.warning(
                    "parse_error",
                    extra={
                        "run_id": run_id,
                        "iteration": iteration,
                        "error": step.parse_error,
                        "count": parse_error_count,
                    },
                )
                if parse_error_count >= self._max_parse_errors:
                    fail_reason = f"Too many parse errors: {step.parse_error}"
                    state = State.FAILED
                    break
                # Ask the LLM to reformat and retry this iteration
                current_prompt = current_prompt + build_corrective_prompt(step)
                continue

            parse_error_count = 0   # reset on successful parse

            self._print_thought(iteration, step.thought)

            # ── Loop detection ─────────────────────────────────────────
            loop_key = f"{step.action}::{step.action_input}"
            action_counts[loop_key] += 1
            if action_counts[loop_key] >= self._loop_threshold:
                fail_reason = (
                    f"Loop detected: action '{step.action}' with input "
                    f"'{step.action_input[:60]}' repeated {self._loop_threshold} times."
                )
                state = State.FAILED
                logger.warning("loop_detected", extra={"run_id": run_id, "loop_key": loop_key})
                break

            # ── DONE check ─────────────────────────────────────────────
            if step.is_finish:
                answer = step.action_input
                state = State.DONE
                self._print_finish(answer)
                break

            # ── ACTING ────────────────────────────────────────────────
            state = State.ACTING
            self._print_action(step.action, step.action_input)

            observation, latency = router.dispatch(step.action, step.action_input)
            tools_used.append(step.action)

            # ── OBSERVING ─────────────────────────────────────────────
            state = State.OBSERVING
            self._print_observation(observation, latency)

            # Record step in trace
            trace_step = {
                "iteration": iteration,
                "thought": step.thought,
                "action": step.action,
                "action_input": step.action_input,
                "observation": observation,
                "latency_s": round(latency, 3),
            }
            trace.append(trace_step)
            history.append({
                "thought": step.thought,
                "action": step.action,
                "action_input": step.action_input,
                "observation": observation,
            })

            # Rebuild prompt with updated history
            current_prompt = self._prompt_builder.build_continuation_prompt(
                initial_prompt, history
            )

        else:
            # Loop exhausted without break
            fail_reason = f"Reached max iterations ({self._max_iter}) without a final answer."
            state = State.FAILED

        # ── Finalize ───────────────────────────────────────────────────
        if state == State.FAILED and not answer:
            last_obs = history[-1]["observation"] if history else ""
            answer = (
                f"Agent could not complete the task. Reason: {fail_reason}\n"
                + (f"Last observation: {last_obs}" if last_obs else "")
            )

        duration = round(time.time() - start_time, 3)

        result = AgentResult(
            run_id=run_id,
            query=query,
            answer=answer,
            outcome=state,
            fail_reason=fail_reason,
            steps=len(trace),
            tools_used=tools_used,
            duration_s=duration,
            trace=trace,
        )

        # ── Persist ────────────────────────────────────────────────────
        self._db.log_run(
            run_id=run_id,
            query=query,
            start_time=start_time,
            steps=result.steps,
            tools_used=tools_used,
            outcome=state,
            trace=trace,
            fail_reason=fail_reason,
        )

        trace_summary = ChromaStore.build_trace_summary(query, trace, state)
        self._chroma.store_trace(run_id, query, trace_summary)

        self._print_run_end(result)
        logger.info(
            "run_end",
            extra={
                "run_id": run_id,
                "outcome": state,
                "steps": result.steps,
                "duration_s": duration,
                "tools_used": tools_used,
            },
        )

        return result

    # ── Rich terminal output ───────────────────────────────────────────

    def _print_run_start(self, run_id: str, query: str) -> None:
        console.print(Panel(
            f"[bold]Query:[/bold] {query}\n[dim]run_id: {run_id}[/dim]",
            title="[bold cyan]Agent Run Started[/bold cyan]",
            border_style="cyan",
        ))

    def _print_thought(self, iteration: int, thought: str) -> None:
        console.print(
            f"[dim][{iteration}][/dim] [yellow]Thought:[/yellow] {thought}"
        )

    def _print_action(self, action: str, action_input: str) -> None:
        console.print(
            f"    [bold blue]Action:[/bold blue] {action}({action_input!r})"
        )

    def _print_observation(self, observation: str, latency: float) -> None:
        preview = observation[:200] + ("..." if len(observation) > 200 else "")
        console.print(
            f"    [green]Observation[/green] [dim]({latency:.2f}s):[/dim] {preview}"
        )

    def _print_parse_error(self, iteration: int, error: str) -> None:
        console.print(
            f"[{iteration}] [bold red]Parse error:[/bold red] {error}"
        )

    def _print_finish(self, answer: str) -> None:
        console.print(Panel(
            answer,
            title="[bold green]Final Answer[/bold green]",
            border_style="green",
        ))

    def _print_run_end(self, result: AgentResult) -> None:
        color = "green" if result.outcome == State.DONE else "red"
        console.print(
            f"[{color}]Outcome: {result.outcome}[/{color}] | "
            f"Steps: {result.steps} | Duration: {result.duration_s}s"
        )