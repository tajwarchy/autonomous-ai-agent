"""
Prompt assembly for the ReAct agent.

Keeps all string-building logic out of the agent loop.
Reads the prompt template from prompts/react.txt (path from config).
"""

import logging
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

# Tool descriptions injected into the system prompt
TOOL_DESCRIPTIONS = """\
- search(query): Search the web using DuckDuckGo. Returns top results as text.
- calculator(expression): Evaluate a math expression. E.g. "2 ** 10 + sqrt(144)".
- file_reader(filename): Read a file from the allowed data directory. Pass filename only, no path.
- wikipedia(topic): Get a summary of a topic from Wikipedia.
- finish(answer): Return your final answer and end the agent run."""


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class PromptBuilder:
    """
    Assembles the full prompt for each LLM call in the ReAct loop.

    The prompt has three sections:
      1. System block: ReAct instructions + tool descriptions + memory context
      2. Conversation history: all prior Thought/Action/Observation turns
      3. Current turn cue: nudge the model to produce its next Thought
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or load_config()
        template_path = Path("prompts/react.txt")
        self.template = template_path.read_text()
        logger.debug("prompt_template_loaded", extra={"path": str(template_path)})

    def build_initial_prompt(
        self,
        query: str,
        memory_context: list[str],
    ) -> str:
        """
        Build the opening prompt for a new agent run.

        Args:
            query: The user's question.
            memory_context: List of past trace strings retrieved from ChromaDB.
        """
        memory_block = self._format_memory(memory_context)
        return self.template.format(
            tool_descriptions=TOOL_DESCRIPTIONS,
            memory_context=memory_block,
            query=query,
        )

    def build_continuation_prompt(
        self,
        initial_prompt: str,
        history: list[dict],
    ) -> str:
        """
        Build the prompt for iteration N>1 of the ReAct loop.

        Args:
            initial_prompt: The prompt from build_initial_prompt (unchanged).
            history: List of dicts with keys: thought, action, action_input, observation.
        """
        history_block = self._format_history(history)
        # Always append the next-turn cue so the model knows to continue
        # in ReAct format — not dump free text
        cue = "\nThought:"
        return f"{initial_prompt}\n\n{history_block}{cue}"

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_memory(traces: list[str]) -> str:
        if not traces:
            return "No relevant past experience found."
        lines = []
        for i, trace in enumerate(traces, 1):
            lines.append(f"[Memory {i}]\n{trace.strip()}")
        return "\n\n".join(lines)

    @staticmethod
    def _format_history(history: list[dict]) -> str:
        """
        Renders prior ReAct turns as plain text so the LLM sees its own
        reasoning and can continue from where it left off.

        After the last observation, adds an explicit nudge so smaller models
        (Mistral 7B) understand they must now call finish, not repeat the tool.
        """
        blocks = []
        for i, turn in enumerate(history):
            block = (
                f"Thought: {turn['thought']}\n"
                f"Action: {turn['action']}\n"
                f"Action Input: {turn['action_input']}\n"
                f"Observation: {turn['observation']}"
            )
            # After the last turn, add an explicit finish nudge
            if i == len(history) - 1 and not turn["observation"].startswith("ERROR"):
                block += (
                    f"\n\nThe observation above is your answer. "
                    f"You MUST now call finish. Do NOT call any tool again."
                    f"\nThought: I have the answer from the observation."
                    f"\nAction: finish"
                    f"\nAction Input:"
                )
            blocks.append(block)
        return "\n\n".join(blocks)