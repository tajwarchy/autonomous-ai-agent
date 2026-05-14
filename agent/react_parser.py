"""
ReAct output parser.

Parses raw LLM text into structured ReActStep objects.
Handles malformed output gracefully — never raises, always returns a result
with a clear parse_error field so the agent loop can decide what to do.

Expected LLM format:
    Thought: <reasoning>
    Action: <tool_name>
    Action Input: <input string>
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Tools the agent is allowed to call (validated after parsing)
VALID_ACTIONS = {"search", "calculator", "file_reader", "wikipedia", "finish"}


@dataclass
class ReActStep:
    """
    One parsed iteration of the ReAct loop.

    Fields are None when parsing failed for that field.
    Check `is_valid` before using this step in the agent loop.
    """
    thought: Optional[str] = None
    action: Optional[str] = None
    action_input: Optional[str] = None
    raw: str = ""                          # original LLM output, always preserved
    parse_error: Optional[str] = None     # human-readable reason if parsing failed

    @property
    def is_valid(self) -> bool:
        return (
            self.parse_error is None
            and self.thought is not None
            and self.action is not None
            and self.action_input is not None
        )

    @property
    def is_finish(self) -> bool:
        return self.action == "finish"


class ReActParser:
    """
    Parses LLM output into ReActStep.

    Strategy:
      1. Try strict line-by-line extraction first (fast path).
      2. Fall back to regex search across the whole string (handles extra
         whitespace, blank lines, minor formatting drift).
      3. If either field is still missing, populate parse_error and return.
      4. Validate that action is a known tool name.
    """

    # Patterns: label at start of a line, capture everything after ": "
    _THOUGHT_RE = re.compile(r"(?im)^Thought\s*:\s*(.+?)(?=\n(?:Action|$))", re.DOTALL)
    _ACTION_RE = re.compile(r"(?im)^Action\s*:\s*(.+?)$")
    _INPUT_RE = re.compile(r"(?im)^Action\s*Input\s*:\s*(.+?)(?=\n|$)", re.DOTALL)

    def parse(self, raw: str) -> ReActStep:
        """
        Parse raw LLM output into a ReActStep.
        Never raises — all errors are captured in ReActStep.parse_error.
        """
        step = ReActStep(raw=raw)

        thought = self._extract(self._THOUGHT_RE, raw)
        action = self._extract(self._ACTION_RE, raw)
        action_input = self._extract(self._INPUT_RE, raw)

        # Collect all missing fields in one pass for a complete error message
        missing = []
        if thought is None:
            missing.append("Thought")
        if action is None:
            missing.append("Action")
        if action_input is None:
            missing.append("Action Input")

        if missing:
            step.parse_error = f"Missing fields: {', '.join(missing)}"
            logger.warning(
                "react_parse_error",
                extra={"reason": step.parse_error, "raw_preview": raw[:200]},
            )
            return step

        # Normalise
        action = action.strip().lower()

        # Validate action name
        if action not in VALID_ACTIONS:
            step.parse_error = (
                f"Unknown action '{action}'. Valid actions: {sorted(VALID_ACTIONS)}"
            )
            logger.warning(
                "react_parse_error",
                extra={"reason": step.parse_error, "raw_preview": raw[:200]},
            )
            return step

        step.thought = thought.strip()
        step.action = action
        step.action_input = action_input.strip()

        logger.debug(
            "react_parse_ok",
            extra={"action": step.action, "action_input_preview": step.action_input[:80]},
        )
        return step

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract(pattern: re.Pattern, text: str) -> Optional[str]:
        m = pattern.search(text)
        if not m:
            return None
        return m.group(1).strip()


def build_corrective_prompt(step: ReActStep) -> str:
    """
    Returns a prompt addendum asking the LLM to reformat its last response.
    Injected by the agent loop after a parse failure.
    """
    return (
        f"\n\nYour previous response could not be parsed. Reason: {step.parse_error}\n"
        "Please respond again using EXACTLY this format and nothing else:\n\n"
        "Thought: <your reasoning>\n"
        "Action: <one of: search, calculator, file_reader, wikipedia, finish>\n"
        "Action Input: <plain string input>\n"
    )