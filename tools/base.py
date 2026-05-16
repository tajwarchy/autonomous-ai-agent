"""
Abstract base class for all agent tools.

Every tool:
  - Has a name and description (injected into the LLM prompt)
  - Implements run(input_str) -> str
  - Is wrapped with a timeout enforced by the tool router
  - Never raises to the agent loop — errors are returned as strings
"""

from abc import ABC, abstractmethod


class BaseTool(ABC):
    """
    All tools inherit from this class.
    The tool router calls tool.run(input_str) and expects a plain string back.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool identifier. Must match VALID_ACTIONS in react_parser.py."""

    @property
    @abstractmethod
    def description(self) -> str:
        """One-line description shown to the LLM in the system prompt."""

    @abstractmethod
    def run(self, input_str: str) -> str:
        """
        Execute the tool and return a result string.
        Must never raise — catch all exceptions and return an error string.
        The tool router handles timeouts externally.
        """

    def __repr__(self) -> str:
        return f"<Tool name={self.name!r}>"