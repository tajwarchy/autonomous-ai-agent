"""
Wikipedia summary tool using the wikipediaapi library.
"""

import logging
from typing import Optional

import wikipediaapi
import yaml

from tools.base import BaseTool

logger = logging.getLogger(__name__)


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class WikipediaTool(BaseTool):

    def __init__(self, config: Optional[dict] = None):
        cfg = config or load_config()
        tool_cfg = cfg["tools"]["wikipedia"]
        self._sentences = tool_cfg.get("sentences", 5)
        self._wiki = wikipediaapi.Wikipedia(
            language=tool_cfg.get("language", "en"),
            user_agent="autonomous-agent/1.0 (educational project)",
        )

    @property
    def name(self) -> str:
        return "wikipedia"

    @property
    def description(self) -> str:
        return (
            "Get a summary of a topic from Wikipedia. "
            "Input: a topic name, e.g. 'Python programming language'."
        )

    def run(self, input_str: str) -> str:
        topic = input_str.strip()
        if not topic:
            return "ERROR: wikipedia requires a non-empty topic."

        try:
            page = self._wiki.page(topic)

            if not page.exists():
                return f"ERROR: no Wikipedia page found for '{topic}'. Try a different search term."

            # Extract first N sentences from the summary
            summary = page.summary
            sentences = self._split_sentences(summary)
            excerpt = " ".join(sentences[: self._sentences])

            output = f"Wikipedia — {page.title}\n\n{excerpt}\n\nFull article: {page.fullurl}"
            logger.debug(
                "wikipedia_tool_ok",
                extra={"topic": topic, "title": page.title, "sentences": len(sentences)},
            )
            return output

        except Exception as e:
            logger.warning("wikipedia_tool_error", extra={"topic": topic, "error": str(e)})
            return f"ERROR: wikipedia lookup failed — {e}"

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Naive sentence splitter — good enough for Wikipedia summaries."""
        import re
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s for s in sentences if s]