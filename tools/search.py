"""
DuckDuckGo search tool — no API key required.
"""

import logging
import time
from typing import Optional

import yaml
from duckduckgo_search import DDGS

from tools.base import BaseTool

logger = logging.getLogger(__name__)


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class SearchTool(BaseTool):

    def __init__(self, config: Optional[dict] = None):
        cfg = config or load_config()
        self._max_results = cfg["tools"]["search"]["max_results"]

    @property
    def name(self) -> str:
        return "search"

    @property
    def description(self) -> str:
        return "Search the web using DuckDuckGo. Input: a plain search query string."

    def run(self, input_str: str) -> str:
        # Strip surrounding quotes the LLM sometimes adds
        query = input_str.strip().strip("\"'")
        if not query:
            return "ERROR: search requires a non-empty query."

        try:
            results = None
            for attempt in range(3):
                try:
                    with DDGS() as ddgs:
                        results = list(ddgs.text(query, max_results=self._max_results))
                    break
                except Exception as e:
                    if attempt < 2 and "202" in str(e):
                        time.sleep(2 ** attempt)   # 1s, 2s backoff
                        continue
                    raise

            if not results:
                return f"No results found for query: {query!r}"

            lines = []
            for i, r in enumerate(results, 1):
                title = r.get("title", "No title")
                body = r.get("body", "No snippet")
                href = r.get("href", "")
                lines.append(f"[{i}] {title}\n{body}\nURL: {href}")

            output = "\n\n".join(lines)
            logger.debug(
                "search_tool_ok",
                extra={"query": query, "num_results": len(results)},
            )
            return output

        except Exception as e:
            logger.warning("search_tool_error", extra={"query": query, "error": str(e)})
            return f"ERROR: search failed — {e}"