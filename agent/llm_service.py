"""
LLM abstraction layer.

Current backend: Ollama (local, free, M1 Metal).

To swap to a different provider, replace ONLY the `_call_*` method used in
`generate()`. The ReAct loop, parser, and agent loop never change.

Swap guide (see bottom of file for full examples):
  - OpenAI:    replace `_call_ollama` with `_call_openai`
  - Anthropic: replace `_call_ollama` with `_call_anthropic`
"""

import logging
import time
from typing import Optional

import httpx
import yaml

logger = logging.getLogger(__name__)


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class LLMClient:
    """
    Clean interface for LLM inference.
    Consumers call: client.generate(prompt) -> str
    Nothing outside this class knows which provider is in use.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or load_config()
        self.cfg = cfg["llm"]
        self.provider = self.cfg["provider"]          # ollama | openai | anthropic
        self.model = self.cfg["model"]
        self.temperature = self.cfg["temperature"]
        self.max_tokens = self.cfg["max_tokens"]
        self.timeout = self.cfg["timeout_seconds"]
        self.base_url = self.cfg.get("base_url", "http://localhost:11434")

    # ------------------------------------------------------------------
    # Public interface — the only method the rest of the codebase calls
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> str:
        """
        Send a prompt to the configured LLM and return the response text.
        Raises LLMError on failure after logging.
        """
        start = time.monotonic()
        try:
            if self.provider == "ollama":
                response = self._call_ollama(prompt)
            # ── SWAP POINT ─────────────────────────────────────────────
            # elif self.provider == "openai":
            #     response = self._call_openai(prompt)
            # elif self.provider == "anthropic":
            #     response = self._call_anthropic(prompt)
            # ───────────────────────────────────────────────────────────
            else:
                raise LLMError(f"Unknown provider: {self.provider}")

            latency = time.monotonic() - start
            logger.info(
                "llm_call_success",
                extra={
                    "provider": self.provider,
                    "model": self.model,
                    "latency_s": round(latency, 3),
                    "prompt_chars": len(prompt),
                    "response_chars": len(response),
                },
            )
            return response

        except LLMError:
            raise
        except Exception as e:
            latency = time.monotonic() - start
            logger.error(
                "llm_call_failed",
                extra={"provider": self.provider, "error": str(e), "latency_s": round(latency, 3)},
            )
            raise LLMError(f"LLM call failed: {e}") from e

    # ------------------------------------------------------------------
    # Ollama backend (active)
    # ------------------------------------------------------------------

    def _call_ollama(self, prompt: str) -> str:
        """
        Calls Ollama's /api/generate endpoint (non-streaming).
        Ollama runs locally on M1 Metal — no API key required.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.base_url}/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()

        text = data.get("response", "").strip()
        if not text:
            raise LLMError("Ollama returned an empty response")
        return text

    # ------------------------------------------------------------------
    # ── SWAP POINT: OpenAI backend ─────────────────────────────────────
    #
    # To activate:
    #   1. pip install openai
    #   2. Set provider: openai in config.yaml
    #   3. Set OPENAI_API_KEY in your environment
    #   4. Change model to e.g. gpt-4o in config.yaml
    #
    # def _call_openai(self, prompt: str) -> str:
    #     import openai
    #     client = openai.OpenAI()           # reads OPENAI_API_KEY from env
    #     resp = client.chat.completions.create(
    #         model=self.model,
    #         messages=[{"role": "user", "content": prompt}],
    #         temperature=self.temperature,
    #         max_tokens=self.max_tokens,
    #     )
    #     return resp.choices[0].message.content.strip()
    #
    # ── SWAP POINT: Anthropic backend ─────────────────────────────────
    #
    # To activate:
    #   1. pip install anthropic
    #   2. Set provider: anthropic in config.yaml
    #   3. Set ANTHROPIC_API_KEY in your environment
    #   4. Change model to e.g. claude-sonnet-4-20250514 in config.yaml
    #
    # def _call_anthropic(self, prompt: str) -> str:
    #     import anthropic
    #     client = anthropic.Anthropic()     # reads ANTHROPIC_API_KEY from env
    #     resp = client.messages.create(
    #         model=self.model,
    #         max_tokens=self.max_tokens,
    #         messages=[{"role": "user", "content": prompt}],
    #     )
    #     return resp.content[0].text.strip()
    #
    # ------------------------------------------------------------------


class LLMError(Exception):
    """Raised when the LLM call fails for any reason."""
    pass