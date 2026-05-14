"""
Smoke test for the LLM abstraction layer.
Run from project root: python test_llm.py
Ollama must be running: ollama serve
"""

from agent.logger_setup import setup_logging
from agent.llm_service import LLMClient, LLMError

setup_logging()

client = LLMClient()

print("\n=== Test 1: basic generation ===")
try:
    response = client.generate("Reply with exactly three words: hello world test")
    print(f"Response: {response!r}")
    assert len(response) > 0, "Empty response"
    print("PASS")
except LLMError as e:
    print(f"FAIL: {e}")

print("\n=== Test 2: ReAct-formatted prompt ===")
react_prompt = (
    "You are a ReAct agent. Respond in this exact format:\n"
    "Thought: I need to answer the question.\n"
    "Action: finish\n"
    "Action Input: The capital of France is Paris.\n\n"
    "Question: What is the capital of France?"
)
try:
    response = client.generate(react_prompt)
    print(f"Response:\n{response}")
    print("PASS" if "Thought:" in response or "Action:" in response else "WARN: unexpected format")
except LLMError as e:
    print(f"FAIL: {e}")