"""
Unit tests for the ReAct parser.
Run from project root: python test_parser.py
No Ollama needed — tests only the parser logic.
"""

from agent.react_parser import ReActParser, build_corrective_prompt

parser = ReActParser()
passed = 0
failed = 0


def check(label: str, condition: bool):
    global passed, failed
    if condition:
        print(f"  PASS  {label}")
        passed += 1
    else:
        print(f"  FAIL  {label}")
        failed += 1


# ── Case 1: well-formed output ─────────────────────────────────────────────
print("\n=== Case 1: well-formed output ===")
raw = (
    "Thought: I should search for the latest news.\n"
    "Action: search\n"
    "Action Input: latest AI news 2024"
)
step = parser.parse(raw)
check("is_valid", step.is_valid)
check("thought captured", step.thought == "I should search for the latest news.")
check("action = search", step.action == "search")
check("action_input captured", step.action_input == "latest AI news 2024")
check("no parse_error", step.parse_error is None)

# ── Case 2: finish action ──────────────────────────────────────────────────
print("\n=== Case 2: finish action ===")
raw = (
    "Thought: I have enough information.\n"
    "Action: finish\n"
    "Action Input: The answer is 42."
)
step = parser.parse(raw)
check("is_valid", step.is_valid)
check("is_finish", step.is_finish)
check("action_input captured", step.action_input == "The answer is 42.")

# ── Case 3: extra blank lines and trailing spaces ─────────────────────────
print("\n=== Case 3: extra whitespace ===")
raw = (
    "\nThought:   I need to calculate something.  \n\n"
    "Action:   calculator  \n"
    "Action Input:   2 ** 10\n\n"
)
step = parser.parse(raw)
check("is_valid", step.is_valid)
check("thought stripped", step.thought == "I need to calculate something.")
check("action normalised", step.action == "calculator")
check("action_input stripped", step.action_input == "2 ** 10")

# ── Case 4: missing Thought ────────────────────────────────────────────────
print("\n=== Case 4: missing Thought ===")
raw = (
    "Action: search\n"
    "Action Input: something"
)
step = parser.parse(raw)
check("not valid", not step.is_valid)
check("parse_error set", step.parse_error is not None)
check("Thought in error msg", "Thought" in step.parse_error)

# ── Case 5: unknown action ─────────────────────────────────────────────────
print("\n=== Case 5: unknown action name ===")
raw = (
    "Thought: Let me browse the web.\n"
    "Action: browser\n"
    "Action Input: https://example.com"
)
step = parser.parse(raw)
check("not valid", not step.is_valid)
check("parse_error mentions unknown action", "browser" in step.parse_error)

# ── Case 6: completely garbled output ────────────────────────────────────
print("\n=== Case 6: completely garbled output ===")
raw = "Sure! I'd be happy to help you with that question about AI."
step = parser.parse(raw)
check("not valid", not step.is_valid)
check("raw preserved", step.raw == raw)

# ── Case 7: corrective prompt ─────────────────────────────────────────────
print("\n=== Case 7: corrective prompt generation ===")
prompt = build_corrective_prompt(step)
check("contains format instructions", "Thought:" in prompt and "Action:" in prompt)
check("contains parse reason", step.parse_error in prompt)

# ── Summary ───────────────────────────────────────────────────────────────
print(f"\n{'='*40}")
print(f"Results: {passed} passed, {failed} failed")
if failed == 0:
    print("All tests passed. Parser is ready for Phase 3.")
else:
    print("Fix failures before proceeding.")