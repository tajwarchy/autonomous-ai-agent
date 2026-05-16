"""
Tool system tests.
Run from project root: python test_tools.py
No Ollama needed.
"""

from agent.logger_setup import setup_logging
setup_logging()

from tools.calculator import CalculatorTool
from tools.circuit_breaker import CircuitBreaker
from tools.file_reader import FileReaderTool
from tools.rate_limiter import RateLimiter
from tools.tool_router import ToolRouter
from tools.wikipedia import WikipediaTool
from tools.search import SearchTool
import pathlib

passed = 0
failed = 0


def check(label: str, condition: bool):
    global passed, failed
    status = "  PASS" if condition else "  FAIL"
    if condition:
        passed += 1
    else:
        failed += 1
    print(f"{status}  {label}")


# ── Calculator ─────────────────────────────────────────────────────────────
print("\n=== Calculator ===")
calc = CalculatorTool()
check("basic arithmetic", calc.run("2 + 2") == "4")
check("exponentiation", calc.run("2 ** 10") == "1024")
check("sqrt", calc.run("sqrt(144)") == "12.0")
check("pi available", "3.14" in calc.run("round(pi, 2)"))
check("division by zero", calc.run("1 / 0").startswith("ERROR:"))
check("blocked builtin", calc.run("__import__('os')").startswith("ERROR:"))
check("empty input", calc.run("").startswith("ERROR:"))
check("bad syntax", calc.run("2 +* 3").startswith("ERROR:"))

# ── File Reader ────────────────────────────────────────────────────────────
print("\n=== File Reader ===")
fr = FileReaderTool()
# Create a sample file for testing
sample = pathlib.Path("data/files/sample.txt")
sample.write_text("Hello from sample file.\nLine 2.")
check("reads existing file", "Hello" in fr.run("sample.txt"))
check("missing file error", fr.run("ghost.txt").startswith("ERROR:"))
check("path traversal blocked", fr.run("../config/config.yaml").startswith("ERROR:"))
check("absolute path blocked", fr.run("/etc/passwd").startswith("ERROR:"))
check("empty input", fr.run("").startswith("ERROR:"))

# ── Rate Limiter ───────────────────────────────────────────────────────────
print("\n=== Rate Limiter ===")
rl = RateLimiter()
# search limit is 5
for _ in range(5):
    ok, _ = rl.check("search")
    rl.increment("search")
ok, msg = rl.check("search")
check("blocks after limit", not ok)
check("error msg mentions tool", "search" in msg)
check("usage tracked", rl.usage()["search"]["calls"] == 5)

# ── Circuit Breaker ────────────────────────────────────────────────────────
print("\n=== Circuit Breaker ===")
cb = CircuitBreaker()
check("starts closed", cb.check("wikipedia")[0])
cb.record_failure("wikipedia", "timeout")
cb.record_failure("wikipedia", "timeout")
check("still closed at 2 failures", cb.check("wikipedia")[0])
cb.record_failure("wikipedia", "timeout")
ok, msg = cb.check("wikipedia")
check("opens at threshold (3)", not ok)
check("error msg mentions tool", "wikipedia" in msg)
check("summary reflects open state", cb.summary()["wikipedia"]["is_open"])

# success resets consecutive count but does NOT close once open
cb2 = CircuitBreaker()
cb2.record_failure("search", "err")
cb2.record_success("search")
cb2.record_failure("search", "err")
check("success resets consecutive count", not cb2.is_open("search"))

# ── Tool Router — full dispatch ────────────────────────────────────────────
print("\n=== Tool Router ===")
router = ToolRouter()

obs, lat = router.dispatch("calculator", "10 * 10")
check("calculator via router", obs == "100")
check("latency is float", isinstance(lat, float))

obs, lat = router.dispatch("file_reader", "sample.txt")
check("file_reader via router", "Hello" in obs)

obs, lat = router.dispatch("unknown_tool", "anything")
check("unknown tool error", obs.startswith("ERROR:"))

# Rate limit via router — use calculator (offline, never fails or trips circuit breaker)
router2 = ToolRouter()
for _ in range(10):
    router2.dispatch("calculator", "1 + 1")    # limit is 10
obs, _ = router2.dispatch("calculator", "1 + 1")
check("router enforces rate limit", "rate limit" in obs)

# Circuit breaker via router — simulate by directly tripping it
router3 = ToolRouter()
router3.circuit_breaker.record_failure("wikipedia", "test")
router3.circuit_breaker.record_failure("wikipedia", "test")
router3.circuit_breaker.record_failure("wikipedia", "test")
obs, _ = router3.dispatch("wikipedia", "Python")
check("router enforces circuit breaker", "circuit breaker" in obs)

# ── Wikipedia (live network call) ─────────────────────────────────────────
print("\n=== Wikipedia (network) ===")
wiki = WikipediaTool()
result = wiki.run("Python programming language")
check("returns content", len(result) > 50)
check("contains title", "Python" in result)
check("missing page error", wiki.run("xyzzy_nonexistent_page_abc123").startswith("ERROR:"))

# ── Search (live network call) ─────────────────────────────────────────────
print("\n=== Search (network) ===")
search = SearchTool()
result = search.run("what is the capital of France")
check("returns results", len(result) > 20)
check("empty query error", search.run("").startswith("ERROR:"))

# ── Summary ───────────────────────────────────────────────────────────────
print(f"\n{'='*40}")
print(f"Results: {passed} passed, {failed} failed")
if failed == 0:
    print("All tests passed. Tool system ready for Phase 4.")
else:
    print("Fix failures before proceeding.")