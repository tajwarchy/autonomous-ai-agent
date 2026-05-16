"""
Safe calculator tool.

Uses a restricted eval with an explicit allowlist of names.
No imports, no builtins, no file access possible.
"""

import logging
import math
from typing import Optional

import yaml

from tools.base import BaseTool

logger = logging.getLogger(__name__)


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# Safe math functions always available regardless of config
_MATH_NAMESPACE = {
    "abs": abs, "round": round, "min": min, "max": max,
    "sum": sum, "pow": pow, "len": len,
    # math module functions
    "sqrt": math.sqrt, "floor": math.floor, "ceil": math.ceil,
    "log": math.log, "log10": math.log10, "log2": math.log2,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "pi": math.pi, "e": math.e,
    "factorial": math.factorial,
}


class CalculatorTool(BaseTool):

    def __init__(self, config: Optional[dict] = None):
        cfg = config or load_config()
        # Merge config-specified names with the fixed math namespace
        extra = cfg["tools"]["calculator"].get("allowed_names", [])
        self._namespace = {**_MATH_NAMESPACE}
        # Only allow names that are already in _MATH_NAMESPACE for safety
        for name in extra:
            if name in _MATH_NAMESPACE:
                self._namespace[name] = _MATH_NAMESPACE[name]

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return (
            "Evaluate a math expression. "
            "Supports: +,-,*,/,**,sqrt,log,sin,cos,tan,pi,e,round,abs,min,max. "
            "Input: a plain math expression string, e.g. 'sqrt(144) + 2**8'."
        )

    def run(self, input_str: str) -> str:
        # Strip surrounding quotes the LLM sometimes adds around the expression
        expr = input_str.strip().strip("\"'")
        if not expr:
            return "ERROR: calculator requires a non-empty expression."

        try:
            # compile first to catch syntax errors cleanly
            code = compile(expr, "<calculator>", "eval")

            # Reject any name not in our allowlist
            for name in code.co_names:
                if name not in self._namespace:
                    return (
                        f"ERROR: name '{name}' is not allowed. "
                        f"Allowed: {sorted(self._namespace.keys())}"
                    )

            result = eval(code, {"__builtins__": {}}, self._namespace)  # noqa: S307
            output = str(result)

            logger.debug(
                "calculator_tool_ok",
                extra={"expr": expr, "result": output},
            )
            return output

        except ZeroDivisionError:
            return "ERROR: division by zero."
        except SyntaxError as e:
            return f"ERROR: invalid expression syntax — {e}"
        except Exception as e:
            logger.warning("calculator_tool_error", extra={"expr": expr, "error": str(e)})
            return f"ERROR: calculation failed — {e}"