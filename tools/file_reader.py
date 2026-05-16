"""
File reader tool — sandboxed to a single allowed directory.

Path traversal attacks (e.g. "../../etc/passwd") are blocked by
resolving both the allowed directory and the requested path to their
absolute real paths and confirming the file is inside the sandbox.
"""

import logging
from pathlib import Path
from typing import Optional

import yaml

from tools.base import BaseTool

logger = logging.getLogger(__name__)

MAX_FILE_BYTES = 32 * 1024  # 32 KB — prevent giant files flooding context


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class FileReaderTool(BaseTool):

    def __init__(self, config: Optional[dict] = None):
        cfg = config or load_config()
        self._allowed_dir = Path(
            cfg["tools"]["file_reader"]["allowed_directory"]
        ).resolve()
        self._allowed_dir.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        return "file_reader"

    @property
    def description(self) -> str:
        return (
            "Read a text file from the allowed data directory. "
            "Input: filename only (e.g. 'notes.txt'), no path separators."
        )

    def run(self, input_str: str) -> str:
        filename = input_str.strip()
        if not filename:
            return "ERROR: file_reader requires a filename."

        # Block obvious traversal attempts early
        if "/" in filename or "\\" in filename or ".." in filename:
            return "ERROR: filename must not contain path separators or '..'. Pass a plain filename only."

        target = (self._allowed_dir / filename).resolve()

        # Sandbox check — target must be inside allowed_dir
        try:
            target.relative_to(self._allowed_dir)
        except ValueError:
            logger.warning(
                "file_reader_sandbox_violation",
                extra={"filename": filename, "resolved": str(target)},
            )
            return f"ERROR: access denied — '{filename}' is outside the allowed directory."

        if not target.exists():
            available = [f.name for f in self._allowed_dir.iterdir() if f.is_file()]
            return (
                f"ERROR: file '{filename}' not found. "
                f"Available files: {available if available else '(none)'}"
            )

        if not target.is_file():
            return f"ERROR: '{filename}' is not a regular file."

        size = target.stat().st_size
        if size > MAX_FILE_BYTES:
            return (
                f"ERROR: file '{filename}' is {size} bytes, "
                f"which exceeds the {MAX_FILE_BYTES}-byte limit."
            )

        try:
            content = target.read_text(encoding="utf-8")
            logger.debug(
                "file_reader_ok",
                extra={"filename": filename, "bytes": size},
            )
            return content

        except UnicodeDecodeError:
            return f"ERROR: '{filename}' is not a UTF-8 text file."
        except Exception as e:
            logger.warning("file_reader_error", extra={"filename": filename, "error": str(e)})
            return f"ERROR: could not read '{filename}' — {e}"