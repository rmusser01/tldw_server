"""
SSE line helpers and provider line normalization.

Highlights:
- Normalization drops provider control lines (`event:`, `id:`, `retry:`) and comments by default.
- To preserve provider control lines, set the global env `STREAM_PROVIDER_CONTROL_PASSTHRU=1`
  or pass `provider_control_passthru=True` to iterators/streams (per-endpoint override).
- Unknown/dropped control/comment lines are logged at debug level to aid troubleshooting.
- Use `sse_done()` to emit a single terminal `[DONE]` marker; do not forward provider DONE lines.
"""

import contextlib
import json
from collections.abc import Iterable
from typing import Any, Callable, Optional

from loguru import logger

# SSE control fields (comments, ":" lines, are handled separately).
SSE_CONTROL_FIELD_PREFIXES = ("event:", "id:", "retry:")


def finalize_stream(response: Optional[Any], done_already: bool = False) -> Iterable[str]:
    """Yield a final DONE (if not already sent) and always close the response safely.

    This is a tiny DRY helper to unify end-of-stream handling across providers.
    """
    try:
        if not done_already:
            yield sse_done()
    finally:
        try:
            if response is not None:
                response.close()
        except Exception as response_close_error:
            logger.debug(
                "SSE finalize_stream failed to close response; error_type={}",
                type(response_close_error).__name__,
            )


def sse_data(payload: dict[str, Any]) -> str:
    """Return an SSE data line with a blank line terminator."""
    return f"data: {json.dumps(payload)}\n\n"


def sse_event(name: str, payload: dict[str, Any]) -> str:
    """Return a named SSE event frame (``event:`` line plus one data line)."""
    return f"event: {name}\ndata: {json.dumps(payload)}\n\n"


def sse_done() -> str:
    """Return the SSE end-of-stream sentinel."""
    return "data: [DONE]\n\n"


def ensure_sse_line(line: str) -> str:
    """Ensure an incoming data line is terminated as SSE (with a blank line)."""
    if not line.endswith("\n\n"):
        if line.endswith("\n"):
            return line + "\n"
        return line + "\n\n"
    return line


def ensure_sse_control_line(line: str) -> str:
    """Ensure a control line ends with a single newline (no dispatch blank line)."""
    if line.endswith("\n\n"):
        return line
    if line.endswith("\n"):
        return line
    return line + "\n"


def openai_delta_chunk(text: str) -> str:
    """Wrap a plain text delta into an OpenAI-compatible SSE chunk."""
    return sse_data({"choices": [{"delta": {"content": text}}]})


def is_done_line(line: str) -> bool:
    """Return True when the raw line represents the [DONE] sentinel.

    Deliberately case-insensitive and tolerant of any whitespace after ``data:``
    and of leading BOM/zero-width characters: a provider DONE in any spelling
    must be recognised so it is suppressed rather than forwarded next to our own
    terminal ``sse_done()`` (single-terminal-DONE contract, ADR-025).
    """
    s = line.lstrip("\ufeff\u200b\u200c\u200d\u2060").strip().lower()
    return s.startswith("data:") and s[len("data:") :].strip() == "[done]"


def normalize_provider_line(
    line: str,
    *,
    provider_control_passthru: bool = False,
    control_filter: Optional[Callable[[str, str], Optional[tuple[str, str]]]] = None,
) -> Optional[str]:
    """
    Normalize a raw provider SSE line into a chunk we can forward.

    - Ignores control fields such as event:/id:/retry: and comment lines.
    - Preserves proper data frames using SSE framing.
    - Falls back to wrapping unexpected payload lines as OpenAI deltas.
    """
    stripped = line.strip()
    if not stripped:
        return None

    lower = stripped.lower()
    for prefix in SSE_CONTROL_FIELD_PREFIXES:
        if lower.startswith(prefix):
            name, value = stripped.split(":", 1)
            name = name.strip()
            value = value.strip()
            if provider_control_passthru:
                if control_filter is not None:
                    try:
                        mapped = control_filter(name, value)
                    except Exception:
                        mapped = (name, value)
                    if mapped is None:
                        return None
                    name, value = mapped
                # Preserve control line without dispatching a blank line
                return ensure_sse_control_line(f"{name}: {value}")
            with contextlib.suppress(Exception):
                logger.debug(f"Dropping provider control line: {stripped}")
            return None
    if stripped.startswith(":"):
        with contextlib.suppress(Exception):
            logger.debug(f"Dropping provider comment line: {stripped}")
        return None

    if stripped.startswith("data:"):
        return ensure_sse_line(stripped)

    return openai_delta_chunk(stripped)
