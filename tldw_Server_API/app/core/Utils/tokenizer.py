"""
Simple token counting utilities with configurable strategies.

Strategies:
- whitespace: count whitespace-delimited tokens
- char_approx: approximate by character length (≈4 chars per token)

Configuration via settings (dict-like):
  TOKEN_ESTIMATOR_MODE: "whitespace" | "char_approx"
  TOKEN_CHAR_APPROX_DIVISOR: int (default 4)
"""

from math import ceil
from typing import Optional

from tldw_Server_API.app.core.Utils.image_validation import DATA_URI_REDACT_RE

try:
    # settings is dict-like per project patterns
    from tldw_Server_API.app.core.config import settings  # type: ignore
except Exception:
    settings = {}


def count_tokens(text: Optional[str], strategy: Optional[str] = None) -> int:
    """
    Count tokens using the configured strategy.

    Args:
        text: input text (None treated as empty)
        strategy: override strategy ("whitespace" | "char_approx")

    Returns:
        Estimated token count as int
    """
    if not text:
        return 0

    chosen = (strategy or settings.get("TOKEN_ESTIMATOR_MODE") or "whitespace").lower()

    if chosen == "char_approx":
        divisor = settings.get("TOKEN_CHAR_APPROX_DIVISOR", 4)
        try:
            div = int(divisor) if int(divisor) > 0 else 4
        except Exception:
            div = 4
        return int(ceil(len(text) / div))

    # default: whitespace
    return len(text.split())


def redact_data_uris(text: str) -> str:
    """Redact data:image payloads: the chat path's one redactor, for fingerprints and estimates."""
    return DATA_URI_REDACT_RE.sub(r"\1<omitted>", text)


def estimate_segment_tokens(text: str) -> int:
    """The chat path's one prompt-size estimate: data:image payloads redacted, 4 chars/token rounded up.

    Rate limits, quota and billing reservations, queue admission, the prompt-cost guardrail,
    the post-call stream estimates and every recorded provider call's estimate row
    (usage_accounting) all use it (TASK-13843). Rounded up because every one of them is
    safer over than under.
    """
    if not text:
        return 0
    sanitized = redact_data_uris(text)
    return max(0, (len(sanitized) + 3) // 4)
