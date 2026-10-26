"""Native H1 Continue for saved replies (CM-01).

Continue asks the model to carry on from a saved reply. It reuses the saved
question's admission the same way Regenerate does, keeps the selected reply as
the final model context message, and settles ``saved reply + continuation`` as
a new sibling reply of the same question. The original rows are never changed.

The request opts in with ``metadata.tldw_continue_from_message_id`` together
with ``tldw_history_selection_v1`` and ``tldw_history_branch: true``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from fastapi import HTTPException

CONTINUE_METADATA_KEY = "tldw_continue_from_message_id"
UNSUPPORTED_CONTINUATION = "unsupported_history_continuation"
_REPLY_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,128}")


def _unsupported() -> HTTPException:
    return HTTPException(
        status_code=409,
        detail={"status": "unsupported_history_capability", "code": UNSUPPORTED_CONTINUATION},
    )


def parse_continue_reply_id(
    request_data: Any,
    *,
    conversation_id: str | None,
    regenerate_reply_id: Any,
    explicit_failed_retry: bool,
) -> str | None:
    """Return the saved reply to continue, or ``None`` when Continue was not requested.

    Continue is only defined for a saved, owner-selected reply, so it requires an
    existing conversation and a versioned history selection, and it cannot be
    combined with Regenerate, failed-turn Retry, legacy continuation controls or
    tool definitions.
    """
    metadata = getattr(request_data, "metadata", None)
    reply_id = metadata.get(CONTINUE_METADATA_KEY) if isinstance(metadata, dict) else None
    if reply_id is None:
        return None
    if not isinstance(reply_id, str) or not _REPLY_ID_PATTERN.fullmatch(reply_id):
        raise HTTPException(status_code=400, detail="A valid saved reply ID is required to continue a reply.")
    if (
        regenerate_reply_id is not None
        or explicit_failed_retry
        or not conversation_id
        or getattr(request_data, "tldw_continuation", None) is not None
    ):
        raise HTTPException(status_code=409, detail="Continue requires an existing saved reply and no other reply action.")
    if getattr(request_data, "tldw_history_selection_v1", None) is None:
        raise _unsupported()
    # A continuation extends saved text. Tool calls would be saved behind the
    # old reply text, and continued_reply_prefix already refuses tool replies.
    if getattr(request_data, "tools", None) or getattr(request_data, "tool_choice", None) not in (None, "none"):
        raise _unsupported()
    return reply_id


def continued_reply_prefix(item: Mapping[str, Any]) -> str:
    """Return the saved reply text that a continuation extends.

    Only plain text replies can be extended: a reply with tool calls, images or
    a stored content placeholder has no text the model could carry on.
    """
    from tldw_Server_API.app.core.Chat.history_selection import HistorySelectionError

    extra = item.get("extra_metadata") or {}
    text = item.get("message")
    if (
        not isinstance(text, str)
        or not text.strip()
        or item.get("images")
        or item.get("tool_calls") is not None
        or extra.get("function_call") is not None
        or extra.get("tool_call_id") is not None
        or extra.get("content_placeholder_reason") is not None
    ):
        raise HistorySelectionError(UNSUPPORTED_CONTINUATION)
    return text


def merge_continued_reply(payload: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    """Return an assistant save payload whose content is ``prefix`` plus the continuation."""
    merged = dict(payload)
    content = merged.get("content")
    if content is None:
        merged["content"] = prefix
    elif isinstance(content, str):
        merged["content"] = prefix + content
    elif isinstance(content, list):
        first = content[0] if content else None
        if isinstance(first, Mapping) and first.get("type") == "text" and isinstance(first.get("text"), str):
            merged["content"] = [{**first, "text": prefix + first["text"]}, *content[1:]]
        else:
            merged["content"] = [{"type": "text", "text": prefix}, *content]
    return merged
