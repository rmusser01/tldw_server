"""Public-text and legacy flashcard selection for chat knowledge saves."""

import json
import re
from collections.abc import Mapping
from typing import Any


class KnowledgeFlashcardError(ValueError):
    """The selected public turn cannot supply a complete flashcard."""


def visible_knowledge_text(value: str) -> str:
    """Exclude closed and unfinished model reasoning blocks from saved content."""
    return re.sub(
        r"<(think|reason|reasoning|thought)>.*?(?:</\1>|$)",
        "",
        value,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()


def _public_turn_text(value: Any) -> str:
    """Accept plain stored text, excluding opaque structured message payloads."""
    if not isinstance(value, str):
        return ""
    public = visible_knowledge_text(value)
    try:
        parsed = json.loads(public)
    except json.JSONDecodeError:
        return public
    except RecursionError:
        return ""
    return "" if isinstance(parsed, (dict, list)) else public


def resolve_knowledge_flashcard(
    *,
    snippet: str,
    front: str | None,
    back: str | None,
    message: Mapping[str, Any] | None,
    parent: Mapping[str, Any] | None,
) -> tuple[str, str]:
    """Use reviewed fields or the verified parent question and public answer excerpt.

    The caller must verify conversation, message and parent ownership/scope before
    calling. Legacy requests cannot infer a card from unlinked or private text.
    """
    if front is not None or back is not None:
        if not front or not back:
            raise KnowledgeFlashcardError("A flashcard requires both a question and an answer")
        return front, back

    if (
        not message
        or not parent
        or message.get("sender") != "assistant"
        or parent.get("sender") != "user"
        or not parent.get("id")
        or message.get("parent_message_id") != parent.get("id")
        or message.get("conversation_id") != parent.get("conversation_id")
    ):
        raise KnowledgeFlashcardError("A legacy flashcard requires a linked public question and answer")

    question = _public_turn_text(parent.get("content"))
    answer = _public_turn_text(message.get("content"))
    excerpt = visible_knowledge_text(snippet)
    if not question or not excerpt or " ".join(excerpt.split()) not in " ".join(answer.split()):
        raise KnowledgeFlashcardError("A legacy flashcard requires an excerpt of the linked public answer")
    return question, excerpt
