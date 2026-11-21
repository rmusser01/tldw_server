"""Public-text and legacy flashcard selection for chat knowledge saves."""

import json
import re
from collections.abc import Mapping
from typing import Any


class KnowledgeFlashcardError(ValueError):
    """The selected public turn cannot supply a complete flashcard."""


_REASONING_DELIMITER = re.compile(r"<(/?)(think|reason|reasoning|thought)>", re.IGNORECASE)


def split_reasoning_text(value: str) -> tuple[str, str]:
    """Separate answer and thinking text, preserving characters in each channel."""
    answer: list[str] = []
    thinking: list[str] = []
    open_tags: list[str] = []
    start = 0
    for delimiter in _REASONING_DELIMITER.finditer(value):
        name = delimiter[2].lower()
        if not delimiter[1]:
            if not open_tags:
                answer.append(value[start:delimiter.start()])
                start = delimiter.end()
            open_tags.append(name)
        elif open_tags and name == open_tags[-1]:
            open_tags.pop()
            if not open_tags:
                thinking.append(value[start:delimiter.start()])
                start = delimiter.end()
        elif not open_tags:
            # Old unescaped traces can contain an early closing tag.
            (thinking if thinking else answer).append(value[start:delimiter.start()])
            start = delimiter.end()
    (thinking if open_tags else answer).append(value[start:])
    return "".join(answer), "".join(thinking)


def strip_reasoning_blocks(value: str) -> str:
    """Remove tagged thinking without changing any visible answer characters."""
    return split_reasoning_text(value)[0]


def visible_knowledge_text(value: str) -> str:
    """Exclude closed and unfinished model reasoning blocks from saved content."""
    return strip_reasoning_blocks(value).strip()


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
