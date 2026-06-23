"""LLM-based extraction of character memories from conversation history.

Provides:
- ``build_extraction_prompt`` — builds the system+user messages for memory extraction
- ``parse_extraction_response`` — parses the LLM JSON output
- ``deduplicate_memories`` — filters out near-duplicate entries
- ``extract_character_memories`` — end-to-end orchestrator
"""
from __future__ import annotations

import difflib
import json
import re
from typing import Any

from loguru import logger


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_CATEGORIES = frozenset({"fact", "relationship", "event", "preference"})

_EXTRACTION_SYSTEM_PROMPT = (
    "You are a memory analyst.  Given the recent conversation between {user_name} "
    "and {character_name}, extract noteworthy information about the user that the "
    "character should remember across sessions.\n\n"
    "For each memory, provide:\n"
    "- category: one of \"fact\", \"relationship\", \"event\", \"preference\"\n"
    "- content: concise statement (1-2 sentences max)\n"
    "- salience: float 0.0-1.0 indicating importance (0.8+ = very important)\n\n"
    "Already known memories (do not duplicate):\n{existing_memories}\n\n"
    "Respond ONLY with a JSON object:\n"
    "{{\"memories\": [{{\"category\": \"...\", \"content\": \"...\", \"salience\": 0.8}}]}}\n"
    "If nothing new, respond: {{\"memories\": []}}"
)


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_extraction_prompt(
    messages: list[dict[str, Any]],
    char_name: str,
    user_name: str,
    existing_memories: list[dict[str, Any]],
) -> tuple[str, str]:
    """Return ``(system_message, user_message)`` for the extraction LLM call.

    *messages* is a list of chat messages (dicts with ``role``/``sender`` and ``content``).
    """
    # Format existing memories as bullet list
    existing_lines: list[str] = []
    for mem in existing_memories:
        cat = mem.get("memory_type", "unknown")
        content = mem.get("content", "")
        existing_lines.append(f"- [{cat}] {content}")
    existing_block = "\n".join(existing_lines) if existing_lines else "(none)"

    system_msg = _EXTRACTION_SYSTEM_PROMPT.format(
        user_name=user_name,
        character_name=char_name,
        existing_memories=existing_block,
    )

    # Build conversation transcript for the user message
    transcript_lines: list[str] = []
    for msg in messages:
        role = msg.get("role") or msg.get("sender") or "unknown"
        content = msg.get("content", "")
        if role in ("user", "human"):
            speaker = user_name
        elif role in ("assistant", "ai"):
            speaker = char_name
        else:
            speaker = role
        transcript_lines.append(f"{speaker}: {content}")

    user_msg = (
        "Extract memories from this conversation:\n\n"
        + "\n".join(transcript_lines)
    )
    return system_msg, user_msg


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_extraction_response(raw_text: str) -> list[dict[str, Any]]:
    """Parse the LLM extraction response into a list of memory dicts.

    Handles:
    - Clean JSON arrays
    - JSON embedded in markdown code fences
    - Partial/malformed JSON gracefully
    """
    text = raw_text.strip()

    # Strip markdown code fences
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    def _coerce_memory_payload(parsed: Any) -> tuple[bool, list[dict[str, Any]]]:
        if isinstance(parsed, list):
            return True, _validate_memory_list(parsed)
        if isinstance(parsed, dict):
            memories = parsed.get("memories")
            if isinstance(memories, list):
                return True, _validate_memory_list(memories)
        return False, []

    # Try direct parse
    try:
        parsed = json.loads(text)
        is_memory_payload, memories = _coerce_memory_payload(parsed)
        if is_memory_payload:
            return memories
    except json.JSONDecodeError:
        pass

    # Try to recover a JSON object/array embedded in surrounding LLM text.
    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\[{]", text):
        try:
            parsed, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        is_memory_payload, memories = _coerce_memory_payload(parsed)
        if is_memory_payload:
            return memories

    logger.warning("Failed to parse extraction response as JSON memory payload")
    return []


def _validate_memory_list(items: list[Any]) -> list[dict[str, Any]]:
    """Filter and normalize extracted memory items."""
    result: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        category = str(item.get("category", "fact")).strip().lower()
        if category not in _VALID_CATEGORIES:
            category = "fact"
        try:
            salience = float(item.get("salience", 0.5))
            salience = max(0.0, min(1.0, salience))
        except (TypeError, ValueError):
            salience = 0.5
        result.append({
            "category": category,
            "content": content,
            "salience": salience,
        })
    return result


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

_SIMILARITY_THRESHOLD = 0.85


def deduplicate_memories(
    new_memories: list[dict[str, Any]],
    existing_memories: list[dict[str, Any]],
    threshold: float = _SIMILARITY_THRESHOLD,
) -> list[dict[str, Any]]:
    """Return only *new_memories* that are not near-duplicates of *existing_memories*."""
    existing_contents = [m.get("content", "") for m in existing_memories]
    unique: list[dict[str, Any]] = []
    for mem in new_memories:
        content = mem.get("content", "")
        is_dup = False
        for existing in existing_contents:
            if difflib.SequenceMatcher(None, content.lower(), existing.lower()).ratio() >= threshold:
                is_dup = True
                break
        if not is_dup:
            unique.append(mem)
    return unique


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class ExtractionResult:
    """Result of memory extraction with dedup stats."""
    __slots__ = ("unique", "total_parsed", "duplicates_skipped")

    def __init__(self, unique: list[dict[str, Any]], total_parsed: int, duplicates_skipped: int):
        self.unique = unique
        self.total_parsed = total_parsed
        self.duplicates_skipped = duplicates_skipped


def extract_character_memories(
    messages: list[dict[str, Any]],
    char_name: str,
    user_name: str,
    existing_memories: list[dict[str, Any]],
    *,
    api_endpoint: str,
    api_key: str | None = None,
    model: str | None = None,
    app_config: dict[str, Any] | None = None,
) -> ExtractionResult:
    """Run extraction end-to-end: build prompt → call LLM → parse → dedup.

    Returns an ``ExtractionResult`` with unique memories and dedup stats.
    Uses ``chat_api_call`` from the chat orchestrator.
    """
    from tldw_Server_API.app.core.Chat.chat_orchestrator import chat_api_call

    system_msg, user_msg = build_extraction_prompt(
        messages, char_name, user_name, existing_memories,
    )

    try:
        result = chat_api_call(
            api_endpoint=api_endpoint,
            messages_payload=[{"role": "user", "content": user_msg}],
            system_message=system_msg,
            api_key=api_key,
            model=model,
            temp=0.3,
            max_tokens=1024,
            response_format={"type": "json_object"},
            app_config=app_config,
        )
    except Exception:
        logger.exception("Character memory extraction LLM call failed")
        return ExtractionResult(unique=[], total_parsed=0, duplicates_skipped=0)

    # ``chat_api_call`` returns the response text (or a streaming iterator).
    raw_text = result if isinstance(result, str) else str(result)

    parsed = parse_extraction_response(raw_text)
    unique = deduplicate_memories(parsed, existing_memories)
    duplicates_skipped = len(parsed) - len(unique)
    logger.info(
        "Character memory extraction: parsed={}, unique={}, dupes={} (for {}/{})",
        len(parsed), len(unique), duplicates_skipped, char_name, user_name,
    )
    return ExtractionResult(
        unique=unique, total_parsed=len(parsed), duplicates_skipped=duplicates_skipped,
    )
