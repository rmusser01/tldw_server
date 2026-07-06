"""Utilities for explicit character emote directives in assistant text."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError

EMOTE_EVENT_LIMIT = 5
EMOTE_PROMPT_STATE_LIMIT = 25
CHARACTER_EMOTE_STATE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")

_DIRECTIVE_PATTERN = re.compile(r"^emote:(.*)$", re.IGNORECASE)


class CharacterEmoteEvent(BaseModel):
    """Accepted emote state transition at a UTF-16 character offset."""

    model_config = ConfigDict(extra="forbid")

    state: str = Field(
        ...,
        min_length=1,
        max_length=40,
        pattern=CHARACTER_EMOTE_STATE_PATTERN.pattern,
        strict=True,
    )
    at_char: int = Field(..., ge=0, strict=True)


class CharacterEmoteParseResult(BaseModel):
    """Visible assistant text and emote events parsed from it."""

    clean_text: str
    events: list[CharacterEmoteEvent]


class CharacterEmoteCompletionResult(BaseModel):
    """Resolved assistant text plus explicit-or-fallback mood metadata."""

    clean_text: str
    mood_label: str | None
    mood_confidence: float | None
    mood_topic: str | None
    emote_events: list[CharacterEmoteEvent]


def normalize_character_emote_state(value: object) -> str | None:
    """Return a safe emote slug, or None when the value is not usable."""

    if not isinstance(value, str):
        return None
    normalized = re.sub(r"\s+", "-", value.strip().lower())
    return normalized if CHARACTER_EMOTE_STATE_PATTERN.fullmatch(normalized) else None


def _iter_lines(text: str) -> Iterator[tuple[str, str]]:
    """Yield text lines with their original newline separator."""

    index = 0
    while index < len(text):
        newline_index = text.find("\n", index)
        if newline_index == -1:
            yield text[index:], ""
            break
        yield text[index:newline_index], "\n"
        index = newline_index + 1


def _is_fence_line(line: str) -> bool:
    """Return whether the line toggles a Markdown code fence."""

    return line.strip().startswith("```")


def _js_string_length(value: str) -> int:
    """Return JavaScript string length measured in UTF-16 code units."""

    return len(value.encode("utf-16-le", errors="surrogatepass")) // 2


def _parse_directive_state(line: str) -> str | None | object:
    """Parse one standalone Emote directive line."""

    match = _DIRECTIVE_PATTERN.fullmatch(line.strip())
    return normalize_character_emote_state(match.group(1)) if match else _NO_DIRECTIVE


_NO_DIRECTIVE = object()


def parse_character_emote_directives(text: str) -> CharacterEmoteParseResult:
    """Strip standalone Emote directives and return accepted state events."""

    clean_parts: list[str] = []
    events: list[CharacterEmoteEvent] = []
    in_fence = False
    last_state: str | None = None
    clean_length = 0

    for line, separator in _iter_lines(text):
        if _is_fence_line(line):
            visible = line + separator
            clean_parts.append(visible)
            clean_length += _js_string_length(visible)
            in_fence = not in_fence
            continue

        if not in_fence:
            state = _parse_directive_state(line)
            if state is not _NO_DIRECTIVE:
                if (
                    isinstance(state, str)
                    and state != last_state
                    and len(events) < EMOTE_EVENT_LIMIT
                ):
                    events.append(CharacterEmoteEvent(state=state, at_char=clean_length))
                    last_state = state
                continue

        visible = line + separator
        clean_parts.append(visible)
        clean_length += _js_string_length(visible)

    return CharacterEmoteParseResult(clean_text="".join(clean_parts), events=events)


def validate_emote_events(events: object) -> list[CharacterEmoteEvent]:
    """Validate client-provided emote event shape and count."""

    if not isinstance(events, list):
        raise ValueError("emote events must be a list")
    if len(events) > EMOTE_EVENT_LIMIT:
        raise ValueError("too many emote events")

    validated: list[CharacterEmoteEvent] = []
    for event in events:
        try:
            parsed = (
                event
                if isinstance(event, CharacterEmoteEvent)
                else CharacterEmoteEvent.model_validate(event)
            )
        except ValidationError as exc:
            raise ValueError("invalid emote event") from exc

        if not CHARACTER_EMOTE_STATE_PATTERN.fullmatch(parsed.state):
            raise ValueError("invalid emote state")
        validated.append(parsed)

    return validated


def validate_emote_events_for_text(
    events: object,
    text: str,
) -> list[CharacterEmoteEvent]:
    """Validate emote events against sanitized assistant text length."""

    validated = validate_emote_events(events)
    max_offset = _js_string_length(text)
    previous_offset = -1
    for event in validated:
        if event.at_char > max_offset:
            raise ValueError("emote event offset exceeds assistant content length")
        if event.at_char < previous_offset:
            raise ValueError("emote event offsets must be non-decreasing")
        previous_offset = event.at_char
    return validated


def resolve_character_emote_completion(
    text: str,
    *,
    fallback_mood_label: str | None = None,
    fallback_mood_confidence: float | None = None,
    fallback_mood_topic: str | None = None,
) -> CharacterEmoteCompletionResult:
    """Resolve visible text and mood metadata from an assistant completion."""

    parsed = parse_character_emote_directives(text)
    if parsed.events:
        return CharacterEmoteCompletionResult(
            clean_text=parsed.clean_text,
            mood_label=parsed.events[-1].state,
            mood_confidence=None,
            mood_topic=None,
            emote_events=parsed.events,
        )

    return CharacterEmoteCompletionResult(
        clean_text=parsed.clean_text,
        mood_label=fallback_mood_label,
        mood_confidence=fallback_mood_confidence,
        mood_topic=fallback_mood_topic,
        emote_events=[],
    )


def extract_character_mood_image_states(character: Mapping[str, object]) -> list[str]:
    """Return safe custom emote states from character extension mood images."""

    extensions: object = character.get("extensions") if isinstance(character, Mapping) else None
    if isinstance(extensions, str):
        try:
            extensions = json.loads(extensions)
        except (TypeError, ValueError):
            extensions = {}
    if not isinstance(extensions, Mapping):
        return []

    tldw_source = extensions.get("tldw")
    tldw = tldw_source if isinstance(tldw_source, Mapping) else {}
    for source in (
        tldw.get("mood_images"),
        tldw.get("moodImages"),
        extensions.get("mood_images"),
        extensions.get("moodImages"),
    ):
        if not isinstance(source, Mapping):
            continue
        states: list[str] = []
        seen: set[str] = set()
        for raw_state in source:
            state = normalize_character_emote_state(raw_state)
            if state and state not in seen:
                states.append(state)
                seen.add(state)
        return states
    return []


def _format_prompt_state_list(states: list[str]) -> str:
    """Format a capped emote state list for the system prompt."""

    visible_states = states[:EMOTE_PROMPT_STATE_LIMIT]
    hidden_count = max(0, len(states) - len(visible_states))
    suffix = f" (+{hidden_count} more)" if hidden_count else ""
    return f"{', '.join(visible_states)}{suffix}"


def append_character_emote_prompt_instruction(
    sys_text: str,
    character: Mapping[str, object],
) -> str:
    """Append the Emote directive instruction to a character system prompt."""

    states = extract_character_mood_image_states(character)
    prefer = (
        f" Prefer these available states: {_format_prompt_state_list(states)}."
        if states
        else ""
    )
    instruction = (
        "When the character expression should change, emit a standalone line exactly like "
        "`Emote: <state>`."
        f"{prefer} Do not emit an emote after every sentence."
    )
    base = sys_text.strip()
    return f"{base}\n\n{instruction}" if base else instruction
