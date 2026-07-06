from __future__ import annotations

import re

from pydantic import BaseModel, Field, ValidationError

EMOTE_EVENT_LIMIT = 5
CHARACTER_EMOTE_STATE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")

_DIRECTIVE_PATTERN = re.compile(r"^emote:(.*)$", re.IGNORECASE)


class CharacterEmoteEvent(BaseModel):
    state: str = Field(..., min_length=1, max_length=40)
    at_char: int = Field(..., ge=0)


class CharacterEmoteParseResult(BaseModel):
    clean_text: str
    events: list[CharacterEmoteEvent]


class CharacterEmoteCompletionResult(BaseModel):
    clean_text: str
    mood_label: str | None
    mood_confidence: float | None
    mood_topic: str | None
    emote_events: list[CharacterEmoteEvent]


def normalize_character_emote_state(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = re.sub(r"\s+", "-", value.strip().lower())
    return normalized if CHARACTER_EMOTE_STATE_PATTERN.fullmatch(normalized) else None


def _iter_lines(text: str):
    index = 0
    while index < len(text):
        newline_index = text.find("\n", index)
        if newline_index == -1:
            yield text[index:], ""
            break
        yield text[index:newline_index], "\n"
        index = newline_index + 1


def _is_fence_line(line: str) -> bool:
    return line.strip().startswith("```")


def _parse_directive_state(line: str) -> str | None | object:
    match = _DIRECTIVE_PATTERN.fullmatch(line.strip())
    return normalize_character_emote_state(match.group(1)) if match else _NO_DIRECTIVE


_NO_DIRECTIVE = object()


def parse_character_emote_directives(text: str) -> CharacterEmoteParseResult:
    clean_parts: list[str] = []
    events: list[CharacterEmoteEvent] = []
    in_fence = False
    last_state: str | None = None
    clean_length = 0

    for line, separator in _iter_lines(text):
        if _is_fence_line(line):
            visible = line + separator
            clean_parts.append(visible)
            clean_length += len(visible)
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
        clean_length += len(visible)

    return CharacterEmoteParseResult(clean_text="".join(clean_parts), events=events)


def validate_emote_events(events: object) -> list[CharacterEmoteEvent]:
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


def resolve_character_emote_completion(
    text: str,
    *,
    fallback_mood_label: str | None = None,
    fallback_mood_confidence: float | None = None,
    fallback_mood_topic: str | None = None,
) -> CharacterEmoteCompletionResult:
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
