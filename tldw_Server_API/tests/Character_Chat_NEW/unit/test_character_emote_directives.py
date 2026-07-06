from __future__ import annotations

import json
from unittest import TestCase
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Character_Chat.emote_directives import (
    EMOTE_EVENT_LIMIT,
    parse_character_emote_directives,
    resolve_character_emote_completion,
    validate_emote_events,
)


FIXTURE_PATH = (
    Path(__file__).resolve().parents[4]
    / "apps/packages/ui/src/utils/__fixtures__/character-emote-directives.json"
)
CASE = TestCase()


def _event_dicts(result):
    return [event.model_dump() for event in result.events]


@pytest.mark.parametrize("fixture", json.loads(FIXTURE_PATH.read_text()))
def test_parse_character_emote_directives_matches_shared_fixture(fixture):
    result = parse_character_emote_directives(fixture["input"])

    CASE.assertEqual(result.clean_text, fixture["clean_text"])
    CASE.assertEqual(_event_dicts(result), fixture["events"])


def test_parse_character_emote_directives_counts_offsets_like_js_strings():
    result = parse_character_emote_directives("😀\nEmote: smug\nDone.")

    CASE.assertEqual(result.clean_text, "😀\nDone.")
    CASE.assertEqual(_event_dicts(result), [{"state": "smug", "at_char": 3}])


def test_validate_emote_events_accepts_valid_event():
    events = validate_emote_events([{"state": "smug", "at_char": 0}])

    CASE.assertEqual(
        [event.model_dump() for event in events],
        [{"state": "smug", "at_char": 0}],
    )


@pytest.mark.parametrize(
    "events",
    [
        [{"state": "../../bad", "at_char": 0}],
        [{"state": "smug", "at_char": -1}],
        [{"state": f"state-{index}", "at_char": 0} for index in range(EMOTE_EVENT_LIMIT + 1)],
        [{"state": "smug", "at_char": "1"}],
        [{"state": "smug", "at_char": 1.0}],
        [{"state": "smug", "at_char": True}],
        [{"state": b"smug", "at_char": 0}],
        [{"state": "smug", "at_char": 0, "extra": "ignored"}],
    ],
)
def test_validate_emote_events_rejects_invalid_metadata(events):
    with pytest.raises(ValueError):
        validate_emote_events(events)


def test_resolve_character_emote_completion_prefers_explicit_directives():
    result = resolve_character_emote_completion(
        "Emote: smug\nFine.",
        fallback_mood_label="happy",
        fallback_mood_confidence=0.9,
        fallback_mood_topic="fallback",
    )

    CASE.assertEqual(result.clean_text, "Fine.")
    CASE.assertEqual(result.mood_label, "smug")
    CASE.assertIsNone(result.mood_confidence)
    CASE.assertIsNone(result.mood_topic)
    CASE.assertEqual(
        [event.model_dump() for event in result.emote_events],
        [{"state": "smug", "at_char": 0}],
    )


def test_resolve_character_emote_completion_preserves_fallback_without_directives():
    result = resolve_character_emote_completion(
        "Fine.",
        fallback_mood_label="happy",
        fallback_mood_confidence=0.9,
        fallback_mood_topic="fallback",
    )

    CASE.assertEqual(result.clean_text, "Fine.")
    CASE.assertEqual(result.mood_label, "happy")
    CASE.assertEqual(result.mood_confidence, 0.9)
    CASE.assertEqual(result.mood_topic, "fallback")
    CASE.assertEqual(result.emote_events, [])
