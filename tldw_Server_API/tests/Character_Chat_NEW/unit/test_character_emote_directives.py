from __future__ import annotations

import json
from unittest import TestCase
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Character_Chat.emote_directives import (
    EMOTE_EVENT_LIMIT,
    parse_character_emote_directives,
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
    ],
)
def test_validate_emote_events_rejects_invalid_metadata(events):
    with pytest.raises(ValueError):
        validate_emote_events(events)
