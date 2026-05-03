"""Tests for pure prompt database record helper functions."""

from tldw_Server_API.app.core.DB_Management.prompts_db_helpers import (
    build_structured_prompt_searchable_text,
    deserialize_prompt_record,
    normalize_keyword,
    normalize_text_for_search,
    serialize_prompt_definition,
)


def test_prompt_definition_serialization_round_trips_structured_payload() -> None:
    payload = {
        "blocks": [{"content": "Hello", "role": "user"}],
        "variables": [{"name": "topic"}],
    }

    serialized = serialize_prompt_definition(payload)
    hydrated = deserialize_prompt_record(
        {
            "name": "structured",
            "prompt_format": None,
            "prompt_definition_json": serialized,
        }
    )
    expected_serialized = "".join(
        [
            '{"blocks": [{"content": "Hello", "role": "user"}], ',
            '"variables": [{"name": "topic"}]}',
        ]
    )

    assert serialized == expected_serialized  # nosec B101
    assert hydrated == {  # nosec B101
        "name": "structured",
        "prompt_format": "legacy",
        "prompt_definition": payload,
    }


def test_prompt_definition_serialization_round_trips_raw_string_payload() -> None:
    payload = "raw template with {{var}}"

    serialized = serialize_prompt_definition(payload)
    hydrated = deserialize_prompt_record(
        {
            "name": "legacy-string",
            "prompt_format": None,
            "prompt_definition_json": serialized,
        }
    )

    assert serialized == payload  # nosec B101
    assert hydrated == {  # nosec B101
        "name": "legacy-string",
        "prompt_format": "legacy",
        "prompt_definition": payload,
    }


def test_structured_prompt_search_text_deduplicates_enabled_fields() -> None:
    payload = {
        "variables": [
            {"name": "topic", "label": "Topic", "description": "topic"},
            {"name": "topic"},
        ],
        "blocks": [
            {"enabled": True, "name": "intro", "role": "system", "content": "Topic"},
            {"enabled": False, "name": "disabled", "content": "hidden"},
        ],
    }

    assert build_structured_prompt_searchable_text(payload) == "topic\nTopic\nintro\nsystem"  # nosec B101


def test_prompt_keyword_and_search_normalization_match_legacy_behavior() -> None:
    assert normalize_keyword("  Alpha\t Beta \n Gamma  ") == "Alpha Beta Gamma"  # nosec B101
    assert normalize_text_for_search("İSTANBUL Cafe\u0301") == "istanbul cafe"  # nosec B101
