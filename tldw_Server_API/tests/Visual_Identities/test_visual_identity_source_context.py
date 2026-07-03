from __future__ import annotations

import base64

import pytest

from tldw_Server_API.app.core.Visual_Identities.source_context import (
    canonicalize_source_context,
    source_context_payload_hash,
)


def test_canonical_source_context_sorts_keys_and_preserves_short_metadata() -> None:
    context = canonicalize_source_context(
        {
            "vn_slot_label": "Happy",
            "generated_file_id": 42,
            "source_feature": "vn_assets",
        }
    )

    assert context == {
        "generated_file_id": 42,
        "source_feature": "vn_assets",
        "vn_slot_label": "Happy",
    }
    assert source_context_payload_hash(context) == source_context_payload_hash(
        {
            "source_feature": "vn_assets",
            "vn_slot_label": "Happy",
            "generated_file_id": 42,
        }
    )
    assert (
        source_context_payload_hash(context)
        == "05b05c3a10b7ea7b1e9bdd1f3d6125b6e5dce1e68ee9bbc7d8df364e55c1657c"
    )
    assert source_context_payload_hash(context) != source_context_payload_hash(
        {
            "generated_file_id": 43,
            "source_feature": "vn_assets",
            "vn_slot_label": "Happy",
        }
    )


def test_source_context_allows_valid_nested_list_and_mapping_metadata() -> None:
    context = canonicalize_source_context(
        {
            "source_feature": "vn_assets",
            "meta": {
                "slots": [
                    {"key": "happy", "weight": 1.0},
                    {"key": "sad", "enabled": False},
                ],
                "source_refs": ["vn_asset_item:29", None],
            },
        }
    )

    assert context == {
        "meta": {
            "slots": [
                {"key": "happy", "weight": 1.0},
                {"enabled": False, "key": "sad"},
            ],
            "source_refs": ["vn_asset_item:29", None],
        },
        "source_feature": "vn_assets",
    }


def test_source_context_allows_short_prompt_references() -> None:
    context = canonicalize_source_context(
        {
            "prompt_id": "prompt-123",
            "prompt_ref": "vn-pack/maya/happy",
            "prompt_label": "Maya happy sprite",
        }
    )

    assert context == {
        "prompt_id": "prompt-123",
        "prompt_label": "Maya happy sprite",
        "prompt_ref": "vn-pack/maya/happy",
    }


@pytest.mark.parametrize("value", [[], "text", 7, None])
def test_source_context_rejects_non_object_roots(value: object) -> None:
    with pytest.raises(ValueError, match="invalid_source_context"):
        canonicalize_source_context(value)


@pytest.mark.parametrize(
    "context",
    [
        {"prompt": "draw a full character sprite"},
        {"Prompt": "draw a full character sprite"},
        {"user_prompt": "draw a full character sprite"},
        {"meta": {"System_Prompt": "draw a full character sprite"}},
        {"image": "data:image/png;base64,AAAA"},
        {"blob": "QUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFB"},
        {"blob": base64.b64encode(b"A" * 32).decode("ascii").rstrip("=")},
        {"blob": base64.urlsafe_b64encode(b"\xff" * 32).decode("ascii").rstrip("=")},
    ],
)
def test_source_context_rejects_prompt_text_and_payloads(
    context: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="invalid_source_context"):
        canonicalize_source_context(context)


@pytest.mark.parametrize(
    "context",
    [
        {"binary": b"\x00\x01"},
        {"too_long": "x" * 513},
        {"x" * 65: "value"},
        {f"k{i}": i for i in range(51)},
        {"a": {"b": {"c": {"d": {"e": "too deep"}}}}},
        {f"k{i}": "x" * 512 for i in range(17)},
    ],
)
def test_source_context_rejects_bounds_violations(
    context: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="invalid_source_context"):
        canonicalize_source_context(context)
