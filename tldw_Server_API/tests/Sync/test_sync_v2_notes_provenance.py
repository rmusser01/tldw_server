"""Strict bounded canonical provenance and portable marker contract."""

import copy
import pytest
from hypothesis import given, strategies as st


def payload():
    return {
        "origin": "reviewed_sources",
        "research": {
            "workspace_id": "ws",
            "import_id": "imp",
            "sources": [
                {
                    "mediaId": 1,
                    "evidence": {
                        "importId": "imp",
                        "threadId": None,
                        "snapshot": True,
                        "sources": [
                            {
                                "title": "Source",
                                "excerpt": "Evidence",
                                "type": "website",
                                "originalId": None,
                                "mediaId": None,
                                "sourceType": None,
                            }
                        ],
                    },
                }
            ],
        },
    }


def test_contract_and_canonical_hash():
    from tldw_Server_API.app.core.Sync.v2.notes_provenance_contract import (
        validate_notes_provenance_payload,
        notes_provenance_object_hash,
    )

    value = payload()
    assert validate_notes_provenance_payload(value) == value
    assert notes_provenance_object_hash(value) == notes_provenance_object_hash(dict(reversed(list(value.items()))))
    assert notes_provenance_object_hash(value) != notes_provenance_object_hash(value, deleted=True)


@pytest.mark.parametrize(
    "field,value",
    [
        ("origin", "invented"),
        ("trust_state", "invented"),
        ("thread_id", ""),
        ("thread_id", "x" * 513),
        ("metadata", {"api_key": "secret"}),
    ],
)
def test_reject_top_level_fields(field, value):
    from tldw_Server_API.app.core.Sync.v2.notes_provenance_contract import validate_notes_provenance_payload

    data = {"origin": "knowledge_qa", field: value}
    with pytest.raises(ValueError):
        validate_notes_provenance_payload(data)


@pytest.mark.parametrize(
    "field,value",
    [
        ("mediaId", True),
        ("mediaId", 0),
        ("mediaId", 2**53),
        ("excerpt", "x" * 100001),
        ("url", "x" * 4097),
        ("token", "secret"),
        ("pageNumber", 1.0),
    ],
)
def test_reject_source_fields(field, value):
    from tldw_Server_API.app.core.Sync.v2.notes_provenance_contract import validate_notes_provenance_payload

    data = payload()
    data["research"]["sources"][0]["evidence"]["sources"][0][field] = value
    with pytest.raises(ValueError):
        validate_notes_provenance_payload(data)


def test_array_and_total_bounds():
    from tldw_Server_API.app.core.Sync.v2.notes_provenance_contract import validate_notes_provenance_payload

    data = payload()
    source = data["research"]["sources"][0]
    data["research"]["sources"] = [copy.deepcopy(source) for _ in range(101)]
    with pytest.raises(ValueError):
        validate_notes_provenance_payload(data)
    data["research"]["sources"] = [copy.deepcopy(source) for _ in range(11)]
    for source in data["research"]["sources"]:
        source["evidence"]["sources"][0]["excerpt"] = "x" * 100000
    with pytest.raises(ValueError):
        validate_notes_provenance_payload(data)


def test_marker_roundtrip_preserves_ordinary_comments():
    from tldw_Server_API.app.core.Sync.v2.notes_provenance_contract import (
        read_notes_provenance,
        retain_notes_provenance,
        strip_notes_provenance,
    )

    body = "Text\n<!-- ordinary -->"
    marked = retain_notes_provenance(body, payload())
    assert read_notes_provenance(marked) == payload()
    assert strip_notes_provenance(marked) == body
    malformed = "<!-- tldw-knowledge:v1:%7Bbad -->"
    assert read_notes_provenance(malformed) is None
    assert strip_notes_provenance(malformed) == malformed


@pytest.mark.parametrize("field", ["sources", "include_note_ids", "include_media_ids", "enable_web_fallback"])
def test_scope_rejects_explicit_null(field):
    from tldw_Server_API.app.core.Sync.v2.notes_provenance_contract import validate_notes_provenance_payload

    with pytest.raises(ValueError):
        validate_notes_provenance_payload({"origin": "knowledge_qa", "scope": {field: None}})


def test_direct_knowledge_context_uses_same_source_contract():
    from tldw_Server_API.app.core.Sync.v2.notes_provenance_contract import validate_notes_provenance_payload

    data = {
        "origin": "knowledge_qa",
        "question": "What happened?",
        "scope": {"include_media_ids": [1], "enable_web_fallback": True},
        "trust_reason_codes": ["web_fallback_used"],
        "sources": payload()["research"]["sources"][0]["evidence"]["sources"],
    }
    assert validate_notes_provenance_payload(data) == data
    data["sources"][0]["token"] = "secret"
    with pytest.raises(ValueError):
        validate_notes_provenance_payload(data)


def test_encoded_marker_ceiling_applies_to_canonical_payload():
    from tldw_Server_API.app.core.Sync.v2.notes_provenance_contract import validate_notes_provenance_payload

    source = payload()["research"]["sources"][0]["evidence"]["sources"][0]
    source["excerpt"] = "é" * 100000
    data = {"origin": "knowledge_qa", "sources": [copy.deepcopy(source) for _ in range(4)]}
    with pytest.raises(ValueError):
        validate_notes_provenance_payload(data)


def test_malformed_percent_encoding_grants_no_provenance():
    from tldw_Server_API.app.core.Sync.v2.notes_provenance_contract import read_notes_provenance

    assert (
        read_notes_provenance(
            "<!-- tldw-knowledge:v1:%7B%22origin%22%3A%22knowledge_qa%22%2C%22thread_id%22%3A%22%ZZ%22%7D -->"
        )
        is None
    )


def test_text_limits_match_client_utf16_units():
    from tldw_Server_API.app.core.Sync.v2.notes_provenance_contract import validate_notes_provenance_payload

    assert (
        validate_notes_provenance_payload({"origin": "knowledge_qa", "thread_id": "😀" * 256})["thread_id"]
        == "😀" * 256
    )
    with pytest.raises(ValueError):
        validate_notes_provenance_payload({"origin": "knowledge_qa", "thread_id": "😀" * 257})


@given(st.integers(min_value=1, max_value=9_007_199_254_740_991))
def test_safe_positive_source_ids_roundtrip(source_id):
    from tldw_Server_API.app.core.Sync.v2.notes_provenance_contract import validate_notes_provenance_payload

    data = payload()
    data["research"]["sources"][0]["mediaId"] = source_id
    assert validate_notes_provenance_payload(data) == data
