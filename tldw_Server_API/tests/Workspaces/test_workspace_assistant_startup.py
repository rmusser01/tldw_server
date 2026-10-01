"""Closed, bounded Workspace startup requests and deterministic retry hashes."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.schemas.chat_session_schemas import ChatSessionCreate

pytestmark = pytest.mark.unit

MODULE = "tldw_Server_API.app.api.v1.schemas.workspace_chat_startup_schemas"
TEXT_LIMITS = {
    "workspace_id": 256,
    "title": 4096,
    "state": 16,
    "topic_label": 1024,
    "cluster_id": 256,
    "source": 256,
    "external_ref": 4096,
}
METADATA = ("title", "state", "topic_label", "cluster_id", "source", "external_ref")


def _startup_module():
    """Fail inside tests, not collection, while the new contract is absent."""
    assert importlib.util.find_spec(MODULE) is not None, "Task 1A startup contract is missing"
    return importlib.import_module(MODULE)


def _payload(**values: Any) -> dict[str, Any]:
    """Build a minimal explicit-none request with optional caller fields."""
    return {
        "scope_type": "workspace",
        "workspace_id": "ws",
        "workspace_assistant_selection": "none",
        **values,
    }


def _request(payload: dict[str, Any]):
    """Exercise the production request boundary without transport or DB effects."""
    return _startup_module().WorkspaceChatStartupRequest.model_validate(payload)


@pytest.mark.parametrize("field", ("scope_type", "workspace_id", "workspace_assistant_selection"))
def test_mandatory_fields_cannot_be_omitted(field):
    payload = _payload()
    del payload[field]
    with pytest.raises(ValidationError):
        _request(payload)


@pytest.mark.parametrize("field", ("scope_type", "workspace_id", "workspace_assistant_selection"))
def test_mandatory_fields_cannot_be_null(field):
    with pytest.raises(ValidationError):
        _request(_payload(**{field: None}))


@pytest.mark.parametrize("scope", ("global", "Workspace", " workspace ", "", 1, False))
def test_only_exact_workspace_scope_is_accepted(scope):
    with pytest.raises(ValidationError):
        _request(_payload(scope_type=scope))


@pytest.mark.parametrize("selection", ("", "INHERIT", " inherit ", "explicit", 1, False))
def test_selection_must_be_exact_inherit_or_none(selection):
    with pytest.raises(ValidationError):
        _request(_payload(workspace_assistant_selection=selection))


def test_inherit_requires_supplied_version():
    with pytest.raises(ValidationError):
        _request(_payload(workspace_assistant_selection="inherit"))


@pytest.mark.parametrize("version", (None, 0, -1, True, False, "1", "", 1.0, [], {}))
def test_inherit_rejects_nonpositive_or_nonstrict_versions(version):
    with pytest.raises(ValidationError):
        _request(_payload(workspace_assistant_selection="inherit", workspace_assistant_default_version=version))


@pytest.mark.parametrize("version", (1, 2, 2**63))
def test_inherit_accepts_positive_strict_integer_version(version):
    request = _request(_payload(workspace_assistant_selection="inherit", workspace_assistant_default_version=version))
    assert request.workspace_assistant_default_version == version


@pytest.mark.parametrize("version", (None, 0, 1, -1, True, "1"))
def test_none_forbids_version_presence_even_null(version):
    with pytest.raises(ValidationError):
        _request(_payload(workspace_assistant_default_version=version))


@pytest.mark.parametrize(
    "field,value",
    [
        ("character_id", 1),
        ("assistant_kind", "persona"),
        ("assistant_id", "persona-1"),
        ("persona_memory_mode", "read_only"),
        ("assistant_startup", {}),
        ("participant_character_ids", []),
        ("prompt_preset_id", "preset"),
        ("memory_by_character_id", {}),
        ("provider", "openai"),
        ("model", "model"),
        ("temperature", 0.7),
        ("top_p", 1.0),
        ("repetition_penalty", 1.0),
        ("stop", []),
        ("parent_conversation_id", "parent"),
        ("forked_from_message_id", "message"),
        ("user_id", 1),
        ("created_at", "2026-09-27T00:00:00Z"),
        ("unknown", "value"),
    ],
)
@pytest.mark.parametrize("explicit_null", (False, True))
def test_extra_fields_are_forbidden_even_default_or_null(field, value, explicit_null):
    with pytest.raises(ValidationError) as error:
        _request(_payload(**{field: None if explicit_null else value}))
    assert error.value.errors()[0]["type"] == "extra_forbidden"


@pytest.mark.parametrize("field", METADATA)
def test_metadata_null_and_omission_remain_distinct(field):
    assert _request(_payload(**{field: None})).model_dump(mode="json", exclude_unset=True) == _payload(**{field: None})
    assert _request(_payload()).model_dump(mode="json", exclude_unset=True) == _payload()


@pytest.mark.parametrize("workspace_id", ("", " ", "\t\n", "\u2003"))
def test_workspace_id_cannot_be_blank_after_trimming(workspace_id):
    with pytest.raises(ValidationError):
        _request(_payload(workspace_id=workspace_id))


def test_workspace_id_trims_without_unicode_normalization():
    assert _request(_payload(workspace_id=" \te\u0301\u2003")).workspace_id == "e\u0301"


@pytest.mark.parametrize("field", METADATA)
@pytest.mark.parametrize("value", (1, False, [], {}))
def test_metadata_requires_string_or_null(field, value):
    with pytest.raises(ValidationError):
        _request(_payload(**{field: value}))


@pytest.mark.parametrize("field", TEXT_LIMITS)
@pytest.mark.parametrize("invalid", ("\x00", "\ud800", "\udfff", "\ud800\udc00"))
def test_decoded_text_rejects_nul_and_surrogate_codepoints(field, invalid):
    with pytest.raises(ValidationError):
        _request(_payload(**{field: f"x{invalid}y"}))


@pytest.mark.parametrize("field,limit", TEXT_LIMITS.items())
@pytest.mark.parametrize("character,width", (("x", 1), ("\u00e9", 2), ("\U0001f600", 4)))
def test_text_size_helper_accepts_exact_utf8_boundary(field, limit, character, width):
    assert _startup_module().startup_text_size(character * (limit // width), field) == limit


@pytest.mark.parametrize("field,limit", TEXT_LIMITS.items())
@pytest.mark.parametrize("character,width", (("x", 1), ("\u00e9", 2), ("\U0001f600", 4)))
def test_text_size_helper_rejects_utf8_boundary_plus_one(field, limit, character, width):
    with pytest.raises(ValueError):
        _startup_module().startup_text_size(character * (limit // width) + "x", field)


@pytest.mark.parametrize("field,limit", [(field, limit) for field, limit in TEXT_LIMITS.items() if field != "state"])
@pytest.mark.parametrize("character,width", (("x", 1), ("\u00e9", 2), ("\U0001f600", 4)))
def test_request_accepts_each_text_field_at_utf8_limit(field, limit, character, width):
    text = character * (limit // width)
    assert getattr(_request(_payload(**{field: text})), field) == text


@pytest.mark.parametrize("field,limit", TEXT_LIMITS.items())
@pytest.mark.parametrize("character,width", (("x", 1), ("\u00e9", 2), ("\U0001f600", 4)))
def test_request_rejects_each_text_field_over_utf8_limit(field, limit, character, width):
    with pytest.raises(ValidationError, match="limit"):
        _request(_payload(**{field: character * (limit // width) + "x"}))


def test_character_length_is_checked_before_encoding():
    class MustNotEncode(str):
        def encode(self, *args, **kwargs):
            pytest.fail("Overlong text must be rejected before encoding")

    with pytest.raises(ValueError, match="limit"):
        _startup_module().startup_text_size(MustNotEncode("x" * 257), "workspace_id")


@pytest.mark.parametrize("field,text", (("workspace_id", " " * 256 + "w"), ("state", " resolved" + " " * 8)))
def test_raw_bounds_are_checked_before_trimming_or_state_normalization(field, text):
    with pytest.raises(ValidationError, match="limit"):
        _request(_payload(**{field: text}))


@pytest.mark.parametrize("character,width", (("x", 1), ("\u00e9", 2), ("\U0001f600", 4)))
def test_aggregate_text_accepts_exact_8192_byte_limit(character, width):
    payload = _payload(
        workspace_id=character * (256 // width),
        title=character * (4096 // width),
        external_ref=character * (3840 // width),
    )
    assert _request(payload).model_dump(mode="json", exclude_unset=True) == payload


@pytest.mark.parametrize("character,width", (("x", 1), ("\u00e9", 2), ("\U0001f600", 4)))
def test_aggregate_text_rejects_8193_bytes_with_individually_valid_fields(character, width):
    with pytest.raises(ValidationError, match="combined byte limit"):
        _request(
            _payload(
                workspace_id=character * (256 // width),
                title=character * (4096 // width),
                external_ref=character * (3840 // width) + "x",
            )
        )


def test_aggregate_counts_raw_workspace_id_before_trimming():
    with pytest.raises(ValidationError, match="combined byte limit"):
        _request(_payload(workspace_id=" " * 255 + "w", title="x" * 4096, external_ref="x" * 3841))


def test_aggregate_counts_every_supplied_metadata_string():
    with pytest.raises(ValidationError, match="combined byte limit"):
        _request(
            _payload(
                workspace_id="w" * 256,
                title="x" * 4096,
                topic_label="x" * 1024,
                cluster_id="x" * 256,
                source="x" * 256,
                state="   IN-PROGRESS  ",
                external_ref="x" * 2289,
            )
        )


@pytest.mark.parametrize("state", ("in-progress", "resolved", "backlog", "non-viable"))
def test_existing_state_enum_normalizes(state):
    assert _request(_payload(state=f" {state.upper()} ")).state == state


def test_state_accepts_valid_16_byte_raw_input():
    assert _request(_payload(state="   IN-PROGRESS  ")).state == "in-progress"


@pytest.mark.parametrize("state", ("", " ", "pending", "x" * 16, "\u00e9" * 8))
def test_state_within_byte_bound_still_requires_valid_semantics(state):
    with pytest.raises(ValidationError, match="state"):
        _request(_payload(state=state))


@pytest.mark.parametrize("field", ("workspace_id", "title", "topic_label", "cluster_id", "source", "external_ref"))
def test_valid_unicode_and_json_surrogate_pair_decode_are_preserved(field):
    payload = _payload(**{field: "\u00e9e\u0301\U0001f600"})
    decoded = json.loads(json.dumps(payload, ensure_ascii=True))
    assert getattr(_request(decoded), field) == payload[field]


@pytest.mark.parametrize("field", ("title", "topic_label", "cluster_id", "source", "external_ref"))
def test_metadata_does_not_trim_or_normalize_valid_strings(field):
    text = " \te\u0301\n\u00e9 "
    assert getattr(_request(_payload(**{field: text})), field) == text


@pytest.mark.parametrize("escaped", (r"\u0000", r"\ud800", r"\udfff"))
def test_json_decoded_invalid_unicode_is_rejected(escaped):
    payload = _payload(title=json.loads(f'"{escaped}"'))
    with pytest.raises(ValidationError):
        _request(payload)


def test_canonical_hash_uses_schema_version_and_only_supplied_body():
    expected = hashlib.sha256(
        b'{"body":{"scope_type":"workspace","workspace_assistant_selection":"none","workspace_id":"ws"},"schema_version":1}'
    ).hexdigest()
    assert _startup_module().startup_request_fingerprint(_request(_payload())) == expected


def test_canonical_hash_uses_ascii_escaping_and_normalized_request():
    request = _request(
        _payload(
            workspace_id=" ws ",
            workspace_assistant_selection="inherit",
            workspace_assistant_default_version=2,
            title="\u00e9\U0001f600",
            state=" RESOLVED ",
            external_ref=None,
        )
    )
    canonical = b'{"body":{"external_ref":null,"scope_type":"workspace","state":"resolved","title":"\\u00e9\\ud83d\\ude00","workspace_assistant_default_version":2,"workspace_assistant_selection":"inherit","workspace_id":"ws"},"schema_version":1}'
    assert _startup_module().startup_request_fingerprint(request) == hashlib.sha256(canonical).hexdigest()


@pytest.mark.parametrize("field", METADATA)
def test_canonical_hash_distinguishes_omission_and_null(field):
    fingerprint = _startup_module().startup_request_fingerprint
    assert fingerprint(_request(_payload())) != fingerprint(_request(_payload(**{field: None})))


@pytest.mark.parametrize("field", ("title", "topic_label", "cluster_id", "source", "external_ref"))
def test_canonical_hash_distinguishes_null_and_empty_metadata(field):
    fingerprint = _startup_module().startup_request_fingerprint
    assert fingerprint(_request(_payload(**{field: None}))) != fingerprint(_request(_payload(**{field: ""})))


def test_canonical_hash_distinguishes_unicode_normalization_forms():
    fingerprint = _startup_module().startup_request_fingerprint
    assert fingerprint(_request(_payload(title="\u00e9"))) != fingerprint(_request(_payload(title="e\u0301")))


def test_canonical_hash_distinguishes_selection_and_version():
    fingerprint = _startup_module().startup_request_fingerprint
    requests = [
        _payload(),
        _payload(workspace_assistant_selection="inherit", workspace_assistant_default_version=1),
        _payload(workspace_assistant_selection="inherit", workspace_assistant_default_version=2),
    ]
    assert len({fingerprint(_request(payload)) for payload in requests}) == 3


def test_fingerprinting_does_not_mutate_presence_or_generate_metadata():
    request = _request(_payload())
    _startup_module().startup_request_fingerprint(request)
    assert request.model_dump(mode="json", exclude_unset=True) == _payload()


VALID_UNICODE = st.text(alphabet=st.characters(blacklist_categories=("Cs",), blacklist_characters="\x00"), max_size=48)


@settings(max_examples=100, derandomize=True, database=None)
@given(text=VALID_UNICODE, data=st.data())
def test_canonical_hash_is_independent_of_valid_unicode_input_key_order(text, data):
    payload = _payload(
        workspace_id="w" + text,
        title=text,
        topic_label=text,
        cluster_id=text,
        source=text,
        external_ref=None,
        state=" RESOLVED ",
    )
    keys = data.draw(st.permutations(tuple(payload)))
    fingerprint = _startup_module().startup_request_fingerprint
    assert fingerprint(_request({key: payload[key] for key in keys})) == fingerprint(_request(payload))


def test_exported_bounds_match_transport_and_text_contract():
    module = _startup_module()
    assert (module.STARTUP_TEXT_BYTE_LIMITS, module.STARTUP_TEXT_BYTES_MAX, module.STARTUP_BODY_BYTES_MAX) == (
        TEXT_LIMITS,
        8192,
        65536,
    )


def test_legacy_chat_creation_remains_unbounded_by_strict_startup_contract():
    request = ChatSessionCreate.model_validate({"title": "x" * 4097, "scope_type": "global"})
    assert request.title == "x" * 4097
