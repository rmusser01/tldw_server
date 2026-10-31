"""Reference-only startup values reject invalid authority and bounded storage drift."""

import importlib
import json

import pytest
from hypothesis import given
from hypothesis import strategies as st
from loguru import logger
from pydantic import ValidationError

pytestmark = pytest.mark.unit


def _module():
    """Load the new value boundary inside tests so missing implementation fails RED."""
    return importlib.import_module("tldw_Server_API.app.core.Chat.assistant_startup")


@pytest.mark.parametrize(
    "source,references",
    [
        ("workspace_default", {"workspace_id": "ws", "workspace_version": 2}),
        ("system_fallback", {"workspace_id": "ws", "workspace_version": 2}),
        ("system_fallback", {}),
        ("explicit", {}),
        ("explicit_none", {}),
        ("fork", {}),
        ("unknown", {}),
    ],
)
def test_valid_sources_round_trip_with_only_reference_keys(source, references):
    """Each supported source retains its values without snapshot fields."""
    module = _module()
    value = module.AssistantStartup(source=source, **references)
    encoded = module.encode_assistant_startup(value)
    assert json.loads(encoded) == {
        "schema_version": 1,
        "source": source,
        "workspace_id": references.get("workspace_id"),
        "workspace_version": references.get("workspace_version"),
    }
    assert module.decode_assistant_startup(encoded) == value
    assert len(encoded.encode("utf-8")) <= 1024


@pytest.mark.parametrize(
    "fields",
    [
        {"source": "workspace_default"},
        {"source": "workspace_default", "workspace_id": "ws"},
        {"source": "workspace_default", "workspace_version": 1},
        {"source": "system_fallback", "workspace_id": "ws"},
        {"source": "system_fallback", "workspace_version": 1},
        *[
            {"source": source, "workspace_id": "ws", "workspace_version": 1}
            for source in ("explicit", "explicit_none", "fork", "unknown")
        ],
        {"source": "invented"},
        *[{"schema_version": version} for version in (True, "1", 1.0, 0, 2, None)],
        *[
            {"source": "workspace_default", "workspace_id": "ws", "workspace_version": version}
            for version in (True, "1", 1.0, 0, -1)
        ],
        *[
            {"source": "workspace_default", "workspace_id": workspace_id, "workspace_version": 1}
            for workspace_id in ("", 42, True, b"ws", [], {})
        ],
        *[{field: "PRIVATE-SNAPSHOT"} for field in ("name", "prompt", "assistant_id", "persona_memory_mode")],
    ],
)
def test_invalid_source_reference_and_snapshot_values_are_rejected(fields):
    """No coercion or extra field may turn malformed input into trusted origin."""
    module = _module()
    with pytest.raises(ValidationError):
        module.AssistantStartup(**fields)


@pytest.mark.parametrize(
    "raw",
    [
        None, "broken", "[]", "null", "true", "42", '{"schema_version":true}',
        '{"source":"explicit","prompt":"PRIVATE"}', {}, [], 1, b"{}",
        '{"source":"workspace_default","workspace_id":"\\ud800","workspace_version":1}',
        "\ud800", " " * 1024 + "{}", "[" * 1000,
    ],
)
def test_invalid_storage_returns_fresh_unknown_without_logging_input(raw):
    """Legacy, corrupt and oversized storage cannot fabricate an origin or leak logs."""
    module = _module()
    messages = []
    sink = logger.add(messages.append, format="{message}")
    try:
        first = module.decode_assistant_startup(raw)
        second = module.decode_assistant_startup(raw)
    finally:
        logger.remove(sink)
    assert first.model_dump() == {
        "schema_version": 1, "source": "unknown", "workspace_id": None, "workspace_version": None,
    }
    assert first is not second
    assert "PRIVATE" not in "".join(messages)


def test_canonical_json_retains_escaped_control_and_unicode_id():
    """Encoding is deterministic UTF-8, not ASCII expansion or ID normalization."""
    module = _module()
    value = module.AssistantStartup(source="workspace_default", workspace_id=' \n"\\\u00e9 ', workspace_version=1)
    encoded = module.encode_assistant_startup(value)
    assert encoded == (
        '{"schema_version":1,"source":"workspace_default",'
        '"workspace_id":" \\n\\"\\\\\u00e9 ","workspace_version":1}'
    )
    assert module.decode_assistant_startup(encoded) == value


@pytest.mark.parametrize("workspace_id", ["x" * 936, "\u00e9" * 468, "\ud800"])
def test_construction_rejects_oversized_or_invalid_unicode_origin(workspace_id):
    """Response-only construction cannot bypass the serialized byte/Unicode bound."""
    module = _module()
    with pytest.raises(ValidationError):
        module.AssistantStartup(source="workspace_default", workspace_id=workspace_id, workspace_version=1)


def test_exact_byte_limit_accepts_long_legacy_workspace_id():
    """The complete 89-byte envelope permits a 935-byte ID, with no invented ID cap."""
    module = _module()
    value = module.AssistantStartup(source="workspace_default", workspace_id="x" * 935, workspace_version=1)
    assert len(module.encode_assistant_startup(value).encode("utf-8")) == 1024


@pytest.mark.parametrize("value", [None, {}, "{}", []])
def test_encoding_rejects_non_model_inputs(value):
    """Arbitrary caller data cannot cross the trusted encoding boundary."""
    module = _module()
    with pytest.raises(TypeError):
        module.encode_assistant_startup(value)


@pytest.mark.parametrize("workspace_id", ["x" * 936, "\ud800"])
def test_encoding_revalidates_unchecked_model_values(workspace_id):
    """model_construct cannot bypass validation at the storage boundary."""
    module = _module()
    unchecked = module.AssistantStartup.model_construct(
        source="workspace_default", workspace_id=workspace_id, workspace_version=1,
    )
    with pytest.raises(ValueError):
        module.encode_assistant_startup(unchecked)


def test_startup_value_is_immutable():
    """A validated origin cannot acquire a different source after construction."""
    module = _module()
    value = module.AssistantStartup()
    with pytest.raises(ValidationError):
        value.source = "explicit"


@given(
    workspace_id=st.text(alphabet=st.characters(blacklist_categories=("Cs",)), min_size=1, max_size=140),
    version=st.integers(min_value=1, max_value=2**63 - 1),
    source=st.sampled_from(["workspace_default", "system_fallback"]),
)
def test_reference_round_trips_preserve_arbitrary_text_within_encoded_cap(workspace_id, version, source):
    """Escaping and multibyte text retain their value while every encoding stays bounded."""
    module = _module()
    value = module.AssistantStartup(source=source, workspace_id=workspace_id, workspace_version=version)
    encoded = module.encode_assistant_startup(value)
    assert module.decode_assistant_startup(encoded) == value
    assert len(encoded.encode("utf-8")) <= 1024
