"""Approved bounded wire projections, separate from admission authority."""

import json

import pytest
from pydantic import TypeAdapter, ValidationError

from tldw_Server_API.app.api.v1.schemas import history_selection_schemas as wire
from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import ChatCompletionRequest

pytestmark = pytest.mark.unit
UID = "f61d4e19-3600-4715-96fd-1f7b45cf8d77"
DIGEST = "a" * 64


def selection():
    return {
        "version": 1,
        "owner_key": "owner",
        "conversation_id": "chat",
        "interpretation": {"kind": "parent_graph_v1"},
        "cursor": {"kind": "empty"},
        "selection_revision": 0,
        "purpose": "send",
        "messages": [],
        "fences": {"conversation": "1", "history": "1", "settings": "1"},
        "storage_context_digest": DIGEST,
        "request_context_digest": DIGEST,
        "selection_digest": DIGEST,
    }


def admission():
    return {
        "version": 1,
        "owner_key": "owner",
        "conversation_id": "chat",
        "input_message_id": UID,
        "input_message_revision": "1",
        "selection_digest": DIGEST,
    }


def request_body(kind="selection"):
    history = {"version": 1, "kind": kind, kind: selection() if kind == "selection" else admission()}
    if kind == "admission":
        history["request_context_digest"] = DIGEST
    return {
        "model": "local-model",
        "api_provider": "openai",
        "stream": False,
        "conversation_id": "chat",
        "save_to_db": True,
        "messages": [{"role": "system", "content": "frozen context"}, {"role": "user", "content": " original "}],
        "tldw_turn": {"user_message_id": UID, "history_v1": history, "result_v1": {"version": 1, "sources": []}},
    }


def source(**overrides):
    return {
        "name": "Document",
        "type": "pdf",
        "mode": "rag",
        "url": "",
        "pageContent": "evidence",
        "metadata": {},
        **overrides,
    }


@pytest.mark.parametrize("kind", ["selection", "admission"])
def test_accepts_approved_nested_mode_without_changing_original_user(kind):
    request = ChatCompletionRequest.model_validate(request_body(kind))
    assert request.messages[-1].content == " original "
    assert request.tldw_turn.history_v1.kind == kind
    assert request.tldw_turn.result_v1.sources == ()


@pytest.mark.parametrize(
    "field,value",
    [
        ("stream", None),
        ("stream", "false"),
        ("stream", 0),
        ("model", ""),
        ("model", "auto"),
        ("api_provider", None),
        ("save_to_db", 1),
        ("tools", []),
        ("function_call", "auto"),
        ("functions", []),
        ("tool_choice", "none"),
        ("extra_body", {"tldw_turn": {}}),
        ("extra_body", {"temperature": 0.5}),
        ("history_message_limit", 3),
        ("history_message_order", "asc"),
        ("tldw_retry_failed_turn", True),
        ("tldw_regenerate_from_message_id", UID),
        ("metadata", {"tldw_retry_failed_turn": True}),
        ("tldw_continuation", {"from_message_id": UID, "mode": "branch"}),
        ("tldw_history_selection_v1", selection()),
    ],
)
def test_nested_mode_rejects_inference_defaults_and_other_admission_controls(field, value):
    body = request_body()
    body[field] = value
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(body)


@pytest.mark.parametrize("field", ["model", "api_provider", "stream", "save_to_db"])
def test_nested_mode_requires_explicit_inference_and_persistence_fields(field):
    body = request_body()
    del body[field]
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(body)


@pytest.mark.parametrize(
    "change",
    ["no_result", "result_only", "null_history", "null_result", "fork", "conversation", "input", "digest", "version"],
)
def test_nested_mode_rejects_mismatched_and_partial_envelopes(change):
    body = request_body("admission" if change == "input" else "selection")
    turn = body["tldw_turn"]
    if change == "no_result":
        del turn["result_v1"]
    elif change == "result_only":
        del turn["history_v1"]
    elif change in {"null_history", "null_result"}:
        turn["history_v1" if change == "null_history" else "result_v1"] = None
    elif change == "input":
        turn["history_v1"]["admission"]["input_message_id"] = "3be5bac5-b2d5-4d42-9a53-3a2db0f1c6d5"
    elif change == "version":
        turn["history_v1"]["version"] = True
    else:
        turn["history_v1"]["selection"][
            {"fork": "purpose", "conversation": "conversation_id", "digest": "request_context_digest"}[change]
        ] = {"fork": "fork", "conversation": "other", "digest": "A" * 64}[change]
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(body)


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "user", "content": [{"type": "text", "text": "original"}]}],
        [{"role": "user", "content": " "}],
        [{"role": "user", "content": "original", "tool_calls": []}],
        [{"role": "assistant", "content": "old"}, {"role": "user", "content": "original"}],
        [{"role": "user", "content": "old"}, {"role": "user", "content": "original"}],
        [{"role": "system", "content": []}, {"role": "user", "content": "original"}],
    ],
)
def test_nested_mode_rejects_nonoriginal_or_nontext_messages(messages):
    body = request_body()
    body["messages"] = messages
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(body)


def test_legacy_durable_client_keeps_text_part_normalization_and_omitted_inference():
    request = ChatCompletionRequest.model_validate(
        {
            "conversation_id": "chat",
            "save_to_db": True,
            "tldw_turn": {"user_message_id": UID},
            "messages": [{"role": "user", "content": [{"type": "text", "text": " a "}, {"type": "text", "text": "b"}]}],
        }
    )
    assert request.messages[0].content == " a \nb"
    assert request.tldw_turn.model_dump() == {"user_message_id": UID}


def test_nested_mode_rejects_null_message_array_as_validation_error():
    body = request_body()
    body["messages"] = None
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(body)


@pytest.mark.parametrize("role", [[], {}, ["user"], {"role": "user"}])
def test_nested_mode_rejects_unhashable_roles_as_validation_error(role):
    body = request_body()
    body["messages"][-1]["role"] = role
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(body)


@pytest.mark.parametrize("score", [10**20, -(10**20), 9007199254740993, 1.7976931348623157e308])
def test_source_score_uses_finite_number_domain_not_locator_safe_integer_domain(score):
    value = wire.SourceV1.model_validate(source(metadata={"score": score}))
    assert value.metadata.score == float(score)
    with pytest.raises(ValidationError):
        wire.SourceV1.model_validate(source(metadata={"page": score}))


@pytest.mark.parametrize("mode", ["validation", "serialization"])
def test_recovery_and_turn_schemas_keep_strict_metadata_locator_and_member_types(mode):
    from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import TLDWTurnSpec

    schema = TypeAdapter(wire.HistoryRecoveryReadV1).json_schema(mode=mode)
    definitions = schema["$defs"]
    metadata = definitions["SourceMetadataV1"]
    assert metadata["additionalProperties"] is False
    assert set(metadata["properties"]) == {
        "source",
        "title",
        "chunk_id",
        "retrieval_strategy",
        "source_type",
        "selection_reason",
        "score",
        "page",
        "loc",
        "media_id",
        "author",
        "chunk_index",
        "total_chunks",
        "start_char",
        "end_char",
        "chunk_start",
        "chunk_end",
    }
    assert metadata["properties"]["title"]["type"] == "string"
    assert metadata["properties"]["loc"]["$ref"] == "#/$defs/SourceLocationV1"
    assert definitions["SourceLocationV1"]["additionalProperties"] is False
    lines = definitions["SourceLinesV1"]
    assert lines["additionalProperties"] is False
    assert lines["properties"]["from"]["maximum"] == 9007199254740991
    assert lines["properties"]["from"]["type"] == "integer"
    turn = TLDWTurnSpec.model_json_schema(mode=mode)
    assert turn["additionalProperties"] is False
    assert set(turn["properties"]) == {"user_message_id", "history_v1", "result_v1"}


@pytest.mark.parametrize("mode", ["validation", "serialization"])
def test_result_schema_publishes_approved_hidden_source_bounds_on_payload_metadata_and_receipt(mode):
    bounds = {
        "sources": 20,
        "excerpt_scalars": 1000,
        "excerpt_utf8_bytes": 4000,
        "aggregate_excerpt_scalars": 16000,
        "name_utf8_bytes": 1000,
        "metadata_text_utf8_bytes": 1000,
        "chunk_id_utf8_bytes": 512,
        "media_id_utf8_bytes": 512,
        "paired_character_ranges": True,
        "chunk_index_within_total": True,
        "compact_label_utf8_bytes": 128,
        "url_utf8_bytes": 2048,
        "result_canonical_utf8_bytes": 65536,
        "distinct_excerpts": True,
    }
    for model in (wire.HistoryResultPayloadV1, wire.HistoryResultMetadataV1, wire.HistoryResultV1):
        schema = model.model_json_schema(mode=mode)
        assert schema["x-tldw-source-bounds"] == bounds
        assert schema["additionalProperties"] is False
        assert "x-tldw-source-bounds" not in schema["properties"]
    schema = TypeAdapter(wire.HistoryRecoveryReadV1).json_schema(mode=mode)
    assert schema["$defs"]["HistoryResultV1"]["x-tldw-source-bounds"] == bounds


def test_source_preserves_real_media_attribution_and_both_character_range_shapes():
    metadata = {
        "media_id": "1",
        "author": " Mira Chen ",
        "chunk_index": 0,
        "total_chunks": 1,
        "start_char": 0,
        "end_char": 10,
        "chunk_start": 0,
        "chunk_end": 10,
    }
    assert wire.SourceV1.model_validate(source(metadata=metadata)).model_dump(mode="json")["metadata"] == metadata


@pytest.mark.parametrize(
    "metadata",
    [
        {"media_id": " "},
        {"media_id": "x" * 513},
        {"media_id": None},
        {"media_id": 1},
        {"author": "x" * 1001},
        {"author": None},
        {"chunk_index": True},
        {"chunk_index": -1},
        {"chunk_index": "0"},
        {"total_chunks": 0},
        {"chunk_index": 3, "total_chunks": 3},
        {"start_char": 1},
        {"end_char": 1},
        {"start_char": 2, "end_char": 1},
        {"chunk_start": 1},
        {"chunk_end": 1},
        {"chunk_start": 2, "chunk_end": 1},
        {"chunk_start": 0, "chunk_end": 9007199254740992},
    ],
)
def test_source_rejects_malformed_locator_metadata_without_coercion(metadata):
    with pytest.raises(ValidationError):
        wire.SourceV1.model_validate(source(metadata=metadata))


@pytest.mark.parametrize("field", ["name", "type", "mode", "url", "pageContent", "metadata"])
def test_source_requires_every_top_level_member(field):
    value = source()
    del value[field]
    with pytest.raises(ValidationError):
        wire.SourceV1.model_validate(value)


@pytest.mark.parametrize(
    "value",
    [
        source(name=" "),
        source(type=""),
        source(pageContent="\n"),
        source(mode="web"),
        source(name="\ud800"),
        source(url="\udfff"),
        source(pageContent="a" * 1001),
        source(name="\u00e9" * 501),
        source(type="\u00e9" * 65),
        source(url="x" * 2049),
        source(extra=True),
        source(metadata={"unknown": "unsupported"}),
        source(metadata={"page": True}),
        source(metadata={"page": -1}),
        source(metadata={"page": 9007199254740992}),
        source(metadata={"page": 1.0}),
        source(metadata={"score": True}),
        source(metadata={"score": float("inf")}),
        source(metadata={"score": "0.5"}),
        source(metadata={"title": None}),
        source(metadata={"source": " "}),
        source(metadata={"chunk_id": "a" * 513}),
        source(metadata={"source_type": "a" * 129}),
        source(metadata={"loc": {"lines": {"from": 2, "to": 1}}}),
        source(metadata={"loc": {"lines": {"from": False, "to": 1}}}),
        source(metadata={"loc": {"lines": {"from": 0, "to": 1, "extra": 2}}}),
        source(metadata={"loc": {"lines": {"from": 0, "to": 1}, "extra": 2}}),
    ],
)
def test_source_rejects_lossy_or_unsupported_projection(value):
    with pytest.raises(ValidationError):
        wire.SourceV1.model_validate(value)


@pytest.mark.parametrize(
    "field",
    ["source", "title", "chunk_id", "retrieval_strategy", "source_type", "selection_reason", "score", "page", "loc"],
)
def test_metadata_optional_means_omitted_not_null(field):
    with pytest.raises(ValidationError):
        wire.SourceV1.model_validate(source(metadata={field: None}))


def test_source_preserves_inert_url_finite_score_and_scalar_boundaries():
    value = source(
        url="local provenance: exact",
        pageContent="\U0001f600" * 1000,
        metadata={
            "score": -12.5,
            "page": 9007199254740991,
            "loc": {"lines": {"from": 0, "to": 9007199254740991}},
        },
    )
    assert wire.SourceV1.model_validate(value).model_dump(mode="json") == value
    assert wire.SourceV1.model_validate(source()).model_dump(mode="json")["metadata"] == {}


@pytest.mark.parametrize(
    "sources",
    [
        [source(pageContent=str(i)) for i in range(21)],
        [source(pageContent=str(i) + "x" * 997) for i in range(17)],
        [source(), source()],
        [
            source(pageContent=str(i), name="x" * 1000, url="x" * 2048, metadata={"title": "x" * 1000})
            for i in range(20)
        ],
        [source(pageContent=str(i), name="\u0000" * 999 + "x", url="") for i in range(12)],
    ],
)
def test_result_payload_rejects_count_total_excerpt_duplicate_and_canonical_byte_overflow(sources):
    with pytest.raises(ValidationError):
        wire.HistoryResultPayloadV1.model_validate({"version": 1, "sources": sources})


def test_result_payload_allows_twenty_ordered_distinct_excerpts():
    values = [source(pageContent=str(i)) for i in range(20)]
    payload = wire.HistoryResultPayloadV1.model_validate({"version": 1, "sources": values})
    assert payload.model_dump(mode="json") == {"version": 1, "sources": values}


def test_result_payload_accepts_exact_64k_and_total_scalar_boundaries_but_not_one_extra_byte():
    values = [
        source(name="x" * 1000, pageContent=str(i) + "x" * (1000 - len(str(i))), metadata={"title": "x" * 1000})
        for i in range(16)
    ]
    payload = {"version": 1, "sources": values}
    # ASCII-only fixed keys give an independent equivalent compact byte count.
    remaining = 65536 - len(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
    for value in values:
        amount = min(remaining, 2048)
        value["url"] = "x" * amount
        remaining -= amount
    assert remaining == 0
    assert wire.HistoryResultPayloadV1.model_validate(payload).model_dump(mode="json") == payload
    values[-1]["url"] += "x"
    with pytest.raises(ValidationError):
        wire.HistoryResultPayloadV1.model_validate(payload)


def result():
    return {
        "version": 1,
        "result_message_id": UID,
        "result_message_revision": "1",
        "admission": admission(),
        "request_context_digest": DIGEST,
        "sources": [],
    }


@pytest.mark.parametrize("status", ["input_verified", "result_verified", "unverified"])
def test_strict_recovery_union_projects_only_approved_shape(status):
    value = {"version": 1, "status": status}
    if status == "unverified":
        value["code"] = "no_protected_binding"
    else:
        value["scope"] = {"scope_type": "global", "workspace_id": None}
        value["admission" if status == "input_verified" else "result"] = (
            {**admission(), "messages": [], "originating_selection_revision": 0}
            if status == "input_verified"
            else result()
        )
    adapter = TypeAdapter(wire.HistoryRecoveryReadV1)
    assert adapter.validate_python(value).model_dump(mode="json") == value
    with pytest.raises(ValidationError):
        adapter.validate_python({**value, "protected_json": {}})


@pytest.mark.parametrize(
    "field,value",
    [
        ("result_message_id", "not-uuid"),
        ("result_message_revision", "2"),
        ("request_context_digest", "A" * 64),
        ("version", True),
        ("sources", None),
    ],
)
def test_result_receipt_rejects_invalid_binding_fields(field, value):
    receipt = result()
    receipt[field] = value
    with pytest.raises(ValidationError):
        wire.HistoryResultV1.model_validate(receipt)


@pytest.mark.parametrize(
    "scope",
    [
        {"scope_type": "global", "workspace_id": "workspace"},
        {"scope_type": "workspace", "workspace_id": None},
        {"scope_type": "workspace", "workspace_id": " "},
        {"scope_type": "global"},
    ],
)
def test_recovery_rejects_scope_coercion(scope):
    with pytest.raises(ValidationError):
        TypeAdapter(wire.HistoryScopeV1).validate_python(scope)
