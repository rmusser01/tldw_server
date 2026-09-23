"""Native retained projection: semantic identity, exclusions and coherent reads."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.schemas.native_fork_schemas import (
    NativeForkCaptureRequestV1,
    NativeForkRequestV1,
    NativeScopeV1,
)
from tldw_Server_API.app.core.Character_Chat.character_behavior_snapshot import build_behavior_snapshot
from tldw_Server_API.app.core.Character_Chat.character_conversation_factory import build_materialized_behavior_controls
from tldw_Server_API.app.core.Chat.history_selection import HistorySelectionError
from tldw_Server_API.app.core.Chat.native_fork_projection import (
    AuthorizedNativeOwner,
    capture_native_fork,
    native_fork_request_digest,
    native_fork_request_tuple,
    project_native_fork_context,
    project_native_fork_messages,
)
from tldw_Server_API.app.core.DB_Management.chacha.conversation_resume_store import build_materialized_behavior_settings
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.tests.DB_Management.test_character_behavior_snapshot_migration import _snapshot

pytestmark = pytest.mark.unit
FIXTURE = Path(__file__).parents[2] / "fixtures" / "native_fork_v1.json"


@dataclass(frozen=True)
class AcceptedState:
    """Detached test values, rebuilt through real accepted authority builders."""

    encoded: str

    @property
    def state(self):
        return json.loads(self.encoded)

    @classmethod
    def build(cls, *, summary="old summary", lore="accepted lore", memory=None):
        payload = _snapshot().payload
        payload["participants"][0]["world_books"] = [{"id": 1, "entries": [{"content": lore}]}]
        payload["participants"][0]["default_memory"] = memory
        snapshot = build_behavior_snapshot(payload)
        settings = {
            "model": "accepted-model",
            "provider": "openai",
            "summary": {
                "enabled": True,
                "thresholdMessages": 20,
                "windowMessages": 5,
                "content": summary,
                "sourceRange": {"fromMessageId": "old"},
                "updatedAt": "old",
            },
        }
        effective = {
            "provider": "openai",
            "model": "accepted-model",
            "sampling": {"temperature": 0.2, "top_p": 1.0, "repetition_penalty": 1.0, "stop": []},
        }
        envelope = build_materialized_behavior_settings(
            {
                "base_snapshot": {"schema_version": 1, "digest": snapshot.digest},
                "behavior_controls": build_materialized_behavior_controls(settings),
                "effective_completion": effective,
            }
        )
        settings["roleplayBehaviorV1"] = envelope
        settings["roleplayResumeV1"] = {
            "resumeEligible": True,
            "resumeIneligibleReason": None,
            "effectiveCompletion": effective,
        }
        state = {
            "conversation": {
                "character_id": 1,
                "assistant_kind": "character",
                "assistant_id": "1",
                "scope_type": "global",
                "workspace_id": None,
            },
            "settings": settings,
            "settings_version": 1,
            "settings_present": True,
            "materialized_settings": {"schema_version": 1, "digest": envelope["digest"], "values": envelope["values"]},
            "behavior_snapshot": {
                "status": "valid",
                "payload": snapshot.payload,
                "schema_version": 1,
                "digest": snapshot.digest,
            },
        }
        return cls(json.dumps(state))

    def with_cached_summary(self, content):
        return self.build(summary=content)

    def with_accepted_worldbook_text(self, content):
        return self.build(lore=content)


@pytest.fixture
def accepted_state():
    return AcceptedState.build()


def neutral():
    return {
        "conversation": {"scope_type": "global", "workspace_id": None},
        "settings": None,
        "settings_present": False,
        "behavior_snapshot": {"status": "missing"},
    }


def row(mid="a", *, role="assistant", text="Stopped text", extra=None, tools=None, settled=True):
    return {
        "id": mid,
        "revision": "r1",
        "role": role,
        "settled": settled,
        "parent_id": None,
        "message": text,
        "images": [],
        "tool_calls": tools,
        "extra_metadata": extra,
    }


def test_summary_content_changes_do_not_change_fork_projection(accepted_state):
    before = project_native_fork_context(accepted_state.state, (), ())
    after = project_native_fork_context(accepted_state.with_cached_summary("new cached text").state, (), ())
    assert before.retained_context_digest == after.retained_context_digest
    settings = json.loads(after.settings_json)
    controls = settings["roleplayBehaviorV1"]["values"]["behavior_controls"]
    assert settings["summary"] == {"enabled": True, "thresholdMessages": 20, "windowMessages": 5}
    assert "summary" not in controls["auto_summary"]
    assert controls["auto_summary"]["enabled"] is True
    assert "content" not in controls["applied_overrides"]["summary"]


def test_retained_worldbook_content_changes_fork_projection(accepted_state):
    before = project_native_fork_context(accepted_state.state, (), ())
    after = project_native_fork_context(accepted_state.with_accepted_worldbook_text("different lore").state, (), ())
    assert before.retained_context_digest != after.retained_context_digest
    assert "world_books" in after.required_effects
    assert after.binding.display_name == "Legacy Character"
    assert not hasattr(after.binding, "child_id")


def test_neutral_absent_snapshot_is_valid_and_corrupt_settings_is_not():
    assert project_native_fork_context(neutral(), (), ()).snapshot_json is None
    corrupt = {**neutral(), "settings_present": True}
    with pytest.raises(HistorySelectionError, match="invalid_settings"):
        project_native_fork_context(corrupt, (), ())


def test_corrupt_required_envelope_rejects(accepted_state):
    state = accepted_state.state
    state["settings"]["roleplayBehaviorV1"]["digest"] = "sha256:" + "0" * 64
    with pytest.raises(HistorySelectionError, match="invalid_materialized"):
        project_native_fork_context(state, (), ())


def test_summary_memory_removed_and_snapshot_envelope_rebound():
    state = AcceptedState.build(
        memory={
            "content": "authored",
            "source": "creation_request",
            "version": 1,
            "persona_memory_entries": [
                {"id": "authored", "memory_type": "manual", "content": "keep"},
                {"id": "cache", "memory_type": "summary", "content": "drop"},
            ],
        }
    ).state
    original_digest = state["behavior_snapshot"]["digest"]
    projected = project_native_fork_context(state, (), ())
    payload = json.loads(projected.snapshot_json)
    assert payload["participants"][0]["default_memory"]["persona_memory_entries"] == [
        {"id": "authored", "memory_type": "manual", "content": "keep"}
    ]
    assert projected.snapshot_digest != original_digest
    assert (
        json.loads(projected.settings_json)["roleplayBehaviorV1"]["values"]["base_snapshot"]["digest"]
        == projected.snapshot_digest
    )


@pytest.mark.parametrize(
    "extra", [{"unknown_carrier": {}}, {"documents": [{"url": "/source"}]}, {"function_call": {"name": "legacy"}}]
)
def test_unknown_or_unbound_message_carriers_reject(extra):
    with pytest.raises(HistorySelectionError, match="unsupported_native_fork"):
        project_native_fork_context(neutral(), (row(extra=extra),), ())


def test_active_rows_reject_but_stopped_failed_text_is_retained():
    with pytest.raises(HistorySelectionError, match="unsettled_message"):
        project_native_fork_context(neutral(), (row(settled=False),), ())
    for status in ["stopped", "failed"]:
        projected = project_native_fork_context(neutral(), (row(extra={"status": status}),), ())
        assert projected.retained_context_digest


def test_incomplete_tool_replay_rejects_and_complete_group_has_inventory():
    call = row(tools=[{"id": "call1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}])
    with pytest.raises(HistorySelectionError, match="incomplete_tool_replay"):
        project_native_fork_context(neutral(), (call,), ())
    result = row("b", role="tool", extra={"tool_call_id": "call1"})
    projected = project_native_fork_context(neutral(), (call, result), ())
    assert "tool_replay" in projected.required_effects


def test_generated_event_mirror_is_never_classified_as_plain_text():
    with pytest.raises(HistorySelectionError, match="generated_event"):
        project_native_fork_context(neutral(), (row(text="[[tldw:image-generation-event:v1]]{}"),), ())


def test_selected_pins_only_and_source_run_authority_excluded():
    state = neutral()
    state["settings"] = {"pinnedMessageIds": ["kept", "omitted"], "characterMemoryExtraction": True}
    state["settings_present"] = True
    projected = project_native_fork_context(state, (row("kept"),), ())
    assert json.loads(projected.settings_json) == {"pinnedMessageIds": ["kept"]}


def test_unknown_behavior_settings_fail_closed():
    state = {**neutral(), "settings": {"futureBehavior": True}, "settings_present": True}
    with pytest.raises(HistorySelectionError, match="unsupported_native_fork_settings"):
        project_native_fork_context(state, (), ())


def test_shared_fixed_tuple_vectors():
    for vector in json.loads(FIXTURE.read_text())["vectors"]:
        request = NativeForkRequestV1.model_validate(vector["request"])
        assert native_fork_request_tuple(request) == vector["tuple"]
        assert native_fork_request_digest(request) == vector["digest"]


@given(st.text(min_size=1, max_size=80).filter(lambda value: not any(0xD800 <= ord(c) <= 0xDFFF for c in value)))
def test_exact_unicode_titles_change_semantic_digest(title):
    body = deepcopy(json.loads(FIXTURE.read_text())["vectors"][0]["request"])
    before = native_fork_request_digest(NativeForkRequestV1.model_validate(body))
    body["child_title"] = title
    after = native_fork_request_digest(NativeForkRequestV1.model_validate(body))
    assert (before == after) is (title == "枝 🌿")


@pytest.mark.parametrize("change", ["unknown", "send", "owner", "comparison", "title", "duplicate"])
def test_strict_request_rejects_ambiguous_or_unbounded_input(change):
    body = deepcopy(json.loads(FIXTURE.read_text())["vectors"][0]["request"])
    if change == "unknown":
        body["new_field"] = True
    elif change == "send":
        body["input"]["selection"]["purpose"] = "send"
    elif change == "owner":
        body["input"]["selection"]["owner_key"] = "wrong"
    elif change == "comparison":
        body["input"]["kind"] = "comparison"
    elif change == "title":
        body["child_title"] = "x" * 501
    elif change == "duplicate":
        body["input"]["selection"]["messages"] *= 2
    with pytest.raises(ValidationError):
        NativeForkRequestV1.model_validate(body)


@pytest.fixture
def native_db(tmp_path):
    db = CharactersRAGDB(db_path=str(tmp_path / "native.sqlite"), client_id="alice")
    cid = db.add_conversation({"title": "Source", "character_id": None})
    yield db, cid
    db.close_connection()


def capture_request(cid, mid=None):
    return NativeForkCaptureRequestV1.model_validate(
        {
            "view": {
                "view_session_id": "view",
                "owner_key": "native:alice",
                "conversation_id": cid,
                "interpretation": {"kind": "parent_graph_v1"},
                "cursor": {"kind": "after_message", "message_id": mid} if mid else {"kind": "empty"},
                "selection_revision": 0,
            },
            "fidelity": "strict",
        }
    )


def owner():
    return AuthorizedNativeOwner("alice", "native:alice", NativeScopeV1(kind="global"))


def test_capture_neutral_empty_and_exact_selected_text_without_writes(native_db):
    db, cid = native_db
    empty = capture_native_fork(db, owner(), capture_request(cid))
    assert empty.input.selection.messages == ()
    mid = db.add_message({"conversation_id": cid, "sender": "assistant", "content": "Exact 停止"})
    captured = capture_native_fork(db, owner(), capture_request(cid, mid))
    assert [member.id for member in captured.input.selection.messages] == [mid]
    assert captured.retained_context_digest != empty.retained_context_digest
    assert db.get_roleplay_resume_state(cid)["message_count"] == 1


def test_capture_scope_mismatch_denied(native_db):
    db, cid = native_db
    wrong = AuthorizedNativeOwner("alice", "native:alice", NativeScopeV1(kind="workspace", workspace_id="other"))
    with pytest.raises(HistorySelectionError, match="scope_mismatch"):
        capture_native_fork(db, wrong, capture_request(cid))


def test_capture_raw_settings_formatting_does_not_change_semantic_projection(native_db):
    db, cid = native_db
    db.upsert_conversation_settings(cid, {"model": "m", "provider": "openai"})
    before = capture_native_fork(db, owner(), capture_request(cid))
    db.upsert_conversation_settings(cid, {"provider": "openai", "model": "m"})
    after = capture_native_fork(db, owner(), capture_request(cid))
    assert before.retained_context_digest == after.retained_context_digest
    assert before.input.selection.storage_context_digest != after.input.selection.storage_context_digest


def test_raw_source_row_revision_is_not_a_retained_semantic_revision():
    before = project_native_fork_context(neutral(), (row(),), ())
    after = project_native_fork_context(neutral(), ({**row(), "revision": "metadata-timestamp-only-change"},), ())
    assert before.retained_context_digest == after.retained_context_digest


def test_request_serialization_has_only_semantic_contract_fields():
    body = json.loads(FIXTURE.read_text())["vectors"][0]["request"]
    request = NativeForkRequestV1.model_validate(body)
    assert set(request.model_dump(mode="json")) == set(body)


def test_capture_embedded_images_are_ordered_typed_manifest_entries(native_db):
    import hashlib

    db, cid = native_db
    mid = db.add_message(
        {
            "conversation_id": cid,
            "sender": "user",
            "content": "two images",
            "images": [{"data": b"first", "mime": "image/png"}, {"data": b"second", "mime": "image/png"}],
        }
    )
    capture = capture_native_fork(db, owner(), capture_request(cid, mid))
    assert [asset.hash for asset in capture.asset_manifest] == [
        hashlib.sha256(b"first").hexdigest(),
        hashlib.sha256(b"second").hexdigest(),
    ]
    assert all(
        asset.representation == "embedded_image_v1" and asset.role == "message_image"
        for asset in capture.asset_manifest
    )


def test_capture_saved_retry_id_excludes_source_correlation_from_child(native_db):
    db, cid = native_db
    mid = db.add_message({
        "conversation_id": cid, "sender": "user", "content": "saved turn",
    })
    db.add_message_metadata(mid, extra={"client_message_id": "saved-turn-1"})
    capture = capture_native_fork(db, owner(), capture_request(cid, mid))
    _, _, rows = db.message_store.read_native_fork_source(
        capture_request(cid, mid).view.model_dump(mode="json"),
        owner_client_id="alice", owner_key="native:alice", scope_type="global", workspace_id=None,
    )
    assert capture.input.selection.messages[0].id == mid
    assert "client_message_id" not in project_native_fork_messages(rows)[0]["extra_metadata"]
    assert project_native_fork_context(neutral(), (row(extra={"client_message_id": "other-turn"}),), ()).retained_context_digest == project_native_fork_context(neutral(), (row(),), ()).retained_context_digest


@pytest.mark.parametrize("edited", [False, True])
def test_capture_saved_image_placeholder_preserves_detail_and_literal_edit(native_db, edited):
    db, cid = native_db
    placeholder = "<Image attachment x1>"
    mid = db.add_message({
        "conversation_id": cid, "sender": "user", "content": placeholder,
        "images": [{"data": b"image", "mime": "image/png"}],
    })
    db.add_message_metadata(mid, extra={"content_placeholder_reason": "image_attachment", "image_details": ["high"]})
    if edited:
        db.update_message(mid, {"content": placeholder}, expected_version=1)
    capture = capture_native_fork(db, owner(), capture_request(cid, mid))
    _, _, rows = db.message_store.read_native_fork_source(
        capture_request(cid, mid).view.model_dump(mode="json"),
        owner_client_id="alice", owner_key="native:alice", scope_type="global", workspace_id=None,
    )
    projected = project_native_fork_messages(rows)[0]
    assert capture.asset_manifest[0].role == "message_image"
    assert projected["message"] == (placeholder if edited else "")
    assert dict(projected["extra_metadata"])["image_details"] == ("high",)


@pytest.mark.parametrize("details", [[], ["unsupported"], "high", [None], [{}], [[]]])
def test_invalid_saved_image_details_fail_closed(details):
    image = {**row(role="user", text="caption", extra={"image_details": details}),
             "images": ["data:image/png;base64,YQ=="]}
    with pytest.raises(HistorySelectionError, match="unsupported_native_fork"):
        project_native_fork_context(neutral(), (image,), ())


def test_capture_drops_raw_row_admission_authority_from_semantic_revision(native_db):
    db, cid = native_db
    mid = db.add_message({"conversation_id": cid, "sender": "user", "content": "same"})
    before = capture_native_fork(db, owner(), capture_request(cid, mid))
    # A source storage revision alone must not become retained-message content.
    db.update_message(mid, {"content": "same"}, expected_version=1)
    after = capture_native_fork(db, owner(), capture_request(cid, mid))
    assert before.input.selection.messages == after.input.selection.messages
    assert before.retained_context_digest == after.retained_context_digest


def test_unknown_nested_prompt_carrier_rejects(accepted_state):
    state = accepted_state.state
    payload = state["behavior_snapshot"]["payload"]
    payload["participants"][0]["prompt"]["prompt_relevant_extensions"]["futureEffect"] = {"run": "external"}
    snapshot = build_behavior_snapshot(payload)
    state["behavior_snapshot"].update(payload=snapshot.payload, digest=snapshot.digest)
    values = state["settings"]["roleplayBehaviorV1"]["values"]
    values["base_snapshot"]["digest"] = snapshot.digest
    state["settings"]["roleplayBehaviorV1"] = build_materialized_behavior_settings(values)
    with pytest.raises(HistorySelectionError, match="prompt_extensions"):
        project_native_fork_context(state, (), ())


def test_tool_arguments_cannot_retain_source_credentials():
    tools = [{"id": "c", "type": "function", "function": {"name": "fetch", "arguments": '{"api_key":"source-secret"}'}}]
    rows = (row(tools=tools), row("b", role="tool", extra={"tool_call_id": "c"}))
    with pytest.raises(HistorySelectionError, match="tool_replay_credentials"):
        project_native_fork_context(neutral(), rows, ())


def test_tool_replay_parameters_must_be_valid_finite_json():
    tools = [{"id": "c", "type": "function", "function": {"name": "fetch", "arguments": '{"x":NaN}'}}]
    rows = (row(tools=tools), row("b", role="tool", extra={"tool_call_id": "c"}))
    with pytest.raises(HistorySelectionError, match="tool_replay_arguments"):
        project_native_fork_context(neutral(), rows, ())


def test_pins_and_authored_memory_are_in_effect_inventory():
    state = {
        **neutral(),
        "settings": {
            "pinnedMessageIds": ["a"],
            "authorNote": "retained note",
            "characterMemoryById": {"1": "authored"},
        },
    }
    effects = project_native_fork_context(state, (row(),), ()).required_effects
    assert {"history_pins", "author_note", "memory"} <= set(effects)


def test_capture_reports_omitted_history_pins(native_db):
    db, cid = native_db
    db.upsert_conversation_settings(cid, {"pinnedMessageIds": ["omitted"]})
    capture = capture_native_fork(db, owner(), capture_request(cid))
    assert "omitted_history_pins" in capture.reasons


def test_capture_changed_settings_during_detached_read_is_stale(native_db, monkeypatch):
    db, cid = native_db
    original = db.conversation_resume_store.get_roleplay_resume_state

    def change_after_read(*args, **kwargs):
        result = original(*args, **kwargs)
        db.upsert_conversation_settings(cid, {"model": "changed"})
        return result

    monkeypatch.setattr(db.conversation_resume_store, "get_roleplay_resume_state", change_after_read)
    with pytest.raises(HistorySelectionError, match="stale_source"):
        capture_native_fork(db, owner(), capture_request(cid))


@pytest.mark.parametrize(
    "invalid",
    [
        {"kind": "global", "workspace_id": None},
        {"kind": "global", "workspace_id": "x"},
        {"kind": "workspace"},
        {"kind": "workspace", "workspace_id": ""},
    ],
)
def test_scope_rejects_noncanonical_shapes(invalid):
    with pytest.raises(ValidationError):
        NativeScopeV1.model_validate(invalid)


def test_asset_missing_marker_requires_explicit_review_and_known_role():
    body = json.loads(FIXTURE.read_text())["vectors"][-1]["request"]
    body["fidelity"] = "strict"
    with pytest.raises(ValidationError):
        NativeForkRequestV1.model_validate(body)
    body["fidelity"] = "allow_unavailable"
    body["asset_manifest"][0]["role"] = "unknown"
    with pytest.raises(ValidationError):
        NativeForkRequestV1.model_validate(body)


def test_committed_maps_are_immutable_and_terminal_results_have_no_child():
    from pydantic import TypeAdapter

    from tldw_Server_API.app.api.v1.schemas.native_fork_schemas import NativeOperationResultV1

    adapter = TypeAdapter(NativeOperationResultV1)
    body = {
        "state": "committed",
        "operation_kind": "native_fork_v1",
        "operation_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        "owner_key": "alice",
        "child_id": "child",
        "message_map": {"old": "new"},
    }
    result = adapter.validate_python(body)
    body["message_map"]["old"] = "changed"
    assert result.message_map["old"] == "new"
    assert result.model_dump(mode="json")["message_map"] == {"old": "new"}
    with pytest.raises(TypeError):
        result.message_map["old"] = "changed"
    with pytest.raises(ValidationError):
        adapter.validate_python({**body, "state": "gone", "code": "child_gone"})


def test_historical_citations_keep_excerpt_without_source_url_or_query_authority():
    from tldw_Server_API.app.core.Chat.native_fork_projection import project_native_fork_messages

    metadata = {
        "rag_context": {
            "search_query": "old",
            "timestamp": "old",
            "settings_snapshot": {"api_key": "secret"},
            "retrieved_documents": [
                {
                    "id": "external",
                    "title": "Historical",
                    "excerpt": "Quoted text",
                    "url": "/source/private.pdf",
                    "score": 0.8,
                }
            ],
        }
    }
    projected = project_native_fork_messages((row(extra=metadata),))
    assert projected[0]["extra_metadata"]["rag_context"]["retrieved_documents"][0] == {
        "title": "Historical",
        "excerpt": "Quoted text",
    }
    assert projected[0]["message"] == "Stopped text"
    with pytest.raises(TypeError):
        projected[0]["extra_metadata"]["rag_context"]["new"] = True


@pytest.mark.parametrize(
    "settings",
    [
        {"autoSummaryEnabled": "true"},
        {"model": {"url": "live"}},
        {"turnTakingMode": "round_robin"},
        {"summary": {"enabled": "yes"}},
    ],
)
def test_malformed_or_unsupported_policy_rejects(settings):
    with pytest.raises(HistorySelectionError, match="unsupported_native_fork"):
        project_native_fork_context({**neutral(), "settings": settings}, (), ())


def test_corrupt_required_nested_controls_reject_explicitly(accepted_state):
    state = accepted_state.state
    values = state["settings"]["roleplayBehaviorV1"]["values"]
    values["behavior_controls"]["auto_summary"] = "corrupt"
    state["settings"]["roleplayBehaviorV1"] = build_materialized_behavior_settings(values)
    with pytest.raises(HistorySelectionError, match="invalid_materialized_behavior"):
        project_native_fork_context(state, (), ())


def test_capture_active_selected_node_rejects(native_db):
    db, cid = native_db
    mid = db.add_message({"conversation_id": cid, "sender": "assistant", "content": "partial"})
    # Use existing protected provenance seam on a real owned database.
    with db.transaction() as conn:
        conn.execute(
            "UPDATE messages SET history_admission_json = ? WHERE id = ?",
            (json.dumps({"version": 1, "settled": False, "interpretation": {"kind": "parent_graph_v1"}}), mid),
        )
    with pytest.raises(HistorySelectionError, match="unsettled_message"):
        capture_native_fork(db, owner(), capture_request(cid, mid))


def test_capture_explicit_legacy_projection_uses_reviewed_order(native_db):
    from tldw_Server_API.tests.DB_Management.test_history_selection_transactions import confirm_body

    db, cid = native_db
    a = db.add_message({"conversation_id": cid, "sender": "user", "content": "A"})
    b = db.add_message({"conversation_id": cid, "sender": "assistant", "content": "B"})
    snap = db.get_conversation_history_snapshot(cid, owner_client_id="alice", owner_key="native:alice")
    body = confirm_body(snap, projection_id="reviewed", path=[b, a])
    db.confirm_legacy_history_projection(body, owner_client_id="alice", owner_key="native:alice")
    request = capture_request(cid, a).model_dump(mode="json")
    request["view"]["interpretation"] = {"kind": "legacy_linear_v1", "projection_id": "reviewed"}
    captured = capture_native_fork(db, owner(), NativeForkCaptureRequestV1.model_validate(request))
    assert [member.id for member in captured.input.selection.messages] == [b, a]


def test_capture_owner_client_mismatch_denied(native_db):
    from tldw_Server_API.app.core.DB_Management.db_errors import NotFoundError

    db, cid = native_db
    wrong = AuthorizedNativeOwner("bob", "native:alice", NativeScopeV1(kind="global"))
    with pytest.raises(NotFoundError):
        capture_native_fork(db, wrong, capture_request(cid))


@pytest.mark.parametrize("source_id,allowed", [(None, True), ("source-chat", False)])
def test_category_memory_requires_proven_authored_provenance(source_id, allowed):
    state = AcceptedState.build(
        memory={
            "content": "",
            "source": "persona_memory_entries",
            "version": 1,
            "persona_memory_entries": [
                {"id": "fact", "memory_type": "fact", "content": "retained fact", "source_conversation_id": source_id}
            ],
        }
    ).state
    if allowed:
        result = project_native_fork_context(state, (), ())
        assert "retained fact" in result.snapshot_json
    else:
        with pytest.raises(HistorySelectionError, match="unclassified_derived_memory"):
            project_native_fork_context(state, (), ())


def test_materialized_pending_greeting_is_excluded_from_empty_fork(accepted_state):
    before = project_native_fork_context(accepted_state.state, (), ())
    state = accepted_state.state
    values = state["settings"]["roleplayBehaviorV1"]["values"]
    values["greeting"] = {"content": "pending greeting must not seed", "source": "first_message", "source_index": 0}
    state["settings"]["roleplayBehaviorV1"] = build_materialized_behavior_settings(values)
    after = project_native_fork_context(state, (), ())
    assert before.retained_context_digest == after.retained_context_digest
    assert "greeting" not in json.loads(after.settings_json)["roleplayBehaviorV1"]["values"]
    assert (
        json.loads(after.settings_json)["roleplayBehaviorV1"]["values"]["behavior_controls"]["greeting"]["enabled"]
        is True
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("extra_metadata", False),
        ("extra_metadata", {"sender_name": {"api_key": "secret"}}),
        ("tool_calls", {}),
        ("tool_calls", False),
    ],
)
def test_malformed_metadata_is_not_neutral(field, value):
    with pytest.raises(HistorySelectionError, match="unsupported_native_fork"):
        project_native_fork_context(neutral(), ({**row(), field: value},), ())


def test_accepted_character_sender_is_retained_as_assistant(accepted_state):
    from tldw_Server_API.app.core.Chat.native_fork_projection import project_native_fork_messages

    rows = (row(role="Legacy Character"),)
    context = project_native_fork_context(accepted_state.state, rows, ())
    projected = project_native_fork_messages(rows, context=context)
    assert projected[0]["role"] == "assistant"
    assert projected[0]["message"] == "Stopped text"


@pytest.mark.parametrize(
    "field",
    [
        "operation_id",
        "owner_key",
        "scope",
        "source_conversation_id",
        "retained_context_digest",
        "fidelity",
        "child_title",
        "message_revision",
        "cursor",
        "interpretation",
    ],
)
def test_changed_semantic_tuple_members_change_digest(field):
    body = json.loads(FIXTURE.read_text())["vectors"][0]["request"]
    before = native_fork_request_digest(NativeForkRequestV1.model_validate(body))
    selection = body["input"]["selection"]
    if field == "operation_id":
        body[field] = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
    elif field == "owner_key":
        body[field] = selection["owner_key"] = "another-owner"
    elif field == "scope":
        body[field] = {"kind": "workspace", "workspace_id": "workspace"}
    elif field == "source_conversation_id":
        body[field] = selection["conversation_id"] = "other-source"
    elif field == "retained_context_digest":
        body[field] = "b" * 64
    elif field == "fidelity":
        body[field] = "allow_unavailable"
    elif field == "child_title":
        body[field] = "New title"
    elif field == "message_revision":
        selection["messages"][0]["revision"] = "changed"
    elif field == "cursor":
        selection["cursor"]["kind"] = "before_message"
    else:
        selection["interpretation"] = {"kind": "legacy_linear_v1", "projection_id": "new-review"}
    assert native_fork_request_digest(NativeForkRequestV1.model_validate(body)) != before


def test_asset_order_and_each_reference_member_are_semantic():
    body = json.loads(FIXTURE.read_text())["vectors"][-2]["request"]
    asset = body["asset_manifest"][0]
    body["asset_manifest"].append({**asset, "reference_id": "second"})
    before = native_fork_request_digest(NativeForkRequestV1.model_validate(body))
    body["asset_manifest"].reverse()
    assert native_fork_request_digest(NativeForkRequestV1.model_validate(body)) != before
    for key, value in [
        ("reference_revision", "2"),
        ("asset_id", "different"),
        ("revision", "2"),
        ("representation", "document_original_v1"),
        ("context_enabled", False),
        ("hash", "c" * 64),
    ]:
        changed = deepcopy(body)
        changed["asset_manifest"].reverse()
        changed["asset_manifest"][0][key] = value
        assert native_fork_request_digest(NativeForkRequestV1.model_validate(changed)) != before


def test_capture_corrupt_required_settings_does_not_become_absent(native_db):
    db, cid = native_db
    db.upsert_conversation_settings(cid, {"model": "m"})
    with db.transaction() as conn:
        conn.execute("UPDATE conversation_settings SET settings_json = ? WHERE conversation_id = ?", ("not json", cid))
    with pytest.raises(HistorySelectionError, match="invalid_settings"):
        capture_native_fork(db, owner(), capture_request(cid))


@pytest.mark.parametrize(
    "action,replacement,original_hash,valid",
    [
        ("remove_from_context", None, None, True),
        ("remove_from_context", {"asset_id": "a", "revision": "1"}, None, False),
        ("replace", {"asset_id": "a", "revision": "1"}, None, True),
        ("replace", None, None, False),
        ("restore", {"asset_id": "a", "revision": "1"}, "c" * 64, True),
        ("restore", {"asset_id": "a", "revision": "1"}, None, False),
    ],
)
def test_asset_context_update_requires_reviewed_identity(action, replacement, original_hash, valid):
    from tldw_Server_API.app.api.v1.schemas.native_fork_schemas import NativeAssetContextUpdateV1

    body = {
        "expected_asset_manifest_revision": "1",
        "target_reference_id": "r",
        "expected_reference_revision": "1",
        "expected_asset_revision": "1",
        "action": action,
        "replacement": replacement,
        "expected_original_hash": original_hash,
    }
    if valid:
        assert NativeAssetContextUpdateV1.model_validate(body).action == action
    else:
        with pytest.raises(ValidationError):
            NativeAssetContextUpdateV1.model_validate(body)


def test_capture_legacy_primary_image_preserves_exact_reference_identity(native_db):
    db, cid = native_db
    mid = db.add_message(
        {
            "conversation_id": cid,
            "sender": "user",
            "content": "legacy image",
            "image_data": b"primary",
            "image_mime_type": "image/png",
        }
    )
    snapshot = db.get_conversation_history_snapshot(cid, owner_client_id="alice", owner_key="native:alice")
    original = snapshot.nodes[0]["assets"][0]
    capture = capture_native_fork(db, owner(), capture_request(cid, mid))
    assert capture.asset_manifest[0].reference_id == original["id"]
    assert capture.asset_manifest[0].reference_revision == original["revision"]


def test_accepted_materialized_participant_name_maps_without_live_lookup(accepted_state):
    state = accepted_state.state
    values = state["settings"]["roleplayBehaviorV1"]["values"]
    values["participants"] = deepcopy(state["behavior_snapshot"]["payload"]["participants"])
    values["participants"][0]["identity"]["name"] = "Accepted Rename"
    state["settings"]["roleplayBehaviorV1"] = build_materialized_behavior_settings(values)
    context = project_native_fork_context(state, (row(role="Accepted Rename"),), ())
    assert context.binding.display_name == "Accepted Rename"


def test_corrupt_snapshot_participant_identity_rejects_with_fork_error(accepted_state):
    state = accepted_state.state
    state["behavior_snapshot"]["payload"]["participants"][0]["identity"] = None
    with pytest.raises(HistorySelectionError, match="unsupported_native_fork_invalid_snapshot"):
        project_native_fork_context(state, (), ())


def test_snapshot_bound_child_can_be_forked_again_without_live_card_identity(accepted_state):
    state = accepted_state.state
    state["conversation"].update(
        id="native-child",
        character_id=None,
        assistant_kind="character",
        assistant_id="snapshot:native-child",
        assistant_binding_mode="snapshot_v1",
    )
    projected = project_native_fork_context(state, (), ())
    assert projected.binding.primary_participant_id == "1"


def test_unprotected_snapshot_identity_cannot_impersonate_native_child(accepted_state):
    state = accepted_state.state
    state["conversation"].update(
        id="native-child",
        character_id=None,
        assistant_kind="character",
        assistant_id="snapshot:native-child",
    )
    with pytest.raises(HistorySelectionError, match="unsupported_native_fork_participants"):
        project_native_fork_context(state, (), ())


@pytest.mark.parametrize("reason", ["future_prefill", "tool_calls"])
def test_placeholder_metadata_cannot_hide_retained_text_without_closed_replay(reason):
    with pytest.raises(HistorySelectionError, match="unsupported_native_fork_tool_replay"):
        project_native_fork_context(neutral(), (row(extra={"content_placeholder_reason": reason}),), ())
