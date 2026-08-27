from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Notes_Graph.suggestion_content import content_fingerprint

pytestmark = pytest.mark.unit

DATASET_ID = "dataset-1"
NOW = datetime(2026, 8, 27, 16, 0, tzinfo=timezone.utc)
SOURCE_ID = "00000000-0000-4000-8000-000000000001"
TARGET_ID = "00000000-0000-4000-8000-000000000002"
OTHER_ID = "00000000-0000-4000-8000-000000000003"


@pytest.fixture()
def db(tmp_path) -> CharactersRAGDB:
    database = CharactersRAGDB(str(tmp_path / "suggestions.db"), client_id="owner-1")
    database.add_note("Source", "source body", note_id=SOURCE_ID)
    database.add_note("Target", "target body", note_id=TARGET_ID)
    database.add_note("Other", "other body", note_id=OTHER_ID)
    with database.transaction() as conn:
        conn.execute(
            "INSERT INTO note_task_scope_authority(owner_user_id,dataset_id) VALUES (?,?)",
            (database.client_id, DATASET_ID),
        )
    try:
        yield database
    finally:
        database.close_all_connections()


def _fingerprint(db: CharactersRAGDB, note_id: str) -> str:
    note = db.get_note_by_id(note_id, include_deleted=True)
    assert note is not None
    return content_fingerprint(note["title"], note["content"])


def _admit(
    db: CharactersRAGDB,
    *,
    key: str,
    provider: str = "openai",
    model: str = "model-a",
    source_id: str = SOURCE_ID,
):
    return db.note_graph_suggestion_store.admit_run(
        dataset_id=DATASET_ID,
        source_note_id=source_id,
        source_fingerprint=_fingerprint(db, source_id),
        provider=provider,
        model=model,
        capability_revision="cap-v1",
        prompt_contract_version="prompt-v1",
        idempotency_key=key,
        now=NOW,
    )


def _queue_and_run(db: CharactersRAGDB, admission):
    queued = db.note_graph_suggestion_store.bind_admitted_run(
        dataset_id=DATASET_ID,
        run_id=admission.run.id,
        expected_state="admitting",
        expected_revision=admission.run.revision,
        job_id=f"job-{admission.run.id}",
        completion_token=f"completion-{admission.run.id}",
        replay_envelope={"run_id": admission.run.id, "state": "queued"},
        now=NOW,
    )
    return db.note_graph_suggestion_store.start_run(
        dataset_id=DATASET_ID,
        run_id=queued.id,
        expected_state="queued",
        expected_revision=queued.revision,
        expected_job_id=queued.job_id,
        acquired_completion_token=f"worker-{admission.run.id}",
        now=NOW,
    )


def _tag_candidate(
    *,
    suggestion_id: str,
    normalized_tag: str,
    display_tag: str,
    keyword_sync_id: str | None = None,
):
    return {
        "id": suggestion_id,
        "kind": "tag",
        "normalized_tag": normalized_tag,
        "display_tag": display_tag,
        "keyword_sync_id": keyword_sync_id,
        "match_strength": "possible",
        "rationale": "Bounded rationale",
        "evidence": (),
    }


def _related_candidate(db: CharactersRAGDB, *, suggestion_id: str, target_id: str = TARGET_ID):
    return {
        "id": suggestion_id,
        "kind": "related_note",
        "target_note_id": target_id,
        "target_fingerprint": _fingerprint(db, target_id),
        "match_strength": "strong",
        "rationale": "Bounded rationale",
        "evidence": (
            {
                "side": "source",
                "ordinal": 0,
                "note_id": SOURCE_ID,
                "field": "content",
                "content_fingerprint": _fingerprint(db, SOURCE_ID),
                "start_offset": 0,
                "end_offset": 6,
            },
            {
                "side": "target",
                "ordinal": 0,
                "note_id": target_id,
                "field": "content",
                "content_fingerprint": _fingerprint(db, target_id),
                "start_offset": 0,
                "end_offset": 6,
            },
        ),
    }


def _stage_and_activate(
    db: CharactersRAGDB,
    *,
    key: str,
    suggestion_id: str,
    target_id: str = TARGET_ID,
):
    running = _queue_and_run(db, _admit(db, key=key))
    publishing = db.note_graph_suggestion_store.stage_suggestions(
        dataset_id=DATASET_ID,
        run_id=running.id,
        expected_state="running",
        expected_revision=running.revision,
        expected_job_id=running.job_id,
        expected_completion_token=running.expected_completion_token,
        result_digest=f"sha256:{'1' * 64}",
        candidates=(_related_candidate(db, suggestion_id=suggestion_id, target_id=target_id),),
        invalid_item_count=0,
        now=NOW,
    )
    activated = db.note_graph_suggestion_store.activate_staged_run(
        dataset_id=DATASET_ID,
        run_id=publishing.id,
        expected_state="publishing",
        expected_revision=publishing.revision,
        observed_job_id=publishing.job_id,
        observed_completion_token=publishing.expected_completion_token,
        observed_result_digest=publishing.result_digest,
        now=NOW,
    )
    return activated


def test_request_fingerprints_and_key_digests_are_versioned_bounded_and_safe(db) -> None:
    store = db.note_graph_suggestion_store
    first = store.canonical_request_fingerprint(
        "run_admit",
        {
            "source_note_id": SOURCE_ID,
            "source_fingerprint": _fingerprint(db, SOURCE_ID),
            "provider": "openai",
            "model": "model-a",
            "capability_revision": "cap-v1",
            "prompt_contract_version": "prompt-v1",
        },
    )
    second = store.canonical_request_fingerprint(
        "run_admit",
        {
            "model": "model-a",
            "provider": "openai",
            "capability_revision": "cap-v1",
            "prompt_contract_version": "prompt-v1",
            "source_fingerprint": _fingerprint(db, SOURCE_ID),
            "source_note_id": SOURCE_ID,
        },
    )

    assert first == second
    assert first.startswith("sha256:") and len(first) == 71
    assert store.idempotency_key_digest("bounded-key") == store.idempotency_key_digest("bounded-key")
    with pytest.raises(ValueError, match="notes_graph_request_contract_invalid"):
        store.canonical_request_fingerprint("run_admit", {"note_text": "private body"})
    with pytest.raises(ValueError, match="notes_graph_request_contract_invalid"):
        store.canonical_request_fingerprint(
            "run_admit",
            {
                "source_note_id": SOURCE_ID,
                "source_fingerprint": _fingerprint(db, SOURCE_ID),
                "provider": "openai",
                "model": "model-a",
                "capability_revision": "cap-v1",
                "prompt_contract_version": "prompt-v1",
                "authorization_material": "alternate secret name",
            },
        )
    with pytest.raises(ValueError, match="notes_graph_request_contract_invalid"):
        store.canonical_request_fingerprint("unknown_operation", {})
    with pytest.raises(ValueError, match="notes_graph_replay_envelope_invalid"):
        store._encode_envelope("run_admit", {"run_id": SOURCE_ID, "state": "queued", "extra": 1})
    with pytest.raises(ValueError, match="notes_graph_replay_envelope_invalid"):
        store._encode_envelope("run_admit", {"run_id": 7, "state": "queued"})
    with pytest.raises(ValueError, match="notes_graph_replay_envelope_invalid"):
        store._encode_envelope("run_admit", ["wrong envelope type"])
    with pytest.raises(ValueError, match="notes_graph_replay_envelope_invalid"):
        store._encode_envelope(
            "run_admit",
            {
                "run_id": SOURCE_ID,
                "state": "failed",
                "error_code": "provider said arbitrary private text",
                "guidance_key": "retry_generation",
            },
        )
    with pytest.raises(ValueError, match="notes_graph_replay_envelope_invalid"):
        store._encode_envelope(
            "run_admit",
            {
                "run_id": SOURCE_ID,
                "state": "failed",
                "error_code": "notes_graph_admission_failed",
                "guidance_key": "arbitrary provider guidance",
            },
        )
    with pytest.raises(ValueError, match="notes_graph_replay_envelope_invalid"):
        store._encode_envelope(
            "run_admit",
            {"run_id": "x" * 257, "state": "queued"},
        )
    with pytest.raises(ValueError, match="notes_graph_idempotency_key_invalid"):
        store.idempotency_key_digest("x" * 257)


def test_active_run_identity_uses_full_approved_tuple_and_reports_conflicts(db) -> None:
    first = _admit(db, key="key-a", model="model-a")
    different_model = _admit(db, key="key-b", model="model-b")

    assert first.run.id != different_model.run.id
    with pytest.raises(RuntimeError, match="notes_graph_active_run_conflict"):
        _admit(db, key="key-c", model="model-a")


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("provider", None),
        ("provider", ""),
        ("provider", "   "),
        ("model", None),
        ("model", ""),
        ("capability_revision", None),
        ("capability_revision", " "),
        ("prompt_contract_version", None),
        ("prompt_contract_version", ""),
    ),
)
def test_admission_rejects_missing_or_blank_binding_contract_fields(db, field, value) -> None:
    fields = {
        "provider": "openai",
        "model": "model-a",
        "capability_revision": "cap-v1",
        "prompt_contract_version": "prompt-v1",
    }
    fields[field] = value
    with pytest.raises(ValueError, match="notes_graph_admission_contract_invalid"):
        db.note_graph_suggestion_store.admit_run(
            dataset_id=DATASET_ID,
            source_note_id=SOURCE_ID,
            source_fingerprint=_fingerprint(db, SOURCE_ID),
            idempotency_key=f"invalid-{field}-{value!r}",
            now=NOW,
            **fields,
        )


def test_explicit_run_transition_and_admission_failure_are_fenced_and_receipt_atomic(db) -> None:
    admission = _admit(db, key="admission-failure")
    with pytest.raises(ValueError, match="notes_graph_run_transition_invalid"):
        db.note_graph_suggestion_store.start_run(
            dataset_id=DATASET_ID,
            run_id=admission.run.id,
            expected_state="admitting",
            expected_revision=admission.run.revision,
            expected_job_id="job-invalid-transition",
            acquired_completion_token="worker-invalid-transition",
            now=NOW,
        )

    failed = db.note_graph_suggestion_store.fail_admission(
        dataset_id=DATASET_ID,
        run_id=admission.run.id,
        expected_state="admitting",
        expected_revision=admission.run.revision,
        error_code="notes_graph_admission_failed",
        guidance_key="retry_generation",
        now=NOW,
    )
    assert failed.state.value == "failed"
    receipt = db.execute_query(
        "SELECT state,http_status,replay_envelope FROM note_graph_suggestion_operation_receipts "
        "WHERE id=?",
        (admission.run.admission_receipt_id,),
    ).fetchone()
    assert receipt["state"] == "failed"
    assert int(receipt["http_status"]) == 503
    assert json.loads(receipt["replay_envelope"]) == {
        "error_code": "notes_graph_admission_failed",
        "guidance_key": "retry_generation",
        "run_id": admission.run.id,
        "state": "failed",
    }

    db.execute_query(
        "DELETE FROM note_graph_suggestion_runs WHERE id=?",
        (admission.run.id,),
    )
    replay = _admit(db, key="admission-failure")
    assert replay.disposition == "terminal_replay"
    assert replay.run is None
    assert replay.replay_envelope == json.loads(receipt["replay_envelope"])

    with pytest.raises(RuntimeError, match="notes_graph_suggestion_idempotency_mismatch"):
        _admit(db, key="admission-failure", model="model-b")


def test_missing_in_progress_admission_resource_is_operation_specific(db) -> None:
    admission = _admit(db, key="missing-admission-run")
    db.execute_query("DELETE FROM note_graph_suggestion_runs WHERE id=?", (admission.run.id,))
    with pytest.raises(RuntimeError, match="notes_graph_run_admit_resource_missing"):
        _admit(db, key="missing-admission-run")


def test_receipt_replay_is_terminal_or_operation_specific_and_mismatch_is_stable(db) -> None:
    in_progress = _admit(db, key="receipt-key")
    replay = _admit(db, key="receipt-key")

    assert replay.disposition == "in_progress"
    assert replay.continuation == "resume_run_admission"
    assert replay.replay_envelope is None

    queued = db.note_graph_suggestion_store.bind_admitted_run(
        dataset_id=DATASET_ID,
        run_id=in_progress.run.id,
        expected_state="admitting",
        expected_revision=in_progress.run.revision,
        job_id="job-receipt",
        completion_token="completion-receipt",
        replay_envelope={"run_id": in_progress.run.id, "state": "queued"},
        now=NOW,
    )
    terminal = _admit(db, key="receipt-key")
    assert queued.state.value == "queued"
    assert terminal.disposition == "terminal_replay"
    assert terminal.replay_envelope == {"run_id": in_progress.run.id, "state": "queued"}

    with pytest.raises(RuntimeError, match="notes_graph_suggestion_idempotency_mismatch"):
        db.note_graph_suggestion_store.admit_run(
            dataset_id=DATASET_ID,
            source_note_id=SOURCE_ID,
            source_fingerprint=_fingerprint(db, SOURCE_ID),
            provider="openai",
            model="different-model",
            capability_revision="cap-v1",
            prompt_contract_version="prompt-v1",
            idempotency_key="receipt-key",
            now=NOW,
        )


def test_staged_rows_are_hidden_and_activation_is_atomic_and_supersedes(db) -> None:
    first = _stage_and_activate(db, key="publish-a", suggestion_id="suggestion-a")
    assert first.state.value == "succeeded"

    running = _queue_and_run(db, _admit(db, key="publish-b", model="model-b"))
    publishing = db.note_graph_suggestion_store.stage_suggestions(
        dataset_id=DATASET_ID,
        run_id=running.id,
        expected_state="running",
        expected_revision=running.revision,
        expected_job_id=running.job_id,
        expected_completion_token=running.expected_completion_token,
        result_digest=f"sha256:{'2' * 64}",
        candidates=(_related_candidate(db, suggestion_id="suggestion-b"),),
        invalid_item_count=0,
        now=NOW + timedelta(minutes=1),
    )

    hidden = db.note_graph_suggestion_store.list_suggestions(
        dataset_id=DATASET_ID,
        source_note_id=SOURCE_ID,
        source_fingerprint=_fingerprint(db, SOURCE_ID),
        states=("pending", "accepting"),
        limit=100,
        cursor=None,
        cursor_secret=b"test-cursor-secret",
    )
    assert [item.id for item in hidden.items] == ["suggestion-a"]

    db.note_graph_suggestion_store.activate_staged_run(
        dataset_id=DATASET_ID,
        run_id=publishing.id,
        expected_state="publishing",
        expected_revision=publishing.revision,
        observed_job_id=publishing.job_id,
        observed_completion_token=publishing.expected_completion_token,
        observed_result_digest=publishing.result_digest,
        now=NOW + timedelta(minutes=1),
    )
    rows = db.execute_query(
        "SELECT id,state,decision_reason FROM note_graph_suggestions ORDER BY id"
    ).fetchall()
    assert [(row["id"], row["state"]) for row in rows] == [
        ("suggestion-a", "stale"),
        ("suggestion-b", "pending"),
    ]
    assert rows[0]["decision_reason"] == "superseded_by_run"


def test_reverse_orientation_is_canonical_for_supersession_and_rejection_suppression(db) -> None:
    _stage_and_activate(db, key="forward-run", suggestion_id="forward")
    reverse_running = _queue_and_run(
        db,
        _admit(db, key="reverse-run", source_id=TARGET_ID),
    )
    reverse_publishing = db.note_graph_suggestion_store.stage_suggestions(
        dataset_id=DATASET_ID,
        run_id=reverse_running.id,
        expected_state="running",
        expected_revision=reverse_running.revision,
        expected_job_id=reverse_running.job_id,
        expected_completion_token=reverse_running.expected_completion_token,
        result_digest=f"sha256:{'a' * 64}",
        candidates=(
            {
                "id": "reverse",
                "kind": "related_note",
                "target_note_id": SOURCE_ID,
                "target_fingerprint": _fingerprint(db, SOURCE_ID),
                "match_strength": "strong",
                "rationale": "Bounded rationale",
                "evidence": (),
            },
        ),
        invalid_item_count=0,
        now=NOW,
    )
    db.note_graph_suggestion_store.activate_staged_run(
        dataset_id=DATASET_ID,
        run_id=reverse_publishing.id,
        expected_state="publishing",
        expected_revision=reverse_publishing.revision,
        observed_job_id=reverse_publishing.job_id,
        observed_completion_token=reverse_publishing.expected_completion_token,
        observed_result_digest=reverse_publishing.result_digest,
        now=NOW,
    )
    assert [tuple(row) for row in db.execute_query(
        "SELECT id,state FROM note_graph_suggestions ORDER BY id"
    ).fetchall()] == [("forward", "stale"), ("reverse", "pending")]

    db.note_graph_suggestion_store.reject_suggestion(
        dataset_id=DATASET_ID,
        suggestion_id="reverse",
        expected_revision=1,
        expected_source_fingerprint=_fingerprint(db, TARGET_ID),
        expected_target_fingerprint=_fingerprint(db, SOURCE_ID),
        idempotency_key="reverse-reject",
        now=NOW,
    )
    forward_running = _queue_and_run(db, _admit(db, key="forward-suppressed", model="model-b"))
    forward_publishing = db.note_graph_suggestion_store.stage_suggestions(
        dataset_id=DATASET_ID,
        run_id=forward_running.id,
        expected_state="running",
        expected_revision=forward_running.revision,
        expected_job_id=forward_running.job_id,
        expected_completion_token=forward_running.expected_completion_token,
        result_digest=f"sha256:{'b' * 64}",
        candidates=(_related_candidate(db, suggestion_id="suppressed-forward"),),
        invalid_item_count=0,
        now=NOW,
    )
    activated = db.note_graph_suggestion_store.activate_staged_run(
        dataset_id=DATASET_ID,
        run_id=forward_publishing.id,
        expected_state="publishing",
        expected_revision=forward_publishing.revision,
        observed_job_id=forward_publishing.job_id,
        observed_completion_token=forward_publishing.expected_completion_token,
        observed_result_digest=forward_publishing.result_digest,
        now=NOW,
    )
    assert activated.suggestion_count == 0
    assert db.execute_query(
        "SELECT COUNT(*) AS count FROM note_graph_suggestions WHERE id='suppressed-forward'"
    ).fetchone()["count"] == 0


def test_existing_tag_rename_delete_and_current_membership_are_rechecked_atomically(db) -> None:
    keyword_id = db.add_keyword("Research")
    keyword = db.get_keyword_by_id(keyword_id)
    assert keyword is not None

    rename_running = _queue_and_run(db, _admit(db, key="tag-rename"))
    rename_publishing = db.note_graph_suggestion_store.stage_suggestions(
        dataset_id=DATASET_ID,
        run_id=rename_running.id,
        expected_state="running",
        expected_revision=rename_running.revision,
        expected_job_id=rename_running.job_id,
        expected_completion_token=rename_running.expected_completion_token,
        result_digest=f"sha256:{'c' * 64}",
        candidates=(
            _tag_candidate(
                suggestion_id="renamed-tag",
                normalized_tag="research",
                display_tag="Research",
                keyword_sync_id=keyword["sync_id"],
            ),
        ),
        invalid_item_count=0,
        now=NOW,
    )
    renamed = db.rename_keyword(keyword_id, "Deep Research", expected_version=1)
    activated = db.note_graph_suggestion_store.activate_staged_run(
        dataset_id=DATASET_ID,
        run_id=rename_publishing.id,
        expected_state="publishing",
        expected_revision=rename_publishing.revision,
        observed_job_id=rename_publishing.job_id,
        observed_completion_token=rename_publishing.expected_completion_token,
        observed_result_digest=rename_publishing.result_digest,
        now=NOW,
    )
    assert activated.state.value == "succeeded"
    renamed_row = db.execute_query(
        "SELECT state,normalized_tag,display_tag,keyword_sync_id FROM note_graph_suggestions "
        "WHERE id='renamed-tag'"
    ).fetchone()
    assert tuple(renamed_row) == ("pending", "deep research", "Deep Research", renamed["sync_id"])

    deleted_id = db.add_keyword("Disposable")
    deleted_keyword = db.get_keyword_by_id(deleted_id)
    assert deleted_keyword is not None
    delete_running = _queue_and_run(db, _admit(db, key="tag-delete", model="model-b"))
    delete_publishing = db.note_graph_suggestion_store.stage_suggestions(
        dataset_id=DATASET_ID,
        run_id=delete_running.id,
        expected_state="running",
        expected_revision=delete_running.revision,
        expected_job_id=delete_running.job_id,
        expected_completion_token=delete_running.expected_completion_token,
        result_digest=f"sha256:{'d' * 64}",
        candidates=(
            _tag_candidate(
                suggestion_id="deleted-tag",
                normalized_tag="disposable",
                display_tag="Disposable",
                keyword_sync_id=deleted_keyword["sync_id"],
            ),
            _tag_candidate(
                suggestion_id="surviving-tag",
                normalized_tag="new tag",
                display_tag="New Tag",
            ),
        ),
        invalid_item_count=0,
        now=NOW,
    )
    assert db.soft_delete_keyword(deleted_id, expected_version=1)
    delete_activated = db.note_graph_suggestion_store.activate_staged_run(
        dataset_id=DATASET_ID,
        run_id=delete_publishing.id,
        expected_state="publishing",
        expected_revision=delete_publishing.revision,
        observed_job_id=delete_publishing.job_id,
        observed_completion_token=delete_publishing.expected_completion_token,
        observed_result_digest=delete_publishing.result_digest,
        now=NOW,
    )
    assert delete_activated.state.value == "succeeded"
    assert delete_activated.suggestion_count == 1
    assert [tuple(row) for row in db.execute_query(
        "SELECT id,state FROM note_graph_suggestions WHERE run_id=? ORDER BY id",
        (delete_publishing.id,),
    ).fetchall()] == [("surviving-tag", "pending")]

    member_id = db.add_keyword("Already Present")
    member_keyword = db.get_keyword_by_id(member_id)
    assert member_keyword is not None
    assert db.link_note_to_keyword(SOURCE_ID, member_id)
    member_running = _queue_and_run(db, _admit(db, key="tag-member", model="model-c"))
    member_publishing = db.note_graph_suggestion_store.stage_suggestions(
        dataset_id=DATASET_ID,
        run_id=member_running.id,
        expected_state="running",
        expected_revision=member_running.revision,
        expected_job_id=member_running.job_id,
        expected_completion_token=member_running.expected_completion_token,
        result_digest=f"sha256:{'e' * 64}",
        candidates=(
            _tag_candidate(
                suggestion_id="present-tag",
                normalized_tag="already present",
                display_tag="Already Present",
                keyword_sync_id=member_keyword["sync_id"],
            ),
        ),
        invalid_item_count=0,
        now=NOW,
    )
    member_activated = db.note_graph_suggestion_store.activate_staged_run(
        dataset_id=DATASET_ID,
        run_id=member_publishing.id,
        expected_state="publishing",
        expected_revision=member_publishing.revision,
        observed_job_id=member_publishing.job_id,
        observed_completion_token=member_publishing.expected_completion_token,
        observed_result_digest=member_publishing.result_digest,
        now=NOW,
    )
    assert member_activated.suggestion_count == 0


def test_freshness_failure_discards_entire_staged_set_without_partial_activation(db) -> None:
    running = _queue_and_run(db, _admit(db, key="stale-stage"))
    publishing = db.note_graph_suggestion_store.stage_suggestions(
        dataset_id=DATASET_ID,
        run_id=running.id,
        expected_state="running",
        expected_revision=running.revision,
        expected_job_id=running.job_id,
        expected_completion_token=running.expected_completion_token,
        result_digest=f"sha256:{'3' * 64}",
        candidates=(
            _related_candidate(db, suggestion_id="fresh-candidate"),
            _related_candidate(db, suggestion_id="stale-candidate", target_id=OTHER_ID),
        ),
        invalid_item_count=0,
        now=NOW,
    )
    db.update_note(OTHER_ID, {"content": "changed"}, expected_version=1)

    stale = db.note_graph_suggestion_store.activate_staged_run(
        dataset_id=DATASET_ID,
        run_id=publishing.id,
        expected_state="publishing",
        expected_revision=publishing.revision,
        observed_job_id=publishing.job_id,
        observed_completion_token=publishing.expected_completion_token,
        observed_result_digest=publishing.result_digest,
        now=NOW,
    )

    assert stale.state.value == "stale"
    assert db.execute_query(
        "SELECT COUNT(*) AS count FROM note_graph_suggestions WHERE run_id = ?",
        (publishing.id,),
    ).fetchone()["count"] == 0


def test_reject_and_reset_are_receipt_atomic_with_monotonic_revision(db) -> None:
    _stage_and_activate(db, key="reject-run", suggestion_id="reject-me")
    rejected = db.note_graph_suggestion_store.reject_suggestion(
        dataset_id=DATASET_ID,
        suggestion_id="reject-me",
        expected_revision=1,
        expected_source_fingerprint=_fingerprint(db, SOURCE_ID),
        expected_target_fingerprint=_fingerprint(db, TARGET_ID),
        idempotency_key="reject-key",
        now=NOW,
    )
    assert rejected.envelope == {"suggestion_id": "reject-me", "state": "rejected", "revision": 2}
    assert rejected.rejection_set.revision == 1
    assert rejected.rejection_set.rejection_count == 1
    assert db.execute_query(
        "SELECT COUNT(*) AS count FROM note_graph_suggestion_evidence WHERE suggestion_id = ?",
        ("reject-me",),
    ).fetchone()["count"] == 0
    assert db.execute_query(
        "SELECT rationale FROM note_graph_suggestions WHERE id = ?", ("reject-me",)
    ).fetchone()["rationale"] is None

    replay = db.note_graph_suggestion_store.reject_suggestion(
        dataset_id=DATASET_ID,
        suggestion_id="reject-me",
        expected_revision=1,
        expected_source_fingerprint=_fingerprint(db, SOURCE_ID),
        expected_target_fingerprint=_fingerprint(db, TARGET_ID),
        idempotency_key="reject-key",
        now=NOW,
    )
    assert replay.disposition == "terminal_replay"
    assert replay.envelope == rejected.envelope

    reset = db.note_graph_suggestion_store.reset_rejections(
        dataset_id=DATASET_ID,
        source_note_id=SOURCE_ID,
        source_fingerprint=_fingerprint(db, SOURCE_ID),
        expected_revision=1,
        idempotency_key="reset-key-a",
        now=NOW,
    )
    assert reset.envelope["cleared_count"] == 1
    assert reset.rejection_set.revision == 2
    assert reset.rejection_set.rejection_count == 0

    zero_reset = db.note_graph_suggestion_store.reset_rejections(
        dataset_id=DATASET_ID,
        source_note_id=SOURCE_ID,
        source_fingerprint=_fingerprint(db, SOURCE_ID),
        expected_revision=2,
        idempotency_key="reset-key-b",
        now=NOW,
    )
    assert zero_reset.envelope["cleared_count"] == 0
    assert zero_reset.rejection_set.revision == 3
    with pytest.raises(RuntimeError, match="notes_graph_rejection_set_conflict"):
        db.note_graph_suggestion_store.reset_rejections(
            dataset_id=DATASET_ID,
            source_note_id=SOURCE_ID,
            source_fingerprint=_fingerprint(db, SOURCE_ID),
            expected_revision=2,
            idempotency_key="reset-key-c",
            now=NOW,
        )


def test_reject_terminal_receipt_replays_after_obsolete_detail_cleanup(db) -> None:
    _stage_and_activate(db, key="cleanup-reject-run", suggestion_id="cleanup-reject")
    rejected = db.note_graph_suggestion_store.reject_suggestion(
        dataset_id=DATASET_ID,
        suggestion_id="cleanup-reject",
        expected_revision=1,
        expected_source_fingerprint=_fingerprint(db, SOURCE_ID),
        expected_target_fingerprint=_fingerprint(db, TARGET_ID),
        idempotency_key="cleanup-reject-key",
        now=NOW,
    )
    with db.transaction() as conn:
        conn.execute("UPDATE notes SET content='target changed without hook' WHERE id=?", (TARGET_ID,))
    counts = db.note_graph_suggestion_store.cleanup_retention(
        dataset_id=DATASET_ID,
        now=NOW + timedelta(days=31),
        limit=100,
    )
    assert counts["suggestions"] == 1

    replay = db.note_graph_suggestion_store.reject_suggestion(
        dataset_id=DATASET_ID,
        suggestion_id="cleanup-reject",
        expected_revision=1,
        expected_source_fingerprint=rejected.rejection_set.source_fingerprint,
        expected_target_fingerprint=content_fingerprint("Target", "target body"),
        idempotency_key="cleanup-reject-key",
        now=NOW + timedelta(days=31),
    )
    assert replay.disposition == "terminal_replay"
    assert replay.envelope == rejected.envelope
    assert replay.suggestion is None
    with pytest.raises(RuntimeError, match="notes_graph_suggestion_idempotency_mismatch"):
        db.note_graph_suggestion_store.reject_suggestion(
            dataset_id=DATASET_ID,
            suggestion_id="cleanup-reject",
            expected_revision=2,
            expected_source_fingerprint=rejected.rejection_set.source_fingerprint,
            expected_target_fingerprint=content_fingerprint("Target", "target body"),
            idempotency_key="cleanup-reject-key",
            now=NOW + timedelta(days=31),
        )


def test_acceptance_lease_is_five_minutes_and_stale_fence_cannot_mutate(db) -> None:
    _stage_and_activate(db, key="accept-run", suggestion_id="accept-me")
    first = db.note_graph_suggestion_store.claim_acceptance(
        dataset_id=DATASET_ID,
        suggestion_id="accept-me",
        expected_revision=1,
        expected_source_fingerprint=_fingerprint(db, SOURCE_ID),
        expected_target_fingerprint=_fingerprint(db, TARGET_ID),
        idempotency_key="accept-key-a",
        now=NOW,
    )
    assert first.suggestion.revision == 2
    assert first.suggestion.acceptance_lease_expires_at == (NOW + timedelta(minutes=5)).isoformat()

    second = db.note_graph_suggestion_store.reclaim_expired_acceptance(
        dataset_id=DATASET_ID,
        suggestion_id="accept-me",
        decision_receipt_id=first.suggestion.decision_receipt_id,
        expected_state="accepting",
        expected_revision=2,
        expected_lease_token=first.suggestion.acceptance_lease_token,
        now=NOW + timedelta(minutes=5, seconds=1),
    )
    assert second.revision == 3
    assert second.acceptance_lease_token != first.suggestion.acceptance_lease_token
    with pytest.raises(RuntimeError, match="notes_graph_suggestion_conflict"):
        db.note_graph_suggestion_store.release_acceptance(
            dataset_id=DATASET_ID,
            suggestion_id="accept-me",
            decision_receipt_id=first.suggestion.decision_receipt_id,
            expected_state="accepting",
            expected_revision=2,
            expected_lease_token=first.suggestion.acceptance_lease_token,
            now=NOW + timedelta(minutes=5, seconds=2),
        )


def test_acceptance_replay_and_fences_never_expose_another_receipts_lease(db) -> None:
    _stage_and_activate(db, key="lease-isolation-run", suggestion_id="lease-isolation")
    first = db.note_graph_suggestion_store.claim_acceptance(
        dataset_id=DATASET_ID,
        suggestion_id="lease-isolation",
        expected_revision=1,
        expected_source_fingerprint=_fingerprint(db, SOURCE_ID),
        expected_target_fingerprint=_fingerprint(db, TARGET_ID),
        idempotency_key="old-accept-key",
        now=NOW,
    )
    released = db.note_graph_suggestion_store.release_acceptance(
        dataset_id=DATASET_ID,
        suggestion_id="lease-isolation",
        decision_receipt_id=first.suggestion.decision_receipt_id,
        expected_state="accepting",
        expected_revision=first.suggestion.revision,
        expected_lease_token=first.suggestion.acceptance_lease_token,
        now=NOW + timedelta(minutes=1),
    )
    assert released.disposition == "completed"
    assert released.envelope == {
        "revision": 3,
        "state": "pending",
        "suggestion_id": "lease-isolation",
    }

    second = db.note_graph_suggestion_store.claim_acceptance(
        dataset_id=DATASET_ID,
        suggestion_id="lease-isolation",
        expected_revision=3,
        expected_source_fingerprint=_fingerprint(db, SOURCE_ID),
        expected_target_fingerprint=_fingerprint(db, TARGET_ID),
        idempotency_key="new-accept-key",
        now=NOW + timedelta(minutes=2),
    )
    old_replay = db.note_graph_suggestion_store.claim_acceptance(
        dataset_id=DATASET_ID,
        suggestion_id="lease-isolation",
        expected_revision=1,
        expected_source_fingerprint=_fingerprint(db, SOURCE_ID),
        expected_target_fingerprint=_fingerprint(db, TARGET_ID),
        idempotency_key="old-accept-key",
        now=NOW + timedelta(minutes=2),
    )
    assert old_replay.disposition == "terminal_replay"
    assert old_replay.envelope == released.envelope
    assert old_replay.suggestion is None
    assert second.suggestion.acceptance_lease_token not in json.dumps(old_replay.envelope)

    with pytest.raises(RuntimeError, match="notes_graph_suggestion_conflict"):
        db.note_graph_suggestion_store.reclaim_expired_acceptance(
            dataset_id=DATASET_ID,
            suggestion_id="lease-isolation",
            decision_receipt_id=first.suggestion.decision_receipt_id,
            expected_state="accepting",
            expected_revision=second.suggestion.revision,
            expected_lease_token=second.suggestion.acceptance_lease_token,
            now=NOW + timedelta(minutes=8),
        )
    with pytest.raises(RuntimeError, match="notes_graph_receipt_conflict"):
        db.note_graph_suggestion_store.release_acceptance(
            dataset_id=DATASET_ID,
            suggestion_id="lease-isolation",
            decision_receipt_id=first.suggestion.decision_receipt_id,
            expected_state="accepting",
            expected_revision=second.suggestion.revision,
            expected_lease_token=second.suggestion.acceptance_lease_token,
            now=NOW + timedelta(minutes=3),
        )


def test_missing_in_progress_acceptance_resource_is_operation_specific(db) -> None:
    _stage_and_activate(db, key="missing-decision-run", suggestion_id="missing-decision")
    claim = db.note_graph_suggestion_store.claim_acceptance(
        dataset_id=DATASET_ID,
        suggestion_id="missing-decision",
        expected_revision=1,
        expected_source_fingerprint=_fingerprint(db, SOURCE_ID),
        expected_target_fingerprint=_fingerprint(db, TARGET_ID),
        idempotency_key="missing-decision-key",
        now=NOW,
    )
    assert claim.disposition == "completed"
    db.execute_query("DELETE FROM note_graph_suggestions WHERE id='missing-decision'")
    with pytest.raises(RuntimeError, match="notes_graph_suggestion_accept_resource_missing"):
        db.note_graph_suggestion_store.claim_acceptance(
            dataset_id=DATASET_ID,
            suggestion_id="missing-decision",
            expected_revision=1,
            expected_source_fingerprint=_fingerprint(db, SOURCE_ID),
            expected_target_fingerprint=_fingerprint(db, TARGET_ID),
            idempotency_key="missing-decision-key",
            now=NOW,
        )


def test_terminal_acceptance_receipt_replays_after_resource_cleanup(db) -> None:
    _stage_and_activate(db, key="decision-cleanup-run", suggestion_id="decision-cleanup")
    claim = db.note_graph_suggestion_store.claim_acceptance(
        dataset_id=DATASET_ID,
        suggestion_id="decision-cleanup",
        expected_revision=1,
        expected_source_fingerprint=_fingerprint(db, SOURCE_ID),
        expected_target_fingerprint=_fingerprint(db, TARGET_ID),
        idempotency_key="decision-cleanup-key",
        now=NOW,
    )
    released = db.note_graph_suggestion_store.release_acceptance(
        dataset_id=DATASET_ID,
        suggestion_id="decision-cleanup",
        decision_receipt_id=claim.suggestion.decision_receipt_id,
        expected_state="accepting",
        expected_revision=claim.suggestion.revision,
        expected_lease_token=claim.suggestion.acceptance_lease_token,
        now=NOW,
    )
    db.execute_query("DELETE FROM note_graph_suggestions WHERE id='decision-cleanup'")
    replay = db.note_graph_suggestion_store.claim_acceptance(
        dataset_id=DATASET_ID,
        suggestion_id="decision-cleanup",
        expected_revision=1,
        expected_source_fingerprint=_fingerprint(db, SOURCE_ID),
        expected_target_fingerprint=_fingerprint(db, TARGET_ID),
        idempotency_key="decision-cleanup-key",
        now=NOW + timedelta(days=31),
    )
    assert replay.disposition == "terminal_replay"
    assert replay.envelope == released.envelope
    assert replay.suggestion is None
    with pytest.raises(RuntimeError, match="notes_graph_suggestion_idempotency_mismatch"):
        db.note_graph_suggestion_store.claim_acceptance(
            dataset_id=DATASET_ID,
            suggestion_id="decision-cleanup",
            expected_revision=2,
            expected_source_fingerprint=_fingerprint(db, SOURCE_ID),
            expected_target_fingerprint=_fingerprint(db, TARGET_ID),
            idempotency_key="decision-cleanup-key",
            now=NOW + timedelta(days=31),
        )


def test_pagination_is_stable_bounded_owner_scoped_and_integrity_checked(db) -> None:
    _stage_and_activate(db, key="page-a", suggestion_id="page-a")
    with db.transaction() as conn:
        conn.execute(
            "UPDATE note_graph_suggestion_runs SET state='failed',revision=revision+1 "
            "WHERE owner_user_id=? AND dataset_id=? AND state='succeeded'",
            (db.client_id, DATASET_ID),
        )
        conn.execute(
            "UPDATE note_graph_suggestions SET state='stale',revision=revision+1 "
            "WHERE owner_user_id=? AND dataset_id=? AND id='page-a'",
            (db.client_id, DATASET_ID),
        )
    _stage_and_activate(db, key="page-b", suggestion_id="page-b", target_id=OTHER_ID)
    # Re-expose the first row under a succeeded run to create a deterministic two-row page.
    with db.transaction() as conn:
        conn.execute(
            "UPDATE note_graph_suggestion_runs SET state='succeeded' WHERE id=(SELECT run_id FROM note_graph_suggestions WHERE id='page-a')"
        )
        conn.execute(
            "UPDATE note_graph_suggestions SET state='pending',revision=revision+1,updated_at=? WHERE id='page-a'",
            ((NOW - timedelta(minutes=1)).isoformat(),),
        )

    page_one = db.note_graph_suggestion_store.list_suggestions(
        dataset_id=DATASET_ID,
        source_note_id=SOURCE_ID,
        source_fingerprint=_fingerprint(db, SOURCE_ID),
        states=("pending",),
        limit=1,
        cursor=None,
        cursor_secret=b"test-cursor-secret",
    )
    page_two = db.note_graph_suggestion_store.list_suggestions(
        dataset_id=DATASET_ID,
        source_note_id=SOURCE_ID,
        source_fingerprint=_fingerprint(db, SOURCE_ID),
        states=("pending",),
        limit=1,
        cursor=page_one.next_cursor,
        cursor_secret=b"test-cursor-secret",
    )
    assert [item.id for item in page_one.items + page_two.items] == ["page-b", "page-a"]
    assert page_two.next_cursor is None
    with pytest.raises(ValueError, match="notes_graph_cursor_invalid"):
        db.note_graph_suggestion_store.list_suggestions(
            dataset_id=DATASET_ID,
            source_note_id=SOURCE_ID,
            source_fingerprint=_fingerprint(db, SOURCE_ID),
            states=("pending",),
            limit=1,
            cursor=f"{page_one.next_cursor}x",
            cursor_secret=b"test-cursor-secret",
        )
    with pytest.raises(ValueError, match="notes_graph_page_limit_invalid"):
        db.note_graph_suggestion_store.list_suggestions(
            dataset_id=DATASET_ID,
            source_note_id=SOURCE_ID,
            source_fingerprint=_fingerprint(db, SOURCE_ID),
            states=("pending",),
            limit=101,
            cursor=None,
            cursor_secret=b"test-cursor-secret",
        )


def test_retention_uses_exact_horizons_and_preserves_current_review_state(db) -> None:
    _stage_and_activate(db, key="retention-run", suggestion_id="retention-pending")
    current_fingerprint = _fingerprint(db, SOURCE_ID)
    expired_30 = NOW - timedelta(days=30, seconds=1)
    expired_90 = NOW - timedelta(days=90, seconds=1)
    with db.transaction() as conn:
        conn.execute(
            "UPDATE note_graph_suggestions SET updated_at=?,expires_at=? WHERE id='retention-pending'",
            (expired_90.isoformat(), expired_90.isoformat()),
        )
        conn.execute(
            "INSERT INTO note_graph_suggestion_rejection_sets(owner_user_id,dataset_id,source_note_id,source_fingerprint,revision,rejection_count,updated_at) "
            "VALUES (?,?,?,?,1,0,?)",
            (db.client_id, DATASET_ID, SOURCE_ID, current_fingerprint, expired_90.isoformat()),
        )
        run_id = conn.execute(
            "SELECT run_id FROM note_graph_suggestions WHERE id='retention-pending'"
        ).fetchone()["run_id"]
        conn.execute(
            "INSERT INTO note_graph_suggestions(id,run_id,owner_user_id,dataset_id,kind,source_note_id,source_fingerprint,target_note_id,target_fingerprint,state,revision,decision_reason,created_at,updated_at,expires_at) "
            "VALUES ('stale-detail',?,?,?,?,?,?,?,?, 'stale',1,'obsolete',?,?,?)",
            (
                run_id,
                db.client_id,
                DATASET_ID,
                "related_note",
                SOURCE_ID,
                current_fingerprint,
                OTHER_ID,
                _fingerprint(db, OTHER_ID),
                expired_30.isoformat(),
                expired_30.isoformat(),
                expired_30.isoformat(),
            ),
        )
        conn.execute(
            "INSERT INTO note_graph_suggestions(id,run_id,owner_user_id,dataset_id,kind,source_note_id,source_fingerprint,target_note_id,target_fingerprint,state,revision,decision_reason,created_at,updated_at,expires_at) "
            "VALUES ('accepted-audit',?,?,?,?,?,?,?,?, 'accepted',1,'accepted',?,?,?)",
            (
                run_id,
                db.client_id,
                DATASET_ID,
                "related_note",
                SOURCE_ID,
                current_fingerprint,
                TARGET_ID,
                _fingerprint(db, TARGET_ID),
                expired_90.isoformat(),
                expired_90.isoformat(),
                expired_90.isoformat(),
            ),
        )
        conn.execute(
            "INSERT INTO note_graph_suggestions(id,run_id,owner_user_id,dataset_id,kind,source_note_id,source_fingerprint,target_note_id,target_fingerprint,state,revision,decision_reason,created_at,updated_at,expires_at) "
            "VALUES ('current-rejection',?,?,?,?,?,?,?,?, 'rejected',1,'user_rejected',?,?,?)",
            (
                run_id,
                db.client_id,
                DATASET_ID,
                "related_note",
                SOURCE_ID,
                current_fingerprint,
                OTHER_ID,
                _fingerprint(db, OTHER_ID),
                expired_30.isoformat(),
                expired_30.isoformat(),
                expired_30.isoformat(),
            ),
        )
        conn.execute(
            "INSERT INTO note_graph_suggestion_operation_receipts("
            "id,operation_kind,owner_user_id,dataset_id,source_note_id,resource_identity,"
            "idempotency_key_digest,request_fingerprint,state,http_status,replay_envelope,"
            "created_at,completed_at,expires_at) VALUES ("
            "'expired-receipt','suggestion_reject',?,?,?,?,?,?, 'completed',200,'{}',?,?,?)",
            (
                db.client_id,
                DATASET_ID,
                SOURCE_ID,
                "expired-resource",
                "sha256:key",
                "sha256:request",
                expired_90.isoformat(),
                expired_90.isoformat(),
                expired_90.isoformat(),
            ),
        )
        conn.execute(
            "INSERT INTO note_graph_suggestion_runs("
            "id,owner_user_id,dataset_id,source_note_id,source_fingerprint,"
            "provider,model,capability_revision,prompt_contract_version,"
            "state,revision,created_at,completed_at,expires_at"
            ") VALUES ('expired-failed-run',?,?,?,?,?,?,?,?,'failed',1,?,?,?)",
            (
                db.client_id,
                DATASET_ID,
                SOURCE_ID,
                current_fingerprint,
                "provider",
                "model",
                "capability-v1",
                "prompt-v1",
                expired_30.isoformat(),
                expired_30.isoformat(),
                expired_30.isoformat(),
            ),
        )

    counts = db.note_graph_suggestion_store.cleanup_retention(
        dataset_id=DATASET_ID,
        now=NOW,
        limit=100,
    )

    assert counts == {"suggestions": 2, "receipts": 1, "runs": 1, "rejection_sets": 0}
    assert db.execute_query(
        "SELECT state FROM note_graph_suggestions WHERE id='retention-pending'"
    ).fetchone()["state"] == "pending"
    rejection_set = db.execute_query(
        "SELECT revision,rejection_count FROM note_graph_suggestion_rejection_sets "
        "WHERE owner_user_id=? AND dataset_id=? AND source_note_id=? AND source_fingerprint=?",
        (db.client_id, DATASET_ID, SOURCE_ID, current_fingerprint),
    ).fetchone()
    assert dict(rejection_set) == {"revision": 1, "rejection_count": 0}
    assert db.execute_query(
        "SELECT state FROM note_graph_suggestions WHERE id='current-rejection'"
    ).fetchone()["state"] == "rejected"

    receipt = db.execute_query(
        "SELECT replay_envelope FROM note_graph_suggestion_operation_receipts LIMIT 1"
    ).fetchone()
    assert receipt is not None
    assert "source body" not in json.dumps(dict(receipt))


def test_cleanup_rejects_above_maintenance_cap_and_expires_obsolete_target_rejection(db) -> None:
    with pytest.raises(ValueError, match="notes_graph_cleanup_limit_invalid"):
        db.note_graph_suggestion_store.cleanup_retention(
            dataset_id=DATASET_ID,
            now=NOW,
            limit=101,
        )

    _stage_and_activate(db, key="target-retention-run", suggestion_id="target-retention")
    db.note_graph_suggestion_store.reject_suggestion(
        dataset_id=DATASET_ID,
        suggestion_id="target-retention",
        expected_revision=1,
        expected_source_fingerprint=_fingerprint(db, SOURCE_ID),
        expected_target_fingerprint=_fingerprint(db, TARGET_ID),
        idempotency_key="target-retention-reject",
        now=NOW,
    )
    with db.transaction() as conn:
        conn.execute("UPDATE notes SET title='Obsolete target' WHERE id=?", (TARGET_ID,))
    counts = db.note_graph_suggestion_store.cleanup_retention(
        dataset_id=DATASET_ID,
        now=NOW + timedelta(days=31),
        limit=100,
    )
    assert counts["suggestions"] == 1
