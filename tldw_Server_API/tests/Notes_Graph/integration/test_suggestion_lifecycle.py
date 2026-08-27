from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Notes_Graph.suggestion_content import content_fingerprint

pytestmark = pytest.mark.integration

DATASET_ID = "dataset-1"
NOW = datetime(2026, 8, 27, 16, 0, tzinfo=timezone.utc)


def _new_db(tmp_path, *, backend=None) -> CharactersRAGDB:
    db = CharactersRAGDB(
        str(tmp_path / "suggestion-lifecycle.db") if backend is None else ":memory:",
        client_id="owner-1",
        backend=backend,
    )
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO note_task_scope_authority(owner_user_id,dataset_id) VALUES (?,?)",
            (db.client_id, DATASET_ID),
        )
    return db


def _scope(db: CharactersRAGDB, conn) -> None:
    if db.note_graph_suggestion_store.is_postgres:
        conn.execute("SELECT set_config('app.current_dataset_id', ?, true)", (DATASET_ID,))


def _note_fingerprint(db: CharactersRAGDB, note_id: str) -> str:
    note = db.get_note_by_id(note_id, include_deleted=True)
    assert note is not None
    return content_fingerprint(note["title"], note["content"])


def _seed_run(
    db: CharactersRAGDB,
    *,
    source_id: str,
    state: str,
    run_id: str,
    job_id: str | None = None,
) -> None:
    with db.transaction() as conn:
        _scope(db, conn)
        conn.execute(
            "INSERT INTO note_graph_suggestion_runs("
            "id,owner_user_id,dataset_id,source_note_id,source_fingerprint,provider,model,capability_revision,prompt_contract_version,job_id,expected_completion_token,state,revision,created_at,expires_at"
            ") VALUES (?,?,?,?,?,'openai','model-a','cap-v1','prompt-v1',?,? ,?,1,?,?)",
            (
                run_id,
                db.client_id,
                DATASET_ID,
                source_id,
                _note_fingerprint(db, source_id),
                job_id,
                f"completion-{run_id}" if job_id else None,
                state,
                NOW.isoformat(),
                (NOW + timedelta(days=90)).isoformat(),
            ),
        )


def _seed_related_suggestion(
    db: CharactersRAGDB,
    *,
    run_id: str,
    suggestion_id: str,
    source_id: str,
    target_id: str,
    state: str,
) -> None:
    with db.transaction() as conn:
        _scope(db, conn)
        conn.execute(
            "INSERT INTO note_graph_suggestions("
            "id,run_id,owner_user_id,dataset_id,kind,source_note_id,source_fingerprint,target_note_id,target_fingerprint,match_strength,rationale,state,revision,created_at,updated_at"
            ") VALUES (?,?,?,?, 'related_note',?,?,?,?, 'strong','safe rationale',?,1,?,?)",
            (
                suggestion_id,
                run_id,
                db.client_id,
                DATASET_ID,
                source_id,
                _note_fingerprint(db, source_id),
                target_id,
                _note_fingerprint(db, target_id),
                state,
                NOW.isoformat(),
                NOW.isoformat(),
            ),
        )


def _exercise_source_lifecycle(db: CharactersRAGDB) -> None:
    cases = (
        ("admitting", None, "stale"),
        ("admitting", "job-admitting", "cancelling"),
        ("queued", "job-queued", "stale"),
        ("running", "job-running", "cancelling"),
        ("publishing", "job-publishing", "stale"),
    )
    for index, (initial, job_id, expected) in enumerate(cases):
        source_id = f"00000000-0000-4000-8000-{index + 1:012d}"
        target_id = f"00000000-0000-4000-8000-{index + 101:012d}"
        run_id = f"run-source-{initial}-{index}"
        db.add_note(f"Source {initial}", "body", note_id=source_id)
        db.add_note(f"Target {initial}", "body", note_id=target_id)
        _seed_run(db, source_id=source_id, state=initial, run_id=run_id, job_id=job_id)
        if initial == "publishing":
            _seed_related_suggestion(
                db,
                run_id=run_id,
                suggestion_id="source-publishing-staged",
                source_id=source_id,
                target_id=target_id,
                state="staged",
            )

        assert db.update_note(source_id, {"content": "changed"}, expected_version=1)
        with db.transaction() as conn:
            _scope(db, conn)
            run = conn.execute(
                "SELECT state,revision,error_code FROM note_graph_suggestion_runs "
                "WHERE owner_user_id=? AND dataset_id=? AND id=?",
                (db.client_id, DATASET_ID, run_id),
            ).fetchone()
            receipts = conn.execute(
                "SELECT id,operation_kind,state,resource_identity,replay_envelope "
                "FROM note_graph_suggestion_operation_receipts WHERE owner_user_id=? AND dataset_id=? AND resource_identity=?",
                (db.client_id, DATASET_ID, run_id),
            ).fetchall()
        assert run["state"] == expected
        assert int(run["revision"]) == 2
        if initial in {"queued", "running"} or (initial == "admitting" and job_id):
            assert len(receipts) == 1
            assert receipts[0]["operation_kind"] == "run_cancel"
            assert receipts[0]["state"] == "in_progress"
            assert receipts[0]["id"] == db.note_graph_suggestion_store.cancellation_operation_id(
                dataset_id=DATASET_ID,
                run_id=run_id,
                run_revision=1,
            )
            assert "changed" not in str(receipts[0]["replay_envelope"])
        else:
            assert receipts == []
        if initial == "running":
            assert run["error_code"] == "notes_graph_source_changed"
        if initial == "publishing":
            assert db.execute_query(
                "SELECT COUNT(*) AS count FROM note_graph_suggestions WHERE run_id=?",
                (run_id,),
            ).fetchone()["count"] == 0


def _exercise_target_and_restore_lifecycle(db: CharactersRAGDB) -> None:
    source = "00000000-0000-4000-8000-000000000500"
    target = "00000000-0000-4000-8000-000000000501"
    db.add_note("Source", "body", note_id=source)
    db.add_note("Target", "body", note_id=target)

    _seed_run(db, source_id=source, state="succeeded", run_id="run-pending")
    _seed_related_suggestion(
        db,
        run_id="run-pending",
        suggestion_id="pending-target",
        source_id=source,
        target_id=target,
        state="pending",
    )
    assert db.update_note(target, {"title": "Target changed"}, expected_version=1)
    row = db.execute_query(
        "SELECT state,revision,decision_reason FROM note_graph_suggestions WHERE id='pending-target'"
    ).fetchone()
    assert (row["state"], int(row["revision"]), row["decision_reason"]) == (
        "stale",
        2,
        "target_changed",
    )

    assert db.soft_delete_note(target, expected_version=2)
    assert db.restore_note(target, expected_version=3)
    restored = db.execute_query(
        "SELECT state,revision FROM note_graph_suggestions WHERE id='pending-target'"
    ).fetchone()
    assert (restored["state"], int(restored["revision"])) == ("stale", 2)


def _exercise_source_and_target_suggestion_state_matrix(db: CharactersRAGDB) -> None:
    for role, states in (("source", ("pending", "rejected", "accepting")), ("target", ("rejected",))):
        for index, state in enumerate(states):
            source = f"10000000-0000-4000-8000-{index + (100 if role == 'source' else 200):012d}"
            target = f"10000000-0000-4000-8000-{index + (300 if role == 'source' else 400):012d}"
            run_id = f"run-{role}-{state}"
            suggestion_id = f"suggestion-{role}-{state}"
            db.add_note(f"{role} {state} source", "body", note_id=source)
            db.add_note(f"{role} {state} target", "body", note_id=target)
            _seed_run(db, source_id=source, state="succeeded", run_id=run_id)
            _seed_related_suggestion(
                db,
                run_id=run_id,
                suggestion_id=suggestion_id,
                source_id=source,
                target_id=target,
                state=state,
            )
            changed_note = source if role == "source" else target
            assert db.update_note(changed_note, {"content": "changed"}, expected_version=1)
            row = db.execute_query(
                "SELECT state,revision,decision_reason FROM note_graph_suggestions WHERE id=?",
                (suggestion_id,),
            ).fetchone()
            assert (row["state"], int(row["revision"]), row["decision_reason"]) == (
                "stale",
                2,
                f"{role}_changed",
            )


def _exercise_sync_and_soft_delete_entry_points(db: CharactersRAGDB) -> None:
    sync_source = "20000000-0000-4000-8000-000000000001"
    db.add_note("Sync source", "body", note_id=sync_source)
    _seed_run(db, source_id=sync_source, state="queued", run_id="run-sync", job_id="job-sync")
    assert db.upsert_note_from_sync(
        note_id=sync_source,
        title="Sync source changed",
        content="changed",
        conversation_id=None,
        message_id=None,
        sync_client_id=db.client_id,
        object_revision=2,
        object_hash="ignored",
        expected_product_version=1,
        projection_timestamp=(NOW + timedelta(minutes=1)).isoformat(),
    )
    assert db.execute_query(
        "SELECT state FROM note_graph_suggestion_runs WHERE id='run-sync'"
    ).fetchone()["state"] == "stale"

    for index, operation in enumerate(("soft_delete_note", "delete_note")):
        source = f"20000000-0000-4000-8000-{index + 10:012d}"
        db.add_note(f"Delete source {index}", "body", note_id=source)
        _seed_run(
            db,
            source_id=source,
            state="queued",
            run_id=f"run-delete-entry-{index}",
            job_id=f"job-delete-entry-{index}",
        )
        if operation == "soft_delete_note":
            assert db.soft_delete_note(source, expected_version=1)
        else:
            assert db.delete_note(source, expected_version=1, hard_delete=False)
        assert db.execute_query(
            "SELECT state FROM note_graph_suggestion_runs WHERE id=?",
            (f"run-delete-entry-{index}",),
        ).fetchone()["state"] == "stale"


def _exercise_accepting_target_fence(db: CharactersRAGDB) -> None:
    source = "00000000-0000-4000-8000-000000000600"
    target = "00000000-0000-4000-8000-000000000601"
    db.add_note("Source accepting", "body", note_id=source)
    db.add_note("Target accepting", "body", note_id=target)
    _seed_run(db, source_id=source, state="succeeded", run_id="run-accepting")
    _seed_related_suggestion(
        db,
        run_id="run-accepting",
        suggestion_id="accepting-target",
        source_id=source,
        target_id=target,
        state="accepting",
    )
    with db.transaction() as conn:
        _scope(db, conn)
        conn.execute(
            "UPDATE note_graph_suggestions SET acceptance_lease_token='old-fence',acceptance_lease_expires_at=? "
            "WHERE id='accepting-target'",
            ((NOW + timedelta(minutes=5)).isoformat(),),
        )

    assert db.soft_delete_note(target, expected_version=1)
    row = db.execute_query(
        "SELECT state,revision,acceptance_lease_token,acceptance_lease_expires_at "
        "FROM note_graph_suggestions WHERE id='accepting-target'"
    ).fetchone()
    assert (row["state"], int(row["revision"])) == ("stale", 2)
    assert row["acceptance_lease_token"] is None
    assert row["acceptance_lease_expires_at"] is None


def _exercise_tag_membership_does_not_invalidate(db: CharactersRAGDB) -> None:
    source = "00000000-0000-4000-8000-000000000700"
    target = "00000000-0000-4000-8000-000000000701"
    db.add_note("Source tag", "body", note_id=source)
    db.add_note("Target tag", "body", note_id=target)
    _seed_run(db, source_id=source, state="succeeded", run_id="run-tag")
    _seed_related_suggestion(
        db,
        run_id="run-tag",
        suggestion_id="tag-sibling",
        source_id=source,
        target_id=target,
        state="pending",
    )
    before = _note_fingerprint(db, source)
    keyword_id = db.add_keyword("new tag")
    assert keyword_id is not None
    assert db.link_note_to_keyword(source, keyword_id)
    assert _note_fingerprint(db, source) == before
    sibling = db.execute_query(
        "SELECT state,revision FROM note_graph_suggestions WHERE id='tag-sibling'"
    ).fetchone()
    assert (sibling["state"], int(sibling["revision"])) == ("pending", 1)


def _exercise_hard_delete_cascades(db: CharactersRAGDB) -> None:
    source = "00000000-0000-4000-8000-000000000800"
    target = "00000000-0000-4000-8000-000000000801"
    db.add_note("Source delete", "body", note_id=source)
    db.add_note("Target delete", "body", note_id=target)
    _seed_run(db, source_id=source, state="succeeded", run_id="run-delete")
    _seed_related_suggestion(
        db,
        run_id="run-delete",
        suggestion_id="delete-suggestion",
        source_id=source,
        target_id=target,
        state="pending",
    )
    with db.transaction() as conn:
        _scope(db, conn)
        conn.execute(
            "INSERT INTO note_graph_suggestion_operation_receipts("
            "id,operation_kind,owner_user_id,dataset_id,source_note_id,resource_identity,"
            "idempotency_key_digest,request_fingerprint,state,http_status,replay_envelope,"
            "created_at,completed_at,expires_at) "
            "VALUES ('delete-receipt','run_admit',?,?,?,?,?,?, 'completed',200,?, ?,?,?)",
            (
                db.client_id,
                DATASET_ID,
                source,
                "run-delete",
                "sha256:key",
                "sha256:request",
                '{"run_id":"run-delete","state":"queued"}',
                NOW.isoformat(),
                NOW.isoformat(),
                (NOW + timedelta(days=90)).isoformat(),
            ),
        )
        conn.execute(
            "INSERT INTO note_graph_suggestion_evidence(suggestion_id,owner_user_id,dataset_id,side,ordinal,note_id,field,content_fingerprint,start_offset,end_offset) "
            "VALUES ('delete-suggestion',?,?, 'source',0,?,'content',?,0,4)",
            (db.client_id, DATASET_ID, source, _note_fingerprint(db, source)),
        )
    assert db.delete_note(source, hard_delete=True)
    assert db.execute_query(
        "SELECT COUNT(*) AS count FROM note_graph_suggestion_runs WHERE id='run-delete'"
    ).fetchone()["count"] == 0
    assert db.execute_query(
        "SELECT COUNT(*) AS count FROM note_graph_suggestions WHERE id='delete-suggestion'"
    ).fetchone()["count"] == 0
    assert db.execute_query(
        "SELECT COUNT(*) AS count FROM note_graph_suggestion_evidence "
        "WHERE suggestion_id='delete-suggestion'"
    ).fetchone()["count"] == 0
    assert db.execute_query(
        "SELECT COUNT(*) AS count FROM note_graph_suggestion_operation_receipts "
        "WHERE id='delete-receipt'"
    ).fetchone()["count"] == 0


def _exercise_lifecycle_contract(db: CharactersRAGDB) -> None:
    _exercise_source_lifecycle(db)
    _exercise_target_and_restore_lifecycle(db)
    _exercise_source_and_target_suggestion_state_matrix(db)
    _exercise_sync_and_soft_delete_entry_points(db)
    _exercise_accepting_target_fence(db)
    _exercise_tag_membership_does_not_invalidate(db)
    _exercise_hard_delete_cascades(db)


def _exercise_invalidation_rollback(db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch) -> None:
    source = "30000000-0000-4000-8000-000000000001"
    target = "30000000-0000-4000-8000-000000000002"
    db.add_note("Rollback source", "original body", note_id=source)
    db.add_note("Rollback target", "target body", note_id=target)
    _seed_run(db, source_id=source, state="succeeded", run_id="run-rollback")
    _seed_related_suggestion(
        db,
        run_id="run-rollback",
        suggestion_id="suggestion-rollback",
        source_id=source,
        target_id=target,
        state="pending",
    )
    original = db.note_graph_suggestion_store.invalidate_for_note_change

    def invalidate_then_fail(*, note_id, conn):
        original(note_id=note_id, conn=conn)
        raise RuntimeError("forced invalidation failure")

    monkeypatch.setattr(
        db.note_graph_suggestion_store,
        "invalidate_for_note_change",
        invalidate_then_fail,
    )
    with pytest.raises(RuntimeError, match="forced invalidation failure"):
        db.update_note(source, {"title": "Changed", "content": "changed"}, expected_version=1)
    with pytest.raises(RuntimeError, match="forced invalidation failure"):
        db.soft_delete_note(source, expected_version=1)

    note = db.get_note_by_id(source, include_deleted=True)
    assert note is not None
    assert (note["title"], note["content"], int(note["version"]), bool(note["deleted"])) == (
        "Rollback source",
        "original body",
        1,
        False,
    )
    run = db.execute_query(
        "SELECT state,revision FROM note_graph_suggestion_runs WHERE id='run-rollback'"
    ).fetchone()
    suggestion = db.execute_query(
        "SELECT state,revision FROM note_graph_suggestions WHERE id='suggestion-rollback'"
    ).fetchone()
    assert (run["state"], int(run["revision"])) == ("succeeded", 1)
    assert (suggestion["state"], int(suggestion["revision"])) == ("pending", 1)


def test_sqlite_note_lifecycle_invalidates_in_same_product_transactions(tmp_path) -> None:
    db = _new_db(tmp_path)
    try:
        _exercise_lifecycle_contract(db)
    finally:
        db.close_all_connections()


def test_postgres_note_lifecycle_invalidates_in_same_product_transactions(
    tmp_path,
    pg_database_config,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = _new_db(tmp_path, backend=backend)
    try:
        _exercise_lifecycle_contract(db)
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()


def test_sqlite_note_and_suggestion_invalidation_roll_back_together(tmp_path, monkeypatch) -> None:
    db = _new_db(tmp_path)
    try:
        _exercise_invalidation_rollback(db, monkeypatch)
    finally:
        db.close_all_connections()


def test_postgres_note_and_suggestion_invalidation_roll_back_together(
    tmp_path,
    pg_database_config,
    monkeypatch,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = _new_db(tmp_path, backend=backend)
    try:
        _exercise_invalidation_rollback(db, monkeypatch)
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()


def test_postgres_active_run_identity_matches_sqlite_full_tuple(tmp_path, pg_database_config) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = _new_db(tmp_path, backend=backend)
    source = "00000000-0000-4000-8000-000000000900"
    try:
        db.add_note("Source identity", "body", note_id=source)
        common = {
            "dataset_id": DATASET_ID,
            "source_note_id": source,
            "source_fingerprint": _note_fingerprint(db, source),
            "provider": "openai",
            "capability_revision": "cap-v1",
            "prompt_contract_version": "prompt-v1",
            "now": NOW,
        }
        first = db.note_graph_suggestion_store.admit_run(
            **common, model="model-a", idempotency_key="postgres-key-a"
        )
        second = db.note_graph_suggestion_store.admit_run(
            **common, model="model-b", idempotency_key="postgres-key-b"
        )
        assert first.run.id != second.run.id
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()


def test_postgres_durable_publication_and_decision_state_machine(tmp_path, pg_database_config) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = _new_db(tmp_path, backend=backend)
    source = "00000000-0000-4000-8000-000000000910"
    target = "00000000-0000-4000-8000-000000000911"
    try:
        db.add_note("Postgres source", "source body", note_id=source)
        db.add_note("Postgres target", "target body", note_id=target)
        source_fingerprint = _note_fingerprint(db, source)
        target_fingerprint = _note_fingerprint(db, target)
        store = db.note_graph_suggestion_store
        admission = store.admit_run(
            dataset_id=DATASET_ID,
            source_note_id=source,
            source_fingerprint=source_fingerprint,
            provider="openai",
            model="model-a",
            capability_revision="cap-v1",
            prompt_contract_version="prompt-v1",
            idempotency_key="postgres-state-machine",
            now=NOW,
        )
        assert store.admit_run(
            dataset_id=DATASET_ID,
            source_note_id=source,
            source_fingerprint=source_fingerprint,
            provider="openai",
            model="model-a",
            capability_revision="cap-v1",
            prompt_contract_version="prompt-v1",
            idempotency_key="postgres-state-machine",
            now=NOW,
        ).disposition == "in_progress"
        queued = store.bind_admitted_run(
            dataset_id=DATASET_ID,
            run_id=admission.run.id,
            expected_state="admitting",
            expected_revision=1,
            job_id="postgres-job",
            completion_token="postgres-completion",
            replay_envelope={"run_id": admission.run.id, "state": "queued"},
            now=NOW,
        )
        running = store.start_run(
            dataset_id=DATASET_ID,
            run_id=queued.id,
            expected_state="queued",
            expected_revision=queued.revision,
            now=NOW,
        )
        publishing = store.stage_suggestions(
            dataset_id=DATASET_ID,
            run_id=running.id,
            expected_state="running",
            expected_revision=running.revision,
            result_digest=f"sha256:{'9' * 64}",
            candidates=(
                {
                    "id": "postgres-suggestion",
                    "kind": "related_note",
                    "target_note_id": target,
                    "target_fingerprint": target_fingerprint,
                    "match_strength": "strong",
                    "rationale": "Bounded rationale",
                    "evidence": (
                        {
                            "side": "source",
                            "ordinal": 0,
                            "note_id": source,
                            "field": "content",
                            "content_fingerprint": source_fingerprint,
                            "start_offset": 0,
                            "end_offset": 6,
                        },
                    ),
                },
            ),
            invalid_item_count=0,
            now=NOW,
        )
        succeeded = store.activate_staged_run(
            dataset_id=DATASET_ID,
            run_id=publishing.id,
            expected_state="publishing",
            expected_revision=publishing.revision,
            observed_job_id=publishing.job_id,
            observed_completion_token=publishing.expected_completion_token,
            observed_result_digest=publishing.result_digest,
            now=NOW,
        )
        assert succeeded.state.value == "succeeded"
        claim = store.claim_acceptance(
            dataset_id=DATASET_ID,
            suggestion_id="postgres-suggestion",
            expected_revision=1,
            expected_source_fingerprint=source_fingerprint,
            expected_target_fingerprint=target_fingerprint,
            idempotency_key="postgres-accept",
            now=NOW,
        )
        released = store.release_acceptance(
            dataset_id=DATASET_ID,
            suggestion_id="postgres-suggestion",
            decision_receipt_id=claim.suggestion.decision_receipt_id,
            expected_state="accepting",
            expected_revision=claim.suggestion.revision,
            expected_lease_token=claim.suggestion.acceptance_lease_token,
            now=NOW,
        )
        pending = released.suggestion
        assert pending is not None
        rejected = store.reject_suggestion(
            dataset_id=DATASET_ID,
            suggestion_id="postgres-suggestion",
            expected_revision=pending.revision,
            expected_source_fingerprint=source_fingerprint,
            expected_target_fingerprint=target_fingerprint,
            idempotency_key="postgres-reject",
            now=NOW,
        )
        reset = store.reset_rejections(
            dataset_id=DATASET_ID,
            source_note_id=source,
            source_fingerprint=source_fingerprint,
            expected_revision=rejected.rejection_set.revision,
            idempotency_key="postgres-reset",
            now=NOW,
        )
        assert reset.rejection_set.revision == 2
        assert reset.rejection_set.rejection_count == 0
        assert store.cleanup_retention(
            dataset_id=DATASET_ID,
            now=NOW + timedelta(days=31),
            limit=100,
        ) == {"suggestions": 1, "receipts": 0, "runs": 0, "rejection_sets": 0}
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()


def _admit_for(db: CharactersRAGDB, source: str, key: str, *, model: str = "model-a"):
    return db.note_graph_suggestion_store.admit_run(
        dataset_id=DATASET_ID,
        source_note_id=source,
        source_fingerprint=_note_fingerprint(db, source),
        provider="openai",
        model=model,
        capability_revision="cap-v1",
        prompt_contract_version="prompt-v1",
        idempotency_key=key,
        now=NOW,
    )


def _running_for(db: CharactersRAGDB, source: str, key: str, *, model: str = "model-a"):
    store = db.note_graph_suggestion_store
    admission = _admit_for(db, source, key, model=model)
    queued = store.bind_admitted_run(
        dataset_id=DATASET_ID,
        run_id=admission.run.id,
        expected_state="admitting",
        expected_revision=admission.run.revision,
        job_id=f"job-{key}",
        completion_token=f"completion-{key}",
        replay_envelope={"run_id": admission.run.id, "state": "queued"},
        now=NOW,
    )
    return store.start_run(
        dataset_id=DATASET_ID,
        run_id=queued.id,
        expected_state="queued",
        expected_revision=queued.revision,
        now=NOW,
    )


def _publish_related(
    db: CharactersRAGDB,
    *,
    source: str,
    target: str,
    key: str,
    suggestion_id: str,
    model: str = "model-a",
):
    store = db.note_graph_suggestion_store
    running = _running_for(db, source, key, model=model)
    publishing = store.stage_suggestions(
        dataset_id=DATASET_ID,
        run_id=running.id,
        expected_state="running",
        expected_revision=running.revision,
        result_digest=f"sha256:{'7' * 64}",
        candidates=(
            {
                "id": suggestion_id,
                "kind": "related_note",
                "target_note_id": target,
                "target_fingerprint": _note_fingerprint(db, target),
                "match_strength": "strong",
                "rationale": "Bounded rationale",
                "evidence": (),
            },
        ),
        invalid_item_count=0,
        now=NOW,
    )
    return store.activate_staged_run(
        dataset_id=DATASET_ID,
        run_id=publishing.id,
        expected_state="publishing",
        expected_revision=publishing.revision,
        observed_job_id=publishing.job_id,
        observed_completion_token=publishing.expected_completion_token,
        observed_result_digest=publishing.result_digest,
        now=NOW,
    )


def _exercise_receipt_cleanup_and_acceptance_fences(db: CharactersRAGDB) -> None:
    store = db.note_graph_suggestion_store
    admission_source = "40000000-0000-4000-8000-000000000001"
    db.add_note("Admission receipt source", "body", note_id=admission_source)
    admission = _admit_for(db, admission_source, "admission-terminal")
    terminal_run = store.fail_admission(
        dataset_id=DATASET_ID,
        run_id=admission.run.id,
        expected_state="admitting",
        expected_revision=admission.run.revision,
        error_code="notes_graph_admission_failed",
        guidance_key="retry_generation",
        now=NOW,
    )
    assert terminal_run.state.value == "failed"
    assert store.cleanup_retention(
        dataset_id=DATASET_ID,
        now=NOW + timedelta(days=31),
        limit=100,
    )["runs"] == 1
    admission_replay = _admit_for(db, admission_source, "admission-terminal")
    assert admission_replay.disposition == "terminal_replay"
    assert admission_replay.run is None
    assert admission_replay.replay_envelope["error_code"] == "notes_graph_admission_failed"
    with pytest.raises(RuntimeError, match="notes_graph_suggestion_idempotency_mismatch"):
        _admit_for(db, admission_source, "admission-terminal", model="model-b")

    missing_source = "40000000-0000-4000-8000-000000000002"
    db.add_note("Missing admission source", "body", note_id=missing_source)
    missing_admission = _admit_for(db, missing_source, "admission-missing")
    db.execute_query("DELETE FROM note_graph_suggestion_runs WHERE id=?", (missing_admission.run.id,))
    with pytest.raises(RuntimeError, match="notes_graph_run_admit_resource_missing"):
        _admit_for(db, missing_source, "admission-missing")

    reject_source = "40000000-0000-4000-8000-000000000010"
    reject_target = "40000000-0000-4000-8000-000000000011"
    db.add_note("Reject source", "body", note_id=reject_source)
    db.add_note("Reject target", "body", note_id=reject_target)
    _publish_related(
        db,
        source=reject_source,
        target=reject_target,
        key="reject-publication",
        suggestion_id="reject-cleanup",
    )
    source_fp = _note_fingerprint(db, reject_source)
    target_fp = _note_fingerprint(db, reject_target)
    rejected = store.reject_suggestion(
        dataset_id=DATASET_ID,
        suggestion_id="reject-cleanup",
        expected_revision=1,
        expected_source_fingerprint=source_fp,
        expected_target_fingerprint=target_fp,
        idempotency_key="reject-terminal",
        now=NOW,
    )
    with db.transaction() as conn:
        _scope(db, conn)
        conn.execute("UPDATE notes SET content='obsolete target' WHERE id=?", (reject_target,))
    assert store.cleanup_retention(
        dataset_id=DATASET_ID,
        now=NOW + timedelta(days=31),
        limit=100,
    )["suggestions"] == 1
    reject_replay = store.reject_suggestion(
        dataset_id=DATASET_ID,
        suggestion_id="reject-cleanup",
        expected_revision=1,
        expected_source_fingerprint=source_fp,
        expected_target_fingerprint=target_fp,
        idempotency_key="reject-terminal",
        now=NOW + timedelta(days=31),
    )
    assert reject_replay.disposition == "terminal_replay"
    assert reject_replay.envelope == rejected.envelope
    assert reject_replay.suggestion is None
    with pytest.raises(RuntimeError, match="notes_graph_suggestion_idempotency_mismatch"):
        store.reject_suggestion(
            dataset_id=DATASET_ID,
            suggestion_id="reject-cleanup",
            expected_revision=2,
            expected_source_fingerprint=source_fp,
            expected_target_fingerprint=target_fp,
            idempotency_key="reject-terminal",
            now=NOW + timedelta(days=31),
        )

    decision_source = "40000000-0000-4000-8000-000000000020"
    decision_target = "40000000-0000-4000-8000-000000000021"
    db.add_note("Decision source", "body", note_id=decision_source)
    db.add_note("Decision target", "body", note_id=decision_target)
    _publish_related(
        db,
        source=decision_source,
        target=decision_target,
        key="decision-publication",
        suggestion_id="decision-cleanup",
    )
    decision_source_fp = _note_fingerprint(db, decision_source)
    decision_target_fp = _note_fingerprint(db, decision_target)
    first = store.claim_acceptance(
        dataset_id=DATASET_ID,
        suggestion_id="decision-cleanup",
        expected_revision=1,
        expected_source_fingerprint=decision_source_fp,
        expected_target_fingerprint=decision_target_fp,
        idempotency_key="decision-old-key",
        now=NOW,
    )
    released = store.release_acceptance(
        dataset_id=DATASET_ID,
        suggestion_id="decision-cleanup",
        decision_receipt_id=first.suggestion.decision_receipt_id,
        expected_state="accepting",
        expected_revision=first.suggestion.revision,
        expected_lease_token=first.suggestion.acceptance_lease_token,
        now=NOW + timedelta(minutes=1),
    )
    second = store.claim_acceptance(
        dataset_id=DATASET_ID,
        suggestion_id="decision-cleanup",
        expected_revision=3,
        expected_source_fingerprint=decision_source_fp,
        expected_target_fingerprint=decision_target_fp,
        idempotency_key="decision-new-key",
        now=NOW + timedelta(minutes=2),
    )
    old_replay = store.claim_acceptance(
        dataset_id=DATASET_ID,
        suggestion_id="decision-cleanup",
        expected_revision=1,
        expected_source_fingerprint=decision_source_fp,
        expected_target_fingerprint=decision_target_fp,
        idempotency_key="decision-old-key",
        now=NOW + timedelta(minutes=2),
    )
    assert old_replay.envelope == released.envelope
    assert old_replay.suggestion is None
    assert second.suggestion.acceptance_lease_token not in str(old_replay.envelope)
    with pytest.raises(RuntimeError, match="notes_graph_receipt_conflict"):
        store.release_acceptance(
            dataset_id=DATASET_ID,
            suggestion_id="decision-cleanup",
            decision_receipt_id=first.suggestion.decision_receipt_id,
            expected_state="accepting",
            expected_revision=second.suggestion.revision,
            expected_lease_token=second.suggestion.acceptance_lease_token,
            now=NOW + timedelta(minutes=3),
        )
    released_second = store.release_acceptance(
        dataset_id=DATASET_ID,
        suggestion_id="decision-cleanup",
        decision_receipt_id=second.suggestion.decision_receipt_id,
        expected_state="accepting",
        expected_revision=second.suggestion.revision,
        expected_lease_token=second.suggestion.acceptance_lease_token,
        now=NOW + timedelta(minutes=3),
    )
    db.execute_query("DELETE FROM note_graph_suggestions WHERE id='decision-cleanup'")
    decision_replay = store.claim_acceptance(
        dataset_id=DATASET_ID,
        suggestion_id="decision-cleanup",
        expected_revision=3,
        expected_source_fingerprint=decision_source_fp,
        expected_target_fingerprint=decision_target_fp,
        idempotency_key="decision-new-key",
        now=NOW + timedelta(days=31),
    )
    assert decision_replay.envelope == released_second.envelope
    assert decision_replay.suggestion is None
    with pytest.raises(RuntimeError, match="notes_graph_suggestion_idempotency_mismatch"):
        store.claim_acceptance(
            dataset_id=DATASET_ID,
            suggestion_id="decision-cleanup",
            expected_revision=4,
            expected_source_fingerprint=decision_source_fp,
            expected_target_fingerprint=decision_target_fp,
            idempotency_key="decision-new-key",
            now=NOW + timedelta(days=31),
        )

    missing_decision_source = "40000000-0000-4000-8000-000000000030"
    missing_decision_target = "40000000-0000-4000-8000-000000000031"
    db.add_note("Missing decision source", "body", note_id=missing_decision_source)
    db.add_note("Missing decision target", "body", note_id=missing_decision_target)
    _publish_related(
        db,
        source=missing_decision_source,
        target=missing_decision_target,
        key="missing-decision-publication",
        suggestion_id="missing-decision",
    )
    missing_source_fp = _note_fingerprint(db, missing_decision_source)
    missing_target_fp = _note_fingerprint(db, missing_decision_target)
    store.claim_acceptance(
        dataset_id=DATASET_ID,
        suggestion_id="missing-decision",
        expected_revision=1,
        expected_source_fingerprint=missing_source_fp,
        expected_target_fingerprint=missing_target_fp,
        idempotency_key="missing-decision-key",
        now=NOW,
    )
    db.execute_query("DELETE FROM note_graph_suggestions WHERE id='missing-decision'")
    with pytest.raises(RuntimeError, match="notes_graph_suggestion_accept_resource_missing"):
        store.claim_acceptance(
            dataset_id=DATASET_ID,
            suggestion_id="missing-decision",
            expected_revision=1,
            expected_source_fingerprint=missing_source_fp,
            expected_target_fingerprint=missing_target_fp,
            idempotency_key="missing-decision-key",
            now=NOW,
        )
    with pytest.raises(ValueError, match="notes_graph_cleanup_limit_invalid"):
        store.cleanup_retention(dataset_id=DATASET_ID, now=NOW, limit=101)


def _exercise_reverse_and_tag_activation(db: CharactersRAGDB) -> None:
    store = db.note_graph_suggestion_store
    source = "50000000-0000-4000-8000-000000000001"
    target = "50000000-0000-4000-8000-000000000002"
    db.add_note("Canonical source", "body", note_id=source)
    db.add_note("Canonical target", "body", note_id=target)
    _publish_related(db, source=source, target=target, key="canonical-forward", suggestion_id="forward")
    _publish_related(db, source=target, target=source, key="canonical-reverse", suggestion_id="reverse")
    rows = db.execute_query(
        "SELECT id,state FROM note_graph_suggestions "
        "WHERE id IN ('forward','reverse') ORDER BY id"
    ).fetchall()
    assert [(row["id"], row["state"]) for row in rows] == [
        ("forward", "stale"),
        ("reverse", "pending"),
    ]
    store.reject_suggestion(
        dataset_id=DATASET_ID,
        suggestion_id="reverse",
        expected_revision=1,
        expected_source_fingerprint=_note_fingerprint(db, target),
        expected_target_fingerprint=_note_fingerprint(db, source),
        idempotency_key="canonical-reject",
        now=NOW,
    )
    suppressed = _publish_related(
        db,
        source=source,
        target=target,
        key="canonical-suppressed",
        suggestion_id="suppressed",
        model="model-b",
    )
    assert suppressed.suggestion_count == 0

    keyword_id = db.add_keyword("Research")
    keyword = db.get_keyword_by_id(keyword_id)
    assert keyword is not None
    tag_running = _running_for(db, source, "tag-reresolve", model="model-c")
    tag_publishing = store.stage_suggestions(
        dataset_id=DATASET_ID,
        run_id=tag_running.id,
        expected_state="running",
        expected_revision=tag_running.revision,
        result_digest=f"sha256:{'8' * 64}",
        candidates=(
            {
                "id": "renamed-tag",
                "kind": "tag",
                "normalized_tag": "research",
                "display_tag": "Research",
                "keyword_sync_id": keyword["sync_id"],
                "match_strength": "possible",
                "rationale": "Bounded rationale",
                "evidence": (),
            },
        ),
        invalid_item_count=0,
        now=NOW,
    )
    db.rename_keyword(keyword_id, "Deep Research", expected_version=1)
    tag_result = store.activate_staged_run(
        dataset_id=DATASET_ID,
        run_id=tag_publishing.id,
        expected_state="publishing",
        expected_revision=tag_publishing.revision,
        observed_job_id=tag_publishing.job_id,
        observed_completion_token=tag_publishing.expected_completion_token,
        observed_result_digest=tag_publishing.result_digest,
        now=NOW,
    )
    assert tag_result.state.value == "succeeded"
    renamed_row = db.execute_query(
        "SELECT normalized_tag,display_tag FROM note_graph_suggestions WHERE id='renamed-tag'"
    ).fetchone()
    assert (renamed_row["normalized_tag"], renamed_row["display_tag"]) == (
        "deep research",
        "Deep Research",
    )

    deleted_id = db.add_keyword("Disposable")
    deleted = db.get_keyword_by_id(deleted_id)
    assert deleted is not None
    deleted_running = _running_for(db, source, "tag-deleted", model="model-e")
    deleted_publishing = store.stage_suggestions(
        dataset_id=DATASET_ID,
        run_id=deleted_running.id,
        expected_state="running",
        expected_revision=deleted_running.revision,
        result_digest=f"sha256:{'5' * 64}",
        candidates=(
            {
                "id": "deleted-tag",
                "kind": "tag",
                "normalized_tag": "disposable",
                "display_tag": "Disposable",
                "keyword_sync_id": deleted["sync_id"],
                "match_strength": "possible",
                "rationale": "Bounded rationale",
                "evidence": (),
            },
            {
                "id": "surviving-tag",
                "kind": "tag",
                "normalized_tag": "surviving",
                "display_tag": "Surviving",
                "keyword_sync_id": None,
                "match_strength": "possible",
                "rationale": "Bounded rationale",
                "evidence": (),
            },
        ),
        invalid_item_count=0,
        now=NOW,
    )
    assert db.soft_delete_keyword(deleted_id, expected_version=1)
    deleted_result = store.activate_staged_run(
        dataset_id=DATASET_ID,
        run_id=deleted_publishing.id,
        expected_state="publishing",
        expected_revision=deleted_publishing.revision,
        observed_job_id=deleted_publishing.job_id,
        observed_completion_token=deleted_publishing.expected_completion_token,
        observed_result_digest=deleted_publishing.result_digest,
        now=NOW,
    )
    assert deleted_result.state.value == "succeeded"
    assert deleted_result.suggestion_count == 1
    assert [str(row["id"]) for row in db.execute_query(
        "SELECT id FROM note_graph_suggestions WHERE run_id=? ORDER BY id",
        (deleted_publishing.id,),
    ).fetchall()] == ["surviving-tag"]

    member_id = db.add_keyword("Already Present")
    member = db.get_keyword_by_id(member_id)
    assert member is not None
    assert db.link_note_to_keyword(source, member_id)
    member_running = _running_for(db, source, "tag-membership", model="model-d")
    member_publishing = store.stage_suggestions(
        dataset_id=DATASET_ID,
        run_id=member_running.id,
        expected_state="running",
        expected_revision=member_running.revision,
        result_digest=f"sha256:{'6' * 64}",
        candidates=(
            {
                "id": "present-tag",
                "kind": "tag",
                "normalized_tag": "already present",
                "display_tag": "Already Present",
                "keyword_sync_id": member["sync_id"],
                "match_strength": "possible",
                "rationale": "Bounded rationale",
                "evidence": (),
            },
        ),
        invalid_item_count=0,
        now=NOW,
    )
    member_result = store.activate_staged_run(
        dataset_id=DATASET_ID,
        run_id=member_publishing.id,
        expected_state="publishing",
        expected_revision=member_publishing.revision,
        observed_job_id=member_publishing.job_id,
        observed_completion_token=member_publishing.expected_completion_token,
        observed_result_digest=member_publishing.result_digest,
        now=NOW,
    )
    assert member_result.suggestion_count == 0


def test_sqlite_fix_round_receipts_identities_tags_and_fences(tmp_path) -> None:
    db = _new_db(tmp_path)
    try:
        _exercise_receipt_cleanup_and_acceptance_fences(db)
        _exercise_reverse_and_tag_activation(db)
    finally:
        db.close_all_connections()


def test_postgres_fix_round_receipts_identities_tags_and_fences(tmp_path, pg_database_config) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = _new_db(tmp_path, backend=backend)
    try:
        _exercise_receipt_cleanup_and_acceptance_fences(db)
        _exercise_reverse_and_tag_activation(db)
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()
