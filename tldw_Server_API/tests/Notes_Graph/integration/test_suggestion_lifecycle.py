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
            "id,owner_user_id,dataset_id,source_note_id,source_fingerprint,provider,model,prompt_contract_version,job_id,expected_completion_token,state,revision,created_at,expires_at"
            ") VALUES (?,?,?,?,?,'openai','model-a','prompt-v1',?,? ,?,1,?,?)",
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
        ("queued", "job-queued", "stale"),
        ("running", "job-running", "cancelling"),
        ("publishing", "job-publishing", "stale"),
    )
    for index, (initial, job_id, expected) in enumerate(cases):
        source_id = f"00000000-0000-4000-8000-{index + 1:012d}"
        target_id = f"00000000-0000-4000-8000-{index + 101:012d}"
        run_id = f"run-source-{initial}"
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
        if initial in {"queued", "running"}:
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


def _exercise_lifecycle_contract(db: CharactersRAGDB) -> None:
    _exercise_source_lifecycle(db)
    _exercise_target_and_restore_lifecycle(db)
    _exercise_accepting_target_fence(db)
    _exercise_tag_membership_does_not_invalidate(db)
    _exercise_hard_delete_cascades(db)


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
        running = store.transition_run(
            dataset_id=DATASET_ID,
            run_id=queued.id,
            expected_state="queued",
            expected_revision=queued.revision,
            new_state="running",
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
        pending = store.release_acceptance(
            dataset_id=DATASET_ID,
            suggestion_id="postgres-suggestion",
            expected_state="accepting",
            expected_revision=claim.suggestion.revision,
            expected_lease_token=claim.suggestion.acceptance_lease_token,
            now=NOW,
        )
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
