"""Fresh local admission through the actual store and Jobs coordinator."""

import pytest

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Notes_Graph.suggestion_content import content_fingerprint
from tldw_Server_API.app.core.Notes_Graph.suggestion_jobs import (
    JOB_PAYLOAD_KEYS,
    JOB_QUEUE,
    SuggestionAdmissionService,
)
from tldw_Server_API.app.core.Notes_Graph.suggestion_service import build_suggestion_decision_service
from tldw_Server_API.tests.Notes_Graph.integration import test_suggestion_acceptance as acceptance_fixture
from tldw_Server_API.tests.Notes_Graph.integration.test_suggestion_acceptance import NOW, SOURCE_ID
from tldw_Server_API.tests.Notes_Graph.integration.test_suggestion_legacy_mutations import (
    local_product_db as local_product_db,
)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("scope", ["fresh-local", "canonical-control"])
def test_actual_admission_preserves_content_free_job_and_replay(local_product_db, tmp_path, monkeypatch, scope):
    db = local_product_db
    dataset = f"legacy:{db.client_id}" if scope == "fresh-local" else "canonical-control"
    if scope == "canonical-control":
        with db.transaction() as conn:
            conn.execute(
                "INSERT INTO note_task_scope_authority(owner_user_id,dataset_id) VALUES (?,?)",
                (db.client_id, dataset),
            )
    monkeypatch.setenv("JOBS_ALLOWED_QUEUES_NOTES", JOB_QUEUE)
    jobs = JobManager(tmp_path / "local-jobs.db")
    source = db.get_note_by_id(SOURCE_ID)
    service = SuggestionAdmissionService(store=db.note_graph_suggestion_store, jobs=jobs, owner_user_id=db.client_id)
    arguments = {
        "dataset_id": dataset,
        "source_note_id": SOURCE_ID,
        "source_fingerprint": content_fingerprint(source["title"], source["content"]),
        "provider": "openai",
        "model": "synthetic-no-provider-call",
        "capability_revision": f"sha256:{'a' * 64}",
        "prompt_contract_version": "notes-graph-suggestions-v1",
        "idempotency_key": "local-lifecycle-admission",
        "now": NOW,
    }
    admitted = service.admit(**arguments)
    replay = service.admit(**arguments)
    assert admitted.run.job_id == admitted.job["uuid"]
    assert admitted.job["payload"]["dataset_id"] == dataset
    assert set(admitted.job["payload"]) == JOB_PAYLOAD_KEYS
    assert admitted.job["idempotency_key"] == admitted.run.id
    assert admitted.job["owner_user_id"] == db.client_id
    assert admitted.job["max_retries"] == 0
    assert replay.disposition == "terminal_replay"
    assert replay.replay_envelope["run_id"] == admitted.run.id
    assert jobs.count_jobs(domain="notes", owner_user_id=db.client_id) == 1
    with db.transaction() as conn:
        authority = conn.execute(
            "SELECT dataset_id FROM note_task_scope_authority WHERE owner_user_id=?",
            (db.client_id,),
        ).fetchall()
    assert [row["dataset_id"] for row in authority] == ([] if scope == "fresh-local" else [dataset])


@pytest.mark.parametrize("kind", ["related_note", "tag"])
def test_actual_local_factory_accepts_published_product_and_replays(local_product_db, monkeypatch, kind):
    db = local_product_db
    dataset = f"legacy:{db.client_id}"
    # Reuse the actual admit/start/stage/publish fixture under its explicit local
    # dataset. No shared authority is inserted and no production method is mocked.
    monkeypatch.setattr(acceptance_fixture, "DATASET_ID", dataset)
    acceptance_fixture._publish(db, suggestion_id="fresh-local-decision", kind=kind)
    decisions = build_suggestion_decision_service(note_db=db, owner_user_id=db.client_id, dataset_id=dataset)
    assert decisions is not None
    decisions._clock = lambda: NOW
    arguments = {
        "dataset_id": dataset,
        "suggestion_id": "fresh-local-decision",
        "expected_revision": 1,
        "expected_source_fingerprint": acceptance_fixture._fingerprint(db, SOURCE_ID),
        "expected_target_fingerprint": (
            acceptance_fixture._fingerprint(db, acceptance_fixture.TARGET_ID) if kind == "related_note" else None
        ),
        "idempotency_key": "accept-fresh-local",
    }
    result = decisions.accept(**arguments)
    replay = decisions.accept(**arguments)
    assert result.envelope["state"] == "accepted"
    assert replay.envelope == result.envelope
    if kind == "tag":
        assert [item["keyword"].casefold() for item in db.get_keywords_for_note(SOURCE_ID)] == ["research"]
    else:
        assert db.notes_link_store.get(result.envelope["accepted_resource_identity"]) is not None
    with db.transaction() as conn:
        assert (
            conn.execute("SELECT 1 FROM note_task_scope_authority WHERE owner_user_id=?", (db.client_id,)).fetchone()
            is None
        )


@pytest.mark.parametrize("mismatch", ["owner", "dataset", "canonical-reserved"])
def test_local_factory_does_not_authorize_mismatched_or_retired_scope(local_product_db, mismatch):
    db = local_product_db
    owner = "another-owner" if mismatch == "owner" else db.client_id
    dataset = "legacy:another-owner" if mismatch == "dataset" else f"legacy:{db.client_id}"
    if mismatch == "canonical-reserved":
        with db.transaction() as conn:
            conn.execute(
                "INSERT INTO note_task_scope_authority(owner_user_id,dataset_id) VALUES (?,?)",
                (db.client_id, "real-canonical"),
            )
    assert build_suggestion_decision_service(note_db=db, owner_user_id=owner, dataset_id=dataset) is None


@pytest.mark.parametrize("changed", ["source", "target"])
@pytest.mark.parametrize("mutation", ["edit", "trash"])
def test_local_note_mutation_invalidates_review_in_its_transaction(local_product_db, monkeypatch, changed, mutation):
    db = local_product_db
    dataset = f"legacy:{db.client_id}"
    monkeypatch.setattr(acceptance_fixture, "DATASET_ID", dataset)
    acceptance_fixture._publish(db, suggestion_id="local-note-change", kind="related_note")
    note_id = SOURCE_ID if changed == "source" else acceptance_fixture.TARGET_ID
    with pytest.raises(RuntimeError, match="caller rollback"):
        with db.transaction() as conn:
            if mutation == "edit":
                db.update_note(note_id, {"content": "rolled back edit"}, expected_version=1, conn=conn)
            else:
                db.soft_delete_note(note_id, expected_version=1)
            assert (
                db.note_graph_suggestion_store.get_suggestion(
                    dataset_id=dataset, suggestion_id="local-note-change"
                ).state.value
                == "stale"
            )
            raise RuntimeError("caller rollback")
    assert (
        db.note_graph_suggestion_store.get_suggestion(dataset_id=dataset, suggestion_id="local-note-change").state.value
        == "pending"
    )
    assert db.get_note_by_id(note_id)["version"] == 1
    if mutation == "edit":
        db.update_note(note_id, {"content": "committed edit"}, expected_version=1)
    else:
        db.soft_delete_note(note_id, expected_version=1)
    assert (
        db.note_graph_suggestion_store.get_suggestion(dataset_id=dataset, suggestion_id="local-note-change").state.value
        == "stale"
    )


def test_local_maintenance_discovers_durable_scope_without_authority_row(local_product_db, monkeypatch):
    db = local_product_db
    dataset = f"legacy:{db.client_id}"
    assert db.note_graph_suggestion_store.list_maintenance_dataset_ids() == ()
    monkeypatch.setattr(acceptance_fixture, "DATASET_ID", dataset)
    acceptance_fixture._publish(db, suggestion_id="local-maintenance", kind="tag")
    assert db.note_graph_suggestion_store.list_maintenance_dataset_ids() == (dataset,)
