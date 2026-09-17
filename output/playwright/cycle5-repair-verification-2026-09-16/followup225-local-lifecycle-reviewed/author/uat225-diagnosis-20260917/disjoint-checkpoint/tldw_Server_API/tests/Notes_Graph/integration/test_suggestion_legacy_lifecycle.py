"""Fresh local admission through the actual store and Jobs coordinator."""

import pytest

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Notes_Graph.suggestion_content import content_fingerprint
from tldw_Server_API.app.core.Notes_Graph.suggestion_jobs import (
    JOB_PAYLOAD_KEYS,
    JOB_QUEUE,
    SuggestionAdmissionService,
)
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
