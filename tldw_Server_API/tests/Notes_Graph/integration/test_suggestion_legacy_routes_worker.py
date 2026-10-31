"""Real local HTTP admission and worker pipeline with a synthetic provider reply."""

import json
from datetime import timedelta
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.api.v1.endpoints import notes_graph_suggestions as endpoint
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Notes_Graph.suggestion_generation import parse_and_validate_generation
from tldw_Server_API.app.core.Notes_Graph.suggestion_jobs import (
    JOB_QUEUE,
    SuggestionAdmissionService,
    SuggestionPublisher,
)
from tldw_Server_API.app.core.Notes_Graph.suggestion_service import SuggestionWorker, SuggestionWorkerError
from tldw_Server_API.tests.Notes_Graph.integration import test_suggestion_acceptance as acceptance
from tldw_Server_API.tests.Notes_Graph.integration.test_suggestion_acceptance import NOW
from tldw_Server_API.tests.Notes_Graph.integration.test_suggestion_fresh_reads import (
    fresh_suggestions as fresh_suggestions,
)
from tldw_Server_API.tests.Notes_Graph.integration.test_suggestion_legacy_mutations import (
    local_product_db as local_product_db,
)
from tldw_Server_API.tests.Notes_Graph.integration.test_suggestion_legacy_retirement import (
    _arguments,
    _maintenance,
    _rows,
)
from tldw_Server_API.tests.Notes_Graph.integration.test_suggestion_unbound_scope import _ready_provider

pytestmark = pytest.mark.integration


def test_local_http_evidence_reject_reset_accept_and_receipt_replay(fresh_suggestions, monkeypatch):
    db, client, _, _ = fresh_suggestions
    dataset = "legacy:1"
    monkeypatch.setattr(acceptance, "DATASET_ID", dataset)
    with chacha_operation(independent=True):
        db.add_note("Source", "source body", note_id=acceptance.SOURCE_ID)
        db.add_note("Target", "target body", note_id=acceptance.TARGET_ID)
        acceptance._publish(db, suggestion_id="local-http-review", kind="related_note", include_evidence=True)
    url = f"/api/v1/notes/{acceptance.SOURCE_ID}/graph/suggestions"
    page = client.get(url).json()
    suggestion = page["items"][0]
    assert {item["side"] for item in suggestion["evidence"]} == {"source", "target"}
    body = {
        "expected_revision": suggestion["revision"],
        "expected_source_fingerprint": suggestion["source_fingerprint"],
        "expected_target_fingerprint": suggestion["target_fingerprint"],
    }
    rejected = client.post(url + "/local-http-review/reject", json=body, headers={"Idempotency-Key": "local-reject"})
    assert rejected.status_code == 200 and rejected.json()["state"] == "rejected", rejected.text
    replay = client.post(url + "/local-http-review/reject", json=body, headers={"Idempotency-Key": "local-reject"})
    assert replay.json() == rejected.json()
    rejected_page = client.get(url + "?state=rejected").json()
    reset = client.post(
        url + "/rejections/reset",
        json={
            "expected_rejection_revision": rejected_page["rejection_set_revision"],
            "source_fingerprint": page["current_source_fingerprint"],
            "confirm": True,
        },
        headers={"Idempotency-Key": "local-reset"},
    )
    assert reset.status_code == 200, reset.text
    # Reset permits new generation rather than silently accepting old content.
    with chacha_operation(independent=True):
        acceptance._publish(db, suggestion_id="local-http-accept", kind="related_note", include_evidence=True)
    accepted = client.post(
        url + "/local-http-accept/accept",
        json={**body, "expected_revision": 1},
        headers={"Idempotency-Key": "local-accept"},
    )
    assert accepted.status_code == 200 and accepted.json()["state"] == "accepted", accepted.text
    with chacha_operation(independent=True), db.transaction() as conn:
        assert (
            db.note_graph_suggestion_store.get_suggestion(
                dataset_id=dataset, suggestion_id="local-http-accept"
            ).state.value
            == "accepted"
        )
        assert (
            db.notes_link_store.get(
                db.note_graph_suggestion_store.get_suggestion(
                    dataset_id=dataset, suggestion_id="local-http-accept"
                ).accepted_resource_identity
            )
            is not None
        )
        assert conn.execute("SELECT 1 FROM note_task_scope_authority").fetchone() is None


def test_fresh_http_capabilities_admit_replay_cancel_without_authority_binding(
    fresh_suggestions, tmp_path, monkeypatch
):
    db, client, note, _ = fresh_suggestions
    _ready_provider(monkeypatch)
    monkeypatch.setenv("NOTES_GRAPH_SUGGESTIONS_WORKER_ENABLED", "1")
    monkeypatch.setenv("JOBS_ALLOWED_QUEUES_NOTES", JOB_QUEUE)
    jobs = JobManager(tmp_path / "route-jobs.db")
    client.app.dependency_overrides[endpoint.try_get_job_manager] = lambda: jobs
    url = f"/api/v1/notes/{note}/graph/suggestions"
    caps = client.get(url + "/capabilities")
    assert caps.status_code == 200 and caps.json()["generation_available"] is True
    headers = {"If-Match": caps.headers["etag"], "Idempotency-Key": "actual-local-http"}
    admitted = client.post(url + "/runs", json={}, headers=headers)
    assert admitted.status_code == 202, admitted.text
    replay = client.post(url + "/runs", json={}, headers=headers)
    assert replay.status_code == 202 and replay.json() == admitted.json()
    run = admitted.json()
    read = client.get(url + "/runs/" + run["id"])
    assert read.status_code == 200 and read.json()["state"] == "queued"
    cancel = client.post(
        url + "/runs/" + run["id"] + "/cancel",
        json={"expected_revision": run["revision"]},
        headers={"Idempotency-Key": "local-cancel"},
    )
    assert cancel.status_code == 200 and cancel.json()["state"] == "cancelling", cancel.text
    assert jobs.count_jobs(domain="notes", owner_user_id="1") == 1
    with chacha_operation(independent=True), db.transaction() as conn:
        assert conn.execute("SELECT 1 FROM note_task_scope_authority").fetchone() is None
        assert conn.execute("SELECT 1 FROM note_graph_suggestions").fetchone() is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome", ["publish", "retire-during-provider", "retire-after-completion", "cancel-before-provider"]
)
async def test_actual_local_worker_retrieval_preparation_and_jobs_publication(
    local_product_db, tmp_path, monkeypatch, outcome
):
    db = local_product_db
    monkeypatch.setenv("JOBS_ALLOWED_QUEUES_NOTES", JOB_QUEUE)
    jobs = JobManager(tmp_path / "worker-jobs.db")
    store = db.note_graph_suggestion_store
    args = _arguments(db)
    admitted = SuggestionAdmissionService(store=store, jobs=jobs, owner_user_id=db.client_id).admit(**args)
    job = jobs.acquire_next_job(
        domain="notes", queue=JOB_QUEUE, job_type="note_graph_suggestions", worker_id="local-worker", lease_seconds=120
    )
    assert job is not None
    calls = []

    def synthetic_reply(*, prepared, provider):
        # Preserve real retrieval, prompt/evidence construction and strict output
        # validation. Only the external model reply is synthetic.
        calls.append(prepared.source_evidence_ids)
        reply = json.dumps(
            {
                "relationships": [],
                "tags": [
                    {
                        "existing_tag_id": None,
                        "new_tag": "Research",
                        "rationale": "A useful research category.",
                        "source_evidence_ids": [prepared.source_evidence_ids[0]],
                    }
                ],
            }
        )
        result = parse_and_validate_generation(reply, prepared=prepared)
        if outcome == "retire-during-provider":
            store.reserve_canonical_scope(dataset_id="real-canonical")
        return result

    worker = SuggestionWorker(
        store_factory=lambda owner: store,
        resolve_capability=lambda **kwargs: (
            SimpleNamespace(revision=args["capability_revision"], generation_available=True),
            object(),
        ),
        cancellation_requested=lambda _: outcome == "cancel-before-provider",
        generate=synthetic_reply,
        now=lambda: NOW,
    )
    if outcome in {"retire-during-provider", "cancel-before-provider"}:
        with pytest.raises(SuggestionWorkerError):
            await worker.handle(job)
        assert _rows(db, "note_graph_suggestions") == []
        assert bool(calls) == (outcome == "retire-during-provider")
        if outcome == "retire-during-provider":
            _maintenance(db, jobs).run_pass(now=NOW + timedelta(seconds=1))
            assert _rows(db, "note_graph_suggestion_runs")[0]["state"] == "stale"
        return
    result = await worker.handle(job)
    assert len(calls) == 1 and result["candidate_count"] == 1
    publishing = store.get_run(dataset_id=args["dataset_id"], run_id=admitted.run.id)
    assert publishing.state.value == "publishing"
    assert jobs.complete_job(
        int(job["id"]),
        result=result,
        worker_id="local-worker",
        lease_id=job["lease_id"],
        completion_token=job["lease_id"],
    )
    if outcome == "retire-after-completion":
        from tldw_Server_API.app.core.DB_Management.chacha.note_graph_suggestion_store import (
            NotesGraphDatasetScopeError,
        )

        store.reserve_canonical_scope(dataset_id="real-canonical")
        with pytest.raises(NotesGraphDatasetScopeError):
            SuggestionPublisher(jobs=jobs, store_factory=lambda owner: store).publish(
                run=publishing, job_uuid=job["uuid"], owner_user_id=db.client_id, dataset_id=args["dataset_id"], now=NOW
            )
        _maintenance(db, jobs).run_pass(now=NOW + timedelta(seconds=1))
        assert _rows(db, "note_graph_suggestion_runs")[0]["state"] == "stale"
        assert _rows(db, "note_graph_suggestions") == []
        assert (
            jobs.get_job_or_archived_by_uuid(job["uuid"], domain="notes", owner_user_id=db.client_id)["status"]
            == "completed"
        )
        return
    published = SuggestionPublisher(jobs=jobs, store_factory=lambda owner: store).publish(
        run=publishing, job_uuid=job["uuid"], owner_user_id=db.client_id, dataset_id=args["dataset_id"], now=NOW
    )
    assert published.state.value == "succeeded"
    assert _rows(db, "note_graph_suggestions")[0]["state"] == "pending"
    assert db.get_keywords_for_note(args["source_note_id"]) == []
