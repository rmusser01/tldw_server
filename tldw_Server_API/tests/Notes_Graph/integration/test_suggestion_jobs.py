from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Notes_Graph import suggestion_service
from tldw_Server_API.app.core.Notes_Graph.suggestion_content import content_fingerprint
from tldw_Server_API.app.core.Notes_Graph.suggestion_jobs import (
    JOB_DOMAIN,
    JOB_PAYLOAD_KEYS,
    JOB_QUEUE,
    JOB_RESULT_KEYS,
    JOB_TYPE,
    PublicationReceiptError,
    SuggestionAdmissionService,
    SuggestionPublisher,
    validate_publication_receipt,
)
from tldw_Server_API.app.core.Notes_Graph.suggestion_service import (
    SuggestionWorker,
    SuggestionWorkerCancelled,
)

pytestmark = pytest.mark.integration

NOW = datetime(2026, 8, 27, 16, 0, tzinfo=timezone.utc)
DATASET_ID = "dataset-1"
SOURCE_ID = "10000000-0000-4000-8000-000000000001"


@pytest.fixture()
def stores(tmp_path, monkeypatch):
    monkeypatch.setenv("JOBS_ALLOWED_QUEUES_NOTES", JOB_QUEUE)
    notes = CharactersRAGDB(str(tmp_path / "notes.db"), client_id="owner-1")
    notes.add_note("Source", "source body", note_id=SOURCE_ID)
    with notes.transaction() as conn:
        conn.execute(
            "INSERT INTO note_task_scope_authority(owner_user_id,dataset_id) VALUES (?,?)",
            (notes.client_id, DATASET_ID),
        )
    jobs = JobManager(tmp_path / "jobs.db")
    try:
        yield notes, jobs
    finally:
        notes.close_all_connections()


def _admit(notes, jobs, *, key="request-1", model="model-a", now=NOW):
    note = notes.get_note_by_id(SOURCE_ID, include_deleted=True)
    assert note is not None
    service = SuggestionAdmissionService(
        store=notes.note_graph_suggestion_store,
        jobs=jobs,
        owner_user_id="owner-1",
    )
    return service.admit(
        dataset_id=DATASET_ID,
        source_note_id=SOURCE_ID,
        source_fingerprint=content_fingerprint(note["title"], note["content"]),
        provider="openai",
        model=model,
        capability_revision=f"sha256:{'a' * 64}",
        prompt_contract_version="notes-graph-suggestions-v1",
        idempotency_key=key,
        now=now,
    )


def test_admission_uses_content_free_exact_job_contract_and_replays(stores) -> None:
    notes, jobs = stores
    first = _admit(notes, jobs)
    replay = _admit(notes, jobs)

    assert first.run.id == first.job["idempotency_key"] == replay.run.id
    assert (first.job["domain"], first.job["queue"], first.job["job_type"]) == (
        JOB_DOMAIN,
        JOB_QUEUE,
        JOB_TYPE,
    )
    assert first.job["owner_user_id"] == "owner-1"
    assert int(first.job["max_retries"]) == 0
    assert set(first.job["payload"]) == JOB_PAYLOAD_KEYS
    assert "owner_user_id" not in first.job["payload"]
    assert first.run.job_id == first.job["uuid"]
    assert first.run.expected_completion_token.startswith("placeholder_")
    assert replay.disposition == "terminal_replay"
    assert jobs.count_jobs(domain=JOB_DOMAIN, owner_user_id="owner-1") == 1


def test_admission_recovers_before_enqueue_and_never_calls_provider(stores, monkeypatch) -> None:
    notes, jobs = stores
    original = jobs.create_job
    calls = 0

    def interrupted(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ConnectionError("jobs unavailable")
        return original(**kwargs)

    monkeypatch.setattr(jobs, "create_job", interrupted)
    with pytest.raises(ConnectionError):
        _admit(notes, jobs, key="resume-before-enqueue")
    recovered = _admit(notes, jobs, key="resume-before-enqueue")

    assert recovered.run.state.value == "queued"
    assert calls == 2


def test_admission_enforces_owner_active_and_hourly_limits(stores) -> None:
    notes, jobs = stores
    _admit(notes, jobs, key="active-1")
    with pytest.raises(RuntimeError, match="notes_graph_owner_active_run_conflict"):
        _admit(notes, jobs, key="active-2", model="model-b")


def test_admission_enforces_twenty_per_owner_per_hour(stores) -> None:
    notes, jobs = stores
    note = notes.get_note_by_id(SOURCE_ID, include_deleted=True)
    assert note is not None
    fingerprint = content_fingerprint(note["title"], note["content"])
    payload = {
        "schema_version": 1,
        "run_id": "historical-run",
        "dataset_id": DATASET_ID,
        "source_note_id": SOURCE_ID,
        "source_fingerprint": fingerprint,
        "provider": "openai",
        "model": "model-a",
        "capability_revision": f"sha256:{'a' * 64}",
        "prompt_contract_version": "notes-graph-suggestions-v1",
    }

    for index in range(20):
        jobs.create_job(
            domain=JOB_DOMAIN,
            queue=JOB_QUEUE,
            job_type=JOB_TYPE,
            payload=payload,
            owner_user_id="owner-1",
            idempotency_key=f"historical-{index}",
            max_retries=0,
        )
    conn = jobs._connect()
    try:
        with conn:
            conn.execute(
                "UPDATE jobs SET status='completed',created_at=?,completed_at=? WHERE domain=?",
                (
                    NOW.strftime("%Y-%m-%d %H:%M:%S.%f"),
                    NOW.strftime("%Y-%m-%d %H:%M:%S.%f"),
                    JOB_DOMAIN,
                ),
            )
    finally:
        conn.close()
    with pytest.raises(RuntimeError, match="notes_graph_admission_rate_limited"):
        _admit(notes, jobs, key="rate-21")


def _stage_and_complete(notes, jobs):
    admitted = _admit(notes, jobs)
    acquired = jobs.acquire_next_job(
        domain=JOB_DOMAIN,
        queue=JOB_QUEUE,
        job_type=JOB_TYPE,
        worker_id="worker-1",
        lease_seconds=30,
    )
    assert acquired is not None
    running = notes.note_graph_suggestion_store.start_run(
        dataset_id=DATASET_ID,
        run_id=admitted.run.id,
        expected_state="queued",
        expected_revision=admitted.run.revision,
        expected_job_id=acquired["uuid"],
        acquired_completion_token=acquired["lease_id"],
        now=NOW,
    )
    digest = f"sha256:{'d' * 64}"
    publishing = notes.note_graph_suggestion_store.stage_suggestions(
        dataset_id=DATASET_ID,
        run_id=running.id,
        expected_state="running",
        expected_revision=running.revision,
        expected_job_id=acquired["uuid"],
        expected_completion_token=acquired["lease_id"],
        result_digest=digest,
        candidates=(),
        invalid_item_count=0,
        now=NOW,
    )
    result = {
        "run_id": running.id,
        "result_digest": digest,
        "candidate_count": 0,
        "evidence_count": 0,
        "validated_count": 0,
        "dropped_count": 0,
        "input_tokens": 0,
        "output_tokens": 0,
    }
    assert set(result) == JOB_RESULT_KEYS
    return admitted, acquired, publishing, result


def test_publication_is_stage_before_complete_receipt_bound_and_replay_safe(stores) -> None:
    notes, jobs = stores
    admitted, acquired, publishing, result = _stage_and_complete(notes, jobs)
    publisher = SuggestionPublisher(
        jobs=jobs,
        store_factory=lambda owner: notes.note_graph_suggestion_store,
    )

    with pytest.raises(PublicationReceiptError, match="receipt_pending"):
        publisher.publish(
            run=publishing,
            job_uuid=acquired["uuid"],
            owner_user_id="owner-1",
            dataset_id=DATASET_ID,
            now=NOW,
        )
    assert jobs.complete_job(
        int(acquired["id"]),
        result=result,
        worker_id="worker-1",
        lease_id=acquired["lease_id"],
        completion_token=acquired["lease_id"],
    )
    published = publisher.publish(
        run=publishing,
        job_uuid=acquired["uuid"],
        owner_user_id="owner-1",
        dataset_id=DATASET_ID,
        now=NOW,
    )
    with pytest.raises(RuntimeError, match="notes_graph_publication_receipt_mismatch"):
        publisher.publish(
            run=publishing,
            job_uuid=acquired["uuid"],
            owner_user_id="owner-1",
            dataset_id=DATASET_ID,
            now=NOW,
        )

    assert publishing.state.value == "publishing"
    assert published.state.value == "succeeded"


def test_publication_activates_from_archived_terminal_job(stores) -> None:
    notes, jobs = stores
    _admitted, acquired, publishing, result = _stage_and_complete(notes, jobs)
    assert jobs.complete_job(
        int(acquired["id"]),
        result=result,
        worker_id="worker-1",
        lease_id=acquired["lease_id"],
        completion_token=acquired["lease_id"],
    )
    conn = jobs._connect()
    try:
        with conn:
            conn.execute(
                "UPDATE jobs SET completed_at='2000-01-01 00:00:00' WHERE id=?",
                (int(acquired["id"]),),
            )
    finally:
        conn.close()
    assert jobs.prune_jobs(statuses=["completed"], older_than_days=1) == 1
    assert jobs.get_job_by_uuid(acquired["uuid"]) is None
    archived = jobs.get_job_or_archived_by_uuid(
        acquired["uuid"],
        domain=JOB_DOMAIN,
        owner_user_id="owner-1",
    )
    assert archived is not None and archived["archived"] is True

    published = SuggestionPublisher(
        jobs=jobs,
        store_factory=lambda _owner: notes.note_graph_suggestion_store,
    ).publish(
        run=publishing,
        job_uuid=acquired["uuid"],
        owner_user_id="owner-1",
        dataset_id=DATASET_ID,
        now=NOW,
    )

    assert published is not None and published.state.value == "succeeded"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("uuid", "wrong-job"),
        ("owner_user_id", "wrong-owner"),
        ("domain", "wrong-domain"),
        ("queue", "wrong-queue"),
        ("job_type", "wrong-type"),
        ("status", "failed"),
        ("completion_token", "wrong-token"),
    ],
)
def test_publication_receipt_rejects_every_immutable_mismatch(field, value) -> None:
    run = SimpleNamespace(
        id="run-1",
        job_id="job-1",
        owner_user_id="owner-1",
        expected_completion_token="lease-1",
        result_digest=f"sha256:{'1' * 64}",
    )
    job = {
        "uuid": "job-1",
        "owner_user_id": "owner-1",
        "domain": JOB_DOMAIN,
        "queue": JOB_QUEUE,
        "job_type": JOB_TYPE,
        "status": "completed",
        "completion_token": "lease-1",
        "result": {
            "run_id": "run-1",
            "result_digest": f"sha256:{'1' * 64}",
            "candidate_count": 0,
            "evidence_count": 0,
            "validated_count": 0,
            "dropped_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        },
    }
    job[field] = value
    with pytest.raises(PublicationReceiptError, match="receipt_mismatch"):
        validate_publication_receipt(job=job, run=run, owner_user_id="owner-1")


def test_publication_receipt_rejects_run_and_digest_mismatch() -> None:
    run = SimpleNamespace(
        id="run-1",
        job_id="job-1",
        owner_user_id="owner-1",
        expected_completion_token="lease-1",
        result_digest=f"sha256:{'1' * 64}",
    )
    base = {
        "uuid": "job-1",
        "owner_user_id": "owner-1",
        "domain": JOB_DOMAIN,
        "queue": JOB_QUEUE,
        "job_type": JOB_TYPE,
        "status": "completed",
        "completion_token": "lease-1",
        "result": {
            "run_id": "run-1",
            "result_digest": f"sha256:{'1' * 64}",
            "candidate_count": 0,
            "evidence_count": 0,
            "validated_count": 0,
            "dropped_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        },
    }
    for key, value in (("run_id", "wrong-run"), ("result_digest", f"sha256:{'2' * 64}")):
        job = {**base, "result": {**base["result"], key: value}}
        with pytest.raises(PublicationReceiptError, match="receipt_mismatch"):
            validate_publication_receipt(job=job, run=run, owner_user_id="owner-1")


def test_default_worker_prepare_builds_request_and_freshness_returns_none(
    monkeypatch,
) -> None:
    source = SimpleNamespace(note_id=SOURCE_ID, title="Source", content="body")
    retrieval = SimpleNamespace(source_note_id=SOURCE_ID)
    prepared = object()
    captured: dict[str, object] = {}

    class Store:
        def load_source_note(self, **kwargs):
            assert kwargs == {"dataset_id": DATASET_ID, "note_id": SOURCE_ID}
            return source

    def build_request(**kwargs):
        captured.update(kwargs)
        return prepared

    monkeypatch.setattr(suggestion_service, "build_generation_request", build_request)

    assert (
        suggestion_service._default_prepare(
            store=Store(),
            dataset_id=DATASET_ID,
            retrieval=retrieval,
        )
        is prepared
    )
    assert captured == {
        "retrieval": retrieval,
        "source_title": "Source",
        "source_content": "body",
    }
    assert (
        suggestion_service._default_freshness_check(
            store=Store(),
            dataset_id=DATASET_ID,
            running=SimpleNamespace(
                source_note_id=SOURCE_ID,
                source_fingerprint=content_fingerprint("Source", "body"),
            ),
            generated=SimpleNamespace(relationships=()),
        )
        is None
    )


@pytest.mark.asyncio
async def test_worker_revalidates_immediately_before_one_call_and_fences_stage() -> None:
    events: list[str] = []
    run = SimpleNamespace(
        id="run-1",
        revision=7,
        job_id="job-1",
        expected_completion_token="placeholder-1",
        state=SimpleNamespace(value="queued"),
        source_note_id=SOURCE_ID,
        source_fingerprint=f"sha256:{'a' * 64}",
        provider="openai",
        model="model-a",
        capability_revision="cap-v1",
        prompt_contract_version="notes-graph-suggestions-v1",
    )

    class Store:
        def get_run(self, **kwargs):
            events.append("load")
            assert kwargs == {"dataset_id": DATASET_ID, "run_id": "run-1"}
            return run

        def start_run(self, **kwargs):
            events.append("start")
            assert kwargs["expected_revision"] == 7
            assert kwargs["expected_job_id"] == "job-1"
            assert kwargs["acquired_completion_token"] == "lease-1"
            return SimpleNamespace(**{**run.__dict__, "revision": 8, "expected_completion_token": "lease-1"})

        def stage_suggestions(self, **kwargs):
            events.append("stage")
            assert kwargs["expected_job_id"] == "job-1"
            assert kwargs["expected_completion_token"] == "lease-1"
            return SimpleNamespace(state=SimpleNamespace(value="publishing"))

    async def generate(**_kwargs):
        events.append("provider")
        return SimpleNamespace(relationships=(), tags=(), validation_counts={})

    worker = SuggestionWorker(
        store_factory=lambda _owner: Store(),
        retrieve=lambda **_kwargs: events.append("retrieve") or object(),
        prepare=lambda **_kwargs: events.append("prepare") or object(),
        resolve_capability=lambda **_kwargs: (
            events.append("capability") or (SimpleNamespace(revision="cap-v1", generation_available=True), object())
        ),
        generate=generate,
        freshness_check=lambda **_kwargs: None,
        cancellation_requested=lambda _job: False,
        now=lambda: NOW,
    )
    result = await worker.handle(
        {
            "uuid": "job-1",
            "owner_user_id": "owner-1",
            "domain": JOB_DOMAIN,
            "queue": JOB_QUEUE,
            "job_type": JOB_TYPE,
            "lease_id": "lease-1",
            "payload": {
                "schema_version": 1,
                "run_id": "run-1",
                "dataset_id": DATASET_ID,
                "source_note_id": SOURCE_ID,
                "source_fingerprint": f"sha256:{'a' * 64}",
                "provider": "openai",
                "model": "model-a",
                "capability_revision": "cap-v1",
                "prompt_contract_version": "notes-graph-suggestions-v1",
            },
        }
    )

    assert events == ["load", "start", "retrieve", "prepare", "capability", "provider", "stage"]
    assert set(result) == JOB_RESULT_KEYS


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("state", SimpleNamespace(value="running")),
        ("job_id", "wrong-job"),
        ("source_note_id", "wrong-source"),
        ("source_fingerprint", f"sha256:{'b' * 64}"),
        ("provider", "wrong-provider"),
        ("model", "wrong-model"),
        ("capability_revision", "wrong-capability"),
        ("prompt_contract_version", "wrong-prompt-contract"),
    ],
)
async def test_worker_rejects_run_binding_mismatch_before_start_or_external_work(
    field,
    value,
) -> None:
    payload = {
        "schema_version": 1,
        "run_id": "run-1",
        "dataset_id": DATASET_ID,
        "source_note_id": SOURCE_ID,
        "source_fingerprint": f"sha256:{'a' * 64}",
        "provider": "openai",
        "model": "model-a",
        "capability_revision": "cap-v1",
        "prompt_contract_version": "notes-graph-suggestions-v1",
    }
    immutable_keys = (
        "source_note_id",
        "source_fingerprint",
        "provider",
        "model",
        "capability_revision",
        "prompt_contract_version",
    )
    run_fields = {
        "id": "run-1",
        "revision": 7,
        "job_id": "job-1",
        "state": SimpleNamespace(value="queued"),
        **{key: payload[key] for key in immutable_keys},
    }
    run_fields[field] = value
    calls: list[str] = []

    class Store:
        def get_run(self, **_kwargs):
            calls.append("load")
            return SimpleNamespace(**run_fields)

        def start_run(self, **_kwargs):
            calls.append("start")
            raise AssertionError("mismatched run must not start")

    worker = SuggestionWorker(
        store_factory=lambda owner: calls.append(f"owner:{owner}") or Store(),
        retrieve=lambda **_kwargs: calls.append("retrieve"),
        prepare=lambda **_kwargs: calls.append("prepare"),
        resolve_capability=lambda **_kwargs: calls.append("capability"),
        generate=lambda **_kwargs: calls.append("provider"),
        freshness_check=lambda **_kwargs: calls.append("freshness"),
        cancellation_requested=lambda _job: False,
        now=lambda: NOW,
    )

    with pytest.raises(RuntimeError, match="notes_graph_job_contract_invalid"):
        await worker.handle(
            {
                "uuid": "job-1",
                "owner_user_id": "owner-1",
                "domain": JOB_DOMAIN,
                "queue": JOB_QUEUE,
                "job_type": JOB_TYPE,
                "lease_id": "lease-1",
                "payload": payload,
            }
        )

    assert calls == ["owner:owner-1", "load"]


@pytest.mark.asyncio
async def test_worker_revision_cas_failure_prevents_retrieval_and_provider() -> None:
    calls: list[str] = []
    payload = {
        "schema_version": 1,
        "run_id": "run-1",
        "dataset_id": DATASET_ID,
        "source_note_id": SOURCE_ID,
        "source_fingerprint": f"sha256:{'a' * 64}",
        "provider": "openai",
        "model": "model-a",
        "capability_revision": "cap-v1",
        "prompt_contract_version": "notes-graph-suggestions-v1",
    }
    immutable_keys = (
        "source_note_id",
        "source_fingerprint",
        "provider",
        "model",
        "capability_revision",
        "prompt_contract_version",
    )

    class Store:
        def get_run(self, **_kwargs):
            calls.append("load")
            return SimpleNamespace(
                id="run-1",
                revision=11,
                job_id="job-1",
                state=SimpleNamespace(value="queued"),
                **{key: payload[key] for key in immutable_keys},
            )

        def start_run(self, **kwargs):
            calls.append(f"start:{kwargs['expected_revision']}")
            raise RuntimeError("notes_graph_run_conflict")

    worker = SuggestionWorker(
        store_factory=lambda _owner: Store(),
        retrieve=lambda **_kwargs: calls.append("retrieve"),
        prepare=lambda **_kwargs: calls.append("prepare"),
        resolve_capability=lambda **_kwargs: calls.append("capability"),
        generate=lambda **_kwargs: calls.append("provider"),
        freshness_check=lambda **_kwargs: calls.append("freshness"),
        cancellation_requested=lambda _job: False,
        now=lambda: NOW,
    )

    with pytest.raises(RuntimeError, match="notes_graph_run_conflict"):
        await worker.handle(
            {
                "uuid": "job-1",
                "owner_user_id": "owner-1",
                "domain": JOB_DOMAIN,
                "queue": JOB_QUEUE,
                "job_type": JOB_TYPE,
                "lease_id": "lease-1",
                "payload": payload,
            }
        )

    assert calls == ["load", "start:11"]


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_on_check", [1, 2])
async def test_worker_cancellation_before_or_after_call_never_stages(cancel_on_check) -> None:
    checks = 0
    calls = 0
    staged = False

    def cancel(_job):
        nonlocal checks
        checks += 1
        return checks == cancel_on_check

    async def generate(**_kwargs):
        nonlocal calls
        calls += 1
        return SimpleNamespace(relationships=(), tags=(), validation_counts={})

    class Store:
        def get_run(self, **_kwargs):
            return SimpleNamespace(
                id="run-1",
                revision=2,
                job_id="job-1",
                source_note_id=SOURCE_ID,
                source_fingerprint=f"sha256:{'a' * 64}",
                provider="openai",
                model="model-a",
                capability_revision="cap-v1",
                prompt_contract_version="notes-graph-suggestions-v1",
                state=SimpleNamespace(value="queued"),
            )

        def start_run(self, **_kwargs):
            return SimpleNamespace(id="run-1", revision=3, job_id="job-1", expected_completion_token="lease-1")

        def stage_suggestions(self, **_kwargs):
            nonlocal staged
            staged = True

    worker = SuggestionWorker(
        store_factory=lambda _owner: Store(),
        retrieve=lambda **_kwargs: object(),
        prepare=lambda **_kwargs: object(),
        resolve_capability=lambda **_kwargs: (SimpleNamespace(revision="cap-v1", generation_available=True), object()),
        generate=generate,
        freshness_check=lambda **_kwargs: None,
        cancellation_requested=cancel,
        now=lambda: NOW,
    )
    with pytest.raises(SuggestionWorkerCancelled):
        await worker.handle(
            {
                "uuid": "job-1",
                "owner_user_id": "owner-1",
                "domain": JOB_DOMAIN,
                "queue": JOB_QUEUE,
                "job_type": JOB_TYPE,
                "lease_id": "lease-1",
                "payload": {
                    "schema_version": 1,
                    "run_id": "run-1",
                    "dataset_id": DATASET_ID,
                    "source_note_id": SOURCE_ID,
                    "source_fingerprint": f"sha256:{'a' * 64}",
                    "provider": "openai",
                    "model": "model-a",
                    "capability_revision": "cap-v1",
                    "prompt_contract_version": "notes-graph-suggestions-v1",
                },
            }
        )

    assert calls == (0 if cancel_on_check == 1 else 1)
    assert staged is False
