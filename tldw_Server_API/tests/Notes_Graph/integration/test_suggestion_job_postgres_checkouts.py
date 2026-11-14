"""The notes-graph suggestions job on PostgreSQL, run by WorkerSDK (TASK-13861).

Each pipeline step runs on a worker thread and closes its checkout on that thread.
That must keep working when an operation owns the job's checkouts, which retires
the handle's state on an explicit close.
"""

import asyncio
import functools
import threading
from collections import Counter
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerSDK
from tldw_Server_API.app.core.Notes_Graph.suggestion_content import content_fingerprint
from tldw_Server_API.app.core.Notes_Graph.suggestion_jobs import JOB_QUEUE, SuggestionAdmissionService
from tldw_Server_API.app.core.Notes_Graph.suggestion_service import SuggestionWorker
from tldw_Server_API.app.services import notes_graph_suggestions_worker as worker

pytestmark = pytest.mark.integration

DATASET_ID = "dataset-1"
SOURCE_ID = "10000000-0000-4000-8000-000000000001"
REVISION = f"sha256:{'a' * 64}"


@pytest.fixture
def admitted_job(pg_database_config, tmp_path, monkeypatch):
    """One admitted suggestion job for owner 1, whose handle draws from a recording pool."""
    monkeypatch.setenv("JOBS_ALLOWED_QUEUES_NOTES", JOB_QUEUE)
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    pool = backend.get_pool()
    holders, taken, returned = {}, Counter(), Counter()
    original_get, original_return = pool.get_connection, pool.return_connection

    def get_connection():
        raw = original_get()
        taken[id(raw)] += 1
        holders[id(raw)] = threading.current_thread().name
        return raw

    def return_connection(raw):
        returned[id(raw)] += 1
        return original_return(raw)

    monkeypatch.setattr(pool, "get_connection", get_connection)
    monkeypatch.setattr(pool, "return_connection", return_connection)
    notes = CharactersRAGDB(tmp_path / "notes.db", client_id="1", backend=backend)
    notes.add_note("Source", "source body", note_id=SOURCE_ID)
    with notes.transaction() as conn:
        conn.execute(
            "INSERT INTO note_task_scope_authority(owner_user_id,dataset_id) VALUES (?,?)",
            (notes.client_id, DATASET_ID),
        )
    jobs = JobManager(tmp_path / "jobs.db")
    note = notes.get_note_by_id(SOURCE_ID, include_deleted=True)
    SuggestionAdmissionService(store=notes.note_graph_suggestion_store, jobs=jobs, owner_user_id="1").admit(
        dataset_id=DATASET_ID,
        source_note_id=SOURCE_ID,
        source_fingerprint=content_fingerprint(note["title"], note["content"]),
        provider="openai",
        model="model-a",
        capability_revision=REVISION,
        prompt_contract_version="notes-graph-suggestions-v1",
        idempotency_key="request-1",
        now=datetime.now(timezone.utc),
    )
    notes.close_connection()

    async def owner_database(_user_id, client_id=None):  # noqa: ARG001
        return notes

    at_provider_call = []  # the checkouts out while the provider call runs

    def no_suggestions(**_kwargs):  # the provider call, the only part not under test
        at_provider_call.append(outstanding())
        return SimpleNamespace(relationships=(), tags=(), validation_counts={}, input_tokens=0, output_tokens=0)

    monkeypatch.setattr(worker, "get_chacha_db_for_user_id", owner_database)
    monkeypatch.setattr(
        worker, "resolve_generation_capability",
        lambda **_kwargs: (SimpleNamespace(revision=REVISION, generation_available=True), object()),
    )
    monkeypatch.setattr(worker, "SuggestionWorker", functools.partial(SuggestionWorker, generate=no_suggestions))

    def outstanding():
        return sorted(holders[key] for key, count in (taken - returned).items() for _ in range(count))

    try:
        yield jobs, outstanding, at_provider_call
    finally:
        notes.close_connection()
        pool.close_all()


async def _run_one_job(jobs, handler):
    sdk = WorkerSDK(jobs, worker.build_worker_config(worker_id="notes-graph-test"))
    finished = []

    async def run_once(job):
        try:
            return await handler(job, sdk=sdk)
        finally:
            finished.append(job["id"])
            sdk.stop()

    await asyncio.wait_for(sdk.run(handler=run_once), timeout=60)
    return jobs.get_job(int(finished[0]))


async def test_the_suggestions_job_completes_and_returns_every_checkout(admitted_job):
    jobs, outstanding, at_provider_call = admitted_job

    async def handler(job, *, sdk):
        return await worker.handle_notes_graph_suggestions_job(job, jobs=jobs, worker_id=sdk.cfg.worker_id)

    stored = await _run_one_job(jobs, handler)

    assert stored["status"] == "completed", stored.get("error_message")
    assert outstanding() == []
    # The unwrapped job returns its checkout after every step, so none is held
    # through the provider call; one operation around the job would hold one.
    assert at_provider_call == [[]]


async def test_the_suggestions_job_completes_when_an_operation_owns_its_checkouts(admitted_job):
    jobs, outstanding, _at_provider_call = admitted_job

    async def handler(job, *, sdk):
        with chacha_operation(independent=True):  # as an owner around the whole job would
            return await worker.handle_notes_graph_suggestions_job(job, jobs=jobs, worker_id=sdk.cfg.worker_id)

    stored = await _run_one_job(jobs, handler)

    assert stored["status"] == "completed", stored.get("error_message")
    assert outstanding() == []
