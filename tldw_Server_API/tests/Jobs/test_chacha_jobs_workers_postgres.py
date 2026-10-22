"""Jobs workers that touch ChaCha return their PostgreSQL checkouts, one family per test (TASK-13861).

Each test runs a real worker's job owner through WorkerSDK.run and stubs only the
work that does not decide where the handle's checkouts go.
"""

import asyncio
import itertools
import threading
from collections import Counter
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Chat_Macros import jobs as chat_macro_jobs
from tldw_Server_API.app.core.Chat_Macros.branch_runner import BranchPromptResult
from tldw_Server_API.app.core.Chat_Macros.repository import ChatMacroRepository
from tldw_Server_API.app.core.Chatbooks.services import jobs_worker as chatbooks_worker
from tldw_Server_API.app.core.DB_Management import ChaChaNotes_DB as chacha_module
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK
from tldw_Server_API.app.services import workspace_file_inventory_jobs_worker as inventory_worker

pytestmark = pytest.mark.integration

WRAPUP_BRANCHES = ["action_items", "decisions", "open_questions", "summary"]


@pytest.fixture
def pg_owner(pg_database_config, tmp_path, monkeypatch):
    """Owner 1's long-lived handle on a pool that records which thread holds each checkout."""
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
    monkeypatch.setattr(chacha_module, "get_content_backend", lambda _config: backend)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_dbs"))
    db = CharactersRAGDB(tmp_path / "owner.db", client_id="1", backend=backend)
    db.close_connection()

    async def owner_database(_user_id, client_id=None):  # noqa: ARG001 - the cached per-user handle
        return db

    def outstanding():
        return sorted(holders[key] for key, count in (taken - returned).items() for _ in range(count))

    try:
        yield SimpleNamespace(db=db, owner_database=owner_database, outstanding=outstanding, jobs=JobManager(tmp_path / "q.db"))
    finally:
        db.close_connection()
        pool.close_all()


@pytest.fixture
def chatbooks_cache(tmp_path, monkeypatch):
    """The Chatbooks worker's own per-user handle cache, empty and pointed at tmp_path."""
    monkeypatch.setattr(chatbooks_worker, "_DB_CACHE", {})
    monkeypatch.setattr(chatbooks_worker.DatabasePaths, "get_chacha_db_path", staticmethod(lambda _uid: tmp_path / "cb.db"))


@pytest.fixture
def pending_wrapup_run(pg_owner):
    """A pending run of the built-in wrapup macro (four branch steps) for owner 1."""
    with chacha_operation(independent=True):
        repository = ChatMacroRepository(pg_owner.db)
        repository.ensure_ready()
        run = repository.create_run(
            user_id="1",
            macro_name="wrapup",
            macro_command="wrapup",
            macro_source="builtin",
            macro_version=1,
            normalized_args={},
            status="pending",
            surface="chat",
            output_profile="default",
            context_snapshot={
                "conversation_id": "conv-1",
                "messages": [{"id": "m1", "role": "user", "excerpt": "Please wrap this up."}],
                "selected_message_ids": ["m1"],
                "model_selection": {"api_provider": "openai", "model": "gpt-test"},
                "output_profile": "default",
                "acp": {},
            },
            model_selection={"api_provider": "openai", "model": "gpt-test"},
        )
    return run.run_id


class _AnsweringBranchRunner:
    """The LLM call each branch makes, the only part not under test.

    It records which checkouts are out while the branch "calls the LLM": the event-loop
    thread must hold none, or the executor build's checkout would span every branch.
    """

    def __init__(self, outstanding, seen):
        self.outstanding, self.seen = outstanding, seen

    async def run_branch(self, *, prompt, snapshot, model_selection):  # noqa: ARG002
        self.seen.append(self.outstanding())
        return BranchPromptResult(text=f"answer to {prompt[:20]}", citations=[], usage={})


def _overlapping_branch_writes(original):
    """Hold the first branch write's transaction open across the second's, as concurrent branches can.

    The second write enters while the first is open and leaves after the first has
    committed. Branches that share one connection's transaction depth lose later writes.
    """
    first_in, second_in, first_out = threading.Event(), threading.Event(), threading.Event()
    arrivals = itertools.count()

    def upsert_branch(self, *args, **kwargs):
        arrival = next(arrivals)
        if arrival > 1:
            return original(self, *args, **kwargs)
        if arrival == 1:
            first_in.wait(10)
        with self.db.transaction():
            if arrival == 0:
                first_in.set()
                second_in.wait(10)
            else:
                second_in.set()
                first_out.wait(10)
            result = original(self, *args, **kwargs)
        if arrival == 0:
            first_out.set()
        return result

    return upsert_branch


async def _run(jobs, *, domain, handler, cancel_check=None, job_type="job", payload=None):
    jobs.create_job(domain=domain, queue="default", job_type=job_type, payload=payload or {}, owner_user_id="1")
    sdk = WorkerSDK(jobs, WorkerConfig(domain=domain, queue="default", worker_id="family-test", retry_on_exception=False))
    seen = []

    async def handle(job):
        seen.append(job["id"])
        try:
            return await handler(job)
        finally:
            sdk.stop()

    async def check(job):
        try:
            return await cancel_check(job)
        finally:
            seen.append(job["id"])
            sdk.stop()

    await asyncio.wait_for(sdk.run(handler=handle, cancel_check=check if cancel_check else None), timeout=60)
    return jobs.get_job(int(seen[0]))


async def test_a_cached_handle_used_off_the_loop_and_closed_at_job_end(pg_owner, monkeypatch):
    """Family: workspace file inventory, writing review, research output, visual identity, VN assets, study suggestions, clone."""
    monkeypatch.setattr(inventory_worker, "get_chacha_db_for_user_id", pg_owner.owner_database)
    monkeypatch.setattr(inventory_worker, "_validate_job_payload", lambda job: dict(job["payload"]))

    def scan(_payload, db, _allowed_roots, _resolver):  # the scan itself; it reads through the handle
        db.get_character_card_by_name("nobody")
        return {"status": "scanned"}

    monkeypatch.setattr(inventory_worker, "_handle_workspace_file_inventory_job_sync", scan)

    stored = await _run(pg_owner.jobs, domain="workspaces", handler=inventory_worker.handle_workspace_file_inventory_job)

    assert stored["status"] == "completed", stored.get("error_message")
    assert pg_owner.outstanding() == []


async def test_a_cancel_check_that_reads_the_owners_handle(pg_owner, monkeypatch):
    """Family: Chat_Macros' cancel check mirrors the cancellation into the owner's database."""
    monkeypatch.setattr(chat_macro_jobs, "get_chacha_db_for_user_id", pg_owner.owner_database)
    cancelled = SimpleNamespace(
        get_job=lambda _job_id: {"status": "processing", "cancel_requested_at": "now", "payload": {}},
        finalize_cancelled=lambda *_args, **_kwargs: None,
    )

    async def cancel_check(job):
        return await chat_macro_jobs.should_cancel_chat_macro_job(job, job_manager=cancelled)

    async def never_runs(_job):
        raise AssertionError("a cancelled job must not be handled")

    stored = await _run(
        pg_owner.jobs, domain="chat_macros", handler=never_runs, cancel_check=cancel_check,
        payload={"macro_run_id": "run-1", "user_id": "1"},
    )

    assert stored["status"] == "cancelled"
    assert pg_owner.outstanding() == []


async def test_a_handle_the_worker_keeps_and_never_closes(pg_owner, chatbooks_cache, monkeypatch):
    """Family: Chatbooks and Data Tables keep per-user handles in a worker cache; Persona visuals opens one per job."""
    # The worker's own per-user handle cache (_get_db) is under test; ChatbookService is not.
    monkeypatch.setattr(chatbooks_worker, "_get_service", lambda user_id: SimpleNamespace(db=chatbooks_worker._get_db(user_id)))

    async def export(service, _payload, _job_id):  # the export itself; it reads through the handle
        await asyncio.to_thread(service.db.get_character_card_by_name, "nobody")
        return {"status": "exported"}

    monkeypatch.setattr(chatbooks_worker, "_handle_export", export)

    stored = await _run(
        pg_owner.jobs, domain="chatbooks", handler=chatbooks_worker._handle_job,
        payload={"action": "export", "chatbooks_job_id": "export-1"},
    )

    assert stored["status"] == "completed", stored.get("error_message")
    assert pg_owner.outstanding() == []


async def test_a_macro_run_whose_branches_write_concurrently(pg_owner, pending_wrapup_run, monkeypatch):
    """Family: Chat_Macros runs its branches with asyncio.gather; each branch writes its own row."""
    monkeypatch.setattr(chat_macro_jobs, "get_chacha_db_for_user_id", pg_owner.owner_database)
    during_branches = []
    monkeypatch.setattr(
        chat_macro_jobs, "ChatMacroLLMBranchRunner", lambda: _AnsweringBranchRunner(pg_owner.outstanding, during_branches)
    )
    monkeypatch.setattr(ChatMacroRepository, "upsert_branch", _overlapping_branch_writes(ChatMacroRepository.upsert_branch))

    stored = await _run(
        pg_owner.jobs, domain=chat_macro_jobs.CHAT_MACROS_DOMAIN, handler=chat_macro_jobs.handle_chat_macro_job,
        job_type=chat_macro_jobs.CHAT_MACROS_JOB_TYPE, payload={"macro_run_id": pending_wrapup_run, "user_id": "1"},
    )

    assert stored["status"] == "completed", stored.get("error_message")
    with chacha_operation(independent=True):  # read what the job committed, on a fresh checkout
        repository = ChatMacroRepository(pg_owner.db)
        saved = (repository.get_run(pending_wrapup_run).status, sorted(b.step_id for b in repository.list_branches(pending_wrapup_run)))
    assert saved == ("completed", WRAPUP_BRANCHES)
    assert [held for held in during_branches if "MainThread" in held] == []
    assert pg_owner.outstanding() == []
