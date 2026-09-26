"""StudyPack jobs must return the checkouts used by their real source workers."""

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from cachetools import LRUCache

from tldw_Server_API.app.api.v1.API_Deps import ChaCha_Notes_DB_Deps as deps
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import ExternalConnection, chacha_operation
from tldw_Server_API.app.core.DB_Management.chacha.runtime import ChaChaRuntimeManager
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.StudyPacks import generation_service
from tldw_Server_API.app.services import study_pack_jobs_worker as worker

pytestmark = pytest.mark.integration
EVIDENCE = "The citrine calibration is twenty-seven."


@pytest.fixture
def worker_db(request, tmp_path, monkeypatch):
    kind = getattr(request, "param", "postgresql")
    backend = (
        DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
        if kind == "postgresql"
        else None
    )
    db = CharactersRAGDB(tmp_path / "worker.db", client_id="1", backend=backend)
    backend = db.backend
    note = db.add_note(title="Committed title", content=EVIDENCE)
    db.add_character_card({"name": deps.DEFAULT_CHARACTER_NAME, "description": "Existing fixture"})
    db.close_connection()
    owner_dir = tmp_path / "owner"
    monkeypatch.setattr(deps.DatabasePaths, "get_user_base_directory", staticmethod(lambda _: owner_dir))
    cache = LRUCache(maxsize=4)
    cache[str(owner_dir)] = db
    monkeypatch.setattr(deps, "_chacha_db_instances", cache)
    monkeypatch.setattr(deps, "_CHACHA_RUNTIME", ChaChaRuntimeManager())
    monkeypatch.setattr(deps, "_CHACHA_SHUTTING_DOWN", False)
    monkeypatch.setattr(deps, "_chacha_default_char_tasks", set())
    executor = ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(deps, "_get_chacha_executor", lambda: executor)
    monkeypatch.setattr(worker, "get_media_db_for_owner", lambda _: SimpleNamespace())
    monkeypatch.setattr(
        generation_service, "_resolve_generation_provider_and_model", lambda *_: ("fixture", "fixture", {})
    )
    ledger = {}
    returns = []
    reads = []
    acquisitions = []
    pool = backend.get_pool()
    lock = threading.RLock()
    if kind == "postgresql":
        original_get = pool.get_connection
        original_return = pool.return_connection

        def get_connection():
            raw = original_get()
            with lock:
                assert id(raw) not in ledger
                ledger[id(raw)] = raw
            return raw

        def return_connection(raw):
            with lock:
                original_return(raw)
                returns.append(raw)
                ledger.pop(id(raw), None)

        monkeypatch.setattr(pool, "get_connection", get_connection)
        monkeypatch.setattr(pool, "return_connection", return_connection)
    accessor = deps.get_chacha_db_for_user_id

    async def observe_accessor(*args, **kwargs):
        acquired = await accessor(*args, **kwargs)
        assert acquired is db
        acquisitions.append(acquired)
        return acquired

    monkeypatch.setattr(worker, "get_chacha_db_for_user_id", observe_accessor)
    original_read = db.get_note_by_id

    def read_note(note_id):
        reads.append(threading.get_ident())
        return original_read(note_id)

    monkeypatch.setattr(db, "get_note_by_id", read_note)
    f = SimpleNamespace(
        db=db,
        backend=backend,
        note=note,
        kind=kind,
        ledger=ledger,
        returns=returns,
        reads=reads,
        acquisitions=acquisitions,
    )
    try:
        yield f
    finally:
        executor.shutdown(wait=True)
        db.close_all_connections()
        pool.close_all()


def job(f, number=1):
    return {
        "id": number,
        "owner_user_id": "1",
        "payload": {"title": "Citrine study", "source_items": [{"source_type": "note", "source_id": f.note}]},
    }


async def drain():
    await asyncio.sleep(0)
    while deps._chacha_default_char_tasks:
        await asyncio.gather(*tuple(deps._chacha_default_char_tasks))


def assert_returned(f):
    if f.kind == "postgresql":
        assert not f.ledger, [(raw.info.transaction_status.name, raw.closed) for raw in f.ledger.values()]


def install_model(f, monkeypatch, *, fail=False):
    prompts = []

    async def model(self, *, system_prompt, user_prompt):
        assert EVIDENCE in user_prompt
        prompts.append(user_prompt)
        if fail:
            raise RuntimeError("controlled model failure")
        return json.dumps(
            {
                "cards": [
                    {
                        "front": "What is the calibration?",
                        "back": "Twenty-seven.",
                        "citations": [{"source_type": "note", "source_id": f.note, "citation_text": EVIDENCE}],
                    }
                ]
            }
        )

    monkeypatch.setattr(generation_service.StudyPackGenerationService, "_call_generation_model", model)
    return prompts


@pytest.mark.parametrize("worker_db", ["postgresql", "sqlite"], indirect=True)
@pytest.mark.parametrize("failure", ["source", "model"])
def test_actual_source_or_model_failure_finishes_job_checkouts(worker_db, monkeypatch, failure):
    f = worker_db
    prompts = install_model(f, monkeypatch, fail=True)
    payload = job(f)
    if failure == "source":
        payload["payload"]["source_items"][0]["source_id"] = "missing-fixture-note"

    async def run():
        with pytest.raises(
            ValueError if failure == "source" else RuntimeError,
            match="not found" if failure == "source" else "controlled model failure",
        ):
            await worker.handle_study_pack_job(payload)
        await drain()
        assert_returned(f)  # Check before event-loop/executor shutdown.
        assert f.acquisitions == [f.db]
        assert f.reads and all(identity != threading.get_ident() for identity in f.reads)
        assert len(prompts) == (0 if failure == "source" else 1)

    asyncio.run(run())


@pytest.mark.parametrize("worker_db", ["postgresql", "sqlite"], indirect=True)
@pytest.mark.parametrize("repetitions", [1, 3])
def test_successful_jobs_persist_cards_and_reuse_cached_database(worker_db, monkeypatch, repetitions):
    f = worker_db
    prompts = install_model(f, monkeypatch)

    async def run():
        results = []
        for number in range(repetitions):
            results.append(await worker.handle_study_pack_job(job(f, number + 1)))
            await drain()
            assert_returned(f)
        assert f.acquisitions == [f.db] * repetitions
        assert len(prompts) == repetitions
        assert len({result["pack_id"] for result in results}) == repetitions
        with chacha_operation(independent=True):
            for result in results:
                saved = f.db.get_study_pack(int(result["pack_id"]))
                assert saved["deck_id"] == result["deck_id"]
                memberships = f.db.list_study_pack_cards(int(result["pack_id"]))
                assert len(memberships) == 1
                card = f.db.get_flashcard(memberships[0]["flashcard_uuid"])
                assert card["back"] == "Twenty-seven."

    asyncio.run(run())


@pytest.mark.parametrize("outer", ["owned", "borrowed", "legacy"])
@pytest.mark.parametrize("fail", [False, True], ids=["successful-job", "failed-job"])
@pytest.mark.parametrize("commit", [False, True], ids=["caller-rollback", "caller-commit"])
def test_job_does_not_adopt_outer_pending_write(worker_db, monkeypatch, fail, commit, outer):
    f = worker_db
    install_model(f, monkeypatch, fail=fail)

    async def run():
        borrowed = f.backend.get_pool().get_connection() if outer == "borrowed" else None
        bindings = (ExternalConnection(f.db, borrowed, f.backend),) if borrowed is not None else ()
        scope = nullcontext() if outer == "legacy" else chacha_operation(independent=True, bindings=bindings)
        try:
            with scope:
                raw = f.db._get_thread_connection()
                f.db.execute_query("UPDATE notes SET title=? WHERE id=?", ("Caller pending", f.note))
                if fail:
                    with pytest.raises(RuntimeError, match="controlled model failure"):
                        await worker.handle_study_pack_job(job(f))
                else:
                    await worker.handle_study_pack_job(job(f))
                await drain()
                assert raw.info.transaction_status.name == "INTRANS"
                assert raw not in f.returns
                assert f.db.get_note_by_id(f.note)["title"] == "Caller pending"
                assert f.backend.execute("SELECT title FROM notes WHERE id=%s", (f.note,)).scalar == "Committed title"
                (raw.commit if commit else raw.rollback)()
        finally:
            if borrowed is not None:
                borrowed.rollback()
                f.backend.get_pool().return_connection(borrowed)
            elif outer == "legacy":
                f.db.close_connection()
        assert_returned(f)
        assert f.backend.execute("SELECT title FROM notes WHERE id=%s", (f.note,)).scalar == (
            "Caller pending" if commit else "Committed title"
        )

    asyncio.run(run())


def test_cancellation_defers_return_until_actual_source_query_finishes(worker_db, monkeypatch):
    f = worker_db
    prompts = install_model(f, monkeypatch)
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    returned = threading.Event()
    return_indexes = []
    captured = []
    original = f.backend.execute
    loop_thread = threading.get_ident()

    def execute(query, params=None, **kwargs):
        if threading.get_ident() != loop_thread and "FROM notes" in query and kwargs.get("connection") is not None:
            captured.append(kwargs["connection"])
            return_indexes.append(len(f.returns))
            entered.set()
            assert release.wait(timeout=5)
            try:
                return original(query, params, **kwargs)
            finally:
                finished.set()
        return original(query, params, **kwargs)

    monkeypatch.setattr(f.backend, "execute", execute)
    pool = f.backend.get_pool()
    original_return = pool.return_connection

    def observe_return(raw):
        original_return(raw)
        if captured and raw is captured[0]:
            returned.set()

    monkeypatch.setattr(pool, "return_connection", observe_return)

    async def run():
        task = asyncio.create_task(worker.handle_study_pack_job(job(f)))
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert captured and captured[0] not in f.returns[return_indexes[0] :]
        release.set()
        assert await asyncio.to_thread(finished.wait, 5)
        await drain()
        # Observe this exact loan return while the loop/executor remain alive.
        assert await asyncio.to_thread(returned.wait, 5)
        assert_returned(f)
        assert not prompts

    try:
        asyncio.run(run())
    finally:
        release.set()
