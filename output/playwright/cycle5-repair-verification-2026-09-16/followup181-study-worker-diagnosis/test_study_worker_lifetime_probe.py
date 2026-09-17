"""Private causal probe of actual Study worker/accessor PostgreSQL cleanup."""

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from cachetools import LRUCache

from tldw_Server_API.app.api.v1.API_Deps import ChaCha_Notes_DB_Deps as deps
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
from tldw_Server_API.app.core.DB_Management.chacha.runtime import ChaChaRuntimeManager
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, ConflictError
from tldw_Server_API.app.core.StudyPacks import generation_service
from tldw_Server_API.app.core.StudySuggestions.jobs import STUDY_SUGGESTIONS_REFRESH_JOB_TYPE
from tldw_Server_API.app.services import study_pack_jobs_worker as pack_worker
from tldw_Server_API.app.services import study_suggestions_jobs_worker as suggestions_worker

pytestmark = pytest.mark.integration
PACKET = Path(__file__).parent
EVIDENCE = "Fixture source says the citrine calibration is twenty-seven."


@pytest.fixture
def worker_db(pg_database_config, tmp_path, monkeypatch):
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(tmp_path / "worker.db", client_id="1", backend=backend)
    note = db.add_note(title="Worker lifetime fixture", content=EVIDENCE)
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
    monkeypatch.setattr(pack_worker, "get_media_db_for_owner", lambda _: SimpleNamespace())
    monkeypatch.setattr(generation_service, "_resolve_generation_provider_and_model", lambda *_: ("fixture", "fixture", {}))

    pool = backend.get_pool()
    get_original = pool.get_connection
    return_original = pool.return_connection
    ledger = {}
    events = []
    reads = []
    cleanups = []
    acquisitions = []
    lock = threading.RLock()
    accessor = deps.get_chacha_db_for_user_id

    async def observe_accessor(*args, **kwargs):
        acquired = await accessor(*args, **kwargs)
        acquisitions.append({"database": id(acquired), "thread": threading.get_ident()})
        assert acquired is db
        return acquired

    monkeypatch.setattr(pack_worker, "get_chacha_db_for_user_id", observe_accessor)
    monkeypatch.setattr(suggestions_worker, "get_chacha_db_for_user_id", observe_accessor)

    def get_connection():
        raw = get_original()
        with lock:
            assert id(raw) not in ledger, "same raw connection checked out twice"
            ledger[id(raw)] = raw
            events.append({"event": "checkout", "connection": id(raw), "thread": threading.get_ident()})
        return raw

    def return_connection(raw):
        return_original(raw)
        with lock:
            events.append({"event": "return", "connection": id(raw), "thread": threading.get_ident()})
            ledger.pop(id(raw), None)

    monkeypatch.setattr(pool, "get_connection", get_connection)
    monkeypatch.setattr(pool, "return_connection", return_connection)
    read_note = db.get_note_by_id
    read_rollup = db.get_flashcard_review_session_rollup
    close_original = db.close_connection

    def observe_note(note_id):
        result = read_note(note_id)
        raw = db._get_thread_connection()
        reads.append({"kind": "note", "thread": threading.get_ident(), "connection": id(raw), "found": bool(result)})
        return result

    def observe_rollup(session_id):
        result = read_rollup(session_id)
        raw = db._get_thread_connection()
        reads.append({"kind": "rollup", "thread": threading.get_ident(), "connection": id(raw), "found": bool(result)})
        return result

    def observe_close():
        cleanups.append(threading.get_ident())
        return close_original()

    monkeypatch.setattr(db, "get_note_by_id", observe_note)
    monkeypatch.setattr(db, "get_flashcard_review_session_rollup", observe_rollup)
    monkeypatch.setattr(db, "close_connection", observe_close)

    f = SimpleNamespace(db=db, note=note, ledger=ledger, events=events, reads=reads, cleanups=cleanups, acquisitions=acquisitions)
    try:
        yield f
    finally:
        executor.shutdown(wait=True)
        db.close_all_connections()
        pool.close_all()


async def drain_maintenance():
    while deps._chacha_default_char_tasks:
        await asyncio.gather(*tuple(deps._chacha_default_char_tasks))


def receipt(f, name, loop_thread):
    result = {
        "case": name,
        "cached_database": id(f.db),
        "acquisitions": f.acquisitions,
        "captured_before_event_loop_shutdown": True,
        "loop_thread": loop_thread,
        "reads": f.reads,
        "close_threads": f.cleanups,
        "events": f.events,
        "outstanding": [
            {"connection": key, "status": raw.info.transaction_status.name, "closed": raw.closed}
            for key, raw in f.ledger.items()
        ],
    }
    (PACKET / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


@pytest.mark.parametrize("owned", [False, True], ids=["existing-finally", "private-owner-control"])
def test_real_study_pack_source_handoff_returns_every_checkout(worker_db, monkeypatch, owned):
    f = worker_db
    model_prompts = []

    async def model_failure(self, *, system_prompt, user_prompt):
        assert EVIDENCE in user_prompt
        model_prompts.append(user_prompt)
        raise RuntimeError("controlled model boundary failure")

    monkeypatch.setattr(generation_service.StudyPackGenerationService, "_call_generation_model", model_failure)

    async def run():
        loop_thread = threading.get_ident()
        with chacha_operation(independent=True) if owned else nullcontext():
            with pytest.raises(RuntimeError, match="controlled model boundary failure"):
                await pack_worker.handle_study_pack_job({
                    "id": 1,
                    "owner_user_id": "1",
                    "payload": {
                        "title": "Fixture only",
                        "source_items": [{"source_type": "note", "source_id": f.note}],
                    },
                })
            await drain_maintenance()
        return receipt(f, "pack-owned" if owned else "pack-existing", loop_thread)

    result = asyncio.run(run())
    loop_thread = result["loop_thread"]
    assert len(model_prompts) == 1
    assert len(f.reads) == 1 and f.reads[0]["found"]
    assert f.reads[0]["thread"] != loop_thread
    assert loop_thread in f.cleanups
    assert result["outstanding"] == []


def test_real_suggestions_same_thread_read_failure_returns_every_checkout(worker_db):
    f = worker_db

    async def run():
        loop_thread = threading.get_ident()
        with pytest.raises(ConflictError, match="Flashcard review session not found"):
            await suggestions_worker.handle_study_suggestions_job({
                "id": 2,
                "owner_user_id": "1",
                "job_type": STUDY_SUGGESTIONS_REFRESH_JOB_TYPE,
                "payload": {"anchor_type": "flashcard_review_session", "anchor_id": 999999},
            })
        await drain_maintenance()
        return receipt(f, "suggestions-existing", loop_thread)

    result = asyncio.run(run())
    loop_thread = result["loop_thread"]
    assert len(f.reads) == 1 and not f.reads[0]["found"]
    assert f.reads[0]["thread"] == loop_thread
    assert loop_thread in f.cleanups
    assert result["outstanding"] == []
