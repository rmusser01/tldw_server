"""Study Suggestions workers publish snapshots under the canonical cached owner."""

import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest
from cachetools import LRUCache

from tldw_Server_API.app.api.v1.API_Deps import ChaCha_Notes_DB_Deps as deps
from tldw_Server_API.app.core.DB_Management import ChaChaNotes_DB as db_module
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
from tldw_Server_API.app.core.DB_Management.chacha.runtime import ChaChaRuntimeManager
from tldw_Server_API.app.services import study_suggestions_jobs_worker as worker

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("kind", ["postgresql", "sqlite"])
@pytest.mark.parametrize("warm", [False, True], ids=["cold-worker-first", "warm-owner-first"])
def test_suggestions_worker_preserves_owner_and_authenticated_cache_reuse(request, tmp_path, monkeypatch, kind, warm):
    backend = (
        DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
        if kind == "postgresql"
        else None
    )
    monkeypatch.setattr(db_module, "get_content_backend", lambda _config: backend)
    monkeypatch.setattr(deps.DatabasePaths, "get_user_base_directory", staticmethod(lambda user: tmp_path / str(user)))
    monkeypatch.setattr(
        deps.DatabasePaths, "get_chacha_db_path", staticmethod(lambda user: tmp_path / str(user) / "ChaChaNotes.db")
    )
    monkeypatch.setattr(
        deps.DatabasePaths, "get_user_visual_identities_dir", staticmethod(lambda user: tmp_path / str(user) / "visual")
    )
    cache = LRUCache(maxsize=4)
    monkeypatch.setattr(deps, "_chacha_db_instances", cache)
    monkeypatch.setattr(deps, "_chacha_db_init_events", {})
    monkeypatch.setattr(deps, "_chacha_db_init_errors", {})
    monkeypatch.setattr(deps, "_CHACHA_RUNTIME", ChaChaRuntimeManager())
    monkeypatch.setattr(deps, "_CHACHA_SHUTTING_DOWN", False)
    monkeypatch.setattr(deps, "_chacha_default_char_tasks", set())
    monkeypatch.setattr(deps, "_chacha_default_char_futures", set())
    executor = ThreadPoolExecutor(max_workers=2)
    monkeypatch.setattr(deps, "_get_chacha_executor", lambda: executor)
    seed = db_module.CharactersRAGDB(tmp_path / "2" / "ChaChaNotes.db", client_id="2", backend=backend)
    with chacha_operation(independent=True):
        deck = seed.add_deck("Canonical owner's review")
        card = seed.add_flashcard({"deck_id": deck, "front": "UTC boundary?", "back": "Midnight UTC."})
        session = seed.get_or_create_flashcard_review_session(
            deck_id=deck, review_mode="due", tag_filter=None, scope_key=f"due:{deck}"
        )
        seed.review_flashcard(card, rating=3, review_session_id=session["id"])
        seed.mark_flashcard_review_session_completed(session["id"])
    seed.close_connection()

    async def drain():
        await asyncio.sleep(0)
        while deps._chacha_default_char_tasks:
            await asyncio.gather(*tuple(deps._chacha_default_char_tasks))

    async def run():
        if warm:
            await deps.get_chacha_db_for_user_id(2)
            await drain()
        else:
            assert not cache
        try:
            result = await worker.handle_study_suggestions_job(
                {
                    "id": 1,
                    "owner_user_id": "2",
                    "job_type": "study_suggestions_refresh",
                    "payload": {"anchor_type": "flashcard_review_session", "anchor_id": session["id"]},
                }
            )
        finally:
            await drain()
        cached = cache[str(tmp_path / "2")]
        later_owner = await deps.get_chacha_db_for_owner(2)
        with chacha_operation(independent=True):
            snapshot = later_owner.get_suggestion_snapshot(result["snapshot_id"])
            canonical_snapshot = seed.get_suggestion_snapshot(result["snapshot_id"])
        assert cached.client_id == "2"
        assert later_owner is cached
        assert snapshot["client_id"] == "2"
        assert canonical_snapshot["id"] == snapshot["id"]
        assert snapshot["anchor_id"] == session["id"]
        assert snapshot["payload_json"]["summary"]["total_count"] == 1

    try:
        asyncio.run(run())
    finally:
        executor.shutdown(wait=True)
        for db in tuple(cache.values()):
            db.close_all_connections()
        seed.close_all_connections()
        if backend is not None:
            backend.get_pool().close_all()
