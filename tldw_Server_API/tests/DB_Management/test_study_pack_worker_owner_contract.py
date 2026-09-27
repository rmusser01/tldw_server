"""StudyPack cold and warm factories preserve the canonical resource owner."""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from uuid import uuid4

import pytest
from cachetools import LRUCache

from tldw_Server_API.app.api.v1.API_Deps import ChaCha_Notes_DB_Deps as deps
from tldw_Server_API.app.core.DB_Management import ChaChaNotes_DB as db_module
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
from tldw_Server_API.app.core.DB_Management.chacha.runtime import ChaChaRuntimeManager
from tldw_Server_API.app.core.StudyPacks import generation_service
from tldw_Server_API.app.core.StudyPacks.source_resolver import StudySourceResolver
from tldw_Server_API.app.services import study_pack_jobs_worker as worker

EVIDENCE = "The citrine calibration is twenty-seven."
pytestmark = pytest.mark.integration


def _restricted_source_receipt(backend, db, note_id):
    """Exercise the real source reader with existing RLS under a restricted role."""
    role = backend.escape_identifier(f"study_pack_owner_{uuid4().hex[:12]}")
    created = False
    try:
        with backend.transaction() as conn:
            backend.execute(f"CREATE ROLE {role} NOLOGIN NOSUPERUSER NOBYPASSRLS", connection=conn)
            backend.execute(f"GRANT USAGE ON SCHEMA public TO {role}", connection=conn)
            backend.execute(f"GRANT SELECT ON notes TO {role}", connection=conn)
            backend.execute(f"GRANT SELECT, INSERT, UPDATE ON decks, flashcards TO {role}", connection=conn)
            backend.execute(f"GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO {role}", connection=conn)
            backend.execute(f"GRANT {role} TO CURRENT_USER", connection=conn)
        created = True
        with chacha_operation(independent=True), db.transaction() as conn:
            conn.execute(f"SET LOCAL ROLE {role}")
            flags = dict(
                conn.execute("SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname=current_user").fetchone()
            )
            scope = conn.execute("SELECT current_setting('app.current_user_id') AS owner").fetchone()["owner"]
            try:
                bundle = StudySourceResolver(db=db).resolve([{"source_type": "note", "source_id": note_id}])
            except ValueError as exc:
                if "not found" not in str(exc):
                    raise
                resolved = False
            else:
                assert bundle.items[0].evidence_text == EVIDENCE
                resolved = True
            deck_id = db.add_deck("Restricted-role generated output")
            card_id = db.add_flashcard({"deck_id": deck_id, "front": "Restricted front", "back": "Restricted back"})
            deck_owner = db.get_deck(deck_id)["client_id"]
            card_owner = db.get_flashcard(card_id)["client_id"]
        return {
            "roleFlags": flags,
            "scope": scope,
            "resolved": resolved,
            "deckOwner": deck_owner,
            "cardOwner": card_owner,
        }
    finally:
        if created:
            with backend.transaction() as conn:
                backend.execute(f"DROP OWNED BY {role}", connection=conn)
                backend.execute(f"DROP ROLE {role}", connection=conn)


@pytest.mark.parametrize("kind", ["postgresql", "sqlite"])
@pytest.mark.parametrize("warm", [False, True], ids=["cold-worker-first", "warm-owner-first"])
def test_worker_preserves_canonical_owner(request, tmp_path, monkeypatch, kind, warm):
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
    monkeypatch.setattr(worker, "get_media_db_for_owner", lambda _: SimpleNamespace())
    monkeypatch.setattr(
        generation_service, "_resolve_generation_provider_and_model", lambda *_: ("fixture", "fixture", {})
    )
    seed = db_module.CharactersRAGDB(tmp_path / "2" / "ChaChaNotes.db", client_id="2", backend=backend)
    with chacha_operation(independent=True):
        note_id = seed.add_note(title="Canonical owner's source", content=EVIDENCE)
    seed.close_connection()
    legacy = None
    legacy_deck = None
    if kind == "sqlite":
        legacy = db_module.CharactersRAGDB(tmp_path / "2" / "ChaChaNotes.db", client_id="study-pack-worker-2")
        legacy_deck = legacy.add_deck("Historical worker label")
        legacy.close_connection()

    async def model(self, *, system_prompt, user_prompt):
        assert EVIDENCE in user_prompt
        return json.dumps(
            {
                "cards": [
                    {
                        "front": "What is the calibration?",
                        "back": "Twenty-seven.",
                        "citations": [{"source_type": "note", "source_id": note_id, "citation_text": EVIDENCE}],
                    }
                ]
            }
        )

    monkeypatch.setattr(generation_service.StudyPackGenerationService, "_call_generation_model", model)

    async def drain():
        await asyncio.sleep(0)
        while deps._chacha_default_char_tasks:
            await asyncio.gather(*tuple(deps._chacha_default_char_tasks))

    async def run():
        if warm:
            owner_db = await deps.get_chacha_db_for_user_id(2)
            await drain()
            assert owner_db.client_id == "2"
        else:
            assert not cache
        result = await worker.handle_study_pack_job(
            {
                "id": 1,
                "owner_user_id": "2",
                "payload": {
                    "title": "Synthetic owner proof",
                    "source_items": [{"source_type": "note", "source_id": note_id}],
                },
            }
        )
        await drain()
        worker_db = cache[str(tmp_path / "2")]
        # The canonical owner loader is also real and must reuse the published cache.
        later_owner_db = await deps.get_chacha_db_for_owner(2)
        with chacha_operation(independent=True):
            stored_deck = seed.execute_query(
                "SELECT client_id FROM decks WHERE id=?", (result["deck_id"],), read_only=True
            ).fetchone()
            stored_cards = seed.execute_query(
                "SELECT client_id FROM flashcards WHERE deck_id=?", (result["deck_id"],), read_only=True
            ).fetchall()
            visible = seed.get_deck(result["deck_id"])
        receipt = {
            "backend": kind,
            "warm": warm,
            "workerClientId": worker_db.client_id,
            "laterOwnerClientId": later_owner_db.client_id,
            "sameCachedObject": later_owner_db is worker_db,
            "storedDeckClientId": stored_deck["client_id"],
            "storedCardClientIds": [row["client_id"] for row in stored_cards],
            "canonicalOwnerCanReadDeck": visible is not None,
        }
        if kind == "postgresql":
            receipt["restrictedSource"] = _restricted_source_receipt(backend, worker_db, note_id)
            assert receipt["restrictedSource"]["roleFlags"] == {"rolsuper": False, "rolbypassrls": False}
            assert receipt["restrictedSource"]["resolved"], receipt
            assert receipt["restrictedSource"]["deckOwner"] == "2", receipt
            assert receipt["restrictedSource"]["cardOwner"] == "2", receipt
        else:
            with chacha_operation(independent=True):
                historical = seed.get_deck(legacy_deck)
                assert historical["name"] == "Historical worker label"
                assert historical["client_id"] == "study-pack-worker-2"
        assert receipt["workerClientId"] == "2", receipt
        assert receipt["laterOwnerClientId"] == "2", receipt
        assert receipt["sameCachedObject"], receipt
        assert receipt["storedDeckClientId"] == "2", receipt
        assert receipt["storedCardClientIds"] == ["2"], receipt
        assert receipt["canonicalOwnerCanReadDeck"], receipt
        if kind == "postgresql":
            assert receipt["restrictedSource"]["scope"] == "2", receipt
            assert receipt["restrictedSource"]["resolved"], receipt

    try:
        asyncio.run(run())
    finally:
        executor.shutdown(wait=True)
        for db in tuple(cache.values()):
            db.close_all_connections()
        if legacy is not None:
            legacy.close_all_connections()
        seed.close_all_connections()
        if backend is not None:
            backend.get_pool().close_all()
