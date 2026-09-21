"""StudyPack writes must work in the actual shared Media/ChaCha content schema."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.StudyPacks import generation_service
from tldw_Server_API.app.services import study_pack_jobs_worker as worker

pytestmark = pytest.mark.integration
EVIDENCE = "The synthetic citrine calibration is twenty-seven."


@pytest.mark.parametrize("kind", ["postgresql", "sqlite"])
@pytest.mark.parametrize("media_first", [False, True], ids=["chacha-only", "real-media-first"])
def test_actual_worker_persists_pack_in_each_content_schema(request, tmp_path, monkeypatch, kind, media_first):
    backend = (
        DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
        if kind == "postgresql"
        else None
    )
    media = (
        MediaDatabase(str(tmp_path / "media.db"), client_id="2", backend=backend)
        if media_first is True
        else SimpleNamespace()
    )
    db = CharactersRAGDB(tmp_path / "notes.db", client_id="2", backend=backend)
    try:
        with chacha_operation(independent=True):
            note_id = db.add_note(title="Owned synthetic source", content=EVIDENCE)

        async def acquire(owner):
            assert owner == "2"
            return db, media

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

        monkeypatch.setattr(worker, "_get_databases_for_user", acquire)
        monkeypatch.setattr(
            generation_service, "_resolve_generation_provider_and_model", lambda *_: ("fixture", "fixture", {})
        )
        monkeypatch.setattr(generation_service.StudyPackGenerationService, "_call_generation_model", model)

        if kind == "postgresql":
            columns = {column["name"] for column in backend.get_table_info("sync_log")}
            assert ("entity_uuid" if media_first is True else "entity_id") in columns
        result = asyncio.run(
            worker.handle_study_pack_job(
                {
                    "id": 2,
                    "owner_user_id": "2",
                    "job_type": "study_pack_generate",
                    "payload": {
                        "title": "Synthetic Study Pack",
                        "deck_mode": "new",
                        "source_items": [{"source_type": "note", "source_id": note_id}],
                    },
                }
            )
        )
        with chacha_operation(independent=True):
            pack = db.get_study_pack(result["pack_id"])
            assert pack["client_id"] == "2"
            assert pack["workspace_id"] is None
            assert pack["deck_id"] == result["deck_id"]
            assert db.get_deck(result["deck_id"])["client_id"] == "2"
            memberships = db.list_study_pack_cards(result["pack_id"])
            assert len(memberships) == 1
            card = db.get_flashcard(memberships[0]["flashcard_uuid"])
            assert card["back"] == "Twenty-seven."
            assert card["client_id"] == "2"
            citations = db.list_flashcard_citations(card["uuid"])
            assert citations[0]["source_id"] == note_id
            assert citations[0]["client_id"] == "2"
            rows = [
                dict(row)
                for row in db.execute_query(
                    "SELECT * FROM sync_log WHERE entity IN (?, ?, ?) ORDER BY change_id",
                    ("study_packs", "study_pack_cards", "flashcard_citations"),
                    read_only=True,
                ).fetchall()
            ]
            assert [row["entity"] for row in rows] == ["study_packs", "study_pack_cards", "flashcard_citations"]
            for row in rows:
                payload = json.loads(row["payload"])
                assert row["operation"] == "create"
                assert row["client_id"] == payload["client_id"] == "2"
                assert row.get("entity_id", row.get("entity_uuid")) == str(payload["id"])
                assert payload["version"] == 1
            assert json.loads(rows[0]["payload"])["deck_id"] == result["deck_id"]
            assert json.loads(rows[1]["payload"])["study_pack_id"] == result["pack_id"]
            assert json.loads(rows[2]["payload"])["citation_text"] == EVIDENCE
    finally:
        db.close_all_connections()
        if media_first:
            media.close_connection()
        if backend is not None:
            backend.get_pool().close_all()


@pytest.fixture(params=["postgresql", "sqlite"])
def shared_store(request, tmp_path):
    backend = (
        DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
        if request.param == "postgresql"
        else None
    )
    media = MediaDatabase(str(tmp_path / "media.db"), client_id="2", backend=backend)
    db = CharactersRAGDB(tmp_path / "notes.db", client_id="2", backend=backend)
    try:
        yield db
    finally:
        db.close_all_connections()
        media.close_connection()
        if backend is not None:
            backend.get_pool().close_all()


def _create_graph(db):
    deck = db.add_deck("Synthetic graph")
    card = db.add_flashcard({"deck_id": deck, "front": "Synthetic front", "back": "Synthetic back"})
    pack = db.create_study_pack(
        title="Synthetic pack",
        workspace_id=None,
        deck_id=deck,
        source_bundle_json={"items": [{"source_type": "note", "source_id": "synthetic"}]},
        generation_options_json={"deck_mode": "new"},
    )
    assert db.add_study_pack_cards(pack, [card]) == 1
    assert (
        db.add_flashcard_citations(
            card,
            [
                {
                    "source_type": "note",
                    "source_id": "synthetic",
                    "citation_text": "Synthetic",
                }
            ],
        )
        == 1
    )
    return pack, card


def _create_suggestions(db):
    """Persist a snapshot and action reservation through the real database API."""
    snapshot = db.create_suggestion_snapshot(
        service="flashcards",
        activity_type="flashcard_review_session",
        anchor_type="flashcard_review_session",
        anchor_id=1,
        suggestion_type="study_suggestions",
        payload_json={"topics": []},
    )
    link = {
        "snapshot_id": snapshot,
        "target_service": "flashcards",
        "target_type": "deck",
        "selection_fingerprint": "shared",
    }
    db.create_suggestion_generation_link(**link, target_id="pending:shared")
    assert db.get_suggestion_snapshot(snapshot)["client_id"] == "2"
    assert db.find_suggestion_generation_link_by_fingerprint(**link)["target_id"] == "pending:shared"


def test_shared_sync_writes_remain_in_caller_rollback(shared_store):
    db = shared_store
    with chacha_operation(independent=True):
        note = db.add_note(title="Committed source", content="Keep this committed source")
        with pytest.raises(RuntimeError, match="caller rollback"):
            with db.transaction():
                _create_graph(db)
                _create_suggestions(db)
                raise RuntimeError("caller rollback")
        counts = db.execute_query(
            """SELECT
            (SELECT COUNT(*) FROM decks) AS decks,
            (SELECT COUNT(*) FROM flashcards) AS cards,
            (SELECT COUNT(*) FROM study_packs) AS packs,
            (SELECT COUNT(*) FROM study_pack_cards) AS memberships,
            (SELECT COUNT(*) FROM flashcard_citations) AS citations,
            (SELECT COUNT(*) FROM suggestion_snapshots) AS snapshots,
            (SELECT COUNT(*) FROM suggestion_generation_links) AS links,
            (SELECT COUNT(*) FROM sync_log WHERE entity IN
                ('study_packs', 'study_pack_cards', 'flashcard_citations',
                 'suggestion_snapshots', 'suggestion_generation_links')) AS sync_rows
        """,
            read_only=True,
        ).fetchone()
        assert dict(counts) == dict.fromkeys(
            ("decks", "cards", "packs", "memberships", "citations", "snapshots", "links", "sync_rows"), 0
        )
        assert db.get_note_by_id(note)["content"] == "Keep this committed source"


def test_all_three_triggers_preserve_update_delete_payloads(shared_store):
    db = shared_store
    with chacha_operation(independent=True):
        pack, card = _create_graph(db)
        membership = db.list_study_pack_cards(pack)[0]["id"]
        citation = db.list_flashcard_citations(card)[0]["id"]
        with db.transaction() as conn:
            conn.execute("UPDATE study_packs SET title=?, version=version+1 WHERE id=?", ("Updated pack", pack))
            conn.execute("UPDATE study_pack_cards SET version=version+1 WHERE id=?", (membership,))
            conn.execute(
                "UPDATE flashcard_citations SET citation_text=?, version=version+1 WHERE id=?",
                ("Updated citation", citation),
            )
        assert db.soft_delete_study_pack(pack, expected_version=2)
        assert db.replace_flashcard_citations(card, []) == 0
        with db.transaction() as conn:
            conn.execute("UPDATE study_pack_cards SET deleted=?, version=version+1 WHERE id=?", (True, membership))
        rows = [
            dict(row)
            for row in db.execute_query(
                "SELECT * FROM sync_log WHERE entity IN (?, ?, ?) ORDER BY change_id",
                ("study_packs", "study_pack_cards", "flashcard_citations"),
                read_only=True,
            ).fetchall()
        ]
        assert len(rows) == 9
        for entity, entity_id in (
            ("study_packs", pack),
            ("study_pack_cards", membership),
            ("flashcard_citations", citation),
        ):
            own_rows = [row for row in rows if row["entity"] == entity]
            assert [row["operation"] for row in own_rows] == ["create", "update", "delete"]
            assert [row["version"] for row in own_rows] == [1, 2, 3]
            for row in own_rows:
                payload = json.loads(row["payload"])
                assert row.get("entity_id", row.get("entity_uuid")) == str(entity_id)
                assert row["client_id"] == payload["client_id"] == "2"
                assert payload["id"] == entity_id
            assert bool(json.loads(own_rows[-1]["payload"])["deleted"])


@pytest.mark.parametrize("fail_once", [False, True], ids=["upgrade", "rollback-and-retry"])
def test_upgrade_replaces_existing_v71_trigger_bodies(pg_database_config, tmp_path, monkeypatch, fail_once):
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    media = MediaDatabase(str(tmp_path / "media.db"), client_id="2", backend=backend)
    db = CharactersRAGDB(tmp_path / "notes.db", client_id="2", backend=backend)
    try:
        columns = {column["name"] for column in backend.get_table_info("sync_log")}
        assert "entity_uuid" in columns and "entity_id" not in columns
        with backend.transaction() as conn:
            for name in (
                "study_packs_sync_log_fn",
                "study_pack_cards_sync_log_fn",
                "flashcard_citations_sync_log_fn",
                "suggestion_snapshots_sync_log_fn",
                "suggestion_generation_links_sync_log_fn",
            ):
                definition = backend.execute(
                    "SELECT pg_get_functiondef(oid) AS definition FROM pg_proc "
                    "WHERE proname=%s AND pronamespace='public'::regnamespace",
                    (name,),
                    connection=conn,
                ).rows[0]["definition"]
                # Reproduce the old installed bodies without changing schema version/data.
                backend.execute(
                    definition.replace("sync_log(entity, entity_uuid,", "sync_log(entity, entity_id,"),
                    connection=conn,
                )
            backend.execute(
                "UPDATE db_schema_version SET version=71 WHERE schema_name=%s",
                (db._SCHEMA_NAME,),
                connection=conn,
            )
        db.close_connection()
        if fail_once:
            original_execute = backend.execute

            def fail_final_trigger(query, *args, **kwargs):
                if "CREATE TRIGGER suggestion_generation_links_sync_log" in query:
                    raise DatabaseError("planned trigger upgrade failure")
                return original_execute(query, *args, **kwargs)

            with monkeypatch.context() as patch:
                patch.setattr(backend, "execute", fail_final_trigger)
                with pytest.raises(CharactersRAGDBError, match="planned trigger upgrade failure"):
                    CharactersRAGDB(tmp_path / "failed.db", client_id="2", backend=backend)
            assert (
                backend.execute("SELECT version FROM db_schema_version WHERE schema_name=%s", (db._SCHEMA_NAME,)).scalar
                == 71
            )
        db = CharactersRAGDB(tmp_path / "notes.db", client_id="2", backend=backend)
        assert (
            backend.execute("SELECT version FROM db_schema_version WHERE schema_name=%s", (db._SCHEMA_NAME,)).scalar
            == 72
        )
        with chacha_operation(independent=True):
            pack, card = _create_graph(db)
            assert db.get_study_pack(pack)["client_id"] == "2"
            assert db.list_flashcard_citations(card)[0]["client_id"] == "2"
            _create_suggestions(db)
    finally:
        db.close_all_connections()
        media.close_connection()
        backend.get_pool().close_all()
