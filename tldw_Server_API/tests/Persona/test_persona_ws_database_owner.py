"""Persona's actual WS opener must retain the authenticated storage tenant."""

from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

import pytest

from tldw_Server_API.app.api.v1.endpoints import persona as persona_ep
from tldw_Server_API.app.core.DB_Management import content_backend
from tldw_Server_API.app.core.DB_Management.backends.base import AuthorizationDeniedError, BackendType
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, ConflictError
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Persona.session_manager import SessionManager

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("backend_kind", ["sqlite", "postgresql"])
@pytest.mark.parametrize("owner", ["1", "341"])
def test_actual_websocket_opener_preserves_owned_transcript_and_foreign_refusal(
    backend_kind, owner, request, tmp_path, monkeypatch
):
    """Use normal backend selection and the real opener, not its replacement."""
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "users"))
    monkeypatch.setenv("TLDW_CONTENT_DB_BACKEND", backend_kind)
    if backend_kind == "postgresql":
        cfg = request.getfixturevalue("pg_restricted_backend").config
        dsn = (
            "postgresql://"
            + quote(cfg.pg_user, safe="")
            + ":"
            + quote(cfg.pg_password, safe="")
            + "@"
            + cfg.pg_host
            + ":"
            + str(cfg.pg_port)
            + "/"
            + cfg.pg_database
        )
        monkeypatch.setenv("TLDW_CONTENT_PG_DSN", dsn)
    content_backend.clear_cached_backend()
    handles = []
    pools = {}

    def retain(db):
        if db is None:
            pytest.fail("The normal authenticated WS database could not open")
        handles.append(db)
        if db.backend_type == BackendType.POSTGRESQL:
            pools[id(db.backend)] = db.backend.get_pool()
        return db

    def discard_fixture_checkout(db):
        if db.backend_type == BackendType.POSTGRESQL:
            # Reset workers must not outlive this fixture's immediate pool shutdown.
            db._discard_legacy_checkout()

    try:
        seed = retain(CharactersRAGDB(DatabasePaths.get_chacha_db_path(int(owner)), client_id=owner))
        expected_backend = BackendType.POSTGRESQL if backend_kind == "postgresql" else BackendType.SQLITE
        assert seed.backend_type == expected_backend  # nosec B101 # Native backend selection is part of the regression.
        persona = seed.create_persona_profile({"user_id": owner, "name": "Owned WS guide"})
        session = seed.create_persona_session({"user_id": owner, "persona_id": persona})
        discard_fixture_checkout(seed)
        seed.close_connection()
        ws_db = retain(persona_ep._open_persona_ws_db(owner))
        assert ws_db.backend_type == expected_backend  # nosec B101 # The actual opener cannot silently select SQLite.
        turn = SessionManager().prepare_turn(
            role="user", content="ORBIT-746", metadata={"client_message_id": "owned-ws-turn"}
        )

        @chacha_operation(independent=True)
        def append(db, prepared):
            return db.persona_transcript_store.append_persona_transcript_turn(
                session_id=session, user_id=owner, persona_id=persona, turn=prepared
            )

        @chacha_operation(independent=True)
        def read(db):
            if db.backend_type == BackendType.POSTGRESQL:
                # Finish this test-owned SELECT before its pool's async reset workers retire.
                with db.transaction():
                    return db.persona_transcript_store.get_persona_transcript(session, user_id=owner)
            return db.persona_transcript_store.get_persona_transcript(session, user_id=owner)

        with ThreadPoolExecutor(max_workers=1) as executor:
            saved = executor.submit(append, ws_db, turn).result(timeout=15)
        assert saved["turns"] == [turn]  # nosec B101 # Real scoped write must commit the exact prepared turn.
        discard_fixture_checkout(ws_db)
        ws_db.close_all_connections()
        reopened = retain(persona_ep._open_persona_ws_db(owner))
        before = read(reopened)
        assert before["turns"] == [turn]  # nosec B101 # Reopening must retain IDs, text, timestamp and metadata.
        foreign = retain(persona_ep._open_persona_ws_db("2"))
        assert read(foreign) is None  # nosec B101 # Another real tenant must not read even with a forged owner argument.
        with pytest.raises((ConflictError, AuthorizationDeniedError)):
            append(foreign, SessionManager().prepare_turn(role="user", content="Forbidden replacement"))
        assert read(reopened) == before  # nosec B101 # The refused foreign write cannot change the owner's snapshot.
    finally:
        try:
            for db in reversed(handles):
                discard_fixture_checkout(db)
                db.close_all_connections()
        finally:
            content_backend.clear_cached_backend()
            for pool in pools.values():
                pool.close_all()
