"""Exact Persona history survives runtime replacement without widening authority."""

import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from tldw_Server_API.app.core.DB_Management.chacha import persona_transcript_store as storage
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, ConflictError
from tldw_Server_API.app.core.Persona.session_manager import SessionManager

pytestmark = pytest.mark.integration


@pytest.fixture(params=["sqlite", "postgresql"])
def transcript_db(request, tmp_path):
    backend = None
    if request.param == "postgresql":
        backend = request.getfixturevalue("pg_restricted_backend")
    db = CharactersRAGDB(tmp_path / "transcript.db", client_id="1", owner_user_id="1", backend=backend)
    persona = db.create_persona_profile({"user_id": "1", "name": "Transcript owner"})
    session = db.create_persona_session({"user_id": "1", "persona_id": persona})
    try:
        yield db, persona, session
    finally:
        db.close_all_connections()


def save(fixture, turn, **fields):
    db, persona, session = fixture
    return db.persona_transcript_store.append_persona_transcript_turn(
        session_id=session,
        user_id="1",
        persona_id=persona,
        turn=turn,
        **fields,
    )


def test_exact_replay_is_idempotent_and_conflicting_identity_is_refused(transcript_db, monkeypatch):
    db, _, session = transcript_db
    turn = SessionManager().prepare_turn(role="user", content="ORBIT-744", metadata={"client_message_id": "stable"})
    first = save(transcript_db, turn)
    later = datetime.now(timezone.utc) + timedelta(hours=1)
    monkeypatch.setattr(storage.PersonaTranscriptStore, "_now", staticmethod(lambda: later))
    assert save(transcript_db, deepcopy(turn)) == first  # nosec B101 -- Exact replay cannot renew retention.
    assert db.persona_transcript_store.get_persona_transcript(session, user_id="1")["turns"] == [turn]  # nosec B101
    with pytest.raises(ConflictError, match="different content"):
        save(transcript_db, {**turn, "content": "Conflicting replacement"})


def test_read_expiry_does_not_renew_or_rewrite_payload_and_blocks_late_reply(transcript_db, monkeypatch):
    db, _, session = transcript_db
    manager = SessionManager()
    user = manager.prepare_turn(role="user", content="Held provider question")
    first = save(transcript_db, user)
    now = datetime.now(timezone.utc)
    monkeypatch.setattr(storage.PersonaTranscriptStore, "_now", staticmethod(lambda: now + timedelta(days=2)))
    expired = db.persona_transcript_store.get_persona_transcript(session, user_id="1")
    assert expired["turns"] == [] and expired["revision"] == first["revision"] and expired["retired"]  # nosec B101
    assert db.persona_transcript_store.list_persona_transcript_counts(user_id="1", session_ids=[session])[session]["turn_count"] == 0  # nosec B101
    with pytest.raises(ConflictError, match="originating"):
        save(transcript_db, manager.prepare_turn(role="assistant", content="Orphan"), parent_turn_id=user["turn_id"])
    monkeypatch.setattr(storage.PersonaTranscriptStore, "_now", staticmethod(lambda: now))
    assert db.persona_transcript_store.get_persona_transcript(session, user_id="1")["turns"] == [user]  # nosec B101 -- Expiry reads are read-only.


def test_owner_capacity_retirement_fences_runtime_and_late_assistant(transcript_db, monkeypatch):
    db, persona, session = transcript_db
    monkeypatch.setattr(storage, "MAX_SESSIONS", 1)
    manager = SessionManager()
    manager.create("1", persona, session)
    user = manager.prepare_turn(role="user", content="First question")
    old = save(transcript_db, user)
    assert manager.publish_transcript(user_id="1", snapshot=old)  # nosec B101
    second = db.create_persona_session({"user_id": "1", "persona_id": persona})
    newer = save((db, persona, second), manager.prepare_turn(role="user", content="Second question"))
    for invalidation in newer["invalidations"]:
        manager.publish_transcript(user_id="1", snapshot=invalidation)
    assert not manager.publish_transcript(user_id="1", snapshot=old)  # nosec B101
    assert manager.list_turns(session_id=session, user_id="1") == []  # nosec B101
    assert db.persona_transcript_store.get_persona_transcript(session, user_id="1")["turns"] == []  # nosec B101
    with pytest.raises(ConflictError, match="originating"):
        save(transcript_db, manager.prepare_turn(role="assistant", content="Late"), parent_turn_id=user["turn_id"])


@pytest.mark.parametrize("change", [{"status": "closed"}, {"status": "archived"}, {"deleted": True}])
def test_terminal_or_deleted_session_cannot_append(transcript_db, change):
    db, _, session = transcript_db
    db.update_persona_session(session_id=session, user_id="1", update_data=change)
    with pytest.raises(ConflictError, match="unavailable"):
        save(transcript_db, SessionManager().prepare_turn(role="user", content="Denied"))
    with pytest.raises(ConflictError, match="unavailable"):
        save(transcript_db, SessionManager().prepare_turn(role="user", content="Denied legacy"), allow_unpersisted=True)

    with pytest.raises(ConflictError, match="unavailable"):
        db.persona_transcript_store.validate_persona_transcript_session(
            session_id=session, user_id="1", persona_id=transcript_db[1], allow_unpersisted=True
        )


def test_foreign_owner_or_persona_cannot_read_or_append(transcript_db):
    db, persona, session = transcript_db
    turn = SessionManager().prepare_turn(role="user", content="Private")
    save(transcript_db, turn)
    assert db.persona_transcript_store.get_persona_transcript(session, user_id="2") is None  # nosec B101
    assert db.persona_transcript_store.list_persona_transcript_counts(user_id="2", session_ids=[session]) == {}  # nosec B101
    other = db.create_persona_profile({"user_id": "1", "name": "Other Persona"})
    for owner, bound in (("2", persona), ("1", other)):
        with pytest.raises(ConflictError, match="unavailable"):
            db.persona_transcript_store.append_persona_transcript_turn(session_id=session, user_id=owner, persona_id=bound, turn=turn)


def test_concurrent_appends_have_one_commit_order_and_no_lost_turns(transcript_db):
    db, _, session = transcript_db
    manager = SessionManager()
    turns = [manager.prepare_turn(role="user", content=str(index)) for index in range(6)]

    @chacha_operation(independent=True)
    def append(db, turn):
        return save(transcript_db, turn)

    with ThreadPoolExecutor(max_workers=3) as pool:
        snapshots = list(pool.map(lambda turn: append(db, turn), turns))
    saved = db.persona_transcript_store.get_persona_transcript(session, user_id="1")
    ordered = sorted(snapshots, key=lambda item: item["revision"])
    assert saved["turns"] == ordered[-1]["turns"]  # nosec B101
    assert [item["revision"] for item in ordered] == list(range(1, 7))  # nosec B101
    assert {item["turn_id"] for item in saved["turns"]} == {item["turn_id"] for item in turns}  # nosec B101


def test_worst_case_bounded_text_and_metadata_are_serializable(transcript_db):
    db, _, session = transcript_db
    manager = SessionManager()
    turn = manager.prepare_turn(role="user", content="\u0000" * 8192, metadata={"unicode": "😀" * 4096})
    save(transcript_db, turn)
    saved = db.persona_transcript_store.get_persona_transcript(session, user_id="1")
    assert saved["turns"] == [turn]  # nosec B101
    assert len(json.dumps(turn["metadata"], ensure_ascii=True, sort_keys=True)) <= storage.MAX_METADATA_CHARS  # nosec B101
    assert (
        len(json.dumps(saved["turns"], ensure_ascii=True, sort_keys=True).encode("ascii")) <= storage.MAX_SNAPSHOT_BYTES
    )  # nosec B101


def test_turn_bound_keeps_latest_exact_turns(transcript_db, monkeypatch):
    db, _, session = transcript_db
    monkeypatch.setattr(storage, "MAX_TURNS", 2)
    manager = SessionManager()
    turns = [manager.prepare_turn(role="user", content=str(index)) for index in range(3)]
    for turn in turns:
        save(transcript_db, turn)
    assert db.persona_transcript_store.get_persona_transcript(session, user_id="1")["turns"] == turns[-2:]  # nosec B101


def test_evicted_runtime_is_not_recreated_by_late_publication():
    manager = SessionManager(max_sessions_per_user=1)
    first = manager.create("1", "persona")
    manager.create("1", "persona")
    assert not manager.publish_transcript(
        user_id="1",
        snapshot={  # nosec B101
            "session_id": first.session_id,
            "persona_id": "persona",
            "revision": 1,
            "turns": [manager.prepare_turn(role="assistant", content="Late")],
        },
    )
    assert manager.get(first.session_id) is None  # nosec B101


def test_full_default_capacity_accepts_astral_content(transcript_db):
    db, persona, session = transcript_db
    manager = SessionManager()
    template = manager.prepare_turn(role="user", content="😀" * 8192, metadata={"value": "x" * 4080})
    turns = [{**template, "turn_id": str(index)} for index in range(200)]
    with db.transaction() as conn:
        query, params = db._prepare_backend_statement(
            "INSERT INTO persona_session_transcripts(session_id,user_id,persona_id,revision,last_activity,turn_count,turns_json) "
            "VALUES (?,?,?,?,?,?,?)",
            (
                session,
                "1",
                persona,
                199,
                datetime.now(timezone.utc).isoformat(),
                199,
                json.dumps(turns[:-1], ensure_ascii=True, sort_keys=True),
            ),
        )
        conn.execute(query, params)
    saved = save(transcript_db, turns[-1])
    assert saved["turn_count"] == 200  # nosec B101 -- Full public retention capacity with worst-case Unicode.
    assert db.persona_transcript_store.get_persona_transcript(session, user_id="1")["turns"] == turns  # nosec B101


def test_read_publication_preserves_activity_but_new_turn_refreshes_it():
    manager = SessionManager()
    session = manager.create("1", "persona")
    old = datetime.now(timezone.utc) - timedelta(hours=12)
    session.updated_at = old
    snapshot = {"session_id": session.session_id, "persona_id": "persona", "revision": 1, "turns": []}
    assert manager.publish_transcript(user_id="1", snapshot=snapshot)  # nosec B101
    assert session.updated_at == old  # nosec B101 -- Reading must not extend inactivity retention.
    assert manager.publish_transcript(user_id="1", snapshot={**snapshot, "revision": 2}, touch=True)  # nosec B101
    assert session.updated_at > old  # nosec B101 -- An actual committed turn refreshes the live session.


def test_reversed_publication_accepts_retained_turn_without_overwriting_newer_history():
    manager = SessionManager()
    session = manager.create("1", "persona")
    first = manager.prepare_turn(role="user", content="First")
    second = manager.prepare_turn(role="user", content="Second")
    old = {"session_id": session.session_id, "persona_id": "persona", "revision": 1, "turns": [first]}
    current = {**old, "revision": 2, "turns": [first, second]}
    assert manager.publish_transcript(user_id="1", snapshot=current)  # nosec B101
    assert manager.publish_transcript(user_id="1", snapshot=old, expected_turn=first)  # nosec B101
    assert manager.list_turns(session_id=session.session_id, user_id="1") == [first, second]  # nosec B101
    assert manager.publish_transcript(user_id="1", snapshot={**old, "revision": 3, "turns": []})  # nosec B101
    assert not manager.publish_transcript(user_id="1", snapshot=old, expected_turn=first)  # nosec B101
    assert manager.list_turns(session_id=session.session_id, user_id="1") == []  # nosec B101


def test_counts_are_limited_to_requested_owned_page(transcript_db):
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import InputError

    db, persona, session = transcript_db
    save(transcript_db, SessionManager().prepare_turn(role="user", content="Page one"))
    other = db.create_persona_session({"user_id": "1", "persona_id": persona})
    save((db, persona, other), SessionManager().prepare_turn(role="user", content="Page two"))
    assert set(db.persona_transcript_store.list_persona_transcript_counts(user_id="1", session_ids=[session])) == {session}  # nosec B101
    assert db.persona_transcript_store.list_persona_transcript_counts(user_id="1", session_ids=[]) == {}  # nosec B101
    with pytest.raises(InputError, match="200 sessions"):
        db.persona_transcript_store.list_persona_transcript_counts(user_id="1", session_ids=[str(index) for index in range(201)])


def test_expired_old_read_cannot_seal_a_new_committed_revision(transcript_db, monkeypatch):
    from threading import Event

    db, persona, session = transcript_db
    manager = SessionManager()
    manager.create("1", persona, session)
    now = [datetime.now(timezone.utc)]
    monkeypatch.setattr(storage.PersonaTranscriptStore, "_now", staticmethod(lambda: now[0]))
    first = save(transcript_db, manager.prepare_turn(role="user", content="First"))
    manager.publish_transcript(user_id="1", snapshot=first)
    now[0] += timedelta(seconds=storage.TTL_SECONDS - 1)
    before_insert, release = Event(), Event()
    query = storage.PersonaTranscriptStore._query

    def held_query(self, conn, sql, params):
        if sql.startswith("INSERT INTO persona_session_transcripts"):
            before_insert.set()
            assert release.wait(5)  # nosec B101 -- Hold the actual append after its nonexpired read.
        return query(self, conn, sql, params)

    monkeypatch.setattr(storage.PersonaTranscriptStore, "_query", held_query)
    second = manager.prepare_turn(role="user", content="Second")

    @chacha_operation(independent=True)
    def append(db):
        return save(transcript_db, second)

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(append, db)
            assert before_insert.wait(5)  # nosec B101
            now[0] += timedelta(seconds=2)
            old_read = db.persona_transcript_store.get_persona_transcript(session, user_id="1")
            assert old_read["turns"] == []  # nosec B101
            manager.publish_transcript(user_id="1", snapshot=old_read)
            assert not manager.publish_transcript(user_id="1", snapshot=first)  # nosec B101 -- Retired same-revision payload cannot return.
            release.set()
            committed = pending.result(timeout=10)
        assert manager.publish_transcript(user_id="1", snapshot=committed)  # nosec B101 -- A read cannot consume the next committed revision.
        assert [turn["content"] for turn in manager.list_turns(session_id=session, user_id="1")] == ["First", "Second"]  # nosec B101
    finally:
        release.set()


def test_explicit_resume_renews_only_nonexpired_history(transcript_db, monkeypatch):
    db, _, session = transcript_db
    now = [datetime.now(timezone.utc)]
    monkeypatch.setattr(storage.PersonaTranscriptStore, "_now", staticmethod(lambda: now[0]))
    turn = SessionManager().prepare_turn(role="user", content="Resume without regeneration")
    first = save(transcript_db, turn)
    now[0] += timedelta(hours=23)
    assert db.persona_transcript_store.get_persona_transcript(session, user_id="1")["revision"] == first["revision"]  # nosec B101 -- GET never renews.
    resumed = db.persona_transcript_store.get_persona_transcript(session, user_id="1", renew=True)
    assert resumed["revision"] > first["revision"] and resumed["turns"] == [turn]  # nosec B101
    now[0] += timedelta(hours=2)
    assert db.persona_transcript_store.get_persona_transcript(session, user_id="1")["turns"] == [turn]  # nosec B101 -- More than24h after the turn, within24h after explicit resume.
    now[0] += timedelta(days=2)
    expired = db.persona_transcript_store.get_persona_transcript(session, user_id="1", renew=True)
    assert expired["retired"] and expired["turns"] == []  # nosec B101
    assert expired["revision"] == resumed["revision"]  # nosec B101 -- Expired resume cannot extend or revive history.


def test_legacy_path_requires_true_absence_and_rejects_foreign_or_inactive_parent(transcript_db):
    db, persona, session = transcript_db
    turn = SessionManager().prepare_turn(role="user", content="Temporary scaffold")
    with pytest.raises(ConflictError, match="unavailable"):
        db.persona_transcript_store.append_persona_transcript_turn(session_id="absent", user_id="1", persona_id=persona, turn=turn)
    assert (
        db.persona_transcript_store.append_persona_transcript_turn(  # nosec B101 -- Explicit internal compatibility only for true absence.
            session_id="absent",
            user_id="1",
            persona_id=persona,
            turn=turn,
            allow_unpersisted=True,
        )
        is None
    )
    foreign = db.create_persona_profile({"user_id": "2", "name": "Other owner"})
    foreign_session = db.create_persona_session({"user_id": "2", "persona_id": foreign})
    with pytest.raises(ConflictError, match="unavailable"):
        db.persona_transcript_store.append_persona_transcript_turn(
            session_id=foreign_session, user_id="1", persona_id=persona, turn=turn, allow_unpersisted=True
        )
    db.update_persona_profile(persona_id=persona, user_id="1", update_data={"is_active": False})
    with pytest.raises(ConflictError, match="unavailable"):
        save(transcript_db, turn, allow_unpersisted=True)
    assert db.persona_transcript_store.get_persona_transcript(session, user_id="1") is None  # nosec B101 -- No false durable history.

    assert (
        db.persona_transcript_store.validate_persona_transcript_session(
            session_id="absent", user_id="1", persona_id=persona, allow_unpersisted=True
        )
        is False
    )  # nosec B101
    with pytest.raises(ConflictError, match="unavailable"):
        db.persona_transcript_store.validate_persona_transcript_session(
            session_id=foreign_session, user_id="1", persona_id=persona, allow_unpersisted=True
        )
    with pytest.raises(ConflictError, match="unavailable"):
        db.persona_transcript_store.validate_persona_transcript_session(
            session_id=session, user_id="1", persona_id=persona, allow_unpersisted=True
        )


def test_legacy_lookup_error_never_falls_back(transcript_db, monkeypatch):
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDBError

    query = storage.PersonaTranscriptStore._query

    def failed_lookup(self, conn, sql, params):
        if sql.startswith("SELECT id FROM persona_sessions"):
            raise CharactersRAGDBError("Injected parent lookup failure")
        return query(self, conn, sql, params)

    monkeypatch.setattr(storage.PersonaTranscriptStore, "_query", failed_lookup)
    db, persona, _ = transcript_db
    with pytest.raises(CharactersRAGDBError, match="Injected parent lookup"):
        db.persona_transcript_store.append_persona_transcript_turn(
            session_id="absent",
            user_id="1",
            persona_id=persona,
            turn=SessionManager().prepare_turn(role="user", content="Must fail"),
            allow_unpersisted=True,
        )

    with pytest.raises(CharactersRAGDBError, match="Injected parent lookup"):
        db.persona_transcript_store.validate_persona_transcript_session(
            session_id="absent", user_id="1", persona_id=persona, allow_unpersisted=True
        )


def test_saved_runtime_or_plan_cannot_downgrade_to_ephemeral():
    manager = SessionManager()
    session = manager.create("1", "persona", "saved", persisted=True)
    assert manager.create("1", "persona", "saved").persisted  # nosec B101
    with pytest.raises(ValueError, match="durable turn storage"):
        manager.append_turn(
            session_id=session.session_id,
            user_id="1",
            persona_id="persona",
            role="user",
            content="Not temporary",
            ephemeral_only=True,
        )
    manager.put_plan(
        session_id="reviewed",
        user_id="1",
        persona_id="persona",
        plan_id="plan",
        steps=[{"idx": 0, "tool": "rag_search", "args": {}}],
        requires_persisted_session=True,
    )
    with pytest.raises(ValueError, match="durable turn storage"):
        manager.append_turn(
            session_id="reviewed",
            user_id="1",
            persona_id="persona",
            role="tool",
            content="Not temporary",
            ephemeral_only=True,
        )


def test_saved_plan_cannot_become_legacy_after_parent_hard_deletion(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from tldw_Server_API.app.api.v1.endpoints import persona as endpoint
    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
    from tldw_Server_API.tests.Persona.test_persona_ws import _recv_until, _seed_persona_session, fastapi_app

    async def authenticated(*args, **kwargs):
        return "1", True, True

    monkeypatch.setattr(endpoint, "_resolve_authenticated_user_id", authenticated)
    session = "saved-parent-disappearance"
    _seed_persona_session(tmp_path, monkeypatch, user_id="1", session_id=session, mode="session_scoped")
    manager = SessionManager()
    monkeypatch.setattr(endpoint, "get_session_manager", lambda: manager)
    db = CharactersRAGDB(str(DatabasePaths.get_chacha_db_path(1)), client_id="parent-disappearance")
    try:
        with TestClient(fastapi_app) as client, client.websocket_connect("/api/v1/persona/stream") as ws:
            ws.send_json(
                {
                    "type": "user_message",
                    "session_id": session,
                    "text": "Search my notes",
                    "use_memory_context": False,
                    "use_companion_context": False,
                }
            )
            plan = _recv_until(ws, lambda event: event.get("event") == "tool_plan")
            assert manager.get(session).persisted  # nosec B101
            with db.transaction() as conn:
                conn.execute("DELETE FROM persona_sessions WHERE id=? AND user_id=?", (session, "1"))
            ws.send_json(
                {"type": "confirm_plan", "session_id": session, "plan_id": plan["plan_id"], "approved_steps": [0]}
            )
            outcome = _recv_until(
                ws, lambda event: event.get("event") == "tool_call" or event.get("reason_code") == "SESSION_NOT_FOUND"
            )
            assert outcome["reason_code"] == "SESSION_NOT_FOUND"  # nosec B101 -- No fresh legacy tool authority.
            ws.send_json(
                {
                    "type": "user_message",
                    "session_id": session,
                    "text": "Search my notes again",
                    "use_memory_context": False,
                    "use_companion_context": False,
                }
            )
            outcome = _recv_until(
                ws, lambda event: event.get("event") == "tool_plan" or event.get("reason_code") == "USER_TURN_FAILED"
            )
            assert outcome["reason_code"] == "USER_TURN_FAILED"  # nosec B101 -- Persisted identity cannot silently become temporary.
    finally:
        db.close_all_connections()


@pytest.fixture
def approval_runtime(tmp_path, monkeypatch):
    """Use an isolated real runtime and fail immediately if a refused tool dispatches."""
    from tldw_Server_API.app.api.v1.endpoints import persona as endpoint

    async def authenticated(*args, **kwargs):
        return "1", True, True

    def forbidden_dispatch():
        pytest.fail("A refused or retired Persona approval reached MCP dispatch")

    manager = SessionManager(max_sessions_per_user=1)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "users"))
    monkeypatch.setattr(endpoint, "_resolve_authenticated_user_id", authenticated)
    monkeypatch.setattr(endpoint, "get_session_manager", lambda: manager)
    monkeypatch.setattr(endpoint, "get_mcp_server", forbidden_dispatch)
    return manager


@pytest.mark.parametrize("parent_kind", ["foreign", "deleted"])
@pytest.mark.parametrize("action", ["confirm", "retry"])
def test_legacy_approval_rejects_a_later_canonical_parent(tmp_path, monkeypatch, parent_kind, action, approval_runtime):

    from fastapi.testclient import TestClient

    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
    from tldw_Server_API.tests.Persona.test_persona_ws import _recv_until, fastapi_app

    manager = approval_runtime
    session, plan = "temporary-approval", "reviewed-plan"
    manager.create("1", "research_assistant", session)
    manager.put_plan(
        session_id=session,
        user_id="1",
        persona_id="research_assistant",
        plan_id=plan,
        steps=[{"idx": 0, "tool": "notes.search", "args": {"query": "reviewed"}}],
    )
    retry_key = plan + "|0|notes.search"
    manager.update_preferences(
        session_id=session,
        user_id="1",
        preferences={
            "pending_retry_approvals": {
                retry_key: {
                    "plan_id": plan,
                    "step_idx": 0,
                    "tool": "notes.search",
                    "step_type": "mcp_tool",
                    "args": {"query": "reviewed"},
                },
            }
        },
    )
    db = CharactersRAGDB(str(DatabasePaths.get_chacha_db_path(1)), client_id="approval-parent-collision")
    owner = "2" if parent_kind == "foreign" else "1"
    persona = db.create_persona_profile({"user_id": owner, "name": "Later canonical parent"})
    db.create_persona_session(
        {"id": session, "user_id": owner, "persona_id": persona, "deleted": parent_kind == "deleted"}
    )
    try:
        with TestClient(fastapi_app) as client, client.websocket_connect("/api/v1/persona/stream") as ws:
            message = {"type": "confirm_plan", "session_id": session, "plan_id": plan, "approved_steps": [0]}
            if action == "retry":
                message = {
                    "type": "retry_tool_call",
                    "session_id": session,
                    "plan_id": plan,
                    "step_idx": 0,
                    "tool": "notes.search",
                }
            ws.send_json(message)
            outcome = _recv_until(
                ws, lambda event: event.get("event") == "tool_call" or event.get("reason_code") == "SESSION_NOT_FOUND"
            )
            assert outcome.get("reason_code") == "SESSION_NOT_FOUND"  # nosec B101 -- Deny before effect or approval consumption.
            if action == "confirm":
                assert manager.get_plan(session_id=session, plan_id=plan, user_id="1") is not None  # nosec B101
            else:
                assert retry_key in manager.get_preferences(session_id=session, user_id="1")["pending_retry_approvals"]  # nosec B101
    finally:
        db.close_all_connections()


@pytest.mark.parametrize("boundary", ["promote", "retire", "replace"])
def test_tool_preflight_rechecks_runtime_after_owned_database_await(tmp_path, monkeypatch, boundary, approval_runtime):
    from threading import Event

    from fastapi.testclient import TestClient

    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
    from tldw_Server_API.app.core.Persona.session_materialization import materialize_persona_session
    from tldw_Server_API.tests.Persona.test_persona_ws import _recv_until, fastapi_app

    manager = approval_runtime
    session, plan = "await-boundary", "reviewed-plan"
    manager.put_plan(
        session_id=session,
        user_id="1",
        persona_id="research_assistant",
        plan_id=plan,
        steps=[{"idx": 0, "tool": "notes.search", "args": {"query": "reviewed"}}],
    )
    original = storage.PersonaTranscriptStore.validate_persona_transcript_session
    waiting, release = Event(), Event()
    validations = []

    def held(self, **kwargs):
        result = original(self, **kwargs)
        validations.append(result)
        if len(validations) == 2:
            waiting.set()
            assert release.wait(5)  # nosec B101 -- Hold the per-step validation after the transaction completed.
        return result

    monkeypatch.setattr(storage.PersonaTranscriptStore, "validate_persona_transcript_session", held)
    try:
        with TestClient(fastapi_app) as client, client.websocket_connect("/api/v1/persona/stream") as ws:
            ws.send_json({"type": "confirm_plan", "session_id": session, "plan_id": plan, "approved_steps": [0]})
            assert waiting.wait(5)  # nosec B101
            if boundary == "promote":
                db = CharactersRAGDB(str(DatabasePaths.get_chacha_db_path(1)), client_id="preflight-promotion")
                try:
                    materialize_persona_session(
                        db,
                        session_manager=manager,
                        user_id="1",
                        persona_id="research_assistant",
                        resume_session_id=session,
                    )
                finally:
                    db.close_all_connections()
            else:
                manager.create("1", "research_assistant", "replacement")
                if boundary == "replace":
                    manager.create("1", "research_assistant", session)
            release.set()
            outcome = _recv_until(
                ws, lambda event: event.get("event") == "tool_call" or event.get("reason_code") == "SESSION_NOT_FOUND"
            )
            assert outcome.get("reason_code") == "SESSION_NOT_FOUND"  # nosec B101
    finally:
        release.set()
