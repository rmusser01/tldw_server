"""HTTP operation owners must release their own PostgreSQL transactions."""

import asyncio
import functools
import inspect
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from contextvars import ContextVar
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import anyio
import httpx
import pytest
from cachetools import LRUCache
from fastapi import Depends, FastAPI

from tldw_Server_API.app.api.v1.API_Deps import ChaCha_Notes_DB_Deps as deps
from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints import buddies, flashcards, notes
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.chacha.runtime import ChaChaRuntimeManager
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError

pytestmark = pytest.mark.integration

DUE_PATH = "/api/v1/flashcards/source-review-plans/due"
NOTES_PATH = "/api/v1/notes/"
_request_path = ContextVar("http_lifecycle_test_path", default=None)
_inside_endpoint = ContextVar("http_lifecycle_test_inside_endpoint", default=False)


@pytest.fixture
def pg_http(request, pg_database_config, tmp_path, monkeypatch):
    """Supply the official DB to the real cached dependency, without replacing it."""
    monkeypatch.setattr(CharactersRAGDB, "_NOTES_MOODBOARD_STUDIO_V61_POSTGRES_LOCK_TIMEOUT", "100ms")
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(tmp_path / "http.db", client_id="1", backend=backend)
    note = db.add_note(title="Committed title", content="HTTP lifetime fixture")
    db.sync_note_folders(note, ["Research"])
    db.add_character_card({"name": deps.DEFAULT_CHARACTER_NAME, "description": "Existing fixture"})
    populated = getattr(request, "param", False)
    if populated:
        db.create_source_review_plan(
            title="HTTP due plan",
            starts_on="2020-01-01",
            timezone_name="UTC",
            source_bundle_json={"items": [{"source_type": "note", "source_id": note, "label": "Fixture"}]},
            schedule=[
                {"offset_value": 1, "offset_unit": "day", "activity_type": "quiz", "due_at": "2020-01-02T00:00:00Z"}
            ],
        )
    db.close_connection()

    # Isolate process-level cache/settings, retaining real cache lookup, health
    # checking, runtime admission, default maintenance and the HTTP dependency.
    owner_dir = tmp_path / "owner"
    monkeypatch.setattr(deps.DatabasePaths, "get_user_base_directory", staticmethod(lambda _user_id: owner_dir))
    cache = LRUCache(maxsize=4)
    cache[str(owner_dir)] = db
    monkeypatch.setattr(deps, "_chacha_db_instances", cache)
    monkeypatch.setattr(deps, "_CHACHA_RUNTIME", ChaChaRuntimeManager())
    monkeypatch.setattr(deps, "_CHACHA_SHUTTING_DOWN", False)
    monkeypatch.setattr(deps, "_chacha_default_char_tasks", set())
    executor = ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(deps, "_get_chacha_executor", lambda: executor)

    observations = []
    original_connection = db._get_thread_connection

    def observe_connection():
        raw = original_connection()
        path = _request_path.get()
        if path is not None and _inside_endpoint.get():
            observations.append((path, threading.get_ident(), raw))
        return raw

    monkeypatch.setattr(db, "_get_thread_connection", observe_connection)
    app = FastAPI()
    app.include_router(flashcards.router, prefix="/api/v1")
    app.include_router(buddies.router, prefix="/api/v1/buddies")
    app.include_router(notes.router, prefix="/api/v1/notes")
    user = User(id=1, username="http-owner", email=None, is_active=True)
    limiter = SimpleNamespace(check_user_rate_limit=AsyncMock(return_value=(True, {})))
    app.dependency_overrides[deps.get_request_user] = lambda: user
    app.dependency_overrides[auth_deps.get_request_user] = lambda: user
    app.dependency_overrides[auth_deps.check_rate_limit] = lambda: None
    app.dependency_overrides[auth_deps.get_rate_limiter_dep] = lambda: limiter

    def override_rate_dependencies(dependant):
        for child in dependant.dependencies:
            if getattr(child.call, "_tldw_rate_limit_resource", None):
                app.dependency_overrides[child.call] = lambda: None
            override_rate_dependencies(child)

    for route in app.routes:
        if hasattr(route, "dependant"):
            override_rate_dependencies(route.dependant)

    # The dependency's compatibility check imports main.app, but no main app
    # startup runs. Ensure an unrelated test override cannot replace this seam.
    from tldw_Server_API.app.main import app as main_app

    monkeypatch.delitem(main_app.dependency_overrides, deps.get_chacha_db_for_user, raising=False)

    @app.middleware("http")
    async def identify_request(request, call_next):
        token = _request_path.set(request.url.path)
        try:
            return await call_next(request)
        finally:
            _request_path.reset(token)

    f = SimpleNamespace(
        app=app,
        db=db,
        backend=backend,
        config=pg_database_config,
        tmp_path=tmp_path,
        note=note,
        populated=populated,
        observations=observations,
    )
    try:
        yield f
    finally:
        executor.shutdown(wait=True)
        db.close_all_connections()
        backend.get_pool().close_all()


async def _run(f, operation):
    """Exercise actual async dependencies and one deterministic AnyIO worker."""
    # Observe endpoint work separately from the real dependency's health and
    # default-character workers, without replacing any dependency or handler.
    for route in f.app.routes:
        if not hasattr(route, "dependant"):
            continue
        original = route.dependant.call

        def instrument(handler):
            if inspect.iscoroutinefunction(handler):

                @functools.wraps(handler)
                async def async_body(*args, **kwargs):
                    token = _inside_endpoint.set(True)
                    try:
                        return await handler(*args, **kwargs)
                    finally:
                        _inside_endpoint.reset(token)

                return async_body

            @functools.wraps(handler)
            def sync_body(*args, **kwargs):
                token = _inside_endpoint.set(True)
                try:
                    return handler(*args, **kwargs)
                finally:
                    _inside_endpoint.reset(token)

            return sync_body

        route.dependant.call = instrument(original)
    limiter = anyio.to_thread.current_default_thread_limiter()
    previous = limiter.total_tokens
    limiter.total_tokens = 1
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=f.app, raise_app_exceptions=False), base_url="http://test"
        ) as client:
            await operation(client)
        if deps._chacha_default_char_tasks:
            await asyncio.gather(*tuple(deps._chacha_default_char_tasks))
    finally:
        limiter.total_tokens = previous


def _states(f, path=None):
    unique = {
        raw.info.backend_pid: raw for observed_path, _, raw in f.observations if path is None or observed_path == path
    }
    result = []
    for pid, raw in unique.items():
        rows = f.backend.execute(
            "SELECT DISTINCT c.relname FROM pg_locks l JOIN pg_class c ON c.oid=l.relation "
            "JOIN pg_namespace n ON n.oid=c.relnamespace WHERE l.pid=%s AND n.nspname='public' "
            "AND c.relkind IN ('r','p') ORDER BY c.relname",
            (pid,),
        ).rows
        result.append(
            {
                "pid": pid,
                "closed": raw.closed,
                "transaction": raw.info.transaction_status.name,
                "relations": [row["relname"] for row in rows],
            }
        )
    assert result, "The real request did not reach a database connection"
    return result


def _assert_finished(f, path=None):
    states = _states(f, path)
    assert all(row["transaction"] == "IDLE" and not row["relations"] for row in states), states


def _replacement(f):
    from psycopg.conninfo import make_conninfo

    config = f.config
    replacement_config = replace(
        config,
        connection_string=make_conninfo(
            host=config.pg_host,
            port=str(config.pg_port),
            dbname=config.pg_database,
            user=config.pg_user,
            password=config.pg_password,
            options="-c lock_timeout=100ms",
        ),
    )
    backend = DatabaseBackendFactory.create_backend(replacement_config)
    replacement = None
    try:
        replacement = CharactersRAGDB(f.tmp_path / "replacement.db", client_id="1", backend=backend)
        assert replacement.get_note_by_id(f.note)["content"] == "HTTP lifetime fixture"
    finally:
        if replacement is not None:
            replacement.close_all_connections()
        backend.get_pool().close_all()


@pytest.mark.parametrize("pg_http", [False, True], indirect=True, ids=["empty", "populated"])
def test_real_dependency_http_chain_allows_live_replacement(pg_http):
    f = pg_http

    async def operation(client):
        due = await client.get(DUE_PATH)
        assert due.status_code == 200
        assert due.json()["total"] == int(f.populated)
        assert (await client.get("/api/v1/buddies")).json() == {"buddies": []}
        attachment = await client.get("/api/v1/buddies/attachment")
        assert attachment.status_code == 200
        assert attachment.json()["attachment"] is None
        for path in [
            "/api/v1/notes/keywords/?include_note_counts=true",
            "/api/v1/notes/collections?include_keywords=true",
        ]:
            assert (await client.get(path)).status_code == 200
        notes_response = await client.get(NOTES_PATH)
        assert notes_response.status_code == 200
        assert notes_response.json()["notes"][0]["title"] == "Committed title"
        due_threads = {thread for path, thread, _ in f.observations if path == DUE_PATH}
        buddy_threads = {thread for path, thread, _ in f.observations if path == "/api/v1/buddies"}
        notes_threads = {thread for path, thread, _ in f.observations if path == NOTES_PATH}
        assert due_threads == buddy_threads
        assert due_threads.isdisjoint(notes_threads)
        states = _states(f)
        assert all(not row["closed"] for row in states)
        try:
            _replacement(f)
        except CharactersRAGDBError as exc:
            pytest.fail(f"Live HTTP replacement failed with {type(exc).__name__}; request connection states: {states}")
        _assert_finished(f)

    asyncio.run(_run(f, operation))


@pytest.mark.parametrize("path", [DUE_PATH, "/api/v1/buddies", "/api/v1/buddies/attachment", NOTES_PATH])
def test_completed_http_read_does_not_retain_a_transaction(pg_http, path):
    async def operation(client):
        assert (await client.get(path)).status_code == 200
        _assert_finished(pg_http, path)

    asyncio.run(_run(pg_http, operation))


@pytest.mark.parametrize("mode", ["sync-handoff", "joined-task", "failure", "cancel"])
def test_http_failure_handoff_and_child_work_release_their_owned_reads(pg_http, mode):
    f = pg_http
    entered = asyncio.Event()

    @f.app.get("/probe/read")
    async def read(db: CharactersRAGDB = Depends(deps.get_chacha_db_for_user)):
        def perform_read():
            rows, total = db.list_due_source_review_occurrences(now_utc="2026-09-17T00:00:00Z")
            assert rows == [] and total == 0

        if mode == "sync-handoff":
            await anyio.to_thread.run_sync(perform_read)
        elif mode == "joined-task":

            async def child():
                await asyncio.sleep(0)
                perform_read()

            await asyncio.create_task(child())
        else:
            perform_read()
        if mode == "failure":
            raise RuntimeError("Fixture route failure")
        if mode == "cancel":
            entered.set()
            await asyncio.Event().wait()
        return {"read": True}

    async def operation(client):
        if mode == "cancel":
            pending = asyncio.create_task(client.get("/probe/read"))
            await asyncio.wait_for(entered.wait(), timeout=5)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        else:
            response = await client.get("/probe/read")
            assert response.status_code == (500 if mode == "failure" else 200)
        _assert_finished(f, "/probe/read")

    asyncio.run(_run(f, operation))


@pytest.mark.parametrize("owner", ["implicit", "raw-begin", "chacha", "nested", "external-backend"])
@pytest.mark.parametrize("commit", [False, True], ids=["rollback", "commit"])
def test_http_caller_keeps_pending_write_until_explicit_decision(pg_http, owner, commit):
    f = pg_http

    @f.app.post("/probe/caller")
    async def caller(db: CharactersRAGDB = Depends(deps.get_chacha_db_for_user)):
        raw = db._get_thread_connection()
        with ExitStack() as stack:
            if owner == "raw-begin":
                db.get_connection().execute("BEGIN")
            elif owner == "external-backend":
                stack.enter_context(db.backend.transaction(connection=raw))
            elif owner in {"chacha", "nested"}:
                stack.enter_context(db.transaction())
                if owner == "nested":
                    stack.enter_context(db.transaction())
            db.execute_query("UPDATE notes SET title=? WHERE id=?", ("Pending title", f.note))
            # A nested non-HTTP accessor must keep this caller's owner/connection.
            nested = await deps.get_chacha_db_for_user_id(1)
            assert nested.get_note_by_id(f.note)["title"] == "Pending title"
            assert raw.info.transaction_status.name == "INTRANS"
            assert f.backend.execute("SELECT title FROM notes WHERE id=%s", (f.note,)).scalar == "Committed title"
            if commit:
                raw.commit()
            else:
                raw.rollback()
        return {"decided": True}

    async def operation(client):
        response = await client.post("/probe/caller")
        assert response.status_code == 200
        expected = "Pending title" if commit else "Committed title"
        assert f.backend.execute("SELECT title FROM notes WHERE id=%s", (f.note,)).scalar == expected
        _assert_finished(f, "/probe/caller")

    asyncio.run(_run(f, operation))


@pytest.mark.parametrize("settle_check", [False, True], ids=["no-auto-commit-control", "owned-teardown"])
def test_http_success_does_not_commit_an_undecided_write(pg_http, settle_check):
    f = pg_http

    @f.app.post("/probe/undecided")
    async def undecided(db: CharactersRAGDB = Depends(deps.get_chacha_db_for_user)):
        db.execute_query("UPDATE notes SET title=? WHERE id=?", ("Pending title", f.note))
        assert db.get_note_by_id(f.note)["title"] == "Pending title"
        return {"accepted": True}

    async def operation(client):
        assert (await client.post("/probe/undecided")).status_code == 200
        assert f.backend.execute("SELECT title FROM notes WHERE id=%s", (f.note,)).scalar == "Committed title"
        if settle_check:
            _assert_finished(f, "/probe/undecided")

    asyncio.run(_run(f, operation))


def test_another_http_owner_cannot_observe_or_settle_pending_work(pg_http):
    f = pg_http
    entered = asyncio.Event()
    release = asyncio.Event()

    @f.app.post("/probe/held-write")
    async def held_write(db: CharactersRAGDB = Depends(deps.get_chacha_db_for_user)):
        raw = db._get_thread_connection()
        db.execute_query("UPDATE notes SET title=? WHERE id=?", ("Pending title", f.note))
        entered.set()
        try:
            await release.wait()
            assert not raw.closed
            assert raw.info.transaction_status.name == "INTRANS"
            assert db.get_note_by_id(f.note)["title"] == "Pending title"
            return {"pending_preserved": True}
        finally:
            raw.rollback()

    async def operation(client):
        pending = asyncio.create_task(client.post("/probe/held-write"))
        await asyncio.wait_for(entered.wait(), timeout=5)
        try:
            other = await client.get(NOTES_PATH)
            assert other.status_code == 200
            title = other.json()["notes"][0]["title"]
        finally:
            release.set()
            first = await pending
        assert first.status_code == 200
        assert first.json() == {"pending_preserved": True}
        assert title == "Committed title", "A different HTTP request observed another owner's uncommitted write"

    asyncio.run(_run(f, operation))
