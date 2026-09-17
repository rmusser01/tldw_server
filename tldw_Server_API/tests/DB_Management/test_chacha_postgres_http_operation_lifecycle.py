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
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import ChaChaOperationMiddleware
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
    app.add_middleware(ChaChaOperationMiddleware)
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


@pytest.mark.parametrize("independent_child", [False, True], ids=["closed-owner-rejected", "explicit-new-owner"])
def test_detached_task_must_not_reuse_finished_http_owner(pg_http, independent_child):
    from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import (
        ClosedChaChaOperationError,
        chacha_operation,
    )

    f = pg_http
    release = asyncio.Event()
    children = []
    observed = []

    @f.app.get("/probe/detached")
    async def detach(db: CharactersRAGDB = Depends(deps.get_chacha_db_for_user)):
        db.list_due_source_review_occurrences(now_utc="2026-09-17T00:00:00Z")

        async def child():
            await release.wait()
            if independent_child:
                with chacha_operation(independent=True):
                    observed.append(db._get_thread_connection())
                    db.list_due_source_review_occurrences(now_utc="2026-09-17T00:00:00Z")
            else:
                with pytest.raises(ClosedChaChaOperationError):
                    db.get_note_by_id(f.note)

        children.append(asyncio.create_task(child()))
        return {"scheduled": True}

    async def operation(client):
        assert (await client.get("/probe/detached")).status_code == 200
        release.set()
        await asyncio.gather(*children)
        _assert_finished(f, "/probe/detached")
        assert all(raw.info.transaction_status.name == "IDLE" for raw in observed)

    asyncio.run(_run(f, operation))


@pytest.mark.parametrize("bind", [False, True], ids=["legacy-not-adopted", "explicit-external-binding"])
@pytest.mark.parametrize("commit", [False, True], ids=["rollback", "commit"])
def test_http_scope_preserves_preexisting_external_write(pg_http, bind, commit):
    from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import ExternalConnection, chacha_operation

    f = pg_http
    raw = f.db._get_thread_connection()
    f.db.execute_query("UPDATE notes SET title=? WHERE id=?", ("External pending", f.note))
    assert raw.info.transaction_status.name == "INTRANS"

    @f.app.get("/probe/external")
    async def external(db: CharactersRAGDB = Depends(deps.get_chacha_db_for_user)):
        if bind:
            with chacha_operation(independent=True, bindings=(ExternalConnection(db, raw, f.backend),)):
                with db.transaction():
                    assert db._get_thread_connection() is raw
                    title = db.get_note_by_id(f.note)["title"]
                db.close_connection()  # A borrowed checkout cannot be returned by this helper.
                assert not raw.closed and raw.info.transaction_status.name == "INTRANS"
        else:
            assert db._get_thread_connection() is not raw
            title = db.get_note_by_id(f.note)["title"]
        return {"title": title}

    async def operation(client):
        response = await client.get("/probe/external")
        assert response.status_code == 200
        assert response.json()["title"] == ("External pending" if bind else "Committed title")
        assert not raw.closed and raw.info.transaction_status.name == "INTRANS"
        assert f.backend.execute("SELECT title FROM notes WHERE id=%s", (f.note,)).scalar == "Committed title"
        (raw.commit if commit else raw.rollback)()
        assert f.backend.execute("SELECT title FROM notes WHERE id=%s", (f.note,)).scalar == (
            "External pending" if commit else "Committed title"
        )

    try:
        asyncio.run(_run(f, operation))
    finally:
        raw.rollback()
        f.db.close_connection()


@pytest.mark.parametrize("failure", [False, True], ids=["complete", "background-failure"])
def test_http_owner_lives_through_stream_and_background_work(pg_http, failure):
    from starlette.background import BackgroundTask
    from starlette.responses import StreamingResponse

    f = pg_http
    raw_connections = []
    phases = []

    @f.app.get("/probe/stream")
    async def stream(db: CharactersRAGDB = Depends(deps.get_chacha_db_for_user)):
        def read(phase):
            raw = db._get_thread_connection()
            raw_connections.append(raw)
            db.list_due_source_review_occurrences(now_utc="2026-09-17T00:00:00Z")
            assert raw.info.transaction_status.name == "INTRANS"
            phases.append(phase)

        async def body():
            read("stream-start")
            yield b"first "
            await asyncio.sleep(0)
            read("stream-end")
            yield b"last"

        def background():
            read("background")
            if failure:
                raise RuntimeError("Expected background failure")

        return StreamingResponse(body(), background=BackgroundTask(background))

    async def operation(client):
        response = await client.get("/probe/stream")
        assert response.status_code == 200 and response.text == "first last"
        assert phases == ["stream-start", "stream-end", "background"]
        assert len({id(raw) for raw in raw_connections}) == 1
        assert raw_connections[0].info.transaction_status.name == "IDLE"

    asyncio.run(_run(f, operation))


def test_two_account_databases_have_separate_checkouts_and_visible_notes(pg_http, monkeypatch):
    from fastapi import Request

    f = pg_http
    other = CharactersRAGDB(f.tmp_path / "second-owner.db", client_id="2", backend=f.backend)
    other_note = other.add_note(title="Other owner", content="Private second owner fixture")
    other.close_connection()
    owner_dirs = {1: f.tmp_path / "owner", 2: f.tmp_path / "other-owner"}
    monkeypatch.setattr(
        deps.DatabasePaths, "get_user_base_directory", staticmethod(lambda user_id: owner_dirs[int(user_id)])
    )
    deps._chacha_db_instances[str(owner_dirs[2])] = other
    raw_by_user = {}

    async def identity(request: Request):
        return User(id=int(request.headers.get("X-Fixture-User", "1")), username="fixture", email=None, is_active=True)

    f.app.dependency_overrides[deps.get_request_user] = identity
    f.app.dependency_overrides[auth_deps.get_request_user] = identity
    entered = asyncio.Event()
    release = asyncio.Event()

    @f.app.get("/probe/account")
    async def account(db: CharactersRAGDB = Depends(deps.get_chacha_db_for_user)):
        owner = int(db.client_id)
        raw_by_user[owner] = db._get_thread_connection()
        selected = f.note if owner == 1 else other_note
        note = db.get_note_by_id(selected)
        if owner == 1:
            db.execute_query("UPDATE notes SET title=? WHERE id=?", ("Private pending", f.note))
            entered.set()
            await release.wait()
            assert db.get_note_by_id(f.note)["title"] == "Private pending"
        return {"title": note["title"]}

    async def operation(client):
        first = asyncio.create_task(client.get("/probe/account", headers={"X-Fixture-User": "1"}))
        await asyncio.wait_for(entered.wait(), timeout=5)
        try:
            second = await client.get("/probe/account", headers={"X-Fixture-User": "2"})
            assert second.status_code == 200 and second.json() == {"title": "Other owner"}
            assert raw_by_user[1] is not raw_by_user[2]
            assert raw_by_user[1].info.transaction_status.name == "INTRANS"
        finally:
            release.set()
            completed = await first
        assert completed.status_code == 200
        assert all(raw.info.transaction_status.name == "IDLE" for raw in raw_by_user.values())
        assert f.backend.execute("SELECT title FROM notes WHERE id=%s", (f.note,)).scalar == "Committed title"

    asyncio.run(_run(f, operation))


def test_request_can_finalize_repeatedly_without_reusing_closed_owner(pg_http):
    from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import (
        ClosedChaChaOperationError,
        chacha_operation,
    )

    f = pg_http

    @f.app.get("/probe/finish")
    async def finish(db: CharactersRAGDB = Depends(deps.get_chacha_db_for_user)):
        with chacha_operation() as owner:
            db.list_due_source_review_occurrences(now_utc="2026-09-17T00:00:00Z")
            raw = db._get_thread_connection()
            owner.close()
            owner.close()
            assert raw.info.transaction_status.name == "IDLE"
            with pytest.raises(ClosedChaChaOperationError):
                db.get_note_by_id(f.note)
        return {"finished": True}

    async def operation(client):
        assert (await client.get("/probe/finish")).json() == {"finished": True}
        assert (await client.get(NOTES_PATH)).status_code == 200
        _assert_finished(f)

    asyncio.run(_run(f, operation))


def test_real_initializer_and_default_worker_finish_independent_scopes(pg_http, monkeypatch):
    from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import current_connection_state

    f = pg_http
    created = []
    initialization_states = []
    raw_connections = []

    def construct(*args, **kwargs):
        db = CharactersRAGDB(*args, **kwargs, backend=f.backend)
        initialization_states.append(current_connection_state(db))
        original = db._get_thread_connection

        def observe():
            raw = original()
            raw_connections.append(raw)
            return raw

        monkeypatch.setattr(db, "_get_thread_connection", observe)
        created.append(db)
        return db

    monkeypatch.setattr(deps, "CharactersRAGDB", construct)
    monkeypatch.setattr(deps, "_get_chacha_db_path_for_user", lambda _: f.tmp_path / "published.db")
    published = deps._create_and_prepare_db(3, "3")
    assert published is created[0]
    assert initialization_states[0] is not None
    assert initialization_states[0].conn is None
    # PostgreSQL intentionally skips the SQLite-only bundled visual seed.
    # Exercise the real default-character worker, which does use a checkout.
    asyncio.run(deps._ensure_default_character_async(published, 3))
    assert raw_connections, "The real default worker did not perform database work"
    assert all(raw.info.transaction_status.name == "IDLE" for raw in raw_connections)
    assert published._get_pinned_backend() is None


def test_closed_borrowed_connection_is_not_replaced_or_returned_by_http_scope(pg_http, monkeypatch):
    from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import ExternalConnection, chacha_operation

    f = pg_http
    pool = f.backend.get_pool()
    raw = pool.get_connection()
    returned = []
    original_return = pool.return_connection

    def record(connection):
        returned.append(connection)
        return original_return(connection)

    monkeypatch.setattr(pool, "return_connection", record)

    @f.app.get("/probe/closed-borrowed")
    async def borrowed():
        with chacha_operation(independent=True, bindings=(ExternalConnection(f.db, raw, f.backend),)):
            raw.close()  # Only the external owner closes its borrowed connection.
            with pytest.raises(CharactersRAGDBError, match="borrowed"):
                f.db.get_connection()
        assert raw not in returned
        return {"external_preserved": True}

    async def operation(client):
        response = await client.get("/probe/closed-borrowed")
        assert response.status_code == 200
        assert response.json() == {"external_preserved": True}

    try:
        asyncio.run(_run(f, operation))
    finally:
        original_return(raw)


@pytest.mark.parametrize("cancel", [False, True], ids=["response-finishes", "request-cancelled"])
@pytest.mark.parametrize("surface", ["db", "connection", "cursor"])
@pytest.mark.parametrize("query_error", [False, True], ids=["success", "backend-error"])
def test_inflight_detached_worker_cannot_use_a_returned_http_checkout(
    pg_http, monkeypatch, cancel, surface, query_error
):
    from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import ChaChaOperation

    f = pg_http
    entered = threading.Event()
    finalizing = threading.Event()
    release = threading.Event()
    returned = threading.Event()
    query_finished = threading.Event()
    ready = asyncio.Event()
    workers = []
    checkouts = []
    premature_returns = []
    worker_context = ContextVar("inflight_fixture_worker", default=False)
    original_execute = f.backend.execute
    pool = f.backend.get_pool()
    original_return = pool.return_connection
    original_close = ChaChaOperation.close

    def execute(*args, **kwargs):
        if not worker_context.get():
            return original_execute(*args, **kwargs)
        checkouts.append(kwargs["connection"])
        entered.set()
        assert release.wait(timeout=5)
        try:
            return original_execute(*args, **kwargs)
        finally:
            query_finished.set()

    def return_connection(raw):
        if any(raw is item for item in checkouts):
            premature_returns.append(not query_finished.is_set())
            returned.set()
        return original_return(raw)

    def close(owner):
        if entered.is_set() and not query_finished.is_set():
            finalizing.set()
        return original_close(owner)

    monkeypatch.setattr(f.backend, "execute", execute)
    monkeypatch.setattr(pool, "return_connection", return_connection)
    monkeypatch.setattr(ChaChaOperation, "close", close)

    def controller():
        try:
            assert finalizing.wait(timeout=5)
            returned.wait(timeout=0.2)
        finally:
            release.set()

    control = threading.Thread(target=controller)
    control.start()

    @f.app.get("/probe/inflight")
    async def inflight(db: CharactersRAGDB = Depends(deps.get_chacha_db_for_user)):
        connection = db.get_connection()

        def worker():
            token = worker_context.set(True)
            try:
                query = (
                    "SELECT missing_inflight_column FROM notes WHERE id=?"
                    if query_error
                    else "SELECT id FROM notes WHERE id=?"
                )
                if surface == "db":
                    return db.execute_query(query, (f.note,)).fetchone()
                target = connection if surface == "connection" else connection.cursor()
                return target.execute(query, (f.note,)).fetchone()
            finally:
                worker_context.reset(token)

        workers.append(asyncio.create_task(asyncio.to_thread(worker)))
        assert await asyncio.to_thread(entered.wait, 5)
        ready.set()
        if cancel:
            await asyncio.Event().wait()
        return {"launched": True}

    async def operation(client):
        request = asyncio.create_task(client.get("/probe/inflight"))
        await asyncio.wait_for(ready.wait(), timeout=5)
        if cancel:
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
        else:
            assert (await request).status_code == 200
        rows = await asyncio.gather(*workers, return_exceptions=True)
        if query_error:
            from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError

            assert isinstance(rows[0], (CharactersRAGDBError, DatabaseError))
        else:
            assert rows[0]["id"] == f.note
        assert premature_returns == [False], "The pool received the checkout before its in-flight query finished"
        assert checkouts[0].info.transaction_status.name == "IDLE"

    try:
        asyncio.run(_run(f, operation))
    finally:
        release.set()
        control.join(timeout=5)


@pytest.mark.parametrize("decision", ["commit", "rollback", "error"])
def test_inflight_explicit_transaction_retains_checkout_through_its_decision(pg_http, monkeypatch, decision):
    f = pg_http
    entered = threading.Event()
    release = threading.Event()
    workers = []
    raw_connections = []
    returns = []
    pool = f.backend.get_pool()
    original_return = pool.return_connection

    def record(raw):
        if any(raw is item for item in raw_connections):
            returns.append(raw)
        return original_return(raw)

    monkeypatch.setattr(pool, "return_connection", record)

    @f.app.post("/probe/inflight-transaction")
    async def transaction(db: CharactersRAGDB = Depends(deps.get_chacha_db_for_user)):
        def worker():
            with db.transaction() as conn:
                raw_connections.append(conn._connection)
                conn.execute("UPDATE notes SET title=? WHERE id=?", ("Explicit pending", f.note))
                entered.set()
                assert release.wait(timeout=5)
                if decision == "rollback":
                    conn.rollback()
                elif decision == "error":
                    raise ValueError("Explicit transaction failure")

        workers.append(asyncio.create_task(asyncio.to_thread(worker)))
        assert await asyncio.to_thread(entered.wait, 5)
        return {"transaction_running": True}

    async def operation(client):
        response = await client.post("/probe/inflight-transaction")
        assert response.status_code == 200
        assert not returns, "Response completion returned a checkout with an active explicit transaction"
        assert f.backend.execute("SELECT title FROM notes WHERE id=%s", (f.note,)).scalar == "Committed title"
        release.set()
        result = await asyncio.gather(*workers, return_exceptions=True)
        assert isinstance(result[0], ValueError) if decision == "error" else result == [None]
        assert returns == raw_connections
        assert raw_connections[0].info.transaction_status.name == "IDLE"
        assert f.backend.execute("SELECT title FROM notes WHERE id=%s", (f.note,)).scalar == (
            "Explicit pending" if decision == "commit" else "Committed title"
        )

    try:
        asyncio.run(_run(f, operation))
    finally:
        release.set()


def test_retained_wrapper_cannot_execute_from_a_later_http_owner(pg_http):
    from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import ClosedChaChaOperationError

    f = pg_http
    retained = []

    @f.app.get("/probe/capture-wrapper")
    async def capture(db: CharactersRAGDB = Depends(deps.get_chacha_db_for_user)):
        retained.append(db.get_connection())
        retained.append(retained[0].cursor())
        return {"captured": True}

    @f.app.get("/probe/reuse-wrapper")
    async def reuse(db: CharactersRAGDB = Depends(deps.get_chacha_db_for_user)):
        assert db.get_note_by_id(f.note)["title"] == "Committed title"
        for wrapper in retained:
            with pytest.raises(ClosedChaChaOperationError):
                wrapper.execute("SELECT id FROM notes WHERE id=?", (f.note,))
        return {"rejected": True}

    async def operation(client):
        assert (await client.get("/probe/capture-wrapper")).status_code == 200
        assert (await client.get("/probe/reuse-wrapper")).json() == {"rejected": True}

    asyncio.run(_run(f, operation))


@pytest.mark.parametrize("later_request", [False, True], ids=["outside-owner", "later-owner"])
def test_retained_connection_cannot_rebind_a_new_cursor_after_return(pg_http, monkeypatch, later_request):
    from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import ClosedChaChaOperationError

    f = pg_http
    retained = []

    @f.app.get("/probe/cursor-source")
    async def capture(db: CharactersRAGDB = Depends(deps.get_chacha_db_for_user)):
        retained.append(db.get_connection())
        return {"captured": True}

    @f.app.get("/probe/new-cursor")
    async def attempt():
        assert f.db.get_note_by_id(f.note)["title"] == "Committed title"
        assert f.db.get_connection()._connection is retained[0]._connection
        with pytest.raises(ClosedChaChaOperationError):
            retained[0].cursor()
        return {"rejected": True}

    async def operation(client):
        assert (await client.get("/probe/cursor-source")).status_code == 200
        if later_request:
            await asyncio.sleep(0)
            if deps._chacha_default_char_tasks:
                await asyncio.gather(*tuple(deps._chacha_default_char_tasks))
            pool = f.backend.get_pool()
            original_get = pool.get_connection

            def prefer_returned_checkout():
                held = []
                try:
                    for _ in range(f.config.pool_size):
                        raw = original_get()
                        if raw is retained[0]._connection:
                            return raw
                        held.append(raw)
                    pytest.fail("The original open checkout was not returned to its pool")
                finally:
                    for raw in held:
                        pool.return_connection(raw)

            monkeypatch.setattr(pool, "get_connection", prefer_returned_checkout)
            assert (await client.get("/probe/new-cursor")).json() == {"rejected": True}
        else:
            with pytest.raises(ClosedChaChaOperationError):
                retained[0].cursor()

    asyncio.run(_run(f, operation))
