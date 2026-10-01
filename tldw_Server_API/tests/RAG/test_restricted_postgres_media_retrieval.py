"""Restricted PostgreSQL coverage for the Media RAG retrieval boundary."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from psycopg import sql

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import check_rate_limit
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.media_db.api import search_media
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context
from tldw_Server_API.app.core.RAG.rag_service.database_retrievers import (
    MediaDBRetriever,
    RetrievalConfig,
)
from tldw_Server_API.app.core.RAG.rag_service.unified_pipeline import unified_rag_pipeline
from tldw_Server_API.app.main import app as fastapi_app

pytestmark = pytest.mark.integration


@pytest.fixture(params=["sqlite", "postgresql"])
def restricted_media_store(request, tmp_path):
    """Yield a Media DB that cannot bypass PostgreSQL RLS for the PostgreSQL case."""
    admin = backend = None
    role = None
    if request.param == "postgresql":
        config = request.getfixturevalue("pg_database_config")
        admin = DatabaseBackendFactory.create_backend(config)
        role = "rag_retrieval_" + uuid4().hex
        password = uuid4().hex
        with admin.transaction() as conn:
            conn.execute(
                sql.SQL(
                    "CREATE ROLE {} LOGIN PASSWORD {} NOSUPERUSER NOBYPASSRLS "
                    "NOINHERIT NOCREATEDB NOCREATEROLE"
                ).format(sql.Identifier(role), sql.Literal(password))
            )
            conn.execute(sql.SQL("GRANT USAGE, CREATE ON SCHEMA public TO {}").format(sql.Identifier(role)))
            conn.execute(
                sql.SQL("GRANT CREATE ON DATABASE {} TO {}").format(
                    sql.Identifier(config.pg_database), sql.Identifier(role)
                )
            )
        backend = DatabaseBackendFactory.create_backend(
            replace(config, pg_user=role, pg_password=password, connection_string=None, pool_size=1, max_overflow=1)
        )
        db = MediaDatabase(db_path=":memory:", client_id="1", backend=backend)
        with backend.transaction() as conn:
            flags = conn.execute(
                "SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user"
            ).fetchone()
            assert not flags["rolsuper"] and not flags["rolbypassrls"]
    else:
        db = MediaDatabase(db_path=str(tmp_path / "media.sqlite"), client_id="1")

    try:
        yield db
    finally:
        db.close_connection()
        if backend is not None:
            backend.get_pool().close_all()
        if role is not None:
            with admin.transaction() as conn:
                conn.execute(sql.SQL("DROP OWNED BY {}").format(sql.Identifier(role)))
                conn.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(role)))
        if admin is not None:
            admin.get_pool().close_all()


@pytest.mark.parametrize("id_kind", ["integer", "uuid"])
@pytest.mark.parametrize("sort_by", ["relevance", "title_asc", "title_desc"])
def test_scoped_media_fts_filters_keep_parameter_order_and_owner(
    restricted_media_store, id_kind, sort_by,
):
    """Both WHERE and rank bindings preserve filters, pagination and RLS."""
    db = restricted_media_store
    records = []
    for label in ["alder", "Birch", "Cedar", "Outside"]:
        marker = "Unrelated" if label == "Cedar" else "Rowan"
        with scoped_context(user_id=1, is_admin=False):
            media_id, media_uuid, _ = db.add_media_with_keywords(
                title=f"{label} {marker}",
                media_type="document",
                content=f"{marker} observatory source {label}.",
                owner_user_id=1,
                author="Rowan curator",
                ingestion_date="2026-09-18T12:00:00",
                keywords=["orbit"],
            )
            records.append((media_id, media_uuid))
    ids = [row[0 if id_kind == "integer" else 1] for row in records[:3]]
    with scoped_context(user_id=1, is_admin=False):
        pages = [search_media(
            db,
            search_query="Rowan",
            search_fields=["title", "content", "author"],
            media_ids_filter=ids,
            media_types=["document"],
            date_range={"start_date": datetime(2026, 9, 1), "end_date": datetime(2026, 10, 1)},
            must_have_keywords=["orbit"],
            must_not_have_keywords=["excluded"],
            sort_by=sort_by,
            results_per_page=1,
            page=page,
        ) for page in (1, 2, 3)]
    assert [total for _, total in pages] == [2, 2, 2]
    assert {row["id"] for rows, _ in pages for row in rows} == {records[0][0], records[1][0]}
    assert [len(rows) for rows, _ in pages] == [1, 1, 0]
    if sort_by in {"title_asc", "title_desc"}:
        expected = records[:2] if sort_by == "title_asc" else records[1::-1]
        assert [row["id"] for rows, _ in pages for row in rows] == [record[0] for record in expected]
    with scoped_context(user_id=2, is_admin=False):
        assert search_media(db, search_query="Rowan", media_ids_filter=ids) == ([], 0)


@pytest.mark.asyncio
async def test_owner_retrieves_chunked_media_through_standard_pipeline(restricted_media_store) -> None:
    """A restricted owner can retrieve a persisted Media source through standard RAG."""
    db = restricted_media_store
    with scoped_context(user_id=1, is_admin=False):
        media_id, _media_uuid, _content_hash = db.add_media_with_keywords(
            title="Rowan Observatory",
            media_type="document",
            content="Rowan directs the observatory from the northern station.",
            keywords=["rowan"],
            chunks=[
                {
                    "text": "Rowan directs the observatory from the northern station.",
                    "start_char": 0,
                    "end_char": 57,
                }
            ],
        )
        assert media_id is not None

        media_rows, _total = search_media(
            db,
            search_query="Rowan",
            search_fields=["title", "content"],
            results_per_page=8,
        )
        assert [row["id"] for row in media_rows] == [media_id]

        retriever = MediaDBRetriever(
            db_path=db.db_path_str,
            media_db=db,
            config=RetrievalConfig(
                max_results=8,
                min_score=0.2,
                use_fts=True,
                use_vector=False,
                fts_level="chunk",
            ),
        )
        direct_documents = await retriever.retrieve("Rowan")
        assert [document.metadata.get("media_id") for document in direct_documents] == [str(media_id)]

        result = await unified_rag_pipeline(
            query="Rowan",
            sources=["media_db"],
            search_mode="hybrid",
            fts_level="chunk",
            top_k=8,
            min_score=0.2,
            enable_generation=False,
            enable_cache=False,
            media_db_path=db.db_path_str,
            media_db=db,
            user_id="1",
        )

    assert result.documents
    assert any(
        str(document.get("content", "")).startswith("Rowan")
        for document in result.documents
        if isinstance(document, dict)
    )


def test_owner_retrieves_chunked_media_through_stream_endpoint(
    restricted_media_store,
    monkeypatch: pytest.MonkeyPatch,
    auth_headers: dict[str, str],
) -> None:
    """The stream emits the restricted owner's persisted media as context."""
    import tldw_Server_API.app.api.v1.endpoints.rag_unified as rag_endpoint

    db = restricted_media_store
    with scoped_context(user_id=1, is_admin=False):
        media_id, _media_uuid, _content_hash = db.add_media_with_keywords(
            title="Rowan Observatory",
            media_type="document",
            content="Rowan directs the observatory from the northern station.",
            keywords=["rowan"],
            chunks=[
                {
                    "text": "Rowan directs the observatory from the northern station.",
                    "start_char": 0,
                    "end_char": 57,
                }
            ],
        )
        assert media_id is not None

        async def current_user() -> User:
            return User(id=1, username="owner", email=None, is_active=True)

        async def rate_limit() -> None:
            return None

        async def media_db():
            yield db

        async def generate(context, **_kwargs):
            async def chunks():
                yield "controlled answer"

            context.stream_generator = chunks()
            context.metadata = {"streaming": True}
            return context

        monkeypatch.setattr(rag_endpoint, "_build_credential_runtime", lambda *_args: None)
        monkeypatch.setattr(rag_endpoint, "generate_streaming_response", generate)
        fastapi_app.dependency_overrides[get_request_user] = current_user
        fastapi_app.dependency_overrides[check_rate_limit] = rate_limit
        fastapi_app.dependency_overrides[get_media_db_for_user] = media_db
        try:
            with TestClient(fastapi_app, headers=auth_headers) as client:
                with client.stream(
                    "POST",
                    "/api/v1/rag/search/stream",
                    json={
                        "query": "Rowan",
                        "strategy": "standard",
                        "sources": ["media_db"],
                        "search_mode": "hybrid",
                        "fts_level": "chunk",
                        "min_score": 0.2,
                        "top_k": 8,
                        "enable_generation": True,
                    },
                ) as response:
                    assert response.status_code == 200
                    events = [json.loads(raw) for raw in response.iter_lines() if raw]
        finally:
            fastapi_app.dependency_overrides.clear()

    context_events = [event for event in events if event.get("type") == "contexts"]
    assert context_events
    assert any(
        context.get("source_id") == str(media_id)
        for context in context_events[0]["contexts"]
    )


@pytest.fixture
def authenticated_media_client(restricted_media_store, monkeypatch):
    """Use real route authentication with no inherited content scope."""
    from fastapi import Depends, FastAPI

    from tldw_Server_API.app.api.v1.endpoints import rag_unified
    from tldw_Server_API.app.api.v1.endpoints.media import item
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.DB_Management.scope_context import get_scope

    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "causal-content-scope-key")
    monkeypatch.setenv("SINGLE_USER_FIXED_ID", "1")
    monkeypatch.setenv("EVALS_HEAVY_ADMIN_ONLY", "true")
    reset_settings()
    db = restricted_media_store
    with scoped_context(user_id=1, is_admin=False):
        media_id, _, _ = db.add_media_with_keywords(
            title="Rowan Observatory",
            media_type="document",
            content="Rowan directs the observatory from the northern station.",
            keywords=["rowan"],
            chunks=[{"text": "Rowan directs the observatory from the northern station.",
                     "start_char": 0, "end_char": 57}],
        )
    assert get_scope() is None
    app = FastAPI()
    app.include_router(item.router, prefix="/media")
    app.include_router(rag_unified.router)

    async def media_db(current_user=Depends(get_request_user)):
        yield db

    async def unused_db():
        return None

    async def no_vectors(*_args, **_kwargs):
        return None

    async def generate(context, **_kwargs):
        async def chunks():
            yield "controlled answer"
        context.stream_generator = chunks()
        context.metadata = {"streaming": True}
        return context

    app.dependency_overrides[get_media_db_for_user] = media_db
    for dependency in (rag_unified.get_chacha_db_for_user, rag_unified.get_prompts_db_for_user,
                       rag_unified.get_collections_db_for_user):
        app.dependency_overrides[dependency] = unused_db
    app.dependency_overrides[check_rate_limit] = unused_db
    monkeypatch.setattr(item, "delete_media_vectors", no_vectors)
    monkeypatch.setattr(rag_unified, "_build_credential_runtime", lambda *_args: None)
    monkeypatch.setattr(rag_unified, "generate_streaming_response", generate)
    try:
        with TestClient(app) as client:
            yield client, db, media_id
        assert get_scope() is None
    finally:
        reset_settings()


@pytest.mark.parametrize("header", ["X-API-KEY", "Authorization"])
def test_permission_first_auth_trash_restore_without_preseeded_scope(authenticated_media_client, header):
    """Real permission-first DELETE must see the row authenticated GET can see."""
    client, db, media_id = authenticated_media_client
    key = "causal-content-scope-key"
    headers = {header: f"Bearer {key}" if header == "Authorization" else key}
    assert client.get(f"/media/{media_id}", headers=headers).status_code == 200
    assert client.delete(f"/media/{media_id}", headers=headers).status_code == 204
    with scoped_context(user_id=1, is_admin=False):
        assert db.get_media_by_id(media_id, include_trash=True)["is_trash"]
    response = client.post(f"/media/{media_id}/restore", headers=headers)
    assert response.status_code == 200
    with scoped_context(user_id=1, is_admin=False):
        assert not db.get_media_by_id(media_id, include_trash=True)["is_trash"]


@pytest.mark.parametrize("query,expected", [("Rowan", True), ("zzzxnomatch", False)])
def test_permission_first_auth_stream_without_preseeded_scope(authenticated_media_client, query, expected):
    """Real permission-first streaming retrieval must propagate authenticated scope."""
    client, _db, media_id = authenticated_media_client
    with client.stream("POST", "/api/v1/rag/search/stream", headers={"X-API-KEY": "causal-content-scope-key"},
                       json={"query": query, "strategy": "standard", "sources": ["media_db"],
                             "search_mode": "fts", "fts_level": "chunk", "min_score": 0.2,
                             "top_k": 8, "enable_generation": True, "enable_cache": False}) as response:
        assert response.status_code == 200
        events = [json.loads(raw) for raw in response.iter_lines() if raw]
    contexts = [context for event in events if event.get("type") == "contexts"
                for context in event["contexts"]]
    assert any(context.get("source_id") == str(media_id) for context in contexts) is expected


def test_failed_selected_media_read_emits_error_without_answer(
    authenticated_media_client, monkeypatch,
):
    """An authenticated selected-source QA failure must not generate empty-context prose."""
    from tldw_Server_API.app.api.v1.endpoints import rag_unified
    from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError

    client, db, media_id = authenticated_media_client
    execute = db.execute_query
    calls = []

    def failed_search(query, *args, **kwargs):
        if "COUNT(DISTINCT m.id)" in query:
            raise DatabaseError("private-database-path query-secret")
        return execute(query, *args, **kwargs)

    async def generate(*args, **kwargs):
        calls.append(True)
        raise AssertionError("Failed retrieval must not dispatch answer generation")

    monkeypatch.setattr(db, "execute_query", failed_search)
    monkeypatch.setattr(rag_unified, "generate_streaming_response", generate)
    with client.stream(
        "POST", "/api/v1/rag/search/stream",
        headers={"X-API-KEY": "causal-content-scope-key"},
        json={"query": "Rowan", "strategy": "standard", "sources": ["media_db"],
              "search_mode": "fts", "fts_level": "media", "include_media_ids": [media_id],
              "min_score": 0.0, "enable_generation": True, "enable_cache": False},
    ) as response:
        assert response.status_code == 200
        events = [json.loads(raw) for raw in response.iter_lines() if raw]
    assert [event["type"] for event in events] == ["error"]
    assert not calls
    assert events[0]["allow_non_stream_fallback"] is False
    assert "private-database-path" not in str(events)


@pytest.mark.asyncio
@pytest.mark.parametrize("restricted_media_store", ["postgresql"], indirect=True)
@pytest.mark.parametrize("boundary", ["principal", "user"])
async def test_cached_ordinary_identity_cannot_reuse_another_owners_scope(restricted_media_store, boundary):
    """A cache hit must not inherit the prior owner's elevated database access."""
    from starlette.requests import Request

    from tldw_Server_API.app.core.AuthNZ.auth_principal_resolver import get_auth_principal
    from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal
    from tldw_Server_API.app.core.DB_Management.scope_context import get_scope

    db = restricted_media_store
    with scoped_context(user_id=1, is_admin=False):
        media_id, _, _ = db.add_media_with_keywords(
            title="Private Observatory", media_type="document", content="Owner-only source.",
        )
    assert get_scope() is None
    request = Request({"type": "http", "headers": [], "path": "/", "method": "GET"})
    request.state.auth = AuthContext(principal=AuthPrincipal(kind="user", user_id=2))
    request.state._auth_user = User(id=2, username="other-owner")
    with scoped_context(user_id=1, is_admin=True):
        if boundary == "principal":
            await get_auth_principal(request)
        else:
            await get_request_user(request, api_key=None, token=None)
        assert db.get_media_by_id(media_id, include_trash=True) is None
    assert get_scope() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("fts_level", ["media", "chunk"])
@pytest.mark.parametrize("include_metadata", [True, False])
async def test_media_evidence_keeps_canonical_identity_in_stream_context(
    restricted_media_store, fts_level, include_metadata,
):
    """Source navigation must retain the Media owner, never substitute a chunk ID."""
    from tldw_Server_API.app.core.RAG.rag_service.streaming_executor import _context_events

    db = restricted_media_store
    with scoped_context(user_id=1, is_admin=False):
        media_id, _, _ = db.add_media_with_keywords(
            title="Rowan Observatory identity",
            media_type="document",
            content="Rowan Observatory is directed by Mira Vale.",
            owner_user_id=1,
            chunks=[{"text": "Rowan Observatory is directed by Mira Vale.", "start_char": 0, "end_char": 43}],
        )
        retriever = MediaDBRetriever(
            db_path=db.db_path_str, media_db=db,
            config=RetrievalConfig(
                max_results=8, min_score=0.0, use_fts=True, use_vector=False,
                fts_level=fts_level, include_metadata=include_metadata,
            ),
        )
        documents = await retriever.retrieve("Rowan", allowed_media_ids=[media_id])
        events = _context_events(docs=documents, payload={}, request_defaults={})
    contexts = events[0]["contexts"]
    assert len(contexts) == 1
    assert contexts[0].get("source_id") == str(media_id)
    assert contexts[0].get("source_type") == "media_db"
    assert contexts[0].get("evidence_origin") == "local_library"
    if fts_level == "chunk":
        assert contexts[0]["id"] != str(media_id)
    with scoped_context(user_id=2, is_admin=False):
        assert await retriever.retrieve("Rowan", allowed_media_ids=[media_id]) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("fts_level", ["media", "chunk"])
@pytest.mark.parametrize(
    "visibility,reader,teams,orgs,is_admin,visible",
    [
        ("personal", 1, [], [], False, True),
        ("personal", 2, [], [], False, False),
        ("personal", None, [], [], False, False),
        ("personal", 2, [], [], True, True),
        ("team", 2, [7], [], False, True),
        ("team", 2, [8], [], False, False),
        ("org", 2, [], [9], False, True),
        ("org", 2, [], [10], False, False),
    ],
)
async def test_media_evidence_respects_shared_visibility(
    restricted_media_store, fts_level, visibility, reader, teams, orgs, is_admin, visible,
):
    """Chunk and whole-source search share the same database visibility contract."""
    db = restricted_media_store
    with scoped_context(user_id=1, is_admin=True):
        media_id, _, _ = db.add_media_with_keywords(
            title="Rowan shared source", media_type="document",
            content="Rowan shared evidence.", owner_user_id=1,
            chunks=[{"text": "Rowan shared evidence.", "start_char": 0, "end_char": 22}],
        )
        if visibility != "personal":
            assert db.share_media(media_id, visibility, team_id=7 if visibility == "team" else None,
                                  org_id=9 if visibility == "org" else None)
    retriever = MediaDBRetriever(
        db_path=db.db_path_str, media_db=db,
        config=RetrievalConfig(max_results=1, min_score=0.0, use_fts=True,
                               use_vector=False, fts_level=fts_level),
    )
    with scoped_context(user_id=reader, team_ids=teams, org_ids=orgs, is_admin=is_admin):
        documents = await retriever.retrieve("Rowan", allowed_media_ids=[media_id])
    assert [doc.metadata["source_id"] for doc in documents] == ([str(media_id)] if visible else [])
