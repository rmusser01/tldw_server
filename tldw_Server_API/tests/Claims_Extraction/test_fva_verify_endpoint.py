"""POST /claims/verify/fva end to end: who the request runs for, and its quota refusals (TASK-13881).

The claims router alone, on real media DBs: one SQLite file per user, or PostgreSQL's shared tables
under a role that cannot bypass row-level security, with or without a per-user vector store. Only the
provider call and the query embedding are stubbed: the provider stub runs the quota check the
instrumented provider runs first, and notes who the call is for.
"""

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from fastapi import Depends, FastAPI

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal, get_request_user
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.endpoints import claims as claims_endpoints
from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
from tldw_Server_API.app.core.AuthNZ.settings import get_settings, reset_settings
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_single_user_instance
from tldw_Server_API.app.core.Chat import chat_service
from tldw_Server_API.app.core.Claims_Extraction import claims_engine, claims_service
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context, set_scope
from tldw_Server_API.app.core.Embeddings.Embeddings_Server import Embeddings_Create
from tldw_Server_API.app.core.RAG.rag_service import database_retrievers
from tldw_Server_API.app.core.RAG.rag_service.vector_stores.base import VectorSearchResult
from tldw_Server_API.app.core.Usage import llm_accounting, llm_quota_gate, quota_resolver, usage_tracker
from tldw_Server_API.app.core.Usage.llm_call_context import current_llm_user_id
from tldw_Server_API.app.core.Usage.quota_checks import LLMQuotaRefusal
from tldw_Server_API.tests.AuthNZ_SQLite._user_fixtures import create_authnz_test_user

pytestmark = pytest.mark.unit

OTHER_USER = 2
STRANGER = 5
MEMBER = 7
ADMIN = 9
_BODY = {
    "claims": [{"text": "The sky is green."}],
    "query": "clear day",
    "top_k": 3,
    "fva_config": {"force_claim_types": ["general"], "max_budget_usd": 5.0},
}
_REFUSAL = LLMQuotaRefusal("llm_tokens_month", 100, 100, "Monthly LLM token limit reached", 60)
_ANTI_QUERY_WORDS = ("against", "contradicts", "disproves")  # the first three negation templates
_BOTH_BACKENDS = pytest.mark.parametrize("fva", ["sqlite", "postgresql"], indirect=True)
_BACKENDS_AND_STORES = pytest.mark.parametrize(
    "fva", ["sqlite", "postgresql", "sqlite+vectors", "postgresql+vectors"], indirect=True
)


class _Spies:
    """What the request did, per AC #3: the evidence's DB, the vector store, the quota decisions, the usage rows."""

    def __init__(self, with_vectors: bool) -> None:
        self.with_vectors = with_vectors
        self.admitted = 10**6
        self.decided_for: list[int] = []
        self.calls: list[tuple[str, int | None]] = []
        self.usage_rows: list[tuple[str, int | None]] = []
        self.vector_store_users: list[str] = []
        self.vector_collections: list[str] = []


@pytest.fixture
def restricted_pg_backend(request):
    """A PostgreSQL content backend whose role cannot bypass RLS (as tests/RAG/test_restricted_postgres_media_retrieval.py)."""
    from psycopg import sql

    config = request.getfixturevalue("pg_database_config")
    admin = DatabaseBackendFactory.create_backend(config)
    role, password = "fva_endpoint_" + uuid4().hex, uuid4().hex
    with admin.transaction() as conn:
        conn.execute(
            sql.SQL(
                "CREATE ROLE {} LOGIN PASSWORD {} NOSUPERUSER NOBYPASSRLS NOINHERIT NOCREATEDB NOCREATEROLE"
            ).format(sql.Identifier(role), sql.Literal(password))
        )
        conn.execute(sql.SQL("GRANT USAGE, CREATE ON SCHEMA public TO {}").format(sql.Identifier(role)))
        conn.execute(
            sql.SQL("GRANT CREATE ON DATABASE {} TO {}").format(
                sql.Identifier(config.pg_database), sql.Identifier(role)
            )
        )
    backend = DatabaseBackendFactory.create_backend(
        replace(config, pg_user=role, pg_password=password, connection_string=None, pool_size=2, max_overflow=4)
    )
    try:
        yield backend
    finally:
        backend.get_pool().close_all()
        with admin.transaction() as conn:
            conn.execute("SET LOCAL lock_timeout = '10s'")  # fail, not hang, on a leaked session (TASK-13900)
            conn.execute(sql.SQL("DROP OWNED BY {}").format(sql.Identifier(role)))
            conn.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(role)))
        admin.get_pool().close_all()


@pytest.fixture
def fva(request, monkeypatch, tmp_path: Path):
    """The claims router, a seeded media DB per user, and spies on everything the request decides per user."""
    param = getattr(request, "param", "sqlite")
    backend = request.getfixturevalue("restricted_pg_backend") if param.startswith("postgresql") else None
    spies = _Spies(with_vectors=param.endswith("+vectors"))

    def _open(owner: int, file_owner: int | None = None) -> MediaDatabase:
        """``owner``'s media DB: their own SQLite file (or ``file_owner``'s), else the shared PostgreSQL tables."""
        if backend is not None:
            return MediaDatabase(db_path=":memory:", client_id=str(owner), backend=backend)
        return MediaDatabase(db_path=str(tmp_path / f"user_{file_owner or owner}.db"), client_id=str(owner))

    seeds = [(owner, owner) for owner in (OTHER_USER, MEMBER, ADMIN, int(get_single_user_instance().id))]
    # A row of another account in OTHER_USER's file, as PostgreSQL's shared tables hold. Only OTHER_USER's
    # own content scope, not an admin's, keeps it out of the evidence.
    seeds.append((STRANGER, OTHER_USER))
    rows: dict[int, list[tuple[int, str]]] = {}  # each owner's (media id, content): sky first, counter-evidence second
    for owner, file_owner in seeds:
        db = _open(owner, file_owner)
        with scoped_context(user_id=owner, is_admin=False):
            for title, content in (
                (f"Sky of user {owner}", f"From the home of user {owner} the sky is blue on a clear day."),
                (
                    f"Counter-evidence of user {owner}",
                    f"Evidence against the claim that the sky is green, by user {owner}: it is blue.",
                ),
            ):
                media_id, _, _ = db.add_media_with_keywords(
                    title=title, media_type="text", content=content, owner_user_id=owner, keywords=None
                )
                rows.setdefault(owner, []).append((media_id, content))
        db.close_connection()

    async def _decide(user_id, requested_tokens=1):
        spies.decided_for.append(user_id)
        return None if len(spies.decided_for) <= spies.admitted else _REFUSAL

    def _provider_call(**kwargs):
        kind = "adjudicate" if "supports, contradicts, or is neutral" in str(kwargs.get("system_message")) else "verify"
        spies.calls.append((kind, current_llm_user_id()))  # the user the call's usage row is written for
        llm_quota_gate.check_llm_call_sync()  # what the instrumented provider runs first
        if kind == "adjudicate":
            return json.dumps({"stance": "CONTRADICTS", "confidence": 0.95})
        return json.dumps({"label": "supported", "confidence": 0.95, "rationale": "ok"})

    async def _log_llm_usage(*, user_id, operation, **_kwargs):
        spies.usage_rows.append((operation, user_id))

    class _UserVectorStore:
        """One user's collection: an anti-context query's nearest neighbour is their counter-evidence row."""

        _initialized = True

        def __init__(self, owner: int) -> None:
            self.owner = owner

        async def search(self, collection_name, query_vector, k=10, filter=None, include_metadata=True):
            spies.vector_collections.append(collection_name)
            media_id, content = rows[self.owner][1 if query_vector[0] else 0]
            return [VectorSearchResult(f"{media_id}_0", content, {"media_id": str(media_id)}, 0.8, 0.2)]

    def _vector_store(settings, user_id="0"):
        spies.vector_store_users.append(str(user_id))
        return _UserVectorStore(int(user_id)) if spies.with_vectors else None  # None: FTS alone

    def _embed(texts, *args, **kwargs):
        return [[1.0 if any(word in text.lower() for word in _ANTI_QUERY_WORDS) else 0.0] for text in texts]

    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "1")
    quota_resolver.invalidate_all()
    monkeypatch.setattr(llm_quota_gate, "llm_quota_refusal", _decide)
    monkeypatch.setattr(chat_service, "perform_chat_api_call", _provider_call)
    monkeypatch.setattr(usage_tracker, "log_llm_usage", _log_llm_usage)
    monkeypatch.setattr(database_retrievers, "create_from_settings_for_user", _vector_store)
    monkeypatch.setattr(Embeddings_Create, "get_embedding_config", lambda: {})
    monkeypatch.setattr(Embeddings_Create, "create_embeddings_batch", _embed)
    monkeypatch.setattr(claims_engine, "_claims_local_nli_enabled", lambda: False)
    # The real _claims_user_override_db opens the admin's target: their file, or the shared backend.
    monkeypatch.setattr(claims_service, "get_user_media_db_path", lambda uid: str(tmp_path / f"user_{int(uid)}.db"))
    if backend is not None:
        real_managed = claims_service.managed_media_database
        monkeypatch.setattr(claims_service, "managed_media_database", lambda **kw: real_managed(**kw, backend=backend))

    async def _own_media_db(current_user: User = Depends(get_request_user)):
        db = _open(int(current_user.id))
        try:
            yield db
        finally:
            db.close_connection()

    app = FastAPI()
    app.include_router(claims_endpoints.router, prefix="/api/v1")
    app.dependency_overrides[get_auth_principal] = lambda: None
    app.dependency_overrides[get_media_db_for_user] = _own_media_db

    async def post(query_string: str = "", headers: dict[str, str] | None = None, body: dict | None = None):
        monkeypatch.setattr(llm_accounting, "_accounting_loop", asyncio.get_running_loop())
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                f"/api/v1/claims/verify/fva{query_string}", json=body or _BODY, headers=headers or {}
            )

    def as_multi_user(user_id: int, role: str) -> None:
        """Multi-user mode: the account and content scope multi-user auth resolves for ``role``."""
        monkeypatch.setenv("AUTH_MODE", "multi_user")
        reset_settings()

        async def _account() -> User:
            set_scope(user_id=user_id, is_admin=role == "admin")
            return User(id=user_id, username=f"name-{user_id}", role=role, roles=[role], is_admin=role == "admin")

        app.dependency_overrides[get_request_user] = _account

    return spies, post, as_multi_user


@pytest.fixture
async def real_api_keys(tmp_path, monkeypatch):
    """Multi-user mode for real: an AuthNZ SQLite DB with an admin (ADMIN) and a member (MEMBER), each with an API key."""
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret-key-fva-endpoint-1234567890")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'users.db'}")
    monkeypatch.setenv("VIRTUAL_KEYS_ENABLED", "false")
    reset_settings()
    await reset_db_pool()
    try:
        pool = await get_db_pool()
        ensure_authnz_tables(Path(pool.db_path))
        keys: dict[int, str] = {}
        manager = APIKeyManager()
        await manager.initialize()
        for user_id, role in ((ADMIN, "admin"), (MEMBER, "user")):
            await create_authnz_test_user(
                pool, username=f"u{user_id}", email=f"u{user_id}@example.com", role=role, user_id=user_id, is_verified=True
            )
            role_row = await pool.fetchrow("SELECT id FROM roles WHERE name = ?", role)
            role_id = int(role_row["id"] if isinstance(role_row, dict) else role_row[0])
            async with pool.transaction() as conn:
                await conn.execute("INSERT OR IGNORE INTO user_roles (user_id, role_id) VALUES (?, ?)", (user_id, role_id))
            created = await manager.create_api_key(user_id=user_id, name=f"k{user_id}", scope="write", expires_in_days=7)
            keys[user_id] = str(created["key"])
        yield keys
    finally:
        await reset_db_pool()


def _assert_all_for(spies: _Spies, response: httpx.Response, owner: int) -> None:
    """The evidence, the vector store, every quota decision and every usage row belong to ``owner``."""
    assert response.status_code == 200, response.text
    result = response.json()["results"][0]
    assert result["anti_context_found"] > 0 and result["adjudication"] is not None, result
    evidence = result["supporting_evidence"] + result["contradicting_evidence"]
    assert evidence and all(f"user {owner}" in e["snippet"] for e in evidence), evidence  # the media DB
    assert set(spies.vector_store_users) == {str(owner)}
    assert set(spies.vector_collections) == ({f"user_{owner}_media_embeddings"} if spies.with_vectors else set())
    assert spies.decided_for and set(spies.decided_for) == {owner}
    assert {user for _op, user in spies.usage_rows} == {owner}  # claims_verify rows, by id
    assert {kind for kind, _user in spies.calls} == {"verify", "adjudicate"}
    assert {user for _kind, user in spies.calls} == {owner}


@_BACKENDS_AND_STORES
@pytest.mark.parametrize("query_string", ["", f"?user_id={OTHER_USER}"])
async def test_multi_user_member_runs_for_itself_whatever_user_id_it_sends(fva, query_string):
    """AC #1, AC #3: a non-admin's user_id is ignored for the media DB, vector store, quota and usage rows."""
    spies, post, as_multi_user = fva
    as_multi_user(MEMBER, "user")

    _assert_all_for(spies, await post(query_string), MEMBER)


@_BACKENDS_AND_STORES
async def test_multi_user_admin_runs_for_the_user_it_names(fva):
    """AC #3: an admin's user_id switches all four to that user, and only that user's rows are evidence."""
    spies, post, as_multi_user = fva
    as_multi_user(ADMIN, "admin")

    _assert_all_for(spies, await post(f"?user_id={OTHER_USER}"), OTHER_USER)


async def test_multi_user_admin_without_user_id_runs_for_itself(fva):
    """SQLite only: on PostgreSQL an admin's own scope reads every account's rows, as for any admin read."""
    spies, post, as_multi_user = fva
    as_multi_user(ADMIN, "admin")

    _assert_all_for(spies, await post(), ADMIN)


async def test_a_request_without_max_budget_usd_runs_too(fva):
    """On dev this body failed at MultiDatabaseRetriever(media_db=db), the body with a budget at ClaimsJobContext."""
    spies, post, as_multi_user = fva
    as_multi_user(MEMBER, "user")

    _assert_all_for(spies, await post(body={**_BODY, "fva_config": {"force_claim_types": ["general"]}}), MEMBER)


@_BOTH_BACKENDS
@pytest.mark.parametrize(("caller", "owner"), [(ADMIN, OTHER_USER), (MEMBER, MEMBER)])
async def test_real_multi_user_api_key_auth_resolves_the_same_owner(fva, real_api_keys, caller, owner):
    """The real multi-user get_request_user and AuthNZ DB: the admin's ?user_id= counts, the member's does not."""
    spies, post, _as_multi_user = fva

    response = await post(f"?user_id={OTHER_USER}", headers={"X-API-KEY": real_api_keys[caller]})

    _assert_all_for(spies, response, owner)


@pytest.mark.parametrize("names_other_user", [False, True])
async def test_single_user_mode_honours_user_id_because_the_single_user_is_an_admin(fva, monkeypatch, names_other_user):
    """Real single-user API-key auth: the single user has the admin role, so user_id counts, as for rebuild_claims."""
    spies, post, _as_multi_user = fva
    monkeypatch.setenv("AUTH_MODE", "single_user")
    reset_settings()
    single_user = int(get_single_user_instance().id)

    response = await post(
        f"?user_id={OTHER_USER}" if names_other_user else "",
        headers={"X-API-KEY": get_settings().SINGLE_USER_API_KEY},
    )

    _assert_all_for(spies, response, OTHER_USER if names_other_user else single_user)


@pytest.mark.parametrize(("admitted", "refused"), [(0, "verify"), (1, "adjudicate")])
async def test_a_quota_refusal_at_the_verify_or_adjudication_call_answers_402(fva, admitted, refused):
    """AC #2: the refusal reaches the client as the 402 limit_exceeded body, for the request's user."""
    spies, post, as_multi_user = fva
    as_multi_user(MEMBER, "user")
    spies.admitted = admitted

    response = await post()

    assert response.status_code == 402, response.text
    assert response.json()["detail"]["error"] == "limit_exceeded"
    assert spies.calls[-1] == (refused, MEMBER)
    assert set(spies.decided_for) == {MEMBER}
    assert spies.usage_rows == ([("claims_verify", MEMBER)] if admitted else [])  # the paid verify call stays charged
