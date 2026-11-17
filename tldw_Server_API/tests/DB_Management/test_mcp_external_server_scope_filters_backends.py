"""External discovery inventory uses the same nullable scope filters on both databases."""

import secrets
from types import SimpleNamespace

import pytest
import pytest_asyncio

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
from tldw_Server_API.app.core.AuthNZ.repos.mcp_hub_repo import McpHubRepo
from tldw_Server_API.app.core.AuthNZ.settings import Settings


@pytest_asyncio.fixture(params=["sqlite", "postgres"])
async def external_repo(request, tmp_path, monkeypatch):
    if request.param == "postgres":
        backend = request.getfixturevalue("pg_restricted_backend")
        config = backend.config
        backend.get_pool().close_all()
        url = (
            f"postgresql://{config.pg_user}:{config.pg_password}@{config.pg_host}:{config.pg_port}/{config.pg_database}"
        )
        monkeypatch.setenv("TLDW_USER_DB_BACKEND", "postgres")
    else:
        url = f"sqlite:///{tmp_path / 'users.db'}"
        monkeypatch.setenv("TLDW_USER_DB_BACKEND", "sqlite")
    pool = DatabasePool(
        Settings(
            _env_file=None,
            AUTH_MODE="single_user",
            DATABASE_URL=url,
            JWT_SECRET_KEY=secrets.token_urlsafe(32),
            DATABASE_POOL_MIN_SIZE=1,
            DATABASE_POOL_MAX_SIZE=5,
        )
    )
    try:
        await pool.initialize()
        if request.param == "sqlite":
            ensure_authnz_tables(tmp_path / "users.db")
        repo = McpHubRepo(pool)
        await repo.ensure_tables()
        yield repo
    finally:
        await pool.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("populated", [False, True])
async def test_external_server_nullable_filters_preserve_inventory_and_order(external_repo, populated):
    repo = external_repo
    if populated:
        for server_id, name, scope, owner in (
            ("d", "D", "org", 7),
            ("c", "C", "user", 8),
            ("b2", "B", "user", 7),
            ("a", "A", "global", None),
            ("b1", "B", "user", 7),
        ):
            await repo.upsert_external_server(
                server_id=server_id,
                name=name,
                transport="stdio",
                config_json='{"cmd":"npx"}',
                owner_scope_type=scope,
                owner_scope_id=owner,
                enabled=False,
                actor_id=None,
            )
    for filters, expected in (
        ({}, ["a", "b1", "b2", "c", "d"]),
        ({"owner_scope_type": "global", "owner_scope_id": None}, ["a"]),
        ({"owner_scope_type": "user"}, ["b1", "b2", "c"]),
        ({"owner_scope_type": "user", "owner_scope_id": 7}, ["b1", "b2"]),
        ({"owner_scope_id": 7}, ["b1", "b2", "d"]),
        ({"owner_scope_type": "org", "owner_scope_id": 7}, ["d"]),
        ({"owner_scope_type": "user", "owner_scope_id": 99}, []),
    ):
        rows = await repo.list_external_servers(**filters)
        assert [row["id"] for row in rows] == (expected if populated else [])  # nosec B101 -- real DB filters and name/id order must preserve exact owner inventory.


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method,filters",
    [
        (method, filters)
        for method in ("list_workspace_set_objects", "list_shared_workspace_entries")
        for filters in ({}, {"owner_scope_type": "user", "owner_scope_id": 7})
    ]
    + [("list_shared_workspace_entries", {"workspace_id": "rowan"})],
)
async def test_governance_audit_inventory_nullable_filters(external_repo, method, filters):
    assert await getattr(external_repo, method)(**filters) == []  # nosec B101 -- actual remaining audit inventory queries must work with null and non-null scope filters.


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["update", "delete"])
@pytest.mark.parametrize("exists", [False, True])
async def test_external_server_mutation_acknowledges_actual_persistence(external_repo, operation, exists):
    repo = external_repo
    if exists:
        await repo.upsert_external_server(
            server_id="lifecycle",
            name="Before",
            transport="stdio",
            config_json='{"cmd":"npx"}',
            owner_scope_type="global",
            owner_scope_id=None,
            enabled=True,
            actor_id=None,
        )
    if operation == "update":
        result = await repo.update_external_server(
            "lifecycle",
            name="After",
            transport="stdio",
            config_json='{"cmd":"npx"}',
            owner_scope_type="global",
            owner_scope_id=None,
            enabled=False,
            actor_id=None,
        )
        stored = await repo.get_external_server("lifecycle")
        assert (stored["name"], stored["enabled"]) == ("After", False) if exists else stored is None  # nosec B101 -- first establish actual persisted update/no-create behavior.
        assert result == stored  # nosec B101 -- a committed PostgreSQL update must not be reported as missing.
    else:
        result = await repo.delete_external_server("lifecycle")
        assert await repo.get_external_server("lifecycle") is None  # nosec B101 -- deletion/no-op both leave no server.
        assert result is exists  # nosec B101 -- affected-row acknowledgement must distinguish existing versus missing.


@pytest.mark.parametrize(
    "result,expected",
    [
        ("UPDATE 1", True),
        ("update 2", True),
        ("DELETE 1", True),
        ("INSERT 0 1", True),
        ("UPDATE 0", False),
        ("DELETE 0", False),
        ("INSERT 0 0", False),
        ("UPDATE broken", False),
        ("UPDATE extra 1", False),
        ("INSERT broken 1", False),
        ("SELECT 1", False),
        ("", False),
        (None, False),
        (SimpleNamespace(rowcount=1), True),
        (SimpleNamespace(rowcount=0), False),
        (SimpleNamespace(rowcount=-1), False),
    ],
)
def test_mcp_command_result_requires_positive_actual_row_count(result, expected):
    assert McpHubRepo._command_touched_rows(result) is expected  # nosec B101 -- zero/malformed DB command tags cannot authorize a successful mutation.


@pytest.mark.asyncio
async def test_mcp_trust_policy_stale_write_has_no_success_acknowledgement(external_repo):
    repo = external_repo
    first = await repo.upsert_governance_pack_trust_policy(policy_document={"revision": 1}, actor_id=None)
    await repo.upsert_governance_pack_trust_policy(
        policy_document={"revision": 2},
        actor_id=None,
        expected_policy_document_json=first["policy_document_json_raw"],
    )
    stale = await repo.upsert_governance_pack_trust_policy(
        policy_document={"revision": 3},
        actor_id=None,
        expected_policy_document_json=first["policy_document_json_raw"],
    )
    current = await repo.get_governance_pack_trust_policy()
    assert current["policy_document"] == {"revision": 2}  # nosec B101 -- the competing committed policy must remain authoritative.
    assert stale is None  # nosec B101 -- PostgreSQL INSERT0 0 is a conflict, never a successful stale save.


@pytest.mark.asyncio
async def test_mcp_empty_workspace_rebind_is_not_reported_as_changed(external_repo):
    changed = await external_repo.rebind_policy_assignment_workspaces(old_assignment_id=998, new_assignment_id=999)
    assert changed is False  # nosec B101 -- both backends must reject a success acknowledgement for zero affected workspaces.
