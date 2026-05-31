"""Tests for standalone MCP SQLite storage contracts."""

from __future__ import annotations

import ast
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest


def _tldw_imports_for(path: Path) -> list[str]:
    """Return imports from a Python file that cross into the host package."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(
                alias.name
                for alias in node.names
                if alias.name == "tldw_Server_API"
                or alias.name.startswith("tldw_Server_API.")
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "tldw_Server_API" or node.module.startswith("tldw_Server_API."):
                imports.append(node.module)
    return imports


def _direct_imports_for(path: Path) -> list[str]:
    """Return direct module imports used by a Python source file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    return imports


def test_sqlite_storage_module_has_no_tldw_server_imports() -> None:
    import mcp_unified.storage.sqlite as sqlite_storage

    assert sqlite_storage.__file__ is not None
    sqlite_path = Path(sqlite_storage.__file__).resolve()

    assert _tldw_imports_for(sqlite_path) == []


def test_sqlite_storage_module_uses_db_abstraction_not_direct_sqlite3() -> None:
    import mcp_unified.storage.sqlite as sqlite_storage

    assert sqlite_storage.__file__ is not None
    sqlite_path = Path(sqlite_storage.__file__).resolve()
    direct_imports = _direct_imports_for(sqlite_path)

    assert "sqlite3" not in direct_imports
    assert not any(import_name.startswith("sqlite3.") for import_name in direct_imports)


def test_sqlite_store_initializes_schema_version_idempotently(tmp_path: Path) -> None:
    from mcp_unified.storage import SQLiteMCPStore

    db_path = tmp_path / "mcp.sqlite"
    store = SQLiteMCPStore(db_path)
    store.close()

    reopened = SQLiteMCPStore(db_path)
    reopened.close()

    with sqlite3.connect(db_path) as conn:
        version = conn.execute(
            "SELECT value FROM mcp_storage_meta WHERE key = ?",
            ("schema_version",),
        ).fetchone()
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'",
            )
        }

    assert version == ("1",)
    assert {
        "mcp_storage_meta",
        "mcp_profiles",
        "mcp_profile_assignments",
        "mcp_approval_policies",
        "mcp_credential_grants",
        "mcp_external_servers",
        "mcp_audit_events",
    }.issubset(tables)


def test_sqlite_store_rejects_future_schema_version(tmp_path: Path) -> None:
    from mcp_unified.storage import SQLiteMCPStore

    db_path = tmp_path / "mcp.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE mcp_storage_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO mcp_storage_meta(key, value) VALUES (?, ?)",
            ("schema_version", "999"),
        )

    with pytest.raises(RuntimeError, match="newer than supported"):
        SQLiteMCPStore(db_path)


def test_sqlite_store_uses_thread_safe_timeout_connection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import mcp_unified.storage.sqlite as sqlite_storage
    from mcp_unified.storage import SQLiteMCPStore

    original_create_engine = sqlite_storage.create_engine
    engine_kwargs: dict[str, Any] = {}

    def recording_create_engine(
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        engine_kwargs.update(kwargs)
        return original_create_engine(*args, **kwargs)

    monkeypatch.setattr(sqlite_storage, "create_engine", recording_create_engine)

    store = SQLiteMCPStore(tmp_path / "mcp.sqlite")
    store.close()

    assert engine_kwargs["connect_args"]["timeout"] == 30.0
    assert engine_kwargs["connect_args"]["check_same_thread"] is False


@pytest.mark.asyncio
async def test_sqlite_store_async_methods_offload_database_work(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import mcp_unified.storage.sqlite as sqlite_storage
    from mcp_unified.profiles import MCPProfile
    from mcp_unified.storage import AuditEvent, SQLiteMCPStore

    store = SQLiteMCPStore(tmp_path / "mcp.sqlite")
    calls: list[str] = []

    async def recording_to_thread(function: Any, /, *args: Any, **kwargs: Any) -> Any:
        calls.append(function.__name__)
        return function(*args, **kwargs)

    monkeypatch.setattr(sqlite_storage.asyncio, "to_thread", recording_to_thread)

    await store.upsert_profile(MCPProfile(id="backend", name="Backend"))
    await store.list_profiles()
    await store.append_event(AuditEvent(id="event-1", event_type="tool.allowed"))
    await store.query_events()
    await store.aclose()

    assert "_upsert_profile_sync" in calls
    assert "_list_profiles_sync" in calls
    assert "_append_event_sync" in calls
    assert "_query_events_sync" in calls
    assert "_close_sync" in calls


@pytest.mark.asyncio
async def test_sqlite_store_memory_database_survives_thread_offload() -> None:
    from mcp_unified.profiles import MCPProfile
    from mcp_unified.storage import SQLiteMCPStore

    store = SQLiteMCPStore(":memory:")

    await store.upsert_profile(MCPProfile(id="qa", name="QA Engineer"))
    profiles = await store.list_profiles()

    assert [profile.id for profile in profiles] == ["qa"]

    await store.aclose()


@pytest.mark.asyncio
async def test_sqlite_store_round_trips_profiles_with_copy_isolation(tmp_path: Path) -> None:
    from mcp_unified.profiles import MCPProfile
    from mcp_unified.storage import SQLiteMCPStore

    store = SQLiteMCPStore(tmp_path / "mcp.sqlite")
    profile = MCPProfile(
        id="orchestrator",
        name="Orchestrator",
        metadata={"nested": {"value": "original"}},
    )

    stored = await store.upsert_profile(profile)
    profile.metadata["nested"]["value"] = "mutated"
    stored.metadata["nested"]["value"] = "mutated-again"

    fetched = await store.get_profile("orchestrator")
    listed = await store.list_profiles()
    deleted = await store.delete_profile("orchestrator")
    missing_delete = await store.delete_profile("orchestrator")

    assert fetched is not None
    assert fetched.metadata["nested"]["value"] == "original"
    assert [profile.id for profile in listed] == ["orchestrator"]
    assert deleted is True
    assert missing_delete is False
    assert await store.get_profile("orchestrator") is None

    await store.aclose()


@pytest.mark.asyncio
async def test_sqlite_store_filters_assignment_policy_and_grant_rows(
    tmp_path: Path,
) -> None:
    from mcp_unified.profiles import MCPProfile
    from mcp_unified.storage import (
        ApprovalPolicyDocument,
        CredentialGrant,
        ProfileAssignment,
        SQLiteMCPStore,
    )

    store = SQLiteMCPStore(tmp_path / "mcp.sqlite")
    await store.upsert_profile(MCPProfile(id="backend", name="Backend"))
    await store.upsert_profile(MCPProfile(id="frontend", name="Frontend"))
    await store.upsert_assignment(
        ProfileAssignment(
            id="assignment-user",
            profile_id="backend",
            principal_id="user-1",
        )
    )
    await store.upsert_assignment(
        ProfileAssignment(
            id="assignment-workspace",
            profile_id="backend",
            workspace_id="workspace-1",
        )
    )
    await store.upsert_assignment(
        ProfileAssignment(
            id="assignment-default",
            profile_id="frontend",
            is_default=True,
        )
    )
    await store.upsert_policy(
        ApprovalPolicyDocument(
            id="policy-backend",
            name="Backend writes",
            profile_id="backend",
            required_for=["filesystem.write"],
        )
    )
    await store.upsert_policy(
        ApprovalPolicyDocument(
            id="policy-frontend",
            name="Frontend writes",
            profile_id="frontend",
        )
    )
    await store.upsert_grant(
        CredentialGrant(
            id="grant-search",
            profile_id="backend",
            broker_id="local",
            credential_slot="search",
            external_server_id="search-server",
        )
    )
    await store.upsert_grant(
        CredentialGrant(
            id="grant-docs",
            profile_id="frontend",
            broker_id="local",
            credential_slot="docs",
            external_server_id="docs-server",
        )
    )

    backend_assignments = await store.list_assignments(profile_id="backend")
    principal_assignments = await store.list_assignments(principal_id="user-1")
    workspace_assignments = await store.list_assignments(workspace_id="workspace-1")
    backend_policies = await store.list_policies(profile_id="backend")
    backend_grants = await store.list_grants(profile_id="backend")
    search_grants = await store.list_grants(external_server_id="search-server")

    assert [assignment.id for assignment in backend_assignments] == [
        "assignment-user",
        "assignment-workspace",
    ]
    assert [assignment.id for assignment in principal_assignments] == ["assignment-user"]
    assert [assignment.id for assignment in workspace_assignments] == [
        "assignment-workspace"
    ]
    assert [policy.id for policy in backend_policies] == ["policy-backend"]
    assert [grant.id for grant in backend_grants] == ["grant-search"]
    assert [grant.id for grant in search_grants] == ["grant-search"]
    assert await store.delete_assignment("assignment-user") is True
    assert await store.get_assignment("assignment-user") is None
    assert await store.delete_policy("missing-policy") is False
    assert await store.delete_grant("missing-grant") is False

    await store.aclose()


@pytest.mark.asyncio
async def test_sqlite_store_enforces_profile_foreign_keys_and_cascades(
    tmp_path: Path,
) -> None:
    from mcp_unified.profiles import MCPProfile
    from mcp_unified.storage import (
        ApprovalPolicyDocument,
        CredentialGrant,
        ProfileAssignment,
        SQLiteMCPStore,
    )
    from sqlalchemy.exc import IntegrityError

    store = SQLiteMCPStore(tmp_path / "mcp.sqlite")

    with pytest.raises(IntegrityError):
        await store.upsert_assignment(
            ProfileAssignment(
                id="orphan-assignment",
                profile_id="missing-profile",
                principal_id="user-1",
            )
        )

    await store.upsert_profile(MCPProfile(id="backend", name="Backend"))
    await store.upsert_assignment(
        ProfileAssignment(
            id="assignment-user",
            profile_id="backend",
            principal_id="user-1",
        )
    )
    await store.upsert_policy(
        ApprovalPolicyDocument(
            id="policy-backend",
            name="Backend writes",
            profile_id="backend",
        )
    )
    await store.upsert_grant(
        CredentialGrant(
            id="grant-search",
            profile_id="backend",
            broker_id="local",
            credential_slot="search",
        )
    )

    assert await store.delete_profile("backend") is True
    assert await store.list_assignments(profile_id="backend") == []
    assert await store.list_policies(profile_id="backend") == []
    assert await store.list_grants(profile_id="backend") == []

    await store.aclose()


@pytest.mark.asyncio
async def test_sqlite_store_lists_external_server_definitions(tmp_path: Path) -> None:
    from mcp_unified.storage import ExternalServerDefinition, SQLiteMCPStore

    store = SQLiteMCPStore(tmp_path / "mcp.sqlite")
    await store.upsert_server(
        ExternalServerDefinition(
            id="filesystem",
            name="Filesystem",
            transport="stdio",
            command=["/usr/local/bin/mcp-filesystem"],
        )
    )
    await store.upsert_server(
        ExternalServerDefinition(
            id="search",
            name="Search",
            transport="websocket",
            url="wss://example.test/mcp",
        )
    )
    await store.upsert_server(
        ExternalServerDefinition(
            id="draft",
            name="Draft",
            transport="stdio",
            enabled=False,
        )
    )

    all_servers = await store.list_servers()
    enabled_servers = await store.list_server_definitions(enabled=True)
    disabled_servers = await store.list_server_definitions(enabled=False)

    assert [server.id for server in all_servers] == ["draft", "filesystem", "search"]
    assert [server.id for server in enabled_servers] == ["filesystem", "search"]
    assert [server.id for server in disabled_servers] == ["draft"]
    assert await store.get_server("filesystem") is not None
    assert await store.delete_server("filesystem") is True
    assert await store.get_server("filesystem") is None

    await store.aclose()


@pytest.mark.asyncio
async def test_sqlite_store_appends_and_queries_audit_events(tmp_path: Path) -> None:
    from mcp_unified.storage import AuditEvent, SQLiteMCPStore

    store = SQLiteMCPStore(tmp_path / "mcp.sqlite")
    base_time = datetime(2026, 5, 28, 12, 0, tzinfo=timezone.utc)
    payload: dict[str, Any] = {
        "tool": "filesystem.write",
        "args": {"path": "/repo/README.md"},
    }

    await store.append_event(
        AuditEvent(
            id="old",
            event_type="tool.allowed",
            actor_id="user-1",
            profile_id="backend",
            payload=payload,
            created_at=base_time,
        )
    )
    await store.append_event(
        AuditEvent(
            id="middle",
            event_type="tool.denied",
            actor_id="user-2",
            profile_id="frontend",
            created_at=base_time + timedelta(minutes=1),
        )
    )
    await store.append_event(
        AuditEvent(
            id="new",
            event_type="tool.allowed",
            actor_id="user-1",
            profile_id="backend",
            created_at=base_time + timedelta(minutes=2),
        )
    )
    payload["args"]["path"] = "/repo/CHANGED.md"

    newest_two = await store.query_events(limit=2)
    user_events = await store.query_events(actor_id="user-1")
    backend_events = await store.query_events(profile_id="backend")
    denied_events = await store.query_events(event_type="tool.denied")
    old_event = [event for event in user_events if event.id == "old"][0]

    assert [event.id for event in newest_two] == ["new", "middle"]
    assert await store.query_events(limit=0) == []
    with pytest.raises(ValueError, match="limit must be non-negative"):
        await store.query_events(limit=-1)
    assert [event.id for event in user_events] == ["new", "old"]
    assert [event.id for event in backend_events] == ["new", "old"]
    assert [event.id for event in denied_events] == ["middle"]
    assert old_event.payload["args"]["path"] == "/repo/README.md"

    await store.aclose()


@pytest.mark.asyncio
async def test_sqlite_store_orders_audit_events_by_utc_instant(
    tmp_path: Path,
) -> None:
    from mcp_unified.storage import AuditEvent, SQLiteMCPStore

    store = SQLiteMCPStore(tmp_path / "mcp.sqlite")
    older = datetime(
        2026,
        5,
        29,
        0,
        15,
        tzinfo=timezone(timedelta(hours=2)),
    )
    newer = datetime(
        2026,
        5,
        28,
        23,
        30,
        tzinfo=timezone(timedelta(hours=-2)),
    )

    await store.append_event(
        AuditEvent(id="older-offset", event_type="tool.allowed", created_at=older)
    )
    await store.append_event(
        AuditEvent(id="newer-offset", event_type="tool.allowed", created_at=newer)
    )

    events = await store.query_events()

    assert [event.id for event in events] == ["newer-offset", "older-offset"]
    assert events[0].created_at.utcoffset() == timedelta(0)
    assert events[0].created_at.isoformat() == "2026-05-29T01:30:00+00:00"

    await store.aclose()
