"""Tests for the standalone MCP policy grant stores (approval leases + TTL grants)."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_policy_grant_package_exposes_models_and_factory() -> None:
    from mcp_unified.policy_grants import (
        InMemoryPolicyGrantStore,
        PolicyGrant,
        PolicyGrantStore,
        create_policy_grant_store,
    )

    assert PolicyGrant is not None
    assert PolicyGrantStore is not None
    assert InMemoryPolicyGrantStore is not None
    assert callable(create_policy_grant_store)


def test_memory_store_creates_and_finds_normalized_approval_grant() -> None:
    from mcp_unified.policy_grants import InMemoryPolicyGrantStore

    store = InMemoryPolicyGrantStore()
    grant = store.create_grant(
        profile_id="researcher",
        grant_type="approval",
        subject_type="domain",
        value="https://Example.com/private",
        ttl_seconds=900,
        granted_by="operator",
        reason="one-off research fetch",
    )

    assert grant.grant_id
    assert grant.value == "example.com"
    assert grant.effect == "allow"
    assert grant.ttl_seconds == 900

    found = store.find_active_grant(
        profile_id="researcher",
        grant_type="approval",
        subject_type="domain",
        value="example.com",
    )
    assert found is not None
    assert found.grant_id == grant.grant_id


def test_memory_store_find_misses_on_mismatched_dimensions() -> None:
    from mcp_unified.policy_grants import InMemoryPolicyGrantStore

    store = InMemoryPolicyGrantStore()
    store.create_grant(
        profile_id="researcher",
        grant_type="approval",
        subject_type="domain",
        value="example.com",
        ttl_seconds=900,
    )

    assert (
        store.find_active_grant(
            profile_id="other-profile",
            grant_type="approval",
            subject_type="domain",
            value="example.com",
        )
        is None
    )
    assert (
        store.find_active_grant(
            profile_id="researcher",
            grant_type="path",
            subject_type="path",
            value="example.com",
        )
        is None
    )
    assert (
        store.find_active_grant(
            profile_id="researcher",
            grant_type="approval",
            subject_type="domain",
            value="other.example.org",
        )
        is None
    )


def test_memory_store_expired_grant_is_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    import mcp_unified.policy_grants.memory as memory_grants
    from mcp_unified.policy_grants import InMemoryPolicyGrantStore

    store = InMemoryPolicyGrantStore()
    monkeypatch.setattr(memory_grants.time, "time", lambda: 1_000.0)
    store.create_grant(
        profile_id="researcher",
        grant_type="approval",
        subject_type="domain",
        value="example.com",
        ttl_seconds=10,
    )

    monkeypatch.setattr(memory_grants.time, "time", lambda: 1_011.0)
    assert (
        store.find_active_grant(
            profile_id="researcher",
            grant_type="approval",
            subject_type="domain",
            value="example.com",
        )
        is None
    )
    assert store.list_active_grants(profile_id="researcher") == []


def test_memory_store_session_scoped_grant_only_matches_its_session() -> None:
    from mcp_unified.policy_grants import InMemoryPolicyGrantStore

    store = InMemoryPolicyGrantStore()
    store.create_grant(
        profile_id="researcher",
        grant_type="approval",
        subject_type="domain",
        value="example.com",
        ttl_seconds=900,
        session_id="session-1",
    )

    assert (
        store.find_active_grant(
            profile_id="researcher",
            grant_type="approval",
            subject_type="domain",
            value="example.com",
            session_id="session-1",
        )
        is not None
    )
    assert (
        store.find_active_grant(
            profile_id="researcher",
            grant_type="approval",
            subject_type="domain",
            value="example.com",
            session_id="session-2",
        )
        is None
    )
    assert (
        store.find_active_grant(
            profile_id="researcher",
            grant_type="approval",
            subject_type="domain",
            value="example.com",
        )
        is None
    )


def test_memory_store_global_grant_matches_any_session() -> None:
    from mcp_unified.policy_grants import InMemoryPolicyGrantStore

    store = InMemoryPolicyGrantStore()
    store.create_grant(
        profile_id="researcher",
        grant_type="approval",
        subject_type="tool",
        value="web.fetch",
        ttl_seconds=900,
    )

    for session_id in (None, "session-1", "session-2"):
        found = store.find_active_grant(
            profile_id="researcher",
            grant_type="approval",
            subject_type="tool",
            value="web.fetch",
            session_id=session_id,
        )
        assert found is not None


def test_memory_store_revoke_deactivates_grant() -> None:
    from mcp_unified.policy_grants import InMemoryPolicyGrantStore

    store = InMemoryPolicyGrantStore()
    grant = store.create_grant(
        profile_id="researcher",
        grant_type="approval",
        subject_type="domain",
        value="example.com",
        ttl_seconds=900,
    )

    revoked = store.revoke_grant(grant.grant_id)
    assert revoked is not None
    assert revoked.grant_id == grant.grant_id
    assert (
        store.find_active_grant(
            profile_id="researcher",
            grant_type="approval",
            subject_type="domain",
            value="example.com",
        )
        is None
    )
    assert store.revoke_grant("missing-grant-id") is None


def test_memory_store_list_active_grants_filters() -> None:
    from mcp_unified.policy_grants import InMemoryPolicyGrantStore

    store = InMemoryPolicyGrantStore()
    store.create_grant(
        profile_id="researcher",
        grant_type="approval",
        subject_type="domain",
        value="example.com",
        ttl_seconds=900,
    )
    store.create_grant(
        profile_id="researcher",
        grant_type="path",
        subject_type="path",
        value="docs/scratch",
        actions=("read", "write"),
        ttl_seconds=900,
    )
    store.create_grant(
        profile_id="backend",
        grant_type="approval",
        subject_type="tool",
        value="web.fetch",
        ttl_seconds=900,
    )

    assert len(store.list_active_grants()) == 3
    assert len(store.list_active_grants(profile_id="researcher")) == 2
    approvals = store.list_active_grants(profile_id="researcher", grant_type="approval")
    assert len(approvals) == 1
    assert approvals[0].subject_type == "domain"


def test_memory_store_rejects_invalid_grants() -> None:
    from mcp_unified.policy_grants import InMemoryPolicyGrantStore

    store = InMemoryPolicyGrantStore()
    with pytest.raises(ValueError):
        store.create_grant(
            profile_id="researcher",
            grant_type="unknown",
            subject_type="domain",
            value="example.com",
            ttl_seconds=900,
        )
    with pytest.raises(ValueError):
        store.create_grant(
            profile_id="researcher",
            grant_type="approval",
            subject_type="skill",
            value="anything",
            ttl_seconds=900,
        )
    with pytest.raises(ValueError):
        store.create_grant(
            profile_id="   ",
            grant_type="approval",
            subject_type="domain",
            value="example.com",
            ttl_seconds=900,
        )
    with pytest.raises(ValueError):
        store.create_grant(
            profile_id="researcher",
            grant_type="path",
            subject_type="domain",
            value="docs/scratch",
            ttl_seconds=900,
        )


def test_policy_grant_safe_payload_includes_expiry_metadata() -> None:
    from mcp_unified.policy_grants import InMemoryPolicyGrantStore

    store = InMemoryPolicyGrantStore()
    grant = store.create_grant(
        profile_id="researcher",
        grant_type="approval",
        subject_type="domain",
        value="example.com",
        ttl_seconds=900,
        session_id="session-1",
        granted_by="operator",
    )

    payload = grant.safe_payload()
    assert payload["grant_id"] == grant.grant_id
    assert payload["profile_id"] == "researcher"
    assert payload["grant_type"] == "approval"
    assert payload["subject_type"] == "domain"
    assert payload["value"] == "example.com"
    assert payload["session_id"] == "session-1"
    assert payload["ttl_seconds"] == 900
    assert payload["expires_at"].endswith("+00:00")


def test_sqlite_store_persists_grants_across_instances(tmp_path: Path) -> None:
    from mcp_unified.policy_grants.sqlite import SQLitePolicyGrantStore

    db_path = tmp_path / "grants.db"
    first = SQLitePolicyGrantStore(db_path)
    try:
        grant = first.create_grant(
            profile_id="researcher",
            grant_type="approval",
            subject_type="domain",
            value="https://Example.com/private",
            ttl_seconds=900,
            session_id="session-1",
            granted_by="operator",
            reason="one-off research fetch",
        )
        assert grant.value == "example.com"
    finally:
        first.close()

    second = SQLitePolicyGrantStore(db_path)
    try:
        found = second.find_active_grant(
            profile_id="researcher",
            grant_type="approval",
            subject_type="domain",
            value="example.com",
            session_id="session-1",
        )
        assert found is not None
        assert found.grant_id == grant.grant_id
        assert found.granted_by == "operator"

        revoked = second.revoke_grant(grant.grant_id)
        assert revoked is not None
        assert (
            second.find_active_grant(
                profile_id="researcher",
                grant_type="approval",
                subject_type="domain",
                value="example.com",
                session_id="session-1",
            )
            is None
        )
    finally:
        second.close()


def test_sqlite_store_expired_grant_is_not_found(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mcp_unified.policy_grants.sqlite as sqlite_grants
    from mcp_unified.policy_grants.sqlite import SQLitePolicyGrantStore

    store = SQLitePolicyGrantStore(tmp_path / "grants.db")
    try:
        monkeypatch.setattr(sqlite_grants.time, "time", lambda: 1_000.0)
        store.create_grant(
            profile_id="researcher",
            grant_type="approval",
            subject_type="domain",
            value="example.com",
            ttl_seconds=10,
        )

        monkeypatch.setattr(sqlite_grants.time, "time", lambda: 1_011.0)
        assert (
            store.find_active_grant(
                profile_id="researcher",
                grant_type="approval",
                subject_type="domain",
                value="example.com",
            )
            is None
        )
        assert store.list_active_grants(profile_id="researcher") == []
    finally:
        store.close()


def test_policy_grant_store_factory_backend_selection(tmp_path: Path) -> None:
    from mcp_unified.policy_grants import InMemoryPolicyGrantStore, create_policy_grant_store
    from mcp_unified.policy_grants.sqlite import SQLitePolicyGrantStore

    assert isinstance(create_policy_grant_store(None), InMemoryPolicyGrantStore)
    assert isinstance(
        create_policy_grant_store({"grant_store_backend": "memory"}),
        InMemoryPolicyGrantStore,
    )

    sqlite_store = create_policy_grant_store(
        {
            "grant_store_backend": "sqlite",
            "grant_store_sqlite_path": str(tmp_path / "grants.db"),
        }
    )
    assert isinstance(sqlite_store, SQLitePolicyGrantStore)
    sqlite_store.close()

    with pytest.raises(ValueError):
        create_policy_grant_store({"grant_store_backend": "sqlite"})
    with pytest.raises(ValueError):
        create_policy_grant_store({"grant_store_backend": "redis"})


def test_memory_store_sweep_eventually_removes_late_expired_grants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mcp_unified.policy_grants.memory as memory_grants
    from mcp_unified.policy_grants import InMemoryPolicyGrantStore

    store = InMemoryPolicyGrantStore(sweep_interval=1, max_sweep_entries=2)
    monkeypatch.setattr(memory_grants.time, "time", lambda: 1_000.0)
    for index in range(3):
        store.create_grant(
            profile_id=f"profile-{index}",
            grant_type="approval",
            subject_type="tool",
            value="web.fetch",
            ttl_seconds=10_000,
        )
    expiring = store.create_grant(
        profile_id="expiring",
        grant_type="approval",
        subject_type="tool",
        value="web.fetch",
        ttl_seconds=10,
    )

    monkeypatch.setattr(memory_grants.time, "time", lambda: 1_050.0)
    for _ in range(4):
        store.list_active_grants()

    assert expiring.grant_id not in store._grants
    assert len(store._grants) == 3


def test_sqlite_store_read_paths_do_not_delete_expired_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mcp_unified.policy_grants.sqlite as sqlite_grants
    from mcp_unified.policy_grants.sqlite import SQLitePolicyGrantStore
    from sqlalchemy import func, select

    store = SQLitePolicyGrantStore(tmp_path / "grants.db", cleanup_interval=1)

    def _row_count() -> int:
        with store._engine.begin() as connection:
            return int(connection.execute(select(func.count()).select_from(store._table)).scalar_one())

    try:
        monkeypatch.setattr(sqlite_grants.time, "time", lambda: 1_000.0)
        store.create_grant(
            profile_id="researcher",
            grant_type="approval",
            subject_type="domain",
            value="example.com",
            ttl_seconds=10,
        )

        monkeypatch.setattr(sqlite_grants.time, "time", lambda: 1_050.0)
        for _ in range(3):
            assert store.list_active_grants(profile_id="researcher") == []
        assert (
            store.find_active_grant(
                profile_id="researcher",
                grant_type="approval",
                subject_type="domain",
                value="example.com",
            )
            is None
        )
        assert _row_count() == 1

        store.create_grant(
            profile_id="researcher",
            grant_type="approval",
            subject_type="tool",
            value="web.fetch",
            ttl_seconds=900,
        )
        assert _row_count() == 1
    finally:
        store.close()
def test_memory_store_normalizes_path_grant_prefix() -> None:
    from mcp_unified.policy_grants import InMemoryPolicyGrantStore

    store = InMemoryPolicyGrantStore()
    grant = store.create_grant(
        profile_id="reviewer",
        grant_type="path",
        subject_type="path",
        value="docs\\scratch/./sub/",
        actions=("read", "write"),
        ttl_seconds=900,
    )
    assert grant.value == "docs/scratch/sub"
    assert grant.actions == ("read", "write")

    found = store.find_active_grant(
        profile_id="reviewer",
        grant_type="path",
        subject_type="path",
        value="docs/scratch/sub",
    )
    assert found is not None


def test_memory_store_rejects_unsafe_path_grant_values() -> None:
    from mcp_unified.policy_grants import InMemoryPolicyGrantStore

    store = InMemoryPolicyGrantStore()
    for unsafe_value in ("/etc/passwd", "../outside", "C:/windows", "  "):
        with pytest.raises(ValueError):
            store.create_grant(
                profile_id="reviewer",
                grant_type="path",
                subject_type="path",
                value=unsafe_value,
                actions=("read",),
                ttl_seconds=900,
            )


def test_memory_store_rejects_invalid_path_grant_actions() -> None:
    from mcp_unified.policy_grants import InMemoryPolicyGrantStore

    store = InMemoryPolicyGrantStore()
    with pytest.raises(ValueError):
        store.create_grant(
            profile_id="reviewer",
            grant_type="path",
            subject_type="path",
            value="docs/scratch",
            actions=("read", "launch_missiles"),
            ttl_seconds=900,
        )
    with pytest.raises(ValueError):
        store.create_grant(
            profile_id="reviewer",
            grant_type="path",
            subject_type="path",
            value="docs/scratch",
            actions=(),
            ttl_seconds=900,
        )
