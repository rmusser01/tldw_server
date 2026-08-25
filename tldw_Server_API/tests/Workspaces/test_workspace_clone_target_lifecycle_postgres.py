"""PostgreSQL integration coverage for staged Workspace clone targets."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    ConflictError,
)

pytestmark = [pytest.mark.integration, pytest.mark.timeout(60)]


@pytest.fixture
def postgres_db(pg_database_config: DatabaseConfig) -> Iterator[CharactersRAGDB]:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(":memory:", client_id="user-1", backend=backend)
    try:
        yield db
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()


def _reserve(
    db: CharactersRAGDB,
    *,
    workspace_id: str = "workspace-target",
    operation_id: str = "operation-1",
    request_fingerprint: str = "fingerprint-1",
    name: str = "Target Workspace",
    description: str | None = "Cloned workspace",
    workspace_profile: str = "research",
) -> dict[str, object]:
    return db.reserve_clone_target(
        workspace_id=workspace_id,
        operation_id=operation_id,
        request_fingerprint=request_fingerprint,
        name=name,
        description=description,
        workspace_profile=workspace_profile,
    )


def test_postgres_reserve_publish_replay_and_confirm_are_hidden_and_fenced(
    postgres_db: CharactersRAGDB,
) -> None:
    postgres_db.upsert_workspace("workspace-ordinary", "Ordinary Workspace")

    with pytest.raises(ConflictError):
        _reserve(postgres_db, workspace_id="workspace-ordinary")

    reserved = _reserve(postgres_db, name="  Target   Workspace  ")
    replayed = _reserve(postgres_db, name="Target Workspace")

    assert replayed == reserved
    assert bool(reserved["archived"]) is True
    assert reserved["system_operation_state"] == "staged"
    assert postgres_db.get_workspace("workspace-target") is None
    assert {row["id"] for row in postgres_db.list_workspaces()} == {"workspace-ordinary"}

    with pytest.raises(ConflictError):
        _reserve(postgres_db, operation_id="operation-2")

    published = postgres_db.publish_clone_target(
        workspace_id="workspace-target",
        operation_id="operation-1",
    )
    pending_replay = _reserve(postgres_db)

    assert pending_replay == published
    assert bool(pending_replay["archived"]) is False
    assert pending_replay["system_operation_state"] == "publication_pending"
    assert postgres_db.get_workspace("workspace-target") is None
    assert {row["id"] for row in postgres_db.list_workspaces()} == {"workspace-ordinary"}

    confirmed = postgres_db.confirm_clone_target_publication(
        workspace_id="workspace-target",
        operation_id="operation-1",
    )

    assert confirmed["system_operation_id"] is None
    assert confirmed["system_operation_kind"] is None
    assert confirmed["system_operation_state"] is None
    assert confirmed["system_request_fingerprint"] is None
    assert postgres_db.get_workspace("workspace-target") == confirmed
    assert {row["id"] for row in postgres_db.list_workspaces()} == {
        "workspace-ordinary",
        "workspace-target",
    }


def test_postgres_reconciliation_and_discard_use_bounded_boolean_rowcounts(
    postgres_db: CharactersRAGDB,
) -> None:
    _reserve(postgres_db, workspace_id="workspace-a", operation_id="operation-a")
    _reserve(postgres_db, workspace_id="workspace-b", operation_id="operation-b")
    postgres_db.publish_clone_target(workspace_id="workspace-b", operation_id="operation-b")
    _reserve(postgres_db, workspace_id="workspace-c", operation_id="operation-c")

    limited = postgres_db.list_clone_targets_for_reconciliation(
        operation_ids=["operation-b", "operation-a"],
        limit=1,
    )
    correlated = postgres_db.list_clone_targets_for_reconciliation(
        operation_ids=["operation-b", "operation-a"],
        limit=2,
    )

    assert [(row["id"], row["system_operation_state"]) for row in limited] == [
        ("workspace-a", "staged")
    ]
    assert [(row["id"], row["system_operation_state"]) for row in correlated] == [
        ("workspace-a", "staged"),
        ("workspace-b", "publication_pending"),
    ]

    assert postgres_db.discard_clone_target(
        workspace_id="workspace-a",
        operation_id="operation-other",
    ) is False
    assert postgres_db.discard_clone_target(
        workspace_id="workspace-a",
        operation_id="operation-a",
    ) is True
    assert postgres_db.discard_clone_target(
        workspace_id="workspace-a",
        operation_id="operation-a",
    ) is False
    assert postgres_db.list_clone_targets_for_reconciliation(
        operation_ids=["operation-a"],
    ) == []

    assert postgres_db.discard_clone_target(
        workspace_id="workspace-b",
        operation_id="operation-b",
    ) is True
    remaining = postgres_db.list_clone_targets_for_reconciliation(
        operation_ids=["operation-b", "operation-c"],
    )

    assert [(row["id"], row["system_operation_state"]) for row in remaining] == [
        ("workspace-c", "staged")
    ]
