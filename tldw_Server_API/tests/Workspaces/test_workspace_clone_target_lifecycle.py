"""Deterministic staged lifecycle contracts for shared Workspace clone targets."""

from __future__ import annotations

from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    ConflictError,
    InputError,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def db(tmp_path: Path) -> CharactersRAGDB:
    database = CharactersRAGDB(str(tmp_path / "clone-targets.sqlite"), client_id="user-1")
    yield database
    database.close_all_connections()


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


def test_first_reservation_is_staged_archived_and_hidden(db: CharactersRAGDB) -> None:
    reserved = _reserve(db, name="  Target   Workspace  ")

    assert reserved["id"] == "workspace-target"
    assert reserved["name"] == "Target Workspace"
    assert bool(reserved["archived"]) is True
    assert reserved["system_operation_id"] == "operation-1"
    assert reserved["system_operation_kind"] == "shared_workspace_clone"
    assert reserved["system_operation_state"] == "staged"
    assert reserved["system_request_fingerprint"] == "fingerprint-1"
    assert db.get_workspace("workspace-target") is None
    assert db.get_workspace("workspace-target", include_deleted=True) is None
    assert db.list_workspaces() == []


def test_identical_reservation_is_idempotent(db: CharactersRAGDB) -> None:
    first = _reserve(db, name="Target   Workspace")
    second = _reserve(db, name="  Target Workspace  ")

    assert second == first


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("request_fingerprint", "fingerprint-2"),
        ("name", "Different Workspace"),
        ("description", None),
        ("workspace_profile", "project"),
    ],
)
def test_same_operation_reservation_requires_exact_request_match(
    db: CharactersRAGDB,
    field: str,
    value: object,
) -> None:
    _reserve(db)
    kwargs = {field: value}

    with pytest.raises(ConflictError):
        _reserve(db, **kwargs)


def test_reservation_conflicts_with_ordinary_workspace(db: CharactersRAGDB) -> None:
    db.upsert_workspace("workspace-target", "Ordinary Workspace")

    with pytest.raises(ConflictError):
        _reserve(db)


def test_reservation_conflicts_with_another_operation(db: CharactersRAGDB) -> None:
    _reserve(db)

    with pytest.raises(ConflictError):
        _reserve(db, operation_id="operation-2")


def test_publish_moves_owned_staged_target_to_hidden_publication_pending(
    db: CharactersRAGDB,
) -> None:
    _reserve(db)

    published = db.publish_clone_target(
        workspace_id="workspace-target",
        operation_id="operation-1",
    )

    assert bool(published["archived"]) is False
    assert published["system_operation_state"] == "publication_pending"
    assert db.get_workspace("workspace-target") is None
    assert db.list_workspaces() == []


def test_identical_reservation_replays_hidden_publication_pending_target(
    db: CharactersRAGDB,
) -> None:
    _reserve(db)
    published = db.publish_clone_target(
        workspace_id="workspace-target",
        operation_id="operation-1",
    )

    replayed = _reserve(db)

    assert replayed == published
    assert bool(replayed["archived"]) is False
    assert replayed["system_operation_state"] == "publication_pending"
    assert db.get_workspace("workspace-target") is None
    assert db.get_workspace("workspace-target", include_deleted=True) is None
    assert db.list_workspaces() == []


def test_publish_rejects_wrong_operation(db: CharactersRAGDB) -> None:
    _reserve(db)

    with pytest.raises(ConflictError):
        db.publish_clone_target(
            workspace_id="workspace-target",
            operation_id="operation-2",
        )


def test_confirmation_clears_all_markers_and_exposes_workspace(db: CharactersRAGDB) -> None:
    _reserve(db)
    db.publish_clone_target(workspace_id="workspace-target", operation_id="operation-1")

    confirmed = db.confirm_clone_target_publication(
        workspace_id="workspace-target",
        operation_id="operation-1",
    )

    assert bool(confirmed["archived"]) is False
    assert confirmed["system_operation_id"] is None
    assert confirmed["system_operation_kind"] is None
    assert confirmed["system_operation_state"] is None
    assert confirmed["system_request_fingerprint"] is None
    assert db.get_workspace("workspace-target") == confirmed
    assert db.list_workspaces() == [confirmed]


def test_confirmation_requires_exact_owned_publication_pending_row(
    db: CharactersRAGDB,
) -> None:
    _reserve(db)

    with pytest.raises(ConflictError):
        db.confirm_clone_target_publication(
            workspace_id="workspace-target",
            operation_id="operation-1",
        )


@pytest.mark.parametrize("publish_first", [False, True])
def test_discard_soft_deletes_exact_owned_target(
    db: CharactersRAGDB,
    publish_first: bool,
) -> None:
    _reserve(db)
    if publish_first:
        db.publish_clone_target(workspace_id="workspace-target", operation_id="operation-1")

    assert db.discard_clone_target(
        workspace_id="workspace-target",
        operation_id="operation-1",
    ) is True
    row = db.execute_query(
        "SELECT deleted FROM workspaces WHERE id = ?",
        ("workspace-target",),
    ).fetchone()
    assert bool(row["deleted"]) is True


def test_discard_is_operation_fenced(db: CharactersRAGDB) -> None:
    _reserve(db)

    assert db.discard_clone_target(
        workspace_id="workspace-target",
        operation_id="operation-2",
    ) is False
    row = db.execute_query(
        "SELECT deleted, system_operation_state FROM workspaces WHERE id = ?",
        ("workspace-target",),
    ).fetchone()
    assert bool(row["deleted"]) is False
    assert row["system_operation_state"] == "staged"


def test_reconciliation_lookup_is_caller_correlated_and_excludes_deleted_rows(
    db: CharactersRAGDB,
) -> None:
    _reserve(db, workspace_id="workspace-staged", operation_id="operation-staged")
    _reserve(db, workspace_id="workspace-pending", operation_id="operation-pending")
    db.publish_clone_target(workspace_id="workspace-pending", operation_id="operation-pending")
    _reserve(db, workspace_id="workspace-deleted", operation_id="operation-deleted")
    db.discard_clone_target(workspace_id="workspace-deleted", operation_id="operation-deleted")
    _reserve(db, workspace_id="workspace-unrequested", operation_id="operation-unrequested")
    db.upsert_workspace("workspace-ordinary", "Ordinary")

    rows = db.list_clone_targets_for_reconciliation(
        operation_ids=["operation-pending", "operation-staged", "operation-deleted"],
        limit=2,
    )

    assert [(row["system_operation_id"], row["system_operation_state"]) for row in rows] == [
        ("operation-pending", "publication_pending"),
        ("operation-staged", "staged"),
    ]


def test_reconciliation_lookup_accepts_empty_operation_ids_without_scanning(
    db: CharactersRAGDB,
) -> None:
    _reserve(db)

    assert db.list_clone_targets_for_reconciliation(operation_ids=[]) == []


@pytest.mark.parametrize(
    ("operation_ids", "limit"),
    [
        (["operation-1"], 0),
        (["operation-1"], 101),
        (["operation-1"] * 101, 100),
        ("operation-1", 100),
        (["bad operation"], 100),
    ],
)
def test_reconciliation_lookup_rejects_unbounded_or_invalid_inputs(
    db: CharactersRAGDB,
    operation_ids: object,
    limit: int,
) -> None:
    with pytest.raises(InputError):
        db.list_clone_targets_for_reconciliation(
            operation_ids=operation_ids,  # type: ignore[arg-type]
            limit=limit,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("workspace_id", "bad workspace"),
        ("operation_id", ""),
        ("request_fingerprint", "fingerprint\nvalue"),
        ("name", "   "),
        ("name", "x" * 256),
        ("workspace_profile", "other"),
    ],
)
def test_reservation_validates_clone_identity_and_workspace_fields(
    db: CharactersRAGDB,
    field: str,
    value: object,
) -> None:
    kwargs = {field: value}

    with pytest.raises(InputError):
        _reserve(db, **kwargs)
