from __future__ import annotations

from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    InputError,
)
from tldw_Server_API.app.core.Workspaces.runtime_bindings import (
    normalize_runtime_binding_payload,
)


def test_runtime_binding_normalizer_redacts_path_and_preserves_safe_metadata() -> None:
    payload = normalize_runtime_binding_payload(
        {
            "binding_id": "repo-main",
            "binding_kind": "repo",
            "owner_domain": "workspaces",
            "locator_ref": "repo-123",
            "label": "Main Repo",
            "status": "ready",
            "path_hint": "/Users/example/private/project",
            "portability": "reference",
            "metadata": {"branch": "main", "remote": "origin"},
        }
    )

    assert payload["path_hint"] == "project"
    assert payload["metadata"] == {"branch": "main", "remote": "origin"}
    assert payload["redaction_report"]["redacted_fields"] == ["path_hint"]


def test_runtime_binding_normalizer_normalizes_contract_aliases() -> None:
    payload = normalize_runtime_binding_payload(
        {
            "binding_id": "sandbox-root",
            "binding_kind": "sandbox_root",
            "owner_domain": "sandbox",
            "locator_ref": "sandbox-123",
            "status": "inspect_only",
            "portability": "metadata_only",
        }
    )

    assert payload["status"] == "inspect-only"
    assert payload["portability"] == "metadata-only"


def test_runtime_binding_normalizer_rejects_secret_metadata_keys() -> None:
    with pytest.raises(InputError):
        normalize_runtime_binding_payload(
            {
                "binding_id": "acp-session",
                "binding_kind": "acp_session",
                "owner_domain": "acp",
                "locator_ref": "session-123",
                "label": "ACP Session",
                "status": "ready",
                "portability": "metadata-only",
                "metadata": {"OPENAI_API_KEY": "sk-secret"},
            }
        )


def test_runtime_binding_normalizer_redacts_path_metadata_values() -> None:
    payload = normalize_runtime_binding_payload(
        {
            "binding_id": "sandbox-root",
            "binding_kind": "sandbox_root",
            "owner_domain": "sandbox",
            "locator_ref": "sandbox-123",
            "status": "ready",
            "portability": "metadata-only",
            "metadata": {
                "absolute_root": "/Users/example/private/project",
                "nested": {"mount_path": "client/acme/repo"},
            },
        }
    )

    assert payload["metadata"]["absolute_root"] == "project"
    assert payload["metadata"]["nested"]["mount_path"] == "repo"
    assert payload["redaction_report"]["redacted_fields"] == [
        "metadata.absolute_root",
        "metadata.nested.mount_path",
    ]


def test_workspace_runtime_binding_upsert_list_get_and_archive(tmp_path: Path) -> None:
    db = CharactersRAGDB(db_path=str(tmp_path / "chacha.db"), client_id="user-1")
    try:
        db.upsert_workspace("ws-1", "Workspace")

        created = db.upsert_workspace_runtime_binding(
            "ws-1",
            {
                "binding_id": "repo-main",
                "binding_kind": "repo",
                "owner_domain": "workspaces",
                "locator_ref": "repo-123",
                "label": "Main Repo",
                "status": "ready",
                "path_hint": "/Users/example/project",
                "portability": "reference",
                "metadata": {"branch": "main"},
            },
            user_id="user-1",
        )

        assert created["path_hint"] == "project"
        assert created["metadata"] == {"branch": "main"}
        assert db.get_workspace_runtime_binding("ws-1", "repo-main")["binding_id"] == "repo-main"
        assert [item["binding_id"] for item in db.list_workspace_runtime_bindings("ws-1")] == [
            "repo-main"
        ]

        updated = db.upsert_workspace_runtime_binding(
            "ws-1",
            {
                "binding_id": "repo-main",
                "binding_kind": "repo",
                "owner_domain": "workspaces",
                "locator_ref": "repo-123",
                "label": "Main Repo",
                "status": "missing",
                "path_hint": "/Users/example/project",
                "portability": "reference",
                "metadata": {"branch": "dev"},
            },
            user_id="user-1",
        )

        assert updated["status"] == "missing"
        assert updated["metadata"] == {"branch": "dev"}
        assert updated["version"] == created["version"] + 1

        archived = db.archive_workspace_runtime_binding("ws-1", "repo-main", user_id="user-1")
        assert archived["status"] == "archived"
        assert archived["deleted"] in (True, 1)
        assert db.get_workspace_runtime_binding("ws-1", "repo-main") is None
        assert (
            db.get_workspace_runtime_binding("ws-1", "repo-main", include_deleted=True)["status"]
            == "archived"
        )
    finally:
        db.close_connection()


def test_workspace_runtime_binding_filters_by_kind_and_owner(tmp_path: Path) -> None:
    db = CharactersRAGDB(db_path=str(tmp_path / "chacha.db"), client_id="user-1")
    try:
        db.upsert_workspace("ws-1", "Workspace")
        db.upsert_workspace_runtime_binding(
            "ws-1",
            {
                "binding_id": "repo-main",
                "binding_kind": "repo",
                "owner_domain": "workspaces",
                "locator_ref": "repo-123",
                "status": "ready",
                "portability": "reference",
            },
        )
        db.upsert_workspace_runtime_binding(
            "ws-1",
            {
                "binding_id": "acp-session",
                "binding_kind": "acp_session",
                "owner_domain": "acp",
                "locator_ref": "session-123",
                "status": "runtime_missing",
                "portability": "metadata_only",
            },
        )

        assert [
            item["binding_id"]
            for item in db.list_workspace_runtime_bindings("ws-1", binding_kind="repo")
        ] == ["repo-main"]
        assert [
            item["binding_id"]
            for item in db.list_workspace_runtime_bindings("ws-1", owner_domain="acp")
        ] == ["acp-session"]
    finally:
        db.close_connection()
