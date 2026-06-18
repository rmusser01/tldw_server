from __future__ import annotations

from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    InputError,
)
from tldw_Server_API.app.core.Workspaces import runtime_bindings as runtime_binding_helpers
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


def test_runtime_binding_normalizer_allows_environment_labels() -> None:
    payload = normalize_runtime_binding_payload(
        {
            "binding_id": "repo-main",
            "binding_kind": "repo",
            "owner_domain": "workspaces",
            "locator_ref": "repo-123",
            "status": "ready",
            "portability": "reference",
            "metadata": {"env": "production", "environment": "staging"},
        }
    )

    assert payload["metadata"] == {"env": "production", "environment": "staging"}


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


def test_runtime_binding_normalizer_does_not_redact_nested_non_path_values() -> None:
    payload = normalize_runtime_binding_payload(
        {
            "binding_id": "repo-main",
            "binding_kind": "repo",
            "owner_domain": "workspaces",
            "locator_ref": "repo-123",
            "status": "ready",
            "portability": "reference",
            "metadata": {
                "project_root": {
                    "path": "/Users/example/private/project",
                    "docs_url": "https://example.test/docs/runtime-bindings",
                    "branch": "feature/runtime-bindings",
                },
            },
        }
    )

    assert payload["metadata"]["project_root"]["path"] == "project"
    assert payload["metadata"]["project_root"]["docs_url"] == "https://example.test/docs/runtime-bindings"
    assert payload["metadata"]["project_root"]["branch"] == "feature/runtime-bindings"
    assert payload["redaction_report"]["redacted_fields"] == [
        "metadata.project_root.path",
    ]


def test_runtime_binding_normalizer_ignores_client_redaction_report() -> None:
    payload = normalize_runtime_binding_payload(
        {
            "binding_id": "repo-main",
            "binding_kind": "repo",
            "owner_domain": "workspaces",
            "locator_ref": "repo-123",
            "status": "ready",
            "portability": "reference",
            "metadata": {"branch": "main"},
            "redaction_report": {
                "redacted": True,
                "redacted_fields": ["metadata.branch"],
                "rejected_fields": ["metadata.fake_secret"],
            },
        }
    )

    assert payload["redaction_report"] == {
        "redacted": False,
        "redacted_fields": [],
        "rejected_fields": [],
    }


def test_runtime_binding_json_decode_warning_omits_raw_preview(monkeypatch: pytest.MonkeyPatch) -> None:
    raw_payload = '{"api_key":"sk-should-not-appear",'
    warnings: list[tuple[str, tuple[object, ...]]] = []

    monkeypatch.setattr(
        runtime_binding_helpers.logger,
        "warning",
        lambda message, *args: warnings.append((message, args)),
    )

    assert runtime_binding_helpers.load_runtime_binding_json_object(raw_payload, field_name="metadata_json") == {}

    rendered = "\n".join(f"{message} {args}" for message, args in warnings)
    assert "metadata_json" in rendered
    assert str(len(raw_payload)) in rendered
    assert "sk-should-not-appear" not in rendered
    assert "api_key" not in rendered


def test_runtime_binding_normalizer_rejects_archived_status_on_write() -> None:
    with pytest.raises(InputError, match="DELETE"):
        normalize_runtime_binding_payload(
            {
                "binding_id": "repo-main",
                "binding_kind": "repo",
                "owner_domain": "workspaces",
                "locator_ref": "repo-123",
                "status": "archived",
                "portability": "reference",
            }
        )


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
