from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints.notifications import router as notifications_router
from tldw_Server_API.app.core.AuthNZ.jwt_service import create_access_token, reset_jwt_service

pytestmark = pytest.mark.integration


SCRIPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "Helper_Scripts"
    / "Testing-related"
    / "chatbooks_full_account_uat_fixture.py"
)

_NOTIFICATION_ENDPOINTS = (
    ("GET", "/api/v1/notifications", None),
    ("GET", "/api/v1/notifications/unread-count", None),
    ("GET", "/api/v1/notifications/preferences", None),
    ("GET", "/api/v1/notifications/stream", None),
    ("POST", "/api/v1/notifications/mark-read", {"ids": [999_999]}),
    ("POST", "/api/v1/notifications/999999/dismiss", None),
    ("POST", "/api/v1/notifications/999999/snooze", {"minutes": 15}),
    ("DELETE", "/api/v1/notifications/999999/snooze", None),
    ("PATCH", "/api/v1/notifications/preferences", {"reminder_enabled": False}),
)


def _load_fixture_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("chatbooks_full_account_uat_fixture", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load fixture helper: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _effective_auth_state(fixture: ModuleType, user_id: int) -> tuple[list[str], list[str]]:
    repo = fixture.AuthnzRbacRepo(
        client_id=f"chatbooks_full_account_uat_test_{id(fixture)}_{user_id}"
    )
    roles = sorted(str(row["name"]) for row in repo.get_user_roles(user_id))
    permissions = sorted(str(name) for name in repo.get_effective_permissions(user_id))
    return roles, permissions


def _notification_auth_client(user_id: int, role: str = "user") -> tuple[TestClient, dict[str, str]]:
    reset_jwt_service()
    access_token = create_access_token(user_id, "chatbooks-backup-source", role)
    app = FastAPI()
    app.include_router(notifications_router, prefix="/api/v1")
    return TestClient(app), {"Authorization": f"Bearer {access_token}"}


@pytest.mark.asyncio
async def test_prepare_fails_when_user_lacks_effective_notification_permission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _load_fixture_module()
    real_rbac_repo = fixture.AuthnzRbacRepo

    class _MissingNotificationPermissionRepo:
        def __init__(self, *, client_id: str) -> None:  # noqa: ARG002
            pass

        def get_effective_permissions(self, user_id: int) -> list[str]:  # noqa: ARG002
            return ["notifications.read"]

    monkeypatch.setattr(fixture, "AuthnzRbacRepo", _MissingNotificationPermissionRepo, raising=False)

    with pytest.raises(
        RuntimeError,
        match=r"missing required notification permissions: notifications\.control",
    ):
        await fixture.prepare(tmp_path)

    monkeypatch.setattr(fixture, "AuthnzRbacRepo", real_rbac_repo)
    roles, _ = _effective_auth_state(fixture, 1)
    assert roles == ["user"]


@pytest.mark.asyncio
async def test_prepare_and_reset_create_separate_source_and_empty_destination(tmp_path: Path) -> None:
    fixture = _load_fixture_module()

    prepared = await fixture.prepare(tmp_path)
    archive_path = Path(prepared["archive_path"])
    expected_path = Path(prepared["expected_path"])
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    roles, permissions = _effective_auth_state(fixture, int(prepared["source_user_id"]))

    assert archive_path == tmp_path / "source" / "full-account.chatbook"
    assert archive_path.is_file()
    assert expected_path == tmp_path / "expected.json"
    assert expected["source_user_id"] != expected["destination_user_id"]
    assert expected["media"]["artifact_sha256"]
    assert expected["embeddings"]["collection_ids"]
    assert roles == ["user"]
    assert {"notifications.read", "notifications.control"}.issubset(permissions)

    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        manifest = json.loads(archive.read("manifest.json"))
        profile_payload = json.loads(archive.read("json/account_profile.json"))
        settings_payload = json.loads(archive.read("json/account_settings.json"))
        archived_media = archive.read(expected["media"]["archive_path"])
        archived_files = {
            name: archive.read(name)
            for name in names
            if not name.endswith("/")
        }

    assert profile_payload["schema_version"] == "1.0"
    assert profile_payload["profile"]["identity.email"] == expected["profile"]["identity.email"]
    assert settings_payload["schema_version"] == "1.0"
    assert settings_payload["overrides"] == expected["settings"]
    inventory_paths = {entry["path"] for entry in manifest["file_inventory"]}
    assert "json/account_profile.json" in inventory_paths
    assert "json/account_settings.json" in inventory_paths
    assert manifest["account_inventory_summary"]["counts"]["account_profiles"] == 1
    assert manifest["account_inventory_summary"]["counts"]["account_settings"] == 1
    assert expected["profile"]["identity.email"] not in json.dumps(manifest)
    source_password_hash = fixture._fixture_password_hash("chatbooks-backup-source").encode()
    assert all(source_password_hash not in data for data in archived_files.values())
    assert all(str(tmp_path).encode() not in data for data in archived_files.values())
    assert expected["media"]["archive_path"] in names
    assert fixture.sha256_bytes(archived_media) == expected["media"]["artifact_sha256"]

    reset = await fixture.reset_destination(tmp_path)
    destination_roles, destination_permissions = _effective_auth_state(
        fixture,
        int(reset["destination_user_id"]),
    )

    assert reset["destination_user_id"] == expected["destination_user_id"]
    assert reset["counts"] == {
        "characters": 0,
        "media_records": 0,
        "media_stored_artifacts": 0,
        "embeddings": 0,
    }
    assert destination_roles == ["user"]
    assert {"notifications.read", "notifications.control"}.issubset(destination_permissions)
    assert archive_path.is_file(), "reset-destination must not move or copy the source archive"
    with pytest.raises(fixture.FixtureVerificationError, match="destination"):
        await fixture.verify(tmp_path)


@pytest.mark.asyncio
async def test_prepared_standard_user_authorizes_every_notification_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _load_fixture_module()
    prepared = await fixture.prepare(tmp_path)
    source_user_id = int(prepared["source_user_id"])
    monkeypatch.setenv("NOTIFICATIONS_STREAM_POLL_SEC", "0.01")
    monkeypatch.setenv("NOTIFICATIONS_STREAM_HEARTBEAT_SEC", "0.05")
    monkeypatch.setenv("NOTIFICATIONS_STREAM_MAX_DURATION_SEC", "0.05")

    client, headers = _notification_auth_client(source_user_id)
    with client:
        responses = [
            client.request(method, path, json=payload, headers=headers)
            for method, path, payload in _NOTIFICATION_ENDPOINTS
        ]

    assert all(response.status_code not in {401, 403} for response in responses)


@pytest.mark.asyncio
async def test_prepared_user_explicit_notification_deny_takes_precedence(tmp_path: Path) -> None:
    fixture = _load_fixture_module()
    prepared = await fixture.prepare(tmp_path)
    source_user_id = int(prepared["source_user_id"])
    auth_db = tmp_path / "source" / "users.db"

    with sqlite3.connect(auth_db) as conn:
        conn.execute(
            """
            INSERT INTO user_permissions (user_id, permission_id, granted)
            SELECT ?, id, 0
            FROM permissions
            WHERE name = 'notifications.control'
            """,
            (source_user_id,),
        )
        conn.commit()

    roles, permissions = _effective_auth_state(fixture, source_user_id)

    assert roles == ["user"]
    assert "notifications.read" in permissions
    assert "notifications.control" not in permissions

    client, headers = _notification_auth_client(source_user_id)
    with client:
        assert client.get("/api/v1/notifications", headers=headers).status_code == 200
        assert client.get(
            "/api/v1/notifications/unread-count",
            headers=headers,
        ).status_code == 200
        assert client.post(
            "/api/v1/notifications/mark-read",
            json={"ids": [999_999]},
            headers=headers,
        ).status_code == 403


@pytest.mark.asyncio
async def test_prepared_user_restricted_custom_role_has_no_notification_access(
    tmp_path: Path,
) -> None:
    fixture = _load_fixture_module()
    prepared = await fixture.prepare(tmp_path)
    source_user_id = int(prepared["source_user_id"])
    auth_db = tmp_path / "source" / "users.db"

    with sqlite3.connect(auth_db) as conn:
        restricted_role_id = conn.execute(
            """
            INSERT INTO roles (name, description, is_system)
            VALUES ('notification-restricted', 'UAT role without notification access', 0)
            RETURNING id
            """
        ).fetchone()[0]
        conn.execute("DELETE FROM user_roles WHERE user_id = ?", (source_user_id,))
        conn.execute(
            "INSERT INTO user_roles (user_id, role_id) VALUES (?, ?)",
            (source_user_id, restricted_role_id),
        )
        conn.commit()

    roles, permissions = _effective_auth_state(fixture, source_user_id)

    assert roles == ["notification-restricted"]
    assert "notifications.read" not in permissions
    assert "notifications.control" not in permissions

    client, headers = _notification_auth_client(source_user_id, role="notification-restricted")
    with client:
        responses = [
            client.request(method, path, json=payload, headers=headers)
            for method, path, payload in _NOTIFICATION_ENDPOINTS
        ]

    assert all(response.status_code == 403 for response in responses)
