from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.integration


SCRIPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "Helper_Scripts"
    / "Testing-related"
    / "chatbooks_full_account_uat_fixture.py"
)


def _load_fixture_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("chatbooks_full_account_uat_fixture", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load fixture helper: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _effective_auth_state(db_path: Path, user_id: int) -> tuple[list[str], list[str]]:
    with sqlite3.connect(db_path) as conn:
        roles = [
            str(row[0])
            for row in conn.execute(
                """
                SELECT r.name
                FROM roles r
                JOIN user_roles ur ON ur.role_id = r.id
                WHERE ur.user_id = ?
                ORDER BY r.name
                """,
                (user_id,),
            ).fetchall()
        ]
        permissions = [
            str(row[0])
            for row in conn.execute(
                """
                SELECT DISTINCT p.name
                FROM permissions p
                JOIN role_permissions rp ON rp.permission_id = p.id
                JOIN user_roles ur ON ur.role_id = rp.role_id
                WHERE ur.user_id = ?
                  AND NOT EXISTS (
                      SELECT 1
                      FROM user_permissions up
                      WHERE up.user_id = ur.user_id
                        AND up.permission_id = p.id
                        AND up.granted = 0
                        AND (up.expires_at IS NULL OR up.expires_at > CURRENT_TIMESTAMP)
                  )
                ORDER BY p.name
                """,
                (user_id,),
            ).fetchall()
        ]
    return roles, permissions


@pytest.mark.asyncio
async def test_prepare_fails_when_user_lacks_effective_notification_permission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _load_fixture_module()

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

    roles, _ = _effective_auth_state(tmp_path / "source" / "users.db", 1)
    assert roles == ["user"]


@pytest.mark.asyncio
async def test_prepare_and_reset_create_separate_source_and_empty_destination(tmp_path: Path) -> None:
    fixture = _load_fixture_module()

    prepared = await fixture.prepare(tmp_path)
    archive_path = Path(prepared["archive_path"])
    expected_path = Path(prepared["expected_path"])
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    roles, permissions = _effective_auth_state(
        tmp_path / "source" / "users.db",
        int(prepared["source_user_id"]),
    )

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
        tmp_path / "destination" / "users.db",
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
