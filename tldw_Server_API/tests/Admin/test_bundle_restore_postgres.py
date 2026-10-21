"""Real bundle restore between distinct official restricted PostgreSQL fixtures."""

from __future__ import annotations

import hashlib
import json
import uuid
import zipfile
from pathlib import Path

import pytest

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.password_service import PasswordService
from tldw_Server_API.app.core.AuthNZ.settings import Settings, reset_settings
from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB
from tldw_Server_API.app.services import admin_bundle_service as bundles
from tldw_Server_API.tests._plugins.postgres import pg_restricted_backend as official_restricted_backend

pytestmark = [pytest.mark.integration, pytest.mark.postgres, pytest.mark.asyncio]


@pytest.fixture
def separate_target_backend(pg_secondary_database_config):
    # Reuse every official role guard and finally cleanup for this one target.
    yield from official_restricted_backend.__wrapped__(pg_secondary_database_config)


def dsn_for(backend) -> str:
    config = backend.config
    return f"postgresql://{config.pg_user}:{config.pg_password}@{config.pg_host}:{config.pg_port}/{config.pg_database}"


async def read_user(dsn: str, username: str):
    pool = DatabasePool(Settings(DATABASE_URL=dsn))
    try:
        await pool.initialize()
        users = UsersDB(pool)
        await users.initialize()
        return await users.get_user_by_username(username)
    finally:
        await pool.close()


async def test_real_auth_bundle_restores_identity_into_separate_restricted_database(
    tmp_path, monkeypatch, pg_restricted_backend, separate_target_backend
):
    monkeypatch.chdir(tmp_path)
    source, target = pg_restricted_backend, separate_target_backend
    assert source.config.pg_database != target.config.pg_database  # nosec B101 - Separate official targets are mandatory.
    assert source.config.pg_user != target.config.pg_user  # nosec B101 - Distinct restricted roles must remain distinct.
    assert not target.table_exists("users")  # nosec B101 - The one session target must be fresh before mutation.
    monkeypatch.setenv("TLDW_DB_BACKUP_PATH", str(tmp_path / "backups"))
    monkeypatch.setenv("TLDW_DB_ALLOWED_BASE_DIRS", str(tmp_path))
    source_dsn, target_dsn = dsn_for(source), dsn_for(target)
    monkeypatch.setenv("DATABASE_URL", source_dsn)
    reset_settings()
    bundles._rate_limit_windows.clear()
    try:
        username = "c09_restore_identity_" + uuid.uuid4().hex[:8]
        password_service = PasswordService()
        password = password_service.generate_secure_password(username=username)
        pool = DatabasePool(Settings(DATABASE_URL=source_dsn))
        try:
            await pool.initialize()
            users = UsersDB(pool)
            await users.initialize()
            await users.create_user(
                username=username,
                email=username + "@example.com",
                password_hash=password_service.hash_password(password),
                role="user",
                is_active=True,
                is_superuser=False,
            )
            before = await users.get_user_by_username(username)
        finally:
            await pool.close()
        metadata = bundles.create_bundle(datasets=["authnz"], user_id=None, admin_user_id=7)
        archive = Path(bundles.get_bundle_path(metadata.bundle_id))
        with zipfile.ZipFile(archive) as bundle:
            manifest = json.loads(bundle.read("manifest.json"))
            assert manifest["datasets"] == ["authnz"]  # nosec B101 - Verify the declared real dump coverage.
            assert len(manifest["files"]) == 1  # nosec B101 - Exactly one auth-store artifact is expected.
            for filename, entry in manifest["files"].items():
                content = bundle.read(filename)
                assert content.startswith(b"PGDMP")  # nosec B101 - This is an actual custom-format PostgreSQL dump.
                assert len(content) == int(entry["size_bytes"])  # nosec B101 - Real manifest byte count must match.
                assert hashlib.sha256(content).hexdigest() == entry["sha256"]  # nosec B101 - Real manifest hash must match.
        monkeypatch.setenv("DATABASE_URL", target_dsn)
        reset_settings()
        outcome = bundles.import_bundle(file_path=str(archive), user_id=None, admin_user_id=7)
        assert outcome["status"] == "imported" and outcome["datasets_restored"] == ["authnz"]  # nosec B101 - No partial import is accepted.
        restored = await read_user(target_dsn, username)
        preserved = await read_user(source_dsn, username)
        fields = ("id", "uuid", "username", "email", "role", "is_active", "profile_version", "password_hash")
        restored_matches = restored is not None and all(restored[key] == before[key] for key in fields)
        source_matches = preserved is not None and all(preserved[key] == before[key] for key in fields)
        assert restored_matches  # nosec B101 - Verify identity/credential without printing private hashes.
        assert source_matches  # nosec B101 - Verify unchanged source without printing private hashes.
        assert password_service.verify_password(password, restored["password_hash"])[0]  # nosec B101 - Restored credential must be usable by the real verifier.
    finally:
        bundles._rate_limit_windows.clear()
        reset_settings()
