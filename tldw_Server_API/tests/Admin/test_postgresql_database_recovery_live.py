"""Real all-user database archive and offline restore into a distinct restricted target."""

import shutil
from types import SimpleNamespace
from uuid import uuid4

import pytest
from Helper_Scripts import pg_backup_restore as cli
from psycopg.conninfo import make_conninfo

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.password_service import PasswordService
from tldw_Server_API.app.core.AuthNZ.settings import Settings, reset_settings
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context
from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB
from tldw_Server_API.app.core.exceptions import BundleExportError
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.services import admin_bundle_service as bundles
from tldw_Server_API.tests._plugins.postgres import pg_restricted_backend as official_restricted_backend

pytestmark = [pytest.mark.integration, pytest.mark.postgres, pytest.mark.asyncio]


@pytest.fixture
def distinct_restricted_target(pg_secondary_database_config):
    yield from official_restricted_backend.__wrapped__(pg_secondary_database_config)


def dsn(config):
    # Official restricted fixture credentials are UUID hex, so this URI is unambiguous.
    return f"postgresql://{config.pg_user}:{config.pg_password}@{config.pg_host}:{config.pg_port}/{config.pg_database}"


def backup_dsn(config):
    return make_conninfo(
        host=config.pg_host,
        port=config.pg_port,
        dbname=config.pg_database,
        user=config.pg_user,
        password=config.pg_password,
    )


def select_scope(monkeypatch, url):
    for name in ("DATABASE_URL", "TLDW_CONTENT_PG_DSN", "JOBS_DB_URL"):
        monkeypatch.setenv(name, url)
    monkeypatch.setenv("TLDW_CONTENT_DB_BACKEND", "postgresql")
    for name in ("PGHOSTADDR", "PGPORT", "PGSERVICE", "PGOPTIONS", "CONTENT_DB_MODE"):
        monkeypatch.delenv(name, raising=False)
    reset_settings()


def read_security(backend):
    """Inspect restored security through the DB abstraction before any bootstrap."""
    flags = backend.execute(
        "SELECT c.relname, c.relrowsecurity, c.relforcerowsecurity FROM pg_class c "
        "JOIN pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname='public' "
        "AND c.relkind IN ('r','p') AND c.relrowsecurity ORDER BY c.relname"
    ).rows
    policies = backend.execute(
        "SELECT schemaname, tablename, policyname, permissive, roles, cmd, qual, with_check "
        "FROM pg_policies WHERE schemaname='public' ORDER BY tablename, policyname"
    ).rows
    return flags, policies


@pytest.mark.usefixtures("backup_workspace")
async def test_real_database_bundle_preserves_all_users_owned_media_and_jobs(
    tmp_path,
    monkeypatch,
    pg_restricted_backend,
    pg_database_config,
    distinct_restricted_target,
    pg_secondary_database_config,
):
    source, target = pg_restricted_backend, distinct_restricted_target
    assert source.config.pg_database != target.config.pg_database  # nosec B101 -- separate official databases, never an in-place test restore.
    assert source.config.pg_user != target.config.pg_user  # nosec B101 -- source owner cannot stand in for the target application role.
    source_url, target_url = dsn(source.config), dsn(target.config)
    select_scope(monkeypatch, source_url)
    passwords = PasswordService()
    owners = []
    media_handles = []
    reader_backends = []
    bundles._rate_limit_windows.clear()
    completed = False
    try:
        pool = DatabasePool(Settings(DATABASE_URL=source_url))
        try:
            await pool.initialize()
            users = UsersDB(pool)
            await users.initialize()
            for label in ("alice", "bob"):
                username = "c09_database_" + label + "_" + uuid4().hex[:8]
                secret = passwords.generate_secure_password(username=username)
                await users.create_user(
                    username=username,
                    email=username + "@example.com",
                    password_hash=passwords.hash_password(secret),
                    role="user",
                    is_active=True,
                    is_superuser=False,
                )
                row = await users.get_user_by_username(username)
                owners.append({"user": row, "secret": secret})
        finally:
            await pool.close()
        manager = JobManager(backend="postgres", db_url=source_url)
        for index, owner in enumerate(owners):
            uid = int(owner["user"]["id"])
            with scoped_context(user_id=uid, is_admin=False):
                media = MediaDatabase(str(tmp_path / "unused-source.db"), client_id=str(uid), backend=source)
                media_handles.append(media)
                media_id, media_uuid, _ = media.add_media_with_keywords(
                    title=f"C09 owned source {index}",
                    media_type="document",
                    content=f"Retained database source for owner {index}",
                    keywords=["c09-backup"],
                    owner_user_id=uid,
                    visibility="personal",
                )
                owner["media"] = media.get_media_by_id(media_id)
                owner["media_id"] = media_id
                owner["media_uuid"] = media_uuid
            owner["job"] = manager.create_job(
                domain="c09-backup",
                queue="default",
                job_type="validation",
                payload={"source_uuid": media_uuid, "owner_index": index},
                owner_user_id=str(uid),
                max_retries=1,
            )
        source_security = read_security(source)
        assert any(row["relname"] == "media" and row["relforcerowsecurity"] for row in source_security[0])  # nosec B101 -- the refusal and isolation oracle require actual FORCE RLS.
        # Complete-row pg_dump must refuse the ordinary FORCE-RLS owner; no filtered dump option.
        with pytest.raises(BundleExportError, match="backup_failed:postgresql"):
            bundles.create_bundle(datasets=["postgresql"], user_id=None, admin_user_id=7)
        assert not list((tmp_path / "backups" / "postgresql").glob("*.dump"))  # nosec B101 -- denied output cannot appear as a valid backup.
        monkeypatch.setenv("TLDW_POSTGRES_BACKUP_DSN", backup_dsn(pg_database_config))
        created = bundles.create_bundle(datasets=["postgresql"], user_id=None, admin_user_id=7)
        archive_path = bundles.get_bundle_path(created.bundle_id)
        select_scope(monkeypatch, target_url)
        monkeypatch.setenv("TLDW_POSTGRES_BACKUP_DSN", backup_dsn(pg_secondary_database_config))
        target.get_pool().close_all()
        args = SimpleNamespace(
            bundle_file=archive_path,
            confirm_database=target.config.pg_database,
            confirm_offline=True,
            timeout=180,
            no_drop=False,
        )
        assert cli._restore_bundle(args) == 0  # nosec B101 -- real bounded tools, safety snapshot, offline checks and restricted restore must all complete.
        inspection = DatabaseBackendFactory.create_backend(target.config)
        reader_backends.append(inspection)
        security_matches = read_security(inspection) == source_security
        assert security_matches  # nosec B101 -- no constructor may reinstall a missing restored policy before this oracle.
        for url, config in ((source_url, source.config), (target_url, target.config)):
            pool = DatabasePool(Settings(DATABASE_URL=url))
            try:
                await pool.initialize()
                users = UsersDB(pool)
                await users.initialize(ensure_schema=False)
                for owner in owners:
                    row = await users.get_user_by_username(owner["user"]["username"])
                    fields = (
                        "id",
                        "uuid",
                        "username",
                        "email",
                        "role",
                        "is_active",
                        "profile_version",
                        "password_hash",
                    )
                    identity_matches = row is not None and all(row[key] == owner["user"][key] for key in fields)
                    credential_ok = (
                        row is not None and passwords.verify_password(owner["secret"], row["password_hash"])[0]
                    )
                    assert identity_matches and credential_ok  # nosec B101 -- verify credentials without printing them or their hashes.
            finally:
                await pool.close()
            backend = DatabaseBackendFactory.create_backend(config)
            reader_backends.append(backend)
            restored_jobs = JobManager(backend="postgres", db_url=url)
            for index, owner in enumerate(owners):
                uid = int(owner["user"]["id"])
                other = owners[1 - index]
                with scoped_context(user_id=uid, is_admin=False):
                    media = MediaDatabase(str(tmp_path / "unused-reader.db"), client_id=str(uid), backend=backend)
                    media_handles.append(media)
                    row = media.get_media_by_id(owner["media_id"])
                    assert row["uuid"] == owner["media_uuid"] and row["content"] == owner["media"]["content"]  # nosec B101 -- each actual owned Media row survives in both original and restored DB.
                    assert media.get_media_by_id(other["media_id"]) is None  # nosec B101 -- restored FORCE RLS cannot expose the other owner's row.
                job = restored_jobs.get_job(int(owner["job"]["id"]), owner_user_id=str(uid))
                fields = ("uuid", "owner_user_id", "status", "payload", "retry_count", "max_retries")
                assert all(job[key] == owner["job"][key] for key in fields)  # nosec B101 -- queued work and retry state are preserved, never automatically consumed or reset.
                assert restored_jobs.get_job(int(other["job"]["id"]), owner_user_id=str(uid)) is None  # nosec B101 -- normal Jobs ownership reads remain isolated.
        completed = True
    finally:
        for media in media_handles:
            media.close_connection()
        for backend in reader_backends:
            backend.get_pool().close_all()
        bundles._rate_limit_windows.clear()
        reset_settings()
        if completed and (tmp_path / "backups").is_dir():
            shutil.rmtree(tmp_path / "backups")


async def test_official_restricted_fixture_retires_provisioner_before_offline_handoff(pg_restricted_backend):
    from tldw_Server_API.app.core.DB_Management.DB_Backups import assert_postgres_database_offline

    pg_restricted_backend.get_pool().close_all()
    assert_postgres_database_offline(pg_restricted_backend.config, timeout_seconds=15)
