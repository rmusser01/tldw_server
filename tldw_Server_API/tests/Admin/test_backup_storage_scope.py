"""Backup operations must not substitute stale SQLite files for live PG content."""

from __future__ import annotations

import hashlib
import json
import zipfile
from configparser import ConfigParser
from pathlib import Path

import pytest

from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.sqlite_backend import SQLiteBackend
from tldw_Server_API.app.core.exceptions import AdminDataOpsError, BundleExportError, BundleImportError
from tldw_Server_API.app.services import admin_bundle_service as bundles
from tldw_Server_API.app.services import admin_data_ops_service as data_ops

pytestmark = pytest.mark.unit


def seed_sqlite(path: Path, marker: str) -> None:
    backend = SQLiteBackend(DatabaseConfig(backend_type=BackendType.SQLITE, sqlite_path=str(path)))
    try:
        with backend.transaction() as connection:
            backend.create_tables("CREATE TABLE c09_backup_marker (marker TEXT NOT NULL);", connection=connection)
            backend.execute("INSERT INTO c09_backup_marker (marker) VALUES (?)", (marker,), connection=connection)
    finally:
        backend.get_pool().close_all()


def read_marker(path: Path) -> list[dict]:
    backend = SQLiteBackend(DatabaseConfig(backend_type=BackendType.SQLITE, sqlite_path=str(path)))
    try:
        return backend.execute("SELECT marker FROM c09_backup_marker").rows
    finally:
        backend.get_pool().close_all()


@pytest.fixture
def storage_scope(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///" + str(tmp_path / "auth.db"))
    monkeypatch.setenv("TLDW_DB_BACKUP_PATH", str(tmp_path / "backups"))
    monkeypatch.setenv("TLDW_DB_ALLOWED_BASE_DIRS", str(tmp_path))
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "users"))
    monkeypatch.delenv("CONTENT_DB_MODE", raising=False)
    monkeypatch.delenv("JOBS_DB_URL", raising=False)
    monkeypatch.setenv("TLDW_CONTENT_DB_BACKEND", "postgresql")
    monkeypatch.setenv("TLDW_CONTENT_PG_DSN", "postgresql://c09_scope@127.0.0.1:15492/not_a_live_test_database")
    shadow, incoming = tmp_path / "stale-local.db", tmp_path / "incoming.db"
    seed_sqlite(shadow, "stale local content")
    seed_sqlite(incoming, "incoming SQLite content")
    reset_settings()
    try:
        yield tmp_path, shadow, incoming
    finally:
        reset_settings()


@pytest.mark.parametrize("dataset", ["media", "chacha", "evaluations"])
@pytest.mark.parametrize("operation", ["backup", "restore"])
def test_pg_content_operations_refuse_stale_local_files_before_mutation(storage_scope, monkeypatch, dataset, operation):
    root, shadow, incoming = storage_scope
    monkeypatch.setitem(data_ops.DATASET_DB_RESOLVERS, dataset, lambda _owner: shadow)
    before = hashlib.sha256(shadow.read_bytes()).digest()
    with pytest.raises(AdminDataOpsError, match="unsupported_backup_backend"):
        if operation == "backup":
            data_ops.create_backup_snapshot(dataset=dataset, user_id=7, backup_type="full", max_backups=None)
        else:
            data_ops.restore_backup_file(dataset=dataset, user_id=7, backup_path=str(incoming))
    assert hashlib.sha256(shadow.read_bytes()).digest() == before  # nosec B101 -- wrong local store must remain untouched.
    assert not (root / "backups").exists()  # nosec B101 -- reject before artifact or safety-snapshot creation.


@pytest.mark.parametrize("operation", ["backup", "restore"])
def test_media_legacy_pg_mode_refuses_local_file(storage_scope, monkeypatch, operation):
    _, shadow, incoming = storage_scope
    monkeypatch.setenv("TLDW_CONTENT_DB_BACKEND", "sqlite")
    monkeypatch.setenv("CONTENT_DB_MODE", "postgres")
    monkeypatch.setitem(data_ops.DATASET_DB_RESOLVERS, "media", lambda _owner: shadow)
    with pytest.raises(AdminDataOpsError, match="unsupported_backup_backend"):
        if operation == "backup":
            data_ops.create_backup_snapshot(dataset="media", user_id=7, backup_type="full", max_backups=None)
        else:
            data_ops.restore_backup_file(dataset="media", user_id=7, backup_path=str(incoming))


def test_config_only_pg_selection_refuses_local_file(storage_scope, monkeypatch):
    _, shadow, _ = storage_scope
    monkeypatch.delenv("TLDW_CONTENT_DB_BACKEND")
    parser = ConfigParser()
    parser.read_dict({"Database": {"type": "postgresql"}})
    monkeypatch.setattr(data_ops, "load_comprehensive_config", lambda: parser)
    monkeypatch.setitem(data_ops.DATASET_DB_RESOLVERS, "chacha", lambda _owner: shadow)
    with pytest.raises(AdminDataOpsError, match="unsupported_backup_backend"):
        data_ops.create_backup_snapshot(dataset="chacha", user_id=7, backup_type="full", max_backups=None)


def scope_bundle(root, incoming):
    content = incoming.read_bytes()
    manifest = {"manifest_version": 1, "datasets": ["prompts", "media"], "schema_versions": {}, "files": {}}
    archive = root / "input.zip"
    with zipfile.ZipFile(archive, "w") as output:
        for dataset in manifest["datasets"]:
            name = dataset + ".db"
            manifest["files"][name] = {
                "dataset": dataset,
                "size_bytes": str(len(content)),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
            output.writestr(name, content)
        output.writestr("manifest.json", json.dumps(manifest))
    return archive


def test_bundle_export_preflights_every_dataset_before_any_snapshot(storage_scope, monkeypatch):
    root, shadow, _ = storage_scope
    for dataset in ("prompts", "media"):
        monkeypatch.setitem(data_ops.DATASET_DB_RESOLVERS, dataset, lambda _owner: shadow)
    snapshots = []
    snapshot = bundles.create_backup_snapshot

    def record_snapshot(**kwargs):
        snapshots.append(kwargs["dataset"])
        return snapshot(**kwargs)

    monkeypatch.setattr(bundles, "create_backup_snapshot", record_snapshot)
    with pytest.raises(BundleExportError) as failure:
        bundles.create_bundle(datasets=["prompts", "media"], user_id=7, max_backups=1)
    assert failure.value.error_code == "unsupported_backup_backend"  # nosec B101 -- callers need an actionable capability refusal.
    assert snapshots == []  # nosec B101 -- no earlier dataset backup or pruning may run.
    assert not (root / "backups").exists()  # nosec B101 -- reject the entire unsupported request before artifacts.


@pytest.mark.parametrize("dry_run", [False, True])
def test_bundle_import_preflights_storage_before_staging_or_safety(storage_scope, monkeypatch, dry_run):
    root, shadow, incoming = storage_scope
    for dataset in ("prompts", "media"):
        monkeypatch.setitem(data_ops.DATASET_DB_RESOLVERS, dataset, lambda _owner: shadow)
    archive = scope_bundle(root, incoming)
    snapshots = []
    monkeypatch.setattr(bundles, "create_backup_snapshot", lambda **kwargs: snapshots.append(kwargs))
    before = hashlib.sha256(shadow.read_bytes()).digest()
    with pytest.raises(BundleImportError) as failure:
        bundles.import_bundle(file_path=str(archive), user_id=7, dry_run=dry_run)
    assert failure.value.error_code == "unsupported_backup_backend"  # nosec B101 -- dry run must not claim unsupported compatibility.
    assert snapshots == []  # nosec B101 -- no live safety/write path may run.
    assert hashlib.sha256(shadow.read_bytes()).digest() == before  # nosec B101 -- preserve every target on refusal.
    assert not (root / "backups").exists()  # nosec B101 -- reject before extraction/staging.


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["backup", "restore"])
async def test_backup_endpoints_explain_unsupported_storage(storage_scope, monkeypatch, operation):
    from fastapi import HTTPException

    from tldw_Server_API.app.api.v1.endpoints.admin import admin_data_ops as endpoint
    from tldw_Server_API.app.api.v1.schemas.admin_schemas import BackupCreateRequest, BackupRestoreRequest

    async def permitted(*args, **kwargs):
        return None

    monkeypatch.setattr(endpoint, "_enforce_admin_user_scope", permitted)
    with pytest.raises(HTTPException) as failure:
        if operation == "backup":
            await endpoint.create_backup(BackupCreateRequest(dataset="media", user_id=7), None, None)
        else:
            await endpoint.restore_backup(
                "input.db", BackupRestoreRequest(dataset="media", user_id=7, confirm=True), None, None
            )
    assert failure.value.status_code == 422  # nosec B101 -- this is unsupported storage, not a malformed field.
    assert str(failure.value.detail).startswith("unsupported_backup_backend:media:")  # nosec B101 -- expose only the sanitized scope explanation.


@pytest.mark.parametrize("operation", ["backup", "restore"])
def test_shared_audit_refuses_stale_per_user_file(storage_scope, monkeypatch, operation):
    from tldw_Server_API.app.api.v1.API_Deps import Audit_DB_Deps
    from tldw_Server_API.app.core import config

    root, shadow, incoming = storage_scope
    monkeypatch.setitem(config.settings, "AUDIT_STORAGE_MODE", "shared")
    monkeypatch.setitem(config.settings, "AUDIT_STORAGE_ROLLBACK", False)
    assert Audit_DB_Deps._resolve_audit_storage_mode() == "shared"  # nosec B101 -- confirm the actual HTTP storage owner.
    monkeypatch.setitem(data_ops.DATASET_DB_RESOLVERS, "audit", lambda _owner: shadow)
    before = shadow.read_bytes()
    with pytest.raises(AdminDataOpsError, match="unsupported_backup_backend:audit:"):
        if operation == "backup":
            data_ops.create_backup_snapshot(dataset="audit", user_id=7, backup_type="full", max_backups=None)
        else:
            data_ops.restore_backup_file(dataset="audit", user_id=7, backup_path=str(incoming))
    assert shadow.read_bytes() == before  # nosec B101 -- shared scope may not replace a nominal user store.
    assert not (root / "backups").exists()  # nosec B101 -- no misleading per-user artifact.


@pytest.mark.parametrize("other_store", ["content", "jobs"])
@pytest.mark.parametrize(
    "other_dsn",
    [
        "postgres://different_user@localhost:15492/c09_shared",
        "postgresql://another_user@127.0.0.1:15492/ignored?dbname=c09_shared",
        "postgresql://another_user@127.0.0.1:15492/",
    ],
)
def test_authnz_refuses_shared_or_unresolved_pg_database(storage_scope, monkeypatch, other_store, other_dsn):
    monkeypatch.setenv("DATABASE_URL", "postgresql://auth_user@127.0.0.1:15492/c09_shared")
    monkeypatch.setenv("TLDW_CONTENT_DB_BACKEND", "sqlite")
    monkeypatch.delenv("JOBS_DB_URL", raising=False)
    if other_store == "content":
        monkeypatch.setenv("TLDW_CONTENT_DB_BACKEND", "postgresql")
        monkeypatch.setenv("TLDW_CONTENT_PG_DSN", other_dsn)
    else:
        monkeypatch.setenv("JOBS_DB_URL", other_dsn)
    reset_settings()
    with pytest.raises(AdminDataOpsError, match="unsupported_backup_backend:authnz:"):
        data_ops._resolve_dataset_db_path("authnz", None)


@pytest.mark.parametrize("dataset", ["media", "chacha", "prompts", "evaluations", "audit"])
def test_sqlite_dataset_backup_and_restore_still_use_the_recorded_user(storage_scope, monkeypatch, dataset):
    from tldw_Server_API.app.core import config

    root, target, incoming = storage_scope
    monkeypatch.setenv("TLDW_CONTENT_DB_BACKEND", "sqlite")
    monkeypatch.setitem(config.settings, "AUDIT_STORAGE_MODE", "per_user")
    monkeypatch.setitem(config.settings, "AUDIT_STORAGE_ROLLBACK", False)
    monkeypatch.setitem(data_ops.DATASET_DB_RESOLVERS, dataset, lambda _owner: target)
    before = read_marker(target)
    snapshot = data_ops.create_backup_snapshot(dataset=dataset, user_id=7, backup_type="full", max_backups=None)
    data_ops.restore_backup_file(dataset=dataset, user_id=7, backup_path=str(incoming))
    restored = read_marker(target) == read_marker(incoming)
    assert snapshot.user_id == 7 and Path(snapshot.path).is_relative_to(root / "backups/user_7" / dataset)  # nosec B101 -- retain explicit backup ownership.
    assert read_marker(Path(snapshot.path)) == before  # nosec B101 -- the snapshot must contain the original records.
    assert restored  # nosec B101 -- real SQLite restore remains supported.


@pytest.mark.parametrize("rollback", [True, "true", "1"])
def test_audit_rollback_preserves_per_user_admission(storage_scope, monkeypatch, rollback):
    from tldw_Server_API.app.api.v1.API_Deps import Audit_DB_Deps
    from tldw_Server_API.app.core import config

    _, shadow, _ = storage_scope
    monkeypatch.setitem(config.settings, "AUDIT_STORAGE_MODE", "shared")
    monkeypatch.setitem(config.settings, "AUDIT_STORAGE_ROLLBACK", rollback)
    monkeypatch.setitem(data_ops.DATASET_DB_RESOLVERS, "audit", lambda _owner: shadow)
    assert Audit_DB_Deps._resolve_audit_storage_mode() == "per_user"  # nosec B101 -- rollback has precedence in the actual HTTP owner.
    assert data_ops._resolve_dataset_db_path("audit", 7) == (str(shadow), 7)  # nosec B101 -- the same selector must preserve per-user backup access.


@pytest.mark.parametrize("content_from_fields", [False, True])
def test_demonstrably_distinct_pg_database_names_allow_authnz_backup(storage_scope, monkeypatch, content_from_fields):
    auth_dsn = "postgresql://auth_user@127.0.0.1:15492/C09_Auth"
    monkeypatch.setenv("DATABASE_URL", auth_dsn)
    monkeypatch.setenv("JOBS_DB_URL", "postgresql://jobs_user@127.0.0.1:15492/C09_Jobs")
    if content_from_fields:
        monkeypatch.delenv("TLDW_CONTENT_PG_DSN")
        monkeypatch.setenv("TLDW_CONTENT_PG_DATABASE", "C09_Content")
    else:
        monkeypatch.setenv(
            "TLDW_CONTENT_PG_DSN", "postgresql://content_user@127.0.0.1:15492/C09_Content?sslmode=disable"
        )
    reset_settings()
    assert data_ops._resolve_dataset_db_path("authnz", None) == (auth_dsn, None)  # nosec B101 -- preserve dedicated AuthNZ PostgreSQL support without a connection.


@pytest.mark.parametrize(
    "invalid_shape", ["empty_files", "missing_file", "duplicate_dataset", "undeclared_file", "non_object"]
)
@pytest.mark.parametrize("dry_run", [False, True])
def test_incomplete_or_contradictory_manifest_never_reports_success(storage_scope, monkeypatch, invalid_shape, dry_run):
    root, shadow, incoming = storage_scope
    monkeypatch.setenv("TLDW_CONTENT_DB_BACKEND", "sqlite")
    for dataset in ("prompts", "media"):
        monkeypatch.setitem(data_ops.DATASET_DB_RESOLVERS, dataset, lambda _owner: shadow)
    archive = scope_bundle(root, incoming)
    with zipfile.ZipFile(archive) as original:
        entries = {name: original.read(name) for name in original.namelist()}
    manifest = json.loads(entries["manifest.json"])
    if invalid_shape == "empty_files":
        manifest["files"] = {}
    elif invalid_shape == "missing_file":
        del manifest["files"]["media.db"]
    elif invalid_shape == "duplicate_dataset":
        manifest["datasets"].append("prompts")
    elif invalid_shape == "undeclared_file":
        manifest["datasets"] = ["prompts"]
    else:
        manifest = []
    entries["manifest.json"] = json.dumps(manifest).encode()
    with zipfile.ZipFile(archive, "w") as modified:
        for name, value in entries.items():
            modified.writestr(name, value)
    snapshots = []
    monkeypatch.setattr(bundles, "create_backup_snapshot", lambda **kwargs: snapshots.append(kwargs))
    with pytest.raises(BundleImportError) as failure:
        bundles.import_bundle(file_path=str(archive), user_id=7, dry_run=dry_run)
    assert failure.value.error_code == "invalid_bundle_manifest"  # nosec B101 -- compatibility/import requires exact complete dataset coverage.
    assert snapshots == []  # nosec B101 -- reject before safety snapshots or mutation.
    assert not (root / "backups").exists()  # nosec B101 -- malformed coverage cannot reach staging.


def test_normalized_export_dataset_cannot_bypass_preflight(storage_scope, monkeypatch):
    root, shadow, _ = storage_scope
    for dataset in ("prompts", "media"):
        monkeypatch.setitem(data_ops.DATASET_DB_RESOLVERS, dataset, lambda _owner: shadow)
    with pytest.raises(BundleExportError) as failure:
        bundles.create_bundle(datasets=["prompts", "MEDIA"], user_id=7)
    assert failure.value.error_code == "unsupported_backup_backend"  # nosec B101 -- accepted name normalization must retain scope admission.
    assert not (root / "backups").exists()  # nosec B101 -- no partial snapshot before the normalized refusal.


def test_postgres_cli_target_matches_decoded_connection_query():
    config = data_ops._config_from_postgres_url(
        "postgresql://auth%5Fuser:fixture%2Fpassword@localhost:15492/ignored?host=127.0.0.1&dbname=c09%5Fauth&sslmode=require"
    )
    correct = (
        config.pg_host,
        config.pg_port,
        config.pg_database,
        config.pg_user,
        config.pg_password,
        config.pg_sslmode,
    ) == ("127.0.0.1", 15492, "c09_auth", "auth_user", "fixture/password", "require")
    assert correct  # nosec B101 -- avoid printing credential-bearing configuration even on failure.


@pytest.mark.parametrize(
    "dsn",
    [
        "postgresql://auth@127.0.0.1:15492/",
        "postgresql://127.0.0.1:15492/c09_auth",
        "postgresql://auth@127.0.0.1:15492/c09_auth?service=unresolved",
        "postgresql://auth@127.0.0.1:bad/c09_auth",
    ],
)
def test_authnz_unknown_cli_identity_is_refused_without_other_pg_stores(storage_scope, monkeypatch, dsn):
    monkeypatch.setenv("TLDW_CONTENT_DB_BACKEND", "sqlite")
    monkeypatch.setenv("DATABASE_URL", dsn)
    reset_settings()
    with pytest.raises(AdminDataOpsError, match="unsupported_backup_backend:authnz:"):
        data_ops._resolve_dataset_db_path("authnz", None)


def test_pg_tools_follow_decoded_target_and_required_ssl(storage_scope, monkeypatch):
    from types import SimpleNamespace

    from tldw_Server_API.app.core.DB_Management import DB_Backups

    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql://auth%5Fuser:fixture%2Fpassword@localhost:15492/ignored?host=127.0.0.1&dbname=c09%5Fauth&sslmode=require",
    )
    reset_settings()
    monkeypatch.setattr(DB_Backups.shutil, "which", lambda tool: "/fixture/bin/" + tool)
    commands = []

    def tool_boundary(command, *, env, capture_output, text):
        commands.append((command, env.get("PGSSLMODE"), env.get("PGPASSWORD") == "fixture/password"))
        if command[0].endswith("pg_dump"):
            Path(command[command.index("-f") + 1]).write_bytes(b"PGDMP synthetic tool boundary")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(DB_Backups.subprocess, "run", tool_boundary)
    snapshot = data_ops.create_backup_snapshot(dataset="authnz", user_id=None, backup_type="full", max_backups=None)
    data_ops.restore_backup_file(dataset="authnz", user_id=None, backup_path=snapshot.path)
    correct = len(commands) == 2 and all(
        command[command.index("-h") + 1] == "127.0.0.1"
        and command[command.index("-U") + 1] == "auth_user"
        and sslmode == "require"
        and password_ok
        for command, sslmode, password_ok in commands
    )
    assert correct  # nosec B101 -- command targeting and transport security must match admission; no credential output.
    assert commands[0][0][-1] == "c09_auth" and commands[1][0][commands[1][0].index("-d") + 1] == "c09_auth"  # nosec B101 -- dump and restore use the same decoded database.


@pytest.mark.parametrize("datasets", [["prompts", "prompts"], ["prompts", " PROMPTS "]])
def test_duplicate_export_scope_is_refused_before_artifacts(storage_scope, monkeypatch, datasets):
    root, shadow, _ = storage_scope
    monkeypatch.setitem(data_ops.DATASET_DB_RESOLVERS, "prompts", lambda _owner: shadow)
    with pytest.raises(BundleExportError) as failure:
        bundles.create_bundle(datasets=datasets, user_id=7)
    assert failure.value.error_code == "duplicate_datasets"  # nosec B101 -- a created bundle must be compatible with exact manifest coverage.
    assert not (root / "backups").exists()  # nosec B101 -- reject duplicate scope before creating or pruning backups.


@pytest.mark.parametrize("override_field", ["database", "password"])
def test_content_pg_fields_use_effective_libpq_database_identity(storage_scope, monkeypatch, override_field):
    monkeypatch.setenv("DATABASE_URL", "postgresql://auth_user@127.0.0.1:15492/c09_shared")
    monkeypatch.delenv("TLDW_CONTENT_PG_DSN")
    monkeypatch.setenv(
        "TLDW_CONTENT_PG_DATABASE", "c09_shared " if override_field == "database" else "nominally_separate"
    )
    if override_field == "password":
        monkeypatch.setenv("TLDW_CONTENT_PG_PASSWORD", "fixture dbname=c09_shared")
    reset_settings()
    with pytest.raises(AdminDataOpsError, match="unsupported_backup_backend:authnz:"):
        data_ops._resolve_dataset_db_path("authnz", None)


@pytest.mark.parametrize("explicit", [None, "postgresql://explicit_user@localhost:15492/explicit_db"])
def test_shared_pg_config_conninfo_preserves_runtime_precedence(explicit):
    config = DatabaseConfig(  # nosec B106 -- fixed non-credential parser input; PostgreSQL transport is not contacted.
        backend_type=BackendType.POSTGRESQL,
        connection_string=explicit,
        pg_host="localhost",
        pg_port=15492,
        pg_database="C09_Content",
        pg_user="fixture_user",
        pg_password="fixture",
        pg_sslmode="require",
        connect_timeout=3,
    )
    expected = (
        explicit
        or "host=localhost port=15492 dbname=C09_Content user=fixture_user password=fixture sslmode=require connect_timeout=3"
    )
    assert config.postgres_connection_string() == expected  # nosec B101 -- both runtime paths and scope admission must use the same construction.


@pytest.mark.parametrize("explicit", [None, "postgresql://explicit_user@localhost:15492/explicit_db"])
def test_direct_pg_connection_forwards_the_shared_effective_conninfo(monkeypatch, explicit):
    from types import SimpleNamespace

    from tldw_Server_API.app.core.DB_Management.backends import postgresql_backend

    seen = []
    expected = (
        explicit
        or "host=localhost port=15492 dbname=C09_Content user=fixture_user password=fixture sslmode=require connect_timeout=3"
    )
    config = DatabaseConfig(  # nosec B106 -- fixed non-credential parser input; PostgreSQL transport is not contacted.
        backend_type=BackendType.POSTGRESQL,
        connection_string=explicit,
        pg_host="localhost",
        pg_port=15492,
        pg_database="C09_Content",
        pg_user="fixture_user",
        pg_password="fixture",
        pg_sslmode="require",
        connect_timeout=3,
    )

    def transport_boundary(dsn):
        seen.append(dsn)
        raise ConnectionError("controlled no-network boundary")

    monkeypatch.setattr(postgresql_backend, "PSYCOPG2_AVAILABLE", True)
    monkeypatch.setattr(postgresql_backend, "psycopg", SimpleNamespace(connect=transport_boundary))
    with pytest.raises(ConnectionError, match="controlled no-network boundary"):
        postgresql_backend.PostgreSQLBackend(config).connect()
    assert seen == [expected]  # nosec B101 -- actual direct-connect caller must forward the same effective target.
