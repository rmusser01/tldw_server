"""Explicit whole-database scope; unit tests never connect to PostgreSQL."""

from pathlib import Path

import pytest

from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
from tldw_Server_API.app.core.exceptions import InvalidBackupUserIdError, UnsupportedBackupBackendError
from tldw_Server_API.app.services import admin_data_ops_service as data_ops

pytestmark = pytest.mark.unit
DSN = "postgresql://c09_database@127.0.0.1:15492/c09_database"


@pytest.fixture
def database_scope(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for key, value in {
        "AUTH_MODE": "single_user",
        "DATABASE_URL": DSN,
        "TLDW_CONTENT_DB_BACKEND": "postgresql",
        "TLDW_CONTENT_PG_DSN": DSN,
        "JOBS_DB_URL": DSN,
        "TLDW_DB_BACKUP_PATH": str(tmp_path / "backups"),
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("CONTENT_DB_MODE", raising=False)
    monkeypatch.delenv("TLDW_POSTGRES_BACKUP_DSN", raising=False)
    reset_settings()
    try:
        yield tmp_path
    finally:
        reset_settings()


def test_database_scope_resolves_explicit_colocated_database_without_a_user(database_scope):
    assert data_ops._resolve_dataset_db_path("postgresql", None) == (DSN, None)  # nosec B101 -- exact global target, never a local file.


def test_database_scope_rejects_user_target_before_artifacts(database_scope):
    with pytest.raises(InvalidBackupUserIdError, match="user_id_forbidden"):
        data_ops.create_backup_snapshot(dataset="postgresql", user_id=7, backup_type="full", max_backups=None)
    assert not (database_scope / "backups").exists()  # nosec B101 -- refuse before artifact creation.


@pytest.mark.parametrize(
    "key,value",
    [
        ("JOBS_DB_URL", "postgresql://c09_jobs@127.0.0.1:15492/another_database"),
        ("JOBS_DB_URL", "postgresql://c09_jobs@localhost:15492/c09_database"),
        ("JOBS_DB_URL", "postgresql://c09_jobs@127.0.0.1:15493/c09_database"),
        ("JOBS_DB_URL", "sqlite:///jobs.db"),
        ("JOBS_DB_URL", ""),
        ("TLDW_CONTENT_PG_DSN", DSN + "?dbname=another_database"),
        ("TLDW_CONTENT_DB_BACKEND", "sqlite"),
    ],
)
def test_database_scope_refuses_unproven_colocation_before_artifacts(database_scope, monkeypatch, key, value):
    monkeypatch.setenv(key, value)
    with pytest.raises(UnsupportedBackupBackendError, match="unsupported_backup_backend:postgresql"):
        data_ops.create_backup_snapshot(dataset="postgresql", user_id=None, backup_type="full", max_backups=None)
    assert not (database_scope / "backups").exists()  # nosec B101 -- no output/pruning can precede scope admission.


def test_database_snapshot_uses_existing_pg_dump_owner_and_global_metadata(database_scope, monkeypatch):
    calls = []

    def dump(backend, backup_dir, *, label, timeout_seconds=None):
        calls.append((backend.config.pg_host, backend.config.pg_port, backend.config.pg_database, label))
        artifact = Path(backup_dir) / "postgresql_test.dump"
        artifact.write_bytes(b"PGDMP" + b"unit-boundary-only")
        return str(artifact)

    monkeypatch.setattr(data_ops, "create_postgres_backup", dump)
    item = data_ops.create_backup_snapshot(dataset="postgresql", user_id=None, backup_type="full", max_backups=None)
    assert calls == [("127.0.0.1", 15492, "c09_database", "postgresql")]  # nosec B101 -- the real tool owner receives the exact admitted target.
    assert item.dataset == "postgresql" and item.user_id is None  # nosec B101 -- whole-database metadata must stay global.
    assert Path(item.path).read_bytes().startswith(b"PGDMP")  # nosec B101 -- unit verifies tool dispatch, not a real backup.


def test_database_restore_refuses_the_online_boundary_before_tool_dispatch(database_scope, monkeypatch):
    called = []
    monkeypatch.setattr(data_ops, "restore_postgres_backup", lambda *args, **kwargs: called.append(True))
    with pytest.raises(UnsupportedBackupBackendError, match="offline_restore_required"):
        data_ops.restore_backup_file(
            dataset="postgresql", user_id=None, backup_path=str(database_scope / "incoming.dump")
        )
    assert called == []  # nosec B101 -- the online boundary cannot dispatch a whole-database overwrite.


@pytest.fixture
def dump_boundary(database_scope, monkeypatch):
    calls = []

    def dump(backend, backup_dir, *, label, timeout_seconds=None):
        calls.append(backend.config)
        artifact = Path(backup_dir) / "postgresql_role_test.dump"
        artifact.write_bytes(b"PGDMP" + b"unit-boundary-only")
        return str(artifact)

    monkeypatch.setattr(data_ops, "create_postgres_backup", dump)
    return calls


def test_database_backup_uses_separate_preprovisioned_principal_only_for_dump(
    database_scope, monkeypatch, dump_boundary
):
    backup_dsn = DSN.replace("c09_database@", "c09_backup@")
    monkeypatch.setenv("TLDW_POSTGRES_BACKUP_DSN", backup_dsn)
    data_ops.create_backup_snapshot(dataset="postgresql", user_id=None, backup_type="full", max_backups=None)
    assert len(dump_boundary) == 1 and dump_boundary[0].pg_user == "c09_backup"  # nosec B101 -- backup privilege is an explicit separate principal.
    assert data_ops._resolve_dataset_db_path("postgresql", None) == (DSN, None)  # nosec B101 -- runtime/restore target credentials stay unchanged.


@pytest.mark.parametrize(
    "backup_dsn", [DSN.replace("/c09_database", "/other_database"), DSN.replace("127.0.0.1", "other.example")]
)
def test_database_backup_principal_cannot_redirect_scope(database_scope, monkeypatch, dump_boundary, backup_dsn):
    monkeypatch.setenv("TLDW_POSTGRES_BACKUP_DSN", backup_dsn)
    with pytest.raises(UnsupportedBackupBackendError, match="unsupported_backup_backend:postgresql"):
        data_ops.create_backup_snapshot(dataset="postgresql", user_id=None, backup_type="full", max_backups=None)
    assert dump_boundary == []  # nosec B101 -- never dump another database under privileged credentials.
    assert not (database_scope / "backups").exists()  # nosec B101 -- reject before artifact creation.


def test_database_restore_normalizes_scope_before_offline_gate(database_scope, monkeypatch):
    called = []
    monkeypatch.setattr(data_ops, "restore_postgres_backup", lambda *args, **kwargs: called.append(True))
    with pytest.raises(UnsupportedBackupBackendError, match="offline_restore_required"):
        data_ops.restore_backup_file(
            dataset=" POSTGRESQL ", user_id=None, backup_path=str(database_scope / "incoming.dump")
        )
    assert called == []  # nosec B101 -- spelling normalization cannot escape the offline boundary.


def test_database_snapshot_normalizes_global_scope(database_scope, dump_boundary):
    item = data_ops.create_backup_snapshot(dataset=" POSTGRESQL ", user_id=None, backup_type="full", max_backups=None)
    assert item.dataset == "postgresql" and item.user_id is None and len(dump_boundary) == 1  # nosec B101 -- metadata and PG dispatch stay canonical.


@pytest.mark.parametrize("operation", ["export", "dry_run", "restore"])
def test_legacy_bundle_rejects_database_dataset_before_any_preflight(database_scope, monkeypatch, operation):
    import json
    import zipfile

    from tldw_Server_API.app.core.exceptions import BundleExportError, BundleImportError
    from tldw_Server_API.app.services import admin_bundle_service as bundles

    def unexpected_preflight(*args, **kwargs):
        pytest.fail("legacy global scope reached storage/disk preflight")

    monkeypatch.setattr(bundles, "_preflight_backup_storage", unexpected_preflight)
    if operation == "export":
        with pytest.raises(BundleExportError, match="unsupported_bundle_dataset"):
            bundles.create_bundle(datasets=["postgresql", "authnz"], user_id=None)
        return
    archive = database_scope / "legacy.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr(
            "manifest.json",
            json.dumps(
                {
                    "manifest_version": 1,
                    "datasets": ["postgresql"],
                    "schema_versions": {},
                    "files": {"postgresql.db": {"dataset": "postgresql"}},
                }
            ),
        )
        output.writestr("postgresql.db", b"SQLite format 3\0" + bytes(100))
    bundles._rate_limit_windows.clear()
    with pytest.raises(BundleImportError, match="invalid_bundle_manifest"):
        bundles.import_bundle(file_path=str(archive), user_id=None, dry_run=operation == "dry_run")


def test_database_scope_accepts_canonical_field_configuration(database_scope, monkeypatch):
    from types import SimpleNamespace

    from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseConfig

    config = DatabaseConfig(
        backend_type=BackendType.POSTGRESQL,
        pg_host="127.0.0.1",
        pg_port=15492,
        pg_database="c09_database",
        pg_user="content_owner",
        connect_timeout=3,
    )
    monkeypatch.setattr(
        data_ops,
        "load_content_db_settings",
        lambda _: SimpleNamespace(backend_type=BackendType.POSTGRESQL, database_config=config),
    )
    resolved = data_ops._postgresql_database_config()
    assert resolved.pg_database == "c09_database"  # nosec B101 -- canonical field configuration must retain the admitted database.


@pytest.mark.parametrize("operation", ["dump", "restore"])
def test_pg_tools_preserve_explicit_connect_timeout(database_scope, monkeypatch, operation):
    from types import SimpleNamespace

    from tldw_Server_API.app.core.DB_Management import DB_Backups
    from tldw_Server_API.app.core.DB_Management.backends.base import BackendType

    config = data_ops._config_from_postgres_url(DSN + "?connect_timeout=3")
    backend = SimpleNamespace(backend_type=BackendType.POSTGRESQL, config=config)
    monkeypatch.setattr(DB_Backups.shutil, "which", lambda tool: "/fixture/bin/" + tool)
    seen = []

    def tool(command, *, env, capture_output, text):
        seen.append(env.get("PGCONNECT_TIMEOUT"))
        if operation == "dump":
            Path(command[command.index("-f") + 1]).write_bytes(b"PGDMP unit boundary")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(DB_Backups.subprocess, "run", tool)
    directory = database_scope / "backups"
    directory.mkdir()
    if operation == "dump":
        DB_Backups.create_postgres_backup(backend, backup_dir=str(directory))
    else:
        source = directory / "source.dump"
        source.write_bytes(b"PGDMP unit boundary")
        assert DB_Backups.restore_postgres_backup(backend, dump_file=str(source)) == "ok"  # nosec B101 -- actual restore command boundary completes.
    assert seen == ["3"]  # nosec B101 -- admission and CLI transport must honor the same explicit timeout.


@pytest.mark.parametrize("key", ["PGHOSTADDR", "PGSERVICE", "PGOPTIONS"])
def test_database_scope_refuses_ambient_libpq_overrides(database_scope, monkeypatch, key):
    monkeypatch.setenv(key, "ambiguous-runtime-target")
    with pytest.raises(UnsupportedBackupBackendError, match="unsupported_backup_backend:postgresql"):
        data_ops._postgresql_database_config()


def test_omitted_pg_port_cannot_disagree_with_ambient_runtime_port(database_scope, monkeypatch):
    for key in ("DATABASE_URL", "JOBS_DB_URL", "TLDW_CONTENT_PG_DSN"):
        monkeypatch.setenv(key, DSN.replace(":15492", ""))
    monkeypatch.setenv("PGPORT", "15493")
    reset_settings()
    with pytest.raises(UnsupportedBackupBackendError, match="unsupported_backup_backend:postgresql"):
        data_ops._postgresql_database_config()
