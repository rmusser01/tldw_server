"""Explicit v2 database archive admission; no database or external tool is run."""

import hashlib
import json
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tldw_Server_API.app.core.exceptions import BundleExportError, BundleImportError
from tldw_Server_API.app.services import admin_bundle_service as bundles
from tldw_Server_API.tests.Admin.test_postgresql_database_bundle import (
    database_scope as database_scope,
)
from tldw_Server_API.tests.Admin.test_postgresql_database_bundle import (
    dump_boundary as dump_boundary,
)

pytestmark = pytest.mark.unit
COVERAGE = ["all_database_sql_objects_and_rows", "all_users", "jobs"]
EXCLUSIONS = ["local_sqlite", "uploaded_files", "models", "vector_services", "cluster_roles"]


@pytest.fixture(autouse=True)
def bundle_limits(monkeypatch):
    from types import SimpleNamespace

    from tldw_Server_API.app.core.DB_Management import DB_Backups

    monkeypatch.setattr(DB_Backups.shutil, "which", lambda _: "/fixture/pg_restore")
    monkeypatch.setattr(DB_Backups, "_run_postgres_tool", lambda *args, **kwargs: SimpleNamespace(returncode=0))
    bundles._rate_limit_windows.clear()
    yield
    bundles._rate_limit_windows.clear()


def archive(root, change=None):
    content = b"PGDMP" + bytes([1, 16, 0, 4, 8, 1]) + bytes(32)
    manifest = {
        "manifest_version": 2,
        "recovery_scope": "postgresql_database",
        "user_id": None,
        "datasets": ["postgresql"],
        "coverage": COVERAGE,
        "exclusions": EXCLUSIONS,
        "files": {
            "database.dump": {
                "dataset": "postgresql",
                "sha256": hashlib.sha256(content).hexdigest(),
                "size_bytes": str(len(content)),
                "hash_algorithm": "sha256",
            }
        },
    }
    if change:
        change(manifest)
    path = root / "global.zip"
    with zipfile.ZipFile(path, "w") as output:
        output.writestr("manifest.json", json.dumps(manifest))
        output.writestr("database.dump", content)
    return path


def test_explicit_database_export_declares_v2_global_coverage(database_scope, dump_boundary):
    item = bundles.create_bundle(datasets=["postgresql"], user_id=None)
    with zipfile.ZipFile(bundles.get_bundle_path(item.bundle_id)) as output:
        manifest = json.loads(output.read("manifest.json"))
        assert manifest["manifest_version"] == 2 and manifest["recovery_scope"] == "postgresql_database"  # nosec B101 -- old importers must refuse this new scope.
        assert manifest["coverage"] == COVERAGE and manifest["exclusions"] == EXCLUSIONS  # nosec B101 -- no per-user or external-file coverage claim.
        assert manifest["user_id"] is None and manifest["datasets"] == ["postgresql"]  # nosec B101 -- database archives have no tenant target.
        assert len(manifest["files"]) == 1 and len(dump_boundary) == 1  # nosec B101 -- one complete database dump only.
    assert item.manifest_version == 2  # nosec B101 -- metadata matches the archive contract.


@pytest.mark.parametrize("datasets,user_id", [(["postgresql", "authnz"], None), (["postgresql"], 7)])
def test_database_export_rejects_mixed_or_user_scope_before_artifacts(database_scope, datasets, user_id):
    with pytest.raises(BundleExportError):
        bundles.create_bundle(datasets=datasets, user_id=user_id)
    assert not (database_scope / "backups").exists()  # nosec B101 -- scope validation precedes every artifact.


def test_database_archive_dry_run_is_read_only_and_wet_http_refuses(database_scope, monkeypatch):
    path = archive(database_scope)
    outcome = bundles.import_bundle(file_path=str(path), user_id=None, dry_run=True)
    assert outcome["status"] == "compatible" and outcome["datasets_restored"] == []  # nosec B101 -- dry run never claims a restore.
    assert outcome["validations"][0]["dataset"] == "postgresql" and outcome["validations"][0]["compatible"]  # nosec B101 -- exact global preview target.
    with pytest.raises(BundleImportError, match="offline_restore_required"):
        bundles.import_bundle(file_path=str(path), user_id=None)
    assert not (database_scope / "postgresql:").exists() and not (database_scope / "backups").exists()  # nosec B101 -- no DSN directory, staging or safety write.


@pytest.mark.parametrize(
    "change",
    [
        lambda m: m.pop("recovery_scope"),
        lambda m: m.update(recovery_scope="user"),
        lambda m: m.update(user_id=7),
        lambda m: m.update(datasets=["postgresql", "authnz"]),
        lambda m: m.update(coverage=["all_users"]),
        lambda m: m.update(exclusions=[]),
        lambda m: m["files"]["database.dump"].update(dataset="authnz"),
        lambda m: m["files"]["database.dump"].update(sha256="0" * 64),
        lambda m: m["files"]["database.dump"].update(size_bytes="1"),
    ],
)
def test_database_archive_rejects_unproven_scope_or_integrity(database_scope, change):
    path = archive(database_scope, change)
    with pytest.raises(BundleImportError):
        bundles.import_bundle(file_path=str(path), user_id=None, dry_run=True)
    assert not (database_scope / "backups").exists()  # nosec B101 -- malformed archives cannot create safety/staging outputs.


def test_database_archive_rejects_requested_user_target(database_scope):
    with pytest.raises(BundleImportError, match="user_id_forbidden"):
        bundles.import_bundle(file_path=str(archive(database_scope)), user_id=7, dry_run=True)


def test_database_archive_requires_actual_colocated_pg_target(database_scope, monkeypatch):
    monkeypatch.setenv("JOBS_DB_URL", "sqlite:///jobs.db")
    with pytest.raises(BundleImportError, match="unsupported_backup_backend"):
        bundles.import_bundle(file_path=str(archive(database_scope)), user_id=None, dry_run=True)


def test_database_archives_have_unique_scope_specific_names(database_scope, dump_boundary):
    first = bundles.create_bundle(datasets=["postgresql"], user_id=None)
    second = bundles.create_bundle(datasets=["postgresql"], user_id=None)
    assert first.bundle_id != second.bundle_id and "postgresql" in first.bundle_id  # nosec B101 -- two same-second snapshots cannot overwrite one another or legacy global bundles.
    assert all(Path(bundles.get_bundle_path(i.bundle_id)).is_file() for i in (first, second))  # nosec B101 -- both archives remain available.


@pytest.mark.parametrize("scope", [None, "postgresql_database"])
def test_bundle_retention_cannot_cross_database_recovery_scope(database_scope, scope):
    directory = database_scope / "backups" / "bundles"
    directory.mkdir(parents=True)
    old = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    for name, recovery_scope in [("legacy.zip", None), ("database.zip", "postgresql_database")]:
        manifest = {"user_id": None, "created_at": old}
        if recovery_scope:
            manifest["recovery_scope"] = recovery_scope
        with zipfile.ZipFile(directory / name, "w") as output:
            output.writestr("manifest.json", json.dumps(manifest))
    removed = bundles._prune_expired_bundles(retention_hours=1, user_id=None, recovery_scope=scope)
    assert removed == 1  # nosec B101 -- exactly one matching scope may be pruned.
    expected = "legacy.zip" if scope else "database.zip"
    assert [p.name for p in directory.iterdir()] == [expected]  # nosec B101 -- the other global scope survives.


def test_database_archive_requires_real_tool_inspection_before_compatible(database_scope, monkeypatch):
    from types import SimpleNamespace

    from tldw_Server_API.app.core.DB_Management import DB_Backups

    seen = []
    monkeypatch.setattr(DB_Backups.shutil, "which", lambda _: "/fixture/pg_restore")

    def inspect(command, **kwargs):
        seen.append(command)
        return SimpleNamespace(returncode=1, stdout="", stderr="invalid custom archive")

    monkeypatch.setattr(DB_Backups, "_run_postgres_tool", inspect)
    with pytest.raises(BundleImportError, match="invalid_postgresql_archive"):
        bundles.import_bundle(file_path=str(archive(database_scope)), user_id=None, dry_run=True)
    assert len(seen) == 1 and "--list" in seen[0] and "-d" not in seen[0]  # nosec B101 -- inspect without target database access.
