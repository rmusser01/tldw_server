"""Bundle restore backend/containment contracts through the real service owners."""

from __future__ import annotations

import builtins
import hashlib
import io
import json
import subprocess  # nosec B404 - Construct mocked tool outcomes only; no process launch.
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
from tldw_Server_API.app.core.exceptions import BundleImportError
from tldw_Server_API.app.services import admin_bundle_service as bundles

pytestmark = pytest.mark.unit
PG_URL = "postgresql://c09_fixture@127.0.0.1:15492/c09_disposable_target"


@pytest.fixture
def restore_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("DATABASE_URL", PG_URL)
    monkeypatch.setenv("TLDW_DB_BACKUP_PATH", str(tmp_path / "backups"))
    monkeypatch.setenv("TLDW_DB_ALLOWED_BASE_DIRS", str(tmp_path))
    reset_settings()
    bundles._rate_limit_windows.clear()
    try:
        yield tmp_path
    finally:
        bundles._rate_limit_windows.clear()
        reset_settings()


def write_bundle(root: Path, content: bytes, *, name: str = "authnz.dump", member: str | None = None) -> Path:
    archive = root / "input.zip"
    manifest = {
        "manifest_version": 1,
        "datasets": ["authnz"],
        "schema_versions": {},
        "files": {
            name: {"dataset": "authnz", "size_bytes": str(len(content)), "sha256": hashlib.sha256(content).hexdigest()}
        },
    }
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("manifest.json", json.dumps(manifest))
        if member is not None:
            bundle.writestr(member, content)
        elif not Path(name).is_absolute() and ".." not in Path(name).parts:
            bundle.writestr(name, content)
    return archive


def test_postgres_space_check_never_creates_a_dsn_shaped_local_directory(restore_env):
    archive = write_bundle(restore_env, b"PGDMP test")
    bundles._check_import_disk_space(file_path=str(archive), required_bytes=64, datasets=["authnz"], user_id=None)
    assert not (restore_env / "postgresql:").exists()  # nosec B101 - Assert the isolated restore/containment contract.


def test_postgres_bundle_roundtrip_uses_contained_transactional_pg_restore(restore_env, monkeypatch):
    from tldw_Server_API.app.core.DB_Management import DB_Backups

    commands = []
    monkeypatch.setattr(DB_Backups.shutil, "which", lambda tool: f"/fixture/bin/{tool}")

    def tool_boundary(command, *, env, capture_output, text):
        commands.append(command)
        if command[0].endswith("/pg_dump"):
            Path(command[command.index("-f") + 1]).write_bytes(b"PGDMP synthetic external dump")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(DB_Backups.subprocess, "run", tool_boundary)
    created = bundles.create_bundle(datasets=["authnz"], user_id=None, admin_user_id=7)
    result = bundles.import_bundle(file_path=bundles.get_bundle_path(created.bundle_id), user_id=None, admin_user_id=7)
    restores = [command for command in commands if command[0].endswith("/pg_restore")]
    assert result["datasets_restored"] == ["authnz"]  # nosec B101 - Assert the isolated restore/containment contract.
    assert len(restores) == 1 and "-1" in restores[0]  # nosec B101 - Assert the isolated restore/containment contract.
    assert Path(restores[0][-1]).is_relative_to(restore_env / "backups")  # nosec B101 - Assert the isolated restore/containment contract.
    assert restores[0][restores[0].index("-d") + 1] == "c09_disposable_target"  # nosec B101 - Assert the isolated restore/containment contract.
    assert not (restore_env / "postgresql:").exists()  # nosec B101 - Assert the isolated restore/containment contract.


@pytest.mark.parametrize("target_postgres", [True, False])
def test_bundle_rejects_cross_backend_format_before_safety_or_restore(restore_env, monkeypatch, target_postgres):
    if not target_postgres:
        monkeypatch.setenv("DATABASE_URL", f"sqlite:///{restore_env / 'target.db'}")
        reset_settings()
    content = b"SQLite format 3\0" + bytes(100) if target_postgres else b"PGDMP synthetic external dump"
    archive = write_bundle(restore_env, content, name="source.db" if target_postgres else "source.dump")
    snapshots = []

    def snapshot(**kwargs):
        snapshots.append(kwargs)
        return SimpleNamespace(filename="safety.db")

    monkeypatch.setattr(bundles, "create_backup_snapshot", snapshot)
    with pytest.raises(BundleImportError) as failure:
        bundles.import_bundle(file_path=str(archive), user_id=None, admin_user_id=7)
    assert failure.value.error_code == "backup_format_mismatch"  # nosec B101 - Assert the isolated restore/containment contract.
    assert snapshots == []  # nosec B101 - Assert the isolated restore/containment contract.
    assert not (restore_env / "target.db").exists()  # nosec B101 - Assert the isolated restore/containment contract.
    assert not (restore_env / "postgresql:").exists()  # nosec B101 - Assert the isolated restore/containment contract.


@pytest.mark.parametrize("absolute", [True, False])
def test_manifest_file_path_is_validated_before_any_external_file_read(restore_env, monkeypatch, absolute):
    outside = restore_env / "outside.dump"
    content = b"PGDMP owned sentinel"
    outside.write_bytes(content)
    name = str(outside) if absolute else "../outside.dump"
    archive = write_bundle(restore_env, content, name=name)
    reads = []
    builtin_open, io_open = builtins.open, io.open

    def observe_open(original):
        def open_owned(file, *args, **kwargs):
            if isinstance(file, (str, Path)) and Path(file).resolve() == outside:
                reads.append(outside)
            return original(file, *args, **kwargs)

        return open_owned

    monkeypatch.setattr(builtins, "open", observe_open(builtin_open))
    monkeypatch.setattr(io, "open", observe_open(io_open))
    with pytest.raises(BundleImportError) as failure:
        bundles.import_bundle(file_path=str(archive), user_id=None, dry_run=True, admin_user_id=7)
    assert failure.value.error_code == "path_traversal_detected"  # nosec B101 - Assert the isolated restore/containment contract.
    assert reads == []  # nosec B101 - Assert the isolated restore/containment contract.
    assert outside.read_bytes() == content  # nosec B101 - Assert the isolated restore/containment contract.


def test_failed_postgres_restore_never_reports_an_imported_dataset(restore_env, monkeypatch):
    from tldw_Server_API.app.core.DB_Management import DB_Backups

    archive = write_bundle(restore_env, b"PGDMP owned dump")
    commands = []
    monkeypatch.setattr(bundles, "create_backup_snapshot", lambda **kwargs: SimpleNamespace(filename="safety.dump"))
    monkeypatch.setattr(DB_Backups.shutil, "which", lambda tool: f"/fixture/bin/{tool}")

    def fail_restore(command, *, env, capture_output, text):
        commands.append(command)
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="owned restore rejection")

    monkeypatch.setattr(DB_Backups.subprocess, "run", fail_restore)
    with pytest.raises(BundleImportError) as failure:
        bundles.import_bundle(file_path=str(archive), user_id=None, admin_user_id=7)
    assert failure.value.error_code == "restore_failed"  # nosec B101 - Assert the isolated restore/containment contract.
    assert len(commands) == 1 and "-1" in commands[0]  # nosec B101 - Assert the isolated restore/containment contract.
    assert archive.exists()  # nosec B101 - Assert the isolated restore/containment contract.
    assert not list((restore_env / "backups").glob("tldw_bundle_import_*"))  # nosec B101 - Assert the isolated restore/containment contract.


def test_later_dataset_failure_rolls_postgres_back_through_its_snapshot_owner(restore_env, monkeypatch):
    from tldw_Server_API.app.core.DB_Management import DB_Backups

    archive = write_bundle(restore_env, b"PGDMP owned dump")
    with zipfile.ZipFile(archive) as incoming:
        manifest = json.loads(incoming.read("manifest.json"))
    media = b"SQLite format 3\0" + bytes(100)
    manifest["datasets"].append("media")
    manifest["files"]["media.db"] = {
        "dataset": "media",
        "size_bytes": str(len(media)),
        "sha256": hashlib.sha256(media).hexdigest(),
    }
    with zipfile.ZipFile(archive, "w") as incoming:
        incoming.writestr("manifest.json", json.dumps(manifest))
        incoming.writestr("authnz.dump", b"PGDMP owned dump")
        incoming.writestr("media.db", media)
    safety = restore_env / "backups/authnz/safety.dump"
    safety.parent.mkdir(parents=True)
    safety.write_bytes(b"PGDMP owned safety dump")
    monkeypatch.setenv("USER_DB_BASE_DIR", str(restore_env / "user-databases"))
    monkeypatch.setattr(bundles, "create_backup_snapshot", lambda **kwargs: SimpleNamespace(filename="safety.dump"))
    monkeypatch.setattr(DB_Backups.shutil, "which", lambda tool: f"/fixture/bin/{tool}")
    commands = []

    def tool_boundary(command, *, env, capture_output, text):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(DB_Backups.subprocess, "run", tool_boundary)
    original_restore = bundles.restore_backup_file

    def fail_media(**kwargs):
        if kwargs["dataset"] == "media":
            raise RuntimeError("owned later dataset failure")
        return original_restore(**kwargs)

    monkeypatch.setattr(bundles, "restore_backup_file", fail_media)
    with pytest.raises(BundleImportError) as failure:
        bundles.import_bundle(file_path=str(archive), user_id=7, admin_user_id=7)
    assert failure.value.error_code == "restore_failed"  # nosec B101 - Assert the isolated restore/containment contract.
    assert len(commands) == 2 and all("-1" in command for command in commands)  # nosec B101 - Assert the isolated restore/containment contract.
    assert commands[1][-1] == str(safety)  # nosec B101 - Assert the isolated restore/containment contract.
    assert getattr(failure.value, "rollback_failures", []) == []  # nosec B101 - Assert the isolated restore/containment contract.
