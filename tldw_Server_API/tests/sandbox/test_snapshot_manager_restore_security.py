from __future__ import annotations

import os
import tarfile
import threading
import time
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Sandbox.snapshots import SnapshotManager


def test_restore_snapshot_round_trip(tmp_path: Path) -> None:
    manager = SnapshotManager(storage_path=str(tmp_path / "snapshots"))
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "state.txt").write_text("original", encoding="utf-8")

    snapshot = manager.create_snapshot("sess-1", str(workspace))
    assert "snapshot_id" in snapshot

    (workspace / "state.txt").write_text("mutated", encoding="utf-8")
    (workspace / "new.txt").write_text("new", encoding="utf-8")

    assert manager.restore_snapshot("sess-1", snapshot["snapshot_id"], str(workspace)) is True
    assert (workspace / "state.txt").read_text(encoding="utf-8") == "original"
    assert not (workspace / "new.txt").exists()


def test_restore_snapshot_rejects_path_traversal_member(tmp_path: Path) -> None:
    manager = SnapshotManager(storage_path=str(tmp_path / "snapshots"))
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "keep.txt").write_text("ok", encoding="utf-8")

    session_id = "sess-path"
    snapshot_id = "snap-path"
    snapshot_path = manager._snapshot_path(session_id, snapshot_id)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)

    source = tmp_path / "payload.txt"
    source.write_text("payload", encoding="utf-8")
    with tarfile.open(snapshot_path, "w:gz") as tar:
        tar.add(source, arcname="../outside.txt")

    with pytest.raises(ValueError, match="Path traversal detected"):
        manager.restore_snapshot(session_id, snapshot_id, str(workspace))

    assert (workspace / "keep.txt").read_text(encoding="utf-8") == "ok"
    assert not (workspace.parent / "outside.txt").exists()


def test_restore_snapshot_rejects_symlink_member(tmp_path: Path) -> None:
    manager = SnapshotManager(storage_path=str(tmp_path / "snapshots"))
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "keep.txt").write_text("safe", encoding="utf-8")

    session_id = "sess-link"
    snapshot_id = "snap-link"
    snapshot_path = manager._snapshot_path(session_id, snapshot_id)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(snapshot_path, "w:gz") as tar:
        info = tarfile.TarInfo("danger-link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tar.addfile(info)

    with pytest.raises(ValueError, match="Refusing to extract link member"):
        manager.restore_snapshot(session_id, snapshot_id, str(workspace))

    assert (workspace / "keep.txt").read_text(encoding="utf-8") == "safe"


def test_create_snapshot_during_concurrent_atomic_writes_is_consistent(tmp_path: Path) -> None:
    manager = SnapshotManager(storage_path=str(tmp_path / "snapshots"))
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    state_file = workspace / "state.txt"
    state_file.write_text("version-0000", encoding="utf-8")

    stop = threading.Event()
    versions_lock = threading.Lock()
    versions: list[str] = []

    def _writer() -> None:
        i = 1
        while not stop.is_set():
            content = f"version-{i:05d}-" + ("x" * 1024)
            tmp = workspace / "state.tmp"
            tmp.write_text(content, encoding="utf-8")
            os.replace(tmp, state_file)
            with versions_lock:
                versions.append(content)
                if len(versions) > 500:
                    versions.pop(0)
            i += 1
            time.sleep(0.001)

    writer = threading.Thread(target=_writer, daemon=True)
    writer.start()
    time.sleep(0.05)
    snapshot = manager.create_snapshot("sess-consistency", str(workspace))
    stop.set()
    writer.join(timeout=2.0)

    with versions_lock:
        observed = set(versions)
    assert observed

    state_file.write_text("mutated", encoding="utf-8")
    assert manager.restore_snapshot("sess-consistency", snapshot["snapshot_id"], str(workspace)) is True
    restored = state_file.read_text(encoding="utf-8")
    assert restored in observed
    assert restored.startswith("version-")


def test_create_snapshot_skips_locked_atomic_temp_entry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manager = SnapshotManager(storage_path=str(tmp_path / "snapshots"))
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    state_file = workspace / "state.txt"
    state_file.write_text("stable", encoding="utf-8")
    (workspace / "state.tmp").write_text("in progress", encoding="utf-8")

    original_add = tarfile.TarFile.add

    def _raise_for_atomic_tmp(
        self: tarfile.TarFile,
        name: str,
        *args: object,
        **kwargs: object,
    ) -> None:
        if Path(name).name == "state.tmp":
            raise PermissionError("locked atomic temp file")
        original_add(self, name, *args, **kwargs)

    monkeypatch.setattr(tarfile.TarFile, "add", _raise_for_atomic_tmp)

    snapshot = manager.create_snapshot("sess-locked-temp", str(workspace))

    state_file.write_text("mutated", encoding="utf-8")
    (workspace / "state.tmp").unlink()
    assert manager.restore_snapshot("sess-locked-temp", snapshot["snapshot_id"], str(workspace)) is True
    assert state_file.read_text(encoding="utf-8") == "stable"
    assert not (workspace / "state.tmp").exists()


def test_clone_session_rejects_symlink_escape(tmp_path: Path) -> None:
    manager = SnapshotManager(storage_path=str(tmp_path / "snapshots"))
    source_workspace = tmp_path / "source"
    source_workspace.mkdir(parents=True, exist_ok=True)
    (source_workspace / "safe.txt").write_text("safe", encoding="utf-8")
    outside_secret = tmp_path / "outside_secret.txt"
    outside_secret.write_text("do-not-copy", encoding="utf-8")
    try:
        os.symlink(str(outside_secret), str(source_workspace / "escape"))
    except (NotImplementedError, OSError):
        pytest.skip("symlink creation not supported in this environment")

    new_workspace = tmp_path / "dest"
    with pytest.raises(ValueError, match="Refusing symlink workspace entry"):
        manager.clone_session("sess-src", str(source_workspace), "sess-dst", str(new_workspace))
    assert not new_workspace.exists()


def test_create_snapshot_rejects_symlink_workspace_entry(tmp_path: Path) -> None:
    manager = SnapshotManager(storage_path=str(tmp_path / "snapshots"))
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "safe.txt").write_text("ok", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    try:
        os.symlink(str(outside), str(workspace / "leak"))
    except (NotImplementedError, OSError):
        pytest.skip("symlink creation not supported in this environment")

    with pytest.raises(ValueError, match="Refusing symlink workspace entry"):
        manager.create_snapshot("sess-create-link", str(workspace))


def test_create_snapshot_rejects_symlink_workspace_root(tmp_path: Path) -> None:
    manager = SnapshotManager(storage_path=str(tmp_path / "snapshots"))
    real_workspace = tmp_path / "real-workspace"
    real_workspace.mkdir(parents=True, exist_ok=True)
    (real_workspace / "safe.txt").write_text("ok", encoding="utf-8")
    symlink_root = tmp_path / "workspace-link"
    try:
        os.symlink(str(real_workspace), str(symlink_root))
    except (NotImplementedError, OSError):
        pytest.skip("symlink creation not supported in this environment")

    with pytest.raises(ValueError, match="Refusing symlink workspace root"):
        manager.create_snapshot("sess-create-root-link", str(symlink_root))
