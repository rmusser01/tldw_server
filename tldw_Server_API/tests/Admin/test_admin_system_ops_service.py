from __future__ import annotations

import errno
import json
import os
import stat
import tempfile
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Any

import pytest

from tldw_Server_API.app.services import admin_system_ops_service as service
from tldw_Server_API.app.services.admin_system_ops_service import (
    _normalize_incident_record,
)

pytestmark = pytest.mark.unit

_MAX_STORE_BYTES = 67_108_864


class _TrackedBinaryStream:
    def __init__(self, stream: Any, calls: list[str]) -> None:
        self._stream = stream
        self._calls = calls

    def __enter__(self) -> _TrackedBinaryStream:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self._stream.close()

    def write(self, payload: bytes) -> int:
        return self._stream.write(payload)

    def flush(self) -> None:
        self._calls.append("flush")
        self._stream.flush()

    def fileno(self) -> int:
        return self._stream.fileno()


def _write_exact_size_json(path: Path, size: int) -> None:
    prefix = b"{}"
    assert size >= len(prefix)
    remaining = size - len(prefix)
    chunk = b" " * (1024 * 1024)
    with path.open("wb") as stream:
        stream.write(prefix)
        while remaining:
            part = chunk[: min(remaining, len(chunk))]
            stream.write(part)
            remaining -= len(part)


@pytest.mark.unit
def test_atomic_save_fsyncs_file_replace_and_parent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "nested" / "system_ops.json"
    calls: list[str] = []
    temporary_paths: list[Path] = []
    real_mkstemp = tempfile.mkstemp
    real_fdopen = os.fdopen
    real_fsync = os.fsync
    real_replace = os.replace
    fsync_count = 0

    def tracked_mkstemp(*args: Any, **kwargs: Any) -> tuple[int, str]:
        calls.append("write-temp")
        fd, name = real_mkstemp(*args, **kwargs)
        temporary_paths.append(Path(name))
        return fd, name

    def tracked_fdopen(fd: int, *args: Any, **kwargs: Any) -> _TrackedBinaryStream:
        return _TrackedBinaryStream(real_fdopen(fd, *args, **kwargs), calls)

    def tracked_fsync(fd: int) -> None:
        nonlocal fsync_count
        fsync_count += 1
        calls.append("fsync-file" if fsync_count == 1 else "fsync-parent")
        if fsync_count == 1:
            real_fsync(fd)

    def tracked_replace(source: Path, destination: Path) -> None:
        calls.append("replace")
        real_replace(source, destination)

    monkeypatch.setattr(tempfile, "mkstemp", tracked_mkstemp)
    monkeypatch.setattr(service.os, "fdopen", tracked_fdopen)
    monkeypatch.setattr(service.os, "fsync", tracked_fsync)
    monkeypatch.setattr(service.os, "replace", tracked_replace)

    store = {"incidents": [], "webhooks": []}
    service._atomic_write_store(path, store)

    assert calls == ["write-temp", "flush", "fsync-file", "replace", "fsync-parent"]
    assert json.loads(path.read_text(encoding="utf-8")) == store
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert temporary_paths[0].parent == path.parent
    assert not temporary_paths[0].exists()


@pytest.mark.parametrize("failure_point", ["before_write", "write", "replace"])
@pytest.mark.unit
def test_atomic_save_never_publishes_partial_destination(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_point: str,
) -> None:
    path = tmp_path / "system_ops.json"
    original = {"version": "old", "incidents": []}
    path.write_text(json.dumps(original), encoding="utf-8")
    real_mkstemp = tempfile.mkstemp
    real_fdopen = os.fdopen
    real_fsync = os.fsync
    real_replace = os.replace
    file_fsynced = False

    class FailingWriteStream(_TrackedBinaryStream):
        def write(self, payload: bytes) -> int:
            raise OSError("injected write failure")

    def failing_mkstemp(*args: Any, **kwargs: Any) -> tuple[int, str]:
        if failure_point == "before_write":
            raise OSError("injected temporary-file failure")
        return real_mkstemp(*args, **kwargs)

    def maybe_failing_fdopen(fd: int, *args: Any, **kwargs: Any) -> Any:
        opened = real_fdopen(fd, *args, **kwargs)
        if failure_point == "write":
            return FailingWriteStream(opened, [])
        return opened

    def tracked_fsync(fd: int) -> None:
        nonlocal file_fsynced
        file_fsynced = True
        real_fsync(fd)

    def maybe_failing_replace(source: Path, destination: Path) -> None:
        if failure_point == "replace":
            assert file_fsynced is True
            raise OSError("injected replace failure")
        real_replace(source, destination)

    monkeypatch.setattr(tempfile, "mkstemp", failing_mkstemp)
    monkeypatch.setattr(service.os, "fdopen", maybe_failing_fdopen)
    monkeypatch.setattr(service.os, "fsync", tracked_fsync)
    monkeypatch.setattr(service.os, "replace", maybe_failing_replace)

    with pytest.raises(OSError):
        service._atomic_write_store(path, {"version": "new"})

    assert json.loads(path.read_text(encoding="utf-8")) == original
    assert list(tmp_path.glob(".system_ops.json.*")) == []


@pytest.mark.unit
def test_atomic_save_only_tolerates_unsupported_directory_fsync(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "system_ops.json"
    real_fsync = os.fsync
    calls = 0

    def unsupported_directory_fsync(fd: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            real_fsync(fd)
            return
        raise OSError(errno.EINVAL, "directory fsync unsupported")

    monkeypatch.setattr(service.os, "fsync", unsupported_directory_fsync)
    service._atomic_write_store(path, {"ok": True})
    assert json.loads(path.read_text(encoding="utf-8")) == {"ok": True}

    calls = 0

    def ordinary_directory_failure(fd: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            real_fsync(fd)
            return
        raise OSError(errno.EIO, "directory fsync failed")

    monkeypatch.setattr(service.os, "fsync", ordinary_directory_failure)
    with pytest.raises(OSError):
        service._atomic_write_store(path, {"ok": False})


@pytest.mark.unit
def test_locked_store_retains_process_and_file_locks_during_atomic_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "system_ops.json"
    path.write_text(json.dumps({"incidents": [], "webhooks": []}), encoding="utf-8")
    process_lock = Lock()
    file_lock_active = False
    published: list[dict[str, Any]] = []

    @contextmanager
    def tracked_file_lock(timeout: float = 5.0):
        del timeout
        nonlocal file_lock_active
        file_lock_active = True
        try:
            yield
        finally:
            file_lock_active = False

    def record_publication(destination: Path, store: dict[str, Any]) -> None:
        assert destination == path
        assert process_lock.locked()
        assert file_lock_active is True
        published.append(store.copy())

    monkeypatch.setattr(service, "_STORE_PATH", path)
    monkeypatch.setattr(service, "_STORE_LOCK", process_lock)
    monkeypatch.setattr(service, "_store_file_lock", tracked_file_lock)
    monkeypatch.setattr(service, "_atomic_write_store", record_publication, raising=False)

    with service._locked_store(write=True) as store:
        store["marker"] = "published"

    assert published == [
        {"incidents": [], "webhooks": [], "marker": "published", **{
            key: value
            for key, value in service._default_store().items()
            if key not in {"incidents", "webhooks"}
        }}
    ]
    assert process_lock.locked() is False
    assert file_lock_active is False


@pytest.mark.unit
def test_strict_store_reader_handles_only_absent_or_whitespace_as_empty(
    tmp_path: Path,
) -> None:
    path = tmp_path / "system_ops.json"
    assert service._load_store_strict(path) == {}

    path.write_text(" \n\t", encoding="utf-8")
    assert service._load_store_strict(path) == {}

    path.write_text('{"webhooks": []}', encoding="utf-8")
    assert service._load_store_strict(path) == {"webhooks": []}


@pytest.mark.unit
def test_strict_store_reader_accepts_exact_limit_and_rejects_one_byte_more(
    tmp_path: Path,
) -> None:
    path = tmp_path / "system_ops.json"
    _write_exact_size_json(path, _MAX_STORE_BYTES)
    assert service._load_store_strict(path, max_bytes=_MAX_STORE_BYTES) == {}

    with path.open("wb") as stream:
        stream.truncate(_MAX_STORE_BYTES + 1)
    with pytest.raises(ValueError, match="system ops store exceeds size limit"):
        service._load_store_strict(path, max_bytes=_MAX_STORE_BYTES)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"\xff", "valid UTF-8"),
        (b"{not-json}", "valid JSON"),
        (b"[]", "JSON object"),
    ],
)
@pytest.mark.unit
def test_strict_store_reader_rejects_invalid_content_without_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    payload: bytes,
    message: str,
) -> None:
    path = tmp_path / "system_ops.json"
    path.write_bytes(payload)
    writes: list[object] = []
    monkeypatch.setattr(
        service,
        "_atomic_write_store",
        lambda *args, **kwargs: writes.append((args, kwargs)),
        raising=False,
    )
    class NoWarningLogger:
        def warning(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            pytest.fail("strict reader must not log content")

    monkeypatch.setattr(service, "logger", NoWarningLogger())

    with pytest.raises(ValueError, match=message):
        service._load_store_strict(path)

    assert writes == []


@pytest.mark.unit
def test_strict_store_reader_propagates_read_errors_and_rejects_symlinks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "system_ops.json"
    path.write_text("{}", encoding="utf-8")
    real_open = service.os.open

    def denied_open(*args: Any, **kwargs: Any) -> int:
        del args, kwargs
        raise PermissionError("read denied")

    monkeypatch.setattr(service.os, "open", denied_open)
    with pytest.raises(PermissionError):
        service._load_store_strict(path)
    monkeypatch.setattr(service.os, "open", real_open)

    if hasattr(os, "O_NOFOLLOW"):
        target = tmp_path / "target.json"
        target.write_text("{}", encoding="utf-8")
        path.unlink()
        path.symlink_to(target)
        with pytest.raises(OSError):
            service._load_store_strict(path)


@pytest.mark.unit
def test_normalize_incident_record_leaves_resolution_metrics_empty_for_unresolved_incident() -> None:
    normalized = _normalize_incident_record(
        {
            "id": "incident-1",
            "created_at": "2026-03-01T00:00:00+00:00",
            "resolved_at": None,
            "timeline": [],
        }
    )

    assert normalized["time_to_acknowledge_seconds"] is None
    assert normalized["time_to_resolve_seconds"] is None


@pytest.mark.unit
def test_normalize_incident_record_skips_seed_creation_event_for_acknowledgement() -> None:
    normalized = _normalize_incident_record(
        {
            "id": "incident-2",
            "created_at": "2026-03-01T00:00:00+00:00",
            "resolved_at": None,
            "timeline": [
                {
                    "id": "evt-seed",
                    "message": "Incident created",
                    "created_at": "2026-03-01T00:00:00+00:00",
                },
                {
                    "id": "evt-followup",
                    "message": "Investigating",
                    "created_at": "2026-03-01T00:05:00+00:00",
                },
            ],
        }
    )

    assert normalized["time_to_acknowledge_seconds"] == 300
