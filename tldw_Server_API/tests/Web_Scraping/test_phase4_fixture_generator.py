from __future__ import annotations

import errno
import hashlib
import json
import multiprocessing
import os
import queue
import stat
import subprocess
import tempfile
import traceback
import unicodedata
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from Helper_Scripts import web_scraping_phase4_fixtures as generator

PHASE4_FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "phase4"
ROUTER_PARITY_TEST = Path(__file__).parent / "test_router_yaml_predecessor_parity.py"


def test_router_category_is_registered_with_an_explicit_builder() -> None:
    assert generator.CASE_NAMES == (
        "article_orchestration_fakes",
        "content",
        "extraction",
        "metadata",
        "router",
        "selectors",
    )
    assert callable(generator.build_router_cases)


def test_checked_fixture_manifest_includes_router_category() -> None:
    manifest = json.loads((PHASE4_FIXTURE_ROOT / "manifest.json").read_text(encoding="ascii"))

    assert manifest["cases"]["router"] == "router.json"
    assert {path.name for path in PHASE4_FIXTURE_ROOT.iterdir()} == {
        "article_orchestration_fakes.json",
        "content.json",
        "extraction.json",
        "manifest.json",
        "metadata.json",
        "router.json",
        "selectors.json",
    }


def test_router_fixture_replay_is_self_contained() -> None:
    source = ROUTER_PARITY_TEST.read_text(encoding="utf-8")

    assert "import subprocess" not in source
    assert "/private/tmp" not in source
    assert "TLDW_PHASE4_PREDECESSOR_ROOT" not in source
    assert "pytest.skip" not in source
    assert "_capture_predecessor" not in source


def _symlink_or_skip(
    link: Path,
    target: Path,
    *,
    target_is_directory: bool,
) -> None:
    try:
        link.symlink_to(target, target_is_directory=target_is_directory)
    except OSError as exc:
        unsupported_errnos = {
            errno.EACCES,
            errno.EPERM,
            getattr(errno, "ENOSYS", -1),
            getattr(errno, "ENOTSUP", -1),
            getattr(errno, "EOPNOTSUPP", -1),
        }
        if exc.errno in unsupported_errnos or getattr(exc, "winerror", None) == 1314:
            pytest.skip("symlink creation is unavailable in this environment")
        raise


def _native_samefile_aliases_or_skip(
    tmp_path: Path,
    alias_kind: str,
) -> tuple[Path, Path]:
    if alias_kind == "case":
        output = tmp_path / "fixtures"
        alias = tmp_path / "FIXTURES"
    elif alias_kind == "unicode":
        output = tmp_path / "Cafe\u0301"
        alias = tmp_path / "Caf\u00e9"
    else:  # pragma: no cover - test helper misuse
        raise AssertionError(f"unknown alias kind: {alias_kind}")

    output.mkdir()
    try:
        aliases_same_file = alias.exists() and os.path.samefile(output, alias)
    except OSError:
        aliases_same_file = False
    finally:
        output.rmdir()
    if not aliases_same_file:
        pytest.skip(f"native {alias_kind} same-file aliases are unavailable")
    return output, alias


def _publication_backup_path(output: Path, suffix: str) -> Path:
    return output.parent / f".{output.name}.backup-{suffix}"


def _fixture_payloads(marker: str) -> dict[str, dict[str, Any]]:
    return {
        category: {
            "category": category,
            "cases": [{"marker": marker}],
        }
        for category in generator.CASE_NAMES
    }


def _write_canonical_json(path: Path, payload: object) -> None:
    encoded = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    path.write_text(encoded, encoding="ascii", newline="\n")


def _write_valid_fixture_set(
    output: Path,
    predecessor_commit: str,
    marker: str,
    *,
    categories: tuple[str, ...] = generator.CASE_NAMES,
) -> None:
    output.mkdir()
    case_files = {category: f"{category}.json" for category in categories}
    for category, filename in case_files.items():
        _write_canonical_json(output / filename, _fixture_payloads(marker)[category])
    _write_canonical_json(
        output / "manifest.json",
        {
            "cases": case_files,
            "predecessor_commit": predecessor_commit,
            "schema_version": 1,
        },
    )


def _isolated_lock_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    output: Path,
) -> Path:
    temp_root = tmp_path / "lock-temp"
    temp_root.mkdir()
    monkeypatch.setattr(generator.tempfile, "gettempdir", lambda: str(temp_root))
    return generator._lock_path_for_output(output)


class _CloseFailingLockFile:
    def __init__(
        self,
        lock_file: Any,
        close_detail: str,
        *,
        close_underlying_before_error: bool = True,
        descriptor_owner: Any | None = None,
    ) -> None:
        self._lock_file = lock_file
        self._close_detail = close_detail
        self._close_underlying_before_error = close_underlying_before_error
        self.descriptor = lock_file.fileno()
        self.descriptor_owner = descriptor_owner
        self.close_calls = 0
        self.underlying_close_calls = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._lock_file, name)

    def __enter__(self) -> _CloseFailingLockFile:
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()

    @property
    def closed(self) -> bool:
        return self._lock_file.closed

    def close(self) -> None:
        self.close_calls += 1
        if self._close_underlying_before_error and not self._lock_file.closed:
            self._lock_file.close()
            self.underlying_close_calls += 1
        raise OSError(self._close_detail)


def _install_tracking_owned_descriptors(
    monkeypatch: pytest.MonkeyPatch,
) -> list[Any]:
    real_owned_descriptor = generator._OwnedDescriptor
    owners: list[Any] = []

    class _TrackingOwnedDescriptor(real_owned_descriptor):
        def __init__(self, descriptor: int) -> None:
            super().__init__(descriptor)
            self.initial_descriptor = descriptor
            self.close_calls = 0
            self.detach_calls = 0
            owners.append(self)

        def detach(self) -> int:
            self.detach_calls += 1
            return super().detach()

        def close(self) -> None:
            self.close_calls += 1
            super().close()

    monkeypatch.setattr(generator, "_OwnedDescriptor", _TrackingOwnedDescriptor)
    return owners


def _active_descriptor_owner(owners: list[Any], descriptor: int) -> Any:
    for owner in reversed(owners):
        if owner.initial_descriptor != descriptor:
            continue
        try:
            owner.fileno()
        except RuntimeError:
            continue
        return owner
    raise AssertionError("lock descriptor owner was not found")


def _install_close_failing_fdopen(
    monkeypatch: pytest.MonkeyPatch,
    close_detail: str,
    *,
    close_underlying_before_error: bool = True,
) -> list[_CloseFailingLockFile]:
    real_fdopen = generator.os.fdopen
    descriptor_owners = _install_tracking_owned_descriptors(monkeypatch)
    lock_files: list[_CloseFailingLockFile] = []

    def _close_failing_fdopen(descriptor: int, *args: Any, **kwargs: Any) -> _CloseFailingLockFile:
        lock_file = _CloseFailingLockFile(
            real_fdopen(descriptor, *args, **kwargs),
            close_detail,
            close_underlying_before_error=close_underlying_before_error,
            descriptor_owner=_active_descriptor_owner(descriptor_owners, descriptor),
        )
        lock_files.append(lock_file)
        return lock_file

    monkeypatch.setattr(generator.os, "fdopen", _close_failing_fdopen)
    return lock_files


class _DirectCloseFailingLockFile:
    def __init__(
        self,
        lock_file: Any,
        close_error: BaseException,
        primary_error: BaseException,
        primary_tracebacks: list[Any],
        *,
        close_underlying_before_error: bool = True,
        descriptor_owner: Any | None = None,
    ) -> None:
        self._lock_file = lock_file
        self._close_error = close_error
        self._primary_error = primary_error
        self._primary_tracebacks = primary_tracebacks
        self._close_underlying_before_error = close_underlying_before_error
        self.descriptor = lock_file.fileno()
        self.descriptor_owner = descriptor_owner
        self.close_calls = 0
        self.underlying_close_calls = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._lock_file, name)

    @property
    def closed(self) -> bool:
        return self._lock_file.closed

    def close(self) -> None:
        self.close_calls += 1
        if self._close_underlying_before_error and not self._lock_file.closed:
            self._lock_file.close()
            self.underlying_close_calls += 1
        self._primary_tracebacks.append(self._primary_error.__traceback__)
        raise self._close_error


def _install_direct_close_failing_fdopen(
    monkeypatch: pytest.MonkeyPatch,
    close_error: BaseException,
    primary_error: BaseException,
    primary_tracebacks: list[Any],
    *,
    close_underlying_before_error: bool = True,
) -> list[_DirectCloseFailingLockFile]:
    real_fdopen = generator.os.fdopen
    descriptor_owners = _install_tracking_owned_descriptors(monkeypatch)
    lock_files: list[_DirectCloseFailingLockFile] = []

    def _direct_close_failing_fdopen(
        descriptor: int,
        *args: Any,
        **kwargs: Any,
    ) -> _DirectCloseFailingLockFile:
        lock_file = _DirectCloseFailingLockFile(
            real_fdopen(descriptor, *args, **kwargs),
            close_error,
            primary_error,
            primary_tracebacks,
            close_underlying_before_error=close_underlying_before_error,
            descriptor_owner=_active_descriptor_owner(descriptor_owners, descriptor),
        )
        lock_files.append(lock_file)
        return lock_file

    monkeypatch.setattr(generator.os, "fdopen", _direct_close_failing_fdopen)
    return lock_files


def _assert_lock_file_closed_once(lock_files: list[_CloseFailingLockFile]) -> None:
    assert len(lock_files) == 1
    lock_file = lock_files[0]
    assert lock_file.close_calls == 1
    assert lock_file.underlying_close_calls == 1
    assert lock_file.closed
    descriptor_owner = lock_file.descriptor_owner
    assert descriptor_owner is not None
    assert descriptor_owner.close_calls == 1
    assert descriptor_owner.detach_calls == 1
    with pytest.raises(RuntimeError, match="^Descriptor ownership has already been released$"):
        descriptor_owner.fileno()


def test_lock_close_assertion_is_independent_of_descriptor_number_reuse(tmp_path: Path) -> None:
    descriptor = os.open(tmp_path / "closed-lock", os.O_CREAT | os.O_RDWR, 0o600)
    lock_file = _CloseFailingLockFile(os.fdopen(descriptor, "r+b", buffering=0), "close failed")
    with pytest.raises(OSError, match="^close failed$"):
        lock_file.close()

    def _released_fileno() -> int:
        raise RuntimeError("Descriptor ownership has already been released")

    lock_file.descriptor_owner = SimpleNamespace(
        close_calls=1,
        detach_calls=1,
        fileno=_released_fileno,
    )

    replacement = os.open(tmp_path / "replacement", os.O_CREAT | os.O_RDWR, 0o600)
    if replacement != descriptor:
        os.dup2(replacement, descriptor)
        os.close(replacement)
    try:
        _assert_lock_file_closed_once([lock_file])
    finally:
        os.close(descriptor)


def _stat_with_uid(metadata: os.stat_result, uid: int) -> os.stat_result:
    values = list(metadata)
    values[4] = uid
    return os.stat_result(values)


def _stat_with_inode(metadata: os.stat_result, inode: int) -> os.stat_result:
    values = list(metadata)
    values[1] = inode
    return os.stat_result(values)


def _stat_with_identity_value(
    metadata: os.stat_result,
    field: str,
    value: int,
) -> os.stat_result:
    values = list(metadata)
    values[{"st_ino": 1, "st_dev": 2}[field]] = value
    return os.stat_result(values)


def test_path_identity_accepts_zero_device_with_nonzero_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "identity"
    path.write_bytes(b"")
    real_stat = generator.os.stat
    metadata = real_stat(path, follow_symlinks=False)

    def _zero_device(
        candidate: os.PathLike[str] | str,
        *args: Any,
        **kwargs: Any,
    ) -> os.stat_result:
        result = real_stat(candidate, *args, **kwargs)
        if Path(candidate) == path and kwargs.get("follow_symlinks") is False:
            return _stat_with_identity_value(result, "st_dev", 0)
        return result

    monkeypatch.setattr(generator.os, "stat", _zero_device)

    assert generator._path_identity(path, "identity changed") == (
        0,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
    )


@pytest.mark.parametrize(
    "metadata",
    [
        pytest.param(SimpleNamespace(st_dev=1, st_ino=0, st_mode=stat.S_IFREG), id="zero-inode"),
        pytest.param(SimpleNamespace(st_dev=1, st_mode=stat.S_IFREG), id="missing-inode"),
        pytest.param(
            SimpleNamespace(st_dev=1, st_ino="inode", st_mode=stat.S_IFREG),
            id="non-integer-inode",
        ),
        pytest.param(SimpleNamespace(st_ino=1, st_mode=stat.S_IFREG), id="missing-device"),
        pytest.param(
            SimpleNamespace(st_dev="device", st_ino=1, st_mode=stat.S_IFREG),
            id="non-integer-device",
        ),
    ],
)
def test_unavailable_metadata_identity_remains_fail_closed(metadata: Any) -> None:
    with pytest.raises(
        RuntimeError,
        match="^Fixture filesystem does not provide stable identity$",
    ):
        generator._stable_metadata_identity(metadata)


def _snapshot_path(path: Path) -> tuple[str, object]:
    if path.is_file():
        return "file", path.read_bytes()
    entries = []
    for entry in sorted(path.rglob("*")):
        relative = entry.relative_to(path).as_posix()
        if entry.is_symlink():
            entries.append((relative, "symlink", str(entry.readlink())))
        elif entry.is_dir():
            entries.append((relative, "directory", None))
        else:
            entries.append((relative, "file", entry.read_bytes()))
    return "directory", entries


def _snapshot_worktree_files(source_root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(source_root).as_posix(): path.read_bytes()
        for path in source_root.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(source_root).parts
    }


def _fixture_process_worker(
    source_root: str,
    predecessor_commit: str,
    output: str,
    marker: str,
    started: Any,
    entered_builder: Any,
    release_builder: Any,
    result_queue: Any,
    failure_mode: str = "none",
    lock_attempted: Any | None = None,
) -> None:
    def _build_payloads(_source_root: Path) -> dict[str, dict[str, Any]]:
        entered_builder.set()
        if not release_builder.wait(15):
            raise TimeoutError("test worker payload release timed out")
        if failure_mode == "exception":
            raise RuntimeError("injected payload failure")
        return _fixture_payloads(marker)

    generator.build_case_payloads = _build_payloads
    if lock_attempted is not None:
        real_acquire_file_lock = generator._acquire_file_lock

        def _signaling_acquire_file_lock(lock_file: Any) -> None:
            lock_attempted.set()
            real_acquire_file_lock(lock_file)

        generator._acquire_file_lock = _signaling_acquire_file_lock
    started.set()
    try:
        generator.generate_fixtures(
            predecessor_commit,
            Path(output),
            source_root=Path(source_root),
        )
    except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as exc:
        result_queue.put((marker, "error", type(exc).__name__, str(exc)))
    else:
        result_queue.put((marker, "ok", "", ""))


def _publication_gap_worker(
    source_root: str,
    output: str,
    gap_open: Any,
    release_gap: Any,
    result_queue: Any,
) -> None:
    output_path = Path(output)
    staging = output_path.parent / "reader-test-staging"
    _write_valid_fixture_set(staging, "2" * 40, "published")
    real_replace = Path.replace

    def _pause_in_rename_gap(path: Path, target: Path) -> Path:
        result = real_replace(path, target)
        if path == output_path:
            gap_open.set()
            if not release_gap.wait(15):
                raise TimeoutError("test publisher gap release timed out")
        return result

    Path.replace = _pause_in_rename_gap
    try:
        with generator._publication_lock(output_path, Path(source_root)):
            generator._replace_output_directory(staging, output_path)
    except BaseException as exc:  # noqa: BLE001 - report child-process failures
        result_queue.put(("publisher", "error", type(exc).__name__, str(exc)))
    else:
        result_queue.put(("publisher", "ok", "", ""))


def _cooperative_reader_worker(
    source_root: str,
    output: str,
    attempted: Any,
    entered: Any,
    result_queue: Any,
) -> None:
    attempted.set()
    try:
        with generator.fixture_publication_reader(
            Path(output),
            source_root=Path(source_root),
        ) as fixture_root:
            entered.set()
            manifest = json.loads((fixture_root / "manifest.json").read_text(encoding="ascii"))
            markers = {
                json.loads((fixture_root / filename).read_text(encoding="ascii"))["cases"][0]["marker"]
                for filename in manifest["cases"].values()
            }
    except BaseException as exc:  # noqa: BLE001 - report child-process failures
        result_queue.put(("reader", "error", type(exc).__name__, str(exc)))
    else:
        result_queue.put(("reader", "ok", "", ",".join(sorted(markers))))


def _crash_after_backup_rename_worker(
    source_root: str,
    predecessor_commit: str,
    output: str,
    rename_completed: Any,
) -> None:
    output_path = Path(output)
    real_replace = Path.replace

    def _terminate_after_backup_rename(path: Path, target: Path) -> Path:
        result = real_replace(path, target)
        if path == output_path:
            rename_completed.set()
            os._exit(73)
        return result

    generator.build_case_payloads = lambda _source_root: _fixture_payloads("unpublished")
    Path.replace = _terminate_after_backup_rename
    generator.generate_fixtures(
        predecessor_commit,
        output_path,
        source_root=Path(source_root),
    )


def _crash_after_recovery_record_worker(
    source_root: str,
    predecessor_commit: str,
    output: str,
    record_durable: Any,
) -> None:
    output_path = Path(output)
    real_write_recovery_record = generator._write_recovery_record

    def _terminate_after_recovery_record(*args: Any, **kwargs: Any) -> Any:
        record = real_write_recovery_record(*args, **kwargs)
        record_durable.set()
        os._exit(74)
        return record

    generator.build_case_payloads = lambda _source_root: _fixture_payloads("unpublished")
    generator._write_recovery_record = _terminate_after_recovery_record
    generator.generate_fixtures(
        predecessor_commit,
        output_path,
        source_root=Path(source_root),
    )


def _join_process(process: multiprocessing.Process, timeout: float = 20) -> None:
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join(5)
        pytest.fail(f"fixture worker {process.name} did not exit")


def _queue_results(result_queue: Any, count: int) -> list[tuple[str, str, str, str]]:
    results = []
    for _ in range(count):
        try:
            results.append(result_queue.get(timeout=5))
        except queue.Empty:
            pytest.fail("fixture worker did not report a result")
    return results


def _assert_fixture_marker(output: Path, marker: str) -> None:
    assert {path.name for path in output.iterdir()} == {
        "manifest.json",
        *(f"{category}.json" for category in generator.CASE_NAMES),
    }
    for category in generator.CASE_NAMES:
        payload = json.loads((output / f"{category}.json").read_text(encoding="ascii"))
        assert payload["cases"] == [{"marker": marker}]


def _run_git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _create_clean_source_root(tmp_path: Path) -> tuple[Path, str]:
    source_root = tmp_path / "source"
    source_root.mkdir()
    _run_git(source_root, "init", "--quiet")
    _run_git(source_root, "config", "user.email", "phase4-fixtures@example.invalid")
    _run_git(source_root, "config", "user.name", "Phase 4 Fixtures")
    (source_root / "tracked.txt").write_text("predecessor\n", encoding="utf-8")
    _run_git(source_root, "add", "tracked.txt")
    _run_git(source_root, "commit", "--quiet", "-m", "predecessor")
    return source_root, _run_git(source_root, "rev-parse", "HEAD")


@pytest.mark.parametrize(
    ("relationship", "through_symlink"),
    [
        pytest.param("equal", False, id="equal"),
        pytest.param("equal", True, id="equal-symlink"),
        pytest.param("ancestor", False, id="ancestor"),
        pytest.param("ancestor", True, id="ancestor-symlink"),
    ],
)
def test_output_at_source_or_ancestor_is_rejected_before_payload_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relationship: str,
    through_symlink: bool,
) -> None:
    source_root, source_commit = _create_clean_source_root(tmp_path)
    target = source_root if relationship == "equal" else tmp_path
    output = target
    if through_symlink:
        output = tmp_path / f"{relationship}-link"
        _symlink_or_skip(output, target, target_is_directory=True)
    before_bytes = _snapshot_worktree_files(source_root)
    before_status = _run_git(source_root, "status", "--porcelain=v1", "--untracked-files=all")

    def _payload_build_must_not_run(_source_root: Path) -> dict[str, object]:
        raise AssertionError("payload builder called for rejected output")

    monkeypatch.setattr(generator, "build_case_payloads", _payload_build_must_not_run)

    with pytest.raises(
        ValueError,
        match="^output must not be source-root or an ancestor of source-root$",
    ):
        generator.generate_fixtures(source_commit, output, source_root=source_root)

    assert _snapshot_worktree_files(source_root) == before_bytes
    assert _run_git(source_root, "status", "--porcelain=v1", "--untracked-files=all") == before_status


@pytest.mark.parametrize(
    "symlink_state",
    ["broken-final", "broken-component", "self-loop"],
)
def test_unresolvable_output_symlink_is_rejected_before_payload_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    symlink_state: str,
) -> None:
    source_root, source_commit = _create_clean_source_root(tmp_path)
    output_link = tmp_path / "output-link"
    if symlink_state == "self-loop":
        _symlink_or_skip(output_link, output_link, target_is_directory=True)
    else:
        _symlink_or_skip(
            output_link,
            tmp_path / "missing-target",
            target_is_directory=True,
        )
    output = output_link / "fixtures" if symlink_state == "broken-component" else output_link
    original_link_target = os.readlink(output_link)
    payload_calls = 0

    def _record_payload_build(_source_root: Path) -> dict[str, dict[str, Any]]:
        nonlocal payload_calls
        payload_calls += 1
        return _fixture_payloads("must-not-build")

    monkeypatch.setattr(generator, "build_case_payloads", _record_payload_build)

    with pytest.raises(ValueError, match="^output path could not be resolved$") as exc_info:
        generator.generate_fixtures(source_commit, output, source_root=source_root)

    assert payload_calls == 0
    assert output_link.is_symlink()
    assert os.readlink(output_link) == original_link_target
    assert str(tmp_path) not in str(exc_info.value)


@pytest.mark.parametrize(
    "parent_parts",
    [("..",), ("..", "..")],
    ids=["effective-source", "effective-source-parent"],
)
def test_parent_traversal_through_missing_output_is_rejected_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    parent_parts: tuple[str, ...],
) -> None:
    source_root, source_commit = _create_clean_source_root(tmp_path)
    missing = source_root / "absent"
    output = missing.joinpath(*parent_parts)
    before_bytes = _snapshot_worktree_files(source_root)
    before_status = _run_git(source_root, "status", "--porcelain=v1", "--untracked-files=all")
    payload_calls = 0

    def _record_payload_build(_source_root: Path) -> dict[str, dict[str, Any]]:
        nonlocal payload_calls
        payload_calls += 1
        return _fixture_payloads("must-not-build")

    monkeypatch.setattr(generator, "build_case_payloads", _record_payload_build)

    with pytest.raises(ValueError, match="^output path could not be resolved$") as exc_info:
        generator.generate_fixtures(source_commit, output, source_root=source_root)

    assert payload_calls == 0
    assert not missing.exists()
    assert _snapshot_worktree_files(source_root) == before_bytes
    assert _run_git(source_root, "status", "--porcelain=v1", "--untracked-files=all") == before_status
    assert str(source_root) not in str(exc_info.value)


def test_output_resolution_preserves_parent_traversal_after_valid_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root, _source_commit = _create_clean_source_root(tmp_path)
    source_child = source_root / "child"
    source_child.mkdir()
    (source_child / "tracked.txt").write_text("child\n", encoding="utf-8")
    _run_git(source_root, "add", "child/tracked.txt")
    _run_git(source_root, "commit", "--quiet", "-m", "add child")
    source_commit = _run_git(source_root, "rev-parse", "HEAD")
    alias_root = tmp_path / "aliases"
    alias_root.mkdir()
    child_alias = alias_root / "child-link"
    _symlink_or_skip(child_alias, source_child, target_is_directory=True)
    output = child_alias / ".." / "generated" / "fixtures"
    monkeypatch.setattr(generator, "build_case_payloads", lambda _source_root: _fixture_payloads("alias"))

    generator.generate_fixtures(source_commit, output, source_root=source_root)

    _assert_fixture_marker(source_root / "generated" / "fixtures", "alias")
    assert not (alias_root / "generated").exists()


def test_output_child_inside_source_root_remains_supported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root, source_commit = _create_clean_source_root(tmp_path)
    output = source_root / "generated" / "fixtures"
    monkeypatch.setattr(generator, "build_case_payloads", lambda _source_root: _fixture_payloads("child"))

    generator.generate_fixtures(source_commit, output, source_root=source_root)

    assert json.loads((output / "content.json").read_text(encoding="ascii"))["cases"] == [{"marker": "child"}]


def test_generation_replaces_canonical_prior_category_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root, source_commit = _create_clean_source_root(tmp_path)
    output = tmp_path / "fixtures"
    prior_categories = (
        "article_orchestration_fakes",
        "content",
        "extraction",
        "metadata",
        "selectors",
    )
    _write_valid_fixture_set(
        output,
        "1" * 40,
        "prior",
        categories=prior_categories,
    )
    monkeypatch.setattr(generator, "build_case_payloads", lambda _source_root: _fixture_payloads("current"))

    generator.generate_fixtures(source_commit, output, source_root=source_root)

    _assert_fixture_marker(output, "current")


@pytest.mark.parametrize(
    "invalid_kind",
    [
        "non-directory",
        "arbitrary",
        "extra-entry",
        "non-file-entry",
        "malformed-json",
        "noncanonical-json",
        "invalid-category",
        "invalid-case-entry",
        "invalid-manifest",
    ],
)
def test_invalid_existing_output_is_rejected_unchanged_before_payload_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_kind: str,
) -> None:
    source_root, source_commit = _create_clean_source_root(tmp_path)
    output = tmp_path / "fixtures"
    if invalid_kind == "non-directory":
        output.write_text("sensitive existing file\n", encoding="utf-8")
    elif invalid_kind == "arbitrary":
        output.mkdir()
        (output / "unrelated.txt").write_text("sensitive unrelated directory\n", encoding="utf-8")
    else:
        _write_valid_fixture_set(output, "1" * 40, "old")
        if invalid_kind == "extra-entry":
            (output / "extra.json").write_text("{}\n", encoding="ascii")
        elif invalid_kind == "non-file-entry":
            case_path = output / "content.json"
            case_path.unlink()
            case_path.mkdir()
        elif invalid_kind == "malformed-json":
            (output / "content.json").write_text("{sensitive malformed", encoding="ascii")
        elif invalid_kind == "noncanonical-json":
            payload = _fixture_payloads("old")["content"]
            (output / "content.json").write_text(json.dumps(payload), encoding="ascii")
        elif invalid_kind == "invalid-category":
            _write_canonical_json(
                output / "content.json",
                {"category": "selectors", "cases": [{"marker": "old"}]},
            )
        elif invalid_kind == "invalid-case-entry":
            _write_canonical_json(
                output / "content.json",
                {"category": "content", "cases": ["not-an-object"]},
            )
        elif invalid_kind == "invalid-manifest":
            manifest = json.loads((output / "manifest.json").read_text(encoding="ascii"))
            manifest["unexpected"] = "sensitive manifest value"
            _write_canonical_json(output / "manifest.json", manifest)
    before = _snapshot_path(output)

    def _payload_build_must_not_run(_source_root: Path) -> dict[str, object]:
        raise AssertionError("payload builder called for invalid existing output")

    monkeypatch.setattr(generator, "build_case_payloads", _payload_build_must_not_run)

    with pytest.raises((RuntimeError, ValueError)) as exc_info:
        generator.generate_fixtures(source_commit, output, source_root=source_root)

    assert _snapshot_path(output) == before
    diagnostic = str(exc_info.value)
    assert str(tmp_path) not in diagnostic
    assert "sensitive" not in diagnostic


def test_empty_case_object_remains_a_valid_existing_fixture_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root, source_commit = _create_clean_source_root(tmp_path)
    output = tmp_path / "fixtures"
    _write_valid_fixture_set(output, "1" * 40, "old")
    _write_canonical_json(
        output / "content.json",
        {"category": "content", "cases": [{}]},
    )
    monkeypatch.setattr(generator, "build_case_payloads", lambda _source_root: _fixture_payloads("new"))

    generator.generate_fixtures(source_commit, output, source_root=source_root)

    payload = json.loads((output / "content.json").read_text(encoding="ascii"))
    assert payload["cases"] == [{"marker": "new"}]


def test_valid_existing_fixture_with_different_predecessor_is_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root, source_commit = _create_clean_source_root(tmp_path)
    output = tmp_path / "fixtures"
    _write_valid_fixture_set(output, "1" * 40, "old")
    monkeypatch.setattr(generator, "build_case_payloads", lambda _source_root: _fixture_payloads("new"))

    generator.generate_fixtures(source_commit, output, source_root=source_root)

    manifest = json.loads((output / "manifest.json").read_text(encoding="ascii"))
    assert manifest["predecessor_commit"] == source_commit
    for category in generator.CASE_NAMES:
        payload = json.loads((output / f"{category}.json").read_text(encoding="ascii"))
        assert payload["cases"] == [{"marker": "new"}]


def test_lock_path_is_a_stable_hash_of_the_resolved_output_outside_source(
    tmp_path: Path,
) -> None:
    source_root, _source_commit = _create_clean_source_root(tmp_path)
    output = tmp_path / "fixtures"
    output.mkdir()
    output_link = tmp_path / "fixtures-link"
    _symlink_or_skip(output_link, output, target_is_directory=True)
    normalized_output = os.path.normcase(str(output.resolve())).casefold()
    expected_identity = hashlib.sha256(normalized_output.encode("utf-8")).hexdigest()

    direct_lock = generator._lock_path_for_output(output)
    linked_lock = generator._lock_path_for_output(output_link)

    assert direct_lock == linked_lock
    assert direct_lock.name == f"{expected_identity}.lock"
    effective_uid = os.geteuid() if hasattr(os, "geteuid") else None
    namespace = "tldw-phase4-fixture-locks"
    if effective_uid is not None:
        namespace = f"{namespace}-{effective_uid}"
    assert direct_lock.parent == Path(tempfile.gettempdir()).resolve() / namespace
    assert not direct_lock.is_relative_to(source_root)


def test_lock_path_casefolds_resolved_output_aliases(tmp_path: Path) -> None:
    lower_output = tmp_path / "fixtures"
    upper_output = tmp_path / "FIXTURES"
    normalized_output = os.path.normcase(str(lower_output.resolve())).casefold()
    expected_identity = hashlib.sha256(normalized_output.encode("utf-8")).hexdigest()

    lower_lock = generator._lock_path_for_output(lower_output)
    upper_lock = generator._lock_path_for_output(upper_output)

    assert lower_lock == upper_lock
    assert lower_lock.name == f"{expected_identity}.lock"


@pytest.mark.parametrize(
    ("first_name", "second_name"),
    [
        pytest.param("fixtures", "FIXTURES", id="case"),
        pytest.param("Cafe\u0301", "Caf\u00e9", id="unicode-normalization"),
    ],
)
def test_publication_key_normalizes_case_and_unicode_aliases(
    tmp_path: Path,
    first_name: str,
    second_name: str,
) -> None:
    first_output = tmp_path / first_name
    second_output = tmp_path / second_name
    expected_key = unicodedata.normalize(
        "NFC",
        os.path.normcase(str(first_output.resolve())).casefold(),
    )

    assert generator._publication_key(first_output) == expected_key
    assert generator._publication_key(second_output) == expected_key
    assert generator._lock_path_for_output(first_output) == generator._lock_path_for_output(second_output)
    assert generator._recovery_path_for_output(first_output) == generator._recovery_path_for_output(second_output)
    first_backup = _publication_backup_path(first_output, "a" * 32)
    second_backup = _publication_backup_path(second_output, "a" * 32)
    assert first_backup != second_backup
    assert first_backup.name == f".{first_name}.backup-{'a' * 32}"
    assert second_backup.name == f".{second_name}.backup-{'a' * 32}"


@pytest.mark.parametrize(
    "lock_root_state",
    ["symlink-into-source", "non-directory", "inside-source"],
)
def test_invalid_physical_lock_root_is_rejected_before_lock_file_or_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lock_root_state: str,
) -> None:
    source_root, source_commit = _create_clean_source_root(tmp_path)
    temp_root = tmp_path / "lock-temp"
    temp_root.mkdir()

    if lock_root_state == "inside-source":
        temp_root = source_root
    monkeypatch.setattr(generator.tempfile, "gettempdir", lambda: str(temp_root))
    lock_root = generator._lock_path_for_output(tmp_path / "fixtures").parent
    physical_lock_root = lock_root

    if lock_root_state == "symlink-into-source":
        physical_lock_root = source_root / "tracked-lock-root"
        physical_lock_root.mkdir()
        (physical_lock_root / "tracked.txt").write_text("tracked lock root\n", encoding="utf-8")
        _run_git(source_root, "add", "tracked-lock-root/tracked.txt")
        _run_git(source_root, "commit", "--quiet", "-m", "add tracked lock root")
        source_commit = _run_git(source_root, "rev-parse", "HEAD")
        _symlink_or_skip(lock_root, physical_lock_root, target_is_directory=True)
    elif lock_root_state == "non-directory":
        lock_root.write_text("sensitive lock root file\n", encoding="utf-8")
    payload_calls = 0

    def _record_payload_build(_source_root: Path) -> dict[str, dict[str, Any]]:
        nonlocal payload_calls
        payload_calls += 1
        return _fixture_payloads("must-not-build")

    monkeypatch.setattr(generator, "build_case_payloads", _record_payload_build)

    with pytest.raises(RuntimeError, match="^Fixture publication lock root is invalid$") as exc_info:
        generator.generate_fixtures(
            source_commit,
            tmp_path / "fixtures",
            source_root=source_root,
        )

    assert payload_calls == 0
    if physical_lock_root.is_dir():
        assert not list(physical_lock_root.glob("*.lock"))
    diagnostic = str(exc_info.value)
    assert str(source_root) not in diagnostic
    assert "sensitive" not in diagnostic


@pytest.mark.skipif(os.name != "posix", reason="POSIX lock metadata policy")
@pytest.mark.parametrize("unsafe_state", ["permissive-mode", "wrong-owner"])
def test_unsafe_lock_root_metadata_is_rejected_before_payload_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unsafe_state: str,
) -> None:
    source_root, source_commit = _create_clean_source_root(tmp_path)
    output = tmp_path / "fixtures"
    lock_path = _isolated_lock_path(tmp_path, monkeypatch, output)
    lock_path.parent.mkdir(mode=0o700)
    if unsafe_state == "permissive-mode":
        lock_path.parent.chmod(0o755)
    else:
        real_fstat = generator.os.fstat

        def _wrong_directory_owner(descriptor: int) -> os.stat_result:
            metadata = real_fstat(descriptor)
            if stat.S_ISDIR(metadata.st_mode):
                return _stat_with_uid(metadata, metadata.st_uid + 1)
            return metadata

        monkeypatch.setattr(generator.os, "fstat", _wrong_directory_owner)

    payload_calls = 0

    def _record_payload_build(_source_root: Path) -> dict[str, dict[str, Any]]:
        nonlocal payload_calls
        payload_calls += 1
        return _fixture_payloads("must-not-build")

    monkeypatch.setattr(generator, "build_case_payloads", _record_payload_build)

    with pytest.raises(RuntimeError, match="^Fixture publication lock root is invalid$") as exc_info:
        generator.generate_fixtures(source_commit, output, source_root=source_root)

    assert payload_calls == 0
    assert not lock_path.exists()
    assert str(tmp_path) not in str(exc_info.value)


@pytest.mark.skipif(os.name != "posix", reason="POSIX lock metadata policy")
@pytest.mark.parametrize(
    "unsafe_state",
    ["permissive-mode", "wrong-owner", "multiple-links", "symlink"],
)
def test_unsafe_lock_file_metadata_is_rejected_before_payload_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unsafe_state: str,
) -> None:
    source_root, source_commit = _create_clean_source_root(tmp_path)
    output = tmp_path / "fixtures"
    lock_path = _isolated_lock_path(tmp_path, monkeypatch, output)
    lock_path.parent.mkdir(mode=0o700)
    lock_target = lock_path.parent / "lock-target"
    lock_target.write_bytes(b"")
    lock_target.chmod(0o600)
    if unsafe_state == "symlink":
        _symlink_or_skip(lock_path, lock_target, target_is_directory=False)
    else:
        os.link(lock_target, lock_path)
        if unsafe_state == "permissive-mode":
            lock_path.chmod(0o644)
        elif unsafe_state == "wrong-owner":
            real_fstat = generator.os.fstat

            def _wrong_file_owner(descriptor: int) -> os.stat_result:
                metadata = real_fstat(descriptor)
                if stat.S_ISREG(metadata.st_mode):
                    return _stat_with_uid(metadata, metadata.st_uid + 1)
                return metadata

            monkeypatch.setattr(generator.os, "fstat", _wrong_file_owner)
        elif unsafe_state != "multiple-links":
            raise AssertionError(f"unexpected test state: {unsafe_state}")
        if unsafe_state != "multiple-links":
            lock_target.unlink()

    before = _snapshot_path(lock_path.parent)
    payload_calls = 0

    def _record_payload_build(_source_root: Path) -> dict[str, dict[str, Any]]:
        nonlocal payload_calls
        payload_calls += 1
        return _fixture_payloads("must-not-build")

    monkeypatch.setattr(generator, "build_case_payloads", _record_payload_build)

    with pytest.raises(RuntimeError, match="^Fixture publication lock file is invalid$") as exc_info:
        generator.generate_fixtures(source_commit, output, source_root=source_root)

    assert payload_calls == 0
    assert _snapshot_path(lock_path.parent) == before
    assert str(tmp_path) not in str(exc_info.value)


def test_no_dirfd_fallback_rejects_lock_root_identity_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_root = tmp_path / "locks"
    lock_root.mkdir(mode=0o700)
    monkeypatch.setattr(generator, "fcntl", None)
    real_stat = generator.os.stat
    root_stat_calls = 0

    def _changing_stat(path: os.PathLike[str] | str, *args: Any, **kwargs: Any) -> os.stat_result:
        nonlocal root_stat_calls
        metadata = real_stat(path, *args, **kwargs)
        if Path(path) == lock_root and kwargs.get("follow_symlinks") is False:
            root_stat_calls += 1
            if root_stat_calls > 1:
                return _stat_with_inode(metadata, metadata.st_ino + 1)
        return metadata

    monkeypatch.setattr(generator.os, "stat", _changing_stat)

    with pytest.raises(RuntimeError, match="^Fixture publication lock root is invalid$"):
        generator._open_lock_descriptor(lock_root, "output.lock")


def test_no_dirfd_fallback_rejects_preexisting_lock_identity_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_root = tmp_path / "locks"
    lock_root.mkdir(mode=0o700)
    lock_path = lock_root / "output.lock"
    lock_path.write_bytes(b"")
    lock_path.chmod(0o600)
    monkeypatch.setattr(generator, "fcntl", None)
    real_open = generator.os.open
    real_close = generator.os.close
    real_stat = generator.os.stat
    real_fstat = generator.os.fstat
    lock_stat_calls = 0
    opened_descriptor: int | None = None
    returned_descriptor: int | None = None

    def _tracking_open(
        path: os.PathLike[str] | str,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal opened_descriptor
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if Path(path) == lock_path:
            opened_descriptor = descriptor
        return descriptor

    def _changing_lock_stat(
        path: os.PathLike[str] | str,
        *args: Any,
        **kwargs: Any,
    ) -> os.stat_result:
        nonlocal lock_stat_calls
        metadata = real_stat(path, *args, **kwargs)
        if Path(path) == lock_path and kwargs.get("follow_symlinks") is False:
            lock_stat_calls += 1
            if lock_stat_calls > 1:
                return _stat_with_inode(metadata, metadata.st_ino + 1)
        return metadata

    def _changed_opened_file(descriptor: int) -> os.stat_result:
        metadata = real_fstat(descriptor)
        if descriptor == opened_descriptor:
            return _stat_with_inode(metadata, metadata.st_ino + 1)
        return metadata

    monkeypatch.setattr(generator.os, "open", _tracking_open)
    monkeypatch.setattr(generator.os, "stat", _changing_lock_stat)
    monkeypatch.setattr(generator.os, "fstat", _changed_opened_file)

    try:
        with pytest.raises(RuntimeError, match="^Fixture publication lock file is invalid$"):
            returned_descriptor = generator._open_lock_descriptor(lock_root, lock_path.name)
    finally:
        if returned_descriptor is not None:
            real_close(returned_descriptor)

    assert opened_descriptor is not None
    with pytest.raises(OSError) as closed:
        real_fstat(opened_descriptor)
    assert closed.value.errno == errno.EBADF


def test_no_dirfd_fallback_closes_descriptor_on_direct_baseexception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DirectInspectionFailure(BaseException):
        pass

    lock_root = tmp_path / "locks"
    lock_root.mkdir(mode=0o700)
    lock_path = lock_root / "output.lock"
    monkeypatch.setattr(generator, "fcntl", None)
    real_open = generator.os.open
    real_close = generator.os.close
    real_fstat = generator.os.fstat
    primary_message = "direct lock inspection failure"
    primary_error = DirectInspectionFailure(primary_message)
    opened_descriptors: list[int] = []
    closed_descriptors: list[int] = []
    primary_tracebacks_during_close: list[Any] = []

    def _tracking_open(
        path: os.PathLike[str] | str,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if Path(path) == lock_path:
            opened_descriptors.append(descriptor)
        return descriptor

    def _fail_opened_descriptor_inspection(descriptor: int) -> os.stat_result:
        if descriptor in opened_descriptors:
            raise primary_error
        return real_fstat(descriptor)

    def _tracking_close(descriptor: int) -> None:
        if descriptor in opened_descriptors:
            closed_descriptors.append(descriptor)
            primary_tracebacks_during_close.append(primary_error.__traceback__)
        real_close(descriptor)

    monkeypatch.setattr(generator.os, "open", _tracking_open)
    monkeypatch.setattr(generator.os, "fstat", _fail_opened_descriptor_inspection)
    monkeypatch.setattr(generator.os, "close", _tracking_close)

    try:
        with pytest.raises(DirectInspectionFailure, match=f"^{primary_message}$") as exc_info:
            generator._open_lock_descriptor(lock_root, lock_path.name)

        assert exc_info.value is primary_error
        assert type(exc_info.value) is DirectInspectionFailure
        assert str(exc_info.value) == primary_message
        assert len(opened_descriptors) == 1
        assert closed_descriptors == opened_descriptors
        assert len(primary_tracebacks_during_close) == 1
        final_traceback = exc_info.value.__traceback__
        assert final_traceback is not None
        assert final_traceback.tb_next is primary_tracebacks_during_close[0]
        recorded_tail = primary_tracebacks_during_close[0]
        while recorded_tail is not None and recorded_tail.tb_next is not None:
            recorded_tail = recorded_tail.tb_next
        final_tail = final_traceback
        while final_tail.tb_next is not None:
            final_tail = final_tail.tb_next
        assert final_tail is recorded_tail
        with pytest.raises(OSError) as closed:
            real_fstat(opened_descriptors[0])
        assert closed.value.errno == errno.EBADF
    finally:
        for descriptor in opened_descriptors:
            try:
                real_close(descriptor)
            except OSError:
                pass


def test_no_dirfd_fallback_rejects_junction_like_lock_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    lock_root = tmp_path / "locks"
    monkeypatch.setattr(generator, "fcntl", None)
    monkeypatch.setattr(
        Path,
        "is_junction",
        lambda path: path == lock_root,
        raising=False,
    )

    with pytest.raises(RuntimeError, match="^Fixture publication lock root is invalid$"):
        generator._prepare_lock_root(lock_root, source_root)


@pytest.mark.parametrize(
    ("entry_state", "file_attributes", "reparse_tag", "expected"),
    [
        pytest.param("reparse-attribute", 0x400, 0, True, id="reparse-attribute"),
        pytest.param("reparse-tag", 0, 0xA0000003, True, id="reparse-tag"),
        pytest.param("regular", 0, 0, False, id="regular"),
        pytest.param("missing", 0, 0, False, id="missing"),
    ],
)
def test_link_like_detection_uses_reparse_metadata_without_is_junction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry_state: str,
    file_attributes: int,
    reparse_tag: int,
    expected: bool,
) -> None:
    entry = tmp_path / "entry"
    if entry_state != "missing":
        entry.write_bytes(b"")
    monkeypatch.delattr(Path, "is_junction", raising=False)
    monkeypatch.setattr(
        generator.stat,
        "FILE_ATTRIBUTE_REPARSE_POINT",
        0x400,
        raising=False,
    )
    real_lstat = generator.os.lstat

    def _reparse_lstat(path: os.PathLike[str] | str) -> Any:
        if Path(path) == entry and entry_state != "missing":
            return SimpleNamespace(
                st_mode=stat.S_IFREG | 0o600,
                st_file_attributes=file_attributes,
                st_reparse_tag=reparse_tag,
            )
        return real_lstat(path)

    monkeypatch.setattr(generator.os, "lstat", _reparse_lstat)

    assert generator._is_link_like(entry) is expected


@pytest.mark.skipif(os.name != "posix", reason="POSIX lock metadata policy")
@pytest.mark.parametrize("metadata_target", ["root", "file"])
def test_lock_metadata_without_stable_identity_uses_dedicated_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metadata_target: str,
) -> None:
    lock_root = tmp_path / "locks"
    lock_root.mkdir(mode=0o700)
    real_fstat = generator.os.fstat

    def _zero_inode(descriptor: int) -> os.stat_result:
        metadata = real_fstat(descriptor)
        if metadata_target == "root" and stat.S_ISDIR(metadata.st_mode):
            return _stat_with_identity_value(metadata, "st_ino", 0)
        if metadata_target == "file" and stat.S_ISREG(metadata.st_mode):
            return _stat_with_identity_value(metadata, "st_ino", 0)
        return metadata

    monkeypatch.setattr(generator.os, "fstat", _zero_inode)

    with pytest.raises(
        RuntimeError,
        match="^Fixture filesystem does not provide stable identity$",
    ) as exc_info:
        generator._open_lock_descriptor(lock_root, "output.lock")

    assert str(tmp_path) not in str(exc_info.value)


@pytest.mark.parametrize("metadata_target", ["root", "file"])
def test_no_dirfd_lock_metadata_without_stable_identity_uses_dedicated_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metadata_target: str,
) -> None:
    lock_root = tmp_path / "locks"
    lock_root.mkdir(mode=0o700)
    monkeypatch.setattr(generator, "fcntl", None)
    real_stat = generator.os.stat
    real_fstat = generator.os.fstat

    def _zero_root_identity(
        path: os.PathLike[str] | str,
        *args: Any,
        **kwargs: Any,
    ) -> os.stat_result:
        metadata = real_stat(path, *args, **kwargs)
        if metadata_target == "root" and Path(path) == lock_root:
            return _stat_with_identity_value(metadata, "st_ino", 0)
        return metadata

    def _zero_file_identity(descriptor: int) -> os.stat_result:
        metadata = real_fstat(descriptor)
        if metadata_target == "file" and stat.S_ISREG(metadata.st_mode):
            return _stat_with_identity_value(metadata, "st_ino", 0)
        return metadata

    monkeypatch.setattr(generator.os, "stat", _zero_root_identity)
    monkeypatch.setattr(generator.os, "fstat", _zero_file_identity)

    with pytest.raises(
        RuntimeError,
        match="^Fixture filesystem does not provide stable identity$",
    ) as exc_info:
        generator._open_lock_descriptor(lock_root, "output.lock")

    assert str(tmp_path) not in str(exc_info.value)


def test_fake_windows_locking_branch_remains_usable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    _isolated_lock_path(tmp_path, monkeypatch, tmp_path / "fixtures")

    class _FakeMsvcrt:
        LK_LOCK = 1
        LK_UNLCK = 2

        def __init__(self) -> None:
            self.operations: list[int] = []

        def locking(self, _descriptor: int, operation: int, _length: int) -> None:
            self.operations.append(operation)

    fake_msvcrt = _FakeMsvcrt()
    monkeypatch.setattr(generator, "fcntl", None)
    monkeypatch.setattr(generator, "msvcrt", fake_msvcrt)

    with generator._publication_lock(tmp_path / "fixtures", source_root):
        pass

    assert fake_msvcrt.operations == [fake_msvcrt.LK_LOCK, fake_msvcrt.LK_UNLCK]


@pytest.mark.parametrize(
    "failure_stage",
    ["seek-end", "tell", "write", "flush", "loop-seek"],
)
def test_fake_windows_acquisition_setup_oserror_is_sanitized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    class _SetupFailingLockFile:
        def __init__(self) -> None:
            self.seek_calls = 0

        def seek(self, _offset: int, _whence: int = os.SEEK_SET) -> None:
            self.seek_calls += 1
            stage = "seek-end" if self.seek_calls == 1 else "loop-seek"
            if failure_stage == stage:
                raise OSError(sensitive_detail)

        def tell(self) -> int:
            if failure_stage == "tell":
                raise OSError(sensitive_detail)
            return 0

        def write(self, _payload: bytes) -> None:
            if failure_stage == "write":
                raise OSError(sensitive_detail)

        def flush(self) -> None:
            if failure_stage == "flush":
                raise OSError(sensitive_detail)

        def fileno(self) -> int:
            return 41

    class _FakeMsvcrt:
        LK_LOCK = 1

        def __init__(self) -> None:
            self.operations: list[int] = []

        def locking(self, _descriptor: int, operation: int, _length: int) -> None:
            self.operations.append(operation)

    sensitive_marker = f"sensitive Windows {failure_stage} failure"
    sensitive_path = str(tmp_path / "private-lock-file")
    sensitive_detail = f"{sensitive_marker} at {sensitive_path}"
    fake_msvcrt = _FakeMsvcrt()
    monkeypatch.setattr(generator, "fcntl", None)
    monkeypatch.setattr(generator, "msvcrt", fake_msvcrt)

    with pytest.raises(
        RuntimeError,
        match="^Fixture publication lock could not be acquired$",
    ) as exc_info:
        generator._acquire_file_lock(_SetupFailingLockFile())

    assert exc_info.value.__suppress_context__
    assert fake_msvcrt.operations == []
    formatted_diagnostic = "".join(
        traceback.format_exception(
            type(exc_info.value),
            exc_info.value,
            exc_info.value.__traceback__,
            chain=True,
        )
    )
    assert sensitive_marker not in formatted_diagnostic
    assert sensitive_path not in formatted_diagnostic


def test_fake_windows_acquisition_retries_transient_lock_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _RetryingMsvcrt:
        LK_LOCK = 1

        def __init__(self) -> None:
            self.operations: list[int] = []
            self.transient_errnos = [errno.EACCES, errno.EAGAIN, errno.EDEADLK]

        def locking(self, _descriptor: int, operation: int, _length: int) -> None:
            self.operations.append(operation)
            if self.transient_errnos:
                raise OSError(self.transient_errnos.pop(0), "sensitive transient failure")

    fake_msvcrt = _RetryingMsvcrt()
    monkeypatch.setattr(generator, "fcntl", None)
    monkeypatch.setattr(generator, "msvcrt", fake_msvcrt)

    with tempfile.TemporaryFile() as lock_file:
        generator._acquire_file_lock(lock_file)

    assert fake_msvcrt.operations == [fake_msvcrt.LK_LOCK] * 4
    assert fake_msvcrt.transient_errnos == []


@pytest.mark.skipif(
    generator.fcntl is None or os.open not in os.supports_dir_fd,
    reason="descriptor-relative POSIX lock opening is unavailable",
)
@pytest.mark.parametrize("lock_open_fails", [False, True], ids=["close-only", "primary-open-error"])
def test_lock_root_descriptor_close_failure_preserves_precedence_and_releases_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    lock_open_fails: bool,
) -> None:
    lock_root = tmp_path / "locks"
    lock_root.mkdir(mode=0o700)
    real_open = generator.os.open
    real_close = generator.os.close
    root_descriptor: int | None = None
    lock_descriptor: int | None = None
    unrelated_descriptor: int | None = None
    interleaving_injected = False
    root_close_attempts: list[int] = []
    lock_close_attempts: list[int] = []
    sensitive_marker = "sensitive root close failure"
    sensitive_path = str(tmp_path / "private-root-descriptor")
    unrelated_path = tmp_path / "unrelated-open"

    def _retire_descriptors() -> None:
        for descriptor in (lock_descriptor, root_descriptor, unrelated_descriptor):
            if descriptor is not None:
                try:
                    real_close(descriptor)
                except OSError:
                    pass

    request.addfinalizer(_retire_descriptors)

    def _controlled_open(
        path: os.PathLike[str] | str,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal root_descriptor, lock_descriptor, unrelated_descriptor, interleaving_injected
        is_root_open = dir_fd is None and os.fspath(path) == os.fspath(lock_root)
        is_lock_open = dir_fd == root_descriptor and os.fspath(path) == "output.lock"
        if is_lock_open and not interleaving_injected:
            interleaving_injected = True
            unrelated_descriptor = _controlled_open(
                unrelated_path,
                os.O_CREAT | os.O_RDWR,
                0o600,
            )
        if is_lock_open and lock_open_fails:
            raise OSError("sensitive lock open failure")
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if is_root_open:
            root_descriptor = descriptor
        elif is_lock_open:
            lock_descriptor = descriptor
        return descriptor

    def _failing_root_close(descriptor: int) -> None:
        if descriptor == root_descriptor:
            root_close_attempts.append(descriptor)
            if not lock_open_fails and len(root_close_attempts) == 1:
                raise OSError(f"{sensitive_marker} at {sensitive_path}")
        elif descriptor == lock_descriptor:
            lock_close_attempts.append(descriptor)
        real_close(descriptor)

    monkeypatch.setattr(generator.os, "open", _controlled_open)
    monkeypatch.setattr(
        generator.os,
        "supports_dir_fd",
        {*generator.os.supports_dir_fd, _controlled_open},
    )
    monkeypatch.setattr(generator.os, "close", _failing_root_close)

    expected = (
        "Fixture publication lock could not be opened"
        if lock_open_fails
        else "Fixture publication lock root could not be closed"
    )
    with pytest.raises(RuntimeError, match=f"^{expected}$") as exc_info:
        generator._open_lock_descriptor(lock_root, "output.lock")

    assert root_descriptor is not None
    assert unrelated_descriptor is not None
    os.fstat(unrelated_descriptor)
    assert str(exc_info.value) == expected
    assert root_close_attempts == [root_descriptor]
    if lock_open_fails:
        with pytest.raises(OSError) as root_closed:
            os.fstat(root_descriptor)
        assert root_closed.value.errno == errno.EBADF
    else:
        # A pre-release close failure may leak; retrying the numeric fd is unsafe.
        os.fstat(root_descriptor)
    if lock_descriptor is not None:
        assert lock_close_attempts == [lock_descriptor]
        with pytest.raises(OSError) as lock_closed:
            os.fstat(lock_descriptor)
        assert lock_closed.value.errno == errno.EBADF
    formatted_diagnostic = "".join(
        traceback.format_exception(
            type(exc_info.value),
            exc_info.value,
            exc_info.value.__traceback__,
            chain=True,
        )
    )
    assert sensitive_marker not in formatted_diagnostic
    assert sensitive_path not in formatted_diagnostic


@pytest.mark.skipif(
    generator.fcntl is None or os.open not in os.supports_dir_fd,
    reason="descriptor-relative POSIX lock opening is unavailable",
)
def test_root_close_after_release_cannot_close_reused_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_root = tmp_path / "locks"
    lock_root.mkdir(mode=0o700)
    real_open = generator.os.open
    real_close = generator.os.close
    real_fstat = generator.os.fstat
    root_descriptor: int | None = None
    lock_descriptor: int | None = None
    root_close_attempts: list[int] = []
    lock_close_attempts: list[int] = []
    replacement_descriptor: int | None = None
    replacement_descriptors: list[int] = []
    sensitive_marker = "sensitive after-close failure"
    sensitive_path = str(tmp_path / "private-root-descriptor")
    replacement_path = tmp_path / "replacement-root-descriptor"

    def _tracking_open(
        path: os.PathLike[str] | str,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal root_descriptor, lock_descriptor
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if dir_fd is None:
            root_descriptor = descriptor
        else:
            lock_descriptor = descriptor
        return descriptor

    def _close_after_release(descriptor: int) -> None:
        nonlocal replacement_descriptor
        if descriptor == root_descriptor:
            root_close_attempts.append(descriptor)
            real_close(descriptor)
            if len(root_close_attempts) == 1:
                while replacement_descriptor is None:
                    candidate = real_open(
                        replacement_path,
                        os.O_CREAT | os.O_RDWR,
                        0o600,
                    )
                    replacement_descriptors.append(candidate)
                    if candidate == descriptor:
                        replacement_descriptor = candidate
                    elif candidate > descriptor:
                        raise AssertionError("root descriptor was reused externally")
                raise OSError(f"{sensitive_marker} at {sensitive_path}")
            return
        if descriptor == lock_descriptor:
            lock_close_attempts.append(descriptor)
        real_close(descriptor)

    monkeypatch.setattr(generator.os, "open", _tracking_open)
    monkeypatch.setattr(
        generator.os,
        "supports_dir_fd",
        {*generator.os.supports_dir_fd, _tracking_open},
    )
    monkeypatch.setattr(generator.os, "close", _close_after_release)

    try:
        with pytest.raises(
            RuntimeError,
            match="^Fixture publication lock root could not be closed$",
        ) as exc_info:
            generator._open_lock_descriptor(lock_root, "output.lock")

        assert root_descriptor is not None
        assert lock_descriptor is not None
        assert root_close_attempts == [root_descriptor]
        assert lock_close_attempts == [lock_descriptor]
        assert replacement_descriptor == root_descriptor
        real_fstat(replacement_descriptor)
        with pytest.raises(OSError) as lock_closed:
            real_fstat(lock_descriptor)
        assert lock_closed.value.errno == errno.EBADF
        formatted_diagnostic = "".join(
            traceback.format_exception(
                type(exc_info.value),
                exc_info.value,
                exc_info.value.__traceback__,
                chain=True,
            )
        )
        assert sensitive_marker not in formatted_diagnostic
        assert sensitive_path not in formatted_diagnostic
    finally:
        for replacement in replacement_descriptors:
            try:
                real_close(replacement)
            except OSError:
                pass
        for descriptor in (lock_descriptor, root_descriptor):
            if descriptor is not None:
                try:
                    real_close(descriptor)
                except OSError:
                    pass


@pytest.mark.skipif(
    generator.fcntl is None or os.open not in os.supports_dir_fd,
    reason="descriptor-relative POSIX lock opening is unavailable",
)
def test_direct_root_close_baseexception_closes_lock_and_preserves_root_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DirectRootCloseFailure(BaseException):
        pass

    class DirectLockCloseFailure(BaseException):
        pass

    lock_root = tmp_path / "locks"
    lock_root.mkdir(mode=0o700)
    real_open = generator.os.open
    real_close = generator.os.close
    real_fstat = generator.os.fstat
    root_descriptor: int | None = None
    lock_descriptor: int | None = None
    root_close_attempts: list[int] = []
    lock_close_attempts: list[int] = []
    primary_message = "direct root descriptor close failure"
    primary_error = DirectRootCloseFailure(primary_message)
    lock_close_error = DirectLockCloseFailure("sensitive direct lock descriptor close failure")
    primary_tracebacks_during_lock_close: list[Any] = []

    def _tracking_open(
        path: os.PathLike[str] | str,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal root_descriptor, lock_descriptor
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if dir_fd is None:
            root_descriptor = descriptor
        else:
            lock_descriptor = descriptor
        return descriptor

    def _failing_close(descriptor: int) -> None:
        if descriptor == root_descriptor:
            root_close_attempts.append(descriptor)
            if len(root_close_attempts) == 1:
                raise primary_error
            real_close(descriptor)
            return
        if descriptor == lock_descriptor:
            lock_close_attempts.append(descriptor)
            primary_tracebacks_during_lock_close.append(primary_error.__traceback__)
            real_close(descriptor)
            raise lock_close_error
        real_close(descriptor)

    monkeypatch.setattr(generator.os, "open", _tracking_open)
    monkeypatch.setattr(
        generator.os,
        "supports_dir_fd",
        {*generator.os.supports_dir_fd, _tracking_open},
    )
    monkeypatch.setattr(generator.os, "close", _failing_close)

    try:
        with pytest.raises(DirectRootCloseFailure, match=f"^{primary_message}$") as exc_info:
            generator._open_lock_descriptor(lock_root, "output.lock")

        assert exc_info.value is primary_error
        assert type(exc_info.value) is DirectRootCloseFailure
        assert str(exc_info.value) == primary_message
        assert root_descriptor is not None
        assert lock_descriptor is not None
        assert root_close_attempts == [root_descriptor]
        assert lock_close_attempts == [lock_descriptor]
        assert len(primary_tracebacks_during_lock_close) == 1
        final_traceback = exc_info.value.__traceback__
        assert final_traceback is not None
        assert final_traceback.tb_next is primary_tracebacks_during_lock_close[0]
        recorded_tail = primary_tracebacks_during_lock_close[0]
        while recorded_tail is not None and recorded_tail.tb_next is not None:
            recorded_tail = recorded_tail.tb_next
        final_tail = final_traceback
        while final_tail.tb_next is not None:
            final_tail = final_tail.tb_next
        assert final_tail is recorded_tail
        real_fstat(root_descriptor)
        with pytest.raises(OSError) as lock_closed:
            real_fstat(lock_descriptor)
        assert lock_closed.value.errno == errno.EBADF
    finally:
        for descriptor in (lock_descriptor, root_descriptor):
            if descriptor is not None:
                try:
                    real_close(descriptor)
                except OSError:
                    pass


@pytest.mark.parametrize("body_raises", [True, False], ids=["primary-error", "unlock-only"])
def test_publication_lock_unlock_failure_preserves_exception_precedence_and_closes_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    body_raises: bool,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    temp_root = tmp_path / "lock-temp"
    temp_root.mkdir()
    monkeypatch.setattr(generator.tempfile, "gettempdir", lambda: str(temp_root))
    released_files: list[Any] = []

    def _failing_unlock(lock_file: Any) -> None:
        released_files.append(lock_file)
        raise OSError("sensitive unlock failure")

    monkeypatch.setattr(generator, "_release_file_lock", _failing_unlock)

    if body_raises:
        with pytest.raises(RuntimeError, match="^primary publication failure$") as exc_info:
            with generator._publication_lock(tmp_path / "fixtures", source_root):
                raise RuntimeError("primary publication failure")
    else:
        with pytest.raises(
            RuntimeError,
            match="^Fixture publication lock could not be released$",
        ) as exc_info:
            with generator._publication_lock(tmp_path / "fixtures", source_root):
                pass

    assert len(released_files) == 1
    assert released_files[0].closed
    assert "sensitive" not in str(exc_info.value)


def test_publication_lock_acquisition_and_close_baseexceptions_preserve_acquisition_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DirectAcquisitionFailure(BaseException):
        pass

    class DirectCloseFailure(BaseException):
        pass

    source_root = tmp_path / "source"
    source_root.mkdir()
    output = tmp_path / "fixtures"
    _isolated_lock_path(tmp_path, monkeypatch, output)
    primary_message = "direct lock acquisition failure"
    primary_error = DirectAcquisitionFailure(primary_message)
    close_error = DirectCloseFailure("sensitive direct close failure")
    primary_tracebacks_during_close: list[Any] = []
    lock_files = _install_direct_close_failing_fdopen(
        monkeypatch,
        close_error,
        primary_error,
        primary_tracebacks_during_close,
    )

    def _fail_acquisition(_lock_file: Any) -> None:
        raise primary_error

    monkeypatch.setattr(generator, "_acquire_file_lock", _fail_acquisition)

    with pytest.raises(DirectAcquisitionFailure, match=f"^{primary_message}$") as exc_info:
        with generator._publication_lock(output, source_root):
            raise AssertionError("publication body must not be entered")

    assert exc_info.value is primary_error
    assert type(exc_info.value) is DirectAcquisitionFailure
    assert str(exc_info.value) == primary_message
    assert len(lock_files) == 1
    assert lock_files[0].close_calls == 1
    assert lock_files[0].underlying_close_calls == 1
    assert lock_files[0].closed
    assert len(primary_tracebacks_during_close) == 1
    final_traceback = exc_info.value.__traceback__
    assert final_traceback is not None
    assert final_traceback.tb_next is not None
    assert final_traceback.tb_next.tb_next is primary_tracebacks_during_close[0]
    with pytest.raises(OSError) as closed_descriptor:
        os.fstat(lock_files[0].descriptor)
    assert closed_descriptor.value.errno == errno.EBADF


def test_publication_lock_unlock_and_close_baseexceptions_preserve_unlock_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DirectUnlockFailure(BaseException):
        pass

    class DirectCloseFailure(BaseException):
        pass

    source_root = tmp_path / "source"
    source_root.mkdir()
    output = tmp_path / "fixtures"
    _isolated_lock_path(tmp_path, monkeypatch, output)
    primary_message = "direct lock unlock failure"
    primary_error = DirectUnlockFailure(primary_message)
    close_error = DirectCloseFailure("sensitive direct close failure")
    primary_tracebacks_during_close: list[Any] = []
    lock_files = _install_direct_close_failing_fdopen(
        monkeypatch,
        close_error,
        primary_error,
        primary_tracebacks_during_close,
    )
    release_calls = 0

    def _fail_unlock(_lock_file: Any) -> None:
        nonlocal release_calls
        release_calls += 1
        raise primary_error

    monkeypatch.setattr(generator, "_release_file_lock", _fail_unlock)

    with pytest.raises(DirectUnlockFailure, match=f"^{primary_message}$") as exc_info:
        with generator._publication_lock(output, source_root):
            pass

    assert exc_info.value is primary_error
    assert type(exc_info.value) is DirectUnlockFailure
    assert str(exc_info.value) == primary_message
    assert release_calls == 1
    assert len(lock_files) == 1
    assert lock_files[0].close_calls == 1
    assert lock_files[0].underlying_close_calls == 1
    assert lock_files[0].closed
    assert len(primary_tracebacks_during_close) == 1
    final_traceback = exc_info.value.__traceback__
    assert final_traceback is not None
    assert final_traceback.tb_next is not None
    assert final_traceback.tb_next.tb_next is primary_tracebacks_during_close[0]
    with pytest.raises(OSError) as closed_descriptor:
        os.fstat(lock_files[0].descriptor)
    assert closed_descriptor.value.errno == errno.EBADF


def test_publication_lock_direct_unlock_baseexception_preserves_exact_body_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PrimaryPublicationFailure(BaseException):
        pass

    class DirectUnlockFailure(BaseException):
        pass

    source_root = tmp_path / "source"
    source_root.mkdir()
    output = tmp_path / "fixtures"
    _isolated_lock_path(tmp_path, monkeypatch, output)
    primary_message = "primary publication failure"
    primary_error = PrimaryPublicationFailure(primary_message)
    unlock_error = DirectUnlockFailure("sensitive direct unlock failure")
    primary_tracebacks_during_unlock: list[Any] = []
    released_files: list[Any] = []

    def _fail_unlock(lock_file: Any) -> None:
        released_files.append(lock_file)
        primary_tracebacks_during_unlock.append(primary_error.__traceback__)
        raise unlock_error

    monkeypatch.setattr(generator, "_release_file_lock", _fail_unlock)

    with pytest.raises(PrimaryPublicationFailure, match=f"^{primary_message}$") as exc_info:
        with generator._publication_lock(output, source_root):
            raise primary_error

    assert exc_info.value is primary_error
    assert type(exc_info.value) is PrimaryPublicationFailure
    assert str(exc_info.value) == primary_message
    assert len(released_files) == 1
    assert released_files[0].closed
    assert len(primary_tracebacks_during_unlock) == 1
    final_traceback = exc_info.value.__traceback__
    assert final_traceback is not None
    recorded_tail = primary_tracebacks_during_unlock[0]
    assert recorded_tail is not None
    assert recorded_tail.tb_next is final_traceback
    while recorded_tail is not None and recorded_tail.tb_next is not None:
        recorded_tail = recorded_tail.tb_next
    assert final_traceback is recorded_tail


def test_publication_lock_direct_close_baseexception_preserves_exact_body_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PrimaryPublicationFailure(BaseException):
        pass

    class DirectCloseFailure(BaseException):
        pass

    class _DirectCloseFailingLockFile:
        def __init__(self, lock_file: Any) -> None:
            self._lock_file = lock_file
            self.descriptor = lock_file.fileno()
            self.close_calls = 0

        def __getattr__(self, name: str) -> Any:
            return getattr(self._lock_file, name)

        @property
        def closed(self) -> bool:
            return self._lock_file.closed

        def close(self) -> None:
            self.close_calls += 1
            if not self._lock_file.closed:
                self._lock_file.close()
            primary_tracebacks_during_close.append(primary_error.__traceback__)
            raise close_error

    source_root = tmp_path / "source"
    source_root.mkdir()
    output = tmp_path / "fixtures"
    _isolated_lock_path(tmp_path, monkeypatch, output)
    primary_message = "primary publication failure"
    primary_error = PrimaryPublicationFailure(primary_message)
    close_error = DirectCloseFailure("sensitive direct close failure")
    primary_tracebacks_during_close: list[Any] = []
    lock_files: list[_DirectCloseFailingLockFile] = []
    real_fdopen = generator.os.fdopen

    def _direct_close_failing_fdopen(
        descriptor: int,
        *args: Any,
        **kwargs: Any,
    ) -> _DirectCloseFailingLockFile:
        lock_file = _DirectCloseFailingLockFile(real_fdopen(descriptor, *args, **kwargs))
        lock_files.append(lock_file)
        return lock_file

    monkeypatch.setattr(generator.os, "fdopen", _direct_close_failing_fdopen)

    with pytest.raises(PrimaryPublicationFailure, match=f"^{primary_message}$") as exc_info:
        with generator._publication_lock(output, source_root):
            raise primary_error

    assert exc_info.value is primary_error
    assert type(exc_info.value) is PrimaryPublicationFailure
    assert str(exc_info.value) == primary_message
    assert len(lock_files) == 1
    assert lock_files[0].close_calls == 1
    assert lock_files[0].closed
    assert len(primary_tracebacks_during_close) == 1
    final_traceback = exc_info.value.__traceback__
    assert final_traceback is not None
    recorded_tail = primary_tracebacks_during_close[0]
    assert recorded_tail is not None
    assert recorded_tail.tb_next is final_traceback
    while recorded_tail is not None and recorded_tail.tb_next is not None:
        recorded_tail = recorded_tail.tb_next
    assert final_traceback is recorded_tail
    with pytest.raises(OSError) as closed_descriptor:
        os.fstat(lock_files[0].descriptor)
    assert closed_descriptor.value.errno == errno.EBADF


def test_publication_lock_close_failure_preserves_body_baseexception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PrimaryPublicationError(BaseException):
        pass

    source_root = tmp_path / "source"
    source_root.mkdir()
    output = tmp_path / "fixtures"
    _isolated_lock_path(tmp_path, monkeypatch, output)
    primary_message = "primary publication failure"
    primary_error = PrimaryPublicationError(primary_message)
    sensitive_marker = "sensitive lock close failure"
    sensitive_path = str(tmp_path / "private-lock-file")
    close_detail = f"{sensitive_marker} at {sensitive_path}"
    lock_files = _install_close_failing_fdopen(monkeypatch, close_detail)

    with pytest.raises(
        PrimaryPublicationError,
        match="^primary publication failure$",
    ) as exc_info:
        with generator._publication_lock(output, source_root):
            raise primary_error

    assert exc_info.value is primary_error
    assert type(exc_info.value) is PrimaryPublicationError
    assert str(exc_info.value) == primary_message
    formatted_diagnostic = "".join(
        traceback.format_exception(
            type(exc_info.value),
            exc_info.value,
            exc_info.value.__traceback__,
            chain=True,
        )
    )
    assert sensitive_marker not in formatted_diagnostic
    assert sensitive_path not in formatted_diagnostic
    _assert_lock_file_closed_once(lock_files)


def test_publication_lock_close_failure_preserves_sanitized_unlock_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    output = tmp_path / "fixtures"
    _isolated_lock_path(tmp_path, monkeypatch, output)
    sensitive_marker = "sensitive lock close failure"
    sensitive_path = str(tmp_path / "private-lock-file")
    close_detail = f"{sensitive_marker} at {sensitive_path}"
    lock_files = _install_close_failing_fdopen(monkeypatch, close_detail)
    release_calls = 0

    def _fail_unlock(_lock_file: Any) -> None:
        nonlocal release_calls
        release_calls += 1
        raise OSError("sensitive unlock failure")

    monkeypatch.setattr(generator, "_release_file_lock", _fail_unlock)

    with pytest.raises(
        RuntimeError,
        match="^Fixture publication lock could not be released$",
    ) as exc_info:
        with generator._publication_lock(output, source_root):
            pass

    assert release_calls == 1
    assert str(exc_info.value) == "Fixture publication lock could not be released"
    formatted_diagnostic = "".join(
        traceback.format_exception(
            type(exc_info.value),
            exc_info.value,
            exc_info.value.__traceback__,
            chain=True,
        )
    )
    assert sensitive_marker not in formatted_diagnostic
    assert sensitive_path not in formatted_diagnostic
    _assert_lock_file_closed_once(lock_files)


def test_publication_lock_standalone_close_failure_is_sanitized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    output = tmp_path / "fixtures"
    _isolated_lock_path(tmp_path, monkeypatch, output)
    sensitive_marker = "sensitive lock close failure"
    sensitive_path = str(tmp_path / "private-lock-file")
    close_detail = f"{sensitive_marker} at {sensitive_path}"
    lock_files = _install_close_failing_fdopen(monkeypatch, close_detail)

    with pytest.raises(
        RuntimeError,
        match="^Fixture publication lock file could not be closed$",
    ) as exc_info:
        with generator._publication_lock(output, source_root):
            pass

    assert exc_info.value.__suppress_context__
    assert str(exc_info.value) == "Fixture publication lock file could not be closed"
    formatted_diagnostic = "".join(
        traceback.format_exception(
            type(exc_info.value),
            exc_info.value,
            exc_info.value.__traceback__,
            chain=True,
        )
    )
    assert sensitive_marker not in formatted_diagnostic
    assert sensitive_path not in formatted_diagnostic
    _assert_lock_file_closed_once(lock_files)


def test_publication_lock_standalone_direct_close_baseexception_is_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DirectCloseFailure(BaseException):
        pass

    source_root = tmp_path / "source"
    source_root.mkdir()
    output = tmp_path / "fixtures"
    _isolated_lock_path(tmp_path, monkeypatch, output)
    close_message = "direct standalone close failure"
    close_error = DirectCloseFailure(close_message)
    close_tracebacks: list[Any] = []
    lock_files = _install_direct_close_failing_fdopen(
        monkeypatch,
        close_error,
        close_error,
        close_tracebacks,
    )
    real_close_descriptor_quietly = generator._close_descriptor_quietly
    fallback_descriptors: list[int | None] = []

    def _tracking_descriptor_fallback(descriptor: int | None) -> None:
        fallback_descriptors.append(descriptor)
        real_close_descriptor_quietly(descriptor)

    monkeypatch.setattr(
        generator,
        "_close_descriptor_quietly",
        _tracking_descriptor_fallback,
    )

    with pytest.raises(DirectCloseFailure, match=f"^{close_message}$") as exc_info:
        with generator._publication_lock(output, source_root):
            pass

    assert exc_info.value is close_error
    assert type(exc_info.value) is DirectCloseFailure
    assert str(exc_info.value) == close_message
    assert len(lock_files) == 1
    assert lock_files[0].close_calls == 1
    assert lock_files[0].underlying_close_calls == 1
    assert lock_files[0].closed
    assert fallback_descriptors == []
    with pytest.raises(OSError) as closed_descriptor:
        os.fstat(lock_files[0].descriptor)
    assert closed_descriptor.value.errno == errno.EBADF


def test_publication_lock_failed_view_cannot_close_reused_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DirectCloseFailure(BaseException):
        pass

    source_root = tmp_path / "source"
    source_root.mkdir()
    output = tmp_path / "fixtures"
    _isolated_lock_path(tmp_path, monkeypatch, output)
    close_message = "direct standalone close failure"
    close_error = DirectCloseFailure(close_message)
    primary_tracebacks: list[Any] = []
    lock_files = _install_direct_close_failing_fdopen(
        monkeypatch,
        close_error,
        close_error,
        primary_tracebacks,
        close_underlying_before_error=False,
    )
    real_close = generator.os.close
    real_fstat = generator.os.fstat
    raw_close_attempts: list[int] = []
    replacement_descriptor: int | None = None
    replacement_descriptors: list[int] = []
    replacement_path = tmp_path / "replacement-descriptor"

    def _close_and_reuse(descriptor: int) -> None:
        nonlocal replacement_descriptor
        real_close(descriptor)
        if lock_files and descriptor == lock_files[0].descriptor:
            raw_close_attempts.append(descriptor)
            if replacement_descriptor is None:
                while replacement_descriptor is None:
                    candidate = os.open(
                        replacement_path,
                        os.O_CREAT | os.O_RDWR,
                        0o600,
                    )
                    replacement_descriptors.append(candidate)
                    if candidate == descriptor:
                        replacement_descriptor = candidate
                    elif candidate > descriptor:
                        raise AssertionError("lock descriptor was reused externally")

    monkeypatch.setattr(generator.os, "close", _close_and_reuse)

    try:
        with pytest.raises(DirectCloseFailure, match=f"^{close_message}$") as exc_info:
            with generator._publication_lock(output, source_root):
                pass

        assert exc_info.value is close_error
        assert type(exc_info.value) is DirectCloseFailure
        assert str(exc_info.value) == close_message
        assert len(lock_files) == 1
        lock_file = lock_files[0]
        assert lock_file.close_calls == 1
        assert lock_file.underlying_close_calls == 0
        assert raw_close_attempts == [lock_file.descriptor]
        assert replacement_descriptor == lock_file.descriptor

        lock_file._lock_file.close()
        assert lock_file._lock_file.closed
        assert replacement_descriptor is not None
        real_fstat(replacement_descriptor)

        final_traceback = exc_info.value.__traceback__
        assert final_traceback is not None
        final_tail = final_traceback
        while final_tail.tb_next is not None:
            final_tail = final_tail.tb_next
        assert final_tail.tb_frame.f_code.co_name == "close"
    finally:
        for lock_file in lock_files:
            try:
                lock_file._lock_file.close()
            except OSError:
                pass
        for replacement in replacement_descriptors:
            try:
                real_close(replacement)
            except OSError:
                pass


def test_publication_lock_oserror_close_before_underlying_is_sanitized_and_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    output = tmp_path / "fixtures"
    _isolated_lock_path(tmp_path, monkeypatch, output)
    sensitive_marker = "sensitive before-close failure"
    sensitive_path = str(tmp_path / "private-lock-file")
    lock_files = _install_close_failing_fdopen(
        monkeypatch,
        f"{sensitive_marker} at {sensitive_path}",
        close_underlying_before_error=False,
    )

    def _retire_file_objects() -> None:
        for lock_file in lock_files:
            try:
                lock_file._lock_file.close()
            except OSError:
                pass

    request.addfinalizer(_retire_file_objects)

    with pytest.raises(
        RuntimeError,
        match="^Fixture publication lock file could not be closed$",
    ) as exc_info:
        with generator._publication_lock(output, source_root):
            pass

    assert exc_info.value.__suppress_context__
    assert len(lock_files) == 1
    assert lock_files[0].close_calls == 1
    assert lock_files[0].underlying_close_calls == 0
    formatted_diagnostic = "".join(
        traceback.format_exception(
            type(exc_info.value),
            exc_info.value,
            exc_info.value.__traceback__,
            chain=True,
        )
    )
    assert sensitive_marker not in formatted_diagnostic
    assert sensitive_path not in formatted_diagnostic
    with pytest.raises(OSError) as closed_descriptor:
        os.fstat(lock_files[0].descriptor)
    assert closed_descriptor.value.errno == errno.EBADF


def test_publication_lock_fdopen_failure_closes_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    output = tmp_path / "fixtures"
    _isolated_lock_path(tmp_path, monkeypatch, output)
    open_lock_descriptor = generator._open_lock_descriptor
    descriptors: list[int] = []
    fdopen_error = OSError("sensitive fdopen failure")

    def _record_lock_descriptor(lock_root: Path, lock_name: str) -> int:
        descriptor = open_lock_descriptor(lock_root, lock_name)
        descriptors.append(descriptor)
        return descriptor

    def _fail_fdopen(_descriptor: int, *_args: Any, **_kwargs: Any) -> Any:
        raise fdopen_error

    monkeypatch.setattr(generator, "_open_lock_descriptor", _record_lock_descriptor)
    monkeypatch.setattr(generator.os, "fdopen", _fail_fdopen)

    try:
        with pytest.raises(OSError) as exc_info:
            with generator._publication_lock(output, source_root):
                pass

        assert exc_info.value is fdopen_error
        assert len(descriptors) == 1
        with pytest.raises(OSError) as closed_descriptor:
            os.fstat(descriptors[0])
        assert closed_descriptor.value.errno == errno.EBADF
    finally:
        for descriptor in descriptors:
            try:
                os.close(descriptor)
            except OSError:
                pass


def test_same_output_processes_serialize_without_crossed_fixture_sets(tmp_path: Path) -> None:
    source_root, source_commit = _create_clean_source_root(tmp_path)
    output = tmp_path / "fixtures"
    context = multiprocessing.get_context("spawn")
    first_started = context.Event()
    first_entered = context.Event()
    first_release = context.Event()
    second_started = context.Event()
    second_lock_attempted = context.Event()
    second_entered = context.Event()
    second_release = context.Event()
    second_release.set()
    result_queue = context.Queue()
    first = context.Process(
        target=_fixture_process_worker,
        args=(
            str(source_root),
            source_commit,
            str(output),
            "first",
            first_started,
            first_entered,
            first_release,
            result_queue,
        ),
    )
    second = context.Process(
        target=_fixture_process_worker,
        args=(
            str(source_root),
            source_commit,
            str(output),
            "second",
            second_started,
            second_entered,
            second_release,
            result_queue,
        ),
        kwargs={"lock_attempted": second_lock_attempted},
    )

    first.start()
    try:
        assert first_started.wait(10)
        assert first_entered.wait(10)
        second.start()
        assert second_started.wait(10)
        assert second_lock_attempted.wait(10)
        assert not second_entered.wait(1)
        first_release.set()
        assert second_entered.wait(10)
    finally:
        first_release.set()
        second_release.set()
        _join_process(first)
        if second.pid is not None:
            _join_process(second)

    assert sorted(_queue_results(result_queue, 2)) == [
        ("first", "ok", "", ""),
        ("second", "ok", "", ""),
    ]
    _assert_fixture_marker(output, "second")


@pytest.mark.parametrize("alias_kind", ["case", "unicode"])
def test_native_samefile_output_aliases_serialize_publishers(
    tmp_path: Path,
    alias_kind: str,
) -> None:
    source_root, source_commit = _create_clean_source_root(tmp_path)
    output, alias = _native_samefile_aliases_or_skip(tmp_path, alias_kind)
    _write_valid_fixture_set(output, "0" * 40, "original")
    assert os.path.samefile(output, alias)

    context = multiprocessing.get_context("spawn")
    first_started = context.Event()
    first_entered = context.Event()
    first_release = context.Event()
    second_started = context.Event()
    second_lock_attempted = context.Event()
    second_entered = context.Event()
    second_release = context.Event()
    second_release.set()
    result_queue = context.Queue()
    first = context.Process(
        target=_fixture_process_worker,
        args=(
            str(source_root),
            source_commit,
            str(output),
            "first-alias",
            first_started,
            first_entered,
            first_release,
            result_queue,
        ),
    )
    second = context.Process(
        target=_fixture_process_worker,
        args=(
            str(source_root),
            source_commit,
            str(alias),
            "second-alias",
            second_started,
            second_entered,
            second_release,
            result_queue,
        ),
        kwargs={"lock_attempted": second_lock_attempted},
    )

    first.start()
    try:
        assert first_started.wait(10)
        assert first_entered.wait(10)
        second.start()
        assert second_started.wait(10)
        assert second_lock_attempted.wait(10)
        assert not second_entered.wait(1)
        first_release.set()
        assert second_entered.wait(10)
    finally:
        first_release.set()
        second_release.set()
        _join_process(first)
        if second.pid is not None:
            _join_process(second)

    assert sorted(_queue_results(result_queue, 2)) == [
        ("first-alias", "ok", "", ""),
        ("second-alias", "ok", "", ""),
    ]
    _assert_fixture_marker(alias, "second-alias")


def test_different_output_processes_do_not_share_a_global_lock(tmp_path: Path) -> None:
    source_root, source_commit = _create_clean_source_root(tmp_path)
    context = multiprocessing.get_context("spawn")
    first_started = context.Event()
    first_entered = context.Event()
    first_release = context.Event()
    second_started = context.Event()
    second_entered = context.Event()
    second_release = context.Event()
    second_release.set()
    result_queue = context.Queue()
    first = context.Process(
        target=_fixture_process_worker,
        args=(
            str(source_root),
            source_commit,
            str(tmp_path / "fixtures-a"),
            "first",
            first_started,
            first_entered,
            first_release,
            result_queue,
        ),
    )
    second = context.Process(
        target=_fixture_process_worker,
        args=(
            str(source_root),
            source_commit,
            str(tmp_path / "fixtures-b"),
            "second",
            second_started,
            second_entered,
            second_release,
            result_queue,
        ),
    )

    first.start()
    try:
        assert first_started.wait(10)
        assert first_entered.wait(10)
        second.start()
        assert second_started.wait(10)
        assert second_entered.wait(10)
        _join_process(second)
        _assert_fixture_marker(tmp_path / "fixtures-b", "second")
    finally:
        first_release.set()
        _join_process(first)
        if second.is_alive():
            second.terminate()
            second.join(5)

    assert sorted(_queue_results(result_queue, 2)) == [
        ("first", "ok", "", ""),
        ("second", "ok", "", ""),
    ]


def test_same_output_lock_is_released_after_payload_exception(tmp_path: Path) -> None:
    source_root, source_commit = _create_clean_source_root(tmp_path)
    output = tmp_path / "fixtures"
    context = multiprocessing.get_context("spawn")
    failed_started = context.Event()
    failed_entered = context.Event()
    failed_release = context.Event()
    failed_release.set()
    result_queue = context.Queue()
    failed = context.Process(
        target=_fixture_process_worker,
        args=(
            str(source_root),
            source_commit,
            str(output),
            "failed",
            failed_started,
            failed_entered,
            failed_release,
            result_queue,
            "exception",
        ),
    )
    failed.start()
    assert failed_started.wait(10)
    assert failed_entered.wait(10)
    _join_process(failed)

    successor_started = context.Event()
    successor_entered = context.Event()
    successor_release = context.Event()
    successor_release.set()
    successor = context.Process(
        target=_fixture_process_worker,
        args=(
            str(source_root),
            source_commit,
            str(output),
            "successor",
            successor_started,
            successor_entered,
            successor_release,
            result_queue,
        ),
    )
    successor.start()
    assert successor_started.wait(10)
    assert successor_entered.wait(10)
    _join_process(successor)

    assert sorted(_queue_results(result_queue, 2)) == [
        ("failed", "error", "RuntimeError", "injected payload failure"),
        ("successor", "ok", "", ""),
    ]
    _assert_fixture_marker(output, "successor")


def test_same_output_lock_is_os_released_after_process_termination(tmp_path: Path) -> None:
    source_root, source_commit = _create_clean_source_root(tmp_path)
    output = tmp_path / "fixtures"
    context = multiprocessing.get_context("spawn")
    crashed_started = context.Event()
    crashed_entered = context.Event()
    crashed_release = context.Event()
    crashed_results = context.Queue()
    crashed = context.Process(
        target=_fixture_process_worker,
        args=(
            str(source_root),
            source_commit,
            str(output),
            "crashed",
            crashed_started,
            crashed_entered,
            crashed_release,
            crashed_results,
        ),
    )
    crashed.start()
    assert crashed_started.wait(10)
    assert crashed_entered.wait(10)
    crashed.terminate()
    _join_process(crashed)
    assert crashed.exitcode not in (None, 0)

    successor_started = context.Event()
    successor_entered = context.Event()
    successor_release = context.Event()
    successor_release.set()
    successor_results = context.Queue()
    successor = context.Process(
        target=_fixture_process_worker,
        args=(
            str(source_root),
            source_commit,
            str(output),
            "successor",
            successor_started,
            successor_entered,
            successor_release,
            successor_results,
        ),
    )
    successor.start()
    assert successor_started.wait(10)
    assert successor_entered.wait(10)
    _join_process(successor)

    assert _queue_results(successor_results, 1) == [("successor", "ok", "", "")]
    _assert_fixture_marker(output, "successor")


def test_cooperative_reader_cannot_observe_two_rename_publication_gap(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    output = tmp_path / "fixtures"
    _write_valid_fixture_set(output, "1" * 40, "original")
    context = multiprocessing.get_context("spawn")
    gap_open = context.Event()
    release_gap = context.Event()
    reader_attempted = context.Event()
    reader_entered = context.Event()
    result_queue = context.Queue()
    publisher = context.Process(
        target=_publication_gap_worker,
        args=(str(source_root), str(output), gap_open, release_gap, result_queue),
    )
    reader = context.Process(
        target=_cooperative_reader_worker,
        args=(
            str(source_root),
            str(output),
            reader_attempted,
            reader_entered,
            result_queue,
        ),
    )

    publisher.start()
    try:
        assert gap_open.wait(10)
        assert not output.exists()
        reader.start()
        assert reader_attempted.wait(10)
        assert not reader_entered.wait(1)
        release_gap.set()
        assert reader_entered.wait(10)
    finally:
        release_gap.set()
        _join_process(publisher)
        if reader.pid is not None:
            _join_process(reader)

    assert sorted(_queue_results(result_queue, 2)) == [
        ("publisher", "ok", "", ""),
        ("reader", "ok", "", "published"),
    ]


def test_recovery_journal_is_validated_and_closed_before_atomic_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "fixtures"
    staging = tmp_path / "staging"
    _write_valid_fixture_set(output, "1" * 40, "old")
    _write_valid_fixture_set(staging, "2" * 40, "new")
    old_snapshot = generator._validate_existing_output(output)
    new_snapshot = generator._validate_fixture_set(staging, predecessor_commit=None)
    assert old_snapshot is not None
    parent_identity = generator._path_identity(tmp_path, "parent changed")
    backup = tmp_path / f".fixtures.backup-{'a' * 32}"
    canonical = tmp_path / ".fixtures.publication-recovery.json"
    real_open = generator.os.open
    real_close = generator.os.close
    real_fsync = generator.os.fsync
    real_read_recovery_file = generator._read_recovery_file
    real_replace = Path.replace
    real_fsync_directory = generator._fsync_directory
    writer_descriptor: int | None = None
    writer_fsynced = False
    writer_closed = False
    temp_validated = False
    atomic_replace_seen = False
    parent_fsynced_after_replace = False

    def _track_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        nonlocal writer_descriptor
        descriptor = real_open(path, flags, *args, **kwargs)
        candidate = Path(path)
        if candidate.parent == tmp_path and candidate.name.startswith(f"{canonical.name}.tmp-"):
            if flags & os.O_WRONLY:
                writer_descriptor = descriptor
        return descriptor

    def _track_fsync(descriptor: int) -> None:
        nonlocal writer_fsynced
        if descriptor == writer_descriptor:
            writer_fsynced = True
        real_fsync(descriptor)

    def _track_close(descriptor: int) -> None:
        nonlocal writer_closed
        real_close(descriptor)
        if descriptor == writer_descriptor:
            writer_closed = True

    def _track_read_recovery_file(path: Path) -> tuple[tuple[int, int, int], bytes]:
        nonlocal temp_validated
        if path != canonical:
            temp_validated = True
        return real_read_recovery_file(path)

    def _track_replace(path: Path, target: Path) -> Path:
        nonlocal atomic_replace_seen
        if target == canonical:
            assert path.parent == canonical.parent
            assert path.name.startswith(f"{canonical.name}.tmp-")
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
            assert writer_fsynced
            assert writer_closed
            assert temp_validated
            atomic_replace_seen = True
        return real_replace(path, target)

    def _track_fsync_directory(
        directory: Path,
        expected_identity: tuple[int, int, int],
        error_message: str,
    ) -> None:
        nonlocal parent_fsynced_after_replace
        if directory == tmp_path and canonical.exists():
            parent_fsynced_after_replace = True
        real_fsync_directory(directory, expected_identity, error_message)

    monkeypatch.setattr(generator.os, "open", _track_open)
    monkeypatch.setattr(generator.os, "fsync", _track_fsync)
    monkeypatch.setattr(generator.os, "close", _track_close)
    monkeypatch.setattr(generator, "_read_recovery_file", _track_read_recovery_file)
    monkeypatch.setattr(Path, "replace", _track_replace)
    monkeypatch.setattr(generator, "_fsync_directory", _track_fsync_directory)

    record = generator._write_recovery_record(
        output,
        backup,
        parent_identity,
        old_snapshot,
        new_snapshot,
    )

    assert record.path == canonical
    assert atomic_replace_seen
    assert parent_fsynced_after_replace
    assert not list(tmp_path.glob(f"{canonical.name}.tmp-*"))
    payload = json.loads(canonical.read_text(encoding="ascii"))
    assert set(payload) == {
        "backup_name",
        "output_name",
        "output_snapshot",
        "parent_identity",
        "publication_identity",
        "schema_version",
        "staged_snapshot",
    }
    assert type(payload["schema_version"]) is int
    assert payload["schema_version"] == 2
    assert payload["output_name"] == output.name
    assert payload["publication_identity"] == generator._publication_key(output)


def test_torn_recovery_journal_write_leaves_no_canonical_or_temp_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "fixtures"
    staging = tmp_path / "staging"
    _write_valid_fixture_set(output, "1" * 40, "old")
    _write_valid_fixture_set(staging, "2" * 40, "new")
    old_snapshot = generator._validate_existing_output(output)
    new_snapshot = generator._validate_fixture_set(staging, predecessor_commit=None)
    assert old_snapshot is not None
    parent_identity = generator._path_identity(tmp_path, "parent changed")
    backup = tmp_path / f".fixtures.backup-{'b' * 32}"
    canonical = tmp_path / ".fixtures.publication-recovery.json"
    real_open = generator.os.open
    real_write = generator.os.write
    recovery_descriptors: set[int] = set()

    def _track_open(path: Any, *args: Any, **kwargs: Any) -> int:
        descriptor = real_open(path, *args, **kwargs)
        candidate = Path(path)
        if candidate.parent == tmp_path and ".publication-recovery.json" in candidate.name:
            recovery_descriptors.add(descriptor)
        return descriptor

    def _write_prefix_then_fail(descriptor: int, data: bytes) -> int:
        if descriptor in recovery_descriptors:
            real_write(descriptor, data[:17])
            raise OSError("sensitive torn write")
        return real_write(descriptor, data)

    monkeypatch.setattr(generator.os, "open", _track_open)
    monkeypatch.setattr(generator.os, "write", _write_prefix_then_fail)

    with pytest.raises(RuntimeError, match=generator._RECOVERY_ERROR) as exc_info:
        generator._write_recovery_record(
            output,
            backup,
            parent_identity,
            old_snapshot,
            new_snapshot,
        )

    assert not canonical.exists()
    assert not list(tmp_path.glob(f"{canonical.name}.tmp-*"))
    assert "sensitive" not in str(exc_info.value)


@pytest.mark.parametrize(
    ("state", "expected_marker"),
    [
        pytest.param("old-output-no-backup", "old", id="old-output-no-backup"),
        pytest.param("no-output-old-backup", "old", id="no-output-old-backup"),
        pytest.param("new-output-old-backup", "new", id="new-output-old-backup"),
        pytest.param("new-output-no-backup", "new", id="new-output-no-backup"),
    ],
)
def test_recovery_accepts_exact_content_bound_publication_states(
    tmp_path: Path,
    state: str,
    expected_marker: str,
) -> None:
    output = tmp_path / "fixtures"
    staging = tmp_path / "staging"
    _write_valid_fixture_set(output, "1" * 40, "old")
    _write_valid_fixture_set(staging, "2" * 40, "new")
    old_snapshot = generator._validate_existing_output(output)
    new_snapshot = generator._validate_fixture_set(staging, predecessor_commit=None)
    assert old_snapshot is not None
    parent_identity = generator._path_identity(tmp_path, "parent changed")
    backup = tmp_path / f".fixtures.backup-{'c' * 32}"
    record = generator._write_recovery_record(
        output,
        backup,
        parent_identity,
        old_snapshot,
        new_snapshot,
    )

    if state != "old-output-no-backup":
        output.replace(backup)
    if state.startswith("new-output"):
        staging.replace(output)
    if state == "new-output-no-backup":
        generator.shutil.rmtree(backup)

    generator._recover_interrupted_publication(output)

    _assert_fixture_marker(output, expected_marker)
    assert not backup.exists()
    assert not record.path.exists()


@pytest.mark.parametrize("alias_kind", ["case", "unicode"])
@pytest.mark.parametrize(
    ("state", "expected_marker"),
    [
        pytest.param("old-output-no-backup", "old", id="old-output-no-backup"),
        pytest.param("no-output-old-backup", "old", id="no-output-old-backup"),
        pytest.param("new-output-old-backup", "new", id="new-output-old-backup"),
        pytest.param("new-output-no-backup", "new", id="new-output-no-backup"),
    ],
)
def test_recovery_accepts_all_exact_states_through_native_samefile_alias(
    tmp_path: Path,
    alias_kind: str,
    state: str,
    expected_marker: str,
) -> None:
    output, alias = _native_samefile_aliases_or_skip(tmp_path, alias_kind)
    staging = tmp_path / "staging"
    _write_valid_fixture_set(output, "1" * 40, "old")
    _write_valid_fixture_set(staging, "2" * 40, "new")
    old_snapshot = generator._validate_existing_output(output)
    new_snapshot = generator._validate_fixture_set(staging, predecessor_commit=None)
    assert old_snapshot is not None
    parent_identity = generator._path_identity(tmp_path, "parent changed")
    backup = _publication_backup_path(output, "c" * 32)
    record = generator._write_recovery_record(
        output,
        backup,
        parent_identity,
        old_snapshot,
        new_snapshot,
    )
    assert record.path == generator._recovery_path_for_output(alias)

    if state != "old-output-no-backup":
        output.replace(backup)
    if state.startswith("new-output"):
        staging.replace(output)
    if state == "new-output-no-backup":
        generator.shutil.rmtree(backup)

    loaded = generator._load_recovery_record(alias)
    assert loaded is not None
    assert loaded.backup == backup
    generator._recover_interrupted_publication(alias)

    _assert_fixture_marker(alias, expected_marker)
    assert not backup.exists()
    assert not record.path.exists()


def test_recovery_rejects_normalized_key_collision_for_distinct_absent_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "primary-fixtures"
    wrong_target = tmp_path / "colliding-fixtures"
    staging = tmp_path / "staging"
    _write_valid_fixture_set(output, "1" * 40, "old")
    _write_valid_fixture_set(staging, "2" * 40, "new")
    old_snapshot = generator._validate_existing_output(output)
    new_snapshot = generator._validate_fixture_set(staging, predecessor_commit=None)
    assert old_snapshot is not None
    parent_identity = generator._path_identity(tmp_path, "parent changed")
    backup = _publication_backup_path(output, "f" * 32)
    record = generator._write_recovery_record(
        output,
        backup,
        parent_identity,
        old_snapshot,
        new_snapshot,
    )
    output.replace(backup)
    publication_key = generator._publication_key(output)
    real_publication_key = generator._publication_key

    def _colliding_publication_key(candidate: Path) -> str:
        if candidate == wrong_target:
            return publication_key
        return real_publication_key(candidate)

    monkeypatch.setattr(generator, "_publication_key", _colliding_publication_key)
    before_backup = _snapshot_path(backup)
    before_record = record.path.read_bytes()

    with pytest.raises(RuntimeError, match=f"^{generator._RECOVERY_ERROR}$"):
        generator._recover_interrupted_publication(wrong_target)

    assert not output.exists()
    assert not wrong_target.exists()
    assert _snapshot_path(backup) == before_backup
    assert record.path.read_bytes() == before_record


@pytest.mark.parametrize(
    "failure_type",
    [pytest.param(OSError, id="oserror"), pytest.param(UnicodeError, id="unicode-error")],
)
def test_recovery_alias_proof_failure_is_sanitized_and_retains_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_type: type[Exception],
) -> None:
    output, alias = _native_samefile_aliases_or_skip(tmp_path, "case")
    staging = tmp_path / "staging"
    _write_valid_fixture_set(output, "1" * 40, "old")
    _write_valid_fixture_set(staging, "2" * 40, "new")
    old_snapshot = generator._validate_existing_output(output)
    new_snapshot = generator._validate_fixture_set(staging, predecessor_commit=None)
    assert old_snapshot is not None
    parent_identity = generator._path_identity(tmp_path, "parent changed")
    backup = _publication_backup_path(output, "a" * 32)
    record = generator._write_recovery_record(
        output,
        backup,
        parent_identity,
        old_snapshot,
        new_snapshot,
    )
    before_output = _snapshot_path(output)
    before_record = record.path.read_bytes()
    sensitive_detail = f"sensitive alias path at {tmp_path}"

    def _fail_samefile(_first: object, _second: object) -> bool:
        raise failure_type(sensitive_detail)

    monkeypatch.setattr(generator.os.path, "samefile", _fail_samefile)

    with pytest.raises(RuntimeError, match=f"^{generator._RECOVERY_ERROR}$") as exc_info:
        generator._recover_interrupted_publication(alias)

    assert sensitive_detail not in str(exc_info.value)
    assert exc_info.value.__suppress_context__
    assert _snapshot_path(output) == before_output
    assert record.path.read_bytes() == before_record
    assert not backup.exists()


def test_recovery_record_rejects_mismatched_publication_identity_through_alias(
    tmp_path: Path,
) -> None:
    output, alias = _native_samefile_aliases_or_skip(tmp_path, "case")
    staging = tmp_path / "staging"
    _write_valid_fixture_set(output, "1" * 40, "old")
    _write_valid_fixture_set(staging, "2" * 40, "new")
    old_snapshot = generator._validate_existing_output(output)
    new_snapshot = generator._validate_fixture_set(staging, predecessor_commit=None)
    assert old_snapshot is not None
    parent_identity = generator._path_identity(tmp_path, "parent changed")
    backup = _publication_backup_path(output, "d" * 32)
    record = generator._write_recovery_record(
        output,
        backup,
        parent_identity,
        old_snapshot,
        new_snapshot,
    )
    payload = json.loads(record.path.read_text(encoding="ascii"))
    payload["publication_identity"] = generator._publication_key(tmp_path / "other-output")
    _write_canonical_json(record.path, payload)
    before = record.path.read_bytes()

    with pytest.raises(RuntimeError, match=generator._RECOVERY_ERROR):
        generator._load_recovery_record(alias)

    assert record.path.read_bytes() == before
    _assert_fixture_marker(alias, "old")
    assert not backup.exists()


def test_recovery_journal_identity_resolution_failure_is_sanitized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sensitive_detail = f"sensitive publication path at {tmp_path}"

    def _fail_publication_key(_output: Path) -> str:
        raise OSError(sensitive_detail)

    monkeypatch.setattr(generator, "_publication_key", _fail_publication_key)

    with pytest.raises(RuntimeError, match=f"^{generator._RECOVERY_ERROR}$") as exc_info:
        generator._load_recovery_record(tmp_path / "fixtures")

    assert sensitive_detail not in str(exc_info.value)
    assert exc_info.value.__suppress_context__


def test_recovery_rejects_ambiguous_old_output_and_unvalidated_backup(
    tmp_path: Path,
) -> None:
    output = tmp_path / "fixtures"
    staging = tmp_path / "staging"
    _write_valid_fixture_set(output, "1" * 40, "old")
    _write_valid_fixture_set(staging, "2" * 40, "new")
    old_snapshot = generator._validate_existing_output(output)
    new_snapshot = generator._validate_fixture_set(staging, predecessor_commit=None)
    assert old_snapshot is not None
    parent_identity = generator._path_identity(tmp_path, "parent changed")
    backup = tmp_path / f".fixtures.backup-{'d' * 32}"
    record = generator._write_recovery_record(
        output,
        backup,
        parent_identity,
        old_snapshot,
        new_snapshot,
    )
    _write_valid_fixture_set(backup, "3" * 40, "ambiguous")
    before_output = _snapshot_path(output)
    before_backup = _snapshot_path(backup)
    before_record = record.path.read_bytes()

    with pytest.raises(RuntimeError, match=generator._RECOVERY_ERROR):
        generator._recover_interrupted_publication(output)

    assert _snapshot_path(output) == before_output
    assert _snapshot_path(backup) == before_backup
    assert record.path.read_bytes() == before_record


def test_prior_five_category_upgrade_journal_recovers_between_renames(
    tmp_path: Path,
) -> None:
    output = tmp_path / "fixtures"
    staging = tmp_path / "staging"
    _write_valid_fixture_set(
        output,
        "1" * 40,
        "old-five",
        categories=generator._PRIOR_CASE_NAMES,
    )
    _write_valid_fixture_set(staging, "2" * 40, "new-six")
    old_snapshot = generator._validate_existing_output(output)
    new_snapshot = generator._validate_fixture_set(staging, predecessor_commit=None)
    assert old_snapshot is not None
    parent_identity = generator._path_identity(tmp_path, "parent changed")
    backup = tmp_path / f".fixtures.backup-{'e' * 32}"
    record = generator._write_recovery_record(
        output,
        backup,
        parent_identity,
        old_snapshot,
        new_snapshot,
    )
    output.replace(backup)

    loaded = generator._load_recovery_record(output)
    assert loaded is not None
    assert loaded.output_snapshot == old_snapshot
    assert loaded.staged_snapshot == new_snapshot
    generator._recover_interrupted_publication(output)

    assert {path.name for path in output.iterdir()} == {
        "manifest.json",
        *(f"{category}.json" for category in generator._PRIOR_CASE_NAMES),
    }
    assert json.loads((output / "content.json").read_text(encoding="ascii"))["cases"] == [{"marker": "old-five"}]
    assert not backup.exists()
    assert not record.path.exists()


@pytest.mark.parametrize("alias_kind", ["case", "unicode"])
def test_prior_five_category_upgrade_journal_recovers_through_samefile_alias(
    tmp_path: Path,
    alias_kind: str,
) -> None:
    output, alias = _native_samefile_aliases_or_skip(tmp_path, alias_kind)
    staging = tmp_path / "staging"
    _write_valid_fixture_set(
        output,
        "1" * 40,
        "old-five",
        categories=generator._PRIOR_CASE_NAMES,
    )
    _write_valid_fixture_set(staging, "2" * 40, "new-six")
    old_snapshot = generator._validate_existing_output(output)
    new_snapshot = generator._validate_fixture_set(staging, predecessor_commit=None)
    assert old_snapshot is not None
    parent_identity = generator._path_identity(tmp_path, "parent changed")
    backup = _publication_backup_path(output, "e" * 32)
    record = generator._write_recovery_record(
        output,
        backup,
        parent_identity,
        old_snapshot,
        new_snapshot,
    )
    output.replace(backup)

    loaded = generator._load_recovery_record(alias)
    assert loaded is not None
    assert loaded.output_snapshot == old_snapshot
    assert loaded.staged_snapshot == new_snapshot
    generator._recover_interrupted_publication(alias)

    assert {path.name for path in alias.iterdir()} == {
        "manifest.json",
        *(f"{category}.json" for category in generator._PRIOR_CASE_NAMES),
    }
    assert json.loads((alias / "content.json").read_text(encoding="ascii"))["cases"] == [{"marker": "old-five"}]
    assert not backup.exists()
    assert not record.path.exists()


def test_successful_publication_removes_validated_backup_before_journal_clear(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "fixtures"
    staging = tmp_path / "staging"
    _write_valid_fixture_set(output, "1" * 40, "old")
    _write_valid_fixture_set(staging, "2" * 40, "new")
    real_clear_recovery_record = generator._clear_recovery_record
    clear_seen = False

    def _clear_after_backup_cleanup(record: Any) -> None:
        nonlocal clear_seen
        _assert_fixture_marker(output, "new")
        assert not record.backup.exists()
        clear_seen = True
        real_clear_recovery_record(record)

    monkeypatch.setattr(generator, "_clear_recovery_record", _clear_after_backup_cleanup)

    generator._replace_output_directory(staging, output)

    assert clear_seen
    _assert_fixture_marker(output, "new")
    assert not list(tmp_path.glob(".fixtures.backup-*"))
    assert not list(tmp_path.glob(".fixtures.publication-recovery.json"))


def test_rollback_journal_cleanup_failure_does_not_mask_active_baseexception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PrimaryPublicationFailure(BaseException):
        pass

    output = tmp_path / "fixtures"
    staging = tmp_path / "staging"
    _write_valid_fixture_set(output, "1" * 40, "old")
    _write_valid_fixture_set(staging, "2" * 40, "new")
    real_replace = Path.replace
    primary = PrimaryPublicationFailure("primary publication failure")

    def _publish_then_fail(path: Path, target: Path) -> Path:
        result = real_replace(path, target)
        if path == staging and target == output:
            raise primary
        return result

    def _fail_journal_clear(_record: Any) -> None:
        raise RuntimeError("secondary journal clear failure")

    monkeypatch.setattr(Path, "replace", _publish_then_fail)
    monkeypatch.setattr(generator, "_clear_recovery_record", _fail_journal_clear)

    with pytest.raises(PrimaryPublicationFailure) as exc_info:
        generator._replace_output_directory(staging, output)

    assert exc_info.value is primary
    _assert_fixture_marker(output, "old")
    _assert_fixture_marker(staging, "new")
    assert not list(tmp_path.glob(".fixtures.backup-*"))
    assert len(list(tmp_path.glob(".fixtures.publication-recovery.json"))) == 1


def test_crash_between_renames_is_recovered_before_payload_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root, source_commit = _create_clean_source_root(tmp_path)
    output = tmp_path / "fixtures"
    _write_valid_fixture_set(output, "1" * 40, "original")
    context = multiprocessing.get_context("spawn")
    rename_completed = context.Event()
    crashed = context.Process(
        target=_crash_after_backup_rename_worker,
        args=(str(source_root), source_commit, str(output), rename_completed),
    )

    crashed.start()
    assert rename_completed.wait(10)
    _join_process(crashed)
    assert crashed.exitcode == 73
    assert not output.exists()
    assert len(list(tmp_path.glob(".fixtures.backup-*"))) == 1
    assert len(list(tmp_path.glob(".fixtures.publication-recovery.json"))) == 1

    def _stop_after_recovery(_source_root: Path) -> dict[str, dict[str, Any]]:
        raise RuntimeError("stop after recovery")

    monkeypatch.setattr(generator, "build_case_payloads", _stop_after_recovery)
    with pytest.raises(RuntimeError, match="^stop after recovery$"):
        generator.generate_fixtures(
            source_commit,
            output,
            source_root=source_root,
        )

    _assert_fixture_marker(output, "original")
    assert not list(tmp_path.glob(".fixtures.backup-*"))
    assert not list(tmp_path.glob(".fixtures.publication-recovery.json"))


def test_recovery_rejects_in_place_backup_mutation_without_deleting_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root, source_commit = _create_clean_source_root(tmp_path)
    output = tmp_path / "fixtures"
    _write_valid_fixture_set(output, "1" * 40, "original")
    context = multiprocessing.get_context("spawn")
    rename_completed = context.Event()
    crashed = context.Process(
        target=_crash_after_backup_rename_worker,
        args=(str(source_root), source_commit, str(output), rename_completed),
    )

    crashed.start()
    assert rename_completed.wait(10)
    _join_process(crashed)
    backup = next(tmp_path.glob(".fixtures.backup-*"))
    recovery_record = tmp_path / ".fixtures.publication-recovery.json"
    _write_canonical_json(backup / "content.json", _fixture_payloads("tampered")["content"])
    before_backup = _snapshot_path(backup)
    before_record = recovery_record.read_bytes()
    build_calls = 0

    def _record_payload_build(_source_root: Path) -> dict[str, dict[str, Any]]:
        nonlocal build_calls
        build_calls += 1
        return _fixture_payloads("must-not-build")

    monkeypatch.setattr(generator, "build_case_payloads", _record_payload_build)
    with pytest.raises(
        RuntimeError,
        match=("^Fixture publication recovery could not be completed safely; " "manual recovery is required$"),
    ) as exc_info:
        generator.generate_fixtures(
            source_commit,
            output,
            source_root=source_root,
        )

    assert build_calls == 0
    assert not output.exists()
    assert _snapshot_path(backup) == before_backup
    assert recovery_record.read_bytes() == before_record
    assert str(tmp_path) not in str(exc_info.value)
    assert "tampered" not in str(exc_info.value)


def test_crash_after_durable_record_before_first_rename_recovers_as_noop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root, source_commit = _create_clean_source_root(tmp_path)
    output = tmp_path / "fixtures"
    _write_valid_fixture_set(output, "1" * 40, "original")
    context = multiprocessing.get_context("spawn")
    record_durable = context.Event()
    crashed = context.Process(
        target=_crash_after_recovery_record_worker,
        args=(str(source_root), source_commit, str(output), record_durable),
    )

    crashed.start()
    assert record_durable.wait(10)
    _join_process(crashed)
    assert crashed.exitcode == 74
    _assert_fixture_marker(output, "original")
    assert not list(tmp_path.glob(".fixtures.backup-*"))
    assert len(list(tmp_path.glob(".fixtures.publication-recovery.json"))) == 1

    def _stop_after_recovery(_source_root: Path) -> dict[str, dict[str, Any]]:
        raise RuntimeError("stop after no-op recovery")

    monkeypatch.setattr(generator, "build_case_payloads", _stop_after_recovery)
    with pytest.raises(RuntimeError, match="^stop after no-op recovery$"):
        generator.generate_fixtures(
            source_commit,
            output,
            source_root=source_root,
        )

    _assert_fixture_marker(output, "original")
    assert not list(tmp_path.glob(".fixtures.backup-*"))
    assert not list(tmp_path.glob(".fixtures.publication-recovery.json"))


def test_recovery_record_rejects_boolean_schema_version(tmp_path: Path) -> None:
    output = tmp_path / "fixtures"
    staging = tmp_path / "staging"
    _write_valid_fixture_set(output, "1" * 40, "original")
    _write_valid_fixture_set(staging, "2" * 40, "staged")
    output_snapshot = generator._capture_fixture_set_snapshot(output, "snapshot failed")
    staged_snapshot = generator._capture_fixture_set_snapshot(staging, "snapshot failed")
    parent_identity = generator._path_identity(tmp_path, "parent changed")
    backup = tmp_path / f".fixtures.backup-{'a' * 32}"
    record = generator._write_recovery_record(
        output,
        backup,
        parent_identity,
        output_snapshot,
        staged_snapshot,
    )
    payload = json.loads(record.path.read_text(encoding="ascii"))
    payload["schema_version"] = True
    _write_canonical_json(record.path, payload)
    before = record.path.read_bytes()

    with pytest.raises(
        RuntimeError,
        match=generator._RECOVERY_ERROR,
    ):
        generator._load_recovery_record(output)

    assert record.path.read_bytes() == before
    _assert_fixture_marker(output, "original")
    assert not backup.exists()


@pytest.mark.parametrize(
    "backup_name",
    [
        pytest.param("../unvalidated-backup", id="traversal"),
        pytest.param(f".other-output.backup-{'f' * 32}", id="arbitrary-sibling"),
    ],
)
def test_recovery_record_rejects_unvalidated_backup_path(
    tmp_path: Path,
    backup_name: str,
) -> None:
    output = tmp_path / "fixtures"
    staging = tmp_path / "staging"
    _write_valid_fixture_set(output, "1" * 40, "original")
    _write_valid_fixture_set(staging, "2" * 40, "staged")
    output_snapshot = generator._capture_fixture_set_snapshot(output, "snapshot failed")
    staged_snapshot = generator._capture_fixture_set_snapshot(staging, "snapshot failed")
    parent_identity = generator._path_identity(tmp_path, "parent changed")
    backup = tmp_path / f".fixtures.backup-{'f' * 32}"
    record = generator._write_recovery_record(
        output,
        backup,
        parent_identity,
        output_snapshot,
        staged_snapshot,
    )
    payload = json.loads(record.path.read_text(encoding="ascii"))
    payload["backup_name"] = backup_name
    _write_canonical_json(record.path, payload)
    before = record.path.read_bytes()

    with pytest.raises(RuntimeError, match=generator._RECOVERY_ERROR):
        generator._load_recovery_record(output)

    assert record.path.read_bytes() == before
    _assert_fixture_marker(output, "original")


@pytest.mark.parametrize(
    "output_name",
    [
        pytest.param("../fixtures", id="traversal"),
        pytest.param("other-output", id="wrong-normalized-component"),
    ],
)
def test_recovery_record_rejects_unvalidated_recorded_output_name(
    tmp_path: Path,
    output_name: str,
) -> None:
    output = tmp_path / "fixtures"
    staging = tmp_path / "staging"
    _write_valid_fixture_set(output, "1" * 40, "original")
    _write_valid_fixture_set(staging, "2" * 40, "staged")
    output_snapshot = generator._capture_fixture_set_snapshot(output, "snapshot failed")
    staged_snapshot = generator._capture_fixture_set_snapshot(staging, "snapshot failed")
    parent_identity = generator._path_identity(tmp_path, "parent changed")
    backup = _publication_backup_path(output, "f" * 32)
    record = generator._write_recovery_record(
        output,
        backup,
        parent_identity,
        output_snapshot,
        staged_snapshot,
    )
    payload = json.loads(record.path.read_text(encoding="ascii"))
    payload["output_name"] = output_name
    payload["backup_name"] = f".{output_name}.backup-{'f' * 32}"
    _write_canonical_json(record.path, payload)
    before = record.path.read_bytes()

    with pytest.raises(RuntimeError, match=generator._RECOVERY_ERROR):
        generator._load_recovery_record(output)

    assert record.path.read_bytes() == before
    _assert_fixture_marker(output, "original")
    assert not backup.exists()


def test_recovery_write_preserves_body_baseexception_over_close_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PrimaryWriteFailure(BaseException):
        pass

    output = tmp_path / "fixtures"
    staging = tmp_path / "staging"
    _write_valid_fixture_set(output, "1" * 40, "original")
    _write_valid_fixture_set(staging, "2" * 40, "staged")
    output_snapshot = generator._capture_fixture_set_snapshot(output, "snapshot failed")
    staged_snapshot = generator._capture_fixture_set_snapshot(staging, "snapshot failed")
    parent_identity = generator._path_identity(tmp_path, "parent changed")
    recovery_path = tmp_path / ".fixtures.publication-recovery.json"
    real_open = generator.os.open
    real_close = generator.os.close
    real_write = generator.os.write
    descriptors: list[int] = []
    primary = PrimaryWriteFailure("primary write failure")

    def _record_open(path: Any, *args: Any, **kwargs: Any) -> int:
        descriptor = real_open(path, *args, **kwargs)
        if Path(path).parent == tmp_path and Path(path).name.startswith(f"{recovery_path.name}.tmp-"):
            descriptors.append(descriptor)
        return descriptor

    def _fail_write(descriptor: int, data: bytes) -> int:
        if descriptor in descriptors:
            raise primary
        return real_write(descriptor, data)

    def _fail_close(descriptor: int) -> None:
        if descriptor in descriptors:
            raise OSError("secondary close failure")
        real_close(descriptor)

    monkeypatch.setattr(generator.os, "open", _record_open)
    monkeypatch.setattr(generator.os, "write", _fail_write)
    monkeypatch.setattr(generator.os, "close", _fail_close)
    try:
        with pytest.raises(PrimaryWriteFailure) as exc_info:
            generator._write_recovery_record(
                output,
                tmp_path / f".fixtures.backup-{'b' * 32}",
                parent_identity,
                output_snapshot,
                staged_snapshot,
            )
        assert exc_info.value is primary
    finally:
        for descriptor in descriptors:
            try:
                real_close(descriptor)
            except OSError:
                pass
    assert not recovery_path.exists()
    assert not list(tmp_path.glob(f"{recovery_path.name}.tmp-*"))


def test_recovery_read_preserves_body_baseexception_over_close_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PrimaryReadFailure(BaseException):
        pass

    recovery_path = tmp_path / "recovery.json"
    recovery_path.write_bytes(b"{}\n")
    recovery_path.chmod(0o600)
    real_open = generator.os.open
    real_close = generator.os.close
    descriptors: list[int] = []
    primary = PrimaryReadFailure("primary read failure")

    def _record_open(path: Any, *args: Any, **kwargs: Any) -> int:
        descriptor = real_open(path, *args, **kwargs)
        if Path(path) == recovery_path:
            descriptors.append(descriptor)
        return descriptor

    def _fail_read(descriptor: int, _size: int) -> bytes:
        if descriptor in descriptors:
            raise primary
        raise AssertionError("unexpected descriptor")

    def _fail_close(descriptor: int) -> None:
        if descriptor in descriptors:
            raise OSError("secondary close failure")
        real_close(descriptor)

    monkeypatch.setattr(generator.os, "open", _record_open)
    monkeypatch.setattr(generator.os, "read", _fail_read)
    monkeypatch.setattr(generator.os, "close", _fail_close)
    try:
        with pytest.raises(PrimaryReadFailure) as exc_info:
            generator._read_recovery_file(recovery_path)
        assert exc_info.value is primary
    finally:
        for descriptor in descriptors:
            try:
                real_close(descriptor)
            except OSError:
                pass


def test_recovery_directory_fsync_preserves_body_baseexception_over_close_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PrimaryFsyncFailure(BaseException):
        pass

    parent_identity = generator._path_identity(tmp_path, "parent changed")
    real_open = generator.os.open
    real_close = generator.os.close
    descriptors: list[int] = []
    primary = PrimaryFsyncFailure("primary fsync failure")

    def _record_open(path: Any, *args: Any, **kwargs: Any) -> int:
        descriptor = real_open(path, *args, **kwargs)
        if Path(path) == tmp_path:
            descriptors.append(descriptor)
        return descriptor

    def _fail_fsync(descriptor: int) -> None:
        if descriptor in descriptors:
            raise primary
        raise AssertionError("unexpected descriptor")

    def _fail_close(descriptor: int) -> None:
        if descriptor in descriptors:
            raise OSError("secondary close failure")
        real_close(descriptor)

    monkeypatch.setattr(generator.os, "open", _record_open)
    monkeypatch.setattr(generator.os, "fsync", _fail_fsync)
    monkeypatch.setattr(generator.os, "close", _fail_close)
    try:
        with pytest.raises(PrimaryFsyncFailure) as exc_info:
            generator._fsync_directory(tmp_path, parent_identity, generator._RECOVERY_ERROR)
        assert exc_info.value is primary
    finally:
        for descriptor in descriptors:
            try:
                real_close(descriptor)
            except OSError:
                pass


def test_cli_rejects_source_root_at_a_different_commit(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_root, source_commit = _create_clean_source_root(tmp_path)
    mismatched_commit = "0" * 40
    assert source_commit != mismatched_commit

    exit_code = generator.main(
        [
            "--predecessor-commit",
            mismatched_commit,
            "--output",
            str(tmp_path / "output"),
            "--source-root",
            str(source_root),
        ]
    )

    assert exit_code == 2
    assert "does not match source-root HEAD" in capsys.readouterr().err


def test_cli_rejects_dirty_source_root(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_root, source_commit = _create_clean_source_root(tmp_path)
    (source_root / "untracked.txt").write_text("dirty\n", encoding="utf-8")

    exit_code = generator.main(
        [
            "--predecessor-commit",
            source_commit,
            "--output",
            str(tmp_path / "output"),
            "--source-root",
            str(source_root),
        ]
    )

    assert exit_code == 2
    assert "source-root is not clean" in capsys.readouterr().err


def test_write_failure_leaves_existing_output_untouched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root, source_commit = _create_clean_source_root(tmp_path)
    output = tmp_path / "fixtures"
    _write_valid_fixture_set(output, "1" * 40, "original")
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    payloads = _fixture_payloads("replacement")
    monkeypatch.setattr(generator, "build_case_payloads", lambda _source_root: payloads)
    original_write_json = generator._write_json
    write_count = 0

    def _fail_during_write(path: Path, payload: object) -> None:
        nonlocal write_count
        write_count += 1
        if write_count == 2:
            raise RuntimeError("injected write failure")
        original_write_json(path, payload)

    monkeypatch.setattr(generator, "_write_json", _fail_during_write)

    with pytest.raises(RuntimeError, match="injected write failure"):
        generator.generate_fixtures(
            source_commit,
            output,
            source_root=source_root,
        )

    after = {path.name: path.read_bytes() for path in output.iterdir()}
    assert after == before


def test_parent_identity_failure_happens_before_staging_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_root, source_commit = _create_clean_source_root(tmp_path)
    output = tmp_path / "fixtures"
    _isolated_lock_path(tmp_path, monkeypatch, output)
    monkeypatch.setattr(
        generator,
        "build_case_payloads",
        lambda _source_root: _fixture_payloads("replacement"),
    )
    real_path_identity = generator._path_identity
    sensitive_marker = "sensitive parent identity failure"
    sensitive_detail = f"{sensitive_marker} at {tmp_path}"

    def _fail_parent_identity(path: Path, error_message: str) -> tuple[int, int, int]:
        if path == output.parent:
            raise RuntimeError(sensitive_detail)
        return real_path_identity(path, error_message)

    monkeypatch.setattr(generator, "_path_identity", _fail_parent_identity)
    real_mkdtemp = generator.tempfile.mkdtemp
    staging_paths: list[Path] = []

    def _record_mkdtemp(*args: Any, **kwargs: Any) -> str:
        staging = Path(real_mkdtemp(*args, **kwargs))
        staging_paths.append(staging)
        return str(staging)

    monkeypatch.setattr(generator.tempfile, "mkdtemp", _record_mkdtemp)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(
            RuntimeError,
            match="^Fixture staging directory could not be cleaned up$",
        ) as exc_info:
            generator.generate_fixtures(
                source_commit,
                output,
                source_root=source_root,
            )

    assert staging_paths == []
    assert not list(output.parent.glob(f".{output.name}.staging-*"))
    assert str(tmp_path) not in str(exc_info.value)
    assert sensitive_marker not in str(exc_info.value)
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("staging_kind", ["symlink", "junction-like"])
def test_link_like_staging_substitution_before_identity_does_not_write_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    staging_kind: str,
) -> None:
    source_root, source_commit = _create_clean_source_root(tmp_path)
    output = tmp_path / "fixtures"
    _isolated_lock_path(tmp_path, monkeypatch, output)
    protected_marker = "protected-target-content"
    target = tmp_path / "private-link-target"
    _write_valid_fixture_set(target, "f" * 40, protected_marker)
    target_before = _snapshot_path(target)
    build_calls = 0

    def _record_payload_build(_source_root: Path) -> dict[str, dict[str, Any]]:
        nonlocal build_calls
        build_calls += 1
        return _fixture_payloads("must-not-write")

    monkeypatch.setattr(generator, "build_case_payloads", _record_payload_build)
    real_mkdtemp = generator.tempfile.mkdtemp
    real_stat = generator.os.stat
    real_lstat = generator.os.lstat
    real_rmtree = generator.shutil.rmtree
    staging_paths: list[Path] = []
    original_staging = tmp_path / "original-staging"

    def _substitute_after_mkdtemp(*args: Any, **kwargs: Any) -> str:
        staging = Path(real_mkdtemp(*args, **kwargs))
        staging.replace(original_staging)
        _symlink_or_skip(staging, target, target_is_directory=True)
        staging_paths.append(staging)
        return str(staging)

    monkeypatch.setattr(generator.tempfile, "mkdtemp", _substitute_after_mkdtemp)

    if staging_kind == "junction-like":

        def _junction_stat(
            path: os.PathLike[str] | str,
            *args: Any,
            **kwargs: Any,
        ) -> os.stat_result:
            if staging_paths and Path(path) == staging_paths[0] and kwargs.get("follow_symlinks") is False:
                return real_stat(target, follow_symlinks=False)
            return real_stat(path, *args, **kwargs)

        def _junction_lstat(
            path: os.PathLike[str] | str,
            *args: Any,
            **kwargs: Any,
        ) -> os.stat_result:
            if staging_paths and Path(path) == staging_paths[0]:
                return real_lstat(target)
            return real_lstat(path, *args, **kwargs)

        monkeypatch.setattr(generator.os, "stat", _junction_stat)
        monkeypatch.setattr(generator.os, "lstat", _junction_lstat)
        monkeypatch.setattr(
            Path,
            "is_junction",
            lambda path: bool(staging_paths and path == staging_paths[0]),
            raising=False,
        )

    write_calls: list[Path] = []
    write_fixture_set = generator._write_fixture_set

    def _record_fixture_write(
        staging: Path,
        predecessor_commit: str,
        payloads: dict[str, dict[str, Any]],
    ) -> None:
        write_calls.append(staging)
        write_fixture_set(staging, predecessor_commit, payloads)

    monkeypatch.setattr(generator, "_write_fixture_set", _record_fixture_write)
    cleanup_calls: list[Path] = []

    def _prevent_vulnerable_cleanup(path: Path, *args: Any, **kwargs: Any) -> None:
        if staging_paths and path == staging_paths[0]:
            cleanup_calls.append(path)
            raise OSError(f"sensitive cleanup failure at {tmp_path}")
        real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(generator.shutil, "rmtree", _prevent_vulnerable_cleanup)

    publication_error: RuntimeError | None = None
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        try:
            generator.generate_fixtures(
                source_commit,
                output,
                source_root=source_root,
            )
        except RuntimeError as exc:
            publication_error = exc

    assert build_calls == 1
    assert write_calls == []
    assert cleanup_calls == []
    assert publication_error is not None
    assert str(publication_error) == "Fixture staging directory could not be cleaned up"
    assert str(tmp_path) not in str(publication_error)
    assert protected_marker not in str(publication_error)
    assert len(staging_paths) == 1
    staging = staging_paths[0]
    assert stat.S_ISLNK(real_lstat(staging).st_mode)
    assert original_staging.is_dir()
    assert _snapshot_path(target) == target_before
    assert not output.exists()
    diagnostic = capsys.readouterr().err
    assert diagnostic == "warning: fixture staging directory retained for manual cleanup\n"
    assert str(tmp_path) not in diagnostic
    assert staging.name not in diagnostic
    assert target.name not in diagnostic
    assert protected_marker not in diagnostic


def test_parent_substitution_before_publication_does_not_publish_fixtures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_root, source_commit = _create_clean_source_root(tmp_path)
    publication_parent = tmp_path / "publication"
    publication_parent.mkdir()
    output = publication_parent / "fixtures"
    original_parent = tmp_path / "original-publication-parent"
    _isolated_lock_path(tmp_path, monkeypatch, output)
    monkeypatch.setattr(
        generator,
        "build_case_payloads",
        lambda _source_root: _fixture_payloads("replacement"),
    )
    replace_output_directory = generator._replace_output_directory
    substitution_performed = False

    def _substitute_parent_before_publication(
        staging: Path,
        target: Path,
        **kwargs: Any,
    ) -> None:
        nonlocal substitution_performed
        publication_parent.replace(original_parent)
        publication_parent.mkdir()
        (original_parent / staging.name).replace(staging)
        substitution_performed = True
        replace_output_directory(staging, target, **kwargs)

    monkeypatch.setattr(
        generator,
        "_replace_output_directory",
        _substitute_parent_before_publication,
    )

    with pytest.raises(
        RuntimeError,
        match="^Fixture output parent changed during publication$",
    ) as exc_info:
        generator.generate_fixtures(
            source_commit,
            output,
            source_root=source_root,
        )

    assert substitution_performed
    assert not output.exists()
    assert not (original_parent / output.name).exists()
    retained_staging = list(publication_parent.iterdir())
    assert len(retained_staging) == 1
    assert retained_staging[0].name.startswith(f".{output.name}.staging-")
    assert retained_staging[0].is_dir()
    assert str(tmp_path) not in str(exc_info.value)
    diagnostic = capsys.readouterr().err
    assert diagnostic == "warning: fixture staging directory retained for manual cleanup\n"
    assert str(tmp_path) not in diagnostic
    assert retained_staging[0].name not in diagnostic


def test_staging_substitution_before_publication_is_not_published(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_root, source_commit = _create_clean_source_root(tmp_path)
    output = tmp_path / "fixtures"
    _isolated_lock_path(tmp_path, monkeypatch, output)
    safe_marker = "validated-original"
    substitute_marker = "untrusted-substitute"
    substitute_commit = "f" * 40
    monkeypatch.setattr(
        generator,
        "build_case_payloads",
        lambda _source_root: _fixture_payloads(safe_marker),
    )
    replace_output_directory = generator._replace_output_directory
    validated_original = tmp_path / "validated-original-staging"
    substituted_staging: Path | None = None

    def _substitute_staging_before_publication(
        staging: Path,
        target: Path,
        **kwargs: Any,
    ) -> None:
        nonlocal substituted_staging
        staging.replace(validated_original)
        _write_valid_fixture_set(staging, substitute_commit, substitute_marker)
        substituted_staging = staging
        replace_output_directory(staging, target, **kwargs)

    monkeypatch.setattr(
        generator,
        "_replace_output_directory",
        _substitute_staging_before_publication,
    )

    publication_error: RuntimeError | None = None
    try:
        generator.generate_fixtures(
            source_commit,
            output,
            source_root=source_root,
        )
    except RuntimeError as exc:
        publication_error = exc

    assert substituted_staging is not None
    assert not output.exists()
    assert publication_error is not None
    assert str(publication_error) == "Fixture staging directory changed during publication"
    assert str(tmp_path) not in str(publication_error)
    _assert_fixture_marker(validated_original, safe_marker)
    _assert_fixture_marker(substituted_staging, substitute_marker)
    original_manifest = json.loads((validated_original / "manifest.json").read_text(encoding="ascii"))
    substitute_manifest = json.loads((substituted_staging / "manifest.json").read_text(encoding="ascii"))
    assert original_manifest["predecessor_commit"] == source_commit
    assert substitute_manifest["predecessor_commit"] == substitute_commit
    assert not list(tmp_path.glob(".fixtures.backup-*"))
    diagnostic = capsys.readouterr().err
    assert diagnostic == "warning: fixture staging directory retained for manual cleanup\n"
    assert str(tmp_path) not in diagnostic
    assert validated_original.name not in diagnostic
    assert substituted_staging.name not in diagnostic
    assert safe_marker not in diagnostic
    assert substitute_marker not in diagnostic


def test_staging_identity_failure_retains_staging_with_fixed_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_root, source_commit = _create_clean_source_root(tmp_path)
    output = tmp_path / "fixtures"
    _isolated_lock_path(tmp_path, monkeypatch, output)
    monkeypatch.setattr(
        generator,
        "build_case_payloads",
        lambda _source_root: _fixture_payloads("replacement"),
    )
    real_path_identity = generator._path_identity
    staging_paths: list[Path] = []
    sensitive_marker = "sensitive staging identity failure"
    sensitive_detail = f"{sensitive_marker} at {tmp_path}"

    real_mkdtemp = generator.tempfile.mkdtemp

    def _record_mkdtemp(*args: Any, **kwargs: Any) -> str:
        staging = Path(real_mkdtemp(*args, **kwargs))
        staging_paths.append(staging)
        return str(staging)

    monkeypatch.setattr(generator.tempfile, "mkdtemp", _record_mkdtemp)

    def _fail_staging_identity(path: Path, error_message: str) -> tuple[int, int, int]:
        if staging_paths and path == staging_paths[0]:
            raise RuntimeError(sensitive_detail)
        return real_path_identity(path, error_message)

    monkeypatch.setattr(generator, "_path_identity", _fail_staging_identity)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(
            RuntimeError,
            match="^Fixture staging directory could not be cleaned up$",
        ) as exc_info:
            generator.generate_fixtures(
                source_commit,
                output,
                source_root=source_root,
            )

    assert len(staging_paths) == 1
    staging = staging_paths[0]
    assert staging.is_dir()
    assert str(tmp_path) not in str(exc_info.value)
    assert sensitive_marker not in str(exc_info.value)
    diagnostic = capsys.readouterr().err
    assert diagnostic == "warning: fixture staging directory retained for manual cleanup\n"
    assert str(tmp_path) not in diagnostic
    assert staging.name not in diagnostic
    assert sensitive_marker not in diagnostic


def test_direct_staging_identity_baseexception_retains_staging_and_primary_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class DirectStagingIdentityFailure(BaseException):
        pass

    source_root, source_commit = _create_clean_source_root(tmp_path)
    output = tmp_path / "fixtures"
    _isolated_lock_path(tmp_path, monkeypatch, output)
    monkeypatch.setattr(
        generator,
        "build_case_payloads",
        lambda _source_root: _fixture_payloads("replacement"),
    )
    real_path_identity = generator._path_identity
    real_mkdtemp = generator.tempfile.mkdtemp
    report_staging_retained = generator._report_staging_retained
    staging_paths: list[Path] = []
    primary_message = "direct staging identity failure"
    primary_error = DirectStagingIdentityFailure(primary_message)
    primary_tracebacks_during_report: list[Any] = []

    def _record_mkdtemp(*args: Any, **kwargs: Any) -> str:
        staging = Path(real_mkdtemp(*args, **kwargs))
        staging_paths.append(staging)
        return str(staging)

    def _fail_staging_identity(path: Path, error_message: str) -> tuple[int, int, int]:
        if staging_paths and path == staging_paths[0]:
            raise primary_error
        return real_path_identity(path, error_message)

    def _record_retention_report() -> None:
        primary_tracebacks_during_report.append(primary_error.__traceback__)
        report_staging_retained()

    monkeypatch.setattr(generator.tempfile, "mkdtemp", _record_mkdtemp)
    monkeypatch.setattr(generator, "_path_identity", _fail_staging_identity)
    monkeypatch.setattr(generator, "_report_staging_retained", _record_retention_report)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(
            DirectStagingIdentityFailure,
            match=f"^{primary_message}$",
        ) as exc_info:
            generator.generate_fixtures(
                source_commit,
                output,
                source_root=source_root,
            )

    assert exc_info.value is primary_error
    assert type(exc_info.value) is DirectStagingIdentityFailure
    assert str(exc_info.value) == primary_message
    assert len(staging_paths) == 1
    staging = staging_paths[0]
    assert staging.is_dir()
    assert len(primary_tracebacks_during_report) == 1
    final_traceback = exc_info.value.__traceback__
    assert final_traceback is not None
    assert final_traceback.tb_next is primary_tracebacks_during_report[0]
    recorded_tail = primary_tracebacks_during_report[0]
    while recorded_tail is not None and recorded_tail.tb_next is not None:
        recorded_tail = recorded_tail.tb_next
    final_tail = final_traceback
    while final_tail.tb_next is not None:
        final_tail = final_tail.tb_next
    assert final_tail is recorded_tail
    diagnostic = capsys.readouterr().err
    assert diagnostic == "warning: fixture staging directory retained for manual cleanup\n"
    assert str(tmp_path) not in diagnostic
    assert staging.name not in diagnostic
    assert primary_message not in diagnostic


def test_staging_cleanup_failure_does_not_replace_primary_publication_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class PrimaryFixturePublicationError(BaseException):
        pass

    class DirectCleanupFailure(BaseException):
        pass

    source_root, source_commit = _create_clean_source_root(tmp_path)
    output = tmp_path / "fixtures"
    _isolated_lock_path(tmp_path, monkeypatch, output)
    monkeypatch.setattr(
        generator,
        "build_case_payloads",
        lambda _source_root: _fixture_payloads("replacement"),
    )
    primary_message = "primary fixture publication failure"
    primary_error = PrimaryFixturePublicationError(primary_message)
    sensitive_marker = "sensitive staging cleanup failure"
    sensitive_path = str(tmp_path / "private-fixture-staging")
    cleanup_detail = f"{sensitive_marker} at {sensitive_path}"
    cleanup_attempts: list[Path] = []
    primary_tracebacks_during_cleanup: list[Any] = []

    def _fail_publication(
        _staging: Path,
        _output: Path,
        *,
        expected_parent_identity: tuple[int, int, int] | None = None,
        expected_staging_identity: tuple[int, int, int] | None = None,
    ) -> None:
        del expected_parent_identity, expected_staging_identity
        raise primary_error

    def _fail_staging_cleanup(path: Path) -> None:
        cleanup_attempts.append(path)
        primary_tracebacks_during_cleanup.append(primary_error.__traceback__)
        raise DirectCleanupFailure(cleanup_detail)

    monkeypatch.setattr(generator, "_replace_output_directory", _fail_publication)
    monkeypatch.setattr(generator.shutil, "rmtree", _fail_staging_cleanup)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(
            PrimaryFixturePublicationError,
            match="^primary fixture publication failure$",
        ) as exc_info:
            generator.generate_fixtures(
                source_commit,
                output,
                source_root=source_root,
            )

    assert exc_info.value is primary_error
    assert type(exc_info.value) is PrimaryFixturePublicationError
    assert str(exc_info.value) == primary_message
    assert len(primary_tracebacks_during_cleanup) == 1
    final_traceback = exc_info.value.__traceback__
    assert final_traceback is not None
    assert final_traceback.tb_next is primary_tracebacks_during_cleanup[0]
    recorded_tail = primary_tracebacks_during_cleanup[0]
    while recorded_tail is not None and recorded_tail.tb_next is not None:
        recorded_tail = recorded_tail.tb_next
    final_tail = final_traceback
    while final_tail is not None and final_tail.tb_next is not None:
        final_tail = final_tail.tb_next
    assert final_tail is recorded_tail
    assert len(cleanup_attempts) == 1
    assert cleanup_attempts[0].parent == output.parent
    assert cleanup_attempts[0].name.startswith(f".{output.name}.staging-")
    formatted_diagnostic = "".join(
        traceback.format_exception(
            type(exc_info.value),
            exc_info.value,
            exc_info.value.__traceback__,
            chain=True,
        )
    )
    assert sensitive_marker not in formatted_diagnostic
    assert sensitive_path not in formatted_diagnostic
    assert "_fail_publication" in formatted_diagnostic
    assert "_fail_staging_cleanup" not in formatted_diagnostic
    diagnostic = capsys.readouterr().err
    assert diagnostic == "warning: fixture staging directory retained for manual cleanup\n"
    assert sensitive_marker not in diagnostic
    assert sensitive_path not in diagnostic


def test_recreated_staging_path_is_retained_after_cooperative_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root, source_commit = _create_clean_source_root(tmp_path)
    output = tmp_path / "fixtures"
    _isolated_lock_path(tmp_path, monkeypatch, output)
    monkeypatch.setattr(
        generator,
        "build_case_payloads",
        lambda _source_root: _fixture_payloads("published"),
    )
    replace_output_directory = generator._replace_output_directory
    staging_paths: list[Path] = []
    cleanup_attempts: list[Path] = []
    replacement_marker = "replacement staging object"
    cleanup_detail = f"sensitive cleanup failure at {tmp_path}"

    def _publish_and_leave_staging(
        staging: Path,
        target: Path,
        *,
        expected_parent_identity: tuple[int, int, int] | None = None,
        expected_staging_identity: tuple[int, int, int] | None = None,
    ) -> None:
        replace_output_directory(
            staging,
            target,
            expected_parent_identity=expected_parent_identity,
            expected_staging_identity=expected_staging_identity,
        )
        staging.mkdir()
        (staging / "replacement.txt").write_text(replacement_marker, encoding="utf-8")
        staging_paths.append(staging)

    def _record_unexpected_staging_cleanup(path: Path) -> None:
        cleanup_attempts.append(path)
        raise OSError(cleanup_detail)

    monkeypatch.setattr(
        generator,
        "_replace_output_directory",
        _publish_and_leave_staging,
    )
    monkeypatch.setattr(
        generator.shutil,
        "rmtree",
        _record_unexpected_staging_cleanup,
    )

    with pytest.raises(
        RuntimeError,
        match="^Fixture staging directory could not be cleaned up$",
    ) as exc_info:
        generator.generate_fixtures(
            source_commit,
            output,
            source_root=source_root,
        )

    assert len(staging_paths) == 1
    replacement = staging_paths[0]
    assert (replacement / "replacement.txt").read_text(encoding="utf-8") == replacement_marker
    assert cleanup_attempts == []
    assert exc_info.value.__suppress_context__
    assert str(exc_info.value) == "Fixture staging directory could not be cleaned up"
    assert "sensitive" not in str(exc_info.value)
    assert str(tmp_path) not in str(exc_info.value)
    _assert_fixture_marker(output, "published")


def test_original_staging_cleanup_oserror_is_sanitized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root, source_commit = _create_clean_source_root(tmp_path)
    output = tmp_path / "fixtures"
    _isolated_lock_path(tmp_path, monkeypatch, output)
    monkeypatch.setattr(
        generator,
        "build_case_payloads",
        lambda _source_root: _fixture_payloads("unpublished"),
    )
    staging_paths: list[Path] = []
    cleanup_attempts: list[Path] = []
    cleanup_detail = f"sensitive cleanup failure at {tmp_path}"

    def _leave_original_staging(
        staging: Path,
        _target: Path,
        *,
        expected_parent_identity: tuple[int, int, int] | None = None,
        expected_staging_identity: tuple[int, int, int] | None = None,
    ) -> None:
        del expected_parent_identity, expected_staging_identity
        staging_paths.append(staging)

    def _fail_staging_cleanup(path: Path) -> None:
        cleanup_attempts.append(path)
        raise OSError(cleanup_detail)

    monkeypatch.setattr(
        generator,
        "_replace_output_directory",
        _leave_original_staging,
    )
    monkeypatch.setattr(generator.shutil, "rmtree", _fail_staging_cleanup)

    with pytest.raises(
        RuntimeError,
        match="^Fixture staging directory could not be cleaned up$",
    ) as exc_info:
        generator.generate_fixtures(
            source_commit,
            output,
            source_root=source_root,
        )

    assert len(staging_paths) == 1
    assert cleanup_attempts == staging_paths
    assert exc_info.value.__suppress_context__
    assert str(exc_info.value) == "Fixture staging directory could not be cleaned up"
    assert "sensitive" not in str(exc_info.value)
    assert str(tmp_path) not in str(exc_info.value)
    assert not output.exists()
    _assert_fixture_marker(staging_paths[0], "unpublished")


@pytest.mark.parametrize("staging_kind", ["symlink", "junction-like"])
def test_standalone_publication_rejects_link_like_staging_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    staging_kind: str,
) -> None:
    output = tmp_path / "fixtures"
    staging = tmp_path / "staging"
    if staging_kind == "symlink":
        fixture_root = tmp_path / "fixture-root"
        _write_valid_fixture_set(fixture_root, "1" * 40, "staged")
        _symlink_or_skip(staging, fixture_root, target_is_directory=True)
    else:
        fixture_root = staging
        _write_valid_fixture_set(fixture_root, "1" * 40, "staged")
        monkeypatch.setattr(
            Path,
            "is_junction",
            lambda path: path == staging,
            raising=False,
        )
    before_fixture_root = _snapshot_path(fixture_root)

    with pytest.raises(
        RuntimeError,
        match="^Fixture staging directory changed during publication$",
    ) as exc_info:
        generator._replace_output_directory(staging, output)

    assert not output.exists()
    assert _snapshot_path(fixture_root) == before_fixture_root
    if staging_kind == "symlink":
        assert staging.is_symlink()
    else:
        assert staging.is_dir()
    assert str(tmp_path) not in str(exc_info.value)


@pytest.mark.parametrize(
    "identity_state",
    ["missing-parent", "mismatched-parent", "mismatched-staging"],
)
def test_malformed_staging_validation_precedes_identity_checks(
    tmp_path: Path,
    identity_state: str,
) -> None:
    staging = tmp_path / "staging"
    _write_valid_fixture_set(staging, "1" * 40, "staged")
    (staging / "content.json").write_text("{sensitive malformed", encoding="ascii")
    before_staging = _snapshot_path(staging)

    replace_kwargs: dict[str, tuple[int, int, int]] = {}
    if identity_state == "missing-parent":
        output = tmp_path / "missing-parent" / "fixtures"
    else:
        output = tmp_path / "fixtures"
        identity_path = output.parent if identity_state == "mismatched-parent" else staging
        path_identity = generator._path_identity(identity_path, "identity failure")
        identity_key = (
            "expected_parent_identity" if identity_state == "mismatched-parent" else "expected_staging_identity"
        )
        replace_kwargs[identity_key] = (
            path_identity[0],
            path_identity[1],
            path_identity[2] + 1,
        )

    with pytest.raises(
        RuntimeError,
        match=r"^Fixture set contains invalid JSON: content\.json$",
    ) as exc_info:
        generator._replace_output_directory(staging, output, **replace_kwargs)

    assert _snapshot_path(staging) == before_staging
    assert not output.exists()
    if identity_state == "missing-parent":
        assert not output.parent.exists()
    assert str(tmp_path) not in str(exc_info.value)
    assert "sensitive" not in str(exc_info.value)


@pytest.mark.parametrize("identity_boundary", ["parent", "output"])
def test_publication_rejects_zero_inode_with_dedicated_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    identity_boundary: str,
) -> None:
    output = tmp_path / "fixtures"
    _write_valid_fixture_set(output, "1" * 40, "old-output")
    staging = tmp_path / "staging"
    _write_valid_fixture_set(staging, "2" * 40, "new-output")
    before_output = _snapshot_path(output)
    before_staging = _snapshot_path(staging)
    target = output.parent if identity_boundary == "parent" else output
    real_stat = generator.os.stat

    def _zero_identity(
        path: os.PathLike[str] | str,
        *args: Any,
        **kwargs: Any,
    ) -> os.stat_result:
        metadata = real_stat(path, *args, **kwargs)
        if Path(path) == target and kwargs.get("follow_symlinks") is False:
            return _stat_with_identity_value(metadata, "st_ino", 0)
        return metadata

    monkeypatch.setattr(generator.os, "stat", _zero_identity)

    with pytest.raises(
        RuntimeError,
        match="^Fixture filesystem does not provide stable identity$",
    ) as exc_info:
        generator._replace_output_directory(staging, output)

    assert _snapshot_path(output) == before_output
    assert _snapshot_path(staging) == before_staging
    assert str(tmp_path) not in str(exc_info.value)


def test_swap_failure_restores_old_output_and_propagates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "fixtures"
    _write_valid_fixture_set(output, "1" * 40, "old-output")
    before = _snapshot_path(output)
    staging = tmp_path / "staging"
    _write_valid_fixture_set(staging, "2" * 40, "new-output")
    original_replace = Path.replace

    def _fail_staging_swap(path: Path, target: Path) -> Path:
        if path == staging:
            raise OSError("injected staging swap failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", _fail_staging_swap)

    with pytest.raises(OSError, match="injected staging swap failure"):
        generator._replace_output_directory(staging, output)

    assert _snapshot_path(output) == before
    assert json.loads((staging / "content.json").read_text(encoding="ascii"))["cases"] == [{"marker": "new-output"}]
    assert not list(tmp_path.glob(".fixtures.backup-*"))


def test_staging_substitution_during_output_inspection_does_not_rename_old_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "fixtures"
    old_marker = "old-output"
    _write_valid_fixture_set(output, "1" * 40, old_marker)
    before_output = _snapshot_path(output)
    staging = tmp_path / "staging"
    safe_marker = "validated-staging"
    substitute_marker = "substituted-staging"
    _write_valid_fixture_set(staging, "2" * 40, safe_marker)
    parent_identity = generator._path_identity(output.parent, "identity failure")
    staging_identity = generator._path_identity(staging, "identity failure")
    validated_staging = tmp_path / "validated-staging"
    original_validate = generator._validate_existing_output
    original_replace = Path.replace
    output_rename_targets: list[Path] = []

    def _substitute_during_output_inspection(
        path: Path,
    ) -> tuple[int, int, int] | None:
        identity = original_validate(path)
        original_replace(staging, validated_staging)
        _write_valid_fixture_set(staging, "3" * 40, substitute_marker)
        return identity

    def _record_output_rename(path: Path, target: Path) -> Path:
        if path == output:
            output_rename_targets.append(target)
        return original_replace(path, target)

    monkeypatch.setattr(
        generator,
        "_validate_existing_output",
        _substitute_during_output_inspection,
    )
    monkeypatch.setattr(Path, "replace", _record_output_rename)

    with pytest.raises(
        RuntimeError,
        match="^Fixture staging directory changed during publication$",
    ) as exc_info:
        generator._replace_output_directory(
            staging,
            output,
            expected_parent_identity=parent_identity,
            expected_staging_identity=staging_identity,
        )

    assert output_rename_targets == []
    assert _snapshot_path(output) == before_output
    _assert_fixture_marker(output, old_marker)
    _assert_fixture_marker(validated_staging, safe_marker)
    _assert_fixture_marker(staging, substitute_marker)
    assert not list(tmp_path.glob(".fixtures.backup-*"))
    diagnostic = str(exc_info.value)
    assert str(tmp_path) not in diagnostic
    assert old_marker not in diagnostic
    assert safe_marker not in diagnostic
    assert substitute_marker not in diagnostic
    assert capsys.readouterr().err == ""


def test_post_rename_identity_check_failure_restores_old_output_and_propagates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "fixtures"
    _write_valid_fixture_set(output, "1" * 40, "old-output")
    staging = tmp_path / "staging"
    _write_valid_fixture_set(staging, "2" * 40, "new-output")
    original_require_absent = generator._require_path_absent
    injected = False

    def _fail_once_after_output_rename(path: Path, error_message: str) -> None:
        nonlocal injected
        original_require_absent(path, error_message)
        if path == output and not injected:
            injected = True
            raise RuntimeError("primary post-rename identity failure")

    monkeypatch.setattr(generator, "_require_path_absent", _fail_once_after_output_rename)

    with pytest.raises(RuntimeError, match="^primary post-rename identity failure$"):
        generator._replace_output_directory(staging, output)

    assert injected
    _assert_fixture_marker(output, "old-output")
    _assert_fixture_marker(staging, "new-output")
    assert not list(tmp_path.glob(".fixtures.backup-*"))


def test_in_place_staging_mutation_after_validation_does_not_replace_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "fixtures"
    _write_valid_fixture_set(output, "1" * 40, "old-output")
    staging = tmp_path / "staging"
    _write_valid_fixture_set(staging, "2" * 40, "new-output")
    before_output = _snapshot_path(output)
    original_validate = generator._validate_existing_output

    def _mutate_staging_after_validation(path: Path) -> object:
        identity = original_validate(path)
        _write_canonical_json(
            staging / "content.json",
            _fixture_payloads("tampered-sensitive-staging")["content"],
        )
        return identity

    monkeypatch.setattr(
        generator,
        "_validate_existing_output",
        _mutate_staging_after_validation,
    )

    with pytest.raises(
        RuntimeError,
        match="^Fixture staging directory changed during publication$",
    ) as exc_info:
        generator._replace_output_directory(staging, output)

    assert _snapshot_path(output) == before_output
    _assert_fixture_marker(output, "old-output")
    assert not list(tmp_path.glob(".fixtures.backup-*"))
    assert str(tmp_path) not in str(exc_info.value)
    assert "tampered-sensitive-staging" not in str(exc_info.value)


def test_in_place_output_mutation_after_validation_aborts_before_first_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "fixtures"
    _write_valid_fixture_set(output, "1" * 40, "old-output")
    staging = tmp_path / "staging"
    _write_valid_fixture_set(staging, "2" * 40, "new-output")
    original_validate = generator._validate_existing_output

    def _mutate_output_after_validation(path: Path) -> object:
        identity = original_validate(path)
        _write_canonical_json(
            output / "content.json",
            _fixture_payloads("tampered-sensitive-output")["content"],
        )
        return identity

    monkeypatch.setattr(
        generator,
        "_validate_existing_output",
        _mutate_output_after_validation,
    )

    with pytest.raises(
        RuntimeError,
        match="^Fixture output changed during publication$",
    ) as exc_info:
        generator._replace_output_directory(staging, output)

    _assert_fixture_marker(staging, "new-output")
    assert json.loads((output / "content.json").read_text(encoding="ascii"))["cases"] == [
        {"marker": "tampered-sensitive-output"}
    ]
    assert not list(tmp_path.glob(".fixtures.backup-*"))
    assert str(tmp_path) not in str(exc_info.value)
    assert "tampered-sensitive-output" not in str(exc_info.value)


def test_output_mutation_after_durable_record_aborts_before_first_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "fixtures"
    _write_valid_fixture_set(output, "1" * 40, "old-output")
    staging = tmp_path / "staging"
    _write_valid_fixture_set(staging, "2" * 40, "new-output")
    original_write_recovery_record = generator._write_recovery_record
    original_replace = Path.replace
    output_rename_calls = 0

    def _mutate_after_durable_record(*args: Any, **kwargs: Any) -> Any:
        record = original_write_recovery_record(*args, **kwargs)
        _write_canonical_json(
            output / "content.json",
            _fixture_payloads("tampered-after-journal")["content"],
        )
        return record

    def _record_output_rename(path: Path, target: Path) -> Path:
        nonlocal output_rename_calls
        if path == output:
            output_rename_calls += 1
        return original_replace(path, target)

    monkeypatch.setattr(generator, "_write_recovery_record", _mutate_after_durable_record)
    monkeypatch.setattr(Path, "replace", _record_output_rename)

    with pytest.raises(RuntimeError, match="^Fixture output changed during publication$"):
        generator._replace_output_directory(staging, output)

    assert output_rename_calls == 0
    assert json.loads((output / "content.json").read_text(encoding="ascii"))["cases"] == [
        {"marker": "tampered-after-journal"}
    ]
    _assert_fixture_marker(staging, "new-output")
    assert not list(tmp_path.glob(".fixtures.backup-*"))
    assert not list(tmp_path.glob(".fixtures.publication-recovery.json"))


def test_staging_mutation_during_final_backup_scan_aborts_before_second_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "fixtures"
    _write_valid_fixture_set(output, "1" * 40, "old-output")
    staging = tmp_path / "staging"
    _write_valid_fixture_set(staging, "2" * 40, "new-output")
    original_require_snapshot = generator._require_fixture_set_snapshot
    original_replace = Path.replace
    staging_rename_calls = 0
    mutated = False

    def _mutate_during_backup_scan(path: Path, expected: Any, error_message: str) -> None:
        nonlocal mutated
        original_require_snapshot(path, expected, error_message)
        if ".backup-" in path.name and not mutated:
            mutated = True
            _write_canonical_json(
                staging / "content.json",
                _fixture_payloads("tampered-before-second-rename")["content"],
            )

    def _record_staging_rename(path: Path, target: Path) -> Path:
        nonlocal staging_rename_calls
        if path == staging:
            staging_rename_calls += 1
        return original_replace(path, target)

    monkeypatch.setattr(generator, "_require_fixture_set_snapshot", _mutate_during_backup_scan)
    monkeypatch.setattr(Path, "replace", _record_staging_rename)

    with pytest.raises(RuntimeError, match="^Fixture staging directory changed during publication$"):
        generator._replace_output_directory(staging, output)

    assert mutated
    assert staging_rename_calls == 0
    _assert_fixture_marker(output, "old-output")
    assert json.loads((staging / "content.json").read_text(encoding="ascii"))["cases"] == [
        {"marker": "tampered-before-second-rename"}
    ]
    assert not list(tmp_path.glob(".fixtures.backup-*"))
    assert not list(tmp_path.glob(".fixtures.publication-recovery.json"))


def test_live_rollback_rejects_mutated_backup_and_retains_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "fixtures"
    _write_valid_fixture_set(output, "1" * 40, "old-output")
    staging = tmp_path / "staging"
    _write_valid_fixture_set(staging, "2" * 40, "new-output")
    original_replace = Path.replace
    backup: Path | None = None

    def _mutate_backup_after_first_rename(path: Path, target: Path) -> Path:
        nonlocal backup
        result = original_replace(path, target)
        if path == output:
            backup = target
            _write_canonical_json(
                target / "content.json",
                _fixture_payloads("tampered-live-backup")["content"],
            )
        return result

    monkeypatch.setattr(Path, "replace", _mutate_backup_after_first_rename)

    with pytest.raises(
        RuntimeError,
        match=("^Fixture output rollback could not be completed safely; " "manual recovery is required$"),
    ):
        generator._replace_output_directory(staging, output)

    assert backup is not None
    assert not output.exists()
    assert json.loads((backup / "content.json").read_text(encoding="ascii"))["cases"] == [
        {"marker": "tampered-live-backup"}
    ]
    _assert_fixture_marker(staging, "new-output")
    assert len(list(tmp_path.glob(".fixtures.publication-recovery.json"))) == 1


def test_live_rollback_revalidates_restored_output_and_retains_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "fixtures"
    _write_valid_fixture_set(output, "1" * 40, "old-output")
    staging = tmp_path / "staging"
    _write_valid_fixture_set(staging, "2" * 40, "new-output")
    original_replace = Path.replace
    backup: Path | None = None

    def _fail_publication_then_mutate_restore(path: Path, target: Path) -> Path:
        nonlocal backup
        if path == output:
            backup = target
            return original_replace(path, target)
        if path == staging:
            raise OSError("injected second rename failure")
        result = original_replace(path, target)
        if backup is not None and path == backup and target == output:
            _write_canonical_json(
                output / "content.json",
                _fixture_payloads("tampered-restored-output")["content"],
            )
        return result

    monkeypatch.setattr(Path, "replace", _fail_publication_then_mutate_restore)

    with pytest.raises(
        RuntimeError,
        match=("^Fixture output rollback could not be completed safely; " "manual recovery is required$"),
    ):
        generator._replace_output_directory(staging, output)

    assert backup is not None
    assert not output.exists()
    assert json.loads((backup / "content.json").read_text(encoding="ascii"))["cases"] == [
        {"marker": "tampered-restored-output"}
    ]
    _assert_fixture_marker(staging, "new-output")
    assert len(list(tmp_path.glob(".fixtures.publication-recovery.json"))) == 1


def test_in_place_staging_mutation_after_publication_enters_manual_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "fixtures"
    _write_valid_fixture_set(output, "1" * 40, "old-output")
    staging = tmp_path / "staging"
    _write_valid_fixture_set(staging, "2" * 40, "new-output")
    original_replace = Path.replace

    def _mutate_after_staging_rename(path: Path, target: Path) -> Path:
        result = original_replace(path, target)
        if path == staging:
            _write_canonical_json(
                output / "content.json",
                _fixture_payloads("tampered-sensitive-published")["content"],
            )
        return result

    monkeypatch.setattr(Path, "replace", _mutate_after_staging_rename)

    with pytest.raises(
        RuntimeError,
        match=("^Fixture output rollback could not be completed safely; " "manual recovery is required$"),
    ) as exc_info:
        generator._replace_output_directory(staging, output)

    assert json.loads((output / "content.json").read_text(encoding="ascii"))["cases"] == [
        {"marker": "tampered-sensitive-published"}
    ]
    backups = list(tmp_path.glob(".fixtures.backup-*"))
    assert len(backups) == 1
    _assert_fixture_marker(backups[0], "old-output")
    assert len(list(tmp_path.glob(".fixtures.publication-recovery.json"))) == 1
    assert str(tmp_path) not in str(exc_info.value)
    assert "tampered-sensitive-published" not in str(exc_info.value)


def test_output_substitution_after_validation_is_not_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "fixtures"
    _write_valid_fixture_set(output, "1" * 40, "old-output")
    staging = tmp_path / "staging"
    _write_valid_fixture_set(staging, "2" * 40, "new-output")
    validated_output = tmp_path / "validated-output"
    original_validate = generator._validate_existing_output

    def _substitute_after_validation(path: Path) -> object:
        identity = original_validate(path)
        path.replace(validated_output)
        _write_valid_fixture_set(path, "3" * 40, "substituted-output")
        return identity

    monkeypatch.setattr(generator, "_validate_existing_output", _substitute_after_validation)

    with pytest.raises(RuntimeError, match="^Fixture output changed during publication$"):
        generator._replace_output_directory(staging, output)

    _assert_fixture_marker(output, "substituted-output")
    _assert_fixture_marker(validated_output, "old-output")
    _assert_fixture_marker(staging, "new-output")
    assert not list(tmp_path.glob(".fixtures.backup-*"))


def test_output_parent_substitution_after_validation_is_not_modified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication_parent = tmp_path / "publication"
    publication_parent.mkdir()
    output = publication_parent / "fixtures"
    _write_valid_fixture_set(output, "1" * 40, "old-output")
    staging = tmp_path / "staging"
    _write_valid_fixture_set(staging, "2" * 40, "new-output")
    validated_parent = tmp_path / "validated-parent"
    original_validate = generator._validate_existing_output

    def _substitute_parent_after_validation(path: Path) -> object:
        identity = original_validate(path)
        publication_parent.replace(validated_parent)
        publication_parent.mkdir()
        _write_valid_fixture_set(output, "3" * 40, "substituted-output")
        return identity

    monkeypatch.setattr(generator, "_validate_existing_output", _substitute_parent_after_validation)

    with pytest.raises(
        RuntimeError,
        match="^Fixture output parent changed during publication$",
    ):
        generator._replace_output_directory(staging, output)

    _assert_fixture_marker(output, "substituted-output")
    _assert_fixture_marker(validated_parent / "fixtures", "old-output")
    _assert_fixture_marker(staging, "new-output")
    assert not list(publication_parent.glob(".fixtures.backup-*"))


def test_backup_substitution_after_output_rename_is_retained(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "fixtures"
    _write_valid_fixture_set(output, "1" * 40, "old-output")
    staging = tmp_path / "staging"
    _write_valid_fixture_set(staging, "2" * 40, "new-output")
    validated_backup = tmp_path / "validated-backup"
    original_replace = Path.replace
    backup_path: Path | None = None

    def _substitute_backup(path: Path, target: Path) -> Path:
        nonlocal backup_path
        result = original_replace(path, target)
        if path == output:
            backup_path = target
            original_replace(target, validated_backup)
            _write_valid_fixture_set(target, "3" * 40, "substituted-backup")
        return result

    monkeypatch.setattr(Path, "replace", _substitute_backup)

    with pytest.raises(
        RuntimeError,
        match=("^Fixture output rollback could not be completed safely; " "manual recovery is required$"),
    ):
        generator._replace_output_directory(staging, output)

    assert backup_path is not None
    assert not output.exists()
    _assert_fixture_marker(backup_path, "substituted-backup")
    _assert_fixture_marker(validated_backup, "old-output")
    _assert_fixture_marker(staging, "new-output")


def test_backup_substitution_before_rollback_is_not_restored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "fixtures"
    _write_valid_fixture_set(output, "1" * 40, "old-output")
    staging = tmp_path / "staging"
    _write_valid_fixture_set(staging, "2" * 40, "new-output")
    validated_backup = tmp_path / "validated-backup"
    original_replace = Path.replace
    backup_path: Path | None = None

    def _fail_with_substituted_backup(path: Path, target: Path) -> Path:
        nonlocal backup_path
        if path == output:
            backup_path = target
            return original_replace(path, target)
        if path == staging:
            assert backup_path is not None
            original_replace(backup_path, validated_backup)
            _write_valid_fixture_set(backup_path, "3" * 40, "substituted-backup")
            raise OSError("sensitive staging failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", _fail_with_substituted_backup)

    with pytest.raises(
        RuntimeError,
        match=("^Fixture output rollback could not be completed safely; " "manual recovery is required$"),
    ) as exc_info:
        generator._replace_output_directory(staging, output)

    assert backup_path is not None
    assert not output.exists()
    _assert_fixture_marker(backup_path, "substituted-backup")
    _assert_fixture_marker(validated_backup, "old-output")
    _assert_fixture_marker(staging, "new-output")
    assert "sensitive" not in str(exc_info.value)
    assert str(tmp_path) not in str(exc_info.value)


def test_backup_substitution_before_cleanup_is_retained_with_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "fixtures"
    _write_valid_fixture_set(output, "1" * 40, "old-output")
    staging = tmp_path / "staging"
    _write_valid_fixture_set(staging, "2" * 40, "new-output")
    validated_backup = tmp_path / "validated-backup"
    original_replace = Path.replace
    backup_path: Path | None = None

    def _commit_with_substituted_backup(path: Path, target: Path) -> Path:
        nonlocal backup_path
        result = original_replace(path, target)
        if path == output:
            backup_path = target
        elif path == staging:
            assert backup_path is not None
            original_replace(backup_path, validated_backup)
            _write_valid_fixture_set(backup_path, "3" * 40, "substituted-backup")
        return result

    monkeypatch.setattr(Path, "replace", _commit_with_substituted_backup)

    generator._replace_output_directory(staging, output)

    assert backup_path is not None
    _assert_fixture_marker(output, "new-output")
    _assert_fixture_marker(backup_path, "substituted-backup")
    _assert_fixture_marker(validated_backup, "old-output")
    diagnostic = capsys.readouterr().err
    assert "fixture output committed; backup cleanup failed" in diagnostic.lower()
    assert backup_path.name in diagnostic
    assert str(tmp_path) not in diagnostic


def test_backup_identity_warning_write_failure_is_nonfatal_after_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DirectStderrFailure(BaseException):
        pass

    class _FailingStderr:
        def __init__(self) -> None:
            self.write_calls = 0

        def write(self, _text: str) -> int:
            self.write_calls += 1
            raise DirectStderrFailure("sensitive stderr failure")

    output = tmp_path / "fixtures"
    _write_valid_fixture_set(output, "1" * 40, "old-output")
    staging = tmp_path / "staging"
    _write_valid_fixture_set(staging, "2" * 40, "new-output")
    validated_backup = tmp_path / "validated-backup"
    original_replace = Path.replace
    backup_path: Path | None = None
    failing_stderr = _FailingStderr()

    def _commit_with_substituted_backup(path: Path, target: Path) -> Path:
        nonlocal backup_path
        result = original_replace(path, target)
        if path == output:
            backup_path = target
        elif path == staging:
            assert backup_path is not None
            original_replace(backup_path, validated_backup)
            _write_valid_fixture_set(backup_path, "3" * 40, "substituted-backup")
        return result

    monkeypatch.setattr(Path, "replace", _commit_with_substituted_backup)
    monkeypatch.setattr(generator.sys, "stderr", failing_stderr)

    generator._replace_output_directory(staging, output)

    assert failing_stderr.write_calls == 1
    assert backup_path is not None
    _assert_fixture_marker(output, "new-output")
    _assert_fixture_marker(backup_path, "substituted-backup")
    _assert_fixture_marker(validated_backup, "old-output")


def test_backup_cleanup_warning_write_failure_is_nonfatal_after_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DirectStderrFailure(BaseException):
        pass

    class _FailingStderr:
        def __init__(self) -> None:
            self.write_calls = 0

        def write(self, _text: str) -> int:
            self.write_calls += 1
            raise DirectStderrFailure("sensitive stderr failure")

    output = tmp_path / "fixtures"
    _write_valid_fixture_set(output, "1" * 40, "old-output")
    staging = tmp_path / "staging"
    _write_valid_fixture_set(staging, "2" * 40, "new-output")
    failing_stderr = _FailingStderr()

    def _fail_backup_cleanup(_path: Path) -> None:
        raise OSError("sensitive backup cleanup failure")

    monkeypatch.setattr(generator.shutil, "rmtree", _fail_backup_cleanup)
    monkeypatch.setattr(generator.sys, "stderr", failing_stderr)

    generator._replace_output_directory(staging, output)

    assert failing_stderr.write_calls == 1
    _assert_fixture_marker(output, "new-output")
    backups = list(tmp_path.glob(".fixtures.backup-*"))
    assert len(backups) == 1
    _assert_fixture_marker(backups[0], "old-output")


def test_backup_cleanup_failure_keeps_committed_output_and_reports_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "fixtures"
    _write_valid_fixture_set(output, "1" * 40, "old-sensitive-output")
    staging = tmp_path / "staging"
    _write_valid_fixture_set(staging, "2" * 40, "new-sensitive-output")

    def _fail_backup_cleanup(_path: Path) -> None:
        raise OSError("injected cleanup failure")

    monkeypatch.setattr(generator.shutil, "rmtree", _fail_backup_cleanup)

    generator._replace_output_directory(staging, output)

    assert json.loads((output / "content.json").read_text(encoding="ascii"))["cases"] == [
        {"marker": "new-sensitive-output"}
    ]
    backups = list(tmp_path.glob(".fixtures.backup-*"))
    assert len(backups) == 1
    assert json.loads((backups[0] / "content.json").read_text(encoding="ascii"))["cases"] == [
        {"marker": "old-sensitive-output"}
    ]
    diagnostic = capsys.readouterr().err
    assert "fixture output committed; backup cleanup failed" in diagnostic.lower()
    assert backups[0].name in diagnostic
    assert str(tmp_path) not in diagnostic
    assert "old-sensitive-output" not in diagnostic
    assert "new-sensitive-output" not in diagnostic


def test_backup_cleanup_failure_is_nonfatal_when_runtime_warnings_are_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "fixtures"
    _write_valid_fixture_set(output, "1" * 40, "old-sensitive-output")
    staging = tmp_path / "staging"
    _write_valid_fixture_set(staging, "2" * 40, "new-sensitive-output")

    def _fail_backup_cleanup(_path: Path) -> None:
        raise OSError("injected cleanup failure")

    monkeypatch.setattr(generator.shutil, "rmtree", _fail_backup_cleanup)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        generator._replace_output_directory(staging, output)

    assert json.loads((output / "content.json").read_text(encoding="ascii"))["cases"] == [
        {"marker": "new-sensitive-output"}
    ]
    backups = list(tmp_path.glob(".fixtures.backup-*"))
    assert len(backups) == 1
    assert json.loads((backups[0] / "content.json").read_text(encoding="ascii"))["cases"] == [
        {"marker": "old-sensitive-output"}
    ]
    diagnostic = capsys.readouterr().err
    assert "fixture output committed; backup cleanup failed" in diagnostic.lower()
    assert backups[0].name in diagnostic
    assert str(tmp_path) not in diagnostic
    assert "old-sensitive-output" not in diagnostic
    assert "new-sensitive-output" not in diagnostic
