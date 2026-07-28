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
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from Helper_Scripts import web_scraping_phase4_fixtures as generator


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


def _write_valid_fixture_set(output: Path, predecessor_commit: str, marker: str) -> None:
    output.mkdir()
    case_files = {category: f"{category}.json" for category in generator.CASE_NAMES}
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
    expected_identity = hashlib.sha256(os.path.normcase(str(output.resolve())).encode("utf-8")).hexdigest()

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


@pytest.mark.skipif(
    generator.fcntl is None or os.open not in os.supports_dir_fd,
    reason="descriptor-relative POSIX lock opening is unavailable",
)
@pytest.mark.parametrize("lock_open_fails", [False, True], ids=["close-only", "primary-open-error"])
def test_lock_root_descriptor_close_failure_preserves_precedence_and_releases_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lock_open_fails: bool,
) -> None:
    lock_root = tmp_path / "locks"
    lock_root.mkdir(mode=0o700)
    real_open = generator.os.open
    real_close = generator.os.close
    root_descriptor: int | None = None
    lock_descriptor: int | None = None

    def _controlled_open(
        path: os.PathLike[str] | str,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal root_descriptor, lock_descriptor
        if dir_fd is not None and lock_open_fails:
            raise OSError("sensitive lock open failure")
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if dir_fd is None:
            root_descriptor = descriptor
        else:
            lock_descriptor = descriptor
        return descriptor

    def _failing_root_close(descriptor: int) -> None:
        real_close(descriptor)
        if descriptor == root_descriptor:
            raise OSError("sensitive root close failure")

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
    with pytest.raises(OSError) as root_closed:
        os.fstat(root_descriptor)
    assert root_closed.value.errno == errno.EBADF
    if lock_descriptor is not None:
        with pytest.raises(OSError) as lock_closed:
            os.fstat(lock_descriptor)
        assert lock_closed.value.errno == errno.EBADF
    assert "sensitive" not in str(exc_info.value)


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


def test_staging_cleanup_failure_does_not_replace_primary_publication_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root, source_commit = _create_clean_source_root(tmp_path)
    output = tmp_path / "fixtures"
    _isolated_lock_path(tmp_path, monkeypatch, output)
    monkeypatch.setattr(
        generator,
        "build_case_payloads",
        lambda _source_root: _fixture_payloads("replacement"),
    )
    primary_error = RuntimeError("primary fixture publication failure")
    cleanup_attempts: list[Path] = []

    def _fail_publication(_staging: Path, _output: Path) -> None:
        raise primary_error

    def _fail_staging_cleanup(path: Path) -> None:
        cleanup_attempts.append(path)
        raise OSError(f"sensitive staging cleanup failure at {tmp_path}")

    monkeypatch.setattr(generator, "_replace_output_directory", _fail_publication)
    monkeypatch.setattr(generator.shutil, "rmtree", _fail_staging_cleanup)

    with pytest.raises(
        RuntimeError,
        match="^primary fixture publication failure$",
    ) as exc_info:
        generator.generate_fixtures(
            source_commit,
            output,
            source_root=source_root,
        )

    assert exc_info.value is primary_error
    assert len(cleanup_attempts) == 1
    assert cleanup_attempts[0].parent == output.parent
    assert cleanup_attempts[0].name.startswith(f".{output.name}.staging-")
    assert "sensitive" not in str(exc_info.value)
    assert str(tmp_path) not in str(exc_info.value)


def test_staging_cleanup_only_failure_is_sanitized_after_atomic_publication(
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

    def _publish_and_leave_staging(staging: Path, target: Path) -> None:
        replace_output_directory(staging, target)
        staging.mkdir()
        staging_paths.append(staging)

    def _fail_staging_cleanup(path: Path) -> None:
        cleanup_attempts.append(path)
        raise OSError(f"sensitive staging cleanup failure at {tmp_path}")

    monkeypatch.setattr(
        generator,
        "_replace_output_directory",
        _publish_and_leave_staging,
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
    _assert_fixture_marker(output, "published")


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
