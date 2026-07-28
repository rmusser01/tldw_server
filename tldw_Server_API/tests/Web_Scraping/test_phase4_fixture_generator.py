from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import queue
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Any

import pytest
from Helper_Scripts import web_scraping_phase4_fixtures as generator


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
) -> None:
    def _build_payloads(_source_root: Path) -> dict[str, dict[str, Any]]:
        entered_builder.set()
        if not release_builder.wait(15):
            raise TimeoutError("test worker payload release timed out")
        if failure_mode == "exception":
            raise RuntimeError("injected payload failure")
        return _fixture_payloads(marker)

    generator.build_case_payloads = _build_payloads
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
        output.symlink_to(target, target_is_directory=True)
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
    output_link.symlink_to(output, target_is_directory=True)
    expected_identity = hashlib.sha256(os.path.normcase(str(output.resolve())).encode("utf-8")).hexdigest()

    direct_lock = generator._lock_path_for_output(output)
    linked_lock = generator._lock_path_for_output(output_link)

    assert direct_lock == linked_lock
    assert direct_lock.name == f"{expected_identity}.lock"
    assert direct_lock.parent == Path(tempfile.gettempdir()).resolve() / "tldw-phase4-fixture-locks"
    assert not direct_lock.is_relative_to(source_root)


def test_same_output_processes_serialize_without_crossed_fixture_sets(tmp_path: Path) -> None:
    source_root, source_commit = _create_clean_source_root(tmp_path)
    output = tmp_path / "fixtures"
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
    )

    first.start()
    try:
        assert first_started.wait(10)
        assert first_entered.wait(10)
        second.start()
        assert second_started.wait(10)
        second_was_blocked = not second_entered.wait(2)
    finally:
        first_release.set()
        second_release.set()
        _join_process(first)
        if second.pid is not None:
            _join_process(second)

    assert second_was_blocked
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
