from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from Helper_Scripts import web_scraping_phase4_fixtures as generator


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
    output.mkdir()
    (output / "existing.json").write_text('{"state":"original"}\n', encoding="utf-8")
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    payloads = {category: {"category": category, "cases": [{"name": category}]} for category in generator.CASE_NAMES}
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
