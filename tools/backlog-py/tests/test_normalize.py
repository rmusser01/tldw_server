"""Normalizer tests on real task files copied from this repository's backlog.

The mixed fixture repo holds verbatim copies of task files that the Node Backlog.md
CLI and backlog-py corrupted between them: Node ``SECTION:NOTES`` blocks, NOTES
wrapping IMPLEMENTATION_NOTES, repeated top-level notes blocks, duplicated and
orphaned FINAL_SUMMARY markers, and an orphaned DESCRIPTION END.
"""

from __future__ import annotations

import re
import shutil
from collections import Counter
from pathlib import Path

import pytest
from backlog_py.cli.main import main
from backlog_py.core.repository import MutableRepository, TaskMutationError
from backlog_py.markdown.task_parser import normalize_task_markdown, parse_task_markdown
from click.testing import CliRunner

FIXTURES = Path(__file__).parent / "fixtures" / "repos"
MIXED_REPO = FIXTURES / "mixed"
MIXED_TASKS = sorted((MIXED_REPO / "backlog" / "tasks").glob("*.md"))
CANONICAL_TASK = FIXTURES / "basic" / "backlog" / "tasks" / "task-1 - Example-task.md"

_SECTION_MARKER = re.compile(r"^<!-- SECTION:(?P<name>[A-Z0-9_ -]+):(?P<edge>BEGIN|END) -->\s*$")
# Headings of a repeated notes block disappear when the block merges into the first one.
_MERGED_HEADINGS = {"## Implementation Notes", "## Notes"}


def _read(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return handle.read()


def _text_lines(source: str) -> Counter[str]:
    return Counter(
        line.rstrip()
        for line in source.splitlines()
        if line.strip() and _SECTION_MARKER.match(line) is None
    )


def _copy_mixed(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    shutil.copytree(MIXED_REPO, repo)
    return repo


def _invoke(repo: Path, *args: str):
    return CliRunner().invoke(main, ["--cwd", str(repo), *args])


def test_every_mixed_fixture_needs_normalization():
    assert len(MIXED_TASKS) == 7
    for path in MIXED_TASKS:
        source = _read(path)
        assert normalize_task_markdown(source) != source, path.name


@pytest.mark.parametrize("path", MIXED_TASKS, ids=lambda path: path.name.split(" - ")[0])
def test_normalized_task_has_one_canonical_block_per_section(path: Path):
    normalized = normalize_task_markdown(_read(path))

    assert "SECTION:NOTES:" not in normalized
    markers = Counter(
        (match["name"], match["edge"])
        for match in map(_SECTION_MARKER.match, normalized.splitlines())
        if match is not None
    )
    for (name, _edge), count in markers.items():
        assert count == 1, f"{name} has {count} markers"
        assert markers[(name, "BEGIN")] == markers[(name, "END")] == 1
    assert ("IMPLEMENTATION_NOTES", "BEGIN") in markers
    assert normalized.count("## Implementation Notes") <= 1
    assert "\n\n\n" not in normalized


@pytest.mark.parametrize("path", MIXED_TASKS, ids=lambda path: path.name.split(" - ")[0])
def test_normalize_is_lossless_and_keeps_frontmatter_bytes(path: Path):
    source = _read(path)
    normalized = normalize_task_markdown(source)

    raw_frontmatter = parse_task_markdown(source).raw_frontmatter
    assert raw_frontmatter is not None
    assert normalized.startswith(raw_frontmatter)
    before, after = _text_lines(source), _text_lines(normalized)
    for heading in _MERGED_HEADINGS:
        assert after[heading] <= before[heading]
        before.pop(heading, None)
        after.pop(heading, None)
    assert after == before


@pytest.mark.parametrize("path", MIXED_TASKS, ids=lambda path: path.name.split(" - ")[0])
def test_normalize_is_idempotent(path: Path):
    once = normalize_task_markdown(_read(path))

    assert normalize_task_markdown(once) == once


def test_canonical_task_is_returned_unchanged():
    source = _read(CANONICAL_TASK)

    assert normalize_task_markdown(source) == source


def test_nested_notes_keep_inner_text_before_outer_text():
    path = next(path for path in MIXED_TASKS if path.name.startswith("task-12849 "))
    notes = parse_task_markdown(normalize_task_markdown(_read(path))).sections["IMPLEMENTATION_NOTES"].content

    assert notes.startswith("- TDD RED captured before production implementation:")
    assert notes.index("- Follow-up verification recorded") < notes.index("Quality review follow-up:")


def test_repeated_notes_block_merges_into_the_first_in_document_order():
    path = next(path for path in MIXED_TASKS if path.name.startswith("task-10003 "))
    normalized = normalize_task_markdown(_read(path))
    notes = parse_task_markdown(normalized).sections["IMPLEMENTATION_NOTES"].content

    assert notes.index("Touched files:") < notes.index("PR #2459 rebase/review follow-up:")
    assert normalized.index("SECTION:IMPLEMENTATION_NOTES:END") < normalized.index("## Final Summary")


def test_duplicated_final_summary_markers_collapse_and_keep_text():
    path = next(path for path in MIXED_TASKS if path.name.startswith("task-12049 "))
    source = _read(path)
    normalized = normalize_task_markdown(source)

    summary = parse_task_markdown(normalized).sections["FINAL_SUMMARY"].content
    assert summary == parse_task_markdown(source).sections["FINAL_SUMMARY"].content
    assert summary.startswith("Fixed the current PR #1982 Notes UI remediation failure")
    assert normalized.count("<!-- SECTION:FINAL_SUMMARY:END -->") == 1


def test_crossed_section_markers_are_rejected():
    source = (
        "---\nid: TASK-9\n---\n\n"
        "<!-- SECTION:NOTES:BEGIN -->\n"
        "<!-- SECTION:PLAN:BEGIN -->\n"
        "text\n"
        "<!-- SECTION:NOTES:END -->\n"
        "<!-- SECTION:PLAN:END -->\n"
    )

    with pytest.raises(ValueError, match="crossed"):
        normalize_task_markdown(source)


def test_cli_check_lists_files_without_writing(tmp_path):
    repo = _copy_mixed(tmp_path)
    before = {path.name: _read(path) for path in (repo / "backlog" / "tasks").glob("*.md")}

    result = _invoke(repo, "task", "normalize", "--check")

    assert result.exit_code == 1
    listed = result.output.splitlines()
    assert sorted(listed) == sorted(f"backlog/tasks/{name}" for name in before)
    assert {path.name: _read(path) for path in (repo / "backlog" / "tasks").glob("*.md")} == before


def test_cli_normalize_rewrites_then_check_is_clean(tmp_path):
    repo = _copy_mixed(tmp_path)

    rewrite = _invoke(repo, "task", "normalize")
    check = _invoke(repo, "task", "normalize", "--check")

    assert rewrite.exit_code == 0
    assert len(rewrite.output.splitlines()) == len(MIXED_TASKS)
    assert check.exit_code == 0
    assert check.output == ""
    for path in (repo / "backlog" / "tasks").glob("*.md"):
        assert "SECTION:NOTES:" not in _read(path)


def test_cli_normalize_accepts_explicit_paths(tmp_path):
    repo = _copy_mixed(tmp_path)
    target = next((repo / "backlog" / "tasks").glob("task-12049 *.md"))

    result = _invoke(repo, "task", "normalize", "--check", str(target))

    assert result.exit_code == 1
    assert result.output.splitlines() == [f"backlog/tasks/{target.name}"]


def test_normalize_rejects_paths_outside_the_backlog(tmp_path):
    repo = _copy_mixed(tmp_path)
    outside = tmp_path / "outside.md"
    outside.write_text(_read(MIXED_TASKS[0]), encoding="utf-8")

    with pytest.raises(TaskMutationError, match="outside allowed base"):
        MutableRepository.from_path(repo).normalize_tasks([outside])

    assert outside.read_text(encoding="utf-8") == _read(MIXED_TASKS[0])


def test_edit_normalizes_node_notes_before_appending(tmp_path):
    repo = _copy_mixed(tmp_path)

    MutableRepository.from_path(repo).edit_task("TASK-10003", append_notes="Appended by backlog-py.")

    written = _read(next((repo / "backlog" / "tasks").glob("task-10003 *.md")))
    assert "SECTION:NOTES:" not in written
    assert written.count("<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->") == 1
    notes = parse_task_markdown(written).sections["IMPLEMENTATION_NOTES"].content
    assert notes.index("Touched files:") < notes.index("PR #2459") < notes.index("Appended by backlog-py.")
