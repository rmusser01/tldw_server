from __future__ import annotations

import hashlib
import os
import shutil
import subprocess  # nosec B404 - fixed local Bash command executes a repository script
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
REFRESH_SCRIPT = REPO_ROOT / "Helper_Scripts" / "refresh_docs_published.sh"


def _tree_manifest(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _refresh(
    *,
    source: Path | None = None,
    destination: Path | None = None,
    fail_after_backup: bool = False,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if source is not None:
        env["TLDW_DOCS_SOURCE_DIR"] = str(source)
    if destination is not None:
        env["TLDW_DOCS_PUBLISHED_DIR"] = str(destination)
    if fail_after_backup:
        env["TLDW_DOCS_TEST_FAIL_AFTER_BACKUP"] = "1"
    return subprocess.run(  # nosec B603
        ["/bin/bash", str(REFRESH_SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_refresh_script_has_safe_destination_override_and_no_silent_copy_failures() -> None:
    script = REFRESH_SCRIPT.read_text(encoding="utf-8")

    assert "TLDW_DOCS_PUBLISHED_DIR" in script
    assert "2>/dev/null || true" not in script


def test_refresh_replaces_clean_and_stale_destinations_deterministically(
    tmp_path: Path,
) -> None:
    clean = tmp_path / "clean"
    stale = tmp_path / "stale"
    stale.mkdir()
    (stale / "unknown.md").write_text("stale\n", encoding="utf-8")
    (stale / "index.md").write_text("old landing page\n", encoding="utf-8")

    clean_result = _refresh(destination=clean)
    stale_result = _refresh(destination=stale)

    assert clean_result.returncode == 0, clean_result.stderr
    assert stale_result.returncode == 0, stale_result.stderr
    assert _tree_manifest(stale) == _tree_manifest(clean)
    assert _tree_manifest(clean) == _tree_manifest(REPO_ROOT / "Docs" / "Published")


def test_refresh_removes_unknown_stale_file(tmp_path: Path) -> None:
    destination = tmp_path / "published"
    destination.mkdir()
    stale_file = destination / "unknown.md"
    stale_file.write_text("stale\n", encoding="utf-8")

    result = _refresh(destination=destination)

    assert result.returncode == 0, result.stderr
    assert not stale_file.exists()


def test_refresh_preserves_getting_started_readme_without_sibling_index(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "published"

    result = _refresh(destination=destination)

    assert result.returncode == 0, result.stderr
    assert (destination / "Getting_Started" / "README.md").is_file()
    assert not (destination / "Getting_Started" / "index.md").exists()


def test_refresh_omits_code_documentation_readme_with_sibling_index(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "published"

    result = _refresh(destination=destination)

    assert result.returncode == 0, result.stderr
    assert (destination / "Code_Documentation" / "index.md").is_file()
    assert not (destination / "Code_Documentation" / "README.md").exists()


def test_refresh_recursively_omits_readme_with_sibling_index(tmp_path: Path) -> None:
    source = tmp_path / "Docs"
    shutil.copytree(
        REPO_ROOT / "Docs",
        source,
        ignore=shutil.ignore_patterns("Published", "site"),
    )
    collision = source / "Code_Documentation" / "Synthetic_Collision"
    collision.mkdir()
    (collision / "index.md").write_text("# Landing\n", encoding="utf-8")
    (collision / "README.md").write_text("# Duplicate\n", encoding="utf-8")
    destination = tmp_path / "published"

    result = _refresh(source=source, destination=destination)

    assert result.returncode == 0, result.stderr
    assert (destination / "Code_Documentation" / "Synthetic_Collision" / "index.md").is_file()
    assert not (destination / "Code_Documentation" / "Synthetic_Collision" / "README.md").exists()


@pytest.mark.parametrize(
    "relative_path",
    (
        "index.md",
        "RELEASE_NOTES.md",
        "Architecture.md",
        "Env_Vars.md",
        "Overview/Feature_Status.md",
        "assets/logo.png",
        "assets/favicon.png",
    ),
)
def test_refresh_publishes_every_special_file(
    tmp_path: Path,
    relative_path: str,
) -> None:
    destination = tmp_path / "published"

    result = _refresh(destination=destination)

    assert result.returncode == 0, result.stderr
    assert (destination / relative_path).is_file()


def test_refresh_restores_destination_after_failure_post_backup(tmp_path: Path) -> None:
    destination = tmp_path / "published"
    destination.mkdir()
    sentinel = destination / "sentinel.md"
    sentinel.write_text("keep me\n", encoding="utf-8")

    result = _refresh(destination=destination, fail_after_backup=True)

    assert result.returncode != 0
    assert sentinel.read_text(encoding="utf-8") == "keep me\n"
    assert not destination.with_name(f"{destination.name}.stage").exists()
    assert not destination.with_name(f"{destination.name}.backup").exists()
