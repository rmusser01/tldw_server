from __future__ import annotations

import hashlib
import os
import shutil

# Fixed local commands execute the repository refresh script and Git queries.
import subprocess  # nosec B404
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
REFRESH_SCRIPT = REPO_ROOT / "Helper_Scripts" / "refresh_docs_published.sh"
REVIEWED_PUBLISHED_JSON = (
    "Docs/Published/Deployment/sidecar_workers_manifest.json",
    "Docs/Published/Evaluations/samples/dataset_quick.json",
    "Docs/Published/Evaluations/samples/rag_pipeline_eval_inline.json",
    "Docs/Published/Evaluations/samples/run_request.json",
    "Docs/Published/Evaluations/baselines/web_retrieval_quality_v1.json",
    "Docs/Published/Monitoring/Grafana_Streaming_Basics.json",
)


def _tree_manifest(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _copy_docs_source(destination: Path) -> Path:
    shutil.copytree(
        REPO_ROOT / "Docs",
        destination,
        ignore=shutil.ignore_patterns("Published", "site"),
    )
    return destination


def _tracked_published_files() -> set[str]:
    result = subprocess.run(  # nosec B603 B607
        ["git", "ls-files", "Docs/Published"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    prefix = "Docs/Published/"
    return {path.removeprefix(prefix) for path in result.stdout.splitlines() if path.startswith(prefix)}


def _clean_env() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if not key.startswith("TLDW_DOCS_")}


def _run_refresh_script(
    script: Path,
    env_updates: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = _clean_env()
    env.update(env_updates or {})
    return subprocess.run(  # nosec B603
        ["/bin/bash", str(script)],
        cwd=script.parent.parent,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _refresh(
    *,
    source: Path | None = None,
    destination: Path | None = None,
    fail_after_backup: bool = False,
    fail_during_backup_cleanup: bool = False,
) -> subprocess.CompletedProcess[str]:
    env = _clean_env()
    env["TLDW_DOCS_TEST_MODE"] = "1"
    if source is not None:
        env["TLDW_DOCS_SOURCE_DIR"] = str(source)
    if destination is not None:
        env["TLDW_DOCS_PUBLISHED_DIR"] = str(destination)
    if fail_after_backup:
        env["TLDW_DOCS_TEST_FAIL_AFTER_BACKUP"] = "1"
    if fail_during_backup_cleanup:
        env["TLDW_DOCS_TEST_FAIL_DURING_BACKUP_CLEANUP"] = "1"
    return _run_refresh_script(REFRESH_SCRIPT, env)


def _isolated_refresh_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    source = _copy_docs_source(repo / "Docs")
    script = repo / "Helper_Scripts" / "refresh_docs_published.sh"
    script.parent.mkdir()
    shutil.copy2(REFRESH_SCRIPT, script)
    return script, source


def _git_check_ignore(path: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # nosec B603 B607
        ["git", "check-ignore", "--no-index", "--quiet", path],
        cwd=REPO_ROOT,
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


def test_refresh_output_matches_tracked_published_files(tmp_path: Path) -> None:
    destination = tmp_path / "published"

    result = _refresh(destination=destination)

    assert result.returncode == 0, result.stderr
    assert set(_tree_manifest(destination)) == _tracked_published_files()


def test_only_reviewed_published_json_files_are_unignored_and_tracked() -> None:
    tracked = {f"Docs/Published/{path}" for path in _tracked_published_files()}

    for path in REVIEWED_PUBLISHED_JSON:
        assert _git_check_ignore(path).returncode == 1
        assert path in tracked

    assert _git_check_ignore("Docs/Published/API-related/private_credentials.json").returncode == 0
    assert _git_check_ignore("unrelated.json").returncode == 0
    assert _git_check_ignore("Docs/_site/private_credentials.json").returncode == 0


@pytest.mark.parametrize(
    "seam_name",
    (
        "TLDW_DOCS_SOURCE_DIR",
        "TLDW_DOCS_PUBLISHED_DIR",
        "TLDW_DOCS_TEST_FAIL_AFTER_BACKUP",
        "TLDW_DOCS_TEST_FAIL_DURING_BACKUP_CLEANUP",
    ),
)
def test_refresh_rejects_ungated_test_seams_before_mutation(
    tmp_path: Path,
    seam_name: str,
) -> None:
    script, source = _isolated_refresh_repo(tmp_path)
    source_sentinel = source / "sentinel.md"
    source_sentinel.write_text("keep source\n", encoding="utf-8")
    override_destination = tmp_path / "override-published"
    values = {
        "TLDW_DOCS_SOURCE_DIR": str(source),
        "TLDW_DOCS_PUBLISHED_DIR": str(override_destination),
        "TLDW_DOCS_TEST_FAIL_AFTER_BACKUP": "1",
        "TLDW_DOCS_TEST_FAIL_DURING_BACKUP_CLEANUP": "1",
    }

    result = _run_refresh_script(script, {seam_name: values[seam_name]})

    assert result.returncode != 0
    assert "TLDW_DOCS_TEST_MODE=1" in result.stderr
    assert source_sentinel.read_text(encoding="utf-8") == "keep source\n"
    assert not (source / "Published").exists()
    assert not override_destination.exists()


def test_refresh_rejects_destination_equal_to_source_before_mutation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "Docs"
    source.mkdir()
    sentinel = source / "sentinel.md"
    sentinel.write_text("keep me\n", encoding="utf-8")

    result = _refresh(source=source, destination=source)

    assert result.returncode != 0
    assert sentinel.read_text(encoding="utf-8") == "keep me\n"
    assert not source.with_name(f"{source.name}.stage").exists()
    assert not source.with_name(f"{source.name}.backup").exists()


def test_refresh_rejects_destination_ancestor_of_source_before_mutation(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "public-root"
    source = _copy_docs_source(destination / "Docs")
    sentinel = destination / "sentinel.md"
    sentinel.write_text("keep me\n", encoding="utf-8")
    source_sentinel = source / "source-sentinel.md"
    source_sentinel.write_text("keep source\n", encoding="utf-8")

    result = _refresh(source=source, destination=destination)

    assert result.returncode != 0
    assert sentinel.read_text(encoding="utf-8") == "keep me\n"
    assert source_sentinel.read_text(encoding="utf-8") == "keep source\n"
    assert not destination.with_name(f"{destination.name}.stage").exists()
    assert not destination.with_name(f"{destination.name}.backup").exists()


def test_refresh_rejects_canonical_symlink_equivalent_paths_before_mutation(
    tmp_path: Path,
) -> None:
    destination = _copy_docs_source(tmp_path / "Docs")
    source = tmp_path / "docs-link"
    source.symlink_to(destination, target_is_directory=True)
    sentinel = destination / "sentinel.md"
    sentinel.write_text("keep me\n", encoding="utf-8")

    result = _refresh(source=source, destination=destination)

    assert result.returncode != 0
    assert source.is_symlink()
    assert sentinel.read_text(encoding="utf-8") == "keep me\n"
    assert not destination.with_name(f"{destination.name}.stage").exists()
    assert not destination.with_name(f"{destination.name}.backup").exists()


def test_refresh_removes_unknown_stale_file(tmp_path: Path) -> None:
    destination = tmp_path / "published"
    destination.mkdir()
    stale_file = destination / "unknown.md"
    stale_file.write_text("stale\n", encoding="utf-8")

    result = _refresh(destination=destination)

    assert result.returncode == 0, result.stderr
    assert not stale_file.exists()
    assert not destination.with_name(f"{destination.name}.backup").exists()


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
    source = _copy_docs_source(tmp_path / "Docs")
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
    assert not destination.with_name(f"{destination.name}.backup").exists()
    assert not destination.with_name(f"{destination.name}.lock").exists()
    assert not list(tmp_path.glob(f"{destination.name}.stage.*"))


def test_refresh_preserves_committed_destination_when_backup_cleanup_fails(
    tmp_path: Path,
) -> None:
    source = _copy_docs_source(tmp_path / "source" / "Docs")
    source_sentinel = source / "source-sentinel.md"
    source_sentinel.write_text("keep source\n", encoding="utf-8")
    destination = tmp_path / "published"
    destination.mkdir()
    old_sentinel = destination / "old-sentinel.md"
    old_sentinel.write_text("keep old\n", encoding="utf-8")
    backup = destination.with_name(f"{destination.name}.backup")

    result = _refresh(
        source=source,
        destination=destination,
        fail_during_backup_cleanup=True,
    )

    assert result.returncode != 0
    assert _tree_manifest(destination) == _tree_manifest(REPO_ROOT / "Docs" / "Published")
    assert not old_sentinel.exists()
    assert (backup / "old-sentinel.md").read_text(encoding="utf-8") == "keep old\n"
    assert source_sentinel.read_text(encoding="utf-8") == "keep source\n"
    assert not destination.with_name(f"{destination.name}.lock").exists()
    assert not list(tmp_path.glob(f"{destination.name}.stage.*"))


def test_refresh_helper_clears_inherited_backup_cleanup_failure_seam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TLDW_DOCS_TEST_FAIL_DURING_BACKUP_CLEANUP", "1")

    assert "TLDW_DOCS_TEST_FAIL_DURING_BACKUP_CLEANUP" not in _clean_env()


def test_refresh_fails_closed_when_lock_already_exists(tmp_path: Path) -> None:
    destination = tmp_path / "published"
    destination.mkdir()
    sentinel = destination / "sentinel.md"
    sentinel.write_text("keep me\n", encoding="utf-8")
    lock = destination.with_name(f"{destination.name}.lock")
    lock.mkdir()
    lock_sentinel = lock / "owner"
    lock_sentinel.write_text("other run\n", encoding="utf-8")

    result = _refresh(destination=destination)

    assert result.returncode != 0
    assert sentinel.read_text(encoding="utf-8") == "keep me\n"
    assert lock_sentinel.read_text(encoding="utf-8") == "other run\n"
    assert not destination.with_name(f"{destination.name}.backup").exists()
    assert not list(tmp_path.glob(f"{destination.name}.stage.*"))


def test_refresh_restores_backup_only_state_and_requires_rerun(tmp_path: Path) -> None:
    destination = tmp_path / "published"
    backup = destination.with_name(f"{destination.name}.backup")
    backup.mkdir()
    backup_sentinel = backup / "sentinel.md"
    backup_sentinel.write_text("recover me\n", encoding="utf-8")

    result = _refresh(destination=destination)

    assert result.returncode != 0
    assert (destination / "sentinel.md").read_text(encoding="utf-8") == "recover me\n"
    assert not backup.exists()
    assert not destination.with_name(f"{destination.name}.lock").exists()
    assert not list(tmp_path.glob(f"{destination.name}.stage.*"))


def test_refresh_preserves_ambiguous_destination_and_backup(tmp_path: Path) -> None:
    destination = tmp_path / "published"
    destination.mkdir()
    destination_sentinel = destination / "destination.md"
    destination_sentinel.write_text("current\n", encoding="utf-8")
    backup = destination.with_name(f"{destination.name}.backup")
    backup.mkdir()
    backup_sentinel = backup / "backup.md"
    backup_sentinel.write_text("previous\n", encoding="utf-8")

    result = _refresh(destination=destination)

    assert result.returncode != 0
    assert destination_sentinel.read_text(encoding="utf-8") == "current\n"
    assert backup_sentinel.read_text(encoding="utf-8") == "previous\n"
    assert not destination.with_name(f"{destination.name}.lock").exists()
    assert not list(tmp_path.glob(f"{destination.name}.stage.*"))


def test_refresh_does_not_delete_foreign_stage(tmp_path: Path) -> None:
    destination = tmp_path / "published"
    foreign_stage = destination.with_name(f"{destination.name}.stage")
    foreign_stage.mkdir()
    foreign_sentinel = foreign_stage / "owner"
    foreign_sentinel.write_text("other run\n", encoding="utf-8")

    result = _refresh(destination=destination)

    assert result.returncode == 0, result.stderr
    assert foreign_sentinel.read_text(encoding="utf-8") == "other run\n"
    assert not destination.with_name(f"{destination.name}.lock").exists()
    assert not destination.with_name(f"{destination.name}.backup").exists()


def test_refresh_rejects_destination_symlink_and_preserves_target(tmp_path: Path) -> None:
    source = _copy_docs_source(tmp_path / "source" / "Docs")
    source_sentinel = source / "sentinel.md"
    source_sentinel.write_text("keep source\n", encoding="utf-8")
    target = tmp_path / "target"
    target.mkdir()
    target_sentinel = target / "sentinel.md"
    target_sentinel.write_text("keep target\n", encoding="utf-8")
    destination = tmp_path / "published"
    destination.symlink_to(target, target_is_directory=True)

    result = _refresh(source=source, destination=destination)

    assert result.returncode != 0
    assert "Docs destination must be a real directory path" in result.stderr
    assert destination.is_symlink()
    assert source_sentinel.read_text(encoding="utf-8") == "keep source\n"
    assert target_sentinel.read_text(encoding="utf-8") == "keep target\n"


def test_refresh_rejects_broken_destination_symlink(tmp_path: Path) -> None:
    source = _copy_docs_source(tmp_path / "source" / "Docs")
    source_sentinel = source / "sentinel.md"
    source_sentinel.write_text("keep source\n", encoding="utf-8")
    target = tmp_path / "missing-target"
    destination = tmp_path / "published"
    destination.symlink_to(target, target_is_directory=True)

    result = _refresh(source=source, destination=destination)

    assert result.returncode != 0
    assert "Docs destination must be a real directory path" in result.stderr
    assert destination.is_symlink()
    assert os.readlink(destination) == str(target)
    assert not target.exists()
    assert source_sentinel.read_text(encoding="utf-8") == "keep source\n"


def test_refresh_rejects_existing_destination_file(tmp_path: Path) -> None:
    source = _copy_docs_source(tmp_path / "source" / "Docs")
    source_sentinel = source / "sentinel.md"
    source_sentinel.write_text("keep source\n", encoding="utf-8")
    destination = tmp_path / "published"
    destination.write_text("keep destination\n", encoding="utf-8")

    result = _refresh(source=source, destination=destination)

    assert result.returncode != 0
    assert "Docs destination must be a real directory path" in result.stderr
    assert destination.read_text(encoding="utf-8") == "keep destination\n"
    assert source_sentinel.read_text(encoding="utf-8") == "keep source\n"


def test_refresh_prefers_evaluations_when_both_sources_exist(tmp_path: Path) -> None:
    source = _copy_docs_source(tmp_path / "Docs")
    preferred = source / "Evaluations"
    preferred.mkdir()
    (preferred / "preferred.md").write_text("preferred\n", encoding="utf-8")
    (source / "Evals" / "fallback-only.md").write_text(
        "fallback\n",
        encoding="utf-8",
    )
    destination = tmp_path / "published"

    result = _refresh(source=source, destination=destination)

    assert result.returncode == 0, result.stderr
    assert (destination / "Evaluations" / "preferred.md").is_file()
    assert not (destination / "Evaluations" / "fallback-only.md").exists()
