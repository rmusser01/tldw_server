from __future__ import annotations

import json
import subprocess  # nosec B404
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import Helper_Scripts.release as release_module  # noqa: E402
from Helper_Scripts.release import (  # noqa: E402
    ShellReleaseRunner,
    bump_version,
    classify_release_state,
    extract_required_check_names,
    find_exact_duplicate_bullets,
    normalize_bullet_text,
    orchestrate_release,
    read_current_version,
    promote_changelog_unreleased_section,
    update_pyproject_version,
    update_mkdocs_version_metadata,
    update_readme_release_references,
    update_release_notes_entry_point,
    validate_release_branch,
)


def test_read_current_version_from_pyproject(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
[project]
name = "example"
version = "9.8.7"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    assert read_current_version(pyproject) == "9.8.7"


def test_bump_version_patch() -> None:
    assert bump_version("0.1.30", "patch") == "0.1.31"


def test_bump_version_minor() -> None:
    assert bump_version("0.1.30", "minor") == "0.2.0"


def test_update_pyproject_version_supports_single_quoted_toml() -> None:
    pyproject_text = "[project]\nname = \"example\"\nversion = '0.1.30'\n"

    updated_text = update_pyproject_version(pyproject_text, "0.1.31")

    assert updated_text == "[project]\nname = \"example\"\nversion = '0.1.31'\n"


def test_extract_required_check_names() -> None:
    checks = extract_required_check_names(
        REPO_ROOT / "Docs" / "Development" / "CI_REQUIRED_GATES.md"
    )
    assert checks == [
        "backend-required",
        "security-required",
        "coverage-required",
        "frontend-required",
        "e2e-required",
        "container-build-check",
    ]


def test_extract_required_check_names_ignores_nested_subsections(tmp_path: Path) -> None:
    doc = tmp_path / "CI_REQUIRED_GATES.md"
    doc.write_text(
        """
# CI Required Gates

## Required Check Names

1. `backend-required`
2. `security-required`

### Nested Details

1. `ignored-required`
""".strip()
        + "\n",
        encoding="utf-8",
    )

    assert extract_required_check_names(doc) == [
        "backend-required",
        "security-required",
    ]


@pytest.mark.parametrize(
    "doc_text",
    [
        (
            """
# CI Required Gates

## Something Else

1. `backend-required`
""".strip()
            + "\n"
        ),
        (
            """
# CI Required Gates

## Required Check Names

### Nested Details

No enumerated checks here.
""".strip()
            + "\n"
        ),
    ],
)
def test_extract_required_check_names_fails_closed_when_section_missing_or_empty(
    tmp_path: Path,
    doc_text: str,
) -> None:
    doc = tmp_path / "CI_REQUIRED_GATES.md"
    doc.write_text(doc_text, encoding="utf-8")

    with pytest.raises(ValueError, match="(?i)required check names|missing|empty"):
        extract_required_check_names(doc)


def test_normalize_bullet_text_collapses_internal_whitespace() -> None:
    assert normalize_bullet_text("  Add   release   notes \n with tabs\t") == (
        "Add release notes with tabs"
    )


def test_validate_release_branch_rejects_unsupported_branch() -> None:
    with pytest.raises(ValueError, match="main"):
        validate_release_branch("dev")


def test_find_exact_duplicate_bullets_only_within_same_section() -> None:
    duplicate_bullets = find_exact_duplicate_bullets(
        {
            "Added": [
                "  Add release helper core  ",
                "Add   release   helper core",
                "Keep release body stable",
            ],
            "Changed": [
                "Add release helper core",
            ],
            "Fixed": [
                "Separate fix text",
                "Separate    fix   text",
            ],
        }
    )

    assert duplicate_bullets == [
        ("Added", "Add release helper core"),
        ("Fixed", "Separate fix text"),
    ]


def test_promote_changelog_unreleased_section_moves_content_and_resets_headings() -> None:
    changelog_text = (
        "# Changelog\n\n"
        "## [Unreleased]\n\n"
        "### Added\n\n"
        "- Add release helper core\n"
        "- Expand release helper warnings\n\n"
        "### Changed\n\n"
        "- Update release metadata coherently\n\n"
        "### Fixed\n\n"
        "- Fix release helper output\n\n"
        "### Removed\n\n"
        "## [0.1.29] - 2026-03-29\n\n"
        "### Added\n\n"
        "- Existing release note\n"
    )

    promoted_text, warnings = promote_changelog_unreleased_section(
        changelog_text=changelog_text,
        version="0.1.30",
        release_date="2026-04-22",
    )

    assert warnings == []
    assert "## [0.1.30] - 2026-04-22" in promoted_text
    unreleased_block = promoted_text.split("## [Unreleased]", 1)[1].split(
        "## [0.1.30] - 2026-04-22", 1
    )[0]
    assert "### Added" in unreleased_block
    assert "### Changed" in unreleased_block
    assert "### Fixed" in unreleased_block
    assert "### Removed" in unreleased_block
    assert "- Add release helper core" not in unreleased_block
    assert "- Add release helper core" in promoted_text
    assert promoted_text.index("## [Unreleased]") < promoted_text.index("## [0.1.30] - 2026-04-22")


def test_promote_changelog_unreleased_section_preserves_wrapped_bullets() -> None:
    changelog_text = (
        "# Changelog\n\n"
        "## [Unreleased]\n\n"
        "### Added\n\n"
        "- Add release helper core\n"
        "  with continuation details\n"
        "  - and a nested note\n\n"
        "### Fixed\n\n"
        "- Keep release output stable\n"
    )

    promoted_text, warnings = promote_changelog_unreleased_section(
        changelog_text=changelog_text,
        version="0.1.30",
        release_date="2026-04-22",
    )

    assert warnings == []
    assert "- Add release helper core with continuation details - and a nested note" in promoted_text
    assert "- Keep release output stable" in promoted_text


def test_promote_changelog_unreleased_section_rejects_exact_duplicates_within_subsection() -> None:
    changelog_text = (
        "# Changelog\n\n"
        "## [Unreleased]\n\n"
        "### Added\n\n"
        "- Duplicate release helper bullet\n"
        "- Duplicate   release   helper   bullet\n\n"
        "### Changed\n\n"
        "- Update release metadata coherently\n"
    )

    with pytest.raises(ValueError, match="(?i)duplicate"):
        promote_changelog_unreleased_section(
            changelog_text=changelog_text,
            version="0.1.30",
            release_date="2026-04-22",
        )


def test_promote_changelog_unreleased_section_reports_cross_section_near_duplicates_as_warnings() -> None:
    changelog_text = (
        "# Changelog\n\n"
        "## [Unreleased]\n\n"
        "### Added\n\n"
        "- Add release helper core\n\n"
        "### Changed\n\n"
        "- Add release helper core docs and metadata\n"
    )

    promoted_text, warnings = promote_changelog_unreleased_section(
        changelog_text=changelog_text,
        version="0.1.30",
        release_date="2026-04-22",
    )

    assert "## [0.1.30] - 2026-04-22" in promoted_text
    assert warnings
    assert any("Added" in warning and "Changed" in warning for warning in warnings)


def test_promote_changelog_unreleased_section_rejects_headings_without_bullets() -> None:
    changelog_text = (
        "# Changelog\n\n"
        "## [Unreleased]\n\n"
        "### Added\n\n"
        "### Changed\n\n"
        "### Fixed\n\n"
        "### Removed\n\n"
        "## [0.1.30] - 2026-04-20\n\n"
        "### Added\n\n"
        "- Existing release entry\n"
    )

    with pytest.raises(ValueError, match="(?i)unreleased.*no bullets|no bullets"):
        promote_changelog_unreleased_section(
            changelog_text=changelog_text,
            version="0.1.31",
            release_date="2026-04-27",
        )


@pytest.mark.parametrize(
    "unsupported_slice",
    [
        (
            "# Changelog\n\n"
            "## [Unreleased]\n\n"
            "Some prose that should not be silently dropped.\n"
        ),
        (
            "# Changelog\n\n"
            "## [Unreleased]\n\n"
            "### Notes\n\n"
            "- Unsupported subsection content\n"
        ),
        (
            "# Changelog\n\n"
            "## [Unreleased]\n\n"
            "### Added\n\n"
            "Ambiguous subsection prose\n"
        ),
    ],
)
def test_promote_changelog_unreleased_section_rejects_unsupported_content(unsupported_slice: str) -> None:
    with pytest.raises(ValueError, match="(?i)unreleased|unsupported|ambiguous"):
        promote_changelog_unreleased_section(
            changelog_text=unsupported_slice,
            version="0.1.30",
            release_date="2026-04-22",
        )


def test_update_readme_release_references_raises_when_expected_anchors_missing() -> None:
    with pytest.raises(ValueError, match="(?i)readme|release line|anchor"):
        update_readme_release_references("README without current release anchors", "0.1.30")


def test_update_mkdocs_version_metadata_raises_when_expected_anchors_missing() -> None:
    with pytest.raises(ValueError, match="(?i)mkdocs|anchor|version"):
        update_mkdocs_version_metadata("extra:\n  generator: false\n", "0.1.30")


def test_update_release_notes_entry_point_raises_when_anchor_missing() -> None:
    with pytest.raises(ValueError, match="(?i)release notes|anchor"):
        update_release_notes_entry_point("Published release notes entry point.", "Docs/Development/Release_Process.md")


def test_classify_release_state_rejects_inconsistent_snapshot() -> None:
    with pytest.raises(ValueError, match="inconsistent"):
        classify_release_state(
            local_release_commit_exists=True,
            local_tag_exists=False,
            remote_tag_exists=False,
            github_release_exists=True,
        )


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {
                "local_release_commit_exists": True,
                "local_tag_exists": False,
                "remote_tag_exists": False,
                "github_release_exists": False,
            },
            "local_release_commit_only",
        ),
        (
            {
                "local_release_commit_exists": False,
                "local_tag_exists": True,
                "remote_tag_exists": False,
                "github_release_exists": False,
            },
            "local_tag_only",
        ),
        (
            {
                "local_release_commit_exists": False,
                "local_tag_exists": False,
                "remote_tag_exists": True,
                "github_release_exists": False,
            },
            "remote_tag_without_github_release",
        ),
        (
            {
                "local_release_commit_exists": True,
                "local_tag_exists": True,
                "remote_tag_exists": True,
                "github_release_exists": True,
            },
            "existing_github_release",
        ),
    ],
)
def test_classify_release_state(kwargs: dict[str, bool], expected: str) -> None:
    assert classify_release_state(**kwargs) == expected


class StubReleaseRunner:
    def __init__(self) -> None:
        self.branch = "main"
        self.dry_run = False
        self.worktree_clean = True
        self.origin_main_sha = "abc123prebump"
        self.head_sha = self.origin_main_sha
        self.local_tag_target_sha: str | None = self.origin_main_sha
        self.required_checks = {
            "backend-required": "success",
            "security-required": "success",
            "coverage-required": "success",
            "frontend-required": "success",
            "e2e-required": "success",
            "container-build-check": "success",
        }
        self.release_state = "fresh"
        self.remote_main_sha = self.origin_main_sha
        self.push_main_error: str | None = None
        self.recorded_check_sha: list[str] = []
        self.metadata_warnings: list[str] = []
        self.metadata_prepared = False
        self.release_notes_from_tag_prepared = False
        self.commit_created = False
        self.tag_created = False
        self.github_release_created = False
        self.pushed_main = False
        self.pushed_tag = False
        self.operations: list[str] = []

    def fetch_origin_main(self) -> str:
        self.operations.append("fetch_origin_main")
        return self.origin_main_sha

    def get_current_branch(self) -> str:
        self.operations.append("get_current_branch")
        return self.branch

    def is_worktree_clean(self) -> bool:
        self.operations.append("is_worktree_clean")
        return self.worktree_clean

    def get_origin_main_sha(self) -> str:
        self.operations.append("get_origin_main_sha")
        return self.origin_main_sha

    def get_head_sha(self) -> str:
        self.operations.append("get_head_sha")
        return self.head_sha

    def get_local_tag_target_sha(self, version: str) -> str | None:
        self.operations.append(f"get_local_tag_target_sha:{version}")
        return self.local_tag_target_sha

    def get_release_state(self, version: str) -> str:
        self.operations.append(f"get_release_state:{version}")
        return self.release_state

    def ensure_required_checks_green(self, sha: str, required_checks: list[str]) -> None:
        self.operations.append(f"ensure_required_checks_green:{sha}")
        self.recorded_check_sha.append(sha)
        failing = [
            check_name
            for check_name in required_checks
            if self.required_checks.get(check_name) != "success"
        ]
        if failing:
            raise RuntimeError(f"Required checks not green for {sha}: {failing}")

    def prepare_metadata(self, current_version: str, next_version: str) -> None:
        self.operations.append(f"prepare_metadata:{current_version}->{next_version}")
        self.metadata_prepared = True

    def get_metadata_warnings(self) -> list[str]:
        self.operations.append("get_metadata_warnings")
        return self.metadata_warnings

    def prepare_release_notes_from_tag(self, version: str) -> None:
        self.operations.append(f"prepare_release_notes_from_tag:{version}")
        self.release_notes_from_tag_prepared = True

    def create_release_commit(self, next_version: str) -> str:
        self.operations.append(f"create_release_commit:{next_version}")
        self.commit_created = True
        return f"release-commit-{next_version}"

    def create_or_update_tag(self, version: str) -> None:
        self.operations.append(f"create_or_update_tag:{version}")
        self.tag_created = True

    def get_remote_main_sha(self) -> str:
        self.operations.append("get_remote_main_sha")
        return self.remote_main_sha

    def push_main(self) -> None:
        self.operations.append("push_main")
        if self.push_main_error is not None:
            raise RuntimeError(self.push_main_error)
        self.pushed_main = True

    def push_tag(self, version: str) -> None:
        self.operations.append(f"push_tag:{version}")
        self.pushed_tag = True

    def create_github_release(self, version: str) -> bool:
        self.operations.append(f"create_github_release:{version}")
        if self.dry_run:
            return False
        self.github_release_created = True
        return True


def test_orchestrate_release_requires_main_branch() -> None:
    runner = StubReleaseRunner()
    runner.branch = "dev"

    with pytest.raises(ValueError, match="expected 'main'"):
        orchestrate_release(
            bump="patch",
            runner=runner,
            pyproject_path=REPO_ROOT / "pyproject.toml",
            ci_gates_doc_path=REPO_ROOT / "Docs" / "Development" / "CI_REQUIRED_GATES.md",
        )

    assert runner.operations[:2] == ["fetch_origin_main", "get_current_branch"]
    assert "prepare_metadata" not in " ".join(runner.operations)


def test_orchestrate_release_requires_clean_worktree() -> None:
    runner = StubReleaseRunner()
    runner.worktree_clean = False

    with pytest.raises(RuntimeError, match="clean worktree"):
        orchestrate_release(
            bump="patch",
            runner=runner,
            pyproject_path=REPO_ROOT / "pyproject.toml",
            ci_gates_doc_path=REPO_ROOT / "Docs" / "Development" / "CI_REQUIRED_GATES.md",
        )

    assert runner.operations[:3] == [
        "fetch_origin_main",
        "get_current_branch",
        "is_worktree_clean",
    ]
    assert runner.metadata_prepared is False


def test_orchestrate_release_hard_aborts_on_non_fast_forward_push_failure() -> None:
    runner = StubReleaseRunner()
    runner.remote_main_sha = "def456moved"
    runner.push_main_error = "git push origin main failed: non-fast-forward"

    with pytest.raises(RuntimeError, match="non-fast-forward"):
        orchestrate_release(
            bump="patch",
            runner=runner,
            pyproject_path=REPO_ROOT / "pyproject.toml",
            ci_gates_doc_path=REPO_ROOT / "Docs" / "Development" / "CI_REQUIRED_GATES.md",
        )

    assert runner.recorded_check_sha == ["abc123prebump"]
    assert runner.commit_created is True
    assert runner.tag_created is True
    assert runner.pushed_main is False
    assert runner.pushed_tag is False
    assert runner.github_release_created is False
    assert runner.operations.count("fetch_origin_main") == 1


def test_orchestrate_release_resumes_from_remote_tag_without_github_release() -> None:
    runner = StubReleaseRunner()
    runner.release_state = "remote_tag_without_github_release"

    result = orchestrate_release(
        bump="patch",
        runner=runner,
        pyproject_path=REPO_ROOT / "pyproject.toml",
        ci_gates_doc_path=REPO_ROOT / "Docs" / "Development" / "CI_REQUIRED_GATES.md",
    )

    assert result["state"] == "remote_tag_without_github_release"
    assert result["github_release_created"] is True
    assert runner.release_notes_from_tag_prepared is True
    assert runner.github_release_created is True
    assert runner.commit_created is False
    assert runner.tag_created is False
    assert runner.pushed_main is False
    assert runner.pushed_tag is False
    assert "prepare_metadata" not in " ".join(runner.operations)
    assert runner.operations.index("prepare_release_notes_from_tag:0.1.32") < runner.operations.index(
        "create_github_release:0.1.32"
    )


def test_orchestrate_release_local_release_commit_only_reads_head_sha_once() -> None:
    runner = StubReleaseRunner()
    runner.release_state = "local_release_commit_only"
    runner.head_sha = "release-commit-on-head"

    result = orchestrate_release(
        bump="patch",
        runner=runner,
        pyproject_path=REPO_ROOT / "pyproject.toml",
        ci_gates_doc_path=REPO_ROOT / "Docs" / "Development" / "CI_REQUIRED_GATES.md",
    )

    assert result["release_commit_sha"] == "release-commit-on-head"
    assert runner.operations.count("get_head_sha") == 1


def test_orchestrate_release_reports_dry_run_without_created_release() -> None:
    runner = StubReleaseRunner()
    runner.dry_run = True

    result = orchestrate_release(
        bump="patch",
        runner=runner,
        pyproject_path=REPO_ROOT / "pyproject.toml",
        ci_gates_doc_path=REPO_ROOT / "Docs" / "Development" / "CI_REQUIRED_GATES.md",
    )

    assert result["dry_run"] is True
    assert result["github_release_created"] is False
    assert runner.github_release_created is False


def test_orchestrate_release_includes_metadata_warnings() -> None:
    runner = StubReleaseRunner()
    runner.metadata_warnings = [
        "Near-duplicate changelog bullets across Added and Changed: 'a' ~ 'a docs'"
    ]

    result = orchestrate_release(
        bump="patch",
        runner=runner,
        pyproject_path=REPO_ROOT / "pyproject.toml",
        ci_gates_doc_path=REPO_ROOT / "Docs" / "Development" / "CI_REQUIRED_GATES.md",
    )

    assert result["warnings"] == runner.metadata_warnings


def test_orchestrate_release_uses_prebump_origin_main_sha_for_required_checks_only() -> None:
    runner = StubReleaseRunner()

    result = orchestrate_release(
        bump="patch",
        runner=runner,
        pyproject_path=REPO_ROOT / "pyproject.toml",
        ci_gates_doc_path=REPO_ROOT / "Docs" / "Development" / "CI_REQUIRED_GATES.md",
    )

    assert result["validated_sha"] == "abc123prebump"
    assert runner.recorded_check_sha == ["abc123prebump"]
    assert all("release-commit-" not in operation for operation in runner.operations if operation.startswith("ensure_required_checks_green:"))
    assert runner.commit_created is True
    assert runner.pushed_main is True
    assert runner.pushed_tag is True
    assert runner.github_release_created is True


def test_orchestrate_release_local_tag_only_fails_closed_when_tag_points_behind_head() -> None:
    runner = StubReleaseRunner()
    runner.release_state = "local_tag_only"
    runner.head_sha = "newer-head-commit"
    runner.local_tag_target_sha = "older-release-commit"

    with pytest.raises(RuntimeError, match="local release tag.*HEAD|tag.*behind"):
        orchestrate_release(
            bump="patch",
            runner=runner,
            pyproject_path=REPO_ROOT / "pyproject.toml",
            ci_gates_doc_path=REPO_ROOT / "Docs" / "Development" / "CI_REQUIRED_GATES.md",
        )

    assert runner.pushed_main is False
    assert runner.pushed_tag is False
    assert runner.github_release_created is False


def test_shell_release_runner_prepare_metadata_allows_cross_section_warning(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo_root = tmp_path
    (repo_root / "Docs" / "Published").mkdir(parents=True)

    (repo_root / "pyproject.toml").write_text(
        """
[project]
name = "tldw-server"
version = "0.1.30"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (repo_root / "CHANGELOG.md").write_text(
        "# Changelog\n\n"
        "## [Unreleased]\n\n"
        "### Added\n\n"
        "- Add release helper core\n\n"
        "### Changed\n\n"
        "- Add release helper core docs and metadata\n\n"
        "### Fixed\n\n"
        "- Fix release helper output\n\n"
        "### Removed\n\n"
        "## [0.1.30] - 2026-04-20\n\n"
        "### Added\n\n"
        "- Existing release entry\n",
        encoding="utf-8",
    )
    (repo_root / "README.md").write_text(
        "- `0.1.30` Beta status. Expect rough edges and please report issues.\n"
        "- The `dev` branch currently contains additional unreleased work beyond `0.1.30`; see [CHANGELOG.md](CHANGELOG.md) for branch-level detail and [Docs/Published/RELEASE_NOTES.md](Docs/Published/RELEASE_NOTES.md) for the published release entry point.\n"
        "Currently landing on `dev` (post-`0.1.30` branch work):\n",
        encoding="utf-8",
    )
    (repo_root / "Docs" / "mkdocs.yml").write_text(
        'extra:\n  version: v0.1.30\ncopyright: |\n  © 2024-2025 tldw_Server - v0.1.30 - <a href="https://github.com/rmusser01/tldw_server">GitHub</a>\n',
        encoding="utf-8",
    )
    (repo_root / "Docs" / "Published" / "RELEASE_NOTES.md").write_text(
        "Published release notes entry point.\n\nFor release process details, see `Docs/Development/Release_Process.md`.\n",
        encoding="utf-8",
    )

    runner = ShellReleaseRunner(repo_root=repo_root, dry_run=True)

    runner.prepare_metadata("0.1.30", "0.1.31")

    assert runner._release_notes_body is not None
    assert "## [0.1.31] - " in runner._release_notes_body
    assert "Add release helper core docs and metadata" in runner._release_notes_body
    assert runner.get_metadata_warnings()
    assert all("Docs/site" not in str(path) for path in runner._prepared_paths)


def test_shell_release_runner_run_command_uses_absolute_executable_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    recorded_commands: list[list[str]] = []

    def _fake_run(
        command_args: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        recorded_commands.append(list(command_args))
        return subprocess.CompletedProcess(command_args, 0, "ok", "")

    monkeypatch.setattr(release_module.subprocess, "run", _fake_run)

    runner = ShellReleaseRunner(repo_root=tmp_path, dry_run=False)

    runner._run_command(["git", "status"])

    assert recorded_commands
    assert Path(recorded_commands[0][0]).is_absolute()


def test_shell_release_runner_run_command_rejects_relative_resolved_executable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def _fake_which(executable: str) -> str:
        assert executable == "git"
        return "relative-bin/git"

    def _fake_run(
        command_args: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        raise AssertionError(f"subprocess should not run relative command: {command_args}")

    monkeypatch.setattr(release_module.shutil, "which", _fake_which)
    monkeypatch.setattr(release_module.subprocess, "run", _fake_run)

    runner = ShellReleaseRunner(repo_root=tmp_path, dry_run=False)

    with pytest.raises(FileNotFoundError, match="absolute path"):
        runner._run_command(["git", "status"])


def test_shell_release_runner_create_or_update_tag_uses_annotated_tag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    recorded_commands: list[list[str]] = []

    def _fake_run_command(
        self: ShellReleaseRunner,
        args: list[str],
        *,
        check: bool = True,
        capture_output: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        recorded_commands.append(list(args))
        return subprocess.CompletedProcess(list(args), 0, "", "")

    monkeypatch.setattr(ShellReleaseRunner, "_run_command", _fake_run_command)

    runner = ShellReleaseRunner(repo_root=tmp_path, dry_run=False)

    runner.create_or_update_tag("1.2.3")

    assert recorded_commands == [["git", "tag", "-a", "v1.2.3", "-m", "v1.2.3"]]


def test_shell_release_runner_required_checks_reads_paginated_check_runs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []

    def _fake_run_text(self: ShellReleaseRunner, args: list[str]) -> str:
        calls.append(list(args))
        if args[:3] == ["gh", "repo", "view"]:
            return "example/repo"
        if args[:2] == ["gh", "api"]:
            return json.dumps(
                [
                    {
                        "check_runs": [
                            {
                                "name": "backend-required",
                                "status": "completed",
                                "conclusion": "success",
                            }
                        ]
                    },
                    {
                        "check_runs": [
                            {
                                "name": "security-required",
                                "status": "completed",
                                "conclusion": "success",
                            }
                        ]
                    },
                ]
            )
        raise AssertionError(f"Unexpected command: {args}")

    monkeypatch.setattr(ShellReleaseRunner, "_run_text", _fake_run_text)

    runner = ShellReleaseRunner(repo_root=tmp_path, dry_run=False)

    runner.ensure_required_checks_green(
        "abc123",
        ["backend-required", "security-required"],
    )

    api_calls = [call for call in calls if call[:2] == ["gh", "api"]]
    assert api_calls == [
        [
            "gh",
            "api",
            "--paginate",
            "--slurp",
            "repos/example/repo/commits/abc123/check-runs?per_page=100",
        ]
    ]
