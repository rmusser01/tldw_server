from __future__ import annotations

import subprocess  # nosec B404
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import Helper_Scripts.release as release  # noqa: E402
from Helper_Scripts.release import (  # noqa: E402
    read_current_version,
    update_mkdocs_version_metadata,
    update_readme_release_references,
    update_release_notes_entry_point,
)


def test_readme_release_references_update_to_target_version() -> None:
    target_version = read_current_version(REPO_ROOT / "pyproject.toml")
    readme_text = (
        "## Current Status\n\n"
        "Current release line:\n"
        "- `0.3.1` Beta status. Expect rough edges and please report issues.\n"
        "- Primary client surfaces are the Next.js WebUI, Admin UI, and browser extension.\n"
        "- The `dev` branch currently contains additional unreleased work beyond `0.3.1`; "
        "see [CHANGELOG.md](CHANGELOG.md) for branch-level detail and "
        "[Docs/Published/RELEASE_NOTES.md](Docs/Published/RELEASE_NOTES.md) for the published "
        "release entry point.\n\n"
        "Currently landing on `dev` (post-`0.3.1` branch work):\n"
        "- Placeholder\n"
    )

    updated_text = update_readme_release_references(readme_text, target_version)

    assert f"`{target_version}` Beta status. Expect rough edges and please report issues." in updated_text
    assert (
        f"The `dev` branch currently contains additional unreleased work beyond `{target_version}`;"
        in updated_text
    )
    assert (
        f"Currently landing on `dev` (post-`{target_version}` branch work):" in updated_text
    )


def test_mkdocs_version_metadata_updates_coherently() -> None:
    target_version = read_current_version(REPO_ROOT / "pyproject.toml")
    mkdocs_text = (
        "extra:\n"
        "  generator: false\n"
        "  version: v0.1.19\n"
        "  social:\n"
        "    - icon: fontawesome/brands/github\n"
        "      link: https://github.com/rmusser01/tldw_server\n"
        "      name: GitHub\n"
        "copyright: |\n"
        "  © 2024-2025 tldw_Server - v0.1.19 - <a href=\"https://github.com/rmusser01/tldw_server\">GitHub</a>\n"
    )

    updated_text = update_mkdocs_version_metadata(mkdocs_text, target_version)

    assert f"version: v{target_version}" in updated_text
    assert f"v{target_version} - <a href=\"https://github.com/rmusser01/tldw_server\">GitHub</a>" in updated_text
    assert "© 2024-2025 tldw_Server" in updated_text


def test_mkdocs_version_metadata_does_not_depend_on_copyright_url() -> None:
    mkdocs_text = (
        "extra:\n"
        "  generator: false\n"
        "  version: v0.1.19\n"
        "copyright: |\n"
        "  © 2024-2025 tldw_Server - v0.1.19 - <a href=\"https://example.com/project\">Project</a>\n"
    )

    updated_text = update_mkdocs_version_metadata(mkdocs_text, "0.1.31")

    assert "version: v0.1.31" in updated_text
    assert "v0.1.31 - <a href=\"https://example.com/project\">Project</a>" in updated_text


def test_mkdocs_version_metadata_updates_version_inside_multiline_copyright() -> None:
    mkdocs_text = (
        "extra:\n"
        "  generator: false\n"
        "  version: v0.1.19\n"
        "copyright: |\n"
        "  Maintained by tldw_Server contributors.\n"
        "  Release train: v0.1.19\n"
        "  <a href=\"https://example.com/project\">Project</a>\n"
    )

    updated_text = update_mkdocs_version_metadata(mkdocs_text, "0.1.31")

    assert "version: v0.1.31" in updated_text
    assert "Release train: v0.1.31" in updated_text
    assert "https://example.com/project" in updated_text


def test_repository_release_metadata_matches_pyproject() -> None:
    current_version = read_current_version(REPO_ROOT / "pyproject.toml")
    readme_text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    mkdocs_text = (REPO_ROOT / "Docs" / "mkdocs.yml").read_text(encoding="utf-8")

    assert f"`{current_version}` Beta status. Expect rough edges and please report issues." in readme_text
    assert f"beyond `{current_version}`" in readme_text
    assert f"post-`{current_version}` branch work" in readme_text
    assert f"version: v{current_version}" in mkdocs_text
    assert f"v{current_version}" in mkdocs_text


def test_mkdocs_version_metadata_raises_for_missing_anchor() -> None:
    with pytest.raises(ValueError, match="(?i)mkdocs|anchor|version"):
        update_mkdocs_version_metadata("extra:\n  generator: false\n", "0.1.30")


def test_release_notes_entry_point_points_to_authoritative_release_process_doc() -> None:
    release_notes_text = "# Release Notes\n\nPublished release notes entry point.\n\nFor release process details, see `Docs/Release_Checklist.md`.\n"

    updated_text = update_release_notes_entry_point(
        release_notes_text,
        "Docs/Development/Release_Process.md",
    )

    assert "Docs/Development/Release_Process.md" in updated_text
    assert "Docs/Release_Checklist.md" not in updated_text


def test_release_notes_entry_point_raises_for_missing_anchor() -> None:
    with pytest.raises(ValueError, match="(?i)release notes|anchor"):
        update_release_notes_entry_point(
            "# Release Notes\n\nNo process pointer here.\n",
            "Docs/Development/Release_Process.md",
        )


def test_docs_site_repo_policy_keeps_generated_site_untracked() -> None:
    gitignore_lines = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert "/Docs/site/" in gitignore_lines
    assert "!Docs/site/**/*.json" not in gitignore_lines

    result = subprocess.run(  # nosec B603 B607
        ["git", "ls-files", "Docs/site"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""


def test_release_helper_does_not_manage_generated_docs_site_outputs() -> None:
    assert not hasattr(release, "update_docs_site_version_bearing_outputs")


def test_release_process_doc_is_authoritative_operator_path() -> None:
    release_process_path = REPO_ROOT / "Docs/Development/Release_Process.md"
    release_notes_path = REPO_ROOT / "Docs/Published/RELEASE_NOTES.md"
    release_checklist_path = REPO_ROOT / "Docs/Release_Checklist.md"

    assert release_process_path.exists(), "expected release process operator doc to exist"

    release_process_text = release_process_path.read_text(encoding="utf-8")
    release_notes_text = release_notes_path.read_text(encoding="utf-8")
    release_checklist_text = release_checklist_path.read_text(encoding="utf-8")

    assert all(
        command in release_process_text
        for command in ("`make release`", "`make release-patch`", "`make release-minor`")
    )
    assert "`main`" in release_process_text
    assert "Docs/Development/CI_REQUIRED_GATES.md" in release_process_text
    assert "formal release artifacts" in release_process_text.lower()
    assert "main snapshots" in release_process_text.lower()
    assert "release commit" in release_process_text.lower()
    assert "republishes" in release_process_text.lower()
    assert "Docs/Release_Checklist.md" in release_process_text
    assert "retry" in release_process_text.lower() or "rerun" in release_process_text.lower()
    assert "recover" in release_process_text.lower()
    assert "PyPI" in release_process_text
    assert "manual" in release_process_text.lower()

    assert "Docs/Development/Release_Process.md" in release_notes_text
    assert "](../Development/Release_Process.md)" in release_notes_text
    assert "](../Release_Checklist.md)" in release_notes_text

    assert "Docs/Development/Release_Process.md" in release_checklist_text
    assert "broad readiness checklist" in release_checklist_text.lower()
