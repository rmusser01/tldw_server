from __future__ import annotations

import shutil
import subprocess  # nosec B404
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

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

SUPPLY_CHAIN_RUNBOOK = REPO_ROOT / "Docs/Development/Software_Supply_Chain.md"
RELEASE_PROCESS = REPO_ROOT / "Docs/Development/Release_Process.md"
PYPI_GUIDE = REPO_ROOT / "Docs/Development/PyPI_Publishing.md"
PINNED_SUPPLY_CHAIN_TOOLS = (
    "ghcr.io/astral-sh/uv:0.12.7@sha256:95f2aa1fe59274951cfe9b0cbc7972e879ff1004bc8945d130a32eb0dbd85945",
    "ghcr.io/cdxgen/cdxgen:v13@sha256:0be75639a833b59d1ba29b3c8ac00dfd2e41e7568d56b6c039007caadebebc0d",
    "docker.io/cyclonedx/cyclonedx-cli:0.33.1@sha256:252c2e26f468c25fea1e63ecde1bc3198ad6e9dbb57f5ed3236bddcb2281b3a7",
    "ghcr.io/aquasecurity/trivy:0.74.0@sha256:62b1e65e8869bc4b4c6aa4fa2b21595256c7c2f6018a9d9ad61caf87187c1969",
)


def _workflow(relative_path: str) -> dict[str, object]:
    # BaseLoader reads trusted local workflow text without constructing Python objects.
    loaded = yaml.load(  # nosec B506
        (REPO_ROOT / relative_path).read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    assert isinstance(loaded, dict)
    return loaded


def _mkdocs_config() -> dict[str, object]:
    loaded = yaml.safe_load((REPO_ROOT / "Docs/mkdocs.yml").read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def test_mkdocs_output_does_not_alias_canonical_site_sources() -> None:
    config = _mkdocs_config()
    config_dir = REPO_ROOT / "Docs"
    configured_site_dir = str(config.get("site_dir", "site"))
    canonical_source = (config_dir / "Site").resolve()
    output = (config_dir / configured_site_dir).resolve()

    assert configured_site_dir == "_site"
    assert canonical_source.as_posix().casefold() != output.as_posix().casefold()


def test_mkdocs_workflow_uploads_configured_site_output() -> None:
    config = _mkdocs_config()
    deploy = _workflow(".github/workflows/mkdocs.yml")
    deploy_steps = deploy["jobs"]["build"]["steps"]
    upload_step = next(step for step in deploy_steps if step["name"] == "Upload artifact")

    assert upload_step["with"]["path"] == f"Docs/{config['site_dir']}"


def test_docs_workflows_enforce_strict_build_and_boundary_paths() -> None:
    deploy = _workflow(".github/workflows/mkdocs.yml")
    gate = _workflow(".github/workflows/onboarding-docs-gate.yml")
    strict = "mkdocs build --strict -f Docs/mkdocs.yml"

    gate_on = gate["on"]
    assert isinstance(gate_on, dict)
    for event in ("pull_request", "push"):
        paths = gate_on[event]["paths"]
        assert "Helper_Scripts/refresh_docs_published.sh" in paths
        assert ".github/workflows/mkdocs.yml" in paths

    gate_steps = gate["jobs"]["onboarding-docs-gate"]["steps"]
    deploy_steps = deploy["jobs"]["build"]["steps"]
    gate_by_name = {step["name"]: step for step in gate_steps}
    deploy_by_name = {step["name"]: step for step in deploy_steps}

    assert strict in gate_by_name["MkDocs build"]["run"].splitlines()
    assert strict in deploy_by_name["Build site"]["run"].splitlines()
    assert "continue-on-error" not in gate_by_name["MkDocs build"]
    assert "continue-on-error" not in deploy_by_name["Build site"]
    assert (
        "python Helper_Scripts/docs/check_public_private_boundary.py"
        in gate_by_name["Check public/private docs boundary"]["run"].splitlines()
    )

    gate_names = [step["name"] for step in gate_steps]
    assert (
        gate_names.index("Refresh curated docs")
        < gate_names.index("Check public/private docs boundary")
        < gate_names.index("Onboarding command boundary check")
        < gate_names.index("Onboarding endpoint drift check")
        < gate_names.index("Docs test suite")
        < gate_names.index("MkDocs build")
    )
    deploy_names = [step["name"] for step in deploy_steps]
    assert (
        deploy_names.index("Refresh curated docs")
        < deploy_names.index("Check public/private docs boundary")
        < deploy_names.index("Build site")
    )


def test_docs_site_guide_requires_strict_for_every_operator_build() -> None:
    guide = (REPO_ROOT / "Docs/Code_Documentation/Docs_Site_Guide.md").read_text(encoding="utf-8")
    build_lines = [line for line in guide.splitlines() if "mkdocs build" in line]

    assert build_lines
    assert all("mkdocs build --strict -f Docs/mkdocs.yml" in line for line in build_lines)


def test_docs_site_guide_describes_dev_build_without_deployment() -> None:
    guide = (REPO_ROOT / "Docs/Code_Documentation/Docs_Site_Guide.md").read_text(encoding="utf-8")

    assert "pushes to `dev`, `main`, and `PG-Backend`" in guide
    assert "`dev` builds are validated but are not deployed" in guide


def test_strict_local_build_preserves_canonical_site_sources() -> None:
    config = _mkdocs_config()
    canonical_dir = (REPO_ROOT / "Docs/Site").resolve()
    configured_site_dir = str(config.get("site_dir", "site"))
    output = (REPO_ROOT / "Docs" / configured_site_dir).resolve()
    canonical_files = (canonical_dir / "index.md", canonical_dir / "RELEASE_NOTES.md")
    before = {path: path.read_bytes() for path in canonical_files}

    assert configured_site_dir == "_site"
    assert output.parent == (REPO_ROOT / "Docs").resolve()
    assert canonical_dir.as_posix().casefold() != output.as_posix().casefold()
    if output.exists():
        shutil.rmtree(output)
    try:
        result = subprocess.run(  # nosec B603
            [
                sys.executable,
                "-m",
                "mkdocs",
                "build",
                "--strict",
                "-f",
                "Docs/mkdocs.yml",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert output.is_dir()
        assert before == {path: path.read_bytes() for path in canonical_files}
    finally:
        if output.exists():
            shutil.rmtree(output)


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
    assert f"The `dev` branch currently contains additional unreleased work beyond `{target_version}`;" in updated_text
    assert f"Currently landing on `dev` (post-`{target_version}` branch work):" in updated_text


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
        '  © 2024-2025 tldw_Server - v0.1.19 - <a href="https://github.com/rmusser01/tldw_server">GitHub</a>\n'
    )

    updated_text = update_mkdocs_version_metadata(mkdocs_text, target_version)

    assert f"version: v{target_version}" in updated_text
    assert f'v{target_version} - <a href="https://github.com/rmusser01/tldw_server">GitHub</a>' in updated_text
    assert "© 2024-2025 tldw_Server" in updated_text


def test_mkdocs_version_metadata_does_not_depend_on_copyright_url() -> None:
    mkdocs_text = (
        "extra:\n"
        "  generator: false\n"
        "  version: v0.1.19\n"
        "copyright: |\n"
        '  © 2024-2025 tldw_Server - v0.1.19 - <a href="https://example.com/project">Project</a>\n'
    )

    updated_text = update_mkdocs_version_metadata(mkdocs_text, "0.1.31")

    assert "version: v0.1.31" in updated_text
    assert 'v0.1.31 - <a href="https://example.com/project">Project</a>' in updated_text


def test_mkdocs_version_metadata_updates_version_inside_multiline_copyright() -> None:
    mkdocs_text = (
        "extra:\n"
        "  generator: false\n"
        "  version: v0.1.19\n"
        "copyright: |\n"
        "  Maintained by tldw_Server contributors.\n"
        "  Release train: v0.1.19\n"
        '  <a href="https://example.com/project">Project</a>\n'
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


def test_release_helper_updates_repository_readme_release_references() -> None:
    readme_text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    target_version = "9.9.9"

    updated_text = update_readme_release_references(readme_text, target_version)

    assert f"`{target_version}` Beta status. Expect rough edges and please report issues." in updated_text
    assert f"beyond `{target_version}`" in updated_text
    assert f"post-`{target_version}` branch work" in updated_text


def test_release_helper_raises_when_post_release_anchor_is_missing() -> None:
    readme_text = (
        "## Current Status\n\n"
        "- `0.1.34` Beta status. Expect rough edges and please report issues.\n"
        "- The `dev` branch carries work beyond `0.1.34`.\n"
    )

    with pytest.raises(ValueError, match="post-release reference"):
        update_readme_release_references(readme_text, "9.9.9")


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

    assert "/Docs/_site/" in gitignore_lines
    assert "/Docs/site/" not in gitignore_lines
    assert "!Docs/_site/**/*.json" not in gitignore_lines

    result = subprocess.run(  # nosec B603 B607
        ["git", "ls-files", "Docs/_site"],
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
    canonical_release_notes_path = REPO_ROOT / "Docs/Site/RELEASE_NOTES.md"
    published_release_notes_path = REPO_ROOT / "Docs/Published/RELEASE_NOTES.md"
    release_checklist_path = REPO_ROOT / "Docs/Release_Checklist.md"

    assert release_process_path.exists(), "expected release process operator doc to exist"

    release_process_text = release_process_path.read_text(encoding="utf-8")
    canonical_release_notes_text = canonical_release_notes_path.read_text(encoding="utf-8")
    published_release_notes_text = published_release_notes_path.read_text(encoding="utf-8")
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
    assert "`Docs/_site/`" in release_process_text
    assert "`Docs/site/`" not in release_process_text

    assert canonical_release_notes_text == published_release_notes_text
    release_process_url = "https://github.com/rmusser01/tldw_server/blob/main/" "Docs/Development/Release_Process.md"
    release_checklist_url = "https://github.com/rmusser01/tldw_server/blob/main/Docs/Release_Checklist.md"
    for release_notes_text in (
        canonical_release_notes_text,
        published_release_notes_text,
    ):
        assert "Docs/Development/Release_Process.md" in release_notes_text
        assert f"]({release_process_url})" in release_notes_text
        assert f"]({release_checklist_url})" in release_notes_text
        assert "](../Development/Release_Process.md)" not in release_notes_text
        assert "](../Release_Checklist.md)" not in release_notes_text

    assert "Docs/Development/Release_Process.md" in release_checklist_text
    assert "broad readiness checklist" in release_checklist_text.lower()


def test_supply_chain_runbook_pins_tools_locks_and_source_evidence() -> None:
    text = SUPPLY_CHAIN_RUNBOOK.read_text(encoding="utf-8")

    for tool in PINNED_SUPPLY_CHAIN_TOOLS:
        assert tool in text
    for command in (
        "uv lock --check",
        "bun install --frozen-lockfile",
    ):
        assert command in text
    for sbom in (
        "sbom-python-root.cdx.json",
        "sbom-apps-workspace.cdx.json",
        "sbom-admin-ui.cdx.json",
        "sbom-source-aggregate.cdx.json",
    ):
        assert sbom in text


def test_supply_chain_runbook_documents_policy_and_exception_lifecycle() -> None:
    text = SUPPLY_CHAIN_RUNBOOK.read_text(encoding="utf-8")

    assert "zero unexcepted Critical or High" in text
    assert "ignore-unfixed=false" in text
    assert "7 days" in text
    assert "30 days" in text
    for field in (
        "`owner`",
        "`rationale`",
        "`mitigation`",
        "`approval`",
        "`created_on`",
        "`expires_on`",
    ):
        assert field in text
    assert "renew" in text.lower()
    assert "past clean scan" in text.lower()
    assert "future safety" in text.lower()


def test_supply_chain_runbook_covers_every_image_and_publication_boundary() -> None:
    text = SUPPLY_CHAIN_RUNBOOK.read_text(encoding="utf-8")

    for name in (
        "app",
        "worker",
        "audio-worker",
        "webui",
        "admin-ui",
        "caddy",
        "postgres",
        "redis",
        "prometheus",
        "alertmanager",
        "grafana",
    ):
        assert f"`{name}`" in text
    assert "`linux/amd64`" in text
    assert "build-and-scan-only" in text
    assert "WebUI and Admin UI" in text
    assert "third-party provenance" in text.lower()
    assert "do not" in text.lower()


def test_supply_chain_runbook_explains_digest_and_attestation_verification() -> None:
    text = SUPPLY_CHAIN_RUNBOOK.read_text(encoding="utf-8")

    assert "tag@sha256:" in text
    assert "subject/index digest" in text
    assert "child manifest digest" in text
    assert "gh attestation verify oci://" in text
    assert "--repo rmusser01/tldw_server" in text
    assert "--signer-workflow" in text
    assert "release-manifest.json" in text
    assert "SHA256SUMS.release" in text


def test_release_and_pypi_guides_describe_current_admission_flow() -> None:
    release_text = RELEASE_PROCESS.read_text(encoding="utf-8")
    pypi_text = PYPI_GUIDE.read_text(encoding="utf-8")

    assert "existing draft GitHub Release" in release_text
    assert "publish <tag>" in release_text
    assert "full-version" in release_text
    assert "floating aliases" in release_text
    assert "Software_Supply_Chain.md" in release_text
    assert "attestations: true" in pypi_text
    assert "Trusted Publishing" in pypi_text
    assert "SHA256SUMS" in pypi_text
    assert "pypi-attestations verify pypi" in pypi_text
    assert "--repository https://github.com/rmusser01/tldw_server" in pypi_text
