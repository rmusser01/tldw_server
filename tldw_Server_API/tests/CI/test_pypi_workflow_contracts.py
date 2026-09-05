"""Security contracts for reproducible and attested Python package publishing."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import tomllib
import yaml

pytestmark = pytest.mark.unit

PUBLISH_WORKFLOW = Path(".github/workflows/publish-pypi.yml")
PACKAGE_WORKFLOW = Path(".github/workflows/pypi-package.yml")
MAKEFILE = Path("Makefile")
PINNED_UV = (
    "ghcr.io/astral-sh/uv:0.12.7@sha256:"
    "95f2aa1fe59274951cfe9b0cbc7972e879ff1004bc8945d130a32eb0dbd85945"
)


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _workflow_on(workflow: dict) -> dict:
    return workflow[True]


def _get_step(steps: list[dict], name: str) -> dict:
    matching = [step for step in steps if step.get("name") == name]
    assert matching, f"{name} step missing"
    return matching[0]


def _run(step: dict) -> str:
    script = step.get("run")
    assert isinstance(script, str), step
    return script


def _assert_changed_actions_are_pinned(path: Path) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("uses:") and not stripped.endswith(".yml"):
            target = stripped.removeprefix("uses:").strip()
            if target.startswith("./"):
                continue
            assert re.fullmatch(r"[^@\s]+@[0-9a-f]{40}", target)


def test_publish_workflow_keeps_manual_and_guarded_main_triggers() -> None:
    workflow = _load(PUBLISH_WORKFLOW)
    triggers = _workflow_on(workflow)

    assert "workflow_dispatch" in triggers
    assert triggers["push"]["branches"] == ["main"]
    assert "pyproject.toml" in triggers["push"]["paths"]
    assert "tldw_Server_API/**" in triggers["push"]["paths"]


@pytest.mark.parametrize("path", [PUBLISH_WORKFLOW, PACKAGE_WORKFLOW])
def test_package_build_uses_pinned_uv_and_locked_release_group(path: Path) -> None:
    workflow = _load(path)
    build = workflow["jobs"]["build"]

    assert workflow["env"]["UV_IMAGE"] == PINNED_UV
    _get_step(build["steps"], "Install pinned uv")
    sync = _run(_get_step(build["steps"], "Sync locked release tools"))
    assert "uv sync --locked --only-group release" in sync
    package_step = next(
        step for step in build["steps"] if step.get("name") in {
            "Build and validate package",
            "Build and check package",
        }
    )
    assert "PYPI_BUILD_ARGS=--no-isolation" in _run(package_step)
    assert "pip install build" not in path.read_text(encoding="utf-8")


def test_publish_requires_source_admission_before_build_and_publish() -> None:
    workflow = _load(PUBLISH_WORKFLOW)
    jobs = workflow["jobs"]

    assert jobs["source-admission"]["uses"] == "./.github/workflows/sbom.yml"
    assert "source-admission" in jobs["build"]["needs"]
    assert "build" in jobs["publish-test"]["needs"]
    assert "build" in jobs["publish-pypi"]["needs"]


@pytest.mark.parametrize("path", [PUBLISH_WORKFLOW, PACKAGE_WORKFLOW])
def test_package_artifact_includes_distribution_checksums(path: Path) -> None:
    workflow = _load(path)
    steps = workflow["jobs"]["build"]["steps"]
    checksum = _run(_get_step(steps, "Hash checked distributions"))
    upload = _get_step(steps, "Upload distributions")

    assert "SHA256SUMS" in checksum
    assert "xargs -0 -r sha256sum" in checksum
    assert "-name '*.whl'" in checksum
    assert "-name '*.tar.gz'" in checksum
    assert upload["with"]["path"] == "dist/"
    assert upload["with"]["if-no-files-found"] == "error"


@pytest.mark.parametrize("job_name", ["publish-test", "publish-pypi"])
def test_publish_jobs_verify_same_artifact_and_request_attestations(job_name: str) -> None:
    workflow = _load(PUBLISH_WORKFLOW)
    steps = workflow["jobs"][job_name]["steps"]
    names = [step.get("name") for step in steps]
    download = _get_step(steps, "Download checked distributions")
    verify = _get_step(steps, "Verify distribution checksums")
    publish = next(step for step in steps if str(step.get("uses", "")).startswith("pypa/"))

    assert download["with"]["name"] == "python-distributions"
    assert download["with"]["path"] == "dist"
    assert "sha256sum -c SHA256SUMS" in _run(verify)
    assert "rm SHA256SUMS" in _run(verify)
    assert names.index("Verify distribution checksums") < steps.index(publish)
    assert publish["with"]["attestations"] is True
    if job_name == "publish-test":
        assert publish["with"]["repository-url"] == "https://test.pypi.org/legacy/"


def test_version_detection_handles_registry_failures_explicitly() -> None:
    workflow = _load(PUBLISH_WORKFLOW)
    script = _run(
        _get_step(workflow["jobs"]["detect-version"]["steps"], "Detect version state")
    )

    assert "JSONDecodeError" in script
    assert "TimeoutError" in script


def test_pypi_make_targets_do_not_require_pip_in_the_locked_environment() -> None:
    packaging_targets = MAKEFILE.read_text(encoding="utf-8").split(
        "# MCP Unified standalone RC", maxsplit=1
    )[0]

    assert "-m pip show" not in packaging_targets
    assert '-c "import build"' in packaging_targets
    assert '-c "import twine"' in packaging_targets
    assert '-c "import loguru"' in packaging_targets


def test_release_group_includes_artifact_checker_logging_dependency() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert "loguru==0.7.3" in project["dependency-groups"]["release"]


@pytest.mark.parametrize("path", [PUBLISH_WORKFLOW, PACKAGE_WORKFLOW])
def test_changed_pypi_workflow_actions_are_commit_pinned(path: Path) -> None:
    _assert_changed_actions_are_pinned(path)
