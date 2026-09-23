"""Side-effect-free helpers for MCP Unified distribution consumer tests."""

from __future__ import annotations

import importlib.metadata as importlib_metadata
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[5]
STANDALONE_PROJECT_ROOT = REPO_ROOT / "apps" / "mcp-unified"


def _declared_build_requirements() -> list[str]:
    """Return the standalone project's declared build-system requirements."""

    pyproject = STANDALONE_PROJECT_ROOT / "pyproject.toml"
    with pyproject.open("rb") as handle:
        return list(tomllib.load(handle)["build-system"]["requires"])


def _requirement_package_name(requirement: str) -> str:
    """Return the bare package name from a requirement string."""

    for separator in (">=", "==", "~=", ">", "<", "[", ";"):
        requirement = requirement.split(separator, 1)[0]
    return requirement.strip().replace("_", "-").lower()


def skip_unless_offline_build_is_possible() -> None:
    """Skip when this environment cannot run the standalone build at all.

    The build below uses ``PIP_NO_INDEX=1`` with ``--no-isolation``, so every
    build-system requirement must already be installed *with dist-info metadata*. A venv
    where setuptools is importable but unregistered satisfies neither: ``python -m build``
    fails with "Backend 'setuptools.build_meta' is not available", and the tests ERRORed
    instead of skipping -- an unrunnable environment reported as a regression.

    This is the necessary condition, checked here because both
    ``test_runtime_package_boundary`` and ``test_gateway_protocol_artifact_consumer``
    build through this helper. ``test_runtime_package_boundary`` additionally checks that
    the installed versions satisfy the declared minimums; that is a stricter guard
    nesting inside this one, not a competing copy of it. See TASK-13291.
    """

    unregistered = [
        requirement
        for requirement in _declared_build_requirements()
        if not _has_distribution_metadata(_requirement_package_name(requirement))
    ]
    if unregistered:
        pytest.skip(
            "standalone build needs its build-system requirements installed with "
            "dist-info metadata (PIP_NO_INDEX=1 with --no-isolation); missing metadata "
            f"for: {', '.join(unregistered)}"
        )


def _has_distribution_metadata(package_name: str) -> bool:
    """Return whether an installed distribution is discoverable by name."""

    try:
        importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return False
    return True


def _subprocess_env(extra_env: dict[str, str] | None = None) -> dict[str, str]:
    """Return a copied environment without changing interpreter state."""

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    return env


def _assert_subprocess_succeeded(
    result: subprocess.CompletedProcess[str],
    command_label: str,
) -> None:
    """Assert a captured build subprocess succeeded with useful diagnostics."""

    if result.returncode != 0:
        raise AssertionError(
            f"{command_label} failed with exit code {result.returncode}:\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )


def build_standalone_distributions(tmp_path: Path) -> tuple[Path, Path]:
    """Build one standalone wheel and sdist from an isolated source copy."""

    skip_unless_offline_build_is_possible()
    package_source = tmp_path / "mcp_unified_source"
    shutil.copytree(
        STANDALONE_PROJECT_ROOT,
        package_source,
        ignore=shutil.ignore_patterns(
            "__pycache__",
            "build",
            "dist",
            "*.egg-info",
        ),
    )
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()

    result = subprocess.run(  # nosec B603
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--sdist",
            "--no-isolation",
            "--outdir",
            str(dist_dir),
            str(package_source),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=_subprocess_env(
            {
                "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                "PIP_NO_INDEX": "1",
            }
        ),
    )
    _assert_subprocess_succeeded(result, "python -m build")

    wheels = sorted(
        [
            *dist_dir.glob("mcp_unified-*.whl"),
            *dist_dir.glob("mcp-unified-*.whl"),
        ]
    )
    sdists = sorted(
        [
            *dist_dir.glob("mcp_unified-*.tar.gz"),
            *dist_dir.glob("mcp-unified-*.tar.gz"),
        ]
    )
    if len(wheels) != 1 or len(sdists) != 1:
        raise AssertionError("standalone build must produce exactly one wheel and one sdist")
    return wheels[0], sdists[0]


def assert_strict_consumer_output(
    result: subprocess.CompletedProcess[str],
    success_marker: str,
) -> None:
    """Require exactly one stdout marker and no downstream stderr output."""

    assert result.stdout == f"{success_marker}\n"  # nosec B101
    assert result.stderr == ""  # nosec B101
