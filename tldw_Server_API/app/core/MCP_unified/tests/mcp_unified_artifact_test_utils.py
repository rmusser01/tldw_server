"""Side-effect-free helpers for MCP Unified distribution consumer tests."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
STANDALONE_PROJECT_ROOT = REPO_ROOT / "apps" / "mcp-unified"


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
