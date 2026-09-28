"""TASK-13380: test runs must import this checkout's first-party packages.

The shared venv installs mcp_unified and tldw_profile_core in editable mode, and those
.pth files pin absolute paths into the main checkout. Plain ``python`` from a worktree
therefore imports the MAIN checkout's copy. pytest does not, because
``[tool.pytest.ini_options] pythonpath`` puts this checkout's src dirs first -- which is
the only thing standing between a worktree test run and silently testing code that was
never changed. These tests keep that true: every src-layout package must be listed, and
each must actually resolve inside the checkout running the tests.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_LAYOUT_GLOBS = ("apps/*/src", "packages/*/src")


def _src_dirs() -> list[Path]:
    """Every src-layout directory (apps/*/src, packages/*/src) in this checkout."""
    return sorted(p for pattern in SRC_LAYOUT_GLOBS for p in REPO_ROOT.glob(pattern) if p.is_dir())


def _first_party_packages() -> list[tuple[str, Path]]:
    """(package name, its src dir) for each importable package under those dirs."""
    return [
        (pkg.name, src)
        for src in _src_dirs()
        for pkg in sorted(src.iterdir())
        if (pkg / "__init__.py").is_file()
    ]


def test_pytest_pythonpath_lists_every_src_layout_package() -> None:
    """A new apps/*/src or packages/*/src must be added, or worktree runs test main's copy."""
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    listed = set(config["tool"]["pytest"]["ini_options"]["pythonpath"])
    expected = {src.relative_to(REPO_ROOT).as_posix() for src in _src_dirs()}

    missing = expected - listed
    assert not missing, (
        f"{sorted(missing)} are src-layout packages missing from "
        "[tool.pytest.ini_options] pythonpath; without them a worktree test run imports "
        "the editable install's copy from another checkout (TASK-13380)"
    )


@pytest.mark.parametrize(("package", "src"), _first_party_packages(), ids=lambda v: getattr(v, "name", v))
def test_first_party_packages_resolve_inside_this_checkout(package: str, src: Path) -> None:
    """Live check: in a worktree this is what fails if the pinning regresses."""
    module = importlib.import_module(package)
    origin = Path(module.__file__).resolve()

    assert origin.is_relative_to(src.resolve()), (
        f"{package} imported from {origin}, not from this checkout's {src}"
    )
