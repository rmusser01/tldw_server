"""Runtime and package-boundary checks for the standalone MCP Unified package."""

from __future__ import annotations

import ast
import configparser
import importlib
import importlib.util
import json
import os
import re
import runpy
import shutil
import subprocess
import sys
import tarfile
import zipfile
from datetime import datetime
from email.message import Message
from email.parser import Parser
from importlib import metadata as importlib_metadata
from pathlib import Path

import pytest
import yaml
from packaging.requirements import Requirement
from pydantic import ValidationError

pytestmark = pytest.mark.unit

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility.
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[5]
ROOT_PYPROJECT = REPO_ROOT / "pyproject.toml"
STANDALONE_PROJECT_ROOT = REPO_ROOT / "apps" / "mcp-unified"
_ARTIFACT_UTILS = runpy.run_path(
    str(Path(__file__).with_name("mcp_unified_artifact_test_utils.py"))
)
_build_standalone_distributions = _ARTIFACT_UTILS["build_standalone_distributions"]
STANDALONE_SRC_ROOT = STANDALONE_PROJECT_ROOT / "src"
PACKAGE_ROOT = STANDALONE_SRC_ROOT / "mcp_unified"
STANDALONE_PYPROJECT = STANDALONE_PROJECT_ROOT / "pyproject.toml"
PY_TYPED_MARKER = PACKAGE_ROOT / "py.typed"
PACKAGE_README = STANDALONE_PROJECT_ROOT / "README.md"
PACKAGE_USER_GUIDE = STANDALONE_PROJECT_ROOT / "USER_GUIDE.md"
PACKAGE_LICENSE = STANDALONE_PROJECT_ROOT / "LICENSE"
PACKAGE_RESOURCE_README = PACKAGE_ROOT / "README.md"
PACKAGE_RESOURCE_USER_GUIDE = PACKAGE_ROOT / "USER_GUIDE.md"

if str(STANDALONE_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(STANDALONE_SRC_ROOT))

import mcp_unified

REQUIRES_DIST_NAME_PATTERN = re.compile(r"^\s*([A-Za-z0-9_.-]+)")
REQUIRES_DIST_EXTRA_PATTERN = re.compile(r"extra\s*==\s*['\"]([^'\"]+)['\"]")


def test_mcp_unified_package_project_lives_under_apps() -> None:
    """The standalone project must live under apps/mcp-unified."""

    assert STANDALONE_PROJECT_ROOT.is_dir()  # nosec B101
    assert STANDALONE_PYPROJECT.is_file()  # nosec B101
    assert PACKAGE_ROOT.is_dir()  # nosec B101
    assert not (REPO_ROOT / "mcp_unified").exists()  # nosec B101


def test_pinned_protocol_schemas_are_checked_out_with_lf_endings() -> None:
    """Pinned byte-for-byte schema fixtures must not receive Windows CRLF conversion."""

    fixture_root = Path(__file__).with_name("fixtures") / "mcp_protocol"
    schema_paths = sorted(fixture_root.glob("*/schema.json"))
    relative_paths = [path.relative_to(REPO_ROOT).as_posix() for path in schema_paths]
    result = subprocess.run(
        ["git", "check-attr", "text", "eol", "--", *relative_paths],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    resolved = {
        (path, attribute): value
        for line in result.stdout.splitlines()
        for path, attribute, value in [line.split(": ", 2)]
    }

    assert len(schema_paths) == 5  # nosec B101
    assert all(resolved[(path, "text")] == "set" for path in relative_paths)  # nosec B101
    assert all(resolved[(path, "eol")] == "lf" for path in relative_paths)  # nosec B101


def _dependency_package_name(dependency: str) -> str:
    """Return a normalized package name from a dependency declaration."""

    name = dependency.strip()
    for separator in ("[", "<", ">", "=", "!", "~", ";"):
        name = name.split(separator, 1)[0]
    return name.strip().lower().replace("_", "-")


def _load_standalone_pyproject() -> dict[str, object]:
    """Load the package-local MCP Unified pyproject document."""

    assert STANDALONE_PYPROJECT.is_file()
    with STANDALONE_PYPROJECT.open("rb") as pyproject_file:
        return tomllib.load(pyproject_file)


def _load_root_pyproject() -> dict[str, object]:
    """Load the root tldw-server pyproject document."""

    assert ROOT_PYPROJECT.is_file()
    with ROOT_PYPROJECT.open("rb") as pyproject_file:
        return tomllib.load(pyproject_file)


def _dependency_names(dependencies: list[str] | tuple[str, ...]) -> set[str]:
    """Return normalized package names from dependency declarations."""

    return {
        _dependency_package_name(dependency)
        for dependency in dependencies
    }


def _version_release_parts(version: str) -> tuple[int, ...] | None:
    """Return leading numeric release parts from a version string."""

    parts: list[int] = []
    for part in version.split("+", 1)[0].split("-", 1)[0].split("."):
        if not part.isdigit():
            break
        parts.append(int(part))
    if not parts:
        return None
    return tuple(parts)


def _minimum_version(requirement: str) -> tuple[int, ...] | None:
    """Return a simple >= minimum version tuple from a requirement string."""

    if ">=" not in requirement:
        return None
    minimum = requirement.split(">=", 1)[1].split(",", 1)[0].strip()
    return _version_release_parts(minimum)


def _version_satisfies_minimum(
    installed_version: str,
    minimum_version: tuple[int, ...] | None,
) -> bool:
    """Return whether an installed version satisfies a simple minimum tuple."""

    if minimum_version is None:
        return True
    installed_parts = _version_release_parts(installed_version)
    if installed_parts is None:
        return False
    width = max(len(installed_parts), len(minimum_version))
    return installed_parts + (0,) * (width - len(installed_parts)) >= (
        minimum_version + (0,) * (width - len(minimum_version))
    )


def _offline_build_tool_issues() -> list[str]:
    """Return missing or incompatible offline build tool requirements."""
    pyproject = _load_standalone_pyproject()
    build_requires = pyproject["build-system"]["requires"]
    missing: list[str] = []
    incompatible: list[str] = []
    for requirement in build_requires:
        package_name = _dependency_package_name(requirement)
        try:
            installed_version = importlib_metadata.version(package_name)
        except importlib_metadata.PackageNotFoundError:
            missing.append(requirement)
            continue
        minimum = _minimum_version(requirement)
        if not _version_satisfies_minimum(installed_version, minimum):
            incompatible.append(f"{requirement} (found {installed_version})")

    details = []
    if missing:
        details.append(f"missing: {', '.join(missing)}")
    if incompatible:
        details.append(f"incompatible: {', '.join(incompatible)}")
    return details


def _require_offline_build_tools() -> None:
    """Skip the offline build smoke when local build tools are unavailable."""

    details = _offline_build_tool_issues()
    if details:
        pytest.skip(
            "Offline standalone package smoke requires preinstalled build-system "
            "requirements because it uses PIP_NO_INDEX=1 and "
            f"--no-build-isolation; {'; '.join(details)}."
        )


def _assert_subprocess_succeeded(
    result: subprocess.CompletedProcess[str],
    command_label: str,
) -> None:
    """Assert a captured subprocess succeeded while preserving diagnostics."""

    if result.returncode != 0:
        raise AssertionError(
            f"{command_label} failed with exit code {result.returncode}:\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )


def _subprocess_env(
    extra_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Return a subprocess environment without injecting checkout source paths."""

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    return env


def _subprocess_env_with_standalone_src(
    extra_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Return subprocess env with relocated standalone src on PYTHONPATH."""

    env = _subprocess_env(extra_env)
    pythonpath_entries = [str(STANDALONE_SRC_ROOT)]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    return env


def _read_wheel_metadata(wheel: Path) -> Message:
    """Read the wheel distribution metadata."""

    with zipfile.ZipFile(wheel) as archive:
        metadata_members = [
            name
            for name in archive.namelist()
            if name.endswith(".dist-info/METADATA")
        ]
        assert len(metadata_members) == 1  # nosec B101
        raw_metadata = archive.read(metadata_members[0]).decode("utf-8")

    return Parser().parsestr(raw_metadata)


def _read_sdist_metadata(sdist: Path) -> Message:
    """Read the source distribution PKG-INFO metadata."""

    with tarfile.open(sdist, "r:gz") as archive:
        metadata_members = [
            member
            for member in archive.getmembers()
            if member.name.endswith("/PKG-INFO")
            and len(Path(member.name).parts) == 2
        ]
        assert len(metadata_members) == 1  # nosec B101
        extracted = archive.extractfile(metadata_members[0])
        assert extracted is not None  # nosec B101
        raw_metadata = extracted.read().decode("utf-8")

    return Parser().parsestr(raw_metadata)


def _base_requirement(metadata: Message, distribution_name: str) -> Requirement:
    """Return one unmarked base requirement from distribution metadata."""

    normalized_name = distribution_name.lower().replace("_", "-")
    matches = [
        requirement
        for value in metadata.get_all("Requires-Dist") or []
        if (requirement := Requirement(value)).name.lower().replace("_", "-")
        == normalized_name
        and requirement.marker is None
    ]
    assert len(matches) == 1  # nosec B101
    return matches[0]


def _read_wheel_entry_points(wheel: Path) -> configparser.ConfigParser:
    """Read wheel entry points as a ConfigParser document."""

    parser = configparser.ConfigParser()
    with zipfile.ZipFile(wheel) as archive:
        entry_point_members = [
            name
            for name in archive.namelist()
            if name.endswith(".dist-info/entry_points.txt")
        ]
        assert len(entry_point_members) == 1  # nosec B101
        parser.read_string(archive.read(entry_point_members[0]).decode("utf-8"))
    return parser


def _read_wheel_members(wheel: Path) -> set[str]:
    """Return normalized member names from a standalone wheel."""

    with zipfile.ZipFile(wheel) as archive:
        return set(archive.namelist())


def _wheel_declared_dependency_names(distribution_metadata: Message) -> set[str]:
    """Return normalized dependency names declared by wheel metadata."""

    dependencies = distribution_metadata.get_all("Requires-Dist") or []
    names: set[str] = set()
    for dependency in dependencies:
        match = REQUIRES_DIST_NAME_PATTERN.match(dependency)
        assert match is not None  # nosec B101
        names.add(match.group(1).lower().replace("_", "-"))
    return names


def _wheel_extra_dependency_names(
    distribution_metadata: Message,
) -> dict[str, set[str]]:
    """Return normalized dependency names grouped by wheel extra marker."""

    dependencies = distribution_metadata.get_all("Requires-Dist") or []
    grouped: dict[str, set[str]] = {}
    for dependency in dependencies:
        extra_match = REQUIRES_DIST_EXTRA_PATTERN.search(dependency)
        if extra_match is None:
            continue
        name_match = REQUIRES_DIST_NAME_PATTERN.match(dependency)
        assert name_match is not None  # nosec B101
        grouped.setdefault(extra_match.group(1), set()).add(
            name_match.group(1).lower().replace("_", "-")
        )
    return grouped


def _read_sdist_members(sdist: Path) -> set[str]:
    """Return archive member names from a standalone source distribution."""

    with tarfile.open(sdist, "r:gz") as archive:
        return {member.name for member in archive.getmembers()}


def _sdist_project_members(sdist_members: set[str]) -> set[str]:
    """Return sdist members relative to the distribution project root."""

    members = {
        member.removeprefix("./").strip("/")
        for member in sdist_members
        if member.removeprefix("./").strip("/")
    }
    if "pyproject.toml" in members:
        return members

    distribution_roots = {
        member.split("/", 1)[0]
        for member in members
    }
    assert len(distribution_roots) == 1, sorted(distribution_roots)  # nosec B101
    distribution_root = next(iter(distribution_roots))
    return {
        member.removeprefix(f"{distribution_root}/")
        for member in members
        if member != distribution_root
    }


@pytest.fixture(scope="module")
def standalone_distributions(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path]:
    """Build standalone artifacts once for distribution metadata tests."""

    return _build_standalone_distributions(
        tmp_path_factory.mktemp("mcp_unified_distribution")
    )


def test_subprocess_failure_assertion_includes_captured_output() -> None:
    """Subprocess smoke failures should expose captured stdout and stderr."""

    result = subprocess.CompletedProcess(
        args=["python", "-m", "pip", "wheel"],
        returncode=2,
        stdout="build stdout",
        stderr="build stderr",
    )

    with pytest.raises(AssertionError) as exc_info:
        _assert_subprocess_succeeded(result, "pip wheel")

    message = str(exc_info.value)
    assert "pip wheel failed with exit code 2" in message
    assert "STDOUT:\nbuild stdout" in message
    assert "STDERR:\nbuild stderr" in message


def test_build_subprocess_env_does_not_inject_checkout_src(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Build subprocesses should not see checkout src via PYTHONPATH."""

    monkeypatch.setenv("PYTHONPATH", "existing-pythonpath")

    env = _subprocess_env({"PIP_NO_INDEX": "1"})

    assert env["PYTHONPATH"] == "existing-pythonpath"  # nosec B101
    assert str(STANDALONE_SRC_ROOT) not in env["PYTHONPATH"]  # nosec B101
    assert env["PIP_NO_INDEX"] == "1"  # nosec B101


def test_runtime_subprocess_env_injects_standalone_src(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime subprocesses should get the relocated standalone src path."""

    monkeypatch.setenv("PYTHONPATH", "existing-pythonpath")

    env = _subprocess_env_with_standalone_src()

    pythonpath_entries = env["PYTHONPATH"].split(os.pathsep)
    assert pythonpath_entries == [str(STANDALONE_SRC_ROOT), "existing-pythonpath"]  # nosec B101


def _tldw_imports_for(path: Path) -> list[str]:
    """Return host-package imports found in a Python source file."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(
                alias.name
                for alias in node.names
                if alias.name == "tldw_Server_API"
                or alias.name.startswith("tldw_Server_API.")
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "tldw_Server_API" or node.module.startswith("tldw_Server_API."):
                imports.append(node.module)
    return imports


def test_runtime_package_boundary_has_no_tldw_server_imports() -> None:
    """Standalone package modules must not import the host tldw server package."""

    assert PACKAGE_ROOT.exists()
    offenders: dict[str, list[str]] = {}
    for path in PACKAGE_ROOT.rglob("*.py"):
        imports = _tldw_imports_for(path)
        if imports:
            offenders[str(path)] = imports
    assert offenders == {}


def test_mcp_unified_package_metadata_declares_release_gate() -> None:
    """Release-gate package metadata should advertise the published experimental status."""

    metadata = importlib.import_module("mcp_unified.package_metadata")

    assert metadata.PACKAGE_NAME == "mcp-unified"
    assert metadata.PACKAGE_STATUS == "public-alpha"
    assert metadata.PUBLISHING_STATUS == "published"
    assert metadata.LICENSE_EXPRESSION == "GPL-3.0-only"
    assert "jsonschema" in metadata.PROJECT_DEPENDENCIES

    extras = metadata.OPTIONAL_EXTRAS
    assert set(extras) == {  # nosec B101
        "core",
        "fastapi",
        "sqlite",
        "docs-web",
        "federation",
        "gateway",
        "dev",
    }
    assert all(
        isinstance(dependency, str)
        for values in extras.values()
        for dependency in values
    )
    assert all(
        isinstance(dependency, str)
        for dependency in metadata.PROJECT_DEPENDENCIES
    )
    assert all(
        dependency == _dependency_package_name(dependency)
        for dependency in metadata.PROJECT_DEPENDENCIES
    )
    assert set(metadata.CORE_DEPENDENCIES).issubset(metadata.PROJECT_DEPENDENCIES)
    assert set(metadata.NETWORK_DEPENDENCIES).issubset(
        metadata.PROJECT_DEPENDENCIES
    )
    assert set(metadata.NETWORK_DEPENDENCIES).issubset(
        metadata.GATEWAY_DEPENDENCIES
    )
    assert all(
        dependency == _dependency_package_name(dependency)
        for values in extras.values()
        for dependency in values
    )

    forbidden_dependency_names = {
        "chromadb",
        "faster-whisper",
        "torch",
        "yt-dlp",
        "next",
        "rag",
        "tts",
        "stt",
    }
    dependency_names = {
        _dependency_package_name(dependency)
        for values in extras.values()
        for dependency in values
    }
    assert forbidden_dependency_names.isdisjoint(dependency_names)

    summary = metadata.package_metadata_summary()
    assert summary["ok"] is True
    assert summary["package_name"] == metadata.PACKAGE_NAME
    assert summary["dependency_version_policy"] == metadata.DEPENDENCY_VERSION_POLICY
    assert summary["base_dependencies"] == list(metadata.PROJECT_DEPENDENCIES)
    assert summary["optional_extras"] == {
        key: list(value)
        for key, value in extras.items()
    }


def test_mcp_unified_publish_metadata_is_ready_for_public_alpha() -> None:
    """Standalone package metadata should be published but still experimental."""

    metadata = importlib.import_module("mcp_unified.package_metadata")
    pyproject = _load_standalone_pyproject()
    project = pyproject["project"]
    setuptools_config = pyproject["tool"]["setuptools"]

    expected_urls = {
        "Homepage": "https://tldwproject.com",
        "Repository": "https://github.com/rmusser01/tldw_server",
        "Issues": "https://github.com/rmusser01/tldw_server/issues",
        "Source Package": "https://github.com/rmusser01/tldw_server/tree/dev/apps/mcp-unified",
        "User Guide": "https://github.com/rmusser01/tldw_server/blob/dev/apps/mcp-unified/USER_GUIDE.md",
    }
    expected_classifiers = {
        "Development Status :: 3 - Alpha",
        "Framework :: FastAPI",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Software Development :: Libraries :: Python Modules",
    }

    assert project["authors"] == list(metadata.PACKAGE_AUTHORS)  # nosec B101
    assert project["maintainers"] == list(metadata.PACKAGE_MAINTAINERS)  # nosec B101
    assert project["keywords"] == list(metadata.PACKAGE_KEYWORDS)  # nosec B101
    assert set(project["classifiers"]) >= expected_classifiers  # nosec B101
    assert project["urls"] == expected_urls  # nosec B101
    assert setuptools_config["license-files"] == ["LICENSE"]  # nosec B101
    assert expected_urls == metadata.PACKAGE_URLS  # nosec B101
    assert metadata.LICENSE_FILES == ("LICENSE",)  # nosec B101

    summary = metadata.package_metadata_summary()
    assert summary["publishing_status"] == "published"  # nosec B101
    assert summary["package_status"] == "public-alpha"  # nosec B101
    assert summary["authors"] == list(metadata.PACKAGE_AUTHORS)  # nosec B101
    assert summary["maintainers"] == list(metadata.PACKAGE_MAINTAINERS)  # nosec B101
    assert summary["keywords"] == list(metadata.PACKAGE_KEYWORDS)  # nosec B101
    assert summary["classifiers"] == list(metadata.PACKAGE_CLASSIFIERS)  # nosec B101
    assert summary["urls"] == expected_urls  # nosec B101
    assert summary["license_files"] == ["LICENSE"]  # nosec B101


def test_mcp_unified_package_license_file_is_local_to_project() -> None:
    """Standalone artifacts should include a package-local license file."""

    root_license = REPO_ROOT / "LICENSE"

    assert root_license.is_file()  # nosec B101
    assert PACKAGE_LICENSE.is_file()  # nosec B101
    assert PACKAGE_LICENSE.read_text(encoding="utf-8") == root_license.read_text(encoding="utf-8")  # nosec B101


def test_mcp_unified_package_declares_pep561_typed_marker() -> None:
    """The standalone package source must advertise typed-package support."""

    assert PY_TYPED_MARKER.is_file()  # nosec B101
    assert PY_TYPED_MARKER.read_text(encoding="utf-8").strip() == ""  # nosec B101


def test_mcp_unified_package_docs_are_local_to_package_boundary() -> None:
    """Package-local docs must accompany the standalone source boundary."""

    assert PACKAGE_README.is_file()  # nosec B101
    assert PACKAGE_USER_GUIDE.is_file()  # nosec B101
    assert PACKAGE_RESOURCE_README.is_file()  # nosec B101
    assert PACKAGE_RESOURCE_USER_GUIDE.is_file()  # nosec B101

    readme = PACKAGE_README.read_text(encoding="utf-8")
    user_guide = PACKAGE_USER_GUIDE.read_text(encoding="utf-8")
    resource_readme = PACKAGE_RESOURCE_README.read_text(encoding="utf-8")
    resource_user_guide = PACKAGE_RESOURCE_USER_GUIDE.read_text(encoding="utf-8")

    assert resource_readme == readme  # nosec B101
    assert resource_user_guide == user_guide  # nosec B101

    assert "# MCP Unified" in readme  # nosec B101
    assert "USER_GUIDE.md" in readme  # nosec B101
    assert "internal/experimental" in readme  # nosec B101
    assert "Publishing Readiness" in readme  # nosec B101
    assert "make mcp-unified-rc" in readme  # nosec B101
    assert "make mcp-unified-publish-dry-run" in readme  # nosec B101
    assert "TestPyPI" in readme  # nosec B101
    assert "published on PyPI" in readme  # nosec B101
    assert "not-published" not in readme  # nosec B101
    assert "mcp-unified-gateway package-info" in readme  # nosec B101
    assert "filesystem advisory lock backends" in readme  # nosec B101
    assert "memory backend" in readme  # nosec B101
    assert "optional SQLite backend" in readme  # nosec B101
    assert "same local database file" in readme  # nosec B101
    assert "# MCP Unified User Guide" in user_guide  # nosec B101
    assert "Publishing Readiness" in user_guide  # nosec B101
    assert "published on PyPI" in user_guide  # nosec B101
    assert "python -m pip install \"mcp-unified[gateway]\"" in user_guide  # nosec B101
    assert "make mcp-unified-rc" in user_guide  # nosec B101
    assert "make mcp-unified-publish-dry-run" in user_guide  # nosec B101
    assert "MCP_UNIFIED_ALLOW_PUBLISH=1" in user_guide  # nosec B101
    assert "not-published" not in user_guide  # nosec B101
    assert "profiles" in user_guide  # nosec B101
    assert "external servers" in user_guide  # nosec B101
    assert "credential grants" in user_guide  # nosec B101
    assert "configuration snapshots" in user_guide  # nosec B101
    assert "tool-events report --group-by profile" in user_guide  # nosec B101
    assert "tool-events export --format jsonl --since 7d" in user_guide  # nosec B101
    assert "tool-events cleanup --max-age-days 30 --max-events 100000" in user_guide  # nosec B101
    assert "does not capture tool arguments" in user_guide  # nosec B101
    assert "evaluator-labeled task outcomes" in user_guide  # nosec B101
    assert "lock_manager_backend" in user_guide  # nosec B101
    assert "lock_manager_sqlite_path" in user_guide  # nosec B101
    assert "lock_manager_sqlite_timeout_seconds" in user_guide  # nosec B101
    assert "lock_manager_cleanup_interval" in user_guide  # nosec B101
    assert "lock_manager_cleanup_limit" in user_guide  # nosec B101
    assert "not a distributed lock across hosts" in user_guide  # nosec B101
    assert "not a model or agent filesystem tool path" in user_guide  # nosec B101
    assert "Tool-Use Reporting" in readme  # nosec B101


def test_mcp_unified_reporting_imports_do_not_eagerly_load_db_adapters() -> None:
    """Lightweight reporting imports must not require optional DB adapters."""

    import_names = (
        "mcp_unified.tool_use_reporting",
        "mcp_unified.gateway.tool_use_reporting",
        "mcp_unified.gateway.config",
        "mcp_unified.gateway.cli",
    )
    script = (
        "import importlib, sys\n"
        f"for name in {import_names!r}:\n"
        "    importlib.import_module(name)\n"
        "forbidden = sorted(\n"
        "    name for name in sys.modules\n"
        "    if name == 'sqlalchemy'\n"
        "    or name.startswith('sqlalchemy.')\n"
        "    or name == 'sqlite3'\n"
        "    or name == 'mcp_unified.tool_use_reporting.sqlite'\n"
        ")\n"
        "print('\\n'.join(forbidden))\n"
        "raise SystemExit(1 if forbidden else 0)\n"
    )
    result = subprocess.run(  # nosec B603
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=_subprocess_env_with_standalone_src(),
    )

    assert result.returncode == 0, result.stdout + result.stderr  # nosec B101


def test_filesystem_lock_star_import_does_not_eagerly_load_sqlite_backend() -> None:
    """Star imports from core lock exports must not require optional SQLAlchemy."""

    result = subprocess.run(  # nosec B603
        [
            sys.executable,
            "-c",
            (
                "import json, sys; "
                "namespace = {}; "
                "exec('from mcp_unified.filesystem_locks import *', namespace); "
                "blocked = ["
                "name for name in ("
                "'sqlalchemy', 'mcp_unified.filesystem_locks.sqlite'"
                ") if name in sys.modules"
                "]; "
                "print(json.dumps(blocked))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=_subprocess_env_with_standalone_src(),
    )

    assert json.loads(result.stdout) == []  # nosec B101


def test_mcp_unified_standalone_pyproject_matches_release_metadata() -> None:
    """Standalone package metadata must stay aligned with release gate metadata."""

    metadata = importlib.import_module("mcp_unified.package_metadata")
    pyproject = _load_standalone_pyproject()
    project = pyproject["project"]

    assert pyproject["build-system"]["requires"] == ["setuptools>=61.0"]
    assert project["name"] == metadata.PACKAGE_NAME
    assert project["version"] == mcp_unified.__version__
    assert project["readme"] == "README.md"
    assert project["license"]["text"] == metadata.LICENSE_EXPRESSION
    assert project["scripts"]["mcp-unified-gateway"] == "mcp_unified.gateway.cli:main"
    setuptools_config = pyproject["tool"]["setuptools"]
    assert setuptools_config["packages"] == [  # nosec B101
        "mcp_unified",
        "mcp_unified.docs",
        "mcp_unified.docs.acquisition",
        "mcp_unified.docs.importers",
        "mcp_unified.docs.retrieval",
        "mcp_unified.docs.store",
        "mcp_unified.federation",
        "mcp_unified.filesystem_locks",
        "mcp_unified.gateway",
        "mcp_unified.interfaces",
        "mcp_unified.policy_grants",
        "mcp_unified.profiles",
        "mcp_unified.smoke",
        "mcp_unified.storage",
        "mcp_unified.tool_hooks",
        "mcp_unified.tool_use_reporting",
    ]
    assert setuptools_config["package-dir"] == {"": "src"}  # nosec B101
    assert pyproject["tool"]["setuptools"]["package-data"] == {  # nosec B101
        "mcp_unified": ["py.typed", "README.md", "USER_GUIDE.md"],
        "mcp_unified.docs.store": ["schema.sql"],
    }

    assert _dependency_names(project["dependencies"]) == set(metadata.PROJECT_DEPENDENCIES)
    assert "jsonschema>=4.23,<5" in project["dependencies"]
    assert "jsonschema>=4.23,<5" in _load_root_pyproject()["project"]["dependencies"]

    optional_dependencies = project["optional-dependencies"]
    assert set(optional_dependencies) == set(metadata.OPTIONAL_EXTRAS)
    assert {
        extra: _dependency_names(dependencies)
        for extra, dependencies in optional_dependencies.items()
    } == {
        extra: set(dependencies)
        for extra, dependencies in metadata.OPTIONAL_EXTRAS.items()
    }

    forbidden_dependency_names = {
        "chromadb",
        "docling",
        "faster-whisper",
        "gradio",
        "llama-cpp-python",
        "nemo-toolkit",
        "next",
        "qwen",
        "torch",
        "tts",
        "yt-dlp",
    }
    standalone_dependency_names = _dependency_names(project["dependencies"])
    for dependencies in optional_dependencies.values():
        standalone_dependency_names.update(_dependency_names(dependencies))

    assert forbidden_dependency_names.isdisjoint(standalone_dependency_names)


def test_mcp_unified_console_scripts_are_standalone_package_owned() -> None:
    """Root package must not advertise standalone MCP Unified console scripts."""

    standalone_project = _load_standalone_pyproject()["project"]
    root_project = _load_root_pyproject()["project"]

    standalone_scripts = standalone_project["scripts"]
    root_scripts = root_project["scripts"]

    assert standalone_scripts["mcp-unified-gateway"] == "mcp_unified.gateway.cli:main"  # nosec B101
    assert standalone_scripts["mcp-unified-smoke"] == "mcp_unified.smoke.cli:main"  # nosec B101
    assert "mcp-unified-gateway" not in root_scripts  # nosec B101
    assert "mcp-unified-smoke" not in root_scripts  # nosec B101


def test_root_tldw_package_discovers_mcp_unified_runtime_imports() -> None:
    """Root installs must include package-owned modules used by host MCP shims."""

    root_pyproject = _load_root_pyproject()
    find_config = root_pyproject["tool"]["setuptools"]["packages"]["find"]

    assert "apps/mcp-unified/src" in find_config["where"]  # nosec B101
    assert "mcp_unified" in find_config["include"]  # nosec B101
    assert "mcp_unified.*" in find_config["include"]  # nosec B101
    assert "tldw_Server_API" in find_config["include"]  # nosec B101
    assert "tldw_Server_API.*" in find_config["include"]  # nosec B101


def test_host_mcp_import_bootstraps_standalone_src_for_source_checkout() -> None:
    """Source checkout imports must find relocated MCP Unified package modules."""

    script = (
        "import json, sys; "
        "sys.path = [p for p in sys.path if 'apps/mcp-unified/src' not in p]; "
        "import tldw_Server_API.app.core.MCP_unified.modules.base; "
        "import mcp_unified; "
        "print(json.dumps({'file': mcp_unified.__file__, 'paths': ["
        "p for p in sys.path if 'apps/mcp-unified/src' in p"
        "]}))"
    )
    result = subprocess.run(  # nosec B603
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "PYTHONPATH": str(REPO_ROOT),
        },
    )
    _assert_subprocess_succeeded(result, "host MCP source import")

    payload = json.loads(result.stdout)
    assert Path(payload["file"]).is_relative_to(STANDALONE_SRC_ROOT)  # nosec B101
    assert payload["paths"] == [str(STANDALONE_SRC_ROOT)]  # nosec B101


def test_mcp_unified_standalone_distribution_metadata_matches_extras(
    standalone_distributions: tuple[Path, Path],
) -> None:
    """Built standalone artifacts must preserve the package-local extras contract."""

    metadata = importlib.import_module("mcp_unified.package_metadata")
    wheel, _sdist = standalone_distributions
    distribution_metadata = _read_wheel_metadata(wheel)
    entry_points = _read_wheel_entry_points(wheel)

    assert distribution_metadata["Name"] == metadata.PACKAGE_NAME  # nosec B101
    assert distribution_metadata["Version"] == mcp_unified.__version__  # nosec B101
    provides_extra = distribution_metadata.get_all("Provides-Extra") or []
    assert set(provides_extra) == set(metadata.OPTIONAL_EXTRAS)  # nosec B101
    assert _wheel_extra_dependency_names(distribution_metadata) == {
        extra: set(dependencies)
        for extra, dependencies in metadata.OPTIONAL_EXTRAS.items()
    }  # nosec B101
    assert (
        entry_points["console_scripts"]["mcp-unified-gateway"]
        == "mcp_unified.gateway.cli:main"
    )  # nosec B101
    assert (
        entry_points["console_scripts"]["mcp-unified-smoke"]
        == "mcp_unified.smoke.cli:main"
    )  # nosec B101

    forbidden_dependency_names = {
        "chromadb",
        "docling",
        "faster-whisper",
        "gradio",
        "llama-cpp-python",
        "nemo-toolkit",
        "next",
        "qwen",
        "torch",
        "tts",
        "yt-dlp",
    }
    assert forbidden_dependency_names.isdisjoint(
        _wheel_declared_dependency_names(distribution_metadata)
    )  # nosec B101


def test_mcp_unified_artifacts_declare_bounded_jsonschema_base_dependency(
    standalone_distributions: tuple[Path, Path],
) -> None:
    """Wheel and sdist metadata must carry the validator as a base dependency."""

    wheel, sdist = standalone_distributions
    for distribution_metadata in (
        _read_wheel_metadata(wheel),
        _read_sdist_metadata(sdist),
    ):
        requirement = _base_requirement(distribution_metadata, "jsonschema")
        assert str(requirement.specifier) == "<5,>=4.23"  # nosec B101


def test_mcp_unified_standalone_sdist_contains_only_package_boundary(
    standalone_distributions: tuple[Path, Path],
) -> None:
    """Built standalone sdist must not include the host server package tree."""

    _wheel, sdist = standalone_distributions
    members = _sdist_project_members(_read_sdist_members(sdist))

    allowed_project_root_members = {
        "LICENSE",
        "PKG-INFO",
        "README.md",
        "USER_GUIDE.md",
        "pyproject.toml",
        "setup.cfg",
        "src",
    }
    project_root_members = {
        member.split("/", 1)[0]
        for member in members
    }
    assert project_root_members <= allowed_project_root_members  # nosec B101

    forbidden_host_root_members = {
        ".github",
        "Databases",
        "Dockerfiles",
        "Docs",
        "Helper_Scripts",
        "apps",
        "mock_openai_server",
        "models",
        "tldw_Server_API",
    }
    forbidden_members = sorted(
        member
        for member in members
        if member.split("/", 1)[0] in forbidden_host_root_members
    )
    assert forbidden_members == []  # nosec B101

    assert "pyproject.toml" in members  # nosec B101
    assert "src/mcp_unified/__init__.py" in members  # nosec B101
    assert "src/mcp_unified/filesystem_locks/__init__.py" in members  # nosec B101
    assert "src/mcp_unified/gateway/cli.py" in members  # nosec B101


def test_mcp_unified_standalone_artifacts_include_typed_marker(
    standalone_distributions: tuple[Path, Path],
) -> None:
    """Built standalone artifacts must carry the PEP 561 marker."""

    wheel, sdist = standalone_distributions
    wheel_members = _read_wheel_members(wheel)
    sdist_members = _sdist_project_members(_read_sdist_members(sdist))

    assert "mcp_unified/py.typed" in wheel_members  # nosec B101
    assert "src/mcp_unified/py.typed" in sdist_members  # nosec B101


def test_mcp_unified_standalone_artifacts_include_package_docs(
    standalone_distributions: tuple[Path, Path],
) -> None:
    """Built standalone artifacts must carry package-local user docs."""

    wheel, sdist = standalone_distributions
    wheel_members = _read_wheel_members(wheel)
    sdist_members = _sdist_project_members(_read_sdist_members(sdist))

    assert "mcp_unified/README.md" in wheel_members  # nosec B101
    assert "mcp_unified/USER_GUIDE.md" in wheel_members  # nosec B101
    assert "mcp_unified/filesystem_locks/__init__.py" in wheel_members  # nosec B101
    assert "mcp_unified/policy_grants/__init__.py" in wheel_members  # nosec B101
    assert "src/mcp_unified/README.md" in sdist_members  # nosec B101
    assert "src/mcp_unified/USER_GUIDE.md" in sdist_members  # nosec B101
    assert "src/mcp_unified/filesystem_locks/__init__.py" in sdist_members  # nosec B101
    assert "src/mcp_unified/policy_grants/__init__.py" in sdist_members  # nosec B101


def _load_workflow(path: Path) -> dict[str, object]:
    """Load a GitHub Actions workflow document."""

    assert path.is_file()  # nosec B101
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(workflow, dict)  # nosec B101
    return workflow


def _workflow_triggers(workflow: dict[str, object]) -> dict[str, object]:
    """Return GitHub workflow triggers with YAML boolean-key fallback."""

    triggers = workflow.get("on") or workflow.get(True)
    assert isinstance(triggers, dict)  # nosec B101
    return triggers


def _workflow_run_blocks(workflow: dict[str, object]) -> list[str]:
    """Return shell run blocks from every workflow job step."""

    return [
        str(step.get("run", ""))
        for job in workflow["jobs"].values()
        for step in job["steps"]
    ]


def _workflow_trigger_paths(workflow: dict[str, object]) -> dict[str, list[str]]:
    """Return GitHub workflow trigger paths with YAML boolean-key fallback."""

    triggers = _workflow_triggers(workflow)
    return {
        trigger_name: trigger_config.get("paths", [])
        for trigger_name, trigger_config in triggers.items()
        if isinstance(trigger_config, dict)
    }


def _contains_editable_install(run_block: str, package_path: str) -> bool:
    """Return whether a shell run block editable-installs a package path."""

    editable_pattern = re.compile(
        rf"(?:^|\s)(?:-e|--editable)(?:\s+|=)[\"']?{re.escape(package_path)}"
    )
    return bool(editable_pattern.search(run_block))


def _make_target_commands(makefile: str, target_name: str) -> list[str]:
    """Return tab-indented command lines for one Makefile target."""

    match = re.search(
        rf"(?ms)^{re.escape(target_name)}:\n"
        r"(?P<body>(?:\t[^\n]*\n)+)",
        makefile,
    )
    assert match is not None, f"missing Makefile target: {target_name}"  # nosec B101
    return [
        line.removeprefix("\t").strip()
        for line in match.group("body").splitlines()
        if line.startswith("\t") and line.strip()
    ]


def test_mcp_unified_rc_workflow_uses_private_permissions() -> None:
    """Internal RC workflow must stay private and source-scoped."""

    workflow_path = REPO_ROOT / ".github" / "workflows" / "mcp-unified-rc.yml"
    workflow = _load_workflow(workflow_path)
    serialized_workflow = yaml.safe_dump(workflow, sort_keys=True)
    run_blocks = _workflow_run_blocks(workflow)
    install_runs = "\n".join(run_blocks)
    trigger_paths = _workflow_trigger_paths(workflow)
    steps = workflow["jobs"]["internal-rc"]["steps"]
    checkout_step = next(step for step in steps if step.get("name") == "Checkout")
    setup_python_step = next(step for step in steps if step.get("name") == "Setup Python")
    upload_step = next(
        step for step in steps if step.get("name") == "Upload MCP Unified RC artifacts"
    )

    assert workflow["permissions"] == {"contents": "read"}  # nosec B101
    assert "id-token" not in workflow["permissions"]  # nosec B101
    assert checkout_step["uses"] == "actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5"  # nosec B101
    assert checkout_step["with"]["persist-credentials"] is False  # nosec B101
    assert setup_python_step["uses"] == "actions/setup-python@a309ff8b426b58ec0e2a45f0f869d46889d02405"  # nosec B101
    assert upload_step["if"] == "always()"  # nosec B101
    assert upload_step["uses"] == "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a"  # nosec B101
    assert "apps/mcp-unified" in serialized_workflow  # nosec B101
    assert "Makefile" in trigger_paths["pull_request"]  # nosec B101
    assert "make mcp-unified-rc" in serialized_workflow  # nosec B101
    assert '"pydantic>=2.0.0"' in install_runs  # nosec B101
    assert '"PyYAML>=6.0.0"' in install_runs  # nosec B101
    assert not any(  # nosec B101
        _contains_editable_install(run_block, "apps/mcp-unified")
        for run_block in run_blocks
    )
    assert "pip install -e" not in install_runs  # nosec B101
    assert "pip install --editable" not in install_runs  # nosec B101


def test_mcp_unified_publish_workflow_is_manual_and_gated() -> None:
    """Standalone publish workflow must be manual and explicitly gated."""

    workflow_path = REPO_ROOT / ".github" / "workflows" / "mcp-unified-publish.yml"
    workflow = _load_workflow(workflow_path)
    triggers = _workflow_triggers(workflow)
    serialized_workflow = yaml.safe_dump(workflow, sort_keys=True)
    run_blocks = "\n".join(_workflow_run_blocks(workflow))
    jobs = workflow["jobs"]
    testpypi_job = jobs["publish-testpypi"]
    pypi_job = jobs["publish-pypi"]

    assert set(triggers) == {"workflow_dispatch"}  # nosec B101
    inputs = triggers["workflow_dispatch"]["inputs"]
    assert inputs["target"]["options"] == ["dry-run", "testpypi", "pypi"]  # nosec B101
    assert inputs["target"]["default"] == "dry-run"  # nosec B101
    assert inputs["confirm_publish"]["required"] is False  # nosec B101
    assert workflow["permissions"] == {"contents": "read"}  # nosec B101
    assert "pull_request" not in serialized_workflow  # nosec B101
    assert "push:" not in serialized_workflow  # nosec B101
    assert "make mcp-unified-rc" in run_blocks  # nosec B101
    assert "mcp-unified-publish-dry-run" in run_blocks  # nosec B101
    assert "MCP_UNIFIED_ALLOW_PUBLISH=1" in run_blocks  # nosec B101

    plan_job = jobs["publish-plan"]
    assert plan_job["permissions"] == {"contents": "read"}  # nosec B101
    for job_name, environment_name, permissions in (
        ("publish-testpypi", "testpypi", {"contents": "read"}),
        (
            "publish-pypi",
            "pypi",
            {"actions": "read", "contents": "read", "id-token": "write"},
        ),
    ):
        job = jobs[job_name]
        assert job["needs"] == "publish-plan"  # nosec B101
        assert "inputs.confirm_publish == 'MCP_UNIFIED_PUBLISH'" in job["if"]  # nosec B101
        assert job["permissions"] == permissions  # nosec B101
        assert job["environment"]["name"] == environment_name  # nosec B101
    assert testpypi_job["env"]["TWINE_USERNAME"] == "__token__"  # nosec B101
    assert "MCP_UNIFIED_TESTPYPI_API_TOKEN" in testpypi_job["env"]["TWINE_PASSWORD"]  # nosec B101
    assert "env" not in pypi_job  # nosec B101
    assert any(  # nosec B101
        step.get("uses", "").startswith("pypa/gh-action-pypi-publish@")
        for step in pypi_job["steps"]
    )


def test_mcp_unified_publish_workflow_installs_direct_validator_dependency() -> None:
    """RC and upload build jobs must install the direct schema dependency."""

    workflow_path = REPO_ROOT / ".github" / "workflows" / "mcp-unified-publish.yml"
    workflow = _load_workflow(workflow_path)
    install_blocks = [
        str(step["run"])
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
        if step.get("name") == "Install packaging tools"
    ]

    assert len(install_blocks) == 2  # nosec B101
    assert all('"jsonschema>=4.23,<5"' in block for block in install_blocks)  # nosec B101


def test_mcp_unified_make_targets_do_not_call_root_pypi_check() -> None:
    """Standalone RC targets must not delegate to the root PyPI package check."""

    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    for target_name in (
        "mcp-unified-build",
        "mcp-unified-check",
        "mcp-unified-uat",
        "mcp-unified-rc",
        "mcp-unified-publish-dry-run",
    ):
        assert re.search(rf"^{target_name}:", makefile, flags=re.MULTILINE)  # nosec B101

    assert _make_target_commands(makefile, "mcp-unified-build") == [  # nosec B101
        "$(MCP_UNIFIED_RC) build",
    ]
    assert _make_target_commands(makefile, "mcp-unified-check") == [  # nosec B101
        "$(MCP_UNIFIED_RC) build",
        "$(MCP_UNIFIED_RC) artifact-gate",
        "$(MCP_UNIFIED_RC) install-smoke",
    ]
    assert _make_target_commands(makefile, "mcp-unified-uat") == [  # nosec B101
        "$(MCP_UNIFIED_RC) cli-uat",
        "$(MCP_UNIFIED_RC) smoke-uat",
        "$(MCP_UNIFIED_RC) extras-matrix",
    ]
    assert _make_target_commands(makefile, "mcp-unified-rc") == [  # nosec B101
        "$(MCP_UNIFIED_RC) all",
    ]
    assert _make_target_commands(makefile, "mcp-unified-publish-dry-run") == [  # nosec B101
        "$(MCP_UNIFIED_RC) build",
        "$(MCP_UNIFIED_RC) publish-plan --target testpypi --dry-run",
    ]


def test_root_pypi_package_workflow_is_tldw_server_only() -> None:
    """Root package-check workflow must not carry MCP Unified artifact gates."""

    workflow_path = REPO_ROOT / ".github" / "workflows" / "pypi-package.yml"
    workflow = _load_workflow(workflow_path)
    serialized_workflow = yaml.safe_dump(workflow, sort_keys=True)
    trigger_paths = _workflow_trigger_paths(workflow)
    run_blocks = _workflow_run_blocks(workflow)
    upload_names = [
        step.get("with", {}).get("name")
        for job in workflow["jobs"].values()
        for step in job["steps"]
        if "upload-artifact" in str(step.get("uses", ""))
    ]

    assert workflow["name"] == "tldw-server PyPI Package Check"  # nosec B101
    assert "apps/mcp-unified/**" not in trigger_paths["pull_request"]  # nosec B101
    assert "apps/mcp-unified/**" not in trigger_paths["push"]  # nosec B101
    assert "mcp_unified/**" not in trigger_paths["pull_request"]  # nosec B101
    assert "mcp_unified/**" not in trigger_paths["push"]  # nosec B101
    assert "test_mcp_unified_artifact_gate.py" not in serialized_workflow  # nosec B101
    assert not any("apps/mcp-unified" in run_block for run_block in run_blocks)  # nosec B101
    assert not any("mcp_unified[dev]" in run_block for run_block in run_blocks)  # nosec B101
    assert not any("mcp_unified/" in run_block for run_block in run_blocks)  # nosec B101
    assert "tldw-server-pypi-dist" in upload_names  # nosec B101


def test_root_pypi_publish_workflow_is_labeled_for_tldw_server() -> None:
    """Root publish workflow must stay unambiguous about tldw-server artifacts."""

    workflow_path = REPO_ROOT / ".github" / "workflows" / "publish-pypi.yml"
    workflow = _load_workflow(workflow_path)
    serialized_workflow = yaml.safe_dump(workflow, sort_keys=True)

    assert workflow["name"] == "Publish tldw-server PyPI Package"  # nosec B101
    assert "mcp-unified" not in serialized_workflow  # nosec B101
    assert "tldw-server-pypi-dist" in serialized_workflow  # nosec B101


def test_mcp_unified_publish_workflow_uses_trusted_publishing_for_pypi() -> None:
    """Production MCP Unified publishing must use the configured PyPI OIDC publisher."""

    workflow_path = REPO_ROOT / ".github" / "workflows" / "mcp-unified-publish.yml"
    workflow_text = workflow_path.read_text(encoding="utf-8")
    workflow = _load_workflow(workflow_path)

    testpypi_job = workflow["jobs"]["publish-testpypi"]
    pypi_job = workflow["jobs"]["publish-pypi"]
    pypi_steps = {step["name"]: step for step in pypi_job["steps"]}
    download_step = pypi_steps["Download MCP Unified distributions"]
    publish_step = pypi_steps["Publish MCP Unified to PyPI"]

    assert testpypi_job["env"]["TWINE_USERNAME"] == "__token__"  # nosec B101
    assert "MCP_UNIFIED_TESTPYPI_API_TOKEN" in testpypi_job["env"]["TWINE_PASSWORD"]  # nosec B101
    assert pypi_job["environment"]["name"] == "pypi"  # nosec B101
    assert pypi_job["permissions"] == {  # nosec B101
        "actions": "read",
        "contents": "read",
        "id-token": "write",
    }
    assert [step["name"] for step in pypi_job["steps"]] == [  # nosec B101
        "Download MCP Unified distributions",
        "Publish MCP Unified to PyPI",
    ]
    assert "env" not in pypi_job  # nosec B101
    assert all("run" not in step for step in pypi_job["steps"])  # nosec B101
    assert download_step["uses"] == (  # nosec B101
        "actions/download-artifact@3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c"
    )
    assert download_step["with"] == {  # nosec B101
        "name": "mcp-unified-publish-plan",
        "path": ".artifacts/mcp-unified-rc",
    }
    assert publish_step["uses"] == (  # nosec B101
        "pypa/gh-action-pypi-publish@ed0c53931b1dc9bd32cbe73a98c7f6766f8a527e"
    )
    assert publish_step["with"]["packages-dir"] == ".artifacts/mcp-unified-rc/dist/"  # nosec B101
    assert "MCP_UNIFIED_PYPI_API_TOKEN" not in workflow_text  # nosec B101
    assert "TWINE_PASSWORD: ${{ secrets.MCP_UNIFIED_PYPI_API_TOKEN }}" not in workflow_text  # nosec B101
    assert "pypa/gh-action-pypi-publish@v1.13.0" not in workflow_text  # nosec B101


@pytest.mark.smoke
def test_mcp_unified_standalone_package_installs_without_root_dependencies(
    tmp_path: Path,
) -> None:
    """Install the standalone package into an isolated target without root deps."""

    _load_standalone_pyproject()
    _require_offline_build_tools()
    package_source = tmp_path / "mcp_unified_source"
    shutil.copytree(
        STANDALONE_PROJECT_ROOT,
        package_source,
        ignore=shutil.ignore_patterns(
            "__pycache__",
            "build",
            "*.egg-info",
        ),
    )
    wheel_dir = tmp_path / "dist"
    wheel_dir.mkdir()
    build_env = _subprocess_env(
        {
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INDEX": "1",
        }
    )
    build_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-build-isolation",
            "--no-deps",
            "--wheel-dir",
            str(wheel_dir),
            str(package_source),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=build_env,
    )
    _assert_subprocess_succeeded(build_result, "pip wheel")

    wheels = sorted(wheel_dir.glob("mcp_unified-*.whl"))
    assert wheels, "No mcp_unified wheel found after pip wheel completed."

    install_dir = tmp_path / "install"

    install_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-index",
            "--target",
            str(install_dir),
            str(wheels[-1]),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=build_env,
    )
    _assert_subprocess_succeeded(install_result, "pip install")
    import_result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            (
                "import json, sys; "
                f"sys.path.insert(0, {str(install_dir)!r}); "
                "import mcp_unified; "
                "import mcp_unified.package_metadata as metadata; "
                "blocked = ["
                "name for name in ("
                "'tldw_Server_API', 'chromadb', 'torch', 'faster_whisper', "
                "'yt_dlp', 'fastapi', 'sqlalchemy'"
                ") if name in sys.modules"
                "]; "
                "print(json.dumps({"
                "'version': mcp_unified.__version__, "
                "'package': metadata.PACKAGE_NAME, "
                "'blocked': blocked"
                "}))"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": "",
        },
    )
    _assert_subprocess_succeeded(import_result, "standalone package import")

    assert json.loads(import_result.stdout) == {
        "version": mcp_unified.__version__,
        "package": "mcp-unified",
        "blocked": [],
    }


def test_mcp_unified_core_import_smoke_stays_minimal() -> None:
    """Importing the package metadata must not pull in host or heavy stacks."""

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, sys; "
                "import mcp_unified; "
                "import mcp_unified.package_metadata; "
                "blocked = ["
                "name for name in ("
                "'tldw_Server_API', 'chromadb', 'torch', 'faster_whisper', "
                "'yt_dlp', 'next'"
                ") if name in sys.modules"
                "]; "
                "print(json.dumps(blocked))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=_subprocess_env_with_standalone_src(),
    )

    assert json.loads(result.stdout) == []


@pytest.mark.parametrize(
    ("module_name", "blocked_modules"),
    [
        ("mcp_unified.storage", ("mcp_unified.storage.sqlite", "sqlalchemy")),
        ("mcp_unified.federation", ("mcp_unified.storage.sqlite", "sqlalchemy")),
        ("mcp_unified.tool_use_reporting", ("sqlalchemy",)),
        ("mcp_unified.tool_use_reporting.recorder", ("sqlalchemy",)),
        ("mcp_unified.tool_use_reporting.builders", ("sqlalchemy",)),
        ("mcp_unified.tool_use_reporting.store", ("sqlalchemy",)),
        ("mcp_unified.tool_use_reporting.reporting", ("sqlalchemy",)),
    ],
)
def test_package_imports_do_not_eagerly_load_sqlite_backend(
    module_name: str,
    blocked_modules: tuple[str, ...],
) -> None:
    """Core and federation imports must not require SQLite storage dependencies."""

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib, json, sys; "
                f"importlib.import_module({module_name!r}); "
                f"blocked = [name for name in {blocked_modules!r} if name in sys.modules]; "
                "print(json.dumps(blocked))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=_subprocess_env_with_standalone_src(),
    )

    assert json.loads(result.stdout) == []


def test_host_interface_shims_reexport_package_contracts() -> None:
    """Host interface shims should re-export package-owned contracts."""

    package_policy = importlib.import_module("mcp_unified.interfaces.policy")
    package_runtime = importlib.import_module("mcp_unified.interfaces.runtime")
    package_storage = importlib.import_module("mcp_unified.interfaces.storage")
    host_policy = importlib.import_module(
        "tldw_Server_API.app.core.MCP_unified.interfaces.policy"
    )
    host_runtime = importlib.import_module(
        "tldw_Server_API.app.core.MCP_unified.interfaces.runtime"
    )
    host_storage = importlib.import_module(
        "tldw_Server_API.app.core.MCP_unified.interfaces.storage"
    )
    assert host_policy.ApprovalEvaluator is package_policy.ApprovalEvaluator
    assert host_runtime.MCPRuntimeDependencies is package_runtime.MCPRuntimeDependencies
    assert host_runtime.ModuleRegistry is package_runtime.ModuleRegistry
    assert host_storage.ProfileStore is package_storage.ProfileStore


def test_host_external_config_schema_shim_reexports_package_contracts() -> None:
    """Host external config-schema shim should re-export package contracts."""

    package_schema = importlib.import_module("mcp_unified.federation.config_schema")
    host_schema = importlib.import_module(
        "tldw_Server_API.app.core.MCP_unified.external_servers.config_schema"
    )

    assert host_schema.ExternalAuthMode is package_schema.ExternalAuthMode
    assert host_schema.ExternalMCPServerConfig is package_schema.ExternalMCPServerConfig
    assert host_schema.ExternalServerRegistryConfig is package_schema.ExternalServerRegistryConfig
    assert host_schema.ExternalTransportType is package_schema.ExternalTransportType
    assert host_schema.parse_external_server_registry is package_schema.parse_external_server_registry


def test_host_external_transport_base_reexports_package_contracts() -> None:
    """Host transport base must reuse the package-owned external contracts."""
    package_models = importlib.import_module("mcp_unified.federation.models")
    package_federation = importlib.import_module("mcp_unified.federation")
    host_base = importlib.import_module(
        "tldw_Server_API.app.core.MCP_unified.external_servers.transports.base"
    )

    assert host_base.ExternalToolDefinition is package_models.ExternalToolDefinition
    assert host_base.ExternalToolCallResult is package_models.ExternalToolCallResult
    assert host_base.BrokeredExternalCredential is package_models.BrokeredExternalCredential
    assert package_federation.BrokeredExternalCredential is package_models.BrokeredExternalCredential


def test_brokered_external_credential_copy_returns_caller_owned_data() -> None:
    """Brokered credential copies must not expose mutable source state."""
    package_models = importlib.import_module("mcp_unified.federation.models")
    credential = package_models.BrokeredExternalCredential(
        headers={"Authorization": "Bearer token"},
        env={"TOKEN": "secret"},
        metadata={"nested": {"source": "broker"}},
    )

    copied = credential.copy()
    copied.headers["Authorization"] = "changed"
    copied.env["TOKEN"] = "changed"
    copied.metadata["nested"]["source"] = "changed"

    assert credential.headers == {"Authorization": "Bearer token"}
    assert credential.env == {"TOKEN": "secret"}
    assert credential.metadata == {"nested": {"source": "broker"}}


def test_host_external_manager_reexports_package_virtual_tool_contract() -> None:
    """Host external manager must reuse the package-owned virtual tool contract."""
    package_models = importlib.import_module("mcp_unified.federation.models")
    package_federation = importlib.import_module("mcp_unified.federation")
    host_external = importlib.import_module(
        "tldw_Server_API.app.core.MCP_unified.external_servers"
    )
    host_manager = importlib.import_module(
        "tldw_Server_API.app.core.MCP_unified.external_servers.manager"
    )

    assert host_manager.VirtualExternalTool is package_models.VirtualExternalTool
    assert host_external.VirtualExternalTool is package_models.VirtualExternalTool
    assert package_federation.VirtualExternalTool is package_models.VirtualExternalTool


def test_gateway_external_runtime_exports_are_package_owned() -> None:
    """Gateway runtime exports must resolve without host package imports."""
    package_gateway = importlib.import_module("mcp_unified.gateway")
    package_config = importlib.import_module("mcp_unified.gateway.config")
    package_adapter = importlib.import_module("mcp_unified.gateway.external_runtime_adapter")
    package_lifecycle = importlib.import_module("mcp_unified.gateway.lifecycle")
    package_runtime = importlib.import_module("mcp_unified.gateway.external_runtime")

    assert package_gateway.GatewayExternalRuntimeManager is package_runtime.GatewayExternalRuntimeManager
    assert package_gateway.GatewayExternalRuntimeError is package_runtime.GatewayExternalRuntimeError
    assert (
        package_gateway.ExternalRuntimeGatewayRuntime
        is package_adapter.ExternalRuntimeGatewayRuntime
    )
    assert (
        package_gateway.GatewayExternalRuntimeBootstrapConfig
        is package_config.GatewayExternalRuntimeBootstrapConfig
    )
    assert (
        package_gateway.GatewayExternalRuntimeLifecycleConfig
        is package_lifecycle.GatewayExternalRuntimeLifecycleConfig
    )


def test_gateway_external_runtime_adapter_import_does_not_import_fastapi_transport() -> None:
    """Importing the external runtime adapter must not require FastAPI helpers."""

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import mcp_unified.gateway.external_runtime_adapter; "
                "print('mcp_unified.gateway.fastapi' in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=_subprocess_env_with_standalone_src(),
    )

    assert result.stdout.strip() == "False"


def test_federation_installer_contracts_are_public_exports() -> None:
    """Installer contracts should be importable from the public federation package."""
    package_federation = importlib.import_module("mcp_unified.federation")
    package_installers = importlib.import_module("mcp_unified.federation.installers")

    assert package_federation.ExternalServerInstaller is package_installers.ExternalServerInstaller
    assert package_federation.NullExternalServerInstaller is package_installers.NullExternalServerInstaller


def test_federation_process_policy_contracts_are_public_exports() -> None:
    """Stdio process-policy contracts should be public federation exports."""
    package_federation = importlib.import_module("mcp_unified.federation")
    package_policy = importlib.import_module("mcp_unified.federation.process_policy")

    assert package_federation.StdioProcessPolicy is package_policy.StdioProcessPolicy
    assert (
        package_federation.coerce_stdio_process_policy
        is package_policy.coerce_stdio_process_policy
    )


def test_virtual_external_tool_copy_returns_caller_owned_data() -> None:
    """Virtual tool copies must not expose mutable source state."""
    package_models = importlib.import_module("mcp_unified.federation.models")
    virtual_tool = package_models.VirtualExternalTool(
        virtual_name="ext.docs.search",
        server_id="docs",
        upstream_tool_name="search",
        description="Search docs",
        input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
        metadata={"nested": {"source": "discovery"}},
        is_write=False,
    )

    copied = virtual_tool.copy()
    copied.input_schema["properties"]["q"]["type"] = "number"
    copied.metadata["nested"]["source"] = "changed"

    assert virtual_tool.input_schema == {
        "type": "object",
        "properties": {"q": {"type": "string"}},
    }
    assert virtual_tool.metadata == {"nested": {"source": "discovery"}}


def test_profile_defaults_are_safe_and_preserve_extension_metadata() -> None:
    """Profile defaults should stay safe while preserving extension metadata."""

    from mcp_unified.profiles.models import MCPProfile

    profile = MCPProfile(
        id="architect",
        name="Architect",
        approval_policy=None,
        path_scopes=None,
        external_server_grants=None,
        credential_grants=None,
        policy_document={
            "allowed_tools": None,
            "capabilities": None,
            "resource_constraints": None,
            "policy_extension": {"level": "experimental"},
        },
        metadata={"agent_metadata": {"system_prompt": "review architecture"}},
        profile_extension={"owner": "frontend"},
    )

    assert profile.enabled is True
    assert profile.policy_document.allowed_tools == []
    assert profile.policy_document.capabilities == []
    assert profile.policy_document.resource_constraints == {}
    assert profile.credential_grants == []
    assert profile.external_server_grants == []
    assert profile.metadata["agent_metadata"]["system_prompt"] == "review architecture"
    dumped = profile.model_dump()
    assert dumped["profile_extension"] == {"owner": "frontend"}
    assert dumped["policy_document"]["policy_extension"] == {"level": "experimental"}


def test_profile_rejects_naive_timestamps() -> None:
    """Profile models should reject naive timestamp values."""

    from mcp_unified.profiles.models import MCPProfile

    with pytest.raises(ValidationError):
        MCPProfile(
            id="architect",
            name="Architect",
            created_at=datetime(2026, 5, 27, 5, 0, 0),
        )
