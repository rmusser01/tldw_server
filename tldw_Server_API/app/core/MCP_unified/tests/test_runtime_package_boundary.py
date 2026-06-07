from __future__ import annotations

import ast
import configparser
import importlib
import importlib.util
import json
import os
import re
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

import mcp_unified
import pytest
import yaml
from pydantic import ValidationError

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility.
    import tomli as tomllib

PACKAGE_ROOT = Path(mcp_unified.__file__).resolve().parent
STANDALONE_PYPROJECT = PACKAGE_ROOT / "pyproject.toml"
PY_TYPED_MARKER = PACKAGE_ROOT / "py.typed"
PACKAGE_README = PACKAGE_ROOT / "README.md"
PACKAGE_USER_GUIDE = PACKAGE_ROOT / "USER_GUIDE.md"
REQUIRES_DIST_NAME_PATTERN = re.compile(r"^\s*([A-Za-z0-9_.-]+)")
REQUIRES_DIST_EXTRA_PATTERN = re.compile(r"extra\s*==\s*['\"]([^'\"]+)['\"]")


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


def _assert_artifact_gate_build_tools_available() -> None:
    """Assert the mandatory artifact gate build tools are available."""

    details = _offline_build_tool_issues()
    if importlib.util.find_spec("build") is None:
        details.append("missing: build")
    if details:
        raise AssertionError(
            "Standalone artifact gate requires preinstalled build tools; "
            f"{'; '.join(details)}."
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


def _build_standalone_distributions(tmp_path: Path) -> tuple[Path, Path]:
    """Build standalone MCP Unified wheel and sdist into a temporary directory."""

    _assert_artifact_gate_build_tools_available()

    package_source = tmp_path / "mcp_unified_source"
    shutil.copytree(
        PACKAGE_ROOT,
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
        env={
            **os.environ,
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INDEX": "1",
        },
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
    assert len(wheels) == 1  # nosec B101
    assert len(sdists) == 1  # nosec B101
    return wheels[0], sdists[0]


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
    """Return normalized member names from a standalone source distribution."""

    with tarfile.open(sdist, "r:gz") as archive:
        return {member.name for member in archive.getmembers()}


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


def _tldw_imports_for(path: Path) -> list[str]:
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
    assert PACKAGE_ROOT.exists()
    offenders: dict[str, list[str]] = {}
    for path in PACKAGE_ROOT.rglob("*.py"):
        imports = _tldw_imports_for(path)
        if imports:
            offenders[str(path)] = imports
    assert offenders == {}


def test_mcp_unified_package_metadata_declares_release_gate() -> None:
    metadata = importlib.import_module("mcp_unified.package_metadata")

    assert metadata.PACKAGE_NAME == "mcp-unified"
    assert metadata.PACKAGE_STATUS == "internal-experimental"
    assert metadata.PUBLISHING_STATUS == "not-published"
    assert metadata.LICENSE_EXPRESSION == "GPL-3.0-only"

    extras = metadata.OPTIONAL_EXTRAS
    assert set(extras) == {
        "core",
        "fastapi",
        "sqlite",
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
    assert summary["optional_extras"] == {
        key: list(value)
        for key, value in extras.items()
    }


def test_mcp_unified_package_declares_pep561_typed_marker() -> None:
    """The standalone package source must advertise typed-package support."""

    assert PY_TYPED_MARKER.is_file()  # nosec B101
    assert PY_TYPED_MARKER.read_text(encoding="utf-8").strip() == ""  # nosec B101


def test_mcp_unified_package_docs_are_local_to_package_boundary() -> None:
    """Package-local docs must accompany the standalone source boundary."""

    assert PACKAGE_README.is_file()  # nosec B101
    assert PACKAGE_USER_GUIDE.is_file()  # nosec B101

    readme = PACKAGE_README.read_text(encoding="utf-8")
    user_guide = PACKAGE_USER_GUIDE.read_text(encoding="utf-8")

    assert "# MCP Unified" in readme  # nosec B101
    assert "USER_GUIDE.md" in readme  # nosec B101
    assert "internal/experimental" in readme  # nosec B101
    assert "mcp-unified-gateway package-info" in readme  # nosec B101
    assert "# MCP Unified User Guide" in user_guide  # nosec B101
    assert "profiles" in user_guide  # nosec B101
    assert "external servers" in user_guide  # nosec B101
    assert "credential grants" in user_guide  # nosec B101
    assert "configuration snapshots" in user_guide  # nosec B101
    assert "tool-events report --group-by profile" in user_guide  # nosec B101
    assert "tool-events export --format jsonl --since 7d" in user_guide  # nosec B101
    assert "tool-events cleanup --max-age-days 30 --max-events 100000" in user_guide  # nosec B101
    assert "does not capture tool arguments" in user_guide  # nosec B101
    assert "evaluator-labeled task outcomes" in user_guide  # nosec B101
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
    )

    assert result.returncode == 0, result.stdout + result.stderr  # nosec B101


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
    assert pyproject["tool"]["setuptools"]["package-data"] == {  # nosec B101
        "mcp_unified": ["py.typed", "README.md", "USER_GUIDE.md"],
    }

    assert _dependency_names(project["dependencies"]) == set(metadata.CORE_DEPENDENCIES)

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


def test_mcp_unified_standalone_sdist_contains_only_package_boundary(
    standalone_distributions: tuple[Path, Path],
) -> None:
    """Built standalone sdist must not include the host server package tree."""

    _wheel, sdist = standalone_distributions
    members = _read_sdist_members(sdist)

    assert any(member.endswith("/pyproject.toml") for member in members)  # nosec B101
    assert any(member.endswith("/__init__.py") for member in members)  # nosec B101
    assert any(member.endswith("/gateway/cli.py") for member in members)  # nosec B101
    assert not any("/tldw_Server_API/" in member for member in members)  # nosec B101
    assert not any("/apps/tldw-frontend/" in member for member in members)  # nosec B101


def test_mcp_unified_standalone_artifacts_include_typed_marker(
    standalone_distributions: tuple[Path, Path],
) -> None:
    """Built standalone artifacts must carry the PEP 561 marker."""

    wheel, sdist = standalone_distributions
    wheel_members = _read_wheel_members(wheel)
    sdist_members = _read_sdist_members(sdist)

    assert "mcp_unified/py.typed" in wheel_members  # nosec B101
    assert any(member.endswith("/py.typed") for member in sdist_members)  # nosec B101


def test_mcp_unified_standalone_artifacts_include_package_docs(
    standalone_distributions: tuple[Path, Path],
) -> None:
    """Built standalone artifacts must carry package-local user docs."""

    wheel, sdist = standalone_distributions
    wheel_members = _read_wheel_members(wheel)
    sdist_members = _read_sdist_members(sdist)

    assert "mcp_unified/README.md" in wheel_members  # nosec B101
    assert "mcp_unified/USER_GUIDE.md" in wheel_members  # nosec B101
    assert any(member.endswith("/README.md") for member in sdist_members)  # nosec B101
    assert any(member.endswith("/USER_GUIDE.md") for member in sdist_members)  # nosec B101


def test_pypi_workflow_runs_mcp_unified_standalone_artifact_gate() -> None:
    """PyPI validation workflow must also gate package-local mcp_unified changes."""

    artifact_gate_config = "mcp_unified/pytest-artifact-gate.ini"
    config_path = PACKAGE_ROOT.parent / artifact_gate_config
    assert config_path.is_file()  # nosec B101
    pytest_config = configparser.ConfigParser()
    pytest_config.read(config_path, encoding="utf-8")
    assert "--noconftest" in pytest_config["pytest"]["addopts"]  # nosec B101

    workflow_path = (
        PACKAGE_ROOT.parent / ".github" / "workflows" / "pypi-package.yml"
    )
    assert workflow_path.is_file()  # nosec B101
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    triggers = workflow.get("on") or workflow.get(True)

    assert "mcp_unified/**" in triggers["pull_request"]["paths"]  # nosec B101
    assert "mcp_unified/**" in triggers["push"]["paths"]  # nosec B101

    steps = workflow["jobs"]["build-and-check"]["steps"]
    run_blocks = [step.get("run", "") for step in steps]
    artifact_gate_test_path = ".github/tests/test_mcp_unified_artifact_gate.py"
    artifact_gate_nodeids = (
        "test_mcp_unified_standalone_distribution_metadata_matches_extras",
        "test_mcp_unified_standalone_sdist_contains_only_package_boundary",
        "test_mcp_unified_standalone_artifacts_include_typed_marker",
        "test_mcp_unified_standalone_artifacts_include_package_docs",
    )
    assert any(
        all(f"{artifact_gate_test_path}::{nodeid}" in run_block for nodeid in artifact_gate_nodeids)
        and f"-c {artifact_gate_config}" in run_block
        for run_block in run_blocks
    )  # nosec B101
    assert any(
        run_block.strip() == "make pypi-check"
        for run_block in run_blocks
    )  # nosec B101
    assert any(
        str(step.get("uses", "")).startswith("actions/upload-artifact@")
        and step.get("with", {}).get("path") == "dist/*"
        for step in steps
    )  # nosec B101


@pytest.mark.smoke
def test_mcp_unified_standalone_package_installs_without_root_dependencies(
    tmp_path: Path,
) -> None:
    """Install the standalone package into an isolated target without root deps."""

    _load_standalone_pyproject()
    _require_offline_build_tools()
    package_source = tmp_path / "mcp_unified_source"
    shutil.copytree(
        PACKAGE_ROOT,
        package_source,
        ignore=shutil.ignore_patterns(
            "__pycache__",
            "build",
            "*.egg-info",
        ),
    )
    wheel_dir = tmp_path / "dist"
    wheel_dir.mkdir()
    build_env = {
        **os.environ,
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_INDEX": "1",
    }
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
    )

    assert json.loads(result.stdout) == []


def test_host_interface_shims_reexport_package_contracts() -> None:
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
    from mcp_unified.profiles.models import MCPProfile

    with pytest.raises(ValidationError):
        MCPProfile(
            id="architect",
            name="Architect",
            created_at=datetime(2026, 5, 27, 5, 0, 0),
        )
