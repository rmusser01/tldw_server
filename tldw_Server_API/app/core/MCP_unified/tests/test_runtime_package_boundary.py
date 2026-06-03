from __future__ import annotations

import ast
import importlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import mcp_unified
import pytest
from pydantic import ValidationError

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility.
    import tomli as tomllib

PACKAGE_ROOT = Path(mcp_unified.__file__).resolve().parent
STANDALONE_PYPROJECT = PACKAGE_ROOT / "pyproject.toml"


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


def test_mcp_unified_standalone_pyproject_matches_release_metadata() -> None:
    """Standalone package metadata must stay aligned with release gate metadata."""

    metadata = importlib.import_module("mcp_unified.package_metadata")
    pyproject = _load_standalone_pyproject()
    project = pyproject["project"]

    assert project["name"] == metadata.PACKAGE_NAME
    assert project["version"] == mcp_unified.__version__
    assert project["license"]["text"] == metadata.LICENSE_EXPRESSION
    assert project["scripts"]["mcp-unified-gateway"] == "mcp_unified.gateway.cli:main"

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


@pytest.mark.smoke
def test_mcp_unified_standalone_package_installs_without_root_dependencies(
    tmp_path: Path,
) -> None:
    """Install the standalone package into an isolated target without root deps."""

    _load_standalone_pyproject()
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
        check=True,
        capture_output=True,
        text=True,
        env=build_env,
    )

    wheels = sorted(wheel_dir.glob("mcp_unified-*.whl"))
    assert wheels, build_result.stdout + build_result.stderr

    install_dir = tmp_path / "install"

    subprocess.run(
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
        check=True,
        capture_output=True,
        text=True,
        env=build_env,
    )
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
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": "",
        },
    )

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
