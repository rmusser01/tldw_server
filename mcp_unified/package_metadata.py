"""Release-readiness metadata for the in-repo MCP Unified package boundary."""

from __future__ import annotations

from types import MappingProxyType
from typing import Final

PACKAGE_NAME: Final = "mcp-unified"
PACKAGE_IMPORT_NAME: Final = "mcp_unified"
PACKAGE_STATUS: Final = "internal-experimental"
PUBLISHING_STATUS: Final = "not-published"
LICENSE_EXPRESSION: Final = "GPL-3.0-only"
SOURCE_DISTRIBUTION: Final = "tldw-server"
DEPENDENCY_VERSION_POLICY: Final = "names-only"

CORE_DEPENDENCIES: Final = (
    "pydantic",
    "loguru",
    "pyyaml",
)
FASTAPI_DEPENDENCIES: Final = (
    "fastapi",
    "starlette",
)
SQLITE_DEPENDENCIES: Final = ("sqlalchemy",)
FEDERATION_DEPENDENCIES: Final = CORE_DEPENDENCIES
GATEWAY_DEPENDENCIES: Final = (
    *CORE_DEPENDENCIES,
    *FASTAPI_DEPENDENCIES,
    *SQLITE_DEPENDENCIES,
    "uvicorn",
)
DEV_DEPENDENCIES: Final = (
    "pytest",
    "pytest-asyncio",
    "bandit",
)

OPTIONAL_EXTRAS: Final = MappingProxyType(
    {
        "core": CORE_DEPENDENCIES,
        "fastapi": FASTAPI_DEPENDENCIES,
        "sqlite": SQLITE_DEPENDENCIES,
        "federation": FEDERATION_DEPENDENCIES,
        "gateway": GATEWAY_DEPENDENCIES,
        "dev": DEV_DEPENDENCIES,
    }
)


def package_metadata_summary() -> dict[str, object]:
    """Return JSON-safe MCP Unified package release metadata."""

    return {
        "ok": True,
        "package_name": PACKAGE_NAME,
        "package_import_name": PACKAGE_IMPORT_NAME,
        "package_status": PACKAGE_STATUS,
        "publishing_status": PUBLISHING_STATUS,
        "license_expression": LICENSE_EXPRESSION,
        "source_distribution": SOURCE_DISTRIBUTION,
        "dependency_version_policy": DEPENDENCY_VERSION_POLICY,
        "optional_extras": {
            extra: list(dependencies)
            for extra, dependencies in OPTIONAL_EXTRAS.items()
        },
    }
