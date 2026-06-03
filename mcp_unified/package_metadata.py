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

CORE_DEPENDENCIES: Final = (
    "pydantic>=2.0.0",
    "loguru>=0.7.0",
    "PyYAML>=6.0.0",
)
FASTAPI_DEPENDENCIES: Final = (
    "fastapi>=0.104.0",
    "starlette>=0.27.0",
)
SQLITE_DEPENDENCIES: Final = ("SQLAlchemy>=2.0.29",)
FEDERATION_DEPENDENCIES: Final = CORE_DEPENDENCIES
GATEWAY_DEPENDENCIES: Final = (
    *CORE_DEPENDENCIES,
    *FASTAPI_DEPENDENCIES,
    *SQLITE_DEPENDENCIES,
    "uvicorn>=0.24.0",
)
DEV_DEPENDENCIES: Final = (
    "pytest>=8.0.0",
    "pytest-asyncio>=1.0.0",
    "bandit>=1.7.0",
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
        "optional_extras": {
            extra: list(dependencies)
            for extra, dependencies in OPTIONAL_EXTRAS.items()
        },
    }
