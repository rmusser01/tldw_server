"""Release-readiness metadata for the in-repo MCP Unified package boundary."""

from __future__ import annotations

from types import MappingProxyType
from typing import Final

PACKAGE_NAME: Final = "mcp-unified"
PACKAGE_IMPORT_NAME: Final = "mcp_unified"
PACKAGE_STATUS: Final = "internal-experimental"
PUBLISHING_STATUS: Final = "published"
LICENSE_EXPRESSION: Final = "GPL-3.0-only"
SOURCE_DISTRIBUTION: Final = "tldw-server"
DEPENDENCY_VERSION_POLICY: Final = "names-only"
PACKAGE_AUTHORS: Final = (
    {"name": "Robert Musser", "email": "contact@tldwproject.com"},
)
PACKAGE_MAINTAINERS: Final = (
    {"name": "Robert Musser", "email": "contact@tldwproject.com"},
)
PACKAGE_KEYWORDS: Final = (
    "mcp",
    "model-context-protocol",
    "gateway",
    "agent-tools",
    "tool-governance",
)
PACKAGE_CLASSIFIERS: Final = (
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
)
PACKAGE_URLS: Final = {
    "Homepage": "https://tldwproject.com",
    "Repository": "https://github.com/rmusser01/tldw_server",
    "Issues": "https://github.com/rmusser01/tldw_server/issues",
    "Source Package": "https://github.com/rmusser01/tldw_server/tree/dev/apps/mcp-unified",
    "User Guide": "https://github.com/rmusser01/tldw_server/blob/dev/apps/mcp-unified/USER_GUIDE.md",
}
LICENSE_FILES: Final = ("LICENSE",)

CORE_DEPENDENCIES: Final = (
    "pydantic",
    "loguru",
    "pyyaml",
)
NETWORK_DEPENDENCIES: Final = (
    "httpx",
    "websockets",
)
PROJECT_DEPENDENCIES: Final = (
    *CORE_DEPENDENCIES,
    *NETWORK_DEPENDENCIES,
)
FASTAPI_DEPENDENCIES: Final = (
    "fastapi",
    "starlette",
)
SQLITE_DEPENDENCIES: Final = ("sqlalchemy",)
FEDERATION_DEPENDENCIES: Final = CORE_DEPENDENCIES
GATEWAY_DEPENDENCIES: Final = (
    *CORE_DEPENDENCIES,
    *NETWORK_DEPENDENCIES,
    *FASTAPI_DEPENDENCIES,
    *SQLITE_DEPENDENCIES,
    "uvicorn",
)
DEV_DEPENDENCIES: Final = (
    "build",
    "setuptools",
    "tomli",
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
        "authors": list(PACKAGE_AUTHORS),
        "maintainers": list(PACKAGE_MAINTAINERS),
        "keywords": list(PACKAGE_KEYWORDS),
        "classifiers": list(PACKAGE_CLASSIFIERS),
        "urls": dict(PACKAGE_URLS),
        "license_files": list(LICENSE_FILES),
        "base_dependencies": list(PROJECT_DEPENDENCIES),
        "optional_extras": {
            extra: list(dependencies)
            for extra, dependencies in OPTIONAL_EXTRAS.items()
        },
    }
