from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path
from typing import Any, Mapping

from .mcp_module import DocsMCPToolProvider
from .models import AccessScope
from .settings import DocsSettings


class StandaloneDocsProfile(str, Enum):
    """Deployment profiles for standalone docs corpus mounting."""

    LOCKED_DOWN = "locked_down"
    LOCAL_FIRST = "local_first"
    ONLINE_CAPABLE = "online_capable"


@dataclass(frozen=True)
class StandaloneDocsMount:
    """A mounted docs provider with its resolved settings and public tool surface."""

    module_id: str
    name: str
    settings: DocsSettings
    provider: DocsMCPToolProvider

    def tool_definitions(self) -> list[dict[str, Any]]:
        """Return the MCP tool definitions exposed by this mount."""

        return self.provider.tool_definitions()

    def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None,
        *,
        scope: AccessScope | None = None,
    ) -> Any:
        """Execute a docs tool in the supplied or default access scope."""

        return self.provider.execute(tool_name, arguments or {}, scope=scope or self.settings.default_scope)


def _default_standalone_db_path() -> Path:
    data_home = os.environ.get("XDG_DATA_HOME")
    base_dir = Path(data_home).expanduser() if data_home else Path.home() / ".local" / "share"
    return base_dir / "tldw_mcp" / "docs.db"


def standalone_docs_settings_for_profile(
    profile: StandaloneDocsProfile | str = StandaloneDocsProfile.LOCKED_DOWN,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> DocsSettings:
    """Build docs settings from a named standalone deployment profile."""

    profile_value = StandaloneDocsProfile(profile)
    values: dict[str, Any] = {
        "db_path": _default_standalone_db_path(),
        "enable_web_acquisition": False,
        "web_source_profile": profile_value.value,
        "allow_arbitrary_public_domains": False,
    }
    if profile_value in {StandaloneDocsProfile.LOCAL_FIRST, StandaloneDocsProfile.ONLINE_CAPABLE}:
        values["enable_web_acquisition"] = True
    if overrides:
        values.update(dict(overrides))
    return DocsSettings.from_mapping(values)


def create_standalone_docs_mount(
    settings: DocsSettings | Mapping[str, Any] | None = None,
    *,
    profile: StandaloneDocsProfile | str = StandaloneDocsProfile.LOCKED_DOWN,
    module_id: str = "docs",
    name: str = "Docs Corpus",
) -> StandaloneDocsMount:
    """Create a runtime-neutral docs MCP mount for the standalone server."""

    if isinstance(settings, DocsSettings):
        resolved_settings = settings
    else:
        resolved_settings = standalone_docs_settings_for_profile(profile, overrides=settings)
    provider = DocsMCPToolProvider(settings=resolved_settings)
    return StandaloneDocsMount(module_id=module_id, name=name, settings=resolved_settings, provider=provider)
