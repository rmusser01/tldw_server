from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from .mcp_module import DocsMCPToolProvider
from .models import AccessScope
from .settings import DocsSettings


class StandaloneDocsProfile(str, Enum):
    LOCKED_DOWN = "locked_down"
    LOCAL_FIRST = "local_first"
    ONLINE_CAPABLE = "online_capable"


@dataclass(frozen=True)
class StandaloneDocsMount:
    module_id: str
    name: str
    settings: DocsSettings
    provider: DocsMCPToolProvider

    def tool_definitions(self) -> list[dict[str, Any]]:
        return self.provider.tool_definitions()

    def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None,
        *,
        scope: AccessScope | None = None,
    ) -> Any:
        return self.provider.execute(tool_name, arguments or {}, scope=scope or self.settings.default_scope)


def standalone_docs_settings_for_profile(
    profile: StandaloneDocsProfile | str = StandaloneDocsProfile.LOCKED_DOWN,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> DocsSettings:
    profile_value = StandaloneDocsProfile(profile)
    values: dict[str, Any] = {
        "db_path": "Databases/mcp_docs.db",
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
    if isinstance(settings, DocsSettings):
        resolved_settings = settings
    else:
        resolved_settings = standalone_docs_settings_for_profile(profile, overrides=settings)
    provider = DocsMCPToolProvider(settings=resolved_settings)
    return StandaloneDocsMount(module_id=module_id, name=name, settings=resolved_settings, provider=provider)
