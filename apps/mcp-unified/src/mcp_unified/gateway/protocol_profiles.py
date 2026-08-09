"""Immutable MCP revision profiles for strict gateway protocol dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

CURRENT_PROTOCOL_VERSION = "2026-07-28"
PREFERRED_LEGACY_PROTOCOL_VERSION = "2025-11-25"
SUPPORTED_PROTOCOL_VERSIONS = (
    CURRENT_PROTOCOL_VERSION,
    PREFERRED_LEGACY_PROTOCOL_VERSION,
    "2025-06-18",
    "2025-03-26",
    "2024-11-05",
)
SUPPORTED_MODERN_PROTOCOL_VERSIONS = (CURRENT_PROTOCOL_VERSION,)
SUPPORTED_LEGACY_PROTOCOL_VERSIONS = SUPPORTED_PROTOCOL_VERSIONS[1:]


@dataclass(frozen=True, slots=True)
class GatewayProtocolProfile:
    """Version-specific behavior used by the strict MCP protocol layer."""

    version: str
    era: Literal["modern", "legacy"]
    requires_initialize: bool
    accepts_batches: bool
    requires_result_type: bool
    cache_hints: bool
    supports_titles: bool
    supports_icons: bool
    supports_resource_links: bool
    structured_content_mode: Literal["any", "object", "none"]
    missing_resource_code: int
    schema_dialect: str


PROTOCOL_PROFILES = MappingProxyType(
    {
        "2026-07-28": GatewayProtocolProfile(
            version="2026-07-28",
            era="modern",
            requires_initialize=False,
            accepts_batches=False,
            requires_result_type=True,
            cache_hints=True,
            supports_titles=True,
            supports_icons=True,
            supports_resource_links=True,
            structured_content_mode="any",
            missing_resource_code=-32602,
            schema_dialect="https://json-schema.org/draft/2020-12/schema",
        ),
        "2025-11-25": GatewayProtocolProfile(
            version="2025-11-25",
            era="legacy",
            requires_initialize=True,
            accepts_batches=False,
            requires_result_type=False,
            cache_hints=False,
            supports_titles=True,
            supports_icons=True,
            supports_resource_links=True,
            structured_content_mode="object",
            missing_resource_code=-32002,
            schema_dialect="https://json-schema.org/draft/2020-12/schema",
        ),
        "2025-06-18": GatewayProtocolProfile(
            version="2025-06-18",
            era="legacy",
            requires_initialize=True,
            accepts_batches=False,
            requires_result_type=False,
            cache_hints=False,
            supports_titles=True,
            supports_icons=False,
            supports_resource_links=True,
            structured_content_mode="object",
            missing_resource_code=-32002,
            schema_dialect="http://json-schema.org/draft-07/schema#",
        ),
        "2025-03-26": GatewayProtocolProfile(
            version="2025-03-26",
            era="legacy",
            requires_initialize=True,
            accepts_batches=True,
            requires_result_type=False,
            cache_hints=False,
            supports_titles=False,
            supports_icons=False,
            supports_resource_links=False,
            structured_content_mode="none",
            missing_resource_code=-32002,
            schema_dialect="http://json-schema.org/draft-07/schema#",
        ),
        "2024-11-05": GatewayProtocolProfile(
            version="2024-11-05",
            era="legacy",
            requires_initialize=True,
            accepts_batches=False,
            requires_result_type=False,
            cache_hints=False,
            supports_titles=False,
            supports_icons=False,
            supports_resource_links=False,
            structured_content_mode="none",
            missing_resource_code=-32002,
            schema_dialect="http://json-schema.org/draft-07/schema#",
        ),
    }
)


__all__ = [
    "CURRENT_PROTOCOL_VERSION",
    "PREFERRED_LEGACY_PROTOCOL_VERSION",
    "PROTOCOL_PROFILES",
    "SUPPORTED_LEGACY_PROTOCOL_VERSIONS",
    "SUPPORTED_MODERN_PROTOCOL_VERSIONS",
    "SUPPORTED_PROTOCOL_VERSIONS",
    "GatewayProtocolProfile",
]
