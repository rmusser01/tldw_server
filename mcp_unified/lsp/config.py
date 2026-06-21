"""Runtime limits and timeout configuration for LSP tooling."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Mapping


@dataclass(frozen=True, slots=True)
class LspRuntimeConfig:
    """Conservative runtime settings for LSP subprocess integrations."""

    request_timeout_seconds: float = 5.0
    startup_timeout_seconds: float = 10.0
    idle_ttl_seconds: int = 300
    max_diagnostics: int = 500
    max_symbols: int = 500
    max_references: int = 500
    max_hover_bytes: int = 16_000
    max_preview_bytes: int = 200_000
    max_stderr_bytes: int = 8_000

    @classmethod
    def from_mapping(cls, settings: Mapping[str, object]) -> "LspRuntimeConfig":
        """Build a config from a mapping, ignoring unknown keys."""

        field_names = {field.name for field in fields(cls)}
        values = {key: value for key, value in settings.items() if key in field_names}
        return cls(**values)


DEFAULT_LSP_CONFIG = LspRuntimeConfig()
