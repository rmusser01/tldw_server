"""Runtime limits and timeout configuration for LSP tooling."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Mapping


_FLOAT_FIELDS = frozenset({"request_timeout_seconds", "startup_timeout_seconds"})
_INT_FIELDS = frozenset(
    {
        "idle_ttl_seconds",
        "max_diagnostics",
        "max_symbols",
        "max_references",
        "max_hover_bytes",
        "max_preview_bytes",
        "max_stderr_bytes",
    }
)


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

    def __post_init__(self) -> None:
        for field_name in _FLOAT_FIELDS:
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field_name} must be a positive number")
            if value <= 0:
                raise ValueError(f"{field_name} must be greater than zero")
            object.__setattr__(self, field_name, float(value))

        for field_name in _INT_FIELDS:
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be a positive integer")
            if value <= 0:
                raise ValueError(f"{field_name} must be greater than zero")

    @classmethod
    def from_mapping(cls, settings: Mapping[str, object]) -> "LspRuntimeConfig":
        """Build a config from a mapping, ignoring unknown keys."""

        field_names = {field.name for field in fields(cls)}
        values = {key: value for key, value in settings.items() if key in field_names}
        return cls(**values)


DEFAULT_LSP_CONFIG = LspRuntimeConfig()
