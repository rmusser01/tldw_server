from __future__ import annotations

import configparser
from typing import Any, Protocol

# Errors a caller swallows when config.txt cannot be loaded (fail-soft: fall back to the default).
CONFIG_LOAD_ERRORS = (
    AttributeError,
    FileNotFoundError,
    ImportError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
    configparser.Error,
)


class ConfigParserLike(Protocol):
    def get(self, section: str, option: str, *, fallback: Any) -> Any:
        ...
