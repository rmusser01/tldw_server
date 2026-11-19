from __future__ import annotations

from typing import Any, Protocol


class ConfigParserLike(Protocol):
    def get(self, section: str, option: str, *, fallback: Any) -> Any:
        ...
