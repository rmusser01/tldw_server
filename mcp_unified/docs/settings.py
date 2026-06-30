from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

from .models import AccessScope

_TRUE_VALUES = {"true", "1", "yes", "on"}
_FALSE_VALUES = {"false", "0", "no", "off"}


def _coerce_bool(value: object, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_VALUES:
            return True
        if normalized in _FALSE_VALUES:
            return False
        raise ValueError(f"{field_name} must be a recognized boolean string")
    return bool(value)


def _coerce_trusted_roots(value: object) -> tuple[Path, ...]:
    if value is None or value == "":
        return ()
    if isinstance(value, str | PathLike):
        items = (value,)
    else:
        items = tuple(value) if isinstance(value, Iterable) else (value,)
    return tuple(Path(item).expanduser().resolve() for item in items)


def _coerce_positive_int(value: object, field_name: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{field_name} must be positive")
    return result


@dataclass(frozen=True)
class DocsSettings:
    db_path: Path
    trusted_roots: tuple[Path, ...] = ()
    max_import_file_bytes: int = 2_000_000
    default_scope: AccessScope = AccessScope()
    enable_web_acquisition: bool = False

    @classmethod
    def from_mapping(cls, values: dict) -> "DocsSettings":
        roots = _coerce_trusted_roots(values.get("trusted_roots"))
        return cls(
            db_path=Path(values.get("db_path", "Databases/mcp_docs.db")).expanduser(),
            trusted_roots=roots,
            max_import_file_bytes=_coerce_positive_int(
                values.get("max_import_file_bytes", 2_000_000),
                "max_import_file_bytes",
            ),
            enable_web_acquisition=_coerce_bool(
                values.get("enable_web_acquisition", False),
                "enable_web_acquisition",
            ),
        )
