"""Lazy Slides module entry points."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "ConflictError",
    "InputError",
    "SchemaError",
    "SlidesDatabase",
    "SlidesDatabaseError",
    "SlidesGenerator",
    "export_presentation_bundle",
    "export_presentation_json",
    "export_presentation_markdown",
    "export_presentation_pdf",
]

_EXPORTS = {
    "ConflictError": (".slides_db", "ConflictError"),
    "InputError": (".slides_db", "InputError"),
    "SchemaError": (".slides_db", "SchemaError"),
    "SlidesDatabase": (".slides_db", "SlidesDatabase"),
    "SlidesDatabaseError": (".slides_db", "SlidesDatabaseError"),
    "SlidesGenerator": (".slides_generator", "SlidesGenerator"),
    "export_presentation_bundle": (".slides_export", "export_presentation_bundle"),
    "export_presentation_json": (".slides_export", "export_presentation_json"),
    "export_presentation_markdown": (".slides_export", "export_presentation_markdown"),
    "export_presentation_pdf": (".slides_export", "export_presentation_pdf"),
}


def __getattr__(name: str) -> Any:
    """Resolve the historical package-level exports only when requested."""
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazy public exports in interactive discovery."""
    return sorted({*globals(), *__all__})
