"""Selection helpers for deriving router groups from canonical specs."""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Iterable

from tldw_Server_API.app.api.v1.router_groups.spec import RouterSpec


@dataclass(frozen=True, slots=True)
class RouterSpecOverride:
    """Explicit metadata overrides for a selected router spec."""

    prefix: str | None = None
    tags: tuple[str, ...] | None = None
    route_key: str | None = None
    default_stable: bool | None = None
    name: str | None = None
    skip_context: str | None = None
    skip_exceptions: tuple[type[Exception], ...] | None = None


def select_router_specs_by_name(
    specs: Iterable[RouterSpec],
    names: Iterable[str],
    *,
    overrides: dict[str, RouterSpecOverride] | None = None,
) -> list[RouterSpec]:
    """Select router specs by canonical name while preserving metadata."""
    requested_names = tuple(names)
    requested_name_set = set(requested_names)
    by_name: dict[str, RouterSpec] = {}
    for spec in specs:
        key = spec.name or spec.route_key
        if not key or key not in requested_name_set:
            continue
        if key in by_name:
            raise ValueError(f"Duplicate router spec selection key: {key}")
        by_name[key] = spec

    selected: list[RouterSpec] = []
    override_map = overrides or {}
    for name in requested_names:
        try:
            spec = by_name[name]
        except KeyError as exc:
            raise KeyError(f"Router spec not found: {name}") from exc

        override = override_map.get(name)
        if override is not None:
            replacement_kwargs = {
                field.name: value
                for field in fields(override)
                if (value := getattr(override, field.name)) is not None
            }
            if replacement_kwargs:
                spec = replace(spec, **replacement_kwargs)
        selected.append(spec)
    return selected
