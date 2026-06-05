"""Helpers for selecting router specs from canonical router groups."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from tldw_Server_API.app.api.v1.router_groups.spec import RouterSpec


@dataclass(frozen=True)
class RouterSpecOverride:
    """Optional field overrides applied when reusing a canonical router spec."""

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
    overrides: Mapping[str, RouterSpecOverride] | None = None,
) -> list[RouterSpec]:
    """Return router specs matching the requested names while preserving metadata."""
    specs_by_name = {spec.name or spec.route_key: spec for spec in specs}
    selected_specs: list[RouterSpec] = []
    override_map = overrides or {}

    for requested_name in names:
        try:
            spec = specs_by_name[requested_name]
        except KeyError as exc:
            raise KeyError(f"Router spec '{requested_name}' was not found") from exc

        override = override_map.get(requested_name, RouterSpecOverride())
        selected_specs.append(
            RouterSpec(
                router=spec.router,
                prefix=(
                    override.prefix
                    if override.prefix is not None
                    else spec.prefix
                ),
                tags=override.tags if override.tags is not None else spec.tags,
                route_key=(
                    override.route_key
                    if override.route_key is not None
                    else spec.route_key
                ),
                default_stable=(
                    override.default_stable
                    if override.default_stable is not None
                    else spec.default_stable
                ),
                name=override.name if override.name is not None else spec.name,
                skip_context=(
                    override.skip_context
                    if override.skip_context is not None
                    else spec.skip_context
                ),
                skip_exceptions=(
                    override.skip_exceptions
                    if override.skip_exceptions is not None
                    else spec.skip_exceptions
                ),
            )
        )

    return selected_specs
