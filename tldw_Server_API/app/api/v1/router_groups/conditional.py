"""Helpers for optional router imports used by router groups."""
from __future__ import annotations

import importlib
from dataclasses import dataclass

from tldw_Server_API.app.api.v1.router_groups.spec import RouterSpec


class OptionalRouterMissingModule(ImportError):
    """Raised when the target optional router module itself is unavailable."""


class OptionalRouterMissingAttribute(AttributeError):
    """Raised when an optional router module is missing the configured router attr."""


@dataclass(frozen=True)
class ImportedRouterSpec:
    """RouterSpec metadata for a router imported inside a router group."""

    import_path: str
    log_name: str
    prefix: str = ""
    tags: tuple[str, ...] = ()
    route_key: str = ""
    default_stable: bool = True
    attr_name: str = "router"
    skip_context: str = ""
    skip_exceptions: tuple[type[Exception], ...] = (
        OptionalRouterMissingModule,
        OptionalRouterMissingAttribute,
    )


def append_imported_router_spec(
    specs: list[RouterSpec],
    definition: ImportedRouterSpec,
) -> None:
    """Append an imported router spec without importing until registration."""
    def _router_factory():
        try:
            module = importlib.import_module(definition.import_path)
        except ModuleNotFoundError as e:
            if e.name == definition.import_path:
                raise OptionalRouterMissingModule(str(e)) from e
            raise
        try:
            return getattr(module, definition.attr_name)
        except AttributeError as e:
            raise OptionalRouterMissingAttribute(
                f"{definition.import_path}.{definition.attr_name}"
            ) from e

    specs.append(
        RouterSpec(
            router=_router_factory,
            prefix=definition.prefix,
            tags=definition.tags,
            route_key=definition.route_key,
            default_stable=definition.default_stable,
            name=definition.log_name,
            skip_context=definition.skip_context,
            skip_exceptions=definition.skip_exceptions,
        )
    )
