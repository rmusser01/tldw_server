"""Helpers for optional router imports used by router groups."""
from __future__ import annotations

import importlib
from dataclasses import dataclass

from loguru import logger

from tldw_Server_API.app.api.v1.router_groups.spec import RouterSpec


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


def append_imported_router_spec(
    specs: list[RouterSpec],
    definition: ImportedRouterSpec,
) -> None:
    """Append an optional imported router spec, preserving existing skip logging."""
    try:
        module = importlib.import_module(definition.import_path)
    except ImportError as e:
        context = f" {definition.skip_context}" if definition.skip_context else ""
        logger.debug(f"Skipping {definition.log_name} router{context}: {e}")
        return

    def _router_factory():
        try:
            return getattr(module, definition.attr_name)
        except AttributeError as e:
            context = f" {definition.skip_context}" if definition.skip_context else ""
            logger.debug(f"Skipping {definition.log_name} router{context}: {e}")
            raise

    specs.append(
        RouterSpec(
            router=_router_factory,
            prefix=definition.prefix,
            tags=definition.tags,
            route_key=definition.route_key,
            default_stable=definition.default_stable,
        )
    )
