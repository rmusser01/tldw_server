from __future__ import annotations

from typing import Iterable

from fastapi import APIRouter, FastAPI

from tldw_Server_API.app.api.v1.router_groups.admin import iter_admin_router_specs
from tldw_Server_API.app.api.v1.router_groups.content import iter_content_router_specs
from tldw_Server_API.app.api.v1.router_groups.core import iter_core_router_specs
from tldw_Server_API.app.api.v1.router_groups.spec import RouterSpec


def include_router_idempotent(
    app: FastAPI,
    router: APIRouter,
    *,
    prefix: str = "",
    tags: list[str] | tuple[str, ...] | None = None,
) -> bool:
    """Include router exactly once for the same router/prefix/tags signature."""
    normalized_tags = tuple(tags or ())
    registry_key = (id(router), prefix, normalized_tags)
    registry = getattr(app.state, "_tldw_router_registry", None)
    if registry is None:
        registry = set()
        app.state._tldw_router_registry = registry

    if registry_key in registry:
        return False

    include_kwargs: dict[str, object] = {"prefix": prefix}
    if normalized_tags:
        include_kwargs["tags"] = list(normalized_tags)
    app.include_router(router, **include_kwargs)
    registry.add(registry_key)
    return True


def register_router_specs(app: FastAPI, specs: Iterable[RouterSpec]) -> int:
    """Register router specs, respecting route_key gating via route_enabled()."""
    from loguru import logger

    count = 0
    for spec in specs:
        if spec.route_key:
            try:
                from tldw_Server_API.app.core.config import route_enabled

                if not route_enabled(spec.route_key, default_stable=spec.default_stable):
                    logger.info(f"Route disabled by policy: {spec.route_key}")
                    continue
            except Exception:  # noqa: BLE001
                pass  # If route_enabled fails, include by default
        if include_router_idempotent(app, spec.router, prefix=spec.prefix, tags=spec.tags):
            count += 1
    return count


def register_all_routers(app: FastAPI) -> int:
    """Register all grouped API routers via idempotent include semantics."""
    total = 0
    total += register_router_specs(app, iter_core_router_specs())
    total += register_router_specs(app, iter_content_router_specs())
    total += register_router_specs(app, iter_admin_router_specs())
    return total
