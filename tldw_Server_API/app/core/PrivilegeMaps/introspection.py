from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.routing import APIRoute
from loguru import logger

from tldw_Server_API.app.core.AuthNZ.privilege_catalog import PrivilegeCatalog


@dataclass(frozen=True)
class DependencyMetadata:
    """Normalized metadata for dependencies contributing to privilege checks."""

    id: str
    type: str
    module: str


@dataclass(frozen=True)
class RouteMetadata:
    """Describes a FastAPI route relevant for privilege mapping."""

    path: str
    methods: tuple[str, ...]
    name: str
    tags: tuple[str, ...]
    endpoint: str
    dependencies: tuple[DependencyMetadata, ...] = field(default_factory=tuple)
    dependency_sources: tuple[str, ...] = field(default_factory=tuple)
    rate_limit_resources: tuple[str, ...] = field(default_factory=tuple)
    summary: str | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        if not self.dependency_sources and self.dependencies:
            object.__setattr__(
                self,
                "dependency_sources",
                tuple(dep.module for dep in self.dependencies),
            )

    @property
    def methods_or_any(self) -> tuple[str, ...]:
        return self.methods or ("ANY",)


def _normalize_dependency_name(callable_obj: object) -> tuple[str, str, str]:
    module = getattr(callable_obj, "__module__", "") or ""
    qualname = getattr(callable_obj, "__qualname__", getattr(callable_obj, "__name__", repr(callable_obj))) or ""
    simple = qualname.split(".")[-1]
    qualified = f"{module}.{qualname}".strip(".")
    return simple, module, qualified


def _build_dependency_id(simple: str, module: str) -> str:
    if module:
        suffix = module.rsplit(".", 1)[-1]
        if suffix and suffix != simple:
            return f"{suffix}.{simple}"
    return simple

def _collect_candidate_scopes(callable_obj: object) -> set[str]:
    candidates: set[str] = set()
    endpoint_id = getattr(callable_obj, "_tldw_endpoint_id", None)
    if isinstance(endpoint_id, str):
        candidates.add(endpoint_id)
    scope_attr = getattr(callable_obj, "_tldw_scope_name", None)
    if isinstance(scope_attr, str):
        candidates.add(scope_attr)
    return candidates


def _extract_scope_matches(callable_obj: object, known_scopes: set[str]) -> tuple[set[str], set[str]]:
    candidates = _collect_candidate_scopes(callable_obj)
    matches = {scope for scope in candidates if scope in known_scopes}
    unknown = {scope for scope in candidates if scope not in known_scopes}
    return matches, unknown


def _extract_rate_limit_resources(callable_obj: object) -> tuple[str, ...]:
    resources: list[str] = []
    singular = getattr(callable_obj, "_tldw_rate_limit_resource", None)
    if isinstance(singular, str) and singular:
        resources.append(singular)
    plural = getattr(callable_obj, "_tldw_rate_limit_resources", ())
    if isinstance(plural, (list, tuple)):
        resources.extend(
            resource
            for resource in plural
            if isinstance(resource, str) and resource
        )
    return tuple(dict.fromkeys(resources))


def collect_privilege_route_registry(
    app: FastAPI,
    catalog: PrivilegeCatalog,
    *,
    strict: bool = False,
) -> dict[str, list[RouteMetadata]]:
    """
    Build a registry mapping privilege scope identifiers to relevant FastAPI route metadata.
    """
    scope_ids = {scope.id for scope in catalog.scopes}
    registry: dict[str, dict[tuple[str, tuple[str, ...]], RouteMetadata]] = {}
    unknown_scope_refs: set[tuple[str, str, str]] = set()

    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue

        dependant_deps = getattr(route.dependant, "dependencies", []) or []
        explicit_deps = list(getattr(route, "dependencies", []) or [])
        aggregated_deps = list(dependant_deps) + explicit_deps
        if not aggregated_deps:
            # Even if no dependencies, there might still be catalog mappings via tags or other mechanisms.
            aggregated_deps = []

        dependency_entries: list[DependencyMetadata] = []
        dependency_sources: list[str] = []
        seen_dependency_keys: set[tuple[str, str]] = set()
        seen_dependency_sources: set[str] = set()
        rate_resources: list[str] = []
        seen_rate_resources: set[str] = set()
        scope_matches: set[str] = set()

        for dep in aggregated_deps:
            callable_obj = getattr(dep, "call", None)
            if callable_obj is None:
                callable_obj = getattr(dep, "dependency", None)
            if not callable(callable_obj):
                continue
            simple, module_path, qualified = _normalize_dependency_name(callable_obj)
            module_path = module_path or (qualified.rsplit(".", 1)[0] if "." in qualified else "")
            dep_key = (simple, module_path)
            if dep_key not in seen_dependency_keys:
                seen_dependency_keys.add(dep_key)
                dependency_entries.append(
                    DependencyMetadata(
                        id=_build_dependency_id(simple, module_path),
                        type="dependency",
                        module=module_path,
                    )
                )
            if qualified not in seen_dependency_sources:
                seen_dependency_sources.add(qualified)
                dependency_sources.append(qualified)
            for resource in _extract_rate_limit_resources(callable_obj):
                if resource not in seen_rate_resources:
                    seen_rate_resources.add(resource)
                    rate_resources.append(resource)
            matches, unknown = _extract_scope_matches(callable_obj, scope_ids)
            scope_matches.update(matches)
            if unknown:
                for scope_id in unknown:
                    unknown_scope_refs.add((scope_id, qualified, route.path))

        if not scope_matches:
            continue

        tags = tuple(route.tags or ())
        methods = tuple(sorted(route.methods or []))
        metadata = RouteMetadata(
            path=route.path,
            methods=methods,
            name=route.name or "",
            tags=tags,
            endpoint=f"{route.endpoint.__module__}.{route.endpoint.__qualname__}",
            dependencies=tuple(dependency_entries),
            dependency_sources=tuple(dependency_sources),
            rate_limit_resources=tuple(rate_resources),
            summary=route.summary,
            description=route.description,
        )

        for scope_id in scope_matches:
            scope_bucket = registry.setdefault(scope_id, {})
            key = (metadata.path, metadata.methods)
            scope_bucket[key] = metadata

    if unknown_scope_refs:
        rendered = ", ".join(
            f"{scope_id} via {dependency} on {route_path}"
            for scope_id, dependency, route_path in sorted(unknown_scope_refs)
        )
        message = f"Unknown privilege scopes referenced in route dependencies: {rendered}"
        if strict:
            raise ValueError(message)
        logger.warning(message)

    return {scope_id: list(bucket.values()) for scope_id, bucket in registry.items()}


def serialize_route_registry(registry: dict[str, list[RouteMetadata]]) -> dict[str, list[dict[str, Any]]]:
    """Convert the route registry into a JSON-serializable structure with deterministic ordering."""

    serialized: dict[str, list[dict[str, Any]]] = {}
    for scope_id in sorted(registry.keys()):
        entries = []
        for meta in sorted(registry[scope_id], key=lambda item: (item.path, tuple(item.methods_or_any))):
            dependencies = sorted(
                meta.dependencies,
                key=lambda dep: (dep.id, dep.module or "", dep.type),
            )
            entries.append(
                {
                    "path": meta.path,
                    "methods": list(meta.methods_or_any),
                    "name": meta.name,
                    "tags": list(meta.tags),
                    "endpoint": meta.endpoint,
                    "dependencies": [
                        {"id": dep.id, "type": dep.type, "module": dep.module} for dep in dependencies
                    ],
                    "rate_limit_resources": sorted(meta.rate_limit_resources),
                    "summary": meta.summary,
                    "description": meta.description,
                }
            )
        serialized[scope_id] = entries
    return serialized


def write_route_registry_snapshot(
    app: FastAPI,
    catalog: PrivilegeCatalog,
    destination: Path,
    *,
    strict: bool = True,
) -> None:
    """
    Generate a deterministic JSON snapshot of the privilege-aware route registry.

    This helper enables CI diffs (git diff) to detect routing or dependency changes that may
    require catalog updates.
    """

    registry = collect_privilege_route_registry(app, catalog, strict=strict)
    payload = serialize_route_registry(registry)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
