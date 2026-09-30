"""Resolve which Resource Governor policy governs a request.

Order (ADR-057):
1. ``route_map.by_path`` globs, first match.
2. The innermost mapped tag of the served route.
3. ``default`` for any other ``/api/`` path.

Anything else is ungoverned. The middleware, both coverage audits and the CI
route-map lint call this module, so what they report is what is enforced.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from starlette.routing import compile_path

from tldw_Server_API.app.core.Utils.fastapi_routes import ServedRoute, iter_served_routes

from .policy_eval import DEFAULT_POLICY_ID

_PREFIX_DEPTH = 4
_CACHE_SIZE = 4096


def compile_route_glob(pattern: str) -> re.Pattern[str]:
    """Compile a ``by_path`` glob: ``*`` matches anything, anchored unless it ends with ``*``."""
    regex = re.escape(pattern).replace("\\*", ".*")
    if not pattern.endswith("*"):
        regex += "$"
    return re.compile(regex)


@dataclass(frozen=True)
class _IndexedRoute:
    regex: re.Pattern[str]
    methods: frozenset[str]
    tags: tuple[str, ...]


def _static_prefix(path: str) -> tuple[str, ...]:
    out: list[str] = []
    for seg in path.strip("/").split("/"):
        if not seg or "{" in seg or len(out) == _PREFIX_DEPTH:
            break
        out.append(seg)
    return tuple(out)


class PolicyResolver:
    """Resolve (path, method) to a policy ID, or None when ungoverned."""

    def __init__(self, route_map: Mapping[str, Any], served_routes: Iterable[ServedRoute]) -> None:
        self._by_path = [
            (compile_route_glob(str(pattern)), str(policy))
            for pattern, policy in dict(route_map.get("by_path") or {}).items()
        ]
        self._by_tag = {str(tag): str(policy) for tag, policy in dict(route_map.get("by_tag") or {}).items()}
        self._groups: dict[tuple[str, ...], list[_IndexedRoute]] = {}
        for route in served_routes:
            if not route.path or not route.methods:
                continue  # mounts and websockets carry no HTTP methods
            regex, _fmt, _conv = compile_path(route.path)
            self._groups.setdefault(_static_prefix(route.path), []).append(
                _IndexedRoute(regex, frozenset(route.methods), tuple(route.tags))
            )
        self._cache: OrderedDict[tuple[str, str], str | None] = OrderedDict()

    def resolve(self, path: str, method: str) -> str | None:
        key = (method.upper(), path)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        result = self._resolve_uncached(path, key[0])
        self._cache[key] = result
        if len(self._cache) > _CACHE_SIZE:
            self._cache.popitem(last=False)
        return result

    def _resolve_uncached(self, path: str, method: str) -> str | None:
        for regex, policy in self._by_path:
            if regex.match(path):
                return policy
        tagged = self._resolve_tag(path, method)
        if tagged is not None:
            return tagged
        return DEFAULT_POLICY_ID if path.startswith("/api/") else None

    def _resolve_tag(self, path: str, method: str) -> str | None:
        # ponytail: tries the longest static-prefix group first, then served order within a
        # group (as Starlette does). Strict served order across groups would cost a full scan.
        segments = [s for s in path.strip("/").split("/") if s]
        for depth in range(min(len(segments), _PREFIX_DEPTH), -1, -1):
            for route in self._groups.get(tuple(segments[:depth]), ()):
                if method in route.methods and route.regex.match(path):
                    for tag in reversed(route.tags):
                        if tag in self._by_tag:
                            return self._by_tag[tag]
                    return None
        return None


def _routes_version(app: Any) -> Any:
    # ponytail: O(1) top-level counter (bumped by app.include_router / add_api_route), read per
    # request. _get_routes_version() would be exact for nested routers but walks every route
    # (~2k) on each call. Upgrade path: poll it on a timer if nested routers ever mutate
    # after startup.
    router = getattr(app, "router", None)
    version = getattr(router, "_routes_version", None)  # FastAPI private attribute, pinned
    if isinstance(version, int):
        return (version, len(getattr(app, "routes", None) or []))
    return len(getattr(app, "routes", None) or [])


def get_policy_resolver(app: Any) -> PolicyResolver | None:
    """Return the app's resolver, rebuilt when the route map or the route table changes."""
    state = getattr(app, "state", None)
    loader = getattr(state, "rg_policy_loader", None)
    if loader is None:
        return None
    try:
        snap = loader.get_snapshot()
    except (AttributeError, RuntimeError):
        return None
    version = _routes_version(app)
    cached = getattr(state, "rg_policy_resolver", None)
    if cached is not None and cached[0] is snap and cached[1] == version:
        return cached[2]
    resolver = PolicyResolver(getattr(snap, "route_map", {}) or {}, iter_served_routes(getattr(app, "routes", []) or []))
    state.rg_policy_resolver = (snap, version, resolver)
    return resolver
