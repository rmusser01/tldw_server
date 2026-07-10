from __future__ import annotations

import os
from collections.abc import Callable
from inspect import Parameter, signature

from loguru import logger

from tldw_Server_API.app.core.testing import is_explicit_pytest_runtime, is_truthy

_WORKER_POLICY_EXCEPTIONS = (
    AttributeError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


def env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return bool(default)
    return is_truthy(raw)


def _route_enabled_accepts_default_stable(route_enabled: Callable[..., bool]) -> bool | None:
    """Return whether a route callback advertises default_stable support."""

    try:
        parameters = signature(route_enabled).parameters.values()
    except (TypeError, ValueError):
        return None

    return any(
        parameter.kind is Parameter.VAR_KEYWORD
        or parameter.name == "default_stable"
        for parameter in parameters
    )


def worker_route_default(
    route_key: str,
    *,
    default_stable: bool = True,
    test_mode: bool = False,
    route_enabled: Callable[..., bool] | None = None,
) -> bool:
    if test_mode:
        return False

    if route_enabled is not None:
        try:
            accepts_default_stable = _route_enabled_accepts_default_stable(route_enabled)
            if accepts_default_stable is True:
                return bool(route_enabled(route_key, default_stable=default_stable))
            else:
                return bool(route_enabled(route_key))
        except _WORKER_POLICY_EXCEPTIONS as exc:
            logger.debug("Worker startup policy route check failed for {}: {}", route_key, exc)
            return False

    try:
        from tldw_Server_API.app.core.config import refresh_config_cache, route_enabled as config_route_enabled

        if is_explicit_pytest_runtime():
            refresh_config_cache()

        return bool(config_route_enabled(route_key, default_stable=default_stable))
    except _WORKER_POLICY_EXCEPTIONS as exc:
        logger.debug("Worker startup policy route check failed for {}: {}", route_key, exc)
        return bool(default_stable)


def worker_path_enabled(
    flag_key: str,
    route_key: str,
    *,
    default_stable: bool = True,
    test_mode: bool = False,
    route_enabled: Callable[..., bool] | None = None,
) -> bool:
    route_default = worker_route_default(
        route_key,
        default_stable=default_stable,
        test_mode=test_mode,
        route_enabled=route_enabled,
    )
    return env_flag(flag_key, route_default)


def should_start_inprocess_worker(
    flag_key: str,
    route_key: str,
    *,
    sidecar_mode: bool,
    default_stable: bool = True,
    test_mode: bool = False,
    route_enabled: Callable[..., bool] | None = None,
) -> bool:
    if sidecar_mode:
        return False

    return worker_path_enabled(
        flag_key,
        route_key,
        default_stable=default_stable,
        test_mode=test_mode,
        route_enabled=route_enabled,
    )
