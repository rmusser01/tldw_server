from __future__ import annotations

import os

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


def worker_route_default(
    route_key: str,
    *,
    default_stable: bool = True,
    test_mode: bool = False,
) -> bool:
    if test_mode:
        return False

    try:
        from tldw_Server_API.app.core.config import refresh_config_cache, route_enabled

        if is_explicit_pytest_runtime():
            refresh_config_cache()

        return bool(route_enabled(route_key, default_stable=default_stable))
    except _WORKER_POLICY_EXCEPTIONS as exc:
        logger.debug("Worker startup policy route check failed for {}: {}", route_key, exc)
        return bool(default_stable)


def worker_path_enabled(
    flag_key: str,
    route_key: str,
    *,
    default_stable: bool = True,
    test_mode: bool = False,
) -> bool:
    route_default = worker_route_default(
        route_key,
        default_stable=default_stable,
        test_mode=test_mode,
    )
    return env_flag(flag_key, route_default)


def should_start_inprocess_worker(
    flag_key: str,
    route_key: str,
    *,
    sidecar_mode: bool,
    default_stable: bool = True,
    test_mode: bool = False,
) -> bool:
    if sidecar_mode:
        return False

    return worker_path_enabled(
        flag_key,
        route_key,
        default_stable=default_stable,
        test_mode=test_mode,
    )
