"""
Startup pre-core helper extracted from the application lifespan.
"""

from __future__ import annotations

from typing import Any, Callable


async def prepare_startup_pre_core(
    *,
    app: Any,
    logger: Any,
    readiness_state: Any,
    shared_is_truthy: Callable[..., bool],
    route_enabled: Callable[..., bool],
    get_mcp_config: Callable[..., Any],
    validate_mcp_config: Callable[..., Any],
    startup_guard_exceptions: tuple[type[BaseException], ...],
    import_exceptions: tuple[type[BaseException], ...],
    test_mode: bool,
) -> bool:
    """Run the remaining startup pre-core block in the legacy order."""
    _validate_startup_test_runtime(
        logger=logger,
        import_exceptions=import_exceptions,
    )
    _apply_startup_transition_gate(
        app=app,
        readiness_state=readiness_state,
        import_exceptions=import_exceptions,
    )
    await _run_startup_preflight_checks(
        logger=logger,
        startup_guard_exceptions=startup_guard_exceptions,
    )
    defer_heavy = _resolve_deferred_heavy_startup(
        shared_is_truthy=shared_is_truthy,
        startup_guard_exceptions=startup_guard_exceptions,
    )
    _prepare_startup_bg_tasks(
        app=app,
        startup_guard_exceptions=startup_guard_exceptions,
    )
    _start_prompts_close_worker(
        logger=logger,
        startup_guard_exceptions=startup_guard_exceptions,
    )
    _validate_startup_mcp_configuration(
        get_mcp_config=get_mcp_config,
        validate_mcp_config=validate_mcp_config,
        logger=logger,
        startup_guard_exceptions=startup_guard_exceptions,
    )
    _validate_startup_acp_configuration(
        route_enabled=route_enabled,
        logger=logger,
        startup_guard_exceptions=startup_guard_exceptions,
    )
    _validate_startup_content_backend(logger=logger)
    _validate_startup_claims_prompt_validation(
        logger=logger,
        startup_guard_exceptions=startup_guard_exceptions,
    )
    _warm_lazy_evaluations_managers(
        route_enabled=route_enabled,
        logger=logger,
        startup_guard_exceptions=startup_guard_exceptions,
        test_mode=test_mode,
    )
    _initialize_startup_telemetry(
        app=app,
        logger=logger,
        startup_guard_exceptions=startup_guard_exceptions,
    )
    _initialize_startup_sentry(
        logger=logger,
        startup_guard_exceptions=startup_guard_exceptions,
        import_exceptions=import_exceptions,
    )
    return defer_heavy


def _validate_startup_test_runtime(**kwargs) -> None:
    from tldw_Server_API.app.services.startup_test_runtime_guard import (
        validate_startup_test_runtime,
    )

    validate_startup_test_runtime(**kwargs)


def _apply_startup_transition_gate(**kwargs) -> None:
    from tldw_Server_API.app.services.startup_transition_gate import (
        apply_startup_transition_gate,
    )

    apply_startup_transition_gate(**kwargs)


async def _run_startup_preflight_checks(**kwargs) -> None:
    from tldw_Server_API.app.services.startup_preflight_reporting import (
        run_startup_preflight_checks,
    )

    await run_startup_preflight_checks(**kwargs)


def _resolve_deferred_heavy_startup(**kwargs) -> bool:
    from tldw_Server_API.app.services.startup_heavy_policy import (
        resolve_deferred_heavy_startup,
    )

    return resolve_deferred_heavy_startup(**kwargs)


def _prepare_startup_bg_tasks(**kwargs) -> None:
    from tldw_Server_API.app.services.startup_bg_tasks import prepare_startup_bg_tasks

    prepare_startup_bg_tasks(**kwargs)


def _start_prompts_close_worker(**kwargs) -> None:
    from tldw_Server_API.app.services.startup_prompts_close_worker import (
        start_prompts_close_worker,
    )

    start_prompts_close_worker(**kwargs)


def _validate_startup_mcp_configuration(**kwargs) -> None:
    from tldw_Server_API.app.services.startup_mcp_validation import (
        validate_startup_mcp_configuration,
    )

    validate_startup_mcp_configuration(**kwargs)


def _validate_startup_acp_configuration(**kwargs) -> None:
    from tldw_Server_API.app.services.startup_acp_validation import (
        validate_startup_acp_configuration,
    )

    validate_startup_acp_configuration(**kwargs)


def _validate_startup_content_backend(**kwargs) -> None:
    from tldw_Server_API.app.services.startup_content_backend_validation import (
        validate_startup_content_backend,
    )

    validate_startup_content_backend(**kwargs)


def _validate_startup_claims_prompt_validation(**kwargs) -> None:
    from tldw_Server_API.app.services.startup_claims_prompt_validation import (
        validate_startup_claims_prompt_validation,
    )

    validate_startup_claims_prompt_validation(**kwargs)


def _warm_lazy_evaluations_managers(**kwargs) -> None:
    from tldw_Server_API.app.services.startup_evaluations_warmup import (
        warm_lazy_evaluations_managers,
    )

    warm_lazy_evaluations_managers(**kwargs)


def _initialize_startup_telemetry(**kwargs) -> None:
    from tldw_Server_API.app.services.startup_telemetry import (
        initialize_startup_telemetry,
    )

    initialize_startup_telemetry(**kwargs)


def _initialize_startup_sentry(**kwargs) -> None:
    from tldw_Server_API.app.services.startup_sentry import initialize_startup_sentry

    initialize_startup_sentry(**kwargs)
