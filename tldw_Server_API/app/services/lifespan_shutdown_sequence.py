"""
Shutdown sequence orchestration extracted from the application lifespan.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import FastAPI

from tldw_Server_API.app.services.lifecycle_worker_engine import LifecycleWorkerEngine
from tldw_Server_API.app.services.lifecycle_workers import ShutdownPhase
from tldw_Server_API.app.services.lifespan_worker_runtime_state import (
    LifespanWorkerRuntimeState,
)


async def run_lifespan_shutdown_sequence(
    *,
    app: FastAPI,
    worker_runtime: LifespanWorkerRuntimeState,
    readiness_state: dict[str, Any],
    db_pool: Any,
    session_manager: Any,
    heavy_startup_handles: Any,
    build_legacy_shutdown_context: Any,
    apply_shutdown_transition_gate: Any,
    quiesce_owned_job_pollers_for_shutdown: Any,
    run_coordinated_shutdown: Any,
    startup_guard_exceptions: tuple[type[BaseException], ...],
    import_exceptions: tuple[type[BaseException], ...],
    in_pytest_runtime: bool,
    test_db_instance_ref: Any,
    timed_shutdown_segment: Any,
    record_shutdown_timing_total: Any,
    monotonic: Any = time.monotonic,
) -> None:
    """Run the post-yield lifespan shutdown sequence in the existing order."""
    shutdown_started = monotonic()
    try:
        app.state._tldw_shutdown_timing_segments = []
    except startup_guard_exceptions:
        pass

    legacy_shutdown_plan: list[Any] = []
    with timed_shutdown_segment(app, "transition_handoff"):
        from tldw_Server_API.app.services.shutdown_transition_handoff import (
            shutdown_transition_handoff,
        )

        transition_handoff_handles = await shutdown_transition_handoff(
            app=app,
            readiness_state=readiness_state,
            build_legacy_shutdown_context=build_legacy_shutdown_context,
            apply_shutdown_transition_gate=apply_shutdown_transition_gate,
            startup_guard_exceptions=startup_guard_exceptions,
            import_exceptions=import_exceptions,
        )
        legacy_shutdown_plan = transition_handoff_handles.legacy_shutdown_plan

    lifecycle_worker_engine = LifecycleWorkerEngine()
    worker_lifecycle_session = worker_runtime.worker_lifecycle_session

    from tldw_Server_API.app.services.shutdown_job_poller_handoff import (
        run_shutdown_job_poller_handoff,
    )

    await run_shutdown_job_poller_handoff(
        app=app,
        worker_lifecycle_session=worker_lifecycle_session,
        lifecycle_worker_engine=lifecycle_worker_engine,
        quiesce_owned_job_pollers_for_shutdown=quiesce_owned_job_pollers_for_shutdown,
        startup_guard_exceptions=startup_guard_exceptions,
        import_exceptions=import_exceptions,
    )

    stopped_background_worker_names: set[str] = set()
    with timed_shutdown_segment(app, "background_worker_shutdown"):
        if worker_lifecycle_session is not None:
            await lifecycle_worker_engine.stop_phase(
                worker_lifecycle_session,
                ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
            )
        stopped_background_worker_names = set(
            getattr(app.state, "_tldw_shutdown_stopped_background_worker_names", [])
        )

    from tldw_Server_API.app.services.shutdown_coordinated_legacy_components import (
        run_shutdown_coordinated_legacy_components,
    )

    await run_shutdown_coordinated_legacy_components(
        app=app,
        legacy_shutdown_plan=legacy_shutdown_plan,
        run_coordinated_shutdown=run_coordinated_shutdown,
        startup_guard_exceptions=startup_guard_exceptions,
        import_exceptions=import_exceptions,
        stopped_background_worker_names=stopped_background_worker_names,
    )

    from tldw_Server_API.app.services.shutdown_pre_worker_cleanup import (
        run_shutdown_pre_worker_cleanup,
    )

    await run_shutdown_pre_worker_cleanup(
        app=app,
        guard_exceptions=startup_guard_exceptions,
    )

    if worker_lifecycle_session is not None:
        await lifecycle_worker_engine.stop_phase(
            worker_lifecycle_session,
            ShutdownPhase.POST_WORKER_SHUTDOWN,
        )

    from tldw_Server_API.app.services.shutdown_post_worker_services import (
        run_shutdown_post_worker_non_worker_cleanup,
    )

    await run_shutdown_post_worker_non_worker_cleanup(
        guard_exceptions=startup_guard_exceptions,
    )

    from tldw_Server_API.app.services.shutdown_final_cleanup_tail import (
        shutdown_final_cleanup_tail,
    )

    await shutdown_final_cleanup_tail(
        app=app,
        db_pool=db_pool,
        session_manager=session_manager,
        heavy_startup_handles=heavy_startup_handles,
        in_pytest_for_db_pool_shutdown=in_pytest_runtime,
        in_pytest_for_tts_shutdown=in_pytest_runtime,
        import_exceptions=import_exceptions,
        startup_guard_exceptions=startup_guard_exceptions,
        test_db_instance_ref=test_db_instance_ref,
        timed_shutdown_segment=timed_shutdown_segment,
    )

    record_shutdown_timing_total(
        app,
        int((monotonic() - shutdown_started) * 1000),
    )
