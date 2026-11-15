"""Lifecycle ownership of lazy moderation supervision, without service activation."""

from collections.abc import Awaitable, Callable, Sequence

from tldw_Server_API.app.services.lifecycle_worker_specs import (
    WorkerLifecycleContext,
    WorkerSpec,
    WorkerStrategy,
)
from tldw_Server_API.app.services.lifecycle_workers import ShutdownPhase


def provide_moderation_worker_specs(context: WorkerLifecycleContext) -> Sequence[WorkerSpec]:
    """Register a shutdown owner without spawning a worker at startup/import."""
    from tldw_Server_API.app.core.Moderation.async_execution import prepare_moderation_execution
    from tldw_Server_API.app.core.Moderation.execution_incidents import (
        _reporter_failure,
        prepare_moderation_incident_store,
    )

    prepare_moderation_execution()
    try:
        prepare_moderation_incident_store()
    except Exception:  # noqa: BLE001 - degraded observability must retain shutdown ownership.
        _reporter_failure()
    return (
        WorkerSpec(
            name="moderation_execution",
            task_name="moderation_execution",
            category="moderation",
            phase=ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
            timeout_sec=1.5,
            strategy=WorkerStrategy.CALLBACK_ONLY,
            shutdown_callback_factory=_shutdown_factory,
        ),
    )


def _shutdown_factory(context: WorkerLifecycleContext) -> Callable[[], Awaitable[None]]:
    from tldw_Server_API.app.core.Moderation.async_execution import close_moderation_offloads
    from tldw_Server_API.app.core.Moderation.execution_incidents import close_moderation_incident_reporting
    from tldw_Server_API.app.core.Moderation.execution_runner import get_execution_runner

    async def shutdown() -> None:
        close_moderation_offloads()
        # No generic executor/pool owns child spawn or cleanup. This bounded
        # close is only a shutdown callback, never request-path execution.
        get_execution_runner().close(timeout_s=0.5)
        await close_moderation_incident_reporting()

    return shutdown
