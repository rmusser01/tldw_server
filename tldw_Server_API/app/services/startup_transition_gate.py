"""
Startup transition gate extracted from the application lifespan.
"""

from __future__ import annotations

from typing import Any, MutableMapping


def apply_startup_transition_gate(
    *,
    app: Any,
    readiness_state: MutableMapping[str, bool],
    import_exceptions: tuple[type[BaseException], ...],
) -> None:
    try:
        _mark_lifecycle_startup(app, readiness_state)
        _disable_job_acquire_gate()
    except import_exceptions:
        pass


def _mark_lifecycle_startup(app: Any, readiness_state: MutableMapping[str, bool]) -> None:
    from tldw_Server_API.app.services.app_lifecycle import mark_lifecycle_startup

    mark_lifecycle_startup(app, readiness_state)


def _disable_job_acquire_gate() -> None:
    from tldw_Server_API.app.core.Jobs.manager import JobManager

    JobManager.set_acquire_gate(False)
