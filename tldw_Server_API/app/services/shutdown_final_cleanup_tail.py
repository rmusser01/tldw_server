"""
Final shutdown cleanup tail helper extracted from the application lifespan.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any

from tldw_Server_API.app.services.shutdown_cleanup_timed_segments import (
    CleanupTimedShutdownHandles,
)


async def shutdown_final_cleanup_tail(
    *,
    app: Any,
    db_pool: Any | None,
    session_manager: Any | None,
    heavy_startup_handles: Any | None,
    in_pytest_for_db_pool_shutdown: bool,
    in_pytest_for_tts_shutdown: bool,
    import_exceptions: tuple[type[BaseException], ...],
    startup_guard_exceptions: tuple[type[BaseException], ...],
    test_db_instance_ref: Any,
    timed_shutdown_segment: Callable[[Any, str], AbstractContextManager[Any]],
) -> CleanupTimedShutdownHandles:
    """Run the remaining final cleanup tail in the legacy shutdown order."""
    cleanup_timed_shutdown_handles = await _shutdown_cleanup_timed_segments(
        app=app,
        db_pool=db_pool,
        session_manager=session_manager,
        heavy_startup_handles=heavy_startup_handles,
        in_pytest_for_db_pool_shutdown=in_pytest_for_db_pool_shutdown,
        in_pytest_for_tts_shutdown=in_pytest_for_tts_shutdown,
        import_exceptions=import_exceptions,
        startup_guard_exceptions=startup_guard_exceptions,
        timed_shutdown_segment=timed_shutdown_segment,
    )
    await _shutdown_post_runtime_cleanup(
        test_db_instance_ref=test_db_instance_ref,
        startup_guard_exceptions=startup_guard_exceptions,
        import_exceptions=import_exceptions,
    )
    return cleanup_timed_shutdown_handles


async def _shutdown_cleanup_timed_segments(**kwargs) -> CleanupTimedShutdownHandles:
    from tldw_Server_API.app.services.shutdown_cleanup_timed_segments import (
        shutdown_cleanup_timed_segments,
    )

    return await shutdown_cleanup_timed_segments(**kwargs)


async def _shutdown_post_runtime_cleanup(**kwargs) -> None:
    from tldw_Server_API.app.services.shutdown_post_runtime_cleanup import (
        shutdown_post_runtime_cleanup,
    )

    await shutdown_post_runtime_cleanup(**kwargs)
