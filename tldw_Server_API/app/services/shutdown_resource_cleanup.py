"""
Resource-cleanup shutdown helpers extracted from the application lifespan.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger


async def shutdown_resource_cleanup(
    *,
    app: Any,
    session_manager: Any | None,
    heavy_startup_handles: Any | None,
    in_pytest_for_tts_shutdown: bool,
    import_exceptions: tuple[type[BaseException], ...],
    startup_guard_exceptions: tuple[type[BaseException], ...],
    run_in_thread: Callable[[Callable[..., Any]], Awaitable[Any]] = asyncio.to_thread,
) -> None:
    """Run the shutdown resource-cleanup tail before timed shutdown segments."""
    await _shutdown_session_manager(
        session_manager=session_manager,
        guard_exceptions=startup_guard_exceptions,
    )
    await _shutdown_mcp_server(
        heavy_startup_handles=heavy_startup_handles,
        guard_exceptions=startup_guard_exceptions,
    )
    await _shutdown_mcp_rate_limiter(
        guard_exceptions=startup_guard_exceptions,
    )
    await _shutdown_tts_service(
        in_pytest_for_tts_shutdown=in_pytest_for_tts_shutdown,
        guard_exceptions=startup_guard_exceptions,
    )
    await _shutdown_tts_resource_manager(
        in_pytest_for_tts_shutdown=in_pytest_for_tts_shutdown,
        guard_exceptions=startup_guard_exceptions,
    )
    await _shutdown_http_client(
        guard_exceptions=startup_guard_exceptions,
    )
    await _shutdown_chacha_resources(
        guard_exceptions=startup_guard_exceptions,
    )
    await _shutdown_prompts_resources(
        guard_exceptions=startup_guard_exceptions,
    )
    await _shutdown_chat_workflows_resources(
        app=app,
        guard_exceptions=startup_guard_exceptions,
    )
    logger.info("App Shutdown: Cleaning up Chat module components...")
    await _shutdown_provider_manager(
        heavy_startup_handles=heavy_startup_handles,
        guard_exceptions=startup_guard_exceptions,
    )
    await _shutdown_request_queue(
        heavy_startup_handles=heavy_startup_handles,
        guard_exceptions=startup_guard_exceptions,
    )
    await _shutdown_local_llm_manager(
        app=app,
        guard_exceptions=startup_guard_exceptions,
        run_in_thread=run_in_thread,
    )


async def _shutdown_session_manager(
    *,
    session_manager: Any | None,
    guard_exceptions: tuple[type[BaseException], ...],
) -> None:
    try:
        if session_manager is not None:
            await session_manager.shutdown()
            logger.info("App Shutdown: Session manager shutdown")
    except guard_exceptions as exc:
        logger.exception(f"App Shutdown: Error shutting down session manager: {exc}")


async def _shutdown_mcp_server(
    *,
    heavy_startup_handles: Any | None,
    guard_exceptions: tuple[type[BaseException], ...],
) -> None:
    try:
        mcp_server = getattr(heavy_startup_handles, "mcp_server", None)
        if mcp_server is not None:
            await mcp_server.shutdown()
            logger.info("App Shutdown: MCP Unified server shutdown")
    except guard_exceptions as exc:
        logger.exception(f"App Shutdown: Error shutting down MCP Unified server: {exc}")


async def _shutdown_mcp_rate_limiter(
    *,
    guard_exceptions: tuple[type[BaseException], ...],
) -> None:
    try:
        from tldw_Server_API.app.core.MCP_unified.auth.rate_limiter import (
            shutdown_rate_limiter as _shutdown_rate_limiter,
        )

        await _shutdown_rate_limiter()
        logger.info("App Shutdown: MCP rate limiter cleanup task cancelled")
    except guard_exceptions as exc:
        logger.debug(f"App Shutdown: MCP rate limiter shutdown skipped/failed: {exc}")


async def _shutdown_tts_service(
    *,
    in_pytest_for_tts_shutdown: bool,
    guard_exceptions: tuple[type[BaseException], ...],
) -> None:
    if in_pytest_for_tts_shutdown:
        logger.info("App Shutdown: Skipping TTS service shutdown in test context")
        return

    try:
        await _shutdown_tts_service_components()
        logger.info("App Shutdown: TTS service shutdown complete")
    except guard_exceptions as exc:
        logger.exception(f"App Shutdown: Error shutting down TTS service: {exc}")


async def _shutdown_tts_resource_manager(
    *,
    in_pytest_for_tts_shutdown: bool,
    guard_exceptions: tuple[type[BaseException], ...],
) -> None:
    if in_pytest_for_tts_shutdown:
        logger.info("App Shutdown: Skipping TTS resource manager shutdown in test context")
        return

    try:
        await _shutdown_tts_resource_manager_components()
        logger.info("App Shutdown: TTS resource manager shutdown complete")
    except guard_exceptions as exc:
        logger.exception(f"App Shutdown: Error shutting down TTS resource manager: {exc}")


async def _shutdown_http_client(
    *,
    guard_exceptions: tuple[type[BaseException], ...],
) -> None:
    try:
        from tldw_Server_API.app.core.http_client import shutdown_http_client

        await shutdown_http_client()
        logger.info("App Shutdown: HTTP client sessions shutdown complete")
    except guard_exceptions as exc:
        logger.exception(f"App Shutdown: Error shutting down HTTP client sessions: {exc}")


async def _shutdown_chacha_resources(
    *,
    guard_exceptions: tuple[type[BaseException], ...],
) -> None:
    try:
        from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import (
            shutdown_chacha_resources,
        )

        await shutdown_chacha_resources()
        logger.info("App Shutdown: ChaChaNotes resources cleaned up")
    except guard_exceptions as exc:
        logger.exception(f"App Shutdown: Error shutting down ChaChaNotes resources: {exc}")


async def _shutdown_prompts_resources(
    *,
    guard_exceptions: tuple[type[BaseException], ...],
) -> None:
    try:
        from tldw_Server_API.app.api.v1.API_Deps.Prompts_DB_Deps import (
            close_all_cached_prompts_db_instances,
            stop_prompts_pending_close_worker,
        )

        await close_all_cached_prompts_db_instances()
        await stop_prompts_pending_close_worker()
        logger.info("App Shutdown: Prompts DB resources cleaned up")
    except guard_exceptions as exc:
        logger.exception(f"App Shutdown: Error shutting down Prompts DB resources: {exc}")


async def _shutdown_chat_workflows_resources(
    *,
    app: Any,
    guard_exceptions: tuple[type[BaseException], ...],
) -> None:
    try:
        from tldw_Server_API.app.api.v1.API_Deps.chat_workflows_deps import (
            shutdown_chat_workflows_deps,
        )

        shutdown_chat_workflows_deps(app)
        logger.info("App Shutdown: Chat workflows resources cleaned up")
    except guard_exceptions as exc:
        logger.exception(f"App Shutdown: Error shutting down chat workflows resources: {exc}")


async def _shutdown_provider_manager(
    *,
    heavy_startup_handles: Any | None,
    guard_exceptions: tuple[type[BaseException], ...],
) -> None:
    try:
        provider_manager = getattr(heavy_startup_handles, "provider_manager", None)
        if provider_manager is not None:
            await provider_manager.stop_health_checks()
            logger.info("App Shutdown: Provider manager stopped")
    except guard_exceptions as exc:
        logger.exception(f"App Shutdown: Error stopping provider manager: {exc}")


async def _shutdown_request_queue(
    *,
    heavy_startup_handles: Any | None,
    guard_exceptions: tuple[type[BaseException], ...],
) -> None:
    try:
        request_queue = getattr(heavy_startup_handles, "request_queue", None)
        if request_queue is not None:
            await request_queue.stop()
            logger.info("App Shutdown: Request queue stopped")
    except guard_exceptions as exc:
        logger.exception(f"App Shutdown: Error stopping request queue: {exc}")


async def _shutdown_local_llm_manager(
    *,
    app: Any,
    guard_exceptions: tuple[type[BaseException], ...],
    run_in_thread: Callable[[Callable[..., Any]], Awaitable[Any]],
) -> None:
    reconciler = getattr(getattr(app, "state", None), "llamacpp_runtime_reconciler", None)
    if reconciler is not None and hasattr(reconciler, "shutdown"):
        try:
            await reconciler.shutdown()
            app.state.llamacpp_runtime_reconciler = None
            logger.info("App Shutdown: llama.cpp runtime reconciler shutdown complete")
        except guard_exceptions as exc:
            logger.exception(f"App Shutdown: Error stopping llama.cpp runtime reconciler: {exc}")

    try:
        llm_manager = getattr(getattr(app, "state", None), "llm_manager", None)
        if llm_manager is not None and hasattr(llm_manager, "cleanup_on_exit"):
            await run_in_thread(llm_manager.cleanup_on_exit)
            logger.info("App Shutdown: Local LLM manager cleanup complete")
    except guard_exceptions as exc:
        logger.exception(f"App Shutdown: Error cleaning up local LLM manager: {exc}")


async def _shutdown_tts_service_components() -> None:
    from tldw_Server_API.app.core.TTS.tts_service_v2 import close_tts_service_v2
    from tldw_Server_API.app.core.TTS.voice_manager import shutdown_voice_manager

    await shutdown_voice_manager()
    await close_tts_service_v2()


async def _shutdown_tts_resource_manager_components() -> None:
    from tldw_Server_API.app.core.TTS.tts_resource_manager import (
        close_resource_manager as _close_tts_resource_manager,
    )

    await _close_tts_resource_manager()
