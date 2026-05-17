"""
Heavy startup initialization helpers extracted from the application lifespan.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from tldw_Server_API.app.core.testing import is_truthy as _shared_is_truthy

_STARTUP_GUARD_EXCEPTIONS = (
    AttributeError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


@dataclass
class HeavyStartupHandles:
    """Long-lived startup resources that still require coordinated shutdown."""

    mcp_server: Any | None = None
    provider_manager: Any | None = None
    request_queue: Any | None = None


async def start_heavy_initializations(
    app: Any,
    *,
    route_enabled: Callable[[str], bool],
    defer_heavy: bool,
) -> HeavyStartupHandles:
    """Run heavy startup inline or schedule it in the background."""
    handles = HeavyStartupHandles()
    if defer_heavy:
        bg_tasks = getattr(app.state, "bg_tasks", None)
        if bg_tasks is None:
            bg_tasks = {}
            app.state.bg_tasks = bg_tasks
        bg_tasks["deferred_startup"] = asyncio.create_task(
            run_heavy_initializations(
                app,
                handles=handles,
                route_enabled=route_enabled,
                deferred=True,
            )
        )
    else:
        await run_heavy_initializations(
            app,
            handles=handles,
            route_enabled=route_enabled,
            deferred=False,
        )
    return handles


async def run_heavy_initializations(
    app: Any,
    *,
    handles: HeavyStartupHandles,
    route_enabled: Callable[[str], bool],
    deferred: bool,
) -> None:
    """Run the existing heavy startup sequence in the current order."""
    if deferred:
        logger.info("Deferred startup: beginning non-critical initializations in background")
    await _init_local_llm_manager(app, route_enabled, deferred=deferred)
    handles.mcp_server = await _init_mcp_server(app, deferred=deferred)
    handles.provider_manager = await _init_provider_manager(deferred=deferred)
    handles.request_queue = await _init_request_queue(deferred=deferred)
    await _init_rate_limiter(deferred=deferred)
    await _init_tts_service(deferred=deferred)
    await _init_chunking_templates(deferred=deferred)
    await _init_embeddings_dim_check(deferred=deferred)
    if deferred:
        logger.info("Deferred startup: completed non-critical initializations")


async def _init_local_llm_manager(
    app: Any,
    route_enabled: Callable[[str], bool],
    *,
    deferred: bool,
) -> None:
    try:
        if getattr(app.state, "llm_manager", None) is not None:
            return

        try:
            _llm_routes_enabled = route_enabled("llamacpp") or route_enabled("llm")
        except _STARTUP_GUARD_EXCEPTIONS:
            _llm_routes_enabled = True
        if not _llm_routes_enabled:
            logger.debug("Local LLM inference manager skipped (llm/llamacpp routes disabled)")
            return

        from tldw_Server_API.app.core.config import get_llamacpp_handler_config
        from tldw_Server_API.app.core.Local_LLM import LLMInferenceManager, LLMManagerConfig

        _llama_cfg = get_llamacpp_handler_config()
        cfg_kwargs = {}
        if _llama_cfg:
            cfg_kwargs["llamacpp"] = _llama_cfg

        manager = await asyncio.to_thread(LLMInferenceManager, LLMManagerConfig(**cfg_kwargs))
        app.state.llm_manager = manager
        await _init_llamacpp_runtime_reconciler(app, manager, deferred=deferred)
        try:
            from tldw_Server_API.app.api.v1.endpoints import llamacpp as _llamacpp_module

            _llamacpp_module.llm_manager = manager
        except _STARTUP_GUARD_EXCEPTIONS as _llm_ep_err:
            logger.debug(f"LLM manager initialized but not injected into llama.cpp endpoints: {_llm_ep_err}")
        logger.info(
            ("Deferred startup: " if deferred else "App Startup: ")
            + "Local LLM inference manager initialized"
        )
    except _STARTUP_GUARD_EXCEPTIONS as _llm_init_err:
        if deferred:
            logger.debug(f"Deferred startup: local LLM manager skipped/failed: {_llm_init_err}")
        else:
            logger.warning(
                "Local LLM inference manager not initialized; llama.cpp endpoints will return 503: "
                f"{_llm_init_err}"
            )


async def _init_llamacpp_runtime_reconciler(app: Any, manager: Any, *, deferred: bool) -> None:
    supervisor = getattr(manager, "llamacpp_supervisor", None)
    if supervisor is None:
        return
    try:
        from tldw_Server_API.app.core.Local_LLM import llamacpp_runtime_reconciler as reconciler_module

        reconciler = reconciler_module.LlamaCppRuntimeReconciler(supervisor)
        app.state.llamacpp_runtime_reconciler = reconciler
        await reconciler.reconcile_startup()
        logger.info(
            ("Deferred startup: " if deferred else "App Startup: ")
            + "llama.cpp runtime reconciliation complete"
        )
    except Exception as exc:  # noqa: BLE001 - autostart reconciliation must not block API startup.
        logger.warning(
            ("Deferred startup: " if deferred else "App Startup: ")
            + f"llama.cpp runtime reconciliation skipped/failed: {exc}"
        )


async def _init_mcp_server(app: Any, *, deferred: bool) -> Any | None:
    try:
        from tldw_Server_API.app.core.MCP_unified import get_mcp_server

        mcp_server = get_mcp_server()
        if not deferred:
            logger.info("App Startup: Initializing MCP Unified server...")
        await mcp_server.initialize()
        logger.info(
            ("Deferred startup: " if deferred else "App Startup: ")
            + "MCP Unified server initialized successfully"
        )
        return mcp_server
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        if deferred:
            logger.debug(f"Deferred startup: MCP Unified server skipped/failed: {exc}")
        else:
            logger.exception(f"App Startup: Failed to initialize MCP Unified server: {exc}")
            logger.warning("Ensure MCP_JWT_SECRET and MCP_API_KEY_SALT environment variables are set")
        return None


async def _init_provider_manager(*, deferred: bool) -> Any | None:
    try:
        from tldw_Server_API.app.core.Chat.provider_manager import initialize_provider_manager
        from tldw_Server_API.app.core.LLM_Calls.adapter_registry import get_registry

        providers = get_registry().list_providers()
        provider_manager = initialize_provider_manager(
            providers, primary_provider=providers[0] if providers else None
        )
        await provider_manager.start_health_checks()
        if deferred:
            logger.info(f"Deferred startup: Provider manager ready ({len(providers)} providers)")
        else:
            logger.info(f"App Startup: Provider manager initialized with {len(providers)} providers")
        return provider_manager
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        if deferred:
            logger.debug(f"Deferred startup: provider manager skipped/failed: {exc}")
        else:
            logger.exception(f"App Startup: Failed to initialize provider manager: {exc}")
        return None


async def _init_request_queue(*, deferred: bool) -> Any | None:
    try:
        from tldw_Server_API.app.core.Chat.request_queue import initialize_request_queue
        from tldw_Server_API.app.core.config import load_comprehensive_config

        cfg = load_comprehensive_config()
        chat_cfg = {}
        if cfg and cfg.has_section("Chat-Module"):
            chat_cfg = dict(cfg.items("Chat-Module"))
        queued_execution_enabled = False
        try:
            env_queued = os.getenv("CHAT_QUEUED_EXECUTION")
            if env_queued is not None:
                queued_execution_enabled = _shared_is_truthy(env_queued)
            else:
                queued_execution_enabled = _shared_is_truthy(str(chat_cfg.get("queued_execution", "False")))
        except _STARTUP_GUARD_EXCEPTIONS:
            queued_execution_enabled = False
        if queued_execution_enabled:
            request_queue = initialize_request_queue(
                max_queue_size=int(chat_cfg.get("max_queue_size", 100)),
                max_concurrent=int(chat_cfg.get("max_concurrent_requests", 10)),
                global_rate_limit=int(chat_cfg.get("rate_limit_per_minute", 60)),
                per_client_rate_limit=int(chat_cfg.get("rate_limit_per_conversation_per_minute", 20)),
            )
            await request_queue.start(num_workers=4)
            if deferred:
                logger.info("Deferred startup: Request queue online")
            else:
                logger.info("App Startup: Request queue initialized with 4 workers")
            return request_queue
        if not deferred:
            logger.info("App Startup: Request queue disabled (QUEUED_EXECUTION is off)")
        return None
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        if deferred:
            logger.debug(f"Deferred startup: request queue skipped/failed: {exc}")
        else:
            logger.exception(f"App Startup: Failed to initialize request queue: {exc}")
        return None


async def _init_rate_limiter(*, deferred: bool) -> None:
    try:
        from tldw_Server_API.app.core.config import rg_enabled as _rg_enabled_flag

        if _rg_enabled_flag(False):
            logger.info(
                ("Deferred startup: " if deferred else "App Startup: ")
                + "Rate limiter skipped (RG enabled)"
            )
            return
        from tldw_Server_API.app.core.Chat.rate_limiter import RateLimitConfig, initialize_rate_limiter
        from tldw_Server_API.app.core.config import load_comprehensive_config

        cfg = load_comprehensive_config()
        chat_cfg = {}
        if cfg and cfg.has_section("Chat-Module"):
            chat_cfg = dict(cfg.items("Chat-Module"))
        rl_cfg = RateLimitConfig(
            global_rpm=int(chat_cfg.get("rate_limit_per_minute", 60)),
            per_user_rpm=int(chat_cfg.get("rate_limit_per_user_per_minute", 20)),
            per_conversation_rpm=int(chat_cfg.get("rate_limit_per_conversation_per_minute", 10)),
            per_user_tokens_per_minute=int(chat_cfg.get("rate_limit_tokens_per_minute", 10000)),
        )
        initialize_rate_limiter(rl_cfg)
        logger.info(
            ("Deferred startup: " if deferred else "App Startup: ")
            + "Rate limiter "
            + ("online" if deferred else "initialized")
        )
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        if deferred:
            logger.debug(f"Deferred startup: rate limiter skipped/failed: {exc}")
        else:
            logger.exception(f"App Startup: Failed to initialize rate limiter: {exc}")


async def _init_tts_service(*, deferred: bool) -> None:
    try:
        from tldw_Server_API.app.core.config import load_comprehensive_config_with_tts
        from tldw_Server_API.app.core.TTS.tts_service_v2 import get_tts_service_v2
        from tldw_Server_API.app.core.TTS.voice_manager import init_voice_manager

        cfg_obj = load_comprehensive_config_with_tts()
        tts_cfg_dict = cfg_obj.get_tts_config() if hasattr(cfg_obj, "get_tts_config") else None
        await get_tts_service_v2(config=tts_cfg_dict)
        await init_voice_manager()
        logger.info(
            ("Deferred startup: " if deferred else "App Startup: ")
            + "TTS service "
            + ("ready" if deferred else "initialized successfully")
        )
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        if deferred:
            logger.debug(f"Deferred startup: TTS skipped/failed: {exc}")
        else:
            logger.exception(f"App Startup: Failed to initialize TTS service: {exc}")
            logger.warning("TTS functionality will be unavailable")


async def _init_chunking_templates(*, deferred: bool) -> None:
    try:
        from tldw_Server_API.app.core.Chunking.template_initialization import ensure_templates_initialized

        ok = ensure_templates_initialized()
        if ok:
            logger.info(
                ("Deferred startup: " if deferred else "App Startup: ")
                + "Chunking templates "
                + ("ready" if deferred else "initialized successfully")
            )
        else:
            if deferred:
                logger.debug("Deferred startup: Chunking templates incomplete")
            else:
                logger.warning("App Startup: Chunking templates initialization incomplete")
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        if deferred:
            logger.debug(f"Deferred startup: chunking templates skipped/failed: {exc}")
        else:
            logger.exception(f"App Startup: Failed to initialize chunking templates: {exc}")


async def _init_embeddings_dim_check(*, deferred: bool) -> None:
    try:
        enabled = os.getenv("EMBEDDINGS_STARTUP_DIM_CHECK_ENABLED", "false").lower() in {
            "true",
            "1",
            "yes",
            "y",
            "on",
        }
        if not enabled:
            return
        strict_mode = os.getenv("EMBEDDINGS_DIM_CHECK_STRICT", "false").lower() in {
            "true",
            "1",
            "yes",
            "y",
            "on",
        }
        if not deferred:
            logger.info("App Startup: Running embeddings dimension sanity check (opt-in)")

        from tldw_Server_API.app.core.config import settings as _emb_settings
        from tldw_Server_API.app.core.Embeddings.ChromaDB_Library import ChromaDBManager

        def _check_user(user_id: str) -> list[tuple[str, int, int, str]]:
            mismatches: list[tuple[str, int, int, str]] = []
            mgr = ChromaDBManager(user_id=user_id, user_embedding_config=_emb_settings)
            client = getattr(mgr, "client", None)
            list_fn = getattr(client, "list_collections", None)
            collections = list_fn() if callable(list_fn) else []
            for col in collections:
                try:
                    name = getattr(col, "name", None) or (col.get("name") if isinstance(col, dict) else None)
                    if not name:
                        continue
                    get_fn = getattr(client, "get_collection", None)
                    collection = get_fn(name=name) if callable(get_fn) else col
                    meta = getattr(collection, "metadata", None) or {}
                    expected = None
                    if isinstance(meta, dict) and meta.get("embedding_dimension"):
                        try:
                            expected = int(meta.get("embedding_dimension"))
                        except _STARTUP_GUARD_EXCEPTIONS:
                            expected = None
                    actual = None
                    if hasattr(collection, "get") and callable(collection.get):
                        try:
                            res = collection.get(limit=1, include=["embeddings"])
                            embs = res.get("embeddings") if isinstance(res, dict) else None
                            if embs and len(embs) > 0:
                                first = embs[0]
                                if first and hasattr(first, "__len__"):
                                    actual = len(first)
                        except _STARTUP_GUARD_EXCEPTIONS:
                            pass
                    if expected is not None and actual is not None and expected != actual:
                        mismatches.append((name, expected, actual, user_id))
                except _STARTUP_GUARD_EXCEPTIONS:
                    pass
            with suppress(_STARTUP_GUARD_EXCEPTIONS):
                mgr.close()
            return mismatches

        auth_mode = str(_emb_settings.get("AUTH_MODE", os.getenv("AUTH_MODE", "single_user")))
        mismatches: list[tuple[str, int, int, str]] = []
        if auth_mode == "multi_user":
            base: Path = _emb_settings.get("USER_DB_BASE_DIR")
            if base and Path(base).exists():
                for entry in Path(base).iterdir():
                    if entry.is_dir():
                        with suppress(_STARTUP_GUARD_EXCEPTIONS):
                            mismatches.extend(_check_user(entry.name))
            elif not deferred:
                logger.warning(
                    "Embeddings dimension check: USER_DB_BASE_DIR missing or does not exist in multi_user mode"
                )
        else:
            user_id = str(_emb_settings.get("SINGLE_USER_FIXED_ID", "1"))
            mismatches.extend(_check_user(user_id))

        if mismatches:
            for name, expected, actual, user_id in mismatches:
                logger.error(
                    ("Deferred startup: " if deferred else "")
                    + f"Embeddings dimension mismatch{' (deferred)' if deferred else ' at startup'} (user={user_id}) in collection '{name}': expected={expected}, actual={actual}"
                )
            if strict_mode:
                raise RuntimeError("EMBEDDINGS_STARTUP_DIM_CHECK_FAILED")
        else:
            logger.info(
                ("Deferred startup: " if deferred else "")
                + (
                    "Embeddings dimension check OK"
                    if deferred
                    else "Embeddings dimension sanity check: OK (no mismatches)"
                )
            )
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        if isinstance(exc, RuntimeError) and str(exc) == "EMBEDDINGS_STARTUP_DIM_CHECK_FAILED":
            raise
        if deferred:
            logger.debug(f"Deferred startup: embeddings dimension check skipped/failed: {exc}")
        else:
            logger.exception(f"Embeddings dimension sanity check failed: {exc}")
