"""Opt-in reset fixtures for process-global singletons that lack a per-test hook.

The audit inventory (audits/2026-07-04-singleton-inventory.md) found the
Embeddings drain singletons have only an ``atexit`` shutdown — no per-test
reset — so a suite that drains them could hand drained state to the next
(#2581). These fixtures let a suite guarantee a clean slate.

Kept OPT-IN (not autouse): the singleton guard currently reports zero leaks on
the sampled lanes, so a suite-wide autouse reset would add teardown cost for no
demonstrated benefit and risk resetting state a test legitimately set up.
Promote to autouse only if the guard surfaces a real leak on a lane.
"""
from __future__ import annotations

import asyncio

import pytest
from loguru import logger

# per-singleton await budget so a hung shutdown cannot stall the loop
_SHUTDOWN_TIMEOUT_S = 5.0


async def _reset_embeddings_singletons() -> None:
    """Best-effort null of the Embeddings drain singletons.

    Awaits the async shutdowns directly (never the atexit-oriented
    ``_shutdown_async_embedding_service_sync`` helper, which blocks on
    ``run_coroutine_threadsafe(...).result()`` and would stall a running event
    loop). Failures are logged at debug, not silently swallowed.
    """
    try:
        from tldw_Server_API.app.core.Embeddings import connection_pool as cp

        await asyncio.wait_for(cp.cleanup_connection_pools(), timeout=_SHUTDOWN_TIMEOUT_S)
    except Exception as exc:
        logger.debug(f"reset_embeddings_singletons: pool cleanup skipped: {exc!r}")
    try:
        from tldw_Server_API.app.core.Embeddings import async_embeddings as ae

        service = ae._async_service_fallback
        if service is not None:
            await asyncio.wait_for(service.shutdown(), timeout=_SHUTDOWN_TIMEOUT_S)
        ae._async_service_fallback = None
    except Exception as exc:
        logger.debug(f"reset_embeddings_singletons: async service reset skipped: {exc!r}")
    try:
        from tldw_Server_API.app.core.Embeddings import request_batching as rb

        batcher = rb._batcher_fallback
        if batcher is not None:
            await asyncio.wait_for(batcher.shutdown(), timeout=_SHUTDOWN_TIMEOUT_S)
        rb._batcher_fallback = None
    except Exception as exc:
        logger.debug(f"reset_embeddings_singletons: batcher reset skipped: {exc!r}")


@pytest.fixture
async def reset_embeddings_singletons():
    """Reset the Embeddings drain singletons before and after the test."""
    await _reset_embeddings_singletons()
    yield
    await _reset_embeddings_singletons()
