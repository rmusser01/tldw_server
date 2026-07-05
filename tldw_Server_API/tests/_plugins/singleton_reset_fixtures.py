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

import pytest


async def _reset_embeddings_singletons() -> None:
    """Best-effort null of the Embeddings drain singletons."""
    try:
        from tldw_Server_API.app.core.Embeddings import connection_pool as cp

        await cp.cleanup_connection_pools()
    except Exception:
        pass
    try:
        from tldw_Server_API.app.core.Embeddings import async_embeddings as ae

        ae._shutdown_async_embedding_service_sync()
    except Exception:
        pass
    try:
        from tldw_Server_API.app.core.Embeddings import request_batching as rb

        batcher = rb._batcher_fallback
        if batcher is not None:
            await batcher.shutdown()
        rb._batcher_fallback = None
    except Exception:
        pass


@pytest.fixture
async def reset_embeddings_singletons():
    """Reset the Embeddings drain singletons before and after the test."""
    await _reset_embeddings_singletons()
    yield
    await _reset_embeddings_singletons()
