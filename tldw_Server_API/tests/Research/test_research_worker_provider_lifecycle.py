"""Standalone Research must initialize and drain its provider authority on its own loop."""

import asyncio

import pytest

from tldw_Server_API.app.core.AuthNZ import llm_provider_overrides as overrides
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Research import jobs_worker

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize("outcome", ["completed", "error", "cancelled", "store-unavailable"])
async def test_standalone_research_owns_provider_policy_lifecycle(monkeypatch, tmp_path, outcome):
    events = []
    overrides.set_llm_provider_overrides_cache_for_tests({}, healthy=False)

    class Pool:
        async def close(self):
            assert overrides._OVERRIDE_REFRESH_SERVICE_TASK is None  # nosec B101 -- Refresh tasks must drain before their pool.
            events.append("close")

    pool = Pool()

    async def get_pool():
        events.append("pool")
        return pool

    class Repository:
        def __init__(self, actual):
            assert actual is pool  # nosec B101 -- The worker must use its own initialized pool.

        async def list_overrides(self):
            events.append("policy")
            if outcome == "store-unavailable":
                raise RuntimeError("controlled unavailable store")
            return []

    class SDK:
        def __init__(self, manager, config):
            pass

        async def run(self, **kwargs):
            # This is the same fail-closed lookup the provider adapter makes on first use.
            assert overrides.get_llm_provider_override("llama.cpp") is None  # nosec B101 -- Empty healthy policy permits normal fallback; cold state must fail.
            events.append("run")
            if outcome == "error":
                raise RuntimeError("controlled worker failure")
            if outcome == "cancelled":
                raise asyncio.CancelledError

    monkeypatch.setattr(jobs_worker, "get_db_pool", get_pool)
    monkeypatch.setattr(jobs_worker, "_jobs_manager", lambda: JobManager(tmp_path / "jobs.db", backend="sqlite"))
    monkeypatch.setattr(jobs_worker, "WorkerSDK", SDK)
    monkeypatch.setattr(overrides, "AuthnzLLMProviderOverridesRepo", Repository)
    try:
        if outcome == "error":
            with pytest.raises(RuntimeError, match="controlled worker failure"):
                await jobs_worker.main()
        elif outcome == "cancelled":
            with pytest.raises(asyncio.CancelledError):
                await jobs_worker.main()
        elif outcome == "store-unavailable":
            with pytest.raises(overrides.LLMProviderOverridesRefreshError):
                await jobs_worker.main()
        else:
            await jobs_worker.main()
        assert events == (
            ["pool", "policy", "close"] if outcome == "store-unavailable" else ["pool", "policy", "run", "close"]
        )  # nosec B101 -- Verify first-use ordering and every exit's cleanup.
    finally:
        await overrides.shutdown_llm_provider_override_recovery()
