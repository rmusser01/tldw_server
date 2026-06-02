from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_run_shutdown_post_worker_non_worker_cleanup_runs_personalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.services import shutdown_post_worker_services as shutdown_services

    calls: list[dict[str, object]] = []

    async def _record_personalization(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        shutdown_services,
        "_shutdown_personalization_consolidation",
        _record_personalization,
    )

    handles = await shutdown_services.run_shutdown_post_worker_non_worker_cleanup(
        guard_exceptions=(RuntimeError,),
    )

    assert isinstance(handles, shutdown_services.PostWorkerNonWorkerCleanupHandles)
    assert calls == [{"guard_exceptions": (RuntimeError,)}]


@pytest.mark.asyncio
async def test_run_shutdown_post_worker_non_worker_cleanup_reraises_non_guarded_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.services import shutdown_post_worker_services as shutdown_services

    async def _raise_personalization(**_kwargs):
        raise ValueError("unexpected cleanup failure")

    monkeypatch.setattr(
        shutdown_services,
        "_shutdown_personalization_consolidation",
        _raise_personalization,
    )

    with pytest.raises(ValueError, match="unexpected cleanup failure"):
        await shutdown_services.run_shutdown_post_worker_non_worker_cleanup(
            guard_exceptions=(RuntimeError,),
        )


@pytest.mark.asyncio
async def test_run_shutdown_post_worker_non_worker_cleanup_guards_personalization_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.services import shutdown_post_worker_services as shutdown_services

    async def _raise_personalization(**_kwargs):
        raise RuntimeError("skip cleanup")

    monkeypatch.setattr(
        shutdown_services,
        "_shutdown_personalization_consolidation",
        _raise_personalization,
    )

    handles = await shutdown_services.run_shutdown_post_worker_non_worker_cleanup(
        guard_exceptions=(RuntimeError,),
    )

    assert isinstance(handles, shutdown_services.PostWorkerNonWorkerCleanupHandles)
