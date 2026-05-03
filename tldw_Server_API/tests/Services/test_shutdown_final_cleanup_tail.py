from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_shutdown_final_cleanup_tail_runs_helpers_in_order_and_returns_handles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.services import shutdown_final_cleanup_tail as shutdown_tail

    calls: list[tuple[str, dict[str, object]]] = []

    async def _record_cleanup(**kwargs):
        calls.append(("cleanup", kwargs))
        return SimpleNamespace()

    async def _record_post_runtime(**kwargs):
        calls.append(("post_runtime", kwargs))

    monkeypatch.setattr(shutdown_tail, "_shutdown_cleanup_timed_segments", _record_cleanup)
    monkeypatch.setattr(shutdown_tail, "_shutdown_post_runtime_cleanup", _record_post_runtime)

    handles = await shutdown_tail.shutdown_final_cleanup_tail(
        app="app",
        db_pool="db-pool",
        session_manager="session-manager",
        heavy_startup_handles="heavy-startup",
        in_pytest_for_db_pool_shutdown=True,
        in_pytest_for_tts_shutdown=True,
        import_exceptions=(ImportError,),
        startup_guard_exceptions=(RuntimeError,),
        test_db_instance_ref="test-db-ref",
        timed_shutdown_segment=lambda app, name: nullcontext(),
    )

    assert [name for name, _ in calls] == ["cleanup", "post_runtime"]
    assert calls[0][1]["db_pool"] == "db-pool"
    assert calls[0][1]["session_manager"] == "session-manager"
    assert calls[0][1]["heavy_startup_handles"] == "heavy-startup"
    assert calls[0][1]["timed_shutdown_segment"] is not None
    assert calls[1][1]["test_db_instance_ref"] == "test-db-ref"
    assert calls[1][1]["startup_guard_exceptions"] == (RuntimeError,)
    assert calls[1][1]["import_exceptions"] == (ImportError,)
    assert handles == SimpleNamespace()
    assert not hasattr(shutdown_tail, "_maybe_stop_authnz_scheduler")
