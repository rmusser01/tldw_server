from __future__ import annotations

import importlib
import sys

import pytest


pytestmark = pytest.mark.unit


def _import_shutdown_post_runtime_cleanup():
    sys.modules.pop("tldw_Server_API.app.services.shutdown_post_runtime_cleanup", None)
    return importlib.import_module("tldw_Server_API.app.services.shutdown_post_runtime_cleanup")


@pytest.mark.asyncio
async def test_shutdown_post_runtime_cleanup_runs_steps_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_post = _import_shutdown_post_runtime_cleanup()
    calls: list[str] = []

    async def _record_reset_media_db_cache(*, import_exceptions):
        assert import_exceptions == (LookupError,)
        calls.append("media-cache")

    async def _record_shutdown_content_backend(*, guard_exceptions):
        assert guard_exceptions == (RuntimeError,)
        calls.append("content-backend")

    async def _record_close_test_db_connections(*, test_db_instance_ref):
        assert test_db_instance_ref == "test-db"
        calls.append("test-db")

    async def _record_reset_jobs_acquire_gate(*, guard_exceptions):
        assert guard_exceptions == (RuntimeError,)
        calls.append("jobs-gate")

    monkeypatch.setattr(shutdown_post, "_reset_media_db_cache", _record_reset_media_db_cache)
    monkeypatch.setattr(shutdown_post, "_shutdown_content_backend", _record_shutdown_content_backend)
    monkeypatch.setattr(shutdown_post, "_close_test_db_connections", _record_close_test_db_connections)
    monkeypatch.setattr(shutdown_post, "_reset_jobs_acquire_gate", _record_reset_jobs_acquire_gate)

    await shutdown_post.shutdown_post_runtime_cleanup(
        test_db_instance_ref="test-db",
        startup_guard_exceptions=(RuntimeError,),
        import_exceptions=(LookupError,),
    )

    assert calls == ["media-cache", "content-backend", "test-db", "jobs-gate"]


@pytest.mark.asyncio
async def test_reset_media_db_cache_handles_import_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_post = _import_shutdown_post_runtime_cleanup()

    def _failing_reset_media_db_cache():
        raise LookupError("boom")

    monkeypatch.setattr(
        shutdown_post,
        "_reset_media_db_cache_service",
        _failing_reset_media_db_cache,
    )

    await shutdown_post._reset_media_db_cache(
        import_exceptions=(LookupError,),
    )


@pytest.mark.asyncio
async def test_shutdown_content_backend_handles_guard_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_post = _import_shutdown_post_runtime_cleanup()

    def _failing_shutdown_content_backend():
        raise RuntimeError("boom")

    monkeypatch.setattr(
        shutdown_post,
        "_shutdown_content_backend_service",
        _failing_shutdown_content_backend,
    )

    await shutdown_post._shutdown_content_backend(
        guard_exceptions=(RuntimeError,),
    )


@pytest.mark.asyncio
async def test_close_test_db_connections_closes_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del monkeypatch
    shutdown_post = _import_shutdown_post_runtime_cleanup()
    calls: list[str] = []

    class _TestDB:
        def close_all_connections(self) -> None:
            calls.append("close")

    await shutdown_post._close_test_db_connections(
        test_db_instance_ref=_TestDB(),
    )

    assert calls == ["close"]


@pytest.mark.asyncio
async def test_reset_jobs_acquire_gate_handles_guard_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_post = _import_shutdown_post_runtime_cleanup()

    def _failing_reset_jobs_acquire_gate(*, enabled):
        del enabled
        raise RuntimeError("boom")

    monkeypatch.setattr(
        shutdown_post,
        "_set_jobs_acquire_gate_service",
        _failing_reset_jobs_acquire_gate,
    )

    await shutdown_post._reset_jobs_acquire_gate(
        guard_exceptions=(RuntimeError,),
    )
