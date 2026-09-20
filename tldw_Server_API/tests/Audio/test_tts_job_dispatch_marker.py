"""Durable dispatch-marker regressions for long-form TTS jobs."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.TTS import tts_jobs_worker

_DISPATCH_MARKER = "tts_provider_dispatch_started"


def _new_manager(tmp_path) -> JobManager:
    return JobManager(tmp_path / "tts-dispatch-marker.db")


def _create_job(jm: JobManager, *, text: str) -> dict[str, Any]:
    return jm.create_job(
        domain="audio",
        queue="default",
        job_type="tts_longform",
        payload={
            "speech_request": {
                "model": "tts-1",
                "input": text,
                "voice": "alloy",
                "response_format": "mp3",
                "stream": False,
            },
            "provider_hint": "openai",
        },
        owner_user_id="1",
        max_retries=3,
    )


def _acquire(jm: JobManager, *, worker_id: str) -> dict[str, Any]:
    job = jm.acquire_next_job(
        domain="audio",
        queue="default",
        job_type="tts_longform",
        lease_seconds=30,
        worker_id=worker_id,
    )
    assert job is not None
    return job


def _expire(jm: JobManager, job_id: int) -> None:
    conn = jm._connect()
    try:
        conn.execute(
            "UPDATE jobs SET leased_until=DATETIME('now', '-10 minutes') WHERE id=?",
            (job_id,),
        )
        conn.commit()
    finally:
        conn.close()


def _patch_success_runtime(
    monkeypatch,
    tmp_path,
    *,
    jm: JobManager,
    dispatches: list[int],
    resolution_gate: asyncio.Event | None = None,
    expected_resolution_count: int = 1,
    events: list[tuple[str, dict[str, Any]]] | None = None,
) -> list[int]:
    resolution_calls = [0]

    async def _resolve_tts_byok(*, current_user, **_kwargs):
        resolution_calls[0] += 1
        if resolution_gate is not None:
            if resolution_calls[0] >= expected_resolution_count:
                resolution_gate.set()
            await resolution_gate.wait()
        return int(current_user.id), None, None

    class _Service:
        def generate_speech(self, request, **_kwargs):
            persisted = jm.get_job(int(request.input.rsplit(" ", 1)[-1]))
            assert persisted["progress_message"] == _DISPATCH_MARKER
            dispatches[0] += 1

            async def _chunks():
                yield b"one-dispatch"

            return _chunks()

    class _Collections:
        def resolve_output_storage_path(self, name):
            return name

        def create_output_artifact(self, **kwargs):
            return SimpleNamespace(
                id=501,
                storage_path=kwargs["storage_path"],
                format=kwargs["format_"],
            )

        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc, _tb):
            return False

    async def _get_service():
        return _Service()

    monkeypatch.setattr(tts_jobs_worker, "_resolve_tts_byok", _resolve_tts_byok)
    monkeypatch.setattr(tts_jobs_worker, "get_tts_service_v2", _get_service)
    monkeypatch.setattr(tts_jobs_worker, "JobManager", lambda: jm)
    monkeypatch.setattr(
        tts_jobs_worker.settings,
        "TTS_HISTORY_ENABLED",
        False,
        raising=False,
    )
    monkeypatch.setattr(
        tts_jobs_worker.DatabasePaths,
        "get_user_outputs_dir",
        lambda _user_id: tmp_path,
    )
    monkeypatch.setattr(
        tts_jobs_worker.CollectionsDatabase,
        "for_user",
        lambda user_id: _Collections(),
    )
    monkeypatch.setattr(
        tts_jobs_worker,
        "emit_job_event",
        lambda name, *, job, attrs: (
            events.append((name, dict(attrs))) if events is not None else None
        ),
    )
    return resolution_calls


@pytest.mark.asyncio
async def test_marked_reclaimed_job_blocks_before_credentials_or_adapter(
    monkeypatch,
    tmp_path,
) -> None:
    """A durable dispatch marker makes crash recovery terminal and secret-free."""

    jm = _new_manager(tmp_path)
    created = _create_job(jm, text="marked reclaim")
    first = _acquire(jm, worker_id="worker-before-crash")
    assert jm.renew_job_lease(
        int(created["id"]),
        seconds=30,
        worker_id=str(first["worker_id"]),
        lease_id=str(first["lease_id"]),
        progress_message=_DISPATCH_MARKER,
        enforce=True,
    )
    _expire(jm, int(created["id"]))
    reclaimed = _acquire(jm, worker_id="worker-after-crash")

    credential_calls = 0
    dispatches = 0

    async def _unexpected_resolution(**_kwargs):
        nonlocal credential_calls
        credential_calls += 1
        return 1, None, None

    async def _get_service():
        class _Service:
            def generate_speech(self, *_args, **_kwargs):
                nonlocal dispatches
                dispatches += 1
                raise AssertionError("marked reclaimed jobs cannot dispatch")

        return _Service()

    monkeypatch.setattr(tts_jobs_worker, "_resolve_tts_byok", _unexpected_resolution)
    monkeypatch.setattr(tts_jobs_worker, "get_tts_service_v2", _get_service)

    with pytest.raises(tts_jobs_worker.TTSJobError) as exc_info:
        await tts_jobs_worker._handle_tts_job(reclaimed)

    assert credential_calls == 0
    assert dispatches == 0
    assert exc_info.value.retryable is False
    assert exc_info.value.failure_code == "tts_replay_blocked"
    assert str(exc_info.value) == "TTS provider dispatch was already started"


@pytest.mark.asyncio
async def test_sqlite_stale_and_current_leases_only_dispatch_current_worker(
    monkeypatch,
    tmp_path,
) -> None:
    """A stale worker cannot mark or dispatch after a concurrent lease reclaim."""

    jm = _new_manager(tmp_path)
    created = _create_job(jm, text="concurrent lease")
    job_id = int(created["id"])
    stale = _acquire(jm, worker_id="stale-worker")
    _expire(jm, job_id)
    current = _acquire(jm, worker_id="current-worker")
    stale["payload"]["speech_request"]["input"] = f"job {job_id}"
    current["payload"]["speech_request"]["input"] = f"job {job_id}"

    dispatches = [0]
    resolution_gate = asyncio.Event()
    events: list[tuple[str, dict[str, Any]]] = []
    _patch_success_runtime(
        monkeypatch,
        tmp_path,
        jm=jm,
        dispatches=dispatches,
        resolution_gate=resolution_gate,
        expected_resolution_count=2,
        events=events,
    )
    renew_calls: list[tuple[str | None, str | None, bool]] = []
    original_renew = jm.renew_job_lease

    def _record_renew(job_id, **kwargs):
        ok = original_renew(job_id, **kwargs)
        renew_calls.append((kwargs.get("worker_id"), kwargs.get("lease_id"), ok))
        return ok

    monkeypatch.setattr(jm, "renew_job_lease", _record_renew)

    stale_result, current_result = await asyncio.gather(
        tts_jobs_worker._handle_tts_job(stale),
        tts_jobs_worker._handle_tts_job(current),
        return_exceptions=True,
    )

    assert isinstance(stale_result, tts_jobs_worker.TTSJobError)
    assert stale_result.retryable is False
    assert stale_result.failure_code == "tts_dispatch_lease_lost"
    assert isinstance(current_result, dict)
    assert dispatches == [1]
    assert len(renew_calls) == 2
    assert [ok for _worker, _lease, ok in renew_calls].count(True) == 1
    assert jm.get_job(job_id)["progress_message"] == _DISPATCH_MARKER
    assert any(
        name == "job.progress" and attrs.get("progress_message") == "tts_completed"
        for name, attrs in events
    )


@pytest.mark.asyncio
async def test_current_lease_marks_once_before_dispatch_and_keeps_marker(
    monkeypatch,
    tmp_path,
) -> None:
    """The current lease durably marks once before the adapter boundary."""

    jm = _new_manager(tmp_path)
    created = _create_job(jm, text="current lease")
    job_id = int(created["id"])
    current = _acquire(jm, worker_id="current-worker")
    current["payload"]["speech_request"]["input"] = f"job {job_id}"
    original_payload = json.loads(json.dumps(current["payload"]))
    dispatches = [0]
    _patch_success_runtime(
        monkeypatch,
        tmp_path,
        jm=jm,
        dispatches=dispatches,
    )
    renew_calls = 0
    original_renew = jm.renew_job_lease

    def _record_renew(*args, **kwargs):
        nonlocal renew_calls
        renew_calls += 1
        return original_renew(*args, **kwargs)

    monkeypatch.setattr(jm, "renew_job_lease", _record_renew)

    result = await tts_jobs_worker._handle_tts_job(current)

    assert result["output_id"] == 501
    assert dispatches == [1]
    assert renew_calls == 1
    assert jm.get_job(job_id)["progress_message"] == _DISPATCH_MARKER
    assert current["payload"] == original_payload


@pytest.mark.asyncio
async def test_unmarked_reclaimed_attempt_can_mark_and_dispatch_once(
    monkeypatch,
    tmp_path,
) -> None:
    """A crash before dispatch marking remains recoverable within the retry budget."""

    jm = _new_manager(tmp_path)
    created = _create_job(jm, text="unmarked reclaim")
    job_id = int(created["id"])
    _acquire(jm, worker_id="worker-before-predispatch-crash")
    _expire(jm, job_id)
    reclaimed = _acquire(jm, worker_id="worker-after-predispatch-crash")
    reclaimed["payload"]["speech_request"]["input"] = f"job {job_id}"
    dispatches = [0]
    _patch_success_runtime(
        monkeypatch,
        tmp_path,
        jm=jm,
        dispatches=dispatches,
    )

    result = await tts_jobs_worker._handle_tts_job(reclaimed)

    assert result["output_id"] == 501
    assert dispatches == [1]
    assert int(reclaimed["retry_count"]) == 1
    assert jm.get_job(job_id)["progress_message"] == _DISPATCH_MARKER
