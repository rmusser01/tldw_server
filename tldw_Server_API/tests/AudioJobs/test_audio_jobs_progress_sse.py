from __future__ import annotations

import asyncio
import json

import pytest

from tldw_Server_API.app.api.v1.endpoints.audio import audio_jobs
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.Jobs.manager import JobManager


@pytest.mark.asyncio
async def test_audio_job_progress_stream_returns_public_attrs_shape(monkeypatch, tmp_path):
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")
    monkeypatch.setenv("JOBS_EVENTS_POLL_INTERVAL", "0.05")
    jm = JobManager(db_path=tmp_path / "jobs.db")
    job = jm.create_job(
        domain="audio",
        queue="default",
        job_type="audio_download",
        payload={"url": "https://example.com/audio.mp3"},
        owner_user_id="1",
    )

    response = await audio_jobs.stream_audio_job_progress(
        job_id=int(job["id"]),
        current_user=User(
            id=1,
            username="owner",
            email="owner@example.com",
            role="user",
            is_active=True,
        ),
        jm=jm,
        request=None,
        after_id=0,
    )
    observed_snapshot = None
    observed_created = None

    iterator = response.body_iterator
    try:
        for _ in range(10):
            line = await asyncio.wait_for(anext(iterator), timeout=3)
            line_text = line.decode("utf-8", errors="ignore") if isinstance(line, bytes) else str(line)
            if not line_text.startswith("data:"):
                continue
            payload = line_text.split(":", 1)[1].strip()
            if not payload or payload == "[DONE]":
                continue
            data = json.loads(payload)
            if not isinstance(data, dict):
                continue
            if data.get("event") == "job.snapshot":
                observed_snapshot = data
                continue
            if data.get("event") == "job.created":
                observed_created = data
                break
    finally:
        await iterator.aclose()

    assert observed_snapshot is not None
    assert isinstance(observed_snapshot.get("attrs"), dict)
    assert observed_created is not None
    assert isinstance(observed_created.get("attrs"), dict)
    assert observed_created["attrs"]["owner_user_id"] == "1"
    assert "attrs_json" not in observed_created
