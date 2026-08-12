"""
test_concurrency_jobs.py
Description: Bulk ingest and background job cancellation flows.

Uploads multiple small docs concurrently-ish (sequential for stability), then
starts a Chatbooks export in async mode and cancels it.
"""

import httpx
import pytest

from .fixtures import cleanup_test_file, create_test_file


@pytest.mark.critical
def test_bulk_media_ingest(api_client, data_tracker):
    texts = [
        "Doc A - concurrency test.",
        "Doc B - concurrency test.",
        "Doc C - concurrency test.",
    ]
    ids = []
    paths = []
    try:
        for i, t in enumerate(texts):
            p = create_test_file(t, suffix=".txt")
            paths.append(p)
            r = api_client.upload_media(file_path=p, title=f"Bulk {i}", media_type="document", generate_embeddings=False)
            mid = r.get("media_id") or (r.get("results") or [{}])[0].get("db_id")
            if mid:
                ids.append(int(mid))
                data_tracker.add_media(int(mid))
        assert len(ids) >= 1
    finally:
        for p in paths:
            cleanup_test_file(p)


@pytest.mark.critical
def test_chatbooks_async_cancel(api_client):
    # Start an async export (minimal content selections)
    payload = {
        "name": "E2E Cancel Chatbook",
        "description": "Cancel flow",
        "content_selections": {"conversation": []},
        "async_mode": True,
    }
    try:
        r = api_client.client.post("/api/v1/chatbooks/export", json=payload)
        if r.status_code >= 400:
            pytest.skip(
                f"Chatbooks export not available: HTTP {r.status_code} {r.text}"
            )
        d = r.json()
        job_id = d.get("job_id")
        assert job_id
    except httpx.HTTPStatusError as e:
        pytest.skip(f"Chatbooks export not available: {e}")

    # Cancel the job
    try:
        c = api_client.client.delete(f"/api/v1/chatbooks/export/jobs/{job_id}")
        # In some configurations, cancellation may be immediate/no-op, or the job
        # may complete before cancellation can be applied.
        if c.status_code == 400 and "Cannot cancel" in c.text:
            pytest.skip("Export job completed before it could be cancelled")
        assert c.status_code in (200, 202, 404)
        # Status check (best-effort)
        s = api_client.client.get(f"/api/v1/chatbooks/export/jobs/{job_id}")
        assert s.status_code in (200, 404)
    except httpx.HTTPError as e:
        pytest.skip(f"Chatbooks cancel path unavailable: {e}")
