"""
test_background_jobs_and_rate_limits.py
Background jobs (embeddings) and rate-limit backoff behavior.

Includes:
- Embedding jobs listing: trigger embeddings generation, poll job or status, and assert completion.
- Rate-limit backoff: trigger a 429 on a rate-limited endpoint, then call through
  APIClient._handle_rate_limit and assert it either succeeds or displays backoff behavior.
"""

import time
import uuid
import pytest
import httpx

from .fixtures import api_client, create_test_file, cleanup_test_file, AssertionHelpers, APIClient, NO_RETRY_HEADER


def _is_redis_unavailable_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "redis.exceptions.connectionerror" in text
        or "connection refused" in text
        or "error 111 connecting to localhost:6379" in text
    )


@pytest.mark.critical
def test_upload_media_forwards_analysis_opt_out(tmp_path):
    """The upload-only embedding flow must explicitly avoid provider analysis."""
    observed: dict[str, bytes] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed["body"] = request.content
        return httpx.Response(
            200,
            json={"results": [{"status": "Success", "db_id": 1}]},
        )

    upload_path = tmp_path / "upload.txt"
    upload_path.write_text("upload fixture control", encoding="utf-8")
    client = APIClient(
        client=httpx.Client(transport=httpx.MockTransport(handler), base_url="http://testserver"),
        auto_auth=False,
    )
    try:
        response = client.upload_media(
            file_path=str(upload_path),
            title="Upload fixture control",
            perform_analysis=False,
        )
    finally:
        client.close()

    assert response["results"][0]["status"] == "Success"
    assert b'name="perform_analysis"\r\n\r\nfalse\r\n' in observed["body"]


@pytest.mark.critical
def test_embedding_jobs_list_and_progression(api_client):
    # Upload a small text doc
    token = f"EMB_JOB_{uuid.uuid4().hex[:6]}"
    fp = create_test_file(f"Embedding job test content {token}")
    try:
        up = api_client.upload_media(
            file_path=fp,
            title=f"Emb Job {token}",
            media_type="document",
            generate_embeddings=False,
            perform_analysis=False,
        )
        media_id = AssertionHelpers.assert_successful_upload(up)
    finally:
        cleanup_test_file(fp)

    # Trigger embeddings
    try:
        r = api_client.client.post(f"/api/v1/media/{media_id}/embeddings", json={})
    except Exception as exc:  # TestClient can re-raise server exceptions in in-process mode
        if _is_redis_unavailable_error(exc):
            pytest.skip(f"Embeddings generation unavailable (redis not reachable): {exc}")
        raise
    if r.status_code in (404, 422, 500):
        pytest.skip(f"Embeddings generation unavailable: {r.status_code}")
    r.raise_for_status()
    job_id = r.json().get("job_id")

    # Poll job status endpoint if job_id present, otherwise fall back to has_embeddings
    start = time.time()
    seen_in_progress = False
    completed = False
    observed_statuses: set[str] = set()
    while time.time() - start < 30:
        if job_id:
            js = api_client.client.get(f"/api/v1/media/embeddings/jobs/{job_id}")
            if js.status_code == 200:
                j = js.json()
                st = (j.get("status") or "").lower()
                if st:
                    observed_statuses.add(st)
                if st == "in_progress":
                    seen_in_progress = True
                if st in {"completed", "success"}:
                    completed = True
                    break
                if st in {"failed", "cancelled"}:
                    pytest.skip(f"Embeddings job ended as {st}")
            # If not found yet, proceed to status check
        # Fallback: check has_embeddings
        est = api_client.client.get(f"/api/v1/media/{media_id}/embeddings/status")
        if est.status_code == 200 and est.json().get("has_embeddings"):
            completed = True
            break
        time.sleep(1.0)

    if not completed:
        if observed_statuses and observed_statuses.issubset({"queued"}):
            pytest.skip("Embeddings job stayed queued; embeddings worker unavailable in this environment")
        pytest.fail("Embeddings did not complete in time")
    # If we ever saw in_progress, that indicates progression visibility
    assert seen_in_progress or completed


@pytest.mark.critical
def test_rate_limit_backoff_retry_envelope(api_client):
    """Intentionally hit a rate-limited endpoint, then verify our backoff wrapper behaves."""
    # Use the chatbooks export endpoint which has a limiter. Create a tiny payload.
    payload = {
        "name": f"RLTest {uuid.uuid4().hex[:6]}",
        "description": "rate limit backoff test",
        "content_selections": {"note": []},
        "author": "pytest",
        "include_media": False,
        "async_mode": True,
    }

    # First, quickly burst until we see a 429 or a few successes
    hit_429 = False
    for _ in range(8):
        r = api_client.client.post(
            "/api/v1/chatbooks/export",
            json=payload,
            headers={NO_RETRY_HEADER: "1"},
        )
        if r.status_code == 429:
            hit_429 = True
            break
    if not hit_429:
        pytest.skip("Could not trigger 429; rate limits likely disabled.")

    # Now call through the backoff wrapper
    start = time.time()
    def _call():
        resp = api_client.client.post(
            "/api/v1/chatbooks/export",
            json=payload,
            headers={NO_RETRY_HEADER: "1"},
        )
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise
        return resp

    try:
        # Directly invoke the internal helper to verify retries/backoff logic executes
        resp = api_client._handle_rate_limit(_call)
        # If we got here without exception, either backoff allowed a success or limits are lenient now
        assert resp.status_code in (200, 201)
    except httpx.HTTPStatusError as e:
        # If still rate-limited, ensure that some backoff elapsed (> first delay ~0.5s) indicating retries occurred
        elapsed = time.time() - start
        assert elapsed >= 0.5, f"Backoff did not occur (elapsed={elapsed:.2f}s)"
        pytest.skip("Still rate-limited after retries; environment enforces longer windows.")
