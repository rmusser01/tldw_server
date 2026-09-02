import re
import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app


@pytest.mark.unit
def test_prometheus_metrics_contains_orchestrator_gauges(disable_heavy_startup, admin_user, redis_client):
    redis_client.run(redis_client.xadd("embeddings:embedding", {"seq": "0"}))
    # Trigger snapshot so gauges are set
    client = TestClient(app)
    r0 = client.get("/api/v1/embeddings/orchestrator/summary")
    assert r0.status_code == 200

    # Fetch metrics in Prometheus text format
    r = client.get("/api/v1/metrics/text")
    assert r.status_code == 200
    text = r.text
    # Presence checks
    assert "embedding_queue_age_current_seconds" in text
    assert "embedding_stage_flag" in text
    # New counters should exist (may be zero)
    assert "orchestrator_summary_failures_total" in text


@pytest.mark.unit
def test_summary_failure_increments_counter(disable_heavy_startup, admin_user, monkeypatch):
     # Force Redis connection failure to trigger fallback and counter increment
    import redis.asyncio as aioredis

    async def fake_from_url(*args, **kwargs):
        raise ConnectionError("cannot connect")

    monkeypatch.setattr(aioredis, "from_url", fake_from_url)

    client = TestClient(app)
    r0 = client.get("/api/v1/embeddings/orchestrator/summary")
    assert r0.status_code == 200
    # Fetch metrics and assert the counter incremented
    r = client.get("/api/v1/metrics/text")
    assert r.status_code == 200
    m = re.search(r"^orchestrator_summary_failures_total\s+(\d+(?:\.\d+)?)$", r.text, re.M)
    assert m, f"orchestrator_summary_failures_total not found in metrics: {r.text[:4000]}"
    assert float(m.group(1)) >= 1.0


@pytest.mark.unit
def test_sse_disconnect_increments_counter(
    disable_heavy_startup, admin_user, redis_client, monkeypatch
):
     # Call the endpoint function directly and close its generator to trigger disconnect accounting
    from tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced import orchestrator_events
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User

    # Create admin user for direct call
    admin = User(id=1, username="admin", email="a@x", is_active=True, is_admin=True)

    # Patch redis client factory to use the shared redis_client
    import redis.asyncio as aioredis

    async def fake_from_url(url, decode_responses=True):
        return redis_client.client

    monkeypatch.setattr(aioredis, "from_url", fake_from_url)

    # Run SSE endpoint to get StreamingResponse; then consume once and close
    async def _run_once_and_close():
        resp = await orchestrator_events(_current_user=admin)
        agen = resp.body_iterator
        # consume one chunk then close to trigger finally{}
        try:
            await agen.__anext__()
        except StopAsyncIteration:
            _ = None
        try:
            await agen.aclose()
        except Exception:
            _ = None

    redis_client.run(_run_once_and_close())

    client = TestClient(app)
    r2 = client.get("/api/v1/metrics/text")
    assert r2.status_code == 200
    text = r2.text
    m = re.search(r"^orchestrator_sse_disconnects_total\s+(\d+(?:\.\d+)?)$", text, re.M)
    assert m, f"orchestrator_sse_disconnects_total not found in metrics: {text[:4000]}"
    assert float(m.group(1)) >= 1.0


@pytest.mark.unit
def test_stage_flag_metric_after_pause(disable_heavy_startup, admin_user, redis_client):
    client = TestClient(app)
    # Pause embedding stage via admin API
    r0 = client.post("/api/v1/embeddings/stage/control", json={"stage": "embedding", "action": "pause"})
    assert r0.status_code == 200
    # Build snapshot to set gauges
    r1 = client.get("/api/v1/embeddings/orchestrator/summary")
    assert r1.status_code == 200
    # Check metrics text contains embedding_stage_flag for embedding paused = 1
    r2 = client.get("/api/v1/metrics/text")
    assert r2.status_code == 200
    text = r2.text
    paused_lines = [
        line for line in text.splitlines()
        if line.startswith("embedding_stage_flag{")
        and 'stage="embedding"' in line
        and 'flag="paused"' in line
    ]
    assert len(paused_lines) == 1
    metric_value = paused_lines[0].split("}", 1)[1].strip()
    assert float(metric_value) == 1.0
