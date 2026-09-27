import json
import os

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_redis_streams_roundtrip_smoke():
    """Basic Redis Streams roundtrip using a real Redis (CI service)."""
    try:
        import redis.asyncio as aioredis  # type: ignore
    except Exception:
        pytest.skip("redis-py not available")

    url = os.getenv("REDIS_URL", "redis://localhost:6379")
    # Use a short connect timeout so CI/sandbox environments fail fast
    client = await aioredis.from_url(
        url,
        decode_responses=True,
        socket_connect_timeout=0.25,
    )
    # Guard: if Redis is not reachable, skip this smoke test
    try:
        await client.ping()
    except Exception as e:  # pragma: no cover - environment dependent
        await client.close()
        pytest.skip(f"Redis not reachable: {e}")
    try:
        stream = "embeddings:integration:smoke"
        # XADD an entry
        eid = await client.xadd(stream, {"hello": "world", "n": json.dumps(1)})
        assert isinstance(eid, str)
        # XRANGE and confirm fields
        items = await client.xrange(stream, "-", "+", count=1)
        assert items and items[0][1]["hello"] == "world"
        # Acknowledge path (create group, read, ack)
        try:
            await client.xgroup_create(stream, "g", id="0", mkstream=True)
        except aioredis.ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise
        msgs = await client.xreadgroup("g", "c1", {stream: ">"}, count=1)
        assert msgs
        _, arr = msgs[0]
        mid = arr[0][0]
        assert await client.xack(stream, "g", mid) == 1
    finally:
        await client.close()
