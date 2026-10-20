"""Audio streams hold a per-user slot from can_start_stream to finish_stream (TASK-13435)."""

import asyncio

import pytest

from tldw_Server_API.app.core.Resource_Governance.governor import MemoryResourceGovernor
from tldw_Server_API.app.core.Usage import audio_quota, concurrency_quota

pytestmark = pytest.mark.unit


@pytest.fixture()
def streams_limit(monkeypatch: pytest.MonkeyPatch) -> dict:
    """A per-key limit the test sets, served to both the slot store and the audio limits view."""
    values: dict = {}

    async def _user_quota(_user_id: int, key: str) -> object:
        """Serve the test's limit for this key."""
        return values.get(key)

    monkeypatch.setattr(concurrency_quota, "user_quota", _user_quota)
    monkeypatch.setattr(audio_quota, "user_quota", _user_quota)
    monkeypatch.setattr(concurrency_quota, "_governor", MemoryResourceGovernor(policy_loader=concurrency_quota._QuotaPolicies()))
    return values


async def _in_own_task(coro):
    """Run a coroutine in a task of its own, as each WebSocket handler does."""
    return await asyncio.create_task(coro)


async def test_second_stream_is_refused_at_the_limit_and_admitted_after_finish(streams_limit: dict) -> None:
    """With limits.audio_concurrent_streams = 1 a second stream is refused until the first finishes."""
    streams_limit[audio_quota.STREAMS_KEY] = 1
    held = asyncio.Event()
    done = asyncio.Event()

    async def _first_stream() -> None:
        """Hold a stream until told to finish."""
        assert await audio_quota.can_start_stream(7) == (True, "OK")
        held.set()
        await done.wait()
        await audio_quota.finish_stream(7)

    first = asyncio.create_task(_first_stream())
    await held.wait()
    assert await _in_own_task(audio_quota.can_start_stream(7)) == (False, "Concurrent streams limit reached (1)")
    done.set()
    await first
    assert await _in_own_task(audio_quota.can_start_stream(7)) == (True, "OK")


async def test_finish_releases_the_finishing_streams_own_slot(streams_limit: dict) -> None:
    """Of two open streams, the one that finishes gives back its own lease; the other's lease stays held in the store."""
    streams_limit[audio_quota.STREAMS_KEY] = 2
    store = concurrency_quota._governor
    a_open, b_open, a_done, b_done = asyncio.Event(), asyncio.Event(), asyncio.Event(), asyncio.Event()
    handles: dict[str, str] = {}

    async def _stream(name: str, opened: asyncio.Event, finish: asyncio.Event) -> None:
        """Open a stream, note its lease, finish when told."""
        await audio_quota.can_start_stream(7)
        handles[name] = audio_quota._stream_slot.get()._handle_id
        opened.set()
        await finish.wait()
        await audio_quota.finish_stream(7)

    a = asyncio.create_task(_stream("a", a_open, a_done))
    await a_open.wait()
    b = asyncio.create_task(_stream("b", b_open, b_done))
    await b_open.wait()
    a_done.set()
    await a
    held = store._leases[(audio_quota.STREAMS_KEY, "streams", "user", "7")]
    assert {lease_id.split(":")[0] for lease_id in held} == {handles["b"]}
    b_done.set()
    await b


async def test_finish_without_a_slot_is_a_noop(streams_limit: dict) -> None:
    """finish_stream in a task that never started a stream does nothing and does not raise."""
    assert await _in_own_task(audio_quota.finish_stream(7)) is None


async def test_limits_view_reports_the_streams_limit(streams_limit: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """GET /audio/stream/limits shows limits.audio_concurrent_streams as concurrent_streams."""
    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "1")
    streams_limit[audio_quota.STREAMS_KEY] = 3
    assert (await audio_quota.get_limits_for_user(7))["concurrent_streams"] == 3
