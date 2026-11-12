"""Per-user concurrency slots: unset is unlimited, 0 blocks, every exit path gives the slot back (TASK-13435)."""

import asyncio

import anyio
import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from tldw_Server_API.app.core.Resource_Governance.governor import MemoryResourceGovernor
from tldw_Server_API.app.core.Resource_Governance.governor_redis import RedisResourceGovernor
from tldw_Server_API.app.core.Usage import concurrency_quota

pytestmark = pytest.mark.unit
KEY = "limits.media_concurrent_jobs"


@pytest.fixture()
def limits(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Per-key limits the test fills in, and a fresh in-memory lease store."""
    values: dict = {}

    async def _user_quota(user_id: int | None, key: str) -> object:
        """Serve the test's limit for this key; like the real resolver, no user id means unlimited."""
        return None if user_id is None else values.get(key)

    monkeypatch.setattr(concurrency_quota, "user_quota", _user_quota)
    monkeypatch.setattr(concurrency_quota, "_governor", MemoryResourceGovernor(policy_loader=concurrency_quota._QuotaPolicies()))
    return values


class _RecordingGovernor:
    """A lease store that admits everything and records what it was asked."""

    def __init__(self) -> None:
        """Start with empty records."""
        self.calls: list[tuple[str, object]] = []

    async def reserve(self, req):
        """Admit and record the request."""
        self.calls.append(("reserve", req))
        return type("D", (), {"allowed": True, "retry_after": None})(), f"h{len(self.calls)}"

    async def renew(self, handle_id, ttl_s):
        """Record a renewal."""
        self.calls.append(("renew", handle_id))

    async def release(self, handle_id):
        """Record a release after one scheduling point, as a networked store would."""
        await asyncio.sleep(0)
        self.calls.append(("release", handle_id))


class _DroppedRedisGovernor(_RecordingGovernor):
    """Admits, then Redis goes away: every renewal and release fails the way redis-py fails."""

    async def renew(self, handle_id, ttl_s):
        """Record the attempt, then fail like a dropped connection."""
        self.calls.append(("renew", handle_id))
        raise RedisConnectionError("Connection closed by server.")

    async def release(self, handle_id):
        """Record the attempt, then fail like a dropped connection."""
        self.calls.append(("release", handle_id))
        raise RedisConnectionError("Connection closed by server.")


async def test_no_limit_is_unlimited_and_never_touches_the_store(limits: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """A user with no value gets a slot that holds nothing; the lease store is never built."""
    built: list[str] = []

    def _recording_store() -> _RecordingGovernor:
        """Record that the store was built."""
        built.append("store")
        return _RecordingGovernor()

    monkeypatch.setattr(concurrency_quota, "_lease_governor", _recording_store)
    slots = [await concurrency_quota.acquire_slot(7, KEY, "jobs") for _ in range(5)]
    assert all(slot._handle_id is None for slot in slots)
    for slot in slots:
        await slot.release()
    assert built == []


async def test_limit_is_enforced_and_release_frees_a_slot(limits: dict) -> None:
    """With a limit of 1 the second slot is refused with a retry hint; releasing the first admits the next."""
    limits[KEY] = 1
    first = await concurrency_quota.acquire_slot(7, KEY, "jobs")
    with pytest.raises(concurrency_quota.ConcurrencyQuotaExceeded) as exc:
        await concurrency_quota.acquire_slot(7, KEY, "jobs")
    assert exc.value.limit == 1 and exc.value.retry_after >= 1
    await first.release()
    second = await concurrency_quota.acquire_slot(7, KEY, "jobs")
    await second.release()


async def test_zero_blocks_without_a_retry_hint(limits: dict) -> None:
    """A limit of 0 refuses every request, and there is no point retrying."""
    limits[KEY] = 0
    with pytest.raises(concurrency_quota.ConcurrencyQuotaExceeded) as exc:
        await concurrency_quota.acquire_slot(7, KEY, "jobs")
    assert exc.value.limit == 0 and exc.value.retry_after is None


async def test_users_and_keys_do_not_share_slots(limits: dict) -> None:
    """One user's held slot neither blocks another user nor a different concurrency key."""
    limits[KEY] = 1
    limits["limits.audio_concurrent_transcriptions"] = 1
    held = await concurrency_quota.acquire_slot(7, KEY, "jobs")
    other_user = await concurrency_quota.acquire_slot(8, KEY, "jobs")
    other_key = await concurrency_quota.acquire_slot(7, "limits.audio_concurrent_transcriptions", "jobs")
    for slot in (held, other_user, other_key):
        await slot.release()


async def test_release_is_idempotent(limits: dict) -> None:
    """Releasing twice frees one slot, not two."""
    limits[KEY] = 2
    a = await concurrency_quota.acquire_slot(7, KEY, "jobs")
    b = await concurrency_quota.acquire_slot(7, KEY, "jobs")
    await a.release()
    await a.release()
    c = await concurrency_quota.acquire_slot(7, KEY, "jobs")
    with pytest.raises(concurrency_quota.ConcurrencyQuotaExceeded):
        await concurrency_quota.acquire_slot(7, KEY, "jobs")
    for slot in (b, c):
        await slot.release()


async def test_cancelled_request_gives_its_slot_back(limits: dict) -> None:
    """A request cancelled mid-flight (client gone) releases in its finally, and the slot is free again."""
    limits[KEY] = 1
    started = asyncio.Event()

    async def _request() -> None:
        """Hold a slot until cancelled."""
        slot = await concurrency_quota.acquire_slot(7, KEY, "jobs")
        try:
            started.set()
            await asyncio.Event().wait()
        finally:
            await slot.release()

    task = asyncio.create_task(_request())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    slot = await concurrency_quota.acquire_slot(7, KEY, "jobs")
    await slot.release()


async def test_release_completes_inside_a_cancelled_scope(limits: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """A server that cancels the request's scope (anyio level cancellation) cannot abort the release."""
    limits[KEY] = 1
    store = _RecordingGovernor()
    monkeypatch.setattr(concurrency_quota, "_governor", store)
    slot = await concurrency_quota.acquire_slot(7, KEY, "jobs")
    with anyio.CancelScope() as scope:
        scope.cancel()
        try:
            await anyio.sleep(1)
        finally:
            await slot.release()
    assert ("release", "h1") in store.calls


async def test_keep_alive_renews_until_the_owner_ends(limits: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """A slot renews its lease while its task runs, and stops once the task ends without releasing."""
    limits[KEY] = 1
    store = _RecordingGovernor()
    monkeypatch.setattr(concurrency_quota, "_governor", store)
    monkeypatch.setattr(concurrency_quota, "RENEW_EVERY_SEC", 0.01)

    async def _leaky_request() -> None:
        """Take a slot, run a while, and end without releasing it (a bug the backstop covers)."""
        await concurrency_quota.acquire_slot(7, KEY, "jobs")
        await asyncio.sleep(0.05)

    await asyncio.create_task(_leaky_request())
    renewals = sum(1 for call in store.calls if call[0] == "renew")
    await asyncio.sleep(0.05)
    assert renewals >= 2
    assert sum(1 for call in store.calls if call[0] == "renew") == renewals


async def test_store_error_admits_the_request(limits: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """A lease store that fails open: the request goes ahead with a slot that holds nothing."""
    limits[KEY] = 1

    class _Broken:
        """A lease store whose Redis connection has dropped."""

        async def reserve(self, req):
            """Fail the way redis-py does: its ConnectionError is not the builtin one."""
            raise RedisConnectionError("Connection closed by server.")

    monkeypatch.setattr(concurrency_quota, "_governor", _Broken())
    slot = await concurrency_quota.acquire_slot(7, KEY, "jobs")
    assert slot._handle_id is None
    await slot.release()


async def test_parallel_requests_get_exactly_the_limit(limits: dict) -> None:
    """Six requests from one user racing for a limit of 2: two get slots, four get the 429 refusal."""
    limits[KEY] = 2
    results = await asyncio.gather(
        *(concurrency_quota.acquire_slot(7, KEY, "jobs") for _ in range(6)), return_exceptions=True
    )
    held = [r for r in results if isinstance(r, concurrency_quota.ConcurrencySlot)]
    refused = [r for r in results if isinstance(r, concurrency_quota.ConcurrencyQuotaExceeded)]
    assert (len(held), len(refused)) == (2, 4)
    for slot in held:
        await slot.release()


async def test_lowering_the_limit_refuses_new_requests_until_usage_drops(limits: dict) -> None:
    """Running requests keep their slots when an admin lowers the limit; new ones wait until usage is under it."""
    limits[KEY] = 3
    running = [await concurrency_quota.acquire_slot(7, KEY, "jobs") for _ in range(3)]
    limits[KEY] = 1
    await running[0].release()
    with pytest.raises(concurrency_quota.ConcurrencyQuotaExceeded):
        await concurrency_quota.acquire_slot(7, KEY, "jobs")
    await running[1].release()
    await running[2].release()
    slot = await concurrency_quota.acquire_slot(7, KEY, "jobs")
    await slot.release()


async def test_switching_quotas_off_admits_at_once(limits: dict) -> None:
    """With the user's slot held, turning the limit off (switch off or value removed) admits the next request."""
    limits[KEY] = 1
    held = await concurrency_quota.acquire_slot(7, KEY, "jobs")
    limits.pop(KEY)
    unlimited = await concurrency_quota.acquire_slot(7, KEY, "jobs")
    await unlimited.release()
    await held.release()


async def test_a_redis_outage_never_escapes_renew_or_release(limits: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redis drops mid-request: renewals keep trying, the release does not raise, and the request's own outcome stands."""
    limits[KEY] = 1
    store = _DroppedRedisGovernor()
    monkeypatch.setattr(concurrency_quota, "_governor", store)
    monkeypatch.setattr(concurrency_quota, "RENEW_EVERY_SEC", 0.01)
    slot = await concurrency_quota.acquire_slot(7, KEY, "jobs")
    await asyncio.sleep(0.05)
    assert sum(1 for call in store.calls if call[0] == "renew") >= 2
    assert not slot._keepalive.done()
    await slot.release()
    assert ("release", "h1") in store.calls


async def test_a_non_numeric_user_is_unlimited(limits: dict) -> None:
    """A principal without a numeric user id is never limited and never crashes the request."""
    limits[KEY] = 0
    for user_id in ("api-key-7", None):
        slot = await concurrency_quota.acquire_slot(user_id, KEY, "jobs")
        await slot.release()


def test_store_follows_rg_backend_even_with_rg_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """The lease store is Redis when RG_BACKEND=redis, and RG_ENABLED=false does not remove it (ADR-076)."""
    monkeypatch.setattr(concurrency_quota, "_governor", None)
    monkeypatch.setenv("RG_ENABLED", "false")
    monkeypatch.setenv("RG_BACKEND", "redis")
    assert isinstance(concurrency_quota._lease_governor(), RedisResourceGovernor)
    monkeypatch.setattr(concurrency_quota, "_governor", None)
    monkeypatch.setenv("RG_BACKEND", "memory")
    assert isinstance(concurrency_quota._lease_governor(), MemoryResourceGovernor)
