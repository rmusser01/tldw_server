"""On real Redis a request's own max_concurrent holds across workers (TASK-13435)."""

import pytest
import redis.asyncio as aioredis
from redis.asyncio.retry import Retry
from redis.backoff import NoBackoff
from redis.exceptions import OutOfMemoryError, ReadOnlyError
from redis.exceptions import TimeoutError as RedisTimeoutError

from tldw_Server_API.app.core.Resource_Governance import RedisResourceGovernor, RGRequest

pytestmark = pytest.mark.rate_limit


class _NoJobsBlock:
    """A policy without a jobs block, as the usage-quota lease store uses."""

    def get_policy(self, _policy_id: str) -> dict:
        """The same block-less policy for any id."""
        return {"scopes": ["user"]}


def _lease() -> RGRequest:
    """One jobs lease held to a limit of 1."""
    return RGRequest(
        entity="user:42",
        categories={"jobs": {"units": 1, "max_concurrent": 1, "ttl_sec": 60}},
        tags={"policy_id": "limits.test_concurrent"},
    )


async def test_request_limit_holds_across_two_workers(real_redis, rg_unique_ns) -> None:
    """Two governors on one Redis namespace (two workers) share the user's limit; a release frees it for the other."""
    worker_a = RedisResourceGovernor(policy_loader=_NoJobsBlock(), ns=rg_unique_ns)
    worker_b = RedisResourceGovernor(policy_loader=_NoJobsBlock(), ns=rg_unique_ns)
    if not await worker_a._is_real_redis():
        pytest.skip("Redis client is not real; using in-memory stub")
    d1, h1 = await worker_a.reserve(_lease())
    d2, h2 = await worker_b.reserve(_lease())
    assert d1.allowed and h1
    assert not d2.allowed and h2 is None
    await worker_a.release(h1)
    d3, h3 = await worker_b.reserve(_lease())
    assert d3.allowed and h3


class _Policy:
    """One policy for any id: ``body``, with the scope and fail mode the test names."""

    def __init__(self, fail_mode: str, **body: dict) -> None:
        """Remember the policy's fail mode and its category blocks."""
        self._policy = {"scopes": ["user"], "fail_mode": fail_mode, **body}

    def get_policy(self, _policy_id: str) -> dict:
        """The same policy for any id."""
        return dict(self._policy)


# What a Redis restart or failover leaves in a worker: a cached SHA the server no longer knows (NOSCRIPT).
# Never SCRIPT FLUSH a shared Redis to get there.
_LOST_SHA = "f" * 40


@pytest.mark.parametrize("fail_mode", ["fallback_memory", "fail_closed", "fail_open"])
async def test_a_lost_lease_script_falls_back_to_counting_in_python(real_redis, rg_unique_ns, fail_mode) -> None:
    """A lease reserve that cannot run its script still counts leases, then loads the script again (TASK-13859)."""
    gov = RedisResourceGovernor(policy_loader=_Policy(fail_mode), ns=rg_unique_ns)
    if not await gov._is_real_redis():
        pytest.skip("Redis client is not real; using in-memory stub")
    lease = RGRequest(
        entity="user:42",
        categories={"jobs": {"units": 1, "max_concurrent": 2, "ttl_sec": 60}},
        tags={"policy_id": "limits.test_concurrent"},
    )
    first, _ = await gov.reserve(lease)  # loads the script
    gov._concurrency_lua_sha = _LOST_SHA

    second, _ = await gov.reserve(lease)
    third, _ = await gov.reserve(lease)

    assert [first.allowed, second.allowed, third.allowed] == [True, True, False]


async def test_a_lost_rate_script_falls_back_once_then_reloads(real_redis, rg_unique_ns) -> None:
    """A rate reserve that cannot run its script uses the Python windows once, then loads the script again (TASK-13859)."""
    gov = RedisResourceGovernor(policy_loader=_Policy("fallback_memory", requests={"rpm": 5}), ns=rg_unique_ns)
    if not await gov._is_real_redis():
        pytest.skip("Redis client is not real; using in-memory stub")
    req = RGRequest(entity="user:42", categories={"requests": {"units": 1}}, tags={"policy_id": "rate.test"})
    first, _ = await gov.reserve(req)  # loads the script
    gov._multi_lua_sha = _LOST_SHA

    second, _ = await gov.reserve(req)
    third, _ = await gov.reserve(req)

    assert [first.allowed, second.allowed, third.allowed] == [True, True, True]
    assert gov._multi_lua_sha not in (None, _LOST_SHA)


@pytest.mark.parametrize("via", ["reserve", "peek"])
async def test_a_lost_window_estimate_script_reloads_after_one_failure(real_redis, rg_unique_ns, via) -> None:
    """A full window's reset estimate falls back to the whole window once, then loads its script again (TASK-13859)."""
    now = [1000.0]
    gov = RedisResourceGovernor(policy_loader=_Policy("fallback_memory", tokens={"per_min": 2}), ns=rg_unique_ns, time_source=lambda: now[0])
    if not await gov._is_real_redis():
        pytest.skip("Redis client is not real; using in-memory stub")
    req = RGRequest(entity="user:42", categories={"tokens": {"units": 1}}, tags={"policy_id": "tokens.test"})
    await gov.reserve(req)
    await gov.reserve(req)  # the window is full until 1060
    gov._tokens_lua_sha = _LOST_SHA
    now[0] = 1030.0

    async def reset_estimate() -> int:
        if via == "peek":
            return (await gov.peek_with_policy("user:42", ["tokens"], "tokens.test"))["tokens"]["reset"]
        decision, _ = await gov.reserve(req)
        return decision.retry_after

    assert [await reset_estimate(), await reset_estimate()] == [60, 30]


def _reply_lost(run):
    """Run the script on Redis, then lose its reply, as a redis-py socket timeout or a connection reset does."""

    async def _command(*args, **kwargs):
        await run(*args, **kwargs)
        raise RedisTimeoutError("Timeout reading from socket")

    return _command


async def test_a_lost_lease_script_reply_does_not_count_its_own_lease(real_redis, rg_unique_ns, monkeypatch) -> None:
    """The lease script takes the lease, then its reply is lost: the user's only slot is still this request's (TASK-13892)."""
    monkeypatch.delenv("RG_TEST_PURGE_LEASES_BEFORE_RESERVE")
    gov = RedisResourceGovernor(policy_loader=_NoJobsBlock(), ns=rg_unique_ns)
    if not await gov._is_real_redis():
        pytest.skip("Redis client is not real; using in-memory stub")
    client = await gov._client_get()
    monkeypatch.setattr(client, "evalsha", _reply_lost(client.evalsha))
    key = gov._keys.lease("limits.test_concurrent", "jobs", "user", "42")

    decision, handle = await gov.reserve(_lease())
    held = await client.zcard(key)
    await gov.release(handle)

    assert (decision.allowed, held, await client.zcard(key)) == (True, 1, 0)


async def test_a_lost_rate_script_reply_charges_the_request_once(real_redis, rg_unique_ns, monkeypatch) -> None:
    """The multi-key rate script charges the window, then its reply is lost: the window holds one charge, not two (TASK-13892)."""
    gov = RedisResourceGovernor(policy_loader=_Policy("fallback_memory", requests={"rpm": 5}), ns=rg_unique_ns)
    if not await gov._is_real_redis():
        pytest.skip("Redis client is not real; using in-memory stub")
    client = await gov._client_get()
    monkeypatch.setattr(client, "evalsha", _reply_lost(client.evalsha))
    req = RGRequest(entity="user:42", categories={"requests": {"units": 1}}, tags={"policy_id": "rate.test"})

    decision, _ = await gov.reserve(req)

    assert (decision.allowed, await client.zcard(gov._keys.win("rate.test", "requests", "user", "42"))) == (True, 1)


def _rejecting(error: Exception):
    """A Redis command that fails with ``error``, as a replica (READONLY) or a full server (OOM) answers a write."""

    async def _command(*_args, **_kwargs):
        raise error

    return _command


# Redis still answers the probe (ZCARD) but rejects writes. A replica (READONLY) rejects every write; a full
# server (OOM) rejects the commands that grow memory, scripts included, but still trims windows and leases.
_READ_ONLY = (ReadOnlyError("You can't write against a read only replica."), ("evalsha", "zadd", "zremrangebyscore", "set", "hset", "expire", "delete"))
_OUT_OF_MEMORY = (OutOfMemoryError("command not allowed when used memory > 'maxmemory'."), ("evalsha", "zadd", "set", "hset"))


@pytest.mark.parametrize(
    ("fail_mode", "body", "categories", "fault"),
    [
        ("fallback_memory", {}, {"jobs": {"units": 1, "max_concurrent": 2, "ttl_sec": 60}}, _READ_ONLY),
        ("fail_closed", {}, {"jobs": {"units": 1, "max_concurrent": 2, "ttl_sec": 60}}, _OUT_OF_MEMORY),
        ("fail_closed", {"requests": {"rpm": 5}}, {"requests": {"units": 1}}, _READ_ONLY),
        ("fail_open", {"requests": {"rpm": 5}}, {"requests": {"units": 1}}, _OUT_OF_MEMORY),
    ],
    ids=["lease-readonly", "lease-oom", "rate-readonly", "rate-oom"],
)
async def test_redis_rejecting_writes_is_logged(real_redis, rg_unique_ns, monkeypatch, fail_mode, body, categories, fault) -> None:
    """A Redis that answers reads but rejects writes makes each worker count on its own; that is logged (TASK-13859)."""
    from loguru import logger

    from tldw_Server_API.app.core.Resource_Governance import policy_eval

    gov = RedisResourceGovernor(policy_loader=_Policy(fail_mode, **body), ns=rg_unique_ns)
    if not await gov._is_real_redis():
        pytest.skip("Redis client is not real; using in-memory stub")
    monkeypatch.setattr(policy_eval, "_last_redis_warning", {}, raising=False)  # no warning sent yet
    error, commands = fault
    client = await gov._client_get()
    for command in commands:
        monkeypatch.setattr(client, command, _rejecting(error))
    req = RGRequest(entity="user:42", categories=categories, tags={"policy_id": "write.test"})
    seen = []
    sink = logger.add(lambda m: seen.append(m.record["message"]), level="WARNING")
    try:
        await gov.reserve(req)
        await gov.reserve(req)
    finally:
        logger.remove(sink)

    assert [m for m in seen if "Redis backend" in m] == [
        f"Resource Governor Redis backend {rg_unique_ns!r} is failing ({type(error).__name__}); "
        "this worker enforces its limits on its own until it recovers"
    ]


# 7 requests against a limit of 5 while Redis rejects writes, as while it is unreachable: what each fail mode
# admits, and the shortest Retry-After on a denial.
_WHILE_REJECTING_WRITES = {
    ("fail_open", "requests"): ([True] * 5 + [False] * 2, 1),  # each worker's acceptance window holds requests
    ("fail_open", "tokens"): ([True] * 7, 1),  # admitted: tokens has no per-worker window
    ("fallback_memory", "requests"): ([True] * 5 + [False] * 2, 1),  # the worker's memory governor counts
    ("fallback_memory", "tokens"): ([True] * 5 + [False] * 2, 1),
    ("fail_closed", "requests"): ([False] * 7, 59),  # denied for the window, not a retry storm
    ("fail_closed", "tokens"): ([False] * 7, 59),
}


@pytest.mark.parametrize("fault", [_READ_ONLY, _OUT_OF_MEMORY], ids=["readonly", "oom"])
@pytest.mark.parametrize("fail_mode", ["fail_open", "fallback_memory", "fail_closed"])
@pytest.mark.parametrize(("category", "limits"), [("requests", {"rpm": 5}), ("tokens", {"per_min": 5})], ids=["requests", "tokens"])
async def test_redis_rejecting_writes_follows_the_fail_mode(real_redis, rg_unique_ns, monkeypatch, category, limits, fail_mode, fault) -> None:
    """A Redis that answers reads but rejects writes gets the fail mode, as an unreachable one does; reserve never raises (TASK-13859)."""
    gov = RedisResourceGovernor(policy_loader=_Policy(fail_mode, **{category: limits}), ns=rg_unique_ns)
    if not await gov._is_real_redis():
        pytest.skip("Redis client is not real; using in-memory stub")
    error, commands = fault
    client = await gov._client_get()
    for command in commands:
        monkeypatch.setattr(client, command, _rejecting(error))
    req = RGRequest(entity="user:42", categories={category: {"units": 1}}, tags={"policy_id": "write.test"})

    decisions = [(await gov.reserve(req))[0] for _ in range(7)]

    admitted, min_retry_after = _WHILE_REJECTING_WRITES[(fail_mode, category)]
    assert [d.allowed for d in decisions] == admitted
    assert all(d.retry_after >= min_retry_after for d in decisions if not d.allowed)


@pytest.mark.parametrize("fail_mode, admitted", [("fail_open", True), ("fallback_memory", True), ("fail_closed", False)])
@pytest.mark.parametrize(("category", "limits"), [("requests", {"rpm": 5}), ("tokens", {"per_min": 5})], ids=["requests", "tokens"])
async def test_redis_dropping_after_the_check_gets_the_fail_mode(real_redis, rg_unique_ns, monkeypatch, category, limits, fail_mode, admitted) -> None:
    """Redis drops between check() and the rate add: reserve applies the fail mode and never raises (TASK-13859)."""
    gov = RedisResourceGovernor(policy_loader=_Policy(fail_mode, **{category: limits}), ns=rg_unique_ns)
    if not await gov._is_real_redis():
        pytest.skip("Redis client is not real; using in-memory stub")
    check = gov.check

    async def check_then_drop(req: RGRequest):
        decision = await check(req)
        # Redis goes away now. Nothing listens on port 1, and retries are off, so each command fails at once.
        gov._client = aioredis.Redis(host="127.0.0.1", port=1, socket_connect_timeout=0.2, retry=Retry(NoBackoff(), 0))
        return decision

    monkeypatch.setattr(gov, "check", check_then_drop)
    decision, handle = await gov.reserve(RGRequest(entity="user:42", categories={category: {"units": 1}}, tags={"policy_id": "drop.test"}))

    assert (decision.allowed, handle is not None) == (admitted, admitted)
