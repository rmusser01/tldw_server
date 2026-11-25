import asyncio
import contextlib
import socket
import time

import pytest
import redis.asyncio as aioredis
from redis.asyncio.retry import Retry
from redis.backoff import NoBackoff
from redis.exceptions import BusyLoadingError, NoScriptError
from redis.exceptions import ConnectionError as RedisConnectionError

pytestmark = pytest.mark.rate_limit

from tldw_Server_API.app.core.Infrastructure.redis_factory import InMemoryAsyncRedis
from tldw_Server_API.app.core.Resource_Governance import RedisResourceGovernor, RGRequest


class _BrokenClient:
    async def evalsha(self, *a, **k):
        raise RuntimeError("boom")
    async def script_load(self, *a, **k):
        raise RuntimeError("boom")
    async def zadd(self, *a, **k):
        raise RuntimeError("boom")
    async def zremrangebyscore(self, *a, **k):
        raise RuntimeError("boom")
    async def zcard(self, *a, **k):
        raise RuntimeError("boom")
    async def set(self, *a, **k):
        raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_fail_open_tokens_allows_on_error(monkeypatch):
    class _Loader:
        def get_policy(self, pid):
            return {"tokens": {"per_min": 1, "fail_mode": "fail_open"}, "scopes": ["global", "user"]}

    rg = RedisResourceGovernor(policy_loader=_Loader(), ns="rg_fail_open")

    async def _broken():
        return _BrokenClient()

    monkeypatch.setattr(rg, "_client_get", _broken)
    d, h = await rg.reserve(RGRequest(entity="user:x", categories={"tokens": {"units": 1}}, tags={"policy_id": "p"}))
    assert d.allowed and h


@pytest.mark.asyncio
async def test_fail_closed_requests_denies_on_error(monkeypatch):
    class _Loader:
        def get_policy(self, pid):
            return {"requests": {"rpm": 1, "fail_mode": "fail_closed"}, "scopes": ["global", "user"]}

    rg = RedisResourceGovernor(policy_loader=_Loader(), ns="rg_fail_closed")

    async def _broken():
        return _BrokenClient()

    monkeypatch.setattr(rg, "_client_get", _broken)
    d, h = await rg.reserve(RGRequest(entity="user:x", categories={"requests": {"units": 1}}, tags={"policy_id": "p"}))
    assert (not d.allowed) and (h is None)


@pytest.mark.asyncio
async def test_fallback_memory_requests_allows_when_redis_broken(monkeypatch):
    class _Loader:
        def get_policy(self, pid):
            return {"requests": {"rpm": 1, "fail_mode": "fallback_memory"}, "scopes": ["global", "user"]}

    rg = RedisResourceGovernor(policy_loader=_Loader(), ns="rg_fallback_mem")

    async def _broken():
        return _BrokenClient()

    monkeypatch.setattr(rg, "_client_get", _broken)
    req = RGRequest(entity="user:x", categories={"requests": {"units": 1}}, tags={"policy_id": "p"})
    d1, h1 = await rg.reserve(req, op_id="f1")
    assert d1.allowed and h1
    d2, h2 = await rg.reserve(req, op_id="f2")
    # Memory fallback should enforce rpm=1
    assert (not d2.allowed) and (h2 is None)


def _redis_down_client() -> aioredis.Redis:
    """A real redis-py client whose server is gone: every command raises redis.exceptions.ConnectionError.

    Nothing listens on port 1, and retries are off, so each command fails at once.
    """
    return aioredis.Redis(host="127.0.0.1", port=1, socket_connect_timeout=0.2, retry=Retry(NoBackoff(), 0), decode_responses=True)


def _governor(policy: dict, ns: str, time_source=time.time) -> RedisResourceGovernor:
    class _Loader:
        def get_policy(self, pid):
            return policy

    return RedisResourceGovernor(policy_loader=_Loader(), ns=ns, time_source=time_source)


@pytest.mark.asyncio
@pytest.mark.parametrize("category, limits", [("requests", {"rpm": 5}), ("tokens", {"per_min": 5})])
@pytest.mark.parametrize("fail_mode, admitted", [("fail_open", True), ("fallback_memory", True), ("fail_closed", False)])
async def test_reserve_honours_the_fail_mode_when_redis_raises_connection_error(category, limits, fail_mode, admitted):
    rg = _governor({category: limits, "scopes": ["user"], "fail_mode": fail_mode}, ns=f"rg_down_{category}_{fail_mode}")
    rg._client = _redis_down_client()  # the worker connected once; Redis has gone away since
    req = RGRequest(entity="user:x", categories={category: {"units": 1}}, tags={"policy_id": "p"})

    decision, handle = await rg.reserve(req)

    assert (decision.allowed, handle is not None) == (admitted, admitted)


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_mode", ["fail_open", "fail_closed", "fallback_memory"])
async def test_renew_and_commit_survive_redis_raising_connection_error(fail_mode, monkeypatch):
    # The RG conftest's test-only lease purge would drop the in-process lease for us; production never runs it.
    monkeypatch.delenv("RG_TEST_PURGE_LEASES_BEFORE_RESERVE")
    rg = _governor({"jobs": {"max_concurrent": 1, "ttl_sec": 60}, "scopes": ["user"], "fail_mode": fail_mode}, ns=f"rg_down_lease_{fail_mode}")
    rg._client = InMemoryAsyncRedis()
    req = RGRequest(entity="user:x", categories={"jobs": {"units": 1}}, tags={"policy_id": "p"})
    decision, handle = await rg.reserve(req)
    assert decision.allowed and handle

    rg._client = _redis_down_client()  # Redis goes away while the lease is held
    await rg.renew(handle, 60)
    await rg.commit(handle)  # the request ends: its only slot frees, with Redis still down
    again, again_handle = await rg.reserve(req)

    assert again.allowed and again_handle


@pytest.mark.asyncio
async def test_peek_reports_unknown_headroom_when_redis_raises_connection_error():
    rg = _governor({"requests": {"rpm": 5}, "scopes": ["user"]}, ns="rg_down_peek")
    rg._client = _redis_down_client()

    # Ingress headers and the character rate limiter's usage stats read this; it must not raise.
    assert await rg.peek_with_policy("user:x", ["requests"], "p") == {"requests": {"remaining": None, "reset": None}}


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_mode", ["fail_open", "fail_closed", "fallback_memory"])
async def test_a_lease_committed_while_redis_is_down_frees_its_slot(fail_mode, monkeypatch):
    # The RG conftest's test-only lease purge would drop the in-process lease for us; production never runs it.
    monkeypatch.delenv("RG_TEST_PURGE_LEASES_BEFORE_RESERVE")
    rg = _governor({"jobs": {"max_concurrent": 1, "ttl_sec": 60}, "scopes": ["user"], "fail_mode": fail_mode}, ns=f"rg_down_free_{fail_mode}")
    rg._client = InMemoryAsyncRedis()
    req = RGRequest(entity="user:x", categories={"jobs": {"units": 1}}, tags={"policy_id": "p"})
    _decision, handle = await rg.reserve(req)

    rg._client = _redis_down_client()
    await rg.commit(handle)
    decision, again = await rg.reserve(req)

    assert decision.allowed and again


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_mode", ["fail_open", "fail_closed", "fallback_memory"])
async def test_leases_count_per_worker_while_redis_is_down(fail_mode, monkeypatch):
    # The RG conftest's test-only lease purge drops the in-process leases once Redis is down; production never runs it.
    monkeypatch.delenv("RG_TEST_PURGE_LEASES_BEFORE_RESERVE")
    rg = _governor({"jobs": {"max_concurrent": 1, "ttl_sec": 60}, "scopes": ["user"], "fail_mode": fail_mode}, ns=f"rg_down_count_{fail_mode}")
    rg._client = _redis_down_client()
    req = RGRequest(entity="user:x", categories={"jobs": {"units": 1}}, tags={"policy_id": "p"})

    first, handle = await rg.reserve(req)
    second, _ = await rg.reserve(req)
    await rg.commit(handle)
    third, _ = await rg.reserve(req)

    assert [first.allowed, second.allowed, third.allowed] == [True, False, True]


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_mode", ["fail_open", "fail_closed", "fallback_memory"])
async def test_renew_while_redis_is_down_keeps_the_lease(fail_mode, monkeypatch):
    monkeypatch.delenv("RG_TEST_PURGE_LEASES_BEFORE_RESERVE")
    now = [1000.0]
    rg = _governor(
        {"jobs": {"max_concurrent": 1, "ttl_sec": 60}, "scopes": ["user"], "fail_mode": fail_mode},
        ns=f"rg_down_renew_{fail_mode}",
        time_source=lambda: now[0],
    )
    rg._client = _redis_down_client()
    req = RGRequest(entity="user:x", categories={"jobs": {"units": 1}}, tags={"policy_id": "p"})
    first, handle = await rg.reserve(req)  # the lease runs to 1060

    now[0] = 1050.0
    await rg.renew(handle, 60)  # now to 1110
    now[0] = 1070.0
    second, _ = await rg.reserve(req)

    assert first.allowed and not second.allowed


@pytest.mark.asyncio
# Not fallback_memory: there the memory governor holds the lease, which Redis does not see once it returns (ADR-078).
@pytest.mark.parametrize("fail_mode", ["fail_open", "fail_closed"])
async def test_renew_after_redis_returns_keeps_a_lease_taken_during_the_outage(fail_mode, monkeypatch):
    monkeypatch.delenv("RG_TEST_PURGE_LEASES_BEFORE_RESERVE")
    now = [1000.0]
    rg = _governor(
        {"jobs": {"max_concurrent": 1, "ttl_sec": 60}, "scopes": ["user"], "fail_mode": fail_mode},
        ns=f"rg_back_renew_{fail_mode}",
        time_source=lambda: now[0],
    )
    rg._client = _redis_down_client()
    req = RGRequest(entity="user:x", categories={"jobs": {"units": 1}}, tags={"policy_id": "p"})
    first, handle = await rg.reserve(req)  # held by this worker only; the lease runs to 1060

    rg._client = InMemoryAsyncRedis()  # Redis is back, with no record of the lease
    now[0] = 1050.0
    await rg.renew(handle, 60)  # now to 1110
    now[0] = 1070.0
    second, _ = await rg.reserve(req)

    assert first.allowed and not second.allowed


@contextlib.contextmanager
def _unreachable_redis(kind: str):
    """Yield ``host:port`` of a Redis RG cannot reach: unroutable, stopped (nothing listens) or silent (never answers)."""
    if kind == "unroutable":
        # Black-holed: a connect waits for the connect timeout, unless the network answers "unreachable" at once.
        yield "10.255.255.1:6379"
        return
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        address = f"127.0.0.1:{sock.getsockname()[1]}"
        if kind == "silent":
            sock.listen(8)  # the kernel completes the connect; nothing reads or replies, so each command times out
            yield address
            return
    yield address  # stopped: the port is free again, so a connect is refused


def _redis_gone_since_connect(address: str, sent: list):
    """A stand-in for ``create_async_redis_client``: the client it hands RG was built before Redis went away (so no
    ping), with 0.2 s timeouts to keep the test fast, and it records each command it sends, PINGs included."""

    async def _create(**_kwargs):
        client = aioredis.from_url(f"redis://{address}", socket_connect_timeout=0.2, socket_timeout=0.2, decode_responses=True)
        run = client.execute_command

        async def _counted(*args, **options):
            sent.append(args[0])
            return await run(*args, **options)

        client.execute_command = _counted
        return client

    return _create


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["unroutable", "stopped", "silent"])
@pytest.mark.parametrize("fail_mode, admitted", [("fail_open", True), ("fallback_memory", True), ("fail_closed", False)])
async def test_rg_stops_calling_an_unreachable_redis_until_the_cooldown_ends(kind, fail_mode, admitted, monkeypatch):
    """One command and the PING after it fail to reach Redis; RG answers the rest by its fail mode without calling Redis (TASK-13882)."""
    from tldw_Server_API.app.core.Resource_Governance import governor_redis

    sent = []
    with _unreachable_redis(kind) as address:
        monkeypatch.setattr(governor_redis, "create_async_redis_client", _redis_gone_since_connect(address, sent))
        rg = _governor({"requests": {"rpm": 5}, "scopes": ["user"], "fail_mode": fail_mode}, ns=f"rg_unreachable_{fail_mode}")
        req = RGRequest(entity="user:x", categories={"requests": {"units": 1}}, tags={"policy_id": "p"})

        decisions = [(await rg.reserve(req))[0] for _ in range(3)]

    assert ([d.allowed for d in decisions], len(sent), sent[1:2]) == ([admitted] * 3, 2, ["PING"])


@pytest.mark.asyncio
async def test_a_zero_cooldown_keeps_calling_an_unreachable_redis(monkeypatch):
    """RG_REDIS_UNREACHABLE_COOLDOWN_SEC=0 turns the cooldown off: every reserve calls Redis, and nothing PINGs it (TASK-13882)."""
    from tldw_Server_API.app.core.Resource_Governance import governor_redis

    monkeypatch.setenv("RG_REDIS_UNREACHABLE_COOLDOWN_SEC", "0")
    sent = []
    with _unreachable_redis("stopped") as address:
        monkeypatch.setattr(governor_redis, "create_async_redis_client", _redis_gone_since_connect(address, sent))
        rg = _governor({"requests": {"rpm": 5}, "scopes": ["user"], "fail_mode": "fail_open"}, ns="rg_unreachable_off")
        req = RGRequest(entity="user:x", categories={"requests": {"units": 1}}, tags={"policy_id": "p"})
        per_reserve = []
        for _ in range(3):
            before = len(sent)
            await rg.reserve(req)
            per_reserve.append(len(sent) - before)

    assert (min(per_reserve) > 0, "PING" in sent) == (True, False)


@pytest.mark.parametrize(
    "raw, seconds",
    [(None, 10.0), ("", 10.0), ("0", 0.0), ("2.5", 2.5), ("-1", 10.0), ("ten", 10.0), ("inf", 10.0), ("nan", 10.0)],
)
def test_the_cooldown_comes_from_its_env_var(raw, seconds, monkeypatch):
    """RG_REDIS_UNREACHABLE_COOLDOWN_SEC sets the cooldown; unset, or not a number of seconds >= 0, it is 10 s (TASK-13882)."""
    from tldw_Server_API.app.core.Resource_Governance import governor_redis

    if raw is None:
        monkeypatch.delenv("RG_REDIS_UNREACHABLE_COOLDOWN_SEC", raising=False)
    else:
        monkeypatch.setenv("RG_REDIS_UNREACHABLE_COOLDOWN_SEC", raw)

    assert governor_redis._redis_unreachable_cooldown_s() == seconds


@pytest.mark.asyncio
async def test_rg_builds_its_redis_client_with_5_s_timeouts(monkeypatch):
    """RG's client waits at most 5 s to connect and for each reply, on any redis-py release the pin allows (TASK-13882)."""
    from tldw_Server_API.app.core.Resource_Governance import governor_redis

    async def _create(**kwargs):
        # Builds the client from the governor's options, as redis_factory does; it never connects in this test.
        return aioredis.from_url("redis://127.0.0.1:1", **(kwargs.get("redis_kwargs") or {}))

    monkeypatch.setattr(governor_redis, "create_async_redis_client", _create)
    client = await _governor({"scopes": ["user"]}, ns="rg_timeouts")._client_get()
    options = client.connection_pool.connection_kwargs

    assert (options.get("socket_connect_timeout"), options.get("socket_timeout")) == (5.0, 5.0)


class _ScriptedRedis:
    """A Redis stand-in that records each command in ``sent`` and answers with the next of ``answers`` (raising an exception)."""

    def __init__(self, answers: list, sent: list):
        self.answers, self.sent = answers, sent

    async def execute_command(self, *args, **_options):
        self.sent.append(args[0])
        await asyncio.sleep(0)  # in flight: a concurrent command runs now
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


_LOADING = BusyLoadingError("Redis is loading the dataset in memory")


@pytest.mark.asyncio
# Any answer from Redis shows it is reachable, an error reply too: after a restart the probe is often NOSCRIPT, or LOADING.
@pytest.mark.parametrize("probe_answer", ["PONG", NoScriptError("No matching script."), _LOADING], ids=["reply", "error-reply", "loading"])
async def test_rg_calls_redis_again_once_it_answers_after_the_cooldown(probe_answer):
    """After the cooldown one command goes to Redis while the others still fail fast; once Redis answers, all go (TASK-13882)."""
    from tldw_Server_API.app.core.Resource_Governance import governor_redis

    # The failed command, the PING that confirms Redis is down, the probe after the cooldown, the next command.
    answers = [RedisConnectionError("Connection refused"), RedisConnectionError("Connection refused"), probe_answer, "after"]
    sent = []

    client = governor_redis._stop_calling_redis_while_unreachable(_ScriptedRedis(answers, sent), cooldown_s=0.5)
    with pytest.raises(RedisConnectionError):
        await client.execute_command("ZCARD")  # Redis is down, and a PING confirms it
    with pytest.raises(RedisConnectionError):
        await client.execute_command("ZCARD")  # within the cooldown: not sent
    await asyncio.sleep(0.6)
    probe, other = await asyncio.gather(client.execute_command("ZCARD"), client.execute_command("ZCARD"), return_exceptions=True)

    assert (probe is probe_answer, type(other), await client.execute_command("ZCARD"), sent) == (True, RedisConnectionError, "after", ["ZCARD", "PING", "ZCARD", "ZCARD"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answers, expected_sent",
    [([_LOADING, "after"], ["ZCARD", "ZCARD"]), ([RedisConnectionError("Connection reset"), _LOADING, "after"], ["ZCARD", "PING", "ZCARD"])],
    ids=["command-loading", "ping-loading"],
)
async def test_a_loading_redis_does_not_start_the_cooldown(answers, expected_sent):
    """LOADING is redis-py's BusyLoadingError, a ConnectionError, but Redis sent it: a LOADING reply to a command, or to
    the PING after a failed one, starts no cooldown, so RG calls Redis again the moment it has loaded (TASK-13882)."""
    from tldw_Server_API.app.core.Resource_Governance import governor_redis

    sent = []
    client = governor_redis._stop_calling_redis_while_unreachable(_ScriptedRedis(answers, sent), cooldown_s=10)
    with pytest.raises(RedisConnectionError):
        await client.execute_command("ZCARD")

    assert (await client.execute_command("ZCARD"), sent) == ("after", expected_sent)


@contextlib.asynccontextmanager
async def _redis_that_closes_its_first_connection():
    """Yield ``host:port`` of a Redis stand-in that handles the client handshake, and closes its first connection after
    the first PING, as Redis's ``timeout`` setting (or a proxy's idle timeout) closes an idle client."""
    connections = []
    handlers = []
    closing = False

    async def _serve(reader, writer, first):
        # Decode complete command frames with redis-py, including split/coalesced TCP reads.
        incoming = aioredis.Connection()
        incoming._reader = reader
        incoming._parser.on_connect(incoming)
        try:
            while True:
                command = await incoming.read_response()
                name = command[0].upper()
                if name == b"HELLO":
                    reply = b"%1\r\n+proto\r\n:3\r\n" if command[1] == b"3" else b"*2\r\n+proto\r\n:2\r\n"
                elif name == b"PING":
                    reply = b"+PONG\r\n"
                else:
                    reply = b"+OK\r\n"
                writer.write(reply)
                await writer.drain()
                if first and name == b"PING":
                    break
        except (RedisConnectionError, ConnectionError):
            pass  # The peer closed or reset its socket.
        finally:
            writer.close()
            await asyncio.wait_for(writer.wait_closed(), timeout=1)

    def accept(reader, writer):
        first = not connections
        # Own the socket and task before the asynchronous handler can be delayed.
        connections.append(writer)
        handlers.append(asyncio.create_task(_serve(reader, writer, first)))
        if closing:
            writer.close()

    server = await asyncio.start_server(accept, "127.0.0.1", 0)
    try:
        yield f"127.0.0.1:{server.sockets[0].getsockname()[1]}"
    finally:
        closing = True
        server.close()
        for writer in connections:
            writer.close()

        async def drain():
            await server.wait_closed()
            return await asyncio.gather(*handlers, return_exceptions=True)

        results = await asyncio.wait_for(drain(), timeout=1)
        for result in results:
            if isinstance(result, BaseException):
                raise result


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", [None, 2, 3], ids=["default", "resp2", "resp3"])
async def test_a_connection_redis_closed_while_idle_does_not_start_the_cooldown(protocol):
    """Redis closed the pooled connection while it sat idle: that command may fail, as before, but the PING after it gets
    an answer, so RG keeps calling Redis. Starting the cooldown here would refuse every user under fail_closed (TASK-13882)."""
    from tldw_Server_API.app.core.Resource_Governance import governor_redis

    async with _redis_that_closes_its_first_connection() as address:
        options = {} if protocol is None else {"protocol": protocol}
        raw = aioredis.from_url(f"redis://{address}", socket_timeout=1, **options)
        try:
            client = governor_redis._stop_calling_redis_while_unreachable(raw, cooldown_s=10)
            assert await client.ping()  # the connection the stand-in then closes
            await asyncio.sleep(0.05)
            # redis-py 8.1 sends this on the closed connection, so it fails, as without the cooldown; a redis-py that
            # notices the close first reconnects instead. Either way the next command must reach Redis.
            with contextlib.suppress(RedisConnectionError):
                await client.ping()
            answered = await client.ping()
        finally:
            await asyncio.wait_for(raw.aclose(), timeout=1)

    assert answered is True


@pytest.mark.asyncio
@pytest.mark.parametrize("body_error", [False, True], ids=["normal-exit", "original-error"])
async def test_idle_disconnect_fixture_closes_handlers_before_their_first_step(monkeypatch, body_error):
    loop = asyncio.get_running_loop()
    create_task = loop.create_task
    accepted, release = asyncio.Event(), asyncio.Event()
    handlers, clients = [], []

    def defer_handler(coro, *args, **kwargs):
        if getattr(getattr(coro, "cr_code", None), "co_name", None) != "_serve":
            return create_task(coro, *args, **kwargs)

        async def delayed():
            try:
                await release.wait()
                return await coro
            finally:
                coro.close()

        task = create_task(delayed(), *args, **kwargs)
        handlers.append(task)
        accepted.set()
        return task

    monkeypatch.setattr(loop, "create_task", defer_handler)
    original_error = RuntimeError("original fixture body failure")

    async def use_fixture():
        async with _redis_that_closes_its_first_connection() as address:
            host, port = address.split(":")
            clients.append(await asyncio.wait_for(asyncio.open_connection(host, int(port)), timeout=1))
            await asyncio.wait_for(accepted.wait(), timeout=1)
            loop.call_soon(release.set)
            if body_error:
                raise original_error

    try:
        if body_error:
            with pytest.raises(RuntimeError, match="original fixture body failure") as caught:
                await use_fixture()
            assert caught.value is original_error
        else:
            await use_fixture()
        assert handlers and all(task.done() and not task.cancelled() for task in handlers)
        assert all(task.exception() is None for task in handlers)
        for reader, _writer in clients:
            with contextlib.suppress(ConnectionResetError):
                assert await asyncio.wait_for(reader.read(), timeout=1) == b""
    finally:
        release.set()
        for _reader, writer in clients:
            writer.close()
            await asyncio.wait_for(writer.wait_closed(), timeout=1)
        for task in handlers:
            if not task.done():
                task.cancel()
        await asyncio.wait_for(asyncio.gather(*handlers, return_exceptions=True), timeout=1)


@pytest.mark.asyncio
async def test_a_redis_outage_is_logged_as_a_warning_once_a_minute(monkeypatch):
    from loguru import logger

    from tldw_Server_API.app.core.Resource_Governance import policy_eval

    async def _connection_refused(*_args, **_kwargs):
        raise RedisConnectionError("Connection refused")

    monkeypatch.setattr(policy_eval, "_last_redis_warning", {}, raising=False)  # no warning sent yet
    rg = _governor({"requests": {"rpm": 5}, "scopes": ["user"], "fail_mode": "fail_open"}, ns="rg_down_warn")
    rg._client = _redis_down_client()
    # Fix the logging stimulus, not the OS-dependent connect outcome.
    monkeypatch.setattr(rg._client, "execute_command", _connection_refused)
    req = RGRequest(entity="user:x", categories={"requests": {"units": 1}}, tags={"policy_id": "p"})
    seen = []
    sink = logger.add(lambda m: seen.append(m.record["message"]), level="WARNING")
    try:
        await rg.reserve(req)
        await rg.reserve(req)
    finally:
        logger.remove(sink)

    assert [m for m in seen if "Redis" in m] == [
        "Resource Governor Redis backend 'rg_down_warn' is failing (ConnectionError); this worker enforces its limits on its own until it recovers"
    ]
