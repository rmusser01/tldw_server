from __future__ import annotations

import asyncio
import threading
import types
import uuid

import pytest


pytestmark = pytest.mark.timeout(10)


@pytest.mark.asyncio
async def test_hub_resume_tail_and_gap() -> None:
    from tldw_Server_API.app.core.Sandbox.streams import get_hub

    hub = get_hub()
    loop = asyncio.get_running_loop()
    hub.set_loop(loop)

    run_id = f"run-resume-{uuid.uuid4().hex}"

    # Publish enough frames to exceed the 100-frame buffer and force trimming
    for i in range(105):
        hub.publish_stdout(run_id, f"line-{i}\n".encode("utf-8"), max_log_bytes=1_000_000)

    # Wait until buffer has 100 frames and last seq appears to be 105
    async def _wait_for_buffer(target_len: int, last_seq: int, timeout: float = 2.0):
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            buf = hub.get_buffer_snapshot(run_id)
            if len(buf) == target_len and buf[-1].get("seq") == last_seq:
                return True
            await asyncio.sleep(0.01)
        return False

    assert await _wait_for_buffer(100, 105)

    # Tail: resume from last known seq → should receive exactly that seq (prefill equality allowed)
    q_tail = hub.subscribe_with_buffer_from_seq(run_id, 105)
    f_tail = await asyncio.wait_for(q_tail.get(), timeout=1.0)
    assert f_tail.get("type") in {"stdout", "stderr"}
    assert int(f_tail.get("seq")) == 105
    # No more buffered frames should arrive immediately
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(q_tail.get(), timeout=0.1)

    # Gap: resume from a very old seq → should start at earliest buffered
    buf = hub.get_buffer_snapshot(run_id)
    earliest_seq = int(buf[0].get("seq"))
    latest_seq = int(buf[-1].get("seq"))

    q_gap = hub.subscribe_with_buffer_from_seq(run_id, 1)
    got = []
    for _ in range(len(buf)):
        got.append(await asyncio.wait_for(q_gap.get(), timeout=1.0))

    seqs = [int(f.get("seq")) for f in got]
    assert seqs[0] == earliest_seq
    assert seqs[-1] == latest_seq
    # Strictly increasing, no duplicates
    assert seqs == sorted(seqs)
    assert len(seqs) == len(set(seqs))

    # Cleanup
    hub.cleanup_run(run_id)


@pytest.mark.asyncio
async def test_hub_multi_subscriber_ordering() -> None:
    from tldw_Server_API.app.core.Sandbox.streams import get_hub

    hub = get_hub()
    loop = asyncio.get_running_loop()
    hub.set_loop(loop)

    run_id = f"run-multi-{uuid.uuid4().hex}"

    # Seed frames 1..3
    for i in range(3):
        hub.publish_stdout(run_id, f"seed-{i}\n".encode("utf-8"), max_log_bytes=1_000_000)

    # Wait until seq reaches 3 to avoid races
    async def _wait_last_seq(n: int, timeout: float = 1.0):
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            buf = hub.get_buffer_snapshot(run_id)
            if buf and int(buf[-1].get("seq", 0)) >= n:
                return True
            await asyncio.sleep(0.01)
        return False

    assert await _wait_last_seq(3)

    # Subscriber A resumes from seq=2 (prefill 2..3)
    qa = hub.subscribe_with_buffer_from_seq(run_id, 2)
    a_prefill = [await asyncio.wait_for(qa.get(), timeout=1.0), await asyncio.wait_for(qa.get(), timeout=1.0)]
    a_prefill_seqs = [int(f.get("seq")) for f in a_prefill]
    assert a_prefill_seqs == [2, 3]

    # Subscriber B subscribes for live frames only
    qb = hub.subscribe(run_id)

    # Publish frames 4..6 (live)
    for i in range(3):
        hub.publish_stdout(run_id, f"live-{i}\n".encode("utf-8"), max_log_bytes=1_000_000)

    # Collect: A should see 2..6; B should see 4..6
    a_more = [await asyncio.wait_for(qa.get(), timeout=1.0) for _ in range(3)]
    b_all = [await asyncio.wait_for(qb.get(), timeout=1.0) for _ in range(3)]

    a_seqs = a_prefill_seqs + [int(f.get("seq")) for f in a_more]
    b_seqs = [int(f.get("seq")) for f in b_all]

    assert a_seqs == [2, 3, 4, 5, 6]
    assert b_seqs == [4, 5, 6]
    # Overlapping range should match order and values across subscribers
    assert a_seqs[-3:] == b_seqs

    # Cleanup
    hub.cleanup_run(run_id)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_hub_subscriber_attaching_before_dispatch_sees_each_frame_once() -> None:
    """Frames published but not yet dispatched reach a new subscriber exactly once, in seq order."""
    from tldw_Server_API.app.core.Sandbox.streams import RunStreamHub

    hub = RunStreamHub()
    hub.set_loop(asyncio.get_running_loop())
    run_id = f"run-attach-{uuid.uuid4().hex}"

    early = hub.subscribe_with_buffer(run_id)
    # Publishing from the loop thread schedules dispatch; it cannot run until we await.
    hub.publish_event(run_id, "start", {})
    hub.publish_heartbeat(run_id)
    hub.publish_stdout(run_id, b"A\n", max_log_bytes=1024)
    late = hub.subscribe_with_buffer(run_id)
    resumed = hub.subscribe_with_buffer_from_seq(run_id, 1)

    # Seq follows publish order. Heartbeats are live-only (never buffered), so
    # only the subscriber attached before it was published sees seq 2.
    history = [("event", 1), ("stdout", 3)]
    for q, expected in (
        (early, [("event", 1), ("heartbeat", 2), ("stdout", 3)]),
        (late, history),
        (resumed, history),
    ):
        got = [await asyncio.wait_for(q.get(), timeout=1.0) for _ in expected]
        assert [(f["type"], f["seq"]) for f in got] == expected
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(q.get(), timeout=0.1)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_drain_buffer_numbers_frames_in_publish_order() -> None:
    """drain_buffer must not number a buffered frame ahead of an earlier queued heartbeat."""
    from tldw_Server_API.app.core.Sandbox.streams import RunStreamHub

    hub = RunStreamHub()
    hub.set_loop(asyncio.get_running_loop())
    run_id = f"run-drain-{uuid.uuid4().hex}"

    # Dispatch is scheduled on this loop and cannot run until we await.
    hub.publish_heartbeat(run_id)
    hub.publish_stdout(run_id, b"A\n", max_log_bytes=1024)
    drained: asyncio.Queue = asyncio.Queue()
    hub.drain_buffer(run_id, drained)

    frame = drained.get_nowait()
    assert (frame["type"], frame["seq"]) == ("stdout", 2)
    assert drained.empty()


@pytest.mark.unit
def test_hub_slow_fanout_on_one_run_does_not_block_other_runs() -> None:
    """A dispatcher stuck copying one run's frame must not stall other runs."""
    from tldw_Server_API.app.core.Sandbox.streams import RunStreamHub

    copying = threading.Event()
    release = threading.Event()

    class _SlowPayload:
        def __deepcopy__(self, memo: dict) -> _SlowPayload:
            copying.set()
            release.wait(5)
            return self

    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
    loop_thread.start()
    try:
        hub = RunStreamHub()
        hub.set_loop(loop)
        hub.subscribe("run-slow")
        hub.publish_event("run-slow", "start", {"payload": _SlowPayload()})
        assert copying.wait(2), "dispatcher never reached the slow fan-out"

        def _other_run() -> None:
            hub.subscribe_with_buffer("run-fast")
            hub.publish_stdout("run-fast", b"hi\n", max_log_bytes=1024)

        other = threading.Thread(target=_other_run, daemon=True)
        other.start()
        other.join(2)
        assert not other.is_alive(), "another run's subscribe/publish blocked behind a slow fan-out"
    finally:
        release.set()
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(2)
        loop.close()


@pytest.mark.asyncio
async def test_hub_redis_no_duplicate_local_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    # Enable fan-out and inject a fake redis that just records publish calls
    chan = f"test:sandbox:dedup:{uuid.uuid4().hex}"
    monkeypatch.setenv("SANDBOX_WS_REDIS_FANOUT", "true")
    monkeypatch.setenv("SANDBOX_REDIS_URL", "redis://fake")
    monkeypatch.setenv("SANDBOX_WS_REDIS_CHANNEL", chan)

    class _FakeRedis:
        def __init__(self):
            self.publishes: list[tuple[str, bytes]] = []

        @classmethod
        def from_url(cls, url: str):
            return cls()

        def ping(self):

            return True

        def publish(self, channel: str, data: bytes):
            self.publishes.append((channel, data))

        def pubsub(self, ignore_subscribe_messages: bool = True):
            # Not used in this test
            class _Noop:
                def subscribe(self, channel: str) -> None:
                    return None

                def listen(self):  # pragma: no cover - not exercised here
                    if False:
                        yield None

            return _Noop()

    fake_mod = types.SimpleNamespace(Redis=_FakeRedis)
    monkeypatch.setitem(__import__("sys").modules, "redis", fake_mod)

    # Construct a fresh hub so that fan-out initialization runs with fakes
    from tldw_Server_API.app.core.Sandbox.streams import RunStreamHub

    hub = RunStreamHub()
    loop = asyncio.get_running_loop()
    hub.set_loop(loop)

    run_id = f"run-redis-dedup-{uuid.uuid4().hex}"
    q = hub.subscribe_with_buffer(run_id)
    hub.publish_stdout(run_id, b"hello", max_log_bytes=1024)

    # Expect exactly one local delivery
    frame = await asyncio.wait_for(q.get(), timeout=1.0)
    assert frame.get("type") == "stdout"
    # Ensure no duplicate immediately follows
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(q.get(), timeout=0.2)
