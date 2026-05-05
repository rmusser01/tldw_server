from __future__ import annotations

import pytest


@pytest.mark.unit
def test_hub_end_dedup_and_cleanup_drain_buffer() -> None:
     # Use a fresh hub instance to avoid global state
    from tldw_Server_API.app.core.Sandbox.streams import RunStreamHub

    hub = RunStreamHub()
    run_id = "run-hub-1"

    q = hub.subscribe(run_id)
    hub.publish_event(run_id, "start", {"ok": True})
    hub.publish_stdout(run_id, b"hello", max_log_bytes=64)
    hub.publish_event(run_id, "end", {})
    # Duplicate end should be ignored by deduplication
    hub.publish_event(run_id, "end", {})

    # Drain buffered frames into the queue (no loop configured)
    hub.drain_buffer(run_id, q)

    frames = []
    while True:
        try:
            frames.append(q.get_nowait())
        except Exception:
            break

    # Exactly one end event
    end_frames = [f for f in frames if f.get("type") == "event" and f.get("event") == "end"]
    assert len(end_frames) == 1

    # Sequence numbers should be present and strictly increasing
    seqs = [int(f["seq"]) for f in frames if "seq" in f]
    assert seqs == sorted(seqs) and len(seqs) == len(set(seqs))

    # Close should cleanup all per-run state
    hub.close(run_id)
    assert run_id not in hub._queues
    assert run_id not in hub._buffers
    assert run_id not in hub._log_bytes
    assert run_id not in hub._truncated
    assert run_id not in hub._ended
    assert run_id not in hub._seq


@pytest.mark.unit
def test_hub_close_publishes_end_and_cleans(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.core.Sandbox.streams import RunStreamHub

    hub = RunStreamHub()
    run_id = "run-hub-2"

    published: list[tuple[str, dict]] = []

    def _fake_publish(rid: str, frame: dict) -> None:
        published.append((rid, frame))

    # Intercept low-level publish to observe frames emitted by close()
    monkeypatch.setattr(hub, "_publish", _fake_publish, raising=True)

    # No prior end event; close should emit exactly one end event then cleanup
    hub.close(run_id)

    ends = [f for rid, f in published if rid == run_id and f.get("type") == "event" and f.get("event") == "end"]
    assert len(ends) == 1

    # Cleanup performed
    assert run_id not in hub._queues
    assert run_id not in hub._buffers
    assert run_id not in hub._log_bytes
    assert run_id not in hub._truncated
    assert run_id not in hub._ended
    assert run_id not in hub._seq


@pytest.mark.unit
def test_hub_close_respects_dedup_after_end(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.core.Sandbox.streams import RunStreamHub

    hub = RunStreamHub()
    run_id = "run-hub-3"

    published: list[tuple[str, dict]] = []

    def _fake_publish(rid: str, frame: dict) -> None:
        published.append((rid, frame))

    monkeypatch.setattr(hub, "_publish", _fake_publish, raising=True)

    # First end emission via publish_event
    hub.publish_event(run_id, "end", {})
    ends_before_close = [f for rid, f in published if rid == run_id and f.get("type") == "event" and f.get("event") == "end"]
    assert len(ends_before_close) == 1

    # close() should not emit another end due to deduplication, and should cleanup
    hub.close(run_id)
    ends_after_close = [f for rid, f in published if rid == run_id and f.get("type") == "event" and f.get("event") == "end"]
    assert len(ends_after_close) == 1  # unchanged

    # Cleanup performed
    assert run_id not in hub._queues
    assert run_id not in hub._buffers
    assert run_id not in hub._log_bytes
    assert run_id not in hub._truncated
    assert run_id not in hub._ended
    assert run_id not in hub._seq


@pytest.mark.unit
def test_hub_unsubscribe_removes_only_target_queue() -> None:
    from tldw_Server_API.app.core.Sandbox.streams import RunStreamHub

    hub = RunStreamHub()
    run_id = "run-hub-4"
    q1 = hub.subscribe(run_id)
    q2 = hub.subscribe(run_id)

    hub.unsubscribe(run_id, q1)
    remaining = hub._queues.get(run_id) or []
    assert len(remaining) == 1
    assert remaining[0][1] is q2

    hub.unsubscribe(run_id, q2)
    assert run_id not in hub._queues


@pytest.mark.unit
def test_hub_marks_single_oversized_chunk_as_log_truncated() -> None:
    from tldw_Server_API.app.core.Sandbox.streams import RunStreamHub

    hub = RunStreamHub()
    run_id = "run-hub-5"
    queue = hub.subscribe(run_id)

    hub.publish_stdout(run_id, b"abcdef", max_log_bytes=5)
    hub.drain_buffer(run_id, queue)

    frames = []
    while True:
        try:
            frames.append(queue.get_nowait())
        except Exception:
            break

    assert hub.get_log_bytes(run_id) == 5
    assert hub.is_log_truncated(run_id)
    assert any(
        frame.get("type") == "truncated" and frame.get("reason") == "log_cap"
        for frame in frames
    )


@pytest.mark.unit
def test_hub_log_truncation_publishes_after_releasing_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.core.Sandbox.streams import RunStreamHub

    hub = RunStreamHub()
    run_id = "run-hub-6"
    published: list[dict[str, object]] = []

    def _publish_without_lock(_run_id: str, frame: dict[str, object]) -> None:
        assert not hub._lock._is_owned()  # type: ignore[attr-defined]
        published.append(dict(frame))

    monkeypatch.setattr(hub, "_publish", _publish_without_lock)

    hub.mark_log_truncated(run_id)

    assert hub.is_log_truncated(run_id)
    assert published == [{"type": "truncated", "reason": "log_cap"}]
