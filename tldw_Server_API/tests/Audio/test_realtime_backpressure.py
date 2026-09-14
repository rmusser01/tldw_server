"""Regressions for bounded generation and cancellation on a slow WebSocket."""

import asyncio
import contextlib
import json

import pytest

from tldw_Server_API.app.core.Audio.Realtime.handler import _run_realtime_loop
from tldw_Server_API.app.core.Audio.Realtime.models import (
    CommitAudioCommand,
    CreateResponseCommand,
    RealtimeLimits,
    ResponseDoneEvent,
)
from tldw_Server_API.app.core.Audio.Realtime.pipeline import RealtimePipelineTextDelta
from tldw_Server_API.app.core.Audio.Realtime.session import RealtimeSession

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


class SlowSocket:
    """Hold the first send while the test controls inbound messages."""

    def __init__(self):
        self.inbound = asyncio.Queue()
        self.send_started = asyncio.Event()
        self.release_send = asyncio.Event()
        self.sent = []

    async def receive(self):
        return await self.inbound.get()

    async def send_json(self, event):
        self.send_started.set()
        await self.release_send.wait()
        self.sent.append(event)

    async def command(self, event_type):
        await self.inbound.put({"type": "websocket.receive", "text": json.dumps({"type": event_type})})


class FastPipeline:
    """Generate a long answer without external timing or services."""

    def __init__(self):
        self.finished = False

    async def stream_turn(self, transcript, *, config):
        for _ in range(1000):
            yield RealtimePipelineTextDelta("chunk")
        self.finished = True
        await asyncio.Event().wait()


async def test_slow_socket_stops_generation_before_entire_answer_is_buffered():
    socket = SlowSocket()
    pipeline = FastPipeline()
    loop = asyncio.create_task(_run_realtime_loop(socket, RealtimeSession(pipeline=pipeline), RealtimeLimits()))
    try:
        await socket.command("response.create")
        await asyncio.wait_for(socket.send_started.wait(), timeout=1)
        for _ in range(30):
            await asyncio.sleep(0)
        assert not pipeline.finished
    finally:
        loop.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await loop


async def test_cancel_discards_buffered_deltas_before_next_session_event():
    socket = SlowSocket()
    loop = asyncio.create_task(_run_realtime_loop(socket, RealtimeSession(pipeline=FastPipeline()), RealtimeLimits()))
    try:
        await socket.command("response.create")
        await asyncio.wait_for(socket.send_started.wait(), timeout=1)
        await socket.command("response.cancel")
        socket.release_send.set()
        for _ in range(60):
            await asyncio.sleep(0)
        cancelled = next(
            i for i, event in enumerate(socket.sent) if event.get("response", {}).get("status") == "cancelled"
        )
        assert all(not event["type"].startswith("response.") for event in socket.sent[cancelled + 1 :])
    finally:
        loop.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await loop


async def test_response_done_dequeued_during_audio_commit_is_still_sent():
    release = asyncio.Event()
    active = asyncio.Event()
    committed = asyncio.Event()

    class Session:
        active_task = None

        def set_active_task(self, task):
            self.active_task = task

        async def handle_command(self, command):
            if isinstance(command, CreateResponseCommand):
                active.set()
                await release.wait()
                yield ResponseDoneEvent(event_id=None, response_id="response", status="completed")
            elif isinstance(command, CommitAudioCommand):
                release.set()
                await asyncio.gather(self.active_task)
                await asyncio.sleep(0)
                committed.set()

    socket = SlowSocket()
    socket.release_send.set()
    loop = asyncio.create_task(_run_realtime_loop(socket, Session(), RealtimeLimits()))
    try:
        await socket.command("response.create")
        await asyncio.wait_for(active.wait(), timeout=1)
        await socket.command("input_audio_buffer.commit")
        await asyncio.wait_for(committed.wait(), timeout=1)
        await socket.inbound.put({"type": "websocket.disconnect"})
        await asyncio.wait_for(loop, timeout=1)
        assert [event["type"] for event in socket.sent] == ["response.done"]
    finally:
        loop.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await loop
