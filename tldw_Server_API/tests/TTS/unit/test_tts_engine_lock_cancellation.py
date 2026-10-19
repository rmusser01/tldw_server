"""A cancelled request keeps a local TTS engine until its thread call returns (TASK-13856).

asyncio.to_thread cannot stop a thread, so an adapter that guards its engine with a lock must not
hand the engine to the next request while a cancelled request's call still runs. Starlette cancels
a disconnected request through an anyio cancel scope, which re-cancels on every await, so each
site is checked with a plain task cancel and with an anyio scope.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Callable

import anyio
import numpy as np
import pytest

from tldw_Server_API.app.core.TTS.adapters.base import AudioFormat, ProviderStatus, TTSRequest

pytestmark = pytest.mark.unit

CANCEL_MODES = pytest.mark.parametrize("via_anyio_scope", [False, True], ids=["task-cancel", "anyio-scope"])


class _Gate:
    """Engine calls record that they started, then block until released, so a caller can be cancelled mid-call."""

    def __init__(self) -> None:
        self.started: list[str] = []
        self.release = threading.Event()

    def enter(self, name: str) -> None:
        self.started.append(name)
        if not self.release.wait(5):
            raise RuntimeError("the test never released the engine call")


async def _assert_cancelled_call_keeps_the_engine(
    call: Callable[[], Awaitable[object]],
    gate: _Gate,
    *,
    via_anyio_scope: bool,
) -> None:
    """Cancel one call mid-engine-call while a second waits; the second must not start until the first returns."""
    scopes: list[anyio.CancelScope] = []

    async def _first() -> None:
        if not via_anyio_scope:
            await call()
            return
        with anyio.CancelScope() as scope:
            scopes.append(scope)
            await call()

    first = asyncio.create_task(_first())
    try:
        for _ in range(500):
            if gate.started:
                break
            await asyncio.sleep(0.01)
        assert gate.started, "the first call never reached the engine"
        second = asyncio.create_task(call())
        await asyncio.sleep(0.05)
        if via_anyio_scope:
            scopes[0].cancel()
        else:
            first.cancel()
        await asyncio.sleep(0.1)
        # The cancelled request's engine call is still running, so the next one has not started.
        assert len(gate.started) == 1
    finally:
        gate.release.set()
    # Bounded waits: a lock-leak regression must fail here, not hang the run until pytest-timeout kills it (Review M1).
    await asyncio.wait_for(second, 5)
    assert len(gate.started) == 2
    if via_anyio_scope:
        await asyncio.wait_for(first, 5)
    else:
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(first, 5)


_WAV_HEADER = b"RIFF\x24\x00\x00\x00WAVEfmt "


def _pocket_adapter(gate: _Gate):
    from tldw_Server_API.app.core.TTS.adapters.pocket_tts_adapter import PocketTTSOnnxAdapter

    class _Engine:
        def generate(self, text, voice, max_frames=500):
            gate.enter("generate")
            return np.zeros(240, dtype=np.float32)

        def stream(self, text, voice, **_kwargs):
            gate.enter("stream")
            yield np.zeros(240, dtype=np.float32)

    adapter = PocketTTSOnnxAdapter({})
    adapter._initialized = True
    adapter._status = ProviderStatus.AVAILABLE
    adapter._engine = _Engine()
    return adapter


def _pocket_request(*, stream: bool) -> TTSRequest:
    return TTSRequest(
        text="hello",
        voice="clone",
        format=AudioFormat.PCM,
        stream=stream,
        voice_reference=_WAV_HEADER,
        extra_params={"validate_reference": False, "convert_reference": False},
    )


@pytest.mark.asyncio
@CANCEL_MODES
async def test_pocket_tts_generate_keeps_the_engine_until_its_call_returns(via_anyio_scope):
    gate = _Gate()
    adapter = _pocket_adapter(gate)

    async def call():
        return await adapter.generate(_pocket_request(stream=False))

    await _assert_cancelled_call_keeps_the_engine(call, gate, via_anyio_scope=via_anyio_scope)


@pytest.mark.asyncio
@CANCEL_MODES
async def test_pocket_tts_stream_keeps_the_engine_until_its_call_returns(via_anyio_scope):
    gate = _Gate()
    adapter = _pocket_adapter(gate)

    async def call():
        response = await adapter.generate(_pocket_request(stream=True))
        return [chunk async for chunk in response.audio_stream]

    await _assert_cancelled_call_keeps_the_engine(call, gate, via_anyio_scope=via_anyio_scope)
