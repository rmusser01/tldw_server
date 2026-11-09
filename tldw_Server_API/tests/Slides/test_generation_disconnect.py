"""Real ASGI receive ownership and cleanup for structured Slides requests."""

import asyncio

import pytest
from starlette.requests import Request

from tldw_Server_API.app.core.Slides.generation_disconnect import (
    SlidesGenerationDisconnectMiddleware,
    generation_disconnected,
)


async def discard(message):
    """No response content is needed by these transport ownership checks."""


@pytest.mark.parametrize(
    "scope",
    [
        {"type": "websocket", "path": "/api/v1/slides/generate"},
        {"type": "http", "method": "GET", "path": "/api/v1/slides/generate"},
        {"type": "http", "method": "POST", "path": "/api/v1/slides/presentations"},
    ],
)
def test_other_requests_keep_the_original_receive(scope):
    async def exercise():
        async def receive():
            raise AssertionError("Unrelated receive must not be consumed")

        async def app(actual_scope, actual_receive, actual_send):
            assert actual_scope is scope and actual_receive is receive and actual_send is discard  # nosec B101 - unchanged ASGI boundary.

        await SlidesGenerationDisconnectMiddleware(app)(scope, receive, discard)

    asyncio.run(exercise())


def test_segmented_body_is_preserved_and_connected_reader_retires():
    async def exercise():
        incoming = asyncio.Queue()
        incoming.put_nowait({"type": "http.request", "body": b"Row", "more_body": True})
        incoming.put_nowait({"type": "http.request", "body": b"an", "more_body": False})
        waiting, retired = asyncio.Event(), asyncio.Event()

        async def receive():
            if not incoming.empty():
                return await incoming.get()
            waiting.set()
            try:
                return await incoming.get()
            finally:
                retired.set()

        async def app(scope, receive, send):
            request = Request(scope, receive)
            assert await request.body() == b"Rowan"  # nosec B101 - preserve every input segment.
            assert not await generation_disconnected(request)  # nosec B101 - a connected caller remains admitted.
            assert waiting.is_set() and not retired.is_set()  # nosec B101 - the observer really owns the pending receive.

        await SlidesGenerationDisconnectMiddleware(app)(
            {"type": "http", "method": "POST", "path": "/api/v1/slides/generate"}, receive, discard
        )
        assert retired.is_set()  # nosec B101 - no receive survives the completed handler.

    asyncio.run(exercise())


@pytest.mark.parametrize("unexpected_message", [False, True])
def test_receive_failure_prevents_admission_and_preserves_primary(unexpected_message):
    async def exercise():
        calls = 0

        async def receive():
            nonlocal calls
            calls += 1
            if calls == 1:
                return {"type": "http.request", "body": b"{}", "more_body": False}
            if unexpected_message:
                return {"type": "http.request", "body": b"extra", "more_body": False}
            raise ValueError("Controlled receive failure")

        async def app(scope, receive, send):
            request = Request(scope, receive)
            await request.body()
            with pytest.raises(RuntimeError if unexpected_message else ValueError):
                await generation_disconnected(request)
            raise LookupError("Primary handler failure")

        with pytest.raises(LookupError, match="Primary handler failure") as caught:
            await SlidesGenerationDisconnectMiddleware(app)(
                {"type": "http", "method": "POST", "path": "/api/v1/slides/generate"}, receive, discard
            )
        assert any("disconnect observer also failed" in note for note in caught.value.__notes__)  # nosec B101 - retain both failure identities.

    asyncio.run(exercise())


def test_caller_cancellation_retires_the_pending_receive():
    async def exercise():
        entered, retired = asyncio.Event(), asyncio.Event()
        calls = 0

        async def receive():
            nonlocal calls
            calls += 1
            if calls == 1:
                return {"type": "http.request", "body": b"{}", "more_body": False}
            try:
                await asyncio.Event().wait()
            finally:
                retired.set()

        async def app(scope, receive, send):
            request = Request(scope, receive)
            await request.body()
            assert not await generation_disconnected(request)  # nosec B101 - cancellation starts from an admitted caller.
            entered.set()
            await asyncio.Event().wait()

        task = asyncio.create_task(
            SlidesGenerationDisconnectMiddleware(app)(
                {"type": "http", "method": "POST", "path": "/api/v1/slides/generate"}, receive, discard
            )
        )
        try:
            await asyncio.wait_for(entered.wait(), 1)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert retired.is_set()  # nosec B101 - cancellation cannot abandon its receive task.

    asyncio.run(exercise())
