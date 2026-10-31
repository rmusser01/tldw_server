"""Shared helpers for workflow adapter tests.

The ffmpeg-backed audio and video adapters await
``subprocess_utils.run_checked_async`` rather than calling ``subprocess.run``,
so that a media job cannot block the API event loop. Tests therefore patch that
seam instead of the global ``subprocess.run``.

Existing mocks are written as plain synchronous functions that take
``(cmd, **kwargs)`` and return a ``CompletedProcess``. Both helpers below accept
those unchanged and adapt them to the awaited interface, so a mock body does not
need to know whether the adapter is sync or async.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Iterator
from contextlib import ExitStack, contextmanager
from typing import Any
from unittest.mock import patch

import pytest

# Every adapter module that awaits the ffmpeg seam.
FFMPEG_ADAPTER_MODULES = (
    "tldw_Server_API.app.core.Workflows.adapters.audio.processing",
    "tldw_Server_API.app.core.Workflows.adapters.video.processing",
    "tldw_Server_API.app.core.Workflows.adapters.video.subtitles",
)


def as_async_run(fn: Callable[..., Any]) -> Callable[..., Awaitable[Any]]:
    """Adapt a synchronous ``subprocess.run``-shaped mock to the awaited seam."""

    async def _run(cmd: list[str], **kwargs: Any) -> Any:
        """Call the wrapped mock, awaiting it when it returns a coroutine."""
        result = fn(cmd, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    return _run


@contextmanager
def patch_ffmpeg(fn: Callable[..., Any]) -> Iterator[Callable[..., Awaitable[Any]]]:
    """Patch ``run_checked_async`` in every ffmpeg adapter module.

    Drop-in for ``patch("subprocess.run", fn)`` in these tests.
    """
    runner = as_async_run(fn)
    with ExitStack() as stack:
        for module in FFMPEG_ADAPTER_MODULES:
            stack.enter_context(patch(f"{module}.run_checked_async", runner))
        yield runner


def setattr_ffmpeg(
    monkeypatch: pytest.MonkeyPatch, fn: Callable[..., Any]
) -> Callable[..., Awaitable[Any]]:
    """monkeypatch-based equivalent of :func:`patch_ffmpeg`."""
    runner = as_async_run(fn)
    for module in FFMPEG_ADAPTER_MODULES:
        monkeypatch.setattr(f"{module}.run_checked_async", runner)
    return runner
