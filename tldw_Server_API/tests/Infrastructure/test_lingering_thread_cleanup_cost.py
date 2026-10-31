"""The per-test lingering-thread sweep must not pay for the same thread twice."""

from __future__ import annotations

import logging
import threading
import time

import pytest

pytestmark = pytest.mark.unit


def test_a_known_lingering_thread_is_not_joined_again() -> None:
    """Running threads cannot be made daemon, so re-joining them cost 1s per test each.

    Two autouse fixtures run the sweep after every test; a handful of idle
    executor workers made that several seconds per test for the rest of the
    session, which is what pushed whole CI shards into the job timeout.
    """
    from tldw_Server_API.tests.conftest import _cleanup_lingering_threads

    release = threading.Event()
    worker = threading.Thread(target=release.wait, name="lingering-probe", daemon=False)
    worker.start()
    try:
        log = logging.getLogger(__name__)
        _cleanup_lingering_threads(log, context="first")
        assert worker.is_alive()

        started = time.monotonic()
        _cleanup_lingering_threads(log, context="second")
        assert time.monotonic() - started < 0.5, "an already-reported thread was joined again"
    finally:
        release.set()
        worker.join(timeout=5)
