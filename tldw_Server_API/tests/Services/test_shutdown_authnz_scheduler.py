from __future__ import annotations

import importlib.util

import pytest

pytestmark = pytest.mark.unit


def test_shutdown_authnz_scheduler_direct_stop_module_was_removed() -> None:
    assert importlib.util.find_spec("tldw_Server_API.app.services.shutdown_authnz_scheduler") is None
