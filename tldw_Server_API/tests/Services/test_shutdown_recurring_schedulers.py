from __future__ import annotations

import importlib.util

import pytest

pytestmark = pytest.mark.unit


def test_shutdown_recurring_schedulers_direct_stop_module_was_removed() -> None:
    assert (
        importlib.util.find_spec(
            "tldw_Server_API.app.services.shutdown_recurring_schedulers"
        )
        is None
    )
