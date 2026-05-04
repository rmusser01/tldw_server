from __future__ import annotations

import importlib.util

import pytest

pytestmark = pytest.mark.unit


def test_shutdown_usage_aggregators_direct_stop_module_is_removed() -> None:
    assert (
        importlib.util.find_spec("tldw_Server_API.app.services.shutdown_usage_aggregators")
        is None
    )


def test_post_worker_tail_no_longer_has_usage_aggregator_direct_stop_adapter() -> None:
    from tldw_Server_API.app.services import shutdown_post_worker_services as shutdown_services

    assert not hasattr(shutdown_services, "_stop_usage_aggregators")
