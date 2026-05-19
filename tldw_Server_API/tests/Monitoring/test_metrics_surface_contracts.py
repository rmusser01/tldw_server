import pytest

import tldw_Server_API.app.api.v1.endpoints.metrics as metrics_endpoint
import tldw_Server_API.app.core.Chat.chat_metrics as chat_metrics_module
from tldw_Server_API.app.core.Chat.chat_metrics import get_chat_metrics
from tldw_Server_API.app.core.Metrics.metrics_manager import (
    MetricDefinition,
    MetricType,
    get_metrics_registry,
)
from tldw_Server_API.app.main import metrics as root_metrics

pytestmark = pytest.mark.monitoring


@pytest.mark.asyncio
async def test_root_metrics_matches_router_text_export(monkeypatch):
    registry = get_metrics_registry()
    registry.reset()
    registry.register_metric(
        MetricDefinition(
            name="surface_contract_counter_total",
            type=MetricType.COUNTER,
            description="surface parity test",
            labels=["source"],
        )
    )
    registry.increment("surface_contract_counter_total", labels={"source": "test"})

    response_root = await root_metrics()
    response_router = await metrics_endpoint.get_prometheus_metrics()

    assert response_root.body == response_router.body
    assert response_root.headers["cache-control"] == response_router.headers["cache-control"]
    assert response_root.media_type == response_router.media_type


@pytest.mark.asyncio
async def test_chat_metrics_endpoint_reports_emitted_request_totals(monkeypatch):
    monkeypatch.setattr(chat_metrics_module, "_chat_metrics_collector", None, raising=False)
    chat_metrics_module.reset_endpoint_metrics_snapshot()
    collector = get_chat_metrics()

    async with collector.track_request(
        provider="openai",
        model="gpt-4",
        streaming=False,
        client_id="client-1",
    ):
        pass

    collector.track_tokens(
        prompt_tokens=11,
        completion_tokens=7,
        model="gpt-4",
        provider="openai",
    )

    payload = await metrics_endpoint.get_chat_metrics_endpoint()

    request_stats = payload["metrics"]["chat_requests_total"]
    duration_stats = payload["metrics"]["chat_request_duration_seconds"]
    prompt_stats = payload["metrics"]["chat_tokens_prompt"]

    assert float(request_stats["count"]) == 1.0
    assert float(request_stats["sum"]) == 1.0
    assert float(request_stats["latest"]) == 1.0

    assert float(duration_stats["count"]) == 1.0
    assert float(duration_stats["sum"]) == float(duration_stats["latest"])

    assert float(prompt_stats["count"]) == 1.0
    assert float(prompt_stats["sum"]) == 11.0
    assert float(prompt_stats["latest"]) == 11.0


@pytest.mark.asyncio
async def test_metrics_reset_clears_chat_endpoint_snapshot(monkeypatch):
    monkeypatch.setattr(chat_metrics_module, "_chat_metrics_collector", None, raising=False)
    chat_metrics_module.reset_endpoint_metrics_snapshot()
    collector = get_chat_metrics()

    async with collector.track_request(
        provider="openai",
        model="gpt-4",
        streaming=False,
        client_id="client-reset",
    ):
        pass

    collector.track_tokens(
        prompt_tokens=13,
        completion_tokens=3,
        model="gpt-4",
        provider="openai",
    )

    payload_before = await metrics_endpoint.get_chat_metrics_endpoint()
    assert float(payload_before["metrics"]["chat_requests_total"]["sum"]) == 1.0
    assert float(payload_before["metrics"]["chat_tokens_prompt"]["sum"]) == 13.0

    await metrics_endpoint.reset_metrics()

    payload_after = await metrics_endpoint.get_chat_metrics_endpoint()
    assert payload_after["metrics"] == {}
    assert payload_after["active_operations"] == {
        "active_requests": 0,
        "active_streams": 0,
        "active_transactions": 0,
    }
