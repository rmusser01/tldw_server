import json
import os
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints import persona as persona_ep
from tldw_Server_API.app.core.Metrics.metrics_manager import get_metrics_registry


@pytest.mark.asyncio
async def test_persona_ws_emits_ws_latency_metrics(monkeypatch):
    """Ensure Persona WS sends frames via WebSocketStream and increments ws_send_latency_ms with labels."""
    async def _fake_resolve(*args, **kwargs):
        return "1", True, True

    monkeypatch.setattr(persona_ep, "_resolve_authenticated_user_id", _fake_resolve)
    monkeypatch.setattr("tldw_Server_API.app.core.feature_flags.is_persona_enabled", lambda: True)

    app = FastAPI()
    app.include_router(persona_ep.router, prefix="/api/v1/persona")

    reg = get_metrics_registry()
    before = reg.get_metric_stats(
        "ws_send_latency_ms",
        labels={"component": "persona", "endpoint": "persona_ws", "transport": "ws"},
    ).get("count", 0)

    with TestClient(app) as client:
        try:
            ws = client.websocket_connect("/api/v1/persona/stream")
        except Exception:
            pytest.skip("persona WebSocket endpoint not available")

        with ws as ws:
            # Expect a server notice on connect; then send a simple user message
            try:
                _ = ws.receive_json()
            except Exception:
                _ = None
            ws.send_text(json.dumps({"type": "user_message", "text": "hello"}))
            try:
                _ = ws.receive_json()
            except Exception:
                _ = None

    after = reg.get_metric_stats(
        "ws_send_latency_ms",
        labels={"component": "persona", "endpoint": "persona_ws", "transport": "ws"},
    ).get("count", 0)

    assert after >= before + 1
