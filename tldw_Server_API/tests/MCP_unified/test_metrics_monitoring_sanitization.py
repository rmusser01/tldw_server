import asyncio

import pytest

from tldw_Server_API.app.core.MCP_unified.monitoring import metrics as metrics_mod
from tldw_Server_API.app.core.MCP_unified.monitoring.metrics import MetricsCollector


@pytest.mark.asyncio
async def test_metrics_collection_fallback_log_sanitizes_exception_detail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sensitive_detail = "metrics failed for /tmp/private-token.db with sk-mcp-secret-token"
    collector = MetricsCollector(enable_prometheus=False)

    async def _raise_leaky_metrics_error() -> None:
        raise RuntimeError(sensitive_detail)

    async def _cancel_after_fallback_sleep(_delay: float) -> None:
        raise asyncio.CancelledError

    messages: list[str] = []
    sink_id = metrics_mod.logger.add(
        lambda message: messages.append(str(message.record.get("message") or "")),
        level="ERROR",
    )
    monkeypatch.setattr(collector, "_sample_system_metrics", _raise_leaky_metrics_error)
    monkeypatch.setattr(metrics_mod.asyncio, "sleep", _cancel_after_fallback_sleep)

    try:
        with pytest.raises(asyncio.CancelledError):
            await collector._collection_loop()
    finally:
        metrics_mod.logger.remove(sink_id)

    rendered_logs = "\n".join(messages)
    assert "Error in metrics collection" in rendered_logs
    assert sensitive_detail not in rendered_logs
    assert "/tmp/private-token.db" not in rendered_logs
    assert "sk-mcp-secret-token" not in rendered_logs
