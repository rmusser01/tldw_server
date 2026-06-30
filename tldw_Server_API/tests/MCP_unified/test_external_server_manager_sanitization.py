import pytest

from tldw_Server_API.app.core.MCP_unified.external_servers import manager as manager_mod


@pytest.mark.asyncio
async def test_external_server_manager_shutdown_sanitizes_adapter_close_log():
    sensitive_error = "adapter close leaked /tmp/mcp-secret-token"

    class BrokenAdapter:
        async def close(self):
            raise RuntimeError(sensitive_error)

    manager = manager_mod.ExternalServerManager()
    manager._adapters = {"private-server": BrokenAdapter()}

    messages: list[str] = []
    sink_id = manager_mod.logger.add(
        lambda message: messages.append(str(message)),
        level="WARNING",
    )
    try:
        await manager.shutdown()
    finally:
        manager_mod.logger.remove(sink_id)

    rendered_logs = "\n".join(messages)
    assert manager._adapters == {}
    assert "External MCP adapter close failed" in rendered_logs
    assert sensitive_error not in rendered_logs
    assert "/tmp/mcp-secret-token" not in rendered_logs
