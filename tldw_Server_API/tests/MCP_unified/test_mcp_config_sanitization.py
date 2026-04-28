import pytest

from tldw_Server_API.app.core.MCP_unified import config as mcp_config


def _capture_logs(level: str = "ERROR") -> tuple[list[str], int]:
    messages: list[str] = []
    sink_id = mcp_config.logger.add(
        lambda message: messages.append(str(message.record.get("message") or "")),
        level=level,
    )
    return messages, sink_id


def test_get_config_failure_log_is_sanitized(monkeypatch):
    sensitive_detail = "config failed at /tmp/mcp-secret-token"

    class BrokenConfig:
        def __init__(self):
            raise RuntimeError(sensitive_detail)

    mcp_config.get_config.cache_clear()
    monkeypatch.setattr(mcp_config, "MCPConfig", BrokenConfig)
    messages, sink_id = _capture_logs()

    try:
        with pytest.raises(RuntimeError):
            mcp_config.get_config()
    finally:
        mcp_config.logger.remove(sink_id)
        mcp_config.get_config.cache_clear()

    rendered_logs = "\n".join(messages)
    assert "Failed to load configuration" in rendered_logs
    assert sensitive_detail not in rendered_logs
    assert "/tmp/mcp-secret-token" not in rendered_logs


def test_validate_config_failure_log_is_sanitized(monkeypatch):
    sensitive_detail = "validation failed for token sk-mcp-secret"

    def fail_get_config():
        raise RuntimeError(sensitive_detail)

    monkeypatch.setattr(mcp_config, "get_config", fail_get_config)
    messages, sink_id = _capture_logs()

    try:
        result = mcp_config.validate_config()
    finally:
        mcp_config.logger.remove(sink_id)

    rendered_logs = "\n".join(messages)
    assert result is False
    assert "Configuration validation failed" in rendered_logs
    assert sensitive_detail not in rendered_logs
    assert "sk-mcp-secret" not in rendered_logs
