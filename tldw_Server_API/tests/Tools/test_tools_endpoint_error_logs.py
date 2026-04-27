from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints import tools


pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_execute_tool_sanitizes_denied_or_invalid_log(monkeypatch):
    logger_stub = MagicMock()

    class _FailingExecutor:
        async def execute(self, **_kwargs):
            raise tools.ToolExecutionError("tool denied after reading /private/tool-config.json")

    monkeypatch.setattr(tools, "logger", logger_stub)
    monkeypatch.setattr(tools, "ToolExecutor", lambda: _FailingExecutor())

    with pytest.raises(HTTPException) as excinfo:
        await tools.execute_tool_endpoint(
            req=tools.ExecuteToolRequest(tool_name="secret.tool"),
            current_user=SimpleNamespace(id=42, username="reader"),
        )

    assert excinfo.value.status_code == 403
    assert excinfo.value.detail == "tool denied after reading /private/tool-config.json"
    logger_stub.warning.assert_called_once_with("tools_execute_denied_or_invalid")
