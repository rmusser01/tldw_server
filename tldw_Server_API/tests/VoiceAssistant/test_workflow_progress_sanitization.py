from __future__ import annotations

import pytest

from tldw_Server_API.app.api.v1.endpoints import voice_assistant


pytestmark = pytest.mark.unit


class WebSocketRecorder:
    def __init__(self) -> None:
        self.sent_json: list[dict] = []

    async def send_json(self, payload: dict) -> None:
        self.sent_json.append(payload)


class FailingWorkflowRouter:
    async def stream_workflow_progress(self, *args, **kwargs):
        raise RuntimeError("workflow backend exploded /private/workflow.db")
        yield None


@pytest.mark.asyncio
async def test_stream_workflow_progress_sanitizes_failure_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logged_errors = []

    class LoggerStub:
        def error(self, message, *args, **kwargs):
            logged_errors.append((message, args, kwargs))

    monkeypatch.setattr(voice_assistant, "logger", LoggerStub())

    websocket = WebSocketRecorder()
    await voice_assistant._stream_workflow_progress(
        websocket=websocket,
        run_id="run-1",
        user_id=1,
        router_instance=FailingWorkflowRouter(),
    )

    assert logged_errors == [("Workflow progress streaming failed", (), {})]
