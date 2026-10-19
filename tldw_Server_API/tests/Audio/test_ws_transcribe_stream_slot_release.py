"""/audio/stream/transcribe never leaves a stream slot held when session setup fails (TASK-13435)."""

import pytest
from starlette.websockets import WebSocketDisconnect

from tldw_Server_API.tests.Audio.ws_test_helpers import ws_client_without_lifespan

pytestmark = pytest.mark.unit


def test_setup_failure_after_admission_leaves_no_slot_held(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failure while the session is being set up must not strand a slot: every start has its finish."""
    import tldw_Server_API.app.api.v1.endpoints.audio.audio as audio_ep
    import tldw_Server_API.app.api.v1.endpoints.audio.audio_streaming as audio_streaming
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings
    from tldw_Server_API.app.main import app

    started: list[int] = []
    finished: list[int] = []

    async def _start(user_id: int) -> tuple[bool, str]:
        """Admit and record the start."""
        started.append(user_id)
        return True, "OK"

    async def _finish(user_id: int) -> None:
        """Record the finish."""
        finished.append(user_id)

    def _broken_setup() -> float:
        """Fail during session setup, after admission on the old code path."""
        raise RuntimeError("setup failed")

    monkeypatch.setattr(audio_ep, "can_start_stream", _start)
    monkeypatch.setattr(audio_ep, "finish_stream", _finish)
    monkeypatch.setattr(audio_streaming, "_get_failopen_cap_minutes", _broken_setup)
    token = get_settings().SINGLE_USER_API_KEY

    with ws_client_without_lifespan(app) as client:
        try:
            with client.websocket_connect(f"/api/v1/audio/stream/transcribe?token={token}") as ws:
                ws.receive_json()
        except (WebSocketDisconnect, RuntimeError):
            pass

    assert len(started) == len(finished)
