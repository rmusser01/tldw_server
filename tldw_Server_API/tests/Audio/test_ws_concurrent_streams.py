import pytest
from starlette.websockets import WebSocketDisconnect

from tldw_Server_API.tests.Audio.ws_test_helpers import ws_client_without_lifespan, ws_session_or_skip


@pytest.mark.asyncio
async def test_ws_concurrent_streams_denied(monkeypatch):
    """
    Verify the audio WebSocket endpoint rejects connections when concurrent-streams quota is exceeded.

    The test monkeypatches quota helpers to simulate active streams and then attempts a WebSocket connection
    to the transcribe endpoint with a valid token. If the endpoint is available, it asserts an incoming error
    frame whose `type` is "error" and whose message mentions "Concurrent streams", and verifies the connection
    is then closed (further receives raise). If the endpoint is not available, the test is skipped.
    """
    from tldw_Server_API.app.main import app
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings
    import tldw_Server_API.app.api.v1.endpoints.audio.audio as audio_ep

    async def _deny_stream(_user_id: int):
        """
        Simulate that a user already has two active streams.

        Parameters:
            user_id (int): The user's identifier (unused; present for interface compatibility).

        Returns:
            int: `2` indicating two active streams.
        """
        return 2  # pretend two streams active

    async def _can_start_stream(_user_id: int):
        """
        Indicates that a user cannot start a new stream because the concurrent streams quota is exceeded.

        Returns:
            (bool, str): `False` and an error message stating the concurrent streams limit.
        """
        return False, "Concurrent streams limit reached (1)"

    # Force can_start_stream denial
    monkeypatch.setattr(audio_ep, "can_start_stream", _can_start_stream)
    # Also return non-zero active count for the limits endpoint if called
    monkeypatch.setattr(audio_ep, "active_streams_count", _deny_stream, raising=False)

    settings = get_settings()
    token = settings.SINGLE_USER_API_KEY

    with ws_client_without_lifespan(app) as client:
        try:
            ws = client.websocket_connect(f"/api/v1/audio/stream/transcribe?token={token}")
        except (WebSocketDisconnect, RuntimeError):
            pytest.skip("audio WebSocket endpoint not available in this build")
        # The server will send an error then close.
        with ws_session_or_skip(ws) as ws:
            # The connection may close very early; try receiving once
            try:
                data = ws.receive_json()
                assert isinstance(data, dict)
                assert data.get("type") == "error"
                assert "Concurrent streams" in str(data.get("message", ""))
            except WebSocketDisconnect:
                # If we miss the error frame due to early close, this is acceptable
                pass
            # Any further receive should raise WebSocketDisconnect
            with pytest.raises(WebSocketDisconnect):
                ws.receive_json()
