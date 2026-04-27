# test_ws_integration.py
# WebSocket integration tests for Voice Assistant endpoint
#
#######################################################################################################################
import asyncio
import base64
import json
from types import SimpleNamespace
from typing import Any, AsyncIterator, Dict, Iterable, List, Optional

import pytest

from tldw_Server_API.app.api.v1.endpoints import voice_assistant


class DummyWebSocket:
    """In-memory WebSocket stub for Voice Assistant WebSocket tests."""

    def __init__(self, messages: Iterable[Dict[str, Any] | str]) -> None:
        self.headers: Dict[str, str] = {}
        self.query_params: Dict[str, str] = {}
        self.client = SimpleNamespace(host="127.0.0.1")
        self._messages: List[str] = [
            json.dumps(m) if isinstance(m, dict) else m for m in messages
        ]
        self.sent_bytes: List[bytes] = []
        self.sent_json: List[Dict[str, Any]] = []
        self.accepted: bool = False
        self.closed: bool = False
        self.close_code: Optional[int] = None
        self.close_calls: List[int] = []

    async def accept(self) -> None:
        """Mark the WebSocket as accepted."""
        self.accepted = True

    async def receive_json(self) -> Dict[str, Any]:
        """Return the next queued JSON message, or raise when exhausted."""
        if not self._messages:
            from fastapi import WebSocketDisconnect
            raise WebSocketDisconnect(code=1000)
        msg = self._messages.pop(0)
        if isinstance(msg, str):
            return json.loads(msg)
        return msg

    async def send_bytes(self, data: bytes) -> None:
        """Record bytes sent over the WebSocket."""
        self.sent_bytes.append(data)

    async def send_json(self, payload: Dict[str, Any]) -> None:
        """Record JSON payloads sent over the WebSocket."""
        self.sent_json.append(payload)

    async def close(self, code: int = 1000, reason: Optional[str] = None) -> None:
        """Record the close code and mark the WebSocket as closed."""
        self.close_calls.append(code)
        if not self.closed:
            self.close_code = code
        self.closed = True


class DummySessionManager:
    """Session manager stub for testing."""

    MAX_SESSIONS_PER_USER = 5

    def __init__(self) -> None:
        self.sessions: Dict[str, Any] = {}
        self._user_sessions: Dict[int, set] = {}
        self.session_counter = 0

    async def start(self) -> None:
        """No-op start hook to match VoiceSessionManager."""
        return None

    async def create_session(self, user_id: int, metadata: Optional[Dict] = None) -> Any:
        """Create a dummy session."""
        from tldw_Server_API.app.core.VoiceAssistant.schemas import VoiceSessionContext, VoiceSessionState
        from datetime import datetime
        import uuid

        session_id = str(uuid.uuid4())
        session = VoiceSessionContext(
            session_id=session_id,
            user_id=user_id,
            state=VoiceSessionState.IDLE,
            metadata=metadata or {},
            created_at=datetime.now(),
            last_activity=datetime.now(),
        )
        self.sessions[session_id] = session

        if user_id not in self._user_sessions:
            self._user_sessions[user_id] = set()
        self._user_sessions[user_id].add(session_id)

        return session

    async def get_session(self, session_id: str, touch: bool = True) -> Optional[Any]:
        """Get a session by ID."""
        return self.sessions.get(session_id)

    async def get_or_create_session(
        self, session_id: Optional[str], user_id: int, **kwargs
    ) -> tuple[Any, bool]:
        """Get or create a session."""
        if session_id and session_id in self.sessions:
            return self.sessions[session_id], False
        session = await self.create_session(user_id)
        return session, True

    async def update_state(self, session_id: str, state: Any) -> bool:
        """Update session state."""
        if session_id in self.sessions:
            self.sessions[session_id].state = state
            return True
        return False

    async def set_pending_intent(self, session_id: str, intent: Any) -> None:
        """Set pending intent."""
        if session_id in self.sessions:
            self.sessions[session_id].pending_intent = intent

    async def end_session(self, session_id: str) -> bool:
        """End a session."""
        if session_id in self.sessions:
            session = self.sessions.pop(session_id)
            if session.user_id in self._user_sessions:
                self._user_sessions[session.user_id].discard(session_id)
            return True
        return False


class DummyCommandRouter:
    """Voice command router stub for testing."""

    def __init__(self) -> None:
        self.processed_commands: List[str] = []

    async def process_command(
        self,
        text: str,
        user_id: int,
        session_id: str,
        db: Any = None,
        persona_id: Optional[str] = None,
    ) -> tuple[Any, str]:
        """Process a command and return a dummy result."""
        from tldw_Server_API.app.core.VoiceAssistant.schemas import ActionResult, ActionType

        self.processed_commands.append(text)

        result = ActionResult(
            success=True,
            action_type=ActionType.LLM_CHAT,
            response_text=f"Processed: {text}",
            result_data={"echo": text},
            execution_time_ms=10.0,
        )
        return result, session_id

    async def get_workflow_status(self, run_id: str, user_id: int) -> Optional[Dict]:
        """Get workflow status."""
        return None

    async def cancel_workflow(self, run_id: str, user_id: int) -> bool:
        """Cancel a workflow."""
        return False

    async def stream_workflow_progress(
        self, run_id: str, user_id: int, poll_interval: float, timeout_seconds: float
    ) -> AsyncIterator[Any]:
        """Stream workflow progress."""
        return
        yield  # Make it a generator


class DummyRegistry:
    """Voice command registry stub."""

    def load_defaults(self) -> None:
        pass

    def get_all_commands(
        self, user_id: int, include_system: bool = True, include_disabled: bool = False
    ) -> List:
        return []


@pytest.fixture
def mock_ws_dependencies(monkeypatch: pytest.MonkeyPatch) -> tuple:
    """Fixture that sets up common mocks for Voice Assistant WebSocket tests."""

    # Mock authentication
    async def _auth(websocket, token: Optional[str]) -> tuple[bool, Optional[int]]:
        if token == "valid-token":
            return True, 1
        return False, None

    # Create instances
    session_manager = DummySessionManager()
    command_router = DummyCommandRouter()
    registry = DummyRegistry()

    # Mock getters
    monkeypatch.setattr(voice_assistant, "_authenticate_websocket", _auth)
    monkeypatch.setattr(
        voice_assistant,
        "get_voice_session_manager",
        lambda: session_manager,
    )
    monkeypatch.setattr(
        voice_assistant,
        "get_voice_command_router",
        lambda: command_router,
    )
    monkeypatch.setattr(
        voice_assistant,
        "get_voice_command_registry",
        lambda: registry,
    )

    # Mock TTS generation
    async def _generate_tts(text, provider=None, model=None, voice=None, response_format="mp3"):
        return b"audio-data", "audio/mpeg"

    monkeypatch.setattr(voice_assistant, "_generate_tts_audio", _generate_tts)

    async def _stream_tts_response(websocket, text, config):
        return None

    monkeypatch.setattr(voice_assistant, "_stream_tts_response", _stream_tts_response)

    # Mock transcription
    async def _transcribe(audio_bytes, config):
        return "transcribed text"

    monkeypatch.setattr(voice_assistant, "_transcribe_audio", _transcribe)

    return session_manager, command_router


class TestWebSocketAuthentication:
    """Tests for WebSocket authentication."""

    @pytest.mark.asyncio
    async def test_auth_sanitizes_db_init_failure_log(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_ws_dependencies,
    ) -> None:
        """Test DB init failures during auth do not leak backend details to logs."""
        logged_warnings = []

        class LoggerStub:
            def warning(self, message, *args, **kwargs):
                logged_warnings.append((message, args, kwargs))

            def info(self, *args, **kwargs):
                return None

            def error(self, *args, **kwargs):
                return None

        async def _fail_get_db_for_user_id(*args, **kwargs):
            raise RuntimeError("voice db exploded /private/voice.db")

        monkeypatch.setattr(voice_assistant, "logger", LoggerStub())
        monkeypatch.setattr(
            voice_assistant,
            "get_chacha_db_for_user_id",
            _fail_get_db_for_user_id,
        )

        messages = [
            {"type": "auth", "token": "valid-token"},
            {"type": "config"},
        ]
        ws = DummyWebSocket(messages)

        await voice_assistant.websocket_voice_assistant(ws, token=None)

        assert logged_warnings == [("Voice assistant DB init failed", (), {})]

    @pytest.mark.asyncio
    async def test_websocket_outer_error_sanitizes_failure_log(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_ws_dependencies,
    ) -> None:
        """Test top-level WebSocket failures do not leak backend details to logs."""
        logged_errors = []

        class LoggerStub:
            def error(self, message, *args, **kwargs):
                logged_errors.append((message, args, kwargs))

            def info(self, *args, **kwargs):
                return None

            def warning(self, *args, **kwargs):
                return None

        def _fail_get_session_manager():
            raise RuntimeError("session manager exploded /private/ws-session.db")

        monkeypatch.setattr(voice_assistant, "logger", LoggerStub())
        monkeypatch.setattr(
            voice_assistant,
            "get_voice_session_manager",
            _fail_get_session_manager,
        )

        messages = [
            {"type": "auth", "token": "valid-token"},
        ]
        ws = DummyWebSocket(messages)

        await voice_assistant.websocket_voice_assistant(ws, token=None)

        assert logged_errors == [("Voice assistant WebSocket error", (), {})]

    @pytest.mark.asyncio
    async def test_auth_success(self, mock_ws_dependencies) -> None:
        """Test successful WebSocket authentication."""
        session_manager, _ = mock_ws_dependencies
        messages = [
            {"type": "auth", "token": "valid-token"},
            {"type": "config"},
        ]
        ws = DummyWebSocket(messages)

        # Run the websocket handler - it will exit on disconnect
        await voice_assistant.websocket_voice_assistant(ws, token=None)

        assert ws.accepted is True
        # Check AUTH_OK was sent
        auth_ok = [m for m in ws.sent_json if m.get("type") == "auth_ok"]
        assert len(auth_ok) == 1
        assert auth_ok[0]["user_id"] == 1
        assert "session_id" in auth_ok[0]

    @pytest.mark.asyncio
    async def test_auth_failure_invalid_token(self, mock_ws_dependencies) -> None:
        """Test WebSocket authentication failure with invalid token."""
        messages = [
            {"type": "auth", "token": "invalid-token"},
        ]
        ws = DummyWebSocket(messages)

        await voice_assistant.websocket_voice_assistant(ws, token=None)

        assert ws.accepted is True
        # Check AUTH_ERROR was sent
        auth_err = [m for m in ws.sent_json if m.get("type") == "auth_error"]
        assert len(auth_err) == 1
        assert "Invalid credentials" in auth_err[0]["error"]
        assert ws.close_code == 4401

    @pytest.mark.asyncio
    async def test_auth_failure_wrong_message_type(self, mock_ws_dependencies) -> None:
        """Test WebSocket authentication failure with wrong message type."""
        messages = [
            {"type": "config"},  # Should be "auth" first
        ]
        ws = DummyWebSocket(messages)

        await voice_assistant.websocket_voice_assistant(ws, token=None)

        assert ws.accepted is True
        auth_err = [m for m in ws.sent_json if m.get("type") == "auth_error"]
        assert len(auth_err) == 1
        assert "Expected AUTH message" in auth_err[0]["error"]
        assert ws.close_code == 4400


class TestWebSocketConfiguration:
    """Tests for WebSocket configuration."""

    @pytest.mark.asyncio
    async def test_config_ack(self, mock_ws_dependencies) -> None:
        """Test configuration acknowledgment."""
        messages = [
            {"type": "auth", "token": "valid-token"},
            {
                "type": "config",
                "stt_model": "parakeet",
                "tts_provider": "kokoro",
                "tts_voice": "af_heart",
            },
        ]
        ws = DummyWebSocket(messages)

        await voice_assistant.websocket_voice_assistant(ws, token=None)

        # Check CONFIG_ACK was sent
        config_ack = [m for m in ws.sent_json if m.get("type") == "config_ack"]
        assert len(config_ack) == 1
        assert config_ack[0]["stt_model"] == "parakeet"
        assert config_ack[0]["tts_provider"] == "kokoro"


class TestWebSocketTextCommands:
    """Tests for text-based voice commands."""

    @pytest.mark.asyncio
    async def test_text_command_processing(self, mock_ws_dependencies) -> None:
        """Test processing a text command via WebSocket."""
        _, command_router = mock_ws_dependencies
        messages = [
            {"type": "auth", "token": "valid-token"},
            {"type": "config"},
            {"type": "text", "text": "search for machine learning"},
        ]
        ws = DummyWebSocket(messages)

        await voice_assistant.websocket_voice_assistant(ws, token=None)

        # Verify command was processed
        assert "search for machine learning" in command_router.processed_commands

        # Check INTENT was sent
        intent_msgs = [m for m in ws.sent_json if m.get("type") == "intent"]
        assert len(intent_msgs) == 1

        # Check ACTION_RESULT was sent
        result_msgs = [m for m in ws.sent_json if m.get("type") == "action_result"]
        assert len(result_msgs) == 1
        assert result_msgs[0]["success"] is True
        assert "Processed:" in result_msgs[0]["response_text"]

        # Check STATE_CHANGE to idle at end
        state_msgs = [m for m in ws.sent_json if m.get("type") == "state_change"]
        assert len(state_msgs) > 0
        # Last state should be idle
        assert state_msgs[-1]["state"] == "idle"


class TestWebSocketAudioCommands:
    """Tests for audio-based voice commands."""

    @pytest.mark.asyncio
    async def test_audio_command_pipeline(self, mock_ws_dependencies) -> None:
        """Test the full audio command pipeline: AUDIO -> COMMIT -> process."""
        _, command_router = mock_ws_dependencies
        audio_data = base64.b64encode(b"fake-audio-data").decode("ascii")
        messages = [
            {"type": "auth", "token": "valid-token"},
            {"type": "config"},
            {"type": "audio", "data": audio_data, "sequence": 0},
            {"type": "audio", "data": audio_data, "sequence": 1},
            {"type": "commit"},
        ]
        ws = DummyWebSocket(messages)

        await voice_assistant.websocket_voice_assistant(ws, token=None)

        # Should have transcription message
        transcription_msgs = [m for m in ws.sent_json if m.get("type") == "transcription"]
        assert len(transcription_msgs) == 1
        assert transcription_msgs[0]["text"] == "transcribed text"
        assert transcription_msgs[0]["is_final"] is True

        # Verify command was processed with transcribed text
        assert "transcribed text" in command_router.processed_commands

    @pytest.mark.asyncio
    async def test_commit_without_audio_returns_error(self, mock_ws_dependencies) -> None:
        """Test that COMMIT without audio returns an error."""
        messages = [
            {"type": "auth", "token": "valid-token"},
            {"type": "config"},
            {"type": "commit"},  # No audio sent
        ]
        ws = DummyWebSocket(messages)

        await voice_assistant.websocket_voice_assistant(ws, token=None)

        # Check ERROR was sent
        error_msgs = [m for m in ws.sent_json if m.get("type") == "error"]
        assert len(error_msgs) == 1
        assert "No audio data" in error_msgs[0]["error"]
        assert error_msgs[0]["recoverable"] is True


class TestWebSocketCancel:
    """Tests for cancel operation."""

    @pytest.mark.asyncio
    async def test_cancel_clears_audio_buffer(self, mock_ws_dependencies) -> None:
        """Test that CANCEL clears the audio buffer."""
        audio_data = base64.b64encode(b"fake-audio-data").decode("ascii")
        messages = [
            {"type": "auth", "token": "valid-token"},
            {"type": "config"},
            {"type": "audio", "data": audio_data},
            {"type": "cancel"},
            {"type": "commit"},  # Should have no audio to process
        ]
        ws = DummyWebSocket(messages)

        await voice_assistant.websocket_voice_assistant(ws, token=None)

        # Cancel should trigger state change to idle
        state_msgs = [m for m in ws.sent_json if m.get("type") == "state_change"]
        idle_states = [m for m in state_msgs if m.get("state") == "idle"]
        assert len(idle_states) > 0

        # Commit after cancel should return error (no audio)
        error_msgs = [m for m in ws.sent_json if m.get("type") == "error"]
        assert len(error_msgs) >= 1


class TestWebSocketStateChanges:
    """Tests for state change notifications."""

    @pytest.mark.asyncio
    async def test_state_change_listening_on_audio(self, mock_ws_dependencies) -> None:
        """Test that state changes to LISTENING when receiving audio."""
        audio_data = base64.b64encode(b"fake-audio-data").decode("ascii")
        messages = [
            {"type": "auth", "token": "valid-token"},
            {"type": "config"},
            {"type": "audio", "data": audio_data},
        ]
        ws = DummyWebSocket(messages)

        await voice_assistant.websocket_voice_assistant(ws, token=None)

        state_msgs = [m for m in ws.sent_json if m.get("type") == "state_change"]
        listening_states = [m for m in state_msgs if m.get("state") == "listening"]
        assert len(listening_states) > 0


class TestWebSocketUnknownMessage:
    """Tests for unknown message handling."""

    @pytest.mark.asyncio
    async def test_unknown_message_type_returns_error(self, mock_ws_dependencies) -> None:
        """Test that unknown message types return an error."""
        messages = [
            {"type": "auth", "token": "valid-token"},
            {"type": "config"},
            {"type": "unknown_type"},
        ]
        ws = DummyWebSocket(messages)

        await voice_assistant.websocket_voice_assistant(ws, token=None)

        error_msgs = [m for m in ws.sent_json if m.get("type") == "error"]
        assert len(error_msgs) == 1
        assert "Unknown message type" in error_msgs[0]["error"]
        assert error_msgs[0]["recoverable"] is True


class TestWebSocketWorkflows:
    """Tests for workflow-related WebSocket messages."""

    @pytest.mark.asyncio
    async def test_workflow_cancel_not_found(self, mock_ws_dependencies) -> None:
        """Test canceling a workflow that doesn't exist."""
        messages = [
            {"type": "auth", "token": "valid-token"},
            {"type": "config"},
            {"type": "workflow_cancel", "run_id": "nonexistent-run"},
        ]
        ws = DummyWebSocket(messages)

        await voice_assistant.websocket_voice_assistant(ws, token=None)

        # Should return error (workflow not found)
        error_msgs = [m for m in ws.sent_json if m.get("type") == "error"]
        assert len(error_msgs) == 1
        assert error_msgs[0]["recoverable"] is True


#
# End of test_ws_integration.py
#######################################################################################################################
