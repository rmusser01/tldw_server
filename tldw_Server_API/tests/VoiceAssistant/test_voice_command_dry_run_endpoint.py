from __future__ import annotations

from types import SimpleNamespace

import pytest

from tldw_Server_API.app.api.v1.endpoints import voice_assistant


pytestmark = pytest.mark.unit


class _RegistryStub:
    def __init__(self) -> None:
        self.load_defaults_called = False
        self.refresh_calls: list[tuple[object, int]] = []

    def load_defaults(self) -> None:
        self.load_defaults_called = True

    def refresh_user_commands(self, db, *, user_id: int, include_disabled: bool = True, persona_id=None):
        self.refresh_calls.append((db, user_id))
        return []


class _ParserStub:
    def __init__(self, *, registry) -> None:
        self.registry = registry

    async def parse(self, *, text: str, user_id: int, context=None):
        assert text == "search for status page"
        assert user_id == 7
        return SimpleNamespace(
            match_method="keyword",
            matched_phrase="search for",
            alternatives=[],
            intent=SimpleNamespace(
                action_type=SimpleNamespace(value="mcp_tool"),
                action_config={"tool_name": "media.search"},
                confidence=0.9,
            ),
        )


@pytest.mark.asyncio
async def test_voice_command_dry_run_loads_defaults_and_refreshes_user_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _RegistryStub()
    monkeypatch.setattr(voice_assistant, "get_voice_command_registry", lambda: registry)

    from tldw_Server_API.app.core.VoiceAssistant import intent_parser as intent_parser_module

    monkeypatch.setattr(intent_parser_module, "IntentParser", _ParserStub)

    db = object()
    response = await voice_assistant.voice_command_dry_run(
        payload=voice_assistant.VoiceCommandDryRunRequest(phrase="search for status page"),
        request=SimpleNamespace(),
        current_user=SimpleNamespace(id=7),
        db=db,
    )

    assert registry.load_defaults_called is True
    assert registry.refresh_calls == [(db, 7)]
    assert response.matched is True
    assert response.action_type == "mcp_tool"


@pytest.mark.asyncio
async def test_generate_tts_audio_sanitizes_backend_failure_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logged_errors = []

    class LoggerStub:
        def error(self, message, *args, **kwargs):
            logged_errors.append((message, args, kwargs))

    async def fake_get_tts_service_v2():
        raise RuntimeError("tts backend exploded /private/tts-cache")

    from tldw_Server_API.app.core.TTS import tts_service_v2

    monkeypatch.setattr(voice_assistant, "logger", LoggerStub())
    monkeypatch.setattr(tts_service_v2, "get_tts_service_v2", fake_get_tts_service_v2)

    audio_bytes, mime_type = await voice_assistant._generate_tts_audio("speak this")

    assert audio_bytes == b""
    assert mime_type == "audio/mpeg"
    assert logged_errors == [("TTS generation failed", (), {})]
