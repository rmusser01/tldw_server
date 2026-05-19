import pytest

from tldw_Server_API.app.core.TTS.adapters import base as base_mod
from tldw_Server_API.app.core.TTS.adapters.base import (
    AudioFormat,
    TTSCapabilities,
    TTSAdapter,
    TTSRequest,
    TTSResponse,
)


class _BadComparable:
    def __lt__(self, _other):
        raise RuntimeError("speed normalization leaked token=secret")

    def __gt__(self, _other):
        raise RuntimeError("speed normalization leaked token=secret")


class _BadLowerStr(str):
    def lower(self):
        raise RuntimeError("lowercase normalization leaked token=secret")


class _FailingInitializeAdapter(TTSAdapter):
    provider_name = "failing-init"

    async def initialize(self):
        raise RuntimeError("init backend exploded token=secret")

    async def generate(self, request: TTSRequest):
        return TTSResponse(audio_data=b"", format=request.format)

    async def get_capabilities(self):
        return TTSCapabilities(
            provider_name=self.provider_name,
            supported_languages={"en"},
            supported_voices=[],
            supported_formats={AudioFormat.MP3},
            max_text_length=1000,
        )


class _FailingCleanupAdapter(_FailingInitializeAdapter):
    provider_name = "failing-cleanup"

    async def initialize(self):
        return True

    async def _cleanup_resources(self):
        raise RuntimeError("cleanup backend exploded token=secret")


def _capture_debug_messages_and_extra(call):
    records: list[str] = []
    sink_id = base_mod.logger.add(
        lambda message: records.append(
            f"{message.record['message']}\n{message.record.get('extra', {})}"
        ),
        level="DEBUG",
    )
    try:
        call()
    finally:
        base_mod.logger.remove(sink_id)
    return "\n".join(records)


def test_request_speed_normalization_log_sanitizes_exception_extra():
    log_output = _capture_debug_messages_and_extra(
        lambda: TTSRequest(text="hello", speed=_BadComparable())
    )

    assert "Voice settings speed normalization failed" in log_output
    assert "speed normalization leaked" not in log_output
    assert "token=secret" not in log_output


def test_request_provider_model_lowercase_log_sanitizes_exception_extra():
    log_output = _capture_debug_messages_and_extra(
        lambda: TTSRequest(text="hello", provider=_BadLowerStr("Provider"))
    )

    assert "TTS provider/model lowercase normalization failed" in log_output
    assert "lowercase normalization leaked" not in log_output
    assert "token=secret" not in log_output


def test_request_voice_settings_coercion_log_sanitizes_exception_extra():
    log_output = _capture_debug_messages_and_extra(
        lambda: TTSRequest(
            text="hello",
            voice_settings={"unexpected token=secret": "value"},
        )
    )

    assert "Voice settings coercion from dict failed" in log_output
    assert "unexpected token=secret" not in log_output


@pytest.mark.asyncio
async def test_ensure_initialized_failure_log_sanitizes_exception_text():
    logged_messages: list[str] = []
    adapter = _FailingInitializeAdapter(config={})
    sink_id = base_mod.logger.add(
        lambda message: logged_messages.append(message.record["message"]),
        level="ERROR",
    )

    try:
        initialized = await adapter.ensure_initialized()
    finally:
        base_mod.logger.remove(sink_id)

    assert initialized is False
    assert any("failing-init initialization failed" in message for message in logged_messages)
    assert all("init backend exploded" not in message for message in logged_messages)
    assert all("token=secret" not in message for message in logged_messages)
    assert all("RuntimeError" in message for message in logged_messages)


@pytest.mark.asyncio
async def test_close_failure_log_sanitizes_exception_text():
    logged_messages: list[str] = []
    adapter = _FailingCleanupAdapter(config={})
    sink_id = base_mod.logger.add(
        lambda message: logged_messages.append(message.record["message"]),
        level="ERROR",
    )

    try:
        await adapter.close()
    finally:
        base_mod.logger.remove(sink_id)

    assert any("Error closing failing-cleanup adapter" in message for message in logged_messages)
    assert all("cleanup backend exploded" not in message for message in logged_messages)
    assert all("token=secret" not in message for message in logged_messages)
    assert all("RuntimeError" in message for message in logged_messages)
