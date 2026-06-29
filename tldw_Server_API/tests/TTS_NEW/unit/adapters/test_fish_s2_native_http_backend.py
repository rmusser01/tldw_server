from __future__ import annotations

import pytest

from tldw_Server_API.app.core.TTS.tts_exceptions import TTSAuthenticationError, TTSProviderError, TTSRateLimitError


class _FakeResponse:
    def __init__(
        self,
        status_code: int = 200,
        content: bytes = b"audio",
        text: str = "ok",
        json_data=None,
        headers=None,
    ):
        self.status_code = status_code
        self.content = content
        self.text = text
        self._json_data = json_data
        self.headers = headers or {}

    def json(self):
        return self._json_data


@pytest.mark.asyncio
async def test_backend_builds_tts_payload_from_request(monkeypatch):
    from tldw_Server_API.app.core.TTS.backends.fish_s2_native_http import FishS2NativeHttpBackend

    backend = FishS2NativeHttpBackend(
        {"base_url": "http://fish.local", "api_key": "secret", "timeout": 30}
    )

    captured: dict[str, object] = {}

    async def fake_fetch(*, method, url, json=None, headers=None, timeout=None, **_kwargs):
        captured["method"] = method
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout
        return _FakeResponse()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.backends.fish_s2_native_http.afetch",
        fake_fetch,
    )

    audio = await backend.synthesize(
        text="hello",
        response_format="wav",
        streaming=False,
        reference_id="tldw_u1_vabc",
        extra_params={"chunk_length": 200, "normalize": True},
    )

    assert audio == b"audio"
    assert captured["method"] == "POST"
    assert captured["url"] == "http://fish.local/v1/tts"
    assert captured["json"] == {
        "text": "hello",
        "format": "wav",
        "streaming": False,
        "reference_id": "tldw_u1_vabc",
        "chunk_length": 200,
        "normalize": True,
    }
    assert captured["headers"] == {"Authorization": "Bearer secret"}
    assert captured["timeout"] == 30


@pytest.mark.asyncio
async def test_backend_uses_streaming_helper_for_streaming_requests(monkeypatch):
    from tldw_Server_API.app.core.TTS.backends.fish_s2_native_http import FishS2NativeHttpBackend

    backend = FishS2NativeHttpBackend({"base_url": "http://fish.local", "api_key": "secret", "timeout": 12})
    captured: dict[str, object] = {}

    async def fake_stream(*, method, url, json=None, headers=None, timeout=None, **_kwargs):
        captured["method"] = method
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout
        yield b"chunk1"
        yield b"chunk2"

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.backends.fish_s2_native_http.astream_bytes",
        fake_stream,
    )

    stream = await backend.synthesize(
        text="hello",
        response_format="wav",
        streaming=True,
        reference_id=None,
        extra_params=None,
    )

    chunks = [chunk async for chunk in stream]
    assert chunks == [b"chunk1", b"chunk2"]
    assert captured["method"] == "POST"
    assert captured["url"] == "http://fish.local/v1/tts"
    assert captured["json"] == {
        "text": "hello",
        "format": "wav",
        "streaming": True,
    }
    assert captured["headers"] == {"Authorization": "Bearer secret"}
    assert captured["timeout"] == 12


@pytest.mark.asyncio
async def test_backend_maps_401_to_authentication_error(monkeypatch):
    from tldw_Server_API.app.core.TTS.backends.fish_s2_native_http import FishS2NativeHttpBackend

    backend = FishS2NativeHttpBackend({"base_url": "http://fish.local", "api_key": "secret"})

    async def fake_fetch(**_kwargs):
        return _FakeResponse(status_code=401, text="invalid token")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.backends.fish_s2_native_http.afetch",
        fake_fetch,
    )

    with pytest.raises(TTSAuthenticationError):
        await backend.synthesize(
            text="hello",
            response_format="wav",
            streaming=False,
            reference_id=None,
            extra_params=None,
        )


@pytest.mark.asyncio
async def test_backend_maps_non_integer_retry_after_to_rate_limit_error(monkeypatch):
    from tldw_Server_API.app.core.TTS.backends.fish_s2_native_http import FishS2NativeHttpBackend

    backend = FishS2NativeHttpBackend({"base_url": "http://fish.local", "api_key": "secret"})

    async def fake_fetch(**_kwargs):
        return _FakeResponse(
            status_code=429,
            text="rate limited",
            headers={"retry-after": "Wed, 21 Oct 2015 07:28:00 GMT"},
        )

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.backends.fish_s2_native_http.afetch",
        fake_fetch,
    )

    with pytest.raises(TTSRateLimitError) as exc_info:
        await backend.synthesize(
            text="hello",
            response_format="wav",
            streaming=False,
            reference_id=None,
            extra_params=None,
        )

    assert exc_info.value.retry_after is None


def test_backend_response_json_logs_parse_failures(monkeypatch):
    from tldw_Server_API.app.core.TTS.backends import fish_s2_native_http
    from tldw_Server_API.app.core.TTS.backends.fish_s2_native_http import FishS2NativeHttpBackend

    class _LoggerStub:
        def __init__(self):
            self.debug_calls = []

        def debug(self, *args, **kwargs):
            self.debug_calls.append((args, kwargs))

    class _InvalidJsonResponse:
        def json(self):
            raise ValueError("not json")

    logger_stub = _LoggerStub()
    monkeypatch.setattr(fish_s2_native_http, "logger", logger_stub)

    assert FishS2NativeHttpBackend._response_json(_InvalidJsonResponse()) is None
    assert logger_stub.debug_calls


@pytest.mark.asyncio
async def test_backend_error_log_omits_upstream_body(monkeypatch):
    from tldw_Server_API.app.core.TTS.backends import fish_s2_native_http
    from tldw_Server_API.app.core.TTS.backends.fish_s2_native_http import FishS2NativeHttpBackend

    class _LoggerStub:
        def __init__(self):
            self.error_calls = []

        def error(self, *args, **kwargs):
            self.error_calls.append((args, kwargs))

    logger_stub = _LoggerStub()
    monkeypatch.setattr(fish_s2_native_http, "logger", logger_stub)

    async def fake_fetch(**_kwargs):
        return _FakeResponse(status_code=500, text="token leaked /private/fish-body.txt")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.backends.fish_s2_native_http.afetch",
        fake_fetch,
    )

    backend = FishS2NativeHttpBackend({"base_url": "http://fish.local"})

    with pytest.raises(TTSProviderError):
        await backend.synthesize(
            text="hello",
            response_format="wav",
            streaming=False,
            reference_id=None,
            extra_params=None,
        )

    assert logger_stub.error_calls
    rendered_calls = repr(logger_stub.error_calls)
    assert "token leaked" not in rendered_calls
    assert "/private/fish-body.txt" not in rendered_calls


@pytest.mark.asyncio
async def test_backend_add_reference_posts_expected_payload(monkeypatch):
    from tldw_Server_API.app.core.TTS.backends.fish_s2_native_http import FishS2NativeHttpBackend

    backend = FishS2NativeHttpBackend({"base_url": "http://fish.local", "api_key": "secret"})

    captured = {}

    async def fake_fetch(*, method, url, json=None, data=None, files=None, headers=None, timeout=None, **_kwargs):
        captured["method"] = method
        captured["url"] = url
        captured["json"] = json
        captured["data"] = data
        captured["files"] = files
        captured["headers"] = headers
        captured["timeout"] = timeout
        return _FakeResponse(json_data={"reference_id": "tldw_u1_voice-1"})

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.backends.fish_s2_native_http.afetch",
        fake_fetch,
    )

    result = await backend.add_reference(
        reference_id="tldw_u1_voice-1",
        audio_b64="QUJD",
        reference_text="hello there",
    )

    assert result == {"reference_id": "tldw_u1_voice-1"}
    assert captured["method"] == "POST"
    assert captured["url"] == "http://fish.local/v1/references/add"
    assert captured["json"] is None
    assert captured["data"] == {
        "id": "tldw_u1_voice-1",
        "text": "hello there",
    }
    assert captured["files"] == {
        "audio": ("reference.wav", b"ABC", "audio/wav"),
    }
    assert captured["headers"] == {"Authorization": "Bearer secret"}
    assert captured["timeout"] == backend.timeout


@pytest.mark.asyncio
async def test_backend_delete_reference_posts_expected_payload(monkeypatch):
    from tldw_Server_API.app.core.TTS.backends.fish_s2_native_http import FishS2NativeHttpBackend

    backend = FishS2NativeHttpBackend({"base_url": "http://fish.local", "api_key": "secret", "timeout": 12})

    captured = {}

    async def fake_fetch(*, method, url, json=None, headers=None, timeout=None, **_kwargs):
        captured["method"] = method
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout
        return _FakeResponse(status_code=204)

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.backends.fish_s2_native_http.afetch",
        fake_fetch,
    )

    deleted = await backend.delete_reference(reference_id="tldw_u1_voice-1")

    assert deleted is True
    assert captured["method"] == "DELETE"
    assert captured["url"] == "http://fish.local/v1/references/delete"
    assert captured["json"] == {"reference_id": "tldw_u1_voice-1"}
    assert captured["headers"] == {"Authorization": "Bearer secret"}
    assert captured["timeout"] == 12
