import importlib.util
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import tldw_Server_API.app.api.v1.endpoints.audio.audio_health as audio_health
import tldw_Server_API.app.core.TTS.adapter_registry as adapter_registry


class _FakeRegistry:
    def __init__(self, kokoro_adapter):
        self._kokoro_adapter = kokoro_adapter

    def list_capabilities(self, include_disabled=True):
        assert include_disabled is True
        return [
            {
                "provider": "kokoro",
                "availability": "enabled",
                "capabilities": None,
            },
            {
                "provider": "openai",
                "availability": "enabled",
                "capabilities": None,
            },
        ]

    async def get_adapter(self, provider):
        if str(getattr(provider, "value", provider)) == "kokoro":
            return self._kokoro_adapter
        return None


class _FakeFactory:
    def __init__(self, kokoro_adapter):
        self.registry = _FakeRegistry(kokoro_adapter)


class _FakeTTSService:
    def get_status(self):
        return {
            "providers": {
                "kokoro": {
                    "status": "available",
                    "availability": "enabled",
                    "initialized": True,
                    "failed": False,
                },
                "openai": {
                    "status": "available",
                    "availability": "enabled",
                    "initialized": True,
                    "failed": False,
                },
            },
            "available": 2,
            "total_providers": 2,
            "circuit_breakers": {},
        }

    async def get_capabilities(self):
        return {}


class _FailingTTSHealthService:
    def get_status(self):
        raise RuntimeError("TTS health backend leaked /private/tts-health.json")


class _FakeConfigManager:
    def __init__(self, provider_configs):
        self._provider_configs = dict(provider_configs)

    def get_provider_config(self, provider):
        return self._provider_configs.get(provider)


class _FakeCircuitManager:
    def __init__(self, statuses):
        self._statuses = dict(statuses)

    def get_all_status(self, detailed=False):
        assert detailed is True
        return dict(self._statuses)


@pytest.mark.asyncio
async def test_tts_health_error_log_is_sanitized(monkeypatch):
    fake_logger = MagicMock()
    monkeypatch.setattr(audio_health, "logger", fake_logger)
    monkeypatch.setattr(audio_health, "ensure_request_id", lambda _request: "req-test")

    health = await audio_health.get_tts_health(
        request=MagicMock(),
        tts_service=_FailingTTSHealthService(),
    )

    assert health["status"] == "error"
    assert health["message"] == "TTS health check failed"
    assert health["request_id"] == "req-test"
    fake_logger.error.assert_called_once_with("Error getting TTS health")


@pytest.mark.asyncio
async def test_kokoro_health_marks_pytorch_runtime_missing_pipeline_as_unhealthy(monkeypatch):
    kokoro_adapter = SimpleNamespace(
        use_onnx=False,
        device="cpu",
        model_path="models/kokoro/kokoro-v1_0.pth",
        voices_json="models/kokoro/voices",
    )

    async def _fake_get_tts_factory():
        return _FakeFactory(kokoro_adapter)

    def _fake_find_spec(name):
        if name == "kokoro.pipeline":
            return None
        return object()

    monkeypatch.setattr(adapter_registry, "get_tts_factory", _fake_get_tts_factory)
    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec)

    health = await audio_health.get_tts_health(
        request=MagicMock(),
        tts_service=_FakeTTSService(),
    )

    kokoro_detail = health["providers"]["details"]["kokoro"]
    assert kokoro_detail["availability"] == "unhealthy"
    assert kokoro_detail["status"] == "unhealthy"
    assert kokoro_detail["runtime_ready"] is False
    assert kokoro_detail["runtime_reason"] == "kokoro_pipeline_missing"
    assert health["providers"]["available"] == 1
    assert health["status"] == "healthy"


@pytest.mark.asyncio
async def test_openai_health_marks_missing_api_key_as_unhealthy(monkeypatch):
    async def _fake_get_tts_factory():
        return _FakeFactory(kokoro_adapter=None)

    monkeypatch.setattr(adapter_registry, "get_tts_factory", _fake_get_tts_factory)
    monkeypatch.setattr(
        audio_health,
        "get_tts_config_manager",
        lambda: _FakeConfigManager(
            {
                "openai": SimpleNamespace(enabled=True, api_key=None),
                "kokoro": SimpleNamespace(enabled=True, api_key=None),
            }
        ),
    )

    health = await audio_health.get_tts_health(
        request=MagicMock(),
        tts_service=_FakeTTSService(),
    )

    openai_detail = health["providers"]["details"]["openai"]
    assert openai_detail["availability"] == "unhealthy"
    assert openai_detail["status"] == "unhealthy"
    assert openai_detail["auth_configured"] is False
    assert openai_detail["auth_ready"] is False
    assert openai_detail["auth_reason"] == "api_key_missing"
    assert health["providers"]["available"] == 1
    assert health["status"] == "healthy"


@pytest.mark.asyncio
async def test_openai_health_marks_known_auth_failure_as_unhealthy(monkeypatch):
    async def _fake_get_tts_factory():
        return _FakeFactory(kokoro_adapter=None)

    monkeypatch.setattr(adapter_registry, "get_tts_factory", _fake_get_tts_factory)
    monkeypatch.setattr(
        audio_health,
        "get_tts_config_manager",
        lambda: _FakeConfigManager(
            {
                "openai": SimpleNamespace(enabled=True, api_key="sk-live-but-bad"),
                "kokoro": SimpleNamespace(enabled=True, api_key=None),
            }
        ),
    )

    service = _FakeTTSService()
    service.circuit_manager = _FakeCircuitManager(
        {
            "openai": {
                "provider": "openai",
                "state": "closed",
                "error_analysis": {
                    "error_categories": {
                        "authentication": 1,
                    }
                },
            }
        }
    )

    health = await audio_health.get_tts_health(
        request=MagicMock(),
        tts_service=service,
    )

    openai_detail = health["providers"]["details"]["openai"]
    assert openai_detail["availability"] == "unhealthy"
    assert openai_detail["status"] == "unhealthy"
    assert openai_detail["auth_configured"] is True
    assert openai_detail["auth_ready"] is False
    assert openai_detail["auth_reason"] == "authentication_failed"
    assert health["providers"]["available"] == 1
    assert health["status"] == "healthy"


@pytest.mark.asyncio
async def test_kokoro_health_redacts_absolute_path_details(monkeypatch):
    kokoro_adapter = SimpleNamespace(
        use_onnx=True,
        device="cpu",
        model_path="/Users/private/models/kokoro/model.onnx",
        voices_json="/Users/private/models/kokoro/voices.json",
    )

    async def _fake_get_tts_factory():
        return _FakeFactory(kokoro_adapter)

    def _fake_find_spec(_name):
        return object()

    def _fake_exists(path):
        return str(path).endswith("libespeak-ng.so.1")

    monkeypatch.setattr(adapter_registry, "get_tts_factory", _fake_get_tts_factory)
    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec)
    monkeypatch.setattr(
        audio_health,
        "_discover_kokoro_espeak_library",
        lambda _adapter: "/Users/private/lib/libespeak-ng.so.1",
    )
    monkeypatch.setattr(audio_health.os.path, "exists", _fake_exists)
    monkeypatch.setenv(
        "PHONEMIZER_ESPEAK_LIBRARY",
        "/Users/private/lib/libespeak-ng.so.1",
    )

    health = await audio_health.get_tts_health(
        request=MagicMock(),
        tts_service=_FakeTTSService(),
    )

    kokoro_info = health["providers"]["kokoro"]

    assert kokoro_info["model_path"] == "model.onnx"
    assert kokoro_info["voices_json"] == "voices.json"
    assert kokoro_info["espeak_lib_env"] == "libespeak-ng.so.1"
    assert kokoro_info["espeak_lib_path"] == "libespeak-ng.so.1"
    assert "/Users/private" not in str(kokoro_info)
