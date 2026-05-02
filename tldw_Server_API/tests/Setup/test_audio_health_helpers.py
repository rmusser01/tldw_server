from fastapi import HTTPException

import pytest
import time
from unittest.mock import MagicMock

from tldw_Server_API.app.api.v1.endpoints.audio import audio_health


def test_public_tts_health_helpers_document_sanitization_contracts():
    assert audio_health._sanitize_public_provider_detail.__doc__
    assert audio_health._normalize_public_health_key.__doc__
    assert audio_health._derive_omnivoice_supervisor_health.__doc__


@pytest.mark.asyncio
async def test_collect_setup_stt_health_normalizes_http_exception(mocker):
    mocker.patch.object(
        audio_health,
        "get_stt_health",
        side_effect=HTTPException(
            status_code=400,
            detail={"message": "Invalid transcription model identifier"},
        ),
    )

    result = await audio_health.collect_setup_stt_health(model="bad-model")

    assert result["usable"] is False
    assert result["available"] is False
    assert result["status_code"] == 400
    assert result["message"] == "Invalid transcription model identifier"
    assert result["model"] == "bad-model"


@pytest.mark.asyncio
async def test_collect_setup_tts_health_normalizes_service_bootstrap_failure(mocker):
    mocker.patch.object(
        audio_health,
        "get_tts_service",
        side_effect=RuntimeError("adapter bootstrap exploded"),
    )

    result = await audio_health.collect_setup_tts_health()

    assert result["status"] == "error"
    assert result["providers"] == {"total": 0, "available": 0, "details": {}}
    assert result["message"] == "TTS health check failed"
    assert result["status_code"] == 500


@pytest.mark.asyncio
async def test_get_stt_health_sanitizes_suspicious_runtime_strings(mocker):
    mocker.patch(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib.parse_transcription_model",
        return_value=("whisper", None, None),
    )
    mocker.patch(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib.validate_whisper_model_identifier",
        side_effect=lambda value: value,
    )
    mocker.patch(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Files.check_transcription_model_status",
        return_value={
            "available": False,
            "usable": False,
            "message": "Traceback: /Users/private/model.bin\nRuntimeError: boom",
            "details": "/Users/private/model.bin",
            "model": "whisper-1",
        },
    )

    result = await audio_health.get_stt_health(
        audio_health._build_internal_health_request("/api/v1/audio/transcriptions/health"),
        model="whisper-1",
        warm=False,
    )

    assert result["message"] == "Internal health diagnostics were suppressed."
    assert "details" not in result


@pytest.mark.asyncio
async def test_get_stt_health_warmup_failure_log_is_sanitized(mocker, monkeypatch):
    mocker.patch(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib.parse_transcription_model",
        return_value=("whisper", None, None),
    )
    mocker.patch(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib.validate_whisper_model_identifier",
        side_effect=lambda value: value,
    )
    mocker.patch(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Files.check_transcription_model_status",
        return_value={
            "available": True,
            "usable": True,
            "message": "ready",
            "model": "whisper-1",
        },
    )
    mocker.patch(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib.processing_choice",
        "cpu",
    )
    mocker.patch(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib.get_whisper_model",
        side_effect=RuntimeError("warm-up leak /private/whisper-model.bin"),
    )

    fake_logger = MagicMock()
    monkeypatch.setattr(audio_health, "logger", fake_logger)

    result = await audio_health.get_stt_health(
        audio_health._build_internal_health_request("/api/v1/audio/transcriptions/health"),
        model="whisper-1",
        warm=True,
    )

    assert result["warm"] == {
        "ok": False,
        "device": "cpu",
        "error": "Model initialization failed.",
    }
    fake_logger.exception.assert_called_once_with("STT health warm-up failed")


@pytest.mark.asyncio
async def test_get_tts_health_surfaces_sanitized_omnivoice_sidecar_status(monkeypatch):
    class _FakeRegistry:
        def list_capabilities(self, include_disabled=True):
            assert include_disabled is True
            return [
                {
                    "provider": "omnivoice",
                    "availability": "enabled",
                    "capabilities": {
                        "provider_name": "OmniVoice",
                        "metadata": {"runtime": "sidecar"},
                    },
                }
            ]

        async def get_adapter(self, _provider):
            return None

    class _FakeFactory:
        registry = _FakeRegistry()

    class _FakeTTSService:
        def __init__(self):
            self._omnivoice_supervisor = MagicMock(
                _closing=False,
                _process=MagicMock(returncode=None),
                _base_url="http://127.0.0.1:8039",
                _last_activity_at=time.time(),
                last_failure_at=time.time(),
                _startup_backoff_seconds=5.0,
            )

        def get_status(self):
            return {
                "providers": {
                    "omnivoice": {
                        "status": "available",
                        "availability": "enabled",
                        "initialized": True,
                        "failed": False,
                        "token": "secret-token",
                        "command": ["python", "/Users/private/omnivoice_sidecar.py"],
                        "traceback": "Traceback: /Users/private/omnivoice_sidecar.py",
                        "repo_path": "/Users/private/OmniVoice",
                        "authToken": "secret-auth-token",
                        "apiKey": "secret-api-key",
                        "baseURL": "http://127.0.0.1:8039",
                        "repoPath": "/Users/private/OmniVoiceCamel",
                        "stackTrace": "Traceback: /Users/private/camel.py",
                    }
                },
                "available": 1,
                "total_providers": 1,
                "circuit_breakers": {},
            }

        async def get_capabilities(self):
            return {}

    async def _fake_get_tts_factory():
        return _FakeFactory()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.adapter_registry.get_tts_factory",
        _fake_get_tts_factory,
        raising=True,
    )

    health = await audio_health.get_tts_health(
        request=MagicMock(),
        tts_service=_FakeTTSService(),
    )

    omnivoice_detail = health["providers"]["details"]["omnivoice"]
    assert omnivoice_detail["availability"] == "enabled"
    assert omnivoice_detail["runtime"] == "sidecar"
    assert omnivoice_detail["sidecar_state"] == "ready"
    assert "last_error_code" not in omnivoice_detail
    assert "token" not in omnivoice_detail
    assert "command" not in omnivoice_detail
    assert "traceback" not in omnivoice_detail
    assert "repo_path" not in omnivoice_detail
    assert "authToken" not in omnivoice_detail
    assert "apiKey" not in omnivoice_detail
    assert "baseURL" not in omnivoice_detail
    assert "repoPath" not in omnivoice_detail
    assert "stackTrace" not in omnivoice_detail

    envelope = health["capabilities_envelope"][0]
    assert envelope["provider"] == "omnivoice"
    assert envelope["runtime"] == "sidecar"


@pytest.mark.asyncio
async def test_get_tts_health_top_level_failure_log_is_sanitized(monkeypatch):
    class _FailingTTSService:
        def get_status(self):
            raise RuntimeError("status leak /private/tts-status.json")

    fake_logger = MagicMock()
    monkeypatch.setattr(audio_health, "logger", fake_logger)

    health = await audio_health.get_tts_health(
        request=audio_health._build_internal_health_request("/api/v1/audio/health"),
        tts_service=_FailingTTSService(),
    )

    assert health["status"] == "error"
    assert health["message"] == "TTS health check failed"
    fake_logger.error.assert_called_once_with("Error getting TTS health", exc_info=True)


@pytest.mark.asyncio
async def test_get_tts_health_envelope_enrichment_failure_log_is_sanitized(monkeypatch):
    class _FakeRegistry:
        def list_capabilities(self, include_disabled=True):
            assert include_disabled is True
            raise RuntimeError("registry leak /private/tts-registry.json")

        async def get_adapter(self, _provider):
            return None

    class _FakeFactory:
        registry = _FakeRegistry()

    class _FakeTTSService:
        def get_status(self):
            return {"providers": {}, "available": 0, "total_providers": 0, "circuit_breakers": {}}

        async def get_capabilities(self):
            return {}

    async def _fake_get_tts_factory():
        return _FakeFactory()

    fake_logger = MagicMock()
    monkeypatch.setattr(audio_health, "logger", fake_logger)
    monkeypatch.setattr(audio_health, "_enrich_external_provider_auth_health", lambda *args: None)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.adapter_registry.get_tts_factory",
        _fake_get_tts_factory,
        raising=True,
    )

    health = await audio_health.get_tts_health(
        request=MagicMock(),
        tts_service=_FakeTTSService(),
    )

    assert health["status"] == "unhealthy"
    fake_logger.debug.assert_called_once_with("TTS health envelope enrichment failed")


@pytest.mark.asyncio
async def test_get_tts_health_kokoro_espeak_introspection_failure_log_is_sanitized(monkeypatch):
    class _FakeAdapter:
        use_onnx = False
        device = "cpu"

    class _FakeRegistry:
        def list_capabilities(self, include_disabled=True):
            assert include_disabled is True
            return []

        async def get_adapter(self, _provider):
            return _FakeAdapter()

    class _FakeFactory:
        registry = _FakeRegistry()

    class _FakeTTSService:
        def get_status(self):
            return {
                "providers": {"kokoro": {"status": "available"}},
                "available": 1,
                "total_providers": 1,
                "circuit_breakers": {},
            }

        async def get_capabilities(self):
            return {}

    async def _fake_get_tts_factory():
        return _FakeFactory()

    def _fail_getenv(name, default=None):
        if name == "PHONEMIZER_ESPEAK_LIBRARY":
            raise RuntimeError("env lookup leaked /private/espeak.env")
        return default

    fake_logger = MagicMock()
    monkeypatch.setattr(audio_health, "logger", fake_logger)
    monkeypatch.setattr(audio_health.os, "getenv", _fail_getenv)
    monkeypatch.setattr(audio_health, "_module_spec_available", lambda _name: True)
    monkeypatch.setattr(audio_health, "_enrich_external_provider_auth_health", lambda *args: None)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.adapter_registry.get_tts_factory",
        _fake_get_tts_factory,
        raising=True,
    )

    health = await audio_health.get_tts_health(
        request=MagicMock(),
        tts_service=_FakeTTSService(),
    )

    assert health["providers"]["kokoro"]["device"] == "cpu"
    fake_logger.debug.assert_called_once_with("Kokoro health eSpeak library introspection failed")


@pytest.mark.asyncio
async def test_get_tts_health_kokoro_enrichment_failure_log_is_sanitized(monkeypatch):
    class _FakeRegistry:
        def list_capabilities(self, include_disabled=True):
            assert include_disabled is True
            return []

        async def get_adapter(self, _provider):
            raise RuntimeError("kokoro adapter leak /private/kokoro-adapter.json")

    class _FakeFactory:
        registry = _FakeRegistry()

    class _FakeTTSService:
        def get_status(self):
            return {"providers": {}, "available": 0, "total_providers": 0, "circuit_breakers": {}}

        async def get_capabilities(self):
            return {}

    async def _fake_get_tts_factory():
        return _FakeFactory()

    fake_logger = MagicMock()
    monkeypatch.setattr(audio_health, "logger", fake_logger)
    monkeypatch.setattr(audio_health, "_enrich_external_provider_auth_health", lambda *args: None)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.adapter_registry.get_tts_factory",
        _fake_get_tts_factory,
        raising=True,
    )

    health = await audio_health.get_tts_health(
        request=MagicMock(),
        tts_service=_FakeTTSService(),
    )

    assert health["status"] == "unhealthy"
    fake_logger.debug.assert_called_once_with("Kokoro health enrichment failed")


def test_tts_health_capability_serializer_failure_log_is_sanitized(monkeypatch):
    class _FailingTTSService:
        def _serialize_capabilities(self, _caps):
            raise RuntimeError("capability serializer exploded at /private/tts-caps.json")

    fake_logger = MagicMock()
    monkeypatch.setattr(audio_health, "logger", fake_logger)

    result = audio_health._serialize_tts_caps_for_health(
        _FailingTTSService(),
        {"provider": "kokoro", "voices": ["af_heart"]},
    )

    assert result == {"provider": "kokoro", "voices": ["af_heart"]}
    fake_logger.debug.assert_called_once()
    message = fake_logger.debug.call_args.args[0]
    assert "exploded" not in message
    assert "/private/" not in message
    assert not fake_logger.debug.call_args.kwargs


def test_tts_health_capability_model_dump_failure_log_is_sanitized(monkeypatch):
    fake_logger = MagicMock()
    monkeypatch.setattr(audio_health, "logger", fake_logger)
    monkeypatch.setattr(audio_health, "is_dataclass", lambda _caps: False)

    def _fail_model_dump(_caps):
        raise RuntimeError("capability model dump exploded at /private/tts-caps.json")

    monkeypatch.setattr(audio_health, "model_dump_compat", _fail_model_dump)

    result = audio_health._serialize_tts_caps_for_health(
        object(),
        object(),
    )

    assert result is None
    fake_logger.debug.assert_called_once()
    message = fake_logger.debug.call_args.args[0]
    assert "exploded" not in message
    assert "/private/" not in message
    assert not fake_logger.debug.call_args.kwargs


def test_tts_health_capability_dataclass_failure_log_is_sanitized(monkeypatch):
    fake_logger = MagicMock()
    monkeypatch.setattr(audio_health, "logger", fake_logger)
    monkeypatch.setattr(audio_health, "model_dump_compat", lambda _caps: None)
    monkeypatch.setattr(audio_health, "is_dataclass", lambda _caps: True)

    def _fail_asdict(_caps):
        raise RuntimeError("capability dataclass exploded at /private/tts-caps.json")

    monkeypatch.setattr(audio_health, "asdict", _fail_asdict)

    result = audio_health._serialize_tts_caps_for_health(
        object(),
        object(),
    )

    assert result is None
    fake_logger.debug.assert_called_once()
    message = fake_logger.debug.call_args.args[0]
    assert "exploded" not in message
    assert "/private/" not in message
    assert not fake_logger.debug.call_args.kwargs


def test_tts_health_auth_config_lookup_failure_log_is_sanitized(monkeypatch):
    def _fail_config_manager():
        raise RuntimeError("auth config leaked /private/tts-auth.toml")

    fake_logger = MagicMock()
    monkeypatch.setattr(audio_health, "logger", fake_logger)
    monkeypatch.setattr(audio_health, "get_tts_config_manager", _fail_config_manager)

    loaded, configs = audio_health._load_auth_provider_configs()

    assert loaded is False
    assert configs == {}
    fake_logger.debug.assert_called_once_with("TTS health auth config lookup failed")


def test_tts_health_detailed_circuit_breaker_failure_log_is_sanitized(monkeypatch):
    class _FailingCircuitManager:
        def get_all_status(self, detailed: bool = False):
            assert detailed is True
            raise RuntimeError("breaker details leaked /private/tts-breakers.json")

    class _Service:
        circuit_manager = _FailingCircuitManager()

    fake_logger = MagicMock()
    monkeypatch.setattr(audio_health, "logger", fake_logger)

    result = audio_health._load_detailed_circuit_breakers(_Service())

    assert result == {}
    fake_logger.debug.assert_called_once_with("TTS health detailed circuit-breaker lookup failed")


def test_tts_health_espeak_ctypes_discovery_failure_log_is_sanitized(monkeypatch):
    class _Adapter:
        config = {}

    def _fail_find_library(_name):
        raise RuntimeError("ctypes lookup leaked /private/libespeak-ng.dylib")

    fake_logger = MagicMock()
    monkeypatch.setattr(audio_health, "logger", fake_logger)
    monkeypatch.setattr(audio_health, "_ctypes_find_library", _fail_find_library)
    monkeypatch.setattr(audio_health.os.path, "exists", lambda _path: False)
    monkeypatch.delenv("PHONEMIZER_ESPEAK_LIBRARY", raising=False)

    result = audio_health._discover_kokoro_espeak_library(_Adapter())

    assert result is None
    fake_logger.debug.assert_called_once_with("Unable to discover eSpeak library via ctypes lookup")


@pytest.mark.asyncio
async def test_get_tts_health_derives_omnivoice_backoff_state_from_supervisor(monkeypatch):
    class _FakeRegistry:
        def list_capabilities(self, include_disabled=True):
            assert include_disabled is True
            return [
                {
                    "provider": "omnivoice",
                    "availability": "enabled",
                    "capabilities": {
                        "provider_name": "OmniVoice",
                        "metadata": {"runtime": "sidecar"},
                    },
                }
            ]

        async def get_adapter(self, _provider):
            return None

    class _FakeFactory:
        registry = _FakeRegistry()

    class _FakeTTSService:
        def __init__(self):
            self._omnivoice_supervisor = MagicMock(
                _closing=False,
                _process=None,
                _base_url=None,
                last_failure_at=time.time(),
                _startup_backoff_seconds=30.0,
            )

        def get_status(self):
            return {
                "providers": {
                    "omnivoice": {
                        "status": "available",
                        "availability": "enabled",
                        "initialized": True,
                        "failed": False,
                    }
                },
                "available": 1,
                "total_providers": 1,
                "circuit_breakers": {},
            }

        async def get_capabilities(self):
            return {}

    async def _fake_get_tts_factory():
        return _FakeFactory()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.adapter_registry.get_tts_factory",
        _fake_get_tts_factory,
        raising=True,
    )

    health = await audio_health.get_tts_health(
        request=MagicMock(),
        tts_service=_FakeTTSService(),
    )

    omnivoice_detail = health["providers"]["details"]["omnivoice"]
    assert omnivoice_detail["runtime"] == "sidecar"
    assert omnivoice_detail["sidecar_state"] == "degraded"
    assert omnivoice_detail["last_error_code"] == "startup_backoff"
    assert omnivoice_detail["status"] == "degraded"
    assert omnivoice_detail["availability"] == "degraded"
    assert omnivoice_detail["failed"] is True
    assert health["providers"]["available"] == 0
    assert health["status"] == "unhealthy"
    assert health["capabilities_envelope"][0]["availability"] == "degraded"


@pytest.mark.asyncio
async def test_get_tts_health_marks_enabled_omnivoice_without_supervisor_idle_stopped(monkeypatch):
    class _FakeRegistry:
        def list_capabilities(self, include_disabled=True):
            assert include_disabled is True
            return [
                {
                    "provider": "omnivoice",
                    "availability": "enabled",
                    "capabilities": {
                        "provider_name": "OmniVoice",
                    },
                }
            ]

        async def get_adapter(self, _provider):
            return None

    class _FakeFactory:
        registry = _FakeRegistry()

    class _FakeTTSService:
        def get_status(self):
            return {
                "providers": {
                    "omnivoice": {
                        "status": "enabled",
                        "availability": "enabled",
                        "initialized": False,
                        "failed": False,
                    }
                },
                "available": 1,
                "total_providers": 1,
                "circuit_breakers": {},
            }

        async def get_capabilities(self):
            return {}

    async def _fake_get_tts_factory():
        return _FakeFactory()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.adapter_registry.get_tts_factory",
        _fake_get_tts_factory,
        raising=True,
    )

    health = await audio_health.get_tts_health(
        request=MagicMock(),
        tts_service=_FakeTTSService(),
    )

    omnivoice_detail = health["providers"]["details"]["omnivoice"]
    assert omnivoice_detail["runtime"] == "sidecar"
    assert omnivoice_detail["sidecar_state"] == "idle_stopped"


@pytest.mark.asyncio
async def test_get_tts_health_keeps_live_omnivoice_startup_as_starting(monkeypatch):
    class _FakeRegistry:
        def list_capabilities(self, include_disabled=True):
            assert include_disabled is True
            return [
                {
                    "provider": "omnivoice",
                    "availability": "enabled",
                    "capabilities": {
                        "provider_name": "OmniVoice",
                    },
                }
            ]

        async def get_adapter(self, _provider):
            return None

    class _FakeFactory:
        registry = _FakeRegistry()

    class _FakeTTSService:
        def __init__(self):
            self._omnivoice_supervisor = MagicMock(
                _closing=False,
                _process=MagicMock(returncode=None),
                _base_url="http://127.0.0.1:8039",
                _last_activity_at=None,
                last_failure_at=None,
                _startup_backoff_seconds=30.0,
            )

        def get_status(self):
            return {
                "providers": {
                    "omnivoice": {
                        "status": "enabled",
                        "availability": "enabled",
                        "initialized": False,
                        "failed": False,
                    }
                },
                "available": 1,
                "total_providers": 1,
                "circuit_breakers": {},
            }

        async def get_capabilities(self):
            return {}

    async def _fake_get_tts_factory():
        return _FakeFactory()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.adapter_registry.get_tts_factory",
        _fake_get_tts_factory,
        raising=True,
    )

    health = await audio_health.get_tts_health(
        request=MagicMock(),
        tts_service=_FakeTTSService(),
    )

    omnivoice_detail = health["providers"]["details"]["omnivoice"]
    assert omnivoice_detail["runtime"] == "sidecar"
    assert omnivoice_detail["sidecar_state"] == "starting"
