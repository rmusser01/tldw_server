from __future__ import annotations

import importlib
import io
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import numpy as np
import pytest
import soundfile as sf
from fastapi import FastAPI
from fastapi.testclient import TestClient

import tldw_Server_API.app.core.Metrics.metrics_manager as metrics_manager
from tldw_Server_API.app.api.v1.endpoints.audio.audio import router as audio_router
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal

TEST_API_KEY = "test-api-key-1234567890"


def _make_wav_bytes(duration_sec: float = 0.1, sr: int = 16000) -> bytes:
    buf = io.BytesIO()
    data = np.zeros(int(sr * duration_sec), dtype=np.float32)
    sf.write(buf, data, sr, format="WAV")
    return buf.getvalue()


def _load_stt_policy_module():
    spec = importlib.util.find_spec(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_policy"
    )
    assert spec is not None
    return importlib.import_module(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_policy"
    )


def _single_user_principal() -> AuthPrincipal:
    return AuthPrincipal(
        kind="user",
        user_id=1,
        api_key_id=None,
        username="single_user",
        email=None,
        subject="single_user",
        token_type="access",
        jti=None,
        roles=["admin"],
        permissions=["*"],
        is_admin=True,
        org_ids=[],
        team_ids=[],
    )


def _setup_stubbed_audio_app(
    monkeypatch: pytest.MonkeyPatch,
    *,
    transcript_text: str,
    temp_outputs_dir: Path,
    storage_service: Any | None = None,
) -> FastAPI:
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", TEST_API_KEY)
    monkeypatch.setenv("SINGLE_USER_TEST_API_KEY", TEST_API_KEY)
    monkeypatch.setenv("SINGLE_USER_FIXED_ID", "1")

    from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
        get_auth_principal,
        get_db_transaction,
    )
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
    import tldw_Server_API.app.api.v1.endpoints.audio.audio as audio_ep
    import tldw_Server_API.app.api.v1.endpoints.audio.audio_transcriptions as audio_transcriptions
    import tldw_Server_API.app.core.DB_Management.db_path_utils as db_path_utils
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_policy as stt_policy
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_provider_adapter as stt_adapter

    async def _fake_get_request_user() -> User:
        return User(id=1, username="single_user")

    async def _fake_get_auth_principal() -> AuthPrincipal:
        return _single_user_principal()

    async def _fake_get_db_transaction():
        yield SimpleNamespace()

    async def _allow_job(*_args: Any, **_kwargs: Any) -> tuple[bool, None]:
        return True, None

    async def _noop_async(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def _get_limits_for_user(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {
            "daily_minutes": 30.0,
            "concurrent_streams": 1,
            "concurrent_jobs": 1,
            "max_file_size_mb": 25,
        }

    class _StubAdapter:
        def transcribe_batch(
            self,
            _audio_path: str,
            *,
            model: str | None = None,
            language: str | None = None,
            task: str = "transcribe",
            word_timestamps: bool = False,
            prompt: str | None = None,
            hotwords: list[str] | None = None,
            base_dir: Path | None = None,
        ) -> dict[str, Any]:
            _ = (task, word_timestamps, prompt, hotwords, base_dir)
            return {
                "text": transcript_text,
                "language": language or "en",
                "segments": [
                    {
                        "start_seconds": 0.0,
                        "end_seconds": 1.0,
                        "Text": transcript_text,
                    }
                ],
                "metadata": {"provider": "external", "model": model or "external:stub"},
            }

    class _StubRegistry:
        def resolve_provider_for_model(self, _model: str) -> tuple[str, str, str | None]:
            return "external", "external:stub", None

        def list_capabilities(self, include_disabled: bool = True) -> list[dict[str, Any]]:
            _ = include_disabled
            return [{"provider": "external", "availability": "available", "capabilities": ["batch"]}]

        def get_adapter(self, _provider: str) -> _StubAdapter:
            return _StubAdapter()

    monkeypatch.setattr(audio_ep, "can_start_job", _allow_job)
    monkeypatch.setattr(audio_ep, "increment_jobs_started", _noop_async)
    monkeypatch.setattr(audio_ep, "finish_job", _noop_async)
    monkeypatch.setattr(audio_ep, "check_daily_minutes_allow", _allow_job)
    monkeypatch.setattr(audio_ep, "add_daily_minutes", _noop_async)
    monkeypatch.setattr(audio_ep, "get_limits_for_user", _get_limits_for_user)
    monkeypatch.setattr(audio_ep, "get_job_heartbeat_interval_seconds", lambda: 0)
    monkeypatch.setattr(audio_ep, "heartbeat_jobs", _noop_async)
    monkeypatch.setattr(stt_adapter, "get_stt_provider_registry", lambda: _StubRegistry())
    monkeypatch.setattr(atlib, "convert_to_wav", lambda path, *args, **kwargs: path)
    monkeypatch.setattr(db_path_utils.DatabasePaths, "get_user_outputs_dir", lambda _user_id: temp_outputs_dir)

    if storage_service is not None:
        async def _get_storage_service():
            return storage_service

        monkeypatch.setattr(stt_policy, "get_storage_service", _get_storage_service)

    app = FastAPI()
    app.dependency_overrides[get_request_user] = _fake_get_request_user
    app.dependency_overrides[get_auth_principal] = _fake_get_auth_principal
    app.dependency_overrides[get_db_transaction] = _fake_get_db_transaction
    app.include_router(audio_router, prefix="/api/v1/audio")
    return app


def test_stt_policy_accepts_stricter_request_overrides() -> None:
    stt_policy = _load_stt_policy_module()

    base = stt_policy.STTPolicy(
        org_id=7,
        delete_audio_after_success=False,
        audio_retention_hours=24.0,
        redact_pii=True,
        allow_unredacted_partials=False,
        redact_categories=["pii_email"],
    )

    merged = stt_policy.merge_request_overrides(
        base,
        audio_retention_hours=1.0,
        redact_categories=["pii_email", "pii_phone"],
        delete_audio_after_success=True,
    )

    assert merged.delete_audio_after_success is True
    assert merged.audio_retention_hours == 1.0
    assert merged.redact_categories == ["pii_email", "pii_phone"]


def test_stt_policy_rejects_weaker_request_overrides() -> None:
    stt_policy = _load_stt_policy_module()

    base = stt_policy.STTPolicy(
        org_id=7,
        delete_audio_after_success=True,
        audio_retention_hours=0.0,
        redact_pii=True,
        allow_unredacted_partials=False,
        redact_categories=["pii_email", "pii_phone"],
    )

    with pytest.raises(ValueError):
        stt_policy.merge_request_overrides(
            base,
            delete_audio_after_success=False,
            redact_pii=False,
            allow_unredacted_partials=True,
            redact_categories=["pii_email"],
        )


def test_audio_transcriptions_redacts_text_and_segments_when_stt_redaction_enabled(
    monkeypatch: pytest.MonkeyPatch,
    bypass_api_limits,
) -> None:
    monkeypatch.setenv("STT_REDACT_PII", "1")
    monkeypatch.setenv("STT_REDACT_CATEGORIES", "pii_email")
    monkeypatch.setenv("STT_ALLOW_UNREDACTED_PARTIALS", "0")

    with tempfile.TemporaryDirectory() as tmpdir:
        app = _setup_stubbed_audio_app(
            monkeypatch,
            transcript_text="contact alice@example.com",
            temp_outputs_dir=Path(tmpdir),
        )

        with bypass_api_limits(app), TestClient(app) as client:
            wav_bytes = _make_wav_bytes()
            files = {"file": ("sample.wav", io.BytesIO(wav_bytes), "audio/wav")}
            data = {"model": "external:stub", "response_format": "json"}
            resp = client.post(
                "/api/v1/audio/transcriptions",
                headers={"X-API-KEY": TEST_API_KEY},
                files=files,
                data=data,
            )

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["text"] == "contact [PII]"
        assert body["segments"][0]["text"] == "contact [PII]"


def test_audio_transcriptions_emit_bounded_stt_metrics_when_redaction_applies(
    monkeypatch: pytest.MonkeyPatch,
    bypass_api_limits,
) -> None:
    monkeypatch.setenv("STT_REDACT_PII", "1")
    monkeypatch.setenv("STT_REDACT_CATEGORIES", "pii_email")
    monkeypatch.setenv("STT_ALLOW_UNREDACTED_PARTIALS", "0")

    metrics_manager._metrics_registry = None
    registry = metrics_manager.get_metrics_registry()

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            app = _setup_stubbed_audio_app(
                monkeypatch,
                transcript_text="contact alice@example.com",
                temp_outputs_dir=Path(tmpdir),
            )

            with bypass_api_limits(app), TestClient(app) as client:
                wav_bytes = _make_wav_bytes()
                files = {"file": ("sample.wav", io.BytesIO(wav_bytes), "audio/wav")}
                data = {"model": "external:stub", "response_format": "json"}
                resp = client.post(
                    "/api/v1/audio/transcriptions",
                    headers={"X-API-KEY": TEST_API_KEY},
                    files=files,
                    data=data,
                )

            assert resp.status_code == 200, resp.text
            assert registry.get_cumulative_counter_total("audio_stt_requests_total") == 1
            assert registry.get_cumulative_counter_totals_by_label("audio_stt_requests_total", "endpoint") == {
                "audio.transcriptions": 1.0
            }
            assert registry.get_cumulative_counter_totals_by_label("audio_stt_requests_total", "provider") == {
                "external": 1.0
            }
            assert registry.get_cumulative_counter_totals_by_label("audio_stt_requests_total", "model") == {
                "other": 1.0
            }
            assert registry.get_cumulative_counter_totals_by_label("audio_stt_requests_total", "status") == {
                "ok": 1.0
            }
            assert registry.get_cumulative_counter(
                "audio_stt_redaction_total",
                {"endpoint": "audio.transcriptions", "redaction_outcome": "applied"},
            ) == 1
            latency_stats = registry.get_metric_stats(
                "audio_stt_latency_seconds",
                labels={"endpoint": "audio.transcriptions", "provider": "external", "model": "other"},
            )
            assert latency_stats.get("count", 0) == 1
    finally:
        metrics_manager._metrics_registry = None


def test_audio_transcriptions_registers_retained_audio_when_retention_enabled(
    monkeypatch: pytest.MonkeyPatch,
    bypass_api_limits,
) -> None:
    monkeypatch.setenv("STT_DELETE_AUDIO_AFTER_SUCCESS", "0")
    monkeypatch.setenv("STT_AUDIO_RETENTION_HOURS", "1")
    monkeypatch.delenv("STT_REDACT_PII", raising=False)

    mock_storage = AsyncMock()
    mock_storage.register_generated_file = AsyncMock(return_value={"id": 7})

    with tempfile.TemporaryDirectory() as tmpdir:
        outputs_dir = Path(tmpdir)
        app = _setup_stubbed_audio_app(
            monkeypatch,
            transcript_text="retained transcript",
            temp_outputs_dir=outputs_dir,
            storage_service=mock_storage,
        )

        with bypass_api_limits(app), TestClient(app) as client:
            wav_bytes = _make_wav_bytes()
            files = {"file": ("sample.wav", io.BytesIO(wav_bytes), "audio/wav")}
            data = {"model": "external:stub", "response_format": "json"}
            resp = client.post(
                "/api/v1/audio/transcriptions",
                headers={"X-API-KEY": TEST_API_KEY},
                files=files,
                data=data,
            )

        assert resp.status_code == 200, resp.text
        assert mock_storage.register_generated_file.await_count == 1
        call_kwargs = mock_storage.register_generated_file.call_args.kwargs
        assert call_kwargs["user_id"] == 1
        assert call_kwargs["file_category"] == "stt_audio"
        assert call_kwargs["source_feature"] == "stt"
        assert call_kwargs["expires_at"] is not None
        retained_path = outputs_dir / call_kwargs["storage_path"]
        assert retained_path.exists()
