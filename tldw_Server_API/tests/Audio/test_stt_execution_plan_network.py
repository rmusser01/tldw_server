"""Execution-plan tests for native network-backed STT providers."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.exceptions import (
    STTExecutionUnsupportedError,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import (
    Audio_Transcription_External_Provider as external,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import (
    Audio_Transcription_Qwen3ASR as qwen3,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import (
    Audio_Transcription_VibeVoice as vibe,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import (
    stt_provider_adapter as spa,
)

_AUDIO_MODULE = (
    "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio"
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "module_order",
    (
        (
            "stt_execution_contract",
            "Audio_Transcription_Qwen3ASR",
            "Audio_Transcription_VibeVoice",
            "Audio_Transcription_External_Provider",
            "stt_provider_adapter",
        ),
        (
            "stt_provider_adapter",
            "Audio_Transcription_External_Provider",
            "Audio_Transcription_VibeVoice",
            "Audio_Transcription_Qwen3ASR",
            "stt_execution_contract",
        ),
        (
            "Audio_Transcription_Qwen3ASR",
            "stt_execution_contract",
            "Audio_Transcription_External_Provider",
            "stt_provider_adapter",
            "Audio_Transcription_VibeVoice",
        ),
    ),
)
def test_network_plan_modules_import_in_any_order(
    module_order: tuple[str, ...],
) -> None:
    imports = "\n".join(
        f"import {_AUDIO_MODULE}.{module}" for module in module_order
    )

    completed = subprocess.run(
        [sys.executable, "-c", imports],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.unit
@pytest.mark.parametrize(
    "url",
    (
        "http://localhost:8000/v1/audio/transcriptions",
        "http://127.0.0.1/v1/audio/transcriptions",
        "http://127.42.0.9/v1/audio/transcriptions",
        "http://[::1]/v1/audio/transcriptions",
        "http://[::ffff:127.0.0.1]/v1/audio/transcriptions",
    ),
)
def test_classify_audio_egress_accepts_only_literal_loopback(url: str) -> None:
    assert spa._classify_audio_egress(url) is spa.SttAudioEgress.LOOPBACK


@pytest.mark.unit
@pytest.mark.parametrize(
    "url",
    (
        "https://api.localhost/v1/audio/transcriptions",
        "https://localhost.example/v1/audio/transcriptions",
        "http://192.168.1.10/v1/audio/transcriptions",
        "https://example.com/v1/audio/transcriptions",
        "http://[::]/v1/audio/transcriptions",
    ),
)
def test_classify_audio_egress_treats_every_other_host_as_remote(url: str) -> None:
    assert spa._classify_audio_egress(url) is spa.SttAudioEgress.REMOTE


@pytest.mark.unit
@pytest.mark.parametrize(
    "url",
    (
        "",
        "localhost:8000/v1/audio/transcriptions",
        "ftp://localhost/v1/audio/transcriptions",
        "http:///v1/audio/transcriptions",
        "http://user@example.com/v1/audio/transcriptions",
        "http://example.com/v1/audio/transcriptions?secret=value",
        "http://example.com/v1/audio/transcriptions#fragment",
        "http://example.com:",
        "http://example.com:not-a-port/v1/audio/transcriptions",
        "http://example.com:70000/v1/audio/transcriptions",
        "http://::1/v1/audio/transcriptions",
    ),
)
def test_classify_audio_egress_rejects_malformed_or_ambiguous_urls(url: str) -> None:
    with pytest.raises(STTExecutionUnsupportedError):
        spa._classify_audio_egress(url)


@pytest.mark.unit
def test_normalized_endpoint_identity_is_opaque_and_covers_complete_path() -> None:
    first, first_egress, first_id = spa._normalize_audio_endpoint(
        "HTTPS://EXAMPLE.COM:443/v1/audio/../audio/transcriptions"
    )
    equivalent, equivalent_egress, equivalent_id = spa._normalize_audio_endpoint(
        "https://example.com/v1/audio/transcriptions"
    )
    other_path, _, other_id = spa._normalize_audio_endpoint(
        "https://example.com/private/transcriptions"
    )

    assert first == equivalent == "https://example.com/v1/audio/transcriptions"
    assert first_egress is equivalent_egress is spa.SttAudioEgress.REMOTE
    assert first_id == equivalent_id
    assert other_path == "https://example.com/private/transcriptions"
    assert other_id != first_id
    assert first_id.startswith("sha256:")
    assert len(first_id) == len("sha256:") + 64
    assert "example.com" not in first_id
    assert "transcriptions" not in first_id


def _plan(
    adapter: spa.SttProviderAdapter,
    *,
    model: str,
    language: str | None = "en",
    task: str = "transcribe",
    word_timestamps: bool = False,
    prompt: str | None = None,
    hotwords: tuple[str, ...] = (),
    diarization: bool = False,
    mode: str = "neutral-v1",
) -> spa.SttBatchExecutionPlan:
    return adapter.plan_batch_execution(
        model=model,
        language=language,
        task=task,
        word_timestamps=word_timestamps,
        prompt=prompt,
        hotwords=hotwords,
        diarization=diarization,
        mode=mode,
    )


@pytest.mark.unit
def test_qwen3_vllm_plan_freezes_endpoint_and_reports_actual_route(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    audio = tmp_path / "qwen.wav"
    audio.write_bytes(b"audio")
    settings: dict[str, Any] = {
        "enabled": True,
        "backend": "vllm",
        "vllm_base_url": "HTTP://LOCALHOST:8000/proxy/..",
        "model_path": "/ignored/local/model",
        "model_revision": "",
        "device": "cuda",
        "dtype": "bfloat16",
        "allow_download": False,
        "sample_rate": 16000,
        "max_new_tokens": 4096,
        "aligner_enabled": False,
    }
    monkeypatch.setattr(
        qwen3,
        "_resolve_settings",
        lambda: dict(settings),
    )
    adapter = spa.Qwen3ASRAdapter()

    plan = _plan(
        adapter,
        model="Qwen/Qwen3-ASR-1.7B",
    )
    route = plan.descriptor.primary_route
    runtime = plan.runtime_values()

    assert route.backend == "vllm_http"
    assert route.source == "vllm_http"
    assert route.audio_egress is spa.SttAudioEgress.LOOPBACK
    assert route.endpoint_id == spa._normalize_audio_endpoint(
        "http://localhost:8000/v1/audio/transcriptions"
    )[2]
    assert runtime["endpoint"] == (
        "http://localhost:8000/v1/audio/transcriptions"
    )
    assert "localhost" not in str(plan.descriptor.as_safe_dict())

    settings.update(
        backend="transformers",
        vllm_base_url="https://mutated.example/private",
        model_path="/mutated/model",
        device="cpu",
        dtype="float32",
    )
    monkeypatch.setattr(
        qwen3,
        "_resolve_settings",
        lambda: (_ for _ in ()).throw(
            AssertionError("planned execution reread Qwen config")
        ),
    )
    captured: dict[str, Any] = {}

    def fake_vllm(
        audio_path: Path,
        frozen_settings: dict[str, Any],
        language: str | None,
        cancel_check: Any,
    ) -> dict[str, Any]:
        captured.update(frozen_settings)
        assert audio_path == audio
        assert language == "en"
        return {
            "text": "planned qwen",
            "language": "en",
            "segments": [
                {
                    "start_seconds": 0.0,
                    "end_seconds": 1.0,
                    "Text": "planned qwen",
                }
            ],
            "metadata": {"source": "hostile-provider-metadata"},
        }

    monkeypatch.setattr(qwen3, "_transcribe_vllm_http", fake_vllm)

    artifact = adapter.transcribe_batch(
        str(audio),
        model="Qwen/Qwen3-ASR-1.7B",
        language="en",
        base_dir=tmp_path,
        execution_plan=plan,
    )

    assert captured["endpoint"] == runtime["endpoint"]
    assert artifact["actual_execution"]["route_id"] == route.route_id
    assert artifact["actual_execution"]["source"] == "vllm_http"
    assert artifact["actual_execution"]["backend"] == "vllm_http"
    assert "metadata" not in artifact


@pytest.mark.unit
def test_qwen3_local_plan_freezes_model_device_and_dtype(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "qwen-model"
    model_dir.mkdir()
    revision = "a" * 40
    settings: dict[str, Any] = {
        "enabled": True,
        "backend": "transformers",
        "vllm_base_url": "",
        "model_path": str(model_dir),
        "model_revision": revision,
        "device": "cpu",
        "dtype": "float32",
        "allow_download": False,
        "sample_rate": 16000,
        "max_new_tokens": 1024,
        "aligner_enabled": False,
    }
    monkeypatch.setattr(qwen3, "_resolve_settings", lambda: dict(settings))

    plan = _plan(
        spa.Qwen3ASRAdapter(),
        model=str(model_dir),
        language=None,
    )
    route = plan.descriptor.primary_route

    assert route.backend == "transformers"
    assert route.source == "local"
    assert route.audio_egress is spa.SttAudioEgress.NONE
    assert route.artifact_id == revision
    assert route.device == "cpu"
    assert route.dtype == "float32"
    assert plan.runtime_values()["model_path"] == str(model_dir.resolve())


@pytest.mark.unit
@pytest.mark.parametrize(
    "overrides",
    (
        {"task": "translate"},
        {"word_timestamps": True},
        {"prompt": "private prompt"},
        {"hotwords": ("private-hotword",)},
        {"diarization": True},
    ),
)
def test_qwen3_neutral_plan_rejects_unsupported_semantics(
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, Any],
) -> None:
    monkeypatch.setattr(
        qwen3,
        "_resolve_settings",
        lambda: {
            "enabled": True,
            "backend": "vllm",
            "vllm_base_url": "http://localhost:8000",
            "model_path": "/ignored",
            "model_revision": "",
            "device": "cpu",
            "dtype": "float32",
            "allow_download": False,
            "sample_rate": 16000,
            "max_new_tokens": 1024,
            "aligner_enabled": False,
        },
    )
    kwargs: dict[str, Any] = {
        "model": "Qwen/Qwen3-ASR-1.7B",
        "language": "en",
        **overrides,
    }

    with pytest.raises(STTExecutionUnsupportedError):
        _plan(spa.Qwen3ASRAdapter(), **kwargs)


@pytest.mark.unit
def test_qwen3_local_neutral_plan_rejects_dropped_language(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "qwen-model"
    model_dir.mkdir()
    monkeypatch.setattr(
        qwen3,
        "_resolve_settings",
        lambda: {
            "enabled": True,
            "backend": "transformers",
            "vllm_base_url": "",
            "model_path": str(model_dir),
            "model_revision": "b" * 40,
            "device": "cpu",
            "dtype": "float32",
            "allow_download": False,
            "sample_rate": 16000,
            "max_new_tokens": 1024,
            "aligner_enabled": False,
        },
    )

    with pytest.raises(STTExecutionUnsupportedError, match="language"):
        _plan(
            spa.Qwen3ASRAdapter(),
            model=str(model_dir),
            language="en-US",
        )


def _vibe_settings(
    model_dir: Path,
    *,
    vllm_enabled: bool,
    local_enabled: bool = True,
) -> dict[str, Any]:
    return {
        "enabled": local_enabled,
        "model_id": str(model_dir),
        "model_revision": "c" * 40,
        "device": "cpu",
        "dtype": "float32",
        "cache_dir": str(model_dir.parent / "cache"),
        "allow_download": False,
        "sample_rate": 16000,
        "max_new_tokens": 1024,
        "vllm_enabled": vllm_enabled,
        "vllm_base_url": "http://127.0.0.1:9000/api",
        "vllm_model_id": str(model_dir),
        "vllm_api_key": "vibe-api-secret",
        "vllm_timeout_seconds": 45,
    }


@pytest.mark.unit
def test_vibevoice_local_neutral_plan_pins_one_strict_route(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "vibe-model"
    model_dir.mkdir()
    monkeypatch.setattr(
        vibe,
        "_resolve_settings",
        lambda: _vibe_settings(
            model_dir,
            vllm_enabled=False,
        ),
    )

    plan = _plan(
        spa.VibeVoiceAdapter(),
        model=str(model_dir),
    )
    route = plan.descriptor.primary_route

    assert len(plan.descriptor.routes) == 1
    assert route.backend == "transformers"
    assert route.source == "local"
    assert route.audio_egress is spa.SttAudioEgress.NONE
    assert route.device == "cpu"
    assert route.dtype == "float32"
    assert plan.runtime_values()["strict_semantics"] is True


@pytest.mark.unit
def test_vibevoice_neutral_vllm_plan_never_falls_back(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    audio = tmp_path / "vibe.wav"
    audio.write_bytes(b"audio")
    model_dir = tmp_path / "vibe-model"
    model_dir.mkdir()
    settings = _vibe_settings(
        model_dir,
        vllm_enabled=True,
    )
    monkeypatch.setattr(
        vibe,
        "_resolve_settings",
        lambda: dict(settings),
    )
    adapter = spa.VibeVoiceAdapter()
    plan = _plan(
        adapter,
        model=str(model_dir),
    )

    assert len(plan.descriptor.routes) == 1
    assert plan.descriptor.primary_route.backend == "vllm_http"

    monkeypatch.setattr(
        vibe,
        "_resolve_settings",
        lambda: (_ for _ in ()).throw(
            AssertionError("planned execution reread VibeVoice config")
        ),
    )
    monkeypatch.setattr(
        vibe,
        "_transcribe_via_vllm_http",
        lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError(
                "vLLM failed at https://private.invalid/secret/path "
                "with vibe-api-secret"
            )
        ),
    )
    monkeypatch.setattr(
        vibe,
        "_transcribe_local",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("neutral plan fell back to local")
        ),
    )

    with pytest.raises(spa.STTTranscriptionError) as exc_info:
        adapter.transcribe_batch(
            str(audio),
            model=str(model_dir),
            language="en",
            base_dir=tmp_path,
            execution_plan=plan,
        )

    for secret in (
        "private.invalid",
        "secret/path",
        "vibe-api-secret",
    ):
        assert secret not in str(exc_info.value)
        assert secret not in caplog.text


@pytest.mark.unit
def test_vibevoice_production_plan_freezes_exact_fallback_and_actual_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    audio = tmp_path / "vibe.wav"
    audio.write_bytes(b"audio")
    model_dir = tmp_path / "vibe-model"
    model_dir.mkdir()
    settings = _vibe_settings(
        model_dir,
        vllm_enabled=True,
    )
    monkeypatch.setattr(
        vibe,
        "_resolve_settings",
        lambda: dict(settings),
    )
    adapter = spa.VibeVoiceAdapter()
    plan = _plan(
        adapter,
        model=str(model_dir),
        hotwords=("private-hotword",),
        mode="production-v1",
    )

    assert [
        route.backend for route in plan.descriptor.routes
    ] == ["vllm_http", "transformers"]
    assert [
        route.route_id for route in plan.descriptor.routes
    ] == ["vllm-http-1", "local-2"]
    assert dict(plan.descriptor.decoding_settings) == {
        "hotword_count": 1,
        "prompt_present": False,
    }
    safe_plan = str(plan.descriptor.as_safe_dict())
    assert "private-hotword" not in safe_plan
    assert "vibe-api-secret" not in safe_plan
    assert "127.0.0.1" not in safe_plan
    assert "private-hotword" not in repr(plan)
    assert "vibe-api-secret" not in repr(plan)

    settings.update(
        vllm_base_url="https://mutated.example/private",
        vllm_api_key="mutated-secret",
        model_id="/mutated/model",
        device="cuda",
        dtype="bfloat16",
    )
    monkeypatch.setattr(
        vibe,
        "_resolve_settings",
        lambda: (_ for _ in ()).throw(
            AssertionError("planned execution reread VibeVoice config")
        ),
    )
    attempts: list[str] = []

    def fail_vllm(**kwargs: Any) -> dict[str, Any]:
        attempts.append("vllm")
        assert kwargs["settings"]["endpoint"] == (
            "http://127.0.0.1:9000/api/v1/audio/transcriptions"
        )
        assert kwargs["settings"]["vllm_api_key"] == "vibe-api-secret"
        raise RuntimeError("planned vLLM unavailable")

    def succeed_local(**kwargs: Any) -> dict[str, Any]:
        attempts.append("local")
        assert kwargs["settings"]["model_id"] == str(
            model_dir.resolve()
        )
        assert kwargs["settings"]["device"] == "cpu"
        assert kwargs["settings"]["dtype"] == "float32"
        assert kwargs["settings"]["strict_semantics"] is True
        assert kwargs["language"] == "en"
        assert kwargs["hotwords"] == ["private-hotword"]
        return {
            "text": "local fallback",
            "language": "en",
            "segments": [
                {
                    "start_seconds": 0.0,
                    "end_seconds": 1.0,
                    "Text": "local fallback",
                }
            ],
            "metadata": {
                "hotwords": ["private-hotword"],
                "source": "untrusted",
            },
        }

    monkeypatch.setattr(
        vibe,
        "_transcribe_via_vllm_http",
        fail_vllm,
    )
    monkeypatch.setattr(
        vibe,
        "_load_audio",
        lambda *_args, **_kwargs: (
            object(),
            16000,
            1.0,
        ),
    )
    monkeypatch.setattr(vibe, "_transcribe_local", succeed_local)
    real_finalize = spa.finalize_stt_artifact

    def assert_fallback_before_finalization(
        artifact: dict[str, Any],
        *,
        plan: spa.SttBatchExecutionPlan,
        actual: spa.SttActualExecution,
    ) -> dict[str, Any]:
        assert actual.route_id == "local-2"
        assert actual.backend == "transformers"
        assert actual.source == "local"
        return real_finalize(
            artifact,
            plan=plan,
            actual=actual,
        )

    monkeypatch.setattr(
        spa,
        "finalize_stt_artifact",
        assert_fallback_before_finalization,
    )

    artifact = adapter.transcribe_batch(
        str(audio),
        model=str(model_dir),
        language="en",
        hotwords=("private-hotword",),
        base_dir=tmp_path,
        execution_plan=plan,
    )

    assert attempts == ["vllm", "local"]
    assert artifact["actual_execution"]["route_id"] == "local-2"
    assert artifact["actual_execution"]["backend"] == "transformers"
    assert artifact["actual_execution"]["source"] == "local"
    assert "metadata" not in artifact
    assert "private-hotword" not in str(artifact)


@pytest.mark.unit
@pytest.mark.parametrize(
    "overrides",
    (
        {"task": "translate"},
        {"word_timestamps": True},
        {"prompt": "private prompt"},
        {"hotwords": ("private-hotword",)},
    ),
)
def test_vibevoice_neutral_plan_rejects_unsupported_semantics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    overrides: dict[str, Any],
) -> None:
    model_dir = tmp_path / "vibe-model"
    model_dir.mkdir()
    monkeypatch.setattr(
        vibe,
        "_resolve_settings",
        lambda: _vibe_settings(
            model_dir,
            vllm_enabled=True,
        ),
    )

    with pytest.raises(STTExecutionUnsupportedError):
        _plan(
            spa.VibeVoiceAdapter(),
            model=str(model_dir),
            **overrides,
        )


def _external_config() -> external.ExternalProviderConfig:
    return external.ExternalProviderConfig(
        base_url="HTTPS://API.EXAMPLE.COM:443/proxy/..",
        api_key="external-api-secret",
        model="whisper-large",
        timeout=91.5,
        max_retries=4,
        verify_ssl=False,
        custom_headers={
            "X-Private-Token": "external-header-secret",
        },
        response_format="json",
        temperature=0.25,
        language=None,
        prompt=None,
    )


@pytest.mark.unit
def test_external_plan_freezes_config_and_returns_safe_typed_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    audio = tmp_path / "external.wav"
    audio.write_bytes(b"audio")
    config = _external_config()
    monkeypatch.setattr(
        external,
        "load_external_provider_config",
        lambda _name="default": config,
    )
    adapter = spa.ExternalAdapter()
    plan = _plan(
        adapter,
        model="external:custom",
    )
    route = plan.descriptor.primary_route

    assert route.backend == "openai_compatible"
    assert route.source == "external_http"
    assert route.audio_egress is spa.SttAudioEgress.REMOTE
    assert route.endpoint_id == spa._normalize_audio_endpoint(
        "https://api.example.com/v1/audio/transcriptions"
    )[2]
    safe_descriptor = str(plan.descriptor.as_safe_dict())
    for secret in (
        "api.example.com",
        "external-api-secret",
        "X-Private-Token",
        "external-header-secret",
    ):
        assert secret not in safe_descriptor
        assert secret not in repr(plan)

    config.base_url = "https://mutated.example/private"
    config.api_key = "mutated-api-secret"
    config.model = "mutated-model"
    config.timeout = 1
    config.max_retries = 1
    config.verify_ssl = True
    config.custom_headers = {"X-Mutated": "mutated-header"}
    config.response_format = "text"
    config.temperature = 2.0
    config.language = "fr"
    config.prompt = "mutated-prompt"
    monkeypatch.setattr(
        external,
        "load_external_provider_config",
        lambda _name="default": (_ for _ in ()).throw(
            AssertionError("planned external execution reread config")
        ),
    )
    captured: dict[str, Any] = {}

    async def fake_async(
        _audio_data: object,
        _sample_rate: int = 16000,
        _provider_name: str = "default",
        frozen_config: external.ExternalProviderConfig | None = None,
        _base_dir: Path | None = None,
        **_kwargs: Any,
    ) -> str:
        assert frozen_config is not None
        captured.update(vars(frozen_config))
        return "planned external"

    monkeypatch.setattr(
        external,
        "transcribe_with_external_provider_async",
        fake_async,
    )

    artifact = adapter.transcribe_batch(
        str(audio),
        model="external:custom",
        language="en",
        base_dir=tmp_path,
        execution_plan=plan,
    )

    assert captured == {
        "base_url": (
            "https://api.example.com/v1/audio/transcriptions"
        ),
        "api_key": "external-api-secret",
        "model": "whisper-large",
        "timeout": 91.5,
        "max_retries": 4,
        "verify_ssl": False,
        "custom_headers": {
            "X-Private-Token": "external-header-secret",
        },
        "response_format": "json",
        "temperature": 0.25,
        "language": "en",
        "prompt": None,
    }
    assert artifact["text"] == "planned external"
    assert artifact["actual_execution"]["route_id"] == route.route_id
    assert artifact["actual_execution"]["endpoint_id"] == (
        route.endpoint_id
    )
    assert "metadata" not in artifact
    for secret in (
        "external-api-secret",
        "external-header-secret",
        "api.example.com",
    ):
        assert secret not in str(artifact)


@pytest.mark.unit
def test_external_planned_sentinel_becomes_typed_redacted_error(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    audio = tmp_path / "external.wav"
    audio.write_bytes(b"audio")
    monkeypatch.setattr(
        external,
        "load_external_provider_config",
        lambda _name="default": _external_config(),
    )
    adapter = spa.ExternalAdapter()
    plan = _plan(
        adapter,
        model="external:custom",
    )

    async def sentinel(*_args: Any, **_kwargs: Any) -> str:
        return (
            "[Error: leaked https://api.example.com/private/path "
            "external-api-secret X-Private-Token "
            "external-header-secret private-hotword]"
        )

    monkeypatch.setattr(
        external,
        "transcribe_with_external_provider_async",
        sentinel,
    )

    with pytest.raises(spa.STTTranscriptionError) as exc_info:
        adapter.transcribe_batch(
            str(audio),
            model="external:custom",
            language="en",
            base_dir=tmp_path,
            execution_plan=plan,
        )

    assert str(exc_info.value) == (
        "Planned local STT transcription failed"
    )
    for secret in (
        "api.example.com",
        "private/path",
        "external-api-secret",
        "X-Private-Token",
        "external-header-secret",
        "private-hotword",
    ):
        assert secret not in str(exc_info.value)
        assert secret not in caplog.text


@pytest.mark.unit
@pytest.mark.parametrize(
    "overrides",
    (
        {"task": "translate"},
        {"word_timestamps": True},
        {"prompt": "private prompt"},
        {"hotwords": ("private-hotword",)},
        {"diarization": True},
    ),
)
def test_external_neutral_plan_rejects_unsupported_semantics(
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, Any],
) -> None:
    monkeypatch.setattr(
        external,
        "load_external_provider_config",
        lambda _name="default": _external_config(),
    )

    with pytest.raises(STTExecutionUnsupportedError):
        _plan(
            spa.ExternalAdapter(),
            model="external:custom",
            **overrides,
        )


@pytest.mark.unit
def test_external_neutral_plan_rejects_configured_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _external_config()
    config.prompt = "configured-private-prompt"
    monkeypatch.setattr(
        external,
        "load_external_provider_config",
        lambda _name="default": config,
    )

    with pytest.raises(STTExecutionUnsupportedError, match="prompt"):
        _plan(
            spa.ExternalAdapter(),
            model="external:custom",
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_external_network_call_disables_redirects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    audio = tmp_path / "external.wav"
    audio.write_bytes(b"audio")
    captured: dict[str, Any] = {}

    class Response:
        status_code = 200

        def json(self) -> dict[str, str]:
            return {"text": "redirect-safe"}

        async def aclose(self) -> None:
            return None

    async def fake_afetch(**kwargs: Any) -> Response:
        captured.update(kwargs)
        return Response()

    monkeypatch.setattr(external, "afetch", fake_afetch)
    result = await external.transcribe_with_external_provider_async(
        audio,
        config=_external_config(),
        base_dir=tmp_path,
    )

    assert result == "redirect-safe"
    assert captured["allow_redirects"] is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_external_sync_bridge_forwards_frozen_plan_from_running_loop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    audio = tmp_path / "external.wav"
    audio.write_bytes(b"audio")
    config = _external_config()
    monkeypatch.setattr(
        external,
        "load_external_provider_config",
        lambda _name="default": config,
    )
    plan = _plan(
        spa.ExternalAdapter(),
        model="external:custom",
    )
    captured: dict[str, Any] = {}

    async def fake_async(
        *_args: Any,
        execution_plan: spa.SttBatchExecutionPlan | None = None,
        **_kwargs: Any,
    ) -> str:
        captured["execution_plan"] = execution_plan
        return "threaded planned external"

    monkeypatch.setattr(
        external,
        "transcribe_with_external_provider_async",
        fake_async,
    )

    outcome = external.transcribe_with_external_provider(
        audio,
        config=config,
        base_dir=tmp_path,
        execution_plan=plan,
    )

    assert captured["execution_plan"] is plan
    assert isinstance(outcome, spa.SttTranscriptionOutcome)
    assert outcome.artifact["text"] == "threaded planned external"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_external_async_planned_call_never_loads_live_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    audio = tmp_path / "external.wav"
    audio.write_bytes(b"audio")
    monkeypatch.setattr(
        external,
        "load_external_provider_config",
        lambda _name="default": _external_config(),
    )
    plan = _plan(
        spa.ExternalAdapter(),
        model="external:custom",
    )
    config_loads: list[str] = []
    monkeypatch.setattr(
        external,
        "load_external_provider_config",
        lambda name="default": config_loads.append(name),
    )

    with pytest.raises(
        spa.STTExecutionPlanError,
        match="frozen configuration",
    ):
        await external.transcribe_with_external_provider_async(
            audio,
            config=None,
            base_dir=tmp_path,
            execution_plan=plan,
        )

    assert config_loads == []
