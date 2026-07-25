"""Execution-plan tests for native network-backed STT providers."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import Helper_Scripts.benchmarks.stt_bench as stt_bench
import httpx
import pytest
from yarl import URL

from tldw_Server_API.app.core import http_client
from tldw_Server_API.app.core.exceptions import (
    CancelCheckError,
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
        "http://example.com:0/v1/audio/transcriptions",
        "http://example.com:70000/v1/audio/transcriptions",
        "http://::1/v1/audio/transcriptions",
        "https://éxample.com/v1/audio/transcriptions",
        "https://%65xample.com/v1/audio/transcriptions",
        "https://[fe80::1%25en0]/v1/audio/transcriptions",
        "https://example.com./v1/audio/transcriptions",
        "https://foo..example.com/v1/audio/transcriptions",
        "https://example.com/\x01/v1/audio/transcriptions",
        "https://user%40example.com@example.net/v1/audio/transcriptions",
        "https://example.com/%2e%2e/private/transcriptions",
        "https://example.com/%7E/private/transcriptions",
        "https://example.com/%ZZ/private/transcriptions",
        "https://example.com/a b/transcriptions",
        "https://example.com/{x}/transcriptions",
        "https://example.com/a|b/transcriptions",
        "https://example.com/[x]/transcriptions",
        "https://example.com/proxy;private/v1/audio/transcriptions",
        "https://example.com//v1///audio/transcriptions",
        "http://127.000.0.1/v1/audio/transcriptions",
        "http://0177.0.0.1/v1/audio/transcriptions",
        "http://127.1/v1/audio/transcriptions",
        "http://2130706433/v1/audio/transcriptions",
        "http://0x7f000001/v1/audio/transcriptions",
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
    assert first_id == (
        "sha256:"
        + hashlib.sha256(first.encode("ascii")).hexdigest()
    )
    assert "example.com" not in first_id
    assert "transcriptions" not in first_id


@pytest.mark.unit
@pytest.mark.parametrize(
    ("source", "expected"),
    (
        (
            "http://example.com:080/v1/audio/transcriptions",
            "http://example.com/v1/audio/transcriptions",
        ),
        (
            "https://example.com:0443/v1/audio/transcriptions",
            "https://example.com/v1/audio/transcriptions",
        ),
        (
            "http://example.com:081/v1/audio/transcriptions",
            "http://example.com:81/v1/audio/transcriptions",
        ),
        (
            "http://[0:0:0:0:0:0:0:1]/v1/audio/transcriptions",
            "http://[::1]/v1/audio/transcriptions",
        ),
    ),
)
def test_normalized_endpoint_matches_httpx_and_yarl_transport_authority(
    source: str,
    expected: str,
) -> None:
    normalized, _egress, endpoint_id = (
        spa._normalize_audio_endpoint(source)
    )

    assert normalized == expected
    assert str(httpx.URL(normalized)) == expected
    assert str(URL(normalized)) == expected
    assert endpoint_id == (
        "sha256:"
        + hashlib.sha256(expected.encode("ascii")).hexdigest()
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("provider", "unsafe_base"),
    (
        ("qwen3", "https://example.com/proxy?api_key=private"),
        ("vibevoice", "https://example.com/proxy?api_key=private"),
        ("external", "https://example.com/proxy?api_key=private"),
        ("qwen3", "https://user%40name@example.com/proxy"),
        ("vibevoice", "https://user%40name@example.com/proxy"),
        ("external", "https://user%40name@example.com/proxy"),
        ("qwen3", "https://example.com/proxy;private"),
        ("vibevoice", "https://example.com/proxy;private"),
        ("external", "https://example.com/proxy;private"),
    ),
)
def test_network_plans_validate_configured_base_before_appending_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    unsafe_base: str,
) -> None:
    if provider == "qwen3":
        monkeypatch.setattr(
            qwen3,
            "_resolve_settings",
            lambda: {
                "enabled": True,
                "backend": "vllm",
                "vllm_base_url": unsafe_base,
                "model_path": "Qwen/Qwen3-ASR-1.7B",
                "sample_rate": 16000,
            },
        )
        adapter: spa.SttProviderAdapter = spa.Qwen3ASRAdapter()
        model = "Qwen/Qwen3-ASR-1.7B"
    elif provider == "vibevoice":
        monkeypatch.setattr(
            vibe,
            "_resolve_settings",
            lambda: {
                "enabled": False,
                "vllm_enabled": True,
                "vllm_base_url": unsafe_base,
                "vllm_model_id": "microsoft/VibeVoice-ASR",
                "model_id": "microsoft/VibeVoice-ASR",
                "sample_rate": 16000,
            },
        )
        adapter = spa.VibeVoiceAdapter()
        model = "microsoft/VibeVoice-ASR"
    else:
        config = _external_config()
        config.base_url = unsafe_base
        monkeypatch.setattr(
            external,
            "load_external_provider_config",
            lambda _name="default": config,
        )
        adapter = spa.ExternalAdapter()
        model = "external:custom"

    with pytest.raises(STTExecutionUnsupportedError):
        _plan(adapter, model=model)


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
    assert runtime["request_model"] == "Qwen/Qwen3-ASR-1.7B"
    assert route.model_label == "Qwen/Qwen3-ASR-1.7B"
    assert {
        "tldw_Server_API.app.core.http_client",
        "tldw_Server_API.app.core.stt_observability_context",
    }.issubset(plan.descriptor.source_modules)
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
    assert captured["request_model"] == runtime["request_model"]
    assert artifact["actual_execution"]["route_id"] == route.route_id
    assert artifact["actual_execution"]["source"] == "vllm_http"
    assert artifact["actual_execution"]["backend"] == "vllm_http"
    assert artifact["actual_execution"]["model_label"] == (
        runtime["request_model"]
    )
    assert "metadata" not in artifact


@pytest.mark.unit
def test_planned_qwen3_upload_uses_hardened_scoped_http_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    audio = tmp_path / "qwen.wav"
    audio.write_bytes(b"audio")
    endpoint = "http://127.0.0.1:1/v1/audio/transcriptions"
    endpoint_id = spa._normalize_audio_endpoint(endpoint)[2]
    captured: dict[str, Any] = {}
    client = object()

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, str]:
            return {"text": "scoped upload"}

    class ClientContext:
        def __enter__(self) -> object:
            return client

        def __exit__(self, *_args: object) -> None:
            return None

    def fake_create_client(**kwargs: Any) -> ClientContext:
        captured["client_settings"] = kwargs
        return ClientContext()

    def fake_fetch(**kwargs: Any) -> Response:
        captured.update(kwargs)
        return Response()

    monkeypatch.setenv("HTTP_TRUST_ENV", "true")
    monkeypatch.setattr(http_client, "create_client", fake_create_client)
    monkeypatch.setattr(http_client, "fetch", fake_fetch)
    monkeypatch.setattr(
        qwen3,
        "_load_audio",
        lambda *_args, **_kwargs: (object(), 16000, 1.0),
    )

    artifact = qwen3._transcribe_vllm_http(
        audio,
        {
            "endpoint": endpoint,
            "endpoint_id": endpoint_id,
            "request_model": "Qwen/Qwen3-ASR-1.7B",
            "sample_rate": 16000,
        },
        "en",
        None,
    )

    assert artifact["text"] == "scoped upload"
    assert captured["method"] == "POST"
    assert captured["url"] == endpoint
    assert captured["allow_redirects"] is False
    assert captured["configured_endpoint"].matches(endpoint)
    assert captured["retry"].attempts == 1
    assert captured["client"] is client
    assert captured["client_settings"]["trust_env"] is False


@pytest.mark.unit
def test_qwen3_local_plan_freezes_model_device_and_dtype(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "qwen-model"
    model_dir.mkdir()
    (model_dir / "weights.bin").write_bytes(b"qwen-test-model")
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
    assert route.artifact_id is not None
    assert route.artifact_id.startswith("sha256:")
    assert route.identity_resolved is True
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
    model_dir.mkdir(parents=True, exist_ok=True)
    weights = model_dir / "weights.bin"
    if not weights.exists():
        weights.write_bytes(b"vibe-test-model")
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
        "vllm_model_id": "microsoft/VibeVoice-ASR-HTTP",
        "vllm_api_key": "vibe-api-secret",
        "vllm_timeout_seconds": 45,
    }


@pytest.mark.unit
def test_vibevoice_local_neutral_plan_pins_one_mismatch_tracking_route(
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
    assert plan.runtime_values()["strict_semantics"] is False
    contract_json, contract_hash = stt_bench.build_execution_contract(
        plan=plan,
        git_commit="a" * 40,
        safe_target_settings={
            "mode": "neutral-v1",
            "task": "transcribe",
            "language": "en",
            "word_timestamps": False,
            "diarization": False,
            "prompt_present": False,
            "hotword_count": 0,
        },
    )
    prepared = stt_bench.PreparedTarget(
        target_id="target-vibevoice",
        provider="vibevoice",
        model_label=plan.descriptor.requested_model_label,
        plan=plan,
        adapter_factory_path="unused:factory",
        execution_contract_json=contract_json,
        execution_contract_hash=contract_hash,
    )

    stt_bench._verify_worker_target(prepared)


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
def test_vibevoice_planned_request_uses_revalidated_canonical_endpoint(
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
    settings["vllm_base_url"] = "http://[::1]:9000/api"
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
    expanded_endpoint = (
        "http://[0:0:0:0:0:0:0:1]:9000"
        "/api/v1/audio/transcriptions"
    )
    mutated = replace(
        plan,
        runtime_settings=tuple(
            (
                key,
                expanded_endpoint if key == "endpoint" else value,
            )
            for key, value in plan.runtime_settings
        ),
    )

    def fake_vllm(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["settings"]["endpoint"] == (
            "http://[::1]:9000/api/v1/audio/transcriptions"
        )
        return {
            "text": "canonical endpoint",
            "segments": [],
        }

    monkeypatch.setattr(
        vibe,
        "_transcribe_via_vllm_http",
        fake_vllm,
    )

    artifact = adapter.transcribe_batch(
        str(audio),
        model=str(model_dir),
        language="en",
        base_dir=tmp_path,
        execution_plan=mutated,
    )

    assert artifact["text"] == "canonical endpoint"


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
    assert [
        route.model_label for route in plan.descriptor.routes
    ] == ["microsoft/VibeVoice-ASR-HTTP", "local-model"]
    assert {
        "tldw_Server_API.app.core.Security.egress",
        "tldw_Server_API.app.core.http_client",
        "tldw_Server_API.app.core.stt_observability_context",
    }.issubset(plan.descriptor.source_modules)
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
        assert kwargs["settings"]["vllm_model_id"] == (
            "microsoft/VibeVoice-ASR-HTTP"
        )
        raise RuntimeError("planned vLLM unavailable")

    def succeed_local(**kwargs: Any) -> dict[str, Any]:
        attempts.append("local")
        assert kwargs["settings"]["model_id"] == str(
            model_dir.resolve()
        )
        assert kwargs["settings"]["device"] == "cpu"
        assert kwargs["settings"]["dtype"] == "float32"
        assert kwargs["settings"]["strict_semantics"] is False
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
        runtime_mismatches: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        assert actual.route_id == "local-2"
        assert actual.backend == "transformers"
        assert actual.source == "local"
        return real_finalize(
            artifact,
            plan=plan,
            actual=actual,
            runtime_mismatches=runtime_mismatches,
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
    assert artifact["actual_execution"]["model_label"] == (
        "local-model"
    )
    assert "metadata" not in artifact
    assert "private-hotword" not in str(artifact)


@pytest.mark.unit
def test_vibevoice_planned_fallback_propagates_cancel_check_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    audio = tmp_path / "vibe.wav"
    audio.write_bytes(b"audio")
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
    adapter = spa.VibeVoiceAdapter()
    plan = _plan(
        adapter,
        model=str(model_dir),
        mode="production-v1",
    )
    cancel_error = CancelCheckError("cancel callback failed")
    monkeypatch.setattr(
        vibe,
        "_transcribe_via_vllm_http",
        lambda **_kwargs: (_ for _ in ()).throw(cancel_error),
    )
    monkeypatch.setattr(
        vibe,
        "_transcribe_local",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("cancellation error used fallback")
        ),
    )

    with pytest.raises(CancelCheckError) as exc_info:
        adapter.transcribe_batch(
            str(audio),
            model=str(model_dir),
            language="en",
            base_dir=tmp_path,
            execution_plan=plan,
        )

    assert exc_info.value is cancel_error


@pytest.mark.unit
@pytest.mark.parametrize(
    "invalid_vllm_artifact",
    (
        {
            "text": "[Error: private route failure]",
            "segments": [],
        },
        {"text": "", "segments": []},
        {"text": "malformed", "segments": "not-a-list"},
        ["not", "a", "mapping"],
    ),
)
def test_vibevoice_production_validates_attempt_before_declared_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    invalid_vllm_artifact: object,
) -> None:
    audio = tmp_path / "vibe.wav"
    audio.write_bytes(b"audio")
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
    adapter = spa.VibeVoiceAdapter()
    plan = _plan(
        adapter,
        model=str(model_dir),
        mode="production-v1",
    )
    attempts: list[str] = []

    def invalid_vllm(**_kwargs: Any) -> Any:
        attempts.append("vllm")
        return invalid_vllm_artifact

    def valid_local(**_kwargs: Any) -> dict[str, Any]:
        attempts.append("local")
        return {
            "text": "declared local fallback",
            "segments": [
                {
                    "start_seconds": 0.0,
                    "end_seconds": 1.0,
                    "Text": "declared local fallback",
                }
            ],
        }

    monkeypatch.setattr(
        vibe,
        "_transcribe_via_vllm_http",
        invalid_vllm,
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
    monkeypatch.setattr(vibe, "_transcribe_local", valid_local)

    artifact = adapter.transcribe_batch(
        str(audio),
        model=str(model_dir),
        language="en",
        base_dir=tmp_path,
        execution_plan=plan,
    )

    assert attempts == ["vllm", "local"]
    assert artifact["text"] == "declared local fallback"
    assert artifact["actual_execution"]["route_id"] == "local-2"


@pytest.mark.unit
def test_vibevoice_neutral_invalid_attempt_never_uses_undeclared_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    audio = tmp_path / "vibe.wav"
    audio.write_bytes(b"audio")
    model_dir = tmp_path / "vibe-model"
    model_dir.mkdir()
    monkeypatch.setattr(
        vibe,
        "_resolve_settings",
        lambda: _vibe_settings(
            model_dir,
            vllm_enabled=True,
            local_enabled=False,
        ),
    )
    adapter = spa.VibeVoiceAdapter()
    plan = _plan(adapter, model=str(model_dir))
    monkeypatch.setattr(
        vibe,
        "_transcribe_via_vllm_http",
        lambda **_kwargs: {
            "text": "[No transcription produced]",
            "segments": [],
        },
    )
    monkeypatch.setattr(
        vibe,
        "_transcribe_local",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("neutral plan used undeclared local route")
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

    assert str(exc_info.value) == "Planned STT transcription failed"


@pytest.mark.unit
def test_vibevoice_production_runtime_semantic_loss_is_retained(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    audio = tmp_path / "vibe.wav"
    audio.write_bytes(b"audio")
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
    adapter = spa.VibeVoiceAdapter()
    plan = _plan(
        adapter,
        model=str(model_dir),
        language="en",
        hotwords=("private-hotword",),
        mode="production-v1",
    )
    calls: list[dict[str, Any]] = []

    class Model:
        def transcribe(
            self,
            _audio: object,
            **kwargs: Any,
        ) -> object:
            calls.append(dict(kwargs))
            if "language" in kwargs or kwargs.get("hotwords"):
                raise TypeError("optional semantics unsupported")
            return {"text": "fallback semantics"}

    monkeypatch.setattr(
        vibe,
        "_load_audio",
        lambda *_args, **_kwargs: (
            object(),
            16000,
            1.0,
        ),
    )
    monkeypatch.setattr(
        vibe,
        "_load_local_components",
        lambda _settings: (object(), Model(), "cpu"),
    )

    artifact = adapter.transcribe_batch(
        str(audio),
        model=str(model_dir),
        language="en",
        hotwords=("private-hotword",),
        base_dir=tmp_path,
        execution_plan=plan,
    )

    assert len(calls) == 2
    assert calls[0]["language"] == "en"
    assert calls[0]["hotwords"] == ["private-hotword"]
    assert "language" not in calls[1]
    assert "hotwords" not in calls[1]
    assert artifact["execution_mismatch"] == [
        "hotwords",
        "language",
    ]


@pytest.mark.unit
def test_vibevoice_neutral_runtime_semantic_loss_is_retained(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    audio = tmp_path / "vibe.wav"
    audio.write_bytes(b"audio")
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
    adapter = spa.VibeVoiceAdapter()
    plan = _plan(
        adapter,
        model=str(model_dir),
        language="en",
    )

    class Model:
        def transcribe(
            self,
            _audio: object,
            **kwargs: Any,
        ) -> object:
            if "language" in kwargs:
                raise TypeError("language unsupported")
            return {"text": "fallback semantics"}

    monkeypatch.setattr(
        vibe,
        "_load_audio",
        lambda *_args, **_kwargs: (
            object(),
            16000,
            1.0,
        ),
    )
    monkeypatch.setattr(
        vibe,
        "_load_local_components",
        lambda _settings: (object(), Model(), "cpu"),
    )

    artifact = adapter.transcribe_batch(
        str(audio),
        model=str(model_dir),
        language="en",
        base_dir=tmp_path,
        execution_plan=plan,
    )

    assert artifact["execution_mismatch"] == ["language"]


@pytest.mark.unit
def test_network_request_models_are_bound_directly_to_route_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"audio")

    monkeypatch.setattr(
        qwen3,
        "_resolve_settings",
        lambda: {
            "enabled": True,
            "backend": "vllm",
            "vllm_base_url": "http://localhost:8000",
            "model_path": "Qwen/Qwen3-ASR-1.7B",
            "sample_rate": 16000,
        },
    )
    qwen_adapter = spa.Qwen3ASRAdapter()
    qwen_plan = _plan(
        qwen_adapter,
        model="Qwen/Qwen3-ASR-1.7B",
    )
    qwen_mutated = replace(
        qwen_plan,
        runtime_settings=tuple(
            (
                key,
                "different-safe-model"
                if key == "request_model"
                else value,
            )
            for key, value in qwen_plan.runtime_settings
        ),
    )
    monkeypatch.setattr(
        qwen3,
        "_transcribe_vllm_http",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("mutated Qwen request was sent")
        ),
    )
    with pytest.raises(spa.STTExecutionPlanError):
        qwen_adapter.transcribe_batch(
            str(audio),
            model="Qwen/Qwen3-ASR-1.7B",
            language="en",
            base_dir=tmp_path,
            execution_plan=qwen_mutated,
        )

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
    vibe_adapter = spa.VibeVoiceAdapter()
    vibe_plan = _plan(
        vibe_adapter,
        model=str(model_dir),
    )
    vibe_mutated = replace(
        vibe_plan,
        runtime_settings=tuple(
            (
                key,
                "different-safe-model"
                if key == "vllm_model_id"
                else value,
            )
            for key, value in vibe_plan.runtime_settings
        ),
    )
    monkeypatch.setattr(
        vibe,
        "_transcribe_via_vllm_http",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("mutated VibeVoice request was sent")
        ),
    )
    with pytest.raises(spa.STTExecutionPlanError):
        vibe_adapter.transcribe_batch(
            str(audio),
            model=str(model_dir),
            language="en",
            base_dir=tmp_path,
            execution_plan=vibe_mutated,
        )


@pytest.mark.unit
def test_planned_remote_request_models_are_trimmed_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        qwen3,
        "_resolve_settings",
        lambda: {
            "enabled": True,
            "backend": "vllm",
            "vllm_base_url": "http://localhost:8000",
            "model_path": "Qwen/Qwen3-ASR-1.7B",
            "sample_rate": 16000,
        },
    )
    qwen_plan = _plan(
        spa.Qwen3ASRAdapter(),
        model="  Qwen/Qwen3-ASR-1.7B  ",
    )
    assert (
        qwen_plan.runtime_values()["request_model"]
        == qwen_plan.descriptor.primary_route.model_label
        == "Qwen/Qwen3-ASR-1.7B"
    )

    model_dir = tmp_path / "vibe-model"
    model_dir.mkdir()
    vibe_settings = _vibe_settings(
        model_dir,
        vllm_enabled=True,
    )
    vibe_settings["vllm_model_id"] = (
        "  microsoft/VibeVoice-ASR-HTTP  "
    )
    monkeypatch.setattr(
        vibe,
        "_resolve_settings",
        lambda: dict(vibe_settings),
    )
    vibe_plan = _plan(
        spa.VibeVoiceAdapter(),
        model=str(model_dir),
    )
    assert (
        vibe_plan.runtime_values()["vllm_model_id"]
        == vibe_plan.descriptor.primary_route.model_label
        == "microsoft/VibeVoice-ASR-HTTP"
    )

    config = _external_config()
    config.model = "  whisper-large  "
    monkeypatch.setattr(
        external,
        "load_external_provider_config",
        lambda _name="default": config,
    )
    external_plan = _plan(
        spa.ExternalAdapter(),
        model="external:custom",
    )
    assert (
        external_plan.runtime_values()["external_model"]
        == external_plan.descriptor.primary_route.model_label
        == "whisper-large"
    )


@pytest.mark.unit
def test_typed_outcome_runtime_mismatches_are_bounded_safe_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        qwen3,
        "_resolve_settings",
        lambda: {
            "enabled": True,
            "backend": "vllm",
            "vllm_base_url": "http://localhost:8000",
            "model_path": "Qwen/Qwen3-ASR-1.7B",
            "sample_rate": 16000,
        },
    )
    plan = _plan(
        spa.Qwen3ASRAdapter(),
        model="Qwen/Qwen3-ASR-1.7B",
    )
    actual = spa.actual_execution_from_route(
        plan.descriptor.primary_route,
        device=None,
    )
    artifact = {"text": "ok", "segments": []}

    default_outcome = spa.SttTranscriptionOutcome(
        artifact=artifact,
        actual_execution=actual,
    )
    assert default_outcome.runtime_mismatches == ()

    safe_outcome = spa.SttTranscriptionOutcome(
        artifact=artifact,
        actual_execution=actual,
        runtime_mismatches=("hotwords", "language"),
    )
    assert safe_outcome.runtime_mismatches == (
        "hotwords",
        "language",
    )

    secret = "private-api-key"
    with pytest.raises(ValueError) as exc_info:
        spa.SttTranscriptionOutcome(
            artifact=artifact,
            actual_execution=actual,
            runtime_mismatches=(secret,),
        )
    assert secret not in str(exc_info.value)

    with pytest.raises(ValueError, match="runtime_mismatches are invalid"):
        spa.SttTranscriptionOutcome(
            artifact=artifact,
            actual_execution=actual,
            runtime_mismatches=(["not-hashable"],),  # type: ignore[arg-type]
        )

    with pytest.raises(
        spa.STTExecutionPlanError,
        match="runtime mismatches are invalid",
    ):
        spa.finalize_stt_artifact(
            artifact,
            plan=plan,
            actual=actual,
            runtime_mismatches=(["not-hashable"],),  # type: ignore[arg-type]
        )


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
def test_legacy_external_endpoint_keeps_urljoin_behavior() -> None:
    assert external._resolve_transcription_endpoint(
        "https://example.com/prefix?legacy=value"
    ) == "https://example.com/v1/audio/transcriptions"


@pytest.mark.unit
def test_external_plan_freezes_config_and_returns_safe_typed_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    audio = tmp_path / "external.wav"
    audio.write_bytes(b"audio")
    config = _external_config()
    monkeypatch.setattr(http_client, "aiohttp", object())
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

    assert plan.descriptor.requested_model_label == "external:custom"
    assert plan.descriptor.resolved_model_label == "whisper-large"
    assert route.model_label == "whisper-large"
    assert route.backend == "openai_compatible"
    assert route.source == "external_http"
    assert route.transport == "aiohttp"
    assert route.audio_egress is spa.SttAudioEgress.REMOTE
    assert route.endpoint_id == spa._normalize_audio_endpoint(
        "https://api.example.com/v1/audio/transcriptions"
    )[2]
    assert plan.runtime_values()["external_transport"] == "aiohttp"
    assert plan.descriptor.dependency_distributions == ("aiohttp",)
    assert {
        "tldw_Server_API.app.core.Security.egress",
        "tldw_Server_API.app.core.http_client",
        "tldw_Server_API.app.core.stt_observability_context",
    }.issubset(plan.descriptor.source_modules)
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
        captured["transport"] = _kwargs.get("transport")
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
        "transport": "aiohttp",
    }
    assert artifact["text"] == "planned external"
    assert artifact["actual_execution"]["route_id"] == route.route_id
    assert artifact["actual_execution"]["endpoint_id"] == (
        route.endpoint_id
    )
    assert artifact["actual_execution"]["model_label"] == (
        "whisper-large"
    )
    assert artifact["actual_execution"]["transport"] == "aiohttp"
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
        "Planned STT transcription failed"
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
async def test_external_legacy_network_call_keeps_default_redirect_behavior(
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
    assert "transport" not in captured


@pytest.mark.unit
@pytest.mark.asyncio
async def test_external_planned_network_call_disables_redirects(
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
    runtime = plan.runtime_values()
    frozen = external.ExternalProviderConfig(
        base_url=str(runtime["external_base_url"]),
        api_key=str(runtime["external_api_key"]),
        model=str(runtime["external_model"]),
        timeout=float(runtime["external_timeout"]),
        max_retries=int(runtime["external_max_retries"]),
        verify_ssl=bool(runtime["external_verify_ssl"]),
        custom_headers=dict(
            zip(
                runtime["external_header_names"],
                runtime["external_header_values"],
            )
        ),
        response_format=str(runtime["external_response_format"]),
        temperature=float(runtime["external_temperature"]),
        language=str(runtime["external_language"]),
    )
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
        config=frozen,
        base_dir=tmp_path,
        execution_plan=plan,
        transport=str(runtime["external_transport"]),
    )

    assert result == "redirect-safe"
    assert captured["allow_redirects"] is False
    assert captured["transport"] == runtime["external_transport"]
    assert captured["data"]["model"] == "whisper-large"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_external_planned_transport_fails_closed_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    audio = tmp_path / "external.wav"
    audio.write_bytes(b"audio")
    config = _external_config()
    monkeypatch.setattr(http_client, "aiohttp", object())
    monkeypatch.setattr(
        external,
        "load_external_provider_config",
        lambda _name="default": config,
    )
    plan = _plan(
        spa.ExternalAdapter(),
        model="external:custom",
    )
    runtime = plan.runtime_values()
    frozen = external.ExternalProviderConfig(
        base_url=str(runtime["external_base_url"]),
        api_key=str(runtime["external_api_key"]),
        model=str(runtime["external_model"]),
        timeout=float(runtime["external_timeout"]),
        max_retries=int(runtime["external_max_retries"]),
        verify_ssl=bool(runtime["external_verify_ssl"]),
        custom_headers=dict(
            zip(
                runtime["external_header_names"],
                runtime["external_header_values"],
            )
        ),
        response_format=str(runtime["external_response_format"]),
        temperature=float(runtime["external_temperature"]),
        language=str(runtime["external_language"]),
    )
    monkeypatch.setattr(http_client, "aiohttp", None)
    monkeypatch.setattr(
        external,
        "afetch",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("unavailable frozen transport was used")
        ),
    )

    with pytest.raises(
        spa.STTExecutionPlanError,
        match="transport",
    ):
        await external.transcribe_with_external_provider_async(
            audio,
            config=frozen,
            base_dir=tmp_path,
            execution_plan=plan,
            transport=str(runtime["external_transport"]),
        )


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
