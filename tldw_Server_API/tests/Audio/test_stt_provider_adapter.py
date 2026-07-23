import builtins
import importlib
import importlib.machinery
import inspect
import pickle
import subprocess
import sys
import types
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path
from typing import Any

import pytest

# Stub heavyweight audio deps before adapter imports to avoid local
# ctranslate2/torch dynamic-load aborts in constrained test environments.
if "torch" not in sys.modules:
    _fake_torch = types.ModuleType("torch")
    _fake_torch.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)
    _fake_torch.Tensor = object
    _fake_torch.float16 = "float16"
    _fake_torch.float32 = "float32"
    _fake_torch.bfloat16 = "bfloat16"
    _fake_torch.nn = types.SimpleNamespace(Module=object)
    _fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    sys.modules["torch"] = _fake_torch

if "faster_whisper" not in sys.modules:
    _fake_fw = types.ModuleType("faster_whisper")
    _fake_fw.__spec__ = importlib.machinery.ModuleSpec("faster_whisper", loader=None)

    class _StubWhisperModel:
        def __init__(self, *args, **kwargs):
            pass

    _fake_fw.WhisperModel = _StubWhisperModel
    _fake_fw.BatchedInferencePipeline = _StubWhisperModel
    sys.modules["faster_whisper"] = _fake_fw

if "transformers" not in sys.modules:
    _fake_tf = types.ModuleType("transformers")
    _fake_tf.__spec__ = importlib.machinery.ModuleSpec("transformers", loader=None)

    class _StubProcessor:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class _StubModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    _fake_tf.AutoProcessor = _StubProcessor
    _fake_tf.Qwen2AudioForConditionalGeneration = _StubModel
    sys.modules["transformers"] = _fake_tf


_EXCEPTIONS_MODULE = "tldw_Server_API.app.core.exceptions"
_AUDIO_LIB_MODULE = "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib"


def _install_py39_compat_stubs() -> None:
    exceptions_stub = types.ModuleType(_EXCEPTIONS_MODULE)
    exceptions_stub.BadRequestError = type("BadRequestError", (Exception,), {})
    exceptions_stub.CancelCheckError = type("CancelCheckError", (Exception,), {})
    exceptions_stub.TranscriptionCancelled = type("TranscriptionCancelled", (Exception,), {})
    exceptions_stub.InvalidStoragePathError = type("InvalidStoragePathError", (Exception,), {})
    exceptions_stub.StorageUnavailableError = type("StorageUnavailableError", (Exception,), {})
    exceptions_stub.NetworkError = type("NetworkError", (Exception,), {})
    exceptions_stub.RetryExhaustedError = type("RetryExhaustedError", (Exception,), {})
    exceptions_stub.__file__ = __file__

    def _exception_getattr(name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        return type(str(name), (Exception,), {})

    exceptions_stub.__getattr__ = _exception_getattr  # type: ignore[assignment]
    sys.modules[_EXCEPTIONS_MODULE] = exceptions_stub

    audio_lib_stub = types.ModuleType(_AUDIO_LIB_MODULE)
    audio_lib_stub.__file__ = __file__

    def _parse_transcription_model(model_name: str):
        normalized = (model_name or "").strip()
        lowered = normalized.lower()
        if lowered.startswith("parakeet"):
            return "parakeet", normalized, None
        if lowered.startswith("qwen2audio"):
            return "qwen2audio", normalized, None
        if lowered.startswith("vibevoice"):
            return "vibevoice", normalized, None
        if lowered.startswith("external:"):
            return "external", normalized, None
        return "whisper", normalized, None

    def _speech_to_text(*args, **kwargs):
        return [], kwargs.get("selected_source_lang")

    audio_lib_stub.parse_transcription_model = _parse_transcription_model
    audio_lib_stub.speech_to_text = _speech_to_text
    audio_lib_stub.strip_whisper_metadata_header = lambda segments: segments
    sys.modules[_AUDIO_LIB_MODULE] = audio_lib_stub


def _import_module():
    module_name = "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_provider_adapter"
    try:
        # Local import so tests don't break when heavy STT deps are absent.
        return importlib.import_module(module_name)
    except TypeError as exc:
        # Python 3.9 cannot import some project modules that use PEP-604
        # runtime unions. Inject a minimal exceptions stub for STT tests.
        if "unsupported operand type(s) for |" not in str(exc):
            raise
        _install_py39_compat_stubs()
        sys.modules.pop(module_name, None)
        return importlib.import_module(module_name)


@pytest.mark.unit
def test_default_provider_name_uses_stt_settings(monkeypatch):
    spa = _import_module()

    # Simulate STT-Settings with both keys present; default_transcriber should win.
    def fake_get_stt_config():
        return {
            "default_stt_provider": "parakeet",
            "default_transcriber": "faster_whisper",
        }

    monkeypatch.setattr(spa, "get_stt_config", fake_get_stt_config)

    registry = spa.SttProviderRegistry()
    assert registry.get_default_provider_name() == "faster-whisper"


@pytest.mark.unit
def test_default_provider_name_falls_back_to_stt_provider(monkeypatch):
    spa = _import_module()

    def fake_get_stt_config():

        return {
            "default_stt_provider": "parakeet",
            # No default_transcriber key
        }

    monkeypatch.setattr(spa, "get_stt_config", fake_get_stt_config)

    registry = spa.SttProviderRegistry()
    assert registry.get_default_provider_name() == "parakeet"


def _make_execution_plan(
    spa,
    *,
    provider="faster-whisper",
    model_label="tiny",
    task="transcribe",
    language="en",
    decoding_settings=(("configuration_id", "neutral-1"),),
    runtime_settings=(),
):
    route = spa.SttExecutionRoute(
        route_id="neutral-1",
        provider=provider,
        model_label=model_label,
        artifact_id=f"sha256:{'a' * 64}",
        identity_resolved=True,
        backend="ctranslate2",
        source="local",
        audio_egress=spa.SttAudioEgress.NONE,
        endpoint_id=None,
        device="cpu",
        compute_type="int8",
        dtype=None,
        decoding_ids=tuple(key for key, _ in decoding_settings),
        local_model_available=True,
        would_download=False,
    )
    descriptor = spa.SttExecutionDescriptor(
        requested_provider=provider,
        requested_model_label=model_label,
        resolved_provider=provider,
        resolved_model_label=model_label,
        routes=(route,),
        honors_task=True,
        honors_language=True,
        honors_prompt_absence=True,
        honors_hotword_absence=True,
        honors_diarization=True,
        honors_word_timestamps=True,
        decoding_settings=decoding_settings,
        source_modules=(
            "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_provider_adapter",
        ),
        dependency_distributions=("faster-whisper",),
    )
    return spa.SttBatchExecutionPlan(
        descriptor=descriptor,
        task=task,
        language=language,
        runtime_settings=runtime_settings,
    )


@pytest.mark.unit
def test_execution_plan_is_frozen_and_pickleable():
    spa = _import_module()
    plan = _make_execution_plan(spa)

    assert pickle.loads(pickle.dumps(plan)) == plan
    with pytest.raises(FrozenInstanceError):
        plan.task = "translate"
    with pytest.raises(FrozenInstanceError):
        plan.descriptor.resolved_provider = "parakeet"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field_name", "settings"),
    [
        (
            "decoding",
            (("prompt_present", False), ("hotword_count", 0)),
        ),
        (
            "decoding",
            (("configuration_id", "one"), ("configuration_id", "two")),
        ),
        ("runtime", (("token", "one"), ("api_key", "two"))),
        ("runtime", (("api_key", "one"), ("api_key", "two"))),
    ],
)
def test_execution_plan_rejects_duplicate_or_noncanonical_setting_keys(
    field_name,
    settings,
):
    spa = _import_module()

    with pytest.raises(ValueError):
        if field_name == "decoding":
            _make_execution_plan(spa, decoding_settings=settings)
        else:
            _make_execution_plan(spa, runtime_settings=settings)


@pytest.mark.unit
def test_execution_plan_keeps_runtime_secrets_out_of_repr_and_safe_descriptor():
    spa = _import_module()
    plan = _make_execution_plan(
        spa,
        runtime_settings=(
            ("api_key", "secret-token"),
            ("endpoint_url", "https://secret.example"),
        ),
    )
    plan = replace(
        plan,
        prompt="secret prompt",
        hotwords=("secret hotword",),
    )

    rendered = repr(plan)
    safe = plan.descriptor.as_safe_dict()

    assert "secret-token" not in rendered
    assert "secret.example" not in rendered
    assert "secret prompt" not in rendered
    assert "secret hotword" not in rendered
    assert "secret-token" not in repr(safe)
    assert "secret.example" not in repr(safe)


@pytest.mark.unit
def test_safe_descriptor_contains_only_declared_fields():
    spa = _import_module()
    plan = _make_execution_plan(spa)

    safe = plan.descriptor.as_safe_dict()

    assert set(safe) == {field.name for field in fields(spa.SttExecutionDescriptor)}
    assert set(safe["routes"][0]) == {
        field.name for field in fields(spa.SttExecutionRoute)
    }
    assert safe["routes"][0]["audio_egress"] == "none"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("target", "changes"),
    [
        ("route", {"model_label": "file:/etc/passwd"}),
        ("route", {"model_label": "/opt/models/private"}),
        ("route", {"model_label": r"C:\models\private"}),
        ("route", {"model_label": "vendor/model?access_token=secret"}),
        ("route", {"model_label": "vendor/model\n"}),
        ("route", {"backend": "https://user:secret@example.test/runtime"}),
        ("route", {"source": "Bearer secret"}),
        ("route", {"device": "api_key"}),
        ("actual", {"source": r"\\server\share\credentials"}),
        ("actual", {"backend": "file:/etc/passwd"}),
        ("descriptor", {"requested_provider": "user@example.test"}),
        (
            "descriptor",
            {
                "decoding_settings": (
                    ("configuration_id", "Authorization: Bearer secret"),
                )
            },
        ),
    ],
)
def test_serialized_execution_values_reject_hostile_strings(target, changes):
    spa = _import_module()
    plan = _make_execution_plan(spa)
    route = plan.descriptor.primary_route

    with pytest.raises(ValueError):
        if target == "route":
            replace(route, **changes)
        elif target == "descriptor":
            replace(plan.descriptor, **changes)
        else:
            actual_values = {
                "route_id": route.route_id,
                "provider": route.provider,
                "model_label": route.model_label,
                "artifact_id": route.artifact_id,
                "backend": route.backend,
                "audio_egress": route.audio_egress,
                "endpoint_id": route.endpoint_id,
                "source": route.source,
                "device": route.device,
                "compute_type": route.compute_type,
                "dtype": route.dtype,
                "decoding_ids": route.decoding_ids,
            }
            actual_values.update(changes)
            spa.SttActualExecution(**actual_values)


@pytest.mark.unit
def test_safe_descriptor_allows_fixed_language_contract():
    spa = _import_module()
    plan = _make_execution_plan(
        spa,
        decoding_settings=(("language_contract", "fixed:en"),),
    )

    assert plan.descriptor.as_safe_dict()["decoding_settings"] == [
        ["language_contract", "fixed:en"]
    ]


@pytest.mark.unit
@pytest.mark.parametrize(
    "decoding_settings",
    [
        (("prompt", "transcript"),),
        (("hotwords", "alpha"),),
        (("headers", "alpha"),),
        (("credential", "alpha"),),
        (("language_contract", "transcript"),),
        (("prompt_present", "alpha"),),
        (("hotword_count", "sk_proj_123"),),
        (("hotword_count", True),),
        (("hotword_count", -1),),
        (("configuration_id", "sk_proj_123"),),
    ],
)
def test_descriptor_rejects_decoding_values_outside_v1_schema(
    decoding_settings,
):
    spa = _import_module()

    with pytest.raises(ValueError):
        _make_execution_plan(
            spa,
            decoding_settings=decoding_settings,
        )


@pytest.mark.unit
def test_descriptor_accepts_all_v1_decoding_settings():
    spa = _import_module()
    decoding_settings = (
        ("configuration_id", "neutral-1"),
        ("hotword_count", 0),
        ("language_contract", "fixed:en"),
        ("prompt_present", False),
    )
    plan = _make_execution_plan(
        spa,
        decoding_settings=decoding_settings,
    )

    assert plan.descriptor.as_safe_dict()["decoding_settings"] == [
        [key, value] for key, value in decoding_settings
    ]


@pytest.mark.unit
@pytest.mark.parametrize("target", ["route", "actual"])
def test_route_and_actual_reject_decoding_ids_outside_v1_schema(target):
    spa = _import_module()
    plan = _make_execution_plan(spa)
    route = plan.descriptor.primary_route

    with pytest.raises(ValueError):
        if target == "route":
            replace(route, decoding_ids=("prompt",))
        else:
            spa.SttActualExecution(
                route_id=route.route_id,
                provider=route.provider,
                model_label=route.model_label,
                artifact_id=route.artifact_id,
                backend=route.backend,
                audio_egress=route.audio_egress,
                endpoint_id=route.endpoint_id,
                source=route.source,
                device=route.device,
                compute_type=route.compute_type,
                dtype=route.dtype,
                decoding_ids=("prompt",),
            )


@pytest.mark.unit
def test_safe_route_allows_opaque_endpoint_hash():
    spa = _import_module()
    plan = _make_execution_plan(spa)
    endpoint_id = f"sha256:{'b' * 64}"

    route = replace(
        plan.descriptor.primary_route,
        audio_egress=spa.SttAudioEgress.REMOTE,
        endpoint_id=endpoint_id,
    )

    assert route.as_safe_dict()["endpoint_id"] == endpoint_id


@pytest.mark.unit
@pytest.mark.parametrize(
    "module_order",
    [
        (
            "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_execution_contract",
            "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_provider_adapter",
        ),
        (
            "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_provider_adapter",
            "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_execution_contract",
        ),
    ],
)
def test_execution_contract_and_adapter_import_in_either_order(module_order):
    command = "; ".join(f"import {module}" for module in module_order)

    result = subprocess.run(
        [sys.executable, "-c", command],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.unit
@pytest.mark.parametrize(
    "module_order",
    [
        (
            "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_execution_contract",
            "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib",
            "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo",
            "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_ONNX",
            "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX",
            "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_provider_adapter",
        ),
        (
            "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_provider_adapter",
            "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX",
            "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_ONNX",
            "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo",
            "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib",
            "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_execution_contract",
        ),
    ],
)
def test_local_execution_modules_import_in_either_order(module_order):
    command = "; ".join(f"import {module}" for module in module_order)

    result = subprocess.run(
        [sys.executable, "-c", command],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.unit
def test_dependency_neutral_contract_does_not_import_adapter():
    contract_name = (
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio."
        "stt_execution_contract"
    )
    adapter_name = (
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio."
        "stt_provider_adapter"
    )
    command = (
        f"import sys; import {contract_name}; "
        f"assert {adapter_name!r} not in sys.modules"
    )

    result = subprocess.run(
        [sys.executable, "-c", command],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.unit
def test_neutral_sentinel_predicate_does_not_import_runtime_or_adapter():
    package = "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio"
    library_name = f"{package}.Audio_Transcription_Lib"
    adapter_name = f"{package}.stt_provider_adapter"
    command = (
        f"import sys; from {package} import stt_execution_contract as contract; "
        "assert contract.is_planned_stt_sentinel('[Error: private]'); "
        f"assert {library_name!r} not in sys.modules; "
        f"assert {adapter_name!r} not in sys.modules"
    )

    result = subprocess.run(
        [sys.executable, "-c", command],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.unit
@pytest.mark.parametrize(
    "provider_module",
    [
        "Audio_Transcription_Lib",
        "Audio_Transcription_Nemo",
        "Audio_Transcription_Parakeet_ONNX",
        "Audio_Transcription_Parakeet_MLX",
        "Audio_Transcription_Qwen3ASR",
        "Audio_Transcription_VibeVoice",
        "Audio_Transcription_External_Provider",
    ],
)
def test_provider_runtime_module_does_not_import_adapter(provider_module):
    package = "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio"
    adapter_name = f"{package}.stt_provider_adapter"
    command = (
        f"import sys; import {package}.{provider_module}; "
        f"assert {adapter_name!r} not in sys.modules"
    )

    result = subprocess.run(
        [sys.executable, "-c", command],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.unit
def test_get_adapter_strict_rejects_unknown_but_legacy_lookup_still_falls_back():
    spa = _import_module()
    registry = spa.SttProviderRegistry()

    with pytest.raises(spa.STTExecutionPlanError):
        registry.get_adapter_strict("unknown-provider")
    assert registry.get_adapter("unknown-provider").name.value == "faster-whisper"


@pytest.mark.unit
def test_default_planner_fails_closed_for_unimplemented_provider():
    spa = _import_module()

    with pytest.raises(spa.STTExecutionUnsupportedError):
        spa.FasterWhisperAdapter().plan_batch_execution(
            model="tiny",
            language="en",
            task="transcribe",
            word_timestamps=False,
            prompt=None,
            hotwords=None,
            diarization=False,
            mode="neutral-v1",
        )


@pytest.mark.unit
def test_all_batch_adapters_keep_execution_plan_optional():
    spa = _import_module()

    for adapter_type in spa.SttProviderRegistry.DEFAULT_ADAPTERS.values():
        parameter = inspect.signature(adapter_type.transcribe_batch).parameters[
            "execution_plan"
        ]
        assert parameter.default is None


@pytest.mark.unit
def test_planned_provider_mismatch_fails_before_provider_helper(monkeypatch):
    spa = _import_module()
    plan = _make_execution_plan(spa, provider="parakeet")
    calls = []

    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib

    monkeypatch.setattr(
        atlib,
        "speech_to_text",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    with pytest.raises(spa.STTExecutionPlanError):
        spa.FasterWhisperAdapter().transcribe_batch(
            "not-opened.wav",
            model="tiny",
            language="en",
            execution_plan=plan,
        )

    assert calls == []


@pytest.mark.unit
@pytest.mark.parametrize("requested_model", [None, "tiny"])
def test_planned_model_must_match_primary_route_before_runtime_access(
    requested_model,
):
    spa = _import_module()
    plan = _make_execution_plan(spa)
    route = replace(plan.descriptor.primary_route, model_label="base")
    plan = replace(
        plan,
        descriptor=replace(plan.descriptor, routes=(route,)),
    )

    with pytest.raises(spa.STTExecutionPlanError) as exc_info:
        spa.FasterWhisperAdapter().transcribe_batch(
            "not-opened.wav",
            model=requested_model,
            language="en",
            execution_plan=plan,
        )

    assert exc_info.type is spa.STTExecutionPlanError


@pytest.mark.unit
@pytest.mark.parametrize(
    "call_overrides",
    [
        {"task": "translate"},
        {"language": "fr"},
        {"prompt": "different prompt"},
        {"hotwords": ("different",)},
        {"word_timestamps": True},
    ],
)
def test_planned_semantics_must_match_request_before_runtime_access(
    call_overrides,
):
    spa = _import_module()
    plan = _make_execution_plan(spa)
    call = {
        "model": "tiny",
        "language": "en",
        "execution_plan": plan,
        **call_overrides,
    }

    with pytest.raises(spa.STTExecutionPlanError):
        spa.FasterWhisperAdapter().transcribe_batch(
            "not-opened.wav",
            **call,
        )


@pytest.mark.unit
def test_shared_planned_runner_finalizes_typed_provider_outcome(monkeypatch):
    spa = _import_module()
    plan = _make_execution_plan(spa)
    route = plan.descriptor.primary_route
    actual = spa.SttActualExecution(
        route_id=route.route_id,
        provider=route.provider,
        model_label=route.model_label,
        artifact_id=route.artifact_id,
        backend=route.backend,
        audio_egress=route.audio_egress,
        endpoint_id=route.endpoint_id,
        source=route.source,
        device=route.device,
        compute_type=route.compute_type,
        dtype=route.dtype,
        decoding_ids=route.decoding_ids,
    )
    calls = []

    class PlannedFasterWhisperAdapter(spa.FasterWhisperAdapter):
        def _transcribe_planned_batch(
            self,
            audio_path,
            *,
            execution_plan,
            base_dir,
            cancel_check,
        ):
            calls.append(
                (audio_path, execution_plan, base_dir, cancel_check)
            )
            return spa.SttTranscriptionOutcome(
                artifact={
                    "text": "planned transcript",
                    "segments": [],
                    "language": "en",
                    "metadata": {"authorization": "Bearer secret"},
                },
                actual_execution=actual,
            )

    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib

    monkeypatch.setattr(
        atlib,
        "is_transcription_error_message",
        lambda text: False,
        raising=False,
    )
    adapter = PlannedFasterWhisperAdapter()
    artifact = adapter.transcribe_batch(
        "not-opened.wav",
        model="tiny",
        language="en",
        execution_plan=plan,
    )

    assert calls == [("not-opened.wav", plan, None, None)]
    assert artifact == {
        "text": "planned transcript",
        "segments": [],
        "language": "en",
        "actual_execution": actual.as_safe_dict(),
    }


@pytest.mark.unit
def test_shared_planned_runner_requires_typed_provider_outcome():
    spa = _import_module()
    plan = _make_execution_plan(spa)

    class UntypedFasterWhisperAdapter(spa.FasterWhisperAdapter):
        def _transcribe_planned_batch(
            self,
            audio_path,
            *,
            execution_plan,
            base_dir,
            cancel_check,
        ):
            return {"text": "untyped", "segments": []}

    with pytest.raises(spa.STTExecutionPlanError) as exc_info:
        UntypedFasterWhisperAdapter().transcribe_batch(
            "not-opened.wav",
            model="tiny",
            language="en",
            execution_plan=plan,
        )

    assert exc_info.type is spa.STTExecutionPlanError


@pytest.mark.unit
@pytest.mark.parametrize(
    "sentinel",
    [
        "[Error: {secret}]",
        "[No transcription produced]",
        "[No speech detected]",
    ],
)
def test_finalize_stt_artifact_sanitizes_recognized_error_sentinel(
    caplog,
    sentinel,
):
    spa = _import_module()
    plan = _make_execution_plan(spa)
    route = plan.descriptor.primary_route
    actual = spa.SttActualExecution(
        route_id=route.route_id,
        provider=route.provider,
        model_label=route.model_label,
        artifact_id=route.artifact_id,
        backend=route.backend,
        audio_egress=route.audio_egress,
        endpoint_id=route.endpoint_id,
        source=route.source,
        device=route.device,
        compute_type=route.compute_type,
        dtype=route.dtype,
        decoding_ids=route.decoding_ids,
    )
    secret = "secret-token /Users/alice/private-transcript.txt"
    raw_sentinel = sentinel.format(secret=secret)

    with pytest.raises(spa.STTTranscriptionError) as exc_info:
        spa.finalize_stt_artifact(
            {"text": raw_sentinel, "segments": []},
            plan=plan,
            actual=actual,
        )

    assert str(exc_info.value) == "Planned local STT transcription failed"
    assert secret not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert secret not in caplog.text
    assert raw_sentinel not in caplog.text


@pytest.mark.unit
def test_finalize_stt_artifact_replaces_hostile_actual_execution_metadata(
    monkeypatch,
):
    spa = _import_module()
    plan = _make_execution_plan(spa)
    route = plan.descriptor.primary_route
    actual = spa.SttActualExecution(
        route_id=route.route_id,
        provider=route.provider,
        model_label=route.model_label,
        artifact_id=route.artifact_id,
        backend=route.backend,
        audio_egress=route.audio_egress,
        endpoint_id=route.endpoint_id,
        source=route.source,
        device=route.device,
        compute_type=route.compute_type,
        dtype=route.dtype,
        decoding_ids=route.decoding_ids,
    )

    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib

    monkeypatch.setattr(
        atlib,
        "is_transcription_error_message",
        lambda text: False,
        raising=False,
    )
    artifact = {
        "text": "hello",
        "segments": [],
        "language": "en",
        "metadata": {
            "authorization": "Bearer secret",
            "actual_execution": {"provider": "attacker"},
        },
        "actual_execution": {
            "endpoint_url": "https://secret.example",
            "provider": "attacker",
        },
    }

    finalized = spa.finalize_stt_artifact(
        artifact,
        plan=plan,
        actual=actual,
    )

    assert "metadata" not in finalized
    assert finalized["actual_execution"] == actual.as_safe_dict()
    assert "secret" not in repr(finalized)
    assert set(finalized["actual_execution"]) == {
        field.name for field in fields(spa.SttActualExecution)
    }


@pytest.mark.unit
def test_finalize_stt_artifact_rejects_undeclared_actual_route(monkeypatch):
    spa = _import_module()
    plan = _make_execution_plan(spa)
    route = plan.descriptor.primary_route
    actual = spa.SttActualExecution(
        route_id="undeclared",
        provider=route.provider,
        model_label=route.model_label,
        artifact_id=route.artifact_id,
        backend=route.backend,
        audio_egress=route.audio_egress,
        endpoint_id=route.endpoint_id,
        source=route.source,
        device=route.device,
        compute_type=route.compute_type,
        dtype=route.dtype,
        decoding_ids=route.decoding_ids,
    )

    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib

    monkeypatch.setattr(
        atlib,
        "is_transcription_error_message",
        lambda text: False,
        raising=False,
    )

    with pytest.raises(spa.STTExecutionPlanError):
        spa.finalize_stt_artifact(
            {"text": "hello", "segments": []},
            plan=plan,
            actual=actual,
        )


@pytest.mark.unit
def test_finalize_stt_artifact_matches_only_declared_non_null_route_fields(
    monkeypatch,
):
    spa = _import_module()
    plan = _make_execution_plan(spa)
    route = replace(
        plan.descriptor.primary_route,
        artifact_id=None,
        identity_resolved=False,
        device=None,
        compute_type=None,
    )
    plan = replace(
        plan,
        descriptor=replace(plan.descriptor, routes=(route,)),
    )
    actual = spa.SttActualExecution(
        route_id=route.route_id,
        provider=route.provider,
        model_label=route.model_label,
        artifact_id=f"sha256:{'b' * 64}",
        backend=route.backend,
        audio_egress=route.audio_egress,
        endpoint_id=route.endpoint_id,
        source=route.source,
        device="cpu",
        compute_type="int8",
        dtype=route.dtype,
        decoding_ids=route.decoding_ids,
    )

    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib

    monkeypatch.setattr(
        atlib,
        "is_transcription_error_message",
        lambda text: False,
        raising=False,
    )

    finalized = spa.finalize_stt_artifact(
        {"text": "hello", "segments": []},
        plan=plan,
        actual=actual,
    )

    assert finalized["actual_execution"] == actual.as_safe_dict()


@pytest.mark.unit
def test_get_adapter_unknown_provider_falls_back_to_faster_whisper(monkeypatch):
    spa = _import_module()

    def fake_get_stt_config():

        return {
            "default_stt_provider": "parakeet",
        }

    monkeypatch.setattr(spa, "get_stt_config", fake_get_stt_config)

    registry = spa.SttProviderRegistry()
    adapter = registry.get_adapter("unknown-provider")
    assert adapter.name.value == "faster-whisper"


@pytest.mark.unit
def test_resolve_provider_for_model_uses_parser(monkeypatch):
    spa = _import_module()

    # Provide a simple, deterministic parser implementation so we don't depend
    # on the exact behavior of Audio_Transcription_Lib here.
    def fake_parse_transcription_model(model_name: str):
        if model_name.startswith("parakeet"):
            return ("parakeet", "parakeet", "onnx")
        if model_name.startswith("qwen2audio"):
            return ("qwen2audio", model_name, None)
        if model_name.startswith("vibevoice"):
            return ("vibevoice", model_name, None)
        return ("whisper", model_name, None)

    monkeypatch.setattr(spa, "parse_transcription_model", fake_parse_transcription_model)

    registry = spa.SttProviderRegistry()

    provider, model, variant = registry.resolve_provider_for_model("parakeet-onnx")
    assert provider == "parakeet"
    assert model == "parakeet"
    assert variant == "onnx"

    provider, model, variant = registry.resolve_provider_for_model("qwen2audio-test")
    assert provider == "qwen2audio"
    assert model == "qwen2audio-test"
    assert variant is None

    provider, model, variant = registry.resolve_provider_for_model("vibevoice-asr")
    assert provider == "vibevoice"
    assert model == "vibevoice-asr"
    assert variant is None

    # Whisper-family models should normalize to faster-whisper.
    provider, model, variant = registry.resolve_provider_for_model("whisper-1")
    assert provider == "faster-whisper"
    assert model == "whisper-1"
    assert variant is None


@pytest.mark.unit
def test_resolve_provider_for_model_keeps_parakeet_when_parser_import_fails(monkeypatch):
    spa = _import_module()
    registry = spa.SttProviderRegistry()
    real_import = builtins.__import__

    monkeypatch.delitem(sys.modules, _AUDIO_LIB_MODULE, raising=False)

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 1 and name == "Audio_Transcription_Lib":
            raise ImportError("simulated parser import failure")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    provider, model, variant = registry.resolve_provider_for_model("parakeet-onnx")

    assert provider == "parakeet"
    assert model == "parakeet"
    assert variant == "onnx"


@pytest.mark.unit
def test_resolve_provider_for_model_allows_external_prefix():
    spa = _import_module()

    registry = spa.SttProviderRegistry()
    provider, model, variant = registry.resolve_provider_for_model("external:custom")
    assert provider == "external"
    assert model == "external:custom"
    assert variant is None


@pytest.mark.unit
def test_resolve_provider_for_model_uses_config_default(monkeypatch):
    spa = _import_module()

    def fake_get_stt_config():
        return {
            "default_transcriber": "parakeet",
            "nemo_model_variant": "mlx",
        }

    monkeypatch.setattr(spa, "get_stt_config", fake_get_stt_config)

    registry = spa.SttProviderRegistry()
    provider, model, variant = registry.resolve_provider_for_model(None)
    assert provider == "parakeet"
    assert model == "parakeet-mlx"
    assert variant == "mlx"


@pytest.mark.unit
def test_resolve_provider_for_model_uses_vibevoice_defaults(monkeypatch):
    spa = _import_module()

    def fake_get_stt_config():
        return {
            "default_transcriber": "vibevoice-asr",
            "vibevoice_model_id": "microsoft/VibeVoice-ASR",
        }

    monkeypatch.setattr(spa, "get_stt_config", fake_get_stt_config)

    registry = spa.SttProviderRegistry()
    provider, model, variant = registry.resolve_provider_for_model(None)
    assert provider == "vibevoice"
    assert model == "microsoft/VibeVoice-ASR"
    assert variant is None


@pytest.mark.unit
def test_resolve_default_transcription_model_uses_whisper_fallback(monkeypatch):
    spa = _import_module()

    def fake_get_stt_config():
        return {"default_transcriber": "faster-whisper"}

    monkeypatch.setattr(spa, "get_stt_config", fake_get_stt_config)

    default_model = spa.resolve_default_transcription_model("whisper-1")
    assert default_model == "whisper-1"


@pytest.mark.unit
def test_resolve_default_transcription_model_prefers_batch_default(monkeypatch):
    spa = _import_module()

    def fake_get_stt_config():
        return {
            "default_batch_transcription_model": "parakeet-onnx",
            "default_transcriber": "faster-whisper",
            "default_stt_provider": "faster-whisper",
        }

    monkeypatch.setattr(spa, "get_stt_config", fake_get_stt_config)

    default_model = spa.resolve_default_transcription_model("whisper-1")
    assert default_model == "parakeet-onnx"


@pytest.mark.unit
def test_resolve_default_transcription_model_uses_canonical_parakeet_onnx_alias(monkeypatch):
    spa = _import_module()

    def fake_get_stt_config():
        return {
            "default_transcriber": "parakeet",
            "default_stt_provider": "parakeet",
            "nemo_model_variant": "onnx",
        }

    monkeypatch.setattr(spa, "get_stt_config", fake_get_stt_config)

    default_model = spa.resolve_default_transcription_model("whisper-1")
    assert default_model == "parakeet-tdt-0.6b-v3-onnx"


@pytest.mark.unit
def test_capabilities_exposed_for_known_providers():
    spa = _import_module()

    registry = spa.SttProviderRegistry()

    fw_caps = registry.get_capabilities("faster-whisper")
    assert fw_caps.supports_batch is True
    assert fw_caps.supports_streaming is True

    parakeet_caps = registry.get_capabilities("parakeet")
    assert parakeet_caps.supports_batch is True
    assert parakeet_caps.supports_streaming is True

    canary_caps = registry.get_capabilities("canary")
    assert canary_caps.supports_batch is True
    assert canary_caps.supports_streaming is False

    qwen_caps = registry.get_capabilities("qwen2audio")
    assert qwen_caps.supports_batch is True
    assert qwen_caps.supports_streaming is False

    vibe_caps = registry.get_capabilities("vibevoice")
    assert vibe_caps.supports_batch is True
    assert vibe_caps.supports_streaming is False
    assert vibe_caps.supports_diarization is True

    qwen3_caps = registry.get_capabilities("qwen3-asr")
    assert qwen3_caps.supports_batch is True
    assert qwen3_caps.supports_streaming is False
    assert qwen3_caps.supports_diarization is False
    assert "word timestamps" in (qwen3_caps.notes or "").lower()


@pytest.mark.unit
def test_resolve_provider_for_model_qwen3_asr_variants(monkeypatch):
    spa = _import_module()

    # Test qwen3-asr-1.7b
    registry = spa.SttProviderRegistry()
    provider, model, variant = registry.resolve_provider_for_model("qwen3-asr-1.7b")
    assert provider == "qwen3-asr"
    assert model == "Qwen/Qwen3-ASR-1.7B"
    assert variant is None

    # Test qwen3-asr-0.6b
    provider, model, variant = registry.resolve_provider_for_model("qwen3-asr-0.6b")
    assert provider == "qwen3-asr"
    assert model == "Qwen/Qwen3-ASR-0.6B"
    assert variant is None

    # Test bare qwen3-asr defaults to 1.7B
    provider, model, variant = registry.resolve_provider_for_model("qwen3-asr")
    assert provider == "qwen3-asr"
    assert model == "Qwen/Qwen3-ASR-1.7B"
    assert variant is None


@pytest.mark.unit
def test_resolve_provider_for_model_qwen3_asr_aliases(monkeypatch):
    spa = _import_module()

    registry = spa.SttProviderRegistry()

    # Test underscore variant
    provider, model, variant = registry.resolve_provider_for_model("qwen3_asr_1.7b")
    assert provider == "qwen3-asr"
    assert model == "Qwen/Qwen3-ASR-1.7B"

    # Test mixed case
    provider, model, variant = registry.resolve_provider_for_model("Qwen3-ASR-0.6B")
    assert provider == "qwen3-asr"
    assert model == "Qwen/Qwen3-ASR-0.6B"


@pytest.mark.unit
def test_normalize_provider_name_qwen3_asr():
    spa = _import_module()
    registry = spa.SttProviderRegistry()

    # Test various aliases
    assert registry.normalize_provider_name("qwen3-asr") == "qwen3-asr"
    assert registry.normalize_provider_name("qwen3_asr") == "qwen3-asr"
    assert registry.normalize_provider_name("qwen3asr") == "qwen3-asr"
    assert registry.normalize_provider_name("Qwen3-ASR") == "qwen3-asr"


@pytest.mark.unit
def test_normalize_provider_name_additional_aliases():
    spa = _import_module()
    registry = spa.SttProviderRegistry()

    assert registry.normalize_provider_name("whisper") == "faster-whisper"
    assert registry.normalize_provider_name("vibevoice_asr") == "vibevoice"
    assert registry.normalize_provider_name("nemo-parakeet") == "parakeet"


@pytest.mark.unit
def test_default_provider_name_whisper_alias_maps_to_faster_whisper(monkeypatch):
    spa = _import_module()

    def fake_get_stt_config():
        return {"default_transcriber": "whisper"}

    monkeypatch.setattr(spa, "get_stt_config", fake_get_stt_config)
    registry = spa.SttProviderRegistry()

    assert registry.get_default_provider_name() == "faster-whisper"


@pytest.mark.unit
def test_transcribe_batch_whisper_normalizes_artifact(monkeypatch, tmp_path):
    spa = _import_module()

    audio_file = tmp_path / "sample.wav"
    audio_file.write_bytes(b"\x00" * 2048)

    def fake_speech_to_text(
        path,
        whisper_model,
        selected_source_lang,
        vad_filter,
        diarize,
        word_timestamps,
        return_language,
        initial_prompt=None,
        task="transcribe",
        base_dir=None,
        cancel_check=None,
    ):

        assert str(path) == str(audio_file)
        assert whisper_model == "tiny"
        assert selected_source_lang is None
        assert task == "transcribe"

        segments = [
            {"Text": "hello", "start_seconds": 0.0, "end_seconds": 0.5},
            {"Text": "world", "start_seconds": 0.5, "end_seconds": 1.0},
        ]
        return segments, "en"

    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib

    monkeypatch.setattr(atlib, "speech_to_text", fake_speech_to_text)
    monkeypatch.setattr(atlib, "strip_whisper_metadata_header", lambda segs: segs)

    adapter = spa.FasterWhisperAdapter()
    artifact = adapter.transcribe_batch(
        str(audio_file),
        model="tiny",
        language=None,
        task="transcribe",
        word_timestamps=False,
    )

    assert artifact["text"] == "hello world"
    assert artifact["language"] == "en"
    assert isinstance(artifact["segments"], list)
    # Default diarization and usage contract
    assert artifact["diarization"]["enabled"] is False
    assert artifact["diarization"]["speakers"] is None
    assert artifact["usage"]["duration_ms"] is None
    assert "actual_execution" not in artifact


@pytest.mark.unit
def test_transcribe_batch_parakeet_normalizes_artifact(monkeypatch, tmp_path):
    spa = _import_module()

    audio_file = tmp_path / "sample_parakeet.wav"
    audio_file.write_bytes(b"\x00" * 1024)

    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib

    def fake_speech_to_text(
        path,
        whisper_model,
        selected_source_lang,
        vad_filter,
        diarize,
        return_language,
        base_dir=None,
        cancel_check=None,
    ):

        assert str(path) == str(audio_file)
        # Parakeet adapter encodes model name into whisper_model
        assert whisper_model == "parakeet-standard"
        segments = [
            {"Text": "parakeet", "start_seconds": 0.0, "end_seconds": 0.5},
            {"Text": "ok", "start_seconds": 0.5, "end_seconds": 1.0},
        ]
        return segments, "en"

    monkeypatch.setattr(atlib, "speech_to_text", fake_speech_to_text, raising=True)

    adapter = spa.ParakeetAdapter()
    artifact = adapter.transcribe_batch(
        str(audio_file),
        model="parakeet-standard",
        language="en",
    )

    assert artifact["text"] == "parakeet ok"
    assert artifact["language"] == "en"
    assert isinstance(artifact["segments"], list)
    assert artifact["metadata"]["provider"] == "parakeet"
    assert artifact["metadata"]["model"] == "parakeet-standard"
    assert artifact["diarization"]["enabled"] is False
    assert "actual_execution" not in artifact


@pytest.mark.unit
def test_transcribe_batch_qwen2audio_normalizes_artifact(monkeypatch, tmp_path):
    spa = _import_module()
    audio_file = tmp_path / "sample_qwen2audio.wav"
    audio_file.write_bytes(b"\x00" * 1024)

    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib

    def fake_speech_to_text(
        path,
        whisper_model,
        selected_source_lang,
        vad_filter,
        diarize,
        return_language,
        base_dir=None,
        cancel_check=None,
    ):
        assert str(path) == str(audio_file)
        assert whisper_model == "qwen2audio"
        return [
            {
                "Text": "qwen2 audio",
                "start_seconds": 0.0,
                "end_seconds": 1.0,
            }
        ], "en"

    monkeypatch.setattr(atlib, "speech_to_text", fake_speech_to_text)

    artifact = spa.Qwen2AudioAdapter().transcribe_batch(
        str(audio_file),
        model="qwen2audio",
        language="en",
    )

    assert artifact == {
        "text": "qwen2 audio",
        "language": "en",
        "segments": [
            {
                "Text": "qwen2 audio",
                "start_seconds": 0.0,
                "end_seconds": 1.0,
            }
        ],
        "diarization": {"enabled": False, "speakers": None},
        "usage": {"duration_ms": None, "tokens": None},
        "metadata": {"provider": "qwen2audio", "model": "qwen2audio"},
    }


@pytest.mark.unit
def test_transcribe_batch_canary_normalizes_artifact(monkeypatch, tmp_path):
    spa = _import_module()

    # Create a minimal valid WAV file for soundfile to read
    import numpy as np
    import soundfile as sf

    audio_file = tmp_path / "sample_canary.wav"
    data = np.zeros(1600, dtype="float32")
    sf.write(str(audio_file), data, 16000)

    # Provide a lightweight fake Nemo module so we don't depend on real Nemo.
    import sys
    fake_nemo_mod = types.ModuleType(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo"
    )

    def fake_transcribe_with_canary(audio_np, sample_rate, language, task="transcribe", target_language=None):

        assert sample_rate == 16000
        return "canary transcript"

    fake_nemo_mod.transcribe_with_canary = fake_transcribe_with_canary
    monkeypatch.setitem(
        sys.modules,
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo",
        fake_nemo_mod,
    )

    adapter = spa.CanaryAdapter()
    artifact = adapter.transcribe_batch(
        str(audio_file),
        model="nemo-canary-1b",
        language="en",
    )

    assert artifact["text"] == "canary transcript"
    assert isinstance(artifact["segments"], list)
    assert artifact["segments"][0]["Text"] == "canary transcript"
    assert artifact["metadata"]["provider"] == "canary"
    assert artifact["diarization"]["enabled"] is False
    assert "actual_execution" not in artifact


@pytest.mark.unit
def test_transcribe_batch_canary_converts_compressed_input_before_soundfile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Canary adapter converts compressed input before soundfile reads it."""
    spa = _import_module()

    import numpy as np
    import soundfile as sf
    import sys

    compressed_file = tmp_path / "sample_canary.mp3"
    compressed_file.write_bytes(b"not really mp3")
    converted_wav = tmp_path / "sample_canary.wav"
    sf.write(str(converted_wav), np.zeros(1600, dtype="float32"), 16000)

    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib

    captured = {}

    def fake_convert_to_wav(path: str, *args: object, **kwargs: object) -> str:
        captured["input_path"] = str(path)
        captured["overwrite"] = kwargs.get("overwrite")
        return str(converted_wav)

    monkeypatch.setattr(atlib, "convert_to_wav", fake_convert_to_wav)

    fake_nemo_mod = types.ModuleType(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo"
    )

    def fake_transcribe_with_canary(
        audio_np: Any,
        sample_rate: int,
        language: str,
        task: str = "transcribe",
        target_language: str | None = None,
    ) -> str:
        assert sample_rate == 16000
        assert len(audio_np) == 1600
        return "canary transcript"

    fake_nemo_mod.transcribe_with_canary = fake_transcribe_with_canary
    monkeypatch.setitem(
        sys.modules,
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo",
        fake_nemo_mod,
    )

    adapter = spa.CanaryAdapter()
    artifact = adapter.transcribe_batch(
        str(compressed_file),
        model="nemo-canary-1b",
        language="en",
        base_dir=tmp_path,
    )

    assert artifact["text"] == "canary transcript"
    assert captured["input_path"] == str(compressed_file)
    assert captured["overwrite"] is True


@pytest.mark.unit
def test_transcribe_batch_qwen3_asr_converts_compressed_input(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Qwen3-ASR adapter passes a freshly converted WAV path to the provider."""
    spa = _import_module()

    import sys
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib

    compressed_file = tmp_path / "sample_qwen3.webm"
    compressed_file.write_bytes(b"not really webm")
    converted_wav = tmp_path / "sample_qwen3.wav"
    converted_wav.write_bytes(b"RIFFfakeWAVE")
    captured = {}

    def fake_convert_to_wav(path: str, *args: object, **kwargs: object) -> str:
        captured["convert_input"] = str(path)
        captured["overwrite"] = kwargs.get("overwrite")
        return str(converted_wav)

    monkeypatch.setattr(atlib, "convert_to_wav", fake_convert_to_wav)

    fake_qwen3_mod = types.ModuleType(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Qwen3ASR"
    )

    def fake_transcribe_with_qwen3_asr(
        audio_path: str,
        *,
        model_path: str | None = None,
        language: str | None = None,
        word_timestamps: bool = False,
        base_dir: Path | None = None,
        cancel_check: object = None,
    ) -> dict[str, Any]:
        captured["audio_path"] = str(audio_path)
        captured["base_dir"] = base_dir
        return {
            "text": "qwen3 transcript",
            "language": language or "en",
            "segments": [
                {"start_seconds": 0.0, "end_seconds": 1.0, "Text": "qwen3 transcript"}
            ],
            "diarization": {"enabled": False, "speakers": None},
            "usage": {"duration_ms": 1000, "tokens": None},
            "metadata": {"provider": "qwen3-asr", "model": model_path or "model"},
        }

    fake_qwen3_mod.transcribe_with_qwen3_asr = fake_transcribe_with_qwen3_asr
    monkeypatch.setitem(
        sys.modules,
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Qwen3ASR",
        fake_qwen3_mod,
    )

    adapter = spa.Qwen3ASRAdapter()
    artifact = adapter.transcribe_batch(
        str(compressed_file),
        model="./models/qwen3_asr/1.7B",
        language="en",
        base_dir=tmp_path,
    )

    assert artifact["text"] == "qwen3 transcript"
    assert captured["convert_input"] == str(compressed_file)
    assert captured["overwrite"] is True
    assert captured["audio_path"] == str(converted_wav)
    assert captured["base_dir"] == tmp_path


@pytest.mark.unit
def test_transcribe_batch_vibevoice_converts_compressed_input(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """VibeVoice adapter passes a freshly converted WAV path to the provider."""
    spa = _import_module()

    import sys
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib

    compressed_file = tmp_path / "sample_vibevoice.m4a"
    compressed_file.write_bytes(b"not really m4a")
    converted_wav = tmp_path / "sample_vibevoice.wav"
    converted_wav.write_bytes(b"RIFFfakeWAVE")
    captured = {}

    def fake_convert_to_wav(path: str, *args: object, **kwargs: object) -> str:
        captured["convert_input"] = str(path)
        captured["overwrite"] = kwargs.get("overwrite")
        return str(converted_wav)

    monkeypatch.setattr(atlib, "convert_to_wav", fake_convert_to_wav)

    fake_vibe_mod = types.ModuleType(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_VibeVoice"
    )

    def fake_transcribe_with_vibevoice(
        audio_path: str,
        *,
        model_id: str | None = None,
        language: str | None = None,
        hotwords: list[str] | None = None,
        base_dir: Path | None = None,
        cancel_check: object = None,
    ) -> dict[str, Any]:
        captured["audio_path"] = str(audio_path)
        captured["base_dir"] = base_dir
        return {
            "text": "vibe transcript",
            "language": language or "en",
            "segments": [
                {"start_seconds": 0.0, "end_seconds": 1.0, "Text": "vibe transcript"}
            ],
            "diarization": {"enabled": False, "speakers": None},
            "usage": {"duration_ms": 1000, "tokens": None},
            "metadata": {"provider": "vibevoice", "model": model_id or "model"},
        }

    fake_vibe_mod.transcribe_with_vibevoice = fake_transcribe_with_vibevoice
    monkeypatch.setitem(
        sys.modules,
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_VibeVoice",
        fake_vibe_mod,
    )

    adapter = spa.VibeVoiceAdapter()
    artifact = adapter.transcribe_batch(
        str(compressed_file),
        model="microsoft/VibeVoice-ASR",
        language="en",
        base_dir=tmp_path,
    )

    assert artifact["text"] == "vibe transcript"
    assert captured["convert_input"] == str(compressed_file)
    assert captured["overwrite"] is True
    assert captured["audio_path"] == str(converted_wav)
    assert captured["base_dir"] == tmp_path


@pytest.mark.unit
def test_transcribe_batch_external_normalizes_artifact(monkeypatch, tmp_path):
    spa = _import_module()

    audio_file = tmp_path / "sample_external.wav"
    audio_file.write_bytes(b"\x00" * 1024)

    # Stub external provider module to avoid real HTTP calls
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_External_Provider as ext_mod

    def fake_transcribe_with_external_provider(
        path,
        provider_name="default",
        language=None,
        sample_rate=None,
        base_dir=None,
    ):

        assert str(path) == str(audio_file)
        assert base_dir is None
        return "external transcript"

    monkeypatch.setattr(
        ext_mod,
        "transcribe_with_external_provider",
        fake_transcribe_with_external_provider,
        raising=True,
    )

    adapter = spa.ExternalAdapter()
    artifact = adapter.transcribe_batch(
        str(audio_file),
        model="external:myprovider",
        language="en",
    )

    assert artifact["text"] == "external transcript"
    assert isinstance(artifact["segments"], list)
    assert artifact["segments"][0]["Text"] == "external transcript"
    assert artifact["metadata"]["provider"] == "external"
    assert artifact["metadata"]["external_provider_name"] == "myprovider"
    assert artifact["diarization"]["enabled"] is False
    assert "actual_execution" not in artifact


@pytest.mark.unit
def test_transcribe_batch_external_passes_base_dir(monkeypatch, tmp_path):
    spa = _import_module()

    audio_file = tmp_path / "external_base_dir.wav"
    audio_file.write_bytes(b"\x00" * 2048)
    base_dir = tmp_path / "base"
    base_dir.mkdir()

    captured = {}

    def fake_transcribe_with_external_provider(
        path,
        provider_name="default",
        language=None,
        sample_rate=None,
        base_dir=None,
    ):

        captured["path"] = str(path)
        captured["base_dir"] = base_dir
        return "external ok"

    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_External_Provider as ext_mod

    monkeypatch.setattr(
        ext_mod,
        "transcribe_with_external_provider",
        fake_transcribe_with_external_provider,
        raising=True,
    )

    adapter = spa.ExternalAdapter()
    artifact = adapter.transcribe_batch(
        str(audio_file),
        model="external:stub",
        language=None,
        base_dir=base_dir,
    )

    assert artifact["text"] == "external ok"
    assert captured["path"] == str(audio_file)
    assert captured["base_dir"] == base_dir


@pytest.mark.unit
def test_transcribe_batch_qwen3_asr_normalizes_artifact(monkeypatch, tmp_path):
    spa = _import_module()

    audio_file = tmp_path / "sample_qwen3.wav"
    audio_file.write_bytes(b"\x00" * 2048)

    import sys

    fake_qwen3_mod = types.ModuleType(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Qwen3ASR"
    )

    def fake_transcribe_with_qwen3_asr(
        audio_path,
        *,
        model_path=None,
        language=None,
        word_timestamps=False,
        base_dir=None,
        cancel_check=None,
    ):
        return {
            "text": "qwen3 transcript",
            "language": language or "en",
            "segments": [
                {"start_seconds": 0.0, "end_seconds": 1.0, "Text": "qwen3 transcript"}
            ],
            "diarization": {"enabled": False, "speakers": None},
            "usage": {"duration_ms": 1000, "tokens": None},
            "metadata": {
                "provider": "qwen3-asr",
                "model": model_path or "./models/qwen3_asr/1.7B",
                "source": "local",
            },
        }

    fake_qwen3_mod.transcribe_with_qwen3_asr = fake_transcribe_with_qwen3_asr
    monkeypatch.setitem(
        sys.modules,
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Qwen3ASR",
        fake_qwen3_mod,
    )

    def fake_get_stt_config():
        return {
            "qwen3_asr_enabled": True,
            "qwen3_asr_model_path": "./models/qwen3_asr/1.7B",
        }

    monkeypatch.setattr(spa, "get_stt_config", fake_get_stt_config)

    adapter = spa.Qwen3ASRAdapter()
    artifact = adapter.transcribe_batch(
        str(audio_file),
        model="./models/qwen3_asr/1.7B",
        language="en",
    )

    assert artifact["text"] == "qwen3 transcript"
    assert artifact["language"] == "en"
    assert isinstance(artifact["segments"], list)
    assert artifact["segments"][0]["Text"] == "qwen3 transcript"
    assert artifact["metadata"]["provider"] == "qwen3-asr"
    assert artifact["diarization"]["enabled"] is False
    assert "actual_execution" not in artifact


@pytest.mark.unit
def test_transcribe_batch_vibevoice_preserves_legacy_artifact(
    monkeypatch,
    tmp_path,
):
    spa = _import_module()
    audio_file = tmp_path / "sample_vibevoice.wav"
    audio_file.write_bytes(b"\x00" * 1024)
    fake_vibe_mod = types.ModuleType(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio."
        "Audio_Transcription_VibeVoice"
    )
    expected = {
        "text": "vibevoice transcript",
        "language": "en",
        "segments": [
            {
                "Text": "vibevoice transcript",
                "start_seconds": 0.0,
                "end_seconds": 1.0,
            }
        ],
        "diarization": {"enabled": True, "speakers": 1},
        "usage": {"duration_ms": 1000, "tokens": None},
        "metadata": {
            "provider": "vibevoice",
            "model": "microsoft/VibeVoice-ASR",
        },
    }

    def fake_transcribe_with_vibevoice(
        audio_path,
        *,
        model_id,
        language,
        hotwords,
        base_dir,
        cancel_check,
    ):
        assert str(audio_path) == str(audio_file)
        assert model_id == "microsoft/VibeVoice-ASR"
        return expected

    fake_vibe_mod.transcribe_with_vibevoice = fake_transcribe_with_vibevoice
    monkeypatch.setitem(
        sys.modules,
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio."
        "Audio_Transcription_VibeVoice",
        fake_vibe_mod,
    )

    artifact = spa.VibeVoiceAdapter().transcribe_batch(
        str(audio_file),
        model="microsoft/VibeVoice-ASR",
        language="en",
    )

    assert artifact is expected
    assert "actual_execution" not in artifact


@pytest.mark.unit
def test_transcribe_batch_qwen3_asr_with_word_timestamps(monkeypatch, tmp_path):
    spa = _import_module()

    audio_file = tmp_path / "sample_qwen3_timestamps.wav"
    audio_file.write_bytes(b"\x00" * 2048)

    import sys

    fake_qwen3_mod = types.ModuleType(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Qwen3ASR"
    )

    captured = {}

    def fake_transcribe_with_qwen3_asr(
        audio_path,
        *,
        model_path=None,
        language=None,
        word_timestamps=False,
        base_dir=None,
        cancel_check=None,
    ):
        captured["word_timestamps"] = word_timestamps
        artifact = {
            "text": "hello world",
            "language": "en",
            "segments": [
                {"start_seconds": 0.0, "end_seconds": 1.0, "Text": "hello world"}
            ],
            "diarization": {"enabled": False, "speakers": None},
            "usage": {"duration_ms": 1000, "tokens": None},
            "metadata": {
                "provider": "qwen3-asr",
                "model": model_path or "./models/qwen3_asr/1.7B",
                "source": "local",
            },
        }
        if word_timestamps:
            artifact["words"] = [
                {"word": "hello", "start": 0.0, "end": 0.4},
                {"word": "world", "start": 0.5, "end": 1.0},
            ]
        return artifact

    fake_qwen3_mod.transcribe_with_qwen3_asr = fake_transcribe_with_qwen3_asr
    monkeypatch.setitem(
        sys.modules,
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Qwen3ASR",
        fake_qwen3_mod,
    )

    def fake_get_stt_config():
        return {
            "qwen3_asr_enabled": True,
            "qwen3_asr_model_path": "./models/qwen3_asr/1.7B",
            "qwen3_asr_aligner_enabled": True,
        }

    monkeypatch.setattr(spa, "get_stt_config", fake_get_stt_config)

    adapter = spa.Qwen3ASRAdapter()
    artifact = adapter.transcribe_batch(
        str(audio_file),
        model="./models/qwen3_asr/1.7B",
        language="en",
        word_timestamps=True,
    )

    assert captured["word_timestamps"] is True
    assert "words" in artifact
    assert len(artifact["words"]) == 2
    assert artifact["words"][0]["word"] == "hello"


@pytest.mark.unit
def test_qwen3_asr_adapter_uses_config_default(monkeypatch):
    spa = _import_module()

    def fake_get_stt_config():
        return {
            "default_transcriber": "qwen3-asr",
            "qwen3_asr_model_path": "./custom/model/path",
        }

    monkeypatch.setattr(spa, "get_stt_config", fake_get_stt_config)

    registry = spa.SttProviderRegistry()
    provider, model, variant = registry.resolve_provider_for_model(None)

    assert provider == "qwen3-asr"
    assert model == "./custom/model/path"
    assert variant is None
