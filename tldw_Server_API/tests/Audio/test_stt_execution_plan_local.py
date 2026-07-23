"""Plan-enforcement tests for local native STT providers."""

from __future__ import annotations

import importlib
import importlib.machinery
import sys
import types
from typing import Any

import numpy as np
import pytest

# Keep optional native runtimes out of ordinary unit tests.
if "torch" not in sys.modules:
    fake_torch = types.ModuleType("torch")
    fake_torch.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)
    fake_torch.Tensor = object
    fake_torch.float16 = "float16"
    fake_torch.float32 = "float32"
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.nn = types.SimpleNamespace(Module=object)
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    fake_torch.no_grad = lambda: types.SimpleNamespace(
        __enter__=lambda self: None,
        __exit__=lambda self, *_args: None,
    )
    sys.modules["torch"] = fake_torch

if "faster_whisper" not in sys.modules:
    fake_fw = types.ModuleType("faster_whisper")
    fake_fw.__spec__ = importlib.machinery.ModuleSpec(
        "faster_whisper",
        loader=None,
    )
    fake_fw.WhisperModel = object
    fake_fw.BatchedInferencePipeline = object
    sys.modules["faster_whisper"] = fake_fw

if "transformers" not in sys.modules:
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.__spec__ = importlib.machinery.ModuleSpec(
        "transformers",
        loader=None,
    )
    fake_transformers.AutoProcessor = object
    fake_transformers.Qwen2AudioForConditionalGeneration = object
    sys.modules["transformers"] = fake_transformers

from tldw_Server_API.app.core.exceptions import (  # noqa: E402
    STTExecutionPlanError,
    STTExecutionUnsupportedError,
    STTTranscriptionError,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import (  # noqa: E402
    Audio_Transcription_Lib as atlib,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import (  # noqa: E402
    Audio_Transcription_Nemo as nemo,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import (  # noqa: E402
    Audio_Transcription_Parakeet_MLX as parakeet_mlx,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import (  # noqa: E402
    Audio_Transcription_Parakeet_ONNX as parakeet_onnx,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import (  # noqa: E402
    stt_provider_adapter as spa,
)


def _route(
    *,
    provider: str,
    model_label: str,
    backend: str,
    device: str | None,
    compute_type: str | None = None,
    dtype: str | None = None,
) -> spa.SttExecutionRoute:
    return spa.SttExecutionRoute(
        route_id="neutral-1",
        provider=provider,
        model_label=model_label,
        artifact_id=None,
        identity_resolved=False,
        backend=backend,
        source="local",
        audio_egress=spa.SttAudioEgress.NONE,
        endpoint_id=None,
        device=device,
        compute_type=compute_type,
        dtype=dtype,
        decoding_ids=(),
        local_model_available=True,
        would_download=False,
    )


def _plan(
    route: spa.SttExecutionRoute,
    *,
    language: str = "en",
    runtime_settings: tuple[tuple[str, spa.SttPlanScalar], ...],
) -> spa.SttBatchExecutionPlan:
    descriptor = spa.SttExecutionDescriptor(
        requested_provider=route.provider,
        requested_model_label=route.model_label,
        resolved_provider=route.provider,
        resolved_model_label=route.model_label,
        routes=(route,),
        honors_task=True,
        honors_language=True,
        honors_prompt_absence=True,
        honors_hotword_absence=True,
        honors_diarization=True,
        honors_word_timestamps=True,
        decoding_settings=(),
        source_modules=(
            "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_provider_adapter",
        ),
        dependency_distributions=("pytest",),
    )
    return spa.SttBatchExecutionPlan(
        descriptor=descriptor,
        task="transcribe",
        language=language,
        runtime_settings=runtime_settings,
    )


@pytest.fixture(autouse=True)
def _clear_local_model_caches(monkeypatch):
    atlib.whisper_model_cache.clear()
    atlib.qwen_processor = None
    atlib.qwen_model = None
    nemo._model_cache.clear()
    parakeet_onnx._onnx_model_cache.clear()
    parakeet_mlx._mlx_model_cache = None
    yield
    atlib.whisper_model_cache.clear()
    atlib.qwen_processor = None
    atlib.qwen_model = None
    nemo._model_cache.clear()
    parakeet_onnx._onnx_model_cache.clear()
    parakeet_mlx._mlx_model_cache = None


@pytest.mark.unit
def test_planned_whisper_requires_local_model_before_library_entry(
    monkeypatch,
):
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(atlib, "check_model_exists", lambda _name: False)
    monkeypatch.setattr(
        atlib,
        "WhisperModel",
        lambda **kwargs: calls.append(kwargs),
    )
    route = _route(
        provider="faster-whisper",
        model_label="tiny",
        backend="ctranslate2",
        device="cpu",
        compute_type="int8",
    )

    with pytest.raises(STTExecutionUnsupportedError, match="available locally"):
        atlib.get_whisper_model(
            "tiny",
            "cpu",
            compute_type_override="int8",
            local_files_only=True,
            allow_device_fallback=False,
            execution_route=route,
        )

    assert calls == []


@pytest.mark.unit
def test_planned_whisper_passes_offline_device_and_compute_and_reports_loaded_values(
    monkeypatch,
):
    calls: list[dict[str, Any]] = []

    class FakeWhisper:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.model = types.SimpleNamespace(
                device="cpu",
                compute_type="int8_float32",
            )

    monkeypatch.setattr(atlib, "check_model_exists", lambda _name: True)
    monkeypatch.setattr(atlib, "WhisperModel", FakeWhisper)
    route = _route(
        provider="faster-whisper",
        model_label="tiny",
        backend="ctranslate2",
        device="cpu",
        compute_type=None,
    )

    loaded = atlib.get_whisper_model(
        "tiny",
        "cpu",
        compute_type_override="int8",
        local_files_only=True,
        allow_device_fallback=False,
        execution_route=route,
    )

    assert isinstance(loaded, spa.SttLoadedRuntime)
    assert calls == [
        {
            "model_size_or_path": "tiny",
            "device": "cpu",
            "compute_type": "int8",
            "local_files_only": True,
        }
    ]
    assert loaded.actual_execution.device == "cpu"
    assert loaded.actual_execution.compute_type == "int8_float32"


@pytest.mark.unit
def test_planned_whisper_cuda_failure_never_retries_cpu(monkeypatch):
    calls: list[dict[str, Any]] = []

    def fail_cuda(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("CUDA initialization failed")

    monkeypatch.setattr(atlib, "check_model_exists", lambda _name: True)
    monkeypatch.setattr(atlib, "WhisperModel", fail_cuda)
    route = _route(
        provider="faster-whisper",
        model_label="tiny",
        backend="ctranslate2",
        device="cuda",
        compute_type="float16",
    )

    with pytest.raises(RuntimeError, match="CUDA"):
        atlib.get_whisper_model(
            "tiny",
            "cuda",
            compute_type_override="float16",
            local_files_only=True,
            allow_device_fallback=False,
            execution_route=route,
        )

    assert len(calls) == 1
    assert calls[0]["device"] == "cuda"


@pytest.mark.unit
def test_planned_whisper_rejects_unprovable_effective_runtime(monkeypatch):
    monkeypatch.setattr(atlib, "check_model_exists", lambda _name: True)
    monkeypatch.setattr(
        atlib,
        "WhisperModel",
        lambda **_kwargs: object(),
    )
    route = _route(
        provider="faster-whisper",
        model_label="tiny",
        backend="ctranslate2",
        device="cpu",
        compute_type="int8",
    )

    with pytest.raises(STTExecutionPlanError, match="effective"):
        atlib.get_whisper_model(
            "tiny",
            "cpu",
            compute_type_override="int8",
            local_files_only=True,
            allow_device_fallback=False,
            execution_route=route,
        )


@pytest.mark.unit
def test_planned_qwen_uses_immutable_local_settings_and_loaded_runtime(
    monkeypatch,
    tmp_path,
):
    model_dir = tmp_path / "qwen"
    model_dir.mkdir()
    calls: list[tuple[str, str, dict[str, Any]]] = []

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.append(("processor", model_id, kwargs))
            return cls()

    class FakeModel:
        device = "mps"
        dtype = "bfloat16"

        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.append(("model", model_id, kwargs))
            return cls()

    fake_torch = types.SimpleNamespace(
        float16="float16",
        float32="float32",
        bfloat16="bfloat16",
    )
    monkeypatch.setattr(atlib, "WHISPER_MODEL_BASE_DIR", tmp_path)
    monkeypatch.setattr(atlib, "_get_torch", lambda **_kwargs: fake_torch)
    monkeypatch.setattr(
        atlib,
        "_get_qwen2audio_classes",
        lambda: (FakeProcessor, FakeModel),
    )
    monkeypatch.setattr(
        atlib,
        "load_and_log_configs",
        lambda: (_ for _ in ()).throw(AssertionError("config reread")),
    )
    route = _route(
        provider="qwen2audio",
        model_label="local-model",
        backend="transformers",
        device=None,
        dtype=None,
    )

    loaded = atlib.load_qwen2audio(
        model_id=str(model_dir),
        revision="a" * 40,
        local_files_only=True,
        device_map="mps",
        dtype_name="bfloat16",
        execution_route=route,
    )

    assert isinstance(loaded, spa.SttLoadedRuntime)
    assert calls == [
        (
            "processor",
            str(model_dir),
            {"revision": "a" * 40, "local_files_only": True},
        ),
        (
            "model",
            str(model_dir),
            {
                "revision": "a" * 40,
                "local_files_only": True,
                "torch_dtype": "bfloat16",
                "device_map": "mps",
            },
        ),
    ]
    assert loaded.actual_execution.device == "mps"
    assert loaded.actual_execution.dtype == "bfloat16"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("loader_name", "provider", "backend"),
    [
        ("load_canary_model", "canary", "nemo"),
        ("load_parakeet_model", "parakeet", "nemo"),
    ],
)
def test_planned_nemo_requires_explicit_local_nemo_artifact(
    loader_name,
    provider,
    backend,
):
    route = _route(
        provider=provider,
        model_label="local-model",
        backend=backend,
        device="cpu",
        dtype="float32",
    )
    loader = getattr(nemo, loader_name)
    kwargs = {
        "model_path": "missing.nemo",
        "device": "cpu",
        "allow_download": False,
        "execution_route": route,
    }
    if loader_name == "load_canary_model":
        kwargs["dtype_name"] = "float32"
    else:
        kwargs["compute_type"] = "float32"
        kwargs["allow_variant_fallback"] = False

    with pytest.raises(STTExecutionUnsupportedError, match=r"\.nemo"):
        loader(**kwargs)


@pytest.mark.unit
def test_planned_canary_uses_restore_from_and_loaded_device_dtype(
    monkeypatch,
    tmp_path,
):
    model_path = tmp_path / "canary.nemo"
    model_path.write_bytes(b"local")
    calls: list[tuple[str, Any]] = []

    class FakeModel:
        device = "cpu"
        dtype = "float32"

        def to(self, **kwargs):
            calls.append(("to", kwargs))
            return self

        def eval(self):
            calls.append(("eval", None))

    class FakeCanary:
        @classmethod
        def restore_from(cls, path, map_location=None):
            calls.append(("restore_from", (path, map_location)))
            return FakeModel()

        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            raise AssertionError("planned Canary must not download")

    fake_asr = types.ModuleType("nemo.collections.asr")
    fake_asr.models = types.SimpleNamespace(EncDecMultiTaskModel=FakeCanary)
    _install_nemo_modules(monkeypatch, fake_asr)
    monkeypatch.setattr(
        nemo,
        "get_stt_config",
        lambda: (_ for _ in ()).throw(AssertionError("config reread")),
    )
    route = _route(
        provider="canary",
        model_label="local-model",
        backend="nemo",
        device="cpu",
        dtype="float32",
    )

    loaded = nemo.load_canary_model(
        model_path=str(model_path),
        device="cpu",
        dtype_name="float32",
        allow_download=False,
        execution_route=route,
    )

    assert isinstance(loaded, spa.SttLoadedRuntime)
    assert calls[0] == ("restore_from", (str(model_path), "cpu"))
    assert loaded.actual_execution.device == "cpu"
    assert loaded.actual_execution.dtype == "float32"


@pytest.mark.unit
def test_planned_canary_typed_failure_does_not_enter_temp_file_fallback(
    monkeypatch,
):
    route = _route(
        provider="canary",
        model_label="nemo-canary-1b",
        backend="nemo",
        device="cpu",
        dtype="float32",
    )
    plan = _plan(
        route,
        runtime_settings=(
            ("device", "cpu"),
            ("dtype", "float32"),
            ("model_path", "canary.nemo"),
            ("variant", "standard"),
        ),
    )
    calls: list[bool] = []

    class FakeModel:
        def transcribe(self, *_args, **_kwargs):
            calls.append(True)
            return ["[Error: planned Canary failed]"]

    monkeypatch.setattr(
        nemo,
        "load_canary_model",
        lambda **_kwargs: spa.SttLoadedRuntime(
            components=(FakeModel(),),
            actual_execution=spa.actual_execution_from_route(
                route,
                device="cpu",
                dtype="float32",
            ),
        ),
    )

    with pytest.raises(STTTranscriptionError, match="planned Canary failed"):
        nemo.transcribe_with_canary(
            np.zeros(1600, dtype=np.float32),
            execution_plan=plan,
        )

    assert calls == [True]


def _install_nemo_modules(monkeypatch, asr_module):
    nemo_package = types.ModuleType("nemo")
    nemo_package.__path__ = []
    collections_package = types.ModuleType("nemo.collections")
    collections_package.__path__ = []
    collections_package.asr = asr_module
    nemo_package.collections = collections_package
    monkeypatch.setitem(sys.modules, "nemo", nemo_package)
    monkeypatch.setitem(sys.modules, "nemo.collections", collections_package)
    monkeypatch.setitem(sys.modules, "nemo.collections.asr", asr_module)


@pytest.mark.unit
def test_planned_parakeet_onnx_never_calls_snapshot_download(
    monkeypatch,
    tmp_path,
):
    model_dir = tmp_path / "onnx"
    model_dir.mkdir()
    (model_dir / "model.onnx").write_bytes(b"graph")

    class FakeSession:
        def __init__(self, *_args, **_kwargs):
            pass

        def get_inputs(self):
            return []

        def get_providers(self):
            return ["CPUExecutionProvider", "CUDAExecutionProvider"]

    fake_ort = types.SimpleNamespace(
        InferenceSession=FakeSession,
        SessionOptions=lambda: types.SimpleNamespace(
            graph_optimization_level=None,
        ),
        GraphOptimizationLevel=types.SimpleNamespace(ORT_ENABLE_ALL="all"),
    )
    monkeypatch.setattr(parakeet_onnx, "ort", fake_ort)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setattr(
        parakeet_onnx,
        "snapshot_download",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("snapshot_download called")
        ),
    )
    route = _route(
        provider="parakeet",
        model_label="parakeet-tdt-0.6b-v3-onnx",
        backend="onnxruntime",
        device=None,
    )

    loaded = parakeet_onnx.load_parakeet_onnx_model(
        str(model_dir),
        "cuda",
        allow_download=False,
        execution_route=route,
    )

    assert isinstance(loaded, spa.SttLoadedRuntime)
    assert loaded.actual_execution.device == "cpu"


@pytest.mark.unit
def test_planned_parakeet_mlx_requires_existing_explicit_local_artifact(
    monkeypatch,
):
    monkeypatch.setattr(parakeet_mlx, "IS_MACOS", True)
    monkeypatch.setattr(parakeet_mlx, "check_mlx_available", lambda: True)
    monkeypatch.setattr(parakeet_mlx, "check_parakeet_mlx_installed", lambda: True)
    route = _route(
        provider="parakeet",
        model_label="parakeet-mlx",
        backend="mlx",
        device="mps",
        dtype="bfloat16",
    )

    with pytest.raises(STTExecutionUnsupportedError, match="existing"):
        parakeet_mlx.load_parakeet_mlx_model(
            model_path="missing-local-model",
            allow_download=False,
            execution_route=route,
        )


@pytest.mark.unit
def test_planned_parakeet_mlx_platform_failure_is_typed(
    monkeypatch,
    tmp_path,
):
    model_dir = tmp_path / "mlx"
    model_dir.mkdir()
    monkeypatch.setattr(parakeet_mlx, "IS_MACOS", False)
    route = _route(
        provider="parakeet",
        model_label="parakeet-mlx",
        backend="mlx",
        device="mps",
        dtype="bfloat16",
    )

    with pytest.raises(STTExecutionUnsupportedError, match="macOS"):
        parakeet_mlx.load_parakeet_mlx_model(
            model_path=str(model_dir),
            allow_download=False,
            execution_route=route,
        )


@pytest.mark.unit
def test_planned_parakeet_mlx_never_retries_or_falls_back(
    monkeypatch,
    tmp_path,
):
    model_dir = tmp_path / "mlx"
    model_dir.mkdir()
    calls: list[tuple[str, dict[str, Any]]] = []

    class FakeModel:
        device = "mps"
        dtype = "bfloat16"

    fake_package = types.ModuleType("parakeet_mlx")

    def from_pretrained(model_id, **kwargs):
        calls.append((model_id, kwargs))
        return FakeModel()

    fake_package.from_pretrained = from_pretrained
    monkeypatch.setitem(sys.modules, "parakeet_mlx", fake_package)
    monkeypatch.setattr(parakeet_mlx, "IS_MACOS", True)
    monkeypatch.setattr(parakeet_mlx, "check_mlx_available", lambda: True)
    monkeypatch.setattr(parakeet_mlx, "check_parakeet_mlx_installed", lambda: True)
    route = _route(
        provider="parakeet",
        model_label="parakeet-mlx",
        backend="mlx",
        device="mps",
        dtype="bfloat16",
    )

    loaded = parakeet_mlx.load_parakeet_mlx_model(
        model_path=str(model_dir),
        allow_download=False,
        execution_route=route,
    )

    assert isinstance(loaded, spa.SttLoadedRuntime)
    assert calls == [(str(model_dir), {})]
    assert loaded.actual_execution.device == "mps"
    assert loaded.actual_execution.dtype == "bfloat16"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("adapter_type", "model", "language"),
    [
        (spa.ParakeetAdapter, "parakeet-standard", "en"),
        (spa.ParakeetAdapter, "parakeet-standard", "en-US"),
        (spa.Qwen2AudioAdapter, "local-model", "en-GB"),
    ],
)
def test_fixed_english_planners_accept_english_primary_subtag(
    adapter_type,
    model,
    language,
    monkeypatch,
    tmp_path,
):
    local_path = tmp_path / (
        "parakeet.nemo" if adapter_type is spa.ParakeetAdapter else "qwen"
    )
    if local_path.suffix:
        local_path.write_bytes(b"local")
    else:
        local_path.mkdir()
    config = {
        "nemo_device": "cpu",
        "parakeet_model_path": str(local_path),
        "qwen2audio_model_id": str(local_path),
        "qwen2audio_device_map": "cpu",
        "qwen2audio_dtype": "float32",
    }
    monkeypatch.setattr(spa, "get_stt_config", lambda: config)

    plan = adapter_type().plan_batch_execution(
        model=model,
        language=language,
        task="translate",
        word_timestamps=True,
        prompt="ignored",
        hotwords=("ignored",),
        diarization=True,
        mode="neutral-v1",
    )

    assert plan.task == "transcribe"
    assert plan.language == language
    assert plan.prompt is None
    assert plan.hotwords == ()
    assert plan.diarization is False
    assert plan.word_timestamps is False
    assert dict(plan.descriptor.decoding_settings)["language_contract"] == "fixed:en"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("adapter", "model"),
    [
        (spa.ParakeetAdapter(), "parakeet-standard"),
        (spa.Qwen2AudioAdapter(), "local-model"),
    ],
)
def test_fixed_english_planners_reject_non_english_before_audio_open(
    adapter,
    model,
    monkeypatch,
    tmp_path,
):
    local_nemo = tmp_path / "parakeet.nemo"
    local_nemo.write_bytes(b"local")
    local_qwen = tmp_path / "qwen"
    local_qwen.mkdir()
    monkeypatch.setattr(
        spa,
        "get_stt_config",
        lambda: {
            "parakeet_model_path": str(local_nemo),
            "qwen2audio_model_id": str(local_qwen),
        },
    )

    with pytest.raises(STTExecutionUnsupportedError, match="English"):
        adapter.plan_batch_execution(
            model=model,
            language="fr-FR",
            task="transcribe",
            word_timestamps=False,
            prompt=None,
            hotwords=None,
            diarization=False,
            mode="neutral-v1",
        )


@pytest.mark.unit
def test_parakeet_planner_rejects_unsupported_variant_before_audio_open():
    with pytest.raises(STTExecutionUnsupportedError, match="variant"):
        spa.ParakeetAdapter().plan_batch_execution(
            model="parakeet-unknown",
            language="en",
            task="transcribe",
            word_timestamps=False,
            prompt=None,
            hotwords=None,
            diarization=False,
            mode="neutral-v1",
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("adapter_type", "model", "config_key", "artifact_kind", "language"),
    [
        (
            spa.FasterWhisperAdapter,
            "local-model",
            None,
            "directory",
            "fr",
        ),
        (
            spa.CanaryAdapter,
            "local-model",
            None,
            "nemo",
            "fr",
        ),
        (
            spa.ParakeetAdapter,
            "parakeet-onnx",
            "parakeet_onnx_model_id",
            "directory",
            "en",
        ),
    ],
)
def test_supported_production_planners_snapshot_semantics(
    adapter_type,
    model,
    config_key,
    artifact_kind,
    language,
    monkeypatch,
    tmp_path,
):
    artifact = tmp_path / ("model.nemo" if artifact_kind == "nemo" else "model")
    if artifact_kind == "nemo":
        artifact.write_bytes(b"local")
    else:
        artifact.mkdir()
    config = {
        "nemo_device": "cpu",
        "nemo_compute_type": "float32",
        "stt_benchmark_configuration_id": "production-1",
        "whisper_compute_type": "int8",
        "whisper_device": "cpu",
    }
    if config_key is not None:
        config[config_key] = str(artifact)
    monkeypatch.setattr(spa, "get_stt_config", lambda: config)

    plan = adapter_type().plan_batch_execution(
        model=str(artifact) if model == "local-model" else model,
        language=language,
        task="transcribe",
        word_timestamps=True,
        prompt="private prompt",
        hotwords=("private hotword",),
        diarization=True,
        mode="production-v1",
    )

    assert plan.prompt == "private prompt"
    assert plan.hotwords == ("private hotword",)
    assert plan.diarization is True
    assert plan.word_timestamps is True
    assert dict(plan.descriptor.decoding_settings) == {
        "configuration_id": "production-1",
        "hotword_count": 1,
        **(
            {"language_contract": "fixed:en"}
            if adapter_type is spa.ParakeetAdapter
            else {}
        ),
        "prompt_present": True,
    }
    serialized = plan.descriptor.as_safe_dict()
    assert "private prompt" not in str(serialized)
    assert "private hotword" not in str(serialized)


@pytest.mark.unit
def test_qwen_production_planner_fails_closed():
    with pytest.raises(STTExecutionUnsupportedError, match="fallback"):
        spa.Qwen2AudioAdapter().plan_batch_execution(
            model="qwen2audio",
            language="en",
            task="transcribe",
            word_timestamps=False,
            prompt=None,
            hotwords=None,
            diarization=False,
            mode="production-v1",
        )


@pytest.mark.unit
def test_planned_faster_whisper_uses_typed_outcome_and_skips_custom_vocab(
    monkeypatch,
    tmp_path,
):
    model_dir = tmp_path / "whisper"
    model_dir.mkdir()
    audio_file = tmp_path / "sample.wav"
    audio_file.write_bytes(b"audio")
    route = _route(
        provider="faster-whisper",
        model_label="local-model",
        backend="ctranslate2",
        device="cpu",
        compute_type="int8",
    )
    plan = _plan(
        route,
        runtime_settings=(
            ("compute_type", "int8"),
            ("device", "cpu"),
            ("model_path", str(model_dir)),
        ),
    )
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
    )
    calls: list[dict[str, Any]] = []

    def fake_speech_to_text(path, **kwargs):
        calls.append(kwargs)
        return spa.SttTranscriptionOutcome(
            artifact={
                "text": "planned transcript",
                "segments": [{"Text": "planned transcript"}],
                "language": "en",
            },
            actual_execution=actual,
        )

    monkeypatch.setattr(atlib, "speech_to_text", fake_speech_to_text)
    monkeypatch.setattr(
        spa,
        "get_stt_config",
        lambda: (_ for _ in ()).throw(AssertionError("config reread")),
    )

    artifact = spa.FasterWhisperAdapter().transcribe_batch(
        str(audio_file),
        model=str(model_dir),
        language="en",
        execution_plan=plan,
    )

    assert artifact["text"] == "planned transcript"
    assert artifact["actual_execution"]["compute_type"] == "int8"
    assert calls[0]["execution_plan"] is plan
    assert calls[0]["initial_prompt"] is None


@pytest.mark.unit
def test_planned_qwen_failure_never_falls_back_to_whisper(
    monkeypatch,
    tmp_path,
):
    model_dir = tmp_path / "qwen"
    model_dir.mkdir()
    audio_file = tmp_path / "sample.wav"
    audio_file.write_bytes(b"audio")
    route = _route(
        provider="qwen2audio",
        model_label="local-model",
        backend="transformers",
        device=None,
        dtype=None,
    )
    plan = _plan(
        route,
        runtime_settings=(
            ("device_map", "cpu"),
            ("dtype", "float32"),
            ("model_path", str(model_dir)),
            ("revision", "a" * 40),
        ),
    )
    whisper_calls: list[bool] = []

    monkeypatch.setattr(
        atlib,
        "speech_to_text_qwen2audio",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("planned qwen failed")
        ),
    )
    monkeypatch.setattr(
        atlib,
        "get_whisper_model",
        lambda *_args, **_kwargs: whisper_calls.append(True),
    )

    with pytest.raises(RuntimeError, match="planned qwen failed"):
        atlib.speech_to_text(
            str(audio_file),
            whisper_model=str(model_dir),
            selected_source_lang="en",
            execution_plan=plan,
        )

    assert whisper_calls == []


@pytest.mark.unit
def test_local_adapters_require_typed_outcome_in_planned_mode(
    monkeypatch,
    tmp_path,
):
    audio_file = tmp_path / "sample.wav"
    audio_file.write_bytes(b"audio")
    adapters_and_routes = (
        (
            spa.FasterWhisperAdapter(),
            _route(
                provider="faster-whisper",
                model_label="tiny",
                backend="ctranslate2",
                device="cpu",
                compute_type="int8",
            ),
        ),
        (
            spa.ParakeetAdapter(),
            _route(
                provider="parakeet",
                model_label="parakeet-standard",
                backend="nemo",
                device="cpu",
                dtype="float32",
            ),
        ),
        (
            spa.CanaryAdapter(),
            _route(
                provider="canary",
                model_label="nemo-canary-1b",
                backend="nemo",
                device="cpu",
                dtype="float32",
            ),
        ),
        (
            spa.Qwen2AudioAdapter(),
            _route(
                provider="qwen2audio",
                model_label="qwen2audio",
                backend="transformers",
                device=None,
                dtype=None,
            ),
        ),
    )
    monkeypatch.setattr(
        atlib,
        "speech_to_text",
        lambda *_args, **_kwargs: ([], "en"),
    )

    for adapter, route in adapters_and_routes:
        plan = _plan(
            route,
            runtime_settings=(("model_path", str(audio_file)),),
        )
        with pytest.raises(STTExecutionPlanError, match="typed"):
            adapter.transcribe_batch(
                str(audio_file),
                model=route.model_label,
                language="en",
                execution_plan=plan,
            )
