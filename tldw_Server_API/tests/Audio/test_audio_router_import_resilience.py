import importlib
import importlib.machinery
import sys
from types import ModuleType
import types

import pytest
from tldw_Server_API.app.core.Utils.fastapi_routes import iter_served_routes


def _module_names(*names: str) -> list[str]:
    return [name for name in names if name]


def _pop_modules(*names: str) -> dict[str, ModuleType]:
    original: dict[str, ModuleType] = {}
    for name in _module_names(*names):
        module = sys.modules.pop(name, None)
        if isinstance(module, ModuleType):
            original[name] = module
    return original


def _restore_modules(original: dict[str, ModuleType], *names: str) -> None:
    for name in _module_names(*names):
        sys.modules.pop(name, None)
        module = original.get(name)
        if module is not None:
            sys.modules[name] = module


def _new_stub_module(name: str) -> ModuleType:
    module = ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    return module


def _install_lightweight_audio_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = _new_stub_module("torch")
    fake_torch.Tensor = object
    fake_torch.nn = types.SimpleNamespace(Module=object)
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    fake_fw = _new_stub_module("faster_whisper")

    class _StubWhisperModel:
        def __init__(self, *args, **kwargs):
            pass

    fake_fw.WhisperModel = _StubWhisperModel
    fake_fw.BatchedInferencePipeline = _StubWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_fw)

    fake_tf = _new_stub_module("transformers")

    class _StubProcessor:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class _StubModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    fake_tf.AutoProcessor = _StubProcessor
    fake_tf.Qwen2AudioForConditionalGeneration = _StubModel
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)


@pytest.mark.unit
def test_audio_router_import_survives_broken_streaming_module(monkeypatch):
    """Aggregate audio router should still expose REST routes when streaming import breaks."""
    audio_module_name = "tldw_Server_API.app.api.v1.endpoints.audio.audio"
    streaming_module_name = "tldw_Server_API.app.api.v1.endpoints.audio.audio_streaming"
    original_audio_module = sys.modules.pop(audio_module_name, None)
    original_streaming_module = sys.modules.get(streaming_module_name)

    class _BrokenStreamingModule(ModuleType):
        def __getattr__(self, name: str):
            raise RuntimeError(f"streaming dependency unavailable for {name}")

    monkeypatch.setitem(sys.modules, streaming_module_name, _BrokenStreamingModule(streaming_module_name))

    try:
        audio_module = importlib.import_module(audio_module_name)
        route_paths = {route.path for route in iter_served_routes(audio_module.router.routes)}

        assert "/transcriptions" in route_paths
        assert "/transcriptions/health" in route_paths
        assert all(not path.startswith("/stream") for path in route_paths)
        assert audio_module.ws_router.routes == []
    finally:
        sys.modules.pop(audio_module_name, None)
        if original_streaming_module is not None:
            sys.modules[streaming_module_name] = original_streaming_module
        else:
            sys.modules.pop(streaming_module_name, None)
        if original_audio_module is not None:
            sys.modules[audio_module_name] = original_audio_module


@pytest.mark.unit
def test_audio_streaming_import_survives_broken_unified_backend(monkeypatch):
    """Streaming route module should import without touching unified backend symbols."""
    module_name = "tldw_Server_API.app.api.v1.endpoints.audio.audio_streaming"
    unified_module_name = (
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Unified"
    )
    _install_lightweight_audio_stubs(monkeypatch)
    original_modules = _pop_modules(module_name, unified_module_name)

    class _BrokenUnifiedModule(ModuleType):
        def __getattr__(self, name: str):
            raise RuntimeError(f"unified backend unavailable for {name}")

    broken_unified = _BrokenUnifiedModule(unified_module_name)
    broken_unified.__spec__ = importlib.machinery.ModuleSpec(unified_module_name, loader=None)
    monkeypatch.setitem(sys.modules, unified_module_name, broken_unified)

    try:
        audio_streaming = importlib.import_module(module_name)
        assert getattr(audio_streaming, "router", None) is not None
        assert getattr(audio_streaming, "ws_router", None) is not None
    finally:
        _restore_modules(original_modules, module_name, unified_module_name)


@pytest.mark.unit
def test_app_main_import_survives_broken_unified_backend(monkeypatch):
    """App startup should not require importing the unified audio backend."""
    module_name = "tldw_Server_API.app.main"
    router_module_name = "tldw_Server_API.app.api.v1.router"
    audio_module_name = "tldw_Server_API.app.api.v1.endpoints.audio.audio"
    streaming_module_name = "tldw_Server_API.app.api.v1.endpoints.audio.audio_streaming"
    unified_module_name = (
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Unified"
    )
    _install_lightweight_audio_stubs(monkeypatch)
    original_modules = _pop_modules(
        module_name,
        router_module_name,
        audio_module_name,
        streaming_module_name,
        unified_module_name,
    )

    class _BrokenUnifiedModule(ModuleType):
        def __getattr__(self, name: str):
            raise RuntimeError(f"unified backend unavailable for {name}")

    broken_unified = _BrokenUnifiedModule(unified_module_name)
    broken_unified.__spec__ = importlib.machinery.ModuleSpec(unified_module_name, loader=None)
    monkeypatch.setitem(sys.modules, unified_module_name, broken_unified)

    try:
        app_main = importlib.import_module(module_name)
        assert getattr(app_main, "app", None) is not None
    finally:
        _restore_modules(
            original_modules,
            module_name,
            router_module_name,
            audio_module_name,
            streaming_module_name,
            unified_module_name,
        )
