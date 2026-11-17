import importlib
import importlib.machinery
import sys
import types
from types import ModuleType

import pytest

from tldw_Server_API.app.core.Utils.fastapi_routes import iter_served_routes


def _module_names(*names: str) -> list[str]:
    return [name for name in names if name]


_MISSING = object()


def _pop_modules(*names: str) -> dict[str, tuple[ModuleType | None, ModuleType | None, object]]:
    original = {}
    for name in _module_names(*names):
        parent_name, _, child_name = name.rpartition(".")
        parent = sys.modules.get(parent_name)
        binding = vars(parent).get(child_name, _MISSING) if parent is not None else _MISSING
        original[name] = (sys.modules.pop(name, None), parent, binding)
    return original


def _restore_modules(original: dict[str, tuple[ModuleType | None, ModuleType | None, object]], *names: str) -> None:
    for name in _module_names(*names):
        sys.modules.pop(name, None)
        module, parent, binding = original[name]
        if module is not None:
            sys.modules[name] = module
        # Imports also bind children on their parent package, independently of sys.modules.
        if parent is not None:
            child_name = name.rpartition(".")[2]
            if binding is _MISSING:
                vars(parent).pop(child_name, None)
            else:
                setattr(parent, child_name, binding)


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
    original_modules = _pop_modules(audio_module_name, streaming_module_name)

    class _BrokenStreamingModule(ModuleType):
        def __getattr__(self, name: str):
            raise RuntimeError(f"streaming dependency unavailable for {name}")

    sys.modules[streaming_module_name] = _BrokenStreamingModule(streaming_module_name)

    try:
        audio_module = importlib.import_module(audio_module_name)
        route_paths = {route.path for route in iter_served_routes(audio_module.router.routes)}

        assert "/transcriptions" in route_paths
        assert "/transcriptions/health" in route_paths
        assert all(not path.startswith("/stream") for path in route_paths)
        assert audio_module.ws_router.routes == []
    finally:
        _restore_modules(original_modules, audio_module_name, streaming_module_name)


@pytest.mark.unit
def test_audio_streaming_import_survives_broken_unified_backend(monkeypatch):
    """Streaming route module should import without touching unified backend symbols."""
    module_name = "tldw_Server_API.app.api.v1.endpoints.audio.audio_streaming"
    unified_module_name = "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Unified"
    _install_lightweight_audio_stubs(monkeypatch)
    original_modules = _pop_modules(module_name, unified_module_name)

    class _BrokenUnifiedModule(ModuleType):
        def __getattr__(self, name: str):
            raise RuntimeError(f"unified backend unavailable for {name}")

    broken_unified = _BrokenUnifiedModule(unified_module_name)
    broken_unified.__spec__ = importlib.machinery.ModuleSpec(unified_module_name, loader=None)
    sys.modules[unified_module_name] = broken_unified

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
    unified_module_name = "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Unified"
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
    sys.modules[unified_module_name] = broken_unified

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


@pytest.mark.unit
@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("bound", [False, True])
def test_import_isolation_restores_parent_binding(monkeypatch, cached, bound):
    parent = ModuleType("audio_isolation_parent")
    name = "audio_isolation_parent.child"
    original = ModuleType(name)
    monkeypatch.setitem(sys.modules, parent.__name__, parent)
    if cached:
        monkeypatch.setitem(sys.modules, name, original)
    if bound:
        parent.child = original
    snapshot = _pop_modules(name)
    replacement = ModuleType(name)
    monkeypatch.setitem(sys.modules, name, replacement)
    parent.child = replacement

    _restore_modules(snapshot, name)

    assert sys.modules.get(name) is (original if cached else None)
    assert vars(parent).get("child") is (original if bound else None)
