import importlib
import importlib.machinery
import sys
from types import ModuleType

import pytest


def _parent_module_binding(name: str) -> tuple[ModuleType | None, str]:
    parent_name, _, attr = name.rpartition(".")
    parent = sys.modules.get(parent_name)
    return parent if isinstance(parent, ModuleType) else None, attr


def _pop_module(name: str) -> ModuleType | None:
    module = sys.modules.pop(name, None)
    parent, attr = _parent_module_binding(name)
    if parent is not None and hasattr(parent, attr):
        delattr(parent, attr)
    return module if isinstance(module, ModuleType) else None


def _restore_module(name: str, module: ModuleType | None) -> None:
    sys.modules.pop(name, None)
    parent, attr = _parent_module_binding(name)
    if parent is not None and hasattr(parent, attr):
        delattr(parent, attr)
    if module is not None:
        sys.modules[name] = module
        if parent is not None:
            setattr(parent, attr, module)


@pytest.mark.unit
def test_audio_transcription_lib_import_survives_broken_optional_backends(monkeypatch):
    module_name = "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib"
    original_module = _pop_module(module_name)

    class _BrokenModule(ModuleType):
        def __getattr__(self, name: str):
            raise RuntimeError(f"optional backend unavailable for {name}")

    broken_fw = _BrokenModule("faster_whisper")
    broken_fw.__spec__ = importlib.machinery.ModuleSpec("faster_whisper", loader=None)
    broken_tf = _BrokenModule("transformers")
    broken_tf.__spec__ = importlib.machinery.ModuleSpec("transformers", loader=None)
    monkeypatch.setitem(sys.modules, "faster_whisper", broken_fw)
    monkeypatch.setitem(sys.modules, "transformers", broken_tf)

    try:
        imported = importlib.import_module(module_name)
        assert hasattr(imported, "WhisperModel")
        assert hasattr(imported, "load_qwen2audio")
    finally:
        _restore_module(module_name, original_module)
