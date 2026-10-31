import json
from pathlib import Path

import numpy as np
import pytest


pytestmark = pytest.mark.unit


def test_missing_dependency_helpers_preserve_context_and_log(monkeypatch):
    from tldw_Server_API.app.core.TTS.vendors import kittentts_compat as mod

    warnings: list[str] = []

    def fake_warning(message, *args, **_kwargs):
        warnings.append(str(message).format(*args))

    monkeypatch.setattr(mod.logger, "warning", fake_warning)

    exc = ImportError("broken optional dependency")
    mod._log_optional_dependency_fallback("onnxruntime", exc)
    missing = mod._missing_dependency_callable("onnxruntime", "runtime sessions", exc)

    with pytest.raises(ImportError, match="onnxruntime is required for KittenTTS runtime sessions"):
        missing()

    assert warnings
    assert "onnxruntime" in warnings[0]



def test_missing_phonemizer_wrapper_raises_import_error_not_name_error(monkeypatch):
    """The fallback EspeakWrapper runs after its except block has unbound `exc`."""
    import importlib
    import sys

    from tldw_Server_API.app.core.TTS.vendors import kittentts_compat as mod

    monkeypatch.setitem(sys.modules, "phonemizer", None)
    monkeypatch.setitem(sys.modules, "phonemizer.backend.espeak.wrapper", None)
    try:
        fallback = importlib.reload(mod)
        with pytest.raises(ImportError, match="phonemizer is required"):
            fallback.EspeakWrapper.set_library("/nowhere")
        with pytest.raises(ImportError, match="phonemizer is required"):
            fallback.EspeakWrapper.set_data_path("/nowhere")
    finally:
        monkeypatch.undo()
        importlib.reload(mod)

def test_initialize_espeak_paths_uses_espeakng_loader(monkeypatch):
    from tldw_Server_API.app.core.TTS.vendors import kittentts_compat as mod

    calls: list[tuple[str, str]] = []

    class FakeWrapper:
        @staticmethod
        def set_library(path):
            calls.append(("library", path))

        @staticmethod
        def set_data_path(path):
            calls.append(("data", path))

    monkeypatch.setattr(mod, "EspeakWrapper", FakeWrapper)
    monkeypatch.setattr(mod.espeakng_loader, "get_library_path", lambda: "/tmp/libespeak.so")
    monkeypatch.setattr(mod.espeakng_loader, "get_data_path", lambda: "/tmp/espeak-data")

    mod.initialize_espeak_paths()

    assert calls == [
        ("library", "/tmp/libespeak.so"),
        ("data", "/tmp/espeak-data"),
    ]


def test_download_model_assets_uses_config_json_file_selection(monkeypatch, tmp_path):
    from tldw_Server_API.app.core.TTS.vendors import kittentts_compat as mod

    config_path = tmp_path / "config.json"
    model_path = tmp_path / "kitten.onnx"
    voices_path = tmp_path / "voices.npz"

    config_path.write_text(
        json.dumps(
            {
                "type": "ONNX1",
                "model_file": model_path.name,
                "voices": voices_path.name,
                "speed_priors": {"expr-voice-5-m": 1.1},
                "voice_aliases": {"Bella": "expr-voice-2-f"},
            }
        ),
        encoding="utf-8",
    )
    model_path.write_bytes(b"onnx")
    voices_path.write_bytes(b"npz")

    downloads = []

    def fake_hf_hub_download(*, repo_id, filename, cache_dir=None, revision=None, local_files_only=False):
        downloads.append((repo_id, filename, cache_dir, revision, local_files_only))
        mapping = {
            "config.json": config_path,
            model_path.name: model_path,
            voices_path.name: voices_path,
        }
        return str(mapping[filename])

    monkeypatch.setattr(mod, "hf_hub_download", fake_hf_hub_download)

    assets = mod.download_model_assets(
        "kitten-tts-nano-0.8",
        cache_dir=str(tmp_path / "cache"),
        auto_download=False,
    )

    assert assets.repo_id == "KittenML/kitten-tts-nano-0.8-fp32"
    assert assets.revision == mod.PINNED_MODEL_REVISIONS["KittenML/kitten-tts-nano-0.8-fp32"]
    assert assets.model_path == model_path
    assert assets.voices_path == voices_path
    assert assets.voice_aliases["Bella"] == "expr-voice-2-f"
    assert downloads[0] == (
        "KittenML/kitten-tts-nano-0.8-fp32",
        "config.json",
        str(tmp_path / "cache"),
        mod.PINNED_MODEL_REVISIONS["KittenML/kitten-tts-nano-0.8-fp32"],
        True,
    )


def test_download_model_assets_accepts_explicit_commit_revision(monkeypatch, tmp_path):
    from tldw_Server_API.app.core.TTS.vendors import kittentts_compat as mod

    config_path = tmp_path / "config.json"
    model_path = tmp_path / "kitten.onnx"
    voices_path = tmp_path / "voices.npz"

    config_path.write_text(
        json.dumps(
            {
                "type": "ONNX1",
                "model_file": model_path.name,
                "voices": voices_path.name,
            }
        ),
        encoding="utf-8",
    )
    model_path.write_bytes(b"onnx")
    voices_path.write_bytes(b"npz")

    downloads = []

    def fake_hf_hub_download(*, repo_id, filename, cache_dir=None, revision=None, local_files_only=False):
        downloads.append((repo_id, filename, revision, local_files_only))
        mapping = {
            "config.json": config_path,
            model_path.name: model_path,
            voices_path.name: voices_path,
        }
        return str(mapping[filename])

    monkeypatch.setattr(mod, "hf_hub_download", fake_hf_hub_download)

    assets = mod.download_model_assets(
        "KittenML/kitten-tts-mini-0.8",
        auto_download=True,
        revision="deadbeef1",
    )

    assert assets.repo_id == "KittenML/kitten-tts-mini-0.8"
    assert assets.revision == "deadbeef1"
    assert downloads[0] == (
        "KittenML/kitten-tts-mini-0.8",
        "config.json",
        "deadbeef1",
        False,
    )


def test_download_model_assets_rejects_branch_revision(monkeypatch):
    from tldw_Server_API.app.core.TTS.vendors import kittentts_compat as mod

    monkeypatch.setattr(
        mod,
        "hf_hub_download",
        lambda **_kwargs: pytest.fail("revision validation should happen before downloads"),
    )

    with pytest.raises(ValueError, match="immutable Hugging Face commit hash"):
        mod.download_model_assets(
            "KittenML/kitten-tts-mini-0.8",
            auto_download=False,
            revision="main",
        )


def test_kitten_runtime_resolves_display_names_case_insensitively(monkeypatch, tmp_path):
    from tldw_Server_API.app.core.TTS.vendors import kittentts_compat as mod

    assets = mod.KittenModelAssets(
        repo_id="KittenML/kitten-tts-nano-0.8-fp32",
        revision="8d6d5a1851ffd13c894c40227c888302c2a86ef7",
        config_path=tmp_path / "config.json",
        model_path=tmp_path / "kitten.onnx",
        voices_path=tmp_path / "voices.npz",
        speed_priors={},
        voice_aliases={"Bella": "expr-voice-2-f", "Leo": "expr-voice-5-f"},
    )

    monkeypatch.setattr(mod, "initialize_espeak_paths", lambda: None)
    monkeypatch.setattr(mod.ort, "InferenceSession", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        mod.np,
        "load",
        lambda *_args, **_kwargs: {
            "expr-voice-2-f": np.ones((2, 3), dtype=np.float32),
            "expr-voice-5-f": np.ones((2, 3), dtype=np.float32),
        },
    )

    class FakeBackend:
        def phonemize(self, texts):
            return texts

    monkeypatch.setattr(
        mod.phonemizer.backend,
        "EspeakBackend",
        lambda **_kwargs: FakeBackend(),
    )
    monkeypatch.setattr(mod, "TextPreprocessor", lambda **_kwargs: (lambda text: text))

    runtime = mod.KittenRuntime(assets)

    assert runtime.available_voices == ["Bella", "Leo"]
    assert runtime.resolve_voice("bella") == "expr-voice-2-f"
    assert runtime.resolve_voice("Leo") == "expr-voice-5-f"
