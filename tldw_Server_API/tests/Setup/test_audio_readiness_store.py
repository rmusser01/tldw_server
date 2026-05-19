import pytest
import sys

from tldw_Server_API.app.core.Setup import install_manager
from tldw_Server_API.app.core.Setup import audio_readiness_store
from tldw_Server_API.app.core.Setup.audio_readiness_store import AudioReadinessStore


class _CapturingLogger:
    def __init__(self):
        self.records = []

    def debug(self, message, *args, **kwargs):
        self._capture("debug", message, args, kwargs)

    def warning(self, message, *args, **kwargs):
        self._capture("warning", message, args, kwargs)

    def _capture(self, level, message, args, kwargs):
        captured_kwargs = dict(kwargs)
        if captured_kwargs.get("exc_info"):
            captured_kwargs["exc_info_repr"] = repr(sys.exc_info()[1])
        self.records.append((level, message, args, captured_kwargs))


def _joined_logs(logger):
    return "\n".join(f"{level} {message} {args!r} {kwargs!r}" for level, message, args, kwargs in logger.records)


def test_audio_readiness_defaults_to_not_started(tmp_path):
    store = AudioReadinessStore(tmp_path / "audio_readiness.json")

    readiness = store.load()

    assert readiness["status"] == "not_started"
    assert readiness["selected_bundle_id"] is None
    assert readiness["selected_resource_profile"] == "balanced"


def test_audio_readiness_update_persists_to_disk(tmp_path):
    store = AudioReadinessStore(tmp_path / "audio_readiness.json")

    store.update(
        status="provisioning",
        selected_bundle_id="cpu_local",
        selected_resource_profile="light",
        tts_choice="kitten_tts",
        selection_key="v2:cpu_local:light:kitten_tts",
        remediation_items=["Verification still pending"],
    )

    reloaded = AudioReadinessStore(tmp_path / "audio_readiness.json").load()

    assert reloaded["status"] == "provisioning"
    assert reloaded["selected_bundle_id"] == "cpu_local"
    assert reloaded["selected_resource_profile"] == "light"
    assert reloaded["tts_choice"] == "kitten_tts"
    assert reloaded["selection_key"] == "v2:cpu_local:light:kitten_tts"
    assert reloaded["remediation_items"] == ["Verification still pending"]


def test_readiness_defaults_missing_profile_to_balanced(tmp_path):
    readiness_path = tmp_path / "audio_readiness.json"
    readiness_path.write_text('{"status": "ready", "selected_bundle_id": "cpu_local"}', encoding="utf-8")

    readiness = AudioReadinessStore(readiness_path).load()

    assert readiness["selected_resource_profile"] == "balanced"


def test_readiness_canonicalizes_default_tts_choice_identity(tmp_path):
    readiness_path = tmp_path / "audio_readiness.json"
    readiness_path.write_text(
        (
            '{"status":"ready","selected_bundle_id":"cpu_local","selected_resource_profile":"balanced",'
            '"tts_choice":"kokoro","selection_key":"v2:cpu_local:balanced:kokoro"}'
        ),
        encoding="utf-8",
    )

    readiness = AudioReadinessStore(readiness_path).load()

    assert readiness["tts_choice"] is None
    assert readiness["selection_key"] == "v2:cpu_local:balanced"


def test_readiness_save_rewrites_stale_selection_key_to_canonical_identity(tmp_path):
    store = AudioReadinessStore(tmp_path / "audio_readiness.json")

    saved = store.save(
        {
            "status": "ready",
            "selected_bundle_id": "cpu_local",
            "selected_resource_profile": "balanced",
            "tts_choice": "kokoro",
            "selection_key": "v2:cpu_local:balanced:kokoro",
        }
    )

    assert saved["tts_choice"] is None
    assert saved["selection_key"] == "v2:cpu_local:balanced"


def test_readiness_save_does_not_swallow_unexpected_catalog_errors(tmp_path, monkeypatch):
    store = AudioReadinessStore(tmp_path / "audio_readiness.json")

    class _BrokenCatalog:
        def bundle_by_id(self, bundle_id):
            raise RuntimeError(f"broken catalog lookup for {bundle_id}")

    monkeypatch.setattr(audio_readiness_store, "get_audio_bundle_catalog", lambda: _BrokenCatalog())

    with pytest.raises(RuntimeError, match="broken catalog lookup"):
        store.save(
            {
                "status": "ready",
                "selected_bundle_id": "cpu_local",
                "selected_resource_profile": "balanced",
                "tts_choice": "kokoro",
            }
        )


def test_install_plan_success_marks_audio_readiness_partial(tmp_path, mocker):
    store = AudioReadinessStore(tmp_path / "audio_readiness.json")
    plan_payload = {
        "stt": [{"engine": "faster_whisper", "models": ["medium"]}],
        "tts": [],
        "embeddings": {
            "huggingface": [],
            "custom": [],
            "onnx": [],
        },
    }

    mocker.patch.object(
        install_manager.audio_readiness_store,
        "get_audio_readiness_store",
        return_value=store,
    )
    mocker.patch.object(install_manager, "_install_dependencies")
    mocker.patch.object(install_manager, "_install_stt")
    mocker.patch.object(install_manager, "_install_tts")
    mocker.patch.object(install_manager, "_install_embeddings")

    install_manager.execute_install_plan(plan_payload)

    readiness = store.load()
    assert readiness["status"] == "partial"
    assert readiness["remediation_items"] == ["Run audio verification to confirm readiness."]


def test_audio_readiness_save_keeps_existing_file_when_atomic_replace_fails(tmp_path, monkeypatch):
    readiness_path = tmp_path / "audio_readiness.json"
    store = AudioReadinessStore(readiness_path)
    store.update(status="ready", selected_bundle_id="cpu_local")
    initial_contents = readiness_path.read_text(encoding="utf-8")

    def _raise_replace(_src, _dst):
        raise OSError("replace failed")

    monkeypatch.setattr(audio_readiness_store.os, "replace", _raise_replace)

    with pytest.raises(OSError, match="replace failed"):
        store.update(status="failed")

    assert readiness_path.read_text(encoding="utf-8") == initial_contents
    assert list(tmp_path.glob("audio_readiness.json.*.tmp")) == []


def test_resolve_readiness_file_candidate_failure_log_is_sanitized(tmp_path, monkeypatch):
    readiness_path = tmp_path / "private" / "setup_audio_readiness.json"
    logger = _CapturingLogger()

    def _raise_write_text(self, *_args, **_kwargs):
        raise OSError(f"write denied at {readiness_path}")

    monkeypatch.setattr(audio_readiness_store, "logger", logger)
    monkeypatch.setattr(audio_readiness_store, "_candidate_readiness_files", lambda: [readiness_path])
    monkeypatch.setattr(audio_readiness_store.Path, "write_text", _raise_write_text)

    assert audio_readiness_store._resolve_readiness_file() is None

    logs = _joined_logs(logger)
    assert "Audio readiness path" not in logs
    assert "Audio readiness candidate path not writable" in logs
    assert str(readiness_path) not in logs
    assert "write denied" not in logs
    assert "exc_info" not in logs


def test_audio_readiness_load_failure_log_is_sanitized(tmp_path, monkeypatch):
    readiness_path = tmp_path / "private" / "setup_audio_readiness.json"
    readiness_path.parent.mkdir()
    readiness_path.write_text("{invalid-json", encoding="utf-8")
    logger = _CapturingLogger()

    monkeypatch.setattr(audio_readiness_store, "logger", logger)

    readiness = AudioReadinessStore(readiness_path).load()

    assert readiness["status"] == "not_started"
    logs = _joined_logs(logger)
    assert "Failed to read audio readiness" in logs
    assert str(readiness_path) not in logs
    assert "Expecting property name" not in logs
    assert "exc_info" not in logs
