import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types

import pytest

from tldw_Server_API.app.core.Setup import install_manager
from tldw_Server_API.app.core.Setup.install_schema import InstallPlan, TTSInstall


class _CapturingLogger:
    def __init__(self):
        self.records = []

    def debug(self, message, *args, **kwargs):
        self.records.append(("debug", message, args, dict(kwargs)))

    def warning(self, message, *args, **kwargs):
        self.records.append(("warning", message, args, dict(kwargs)))


def _joined_records(logger: _CapturingLogger) -> str:
    return "\n".join(f"{level} {message} {args!r} {kwargs!r}" for level, message, args, kwargs in logger.records)


@pytest.fixture(autouse=True)
def reset_dependency_cache():
    install_manager._INSTALLED_DEPENDENCIES.clear()  # noqa: SLF001
    yield
    install_manager._INSTALLED_DEPENDENCIES.clear()  # noqa: SLF001


def _read_status(path: str):
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def test_install_status_file_candidate_failure_log_is_sanitized(tmp_path, monkeypatch):
    status_dir = tmp_path / "private" / "install-state"
    logger = _CapturingLogger()
    original_open = Path.open

    def _raise_write_probe(self, *args, **kwargs):
        if self.name == ".write_test":
            raise OSError(f"install status write denied at {self}")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(install_manager, "logger", logger)
    monkeypatch.setattr(install_manager, "_candidate_status_dirs", lambda: [status_dir])
    monkeypatch.setattr(install_manager.Path, "open", _raise_write_probe)

    assert install_manager._resolve_status_file() is None

    joined = _joined_records(logger)
    assert "Install status directory not writable" in joined
    assert str(status_dir) not in joined
    assert "install status write denied" not in joined
    assert "exc_info" not in joined


def test_dependencies_skipped_when_pip_disabled(monkeypatch):


    plan = {
        'stt': [{'engine': 'faster_whisper', 'models': ['small']}],
        'tts': [],
        'embeddings': {'huggingface': [], 'custom': [], 'onnx': []},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setenv('TLDW_INSTALL_STATE_DIR', tmpdir)
        monkeypatch.setenv('TLDW_SETUP_SKIP_PIP', '1')
        monkeypatch.setenv('TLDW_SETUP_SKIP_DOWNLOADS', '1')

        executed = []

        def fake_subprocess(cmd, check=False, capture_output=True, text=True):  # noqa: ARG001
            executed.append(cmd)
            return

        monkeypatch.setattr(install_manager, '_run_subprocess', fake_subprocess)

        install_manager.execute_install_plan(plan)

        status_path = os.path.join(tmpdir, install_manager.STATUS_FILENAME)
        payload = _read_status(status_path)
        step_names = {entry['name']: entry['status'] for entry in payload['steps']}

        assert step_names.get('deps:stt:faster_whisper') in {'skipped', 'completed'}
        assert not executed, "Subprocess should not run when pip is disabled"


def test_dependencies_trigger_pip_install(monkeypatch):


    plan = {
        'stt': [{'engine': 'faster_whisper', 'models': ['small']}],
        'tts': [],
        'embeddings': {'huggingface': [], 'custom': [], 'onnx': []},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setenv('TLDW_INSTALL_STATE_DIR', tmpdir)
        monkeypatch.delenv('TLDW_SETUP_SKIP_PIP', raising=False)
        monkeypatch.setenv('TLDW_SETUP_SKIP_DOWNLOADS', '1')

        commands = []

        original_find_spec = importlib.util.find_spec

        def fake_find_spec(name):

            if name == 'faster_whisper':
                return None
            return original_find_spec(name)

        monkeypatch.setattr(importlib.util, 'find_spec', fake_find_spec)

        def fake_subprocess(cmd, check=False, capture_output=True, text=True):  # noqa: ARG001
            commands.append(cmd)
            return

        monkeypatch.setattr(install_manager, '_run_subprocess', fake_subprocess)

        install_manager.execute_install_plan(plan)

        assert commands, "Expected pip install command to execute"
        pip_cmd = commands[0]
        assert pip_cmd[:4] == [install_manager.sys.executable, '-m', 'pip', 'install']
        assert any('faster-whisper' in part for part in pip_cmd)


def test_install_plan_accepts_kitten_tts():
    plan = InstallPlan(tts=[TTSInstall(engine='kitten_tts', variants=['nano'])])

    assert plan.tts[0].engine == 'kitten_tts'
    assert plan.tts[0].variants == ['nano']


def test_install_plan_accepts_omnivoice():
    plan = InstallPlan(tts=[TTSInstall(engine='omnivoice')])

    assert plan.tts[0].engine == 'omnivoice'
    assert plan.tts[0].variants == []


def test_kitten_tts_dependencies_trigger_pip_install(monkeypatch):

    plan = {
        'stt': [],
        'tts': [{'engine': 'kitten_tts', 'variants': ['nano']}],
        'embeddings': {'huggingface': [], 'custom': [], 'onnx': []},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setenv('TLDW_INSTALL_STATE_DIR', tmpdir)
        monkeypatch.delenv('TLDW_SETUP_SKIP_PIP', raising=False)
        monkeypatch.setenv('TLDW_SETUP_SKIP_DOWNLOADS', '1')

        commands = []

        original_find_spec = importlib.util.find_spec

        def fake_find_spec(name):
            if name == 'phonemizer':
                return object()
            if name in {'espeakng_loader', 'huggingface_hub'}:
                return None
            return original_find_spec(name)

        monkeypatch.setattr(importlib.util, 'find_spec', fake_find_spec)
        monkeypatch.setattr(install_manager, '_install_kitten_tts', lambda _variants: None, raising=False)

        def fake_subprocess(cmd, check=False, capture_output=True, text=True):  # noqa: ARG001
            commands.append(cmd)
            return

        monkeypatch.setattr(install_manager, '_run_subprocess', fake_subprocess)

        install_manager.execute_install_plan(plan)

        assert commands, "Expected pip install commands for KittenTTS dependencies"
        flattened = ' '.join(' '.join(cmd) for cmd in commands)
        assert 'phonemizer-fork' in flattened
        assert 'espeakng_loader' in flattened


def test_install_kitten_tts_rejects_unknown_variants(monkeypatch):
    monkeypatch.setattr(install_manager, "_ensure_downloads_allowed", lambda _label: None)
    monkeypatch.setattr(
        install_manager,
        "_resolve_kitten_tts_prefetch_settings",
        lambda: {"cache_dir": "cache/kitten_tts", "revision": None},
        raising=False,
    )

    fake_module = types.SimpleNamespace(
        download_model_assets=lambda *_args, **_kwargs: pytest.fail("unknown variants should fail before downloads")
    )
    monkeypatch.setitem(
        sys.modules,
        "tldw_Server_API.app.core.TTS.vendors.kittentts_compat",
        fake_module,
    )

    with pytest.raises(ValueError, match="Unsupported KittenTTS variants: custom"):
        install_manager._install_kitten_tts(["custom"])


def test_cuda_available_requires_successful_nvidia_probe(monkeypatch):
    monkeypatch.delenv("TLDW_SETUP_FORCE_CPU", raising=False)
    monkeypatch.delenv("TLDW_SETUP_FORCE_GPU", raising=False)
    monkeypatch.setenv("CUDA_HOME", "/opt/cuda")
    monkeypatch.setattr(install_manager.shutil, "which", lambda _name: None)

    def fake_run(*_args, **_kwargs):
        raise AssertionError("nvidia-smi probe should not run when it is unavailable")

    monkeypatch.setattr(install_manager.subprocess, "run", fake_run)

    assert install_manager._cuda_available() is False


def test_cuda_available_accepts_verified_nvidia_smi(monkeypatch):
    monkeypatch.delenv("TLDW_SETUP_FORCE_CPU", raising=False)
    monkeypatch.delenv("TLDW_SETUP_FORCE_GPU", raising=False)
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.setattr(install_manager.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")

    def fake_run(cmd, check, capture_output, text, timeout):  # noqa: ARG001
        assert cmd == ["/usr/bin/nvidia-smi", "-L"]
        assert timeout == 3
        return types.SimpleNamespace(returncode=0, stdout="GPU 0: Test GPU\n", stderr="")

    monkeypatch.setattr(install_manager.subprocess, "run", fake_run)

    assert install_manager._cuda_available() is True


def test_install_kitten_tts_prefetch_uses_configured_cache_dir(monkeypatch):
    monkeypatch.setattr(install_manager, "_ensure_downloads_allowed", lambda _label: None)
    monkeypatch.setattr(
        install_manager,
        "_resolve_kitten_tts_prefetch_settings",
        lambda: {"cache_dir": "cache/kitten_tts", "revision": None},
        raising=False,
    )

    download_calls: list[tuple[str, str | None, bool, str | None]] = []

    fake_module = types.SimpleNamespace(
        download_model_assets=lambda repo_id, *, cache_dir=None, auto_download=True, revision=None: download_calls.append(
            (repo_id, cache_dir, auto_download, revision)
        )
    )
    monkeypatch.setitem(
        sys.modules,
        "tldw_Server_API.app.core.TTS.vendors.kittentts_compat",
        fake_module,
    )

    install_manager._install_kitten_tts(["nano"])

    assert download_calls == [
        ("KittenML/kitten-tts-nano-0.8", "cache/kitten_tts", True, None)
    ]


def _patch_omnivoice_sidecar_installer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    config_patched: bool,
) -> list[tuple[str, object]]:
    from Helper_Scripts.TTS_Installers import install_tts_omnivoice_sidecar as installer

    calls: list[tuple[str, object]] = []
    source_checkout = tmp_path / "OmniVoice"
    model_path = tmp_path / "models" / "OmniVoice"
    model_path.mkdir(parents=True)
    monkeypatch.setenv("TLDW_OMNIVOICE_MODEL_PATH", str(model_path))

    monkeypatch.setattr(
        installer,
        "resolve_source_checkout",
        lambda repo_root=None: calls.append(("resolve_source_checkout", repo_root)) or source_checkout,
        raising=True,
    )
    monkeypatch.setattr(
        installer,
        "build_runtime_layout",
        lambda runtime_base, repo_root=None: calls.append(("build_runtime_layout", runtime_base)) or types.SimpleNamespace(
            runtime_base=Path(runtime_base),
            venv_dir=Path(runtime_base) / ".venv",
            runtime_dir=Path(runtime_base) / "runtime",
            logs_dir=Path(runtime_base) / "logs",
            interpreter_path=Path(runtime_base) / ".venv" / "bin" / "python",
        ),
        raising=True,
    )
    monkeypatch.setattr(
        installer,
        "create_runtime_layout",
        lambda layout: calls.append(("create_runtime_layout", layout.runtime_base)),
        raising=True,
    )
    monkeypatch.setattr(
        installer,
        "clone_repository",
        lambda repo_url, source_dir: calls.append(("clone_repository", repo_url, source_dir)),
        raising=True,
    )
    monkeypatch.setattr(
        installer,
        "create_virtualenv",
        lambda venv_dir: calls.append(("create_virtualenv", venv_dir)),
        raising=True,
    )
    monkeypatch.setattr(
        installer,
        "install_sidecar_runtime",
        lambda *, interpreter_path, repo_root, source_checkout: calls.append(
            ("install_sidecar_runtime", interpreter_path, repo_root, source_checkout)
        ),
        raising=True,
    )
    monkeypatch.setattr(
        installer,
        "validate_runtime_layout",
        lambda layout: calls.append(("validate_runtime_layout", layout.runtime_base)) or [],
        raising=True,
    )
    monkeypatch.setattr(
        installer,
        "validate_local_model_path",
        lambda path: calls.append(("validate_local_model_path", path)) or path,
        raising=True,
    )
    monkeypatch.setattr(
        installer,
        "patch_tts_config",
        lambda **kwargs: calls.append(("patch_tts_config", kwargs["config_path"], kwargs["model_path"])) or config_patched,
        raising=True,
    )

    return calls


def test_omnivoice_install_routes_to_sidecar_installer(monkeypatch, tmp_path):
    calls = _patch_omnivoice_sidecar_installer(
        monkeypatch,
        tmp_path,
        config_patched=True,
    )

    install_manager._install_omnivoice()

    assert [entry[0] for entry in calls] == [
        "validate_local_model_path",
        "resolve_source_checkout",
        "build_runtime_layout",
        "create_runtime_layout",
        "clone_repository",
        "create_virtualenv",
        "install_sidecar_runtime",
        "validate_runtime_layout",
        "patch_tts_config",
    ]
    patch_call = calls[-1]
    assert patch_call[0] == "patch_tts_config"
    assert patch_call[2] == tmp_path / "models" / "OmniVoice"


def test_omnivoice_install_fails_when_config_patch_does_not_update_provider(
    monkeypatch,
    tmp_path,
):
    calls = _patch_omnivoice_sidecar_installer(
        monkeypatch,
        tmp_path,
        config_patched=False,
    )

    with pytest.raises(RuntimeError, match="provider configuration could not be updated"):
        install_manager._install_omnivoice()

    assert [entry[0] for entry in calls][-1] == "patch_tts_config"


def test_omnivoice_install_requires_explicit_model_path(monkeypatch):
    monkeypatch.setattr(install_manager, "_ensure_downloads_allowed", lambda _label: None, raising=True)
    monkeypatch.delenv("TLDW_OMNIVOICE_MODEL_PATH", raising=False)

    with pytest.raises(RuntimeError, match="TLDW_OMNIVOICE_MODEL_PATH"):
        install_manager._install_omnivoice()


def test_omnivoice_install_rejects_invalid_model_path(monkeypatch, tmp_path):
    from Helper_Scripts.TTS_Installers import install_tts_omnivoice_sidecar as installer

    missing_model_path = tmp_path / "missing"

    monkeypatch.setattr(install_manager, "_ensure_downloads_allowed", lambda _label: None, raising=True)
    monkeypatch.setenv("TLDW_OMNIVOICE_MODEL_PATH", str(missing_model_path))
    monkeypatch.setattr(
        installer,
        "resolve_source_checkout",
        lambda repo_root=None: pytest.fail("source checkout should not resolve for invalid model path"),
        raising=True,
    )

    with pytest.raises(RuntimeError, match="OmniVoice model path"):
        install_manager._install_omnivoice()


def test_omnivoice_install_checks_download_policy_before_running_installer(monkeypatch):
    calls: list[tuple[str, object]] = []

    def _blocked(label):
        calls.append(("ensure_downloads_allowed", label))
        raise install_manager.DownloadBlockedError("downloads blocked for test")

    monkeypatch.setattr(install_manager, "_ensure_downloads_allowed", _blocked, raising=True)

    with pytest.raises(install_manager.DownloadBlockedError, match="downloads blocked for test"):
        install_manager._install_omnivoice()

    assert calls == [("ensure_downloads_allowed", "OmniVoice sidecar runtime")]
