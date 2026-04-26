from pathlib import Path

import pytest
import yaml


@pytest.mark.unit
def test_omnivoice_installer_prefers_local_checkout(tmp_path):
    from Helper_Scripts.TTS_Installers.install_tts_omnivoice_sidecar import resolve_source_checkout

    local_checkout = tmp_path / "OmniVoice"
    local_checkout.mkdir()

    assert resolve_source_checkout(default_probe=local_checkout) == local_checkout.resolve()  # nosec B101


@pytest.mark.unit
def test_omnivoice_installer_builds_dedicated_runtime_layout():
    from Helper_Scripts.TTS_Installers.install_tts_omnivoice_sidecar import build_runtime_layout

    repo_root = Path(__file__).resolve().parents[4]
    layout = build_runtime_layout(Path("models") / "omnivoice_sidecar", repo_root=repo_root)

    assert layout.provider_name == "omnivoice"  # nosec B101
    assert layout.venv_dir.relative_to(repo_root).as_posix() == "models/omnivoice_sidecar/.venv"  # nosec B101
    assert layout.runtime_dir.relative_to(repo_root).as_posix() == "models/omnivoice_sidecar/runtime"  # nosec B101
    assert layout.logs_dir.relative_to(repo_root).as_posix() == "models/omnivoice_sidecar/logs"  # nosec B101
    assert layout.interpreter_path.relative_to(repo_root).as_posix() == "models/omnivoice_sidecar/.venv/bin/python"  # nosec B101


@pytest.mark.unit
def test_omnivoice_installer_creates_runtime_layout(tmp_path):
    from Helper_Scripts.TTS_Installers.install_tts_omnivoice_sidecar import build_runtime_layout, create_runtime_layout

    layout = build_runtime_layout(tmp_path / "models" / "omnivoice_sidecar", repo_root=tmp_path)
    create_runtime_layout(layout)

    assert layout.runtime_base.is_dir()  # nosec B101
    assert layout.runtime_dir.is_dir()  # nosec B101
    assert layout.logs_dir.is_dir()  # nosec B101


@pytest.mark.unit
def test_omnivoice_installer_creates_posix_virtualenv_with_symlinks(tmp_path, monkeypatch):
    import os

    from Helper_Scripts.TTS_Installers import install_tts_omnivoice_sidecar as installer

    created = {}

    class _FakeBuilder:
        def __init__(self, *, with_pip, symlinks):
            created["with_pip"] = with_pip
            created["symlinks"] = symlinks

        def create(self, target):
            created["target"] = Path(target)

    monkeypatch.setattr(installer.venv, "EnvBuilder", _FakeBuilder)

    venv_dir = tmp_path / ".venv"
    installer.create_virtualenv(venv_dir)

    assert created["with_pip"] is True  # nosec B101
    assert created["symlinks"] is (os.name != "nt")  # nosec B101
    assert created["target"] == venv_dir  # nosec B101


@pytest.mark.unit
def test_omnivoice_installer_archives_broken_virtualenv_before_recreate(tmp_path, monkeypatch):
    from Helper_Scripts.TTS_Installers import install_tts_omnivoice_sidecar as installer

    events = {}

    class _FakeBuilder:
        def __init__(self, *, with_pip, symlinks):
            events["with_pip"] = with_pip
            events["symlinks"] = symlinks

        def create(self, target):
            events["created_target"] = Path(target)

    def _fake_archive(target, *, reason):
        events["archived_target"] = Path(target)
        events["archived_reason"] = reason
        return target.with_name(f"{target.name}.{reason}-stub")

    monkeypatch.setattr(installer, "is_virtualenv_interpreter_usable", lambda _path: False)
    monkeypatch.setattr(installer, "archive_existing_virtualenv", _fake_archive)
    monkeypatch.setattr(installer.venv, "EnvBuilder", _FakeBuilder)

    venv_dir = tmp_path / ".venv"
    venv_dir.mkdir()

    installer.create_virtualenv(venv_dir)

    assert events["archived_target"] == venv_dir  # nosec B101
    assert events["archived_reason"] == "broken"  # nosec B101
    assert events["with_pip"] is True  # nosec B101
    assert events["created_target"] == venv_dir  # nosec B101


@pytest.mark.unit
def test_omnivoice_installer_recreate_flag_forces_virtualenv_rebuild(tmp_path, monkeypatch):
    from Helper_Scripts.TTS_Installers import install_tts_omnivoice_sidecar as installer

    events = {}

    class _FakeBuilder:
        def __init__(self, *, with_pip, symlinks):
            events["with_pip"] = with_pip
            events["symlinks"] = symlinks

        def create(self, target):
            events["created_target"] = Path(target)

    def _fake_archive(target, *, reason):
        events["archived_target"] = Path(target)
        events["archived_reason"] = reason
        return target.with_name(f"{target.name}.{reason}-stub")

    monkeypatch.setattr(installer, "is_virtualenv_interpreter_usable", lambda _path: True)
    monkeypatch.setattr(installer, "archive_existing_virtualenv", _fake_archive)
    monkeypatch.setattr(installer.venv, "EnvBuilder", _FakeBuilder)

    venv_dir = tmp_path / ".venv"
    venv_dir.mkdir()

    installer.create_virtualenv(venv_dir, recreate=True)

    assert events["archived_target"] == venv_dir  # nosec B101
    assert events["archived_reason"] == "recreate"  # nosec B101
    assert events["with_pip"] is True  # nosec B101
    assert events["created_target"] == venv_dir  # nosec B101


@pytest.mark.unit
def test_omnivoice_installer_parse_args_supports_recreate_venv():
    from Helper_Scripts.TTS_Installers.install_tts_omnivoice_sidecar import parse_args

    args = parse_args(["--recreate-venv"])

    assert args.recreate_venv is True  # nosec B101


@pytest.mark.unit
def test_omnivoice_installer_parse_args_supports_install_inference_deps():
    from Helper_Scripts.TTS_Installers.install_tts_omnivoice_sidecar import parse_args

    args = parse_args(["--install-inference-deps"])

    assert args.install_inference_deps is True  # nosec B101


@pytest.mark.unit
def test_omnivoice_installer_installs_source_with_dependencies_when_requested(tmp_path, monkeypatch):
    from Helper_Scripts.TTS_Installers import install_tts_omnivoice_sidecar as installer

    commands = []
    source_checkout = tmp_path / "OmniVoice"
    source_checkout.mkdir()

    def _fake_run(command, **kwargs):  # noqa: ARG001
        commands.append(command)

    monkeypatch.setattr(installer.subprocess, "run", _fake_run)

    installer.install_sidecar_runtime(
        interpreter_path=tmp_path / ".venv" / "bin" / "python",
        repo_root=tmp_path,
        source_checkout=source_checkout,
        install_inference_deps=True,
    )

    editable_install = commands[-1]
    assert "-e" in editable_install  # nosec B101
    assert "--no-deps" not in editable_install  # nosec B101
    assert str(source_checkout) in editable_install  # nosec B101


@pytest.mark.unit
def test_omnivoice_installer_updates_only_provider_block(tmp_path):
    from Helper_Scripts.TTS_Installers.install_tts_omnivoice_sidecar import (
        build_runtime_layout,
        patch_tts_config,
    )

    config_path = tmp_path / "tts_providers_config.yaml"
    config_path.write_text(
        """
providers:
  kitten_tts:
    enabled: false
    model: "KittenML/kitten-tts-nano-0.8"

  omnivoice:
    enabled: false
    runtime: "sidecar"
    extra_params:
      repo_path: "../old-OmniVoice"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    layout = build_runtime_layout(Path("models") / "omnivoice_sidecar", repo_root=tmp_path)
    source_checkout = tmp_path.parent / "OmniVoice"
    source_checkout.mkdir(exist_ok=True)

    changed = patch_tts_config(
        config_path=config_path,
        layout=layout,
        source_checkout=source_checkout,
        repo_root=tmp_path,
    )

    content = config_path.read_text(encoding="utf-8")
    assert changed is True  # nosec B101
    assert 'kitten_tts:\n    enabled: false' in content  # nosec B101
    assert 'omnivoice:\n    enabled: true' in content  # nosec B101
    assert 'runtime: "sidecar"' in content  # nosec B101
    assert 'runtime_mode: "real"' in content  # nosec B101
    assert 'model_id: "k2-fsa/OmniVoice"' in content  # nosec B101
    assert 'python_path: "models/omnivoice_sidecar/.venv/bin/python"' in content  # nosec B101
    assert 'runtime_path: "models/omnivoice_sidecar/runtime"' in content  # nosec B101
    assert 'logs_path: "models/omnivoice_sidecar/logs"' in content  # nosec B101
    assert 'repo_path: "../OmniVoice"' in content  # nosec B101


@pytest.mark.unit
def test_omnivoice_installer_inserts_valid_yaml_when_provider_block_is_missing(tmp_path):
    from Helper_Scripts.TTS_Installers.install_tts_omnivoice_sidecar import (
        build_runtime_layout,
        patch_tts_config,
    )

    config_path = tmp_path / "tts_providers_config.yaml"
    config_path.write_text(
        """
providers:
  kitten_tts:
    enabled: false
""".strip()
        + "\n",
        encoding="utf-8",
    )
    layout = build_runtime_layout(Path("models") / "omnivoice_sidecar", repo_root=tmp_path)
    source_checkout = tmp_path / "external" / "OmniVoice"
    source_checkout.mkdir(parents=True)

    changed = patch_tts_config(
        config_path=config_path,
        layout=layout,
        source_checkout=source_checkout,
        repo_root=tmp_path,
    )

    content = config_path.read_text(encoding="utf-8")
    parsed = yaml.safe_load(content)

    assert changed is True
    assert "  omnivoice:" in content
    assert parsed["providers"]["omnivoice"]["enabled"] is True
    assert parsed["providers"]["omnivoice"]["extra_params"]["repo_path"] == "external/OmniVoice"


@pytest.mark.unit
def test_omnivoice_installer_skips_complex_yaml_constructs(tmp_path):
    from Helper_Scripts.TTS_Installers.install_tts_omnivoice_sidecar import (
        build_runtime_layout,
        patch_tts_config,
    )

    config_path = tmp_path / "tts_providers_config.yaml"
    original = (
        """
providers:
  kitten_tts: # keep legacy comment
    enabled: false
""".strip()
        + "\n"
    )
    config_path.write_text(original, encoding="utf-8")
    layout = build_runtime_layout(Path("models") / "omnivoice_sidecar", repo_root=tmp_path)
    source_checkout = tmp_path / "external" / "OmniVoice"
    source_checkout.mkdir(parents=True)

    changed = patch_tts_config(
        config_path=config_path,
        layout=layout,
        source_checkout=source_checkout,
        repo_root=tmp_path,
    )

    assert changed is False
    assert config_path.read_text(encoding="utf-8") == original


@pytest.mark.unit
def test_omnivoice_installer_rejects_malformed_repo_urls(tmp_path, monkeypatch):
    from Helper_Scripts.TTS_Installers.install_tts_omnivoice_sidecar import clone_repository

    monkeypatch.setattr(
        "Helper_Scripts.TTS_Installers.install_tts_omnivoice_sidecar._run_checked_command",
        lambda command, cwd=None: pytest.fail(f"clone should not run for invalid repo URL: {command}"),  # noqa: ARG005
        raising=True,
    )
    monkeypatch.setattr(
        "Helper_Scripts.TTS_Installers.install_tts_omnivoice_sidecar.shutil.which",
        lambda name: "/usr/bin/git" if name == "git" else None,
        raising=True,
    )

    with pytest.raises(SystemExit, match="Invalid repository URL"):
        clone_repository("https://github.com/example/repo.git; rm -rf /", tmp_path / "OmniVoice")
