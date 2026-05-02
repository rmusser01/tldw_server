from pathlib import Path

import pytest
import yaml


@pytest.mark.unit
def test_omnivoice_installer_prefers_local_checkout(tmp_path):
    from Helper_Scripts.TTS_Installers.install_tts_omnivoice_sidecar import resolve_source_checkout

    local_checkout = tmp_path / "OmniVoice"
    local_checkout.mkdir()

    assert resolve_source_checkout(default_probe=local_checkout) == local_checkout.resolve()


@pytest.mark.unit
def test_omnivoice_installer_builds_dedicated_runtime_layout():
    from Helper_Scripts.TTS_Installers.install_tts_omnivoice_sidecar import build_runtime_layout

    repo_root = Path(__file__).resolve().parents[4]
    layout = build_runtime_layout(Path("models") / "omnivoice_sidecar", repo_root=repo_root)

    assert layout.provider_name == "omnivoice"
    assert layout.venv_dir.relative_to(repo_root).as_posix() == "models/omnivoice_sidecar/.venv"
    assert layout.runtime_dir.relative_to(repo_root).as_posix() == "models/omnivoice_sidecar/runtime"
    assert layout.logs_dir.relative_to(repo_root).as_posix() == "models/omnivoice_sidecar/logs"
    assert layout.interpreter_path.relative_to(repo_root).as_posix() == "models/omnivoice_sidecar/.venv/bin/python"


@pytest.mark.unit
def test_omnivoice_installer_creates_runtime_layout(tmp_path):
    from Helper_Scripts.TTS_Installers.install_tts_omnivoice_sidecar import build_runtime_layout, create_runtime_layout

    layout = build_runtime_layout(tmp_path / "models" / "omnivoice_sidecar", repo_root=tmp_path)
    create_runtime_layout(layout)

    assert layout.runtime_base.is_dir()
    assert layout.runtime_dir.is_dir()
    assert layout.logs_dir.is_dir()


@pytest.mark.unit
def test_omnivoice_installer_installs_source_dependencies_by_default(tmp_path, monkeypatch):
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
    assert changed is True
    assert 'kitten_tts:\n    enabled: false' in content
    assert 'omnivoice:\n    enabled: true' in content
    assert 'runtime: "sidecar"' in content
    assert 'python_path: "models/omnivoice_sidecar/.venv/bin/python"' in content
    assert 'runtime_path: "models/omnivoice_sidecar/runtime"' in content
    assert 'logs_path: "models/omnivoice_sidecar/logs"' in content
    assert 'repo_path: "../OmniVoice"' in content


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
