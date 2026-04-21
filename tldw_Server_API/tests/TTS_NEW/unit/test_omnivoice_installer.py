from pathlib import Path

import pytest


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
    source_checkout.mkdir()

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
