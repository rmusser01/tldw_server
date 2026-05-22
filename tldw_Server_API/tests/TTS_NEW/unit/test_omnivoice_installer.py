from pathlib import Path, PureWindowsPath

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
    model_path = tmp_path / "models" / "OmniVoice"
    model_path.mkdir(parents=True)

    changed = patch_tts_config(
        config_path=config_path,
        layout=layout,
        source_checkout=source_checkout,
        model_path=model_path,
        repo_root=tmp_path,
    )

    content = config_path.read_text(encoding="utf-8")
    parsed = yaml.safe_load(content)
    assert changed is True
    assert 'kitten_tts:\n    enabled: false' in content
    assert 'omnivoice:\n    enabled: true' in content
    assert 'runtime: "sidecar"' in content
    assert 'python_path: "models/omnivoice_sidecar/.venv/bin/python"' in content
    assert 'runtime_path: "models/omnivoice_sidecar/runtime"' in content
    assert 'scratch_dir: "models/omnivoice_sidecar/runtime/scratch"' in content
    assert 'logs_path: "models/omnivoice_sidecar/logs"' in content
    assert 'repo_path: "../OmniVoice"' in content
    extra_params = parsed["providers"]["omnivoice"]["extra_params"]
    assert extra_params["model_path"] == "models/OmniVoice"
    assert extra_params["runtime_path"] == "models/omnivoice_sidecar/runtime"
    assert extra_params["scratch_dir"] == "models/omnivoice_sidecar/runtime/scratch"


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
    model_path = tmp_path / "models" / "OmniVoice"
    model_path.mkdir(parents=True)

    changed = patch_tts_config(
        config_path=config_path,
        layout=layout,
        source_checkout=source_checkout,
        model_path=model_path,
        repo_root=tmp_path,
    )

    content = config_path.read_text(encoding="utf-8")
    parsed = yaml.safe_load(content)

    assert changed is True
    assert "  omnivoice:" in content
    assert parsed["providers"]["omnivoice"]["enabled"] is True
    assert parsed["providers"]["omnivoice"]["extra_params"]["repo_path"] == "external/OmniVoice"
    assert parsed["providers"]["omnivoice"]["extra_params"]["model_path"] == "models/OmniVoice"


@pytest.mark.unit
def test_omnivoice_installer_rejects_unsupported_flow_style_providers_without_duplicate(tmp_path):
    from Helper_Scripts.TTS_Installers.install_tts_omnivoice_sidecar import (
        build_runtime_layout,
        patch_tts_config,
    )

    config_path = tmp_path / "tts_providers_config.yaml"
    original = 'providers: { kitten_tts: { enabled: false } }\n'
    config_path.write_text(original, encoding="utf-8")
    layout = build_runtime_layout(Path("models") / "omnivoice_sidecar", repo_root=tmp_path)
    source_checkout = tmp_path / "external" / "OmniVoice"
    source_checkout.mkdir(parents=True)
    model_path = tmp_path / "models" / "OmniVoice"
    model_path.mkdir(parents=True)

    changed = patch_tts_config(
        config_path=config_path,
        layout=layout,
        source_checkout=source_checkout,
        model_path=model_path,
        repo_root=tmp_path,
    )

    assert changed is False
    assert config_path.read_text(encoding="utf-8") == original


@pytest.mark.unit
def test_omnivoice_installer_patches_when_comments_are_outside_provider_block(tmp_path):
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

  omnivoice:
    enabled: false
""".strip()
        + "\n"
    )
    config_path.write_text(original, encoding="utf-8")
    layout = build_runtime_layout(Path("models") / "omnivoice_sidecar", repo_root=tmp_path)
    source_checkout = tmp_path / "external" / "OmniVoice"
    source_checkout.mkdir(parents=True)
    model_path = tmp_path / "models" / "OmniVoice"
    model_path.mkdir(parents=True)

    changed = patch_tts_config(
        config_path=config_path,
        layout=layout,
        source_checkout=source_checkout,
        model_path=model_path,
        repo_root=tmp_path,
    )

    assert changed is True
    parsed = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert parsed["providers"]["omnivoice"]["enabled"] is True


@pytest.mark.unit
def test_omnivoice_installer_skips_complex_yaml_constructs_in_provider_block(tmp_path):
    from Helper_Scripts.TTS_Installers.install_tts_omnivoice_sidecar import (
        build_runtime_layout,
        patch_tts_config,
    )

    config_path = tmp_path / "tts_providers_config.yaml"
    original = (
        """
providers:
  kitten_tts:
    enabled: false

  omnivoice:
    enabled: false
    sample_rate: 24000 # keep legacy comment
""".strip()
        + "\n"
    )
    config_path.write_text(original, encoding="utf-8")
    layout = build_runtime_layout(Path("models") / "omnivoice_sidecar", repo_root=tmp_path)
    source_checkout = tmp_path / "external" / "OmniVoice"
    source_checkout.mkdir(parents=True)
    model_path = tmp_path / "models" / "OmniVoice"
    model_path.mkdir(parents=True)

    changed = patch_tts_config(
        config_path=config_path,
        layout=layout,
        source_checkout=source_checkout,
        model_path=model_path,
        repo_root=tmp_path,
    )

    assert changed is False
    assert config_path.read_text(encoding="utf-8") == original


@pytest.mark.unit
def test_omnivoice_installer_patches_checked_in_default_config_style(tmp_path):
    from Helper_Scripts.TTS_Installers.install_tts_omnivoice_sidecar import (
        build_runtime_layout,
        patch_tts_config,
    )

    repo_root = Path(__file__).resolve().parents[4]
    source_config = repo_root / "tldw_Server_API" / "Config_Files" / "tts_providers_config.yaml"
    config_path = tmp_path / "tts_providers_config.yaml"
    config_path.write_text(source_config.read_text(encoding="utf-8"), encoding="utf-8")
    layout = build_runtime_layout(Path("models") / "omnivoice_sidecar", repo_root=tmp_path)
    source_checkout = tmp_path / "external" / "OmniVoice"
    source_checkout.mkdir(parents=True)
    model_path = tmp_path / "models" / "OmniVoice"
    model_path.mkdir(parents=True)

    changed = patch_tts_config(
        config_path=config_path,
        layout=layout,
        source_checkout=source_checkout,
        model_path=model_path,
        repo_root=tmp_path,
    )

    parsed = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert changed is True
    assert parsed["providers"]["omnivoice"]["enabled"] is True
    assert parsed["providers"]["omnivoice"]["extra_params"]["model_path"] == "models/OmniVoice"


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


@pytest.mark.unit
def test_omnivoice_installer_rejects_missing_model_path(tmp_path):
    from Helper_Scripts.TTS_Installers.install_tts_omnivoice_sidecar import validate_local_model_path

    with pytest.raises(SystemExit, match="model"):
        validate_local_model_path(tmp_path / "missing")


@pytest.mark.unit
def test_omnivoice_installer_resolves_existing_model_path(tmp_path):
    from Helper_Scripts.TTS_Installers.install_tts_omnivoice_sidecar import validate_local_model_path

    model_path = tmp_path / "models" / "OmniVoice"
    model_path.mkdir(parents=True)

    assert validate_local_model_path(model_path) == model_path.resolve()


@pytest.mark.unit
def test_omnivoice_installer_main_requires_model_path_before_patching(tmp_path, monkeypatch):
    from Helper_Scripts.TTS_Installers import install_tts_omnivoice_sidecar as installer

    config_path = tmp_path / "tts_providers_config.yaml"
    config_path.write_text("providers: {}\n", encoding="utf-8")

    monkeypatch.setattr(installer, "_ensure_prerequisites", lambda: None)
    monkeypatch.setattr(installer, "resolve_repo_root", lambda: tmp_path)
    monkeypatch.setattr(installer, "create_runtime_layout", lambda layout: layout)
    monkeypatch.setattr(installer, "create_virtualenv", lambda venv_dir: None)
    monkeypatch.setattr(installer, "validate_runtime_layout", lambda layout: [])
    monkeypatch.setattr(installer, "patch_tts_config", lambda **kwargs: pytest.fail("config should not be patched"))

    with pytest.raises(SystemExit, match="model"):
        installer.main(
            [
                "--skip-clone",
                "--skip-install",
                "--config-path",
                str(config_path),
                "--runtime-base",
                str(tmp_path / "models" / "omnivoice_sidecar"),
            ]
        )


@pytest.mark.unit
def test_omnivoice_installer_parse_args_accepts_model_path_and_skip_check():
    from Helper_Scripts.TTS_Installers.install_tts_omnivoice_sidecar import parse_args

    args = parse_args(["--model-path", "models/OmniVoice", "--skip-model-check"])

    assert args.model_path == "models/OmniVoice"
    assert args.skip_model_check is True


@pytest.mark.unit
def test_omnivoice_installer_main_fails_when_config_patch_is_skipped(tmp_path, monkeypatch):
    from Helper_Scripts.TTS_Installers import install_tts_omnivoice_sidecar as installer

    config_path = tmp_path / "tts_providers_config.yaml"
    config_path.write_text("providers: {}\n", encoding="utf-8")
    model_path = tmp_path / "models" / "OmniVoice"
    model_path.mkdir(parents=True)

    monkeypatch.setattr(installer, "_ensure_prerequisites", lambda: None)
    monkeypatch.setattr(installer, "resolve_repo_root", lambda: tmp_path)
    monkeypatch.setattr(installer, "create_runtime_layout", lambda layout: layout)
    monkeypatch.setattr(installer, "create_virtualenv", lambda venv_dir: None)
    monkeypatch.setattr(installer, "validate_runtime_layout", lambda layout: [])
    monkeypatch.setattr(installer, "patch_tts_config", lambda **kwargs: False)

    with pytest.raises(SystemExit, match="configuration"):
        installer.main(
            [
                "--skip-clone",
                "--skip-install",
                "--model-path",
                "models/OmniVoice",
                "--config-path",
                str(config_path),
                "--runtime-base",
                str(tmp_path / "models" / "omnivoice_sidecar"),
            ]
        )


@pytest.mark.unit
def test_omnivoice_installer_resolves_relative_model_path_against_repo_root(tmp_path, monkeypatch):
    from Helper_Scripts.TTS_Installers import install_tts_omnivoice_sidecar as installer

    repo_root = tmp_path / "repo"
    cwd = tmp_path / "elsewhere"
    repo_root.mkdir()
    cwd.mkdir()
    config_path = repo_root / "tts_providers_config.yaml"
    config_path.write_text("providers:\n  omnivoice:\n    enabled: false\n", encoding="utf-8")
    model_path = repo_root / "models" / "OmniVoice"
    model_path.mkdir(parents=True)

    monkeypatch.chdir(cwd)
    monkeypatch.setattr(installer, "_ensure_prerequisites", lambda: None)
    monkeypatch.setattr(installer, "resolve_repo_root", lambda: repo_root)
    monkeypatch.setattr(installer, "create_runtime_layout", lambda layout: layout)
    monkeypatch.setattr(installer, "create_virtualenv", lambda venv_dir: None)
    monkeypatch.setattr(installer, "validate_runtime_layout", lambda layout: [])

    installer.main(
        [
            "--skip-clone",
            "--skip-install",
            "--model-path",
            "models/OmniVoice",
            "--config-path",
            str(config_path),
            "--runtime-base",
            str(repo_root / "models" / "omnivoice_sidecar"),
        ]
    )

    parsed = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert parsed["providers"]["omnivoice"]["extra_params"]["model_path"] == "models/OmniVoice"


@pytest.mark.unit
def test_omnivoice_installer_resolves_relative_config_path_against_repo_root(tmp_path, monkeypatch):
    from Helper_Scripts.TTS_Installers import install_tts_omnivoice_sidecar as installer

    repo_root = tmp_path / "repo"
    cwd = tmp_path / "elsewhere"
    repo_root.mkdir()
    cwd.mkdir()
    config_path = repo_root / "tts_providers_config.yaml"
    config_path.write_text("providers:\n  omnivoice:\n    enabled: false\n", encoding="utf-8")
    model_path = repo_root / "models" / "OmniVoice"
    model_path.mkdir(parents=True)

    monkeypatch.chdir(cwd)
    monkeypatch.setattr(installer, "_ensure_prerequisites", lambda: None)
    monkeypatch.setattr(installer, "resolve_repo_root", lambda: repo_root)
    monkeypatch.setattr(installer, "create_runtime_layout", lambda layout: layout)
    monkeypatch.setattr(installer, "create_virtualenv", lambda venv_dir: None)
    monkeypatch.setattr(installer, "validate_runtime_layout", lambda layout: [])

    installer.main(
        [
            "--skip-clone",
            "--skip-install",
            "--model-path",
            "models/OmniVoice",
            "--config-path",
            "tts_providers_config.yaml",
            "--runtime-base",
            str(repo_root / "models" / "omnivoice_sidecar"),
        ]
    )

    parsed = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert parsed["providers"]["omnivoice"]["extra_params"]["model_path"] == "models/OmniVoice"


@pytest.mark.unit
def test_omnivoice_installer_rejects_unsafe_config_path_scalars(tmp_path):
    from Helper_Scripts.TTS_Installers.install_tts_omnivoice_sidecar import (
        build_runtime_layout,
        patch_tts_config,
    )

    config_path = tmp_path / "tts_providers_config.yaml"
    original = "providers:\n  omnivoice:\n    enabled: false\n"
    config_path.write_text(original, encoding="utf-8")
    layout = build_runtime_layout(Path("models") / "omnivoice_sidecar", repo_root=tmp_path)
    source_checkout = tmp_path / "external" / "OmniVoice"
    source_checkout.mkdir(parents=True)

    with pytest.raises(SystemExit, match="Unsafe"):
        patch_tts_config(
            config_path=config_path,
            layout=layout,
            source_checkout=source_checkout,
            model_path=Path('models/bad"model'),
            repo_root=tmp_path,
        )

    assert config_path.read_text(encoding="utf-8") == original


@pytest.mark.unit
def test_omnivoice_installer_requires_model_path_before_prerequisites(monkeypatch):
    from Helper_Scripts.TTS_Installers import install_tts_omnivoice_sidecar as installer

    monkeypatch.setattr(installer, "_ensure_prerequisites", lambda: pytest.fail("prerequisites should not be checked"))

    with pytest.raises(SystemExit, match="model path"):
        installer.main(["--skip-clone", "--skip-install"])
