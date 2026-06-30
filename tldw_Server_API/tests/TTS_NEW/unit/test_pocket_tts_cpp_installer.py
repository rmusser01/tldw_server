from pathlib import Path

import pytest


@pytest.mark.unit
def test_pocket_tts_cpp_installer_resolves_repo_root_from_nested_path():
    from Helper_Scripts.TTS_Installers.install_tts_pocket_tts_cpp import resolve_repo_root

    repo_root = Path(__file__).resolve().parents[4]
    probe = repo_root / "tldw_Server_API" / "tests" / "TTS_NEW" / "unit" / "nested" / "probe.py"

    assert resolve_repo_root(probe) == repo_root


@pytest.mark.unit
def test_pocket_tts_cpp_installer_builds_separate_runtime_layout():
    from Helper_Scripts.TTS_Installers.install_tts_pocket_tts_cpp import build_runtime_layout

    repo_root = Path(__file__).resolve().parents[4]
    layout = build_runtime_layout(
        Path("models") / "pocket_tts_cpp",
        repo_root=repo_root,
        platform_name="linux",
    )

    assert layout.binary_path.relative_to(repo_root).as_posix() == "bin/pocket-tts"
    assert layout.tokenizer_path.relative_to(repo_root).as_posix() == "models/pocket_tts_cpp/tokenizer.model"
    assert layout.model_dir.relative_to(repo_root).as_posix() == "models/pocket_tts_cpp/onnx"
    assert layout.provider_name == "pocket_tts_cpp"


@pytest.mark.unit
def test_pocket_tts_cpp_installer_updates_only_cpp_provider_block(tmp_path):
    from Helper_Scripts.TTS_Installers.install_tts_pocket_tts_cpp import patch_tts_config

    config_path = tmp_path / "tts_providers_config.yaml"
    config_path.write_text(
        """
providers:
  pocket_tts:
    enabled: false
    model_path: "models/pocket_tts_onnx/onnx"
    tokenizer_path: "models/pocket_tts_onnx/tokenizer.model"
    module_path: "models/pocket_tts_onnx"

  pocket_tts_cpp:
    enabled: false
    binary_path: "models/pocket_tts_cpp/pocket_tts_cpp"
    tokenizer_path: "models/pocket_tts_cpp/tokenizer.model"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    changed = patch_tts_config(
        config_path=config_path,
        binary_path=Path("bin") / "pocket-tts",
        tokenizer_path=Path("models") / "pocket_tts_cpp" / "tokenizer.model",
        model_dir=Path("models") / "pocket_tts_cpp" / "onnx",
        repo_root=tmp_path,
    )

    content = config_path.read_text(encoding="utf-8")
    assert changed is True
    assert "pocket_tts:\n    enabled: false" in content
    assert "pocket_tts_cpp:\n    enabled: true" in content
    assert 'binary_path: "bin/pocket-tts"' in content
    assert 'model_path: "models/pocket_tts_cpp/onnx"' in content
    assert 'tokenizer_path: "models/pocket_tts_cpp/tokenizer.model"' in content


@pytest.mark.unit
def test_pocket_tts_cpp_installer_reports_missing_prerequisite_commands_without_running_them():
    from Helper_Scripts.TTS_Installers.install_tts_pocket_tts_cpp import missing_prerequisite_commands

    missing = missing_prerequisite_commands(available_commands={"git", "cmake"})

    assert missing == ["c++"]


@pytest.mark.unit
def test_pocket_tts_cpp_installer_exports_binary_tokenizer_and_onnx_layout(tmp_path):
    from Helper_Scripts.TTS_Installers.install_tts_pocket_tts_cpp import (
        build_runtime_layout,
        export_runtime_artifacts,
    )

    repo_root = tmp_path / "repo"
    runtime_base = repo_root / "models" / "pocket_tts_cpp"
    build_dir = tmp_path / "build"
    layout = build_runtime_layout(runtime_base, repo_root=repo_root)

    (build_dir / "bin").mkdir(parents=True, exist_ok=True)
    (build_dir / "assets").mkdir(parents=True, exist_ok=True)
    (build_dir / "onnx").mkdir(parents=True, exist_ok=True)
    (build_dir / "bin" / "pocket_tts_cpp").write_text("binary", encoding="utf-8")
    (build_dir / "assets" / "tokenizer.model").write_text("tokenizer", encoding="utf-8")
    (build_dir / "onnx" / "flow_lm_main_int8.onnx").write_text("model", encoding="utf-8")

    export_runtime_artifacts(build_dir=build_dir, layout=layout)

    assert layout.binary_path.read_text(encoding="utf-8") == "binary"
    assert layout.tokenizer_path.read_text(encoding="utf-8") == "tokenizer"
    assert (layout.model_dir / "flow_lm_main_int8.onnx").read_text(encoding="utf-8") == "model"


@pytest.mark.unit
def test_pocket_tts_cpp_installer_exports_from_install_layout_when_build_tree_is_incomplete(tmp_path):
    from Helper_Scripts.TTS_Installers.install_tts_pocket_tts_cpp import (
        build_runtime_layout,
        export_runtime_artifacts,
    )

    repo_root = tmp_path / "repo"
    runtime_base = repo_root / "models" / "pocket_tts_cpp"
    build_dir = tmp_path / "build"
    install_dir = tmp_path / "install"
    layout = build_runtime_layout(runtime_base, repo_root=repo_root)

    (install_dir / "bin").mkdir(parents=True, exist_ok=True)
    (install_dir / "onnx").mkdir(parents=True, exist_ok=True)
    (install_dir / "bin" / "pocket-tts").write_text("installed-binary", encoding="utf-8")
    (install_dir / "tokenizer.model").write_text("installed-tokenizer", encoding="utf-8")
    (install_dir / "onnx" / "flow_lm_main_int8.onnx").write_text("installed-model", encoding="utf-8")

    export_runtime_artifacts(build_dir=build_dir, install_dir=install_dir, layout=layout)

    assert layout.binary_path.read_text(encoding="utf-8") == "installed-binary"
    assert layout.tokenizer_path.read_text(encoding="utf-8") == "installed-tokenizer"
    assert (layout.model_dir / "flow_lm_main_int8.onnx").read_text(encoding="utf-8") == "installed-model"


@pytest.mark.unit
def test_pocket_tts_cpp_installer_builds_windows_binary_layout():
    from Helper_Scripts.TTS_Installers.install_tts_pocket_tts_cpp import build_runtime_layout

    repo_root = Path(__file__).resolve().parents[4]
    layout = build_runtime_layout(
        Path("models") / "pocket_tts_cpp",
        repo_root=repo_root,
        platform_name="win32",
    )

    assert layout.binary_path.relative_to(repo_root).as_posix() == "bin/pocket-tts.exe"


@pytest.mark.unit
def test_pocket_tts_cpp_installer_runs_install_step_before_export(tmp_path, monkeypatch):
    from Helper_Scripts.TTS_Installers import install_tts_pocket_tts_cpp as installer

    repo_root = tmp_path / "repo"
    runtime_base = repo_root / "models" / "pocket_tts_cpp"
    build_dir = tmp_path / "build"
    source_dir = tmp_path / "src"
    calls: list[tuple] = []

    monkeypatch.setattr(installer, "_ensure_prerequisites", lambda: None, raising=True)
    monkeypatch.setattr(installer, "resolve_repo_root", lambda start=None: repo_root, raising=True)
    monkeypatch.setattr(
        installer,
        "configure_build",
        lambda source_dir, build_dir, install_dir: calls.append(("configure", source_dir, build_dir, install_dir)),
        raising=True,
    )
    monkeypatch.setattr(
        installer,
        "build_project",
        lambda build_dir: calls.append(("build", build_dir)),
        raising=True,
    )
    monkeypatch.setattr(
        installer,
        "install_project",
        lambda build_dir: calls.append(("install", build_dir)),
        raising=True,
    )
    monkeypatch.setattr(
        installer,
        "export_runtime_artifacts",
        lambda *, build_dir, install_dir, layout: calls.append(("export", build_dir, install_dir, layout.binary_path)),
        raising=True,
    )
    monkeypatch.setattr(installer, "validate_runtime_layout", lambda layout: [], raising=True)
    monkeypatch.setattr(installer, "patch_tts_config", lambda **kwargs: True, raising=True)

    exit_code = installer.main(
        [
            "--runtime-base",
            str(runtime_base),
            "--build-dir",
            str(build_dir),
            "--source-dir",
            str(source_dir),
            "--config-path",
            str(tmp_path / "tts_providers_config.yaml"),
            "--no-clone",
        ]
    )

    assert exit_code == 0
    assert [entry[0] for entry in calls] == ["configure", "build", "install", "export"]
    assert calls[0][1] == source_dir
    assert calls[0][2] == build_dir
    assert calls[0][3] == runtime_base
    assert calls[2][1] == build_dir
    assert calls[3][1] == build_dir
    assert calls[3][2] == runtime_base


@pytest.mark.unit
def test_pocket_tts_cpp_installer_aborts_before_config_patch_when_runtime_is_incomplete(
    tmp_path,
    monkeypatch,
):
    from Helper_Scripts.TTS_Installers import install_tts_pocket_tts_cpp as installer

    repo_root = tmp_path / "repo"
    runtime_base = repo_root / "models" / "pocket_tts_cpp"
    build_dir = tmp_path / "empty_build"
    build_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(installer, "_ensure_prerequisites", lambda: None, raising=True)
    monkeypatch.setattr(installer, "resolve_repo_root", lambda start=None: repo_root, raising=True)
    monkeypatch.setattr(installer, "clone_repository", lambda *args, **kwargs: None, raising=True)
    monkeypatch.setattr(installer, "configure_build", lambda *args, **kwargs: None, raising=True)
    monkeypatch.setattr(installer, "build_project", lambda *args, **kwargs: None, raising=True)

    patch_calls: list[object] = []

    def _record_patch(*args, **kwargs):
        patch_calls.append((args, kwargs))
        return True

    monkeypatch.setattr(installer, "patch_tts_config", _record_patch, raising=True)

    with pytest.raises(SystemExit, match="runtime export incomplete"):
        installer.main(
            [
                "--runtime-base",
                str(runtime_base),
                "--build-dir",
                str(build_dir),
                "--config-path",
                str(tmp_path / "tts_providers_config.yaml"),
                "--no-clone",
                "--no-build",
            ]
        )

    assert patch_calls == []
