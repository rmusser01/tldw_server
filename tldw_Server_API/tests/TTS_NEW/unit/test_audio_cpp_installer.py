from __future__ import annotations

import shutil
from pathlib import Path

import pytest


def _workspace_test_dir(name: str) -> Path:
    root = Path.cwd() / "models" / "audio_cpp" / "test_artifacts" / name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    return root


@pytest.mark.unit
def test_audio_cpp_installer_builds_repo_local_runtime_layout():
    from Helper_Scripts.install_tts_audio_cpp import build_runtime_layout

    repo_root = Path(__file__).resolve().parents[4]

    layout = build_runtime_layout(Path("models") / "audio_cpp", repo_root=repo_root, platform_name="linux")

    assert layout.provider_name == "audio_cpp"
    assert layout.runtime_base.relative_to(repo_root).as_posix() == "models/audio_cpp"
    assert layout.binary_path.relative_to(repo_root).as_posix() == "bin/audiocpp_server"
    assert layout.model_path.relative_to(repo_root).as_posix() == "models/audio_cpp/pocket-tts"
    assert layout.server_config_path.relative_to(repo_root).as_posix() == "models/audio_cpp/server.json"
    assert layout.shared_scratch_dir.relative_to(repo_root).as_posix() == "models/audio_cpp/runtime/scratch"


@pytest.mark.unit
def test_audio_cpp_installer_patches_config_without_enabling_provider_by_default():
    from Helper_Scripts.install_tts_audio_cpp import build_runtime_layout, patch_tts_config

    test_root = _workspace_test_dir("installer_patch_disabled")
    config_path = test_root / "tts_providers_config.yaml"
    config_path.write_text(
        """
providers:
  audio_cpp:
    enabled: false
    base_url: "http://127.0.0.1:8080"
    extra_params:
      managed: false
""".strip()
        + "\n",
        encoding="utf-8",
    )
    repo_root = test_root / "repo"
    layout = build_runtime_layout(Path("models") / "audio_cpp", repo_root=repo_root, platform_name="linux")

    changed = patch_tts_config(
        config_path=config_path,
        layout=layout,
        repo_root=repo_root,
        enable_provider=False,
        base_url="http://127.0.0.1:9010",
    )

    content = config_path.read_text(encoding="utf-8")
    assert changed is True
    assert "audio_cpp:\n    enabled: false" in content
    assert 'base_url: "http://127.0.0.1:9010"' in content
    assert 'binary_path: "bin/audiocpp_server"' in content
    assert 'model_path: "models/audio_cpp/pocket-tts"' in content
    assert 'server_config_path: "models/audio_cpp/server.json"' in content
    assert "HF_TOKEN" not in content
    assert "api_key" not in content


@pytest.mark.unit
def test_audio_cpp_installer_can_enable_provider_when_requested():
    from Helper_Scripts.install_tts_audio_cpp import build_runtime_layout, patch_tts_config

    test_root = _workspace_test_dir("installer_patch_enabled")
    config_path = test_root / "tts_providers_config.yaml"
    config_path.write_text("providers:\n", encoding="utf-8")
    repo_root = test_root / "repo"
    layout = build_runtime_layout(Path("models") / "audio_cpp", repo_root=repo_root, platform_name="linux")

    patch_tts_config(
        config_path=config_path,
        layout=layout,
        repo_root=repo_root,
        enable_provider=True,
        base_url="http://127.0.0.1:8080",
    )

    content = config_path.read_text(encoding="utf-8")
    assert "audio_cpp:\n    enabled: true" in content
    assert "extra_params:\n      managed: true" in content


@pytest.mark.unit
def test_audio_cpp_installer_builds_explicit_clone_build_and_model_manager_commands():
    from Helper_Scripts.install_tts_audio_cpp import (
        build_clone_command,
        build_cmake_build_command,
        build_cmake_configure_command,
        build_model_manager_command,
    )

    test_root = _workspace_test_dir("installer_commands")
    source_dir = test_root / "external" / "audio.cpp"
    build_dir = test_root / "build"
    install_dir = test_root / "install"

    assert build_clone_command("https://github.com/0xShug0/audio.cpp", source_dir) == [
        "git",
        "clone",
        "--depth",
        "1",
        "https://github.com/0xShug0/audio.cpp",
        str(source_dir),
    ]
    assert build_cmake_configure_command(
        source_dir=source_dir,
        build_dir=build_dir,
        install_dir=install_dir,
        backend="cuda",
    ) == [
        "cmake",
        "-S",
        str(source_dir),
        "-B",
        str(build_dir),
        f"-DCMAKE_INSTALL_PREFIX={install_dir}",
        "-DAUDIOCPP_BACKEND=cuda",
    ]
    assert build_cmake_build_command(build_dir) == ["cmake", "--build", str(build_dir), "--config", "Release"]
    assert build_model_manager_command(
        source_dir=source_dir,
        package_id="pocket-tts",
        models_root=install_dir / "models",
        python_executable="python",
    ) == [
        "python",
        str(source_dir / "tools" / "model_manager.py"),
        "install",
        "pocket-tts",
        "--models-root",
        str(install_dir / "models"),
    ]


@pytest.mark.unit
def test_audio_cpp_installer_builds_windows_binary_layout():
    from Helper_Scripts.install_tts_audio_cpp import build_runtime_layout

    repo_root = Path(__file__).resolve().parents[4]

    layout = build_runtime_layout(Path("models") / "audio_cpp", repo_root=repo_root, platform_name="win32")

    assert layout.binary_path.relative_to(repo_root).as_posix() == "bin/audiocpp_server.exe"
