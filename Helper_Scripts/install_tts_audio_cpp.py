#!/usr/bin/env python3
"""Install/config helper for the optional audio.cpp TTS runtime."""

from __future__ import annotations

import argparse
import subprocess  # nosec B404 - installer CLI intentionally runs explicit argv lists
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

DEFAULT_REPO_URL = "https://github.com/0xShug0/audio.cpp"
DEFAULT_RUNTIME_BASE = Path("models") / "audio_cpp"
DEFAULT_CONFIG_PATH = Path("tldw_Server_API") / "Config_Files" / "tts_providers_config.yaml"
DEFAULT_SOURCE_DIR = Path("external") / "audio.cpp"
PROVIDER_NAME = "audio_cpp"


@dataclass(frozen=True)
class AudioCppRuntimeLayout:
    """Resolved repo-local paths for an audio.cpp sidecar runtime."""

    provider_name: str
    runtime_base: Path
    binary_path: Path
    model_path: Path
    server_config_path: Path
    shared_scratch_dir: Path
    source_dir: Path
    build_dir: Path


def default_binary_name(platform_name: str | None = None) -> str:
    platform_key = (platform_name or sys.platform).lower()
    if platform_key.startswith("win"):
        return "audiocpp_server.exe"
    return "audiocpp_server"


def resolve_repo_root(start: Path | None = None) -> Path:
    probe = (start or Path(__file__)).resolve()
    candidates = (probe,) + tuple(probe.parents)
    for candidate in candidates:
        if (candidate / "pyproject.toml").exists() and (candidate / "tldw_Server_API").is_dir():
            return candidate
    raise FileNotFoundError(f"Unable to resolve repository root from {probe}")


def build_runtime_layout(
    runtime_base: Path,
    repo_root: Path | None = None,
    *,
    platform_name: str | None = None,
) -> AudioCppRuntimeLayout:
    root = repo_root if repo_root is not None else resolve_repo_root()
    base_candidate = runtime_base.expanduser()
    base = base_candidate if base_candidate.is_absolute() else root / base_candidate
    return AudioCppRuntimeLayout(
        provider_name=PROVIDER_NAME,
        runtime_base=base,
        binary_path=root / "bin" / default_binary_name(platform_name),
        model_path=base / "pocket-tts",
        server_config_path=base / "server.json",
        shared_scratch_dir=base / "runtime" / "scratch",
        source_dir=root / DEFAULT_SOURCE_DIR,
        build_dir=base / "_build",
    )


def _path_for_config(path: Path, repo_root: Path | None) -> str:
    if not path.is_absolute():
        return path.as_posix()
    if repo_root is not None:
        try:
            return path.relative_to(repo_root).as_posix()
        except ValueError:
            pass
    return path.as_posix()


def _find_provider_block(lines: list[str], provider_name: str) -> tuple[int | None, int | None, int | None]:
    in_providers = False
    providers_indent: int | None = None
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if not in_providers:
            if stripped == "providers:":
                in_providers = True
                providers_indent = indent
            continue
        if providers_indent is not None and indent <= providers_indent:
            in_providers = False
            continue
        if stripped.startswith(f"{provider_name}:"):
            block_start = idx
            block_indent = indent
            block_end = idx + 1
            while block_end < len(lines):
                next_line = lines[block_end]
                next_stripped = next_line.strip()
                if not next_stripped or next_stripped.startswith("#"):
                    block_end += 1
                    continue
                next_indent = len(next_line) - len(next_line.lstrip(" "))
                if next_indent <= block_indent:
                    break
                block_end += 1
            return block_start, block_end, block_indent
    return None, None, None


def _find_providers_insert_at(lines: list[str]) -> tuple[int, int]:
    for idx, line in enumerate(lines):
        if line.strip() == "providers:":
            providers_indent = len(line) - len(line.lstrip(" "))
            insert_at = len(lines)
            for probe in range(idx + 1, len(lines)):
                stripped = lines[probe].strip()
                if not stripped or stripped.startswith("#"):
                    continue
                indent = len(lines[probe]) - len(lines[probe].lstrip(" "))
                if indent <= providers_indent:
                    insert_at = probe
                    break
            return insert_at, providers_indent + 2
    lines.extend(["", "providers:"])
    return len(lines), 2


def _render_provider_block(
    *,
    layout: AudioCppRuntimeLayout,
    repo_root: Path | None,
    enable_provider: bool,
    base_url: str,
    provider_indent: int,
) -> list[str]:
    provider_prefix = " " * provider_indent
    key_prefix = provider_prefix + "  "
    nested_prefix = key_prefix + "  "
    server_prefix = nested_prefix + "  "
    model_prefix = server_prefix + "  "
    model_nested_prefix = model_prefix + "  "

    binary_path = _path_for_config(layout.binary_path, repo_root)
    model_path = _path_for_config(layout.model_path, repo_root)
    server_config_path = _path_for_config(layout.server_config_path, repo_root)
    models_root = _path_for_config(layout.runtime_base, repo_root)
    shared_scratch_dir = _path_for_config(layout.shared_scratch_dir, repo_root)
    return [
        f"{provider_prefix}{PROVIDER_NAME}:",
        f"{key_prefix}enabled: {'true' if enable_provider else 'false'}",
        f'{key_prefix}backend: "cuda"',
        f'{key_prefix}base_url: "{base_url}"',
        f'{key_prefix}model: "audio-cpp/pocket-tts"',
        f'{key_prefix}model_path: "{model_path}"',
        f'{key_prefix}binary_path: "{binary_path}"',
        f"{key_prefix}device: cuda",
        f"{key_prefix}timeout: 300",
        f"{key_prefix}sample_rate: 24000",
        f"{key_prefix}max_concurrent_generations: 1",
        f"{key_prefix}auto_download: false",
        f"{key_prefix}extra_params:",
        f"{nested_prefix}managed: true",
        f"{nested_prefix}allow_remote_base_url: false",
        f'{nested_prefix}external_voice_reference_mode: "disabled"',
        f"{nested_prefix}retain_request_artifacts: false",
        f"{nested_prefix}server:",
        f'{server_prefix}host: "127.0.0.1"',
        f"{server_prefix}port: 8080",
        f"{server_prefix}autoselect_port: true",
        f"{server_prefix}port_probe_max: 10",
        f"{server_prefix}startup_timeout_seconds: 30",
        f"{server_prefix}healthcheck_interval_seconds: 0.25",
        f"{server_prefix}startup_backoff_seconds: 5",
        f"{server_prefix}idle_shutdown_seconds: 900",
        f"{server_prefix}terminate_timeout_seconds: 10",
        f'{server_prefix}server_config_path: "{server_config_path}"',
        f'{server_prefix}models_root: "{models_root}"',
        f'{server_prefix}shared_scratch_dir: "{shared_scratch_dir}"',
        f"{server_prefix}lazy_load: true",
        f"{server_prefix}device: 0",
        f"{server_prefix}threads: 1",
        f"{server_prefix}model:",
        f'{model_prefix}id: "pocket-tts"',
        f'{model_prefix}family: "pocket_tts"',
        f'{model_prefix}path: "{model_path}"',
        f'{model_prefix}task: "tts"',
        f'{model_prefix}mode: "offline"',
        f"{model_prefix}load_options:",
        f'{model_nested_prefix}language: "english"',
        f"{model_prefix}session_options:",
        f'{model_nested_prefix}language: "english"',
        f"{nested_prefix}request_option_allowlist:",
        f'{server_prefix}- "max_tokens"',
        f'{server_prefix}- "seed"',
    ]


def patch_tts_config(
    *,
    config_path: Path,
    layout: AudioCppRuntimeLayout,
    repo_root: Path | None = None,
    enable_provider: bool = False,
    base_url: str = "http://127.0.0.1:8080",
) -> bool:
    """Patch only the audio_cpp provider block."""
    if not config_path.exists():
        logger.warning("Config file not found at {}; skipping update.", config_path)
        return False

    lines = config_path.read_text(encoding="utf-8").splitlines()
    block_start, block_end, block_indent = _find_provider_block(lines, PROVIDER_NAME)
    if block_indent is None:
        insert_at, provider_indent = _find_providers_insert_at(lines)
    else:
        insert_at = block_start if block_start is not None else len(lines)
        provider_indent = block_indent

    block_lines = _render_provider_block(
        layout=layout,
        repo_root=repo_root,
        enable_provider=enable_provider,
        base_url=base_url,
        provider_indent=provider_indent,
    )
    if block_start is not None and block_end is not None:
        lines[block_start:block_end] = block_lines
    else:
        lines[insert_at:insert_at] = block_lines

    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Updated audio.cpp provider configuration at {}", config_path)
    return True


def build_clone_command(repo_url: str, source_dir: Path) -> list[str]:
    return ["git", "clone", "--depth", "1", repo_url, str(source_dir)]


def build_cmake_configure_command(
    *,
    source_dir: Path,
    build_dir: Path,
    install_dir: Path,
    backend: str = "cuda",
) -> list[str]:
    return [
        "cmake",
        "-S",
        str(source_dir),
        "-B",
        str(build_dir),
        f"-DCMAKE_INSTALL_PREFIX={install_dir}",
        f"-DAUDIOCPP_BACKEND={backend}",
    ]


def build_cmake_build_command(build_dir: Path) -> list[str]:
    return ["cmake", "--build", str(build_dir), "--config", "Release"]


def build_model_manager_command(
    *,
    source_dir: Path,
    package_id: str,
    models_root: Path,
    python_executable: str = sys.executable,
) -> list[str]:
    return [
        python_executable,
        str(source_dir / "tools" / "model_manager.py"),
        "install",
        package_id,
        "--models-root",
        str(models_root),
    ]


def _run_checked(command: Sequence[str]) -> None:
    subprocess.run([str(part) for part in command], check=True)  # nosec B603


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install/configure the optional audio.cpp TTS runtime")
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--runtime-base", default=str(DEFAULT_RUNTIME_BASE))
    parser.add_argument("--source-dir")
    parser.add_argument("--build-dir")
    parser.add_argument("--config-path", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--package-id", default="pocket-tts")
    parser.add_argument("--enable-provider", action="store_true")
    parser.add_argument("--clone", action="store_true", help="Clone audio.cpp")
    parser.add_argument("--configure", action="store_true", help="Run cmake configure")
    parser.add_argument("--build", action="store_true", help="Run cmake build")
    parser.add_argument("--install-model", action="store_true", help="Run upstream model_manager.py install")
    parser.add_argument("--patch-config", action="store_true", help="Patch tts_providers_config.yaml")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = resolve_repo_root()
    layout = build_runtime_layout(Path(args.runtime_base), repo_root=repo_root)
    source_dir = Path(args.source_dir).expanduser() if args.source_dir else layout.source_dir
    build_dir = Path(args.build_dir).expanduser() if args.build_dir else layout.build_dir

    if args.clone:
        _run_checked(build_clone_command(args.repo_url, source_dir))
    if args.configure:
        _run_checked(
            build_cmake_configure_command(
                source_dir=source_dir,
                build_dir=build_dir,
                install_dir=layout.runtime_base,
                backend=args.backend,
            )
        )
    if args.build:
        _run_checked(build_cmake_build_command(build_dir))
    if args.install_model:
        _run_checked(
            build_model_manager_command(
                source_dir=source_dir,
                package_id=args.package_id,
                models_root=layout.runtime_base,
            )
        )
    if args.patch_config:
        patch_tts_config(
            config_path=repo_root / Path(args.config_path),
            layout=layout,
            repo_root=repo_root,
            enable_provider=args.enable_provider,
        )

    logger.info("audio.cpp runtime layout: {}", layout.runtime_base)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
