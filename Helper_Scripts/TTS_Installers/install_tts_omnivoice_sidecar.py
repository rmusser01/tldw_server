#!/usr/bin/env python3
"""
Install and configure the OmniVoice sidecar runtime.

This helper provisions a dedicated runtime under ``models/omnivoice_sidecar`` and
patches only the ``omnivoice`` provider block in ``tts_providers_config.yaml``.
Pure helpers are kept separate from the CLI flow so the install plan can test
layout/config logic without creating environments or downloading anything.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess  # nosec B404
import sys
import venv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from loguru import logger


DEFAULT_REPO_URL = "https://github.com/k2-fsa/OmniVoice"
DEFAULT_RUNTIME_BASE = Path("models") / "omnivoice_sidecar"
DEFAULT_CONFIG_PATH = Path("tldw_Server_API") / "Config_Files" / "tts_providers_config.yaml"
DEFAULT_SOURCE_CHECKOUT = Path("external") / "OmniVoice"
DEFAULT_LOCAL_CHECKOUT = Path("..") / "OmniVoice"
PROVIDER_NAME = "omnivoice"


@dataclass(frozen=True)
class OmniVoiceRuntimeLayout:
    """Resolved runtime paths for the OmniVoice sidecar."""

    provider_name: str
    runtime_base: Path
    venv_dir: Path
    runtime_dir: Path
    logs_dir: Path
    interpreter_path: Path


def resolve_repo_root(start: Optional[Path] = None) -> Path:
    """Resolve the repository root from a probe path."""

    probe = (start or Path(__file__)).resolve()
    candidates = (probe,) + tuple(probe.parents)
    for candidate in candidates:
        if (candidate / "pyproject.toml").exists() and (candidate / "tldw_Server_API").is_dir():
            return candidate
    raise FileNotFoundError(f"Unable to resolve repository root from {probe}")


def resolve_sidecar_python_path(venv_dir: Path, *, platform_name: Optional[str] = None) -> Path:
    """Return the expected Python interpreter path inside a virtual environment."""

    platform_key = (platform_name or sys.platform).lower()
    if platform_key.startswith("win"):
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def resolve_source_checkout(
    *,
    repo_root: Optional[Path] = None,
    source_dir: Optional[Path] = None,
    default_probe: Optional[Path] = None,
) -> Path:
    """Resolve the OmniVoice checkout, preferring ``../OmniVoice`` when present."""

    if source_dir is not None:
        return source_dir.expanduser().resolve()

    probe = default_probe.expanduser().resolve() if default_probe is not None else None
    if probe is not None and probe.exists():
        return probe

    root = repo_root if repo_root is not None else resolve_repo_root()
    preferred = (root / DEFAULT_LOCAL_CHECKOUT).resolve()
    if preferred.exists():
        return preferred

    return (root / DEFAULT_SOURCE_CHECKOUT).resolve()


def build_runtime_layout(
    runtime_base: Path,
    repo_root: Optional[Path] = None,
    *,
    platform_name: Optional[str] = None,
) -> OmniVoiceRuntimeLayout:
    """Build the expected OmniVoice sidecar runtime layout."""

    root = repo_root if repo_root is not None else resolve_repo_root()
    base_candidate = runtime_base.expanduser()
    base = base_candidate if base_candidate.is_absolute() else (root / base_candidate)
    venv_dir = base / ".venv"
    return OmniVoiceRuntimeLayout(
        provider_name=PROVIDER_NAME,
        runtime_base=base,
        venv_dir=venv_dir,
        runtime_dir=base / "runtime",
        logs_dir=base / "logs",
        interpreter_path=resolve_sidecar_python_path(venv_dir, platform_name=platform_name),
    )


def create_runtime_layout(layout: OmniVoiceRuntimeLayout) -> OmniVoiceRuntimeLayout:
    """Create the directory layout required by the sidecar runtime."""

    layout.runtime_base.mkdir(parents=True, exist_ok=True)
    layout.runtime_dir.mkdir(parents=True, exist_ok=True)
    layout.logs_dir.mkdir(parents=True, exist_ok=True)
    return layout


def validate_runtime_layout(layout: OmniVoiceRuntimeLayout) -> list[str]:
    """Return a list of missing required runtime artifacts."""

    missing: list[str] = []
    for path in (layout.runtime_base, layout.venv_dir, layout.runtime_dir, layout.logs_dir, layout.interpreter_path):
        if not path.exists():
            missing.append(str(path))
    return missing


def _path_for_config(path: Path, repo_root: Optional[Path]) -> str:
    if not path.is_absolute():
        return path.as_posix()
    if repo_root is not None:
        try:
            return os.path.relpath(path, repo_root).replace(os.sep, "/")
        except ValueError:
            pass
    return str(path)


def _find_provider_block(lines: list[str], provider_name: str) -> tuple[Optional[int], Optional[int], Optional[int]]:
    in_providers = False
    providers_indent: Optional[int] = None
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


def _insert_provider_block(lines: list[str], provider_name: str, block_lines: list[str]) -> list[str]:
    block_start, block_end, _block_indent = _find_provider_block(lines, provider_name)
    if block_start is not None and block_end is not None:
        lines[block_start:block_end] = block_lines
        return lines

    providers_start = None
    for idx, line in enumerate(lines):
        if line.strip() == "providers:":
            providers_start = idx
            break
    if providers_start is None:
        lines.extend(["", "providers:"])
        providers_start = len(lines) - 1

    insert_at = len(lines)
    providers_indent = len(lines[providers_start]) - len(lines[providers_start].lstrip(" "))
    for idx in range(providers_start + 1, len(lines)):
        stripped = lines[idx].strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(lines[idx]) - len(lines[idx].lstrip(" "))
        if indent <= providers_indent:
            insert_at = idx
            break

    lines[insert_at:insert_at] = block_lines
    return lines


def patch_tts_config(
    *,
    config_path: Path,
    layout: OmniVoiceRuntimeLayout,
    source_checkout: Path,
    repo_root: Optional[Path] = None,
) -> bool:
    """Patch only the OmniVoice provider block."""

    if not config_path.exists():
        logger.warning("Config file not found at {}; skipping update.", config_path)
        return False

    lines = config_path.read_text(encoding="utf-8").splitlines()
    block_start, block_end, block_indent = _find_provider_block(lines, PROVIDER_NAME)
    provider_indent = " " * (block_indent or 0)
    key_indent = provider_indent + "  "
    nested_indent = key_indent + "  "
    block_lines = [
        f"{provider_indent}{PROVIDER_NAME}:",
        f"{key_indent}enabled: true",
        f'{key_indent}runtime: "sidecar"',
        f'{key_indent}model: "omnivoice"',
        f"{key_indent}sample_rate: 24000",
        f"{key_indent}max_concurrent_generations: 1",
        f"{key_indent}extra_params:",
        f'{nested_indent}repo_path: "{_path_for_config(source_checkout, repo_root)}"',
        f'{nested_indent}python_path: "{_path_for_config(layout.interpreter_path, repo_root)}"',
        f'{nested_indent}runtime_path: "{_path_for_config(layout.runtime_dir, repo_root)}"',
        f'{nested_indent}logs_path: "{_path_for_config(layout.logs_dir, repo_root)}"',
        f'{nested_indent}host: "127.0.0.1"',
        f"{nested_indent}port: 8039",
        f"{nested_indent}autoselect_port: true",
        f"{nested_indent}warmup_on_startup: false",
        f"{nested_indent}idle_shutdown_seconds: 900",
        f"{nested_indent}resident_mode: false",
    ]
    if block_start is None or block_end is None:
        lines = _insert_provider_block(lines, PROVIDER_NAME, block_lines)
    else:
        lines[block_start:block_end] = block_lines

    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Updated OmniVoice provider configuration at {}", config_path)
    return True


def missing_prerequisite_commands(*, available_commands: Optional[set[str]] = None) -> list[str]:
    """Return missing external commands required by the CLI path."""

    commands = ("git",)
    detected = available_commands if available_commands is not None else {cmd for cmd in commands if shutil.which(cmd)}
    return [cmd for cmd in commands if cmd not in detected]


def _ensure_prerequisites() -> None:
    missing = missing_prerequisite_commands()
    if missing:
        raise SystemExit(f"Missing prerequisites: {', '.join(missing)}")


def clone_repository(repo_url: str, source_dir: Path) -> None:
    """Clone OmniVoice if the checkout does not already exist."""

    if source_dir.exists():
        logger.info("Using existing OmniVoice checkout at {}", source_dir)
        return
    git_executable = shutil.which("git")
    if not git_executable:
        raise SystemExit("Missing prerequisites: git")
    source_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([git_executable, "clone", repo_url, str(source_dir)], check=True)  # nosec B603


def create_virtualenv(venv_dir: Path) -> None:
    """Create a dedicated virtual environment for the sidecar."""

    if venv_dir.exists():
        logger.info("Using existing OmniVoice sidecar virtualenv at {}", venv_dir)
        return
    builder = venv.EnvBuilder(with_pip=True)
    builder.create(venv_dir)


def install_sidecar_runtime(
    *,
    interpreter_path: Path,
    repo_root: Path,
    source_checkout: Path,
) -> None:
    """Install the minimal sidecar runtime into the dedicated environment."""

    subprocess.run([str(interpreter_path), "-m", "pip", "install", "--upgrade", "pip"], check=True)  # nosec B603
    subprocess.run(  # nosec B603
        [
            str(interpreter_path),
            "-m",
            "pip",
            "install",
            "fastapi>=0.110.0",
            "uvicorn>=0.30.0",
            "httpx>=0.27.0",
            "pydantic>=2.7.0",
            "loguru>=0.7.0",
        ],
        check=True,
    )
    if source_checkout.exists():
        subprocess.run(  # nosec B603
            [
                str(interpreter_path),
                "-m",
                "pip",
                "install",
                "--no-deps",
                "-e",
                str(source_checkout),
            ],
            check=True,
            cwd=str(repo_root),
        )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install the OmniVoice sidecar runtime")
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--runtime-base", default=str(DEFAULT_RUNTIME_BASE))
    parser.add_argument("--source-dir")
    parser.add_argument("--config-path", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--skip-clone", action="store_true")
    parser.add_argument("--skip-install", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    _ensure_prerequisites()

    repo_root = resolve_repo_root()
    runtime_base = Path(args.runtime_base)
    config_path = Path(args.config_path)
    source_checkout = resolve_source_checkout(
        repo_root=repo_root,
        source_dir=Path(args.source_dir) if args.source_dir else None,
    )
    layout = build_runtime_layout(runtime_base, repo_root=repo_root)
    create_runtime_layout(layout)

    if not args.skip_clone:
        clone_repository(args.repo_url, source_checkout)

    create_virtualenv(layout.venv_dir)
    if not args.skip_install:
        install_sidecar_runtime(
            interpreter_path=layout.interpreter_path,
            repo_root=repo_root,
            source_checkout=source_checkout,
        )

    missing = validate_runtime_layout(layout)
    if missing:
        raise SystemExit(f"OmniVoice runtime layout incomplete: {', '.join(missing)}")

    patch_tts_config(
        config_path=config_path,
        layout=layout,
        source_checkout=source_checkout,
        repo_root=repo_root,
    )
    logger.info("OmniVoice sidecar runtime ready at {}", layout.runtime_base)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
