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
import re
import shutil
import subprocess  # nosec B404 - installer intentionally launches vetted local git/pip argv without a shell
import sys
import venv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence
from urllib.parse import urlparse

from loguru import logger


DEFAULT_REPO_URL = "https://github.com/k2-fsa/OmniVoice"
DEFAULT_RUNTIME_BASE = Path("models") / "omnivoice_sidecar"
DEFAULT_CONFIG_PATH = Path("tldw_Server_API") / "Config_Files" / "tts_providers_config.yaml"
DEFAULT_SOURCE_CHECKOUT = Path("external") / "OmniVoice"
DEFAULT_LOCAL_CHECKOUT = Path("..") / "OmniVoice"
PROVIDER_NAME = "omnivoice"
_GIT_SCP_URL_RE = re.compile(r"^[A-Za-z0-9_.-]+@[A-Za-z0-9_.-]+:[A-Za-z0-9_./-]+(?:\.git)?$")


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
    (layout.runtime_dir / "scratch").mkdir(parents=True, exist_ok=True)
    layout.logs_dir.mkdir(parents=True, exist_ok=True)
    return layout


def validate_local_model_path(model_path: Path) -> Path:
    """Resolve and validate an existing local OmniVoice model directory."""

    resolved = model_path.expanduser().resolve()
    if not resolved.is_dir():
        raise SystemExit(f"OmniVoice model path is not a directory: {resolved}")
    return resolved


def validate_runtime_layout(layout: OmniVoiceRuntimeLayout) -> list[str]:
    """Return a list of missing required runtime artifacts."""

    return [
        str(path)
        for path in (layout.runtime_base, layout.venv_dir, layout.runtime_dir, layout.logs_dir, layout.interpreter_path)
        if not path.exists()
    ]


def _path_for_config(path: Path, repo_root: Optional[Path]) -> str:
    path_posix = path.as_posix()
    if not path.is_absolute():
        return path_posix
    if repo_root is not None:
        try:
            rel_path = os.path.relpath(path, repo_root)
            rel_path_posix = rel_path.replace("\\", "/").replace(os.sep, "/")
            if ":" in rel_path_posix and getattr(path, "drive", ""):
                return path_posix
            return rel_path_posix
        except (TypeError, ValueError):
            pass
    return path_posix


def _validate_config_path_scalar(value: str) -> str:
    if any(character in value for character in ('"', "\n", "\r")):
        raise SystemExit(f"Unsafe path value for YAML config: {value!r}")
    return value


def _safe_path_for_config(path: Path, repo_root: Optional[Path]) -> str:
    return _validate_config_path_scalar(_path_for_config(path, repo_root))


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
                next_indent = len(next_line) - len(next_line.lstrip(" "))
                if not next_stripped or next_stripped.startswith("#"):
                    lookahead = block_end + 1
                    while lookahead < len(lines):
                        lookahead_line = lines[lookahead]
                        lookahead_stripped = lookahead_line.strip()
                        if lookahead_stripped and not lookahead_stripped.startswith("#"):
                            break
                        lookahead += 1
                    if lookahead >= len(lines):
                        break
                    lookahead_line = lines[lookahead]
                    lookahead_indent = len(lookahead_line) - len(lookahead_line.lstrip(" "))
                    if lookahead_indent <= block_indent:
                        break
                    block_end += 1
                    continue
                if next_indent <= block_indent:
                    break
                block_end += 1
            return block_start, block_end, block_indent
    return None, None, None


def _find_providers_indent(lines: list[str]) -> Optional[int]:
    for line in lines:
        if line.strip() == "providers:":
            return len(line) - len(line.lstrip(" "))
    return None


def _has_unsupported_providers_declaration(lines: list[str]) -> bool:
    return any(line.strip().startswith("providers:") for line in lines)


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


def _line_has_inline_yaml_comment(line: str) -> bool:
    in_single_quote = False
    in_double_quote = False
    saw_mapping_separator = False

    for character in line:
        if character == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
            continue
        if character == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
            continue
        if in_single_quote or in_double_quote:
            continue
        if character == ":":
            saw_mapping_separator = True
            continue
        if character == "#" and saw_mapping_separator:
            return True
    return False


def _find_unsupported_yaml_construct(lines: list[str]) -> str | None:
    for index, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if _line_has_inline_yaml_comment(line):
            return f"inline comment on line {index}"
        if "{" in stripped or "}" in stripped:
            return f"flow-style mapping on line {index}"
        if re.search(r":\s*[>|](?:\s|$)", line):
            return f"multiline scalar on line {index}"
    return None


def _validate_repository_url(repo_url: str) -> str:
    candidate = str(repo_url or "").strip()
    if not candidate:
        raise SystemExit("Repository URL is required")
    if any(token in candidate for token in (";", "|", "&", "$", "\n", "\r")):
        raise SystemExit(f"Invalid repository URL: {repo_url}")
    if _GIT_SCP_URL_RE.match(candidate):
        return candidate

    parsed = urlparse(candidate)
    if parsed.scheme in {"https", "http", "ssh", "git"} and parsed.netloc and parsed.path:
        return candidate

    raise SystemExit(f"Invalid repository URL: {repo_url}")


def patch_tts_config(
    *,
    config_path: Path,
    layout: OmniVoiceRuntimeLayout,
    source_checkout: Path,
    model_path: Path,
    repo_root: Optional[Path] = None,
) -> bool:
    """Patch only the OmniVoice provider block."""

    if not config_path.exists():
        logger.warning("Config file not found at {}; skipping update.", config_path)
        return False

    lines = config_path.read_text(encoding="utf-8").splitlines()
    block_start, block_end, block_indent = _find_provider_block(lines, PROVIDER_NAME)
    if block_start is not None and block_end is not None:
        unsupported_construct = _find_unsupported_yaml_construct(lines[block_start:block_end])
        if unsupported_construct is not None:
            logger.warning(
                "Skipping OmniVoice provider config patch at {} because the provider block contains unsupported "
                "constructs ({})",
                config_path,
                unsupported_construct,
            )
            return False
    providers_indent = _find_providers_indent(lines)
    if providers_indent is None and _has_unsupported_providers_declaration(lines):
        logger.warning(
            "Skipping OmniVoice provider config patch at {} because the providers declaration is unsupported",
            config_path,
        )
        return False
    effective_block_indent = block_indent if block_indent is not None else ((providers_indent or 0) + 2)
    provider_indent = " " * effective_block_indent
    key_indent = provider_indent + "  "
    nested_indent = key_indent + "  "
    scratch_dir = layout.runtime_dir / "scratch"
    block_lines = [
        f"{provider_indent}{PROVIDER_NAME}:",
        f"{key_indent}enabled: true",
        f'{key_indent}runtime: "sidecar"',
        f'{key_indent}model: "omnivoice"',
        f"{key_indent}sample_rate: 24000",
        f"{key_indent}max_concurrent_generations: 1",
        f"{key_indent}extra_params:",
        f'{nested_indent}repo_path: "{_path_for_config(source_checkout, repo_root)}"',
        f'{nested_indent}model_path: "{_path_for_config(model_path, repo_root)}"',
        f'{nested_indent}python_path: "{_path_for_config(layout.interpreter_path, repo_root)}"',
        f'{nested_indent}runtime_path: "{_path_for_config(layout.runtime_dir, repo_root)}"',
        f'{nested_indent}scratch_dir: "{_path_for_config(scratch_dir, repo_root)}"',
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


def _run_checked_command(command: Sequence[str], *, cwd: Optional[Path] = None) -> None:
    subprocess.run(  # nosec B603 - fixed argv list, no shell, installer-controlled executable paths
        [str(part) for part in command],
        check=True,
        cwd=str(cwd) if cwd is not None else None,
    )


def clone_repository(repo_url: str, source_dir: Path) -> None:
    """Clone OmniVoice if the checkout does not already exist."""

    if source_dir.exists():
        logger.info("Using existing OmniVoice checkout at {}", source_dir)
        return
    validated_repo_url = _validate_repository_url(repo_url)
    git_executable = shutil.which("git")
    if not git_executable:
        raise SystemExit("Missing prerequisites: git")
    source_dir.parent.mkdir(parents=True, exist_ok=True)
    _run_checked_command([git_executable, "clone", validated_repo_url, str(source_dir)])


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
    install_inference_deps: bool = True,
) -> None:
    """Install the sidecar runtime and OmniVoice source dependencies."""

    _run_checked_command([str(interpreter_path), "-m", "pip", "install", "--upgrade", "pip"])
    _run_checked_command(
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
        ]
    )
    if source_checkout.exists():
        install_command = [
            str(interpreter_path),
            "-m",
            "pip",
            "install",
        ]
        if not install_inference_deps:
            logger.warning(
                "OmniVoice source dependencies are required in the sidecar venv; "
                "installing dependencies despite install_inference_deps=False"
            )
        install_command.extend(["-e", str(source_checkout)])
        _run_checked_command(install_command, cwd=repo_root)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install the OmniVoice sidecar runtime")
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--runtime-base", default=str(DEFAULT_RUNTIME_BASE))
    parser.add_argument("--source-dir")
    parser.add_argument("--model-path", help="Resolved local OmniVoice model directory")
    parser.add_argument("--config-path", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--skip-clone", action="store_true")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--skip-model-check", action="store_true")
    parser.add_argument("--recreate-venv", action="store_true")
    parser.add_argument(
        "--install-inference-deps",
        action="store_true",
        default=True,
        help="Retained for compatibility; OmniVoice source dependencies are installed by default.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not args.model_path:
        raise SystemExit("OmniVoice model path is required; pass --model-path")

    _ensure_prerequisites()

    repo_root = resolve_repo_root()
    if not args.model_path:
        raise SystemExit("OmniVoice model path is required; pass --model-path")

    model_path = Path(args.model_path).expanduser().resolve()
    if not args.skip_model_check:
        model_path = validate_local_model_path(Path(args.model_path))

    runtime_base = Path(args.runtime_base)
    config_path = _resolve_path_from_repo_root(Path(args.config_path), repo_root)
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

    config_updated = patch_tts_config(
        config_path=config_path,
        layout=layout,
        source_checkout=source_checkout,
        model_path=model_path,
        repo_root=repo_root,
    )
    if not config_updated:
        raise SystemExit("OmniVoice provider configuration was not updated")
    logger.info("OmniVoice sidecar runtime ready at {}", layout.runtime_base)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
