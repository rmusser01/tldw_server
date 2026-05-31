from __future__ import annotations

import asyncio
import difflib
import json
import os
import socket

# The wizard starts fixed Python module commands with shell=False.
import subprocess  # nosec B404
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import typer
from loguru import logger

from tldw_Server_API.app.core.testing import is_truthy

from . import profile_verify
from . import profiles as profile_utils
from .utils import detect as detect_utils
from .utils import env as env_utils
from .utils import files as files_utils
from .utils import git as git_utils

app = typer.Typer(add_completion=False, no_args_is_help=True, help="tldw_server setup wizard CLI")
_WIZARD_NONCRITICAL_EXCEPTIONS = (
    AttributeError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    json.JSONDecodeError,
    subprocess.SubprocessError,
)
_ENV_GITIGNORE_ENTRIES = [".env", ".env.*.bak", ".env.local", "wizard.log"]


def _emit(result: dict[str, Any], use_json: bool) -> None:
    if use_json:
        typer.echo(json.dumps(result, indent=2))
    else:
        # Minimal human-friendly print
        status = result.get("status", "ok")
        typer.echo(f"Status: {status}")
        for k in ("actions", "facts", "notes"):
            if k in result and result[k]:
                typer.echo(f"{k.capitalize()}: {result[k]}")


def _resolve_database_url(env_path: Path, *, prefer_env_file: bool = False) -> str | None:
    env_values = env_utils.load_env(env_path)
    if prefer_env_file:
        value = env_values.get("DATABASE_URL") or os.getenv("DATABASE_URL")
    else:
        value = os.getenv("DATABASE_URL") or env_values.get("DATABASE_URL")
    if value is None:
        return None
    trimmed = value.strip()
    if not trimmed or trimmed.lower() in {"none", "null", "nil"}:
        return None
    return trimmed


def _validate_database_url(db_url: str) -> tuple[bool, str]:
    try:
        parsed = urlsplit(db_url)
    except ValueError:
        return False, "unable to parse DATABASE_URL"
    scheme = (parsed.scheme or "").split("+", 1)[0].lower()
    if scheme in {"postgres", "postgresql"}:
        if not parsed.netloc:
            return False, "missing host or credentials"
        return True, ""
    if scheme in {"sqlite", "file", ""}:
        return False, "sqlite/file URLs are not supported for multi_user"
    return False, f"unsupported scheme '{scheme}'"


def _resolve_sqlite_db_path(db_url: str) -> Path | None:
    try:
        from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
    except ImportError:
        return None
    _, _, fs_path = DatabasePool._resolve_sqlite_paths(db_url)
    if not fs_path or fs_path == ":memory:":
        return None
    path = Path(fs_path).expanduser()
    path = (Path.cwd() / path).resolve() if not path.is_absolute() else path.resolve()
    return path


def _resolve_user_db_base_dir_for_dry_run() -> Path:
    raw = os.getenv("USER_DB_BASE_DIR") or os.getenv("USER_DB_BASE")
    if raw:
        try:
            candidate = Path(raw).expanduser()
        except (OSError, TypeError, ValueError):
            candidate = Path(raw)
        candidate = (Path.cwd() / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
        return candidate
    return (Path.cwd() / "Databases" / "user_databases").resolve()


def _timestamped_backup(path: Path, content: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    backup = path.with_name(f"{path.name}.{timestamp}.bak")
    counter = 1
    while backup.exists():
        backup = path.with_name(f"{path.name}.{timestamp}.{counter}.bak")
        counter += 1
    files_utils.atomic_write(backup, content)
    return backup


def _render_unified_diff(before: str, after: str, *, label: str) -> str:
    diff = difflib.unified_diff(
        before.splitlines(),
        after.splitlines(),
        fromfile=f"{label} (before)",
        tofile=f"{label} (after)",
        lineterm="",
    )
    return "\n".join(diff)


def _split_ini_inline_comment(value: str) -> tuple[str, str]:
    for idx, ch in enumerate(value):
        if ch in {"#", ";"} and (idx == 0 or value[idx - 1].isspace()):
            return value[:idx].rstrip(), value[idx:].rstrip()
    return value.rstrip(), ""


def _update_ini_section(content: str, section: str, updates: dict[str, str]) -> tuple[str, bool, list[str]]:
    if not updates:
        return content, False, []
    lines = content.splitlines()
    target = section.strip()
    section_header = f"[{target}]"
    in_section = False
    section_found = False
    last_index: dict[str, int] = {}
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_section = stripped == section_header
            if in_section:
                section_found = True
            continue
        if not in_section:
            continue
        if not stripped or stripped.startswith("#") or stripped.startswith(";"):
            continue
        if "=" not in line:
            continue
        key = line.split("=", 1)[0].strip()
        if key in updates:
            last_index[key] = idx

    missing = [key for key in updates if key not in last_index]
    rendered: list[str] = []
    in_section = False
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            if in_section and missing:
                for key in missing:
                    rendered.append(f"{key} = {updates[key]}")
                missing = []
            in_section = stripped == section_header
            rendered.append(line)
            if in_section:
                section_found = True
            continue
        if not in_section:
            rendered.append(line)
            continue
        if not stripped or stripped.startswith("#") or stripped.startswith(";") or "=" not in line:
            rendered.append(line)
            continue
        key_part, raw_value = line.split("=", 1)
        key = key_part.strip()
        if key not in updates:
            if last_index.get(key) != idx:
                continue
            rendered.append(line)
            continue
        if last_index.get(key) != idx:
            continue
        value_part, comment = _split_ini_inline_comment(raw_value)
        prefix = key_part[: len(key_part) - len(key_part.lstrip())]
        suffix = f" {comment}" if comment else ""
        rendered.append(f"{prefix}{key} = {updates[key]}{suffix}")

    if in_section and missing:
        for key in missing:
            rendered.append(f"{key} = {updates[key]}")
        missing = []

    if not section_found:
        if rendered and rendered[-1].strip():
            rendered.append("")
        rendered.append(section_header)
        for key, value in updates.items():
            rendered.append(f"{key} = {value}")

    new_content = "\n".join(rendered)
    if new_content and not new_content.endswith("\n"):
        new_content += "\n"
    return new_content, new_content != content, missing


_PROVIDER_SOURCES = [
    {"name": "openai", "label": "OpenAI", "env_keys": ["OPENAI_API_KEY"], "config_key": "openai_api_key"},
    {"name": "anthropic", "label": "Anthropic", "env_keys": ["ANTHROPIC_API_KEY"], "config_key": "anthropic_api_key"},
    {"name": "cohere", "label": "Cohere", "env_keys": ["COHERE_API_KEY"], "config_key": "cohere_api_key"},
    {"name": "groq", "label": "Groq", "env_keys": ["GROQ_API_KEY"], "config_key": "groq_api_key"},
    {
        "name": "huggingface",
        "label": "HuggingFace",
        "env_keys": ["HUGGINGFACE_API_KEY"],
        "config_key": "huggingface_api_key",
    },
    {
        "name": "openrouter",
        "label": "OpenRouter",
        "env_keys": ["OPENROUTER_API_KEY"],
        "config_key": "openrouter_api_key",
    },
    {"name": "deepseek", "label": "DeepSeek", "env_keys": ["DEEPSEEK_API_KEY"], "config_key": "deepseek_api_key"},
    {"name": "qwen", "label": "Qwen", "env_keys": ["QWEN_API_KEY"], "config_key": "qwen_api_key"},
    {"name": "mistral", "label": "Mistral", "env_keys": ["MISTRAL_API_KEY"], "config_key": "mistral_api_key"},
    {"name": "google", "label": "Google", "env_keys": ["GOOGLE_API_KEY"], "config_key": "google_api_key"},
    {
        "name": "elevenlabs",
        "label": "ElevenLabs",
        "env_keys": ["ELEVENLABS_API_KEY"],
        "config_key": "elevenlabs_api_key",
    },
    {
        "name": "bedrock",
        "label": "Bedrock",
        "env_keys": ["BEDROCK_API_KEY", "AWS_BEARER_TOKEN_BEDROCK"],
        "config_key": "bedrock_api_key",
    },
    {
        "name": "custom_openai",
        "label": "Custom OpenAI",
        "env_keys": ["CUSTOM_OPENAI_API_KEY"],
        "config_key": "custom_openai_api_key",
    },
]

_MCP_CLIENTS = {
    "cursor": {
        "label": "Cursor",
        "darwin": [Path("~/Library/Application Support/Cursor/User/settings.json")],
        "linux": [Path("~/.config/Cursor/User/settings.json")],
        "windows": [Path("Cursor/User/settings.json")],
    },
    "claude": {
        "label": "Claude",
        "darwin": [Path("~/Library/Application Support/Claude/claude_desktop_config.json")],
        "linux": [Path("~/.config/Claude/claude_desktop_config.json")],
        "windows": [Path("Claude/claude_desktop_config.json")],
    },
    "vscode": {
        "label": "VS Code",
        "darwin": [Path("~/Library/Application Support/Code/User/settings.json")],
        "linux": [Path("~/.config/Code/User/settings.json")],
        "windows": [Path("Code/User/settings.json")],
    },
    "zed": {
        "label": "Zed",
        "darwin": [Path("~/Library/Application Support/Zed/settings.json")],
        "linux": [Path("~/.config/zed/settings.json")],
        "windows": [Path("Zed/settings.json")],
    },
}

_DEFAULT_MCP_URL = "ws://127.0.0.1:8000/api/v1/mcp/ws"
_MCP_SERVER_NAME = "tldw_server"


def _platform_key() -> str:
    if sys.platform.startswith("win"):
        return "windows"
    if sys.platform == "darwin":
        return "darwin"
    return "linux"


def _resolve_mcp_candidate_paths(client: str) -> list[Path]:
    data = _MCP_CLIENTS.get(client, {})
    key = _platform_key()
    candidates = data.get(key, [])
    resolved: list[Path] = []
    if key == "windows":
        base = os.getenv("APPDATA") or os.getenv("LOCALAPPDATA")
        base_path = Path(base).expanduser() if base else (Path.home() / "AppData" / "Roaming")
        for path in candidates:
            candidate = path
            if not candidate.is_absolute():
                candidate = base_path / candidate
            resolved.append(candidate.expanduser().resolve())
    else:
        for path in candidates:
            resolved.append(path.expanduser().resolve())
    return resolved


def _load_json_file(path: Path) -> tuple[dict[str, Any], str]:
    if not path.exists():
        return {}, ""
    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        return {}, raw
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a JSON object at the top level.")
    return data, raw


def _render_json(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def _ensure_writable_file(path: Path) -> bool:
    existed = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a"):
        pass
    return not existed


def _port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _probe_endpoint(base_url: str, path: str, *, timeout: float = 2.0) -> dict[str, Any]:
    import httpx

    url = f"{base_url}{path}"
    try:
        resp = httpx.get(url, timeout=timeout)
        return {"url": url, "status_code": resp.status_code, "ok": resp.status_code < 400}
    except (httpx.HTTPError, OSError, TimeoutError, ValueError) as exc:
        return {"url": url, "ok": False, "error": str(exc)}


def _check_endpoints(base_url: str) -> dict[str, dict[str, Any]]:
    return {
        "api_health": _probe_endpoint(base_url, "/api/v1/health"),
        "healthz": _probe_endpoint(base_url, "/api/v1/healthz"),
        "mcp_status": _probe_endpoint(base_url, "/api/v1/mcp/status"),
    }


def _start_ephemeral_server(port: int, env: dict[str, str]) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "tldw_Server_API.app.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "warning",
    ]
    return subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # nosec B603


def _stop_process(proc: subprocess.Popen, timeout: float = 5.0) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()


def _build_initializer_env(env_path: Path, updates: dict[str, str | None]) -> dict[str, str]:
    env = os.environ.copy()
    env.update(env_utils.load_env(env_path))
    env.update({key: str(value) for key, value in updates.items() if value is not None})
    return env


@app.command()
def init(
    default: bool = typer.Option(False, "--default", help="Apply safe defaults without prompting"),
    install_dir: Path = typer.Option(Path.cwd(), "--install-dir", help="Installation directory (default: CWD)"),
    profile: str | None = typer.Option(None, "--profile", help="Public setup profile"),
    env_file: Path | None = typer.Option(None, "--env-file", help="Explicit env file path"),
    admin_username: str | None = typer.Option(None, "--admin-username", help="Initial multi-user admin username"),
    admin_password: str | None = typer.Option(None, "--admin-password", help="Initial multi-user admin password"),
    admin_email: str | None = typer.Option(None, "--admin-email", help="Initial multi-user admin email"),
    non_interactive: bool = typer.Option(False, "--non-interactive", help="Run without prompts using env vars"),
    debug: bool = typer.Option(False, "--debug", help="Verbose logging"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview actions without writing"),
    json_out: bool = typer.Option(False, "--json", help="Emit machine-readable JSON output"),
    yes: bool = typer.Option(False, "--yes", "--no-input", help="Assume 'yes' for prompts (non-interactive)"),
    no_format: bool = typer.Option(False, "--no-format", help="Skip formatting changed files"),
    no_color: bool = typer.Option(False, "--no-color", help="Disable colored output"),
    quiet: bool = typer.Option(False, "--quiet", help="Minimal output"),
):
    """Full guided setup (scaffold)."""
    if debug:
        logger.remove()
        logger.add(lambda m: typer.echo(m, nl=False), level="DEBUG")

    base = Path(install_dir).resolve()
    facts = {
        "cwd": str(Path.cwd()),
        "install_dir": str(base),
        "git": git_utils.is_git_repo(base),
        "ffmpeg": detect_utils.has_ffmpeg(),
        "cuda": detect_utils.has_cuda(),
    }

    # Plan actions (skeleton)
    try:
        setup_profile = profile_utils.normalize_profile(profile) if profile else None
    except ValueError as exc:
        result = {
            "command": "init",
            "status": "error",
            "facts": facts,
            "actions": [{"profile": {"valid": False, "reason": str(exc)}}],
            "notes": [str(exc)],
        }
        _emit(result, json_out)
        raise typer.Exit(2) from exc

    env_path = (
        profile_utils.resolve_env_path(
            profile=setup_profile,
            start_dir=base,
            explicit_env_file=env_file,
        )
        if setup_profile
        else (env_file.expanduser().resolve() if env_file is not None else base / ".env")
    )
    actions = []
    if not env_path.exists():
        actions.append({"create": str(env_path)})
    actions.append({"ensure_gitignore": _ENV_GITIGNORE_ENTRIES})

    uses_structured_postgres = bool(setup_profile and profile_utils.profile_uses_structured_postgres(setup_profile))
    existing_env = env_utils.load_env(env_path, raw_values=uses_structured_postgres)
    updates: dict[str, str | None] = {}
    if setup_profile:
        requested_admin_username = admin_username or os.getenv("ADMIN_USERNAME")
        try:
            updates.update(
                profile_utils.build_profile_env(
                    profile=setup_profile,
                    existing_env=existing_env,
                    admin_username=requested_admin_username,
                    admin_password=admin_password or os.getenv("ADMIN_PASSWORD"),
                    admin_email=admin_email or os.getenv("ADMIN_EMAIL"),
                )
            )
        except ValueError as exc:
            result = {
                "command": "init",
                "status": "error",
                "facts": facts,
                "actions": [{"admin_username": {"valid": False, "reason": str(exc)}}],
                "notes": [str(exc)],
            }
            _emit(result, json_out)
            raise typer.Exit(2) from exc
        auth_mode = updates["AUTH_MODE"]
    else:
        if env_file is not None:
            auth_mode = (
                existing_env.get("AUTH_MODE") or os.getenv("AUTH_MODE") or ("single_user" if default or yes else "")
            )
        else:
            auth_mode = (
                os.getenv("AUTH_MODE") or existing_env.get("AUTH_MODE") or ("single_user" if default or yes else "")
            )
    initializer_action: dict[str, Any] | None = None
    validation_action: dict[str, Any] | None = None
    if auth_mode:
        updates["AUTH_MODE"] = auth_mode
    if auth_mode == "single_user":
        if setup_profile and updates.get("SINGLE_USER_API_KEY"):
            existing_key = updates["SINGLE_USER_API_KEY"]
        elif env_file is not None:
            existing_key = (
                existing_env.get("SINGLE_USER_API_KEY") or os.getenv("SINGLE_USER_API_KEY") or os.getenv("API_KEY")
            )
        else:
            existing_key = (
                os.getenv("SINGLE_USER_API_KEY") or os.getenv("API_KEY") or existing_env.get("SINGLE_USER_API_KEY")
            )
        if not existing_key:
            existing_key = env_utils.generate_single_user_api_key()
        updates["SINGLE_USER_API_KEY"] = existing_key
    if auth_mode == "multi_user":
        if uses_structured_postgres:
            db_url = profile_utils.build_postgres_database_url(
                user=updates.get("POSTGRES_USER") or "tldw_user",
                password=updates.get("POSTGRES_PASSWORD") or "",
                db=updates.get("POSTGRES_DB") or "tldw_users",
            )
        else:
            db_url = updates.get("DATABASE_URL") or _resolve_database_url(
                env_path,
                prefer_env_file=env_file is not None,
            )
        if not db_url:
            result = {
                "command": "init",
                "status": "error",
                "facts": facts,
                "actions": [{"validate_database_url": {"present": False, "valid": False, "reason": "missing"}}],
                "notes": ["DATABASE_URL is required for multi_user mode."],
            }
            _emit(result, json_out)
            raise typer.Exit(2)
        valid, reason = _validate_database_url(db_url)
        if not valid:
            result = {
                "command": "init",
                "status": "error",
                "facts": facts,
                "actions": [{"validate_database_url": {"present": True, "valid": False, "reason": reason}}],
                "notes": [f"DATABASE_URL invalid for multi_user: {reason}"],
            }
            _emit(result, json_out)
            raise typer.Exit(2)
        validation_action = {"validate_database_url": {"present": True, "valid": True, "reason": None}}
        if not uses_structured_postgres:
            updates["DATABASE_URL"] = db_url
        cmd = [sys.executable, "-m", "tldw_Server_API.app.core.AuthNZ.initialize"]
        if setup_profile and setup_profile.docker:
            initializer_action = {
                "command": "AuthNZ initializer",
                "status": "deferred_to_docker",
                "reason": "Docker profiles initialize AuthNZ inside the container.",
            }
        elif dry_run:
            if yes:
                initializer_action = {"command": " ".join(cmd), "status": "would_run"}
            elif non_interactive or not sys.stdin.isatty():
                initializer_action = {"command": " ".join(cmd), "status": "skipped_non_interactive"}
            else:
                initializer_action = {"command": " ".join(cmd), "status": "would_prompt"}
        else:
            if yes:
                proc = subprocess.run(cmd, check=False, env=_build_initializer_env(env_path, updates))  # nosec
                initializer_action = {"command": " ".join(cmd), "returncode": proc.returncode}
            elif non_interactive or not sys.stdin.isatty():
                initializer_action = {"command": " ".join(cmd), "status": "skipped_non_interactive"}
            else:
                if typer.confirm("Run AuthNZ initializer now?", default=False):
                    proc = subprocess.run(cmd, check=False, env=_build_initializer_env(env_path, updates))  # nosec
                    initializer_action = {"command": " ".join(cmd), "returncode": proc.returncode}
                else:
                    initializer_action = {"command": "AuthNZ initializer", "status": "skipped"}

    if dry_run:
        if updates:
            actions.append({"set_env": env_utils.mask_env_values({k: v for k, v in updates.items() if v is not None})})
        if validation_action:
            actions.append(validation_action)
        if initializer_action:
            actions.append({"authnz_initializer": initializer_action})
        result = {
            "command": "init",
            "status": "ok",
            "facts": facts,
            "actions": actions,
            "paths": {"env": str(env_path)},
            "notes": [
                "dry-run only; no changes made",
            ],
        }
        _emit(result, json_out)
        raise typer.Exit(0)
    env_utils.ensure_env(
        env_path,
        updates=updates,
        remove_keys={"DATABASE_URL", "JOBS_DB_URL"} if uses_structured_postgres else None,
        raw_values=uses_structured_postgres,
    )

    # Ensure .gitignore entries
    files_utils.ensure_gitignore(base / ".gitignore", entries=_ENV_GITIGNORE_ENTRIES)

    if updates:
        actions.append({"set_env": env_utils.mask_env_values({k: v for k, v in updates.items() if v is not None})})
    if validation_action:
        actions.append(validation_action)
    if initializer_action:
        actions.append({"authnz_initializer": initializer_action})

    result = {
        "command": "init",
        "status": "ok",
        "facts": facts,
        "actions": actions,
        "paths": {"env": str(env_path)},
    }
    _emit(result, json_out)


@app.command()
def auth(
    mode: str = typer.Option(
        "single_user",
        "--mode",
        help="Authentication mode",
        case_sensitive=False,
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit machine-readable JSON output"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview actions without writing"),
    yes: bool = typer.Option(False, "--yes", "--no-input", help="Assume 'yes' for prompts (non-interactive)"),
):
    """Configure Auth mode (scaffold)."""
    mode = mode.lower().strip()
    if mode not in {"single_user", "multi_user"}:
        typer.echo("Invalid mode. Use single_user or multi_user.")
        raise typer.Exit(2)
    env_path = Path.cwd() / ".env"
    existing_env = env_utils.load_env(env_path)
    updates: dict[str, str | None] = {"AUTH_MODE": mode}
    notes = []
    actions = []
    validation: dict[str, Any] = {}
    initializer_action: dict[str, Any] | None = None
    if mode == "single_user":
        existing_key = (
            os.getenv("SINGLE_USER_API_KEY") or os.getenv("API_KEY") or existing_env.get("SINGLE_USER_API_KEY")
        )
        if not existing_key:
            if dry_run:
                notes.append("would_generate_single_user_api_key")
            else:
                existing_key = env_utils.generate_single_user_api_key()
        if existing_key:
            updates["SINGLE_USER_API_KEY"] = existing_key
    if mode == "multi_user":
        db_url = _resolve_database_url(env_path)
        if not db_url:
            validation = {"database_url": {"present": False, "valid": False, "reason": "missing"}}
            notes.append("DATABASE_URL is required for multi_user mode.")
            result = {
                "command": "auth",
                "status": "error",
                "mode": mode,
                "actions": [{"validate_database_url": validation}],
                "dry_run": dry_run,
                "notes": notes,
                "paths": {"env": str(env_path)},
            }
            _emit(result, json_out)
            raise typer.Exit(2)
        valid, reason = _validate_database_url(db_url)
        validation = {"database_url": {"present": True, "valid": valid, "reason": reason or None}}
        if not valid:
            notes.append(f"DATABASE_URL invalid for multi_user: {reason}")
            result = {
                "command": "auth",
                "status": "error",
                "mode": mode,
                "actions": [{"validate_database_url": validation}],
                "dry_run": dry_run,
                "notes": notes,
                "paths": {"env": str(env_path)},
            }
            _emit(result, json_out)
            raise typer.Exit(2)
        cmd = [sys.executable, "-m", "tldw_Server_API.app.core.AuthNZ.initialize"]
        if dry_run:
            if yes:
                initializer_action = {"command": " ".join(cmd), "status": "would_run"}
            elif sys.stdin.isatty():
                initializer_action = {"command": " ".join(cmd), "status": "would_prompt"}
            else:
                initializer_action = {"command": " ".join(cmd), "status": "skipped_non_interactive"}
                notes.append("Non-interactive session; skipping AuthNZ initializer prompt.")
        else:
            if yes:
                proc = subprocess.run(cmd, check=False)  # nosec B603
                initializer_action = {"command": " ".join(cmd), "returncode": proc.returncode}
                if proc.returncode != 0:
                    notes.append("AuthNZ initializer failed; see output for details.")
            elif sys.stdin.isatty():
                if typer.confirm("Run AuthNZ initializer now?", default=False):
                    proc = subprocess.run(cmd, check=False)  # nosec B603
                    initializer_action = {"command": " ".join(cmd), "returncode": proc.returncode}
                    if proc.returncode != 0:
                        notes.append("AuthNZ initializer failed; see output for details.")
                else:
                    initializer_action = {"command": "AuthNZ initializer", "status": "skipped"}
                    notes.append("Skipped AuthNZ initializer; run it later if needed.")
            else:
                notes.append("Non-interactive session; skipping AuthNZ initializer prompt.")
    env_result = env_utils.ensure_env(env_path, updates=updates, dry_run=dry_run)
    masked = env_utils.mask_env_values({k: v for k, v in updates.items() if v is not None})
    actions.append({"set_env": masked})
    if env_result.backup_path:
        actions.append({"backup": str(env_result.backup_path)})
    if validation:
        actions.append({"validate_database_url": validation})
    if initializer_action:
        actions.append({"authnz_initializer": initializer_action})
    result = {
        "command": "auth",
        "status": "ok",
        "mode": mode,
        "actions": actions,
        "dry_run": dry_run,
        "notes": notes,
        "paths": {"env": str(env_path)},
    }
    _emit(result, json_out)


@app.command()
def verify(
    json_out: bool = typer.Option(False, "--json", help="Emit machine-readable JSON output"),
    check_provider: bool = typer.Option(
        False, "--check-provider", help="Attempt provider checks (offline/mock in scaffold)"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview actions without writing"),
    profile: str | None = typer.Option(None, "--profile", help="Public setup profile"),
    env_file: Path | None = typer.Option(None, "--env-file", help="Explicit env file path"),
    base_url: str | None = typer.Option(None, "--base-url", help="API base URL for --profile verification"),
    webui_url: str | None = typer.Option(None, "--webui-url", help="WebUI base URL for --profile verification"),
    first_value: bool = typer.Option(False, "--first-value", help="Run first ingest/search checks"),
):
    """Run verification checks (Stage 4)."""
    facts: dict[str, Any] = {
        "ffmpeg": detect_utils.has_ffmpeg(),
        "cuda": detect_utils.has_cuda(),
    }
    notes: list[str] = []
    actions: list[dict[str, Any]] = []

    if profile:
        try:
            setup_profile = profile_utils.normalize_profile(profile)
        except ValueError as exc:
            result = {
                "command": "verify",
                "status": "error",
                "facts": facts,
                "actions": [{"profile": {"valid": False, "reason": str(exc)}}],
                "notes": [str(exc)],
                "check_provider": bool(check_provider),
                "dry_run": dry_run,
            }
            _emit(result, json_out)
            raise typer.Exit(2) from exc

        env_path = profile_utils.resolve_env_path(
            profile=setup_profile,
            start_dir=Path.cwd(),
            explicit_env_file=env_file,
        )
        resolved_base_url = base_url or setup_profile.default_base_url
        resolved_webui_url = webui_url if webui_url is not None else setup_profile.default_webui_url
        facts.update(
            {
                "profile": setup_profile.name,
                "base_url": profile_verify._sanitize_url_userinfo(resolved_base_url),
                "webui_url": profile_verify._sanitize_url_userinfo(resolved_webui_url),
            }
        )
        if dry_run:
            result = {
                "command": "verify",
                "status": "ok",
                "facts": facts,
                "actions": [{"server": {"mode": "dry_run", "profile": setup_profile.name}}],
                "notes": ["dry-run only; skipping profile probes."],
                "check_provider": bool(check_provider),
                "dry_run": True,
                "paths": {"env": str(env_path)},
            }
            _emit(result, json_out)
            raise typer.Exit(0)

        check_result = profile_verify.run_profile_checks(
            profile=setup_profile,
            base_url=resolved_base_url,
            webui_url=resolved_webui_url,
            env_path=env_path,
            first_value=first_value,
            check_provider=bool(check_provider),
            timeout=5.0,
        )
        result = {
            "command": "verify",
            "status": check_result.get("status", "error"),
            "facts": facts,
            "actions": check_result.get("actions", []),
            "notes": check_result.get("notes", []),
            "check_provider": bool(check_provider),
            "dry_run": False,
            "paths": {"env": str(env_path)},
        }
        _emit(result, json_out)
        if result["status"] != "ok":
            raise typer.Exit(2)
        raise typer.Exit(0)

    env_port = os.getenv("TLDW_SERVER_PORT")
    preferred_port = int(env_port) if env_port and env_port.isdigit() else 8000
    base_url = f"http://127.0.0.1:{preferred_port}"

    if dry_run:
        actions.append({"server": {"mode": "dry_run", "port": preferred_port}})
        actions.append(
            {
                "endpoints": {
                    "api_health": {"url": f"{base_url}/api/v1/health", "status": "skipped_dry_run"},
                    "healthz": {"url": f"{base_url}/api/v1/healthz", "status": "skipped_dry_run"},
                    "mcp_status": {"url": f"{base_url}/api/v1/mcp/status", "status": "skipped_dry_run"},
                }
            }
        )
        notes.append("dry-run only; skipping endpoint probes and server spawn.")
        result = {
            "command": "verify",
            "status": "ok",
            "facts": facts,
            "actions": actions,
            "notes": notes,
            "check_provider": bool(check_provider),
            "dry_run": True,
        }
        _emit(result, json_out)
        raise typer.Exit(0)
    probe = _probe_endpoint(base_url, "/api/v1/healthz")
    server_running = "status_code" in probe
    server_mode = "existing" if server_running else "spawned"
    server_port = preferred_port
    proc: subprocess.Popen | None = None

    if not server_running:
        if env_port:
            if not _port_available(preferred_port):
                facts.update({"server_mode": "unavailable", "server_port": preferred_port})
                result = {
                    "command": "verify",
                    "status": "error",
                    "facts": facts,
                    "actions": [{"server": {"port": preferred_port, "error": "port_in_use"}}],
                    "notes": ["TLDW_SERVER_PORT is in use; cannot start ephemeral server."],
                    "check_provider": bool(check_provider),
                    "dry_run": dry_run,
                }
                _emit(result, json_out)
                raise typer.Exit(2)
            server_port = preferred_port
        else:
            server_port = _pick_free_port()

        base_url = f"http://127.0.0.1:{server_port}"
        env = os.environ.copy()
        env.setdefault("DEFER_HEAVY_STARTUP", "true")
        proc = _start_ephemeral_server(server_port, env)

        ready = False
        for _ in range(20):
            time.sleep(0.5)
            ready_probe = _probe_endpoint(base_url, "/api/v1/healthz")
            if "status_code" in ready_probe:
                ready = True
                break
        if not ready:
            if proc:
                _stop_process(proc)
            facts.update({"server_mode": "unavailable", "server_port": server_port})
            result = {
                "command": "verify",
                "status": "error",
                "facts": facts,
                "actions": [{"server": {"port": server_port, "error": "startup_timeout"}}],
                "notes": ["Ephemeral server failed to start within timeout."],
                "check_provider": bool(check_provider),
                "dry_run": dry_run,
            }
            _emit(result, json_out)
            raise typer.Exit(2)

    facts.update({"server_mode": server_mode, "server_port": server_port})
    if proc:
        actions.append({"server": {"mode": "spawned", "port": server_port}})
    else:
        actions.append({"server": {"mode": "existing", "port": server_port}})

    endpoint_results = _check_endpoints(base_url)
    actions.append({"endpoints": endpoint_results})

    if proc:
        _stop_process(proc)

    status = "ok" if all(result.get("ok") for result in endpoint_results.values()) else "error"
    if status != "ok":
        notes.append("One or more endpoints failed checks.")
    result = {
        "command": "verify",
        "status": status,
        "facts": facts,
        "actions": actions,
        "notes": notes,
        "check_provider": bool(check_provider),
        "dry_run": dry_run,
    }
    _emit(result, json_out)
    if status != "ok":
        raise typer.Exit(2)


@app.command()
def providers(
    json_out: bool = typer.Option(False, "--json", help="Emit machine-readable JSON output"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview actions without writing"),
    check_provider: bool = typer.Option(
        False, "--check-provider", help="Attempt provider checks (offline/mock in scaffold)"
    ),
    write_config: bool = typer.Option(False, "--write-config", help="Write provider keys to config.txt when set"),
):
    """Collect/store provider keys."""
    env_path = Path.cwd() / ".env"
    existing_env = env_utils.load_env(env_path)
    actions: list[dict[str, Any]] = []
    notes: list[str] = []
    env_updates: dict[str, str] = {}
    config_updates: dict[str, str] = {}
    provider_status: list[dict[str, Any]] = []

    for provider in _PROVIDER_SOURCES:
        value = None
        source = None
        used_key = None
        for env_key in provider["env_keys"]:
            env_value = os.getenv(env_key)
            if env_value:
                value = env_value
                source = "env"
                used_key = env_key
                break
        if value is None:
            for env_key in provider["env_keys"]:
                env_value = existing_env.get(env_key)
                if env_value:
                    value = env_value
                    source = "env_file"
                    used_key = env_key
                    break

        if value:
            provider_status.append(
                {
                    "provider": provider["name"],
                    "label": provider["label"],
                    "env_key": used_key,
                    "source": source,
                    "status": "found",
                }
            )
            if source == "env":
                env_updates[used_key] = value
            if write_config:
                config_updates[provider["config_key"]] = value
        else:
            provider_status.append(
                {
                    "provider": provider["name"],
                    "label": provider["label"],
                    "env_key": provider["env_keys"][0],
                    "status": "missing",
                }
            )

    actions.append({"providers": provider_status})

    if env_updates:
        env_result = env_utils.ensure_env(env_path, updates=env_updates, dry_run=dry_run)
        actions.append({"set_env": env_utils.mask_env_values(env_updates)})
        if env_result.backup_path:
            actions.append({"backup": str(env_result.backup_path)})
    else:
        notes.append("No provider keys found in environment variables.")

    if write_config:
        from tldw_Server_API.app.core.config_paths import resolve_config_file

        config_path = resolve_config_file()
        config_content = config_path.read_text(encoding="utf-8") if config_path.exists() else ""
        created = False
        if not config_path.exists():
            minimal = "[API]\n"
            if not dry_run:
                files_utils.atomic_write(config_path, minimal)
            config_content = minimal
            created = True

        masked_updates = {key: env_utils.mask_value(value) for key, value in config_updates.items()}
        updated_content, changed, _missing = _update_ini_section(config_content, "API", config_updates)
        config_action: dict[str, Any] = {
            "path": str(config_path),
            "created": created,
            "updated_keys": list(masked_updates.keys()),
        }
        if changed:
            if dry_run:
                config_action["status"] = "would_update"
                config_action["diff"] = _render_unified_diff(config_content, updated_content, label=str(config_path))
            else:
                backup_path = None
                if config_path.exists() and config_content:
                    backup_path = _timestamped_backup(config_path, config_content)
                files_utils.atomic_write(config_path, updated_content)
                config_action["status"] = "updated"
                if backup_path:
                    config_action["backup"] = str(backup_path)
        else:
            config_action["status"] = "unchanged"
        config_action["masked_values"] = masked_updates
        actions.append({"config_txt": config_action})

    should_check = check_provider or is_truthy(os.getenv("TLDW_CHECK_PROVIDER"))
    if should_check:
        checks = []
        for entry in provider_status:
            if entry.get("source"):
                checks.append({"provider": entry["provider"], "status": "skipped_offline"})
        actions.append({"provider_checks": checks})

    result = {
        "command": "providers",
        "status": "ok",
        "actions": actions,
        "dry_run": dry_run,
        "check_provider": bool(should_check),
        "notes": notes,
        "paths": {"env": str(env_path)},
    }
    _emit(result, json_out)


@app.command()
def db(
    json_out: bool = typer.Option(False, "--json", help="Emit machine-readable JSON output"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview actions without writing"),
):
    """Initialize/validate databases (Stage 3)."""
    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths

    env_path = Path.cwd() / ".env"
    notes: list[str] = []
    actions: list[dict[str, Any]] = []

    existing_env = env_utils.load_env(env_path)
    auth_mode = os.getenv("AUTH_MODE") or existing_env.get("AUTH_MODE") or "single_user"
    db_url = _resolve_database_url(env_path)

    if auth_mode == "multi_user":
        if not db_url:
            result = {
                "command": "db",
                "status": "error",
                "actions": [{"validate_database_url": {"present": False, "valid": False, "reason": "missing"}}],
                "notes": ["DATABASE_URL is required for multi_user mode."],
                "dry_run": dry_run,
            }
            _emit(result, json_out)
            raise typer.Exit(2)
        valid, reason = _validate_database_url(db_url)
        if not valid:
            result = {
                "command": "db",
                "status": "error",
                "actions": [{"validate_database_url": {"present": True, "valid": False, "reason": reason}}],
                "notes": [f"DATABASE_URL invalid for multi_user: {reason}"],
                "dry_run": dry_run,
            }
            _emit(result, json_out)
            raise typer.Exit(2)

    if not db_url:
        db_url = "sqlite:///./Databases/users.db"
        notes.append("DATABASE_URL not set; using default sqlite path for AuthNZ.")

    scheme = (urlsplit(db_url).scheme or "").split("+", 1)[0].lower()
    if scheme and scheme not in {"sqlite", "file", "postgres", "postgresql"}:
        result = {
            "command": "db",
            "status": "error",
            "actions": [
                {"validate_database_url": {"present": True, "valid": False, "reason": f"unsupported scheme '{scheme}'"}}
            ],
            "notes": [f"Unsupported DATABASE_URL scheme: {scheme}"],
            "dry_run": dry_run,
        }
        _emit(result, json_out)
        raise typer.Exit(2)

    if dry_run:
        if scheme in {"postgres", "postgresql"}:
            actions.append({"postgres_check": {"status": "skipped_dry_run"}})
        else:
            auth_db_path = _resolve_sqlite_db_path(db_url)
            if auth_db_path is None:
                notes.append("AuthNZ sqlite path not resolved (maybe :memory:); skipping file creation.")
            else:
                actions.append({"authnz_db": {"path": str(auth_db_path), "status": "would_create"}})

        user_id = DatabasePaths.get_single_user_id()
        base_dir = _resolve_user_db_base_dir_for_dry_run()
        user_dir = base_dir / str(user_id)
        sqlite_files = [
            {"name": "media", "path": str(user_dir / DatabasePaths.MEDIA_DB_NAME), "status": "would_create"},
            {"name": "chacha", "path": str(user_dir / DatabasePaths.CHACHA_DB_NAME), "status": "would_create"},
            {
                "name": "evaluations",
                "path": str(user_dir / DatabasePaths.EVALUATIONS_SUBDIR / DatabasePaths.EVALUATIONS_DB_NAME),
                "status": "would_create",
            },
            {
                "name": "evaluations_shared",
                "path": str((Path.cwd() / "Databases" / DatabasePaths.EVALUATIONS_DB_NAME).resolve()),
                "status": "would_create",
            },
        ]
        actions.append({"sqlite_files": sqlite_files})
        result = {"command": "db", "status": "ok", "actions": actions, "notes": notes, "dry_run": True}
        _emit(result, json_out)
        raise typer.Exit(0)

    if scheme in {"postgres", "postgresql"}:
        try:
            from tldw_Server_API.app.core.AuthNZ.database import test_database_connection

            ok = asyncio.run(test_database_connection())
        except _WIZARD_NONCRITICAL_EXCEPTIONS as exc:
            ok = False
            notes.append(f"Postgres validation failed: {exc}")
        actions.append({"postgres_check": {"status": "ok" if ok else "error"}})
        if not ok:
            result = {"command": "db", "status": "error", "actions": actions, "notes": notes, "dry_run": dry_run}
            _emit(result, json_out)
            raise typer.Exit(2)
    else:
        auth_db_path = _resolve_sqlite_db_path(db_url)
        if auth_db_path is None:
            notes.append("AuthNZ sqlite path not resolved (maybe :memory:); skipping file creation.")
        else:
            try:
                created = _ensure_writable_file(auth_db_path)
                actions.append({"authnz_db": {"path": str(auth_db_path), "created": created}})
            except OSError as exc:
                result = {
                    "command": "db",
                    "status": "error",
                    "actions": [{"authnz_db": {"path": str(auth_db_path), "error": str(exc)}}],
                    "dry_run": dry_run,
                }
                _emit(result, json_out)
                raise typer.Exit(2) from exc

    user_id = DatabasePaths.get_single_user_id()
    if not DatabasePaths.validate_database_structure(user_id):
        result = {
            "command": "db",
            "status": "error",
            "actions": [{"sqlite_structure": {"status": "error", "user_id": str(user_id)}}],
            "dry_run": dry_run,
        }
        _emit(result, json_out)
        raise typer.Exit(2)

    sqlite_files: list[dict[str, Any]] = []
    db_paths = {
        "media": DatabasePaths.get_media_db_path(user_id),
        "chacha": DatabasePaths.get_chacha_db_path(user_id),
        "evaluations": DatabasePaths.get_evaluations_db_path(user_id),
        "evaluations_shared": (Path.cwd() / "Databases" / "evaluations.db").resolve(),
    }
    for name, path in db_paths.items():
        try:
            created = _ensure_writable_file(path)
            sqlite_files.append({"name": name, "path": str(path), "created": created})
        except OSError as exc:
            result = {
                "command": "db",
                "status": "error",
                "actions": [{"sqlite_db": {"name": name, "path": str(path), "error": str(exc)}}],
                "dry_run": dry_run,
            }
            _emit(result, json_out)
            raise typer.Exit(2) from exc

    actions.append({"sqlite_files": sqlite_files})
    result = {"command": "db", "status": "ok", "actions": actions, "notes": notes, "dry_run": dry_run}
    _emit(result, json_out)


@app.command()
def mcp(
    action: str = typer.Argument("add", metavar="[add|remove]", help="Add or remove MCP client configs"),
    clients: list[str] = typer.Option(None, "--client", "-c", help="Client(s) to configure"),
    config_path: Path | None = typer.Option(None, "--config-path", help="Override config path (single client)"),
    server_url: str | None = typer.Option(None, "--server-url", help="MCP server URL"),
    json_out: bool = typer.Option(False, "--json", help="Emit machine-readable JSON output"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview actions without writing"),
    yes: bool = typer.Option(False, "--yes", "--no-input", help="Assume 'yes' for prompts (non-interactive)"),
):
    """Install/remove MCP client config."""
    action = action.lower().strip()
    if action not in {"add", "remove"}:
        typer.echo("Invalid action. Use add or remove.")
        raise typer.Exit(2)

    requested = clients or list(_MCP_CLIENTS.keys())
    normalized: list[str] = []
    for client in requested:
        key = client.lower().strip()
        if key not in _MCP_CLIENTS:
            typer.echo(f"Unknown client '{client}'. Supported: {', '.join(sorted(_MCP_CLIENTS.keys()))}")
            raise typer.Exit(2)
        normalized.append(key)

    if config_path and len(normalized) != 1:
        typer.echo("--config-path requires a single --client selection.")
        raise typer.Exit(2)

    server_url = server_url or os.getenv("TLDW_MCP_URL") or _DEFAULT_MCP_URL
    transport = "websocket" if server_url.startswith("ws") else "http"
    entry = {
        "url": server_url,
        "transport": transport,
        "headers": {"X-API-KEY": "YOUR_API_KEY"},
    }

    if action == "remove" and not yes:
        if sys.stdin.isatty():
            if not typer.confirm("Remove MCP config from selected clients?", default=False):
                actions = []
                for client in normalized:
                    actions.append(
                        {
                            "mcp_client": {
                                "client": client,
                                "label": _MCP_CLIENTS[client]["label"],
                                "status": "skipped",
                                "reason": "cancelled",
                            }
                        }
                    )
                result = {"command": "mcp", "status": "ok", "actions": actions, "dry_run": dry_run}
                _emit(result, json_out)
                return
        else:
            actions = []
            for client in normalized:
                actions.append(
                    {
                        "mcp_client": {
                            "client": client,
                            "label": _MCP_CLIENTS[client]["label"],
                            "status": "skipped_non_interactive",
                        }
                    }
                )
            result = {"command": "mcp", "status": "ok", "actions": actions, "dry_run": dry_run}
            _emit(result, json_out)
            return

    actions: list[dict[str, Any]] = []
    for client in normalized:
        label = _MCP_CLIENTS[client]["label"]
        if config_path:
            candidate_path = config_path.expanduser().resolve()
            detected = True
        else:
            candidates = _resolve_mcp_candidate_paths(client)
            candidate_path = candidates[0] if candidates else Path.cwd() / f"{client}_settings.json"
            detected = any(path.exists() or path.parent.exists() for path in candidates)

        if not detected and not config_path:
            actions.append(
                {
                    "mcp_client": {
                        "client": client,
                        "label": label,
                        "path": str(candidate_path),
                        "status": "not_found",
                    }
                }
            )
            continue

        if action == "remove" and not candidate_path.exists():
            actions.append(
                {
                    "mcp_client": {
                        "client": client,
                        "label": label,
                        "path": str(candidate_path),
                        "status": "missing",
                    }
                }
            )
            continue

        try:
            data, raw = _load_json_file(candidate_path)
        except _WIZARD_NONCRITICAL_EXCEPTIONS as exc:
            actions.append(
                {
                    "mcp_client": {
                        "client": client,
                        "label": label,
                        "path": str(candidate_path),
                        "status": "error",
                        "error": str(exc),
                    }
                }
            )
            continue

        mcp_servers = data.get("mcpServers")
        if mcp_servers is None:
            mcp_servers = {}
        if not isinstance(mcp_servers, dict):
            actions.append(
                {
                    "mcp_client": {
                        "client": client,
                        "label": label,
                        "path": str(candidate_path),
                        "status": "error",
                        "error": "mcpServers must be a JSON object",
                    }
                }
            )
            continue

        changed = False
        status = "unchanged"

        if action == "add":
            existing = mcp_servers.get(_MCP_SERVER_NAME)
            if existing != entry:
                mcp_servers[_MCP_SERVER_NAME] = entry
                data["mcpServers"] = mcp_servers
                changed = True
                status = "updated" if candidate_path.exists() else "created"
        else:
            if _MCP_SERVER_NAME in mcp_servers:
                mcp_servers.pop(_MCP_SERVER_NAME, None)
                if mcp_servers:
                    data["mcpServers"] = mcp_servers
                else:
                    data.pop("mcpServers", None)
                changed = True
                status = "updated"

        new_content = _render_json(data)
        client_action: dict[str, Any] = {
            "client": client,
            "label": label,
            "path": str(candidate_path),
            "status": status,
        }
        if changed:
            if dry_run:
                client_action["diff"] = _render_unified_diff(raw or "{}", new_content, label=str(candidate_path))
            else:
                backup_path = None
                if candidate_path.exists() and raw:
                    backup_path = _timestamped_backup(candidate_path, raw)
                candidate_path.parent.mkdir(parents=True, exist_ok=True)
                files_utils.atomic_write(candidate_path, new_content)
                if backup_path:
                    client_action["backup"] = str(backup_path)
        actions.append({"mcp_client": client_action})

    result = {"command": "mcp", "status": "ok", "actions": actions, "dry_run": dry_run}
    _emit(result, json_out)


@app.command()
def format(
    json_out: bool = typer.Option(False, "--json", help="Emit machine-readable JSON output"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview actions without writing"),
):
    """Format changed files with Black/Ruff when available (scaffold)."""
    base = Path.cwd()
    actions: dict[str, Any] = {"formatted": []}
    if git_utils.is_git_repo(base):
        changed = git_utils.changed_or_untracked_files(base)
        if changed:
            try:
                if dry_run:
                    actions["would_format"] = changed
                else:
                    from .utils import format as format_utils

                    format_utils.maybe_format(changed)
                    actions["formatted"] = changed
            except _WIZARD_NONCRITICAL_EXCEPTIONS as e:
                actions["error"] = str(e)
    result = {"command": "format", "status": "ok", "actions": actions, "dry_run": dry_run}
    _emit(result, json_out)


@app.command()
def doctor(
    json_out: bool = typer.Option(False, "--json", help="Emit machine-readable JSON output"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview actions without writing"),
    yes: bool = typer.Option(False, "--yes", "--no-input", help="Assume 'yes' for prompts (non-interactive)"),
):
    """Detect issues and propose/apply fixes (scaffold)."""
    facts = {
        "ffmpeg": detect_utils.has_ffmpeg(),
        "git": git_utils.is_git_repo(Path.cwd()),
    }
    actions: list[dict[str, Any]] = []
    notes: list[str] = []
    had_error = False

    apply_fixes = False
    if dry_run:
        apply_fixes = False
    elif yes:
        apply_fixes = True
    elif sys.stdin.isatty():
        apply_fixes = typer.confirm("Apply recommended fixes?", default=False)
    else:
        notes.append("Non-interactive session; use --yes to apply fixes.")

    env_path = Path.cwd() / ".env"
    existing_env = env_utils.load_env(env_path)
    env_exists = env_path.exists()
    updates: dict[str, str] = {}
    env_action: dict[str, Any] = {"path": str(env_path), "status": "ok"}
    missing_keys: list[str] = []

    auth_mode = os.getenv("AUTH_MODE") or existing_env.get("AUTH_MODE")
    if not auth_mode:
        auth_mode = "single_user"
        missing_keys.append("AUTH_MODE")
        updates["AUTH_MODE"] = auth_mode
    else:
        if "AUTH_MODE" not in existing_env and os.getenv("AUTH_MODE"):
            updates["AUTH_MODE"] = auth_mode

    if auth_mode == "single_user":
        existing_key = (
            os.getenv("SINGLE_USER_API_KEY")
            or os.getenv("API_KEY")
            or existing_env.get("SINGLE_USER_API_KEY")
            or existing_env.get("API_KEY")
        )
        if not existing_key:
            missing_keys.append("SINGLE_USER_API_KEY")
            updates["SINGLE_USER_API_KEY"] = env_utils.generate_single_user_api_key()
        else:
            if "SINGLE_USER_API_KEY" not in existing_env and os.getenv("SINGLE_USER_API_KEY"):
                updates["SINGLE_USER_API_KEY"] = existing_key
    elif auth_mode == "multi_user":
        db_url = _resolve_database_url(env_path)
        if not db_url:
            actions.append({"validate_database_url": {"present": False, "valid": False, "reason": "missing"}})
        else:
            valid, reason = _validate_database_url(db_url)
            actions.append({"validate_database_url": {"present": True, "valid": valid, "reason": reason or None}})
            if valid and "DATABASE_URL" not in existing_env and os.getenv("DATABASE_URL"):
                updates["DATABASE_URL"] = db_url

    if not env_exists:
        env_action["status"] = "missing"
    if missing_keys:
        env_action["missing_keys"] = missing_keys

    port_raw = os.getenv("TLDW_SERVER_PORT") or existing_env.get("TLDW_SERVER_PORT")
    port = int(port_raw) if port_raw and port_raw.isdigit() else 8000
    if not _port_available(port):
        suggested = _pick_free_port()
        actions.append({"port": {"status": "in_use", "port": port, "suggested_port": suggested}})
        updates["TLDW_SERVER_PORT"] = str(suggested)

    if updates:
        masked = env_utils.mask_env_values(updates)
        if apply_fixes:
            try:
                env_result = env_utils.ensure_env(env_path, updates=updates, dry_run=False)
                env_action["status"] = "created" if env_result.created else "updated"
                actions.append({"set_env": masked})
                if env_result.backup_path:
                    actions.append({"backup": str(env_result.backup_path)})
            except _WIZARD_NONCRITICAL_EXCEPTIONS as exc:
                env_action["status"] = "error"
                env_action["error"] = str(exc)
                had_error = True
        else:
            env_action["status"] = "would_update" if dry_run else "recommended"
            actions.append({"set_env": masked})

    if env_action.get("status") != "ok":
        actions.append({"env": env_action})

    gitignore_path = Path.cwd() / ".gitignore"
    desired_entries = _ENV_GITIGNORE_ENTRIES
    existing_lines = gitignore_path.read_text(encoding="utf-8").splitlines() if gitignore_path.exists() else []
    existing_set = {line.strip() for line in existing_lines if line.strip()}
    missing_entries = [entry for entry in desired_entries if entry not in existing_set]
    if missing_entries:
        git_action = {"path": str(gitignore_path), "missing": missing_entries}
        if apply_fixes:
            try:
                files_utils.ensure_gitignore(gitignore_path, entries=desired_entries)
                git_action["status"] = "updated"
            except _WIZARD_NONCRITICAL_EXCEPTIONS as exc:
                git_action["status"] = "error"
                git_action["error"] = str(exc)
                had_error = True
        else:
            git_action["status"] = "would_update" if dry_run else "recommended"
        actions.append({"gitignore": git_action})

    if not facts["ffmpeg"]:
        actions.append(
            {
                "ffmpeg": {
                    "status": "missing",
                    "hint": "Install via brew/apt/choco or set PATH to ffmpeg.",
                }
            }
        )

    status = "error" if had_error else "ok"
    result = {
        "command": "doctor",
        "status": status,
        "facts": facts,
        "actions": actions,
        "dry_run": dry_run,
        "notes": notes,
    }
    _emit(result, json_out)
    if had_error:
        raise typer.Exit(2)


def main() -> None:
    """Console script entry point for tldw-setup."""
    app()


if __name__ == "__main__":  # pragma: no cover - script mode
    main()
