from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import SplitResult, urlsplit

from loguru import logger

from .files import atomic_write

_ENV_KEY_RE = re.compile(r"^\s*(export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)$")
_RAW_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SENSITIVE_TOKENS = ("KEY", "SECRET", "TOKEN", "PASSWORD")


@dataclass(frozen=True)
class ParsedEnvLine:
    raw: str
    key: str | None
    value: str | None
    export: bool


@dataclass(frozen=True)
class EnvUpdateResult:
    path: Path
    created: bool
    changed: bool
    backup_path: Path | None
    updated_keys: tuple[str, ...]
    added_keys: tuple[str, ...]
    dry_run: bool


def _strip_inline_comment(value: str) -> str:
    in_single = False
    in_double = False
    for idx, ch in enumerate(value):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double and (idx == 0 or value[idx - 1].isspace()):
            return value[:idx].rstrip()
    return value


def _normalize_value(raw_value: str) -> str:
    value = _strip_inline_comment(raw_value).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        quote = value[0]
        value = value[1:-1]
        if quote == '"':
            value = value.replace('\\"', '"').replace("\\\\", "\\")
    return value


def _format_env_value(value: str) -> str:
    if value == "":
        return ""
    needs_quotes = any(ch.isspace() for ch in value) or "#" in value
    if not needs_quotes:
        return value
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _format_env_line(key: str, value: str, export: bool = False, raw_values: bool = False) -> str:
    prefix = "export " if export else ""
    if raw_values:
        if "\n" in value or "\r" in value:
            raise ValueError(f"Env value for {key} cannot contain newline characters in raw mode")
        return f"{prefix}{key}={value}"
    return f"{prefix}{key}={_format_env_value(value)}"


def _parse_line(line: str) -> ParsedEnvLine:
    match = _ENV_KEY_RE.match(line)
    if not match:
        return ParsedEnvLine(raw=line, key=None, value=None, export=False)
    export_prefix = bool(match.group(1))
    key = match.group(2)
    value = _normalize_value(match.group(3) or "")
    return ParsedEnvLine(raw=line, key=key, value=value, export=export_prefix)


def _parse_raw_line(line: str) -> ParsedEnvLine:
    if not line.strip() or line.lstrip().startswith("#") or "=" not in line:
        return ParsedEnvLine(raw=line, key=None, value=None, export=False)
    key, value = line.split("=", 1)
    if not _RAW_ENV_KEY_RE.match(key):
        return ParsedEnvLine(raw=line, key=None, value=None, export=False)
    return ParsedEnvLine(raw=line, key=key, value=value, export=False)


def _backup_env(path: Path, content: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    backup = path.with_name(f"{path.name}.{timestamp}.bak")
    counter = 1
    while backup.exists():
        backup = path.with_name(f"{path.name}.{timestamp}.{counter}.bak")
        counter += 1
    atomic_write(backup, content)
    _chmod_600(backup)
    return backup


def _chmod_600(path: Path) -> None:
    try:
        os.chmod(path, 0o600)
    except Exception as exc:
        logger.debug(f"chmod on {path} ignored: {exc}")


def load_env(path: Path, *, raw_values: bool = False) -> dict[str, str]:
    if not path.exists():
        return {}
    content = path.read_text(encoding="utf-8")
    values: dict[str, str] = {}
    for line_number, line in enumerate(content.splitlines(), start=1):
        if raw_values and "\r" in line:
            raise ValueError(f"Invalid carriage return in env file {path}:{line_number}")
        parsed = _parse_raw_line(line) if raw_values else _parse_line(line)
        if raw_values and parsed.key is None and line.strip() and not line.lstrip().startswith("#"):
            raise ValueError(f"Invalid raw env line in {path}:{line_number}")
        if parsed.key:
            values[parsed.key] = parsed.value or ""
    return values


def is_sensitive_key(key: str) -> bool:
    key_upper = key.upper()
    return any(token in key_upper for token in _SENSITIVE_TOKENS)


def mask_value(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 4:
        return "*" * len(value)
    return "*" * (len(value) - 4) + value[-4:]


def _fallback_mask_url_credentials(value: str) -> str:
    marker = "://"
    marker_index = value.find(marker)
    if marker_index < 0:
        return value
    authority_start = marker_index + len(marker)
    at_index = value.find("@", authority_start)
    if at_index < 0:
        return value
    userinfo = value[authority_start:at_index]
    username = userinfo.split(":", 1)[0]
    credentials = f"{username}:********" if username else "********"
    return f"{value[:authority_start]}{credentials}{value[at_index:]}"


def _mask_url_credentials(value: str) -> str:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return _fallback_mask_url_credentials(value)
    if not parsed.scheme or not parsed.netloc or parsed.password is None:
        return _fallback_mask_url_credentials(value)

    if "@" not in parsed.netloc:
        return _fallback_mask_url_credentials(value)
    userinfo, hostinfo = parsed.netloc.rsplit("@", 1)
    username = userinfo.split(":", 1)[0]
    credentials = f"{username}:********" if username else "********"
    netloc = f"{credentials}@{hostinfo}"
    return SplitResult(parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment).geturl()


def mask_env_values(values: dict[str, str]) -> dict[str, str]:
    masked: dict[str, str] = {}
    for key, value in values.items():
        if is_sensitive_key(key):
            masked[key] = mask_value(value)
        else:
            masked[key] = _mask_url_credentials(value)
    return masked


def generate_single_user_api_key() -> str:
    from tldw_Server_API.app.core.AuthNZ.api_key_crypto import (
        format_api_key,
        generate_api_key_id,
        generate_api_key_secret,
    )

    return format_api_key(generate_api_key_id(), generate_api_key_secret())


def ensure_env(
    path: Path,
    *,
    updates: dict[str, str | None] | None = None,
    defaults: dict[str, str | None] | None = None,
    remove_keys: set[str] | frozenset[str] | tuple[str, ...] | list[str] | None = None,
    dry_run: bool = False,
    raw_values: bool = False,
) -> EnvUpdateResult:
    """Create or update a .env file idempotently with backups."""
    updates = updates or {}
    defaults = defaults or {}
    remove_keys_set = set(remove_keys or ())
    updates_clean = {k: v for k, v in updates.items() if v is not None}
    defaults_clean = {k: v for k, v in defaults.items() if v not in (None, "")}

    path.parent.mkdir(parents=True, exist_ok=True)
    existed = path.exists()
    content = path.read_text(encoding="utf-8") if existed else ""
    lines = content.splitlines() if content else []
    parsed = [_parse_raw_line(line) if raw_values else _parse_line(line) for line in lines]

    last_index: dict[str, int] = {}
    for idx, entry in enumerate(parsed):
        if entry.key:
            last_index[entry.key] = idx

    existing_keys = set(last_index.keys())
    updated_keys: list[str] = []
    added_keys: list[str] = []
    rendered: list[str] = []

    for idx, entry in enumerate(parsed):
        if entry.key is None:
            rendered.append(entry.raw)
            continue
        if last_index.get(entry.key) != idx:
            continue
        if entry.key in remove_keys_set:
            updated_keys.append(entry.key)
            continue
        if entry.key in updates_clean:
            rendered.append(
                _format_env_line(
                    entry.key,
                    str(updates_clean[entry.key]),
                    export=entry.export,
                    raw_values=raw_values,
                )
            )
            updated_keys.append(entry.key)
            continue
        rendered.append(entry.raw)

    for key, value in updates_clean.items():
        if key not in existing_keys:
            rendered.append(_format_env_line(key, str(value), raw_values=raw_values))
            added_keys.append(key)

    for key, value in defaults_clean.items():
        if key not in existing_keys and key not in updates_clean:
            rendered.append(_format_env_line(key, str(value), raw_values=raw_values))
            added_keys.append(key)

    new_content = "\n".join(rendered).rstrip("\n") + "\n" if rendered else ""

    changed = new_content != content
    backup_path: Path | None = None
    if existed and changed and not dry_run:
        backup_path = _backup_env(path, content)

    if not dry_run and (changed or not existed):
        atomic_write(path, new_content)
        _chmod_600(path)
    elif existed and not dry_run:
        _chmod_600(path)

    return EnvUpdateResult(
        path=path,
        created=not existed,
        changed=changed,
        backup_path=backup_path,
        updated_keys=tuple(updated_keys),
        added_keys=tuple(added_keys),
        dry_run=dry_run,
    )
