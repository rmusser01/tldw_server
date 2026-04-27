from __future__ import annotations

import base64
import os
import secrets
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from cryptography.fernet import Fernet

from tldw_Server_API.app.core.AuthNZ.api_key_crypto import parse_api_key
from tldw_Server_API.app.core.AuthNZ.username_utils import normalize_admin_username
from tldw_Server_API.cli.wizard.utils import env as env_utils


@dataclass(frozen=True)
class SetupProfile:
    name: str
    auth_mode: str
    docker: bool
    includes_webui: bool
    includes_postgres: bool
    default_base_url: str = "http://127.0.0.1:8000"
    default_webui_url: str | None = None


_PROFILES: dict[str, SetupProfile] = {
    "docker-single-webui": SetupProfile(
        name="docker-single-webui",
        auth_mode="single_user",
        docker=True,
        includes_webui=True,
        includes_postgres=False,
        default_webui_url="http://127.0.0.1:8080",
    ),
    "docker-multi-postgres": SetupProfile(
        name="docker-multi-postgres",
        auth_mode="multi_user",
        docker=True,
        includes_webui=False,
        includes_postgres=True,
    ),
    "local-single": SetupProfile(
        name="local-single",
        auth_mode="single_user",
        docker=False,
        includes_webui=False,
        includes_postgres=False,
    ),
}


def normalize_profile(value: str | None) -> SetupProfile:
    name = (value or "local-single").strip().lower().replace("_", "-")
    aliases = {
        "docker-single": "docker-single-webui",
        "docker-webui": "docker-single-webui",
        "docker-multi": "docker-multi-postgres",
        "local": "local-single",
    }
    name = aliases.get(name, name)
    try:
        return _PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(_PROFILES))
        raise ValueError(f"Unsupported setup profile '{value}'. Use one of: {choices}") from exc


def resolve_repo_root(start_dir: Path | None = None) -> Path | None:
    current = (start_dir or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "tldw_Server_API").is_dir():
            return candidate
    return None


def resolve_env_path(
    *,
    profile: SetupProfile,
    start_dir: Path | None = None,
    explicit_env_file: Path | None = None,
) -> Path:
    del profile  # Reserved for later profile-specific env placement.
    if explicit_env_file is not None:
        return explicit_env_file.expanduser().resolve()
    repo_root = resolve_repo_root(start_dir)
    if repo_root is not None:
        return repo_root / "tldw_Server_API" / "Config_Files" / ".env"
    return (start_dir or Path.cwd()).resolve() / ".env"


def _secret_token() -> str:
    return secrets.token_urlsafe(32)


def _postgres_password() -> str:
    return secrets.token_urlsafe(32)


def _fernet_key() -> str:
    return Fernet.generate_key().decode("ascii")


def _is_valid_fernet_key(value: str) -> bool:
    try:
        Fernet(value.encode("ascii"))
    except (TypeError, ValueError):
        return False
    return True


def _is_valid_single_user_api_key(value: str) -> bool:
    return parse_api_key(value) is not None


def _byok_key() -> str:
    return base64.urlsafe_b64encode(os.urandom(32)).decode("ascii")


def _is_placeholder(value: str | None) -> bool:
    if value is None:
        return True
    normalized = value.strip().lower()
    return (
        normalized == ""
        or normalized in {"change-me", "changeme", "default", "test-key"}
        or normalized.startswith("change_me")
        or normalized.startswith("replace-with-")
    )


def _existing_or_generated(existing_env: Mapping[str, str], key: str, generator: Callable[[], str]) -> str:
    value = existing_env.get(key)
    if not _is_placeholder(value):
        if key == "SINGLE_USER_API_KEY" and not _is_valid_single_user_api_key(value):
            return generator()
        if key == "SESSION_ENCRYPTION_KEY" and not _is_valid_fernet_key(value):
            return generator()
        return value
    return generator()


def profile_uses_structured_postgres(profile: SetupProfile) -> bool:
    return profile.docker and profile.includes_postgres


def build_postgres_database_url(
    *,
    user: str,
    password: str,
    db: str,
    host: str = "postgres",
    port: str = "5432",
) -> str:
    return f"postgresql://{quote(user, safe='')}:{quote(password, safe='')}" f"@{host}:{port}/{quote(db, safe='')}"


def build_profile_env(
    *,
    profile: SetupProfile,
    existing_env: Mapping[str, str],
    admin_username: str | None = None,
    admin_password: str | None = None,
    admin_email: str | None = None,
) -> dict[str, str]:
    values: dict[str, str] = {"AUTH_MODE": profile.auth_mode}
    if profile.auth_mode == "single_user":
        values["DATABASE_URL"] = existing_env.get("DATABASE_URL") or "sqlite:///./Databases/users.db"
        values["SINGLE_USER_API_KEY"] = _existing_or_generated(
            existing_env,
            "SINGLE_USER_API_KEY",
            env_utils.generate_single_user_api_key,
        )
        return values

    postgres_user = existing_env.get("POSTGRES_USER") or "tldw_user"
    postgres_db = existing_env.get("POSTGRES_DB") or "tldw_users"
    postgres_password = _existing_or_generated(existing_env, "POSTGRES_PASSWORD", _postgres_password)
    values["POSTGRES_USER"] = postgres_user
    values["POSTGRES_DB"] = postgres_db
    values["POSTGRES_PASSWORD"] = postgres_password
    if not profile_uses_structured_postgres(profile):
        database_url = build_postgres_database_url(user=postgres_user, password=postgres_password, db=postgres_db)
        values["DATABASE_URL"] = database_url
        values["JOBS_DB_URL"] = database_url
    values["JWT_SECRET_KEY"] = _existing_or_generated(existing_env, "JWT_SECRET_KEY", _secret_token)
    values["SESSION_ENCRYPTION_KEY"] = _existing_or_generated(existing_env, "SESSION_ENCRYPTION_KEY", _fernet_key)
    values["MCP_JWT_SECRET"] = _existing_or_generated(existing_env, "MCP_JWT_SECRET", _secret_token)
    values["MCP_API_KEY_SALT"] = _existing_or_generated(existing_env, "MCP_API_KEY_SALT", _secret_token)
    values["BYOK_ENCRYPTION_KEY"] = _existing_or_generated(existing_env, "BYOK_ENCRYPTION_KEY", _byok_key)
    if admin_username:
        values["ADMIN_USERNAME"] = normalize_admin_username(admin_username)
    if admin_password:
        values["ADMIN_PASSWORD"] = admin_password
    if admin_email:
        values["ADMIN_EMAIL"] = admin_email
    return values
