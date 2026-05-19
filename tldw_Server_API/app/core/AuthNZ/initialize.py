#!/usr/bin/env python3
# initialize.py
# Description: Initialize AuthNZ module for first-time setup
#
# This script sets up the AuthNZ module including:
# - Database creation and migrations
# - Initial admin user creation (multi-user mode)
# - Encryption key generation
# - Configuration validation
#

import argparse
import asyncio
import os
import secrets
import sys
from collections.abc import Iterable
from getpass import getpass
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

from dotenv import dotenv_values, load_dotenv
from loguru import logger

TEST_SETUP_API_KEY = "THIS-IS-NOT-A-SECURE-KEY-123-CHANGE-ME"
SINGLE_USER_API_KEY_PLACEHOLDERS = {
    "CHANGE_ME_TO_SECURE_API_KEY",
    "default-secret-key-for-single-user",
    "change-me-in-production",
    "CHANGE-ME-to-a-secure-key-at-least-16-chars",
}

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

import contextlib

from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager, get_api_key_manager
from tldw_Server_API.app.core.AuthNZ.database import DatabasePool, get_db_pool
from tldw_Server_API.app.core.AuthNZ.migrations import check_migration_status, ensure_authnz_tables
from tldw_Server_API.app.core.AuthNZ.monitoring import get_authnz_monitor
from tldw_Server_API.app.core.AuthNZ.password_service import PasswordService
from tldw_Server_API.app.core.AuthNZ.scheduler import start_authnz_scheduler
from tldw_Server_API.app.core.AuthNZ.settings import get_settings, reset_settings
from tldw_Server_API.app.core.AuthNZ.username_utils import normalize_admin_username
from tldw_Server_API.app.core.DB_Management.Users_DB import ensure_user_directories, get_users_db
from tldw_Server_API.app.core.testing import is_test_mode

_AUTHNZ_INIT_NONCRITICAL_EXCEPTIONS = (
    asyncio.CancelledError,
    asyncio.TimeoutError,
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    ImportError,
    IndexError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    UnicodeDecodeError,
)

#######################################################################################################################
#
# Initialization Functions
#


def _sanitize_db_url(url: Optional[str]) -> str:
    """Strip credentials from DB URL for safe diagnostics."""
    if not url:
        return "unknown"

    try:
        parsed = urlsplit(url)

        # For file-based URLs (e.g., sqlite) that lack a netloc, return as-is.
        if not parsed.netloc:
            return url

        netloc_no_auth = parsed.netloc.split("@", 1)[-1]
        host = parsed.hostname or netloc_no_auth
        port = f":{parsed.port}" if parsed.port else ""
        # urlsplit strips IPv6 brackets; restore them when reconstructing
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"

        sanitized_netloc = f"{host}{port}"
        prefix = f"{parsed.scheme}://" if parsed.scheme else ""
        sanitized_url = f"{prefix}{sanitized_netloc}{parsed.path}"

        if parsed.query:
            sanitized_url += f"?{parsed.query}"
        if parsed.fragment:
            sanitized_url += f"#{parsed.fragment}"

        return sanitized_url or url
    except _AUTHNZ_INIT_NONCRITICAL_EXCEPTIONS:
        # Fall back to the original string if parsing fails; avoid raising during diagnostics.
        return url


def _resolve_sqlite_db_path(db_url: str) -> Optional[Path]:
    """Resolve a sqlite/file DATABASE_URL into a filesystem path, if applicable."""
    try:
        parsed = urlsplit(db_url)
        scheme = (parsed.scheme or "").lower().split("+", 1)[0]
    except _AUTHNZ_INIT_NONCRITICAL_EXCEPTIONS:
        scheme = ""

    if scheme not in {"sqlite", "file", ""}:
        return None

    _, _, fs_path = DatabasePool._resolve_sqlite_paths(db_url)
    if not fs_path or fs_path == ":memory:":
        return None

    return Path(fs_path)

def _normalize_env_value(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "nil"}:
        return None
    return text


def _resolve_env_locations() -> tuple[list[Path], list[Path], Path]:
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    cfg_dir = project_root / "Config_Files"
    env_candidates = [
        cfg_dir / ".env",
        cfg_dir / ".ENV",
    ]
    template_candidates = [cfg_dir / ".env.example"]
    return env_candidates, template_candidates, cfg_dir


def _create_env_from_template(target: Path, templates: Iterable[Path]) -> bool:
    for template in templates:
        if template.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(template.read_text(encoding="utf-8"), encoding="utf-8")
            return True
    return False


def _read_env_values(env_path: Optional[Path]) -> dict[str, str]:
    if not env_path or not env_path.exists():
        return {}
    try:
        raw = dotenv_values(str(env_path))
    except _AUTHNZ_INIT_NONCRITICAL_EXCEPTIONS:
        return {}
    return {str(k): str(v) for k, v in raw.items() if k}


def _effective_env_value(key: str, env_values: dict[str, str]) -> Optional[str]:
    return _normalize_env_value(os.getenv(key) or env_values.get(key))


def _detect_env_issues(auth_mode: str, env_values: dict[str, str]) -> tuple[set[str], list[str]]:
    missing_keys: set[str] = set()
    issues: list[str] = []

    mode = (auth_mode or "single_user").strip().lower()
    if mode not in {"single_user", "multi_user"}:
        issues.append(f"AUTH_MODE must be 'single_user' or 'multi_user' (found: {auth_mode})")
        return missing_keys, issues

    jwt_placeholders = {
        "CHANGE_ME_TO_SECURE_RANDOM_KEY_MIN_32_CHARS",
    }

    if mode == "single_user":
        single_key = (
            _effective_env_value("SINGLE_USER_API_KEY", env_values)
            or _effective_env_value("API_KEY", env_values)
        )
        if not single_key:
            missing_keys.add("SINGLE_USER_API_KEY")
            issues.append("SINGLE_USER_API_KEY is required for single-user mode")
        elif single_key in SINGLE_USER_API_KEY_PLACEHOLDERS:
            missing_keys.add("SINGLE_USER_API_KEY")
            issues.append("SINGLE_USER_API_KEY still uses the default placeholder")
        elif len(single_key) < 16:
            missing_keys.add("SINGLE_USER_API_KEY")
            issues.append("SINGLE_USER_API_KEY must be at least 16 characters")

    if mode == "multi_user":
        jwt_key = _effective_env_value("JWT_SECRET_KEY", env_values)
        if not jwt_key:
            missing_keys.add("JWT_SECRET_KEY")
            issues.append("JWT_SECRET_KEY must be set for multi-user mode")
        elif jwt_key in jwt_placeholders:
            missing_keys.add("JWT_SECRET_KEY")
            issues.append("JWT_SECRET_KEY still uses the default placeholder")
        elif len(jwt_key) < 32:
            missing_keys.add("JWT_SECRET_KEY")
            issues.append("JWT_SECRET_KEY must be at least 32 characters")

    # MCP Unified requires explicit secrets in production; generate during setup.
    mcp_jwt = _effective_env_value("MCP_JWT_SECRET", env_values)
    if not mcp_jwt:
        missing_keys.add("MCP_JWT_SECRET")
        issues.append("MCP_JWT_SECRET must be set for MCP security hardening")
    elif len(mcp_jwt) < 32:
        missing_keys.add("MCP_JWT_SECRET")
        issues.append("MCP_JWT_SECRET must be at least 32 characters")

    mcp_salt = _effective_env_value("MCP_API_KEY_SALT", env_values)
    if not mcp_salt:
        missing_keys.add("MCP_API_KEY_SALT")
        issues.append("MCP_API_KEY_SALT must be set for MCP security hardening")
    elif len(mcp_salt) < 32:
        missing_keys.add("MCP_API_KEY_SALT")
        issues.append("MCP_API_KEY_SALT must be at least 32 characters")

    return missing_keys, issues


def _write_env_values(env_path: Path, values: dict[str, str]) -> None:
    env_path.parent.mkdir(parents=True, exist_ok=True)
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []

    updated_keys: set[str] = set()
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        candidate = stripped
        if candidate.startswith("export "):
            candidate = candidate[7:]
        key, sep, _value = candidate.partition("=")
        if not sep:
            continue
        key = key.strip()
        if key in values:
            lines[idx] = f"{key}={values[key]}"
            updated_keys.add(key)

    if updated_keys != set(values.keys()):
        if lines and lines[-1].strip():
            lines.append("")
        lines.append("# Added by AuthNZ initialize")
        for key, value in values.items():
            if key not in updated_keys:
                lines.append(f"{key}={value}")

    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _select_keys_to_write(
    generated: dict[str, str],
    required_keys: set[str],
) -> dict[str, str]:
    keys_to_write: dict[str, str] = {}
    for key in required_keys:
        value = generated.get(key)
        if value:
            keys_to_write[key] = value
    return keys_to_write


def _prompt_yes_no(prompt: str, default_yes: bool, non_interactive: bool) -> bool:
    if non_interactive:
        return default_yes
    suffix = "Y/n" if default_yes else "y/N"
    response = input(f"{prompt} ({suffix}): ").strip().lower()
    if not response:
        return default_yes
    return response in {"y", "yes"}


def _ensure_env_file() -> Path:
    env_candidates, template_candidates, _ = _resolve_env_locations()
    selected_env = next((p for p in env_candidates if p.exists()), None) or env_candidates[0]
    if not selected_env.exists():
        created = _create_env_from_template(selected_env, template_candidates)
        if not created:
            selected_env.parent.mkdir(parents=True, exist_ok=True)
            selected_env.write_text("", encoding="utf-8")
    return selected_env


def _apply_test_setup_env() -> Path:
    env_path = _ensure_env_file()
    values = {
        "AUTH_MODE": "single_user",
        "SINGLE_USER_API_KEY": TEST_SETUP_API_KEY,
        "MCP_JWT_SECRET": secrets.token_urlsafe(32),
        "MCP_API_KEY_SALT": secrets.token_urlsafe(32),
    }
    _write_env_values(env_path, values)
    load_dotenv(dotenv_path=str(env_path), override=True)
    reset_settings()
    return env_path



def print_banner():
    """Print initialization banner"""
    print("\n" + "=" * 60)
    print("       AuthNZ Module Initialization")
    print("=" * 60)
    print()



def check_environment():
    """Check and validate environment configuration

    Preference order for .env resolution:
      1) tldw_Server_API/Config_Files/.env (project Config_Files directory)
    The first found file is loaded into process env (non-overriding).
    """
    print("📋 Checking environment configuration...")

    env_candidates, template_candidates, _ = _resolve_env_locations()
    selected_env: Optional[Path] = next((p for p in env_candidates if p.exists()), None)

    if selected_env is None:
        selected_env = env_candidates[0]
        print("❌ No .env file found in Config_Files!")
        print(f"   Creating at: {selected_env}")
        created = _create_env_from_template(selected_env, template_candidates)
        if created:
            print("✅ Created .env file from template")
        else:
            selected_env.parent.mkdir(parents=True, exist_ok=True)
            selected_env.write_text("", encoding="utf-8")
            print("⚠️  Template file not found; created empty .env")

    # Load the chosen .env without overriding any already-set environment vars
    try:
        load_dotenv(dotenv_path=str(selected_env), override=False)
        print(f"✅ Loaded environment variables from: {selected_env}")
    except _AUTHNZ_INIT_NONCRITICAL_EXCEPTIONS as e:
        print(f"⚠️  Failed to load .env at {selected_env}: {e}")

    env_values = _read_env_values(selected_env)
    auth_mode = _effective_env_value("AUTH_MODE", env_values) or "single_user"
    missing_keys, issues = _detect_env_issues(auth_mode, env_values)

    if issues:
        print("\n⚠️  Configuration issues found:")
        for issue in issues:
            print(f"   - {issue}")

    if missing_keys:
        return {
            "ok": False,
            "env_path": selected_env,
            "missing_keys": missing_keys,
            "issues": issues,
            "auth_mode": auth_mode,
        }

    try:
        settings = get_settings()
    except _AUTHNZ_INIT_NONCRITICAL_EXCEPTIONS as exc:
        print(f"\n❌ Configuration validation failed: {exc}")
        return {
            "ok": False,
            "env_path": selected_env,
            "missing_keys": set(),
            "issues": [str(exc)],
            "auth_mode": auth_mode,
        }

    print("✅ Environment configuration valid")
    print(f"   Mode: {settings.AUTH_MODE}")
    db_url_safe = _sanitize_db_url(settings.DATABASE_URL)
    print(f"   Database: {db_url_safe}")

    return {
        "ok": True,
        "env_path": selected_env,
        "missing_keys": set(),
        "issues": [],
        "auth_mode": settings.AUTH_MODE,
    }

def generate_secure_keys(requested_keys: Optional[Iterable[str]] = None):
    """Generate secure keys for configuration"""
    print("\n🔑 Generating secure keys...")

    from tldw_Server_API.app.core.AuthNZ.api_key_crypto import (
        format_api_key,
        generate_api_key_id,
        generate_api_key_secret,
    )

    requested = set(requested_keys) if requested_keys else None
    keys: dict[str, str] = {}

    if requested is None or "JWT_SECRET_KEY" in requested:
        keys['JWT_SECRET_KEY'] = secrets.token_urlsafe(32)
    if requested is None or "SINGLE_USER_API_KEY" in requested:
        keys['SINGLE_USER_API_KEY'] = format_api_key(
            generate_api_key_id(),
            generate_api_key_secret(),
        )
    if requested is None or "API_KEY_PEPPER" in requested:
        keys['API_KEY_PEPPER'] = secrets.token_hex(32)
    if requested is None or "SESSION_ENCRYPTION_KEY" in requested:
        # Generate Fernet key for session encryption
        from cryptography.fernet import Fernet

        keys['SESSION_ENCRYPTION_KEY'] = Fernet.generate_key().decode()
    if requested is None or "MCP_JWT_SECRET" in requested:
        keys["MCP_JWT_SECRET"] = secrets.token_urlsafe(32)
    if requested is None or "MCP_API_KEY_SALT" in requested:
        keys["MCP_API_KEY_SALT"] = secrets.token_urlsafe(32)

    print("\n📝 Generated keys (save these in your .env file):")
    print("-" * 50)
    for key, value in keys.items():
        print(f"{key}={value}")
    print("-" * 50)

    return keys

async def setup_database():
    """Setup database and run migrations"""
    print("\n🗄️  Setting up database...")

    settings = get_settings()

    # Extract database path
    db_url = settings.DATABASE_URL
    db_path = _resolve_sqlite_db_path(db_url)
    if db_path is not None:

        # Ensure directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"   Database path: {db_path}")

        # Check migration status
        status = check_migration_status(db_path)
        print(f"   Current version: {status['current_version']}")
        print(f"   Latest version: {status['latest_version']}")

        if not status['is_up_to_date']:
            print(f"   Pending migrations: {len(status['pending_migrations'])}")

            # Apply migrations
            ensure_authnz_tables(db_path)
            print("✅ Database migrations applied")
        else:
            print("✅ Database is up to date")
    else:
        # Basic Postgres bootstrap: ensure required tables exist
        print("⚙️  Non-SQLite database detected - attempting basic schema bootstrap (users, sessions, api_keys, RBAC)...")
        try:
            # Ensure connection pool and users table
            users_db = await get_users_db()
            await users_db.initialize()

            from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
            from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import (
                ensure_api_keys_tables_pg,
                ensure_authnz_core_tables_pg,
                ensure_billing_tables_pg,
                ensure_byok_oauth_state_pg,
                ensure_generated_files_table_pg,
                ensure_org_provider_secrets_pg,
                ensure_usage_tables_pg,
                ensure_user_provider_secrets_pg,
                ensure_virtual_key_counters_pg,
            )

            pool = await get_db_pool()

            # Ensure core AuthNZ tables (audit_logs, sessions, registration_codes, RBAC, orgs/teams)
            await ensure_authnz_core_tables_pg(pool)

            # Ensure billing tables used by webhook/invoice/audit paths.
            ok_billing_tables = await ensure_billing_tables_pg(pool)
            if not ok_billing_tables:
                raise RuntimeError("Failed to ensure Postgres billing tables")

            # Seed baseline RBAC roles and permissions (centralized helper to avoid drift)
            from tldw_Server_API.app.core.AuthNZ.rbac_seed import ensure_baseline_rbac_seed
            async with pool.transaction() as conn:
                await ensure_baseline_rbac_seed(
                    conn,
                    include_mcp_permissions=False,
                    is_postgres=True,
                )

            # Ensure API key tables after org/team tables exist
            ok_api_keys = await ensure_api_keys_tables_pg(pool)
            if not ok_api_keys:
                raise RuntimeError("Failed to ensure Postgres api_keys tables")
            # Ensure BYOK secrets tables
            await ensure_user_provider_secrets_pg(pool)
            await ensure_org_provider_secrets_pg(pool)
            await ensure_byok_oauth_state_pg(pool)

            # Ensure usage/LLM usage tables and virtual-key counters for Postgres.
            # The SQLite path is covered by AuthNZ migrations; on Postgres we rely
            # on these additive helpers instead of inline DDL here.
            # Capture a sanitized view of the DB URL for diagnostics without leaking credentials.
            db_url_safe = "unknown"
            try:
                raw_url = get_settings().DATABASE_URL
                db_url_safe = _sanitize_db_url(raw_url)
            except _AUTHNZ_INIT_NONCRITICAL_EXCEPTIONS as settings_err:
                # Settings resolution failures during bootstrap are non-fatal here; keep "unknown".
                logger.debug(
                    f"DB URL extraction for diagnostics failed: {settings_err}"
                )
            try:
                await ensure_usage_tables_pg(pool)
            except _AUTHNZ_INIT_NONCRITICAL_EXCEPTIONS as usage_err:
                logger.warning(
                    "AuthNZ initialize: ensure_usage_tables_pg failed for Postgres backend "
                    f"(db={db_url_safe}); usage tables may be missing: {usage_err}"
                )
            try:
                await ensure_generated_files_table_pg(pool)
            except _AUTHNZ_INIT_NONCRITICAL_EXCEPTIONS as files_err:
                logger.warning(
                    "AuthNZ initialize: ensure_generated_files_table_pg failed for Postgres backend "
                    f"(db={db_url_safe}); generated_files table may be missing: {files_err}"
                )
            try:
                await ensure_virtual_key_counters_pg(pool)
            except _AUTHNZ_INIT_NONCRITICAL_EXCEPTIONS as vk_err:
                logger.warning(
                    "AuthNZ initialize: ensure_virtual_key_counters_pg failed for Postgres backend "
                    f"(db={db_url_safe}); virtual-key counters tables may be missing: {vk_err}"
                )

            print(
                "✅ Basic schema ensured for Postgres (users, api keys, sessions, "
                "registration_codes, RBAC, orgs/teams, usage tables)"
            )
        except _AUTHNZ_INIT_NONCRITICAL_EXCEPTIONS as e:
            print(f"❌ Failed to bootstrap Postgres schema: {e}")
            logger.exception("Postgres schema bootstrap error")
            return False

    return True


#######################################################################################################################
#
# Async startup helpers (app/tests)

_SCHEMA_ENSURED_KEYS: set[str] = set()
_SCHEMA_ENSURE_LOCK = asyncio.Lock()


async def ensure_authnz_schema_ready_once() -> None:
    """Ensure AuthNZ schema is present for SQLite backends exactly once per process.

    - Obtains the shared DB pool via get_db_pool.
    - If backend is SQLite, calls ensure_authnz_tables in a thread (safe to call repeatedly).
    - Guarded by an in‑memory flag + lock to avoid repeated work across startup and tests.
    """
    global _SCHEMA_ENSURED_KEYS
    async with _SCHEMA_ENSURE_LOCK:
        try:
            pool = await get_db_pool()
        except _AUTHNZ_INIT_NONCRITICAL_EXCEPTIONS as e:
            with contextlib.suppress(_AUTHNZ_INIT_NONCRITICAL_EXCEPTIONS):
                logger.debug(f"AuthNZ schema ensure: failed to acquire DB pool; skipping: {e}")
            return

        try:
            # If asyncpg pool exists, we're on Postgres; no SQLite migration ensure needed.
            if getattr(pool, 'pool', None):
                return

            db_fs_path = getattr(pool, '_sqlite_fs_path', None) or getattr(pool, 'db_path', None)
            key = str(db_fs_path or '')
            if key in _SCHEMA_ENSURED_KEYS:
                return
            if db_fs_path and str(db_fs_path) != ':memory:':
                try:
                    await asyncio.to_thread(ensure_authnz_tables, Path(str(db_fs_path)))
                    logger.info(f"AuthNZ Startup: ensured SQLite schema at {db_fs_path}")
                except _AUTHNZ_INIT_NONCRITICAL_EXCEPTIONS as mig_err:
                    logger.debug(f"AuthNZ Startup: ensure_authnz_tables skipped/failed: {mig_err}")
            _SCHEMA_ENSURED_KEYS.add(key)
        except _AUTHNZ_INIT_NONCRITICAL_EXCEPTIONS as e:
            # Do not raise during startup; log for diagnostics
            logger.debug(f"AuthNZ Startup: schema ensure encountered error: {e}")
            with contextlib.suppress(_AUTHNZ_INIT_NONCRITICAL_EXCEPTIONS):
                _SCHEMA_ENSURED_KEYS.add(str(getattr(pool, '_sqlite_fs_path', '') or getattr(pool, 'db_path', '') or ''))


async def ensure_single_user_rbac_seed_if_needed() -> None:
    """Ensure baseline RBAC seed exists in single-user mode for any backend.

    Idempotent: inserts roles/permissions only if missing. Intended to backstop
    environments where migrations or bootstrap did not seed RBAC yet.
    """
    settings = get_settings()
    # In test suites we may switch DATABASE_URL or AUTH_MODE between runs (e.g., SQLite → Postgres).
    # Detect and realign settings/pools so the seed targets the active backend.
    try:
        effective_db_url = os.getenv("DATABASE_URL")
    except _AUTHNZ_INIT_NONCRITICAL_EXCEPTIONS:
        effective_db_url = None
    try:
        effective_auth_mode = os.getenv("AUTH_MODE")
    except _AUTHNZ_INIT_NONCRITICAL_EXCEPTIONS:
        effective_auth_mode = None
    try:
        effective_single_user_api_key = os.getenv("SINGLE_USER_API_KEY") or os.getenv("API_KEY")
    except _AUTHNZ_INIT_NONCRITICAL_EXCEPTIONS:
        effective_single_user_api_key = None

    need_reset = False
    if effective_db_url and effective_db_url != settings.DATABASE_URL:
        need_reset = True
    if effective_auth_mode and effective_auth_mode.lower() != settings.AUTH_MODE:
        need_reset = True
    if effective_single_user_api_key and effective_single_user_api_key != settings.SINGLE_USER_API_KEY:
        need_reset = True

    test_mode = is_test_mode()
    if test_mode:
        need_reset = True

    if need_reset:
        try:
            from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool

            await reset_db_pool()
        except _AUTHNZ_INIT_NONCRITICAL_EXCEPTIONS as reset_err:
            logger.debug(f"ensure_single_user_rbac_seed_if_needed: reset_db_pool skipped: {reset_err}")
        reset_settings()
        settings = get_settings()

    if settings.AUTH_MODE != "single_user":
        # In multi-user modes we rely on the normal RBAC/bootstrap paths for
        # roles and permissions. Forcing the single-user seed (including an
        # explicit ``id = SINGLE_USER_FIXED_ID`` row in ``users``) would
        # interfere with Postgres SERIAL/identity sequences and tests that
        # exercise multi-user registration flows. Only single-user profile
        # (AUTH_MODE=single_user) should reach the seed logic below.
        logger.debug(
            "ensure_single_user_rbac_seed_if_needed: skipping seed; AUTH_MODE={}",
            settings.AUTH_MODE,
        )
        return

    # Best-effort schema backstops:
    # - SQLite: ensure migrations have been applied (file-backed DBs).
    # - Postgres: ensure core AuthNZ tables (including RBAC) exist via PG extras.
    try:
        await ensure_authnz_schema_ready_once()
    except _AUTHNZ_INIT_NONCRITICAL_EXCEPTIONS as schema_err:
        logger.debug(
            "ensure_single_user_rbac_seed_if_needed: schema ensure skipped/failed: {}",
            schema_err,
        )
    try:
        # Acquire a connection via pool/transaction abstraction
        from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
        pool = await get_db_pool()
        is_postgres_backend = bool(getattr(pool, "pool", None))

        # Postgres-only: ensure core AuthNZ tables (including RBAC) via bootstrap backstop.
        if is_postgres_backend:
            try:
                from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import (
                    ensure_authnz_core_tables_pg,
                )

                await ensure_authnz_core_tables_pg(pool)
            except _AUTHNZ_INIT_NONCRITICAL_EXCEPTIONS as ensure_err:
                logger.debug(
                    "ensure_single_user_rbac_seed_if_needed: PG ensure_authnz_core_tables_pg failed/skipped: {}",
                    ensure_err,
                )

        from tldw_Server_API.app.core.AuthNZ.rbac_seed import (
            ensure_baseline_rbac_seed,
            ensure_sqlite_rbac_tables,
        )

        # SQLite in-memory DBs cannot run file-based migrations; create minimal
        # RBAC tables as a backstop so baseline seed can succeed.
        if not is_postgres_backend:
            sqlite_fs_path = str(
                getattr(pool, "_sqlite_fs_path", None)
                or getattr(pool, "db_path", None)
                or ""
            )
            sqlite_is_memory = sqlite_fs_path.strip() == ":memory:"
            try:
                db_path_str = str(getattr(pool, "db_path", "") or "")
                if "mode=memory" in db_path_str.lower():
                    sqlite_is_memory = True
            except _AUTHNZ_INIT_NONCRITICAL_EXCEPTIONS:
                pass

            if sqlite_is_memory:
                async with pool.transaction() as conn:  # type: ignore[attr-defined]
                    try:
                        await ensure_sqlite_rbac_tables(conn)
                    except _AUTHNZ_INIT_NONCRITICAL_EXCEPTIONS as table_err:
                        logger.debug(
                            "SQLite in-memory RBAC table creation skipped (tables may already exist): {}",
                            table_err,
                        )
                    with contextlib.suppress(_AUTHNZ_INIT_NONCRITICAL_EXCEPTIONS):
                        await conn.commit()  # type: ignore[attr-defined]

        async with pool.transaction() as conn:
            await ensure_baseline_rbac_seed(
                conn,
                include_mcp_permissions=True,
                is_postgres=is_postgres_backend,
            )
            if not is_postgres_backend:
                with contextlib.suppress(_AUTHNZ_INIT_NONCRITICAL_EXCEPTIONS):
                    await conn.commit()  # type: ignore[attr-defined]

        single_user_id = settings.SINGLE_USER_FIXED_ID
        from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo

        users_repo = AuthnzUsersRepo(db_pool=pool)
        await users_repo.ensure_single_user_admin_user(
            user_id=single_user_id,
            username="single_user",
            email="single_user@example.local",
            # Single-user auth uses the API key; this fixed account has no password login.
            password_hash="",  # nosec B106
        )

        # Ensure single-user role assignment + primary/test API keys.
        try:
            from tldw_Server_API.app.core.AuthNZ.repos.api_keys_repo import AuthnzApiKeysRepo

            await users_repo.assign_role_if_missing(user_id=single_user_id, role_name="admin")

            api_repo = AuthnzApiKeysRepo(pool)
            api_manager: APIKeyManager | None = None

            primary_api_key = (settings.SINGLE_USER_API_KEY or "").strip()
            if primary_api_key and primary_api_key != "CHANGE_ME_TO_SECURE_API_KEY":
                api_manager = APIKeyManager(db_pool=pool)
                key_hash = api_manager.hash_api_key(primary_api_key)
                key_prefix = (
                    (primary_api_key[:10] + "...")
                    if len(primary_api_key) > 10
                    else primary_api_key
                )
                key_identifier = None
                try:
                    from tldw_Server_API.app.core.AuthNZ.api_key_crypto import parse_api_key

                    parsed = parse_api_key(primary_api_key)
                    if parsed:
                        key_identifier, _secret = parsed
                except _AUTHNZ_INIT_NONCRITICAL_EXCEPTIONS as exc:
                    logger.opt(exception=True).debug(
                        "Failed to parse SINGLE_USER_API_KEY for seed key identifier extraction: {}",
                        exc,
                    )
                await api_repo.upsert_primary_key(
                    user_id=single_user_id,
                    key_hash=key_hash,
                    key_identifier=key_identifier,
                    key_prefix=key_prefix,
                    name="single-user primary key",
                    description="Primary API key for single-user profile",
                    scope="admin",
                    is_virtual=False,
                )

            test_api_key = os.getenv("SINGLE_USER_TEST_API_KEY")
            if test_mode and test_api_key:
                api_manager = api_manager or APIKeyManager(db_pool=pool)
                test_key_hash = api_manager.hash_api_key(test_api_key)
                test_key_prefix = (
                    (test_api_key[:10] + "...")
                    if len(test_api_key) > 10
                    else test_api_key
                )
                test_key_identifier = None
                try:
                    from tldw_Server_API.app.core.AuthNZ.api_key_crypto import parse_api_key

                    parsed = parse_api_key(test_api_key)
                    if parsed:
                        test_key_identifier, _secret = parsed
                except _AUTHNZ_INIT_NONCRITICAL_EXCEPTIONS as exc:
                    logger.opt(exception=True).debug(
                        "Failed to parse SINGLE_USER_TEST_API_KEY for seed key identifier extraction: {}",
                        exc,
                    )
                await api_repo.upsert_primary_key(
                    user_id=single_user_id,
                    key_hash=test_key_hash,
                    key_identifier=test_key_identifier,
                    key_prefix=test_key_prefix,
                    name="single-user test key",
                    description="Deterministic API key for test automation",
                    scope="admin",
                    is_virtual=False,
                )
        except _AUTHNZ_INIT_NONCRITICAL_EXCEPTIONS as role_assign_err:
            # Log at warning level with context so repeated failures surface operationally
            logger.warning(
                "Single-user admin role assignment skipped in ensure_single_user_rbac_seed_if_needed "
                "(AUTH_MODE={}, db_url={}): {}",
                settings.AUTH_MODE,
                _sanitize_db_url(settings.DATABASE_URL),
                role_assign_err,
            )
    except _AUTHNZ_INIT_NONCRITICAL_EXCEPTIONS as e:
        # Non-fatal but important for observability: surface failures at warning level
        logger.opt(exception=True).warning(
            "Single-user RBAC seed ensure skipped or failed in ensure_single_user_rbac_seed_if_needed "
            "(AUTH_MODE={}, db_url={}): {}",
            settings.AUTH_MODE,
            _sanitize_db_url(settings.DATABASE_URL),
            e,
        )


async def _collect_single_user_invariant_errors(
    pool: DatabasePool,
    *,
    expected_user_id: int,
    expected_key_hash: Optional[str],
    allowed_non_virtual_key_hashes: Optional[set[str]] = None,
    check_keys: bool,
) -> list[str]:
    """Return a list of invariant violations for single-user bootstrap."""
    from tldw_Server_API.app.core.AuthNZ.repos.api_keys_repo import AuthnzApiKeysRepo
    from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo

    errors: list[str] = []

    users_repo = AuthnzUsersRepo(db_pool=pool)
    api_keys_repo = AuthnzApiKeysRepo(db_pool=pool)

    try:
        active_rows, _active_total = await users_repo.list_users(
            offset=0,
            limit=10_000,
            is_active=True,
        )
        active_ids = sorted(
            {
                int(row["id"])
                for row in active_rows
                if row.get("id") is not None
            }
        )
        if expected_user_id not in active_ids:
            errors.append(
                f"Single-user admin id={expected_user_id} is missing or inactive."
            )
        extra_active = [uid for uid in active_ids if uid != expected_user_id]
        if extra_active:
            errors.append(
                "Multiple active users detected in single-user profile: "
                + ", ".join(str(uid) for uid in extra_active)
            )
    except _AUTHNZ_INIT_NONCRITICAL_EXCEPTIONS as exc:
        errors.append(f"Failed to verify active users: {exc}")

    try:
        admin_rows, _admin_total = await users_repo.list_users(
            offset=0,
            limit=10_000,
            role="admin",
            is_active=True,
        )
        admin_ids = sorted(
            {
                int(row["id"])
                for row in admin_rows
                if row.get("id") is not None
            }
        )
        extra_admins = [uid for uid in admin_ids if uid != expected_user_id]
        if extra_admins:
            errors.append(
                "Multiple admin users detected in single-user profile: "
                + ", ".join(str(uid) for uid in extra_admins)
            )
    except _AUTHNZ_INIT_NONCRITICAL_EXCEPTIONS as exc:
        errors.append(f"Failed to verify admin users: {exc}")

    if check_keys and expected_key_hash:
        try:
            key_rows = await api_keys_repo.list_user_keys(
                user_id=expected_user_id,
                include_revoked=False,
            )
            allowed_hashes: set[str] = {str(expected_key_hash)}
            if allowed_non_virtual_key_hashes:
                allowed_hashes.update(
                    str(item)
                    for item in allowed_non_virtual_key_hashes
                    if item
                )
            active_non_virtual_rows = [
                row
                for row in key_rows
                if not bool(row.get("is_virtual", False))
            ]
            key_ids = [
                int(row["id"])
                for row in active_non_virtual_rows
                if row.get("id") is not None
            ]
            if not key_ids:
                errors.append(
                    "No active non-virtual API key found for the single-user admin."
                )
                return errors

            primary_rows = [
                row
                for row in active_non_virtual_rows
                if str(row.get("key_hash") or "") == str(expected_key_hash)
            ]
            if len(primary_rows) != 1:
                errors.append(
                    "Active primary API key does not match SINGLE_USER_API_KEY."
                )

            unexpected_rows = [
                row
                for row in active_non_virtual_rows
                if str(row.get("key_hash") or "") not in allowed_hashes
            ]
            if unexpected_rows:
                unexpected_ids = [
                    str(int(row["id"]))
                    for row in unexpected_rows
                    if row.get("id") is not None
                ]
                if unexpected_ids:
                    errors.append(
                        "Unexpected active non-virtual API keys found for the single-user admin: "
                        + ", ".join(unexpected_ids)
                    )
                else:
                    errors.append(
                        "Unexpected active non-virtual API keys found for the single-user admin."
                    )
        except _AUTHNZ_INIT_NONCRITICAL_EXCEPTIONS as exc:
            errors.append(f"Failed to verify single-user API key invariants: {exc}")

    return errors

async def create_admin_user():
    """Create initial admin user for multi-user mode"""
    settings = get_settings()

    if settings.AUTH_MODE != "multi_user":
        print("\n📝 Single-user mode - skipping admin user creation")
        return True

    print("\n👤 Creating admin user...")

    # Get user input
    while True:
        raw_username = input("   Admin username (default: tldw_admin): ").strip() or "tldw_admin"
        try:
            username = normalize_admin_username(raw_username)
            break
        except ValueError as exc:
            print(f"   {exc}")

    while True:
        email = input("   Admin email: ").strip()
        if "@" in email and "." in email:
            break
        print("   Please enter a valid email address")

    while True:
        password = getpass("   Admin password (min 10 chars): ")
        if len(password) >= 10:
            confirm = getpass("   Confirm password: ")
            if password == confirm:
                break
            else:
                print("   Passwords don't match!")
        else:
            print("   Password must be at least 10 characters")

    try:
        # Hash password
        password_service = PasswordService()
        password_hash = password_service.hash_password(password)

        # Create user
        users_db = await get_users_db()
        admin_user = await users_db.create_user(
            username=username,
            email=email,
            password_hash=password_hash,
            role="admin",
            is_superuser=True
        )

        # Create initial API key for admin
        api_manager = await get_api_key_manager()
        api_key_result = await api_manager.create_api_key(
            user_id=admin_user['id'],
            name="Initial Admin API Key",
            description="Auto-generated during setup",
            scope="admin",
            expires_in_days=365
        )

        print("\n✅ Admin user created successfully!")
        print(f"   User ID: {admin_user['id']}")
        print(f"   Username: {admin_user['username']}")
        print("\n🔑 Admin API Key (save this - won't be shown again):")
        print(f"   {api_key_result['key']}")

        # Ensure user directories exist
        await ensure_user_directories(admin_user['id'])

        return True

    except _AUTHNZ_INIT_NONCRITICAL_EXCEPTIONS as e:
        print(f"❌ Failed to create admin user: {e}")
        return False


async def bootstrap_single_user_profile() -> bool:
    """
    Bootstrap single-user profile using normal AuthNZ flows.

    This helper is idempotent and ensures:
    - A single admin user exists with id = SINGLE_USER_FIXED_ID.
    - A primary API key exists for that user matching SINGLE_USER_API_KEY
      (hashed via the centralized API key HMAC logic).
    """
    settings = get_settings()
    if settings.AUTH_MODE != "single_user":
        return True

    print("\n👤 Bootstrapping single-user profile (admin user + primary API key)...")
    logger.info("Bootstrapping single-user profile (admin user + primary API key)...")

    # Ensure RBAC seed for the single-user account (roles, permissions, user row)
    try:
        await ensure_single_user_rbac_seed_if_needed()
    except _AUTHNZ_INIT_NONCRITICAL_EXCEPTIONS as e:
        print(f"⚠️  Single-user RBAC seed failed (continuing): {e}")
        logger.opt(exception=True).warning(
            "Single-user RBAC seed failed in bootstrap_single_user_profile "
            "(continuing): {}",
            e,
        )

    # The RBAC seed path may reset settings/DB pools; refresh settings to reflect
    # the current environment before reading SINGLE_USER_* values.
    settings = get_settings()
    pool = await get_db_pool()
    expected_user_id = settings.SINGLE_USER_FIXED_ID
    api_key_value = settings.SINGLE_USER_API_KEY or ""
    if not api_key_value or api_key_value == "CHANGE_ME_TO_SECURE_API_KEY":
        errors = await _collect_single_user_invariant_errors(
            pool,
            expected_user_id=expected_user_id,
            expected_key_hash=None,
            check_keys=False,
        )
        if errors:
            message = (
                "Single-user bootstrap invariant check failed:\n - "
                + "\n - ".join(errors)
                + "\nResolve conflicts (deactivate extra users, revoke extra API keys) and re-run bootstrap."
            )
            print(f"❌ {message}")
            logger.error(message)
            return False
        print(
            "⚠️  SINGLE_USER_API_KEY is not set or uses the default placeholder; "
            "skipping primary API key bootstrap."
        )
        logger.warning(
            "SINGLE_USER_API_KEY is not set or uses the default placeholder; "
            "skipping single-user primary API key bootstrap."
        )
        return True

    try:
        # Use APIKeyManager to ensure tables and compute key hash
        manager = APIKeyManager(db_pool=pool)
        await manager.initialize()

        key_hash = manager.hash_api_key(api_key_value)
        key_prefix = (api_key_value[:10] + "...") if len(api_key_value) > 10 else api_key_value
        key_identifier = None
        try:
            from tldw_Server_API.app.core.AuthNZ.api_key_crypto import parse_api_key

            parsed = parse_api_key(api_key_value)
            if parsed:
                key_identifier, _secret = parsed
        except _AUTHNZ_INIT_NONCRITICAL_EXCEPTIONS as exc:  # noqa: BLE001 - defensive: parsing failures must not block bootstrap
            logger.opt(exception=True).debug(
                "Failed to parse SINGLE_USER_API_KEY for identifier extraction: {}",
                exc,
            )
            key_identifier = None

        from tldw_Server_API.app.core.AuthNZ.repos.api_keys_repo import AuthnzApiKeysRepo

        repo = AuthnzApiKeysRepo(pool)
        await repo.upsert_primary_key(
            user_id=settings.SINGLE_USER_FIXED_ID,
            key_hash=key_hash,
            key_identifier=key_identifier,
            key_prefix=key_prefix,
            name="single-user primary key",
            description="Primary API key for single-user profile",
            scope="admin",
            is_virtual=False,
        )

        print("✅ Single-user primary API key ensured in AuthNZ store")
        logger.info("Single-user primary API key ensured in AuthNZ store")
        allowed_hashes: set[str] = {key_hash}
        test_api_key = os.getenv("SINGLE_USER_TEST_API_KEY")
        if (
            is_test_mode()
            and test_api_key
            and test_api_key != api_key_value
        ):
            allowed_hashes.add(manager.hash_api_key(test_api_key))
        errors = await _collect_single_user_invariant_errors(
            pool,
            expected_user_id=expected_user_id,
            expected_key_hash=key_hash,
            allowed_non_virtual_key_hashes=allowed_hashes,
            check_keys=True,
        )
        if errors:
            message = (
                "Single-user bootstrap invariant check failed:\n - "
                + "\n - ".join(errors)
                + "\nResolve conflicts (deactivate extra users, revoke extra API keys) and re-run bootstrap."
            )
            print(f"❌ {message}")
            logger.error(message)
            return False
        return True
    except _AUTHNZ_INIT_NONCRITICAL_EXCEPTIONS as e:
        print(f"⚠️  Failed to bootstrap single-user primary API key (continuing): {e}")
        logger.opt(exception=True).warning(
            "Failed to bootstrap single-user primary API key (continuing): {}",
            e,
        )
        return False

async def test_authentication():
    """Test authentication system"""
    print("\n🧪 Testing authentication system...")

    settings = get_settings()

    try:
        if settings.AUTH_MODE == "single_user":
            # Test API key validation
            print("   Testing single-user API key...")
            # This would normally test the actual API key validation
            print("✅ Single-user authentication configured")
        else:
            # Test JWT system
            from tldw_Server_API.app.core.AuthNZ.jwt_service import get_jwt_service

            # Initialize JWT service (sync accessor)
            jwt_service = get_jwt_service()

            # Create a minimal, valid access token
            test_token = jwt_service.create_access_token(
                user_id=1,
                username="test_user",
                role="user",
            )

            # Validate the token via access-token decoder
            decoded = jwt_service.decode_access_token(test_token)

            # sub contains user_id; username holds the display name
            if decoded and decoded.get("username") == "test_user":
                print("✅ JWT authentication system working")
            else:
                print("❌ JWT validation failed")
                return False

        return True

    except _AUTHNZ_INIT_NONCRITICAL_EXCEPTIONS as e:
        print(f"❌ Authentication test failed: {e}")
        return False

async def start_services():
    """Start background services"""
    print("\n🚀 Starting background services...")

    try:
        # Start scheduler
        await start_authnz_scheduler()
        print("✅ Scheduler started")

        # Initialize monitor
        monitor = await get_authnz_monitor()
        print("✅ Monitoring system initialized")

        # Get initial metrics
        metrics = await monitor.get_metrics_summary(60)
        print(f"   Health status: {monitor._calculate_health_status(metrics)}")

        return True

    except _AUTHNZ_INIT_NONCRITICAL_EXCEPTIONS as e:
        print(f"❌ Failed to start services: {e}")
        return False

async def main(*, non_interactive: bool = False, test_setup: bool = False):
    """Main initialization function"""
    print_banner()

    generated_keys_written = False
    if test_setup:
        non_interactive = True
        env_path = _apply_test_setup_env()
        generated_keys_written = True
        print(f"✅ Test setup .env prepared at: {env_path}")
        print(f"⚠️  SINGLE_USER_API_KEY set to insecure test value: {TEST_SETUP_API_KEY}")

    # Step 1: Check environment
    env_status = check_environment()

    if not env_status.get("ok"):
        missing_keys = env_status.get("missing_keys", set())
        if missing_keys:
            env_path = env_status.get("env_path")
            prompt_path = env_path or Path(".env").resolve()
            should_generate = _prompt_yes_no(
                f"\n📝 Missing required keys. Generate and write to {prompt_path}?",
                default_yes=True,
                non_interactive=non_interactive,
            )
            if should_generate:
                generated = generate_secure_keys(requested_keys=missing_keys)
                keys_to_write = _select_keys_to_write(generated, set(missing_keys))
                if env_path and keys_to_write:
                    _write_env_values(env_path, keys_to_write)
                    load_dotenv(dotenv_path=str(env_path), override=True)
                    reset_settings()
                    generated_keys_written = True
                    print(f"✅ Wrote generated keys to: {env_path}")
                else:
                    print("\n⚠️  Could not write generated keys; update your .env manually.")
                    sys.exit(1)
            else:
                print("\n⚠️  Please configure your environment and run again.")
                print("   1. Edit .env file with secure values")
                print("   2. Run: python -m tldw_Server_API.app.core.AuthNZ.initialize")
                sys.exit(1)
        else:
            print("\n⚠️  Please configure your environment and run again.")
            print("   1. Edit .env file with secure values")
            print("   2. Run: python -m tldw_Server_API.app.core.AuthNZ.initialize")
            sys.exit(1)

    # Step 2: Offer to generate keys if needed
    if not generated_keys_written:
        should_generate = _prompt_yes_no(
            "\n📝 Generate new secure keys?",
            default_yes=False,
            non_interactive=non_interactive,
        )
        if should_generate:
            generated = generate_secure_keys()
            env_path = env_status.get("env_path")
            if env_path:
                should_write = _prompt_yes_no(
                    f"\n📝 Write generated keys to {env_path}?",
                    default_yes=False,
                    non_interactive=non_interactive,
                )
                if should_write:
                    _write_env_values(env_path, generated)
                    load_dotenv(dotenv_path=str(env_path), override=True)
                    reset_settings()
                    print(f"✅ Wrote generated keys to: {env_path}")
                else:
                    print("\n⚠️  Update your .env file with these keys and run again.")
                    sys.exit(0)
            else:
                print("\n⚠️  Update your .env file with these keys and run again.")
                sys.exit(0)

    # Step 3: Setup database
    if not await setup_database():
        print("\n❌ Database setup failed")
        sys.exit(1)

    # Step 4: Create admin user / bootstrap profile
    settings = get_settings()
    if settings.AUTH_MODE == "multi_user":
        # Check if any users exist
        try:
            users_db = await get_users_db()
            existing_users = await users_db.list_users(limit=1)

            if not existing_users:
                if non_interactive:
                    print("\n⚠️  No users found and --non-interactive is set; skipping admin user creation.")
                else:
                    should_create = _prompt_yes_no(
                        "\n📝 No users found. Create admin user?",
                        default_yes=True,
                        non_interactive=non_interactive,
                    )
                    if should_create and not await create_admin_user():
                        print("\n⚠️  Admin user creation failed")
            else:
                print(f"\n✅ Found {len(existing_users)} existing user(s)")
        except _AUTHNZ_INIT_NONCRITICAL_EXCEPTIONS as e:
            logger.warning(f"Could not check existing users: {e}")
            if non_interactive:
                print("\n⚠️  --non-interactive is set; skipping admin user creation.")
            else:
                should_create = _prompt_yes_no(
                    "\n📝 Create admin user?",
                    default_yes=True,
                    non_interactive=non_interactive,
                )
                if should_create:
                    await create_admin_user()
    else:
        # Single-user profile: ensure bootstrap user + primary API key
        bootstrap_ok = await bootstrap_single_user_profile()
        if not bootstrap_ok:
            print("\n❌ Single-user bootstrap failed")
            logger.error("Single-user bootstrap failed during AuthNZ initialization")
            test_mode = is_test_mode()
            if not test_mode:
                print("❌ Exiting due to single-user bootstrap failure.")
                sys.exit(1)
            print(
                "⚠️  TEST_MODE is enabled; continuing despite single-user bootstrap failure."
            )
            logger.warning(
                "TEST_MODE enabled; continuing despite single-user bootstrap failure"
            )

    # Step 5: Test authentication
    if not await test_authentication():
        print("\n⚠️  Authentication test failed")

    # Step 6: Start services (optional)
    should_start_services = test_setup or _prompt_yes_no(
        "\n🚀 Start background services?",
        default_yes=False,
        non_interactive=non_interactive,
    )
    if should_start_services:
        await start_services()

    # Summary
    print("\n" + "=" * 60)
    print("✅ AuthNZ Initialization Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review your configuration in .env")
    print("2. Test authentication endpoints")
    print("3. Configure monitoring and alerting")
    print("4. Set up regular backups")
    print("\nTo start the application:")
    print("   python -m uvicorn tldw_Server_API.app.main:app --reload")
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize AuthNZ module")
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Run without prompts (uses defaults and auto-generates missing keys).",
    )
    parser.add_argument(
        "--test-setup",
        action="store_true",
        help=(
            "Prepare a rapid test environment (writes an insecure SINGLE_USER_API_KEY, "
            "populates the .env file, and auto-starts background services)."
        ),
    )
    try:
        args = parser.parse_args()
        asyncio.run(main(non_interactive=args.non_interactive, test_setup=args.test_setup))
    except KeyboardInterrupt:
        print("\n\n⚠️  Initialization cancelled by user")
        sys.exit(0)
    except _AUTHNZ_INIT_NONCRITICAL_EXCEPTIONS as e:
        print(f"\n❌ Initialization failed: {e}")
        logger.exception("Initialization error")
        sys.exit(1)
