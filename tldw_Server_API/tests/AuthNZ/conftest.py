"""
Shared fixtures and configuration for AuthNZ tests.
Provides PostgreSQL test database isolation with transaction rollback.
"""

import os
import json
import contextlib
import shutil
import subprocess
import pytest
import pytest_asyncio
import asyncio
import uuid
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, AsyncGenerator, Optional
from unittest.mock import Mock, AsyncMock, MagicMock

import asyncpg
from loguru import logger

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.password_service import PasswordService
from tldw_Server_API.app.core.AuthNZ.jwt_service import JWTService
from tldw_Server_API.app.core.AuthNZ.settings import Settings
from tldw_Server_API.app.core.AuthNZ.session_manager import SessionManager
from tldw_Server_API.app.core.AuthNZ.rate_limiter import RateLimiter, reset_rate_limiter
from tldw_Server_API.app.core.AuthNZ.lockout_tracker import reset_lockout_tracker
from tldw_Server_API.app.services.registration_service import RegistrationService
from tldw_Server_API.app.core.Audit.unified_audit_service import UnifiedAuditService
from tldw_Server_API.app.services.storage_quota_service import StorageQuotaService
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.AuthNZ.scheduler import reset_authnz_scheduler
from tldw_Server_API.app.core.AuthNZ.token_blacklist import reset_token_blacklist
from tldw_Server_API.app.core.AuthNZ.alerting import reset_security_alert_dispatcher

# Test database configuration
# Allow a full Postgres DSN to configure tests easily
_TEST_DSN = os.getenv("TEST_DATABASE_URL") or os.getenv("DATABASE_URL") or ""
_TEST_DSN = _TEST_DSN.strip()

def _parse_pg_dsn(dsn: str):
    try:
        from urllib.parse import urlparse
        parsed = urlparse(dsn)
        if not parsed.scheme.startswith("postgres"):
            return None
        host = parsed.hostname or "localhost"
        port = int(parsed.port or 5432)
        user = parsed.username or "tldw_user"
        password = parsed.password or "TestPassword123!"
        db = (parsed.path or "/tldw_test").lstrip("/") or "tldw_test"
        return {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "db": db,
        }
    except Exception:
        return None

_parsed = _parse_pg_dsn(_TEST_DSN) if _TEST_DSN else None

TEST_DB_NAME = (_parsed or {}).get("db") or os.getenv("TEST_DB_NAME", "tldw_test")
TEST_DB_HOST = (_parsed or {}).get("host") or os.getenv("TEST_DB_HOST", "localhost")
TEST_DB_PORT = int(((_parsed or {}).get("port")) or int(os.getenv("TEST_DB_PORT", "5432")))
TEST_DB_USER = (_parsed or {}).get("user") or os.getenv("TEST_DB_USER", "tldw_user")
TEST_DB_PASSWORD = (_parsed or {}).get("password") or os.getenv("TEST_DB_PASSWORD", "TestPassword123!")

# Import TestClient for isolated environment
from fastapi.testclient import TestClient



class _StubAuditService:
    """No-op audit service used in TEST_MODE to avoid background tasks."""
    def __init__(self, *args, **kwargs) -> None:
        pass

    async def initialize(self) -> None:
        return None

    async def log_event(self, *args, **kwargs) -> None:
        return None

    async def log_login(self, *args, **kwargs) -> None:
        return None

    async def shutdown(self) -> None:
        return None


class _StubPersonalizationDB:
    """No-op personalization DB used to avoid filesystem writes in tests."""

    def insert_usage_event(self, *args, **kwargs):
        return None

    def __getattr__(self, item):
        # Allow any other method calls without side effects
        def _noop(*_args, **_kwargs):
            return None
        return _noop


async def _can_connect_postgres(host: str, port: int, user: str, password: str, database: str = "postgres") -> bool:
    try:
        conn = await asyncpg.connect(host=host, port=port, user=user, password=password, database=database)
        await conn.close()
        return True
    except Exception as e:
        logger.debug(f"Postgres connectivity check failed: {e}")
        return False


async def _ensure_postgres_available(host: str, port: int, user: str, password: str, *, require_pg: bool, default_db: str = "postgres") -> bool:
    """Try to connect; if not available and local, attempt to start docker, then retry.

    Returns True if Postgres becomes reachable; otherwise False (caller may skip tests).
    """
    if await _can_connect_postgres(host, port, user, password, default_db):
        return True

    # Only attempt Docker on local hostnames
    if str(host) not in {"localhost", "127.0.0.1", "::1"}:
        return False

    if os.getenv("TLDW_TEST_NO_DOCKER", "").lower() in ("1", "true", "yes"):
        return False

    docker_bin = shutil.which("docker")
    if not docker_bin:
        logger.info("Docker not found in PATH; cannot auto-start Postgres for tests")
        return False

    image = os.getenv("TLDW_TEST_PG_IMAGE", "postgres:18")
    container = os.getenv("TLDW_TEST_PG_CONTAINER_NAME", "tldw_postgres_test")

    # Stop and remove an existing container with same name (best-effort)
    try:
        await asyncio.to_thread(subprocess.run, [docker_bin, "rm", "-f", container], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        _ = None

    envs = [
        "-e", f"POSTGRES_USER={user}",
        "-e", f"POSTGRES_PASSWORD={password}",
        # Create a default DB; per-test DBs will be created later as needed
        "-e", f"POSTGRES_DB={default_db}",
    ]
    ports = ["-p", f"{port}:5432"]

    run_cmd = [docker_bin, "run", "-d", "--name", container, *envs, *ports, image]
    logger.info(f"Attempting to start Postgres test container: {' '.join(run_cmd)}")
    try:
        res = await asyncio.to_thread(subprocess.run, run_cmd, check=False, capture_output=True, text=True)
        if res.returncode != 0:
            logger.warning(f"Docker run failed (code {res.returncode}): {res.stderr.strip()}")
            # If container already running under same name, try to reuse without starting
    except Exception as e:
        logger.warning(f"Failed to start Docker Postgres: {e}")
        return False

    # Wait up to ~30 seconds for readiness, trying to connect
    for _ in range(30):
        if await _can_connect_postgres(host, port, user, password, default_db):
            logger.info("Postgres became reachable after docker start")
            return True
        await asyncio.sleep(1)

    logger.warning("Postgres did not become reachable after docker start attempts")
    return False


async def _ensure_postgres_schema_extensions(conn: asyncpg.Connection) -> None:
    """Ensure additional AuthNZ Postgres tables needed by integration tests."""
    statements = [
        # Core org/team hierarchy and memberships
        """
        CREATE TABLE IF NOT EXISTS organizations (
            id SERIAL PRIMARY KEY,
            uuid UUID UNIQUE DEFAULT gen_random_uuid(),
            name VARCHAR(255) UNIQUE NOT NULL,
            slug VARCHAR(255) UNIQUE,
            owner_user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
            is_active BOOLEAN DEFAULT TRUE,
            metadata JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_orgs_owner ON organizations(owner_user_id)",
        """
        CREATE TABLE IF NOT EXISTS org_members (
            org_id INTEGER NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            role VARCHAR(32) DEFAULT 'member',
            status VARCHAR(32) DEFAULT 'active',
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (org_id, user_id)
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_org_members_user ON org_members(user_id)",
        """
        CREATE TABLE IF NOT EXISTS teams (
            id SERIAL PRIMARY KEY,
            org_id INTEGER NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
            name VARCHAR(255) NOT NULL,
            slug VARCHAR(255),
            description TEXT,
            is_active BOOLEAN DEFAULT TRUE,
            metadata JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (org_id, name)
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_teams_org ON teams(org_id)",
        """
        CREATE TABLE IF NOT EXISTS team_members (
            team_id INTEGER NOT NULL REFERENCES teams(id) ON DELETE CASCADE,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            role VARCHAR(32) DEFAULT 'member',
            status VARCHAR(32) DEFAULT 'active',
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (team_id, user_id)
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_team_members_user ON team_members(user_id)",
        # RBAC and scoped permissions
        """
        CREATE TABLE IF NOT EXISTS permissions (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) UNIQUE NOT NULL,
            description TEXT,
            category VARCHAR(100)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS role_permissions (
            role_id INTEGER NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
            permission_id INTEGER NOT NULL REFERENCES permissions(id) ON DELETE CASCADE,
            UNIQUE(role_id, permission_id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS user_roles (
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            role_id INTEGER NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
            UNIQUE(user_id, role_id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS user_permissions (
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            permission_id INTEGER NOT NULL REFERENCES permissions(id) ON DELETE CASCADE,
            granted BOOLEAN NOT NULL DEFAULT TRUE,
            expires_at TIMESTAMP,
            PRIMARY KEY (user_id, permission_id)
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_user_permissions_user ON user_permissions(user_id)",
        # API key audit log used by API key repository tests
        """
        CREATE TABLE IF NOT EXISTS api_key_audit_log (
            id SERIAL PRIMARY KEY,
            api_key_id INTEGER NOT NULL REFERENCES api_keys(id) ON DELETE CASCADE,
            action VARCHAR(50) NOT NULL,
            user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
            ip_address VARCHAR(45),
            user_agent TEXT,
            details JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_api_key_audit_log_api_key_id ON api_key_audit_log(api_key_id)",
        "CREATE INDEX IF NOT EXISTS idx_api_key_audit_log_created_at ON api_key_audit_log(created_at)",
        # Usage tables
        """
        CREATE TABLE IF NOT EXISTS usage_log (
            id SERIAL PRIMARY KEY,
            ts TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
            key_id INTEGER REFERENCES api_keys(id) ON DELETE SET NULL,
            endpoint TEXT,
            status INTEGER,
            latency_ms INTEGER,
            bytes BIGINT,
            bytes_in BIGINT,
            meta JSONB,
            request_id TEXT
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_usage_log_ts ON usage_log(ts)",
        "CREATE INDEX IF NOT EXISTS idx_usage_log_user ON usage_log(user_id)",
        "CREATE INDEX IF NOT EXISTS idx_usage_log_status ON usage_log(status)",
        "CREATE INDEX IF NOT EXISTS idx_usage_log_endpoint ON usage_log(endpoint)",
        "CREATE INDEX IF NOT EXISTS idx_usage_log_request_id ON usage_log(request_id)",
        """
        CREATE TABLE IF NOT EXISTS usage_daily (
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            day DATE NOT NULL,
            requests INTEGER DEFAULT 0,
            errors INTEGER DEFAULT 0,
            bytes_total BIGINT DEFAULT 0,
            bytes_in_total BIGINT DEFAULT 0,
            latency_avg_ms DOUBLE PRECISION,
            PRIMARY KEY (user_id, day)
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_usage_daily_day_user ON usage_daily(day, user_id)",
        """
        CREATE TABLE IF NOT EXISTS llm_usage_log (
            id SERIAL PRIMARY KEY,
            ts TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
            key_id INTEGER REFERENCES api_keys(id) ON DELETE SET NULL,
            endpoint TEXT,
            operation TEXT,
            provider TEXT,
            model TEXT,
            status INTEGER,
            latency_ms INTEGER,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            total_tokens INTEGER,
            prompt_cost_usd DOUBLE PRECISION,
            completion_cost_usd DOUBLE PRECISION,
            total_cost_usd DOUBLE PRECISION,
            currency TEXT DEFAULT 'USD',
            estimated BOOLEAN DEFAULT FALSE,
            request_id TEXT
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_llm_usage_log_ts ON llm_usage_log(ts)",
        "CREATE INDEX IF NOT EXISTS idx_llm_usage_log_user ON llm_usage_log(user_id)",
        "CREATE INDEX IF NOT EXISTS idx_llm_usage_log_provider_model ON llm_usage_log(provider, model)",
        "CREATE INDEX IF NOT EXISTS idx_llm_usage_log_op_ts ON llm_usage_log(operation, ts)",
        "CREATE INDEX IF NOT EXISTS idx_llm_usage_log_key_ts ON llm_usage_log(key_id, ts)",
        """
        CREATE TABLE IF NOT EXISTS llm_usage_daily (
            day DATE NOT NULL,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            operation TEXT NOT NULL,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            requests INTEGER DEFAULT 0,
            errors INTEGER DEFAULT 0,
            input_tokens BIGINT DEFAULT 0,
            output_tokens BIGINT DEFAULT 0,
            total_tokens BIGINT DEFAULT 0,
            total_cost_usd DOUBLE PRECISION DEFAULT 0.0,
            latency_avg_ms DOUBLE PRECISION,
            PRIMARY KEY (day, user_id, operation, provider, model)
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_llm_usage_daily_day_user_op_prov_model ON llm_usage_daily(day, user_id, operation, provider, model)",
        # Generated files table used by storage quota and cleanup paths
        """
        CREATE TABLE IF NOT EXISTS generated_files (
            id SERIAL PRIMARY KEY,
            uuid TEXT UNIQUE NOT NULL,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            org_id INTEGER REFERENCES organizations(id) ON DELETE SET NULL,
            team_id INTEGER REFERENCES teams(id) ON DELETE SET NULL,
            filename TEXT NOT NULL,
            original_filename TEXT,
            storage_path TEXT NOT NULL,
            mime_type TEXT,
            file_size_bytes BIGINT NOT NULL DEFAULT 0,
            checksum TEXT,
            file_category TEXT NOT NULL,
            source_feature TEXT NOT NULL,
            source_ref TEXT,
            folder_tag TEXT,
            tags JSONB,
            is_transient BOOLEAN DEFAULT FALSE,
            expires_at TIMESTAMP,
            retention_policy TEXT DEFAULT 'user_default',
            is_deleted BOOLEAN DEFAULT FALSE,
            deleted_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            accessed_at TIMESTAMP
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_generated_files_user_id ON generated_files(user_id)",
        "CREATE INDEX IF NOT EXISTS idx_generated_files_org_id ON generated_files(org_id)",
        "CREATE INDEX IF NOT EXISTS idx_generated_files_team_id ON generated_files(team_id)",
        "CREATE INDEX IF NOT EXISTS idx_generated_files_uuid ON generated_files(uuid)",
        "CREATE INDEX IF NOT EXISTS idx_generated_files_category ON generated_files(file_category)",
        "CREATE INDEX IF NOT EXISTS idx_generated_files_source_feature ON generated_files(source_feature)",
        "CREATE INDEX IF NOT EXISTS idx_generated_files_folder_tag ON generated_files(folder_tag)",
        "CREATE INDEX IF NOT EXISTS idx_generated_files_is_deleted ON generated_files(is_deleted)",
        "CREATE INDEX IF NOT EXISTS idx_generated_files_expires_at ON generated_files(expires_at)",
        "CREATE INDEX IF NOT EXISTS idx_generated_files_created_at ON generated_files(created_at)",
        "CREATE INDEX IF NOT EXISTS idx_generated_files_user_category ON generated_files(user_id, file_category, is_deleted)",
    ]

    for sql in statements:
        await conn.execute(sql)

    await conn.execute(
        """
        INSERT INTO roles (name, description, is_system) VALUES
        ('admin','Administrator', TRUE),
        ('user','Standard user', TRUE)
        ON CONFLICT (name) DO NOTHING
        """
    )
    for name, desc, cat in (
        ("media.read", "Read media", "media"),
        ("media.create", "Create media", "media"),
    ):
        await conn.execute(
            """
            INSERT INTO permissions (name, description, category)
            VALUES ($1, $2, $3)
            ON CONFLICT (name) DO NOTHING
            """,
            name,
            desc,
            cat,
        )

    role_rows = await conn.fetch("SELECT id, name FROM roles WHERE name IN ('admin','user')")
    perm_rows = await conn.fetch("SELECT id, name FROM permissions WHERE name IN ('media.read','media.create')")
    role_id = {r["name"]: r["id"] for r in role_rows or []}
    perm_id = {p["name"]: p["id"] for p in perm_rows or []}
    for pname in ("media.read", "media.create"):
        if "user" in role_id and pname in perm_id:
            await conn.execute(
                """
                INSERT INTO role_permissions (role_id, permission_id)
                VALUES ($1, $2)
                ON CONFLICT (role_id, permission_id) DO NOTHING
                """,
                role_id["user"],
                perm_id[pname],
            )
        if "admin" in role_id and pname in perm_id:
            await conn.execute(
                """
                INSERT INTO role_permissions (role_id, permission_id)
                VALUES ($1, $2)
                ON CONFLICT (role_id, permission_id) DO NOTHING
                """,
                role_id["admin"],
                perm_id[pname],
            )


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(autouse=True)
async def reset_singletons(request):
    """Auto-reset all singletons before and after each test for clean state."""
    # No session-wide default DB. Tests must use isolated DB fixtures or mocks.
    # Reset before test
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.session_manager import reset_session_manager
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.services.registration_service import reset_registration_service
    from tldw_Server_API.app.services.org_invite_service import reset_invite_service
    from tldw_Server_API.app.core.Billing.enforcement import reset_billing_enforcer
    from tldw_Server_API.app.core.Billing.subscription_service import reset_subscription_service
    from tldw_Server_API.app.core.Audit.unified_audit_service import shutdown_audit_service
    from tldw_Server_API.app.core.AuthNZ.jwt_service import reset_jwt_service
    from tldw_Server_API.app.core.AuthNZ.api_key_manager import reset_api_key_manager
    from tldw_Server_API.app.core.DB_Management.Users_DB import reset_users_db
    from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import close_all_chacha_db_instances

    # Disable CSRF protection for tests
    original_csrf_setting = settings.get('CSRF_ENABLED')
    settings['CSRF_ENABLED'] = False

    close_all_chacha_db_instances()

    await reset_db_pool()
    await reset_session_manager()
    await reset_token_blacklist()
    await reset_security_alert_dispatcher()
    await reset_authnz_scheduler()
    await reset_rate_limiter()
    await reset_lockout_tracker()
    reset_settings()
    reset_jwt_service()
    await reset_registration_service()
    await reset_invite_service()
    await reset_subscription_service()
    reset_billing_enforcer()
    try:
        from tldw_Server_API.app.services.storage_cleanup_service import (
            reset_cleanup_service as _reset_cleanup_service,
        )
        from tldw_Server_API.app.services.storage_quota_service import (
            reset_storage_service as _reset_storage_service,
        )

        await _reset_cleanup_service()
        await _reset_storage_service()
    except Exception:
        _ = None
    await shutdown_audit_service()
    await reset_api_key_manager()
    await reset_users_db()

    # Clear any FastAPI dependency overrides and stub audit unless real audit requested
    try:
        from tldw_Server_API.app.main import app as _app
        _app.dependency_overrides.clear()
        # In TEST_MODE, stub audit service to avoid background task group errors
        try:
            from tldw_Server_API.app.api.v1.API_Deps.Audit_DB_Deps import get_audit_service_for_user
            from tldw_Server_API.app.api.v1.API_Deps.personalization_deps import (
                get_personalization_db_for_user,
                get_usage_event_logger,
                UsageEventLogger,
            )

            async def _override_audit_dep(current_user=None):
                return _StubAuditService()

            def _override_personalization_db(current_user=None):
                return None

            def _override_usage_logger(request=None, user=None, db=None):
                return UsageEventLogger(
                    user_id=str(getattr(user, "id", "test")),
                    db=_StubPersonalizationDB(),
                )

            if not request.node.get_closest_marker("real_audit"):
                _app.dependency_overrides[get_audit_service_for_user] = _override_audit_dep
                _app.dependency_overrides[get_personalization_db_for_user] = _override_personalization_db
                _app.dependency_overrides[get_usage_event_logger] = _override_usage_logger
        except Exception:
            # If import fails here, tests that don't hit audit won't care
            _ = None

        # Also, in TEST_MODE, strip non-essential middlewares that may perform
        # background DB work after response (to avoid TaskGroup noise in full runs)
        try:
            from tldw_Server_API.app.core.Metrics.http_middleware import HTTPMetricsMiddleware as _HTTPMM
            from tldw_Server_API.app.core.Security.middleware import SecurityHeadersMiddleware as _SHM
            from tldw_Server_API.app.core.Security.request_id_middleware import RequestIDMiddleware as _RID
            kept = []
            for m in getattr(_app, 'user_middleware', []):
                if getattr(m, 'cls', None) in (_HTTPMM, _SHM, _RID):
                    continue
                kept.append(m)
            if len(kept) != len(getattr(_app, 'user_middleware', [])):
                _app.user_middleware = kept
                # Rebuild the Starlette middleware stack
                _app.middleware_stack = _app.build_middleware_stack()
        except Exception:
            _ = None
    except Exception:
        _ = None

    yield

    # Reset after test
    await reset_db_pool()
    await reset_session_manager()
    await reset_token_blacklist()
    await reset_security_alert_dispatcher()
    await reset_authnz_scheduler()
    await reset_rate_limiter()
    await reset_lockout_tracker()
    reset_settings()
    reset_jwt_service()
    await reset_registration_service()
    await reset_invite_service()
    await reset_subscription_service()
    reset_billing_enforcer()
    await shutdown_audit_service()
    await reset_api_key_manager()
    await reset_users_db()
    try:
        close_all_chacha_db_instances()
    except Exception:
        _ = None
    try:
        from tldw_Server_API.app.main import app as _app
        _app.dependency_overrides.clear()
    except Exception:
        _ = None

    # Restore original CSRF setting
    if original_csrf_setting is not None:
        settings['CSRF_ENABLED'] = original_csrf_setting
    else:
        settings.pop('CSRF_ENABLED', None)


@pytest_asyncio.fixture
async def real_audit_service(tmp_path):
    """Enable real UnifiedAuditService for this test and isolate per-user DBs.

    - Sets USER_DB_BASE_DIR to a per-test tmp directory
    - Resets settings so config picks up new base dir
    - Ensures audit services are shut down after the test
    """
    import os as _os
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings as _reset_settings
    from tldw_Server_API.app.api.v1.API_Deps.Audit_DB_Deps import shutdown_all_audit_services as _shutdown_all

    _os.environ['USER_DB_BASE_DIR'] = str((tmp_path / 'user_databases').resolve())
    _reset_settings()
    try:
        yield
    finally:
        try:
            await _shutdown_all()
        except Exception:
            _ = None


@pytest_asyncio.fixture
async def isolated_test_environment(monkeypatch, tmp_path):
    """Create isolated DB and app instance for each test - TRUE ONE DB PER TEST."""
    import uuid as uuid_lib

    # Disable CSRF protection for tests
    settings['CSRF_ENABLED'] = False

    # 1. Generate unique DB name for this test
    db_name = f"tldw_test_{uuid_lib.uuid4().hex[:8]}"
    logger.info(f"Creating isolated test database: {db_name}")

    # 2. Create the unique database (skip gracefully if Postgres is unavailable and not required)
    require_pg = os.getenv("TLDW_TEST_POSTGRES_REQUIRED", "").lower() in ("1", "true", "yes")
    # Ensure Postgres is reachable, optionally starting a local dockerized instance
    ok = await _ensure_postgres_available(TEST_DB_HOST, TEST_DB_PORT, TEST_DB_USER, TEST_DB_PASSWORD, require_pg=require_pg, default_db="postgres")
    if not ok:
        if not require_pg:
            import pytest as _pytest
            _pytest.skip("PostgreSQL not available; attempted docker start; skipping AuthNZ integration tests. Set TLDW_TEST_POSTGRES_REQUIRED=1 to enforce.")
        raise RuntimeError("PostgreSQL not available and docker start failed under TLDW_TEST_POSTGRES_REQUIRED=1")
    conn = await asyncpg.connect(
        host=TEST_DB_HOST,
        port=TEST_DB_PORT,
        user=TEST_DB_USER,
        password=TEST_DB_PASSWORD,
        database="postgres"
    )

    try:
        # Drop if exists (cleanup from failed tests)
        await conn.execute(
            """
            SELECT pg_terminate_backend(pid)
            FROM pg_stat_activity
            WHERE datname = $1 AND pid <> pg_backend_pid()
            """,
            db_name,
        )
        await conn.execute(f"DROP DATABASE IF EXISTS {db_name}")

        # Create new database
        await conn.execute(f"CREATE DATABASE {db_name}")
        logger.info(f"Created test database: {db_name}")
    finally:
        await conn.close()

    # 3. Create schema in the new database
    test_conn = await asyncpg.connect(
        host=TEST_DB_HOST,
        port=TEST_DB_PORT,
        user=TEST_DB_USER,
        password=TEST_DB_PASSWORD,
        database=db_name
    )

    try:
        await test_conn.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
        # Create all required tables
        await test_conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                uuid UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
                username VARCHAR(255) UNIQUE NOT NULL,
                email VARCHAR(255) UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                metadata JSONB DEFAULT '{}'::jsonb,
                role VARCHAR(50) NOT NULL DEFAULT 'user',
                is_active BOOLEAN DEFAULT TRUE,
                is_verified BOOLEAN DEFAULT FALSE,
                is_superuser BOOLEAN DEFAULT FALSE,
                failed_login_attempts INTEGER DEFAULT 0,
                locked_until TIMESTAMP,
                storage_quota_mb INTEGER DEFAULT 5120,
                storage_used_mb FLOAT DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                email_verified_at TIMESTAMP,
                two_factor_enabled BOOLEAN DEFAULT FALSE,
                two_factor_secret TEXT,
                totp_secret TEXT,
                backup_codes TEXT,
                created_by INTEGER REFERENCES users(id),
                password_changed_at TIMESTAMP
            )
        """)

        await test_conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                token_hash TEXT UNIQUE NOT NULL,
                refresh_token_hash TEXT UNIQUE,
                encrypted_token TEXT,
                encrypted_refresh TEXT,
                expires_at TIMESTAMP NOT NULL,
                refresh_expires_at TIMESTAMP,
                access_jti VARCHAR(255),
                refresh_jti VARCHAR(255),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ip_address VARCHAR(45),
                user_agent TEXT,
                device_id VARCHAR(255),
                is_active BOOLEAN DEFAULT TRUE,
                is_revoked BOOLEAN DEFAULT FALSE,
                revoked_at TIMESTAMP,
                revoked_by INTEGER REFERENCES users(id),
                revoke_reason TEXT
            )
        """)

        await test_conn.execute("""
            CREATE TABLE IF NOT EXISTS password_history (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        await test_conn.execute("""
            CREATE TABLE IF NOT EXISTS roles (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) UNIQUE NOT NULL,
                description TEXT,
                is_system BOOLEAN DEFAULT FALSE
            )
        """)

        await test_conn.execute("""
            CREATE TABLE IF NOT EXISTS organizations (
                id SERIAL PRIMARY KEY,
                uuid UUID UNIQUE DEFAULT gen_random_uuid(),
                name VARCHAR(255) UNIQUE NOT NULL,
                slug VARCHAR(255) UNIQUE,
                owner_user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
                is_active BOOLEAN DEFAULT TRUE,
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await test_conn.execute("CREATE INDEX IF NOT EXISTS idx_orgs_owner ON organizations(owner_user_id)")

        await test_conn.execute("""
            CREATE TABLE IF NOT EXISTS org_members (
                org_id INTEGER NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                role VARCHAR(32) DEFAULT 'member',
                status VARCHAR(32) DEFAULT 'active',
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (org_id, user_id)
            )
        """)
        await test_conn.execute("CREATE INDEX IF NOT EXISTS idx_org_members_user ON org_members(user_id)")

        await test_conn.execute("""
            CREATE TABLE IF NOT EXISTS teams (
                id SERIAL PRIMARY KEY,
                org_id INTEGER NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
                name VARCHAR(255) NOT NULL,
                slug VARCHAR(255),
                description TEXT,
                is_active BOOLEAN DEFAULT TRUE,
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (org_id, name)
            )
        """)
        await test_conn.execute("CREATE INDEX IF NOT EXISTS idx_teams_org ON teams(org_id)")

        await test_conn.execute("""
            CREATE TABLE IF NOT EXISTS team_members (
                team_id INTEGER NOT NULL REFERENCES teams(id) ON DELETE CASCADE,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                role VARCHAR(32) DEFAULT 'member',
                status VARCHAR(32) DEFAULT 'active',
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (team_id, user_id)
            )
        """)
        await test_conn.execute("CREATE INDEX IF NOT EXISTS idx_team_members_user ON team_members(user_id)")

        await test_conn.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                key_hash TEXT UNIQUE NOT NULL,
                key_id VARCHAR(32),
                key_prefix VARCHAR(16) NOT NULL,
                name VARCHAR(255),
                description TEXT,
                scope VARCHAR(50),
                status VARCHAR(20) DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                last_used_at TIMESTAMP,
                last_used_ip VARCHAR(45),
                usage_count INTEGER DEFAULT 0,
                rate_limit INTEGER,
                allowed_ips TEXT,
                metadata JSONB,
                rotated_from INTEGER REFERENCES api_keys(id),
                rotated_to INTEGER REFERENCES api_keys(id),
                revoked_at TIMESTAMP,
                revoked_by INTEGER,
                revoke_reason TEXT,
                is_virtual BOOLEAN DEFAULT FALSE,
                parent_key_id INTEGER REFERENCES api_keys(id),
                org_id INTEGER,
                team_id INTEGER,
                llm_budget_day_tokens BIGINT,
                llm_budget_month_tokens BIGINT,
                llm_budget_day_usd DOUBLE PRECISION,
                llm_budget_month_usd DOUBLE PRECISION,
                llm_allowed_endpoints TEXT,
                llm_allowed_providers TEXT,
                llm_allowed_models TEXT
            )
        """)
        await test_conn.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id)")
        await test_conn.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_key_hash ON api_keys(key_hash)")
        await test_conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_api_keys_key_id ON api_keys(key_id)")
        await test_conn.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_status ON api_keys(status)")
        await test_conn.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_expires_at ON api_keys(expires_at)")

        await test_conn.execute("""
            CREATE TABLE IF NOT EXISTS registration_codes (
                id SERIAL PRIMARY KEY,
                code VARCHAR(255) UNIQUE NOT NULL,
                max_uses INTEGER DEFAULT 1,
                times_used INTEGER DEFAULT 0,
                expires_at TIMESTAMP,
                created_by INTEGER REFERENCES users(id),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                role_to_grant VARCHAR(50) DEFAULT 'user',
                is_active BOOLEAN DEFAULT TRUE,
                description TEXT,
                allowed_email_domain TEXT,
                metadata JSONB,
                role_id INTEGER REFERENCES roles(id),
                org_id INTEGER,
                org_role VARCHAR(50),
                team_id INTEGER
            )
        """)

        await test_conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
                action VARCHAR(255) NOT NULL,
                target_type VARCHAR(100),
                target_id INTEGER,
                success BOOLEAN DEFAULT TRUE,
                details TEXT,
                ip_address VARCHAR(45),
                user_agent TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        await test_conn.execute("""
            CREATE TABLE IF NOT EXISTS token_blacklist (
                id SERIAL PRIMARY KEY,
                jti VARCHAR(255) UNIQUE NOT NULL,
                user_id INTEGER,
                token_type VARCHAR(50),
                revoked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                reason VARCHAR(255),
                revoked_by INTEGER,
                ip_address VARCHAR(45),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)
        await test_conn.execute("CREATE INDEX IF NOT EXISTS idx_blacklist_jti ON token_blacklist(jti)")
        await test_conn.execute("CREATE INDEX IF NOT EXISTS idx_blacklist_expires ON token_blacklist(expires_at)")
        await test_conn.execute("CREATE INDEX IF NOT EXISTS idx_blacklist_user ON token_blacklist(user_id)")

        await test_conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
                action VARCHAR(255) NOT NULL,
                resource_type VARCHAR(128),
                resource_id INTEGER,
                ip_address VARCHAR(45),
                user_agent TEXT,
                status VARCHAR(32),
                details TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await test_conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit_logs(user_id)")
        await test_conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_logs_action ON audit_logs(action)")
        await test_conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at ON audit_logs(created_at)")

        await test_conn.execute("""
            CREATE TABLE IF NOT EXISTS token_blacklist (
                id SERIAL PRIMARY KEY,
                jti VARCHAR(255) UNIQUE NOT NULL,
                user_id INTEGER,
                token_type VARCHAR(50),
                revoked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                reason VARCHAR(255),
                revoked_by INTEGER,
                ip_address VARCHAR(45),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)
        await test_conn.execute("CREATE INDEX IF NOT EXISTS idx_blacklist_jti ON token_blacklist(jti)")
        await test_conn.execute("CREATE INDEX IF NOT EXISTS idx_blacklist_expires ON token_blacklist(expires_at)")
        await test_conn.execute("CREATE INDEX IF NOT EXISTS idx_blacklist_user ON token_blacklist(user_id)")

        # RBAC core tables (minimal for tests)
        await test_conn.execute("""
            CREATE TABLE IF NOT EXISTS roles (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100) UNIQUE NOT NULL,
                description TEXT,
                is_system BOOLEAN DEFAULT FALSE
            )
        """)
        await test_conn.execute("""
            CREATE TABLE IF NOT EXISTS permissions (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) UNIQUE NOT NULL,
                description TEXT,
                category VARCHAR(100)
            )
        """)
        await test_conn.execute("""
            CREATE TABLE IF NOT EXISTS role_permissions (
                role_id INTEGER NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
                permission_id INTEGER NOT NULL REFERENCES permissions(id) ON DELETE CASCADE,
                UNIQUE(role_id, permission_id)
            )
        """)
        await test_conn.execute("""
            CREATE TABLE IF NOT EXISTS user_roles (
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                role_id INTEGER NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
                UNIQUE(user_id, role_id)
            )
        """)
        await test_conn.execute("""
            CREATE TABLE IF NOT EXISTS user_permissions (
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                permission_id INTEGER NOT NULL REFERENCES permissions(id) ON DELETE CASCADE,
                granted BOOLEAN NOT NULL DEFAULT TRUE,
                expires_at TIMESTAMP,
                PRIMARY KEY (user_id, permission_id)
            )
        """)

        # Create indexes
        await test_conn.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
        await test_conn.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")
        await test_conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id)")
        await test_conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_token_hash ON sessions(token_hash)")
        await test_conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_log_user_id ON audit_log(user_id)")
        await test_conn.execute("CREATE INDEX IF NOT EXISTS idx_user_permissions_user ON user_permissions(user_id)")

        # Seed minimal roles expected by tests
        await test_conn.execute("""
            INSERT INTO roles (name, description, is_system) VALUES
            ('admin','Administrator', TRUE)
            ON CONFLICT (name) DO NOTHING
        """)
        await test_conn.execute("""
            INSERT INTO roles (name, description, is_system) VALUES
            ('user','Standard user', TRUE)
            ON CONFLICT (name) DO NOTHING
        """)
        # Seed baseline permissions for default roles to align with application migrations
        perm_defs = [
            ("media.read", "Read media", "media"),
            ("media.create", "Create media", "media"),
        ]
        for name, desc, cat in perm_defs:
            await test_conn.execute(
                """
                INSERT INTO permissions (name, description, category)
                VALUES ($1, $2, $3)
                ON CONFLICT (name) DO NOTHING
                """,
                name,
                desc,
                cat,
            )

        role_rows = await test_conn.fetch("SELECT id, name FROM roles WHERE name IN ('admin','user')")
        perm_rows = await test_conn.fetch("SELECT id, name FROM permissions WHERE name IN ('media.read','media.create')")
        role_id = {r["name"]: r["id"] for r in role_rows or []}
        perm_id = {p["name"]: p["id"] for p in perm_rows or []}

        for pname in ("media.read", "media.create"):
            if "user" in role_id and pname in perm_id:
                await test_conn.execute(
                    """
                    INSERT INTO role_permissions (role_id, permission_id)
                    VALUES ($1, $2)
                    ON CONFLICT (role_id, permission_id) DO NOTHING
                    """,
                    role_id["user"],
                    perm_id[pname],
                )
            if "admin" in role_id and pname in perm_id:
                await test_conn.execute(
                    """
                    INSERT INTO role_permissions (role_id, permission_id)
                    VALUES ($1, $2)
                    ON CONFLICT (role_id, permission_id) DO NOTHING
                    """,
                    role_id["admin"],
                    perm_id[pname],
                )

        # Billing-related tables used by integration tests.
        # NOTE: Keep these definitions in sync with the corresponding
        # AuthNZ migrations:
        #   - migration_032_create_stripe_webhook_events
        #   - migration_033_create_payment_history
        #   - migration_034_create_billing_audit_log
        await test_conn.execute(
            """
            CREATE TABLE IF NOT EXISTS stripe_webhook_events (
                id SERIAL PRIMARY KEY,
                stripe_event_id TEXT UNIQUE NOT NULL,
                event_type TEXT NOT NULL,
                event_data JSONB NOT NULL,
                status TEXT DEFAULT 'pending',
                processed_at TIMESTAMPTZ,
                error_message TEXT,
                retry_count INTEGER DEFAULT 0,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        await test_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_stripe_events_event_id ON stripe_webhook_events(stripe_event_id)"
        )
        await test_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_stripe_events_type ON stripe_webhook_events(event_type)"
        )
        await test_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_stripe_events_status ON stripe_webhook_events(status)"
        )

        await test_conn.execute(
            """
            CREATE TABLE IF NOT EXISTS payment_history (
                id SERIAL PRIMARY KEY,
                org_id INTEGER NOT NULL,
                stripe_invoice_id TEXT,
                stripe_payment_intent_id TEXT,
                amount_cents INTEGER NOT NULL,
                currency TEXT DEFAULT 'usd',
                status TEXT NOT NULL,
                description TEXT,
                invoice_pdf_url TEXT,
                receipt_url TEXT,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        await test_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_payment_history_org ON payment_history(org_id)"
        )
        await test_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_payment_history_org_date ON payment_history(org_id, created_at)"
        )
        await test_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_payment_history_stripe_invoice ON payment_history(stripe_invoice_id)"
        )

        await test_conn.execute(
            """
            CREATE TABLE IF NOT EXISTS billing_audit_log (
                id SERIAL PRIMARY KEY,
                org_id INTEGER NOT NULL,
                user_id INTEGER,
                action TEXT NOT NULL,
                details TEXT,
                ip_address TEXT,
                user_agent TEXT,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        await test_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_billing_audit_org ON billing_audit_log(org_id)"
        )
        await test_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_billing_audit_action ON billing_audit_log(action)"
        )
        await test_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_billing_audit_created ON billing_audit_log(created_at)"
        )

        # Core billing tables used by billing endpoints.
        await test_conn.execute(
            """
            CREATE TABLE IF NOT EXISTS subscription_plans (
                id SERIAL PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                display_name TEXT NOT NULL,
                description TEXT,
                stripe_product_id TEXT,
                stripe_price_id TEXT,
                stripe_price_id_yearly TEXT,
                price_usd_monthly DOUBLE PRECISION DEFAULT 0,
                price_usd_yearly DOUBLE PRECISION DEFAULT 0,
                limits_json JSONB NOT NULL,
                is_active BOOLEAN DEFAULT TRUE,
                is_public BOOLEAN DEFAULT FALSE,
                sort_order INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        await test_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_subscription_plans_name ON subscription_plans(name)"
        )
        await test_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_subscription_plans_active ON subscription_plans(is_active)"
        )

        default_plans = [
            {
                "name": "free",
                "display_name": "Free",
                "description": "Internal/default plan (not publicly listed)",
                "price_usd_monthly": 0,
                "price_usd_yearly": 0,
                "sort_order": 0,
                "limits": {
                    "storage_mb": 1024,
                    "api_calls_day": 100,
                    "api_calls_month": 3000,
                    "llm_tokens_day": 10000,
                    "llm_tokens_month": 300000,
                    "llm_cost_month_usd": 0,
                    "transcription_minutes_month": 10,
                    "rag_queries_day": 50,
                    "concurrent_jobs": 1,
                    "team_members": 1,
                    "rate_limit_rpm": 10,
                    "features": ["basic_search", "fts5_search", "basic_chat"],
                },
            },
        ]

        for plan in default_plans:
            await test_conn.execute(
                """
                INSERT INTO subscription_plans
                (name, display_name, description, price_usd_monthly, price_usd_yearly, limits_json, sort_order)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (name) DO NOTHING
                """,
                plan["name"],
                plan["display_name"],
                plan["description"],
                plan["price_usd_monthly"],
                plan["price_usd_yearly"],
                json.dumps(plan["limits"]),
                plan["sort_order"],
            )

        await test_conn.execute(
            """
            CREATE TABLE IF NOT EXISTS org_subscriptions (
                id SERIAL PRIMARY KEY,
                org_id INTEGER NOT NULL UNIQUE REFERENCES organizations(id) ON DELETE CASCADE,
                plan_id INTEGER NOT NULL REFERENCES subscription_plans(id) ON DELETE RESTRICT,
                stripe_customer_id TEXT,
                stripe_subscription_id TEXT,
                stripe_subscription_status TEXT,
                billing_cycle TEXT DEFAULT 'monthly',
                current_period_start TIMESTAMP,
                current_period_end TIMESTAMP,
                status TEXT DEFAULT 'active',
                trial_start TIMESTAMP,
                trial_end TIMESTAMP,
                canceled_at TIMESTAMP,
                cancel_at_period_end BOOLEAN DEFAULT FALSE,
                custom_limits_json JSONB,
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        await test_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_org_subs_org ON org_subscriptions(org_id)"
        )
        await test_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_org_subs_stripe_customer ON org_subscriptions(stripe_customer_id)"
        )
        await test_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_org_subs_stripe_sub ON org_subscriptions(stripe_subscription_id)"
        )
        await test_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_org_subs_status ON org_subscriptions(status)"
        )

        # Org invites tables used by org invite endpoints.
        await test_conn.execute(
            """
            CREATE TABLE IF NOT EXISTS org_invites (
                id SERIAL PRIMARY KEY,
                code TEXT UNIQUE NOT NULL,
                org_id INTEGER NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
                team_id INTEGER REFERENCES teams(id) ON DELETE CASCADE,
                role_to_grant TEXT DEFAULT 'member',
                created_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMPTZ NOT NULL,
                max_uses INTEGER DEFAULT 1,
                uses_count INTEGER DEFAULT 0,
                is_active BOOLEAN DEFAULT TRUE,
                allowed_email_domain TEXT,
                description TEXT,
                metadata JSONB
            )
            """
        )
        await test_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_org_invites_code ON org_invites(code)"
        )
        await test_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_org_invites_org_active ON org_invites(org_id, is_active)"
        )
        await test_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_org_invites_expires ON org_invites(expires_at)"
        )

        await test_conn.execute(
            """
            CREATE TABLE IF NOT EXISTS org_invite_redemptions (
                id SERIAL PRIMARY KEY,
                invite_id INTEGER NOT NULL REFERENCES org_invites(id) ON DELETE CASCADE,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                redeemed_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                ip_address TEXT,
                user_agent TEXT,
                UNIQUE(invite_id, user_id)
            )
            """
        )
        await test_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_invite_redemptions_invite ON org_invite_redemptions(invite_id)"
        )
        await test_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_invite_redemptions_user ON org_invite_redemptions(user_id)"
        )

        await _ensure_postgres_schema_extensions(test_conn)
        logger.info(f"Created schema in test database: {db_name}")
    finally:
        await test_conn.close()

    # 4. Set environment variables for this test
    db_url = f"postgresql://{TEST_DB_USER}:{TEST_DB_PASSWORD}@{TEST_DB_HOST}:{TEST_DB_PORT}/{db_name}"
    monkeypatch.setenv("TEST_DATABASE_URL", db_url)
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", db_url)
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret-key-for-testing-only")
    monkeypatch.setenv("ENABLE_REGISTRATION", "true")
    monkeypatch.setenv("REQUIRE_REGISTRATION_CODE", "false")
    monkeypatch.setenv("EMAIL_VERIFICATION_REQUIRED", "false")
    monkeypatch.setenv("USER_DB_BASE_DIR", str((tmp_path / "user_databases").resolve()))
    # Defer heavy startup (embeddings, TTS, request queue, etc.) to prevent local hangs
    monkeypatch.setenv("DEFER_HEAVY_STARTUP", "true")
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTHNZ_FORCE_REAL_SESSION_MANAGER", "1")

    # 5. Reset ALL singletons to force fresh initialization with new DB
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.session_manager import reset_session_manager
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.AuthNZ.api_key_manager import reset_api_key_manager
    from tldw_Server_API.app.core.AuthNZ.rate_limiter import reset_rate_limiter
    from tldw_Server_API.app.core.AuthNZ.lockout_tracker import reset_lockout_tracker
    from tldw_Server_API.app.services.registration_service import reset_registration_service
    from tldw_Server_API.app.core.Audit.unified_audit_service import shutdown_audit_service
    from tldw_Server_API.app.core.DB_Management.Users_DB import reset_users_db
    from tldw_Server_API.app.core.AuthNZ.jwt_service import reset_jwt_service
    from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import (
        ensure_billing_tables_pg,
        ensure_identity_federation_tables_pg,
    )

    await reset_db_pool()
    await reset_session_manager()
    await reset_api_key_manager()
    await reset_rate_limiter()
    await reset_lockout_tracker()
    reset_settings()
    reset_jwt_service()
    await reset_registration_service()
    await shutdown_audit_service()
    await reset_users_db()

    # Run the same Postgres compatibility DDL used by production bootstrap,
    # then tear the pool back down so the TestClient creates a fresh pool on
    # its own event loop.
    try:
        bootstrap_pool = await get_db_pool()
        await ensure_billing_tables_pg(bootstrap_pool)
        await ensure_identity_federation_tables_pg(bootstrap_pool)
    finally:
        await reset_db_pool()

    # 5.1 Skip forcing a DatabasePool into the app to avoid cross-event-loop issues.
    #     Let the FastAPI app create its own pool within its own loop when handling requests.
    # 5.2 We already created the minimal schema required for registration/login above.
    #     Avoid calling module bootstrap that could prime a global pool on the fixture loop.

    # 7. Create TestClient (DB exists, singletons reset, env vars set)
    from tldw_Server_API.app.main import app as _app
    # Diagnostics: verify settings and DB URL are pointing to our per-test DB
    try:
        from tldw_Server_API.app.core.AuthNZ.settings import get_settings as _get_settings
        _s = _get_settings()
        logger.info(f"AuthNZ test fixture DB URL: {_s.DATABASE_URL}")
        logger.info(f"AuthNZ mode: {_s.AUTH_MODE} | CSRF_ENABLED={settings.get('CSRF_ENABLED')}")
    except Exception as _diag_e:
        logger.warning(f"AuthNZ test fixture diagnostics failed: {_diag_e}")
    with TestClient(_app) as client:
        yield client, db_name

    # 8. Cleanup: reset singletons again
    await reset_db_pool()
    await reset_session_manager()
    await reset_token_blacklist()
    await reset_authnz_scheduler()
    await reset_rate_limiter()
    await reset_lockout_tracker()
    reset_settings()
    try:
        from tldw_Server_API.app.services.storage_cleanup_service import (
            reset_cleanup_service as _reset_cleanup_service,
        )
        from tldw_Server_API.app.services.storage_quota_service import (
            reset_storage_service as _reset_storage_service,
        )

        await _reset_cleanup_service()
        await _reset_storage_service()
    except Exception:
        _ = None
    await reset_registration_service()
    await shutdown_audit_service()
    await reset_users_db()

    # 9. Drop the unique database
    cleanup_conn = await asyncpg.connect(
        host=TEST_DB_HOST,
        port=TEST_DB_PORT,
        user=TEST_DB_USER,
        password=TEST_DB_PASSWORD,
        database="postgres"
    )

    try:
        await cleanup_conn.execute(
            """
            SELECT pg_terminate_backend(pid)
            FROM pg_stat_activity
            WHERE datname = $1 AND pid <> pg_backend_pid()
            """,
            db_name,
        )
        await cleanup_conn.execute(f"DROP DATABASE IF EXISTS {db_name}")
        logger.info(f"Dropped test database: {db_name}")
    finally:
        await cleanup_conn.close()

    # Re-enable CSRF protection after test
    settings.pop('CSRF_ENABLED', None)


@pytest_asyncio.fixture
async def setup_test_database(monkeypatch):
    """Create and setup the test database for the test session."""
    # Ensure FastAPI + core settings pick Postgres for this test DB
    require_pg = os.getenv("TLDW_TEST_POSTGRES_REQUIRED", "").lower() in ("1", "true", "yes")
    test_dsn = f"postgresql://{TEST_DB_USER}:{TEST_DB_PASSWORD}@{TEST_DB_HOST}:{TEST_DB_PORT}/{TEST_DB_NAME}"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", test_dsn)
    try:
        from tldw_Server_API.app.core.AuthNZ.settings import reset_settings as _reset_settings
        from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool as _reset_db_pool
        from tldw_Server_API.app.core.AuthNZ.rate_limiter import reset_rate_limiter as _reset_rate_limiter
        from tldw_Server_API.app.core.AuthNZ.lockout_tracker import reset_lockout_tracker as _reset_lockout_tracker
        _reset_settings()
        # Reset any pre-existing pool so app endpoints use Postgres on first access
        # (some tests hit endpoints that call get_db_pool inside request handlers)
        await _reset_db_pool()
        await _reset_rate_limiter()
        await _reset_lockout_tracker()
    except Exception:
        _ = None
    # Ensure Postgres reachable before creating the session DB
    ok = await _ensure_postgres_available(TEST_DB_HOST, TEST_DB_PORT, TEST_DB_USER, TEST_DB_PASSWORD, require_pg=require_pg, default_db="postgres")
    if not ok:
        if not require_pg:
            import pytest as _pytest
            _pytest.skip("PostgreSQL not available; attempted docker start; skipping AuthNZ Postgres-backed tests. Set TLDW_TEST_POSTGRES_REQUIRED=1 to enforce.")
        raise RuntimeError("PostgreSQL not available and docker start failed under TLDW_TEST_POSTGRES_REQUIRED=1")
    # Connect to postgres database to create test database
    conn = await asyncpg.connect(
        host=TEST_DB_HOST,
        port=TEST_DB_PORT,
        user=TEST_DB_USER,
        password=TEST_DB_PASSWORD,
        database="postgres"
    )

    try:
        # Drop test database if it exists
        await conn.execute(
            """
            SELECT pg_terminate_backend(pid)
            FROM pg_stat_activity
            WHERE datname = $1 AND pid <> pg_backend_pid()
            """,
            TEST_DB_NAME,
        )
        await conn.execute(f"DROP DATABASE IF EXISTS {TEST_DB_NAME}")

        # Create test database
        await conn.execute(f"CREATE DATABASE {TEST_DB_NAME}")
        logger.info(f"Created test database: {TEST_DB_NAME}")

    finally:
        await conn.close()

    # Connect to test database and create base minimal schema (core tables)
    test_conn = await asyncpg.connect(
        host=TEST_DB_HOST,
        port=TEST_DB_PORT,
        user=TEST_DB_USER,
        password=TEST_DB_PASSWORD,
        database=TEST_DB_NAME
    )

    try:
        await test_conn.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
        # Create tables
        await test_conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                uuid UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
                username VARCHAR(255) UNIQUE NOT NULL,
                email VARCHAR(255) UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                metadata JSONB DEFAULT '{}'::jsonb,
                role VARCHAR(50) NOT NULL DEFAULT 'user',
                is_active BOOLEAN DEFAULT TRUE,
                is_verified BOOLEAN DEFAULT FALSE,
                is_superuser BOOLEAN DEFAULT FALSE,
                failed_login_attempts INTEGER DEFAULT 0,
                locked_until TIMESTAMP,
                storage_quota_mb INTEGER DEFAULT 5120,
                storage_used_mb FLOAT DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                email_verified_at TIMESTAMP,
                two_factor_enabled BOOLEAN DEFAULT FALSE,
                two_factor_secret TEXT,
                totp_secret TEXT,
                backup_codes TEXT,
                created_by INTEGER REFERENCES users(id),
                password_changed_at TIMESTAMP
            )
        """)

        await test_conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                token_hash TEXT UNIQUE NOT NULL,
                refresh_token_hash TEXT UNIQUE,
                encrypted_token TEXT,
                encrypted_refresh TEXT,
                expires_at TIMESTAMP NOT NULL,
                refresh_expires_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ip_address VARCHAR(45),
                user_agent TEXT,
                device_id VARCHAR(255),
                is_active BOOLEAN DEFAULT TRUE,
                is_revoked BOOLEAN DEFAULT FALSE,
                revoked_at TIMESTAMP,
                revoked_by INTEGER REFERENCES users(id),
                revoke_reason TEXT,
                access_jti VARCHAR(255),
                refresh_jti VARCHAR(255)
            )
        """)

        await test_conn.execute("""
            CREATE TABLE IF NOT EXISTS password_history (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        await test_conn.execute("""
            CREATE TABLE IF NOT EXISTS roles (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) UNIQUE NOT NULL,
                description TEXT,
                is_system BOOLEAN DEFAULT FALSE
            )
        """)

        await test_conn.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                key_hash TEXT UNIQUE NOT NULL,
                key_id VARCHAR(32),
                key_prefix VARCHAR(16) NOT NULL,
                name VARCHAR(255),
                description TEXT,
                scope VARCHAR(50),
                status VARCHAR(20) DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                last_used_at TIMESTAMP,
                last_used_ip VARCHAR(45),
                usage_count INTEGER DEFAULT 0,
                rate_limit INTEGER,
                allowed_ips TEXT,
                metadata JSONB,
                rotated_from INTEGER REFERENCES api_keys(id),
                rotated_to INTEGER REFERENCES api_keys(id),
                revoked_at TIMESTAMP,
                revoked_by INTEGER,
                revoke_reason TEXT,
                is_virtual BOOLEAN DEFAULT FALSE,
                parent_key_id INTEGER REFERENCES api_keys(id),
                org_id INTEGER,
                team_id INTEGER,
                llm_budget_day_tokens BIGINT,
                llm_budget_month_tokens BIGINT,
                llm_budget_day_usd DOUBLE PRECISION,
                llm_budget_month_usd DOUBLE PRECISION,
                llm_allowed_endpoints TEXT,
                llm_allowed_providers TEXT,
                llm_allowed_models TEXT
            )
        """)
        await test_conn.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id)")
        await test_conn.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_key_hash ON api_keys(key_hash)")
        await test_conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_api_keys_key_id ON api_keys(key_id)")
        await test_conn.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_status ON api_keys(status)")
        await test_conn.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_expires_at ON api_keys(expires_at)")

        await test_conn.execute("""
            CREATE TABLE IF NOT EXISTS registration_codes (
                id SERIAL PRIMARY KEY,
                code VARCHAR(255) UNIQUE NOT NULL,
                max_uses INTEGER DEFAULT 1,
                times_used INTEGER DEFAULT 0,
                expires_at TIMESTAMP,
                created_by INTEGER REFERENCES users(id),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                role_to_grant VARCHAR(50) DEFAULT 'user',
                is_active BOOLEAN DEFAULT TRUE,
                description TEXT,
                allowed_email_domain TEXT,
                metadata JSONB,
                role_id INTEGER REFERENCES roles(id),
                org_id INTEGER,
                org_role VARCHAR(50),
                team_id INTEGER
            )
        """)

        await test_conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
                action VARCHAR(255) NOT NULL,
                target_type VARCHAR(100),
                target_id INTEGER,
                success BOOLEAN DEFAULT TRUE,
                details TEXT,
                ip_address VARCHAR(45),
                user_agent TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        await test_conn.execute("""
            CREATE TABLE IF NOT EXISTS rate_limits (
                identifier TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                request_count INTEGER NOT NULL,
                window_start TIMESTAMPTZ NOT NULL,
                PRIMARY KEY (identifier, endpoint, window_start)
            )
        """)
        await test_conn.execute("""
            CREATE TABLE IF NOT EXISTS failed_attempts (
                identifier TEXT NOT NULL,
                attempt_type TEXT NOT NULL,
                attempt_count INTEGER NOT NULL,
                window_start TIMESTAMPTZ NOT NULL,
                PRIMARY KEY (identifier, attempt_type)
            )
        """)
        await test_conn.execute("""
            CREATE TABLE IF NOT EXISTS account_lockouts (
                identifier TEXT NOT NULL,
                attempt_type TEXT NOT NULL,
                locked_until TIMESTAMPTZ NOT NULL,
                reason TEXT,
                PRIMARY KEY (identifier, attempt_type)
            )
        """)
        await test_conn.execute("CREATE INDEX IF NOT EXISTS idx_rate_limits_identifier ON rate_limits(identifier)")

        # Create indexes
        await test_conn.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
        await test_conn.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")
        await test_conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id)")
        await test_conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_token_hash ON sessions(token_hash)")
        await test_conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_log_user_id ON audit_log(user_id)")

        await _ensure_postgres_schema_extensions(test_conn)
        logger.info("Created test database schema")

    finally:
        await test_conn.close()

    # Also run the AuthNZ module's Postgres bootstrap to ensure full schema parity
    # (sessions, registration_codes, RBAC, API keys, usage tables, etc.)
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"postgresql://{TEST_DB_USER}:{TEST_DB_PASSWORD}@{TEST_DB_HOST}:{TEST_DB_PORT}/{TEST_DB_NAME}")
    monkeypatch.setenv("JWT_SECRET_KEY", os.environ.get("JWT_SECRET_KEY", "test-secret-key-for-testing-only"))
    monkeypatch.setenv("AUTHNZ_FORCE_REAL_SESSION_MANAGER", "1")
    try:
        from tldw_Server_API.app.core.AuthNZ.initialize import setup_database as _authnz_setup_db
        await _authnz_setup_db()
        logger.info("AuthNZ Postgres schema bootstrap completed for session test DB")
    except Exception as exc:
        logger.exception(f"AuthNZ schema bootstrap failed in setup_test_database: {exc}")
        raise RuntimeError("AuthNZ Postgres schema bootstrap failed in setup_test_database") from exc

    yield

    # Cleanup: Drop test database after all tests
    cleanup_conn = await asyncpg.connect(
        host=TEST_DB_HOST,
        port=TEST_DB_PORT,
        user=TEST_DB_USER,
        password=TEST_DB_PASSWORD,
        database="postgres"
    )

    try:
        await cleanup_conn.execute(
            """
            SELECT pg_terminate_backend(pid)
            FROM pg_stat_activity
            WHERE datname = $1 AND pid <> pg_backend_pid()
            """,
            TEST_DB_NAME,
        )
        await cleanup_conn.execute(f"DROP DATABASE IF EXISTS {TEST_DB_NAME}")
        logger.info(f"Dropped test database: {TEST_DB_NAME}")
    finally:
        await cleanup_conn.close()


@pytest_asyncio.fixture(scope="function")
async def clean_database(setup_test_database):
    """Ensure database is clean before each test."""
    tables = [
        "generated_files",
        "llm_usage_daily",
        "llm_usage_log",
        "usage_daily",
        "usage_log",
        "org_invite_redemptions",
        "org_invites",
        "team_members",
        "teams",
        "org_members",
        "organizations",
        "billing_audit_log",
        "payment_history",
        "stripe_webhook_events",
        "org_subscriptions",
        "subscription_plans",
        "api_key_audit_log",
        "api_keys",
        "token_blacklist",
        "user_permissions",
        "user_roles",
        "role_permissions",
        "permissions",
        "audit_logs",
        "audit_log",
        "registration_codes",
        "password_history",
        "sessions",
        "account_lockouts",
        "failed_attempts",
        "rate_limits",
        "users",
    ]

    async def _truncate_all(connection: asyncpg.Connection) -> None:
        for table in tables:
            with contextlib.suppress(Exception):
                await connection.execute(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE")

    conn = await asyncpg.connect(
        host=TEST_DB_HOST,
        port=TEST_DB_PORT,
        user=TEST_DB_USER,
        password=TEST_DB_PASSWORD,
        database=TEST_DB_NAME
    )

    try:
        await _truncate_all(conn)

        logger.debug("Cleaned test database tables")
    finally:
        await conn.close()

    yield

    # Clean up after test
    cleanup_conn = await asyncpg.connect(
        host=TEST_DB_HOST,
        port=TEST_DB_PORT,
        user=TEST_DB_USER,
        password=TEST_DB_PASSWORD,
        database=TEST_DB_NAME
    )

    try:
        await _truncate_all(cleanup_conn)
    finally:
        await cleanup_conn.close()


@pytest_asyncio.fixture
async def test_db_pool(setup_test_database, clean_database):
    """Create a test database pool connected to the test PostgreSQL database."""
    test_database_url = f"postgresql://{TEST_DB_USER}:{TEST_DB_PASSWORD}@{TEST_DB_HOST}:{TEST_DB_PORT}/{TEST_DB_NAME}"

    test_settings = Settings(
        AUTH_MODE="multi_user",
        DATABASE_URL=test_database_url,
        JWT_SECRET_KEY="test-secret-key-for-testing-only",
        ENABLE_REGISTRATION=True,
        REQUIRE_REGISTRATION_CODE=False,
    )

    pool = DatabasePool(test_settings)
    await pool.initialize()

    try:
        yield pool
    finally:
        await pool.close()


@pytest_asyncio.fixture
async def mock_db_pool():
    """Create a mock database pool for unit testing."""
    pool = AsyncMock(spec=DatabasePool)

    # Mock connection context manager
    mock_conn = AsyncMock()
    pool.transaction.return_value.__aenter__.return_value = mock_conn
    pool.transaction.return_value.__aexit__.return_value = None

    # Mock fetchone for user queries
    pool.fetchone = AsyncMock()
    pool.fetchrow = AsyncMock()
    pool.fetch = AsyncMock()
    pool.fetchval = AsyncMock()

    # Mock execute for updates
    pool.execute = AsyncMock()
    pool.acquire = AsyncMock()
    pool.release = AsyncMock()

    return pool


@pytest.fixture
def password_service():
    """Create a password service instance for testing."""
    return PasswordService()


@pytest.fixture
def jwt_settings():
    """Create JWT settings for testing."""
    return Settings(
        AUTH_MODE="multi_user",
        JWT_SECRET_KEY="test-secret-key-for-testing-only-needs-32-chars-minimum",
        JWT_ALGORITHM="HS256",
        ACCESS_TOKEN_EXPIRE_MINUTES=30,
        REFRESH_TOKEN_EXPIRE_DAYS=7,
        SESSION_CLEANUP_INTERVAL_HOURS=24,
        SESSION_MAX_AGE_DAYS=30,
        RATE_LIMIT_MAX_REQUESTS=100,
        RATE_LIMIT_WINDOW_SECONDS=60,
        PASSWORD_MIN_LENGTH=8,
        PASSWORD_REQUIRE_UPPERCASE=True,
        PASSWORD_REQUIRE_LOWERCASE=True,
        PASSWORD_REQUIRE_DIGIT=True,
        PASSWORD_REQUIRE_SPECIAL=False,
        REGISTRATION_ENABLED=True,
        REGISTRATION_REQUIRE_CODE=False,
        REGISTRATION_CODES=[],
        DEFAULT_USER_ROLE="user",
        DEFAULT_STORAGE_QUOTA_MB=1000,
        EMAIL_VERIFICATION_REQUIRED=False,
        CORS_ORIGINS=["*"],
        API_PREFIX="/api/v1"
    )


@pytest.fixture
def jwt_service(jwt_settings):
    """Create a JWT service instance for testing."""
    return JWTService(settings=jwt_settings)


@pytest_asyncio.fixture
async def session_manager(test_db_pool):
    """Create a session manager instance for testing."""
    manager = SessionManager(db_pool=test_db_pool)
    yield manager
    # Cleanup
    await manager.shutdown()


@pytest_asyncio.fixture
async def rate_limiter():
    """Create a rate limiter instance for testing."""
    limiter = RateLimiter()
    yield limiter


@pytest_asyncio.fixture
async def registration_service(test_db_pool, password_service):
    """Create a registration service instance for testing."""
    return RegistrationService(
        db_pool=test_db_pool,
        password_service=password_service
    )


@pytest_asyncio.fixture
async def audit_service(test_db_pool):
    """Create an audit service instance for testing."""
    service = UnifiedAuditService()
    await service.initialize()
    try:
        yield service
    finally:
        await service.stop()


@pytest_asyncio.fixture
async def storage_service(test_db_pool):
    """Create a storage quota service instance for testing."""
    return StorageQuotaService(test_db_pool)


@pytest_asyncio.fixture
async def test_user(test_db_pool, password_service):
    """Create a test user in the database."""
    user_uuid = str(uuid.uuid4())
    password = "Test@Pass#2024!"
    password_hash = password_service.hash_password(password)

    async with test_db_pool.acquire() as conn:
        user = await conn.fetchrow("""
            INSERT INTO users (
                uuid, username, email, password_hash, role,
                is_active, is_verified, storage_quota_mb, storage_used_mb
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            RETURNING id, uuid, username, email, role, is_active, is_verified,
                      storage_quota_mb, storage_used_mb, created_at
        """, user_uuid, "testuser", "test@example.com", password_hash,
            "user", True, True, 5120, 0.0)

    return {
        "id": user["id"],
        "uuid": str(user["uuid"]),
        "username": user["username"],
        "email": user["email"],
        "role": user["role"],
        "is_active": user["is_active"],
        "is_verified": user["is_verified"],
        "storage_quota_mb": user["storage_quota_mb"],
        "storage_used_mb": user["storage_used_mb"],
        "created_at": user["created_at"],
        "password": password,
        "password_hash": password_hash
    }


@pytest_asyncio.fixture
async def admin_user(test_db_pool, password_service):
    """Create an admin test user in the database."""
    user_uuid = str(uuid.uuid4())
    password = "Admin@Pass#2024!"
    password_hash = password_service.hash_password(password)

    async with test_db_pool.acquire() as conn:
        user = await conn.fetchrow("""
            INSERT INTO users (
                uuid, username, email, password_hash, role,
                is_active, is_verified, storage_quota_mb, storage_used_mb
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            RETURNING id, uuid, username, email, role, is_active, is_verified,
                      storage_quota_mb, storage_used_mb, created_at
        """, user_uuid, "admin", "admin@example.com", password_hash,
            "admin", True, True, 10240, 0.0)

    return {
        "id": user["id"],
        "uuid": str(user["uuid"]),
        "username": user["username"],
        "email": user["email"],
        "role": user["role"],
        "is_active": user["is_active"],
        "is_verified": user["is_verified"],
        "storage_quota_mb": user["storage_quota_mb"],
        "storage_used_mb": user["storage_used_mb"],
        "created_at": user["created_at"],
        "password": password,
        "password_hash": password_hash
    }


@pytest_asyncio.fixture
async def inactive_user(test_db_pool, password_service):
    """Create an inactive test user in the database."""
    user_uuid = str(uuid.uuid4())
    password = "Inactive@Pass#2024!"
    password_hash = password_service.hash_password(password)

    async with test_db_pool.acquire() as conn:
        user = await conn.fetchrow("""
            INSERT INTO users (
                uuid, username, email, password_hash, role,
                is_active, is_verified, storage_quota_mb, storage_used_mb
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            RETURNING id, uuid, username, email, role, is_active, is_verified,
                      storage_quota_mb, storage_used_mb, created_at
        """, user_uuid, "inactiveuser", "inactive@example.com", password_hash,
            "user", False, True, 5120, 0.0)

    return {
        "id": user["id"],
        "uuid": str(user["uuid"]),
        "username": user["username"],
        "email": user["email"],
        "role": user["role"],
        "is_active": user["is_active"],
        "is_verified": user["is_verified"],
        "storage_quota_mb": user["storage_quota_mb"],
        "storage_used_mb": user["storage_used_mb"],
        "created_at": user["created_at"],
        "password": password,
        "password_hash": password_hash
    }


@pytest.fixture
def valid_access_token(jwt_service, test_user):
    """Create a valid access token for testing."""
    return jwt_service.create_access_token(
        user_id=test_user['id'],
        username=test_user['username'],
        role=test_user['role']
    )


@pytest.fixture
def valid_refresh_token(jwt_service, test_user):
    """Create a valid refresh token for testing."""
    return jwt_service.create_refresh_token(
        user_id=test_user['id'],
        username=test_user['username']
    )


@pytest.fixture
def expired_access_token(jwt_service, test_user):
    """Create an expired access token for testing."""
    # Temporarily override expiry
    original_expire = jwt_service.settings.ACCESS_TOKEN_EXPIRE_MINUTES
    jwt_service.settings.ACCESS_TOKEN_EXPIRE_MINUTES = -1  # Expired
    token = jwt_service.create_access_token(
        user_id=test_user['id'],
        username=test_user['username'],
        role=test_user['role']
    )
    jwt_service.settings.ACCESS_TOKEN_EXPIRE_MINUTES = original_expire
    return token


@pytest.fixture
def auth_headers(valid_access_token):
    """Create authorization headers with valid token."""
    return {"Authorization": f"Bearer {valid_access_token}"}


@pytest.fixture
def api_key_headers():
    """Create API key headers for single-user mode."""
    return {"X-API-KEY": settings.get("SINGLE_USER_API_KEY", "test-api-key")}


@pytest.fixture(autouse=True)
def clear_app_overrides():
    """Clear FastAPI app dependency overrides after each test."""
    yield
    from tldw_Server_API.app.main import app
    app.dependency_overrides.clear()
