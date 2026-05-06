# migrations.py
# Description: Database migrations for AuthNZ module tables
#
# Imports
import contextlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

#
# 3rd-party imports
from loguru import logger

#
# Local imports
from tldw_Server_API.app.core.AuthNZ.admin_webhook_secrets import encrypt_admin_webhook_secret
from tldw_Server_API.app.core.DB_Management.migrations import Migration, MigrationManager
from tldw_Server_API.app.core.Infrastructure.distributed_lock import acquire_migration_lock
from tldw_Server_API.app.core.testing import is_test_mode as _is_test_mode
from tldw_Server_API.app.core.testing import is_truthy as _is_truthy

_AUTHNZ_MIGRATIONS_NONCRITICAL_EXCEPTIONS = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    AttributeError,
    ConnectionError,
    TimeoutError,
    sqlite3.Error,
    json.JSONDecodeError,
)


def _sqlite_table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    """Return True when the given SQLite table already exists."""
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None

#######################################################################################################################
#
# AuthNZ Migrations
#

def migration_001_create_users_table(conn: sqlite3.Connection) -> None:
    """Create the users table for authentication"""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            is_active INTEGER DEFAULT 1,
            is_superuser INTEGER DEFAULT 0,
            role TEXT DEFAULT 'user',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP,
            email_verified INTEGER DEFAULT 0,
            storage_quota_mb INTEGER DEFAULT 5120,
            storage_used_mb INTEGER DEFAULT 0
        )
    """)

    # Create indexes
    conn.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_users_role ON users(role)")

    conn.commit()
    logger.info("Migration 001: Created users table")


def migration_002_create_sessions_table(conn: sqlite3.Connection) -> None:
    """Create the sessions table for session management"""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            token_hash TEXT NOT NULL,
            refresh_token_hash TEXT,
            encrypted_token TEXT,
            encrypted_refresh TEXT,
            expires_at TIMESTAMP NOT NULL,
            refresh_expires_at TIMESTAMP,
            ip_address TEXT,
            user_agent TEXT,
            device_id TEXT,
            is_active INTEGER DEFAULT 1,
            is_revoked INTEGER DEFAULT 0,
            revoked_at TIMESTAMP,
            revoked_by INTEGER,
            revoke_reason TEXT,
            access_jti TEXT,
            refresh_jti TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)

    # Create indexes
    try:
        cursor = conn.execute("PRAGMA table_info(sessions)")
        columns = {row[1] for row in cursor.fetchall()}

        def add_col(name: str, decl: str):
            if name not in columns:
                conn.execute(f"ALTER TABLE sessions ADD COLUMN {decl}")
                columns.add(name)

        # Legacy schemas may lack critical columns; add safe defaults where possible
        add_col('token_hash', "token_hash TEXT")
        add_col('refresh_token_hash', "refresh_token_hash TEXT")
        add_col('encrypted_token', "encrypted_token TEXT")
        add_col('encrypted_refresh', "encrypted_refresh TEXT")
        add_col('refresh_expires_at', "refresh_expires_at TIMESTAMP")
        add_col('access_jti', "access_jti TEXT")
        add_col('refresh_jti', "refresh_jti TEXT")
        add_col('is_active', "is_active INTEGER DEFAULT 1")
        add_col('is_revoked', "is_revoked INTEGER DEFAULT 0")
        add_col('revoked_at', "revoked_at TIMESTAMP")
        add_col('revoked_by', "revoked_by INTEGER")
        add_col('revoke_reason', "revoke_reason TEXT")
    except _AUTHNZ_MIGRATIONS_NONCRITICAL_EXCEPTIONS:
        pass

    conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_token_hash ON sessions(token_hash)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON sessions(expires_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_access_jti ON sessions(access_jti)")

    conn.commit()
    logger.info("Migration 002: Created sessions table")


def migration_003_create_api_keys_table(conn: sqlite3.Connection) -> None:
    """Create the api_keys table for API key management"""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS api_keys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            key_hash TEXT UNIQUE NOT NULL,
            key_id TEXT,
            key_prefix TEXT,
            name TEXT,
            description TEXT,
            scope TEXT,
            status TEXT DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP,
            last_used_at TIMESTAMP,
            last_used_ip TEXT,
            usage_count INTEGER DEFAULT 0,
            rate_limit INTEGER,
            allowed_ips TEXT,
            metadata TEXT,
            rotated_from INTEGER REFERENCES api_keys(id),
            rotated_to INTEGER REFERENCES api_keys(id),
            revoked_at TIMESTAMP,
            revoked_by INTEGER,
            revoke_reason TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)

    # Harmonize legacy schemas: add missing columns if needed
    try:
        cur = conn.execute("PRAGMA table_info(api_keys)")
        cols = {row[1] for row in cur.fetchall()}

        def add_col(name: str, decl: str):
            if name not in cols:
                conn.execute(f"ALTER TABLE api_keys ADD COLUMN {decl}")
                cols.add(name)

        add_col('key_id', "key_id TEXT")
        add_col('key_prefix', "key_prefix TEXT")
        add_col('description', "description TEXT")
        add_col('scope', "scope TEXT")
        add_col('status', "status TEXT DEFAULT 'active'")
        add_col('last_used_at', "last_used_at TIMESTAMP")
        add_col('last_used_ip', "last_used_ip TEXT")
        add_col('usage_count', "usage_count INTEGER DEFAULT 0")
        add_col('rate_limit', "rate_limit INTEGER")
        add_col('allowed_ips', "allowed_ips TEXT")
        add_col('metadata', "metadata TEXT")
        add_col('rotated_from', "rotated_from INTEGER REFERENCES api_keys(id)")
        add_col('rotated_to', "rotated_to INTEGER REFERENCES api_keys(id)")
        add_col('revoked_at', "revoked_at TIMESTAMP")
        add_col('revoked_by', "revoked_by INTEGER")
        add_col('revoke_reason', "revoke_reason TEXT")
    except _AUTHNZ_MIGRATIONS_NONCRITICAL_EXCEPTIONS:
        pass

    # Create indexes (only if columns exist)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_key_hash ON api_keys(key_hash)")
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_api_keys_key_id ON api_keys(key_id)")
    with contextlib.suppress(_AUTHNZ_MIGRATIONS_NONCRITICAL_EXCEPTIONS):
        conn.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_status ON api_keys(status)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_expires_at ON api_keys(expires_at)")

    conn.commit()
    logger.info("Migration 003: Created api_keys table")


def migration_004_create_api_key_audit_log(conn: sqlite3.Connection) -> None:
    """Create the api_key_audit_log table for tracking API key actions"""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS api_key_audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            api_key_id INTEGER NOT NULL,
            action TEXT NOT NULL,
            user_id INTEGER,
            ip_address TEXT,
            user_agent TEXT,
            details TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (api_key_id) REFERENCES api_keys(id) ON DELETE CASCADE
        )
    """)

    # Create index
    conn.execute("CREATE INDEX IF NOT EXISTS idx_api_key_audit_log_api_key_id ON api_key_audit_log(api_key_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_api_key_audit_log_created_at ON api_key_audit_log(created_at)")

    conn.commit()
    logger.info("Migration 004: Created api_key_audit_log table")


def migration_005_create_rate_limits_table(conn: sqlite3.Connection) -> None:
    """Create the rate_limits table for rate limiting"""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS rate_limits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            identifier TEXT NOT NULL,
            endpoint TEXT NOT NULL,
            request_count INTEGER DEFAULT 1,
            window_start TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(identifier, endpoint, window_start)
        )
    """)

    # Create indexes
    conn.execute("CREATE INDEX IF NOT EXISTS idx_rate_limits_identifier ON rate_limits(identifier)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_rate_limits_window_start ON rate_limits(window_start)")

    conn.commit()
    logger.info("Migration 005: Created rate_limits table")


def migration_006_create_registration_codes_table(conn: sqlite3.Connection) -> None:
    """Create the registration_codes table for controlled registration"""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS registration_codes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT UNIQUE NOT NULL,
            created_by INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP,
            max_uses INTEGER DEFAULT 1,
            uses_count INTEGER DEFAULT 0,
            is_active INTEGER DEFAULT 1,
            metadata TEXT,
            FOREIGN KEY (created_by) REFERENCES users(id)
        )
    """)

    # Create indexes
    conn.execute("CREATE INDEX IF NOT EXISTS idx_registration_codes_code ON registration_codes(code)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_registration_codes_expires_at ON registration_codes(expires_at)")

    conn.commit()
    logger.info("Migration 006: Created registration_codes table")


def migration_007_create_audit_logs_table(conn: sqlite3.Connection) -> None:
    """Create the audit_logs table for security auditing"""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            action TEXT NOT NULL,
            resource_type TEXT,
            resource_id INTEGER,
            ip_address TEXT,
            user_agent TEXT,
            status TEXT,
            details TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    # Create indexes
    conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit_logs(user_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_logs_action ON audit_logs(action)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at ON audit_logs(created_at)")

    conn.commit()
    logger.info("Migration 007: Created audit_logs table")


def migration_008_add_password_history_table(conn: sqlite3.Connection) -> None:
    """Create the password_history table to prevent password reuse"""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS password_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)

    # Create indexes
    conn.execute("CREATE INDEX IF NOT EXISTS idx_password_history_user_id ON password_history(user_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_password_history_created_at ON password_history(created_at)")

    conn.commit()
    logger.info("Migration 008: Created password_history table")


def migration_009_add_session_encryption_columns(conn: sqlite3.Connection) -> None:
    """Add encryption columns to sessions table if they don't exist"""
    # Check if columns already exist
    cursor = conn.execute("PRAGMA table_info(sessions)")
    columns = [row[1] for row in cursor.fetchall()]

    if 'encrypted_token' not in columns:
        conn.execute("ALTER TABLE sessions ADD COLUMN encrypted_token TEXT")
        logger.info("Added encrypted_token column to sessions table")

    if 'encrypted_refresh' not in columns:
        conn.execute("ALTER TABLE sessions ADD COLUMN encrypted_refresh TEXT")
        logger.info("Added encrypted_refresh column to sessions table")

    conn.commit()
    logger.info("Migration 009: Added session encryption columns")


def migration_010_add_2fa_columns(conn: sqlite3.Connection) -> None:
    """Add two-factor authentication columns to users table"""
    # Check if columns already exist
    cursor = conn.execute("PRAGMA table_info(users)")
    columns = [row[1] for row in cursor.fetchall()]

    if 'totp_secret' not in columns:
        conn.execute("ALTER TABLE users ADD COLUMN totp_secret TEXT")
        logger.info("Added totp_secret column to users table")

    if 'two_factor_enabled' not in columns:
        conn.execute("ALTER TABLE users ADD COLUMN two_factor_enabled INTEGER DEFAULT 0")
        logger.info("Added two_factor_enabled column to users table")

    if 'backup_codes' not in columns:
        conn.execute("ALTER TABLE users ADD COLUMN backup_codes TEXT")
        logger.info("Added backup_codes column to users table")

    conn.commit()
    logger.info("Migration 010: Added 2FA columns to users table")


# Rollback functions (for migrations that can be rolled back)

def rollback_001_drop_users_table(conn: sqlite3.Connection) -> None:
    """Rollback: Drop the users table"""
    conn.execute("DROP TABLE IF EXISTS users")
    conn.commit()
    logger.info("Rollback 001: Dropped users table")


def rollback_002_drop_sessions_table(conn: sqlite3.Connection) -> None:
    """Rollback: Drop the sessions table"""
    conn.execute("DROP TABLE IF EXISTS sessions")
    conn.commit()
    logger.info("Rollback 002: Dropped sessions table")


def rollback_003_drop_api_keys_table(conn: sqlite3.Connection) -> None:
    """Rollback: Drop the api_keys table"""
    conn.execute("DROP TABLE IF EXISTS api_keys")
    conn.commit()
    logger.info("Rollback 003: Dropped api_keys table")


def migration_011_add_enhanced_auth_tables(conn: sqlite3.Connection) -> None:
    """Create tables for enhanced authentication features"""
    logger.info("Migration 011: START enhanced auth tables + uuid")

    # Password reset tokens table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS password_reset_tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            token_hash TEXT UNIQUE,
            expires_at TIMESTAMP NOT NULL,
            used_at TIMESTAMP,
            ip_address TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)
    # Harmonize legacy column name 'token' -> ensure 'token_hash' exists
    try:
        cur = conn.execute("PRAGMA table_info(password_reset_tokens)")
        cols = {row[1] for row in cur.fetchall()}
        if 'token_hash' not in cols:
            conn.execute("ALTER TABLE password_reset_tokens ADD COLUMN token_hash TEXT UNIQUE")
    except _AUTHNZ_MIGRATIONS_NONCRITICAL_EXCEPTIONS:
        pass
    # Indexes
    with contextlib.suppress(_AUTHNZ_MIGRATIONS_NONCRITICAL_EXCEPTIONS):
        conn.execute("CREATE INDEX IF NOT EXISTS idx_reset_tokens_hash ON password_reset_tokens(token_hash)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_reset_tokens_user ON password_reset_tokens(user_id)")

    # Failed attempts table for lockout tracking
    conn.execute("""
        CREATE TABLE IF NOT EXISTS failed_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            identifier TEXT NOT NULL,
            attempt_type TEXT NOT NULL,
            attempt_count INTEGER DEFAULT 1,
            window_start TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(identifier, attempt_type)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_failed_attempts_identifier ON failed_attempts(identifier)")

    # Account lockouts table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS account_lockouts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            identifier TEXT UNIQUE NOT NULL,
            locked_until TIMESTAMP NOT NULL,
            reason TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_lockouts_identifier ON account_lockouts(identifier)")

    # Token blacklist table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS token_blacklist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            jti TEXT UNIQUE NOT NULL,
            user_id INTEGER,
            token_type TEXT,
            revoked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP NOT NULL,
            reason TEXT,
            revoked_by INTEGER,
            ip_address TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_blacklist_jti ON token_blacklist(jti)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_blacklist_expires ON token_blacklist(expires_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_blacklist_user ON token_blacklist(user_id)")

    # Add columns to users if not exists
    cursor = conn.execute("PRAGMA table_info(users)")
    columns = [row[1] for row in cursor.fetchall()]

    if 'uuid' not in columns:
        # SQLite cannot add a UNIQUE constraint via ALTER TABLE. Add column first,
        # then create a unique index to enforce uniqueness. This keeps the
        # migration compatible with fresh DBs and legacy ones.
        conn.execute("ALTER TABLE users ADD COLUMN uuid TEXT")
        try:
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_uuid ON users(uuid)")
        except _AUTHNZ_MIGRATIONS_NONCRITICAL_EXCEPTIONS:
            # If an older schema already enforced uniqueness differently, ignore
            pass
        logger.info("Added uuid column to users table with unique index")

    if 'email_verified_at' not in columns:
        conn.execute("ALTER TABLE users ADD COLUMN email_verified_at TIMESTAMP")
        logger.info("Added email_verified_at column to users table")

    if 'is_verified' not in columns:
        conn.execute("ALTER TABLE users ADD COLUMN is_verified INTEGER DEFAULT 0")
        logger.info("Added is_verified column to users table")

    conn.commit()
    logger.info("Migration 011: Created enhanced authentication tables")


def migration_084_scope_account_lockouts_by_attempt_type(conn: sqlite3.Connection) -> None:
    """Rebuild account_lockouts so lockouts are scoped by attempt_type."""
    logger.info("Migration 084: START scope account_lockouts by attempt_type")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS account_lockouts_new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            identifier TEXT NOT NULL,
            attempt_type TEXT NOT NULL,
            locked_until TIMESTAMP NOT NULL,
            reason TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(identifier, attempt_type)
        )
        """
    )
    conn.execute(
        """
        INSERT OR IGNORE INTO account_lockouts_new
            (identifier, attempt_type, locked_until, reason, created_at)
        SELECT
            identifier,
            'login',
            locked_until,
            reason,
            COALESCE(created_at, CURRENT_TIMESTAMP)
        FROM account_lockouts
        """
    )
    conn.execute("DROP TABLE IF EXISTS account_lockouts")
    conn.execute("ALTER TABLE account_lockouts_new RENAME TO account_lockouts")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_lockouts_identifier_attempt_type "
        "ON account_lockouts(identifier, attempt_type)"
    )
    conn.commit()
    logger.info("Migration 084: COMPLETE scope account_lockouts by attempt_type")


def migration_085_remove_api_keys_scope_default(conn: sqlite3.Connection) -> None:
    """Rebuild api_keys so omitted scope no longer defaults to read."""
    logger.info("Migration 085: START remove api_keys.scope default")

    pragma_rows = conn.execute("PRAGMA table_info(api_keys)").fetchall()
    if not pragma_rows:
        logger.info("Migration 085: api_keys table not present; skipping")
        return

    desired_defs = {
        "id": "id INTEGER PRIMARY KEY AUTOINCREMENT",
        "user_id": "user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE",
        "key_hash": "key_hash TEXT UNIQUE NOT NULL",
        "key_id": "key_id TEXT",
        "key_prefix": "key_prefix TEXT",
        "name": "name TEXT",
        "description": "description TEXT",
        "scope": "scope TEXT",
        "status": "status TEXT DEFAULT 'active'",
        "created_at": "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
        "expires_at": "expires_at TIMESTAMP",
        "last_used_at": "last_used_at TIMESTAMP",
        "last_used_ip": "last_used_ip TEXT",
        "usage_count": "usage_count INTEGER DEFAULT 0",
        "rate_limit": "rate_limit INTEGER",
        "allowed_ips": "allowed_ips TEXT",
        "metadata": "metadata TEXT",
        "rotated_from": "rotated_from INTEGER REFERENCES api_keys(id)",
        "rotated_to": "rotated_to INTEGER REFERENCES api_keys(id)",
        "revoked_at": "revoked_at TIMESTAMP",
        "revoked_by": "revoked_by INTEGER",
        "revoke_reason": "revoke_reason TEXT",
        "is_virtual": "is_virtual INTEGER DEFAULT 0",
        "parent_key_id": "parent_key_id INTEGER REFERENCES api_keys(id)",
        "org_id": "org_id INTEGER REFERENCES organizations(id) ON DELETE SET NULL",
        "team_id": "team_id INTEGER REFERENCES teams(id) ON DELETE SET NULL",
        "llm_budget_day_tokens": "llm_budget_day_tokens INTEGER",
        "llm_budget_month_tokens": "llm_budget_month_tokens INTEGER",
        "llm_budget_day_usd": "llm_budget_day_usd REAL",
        "llm_budget_month_usd": "llm_budget_month_usd REAL",
        "llm_allowed_endpoints": "llm_allowed_endpoints TEXT",
        "llm_allowed_providers": "llm_allowed_providers TEXT",
        "llm_allowed_models": "llm_allowed_models TEXT",
    }

    pragma_by_name = {row[1]: row for row in pragma_rows}
    current_columns = [row[1] for row in pragma_rows]

    def _fallback_definition(column_name: str) -> str:
        row = pragma_by_name[column_name]
        column_type = row[2] or "TEXT"
        not_null = bool(row[3])
        default = row[4]
        primary_key = bool(row[5])

        parts = [column_name, column_type]
        if primary_key:
            parts.append("PRIMARY KEY")
        elif not_null:
            parts.append("NOT NULL")
        if default is not None and column_name != "scope":
            parts.append(f"DEFAULT {default}")
        return " ".join(parts)

    column_defs = [desired_defs.get(name, _fallback_definition(name)) for name in current_columns]

    # Use the foreign-key-off rebuild pattern to avoid FK cascade
    # actions (e.g. ON DELETE CASCADE on api_key_audit_log, ON DELETE
    # SET NULL on usage_log / llm_usage_log) that SQLite would otherwise
    # trigger when the live api_keys table is dropped.
    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS api_keys_new (
                {', '.join(column_defs)}
            )
            """
        )
        conn.execute(
            f"""
            INSERT INTO api_keys_new ({', '.join(current_columns)})
            SELECT {', '.join(current_columns)}
            FROM api_keys
            """
        )
        conn.execute("DROP TABLE IF EXISTS api_keys")
        conn.execute("ALTER TABLE api_keys_new RENAME TO api_keys")

        conn.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_key_hash ON api_keys(key_hash)")
        if "key_id" in current_columns:
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_api_keys_key_id ON api_keys(key_id)")
        if "status" in current_columns:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_status ON api_keys(status)")
        if "expires_at" in current_columns:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_expires_at ON api_keys(expires_at)")
        if "is_virtual" in current_columns:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_virtual ON api_keys(is_virtual)")
        if "org_id" in current_columns:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_org ON api_keys(org_id)")
        if "team_id" in current_columns:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_team ON api_keys(team_id)")
    finally:
        conn.execute("PRAGMA foreign_keys = ON")

    conn.commit()
    logger.info("Migration 085: COMPLETE remove api_keys.scope default")


def migration_086_create_prototype_workspace_tables(conn: sqlite3.Connection) -> None:
    """Create shared metadata tables for prototype workspaces and collaboration sessions."""
    logger.info("Migration 086: START prototype workspace metadata tables")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS prototype_workspaces (
            id TEXT PRIMARY KEY,
            owner_user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            description TEXT,
            creation_source TEXT NOT NULL
                CHECK (creation_source IN ('prompt', 'template', 'existing_workspace')),
            canonical_snapshot_id TEXT,
            last_known_good_snapshot_id TEXT,
            canonical_preview_status TEXT,
            publish_validation_status TEXT,
            preview_policy_json TEXT NOT NULL DEFAULT '{}',
            share_policy_json TEXT NOT NULL DEFAULT '{}',
            runtime_policy_json TEXT NOT NULL DEFAULT '{}',
            designated_promoter_ids_json TEXT NOT NULL DEFAULT '[]',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            archived_at TIMESTAMP,
            FOREIGN KEY (owner_user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_prototype_workspaces_owner_updated "
        "ON prototype_workspaces(owner_user_id, updated_at)"
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS prototype_snapshots (
            id TEXT PRIMARY KEY,
            prototype_workspace_id TEXT NOT NULL,
            parent_snapshot_id TEXT,
            created_from_session_id TEXT,
            author_user_id INTEGER,
            author_shared_actor_id TEXT,
            storage_ref TEXT,
            diff_summary_json TEXT NOT NULL DEFAULT '{}',
            prompt_summary TEXT,
            preview_health_json TEXT NOT NULL DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (prototype_workspace_id) REFERENCES prototype_workspaces(id) ON DELETE CASCADE,
            UNIQUE (id, prototype_workspace_id),
            CHECK (
                (author_user_id IS NOT NULL AND author_shared_actor_id IS NULL)
                OR (author_user_id IS NULL AND author_shared_actor_id IS NOT NULL)
            )
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_prototype_snapshots_workspace_created "
        "ON prototype_snapshots(prototype_workspace_id, created_at)"
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS prototype_shared_actors (
            id TEXT PRIMARY KEY,
            prototype_workspace_id TEXT NOT NULL,
            share_link_id INTEGER NOT NULL,
            display_name TEXT NOT NULL,
            session_binding_id TEXT,
            runtime_policy_profile TEXT NOT NULL,
            quota_policy_json TEXT NOT NULL DEFAULT '{}',
            last_activity_at TIMESTAMP,
            expires_at TIMESTAMP,
            revoked_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (prototype_workspace_id) REFERENCES prototype_workspaces(id) ON DELETE CASCADE,
            UNIQUE (id, prototype_workspace_id)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_prototype_shared_actors_workspace "
        "ON prototype_shared_actors(prototype_workspace_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_prototype_shared_actors_share_link_revoked "
        "ON prototype_shared_actors(share_link_id, revoked_at)"
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS prototype_sessions (
            id TEXT PRIMARY KEY,
            prototype_workspace_id TEXT NOT NULL,
            base_snapshot_id TEXT NOT NULL,
            actor_user_id INTEGER,
            actor_shared_actor_id TEXT,
            actor_type TEXT NOT NULL
                CHECK (actor_type IN ('owner', 'internal_collaborator', 'external_collaborator')),
            share_link_id INTEGER,
            acp_session_id TEXT,
            sandbox_session_id TEXT,
            sandbox_run_id TEXT,
            runtime_status TEXT,
            preview_handle TEXT,
            preview_status TEXT,
            last_saved_snapshot_id TEXT,
            last_activity_at TIMESTAMP,
            expires_at TIMESTAMP,
            revoked_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (prototype_workspace_id) REFERENCES prototype_workspaces(id) ON DELETE CASCADE,
            FOREIGN KEY (actor_user_id) REFERENCES users(id),
            FOREIGN KEY (base_snapshot_id, prototype_workspace_id)
                REFERENCES prototype_snapshots(id, prototype_workspace_id),
            FOREIGN KEY (actor_shared_actor_id, prototype_workspace_id)
                REFERENCES prototype_shared_actors(id, prototype_workspace_id),
            UNIQUE (id, prototype_workspace_id),
            CHECK (
                (actor_type = 'external_collaborator'
                    AND actor_user_id IS NULL
                    AND actor_shared_actor_id IS NOT NULL)
                OR (
                    actor_type IN ('owner', 'internal_collaborator')
                    AND actor_user_id IS NOT NULL
                    AND actor_shared_actor_id IS NULL
                )
            )
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_prototype_sessions_workspace_updated "
        "ON prototype_sessions(prototype_workspace_id, updated_at)"
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS prototype_preview_handles (
            id TEXT PRIMARY KEY,
            preview_scope TEXT NOT NULL
                CHECK (preview_scope IN ('canonical', 'session')),
            scope_id TEXT NOT NULL,
            prototype_workspace_id TEXT NOT NULL,
            prototype_session_id TEXT,
            actor_key TEXT NOT NULL,
            target_ref TEXT NOT NULL,
            runtime_policy_profile TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            is_active INTEGER NOT NULL DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            revoked_at TIMESTAMP,
            FOREIGN KEY (prototype_workspace_id) REFERENCES prototype_workspaces(id) ON DELETE CASCADE,
            FOREIGN KEY (prototype_session_id, prototype_workspace_id)
                REFERENCES prototype_sessions(id, prototype_workspace_id)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_prototype_preview_handles_workspace "
        "ON prototype_preview_handles(prototype_workspace_id, created_at)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_prototype_preview_handles_session "
        "ON prototype_preview_handles(prototype_session_id, created_at)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_prototype_preview_handles_scope_active "
        "ON prototype_preview_handles(scope_id, is_active)"
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_prototype_preview_handles_active_scope "
        "ON prototype_preview_handles(scope_id) WHERE is_active = 1"
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS prototype_promotion_requests (
            id TEXT PRIMARY KEY,
            prototype_workspace_id TEXT NOT NULL,
            prototype_session_id TEXT NOT NULL,
            candidate_snapshot_id TEXT NOT NULL,
            requested_by_user_id INTEGER,
            requested_by_shared_actor_id TEXT,
            status TEXT NOT NULL
                CHECK (status IN ('pending', 'approved', 'rejected', 'promoted', 'stale')),
            reviewed_by_user_id INTEGER,
            review_notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (prototype_workspace_id) REFERENCES prototype_workspaces(id) ON DELETE CASCADE,
            FOREIGN KEY (prototype_session_id) REFERENCES prototype_sessions(id) ON DELETE CASCADE,
            FOREIGN KEY (candidate_snapshot_id, prototype_workspace_id)
                REFERENCES prototype_snapshots(id, prototype_workspace_id),
            FOREIGN KEY (prototype_session_id, prototype_workspace_id)
                REFERENCES prototype_sessions(id, prototype_workspace_id),
            CHECK (
                (requested_by_user_id IS NOT NULL AND requested_by_shared_actor_id IS NULL)
                OR (requested_by_user_id IS NULL AND requested_by_shared_actor_id IS NOT NULL)
            )
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_prototype_promotion_requests_workspace_status "
        "ON prototype_promotion_requests(prototype_workspace_id, status)"
    )

    conn.commit()
    logger.info("Migration 086: Created prototype workspace metadata tables")


def migration_087_expand_share_tokens_resource_type_for_prototypes(conn: sqlite3.Connection) -> None:
    """Rebuild share_tokens to support prototype_workspace and backfill legacy encoded rows."""
    logger.info("Migration 087: START expand share_tokens resource_type for prototypes")

    if not _sqlite_table_exists(conn, "share_tokens"):
        logger.info("Migration 087: share_tokens table missing; skipping")
        return

    table_sql_row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'share_tokens'"
    ).fetchone()
    table_sql = str(table_sql_row[0] if table_sql_row else "").lower()
    has_prototype_type = "prototype_workspace" in table_sql
    legacy_count_row = conn.execute(
        """
        SELECT COUNT(*) FROM share_tokens
        WHERE resource_type = 'workspace'
          AND resource_id LIKE 'prototype_workspace::%'
        """
    ).fetchone()
    legacy_count = int(legacy_count_row[0]) if legacy_count_row else 0

    conn.execute("DROP TABLE IF EXISTS share_tokens_new")

    if has_prototype_type and legacy_count == 0:
        logger.info("Migration 087: share_tokens already supports prototype_workspace; skipping")
        return

    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS share_tokens_new (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                token_hash      TEXT UNIQUE NOT NULL,
                token_prefix    TEXT NOT NULL,
                resource_type   TEXT NOT NULL
                    CHECK (resource_type IN ('chatbook', 'workspace', 'prototype_workspace')),
                resource_id     TEXT NOT NULL,
                owner_user_id   INTEGER NOT NULL,
                access_level    TEXT NOT NULL DEFAULT 'view_chat',
                allow_clone     INTEGER NOT NULL DEFAULT 1,
                password_hash   TEXT,
                max_uses        INTEGER,
                use_count       INTEGER NOT NULL DEFAULT 0,
                expires_at      TIMESTAMP,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                revoked_at      TIMESTAMP,
                FOREIGN KEY (owner_user_id) REFERENCES users(id)
            )
            """
        )

        conn.execute(
            """
            INSERT INTO share_tokens_new (
                id, token_hash, token_prefix, resource_type, resource_id,
                owner_user_id, access_level, allow_clone, password_hash,
                max_uses, use_count, expires_at, created_at, revoked_at
            )
            SELECT
                id,
                token_hash,
                token_prefix,
                CASE
                    WHEN resource_type = 'workspace'
                         AND resource_id LIKE 'prototype_workspace::%'
                    THEN 'prototype_workspace'
                    ELSE resource_type
                END AS resource_type,
                CASE
                    WHEN resource_type = 'workspace'
                         AND resource_id LIKE 'prototype_workspace::%'
                    THEN substr(resource_id, 22)
                    ELSE resource_id
                END AS resource_id,
                owner_user_id,
                access_level,
                allow_clone,
                password_hash,
                max_uses,
                use_count,
                expires_at,
                created_at,
                revoked_at
            FROM share_tokens
            """
        )

        conn.execute("DROP TABLE IF EXISTS share_tokens")
        conn.execute("ALTER TABLE share_tokens_new RENAME TO share_tokens")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_share_tokens_prefix ON share_tokens(token_prefix)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_share_tokens_owner ON share_tokens(owner_user_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_share_tokens_resource ON share_tokens(resource_type, resource_id)")
    finally:
        conn.execute("PRAGMA foreign_keys = ON")

    conn.commit()
    logger.info(
        "Migration 087: Expanded share_tokens resource_type and backfilled {} legacy prototype rows",
        legacy_count,
    )


def rollback_086_drop_prototype_workspace_tables(conn: sqlite3.Connection) -> None:
    """Rollback migration 086 by dropping prototype workspace metadata tables."""
    conn.execute("DROP TABLE IF EXISTS prototype_promotion_requests")
    conn.execute("DROP TABLE IF EXISTS prototype_preview_handles")
    conn.execute("DROP TABLE IF EXISTS prototype_sessions")
    conn.execute("DROP TABLE IF EXISTS prototype_shared_actors")
    conn.execute("DROP TABLE IF EXISTS prototype_snapshots")
    conn.execute("DROP TABLE IF EXISTS prototype_workspaces")
    conn.commit()
    logger.info("Rollback 086: Dropped prototype workspace metadata tables")


def migration_012_create_rbac_tables(conn: sqlite3.Connection) -> None:
    """Create core RBAC tables: roles, permissions, mappings, and user overrides."""
    logger.info("Migration 012: START RBAC core tables")
    # Roles
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS roles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            is_system INTEGER DEFAULT 0
        )
        """
    )

    # Permissions (use column name 'name' to match existing DB accessors)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS permissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            category TEXT
        )
        """
    )

    # Role -> Permission mapping
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS role_permissions (
            role_id INTEGER NOT NULL,
            permission_id INTEGER NOT NULL,
            PRIMARY KEY (role_id, permission_id),
            FOREIGN KEY (role_id) REFERENCES roles(id) ON DELETE CASCADE,
            FOREIGN KEY (permission_id) REFERENCES permissions(id) ON DELETE CASCADE
        )
        """
    )

    # User -> Role mapping (with optional expiration)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS user_roles (
            user_id INTEGER NOT NULL,
            role_id INTEGER NOT NULL,
            granted_by INTEGER,
            expires_at TIMESTAMP,
            PRIMARY KEY (user_id, role_id),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY (role_id) REFERENCES roles(id) ON DELETE CASCADE
        )
        """
    )

    # User permission overrides (allow/deny via boolean 'granted')
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS user_permissions (
            user_id INTEGER NOT NULL,
            permission_id INTEGER NOT NULL,
            granted INTEGER NOT NULL DEFAULT 1, -- 1 = allow, 0 = deny
            expires_at TIMESTAMP,
            PRIMARY KEY (user_id, permission_id),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY (permission_id) REFERENCES permissions(id) ON DELETE CASCADE
        )
        """
    )

    # Helpful indexes
    conn.execute("CREATE INDEX IF NOT EXISTS idx_user_roles_user ON user_roles(user_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_user_roles_role ON user_roles(role_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_role_permissions_role ON role_permissions(role_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_user_permissions_user ON user_permissions(user_id)")

    conn.commit()
    logger.info("Migration 012: Created RBAC core tables (roles, permissions, mappings, overrides)")


def migration_013_create_rbac_limits_and_usage(conn: sqlite3.Connection) -> None:
    """Create optional RBAC rate limit and usage tables (SQLite)."""
    logger.info("Migration 013: START RBAC limits + usage tables")
    # Role-level rate limits
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS rbac_role_rate_limits (
            role_id INTEGER NOT NULL,
            resource TEXT NOT NULL,
            limit_per_min INTEGER,
            burst INTEGER,
            PRIMARY KEY (role_id, resource),
            FOREIGN KEY (role_id) REFERENCES roles(id) ON DELETE CASCADE
        )
        """
    )

    # User-level rate limits (override role defaults)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS rbac_user_rate_limits (
            user_id INTEGER NOT NULL,
            resource TEXT NOT NULL,
            limit_per_min INTEGER,
            burst INTEGER,
            PRIMARY KEY (user_id, resource),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """
    )

    # Lightweight per-request usage (for API analytics)
    # In pytest/TEST_MODE, relax foreign keys to simplify isolated table tests.
    try:
        import os as _os
        _relax_fk = (
            _is_truthy(_os.getenv("DISABLE_USAGE_FOREIGN_KEYS", ""))
            or _is_test_mode()
            or _os.getenv("PYTEST_CURRENT_TEST") is not None
        )
    except _AUTHNZ_MIGRATIONS_NONCRITICAL_EXCEPTIONS:
        _relax_fk = False

    if _relax_fk:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS usage_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                user_id INTEGER,
                key_id INTEGER,
                endpoint TEXT,
                status INTEGER,
                latency_ms INTEGER,
                bytes INTEGER,
                bytes_in INTEGER,
                meta TEXT
            )
            """
        )
    else:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS usage_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                user_id INTEGER,
                key_id INTEGER,
                endpoint TEXT,
                status INTEGER,
                latency_ms INTEGER,
                bytes INTEGER,
                bytes_in INTEGER,
                meta TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL,
                FOREIGN KEY (key_id) REFERENCES api_keys(id) ON DELETE SET NULL
            )
            """
        )

    # Daily aggregate for reporting
    if _relax_fk:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS usage_daily (
                user_id INTEGER NOT NULL,
                day DATE NOT NULL,
                requests INTEGER DEFAULT 0,
                errors INTEGER DEFAULT 0,
                bytes_total INTEGER DEFAULT 0,
                latency_avg_ms REAL,
                PRIMARY KEY (user_id, day)
            )
            """
        )
    else:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS usage_daily (
                user_id INTEGER NOT NULL,
                day DATE NOT NULL,
                requests INTEGER DEFAULT 0,
                errors INTEGER DEFAULT 0,
                bytes_total INTEGER DEFAULT 0,
                latency_avg_ms REAL,
                PRIMARY KEY (user_id, day),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """
        )

    conn.commit()
    logger.info("Migration 013: Created RBAC rate limit and usage tables")


def migration_014_seed_roles_permissions(conn: sqlite3.Connection) -> None:
    """Seed default roles and a baseline permission catalog."""
    logger.info("Migration 014: START seed default roles + permissions")
    # Seed roles
    conn.execute(
        """
        INSERT OR IGNORE INTO roles (name, description, is_system)
        VALUES
          ('admin', 'Administrator (full access)', 1),
          ('user', 'Standard user (baseline permissions)', 1),
          ('moderator', 'Moderator (curated elevated access)', 1),
          ('reviewer', 'Claims reviewer', 1)
        """
    )

    # Seed permissions (align with AuthNZ/permissions.py constants; use name column)
    perms = [
        # media
        ('media.create','Create media','media'),
        ('media.read','Read media','media'),
        ('media.update','Update media','media'),
        ('media.delete','Delete media','media'),
        ('media.transcribe','Transcribe audio/video','media'),
        ('media.export','Export media','media'),
        # users
        ('users.create','Create users','users'),
        ('users.read','Read users','users'),
        ('users.update','Update users','users'),
        ('users.delete','Delete users','users'),
        ('users.manage_roles','Manage user roles','users'),
        ('users.invite','Invite users','users'),
        # system
        ('system.configure','Configure system','system'),
        ('system.backup','Backup system','system'),
        ('system.export','Export system data','system'),
        ('system.logs','View system logs','system'),
        ('system.maintenance','Maintenance operations','system'),
        # api
        ('api.generate_keys','Generate API keys','api'),
        ('api.manage_webhooks','Manage webhooks','api'),
        ('api.rate_limit_override','Override rate limits','api'),
        # claims
        ('claims.review','Review claims','claims'),
        ('claims.admin','Administer claims','claims')
    ]
    for name, description, category in perms:
        conn.execute(
            "INSERT OR IGNORE INTO permissions (name, description, category) VALUES (?, ?, ?)",
            (name, description, category),
        )

    # Helper: get id by name
    def _id(table: str, key: str) -> int:
        lookup_sql_template = "SELECT id FROM {table} WHERE name = ?"
        lookup_sql = lookup_sql_template.format_map(locals())  # nosec B608
        cur = conn.execute(lookup_sql, (key,))
        row = cur.fetchone()
        return row[0] if row else None

    admin_id = _id('roles', 'admin')
    user_id = _id('roles', 'user')
    mod_id = _id('roles', 'moderator')
    reviewer_id = _id('roles', 'reviewer')

    # Grant all permissions to admin
    cur = conn.execute("SELECT id FROM permissions")
    for (perm_id,) in cur.fetchall():
        if admin_id is not None:
            conn.execute(
                "INSERT OR IGNORE INTO role_permissions (role_id, permission_id) VALUES (?, ?)",
                (admin_id, perm_id),
            )

    # Baseline user permissions
    baseline = [
        'media.create','media.read','media.update','media.transcribe',
        'users.read'
    ]
    for code in baseline:
        pid = _id('permissions', code)
        if pid is not None and user_id is not None:
            conn.execute(
                "INSERT OR IGNORE INTO role_permissions (role_id, permission_id) VALUES (?, ?)",
                (user_id, pid),
            )

    # Moderator: baseline + delete and some users.manage_roles
    mod_extra = ['media.delete', 'users.manage_roles']
    for code in set(baseline + mod_extra):
        pid = _id('permissions', code)
        if pid is not None and mod_id is not None:
            conn.execute(
                "INSERT OR IGNORE INTO role_permissions (role_id, permission_id) VALUES (?, ?)",
                (mod_id, pid),
            )

    # Reviewer: read media + claims.review
    reviewer_perms = ['media.read', 'claims.review']
    for code in reviewer_perms:
        pid = _id('permissions', code)
        if pid is not None and reviewer_id is not None:
            conn.execute(
                "INSERT OR IGNORE INTO role_permissions (role_id, permission_id) VALUES (?, ?)",
                (reviewer_id, pid),
            )

    conn.commit()
    logger.info("Migration 014: Seeded default roles and permissions")


def migration_015_create_llm_usage_tables(conn: sqlite3.Connection) -> None:
    """Create llm_usage_log and llm_usage_daily tables (SQLite)."""
    logger.info("Migration 015: START LLM usage tables")
    # Per-request LLM usage log
    try:
        import os as _os
        _relax_fk = (
            _is_truthy(_os.getenv("DISABLE_USAGE_FOREIGN_KEYS", ""))
            or _is_test_mode()
            or _os.getenv("PYTEST_CURRENT_TEST") is not None
        )
    except _AUTHNZ_MIGRATIONS_NONCRITICAL_EXCEPTIONS:
        _relax_fk = False

    if _relax_fk:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_usage_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                user_id INTEGER,
                key_id INTEGER,
                endpoint TEXT,
                operation TEXT,
                provider TEXT,
                model TEXT,
                status INTEGER,
                latency_ms INTEGER,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                total_tokens INTEGER,
                prompt_cost_usd REAL,
                completion_cost_usd REAL,
                total_cost_usd REAL,
                currency TEXT DEFAULT 'USD',
                estimated INTEGER DEFAULT 0,
                request_id TEXT,
                remote_ip TEXT,
                user_agent TEXT,
                token_name TEXT,
                conversation_id TEXT
            )
            """
        )
    else:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_usage_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                user_id INTEGER,
                key_id INTEGER,
                endpoint TEXT,
                operation TEXT,
                provider TEXT,
                model TEXT,
                status INTEGER,
                latency_ms INTEGER,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                total_tokens INTEGER,
                prompt_cost_usd REAL,
                completion_cost_usd REAL,
                total_cost_usd REAL,
                currency TEXT DEFAULT 'USD',
                estimated INTEGER DEFAULT 0,
                request_id TEXT,
                remote_ip TEXT,
                user_agent TEXT,
                token_name TEXT,
                conversation_id TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL,
                FOREIGN KEY (key_id) REFERENCES api_keys(id) ON DELETE SET NULL
            )
            """
        )

    # Helpful indexes for common queries
    conn.execute("CREATE INDEX IF NOT EXISTS idx_llm_usage_log_ts ON llm_usage_log(ts)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_llm_usage_log_user ON llm_usage_log(user_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_llm_usage_log_provider_model ON llm_usage_log(provider, model)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_llm_usage_log_op_ts ON llm_usage_log(operation, ts)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_llm_usage_log_remote_ip_ts ON llm_usage_log(remote_ip, ts)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_llm_usage_log_token_name_ts ON llm_usage_log(token_name, ts)")

    # Daily aggregate table
    if _relax_fk:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_usage_daily (
                day DATE NOT NULL,
                user_id INTEGER NOT NULL,
                operation TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                requests INTEGER DEFAULT 0,
                errors INTEGER DEFAULT 0,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                total_cost_usd REAL DEFAULT 0.0,
                latency_avg_ms REAL,
                PRIMARY KEY (day, user_id, operation, provider, model)
            )
            """
        )
    else:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_usage_daily (
                day DATE NOT NULL,
                user_id INTEGER NOT NULL,
                operation TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                requests INTEGER DEFAULT 0,
                errors INTEGER DEFAULT 0,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                total_cost_usd REAL DEFAULT 0.0,
                latency_avg_ms REAL,
                PRIMARY KEY (day, user_id, operation, provider, model),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """
        )

    conn.commit()
    logger.info("Migration 015: Created LLM usage tables (llm_usage_log, llm_usage_daily)")


def migration_016_create_orgs_teams(conn: sqlite3.Connection) -> None:
    """Create Organizations/Teams hierarchy tables (SQLite)."""
    logger.info("Migration 016: START organizations/teams tables")
    # organizations
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS organizations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            uuid TEXT UNIQUE,
            name TEXT UNIQUE NOT NULL,
            slug TEXT UNIQUE,
            owner_user_id INTEGER,
            is_active INTEGER DEFAULT 1,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (owner_user_id) REFERENCES users(id) ON DELETE SET NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_orgs_owner ON organizations(owner_user_id)")

    # org_members
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS org_members (
            org_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            role TEXT DEFAULT 'member',
            status TEXT DEFAULT 'active',
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (org_id, user_id),
            FOREIGN KEY (org_id) REFERENCES organizations(id) ON DELETE CASCADE,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_org_members_user ON org_members(user_id)")

    # teams
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS teams (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            org_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            slug TEXT,
            description TEXT,
            is_active INTEGER DEFAULT 1,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (org_id, name),
            FOREIGN KEY (org_id) REFERENCES organizations(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_teams_org ON teams(org_id)")

    # team_members
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS team_members (
            team_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            role TEXT DEFAULT 'member',
            status TEXT DEFAULT 'active',
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (team_id, user_id),
            FOREIGN KEY (team_id) REFERENCES teams(id) ON DELETE CASCADE,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_team_members_user ON team_members(user_id)")

    conn.commit()
    logger.info("Migration 016: Created organizations, org_members, teams, team_members")


def migration_017_extend_api_keys_virtual(conn: sqlite3.Connection) -> None:
    """Extend api_keys table with Virtual Key fields (SQLite)."""
    # Helper to check column existence
    cur = conn.execute("PRAGMA table_info(api_keys)")
    cols = {row[1] for row in cur.fetchall()}

    def add_col(name: str, decl: str) -> None:
        if name not in cols:
            conn.execute(f"ALTER TABLE api_keys ADD COLUMN {decl}")

    add_col('is_virtual', "is_virtual INTEGER DEFAULT 0")
    add_col('parent_key_id', "parent_key_id INTEGER REFERENCES api_keys(id)")
    add_col('org_id', "org_id INTEGER REFERENCES organizations(id) ON DELETE SET NULL")
    add_col('team_id', "team_id INTEGER REFERENCES teams(id) ON DELETE SET NULL")
    add_col('llm_budget_day_tokens', "llm_budget_day_tokens INTEGER")
    add_col('llm_budget_month_tokens', "llm_budget_month_tokens INTEGER")
    add_col('llm_budget_day_usd', "llm_budget_day_usd REAL")
    add_col('llm_budget_month_usd', "llm_budget_month_usd REAL")
    add_col('llm_allowed_endpoints', "llm_allowed_endpoints TEXT")
    add_col('llm_allowed_providers', "llm_allowed_providers TEXT")
    add_col('llm_allowed_models', "llm_allowed_models TEXT")

    # Helpful indexes
    conn.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_virtual ON api_keys(is_virtual)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_org ON api_keys(org_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_team ON api_keys(team_id)")

    conn.commit()
    logger.info("Migration 017: Extended api_keys with virtual key fields")


def migration_018_add_usage_indexes(conn: sqlite3.Connection) -> None:
    """Add helpful indexes for usage tables (SQLite)."""
    with contextlib.suppress(_AUTHNZ_MIGRATIONS_NONCRITICAL_EXCEPTIONS):
        conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_log_ts ON usage_log(ts)")
    with contextlib.suppress(_AUTHNZ_MIGRATIONS_NONCRITICAL_EXCEPTIONS):
        conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_log_user ON usage_log(user_id)")
    with contextlib.suppress(_AUTHNZ_MIGRATIONS_NONCRITICAL_EXCEPTIONS):
        conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_log_status ON usage_log(status)")
    with contextlib.suppress(_AUTHNZ_MIGRATIONS_NONCRITICAL_EXCEPTIONS):
        conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_daily_day_user ON usage_daily(day, user_id)")
    conn.commit()
    logger.info("Migration 018: Added indexes for usage_log and usage_daily")


def migration_019_usage_log_add_request_id(conn: sqlite3.Connection) -> None:
    """Add request_id column to usage_log and index it (SQLite)."""
    try:
        conn.execute("ALTER TABLE usage_log ADD COLUMN request_id TEXT")
    except _AUTHNZ_MIGRATIONS_NONCRITICAL_EXCEPTIONS:
        # Column may already exist
        pass
    with contextlib.suppress(_AUTHNZ_MIGRATIONS_NONCRITICAL_EXCEPTIONS):
        conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_log_request_id ON usage_log(request_id)")
    conn.commit()
    logger.info("Migration 019: Added request_id column to usage_log")


def migration_020_usage_log_add_bytes_in(conn: sqlite3.Connection) -> None:
    """Add bytes_in column to usage_log (SQLite)."""
    try:
        conn.execute("ALTER TABLE usage_log ADD COLUMN bytes_in INTEGER")
    except _AUTHNZ_MIGRATIONS_NONCRITICAL_EXCEPTIONS:
        # Column may already exist
        pass
    conn.commit()
    logger.info("Migration 020: Added bytes_in column to usage_log")


def migration_021_usage_daily_add_bytes_in_total(conn: sqlite3.Connection) -> None:
    """Add bytes_in_total column to usage_daily (SQLite)."""
    try:
        conn.execute("ALTER TABLE usage_daily ADD COLUMN bytes_in_total INTEGER DEFAULT 0")
    except _AUTHNZ_MIGRATIONS_NONCRITICAL_EXCEPTIONS:
        # Column may already exist
        pass
    conn.commit()
    logger.info("Migration 021: Added bytes_in_total column to usage_daily")


def migration_049_add_llm_usage_log_key_ts_index(conn: sqlite3.Connection) -> None:
    """Add composite index for llm_usage_log key_id + ts (SQLite)."""
    with contextlib.suppress(_AUTHNZ_MIGRATIONS_NONCRITICAL_EXCEPTIONS):
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_llm_usage_log_key_ts ON llm_usage_log(key_id, ts)"
        )
    conn.commit()
    logger.info("Migration 049: Added llm_usage_log key_id + ts index")


def migration_050_create_generated_files_table(conn: sqlite3.Connection) -> None:
    """Create generated_files table for tracking user-generated content files.

    Tracks: TTS audio, images, voice clones, mindmaps, spreadsheets.
    Supports virtual folders via tags and soft delete with retention policies.
    """
    logger.info("Migration 050: START generated_files table")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS generated_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            uuid TEXT UNIQUE NOT NULL,

            -- Ownership
            user_id INTEGER NOT NULL,
            org_id INTEGER,
            team_id INTEGER,

            -- File metadata
            filename TEXT NOT NULL,
            original_filename TEXT,
            storage_path TEXT NOT NULL,
            mime_type TEXT,
            file_size_bytes INTEGER NOT NULL DEFAULT 0,
            checksum TEXT,

            -- Classification
            file_category TEXT NOT NULL,
            source_feature TEXT NOT NULL,
            source_ref TEXT,

            -- Organization (virtual folders via tags)
            folder_tag TEXT,
            tags TEXT,

            -- Lifecycle
            is_transient INTEGER DEFAULT 0,
            expires_at TIMESTAMP,
            retention_policy TEXT DEFAULT 'user_default',

            -- Soft delete
            is_deleted INTEGER DEFAULT 0,
            deleted_at TIMESTAMP,

            -- Timestamps
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            accessed_at TIMESTAMP,

            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY (org_id) REFERENCES organizations(id) ON DELETE SET NULL,
            FOREIGN KEY (team_id) REFERENCES teams(id) ON DELETE SET NULL
        )
        """
    )

    # Indexes for common query patterns
    conn.execute("CREATE INDEX IF NOT EXISTS idx_generated_files_user_id ON generated_files(user_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_generated_files_org_id ON generated_files(org_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_generated_files_team_id ON generated_files(team_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_generated_files_uuid ON generated_files(uuid)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_generated_files_category ON generated_files(file_category)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_generated_files_source_feature ON generated_files(source_feature)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_generated_files_folder_tag ON generated_files(folder_tag)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_generated_files_is_deleted ON generated_files(is_deleted)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_generated_files_expires_at ON generated_files(expires_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_generated_files_created_at ON generated_files(created_at)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_generated_files_user_category "
        "ON generated_files(user_id, file_category, is_deleted)"
    )

    conn.commit()
    logger.info("Migration 050: Created generated_files table with indexes")


def rollback_050_drop_generated_files_table(conn: sqlite3.Connection) -> None:
    """Rollback migration 050 by dropping generated_files table."""
    conn.execute("DROP TABLE IF EXISTS generated_files")
    conn.commit()
    logger.info("Rollback 050: Dropped generated_files table")


def migration_051_create_storage_quotas_table(conn: sqlite3.Connection) -> None:
    """Create storage_quotas table for team/org-level quota management.

    Provides shared pool quotas for teams and organizations.
    User quotas are stored on the users table (storage_quota_mb, storage_used_mb).
    """
    logger.info("Migration 051: START storage_quotas table")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS storage_quotas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            org_id INTEGER,
            team_id INTEGER,
            quota_mb INTEGER NOT NULL DEFAULT 10240,
            used_mb REAL DEFAULT 0,
            soft_limit_pct INTEGER DEFAULT 80,
            hard_limit_pct INTEGER DEFAULT 100,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (org_id) REFERENCES organizations(id) ON DELETE CASCADE,
            FOREIGN KEY (team_id) REFERENCES teams(id) ON DELETE CASCADE,

            CHECK ((org_id IS NOT NULL AND team_id IS NULL) OR
                   (org_id IS NULL AND team_id IS NOT NULL) OR
                   (org_id IS NULL AND team_id IS NULL))
        )
        """
    )

    # Indexes
    conn.execute("CREATE INDEX IF NOT EXISTS idx_storage_quotas_org_id ON storage_quotas(org_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_storage_quotas_team_id ON storage_quotas(team_id)")
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_storage_quotas_org_unique "
        "ON storage_quotas(org_id) WHERE org_id IS NOT NULL"
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_storage_quotas_team_unique "
        "ON storage_quotas(team_id) WHERE team_id IS NOT NULL"
    )

    conn.commit()
    logger.info("Migration 051: Created storage_quotas table with indexes")


def migration_052_create_org_team_role_permissions(conn: sqlite3.Connection) -> None:
    """Create org/team role-to-permission mapping tables."""
    logger.info("Migration 052: START org/team role permission tables")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS org_role_permissions (
            org_role TEXT NOT NULL,
            permission_id INTEGER NOT NULL,
            PRIMARY KEY (org_role, permission_id),
            FOREIGN KEY (permission_id) REFERENCES permissions(id) ON DELETE CASCADE,
            CHECK (org_role IN ('owner', 'admin', 'lead', 'member'))
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_org_role_permissions_permission_id "
        "ON org_role_permissions(permission_id)"
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS team_role_permissions (
            team_role TEXT NOT NULL,
            permission_id INTEGER NOT NULL,
            PRIMARY KEY (team_role, permission_id),
            FOREIGN KEY (permission_id) REFERENCES permissions(id) ON DELETE CASCADE,
            CHECK (team_role IN ('owner', 'admin', 'lead', 'member'))
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_team_role_permissions_permission_id "
        "ON team_role_permissions(permission_id)"
    )

    def _seed(role_name: str, scoped_role: str) -> None:
        conn.execute(
            """
            INSERT OR IGNORE INTO org_role_permissions (org_role, permission_id)
            SELECT ?, rp.permission_id
            FROM role_permissions rp
            JOIN roles r ON r.id = rp.role_id
            WHERE r.name = ?
            """,
            (scoped_role, role_name),
        )
        conn.execute(
            """
            INSERT OR IGNORE INTO team_role_permissions (team_role, permission_id)
            SELECT ?, rp.permission_id
            FROM role_permissions rp
            JOIN roles r ON r.id = rp.role_id
            WHERE r.name = ?
            """,
            (scoped_role, role_name),
        )

    _seed("admin", "owner")
    _seed("admin", "admin")
    _seed("reviewer", "lead")
    _seed("user", "member")

    conn.commit()
    logger.info("Migration 052: Created org/team role permission tables and seeded defaults")


def migration_053_create_byok_oauth_state(conn: sqlite3.Connection) -> None:
    """Create byok_oauth_state table for BYOK OAuth authorize/callback state."""
    logger.info("Migration 053: START byok_oauth_state table")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS byok_oauth_state (
            state TEXT NOT NULL,
            user_id INTEGER NOT NULL,
            provider TEXT NOT NULL,
            auth_session_id TEXT NOT NULL,
            redirect_uri TEXT NOT NULL,
            pkce_verifier_encrypted TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP NOT NULL,
            consumed_at TIMESTAMP,
            return_path TEXT,
            PRIMARY KEY (state, user_id),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """
    )

    try:
        cur = conn.execute("PRAGMA table_info(byok_oauth_state)")
        cols = {row[1] for row in cur.fetchall()}

        def add_col(name: str, decl: str):
            if name not in cols:
                conn.execute(f"ALTER TABLE byok_oauth_state ADD COLUMN {decl}")
                cols.add(name)

        add_col("provider", "provider TEXT")
        add_col("auth_session_id", "auth_session_id TEXT")
        add_col("redirect_uri", "redirect_uri TEXT")
        add_col("pkce_verifier_encrypted", "pkce_verifier_encrypted TEXT")
        add_col("created_at", "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        add_col("expires_at", "expires_at TIMESTAMP")
        add_col("consumed_at", "consumed_at TIMESTAMP")
        add_col("return_path", "return_path TEXT")
    except _AUTHNZ_MIGRATIONS_NONCRITICAL_EXCEPTIONS:
        pass

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_byok_oauth_state_provider_expires "
        "ON byok_oauth_state(provider, expires_at)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_byok_oauth_state_user_provider_consumed "
        "ON byok_oauth_state(user_id, provider, consumed_at)"
    )

    conn.commit()
    logger.info("Migration 053: Created byok_oauth_state table")


def migration_054_add_llm_usage_log_router_analytics_columns(conn: sqlite3.Connection) -> None:
    """Add router-analytics enrichment columns/indexes to llm_usage_log (SQLite)."""
    for column in ("remote_ip", "user_agent", "token_name", "conversation_id"):
        with contextlib.suppress(_AUTHNZ_MIGRATIONS_NONCRITICAL_EXCEPTIONS):
            conn.execute(f"ALTER TABLE llm_usage_log ADD COLUMN {column} TEXT")

    with contextlib.suppress(_AUTHNZ_MIGRATIONS_NONCRITICAL_EXCEPTIONS):
        conn.execute("CREATE INDEX IF NOT EXISTS idx_llm_usage_log_remote_ip_ts ON llm_usage_log(remote_ip, ts)")
    with contextlib.suppress(_AUTHNZ_MIGRATIONS_NONCRITICAL_EXCEPTIONS):
        conn.execute("CREATE INDEX IF NOT EXISTS idx_llm_usage_log_token_name_ts ON llm_usage_log(token_name, ts)")

    conn.commit()
    logger.info("Migration 054: Added llm_usage_log router analytics columns/indexes")


def migration_055_create_mcp_hub_tables(conn: sqlite3.Connection) -> None:
    """Create MCP Hub management tables (SQLite)."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS mcp_acp_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT,
            owner_scope_type TEXT NOT NULL DEFAULT 'user',
            owner_scope_id INTEGER,
            profile_json TEXT NOT NULL DEFAULT '{}',
            is_active INTEGER DEFAULT 1,
            created_by INTEGER,
            updated_by INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(name, owner_scope_type, owner_scope_id)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_mcp_acp_profiles_scope "
        "ON mcp_acp_profiles(owner_scope_type, owner_scope_id)"
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS mcp_external_servers (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            enabled INTEGER DEFAULT 1,
            owner_scope_type TEXT NOT NULL DEFAULT 'user',
            owner_scope_id INTEGER,
            transport TEXT NOT NULL,
            config_json TEXT NOT NULL DEFAULT '{}',
            created_by INTEGER,
            updated_by INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_mcp_external_servers_scope "
        "ON mcp_external_servers(owner_scope_type, owner_scope_id)"
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS mcp_external_server_secrets (
            server_id TEXT PRIMARY KEY,
            encrypted_blob TEXT NOT NULL,
            key_hint TEXT,
            updated_by INTEGER,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (server_id) REFERENCES mcp_external_servers(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_mcp_external_server_secrets_updated_at "
        "ON mcp_external_server_secrets(updated_at)"
    )

    conn.commit()
    logger.info("Migration 055: Created MCP Hub tables")

def migration_056_create_mcp_hub_policy_tables(conn: sqlite3.Connection) -> None:
    """Create MCP Hub policy and approval tables (SQLite)."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS mcp_permission_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT,
            owner_scope_type TEXT NOT NULL DEFAULT 'user',
            owner_scope_id INTEGER,
            mode TEXT NOT NULL DEFAULT 'custom',
            policy_document_json TEXT NOT NULL DEFAULT '{}',
            is_active INTEGER DEFAULT 1,
            created_by INTEGER,
            updated_by INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(name, owner_scope_type, owner_scope_id)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_mcp_permission_profiles_scope "
        "ON mcp_permission_profiles(owner_scope_type, owner_scope_id)"
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS mcp_policy_assignments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            target_type TEXT NOT NULL,
            target_id TEXT,
            owner_scope_type TEXT NOT NULL DEFAULT 'user',
            owner_scope_id INTEGER,
            profile_id INTEGER,
            inline_policy_document_json TEXT NOT NULL DEFAULT '{}',
            approval_policy_id INTEGER,
            is_active INTEGER DEFAULT 1,
            created_by INTEGER,
            updated_by INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (profile_id) REFERENCES mcp_permission_profiles(id) ON DELETE SET NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_mcp_policy_assignments_scope "
        "ON mcp_policy_assignments(owner_scope_type, owner_scope_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_mcp_policy_assignments_target "
        "ON mcp_policy_assignments(target_type, target_id)"
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS mcp_policy_overrides (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            assignment_id INTEGER NOT NULL UNIQUE,
            override_document_json TEXT NOT NULL DEFAULT '{}',
            is_active INTEGER DEFAULT 1,
            broadens_access INTEGER DEFAULT 0,
            grant_authority_snapshot_json TEXT NOT NULL DEFAULT '{}',
            created_by INTEGER,
            updated_by INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (assignment_id) REFERENCES mcp_policy_assignments(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_mcp_policy_overrides_assignment "
        "ON mcp_policy_overrides(assignment_id)"
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS mcp_approval_policies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT,
            owner_scope_type TEXT NOT NULL DEFAULT 'user',
            owner_scope_id INTEGER,
            mode TEXT NOT NULL,
            rules_json TEXT NOT NULL DEFAULT '{}',
            is_active INTEGER DEFAULT 1,
            created_by INTEGER,
            updated_by INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_mcp_approval_policies_scope "
        "ON mcp_approval_policies(owner_scope_type, owner_scope_id)"
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS mcp_approval_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            approval_policy_id INTEGER,
            context_key TEXT NOT NULL,
            conversation_id TEXT,
            tool_name TEXT NOT NULL,
            scope_key TEXT NOT NULL,
            decision TEXT NOT NULL,
            consume_on_match INTEGER DEFAULT 0,
            expires_at TIMESTAMP,
            consumed_at TIMESTAMP,
            created_by INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (approval_policy_id) REFERENCES mcp_approval_policies(id) ON DELETE SET NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_mcp_approval_decisions_context "
        "ON mcp_approval_decisions(context_key, conversation_id)"
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS mcp_credential_bindings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            binding_target_type TEXT NOT NULL,
            binding_target_id TEXT,
            external_server_id TEXT NOT NULL,
            credential_ref TEXT NOT NULL,
            usage_rules_json TEXT NOT NULL DEFAULT '{}',
            created_by INTEGER,
            updated_by INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (external_server_id) REFERENCES mcp_external_servers(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_mcp_credential_bindings_target "
        "ON mcp_credential_bindings(binding_target_type, binding_target_id)"
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS mcp_policy_audit_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            resource_type TEXT NOT NULL,
            resource_id TEXT NOT NULL,
            action TEXT NOT NULL,
            previous_value_json TEXT NOT NULL DEFAULT '{}',
            new_value_json TEXT NOT NULL DEFAULT '{}',
            broadened_access INTEGER DEFAULT 0,
            actor_id INTEGER,
            grant_authority_snapshot_json TEXT NOT NULL DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_mcp_policy_audit_history_resource "
        "ON mcp_policy_audit_history(resource_type, resource_id)"
    )

    conn.commit()
    logger.info("Migration 056: Created MCP Hub policy tables")


def migration_057_add_consumable_mcp_approval_decision_columns(conn: sqlite3.Connection) -> None:
    """Add explicit single-use approval columns to MCP Hub approval decisions."""

    def add_col(name: str, decl: str) -> None:
        cols = {str(row[1]) for row in conn.execute("PRAGMA table_info(mcp_approval_decisions)").fetchall()}
        if name not in cols:
            conn.execute(f"ALTER TABLE mcp_approval_decisions ADD COLUMN {decl}")

    add_col("consume_on_match", "consume_on_match INTEGER DEFAULT 0")
    add_col("consumed_at", "consumed_at TIMESTAMP")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_mcp_approval_decisions_active "
        "ON mcp_approval_decisions(context_key, conversation_id, tool_name, scope_key, decision, consumed_at)"
    )

    conn.commit()
    logger.info("Migration 057: Added MCP approval decision consumption columns")


def migration_058_harden_mcp_policy_override_schema(conn: sqlite3.Connection) -> None:
    """Add override activity tracking and enforce a 1:1 assignment override mapping."""

    cols = {str(row[1]) for row in conn.execute("PRAGMA table_info(mcp_policy_overrides)").fetchall()}
    if "is_active" not in cols:
        conn.execute("ALTER TABLE mcp_policy_overrides ADD COLUMN is_active INTEGER DEFAULT 1")
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_mcp_policy_overrides_assignment "
        "ON mcp_policy_overrides(assignment_id)"
    )

    conn.commit()
    logger.info("Migration 058: Hardened MCP policy override schema")


def migration_059_harden_mcp_external_binding_schema(conn: sqlite3.Connection) -> None:
    """Add managed/legacy external server metadata and tighten credential bindings."""

    external_columns = {
        str(row[1]) for row in conn.execute("PRAGMA table_info(mcp_external_servers)").fetchall()
    }
    if "server_source" not in external_columns:
        conn.execute(
            "ALTER TABLE mcp_external_servers ADD COLUMN server_source TEXT NOT NULL DEFAULT 'managed'"
        )
    if "legacy_source_ref" not in external_columns:
        conn.execute("ALTER TABLE mcp_external_servers ADD COLUMN legacy_source_ref TEXT")
    if "superseded_by_server_id" not in external_columns:
        conn.execute("ALTER TABLE mcp_external_servers ADD COLUMN superseded_by_server_id TEXT")

    binding_columns = {
        str(row[1]) for row in conn.execute("PRAGMA table_info(mcp_credential_bindings)").fetchall()
    }
    if "binding_mode" not in binding_columns:
        conn.execute(
            "ALTER TABLE mcp_credential_bindings ADD COLUMN binding_mode TEXT NOT NULL DEFAULT 'grant'"
        )

    conn.execute(
        """
        UPDATE mcp_credential_bindings
        SET binding_mode = CASE
            WHEN usage_rules_json LIKE '%"binding_mode":"disable"%'
              OR usage_rules_json LIKE '%"binding_mode": "disable"%'
            THEN 'disable'
            ELSE COALESCE(NULLIF(TRIM(binding_mode), ''), 'grant')
        END
        """
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_mcp_credential_bindings_target_server "
        "ON mcp_credential_bindings(binding_target_type, binding_target_id, external_server_id)"
    )

    conn.commit()
    logger.info("Migration 059: Hardened MCP external binding schema")


def _infer_default_external_slot(config_json: str | None) -> tuple[str, str] | None:
    """Infer a safe default slot for obvious single-secret managed auth modes."""
    try:
        config = json.loads(config_json or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(config, dict):
        return None
    auth = config.get("auth")
    if not isinstance(auth, dict):
        return None
    mode = str(auth.get("mode") or "").strip().lower()
    if mode == "bearer_token":
        return ("bearer_token", "bearer_token")
    if mode == "api_key_header":
        return ("api_key", "api_key")
    return None


def migration_060_add_mcp_external_credential_slots(conn: sqlite3.Connection) -> None:
    """Add external credential slot tables and evolve bindings to be slot-aware."""

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS mcp_external_server_credential_slots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            server_id TEXT NOT NULL,
            slot_name TEXT NOT NULL,
            display_name TEXT NOT NULL,
            secret_kind TEXT NOT NULL,
            privilege_class TEXT NOT NULL DEFAULT 'default',
            is_required INTEGER DEFAULT 0,
            created_by INTEGER,
            updated_by INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (server_id) REFERENCES mcp_external_servers(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_mcp_external_server_slots_server_slot "
        "ON mcp_external_server_credential_slots(server_id, slot_name)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_mcp_external_server_slots_server "
        "ON mcp_external_server_credential_slots(server_id)"
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS mcp_external_server_slot_secrets (
            slot_id INTEGER PRIMARY KEY,
            encrypted_blob TEXT NOT NULL,
            key_hint TEXT,
            updated_by INTEGER,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (slot_id) REFERENCES mcp_external_server_credential_slots(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_mcp_external_server_slot_secrets_updated_at "
        "ON mcp_external_server_slot_secrets(updated_at)"
    )

    binding_columns = {
        str(row[1]) for row in conn.execute("PRAGMA table_info(mcp_credential_bindings)").fetchall()
    }
    if "slot_name" not in binding_columns:
        conn.execute(
            "ALTER TABLE mcp_credential_bindings ADD COLUMN slot_name TEXT NOT NULL DEFAULT ''"
        )

    conn.execute("DROP INDEX IF EXISTS uq_mcp_credential_bindings_target_server")
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_mcp_credential_bindings_target_server_slot "
        "ON mcp_credential_bindings(binding_target_type, binding_target_id, external_server_id, slot_name)"
    )

    rows = conn.execute(
        """
        SELECT id, config_json, created_by, updated_by, created_at, updated_at
        FROM mcp_external_servers
        WHERE COALESCE(server_source, 'managed') = 'managed'
          AND superseded_by_server_id IS NULL
        """
    ).fetchall()
    for row in rows:
        server_id = str(row[0] or "")
        inferred = _infer_default_external_slot(row[1])
        if not server_id or inferred is None:
            continue
        slot_name, secret_kind = inferred
        display_name = slot_name.replace("_", " ").title()
        conn.execute(
            """
            INSERT OR IGNORE INTO mcp_external_server_credential_slots (
                server_id, slot_name, display_name, secret_kind, privilege_class, is_required,
                created_by, updated_by, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                server_id,
                slot_name,
                display_name,
                secret_kind,
                "default",
                1,
                row[2],
                row[3],
                row[4],
                row[5],
            ),
        )
        conn.execute(
            """
            UPDATE mcp_credential_bindings
            SET slot_name = ?
            WHERE external_server_id = ?
              AND COALESCE(TRIM(slot_name), '') = ''
            """,
            (slot_name, server_id),
        )
        slot_row = conn.execute(
            """
            SELECT id
            FROM mcp_external_server_credential_slots
            WHERE server_id = ?
              AND slot_name = ?
            """,
            (server_id, slot_name),
        ).fetchone()
        if slot_row is None:
            continue
        slot_id = int(slot_row[0])
        secret_row = conn.execute(
            """
            SELECT encrypted_blob, key_hint, updated_by, updated_at
            FROM mcp_external_server_secrets
            WHERE server_id = ?
            """,
            (server_id,),
        ).fetchone()
        if secret_row is None:
            continue
        conn.execute(
            """
            INSERT OR REPLACE INTO mcp_external_server_slot_secrets (
                slot_id, encrypted_blob, key_hint, updated_by, updated_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (slot_id, secret_row[0], secret_row[1], secret_row[2], secret_row[3]),
        )

    conn.commit()
    logger.info("Migration 060: Added MCP external credential slot schema")


def migration_061_add_mcp_path_scope_objects_and_assignment_workspaces(conn: sqlite3.Connection) -> None:
    """Add reusable path-scope objects and assignment workspace membership tables."""

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS mcp_path_scope_objects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT,
            owner_scope_type TEXT NOT NULL DEFAULT 'user',
            owner_scope_id INTEGER,
            path_scope_document_json TEXT NOT NULL DEFAULT '{}',
            is_active INTEGER DEFAULT 1,
            created_by INTEGER,
            updated_by INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_mcp_path_scope_objects_scope "
        "ON mcp_path_scope_objects(owner_scope_type, owner_scope_id)"
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_mcp_path_scope_objects_scope "
        "ON mcp_path_scope_objects(name, owner_scope_type, owner_scope_id)"
    )

    profile_columns = {
        str(row[1]) for row in conn.execute("PRAGMA table_info(mcp_permission_profiles)").fetchall()
    }
    if "path_scope_object_id" not in profile_columns:
        conn.execute("ALTER TABLE mcp_permission_profiles ADD COLUMN path_scope_object_id INTEGER")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_mcp_permission_profiles_path_scope_object "
        "ON mcp_permission_profiles(path_scope_object_id)"
    )

    assignment_columns = {
        str(row[1]) for row in conn.execute("PRAGMA table_info(mcp_policy_assignments)").fetchall()
    }
    if "path_scope_object_id" not in assignment_columns:
        conn.execute("ALTER TABLE mcp_policy_assignments ADD COLUMN path_scope_object_id INTEGER")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_mcp_policy_assignments_path_scope_object "
        "ON mcp_policy_assignments(path_scope_object_id)"
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS mcp_policy_assignment_workspaces (
            assignment_id INTEGER NOT NULL,
            workspace_id TEXT NOT NULL,
            created_by INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (assignment_id, workspace_id),
            FOREIGN KEY (assignment_id) REFERENCES mcp_policy_assignments(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_mcp_policy_assignment_workspaces_assignment "
        "ON mcp_policy_assignment_workspaces(assignment_id)"
    )

    conn.commit()
    logger.info("Migration 061: Added MCP path scope object schema")


def migration_062_add_mcp_workspace_set_objects(conn: sqlite3.Connection) -> None:
    """Add reusable workspace-set objects and assignment workspace source fields."""

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS mcp_workspace_set_objects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT,
            owner_scope_type TEXT NOT NULL DEFAULT 'user',
            owner_scope_id INTEGER,
            is_active INTEGER DEFAULT 1,
            created_by INTEGER,
            updated_by INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_mcp_workspace_set_objects_scope "
        "ON mcp_workspace_set_objects(owner_scope_type, owner_scope_id)"
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_mcp_workspace_set_objects_scope "
        "ON mcp_workspace_set_objects(name, owner_scope_type, owner_scope_id)"
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS mcp_workspace_set_object_members (
            workspace_set_object_id INTEGER NOT NULL,
            workspace_id TEXT NOT NULL,
            created_by INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (workspace_set_object_id, workspace_id),
            FOREIGN KEY (workspace_set_object_id) REFERENCES mcp_workspace_set_objects(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_mcp_workspace_set_object_members_object "
        "ON mcp_workspace_set_object_members(workspace_set_object_id)"
    )

    assignment_columns = {
        str(row[1]) for row in conn.execute("PRAGMA table_info(mcp_policy_assignments)").fetchall()
    }
    if "workspace_source_mode" not in assignment_columns:
        conn.execute("ALTER TABLE mcp_policy_assignments ADD COLUMN workspace_source_mode TEXT")
    if "workspace_set_object_id" not in assignment_columns:
        conn.execute("ALTER TABLE mcp_policy_assignments ADD COLUMN workspace_set_object_id INTEGER")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_mcp_policy_assignments_workspace_set_object "
        "ON mcp_policy_assignments(workspace_set_object_id)"
    )

    conn.commit()
    logger.info("Migration 062: Added MCP workspace set object schema")


def migration_063_add_mcp_shared_workspaces(conn: sqlite3.Connection) -> None:
    """Add shared workspace registry entries for shared-scope path governance."""

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS mcp_shared_workspaces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            workspace_id TEXT NOT NULL,
            display_name TEXT NOT NULL,
            absolute_root TEXT NOT NULL,
            owner_scope_type TEXT NOT NULL DEFAULT 'team',
            owner_scope_id INTEGER,
            is_active INTEGER DEFAULT 1,
            created_by INTEGER,
            updated_by INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_mcp_shared_workspaces_scope "
        "ON mcp_shared_workspaces(owner_scope_type, owner_scope_id)"
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_mcp_shared_workspaces_scope_workspace "
        "ON mcp_shared_workspaces(owner_scope_type, owner_scope_id, workspace_id)"
    )

    conn.commit()
    logger.info("Migration 063: Added MCP shared workspace registry schema")


def migration_069_add_mcp_governance_pack_schema(conn: sqlite3.Connection) -> None:
    """Add governance-pack provenance tables and immutability flags for MCP Hub."""

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS mcp_governance_packs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pack_id TEXT NOT NULL,
            pack_version TEXT NOT NULL,
            pack_schema_version INTEGER NOT NULL,
            capability_taxonomy_version INTEGER NOT NULL,
            adapter_contract_version INTEGER NOT NULL,
            title TEXT NOT NULL,
            description TEXT,
            owner_scope_type TEXT NOT NULL DEFAULT 'user',
            owner_scope_id INTEGER,
            bundle_digest TEXT NOT NULL,
            manifest_json TEXT NOT NULL DEFAULT '{}',
            normalized_ir_json TEXT NOT NULL DEFAULT '{}',
            created_by INTEGER,
            updated_by INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_mcp_governance_packs_scope "
        "ON mcp_governance_packs(owner_scope_type, owner_scope_id)"
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_mcp_governance_packs_scope_version "
        "ON mcp_governance_packs(pack_id, pack_version, owner_scope_type, owner_scope_id)"
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS mcp_governance_pack_objects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            governance_pack_id INTEGER NOT NULL,
            object_type TEXT NOT NULL,
            object_id TEXT NOT NULL,
            source_object_id TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (governance_pack_id) REFERENCES mcp_governance_packs(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_mcp_governance_pack_objects_pack "
        "ON mcp_governance_pack_objects(governance_pack_id)"
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_mcp_governance_pack_objects_pack_source "
        "ON mcp_governance_pack_objects(governance_pack_id, object_type, source_object_id)"
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_mcp_governance_pack_objects_object "
        "ON mcp_governance_pack_objects(object_type, object_id)"
    )

    profile_columns = {
        str(row[1]) for row in conn.execute("PRAGMA table_info(mcp_permission_profiles)").fetchall()
    }
    if "is_immutable" not in profile_columns:
        conn.execute(
            "ALTER TABLE mcp_permission_profiles ADD COLUMN is_immutable INTEGER NOT NULL DEFAULT 0"
        )

    assignment_columns = {
        str(row[1]) for row in conn.execute("PRAGMA table_info(mcp_policy_assignments)").fetchall()
    }
    if "is_immutable" not in assignment_columns:
        conn.execute(
            "ALTER TABLE mcp_policy_assignments ADD COLUMN is_immutable INTEGER NOT NULL DEFAULT 0"
        )

    approval_columns = {
        str(row[1]) for row in conn.execute("PRAGMA table_info(mcp_approval_policies)").fetchall()
    }
    if "is_immutable" not in approval_columns:
        conn.execute(
            "ALTER TABLE mcp_approval_policies ADD COLUMN is_immutable INTEGER NOT NULL DEFAULT 0"
        )

    conn.commit()
    logger.info("Migration 069: Added MCP governance pack schema")


def migration_070_add_mcp_capability_adapter_mappings(conn: sqlite3.Connection) -> None:
    """Add scope-aware MCP capability adapter mapping storage."""

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS mcp_capability_adapter_mappings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mapping_id TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT,
            owner_scope_type TEXT NOT NULL DEFAULT 'global',
            owner_scope_id INTEGER,
            capability_name TEXT NOT NULL,
            adapter_contract_version INTEGER NOT NULL,
            resolved_policy_document_json TEXT NOT NULL DEFAULT '{}',
            supported_environment_requirements_json TEXT NOT NULL DEFAULT '[]',
            is_active INTEGER NOT NULL DEFAULT 1,
            created_by INTEGER,
            updated_by INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_mcp_capability_adapter_mappings_scope "
        "ON mcp_capability_adapter_mappings(owner_scope_type, owner_scope_id)"
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_mcp_capability_adapter_mappings_mapping_id "
        "ON mcp_capability_adapter_mappings(mapping_id)"
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_mcp_capability_adapter_mappings_active_scope_capability "
        "ON mcp_capability_adapter_mappings("
        "owner_scope_type, IFNULL(owner_scope_id, -1), capability_name"
        ") WHERE is_active = 1"
    )

    conn.commit()
    logger.info("Migration 070: Added MCP capability adapter mapping schema")


def migration_071_add_governance_pack_upgrade_lineage(conn: sqlite3.Connection) -> None:
    """Add governance-pack install-state and upgrade-lineage tracking."""

    governance_pack_columns = {
        str(row[1]) for row in conn.execute("PRAGMA table_info(mcp_governance_packs)").fetchall()
    }
    if "is_active_install" not in governance_pack_columns:
        conn.execute(
            "ALTER TABLE mcp_governance_packs ADD COLUMN is_active_install INTEGER NOT NULL DEFAULT 1"
        )
    if "superseded_by_governance_pack_id" not in governance_pack_columns:
        conn.execute(
            "ALTER TABLE mcp_governance_packs ADD COLUMN superseded_by_governance_pack_id INTEGER"
        )
    if "installed_from_upgrade_id" not in governance_pack_columns:
        conn.execute(
            "ALTER TABLE mcp_governance_packs ADD COLUMN installed_from_upgrade_id INTEGER"
        )

    governance_pack_indexes = {
        str(row[1]) for row in conn.execute("PRAGMA index_list(mcp_governance_packs)").fetchall()
    }
    if "uq_mcp_governance_packs_active_scope" not in governance_pack_indexes:
        conn.execute("UPDATE mcp_governance_packs SET is_active_install = 0")
        conn.execute(
            """
            UPDATE mcp_governance_packs
            SET is_active_install = 1
            WHERE id IN (
                SELECT MAX(id)
                FROM mcp_governance_packs
                GROUP BY pack_id, owner_scope_type, IFNULL(owner_scope_id, -1)
            )
            """
        )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_mcp_governance_packs_active_scope "
            "ON mcp_governance_packs(pack_id, owner_scope_type, IFNULL(owner_scope_id, -1)) "
            "WHERE is_active_install = 1"
        )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS mcp_governance_pack_upgrades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pack_id TEXT NOT NULL,
            owner_scope_type TEXT NOT NULL DEFAULT 'user',
            owner_scope_id INTEGER,
            from_governance_pack_id INTEGER NOT NULL,
            to_governance_pack_id INTEGER NOT NULL,
            from_pack_version TEXT NOT NULL,
            to_pack_version TEXT NOT NULL,
            status TEXT NOT NULL,
            planned_by INTEGER,
            executed_by INTEGER,
            planner_inputs_fingerprint TEXT,
            adapter_state_fingerprint TEXT,
            plan_summary_json TEXT NOT NULL DEFAULT '{}',
            accepted_resolutions_json TEXT NOT NULL DEFAULT '{}',
            failure_summary TEXT,
            planned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            executed_at TIMESTAMP,
            FOREIGN KEY (from_governance_pack_id) REFERENCES mcp_governance_packs(id) ON DELETE CASCADE,
            FOREIGN KEY (to_governance_pack_id) REFERENCES mcp_governance_packs(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_mcp_governance_pack_upgrades_scope "
        "ON mcp_governance_pack_upgrades(pack_id, owner_scope_type, owner_scope_id)"
    )

    conn.commit()
    logger.info("Migration 071: Added governance-pack upgrade lineage schema")


def rollback_053_drop_byok_oauth_state(conn: sqlite3.Connection) -> None:
    """Rollback migration 053 by dropping the byok_oauth_state table."""
    conn.execute("DROP TABLE IF EXISTS byok_oauth_state")
    conn.commit()
    logger.info("Rollback 053: Dropped byok_oauth_state table")



def rollback_051_drop_storage_quotas_table(conn: sqlite3.Connection) -> None:
    """Rollback migration 051 by dropping storage_quotas table."""
    conn.execute("DROP TABLE IF EXISTS storage_quotas")
    conn.commit()
    logger.info("Rollback 051: Dropped storage_quotas table")


def migration_022_create_tool_catalogs(conn: sqlite3.Connection) -> None:
    """Create tables for MCP tool catalogs (SQLite)."""
    # tool_catalogs: scoped by (org_id, team_id) with name unique per scope
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tool_catalogs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT,
            org_id INTEGER,
            team_id INTEGER,
            is_active INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(name, org_id, team_id),
            FOREIGN KEY (org_id) REFERENCES organizations(id) ON DELETE SET NULL,
            FOREIGN KEY (team_id) REFERENCES teams(id) ON DELETE SET NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tool_catalogs_org_team ON tool_catalogs(org_id, team_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tool_catalogs_name ON tool_catalogs(name)")

    # tool_catalog_entries
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tool_catalog_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            catalog_id INTEGER NOT NULL,
            tool_name TEXT NOT NULL,
            module_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(catalog_id, tool_name),
            FOREIGN KEY (catalog_id) REFERENCES tool_catalogs(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tool_catalog_entries_catalog ON tool_catalog_entries(catalog_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tool_catalog_entries_tool ON tool_catalog_entries(tool_name)")
    conn.commit()
    logger.info("Migration 022: Created tool_catalogs and tool_catalog_entries tables")


def migration_023_create_virtual_key_counters(conn: sqlite3.Connection) -> None:
    """Create counters tables for virtual keys (SQLite)."""
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vk_jwt_counters (
                jti TEXT NOT NULL,
                counter_type TEXT NOT NULL,
                count INTEGER DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (jti, counter_type)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vk_api_key_counters (
                api_key_id INTEGER NOT NULL,
                counter_type TEXT NOT NULL,
                count INTEGER DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (api_key_id, counter_type)
            )
            """
        )
        # Helpful indexes for reporting/cleanup (best-effort)
        with contextlib.suppress(_AUTHNZ_MIGRATIONS_NONCRITICAL_EXCEPTIONS):
            conn.execute("CREATE INDEX IF NOT EXISTS idx_vk_jwt_counters_type ON vk_jwt_counters(counter_type)")
        with contextlib.suppress(_AUTHNZ_MIGRATIONS_NONCRITICAL_EXCEPTIONS):
            conn.execute("CREATE INDEX IF NOT EXISTS idx_vk_api_key_counters_type ON vk_api_key_counters(counter_type)")
        conn.commit()
        logger.info("Migration 023: Created virtual key counters tables")
    except _AUTHNZ_MIGRATIONS_NONCRITICAL_EXCEPTIONS as e:
        logger.warning(f"Migration 023 skipped/failed: {e}")


def migration_024_ensure_api_keys_status_column(conn: sqlite3.Connection) -> None:
    """Ensure the api_keys table exposes a status column prior to creating status-based indexes."""
    try:
        teams_exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='teams'"
        ).fetchone()
        orgs_exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='organizations'"
        ).fetchone()
        if not teams_exists or not orgs_exists:
            migration_016_create_orgs_teams(conn)
    except _AUTHNZ_MIGRATIONS_NONCRITICAL_EXCEPTIONS as error:
        logger.warning(f"Migration 024: unable to verify organization/team tables ({error})")

    try:
        cursor = conn.execute("PRAGMA table_info(api_keys)")
        columns = {row[1] for row in cursor.fetchall()}
        if "status" not in columns:
            conn.execute("ALTER TABLE api_keys ADD COLUMN status TEXT DEFAULT 'active'")
            conn.execute("UPDATE api_keys SET status = 'active' WHERE status IS NULL")
    except sqlite3.OperationalError as error:
        logger.warning(f"Migration 024 skipped: api_keys table unavailable ({error})")
        conn.commit()
        return
    except _AUTHNZ_MIGRATIONS_NONCRITICAL_EXCEPTIONS as error:
        logger.warning(f"Migration 024 encountered an unexpected error inspecting api_keys: {error}")
        conn.commit()
        return

    try:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_status ON api_keys(status)")
    except sqlite3.OperationalError as error:
        logger.warning(f"Migration 024: idx_api_keys_status creation skipped ({error})")
    conn.commit()
    logger.info("Migration 024: Ensured api_keys.status column and index")


def migration_025_team_members_added_at(conn: sqlite3.Connection) -> None:
    """Ensure team_members table exposes added_at column and backfill legacy data."""
    try:
        cursor = conn.execute("PRAGMA table_info(team_members)")
        columns = {row[1] for row in cursor.fetchall()}
    except sqlite3.OperationalError as error:
        logger.warning(f"Migration 025 skipped: team_members table unavailable ({error})")
        conn.commit()
        return

    if "added_at" not in columns:
        conn.execute(
            "ALTER TABLE team_members ADD COLUMN added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
        )

    if "joined_at" in columns:
        conn.execute(
            """
            UPDATE team_members
            SET added_at = COALESCE(added_at, joined_at, CURRENT_TIMESTAMP)
            """
        )

    conn.commit()
    logger.info("Migration 025: Ensured team_members.added_at column with backfill")


def migration_026_create_privilege_snapshots_table(conn: sqlite3.Connection) -> None:
    """Create privilege_snapshots table for snapshot storage."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS privilege_snapshots (
            snapshot_id TEXT PRIMARY KEY,
            generated_at TEXT NOT NULL,
            generated_by TEXT NOT NULL,
            org_id TEXT,
            team_id TEXT,
            catalog_version TEXT NOT NULL,
            summary_json TEXT,
            scope_index TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_priv_snapshots_generated_at ON privilege_snapshots(generated_at)"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_priv_snapshots_org ON privilege_snapshots(org_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_priv_snapshots_team ON privilege_snapshots(team_id)")
    conn.commit()
    logger.info("Migration 026: Created privilege_snapshots table")


def rollback_026_drop_privilege_snapshots_table(conn: sqlite3.Connection) -> None:
    """Drop privilege_snapshots table during rollback/testing."""
    conn.execute("DROP TABLE IF EXISTS privilege_snapshots")
    conn.commit()
    logger.info("Rollback 026: Dropped privilege_snapshots table")
#######################################################################################################################
#
# Additional maintenance migrations
#


def migration_027_add_session_revocation_columns(conn: sqlite3.Connection) -> None:
    """Ensure sessions table has revocation bookkeeping columns for legacy upgrades."""
    try:
        cursor = conn.execute("PRAGMA table_info(sessions)")
        rows = cursor.fetchall()
        columns = {row[1] for row in rows}

        def add_col(name: str, decl: str):
            if name not in columns:
                conn.execute(f"ALTER TABLE sessions ADD COLUMN {decl}")
                columns.add(name)

        add_col("is_active", "is_active INTEGER DEFAULT 1")
        add_col("is_revoked", "is_revoked INTEGER DEFAULT 0")
        add_col("revoked_at", "revoked_at TIMESTAMP")
        add_col("revoked_by", "revoked_by INTEGER")
        add_col("revoke_reason", "revoke_reason TEXT")
        conn.commit()
        logger.info("Migration 027: Harmonized session revocation columns")
    except _AUTHNZ_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning(f"Migration 027: Unable to harmonize session columns: {exc}")


def migration_028_create_org_invites(conn: sqlite3.Connection) -> None:
    """Create org_invites table for shareable invite codes."""
    logger.info("Migration 028: START org_invites table")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS org_invites (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT UNIQUE NOT NULL,
            org_id INTEGER NOT NULL,
            team_id INTEGER,
            role_to_grant TEXT DEFAULT 'member',
            created_by INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP NOT NULL,
            max_uses INTEGER DEFAULT 1,
            uses_count INTEGER DEFAULT 0,
            is_active INTEGER DEFAULT 1,
            allowed_email_domain TEXT,
            description TEXT,
            metadata TEXT,
            FOREIGN KEY (org_id) REFERENCES organizations(id) ON DELETE CASCADE,
            FOREIGN KEY (team_id) REFERENCES teams(id) ON DELETE CASCADE,
            FOREIGN KEY (created_by) REFERENCES users(id) ON DELETE SET NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_org_invites_code ON org_invites(code)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_org_invites_org_active ON org_invites(org_id, is_active)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_org_invites_expires ON org_invites(expires_at)")
    conn.commit()
    logger.info("Migration 028: Created org_invites table")


def rollback_028_drop_org_invites_table(conn: sqlite3.Connection) -> None:
    """Drop org_invites table during rollback/testing."""
    conn.execute("DROP TABLE IF EXISTS org_invites")
    conn.commit()
    logger.info("Rollback 028: Dropped org_invites table")


def migration_029_create_org_invite_redemptions(conn: sqlite3.Connection) -> None:
    """Create org_invite_redemptions table to track who redeemed invites."""
    logger.info("Migration 029: START org_invite_redemptions table")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS org_invite_redemptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            invite_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            redeemed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ip_address TEXT,
            user_agent TEXT,
            FOREIGN KEY (invite_id) REFERENCES org_invites(id) ON DELETE CASCADE,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            UNIQUE(invite_id, user_id)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_invite_redemptions_invite ON org_invite_redemptions(invite_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_invite_redemptions_user ON org_invite_redemptions(user_id)")
    conn.commit()
    logger.info("Migration 029: Created org_invite_redemptions table")


def rollback_029_drop_org_invite_redemptions_table(conn: sqlite3.Connection) -> None:
    """Drop org_invite_redemptions table during rollback/testing."""
    conn.execute("DROP TABLE IF EXISTS org_invite_redemptions")
    conn.commit()
    logger.info("Rollback 029: Dropped org_invite_redemptions table")


def migration_030_create_subscription_plans(conn: sqlite3.Connection) -> None:
    """Retired OSS billing migration retained as a compatibility no-op."""
    conn.commit()
    logger.info("Migration 030: Retired public billing plan schema bootstrap")


def rollback_030_drop_subscription_plans_table(conn: sqlite3.Connection) -> None:
    """Drop subscription_plans table during rollback/testing."""
    conn.execute("DROP TABLE IF EXISTS subscription_plans")
    conn.commit()
    logger.info("Rollback 030: Dropped subscription_plans table")


def migration_031_create_org_subscriptions(conn: sqlite3.Connection) -> None:
    """Retired OSS billing migration retained as a compatibility no-op."""
    conn.commit()
    logger.info("Migration 031: Retired public org_subscriptions schema bootstrap")


def rollback_031_drop_org_subscriptions_table(conn: sqlite3.Connection) -> None:
    """Drop org_subscriptions table during rollback/testing."""
    conn.execute("DROP TABLE IF EXISTS org_subscriptions")
    conn.commit()
    logger.info("Rollback 031: Dropped org_subscriptions table")


def migration_032_create_stripe_webhook_events(conn: sqlite3.Connection) -> None:
    """Retired OSS billing migration retained as a compatibility no-op."""
    conn.commit()
    logger.info("Migration 032: Retired public Stripe webhook schema bootstrap")


def rollback_032_drop_stripe_webhook_events_table(conn: sqlite3.Connection) -> None:
    """Drop stripe_webhook_events table during rollback/testing."""
    conn.execute("DROP TABLE IF EXISTS stripe_webhook_events")
    conn.commit()
    logger.info("Rollback 032: Dropped stripe_webhook_events table")


def migration_033_create_payment_history(conn: sqlite3.Connection) -> None:
    """Retired OSS billing migration retained as a compatibility no-op."""
    conn.commit()
    logger.info("Migration 033: Retired public payment history schema bootstrap")


def rollback_033_drop_payment_history_table(conn: sqlite3.Connection) -> None:
    """Drop payment_history table during rollback/testing."""
    conn.execute("DROP TABLE IF EXISTS payment_history")
    conn.commit()
    logger.info("Rollback 033: Dropped payment_history table")


def migration_034_create_billing_audit_log(conn: sqlite3.Connection) -> None:
    """Retired OSS billing migration retained as a compatibility no-op."""
    conn.commit()
    logger.info("Migration 034: Retired public billing audit schema bootstrap")


def rollback_034_drop_billing_audit_log_table(conn: sqlite3.Connection) -> None:
    """Drop billing_audit_log table during rollback/testing."""
    conn.execute("DROP TABLE IF EXISTS billing_audit_log")
    conn.commit()
    logger.info("Rollback 034: Dropped billing_audit_log table")


def rollback_retired_billing_schema_noop(conn: sqlite3.Connection) -> None:
    """Preserve historical billing tables during OSS rollback flows."""
    conn.commit()
    logger.info("Rollback: skipped retired billing schema teardown")


def migration_035_backfill_storage_mb_limits(conn: sqlite3.Connection) -> None:
    """Normalize storage limits to storage_mb in plan and custom limit JSON."""
    logger.info("Migration 035: START storage_mb limit backfill")

    def _normalize_limits_json(raw_json: str) -> Optional[str]:
        if not raw_json:
            return None
        try:
            data = json.loads(raw_json)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(data, dict):
            return None

        changed = False
        if "storage_gb" in data:
            storage_mb = None
            try:
                storage_mb = int(float(data["storage_gb"]) * 1024)
            except (TypeError, ValueError):
                storage_mb = None
            if storage_mb is not None and "storage_mb" not in data:
                data["storage_mb"] = storage_mb
            data.pop("storage_gb", None)
            changed = True
        return json.dumps(data) if changed else None

    if _sqlite_table_exists(conn, "subscription_plans"):
        cur = conn.execute("SELECT id, limits_json FROM subscription_plans")
        rows = cur.fetchall()
        for plan_id, limits_json in rows:
            updated = _normalize_limits_json(limits_json)
            if updated is not None:
                conn.execute(
                    "UPDATE subscription_plans SET limits_json = ? WHERE id = ?",
                    (updated, plan_id),
                )

    if _sqlite_table_exists(conn, "org_subscriptions"):
        cur = conn.execute(
            "SELECT id, custom_limits_json FROM org_subscriptions WHERE custom_limits_json IS NOT NULL"
        )
        rows = cur.fetchall()
        for sub_id, limits_json in rows:
            updated = _normalize_limits_json(limits_json)
            if updated is not None:
                conn.execute(
                    "UPDATE org_subscriptions SET custom_limits_json = ? WHERE id = ?",
                    (updated, sub_id),
                )

    conn.commit()
    logger.info("Migration 035: Completed storage_mb limit backfill")


def migration_036_create_user_provider_secrets(conn: sqlite3.Connection) -> None:
    """Create the user_provider_secrets table for per-user BYOK credentials."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS user_provider_secrets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            provider TEXT NOT NULL,
            encrypted_blob TEXT NOT NULL,
            key_hint TEXT,
            metadata TEXT,
            created_by INTEGER,
            updated_by INTEGER,
            revoked_by INTEGER,
            revoked_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_used_at TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            UNIQUE (user_id, provider)
        )
        """
    )

    # Harmonize legacy schemas: add missing columns if needed
    try:
        cur = conn.execute("PRAGMA table_info(user_provider_secrets)")
        cols = {row[1] for row in cur.fetchall()}

        def add_col(name: str, decl: str):
            if name not in cols:
                conn.execute(f"ALTER TABLE user_provider_secrets ADD COLUMN {decl}")
                cols.add(name)

        add_col("encrypted_blob", "encrypted_blob TEXT")
        add_col("key_hint", "key_hint TEXT")
        add_col("metadata", "metadata TEXT")
        add_col("created_by", "created_by INTEGER")
        add_col("updated_by", "updated_by INTEGER")
        add_col("revoked_by", "revoked_by INTEGER")
        add_col("revoked_at", "revoked_at TIMESTAMP")
        add_col("created_at", "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        add_col("updated_at", "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        add_col("last_used_at", "last_used_at TIMESTAMP")
    except _AUTHNZ_MIGRATIONS_NONCRITICAL_EXCEPTIONS:
        pass

    conn.execute("CREATE INDEX IF NOT EXISTS idx_user_provider_secrets_user_id ON user_provider_secrets(user_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_user_provider_secrets_provider ON user_provider_secrets(provider)")
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_user_provider_secrets_user_provider "
        "ON user_provider_secrets(user_id, provider)"
    )

    conn.commit()
    logger.info("Migration 036: Created user_provider_secrets table")


def rollback_036_drop_user_provider_secrets_table(conn: sqlite3.Connection) -> None:
    """Rollback migration 036 by dropping the user_provider_secrets table."""
    conn.execute("DROP TABLE IF EXISTS user_provider_secrets")
    conn.commit()
    logger.info("Rollback 036: Dropped user_provider_secrets table")


def migration_037_create_org_provider_secrets(conn: sqlite3.Connection) -> None:
    """Create the org_provider_secrets table for org/team shared BYOK credentials."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS org_provider_secrets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scope_type TEXT NOT NULL,
            scope_id INTEGER NOT NULL,
            provider TEXT NOT NULL,
            encrypted_blob TEXT NOT NULL,
            key_hint TEXT,
            metadata TEXT,
            created_by INTEGER,
            updated_by INTEGER,
            revoked_by INTEGER,
            revoked_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_used_at TIMESTAMP,
            UNIQUE (scope_type, scope_id, provider)
        )
        """
    )

    # Harmonize legacy schemas: add missing columns if needed
    try:
        cur = conn.execute("PRAGMA table_info(org_provider_secrets)")
        cols = {row[1] for row in cur.fetchall()}

        def add_col(name: str, decl: str):
            if name not in cols:
                conn.execute(f"ALTER TABLE org_provider_secrets ADD COLUMN {decl}")
                cols.add(name)

        add_col("encrypted_blob", "encrypted_blob TEXT")
        add_col("key_hint", "key_hint TEXT")
        add_col("metadata", "metadata TEXT")
        add_col("created_by", "created_by INTEGER")
        add_col("updated_by", "updated_by INTEGER")
        add_col("revoked_by", "revoked_by INTEGER")
        add_col("revoked_at", "revoked_at TIMESTAMP")
        add_col("created_at", "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        add_col("updated_at", "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        add_col("last_used_at", "last_used_at TIMESTAMP")
    except _AUTHNZ_MIGRATIONS_NONCRITICAL_EXCEPTIONS:
        pass

    conn.execute("CREATE INDEX IF NOT EXISTS idx_org_provider_secrets_scope ON org_provider_secrets(scope_type, scope_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_org_provider_secrets_provider ON org_provider_secrets(provider)")
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_org_provider_secrets_scope_provider "
        "ON org_provider_secrets(scope_type, scope_id, provider)"
    )

    conn.commit()
    logger.info("Migration 037: Created org_provider_secrets table")


def rollback_037_drop_org_provider_secrets_table(conn: sqlite3.Connection) -> None:
    """Rollback migration 037 by dropping the org_provider_secrets table."""
    conn.execute("DROP TABLE IF EXISTS org_provider_secrets")
    conn.commit()
    logger.info("Rollback 037: Dropped org_provider_secrets table")


def migration_038_add_org_provider_secrets_cleanup_triggers(conn: sqlite3.Connection) -> None:
    """Add cleanup triggers for org_provider_secrets when orgs/teams are deleted."""
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS delete_org_provider_secrets_on_org_delete
            AFTER DELETE ON organizations
            FOR EACH ROW
        BEGIN
            DELETE FROM org_provider_secrets WHERE scope_type = 'org' AND scope_id = OLD.id;
        END;
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS delete_org_provider_secrets_on_team_delete
            AFTER DELETE ON teams
            FOR EACH ROW
        BEGIN
            DELETE FROM org_provider_secrets WHERE scope_type = 'team' AND scope_id = OLD.id;
        END;
        """
    )
    conn.commit()
    logger.info("Migration 038: Added org_provider_secrets cleanup triggers")


def rollback_038_drop_org_provider_secrets_cleanup_triggers(conn: sqlite3.Connection) -> None:
    """Rollback migration 038 by dropping org_provider_secrets cleanup triggers."""
    conn.execute("DROP TRIGGER IF EXISTS delete_org_provider_secrets_on_org_delete")
    conn.execute("DROP TRIGGER IF EXISTS delete_org_provider_secrets_on_team_delete")
    conn.commit()
    logger.info("Rollback 038: Dropped org_provider_secrets cleanup triggers")


def migration_039_ensure_user_storage_columns(conn: sqlite3.Connection) -> None:
    """Ensure legacy users tables include storage quota/usage columns."""
    logger.info("Migration 039: START ensure storage columns on users table")
    try:
        cur = conn.execute("PRAGMA table_info(users)")
        columns = {row[1] for row in cur.fetchall()}

        if "storage_quota_mb" not in columns:
            conn.execute("ALTER TABLE users ADD COLUMN storage_quota_mb INTEGER DEFAULT 5120")
            columns.add("storage_quota_mb")

        if "storage_used_mb" not in columns:
            conn.execute("ALTER TABLE users ADD COLUMN storage_used_mb INTEGER DEFAULT 0")
            columns.add("storage_used_mb")

        conn.commit()
        logger.info("Migration 039: storage columns ensured on users table")
    except _AUTHNZ_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Migration 039: failed to ensure storage columns: {}", exc)
        raise


def migration_040_extend_registration_codes_for_org_invites(conn: sqlite3.Connection) -> None:
    """Add org-scoped invite columns and missing registration code fields."""
    logger.info("Migration 040: START registration_codes org invite fields")
    cur = conn.execute("PRAGMA table_info(registration_codes)")
    columns = {row[1] for row in cur.fetchall()}

    def add_col(name: str, decl: str) -> None:
        if name not in columns:
            conn.execute(f"ALTER TABLE registration_codes ADD COLUMN {decl}")
            columns.add(name)

    # Core fields expected by registration/admin flows
    add_col("role_to_grant", "role_to_grant TEXT DEFAULT 'user'")
    add_col("description", "description TEXT")
    add_col("allowed_email_domain", "allowed_email_domain TEXT")
    add_col("times_used", "times_used INTEGER DEFAULT 0")
    add_col("is_active", "is_active INTEGER DEFAULT 1")
    add_col("metadata", "metadata TEXT")

    # Org-scoped invite extension
    add_col("org_id", "org_id INTEGER REFERENCES organizations(id) ON DELETE SET NULL")
    add_col("org_role", "org_role TEXT")
    add_col("team_id", "team_id INTEGER REFERENCES teams(id) ON DELETE SET NULL")

    # Backfill times_used from legacy uses_count if present
    if "uses_count" in columns and "times_used" in columns:
        conn.execute(
            """
            UPDATE registration_codes
            SET times_used = uses_count
            WHERE (times_used IS NULL OR times_used = 0)
              AND uses_count > 0
            """
        )

    conn.commit()
    logger.info("Migration 040: Updated registration_codes for org invites")


def migration_046_add_org_invite_allowlist_domain(conn: sqlite3.Connection) -> None:
    """Add allowed_email_domain to org_invites if missing."""
    logger.info("Migration 046: START org_invites allowlist domain")
    try:
        table_exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='org_invites'"
        ).fetchone()
        if not table_exists:
            logger.info("Migration 046: org_invites table missing, skipping")
            return
        cur = conn.execute("PRAGMA table_info(org_invites)")
        columns = {row[1] for row in cur.fetchall()}
        if "allowed_email_domain" not in columns:
            conn.execute("ALTER TABLE org_invites ADD COLUMN allowed_email_domain TEXT")
        conn.commit()
        logger.info("Migration 046: org_invites allowlist domain ensured")
    except _AUTHNZ_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Migration 046: failed to add org_invites allowlist domain: {}", exc)
        raise


def migration_047_create_user_config_overrides_table(conn: sqlite3.Connection) -> None:
    """Create the user_config_overrides table for profile preferences."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS user_config_overrides (
            user_id INTEGER NOT NULL,
            key TEXT NOT NULL,
            value_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_by INTEGER,
            updated_by INTEGER,
            PRIMARY KEY (user_id, key),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """
    )

    try:
        cursor = conn.execute("PRAGMA table_info(user_config_overrides)")
        columns = {row[1] for row in cursor.fetchall()}

        def add_col(name: str, decl: str):
            if name not in columns:
                conn.execute(f"ALTER TABLE user_config_overrides ADD COLUMN {decl}")
                columns.add(name)

        add_col("value_json", "value_json TEXT")
        add_col("created_at", "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        add_col("updated_at", "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        add_col("created_by", "created_by INTEGER")
        add_col("updated_by", "updated_by INTEGER")
    except _AUTHNZ_MIGRATIONS_NONCRITICAL_EXCEPTIONS:
        pass

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_user_config_overrides_user_id ON user_config_overrides(user_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_user_config_overrides_key ON user_config_overrides(key)"
    )
    conn.commit()
    logger.info("Migration 047: Created user_config_overrides table")


def rollback_047_drop_user_config_overrides(conn: sqlite3.Connection) -> None:
    """Drop user_config_overrides table."""
    conn.execute("DROP TABLE IF EXISTS user_config_overrides")
    conn.commit()
    logger.info("Rollback 047: Dropped user_config_overrides table")


def migration_048_create_org_team_config_overrides_table(conn: sqlite3.Connection) -> None:
    """Create org/team config overrides tables."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS org_config_overrides (
            org_id INTEGER NOT NULL,
            key TEXT NOT NULL,
            value_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_by INTEGER,
            updated_by INTEGER,
            PRIMARY KEY (org_id, key),
            FOREIGN KEY (org_id) REFERENCES organizations(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS team_config_overrides (
            team_id INTEGER NOT NULL,
            key TEXT NOT NULL,
            value_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_by INTEGER,
            updated_by INTEGER,
            PRIMARY KEY (team_id, key),
            FOREIGN KEY (team_id) REFERENCES teams(id) ON DELETE CASCADE
        )
        """
    )

    try:
        cursor = conn.execute("PRAGMA table_info(org_config_overrides)")
        columns = {row[1] for row in cursor.fetchall()}

        def add_col(name: str, decl: str):
            if name not in columns:
                conn.execute(f"ALTER TABLE org_config_overrides ADD COLUMN {decl}")
                columns.add(name)

        add_col("value_json", "value_json TEXT")
        add_col("created_at", "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        add_col("updated_at", "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        add_col("created_by", "created_by INTEGER")
        add_col("updated_by", "updated_by INTEGER")
    except _AUTHNZ_MIGRATIONS_NONCRITICAL_EXCEPTIONS:
        pass

    try:
        cursor = conn.execute("PRAGMA table_info(team_config_overrides)")
        columns = {row[1] for row in cursor.fetchall()}

        def add_col(name: str, decl: str):
            if name not in columns:
                conn.execute(f"ALTER TABLE team_config_overrides ADD COLUMN {decl}")
                columns.add(name)

        add_col("value_json", "value_json TEXT")
        add_col("created_at", "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        add_col("updated_at", "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        add_col("created_by", "created_by INTEGER")
        add_col("updated_by", "updated_by INTEGER")
    except _AUTHNZ_MIGRATIONS_NONCRITICAL_EXCEPTIONS:
        pass

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_org_config_overrides_org_id ON org_config_overrides(org_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_org_config_overrides_key ON org_config_overrides(key)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_team_config_overrides_team_id ON team_config_overrides(team_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_team_config_overrides_key ON team_config_overrides(key)"
    )
    conn.commit()
    logger.info("Migration 048: Created org/team config overrides tables")


def rollback_048_drop_org_team_config_overrides(conn: sqlite3.Connection) -> None:
    """Drop org/team config overrides tables."""
    conn.execute("DROP TABLE IF EXISTS team_config_overrides")
    conn.execute("DROP TABLE IF EXISTS org_config_overrides")
    conn.commit()
    logger.info("Rollback 048: Dropped org/team config overrides tables")


def migration_059_create_data_subject_requests_table(conn: sqlite3.Connection) -> None:
    """Create data_subject_requests table for authoritative DSR intake."""
    logger.info("Migration 059: START data_subject_requests table")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS data_subject_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            client_request_id TEXT NOT NULL UNIQUE,
            requester_identifier TEXT NOT NULL,
            resolved_user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
            request_type TEXT NOT NULL,
            status TEXT NOT NULL,
            selected_categories TEXT NOT NULL DEFAULT '[]',
            preview_summary TEXT NOT NULL DEFAULT '[]',
            coverage_metadata TEXT DEFAULT '{}',
            requested_by_user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
            requested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            notes TEXT
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_data_subject_requests_requester "
        "ON data_subject_requests(requester_identifier)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_data_subject_requests_resolved_user "
        "ON data_subject_requests(resolved_user_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_data_subject_requests_type "
        "ON data_subject_requests(request_type)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_data_subject_requests_status "
        "ON data_subject_requests(status)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_data_subject_requests_requested_at "
        "ON data_subject_requests(requested_at)"
    )
    conn.commit()
    logger.info("Migration 059: Created data_subject_requests table")


def rollback_059_drop_data_subject_requests_table(conn: sqlite3.Connection) -> None:
    """Rollback migration 059 by dropping data_subject_requests."""
    conn.execute("DROP TABLE IF EXISTS data_subject_requests")
    conn.commit()
    logger.info("Rollback 059: Dropped data_subject_requests table")


def migration_060_create_admin_monitoring_tables(conn: sqlite3.Connection) -> None:
    """Create admin monitoring control-plane tables for SQLite backends."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS admin_alert_rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            metric TEXT NOT NULL,
            operator TEXT NOT NULL,
            threshold REAL NOT NULL,
            duration_minutes INTEGER NOT NULL,
            severity TEXT NOT NULL,
            enabled INTEGER DEFAULT 1,
            created_by_user_id INTEGER,
            updated_by_user_id INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (created_by_user_id) REFERENCES users(id) ON DELETE SET NULL,
            FOREIGN KEY (updated_by_user_id) REFERENCES users(id) ON DELETE SET NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_admin_alert_rules_created_at ON admin_alert_rules(created_at)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_admin_alert_rules_metric_enabled "
        "ON admin_alert_rules(metric, enabled)"
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS admin_alert_state (
            alert_identity TEXT PRIMARY KEY,
            assigned_to_user_id INTEGER,
            snoozed_until TIMESTAMP,
            escalated_severity TEXT,
            acknowledged_at TIMESTAMP,
            dismissed_at TIMESTAMP,
            updated_by_user_id INTEGER,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (assigned_to_user_id) REFERENCES users(id) ON DELETE SET NULL,
            FOREIGN KEY (updated_by_user_id) REFERENCES users(id) ON DELETE SET NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_admin_alert_state_updated_at ON admin_alert_state(updated_at)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_admin_alert_state_assigned_user "
        "ON admin_alert_state(assigned_to_user_id)"
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS admin_alert_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_identity TEXT NOT NULL,
            action TEXT NOT NULL,
            actor_user_id INTEGER,
            details_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (actor_user_id) REFERENCES users(id) ON DELETE SET NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_admin_alert_events_identity_created "
        "ON admin_alert_events(alert_identity, created_at)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_admin_alert_events_created_at ON admin_alert_events(created_at)"
    )

    conn.commit()
    logger.info("Migration 060: Created admin monitoring tables")


def rollback_060_drop_admin_monitoring_tables(conn: sqlite3.Connection) -> None:
    """Rollback migration 060 by dropping admin monitoring tables."""
    conn.execute("DROP TABLE IF EXISTS admin_alert_events")
    conn.execute("DROP TABLE IF EXISTS admin_alert_state")
    conn.execute("DROP TABLE IF EXISTS admin_alert_rules")
    conn.commit()
    logger.info("Rollback 060: Dropped admin monitoring tables")


def migration_061_create_backup_schedule_tables(conn: sqlite3.Connection) -> None:
    """Create backup schedule persistence tables (SQLite)."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS backup_schedules (
            id TEXT PRIMARY KEY,
            dataset TEXT NOT NULL,
            target_user_id INTEGER,
            target_scope_key TEXT NOT NULL,
            frequency TEXT NOT NULL,
            time_of_day TEXT NOT NULL,
            timezone TEXT NOT NULL,
            anchor_day_of_week INTEGER,
            anchor_day_of_month INTEGER,
            retention_count INTEGER NOT NULL,
            is_paused INTEGER NOT NULL DEFAULT 0,
            created_by_user_id INTEGER,
            updated_by_user_id INTEGER,
            created_at TIMESTAMP NOT NULL,
            updated_at TIMESTAMP NOT NULL,
            next_run_at TIMESTAMP,
            last_run_at TIMESTAMP,
            last_status TEXT,
            last_job_id TEXT,
            last_error TEXT,
            deleted_at TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_backup_schedules_target_scope_active
        ON backup_schedules(target_scope_key)
        WHERE deleted_at IS NULL
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_backup_schedules_next_run_at "
        "ON backup_schedules(next_run_at)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_backup_schedules_deleted_at "
        "ON backup_schedules(deleted_at)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS backup_schedule_runs (
            id TEXT PRIMARY KEY,
            schedule_id TEXT NOT NULL,
            scheduled_for TIMESTAMP NOT NULL,
            run_slot_key TEXT NOT NULL UNIQUE,
            status TEXT NOT NULL,
            job_id TEXT,
            error TEXT,
            enqueued_at TIMESTAMP,
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            FOREIGN KEY (schedule_id) REFERENCES backup_schedules(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_backup_schedule_runs_schedule_id "
        "ON backup_schedule_runs(schedule_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_backup_schedule_runs_scheduled_for "
        "ON backup_schedule_runs(scheduled_for)"
    )

    conn.commit()
    logger.info("Migration 061: Created backup schedule tables")


def rollback_061_drop_backup_schedule_tables(conn: sqlite3.Connection) -> None:
    """Drop backup schedule persistence tables (SQLite rollback)."""
    conn.execute("DROP TABLE IF EXISTS backup_schedule_runs")
    conn.execute("DROP TABLE IF EXISTS backup_schedules")
    conn.commit()
    logger.info("Rollback 061: Dropped backup schedule tables")


def migration_067_create_maintenance_rotation_runs_table(conn: sqlite3.Connection) -> None:
    """Create maintenance rotation run persistence table (SQLite)."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS maintenance_rotation_runs (
            id TEXT PRIMARY KEY,
            mode TEXT NOT NULL,
            status TEXT NOT NULL,
            domain TEXT,
            queue TEXT,
            job_type TEXT,
            fields_json TEXT NOT NULL,
            "limit" INTEGER,
            affected_count INTEGER,
            requested_by_user_id INTEGER,
            requested_by_label TEXT,
            confirmation_recorded INTEGER NOT NULL DEFAULT 0,
            job_id TEXT,
            scope_summary TEXT NOT NULL,
            key_source TEXT NOT NULL,
            error_message TEXT,
            created_at TIMESTAMP NOT NULL,
            started_at TIMESTAMP,
            completed_at TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_maintenance_rotation_runs_active_execute
        ON maintenance_rotation_runs(mode)
        WHERE mode = 'execute' AND status IN ('queued', 'running')
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_maintenance_rotation_runs_created_at "
        "ON maintenance_rotation_runs(created_at)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_maintenance_rotation_runs_status "
        "ON maintenance_rotation_runs(status)"
    )
    conn.commit()
    logger.info("Migration 067: Created maintenance rotation runs table")


def rollback_067_drop_maintenance_rotation_runs_table(conn: sqlite3.Connection) -> None:
    """Drop maintenance rotation run persistence table (SQLite rollback)."""
    conn.execute("DROP TABLE IF EXISTS maintenance_rotation_runs")
    conn.commit()
    logger.info("Rollback 067: Dropped maintenance rotation runs table")


def migration_068_create_byok_validation_runs_table(conn: sqlite3.Connection) -> None:
    """Create BYOK validation run persistence table (SQLite)."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS byok_validation_runs (
            id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            org_id INTEGER,
            provider TEXT,
            keys_checked INTEGER,
            valid_count INTEGER,
            invalid_count INTEGER,
            error_count INTEGER,
            requested_by_user_id INTEGER,
            requested_by_label TEXT,
            job_id TEXT,
            scope_summary TEXT NOT NULL,
            error_message TEXT,
            created_at TIMESTAMP NOT NULL,
            started_at TIMESTAMP,
            completed_at TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_byok_validation_runs_active
        ON byok_validation_runs((1))
        WHERE status IN ('queued', 'running')
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_byok_validation_runs_created_at "
        "ON byok_validation_runs(created_at)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_byok_validation_runs_status "
        "ON byok_validation_runs(status)"
    )
    conn.commit()
    logger.info("Migration 068: Created BYOK validation runs table")


def rollback_068_drop_byok_validation_runs_table(conn: sqlite3.Connection) -> None:
    """Drop BYOK validation run persistence table (SQLite rollback)."""
    conn.execute("DROP TABLE IF EXISTS byok_validation_runs")
    conn.commit()
    logger.info("Rollback 068: Dropped BYOK validation runs table")


def migration_041_add_llm_provider_overrides(conn: sqlite3.Connection) -> None:
    """Add llm_provider_overrides table for runtime provider overrides."""
    logger.info("Migration 041: START llm_provider_overrides table")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS llm_provider_overrides (
            provider TEXT PRIMARY KEY,
            is_enabled INTEGER,
            allowed_models TEXT,
            config_json TEXT,
            secret_blob TEXT,
            api_key_hint TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_llm_provider_overrides_enabled ON llm_provider_overrides(is_enabled)"
    )
    conn.commit()
    logger.info("Migration 041: Created llm_provider_overrides table")


def rollback_041_drop_llm_provider_overrides(conn: sqlite3.Connection) -> None:
    """Rollback migration 041 by dropping llm_provider_overrides table."""
    conn.execute("DROP TABLE IF EXISTS llm_provider_overrides")
    conn.commit()
    logger.info("Rollback 041: Dropped llm_provider_overrides table")


def migration_042_create_org_budgets(conn: sqlite3.Connection) -> None:
    """Create org_budgets table and migrate legacy budget data."""
    logger.info("Migration 042: START org_budgets table")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS org_budgets (
            org_id INTEGER PRIMARY KEY,
            budgets_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (org_id) REFERENCES organizations(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_org_budgets_org ON org_budgets(org_id)")

    def _normalize_alert_thresholds(value: Any) -> Optional[dict[str, Any]]:
        if value is None:
            return None
        if isinstance(value, list):
            return {"global": value}
        if isinstance(value, dict):
            out: dict[str, Any] = {}
            if "global" in value:
                out["global"] = value.get("global")
            if "per_metric" in value:
                out["per_metric"] = value.get("per_metric")
            return out or None
        return None

    def _normalize_enforcement_mode(value: Any) -> Optional[dict[str, Any]]:
        if value is None:
            return None
        if isinstance(value, str):
            return {"global": value}
        if isinstance(value, dict):
            out: dict[str, Any] = {}
            if "global" in value:
                out["global"] = value.get("global")
            if "per_metric" in value:
                out["per_metric"] = value.get("per_metric")
            return out or None
        return None

    def _inflate_legacy_budgets(legacy: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key in ("budget_day_usd", "budget_month_usd", "budget_day_tokens", "budget_month_tokens"):
            if key in legacy:
                payload[key] = legacy[key]
        thresholds = _normalize_alert_thresholds(legacy.get("alert_thresholds"))
        if thresholds is not None:
            payload["alert_thresholds"] = thresholds
        enforcement = _normalize_enforcement_mode(legacy.get("enforcement_mode"))
        if enforcement is not None:
            payload["enforcement_mode"] = enforcement
        return payload

    if _sqlite_table_exists(conn, "org_subscriptions"):
        cur = conn.execute(
            "SELECT org_id, custom_limits_json FROM org_subscriptions WHERE custom_limits_json IS NOT NULL"
        )
        rows = cur.fetchall()
        for org_id, custom_limits_json in rows:
            if not custom_limits_json:
                continue
            try:
                data = json.loads(custom_limits_json)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(data, dict):
                continue
            legacy_budgets = data.get("budgets")
            if not isinstance(legacy_budgets, dict):
                continue
            payload = _inflate_legacy_budgets(legacy_budgets)
            if payload:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO org_budgets (org_id, budgets_json, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                    """,
                    (org_id, json.dumps(payload)),
                )
            data.pop("budgets", None)
            conn.execute(
                "UPDATE org_subscriptions SET custom_limits_json = ? WHERE org_id = ?",
                (json.dumps(data) if data else None, org_id),
            )

    conn.commit()
    logger.info("Migration 042: Created org_budgets table and migrated legacy budgets")


def rollback_042_drop_org_budgets(conn: sqlite3.Connection) -> None:
    """Rollback migration 042 by dropping org_budgets table."""
    conn.execute("DROP TABLE IF EXISTS org_budgets")
    conn.commit()
    logger.info("Rollback 042: Dropped org_budgets table")


def migration_043_create_retention_policy_overrides(conn: sqlite3.Connection) -> None:
    """Create retention_policy_overrides table for persisted retention settings."""
    logger.info("Migration 043: START retention_policy_overrides table")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS retention_policy_overrides (
            policy_key TEXT PRIMARY KEY,
            days INTEGER NOT NULL,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    logger.info("Migration 043: Created retention_policy_overrides table")


def rollback_043_drop_retention_policy_overrides(conn: sqlite3.Connection) -> None:
    """Rollback migration 043 by dropping retention_policy_overrides table."""
    conn.execute("DROP TABLE IF EXISTS retention_policy_overrides")
    conn.commit()
    logger.info("Rollback 043: Dropped retention_policy_overrides table")


def migration_044_add_api_keys_key_id(conn: sqlite3.Connection) -> None:
    """Add key_id column + index for api_keys (SQLite)."""
    try:
        cur = conn.execute("PRAGMA table_info(api_keys)")
        cols = {row[1] for row in cur.fetchall()}
        if "key_id" not in cols:
            conn.execute("ALTER TABLE api_keys ADD COLUMN key_id TEXT")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_api_keys_key_id ON api_keys(key_id)")
        conn.commit()
        logger.info("Migration 044: Added api_keys.key_id + index")
    except sqlite3.OperationalError as error:
        logger.warning(f"Migration 044 skipped or failed: {error}")


def rollback_044_drop_api_keys_key_id(conn: sqlite3.Connection) -> None:
    """Rollback migration 044 by dropping key_id index (column cannot be dropped in SQLite)."""
    conn.execute("DROP INDEX IF EXISTS idx_api_keys_key_id")
    conn.commit()
    logger.info("Rollback 044: Dropped idx_api_keys_key_id index")


def migration_045_add_users_created_by(conn: sqlite3.Connection) -> None:
    """Add created_by column to users table for admin-created accounts."""
    try:
        cur = conn.execute("PRAGMA table_info(users)")
        cols = {row[1] for row in cur.fetchall()}
        if "created_by" not in cols:
            conn.execute("ALTER TABLE users ADD COLUMN created_by INTEGER")
            logger.info("Migration 045: Added users.created_by column")
        conn.commit()
    except sqlite3.OperationalError as error:
        logger.warning(f"Migration 045 skipped or failed: {error}")


def migration_072_create_identity_providers_table(conn: sqlite3.Connection) -> None:
    """Create the identity_providers table for enterprise federation config."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS identity_providers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            slug TEXT NOT NULL,
            provider_type TEXT NOT NULL DEFAULT 'oidc',
            owner_scope_type TEXT NOT NULL DEFAULT 'global',
            owner_scope_id INTEGER,
            enabled INTEGER NOT NULL DEFAULT 0,
            display_name TEXT,
            issuer TEXT NOT NULL,
            discovery_url TEXT,
            authorization_url TEXT,
            token_url TEXT,
            jwks_url TEXT,
            client_id TEXT,
            client_secret_ref TEXT,
            claim_mapping_json TEXT NOT NULL DEFAULT '{}',
            provisioning_policy_json TEXT NOT NULL DEFAULT '{}',
            created_by INTEGER,
            updated_by INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    try:
        cur = conn.execute("PRAGMA table_info(identity_providers)")
        cols = {row[1] for row in cur.fetchall()}

        def add_col(name: str, decl: str) -> None:
            if name not in cols:
                conn.execute(f"ALTER TABLE identity_providers ADD COLUMN {decl}")
                cols.add(name)

        add_col("provider_type", "provider_type TEXT NOT NULL DEFAULT 'oidc'")
        add_col("owner_scope_type", "owner_scope_type TEXT NOT NULL DEFAULT 'global'")
        add_col("owner_scope_id", "owner_scope_id INTEGER")
        add_col("enabled", "enabled INTEGER NOT NULL DEFAULT 0")
        add_col("display_name", "display_name TEXT")
        add_col("issuer", "issuer TEXT")
        add_col("discovery_url", "discovery_url TEXT")
        add_col("authorization_url", "authorization_url TEXT")
        add_col("token_url", "token_url TEXT")
        add_col("jwks_url", "jwks_url TEXT")
        add_col("client_id", "client_id TEXT")
        add_col("client_secret_ref", "client_secret_ref TEXT")
        add_col("claim_mapping_json", "claim_mapping_json TEXT NOT NULL DEFAULT '{}'")
        add_col("provisioning_policy_json", "provisioning_policy_json TEXT NOT NULL DEFAULT '{}'")
        add_col("created_by", "created_by INTEGER")
        add_col("updated_by", "updated_by INTEGER")
        add_col("created_at", "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        add_col("updated_at", "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
    except _AUTHNZ_MIGRATIONS_NONCRITICAL_EXCEPTIONS:
        pass

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_identity_providers_scope "
        "ON identity_providers(owner_scope_type, owner_scope_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_identity_providers_enabled "
        "ON identity_providers(enabled)"
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_identity_providers_scope_slug "
        "ON identity_providers(slug, owner_scope_type, COALESCE(owner_scope_id, 0))"
    )
    conn.commit()
    logger.info("Migration 072: Created identity_providers table")


def rollback_072_drop_identity_providers_table(conn: sqlite3.Connection) -> None:
    """Rollback migration 072 by dropping the identity_providers table."""
    conn.execute("DROP TABLE IF EXISTS identity_providers")
    conn.commit()
    logger.info("Rollback 072: Dropped identity_providers table")


def migration_073_create_federated_identities_table(conn: sqlite3.Connection) -> None:
    """Create the federated_identities table for local user links."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS federated_identities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            identity_provider_id INTEGER NOT NULL,
            external_subject TEXT NOT NULL,
            user_id INTEGER NOT NULL,
            external_username TEXT,
            external_email TEXT,
            last_claims_hash TEXT,
            last_seen_at TIMESTAMP,
            status TEXT NOT NULL DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (identity_provider_id) REFERENCES identity_providers(id) ON DELETE CASCADE,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """
    )

    try:
        cur = conn.execute("PRAGMA table_info(federated_identities)")
        cols = {row[1] for row in cur.fetchall()}

        def add_col(name: str, decl: str) -> None:
            if name not in cols:
                conn.execute(f"ALTER TABLE federated_identities ADD COLUMN {decl}")
                cols.add(name)

        add_col("external_username", "external_username TEXT")
        add_col("external_email", "external_email TEXT")
        add_col("last_claims_hash", "last_claims_hash TEXT")
        add_col("last_seen_at", "last_seen_at TIMESTAMP")
        add_col("status", "status TEXT NOT NULL DEFAULT 'active'")
        add_col("created_at", "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        add_col("updated_at", "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
    except _AUTHNZ_MIGRATIONS_NONCRITICAL_EXCEPTIONS:
        pass

    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_federated_identities_provider_subject "
        "ON federated_identities(identity_provider_id, external_subject)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_federated_identities_user_id "
        "ON federated_identities(user_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_federated_identities_provider_id "
        "ON federated_identities(identity_provider_id)"
    )
    conn.commit()
    logger.info("Migration 073: Created federated_identities table")


def rollback_073_drop_federated_identities_table(conn: sqlite3.Connection) -> None:
    """Rollback migration 073 by dropping the federated_identities table."""
    conn.execute("DROP TABLE IF EXISTS federated_identities")
    conn.commit()
    logger.info("Rollback 073: Dropped federated_identities table")


def migration_074_create_federated_managed_grants_table(conn: sqlite3.Connection) -> None:
    """Create the federated_managed_grants table for safe grant/revoke provenance."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS federated_managed_grants (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            identity_provider_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            grant_kind TEXT NOT NULL,
            target_ref TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (identity_provider_id) REFERENCES identity_providers(id) ON DELETE CASCADE,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """
    )

    try:
        cur = conn.execute("PRAGMA table_info(federated_managed_grants)")
        cols = {row[1] for row in cur.fetchall()}

        def add_col(name: str, decl: str) -> None:
            if name not in cols:
                conn.execute(f"ALTER TABLE federated_managed_grants ADD COLUMN {decl}")
                cols.add(name)

        add_col("grant_kind", "grant_kind TEXT NOT NULL DEFAULT 'org'")
        add_col("target_ref", "target_ref TEXT NOT NULL DEFAULT ''")
        add_col("created_at", "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        add_col("updated_at", "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
    except _AUTHNZ_MIGRATIONS_NONCRITICAL_EXCEPTIONS:
        pass

    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_federated_managed_grants_target "
        "ON federated_managed_grants(identity_provider_id, user_id, grant_kind, target_ref)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_federated_managed_grants_user_id "
        "ON federated_managed_grants(user_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_federated_managed_grants_provider_id "
        "ON federated_managed_grants(identity_provider_id)"
    )
    conn.commit()
    logger.info("Migration 074: Created federated_managed_grants table")


def rollback_074_drop_federated_managed_grants_table(conn: sqlite3.Connection) -> None:
    """Rollback migration 074 by dropping the federated_managed_grants table."""
    conn.execute("DROP TABLE IF EXISTS federated_managed_grants")
    conn.commit()
    logger.info("Rollback 074: Dropped federated_managed_grants table")


def migration_075_create_secret_backends_table(conn: sqlite3.Connection) -> None:
    """Create the secret_backends table for backend metadata and capabilities."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS secret_backends (
            name TEXT PRIMARY KEY,
            display_name TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'enabled',
            capabilities_json TEXT NOT NULL DEFAULT '{}',
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    try:
        cur = conn.execute("PRAGMA table_info(secret_backends)")
        cols = {row[1] for row in cur.fetchall()}

        def add_col(name: str, decl: str) -> None:
            if name not in cols:
                conn.execute(f"ALTER TABLE secret_backends ADD COLUMN {decl}")
                cols.add(name)

        add_col("display_name", "display_name TEXT")
        add_col("status", "status TEXT NOT NULL DEFAULT 'enabled'")
        add_col("capabilities_json", "capabilities_json TEXT NOT NULL DEFAULT '{}'")
        add_col("metadata_json", "metadata_json TEXT NOT NULL DEFAULT '{}'")
        add_col("created_at", "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        add_col("updated_at", "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
    except _AUTHNZ_MIGRATIONS_NONCRITICAL_EXCEPTIONS:
        pass

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_secret_backends_status "
        "ON secret_backends(status)"
    )
    conn.commit()
    logger.info("Migration 075: Created secret_backends table")


def rollback_075_drop_secret_backends_table(conn: sqlite3.Connection) -> None:
    """Rollback migration 075 by dropping the secret_backends table."""
    conn.execute("DROP TABLE IF EXISTS secret_backends")
    conn.commit()
    logger.info("Rollback 075: Dropped secret_backends table")


def migration_076_create_managed_secret_refs_table(conn: sqlite3.Connection) -> None:
    """Create the managed_secret_refs table for logical secret references."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS managed_secret_refs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            backend_name TEXT NOT NULL,
            owner_scope_type TEXT NOT NULL,
            owner_scope_id INTEGER NOT NULL,
            provider_key TEXT NOT NULL,
            backend_ref TEXT NULL,
            display_name TEXT NULL,
            status TEXT NOT NULL DEFAULT 'active',
            metadata_json TEXT NOT NULL DEFAULT '{}',
            last_resolved_at TIMESTAMP NULL,
            expires_at TIMESTAMP NULL,
            created_by INTEGER NULL,
            updated_by INTEGER NULL,
            revoked_by INTEGER NULL,
            revoked_at TIMESTAMP NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (backend_name) REFERENCES secret_backends(name) ON DELETE RESTRICT
        )
        """
    )

    try:
        cur = conn.execute("PRAGMA table_info(managed_secret_refs)")
        cols = {row[1] for row in cur.fetchall()}

        def add_col(name: str, decl: str) -> None:
            if name not in cols:
                conn.execute(f"ALTER TABLE managed_secret_refs ADD COLUMN {decl}")
                cols.add(name)

        add_col("backend_ref", "backend_ref TEXT")
        add_col("display_name", "display_name TEXT")
        add_col("status", "status TEXT NOT NULL DEFAULT 'active'")
        add_col("metadata_json", "metadata_json TEXT NOT NULL DEFAULT '{}'")
        add_col("last_resolved_at", "last_resolved_at TIMESTAMP")
        add_col("expires_at", "expires_at TIMESTAMP")
        add_col("created_by", "created_by INTEGER")
        add_col("updated_by", "updated_by INTEGER")
        add_col("revoked_by", "revoked_by INTEGER")
        add_col("revoked_at", "revoked_at TIMESTAMP")
        add_col("created_at", "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        add_col("updated_at", "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
    except _AUTHNZ_MIGRATIONS_NONCRITICAL_EXCEPTIONS:
        pass

    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_managed_secret_refs_scope_provider "
        "ON managed_secret_refs(backend_name, owner_scope_type, owner_scope_id, provider_key)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_managed_secret_refs_scope "
        "ON managed_secret_refs(owner_scope_type, owner_scope_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_managed_secret_refs_status "
        "ON managed_secret_refs(status)"
    )
    conn.commit()
    logger.info("Migration 076: Created managed_secret_refs table")


def rollback_076_drop_managed_secret_refs_table(conn: sqlite3.Connection) -> None:
    """Rollback migration 076 by dropping the managed_secret_refs table."""
    conn.execute("DROP TABLE IF EXISTS managed_secret_refs")
    conn.commit()
    logger.info("Rollback 076: Dropped managed_secret_refs table")


def migration_077_create_sharing_tables(conn: sqlite3.Connection) -> None:
    """Create tables for shared workspaces and share tokens."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS shared_workspaces (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            workspace_id     TEXT NOT NULL,
            owner_user_id    INTEGER NOT NULL,
            share_scope_type TEXT NOT NULL DEFAULT 'team'
                CHECK (share_scope_type IN ('team', 'org')),
            share_scope_id   INTEGER NOT NULL,
            access_level     TEXT NOT NULL DEFAULT 'view_chat'
                CHECK (access_level IN ('view_chat', 'view_chat_add', 'full_edit')),
            allow_clone      INTEGER NOT NULL DEFAULT 1,
            created_by       INTEGER NOT NULL,
            created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            revoked_at       TIMESTAMP,
            FOREIGN KEY (owner_user_id) REFERENCES users(id),
            FOREIGN KEY (created_by) REFERENCES users(id),
            UNIQUE(workspace_id, owner_user_id, share_scope_type, share_scope_id)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_shared_ws_owner ON shared_workspaces(owner_user_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_shared_ws_scope ON shared_workspaces(share_scope_type, share_scope_id)")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS share_tokens (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            token_hash      TEXT UNIQUE NOT NULL,
            token_prefix    TEXT NOT NULL,
            resource_type   TEXT NOT NULL
                CHECK (resource_type IN ('chatbook', 'workspace')),
            resource_id     TEXT NOT NULL,
            owner_user_id   INTEGER NOT NULL,
            access_level    TEXT NOT NULL DEFAULT 'view_chat',
            allow_clone     INTEGER NOT NULL DEFAULT 1,
            password_hash   TEXT,
            max_uses        INTEGER,
            use_count       INTEGER NOT NULL DEFAULT 0,
            expires_at      TIMESTAMP,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            revoked_at      TIMESTAMP,
            FOREIGN KEY (owner_user_id) REFERENCES users(id)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_share_tokens_prefix ON share_tokens(token_prefix)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_share_tokens_owner ON share_tokens(owner_user_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_share_tokens_resource ON share_tokens(resource_type, resource_id)")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS share_audit_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type      TEXT NOT NULL,
            actor_user_id   INTEGER,
            resource_type   TEXT NOT NULL,
            resource_id     TEXT NOT NULL,
            owner_user_id   INTEGER NOT NULL,
            share_id        INTEGER,
            token_id        INTEGER,
            metadata_json   TEXT DEFAULT '{}',
            ip_address      TEXT,
            user_agent      TEXT,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_share_audit_created ON share_audit_log(created_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_share_audit_owner ON share_audit_log(owner_user_id)")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS sharing_config (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            scope_type      TEXT NOT NULL DEFAULT 'global'
                CHECK (scope_type IN ('global', 'org', 'team')),
            scope_id        INTEGER,
            config_key      TEXT NOT NULL,
            config_value    TEXT NOT NULL,
            updated_by      INTEGER,
            updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(scope_type, scope_id, config_key)
        )
    """)

    conn.commit()
    logger.info("Migration 077: Created sharing tables (shared_workspaces, share_tokens, share_audit_log, sharing_config)")


def migration_078_add_governance_pack_source_provenance(conn: sqlite3.Connection) -> None:
    """Add governance-pack source provenance fields and prepared candidate storage."""

    governance_pack_columns = {
        str(row[1]) for row in conn.execute("PRAGMA table_info(mcp_governance_packs)").fetchall()
    }
    if "source_type" not in governance_pack_columns:
        conn.execute("ALTER TABLE mcp_governance_packs ADD COLUMN source_type TEXT")
    if "source_location" not in governance_pack_columns:
        conn.execute("ALTER TABLE mcp_governance_packs ADD COLUMN source_location TEXT")
    if "source_ref_requested" not in governance_pack_columns:
        conn.execute("ALTER TABLE mcp_governance_packs ADD COLUMN source_ref_requested TEXT")
    if "source_ref_kind" not in governance_pack_columns:
        conn.execute("ALTER TABLE mcp_governance_packs ADD COLUMN source_ref_kind TEXT")
    if "source_subpath" not in governance_pack_columns:
        conn.execute("ALTER TABLE mcp_governance_packs ADD COLUMN source_subpath TEXT")
    if "source_commit_resolved" not in governance_pack_columns:
        conn.execute("ALTER TABLE mcp_governance_packs ADD COLUMN source_commit_resolved TEXT")
    if "pack_content_digest" not in governance_pack_columns:
        conn.execute("ALTER TABLE mcp_governance_packs ADD COLUMN pack_content_digest TEXT")
    if "source_verified" not in governance_pack_columns:
        conn.execute("ALTER TABLE mcp_governance_packs ADD COLUMN source_verified INTEGER")
    if "source_verification_mode" not in governance_pack_columns:
        conn.execute("ALTER TABLE mcp_governance_packs ADD COLUMN source_verification_mode TEXT")
    if "signer_fingerprint" not in governance_pack_columns:
        conn.execute("ALTER TABLE mcp_governance_packs ADD COLUMN signer_fingerprint TEXT")
    if "signer_identity" not in governance_pack_columns:
        conn.execute("ALTER TABLE mcp_governance_packs ADD COLUMN signer_identity TEXT")
    if "verified_object_type" not in governance_pack_columns:
        conn.execute("ALTER TABLE mcp_governance_packs ADD COLUMN verified_object_type TEXT")
    if "verification_result_code" not in governance_pack_columns:
        conn.execute("ALTER TABLE mcp_governance_packs ADD COLUMN verification_result_code TEXT")
    if "verification_warning_code" not in governance_pack_columns:
        conn.execute("ALTER TABLE mcp_governance_packs ADD COLUMN verification_warning_code TEXT")
    if "source_fetched_at" not in governance_pack_columns:
        conn.execute("ALTER TABLE mcp_governance_packs ADD COLUMN source_fetched_at TIMESTAMP")
    if "fetched_by" not in governance_pack_columns:
        conn.execute("ALTER TABLE mcp_governance_packs ADD COLUMN fetched_by INTEGER")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS mcp_governance_pack_source_candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_type TEXT NOT NULL,
            source_location TEXT NOT NULL,
            source_ref_requested TEXT,
            source_ref_kind TEXT,
            source_subpath TEXT,
            source_commit_resolved TEXT,
            pack_content_digest TEXT NOT NULL,
            pack_document_json TEXT NOT NULL DEFAULT '{}',
            source_verified INTEGER,
            source_verification_mode TEXT,
            signer_fingerprint TEXT,
            signer_identity TEXT,
            verified_object_type TEXT,
            verification_result_code TEXT,
            verification_warning_code TEXT,
            source_fetched_at TIMESTAMP,
            fetched_by INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    source_candidate_columns = {
        str(row[1]) for row in conn.execute("PRAGMA table_info(mcp_governance_pack_source_candidates)").fetchall()
    }
    if "source_ref_kind" not in source_candidate_columns:
        conn.execute("ALTER TABLE mcp_governance_pack_source_candidates ADD COLUMN source_ref_kind TEXT")
    if "pack_document_json" not in source_candidate_columns:
        conn.execute(
            "ALTER TABLE mcp_governance_pack_source_candidates ADD COLUMN pack_document_json TEXT NOT NULL DEFAULT '{}'"
        )
    if "signer_fingerprint" not in source_candidate_columns:
        conn.execute("ALTER TABLE mcp_governance_pack_source_candidates ADD COLUMN signer_fingerprint TEXT")
    if "signer_identity" not in source_candidate_columns:
        conn.execute("ALTER TABLE mcp_governance_pack_source_candidates ADD COLUMN signer_identity TEXT")
    if "verified_object_type" not in source_candidate_columns:
        conn.execute("ALTER TABLE mcp_governance_pack_source_candidates ADD COLUMN verified_object_type TEXT")
    if "verification_result_code" not in source_candidate_columns:
        conn.execute("ALTER TABLE mcp_governance_pack_source_candidates ADD COLUMN verification_result_code TEXT")
    if "verification_warning_code" not in source_candidate_columns:
        conn.execute("ALTER TABLE mcp_governance_pack_source_candidates ADD COLUMN verification_warning_code TEXT")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_mcp_governance_pack_source_candidates_source "
        "ON mcp_governance_pack_source_candidates(source_type, source_location)"
    )

    conn.commit()
    logger.info("Migration 078: Added governance-pack source provenance schema")


def migration_079_add_governance_pack_trust_policy(conn: sqlite3.Connection) -> None:
    """Add deployment-wide governance-pack trust-policy storage."""

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS mcp_governance_pack_trust_policy (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            policy_document_json TEXT NOT NULL DEFAULT '{}',
            updated_by INTEGER,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    conn.commit()
    logger.info("Migration 079: Added governance-pack trust policy schema")


def migration_080_create_admin_webhooks_tables(conn: sqlite3.Connection) -> None:
    """Create admin_webhooks and admin_webhooks_delivery_log tables."""

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS admin_webhooks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL,
            secret_encrypted TEXT NOT NULL,
            secret_key_id TEXT,
            event_types TEXT NOT NULL DEFAULT '[]',
            description TEXT NOT NULL DEFAULT '',
            active INTEGER NOT NULL DEFAULT 1,
            retry_count INTEGER NOT NULL DEFAULT 3 CHECK (retry_count >= 0 AND retry_count <= 10),
            timeout_seconds INTEGER NOT NULL DEFAULT 10 CHECK (timeout_seconds >= 1 AND timeout_seconds <= 120),
            created_by INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS admin_webhooks_delivery_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            webhook_id INTEGER NOT NULL REFERENCES admin_webhooks(id) ON DELETE CASCADE,
            event_type TEXT NOT NULL,
            payload_json TEXT NOT NULL DEFAULT '{}',
            signature TEXT NOT NULL,
            status_code INTEGER,
            response_body TEXT,
            latency_ms INTEGER,
            retry_attempt INTEGER NOT NULL DEFAULT 0,
            error_message TEXT,
            delivered_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_admin_webhooks_delivery_log_webhook_id "
        "ON admin_webhooks_delivery_log(webhook_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_admin_webhooks_delivery_log_created_at "
        "ON admin_webhooks_delivery_log(created_at)"
    )

    conn.commit()
    logger.info("Migration 080: Created admin_webhooks and delivery_log tables")


def migration_081_create_admin_dependency_health_history(conn: sqlite3.Connection) -> None:
    """Create time-series table for dependency health probe results."""

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS admin_dependency_health_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            service_name TEXT NOT NULL,
            status TEXT NOT NULL,
            latency_ms INTEGER,
            error_message TEXT,
            checked_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_admin_dep_health_service_checked "
        "ON admin_dependency_health_history(service_name, checked_at DESC)"
    )

    conn.commit()
    logger.info("Migration 081: Created admin_dependency_health_history table")


def migration_082_harden_admin_webhooks_and_create_admin_settings(conn: sqlite3.Connection) -> None:
    """Create admin_settings and migrate admin_webhooks secrets to encrypted storage."""

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS admin_settings (
            setting_key TEXT PRIMARY KEY,
            value_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )

    if not _sqlite_table_exists(conn, "admin_webhooks"):
        conn.commit()
        logger.info("Migration 082: Created admin_settings table")
        return

    table_sql_row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'admin_webhooks'"
    ).fetchone()
    table_sql = str(table_sql_row[0] if table_sql_row and table_sql_row[0] else "")
    columns = {
        row[1]
        for row in conn.execute("PRAGMA table_info(admin_webhooks)").fetchall()
    }
    needs_rebuild = (
        "secret" in columns
        or "secret_encrypted" not in columns
        or "secret_key_id" not in columns
        or "CHECK (retry_count >=" not in table_sql
        or "CHECK (timeout_seconds >=" not in table_sql
    )
    if not needs_rebuild:
        conn.commit()
        logger.info("Migration 082: Admin webhook schema already hardened")
        return

    previous_row_factory = conn.row_factory
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT * FROM admin_webhooks").fetchall()
    finally:
        conn.row_factory = previous_row_factory

    migrated_rows: list[tuple[Any, ...]] = []
    for row in rows:
        secret_encrypted = row["secret_encrypted"] if "secret_encrypted" in row.keys() else None
        secret_key_id = row["secret_key_id"] if "secret_key_id" in row.keys() else None
        if not secret_encrypted:
            plaintext_secret = row["secret"] if "secret" in row.keys() else None
            if not plaintext_secret:
                raise ValueError(f"Admin webhook {row['id']} is missing a secret for migration")
            encrypted = encrypt_admin_webhook_secret(str(plaintext_secret))
            secret_encrypted = encrypted.encrypted_blob
            secret_key_id = encrypted.key_id

        migrated_rows.append(
            (
                row["id"],
                row["url"],
                secret_encrypted,
                secret_key_id,
                row["event_types"],
                row["description"],
                row["active"],
                row["retry_count"],
                row["timeout_seconds"],
                row["created_by"],
                row["created_at"],
                row["updated_at"],
            )
        )

    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        conn.execute(
            """
            CREATE TABLE admin_webhooks_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT NOT NULL,
                secret_encrypted TEXT NOT NULL,
                secret_key_id TEXT,
                event_types TEXT NOT NULL DEFAULT '[]',
                description TEXT NOT NULL DEFAULT '',
                active INTEGER NOT NULL DEFAULT 1,
                retry_count INTEGER NOT NULL DEFAULT 3 CHECK (retry_count >= 0 AND retry_count <= 10),
                timeout_seconds INTEGER NOT NULL DEFAULT 10 CHECK (timeout_seconds >= 1 AND timeout_seconds <= 120),
                created_by INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        if migrated_rows:
            conn.executemany(
                """
                INSERT INTO admin_webhooks_new (
                    id, url, secret_encrypted, secret_key_id, event_types, description,
                    active, retry_count, timeout_seconds, created_by, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                migrated_rows,
            )
        conn.execute("DROP TABLE admin_webhooks")
        conn.execute("ALTER TABLE admin_webhooks_new RENAME TO admin_webhooks")
    finally:
        conn.execute("PRAGMA foreign_keys = ON")

    conn.commit()
    logger.info("Migration 082: Hardened admin_webhooks schema and created admin_settings")


def migration_083_create_org_stt_settings(conn: sqlite3.Connection) -> None:
    """Create org_stt_settings table for org-scoped STT policy."""
    logger.info("Migration 083: START org_stt_settings table")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS org_stt_settings (
            org_id INTEGER PRIMARY KEY,
            delete_audio_after_success INTEGER NOT NULL DEFAULT 1,
            audio_retention_hours REAL NOT NULL DEFAULT 0.0,
            redact_pii INTEGER NOT NULL DEFAULT 0,
            allow_unredacted_partials INTEGER NOT NULL DEFAULT 0,
            redact_categories_json TEXT NOT NULL DEFAULT '[]',
            updated_by INTEGER,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (org_id) REFERENCES organizations(id) ON DELETE CASCADE,
            FOREIGN KEY (updated_by) REFERENCES users(id) ON DELETE SET NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_org_stt_settings_updated_by ON org_stt_settings(updated_by)")
    conn.commit()
    logger.info("Migration 083: Created org_stt_settings table")


def rollback_083_drop_org_stt_settings(conn: sqlite3.Connection) -> None:
    """Rollback migration 083 by dropping org_stt_settings."""
    conn.execute("DROP TABLE IF EXISTS org_stt_settings")
    conn.commit()
    logger.info("Rollback 083: Dropped org_stt_settings table")


def rollback_081_drop_admin_dependency_health_history(conn: sqlite3.Connection) -> None:
    """Rollback migration 081."""
    conn.execute("DROP TABLE IF EXISTS admin_dependency_health_history")
    conn.commit()
    logger.info("Rollback 081: Dropped admin_dependency_health_history table")


def rollback_080_drop_admin_webhooks_tables(conn: sqlite3.Connection) -> None:
    """Rollback migration 080 by dropping admin webhook tables."""
    conn.execute("DROP TABLE IF EXISTS admin_webhooks_delivery_log")
    conn.execute("DROP TABLE IF EXISTS admin_webhooks")
    conn.commit()
    logger.info("Rollback 080: Dropped admin webhook tables")


def rollback_077_drop_sharing_tables(conn: sqlite3.Connection) -> None:
    """Rollback migration 077 by dropping sharing tables."""
    conn.execute("DROP TABLE IF EXISTS sharing_config")
    conn.execute("DROP TABLE IF EXISTS share_audit_log")
    conn.execute("DROP TABLE IF EXISTS share_tokens")
    conn.execute("DROP TABLE IF EXISTS shared_workspaces")
    conn.commit()
    logger.info("Rollback 077: Dropped sharing tables")


#######################################################################################################################
#
# Migration Registry
#

def get_authnz_migrations() -> list[Migration]:
    """Get all AuthNZ migrations in order"""
    return [
        Migration(1, "Create users table", migration_001_create_users_table, rollback_001_drop_users_table),
        Migration(2, "Create sessions table", migration_002_create_sessions_table, rollback_002_drop_sessions_table),
        Migration(3, "Create api_keys table", migration_003_create_api_keys_table, rollback_003_drop_api_keys_table),
        Migration(4, "Create api_key_audit_log table", migration_004_create_api_key_audit_log),
        Migration(5, "Create rate_limits table", migration_005_create_rate_limits_table),
        Migration(6, "Create registration_codes table", migration_006_create_registration_codes_table),
        Migration(7, "Create audit_logs table", migration_007_create_audit_logs_table),
        Migration(8, "Add password_history table", migration_008_add_password_history_table),
        Migration(9, "Add session encryption columns", migration_009_add_session_encryption_columns),
        Migration(10, "Add 2FA columns to users", migration_010_add_2fa_columns),
        Migration(11, "Enhanced auth tables + uuid", migration_011_add_enhanced_auth_tables),
        Migration(12, "Create RBAC core tables", migration_012_create_rbac_tables),
        Migration(13, "Create RBAC limits and usage tables", migration_013_create_rbac_limits_and_usage),
        Migration(14, "Seed default roles and permissions", migration_014_seed_roles_permissions),
        Migration(15, "Create LLM usage tables", migration_015_create_llm_usage_tables),
        Migration(16, "Create organizations and teams tables", migration_016_create_orgs_teams),
        Migration(17, "Extend api_keys with virtual key fields", migration_017_extend_api_keys_virtual),
        Migration(18, "Add indexes for usage tables", migration_018_add_usage_indexes),
        Migration(19, "Add request_id to usage_log", migration_019_usage_log_add_request_id),
        Migration(20, "Add bytes_in to usage_log", migration_020_usage_log_add_bytes_in),
        Migration(21, "Add bytes_in_total to usage_daily", migration_021_usage_daily_add_bytes_in_total),
        Migration(22, "Create tool catalogs tables", migration_022_create_tool_catalogs),
        Migration(23, "Create virtual key counters tables", migration_023_create_virtual_key_counters),
        Migration(24, "Ensure api_keys status column before index", migration_024_ensure_api_keys_status_column),
        Migration(25, "Backfill team_members added_at column", migration_025_team_members_added_at),
        Migration(
            26,
            "Create privilege_snapshots table",
            migration_026_create_privilege_snapshots_table,
            rollback_026_drop_privilege_snapshots_table,
        ),
        Migration(
            27,
            "Ensure session revocation columns",
            migration_027_add_session_revocation_columns,
        ),
        Migration(28, "Create org_invites table", migration_028_create_org_invites, rollback_028_drop_org_invites_table),
        Migration(
            29,
            "Create org_invite_redemptions table",
            migration_029_create_org_invite_redemptions,
            rollback_029_drop_org_invite_redemptions_table,
        ),
        Migration(
            30,
            "Create subscription_plans table",
            migration_030_create_subscription_plans,
            rollback_retired_billing_schema_noop,
        ),
        Migration(
            31,
            "Create org_subscriptions table",
            migration_031_create_org_subscriptions,
            rollback_retired_billing_schema_noop,
        ),
        Migration(
            32,
            "Create stripe_webhook_events table",
            migration_032_create_stripe_webhook_events,
            rollback_retired_billing_schema_noop,
        ),
        Migration(
            33,
            "Create payment_history table",
            migration_033_create_payment_history,
            rollback_retired_billing_schema_noop,
        ),
        Migration(
            34,
            "Create billing_audit_log table",
            migration_034_create_billing_audit_log,
            rollback_retired_billing_schema_noop,
        ),
        Migration(
            35,
            "Backfill storage_mb limits",
            migration_035_backfill_storage_mb_limits,
        ),
        Migration(
            36,
            "Create user_provider_secrets table",
            migration_036_create_user_provider_secrets,
            rollback_036_drop_user_provider_secrets_table,
        ),
        Migration(
            37,
            "Create org_provider_secrets table",
            migration_037_create_org_provider_secrets,
            rollback_037_drop_org_provider_secrets_table,
        ),
        Migration(
            38,
            "Add org_provider_secrets cleanup triggers",
            migration_038_add_org_provider_secrets_cleanup_triggers,
            rollback_038_drop_org_provider_secrets_cleanup_triggers,
        ),
        Migration(
            39,
            "Ensure users storage columns",
            migration_039_ensure_user_storage_columns,
        ),
        Migration(
            40,
            "Extend registration_codes for org invites",
            migration_040_extend_registration_codes_for_org_invites,
        ),
        Migration(
            41,
            "Add llm_provider_overrides table",
            migration_041_add_llm_provider_overrides,
            rollback_041_drop_llm_provider_overrides,
        ),
        Migration(
            42,
            "Create org_budgets table",
            migration_042_create_org_budgets,
            rollback_042_drop_org_budgets,
        ),
        Migration(
            43,
            "Create retention_policy_overrides table",
            migration_043_create_retention_policy_overrides,
            rollback_043_drop_retention_policy_overrides,
        ),
        Migration(
            44,
            "Add api_keys.key_id column",
            migration_044_add_api_keys_key_id,
            rollback_044_drop_api_keys_key_id,
        ),
        Migration(
            45,
            "Add users.created_by column",
            migration_045_add_users_created_by,
        ),
        Migration(
            46,
            "Add org_invites allowed_email_domain",
            migration_046_add_org_invite_allowlist_domain,
        ),
        Migration(
            47,
            "Create user_config_overrides table",
            migration_047_create_user_config_overrides_table,
            rollback_047_drop_user_config_overrides,
        ),
        Migration(
            48,
            "Create org/team config overrides tables",
            migration_048_create_org_team_config_overrides_table,
            rollback_048_drop_org_team_config_overrides,
        ),
        Migration(
            49,
            "Add llm_usage_log key_id + ts index",
            migration_049_add_llm_usage_log_key_ts_index,
        ),
        Migration(
            50,
            "Create generated_files table",
            migration_050_create_generated_files_table,
            rollback_050_drop_generated_files_table,
        ),
        Migration(
            51,
            "Create storage_quotas table",
            migration_051_create_storage_quotas_table,
            rollback_051_drop_storage_quotas_table,
        ),
        Migration(
            52,
            "Create org/team role permission tables",
            migration_052_create_org_team_role_permissions,
        ),
        Migration(
            53,
            "Create BYOK OAuth state table",
            migration_053_create_byok_oauth_state,
            rollback_053_drop_byok_oauth_state,
        ),
        Migration(
            54,
            "Add llm_usage_log router analytics columns and indexes",
            migration_054_add_llm_usage_log_router_analytics_columns,
        ),
        Migration(
            55,
            "Create MCP Hub tables",
            migration_055_create_mcp_hub_tables,
        ),
        Migration(
            56,
            "Create MCP Hub policy tables",
            migration_056_create_mcp_hub_policy_tables,
        ),
        Migration(
            57,
            "Add consumable MCP approval decision columns",
            migration_057_add_consumable_mcp_approval_decision_columns,
        ),
        Migration(
            58,
            "Harden MCP policy override schema",
            migration_058_harden_mcp_policy_override_schema,
        ),
        Migration(
            59,
            "Create data subject requests table",
            migration_059_create_data_subject_requests_table,
            rollback_059_drop_data_subject_requests_table,
        ),
        Migration(
            60,
            "Create admin monitoring tables",
            migration_060_create_admin_monitoring_tables,
            rollback_060_drop_admin_monitoring_tables,
        ),
        Migration(
            61,
            "Create backup schedule tables",
            migration_061_create_backup_schedule_tables,
            rollback_061_drop_backup_schedule_tables,
        ),
        Migration(
            62,
            "Harden MCP external binding schema",
            migration_059_harden_mcp_external_binding_schema,
        ),
        Migration(
            63,
            "Add MCP external credential slots",
            migration_060_add_mcp_external_credential_slots,
        ),
        Migration(
            64,
            "Add MCP path scope objects and assignment workspaces",
            migration_061_add_mcp_path_scope_objects_and_assignment_workspaces,
        ),
        Migration(
            65,
            "Add MCP workspace set objects",
            migration_062_add_mcp_workspace_set_objects,
        ),
        Migration(
            66,
            "Add MCP shared workspace registry",
            migration_063_add_mcp_shared_workspaces,
        ),
        Migration(
            67,
            "Create maintenance rotation runs table",
            migration_067_create_maintenance_rotation_runs_table,
            rollback_067_drop_maintenance_rotation_runs_table,
        ),
        Migration(
            68,
            "Create BYOK validation runs table",
            migration_068_create_byok_validation_runs_table,
            rollback_068_drop_byok_validation_runs_table,
        ),
        Migration(
            69,
            "Add MCP governance pack schema",
            migration_069_add_mcp_governance_pack_schema,
        ),
        Migration(
            70,
            "Add MCP capability adapter mapping schema",
            migration_070_add_mcp_capability_adapter_mappings,
        ),
        Migration(
            71,
            "Add governance pack upgrade lineage schema",
            migration_071_add_governance_pack_upgrade_lineage,
        ),
        Migration(
            72,
            "Create identity_providers table",
            migration_072_create_identity_providers_table,
            rollback_072_drop_identity_providers_table,
        ),
        Migration(
            73,
            "Create federated_identities table",
            migration_073_create_federated_identities_table,
            rollback_073_drop_federated_identities_table,
        ),
        Migration(
            74,
            "Create federated_managed_grants table",
            migration_074_create_federated_managed_grants_table,
            rollback_074_drop_federated_managed_grants_table,
        ),
        Migration(
            75,
            "Create secret_backends table",
            migration_075_create_secret_backends_table,
            rollback_075_drop_secret_backends_table,
        ),
        Migration(
            76,
            "Create managed_secret_refs table",
            migration_076_create_managed_secret_refs_table,
            rollback_076_drop_managed_secret_refs_table,
        ),
        Migration(
            77,
            "Create sharing tables",
            migration_077_create_sharing_tables,
            rollback_077_drop_sharing_tables,
        ),
        Migration(
            78,
            "Add governance-pack source provenance",
            migration_078_add_governance_pack_source_provenance,
        ),
        Migration(
            79,
            "Add governance-pack trust policy",
            migration_079_add_governance_pack_trust_policy,
        ),
        Migration(
            80,
            "Create admin webhooks tables",
            migration_080_create_admin_webhooks_tables,
            rollback_080_drop_admin_webhooks_tables,
        ),
        Migration(
            81,
            "Create dependency health history table",
            migration_081_create_admin_dependency_health_history,
            rollback_081_drop_admin_dependency_health_history,
        ),
        Migration(
            82,
            "Harden admin webhooks schema and create admin settings table",
            migration_082_harden_admin_webhooks_and_create_admin_settings,
        ),
        Migration(
            83,
            "Create org STT settings table",
            migration_083_create_org_stt_settings,
            rollback_083_drop_org_stt_settings,
        ),
        Migration(
            84,
            "Scope account lockouts by attempt type",
            migration_084_scope_account_lockouts_by_attempt_type,
        ),
        Migration(
            85,
            "Remove api_keys.scope default",
            migration_085_remove_api_keys_scope_default,
        ),
        Migration(
            86,
            "Create prototype workspace metadata tables",
            migration_086_create_prototype_workspace_tables,
            rollback_086_drop_prototype_workspace_tables,
        ),
        Migration(
            87,
            "Expand share_tokens resource_type for prototype workspace links",
            migration_087_expand_share_tokens_resource_type_for_prototypes,
        ),
    ]


def apply_authnz_migrations(db_path: Path, target_version: int = None) -> None:
    """
    Apply AuthNZ migrations to a database

    Args:
        db_path: Path to the database file
        target_version: Target migration version (None = latest)
    """
    import os as _os

    redis_url = _os.getenv("REDIS_URL")
    lock_dir = str(Path(db_path).parent) if db_path else None

    with acquire_migration_lock(
        lock_dir=lock_dir,
        lock_name="authnz_migration",
        redis_url=redis_url,
        timeout=60,
    ):
        _apply_authnz_migrations_locked(db_path, target_version)


def _apply_authnz_migrations_locked(db_path: Path, target_version: int = None) -> None:
    """Inner migration logic, called while holding the distributed lock."""
    manager = MigrationManager(db_path)
    try:
        from loguru import logger as _logger
        _latest = len(get_authnz_migrations())
        _logger.info(
            f"AuthNZ.apply_migrations: db={db_path} target={'latest' if target_version is None else target_version} latest={_latest}"
        )
    except _AUTHNZ_MIGRATIONS_NONCRITICAL_EXCEPTIONS:
        pass

    # Add all migrations to the manager
    for migration in get_authnz_migrations():
        manager.add_migration(migration)

    # Apply migrations
    manager.migrate(target_version)

    logger.info(f"Applied AuthNZ migrations to {db_path}")


def rollback_authnz_migrations(db_path: Path, target_version: int = 0) -> None:
    """
    Rollback AuthNZ migrations

    Args:
        db_path: Path to the database file
        target_version: Target migration version to rollback to
    """
    manager = MigrationManager(db_path)

    # Add all migrations to the manager
    for migration in get_authnz_migrations():
        manager.add_migration(migration)

    # Rollback migrations
    manager.rollback(target_version)

    logger.info(f"Rolled back AuthNZ migrations to version {target_version}")


#######################################################################################################################
#
# Utility Functions
#

def check_migration_status(db_path: Path) -> dict:
    """
    Check the migration status of a database

    Args:
        db_path: Path to the database file

    Returns:
        Dictionary with migration status information
    """
    manager = MigrationManager(db_path)

    # Add all migrations
    for migration in get_authnz_migrations():
        manager.add_migration(migration)

    current_version = manager.get_current_version()
    pending = manager.get_pending_migrations()

    return {
        "current_version": current_version,
        "latest_version": len(get_authnz_migrations()),
        "pending_migrations": [{"version": m.version, "name": m.name} for m in pending],
        "is_up_to_date": len(pending) == 0
    }


def ensure_authnz_tables(db_path: Path) -> None:
    """
    Ensure all AuthNZ tables exist in the database

    Args:
        db_path: Path to the database file
    """
    # Check current status
    status = check_migration_status(db_path)

    if not status["is_up_to_date"]:
        logger.info(
            f"Database needs migrations for {db_path}. Current: {status['current_version']}, Latest: {status['latest_version']}"
        )
        apply_authnz_migrations(db_path)
    else:
        logger.debug("AuthNZ tables are up to date")


#
# End of migrations.py
#######################################################################################################################
