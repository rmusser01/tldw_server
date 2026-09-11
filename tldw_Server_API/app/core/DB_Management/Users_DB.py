# Users_DB.py
# Description: Database operations for user management in multi-user mode
#
# Imports
import asyncio
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

# Guarded optional imports for async drivers. Users_DB relies on the
# unified DatabasePool abstraction and should not hard-depend on these
# modules at import time to support SQLite-only deployments.
try:  # pragma: no cover - presence depends on environment
    import asyncpg  # type: ignore
    _ASYNC_PG_AVAILABLE = True
    try:
        _PG_UniqueViolationError = asyncpg.exceptions.UniqueViolationError  # type: ignore[attr-defined]
    except AttributeError:  # pragma: no cover
        class _PG_UniqueViolationError(Exception):  # type: ignore  # noqa: N801
            pass
except ImportError:  # pragma: no cover
    _ASYNC_PG_AVAILABLE = False
    class _PG_UniqueViolationError(Exception):  # type: ignore  # noqa: N801
        pass

try:  # pragma: no cover - optional in SQLite-only deployments
    import aiosqlite  # type: ignore
    _AIOSQLITE_AVAILABLE = True
    # Provide a safe alias for IntegrityError so except clauses don't NameError
    _AIOSQLITE_IntegrityError = aiosqlite.IntegrityError  # type: ignore[attr-defined]  # noqa: N801
except ImportError:  # pragma: no cover
    _AIOSQLITE_AVAILABLE = False
    # Fallback placeholder so tuple excepts remain valid even when aiosqlite is absent
    class _AIOSQLITE_IntegrityError(Exception):  # type: ignore  # noqa: N801
        pass
#
# 3rd-party imports
from loguru import logger

#
# Local imports
from tldw_Server_API.app.core.AuthNZ.database import DatabasePool, get_db_pool
from tldw_Server_API.app.core.AuthNZ.exceptions import DatabaseError, TransactionError
from tldw_Server_API.app.core.AuthNZ.postgres_profile_version_schema import (
    ensure_postgres_profile_version_on_connection,
)
from tldw_Server_API.app.core.AuthNZ.profile_candidate_schema import (
    PROFILE_CANDIDATE_TABLES,
    profile_candidate_schema_is_valid,
)
from tldw_Server_API.app.core.AuthNZ.profile_user_write_guard import (
    _execute_profile_users_bootstrap,
)
from tldw_Server_API.app.core.AuthNZ.profile_version import (
    PROFILE_VISIBLE_USER_FIELDS,
    VersionedUserWriteGateway,
)
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths

_USERS_DB_NONCRITICAL_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    ConnectionError,
    DatabaseError,
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
    sqlite3.Error,
)

#######################################################################################################################
#
# Exceptions
#

class UserNotFoundError(Exception):
    """Raised when a user is not found in the database"""
    pass

class DuplicateUserError(Exception):
    """Raised when attempting to create a user that already exists"""
    pass

#######################################################################################################################
#
# Users Database Class
#

class UsersDB:
    """Handles all database operations for user management"""

    def __init__(self, db_pool: Optional[DatabasePool] = None):
        """Initialize Users database handler"""
        self.db_pool = db_pool
        self._initialized = False
        self._schema_ensured = False
        self.settings = get_settings()

    def _using_postgres_backend(self) -> bool:
        """Return True when the underlying DatabasePool is backed by PostgreSQL."""
        if self.db_pool is None:
            return False
        return getattr(self.db_pool, "pool", None) is not None

    @staticmethod
    def _normalize_user_row(row: Any, *, is_postgres: bool) -> dict[str, Any]:
        if row is None:
            raise UserNotFoundError("User not found")
        user = dict(row)
        if not is_postgres:
            for field, default in (
                ("is_active", 1),
                ("is_superuser", 0),
                ("email_verified", 0),
                ("is_verified", 0),
            ):
                user[field] = bool(user.get(field, default))
        return user

    async def _get_user_by_id_on_connection(
        self,
        conn: Any,
        user_id: int,
        *,
        is_postgres: bool,
    ) -> dict[str, Any]:
        if is_postgres:
            row = await conn.fetchrow(
                "SELECT * FROM public.users WHERE id = $1",
                user_id,
            )
        else:
            cursor = await conn.execute(
                "SELECT * FROM main.users WHERE id = ?",
                (user_id,),
            )
            row = await cursor.fetchone()
        return self._normalize_user_row(row, is_postgres=is_postgres)

    def _log_storage_failure(self, operation: str, error: BaseException) -> None:
        logger.bind(
            operation=operation,
            backend="postgres" if self._using_postgres_backend() else "sqlite",
            exception_type=type(error).__name__,
        ).error("UsersDB storage operation failed")

    async def initialize(self, *, ensure_schema: bool = True) -> None:
        """Initialize database access and optionally ensure users tables exist."""
        if self._initialized and (not ensure_schema or self._schema_ensured):
            return

        # Get database pool
        if not self.db_pool:
            self.db_pool = await get_db_pool()

        if ensure_schema and not self._schema_ensured:
            await self._create_tables()
            self._schema_ensured = True

        self._initialized = True
        logger.info("UsersDB initialized")

    async def _create_tables(self):
        """Create users and related tables if they don't exist"""
        attempts = 3
        delay = 0.1
        for attempt in range(attempts):
            try:
                async with self.db_pool.transaction() as conn:
                    is_postgres = getattr(self.db_pool, "pool", None) is not None
                    if is_postgres:
                        uuid_default = self._postgres_uuid_default()
                        users_ddl = """
                            CREATE TABLE IF NOT EXISTS public.users (
                                id SERIAL PRIMARY KEY,
                                uuid UUID UNIQUE NOT NULL DEFAULT __UUID_DEFAULT__,
                                username VARCHAR(50) UNIQUE NOT NULL,
                                email VARCHAR(255) UNIQUE NOT NULL,
                                password_hash TEXT NOT NULL,
                                metadata JSONB,
                                is_active BOOLEAN NOT NULL DEFAULT TRUE,
                                is_superuser BOOLEAN NOT NULL DEFAULT FALSE,
                                role VARCHAR(50) NOT NULL DEFAULT 'user',
                                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                                updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                                last_login TIMESTAMPTZ,
                                email_verified BOOLEAN NOT NULL DEFAULT FALSE,
                                is_verified BOOLEAN NOT NULL DEFAULT FALSE,
                                two_factor_enabled BOOLEAN NOT NULL DEFAULT FALSE,
                                failed_login_attempts INTEGER NOT NULL DEFAULT 0,
                                locked_until TIMESTAMPTZ,
                                storage_quota_mb INTEGER NOT NULL DEFAULT 5120,
                                storage_used_mb INTEGER NOT NULL DEFAULT 0,
                                email_verified_at TIMESTAMPTZ,
                                two_factor_secret TEXT,
                                totp_secret TEXT,
                                backup_codes TEXT,
                                created_by INTEGER REFERENCES public.users(id) ON DELETE SET NULL,
                                password_changed_at TIMESTAMPTZ,
                                profile_version TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                            )
                        """.replace("__UUID_DEFAULT__", uuid_default)
                        await _execute_profile_users_bootstrap(
                            conn,
                            users_ddl,
                            backend="postgres",
                        )
                        await ensure_postgres_profile_version_on_connection(conn)

                        # Create indexes
                        await conn.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON public.users(username)")
                        await conn.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON public.users(email)")
                        await conn.execute("CREATE INDEX IF NOT EXISTS idx_users_role ON public.users(role)")
                        await conn.execute("ALTER TABLE public.users ADD COLUMN IF NOT EXISTS metadata JSONB")
                        await conn.execute("ALTER TABLE public.users ADD COLUMN IF NOT EXISTS uuid UUID")
                        await conn.execute(
                            "ALTER TABLE public.users ADD COLUMN IF NOT EXISTS storage_quota_mb INTEGER DEFAULT 5120"
                        )
                        await conn.execute(
                            "ALTER TABLE public.users ADD COLUMN IF NOT EXISTS storage_used_mb INTEGER DEFAULT 0"
                        )
                        await self._ensure_profile_candidate_tables(conn, is_postgres=True)
                        await self._validate_profile_candidate_tables(
                            conn,
                            is_postgres=True,
                        )
                        rows = await conn.fetch("SELECT id FROM public.users WHERE uuid IS NULL")
                        gateway = VersionedUserWriteGateway("postgres")
                        for row in rows:
                            user_id = int(row["id"])
                            await gateway.execute_update(
                                conn,
                                user_id=user_id,
                                profile_visible_fields=("uuid",),
                                statement="UPDATE public.users SET uuid = $1 WHERE id = $2",
                                parameters=(str(uuid.uuid4()), user_id),
                            )
                        await conn.execute("ALTER TABLE public.users ALTER COLUMN uuid SET NOT NULL")
                        try:
                            await conn.execute("ALTER TABLE public.users ALTER COLUMN uuid SET DEFAULT gen_random_uuid()")
                        except _USERS_DB_NONCRITICAL_EXCEPTIONS:
                            try:
                                await conn.execute(
                                    "ALTER TABLE public.users ALTER COLUMN uuid SET DEFAULT uuid_generate_v4()"
                                )
                            except _USERS_DB_NONCRITICAL_EXCEPTIONS as def_err:
                                logger.bind(
                                    operation="users_uuid_default",
                                    exception_type=type(def_err).__name__,
                                ).warning("Could not set PostgreSQL users UUID default")

                    else:
                        # SQLite
                        await _execute_profile_users_bootstrap(conn, """
                            CREATE TABLE IF NOT EXISTS main.users (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                uuid TEXT UNIQUE NOT NULL DEFAULT (lower(hex(randomblob(16)))),
                                username TEXT UNIQUE NOT NULL,
                                email TEXT UNIQUE NOT NULL,
                                password_hash TEXT NOT NULL,
                                metadata TEXT,
                                is_active INTEGER NOT NULL DEFAULT 1,
                                is_superuser INTEGER NOT NULL DEFAULT 0,
                                role TEXT NOT NULL DEFAULT 'user',
                                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                                last_login TIMESTAMP,
                                email_verified INTEGER NOT NULL DEFAULT 0,
                                is_verified INTEGER NOT NULL DEFAULT 0,
                                two_factor_enabled INTEGER NOT NULL DEFAULT 0,
                                failed_login_attempts INTEGER NOT NULL DEFAULT 0,
                                locked_until TIMESTAMP,
                                storage_quota_mb INTEGER NOT NULL DEFAULT 5120,
                                storage_used_mb INTEGER NOT NULL DEFAULT 0,
                                email_verified_at TIMESTAMP,
                                two_factor_secret TEXT,
                                totp_secret TEXT,
                                backup_codes TEXT,
                                created_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
                                password_changed_at TIMESTAMP,
                                profile_version TEXT NOT NULL DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%f000Z', 'now'))
                            )
                        """, backend="sqlite")

                        # Create indexes
                        await conn.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")
                        await conn.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
                        await conn.execute("CREATE INDEX IF NOT EXISTS idx_users_role ON users(role)")
                        cursor = await conn.execute("PRAGMA table_info(users)")
                        columns_info = await cursor.fetchall()
                        columns = {row[1] for row in columns_info}
                        if "metadata" not in columns:
                            await conn.execute("ALTER TABLE users ADD COLUMN metadata TEXT")
                        if "uuid" not in columns:
                            await conn.execute("ALTER TABLE users ADD COLUMN uuid TEXT")
                        if "is_active" not in columns:
                            await conn.execute(
                                "ALTER TABLE users ADD COLUMN is_active INTEGER DEFAULT 1"
                            )
                        if "is_superuser" not in columns:
                            await conn.execute(
                                "ALTER TABLE users ADD COLUMN is_superuser INTEGER DEFAULT 0"
                            )
                        if "email_verified" not in columns:
                            await conn.execute(
                                "ALTER TABLE users ADD COLUMN email_verified INTEGER DEFAULT 0"
                            )
                        if "is_verified" not in columns:
                            await conn.execute(
                                "ALTER TABLE users ADD COLUMN is_verified INTEGER DEFAULT 0"
                            )
                        if "storage_quota_mb" not in columns:
                            await conn.execute("ALTER TABLE users ADD COLUMN storage_quota_mb INTEGER DEFAULT 5120")
                        if "storage_used_mb" not in columns:
                            await conn.execute("ALTER TABLE users ADD COLUMN storage_used_mb INTEGER DEFAULT 0")
                        if "profile_version" not in columns:
                            raise RuntimeError(
                                "AuthNZ users table is missing required columns: "
                                "profile_version"
                            )
                        await self._ensure_profile_candidate_tables(
                            conn,
                            is_postgres=False,
                        )
                        await self._validate_profile_candidate_tables(
                            conn,
                            is_postgres=False,
                        )
                        cursor = await conn.execute(
                            "SELECT id FROM users WHERE uuid IS NULL OR uuid = ''"
                        )
                        rows = await cursor.fetchall()
                        gateway = VersionedUserWriteGateway("sqlite")
                        for row in rows:
                            user_id = int(row[0])
                            await gateway.execute_update(
                                conn,
                                user_id=user_id,
                                profile_visible_fields=("uuid",),
                                statement="UPDATE users SET uuid = ? WHERE id = ?",
                                parameters=(str(uuid.uuid4()), user_id),
                            )
                        try:
                            await conn.execute(
                                "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_uuid ON users(uuid)"
                            )
                        except _USERS_DB_NONCRITICAL_EXCEPTIONS as idx_err:
                            logger.bind(
                                operation="users_uuid_index",
                                exception_type=type(idx_err).__name__,
                            ).warning("Could not create users UUID index")

                    logger.debug("Users table and indexes created/verified")
                return
            except _USERS_DB_NONCRITICAL_EXCEPTIONS as e:
                if attempt < attempts - 1 and "locked" in str(e).lower():
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 1.0)
                    continue
                self._log_storage_failure("create_tables", e)
                if "profile_version" in str(e).lower():
                    raise DatabaseError(
                        "AuthNZ users.profile_version readiness validation failed"
                    ) from None
                raise DatabaseError("Failed to create users table") from None

    @staticmethod
    def _postgres_uuid_default() -> str:
        return "gen_random_uuid()"

    @staticmethod
    async def _ensure_profile_candidate_tables(conn: Any, *, is_postgres: bool) -> None:
        if is_postgres:
            statements = (
                """CREATE TABLE IF NOT EXISTS public.organizations (
                    id SERIAL PRIMARY KEY,
                    uuid VARCHAR(64) UNIQUE,
                    name VARCHAR(255) UNIQUE NOT NULL,
                    slug VARCHAR(255) UNIQUE,
                    owner_user_id INTEGER REFERENCES public.users(id) ON DELETE SET NULL,
                    is_active BOOLEAN DEFAULT TRUE,
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                )""",
                """CREATE TABLE IF NOT EXISTS public.teams (
                    id SERIAL PRIMARY KEY,
                    org_id INTEGER NOT NULL REFERENCES public.organizations(id) ON DELETE CASCADE,
                    name VARCHAR(255) NOT NULL,
                    slug VARCHAR(255),
                    description TEXT,
                    is_active BOOLEAN DEFAULT TRUE,
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (org_id, name)
                )""",
                """CREATE TABLE IF NOT EXISTS public.org_members (
                    org_id INTEGER NOT NULL REFERENCES public.organizations(id) ON DELETE CASCADE,
                    user_id INTEGER NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
                    role VARCHAR(32) DEFAULT 'member',
                    status VARCHAR(32) DEFAULT 'active',
                    added_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (org_id, user_id)
                )""",
                """CREATE TABLE IF NOT EXISTS public.team_members (
                    team_id INTEGER NOT NULL REFERENCES public.teams(id) ON DELETE CASCADE,
                    user_id INTEGER NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
                    role VARCHAR(32) DEFAULT 'member',
                    status VARCHAR(32) DEFAULT 'active',
                    added_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (team_id, user_id)
                )""",
                """CREATE TABLE IF NOT EXISTS public.user_config_overrides (
                    user_id INTEGER NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
                    key TEXT NOT NULL,
                    value_json TEXT,
                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    created_by INTEGER,
                    updated_by INTEGER,
                    PRIMARY KEY (user_id, key)
                )""",
                """CREATE TABLE IF NOT EXISTS public.org_config_overrides (
                    org_id INTEGER NOT NULL REFERENCES public.organizations(id) ON DELETE CASCADE,
                    key TEXT NOT NULL,
                    value_json TEXT,
                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    created_by INTEGER,
                    updated_by INTEGER,
                    PRIMARY KEY (org_id, key)
                )""",
                """CREATE TABLE IF NOT EXISTS public.team_config_overrides (
                    team_id INTEGER NOT NULL REFERENCES public.teams(id) ON DELETE CASCADE,
                    key TEXT NOT NULL,
                    value_json TEXT,
                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    created_by INTEGER,
                    updated_by INTEGER,
                    PRIMARY KEY (team_id, key)
                )""",
            )
        else:
            statements = (
                """CREATE TABLE IF NOT EXISTS main.organizations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    uuid TEXT UNIQUE,
                    name TEXT UNIQUE NOT NULL,
                    slug TEXT UNIQUE,
                    owner_user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
                    is_active INTEGER DEFAULT 1,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )""",
                """CREATE TABLE IF NOT EXISTS main.teams (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    org_id INTEGER NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
                    name TEXT NOT NULL,
                    slug TEXT,
                    description TEXT,
                    is_active INTEGER DEFAULT 1,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (org_id, name)
                )""",
                """CREATE TABLE IF NOT EXISTS main.org_members (
                    org_id INTEGER NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    role TEXT DEFAULT 'member',
                    status TEXT DEFAULT 'active',
                    added_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (org_id, user_id)
                )""",
                """CREATE TABLE IF NOT EXISTS main.team_members (
                    team_id INTEGER NOT NULL REFERENCES teams(id) ON DELETE CASCADE,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    role TEXT DEFAULT 'member',
                    status TEXT DEFAULT 'active',
                    added_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (team_id, user_id)
                )""",
                """CREATE TABLE IF NOT EXISTS main.user_config_overrides (
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    key TEXT NOT NULL,
                    value_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    created_by INTEGER,
                    updated_by INTEGER,
                    PRIMARY KEY (user_id, key)
                )""",
                """CREATE TABLE IF NOT EXISTS main.org_config_overrides (
                    org_id INTEGER NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
                    key TEXT NOT NULL,
                    value_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    created_by INTEGER,
                    updated_by INTEGER,
                    PRIMARY KEY (org_id, key)
                )""",
                """CREATE TABLE IF NOT EXISTS main.team_config_overrides (
                    team_id INTEGER NOT NULL REFERENCES teams(id) ON DELETE CASCADE,
                    key TEXT NOT NULL,
                    value_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    created_by INTEGER,
                    updated_by INTEGER,
                    PRIMARY KEY (team_id, key)
                )""",
            )
        for statement in statements:
            await conn.execute(statement)

    @staticmethod
    async def _validate_profile_candidate_tables(
        conn: Any,
        *,
        is_postgres: bool,
    ) -> None:
        columns_by_table: dict[str, dict[str, dict[str, Any]]] = {
            table_name: {} for table_name in PROFILE_CANDIDATE_TABLES
        }
        primary_key_rows: dict[str, list[tuple[int, str]]] = {
            table_name: [] for table_name in PROFILE_CANDIDATE_TABLES
        }
        unique_key_rows: dict[tuple[str, str], list[tuple[int, str]]] = {}
        foreign_keys_by_table: dict[
            str,
            set[tuple[str, str, str, str, str]],
        ] = {table_name: set() for table_name in PROFILE_CANDIDATE_TABLES}

        if is_postgres:
            placeholders = ", ".join(
                f"${position}"
                for position in range(1, len(PROFILE_CANDIDATE_TABLES) + 1)
            )
            table_filter = f"({placeholders})"
            column_rows = await conn.fetch(
                "SELECT table_name, column_name, data_type, is_nullable, "
                "column_default, is_identity, identity_generation "
                "FROM information_schema.columns "
                "WHERE table_schema = 'public' AND table_name IN "
                + table_filter,  # nosec B608 -- fixed-count placeholders only.
                *PROFILE_CANDIDATE_TABLES,
            )
            for row in column_rows:
                columns_by_table[str(row["table_name"])][str(row["column_name"])] = {
                    "data_type": row["data_type"],
                    "not_null": str(row["is_nullable"]).upper() == "NO",
                    "default": row["column_default"],
                    "is_identity": row["is_identity"],
                    "identity_generation": row["identity_generation"],
                }

            primary_rows = await conn.fetch(
                "SELECT tc.table_name, kcu.column_name, kcu.ordinal_position "
                "FROM information_schema.table_constraints AS tc "
                "JOIN information_schema.key_column_usage AS kcu "
                "ON tc.constraint_name = kcu.constraint_name "
                "AND tc.constraint_schema = kcu.constraint_schema "
                "WHERE tc.table_schema = 'public' "
                "AND tc.constraint_type = 'PRIMARY KEY' "
                "AND tc.table_name IN "
                + table_filter,  # nosec B608 -- fixed-count placeholders only.
                *PROFILE_CANDIDATE_TABLES,
            )
            for row in primary_rows:
                primary_key_rows[str(row["table_name"])].append(
                    (int(row["ordinal_position"]), str(row["column_name"]))
                )

            unique_rows = await conn.fetch(
                "SELECT tc.table_name, tc.constraint_name, kcu.column_name, "
                "kcu.ordinal_position FROM information_schema.table_constraints AS tc "
                "JOIN information_schema.key_column_usage AS kcu "
                "ON tc.constraint_name = kcu.constraint_name "
                "AND tc.constraint_schema = kcu.constraint_schema "
                "WHERE tc.table_schema = 'public' "
                "AND tc.constraint_type = 'UNIQUE' AND tc.table_name IN "
                + table_filter,  # nosec B608 -- fixed-count placeholders only.
                *PROFILE_CANDIDATE_TABLES,
            )
            for row in unique_rows:
                key = (str(row["table_name"]), str(row["constraint_name"]))
                unique_key_rows.setdefault(key, []).append(
                    (int(row["ordinal_position"]), str(row["column_name"]))
                )

            foreign_rows = await conn.fetch(
                "SELECT tc.table_name, kcu.column_name, "
                "ccu.table_schema AS foreign_table_schema, "
                "ccu.table_name AS foreign_table_name, "
                "ccu.column_name AS foreign_column_name, rc.delete_rule "
                "FROM information_schema.table_constraints AS tc "
                "JOIN information_schema.key_column_usage AS kcu "
                "ON tc.constraint_name = kcu.constraint_name "
                "AND tc.constraint_schema = kcu.constraint_schema "
                "JOIN information_schema.referential_constraints AS rc "
                "ON tc.constraint_name = rc.constraint_name "
                "AND tc.constraint_schema = rc.constraint_schema "
                "JOIN information_schema.constraint_column_usage AS ccu "
                "ON rc.unique_constraint_name = ccu.constraint_name "
                "AND rc.unique_constraint_schema = ccu.constraint_schema "
                "WHERE tc.table_schema = 'public' "
                "AND tc.constraint_type = 'FOREIGN KEY' "
                "AND tc.table_name IN "
                + table_filter,  # nosec B608 -- fixed-count placeholders only.
                *PROFILE_CANDIDATE_TABLES,
            )
            for row in foreign_rows:
                foreign_keys_by_table[str(row["table_name"])].add(
                    (
                        str(row["column_name"]),
                        str(row["foreign_table_schema"]),
                        str(row["foreign_table_name"]),
                        str(row["foreign_column_name"]),
                        str(row["delete_rule"]),
                    )
                )
            backend = "postgres"
        else:
            for table_name in PROFILE_CANDIDATE_TABLES:
                cursor = await conn.execute(f'PRAGMA table_info("{table_name}")')  # nosec B608
                table_info = await cursor.fetchall()
                columns_by_table[table_name] = {
                    str(row[1]): {
                        "data_type": row[2],
                        "not_null": bool(row[3]) or int(row[5]) > 0,
                        "default": row[4],
                    }
                    for row in table_info
                }
                primary_key_rows[table_name] = [
                    (int(row[5]), str(row[1]))
                    for row in table_info
                    if int(row[5]) > 0
                ]
                cursor = await conn.execute(
                    "SELECT index_list.name AS index_name, "
                    "index_info.name AS column_name, index_info.seqno "
                    "FROM pragma_index_list(?) AS index_list "
                    "JOIN pragma_index_info(index_list.name) AS index_info "
                    "WHERE index_list.[unique] = 1 "
                    "AND index_list.origin <> 'pk' "
                    "ORDER BY index_list.name, index_info.seqno",
                    (table_name,),
                )
                for row in await cursor.fetchall():
                    unique_key_rows.setdefault(
                        (table_name, str(row[0])),
                        [],
                    ).append((int(row[2]), str(row[1])))
                cursor = await conn.execute(
                    f'PRAGMA foreign_key_list("{table_name}")'  # nosec B608
                )
                foreign_keys_by_table[table_name] = {
                    (
                        str(row[3]),
                        "main",
                        str(row[2]),
                        str(row[4]),
                        str(row[6]),
                    )
                    for row in await cursor.fetchall()
                }
            backend = "sqlite"

        primary_keys = {
            table_name: tuple(
                column for _position, column in sorted(primary_key_rows[table_name])
            )
            for table_name in PROFILE_CANDIDATE_TABLES
        }
        unique_keys: dict[str, set[tuple[str, ...]]] = {
            table_name: set() for table_name in PROFILE_CANDIDATE_TABLES
        }
        for (table_name, _constraint_name), rows in unique_key_rows.items():
            unique_keys[table_name].add(
                tuple(column for _position, column in sorted(rows))
            )
        if not profile_candidate_schema_is_valid(
            backend=backend,
            columns=columns_by_table,
            primary_keys=primary_keys,
            unique_keys=unique_keys,
            foreign_keys=foreign_keys_by_table,
        ):
            raise DatabaseError(
                "Required profile candidate schema validation failed"
            )

    async def get_user_by_id(self, user_id: int) -> Optional[dict[str, Any]]:
        """
        Get user by ID

        Args:
            user_id: User's database ID

        Returns:
            User data dictionary or None if not found

        Raises:
            UserNotFoundError: If user doesn't exist
        """
        if not self._initialized:
            await self.initialize()

        try:
            result = await self.db_pool.fetchone(
                "SELECT * FROM users WHERE id = ?",
                user_id
            )

            if not result:
                raise UserNotFoundError(f"User with ID {user_id} not found")

            # Convert to dictionary
            user_dict = dict(result)

            # Convert boolean fields for SQLite
            if not self._using_postgres_backend():  # SQLite
                user_dict['is_active'] = bool(user_dict.get('is_active', 1))
                user_dict['is_superuser'] = bool(user_dict.get('is_superuser', 0))
                user_dict['email_verified'] = bool(user_dict.get('email_verified', 0))
                user_dict['is_verified'] = bool(user_dict.get('is_verified', 0))

            return user_dict

        except UserNotFoundError:
            raise
        except _USERS_DB_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to get user by ID {user_id}: {e}")
            raise DatabaseError(f"Failed to get user: {e}") from e

    async def get_user_by_username(self, username: str) -> Optional[dict[str, Any]]:
        """
        Get user by username

        Args:
            username: Username to search for

        Returns:
            User data dictionary or None if not found
        """
        if not self._initialized:
            await self.initialize()

        try:
            result = await self.db_pool.fetchone(
                "SELECT * FROM users WHERE username = ?",
                username
            )

            if not result:
                return None

            # Convert to dictionary
            user_dict = dict(result)

            # Convert boolean fields for SQLite
            if not self._using_postgres_backend():  # SQLite
                user_dict['is_active'] = bool(user_dict.get('is_active', 1))
                user_dict['is_superuser'] = bool(user_dict.get('is_superuser', 0))
                user_dict['email_verified'] = bool(user_dict.get('email_verified', 0))
                user_dict['is_verified'] = bool(user_dict.get('is_verified', 0))

            return user_dict

        except _USERS_DB_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to get user by username {username}: {e}")
            raise DatabaseError(f"Failed to get user: {e}") from e

    async def get_user_by_uuid(self, user_uuid: str) -> Optional[dict[str, Any]]:
        """
        Get user by UUID (textual identifier) when available.

        Args:
            user_uuid: UUID string stored with the user.

        Returns:
            User data dictionary or None if not found.
        """
        if not self._initialized:
            await self.initialize()

        if not user_uuid:
            return None

        try:
            result = await self.db_pool.fetchone(
                "SELECT * FROM users WHERE uuid = ?",
                user_uuid
            )

            if not result:
                return None

            user_dict = dict(result)

            if not self._using_postgres_backend():  # SQLite conversions
                user_dict['is_active'] = bool(user_dict.get('is_active', 1))
                user_dict['is_superuser'] = bool(user_dict.get('is_superuser', 0))
                user_dict['email_verified'] = bool(user_dict.get('email_verified', 0))
                user_dict['is_verified'] = bool(user_dict.get('is_verified', 0))

            return user_dict

        except _USERS_DB_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to get user by uuid {user_uuid}: {e}")
            raise DatabaseError(f"Failed to get user: {e}") from e

    async def get_user_by_email(self, email: str) -> Optional[dict[str, Any]]:
        """
        Get user by email

        Args:
            email: Email to search for

        Returns:
            User data dictionary or None if not found
        """
        if not self._initialized:
            await self.initialize()

        try:
            result = await self.db_pool.fetchone(
                "SELECT * FROM users WHERE email = ?",
                email.lower()
            )

            if not result:
                return None

            # Convert to dictionary
            user_dict = dict(result)

            # Convert boolean fields for SQLite
            if not self._using_postgres_backend():  # SQLite
                user_dict['is_active'] = bool(user_dict.get('is_active', 1))
                user_dict['is_superuser'] = bool(user_dict.get('is_superuser', 0))
                user_dict['email_verified'] = bool(user_dict.get('email_verified', 0))
                user_dict['is_verified'] = bool(user_dict.get('is_verified', 0))

            return user_dict

        except _USERS_DB_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to get user by email {email}: {e}")
            raise DatabaseError(f"Failed to get user: {e}") from e

    async def create_user(
        self,
        username: str,
        email: str,
        password_hash: str,
        role: str = "user",
        is_active: bool = True,
        is_verified: bool = False,
        is_superuser: bool = False,
        storage_quota_mb: int = 5120,
        uuid_value: Optional[uuid.UUID | str] = None,
    ) -> dict[str, Any]:
        """
        Create a new user

        Args:
            username: Unique username
            email: User's email address
            password_hash: Hashed password
            role: User role (default: "user")
            is_active: Whether user is active
            is_verified: Whether user is verified
            is_superuser: Whether user is a superuser
            storage_quota_mb: Storage quota in MB
            uuid_value: Optional pre-assigned UUID for the user. When omitted,
                a new UUID4 is generated.

        Returns:
            Created user data

        Raises:
            DuplicateUserError: If username or email already exists
        """
        if not self._initialized:
            await self.initialize()

        # Check for existing user
        existing = await self.get_user_by_username(username)
        if existing:
            raise DuplicateUserError("Username already exists")

        existing = await self.get_user_by_email(email)
        if existing:
            raise DuplicateUserError("Email already exists")

        try:
            generated_uuid = str(uuid_value) if uuid_value is not None else str(uuid.uuid4())
            user_id: Optional[int] = None

            async with self.db_pool.transaction() as conn:
                is_postgres = self._using_postgres_backend()
                gateway = VersionedUserWriteGateway(
                    "postgres" if is_postgres else "sqlite"
                )
                insert_result = await gateway.insert_user(
                    conn,
                    values={
                        "uuid": generated_uuid,
                        "username": username,
                        "email": email.lower(),
                        "password_hash": password_hash,
                        "role": role,
                        "is_active": is_active if is_postgres else int(is_active),
                        "is_verified": is_verified if is_postgres else int(is_verified),
                        "is_superuser": (
                            is_superuser if is_postgres else int(is_superuser)
                        ),
                        "storage_quota_mb": storage_quota_mb,
                    },
                )
                user_id = insert_result.affected_user_ids[0]
            logger.info("Created user")

            if user_id is None:
                raise DatabaseError("Failed to create user: no id returned from insert")

            # Return the created user (outside transaction so row is visible on all connections)
            return await self.get_user_by_id(int(user_id))

        except DuplicateUserError:
            raise
        except _PG_UniqueViolationError as e:
            self._log_storage_failure("create_user_duplicate", e)
            raise DuplicateUserError("Username or email already exists") from None
        except (_AIOSQLITE_IntegrityError, sqlite3.IntegrityError) as e:
            message = str(e).lower()
            if "unique constraint failed" in message or "unique constraint violation" in message:
                self._log_storage_failure("create_user_duplicate", e)
                raise DuplicateUserError("Username or email already exists") from None
            self._log_storage_failure("create_user", e)
            raise DatabaseError("Failed to create user") from None
        except _USERS_DB_NONCRITICAL_EXCEPTIONS as e:
            if isinstance(e, TransactionError):
                try:
                    duplicate = await self.db_pool.fetchone(
                        "SELECT id FROM users WHERE username = ? OR email = ? LIMIT 1",
                        username,
                        email.lower(),
                    )
                except _USERS_DB_NONCRITICAL_EXCEPTIONS:
                    duplicate = None
                if duplicate is not None:
                    self._log_storage_failure("create_user_duplicate", e)
                    raise DuplicateUserError(
                        "Username or email already exists"
                    ) from None
            msg = str(e)
            if "UNIQUE constraint failed" in msg and "users" in msg:
                self._log_storage_failure("create_user_duplicate", e)
                raise DuplicateUserError("Username or email already exists") from None
            self._log_storage_failure("create_user", e)
            raise DatabaseError("Failed to create user") from None

    async def update_user(
        self,
        user_id: int,
        **kwargs
    ) -> dict[str, Any]:
        """
        Update user information

        Args:
            user_id: User ID to update
            **kwargs: Fields to update

        Returns:
            Updated user data
        """
        if not self._initialized:
            await self.initialize()

        # Ensure user exists
        user = await self.get_user_by_id(user_id)
        if not user:
            raise UserNotFoundError(f"User with ID {user_id} not found")

        # Filter allowed fields
        allowed_fields = {
            'email', 'password_hash', 'is_active', 'is_superuser',
            'role', 'last_login', 'email_verified', 'is_verified', 'storage_quota_mb',
            'storage_used_mb'
        }

        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}

        if not updates:
            return user  # Nothing to update

        try:
            # Build update query
            # Note: Build placeholder style per backend; keep deterministic field order
            field_names = list(updates.keys())

            async with self.db_pool.transaction() as conn:
                # Determine backend from DatabasePool state (not conn capability probing).
                is_postgres = self._using_postgres_backend()

                if is_postgres:
                    # PostgreSQL - use $1..$n placeholders
                    set_clause = ", ".join(f"{k} = ${i+1}" for i, k in enumerate(field_names))
                    values = [updates[k] for k in field_names] + [user_id]
                    user_id_param = len(values)
                    update_user_sql_template = (
                        "UPDATE users SET {set_clause}, updated_at = CURRENT_TIMESTAMP "
                        "WHERE id = ${user_id_param}"
                    )
                    query = update_user_sql_template.format_map(locals())  # nosec B608
                    visible_fields = tuple(
                        field
                        for field in field_names
                        if field in PROFILE_VISIBLE_USER_FIELDS
                    )
                    await VersionedUserWriteGateway("postgres").execute_update(
                        conn,
                        user_id=user_id,
                        profile_visible_fields=visible_fields,
                        statement=query,
                        parameters=tuple(values),
                    )
                else:
                    # SQLite - convert bools to ints and use '?' placeholders
                    for key in ['is_active', 'is_superuser', 'email_verified', 'is_verified']:
                        if key in updates:
                            updates[key] = int(bool(updates[key]))
                    set_clause = ", ".join(f"{k} = ?" for k in field_names)
                    values = [updates[k] for k in field_names] + [user_id]
                    update_user_sql_template = "UPDATE users SET {set_clause}, updated_at = CURRENT_TIMESTAMP WHERE id = ?"
                    query = update_user_sql_template.format_map(locals())  # nosec B608
                    visible_fields = tuple(
                        field
                        for field in field_names
                        if field in PROFILE_VISIBLE_USER_FIELDS
                    )
                    await VersionedUserWriteGateway("sqlite").execute_update(
                        conn,
                        user_id=user_id,
                        profile_visible_fields=visible_fields,
                        statement=query,
                        parameters=tuple(values),
                    )

                updated_user = await self._get_user_by_id_on_connection(
                    conn,
                    user_id,
                    is_postgres=is_postgres,
                )
                logger.info("Updated user")
                return updated_user

        except _USERS_DB_NONCRITICAL_EXCEPTIONS as e:
            self._log_storage_failure("update_user", e)
            raise DatabaseError("Failed to update user") from None

    async def delete_user(self, user_id: int) -> bool:
        """
        Delete a user (soft delete by marking inactive)

        Args:
            user_id: User ID to delete

        Returns:
            True if successful
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Soft delete - just mark as inactive
            await self.update_user(user_id, is_active=False)
            logger.info(f"Soft deleted user {user_id}")
            return True

        except _USERS_DB_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to delete user {user_id}: {e}")
            raise DatabaseError(f"Failed to delete user: {e}") from e

    async def update_last_login(self, user_id: int):
        """Update user's last login timestamp"""
        await self.update_user(user_id, last_login=datetime.now(timezone.utc))

    async def list_users(
        self,
        offset: int = 0,
        limit: int = 100,
        role: Optional[str] = None,
        is_active: Optional[bool] = None
    ) -> list[dict[str, Any]]:
        """
        List users with optional filtering

        Args:
            offset: Pagination offset
            limit: Maximum results
            role: Filter by role
            is_active: Filter by active status

        Returns:
            List of user dictionaries
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Build query with filters
            query = "SELECT * FROM users WHERE 1=1"
            params = []

            if role is not None:
                query += " AND role = ?"
                params.append(role)

            if is_active is not None:
                query += " AND is_active = ?"
                params.append(int(is_active) if not self._using_postgres_backend() else is_active)

            query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            results = await self.db_pool.fetchall(query, *params)

            users = []
            for row in results:
                user_dict = dict(row)

                # Convert boolean fields for SQLite
                if not self._using_postgres_backend():  # SQLite
                    user_dict['is_active'] = bool(user_dict.get('is_active', 1))
                    user_dict['is_superuser'] = bool(user_dict.get('is_superuser', 0))
                    user_dict['email_verified'] = bool(user_dict.get('email_verified', 0))
                    user_dict['is_verified'] = bool(user_dict.get('is_verified', 0))

                users.append(user_dict)

            return users

        except _USERS_DB_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to list users: {e}")
            raise DatabaseError(f"Failed to list users: {e}") from e


#######################################################################################################################
#
# Module Functions
#

# Global instance
_users_db: Optional[UsersDB] = None

async def get_users_db() -> UsersDB:
    """Get UsersDB singleton instance"""
    global _users_db
    if not _users_db:
        _users_db = UsersDB()
        await _users_db.initialize()
    return _users_db

async def reset_users_db() -> None:
    """Reset the UsersDB singleton (testing utility)."""
    global _users_db
    _users_db = None

# Convenience functions for backward compatibility
async def get_user_by_id(user_id: int) -> Optional[dict[str, Any]]:
    """Get user by ID (convenience function)"""
    db = await get_users_db()
    return await db.get_user_by_id(user_id)

async def get_user_by_uuid(user_uuid: str) -> Optional[dict[str, Any]]:
    """Get user by UUID (convenience function)"""
    db = await get_users_db()
    return await db.get_user_by_uuid(user_uuid)

async def create_user(username: str, email: str, password_hash: str, **kwargs) -> dict[str, Any]:
    """Create user (convenience function)"""
    db = await get_users_db()
    return await db.create_user(username, email, password_hash, **kwargs)

async def get_user_by_username(username: str) -> Optional[dict[str, Any]]:
    """Get user by username (convenience function)"""
    db = await get_users_db()
    return await db.get_user_by_username(username)


#######################################################################################################################
#
# Per-User Database Path Management
# Each user gets their own SQLite database for their media/content
#

def get_user_db_path(user_id: int, db_name: str = "media") -> str:
    """
    Resolve the canonical path for a user's database file.

    Args:
        user_id: The user's ID
        db_name: Logical database key (media, chacha, prompts, audit, evaluations, personalization, etc.)

    Returns:
        Absolute path to the requested database file as a string.
    """
    db_name_normalized = (db_name or "media").strip().lower()
    path_getters = {
        "media": DatabasePaths.get_media_db_path,
        "chacha": DatabasePaths.get_chacha_db_path,
        "chachanotes": DatabasePaths.get_chacha_db_path,
        "prompts": DatabasePaths.get_prompts_db_path,
        "audit": DatabasePaths.get_audit_db_path,
        "evaluations": DatabasePaths.get_evaluations_db_path,
        "personalization": DatabasePaths.get_personalization_db_path,
        "workflows": DatabasePaths.get_workflows_db_path,
        "workflows_scheduler": DatabasePaths.get_workflows_scheduler_db_path,
    }

    getter = path_getters.get(db_name_normalized)
    if getter:
        return str(getter(user_id))

    # Fallback: place custom databases alongside the canonical user directory
    fallback_path = DatabasePaths.get_user_base_directory(user_id) / f"{db_name_normalized}.db"
    fallback_path.parent.mkdir(parents=True, exist_ok=True)
    return str(fallback_path)


def get_user_chromadb_path(user_id: int) -> str:
    """
    Construct the path for a user's ChromaDB data.

    Args:
        user_id: The user's ID

    Returns:
        Path to the user's ChromaDB directory
    """
    return str(DatabasePaths.get_user_chroma_dir(user_id))


async def get_user_media_db(user_id: int, db_name: str = "media"):
    """
    Get a MediaDatabase instance for a specific user.

    Args:
        user_id: The user's ID
        db_name: Name of the database

    Returns:
        MediaDatabase instance for the user

    Note:
        This creates the user directory structure if it doesn't exist.
    """
    from pathlib import Path

    # Get the database path (ensures directory exists)
    db_path = Path(get_user_db_path(user_id, db_name))

    # Import media DB factory (avoid circular import)
    try:
        from tldw_Server_API.app.core.DB_Management.media_db.api import create_media_database

        # Create and return the database instance via central factory
        db_instance = create_media_database(client_id=str(user_id), db_path=str(db_path))
        return db_instance

    except ImportError as e:
        logger.error(f"Failed to import MediaDatabase: {e}")
        raise ImportError("MediaDatabase class not available") from e


async def ensure_user_directories(user_id: int):
    """
    Ensure all necessary directories exist for a user.

    Args:
        user_id: The user's ID
    """
    from pathlib import Path

    # Ensure database structure via centralized helpers
    DatabasePaths.validate_database_structure(user_id)

    # Ensure Chroma storage directory exists alongside other user assets
    chroma_dir = Path(get_user_chromadb_path(user_id))
    chroma_dir.mkdir(parents=True, exist_ok=True)

    logger.debug(f"Ensured directories exist for user {user_id} -> {DatabasePaths.get_user_base_directory(user_id)}")


async def cleanup_user_data(user_id: int):
    """
    Clean up all data associated with a user (for deletion).

    Args:
        user_id: The user's ID

    Warning:
        This permanently deletes all user data!
    """
    import shutil

    base_dir = DatabasePaths.get_user_base_directory(user_id)
    if base_dir.exists():
        shutil.rmtree(base_dir)
        logger.info(f"Removed user data directory for user {user_id}: {base_dir}")


#
# End of Users_DB.py
#######################################################################################################################
