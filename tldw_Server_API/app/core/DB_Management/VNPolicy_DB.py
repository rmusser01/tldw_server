"""VN policy profile storage and per-user VN profile snapshots."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool, get_db_pool
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


VN_POLICY_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS vn_policy_profiles (
    profile_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    description TEXT,
    definition_json TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    builtin BOOLEAN NOT NULL DEFAULT 0,
    disabled BOOLEAN NOT NULL DEFAULT 0,
    created_by_user_id INTEGER,
    updated_by_user_id INTEGER,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS vn_policy_profile_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    display_name TEXT NOT NULL,
    description TEXT,
    definition_json TEXT NOT NULL,
    created_by_user_id INTEGER,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(profile_id, version)
);

CREATE TABLE IF NOT EXISTS vn_generation_profiles (
    profile_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    description TEXT,
    definition_json TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    builtin BOOLEAN NOT NULL DEFAULT 0,
    disabled BOOLEAN NOT NULL DEFAULT 0,
    created_by_user_id INTEGER,
    updated_by_user_id INTEGER,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS vn_generation_profile_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    display_name TEXT NOT NULL,
    description TEXT,
    definition_json TEXT NOT NULL,
    created_by_user_id INTEGER,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(profile_id, version)
);

CREATE TABLE IF NOT EXISTS vn_profile_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_user_id INTEGER NOT NULL,
    snapshot_type TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    profile_version INTEGER NOT NULL,
    resource_type TEXT NOT NULL,
    resource_id INTEGER,
    definition_json TEXT NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_vn_profile_snapshots_owner_resource
    ON vn_profile_snapshots(owner_user_id, resource_type, resource_id);
CREATE INDEX IF NOT EXISTS idx_vn_policy_profiles_disabled
    ON vn_policy_profiles(disabled);
CREATE INDEX IF NOT EXISTS idx_vn_generation_profiles_disabled
    ON vn_generation_profiles(disabled);
"""

VN_POLICY_SCHEMA_STATEMENTS = tuple(
    statement.strip()
    for statement in VN_POLICY_SCHEMA_SQL.split(";")
    if statement.strip()
)

VN_PROFILE_SNAPSHOT_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS vn_profile_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_user_id INTEGER NOT NULL,
    snapshot_type TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    profile_version INTEGER NOT NULL,
    resource_type TEXT NOT NULL,
    resource_id INTEGER,
    definition_json TEXT NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_vn_profile_snapshots_owner_resource
    ON vn_profile_snapshots(owner_user_id, resource_type, resource_id);
"""

VN_PROFILE_SNAPSHOT_SCHEMA_STATEMENTS = tuple(
    statement.strip()
    for statement in VN_PROFILE_SNAPSHOT_SCHEMA_SQL.split(";")
    if statement.strip()
)

VN_PROFILE_DEFINITION_SCHEMA_SQLITE = """
CREATE TABLE IF NOT EXISTS vn_policy_profiles (
    profile_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    description TEXT,
    definition_json TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    builtin BOOLEAN NOT NULL DEFAULT 0,
    disabled BOOLEAN NOT NULL DEFAULT 0,
    created_by_user_id INTEGER,
    updated_by_user_id INTEGER,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS vn_policy_profile_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    display_name TEXT NOT NULL,
    description TEXT,
    definition_json TEXT NOT NULL,
    created_by_user_id INTEGER,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(profile_id, version)
);

CREATE TABLE IF NOT EXISTS vn_generation_profiles (
    profile_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    description TEXT,
    definition_json TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    builtin BOOLEAN NOT NULL DEFAULT 0,
    disabled BOOLEAN NOT NULL DEFAULT 0,
    created_by_user_id INTEGER,
    updated_by_user_id INTEGER,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS vn_generation_profile_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    display_name TEXT NOT NULL,
    description TEXT,
    definition_json TEXT NOT NULL,
    created_by_user_id INTEGER,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(profile_id, version)
);

CREATE INDEX IF NOT EXISTS idx_global_vn_policy_profiles_disabled
    ON vn_policy_profiles(disabled);
CREATE INDEX IF NOT EXISTS idx_global_vn_generation_profiles_disabled
    ON vn_generation_profiles(disabled);
"""

VN_PROFILE_DEFINITION_SCHEMA_POSTGRES = """
CREATE TABLE IF NOT EXISTS vn_policy_profiles (
    profile_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    description TEXT,
    definition_json TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    builtin BOOLEAN NOT NULL DEFAULT FALSE,
    disabled BOOLEAN NOT NULL DEFAULT FALSE,
    created_by_user_id INTEGER,
    updated_by_user_id INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS vn_policy_profile_versions (
    id BIGSERIAL PRIMARY KEY,
    profile_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    display_name TEXT NOT NULL,
    description TEXT,
    definition_json TEXT NOT NULL,
    created_by_user_id INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(profile_id, version)
);

CREATE TABLE IF NOT EXISTS vn_generation_profiles (
    profile_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    description TEXT,
    definition_json TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    builtin BOOLEAN NOT NULL DEFAULT FALSE,
    disabled BOOLEAN NOT NULL DEFAULT FALSE,
    created_by_user_id INTEGER,
    updated_by_user_id INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS vn_generation_profile_versions (
    id BIGSERIAL PRIMARY KEY,
    profile_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    display_name TEXT NOT NULL,
    description TEXT,
    definition_json TEXT NOT NULL,
    created_by_user_id INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(profile_id, version)
);

CREATE INDEX IF NOT EXISTS idx_global_vn_policy_profiles_disabled
    ON vn_policy_profiles(disabled);
CREATE INDEX IF NOT EXISTS idx_global_vn_generation_profiles_disabled
    ON vn_generation_profiles(disabled);
"""

VN_PROFILE_DEFINITION_SCHEMA_SQLITE_STATEMENTS = tuple(
    statement.strip()
    for statement in VN_PROFILE_DEFINITION_SCHEMA_SQLITE.split(";")
    if statement.strip()
)

VN_PROFILE_DEFINITION_SCHEMA_POSTGRES_STATEMENTS = tuple(
    statement.strip()
    for statement in VN_PROFILE_DEFINITION_SCHEMA_POSTGRES.split(";")
    if statement.strip()
)

LOCAL_DEFAULT_POLICY_DEFINITION: dict[str, Any] = {
    "character_safety": {
        "missing": {"general": "warn", "teen": "warn", "suggestive": "block", "mature": "block"},
        "unknown_or_ambiguous": {"general": "warn", "teen": "warn", "suggestive": "block", "mature": "block"},
        "conflicting": {"default": "block"},
        "imported_untrusted": {"general": "warn", "teen": "warn", "suggestive": "warn", "mature": "block"},
    },
    "acknowledgement_required_for_warnings": True,
}

STRICT_HOSTED_POLICY_DEFINITION: dict[str, Any] = {
    "character_safety": {
        "missing": {"default": "block"},
        "unknown_or_ambiguous": {"default": "block"},
        "conflicting": {"default": "block"},
        "imported_untrusted": {"default": "block"},
    },
    "acknowledgement_required_for_warnings": True,
}

STORY_DEFAULT_GENERATION_DEFINITION: dict[str, Any] = {
    "provider": "local",
    "model": "default",
    "supports_structured_output": True,
    "temperature_default": 0.7,
    "temperature_min": 0.0,
    "temperature_max": 1.2,
    "max_output_tokens": 2048,
    "allowed_content_ratings": ["general", "teen", "mature"],
    "max_choices": 4,
    "max_branch_depth": 12,
    "max_model_expansion_scope": "scene",
    "tts_allowed": True,
    "output_persistence_max_days": 30,
    "audit_mode": "metadata",
}

BUILTIN_POLICY_PROFILES = (
    ("local_default", "Local Default", "Local/self-hosted VN policy defaults.", LOCAL_DEFAULT_POLICY_DEFINITION),
    ("strict_hosted", "Strict Hosted", "Fail-closed policy profile for hosted deployments.", STRICT_HOSTED_POLICY_DEFINITION),
)

BUILTIN_GENERATION_PROFILES = (
    ("story_default", "Story Default", "Default structured story generation profile.", STORY_DEFAULT_GENERATION_DEFINITION),
)

ALLOWED_CONTENT_RATINGS = {"general", "teen", "suggestive", "mature"}
ALLOWED_MODEL_EXPANSION_SCOPES = {"none", "turn", "scene", "session"}
ALLOWED_AUDIT_MODES = {"none", "metadata", "full"}


def ensure_vn_policy_tables(db: CharactersRAGDB) -> None:
    """Create VN policy metadata tables in the provided ChaChaNotes database."""
    _require_sqlite_chacha_db(db)
    with db.transaction() as conn:
        for statement in VN_POLICY_SCHEMA_STATEMENTS:
            conn.execute(statement)


def ensure_vn_profile_snapshot_tables(db: CharactersRAGDB) -> None:
    """Create only per-user VN profile snapshot tables in the provided ChaChaNotes database."""
    _require_sqlite_chacha_db(db)
    with db.transaction() as conn:
        for statement in VN_PROFILE_SNAPSHOT_SCHEMA_STATEMENTS:
            conn.execute(statement)


class BuiltinVNPolicyProfileStore:
    """Async profile store exposing built-in definitions without mutating user databases."""

    async def list_policy_profiles(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        include_disabled: bool = False,
    ) -> tuple[list[dict[str, Any]], int]:
        rows = [_builtin_profile_row(*profile) for profile in BUILTIN_POLICY_PROFILES]
        return _slice_profile_rows(rows, limit=limit, offset=offset), len(rows)

    async def list_generation_profiles(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        include_disabled: bool = False,
    ) -> tuple[list[dict[str, Any]], int]:
        rows = [_builtin_profile_row(*profile) for profile in BUILTIN_GENERATION_PROFILES]
        return _slice_profile_rows(rows, limit=limit, offset=offset), len(rows)

    async def get_policy_profile(self, profile_id: str, *, include_disabled: bool = False) -> dict[str, Any] | None:
        return _get_builtin_profile(BUILTIN_POLICY_PROFILES, profile_id)

    async def get_generation_profile(self, profile_id: str, *, include_disabled: bool = False) -> dict[str, Any] | None:
        return _get_builtin_profile(BUILTIN_GENERATION_PROFILES, profile_id)

    async def create_policy_profile(self, **kwargs: Any) -> dict[str, Any]:
        raise ValueError("global_profile_store_required")

    async def update_policy_profile(self, profile_id: str, **kwargs: Any) -> dict[str, Any]:
        raise ValueError("global_profile_store_required")

    async def create_generation_profile(self, **kwargs: Any) -> dict[str, Any]:
        raise ValueError("global_profile_store_required")

    async def update_generation_profile(self, profile_id: str, **kwargs: Any) -> dict[str, Any]:
        raise ValueError("global_profile_store_required")

    async def disable_policy_profile(self, profile_id: str, **kwargs: Any) -> None:
        raise ValueError("global_profile_store_required")

    async def disable_generation_profile(self, profile_id: str, **kwargs: Any) -> None:
        raise ValueError("global_profile_store_required")


class VNProfileSnapshotRepository:
    """Repository for per-user immutable VN profile snapshots."""

    def __init__(self, db: CharactersRAGDB):
        _require_sqlite_chacha_db(db)
        self.db = db
        self._schema_initialized = False

    @classmethod
    def initialized(cls, db: CharactersRAGDB) -> "VNProfileSnapshotRepository":
        repo = cls(db)
        repo.initialize_schema()
        return repo

    def initialize_schema(self) -> None:
        ensure_vn_profile_snapshot_tables(self.db)
        self._schema_initialized = True

    def create_profile_snapshot(
        self,
        *,
        owner_user_id: int,
        snapshot_type: str,
        profile_id: str,
        profile_version: int,
        resource_type: str,
        resource_id: int | None,
        definition: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Create an immutable profile definition snapshot for a user-owned VN resource."""
        self._ensure_schema_initialized()
        if snapshot_type not in {"policy", "generation"}:
            raise ValueError("invalid_snapshot_type")
        with self.db.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO vn_profile_snapshots (
                    owner_user_id,
                    snapshot_type,
                    profile_id,
                    profile_version,
                    resource_type,
                    resource_id,
                    definition_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    owner_user_id,
                    snapshot_type,
                    profile_id,
                    profile_version,
                    resource_type,
                    resource_id,
                    _json_dump(dict(definition)),
                ),
            )
            snapshot_id = int(cursor.lastrowid)
        row = self.get_profile_snapshot(snapshot_id, owner_user_id=owner_user_id)
        if row is None:
            raise RuntimeError("created_profile_snapshot_not_found")
        return row

    def get_profile_snapshot(self, snapshot_id: int, *, owner_user_id: int) -> dict[str, Any] | None:
        """Return a profile snapshot owned by a user."""
        self._ensure_schema_initialized()
        cursor = self.db.execute_query(
            "SELECT * FROM vn_profile_snapshots WHERE id = ? AND owner_user_id = ?",
            (snapshot_id, owner_user_id),
        )
        row = cursor.fetchone()
        return _decode_snapshot(row) if row is not None else None

    def _ensure_schema_initialized(self) -> None:
        if not self._schema_initialized:
            self.initialize_schema()


class VNPolicyProfileStore:
    """Global VN policy and generation profile definitions stored in AuthNZ DB."""

    def __init__(self, db_pool: DatabasePool | None = None):
        self.db_pool = db_pool
        self._initialized = False

    async def initialize(self) -> None:
        """Create profile definition tables and seed built-ins."""
        if self._initialized:
            return
        if self.db_pool is None:
            self.db_pool = await get_db_pool()
        statements = (
            VN_PROFILE_DEFINITION_SCHEMA_POSTGRES_STATEMENTS
            if self._is_postgres_backend()
            else VN_PROFILE_DEFINITION_SCHEMA_SQLITE_STATEMENTS
        )
        async with self.db_pool.transaction() as conn:
            for statement in statements:
                await conn.execute(statement)
        self._initialized = True
        await self._seed_builtin_profiles()

    async def list_policy_profiles(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        include_disabled: bool = False,
    ) -> tuple[list[dict[str, Any]], int]:
        """Return global policy profiles with total count for offset pagination."""
        return await self._list_profiles(
            table_name="vn_policy_profiles",
            limit=limit,
            offset=offset,
            include_disabled=include_disabled,
        )

    async def list_generation_profiles(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        include_disabled: bool = False,
    ) -> tuple[list[dict[str, Any]], int]:
        """Return global generation profiles with total count for offset pagination."""
        return await self._list_profiles(
            table_name="vn_generation_profiles",
            limit=limit,
            offset=offset,
            include_disabled=include_disabled,
        )

    async def get_policy_profile(self, profile_id: str, *, include_disabled: bool = False) -> dict[str, Any] | None:
        """Return a global policy profile by public profile id."""
        return await self._get_profile(
            table_name="vn_policy_profiles",
            profile_id=profile_id,
            include_disabled=include_disabled,
        )

    async def get_generation_profile(self, profile_id: str, *, include_disabled: bool = False) -> dict[str, Any] | None:
        """Return a global generation profile by public profile id."""
        return await self._get_profile(
            table_name="vn_generation_profiles",
            profile_id=profile_id,
            include_disabled=include_disabled,
        )

    async def create_policy_profile(
        self,
        *,
        profile_id: str,
        display_name: str,
        definition: Mapping[str, Any],
        description: str | None = None,
        created_by_user_id: int | None = None,
    ) -> dict[str, Any]:
        """Create a global policy profile and its first immutable version row."""
        _validate_policy_definition(definition)
        return await self._create_profile(
            table_name="vn_policy_profiles",
            version_table_name="vn_policy_profile_versions",
            profile_id=profile_id,
            display_name=display_name,
            description=description,
            definition=definition,
            created_by_user_id=created_by_user_id,
            builtin=False,
        )

    async def update_policy_profile(
        self,
        profile_id: str,
        *,
        display_name: str | None = None,
        description: str | None = None,
        definition: Mapping[str, Any] | None = None,
        updated_by_user_id: int | None = None,
    ) -> dict[str, Any]:
        """Update a global policy profile and append a new version row."""
        if definition is not None:
            _validate_policy_definition(definition)
        return await self._update_profile(
            table_name="vn_policy_profiles",
            version_table_name="vn_policy_profile_versions",
            profile_id=profile_id,
            display_name=display_name,
            description=description,
            definition=definition,
            updated_by_user_id=updated_by_user_id,
        )

    async def create_generation_profile(
        self,
        *,
        profile_id: str,
        display_name: str,
        definition: Mapping[str, Any],
        description: str | None = None,
        created_by_user_id: int | None = None,
    ) -> dict[str, Any]:
        """Create a global generation profile and its first immutable version row."""
        _validate_generation_profile_definition(definition)
        return await self._create_profile(
            table_name="vn_generation_profiles",
            version_table_name="vn_generation_profile_versions",
            profile_id=profile_id,
            display_name=display_name,
            description=description,
            definition=definition,
            created_by_user_id=created_by_user_id,
            builtin=False,
        )

    async def update_generation_profile(
        self,
        profile_id: str,
        *,
        display_name: str | None = None,
        description: str | None = None,
        definition: Mapping[str, Any] | None = None,
        updated_by_user_id: int | None = None,
    ) -> dict[str, Any]:
        """Update a global generation profile and append a new version row."""
        if definition is not None:
            _validate_generation_profile_definition(definition)
        return await self._update_profile(
            table_name="vn_generation_profiles",
            version_table_name="vn_generation_profile_versions",
            profile_id=profile_id,
            display_name=display_name,
            description=description,
            definition=definition,
            updated_by_user_id=updated_by_user_id,
        )

    async def disable_policy_profile(self, profile_id: str, *, updated_by_user_id: int | None = None) -> None:
        """Disable a global policy profile."""
        await self._disable_profile(
            table_name="vn_policy_profiles",
            profile_id=profile_id,
            updated_by_user_id=updated_by_user_id,
        )

    async def disable_generation_profile(self, profile_id: str, *, updated_by_user_id: int | None = None) -> None:
        """Disable a global generation profile."""
        await self._disable_profile(
            table_name="vn_generation_profiles",
            profile_id=profile_id,
            updated_by_user_id=updated_by_user_id,
        )

    async def list_policy_profile_versions(self, profile_id: str) -> list[dict[str, Any]]:
        """Return global policy profile versions in ascending version order."""
        return await self._list_profile_versions("vn_policy_profile_versions", profile_id)

    async def list_generation_profile_versions(self, profile_id: str) -> list[dict[str, Any]]:
        """Return global generation profile versions in ascending version order."""
        return await self._list_profile_versions("vn_generation_profile_versions", profile_id)

    def _is_postgres_backend(self) -> bool:
        db_pool = self.db_pool
        if db_pool is None:
            return False
        if isinstance(db_pool, DatabasePool):
            return getattr(db_pool, "pool", None) is not None
        sqlite_hint = getattr(db_pool, "_is_sqlite", None)
        if isinstance(sqlite_hint, bool):
            return not sqlite_hint
        return getattr(db_pool, "pool", None) is not None

    async def _execute_tx(self, conn: Any, query: str, *args: Any) -> Any:
        sql = _question_mark_to_dollar(query) if self._is_postgres_backend() else query
        return await conn.execute(sql, *args)

    async def _seed_builtin_profiles(self) -> None:
        for profile_id, display_name, description, definition in BUILTIN_POLICY_PROFILES:
            if await self.get_policy_profile(profile_id, include_disabled=True) is None:
                await self._create_profile(
                    table_name="vn_policy_profiles",
                    version_table_name="vn_policy_profile_versions",
                    profile_id=profile_id,
                    display_name=display_name,
                    description=description,
                    definition=definition,
                    created_by_user_id=None,
                    builtin=True,
                )
        for profile_id, display_name, description, definition in BUILTIN_GENERATION_PROFILES:
            if await self.get_generation_profile(profile_id, include_disabled=True) is None:
                await self._create_profile(
                    table_name="vn_generation_profiles",
                    version_table_name="vn_generation_profile_versions",
                    profile_id=profile_id,
                    display_name=display_name,
                    description=description,
                    definition=definition,
                    created_by_user_id=None,
                    builtin=True,
                )

    async def _list_profiles(
        self,
        *,
        table_name: str,
        limit: int,
        offset: int,
        include_disabled: bool,
    ) -> tuple[list[dict[str, Any]], int]:
        await self._ensure_initialized()
        _validate_table_name(table_name)
        where_clause = "" if include_disabled else "WHERE disabled = 0"
        total_row = await self.db_pool.fetchone(
            f"SELECT COUNT(*) AS total FROM {table_name} {where_clause}"  # nosec B608
        )
        total = _row_int(total_row, "total", 0)
        rows = await self.db_pool.fetchall(
            f"""
            SELECT *
            FROM {table_name}
            {where_clause}
            ORDER BY builtin DESC, profile_id ASC
            LIMIT ? OFFSET ?
            """,  # nosec B608
            max(1, min(int(limit), 100)),
            max(0, int(offset)),
        )
        return [_decode_profile(row) for row in rows], total

    async def _get_profile(
        self,
        *,
        table_name: str,
        profile_id: str,
        include_disabled: bool,
    ) -> dict[str, Any] | None:
        await self._ensure_initialized()
        _validate_table_name(table_name)
        where_clause = "profile_id = ?" if include_disabled else "profile_id = ? AND disabled = 0"
        row = await self.db_pool.fetchone(
            f"SELECT * FROM {table_name} WHERE {where_clause}",  # nosec B608
            profile_id,
        )
        return _decode_profile(row) if row is not None else None

    async def _create_profile(
        self,
        *,
        table_name: str,
        version_table_name: str,
        profile_id: str,
        display_name: str,
        description: str | None,
        definition: Mapping[str, Any],
        created_by_user_id: int | None,
        builtin: bool,
    ) -> dict[str, Any]:
        await self._ensure_initialized()
        _validate_table_name(table_name)
        _validate_version_table_name(version_table_name)
        if await self._get_profile(table_name=table_name, profile_id=profile_id, include_disabled=True) is not None:
            raise ValueError("profile_already_exists")
        async with self.db_pool.transaction() as conn:
            await self._execute_tx(
                conn,
                f"""
                INSERT INTO {table_name} (
                    profile_id,
                    display_name,
                    description,
                    definition_json,
                    version,
                    builtin,
                    created_by_user_id,
                    updated_by_user_id
                )
                VALUES (?, ?, ?, ?, 1, ?, ?, ?)
                """,  # nosec B608
                profile_id,
                display_name,
                description,
                _json_dump(dict(definition)),
                bool(builtin),
                created_by_user_id,
                created_by_user_id,
            )
            await self._insert_version_row(
                conn,
                version_table_name=version_table_name,
                profile_id=profile_id,
                version=1,
                display_name=display_name,
                description=description,
                definition=definition,
                created_by_user_id=created_by_user_id,
            )
        row = await self._get_profile(table_name=table_name, profile_id=profile_id, include_disabled=True)
        if row is None:
            raise RuntimeError("created_profile_not_found")
        return row

    async def _update_profile(
        self,
        *,
        table_name: str,
        version_table_name: str,
        profile_id: str,
        display_name: str | None,
        description: str | None,
        definition: Mapping[str, Any] | None,
        updated_by_user_id: int | None,
    ) -> dict[str, Any]:
        await self._ensure_initialized()
        _validate_table_name(table_name)
        _validate_version_table_name(version_table_name)
        existing = await self._get_profile(table_name=table_name, profile_id=profile_id, include_disabled=True)
        if existing is None:
            raise ValueError("profile_not_found")
        if bool(existing.get("builtin")):
            raise ValueError("builtin_profile_immutable")
        new_display_name = display_name if display_name is not None else str(existing["display_name"])
        new_description = description if description is not None else existing.get("description")
        new_definition = dict(definition) if definition is not None else dict(existing["definition"])
        new_version = int(existing["version"]) + 1
        async with self.db_pool.transaction() as conn:
            await self._execute_tx(
                conn,
                f"""
                UPDATE {table_name}
                SET display_name = ?,
                    description = ?,
                    definition_json = ?,
                    version = ?,
                    updated_by_user_id = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE profile_id = ?
                """,  # nosec B608
                new_display_name,
                new_description,
                _json_dump(new_definition),
                new_version,
                updated_by_user_id,
                profile_id,
            )
            await self._insert_version_row(
                conn,
                version_table_name=version_table_name,
                profile_id=profile_id,
                version=new_version,
                display_name=new_display_name,
                description=new_description,
                definition=new_definition,
                created_by_user_id=updated_by_user_id,
            )
        row = await self._get_profile(table_name=table_name, profile_id=profile_id, include_disabled=True)
        if row is None:
            raise RuntimeError("updated_profile_not_found")
        return row

    async def _disable_profile(
        self,
        *,
        table_name: str,
        profile_id: str,
        updated_by_user_id: int | None,
    ) -> None:
        await self._ensure_initialized()
        _validate_table_name(table_name)
        existing = await self._get_profile(table_name=table_name, profile_id=profile_id, include_disabled=True)
        if existing is None:
            raise ValueError("profile_not_found")
        if bool(existing.get("builtin")):
            raise ValueError("builtin_profile_immutable")
        result = await self.db_pool.execute(
            f"""
            UPDATE {table_name}
            SET disabled = 1,
                updated_by_user_id = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE profile_id = ?
            """,  # nosec B608
            updated_by_user_id,
            profile_id,
        )
        if _affected_rows(result) == 0:
            raise ValueError("profile_not_found")

    async def _list_profile_versions(self, version_table_name: str, profile_id: str) -> list[dict[str, Any]]:
        await self._ensure_initialized()
        _validate_version_table_name(version_table_name)
        rows = await self.db_pool.fetchall(
            f"""
            SELECT *
            FROM {version_table_name}
            WHERE profile_id = ?
            ORDER BY version ASC
            """,  # nosec B608
            profile_id,
        )
        return [_decode_profile_version(row) for row in rows]

    async def _insert_version_row(
        self,
        conn: Any,
        *,
        version_table_name: str,
        profile_id: str,
        version: int,
        display_name: str,
        description: str | None,
        definition: Mapping[str, Any],
        created_by_user_id: int | None,
    ) -> None:
        _validate_version_table_name(version_table_name)
        await self._execute_tx(
            conn,
            f"""
            INSERT INTO {version_table_name} (
                profile_id,
                version,
                display_name,
                description,
                definition_json,
                created_by_user_id
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,  # nosec B608
            profile_id,
            version,
            display_name,
            description,
            _json_dump(dict(definition)),
            created_by_user_id,
        )

    async def _ensure_initialized(self) -> None:
        if not self._initialized:
            await self.initialize()


class SyncVNPolicyProfileStoreAdapter:
    """Async facade over the local repository for unit tests and direct service use."""

    def __init__(self, repo: "VNPolicyRepository"):
        self._repo = repo

    async def list_policy_profiles(self, **kwargs: Any) -> tuple[list[dict[str, Any]], int]:
        return self._repo.list_policy_profiles(**kwargs)

    async def list_generation_profiles(self, **kwargs: Any) -> tuple[list[dict[str, Any]], int]:
        return self._repo.list_generation_profiles(**kwargs)

    async def get_policy_profile(self, profile_id: str, **kwargs: Any) -> dict[str, Any] | None:
        return self._repo.get_policy_profile(profile_id, **kwargs)

    async def get_generation_profile(self, profile_id: str, **kwargs: Any) -> dict[str, Any] | None:
        return self._repo.get_generation_profile(profile_id, **kwargs)

    async def create_policy_profile(self, **kwargs: Any) -> dict[str, Any]:
        return self._repo.create_policy_profile(**kwargs)

    async def update_policy_profile(self, profile_id: str, **kwargs: Any) -> dict[str, Any]:
        return self._repo.update_policy_profile(profile_id, **kwargs)

    async def create_generation_profile(self, **kwargs: Any) -> dict[str, Any]:
        return self._repo.create_generation_profile(**kwargs)

    async def update_generation_profile(self, profile_id: str, **kwargs: Any) -> dict[str, Any]:
        return self._repo.update_generation_profile(profile_id, **kwargs)

    async def disable_policy_profile(self, profile_id: str, **kwargs: Any) -> None:
        self._repo.disable_policy_profile(profile_id, **kwargs)

    async def disable_generation_profile(self, profile_id: str, **kwargs: Any) -> None:
        self._repo.disable_generation_profile(profile_id, **kwargs)


class VNPolicyRepository:
    """Repository for VN policy and generation profile metadata."""

    def __init__(self, db: CharactersRAGDB):
        _require_sqlite_chacha_db(db)
        self.db = db
        self._schema_initialized = False

    @classmethod
    def initialized(cls, db: CharactersRAGDB) -> "VNPolicyRepository":
        repo = cls(db)
        repo.initialize_schema()
        return repo

    def initialize_schema(self) -> None:
        ensure_vn_policy_tables(self.db)
        self._schema_initialized = True
        self._seed_builtin_profiles()

    def list_policy_profiles(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        include_disabled: bool = False,
    ) -> tuple[list[dict[str, Any]], int]:
        """Return policy profiles with total count for offset pagination."""
        return self._list_profiles(
            table_name="vn_policy_profiles",
            limit=limit,
            offset=offset,
            include_disabled=include_disabled,
        )

    def list_generation_profiles(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        include_disabled: bool = False,
    ) -> tuple[list[dict[str, Any]], int]:
        """Return generation profiles with total count for offset pagination."""
        return self._list_profiles(
            table_name="vn_generation_profiles",
            limit=limit,
            offset=offset,
            include_disabled=include_disabled,
        )

    def get_policy_profile(self, profile_id: str, *, include_disabled: bool = False) -> dict[str, Any] | None:
        """Return a policy profile by public profile id."""
        return self._get_profile(
            table_name="vn_policy_profiles",
            profile_id=profile_id,
            include_disabled=include_disabled,
        )

    def get_generation_profile(self, profile_id: str, *, include_disabled: bool = False) -> dict[str, Any] | None:
        """Return a generation profile by public profile id."""
        return self._get_profile(
            table_name="vn_generation_profiles",
            profile_id=profile_id,
            include_disabled=include_disabled,
        )

    def create_policy_profile(
        self,
        *,
        profile_id: str,
        display_name: str,
        definition: Mapping[str, Any],
        description: str | None = None,
        created_by_user_id: int | None = None,
    ) -> dict[str, Any]:
        """Create a policy profile and its first immutable version row."""
        _validate_policy_definition(definition)
        return self._create_profile(
            table_name="vn_policy_profiles",
            version_table_name="vn_policy_profile_versions",
            profile_id=profile_id,
            display_name=display_name,
            description=description,
            definition=definition,
            created_by_user_id=created_by_user_id,
        )

    def update_policy_profile(
        self,
        profile_id: str,
        *,
        display_name: str | None = None,
        description: str | None = None,
        definition: Mapping[str, Any] | None = None,
        updated_by_user_id: int | None = None,
    ) -> dict[str, Any]:
        """Update a policy profile and append a new version row."""
        if definition is not None:
            _validate_policy_definition(definition)
        return self._update_profile(
            table_name="vn_policy_profiles",
            version_table_name="vn_policy_profile_versions",
            profile_id=profile_id,
            display_name=display_name,
            description=description,
            definition=definition,
            updated_by_user_id=updated_by_user_id,
        )

    def create_generation_profile(
        self,
        *,
        profile_id: str,
        display_name: str,
        definition: Mapping[str, Any],
        description: str | None = None,
        created_by_user_id: int | None = None,
    ) -> dict[str, Any]:
        """Create a generation profile and its first immutable version row."""
        _validate_generation_profile_definition(definition)
        return self._create_profile(
            table_name="vn_generation_profiles",
            version_table_name="vn_generation_profile_versions",
            profile_id=profile_id,
            display_name=display_name,
            description=description,
            definition=definition,
            created_by_user_id=created_by_user_id,
        )

    def update_generation_profile(
        self,
        profile_id: str,
        *,
        display_name: str | None = None,
        description: str | None = None,
        definition: Mapping[str, Any] | None = None,
        updated_by_user_id: int | None = None,
    ) -> dict[str, Any]:
        """Update a generation profile and append a new version row."""
        if definition is not None:
            _validate_generation_profile_definition(definition)
        return self._update_profile(
            table_name="vn_generation_profiles",
            version_table_name="vn_generation_profile_versions",
            profile_id=profile_id,
            display_name=display_name,
            description=description,
            definition=definition,
            updated_by_user_id=updated_by_user_id,
        )

    def disable_policy_profile(self, profile_id: str, *, updated_by_user_id: int | None = None) -> None:
        """Disable a policy profile so normal list/read calls no longer expose it."""
        self._disable_profile(
            table_name="vn_policy_profiles",
            profile_id=profile_id,
            updated_by_user_id=updated_by_user_id,
        )

    def disable_generation_profile(self, profile_id: str, *, updated_by_user_id: int | None = None) -> None:
        """Disable a generation profile so normal list/read calls no longer expose it."""
        self._disable_profile(
            table_name="vn_generation_profiles",
            profile_id=profile_id,
            updated_by_user_id=updated_by_user_id,
        )

    def list_policy_profile_versions(self, profile_id: str) -> list[dict[str, Any]]:
        """Return all stored policy profile versions in ascending version order."""
        return self._list_profile_versions("vn_policy_profile_versions", profile_id)

    def list_generation_profile_versions(self, profile_id: str) -> list[dict[str, Any]]:
        """Return all stored generation profile versions in ascending version order."""
        return self._list_profile_versions("vn_generation_profile_versions", profile_id)

    def create_profile_snapshot(
        self,
        *,
        owner_user_id: int,
        snapshot_type: str,
        profile_id: str,
        profile_version: int,
        resource_type: str,
        resource_id: int | None,
        definition: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Create an immutable profile definition snapshot for a user-owned VN resource."""
        self._ensure_schema_initialized()
        if snapshot_type not in {"policy", "generation"}:
            raise ValueError("invalid_snapshot_type")
        with self.db.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO vn_profile_snapshots (
                    owner_user_id,
                    snapshot_type,
                    profile_id,
                    profile_version,
                    resource_type,
                    resource_id,
                    definition_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    owner_user_id,
                    snapshot_type,
                    profile_id,
                    profile_version,
                    resource_type,
                    resource_id,
                    _json_dump(dict(definition)),
                ),
            )
            snapshot_id = int(cursor.lastrowid)
        row = self.get_profile_snapshot(snapshot_id, owner_user_id=owner_user_id)
        if row is None:
            raise RuntimeError("created_profile_snapshot_not_found")
        return row

    def get_profile_snapshot(self, snapshot_id: int, *, owner_user_id: int) -> dict[str, Any] | None:
        """Return a profile snapshot owned by a user."""
        self._ensure_schema_initialized()
        cursor = self.db.execute_query(
            "SELECT * FROM vn_profile_snapshots WHERE id = ? AND owner_user_id = ?",
            (snapshot_id, owner_user_id),
        )
        row = cursor.fetchone()
        return _decode_snapshot(row) if row is not None else None

    def _seed_builtin_profiles(self) -> None:
        for profile_id, display_name, description, definition in BUILTIN_POLICY_PROFILES:
            if self.get_policy_profile(profile_id, include_disabled=True) is None:
                self.create_policy_profile(
                    profile_id=profile_id,
                    display_name=display_name,
                    description=description,
                    definition=definition,
                    created_by_user_id=None,
                )
                with self.db.transaction() as conn:
                    conn.execute(
                        "UPDATE vn_policy_profiles SET builtin = 1 WHERE profile_id = ?",
                        (profile_id,),
                    )
        for profile_id, display_name, description, definition in BUILTIN_GENERATION_PROFILES:
            if self.get_generation_profile(profile_id, include_disabled=True) is None:
                self.create_generation_profile(
                    profile_id=profile_id,
                    display_name=display_name,
                    description=description,
                    definition=definition,
                    created_by_user_id=None,
                )
                with self.db.transaction() as conn:
                    conn.execute(
                        "UPDATE vn_generation_profiles SET builtin = 1 WHERE profile_id = ?",
                        (profile_id,),
                    )

    def _list_profiles(
        self,
        *,
        table_name: str,
        limit: int,
        offset: int,
        include_disabled: bool,
    ) -> tuple[list[dict[str, Any]], int]:
        self._ensure_schema_initialized()
        _validate_table_name(table_name)
        where_clause = "" if include_disabled else "WHERE disabled = 0"
        total_cursor = self.db.execute_query(
            f"SELECT COUNT(*) FROM {table_name} {where_clause}"  # nosec B608
        )
        total = int(total_cursor.fetchone()[0])
        cursor = self.db.execute_query(
            f"""
            SELECT *
            FROM {table_name}
            {where_clause}
            ORDER BY builtin DESC, profile_id ASC
            LIMIT ? OFFSET ?
            """,  # nosec B608
            (max(1, min(int(limit), 100)), max(0, int(offset))),
        )
        return [_decode_profile(row) for row in cursor.fetchall()], total

    def _get_profile(
        self,
        *,
        table_name: str,
        profile_id: str,
        include_disabled: bool,
    ) -> dict[str, Any] | None:
        self._ensure_schema_initialized()
        _validate_table_name(table_name)
        where_clause = "profile_id = ?" if include_disabled else "profile_id = ? AND disabled = 0"
        cursor = self.db.execute_query(
            f"SELECT * FROM {table_name} WHERE {where_clause}",  # nosec B608
            (profile_id,),
        )
        row = cursor.fetchone()
        return _decode_profile(row) if row is not None else None

    def _create_profile(
        self,
        *,
        table_name: str,
        version_table_name: str,
        profile_id: str,
        display_name: str,
        description: str | None,
        definition: Mapping[str, Any],
        created_by_user_id: int | None,
    ) -> dict[str, Any]:
        self._ensure_schema_initialized()
        _validate_table_name(table_name)
        _validate_version_table_name(version_table_name)
        with self.db.transaction() as conn:
            conn.execute(
                f"""
                INSERT INTO {table_name} (
                    profile_id,
                    display_name,
                    description,
                    definition_json,
                    version,
                    created_by_user_id,
                    updated_by_user_id
                )
                VALUES (?, ?, ?, ?, 1, ?, ?)
                """,  # nosec B608
                (
                    profile_id,
                    display_name,
                    description,
                    _json_dump(dict(definition)),
                    created_by_user_id,
                    created_by_user_id,
                ),
            )
            self._insert_version_row(
                conn,
                version_table_name=version_table_name,
                profile_id=profile_id,
                version=1,
                display_name=display_name,
                description=description,
                definition=definition,
                created_by_user_id=created_by_user_id,
            )
        row = self._get_profile(table_name=table_name, profile_id=profile_id, include_disabled=True)
        if row is None:
            raise RuntimeError("created_profile_not_found")
        return row

    def _update_profile(
        self,
        *,
        table_name: str,
        version_table_name: str,
        profile_id: str,
        display_name: str | None,
        description: str | None,
        definition: Mapping[str, Any] | None,
        updated_by_user_id: int | None,
    ) -> dict[str, Any]:
        self._ensure_schema_initialized()
        _validate_table_name(table_name)
        _validate_version_table_name(version_table_name)
        existing = self._get_profile(table_name=table_name, profile_id=profile_id, include_disabled=True)
        if existing is None:
            raise ValueError("profile_not_found")
        new_display_name = display_name if display_name is not None else str(existing["display_name"])
        new_description = description if description is not None else existing.get("description")
        new_definition = dict(definition) if definition is not None else dict(existing["definition"])
        new_version = int(existing["version"]) + 1
        with self.db.transaction() as conn:
            conn.execute(
                f"""
                UPDATE {table_name}
                SET display_name = ?,
                    description = ?,
                    definition_json = ?,
                    version = ?,
                    updated_by_user_id = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE profile_id = ?
                """,  # nosec B608
                (
                    new_display_name,
                    new_description,
                    _json_dump(new_definition),
                    new_version,
                    updated_by_user_id,
                    profile_id,
                ),
            )
            self._insert_version_row(
                conn,
                version_table_name=version_table_name,
                profile_id=profile_id,
                version=new_version,
                display_name=new_display_name,
                description=new_description,
                definition=new_definition,
                created_by_user_id=updated_by_user_id,
            )
        row = self._get_profile(table_name=table_name, profile_id=profile_id, include_disabled=True)
        if row is None:
            raise RuntimeError("updated_profile_not_found")
        return row

    def _disable_profile(
        self,
        *,
        table_name: str,
        profile_id: str,
        updated_by_user_id: int | None,
    ) -> None:
        self._ensure_schema_initialized()
        _validate_table_name(table_name)
        with self.db.transaction() as conn:
            cursor = conn.execute(
                f"""
                UPDATE {table_name}
                SET disabled = 1,
                    updated_by_user_id = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE profile_id = ?
                """,  # nosec B608
                (updated_by_user_id, profile_id),
            )
            if cursor.rowcount == 0:
                raise ValueError("profile_not_found")

    def _list_profile_versions(self, version_table_name: str, profile_id: str) -> list[dict[str, Any]]:
        self._ensure_schema_initialized()
        _validate_version_table_name(version_table_name)
        cursor = self.db.execute_query(
            f"""
            SELECT *
            FROM {version_table_name}
            WHERE profile_id = ?
            ORDER BY version ASC
            """,  # nosec B608
            (profile_id,),
        )
        return [_decode_profile_version(row) for row in cursor.fetchall()]

    def _insert_version_row(
        self,
        conn: Any,
        *,
        version_table_name: str,
        profile_id: str,
        version: int,
        display_name: str,
        description: str | None,
        definition: Mapping[str, Any],
        created_by_user_id: int | None,
    ) -> None:
        _validate_version_table_name(version_table_name)
        conn.execute(
            f"""
            INSERT INTO {version_table_name} (
                profile_id,
                version,
                display_name,
                description,
                definition_json,
                created_by_user_id
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,  # nosec B608
            (
                profile_id,
                version,
                display_name,
                description,
                _json_dump(dict(definition)),
                created_by_user_id,
            ),
        )

    def _ensure_schema_initialized(self) -> None:
        if not self._schema_initialized:
            self.initialize_schema()


def _validate_policy_definition(definition: Mapping[str, Any]) -> None:
    character_safety = definition.get("character_safety")
    if not isinstance(character_safety, Mapping):
        raise ValueError("invalid_policy_profile")
    for key in ("missing", "unknown_or_ambiguous", "conflicting", "imported_untrusted"):
        if not isinstance(character_safety.get(key), Mapping):
            raise ValueError("invalid_policy_profile")


def _validate_generation_profile_definition(definition: Mapping[str, Any]) -> None:
    try:
        temperature_default = float(definition["temperature_default"])
        temperature_min = float(definition["temperature_min"])
        temperature_max = float(definition["temperature_max"])
        max_choices = int(definition["max_choices"])
        max_branch_depth = int(definition["max_branch_depth"])
        max_output_tokens = int(definition["max_output_tokens"])
        output_persistence_max_days = int(definition["output_persistence_max_days"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid_generation_profile") from exc

    ratings = definition.get("allowed_content_ratings")
    expansion_scope = str(definition.get("max_model_expansion_scope") or "")
    audit_mode = str(definition.get("audit_mode") or "")
    if (
        not str(definition.get("provider") or "").strip()
        or not str(definition.get("model") or "").strip()
        or not isinstance(definition.get("supports_structured_output"), bool)
        or not isinstance(definition.get("tts_allowed"), bool)
        or temperature_min < 0
        or temperature_max < temperature_min
        or not (temperature_min <= temperature_default <= temperature_max)
        or max_output_tokens < 1
        or max_choices < 1
        or max_branch_depth < 1
        or output_persistence_max_days < 0
        or not isinstance(ratings, list)
        or not ratings
        or any(str(rating) not in ALLOWED_CONTENT_RATINGS for rating in ratings)
        or expansion_scope not in ALLOWED_MODEL_EXPANSION_SCOPES
        or audit_mode not in ALLOWED_AUDIT_MODES
    ):
        raise ValueError("invalid_generation_profile")


def _decode_profile(row: Any) -> dict[str, Any]:
    data = dict(row)
    data["definition"] = _json_loads(data.pop("definition_json"), {})
    data["builtin"] = bool(data["builtin"])
    data["disabled"] = bool(data["disabled"])
    return data


def _decode_profile_version(row: Any) -> dict[str, Any]:
    data = dict(row)
    data["definition"] = _json_loads(data.pop("definition_json"), {})
    return data


def _decode_snapshot(row: Any) -> dict[str, Any]:
    data = dict(row)
    data["definition"] = _json_loads(data.pop("definition_json"), {})
    return data


def _builtin_profile_row(
    profile_id: str,
    display_name: str,
    description: str,
    definition: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "profile_id": profile_id,
        "display_name": display_name,
        "description": description,
        "definition": dict(definition),
        "version": 1,
        "builtin": True,
        "disabled": False,
        "created_by_user_id": None,
        "updated_by_user_id": None,
        "created_at": None,
        "updated_at": None,
    }


def _get_builtin_profile(
    profiles: tuple[tuple[str, str, str, dict[str, Any]], ...],
    profile_id: str,
) -> dict[str, Any] | None:
    for profile in profiles:
        if profile[0] == profile_id:
            return _builtin_profile_row(*profile)
    return None


def _slice_profile_rows(rows: list[dict[str, Any]], *, limit: int, offset: int) -> list[dict[str, Any]]:
    bounded_limit = max(1, min(int(limit), 100))
    bounded_offset = max(0, int(offset))
    return rows[bounded_offset : bounded_offset + bounded_limit]


def _json_dump(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _json_loads(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    try:
        return json.loads(str(value))
    except (TypeError, ValueError):
        return default


def _row_int(row: Any, key: str, default: int) -> int:
    if row is None:
        return default
    try:
        if isinstance(row, Mapping):
            return int(row.get(key, default))
        return int(row[key])
    except (IndexError, KeyError, TypeError, ValueError):
        try:
            return int(row[0])
        except (IndexError, KeyError, TypeError, ValueError):
            return default


def _affected_rows(result: Any) -> int:
    if isinstance(result, str):
        parts = result.strip().split()
        if len(parts) >= 2 and parts[-1].isdigit():
            return int(parts[-1])
        return 0
    rowcount = getattr(result, "rowcount", None)
    if isinstance(rowcount, int):
        return max(0, rowcount)
    return 0


def _question_mark_to_dollar(query: str) -> str:
    index = 0
    parts: list[str] = []
    for char in query:
        if char == "?":
            index += 1
            parts.append(f"${index}")
        else:
            parts.append(char)
    return "".join(parts)


def _require_sqlite_chacha_db(db: CharactersRAGDB) -> None:
    if getattr(db, "backend_type", None) != BackendType.SQLITE:
        raise NotImplementedError(
            "VN policy metadata currently supports SQLite ChaChaNotes databases only."
        )


_PROFILE_TABLES = {"vn_policy_profiles", "vn_generation_profiles"}
_VERSION_TABLES = {"vn_policy_profile_versions", "vn_generation_profile_versions"}


def _validate_table_name(table_name: str) -> None:
    if table_name not in _PROFILE_TABLES:
        raise ValueError("unsupported_profile_table")


def _validate_version_table_name(table_name: str) -> None:
    if table_name not in _VERSION_TABLES:
        raise ValueError("unsupported_profile_version_table")
