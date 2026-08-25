"""Canonical metadata contract for profile-version candidate source tables."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from tldw_Server_API.app.core.AuthNZ.profile_user_write_guard import (
    _execute_postgres_membership_timestamp_repair,
)

PROFILE_CANDIDATE_TABLES = (
    "organizations",
    "teams",
    "org_members",
    "team_members",
    "user_config_overrides",
    "org_config_overrides",
    "team_config_overrides",
)

SQLITE_PROFILE_CANDIDATE_TABLE_STATEMENTS = (
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

PROFILE_CANDIDATE_COLUMNS = {
    "organizations": {
        "id",
        "uuid",
        "name",
        "slug",
        "owner_user_id",
        "is_active",
        "metadata",
        "created_at",
        "updated_at",
    },
    "teams": {
        "id",
        "org_id",
        "name",
        "slug",
        "description",
        "is_active",
        "metadata",
        "created_at",
        "updated_at",
    },
    "org_members": {"org_id", "user_id", "role", "status", "added_at"},
    "team_members": {"team_id", "user_id", "role", "status", "added_at"},
    "user_config_overrides": {
        "user_id",
        "key",
        "value_json",
        "created_at",
        "updated_at",
        "created_by",
        "updated_by",
    },
    "org_config_overrides": {
        "org_id",
        "key",
        "value_json",
        "created_at",
        "updated_at",
        "created_by",
        "updated_by",
    },
    "team_config_overrides": {
        "team_id",
        "key",
        "value_json",
        "created_at",
        "updated_at",
        "created_by",
        "updated_by",
    },
}

PROFILE_CANDIDATE_PRIMARY_KEYS = {
    "organizations": ("id",),
    "teams": ("id",),
    "org_members": ("org_id", "user_id"),
    "team_members": ("team_id", "user_id"),
    "user_config_overrides": ("user_id", "key"),
    "org_config_overrides": ("org_id", "key"),
    "team_config_overrides": ("team_id", "key"),
}

PROFILE_CANDIDATE_UNIQUE_KEYS = {
    "organizations": {("uuid",), ("name",), ("slug",)},
    "teams": {("org_id", "name")},
    "org_members": set(),
    "team_members": set(),
    "user_config_overrides": set(),
    "org_config_overrides": set(),
    "team_config_overrides": set(),
}

PROFILE_CANDIDATE_FOREIGN_KEYS = {
    "organizations": {
        ("owner_user_id", "users", "id", "SET NULL"),
    },
    "teams": {("org_id", "organizations", "id", "CASCADE")},
    "org_members": {
        ("org_id", "organizations", "id", "CASCADE"),
        ("user_id", "users", "id", "CASCADE"),
    },
    "team_members": {
        ("team_id", "teams", "id", "CASCADE"),
        ("user_id", "users", "id", "CASCADE"),
    },
    "user_config_overrides": {("user_id", "users", "id", "CASCADE")},
    "org_config_overrides": {
        ("org_id", "organizations", "id", "CASCADE")
    },
    "team_config_overrides": {("team_id", "teams", "id", "CASCADE")},
}

_POSTGRES_TIMESTAMP_COLUMNS = {
    "organizations": ("created_at", "updated_at"),
    "teams": ("created_at", "updated_at"),
    "org_members": ("added_at",),
    "team_members": ("added_at",),
    "user_config_overrides": ("created_at", "updated_at"),
    "org_config_overrides": ("created_at", "updated_at"),
    "team_config_overrides": ("created_at", "updated_at"),
}

_POSTGRES_VERSION_SOURCE_COLUMNS = {
    "organizations": ("updated_at", "created_at"),
    "teams": ("updated_at", "created_at"),
    "org_members": ("added_at", None),
    "team_members": ("added_at", None),
    "user_config_overrides": ("updated_at", "created_at"),
    "org_config_overrides": ("updated_at", "created_at"),
    "team_config_overrides": ("updated_at", "created_at"),
}

_COLUMN_KINDS = {
    "organizations": {
        "id": "integer",
        "uuid": "identifier",
        "name": "text",
        "slug": "text",
        "owner_user_id": "integer",
        "is_active": "boolean",
        "metadata": "json",
        "created_at": "timestamp",
        "updated_at": "timestamp",
    },
    "teams": {
        "id": "integer",
        "org_id": "integer",
        "name": "text",
        "slug": "text",
        "description": "text",
        "is_active": "boolean",
        "metadata": "json",
        "created_at": "timestamp",
        "updated_at": "timestamp",
    },
    "org_members": {
        "org_id": "integer",
        "user_id": "integer",
        "role": "text",
        "status": "text",
        "added_at": "timestamp",
    },
    "team_members": {
        "team_id": "integer",
        "user_id": "integer",
        "role": "text",
        "status": "text",
        "added_at": "timestamp",
    },
    "user_config_overrides": {
        "user_id": "integer",
        "key": "text",
        "value_json": "text",
        "created_at": "timestamp",
        "updated_at": "timestamp",
        "created_by": "integer",
        "updated_by": "integer",
    },
    "org_config_overrides": {
        "org_id": "integer",
        "key": "text",
        "value_json": "text",
        "created_at": "timestamp",
        "updated_at": "timestamp",
        "created_by": "integer",
        "updated_by": "integer",
    },
    "team_config_overrides": {
        "team_id": "integer",
        "key": "text",
        "value_json": "text",
        "created_at": "timestamp",
        "updated_at": "timestamp",
        "created_by": "integer",
        "updated_by": "integer",
    },
}

_REQUIRED_NOT_NULL = {
    "organizations": {"name", "updated_at"},
    "teams": {"org_id", "name", "updated_at"},
    "org_members": {"org_id", "user_id", "added_at"},
    "team_members": {"team_id", "user_id", "added_at"},
    "user_config_overrides": {"user_id", "key", "updated_at"},
    "org_config_overrides": {"org_id", "key", "updated_at"},
    "team_config_overrides": {"team_id", "key", "updated_at"},
}

_EXPECTED_DEFAULTS = {
    "organizations": {
        "is_active": "true",
        "created_at": "current_timestamp",
        "updated_at": "current_timestamp",
    },
    "teams": {
        "is_active": "true",
        "created_at": "current_timestamp",
        "updated_at": "current_timestamp",
    },
    "org_members": {
        "role": "member",
        "status": "active",
        "added_at": "current_timestamp",
    },
    "team_members": {
        "role": "member",
        "status": "active",
        "added_at": "current_timestamp",
    },
    "user_config_overrides": {
        "created_at": "current_timestamp",
        "updated_at": "current_timestamp",
    },
    "org_config_overrides": {
        "created_at": "current_timestamp",
        "updated_at": "current_timestamp",
    },
    "team_config_overrides": {
        "created_at": "current_timestamp",
        "updated_at": "current_timestamp",
    },
}

_TRAILING_DEFAULT_CAST = re.compile(
    r"\s*::\s*(?:"
    r"text|boolean|jsonb?|regclass|"
    r"varchar(?:\(\d+\))?|character\s+varying(?:\(\d+\))?"
    r")\s*$",
    re.IGNORECASE,
)


def _normalized_type(value: object) -> str:
    return " ".join(str(value or "").strip().upper().split())


def _type_matches(value: object, *, backend: str, kind: str) -> bool:
    normalized = _normalized_type(value)
    if backend == "postgres":
        allowed = {
            "integer": {"INTEGER", "BIGINT"},
            "text": {"TEXT", "CHARACTER VARYING"},
            "identifier": {"UUID", "TEXT", "CHARACTER VARYING"},
            "boolean": {"BOOLEAN"},
            "json": {"JSON", "JSONB"},
            "timestamp": {"TIMESTAMP WITH TIME ZONE"},
        }
    elif backend == "sqlite":
        allowed = {
            "integer": {"INTEGER"},
            "text": {"TEXT"},
            "identifier": {"TEXT"},
            "boolean": {"INTEGER", "BOOLEAN"},
            "json": {"TEXT"},
            "timestamp": {"TEXT", "TIMESTAMP", "DATETIME"},
        }
    else:
        return False
    return normalized in allowed[kind]


def _normalized_default(value: object) -> str | None:
    if value is None:
        return None
    normalized = " ".join(str(value).strip().casefold().split())
    while normalized.startswith("(") and normalized.endswith(")"):
        normalized = normalized[1:-1].strip()
    while cast_match := _TRAILING_DEFAULT_CAST.search(normalized):
        normalized = normalized[: cast_match.start()].strip()
    if normalized.startswith("'") and normalized.endswith("'"):
        normalized = normalized[1:-1]
    return normalized


def _default_matches(value: object, expected: str) -> bool:
    normalized = _normalized_default(value)
    if expected == "current_timestamp":
        return normalized in {"current_timestamp", "now()"}
    if expected == "true":
        return normalized in {"true", "1"}
    return normalized == expected


def _postgres_generated_id_matches(
    table_name: str,
    metadata: Mapping[str, Any],
) -> bool:
    is_identity = metadata.get("is_identity")
    identity_generation = metadata.get("identity_generation")
    if (
        (is_identity is True or str(is_identity).upper() == "YES")
        and str(identity_generation).upper() in {"ALWAYS", "BY DEFAULT"}
    ):
        return True
    default = " ".join(str(metadata.get("default") or "").strip().casefold().split())
    return bool(
        re.fullmatch(
            rf"nextval\('(?:public\.)?{table_name}_id_seq'::regclass\)",
            default,
        )
    )


def profile_candidate_schema_is_valid(
    *,
    backend: str,
    columns: Mapping[str, Mapping[str, Mapping[str, Any]]],
    primary_keys: Mapping[str, tuple[str, ...]],
    unique_keys: Mapping[str, set[tuple[str, ...]]],
    foreign_keys: Mapping[str, set[tuple[str, str, str, str, str]]],
) -> bool:
    """Return whether introspected metadata satisfies the canonical contract."""
    for table_name in PROFILE_CANDIDATE_TABLES:
        table_columns = columns.get(table_name, {})
        if not table_columns.keys() >= PROFILE_CANDIDATE_COLUMNS[table_name]:
            return False
        for column_name, kind in _COLUMN_KINDS[table_name].items():
            metadata = table_columns[column_name]
            if not _type_matches(
                metadata.get("data_type"),
                backend=backend,
                kind=kind,
            ):
                return False
        if any(
            not bool(table_columns[column_name].get("not_null"))
            for column_name in _REQUIRED_NOT_NULL[table_name]
        ):
            return False
        if backend == "postgres" and table_name in {"organizations", "teams"}:
            if not _postgres_generated_id_matches(
                table_name,
                table_columns["id"],
            ):
                return False
        if any(
            not _default_matches(
                table_columns[column_name].get("default"),
                expected,
            )
            for column_name, expected in _EXPECTED_DEFAULTS[table_name].items()
        ):
            return False
        if primary_keys.get(table_name) != PROFILE_CANDIDATE_PRIMARY_KEYS[table_name]:
            return False
        if not unique_keys.get(table_name, set()) >= PROFILE_CANDIDATE_UNIQUE_KEYS[table_name]:
            return False
        expected_schema = "public" if backend == "postgres" else "main"
        try:
            normalized_foreign_keys = {
                (source, schema, target, target_column, action.upper())
                for source, schema, target, target_column, action in foreign_keys.get(
                    table_name,
                    set(),
                )
            }
        except (TypeError, ValueError):
            return False
        expected_foreign_keys = {
            (source, expected_schema, target, target_column, action)
            for source, target, target_column, action in PROFILE_CANDIDATE_FOREIGN_KEYS[
                table_name
            ]
        }
        if normalized_foreign_keys != expected_foreign_keys:
            return False
    return True


def validate_sqlite_profile_candidate_schema(conn: Any) -> None:
    """Validate canonical candidate metadata on one SQLite connection."""
    placeholders = ", ".join("?" for _ in PROFILE_CANDIDATE_TABLES)
    shadow = conn.execute(
        "SELECT name FROM temp.sqlite_master "
        f"WHERE type IN ('table', 'view') AND name IN ({placeholders}) LIMIT 1",  # nosec B608
        PROFILE_CANDIDATE_TABLES,
    ).fetchone()
    if shadow is not None:
        raise RuntimeError("Required profile candidate schema validation failed")

    columns: dict[str, dict[str, dict[str, Any]]] = {}
    primary_keys: dict[str, tuple[str, ...]] = {}
    unique_keys: dict[str, set[tuple[str, ...]]] = {}
    foreign_keys: dict[str, set[tuple[str, str, str, str, str]]] = {}
    for table_name in PROFILE_CANDIDATE_TABLES:
        table_info = conn.execute(
            f'PRAGMA main.table_info("{table_name}")'  # nosec B608
        ).fetchall()
        columns[table_name] = {
            str(row[1]): {
                "data_type": row[2],
                "not_null": bool(row[3]) or int(row[5]) > 0,
                "default": row[4],
            }
            for row in table_info
        }
        primary_keys[table_name] = tuple(
            str(row[1])
            for row in sorted(table_info, key=lambda item: int(item[5]))
            if int(row[5]) > 0
        )

        table_unique_keys: set[tuple[str, ...]] = set()
        index_rows = conn.execute(
            f'PRAGMA main.index_list("{table_name}")'  # nosec B608
        ).fetchall()
        for index_row in index_rows:
            if not bool(index_row[2]) or str(index_row[3]).casefold() == "pk":
                continue
            index_name = str(index_row[1]).replace('"', '""')
            index_info = conn.execute(
                f'PRAGMA main.index_info("{index_name}")'  # nosec B608
            ).fetchall()
            table_unique_keys.add(
                tuple(
                    str(row[2])
                    for row in sorted(index_info, key=lambda item: int(item[0]))
                )
            )
        unique_keys[table_name] = table_unique_keys

        foreign_rows = conn.execute(
            f'PRAGMA main.foreign_key_list("{table_name}")'  # nosec B608
        ).fetchall()
        foreign_keys[table_name] = {
            (str(row[3]), "main", str(row[2]), str(row[4]), str(row[6]))
            for row in foreign_rows
        }

    if not profile_candidate_schema_is_valid(
        backend="sqlite",
        columns=columns,
        primary_keys=primary_keys,
        unique_keys=unique_keys,
        foreign_keys=foreign_keys,
    ):
        raise RuntimeError("Required profile candidate schema validation failed")


async def validate_postgres_profile_candidate_schema(conn: Any) -> None:
    """Validate canonical candidate metadata on one PostgreSQL connection."""
    placeholders = ", ".join(
        f"${position}" for position in range(1, len(PROFILE_CANDIDATE_TABLES) + 1)
    )
    table_filter = f"({placeholders})"
    columns: dict[str, dict[str, dict[str, Any]]] = {
        table_name: {} for table_name in PROFILE_CANDIDATE_TABLES
    }
    primary_key_rows: dict[str, list[tuple[int, str]]] = {
        table_name: [] for table_name in PROFILE_CANDIDATE_TABLES
    }
    unique_key_rows: dict[tuple[str, str], list[tuple[int, str]]] = {}
    foreign_keys: dict[str, set[tuple[str, str, str, str, str]]] = {
        table_name: set() for table_name in PROFILE_CANDIDATE_TABLES
    }

    column_rows = await conn.fetch(
        "SELECT table_name, column_name, data_type, is_nullable, column_default, "
        "is_identity, identity_generation FROM information_schema.columns "
        "WHERE table_schema = 'public' AND table_name IN "
        + table_filter,  # nosec B608 -- fixed-count placeholders only.
        *PROFILE_CANDIDATE_TABLES,
    )
    for row in column_rows:
        columns[str(row["table_name"])][str(row["column_name"])] = {
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
        "WHERE tc.table_schema = 'public' AND tc.constraint_type = 'PRIMARY KEY' "
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
        "WHERE tc.table_schema = 'public' AND tc.constraint_type = 'UNIQUE' "
        "AND tc.table_name IN "
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
        "WHERE tc.table_schema = 'public' AND tc.constraint_type = 'FOREIGN KEY' "
        "AND tc.table_name IN "
        + table_filter,  # nosec B608 -- fixed-count placeholders only.
        *PROFILE_CANDIDATE_TABLES,
    )
    for row in foreign_rows:
        foreign_keys[str(row["table_name"])].add(
            (
                str(row["column_name"]),
                str(row["foreign_table_schema"]),
                str(row["foreign_table_name"]),
                str(row["foreign_column_name"]),
                str(row["delete_rule"]),
            )
        )

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
        backend="postgres",
        columns=columns,
        primary_keys=primary_keys,
        unique_keys=unique_keys,
        foreign_keys=foreign_keys,
    ):
        raise RuntimeError("Required profile candidate schema validation failed")


async def repair_postgres_profile_candidate_timestamps(conn: Any) -> None:
    """Normalize legacy candidate timestamps inside a caller-owned transaction."""
    for table_name, column_names in _POSTGRES_TIMESTAMP_COLUMNS.items():
        for column_name in column_names:
            data_type = await conn.fetchval(
                "SELECT data_type FROM information_schema.columns "
                "WHERE table_schema = 'public' AND table_name = $1 "
                "AND column_name = $2",
                table_name,
                column_name,
            )
            if data_type == "timestamp without time zone":
                await conn.execute(
                    f"ALTER TABLE public.{table_name} ALTER COLUMN {column_name} "  # nosec B608
                    f"TYPE TIMESTAMPTZ USING {column_name} AT TIME ZONE 'UTC'"
                )
            elif data_type != "timestamp with time zone":
                raise RuntimeError(
                    "Required profile candidate timestamp normalization failed"
                )

    for table_name, (
        source_column,
        fallback_column,
    ) in _POSTGRES_VERSION_SOURCE_COLUMNS.items():
        fallback = (
            f"{fallback_column}, CURRENT_TIMESTAMP"
            if fallback_column is not None
            else "CURRENT_TIMESTAMP"
        )
        if table_name in {"org_members", "team_members"}:
            await _execute_postgres_membership_timestamp_repair(
                conn,
                table_name=table_name,
            )
        else:
            await conn.execute(
                f"UPDATE public.{table_name} SET {source_column} = "  # nosec B608
                f"COALESCE({source_column}, {fallback}) "
                f"WHERE {source_column} IS NULL"
            )
        await conn.execute(
            f"ALTER TABLE public.{table_name} ALTER COLUMN {source_column} "  # nosec B608
            "SET DEFAULT CURRENT_TIMESTAMP"
        )
        await conn.execute(
            f"ALTER TABLE public.{table_name} ALTER COLUMN {source_column} "  # nosec B608
            "SET NOT NULL"
        )
