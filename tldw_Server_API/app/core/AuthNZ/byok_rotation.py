from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.byok_runtime import openai_credential_mutation_lock
from tldw_Server_API.app.core.AuthNZ.database import DatabasePool, get_db_pool
from tldw_Server_API.app.core.AuthNZ.repos.user_provider_secrets_repo import (
    AuthnzUserProviderSecretsRepo,
)
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import (
    decrypt_byok_payload,
    dumps_envelope,
    encrypt_byok_payload,
    loads_envelope,
)
from tldw_Server_API.app.core.LLM_Calls.provider_identity import canonical_provider_name

_ROTATION_TABLES = frozenset({"user_provider_secrets", "org_provider_secrets"})
_OPENAI_PROVIDER = "openai"


@dataclass
class RotationStats:
    processed: int = 0
    updated: int = 0
    skipped: int = 0
    failed: int = 0

    def add(self, other: RotationStats) -> None:
        self.processed += other.processed
        self.updated += other.updated
        self.skipped += other.skipped
        self.failed += other.failed


@dataclass
class RotationSummary:
    tables: dict[str, RotationStats]
    total: RotationStats
    dry_run: bool = False


def _is_postgres_pool(pool: DatabasePool) -> bool:
    """Return backend type from DatabasePool state."""
    return getattr(pool, "pool", None) is not None


def _extract_row_fields(
    row: Any,
    *,
    table: str,
) -> tuple[int, str | None, int | None, str | None]:
    if isinstance(row, dict):
        materialized = row
    else:
        materialized = {key: row[key] for key in row.keys()}  # noqa: SIM118 - aiosqlite.Row iterates values
    user_id = materialized.get("user_id") if table == "user_provider_secrets" else None
    provider = materialized.get("provider") if table == "user_provider_secrets" else None
    return (
        int(materialized.get("id")),
        materialized.get("encrypted_blob"),
        int(user_id) if user_id is not None else None,
        str(provider) if provider is not None else None,
    )


def _validated_table(table: str) -> str:
    if table not in _ROTATION_TABLES:
        raise ValueError("Unsupported BYOK rotation table")
    return table


async def _fetch_rows(
    conn: Any,
    *,
    table: str,
    last_id: int,
    batch_size: int,
    is_postgres: bool,
) -> list[Any]:
    table = _validated_table(table)
    columns = "id, encrypted_blob, user_id, provider" if table == "user_provider_secrets" else "id, encrypted_blob"
    if is_postgres:
        fetch_rows_sql_template = """
            SELECT {columns}
            FROM {table}
            WHERE id > $1
            ORDER BY id
            LIMIT $2
        """
        query = fetch_rows_sql_template.format_map(locals())  # nosec B608
        return await conn.fetch(query, last_id, batch_size)

    fetch_rows_sql_template = """
        SELECT {columns}
        FROM {table}
        WHERE id > ?
        ORDER BY id
        LIMIT ?
    """
    query = fetch_rows_sql_template.format_map(locals())  # nosec B608
    cursor = await conn.execute(query, last_id, batch_size)
    return list(await cursor.fetchall())


async def _fetch_row_by_id(
    executor: Any,
    *,
    table: str,
    row_id: int,
    is_postgres: bool,
) -> dict[str, Any] | None:
    table = _validated_table(table)
    columns = "id, encrypted_blob, user_id, provider" if table == "user_provider_secrets" else "id, encrypted_blob"
    if is_postgres:
        query_template = "SELECT {columns} FROM {table} WHERE id = $1"
        query = query_template.format_map(locals())  # nosec B608
        row = await executor.fetchone(query, row_id)
    else:
        query_template = "SELECT {columns} FROM {table} WHERE id = ?"
        query = query_template.format_map(locals())  # nosec B608
        row = await executor.fetchone(query, row_id)
    return dict(row) if row else None


def _update_changed_one(result: Any) -> bool:
    rowcount = getattr(result, "rowcount", None)
    if isinstance(rowcount, int):
        return rowcount == 1
    if isinstance(result, str):
        return result.rsplit(" ", 1)[-1] == "1"
    return False


async def _apply_update_if_unchanged(
    executor: Any,
    *,
    table: str,
    updated_blob: str,
    row_id: int,
    expected_blob: str,
    is_postgres: bool,
) -> bool:
    """Replace one encrypted blob only if the snapshot is still current."""
    table = _validated_table(table)

    if is_postgres:
        update_blob_sql_template = (
            "UPDATE {table} SET encrypted_blob = $1 WHERE id = $2 AND encrypted_blob = $3"
        )
        query = update_blob_sql_template.format_map(locals())  # nosec B608
        result = await executor.execute(query, updated_blob, row_id, expected_blob)
        return _update_changed_one(result)

    update_blob_sql_template = (
        "UPDATE {table} SET encrypted_blob = ? WHERE id = ? AND encrypted_blob = ?"
    )
    query = update_blob_sql_template.format_map(locals())  # nosec B608
    result = await executor.execute(query, (updated_blob, row_id, expected_blob))
    return _update_changed_one(result)


def _reencrypt_blob(encrypted_blob: str) -> str:
    payload = decrypt_byok_payload(loads_envelope(encrypted_blob))
    return dumps_envelope(encrypt_byok_payload(payload))


async def _rotate_openai_user_row(
    *,
    pool: DatabasePool,
    row_id: int,
    user_id: int,
    is_postgres: bool,
) -> str:
    """Rotate one OpenAI user row on the shared credential-mutation lock."""
    base_repo = AuthnzUserProviderSecretsRepo(pool)
    async with openai_credential_mutation_lock(
        user_id=user_id,
        provider=_OPENAI_PROVIDER,
    ) as locked_repo:
        mutation_repo = locked_repo or base_repo
        current_row = await _fetch_row_by_id(
            mutation_repo.db_pool,
            table="user_provider_secrets",
            row_id=row_id,
            is_postgres=is_postgres,
        )
        if current_row is None:
            return "conflict"
        current_blob = current_row.get("encrypted_blob")
        if not isinstance(current_blob, str) or not current_blob:
            return "skipped"
        updated_blob = _reencrypt_blob(current_blob)
        updated = await _apply_update_if_unchanged(
            mutation_repo.db_pool,
            table="user_provider_secrets",
            updated_blob=updated_blob,
            row_id=row_id,
            expected_blob=current_blob,
            is_postgres=is_postgres,
        )
        return "updated" if updated else "conflict"


async def _rotate_table(
    *,
    pool: DatabasePool,
    table: str,
    batch_size: int,
    dry_run: bool,
    is_postgres: bool,
) -> RotationStats:
    stats = RotationStats()
    last_id = 0

    while True:
        async with pool.acquire() as conn:
            rows = await _fetch_rows(
                conn,
                table=table,
                last_id=last_id,
                batch_size=batch_size,
                is_postgres=is_postgres,
            )
        if not rows:
            break

        max_id = last_id
        for row in rows:
            row_id, encrypted_blob, user_id, provider = _extract_row_fields(row, table=table)
            max_id = max(max_id, row_id)
            stats.processed += 1

            if not isinstance(encrypted_blob, str) or not encrypted_blob:
                stats.skipped += 1
                continue

            try:
                if dry_run:
                    _reencrypt_blob(encrypted_blob)
                    stats.updated += 1
                    continue

                if (
                    table == "user_provider_secrets"
                    and user_id is not None
                    and canonical_provider_name(provider or "") == _OPENAI_PROVIDER
                ):
                    outcome = await _rotate_openai_user_row(
                        pool=pool,
                        row_id=row_id,
                        user_id=user_id,
                        is_postgres=is_postgres,
                    )
                    if outcome == "updated":
                        stats.updated += 1
                    elif outcome == "skipped":
                        stats.skipped += 1
                    else:
                        stats.failed += 1
                        logger.warning("BYOK rotation CAS conflict table={} id={}", table, row_id)
                    continue

                updated_blob = _reencrypt_blob(encrypted_blob)
                updated = await _apply_update_if_unchanged(
                    pool,
                    table=table,
                    updated_blob=updated_blob,
                    row_id=row_id,
                    expected_blob=encrypted_blob,
                    is_postgres=is_postgres,
                )
                if updated:
                    stats.updated += 1
                else:
                    stats.failed += 1
                    logger.warning("BYOK rotation CAS conflict table={} id={}", table, row_id)
            except Exception as exc:
                stats.failed += 1
                logger.warning(
                    "BYOK rotation failed table={} id={} error_type={}",
                    table,
                    row_id,
                    type(exc).__name__,
                )

        last_id = max_id

    return stats


async def rotate_byok_secrets(
    *,
    dry_run: bool = False,
    batch_size: int = 500,
    pool: DatabasePool | None = None,
) -> RotationSummary:
    settings = get_settings()
    if not settings.BYOK_ENCRYPTION_KEY:
        raise ValueError("BYOK_ENCRYPTION_KEY is not configured")
    if not settings.BYOK_SECONDARY_ENCRYPTION_KEY:
        logger.warning(
            "BYOK_SECONDARY_ENCRYPTION_KEY is not set; rotation will only succeed for rows "
            "already encrypted with the primary key"
        )

    db_pool = pool or await get_db_pool()
    is_postgres = _is_postgres_pool(db_pool)

    tables = {}
    total = RotationStats()
    for table in ("user_provider_secrets", "org_provider_secrets"):
        stats = await _rotate_table(
            pool=db_pool,
            table=table,
            batch_size=batch_size,
            dry_run=dry_run,
            is_postgres=is_postgres,
        )
        tables[table] = stats
        total.add(stats)

    return RotationSummary(tables=tables, total=total, dry_run=dry_run)
