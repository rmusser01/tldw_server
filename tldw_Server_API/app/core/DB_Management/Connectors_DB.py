from __future__ import annotations


async def _sqlite_column_names(db, table_name: str) -> set[str]:
    cur = await db.execute(f"PRAGMA table_info({table_name})")
    rows = await cur.fetchall()
    names: set[str] = set()
    for row in rows:
        if hasattr(row, "keys"):
            names.add(str(row["name"]))
        else:
            names.add(str(row[1]))
    return names


async def _ensure_sqlite_column(db, table_name: str, column_name: str, column_sql: str) -> None:
    if column_name in await _sqlite_column_names(db, table_name):
        return
    await db.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_sql}")


async def _mark_sources_needing_rescan(db, *, is_postgres: bool) -> None:
    if is_postgres:
        await db.execute(
            """
            INSERT INTO external_source_sync_state (
                source_id,
                sync_mode,
                needs_full_rescan
            )
            SELECT DISTINCT source_id, 'manual', TRUE
            FROM external_items
            WHERE media_id IS NULL
            ON CONFLICT (source_id) DO UPDATE SET
                needs_full_rescan = TRUE
            """
        )
        return
    await db.execute(
        """
        INSERT OR IGNORE INTO external_source_sync_state (
            source_id,
            sync_mode,
            needs_full_rescan
        )
        SELECT DISTINCT source_id, 'manual', 1
        FROM external_items
        WHERE media_id IS NULL
        """
    )
    await db.execute(
        """
        UPDATE external_source_sync_state
        SET needs_full_rescan = 1
        WHERE source_id IN (
            SELECT DISTINCT source_id
            FROM external_items
            WHERE media_id IS NULL
        )
        """
    )


async def ensure_connectors_tables(db, *, is_postgres: bool) -> None:
    if is_postgres:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS external_accounts (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL,
                provider TEXT NOT NULL,
                display_name TEXT,
                email TEXT,
                access_token TEXT,
                refresh_token TEXT,
                token_expires_at TIMESTAMP NULL,
                scopes TEXT,
                provider_metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS external_sources (
                id SERIAL PRIMARY KEY,
                account_id INTEGER NOT NULL REFERENCES external_accounts(id) ON DELETE CASCADE,
                provider TEXT NOT NULL,
                remote_id TEXT NOT NULL,
                type TEXT NOT NULL,
                path TEXT,
                options JSONB,
                enabled BOOLEAN DEFAULT TRUE,
                last_synced_at TIMESTAMP NULL
            )
            """
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS external_items (
                id SERIAL PRIMARY KEY,
                source_id INTEGER NOT NULL REFERENCES external_sources(id) ON DELETE CASCADE,
                provider TEXT NOT NULL,
                external_id TEXT NOT NULL,
                name TEXT,
                mime TEXT,
                size BIGINT,
                modified_at TIMESTAMP NULL,
                version TEXT,
                hash TEXT,
                last_ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(source_id, provider, external_id)
            )
            """
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS external_reference_items (
                id SERIAL PRIMARY KEY,
                source_id INTEGER NOT NULL REFERENCES external_sources(id) ON DELETE CASCADE,
                provider TEXT NOT NULL,
                provider_item_key TEXT NOT NULL,
                provider_library_id TEXT,
                collection_key TEXT,
                collection_name TEXT,
                provider_version TEXT,
                provider_updated_at TIMESTAMP NULL,
                media_id INTEGER,
                dedupe_match_reason TEXT,
                raw_reference_metadata JSONB,
                first_imported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_imported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(source_id, provider, provider_item_key)
            )
            """
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS org_connector_policy (
                org_id INTEGER PRIMARY KEY,
                enabled_providers TEXT,
                allowed_export_formats TEXT,
                allowed_file_types TEXT,
                max_file_size_mb INTEGER,
                account_linking_role TEXT,
                allowed_account_domains TEXT,
                allowed_remote_paths TEXT,
                denied_remote_paths TEXT,
                allowed_notion_workspaces TEXT,
                denied_notion_workspaces TEXT,
                quotas_per_role JSONB,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS external_oauth_state (
                state TEXT NOT NULL,
                user_id INTEGER NOT NULL,
                provider TEXT NOT NULL,
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (state, user_id)
            )
            """
        )
        await db.execute("ALTER TABLE external_accounts ADD COLUMN IF NOT EXISTS provider_metadata JSONB")
        await db.execute("ALTER TABLE external_oauth_state ADD COLUMN IF NOT EXISTS metadata JSONB")
        await db.execute("ALTER TABLE external_items ADD COLUMN IF NOT EXISTS media_id INTEGER")
        await db.execute("ALTER TABLE external_items ADD COLUMN IF NOT EXISTS sync_status TEXT")
        await db.execute("ALTER TABLE external_items ADD COLUMN IF NOT EXISTS current_version_number INTEGER")
        await db.execute("ALTER TABLE external_items ADD COLUMN IF NOT EXISTS remote_parent_id TEXT")
        await db.execute("ALTER TABLE external_items ADD COLUMN IF NOT EXISTS remote_path TEXT")
        await db.execute("ALTER TABLE external_items ADD COLUMN IF NOT EXISTS last_seen_at TIMESTAMP NULL")
        await db.execute("ALTER TABLE external_items ADD COLUMN IF NOT EXISTS last_content_sync_at TIMESTAMP NULL")
        await db.execute("ALTER TABLE external_items ADD COLUMN IF NOT EXISTS last_metadata_sync_at TIMESTAMP NULL")
        await db.execute("ALTER TABLE external_items ADD COLUMN IF NOT EXISTS remote_deleted_at TIMESTAMP NULL")
        await db.execute("ALTER TABLE external_items ADD COLUMN IF NOT EXISTS access_revoked_at TIMESTAMP NULL")
        await db.execute("ALTER TABLE external_items ADD COLUMN IF NOT EXISTS provider_metadata JSONB")
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS external_source_sync_state (
                source_id INTEGER PRIMARY KEY REFERENCES external_sources(id) ON DELETE CASCADE,
                sync_mode TEXT NOT NULL DEFAULT 'manual',
                cursor TEXT,
                cursor_kind TEXT,
                last_bootstrap_at TIMESTAMP NULL,
                last_sync_started_at TIMESTAMP NULL,
                last_sync_succeeded_at TIMESTAMP NULL,
                last_sync_failed_at TIMESTAMP NULL,
                last_error TEXT,
                retry_backoff_count INTEGER DEFAULT 0,
                webhook_status TEXT,
                webhook_subscription_id TEXT,
                webhook_expires_at TIMESTAMP NULL,
                webhook_metadata JSONB,
                needs_full_rescan BOOLEAN DEFAULT FALSE,
                active_job_id TEXT,
                active_job_started_at TIMESTAMP NULL
            )
            """
        )
        await db.execute(
            """
            ALTER TABLE external_source_sync_state
            ADD COLUMN IF NOT EXISTS webhook_metadata JSONB
            """
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS external_item_events (
                id SERIAL PRIMARY KEY,
                external_item_id INTEGER NOT NULL REFERENCES external_items(id) ON DELETE CASCADE,
                event_type TEXT NOT NULL,
                job_id TEXT,
                payload_json JSONB,
                occurred_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS external_webhook_receipts (
                id SERIAL PRIMARY KEY,
                provider TEXT NOT NULL,
                receipt_key TEXT NOT NULL,
                source_id INTEGER NULL REFERENCES external_sources(id) ON DELETE SET NULL,
                payload_hash TEXT,
                received_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(provider, receipt_key)
            )
            """
        )
    else:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS external_accounts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                provider TEXT NOT NULL,
                display_name TEXT,
                email TEXT,
                access_token TEXT,
                refresh_token TEXT,
                token_expires_at TEXT,
                scopes TEXT,
                provider_metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """,
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS external_sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                account_id INTEGER NOT NULL,
                provider TEXT NOT NULL,
                remote_id TEXT NOT NULL,
                type TEXT NOT NULL,
                path TEXT,
                options TEXT,
                enabled INTEGER DEFAULT 1,
                last_synced_at TEXT
            )
            """,
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS external_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id INTEGER NOT NULL,
                provider TEXT NOT NULL,
                external_id TEXT NOT NULL,
                name TEXT,
                mime TEXT,
                size INTEGER,
                modified_at TEXT,
                version TEXT,
                hash TEXT,
                last_ingested_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(source_id, provider, external_id)
            )
            """,
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS external_reference_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id INTEGER NOT NULL,
                provider TEXT NOT NULL,
                provider_item_key TEXT NOT NULL,
                provider_library_id TEXT,
                collection_key TEXT,
                collection_name TEXT,
                provider_version TEXT,
                provider_updated_at TEXT,
                media_id INTEGER,
                dedupe_match_reason TEXT,
                raw_reference_metadata TEXT,
                first_imported_at TEXT DEFAULT CURRENT_TIMESTAMP,
                last_imported_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(source_id, provider, provider_item_key)
            )
            """,
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS org_connector_policy (
                org_id INTEGER PRIMARY KEY,
                enabled_providers TEXT,
                allowed_export_formats TEXT,
                allowed_file_types TEXT,
                max_file_size_mb INTEGER,
                account_linking_role TEXT,
                allowed_account_domains TEXT,
                allowed_remote_paths TEXT,
                denied_remote_paths TEXT,
                allowed_notion_workspaces TEXT,
                denied_notion_workspaces TEXT,
                quotas_per_role TEXT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """,
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS external_oauth_state (
                state TEXT NOT NULL,
                user_id INTEGER NOT NULL,
                provider TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (state, user_id)
            )
            """,
        )
        await _ensure_sqlite_column(db, "external_accounts", "provider_metadata", "provider_metadata TEXT")
        await _ensure_sqlite_column(db, "external_oauth_state", "metadata", "metadata TEXT")
        await _ensure_sqlite_column(db, "external_items", "media_id", "media_id INTEGER")
        await _ensure_sqlite_column(db, "external_items", "sync_status", "sync_status TEXT")
        await _ensure_sqlite_column(db, "external_items", "current_version_number", "current_version_number INTEGER")
        await _ensure_sqlite_column(db, "external_items", "remote_parent_id", "remote_parent_id TEXT")
        await _ensure_sqlite_column(db, "external_items", "remote_path", "remote_path TEXT")
        await _ensure_sqlite_column(db, "external_items", "last_seen_at", "last_seen_at TEXT")
        await _ensure_sqlite_column(db, "external_items", "last_content_sync_at", "last_content_sync_at TEXT")
        await _ensure_sqlite_column(db, "external_items", "last_metadata_sync_at", "last_metadata_sync_at TEXT")
        await _ensure_sqlite_column(db, "external_items", "remote_deleted_at", "remote_deleted_at TEXT")
        await _ensure_sqlite_column(db, "external_items", "access_revoked_at", "access_revoked_at TEXT")
        await _ensure_sqlite_column(db, "external_items", "provider_metadata", "provider_metadata TEXT")
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS external_source_sync_state (
                source_id INTEGER PRIMARY KEY,
                sync_mode TEXT NOT NULL DEFAULT 'manual',
                cursor TEXT,
                cursor_kind TEXT,
                last_bootstrap_at TEXT,
                last_sync_started_at TEXT,
                last_sync_succeeded_at TEXT,
                last_sync_failed_at TEXT,
                last_error TEXT,
                retry_backoff_count INTEGER DEFAULT 0,
                webhook_status TEXT,
                webhook_subscription_id TEXT,
                webhook_expires_at TEXT,
                webhook_metadata TEXT,
                needs_full_rescan INTEGER DEFAULT 0,
                active_job_id TEXT,
                active_job_started_at TEXT
            )
            """,
        )
        await _ensure_sqlite_column(
            db,
            "external_source_sync_state",
            "webhook_metadata",
            "webhook_metadata TEXT",
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS external_item_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                external_item_id INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                job_id TEXT,
                payload_json TEXT,
                occurred_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """,
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS external_webhook_receipts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                provider TEXT NOT NULL,
                receipt_key TEXT NOT NULL,
                source_id INTEGER,
                payload_hash TEXT,
                received_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(provider, receipt_key)
            )
            """,
        )

    await _mark_sources_needing_rescan(db, is_postgres=is_postgres)
