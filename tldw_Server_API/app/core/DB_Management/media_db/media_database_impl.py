"""Package-native owner for the canonical MediaDatabase class."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext, suppress
from datetime import timezone
from email.utils import parsedate_to_datetime
import json
import logging
import sqlite3
from typing import Any

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.sqlite_policy import begin_immediate_if_needed
from tldw_Server_API.app.core.DB_Management.media_db.runtime.sqlite_bootstrap import (
    apply_sqlite_connection_pragmas,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.bootstrap_lifecycle_ops import (
    _ensure_sqlite_backend,
    initialize_db,
    initialize_media_database,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.backup_ops import (
    _backup_non_sqlite_database,
    backup_database,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.backend_resolution import (
    _resolve_backend,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.chunk_ops import (
    add_media_chunks_in_batches,
    batch_insert_chunks,
    clear_unvectorized_chunks,
    create_chunking_template,
    delete_chunking_template,
    get_chunking_template,
    process_chunks,
    process_unvectorized_chunks,
    update_chunking_template,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.connection_lifecycle import (
    _dec_tx_depth,
    _get_persistent_conn,
    _get_tx_depth,
    _get_txn_conn,
    _inc_tx_depth,
    _set_persistent_conn,
    _set_tx_depth,
    _set_txn_conn,
    close_connection,
    get_connection,
    release_context_connection,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.execution_ops import (
    _execute_with_connection,
    _executemany_with_connection,
    _fetchall_with_connection,
    _fetchone_with_connection,
    execute_many,
    execute_query,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.fts_ops import (
    _delete_fts_keyword,
    _delete_fts_media,
    _update_fts_keyword,
    _update_fts_media,
    sync_refresh_fts_for_entity,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.chunk_fts_ops import (
    ensure_chunk_fts,
    maybe_rebuild_chunk_fts_if_empty,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.backend_prepare_ops import (
    _normalise_params,
    _prepare_backend_many_statement,
    _prepare_backend_statement,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.sync_utility_ops import (
    _generate_uuid,
    _get_current_utc_timestamp_str,
    _get_next_version,
    _log_sync_event,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.scope_resolution_ops import (
    _resolve_scope_ids,
)
from tldw_Server_API.app.core.DB_Management.media_db.schema.bootstrap import (
    ensure_media_schema,
)
from tldw_Server_API.app.core.DB_Management.media_db.schema.backends.sqlite_helpers import (
    bootstrap_sqlite_schema,
)
from tldw_Server_API.app.core.DB_Management.media_db.schema.backends.postgres_helpers import (
    bootstrap_postgres_schema,
)
from tldw_Server_API.app.core.DB_Management.media_db.schema.migrations import (
    get_postgres_migrations,
    run_postgres_migrations,
)
from tldw_Server_API.app.core.DB_Management.media_db.schema.features.postgres_rls import (
    _ensure_postgres_rls,
    _postgres_policy_exists,
)
from tldw_Server_API.app.core.DB_Management.media_db.schema.features.core_media import (
    apply_postgres_core_media_schema,
    apply_sqlite_core_media_schema,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.sync_log_ops import (
    delete_sync_log_entries,
    delete_sync_log_entries_before,
    get_sync_log_entries,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.structure_index_ops import (
    _write_structure_index_records,
    delete_document_structure_for_media,
    write_document_structure_index,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.query_utility_ops import (
    _append_case_insensitive_like,
    _convert_sqlite_placeholders_to_postgres,
    _keyword_order_expression,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.media_file_ops import (
    get_media_file,
    get_media_files,
    has_original_file,
    insert_media_file,
    soft_delete_media_file,
    soft_delete_media_files_for_media,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.visual_document_ops import (
    insert_visual_document,
    list_visual_documents_for_media,
    soft_delete_visual_documents_for_media,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.tts_history_ops import (
    _build_tts_history_filters,
    count_tts_history,
    create_tts_history_entry,
    get_tts_history_entry,
    list_tts_history,
    list_tts_history_user_ids,
    mark_tts_history_artifacts_deleted_for_file_id,
    mark_tts_history_artifacts_deleted_for_output,
    purge_tts_history_for_user,
    soft_delete_tts_history_entry,
    update_tts_history_favorite,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.audio_preset_ops import (
    count_audio_presets,
    create_audio_preset,
    get_audio_preset,
    list_audio_presets,
    soft_delete_audio_preset,
    update_audio_preset,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.document_keyword_ops import (
    create_document_version,
    get_all_document_versions,
    soft_delete_document_version,
    soft_delete_keyword,
    update_keywords_for_media,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.keyword_access_ops import (
    add_keyword,
    fetch_media_for_keywords,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.media_lifecycle_ops import (
    get_media_visibility,
    mark_as_trash,
    restore_from_trash,
    share_media,
    soft_delete_media,
    unshare_media,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.media_entrypoint_ops import (
    run_add_media_with_keywords,
    run_search_media_db,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.data_table_helper_ops import (
    _get_data_table_owner_client_id,
    _normalize_data_table_row_json,
    _resolve_data_table_write_client_id,
    _resolve_data_tables_owner,
    _soft_delete_data_table_children,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.data_table_metadata_ops import (
    count_data_tables,
    create_data_table,
    get_data_table,
    get_data_table_by_uuid,
    list_data_tables,
    soft_delete_data_table,
    update_data_table,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.data_table_generation_ops import (
    persist_data_table_generation,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.data_table_replace_ops import (
    replace_data_table_contents,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.safe_metadata_search_ops import (
    search_by_safe_metadata,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.synced_document_update_ops import (
    apply_synced_document_content_update,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.document_version_rollback_ops import (
    rollback_to_version,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.noncritical import (
    MEDIA_NONCRITICAL_EXCEPTIONS,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.email_state_ops import (
    _ensure_email_backfill_state_row,
    _fetch_email_backfill_state_row,
    _fetch_email_sync_state_row,
    _resolve_email_sync_source_row_id,
    _update_email_backfill_progress,
    get_email_legacy_backfill_state,
    get_email_sync_state,
    mark_email_sync_run_failed,
    mark_email_sync_run_started,
    mark_email_sync_run_succeeded,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.email_message_mutation_ops import (
    _normalize_email_label_values,
    _resolve_email_message_row_for_source_message,
    apply_email_label_delta,
    reconcile_email_message_state,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.email_query_ops import (
    _email_like_clause,
    _parse_email_operator_query,
    get_email_message_detail,
    search_email_messages,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.email_retention_ops import (
    _cleanup_email_orphans_for_tenant,
    enforce_email_retention_policy,
    hard_delete_email_tenant_data,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.email_backfill_ops import (
    run_email_legacy_backfill_batch,
    run_email_legacy_backfill_worker,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.email_graph_persistence_ops import (
    _resolve_email_tenant_id,
    upsert_email_message_graph,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.claims_analytics_export_ops import (
    cleanup_claims_analytics_exports,
    count_claims_analytics_exports,
    create_claims_analytics_export,
    get_claims_analytics_export,
    list_claims_analytics_exports,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.claims_notification_ops import (
    get_claim_notification,
    get_claim_notifications_by_ids,
    get_latest_claim_notification,
    insert_claim_notification,
    list_claim_notifications,
    mark_claim_notifications_delivered,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.claims_review_rule_ops import (
    create_claim_review_rule,
    delete_claim_review_rule,
    get_claim_review_rule,
    list_claim_review_rules,
    update_claim_review_rule,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.claims_review_read_ops import (
    list_claim_review_history,
    list_review_queue,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.claims_read_ops import (
    get_claim_with_media,
    get_claims_by_media,
    get_claims_by_uuid,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.claims_write_ops import (
    soft_delete_claims_for_media,
    update_claim,
    update_claim_review,
    upsert_claims,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.claims_list_ops import (
    list_claims,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.claims_search_ops import (
    search_claims,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.claims_review_metrics_ops import (
    count_claims_review_extractor_metrics_daily,
    get_claims_review_extractor_metrics_daily,
    list_claims_review_extractor_metrics_daily,
    list_claims_review_user_ids,
    upsert_claims_review_extractor_metrics_daily,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.claims_cluster_aggregate_ops import (
    get_claim_cluster_member_counts,
    get_claim_clusters_by_ids,
    update_claim_clusters_watchlist_counts,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.claims_cluster_ops import (
    add_claim_to_cluster,
    create_claim_cluster,
    create_claim_cluster_link,
    delete_claim_cluster_link,
    get_claim_cluster,
    get_claim_cluster_link,
    list_claim_cluster_links,
    list_claim_cluster_members,
    list_claim_clusters,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.claims_cluster_exact_rebuild_ops import (
    rebuild_claim_clusters_exact,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.claims_cluster_assignment_rebuild_ops import (
    rebuild_claim_clusters_from_assignments,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.claims_fts_ops import (
    rebuild_claims_fts,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.claims_monitoring_settings_ops import (
    get_claims_monitoring_settings,
    upsert_claims_monitoring_settings,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.claims_monitoring_config_ops import (
    create_claims_monitoring_config,
    delete_claims_monitoring_config,
    delete_claims_monitoring_configs_by_user,
    get_claims_monitoring_config,
    list_claims_monitoring_configs,
    list_claims_monitoring_user_ids,
    update_claims_monitoring_config,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.claims_monitoring_migration_ops import (
    migrate_legacy_claims_monitoring_alerts,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.claims_monitoring_alert_ops import (
    create_claims_monitoring_alert,
    delete_claims_monitoring_alert,
    get_claims_monitoring_alert,
    list_claims_monitoring_alerts,
    update_claims_monitoring_alert,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.claims_monitoring_health_ops import (
    get_claims_monitoring_health,
    upsert_claims_monitoring_health,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.claims_monitoring_event_ops import (
    get_latest_claims_monitoring_event_delivery,
    insert_claims_monitoring_event,
    list_claims_monitoring_events,
    list_undelivered_claims_monitoring_events,
    mark_claims_monitoring_events_delivered,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.data_table_child_ops import (
    get_data_table_counts,
    insert_data_table_columns,
    insert_data_table_rows,
    insert_data_table_sources,
    list_data_table_columns,
    list_data_table_rows,
    list_data_table_sources,
    soft_delete_data_table_columns,
    soft_delete_data_table_rows,
    soft_delete_data_table_sources,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.query_ops import (
    fetch_all_keywords,
    get_distinct_media_types,
    get_media_by_id,
    get_media_by_hash,
    get_media_status_by_id,
    get_media_by_title,
    get_media_by_url,
    get_media_by_uuid,
    get_paginated_media_list,
    get_paginated_files,
    get_paginated_trash_list,
    has_unvectorized_chunks,
)
from tldw_Server_API.app.core.DB_Management.media_db.schema.migration_bodies.postgres_collections import (
    run_postgres_migrate_to_v12,
    run_postgres_migrate_to_v13,
)
from tldw_Server_API.app.core.DB_Management.media_db.schema.migration_bodies.postgres_early_schema import (
    run_postgres_migrate_to_v5,
    run_postgres_migrate_to_v6,
    run_postgres_migrate_to_v7,
    run_postgres_migrate_to_v8,
)
from tldw_Server_API.app.core.DB_Management.media_db.schema.migration_bodies.postgres_visibility_owner import (
    run_postgres_migrate_to_v9,
)
from tldw_Server_API.app.core.DB_Management.media_db.schema.migration_bodies.postgres_claims import (
    run_postgres_migrate_to_v10,
    run_postgres_migrate_to_v17,
)
from tldw_Server_API.app.core.DB_Management.media_db.schema.migration_bodies.postgres_mediafiles import (
    run_postgres_migrate_to_v11,
)
from tldw_Server_API.app.core.DB_Management.media_db.schema.migration_bodies.postgres_data_tables import (
    run_postgres_migrate_to_v14,
    run_postgres_migrate_to_v15,
)
from tldw_Server_API.app.core.DB_Management.media_db.schema.migration_bodies.postgres_source_hash import (
    run_postgres_migrate_to_v16,
)
from tldw_Server_API.app.core.DB_Management.media_db.schema.migration_bodies.postgres_sequence_sync import (
    run_postgres_migrate_to_v18,
)
from tldw_Server_API.app.core.DB_Management.media_db.schema.migration_bodies.postgres_fts_rls import (
    run_postgres_migrate_to_v19,
)
from tldw_Server_API.app.core.DB_Management.media_db.schema.migration_bodies.postgres_structure_visual_indexes import (
    run_postgres_migrate_to_v21,
)
from tldw_Server_API.app.core.DB_Management.media_db.schema.migration_bodies.postgres_email_schema import (
    run_postgres_migrate_to_v22,
)
from tldw_Server_API.app.core.DB_Management.media_db.schema.migration_bodies.postgres_transcript_run_history import (
    run_postgres_migrate_to_v23,
)
from tldw_Server_API.app.core.DB_Management.media_db.schema.migration_bodies.postgres_tts_history import (
    run_postgres_migrate_to_v20,
)
from tldw_Server_API.app.core.DB_Management.media_db.schema.postgres_schema_version import (
    update_schema_version_postgres,
)
from tldw_Server_API.app.core.DB_Management.media_db.schema.sqlite_schema_version import (
    get_db_version,
)
from tldw_Server_API.app.core.DB_Management.media_db.schema.fts_structures import (
    ensure_fts_structures,
    ensure_postgres_fts,
    ensure_sqlite_fts,
)
from tldw_Server_API.app.core.DB_Management.media_db.schema.email_schema_structures import (
    ensure_postgres_email_schema,
    ensure_sqlite_email_schema,
)
from tldw_Server_API.app.core.DB_Management.media_db.schema.sqlite_post_core_structures import (
    ensure_sqlite_data_tables,
    ensure_sqlite_source_hash_column,
    ensure_sqlite_visibility_columns,
)
from tldw_Server_API.app.core.DB_Management.media_db.schema.sqlite_claims_extensions import (
    ensure_sqlite_claims_extensions,
)
from tldw_Server_API.app.core.DB_Management.media_db.schema.postgres_data_table_structures import (
    ensure_postgres_columns,
    ensure_postgres_data_tables,
    ensure_postgres_data_tables_columns,
)
from tldw_Server_API.app.core.DB_Management.media_db.schema.postgres_tts_source_hash_structures import (
    ensure_postgres_source_hash_column,
    ensure_postgres_tts_history,
)
from tldw_Server_API.app.core.DB_Management.media_db.schema.postgres_audio_preset_structures import (
    ensure_postgres_audio_presets,
)
from tldw_Server_API.app.core.DB_Management.media_db.schema.postgres_claims_collection_structures import (
    ensure_postgres_claims_extensions,
    ensure_postgres_claims_tables,
    ensure_postgres_collections_tables,
)
from tldw_Server_API.app.core.DB_Management.media_db.schema.postgres_sequence_maintenance import (
    sync_postgres_sequences,
)
from tldw_Server_API.app.core.DB_Management.media_db.schema.postgres_sqlite_conversion import (
    _convert_sqlite_sql_to_postgres_statements,
    _transform_sqlite_statement_to_postgres,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.state_ops import (
    mark_embeddings_error,
    update_media_reprocess_state,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.template_structure_ops import (
    list_chunking_templates,
    lookup_section_by_heading,
    lookup_section_for_offset,
    seed_builtin_templates,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.unvectorized_chunk_reads import (
    get_unvectorized_anchor_index_for_offset,
    get_unvectorized_chunk_by_index,
    get_unvectorized_chunk_count,
    get_unvectorized_chunk_index_by_uuid,
    get_unvectorized_chunks_in_range,
)

class MediaDatabase:
    """Canonical package-native Media DB runtime class."""

    _CURRENT_SCHEMA_VERSION = 23  # Transcript run-history schema/bootstrap scaffolding

    # <<< Schema Definition (Version 1) >>>

    _TABLES_SQL_V1 = """
    PRAGMA foreign_keys = ON;

    -- Schema Version Table --
    CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER PRIMARY KEY NOT NULL
    );
    -- Initialize version if table is newly created
    INSERT OR IGNORE INTO schema_version (version) VALUES (0);

    -- Media Table --
    CREATE TABLE IF NOT EXISTS Media (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        url TEXT UNIQUE,
        title TEXT NOT NULL,
        type TEXT NOT NULL,
        content TEXT,
        author TEXT,
        ingestion_date DATETIME,
        transcription_model TEXT,
        is_trash BOOLEAN DEFAULT 0 NOT NULL,
        trash_date DATETIME,
        vector_embedding BLOB,
        chunking_status TEXT DEFAULT 'pending' NOT NULL,
        vector_processing INTEGER DEFAULT 0 NOT NULL,
        content_hash TEXT NOT NULL,
        source_hash TEXT,
        uuid TEXT UNIQUE NOT NULL,
        last_modified DATETIME NOT NULL,
        version INTEGER NOT NULL DEFAULT 1,
        org_id INTEGER,
        team_id INTEGER,
        visibility TEXT DEFAULT 'personal' CHECK (visibility IN ('personal', 'team', 'org')),
        owner_user_id INTEGER,
        latest_transcription_run_id INTEGER,
        next_transcription_run_id INTEGER NOT NULL DEFAULT 1,
        client_id TEXT NOT NULL,
        deleted BOOLEAN NOT NULL DEFAULT 0,
        prev_version INTEGER,
        merge_parent_uuid TEXT
    );

    -- Keywords Table --
    CREATE TABLE IF NOT EXISTS Keywords (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        keyword TEXT NOT NULL UNIQUE COLLATE NOCASE,
        uuid TEXT UNIQUE NOT NULL,
        last_modified DATETIME NOT NULL,
        version INTEGER NOT NULL DEFAULT 1,
        client_id TEXT NOT NULL,
        deleted BOOLEAN NOT NULL DEFAULT 0,
        prev_version INTEGER,
        merge_parent_uuid TEXT
    );

    -- MediaKeywords Table (Junction Table) --
    CREATE TABLE IF NOT EXISTS MediaKeywords (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        media_id INTEGER NOT NULL,
        keyword_id INTEGER NOT NULL,
        UNIQUE (media_id, keyword_id),
        FOREIGN KEY (media_id) REFERENCES Media(id) ON DELETE CASCADE,
        FOREIGN KEY (keyword_id) REFERENCES Keywords(id) ON DELETE CASCADE
    );

    -- Transcripts Table --
    CREATE TABLE IF NOT EXISTS Transcripts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        media_id INTEGER NOT NULL,
        whisper_model TEXT,
        transcription TEXT,
        created_at DATETIME,
        transcription_run_id INTEGER,
        supersedes_run_id INTEGER,
        idempotency_key TEXT,
        uuid TEXT UNIQUE NOT NULL,
        last_modified DATETIME NOT NULL,
        version INTEGER NOT NULL DEFAULT 1,
        client_id TEXT NOT NULL,
        deleted BOOLEAN NOT NULL DEFAULT 0,
        prev_version INTEGER,
        merge_parent_uuid TEXT,
        FOREIGN KEY (media_id) REFERENCES Media(id) ON DELETE CASCADE
    );

    -- MediaChunks Table --
    CREATE TABLE IF NOT EXISTS MediaChunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        media_id INTEGER NOT NULL,
        chunk_text TEXT NOT NULL,
        start_index INTEGER,
        end_index INTEGER,
        chunk_id TEXT UNIQUE,
        uuid TEXT UNIQUE NOT NULL,
        last_modified DATETIME NOT NULL,
        version INTEGER NOT NULL DEFAULT 1,
        client_id TEXT NOT NULL,
        deleted BOOLEAN NOT NULL DEFAULT 0,
        prev_version INTEGER,
        merge_parent_uuid TEXT,
        FOREIGN KEY (media_id) REFERENCES Media(id) ON DELETE CASCADE
    );

    -- UnvectorizedMediaChunks Table --
    CREATE TABLE IF NOT EXISTS UnvectorizedMediaChunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        media_id INTEGER NOT NULL,
        chunk_text TEXT NOT NULL,
        chunk_index INTEGER NOT NULL,
        start_char INTEGER,
        end_char INTEGER,
        chunk_type TEXT,
        creation_date DATETIME,
        last_modified_orig DATETIME,
        is_processed BOOLEAN DEFAULT FALSE NOT NULL,
        metadata TEXT,
        uuid TEXT UNIQUE NOT NULL,
        last_modified DATETIME NOT NULL,
        version INTEGER NOT NULL DEFAULT 1,
        client_id TEXT NOT NULL,
        deleted BOOLEAN NOT NULL DEFAULT 0,
        prev_version INTEGER,
        merge_parent_uuid TEXT,
        UNIQUE (media_id, chunk_index, chunk_type),
        FOREIGN KEY (media_id) REFERENCES Media(id) ON DELETE CASCADE
    );

    -- VisualDocuments Table --
    -- Stores per-media image-derived artifacts (figures, frames, screenshots) with
    -- captions/OCR and soft-delete/versioning semantics.
    CREATE TABLE IF NOT EXISTS VisualDocuments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        media_id INTEGER NOT NULL,
        location TEXT,
        page_number INTEGER,
        frame_index INTEGER,
        timestamp_seconds REAL,
        caption TEXT,
        ocr_text TEXT,
        tags TEXT,
        thumbnail_path TEXT,
        extra_metadata TEXT,
        uuid TEXT UNIQUE NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        version INTEGER NOT NULL DEFAULT 1,
        client_id TEXT NOT NULL,
        deleted BOOLEAN NOT NULL DEFAULT 0,
        prev_version INTEGER,
        merge_parent_uuid TEXT,
        FOREIGN KEY (media_id) REFERENCES Media(id) ON DELETE CASCADE
    );

    -- DocumentVersions Table --
    CREATE TABLE IF NOT EXISTS DocumentVersions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        media_id INTEGER NOT NULL,
        version_number INTEGER NOT NULL,
        prompt TEXT,
        analysis_content TEXT,
        safe_metadata TEXT,
        content TEXT NOT NULL,
        created_at DATETIME,
        uuid TEXT UNIQUE NOT NULL,
        last_modified DATETIME NOT NULL,
        version INTEGER NOT NULL DEFAULT 1,
        client_id TEXT NOT NULL,
        deleted BOOLEAN NOT NULL DEFAULT 0,
        prev_version INTEGER,
        merge_parent_uuid TEXT,
        FOREIGN KEY (media_id) REFERENCES Media(id) ON DELETE CASCADE,
        UNIQUE (media_id, version_number)
    );

    -- DocumentVersionIdentifiers Table --
    CREATE TABLE IF NOT EXISTS DocumentVersionIdentifiers (
        dv_id INTEGER PRIMARY KEY,
        doi TEXT,
        pmid TEXT,
        pmcid TEXT,
        arxiv_id TEXT,
        s2_paper_id TEXT,
        FOREIGN KEY (dv_id) REFERENCES DocumentVersions(id) ON DELETE CASCADE
    );

    -- DocumentStructureIndex Table --
    -- Stores structural boundaries for documents (sections, paragraphs, etc.)
    CREATE TABLE IF NOT EXISTS DocumentStructureIndex (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        media_id INTEGER NOT NULL,
        parent_id INTEGER,
        kind TEXT NOT NULL,           -- e.g., 'section', 'paragraph', 'list', 'table', 'header'
        level INTEGER,                 -- heading depth if applicable
        title TEXT,                    -- section title if applicable
        start_char INTEGER NOT NULL,
        end_char INTEGER NOT NULL,
        order_index INTEGER,           -- ordering within media
        path TEXT,                     -- optional JSON/text path of ancestry titles
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        version INTEGER NOT NULL DEFAULT 1,
        client_id TEXT NOT NULL,
        deleted BOOLEAN NOT NULL DEFAULT 0,
        FOREIGN KEY (media_id) REFERENCES Media(id) ON DELETE CASCADE
    );

    -- Sync Log Table --
    CREATE TABLE IF NOT EXISTS sync_log (
        change_id INTEGER PRIMARY KEY AUTOINCREMENT,
        entity TEXT NOT NULL,
        entity_uuid TEXT NOT NULL,
        operation TEXT NOT NULL CHECK(operation IN ('create','update','delete', 'link', 'unlink')),
        timestamp DATETIME NOT NULL,
        client_id TEXT NOT NULL,
        version INTEGER NOT NULL,
        org_id INTEGER,
        team_id INTEGER,
        payload TEXT
    );

    -- Chunking Templates Table --
    CREATE TABLE IF NOT EXISTS ChunkingTemplates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        uuid TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        description TEXT,
        template_json TEXT NOT NULL,
        is_builtin BOOLEAN DEFAULT 0 NOT NULL,
        tags TEXT,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        version INTEGER NOT NULL DEFAULT 1,
        org_id INTEGER,
        team_id INTEGER,
        client_id TEXT NOT NULL,
        user_id TEXT,
        deleted BOOLEAN NOT NULL DEFAULT 0,
        prev_version INTEGER,
        merge_parent_uuid TEXT
    );
    """

    _INDICES_SQL_V1 = """
    -- Indices (Create after tables exist) --
    CREATE INDEX IF NOT EXISTS idx_media_title ON Media(title);
    CREATE INDEX IF NOT EXISTS idx_media_type ON Media(type);
    CREATE INDEX IF NOT EXISTS idx_media_author ON Media(author);
    CREATE INDEX IF NOT EXISTS idx_media_ingestion_date ON Media(ingestion_date);
    CREATE INDEX IF NOT EXISTS idx_media_chunking_status ON Media(chunking_status);
    CREATE INDEX IF NOT EXISTS idx_media_vector_processing ON Media(vector_processing);
    CREATE INDEX IF NOT EXISTS idx_media_is_trash ON Media(is_trash);
    CREATE INDEX IF NOT EXISTS idx_media_content_hash ON Media(content_hash);
    CREATE INDEX IF NOT EXISTS idx_media_source_hash ON Media(source_hash);
    CREATE UNIQUE INDEX IF NOT EXISTS idx_media_uuid ON Media(uuid);
    CREATE INDEX IF NOT EXISTS idx_media_last_modified ON Media(last_modified);
    CREATE INDEX IF NOT EXISTS idx_media_deleted ON Media(deleted);
    CREATE INDEX IF NOT EXISTS idx_media_prev_version ON Media(prev_version);
    CREATE INDEX IF NOT EXISTS idx_media_merge_parent_uuid ON Media(merge_parent_uuid);
    CREATE INDEX IF NOT EXISTS idx_media_org_id ON Media(org_id);
    CREATE INDEX IF NOT EXISTS idx_media_team_id ON Media(team_id);
    CREATE INDEX IF NOT EXISTS idx_media_visibility ON Media(visibility);
    CREATE INDEX IF NOT EXISTS idx_media_owner_user_id ON Media(owner_user_id);
    CREATE INDEX IF NOT EXISTS idx_media_latest_transcription_run_id ON Media(latest_transcription_run_id);
    CREATE INDEX IF NOT EXISTS idx_media_next_transcription_run_id ON Media(next_transcription_run_id);

    CREATE UNIQUE INDEX IF NOT EXISTS idx_keywords_uuid ON Keywords(uuid);
    CREATE INDEX IF NOT EXISTS idx_keywords_last_modified ON Keywords(last_modified);
    CREATE INDEX IF NOT EXISTS idx_keywords_deleted ON Keywords(deleted);
    CREATE INDEX IF NOT EXISTS idx_keywords_prev_version ON Keywords(prev_version);
    CREATE INDEX IF NOT EXISTS idx_keywords_merge_parent_uuid ON Keywords(merge_parent_uuid);

    CREATE INDEX IF NOT EXISTS idx_mediakeywords_media_id ON MediaKeywords(media_id);
    CREATE INDEX IF NOT EXISTS idx_mediakeywords_keyword_id ON MediaKeywords(keyword_id);

    CREATE INDEX IF NOT EXISTS idx_transcripts_media_id ON Transcripts(media_id);
    CREATE UNIQUE INDEX IF NOT EXISTS idx_transcripts_uuid ON Transcripts(uuid);
    CREATE INDEX IF NOT EXISTS idx_transcripts_last_modified ON Transcripts(last_modified);
    CREATE INDEX IF NOT EXISTS idx_transcripts_deleted ON Transcripts(deleted);
    CREATE INDEX IF NOT EXISTS idx_transcripts_prev_version ON Transcripts(prev_version);
    CREATE INDEX IF NOT EXISTS idx_transcripts_merge_parent_uuid ON Transcripts(merge_parent_uuid);
    CREATE UNIQUE INDEX IF NOT EXISTS idx_transcripts_media_run_id
        ON Transcripts(media_id, transcription_run_id DESC)
        WHERE transcription_run_id IS NOT NULL;
    CREATE INDEX IF NOT EXISTS idx_transcripts_supersedes_run_id ON Transcripts(supersedes_run_id);
    CREATE UNIQUE INDEX IF NOT EXISTS idx_transcripts_media_idempotency_key
        ON Transcripts(media_id, idempotency_key)
        WHERE idempotency_key IS NOT NULL;

    CREATE INDEX IF NOT EXISTS idx_mediachunks_media_id ON MediaChunks(media_id);
    CREATE UNIQUE INDEX IF NOT EXISTS idx_mediachunks_uuid ON MediaChunks(uuid);
    CREATE INDEX IF NOT EXISTS idx_mediachunks_last_modified ON MediaChunks(last_modified);
    CREATE INDEX IF NOT EXISTS idx_mediachunks_deleted ON MediaChunks(deleted);
    CREATE INDEX IF NOT EXISTS idx_mediachunks_prev_version ON MediaChunks(prev_version);
    CREATE INDEX IF NOT EXISTS idx_mediachunks_merge_parent_uuid ON MediaChunks(merge_parent_uuid);

    CREATE INDEX IF NOT EXISTS idx_unvectorized_media_chunks_media_id ON UnvectorizedMediaChunks(media_id);
    CREATE INDEX IF NOT EXISTS idx_unvectorized_media_chunks_is_processed ON UnvectorizedMediaChunks(is_processed);
    CREATE INDEX IF NOT EXISTS idx_unvectorized_media_chunks_chunk_type ON UnvectorizedMediaChunks(chunk_type);
    CREATE UNIQUE INDEX IF NOT EXISTS idx_unvectorizedmediachunks_uuid ON UnvectorizedMediaChunks(uuid);
    CREATE INDEX IF NOT EXISTS idx_unvectorizedmediachunks_last_modified ON UnvectorizedMediaChunks(last_modified);
    CREATE INDEX IF NOT EXISTS idx_unvectorizedmediachunks_deleted ON UnvectorizedMediaChunks(deleted);
    CREATE INDEX IF NOT EXISTS idx_unvectorizedmediachunks_prev_version ON UnvectorizedMediaChunks(prev_version);
    CREATE INDEX IF NOT EXISTS idx_unvectorizedmediachunks_merge_parent_uuid ON UnvectorizedMediaChunks(merge_parent_uuid);

    -- VisualDocuments indices --
    CREATE INDEX IF NOT EXISTS idx_visualdocs_media_id ON VisualDocuments(media_id);
    CREATE UNIQUE INDEX IF NOT EXISTS idx_visualdocs_uuid ON VisualDocuments(uuid);
    CREATE INDEX IF NOT EXISTS idx_visualdocs_last_modified ON VisualDocuments(last_modified);
    CREATE INDEX IF NOT EXISTS idx_visualdocs_deleted ON VisualDocuments(deleted);
    CREATE INDEX IF NOT EXISTS idx_visualdocs_prev_version ON VisualDocuments(prev_version);
    CREATE INDEX IF NOT EXISTS idx_visualdocs_merge_parent_uuid ON VisualDocuments(merge_parent_uuid);
    CREATE INDEX IF NOT EXISTS idx_visualdocs_page_frame ON VisualDocuments(media_id, page_number, frame_index);
    CREATE INDEX IF NOT EXISTS idx_visualdocs_caption ON VisualDocuments(caption);
    CREATE INDEX IF NOT EXISTS idx_visualdocs_tags ON VisualDocuments(tags);

    CREATE INDEX IF NOT EXISTS idx_document_versions_media_id ON DocumentVersions(media_id);
    CREATE INDEX IF NOT EXISTS idx_document_versions_version_number ON DocumentVersions(version_number);
    CREATE UNIQUE INDEX IF NOT EXISTS idx_documentversions_uuid ON DocumentVersions(uuid);
    CREATE INDEX IF NOT EXISTS idx_documentversions_last_modified ON DocumentVersions(last_modified);
    CREATE INDEX IF NOT EXISTS idx_documentversions_deleted ON DocumentVersions(deleted);
    CREATE INDEX IF NOT EXISTS idx_documentversions_prev_version ON DocumentVersions(prev_version);
    CREATE INDEX IF NOT EXISTS idx_documentversions_merge_parent_uuid ON DocumentVersions(merge_parent_uuid);

    CREATE INDEX IF NOT EXISTS idx_dvi_doi ON DocumentVersionIdentifiers(doi);
    CREATE INDEX IF NOT EXISTS idx_dvi_pmid ON DocumentVersionIdentifiers(pmid);
    CREATE INDEX IF NOT EXISTS idx_dvi_pmcid ON DocumentVersionIdentifiers(pmcid);
    CREATE INDEX IF NOT EXISTS idx_dvi_arxiv ON DocumentVersionIdentifiers(arxiv_id);
    CREATE INDEX IF NOT EXISTS idx_dvi_s2 ON DocumentVersionIdentifiers(s2_paper_id);

    -- DocumentStructureIndex Indices --
    CREATE INDEX IF NOT EXISTS idx_dsi_media_kind ON DocumentStructureIndex(media_id, kind);
    CREATE INDEX IF NOT EXISTS idx_dsi_media_start ON DocumentStructureIndex(media_id, start_char);
    CREATE INDEX IF NOT EXISTS idx_dsi_media_parent ON DocumentStructureIndex(parent_id);
    CREATE INDEX IF NOT EXISTS idx_dsi_media_path ON DocumentStructureIndex(media_id, path);

    CREATE INDEX IF NOT EXISTS idx_sync_log_ts ON sync_log(timestamp);
    CREATE INDEX IF NOT EXISTS idx_sync_log_entity_uuid ON sync_log(entity_uuid);
    CREATE INDEX IF NOT EXISTS idx_sync_log_client_id ON sync_log(client_id);
    CREATE INDEX IF NOT EXISTS idx_sync_log_org_id ON sync_log(org_id);
    CREATE INDEX IF NOT EXISTS idx_sync_log_team_id ON sync_log(team_id);

    -- Chunking Templates Indices --
    CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_template_name
        ON ChunkingTemplates(name) WHERE deleted = 0;
    CREATE INDEX IF NOT EXISTS idx_template_is_builtin ON ChunkingTemplates(is_builtin);
    CREATE INDEX IF NOT EXISTS idx_template_deleted ON ChunkingTemplates(deleted);
    CREATE INDEX IF NOT EXISTS idx_template_tags ON ChunkingTemplates(tags);
    """

    _TRIGGERS_SQL_V1 = """
    -- Validation Triggers (Create after tables and indices) --
    DROP TRIGGER IF EXISTS media_validate_sync_update;
    CREATE TRIGGER media_validate_sync_update BEFORE UPDATE ON Media
    BEGIN
        SELECT RAISE(ABORT, 'Sync Error (Media): Version must increment by exactly 1.')
        WHERE NEW.version IS NOT OLD.version + 1;
        SELECT RAISE(ABORT, 'Sync Error (Media): Client ID cannot be NULL or empty.')
        WHERE NEW.client_id IS NULL OR NEW.client_id = '';
        -- Add more checks if needed (e.g., UUID modification)
        SELECT RAISE(ABORT, 'Sync Error (Media): UUID cannot be changed.')
        WHERE NEW.uuid IS NOT OLD.uuid;
    END;

    DROP TRIGGER IF EXISTS keywords_validate_sync_update;
    CREATE TRIGGER keywords_validate_sync_update BEFORE UPDATE ON Keywords
    BEGIN
        SELECT RAISE(ABORT, 'Sync Error (Keywords): Version must increment by exactly 1.')
        WHERE NEW.version IS NOT OLD.version + 1;
        SELECT RAISE(ABORT, 'Sync Error (Keywords): Client ID cannot be NULL or empty.')
        WHERE NEW.client_id IS NULL OR NEW.client_id = '';
        SELECT RAISE(ABORT, 'Sync Error (Keywords): UUID cannot be changed.')
        WHERE NEW.uuid IS NOT OLD.uuid;
    END;

    DROP TRIGGER IF EXISTS transcripts_validate_sync_update;
    CREATE TRIGGER transcripts_validate_sync_update BEFORE UPDATE ON Transcripts
    BEGIN
        SELECT RAISE(ABORT, 'Sync Error (Transcripts): Version must increment by exactly 1.')
        WHERE NEW.version IS NOT OLD.version + 1;
        SELECT RAISE(ABORT, 'Sync Error (Transcripts): Client ID cannot be NULL or empty.')
        WHERE NEW.client_id IS NULL OR NEW.client_id = '';
        SELECT RAISE(ABORT, 'Sync Error (Transcripts): UUID cannot be changed.')
        WHERE NEW.uuid IS NOT OLD.uuid;
    END;

    DROP TRIGGER IF EXISTS mediachunks_validate_sync_update;
    CREATE TRIGGER mediachunks_validate_sync_update BEFORE UPDATE ON MediaChunks
    BEGIN
        SELECT RAISE(ABORT, 'Sync Error (MediaChunks): Version must increment by exactly 1.')
        WHERE NEW.version IS NOT OLD.version + 1;
        SELECT RAISE(ABORT, 'Sync Error (MediaChunks): Client ID cannot be NULL or empty.')
        WHERE NEW.client_id IS NULL OR NEW.client_id = '';
        SELECT RAISE(ABORT, 'Sync Error (MediaChunks): UUID cannot be changed.')
        WHERE NEW.uuid IS NOT OLD.uuid;
    END;

    DROP TRIGGER IF EXISTS unvectorizedmediachunks_validate_sync_update;
    CREATE TRIGGER unvectorizedmediachunks_validate_sync_update BEFORE UPDATE ON UnvectorizedMediaChunks
    BEGIN
        SELECT RAISE(ABORT, 'Sync Error (UnvectorizedMediaChunks): Version must increment by exactly 1.')
        WHERE NEW.version IS NOT OLD.version + 1;
        SELECT RAISE(ABORT, 'Sync Error (UnvectorizedMediaChunks): Client ID cannot be NULL or empty.')
        WHERE NEW.client_id IS NULL OR NEW.client_id = '';
        SELECT RAISE(ABORT, 'Sync Error (UnvectorizedMediaChunks): UUID cannot be changed.')
        WHERE NEW.uuid IS NOT OLD.uuid;
    END;

    DROP TRIGGER IF EXISTS documentversions_validate_sync_update;
    CREATE TRIGGER documentversions_validate_sync_update BEFORE UPDATE ON DocumentVersions
    BEGIN
        SELECT RAISE(ABORT, 'Sync Error (DocumentVersions): Version must increment by exactly 1.')
        WHERE NEW.version IS NOT OLD.version + 1;
        SELECT RAISE(ABORT, 'Sync Error (DocumentVersions): Client ID cannot be NULL or empty.')
        WHERE NEW.client_id IS NULL OR NEW.client_id = '';
        SELECT RAISE(ABORT, 'Sync Error (DocumentVersions): UUID cannot be changed.')
        WHERE NEW.uuid IS NOT OLD.uuid;
    END;
    """

    _FTS_TABLES_SQL = """
    -- FTS Tables (Executed Separately) --
    CREATE VIRTUAL TABLE IF NOT EXISTS media_fts USING fts5(
        title,
        content,
        content='Media',    -- Keep reference to source table
        content_rowid='id' -- Link to Media.id
    );

    CREATE VIRTUAL TABLE IF NOT EXISTS keyword_fts USING fts5(
        keyword,
        content='Keywords',    -- Keep reference to source table
        content_rowid='id'  -- Link to Keywords.id
    );

    -- Optional FTS for Claims (content-backed; Stage 1 has no triggers)
    CREATE VIRTUAL TABLE IF NOT EXISTS claims_fts USING fts5(
        claim_text,
        content='Claims',     -- Keep reference to source table
        content_rowid='id'    -- Link to Claims.id
    );
    """

    _CLAIMS_FTS_TRIGGERS_SQL = """
    -- Keep claims_fts in sync with Claims via triggers
    DROP TRIGGER IF EXISTS claims_ai;
    CREATE TRIGGER IF NOT EXISTS claims_ai AFTER INSERT ON Claims BEGIN
        -- Only index non-deleted claims
        INSERT INTO claims_fts(rowid, claim_text)
        SELECT NEW.id, NEW.claim_text WHERE NEW.deleted = 0;
    END;

    DROP TRIGGER IF EXISTS claims_au;
    CREATE TRIGGER IF NOT EXISTS claims_au AFTER UPDATE ON Claims BEGIN
        -- Remove previous terms then re-index when not deleted
        INSERT INTO claims_fts(claims_fts, rowid, claim_text)
        SELECT 'delete', OLD.id, OLD.claim_text WHERE OLD.deleted = 0;
        INSERT INTO claims_fts(rowid, claim_text)
        SELECT NEW.id, NEW.claim_text WHERE NEW.deleted = 0;
    END;

    DROP TRIGGER IF EXISTS claims_ad;
    CREATE TRIGGER IF NOT EXISTS claims_ad AFTER DELETE ON Claims BEGIN
        INSERT INTO claims_fts(claims_fts, rowid, claim_text)
        SELECT 'delete', OLD.id, OLD.claim_text WHERE OLD.deleted = 0;
    END;
    """

    _CLAIMS_TABLE_SQL = """
    -- Claims table for ingestion-time factual statements tied to media chunks
    CREATE TABLE IF NOT EXISTS Claims (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        media_id INTEGER NOT NULL,
        chunk_index INTEGER NOT NULL,
        span_start INTEGER,
        span_end INTEGER,
        claim_text TEXT NOT NULL,
        confidence REAL,
        extractor TEXT NOT NULL,
        extractor_version TEXT NOT NULL,
        chunk_hash TEXT NOT NULL,
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        uuid TEXT UNIQUE NOT NULL,
        last_modified DATETIME NOT NULL,
        version INTEGER NOT NULL DEFAULT 1,
        client_id TEXT NOT NULL,
        deleted BOOLEAN NOT NULL DEFAULT 0,
        prev_version INTEGER,
        merge_parent_uuid TEXT,
        review_status TEXT NOT NULL DEFAULT 'pending',
        reviewer_id INTEGER,
        review_group TEXT,
        reviewed_at DATETIME,
        review_notes TEXT,
        review_version INTEGER NOT NULL DEFAULT 1,
        review_reason_code TEXT,
        claim_cluster_id INTEGER,
        FOREIGN KEY (media_id) REFERENCES Media(id) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_claims_media_id ON Claims(media_id);
    CREATE INDEX IF NOT EXISTS idx_claims_media_chunk ON Claims(media_id, chunk_index);
    CREATE UNIQUE INDEX IF NOT EXISTS idx_claims_uuid ON Claims(uuid);
    CREATE INDEX IF NOT EXISTS idx_claims_deleted ON Claims(deleted);
    CREATE INDEX IF NOT EXISTS idx_claims_review_status ON Claims(review_status);
    CREATE INDEX IF NOT EXISTS idx_claims_reviewer_id ON Claims(reviewer_id);
    CREATE INDEX IF NOT EXISTS idx_claims_review_group ON Claims(review_group);
    CREATE INDEX IF NOT EXISTS idx_claims_cluster_id ON Claims(claim_cluster_id);

    CREATE TABLE IF NOT EXISTS claims_review_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        claim_id INTEGER NOT NULL,
        old_status TEXT,
        new_status TEXT,
        old_text TEXT,
        new_text TEXT,
        reviewer_id INTEGER,
        notes TEXT,
        reason_code TEXT,
        action_ip TEXT,
        action_user_agent TEXT,
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (claim_id) REFERENCES Claims(id) ON DELETE CASCADE
    );
    CREATE INDEX IF NOT EXISTS idx_claims_review_log_claim ON claims_review_log(claim_id);
    CREATE INDEX IF NOT EXISTS idx_claims_review_log_reviewer ON claims_review_log(reviewer_id);
    CREATE INDEX IF NOT EXISTS idx_claims_review_log_created ON claims_review_log(created_at);

    CREATE TABLE IF NOT EXISTS claims_review_extractor_metrics_daily (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        report_date TEXT NOT NULL,
        extractor TEXT NOT NULL,
        extractor_version TEXT NOT NULL DEFAULT '',
        total_reviewed INTEGER NOT NULL DEFAULT 0,
        approved_count INTEGER NOT NULL DEFAULT 0,
        rejected_count INTEGER NOT NULL DEFAULT 0,
        flagged_count INTEGER NOT NULL DEFAULT 0,
        reassigned_count INTEGER NOT NULL DEFAULT 0,
        edited_count INTEGER NOT NULL DEFAULT 0,
        reason_code_counts_json TEXT,
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(user_id, report_date, extractor, extractor_version)
    );
    CREATE INDEX IF NOT EXISTS idx_claims_review_metrics_user ON claims_review_extractor_metrics_daily(user_id);
    CREATE INDEX IF NOT EXISTS idx_claims_review_metrics_date ON claims_review_extractor_metrics_daily(report_date);
    CREATE INDEX IF NOT EXISTS idx_claims_review_metrics_extractor ON claims_review_extractor_metrics_daily(extractor);

    CREATE TABLE IF NOT EXISTS claims_review_rules (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        priority INTEGER NOT NULL DEFAULT 0,
        predicate_json TEXT NOT NULL,
        reviewer_id INTEGER,
        review_group TEXT,
        active BOOLEAN NOT NULL DEFAULT 1,
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_claims_review_rules_user ON claims_review_rules(user_id);
    CREATE INDEX IF NOT EXISTS idx_claims_review_rules_active ON claims_review_rules(active);

    CREATE TABLE IF NOT EXISTS claims_monitoring_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        event_type TEXT NOT NULL,
        severity TEXT,
        payload_json TEXT,
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        delivered_at DATETIME
    );
    CREATE INDEX IF NOT EXISTS idx_claims_monitoring_events_user ON claims_monitoring_events(user_id);
    CREATE INDEX IF NOT EXISTS idx_claims_monitoring_events_type ON claims_monitoring_events(event_type);
    CREATE INDEX IF NOT EXISTS idx_claims_monitoring_events_delivered ON claims_monitoring_events(delivered_at);

    CREATE TABLE IF NOT EXISTS claims_monitoring_settings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        threshold_ratio REAL,
        baseline_ratio REAL,
        slack_webhook_url TEXT,
        webhook_url TEXT,
        email_recipients TEXT,
        enabled BOOLEAN NOT NULL DEFAULT 1,
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    CREATE UNIQUE INDEX IF NOT EXISTS idx_claims_monitoring_settings_user ON claims_monitoring_settings(user_id);

    CREATE TABLE IF NOT EXISTS claims_monitoring_alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        name TEXT NOT NULL,
        alert_type TEXT NOT NULL,
        threshold_ratio REAL,
        baseline_ratio REAL,
        channels_json TEXT NOT NULL,
        slack_webhook_url TEXT,
        webhook_url TEXT,
        email_recipients TEXT,
        enabled BOOLEAN NOT NULL DEFAULT 1,
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_claims_monitoring_alerts_user ON claims_monitoring_alerts(user_id);

    CREATE TABLE IF NOT EXISTS claims_monitoring_config (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        threshold_ratio REAL,
        baseline_ratio REAL,
        slack_webhook_url TEXT,
        webhook_url TEXT,
        email_recipients TEXT,
        enabled BOOLEAN NOT NULL DEFAULT 1,
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_claims_monitoring_user ON claims_monitoring_config(user_id);

    CREATE TABLE IF NOT EXISTS claims_monitoring_health (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        queue_size INTEGER NOT NULL DEFAULT 0,
        worker_count INTEGER,
        last_worker_heartbeat TEXT,
        last_processed_at TEXT,
        last_failure_at TEXT,
        last_failure_reason TEXT,
        updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    CREATE UNIQUE INDEX IF NOT EXISTS idx_claims_monitoring_health_user ON claims_monitoring_health(user_id);

    CREATE TABLE IF NOT EXISTS claims_analytics_exports (
        export_id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        format TEXT NOT NULL,
        status TEXT NOT NULL,
        payload_json TEXT,
        payload_csv TEXT,
        filters_json TEXT,
        pagination_json TEXT,
        error_message TEXT,
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_claims_analytics_exports_user ON claims_analytics_exports(user_id);

    CREATE TABLE IF NOT EXISTS claims_notifications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        kind TEXT NOT NULL,
        target_user_id TEXT,
        target_review_group TEXT,
        resource_type TEXT,
        resource_id TEXT,
        payload_json TEXT NOT NULL,
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        delivered_at DATETIME
    );
    CREATE INDEX IF NOT EXISTS idx_claims_notifications_user ON claims_notifications(user_id);
    CREATE INDEX IF NOT EXISTS idx_claims_notifications_kind ON claims_notifications(kind);
    CREATE INDEX IF NOT EXISTS idx_claims_notifications_target_user ON claims_notifications(target_user_id);
    CREATE INDEX IF NOT EXISTS idx_claims_notifications_review_group ON claims_notifications(target_review_group);
    CREATE INDEX IF NOT EXISTS idx_claims_notifications_resource ON claims_notifications(resource_type, resource_id);
    CREATE INDEX IF NOT EXISTS idx_claims_notifications_delivered ON claims_notifications(delivered_at);

    CREATE TABLE IF NOT EXISTS claim_clusters (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        canonical_claim_text TEXT,
        representative_claim_id INTEGER,
        summary TEXT,
        cluster_version INTEGER NOT NULL DEFAULT 1,
        watchlist_count INTEGER NOT NULL DEFAULT 0,
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_claim_clusters_user ON claim_clusters(user_id);
    CREATE INDEX IF NOT EXISTS idx_claim_clusters_updated ON claim_clusters(updated_at);

    CREATE TABLE IF NOT EXISTS claim_cluster_membership (
        cluster_id INTEGER NOT NULL,
        claim_id INTEGER NOT NULL,
        similarity_score REAL,
        cluster_joined_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (cluster_id, claim_id)
    );
    CREATE INDEX IF NOT EXISTS idx_claim_cluster_membership_claim ON claim_cluster_membership(claim_id);

    CREATE TABLE IF NOT EXISTS claim_cluster_links (
        parent_cluster_id INTEGER NOT NULL,
        child_cluster_id INTEGER NOT NULL,
        relation_type TEXT,
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (parent_cluster_id, child_cluster_id)
    );
    """

    _MEDIA_FILES_TABLE_SQL = """
    -- MediaFiles Table --
    -- Stores original uploaded files and derived artifacts for media items.
    -- Enables PDF viewing and other original file retrieval features.
    CREATE TABLE IF NOT EXISTS MediaFiles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        media_id INTEGER NOT NULL,
        file_type TEXT NOT NULL DEFAULT 'original',
        storage_path TEXT NOT NULL,
        original_filename TEXT,
        file_size INTEGER,
        mime_type TEXT,
        checksum TEXT,
        uuid TEXT UNIQUE NOT NULL,
        created_at TEXT DEFAULT (datetime('now')),
        last_modified DATETIME NOT NULL,
        version INTEGER NOT NULL DEFAULT 1,
        client_id TEXT NOT NULL,
        deleted BOOLEAN NOT NULL DEFAULT 0,
        prev_version INTEGER,
        merge_parent_uuid TEXT,
        FOREIGN KEY (media_id) REFERENCES Media(id) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_media_files_media_id ON MediaFiles(media_id);
    CREATE INDEX IF NOT EXISTS idx_media_files_type ON MediaFiles(file_type);
    CREATE UNIQUE INDEX IF NOT EXISTS idx_media_files_uuid ON MediaFiles(uuid);
    CREATE INDEX IF NOT EXISTS idx_media_files_deleted ON MediaFiles(deleted);
    """

    _TTS_HISTORY_TABLE_SQL = """
    -- TTS History Table --
    CREATE TABLE IF NOT EXISTS tts_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        created_at TEXT NOT NULL,
        text TEXT,
        text_hash TEXT NOT NULL,
        text_length INTEGER,
        provider TEXT,
        model TEXT,
        voice_id TEXT,
        voice_name TEXT,
        voice_info TEXT,
        format TEXT,
        duration_ms INTEGER,
        generation_time_ms INTEGER,
        params_json TEXT,
        status TEXT,
        segments_json TEXT,
        favorite BOOLEAN NOT NULL DEFAULT 0,
        job_id INTEGER,
        output_id INTEGER,
        artifact_ids TEXT,
        artifact_deleted_at TEXT,
        error_message TEXT,
        deleted BOOLEAN NOT NULL DEFAULT 0,
        deleted_at TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_tts_history_user_created ON tts_history(user_id, created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_tts_history_user_favorite ON tts_history(user_id, favorite);
    CREATE INDEX IF NOT EXISTS idx_tts_history_user_provider ON tts_history(user_id, provider);
    CREATE INDEX IF NOT EXISTS idx_tts_history_user_model ON tts_history(user_id, model);
    CREATE INDEX IF NOT EXISTS idx_tts_history_user_voice_id ON tts_history(user_id, voice_id);
    CREATE INDEX IF NOT EXISTS idx_tts_history_user_text_hash ON tts_history(user_id, text_hash);
    """

    _AUDIO_PRESETS_TABLE_SQL = """
    -- Audio Presets Table --
    CREATE TABLE IF NOT EXISTS audio_presets (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        kind TEXT NOT NULL,
        name TEXT NOT NULL,
        description TEXT,
        favorite BOOLEAN NOT NULL DEFAULT 0,
        is_default BOOLEAN NOT NULL DEFAULT 0,
        config_json TEXT NOT NULL,
        capability_assumptions_json TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        deleted BOOLEAN NOT NULL DEFAULT 0,
        deleted_at TEXT,
        version INTEGER NOT NULL DEFAULT 1
    );
    CREATE INDEX IF NOT EXISTS idx_audio_presets_user_kind_updated
        ON audio_presets(user_id, kind, deleted, updated_at DESC);
    CREATE INDEX IF NOT EXISTS idx_audio_presets_user_kind_favorite
        ON audio_presets(user_id, kind, favorite, deleted);
    CREATE INDEX IF NOT EXISTS idx_audio_presets_user_kind_default
        ON audio_presets(user_id, kind, is_default, deleted);
    CREATE UNIQUE INDEX IF NOT EXISTS ux_audio_presets_user_kind_name_active
        ON audio_presets(user_id, kind, LOWER(name))
        WHERE deleted = FALSE;
    CREATE UNIQUE INDEX IF NOT EXISTS ux_audio_presets_user_kind_default_active
        ON audio_presets(user_id, kind)
        WHERE is_default = TRUE AND deleted = FALSE;
    """

    _DATA_TABLES_SQL = """
    -- Data Tables (LLM-generated structured tables) --
    CREATE TABLE IF NOT EXISTS data_tables (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        uuid TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        description TEXT,
        workspace_tag TEXT,
        prompt TEXT NOT NULL,
        column_hints_json TEXT,
        status TEXT NOT NULL DEFAULT 'queued',
        row_count INTEGER NOT NULL DEFAULT 0,
        generation_model TEXT,
        last_error TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        last_modified DATETIME NOT NULL,
        version INTEGER NOT NULL DEFAULT 1,
        client_id TEXT NOT NULL,
        deleted BOOLEAN NOT NULL DEFAULT 0,
        prev_version INTEGER,
        merge_parent_uuid TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_data_tables_status ON data_tables(status);
    CREATE INDEX IF NOT EXISTS idx_data_tables_updated ON data_tables(updated_at DESC);
    CREATE INDEX IF NOT EXISTS idx_data_tables_deleted ON data_tables(deleted);
    CREATE INDEX IF NOT EXISTS idx_data_tables_workspace_tag ON data_tables(workspace_tag);

    CREATE TABLE IF NOT EXISTS data_table_columns (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        table_id INTEGER NOT NULL,
        column_id TEXT NOT NULL,
        name TEXT NOT NULL,
        type TEXT NOT NULL,
        description TEXT,
        format TEXT,
        position INTEGER NOT NULL,
        created_at TEXT NOT NULL,
        last_modified DATETIME NOT NULL,
        version INTEGER NOT NULL DEFAULT 1,
        client_id TEXT NOT NULL,
        deleted BOOLEAN NOT NULL DEFAULT 0,
        prev_version INTEGER,
        merge_parent_uuid TEXT,
        FOREIGN KEY (table_id) REFERENCES data_tables(id) ON DELETE CASCADE
    );
    CREATE UNIQUE INDEX IF NOT EXISTS ux_data_table_columns_table_column ON data_table_columns(table_id, column_id);
    CREATE INDEX IF NOT EXISTS idx_data_table_columns_table_position ON data_table_columns(table_id, position);
    CREATE UNIQUE INDEX IF NOT EXISTS ux_data_table_columns_table_position_active ON data_table_columns(table_id, position, deleted);

    CREATE TABLE IF NOT EXISTS data_table_rows (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        table_id INTEGER NOT NULL,
        row_id TEXT NOT NULL,
        row_index INTEGER NOT NULL,
        row_json TEXT NOT NULL,
        row_hash TEXT,
        created_at TEXT NOT NULL,
        last_modified DATETIME NOT NULL,
        version INTEGER NOT NULL DEFAULT 1,
        client_id TEXT NOT NULL,
        deleted BOOLEAN NOT NULL DEFAULT 0,
        prev_version INTEGER,
        merge_parent_uuid TEXT,
        FOREIGN KEY (table_id) REFERENCES data_tables(id) ON DELETE CASCADE
    );
    CREATE UNIQUE INDEX IF NOT EXISTS ux_data_table_rows_table_row ON data_table_rows(table_id, row_id);
    CREATE INDEX IF NOT EXISTS idx_data_table_rows_table_index ON data_table_rows(table_id, row_index);
    CREATE UNIQUE INDEX IF NOT EXISTS ux_data_table_rows_table_index_active ON data_table_rows(table_id, row_index, deleted);

    CREATE TABLE IF NOT EXISTS data_table_sources (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        table_id INTEGER NOT NULL,
        source_type TEXT NOT NULL,
        source_id TEXT NOT NULL,
        title TEXT,
        snapshot_json TEXT,
        retrieval_params_json TEXT,
        created_at TEXT NOT NULL,
        last_modified DATETIME NOT NULL,
        version INTEGER NOT NULL DEFAULT 1,
        client_id TEXT NOT NULL,
        deleted BOOLEAN NOT NULL DEFAULT 0,
        prev_version INTEGER,
        merge_parent_uuid TEXT,
        FOREIGN KEY (table_id) REFERENCES data_tables(id) ON DELETE CASCADE
    );
    CREATE INDEX IF NOT EXISTS idx_data_table_sources_table ON data_table_sources(table_id);
    """

    _EMAIL_SCHEMA_SQL = """
    -- Email Sources --
    CREATE TABLE IF NOT EXISTS email_sources (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tenant_id TEXT NOT NULL,
        provider TEXT NOT NULL DEFAULT 'upload',
        source_key TEXT NOT NULL,
        display_name TEXT,
        status TEXT NOT NULL DEFAULT 'active',
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        UNIQUE (tenant_id, provider, source_key)
    );

    -- Email Messages (normalized message identity + denormalized search helper columns)
    CREATE TABLE IF NOT EXISTS email_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tenant_id TEXT NOT NULL,
        media_id INTEGER NOT NULL UNIQUE,
        source_id INTEGER NOT NULL,
        source_message_id TEXT,
        message_id TEXT,
        subject TEXT,
        body_text TEXT,
        internal_date DATETIME,
        from_text TEXT,
        to_text TEXT,
        cc_text TEXT,
        bcc_text TEXT,
        label_text TEXT,
        has_attachments BOOLEAN NOT NULL DEFAULT 0,
        raw_metadata_json TEXT,
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (media_id) REFERENCES Media(id) ON DELETE CASCADE,
        FOREIGN KEY (source_id) REFERENCES email_sources(id) ON DELETE CASCADE
    );

    -- Email Participants --
    CREATE TABLE IF NOT EXISTS email_participants (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tenant_id TEXT NOT NULL,
        email_normalized TEXT NOT NULL,
        display_name TEXT,
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        UNIQUE (tenant_id, email_normalized)
    );

    -- Message <-> Participant role mapping
    CREATE TABLE IF NOT EXISTS email_message_participants (
        email_message_id INTEGER NOT NULL,
        participant_id INTEGER NOT NULL,
        role TEXT NOT NULL CHECK (role IN ('from', 'to', 'cc', 'bcc')),
        PRIMARY KEY (email_message_id, participant_id, role),
        FOREIGN KEY (email_message_id) REFERENCES email_messages(id) ON DELETE CASCADE,
        FOREIGN KEY (participant_id) REFERENCES email_participants(id) ON DELETE CASCADE
    );

    -- Labels and message-label mappings
    CREATE TABLE IF NOT EXISTS email_labels (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tenant_id TEXT NOT NULL,
        label_key TEXT NOT NULL,
        label_name TEXT NOT NULL,
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        UNIQUE (tenant_id, label_key)
    );

    CREATE TABLE IF NOT EXISTS email_message_labels (
        email_message_id INTEGER NOT NULL,
        label_id INTEGER NOT NULL,
        PRIMARY KEY (email_message_id, label_id),
        FOREIGN KEY (email_message_id) REFERENCES email_messages(id) ON DELETE CASCADE,
        FOREIGN KEY (label_id) REFERENCES email_labels(id) ON DELETE CASCADE
    );

    -- Attachment metadata
    CREATE TABLE IF NOT EXISTS email_attachments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email_message_id INTEGER NOT NULL,
        filename TEXT,
        content_type TEXT,
        size_bytes INTEGER,
        content_id TEXT,
        disposition TEXT,
        extracted_text_available BOOLEAN NOT NULL DEFAULT 0,
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (email_message_id) REFERENCES email_messages(id) ON DELETE CASCADE
    );

    -- Sync cursor/checkpoint state
    CREATE TABLE IF NOT EXISTS email_sync_state (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tenant_id TEXT NOT NULL,
        source_id INTEGER NOT NULL,
        cursor TEXT,
        last_run_at DATETIME,
        last_success_at DATETIME,
        error_state TEXT,
        retry_backoff_count INTEGER NOT NULL DEFAULT 0,
        updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        UNIQUE (tenant_id, source_id),
        FOREIGN KEY (source_id) REFERENCES email_sources(id) ON DELETE CASCADE
    );

    -- Legacy media -> normalized email backfill checkpoint state
    CREATE TABLE IF NOT EXISTS email_backfill_state (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tenant_id TEXT NOT NULL,
        backfill_key TEXT NOT NULL,
        last_media_id INTEGER NOT NULL DEFAULT 0,
        processed_count INTEGER NOT NULL DEFAULT 0,
        success_count INTEGER NOT NULL DEFAULT 0,
        skipped_count INTEGER NOT NULL DEFAULT 0,
        failed_count INTEGER NOT NULL DEFAULT 0,
        status TEXT NOT NULL DEFAULT 'idle',
        last_error TEXT,
        started_at DATETIME,
        completed_at DATETIME,
        updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        UNIQUE (tenant_id, backfill_key)
    );
    """

    _EMAIL_INDICES_SQL = """
    CREATE INDEX IF NOT EXISTS idx_email_sources_tenant_provider ON email_sources(tenant_id, provider);
    CREATE INDEX IF NOT EXISTS idx_email_messages_tenant_date ON email_messages(tenant_id, internal_date);
    CREATE INDEX IF NOT EXISTS idx_email_messages_tenant_date_id ON email_messages(tenant_id, internal_date DESC, id DESC);
    CREATE INDEX IF NOT EXISTS idx_email_messages_tenant_has_attachments_date
        ON email_messages(tenant_id, has_attachments, internal_date DESC, id DESC);
    CREATE INDEX IF NOT EXISTS idx_email_messages_source_id ON email_messages(source_id);
    CREATE UNIQUE INDEX IF NOT EXISTS idx_email_messages_tenant_source_message
        ON email_messages(tenant_id, source_id, source_message_id)
        WHERE source_message_id IS NOT NULL AND source_message_id <> '';
    CREATE UNIQUE INDEX IF NOT EXISTS idx_email_messages_tenant_message_id
        ON email_messages(tenant_id, source_id, message_id)
        WHERE message_id IS NOT NULL AND message_id <> '';
    CREATE INDEX IF NOT EXISTS idx_email_participants_tenant_email ON email_participants(tenant_id, email_normalized);
    CREATE INDEX IF NOT EXISTS idx_email_message_participants_role ON email_message_participants(role);
    CREATE INDEX IF NOT EXISTS idx_email_message_participants_message_role
        ON email_message_participants(email_message_id, role, participant_id);
    CREATE INDEX IF NOT EXISTS idx_email_labels_tenant_name ON email_labels(tenant_id, label_name);
    CREATE INDEX IF NOT EXISTS idx_email_message_labels_label ON email_message_labels(label_id);
    CREATE INDEX IF NOT EXISTS idx_email_attachments_message_id ON email_attachments(email_message_id);
    CREATE INDEX IF NOT EXISTS idx_email_sync_state_tenant_source ON email_sync_state(tenant_id, source_id);
    CREATE INDEX IF NOT EXISTS idx_email_backfill_state_status ON email_backfill_state(status);
    """

    _EMAIL_SQLITE_FTS_SQL = """
    CREATE VIRTUAL TABLE IF NOT EXISTS email_fts USING fts5(
        subject,
        body_text,
        from_text,
        to_text,
        cc_text,
        bcc_text,
        label_text,
        content='email_messages',
        content_rowid='id'
    );
    """

    @property
    def _SCHEMA_UPDATE_VERSION_SQL_V1(self) -> str:
        return (
            "DELETE FROM schema_version WHERE version <> 0;\n"  # nosec B608
            f"UPDATE schema_version SET version = {self._CURRENT_SCHEMA_VERSION} WHERE version = 0;"
        )

    @staticmethod
    def _normalize_email_address(value: Any) -> str | None:
        addr = str(value or "").strip().lower()
        return addr if addr and "@" in addr else None

    @staticmethod
    def _parse_email_internal_date(value: Any) -> str | None:
        raw = str(value or "").strip()
        if not raw:
            return None
        with suppress(MEDIA_NONCRITICAL_EXCEPTIONS):
            dt = parsedate_to_datetime(raw)
            if dt is not None:
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc).isoformat()
        return None

    @staticmethod
    def _collect_email_labels(
        metadata: dict[str, Any] | None = None,
        labels: list[str] | str | None = None,
    ) -> list[str]:
        raw_values: list[str] = []
        if isinstance(labels, str):
            raw_values.extend(v.strip() for v in labels.split(","))
        elif isinstance(labels, list):
            raw_values.extend(str(v).strip() for v in labels if v is not None)

        email_meta = (metadata or {}).get("email")
        if isinstance(email_meta, dict):
            meta_labels = email_meta.get("labels")
            if isinstance(meta_labels, str):
                raw_values.extend(v.strip() for v in meta_labels.split(","))
            elif isinstance(meta_labels, list):
                raw_values.extend(str(v).strip() for v in meta_labels if v is not None)

        top_labels = (metadata or {}).get("labels")
        if isinstance(top_labels, str):
            raw_values.extend(v.strip() for v in top_labels.split(","))
        elif isinstance(top_labels, list):
            raw_values.extend(str(v).strip() for v in top_labels if v is not None)

        normalized: list[str] = []
        seen: set[str] = set()
        for value in raw_values:
            label = str(value or "").strip()
            if not label:
                continue
            key = label.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(label)
        return normalized

    @staticmethod
    def _normalize_email_sync_cursor(value: Any) -> str | None:
        text = str(value or "").strip()
        return text or None

    @staticmethod
    def _normalize_email_backfill_key(value: Any) -> str:
        text = str(value or "").strip().lower()
        return text[:128] if text else "legacy_media_email"

    @staticmethod
    def _parse_email_backfill_safe_metadata(raw_safe_metadata: Any) -> dict[str, Any]:
        if isinstance(raw_safe_metadata, dict):
            metadata_map = dict(raw_safe_metadata)
        else:
            raw_text = str(raw_safe_metadata or "").strip()
            if not raw_text:
                return {}
            try:
                parsed = json.loads(raw_text)
            except MEDIA_NONCRITICAL_EXCEPTIONS:
                return {}
            if not isinstance(parsed, dict):
                return {}
            metadata_map = dict(parsed)

        nested_meta = metadata_map.get("metadata")
        if (
            not isinstance(metadata_map.get("email"), dict)
            and isinstance(nested_meta, dict)
            and isinstance(nested_meta.get("email"), dict)
        ):
            # Some older document versions wrap parser metadata under "metadata".
            merged = dict(nested_meta)
            for key, value in metadata_map.items():
                if key not in merged:
                    merged[key] = value
            metadata_map = merged

        return metadata_map

    @staticmethod
    def _derive_email_backfill_source_fields(
        *,
        metadata_map: dict[str, Any],
        media_url: Any,
        tenant_id: str,
    ) -> tuple[str, str, str | None]:
        url_text = str(media_url or "").strip()
        provider_hint = str(metadata_map.get("provider") or "").strip().lower()
        source_hint = str(metadata_map.get("source") or "").strip().lower()
        email_meta = metadata_map.get("email")
        email_map = email_meta if isinstance(email_meta, dict) else {}

        provider = provider_hint
        if not provider:
            if source_hint in {"gmail", "gmail_connector"} or url_text.lower().startswith("gmail://"):
                provider = "gmail"
            else:
                provider = "upload"

        source_key = str(
            metadata_map.get("source_key")
            or email_map.get("source_key")
            or ""
        ).strip()
        source_message_id = str(
            email_map.get("source_message_id")
            or metadata_map.get("source_message_id")
            or ""
        ).strip() or None

        if url_text.lower().startswith("gmail://"):
            path = url_text[len("gmail://") :]
            source_part, sep, message_part = path.partition("/")
            source_part = source_part.strip()
            message_part = message_part.strip()
            if provider == "gmail" and not source_key and source_part:
                source_key = source_part
            if source_message_id is None and sep and message_part:
                source_message_id = message_part

        if not source_key:
            source_key = f"legacy-media:{tenant_id}"

        return provider, source_key, source_message_id

    @contextmanager
    def transaction(self):
        """Provide nested transaction handling across SQLite and PostgreSQL backends."""

        if self.backend_type == BackendType.SQLITE:
            outermost = self._get_txn_conn() is None
            if outermost:
                conn = self._persistent_conn or self.backend.connect()
                with suppress(MEDIA_NONCRITICAL_EXCEPTIONS):
                    conn.row_factory = sqlite3.Row
                try:
                    self._apply_sqlite_connection_pragmas(conn)
                except sqlite3.Error:
                    pass
            else:
                conn = self._get_txn_conn()

            self._inc_tx_depth()
            try:
                if outermost:
                    begin_immediate_if_needed(conn)
                    self._set_txn_conn(conn)
                    logging.debug("Started SQLite transaction.")
                yield conn
                if outermost:
                    conn.commit()
                    logging.debug("Committed SQLite transaction.")
            except MEDIA_NONCRITICAL_EXCEPTIONS:
                logging.exception("SQLite transaction failed, rolling back")
                if outermost:
                    with suppress(sqlite3.Error):
                        conn.rollback()
                raise
            finally:
                self._dec_tx_depth()
                if outermost:
                    self._set_txn_conn(None)
                    if self._persistent_conn is None:
                        with suppress(MEDIA_NONCRITICAL_EXCEPTIONS):
                            conn.close()
            return

        manages_backend_conn = self._get_txn_conn() is None
        conn = (
            self.backend.get_pool().get_connection()
            if manages_backend_conn
            else self._get_txn_conn()
        )

        manages_backend_tx = self._get_tx_depth() == 0
        self._inc_tx_depth()
        ctx = self.backend.transaction(conn) if manages_backend_tx else nullcontext(conn)
        try:
            with ctx as inner_conn:
                self._set_txn_conn(inner_conn)
                yield inner_conn
        finally:
            depth = self._dec_tx_depth()
            if depth == 0:
                self._set_txn_conn(None)
            if manages_backend_conn:
                with suppress(MEDIA_NONCRITICAL_EXCEPTIONS):
                    self.backend.get_pool().return_connection(conn)

MediaDatabase.add_media_chunks_in_batches = add_media_chunks_in_batches
MediaDatabase.batch_insert_chunks = batch_insert_chunks
MediaDatabase.clear_unvectorized_chunks = clear_unvectorized_chunks
MediaDatabase.create_chunking_template = create_chunking_template
MediaDatabase.delete_chunking_template = delete_chunking_template
MediaDatabase.create_document_version = create_document_version
MediaDatabase.fetch_all_keywords = fetch_all_keywords
MediaDatabase.get_all_document_versions = get_all_document_versions
MediaDatabase.get_chunking_template = get_chunking_template
MediaDatabase.get_distinct_media_types = get_distinct_media_types
MediaDatabase.get_media_by_id = get_media_by_id
MediaDatabase.get_media_by_hash = get_media_by_hash
MediaDatabase.get_media_status_by_id = get_media_status_by_id
MediaDatabase.get_media_by_title = get_media_by_title
MediaDatabase.get_media_by_url = get_media_by_url
MediaDatabase.get_media_by_uuid = get_media_by_uuid
MediaDatabase.search_media_db = run_search_media_db
MediaDatabase.get_paginated_media_list = get_paginated_media_list
MediaDatabase.get_paginated_files = get_paginated_files
MediaDatabase.get_paginated_trash_list = get_paginated_trash_list
MediaDatabase.has_unvectorized_chunks = has_unvectorized_chunks
MediaDatabase.get_unvectorized_anchor_index_for_offset = get_unvectorized_anchor_index_for_offset
MediaDatabase.get_unvectorized_chunk_by_index = get_unvectorized_chunk_by_index
MediaDatabase.get_unvectorized_chunk_count = get_unvectorized_chunk_count
MediaDatabase.get_unvectorized_chunk_index_by_uuid = get_unvectorized_chunk_index_by_uuid
MediaDatabase.get_unvectorized_chunks_in_range = get_unvectorized_chunks_in_range
MediaDatabase._postgres_migrate_to_v5 = run_postgres_migrate_to_v5
MediaDatabase._postgres_migrate_to_v6 = run_postgres_migrate_to_v6
MediaDatabase._postgres_migrate_to_v7 = run_postgres_migrate_to_v7
MediaDatabase._postgres_migrate_to_v8 = run_postgres_migrate_to_v8
MediaDatabase._postgres_migrate_to_v9 = run_postgres_migrate_to_v9
MediaDatabase._postgres_migrate_to_v10 = run_postgres_migrate_to_v10
MediaDatabase._postgres_migrate_to_v11 = run_postgres_migrate_to_v11
MediaDatabase._postgres_migrate_to_v12 = run_postgres_migrate_to_v12
MediaDatabase._postgres_migrate_to_v13 = run_postgres_migrate_to_v13
MediaDatabase._postgres_migrate_to_v14 = run_postgres_migrate_to_v14
MediaDatabase._postgres_migrate_to_v15 = run_postgres_migrate_to_v15
MediaDatabase._postgres_migrate_to_v16 = run_postgres_migrate_to_v16
MediaDatabase._postgres_migrate_to_v17 = run_postgres_migrate_to_v17
MediaDatabase._postgres_migrate_to_v18 = run_postgres_migrate_to_v18
MediaDatabase._postgres_migrate_to_v19 = run_postgres_migrate_to_v19
MediaDatabase._postgres_migrate_to_v20 = run_postgres_migrate_to_v20
MediaDatabase._postgres_migrate_to_v21 = run_postgres_migrate_to_v21
MediaDatabase._postgres_migrate_to_v22 = run_postgres_migrate_to_v22
MediaDatabase._postgres_migrate_to_v23 = run_postgres_migrate_to_v23
MediaDatabase._get_db_version = get_db_version
MediaDatabase._update_schema_version_postgres = update_schema_version_postgres
MediaDatabase._sync_postgres_sequences = sync_postgres_sequences
MediaDatabase._convert_sqlite_sql_to_postgres_statements = _convert_sqlite_sql_to_postgres_statements
MediaDatabase._transform_sqlite_statement_to_postgres = _transform_sqlite_statement_to_postgres
MediaDatabase._ensure_fts_structures = ensure_fts_structures
MediaDatabase._ensure_sqlite_fts = ensure_sqlite_fts
MediaDatabase._ensure_postgres_fts = ensure_postgres_fts
MediaDatabase._ensure_sqlite_email_schema = ensure_sqlite_email_schema
MediaDatabase._ensure_postgres_email_schema = ensure_postgres_email_schema
MediaDatabase._ensure_sqlite_visibility_columns = ensure_sqlite_visibility_columns
MediaDatabase._ensure_sqlite_source_hash_column = ensure_sqlite_source_hash_column
MediaDatabase._ensure_sqlite_data_tables = ensure_sqlite_data_tables
MediaDatabase._ensure_sqlite_claims_extensions = ensure_sqlite_claims_extensions
MediaDatabase._ensure_postgres_data_tables = ensure_postgres_data_tables
MediaDatabase._ensure_postgres_columns = ensure_postgres_columns
MediaDatabase._ensure_postgres_data_tables_columns = ensure_postgres_data_tables_columns
MediaDatabase._ensure_postgres_tts_history = ensure_postgres_tts_history
MediaDatabase._ensure_postgres_audio_presets = ensure_postgres_audio_presets
MediaDatabase._ensure_postgres_source_hash_column = ensure_postgres_source_hash_column
MediaDatabase._ensure_postgres_claims_tables = ensure_postgres_claims_tables
MediaDatabase._ensure_postgres_collections_tables = ensure_postgres_collections_tables
MediaDatabase._ensure_postgres_claims_extensions = ensure_postgres_claims_extensions
MediaDatabase.list_chunking_templates = list_chunking_templates
MediaDatabase.lookup_section_by_heading = lookup_section_by_heading
MediaDatabase.lookup_section_for_offset = lookup_section_for_offset
MediaDatabase.process_chunks = process_chunks
MediaDatabase.process_unvectorized_chunks = process_unvectorized_chunks
MediaDatabase.mark_embeddings_error = mark_embeddings_error
MediaDatabase._get_txn_conn = _get_txn_conn
MediaDatabase._set_txn_conn = _set_txn_conn
MediaDatabase._get_tx_depth = _get_tx_depth
MediaDatabase._set_tx_depth = _set_tx_depth
MediaDatabase._inc_tx_depth = _inc_tx_depth
MediaDatabase._dec_tx_depth = _dec_tx_depth
MediaDatabase._get_persistent_conn = _get_persistent_conn
MediaDatabase._set_persistent_conn = _set_persistent_conn
MediaDatabase._prepare_backend_statement = _prepare_backend_statement
MediaDatabase._prepare_backend_many_statement = _prepare_backend_many_statement
MediaDatabase._normalise_params = _normalise_params
MediaDatabase._resolve_scope_ids = _resolve_scope_ids
MediaDatabase._initialize_schema = ensure_media_schema
MediaDatabase._initialize_schema_sqlite = bootstrap_sqlite_schema
MediaDatabase._initialize_schema_postgres = bootstrap_postgres_schema
MediaDatabase._run_postgres_migrations = run_postgres_migrations
MediaDatabase._get_postgres_migrations = get_postgres_migrations
MediaDatabase._apply_schema_v1_sqlite = apply_sqlite_core_media_schema
MediaDatabase._apply_schema_v1_postgres = apply_postgres_core_media_schema
MediaDatabase._postgres_policy_exists = _postgres_policy_exists
MediaDatabase._ensure_postgres_rls = _ensure_postgres_rls
MediaDatabase._generate_uuid = _generate_uuid
MediaDatabase._get_current_utc_timestamp_str = _get_current_utc_timestamp_str
MediaDatabase._get_next_version = _get_next_version
MediaDatabase._log_sync_event = _log_sync_event
MediaDatabase.get_sync_log_entries = get_sync_log_entries
MediaDatabase.delete_sync_log_entries = delete_sync_log_entries
MediaDatabase.delete_sync_log_entries_before = delete_sync_log_entries_before
MediaDatabase._write_structure_index_records = _write_structure_index_records
MediaDatabase.write_document_structure_index = write_document_structure_index
MediaDatabase.delete_document_structure_for_media = delete_document_structure_for_media
MediaDatabase._keyword_order_expression = _keyword_order_expression
MediaDatabase._append_case_insensitive_like = _append_case_insensitive_like
MediaDatabase._convert_sqlite_placeholders_to_postgres = _convert_sqlite_placeholders_to_postgres
MediaDatabase.insert_media_file = insert_media_file
MediaDatabase.get_media_file = get_media_file
MediaDatabase.get_media_files = get_media_files
MediaDatabase.has_original_file = has_original_file
MediaDatabase.soft_delete_media = soft_delete_media
MediaDatabase.share_media = share_media
MediaDatabase.unshare_media = unshare_media
MediaDatabase.get_media_visibility = get_media_visibility
MediaDatabase.mark_as_trash = mark_as_trash
MediaDatabase.restore_from_trash = restore_from_trash
MediaDatabase.add_media_with_keywords = run_add_media_with_keywords
MediaDatabase._resolve_data_tables_owner = _resolve_data_tables_owner
MediaDatabase._resolve_data_table_write_client_id = _resolve_data_table_write_client_id
MediaDatabase._get_data_table_owner_client_id = _get_data_table_owner_client_id
MediaDatabase._soft_delete_data_table_children = _soft_delete_data_table_children
MediaDatabase._normalize_data_table_row_json = _normalize_data_table_row_json
MediaDatabase.create_data_table = create_data_table
MediaDatabase.get_data_table = get_data_table
MediaDatabase.get_data_table_by_uuid = get_data_table_by_uuid
MediaDatabase.list_data_tables = list_data_tables
MediaDatabase.count_data_tables = count_data_tables
MediaDatabase.update_data_table = update_data_table
MediaDatabase.soft_delete_data_table = soft_delete_data_table
MediaDatabase.persist_data_table_generation = persist_data_table_generation
MediaDatabase.replace_data_table_contents = replace_data_table_contents
MediaDatabase.search_by_safe_metadata = search_by_safe_metadata
MediaDatabase.apply_synced_document_content_update = (
    apply_synced_document_content_update
)
MediaDatabase.rollback_to_version = rollback_to_version
MediaDatabase._resolve_email_tenant_id = _resolve_email_tenant_id
MediaDatabase.upsert_email_message_graph = upsert_email_message_graph
MediaDatabase.create_claims_analytics_export = create_claims_analytics_export
MediaDatabase.get_claims_analytics_export = get_claims_analytics_export
MediaDatabase.list_claims_analytics_exports = list_claims_analytics_exports
MediaDatabase.count_claims_analytics_exports = count_claims_analytics_exports
MediaDatabase.cleanup_claims_analytics_exports = cleanup_claims_analytics_exports
MediaDatabase.insert_claim_notification = insert_claim_notification
MediaDatabase.get_claim_notification = get_claim_notification
MediaDatabase.get_latest_claim_notification = get_latest_claim_notification
MediaDatabase.list_claim_notifications = list_claim_notifications
MediaDatabase.get_claim_notifications_by_ids = get_claim_notifications_by_ids
MediaDatabase.mark_claim_notifications_delivered = mark_claim_notifications_delivered
MediaDatabase.list_claim_review_rules = list_claim_review_rules
MediaDatabase.create_claim_review_rule = create_claim_review_rule
MediaDatabase.get_claim_review_rule = get_claim_review_rule
MediaDatabase.update_claim_review_rule = update_claim_review_rule
MediaDatabase.delete_claim_review_rule = delete_claim_review_rule
MediaDatabase.list_claim_review_history = list_claim_review_history
MediaDatabase.list_review_queue = list_review_queue
MediaDatabase.get_claims_by_media = get_claims_by_media
MediaDatabase.get_claim_with_media = get_claim_with_media
MediaDatabase.get_claims_by_uuid = get_claims_by_uuid
MediaDatabase.upsert_claims = upsert_claims
MediaDatabase.update_claim = update_claim
MediaDatabase.update_claim_review = update_claim_review
MediaDatabase.soft_delete_claims_for_media = soft_delete_claims_for_media
MediaDatabase.list_claims = list_claims
MediaDatabase.search_claims = search_claims
MediaDatabase.get_claims_review_extractor_metrics_daily = (
    get_claims_review_extractor_metrics_daily
)
MediaDatabase.upsert_claims_review_extractor_metrics_daily = (
    upsert_claims_review_extractor_metrics_daily
)
MediaDatabase.list_claims_review_extractor_metrics_daily = (
    list_claims_review_extractor_metrics_daily
)
MediaDatabase.count_claims_review_extractor_metrics_daily = (
    count_claims_review_extractor_metrics_daily
)
MediaDatabase.list_claims_review_user_ids = list_claims_review_user_ids
MediaDatabase.get_claim_clusters_by_ids = get_claim_clusters_by_ids
MediaDatabase.get_claim_cluster_member_counts = get_claim_cluster_member_counts
MediaDatabase.update_claim_clusters_watchlist_counts = (
    update_claim_clusters_watchlist_counts
)
MediaDatabase.list_claim_clusters = list_claim_clusters
MediaDatabase.get_claim_cluster = get_claim_cluster
MediaDatabase.get_claim_cluster_link = get_claim_cluster_link
MediaDatabase.list_claim_cluster_links = list_claim_cluster_links
MediaDatabase.create_claim_cluster_link = create_claim_cluster_link
MediaDatabase.delete_claim_cluster_link = delete_claim_cluster_link
MediaDatabase.list_claim_cluster_members = list_claim_cluster_members
MediaDatabase.create_claim_cluster = create_claim_cluster
MediaDatabase.add_claim_to_cluster = add_claim_to_cluster
MediaDatabase.rebuild_claim_clusters_exact = rebuild_claim_clusters_exact
MediaDatabase.rebuild_claim_clusters_from_assignments = (
    rebuild_claim_clusters_from_assignments
)
MediaDatabase.rebuild_claims_fts = rebuild_claims_fts
MediaDatabase.get_claims_monitoring_settings = get_claims_monitoring_settings
MediaDatabase.upsert_claims_monitoring_settings = upsert_claims_monitoring_settings
MediaDatabase.migrate_legacy_claims_monitoring_alerts = (
    migrate_legacy_claims_monitoring_alerts
)
MediaDatabase.delete_claims_monitoring_configs_by_user = (
    delete_claims_monitoring_configs_by_user
)
MediaDatabase.list_claims_monitoring_configs = list_claims_monitoring_configs
MediaDatabase.create_claims_monitoring_config = create_claims_monitoring_config
MediaDatabase.get_claims_monitoring_config = get_claims_monitoring_config
MediaDatabase.update_claims_monitoring_config = update_claims_monitoring_config
MediaDatabase.delete_claims_monitoring_config = delete_claims_monitoring_config
MediaDatabase.list_claims_monitoring_user_ids = list_claims_monitoring_user_ids
MediaDatabase.list_claims_monitoring_alerts = list_claims_monitoring_alerts
MediaDatabase.get_claims_monitoring_alert = get_claims_monitoring_alert
MediaDatabase.create_claims_monitoring_alert = create_claims_monitoring_alert
MediaDatabase.update_claims_monitoring_alert = update_claims_monitoring_alert
MediaDatabase.delete_claims_monitoring_alert = delete_claims_monitoring_alert
MediaDatabase.get_claims_monitoring_health = get_claims_monitoring_health
MediaDatabase.upsert_claims_monitoring_health = upsert_claims_monitoring_health
MediaDatabase.insert_claims_monitoring_event = insert_claims_monitoring_event
MediaDatabase.list_claims_monitoring_events = list_claims_monitoring_events
MediaDatabase.list_undelivered_claims_monitoring_events = (
    list_undelivered_claims_monitoring_events
)
MediaDatabase.mark_claims_monitoring_events_delivered = (
    mark_claims_monitoring_events_delivered
)
MediaDatabase.get_latest_claims_monitoring_event_delivery = (
    get_latest_claims_monitoring_event_delivery
)
MediaDatabase._resolve_email_sync_source_row_id = _resolve_email_sync_source_row_id
MediaDatabase._fetch_email_sync_state_row = _fetch_email_sync_state_row
MediaDatabase.get_email_sync_state = get_email_sync_state
MediaDatabase.mark_email_sync_run_started = mark_email_sync_run_started
MediaDatabase.mark_email_sync_run_succeeded = mark_email_sync_run_succeeded
MediaDatabase.mark_email_sync_run_failed = mark_email_sync_run_failed
MediaDatabase._normalize_email_label_values = staticmethod(_normalize_email_label_values)
MediaDatabase._resolve_email_message_row_for_source_message = (
    _resolve_email_message_row_for_source_message
)
MediaDatabase.apply_email_label_delta = apply_email_label_delta
MediaDatabase.reconcile_email_message_state = reconcile_email_message_state
MediaDatabase._parse_email_operator_query = _parse_email_operator_query
MediaDatabase._email_like_clause = _email_like_clause
MediaDatabase.search_email_messages = search_email_messages
MediaDatabase.get_email_message_detail = get_email_message_detail
MediaDatabase._cleanup_email_orphans_for_tenant = _cleanup_email_orphans_for_tenant
MediaDatabase.enforce_email_retention_policy = enforce_email_retention_policy
MediaDatabase.hard_delete_email_tenant_data = hard_delete_email_tenant_data
MediaDatabase.run_email_legacy_backfill_batch = run_email_legacy_backfill_batch
MediaDatabase.run_email_legacy_backfill_worker = run_email_legacy_backfill_worker
MediaDatabase._fetch_email_backfill_state_row = _fetch_email_backfill_state_row
MediaDatabase._ensure_email_backfill_state_row = _ensure_email_backfill_state_row
MediaDatabase.get_email_legacy_backfill_state = get_email_legacy_backfill_state
MediaDatabase._update_email_backfill_progress = _update_email_backfill_progress
MediaDatabase.get_data_table_counts = get_data_table_counts
MediaDatabase.insert_data_table_columns = insert_data_table_columns
MediaDatabase.list_data_table_columns = list_data_table_columns
MediaDatabase.soft_delete_data_table_columns = soft_delete_data_table_columns
MediaDatabase.insert_data_table_rows = insert_data_table_rows
MediaDatabase.list_data_table_rows = list_data_table_rows
MediaDatabase.soft_delete_data_table_rows = soft_delete_data_table_rows
MediaDatabase.insert_data_table_sources = insert_data_table_sources
MediaDatabase.list_data_table_sources = list_data_table_sources
MediaDatabase.soft_delete_data_table_sources = soft_delete_data_table_sources
MediaDatabase.soft_delete_media_file = soft_delete_media_file
MediaDatabase.soft_delete_media_files_for_media = soft_delete_media_files_for_media
MediaDatabase.insert_visual_document = insert_visual_document
MediaDatabase.list_visual_documents_for_media = list_visual_documents_for_media
MediaDatabase.soft_delete_visual_documents_for_media = soft_delete_visual_documents_for_media
MediaDatabase.create_tts_history_entry = create_tts_history_entry
MediaDatabase._build_tts_history_filters = _build_tts_history_filters
MediaDatabase.list_tts_history = list_tts_history
MediaDatabase.count_tts_history = count_tts_history
MediaDatabase.get_tts_history_entry = get_tts_history_entry
MediaDatabase.update_tts_history_favorite = update_tts_history_favorite
MediaDatabase.soft_delete_tts_history_entry = soft_delete_tts_history_entry
MediaDatabase.mark_tts_history_artifacts_deleted_for_output = (
    mark_tts_history_artifacts_deleted_for_output
)
MediaDatabase.mark_tts_history_artifacts_deleted_for_file_id = (
    mark_tts_history_artifacts_deleted_for_file_id
)
MediaDatabase.purge_tts_history_for_user = purge_tts_history_for_user
MediaDatabase.list_tts_history_user_ids = list_tts_history_user_ids
MediaDatabase.create_audio_preset = create_audio_preset
MediaDatabase.list_audio_presets = list_audio_presets
MediaDatabase.count_audio_presets = count_audio_presets
MediaDatabase.get_audio_preset = get_audio_preset
MediaDatabase.update_audio_preset = update_audio_preset
MediaDatabase.soft_delete_audio_preset = soft_delete_audio_preset
MediaDatabase.get_connection = get_connection
MediaDatabase.close_connection = close_connection
MediaDatabase.release_context_connection = release_context_connection
MediaDatabase._execute_with_connection = _execute_with_connection
MediaDatabase._executemany_with_connection = _executemany_with_connection
MediaDatabase._fetchone_with_connection = _fetchone_with_connection
MediaDatabase._fetchall_with_connection = _fetchall_with_connection
MediaDatabase.execute_query = execute_query
MediaDatabase.execute_many = execute_many
MediaDatabase._update_fts_media = _update_fts_media
MediaDatabase._delete_fts_media = _delete_fts_media
MediaDatabase._update_fts_keyword = _update_fts_keyword
MediaDatabase._delete_fts_keyword = _delete_fts_keyword
MediaDatabase.sync_refresh_fts_for_entity = sync_refresh_fts_for_entity
MediaDatabase.ensure_chunk_fts = ensure_chunk_fts
MediaDatabase.maybe_rebuild_chunk_fts_if_empty = maybe_rebuild_chunk_fts_if_empty
MediaDatabase._apply_sqlite_connection_pragmas = apply_sqlite_connection_pragmas
MediaDatabase.__init__ = initialize_media_database
MediaDatabase.initialize_db = initialize_db
MediaDatabase._ensure_sqlite_backend = _ensure_sqlite_backend
MediaDatabase.backup_database = backup_database
MediaDatabase._backup_non_sqlite_database = _backup_non_sqlite_database
MediaDatabase._resolve_backend = _resolve_backend
MediaDatabase.seed_builtin_templates = seed_builtin_templates
MediaDatabase.add_keyword = add_keyword
MediaDatabase.fetch_media_for_keywords = fetch_media_for_keywords
MediaDatabase.soft_delete_document_version = soft_delete_document_version
MediaDatabase.soft_delete_keyword = soft_delete_keyword
MediaDatabase.update_media_reprocess_state = update_media_reprocess_state
MediaDatabase.update_chunking_template = update_chunking_template
MediaDatabase.update_keywords_for_media = update_keywords_for_media

__all__ = ["MediaDatabase"]
