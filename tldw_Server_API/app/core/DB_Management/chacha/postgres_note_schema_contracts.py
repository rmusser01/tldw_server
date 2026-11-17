"""Fixed PostgreSQL Notes catalog and ordered bootstrap index contracts."""

NOTE_ATTACHMENT_COLUMN_CONTRACTS = (
    ("client_id", "text", True, None),
    ("dataset_id", "text", True, None),
    ("attachment_id", "text", True, None),
    ("note_id", "text", True, None),
    ("file_name", "text", True, None),
    ("normalized_file_name", "text", True, None),
    ("original_file_name", "text", True, None),
    ("content_type", "text", True, None),
    ("size_bytes", "bigint", True, None),
    ("blob_hash", "text", True, None),
    ("object_hash", "text", True, None),
    ("version", "bigint", True, None),
    ("deleted", "boolean", True, "false"),
    ("deleted_at", "timestamp with time zone", False, None),
    ("delete_reason", "text", False, None),
    ("created_at", "timestamp with time zone", True, None),
    ("last_modified", "timestamp with time zone", True, None),
    ("created_by", "text", True, None),
    ("source_kind", "text", True, None),
)

NOTE_TASK_PRIMARY_KEYS = {
    "note_tasks_pkey": ("note_tasks", ("owner_user_id", "dataset_id", "id")),
    "task_note_projections_pkey": ("task_note_projections", ("owner_user_id", "dataset_id", "task_id")),
    "task_events_pkey": ("task_events", ("owner_user_id", "dataset_id", "id")),
    "task_event_read_state_pkey": ("task_event_read_state", ("owner_user_id", "dataset_id", "event_id", "user_id")),
    "note_task_reconciliation_state_pkey": (
        "note_task_reconciliation_state",
        ("owner_user_id", "dataset_id", "note_id"),
    ),
    "task_projection_drifts_pkey": ("task_projection_drifts", ("owner_user_id", "dataset_id", "id")),
    "note_task_scope_authority_pkey": ("note_task_scope_authority", ("owner_user_id",)),
}

POSTGRES_CORE_INDEX_STATEMENTS = (
    "CREATE INDEX IF NOT EXISTS idx_flashcards_created_at ON flashcards(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_flashcards_conversation ON flashcards(conversation_id)",
    "CREATE INDEX IF NOT EXISTS idx_flashcards_message ON flashcards(message_id)",
    "CREATE INDEX IF NOT EXISTS idx_conversations_state ON conversations(state)",
    "CREATE INDEX IF NOT EXISTS idx_conversations_cluster ON conversations(cluster_id)",
    "CREATE INDEX IF NOT EXISTS idx_conversations_last_modified ON conversations(last_modified)",
    "CREATE INDEX IF NOT EXISTS idx_conversations_topic_label ON conversations(topic_label)",
    "CREATE INDEX IF NOT EXISTS idx_conversations_source_external_ref ON conversations(source, external_ref)",
    "CREATE INDEX IF NOT EXISTS idx_conversations_client_deleted_last_modified ON conversations(client_id, "
    "deleted, last_modified DESC)",
    "CREATE INDEX IF NOT EXISTS idx_conversations_client_character_deleted_last_modified ON "
    "conversations(client_id, character_id, deleted, last_modified DESC)",
    "CREATE INDEX IF NOT EXISTS idx_conversations_client_deleted_created_at ON conversations(client_id, "
    "deleted, created_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_notes_conversation ON notes(conversation_id)",
    "CREATE INDEX IF NOT EXISTS idx_notes_message ON notes(message_id)",
    "CREATE INDEX IF NOT EXISTS idx_writing_sessions_last_modified ON writing_sessions(last_modified)",
    "CREATE INDEX IF NOT EXISTS idx_writing_sessions_deleted ON writing_sessions(deleted)",
    "CREATE INDEX IF NOT EXISTS idx_writing_templates_last_modified ON writing_templates(last_modified)",
    "CREATE INDEX IF NOT EXISTS idx_writing_templates_deleted ON writing_templates(deleted)",
    "CREATE INDEX IF NOT EXISTS idx_writing_themes_last_modified ON writing_themes(last_modified)",
    "CREATE INDEX IF NOT EXISTS idx_writing_themes_deleted ON writing_themes(deleted)",
    "CREATE INDEX IF NOT EXISTS idx_writing_themes_order ON writing_themes(order_index)",
    "CREATE INDEX IF NOT EXISTS idx_writing_wordclouds_status ON writing_wordclouds(status)",
    "CREATE INDEX IF NOT EXISTS idx_writing_wordclouds_last_modified ON writing_wordclouds(last_modified)",
    "CREATE INDEX IF NOT EXISTS idx_character_exemplars_character ON character_exemplars(character_id)",
    "CREATE INDEX IF NOT EXISTS idx_character_exemplars_scenario_emotion ON character_exemplars(scenario, emotion)",
    "CREATE INDEX IF NOT EXISTS idx_character_exemplars_novelty ON character_exemplars(novelty_hint)",
)


def require_moodboard_migration_complete(progress: dict) -> None:
    """Require the already-read owner journal to publish completion."""
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import SchemaError

    if progress["status"] != "complete":
        raise SchemaError("Notes moodboard/Studio v61 PostgreSQL migration progress is incomplete.")  # noqa: TRY003


MOODBOARD_STUDIO_V61_INDEX_STATEMENTS = (
    "CREATE INDEX IF NOT EXISTS idx_moodboards_scope_page ON moodboards(owner_user_id,dataset_id,deleted,last_modified,id)",
    "CREATE INDEX IF NOT EXISTS idx_moodboards_scope_sync_id ON moodboards(owner_user_id,dataset_id,sync_id)",
    "CREATE INDEX IF NOT EXISTS idx_moodboard_notes_scope_board_page ON moodboard_notes(owner_user_id,dataset_id,moodboard_id,deleted,order_index,placement_id)",
    "CREATE INDEX IF NOT EXISTS idx_moodboard_notes_scope_note ON moodboard_notes(owner_user_id,dataset_id,note_id,deleted,moodboard_id)",
    "CREATE INDEX IF NOT EXISTS idx_moodboard_notes_scope_placement ON moodboard_notes(owner_user_id,dataset_id,placement_id)",
    "CREATE INDEX IF NOT EXISTS idx_note_studio_documents_scope_page ON note_studio_documents(owner_user_id,dataset_id,deleted,last_modified,note_id)",
    "CREATE INDEX IF NOT EXISTS idx_note_studio_documents_scope_note ON note_studio_documents(owner_user_id,dataset_id,note_id)",
    "CREATE INDEX IF NOT EXISTS idx_note_studio_documents_scope_source ON note_studio_documents(owner_user_id,dataset_id,source_note_id)",
)
