from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

Sensitivity = Literal["public", "personal", "sensitive", "secret"]
RestoreStatus = Literal["restorable", "pointer_only", "non_restorable"]


@dataclass(frozen=True)
class AccountInventoryEntry:
    category: str
    label: str
    source: str
    export_representation: str
    manifest_count_key: str
    import_handler_key: str
    dependencies: tuple[str, ...]
    sensitivity: Sensitivity
    restore_status: RestoreStatus
    warning: str | None = None

    def to_summary(self) -> dict[str, object]:
        return asdict(self)


ACCOUNT_DATA_INVENTORY: tuple[AccountInventoryEntry, ...] = (
    AccountInventoryEntry(
        category="account_profile",
        label="Account profile",
        source="AuthNZ users database: user account/profile rows",
        export_representation="json/account_profile.json",
        manifest_count_key="account_profiles",
        import_handler_key="restore_account_profile",
        dependencies=(),
        sensitivity="sensitive",
        restore_status="restorable",
    ),
    AccountInventoryEntry(
        category="account_settings",
        label="Account settings",
        source="per-user settings tables and Chatbooks export options",
        export_representation="json/account_settings.json",
        manifest_count_key="account_settings",
        import_handler_key="restore_account_settings",
        dependencies=("account_profile",),
        sensitivity="personal",
        restore_status="restorable",
    ),
    AccountInventoryEntry(
        category="conversations",
        label="Conversations",
        source="ChaChaNotes DB: conversations and messages",
        export_representation="content/conversations/*.json",
        manifest_count_key="conversations",
        import_handler_key="restore_conversations",
        dependencies=("characters", "media_pointers"),
        sensitivity="personal",
        restore_status="restorable",
    ),
    AccountInventoryEntry(
        category="notes",
        label="Notes",
        source="ChaChaNotes DB: notes",
        export_representation="content/notes/*.md",
        manifest_count_key="notes",
        import_handler_key="restore_notes",
        dependencies=("tags_categories_relationships",),
        sensitivity="personal",
        restore_status="restorable",
    ),
    AccountInventoryEntry(
        category="characters",
        label="Characters",
        source="ChaChaNotes DB: character_cards",
        export_representation="content/characters/*.json",
        manifest_count_key="characters",
        import_handler_key="restore_characters",
        dependencies=("world_books", "dictionaries"),
        sensitivity="personal",
        restore_status="restorable",
    ),
    AccountInventoryEntry(
        category="world_books",
        label="World books",
        source="ChaChaNotes DB: world_books and entries",
        export_representation="content/world_books/*.json",
        manifest_count_key="world_books",
        import_handler_key="restore_world_books",
        dependencies=("characters",),
        sensitivity="personal",
        restore_status="restorable",
    ),
    AccountInventoryEntry(
        category="dictionaries",
        label="Dictionaries",
        source="ChaChaNotes DB: chat_dictionaries and entries",
        export_representation="content/dictionaries/*.json",
        manifest_count_key="dictionaries",
        import_handler_key="restore_dictionaries",
        dependencies=("characters",),
        sensitivity="personal",
        restore_status="restorable",
    ),
    AccountInventoryEntry(
        category="prompts",
        label="Prompts",
        source="Prompts DB: prompts and prompt metadata",
        export_representation="content/prompts/*.json",
        manifest_count_key="prompts",
        import_handler_key="restore_prompts",
        dependencies=("tags_categories_relationships",),
        sensitivity="personal",
        restore_status="restorable",
    ),
    AccountInventoryEntry(
        category="evaluations",
        label="Evaluations",
        source="Evaluations DB: evaluations, runs, datasets, and recipes",
        export_representation="content/evaluations/*.json",
        manifest_count_key="evaluations",
        import_handler_key="restore_evaluations",
        dependencies=("prompts", "media_records"),
        sensitivity="personal",
        restore_status="restorable",
    ),
    AccountInventoryEntry(
        category="generated_documents",
        label="Generated documents",
        source="ChaChaNotes DB: generated_documents",
        export_representation="content/generated_documents/*.json",
        manifest_count_key="generated_documents",
        import_handler_key="restore_generated_documents",
        dependencies=("conversations",),
        sensitivity="personal",
        restore_status="restorable",
    ),
    AccountInventoryEntry(
        category="explainer_sessions",
        label="Explainer sessions",
        source="Explainer DB: sessions, nodes, selected sources, and citations",
        export_representation="content/explainer_sessions/*.json",
        manifest_count_key="explainer_sessions",
        import_handler_key="restore_explainer_sessions",
        dependencies=("media_records", "notes"),
        sensitivity="personal",
        restore_status="restorable",
    ),
    AccountInventoryEntry(
        category="media_records",
        label="Media records",
        source="Media DB: Media rows and metadata",
        export_representation="content/media/*.json",
        manifest_count_key="media_records",
        import_handler_key="restore_media_records",
        dependencies=("media_pointers",),
        sensitivity="personal",
        restore_status="restorable",
    ),
    AccountInventoryEntry(
        category="media_transcripts",
        label="Media transcripts",
        source="Media DB: Transcripts rows",
        export_representation="content/media/transcripts/*.json",
        manifest_count_key="media_transcripts",
        import_handler_key="restore_media_transcripts",
        dependencies=("media_records",),
        sensitivity="personal",
        restore_status="restorable",
    ),
    AccountInventoryEntry(
        category="media_chunks",
        label="Media chunks",
        source="Media DB: UnvectorizedMediaChunks and derived chunk rows",
        export_representation="content/media/chunks/*.json",
        manifest_count_key="media_chunks",
        import_handler_key="restore_media_chunks",
        dependencies=("media_records", "media_transcripts"),
        sensitivity="personal",
        restore_status="restorable",
    ),
    AccountInventoryEntry(
        category="media_stored_artifacts",
        label="Stored media artifacts",
        source="Media DB: MediaFiles and stored account artifact directories",
        export_representation="content/media/files/*",
        manifest_count_key="media_stored_artifacts",
        import_handler_key="restore_media_stored_artifacts",
        dependencies=("media_records",),
        sensitivity="personal",
        restore_status="restorable",
    ),
    AccountInventoryEntry(
        category="media_pointers",
        label="Media source references",
        source="Media DB: external URLs, local source paths, and provenance pointers",
        export_representation="json/media_pointers.json",
        manifest_count_key="media_pointers",
        import_handler_key="restore_media_pointers",
        dependencies=("media_records",),
        sensitivity="personal",
        restore_status="pointer_only",
        warning="External media URLs and local paths restore as references only; unavailable source bytes are not recreated.",
    ),
    AccountInventoryEntry(
        category="embeddings",
        label="Embeddings",
        source="ChromaDB collections and media vector fields",
        export_representation="content/embeddings/*.json",
        manifest_count_key="embeddings",
        import_handler_key="restore_embeddings",
        dependencies=("media_records", "media_chunks"),
        sensitivity="personal",
        restore_status="restorable",
    ),
    AccountInventoryEntry(
        category="tags_categories_relationships",
        label="Tags, categories, and relationships",
        source="ChaChaNotes, Media, and Prompt relationship/tag tables",
        export_representation="json/relationships.json",
        manifest_count_key="tags_categories_relationships",
        import_handler_key="restore_tags_categories_relationships",
        dependencies=("conversations", "notes", "media_records", "prompts"),
        sensitivity="personal",
        restore_status="non_restorable",
        warning=(
            "Tag/category relationship tables are not serialized in this archive version; "
            "restored content may need tags or relationships rebuilt."
        ),
    ),
    AccountInventoryEntry(
        category="sensitive_user_values",
        label="Sensitive user values",
        source="AuthNZ, provider credential stores, tokens, and deployment-local secrets",
        export_representation="json/sensitive_user_values.redacted.json",
        manifest_count_key="sensitive_user_values",
        import_handler_key="restore_sensitive_user_values",
        dependencies=("account_profile", "account_settings"),
        sensitivity="secret",
        restore_status="non_restorable",
        warning="Secret values and deployment-local credentials are not shown in summaries and may require reconfiguration after restore.",
    ),
)
