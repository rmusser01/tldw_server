"""Versioned ChaCha storage for native fork receipts and retained assets."""

from __future__ import annotations

NATIVE_CHAT_TABLES = (
    "native_chat_operations",
    "native_chat_asset_candidates",
    "native_chat_asset_claims",
    "native_chat_asset_references",
    "native_chat_quota_intents",
)


def native_fork_schema_statements(*, postgres: bool) -> tuple[str, ...]:
    """Return migration DDL without touching data or opening a transaction."""
    boolean = "BOOLEAN" if postgres else "INTEGER"
    false = "FALSE" if postgres else "0"
    return (
        f"ALTER TABLE workspaces ADD COLUMN native_chat_admission_closed {boolean} NOT NULL DEFAULT {false}",
        "ALTER TABLE conversations ADD COLUMN required_projection_version TEXT",
        "ALTER TABLE conversations ADD COLUMN native_creation_operation_kind TEXT",
        "ALTER TABLE conversations ADD COLUMN native_creation_operation_id TEXT",
        "ALTER TABLE conversations ADD COLUMN native_bundle_json TEXT",
        """
        CREATE TABLE native_chat_operations (
            client_id TEXT NOT NULL,
            operation_kind TEXT NOT NULL CHECK (operation_kind IN ('native_fork_v1', 'native_asset_retention_v1')),
            operation_id TEXT NOT NULL,
            owner_key TEXT NOT NULL,
            scope_type TEXT NOT NULL CHECK (scope_type IN ('global', 'workspace')),
            workspace_id TEXT,
            request_digest TEXT NOT NULL,
            canonical_request_json TEXT,
            projection_version TEXT NOT NULL,
            state TEXT NOT NULL CHECK (state IN ('preparing', 'committed', 'rejected', 'expired', 'gone')),
            attempt_generation INTEGER NOT NULL DEFAULT 0 CHECK (attempt_generation >= 0),
            lease_expires_at TEXT,
            source_conversation_id TEXT,
            child_conversation_id TEXT,
            result_json TEXT,
            terminal_reason TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (client_id, operation_kind, operation_id),
            CHECK ((scope_type = 'workspace' AND workspace_id IS NOT NULL)
                OR (scope_type = 'global' AND workspace_id IS NULL))
        )
        """,
        "CREATE INDEX native_chat_operations_child ON native_chat_operations(client_id, child_conversation_id)",
        """
        CREATE TABLE native_chat_asset_candidates (
            client_id TEXT NOT NULL,
            candidate_id TEXT NOT NULL,
            operation_kind TEXT NOT NULL,
            operation_id TEXT NOT NULL,
            owner_key TEXT NOT NULL,
            attempt_generation INTEGER NOT NULL CHECK (attempt_generation >= 1),
            storage_namespace_id TEXT NOT NULL,
            upload_id TEXT NOT NULL,
            expected_hash TEXT NOT NULL,
            expected_size_bytes INTEGER NOT NULL CHECK (expected_size_bytes >= 0),
            mime_type TEXT NOT NULL,
            representation TEXT NOT NULL,
            storage_key TEXT,
            state TEXT NOT NULL CHECK (state IN ('reserved', 'prepared', 'adopted', 'reclaiming', 'discarded')),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (client_id, candidate_id),
            FOREIGN KEY (client_id, operation_kind, operation_id)
                REFERENCES native_chat_operations(client_id, operation_kind, operation_id)
        )
        """,
        "CREATE INDEX native_chat_candidates_operation ON native_chat_asset_candidates(client_id, operation_kind, operation_id)",
        """
        CREATE TABLE native_chat_asset_claims (
            client_id TEXT NOT NULL,
            claim_id TEXT NOT NULL,
            owner_key TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            candidate_id TEXT,
            asset_id TEXT NOT NULL,
            asset_revision INTEGER NOT NULL CHECK (asset_revision >= 1),
            content_hash TEXT NOT NULL,
            size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
            mime_type TEXT NOT NULL,
            representation TEXT NOT NULL,
            state TEXT NOT NULL CHECK (state IN ('live', 'released')),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (client_id, claim_id),
            FOREIGN KEY (client_id, candidate_id)
                REFERENCES native_chat_asset_candidates(client_id, candidate_id)
        )
        """,
        "CREATE INDEX native_chat_claims_conversation ON native_chat_asset_claims(client_id, conversation_id, state)",
        f"""
        CREATE TABLE native_chat_asset_references (
            client_id TEXT NOT NULL,
            reference_id TEXT NOT NULL,
            owner_key TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            claim_id TEXT NOT NULL,
            position INTEGER NOT NULL CHECK (position >= 0),
            revision INTEGER NOT NULL CHECK (revision >= 1),
            context_enabled {boolean} NOT NULL DEFAULT {false},
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (client_id, reference_id),
            FOREIGN KEY (client_id, claim_id)
                REFERENCES native_chat_asset_claims(client_id, claim_id)
        )
        """,
        "CREATE INDEX native_chat_refs_conversation ON native_chat_asset_references(client_id, conversation_id, position)",
        """
        CREATE TABLE native_chat_quota_intents (
            client_id TEXT NOT NULL,
            candidate_id TEXT NOT NULL,
            intent_kind TEXT NOT NULL CHECK (intent_kind IN ('reserve', 'release')),
            owner_key TEXT NOT NULL,
            size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
            state TEXT NOT NULL CHECK (state IN ('pending', 'confirmed')),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (client_id, candidate_id, intent_kind),
            FOREIGN KEY (client_id, candidate_id)
                REFERENCES native_chat_asset_candidates(client_id, candidate_id)
        )
        """,
    )
