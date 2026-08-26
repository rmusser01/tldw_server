-- version: 25
-- description: Add complete operation ownership markers for shared Workspace clone Media
-- idempotent: true

PRAGMA foreign_keys = OFF;
BEGIN TRANSACTION;

ALTER TABLE Media ADD COLUMN system_operation_id TEXT;
ALTER TABLE Media ADD COLUMN system_operation_kind TEXT;
ALTER TABLE Media ADD COLUMN system_source_identity TEXT;
ALTER TABLE Media ADD COLUMN system_content_hash TEXT
    CONSTRAINT ck_media_system_operation_ownership CHECK (
        (
            system_operation_id IS NULL
            AND system_operation_kind IS NULL
            AND system_source_identity IS NULL
            AND system_content_hash IS NULL
        )
        OR
        (
            system_operation_id IS NOT NULL
            AND length(system_operation_id) BETWEEN 1 AND 255
            AND system_operation_kind IS NOT NULL
            AND system_operation_kind = 'shared_workspace_clone'
            AND system_source_identity IS NOT NULL
            AND length(system_source_identity) BETWEEN 1 AND 255
            AND system_content_hash IS NOT NULL
            AND length(system_content_hash) = 64
            AND system_content_hash = lower(system_content_hash)
            AND system_content_hash NOT GLOB '*[^0-9a-f]*'
        )
    );

CREATE UNIQUE INDEX IF NOT EXISTS ux_media_system_operation_source
    ON Media(system_operation_kind, system_operation_id, system_source_identity)
    WHERE system_operation_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS OperationOwnedCloneKeywords (
    media_id INTEGER NOT NULL,
    keyword_id INTEGER NOT NULL,
    operation_id TEXT NOT NULL CHECK (length(operation_id) BETWEEN 1 AND 255),
    source_identity TEXT NOT NULL CHECK (length(source_identity) BETWEEN 1 AND 255),
    created_by_clone BOOLEAN NOT NULL,
    PRIMARY KEY (media_id, keyword_id),
    FOREIGN KEY (media_id) REFERENCES Media(id) ON DELETE CASCADE,
    FOREIGN KEY (keyword_id) REFERENCES Keywords(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_owned_clone_keywords_keyword
    ON OperationOwnedCloneKeywords(keyword_id);
CREATE INDEX IF NOT EXISTS idx_owned_clone_keywords_operation
    ON OperationOwnedCloneKeywords(operation_id, source_identity);

UPDATE schema_version SET version = 25;

COMMIT;
PRAGMA foreign_keys = ON;
