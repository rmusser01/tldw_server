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
            AND system_operation_kind = 'shared_workspace_clone'
            AND system_source_identity IS NOT NULL
            AND system_content_hash IS NOT NULL
            AND length(system_content_hash) = 64
            AND system_content_hash = lower(system_content_hash)
            AND system_content_hash NOT GLOB '*[^0-9a-f]*'
        )
    );

CREATE UNIQUE INDEX IF NOT EXISTS ux_media_system_operation_source
    ON Media(system_operation_kind, system_operation_id, system_source_identity)
    WHERE system_operation_id IS NOT NULL;

UPDATE schema_version SET version = 25;

COMMIT;
PRAGMA foreign_keys = ON;
