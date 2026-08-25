-- version: 26
-- description: Finalize owner-scoped staged clone persistence
-- idempotent: false

PRAGMA foreign_keys = OFF;
BEGIN TRANSACTION;

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

ALTER TABLE OperationOwnedCloneKeywords
    RENAME TO OperationOwnedCloneKeywords_v25;

CREATE TABLE OperationOwnedCloneKeywords (
    media_id INTEGER NOT NULL,
    keyword TEXT NOT NULL CHECK (
        length(keyword) BETWEEN 1 AND 255
        AND keyword = lower(trim(keyword))
    ),
    operation_id TEXT NOT NULL CHECK (length(operation_id) BETWEEN 1 AND 255),
    source_identity TEXT NOT NULL CHECK (length(source_identity) BETWEEN 1 AND 255),
    client_id TEXT NOT NULL CHECK (length(client_id) BETWEEN 1 AND 255),
    PRIMARY KEY (media_id, keyword),
    FOREIGN KEY (media_id) REFERENCES Media(id) ON DELETE CASCADE
);

CREATE VIEW OperationOwnedCloneKeywords_v26_source (
    media_id, keyword_value, operation_id, source_identity, client_value
) AS
SELECT * FROM OperationOwnedCloneKeywords_v25;

INSERT OR IGNORE INTO OperationOwnedCloneKeywords (
    media_id, keyword, operation_id, source_identity, client_id
)
SELECT holds.media_id,
       lower(trim(keywords.keyword)),
       media.system_operation_id,
       media.system_source_identity,
       media.client_id
  FROM OperationOwnedCloneKeywords_v26_source AS holds
  JOIN Keywords AS keywords ON keywords.id = holds.keyword_value
  JOIN Media AS media ON media.id = holds.media_id
 WHERE EXISTS (
        SELECT 1
          FROM pragma_table_info('OperationOwnedCloneKeywords_v25')
         WHERE name = 'keyword_id'
   )
   AND media.system_operation_kind = 'shared_workspace_clone'
   AND length(media.system_operation_id) BETWEEN 1 AND 255
   AND length(media.system_source_identity) BETWEEN 1 AND 255
   AND length(trim(keywords.keyword)) BETWEEN 1 AND 255;

INSERT OR IGNORE INTO OperationOwnedCloneKeywords (
    media_id, keyword, operation_id, source_identity, client_id
)
SELECT media_id, keyword_value, operation_id, source_identity, client_value
  FROM OperationOwnedCloneKeywords_v26_source
 WHERE NOT EXISTS (
        SELECT 1
          FROM pragma_table_info('OperationOwnedCloneKeywords_v25')
         WHERE name = 'keyword_id'
   );

DROP VIEW OperationOwnedCloneKeywords_v26_source;

DROP TABLE OperationOwnedCloneKeywords_v25;

CREATE INDEX idx_owned_clone_keywords_keyword
    ON OperationOwnedCloneKeywords(keyword);
CREATE INDEX idx_owned_clone_keywords_operation
    ON OperationOwnedCloneKeywords(operation_id, source_identity);

UPDATE Media
   SET system_operation_id = NULL,
       system_operation_kind = NULL,
       system_source_identity = NULL,
       system_content_hash = NULL
 WHERE NOT (
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

CREATE TRIGGER media_validate_system_operation_insert_v26
BEFORE INSERT ON Media
WHEN NOT COALESCE(
    (
        NEW.system_operation_id IS NULL
        AND NEW.system_operation_kind IS NULL
        AND NEW.system_source_identity IS NULL
        AND NEW.system_content_hash IS NULL
    )
    OR
    (
        NEW.system_operation_id IS NOT NULL
        AND length(NEW.system_operation_id) BETWEEN 1 AND 255
        AND NEW.system_operation_kind IS NOT NULL
        AND NEW.system_operation_kind = 'shared_workspace_clone'
        AND NEW.system_source_identity IS NOT NULL
        AND length(NEW.system_source_identity) BETWEEN 1 AND 255
        AND NEW.system_content_hash IS NOT NULL
        AND length(NEW.system_content_hash) = 64
        AND NEW.system_content_hash = lower(NEW.system_content_hash)
        AND NEW.system_content_hash NOT GLOB '*[^0-9a-f]*'
    ),
    0
)
BEGIN
    SELECT RAISE(ABORT, 'invalid Media system operation ownership markers');
END;

CREATE TRIGGER media_validate_system_operation_update_v26
BEFORE UPDATE OF system_operation_id, system_operation_kind,
    system_source_identity, system_content_hash ON Media
WHEN NOT COALESCE(
    (
        NEW.system_operation_id IS NULL
        AND NEW.system_operation_kind IS NULL
        AND NEW.system_source_identity IS NULL
        AND NEW.system_content_hash IS NULL
    )
    OR
    (
        NEW.system_operation_id IS NOT NULL
        AND length(NEW.system_operation_id) BETWEEN 1 AND 255
        AND NEW.system_operation_kind IS NOT NULL
        AND NEW.system_operation_kind = 'shared_workspace_clone'
        AND NEW.system_source_identity IS NOT NULL
        AND length(NEW.system_source_identity) BETWEEN 1 AND 255
        AND NEW.system_content_hash IS NOT NULL
        AND length(NEW.system_content_hash) = 64
        AND NEW.system_content_hash = lower(NEW.system_content_hash)
        AND NEW.system_content_hash NOT GLOB '*[^0-9a-f]*'
    ),
    0
)
BEGIN
    SELECT RAISE(ABORT, 'invalid Media system operation ownership markers');
END;

UPDATE schema_version SET version = 26;

COMMIT;
PRAGMA foreign_keys = ON;
