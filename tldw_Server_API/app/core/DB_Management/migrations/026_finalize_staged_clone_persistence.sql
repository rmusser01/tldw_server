-- version: 26
-- description: Finalize owner-scoped staged clone persistence
-- idempotent: false

PRAGMA foreign_keys = OFF;
BEGIN TRANSACTION;

UPDATE Media
   SET system_operation_id = NULL,
       system_operation_kind = NULL,
       system_source_identity = NULL,
       system_content_hash = NULL
 WHERE NOT COALESCE(
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
    ),
    0
 );

CREATE TABLE IF NOT EXISTS MediaKeywords (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    media_id INTEGER NOT NULL,
    keyword_id INTEGER NOT NULL,
    UNIQUE (media_id, keyword_id),
    FOREIGN KEY (media_id) REFERENCES Media(id) ON DELETE CASCADE,
    FOREIGN KEY (keyword_id) REFERENCES Keywords(id) ON DELETE CASCADE
);

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
   AND holds.operation_id = media.system_operation_id
   AND holds.source_identity = media.system_source_identity
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

INSERT OR IGNORE INTO OperationOwnedCloneKeywords (
    media_id, keyword, operation_id, source_identity, client_id
)
SELECT links.media_id,
       lower(trim(keywords.keyword)),
       media.system_operation_id,
       media.system_source_identity,
       media.client_id
  FROM MediaKeywords AS links
  JOIN Keywords AS keywords ON keywords.id = links.keyword_id
  JOIN Media AS media ON media.id = links.media_id
 WHERE media.system_operation_id IS NOT NULL
   AND length(media.system_operation_id) BETWEEN 1 AND 255
   AND media.system_operation_kind = 'shared_workspace_clone'
   AND media.system_source_identity IS NOT NULL
   AND length(media.system_source_identity) BETWEEN 1 AND 255
   AND media.system_content_hash IS NOT NULL
   AND length(media.system_content_hash) = 64
   AND media.system_content_hash = lower(media.system_content_hash)
   AND media.system_content_hash NOT GLOB '*[^0-9a-f]*'
   AND length(trim(keywords.keyword)) BETWEEN 1 AND 255;

CREATE TABLE OperationOwnedCloneKeywords_v26_coverage (
    complete INTEGER NOT NULL CHECK (complete = 1)
);

INSERT INTO OperationOwnedCloneKeywords_v26_coverage (complete)
SELECT CASE WHEN EXISTS (
    SELECT 1
      FROM MediaKeywords AS links
      JOIN Media AS media ON media.id = links.media_id
      LEFT JOIN Keywords AS keywords ON keywords.id = links.keyword_id
     WHERE media.system_operation_id IS NOT NULL
       AND length(media.system_operation_id) BETWEEN 1 AND 255
       AND media.system_operation_kind = 'shared_workspace_clone'
       AND media.system_source_identity IS NOT NULL
       AND length(media.system_source_identity) BETWEEN 1 AND 255
       AND media.system_content_hash IS NOT NULL
       AND length(media.system_content_hash) = 64
       AND media.system_content_hash = lower(media.system_content_hash)
       AND media.system_content_hash NOT GLOB '*[^0-9a-f]*'
       AND (
            keywords.id IS NULL
            OR length(trim(keywords.keyword)) NOT BETWEEN 1 AND 255
            OR NOT EXISTS (
                SELECT 1
                  FROM OperationOwnedCloneKeywords AS pending
                 WHERE pending.media_id = links.media_id
                   AND pending.keyword = lower(trim(keywords.keyword))
                   AND pending.operation_id = media.system_operation_id
                   AND pending.source_identity = media.system_source_identity
                   AND pending.client_id = media.client_id
            )
       )
) THEN 0 ELSE 1 END;

INSERT INTO OperationOwnedCloneKeywords_v26_coverage (complete)
SELECT CASE WHEN EXISTS (
    SELECT 1
      FROM OperationOwnedCloneKeywords_v26_source AS holds
      JOIN Media AS media ON media.id = holds.media_id
      LEFT JOIN Keywords AS keywords ON keywords.id = holds.keyword_value
     WHERE EXISTS (
            SELECT 1
              FROM pragma_table_info('OperationOwnedCloneKeywords_v25')
             WHERE name = 'keyword_id'
       )
       AND media.system_operation_id IS NOT NULL
       AND length(media.system_operation_id) BETWEEN 1 AND 255
       AND media.system_operation_kind = 'shared_workspace_clone'
       AND media.system_source_identity IS NOT NULL
       AND length(media.system_source_identity) BETWEEN 1 AND 255
       AND media.system_content_hash IS NOT NULL
       AND length(media.system_content_hash) = 64
       AND media.system_content_hash = lower(media.system_content_hash)
       AND media.system_content_hash NOT GLOB '*[^0-9a-f]*'
       AND holds.operation_id = media.system_operation_id
       AND holds.source_identity = media.system_source_identity
       AND (
            keywords.id IS NULL
            OR length(trim(keywords.keyword)) NOT BETWEEN 1 AND 255
            OR NOT EXISTS (
                SELECT 1
                  FROM OperationOwnedCloneKeywords AS pending
                 WHERE pending.media_id = holds.media_id
                   AND pending.keyword = lower(trim(keywords.keyword))
                   AND pending.operation_id = media.system_operation_id
                   AND pending.source_identity = media.system_source_identity
                   AND pending.client_id = media.client_id
            )
       )
) THEN 0 ELSE 1 END;

DELETE FROM MediaKeywords
 WHERE EXISTS (
        SELECT 1 FROM Media AS media
         WHERE media.id = MediaKeywords.media_id
           AND media.system_operation_id IS NOT NULL
           AND length(media.system_operation_id) BETWEEN 1 AND 255
           AND media.system_operation_kind = 'shared_workspace_clone'
           AND media.system_source_identity IS NOT NULL
           AND length(media.system_source_identity) BETWEEN 1 AND 255
           AND media.system_content_hash IS NOT NULL
           AND length(media.system_content_hash) = 64
           AND media.system_content_hash = lower(media.system_content_hash)
           AND media.system_content_hash NOT GLOB '*[^0-9a-f]*'
   );

DELETE FROM Keywords
 WHERE NOT EXISTS (
        SELECT 1 FROM MediaKeywords WHERE MediaKeywords.keyword_id = Keywords.id
   )
   AND EXISTS (
        SELECT 1
          FROM OperationOwnedCloneKeywords_v26_source AS holds
          JOIN Media AS media ON media.id = holds.media_id
         WHERE EXISTS (
                SELECT 1
                  FROM pragma_table_info('OperationOwnedCloneKeywords_v25')
                 WHERE name = 'keyword_id'
           )
           AND media.system_operation_kind = 'shared_workspace_clone'
           AND length(media.system_operation_id) BETWEEN 1 AND 255
           AND length(media.system_source_identity) BETWEEN 1 AND 255
           AND length(media.system_content_hash) = 64
           AND media.system_content_hash = lower(media.system_content_hash)
           AND media.system_content_hash NOT GLOB '*[^0-9a-f]*'
           AND holds.operation_id = media.system_operation_id
           AND holds.source_identity = media.system_source_identity
           AND holds.client_value = 1
           AND holds.keyword_value = Keywords.id
   );

INSERT INTO OperationOwnedCloneKeywords_v26_coverage (complete)
SELECT CASE WHEN EXISTS (
    SELECT 1
      FROM MediaKeywords AS links
      JOIN Media AS media ON media.id = links.media_id
     WHERE media.system_operation_id IS NOT NULL
       AND length(media.system_operation_id) BETWEEN 1 AND 255
       AND media.system_operation_kind = 'shared_workspace_clone'
       AND media.system_source_identity IS NOT NULL
       AND length(media.system_source_identity) BETWEEN 1 AND 255
       AND media.system_content_hash IS NOT NULL
       AND length(media.system_content_hash) = 64
       AND media.system_content_hash = lower(media.system_content_hash)
       AND media.system_content_hash NOT GLOB '*[^0-9a-f]*'
) THEN 0 ELSE 1 END;

DROP VIEW OperationOwnedCloneKeywords_v26_source;

DROP TABLE OperationOwnedCloneKeywords_v25;

DROP TABLE OperationOwnedCloneKeywords_v26_coverage;

CREATE INDEX idx_owned_clone_keywords_keyword
    ON OperationOwnedCloneKeywords(keyword);
CREATE INDEX idx_owned_clone_keywords_operation
    ON OperationOwnedCloneKeywords(operation_id, source_identity);

DROP TRIGGER IF EXISTS media_validate_system_operation_insert_v26;
DROP TRIGGER IF EXISTS media_validate_system_operation_update_v26;

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
