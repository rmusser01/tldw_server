
"""

    UUID Columns: Added uuid TEXT UNIQUE NOT NULL to main entity tables. Your application code must generate these UUIDs (e.g., uuid.uuid4()) before inserting new records.

    Sync Metadata: Added last_modified, version, client_id, deleted to all synced entity tables. Your application code must manage these correctly (increment version, update timestamp/client_id on changes).

    Defaults: Used DEFAULT values for sync columns primarily to handle potential ALTER TABLE operations on existing schemas without causing NOT NULL constraint errors immediately. For a greenfield project, you might omit the defaults and rely entirely on your application code setting these values on INSERT.

    sync_log Table: Created as specified, with indices.

    Triggers:

        Included triggers for Media, Keywords, DocumentVersions, and the MediaKeywords relationship.

        You MUST add similar triggers for Transcripts, MediaChunks, and UnvectorizedMediaChunks based on the patterns shown, adjusting the json_object payload to include their specific columns and potentially the media_uuid of their parent Media item.

        The UPDATE triggers check OLD.deleted = NEW.deleted to differentiate data updates from soft delete/undelete operations.

        The UPDATE triggers use ifnull() comparisons to handle NULL values correctly.

        The UPDATE triggers check for changes in last_modified or version as well, ensuring that even operations that only bump metadata (like a forced conflict resolution) get logged.

        The DELETE triggers fire only when deleted goes from 0 to 1 (soft delete).

        MediaKeywords triggers log custom 'link'/'unlink' operations and construct a synthetic UUID for the log entry (media_uuid || '_' || keyword_uuid). They fetch necessary info (UUIDs, client_id, version) from parent tables.

    JSON Payload: The json_object in the triggers defines what data gets sent during sync. Carefully review and include all necessary fields for each entity. Exclude fields that are purely local state (like chunking_status maybe) or too large/volatile (like vector_embedding).

    FTS Tables: Kept the original FTS tables and their triggers as they depend on the main tables' id and content. They don't need direct sync metadata.

    Application Responsibility: Remember, this schema enables sync. Your Python application logic is responsible for:

        Generating UUIDs.

        Managing version, last_modified, client_id, deleted columns correctly before SQL operations.

        Implementing the actual sync communication (API calls) and conflict resolution logic (apply_changes_locally, handle_conflict, etc.).
"""


media_db_schema = """
-- Enable Foreign Key support
PRAGMA foreign_keys = ON;

-- ───────────────────────────────────────────────────────────────────────────
-- Core Data Tables with Sync Metadata
-- ───────────────────────────────────────────────────────────────────────────

-- Media Table (Central Entity)
-- ============================
CREATE TABLE IF NOT EXISTS Media (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT UNIQUE,
    title TEXT NOT NULL,
    type TEXT NOT NULL,
    content TEXT,
    author TEXT,
    ingestion_date TEXT,
    transcription_model TEXT,
    is_trash BOOLEAN DEFAULT 0 NOT NULL, -- Keep for application logic (optional if 'deleted' is used exclusively)
    trash_date DATETIME,
    vector_embedding BLOB, -- Often excluded from sync payload due to size/volatility
    chunking_status TEXT DEFAULT 'pending' NOT NULL, -- Likely managed locally, maybe exclude from sync
    vector_processing INTEGER DEFAULT 0 NOT NULL,    -- Likely managed locally, maybe exclude from sync
    content_hash TEXT UNIQUE NOT NULL,

    -- Sync Metadata Columns --
    uuid TEXT UNIQUE NOT NULL, -- Globally unique identifier for sync
    last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    version INTEGER NOT NULL DEFAULT 1,
    client_id TEXT NOT NULL, -- Identifier of the client that made the last change
    deleted BOOLEAN NOT NULL DEFAULT 0 -- Soft delete flag for sync
);
-- Original Indices
CREATE INDEX IF NOT EXISTS idx_media_title ON Media(title);
CREATE INDEX IF NOT EXISTS idx_media_type ON Media(type);
CREATE INDEX IF NOT EXISTS idx_media_author ON Media(author);
CREATE INDEX IF NOT EXISTS idx_media_ingestion_date ON Media(ingestion_date);
CREATE INDEX IF NOT EXISTS idx_media_chunking_status ON Media(chunking_status);
CREATE INDEX IF NOT EXISTS idx_media_vector_processing ON Media(vector_processing);
CREATE INDEX IF NOT EXISTS idx_media_is_trash ON Media(is_trash);
CREATE UNIQUE INDEX IF NOT EXISTS idx_media_content_hash ON Media(content_hash);
-- Sync Indices
CREATE UNIQUE INDEX IF NOT EXISTS idx_media_uuid ON Media(uuid);
CREATE INDEX IF NOT EXISTS idx_media_last_modified ON Media(last_modified);
CREATE INDEX IF NOT EXISTS idx_media_deleted ON Media(deleted);


-- Keywords Table
-- ==============
CREATE TABLE IF NOT EXISTS Keywords (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    keyword TEXT NOT NULL UNIQUE COLLATE NOCASE,

    -- Sync Metadata Columns --
    uuid TEXT UNIQUE NOT NULL,
    last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    version INTEGER NOT NULL DEFAULT 1,
    client_id TEXT NOT NULL,
    deleted BOOLEAN NOT NULL DEFAULT 0
);
-- Original Indices (keyword covered by UNIQUE)
-- Sync Indices
CREATE UNIQUE INDEX IF NOT EXISTS idx_keywords_uuid ON Keywords(uuid);
CREATE INDEX IF NOT EXISTS idx_keywords_last_modified ON Keywords(last_modified);
CREATE INDEX IF NOT EXISTS idx_keywords_deleted ON Keywords(deleted);


-- MediaKeywords Table (Relationship Table - No direct sync metadata needed)
-- ==========================================================================
CREATE TABLE IF NOT EXISTS MediaKeywords (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    media_id INTEGER NOT NULL,
    keyword_id INTEGER NOT NULL,
    UNIQUE (media_id, keyword_id),
    FOREIGN KEY (media_id) REFERENCES Media(id) ON DELETE CASCADE,
    FOREIGN KEY (keyword_id) REFERENCES Keywords(id) ON DELETE CASCADE
);
-- Original Indices
CREATE INDEX IF NOT EXISTS idx_mediakeywords_media_id ON MediaKeywords(media_id);
CREATE INDEX IF NOT EXISTS idx_mediakeywords_keyword_id ON MediaKeywords(keyword_id);


-- Transcripts Table
-- =================
CREATE TABLE IF NOT EXISTS Transcripts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    media_id INTEGER NOT NULL,
    whisper_model TEXT,
    transcription TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, -- Keep original creation time
    UNIQUE (media_id, whisper_model), -- Original constraint
    FOREIGN KEY (media_id) REFERENCES Media(id) ON DELETE CASCADE,

    -- Sync Metadata Columns --
    uuid TEXT UNIQUE NOT NULL,
    last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    version INTEGER NOT NULL DEFAULT 1,
    client_id TEXT NOT NULL,
    deleted BOOLEAN NOT NULL DEFAULT 0
);
-- Original Indices
CREATE INDEX IF NOT EXISTS idx_transcripts_media_id ON Transcripts(media_id);
-- Sync Indices
CREATE UNIQUE INDEX IF NOT EXISTS idx_transcripts_uuid ON Transcripts(uuid);
CREATE INDEX IF NOT EXISTS idx_transcripts_last_modified ON Transcripts(last_modified);
CREATE INDEX IF NOT EXISTS idx_transcripts_deleted ON Transcripts(deleted);


-- MediaChunks Table
-- =================
CREATE TABLE IF NOT EXISTS MediaChunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    media_id INTEGER NOT NULL,
    chunk_text TEXT,
    start_index INTEGER,
    end_index INTEGER,
    chunk_id TEXT UNIQUE, -- Original potentially local ID
    FOREIGN KEY (media_id) REFERENCES Media(id) ON DELETE CASCADE,

    -- Sync Metadata Columns --
    uuid TEXT UNIQUE NOT NULL,
    last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    version INTEGER NOT NULL DEFAULT 1,
    client_id TEXT NOT NULL,
    deleted BOOLEAN NOT NULL DEFAULT 0
);
-- Original Indices
CREATE INDEX IF NOT EXISTS idx_mediachunks_media_id ON MediaChunks(media_id);
-- Sync Indices
CREATE UNIQUE INDEX IF NOT EXISTS idx_mediachunks_uuid ON MediaChunks(uuid);
CREATE INDEX IF NOT EXISTS idx_mediachunks_last_modified ON MediaChunks(last_modified);
CREATE INDEX IF NOT EXISTS idx_mediachunks_deleted ON MediaChunks(deleted);


-- UnvectorizedMediaChunks Table
-- =============================
CREATE TABLE IF NOT EXISTS UnvectorizedMediaChunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    media_id INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    start_char INTEGER,
    end_char INTEGER,
    chunk_type TEXT,
    creation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP, -- Keep original creation time
    last_modified_orig TIMESTAMP DEFAULT CURRENT_TIMESTAMP, -- Renamed original timestamp if needed for app logic
    is_processed BOOLEAN DEFAULT FALSE NOT NULL, -- Likely local state, exclude from sync payload?
    metadata TEXT, -- Potentially sync this if it contains relevant info
    UNIQUE (media_id, chunk_index, chunk_type), -- Original constraint
    FOREIGN KEY (media_id) REFERENCES Media(id) ON DELETE CASCADE,

    -- Sync Metadata Columns --
    uuid TEXT UNIQUE NOT NULL,
    last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, -- Sync timestamp
    version INTEGER NOT NULL DEFAULT 1,
    client_id TEXT NOT NULL,
    deleted BOOLEAN NOT NULL DEFAULT 0
);
-- Original Indices
CREATE INDEX IF NOT EXISTS idx_unvectorized_media_chunks_media_id ON UnvectorizedMediaChunks(media_id);
CREATE INDEX IF NOT EXISTS idx_unvectorized_media_chunks_is_processed ON UnvectorizedMediaChunks(is_processed);
CREATE INDEX IF NOT EXISTS idx_unvectorized_media_chunks_chunk_type ON UnvectorizedMediaChunks(chunk_type);
-- Sync Indices
CREATE UNIQUE INDEX IF NOT EXISTS idx_unvectorizedmediachunks_uuid ON UnvectorizedMediaChunks(uuid);
CREATE INDEX IF NOT EXISTS idx_unvectorizedmediachunks_last_modified ON UnvectorizedMediaChunks(last_modified);
CREATE INDEX IF NOT EXISTS idx_unvectorizedmediachunks_deleted ON UnvectorizedMediaChunks(deleted);


-- DocumentVersions Table
-- ======================
CREATE TABLE IF NOT EXISTS DocumentVersions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    media_id INTEGER NOT NULL,
    version_number INTEGER NOT NULL, -- Local version sequence per media item
    prompt TEXT,
    analysis_content TEXT,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, -- Keep original creation time
    FOREIGN KEY (media_id) REFERENCES Media(id) ON DELETE CASCADE,
    UNIQUE (media_id, version_number), -- Original constraint

    -- Sync Metadata Columns --
    uuid TEXT UNIQUE NOT NULL,
    last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    version INTEGER NOT NULL DEFAULT 1,
    client_id TEXT NOT NULL,
    deleted BOOLEAN NOT NULL DEFAULT 0
);
-- Original Indices
CREATE INDEX IF NOT EXISTS idx_document_versions_media_id ON DocumentVersions(media_id);
CREATE INDEX IF NOT EXISTS idx_document_versions_version_number ON DocumentVersions(version_number);
-- Sync Indices
CREATE UNIQUE INDEX IF NOT EXISTS idx_documentversions_uuid ON DocumentVersions(uuid);
CREATE INDEX IF NOT EXISTS idx_documentversions_last_modified ON DocumentVersions(last_modified);
CREATE INDEX IF NOT EXISTS idx_documentversions_deleted ON DocumentVersions(deleted);


-- ───────────────────────────────────────────────────────────────────────────
-- Virtual FTS Tables (Keep as is, they reference the main tables)
-- ───────────────────────────────────────────────────────────────────────────

-- FTS for Media
CREATE VIRTUAL TABLE IF NOT EXISTS media_fts USING fts5(
    title,
    content,
    content='Media',
    content_rowid='id' -- Links to Media.id (ROWID)
);
-- FTS Triggers for Media (Keep original triggers, they update FTS based on Media changes)
CREATE TRIGGER IF NOT EXISTS media_ai AFTER INSERT ON Media BEGIN
    INSERT INTO media_fts (rowid, title, content) VALUES (new.id, new.title, new.content);
END;
CREATE TRIGGER IF NOT EXISTS media_ad AFTER DELETE ON Media BEGIN
    DELETE FROM media_fts WHERE rowid = old.id;
END;
CREATE TRIGGER IF NOT EXISTS media_au AFTER UPDATE ON Media BEGIN
    UPDATE media_fts SET title = new.title, content = new.content WHERE rowid = old.id;
END;

-- FTS for Keywords
CREATE VIRTUAL TABLE IF NOT EXISTS keyword_fts USING fts5(
    keyword,
    content='Keywords',
    content_rowid='id' -- Links to Keywords.id (ROWID)
);
-- FTS Triggers for Keywords (Assuming these are added if needed)
CREATE TRIGGER IF NOT EXISTS keywords_fts_ai AFTER INSERT ON Keywords BEGIN
    INSERT INTO keyword_fts(rowid, keyword) VALUES (new.id, new.keyword);
END;
CREATE TRIGGER IF NOT EXISTS keywords_fts_ad AFTER DELETE ON Keywords BEGIN
    DELETE FROM keyword_fts WHERE rowid = old.id;
END;
CREATE TRIGGER IF NOT EXISTS keywords_fts_au AFTER UPDATE ON Keywords BEGIN
    UPDATE keyword_fts SET keyword = new.keyword WHERE rowid = old.id;
END;


-- ───────────────────────────────────────────────────────────────────────────
-- Synchronization Log Table and Indices
-- ───────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sync_log (
    change_id    INTEGER  PRIMARY KEY AUTOINCREMENT, -- Local log entry ID
    entity       TEXT     NOT NULL,   -- Table name (e.g., 'Media', 'Keywords', 'MediaKeywords')
    entity_uuid  TEXT     NOT NULL,   -- UUID of the record changed, or synthetic ID for relationships
    operation    TEXT     NOT NULL CHECK(operation IN ('create','update','delete', 'link', 'unlink')),
    timestamp    DATETIME NOT NULL,   -- Matches the record's last_modified or the time of the link/unlink
    client_id    TEXT     NOT NULL,   -- Source device UUID that made the change
    version      INTEGER  NOT NULL,   -- Version number of the record *after* the change (or parent's version for links)
    payload      TEXT              -- JSON blob of the record's state or link info
);

-- Indices for efficient querying by sync process
CREATE INDEX IF NOT EXISTS idx_sync_log_ts ON sync_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_sync_log_entity_uuid ON sync_log(entity_uuid);
CREATE INDEX IF NOT EXISTS idx_sync_log_client_id ON sync_log(client_id);


-- ───────────────────────────────────────────────────────────────────────────
-- Triggers to Populate sync_log
-- ───────────────────────────────────────────────────────────────────────────

-- ========================
-- Triggers for Media Table
-- ========================
DROP TRIGGER IF EXISTS media_sync_create;
CREATE TRIGGER media_sync_create
AFTER INSERT ON Media
BEGIN
  INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
  VALUES (
    'Media', NEW.uuid, 'create',
    NEW.last_modified, NEW.client_id, NEW.version,
    json_object(
      'uuid', NEW.uuid, 'url', NEW.url, 'title', NEW.title, 'type', NEW.type,
      'content', NEW.content, 'author', NEW.author, 'ingestion_date', NEW.ingestion_date,
      'transcription_model', NEW.transcription_model, 'is_trash', NEW.is_trash, 'trash_date', NEW.trash_date,
      'content_hash', NEW.content_hash, 'last_modified', NEW.last_modified,
      'version', NEW.version, 'client_id', NEW.client_id, 'deleted', NEW.deleted
      -- Exclude: id, vector_embedding, chunking_status, vector_processing (typically)
    )
  );
END;

DROP TRIGGER IF EXISTS media_sync_update;
CREATE TRIGGER media_sync_update
AFTER UPDATE ON Media
-- IMPORTANT: Only trigger update log if it's NOT a delete/undelete operation itself
WHEN OLD.deleted = NEW.deleted AND (
     -- Compare all relevant fields, using ifnull for safety
     ifnull(OLD.url,'') != ifnull(NEW.url,'') OR
     ifnull(OLD.title,'') != ifnull(NEW.title,'') OR
     ifnull(OLD.type,'') != ifnull(NEW.type,'') OR
     ifnull(OLD.content,'') != ifnull(NEW.content,'') OR
     ifnull(OLD.author,'') != ifnull(NEW.author,'') OR
     ifnull(OLD.ingestion_date,'') != ifnull(NEW.ingestion_date,'') OR
     ifnull(OLD.transcription_model,'') != ifnull(NEW.transcription_model,'') OR
     ifnull(OLD.is_trash,0) != ifnull(NEW.is_trash,0) OR
     ifnull(OLD.trash_date,'') != ifnull(NEW.trash_date,'') OR
     ifnull(OLD.content_hash,'') != ifnull(NEW.content_hash,'') OR
     -- Crucially, also trigger if only metadata changed (important for conflict resolution)
     ifnull(OLD.last_modified,'') != ifnull(NEW.last_modified,'') OR
     ifnull(OLD.version,0) != ifnull(NEW.version,0)
)
BEGIN
  INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
  VALUES (
    'Media', NEW.uuid, 'update',
    NEW.last_modified, NEW.client_id, NEW.version,
    json_object( -- Full payload on update
      'uuid', NEW.uuid, 'url', NEW.url, 'title', NEW.title, 'type', NEW.type,
      'content', NEW.content, 'author', NEW.author, 'ingestion_date', NEW.ingestion_date,
      'transcription_model', NEW.transcription_model, 'is_trash', NEW.is_trash, 'trash_date', NEW.trash_date,
      'content_hash', NEW.content_hash, 'last_modified', NEW.last_modified,
      'version', NEW.version, 'client_id', NEW.client_id, 'deleted', NEW.deleted
    )
  );
END;

DROP TRIGGER IF EXISTS media_sync_delete;
CREATE TRIGGER media_sync_delete
AFTER UPDATE ON Media
WHEN OLD.deleted = 0 AND NEW.deleted = 1 -- Trigger specifically on soft delete
BEGIN
  INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
  VALUES (
    'Media', NEW.uuid, 'delete',
    NEW.last_modified, NEW.client_id, NEW.version,
    json_object('uuid', NEW.uuid) -- Minimal payload for delete
  );
END;

-- ==========================
-- Triggers for Keywords Table
-- ==========================
DROP TRIGGER IF EXISTS keywords_sync_create;
CREATE TRIGGER keywords_sync_create
AFTER INSERT ON Keywords
BEGIN
  INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
  VALUES (
    'Keywords', NEW.uuid, 'create',
    NEW.last_modified, NEW.client_id, NEW.version,
    json_object(
      'uuid', NEW.uuid, 'keyword', NEW.keyword,
      'last_modified', NEW.last_modified, 'version', NEW.version,
      'client_id', NEW.client_id, 'deleted', NEW.deleted
    )
  );
END;

DROP TRIGGER IF EXISTS keywords_sync_update;
CREATE TRIGGER keywords_sync_update
AFTER UPDATE ON Keywords
WHEN OLD.deleted = NEW.deleted AND (
    ifnull(OLD.keyword,'') != ifnull(NEW.keyword,'') OR
    ifnull(OLD.last_modified,'') != ifnull(NEW.last_modified,'') OR
    ifnull(OLD.version,0) != ifnull(NEW.version,0)
)
BEGIN
  INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
  VALUES (
    'Keywords', NEW.uuid, 'update',
    NEW.last_modified, NEW.client_id, NEW.version,
     json_object(
      'uuid', NEW.uuid, 'keyword', NEW.keyword,
      'last_modified', NEW.last_modified, 'version', NEW.version,
      'client_id', NEW.client_id, 'deleted', NEW.deleted
    )
  );
END;

DROP TRIGGER IF EXISTS keywords_sync_delete;
CREATE TRIGGER keywords_sync_delete
AFTER UPDATE ON Keywords
WHEN OLD.deleted = 0 AND NEW.deleted = 1
BEGIN
  INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
  VALUES (
    'Keywords', NEW.uuid, 'delete',
    NEW.last_modified, NEW.client_id, NEW.version,
    json_object('uuid', NEW.uuid)
  );
END;

-- =======================================
-- Triggers for MediaKeywords Relationship
-- =======================================
DROP TRIGGER IF EXISTS mediakeywords_sync_link;
CREATE TRIGGER mediakeywords_sync_link
AFTER INSERT ON MediaKeywords
BEGIN
    INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
    SELECT
        'MediaKeywords',                                -- Entity name
        m.uuid || '_' || k.uuid,                        -- Synthetic UUID for the relationship
        'link',                                         -- Custom operation type
        strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime'), -- Timestamp of the link action
        m.client_id,                                    -- Use client_id from Media (best guess)
        m.version,                                      -- Use version from Media (best guess)
        json_object(
            'media_uuid', m.uuid,
            'keyword_uuid', k.uuid
        )
    FROM Media m, Keywords k
    WHERE m.id = NEW.media_id AND k.id = NEW.keyword_id;
END;

DROP TRIGGER IF EXISTS mediakeywords_sync_unlink;
CREATE TRIGGER mediakeywords_sync_unlink
AFTER DELETE ON MediaKeywords
BEGIN
     INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
    SELECT
        'MediaKeywords',                                -- Entity name
        m.uuid || '_' || k.uuid,                        -- Synthetic UUID for the relationship
        'unlink',                                       -- Custom operation type
        strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime'), -- Timestamp of the unlink action
        m.client_id,                                    -- Use client_id from Media (best guess)
        m.version,                                      -- Use version from Media (best guess)
        json_object(
            'media_uuid', m.uuid,
            'keyword_uuid', k.uuid
        )
    FROM Media m, Keywords k
    WHERE m.id = OLD.media_id AND k.id = OLD.keyword_id;
END;

-- ==========================
-- Triggers for Transcripts Table
-- ==========================
DROP TRIGGER IF EXISTS transcripts_sync_create;
CREATE TRIGGER transcripts_sync_create
AFTER INSERT ON Transcripts
BEGIN
  INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
  VALUES (
    'Transcripts', NEW.uuid, 'create',
    NEW.last_modified, NEW.client_id, NEW.version,
    json_object(
      'uuid', NEW.uuid,
      'media_uuid', (SELECT uuid FROM Media WHERE id = NEW.media_id), -- Include parent UUID
      'whisper_model', NEW.whisper_model, 'transcription', NEW.transcription,
      'created_at', NEW.created_at, -- Keep original creation timestamp
      'last_modified', NEW.last_modified, 'version', NEW.version,
      'client_id', NEW.client_id, 'deleted', NEW.deleted
    )
  );
END;

DROP TRIGGER IF EXISTS transcripts_sync_update;
CREATE TRIGGER transcripts_sync_update
AFTER UPDATE ON Transcripts
WHEN OLD.deleted = NEW.deleted AND (
    ifnull(OLD.whisper_model,'') != ifnull(NEW.whisper_model,'') OR
    ifnull(OLD.transcription,'') != ifnull(NEW.transcription,'') OR
    ifnull(OLD.last_modified,'') != ifnull(NEW.last_modified,'') OR
    ifnull(OLD.version,0) != ifnull(NEW.version,0)
)
BEGIN
  INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
  VALUES (
    'Transcripts', NEW.uuid, 'update',
    NEW.last_modified, NEW.client_id, NEW.version,
    json_object(
      'uuid', NEW.uuid,
      'media_uuid', (SELECT uuid FROM Media WHERE id = NEW.media_id),
      'whisper_model', NEW.whisper_model, 'transcription', NEW.transcription,
      'created_at', NEW.created_at,
      'last_modified', NEW.last_modified, 'version', NEW.version,
      'client_id', NEW.client_id, 'deleted', NEW.deleted
    )
  );
END;

DROP TRIGGER IF EXISTS transcripts_sync_delete;
CREATE TRIGGER transcripts_sync_delete
AFTER UPDATE ON Transcripts
WHEN OLD.deleted = 0 AND NEW.deleted = 1
BEGIN
  INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
  VALUES (
    'Transcripts', NEW.uuid, 'delete',
    NEW.last_modified, NEW.client_id, NEW.version,
    json_object('uuid', NEW.uuid)
  );
END;

-- ==========================
-- Triggers for MediaChunks Table
-- ==========================
DROP TRIGGER IF EXISTS mediachunks_sync_create;
CREATE TRIGGER mediachunks_sync_create
AFTER INSERT ON MediaChunks
BEGIN
  INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
  VALUES (
    'MediaChunks', NEW.uuid, 'create',
    NEW.last_modified, NEW.client_id, NEW.version,
    json_object(
      'uuid', NEW.uuid,
      'media_uuid', (SELECT uuid FROM Media WHERE id = NEW.media_id),
      'chunk_text', NEW.chunk_text, 'start_index', NEW.start_index, 'end_index', NEW.end_index,
      'chunk_id', NEW.chunk_id, -- Sync original chunk_id if needed
      'last_modified', NEW.last_modified, 'version', NEW.version,
      'client_id', NEW.client_id, 'deleted', NEW.deleted
    )
  );
END;

DROP TRIGGER IF EXISTS mediachunks_sync_update;
CREATE TRIGGER mediachunks_sync_update
AFTER UPDATE ON MediaChunks
WHEN OLD.deleted = NEW.deleted AND (
    ifnull(OLD.chunk_text,'') != ifnull(NEW.chunk_text,'') OR
    ifnull(OLD.start_index,0) != ifnull(NEW.start_index,0) OR
    ifnull(OLD.end_index,0) != ifnull(NEW.end_index,0) OR
    ifnull(OLD.chunk_id,'') != ifnull(NEW.chunk_id,'') OR
    ifnull(OLD.last_modified,'') != ifnull(NEW.last_modified,'') OR
    ifnull(OLD.version,0) != ifnull(NEW.version,0)
)
BEGIN
  INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
  VALUES (
    'MediaChunks', NEW.uuid, 'update',
    NEW.last_modified, NEW.client_id, NEW.version,
    json_object(
      'uuid', NEW.uuid,
      'media_uuid', (SELECT uuid FROM Media WHERE id = NEW.media_id),
      'chunk_text', NEW.chunk_text, 'start_index', NEW.start_index, 'end_index', NEW.end_index,
      'chunk_id', NEW.chunk_id,
      'last_modified', NEW.last_modified, 'version', NEW.version,
      'client_id', NEW.client_id, 'deleted', NEW.deleted
    )
  );
END;

DROP TRIGGER IF EXISTS mediachunks_sync_delete;
CREATE TRIGGER mediachunks_sync_delete
AFTER UPDATE ON MediaChunks
WHEN OLD.deleted = 0 AND NEW.deleted = 1
BEGIN
  INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
  VALUES (
    'MediaChunks', NEW.uuid, 'delete',
    NEW.last_modified, NEW.client_id, NEW.version,
    json_object('uuid', NEW.uuid)
  );
END;

-- =======================================
-- Triggers for UnvectorizedMediaChunks Table
-- =======================================
DROP TRIGGER IF EXISTS unvectorizedmediachunks_sync_create;
CREATE TRIGGER unvectorizedmediachunks_sync_create
AFTER INSERT ON UnvectorizedMediaChunks
BEGIN
  INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
  VALUES (
    'UnvectorizedMediaChunks', NEW.uuid, 'create',
    NEW.last_modified, NEW.client_id, NEW.version,
    json_object(
      'uuid', NEW.uuid,
      'media_uuid', (SELECT uuid FROM Media WHERE id = NEW.media_id),
      'chunk_text', NEW.chunk_text, 'chunk_index', NEW.chunk_index, 'start_char', NEW.start_char,
      'end_char', NEW.end_char, 'chunk_type', NEW.chunk_type, 'creation_date', NEW.creation_date,
      -- 'is_processed', NEW.is_processed, -- Decide if this local state needs sync
      'metadata', NEW.metadata, -- Sync metadata field?
      'last_modified', NEW.last_modified, 'version', NEW.version,
      'client_id', NEW.client_id, 'deleted', NEW.deleted
    )
  );
END;

DROP TRIGGER IF EXISTS unvectorizedmediachunks_sync_update;
CREATE TRIGGER unvectorizedmediachunks_sync_update
AFTER UPDATE ON UnvectorizedMediaChunks
WHEN OLD.deleted = NEW.deleted AND (
    ifnull(OLD.chunk_text,'') != ifnull(NEW.chunk_text,'') OR
    ifnull(OLD.chunk_index,0) != ifnull(NEW.chunk_index,0) OR
    ifnull(OLD.start_char,0) != ifnull(NEW.start_char,0) OR
    ifnull(OLD.end_char,0) != ifnull(NEW.end_char,0) OR
    ifnull(OLD.chunk_type,'') != ifnull(NEW.chunk_type,'') OR
    -- ifnull(OLD.is_processed,0) != ifnull(NEW.is_processed,0) OR -- Sync local state change?
    ifnull(OLD.metadata,'') != ifnull(NEW.metadata,'') OR
    ifnull(OLD.last_modified,'') != ifnull(NEW.last_modified,'') OR
    ifnull(OLD.version,0) != ifnull(NEW.version,0)
)
BEGIN
  INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
  VALUES (
    'UnvectorizedMediaChunks', NEW.uuid, 'update',
    NEW.last_modified, NEW.client_id, NEW.version,
    json_object(
      'uuid', NEW.uuid,
      'media_uuid', (SELECT uuid FROM Media WHERE id = NEW.media_id),
      'chunk_text', NEW.chunk_text, 'chunk_index', NEW.chunk_index, 'start_char', NEW.start_char,
      'end_char', NEW.end_char, 'chunk_type', NEW.chunk_type, 'creation_date', NEW.creation_date,
      -- 'is_processed', NEW.is_processed,
      'metadata', NEW.metadata,
      'last_modified', NEW.last_modified, 'version', NEW.version,
      'client_id', NEW.client_id, 'deleted', NEW.deleted
    )
  );
END;

DROP TRIGGER IF EXISTS unvectorizedmediachunks_sync_delete;
CREATE TRIGGER unvectorizedmediachunks_sync_delete
AFTER UPDATE ON UnvectorizedMediaChunks
WHEN OLD.deleted = 0 AND NEW.deleted = 1
BEGIN
  INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
  VALUES (
    'UnvectorizedMediaChunks', NEW.uuid, 'delete',
    NEW.last_modified, NEW.client_id, NEW.version,
    json_object('uuid', NEW.uuid)
  );
END;


-- ==================================
-- Triggers for DocumentVersions Table
-- ==================================
DROP TRIGGER IF EXISTS documentversions_sync_create;
CREATE TRIGGER documentversions_sync_create
AFTER INSERT ON DocumentVersions
BEGIN
  INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
  VALUES (
    'DocumentVersions', NEW.uuid, 'create',
    NEW.last_modified, NEW.client_id, NEW.version,
    json_object(
      'uuid', NEW.uuid,
      'media_uuid', (SELECT uuid FROM Media WHERE id = NEW.media_id),
      'version_number', NEW.version_number, 'prompt', NEW.prompt,
      'analysis_content', NEW.analysis_content, 'content', NEW.content,
      'created_at', NEW.created_at,
      'last_modified', NEW.last_modified, 'version', NEW.version,
      'client_id', NEW.client_id, 'deleted', NEW.deleted
    )
  );
END;

DROP TRIGGER IF EXISTS documentversions_sync_update;
CREATE TRIGGER documentversions_sync_update
AFTER UPDATE ON DocumentVersions
WHEN OLD.deleted = NEW.deleted AND (
    ifnull(OLD.prompt,'') != ifnull(NEW.prompt,'') OR
    ifnull(OLD.analysis_content,'') != ifnull(NEW.analysis_content,'') OR
    ifnull(OLD.content,'') != ifnull(NEW.content,'') OR
    ifnull(OLD.last_modified,'') != ifnull(NEW.last_modified,'') OR
    ifnull(OLD.version,0) != ifnull(NEW.version,0)
)
BEGIN
  INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
  VALUES (
    'DocumentVersions', NEW.uuid, 'update',
    NEW.last_modified, NEW.client_id, NEW.version,
    json_object(
       'uuid', NEW.uuid,
       'media_uuid', (SELECT uuid FROM Media WHERE id = NEW.media_id), -- Include parent UUID
       'version_number', NEW.version_number, 'prompt', NEW.prompt,
       'analysis_content', NEW.analysis_content, 'content', NEW.content,
       'created_at', NEW.created_at,
       'last_modified', NEW.last_modified, 'version', NEW.version,
       'client_id', NEW.client_id, 'deleted', NEW.deleted
    )
  );
END;

DROP TRIGGER IF EXISTS documentversions_sync_delete;
CREATE TRIGGER documentversions_sync_delete
AFTER UPDATE ON DocumentVersions
WHEN OLD.deleted = 0 AND NEW.deleted = 1
BEGIN
  INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
  VALUES (
    'DocumentVersions', NEW.uuid, 'delete',
    NEW.last_modified, NEW.client_id, NEW.version,
    json_object('uuid', NEW.uuid)
  );
END;

-- Ensure their JSON payloads include necessary parent context (media_uuid)
-- and all relevant columns specific to those tables.
"""