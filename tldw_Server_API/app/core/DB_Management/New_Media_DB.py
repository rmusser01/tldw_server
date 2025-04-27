
"""
undelete Triggers: Added specific triggers for each table (*_sync_undelete) that fire when deleted goes from 1 to 0. They log an 'update' operation with the full payload, ensuring the restored state is captured by the sync log. The main UPDATE triggers are kept specific to non-delete/undelete changes using WHEN OLD.deleted = NEW.deleted AND (...).

Foreign Keys and Soft Deletes:

    The Issue: Database-level FOREIGN KEY ... ON DELETE CASCADE only works with actual DELETE FROM ... SQL statements. It does not activate when you run UPDATE Media SET deleted = 1 WHERE id = ?.

    Consequence: When you soft-delete a Media item, its related records (in Transcripts, MediaChunks, DocumentVersions, MediaKeywords) still exist in the database and still point to the media_id of the soft-deleted record. The database doesn't automatically hide or soft-delete them.

    What You Need to Do (in Application/Sync Logic):

        Querying: When displaying or using data, always filter by deleted = 0 on the relevant tables unless you specifically want to show/process deleted items (e.g., SELECT * FROM Transcripts WHERE media_id = ? AND deleted = 0).

        Syncing Deletes: When your sync logic processes a delete operation from the sync_log for a Media item, the receiving client needs to know that this implies the related child records (Transcripts, Chunks, etc.) associated with that Media's UUID should also be considered deleted, even if they haven't received explicit delete logs for each child. Alternatively, when soft-deleting a Media item, your application code could also explicitly loop through its children, set their deleted flag to 1, and update their last_modified, version, etc. This would generate individual delete entries in the sync_log for each child, making the log more explicit but requiring more application logic and potentially more log entries. The best approach depends on how your sync service interprets the logs.

        Syncing Unlinks: Similarly, if a Media item is deleted, the entries linking it in MediaKeywords aren't automatically removed by ON DELETE CASCADE. Your application should explicitly DELETE FROM MediaKeywords WHERE media_id = ? when soft-deleting the Media item. The mediakeywords_sync_unlink trigger will then correctly log these removals.

Timestamp Usage: Changed TEXT to DATETIME for ingestion_date, trash_date, last_modified, timestamp (in sync_log), and kept TIMESTAMP for created_at, creation_date. In SQLite, DATETIME and TIMESTAMP have NUMERIC affinity but often store date/time strings as TEXT (like 'YYYY-MM-DD HH:MM:SS') or Unix timestamps as INTEGER, depending on what you insert. Using DATETIME is common practice and clearly signals intent. DEFAULT CURRENT_TIMESTAMP works correctly with this.

MediaChunks.chunk_id: Kept as TEXT UNIQUE assuming global uniqueness is desired. If it only needs to be unique per media item, change it to UNIQUE (media_id, chunk_id).

Chunking Lifecycle: Added comments reflecting the flow: Media ingestion -> UnvectorizedMediaChunks creation (async/delayed) -> potential future processing. MediaChunks table is kept for now, assuming it might serve a purpose or for legacy reasons. If UnvectorizedMediaChunks is the only chunk table needed going forward, you could consider merging/removing MediaChunks.

Payload Exclusions & last_modified:

    Reviewed payloads in triggers. Excluded fields like id (local), vector_embedding (large/volatile), chunking_status, vector_processing, is_processed (local processing state). Included metadata in UnvectorizedMediaChunks as it might contain sync-relevant info.

    Crucially for Sync: Your application code, before inserting or updating a record, must handle setting the correct last_modified timestamp (e.g., using the application's current time or the server's time if available), incrementing the version, and setting the client_id. The triggers use the values present in the NEW row after your INSERT/UPDATE statement finishes but before the transaction commits. The DEFAULT CURRENT_TIMESTAMP on last_modified is a fallback but relying on explicit setting by the app is better for sync accuracy. The version needs explicit incrementing logic in your app. These fields (along with uuid and deleted) are the core components your sync service will use to compare states, detect conflicts (e.g., different client_ids modifying the same version or having later last_modified times), and enable the merge/conflict resolution logic you described.


    Managing last_modified:

        DB Default: DEFAULT CURRENT_TIMESTAMP sets the time the database row was created/modified.

        Why App Role is Crucial: For synchronization, you often need the timestamp to reflect the logical time the change occurred on the client or was committed by the user. Relying purely on the DB timestamp can be problematic if changes are batched, synced later, or if client clocks are slightly off.

        Application Task: Before an INSERT or UPDATE, your application code should determine the correct timestamp (e.g., datetime.now() in Python, new Date() in JS) and explicitly include it in the SQL statement (INSERT INTO Media (..., last_modified, ...) VALUES (..., ?, ...) or UPDATE Media SET ..., last_modified = ?, ... WHERE ...). This ensures the timestamp accurately reflects when the change was intended/made from the application's perspective, which is often better for conflict resolution ("Last Write Wins" strategies).

    Managing version:

        DB Default: DEFAULT 1 only helps on initial INSERT. The database does not automatically increment the version number on UPDATE.

        Why App Role is Crucial: The version number is essential for detecting concurrent edits. If two clients fetch version 5 of a record, both modify it, and try to save, they should both be attempting to save version 6. The first one succeeds, the second one fails (or triggers conflict resolution) because the version in the database is already 6.

        Application Task: Before performing an UPDATE, your application must:

            Read the current version of the record from the database.

            Increment this number by 1.

            Include the new incremented version in the UPDATE statement (UPDATE Media SET ..., version = ?, ... WHERE uuid = ?).

            (Optional but Recommended for Stronger Conflict Detection): Add the original version to the WHERE clause (UPDATE Media SET ..., version = new_version, ... WHERE uuid = ? AND version = original_version). If this UPDATE affects 0 rows, it means another client updated the record first (changed the version), indicating a conflict that the application needs to handle.

    Managing client_id:

        DB Default: None (NOT NULL constraint). The database has no idea which client is making the change.

        Why App Role is Crucial: You need to know who made the change for the sync_log and potentially for conflict resolution rules.

        Application Task: Every instance of your application (on each device/user session) needs a unique ID. When performing any INSERT or UPDATE that should be synced, the application must provide its own client_id (INSERT INTO Media (..., client_id, ...) VALUES (..., ?, ...) or UPDATE Media SET ..., client_id = ?, ... WHERE ...).

    Handling Cascading Effects of Soft Deletes:

        DB Limitation: FOREIGN KEY ... ON DELETE CASCADE only works for actual DELETE statements, not for UPDATE Media SET deleted = 1 WHERE ....

        Why App Role is Crucial: When you soft-delete a parent record (like Media), the database does not automatically:

            Soft-delete the child records (like Transcripts, MediaChunks, DocumentVersions). They remain with deleted = 0.

            Remove the linking entries in many-to-many tables (like MediaKeywords). The links still exist.

        Application Task (Choose one approach):

            A) Explicit Cascade (in Application Code): When soft-deleting a Media record:

                Find all its related child records (Transcripts, MediaChunks, DocumentVersions using the media_id).

                Perform an UPDATE on each of these child tables, setting deleted = 1, updating their last_modified, incrementing their version, and setting the client_id. (This will trigger their respective _sync_delete triggers).

                Perform a DELETE from MediaKeywords where media_id matches. (This will trigger the mediakeywords_sync_unlink trigger).

            B) Implicit Cascade (in Application Logic & Queries):

                Only soft-delete the parent Media record.

                In all application queries that retrieve child records or use links, always join back to the parent Media table and add a condition like AND Media.deleted = 0. This ensures you only ever work with children of non-deleted parents.

                Your sync logic must understand that receiving a delete log for a Media UUID means all its associated children (even without explicit logs) should also be considered deleted on the receiving client.

In Summary:

The database triggers automate the logging of changes based on the data present after an INSERT or UPDATE occurs. However, the application is responsible for preparing the correct data (last_modified, version, client_id) for those operations and for managing the logical relationships (like cascading soft deletes) that the database's foreign key constraints cannot handle automatically in a soft-delete scenario. Getting this application logic right is fundamental for the synchronization mechanism to work correctly and maintain data integrity across clients.


FTS Query Filtering Explained

What is FTS (Full-Text Search)?

Your media_fts and keyword_fts tables are specialized indices created using CREATE VIRTUAL TABLE ... USING fts5(...). They are designed for efficiently searching for words or phrases within large blocks of text (like Media.title, Media.content, Keywords.keyword). They work differently from regular indices. They break down the text into tokens (words) and create an internal structure optimized for the MATCH operator.

The Problem: FTS Tables are Separate

    The media_fts table contains the searchable text (title, content) and the rowid of the corresponding row in the Media table.

    Crucially, media_fts does not contain the deleted or is_trash columns from the Media table.

Therefore, if you run a query only against the FTS table:


-- INCORRECT for filtering deleted/trashed items
SELECT rowid, title FROM media_fts WHERE media_fts MATCH 'some search query';



IGNORE_WHEN_COPYING_START
Use code with caution. SQL
IGNORE_WHEN_COPYING_END

This query will return results from media_fts for all Media rows that match the text query, regardless of whether the corresponding Media row has deleted = 1 or is_trash = 1. The FTS index itself doesn't know about those flags.

The Solution: JOIN Back to the Original Table

To correctly filter FTS results based on status flags (or any other columns) in the original table, you must join the FTS table back to the original table:

    Use the MATCH operator on the FTS table (media_fts) to find potential candidates based on text content.

    Use the rowid from the FTS table to link (JOIN) back to the id (which is typically the rowid alias) of the original Media table.

    Apply your standard filtering conditions (deleted = 0, is_trash = 0, type = '...', etc.) to the columns in the original Media table.

Correct FTS Query Example:


SELECT
    m.id,          -- Select columns from the original Media table
    m.uuid,
    m.title,
    m.type,
    m.last_modified
    -- Add snippet(media_fts, ...) or other FTS functions if needed
FROM
    media_fts fts   -- Start with the FTS table (aliased as fts)
JOIN
    Media m ON fts.rowid = m.id -- Join back to Media (aliased as m) using the rowid link
WHERE
    fts.media_fts MATCH 'your search query' -- Filter based on text content using MATCH
  AND
    m.deleted = 0               -- Filter based on the deleted flag in the Media table
  AND
    m.is_trash = 0;             -- Filter based on the is_trash flag in the Media table
-- Add ORDER BY m.last_modified DESC etc. as needed


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
    ingestion_date DATETIME, -- Changed to DATETIME
    transcription_model TEXT,
    is_trash BOOLEAN DEFAULT 0 NOT NULL, -- For UI Trash Can feature
    trash_date DATETIME,                 -- Changed to DATETIME
    vector_embedding BLOB,               -- Often excluded from sync payload due to size/volatility (Local Only)
    chunking_status TEXT DEFAULT 'pending' NOT NULL, -- Likely managed locally, exclude from sync (Local Only)
    vector_processing INTEGER DEFAULT 0 NOT NULL,    -- Likely managed locally, exclude from sync (Local Only)
    content_hash TEXT UNIQUE NOT NULL,

    -- Sync Metadata Columns --
    uuid TEXT UNIQUE NOT NULL, -- Globally unique identifier for sync
    last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, -- App should ideally set this explicitly on change
    version INTEGER NOT NULL DEFAULT 1,                        -- App must increment this on change
    client_id TEXT NOT NULL,                                   -- Identifier of the client that made the last change (Set by App)
    deleted BOOLEAN NOT NULL DEFAULT 0                         -- Soft delete flag for sync
);
-- Original Indices
CREATE INDEX IF NOT EXISTS idx_media_title ON Media(title);
CREATE INDEX IF NOT EXISTS idx_media_type ON Media(type);
CREATE INDEX IF NOT EXISTS idx_media_author ON Media(author);
CREATE INDEX IF NOT EXISTS idx_media_ingestion_date ON Media(ingestion_date);
CREATE INDEX IF NOT EXISTS idx_media_chunking_status ON Media(chunking_status); -- For local queries
CREATE INDEX IF NOT EXISTS idx_media_vector_processing ON Media(vector_processing); -- For local queries
CREATE INDEX IF NOT EXISTS idx_media_is_trash ON Media(is_trash); -- For UI queries
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
    last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, -- App should ideally set this explicitly on change
    version INTEGER NOT NULL DEFAULT 1,                        -- App must increment this on change
    client_id TEXT NOT NULL,                                   -- Set by App
    deleted BOOLEAN NOT NULL DEFAULT 0
);
-- Original Indices (keyword covered by UNIQUE)
-- Sync Indices
CREATE UNIQUE INDEX IF NOT EXISTS idx_keywords_uuid ON Keywords(uuid);
CREATE INDEX IF NOT EXISTS idx_keywords_last_modified ON Keywords(last_modified);
CREATE INDEX IF NOT EXISTS idx_keywords_deleted ON Keywords(deleted);


-- MediaKeywords Table (Relationship Table - No direct sync metadata needed)
-- ==========================================================================
-- Sync is handled by 'link'/'unlink' operations in sync_log triggered below.
-- NOTE: Soft-deleting Media/Keywords does NOT automatically delete rows here
-- due to use of soft deletes. Application logic must handle unlinking.
CREATE TABLE IF NOT EXISTS MediaKeywords (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    media_id INTEGER NOT NULL,
    keyword_id INTEGER NOT NULL,
    UNIQUE (media_id, keyword_id),
    FOREIGN KEY (media_id) REFERENCES Media(id) ON DELETE CASCADE,     -- Cascade works on HARD delete only
    FOREIGN KEY (keyword_id) REFERENCES Keywords(id) ON DELETE CASCADE -- Cascade works on HARD delete only
);
-- Original Indices
CREATE INDEX IF NOT EXISTS idx_mediakeywords_media_id ON MediaKeywords(media_id);
CREATE INDEX IF NOT EXISTS idx_mediakeywords_keyword_id ON MediaKeywords(keyword_id);


-- Transcripts Table
-- =================
-- NOTE: Soft-deleting parent Media does NOT automatically soft-delete Transcripts.
-- Application/Sync logic must handle this relationship based on Media.deleted flag.
CREATE TABLE IF NOT EXISTS Transcripts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    media_id INTEGER NOT NULL,
    whisper_model TEXT,
    transcription TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, -- Keep original creation time
    UNIQUE (media_id, whisper_model), -- Original constraint
    FOREIGN KEY (media_id) REFERENCES Media(id) ON DELETE CASCADE, -- Cascade works on HARD delete only

    -- Sync Metadata Columns --
    uuid TEXT UNIQUE NOT NULL,
    last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, -- App should ideally set this explicitly on change
    version INTEGER NOT NULL DEFAULT 1,                        -- App must increment this on change
    client_id TEXT NOT NULL,                                   -- Set by App
    deleted BOOLEAN NOT NULL DEFAULT 0
);
-- Original Indices
CREATE INDEX IF NOT EXISTS idx_transcripts_media_id ON Transcripts(media_id);
-- Sync Indices
CREATE UNIQUE INDEX IF NOT EXISTS idx_transcripts_uuid ON Transcripts(uuid);
CREATE INDEX IF NOT EXISTS idx_transcripts_last_modified ON Transcripts(last_modified);
CREATE INDEX IF NOT EXISTS idx_transcripts_deleted ON Transcripts(deleted);


-- MediaChunks Table (Consider if still needed if vector embeddings are not stored here)
-- =================
-- Represents processed/structured chunks of media content.
-- NOTE: Soft-deleting parent Media does NOT automatically soft-delete MediaChunks.
-- Application/Sync logic must handle this relationship based on Media.deleted flag.
CREATE TABLE IF NOT EXISTS MediaChunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    media_id INTEGER NOT NULL,
    chunk_text TEXT,
    start_index INTEGER,
    end_index INTEGER,
    chunk_id TEXT UNIQUE, -- Assumed Globally Unique Identifier for the chunk content/position. If unique per media, use UNIQUE(media_id, chunk_id)
    FOREIGN KEY (media_id) REFERENCES Media(id) ON DELETE CASCADE, -- Cascade works on HARD delete only

    -- Sync Metadata Columns --
    uuid TEXT UNIQUE NOT NULL,
    last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, -- App should ideally set this explicitly on change
    version INTEGER NOT NULL DEFAULT 1,                        -- App must increment this on change
    client_id TEXT NOT NULL,                                   -- Set by App
    deleted BOOLEAN NOT NULL DEFAULT 0
);
-- Original Indices
CREATE INDEX IF NOT EXISTS idx_mediachunks_media_id ON MediaChunks(media_id);
-- Sync Indices
CREATE UNIQUE INDEX IF NOT EXISTS idx_mediachunks_uuid ON MediaChunks(uuid);
CREATE INDEX IF NOT EXISTS idx_mediachunks_last_modified ON MediaChunks(last_modified);
CREATE INDEX IF NOT EXISTS idx_mediachunks_deleted ON MediaChunks(deleted);


-- UnvectorizedMediaChunks Table (Likely precursor to processing/vectorization)
-- =============================
-- Stores chunks identified during initial processing, before vectorization (if any).
-- Can be created asynchronously after Media ingestion.
-- NOTE: Soft-deleting parent Media does NOT automatically soft-delete these chunks.
-- Application/Sync logic must handle this relationship based on Media.deleted flag.
CREATE TABLE IF NOT EXISTS UnvectorizedMediaChunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    media_id INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    start_char INTEGER,
    end_char INTEGER,
    chunk_type TEXT,
    creation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP, -- Keep original creation time
    last_modified_orig TIMESTAMP DEFAULT CURRENT_TIMESTAMP, -- Renamed original timestamp if needed for app logic (Consider removing if sync `last_modified` is sufficient)
    is_processed BOOLEAN DEFAULT FALSE NOT NULL, -- Local state, exclude from sync payload (Local Only)
    metadata TEXT, -- Potentially sync this if it contains relevant info (e.g., user notes)
    UNIQUE (media_id, chunk_index, chunk_type), -- Original constraint
    FOREIGN KEY (media_id) REFERENCES Media(id) ON DELETE CASCADE, -- Cascade works on HARD delete only

    -- Sync Metadata Columns --
    uuid TEXT UNIQUE NOT NULL,
    last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, -- Sync timestamp (App should ideally set explicitly)
    version INTEGER NOT NULL DEFAULT 1,                        -- App must increment this on change
    client_id TEXT NOT NULL,                                   -- Set by App
    deleted BOOLEAN NOT NULL DEFAULT 0
);
-- Original Indices
CREATE INDEX IF NOT EXISTS idx_unvectorized_media_chunks_media_id ON UnvectorizedMediaChunks(media_id);
CREATE INDEX IF NOT EXISTS idx_unvectorized_media_chunks_is_processed ON UnvectorizedMediaChunks(is_processed); -- For local queries
CREATE INDEX IF NOT EXISTS idx_unvectorized_media_chunks_chunk_type ON UnvectorizedMediaChunks(chunk_type);
-- Sync Indices
CREATE UNIQUE INDEX IF NOT EXISTS idx_unvectorizedmediachunks_uuid ON UnvectorizedMediaChunks(uuid);
CREATE INDEX IF NOT EXISTS idx_unvectorizedmediachunks_last_modified ON UnvectorizedMediaChunks(last_modified);
CREATE INDEX IF NOT EXISTS idx_unvectorizedmediachunks_deleted ON UnvectorizedMediaChunks(deleted);


-- DocumentVersions Table
-- ======================
-- Stores snapshots or analysis versions related to a media item.
-- NOTE: Soft-deleting parent Media does NOT automatically soft-delete DocumentVersions.
-- Application/Sync logic must handle this relationship based on Media.deleted flag.
CREATE TABLE IF NOT EXISTS DocumentVersions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    media_id INTEGER NOT NULL,
    version_number INTEGER NOT NULL, -- Local version sequence per media item
    prompt TEXT,
    analysis_content TEXT,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, -- Keep original creation time
    FOREIGN KEY (media_id) REFERENCES Media(id) ON DELETE CASCADE, -- Cascade works on HARD delete only
    UNIQUE (media_id, version_number), -- Original constraint

    -- Sync Metadata Columns --
    uuid TEXT UNIQUE NOT NULL,
    last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, -- App should ideally set this explicitly on change
    version INTEGER NOT NULL DEFAULT 1,                        -- App must increment this on change
    client_id TEXT NOT NULL,                                   -- Set by App
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
-- Note: These triggers work correctly even with soft deletes, as they fire on INSERT/UPDATE/DELETE
-- of the main Media table. Searching FTS won't automatically exclude soft-deleted items unless
-- you join back to Media table in your FTS query (e.g., ... JOIN Media ON media_fts.rowid = Media.id WHERE Media.deleted = 0)
CREATE TRIGGER IF NOT EXISTS media_ai AFTER INSERT ON Media BEGIN
    INSERT INTO media_fts (rowid, title, content) VALUES (new.id, new.title, new.content);
END;
CREATE TRIGGER IF NOT EXISTS media_ad AFTER DELETE ON Media BEGIN
    DELETE FROM media_fts WHERE rowid = old.id; -- Only fires on HARD delete
END;
CREATE TRIGGER IF NOT EXISTS media_au AFTER UPDATE ON Media BEGIN
    -- Update FTS regardless of soft delete status, filtering happens at query time
    UPDATE media_fts SET title = new.title, content = new.content WHERE rowid = old.id;
END;

-- FTS for Keywords
CREATE VIRTUAL TABLE IF NOT EXISTS keyword_fts USING fts5(
    keyword,
    content='Keywords',
    content_rowid='id' -- Links to Keywords.id (ROWID)
);
-- FTS Triggers for Keywords
CREATE TRIGGER IF NOT EXISTS keywords_fts_ai AFTER INSERT ON Keywords BEGIN
    INSERT INTO keyword_fts(rowid, keyword) VALUES (new.id, new.keyword);
END;
CREATE TRIGGER IF NOT EXISTS keywords_fts_ad AFTER DELETE ON Keywords BEGIN
    DELETE FROM keyword_fts WHERE rowid = old.id; -- Only fires on HARD delete
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
    timestamp    DATETIME NOT NULL,   -- Matches the record's last_modified or the time of the link/unlink (Changed to DATETIME)
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
-- (Includes specific UNDELETE triggers)
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
      -- Excluded (Local/Volatile): id, vector_embedding, chunking_status, vector_processing
    )
  );
END;

DROP TRIGGER IF EXISTS media_sync_update;
CREATE TRIGGER media_sync_update
AFTER UPDATE ON Media
-- IMPORTANT: Only trigger for actual data/metadata updates, NOT delete/undelete actions.
WHEN OLD.deleted = NEW.deleted AND (
     ifnull(OLD.url,'') != ifnull(NEW.url,'') OR
     ifnull(OLD.title,'') != ifnull(NEW.title,'') OR
     ifnull(OLD.type,'') != ifnull(NEW.type,'') OR
     ifnull(OLD.content,'') != ifnull(NEW.content,'') OR
     ifnull(OLD.author,'') != ifnull(NEW.author,'') OR
     ifnull(OLD.ingestion_date,'') != ifnull(NEW.ingestion_date,'') OR
     ifnull(OLD.transcription_model,'') != ifnull(NEW.transcription_model,'') OR
     ifnull(OLD.is_trash,0) != ifnull(NEW.is_trash,0) OR -- Include UI trash changes
     ifnull(OLD.trash_date,'') != ifnull(NEW.trash_date,'') OR -- Include UI trash changes
     ifnull(OLD.content_hash,'') != ifnull(NEW.content_hash,'') OR
     -- Include sync metadata changes if they occur without data change (less common but possible)
     ifnull(OLD.last_modified,'') != ifnull(NEW.last_modified,'') OR
     ifnull(OLD.version,0) != ifnull(NEW.version,0) OR
     ifnull(OLD.client_id, '') != ifnull(NEW.client_id, '')
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
      -- Excluded (Local/Volatile): id, vector_embedding, chunking_status, vector_processing
    )
  );
END;

DROP TRIGGER IF EXISTS media_sync_delete;
CREATE TRIGGER media_sync_delete
AFTER UPDATE ON Media
WHEN OLD.deleted = 0 AND NEW.deleted = 1 -- Trigger specifically on soft delete (0 -> 1)
BEGIN
  INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
  VALUES (
    'Media', NEW.uuid, 'delete',
    NEW.last_modified, NEW.client_id, NEW.version,
    json_object('uuid', NEW.uuid, 'last_modified', NEW.last_modified, 'version', NEW.version, 'client_id', NEW.client_id) -- Minimal payload + sync meta for delete
  );
END;

DROP TRIGGER IF EXISTS media_sync_undelete;
CREATE TRIGGER media_sync_undelete
AFTER UPDATE ON Media
WHEN OLD.deleted = 1 AND NEW.deleted = 0 -- Trigger specifically on undelete (1 -> 0)
BEGIN
  INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
  VALUES (
    'Media', NEW.uuid, 'update', -- Log as an 'update' because the item is now active with its current state
    NEW.last_modified, NEW.client_id, NEW.version,
    json_object( -- Full payload needed for restore
      'uuid', NEW.uuid, 'url', NEW.url, 'title', NEW.title, 'type', NEW.type,
      'content', NEW.content, 'author', NEW.author, 'ingestion_date', NEW.ingestion_date,
      'transcription_model', NEW.transcription_model, 'is_trash', NEW.is_trash, 'trash_date', NEW.trash_date,
      'content_hash', NEW.content_hash, 'last_modified', NEW.last_modified,
      'version', NEW.version, 'client_id', NEW.client_id, 'deleted', NEW.deleted
      -- Excluded (Local/Volatile): id, vector_embedding, chunking_status, vector_processing
    )
  );
END;


-- ==========================
-- Triggers for Keywords Table
-- ==========================
DROP TRIGGER IF EXISTS keywords_sync_create;
CREATE TRIGGER keywords_sync_create AFTER INSERT ON Keywords BEGIN
  INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
  VALUES ('Keywords', NEW.uuid, 'create', NEW.last_modified, NEW.client_id, NEW.version,
    json_object('uuid', NEW.uuid, 'keyword', NEW.keyword, 'last_modified', NEW.last_modified, 'version', NEW.version, 'client_id', NEW.client_id, 'deleted', NEW.deleted));
END;

DROP TRIGGER IF EXISTS keywords_sync_update;
CREATE TRIGGER keywords_sync_update AFTER UPDATE ON Keywords
WHEN OLD.deleted = NEW.deleted AND (
    ifnull(OLD.keyword,'') != ifnull(NEW.keyword,'') OR
    ifnull(OLD.last_modified,'') != ifnull(NEW.last_modified,'') OR
    ifnull(OLD.version,0) != ifnull(NEW.version,0) OR
    ifnull(OLD.client_id, '') != ifnull(NEW.client_id, '')
) BEGIN
  INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
  VALUES ('Keywords', NEW.uuid, 'update', NEW.last_modified, NEW.client_id, NEW.version,
     json_object('uuid', NEW.uuid, 'keyword', NEW.keyword, 'last_modified', NEW.last_modified, 'version', NEW.version, 'client_id', NEW.client_id, 'deleted', NEW.deleted));
END;

DROP TRIGGER IF EXISTS keywords_sync_delete;
CREATE TRIGGER keywords_sync_delete AFTER UPDATE ON Keywords
WHEN OLD.deleted = 0 AND NEW.deleted = 1 BEGIN
  INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
  VALUES ('Keywords', NEW.uuid, 'delete', NEW.last_modified, NEW.client_id, NEW.version,
    json_object('uuid', NEW.uuid, 'last_modified', NEW.last_modified, 'version', NEW.version, 'client_id', NEW.client_id));
END;

DROP TRIGGER IF EXISTS keywords_sync_undelete;
CREATE TRIGGER keywords_sync_undelete AFTER UPDATE ON Keywords
WHEN OLD.deleted = 1 AND NEW.deleted = 0 BEGIN
  INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
  VALUES ('Keywords', NEW.uuid, 'update', NEW.last_modified, NEW.client_id, NEW.version,
     json_object('uuid', NEW.uuid, 'keyword', NEW.keyword, 'last_modified', NEW.last_modified, 'version', NEW.version, 'client_id', NEW.client_id, 'deleted', NEW.deleted));
END;


-- =======================================
-- Triggers for MediaKeywords Relationship
-- =======================================
DROP TRIGGER IF EXISTS mediakeywords_sync_link;
CREATE TRIGGER mediakeywords_sync_link
AFTER INSERT ON MediaKeywords
BEGIN
    -- Get parent UUIDs and metadata. Use Media's client/version as representative.
    -- Timestamp is the time of linking.
    SELECT RAISE(ABORT, 'Cannot link keyword: Media record not found or missing UUID')
    WHERE NOT EXISTS (SELECT 1 FROM Media WHERE id = NEW.media_id AND uuid IS NOT NULL);
    SELECT RAISE(ABORT, 'Cannot link keyword: Keyword record not found or missing UUID')
    WHERE NOT EXISTS (SELECT 1 FROM Keywords WHERE id = NEW.keyword_id AND uuid IS NOT NULL);

    INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
    SELECT
        'MediaKeywords',                                -- Entity name
        m.uuid || '_' || k.uuid,                        -- Synthetic UUID for the relationship
        'link',                                         -- Custom operation type
        strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime'), -- Timestamp of the link action
        m.client_id,                                    -- Use client_id from Media (convention)
        m.version,                                      -- Use version from Media (convention)
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
     -- Get parent UUIDs and metadata *before* they might be gone. Use current time.
     -- We need to handle cases where parents might already be soft-deleted but the link existed.
     -- Fetch UUIDs directly if possible, use placeholders if records are fully gone (less ideal).
     INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
     SELECT
         'MediaKeywords',
         ifnull(m.uuid, 'unknown_media_' || OLD.media_id) || '_' || ifnull(k.uuid, 'unknown_keyword_' || OLD.keyword_id),
         'unlink',
         strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime'),
         ifnull(m.client_id, 'unknown'), -- Best guess for client ID
         ifnull(m.version, 0),           -- Best guess for version
         json_object(
             'media_uuid', ifnull(m.uuid, 'unknown_media_' || OLD.media_id),
             'keyword_uuid', ifnull(k.uuid, 'unknown_keyword_' || OLD.keyword_id)
         )
     -- Use LEFT JOIN in case parent records were hard-deleted (though soft delete is expected)
     FROM (SELECT OLD.media_id as media_id, OLD.keyword_id as keyword_id) AS OldIds -- Ensure OLD values are available
     LEFT JOIN Media m ON m.id = OldIds.media_id
     LEFT JOIN Keywords k ON k.id = OldIds.keyword_id;
END;


-- ==========================
-- Triggers for Transcripts Table
-- ==========================
DROP TRIGGER IF EXISTS transcripts_sync_create;
CREATE TRIGGER transcripts_sync_create AFTER INSERT ON Transcripts BEGIN
  INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
  VALUES ('Transcripts', NEW.uuid, 'create', NEW.last_modified, NEW.client_id, NEW.version,
    json_object('uuid', NEW.uuid, 'media_uuid', (SELECT uuid FROM Media WHERE id = NEW.media_id),
      'whisper_model', NEW.whisper_model, 'transcription', NEW.transcription, 'created_at', NEW.created_at,
      'last_modified', NEW.last_modified, 'version', NEW.version, 'client_id', NEW.client_id, 'deleted', NEW.deleted));
END;

DROP TRIGGER IF EXISTS transcripts_sync_update;
CREATE TRIGGER transcripts_sync_update AFTER UPDATE ON Transcripts
WHEN OLD.deleted = NEW.deleted AND (
    ifnull(OLD.whisper_model,'') != ifnull(NEW.whisper_model,'') OR
    ifnull(OLD.transcription,'') != ifnull(NEW.transcription,'') OR
    ifnull(OLD.last_modified,'') != ifnull(NEW.last_modified,'') OR
    ifnull(OLD.version,0) != ifnull(NEW.version,0) OR
    ifnull(OLD.client_id, '') != ifnull(NEW.client_id, '')
) BEGIN
  INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
  VALUES ('Transcripts', NEW.uuid, 'update', NEW.last_modified, NEW.client_id, NEW.version,
    json_object('uuid', NEW.uuid, 'media_uuid', (SELECT uuid FROM Media WHERE id = NEW.media_id),
      'whisper_model', NEW.whisper_model, 'transcription', NEW.transcription, 'created_at', NEW.created_at,
      'last_modified', NEW.last_modified, 'version', NEW.version, 'client_id', NEW.client_id, 'deleted', NEW.deleted));
END;

DROP TRIGGER IF EXISTS transcripts_sync_delete;
CREATE TRIGGER transcripts_sync_delete AFTER UPDATE ON Transcripts
WHEN OLD.deleted = 0 AND NEW.deleted = 1 BEGIN
  INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
  VALUES ('Transcripts', NEW.uuid, 'delete', NEW.last_modified, NEW.client_id, NEW.version,
    json_object('uuid', NEW.uuid, 'media_uuid', (SELECT uuid FROM Media WHERE id = NEW.media_id), 'last_modified', NEW.last_modified, 'version', NEW.version, 'client_id', NEW.client_id));
END;

DROP TRIGGER IF EXISTS transcripts_sync_undelete;
CREATE TRIGGER transcripts_sync_undelete AFTER UPDATE ON Transcripts
WHEN OLD.deleted = 1 AND NEW.deleted = 0 BEGIN
 INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
  VALUES ('Transcripts', NEW.uuid, 'update', NEW.last_modified, NEW.client_id, NEW.version,
    json_object('uuid', NEW.uuid, 'media_uuid', (SELECT uuid FROM Media WHERE id = NEW.media_id),
      'whisper_model', NEW.whisper_model, 'transcription', NEW.transcription, 'created_at', NEW.created_at,
      'last_modified', NEW.last_modified, 'version', NEW.version, 'client_id', NEW.client_id, 'deleted', NEW.deleted));
END;


-- ==========================
-- Triggers for MediaChunks Table
-- ==========================
DROP TRIGGER IF EXISTS mediachunks_sync_create;
CREATE TRIGGER mediachunks_sync_create AFTER INSERT ON MediaChunks BEGIN
  INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
  VALUES ('MediaChunks', NEW.uuid, 'create', NEW.last_modified, NEW.client_id, NEW.version,
    json_object('uuid', NEW.uuid, 'media_uuid', (SELECT uuid FROM Media WHERE id = NEW.media_id),
      'chunk_text', NEW.chunk_text, 'start_index', NEW.start_index, 'end_index', NEW.end_index, 'chunk_id', NEW.chunk_id,
      'last_modified', NEW.last_modified, 'version', NEW.version, 'client_id', NEW.client_id, 'deleted', NEW.deleted));
END;

DROP TRIGGER IF EXISTS mediachunks_sync_update;
CREATE TRIGGER mediachunks_sync_update AFTER UPDATE ON MediaChunks
WHEN OLD.deleted = NEW.deleted AND (
    ifnull(OLD.chunk_text,'') != ifnull(NEW.chunk_text,'') OR
    ifnull(OLD.start_index,0) != ifnull(NEW.start_index,0) OR
    ifnull(OLD.end_index,0) != ifnull(NEW.end_index,0) OR
    ifnull(OLD.chunk_id,'') != ifnull(NEW.chunk_id,'') OR
    ifnull(OLD.last_modified,'') != ifnull(NEW.last_modified,'') OR
    ifnull(OLD.version,0) != ifnull(NEW.version,0) OR
    ifnull(OLD.client_id, '') != ifnull(NEW.client_id, '')
) BEGIN
  INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
  VALUES ('MediaChunks', NEW.uuid, 'update', NEW.last_modified, NEW.client_id, NEW.version,
    json_object('uuid', NEW.uuid, 'media_uuid', (SELECT uuid FROM Media WHERE id = NEW.media_id),
      'chunk_text', NEW.chunk_text, 'start_index', NEW.start_index, 'end_index', NEW.end_index, 'chunk_id', NEW.chunk_id,
      'last_modified', NEW.last_modified, 'version', NEW.version, 'client_id', NEW.client_id, 'deleted', NEW.deleted));
END;

DROP TRIGGER IF EXISTS mediachunks_sync_delete;
CREATE TRIGGER mediachunks_sync_delete AFTER UPDATE ON MediaChunks
WHEN OLD.deleted = 0 AND NEW.deleted = 1 BEGIN
  INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
  VALUES ('MediaChunks', NEW.uuid, 'delete', NEW.last_modified, NEW.client_id, NEW.version,
    json_object('uuid', NEW.uuid, 'media_uuid', (SELECT uuid FROM Media WHERE id = NEW.media_id), 'last_modified', NEW.last_modified, 'version', NEW.version, 'client_id', NEW.client_id));
END;

DROP TRIGGER IF EXISTS mediachunks_sync_undelete;
CREATE TRIGGER mediachunks_sync_undelete AFTER UPDATE ON MediaChunks
WHEN OLD.deleted = 1 AND NEW.deleted = 0 BEGIN
  INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
  VALUES ('MediaChunks', NEW.uuid, 'update', NEW.last_modified, NEW.client_id, NEW.version,
    json_object('uuid', NEW.uuid, 'media_uuid', (SELECT uuid FROM Media WHERE id = NEW.media_id),
      'chunk_text', NEW.chunk_text, 'start_index', NEW.start_index, 'end_index', NEW.end_index, 'chunk_id', NEW.chunk_id,
      'last_modified', NEW.last_modified, 'version', NEW.version, 'client_id', NEW.client_id, 'deleted', NEW.deleted));
END;


-- =======================================
-- Triggers for UnvectorizedMediaChunks Table
-- =======================================
DROP TRIGGER IF EXISTS unvectorizedmediachunks_sync_create;
CREATE TRIGGER unvectorizedmediachunks_sync_create AFTER INSERT ON UnvectorizedMediaChunks BEGIN
  INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
  VALUES ('UnvectorizedMediaChunks', NEW.uuid, 'create', NEW.last_modified, NEW.client_id, NEW.version,
    json_object('uuid', NEW.uuid, 'media_uuid', (SELECT uuid FROM Media WHERE id = NEW.media_id),
      'chunk_text', NEW.chunk_text, 'chunk_index', NEW.chunk_index, 'start_char', NEW.start_char,
      'end_char', NEW.end_char, 'chunk_type', NEW.chunk_type, 'creation_date', NEW.creation_date,
      'metadata', NEW.metadata, -- Sync metadata field (assuming it's not purely local)
      'last_modified', NEW.last_modified, 'version', NEW.version, 'client_id', NEW.client_id, 'deleted', NEW.deleted
      -- Excluded (Local): id, is_processed, last_modified_orig (if purely local)
      ));
END;

DROP TRIGGER IF EXISTS unvectorizedmediachunks_sync_update;
CREATE TRIGGER unvectorizedmediachunks_sync_update AFTER UPDATE ON UnvectorizedMediaChunks
WHEN OLD.deleted = NEW.deleted AND (
    ifnull(OLD.chunk_text,'') != ifnull(NEW.chunk_text,'') OR
    ifnull(OLD.chunk_index,0) != ifnull(NEW.chunk_index,0) OR
    ifnull(OLD.start_char,0) != ifnull(NEW.start_char,0) OR
    ifnull(OLD.end_char,0) != ifnull(NEW.end_char,0) OR
    ifnull(OLD.chunk_type,'') != ifnull(NEW.chunk_type,'') OR
    ifnull(OLD.metadata,'') != ifnull(NEW.metadata,'') OR -- Sync metadata changes
    ifnull(OLD.last_modified,'') != ifnull(NEW.last_modified,'') OR
    ifnull(OLD.version,0) != ifnull(NEW.version,0) OR
    ifnull(OLD.client_id, '') != ifnull(NEW.client_id, '')
    -- Don't trigger sync log on change of is_processed or last_modified_orig (local fields)
) BEGIN
  INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
  VALUES ('UnvectorizedMediaChunks', NEW.uuid, 'update', NEW.last_modified, NEW.client_id, NEW.version,
    json_object('uuid', NEW.uuid, 'media_uuid', (SELECT uuid FROM Media WHERE id = NEW.media_id),
      'chunk_text', NEW.chunk_text, 'chunk_index', NEW.chunk_index, 'start_char', NEW.start_char,
      'end_char', NEW.end_char, 'chunk_type', NEW.chunk_type, 'creation_date', NEW.creation_date,
      'metadata', NEW.metadata,
      'last_modified', NEW.last_modified, 'version', NEW.version, 'client_id', NEW.client_id, 'deleted', NEW.deleted
       -- Excluded (Local): id, is_processed, last_modified_orig (if purely local)
      ));
END;

DROP TRIGGER IF EXISTS unvectorizedmediachunks_sync_delete;
CREATE TRIGGER unvectorizedmediachunks_sync_delete AFTER UPDATE ON UnvectorizedMediaChunks
WHEN OLD.deleted = 0 AND NEW.deleted = 1 BEGIN
  INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
  VALUES ('UnvectorizedMediaChunks', NEW.uuid, 'delete', NEW.last_modified, NEW.client_id, NEW.version,
    json_object('uuid', NEW.uuid, 'media_uuid', (SELECT uuid FROM Media WHERE id = NEW.media_id), 'last_modified', NEW.last_modified, 'version', NEW.version, 'client_id', NEW.client_id));
END;

DROP TRIGGER IF EXISTS unvectorizedmediachunks_sync_undelete;
CREATE TRIGGER unvectorizedmediachunks_sync_undelete AFTER UPDATE ON UnvectorizedMediaChunks
WHEN OLD.deleted = 1 AND NEW.deleted = 0 BEGIN
  INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
  VALUES ('UnvectorizedMediaChunks', NEW.uuid, 'update', NEW.last_modified, NEW.client_id, NEW.version,
    json_object('uuid', NEW.uuid, 'media_uuid', (SELECT uuid FROM Media WHERE id = NEW.media_id),
      'chunk_text', NEW.chunk_text, 'chunk_index', NEW.chunk_index, 'start_char', NEW.start_char,
      'end_char', NEW.end_char, 'chunk_type', NEW.chunk_type, 'creation_date', NEW.creation_date,
      'metadata', NEW.metadata,
      'last_modified', NEW.last_modified, 'version', NEW.version, 'client_id', NEW.client_id, 'deleted', NEW.deleted
       -- Excluded (Local): id, is_processed, last_modified_orig (if purely local)
      ));
END;


-- ==================================
-- Triggers for DocumentVersions Table
-- ==================================
DROP TRIGGER IF EXISTS documentversions_sync_create;
CREATE TRIGGER documentversions_sync_create AFTER INSERT ON DocumentVersions BEGIN
  INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
  VALUES ('DocumentVersions', NEW.uuid, 'create', NEW.last_modified, NEW.client_id, NEW.version,
    json_object('uuid', NEW.uuid, 'media_uuid', (SELECT uuid FROM Media WHERE id = NEW.media_id),
      'version_number', NEW.version_number, 'prompt', NEW.prompt, 'analysis_content', NEW.analysis_content,
      'content', NEW.content, 'created_at', NEW.created_at, 'last_modified', NEW.last_modified,
      'version', NEW.version, 'client_id', NEW.client_id, 'deleted', NEW.deleted));
END;

DROP TRIGGER IF EXISTS documentversions_sync_update;
CREATE TRIGGER documentversions_sync_update AFTER UPDATE ON DocumentVersions
WHEN OLD.deleted = NEW.deleted AND (
    ifnull(OLD.prompt,'') != ifnull(NEW.prompt,'') OR
    ifnull(OLD.analysis_content,'') != ifnull(NEW.analysis_content,'') OR
    ifnull(OLD.content,'') != ifnull(NEW.content,'') OR
    -- version_number change shouldn't trigger sync update if only local, but maybe it should sync? Included for now.
    ifnull(OLD.version_number,0) != ifnull(NEW.version_number,0) OR
    ifnull(OLD.last_modified,'') != ifnull(NEW.last_modified,'') OR
    ifnull(OLD.version,0) != ifnull(NEW.version,0) OR
    ifnull(OLD.client_id, '') != ifnull(NEW.client_id, '')
) BEGIN
  INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
  VALUES ('DocumentVersions', NEW.uuid, 'update', NEW.last_modified, NEW.client_id, NEW.version,
    json_object('uuid', NEW.uuid, 'media_uuid', (SELECT uuid FROM Media WHERE id = NEW.media_id),
      'version_number', NEW.version_number, 'prompt', NEW.prompt, 'analysis_content', NEW.analysis_content,
      'content', NEW.content, 'created_at', NEW.created_at, 'last_modified', NEW.last_modified,
      'version', NEW.version, 'client_id', NEW.client_id, 'deleted', NEW.deleted));
END;

DROP TRIGGER IF EXISTS documentversions_sync_delete;
CREATE TRIGGER documentversions_sync_delete AFTER UPDATE ON DocumentVersions
WHEN OLD.deleted = 0 AND NEW.deleted = 1 BEGIN
  INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
  VALUES ('DocumentVersions', NEW.uuid, 'delete', NEW.last_modified, NEW.client_id, NEW.version,
    json_object('uuid', NEW.uuid, 'media_uuid', (SELECT uuid FROM Media WHERE id = NEW.media_id), 'last_modified', NEW.last_modified, 'version', NEW.version, 'client_id', NEW.client_id));
END;

DROP TRIGGER IF EXISTS documentversions_sync_undelete;
CREATE TRIGGER documentversions_sync_undelete AFTER UPDATE ON DocumentVersions
WHEN OLD.deleted = 1 AND NEW.deleted = 0 BEGIN
  INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
  VALUES ('DocumentVersions', NEW.uuid, 'update', NEW.last_modified, NEW.client_id, NEW.version,
    json_object('uuid', NEW.uuid, 'media_uuid', (SELECT uuid FROM Media WHERE id = NEW.media_id),
      'version_number', NEW.version_number, 'prompt', NEW.prompt, 'analysis_content', NEW.analysis_content,
      'content', NEW.content, 'created_at', NEW.created_at, 'last_modified', NEW.last_modified,
      'version', NEW.version, 'client_id', NEW.client_id, 'deleted', NEW.deleted));
END;
"""



























#Version 2
"""
Okay, let's tackle point #1 first: Enforce metadata integrity at DB-layer.

We'll add BEFORE UPDATE triggers to the six syncable tables (Media, Keywords, Transcripts, MediaChunks, UnvectorizedMediaChunks, DocumentVersions). These triggers will run before any update operation completes and will abort the transaction if the validation rules are violated.

Validation Rules:

    NEW.version must be exactly OLD.version + 1.

    NEW.client_id must not be NULL and not be an empty string ('').

These triggers ensure that even if the application code calling the library has a bug (forgets to increment version or provide client_id), the database itself prevents the inconsistent data from being saved, maintaining the integrity required for synchronization.

Here is the full updated schema including the original structure plus the new validation triggers added at the end:
--------------------------------------------------------------------------------------------------------------------------------------------------------------
Placement: The new validation triggers are added at the very end of the schema script.

BEFORE UPDATE: They use BEFORE UPDATE to check the data before the update is allowed to proceed.

RAISE(ABORT, '...'): If a validation condition (WHERE ...) is met, this function immediately stops the UPDATE operation and rolls back the current transaction (if any), returning the error message.

Checks:

    NEW.version IS NOT OLD.version + 1: Ensures the sync version number is strictly incremented by one.

    NEW.client_id IS NULL OR NEW.client_id = '': Ensures a non-empty client_id is provided with the update.

Coverage: These triggers cover all six tables that contain the sync metadata columns and are expected to be modified directly via UPDATE statements requiring version increments.

No WHEN Clause: These validation triggers apply to all UPDATE stateme
--------------------------------------------------------------------------------------------------------------------------------------------------------------


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
    ingestion_date DATETIME, -- Changed to DATETIME
    transcription_model TEXT,
    is_trash BOOLEAN DEFAULT 0 NOT NULL, -- For UI Trash Can feature
    trash_date DATETIME,                 -- Changed to DATETIME
    vector_embedding BLOB,               -- Often excluded from sync payload due to size/volatility (Local Only)
    chunking_status TEXT DEFAULT 'pending' NOT NULL, -- Likely managed locally, exclude from sync (Local Only)
    vector_processing INTEGER DEFAULT 0 NOT NULL,    -- Likely managed locally, exclude from sync (Local Only)
    content_hash TEXT UNIQUE NOT NULL,

    -- Sync Metadata Columns --
    uuid TEXT UNIQUE NOT NULL, -- Globally unique identifier for sync
    last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, -- App should ideally set this explicitly on change
    version INTEGER NOT NULL DEFAULT 1,                        -- App must increment this on change
    client_id TEXT NOT NULL,                                   -- Identifier of the client that made the last change (Set by App)
    deleted BOOLEAN NOT NULL DEFAULT 0                         -- Soft delete flag for sync
);
-- Original Indices
CREATE INDEX IF NOT EXISTS idx_media_title ON Media(title);
CREATE INDEX IF NOT EXISTS idx_media_type ON Media(type);
CREATE INDEX IF NOT EXISTS idx_media_author ON Media(author);
CREATE INDEX IF NOT EXISTS idx_media_ingestion_date ON Media(ingestion_date);
CREATE INDEX IF NOT EXISTS idx_media_chunking_status ON Media(chunking_status); -- For local queries
CREATE INDEX IF NOT EXISTS idx_media_vector_processing ON Media(vector_processing); -- For local queries
CREATE INDEX IF NOT EXISTS idx_media_is_trash ON Media(is_trash); -- For UI queries
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
    last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, -- App should ideally set this explicitly on change
    version INTEGER NOT NULL DEFAULT 1,                        -- App must increment this on change
    client_id TEXT NOT NULL,                                   -- Set by App
    deleted BOOLEAN NOT NULL DEFAULT 0
);
-- Original Indices (keyword covered by UNIQUE)
-- Sync Indices
CREATE UNIQUE INDEX IF NOT EXISTS idx_keywords_uuid ON Keywords(uuid);
CREATE INDEX IF NOT EXISTS idx_keywords_last_modified ON Keywords(last_modified);
CREATE INDEX IF NOT EXISTS idx_keywords_deleted ON Keywords(deleted);


-- MediaKeywords Table (Relationship Table - No direct sync metadata needed)
-- ==========================================================================
-- Sync is handled by 'link'/'unlink' operations in sync_log triggered below.
-- NOTE: Soft-deleting Media/Keywords does NOT automatically delete rows here
-- due to use of soft deletes. Application logic must handle unlinking.
CREATE TABLE IF NOT EXISTS MediaKeywords (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    media_id INTEGER NOT NULL,
    keyword_id INTEGER NOT NULL,
    UNIQUE (media_id, keyword_id),
    FOREIGN KEY (media_id) REFERENCES Media(id) ON DELETE CASCADE,     -- Cascade works on HARD delete only
    FOREIGN KEY (keyword_id) REFERENCES Keywords(id) ON DELETE CASCADE -- Cascade works on HARD delete only
);
-- Original Indices
CREATE INDEX IF NOT EXISTS idx_mediakeywords_media_id ON MediaKeywords(media_id);
CREATE INDEX IF NOT EXISTS idx_mediakeywords_keyword_id ON MediaKeywords(keyword_id);


-- Transcripts Table
-- =================
-- NOTE: Soft-deleting parent Media does NOT automatically soft-delete Transcripts.
-- Application/Sync logic must handle this relationship based on Media.deleted flag.
CREATE TABLE IF NOT EXISTS Transcripts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    media_id INTEGER NOT NULL,
    whisper_model TEXT,
    transcription TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, -- Keep original creation time
    UNIQUE (media_id, whisper_model), -- Original constraint
    FOREIGN KEY (media_id) REFERENCES Media(id) ON DELETE CASCADE, -- Cascade works on HARD delete only

    -- Sync Metadata Columns --
    uuid TEXT UNIQUE NOT NULL,
    last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, -- App should ideally set this explicitly on change
    version INTEGER NOT NULL DEFAULT 1,                        -- App must increment this on change
    client_id TEXT NOT NULL,                                   -- Set by App
    deleted BOOLEAN NOT NULL DEFAULT 0
);
-- Original Indices
CREATE INDEX IF NOT EXISTS idx_transcripts_media_id ON Transcripts(media_id);
-- Sync Indices
CREATE UNIQUE INDEX IF NOT EXISTS idx_transcripts_uuid ON Transcripts(uuid);
CREATE INDEX IF NOT EXISTS idx_transcripts_last_modified ON Transcripts(last_modified);
CREATE INDEX IF NOT EXISTS idx_transcripts_deleted ON Transcripts(deleted);


-- MediaChunks Table (Consider if still needed if vector embeddings are not stored here)
-- =================
-- Represents processed/structured chunks of media content.
-- NOTE: Soft-deleting parent Media does NOT automatically soft-delete MediaChunks.
-- Application/Sync logic must handle this relationship based on Media.deleted flag.
CREATE TABLE IF NOT EXISTS MediaChunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    media_id INTEGER NOT NULL,
    chunk_text TEXT,
    start_index INTEGER,
    end_index INTEGER,
    chunk_id TEXT UNIQUE, -- Assumed Globally Unique Identifier for the chunk content/position. If unique per media, use UNIQUE(media_id, chunk_id)
    FOREIGN KEY (media_id) REFERENCES Media(id) ON DELETE CASCADE, -- Cascade works on HARD delete only

    -- Sync Metadata Columns --
    uuid TEXT UNIQUE NOT NULL,
    last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, -- App should ideally set this explicitly on change
    version INTEGER NOT NULL DEFAULT 1,                        -- App must increment this on change
    client_id TEXT NOT NULL,                                   -- Set by App
    deleted BOOLEAN NOT NULL DEFAULT 0
);
-- Original Indices
CREATE INDEX IF NOT EXISTS idx_mediachunks_media_id ON MediaChunks(media_id);
-- Sync Indices
CREATE UNIQUE INDEX IF NOT EXISTS idx_mediachunks_uuid ON MediaChunks(uuid);
CREATE INDEX IF NOT EXISTS idx_mediachunks_last_modified ON MediaChunks(last_modified);
CREATE INDEX IF NOT EXISTS idx_mediachunks_deleted ON MediaChunks(deleted);


-- UnvectorizedMediaChunks Table (Likely precursor to processing/vectorization)
-- =============================
-- Stores chunks identified during initial processing, before vectorization (if any).
-- Can be created asynchronously after Media ingestion.
-- NOTE: Soft-deleting parent Media does NOT automatically soft-delete these chunks.
-- Application/Sync logic must handle this relationship based on Media.deleted flag.
CREATE TABLE IF NOT EXISTS UnvectorizedMediaChunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    media_id INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    start_char INTEGER,
    end_char INTEGER,
    chunk_type TEXT,
    creation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP, -- Keep original creation time
    last_modified_orig TIMESTAMP DEFAULT CURRENT_TIMESTAMP, -- Renamed original timestamp if needed for app logic (Consider removing if sync `last_modified` is sufficient)
    is_processed BOOLEAN DEFAULT FALSE NOT NULL, -- Local state, exclude from sync payload (Local Only)
    metadata TEXT, -- Potentially sync this if it contains relevant info (e.g., user notes)
    UNIQUE (media_id, chunk_index, chunk_type), -- Original constraint
    FOREIGN KEY (media_id) REFERENCES Media(id) ON DELETE CASCADE, -- Cascade works on HARD delete only

    -- Sync Metadata Columns --
    uuid TEXT UNIQUE NOT NULL,
    last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, -- Sync timestamp (App should ideally set explicitly)
    version INTEGER NOT NULL DEFAULT 1,                        -- App must increment this on change
    client_id TEXT NOT NULL,                                   -- Set by App
    deleted BOOLEAN NOT NULL DEFAULT 0
);
-- Original Indices
CREATE INDEX IF NOT EXISTS idx_unvectorized_media_chunks_media_id ON UnvectorizedMediaChunks(media_id);
CREATE INDEX IF NOT EXISTS idx_unvectorized_media_chunks_is_processed ON UnvectorizedMediaChunks(is_processed); -- For local queries
CREATE INDEX IF NOT EXISTS idx_unvectorized_media_chunks_chunk_type ON UnvectorizedMediaChunks(chunk_type);
-- Sync Indices
CREATE UNIQUE INDEX IF NOT EXISTS idx_unvectorizedmediachunks_uuid ON UnvectorizedMediaChunks(uuid);
CREATE INDEX IF NOT EXISTS idx_unvectorizedmediachunks_last_modified ON UnvectorizedMediaChunks(last_modified);
CREATE INDEX IF NOT EXISTS idx_unvectorizedmediachunks_deleted ON UnvectorizedMediaChunks(deleted);


-- DocumentVersions Table
-- ======================
-- Stores snapshots or analysis versions related to a media item.
-- NOTE: Soft-deleting parent Media does NOT automatically soft-delete DocumentVersions.
-- Application/Sync logic must handle this relationship based on Media.deleted flag.
CREATE TABLE IF NOT EXISTS DocumentVersions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    media_id INTEGER NOT NULL,
    version_number INTEGER NOT NULL, -- Local version sequence per media item
    prompt TEXT,
    analysis_content TEXT,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, -- Keep original creation time
    FOREIGN KEY (media_id) REFERENCES Media(id) ON DELETE CASCADE, -- Cascade works on HARD delete only
    UNIQUE (media_id, version_number), -- Original constraint

    -- Sync Metadata Columns --
    uuid TEXT UNIQUE NOT NULL,
    last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, -- App should ideally set this explicitly on change
    version INTEGER NOT NULL DEFAULT 1,                        -- App must increment this on change
    client_id TEXT NOT NULL,                                   -- Set by App
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
-- FTS Triggers for Keywords
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
    timestamp    DATETIME NOT NULL,   -- Matches the record's last_modified or the time of the link/unlink (Changed to DATETIME)
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
-- (Includes specific UNDELETE triggers)
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
    )
  );
END;

DROP TRIGGER IF EXISTS media_sync_update;
CREATE TRIGGER media_sync_update
AFTER UPDATE ON Media
WHEN OLD.deleted = NEW.deleted AND (
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
     ifnull(OLD.last_modified,'') != ifnull(NEW.last_modified,'') OR
     ifnull(OLD.version,0) != ifnull(NEW.version,0) OR
     ifnull(OLD.client_id, '') != ifnull(NEW.client_id, '')
)
BEGIN
  INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
  VALUES (
    'Media', NEW.uuid, 'update',
    NEW.last_modified, NEW.client_id, NEW.version,
    json_object(
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
WHEN OLD.deleted = 0 AND NEW.deleted = 1
BEGIN
  INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
  VALUES (
    'Media', NEW.uuid, 'delete',
    NEW.last_modified, NEW.client_id, NEW.version,
    json_object('uuid', NEW.uuid, 'last_modified', NEW.last_modified, 'version', NEW.version, 'client_id', NEW.client_id)
  );
END;

DROP TRIGGER IF EXISTS media_sync_undelete;
CREATE TRIGGER media_sync_undelete
AFTER UPDATE ON Media
WHEN OLD.deleted = 1 AND NEW.deleted = 0
BEGIN
  INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
  VALUES (
    'Media', NEW.uuid, 'update',
    NEW.last_modified, NEW.client_id, NEW.version,
    json_object(
      'uuid', NEW.uuid, 'url', NEW.url, 'title', NEW.title, 'type', NEW.type,
      'content', NEW.content, 'author', NEW.author, 'ingestion_date', NEW.ingestion_date,
      'transcription_model', NEW.transcription_model, 'is_trash', NEW.is_trash, 'trash_date', NEW.trash_date,
      'content_hash', NEW.content_hash, 'last_modified', NEW.last_modified,
      'version', NEW.version, 'client_id', NEW.client_id, 'deleted', NEW.deleted
    )
  );
END;


-- ==========================
-- Triggers for Keywords Table
-- ==========================
DROP TRIGGER IF EXISTS keywords_sync_create;
CREATE TRIGGER keywords_sync_create AFTER INSERT ON Keywords BEGIN
  INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
  VALUES ('Keywords', NEW.uuid, 'create', NEW.last_modified, NEW.client_id, NEW.version,
    json_object('uuid', NEW.uuid, 'keyword', NEW.keyword, 'last_modified', NEW.last_modified, 'version', NEW.version, 'client_id', NEW.client_id, 'deleted', NEW.deleted));
END;

DROP TRIGGER IF EXISTS keywords_sync_update;
CREATE TRIGGER keywords_sync_update AFTER UPDATE ON Keywords
WHEN OLD.deleted = NEW.deleted AND (
    ifnull(OLD.keyword,'') != ifnull(NEW.keyword,'') OR
    ifnull(OLD.last_modified,'') != ifnull(NEW.last_modified,'') OR
    ifnull(OLD.version,0) != ifnull(NEW.version,0) OR
    ifnull(OLD.client_id, '') != ifnull(NEW.client_id, '')
) BEGIN
  INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
  VALUES ('Keywords', NEW.uuid, 'update', NEW.last_modified, NEW.client_id, NEW.version,
     json_object('uuid', NEW.uuid, 'keyword', NEW.keyword, 'last_modified', NEW.last_modified, 'version', NEW.version, 'client_id', NEW.client_id, 'deleted', NEW.deleted));
END;

DROP TRIGGER IF EXISTS keywords_sync_delete;
CREATE TRIGGER keywords_sync_delete AFTER UPDATE ON Keywords
WHEN OLD.deleted = 0 AND NEW.deleted = 1 BEGIN
  INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
  VALUES ('Keywords', NEW.uuid, 'delete', NEW.last_modified, NEW.client_id, NEW.version,
    json_object('uuid', NEW.uuid, 'last_modified', NEW.last_modified, 'version', NEW.version, 'client_id', NEW.client_id));
END;

DROP TRIGGER IF EXISTS keywords_sync_undelete;
CREATE TRIGGER keywords_sync_undelete AFTER UPDATE ON Keywords
WHEN OLD.deleted = 1 AND NEW.deleted = 0 BEGIN
  INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
  VALUES ('Keywords', NEW.uuid, 'update', NEW.last_modified, NEW.client_id, NEW.version,
     json_object('uuid', NEW.uuid, 'keyword', NEW.keyword, 'last_modified', NEW.last_modified, 'version', NEW.version, 'client_id', NEW.client_id, 'deleted', NEW.deleted));
END;


-- =======================================
-- Triggers for MediaKeywords Relationship
-- =======================================
DROP TRIGGER IF EXISTS mediakeywords_sync_link;
CREATE TRIGGER mediakeywords_sync_link
AFTER INSERT ON MediaKeywords
BEGIN
    SELECT RAISE(ABORT, 'Cannot link keyword: Media record not found or missing UUID')
    WHERE NOT EXISTS (SELECT 1 FROM Media WHERE id = NEW.media_id AND uuid IS NOT NULL);
    SELECT RAISE(ABORT, 'Cannot link keyword: Keyword record not found or missing UUID')
    WHERE NOT EXISTS (SELECT 1 FROM Keywords WHERE id = NEW.keyword_id AND uuid IS NOT NULL);

    INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
    SELECT
        'MediaKeywords',
        m.uuid || '_' || k.uuid,
        'link',
        strftime('%Y-%m-%d %H:%M:%S', 'now'), -- Use UTC time 'now'
        m.client_id, -- Convention: Use Media's client_id
        m.version,   -- Convention: Use Media's version
        json_object('media_uuid', m.uuid, 'keyword_uuid', k.uuid)
    FROM Media m, Keywords k
    WHERE m.id = NEW.media_id AND k.id = NEW.keyword_id;
END;

DROP TRIGGER IF EXISTS mediakeywords_sync_unlink;
CREATE TRIGGER mediakeywords_sync_unlink
AFTER DELETE ON MediaKeywords
BEGIN
     INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
     SELECT
         'MediaKeywords',
         ifnull(m.uuid, 'unknown_media_' || OLD.media_id) || '_' || ifnull(k.uuid, 'unknown_keyword_' || OLD.keyword_id),
         'unlink',
         strftime('%Y-%m-%d %H:%M:%S', 'now'), -- Use UTC time 'now'
         ifnull(m.client_id, 'unknown'), -- Best guess
         ifnull(m.version, 0),           -- Best guess
         json_object(
             'media_uuid', ifnull(m.uuid, 'unknown_media_' || OLD.media_id),
             'keyword_uuid', ifnull(k.uuid, 'unknown_keyword_' || OLD.keyword_id)
         )
     FROM (SELECT OLD.media_id as media_id, OLD.keyword_id as keyword_id) AS OldIds
     LEFT JOIN Media m ON m.id = OldIds.media_id
     LEFT JOIN Keywords k ON k.id = OldIds.keyword_id;
END;


-- ==========================
-- Triggers for Transcripts Table
-- ==========================
DROP TRIGGER IF EXISTS transcripts_sync_create;
CREATE TRIGGER transcripts_sync_create AFTER INSERT ON Transcripts BEGIN
  INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
  VALUES ('Transcripts', NEW.uuid, 'create', NEW.last_modified, NEW.client_id, NEW.version,
    json_object('uuid', NEW.uuid, 'media_uuid', (SELECT uuid FROM Media WHERE id = NEW.media_id),
      'whisper_model', NEW.whisper_model, 'transcription', NEW.transcription, 'created_at', NEW.created_at,
      'last_modified', NEW.last_modified, 'version', NEW.version, 'client_id', NEW.client_id, 'deleted', NEW.deleted));
END;

DROP TRIGGER IF EXISTS transcripts_sync_update;
CREATE TRIGGER transcripts_sync_update AFTER UPDATE ON Transcripts
WHEN OLD.deleted = NEW.deleted AND (
    ifnull(OLD.whisper_model,'') != ifnull(NEW.whisper_model,'') OR
    ifnull(OLD.transcription,'') != ifnull(NEW.transcription,'') OR
    ifnull(OLD.last_modified,'') != ifnull(NEW.last_modified,'') OR
    ifnull(OLD.version,0) != ifnull(NEW.version,0) OR
    ifnull(OLD.client_id, '') != ifnull(NEW.client_id, '')
) BEGIN
  INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
  VALUES ('Transcripts', NEW.uuid, 'update', NEW.last_modified, NEW.client_id, NEW.version,
    json_object('uuid', NEW.uuid, 'media_uuid', (SELECT uuid FROM Media WHERE id = NEW.media_id),
      'whisper_model', NEW.whisper_model, 'transcription', NEW.transcription, 'created_at', NEW.created_at,
      'last_modified', NEW.last_modified, 'version', NEW.version, 'client_id', NEW.client_id, 'deleted', NEW.deleted));
END;

DROP TRIGGER IF EXISTS transcripts_sync_delete;
CREATE TRIGGER transcripts_sync_delete AFTER UPDATE ON Transcripts
WHEN OLD.deleted = 0 AND NEW.deleted = 1 BEGIN
  INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
  VALUES ('Transcripts', NEW.uuid, 'delete', NEW.last_modified, NEW.client_id, NEW.version,
    json_object('uuid', NEW.uuid, 'media_uuid', (SELECT uuid FROM Media WHERE id = NEW.media_id), 'last_modified', NEW.last_modified, 'version', NEW.version, 'client_id', NEW.client_id));
END;

DROP TRIGGER IF EXISTS transcripts_sync_undelete;
CREATE TRIGGER transcripts_sync_undelete AFTER UPDATE ON Transcripts
WHEN OLD.deleted = 1 AND NEW.deleted = 0 BEGIN
 INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
  VALUES ('Transcripts', NEW.uuid, 'update', NEW.last_modified, NEW.client_id, NEW.version,
    json_object('uuid', NEW.uuid, 'media_uuid', (SELECT uuid FROM Media WHERE id = NEW.media_id),
      'whisper_model', NEW.whisper_model, 'transcription', NEW.transcription, 'created_at', NEW.created_at,
      'last_modified', NEW.last_modified, 'version', NEW.version, 'client_id', NEW.client_id, 'deleted', NEW.deleted));
END;


-- ==========================
-- Triggers for MediaChunks Table
-- ==========================
DROP TRIGGER IF EXISTS mediachunks_sync_create;
CREATE TRIGGER mediachunks_sync_create AFTER INSERT ON MediaChunks BEGIN
  INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
  VALUES ('MediaChunks', NEW.uuid, 'create', NEW.last_modified, NEW.client_id, NEW.version,
    json_object('uuid', NEW.uuid, 'media_uuid', (SELECT uuid FROM Media WHERE id = NEW.media_id),
      'chunk_text', NEW.chunk_text, 'start_index', NEW.start_index, 'end_index', NEW.end_index, 'chunk_id', NEW.chunk_id,
      'last_modified', NEW.last_modified, 'version', NEW.version, 'client_id', NEW.client_id, 'deleted', NEW.deleted));
END;

DROP TRIGGER IF EXISTS mediachunks_sync_update;
CREATE TRIGGER mediachunks_sync_update AFTER UPDATE ON MediaChunks
WHEN OLD.deleted = NEW.deleted AND (
    ifnull(OLD.chunk_text,'') != ifnull(NEW.chunk_text,'') OR
    ifnull(OLD.start_index,0) != ifnull(NEW.start_index,0) OR
    ifnull(OLD.end_index,0) != ifnull(NEW.end_index,0) OR
    ifnull(OLD.chunk_id,'') != ifnull(NEW.chunk_id,'') OR
    ifnull(OLD.last_modified,'') != ifnull(NEW.last_modified,'') OR
    ifnull(OLD.version,0) != ifnull(NEW.version,0) OR
    ifnull(OLD.client_id, '') != ifnull(NEW.client_id, '')
) BEGIN
  INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
  VALUES ('MediaChunks', NEW.uuid, 'update', NEW.last_modified, NEW.client_id, NEW.version,
    json_object('uuid', NEW.uuid, 'media_uuid', (SELECT uuid FROM Media WHERE id = NEW.media_id),
      'chunk_text', NEW.chunk_text, 'start_index', NEW.start_index, 'end_index', NEW.end_index, 'chunk_id', NEW.chunk_id,
      'last_modified', NEW.last_modified, 'version', NEW.version, 'client_id', NEW.client_id, 'deleted', NEW.deleted));
END;

DROP TRIGGER IF EXISTS mediachunks_sync_delete;
CREATE TRIGGER mediachunks_sync_delete AFTER UPDATE ON MediaChunks
WHEN OLD.deleted = 0 AND NEW.deleted = 1 BEGIN
  INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
  VALUES ('MediaChunks', NEW.uuid, 'delete', NEW.last_modified, NEW.client_id, NEW.version,
    json_object('uuid', NEW.uuid, 'media_uuid', (SELECT uuid FROM Media WHERE id = NEW.media_id), 'last_modified', NEW.last_modified, 'version', NEW.version, 'client_id', NEW.client_id));
END;

DROP TRIGGER IF EXISTS mediachunks_sync_undelete;
CREATE TRIGGER mediachunks_sync_undelete AFTER UPDATE ON MediaChunks
WHEN OLD.deleted = 1 AND NEW.deleted = 0 BEGIN
  INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
  VALUES ('MediaChunks', NEW.uuid, 'update', NEW.last_modified, NEW.client_id, NEW.version,
    json_object('uuid', NEW.uuid, 'media_uuid', (SELECT uuid FROM Media WHERE id = NEW.media_id),
      'chunk_text', NEW.chunk_text, 'start_index', NEW.start_index, 'end_index', NEW.end_index, 'chunk_id', NEW.chunk_id,
      'last_modified', NEW.last_modified, 'version', NEW.version, 'client_id', NEW.client_id, 'deleted', NEW.deleted));
END;


-- =======================================
-- Triggers for UnvectorizedMediaChunks Table
-- =======================================
DROP TRIGGER IF EXISTS unvectorizedmediachunks_sync_create;
CREATE TRIGGER unvectorizedmediachunks_sync_create AFTER INSERT ON UnvectorizedMediaChunks BEGIN
  INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
  VALUES ('UnvectorizedMediaChunks', NEW.uuid, 'create', NEW.last_modified, NEW.client_id, NEW.version,
    json_object('uuid', NEW.uuid, 'media_uuid', (SELECT uuid FROM Media WHERE id = NEW.media_id),
      'chunk_text', NEW.chunk_text, 'chunk_index', NEW.chunk_index, 'start_char', NEW.start_char,
      'end_char', NEW.end_char, 'chunk_type', NEW.chunk_type, 'creation_date', NEW.creation_date,
      'metadata', NEW.metadata,
      'last_modified', NEW.last_modified, 'version', NEW.version, 'client_id', NEW.client_id, 'deleted', NEW.deleted
      ));
END;

DROP TRIGGER IF EXISTS unvectorizedmediachunks_sync_update;
CREATE TRIGGER unvectorizedmediachunks_sync_update AFTER UPDATE ON UnvectorizedMediaChunks
WHEN OLD.deleted = NEW.deleted AND (
    ifnull(OLD.chunk_text,'') != ifnull(NEW.chunk_text,'') OR
    ifnull(OLD.chunk_index,0) != ifnull(NEW.chunk_index,0) OR
    ifnull(OLD.start_char,0) != ifnull(NEW.start_char,0) OR
    ifnull(OLD.end_char,0) != ifnull(NEW.end_char,0) OR
    ifnull(OLD.chunk_type,'') != ifnull(NEW.chunk_type,'') OR
    ifnull(OLD.metadata,'') != ifnull(NEW.metadata,'') OR
    ifnull(OLD.last_modified,'') != ifnull(NEW.last_modified,'') OR
    ifnull(OLD.version,0) != ifnull(NEW.version,0) OR
    ifnull(OLD.client_id, '') != ifnull(NEW.client_id, '')
) BEGIN
  INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
  VALUES ('UnvectorizedMediaChunks', NEW.uuid, 'update', NEW.last_modified, NEW.client_id, NEW.version,
    json_object('uuid', NEW.uuid, 'media_uuid', (SELECT uuid FROM Media WHERE id = NEW.media_id),
      'chunk_text', NEW.chunk_text, 'chunk_index', NEW.chunk_index, 'start_char', NEW.start_char,
      'end_char', NEW.end_char, 'chunk_type', NEW.chunk_type, 'creation_date', NEW.creation_date,
      'metadata', NEW.metadata,
      'last_modified', NEW.last_modified, 'version', NEW.version, 'client_id', NEW.client_id, 'deleted', NEW.deleted
      ));
END;

DROP TRIGGER IF EXISTS unvectorizedmediachunks_sync_delete;
CREATE TRIGGER unvectorizedmediachunks_sync_delete AFTER UPDATE ON UnvectorizedMediaChunks
WHEN OLD.deleted = 0 AND NEW.deleted = 1 BEGIN
  INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
  VALUES ('UnvectorizedMediaChunks', NEW.uuid, 'delete', NEW.last_modified, NEW.client_id, NEW.version,
    json_object('uuid', NEW.uuid, 'media_uuid', (SELECT uuid FROM Media WHERE id = NEW.media_id), 'last_modified', NEW.last_modified, 'version', NEW.version, 'client_id', NEW.client_id));
END;

DROP TRIGGER IF EXISTS unvectorizedmediachunks_sync_undelete;
CREATE TRIGGER unvectorizedmediachunks_sync_undelete AFTER UPDATE ON UnvectorizedMediaChunks
WHEN OLD.deleted = 1 AND NEW.deleted = 0 BEGIN
  INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
  VALUES ('UnvectorizedMediaChunks', NEW.uuid, 'update', NEW.last_modified, NEW.client_id, NEW.version,
    json_object('uuid', NEW.uuid, 'media_uuid', (SELECT uuid FROM Media WHERE id = NEW.media_id),
      'chunk_text', NEW.chunk_text, 'chunk_index', NEW.chunk_index, 'start_char', NEW.start_char,
      'end_char', NEW.end_char, 'chunk_type', NEW.chunk_type, 'creation_date', NEW.creation_date,
      'metadata', NEW.metadata,
      'last_modified', NEW.last_modified, 'version', NEW.version, 'client_id', NEW.client_id, 'deleted', NEW.deleted
      ));
END;


-- ==================================
-- Triggers for DocumentVersions Table
-- ==================================
DROP TRIGGER IF EXISTS documentversions_sync_create;
CREATE TRIGGER documentversions_sync_create AFTER INSERT ON DocumentVersions BEGIN
  INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
  VALUES ('DocumentVersions', NEW.uuid, 'create', NEW.last_modified, NEW.client_id, NEW.version,
    json_object('uuid', NEW.uuid, 'media_uuid', (SELECT uuid FROM Media WHERE id = NEW.media_id),
      'version_number', NEW.version_number, 'prompt', NEW.prompt, 'analysis_content', NEW.analysis_content,
      'content', NEW.content, 'created_at', NEW.created_at, 'last_modified', NEW.last_modified,
      'version', NEW.version, 'client_id', NEW.client_id, 'deleted', NEW.deleted));
END;

DROP TRIGGER IF EXISTS documentversions_sync_update;
CREATE TRIGGER documentversions_sync_update AFTER UPDATE ON DocumentVersions
WHEN OLD.deleted = NEW.deleted AND (
    ifnull(OLD.prompt,'') != ifnull(NEW.prompt,'') OR
    ifnull(OLD.analysis_content,'') != ifnull(NEW.analysis_content,'') OR
    ifnull(OLD.content,'') != ifnull(NEW.content,'') OR
    ifnull(OLD.version_number,0) != ifnull(NEW.version_number,0) OR
    ifnull(OLD.last_modified,'') != ifnull(NEW.last_modified,'') OR
    ifnull(OLD.version,0) != ifnull(NEW.version,0) OR
    ifnull(OLD.client_id, '') != ifnull(NEW.client_id, '')
) BEGIN
  INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
  VALUES ('DocumentVersions', NEW.uuid, 'update', NEW.last_modified, NEW.client_id, NEW.version,
    json_object('uuid', NEW.uuid, 'media_uuid', (SELECT uuid FROM Media WHERE id = NEW.media_id),
      'version_number', NEW.version_number, 'prompt', NEW.prompt, 'analysis_content', NEW.analysis_content,
      'content', NEW.content, 'created_at', NEW.created_at, 'last_modified', NEW.last_modified,
      'version', NEW.version, 'client_id', NEW.client_id, 'deleted', NEW.deleted));
END;

DROP TRIGGER IF EXISTS documentversions_sync_delete;
CREATE TRIGGER documentversions_sync_delete AFTER UPDATE ON DocumentVersions
WHEN OLD.deleted = 0 AND NEW.deleted = 1 BEGIN
  INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
  VALUES ('DocumentVersions', NEW.uuid, 'delete', NEW.last_modified, NEW.client_id, NEW.version,
    json_object('uuid', NEW.uuid, 'media_uuid', (SELECT uuid FROM Media WHERE id = NEW.media_id), 'last_modified', NEW.last_modified, 'version', NEW.version, 'client_id', NEW.client_id));
END;

DROP TRIGGER IF EXISTS documentversions_sync_undelete;
CREATE TRIGGER documentversions_sync_undelete AFTER UPDATE ON DocumentVersions
WHEN OLD.deleted = 1 AND NEW.deleted = 0 BEGIN
  INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
  VALUES ('DocumentVersions', NEW.uuid, 'update', NEW.last_modified, NEW.client_id, NEW.version,
    json_object('uuid', NEW.uuid, 'media_uuid', (SELECT uuid FROM Media WHERE id = NEW.media_id),
      'version_number', NEW.version_number, 'prompt', NEW.prompt, 'analysis_content', NEW.analysis_content,
      'content', NEW.content, 'created_at', NEW.created_at, 'last_modified', NEW.last_modified,
      'version', NEW.version, 'client_id', NEW.client_id, 'deleted', NEW.deleted));
END;

-- ───────────────────────────────────────────────────────────────────────────
-- Synchronization Data Validation Triggers (NEW)
-- ───────────────────────────────────────────────────────────────────────────

-- ========================
-- Validation for Media Table
-- ========================
DROP TRIGGER IF EXISTS media_validate_sync_update;
CREATE TRIGGER media_validate_sync_update
BEFORE UPDATE ON Media
BEGIN
    -- Check version increment (must be exactly +1)
    SELECT RAISE(ABORT, 'Sync Error (Media): Version must increment by exactly 1.')
    WHERE NEW.version IS NOT OLD.version + 1;

    -- Check client_id presence
    SELECT RAISE(ABORT, 'Sync Error (Media): Client ID cannot be NULL or empty.')
    WHERE NEW.client_id IS NULL OR NEW.client_id = '';

    -- Optional: Check timestamp is not older (helps prevent accidental regressions)
    -- SELECT RAISE(ABORT, 'Sync Error (Media): New last_modified timestamp cannot be older than the existing one.')
    -- WHERE NEW.last_modified < OLD.last_modified;
END;

-- ==========================
-- Validation for Keywords Table
-- ==========================
DROP TRIGGER IF EXISTS keywords_validate_sync_update;
CREATE TRIGGER keywords_validate_sync_update
BEFORE UPDATE ON Keywords
BEGIN
    SELECT RAISE(ABORT, 'Sync Error (Keywords): Version must increment by exactly 1.')
    WHERE NEW.version IS NOT OLD.version + 1;

    SELECT RAISE(ABORT, 'Sync Error (Keywords): Client ID cannot be NULL or empty.')
    WHERE NEW.client_id IS NULL OR NEW.client_id = '';
END;

-- ==========================
-- Validation for Transcripts Table
-- ==========================
DROP TRIGGER IF EXISTS transcripts_validate_sync_update;
CREATE TRIGGER transcripts_validate_sync_update
BEFORE UPDATE ON Transcripts
BEGIN
    SELECT RAISE(ABORT, 'Sync Error (Transcripts): Version must increment by exactly 1.')
    WHERE NEW.version IS NOT OLD.version + 1;

    SELECT RAISE(ABORT, 'Sync Error (Transcripts): Client ID cannot be NULL or empty.')
    WHERE NEW.client_id IS NULL OR NEW.client_id = '';
END;

-- ==========================
-- Validation for MediaChunks Table
-- ==========================
DROP TRIGGER IF EXISTS mediachunks_validate_sync_update;
CREATE TRIGGER mediachunks_validate_sync_update
BEFORE UPDATE ON MediaChunks
BEGIN
    SELECT RAISE(ABORT, 'Sync Error (MediaChunks): Version must increment by exactly 1.')
    WHERE NEW.version IS NOT OLD.version + 1;

    SELECT RAISE(ABORT, 'Sync Error (MediaChunks): Client ID cannot be NULL or empty.')
    WHERE NEW.client_id IS NULL OR NEW.client_id = '';
END;

-- =======================================
-- Validation for UnvectorizedMediaChunks Table
-- =======================================
DROP TRIGGER IF EXISTS unvectorizedmediachunks_validate_sync_update;
CREATE TRIGGER unvectorizedmediachunks_validate_sync_update
BEFORE UPDATE ON UnvectorizedMediaChunks
BEGIN
    SELECT RAISE(ABORT, 'Sync Error (UnvectorizedMediaChunks): Version must increment by exactly 1.')
    WHERE NEW.version IS NOT OLD.version + 1;

    SELECT RAISE(ABORT, 'Sync Error (UnvectorizedMediaChunks): Client ID cannot be NULL or empty.')
    WHERE NEW.client_id IS NULL OR NEW.client_id = '';
END;

-- ==================================
-- Validation for DocumentVersions Table
-- ==================================
DROP TRIGGER IF EXISTS documentversions_validate_sync_update;
CREATE TRIGGER documentversions_validate_sync_update
BEFORE UPDATE ON DocumentVersions
BEGIN
    SELECT RAISE(ABORT, 'Sync Error (DocumentVersions): Version must increment by exactly 1.')
    WHERE NEW.version IS NOT OLD.version + 1;

    SELECT RAISE(ABORT, 'Sync Error (DocumentVersions): Client ID cannot be NULL or empty.')
    WHERE NEW.client_id IS NULL OR NEW.client_id = '';
END;
"""