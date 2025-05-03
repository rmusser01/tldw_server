-- migrations/versions/001_initial_schema.sql
-- This file contains the base tables, indices, FTS tables, and validation triggers, but **NO** sync log triggers and **NO** FTS modification triggers.
    PRAGMA foreign_keys = ON;

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
        content_hash TEXT UNIQUE NOT NULL,
        uuid TEXT UNIQUE NOT NULL,
        last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        version INTEGER NOT NULL DEFAULT 1,
        client_id TEXT NOT NULL,
        deleted BOOLEAN NOT NULL DEFAULT 0,
        prev_version INTEGER,
        merge_parent_uuid TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_media_title ON Media(title);
    CREATE INDEX IF NOT EXISTS idx_media_type ON Media(type);
    CREATE INDEX IF NOT EXISTS idx_media_author ON Media(author);
    CREATE INDEX IF NOT EXISTS idx_media_ingestion_date ON Media(ingestion_date);
    CREATE INDEX IF NOT EXISTS idx_media_chunking_status ON Media(chunking_status);
    CREATE INDEX IF NOT EXISTS idx_media_vector_processing ON Media(vector_processing);
    CREATE INDEX IF NOT EXISTS idx_media_is_trash ON Media(is_trash);
    CREATE UNIQUE INDEX IF NOT EXISTS idx_media_content_hash ON Media(content_hash);
    CREATE UNIQUE INDEX IF NOT EXISTS idx_media_uuid ON Media(uuid);
    CREATE INDEX IF NOT EXISTS idx_media_last_modified ON Media(last_modified);
    CREATE INDEX IF NOT EXISTS idx_media_deleted ON Media(deleted);
    CREATE INDEX IF NOT EXISTS idx_media_prev_version ON Media(prev_version);
    CREATE INDEX IF NOT EXISTS idx_media_merge_parent_uuid ON Media(merge_parent_uuid);

    -- Keywords Table --
    CREATE TABLE IF NOT EXISTS Keywords (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        keyword TEXT NOT NULL UNIQUE COLLATE NOCASE,
        uuid TEXT UNIQUE NOT NULL,
        last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        version INTEGER NOT NULL DEFAULT 1,
        client_id TEXT NOT NULL,
        deleted BOOLEAN NOT NULL DEFAULT 0,
        prev_version INTEGER,
        merge_parent_uuid TEXT
    );
    CREATE UNIQUE INDEX IF NOT EXISTS idx_keywords_uuid ON Keywords(uuid);
    CREATE INDEX IF NOT EXISTS idx_keywords_last_modified ON Keywords(last_modified);
    CREATE INDEX IF NOT EXISTS idx_keywords_deleted ON Keywords(deleted);
    CREATE INDEX IF NOT EXISTS idx_keywords_prev_version ON Keywords(prev_version);
    CREATE INDEX IF NOT EXISTS idx_keywords_merge_parent_uuid ON Keywords(merge_parent_uuid);

    -- MediaKeywords Table (Junction Table) --
    CREATE TABLE IF NOT EXISTS MediaKeywords (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        media_id INTEGER NOT NULL,
        keyword_id INTEGER NOT NULL,
        UNIQUE (media_id, keyword_id),
        FOREIGN KEY (media_id) REFERENCES Media(id) ON DELETE CASCADE,
        FOREIGN KEY (keyword_id) REFERENCES Keywords(id) ON DELETE CASCADE
    );
    CREATE INDEX IF NOT EXISTS idx_mediakeywords_media_id ON MediaKeywords(media_id);
    CREATE INDEX IF NOT EXISTS idx_mediakeywords_keyword_id ON MediaKeywords(keyword_id);

    -- Transcripts Table --
    CREATE TABLE IF NOT EXISTS Transcripts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        media_id INTEGER NOT NULL,
        whisper_model TEXT,
        transcription TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        uuid TEXT UNIQUE NOT NULL,
        last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        version INTEGER NOT NULL DEFAULT 1,
        client_id TEXT NOT NULL,
        deleted BOOLEAN NOT NULL DEFAULT 0,
        prev_version INTEGER,
        merge_parent_uuid TEXT,
        UNIQUE (media_id, whisper_model),
        FOREIGN KEY (media_id) REFERENCES Media(id) ON DELETE CASCADE
    );
    CREATE INDEX IF NOT EXISTS idx_transcripts_media_id ON Transcripts(media_id);
    CREATE UNIQUE INDEX IF NOT EXISTS idx_transcripts_uuid ON Transcripts(uuid);
    CREATE INDEX IF NOT EXISTS idx_transcripts_last_modified ON Transcripts(last_modified);
    CREATE INDEX IF NOT EXISTS idx_transcripts_deleted ON Transcripts(deleted);
    CREATE INDEX IF NOT EXISTS idx_transcripts_prev_version ON Transcripts(prev_version);
    CREATE INDEX IF NOT EXISTS idx_transcripts_merge_parent_uuid ON Transcripts(merge_parent_uuid);

    -- MediaChunks Table --
    CREATE TABLE IF NOT EXISTS MediaChunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        media_id INTEGER NOT NULL,
        chunk_text TEXT NOT NULL,
        start_index INTEGER,
        end_index INTEGER,
        chunk_id TEXT UNIQUE,
        uuid TEXT UNIQUE NOT NULL,
        last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        version INTEGER NOT NULL DEFAULT 1,
        client_id TEXT NOT NULL,
        deleted BOOLEAN NOT NULL DEFAULT 0,
        prev_version INTEGER,
        merge_parent_uuid TEXT,
        FOREIGN KEY (media_id) REFERENCES Media(id) ON DELETE CASCADE
    );
    CREATE INDEX IF NOT EXISTS idx_mediachunks_media_id ON MediaChunks(media_id);
    CREATE UNIQUE INDEX IF NOT EXISTS idx_mediachunks_uuid ON MediaChunks(uuid);
    CREATE INDEX IF NOT EXISTS idx_mediachunks_last_modified ON MediaChunks(last_modified);
    CREATE INDEX IF NOT EXISTS idx_mediachunks_deleted ON MediaChunks(deleted);
    CREATE INDEX IF NOT EXISTS idx_mediachunks_prev_version ON MediaChunks(prev_version);
    CREATE INDEX IF NOT EXISTS idx_mediachunks_merge_parent_uuid ON MediaChunks(merge_parent_uuid);

    -- UnvectorizedMediaChunks Table --
    CREATE TABLE IF NOT EXISTS UnvectorizedMediaChunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        media_id INTEGER NOT NULL,
        chunk_text TEXT NOT NULL,
        chunk_index INTEGER NOT NULL,
        start_char INTEGER,
        end_char INTEGER,
        chunk_type TEXT,
        creation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_modified_orig TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        is_processed BOOLEAN DEFAULT FALSE NOT NULL,
        metadata TEXT,
        uuid TEXT UNIQUE NOT NULL,
        last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        version INTEGER NOT NULL DEFAULT 1,
        client_id TEXT NOT NULL,
        deleted BOOLEAN NOT NULL DEFAULT 0,
        prev_version INTEGER,
        merge_parent_uuid TEXT,
        UNIQUE (media_id, chunk_index, chunk_type),
        FOREIGN KEY (media_id) REFERENCES Media(id) ON DELETE CASCADE
    );
    CREATE INDEX IF NOT EXISTS idx_unvectorized_media_chunks_media_id ON UnvectorizedMediaChunks(media_id);
    CREATE INDEX IF NOT EXISTS idx_unvectorized_media_chunks_is_processed ON UnvectorizedMediaChunks(is_processed);
    CREATE INDEX IF NOT EXISTS idx_unvectorized_media_chunks_chunk_type ON UnvectorizedMediaChunks(chunk_type);
    CREATE UNIQUE INDEX IF NOT EXISTS idx_unvectorizedmediachunks_uuid ON UnvectorizedMediaChunks(uuid);
    CREATE INDEX IF NOT EXISTS idx_unvectorizedmediachunks_last_modified ON UnvectorizedMediaChunks(last_modified);
    CREATE INDEX IF NOT EXISTS idx_unvectorizedmediachunks_deleted ON UnvectorizedMediaChunks(deleted);
    CREATE INDEX IF NOT EXISTS idx_unvectorizedmediachunks_prev_version ON UnvectorizedMediaChunks(prev_version);
    CREATE INDEX IF NOT EXISTS idx_unvectorizedmediachunks_merge_parent_uuid ON UnvectorizedMediaChunks(merge_parent_uuid);

    -- DocumentVersions Table --
    CREATE TABLE IF NOT EXISTS DocumentVersions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        media_id INTEGER NOT NULL,
        version_number INTEGER NOT NULL,
        prompt TEXT,
        analysis_content TEXT,
        content TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        uuid TEXT UNIQUE NOT NULL,
        last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        version INTEGER NOT NULL DEFAULT 1,
        client_id TEXT NOT NULL,
        deleted BOOLEAN NOT NULL DEFAULT 0,
        prev_version INTEGER,
        merge_parent_uuid TEXT,
        FOREIGN KEY (media_id) REFERENCES Media(id) ON DELETE CASCADE,
        UNIQUE (media_id, version_number)
    );
    CREATE INDEX IF NOT EXISTS idx_document_versions_media_id ON DocumentVersions(media_id);
    CREATE INDEX IF NOT EXISTS idx_document_versions_version_number ON DocumentVersions(version_number);
    CREATE UNIQUE INDEX IF NOT EXISTS idx_documentversions_uuid ON DocumentVersions(uuid);
    CREATE INDEX IF NOT EXISTS idx_documentversions_last_modified ON DocumentVersions(last_modified);
    CREATE INDEX IF NOT EXISTS idx_documentversions_deleted ON DocumentVersions(deleted);
    CREATE INDEX IF NOT EXISTS idx_documentversions_prev_version ON DocumentVersions(prev_version);
    CREATE INDEX IF NOT EXISTS idx_documentversions_merge_parent_uuid ON DocumentVersions(merge_parent_uuid);

    -- FTS Tables --
    CREATE VIRTUAL TABLE IF NOT EXISTS media_fts USING fts5(title, content, content='Media', content_rowid='id');
    CREATE VIRTUAL TABLE IF NOT EXISTS keyword_fts USING fts5(keyword, content='Keywords', content_rowid='id');

    -- Validation Triggers --
    DROP TRIGGER IF EXISTS media_validate_sync_update;
    CREATE TRIGGER media_validate_sync_update BEFORE UPDATE ON Media BEGIN SELECT RAISE(ABORT, 'Sync Error (Media): Version must increment by exactly 1.') WHERE NEW.version IS NOT OLD.version + 1; SELECT RAISE(ABORT, 'Sync Error (Media): Client ID cannot be NULL or empty.') WHERE NEW.client_id IS NULL OR NEW.client_id = ''; END;
    DROP TRIGGER IF EXISTS keywords_validate_sync_update;
    CREATE TRIGGER keywords_validate_sync_update BEFORE UPDATE ON Keywords BEGIN SELECT RAISE(ABORT, 'Sync Error (Keywords): Version must increment by exactly 1.') WHERE NEW.version IS NOT OLD.version + 1; SELECT RAISE(ABORT, 'Sync Error (Keywords): Client ID cannot be NULL or empty.') WHERE NEW.client_id IS NULL OR NEW.client_id = ''; END;
    DROP TRIGGER IF EXISTS transcripts_validate_sync_update;
    CREATE TRIGGER transcripts_validate_sync_update BEFORE UPDATE ON Transcripts BEGIN SELECT RAISE(ABORT, 'Sync Error (Transcripts): Version must increment by exactly 1.') WHERE NEW.version IS NOT OLD.version + 1; SELECT RAISE(ABORT, 'Sync Error (Transcripts): Client ID cannot be NULL or empty.') WHERE NEW.client_id IS NULL OR NEW.client_id = ''; END;
    DROP TRIGGER IF EXISTS mediachunks_validate_sync_update;
    CREATE TRIGGER mediachunks_validate_sync_update BEFORE UPDATE ON MediaChunks BEGIN SELECT RAISE(ABORT, 'Sync Error (MediaChunks): Version must increment by exactly 1.') WHERE NEW.version IS NOT OLD.version + 1; SELECT RAISE(ABORT, 'Sync Error (MediaChunks): Client ID cannot be NULL or empty.') WHERE NEW.client_id IS NULL OR NEW.client_id = ''; END;
    DROP TRIGGER IF EXISTS unvectorizedmediachunks_validate_sync_update;
    CREATE TRIGGER unvectorizedmediachunks_validate_sync_update BEFORE UPDATE ON UnvectorizedMediaChunks BEGIN SELECT RAISE(ABORT, 'Sync Error (UnvectorizedMediaChunks): Version must increment by exactly 1.') WHERE NEW.version IS NOT OLD.version + 1; SELECT RAISE(ABORT, 'Sync Error (UnvectorizedMediaChunks): Client ID cannot be NULL or empty.') WHERE NEW.client_id IS NULL OR NEW.client_id = ''; END;
    DROP TRIGGER IF EXISTS documentversions_validate_sync_update;
    CREATE TRIGGER documentversions_validate_sync_update BEFORE UPDATE ON DocumentVersions BEGIN SELECT RAISE(ABORT, 'Sync Error (DocumentVersions): Version must increment by exactly 1.') WHERE NEW.version IS NOT OLD.version + 1; SELECT RAISE(ABORT, 'Sync Error (DocumentVersions): Client ID cannot be NULL or empty.') WHERE NEW.client_id IS NULL OR NEW.client_id = ''; END;