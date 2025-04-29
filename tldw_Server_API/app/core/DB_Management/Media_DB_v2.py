# SQLite_DB.py (Refactored for Multi-DB Instances & Internal Sync Meta)
#########################################
# SQLite_DB Library
# Manages SQLite DB operations for specific instances, handling sync metadata internally.
# Requires a client_id during Database initialization.
# Standalone functions require a Database instance passed as an argument.
####
import configparser
import csv
import hashlib
import html
import os
import queue # Keep if chunk queue logic is used elsewhere
import re
import shutil
import sqlite3
import threading
import time
import traceback
import uuid # For UUID generation
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta # Use timezone-aware UTC
from math import ceil
from typing import List, Tuple, Dict, Any, Optional, Type

# Third-Party Libraries (Ensure these are installed if used)
# import gradio as gr # Removed if Gradio interfaces moved out
# import pandas as pd # Removed if Pandas formatting moved out
# import yaml # Keep if Obsidian import uses it

# --- Logging Setup ---
# Assume logger is configured elsewhere or use basic config:
import logging

import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Custom Exceptions ---
class DatabaseError(Exception):
    """Base exception for database related errors."""
    pass

class InputError(ValueError):
    """Custom exception for input validation errors."""
    pass

class ConflictError(DatabaseError):
    """Indicates a conflict due to concurrent modification (version mismatch)."""
    def __init__(self, message="Conflict detected: Record modified concurrently.", entity=None, identifier=None):
        super().__init__(message)
        self.entity = entity
        self.identifier = identifier # Can be id or uuid

    def __str__(self):
        base = super().__str__()
        details = []
        if self.entity: details.append(f"Entity: {self.entity}")
        if self.identifier: details.append(f"ID: {self.identifier}")
        return f"{base} ({', '.join(details)})" if details else base

# --- Database Class ---
class Database:
    """
    Manages SQLite connection and operations for a specific database file,
    handling sync metadata internally. Requires client_id on initialization.
    """
    # <<< Full Schema with Validation Triggers >>>
    _SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

-- Media Table --
CREATE TABLE IF NOT EXISTS Media (
    id INTEGER PRIMARY KEY AUTOINCREMENT, url TEXT UNIQUE, title TEXT NOT NULL, type TEXT NOT NULL, content TEXT,
    author TEXT, ingestion_date DATETIME, transcription_model TEXT, is_trash BOOLEAN DEFAULT 0 NOT NULL,
    trash_date DATETIME, vector_embedding BLOB, chunking_status TEXT DEFAULT 'pending' NOT NULL,
    vector_processing INTEGER DEFAULT 0 NOT NULL, content_hash TEXT UNIQUE NOT NULL,
    uuid TEXT UNIQUE NOT NULL,
    last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    version INTEGER NOT NULL DEFAULT 1,
    client_id TEXT NOT NULL,
    deleted BOOLEAN NOT NULL DEFAULT 0,
    prev_version INTEGER, -- Added for conflict resolution
    merge_parent_uuid TEXT -- Added for conflict resolution (e.g., 3-way merge)
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
CREATE INDEX IF NOT EXISTS idx_media_prev_version ON Media(prev_version); -- Index for new column
CREATE INDEX IF NOT EXISTS idx_media_merge_parent_uuid ON Media(merge_parent_uuid); -- Index for new column

-- Keywords Table --
CREATE TABLE IF NOT EXISTS Keywords (
    id INTEGER PRIMARY KEY AUTOINCREMENT, keyword TEXT NOT NULL UNIQUE COLLATE NOCASE,
    uuid TEXT UNIQUE NOT NULL,
    last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    version INTEGER NOT NULL DEFAULT 1,
    client_id TEXT NOT NULL,
    deleted BOOLEAN NOT NULL DEFAULT 0,
    prev_version INTEGER, -- Added
    merge_parent_uuid TEXT -- Added
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_keywords_uuid ON Keywords(uuid);
CREATE INDEX IF NOT EXISTS idx_keywords_last_modified ON Keywords(last_modified);
CREATE INDEX IF NOT EXISTS idx_keywords_deleted ON Keywords(deleted);
CREATE INDEX IF NOT EXISTS idx_keywords_prev_version ON Keywords(prev_version); -- Index for new column
CREATE INDEX IF NOT EXISTS idx_keywords_merge_parent_uuid ON Keywords(merge_parent_uuid); -- Index for new column

-- MediaKeywords Table (Junction Table - No sync metadata needed here usually) --
CREATE TABLE IF NOT EXISTS MediaKeywords (
    id INTEGER PRIMARY KEY AUTOINCREMENT, media_id INTEGER NOT NULL, keyword_id INTEGER NOT NULL,
    UNIQUE (media_id, keyword_id),
    FOREIGN KEY (media_id) REFERENCES Media(id) ON DELETE CASCADE,
    FOREIGN KEY (keyword_id) REFERENCES Keywords(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_mediakeywords_media_id ON MediaKeywords(media_id);
CREATE INDEX IF NOT EXISTS idx_mediakeywords_keyword_id ON MediaKeywords(keyword_id);

-- Transcripts Table --
CREATE TABLE IF NOT EXISTS Transcripts (
    id INTEGER PRIMARY KEY AUTOINCREMENT, media_id INTEGER NOT NULL, whisper_model TEXT, transcription TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, UNIQUE (media_id, whisper_model),
    FOREIGN KEY (media_id) REFERENCES Media(id) ON DELETE CASCADE,
    uuid TEXT UNIQUE NOT NULL,
    last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    version INTEGER NOT NULL DEFAULT 1,
    client_id TEXT NOT NULL,
    deleted BOOLEAN NOT NULL DEFAULT 0,
    prev_version INTEGER, -- Added
    merge_parent_uuid TEXT -- Added
);
CREATE INDEX IF NOT EXISTS idx_transcripts_media_id ON Transcripts(media_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_transcripts_uuid ON Transcripts(uuid);
CREATE INDEX IF NOT EXISTS idx_transcripts_last_modified ON Transcripts(last_modified);
CREATE INDEX IF NOT EXISTS idx_transcripts_deleted ON Transcripts(deleted);
CREATE INDEX IF NOT EXISTS idx_transcripts_prev_version ON Transcripts(prev_version); -- Index for new column
CREATE INDEX IF NOT EXISTS idx_transcripts_merge_parent_uuid ON Transcripts(merge_parent_uuid); -- Index for new column

-- MediaChunks Table --
CREATE TABLE IF NOT EXISTS MediaChunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT, media_id INTEGER NOT NULL, chunk_text TEXT, start_index INTEGER, end_index INTEGER,
    chunk_id TEXT UNIQUE, FOREIGN KEY (media_id) REFERENCES Media(id) ON DELETE CASCADE,
    uuid TEXT UNIQUE NOT NULL,
    last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    version INTEGER NOT NULL DEFAULT 1,
    client_id TEXT NOT NULL,
    deleted BOOLEAN NOT NULL DEFAULT 0,
    prev_version INTEGER, -- Added
    merge_parent_uuid TEXT -- Added
);
CREATE INDEX IF NOT EXISTS idx_mediachunks_media_id ON MediaChunks(media_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_mediachunks_uuid ON MediaChunks(uuid);
CREATE INDEX IF NOT EXISTS idx_mediachunks_last_modified ON MediaChunks(last_modified);
CREATE INDEX IF NOT EXISTS idx_mediachunks_deleted ON MediaChunks(deleted);
CREATE INDEX IF NOT EXISTS idx_mediachunks_prev_version ON MediaChunks(prev_version); -- Index for new column
CREATE INDEX IF NOT EXISTS idx_mediachunks_merge_parent_uuid ON MediaChunks(merge_parent_uuid); -- Index for new column

-- UnvectorizedMediaChunks Table --
CREATE TABLE IF NOT EXISTS UnvectorizedMediaChunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT, media_id INTEGER NOT NULL, chunk_text TEXT NOT NULL, chunk_index INTEGER NOT NULL,
    start_char INTEGER, end_char INTEGER, chunk_type TEXT, creation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_modified_orig TIMESTAMP DEFAULT CURRENT_TIMESTAMP, is_processed BOOLEAN DEFAULT FALSE NOT NULL, metadata TEXT,
    UNIQUE (media_id, chunk_index, chunk_type), FOREIGN KEY (media_id) REFERENCES Media(id) ON DELETE CASCADE,
    uuid TEXT UNIQUE NOT NULL,
    last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    version INTEGER NOT NULL DEFAULT 1,
    client_id TEXT NOT NULL,
    deleted BOOLEAN NOT NULL DEFAULT 0,
    prev_version INTEGER, -- Added
    merge_parent_uuid TEXT -- Added
);
CREATE INDEX IF NOT EXISTS idx_unvectorized_media_chunks_media_id ON UnvectorizedMediaChunks(media_id);
CREATE INDEX IF NOT EXISTS idx_unvectorized_media_chunks_is_processed ON UnvectorizedMediaChunks(is_processed);
CREATE INDEX IF NOT EXISTS idx_unvectorized_media_chunks_chunk_type ON UnvectorizedMediaChunks(chunk_type);
CREATE UNIQUE INDEX IF NOT EXISTS idx_unvectorizedmediachunks_uuid ON UnvectorizedMediaChunks(uuid);
CREATE INDEX IF NOT EXISTS idx_unvectorizedmediachunks_last_modified ON UnvectorizedMediaChunks(last_modified);
CREATE INDEX IF NOT EXISTS idx_unvectorizedmediachunks_deleted ON UnvectorizedMediaChunks(deleted);
CREATE INDEX IF NOT EXISTS idx_unvectorizedmediachunks_prev_version ON UnvectorizedMediaChunks(prev_version); -- Index for new column
CREATE INDEX IF NOT EXISTS idx_unvectorizedmediachunks_merge_parent_uuid ON UnvectorizedMediaChunks(merge_parent_uuid); -- Index for new column

-- DocumentVersions Table --
CREATE TABLE IF NOT EXISTS DocumentVersions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, media_id INTEGER NOT NULL, version_number INTEGER NOT NULL, prompt TEXT,
    analysis_content TEXT, content TEXT NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (media_id) REFERENCES Media(id) ON DELETE CASCADE, UNIQUE (media_id, version_number),
    uuid TEXT UNIQUE NOT NULL,
    last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    version INTEGER NOT NULL DEFAULT 1,
    client_id TEXT NOT NULL,
    deleted BOOLEAN NOT NULL DEFAULT 0,
    prev_version INTEGER, -- Added
    merge_parent_uuid TEXT -- Added
);
CREATE INDEX IF NOT EXISTS idx_document_versions_media_id ON DocumentVersions(media_id);
CREATE INDEX IF NOT EXISTS idx_document_versions_version_number ON DocumentVersions(version_number);
CREATE UNIQUE INDEX IF NOT EXISTS idx_documentversions_uuid ON DocumentVersions(uuid);
CREATE INDEX IF NOT EXISTS idx_documentversions_last_modified ON DocumentVersions(last_modified);
CREATE INDEX IF NOT EXISTS idx_documentversions_deleted ON DocumentVersions(deleted);
CREATE INDEX IF NOT EXISTS idx_documentversions_prev_version ON DocumentVersions(prev_version); -- Index for new column
CREATE INDEX IF NOT EXISTS idx_documentversions_merge_parent_uuid ON DocumentVersions(merge_parent_uuid); -- Index for new column

-- FTS Tables & Triggers --
CREATE VIRTUAL TABLE IF NOT EXISTS media_fts USING fts5(title, content, content='Media', content_rowid='id');
CREATE TRIGGER IF NOT EXISTS media_ai AFTER INSERT ON Media BEGIN INSERT INTO media_fts (rowid, title, content) VALUES (new.id, new.title, new.content); END;
CREATE TRIGGER IF NOT EXISTS media_ad AFTER DELETE ON Media BEGIN DELETE FROM media_fts WHERE rowid = old.id; END;
CREATE TRIGGER IF NOT EXISTS media_au AFTER UPDATE ON Media BEGIN UPDATE media_fts SET title = new.title, content = new.content WHERE rowid = old.id; END;
CREATE VIRTUAL TABLE IF NOT EXISTS keyword_fts USING fts5(keyword, content='Keywords', content_rowid='id');
CREATE TRIGGER IF NOT EXISTS keywords_fts_ai AFTER INSERT ON Keywords BEGIN INSERT INTO keyword_fts(rowid, keyword) VALUES (new.id, new.keyword); END;
CREATE TRIGGER IF NOT EXISTS keywords_fts_ad AFTER DELETE ON Keywords BEGIN DELETE FROM keyword_fts WHERE rowid = old.id; END;
CREATE TRIGGER IF NOT EXISTS keywords_fts_au AFTER UPDATE ON Keywords BEGIN UPDATE keyword_fts SET keyword = new.keyword WHERE rowid = old.id; END;

-- Sync Log Table & Indices --
CREATE TABLE IF NOT EXISTS sync_log (
    change_id INTEGER PRIMARY KEY AUTOINCREMENT, entity TEXT NOT NULL, entity_uuid TEXT NOT NULL,
    operation TEXT NOT NULL CHECK(operation IN ('create','update','delete', 'link', 'unlink')),
    timestamp DATETIME NOT NULL, client_id TEXT NOT NULL, version INTEGER NOT NULL, payload TEXT
);
CREATE INDEX IF NOT EXISTS idx_sync_log_ts ON sync_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_sync_log_entity_uuid ON sync_log(entity_uuid);
CREATE INDEX IF NOT EXISTS idx_sync_log_client_id ON sync_log(client_id);

-- Sync Log Triggers (Payloads updated to include prev_version, merge_parent_uuid) --

-- Media Triggers --
DROP TRIGGER IF EXISTS media_sync_create;
CREATE TRIGGER media_sync_create AFTER INSERT ON Media BEGIN
    INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
    VALUES ('Media', NEW.uuid, 'create', NEW.last_modified, NEW.client_id, NEW.version,
            json_object('uuid', NEW.uuid, 'url', NEW.url, 'title', NEW.title, 'type', NEW.type,'content', NEW.content, 'author', NEW.author, 'ingestion_date', NEW.ingestion_date,'transcription_model', NEW.transcription_model, 'is_trash', NEW.is_trash, 'trash_date', NEW.trash_date,'content_hash', NEW.content_hash, 'last_modified', NEW.last_modified,'version', NEW.version, 'client_id', NEW.client_id, 'deleted', NEW.deleted, 'prev_version', NEW.prev_version, 'merge_parent_uuid', NEW.merge_parent_uuid));
END;

DROP TRIGGER IF EXISTS media_sync_update;
CREATE TRIGGER media_sync_update AFTER UPDATE ON Media
    WHEN OLD.deleted = NEW.deleted AND (
         ifnull(OLD.url,'') != ifnull(NEW.url,'') OR ifnull(OLD.title,'') != ifnull(NEW.title,'') OR ifnull(OLD.type,'') != ifnull(NEW.type,'') OR ifnull(OLD.content,'') != ifnull(NEW.content,'') OR ifnull(OLD.author,'') != ifnull(NEW.author,'') OR ifnull(OLD.ingestion_date,'') != ifnull(NEW.ingestion_date,'') OR ifnull(OLD.transcription_model,'') != ifnull(NEW.transcription_model,'') OR ifnull(OLD.is_trash,0) != ifnull(NEW.is_trash,0) OR ifnull(OLD.trash_date,'') != ifnull(NEW.trash_date,'') OR ifnull(OLD.content_hash,'') != ifnull(NEW.content_hash,'') OR ifnull(OLD.last_modified,'') != ifnull(NEW.last_modified,'') OR ifnull(OLD.version,0) != ifnull(NEW.version,0) OR ifnull(OLD.client_id, '') != ifnull(NEW.client_id, '') OR ifnull(OLD.prev_version,'') != ifnull(NEW.prev_version,'') OR ifnull(OLD.merge_parent_uuid,'') != ifnull(NEW.merge_parent_uuid,'')
    )
BEGIN
    INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
    VALUES ('Media', NEW.uuid, 'update', NEW.last_modified, NEW.client_id, NEW.version,
            json_object('uuid', NEW.uuid, 'url', NEW.url, 'title', NEW.title, 'type', NEW.type,'content', NEW.content, 'author', NEW.author, 'ingestion_date', NEW.ingestion_date,'transcription_model', NEW.transcription_model, 'is_trash', NEW.is_trash, 'trash_date', NEW.trash_date,'content_hash', NEW.content_hash, 'last_modified', NEW.last_modified,'version', NEW.version, 'client_id', NEW.client_id, 'deleted', NEW.deleted, 'prev_version', NEW.prev_version, 'merge_parent_uuid', NEW.merge_parent_uuid));
END;

DROP TRIGGER IF EXISTS media_sync_delete;
CREATE TRIGGER media_sync_delete AFTER UPDATE ON Media WHEN OLD.deleted=0 AND NEW.deleted=1 BEGIN
    INSERT INTO sync_log(entity, entity_uuid, operation, timestamp, client_id, version, payload)
    VALUES ('Media', NEW.uuid, 'delete', NEW.last_modified, NEW.client_id, NEW.version,
            json_object('uuid',NEW.uuid,'last_modified',NEW.last_modified,'version',NEW.version,'client_id',NEW.client_id)); -- Delete payload is minimal
END;

DROP TRIGGER IF EXISTS media_sync_undelete;
CREATE TRIGGER media_sync_undelete AFTER UPDATE ON Media WHEN OLD.deleted=1 AND NEW.deleted=0 BEGIN
    INSERT INTO sync_log(entity, entity_uuid, operation, timestamp, client_id, version, payload)
    VALUES ('Media', NEW.uuid, 'update', NEW.last_modified, NEW.client_id, NEW.version,
            json_object('uuid', NEW.uuid, 'url', NEW.url, 'title', NEW.title, 'type', NEW.type,'content', NEW.content, 'author', NEW.author, 'ingestion_date', NEW.ingestion_date,'transcription_model', NEW.transcription_model, 'is_trash', NEW.is_trash, 'trash_date', NEW.trash_date,'content_hash', NEW.content_hash, 'last_modified', NEW.last_modified,'version', NEW.version, 'client_id', NEW.client_id, 'deleted', NEW.deleted, 'prev_version', NEW.prev_version, 'merge_parent_uuid', NEW.merge_parent_uuid)); -- Undelete is an 'update' with full payload
END;

-- Keywords Triggers --
DROP TRIGGER IF EXISTS keywords_sync_create;
CREATE TRIGGER keywords_sync_create AFTER INSERT ON Keywords BEGIN
    INSERT INTO sync_log(entity, entity_uuid, operation, timestamp, client_id, version, payload)
    VALUES ('Keywords', NEW.uuid, 'create', NEW.last_modified, NEW.client_id, NEW.version,
            json_object('uuid', NEW.uuid, 'keyword', NEW.keyword, 'last_modified', NEW.last_modified, 'version', NEW.version, 'client_id', NEW.client_id, 'deleted', NEW.deleted, 'prev_version', NEW.prev_version, 'merge_parent_uuid', NEW.merge_parent_uuid));
END;

DROP TRIGGER IF EXISTS keywords_sync_update;
CREATE TRIGGER keywords_sync_update AFTER UPDATE ON Keywords
    WHEN OLD.deleted = NEW.deleted AND (
        ifnull(OLD.keyword,'') != ifnull(NEW.keyword,'') OR ifnull(OLD.last_modified,'') != ifnull(NEW.last_modified,'') OR ifnull(OLD.version,0) != ifnull(NEW.version,0) OR ifnull(OLD.client_id, '') != ifnull(NEW.client_id, '') OR ifnull(OLD.prev_version,'') != ifnull(NEW.prev_version,'') OR ifnull(OLD.merge_parent_uuid,'') != ifnull(NEW.merge_parent_uuid,'')
    )
BEGIN
    INSERT INTO sync_log(entity, entity_uuid, operation, timestamp, client_id, version, payload)
    VALUES ('Keywords', NEW.uuid, 'update', NEW.last_modified, NEW.client_id, NEW.version,
            json_object('uuid', NEW.uuid, 'keyword', NEW.keyword, 'last_modified', NEW.last_modified, 'version', NEW.version, 'client_id', NEW.client_id, 'deleted', NEW.deleted, 'prev_version', NEW.prev_version, 'merge_parent_uuid', NEW.merge_parent_uuid));
END;

DROP TRIGGER IF EXISTS keywords_sync_delete;
CREATE TRIGGER keywords_sync_delete AFTER UPDATE ON Keywords WHEN OLD.deleted=0 AND NEW.deleted=1 BEGIN
    INSERT INTO sync_log(entity, entity_uuid, operation, timestamp, client_id, version, payload)
    VALUES ('Keywords', NEW.uuid, 'delete', NEW.last_modified, NEW.client_id, NEW.version,
            json_object('uuid', NEW.uuid, 'last_modified', NEW.last_modified, 'version', NEW.version, 'client_id', NEW.client_id));
END;

DROP TRIGGER IF EXISTS keywords_sync_undelete;
CREATE TRIGGER keywords_sync_undelete AFTER UPDATE ON Keywords WHEN OLD.deleted=1 AND NEW.deleted=0 BEGIN
    INSERT INTO sync_log(entity, entity_uuid, operation, timestamp, client_id, version, payload)
    VALUES ('Keywords', NEW.uuid, 'update', NEW.last_modified, NEW.client_id, NEW.version,
            json_object('uuid', NEW.uuid, 'keyword', NEW.keyword, 'last_modified', NEW.last_modified, 'version', NEW.version, 'client_id', NEW.client_id, 'deleted', NEW.deleted, 'prev_version', NEW.prev_version, 'merge_parent_uuid', NEW.merge_parent_uuid));
END;

-- MediaKeywords Triggers (Junction table - link/unlink ops, minimal payload) --
DROP TRIGGER IF EXISTS mediakeywords_sync_link;
CREATE TRIGGER mediakeywords_sync_link AFTER INSERT ON MediaKeywords BEGIN
    SELECT RAISE(ABORT, 'Cannot link keyword: Media record not found or missing UUID') WHERE NOT EXISTS (SELECT 1 FROM Media WHERE id = NEW.media_id AND uuid IS NOT NULL);
    SELECT RAISE(ABORT, 'Cannot link keyword: Keyword record not found or missing UUID') WHERE NOT EXISTS (SELECT 1 FROM Keywords WHERE id = NEW.keyword_id AND uuid IS NOT NULL);
    -- Link payload needs info about both sides. Use the *parent* record's sync meta if available?
    -- Using strftime('now') is simpler but less precise than parent last_modified.
    -- Choosing a client_id/version for the link is tricky. Using Media's seems reasonable.
    INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
    SELECT 'MediaKeywords', m.uuid || '_' || k.uuid, 'link', strftime('%Y-%m-%d %H:%M:%S.%f', 'now', 'utc'), m.client_id, m.version,
           json_object('media_uuid', m.uuid, 'keyword_uuid', k.uuid)
    FROM Media m, Keywords k WHERE m.id = NEW.media_id AND k.id = NEW.keyword_id;
END;

DROP TRIGGER IF EXISTS mediakeywords_sync_unlink;
CREATE TRIGGER mediakeywords_sync_unlink AFTER DELETE ON MediaKeywords BEGIN
    -- Unlink payload needs info about both sides. Try to get UUIDs even if parents might be deleted later.
    -- Use strftime('now') for timestamp. Client/Version are problematic here. Use 'unknown' or maybe last known Media info?
    INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
    SELECT 'MediaKeywords', ifnull(m.uuid, 'unknown_media_' || OLD.media_id) || '_' || ifnull(k.uuid, 'unknown_keyword_' || OLD.keyword_id),
           'unlink', strftime('%Y-%m-%d %H:%M:%S.%f', 'now', 'utc'), ifnull(m.client_id, 'unknown'), ifnull(m.version, 0),
           json_object('media_uuid', ifnull(m.uuid, 'unknown_media_' || OLD.media_id),
                       'keyword_uuid', ifnull(k.uuid, 'unknown_keyword_' || OLD.keyword_id))
    FROM (SELECT OLD.media_id as media_id, OLD.keyword_id as keyword_id) AS OldIds
    LEFT JOIN Media m ON m.id = OldIds.media_id
    LEFT JOIN Keywords k ON k.id = OldIds.keyword_id;
END;

-- Transcripts Triggers --
DROP TRIGGER IF EXISTS transcripts_sync_create;
CREATE TRIGGER transcripts_sync_create AFTER INSERT ON Transcripts BEGIN
    INSERT INTO sync_log(entity, entity_uuid, operation, timestamp, client_id, version, payload)
    VALUES ('Transcripts', NEW.uuid, 'create', NEW.last_modified, NEW.client_id, NEW.version,
            json_object('uuid', NEW.uuid, 'media_uuid', (SELECT uuid FROM Media WHERE id = NEW.media_id),'whisper_model', NEW.whisper_model, 'transcription', NEW.transcription, 'created_at', NEW.created_at,'last_modified', NEW.last_modified, 'version', NEW.version, 'client_id', NEW.client_id, 'deleted', NEW.deleted, 'prev_version', NEW.prev_version, 'merge_parent_uuid', NEW.merge_parent_uuid));
END;

DROP TRIGGER IF EXISTS transcripts_sync_update;
CREATE TRIGGER transcripts_sync_update AFTER UPDATE ON Transcripts
    WHEN OLD.deleted = NEW.deleted AND (
        ifnull(OLD.whisper_model,'') != ifnull(NEW.whisper_model,'') OR ifnull(OLD.transcription,'') != ifnull(NEW.transcription,'') OR ifnull(OLD.last_modified,'') != ifnull(NEW.last_modified,'') OR ifnull(OLD.version,0) != ifnull(NEW.version,0) OR ifnull(OLD.client_id, '') != ifnull(NEW.client_id, '') OR ifnull(OLD.prev_version,'') != ifnull(NEW.prev_version,'') OR ifnull(OLD.merge_parent_uuid,'') != ifnull(NEW.merge_parent_uuid,'')
    )
BEGIN
    INSERT INTO sync_log(entity, entity_uuid, operation, timestamp, client_id, version, payload)
    VALUES ('Transcripts', NEW.uuid, 'update', NEW.last_modified, NEW.client_id, NEW.version,
            json_object('uuid', NEW.uuid, 'media_uuid', (SELECT uuid FROM Media WHERE id = NEW.media_id),'whisper_model', NEW.whisper_model, 'transcription', NEW.transcription, 'created_at', NEW.created_at,'last_modified', NEW.last_modified, 'version', NEW.version, 'client_id', NEW.client_id, 'deleted', NEW.deleted, 'prev_version', NEW.prev_version, 'merge_parent_uuid', NEW.merge_parent_uuid));
END;

DROP TRIGGER IF EXISTS transcripts_sync_delete;
CREATE TRIGGER transcripts_sync_delete AFTER UPDATE ON Transcripts WHEN OLD.deleted=0 AND NEW.deleted=1 BEGIN
    INSERT INTO sync_log(entity, entity_uuid, operation, timestamp, client_id, version, payload)
    VALUES ('Transcripts', NEW.uuid, 'delete', NEW.last_modified, NEW.client_id, NEW.version,
            json_object('uuid', NEW.uuid, 'media_uuid', (SELECT uuid FROM Media WHERE id = NEW.media_id), 'last_modified', NEW.last_modified, 'version', NEW.version, 'client_id', NEW.client_id));
END;

DROP TRIGGER IF EXISTS transcripts_sync_undelete;
CREATE TRIGGER transcripts_sync_undelete AFTER UPDATE ON Transcripts WHEN OLD.deleted=1 AND NEW.deleted=0 BEGIN
    INSERT INTO sync_log(entity, entity_uuid, operation, timestamp, client_id, version, payload)
    VALUES ('Transcripts', NEW.uuid, 'update', NEW.last_modified, NEW.client_id, NEW.version,
            json_object('uuid', NEW.uuid, 'media_uuid', (SELECT uuid FROM Media WHERE id = NEW.media_id),'whisper_model', NEW.whisper_model, 'transcription', NEW.transcription, 'created_at', NEW.created_at,'last_modified', NEW.last_modified, 'version', NEW.version, 'client_id', NEW.client_id, 'deleted', NEW.deleted, 'prev_version', NEW.prev_version, 'merge_parent_uuid', NEW.merge_parent_uuid));
END;

-- MediaChunks Triggers --
DROP TRIGGER IF EXISTS mediachunks_sync_create;
CREATE TRIGGER mediachunks_sync_create AFTER INSERT ON MediaChunks BEGIN
    INSERT INTO sync_log(entity, entity_uuid, operation, timestamp, client_id, version, payload)
    VALUES ('MediaChunks', NEW.uuid, 'create', NEW.last_modified, NEW.client_id, NEW.version,
            json_object('uuid', NEW.uuid, 'media_uuid', (SELECT uuid FROM Media WHERE id = NEW.media_id),'chunk_text', NEW.chunk_text, 'start_index', NEW.start_index, 'end_index', NEW.end_index, 'chunk_id', NEW.chunk_id,'last_modified', NEW.last_modified, 'version', NEW.version, 'client_id', NEW.client_id, 'deleted', NEW.deleted, 'prev_version', NEW.prev_version, 'merge_parent_uuid', NEW.merge_parent_uuid));
END;

DROP TRIGGER IF EXISTS mediachunks_sync_update;
CREATE TRIGGER mediachunks_sync_update AFTER UPDATE ON MediaChunks
    WHEN OLD.deleted = NEW.deleted AND (
        ifnull(OLD.chunk_text,'') != ifnull(NEW.chunk_text,'') OR ifnull(OLD.start_index,0) != ifnull(NEW.start_index,0) OR ifnull(OLD.end_index,0) != ifnull(NEW.end_index,0) OR ifnull(OLD.chunk_id,'') != ifnull(NEW.chunk_id,'') OR ifnull(OLD.last_modified,'') != ifnull(NEW.last_modified,'') OR ifnull(OLD.version,0) != ifnull(NEW.version,0) OR ifnull(OLD.client_id, '') != ifnull(NEW.client_id, '') OR ifnull(OLD.prev_version,'') != ifnull(NEW.prev_version,'') OR ifnull(OLD.merge_parent_uuid,'') != ifnull(NEW.merge_parent_uuid,'')
    )
BEGIN
    INSERT INTO sync_log(entity, entity_uuid, operation, timestamp, client_id, version, payload)
    VALUES ('MediaChunks', NEW.uuid, 'update', NEW.last_modified, NEW.client_id, NEW.version,
            json_object('uuid', NEW.uuid, 'media_uuid', (SELECT uuid FROM Media WHERE id = NEW.media_id),'chunk_text', NEW.chunk_text, 'start_index', NEW.start_index, 'end_index', NEW.end_index, 'chunk_id', NEW.chunk_id,'last_modified', NEW.last_modified, 'version', NEW.version, 'client_id', NEW.client_id, 'deleted', NEW.deleted, 'prev_version', NEW.prev_version, 'merge_parent_uuid', NEW.merge_parent_uuid));
END;

DROP TRIGGER IF EXISTS mediachunks_sync_delete;
CREATE TRIGGER mediachunks_sync_delete AFTER UPDATE ON MediaChunks WHEN OLD.deleted=0 AND NEW.deleted=1 BEGIN
    INSERT INTO sync_log(entity, entity_uuid, operation, timestamp, client_id, version, payload)
    VALUES ('MediaChunks', NEW.uuid, 'delete', NEW.last_modified, NEW.client_id, NEW.version,
            json_object('uuid', NEW.uuid, 'media_uuid', (SELECT uuid FROM Media WHERE id = NEW.media_id), 'last_modified', NEW.last_modified, 'version', NEW.version, 'client_id', NEW.client_id));
END;

DROP TRIGGER IF EXISTS mediachunks_sync_undelete;
CREATE TRIGGER mediachunks_sync_undelete AFTER UPDATE ON MediaChunks WHEN OLD.deleted=1 AND NEW.deleted=0 BEGIN
    INSERT INTO sync_log(entity, entity_uuid, operation, timestamp, client_id, version, payload)
    VALUES ('MediaChunks', NEW.uuid, 'update', NEW.last_modified, NEW.client_id, NEW.version,
            json_object('uuid', NEW.uuid, 'media_uuid', (SELECT uuid FROM Media WHERE id = NEW.media_id),'chunk_text', NEW.chunk_text, 'start_index', NEW.start_index, 'end_index', NEW.end_index, 'chunk_id', NEW.chunk_id,'last_modified', NEW.last_modified, 'version', NEW.version, 'client_id', NEW.client_id, 'deleted', NEW.deleted, 'prev_version', NEW.prev_version, 'merge_parent_uuid', NEW.merge_parent_uuid));
END;

-- UnvectorizedMediaChunks Triggers --
DROP TRIGGER IF EXISTS unvectorizedmediachunks_sync_create;
CREATE TRIGGER unvectorizedmediachunks_sync_create AFTER INSERT ON UnvectorizedMediaChunks BEGIN
    INSERT INTO sync_log(entity, entity_uuid, operation, timestamp, client_id, version, payload)
    VALUES ('UnvectorizedMediaChunks', NEW.uuid, 'create', NEW.last_modified, NEW.client_id, NEW.version,
            json_object('uuid', NEW.uuid, 'media_uuid', (SELECT uuid FROM Media WHERE id = NEW.media_id),'chunk_text', NEW.chunk_text, 'chunk_index', NEW.chunk_index, 'start_char', NEW.start_char,'end_char', NEW.end_char, 'chunk_type', NEW.chunk_type, 'creation_date', NEW.creation_date,'metadata', NEW.metadata,'last_modified', NEW.last_modified, 'version', NEW.version, 'client_id', NEW.client_id, 'deleted', NEW.deleted, 'prev_version', NEW.prev_version, 'merge_parent_uuid', NEW.merge_parent_uuid));
END;

DROP TRIGGER IF EXISTS unvectorizedmediachunks_sync_update;
CREATE TRIGGER unvectorizedmediachunks_sync_update AFTER UPDATE ON UnvectorizedMediaChunks
    WHEN OLD.deleted = NEW.deleted AND (
        ifnull(OLD.chunk_text,'') != ifnull(NEW.chunk_text,'') OR ifnull(OLD.chunk_index,0) != ifnull(NEW.chunk_index,0) OR ifnull(OLD.start_char,0) != ifnull(NEW.start_char,0) OR ifnull(OLD.end_char,0) != ifnull(NEW.end_char,0) OR ifnull(OLD.chunk_type,'') != ifnull(NEW.chunk_type,'') OR ifnull(OLD.metadata,'') != ifnull(NEW.metadata,'') OR ifnull(OLD.last_modified,'') != ifnull(NEW.last_modified,'') OR ifnull(OLD.version,0) != ifnull(NEW.version,0) OR ifnull(OLD.client_id, '') != ifnull(NEW.client_id, '') OR ifnull(OLD.prev_version,'') != ifnull(NEW.prev_version,'') OR ifnull(OLD.merge_parent_uuid,'') != ifnull(NEW.merge_parent_uuid,'')
    )
BEGIN
    INSERT INTO sync_log(entity, entity_uuid, operation, timestamp, client_id, version, payload)
    VALUES ('UnvectorizedMediaChunks', NEW.uuid, 'update', NEW.last_modified, NEW.client_id, NEW.version,
            json_object('uuid', NEW.uuid, 'media_uuid', (SELECT uuid FROM Media WHERE id = NEW.media_id),'chunk_text', NEW.chunk_text, 'chunk_index', NEW.chunk_index, 'start_char', NEW.start_char,'end_char', NEW.end_char, 'chunk_type', NEW.chunk_type, 'creation_date', NEW.creation_date,'metadata', NEW.metadata,'last_modified', NEW.last_modified, 'version', NEW.version, 'client_id', NEW.client_id, 'deleted', NEW.deleted, 'prev_version', NEW.prev_version, 'merge_parent_uuid', NEW.merge_parent_uuid));
END;

DROP TRIGGER IF EXISTS unvectorizedmediachunks_sync_delete;
CREATE TRIGGER unvectorizedmediachunks_sync_delete AFTER UPDATE ON UnvectorizedMediaChunks WHEN OLD.deleted=0 AND NEW.deleted=1 BEGIN
    INSERT INTO sync_log(entity, entity_uuid, operation, timestamp, client_id, version, payload)
    VALUES ('UnvectorizedMediaChunks', NEW.uuid, 'delete', NEW.last_modified, NEW.client_id, NEW.version,
            json_object('uuid', NEW.uuid, 'media_uuid', (SELECT uuid FROM Media WHERE id = NEW.media_id), 'last_modified', NEW.last_modified, 'version', NEW.version, 'client_id', NEW.client_id));
END;

DROP TRIGGER IF EXISTS unvectorizedmediachunks_sync_undelete;
CREATE TRIGGER unvectorizedmediachunks_sync_undelete AFTER UPDATE ON UnvectorizedMediaChunks WHEN OLD.deleted=1 AND NEW.deleted=0 BEGIN
    INSERT INTO sync_log(entity, entity_uuid, operation, timestamp, client_id, version, payload)
    VALUES ('UnvectorizedMediaChunks', NEW.uuid, 'update', NEW.last_modified, NEW.client_id, NEW.version,
            json_object('uuid', NEW.uuid, 'media_uuid', (SELECT uuid FROM Media WHERE id = NEW.media_id),'chunk_text', NEW.chunk_text, 'chunk_index', NEW.chunk_index, 'start_char', NEW.start_char,'end_char', NEW.end_char, 'chunk_type', NEW.chunk_type, 'creation_date', NEW.creation_date,'metadata', NEW.metadata,'last_modified', NEW.last_modified, 'version', NEW.version, 'client_id', NEW.client_id, 'deleted', NEW.deleted, 'prev_version', NEW.prev_version, 'merge_parent_uuid', NEW.merge_parent_uuid));
END;

-- DocumentVersions Triggers --
DROP TRIGGER IF EXISTS documentversions_sync_create;
CREATE TRIGGER documentversions_sync_create AFTER INSERT ON DocumentVersions BEGIN
    INSERT INTO sync_log(entity, entity_uuid, operation, timestamp, client_id, version, payload)
    VALUES ('DocumentVersions', NEW.uuid, 'create', NEW.last_modified, NEW.client_id, NEW.version,
            json_object('uuid', NEW.uuid, 'media_uuid', (SELECT uuid FROM Media WHERE id = NEW.media_id),'version_number', NEW.version_number, 'prompt', NEW.prompt, 'analysis_content', NEW.analysis_content,'content', NEW.content, 'created_at', NEW.created_at, 'last_modified', NEW.last_modified,'version', NEW.version, 'client_id', NEW.client_id, 'deleted', NEW.deleted, 'prev_version', NEW.prev_version, 'merge_parent_uuid', NEW.merge_parent_uuid));
END;

DROP TRIGGER IF EXISTS documentversions_sync_update;
CREATE TRIGGER documentversions_sync_update AFTER UPDATE ON DocumentVersions
    WHEN OLD.deleted = NEW.deleted AND (
        ifnull(OLD.prompt,'') != ifnull(NEW.prompt,'') OR ifnull(OLD.analysis_content,'') != ifnull(NEW.analysis_content,'') OR ifnull(OLD.content,'') != ifnull(NEW.content,'') OR ifnull(OLD.version_number,0) != ifnull(NEW.version_number,0) OR ifnull(OLD.last_modified,'') != ifnull(NEW.last_modified,'') OR ifnull(OLD.version,0) != ifnull(NEW.version,0) OR ifnull(OLD.client_id, '') != ifnull(NEW.client_id, '') OR ifnull(OLD.prev_version,'') != ifnull(NEW.prev_version,'') OR ifnull(OLD.merge_parent_uuid,'') != ifnull(NEW.merge_parent_uuid,'')
    )
BEGIN
    INSERT INTO sync_log(entity, entity_uuid, operation, timestamp, client_id, version, payload)
    VALUES ('DocumentVersions', NEW.uuid, 'update', NEW.last_modified, NEW.client_id, NEW.version,
            json_object('uuid', NEW.uuid, 'media_uuid', (SELECT uuid FROM Media WHERE id = NEW.media_id),'version_number', NEW.version_number, 'prompt', NEW.prompt, 'analysis_content', NEW.analysis_content,'content', NEW.content, 'created_at', NEW.created_at, 'last_modified', NEW.last_modified,'version', NEW.version, 'client_id', NEW.client_id, 'deleted', NEW.deleted, 'prev_version', NEW.prev_version, 'merge_parent_uuid', NEW.merge_parent_uuid));
END;

DROP TRIGGER IF EXISTS documentversions_sync_delete;
CREATE TRIGGER documentversions_sync_delete AFTER UPDATE ON DocumentVersions WHEN OLD.deleted=0 AND NEW.deleted=1 BEGIN
    INSERT INTO sync_log(entity, entity_uuid, operation, timestamp, client_id, version, payload)
    VALUES ('DocumentVersions', NEW.uuid, 'delete', NEW.last_modified, NEW.client_id, NEW.version,
            json_object('uuid', NEW.uuid, 'media_uuid', (SELECT uuid FROM Media WHERE id = NEW.media_id), 'last_modified', NEW.last_modified, 'version', NEW.version, 'client_id', NEW.client_id));
END;

DROP TRIGGER IF EXISTS documentversions_sync_undelete;
CREATE TRIGGER documentversions_sync_undelete AFTER UPDATE ON DocumentVersions WHEN OLD.deleted=1 AND NEW.deleted=0 BEGIN
    INSERT INTO sync_log(entity, entity_uuid, operation, timestamp, client_id, version, payload)
    VALUES ('DocumentVersions', NEW.uuid, 'update', NEW.last_modified, NEW.client_id, NEW.version,
            json_object('uuid', NEW.uuid, 'media_uuid', (SELECT uuid FROM Media WHERE id = NEW.media_id),'version_number', NEW.version_number, 'prompt', NEW.prompt, 'analysis_content', NEW.analysis_content,'content', NEW.content, 'created_at', NEW.created_at, 'last_modified', NEW.last_modified,'version', NEW.version, 'client_id', NEW.client_id, 'deleted', NEW.deleted, 'prev_version', NEW.prev_version, 'merge_parent_uuid', NEW.merge_parent_uuid));
END;

-- Validation Triggers (Do not validate prev_version or merge_parent_uuid yet) --
DROP TRIGGER IF EXISTS media_validate_sync_update; CREATE TRIGGER media_validate_sync_update BEFORE UPDATE ON Media BEGIN SELECT RAISE(ABORT, 'Sync Error (Media): Version must increment by exactly 1.') WHERE NEW.version IS NOT OLD.version + 1; SELECT RAISE(ABORT, 'Sync Error (Media): Client ID cannot be NULL or empty.') WHERE NEW.client_id IS NULL OR NEW.client_id = ''; END;
DROP TRIGGER IF EXISTS keywords_validate_sync_update; CREATE TRIGGER keywords_validate_sync_update BEFORE UPDATE ON Keywords BEGIN SELECT RAISE(ABORT, 'Sync Error (Keywords): Version must increment by exactly 1.') WHERE NEW.version IS NOT OLD.version + 1; SELECT RAISE(ABORT, 'Sync Error (Keywords): Client ID cannot be NULL or empty.') WHERE NEW.client_id IS NULL OR NEW.client_id = ''; END;
DROP TRIGGER IF EXISTS transcripts_validate_sync_update; CREATE TRIGGER transcripts_validate_sync_update BEFORE UPDATE ON Transcripts BEGIN SELECT RAISE(ABORT, 'Sync Error (Transcripts): Version must increment by exactly 1.') WHERE NEW.version IS NOT OLD.version + 1; SELECT RAISE(ABORT, 'Sync Error (Transcripts): Client ID cannot be NULL or empty.') WHERE NEW.client_id IS NULL OR NEW.client_id = ''; END;
DROP TRIGGER IF EXISTS mediachunks_validate_sync_update; CREATE TRIGGER mediachunks_validate_sync_update BEFORE UPDATE ON MediaChunks BEGIN SELECT RAISE(ABORT, 'Sync Error (MediaChunks): Version must increment by exactly 1.') WHERE NEW.version IS NOT OLD.version + 1; SELECT RAISE(ABORT, 'Sync Error (MediaChunks): Client ID cannot be NULL or empty.') WHERE NEW.client_id IS NULL OR NEW.client_id = ''; END;
DROP TRIGGER IF EXISTS unvectorizedmediachunks_validate_sync_update; CREATE TRIGGER unvectorizedmediachunks_validate_sync_update BEFORE UPDATE ON UnvectorizedMediaChunks BEGIN SELECT RAISE(ABORT, 'Sync Error (UnvectorizedMediaChunks): Version must increment by exactly 1.') WHERE NEW.version IS NOT OLD.version + 1; SELECT RAISE(ABORT, 'Sync Error (UnvectorizedMediaChunks): Client ID cannot be NULL or empty.') WHERE NEW.client_id IS NULL OR NEW.client_id = ''; END;
DROP TRIGGER IF EXISTS documentversions_validate_sync_update; CREATE TRIGGER documentversions_validate_sync_update BEFORE UPDATE ON DocumentVersions BEGIN SELECT RAISE(ABORT, 'Sync Error (DocumentVersions): Version must increment by exactly 1.') WHERE NEW.version IS NOT OLD.version + 1; SELECT RAISE(ABORT, 'Sync Error (DocumentVersions): Client ID cannot be NULL or empty.') WHERE NEW.client_id IS NULL OR NEW.client_id = ''; END;
"""

    def __init__(self, db_path: str, client_id: str):
        self.is_memory_db = (db_path == ':memory:')
        if not client_id:
            raise ValueError("Client ID cannot be empty or None.")
        self.client_id = client_id

        if self.is_memory_db:
            self.db_path_str = ':memory:'
            logging.info(f"Initializing Database object for :memory: [Client ID: {self.client_id}]")
        else:
            from pathlib import Path
            self.db_path = Path(db_path).resolve()
            self.db_path_str = str(self.db_path)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            logging.info(f"Initializing Database object for path: {self.db_path_str} [Client ID: {self.client_id}]")

        self._local = threading.local()
        try:
            self._ensure_schema()
        except DatabaseError as e:
            logging.critical(f"FATAL: Initial DB schema setup failed for {self.db_path_str}: {e}", exc_info=True)
            raise

    # --- Connection Management ---
    def _get_thread_connection(self) -> sqlite3.Connection:
        conn = getattr(self._local, 'conn', None)
        is_closed = True
        if conn:
            try:
                conn.execute("SELECT 1") # Simple check
                is_closed = False
            except (sqlite3.ProgrammingError, sqlite3.OperationalError):
                 logging.warning(f"Thread-local connection to {self.db_path_str} was closed. Reopening.")
                 is_closed = True
                 try: conn.close()
                 except Exception: pass
                 self._local.conn = None

        if is_closed:
            try:
                conn = sqlite3.connect(
                    self.db_path_str,
                    detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
                    check_same_thread=False,
                    timeout=10
                )
                conn.row_factory = sqlite3.Row
                if not self.is_memory_db:
                    conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA foreign_keys = ON;")
                self._local.conn = conn
                logging.debug(f"Opened/Reopened SQLite connection to {self.db_path_str} [Client: {self.client_id}, Thread: {threading.current_thread().name}]")
            except sqlite3.Error as e:
                logging.error(f"Failed to connect to database at {self.db_path_str}: {e}", exc_info=True)
                self._local.conn = None
                raise DatabaseError(f"Failed to connect to database '{self.db_path_str}': {e}") from e
        return self._local.conn

    def get_connection(self) -> sqlite3.Connection:
        return self._get_thread_connection()

    def close_connection(self):
        if hasattr(self._local, 'conn') and self._local.conn is not None:
            try: self._local.conn.close()
            except sqlite3.Error as e: logging.warning(f"Error closing connection: {e}")
            finally: self._local.conn = None; logging.debug(f"Closed connection.")

    # --- Query Execution ---
    def execute_query(self, query: str, params: tuple = None, *, commit: bool = False) -> sqlite3.Cursor:
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            logging.debug(f"Executing Query: {query[:200]}... Params: {str(params)[:100]}...")
            cursor.execute(query, params or ())
            if commit: conn.commit(); logging.debug("Committed.")
            return cursor
        except sqlite3.Error as e:
            msg = str(e).lower()
            if "sync error" in msg and ("version must increment" in msg or "client id cannot be null" in msg):
                 logging.error(f"Sync Validation Failed: {e}")
                 raise e # Let validation errors propagate clearly
            else:
                 logging.error(f"Query failed: {query[:200]}... Error: {e}", exc_info=True)
                 raise DatabaseError(f"Query execution failed: {e}") from e

    def execute_many(self, query: str, params_list: List[tuple], *, commit: bool = False) -> Optional[sqlite3.Cursor]:
        conn = self.get_connection()
        if not isinstance(params_list, list): raise TypeError("params_list must be a list.")
        if not params_list: return None
        try:
            cursor = conn.cursor()
            logging.debug(f"Executing Many: {query[:150]}... with {len(params_list)} sets.")
            cursor.executemany(query, params_list)
            if commit: conn.commit(); logging.debug("Committed Many.")
            return cursor
        except sqlite3.Error as e:
            logging.error(f"Execute Many failed: {query[:150]}... Error: {e}", exc_info=True)
            raise DatabaseError(f"Execute Many failed: {e}") from e
        except TypeError as te:
            logging.error(f"TypeError during Execute Many: {te}. Check params_list format.", exc_info=True)
            raise TypeError(f"Parameter list format error: {te}") from te

    # --- Transaction Context ---
    @contextmanager
    def transaction(self):
        conn = self.get_connection()
        in_outer = conn.in_transaction
        try:
            if not in_outer: conn.execute("BEGIN")
            yield conn
            if not in_outer: conn.commit()
        except Exception as e:
            if not in_outer:
                logging.error(f"Transaction failed, rolling back: {type(e).__name__} - {e}", exc_info=False)
                try: conn.rollback(); logging.debug("Rollback successful.")
                except sqlite3.Error as rb_err: logging.error(f"Rollback FAILED: {rb_err}", exc_info=True)
            if isinstance(e, (ConflictError, InputError, DatabaseError, sqlite3.Error)): raise e
            else: raise DatabaseError(f"Transaction failed unexpectedly: {e}") from e

    # --- Schema Setup ---
    def _ensure_schema(self):
        conn = self.get_connection()
        try:
            logging.info(f"Ensuring schema exists for: {self.db_path_str}")
            conn.executescript(self._SCHEMA_SQL)
            conn.commit()
            logging.info(f"Schema setup complete for: {self.db_path_str}")
        except sqlite3.Error as e:
            logging.error(f"Schema setup failed: {e}", exc_info=True)
            try: conn.rollback()
            except sqlite3.Error: pass
            raise DatabaseError(f"DB schema setup failed: {e}") from e

    # --- Internal Helpers ---
    def _get_current_utc_timestamp_str(self) -> str:
        return datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

    def _generate_uuid(self) -> str:
        return str(uuid.uuid4())

    def _get_next_version(self, conn: sqlite3.Connection, table: str, id_col: str, id_val: Any) -> Optional[Tuple[int, int]]:
        # Added deleted=0 check
        cursor = conn.execute(f"SELECT version FROM {table} WHERE {id_col} = ? AND deleted = 0", (id_val,))
        result = cursor.fetchone()
        if result:
            current_version = result['version']
            return current_version, current_version + 1
        return None

    # --- Public Mutating Methods (Internalizing Sync Meta) ---
    # (Includes add_keyword, add_media_with_keywords, create_document_version,
    #  update_keywords_for_media, soft_delete_media from previous responses)
    def add_keyword(self, keyword: str) -> Tuple[Optional[int], Optional[str]]:
        if not keyword or not keyword.strip(): raise InputError("Keyword cannot be empty.")
        keyword = keyword.strip().lower(); current_time = self._get_current_utc_timestamp_str(); client_id = self.client_id
        try:
            with self.transaction() as conn:
                cursor = conn.cursor(); cursor.execute('SELECT id, uuid, deleted, version FROM Keywords WHERE keyword = ?', (keyword,)); existing = cursor.fetchone()
                if existing:
                    kw_id, kw_uuid, is_deleted, current_version = existing['id'], existing['uuid'], existing['deleted'], existing['version']
                    if is_deleted:
                        new_version = current_version + 1; logger.info(f"Undeleting keyword '{keyword}' (ID: {kw_id}). New ver: {new_version}")
                        cursor.execute("UPDATE Keywords SET deleted=0, last_modified=?, version=?, client_id=? WHERE id=? AND version=?", (current_time, new_version, client_id, kw_id, current_version))
                        if cursor.rowcount == 0: raise ConflictError("Keywords", kw_id)
                        return kw_id, kw_uuid
                    else: logger.debug(f"Keyword '{keyword}' active."); return kw_id, kw_uuid
                else:
                    new_uuid = self._generate_uuid(); logger.info(f"Adding new keyword '{keyword}' UUID {new_uuid}.")
                    cursor.execute("INSERT INTO Keywords (keyword, uuid, last_modified, version, client_id, deleted) VALUES (?, ?, ?, 1, ?, 0)", (keyword, new_uuid, current_time, client_id)); kw_id = cursor.lastrowid
                    return kw_id, new_uuid
        except (InputError, ConflictError, sqlite3.Error) as e:
             logger.error(f"Error add/undelete keyword '{keyword}': {e}"); raise e if isinstance(e, (InputError, ConflictError, DatabaseError)) else DatabaseError(f"Failed add keyword: {e}") from e
        except Exception as e: logger.error(f"Unexpected keyword error '{keyword}': {e}"); raise DatabaseError(f"Unexpected add keyword error: {e}") from e

    def get_sync_log_entries(self, since_change_id: int = 0, limit: Optional[int] = None) -> List[Dict]:
        """
        Retrieves sync log entries newer than a given change_id.

        Args:
            since_change_id: The minimum change_id (exclusive) to fetch. Defaults to 0 (fetch all).
            limit: Maximum number of entries to return.

        Returns:
            A list of dictionaries representing sync log entries.
        """
        query = "SELECT change_id, entity, entity_uuid, operation, timestamp, client_id, version, payload FROM sync_log WHERE change_id > ? ORDER BY change_id ASC"
        params = [since_change_id]
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        try:
            cursor = self.execute_query(query, tuple(params))
            return [dict(row) for row in cursor.fetchall()]
        except (DatabaseError, sqlite3.Error) as e:
            logger.error(f"Error fetching sync log entries from DB '{self.db_path_str}': {e}")
            raise DatabaseError("Failed to fetch sync log entries") from e

    def delete_sync_log_entries(self, change_ids: List[int]) -> int:
        """
        Deletes specific sync log entries by their change_id.
        Intended for use by an external sync log vacuum process.

        Args:
            change_ids: A list of primary key change_id values to delete.

        Returns:
            The number of rows deleted.

        Raises:
            DatabaseError: If the deletion fails.
        """
        if not change_ids:
            return 0
        if not all(isinstance(cid, int) for cid in change_ids):
            raise ValueError("change_ids must be a list of integers.")

        placeholders = ','.join('?' * len(change_ids))
        query = f"DELETE FROM sync_log WHERE change_id IN ({placeholders})"
        try:
            # Use a transaction for potentially large deletes
            with self.transaction():
                cursor = self.execute_query(query, tuple(change_ids), commit=False) # Commit handled by transaction
                deleted_count = cursor.rowcount
                logger.info(f"Deleted {deleted_count} sync log entries from DB '{self.db_path_str}'.")
                return deleted_count
        except (DatabaseError, sqlite3.Error) as e:
            logger.error(f"Error deleting sync log entries from DB '{self.db_path_str}': {e}")
            raise DatabaseError("Failed to delete sync log entries") from e
        except Exception as e:
            logger.error(f"Unexpected error deleting sync log entries from DB '{self.db_path_str}': {e}")
            raise DatabaseError(f"Unexpected error deleting sync log entries: {e}") from e

    def delete_sync_log_entries_before(self, change_id_threshold: int) -> int:
        """
        Deletes sync log entries with change_id less than or equal to a threshold.
        Intended for use by an external sync log vacuum process.

        Args:
            change_id_threshold: Delete all entries with change_id <= this value.

        Returns:
            The number of rows deleted.

        Raises:
            DatabaseError: If the deletion fails.
        """
        if not isinstance(change_id_threshold, int) or change_id_threshold < 0:
             raise ValueError("change_id_threshold must be a non-negative integer.")

        query = "DELETE FROM sync_log WHERE change_id <= ?"
        try:
             with self.transaction():
                 cursor = self.execute_query(query, (change_id_threshold,), commit=False)
                 deleted_count = cursor.rowcount
                 logger.info(f"Deleted {deleted_count} sync log entries before or at ID {change_id_threshold} from DB '{self.db_path_str}'.")
                 return deleted_count
        except (DatabaseError, sqlite3.Error) as e:
            logger.error(f"Error deleting sync log entries before {change_id_threshold} from DB '{self.db_path_str}': {e}")
            raise DatabaseError("Failed to delete sync log entries before threshold") from e
        except Exception as e:
            logger.error(f"Unexpected error deleting sync log entries before {change_id_threshold} from DB '{self.db_path_str}': {e}")
            raise DatabaseError(f"Unexpected error deleting sync log entries before threshold: {e}") from e

    def soft_delete_media(self, media_id: int, cascade: bool = True) -> bool:
        """
        Soft deletes Media item and optionally cascades to children.
        Uses instance client_id and handles sync metadata internally.

        Args:
            media_id: The ID of the media item to delete.
            cascade (bool): If True, explicitly soft-delete children and unlink keywords.

        Returns:
            bool: True if successful, False if not found/already deleted/conflict.

        Raises: DatabaseError, ConflictError.
        """
        # Now uses self.client_id and self.transaction() etc.
        current_time = self._get_current_utc_timestamp_str()
        client_id = self.client_id
        logger.info(f"Attempting soft delete for Media ID: {media_id} [Client: {client_id}, Cascade: {cascade}]")

        try:
            # Use the instance's transaction context manager
            with self.transaction() as conn:
                cursor = conn.cursor()

                # 1. Get current version & Check if active using internal helper
                version_info = self._get_next_version(conn, "Media", "id", media_id)
                if version_info is None:
                    logger.warning(f"Cannot soft delete: Media ID {media_id} not found or already deleted.")
                    return False
                current_media_version, new_media_version = version_info

                # 2. Soft delete the main Media item with optimistic locking
                logger.debug(f"Soft deleting main Media record ID: {media_id} (Setting version {new_media_version})")
                cursor.execute("""
                    UPDATE Media SET deleted = 1, last_modified = ?, version = ?, client_id = ?
                    WHERE id = ? AND version = ? -- Optimistic lock
                """, (current_time, new_media_version, client_id, media_id, current_media_version))
                if cursor.rowcount == 0:
                    # This indicates a conflict (version changed) or the record was deleted concurrently
                    raise ConflictError(entity="Media", identifier=media_id)
                # Validation & Sync log triggers handle checks and logging

                # 3. Explicit Cascade (if enabled)
                if cascade:
                    logger.info(f"Performing explicit cascade delete for Media ID: {media_id}")
                    # --- 3a. Unlink Keywords ---
                    cursor.execute("DELETE FROM MediaKeywords WHERE media_id = ?", (media_id,))
                    logger.debug(f"Unlinked {cursor.rowcount} keywords.")

                    # --- 3b. Soft Delete Children (using robust versioning) ---
                    child_tables_to_cascade = [
                        ("Transcripts", "media_id", "uuid"),
                        ("MediaChunks", "media_id", "uuid"),
                        ("UnvectorizedMediaChunks", "media_id", "uuid"),
                        ("DocumentVersions", "media_id", "uuid")
                    ]
                    for table, fk_col, uuid_col in child_tables_to_cascade:
                        logger.debug(f"Cascading soft delete to {table} for Media ID: {media_id}")
                        cursor.execute(f"SELECT id, version FROM {table} WHERE {fk_col} = ? AND deleted = 0", (media_id,))
                        children = cursor.fetchall()
                        if not children: continue

                        update_params = []
                        for child in children:
                             child_id = child['id']
                             child_current_version = child['version']
                             child_new_version = child_current_version + 1
                             update_params.append((current_time, child_new_version, client_id, child_id, child_current_version))

                        update_sql = f"""
                            UPDATE {table} SET deleted = 1, last_modified = ?, version = ?, client_id = ?
                            WHERE id = ? AND version = ? AND deleted = 0
                        """
                        # Note: executemany with UPDATE WHERE might not work as expected for optimistic locking verification easily.
                        # Looping might be necessary if strict rowcount checking per child is needed.
                        # For now, attempt bulk update; validation triggers are the primary guard.
                        cursor.executemany(update_sql, update_params)
                        logger.debug(f"Attempted cascade soft delete for {len(children)} records in {table}.")
                else:
                    logger.info(f"Implicit cascade delete for Media ID: {media_id}.")

            logger.info(f"Soft delete successful for Media ID: {media_id}.")
            return True

        except (ConflictError, DatabaseError, sqlite3.Error) as e:
             logger.error(f"Error soft deleting media ID {media_id}: {e}", exc_info=True)
             # Re-raise specific errors or DatabaseError
             if isinstance(e, (ConflictError, DatabaseError)): raise e
             else: raise DatabaseError(f"Failed to soft delete media: {e}") from e
        except Exception as e:
             logger.error(f"Unexpected error soft deleting media ID {media_id}: {e}", exc_info=True)
             raise DatabaseError(f"Unexpected error during soft delete: {e}") from e

    def add_media_with_keywords(self, *, url: Optional[str] = None, title: Optional[str], media_type: Optional[str], content: Optional[str], keywords: Optional[List[str]] = None, prompt: Optional[str] = None, analysis_content: Optional[str] = None, transcription_model: Optional[str] = None, author: Optional[str] = None, ingestion_date: Optional[str] = None, overwrite: bool = False, chunk_options: Optional[Dict] = None, segments: Optional[Any] = None) -> Tuple[Optional[int], Optional[str], str]:
        if content is None: raise InputError("Content cannot be None.")
        title = title or 'Untitled'; media_type = media_type or 'unknown'; keywords_list = [k.strip().lower() for k in keywords if k and k.strip()] if keywords else []
        ingestion_date_str = ingestion_date or self._get_current_utc_timestamp_str().split(" ")[0] # Default today
        content_hash = hashlib.sha256(content.encode()).hexdigest(); current_time = self._get_current_utc_timestamp_str(); client_id = self.client_id
        if not url: url = f"local://{media_type}/{content_hash}"
        logging.info(f"Processing add/update for: URL='{url}', Title='{title}', Client='{client_id}'")
        try:
            with self.transaction() as conn:
                cursor = conn.cursor(); cursor.execute('SELECT id, uuid, version FROM Media WHERE (url = ? OR content_hash = ?) AND deleted = 0 LIMIT 1', (url, content_hash)); existing_media = cursor.fetchone()
                media_id, media_uuid, action = None, None, "skipped"
                if existing_media:
                    media_id, media_uuid, current_version = existing_media['id'], existing_media['uuid'], existing_media['version']
                    if overwrite:
                        action = "updated"; new_version = current_version + 1; logger.info(f"Updating media ID {media_id} to version {new_version}.")
                        cursor.execute('UPDATE Media SET url=?, title=?, type=?, content=?, author=?, ingestion_date=?, transcription_model=?, content_hash=?, is_trash=0, trash_date=NULL, chunking_status="pending", vector_processing=0, last_modified=?, version=?, client_id=?, deleted=0 WHERE id=? AND version=?', (url, title, media_type, content, author, ingestion_date_str, transcription_model, content_hash, current_time, new_version, client_id, media_id, current_version))
                        if cursor.rowcount == 0: raise ConflictError("Media", media_id)
                        self.update_keywords_for_media(media_id, keywords_list); self.create_document_version(media_id=media_id, content=content, prompt=prompt, analysis_content=analysis_content)
                    else: action = "already_exists_skipped"
                else:
                    action = "added"; media_uuid = self._generate_uuid(); logger.info(f"Inserting new media UUID {media_uuid}.")
                    cursor.execute('INSERT INTO Media (url, title, type, content, author, ingestion_date, transcription_model, content_hash, is_trash, chunking_status, vector_processing, uuid, last_modified, version, client_id, deleted) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, "pending", 0, ?, ?, 1, ?, 0)', (url, title, media_type, content, author, ingestion_date_str, transcription_model, content_hash, media_uuid, current_time, client_id)); media_id = cursor.lastrowid
                    self.update_keywords_for_media(media_id, keywords_list); self.create_document_version(media_id=media_id, content=content, prompt=prompt, analysis_content=analysis_content)
            if action in ["added", "updated"] and chunk_options: logger.info(f"Scheduling chunking for media {media_id}") # Placeholder
            if action == "updated": message = f"Media '{title}' updated."
            elif action == "added": message = f"Media '{title}' added."
            else: message = f"Media '{title}' exists, not overwritten."
            return media_id, media_uuid, message
        except (InputError, ConflictError, sqlite3.Error) as e: logger.error(f"Error add/update media {url}: {e}"); raise e if isinstance(e, (InputError, ConflictError, DatabaseError)) else DatabaseError(f"Failed add/update media: {e}") from e
        except Exception as e: logger.error(f"Unexpected error add/update media {url}: {e}"); raise DatabaseError(f"Unexpected error add/update media: {e}") from e

    def create_document_version(self, media_id: int, content: str, prompt: Optional[str] = None, analysis_content: Optional[str] = None) -> Dict[str, Any]:
        if content is None: raise InputError("Content required for doc version.")
        current_time = self._get_current_utc_timestamp_str(); client_id = self.client_id; new_uuid = self._generate_uuid()
        conn = self.get_connection() # Assumes called within existing transaction or manages its own
        try:
             cursor = conn.cursor(); cursor.execute('SELECT COALESCE(MAX(version_number), 0) + 1 FROM DocumentVersions WHERE media_id = ?', (media_id,)); version_number = cursor.fetchone()[0]
             logger.debug(f"Creating doc version {version_number} for media {media_id}.")
             cursor.execute('INSERT INTO DocumentVersions (media_id, version_number, content, prompt, analysis_content, created_at, uuid, last_modified, version, client_id, deleted) VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?, ?, 1, ?, 0)', (media_id, version_number, content, prompt, analysis_content, new_uuid, current_time, client_id))
             return {'media_id': media_id, 'version_number': version_number, 'uuid': new_uuid}
        except sqlite3.Error as e:
            if "foreign key constraint failed" in str(e).lower(): logger.error(f"FK fail: Media ID {media_id} not found."); raise DatabaseError(f"Media ID {media_id} not found.") from e
            logger.error(f"DB error creating doc version for media {media_id}: {e}"); raise DatabaseError(f"Failed create doc version: {e}") from e

    def update_keywords_for_media(self, media_id: int, keywords: List[str]):
        valid_keywords = sorted(list(set([k.strip().lower() for k in keywords if k and k.strip()])))
        current_time = self._get_current_utc_timestamp_str(); client_id = self.client_id
        try:
            with self.transaction() as conn:
                cursor = conn.cursor()
                version_info = self._get_next_version(conn, "Media", "id", media_id)
                if version_info is None: raise DatabaseError(f"Cannot update keywords: Media {media_id} not found/deleted.")
                current_media_version, new_media_version = version_info
                cursor.execute('SELECT k.id FROM Keywords k JOIN MediaKeywords mk ON k.id = mk.keyword_id WHERE mk.media_id = ?', (media_id,)); current_keyword_ids = {row['id'] for row in cursor.fetchall()}
                target_keyword_ids = set()
                if valid_keywords:
                     for kw in valid_keywords:
                         kw_id, _ = self.add_keyword(kw) # Handles internal meta
                         if kw_id: target_keyword_ids.add(kw_id)
                         else: raise DatabaseError(f"Failed get/add keyword '{kw}'")
                ids_to_add = target_keyword_ids - current_keyword_ids; ids_to_remove = current_keyword_ids - target_keyword_ids
                if ids_to_remove: cursor.execute(f"DELETE FROM MediaKeywords WHERE media_id = ? AND keyword_id IN ({','.join('?'*len(ids_to_remove))})", (media_id, *list(ids_to_remove)))
                if ids_to_add: cursor.executemany("INSERT OR IGNORE INTO MediaKeywords (media_id, keyword_id) VALUES (?, ?)", [(media_id, kid) for kid in ids_to_add])
                if ids_to_add or ids_to_remove:
                     logger.debug(f"Updating parent Media {media_id} version for keyword changes.")
                     cursor.execute("UPDATE Media SET last_modified = ?, version = ?, client_id = ? WHERE id = ? AND version = ?", (current_time, new_media_version, client_id, media_id, current_media_version))
                     if cursor.rowcount == 0: raise ConflictError("Media", media_id)
                else: logger.debug(f"No keyword changes for media {media_id}.")
            return True
        except (ConflictError, sqlite3.Error) as e: logger.error(f"Error updating keywords media {media_id}: {e}"); raise e if isinstance(e, (ConflictError, DatabaseError)) else DatabaseError(f"Keyword update failed: {e}") from e
        except Exception as e: logger.error(f"Unexpected keywords error media {media_id}: {e}"); raise DatabaseError(f"Unexpected keyword update error: {e}") from e

    def soft_delete_keyword(self, keyword: str) -> bool:
        if not keyword or not keyword.strip(): raise InputError("Keyword cannot be empty.")
        keyword = keyword.strip().lower(); current_time = self._get_current_utc_timestamp_str(); client_id = self.client_id
        try:
            with self.transaction() as conn:
                cursor = conn.cursor(); cursor.execute('SELECT id, version FROM Keywords WHERE keyword = ? AND deleted = 0', (keyword,)); keyword_info = cursor.fetchone()
                if not keyword_info: logger.warning(f"Keyword '{keyword}' not found/deleted."); return False
                keyword_id, current_version = keyword_info['id'], keyword_info['version']; new_version = current_version + 1
                logger.info(f"Soft deleting keyword '{keyword}' (ID: {keyword_id}). New ver: {new_version}")
                cursor.execute("UPDATE Keywords SET deleted=1, last_modified=?, version=?, client_id=? WHERE id=? AND version=?", (current_time, new_version, client_id, keyword_id, current_version))
                if cursor.rowcount == 0: raise ConflictError("Keywords", keyword_id)
                logger.debug(f"Unlinking keyword ID {keyword_id} from MediaKeywords."); cursor.execute("DELETE FROM MediaKeywords WHERE keyword_id = ?", (keyword_id,)); logger.info(f"Unlinked keyword '{keyword}' from {cursor.rowcount} items.")
            return True
        except (InputError, ConflictError, sqlite3.Error) as e: logger.error(f"Error soft delete keyword '{keyword}': {e}"); raise e if isinstance(e, (InputError, ConflictError, DatabaseError)) else DatabaseError(f"Failed soft delete keyword: {e}") from e
        except Exception as e: logger.error(f"Unexpected soft delete keyword error '{keyword}': {e}"); raise DatabaseError(f"Unexpected soft delete keyword error: {e}") from e

    def soft_delete_document_version(self, version_uuid: str) -> bool:
        if not version_uuid: raise InputError("Version UUID required.")
        current_time = self._get_current_utc_timestamp_str(); client_id = self.client_id
        logger.debug(f"Attempting soft delete DocVersion UUID: {version_uuid}")
        try:
            with self.transaction() as conn:
                cursor = conn.cursor(); cursor.execute("SELECT id, media_id, version FROM DocumentVersions WHERE uuid = ? AND deleted = 0", (version_uuid,)); version_info = cursor.fetchone()
                if not version_info: logger.warning(f"DocVersion UUID {version_uuid} not found/deleted."); return False
                version_id, media_id, current_sync_version = version_info['id'], version_info['media_id'], version_info['version']; new_sync_version = current_sync_version + 1
                cursor.execute("SELECT COUNT(*) FROM DocumentVersions WHERE media_id = ? AND deleted = 0", (media_id,)); active_count = cursor.fetchone()[0]
                if active_count <= 1: logger.warning(f"Cannot delete DocVersion UUID {version_uuid} - last active."); return False
                cursor.execute("UPDATE DocumentVersions SET deleted=1, last_modified=?, version=?, client_id=? WHERE id=? AND version=?", (current_time, new_sync_version, client_id, version_id, current_sync_version))
                if cursor.rowcount == 0: raise ConflictError("DocumentVersions", version_id)
                logger.info(f"Soft deleted DocVersion UUID {version_uuid}. New ver: {new_sync_version}")
                return True
        except (InputError, ConflictError, DatabaseError, sqlite3.Error) as e: logger.error(f"Error soft delete DocVersion UUID {version_uuid}: {e}"); raise e if isinstance(e, (InputError, ConflictError, DatabaseError)) else DatabaseError(f"Failed soft delete doc version: {e}") from e
        except Exception as e: logger.error(f"Unexpected soft delete DocVersion error UUID {version_uuid}: {e}"); raise DatabaseError(f"Unexpected version soft delete error: {e}") from e

    def mark_as_trash(self, media_id: int) -> bool:
        current_time = self._get_current_utc_timestamp_str(); client_id = self.client_id; logger.debug(f"Marking media {media_id} as trash.")
        try:
            with self.transaction() as conn:
                version_info = self._get_next_version(conn, "Media", "id", media_id)
                if version_info is None: logger.warning(f"Cannot trash: Media {media_id} not found/deleted."); return False # Already deleted
                cursor = conn.execute("SELECT is_trash FROM Media WHERE id = ?", (media_id,)); trash_info = cursor.fetchone()
                if trash_info and trash_info['is_trash']: logger.warning(f"Media {media_id} already in trash."); return False # Already trashed
                current_version, new_version = version_info
                cursor = conn.cursor(); cursor.execute("UPDATE Media SET is_trash=1, trash_date=?, last_modified=?, version=?, client_id=? WHERE id=? AND version=?", (current_time, current_time, new_version, client_id, media_id, current_version))
                if cursor.rowcount == 0: raise ConflictError("Media", media_id)
                logger.info(f"Media {media_id} marked as trash. New ver: {new_version}")
                return True
        except (ConflictError, DatabaseError, sqlite3.Error) as e: logger.error(f"Error marking media {media_id} as trash: {e}"); raise e if isinstance(e, (ConflictError, DatabaseError)) else DatabaseError(f"Failed mark as trash: {e}") from e
        except Exception as e: logger.error(f"Unexpected error marking media {media_id} trash: {e}"); raise DatabaseError(f"Unexpected mark trash error: {e}") from e

    def restore_from_trash(self, media_id: int) -> bool:
        current_time = self._get_current_utc_timestamp_str(); client_id = self.client_id; logger.debug(f"Restoring media {media_id} from trash.")
        try:
            with self.transaction() as conn:
                cursor = conn.execute("SELECT version, is_trash FROM Media WHERE id = ? AND deleted = 0", (media_id,)) # Check deleted=0 here
                media_info = cursor.fetchone()
                if not media_info: logger.warning(f"Cannot restore: Media {media_id} not found/deleted."); return False
                if not media_info['is_trash']: logger.warning(f"Cannot restore: Media {media_id} not in trash."); return False
                current_version = media_info['version']; new_version = current_version + 1
                cursor = conn.cursor(); cursor.execute("UPDATE Media SET is_trash=0, trash_date=NULL, last_modified=?, version=?, client_id=? WHERE id=? AND version=?", (current_time, new_version, client_id, media_id, current_version))
                if cursor.rowcount == 0: raise ConflictError("Media", media_id)
                logger.info(f"Media {media_id} restored from trash. New ver: {new_version}")
                return True
        except (ConflictError, DatabaseError, sqlite3.Error) as e: logger.error(f"Error restoring media {media_id} trash: {e}"); raise e if isinstance(e, (ConflictError, DatabaseError)) else DatabaseError(f"Failed restore trash: {e}") from e
        except Exception as e: logger.error(f"Unexpected error restoring media {media_id} trash: {e}"); raise DatabaseError(f"Unexpected restore trash error: {e}") from e

    def rollback_to_version(self, media_id: int, target_version_number: int) -> Dict[str, Any]:
        if not isinstance(target_version_number, int) or target_version_number < 1: raise ValueError("Target version invalid.")
        client_id = self.client_id; current_time = self._get_current_utc_timestamp_str(); logger.debug(f"Rolling back media {media_id} to doc version {target_version_number}.")
        try:
            with self.transaction() as conn:
                cursor = conn.cursor()
                version_info = self._get_next_version(conn, "Media", "id", media_id)
                if version_info is None: return {'error': f'Media {media_id} not found/deleted.'}
                current_media_version, new_media_version = version_info
                # Use standalone function which requires db_instance (self)
                target_version_data = get_document_version(self, media_id, target_version_number, True)
                if target_version_data is None: return {'error': f'Rollback target version {target_version_number} not found/inactive.'}
                cursor.execute("SELECT MAX(version_number) FROM DocumentVersions WHERE media_id=? AND deleted=0", (media_id,)); latest_vn_res = cursor.fetchone()
                if latest_vn_res and target_version_number == latest_vn_res[0]: return {'error': 'Cannot rollback to the current latest version number.'}
                target_content = target_version_data.get('content'); target_prompt = target_version_data.get('prompt'); target_analysis = target_version_data.get('analysis_content')
                if target_content is None: return {'error': f'Version {target_version_number} has no content.'}
                # Call internal method, assumes it runs in current transaction
                new_version_info = self.create_document_version(media_id=media_id, content=target_content, prompt=target_prompt, analysis_content=target_analysis)
                new_doc_version_number = new_version_info.get('version_number')
                new_content_hash = hashlib.sha256(target_content.encode()).hexdigest()
                cursor.execute('UPDATE Media SET content=?, content_hash=?, last_modified=?, version=?, client_id=?, chunking_status="pending", vector_processing=0 WHERE id=? AND version=?', (target_content, new_content_hash, current_time, new_media_version, client_id, media_id, current_media_version))
                if cursor.rowcount == 0: raise ConflictError("Media", media_id)
            logger.info(f"Rolled back media {media_id} to state of doc ver {target_version_number}. New DocVer: {new_doc_version_number}, New MediaVer: {new_media_version}")
            return {'success': f'Rolled back to version {target_version_number}. State saved as new version {new_doc_version_number}.', 'new_version_number': new_doc_version_number, 'new_media_version': new_media_version}
        except (InputError, ValueError, ConflictError, DatabaseError, sqlite3.Error, TypeError) as e: logger.error(f"Rollback error media {media_id}: {e}"); raise e if isinstance(e, (InputError, ValueError, ConflictError, DatabaseError, TypeError)) else DatabaseError(f"DB error during rollback: {e}") from e
        except Exception as e: logger.error(f"Unexpected rollback error media {media_id}: {e}"); raise DatabaseError(f"Unexpected rollback error: {e}") from e

    def process_unvectorized_chunks(self, media_id: int, chunks: List[Dict[str, Any]], batch_size: int = 100):
        if not chunks: logger.warning(f"process_unvectorized_chunks empty list for media {media_id}."); return
        client_id = self.client_id # Use instance client_id
        start_time = time.time(); total_chunks = len(chunks); processed_count = 0; logger.info(f"Processing {total_chunks} unvectorized chunks for media {media_id}.")
        try:
            # Check parent active status outside transaction maybe?
            if not check_media_exists(self, media_id=media_id): # Use standalone check function
                 raise InputError(f"Cannot add chunks: Parent Media {media_id} not found or deleted.")
            with self.transaction() as conn:
                for i in range(0, total_chunks, batch_size):
                    batch = chunks[i:i + batch_size]; chunk_params = []; current_time = self._get_current_utc_timestamp_str()
                    for chunk_dict in batch:
                        chunk_uuid = self._generate_uuid(); chunk_text = chunk_dict.get('chunk_text', chunk_dict.get('text')); chunk_index = chunk_dict.get('chunk_index')
                        if chunk_text is None or chunk_index is None: logger.warning(f"Skipping chunk missing text/index media {media_id}"); continue
                        chunk_params.append((media_id, chunk_text, chunk_index, chunk_dict.get('start_char'), chunk_dict.get('end_char'), chunk_dict.get('chunk_type'), chunk_dict.get('metadata'), chunk_uuid, current_time, 1, client_id, 0))
                    if not chunk_params: continue
                    sql = "INSERT INTO UnvectorizedMediaChunks (media_id, chunk_text, chunk_index, start_char, end_char, chunk_type, metadata, uuid, last_modified, version, client_id, deleted) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                    cursor = conn.cursor(); cursor.executemany(sql, chunk_params); processed_count += len(chunk_params); logger.debug(f"Processed {processed_count}/{total_chunks} unvectorized chunks media {media_id}")
            duration = time.time() - start_time; logger.info(f"Finished processing {processed_count} unvectorized chunks media {media_id}. Duration: {duration:.4f}s")
        except (InputError, DatabaseError, sqlite3.Error) as e: logger.error(f"Error processing unvectorized chunks media {media_id}: {e}"); raise e if isinstance(e, (InputError, DatabaseError)) else DatabaseError(f"Failed process chunks: {e}") from e
        except Exception as e: logger.error(f"Unexpected chunk processing error media {media_id}: {e}"); raise DatabaseError(f"Unexpected chunk error: {e}") from e

    # --- Read Methods (Implemented within class for convenience or standalone) ---
    # These mostly use execute_query and don't handle sync meta directly, just filter.

    def fetch_all_keywords(self) -> List[str]:
        """Fetches all *active* (non-deleted) keywords."""
        try:
            cursor = self.execute_query('SELECT keyword FROM Keywords WHERE deleted = 0 ORDER BY keyword COLLATE NOCASE')
            return [row['keyword'] for row in cursor.fetchall()]
        except DatabaseError as e:
            logger.error(f"Error fetching keywords: {e}")
            raise # Re-raise

    # FIXME
    def get_media_by_id(self, media_id: int) -> Optional[Dict]:
        pass
    def get_media_by_url(self, url: str) -> Optional[Dict]:
        pass
    def get_media_by_hash(self, content_hash: str) -> Optional[Dict]:
        pass
    def get_media_by_title(self, title: str) -> Optional[Dict]:
        pass

def get_document_version(
        db_instance: Database, # REQUIRED first argument
        media_id: int,
        version_number: Optional[int] = None, # Local version number
        include_content: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Get a specific *active* document version or the latest *active* version
    for an *active* media item, using the provided Database instance.

    Args:
        db_instance: The Database instance for the specific user DB.
        media_id: ID of the media item.
        version_number: Specific local version number. If None, gets latest active.
        include_content: Whether to include the full 'content' field.

    Returns:
        Dictionary representing the version, or None if not found/inactive.
        Includes sync metadata fields.

    Raises: DatabaseError, ValueError, TypeError.
    """
    if not isinstance(db_instance, Database): raise TypeError("db_instance must be a Database object.")
    if version_number is not None and (not isinstance(version_number, int) or version_number < 1):
        raise ValueError("Version number must be a positive integer.")

    log_msg = f"Getting {'latest' if version_number is None else f'version {version_number}'} for media_id={media_id}"
    logger.debug(f"{log_msg} (active only) from DB: {db_instance.db_path_str}")

    try:
        # Base columns to select
        select_cols = ("dv.id, dv.uuid, dv.media_id, dv.version_number, dv.created_at, "
                       "dv.prompt, dv.analysis_content, dv.last_modified, dv.version, "
                       "dv.client_id, dv.deleted")
        if include_content:
            select_cols += ", dv.content"

        params = [media_id]
        # Base query joins Media and filters for active records
        query_base = f"""
            FROM DocumentVersions dv
            JOIN Media m ON dv.media_id = m.id
            WHERE dv.media_id = ? AND dv.deleted = 0 AND m.deleted = 0
        """
        order_limit = ""

        if version_number is None:
            # Get latest active version by local version_number
            order_limit = "ORDER BY dv.version_number DESC LIMIT 1"
        else:
            # Get specific active version by local version_number
            query_base += " AND dv.version_number = ?"
            params.append(version_number)

        final_query = f"SELECT {select_cols} {query_base} {order_limit}"

        # Use the execute_query method of the passed instance
        cursor = db_instance.execute_query(final_query, tuple(params))
        result = cursor.fetchone() # Fetch using the row factory

        if not result:
            logger.warning(f"Active version {'latest' if version_number is None else version_number} not found for active media_id {media_id}")
            return None

        return dict(result) # Convert Row to dict

    except (DatabaseError, sqlite3.Error) as e:
        logger.error(f"Error retrieving {log_msg} on DB '{db_instance.db_path_str}': {e}", exc_info=True)
        # Decide: return None or raise? Raising seems more consistent for DB errors.
        raise DatabaseError(f"Database error retrieving version: {e}") from e
    except Exception as e:
         logger.error(f"Unexpected error retrieving {log_msg} on DB '{db_instance.db_path_str}': {e}", exc_info=True)
         raise DatabaseError(f"Unexpected error retrieving version: {e}") from e


# =========================================================================
# Standalone Functions (REQUIRE db_instance passed explicitly)
# =========================================================================

# --- Backup & Integrity (Operate on path, no db_instance needed) ---
def create_incremental_backup(db_path, backup_dir): ... # Keep implementation
def create_automated_backup(db_path, backup_dir): ... # Keep implementation
def rotate_backups(backup_dir, max_backups=10): ... # Keep implementation
def check_database_integrity(db_path): ... # Keep implementation

# --- Utility Checks (Require db_instance) ---
def is_valid_date(date_string: str) -> bool:
    if not date_string: return False
    try: datetime.strptime(date_string, '%Y-%m-%d'); return True
    except (ValueError, TypeError): return False

def check_media_exists(db_instance: Database, media_id: Optional[int] = None, url: Optional[str] = None, content_hash: Optional[str] = None) -> Optional[int]:
    """Checks if *active* media exists by ID, URL, or hash."""
    if not isinstance(db_instance, Database): raise TypeError("db_instance required.")
    query_parts = []
    params = []
    if media_id is not None: query_parts.append("id = ?"); params.append(media_id)
    if url: query_parts.append("url = ?"); params.append(url)
    if content_hash: query_parts.append("content_hash = ?"); params.append(content_hash)
    if not query_parts: raise ValueError("Must provide id, url, or content_hash to check.")

    query = f"SELECT id FROM Media WHERE ({' OR '.join(query_parts)}) AND deleted = 0 LIMIT 1"
    try:
        cursor = db_instance.execute_query(query, tuple(params))
        result = cursor.fetchone()
        return result['id'] if result else None
    except (DatabaseError, sqlite3.Error) as e:
        logger.error(f"Error checking media existence on DB '{db_instance.db_path_str}': {e}")
        return None


# --- Example Implementation for a remaining function ---
def empty_trash(db_instance: Database, days_threshold: int) -> Tuple[int, int]:
    """Moves items older than threshold from UI trash to sync delete state."""
    if not isinstance(db_instance, Database): raise TypeError("db_instance required.")
    if not isinstance(days_threshold, int) or days_threshold < 0: raise ValueError("Days must be non-negative int.")

    threshold_date_str = (datetime.now(timezone.utc) - timedelta(days=days_threshold)).strftime('%Y-%m-%d %H:%M:%S')
    processed_count = 0
    logger.info(f"Emptying trash older than {days_threshold} days ({threshold_date_str}) on DB {db_instance.db_path_str}")

    try:
        # Find items to process first (read-only)
        cursor_find = db_instance.execute_query("""
             SELECT id, title FROM Media
             WHERE is_trash = 1 AND deleted = 0 AND trash_date <= ?
         """, (threshold_date_str,))
        items_to_process = cursor_find.fetchall()

        if not items_to_process:
             logger.info("No items found in trash older than threshold.")
        else:
            logger.info(f"Found {len(items_to_process)} items to process.")
            for item in items_to_process:
                 media_id = item['id']
                 logger.debug(f"Processing item ID {media_id} ('{item['title']}') for sync delete from trash.")
                 try:
                     # Call the instance method which handles transaction, meta, cascade
                     success = db_instance.soft_delete_media(media_id=media_id, cascade=True)
                     if success:
                         processed_count += 1
                     else:
                         # soft_delete_media logs failure details
                         logger.warning(f"Failed to process item ID {media_id} during trash emptying (already deleted or conflict?).")
                 except ConflictError as e:
                      logger.warning(f"Conflict processing item ID {media_id} during trash emptying: {e}")
                 except DatabaseError as e:
                      logger.error(f"Database error processing item ID {media_id} during trash emptying: {e}")
                      # Decide whether to continue or stop the whole process
                      # For now, log and continue
                 except Exception as e:
                      logger.error(f"Unexpected error processing item ID {media_id} during trash emptying: {e}", exc_info=True)

        # Get final count of items remaining in UI trash (and not sync deleted)
        cursor_remain = db_instance.execute_query("SELECT COUNT(*) FROM Media WHERE is_trash = 1 AND deleted = 0")
        remaining_count = cursor_remain.fetchone()[0]

        logger.info(f"Trash emptying complete. Processed (sync deleted): {processed_count}. Remaining in UI trash: {remaining_count}.")
        return processed_count, remaining_count

    except (DatabaseError, sqlite3.Error) as e:
        logger.error(f"Error emptying trash on DB '{db_instance.db_path_str}': {e}", exc_info=True)
        return 0, -1 # Indicate error
    except Exception as e:
        logger.error(f"Unexpected error emptying trash on DB '{db_instance.db_path_str}': {e}", exc_info=True)
        return 0, -1

# --- Utility Checks (Continued) ---

def check_media_and_whisper_model(db_instance: Database, title: Optional[str]=None, url: Optional[str]=None, current_whisper_model: Optional[str]=None) -> Tuple[bool, str]:
    """
    DEPRECATED LIKELY: Checks if *active* media exists and compares whisper model from *latest active* Transcript.
    Prefer check_should_process_by_url which checks the Media table directly.
    Returns (should_process, reason)
    """
    if not isinstance(db_instance, Database): raise TypeError("db_instance required.")
    if not title and not url: return True, "No title or URL provided"
    logger.warning("check_media_and_whisper_model is likely deprecated; prefer checking Media.transcription_model")

    media_id = check_media_exists(db_instance, url=url) # Check active media
    if not media_id:
        return True, "Media not found or is deleted"

    try:
        # Get the latest *active* transcript for this media
        cursor = db_instance.execute_query("""
            SELECT whisper_model FROM Transcripts
            WHERE media_id = ? AND deleted = 0
            ORDER BY created_at DESC LIMIT 1
        """, (media_id,))
        transcript_result = cursor.fetchone()

        if not transcript_result:
            return True, f"No active transcript found for media (ID: {media_id})"

        db_whisper_model = transcript_result['whisper_model']

        if not db_whisper_model:
             return True, f"Active transcript for media (ID: {media_id}) has no model info."

        if not current_whisper_model:
            # If checking without a current model, assume no need to re-process based on model match
            return False, f"Media found (ID: {media_id}) with model '{db_whisper_model}', no current model specified for comparison."

        if db_whisper_model != current_whisper_model:
            return True, f"Different whisper model (DB: {db_whisper_model}, Current: {current_whisper_model})"
        else:
            return False, f"Media found with same whisper model (ID: {media_id})"

    except (DatabaseError, sqlite3.Error) as e:
        logger.error(f"Error checking media/whisper model for ID {media_id}: {e}", exc_info=True)
        # Fail safe? Allow processing on error?
        return True, f"Database error during whisper model check: {e}"


# --- Media Processing State (Requires db_instance) ---

def get_unprocessed_media(db_instance: Database) -> List[Dict]:
    """Gets active media items pending vector processing."""
    if not isinstance(db_instance, Database): raise TypeError("db_instance required.")
    try:
        query = """
        SELECT id, uuid, content, type, title
        FROM Media
        WHERE vector_processing = 0 AND deleted = 0 AND is_trash = 0
        ORDER BY id
        """
        cursor = db_instance.execute_query(query)
        return [dict(row) for row in cursor.fetchall()]
    except (DatabaseError, sqlite3.Error) as e:
        logger.error(f"Error getting unprocessed media on DB '{db_instance.db_path_str}': {e}")
        raise DatabaseError("Failed to get unprocessed media") from e

def mark_media_as_processed(db_instance: Database, media_id: int):
    """Marks media vector_processing=1. DOES NOT update sync metadata."""
    # Note: This updates a local-only processing flag. It should generally NOT
    # increment the main record's sync 'version' or 'last_modified'.
    # If these flags ARE meant to be synced, this needs full sync meta handling.
    if not isinstance(db_instance, Database): raise TypeError("db_instance required.")
    logger.debug(f"Marking media {media_id} vector_processing=1 on DB '{db_instance.db_path_str}'.")
    try:
        # Use a direct query, no transaction needed unless part of larger workflow
        cursor = db_instance.execute_query(
            "UPDATE Media SET vector_processing = 1 WHERE id = ?",
            (media_id,),
            commit=True # Commit this specific change
        )
        if cursor.rowcount == 0:
             logger.warning(f"Attempted to mark media {media_id} as processed, but it was not found.")
             # Raise error or just log? Let's log for now.
             # raise ValueError(f"Media ID {media_id} not found.")
    except (DatabaseError, sqlite3.Error) as e:
        logger.error(f"Error marking media {media_id} as processed on '{db_instance.db_path_str}': {e}")
        raise DatabaseError(f"Failed to mark media {media_id} processed") from e

# --- Ingestion Wrappers (Require db_instance) ---

def ingest_article_to_db_new(
    db_instance: Database, # REQUIRED
    *, # Force keyword arguments
    url: str,
    title: str,
    content: str,
    author: Optional[str] = None,
    keywords: Optional[List[str]] = None,
    summary: Optional[str] = None, # Maps to analysis_content
    ingestion_date: Optional[str] = None,
    custom_prompt: Optional[str] = None, # Maps to prompt
    overwrite: bool = False
) -> Tuple[Optional[int], Optional[str], str]:
    """Ingests a web article using the provided Database instance by calling its internal method."""
    if not isinstance(db_instance, Database): raise TypeError("db_instance required.")
    if not url or not title or content is None: raise InputError("URL, Title, and Content are required.")
    # Call the Database method, which handles client_id, versioning etc internally
    return db_instance.add_media_with_keywords(
        url=url,
        title=title,
        media_type='article', # Hardcode type
        content=content,
        keywords=keywords,
        prompt=custom_prompt,
        analysis_content=summary,
        author=author,
        ingestion_date=ingestion_date,
        overwrite=overwrite
        # chunk_options and segments could be passed if needed
    )

def import_obsidian_note_to_db(db_instance: Database, note_data: Dict[str, Any]) -> Tuple[Optional[int], Optional[str], str]:
    """Imports an Obsidian note using the provided Database instance by calling its internal method."""
    if not isinstance(db_instance, Database): raise TypeError("db_instance required.")
    required_keys = ['title', 'content']
    if not all(key in note_data for key in required_keys):
        raise InputError(f"Obsidian note data missing required keys: {required_keys}")

    # Use title or a generated ID as URL if file_path isn't stable/desired
    url_identifier = f"obsidian://{note_data['title']}" # Example identifier

    # Map Obsidian tags to keywords
    keywords = note_data.get('tags', [])

    # Store frontmatter as analysis_content in the initial version
    frontmatter_str = None
    if 'frontmatter' in note_data and isinstance(note_data['frontmatter'], dict):
        try:
            frontmatter_str = yaml.dump(note_data['frontmatter'])
        except ImportError:
            logger.warning("PyYAML not installed. Cannot store Obsidian frontmatter.")
        except Exception as e:
            logger.error(f"Error dumping frontmatter to YAML: {e}")

    return db_instance.add_media_with_keywords(
        url=url_identifier,
        title=note_data['title'],
        media_type='obsidian_note',
        content=note_data['content'],
        keywords=keywords,
        author=note_data.get('frontmatter', {}).get('author'), # Get author from frontmatter if exists
        # Use 'Obsidian Frontmatter' as the prompt for the initial version
        prompt="Obsidian Frontmatter" if frontmatter_str else None,
        analysis_content=frontmatter_str,
        # Overwrite logic determined by caller or default (False)
        overwrite=note_data.get('overwrite', False) # Allow passing overwrite flag in note_data
    )

# --- Transcript/Analysis/Prompt Reads (Require db_instance) ---

def get_media_transcripts(db_instance: Database, media_id: int) -> List[Dict]:
    """Gets all *active* transcripts for an *active* media item."""
    if not isinstance(db_instance, Database): raise TypeError("db_instance required.")
    logger.debug(f"Fetching transcripts for media_id={media_id} from DB: {db_instance.db_path_str}")
    try:
        query = """
            SELECT t.id, t.uuid, t.whisper_model, t.transcription, t.created_at,
                   t.last_modified, t.version, t.client_id
            FROM Transcripts t
            JOIN Media m ON t.media_id = m.id
            WHERE t.media_id = ? AND t.deleted = 0 AND m.deleted = 0
            ORDER BY t.created_at DESC
        """
        cursor = db_instance.execute_query(query, (media_id,))
        return [dict(row) for row in cursor.fetchall()]
    except (DatabaseError, sqlite3.Error) as e:
        logger.error(f"Error getting transcripts for media {media_id} on '{db_instance.db_path_str}': {e}")
        return [] # Return empty on error

def get_latest_transcription(db_instance: Database, media_id: int) -> Optional[str]:
     """Gets text of latest *active* transcript for an *active* media item."""
     if not isinstance(db_instance, Database): raise TypeError("db_instance required.")
     try:
         query = """
             SELECT t.transcription FROM Transcripts t
             JOIN Media m ON t.media_id = m.id
             WHERE t.media_id = ? AND t.deleted = 0 AND m.deleted = 0
             ORDER BY t.created_at DESC LIMIT 1
         """
         cursor = db_instance.execute_query(query, (media_id,))
         result = cursor.fetchone()
         return result['transcription'] if result else None
     except (DatabaseError, sqlite3.Error) as e:
         logger.error(f"Error getting latest transcript text media {media_id} on '{db_instance.db_path_str}': {e}")
         return None # Or raise? Return None for not found / error

def get_specific_transcript(db_instance: Database, transcript_uuid: str) -> Optional[Dict]:
     """Gets a specific *active* transcript by UUID."""
     if not isinstance(db_instance, Database): raise TypeError("db_instance required.")
     try:
         query = """
             SELECT t.id, t.uuid, t.media_id, t.whisper_model, t.transcription, t.created_at,
                    t.last_modified, t.version, t.client_id
             FROM Transcripts t
             JOIN Media m ON t.media_id = m.id
             WHERE t.uuid = ? AND t.deleted = 0 AND m.deleted = 0
         """
         cursor = db_instance.execute_query(query, (transcript_uuid,))
         result = cursor.fetchone()
         return dict(result) if result else None
     except (DatabaseError, sqlite3.Error) as e:
         logger.error(f"Error getting transcript UUID {transcript_uuid} on '{db_instance.db_path_str}': {e}")
         return None # Or raise?

def get_specific_analysis(db_instance: Database, version_uuid: str) -> Optional[str]:
    """Gets analysis_content from a specific *active* DocumentVersion by UUID."""
    if not isinstance(db_instance, Database): raise TypeError("db_instance required.")
    try:
        query = """
            SELECT dv.analysis_content FROM DocumentVersions dv
            JOIN Media m ON dv.media_id = m.id
            WHERE dv.uuid = ? AND dv.deleted = 0 AND m.deleted = 0
        """
        cursor = db_instance.execute_query(query, (version_uuid,))
        result = cursor.fetchone()
        return result['analysis_content'] if result else None
    except (DatabaseError, sqlite3.Error) as e:
         logger.error(f"Error getting analysis for version UUID {version_uuid} on '{db_instance.db_path_str}': {e}")
         return None

def get_media_prompts(db_instance: Database, media_id: int) -> List[Dict]:
     """Gets all non-empty prompts from *active* DocumentVersions for an *active* media item."""
     if not isinstance(db_instance, Database): raise TypeError("db_instance required.")
     try:
         query = """
             SELECT dv.id, dv.uuid, dv.prompt, dv.created_at
             FROM DocumentVersions dv
             JOIN Media m ON dv.media_id = m.id
             WHERE dv.media_id = ? AND dv.deleted = 0 AND m.deleted = 0
               AND dv.prompt IS NOT NULL AND dv.prompt != ''
             ORDER BY dv.version_number DESC
         """
         cursor = db_instance.execute_query(query, (media_id,))
         # Return dicts containing id, uuid, content (prompt text), created_at
         return [{'id': row['id'], 'uuid': row['uuid'], 'content': row['prompt'], 'created_at': row['created_at']} for row in cursor.fetchall()]
     except (DatabaseError, sqlite3.Error) as e:
         logger.error(f"Error getting prompts for media {media_id} on '{db_instance.db_path_str}': {e}")
         return []

def get_specific_prompt(db_instance: Database, version_uuid: str) -> Optional[str]:
    """Gets prompt from a specific *active* DocumentVersion by UUID."""
    if not isinstance(db_instance, Database): raise TypeError("db_instance required.")
    try:
        query = """
            SELECT dv.prompt FROM DocumentVersions dv
            JOIN Media m ON dv.media_id = m.id
            WHERE dv.uuid = ? AND dv.deleted = 0 AND m.deleted = 0
        """
        cursor = db_instance.execute_query(query, (version_uuid,))
        result = cursor.fetchone()
        return result['prompt'] if result else None
    except (DatabaseError, sqlite3.Error) as e:
         logger.error(f"Error getting prompt for version UUID {version_uuid} on '{db_instance.db_path_str}': {e}")
         return None

# --- Specific Deletes (Need Soft Delete Implementation) ---

def soft_delete_transcript(db_instance: Database, transcript_uuid: str) -> bool:
    """Soft deletes a specific transcript by UUID. Uses instance client_id."""
    if not isinstance(db_instance, Database): raise TypeError("db_instance required.")
    if not transcript_uuid: raise InputError("Transcript UUID required.")
    current_time = db_instance._get_current_utc_timestamp_str(); client_id = db_instance.client_id
    logger.debug(f"Attempting soft delete Transcript UUID: {transcript_uuid}")
    try:
        with db_instance.transaction() as conn:
             cursor = conn.cursor()
             cursor.execute("SELECT id, version FROM Transcripts WHERE uuid = ? AND deleted = 0", (transcript_uuid,))
             transcript_info = cursor.fetchone()
             if not transcript_info: logger.warning(f"Transcript UUID {transcript_uuid} not found/deleted."); return False
             transcript_id, current_version = transcript_info['id'], transcript_info['version']; new_version = current_version + 1
             cursor.execute("UPDATE Transcripts SET deleted=1, last_modified=?, version=?, client_id=? WHERE id=? AND version=?", (current_time, new_version, client_id, transcript_id, current_version))
             if cursor.rowcount == 0: raise ConflictError("Transcripts", transcript_id)
             logger.info(f"Soft deleted Transcript UUID {transcript_uuid}. New ver: {new_version}")
             return True
    except (InputError, ConflictError, DatabaseError, sqlite3.Error) as e: logger.error(f"Error soft delete Transcript UUID {transcript_uuid}: {e}"); raise e if isinstance(e, (InputError, ConflictError, DatabaseError)) else DatabaseError(f"Failed soft delete transcript: {e}") from e
    except Exception as e: logger.error(f"Unexpected soft delete Transcript error UUID {transcript_uuid}: {e}"); raise DatabaseError(f"Unexpected transcript soft delete error: {e}") from e

# soft_delete_document_version is already implemented as instance method

def clear_specific_analysis(db_instance: Database, version_uuid: str) -> bool:
    """Sets analysis_content to NULL for a specific active DocumentVersion. Increments version."""
    if not isinstance(db_instance, Database): raise TypeError("db_instance required.")
    if not version_uuid: raise InputError("Version UUID required.")
    current_time = db_instance._get_current_utc_timestamp_str(); client_id = db_instance.client_id
    logger.debug(f"Clearing analysis for DocVersion UUID: {version_uuid}")
    try:
        with db_instance.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, version FROM DocumentVersions WHERE uuid = ? AND deleted = 0", (version_uuid,))
            version_info = cursor.fetchone()
            if not version_info: logger.warning(f"DocVersion UUID {version_uuid} not found/deleted."); return False
            version_id, current_version = version_info['id'], version_info['version']; new_version = current_version + 1
            cursor.execute("UPDATE DocumentVersions SET analysis_content=NULL, last_modified=?, version=?, client_id=? WHERE id=? AND version=?", (current_time, new_version, client_id, version_id, current_version))
            if cursor.rowcount == 0: raise ConflictError("DocumentVersions", version_id)
            logger.info(f"Cleared analysis for DocVersion UUID {version_uuid}. New ver: {new_version}")
            return True
    except (InputError, ConflictError, DatabaseError, sqlite3.Error) as e: logger.error(f"Error clearing analysis UUID {version_uuid}: {e}"); raise e if isinstance(e, (InputError, ConflictError, DatabaseError)) else DatabaseError(f"Failed clear analysis: {e}") from e
    except Exception as e: logger.error(f"Unexpected error clearing analysis UUID {version_uuid}: {e}"); raise DatabaseError(f"Unexpected clear analysis error: {e}") from e

def clear_specific_prompt(db_instance: Database, version_uuid: str) -> bool:
    """Sets prompt to NULL for a specific active DocumentVersion. Increments version."""
    if not isinstance(db_instance, Database): raise TypeError("db_instance required.")
    if not version_uuid: raise InputError("Version UUID required.")
    current_time = db_instance._get_current_utc_timestamp_str(); client_id = db_instance.client_id
    logger.debug(f"Clearing prompt for DocVersion UUID: {version_uuid}")
    try:
        with db_instance.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, version FROM DocumentVersions WHERE uuid = ? AND deleted = 0", (version_uuid,))
            version_info = cursor.fetchone()
            if not version_info: logger.warning(f"DocVersion UUID {version_uuid} not found/deleted."); return False
            version_id, current_version = version_info['id'], version_info['version']; new_version = current_version + 1
            cursor.execute("UPDATE DocumentVersions SET prompt=NULL, last_modified=?, version=?, client_id=? WHERE id=? AND version=?", (current_time, new_version, client_id, version_id, current_version))
            if cursor.rowcount == 0: raise ConflictError("DocumentVersions", version_id)
            logger.info(f"Cleared prompt for DocVersion UUID {version_uuid}. New ver: {new_version}")
            return True
    except (InputError, ConflictError, DatabaseError, sqlite3.Error) as e: logger.error(f"Error clearing prompt UUID {version_uuid}: {e}"); raise e if isinstance(e, (InputError, ConflictError, DatabaseError)) else DatabaseError(f"Failed clear prompt: {e}") from e
    except Exception as e: logger.error(f"Unexpected error clearing prompt UUID {version_uuid}: {e}"); raise DatabaseError(f"Unexpected clear prompt error: {e}") from e


# --- Other Remaining Functions ---

def get_chunk_text(db_instance: Database, chunk_uuid: str) -> Optional[str]:
     """Gets text of a specific *active* chunk by UUID (assuming MediaChunks or UnvectorizedMediaChunks)."""
     # Determine which table to query based on your primary chunk table
     # Let's assume UnvectorizedMediaChunks for this example
     if not isinstance(db_instance, Database): raise TypeError("db_instance required.")
     target_table = "UnvectorizedMediaChunks" # Or "MediaChunks"
     try:
         query = f"""
             SELECT c.chunk_text FROM {target_table} c
             JOIN Media m ON c.media_id = m.id
             WHERE c.uuid = ? AND c.deleted = 0 AND m.deleted = 0
         """
         cursor = db_instance.execute_query(query, (chunk_uuid,))
         result = cursor.fetchone()
         return result['chunk_text'] if result else None
     except (DatabaseError, sqlite3.Error) as e:
         logger.error(f"Error getting chunk text UUID {chunk_uuid} on '{db_instance.db_path_str}': {e}")
         return None

def get_all_content_from_database(db_instance: Database) -> List[Dict[str, Any]]:
    """Retrieve basic info for all *active* media items."""
    if not isinstance(db_instance, Database): raise TypeError("db_instance required.")
    try:
        cursor = db_instance.execute_query("""
            SELECT id, uuid, content, title, author, type
            FROM Media WHERE deleted = 0 AND is_trash = 0
        """)
        return [dict(item) for item in cursor.fetchall()]
    except (DatabaseError, sqlite3.Error) as e:
        logger.error(f"Error retrieving all content from DB '{db_instance.db_path_str}': {e}")
        raise DatabaseError("Error retrieving all content") from e

def permanently_delete_item(db_instance: Database, media_id: int) -> bool:
    """Performs HARD delete. DANGEROUS FOR SYNC. Use with extreme caution."""
    if not isinstance(db_instance, Database): raise TypeError("db_instance required.")
    logger.warning(f"PERMANENT DELETE initiated for Media ID: {media_id} on DB {db_instance.db_path_str}.")
    try:
        with db_instance.transaction() as conn:
            cursor = conn.cursor()
            # Check existence first (optional but safer)
            cursor.execute("SELECT 1 FROM Media WHERE id = ?", (media_id,))
            if not cursor.fetchone(): logger.warning(f"Permanent delete failed: Media {media_id} not found."); return False
            # Hard delete - Cascades should handle children per schema FKs
            cursor.execute("DELETE FROM Media WHERE id = ?", (media_id,))
            deleted_count = cursor.rowcount
            if deleted_count > 0:
                 logger.info(f"Permanently deleted Media ID: {media_id}. NO sync log entry generated by triggers.")
                 return True
            else: logger.error(f"Permanent delete failed unexpectedly for Media {media_id}."); return False # Should not happen if found
    except sqlite3.Error as e:
        logger.error(f"Error permanently deleting Media {media_id}: {e}", exc_info=True)
        raise DatabaseError(f"Failed to permanently delete item: {e}") from e


# --- Keyword Read Functions ---

def fetch_all_keywords(db_instance: Database) -> List[str]:
    """Fetches all *active* keywords using the provided Database instance."""
    if not isinstance(db_instance, Database): raise TypeError("db_instance must be a Database object.")
    # Delegates to the instance method if you moved it there, otherwise implement here
    # Assuming it's an instance method now:
    # return db_instance.fetch_all_keywords()
    # OR, if kept standalone:
    try:
        cursor = db_instance.execute_query('SELECT keyword FROM Keywords WHERE deleted = 0 ORDER BY keyword COLLATE NOCASE')
        return [row['keyword'] for row in cursor.fetchall()]
    except (DatabaseError, sqlite3.Error) as e:
        logger.error(f"Error fetching all keywords on DB '{db_instance.db_path_str}': {e}")
        raise DatabaseError("Failed to fetch all keywords") from e

def fetch_keywords_for_media(media_id: int, db_instance: Database) -> List[str]:
    """ Fetches active keywords for a specific active media item using the provided Database instance."""
    if not isinstance(db_instance, Database): raise TypeError("db_instance must be a Database object.")
    logger.debug(f"Fetching keywords for media_id={media_id} from DB: {db_instance.db_path_str}")
    try:
        query = '''
            SELECT k.keyword
            FROM Keywords k
            JOIN MediaKeywords mk ON k.id = mk.keyword_id
            JOIN Media m ON mk.media_id = m.id
            WHERE mk.media_id = ? AND k.deleted = 0 AND m.deleted = 0
            ORDER BY k.keyword COLLATE NOCASE
        '''
        cursor = db_instance.execute_query(query, (media_id,))
        return [row['keyword'] for row in cursor.fetchall()]
    except (DatabaseError, sqlite3.Error) as e:
        logger.error(f"Error fetching keywords for media_id {media_id} on '{db_instance.db_path_str}': {e}", exc_info=True)
        raise DatabaseError(f"Failed to fetch keywords for media {media_id}") from e

def fetch_keywords_for_media_batch(media_ids: List[int], db_instance: Database) -> Dict[int, List[str]]:
    """ Fetches active keywords for multiple active media IDs efficiently using the provided Database instance."""
    if not isinstance(db_instance, Database): raise TypeError("db_instance must be a Database object.")
    if not media_ids: return {}

    keywords_map = {media_id: [] for media_id in media_ids}
    # Ensure media_ids are integers before creating placeholders
    safe_media_ids = [int(mid) for mid in media_ids]
    if not safe_media_ids: return {} # Handle case where conversion fails or list becomes empty
    placeholders = ','.join('?' * len(safe_media_ids))

    query = f"""
        SELECT mk.media_id, k.keyword
        FROM MediaKeywords mk
        JOIN Keywords k ON mk.keyword_id = k.id
        JOIN Media m ON mk.media_id = m.id
        WHERE mk.media_id IN ({placeholders}) AND k.deleted = 0 AND m.deleted = 0
        ORDER BY mk.media_id, k.keyword COLLATE NOCASE
    """
    try:
        cursor = db_instance.execute_query(query, tuple(safe_media_ids))
        for row in cursor.fetchall():
            keywords_map[row['media_id']].append(row['keyword'])
        return keywords_map
    except (DatabaseError, sqlite3.Error) as e:
         logger.error(f"Failed to fetch keywords batch on '{db_instance.db_path_str}': {e}", exc_info=True)
         raise DatabaseError("Failed to fetch keywords batch") from e

# --- Search Function ---

def search_media_db(
        db_instance: Database, # REQUIRED first argument
        search_query: Optional[str],
        search_fields: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None,
        page: int = 1,
        results_per_page: int = 20,
        include_trash: bool = False,
        include_deleted: bool = False
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Search for media items using the provided Database instance, FTS, keywords, etc.
    """
    if not isinstance(db_instance, Database): raise TypeError("db_instance must be a Database object.")
    # --- Input Validation ---
    if page < 1: raise ValueError("Page number must be 1 or greater")
    if results_per_page < 1: raise ValueError("Results per page must be 1 or greater")
    if search_query and not search_fields: search_fields = ["title", "content"]
    elif not search_fields: search_fields = []
    valid_fields = {"title", "content", "author", "type"}
    sanitized_fields = [field for field in search_fields if field in valid_fields]
    keyword_list = [k.strip().lower() for k in keywords if k and k.strip()] if keywords else []
    if not search_query and not keyword_list: logging.debug("Executing search with no query or keywords.")

    # --- Query Building ---
    offset = (page - 1) * results_per_page
    base_params = []
    conditions = []
    joins = []
    if not include_deleted: conditions.append("m.deleted = 0")
    if not include_trash: conditions.append("m.is_trash = 0")

    if keyword_list:
        # Using EXISTS subquery for potentially better performance than JOIN for filtering
        kw_placeholders = ','.join('?' * len(keyword_list))
        conditions.append(f"""
            EXISTS (
                SELECT 1 FROM MediaKeywords mk JOIN Keywords k ON mk.keyword_id = k.id
                WHERE mk.media_id = m.id AND k.deleted = 0 AND k.keyword IN ({kw_placeholders})
                GROUP BY mk.media_id
                HAVING COUNT(DISTINCT k.id) = ?
            )
        """)
        base_params.extend(keyword_list)
        base_params.append(len(keyword_list)) # Count for HAVING

    fts_search_requested = "title" in sanitized_fields or "content" in sanitized_fields
    like_fields = {"author", "type"}
    like_search_requested = list(set(sanitized_fields) & like_fields)

    if search_query:
        if fts_search_requested:
            joins.append("JOIN media_fts fts ON fts.rowid = m.id")
            conditions.append("fts.media_fts MATCH ?")
            # Escape FTS query if necessary (e.g., handle quotes, operators) - simple pass-through for now
            base_params.append(search_query)
        if like_search_requested:
            like_conditions = []
            for field in like_search_requested:
                like_conditions.append(f"m.{field} LIKE ? COLLATE NOCASE")
                base_params.append(f"%{search_query}%")
            if like_conditions: conditions.append(f"({' OR '.join(like_conditions)})")
        elif not fts_search_requested and not like_search_requested and sanitized_fields:
             logging.warning(f"Search query provided but no searchable fields selected. Query ignored.")

    join_clause = " ".join(joins)
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # --- Database Interaction ---
    try:
        # Use transaction for consistency (optional for read, but doesn't hurt)
        with db_instance.transaction():
            cursor = db_instance.get_connection().cursor()

            count_query = f"SELECT COUNT(m.id) FROM Media m {join_clause} WHERE {where_clause}"
            logging.debug(f"Search Count Query: {count_query} | Params: {base_params}")
            cursor.execute(count_query, tuple(base_params))
            total_matches = cursor.fetchone()[0]

            results_list = []
            if total_matches > 0 and offset < total_matches:
                results_query = f""" SELECT m.id, m.uuid, m.url, m.title, m.type, m.content, m.author, m.ingestion_date,
                                        m.transcription_model, m.is_trash, m.trash_date, m.chunking_status, m.vector_processing,
                                        m.content_hash, m.last_modified, m.version, m.client_id, m.deleted
                                    FROM Media m {join_clause} WHERE {where_clause}
                                    ORDER BY m.last_modified DESC, m.id DESC LIMIT ? OFFSET ? """
                paginated_params = tuple(base_params + [results_per_page, offset])
                logging.debug(f"Search Results Query: {results_query} | Params: {paginated_params}")
                cursor.execute(results_query, paginated_params)
                results_list = [dict(row) for row in cursor.fetchall()]
                # Optionally fetch keywords batch here using fetch_keywords_for_media_batch(ids, db_instance)
            return results_list, total_matches
    except (sqlite3.Error, DatabaseError) as e:
        if "no such table: media_fts" in str(e):
             logger.error(f"FTS table missing on DB '{db_instance.db_path_str}'", exc_info=True)
             raise DatabaseError("FTS table 'media_fts' not found.") from e
        logger.error(f"Error in search_media_db on '{db_instance.db_path_str}': {e}", exc_info=True)
        raise DatabaseError(f"Failed to search media database: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error in search_media_db on '{db_instance.db_path_str}': {e}", exc_info=True)
        raise DatabaseError(f"An unexpected error occurred during media search: {e}") from e


