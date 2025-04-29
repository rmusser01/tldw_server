# Characters_and_RAG_DB.py
# Database functions for managing character cards and chat histories.
# #
# Imports
import configparser
import sqlite3
import json
import os
import sys
from typing import List, Dict, Optional, Tuple, Any, Union
#
# Local Imports
from tldw_Server_API.app.core.Utils.Utils import get_database_dir, get_project_relative_path, get_database_path, logging
#
#######################################################################################################################
#
# Functions:

# Schema
rag_char_chat_schema = """
PRAGMA foreign_keys = ON;

-- ───────────────────────────────────────────────────────────────────────────
-- 1. Character profiles (with sync metadata + FTS5)
-- ───────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS character_cards (
    id                        INTEGER   PRIMARY KEY AUTOINCREMENT,
    name                      TEXT      UNIQUE NOT NULL,
    description               TEXT,
    personality               TEXT,
    scenario                  TEXT,
    image                     BLOB,
    post_history_instructions TEXT,
    first_message             TEXT,
    message_example           TEXT,
    creator_notes             TEXT,
    system_prompt             TEXT,
    alternate_greetings       TEXT,
    tags                      TEXT,
    creator                   TEXT,
    character_version         TEXT,
    extensions                TEXT,
    created_at                DATETIME  NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_modified             DATETIME  NOT NULL DEFAULT CURRENT_TIMESTAMP, -- Added for sync
    deleted                   BOOLEAN   NOT NULL DEFAULT 0,               -- Added for sync (soft delete)
    client_id                 TEXT      NOT NULL DEFAULT 'unknown',       -- Added for sync
    version                   INTEGER   NOT NULL DEFAULT 1                -- Added for sync
);
CREATE VIRTUAL TABLE IF NOT EXISTS character_cards_fts
USING fts5(
    name,
    description,
    personality,
    scenario,
    system_prompt,
    content='character_cards',
    content_rowid='id'
);
CREATE TRIGGER IF NOT EXISTS character_cards_ai AFTER INSERT ON character_cards BEGIN
  INSERT INTO character_cards_fts(rowid, name, description, personality, scenario, system_prompt)
    VALUES (new.id, new.name, new.description, new.personality, new.scenario, new.system_prompt);
END;
CREATE TRIGGER IF NOT EXISTS character_cards_au AFTER UPDATE ON character_cards BEGIN
  UPDATE character_cards_fts
     SET name        = new.name,
         description = new.description,
         personality = new.personality,
         scenario    = new.scenario,
         system_prompt = new.system_prompt
   WHERE rowid = new.id;
END;
CREATE TRIGGER IF NOT EXISTS character_cards_ad AFTER DELETE ON character_cards BEGIN
  DELETE FROM character_cards_fts WHERE rowid = old.id;
END;

-- ───────────────────────────────────────────────────────────────────────────
-- 2. Conversations (branches) with tombstones & sync metadata
-- ───────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS conversations (
    id                       TEXT      PRIMARY KEY,
    root_id                  TEXT      NOT NULL,
    forked_from_message_id   TEXT      REFERENCES messages(id) ON DELETE SET NULL, -- Clear link if message deleted
    parent_conversation_id   TEXT      REFERENCES conversations(id) ON DELETE SET NULL, -- Clear link if parent convo deleted
    character_id             INTEGER   REFERENCES character_cards(id) ON DELETE CASCADE ON UPDATE CASCADE, -- Delete convos if character deleted
    title                    TEXT,
    rating                   INTEGER   CHECK(rating BETWEEN 1 AND 5),
    created_at               DATETIME  NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_modified            DATETIME  NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted                  BOOLEAN   NOT NULL DEFAULT 0,
    client_id                TEXT      NOT NULL,
    version                  INTEGER   NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_conversations_root   ON conversations(root_id);
CREATE INDEX IF NOT EXISTS idx_conversations_parent ON conversations(parent_conversation_id);
CREATE INDEX IF NOT EXISTS idx_conv_char          ON conversations(character_id); -- Added Index

-- FTS5 for conversation titles
CREATE VIRTUAL TABLE IF NOT EXISTS conversations_fts
USING fts5(
    title,
    content='conversations',
    content_rowid='rowid' -- Note: Using TEXT PK, rowid might not be stable if PK changes, but it's likely UUID so ok.
);
CREATE TRIGGER IF NOT EXISTS conversations_ai AFTER INSERT ON conversations BEGIN
  INSERT INTO conversations_fts(rowid, title)
    VALUES (new.rowid, new.title);
END;
CREATE TRIGGER IF NOT EXISTS conversations_au AFTER UPDATE ON conversations BEGIN
  UPDATE conversations_fts
     SET title = new.title
   WHERE rowid = old.rowid;
END;
CREATE TRIGGER IF NOT EXISTS conversations_ad AFTER DELETE ON conversations BEGIN
  DELETE FROM conversations_fts WHERE rowid = old.rowid;
END;

-- ───────────────────────────────────────────────────────────────────────────
-- 3. Messages with swipe/fork links, rankings, tombstones & sync metadata
-- ───────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS messages (
    id                   TEXT      PRIMARY KEY,
    conversation_id      TEXT      NOT NULL REFERENCES conversations(id) ON DELETE CASCADE, -- Delete messages if convo deleted
    parent_message_id    TEXT      REFERENCES messages(id) ON DELETE SET NULL, -- Clear parent link if parent message deleted
    sender               TEXT      NOT NULL,
    content              TEXT      NOT NULL,
    timestamp            DATETIME  NOT NULL DEFAULT CURRENT_TIMESTAMP, -- This is effectively created_at
    ranking              INTEGER,
    last_modified        DATETIME  NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted              BOOLEAN   NOT NULL DEFAULT 0,
    client_id            TEXT      NOT NULL,
    version              INTEGER   NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_msgs_conversation ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_msgs_parent       ON messages(parent_message_id);
CREATE INDEX IF NOT EXISTS idx_msgs_timestamp    ON messages(timestamp);
CREATE INDEX IF NOT EXISTS idx_msgs_ranking      ON messages(ranking);
CREATE INDEX IF NOT EXISTS idx_msgs_conv_ts      ON messages(conversation_id, timestamp); -- Added Index

-- FTS5 for message content
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts
USING fts5(
    content,
    content='messages',
    content_rowid='rowid' -- Note: Using TEXT PK, rowid might not be stable if PK changes, but it's likely UUID so ok.
);
CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
  INSERT INTO messages_fts(rowid, content)
    VALUES (new.rowid, new.content);
END;
CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
  UPDATE messages_fts
     SET content = new.content
   WHERE rowid = old.rowid;
END;
CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
  DELETE FROM messages_fts WHERE rowid = old.rowid;
END;

-- ───────────────────────────────────────────────────────────────────────────
-- 4. Keywords (with sync metadata + FTS5)
-- ───────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS keywords (
    id          INTEGER  PRIMARY KEY AUTOINCREMENT,
    keyword     TEXT     NOT NULL UNIQUE,
    created_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,       -- Added for consistency
    last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,       -- Added for sync
    deleted       BOOLEAN  NOT NULL DEFAULT 0,                      -- Added for sync (soft delete)
    client_id     TEXT     NOT NULL DEFAULT 'unknown',              -- Added for sync
    version       INTEGER  NOT NULL DEFAULT 1                       -- Added for sync
);
CREATE VIRTUAL TABLE IF NOT EXISTS keywords_fts
USING fts5(
    keyword,
    content='keywords',
    content_rowid='id'
);
CREATE TRIGGER IF NOT EXISTS keywords_ai AFTER INSERT ON keywords BEGIN
  INSERT INTO keywords_fts(rowid, keyword) VALUES (new.id, new.keyword);
END;
CREATE TRIGGER IF NOT EXISTS keywords_au AFTER UPDATE ON keywords BEGIN
  UPDATE keywords_fts SET keyword = new.keyword WHERE rowid = old.id;
END;
CREATE TRIGGER IF NOT EXISTS keywords_ad AFTER DELETE ON keywords BEGIN
  DELETE FROM keywords_fts WHERE rowid = old.id;
END;

-- ───────────────────────────────────────────────────────────────────────────
-- 5. Keyword Collections (with sync metadata + FTS5)
-- ───────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS keyword_collections (
    id          INTEGER  PRIMARY KEY AUTOINCREMENT,
    name        TEXT     NOT NULL UNIQUE,
    parent_id   INTEGER  REFERENCES keyword_collections(id) ON DELETE SET NULL ON UPDATE CASCADE, -- Set parent to NULL if parent deleted
    created_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,       -- Added for consistency
    last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,       -- Added for sync
    deleted       BOOLEAN  NOT NULL DEFAULT 0,                      -- Added for sync (soft delete)
    client_id     TEXT     NOT NULL DEFAULT 'unknown',              -- Added for sync
    version       INTEGER  NOT NULL DEFAULT 1                       -- Added for sync
);
CREATE VIRTUAL TABLE IF NOT EXISTS keyword_collections_fts
USING fts5(
    name,
    content='keyword_collections',
    content_rowid='id'
);
CREATE TRIGGER IF NOT EXISTS keyword_collections_ai AFTER INSERT ON keyword_collections BEGIN
  INSERT INTO keyword_collections_fts(rowid, name) VALUES (new.id, new.name);
END;
CREATE TRIGGER IF NOT EXISTS keyword_collections_au AFTER UPDATE ON keyword_collections BEGIN
  UPDATE keyword_collections_fts SET name = new.name WHERE rowid = old.id;
END;
CREATE TRIGGER IF NOT EXISTS keyword_collections_ad AFTER DELETE ON keyword_collections BEGIN
  DELETE FROM keyword_collections_fts WHERE rowid = old.id;
END;

-- ───────────────────────────────────────────────────────────────────────────
-- 6. Notes (Independent, with sync metadata + FTS5)
-- ───────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS notes (
    id            INTEGER   PRIMARY KEY AUTOINCREMENT,
    -- conversation_id removed - Notes are independent
    title         TEXT      NOT NULL,
    content       TEXT      NOT NULL,
    created_at    DATETIME  NOT NULL DEFAULT CURRENT_TIMESTAMP, -- Renamed from timestamp
    last_modified DATETIME  NOT NULL DEFAULT CURRENT_TIMESTAMP, -- Added for sync
    deleted       BOOLEAN   NOT NULL DEFAULT 0,               -- Added for sync (soft delete)
    client_id     TEXT      NOT NULL DEFAULT 'unknown',       -- Added for sync
    version       INTEGER   NOT NULL DEFAULT 1                -- Added for sync
);
CREATE INDEX IF NOT EXISTS idx_notes_last_modified ON notes(last_modified); -- Added index

CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts
USING fts5(
    title,
    content,
    content='notes',
    content_rowid='id'
);
CREATE TRIGGER IF NOT EXISTS notes_ai AFTER INSERT ON notes BEGIN
  INSERT INTO notes_fts(rowid, title, content)
    VALUES (new.id, new.title, new.content);
END;
CREATE TRIGGER IF NOT EXISTS notes_au AFTER UPDATE ON notes BEGIN
  UPDATE notes_fts
     SET title   = new.title,
         content = new.content
   WHERE rowid = old.id;
END;
CREATE TRIGGER IF NOT EXISTS notes_ad AFTER DELETE ON notes BEGIN
  DELETE FROM notes_fts WHERE rowid = old.id;
END;

-- ───────────────────────────────────────────────────────────────────────────
-- 7. Linking Tables (with created_at for potential sync ordering)
-- ───────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS conversation_keywords (
    conversation_id TEXT    NOT NULL REFERENCES conversations(id) ON DELETE CASCADE, -- Link dies if conversation dies
    keyword_id      INTEGER NOT NULL REFERENCES keywords(id) ON DELETE CASCADE ON UPDATE CASCADE, -- Link dies if keyword dies
    created_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, -- Added for sync log info
    PRIMARY KEY(conversation_id, keyword_id)
);
-- Optional: Indexes on individual columns can be useful if not covered by PK lookup patterns
CREATE INDEX IF NOT EXISTS idx_convkw_kw ON conversation_keywords(keyword_id);

CREATE TABLE IF NOT EXISTS collection_keywords (
    collection_id INTEGER NOT NULL REFERENCES keyword_collections(id) ON DELETE CASCADE ON UPDATE CASCADE, -- Link dies if collection dies
    keyword_id    INTEGER NOT NULL REFERENCES keywords(id) ON DELETE CASCADE ON UPDATE CASCADE, -- Link dies if keyword dies
    created_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, -- Added for sync log info
    PRIMARY KEY(collection_id, keyword_id)
);
-- Optional: Indexes on individual columns can be useful if not covered by PK lookup patterns
CREATE INDEX IF NOT EXISTS idx_collkw_kw ON collection_keywords(keyword_id);

CREATE TABLE IF NOT EXISTS note_keywords (
    note_id    INTEGER NOT NULL REFERENCES notes(id) ON DELETE CASCADE ON UPDATE CASCADE, -- Link dies if note dies
    keyword_id INTEGER NOT NULL REFERENCES keywords(id) ON DELETE CASCADE ON UPDATE CASCADE, -- Link dies if keyword dies
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, -- Added for sync log info
    PRIMARY KEY(note_id, keyword_id)
);
-- Optional: Indexes on individual columns can be useful if not covered by PK lookup patterns
CREATE INDEX IF NOT EXISTS idx_notekw_kw ON note_keywords(keyword_id);


-- ───────────────────────────────────────────────────────────────────────────
-- 8. Sync Log for bi‑directional, offline‑first synchronization
-- ───────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sync_log (
    change_id    INTEGER  PRIMARY KEY AUTOINCREMENT,
    entity       TEXT     NOT NULL,   -- Table name (e.g., 'conversations', 'notes', 'keywords')
    entity_id    TEXT     NOT NULL,   -- Primary Key of the record (casted to TEXT)
    operation    TEXT     NOT NULL   CHECK(operation IN ('create','update','delete')),
    timestamp    DATETIME NOT NULL,   -- Usually record.last_modified or CURRENT_TIMESTAMP for links
    client_id    TEXT     NOT NULL,   -- Source device UUID/ID
    version      INTEGER  NOT NULL,   -- Record version
    payload      TEXT     NOT NULL    -- JSON blob of new record state or minimal info for delete/links
);
CREATE INDEX IF NOT EXISTS idx_sync_log_ts     ON sync_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_sync_log_entity ON sync_log(entity, entity_id); -- Added Index


-- ───────────────────────────────────────────────────────────────────────────
-- SYNC LOG TRIGGERS --
-- ───────────────────────────────────────────────────────────────────────────

-- == Triggers for: messages ==
CREATE TRIGGER IF NOT EXISTS messages_sync_create
AFTER INSERT ON messages
BEGIN
  INSERT INTO sync_log (entity,entity_id,operation,timestamp,client_id,version,payload)
    VALUES (
      'messages', NEW.id, 'create',
       NEW.last_modified, NEW.client_id, NEW.version,
       json_object(
         'id', NEW.id, 'conversation_id', NEW.conversation_id, 'parent_message_id', NEW.parent_message_id,
         'sender', NEW.sender, 'content', NEW.content, 'timestamp', NEW.timestamp,
         'ranking', NEW.ranking, 'last_modified', NEW.last_modified, 'deleted', NEW.deleted,
         'client_id', NEW.client_id, 'version', NEW.version
       )
    );
END;

CREATE TRIGGER IF NOT EXISTS messages_sync_update
AFTER UPDATE ON messages
-- Trigger only for actual data changes relevant to sync, EXCLUDING soft delete/undelete actions handled by _sync_delete trigger
WHEN OLD.deleted = NEW.deleted AND (
     OLD.content           <> NEW.content OR
     OLD.ranking           <> NEW.ranking OR
     OLD.parent_message_id <> NEW.parent_message_id OR -- If parent link change should be synced
     OLD.last_modified     <> NEW.last_modified OR -- Ensure timestamp changes trigger sync
     OLD.version           <> NEW.version -- Ensure version bumps trigger sync
     -- Add other fields here if their changes should trigger a sync update
)
BEGIN
  INSERT INTO sync_log (entity,entity_id,operation,timestamp,client_id,version,payload)
    VALUES (
      'messages', NEW.id, 'update',
       NEW.last_modified, NEW.client_id, NEW.version,
       json_object(
         'id', NEW.id, 'conversation_id', NEW.conversation_id, 'parent_message_id', NEW.parent_message_id,
         'sender', NEW.sender, 'content', NEW.content, 'timestamp', NEW.timestamp,
         'ranking', NEW.ranking, 'last_modified', NEW.last_modified, 'deleted', NEW.deleted,
         'client_id', NEW.client_id, 'version', NEW.version
       )
    );
END;

CREATE TRIGGER IF NOT EXISTS messages_sync_delete -- Handles SOFT delete
AFTER UPDATE ON messages
WHEN OLD.deleted = 0 AND NEW.deleted = 1
BEGIN
  INSERT INTO sync_log (entity,entity_id,operation,timestamp,client_id,version,payload)
    VALUES (
      'messages', NEW.id, 'delete',
       NEW.last_modified, NEW.client_id, NEW.version,
       json_object('id', NEW.id, 'deleted', NEW.deleted) -- Include deleted flag
    );
END;

-- Optional: Trigger for UNDELETE if needed for sync
CREATE TRIGGER IF NOT EXISTS messages_sync_undelete
AFTER UPDATE ON messages
WHEN OLD.deleted = 1 AND NEW.deleted = 0
BEGIN
    -- This requires sending the full state again, similar to an update or create
    INSERT INTO sync_log (entity,entity_id,operation,timestamp,client_id,version,payload)
    VALUES (
      'messages', NEW.id, 'update', -- Or treat as 'create'/'upsert' depending on sync logic
       NEW.last_modified, NEW.client_id, NEW.version,
       json_object(
         'id', NEW.id, 'conversation_id', NEW.conversation_id, 'parent_message_id', NEW.parent_message_id,
         'sender', NEW.sender, 'content', NEW.content, 'timestamp', NEW.timestamp,
         'ranking', NEW.ranking, 'last_modified', NEW.last_modified, 'deleted', NEW.deleted,
         'client_id', NEW.client_id, 'version', NEW.version
       )
    );
END;

-- == Triggers for: conversations ==
CREATE TRIGGER IF NOT EXISTS conversations_sync_create
AFTER INSERT ON conversations
BEGIN
  INSERT INTO sync_log (entity,entity_id,operation,timestamp,client_id,version,payload)
    VALUES (
      'conversations', NEW.id, 'create',
       NEW.last_modified, NEW.client_id, NEW.version,
       json_object(
         'id', NEW.id, 'root_id', NEW.root_id, 'forked_from_message_id', NEW.forked_from_message_id,
         'parent_conversation_id', NEW.parent_conversation_id, 'character_id', NEW.character_id, 'title', NEW.title,
         'rating', NEW.rating, 'last_modified', NEW.last_modified, 'deleted', NEW.deleted,
         'client_id', NEW.client_id, 'version', NEW.version
       )
    );
END;

CREATE TRIGGER IF NOT EXISTS conversations_sync_update
AFTER UPDATE ON conversations
WHEN OLD.deleted = NEW.deleted AND (
     OLD.title                    <> NEW.title OR
     OLD.rating                   <> NEW.rating OR
     OLD.forked_from_message_id   <> NEW.forked_from_message_id OR -- If these hierarchy changes need sync
     OLD.parent_conversation_id   <> NEW.parent_conversation_id OR -- If these hierarchy changes need sync
     OLD.character_id             <> NEW.character_id OR           -- If character reassignment needs sync
     OLD.last_modified            <> NEW.last_modified OR
     OLD.version                  <> NEW.version
)
BEGIN
  INSERT INTO sync_log (entity,entity_id,operation,timestamp,client_id,version,payload)
    VALUES (
      'conversations', NEW.id, 'update',
       NEW.last_modified, NEW.client_id, NEW.version,
       json_object(
         'id', NEW.id, 'root_id', NEW.root_id, 'forked_from_message_id', NEW.forked_from_message_id,
         'parent_conversation_id', NEW.parent_conversation_id, 'character_id', NEW.character_id, 'title', NEW.title,
         'rating', NEW.rating, 'last_modified', NEW.last_modified, 'deleted', NEW.deleted,
         'client_id', NEW.client_id, 'version', NEW.version
       )
    );
END;

CREATE TRIGGER IF NOT EXISTS conversations_sync_delete -- Handles SOFT delete
AFTER UPDATE ON conversations
WHEN OLD.deleted = 0 AND NEW.deleted = 1
BEGIN
  INSERT INTO sync_log (entity,entity_id,operation,timestamp,client_id,version,payload)
    VALUES (
      'conversations', NEW.id, 'delete',
       NEW.last_modified, NEW.client_id, NEW.version,
       json_object('id', NEW.id, 'deleted', NEW.deleted)
    );
END;

-- Optional: Trigger for UNDELETE if needed for sync
CREATE TRIGGER IF NOT EXISTS conversations_sync_undelete
AFTER UPDATE ON conversations
WHEN OLD.deleted = 1 AND NEW.deleted = 0
BEGIN
    INSERT INTO sync_log (entity,entity_id,operation,timestamp,client_id,version,payload)
    VALUES (
      'conversations', NEW.id, 'update', -- Treat as update/upsert
       NEW.last_modified, NEW.client_id, NEW.version,
       json_object(
         'id', NEW.id, 'root_id', NEW.root_id, 'forked_from_message_id', NEW.forked_from_message_id,
         'parent_conversation_id', NEW.parent_conversation_id, 'character_id', NEW.character_id, 'title', NEW.title,
         'rating', NEW.rating, 'last_modified', NEW.last_modified, 'deleted', NEW.deleted,
         'client_id', NEW.client_id, 'version', NEW.version
       )
    );
END;

-- == Triggers for: character_cards ==
CREATE TRIGGER IF NOT EXISTS character_cards_sync_create
AFTER INSERT ON character_cards
BEGIN
  INSERT INTO sync_log (entity,entity_id,operation,timestamp,client_id,version,payload)
    VALUES (
      'character_cards', CAST(NEW.id AS TEXT), 'create',
       NEW.last_modified, NEW.client_id, NEW.version,
       json_object(
         'id', NEW.id, 'name', NEW.name, 'description', NEW.description, 'personality', NEW.personality,
         'scenario', NEW.scenario, -- 'image', NEW.image, -- Avoid large BLOBs in sync log payload if possible
         'post_history_instructions', NEW.post_history_instructions, 'first_message', NEW.first_message,
         'message_example', NEW.message_example, 'creator_notes', NEW.creator_notes, 'system_prompt', NEW.system_prompt,
         'alternate_greetings', NEW.alternate_greetings, 'tags', NEW.tags, 'creator', NEW.creator,
         'character_version', NEW.character_version, 'extensions', NEW.extensions, 'created_at', NEW.created_at,
         'last_modified', NEW.last_modified, 'deleted', NEW.deleted, 'client_id', NEW.client_id, 'version', NEW.version
       )
    );
END;

CREATE TRIGGER IF NOT EXISTS character_cards_sync_update
AFTER UPDATE ON character_cards
WHEN OLD.deleted = NEW.deleted AND (
     OLD.name                      <> NEW.name OR
     OLD.description               <> NEW.description OR
     OLD.personality               <> NEW.personality OR
     OLD.scenario                  <> NEW.scenario OR
     OLD.post_history_instructions <> NEW.post_history_instructions OR
     OLD.first_message             <> NEW.first_message OR
     OLD.message_example           <> NEW.message_example OR
     OLD.creator_notes             <> NEW.creator_notes OR
     OLD.system_prompt             <> NEW.system_prompt OR
     OLD.alternate_greetings       <> NEW.alternate_greetings OR
     OLD.tags                      <> NEW.tags OR
     OLD.creator                   <> NEW.creator OR
     OLD.character_version         <> NEW.character_version OR
     OLD.extensions                <> NEW.extensions OR
     -- OLD.image                  <> NEW.image OR -- Avoid comparing BLOBs directly in trigger if possible
     OLD.last_modified             <> NEW.last_modified OR
     OLD.version                   <> NEW.version
)
BEGIN
  INSERT INTO sync_log (entity,entity_id,operation,timestamp,client_id,version,payload)
    VALUES (
      'character_cards', CAST(NEW.id AS TEXT), 'update',
       NEW.last_modified, NEW.client_id, NEW.version,
       json_object(
         'id', NEW.id, 'name', NEW.name, 'description', NEW.description, 'personality', NEW.personality,
         'scenario', NEW.scenario, -- 'image', '[...]', -- Placeholder or separate sync for image
         'post_history_instructions', NEW.post_history_instructions, 'first_message', NEW.first_message,
         'message_example', NEW.message_example, 'creator_notes', NEW.creator_notes, 'system_prompt', NEW.system_prompt,
         'alternate_greetings', NEW.alternate_greetings, 'tags', NEW.tags, 'creator', NEW.creator,
         'character_version', NEW.character_version, 'extensions', NEW.extensions, 'created_at', NEW.created_at,
         'last_modified', NEW.last_modified, 'deleted', NEW.deleted, 'client_id', NEW.client_id, 'version', NEW.version
       )
    );
END;

CREATE TRIGGER IF NOT EXISTS character_cards_sync_delete -- Handles SOFT delete
AFTER UPDATE ON character_cards
WHEN OLD.deleted = 0 AND NEW.deleted = 1
BEGIN
  INSERT INTO sync_log (entity,entity_id,operation,timestamp,client_id,version,payload)
    VALUES (
      'character_cards', CAST(NEW.id AS TEXT), 'delete',
       NEW.last_modified, NEW.client_id, NEW.version,
       json_object('id', NEW.id, 'deleted', NEW.deleted)
    );
END;

-- Optional: Trigger for UNDELETE character_cards
CREATE TRIGGER IF NOT EXISTS character_cards_sync_undelete
AFTER UPDATE ON character_cards
WHEN OLD.deleted = 1 AND NEW.deleted = 0
BEGIN
    INSERT INTO sync_log (entity,entity_id,operation,timestamp,client_id,version,payload)
    VALUES (
      'character_cards', CAST(NEW.id AS TEXT), 'update', -- Treat as update/upsert
       NEW.last_modified, NEW.client_id, NEW.version,
       json_object(
         'id', NEW.id, 'name', NEW.name, 'description', NEW.description, 'personality', NEW.personality,
         'scenario', NEW.scenario, -- 'image', '[...]',
         'post_history_instructions', NEW.post_history_instructions, 'first_message', NEW.first_message,
         'message_example', NEW.message_example, 'creator_notes', NEW.creator_notes, 'system_prompt', NEW.system_prompt,
         'alternate_greetings', NEW.alternate_greetings, 'tags', NEW.tags, 'creator', NEW.creator,
         'character_version', NEW.character_version, 'extensions', NEW.extensions, 'created_at', NEW.created_at,
         'last_modified', NEW.last_modified, 'deleted', NEW.deleted, 'client_id', NEW.client_id, 'version', NEW.version
       )
    );
END;

-- == Triggers for: notes ==
CREATE TRIGGER IF NOT EXISTS notes_sync_create
AFTER INSERT ON notes
BEGIN
  INSERT INTO sync_log (entity,entity_id,operation,timestamp,client_id,version,payload)
    VALUES (
      'notes', CAST(NEW.id AS TEXT), 'create',
       NEW.last_modified, NEW.client_id, NEW.version,
       json_object(
         'id', NEW.id, 'title', NEW.title, 'content', NEW.content,
         'created_at', NEW.created_at, 'last_modified', NEW.last_modified,
         'deleted', NEW.deleted, 'client_id', NEW.client_id, 'version', NEW.version
       )
    );
END;

CREATE TRIGGER IF NOT EXISTS notes_sync_update
AFTER UPDATE ON notes
WHEN OLD.deleted = NEW.deleted AND (
     OLD.title         <> NEW.title OR
     OLD.content       <> NEW.content OR
     OLD.last_modified <> NEW.last_modified OR
     OLD.version       <> NEW.version
)
BEGIN
  INSERT INTO sync_log (entity,entity_id,operation,timestamp,client_id,version,payload)
    VALUES (
      'notes', CAST(NEW.id AS TEXT), 'update',
       NEW.last_modified, NEW.client_id, NEW.version,
       json_object(
         'id', NEW.id, 'title', NEW.title, 'content', NEW.content,
         'created_at', NEW.created_at, 'last_modified', NEW.last_modified,
         'deleted', NEW.deleted, 'client_id', NEW.client_id, 'version', NEW.version
       )
    );
END;

CREATE TRIGGER IF NOT EXISTS notes_sync_delete -- Handles SOFT delete
AFTER UPDATE ON notes
WHEN OLD.deleted = 0 AND NEW.deleted = 1
BEGIN
  INSERT INTO sync_log (entity,entity_id,operation,timestamp,client_id,version,payload)
    VALUES (
      'notes', CAST(NEW.id AS TEXT), 'delete',
       NEW.last_modified, NEW.client_id, NEW.version,
       json_object('id', NEW.id, 'deleted', NEW.deleted)
    );
END;

-- Optional: Trigger for UNDELETE notes
CREATE TRIGGER IF NOT EXISTS notes_sync_undelete
AFTER UPDATE ON notes
WHEN OLD.deleted = 1 AND NEW.deleted = 0
BEGIN
    INSERT INTO sync_log (entity,entity_id,operation,timestamp,client_id,version,payload)
    VALUES (
      'notes', CAST(NEW.id AS TEXT), 'update', -- Treat as update/upsert
       NEW.last_modified, NEW.client_id, NEW.version,
       json_object(
         'id', NEW.id, 'title', NEW.title, 'content', NEW.content,
         'created_at', NEW.created_at, 'last_modified', NEW.last_modified,
         'deleted', NEW.deleted, 'client_id', NEW.client_id, 'version', NEW.version
       )
    );
END;

-- == Triggers for: keywords ==
CREATE TRIGGER IF NOT EXISTS keywords_sync_create
AFTER INSERT ON keywords
BEGIN
  INSERT INTO sync_log (entity,entity_id,operation,timestamp,client_id,version,payload)
    VALUES (
      'keywords', CAST(NEW.id AS TEXT), 'create',
       NEW.last_modified, NEW.client_id, NEW.version,
       json_object(
         'id', NEW.id, 'keyword', NEW.keyword, 'created_at', NEW.created_at,
         'last_modified', NEW.last_modified, 'deleted', NEW.deleted,
         'client_id', NEW.client_id, 'version', NEW.version
       )
    );
END;

CREATE TRIGGER IF NOT EXISTS keywords_sync_update
AFTER UPDATE ON keywords
WHEN OLD.deleted = NEW.deleted AND (
     OLD.keyword       <> NEW.keyword OR
     OLD.last_modified <> NEW.last_modified OR
     OLD.version       <> NEW.version
)
BEGIN
  INSERT INTO sync_log (entity,entity_id,operation,timestamp,client_id,version,payload)
    VALUES (
      'keywords', CAST(NEW.id AS TEXT), 'update',
       NEW.last_modified, NEW.client_id, NEW.version,
       json_object(
         'id', NEW.id, 'keyword', NEW.keyword, 'created_at', NEW.created_at,
         'last_modified', NEW.last_modified, 'deleted', NEW.deleted,
         'client_id', NEW.client_id, 'version', NEW.version
       )
    );
END;

CREATE TRIGGER IF NOT EXISTS keywords_sync_delete -- Handles SOFT delete
AFTER UPDATE ON keywords
WHEN OLD.deleted = 0 AND NEW.deleted = 1
BEGIN
  INSERT INTO sync_log (entity,entity_id,operation,timestamp,client_id,version,payload)
    VALUES (
      'keywords', CAST(NEW.id AS TEXT), 'delete',
       NEW.last_modified, NEW.client_id, NEW.version,
       json_object('id', NEW.id, 'deleted', NEW.deleted)
    );
END;

-- Optional: Trigger for UNDELETE keywords
CREATE TRIGGER IF NOT EXISTS keywords_sync_undelete
AFTER UPDATE ON keywords
WHEN OLD.deleted = 1 AND NEW.deleted = 0
BEGIN
    INSERT INTO sync_log (entity,entity_id,operation,timestamp,client_id,version,payload)
    VALUES (
      'keywords', CAST(NEW.id AS TEXT), 'update', -- Treat as update/upsert
       NEW.last_modified, NEW.client_id, NEW.version,
       json_object(
         'id', NEW.id, 'keyword', NEW.keyword, 'created_at', NEW.created_at,
         'last_modified', NEW.last_modified, 'deleted', NEW.deleted,
         'client_id', NEW.client_id, 'version', NEW.version
       )
    );
END;

-- == Triggers for: keyword_collections ==
CREATE TRIGGER IF NOT EXISTS keyword_collections_sync_create
AFTER INSERT ON keyword_collections
BEGIN
  INSERT INTO sync_log (entity,entity_id,operation,timestamp,client_id,version,payload)
    VALUES (
      'keyword_collections', CAST(NEW.id AS TEXT), 'create',
       NEW.last_modified, NEW.client_id, NEW.version,
       json_object(
         'id', NEW.id, 'name', NEW.name, 'parent_id', NEW.parent_id, 'created_at', NEW.created_at,
         'last_modified', NEW.last_modified, 'deleted', NEW.deleted,
         'client_id', NEW.client_id, 'version', NEW.version
       )
    );
END;

CREATE TRIGGER IF NOT EXISTS keyword_collections_sync_update
AFTER UPDATE ON keyword_collections
WHEN OLD.deleted = NEW.deleted AND (
     OLD.name          <> NEW.name OR
     OLD.parent_id     <> NEW.parent_id OR
     OLD.last_modified <> NEW.last_modified OR
     OLD.version       <> NEW.version
)
BEGIN
  INSERT INTO sync_log (entity,entity_id,operation,timestamp,client_id,version,payload)
    VALUES (
      'keyword_collections', CAST(NEW.id AS TEXT), 'update',
       NEW.last_modified, NEW.client_id, NEW.version,
       json_object(
         'id', NEW.id, 'name', NEW.name, 'parent_id', NEW.parent_id, 'created_at', NEW.created_at,
         'last_modified', NEW.last_modified, 'deleted', NEW.deleted,
         'client_id', NEW.client_id, 'version', NEW.version
       )
    );
END;

CREATE TRIGGER IF NOT EXISTS keyword_collections_sync_delete -- Handles SOFT delete
AFTER UPDATE ON keyword_collections
WHEN OLD.deleted = 0 AND NEW.deleted = 1
BEGIN
  INSERT INTO sync_log (entity,entity_id,operation,timestamp,client_id,version,payload)
    VALUES (
      'keyword_collections', CAST(NEW.id AS TEXT), 'delete',
       NEW.last_modified, NEW.client_id, NEW.version,
       json_object('id', NEW.id, 'deleted', NEW.deleted)
    );
END;

-- Optional: Trigger for UNDELETE keyword_collections
CREATE TRIGGER IF NOT EXISTS keyword_collections_sync_undelete
AFTER UPDATE ON keyword_collections
WHEN OLD.deleted = 1 AND NEW.deleted = 0
BEGIN
    INSERT INTO sync_log (entity,entity_id,operation,timestamp,client_id,version,payload)
    VALUES (
      'keyword_collections', CAST(NEW.id AS TEXT), 'update', -- Treat as update/upsert
       NEW.last_modified, NEW.client_id, NEW.version,
       json_object(
         'id', NEW.id, 'name', NEW.name, 'parent_id', NEW.parent_id, 'created_at', NEW.created_at,
         'last_modified', NEW.last_modified, 'deleted', NEW.deleted,
         'client_id', NEW.client_id, 'version', NEW.version
       )
    );
END;


-- == Triggers for LINK TABLES (Handle Hard Deletes) ==
-- Note: Using placeholder 'unknown_client' and version 1. Adapt as needed.

-- conversation_keywords
CREATE TRIGGER IF NOT EXISTS conversation_keywords_sync_create
AFTER INSERT ON conversation_keywords
BEGIN
  INSERT INTO sync_log (entity,entity_id,operation,timestamp,client_id,version,payload)
    VALUES (
      'conversation_keywords', NEW.conversation_id || '_' || CAST(NEW.keyword_id AS TEXT), 'create', -- Composite pseudo-ID
       NEW.created_at, 'unknown_client', 1, -- Placeholder client/version
       json_object( 'conversation_id', NEW.conversation_id, 'keyword_id', NEW.keyword_id )
    );
END;

CREATE TRIGGER IF NOT EXISTS conversation_keywords_sync_delete
AFTER DELETE ON conversation_keywords
BEGIN
  INSERT INTO sync_log (entity,entity_id,operation,timestamp,client_id,version,payload)
    VALUES (
      'conversation_keywords', OLD.conversation_id || '_' || CAST(OLD.keyword_id AS TEXT), 'delete', -- Composite pseudo-ID
       CURRENT_TIMESTAMP, 'unknown_client', 1, -- Placeholder client/version
       json_object( 'conversation_id', OLD.conversation_id, 'keyword_id', OLD.keyword_id )
    );
END;

-- collection_keywords
CREATE TRIGGER IF NOT EXISTS collection_keywords_sync_create
AFTER INSERT ON collection_keywords
BEGIN
  INSERT INTO sync_log (entity,entity_id,operation,timestamp,client_id,version,payload)
    VALUES (
      'collection_keywords', CAST(NEW.collection_id AS TEXT) || '_' || CAST(NEW.keyword_id AS TEXT), 'create',
       NEW.created_at, 'unknown_client', 1,
       json_object( 'collection_id', NEW.collection_id, 'keyword_id', NEW.keyword_id )
    );
END;

CREATE TRIGGER IF NOT EXISTS collection_keywords_sync_delete
AFTER DELETE ON collection_keywords
BEGIN
  INSERT INTO sync_log (entity,entity_id,operation,timestamp,client_id,version,payload)
    VALUES (
      'collection_keywords', CAST(OLD.collection_id AS TEXT) || '_' || CAST(OLD.keyword_id AS TEXT), 'delete',
       CURRENT_TIMESTAMP, 'unknown_client', 1,
       json_object( 'collection_id', OLD.collection_id, 'keyword_id', OLD.keyword_id )
    );
END;

-- note_keywords
CREATE TRIGGER IF NOT EXISTS note_keywords_sync_create
AFTER INSERT ON note_keywords
BEGIN
  INSERT INTO sync_log (entity,entity_id,operation,timestamp,client_id,version,payload)
    VALUES (
      'note_keywords', CAST(NEW.note_id AS TEXT) || '_' || CAST(NEW.keyword_id AS TEXT), 'create',
       NEW.created_at, 'unknown_client', 1,
       json_object( 'note_id', NEW.note_id, 'keyword_id', NEW.keyword_id )
    );
END;

CREATE TRIGGER IF NOT EXISTS note_keywords_sync_delete
AFTER DELETE ON note_keywords
BEGIN
  INSERT INTO sync_log (entity,entity_id,operation,timestamp,client_id,version,payload)
    VALUES (
      'note_keywords', CAST(OLD.note_id AS TEXT) || '_' || CAST(OLD.keyword_id AS TEXT), 'delete',
       CURRENT_TIMESTAMP, 'unknown_client', 1,
       json_object( 'note_id', OLD.note_id, 'keyword_id', OLD.keyword_id )
    );
END;
"""





#
# End of Characters_and_RAG_DB.py
#######################################################################################################################
