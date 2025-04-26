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
character_and_rag_db_schema = """
PRAGMA foreign_keys = ON;

-- ───────────────────────────────────────────────────────────────────────────
-- 1. Conversations (branches) with tombstones & sync metadata
-- ───────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS conversations (
    id                       TEXT      PRIMARY KEY,
    root_id                  TEXT      NOT NULL,
    forked_from_message_id   TEXT      REFERENCES messages(id),
    parent_conversation_id   TEXT      REFERENCES conversations(id),
    character_id             INTEGER   REFERENCES character_cards(id),
    title                    TEXT,
    rating                   INTEGER   CHECK(rating BETWEEN 1 AND 5),
    created_at               DATETIME  NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_modified            DATETIME  NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted                  BOOLEAN   NOT NULL DEFAULT 0,
    client_id                TEXT      NOT NULL,
    version                  INTEGER   NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_conversations_root  ON conversations(root_id);
CREATE INDEX IF NOT EXISTS idx_conversations_parent ON conversations(parent_conversation_id);

-- FTS5 for conversation titles
CREATE VIRTUAL TABLE IF NOT EXISTS conversations_fts
USING fts5(
    title,
    content='conversations',
    content_rowid='rowid'
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
-- 2. Messages with swipe/fork links, rankings, tombstones & sync metadata
-- ───────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS messages (
    id                   TEXT      PRIMARY KEY,
    conversation_id      TEXT      NOT NULL REFERENCES conversations(id),
    parent_message_id    TEXT      REFERENCES messages(id),
    sender               TEXT      NOT NULL,
    content              TEXT      NOT NULL,
    timestamp            DATETIME  NOT NULL DEFAULT CURRENT_TIMESTAMP,
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

-- FTS5 for message content
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts
USING fts5(
    content,
    content='messages',
    content_rowid='rowid'
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
-- 3. Character profiles (unchanged schema + FTS5)
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
    created_at                DATETIME  DEFAULT CURRENT_TIMESTAMP
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
-- 4. Keywords, Collections & Notes (from RAG‑QA) + FTS5
-- ───────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS keywords (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    keyword  TEXT    NOT NULL UNIQUE
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

CREATE TABLE IF NOT EXISTS conversation_keywords (
    conversation_id TEXT NOT NULL REFERENCES conversations(id),
    keyword_id      INTEGER NOT NULL REFERENCES keywords(id),
    PRIMARY KEY(conversation_id, keyword_id)
);

CREATE TABLE IF NOT EXISTS keyword_collections (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    name      TEXT    NOT NULL UNIQUE,
    parent_id INTEGER REFERENCES keyword_collections(id)
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

CREATE TABLE IF NOT EXISTS collection_keywords (
    collection_id INTEGER NOT NULL REFERENCES keyword_collections(id),
    keyword_id    INTEGER NOT NULL REFERENCES keywords(id),
    PRIMARY KEY(collection_id, keyword_id)
);

CREATE TABLE IF NOT EXISTS notes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT    NOT NULL REFERENCES conversations(id),
    title           TEXT    NOT NULL,
    content         TEXT    NOT NULL,
    timestamp       DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);
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

CREATE TABLE IF NOT EXISTS note_keywords (
    note_id    INTEGER NOT NULL REFERENCES notes(id),
    keyword_id INTEGER NOT NULL REFERENCES keywords(id),
    PRIMARY KEY(note_id, keyword_id)
);

-- ───────────────────────────────────────────────────────────────────────────
-- 5. Sync Log for bi‑directional, offline‑first synchronization
-- ───────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sync_log (
    change_id    INTEGER  PRIMARY KEY AUTOINCREMENT,
    entity       TEXT     NOT NULL,   -- 'conversations' or 'messages'
    entity_id    TEXT     NOT NULL,   -- UUID of the record
    operation    TEXT     NOT NULL   CHECK(operation IN ('create','update','delete')),
    timestamp    DATETIME NOT NULL,   -- equals record.last_modified
    client_id    TEXT     NOT NULL,   -- source device UUID
    version      INTEGER  NOT NULL,
    payload      TEXT     NOT NULL    -- JSON blob of new record state
);
CREATE INDEX IF NOT EXISTS idx_sync_log_ts ON sync_log(timestamp);

-- Triggers to populate sync_log on INSERT / UPDATE / logical DELETE for messages
CREATE TRIGGER IF NOT EXISTS messages_sync_create
AFTER INSERT ON messages
BEGIN
  INSERT INTO sync_log (entity,entity_id,operation,timestamp,client_id,version,payload)
    VALUES (
      'messages', NEW.id, 'create',
       NEW.last_modified, NEW.client_id, NEW.version,
       json_object(
         'id', NEW.id,
         'conversation_id', NEW.conversation_id,
         'parent_message_id', NEW.parent_message_id,
         'sender', NEW.sender,
         'content', NEW.content,
         'timestamp', NEW.timestamp,
         'ranking', NEW.ranking,
         'last_modified', NEW.last_modified,
         'deleted', NEW.deleted,
         'client_id', NEW.client_id,
         'version', NEW.version
       )
    );
END;

CREATE TRIGGER IF NOT EXISTS messages_sync_update
AFTER UPDATE ON messages
WHEN OLD.deleted = NEW.deleted AND (
     OLD.content      <> NEW.content OR
     OLD.ranking      <> NEW.ranking OR
     OLD.deleted      <> NEW.deleted OR
     OLD.version      <> NEW.version OR
     OLD.last_modified<> NEW.last_modified
)
BEGIN
  INSERT INTO sync_log (entity,entity_id,operation,timestamp,client_id,version,payload)
    VALUES (
      'messages', NEW.id, 'update',
       NEW.last_modified, NEW.client_id, NEW.version,
       json_object(
         'id', NEW.id,
         'conversation_id', NEW.conversation_id,
         'parent_message_id', NEW.parent_message_id,
         'sender', NEW.sender,
         'content', NEW.content,
         'timestamp', NEW.timestamp,
         'ranking', NEW.ranking,
         'last_modified', NEW.last_modified,
         'deleted', NEW.deleted,
         'client_id', NEW.client_id,
         'version', NEW.version
       )
    );
END;

CREATE TRIGGER IF NOT EXISTS messages_sync_delete
AFTER UPDATE ON messages
WHEN OLD.deleted = 0 AND NEW.deleted = 1
BEGIN
  INSERT INTO sync_log (entity,entity_id,operation,timestamp,client_id,version,payload)
    VALUES (
      'messages', NEW.id, 'delete',
       NEW.last_modified, NEW.client_id, NEW.version,
       json_object('id', NEW.id)
    );
END;

-- Triggers to populate sync_log on INSERT / UPDATE / logical DELETE for conversations
CREATE TRIGGER IF NOT EXISTS conversations_sync_create
AFTER INSERT ON conversations
BEGIN
  INSERT INTO sync_log (entity,entity_id,operation,timestamp,client_id,version,payload)
    VALUES (
      'conversations', NEW.id, 'create',
       NEW.last_modified, NEW.client_id, NEW.version,
       json_object(
         'id', NEW.id,
         'root_id', NEW.root_id,
         'forked_from_message_id', NEW.forked_from_message_id,
         'parent_conversation_id', NEW.parent_conversation_id,
         'character_id', NEW.character_id,
         'title', NEW.title,
         'rating', NEW.rating,
         'last_modified', NEW.last_modified,
         'deleted', NEW.deleted,
         'client_id', NEW.client_id,
         'version', NEW.version
       )
    );
END;

CREATE TRIGGER IF NOT EXISTS conversations_sync_update
AFTER UPDATE ON conversations
WHEN OLD.deleted = NEW.deleted AND (
     OLD.title            <> NEW.title OR
     OLD.rating           <> NEW.rating OR
     OLD.deleted          <> NEW.deleted OR
     OLD.version          <> NEW.version OR
     OLD.last_modified    <> NEW.last_modified
)
BEGIN
  INSERT INTO sync_log (entity,entity_id,operation,timestamp,client_id,version,payload)
    VALUES (
      'conversations', NEW.id, 'update',
       NEW.last_modified, NEW.client_id, NEW.version,
       json_object(
         'id', NEW.id,
         'root_id', NEW.root_id,
         'forked_from_message_id', NEW.forked_from_message_id,
         'parent_conversation_id', NEW.parent_conversation_id,
         'character_id', NEW.character_id,
         'title', NEW.title,
         'rating', NEW.rating,
         'last_modified', NEW.last_modified,
         'deleted', NEW.deleted,
         'client_id', NEW.client_id,
         'version', NEW.version
       )
    );
END;

CREATE TRIGGER IF NOT EXISTS conversations_sync_delete
AFTER UPDATE ON conversations
WHEN OLD.deleted = 0 AND NEW.deleted = 1
BEGIN
  INSERT INTO sync_log (entity,entity_id,operation,timestamp,client_id,version,payload)
    VALUES (
      'conversations', NEW.id, 'delete',
       NEW.last_modified, NEW.client_id, NEW.version,
       json_object('id', NEW.id)
    );
END;
"""





#
# End of Characters_and_RAG_DB.py
#######################################################################################################################
