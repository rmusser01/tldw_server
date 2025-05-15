# ChaChaNotes_DB.py
# Description: DB Library for Character Cards, Chats, and Notes.
#
# Imports
import sqlite3
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
import threading
import logging
from typing import List, Dict, Optional, Tuple, Any, Union, Set
#
# Third-Party Libraries
#
# Local Imports
#
########################################################################################################################
#
# Functions:


# --- Logging Setup ---
# It's good practice to have a logger. If not configured elsewhere,
# a basic configuration can be uncommented or set up in the application.
logger = logging.getLogger(__name__)


# Example basic config:
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(name)s - %(threadName)s - %(message)s')


# --- Custom Exceptions ---
class CharactersRAGDBError(Exception):
    """Base exception for CharactersRAGDB related errors."""
    pass


class SchemaError(CharactersRAGDBError):
    """Exception for schema version mismatches or migration failures."""
    pass


class InputError(ValueError):
    """Custom exception for input validation errors."""
    pass


class ConflictError(CharactersRAGDBError):
    """Indicates a conflict due to concurrent modification (version mismatch or unique constraint)."""

    def __init__(self, message="Conflict detected.", entity: Optional[str] = None, entity_id: Any = None):
        super().__init__(message)
        self.entity = entity
        self.entity_id = entity_id

    def __str__(self):
        base = super().__str__()
        details = []
        if self.entity:
            details.append(f"Entity: {self.entity}")
        if self.entity_id:
            details.append(f"ID: {self.entity_id}")
        return f"{base} ({', '.join(details)})" if details else base


# --- Database Class ---
class CharactersRAGDB:
    """
    Manages SQLite connection and operations for the Characters and RAG database.
    Handles sync metadata updates. Requires client_id on initialization.
    Includes schema versioning.
    Relies on SQL triggers for FTS updates and sync_log entries for most main tables.
    Link table sync logs are handled by Python methods.
    """
    _CURRENT_SCHEMA_VERSION = 3 # Incremented schema version
    _SCHEMA_NAME = "rag_char_chat_schema"  # Used for the db_schema_version table

    _FULL_SCHEMA_SQL_V3 = """
/*───────────────────────────────────────────────────────────────
  RAG Character-Chat Schema  –  Version 3   (2025-05-14)
───────────────────────────────────────────────────────────────*/
PRAGMA foreign_keys = ON;

/*----------------------------------------------------------------
  0. Schema-version registry
----------------------------------------------------------------*/
CREATE TABLE IF NOT EXISTS db_schema_version(
  schema_name TEXT PRIMARY KEY NOT NULL,
  version     INTEGER NOT NULL
);
INSERT OR IGNORE INTO db_schema_version(schema_name,version)
VALUES('rag_char_chat_schema',0);

/*----------------------------------------------------------------
  1. Character profiles  (FTS5 external-content)
----------------------------------------------------------------*/
CREATE TABLE IF NOT EXISTS character_cards(
  id            INTEGER  PRIMARY KEY AUTOINCREMENT,
  name          TEXT     UNIQUE NOT NULL,
  description   TEXT,
  personality   TEXT,
  scenario      TEXT,
  system_prompt TEXT,
  image                     BLOB,
  post_history_instructions TEXT,
  first_message             TEXT,
  message_example           TEXT,
  creator_notes             TEXT,
  alternate_greetings       TEXT,
  tags                      TEXT,
  creator                   TEXT,
  character_version         TEXT,
  extensions                TEXT,
  created_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted       BOOLEAN  NOT NULL DEFAULT 0,
  client_id     TEXT     NOT NULL DEFAULT 'unknown',
  version       INTEGER  NOT NULL DEFAULT 1
);

CREATE VIRTUAL TABLE IF NOT EXISTS character_cards_fts
USING fts5(
  name, description, personality, scenario, system_prompt,
  content='character_cards',
  content_rowid='id'
);

DROP TRIGGER IF EXISTS character_cards_ai;
DROP TRIGGER IF EXISTS character_cards_au;
DROP TRIGGER IF EXISTS character_cards_ad;

CREATE TRIGGER character_cards_ai
AFTER INSERT ON character_cards BEGIN
  INSERT INTO character_cards_fts(rowid,name,description,personality,scenario,system_prompt)
  SELECT new.id,new.name,new.description,new.personality,new.scenario,new.system_prompt
  WHERE new.deleted = 0;
END;

CREATE TRIGGER character_cards_au
AFTER UPDATE ON character_cards BEGIN
  INSERT INTO character_cards_fts(character_cards_fts,rowid,
                                  name,description,personality,scenario,system_prompt)
  VALUES('delete',old.id,old.name,old.description,old.personality,old.scenario,old.system_prompt);

  INSERT INTO character_cards_fts(rowid,name,description,personality,scenario,system_prompt)
  SELECT new.id,new.name,new.description,new.personality,new.scenario,new.system_prompt
  WHERE new.deleted = 0;
END;

CREATE TRIGGER character_cards_ad
AFTER DELETE ON character_cards BEGIN
  INSERT INTO character_cards_fts(character_cards_fts,rowid,
                                  name,description,personality,scenario,system_prompt)
  VALUES('delete',old.id,old.name,old.description,old.personality,old.scenario,old.system_prompt);
END;

/*----------------------------------------------------------------
  2. Conversations
----------------------------------------------------------------*/
CREATE TABLE IF NOT EXISTS conversations(
  id                     TEXT PRIMARY KEY,            /* UUID */
  root_id                TEXT NOT NULL,
  forked_from_message_id TEXT REFERENCES messages(id) ON DELETE SET NULL,
  parent_conversation_id TEXT REFERENCES conversations(id) ON DELETE SET NULL,
  character_id           INTEGER REFERENCES character_cards(id)
                          ON DELETE CASCADE ON UPDATE CASCADE,
  title        TEXT,
  rating       INTEGER CHECK(rating BETWEEN 1 AND 5),
  created_at   DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted      BOOLEAN  NOT NULL DEFAULT 0,
  client_id    TEXT     NOT NULL,
  version      INTEGER  NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_conversations_root   ON conversations(root_id);
CREATE INDEX IF NOT EXISTS idx_conversations_parent ON conversations(parent_conversation_id);
CREATE INDEX IF NOT EXISTS idx_conv_char           ON conversations(character_id);

CREATE VIRTUAL TABLE IF NOT EXISTS conversations_fts
USING fts5(
  title,
  content='conversations',
  content_rowid='rowid'
);

DROP TRIGGER IF EXISTS conversations_ai;
DROP TRIGGER IF EXISTS conversations_au;
DROP TRIGGER IF EXISTS conversations_ad;

CREATE TRIGGER conversations_ai
AFTER INSERT ON conversations BEGIN
  INSERT INTO conversations_fts(rowid,title)
  SELECT new.rowid,new.title
  WHERE new.deleted = 0 AND new.title IS NOT NULL;
END;

CREATE TRIGGER conversations_au
AFTER UPDATE ON conversations BEGIN
  INSERT INTO conversations_fts(conversations_fts,rowid,title)
  VALUES('delete',old.rowid,old.title);

  INSERT INTO conversations_fts(rowid,title)
  SELECT new.rowid,new.title
  WHERE new.deleted = 0 AND new.title IS NOT NULL;
END;

CREATE TRIGGER conversations_ad
AFTER DELETE ON conversations BEGIN
  INSERT INTO conversations_fts(conversations_fts,rowid,title)
  VALUES('delete',old.rowid,old.title);
END;

/*----------------------------------------------------------------
  3. Messages
----------------------------------------------------------------*/
CREATE TABLE IF NOT EXISTS messages(
  id                TEXT PRIMARY KEY,                 /* UUID */
  conversation_id   TEXT  NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  parent_message_id TEXT  REFERENCES messages(id)     ON DELETE SET NULL,
  sender            TEXT  NOT NULL,
  content           TEXT  NOT NULL,
  timestamp         DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  ranking           INTEGER,
  last_modified     DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted           BOOLEAN NOT NULL DEFAULT 0,
  client_id         TEXT    NOT NULL,
  version           INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_msgs_conversation ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_msgs_parent       ON messages(parent_message_id);
CREATE INDEX IF NOT EXISTS idx_msgs_timestamp    ON messages(timestamp);
CREATE INDEX IF NOT EXISTS idx_msgs_ranking      ON messages(ranking);
CREATE INDEX IF NOT EXISTS idx_msgs_conv_ts      ON messages(conversation_id,timestamp);

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts
USING fts5(
  content,
  content='messages',
  content_rowid='rowid'
);

DROP TRIGGER IF EXISTS messages_ai;
DROP TRIGGER IF EXISTS messages_au;
DROP TRIGGER IF EXISTS messages_ad;

CREATE TRIGGER messages_ai
AFTER INSERT ON messages BEGIN
  INSERT INTO messages_fts(rowid,content)
  SELECT new.rowid,new.content
  WHERE new.deleted = 0;
END;

CREATE TRIGGER messages_au
AFTER UPDATE ON messages BEGIN
  INSERT INTO messages_fts(messages_fts,rowid,content)
  VALUES('delete',old.rowid,old.content);

  INSERT INTO messages_fts(rowid,content)
  SELECT new.rowid,new.content
  WHERE new.deleted = 0;
END;

CREATE TRIGGER messages_ad
AFTER DELETE ON messages BEGIN
  INSERT INTO messages_fts(messages_fts,rowid,content)
  VALUES('delete',old.rowid,old.content);
END;

/*----------------------------------------------------------------
  4. Keywords
----------------------------------------------------------------*/
CREATE TABLE IF NOT EXISTS keywords(
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  keyword       TEXT    UNIQUE NOT NULL COLLATE NOCASE,
  created_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted       BOOLEAN  NOT NULL DEFAULT 0,
  client_id     TEXT     NOT NULL DEFAULT 'unknown',
  version       INTEGER  NOT NULL DEFAULT 1
);

CREATE VIRTUAL TABLE IF NOT EXISTS keywords_fts
USING fts5(
  keyword,
  content='keywords',
  content_rowid='id'
);

DROP TRIGGER IF EXISTS keywords_ai;
DROP TRIGGER IF EXISTS keywords_au;
DROP TRIGGER IF EXISTS keywords_ad;

CREATE TRIGGER keywords_ai
AFTER INSERT ON keywords BEGIN
  INSERT INTO keywords_fts(rowid,keyword)
  SELECT new.id,new.keyword
  WHERE new.deleted = 0;
END;

CREATE TRIGGER keywords_au
AFTER UPDATE ON keywords BEGIN
  INSERT INTO keywords_fts(keywords_fts,rowid,keyword)
  VALUES('delete',old.id,old.keyword);

  INSERT INTO keywords_fts(rowid,keyword)
  SELECT new.id,new.keyword
  WHERE new.deleted = 0;
END;

CREATE TRIGGER keywords_ad
AFTER DELETE ON keywords BEGIN
  INSERT INTO keywords_fts(keywords_fts,rowid,keyword)
  VALUES('delete',old.id,old.keyword);
END;

/*----------------------------------------------------------------
  5. Keyword collections
----------------------------------------------------------------*/
CREATE TABLE IF NOT EXISTS keyword_collections(
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  name          TEXT    UNIQUE NOT NULL COLLATE NOCASE,
  parent_id     INTEGER REFERENCES keyword_collections(id)
                         ON DELETE SET NULL ON UPDATE CASCADE,
  created_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted       BOOLEAN  NOT NULL DEFAULT 0,
  client_id     TEXT     NOT NULL DEFAULT 'unknown',
  version       INTEGER  NOT NULL DEFAULT 1
);

CREATE VIRTUAL TABLE IF NOT EXISTS keyword_collections_fts
USING fts5(
  name,
  content='keyword_collections',
  content_rowid='id'
);

DROP TRIGGER IF EXISTS keyword_collections_ai;
DROP TRIGGER IF EXISTS keyword_collections_au;
DROP TRIGGER IF EXISTS keyword_collections_ad;

CREATE TRIGGER keyword_collections_ai
AFTER INSERT ON keyword_collections BEGIN
  INSERT INTO keyword_collections_fts(rowid,name)
  SELECT new.id,new.name
  WHERE new.deleted = 0;
END;

CREATE TRIGGER keyword_collections_au
AFTER UPDATE ON keyword_collections BEGIN
  INSERT INTO keyword_collections_fts(keyword_collections_fts,rowid,name)
  VALUES('delete',old.id,old.name);

  INSERT INTO keyword_collections_fts(rowid,name)
  SELECT new.id,new.name
  WHERE new.deleted = 0;
END;

CREATE TRIGGER keyword_collections_ad
AFTER DELETE ON keyword_collections BEGIN
  INSERT INTO keyword_collections_fts(keyword_collections_fts,rowid,name)
  VALUES('delete',old.id,old.name);
END;

/*----------------------------------------------------------------
  6. Notes
----------------------------------------------------------------*/
CREATE TABLE IF NOT EXISTS notes(
  id            TEXT PRIMARY KEY,                     /* UUID */
  title         TEXT NOT NULL,
  content       TEXT NOT NULL,
  created_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted       BOOLEAN  NOT NULL DEFAULT 0,
  client_id     TEXT     NOT NULL DEFAULT 'unknown',
  version       INTEGER  NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_notes_last_modified ON notes(last_modified);

CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts
USING fts5(
  title,content,
  content='notes',
  content_rowid='rowid'
);

DROP TRIGGER IF EXISTS notes_ai;
DROP TRIGGER IF EXISTS notes_au;
DROP TRIGGER IF EXISTS notes_ad;

CREATE TRIGGER notes_ai
AFTER INSERT ON notes BEGIN
  INSERT INTO notes_fts(rowid,title,content)
  SELECT new.rowid,new.title,new.content
  WHERE new.deleted = 0;
END;

CREATE TRIGGER notes_au
AFTER UPDATE ON notes BEGIN
  INSERT INTO notes_fts(notes_fts,rowid,title,content)
  VALUES('delete',old.rowid,old.title,old.content);

  INSERT INTO notes_fts(rowid,title,content)
  SELECT new.rowid,new.title,new.content
  WHERE new.deleted = 0;
END;

CREATE TRIGGER notes_ad
AFTER DELETE ON notes BEGIN
  INSERT INTO notes_fts(notes_fts,rowid,title,content)
  VALUES('delete',old.rowid,old.title,old.content);
END;

/*----------------------------------------------------------------
  7. Linking tables (no FTS)
----------------------------------------------------------------*/
CREATE TABLE IF NOT EXISTS conversation_keywords(
  conversation_id TEXT    NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  keyword_id      INTEGER NOT NULL REFERENCES keywords(id) ON DELETE CASCADE ON UPDATE CASCADE,
  created_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY(conversation_id,keyword_id)
);
CREATE INDEX IF NOT EXISTS idx_convkw_kw ON conversation_keywords(keyword_id);

CREATE TABLE IF NOT EXISTS collection_keywords(
  collection_id INTEGER NOT NULL REFERENCES keyword_collections(id) ON DELETE CASCADE ON UPDATE CASCADE,
  keyword_id    INTEGER NOT NULL REFERENCES keywords(id)            ON DELETE CASCADE ON UPDATE CASCADE,
  created_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY(collection_id,keyword_id)
);
CREATE INDEX IF NOT EXISTS idx_collkw_kw ON collection_keywords(keyword_id);

CREATE TABLE IF NOT EXISTS note_keywords(
  note_id    TEXT    NOT NULL REFERENCES notes(id)                 ON DELETE CASCADE ON UPDATE CASCADE,
  keyword_id INTEGER NOT NULL REFERENCES keywords(id)              ON DELETE CASCADE ON UPDATE CASCADE,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY(note_id,keyword_id)
);
CREATE INDEX IF NOT EXISTS idx_notekw_kw ON note_keywords(keyword_id);

/*----------------------------------------------------------------
  8. Sync log (plus triggers)
----------------------------------------------------------------*/
CREATE TABLE IF NOT EXISTS sync_log(
  change_id   INTEGER  PRIMARY KEY AUTOINCREMENT,
  entity      TEXT     NOT NULL,
  entity_id   TEXT     NOT NULL,
  operation   TEXT     NOT NULL CHECK(operation IN('create','update','delete')),
  timestamp   DATETIME NOT NULL,
  client_id   TEXT     NOT NULL,
  version     INTEGER  NOT NULL,
  payload     TEXT     NOT NULL          /* JSON blob */
);
CREATE INDEX IF NOT EXISTS idx_sync_log_ts     ON sync_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_sync_log_entity ON sync_log(entity,entity_id);

/*—— drop any pre-existing sync triggers ———————————*/
DROP TRIGGER IF EXISTS messages_sync_create;
DROP TRIGGER IF EXISTS messages_sync_update;
DROP TRIGGER IF EXISTS messages_sync_delete;
DROP TRIGGER IF EXISTS messages_sync_undelete;

DROP TRIGGER IF EXISTS conversations_sync_create;
DROP TRIGGER IF EXISTS conversations_sync_update;
DROP TRIGGER IF EXISTS conversations_sync_delete;
DROP TRIGGER IF EXISTS conversations_sync_undelete;

DROP TRIGGER IF EXISTS character_cards_sync_create;
DROP TRIGGER IF EXISTS character_cards_sync_update;
DROP TRIGGER IF EXISTS character_cards_sync_delete;
DROP TRIGGER IF EXISTS character_cards_sync_undelete;

DROP TRIGGER IF EXISTS notes_sync_create;
DROP TRIGGER IF EXISTS notes_sync_update;
DROP TRIGGER IF EXISTS notes_sync_delete;
DROP TRIGGER IF EXISTS notes_sync_undelete;

DROP TRIGGER IF EXISTS keywords_sync_create;
DROP TRIGGER IF EXISTS keywords_sync_update;
DROP TRIGGER IF EXISTS keywords_sync_delete;
DROP TRIGGER IF EXISTS keywords_sync_undelete;

DROP TRIGGER IF EXISTS keyword_collections_sync_create;
DROP TRIGGER IF EXISTS keyword_collections_sync_update;
DROP TRIGGER IF EXISTS keyword_collections_sync_delete;
DROP TRIGGER IF EXISTS keyword_collections_sync_undelete;

/*—— sync triggers: messages ———————————————*/
CREATE TRIGGER messages_sync_create
AFTER INSERT ON messages BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('messages',NEW.id,'create',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'conversation_id',NEW.conversation_id,'parent_message_id',NEW.parent_message_id,
                     'sender',NEW.sender,'content',NEW.content,'timestamp',NEW.timestamp,'ranking',NEW.ranking,
                     'last_modified',NEW.last_modified,'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER messages_sync_update
AFTER UPDATE ON messages
WHEN OLD.deleted = NEW.deleted AND (
     OLD.content IS NOT NEW.content OR
     OLD.ranking IS NOT NEW.ranking OR
     OLD.parent_message_id IS NOT NEW.parent_message_id OR
     OLD.last_modified IS NOT NEW.last_modified OR
     OLD.version IS NOT NEW.version)
BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('messages',NEW.id,'update',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'conversation_id',NEW.conversation_id,'parent_message_id',NEW.parent_message_id,
                     'sender',NEW.sender,'content',NEW.content,'timestamp',NEW.timestamp,'ranking',NEW.ranking,
                     'last_modified',NEW.last_modified,'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER messages_sync_delete
AFTER UPDATE ON messages
WHEN OLD.deleted = 0 AND NEW.deleted = 1
BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('messages',NEW.id,'delete',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'deleted',NEW.deleted,'last_modified',NEW.last_modified,
                     'version',NEW.version,'client_id',NEW.client_id));
END;

CREATE TRIGGER messages_sync_undelete
AFTER UPDATE ON messages
WHEN OLD.deleted = 1 AND NEW.deleted = 0
BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('messages',NEW.id,'update',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'conversation_id',NEW.conversation_id,'parent_message_id',NEW.parent_message_id,
                     'sender',NEW.sender,'content',NEW.content,'timestamp',NEW.timestamp,'ranking',NEW.ranking,
                     'last_modified',NEW.last_modified,'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

/*—— sync triggers: conversations ———————————*/
CREATE TRIGGER conversations_sync_create
AFTER INSERT ON conversations BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('conversations',NEW.id,'create',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'root_id',NEW.root_id,'forked_from_message_id',NEW.forked_from_message_id,
                     'parent_conversation_id',NEW.parent_conversation_id,'character_id',NEW.character_id,'title',NEW.title,
                     'rating',NEW.rating,'created_at',NEW.created_at,'last_modified',NEW.last_modified,'deleted',NEW.deleted,
                     'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER conversations_sync_update
AFTER UPDATE ON conversations
WHEN OLD.deleted = NEW.deleted AND (
     OLD.title IS NOT NEW.title OR
     OLD.rating IS NOT NEW.rating OR
     OLD.forked_from_message_id IS NOT NEW.forked_from_message_id OR
     OLD.parent_conversation_id IS NOT NEW.parent_conversation_id OR
     OLD.character_id IS NOT NEW.character_id OR
     OLD.last_modified IS NOT NEW.last_modified OR
     OLD.version IS NOT NEW.version)
BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('conversations',NEW.id,'update',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'root_id',NEW.root_id,'forked_from_message_id',NEW.forked_from_message_id,
                     'parent_conversation_id',NEW.parent_conversation_id,'character_id',NEW.character_id,'title',NEW.title,
                     'rating',NEW.rating,'created_at',NEW.created_at,'last_modified',NEW.last_modified,'deleted',NEW.deleted,
                     'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER conversations_sync_delete
AFTER UPDATE ON conversations
WHEN OLD.deleted = 0 AND NEW.deleted = 1
BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('conversations',NEW.id,'delete',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'deleted',NEW.deleted,'last_modified',NEW.last_modified,
                     'version',NEW.version,'client_id',NEW.client_id));
END;

CREATE TRIGGER conversations_sync_undelete
AFTER UPDATE ON conversations
WHEN OLD.deleted = 1 AND NEW.deleted = 0
BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('conversations',NEW.id,'update',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'root_id',NEW.root_id,'forked_from_message_id',NEW.forked_from_message_id,
                     'parent_conversation_id',NEW.parent_conversation_id,'character_id',NEW.character_id,'title',NEW.title,
                     'rating',NEW.rating,'created_at',NEW.created_at,'last_modified',NEW.last_modified,'deleted',NEW.deleted,
                     'client_id',NEW.client_id,'version',NEW.version));
END;

/*—— sync triggers: character_cards —————————*/
CREATE TRIGGER character_cards_sync_create
AFTER INSERT ON character_cards BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('character_cards',CAST(NEW.id AS TEXT),'create',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'name',NEW.name,'description',NEW.description,'personality',NEW.personality,
                     'scenario',NEW.scenario,'post_history_instructions',NEW.post_history_instructions,
                     'first_message',NEW.first_message,'message_example',NEW.message_example,'creator_notes',NEW.creator_notes,
                     'system_prompt',NEW.system_prompt,'alternate_greetings',NEW.alternate_greetings,'tags',NEW.tags,'creator',NEW.creator,
                     'character_version',NEW.character_version,'extensions',NEW.extensions,'created_at',NEW.created_at,
                     'last_modified',NEW.last_modified,'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER character_cards_sync_update
AFTER UPDATE ON character_cards
WHEN OLD.deleted = NEW.deleted AND (
     OLD.name IS NOT NEW.name OR
     OLD.description IS NOT NEW.description OR
     OLD.personality IS NOT NEW.personality OR
     OLD.scenario IS NOT NEW.scenario OR
     OLD.image IS NOT NEW.image OR
     OLD.post_history_instructions IS NOT NEW.post_history_instructions OR
     OLD.first_message IS NOT NEW.first_message OR
     OLD.message_example IS NOT NEW.message_example OR
     OLD.creator_notes IS NOT NEW.creator_notes OR
     OLD.system_prompt IS NOT NEW.system_prompt OR
     OLD.alternate_greetings IS NOT NEW.alternate_greetings OR
     OLD.tags IS NOT NEW.tags OR
     OLD.creator IS NOT NEW.creator OR
     OLD.character_version IS NOT NEW.character_version OR
     OLD.extensions IS NOT NEW.extensions OR
     OLD.last_modified IS NOT NEW.last_modified OR
     OLD.version IS NOT NEW.version)
BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('character_cards',CAST(NEW.id AS TEXT),'update',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'name',NEW.name,'description',NEW.description,'personality',NEW.personality,
                     'scenario',NEW.scenario,'post_history_instructions',NEW.post_history_instructions,
                     'first_message',NEW.first_message,'message_example',NEW.message_example,'creator_notes',NEW.creator_notes,
                     'system_prompt',NEW.system_prompt,'alternate_greetings',NEW.alternate_greetings,'tags',NEW.tags,'creator',NEW.creator,
                     'character_version',NEW.character_version,'extensions',NEW.extensions,'created_at',NEW.created_at,
                     'last_modified',NEW.last_modified,'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER character_cards_sync_delete
AFTER UPDATE ON character_cards
WHEN OLD.deleted = 0 AND NEW.deleted = 1
BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('character_cards',CAST(NEW.id AS TEXT),'delete',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'deleted',NEW.deleted,'last_modified',NEW.last_modified,
                     'version',NEW.version,'client_id',NEW.client_id));
END;

CREATE TRIGGER character_cards_sync_undelete
AFTER UPDATE ON character_cards
WHEN OLD.deleted = 1 AND NEW.deleted = 0
BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('character_cards',CAST(NEW.id AS TEXT),'update',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'name',NEW.name,'description',NEW.description,'personality',NEW.personality,
                     'scenario',NEW.scenario,'post_history_instructions',NEW.post_history_instructions,
                     'first_message',NEW.first_message,'message_example',NEW.message_example,'creator_notes',NEW.creator_notes,
                     'system_prompt',NEW.system_prompt,'alternate_greetings',NEW.alternate_greetings,'tags',NEW.tags,'creator',NEW.creator,
                     'character_version',NEW.character_version,'extensions',NEW.extensions,'created_at',NEW.created_at,
                     'last_modified',NEW.last_modified,'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

/*—— sync triggers: notes ———————————————*/
CREATE TRIGGER notes_sync_create
AFTER INSERT ON notes BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('notes',NEW.id,'create',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'title',NEW.title,'content',NEW.content,'created_at',NEW.created_at,
                     'last_modified',NEW.last_modified,'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER notes_sync_update
AFTER UPDATE ON notes
WHEN OLD.deleted = NEW.deleted AND (
     OLD.title IS NOT NEW.title OR
     OLD.content IS NOT NEW.content OR
     OLD.last_modified IS NOT NEW.last_modified OR
     OLD.version IS NOT NEW.version)
BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('notes',NEW.id,'update',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'title',NEW.title,'content',NEW.content,'created_at',NEW.created_at,
                     'last_modified',NEW.last_modified,'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER notes_sync_delete
AFTER UPDATE ON notes
WHEN OLD.deleted = 0 AND NEW.deleted = 1
BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('notes',NEW.id,'delete',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'deleted',NEW.deleted,'last_modified',NEW.last_modified,
                     'version',NEW.version,'client_id',NEW.client_id));
END;

CREATE TRIGGER notes_sync_undelete
AFTER UPDATE ON notes
WHEN OLD.deleted = 1 AND NEW.deleted = 0
BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('notes',NEW.id,'update',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'title',NEW.title,'content',NEW.content,'created_at',NEW.created_at,
                     'last_modified',NEW.last_modified,'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

/*—— sync triggers: keyword_collections ————*/
CREATE TRIGGER keyword_collections_sync_create
AFTER INSERT ON keyword_collections BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('keyword_collections',CAST(NEW.id AS TEXT),'create',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'name',NEW.name,'parent_id',NEW.parent_id,'created_at',NEW.created_at,
                     'last_modified',NEW.last_modified,'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER keyword_collections_sync_update
AFTER UPDATE ON keyword_collections
WHEN OLD.deleted = NEW.deleted AND (
     OLD.name IS NOT NEW.name OR
     OLD.parent_id IS NOT NEW.parent_id OR
     OLD.last_modified IS NOT NEW.last_modified OR
     OLD.version IS NOT NEW.version)
BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('keyword_collections',CAST(NEW.id AS TEXT),'update',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'name',NEW.name,'parent_id',NEW.parent_id,'created_at',NEW.created_at,
                     'last_modified',NEW.last_modified,'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER keyword_collections_sync_delete
AFTER UPDATE ON keyword_collections
WHEN OLD.deleted = 0 AND NEW.deleted = 1
BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('keyword_collections',CAST(NEW.id AS TEXT),'delete',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'deleted',NEW.deleted,'last_modified',NEW.last_modified,
                     'version',NEW.version,'client_id',NEW.client_id));
END;

CREATE TRIGGER keyword_collections_sync_undelete
AFTER UPDATE ON keyword_collections
WHEN OLD.deleted = 1 AND NEW.deleted = 0
BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('keyword_collections',CAST(NEW.id AS TEXT),'update',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'name',NEW.name,'parent_id',NEW.parent_id,'created_at',NEW.created_at,
                     'last_modified',NEW.last_modified,'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

/*----------------------------------------------------------------
  Finalise version bump
----------------------------------------------------------------*/
UPDATE db_schema_version
   SET version = 3
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 3;
"""

    _FULL_SCHEMA_SQL_V2 = """
PRAGMA foreign_keys = ON;

-- Schema Version Table (Added for library management) --
CREATE TABLE IF NOT EXISTS db_schema_version (
    schema_name TEXT PRIMARY KEY NOT NULL,
    version     INTEGER NOT NULL
);
INSERT OR IGNORE INTO db_schema_version (schema_name, version) VALUES ('rag_char_chat_schema', 0);

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
    last_modified             DATETIME  NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted                   BOOLEAN   NOT NULL DEFAULT 0,
    client_id                 TEXT      NOT NULL DEFAULT 'unknown',
    version                   INTEGER   NOT NULL DEFAULT 1
);
 
CREATE VIRTUAL TABLE IF NOT EXISTS character_cards_fts
USING fts5(
  name,
  description,
  personality,          -- safely indexed again
  scenario,
  system_prompt,
  content = 'character_cards',
  content_rowid = 'id'
);

CREATE TRIGGER IF NOT EXISTS character_cards_ai
AFTER INSERT ON character_cards BEGIN
  INSERT INTO character_cards_fts( rowid,
                                   name, description, personality,
                                   scenario, system_prompt )
  SELECT new.id, new.name, new.description, new.personality,
         new.scenario, new.system_prompt
  WHERE new.deleted = 0;
END;

CREATE TRIGGER IF NOT EXISTS character_cards_au
AFTER UPDATE ON character_cards BEGIN
  /* remove the old revision from the index */
  INSERT INTO character_cards_fts( character_cards_fts, rowid,
                                   name, description, personality,
                                   scenario, system_prompt )
  VALUES( 'delete', old.id, old.name, old.description, old.personality,
          old.scenario, old.system_prompt );

  /* add the new revision if it is not a tombstone */
  INSERT INTO character_cards_fts( rowid,
                                   name, description, personality,
                                   scenario, system_prompt )
  SELECT new.id, new.name, new.description, new.personality,
         new.scenario, new.system_prompt
  WHERE new.deleted = 0;
END;

CREATE TRIGGER IF NOT EXISTS character_cards_ad
AFTER DELETE ON character_cards BEGIN
  INSERT INTO character_cards_fts( character_cards_fts, rowid,
                                   name, description, personality,
                                   scenario, system_prompt )
  VALUES( 'delete', old.id, old.name, old.description, old.personality,
          old.scenario, old.system_prompt );
END;

-- ───────────────────────────────────────────────────────────────────────────
-- 2. Conversations (branches) with tombstones & sync metadata
-- ───────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS conversations (
    id                       TEXT      PRIMARY KEY, -- UUID
    root_id                  TEXT      NOT NULL,    -- UUID of the first conversation in a tree
    forked_from_message_id   TEXT      REFERENCES messages(id) ON DELETE SET NULL,
    parent_conversation_id   TEXT      REFERENCES conversations(id) ON DELETE SET NULL,
    character_id             INTEGER   REFERENCES character_cards(id) ON DELETE CASCADE ON UPDATE CASCADE,
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
CREATE INDEX IF NOT EXISTS idx_conv_char          ON conversations(character_id);


-- =========== CONVERSATIONS FTS TABLE ACTIVE, ALL ITS TRIGGERS ACTIVE ============
CREATE VIRTUAL TABLE IF NOT EXISTS conversations_fts
USING fts5(
    title,
    content='conversations',
    content_rowid='rowid' 
);

CREATE TRIGGER IF NOT EXISTS conversations_ai AFTER INSERT ON conversations BEGIN
  INSERT INTO conversations_fts(rowid, title)
    SELECT new.rowid, new.title
    WHERE new.deleted = 0 AND new.title IS NOT NULL; 
END;

CREATE TRIGGER IF NOT EXISTS conversations_au -- AFTER UPDATE
AFTER UPDATE ON conversations BEGIN
  INSERT INTO conversations_fts(conversations_fts, rowid, title)
  VALUES('delete', old.rowid, old.title);

  INSERT INTO conversations_fts(rowid, title)
  SELECT new.rowid, new.title
  WHERE new.deleted = 0 AND new.title IS NOT NULL;
END;

CREATE TRIGGER conversations_ad
AFTER DELETE ON conversations BEGIN
  INSERT INTO conversations_fts(conversations_fts, rowid, title)
  VALUES('delete', old.rowid, old.title);
END;
-- ========= END SECTION FOR CONVERSATIONS FTS TRIGGERS  ========

-- ───────────────────────────────────────────────────────────────────────────
-- 3. Messages with swipe/fork links, rankings, tombstones & sync metadata
-- ───────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS messages (
    id                   TEXT      PRIMARY KEY, -- UUID
    conversation_id      TEXT      NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    parent_message_id    TEXT      REFERENCES messages(id) ON DELETE SET NULL, 
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
CREATE INDEX IF NOT EXISTS idx_msgs_conv_ts      ON messages(conversation_id, timestamp);

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts
USING fts5(
    content,
    content='messages',
    content_rowid='rowid' -- FIXED: Changed from 'id' to 'rowid' as messages.id is TEXT
);
CREATE TRIGGER messages_ai
AFTER INSERT ON messages BEGIN
  INSERT INTO messages_fts(rowid, content)
  SELECT new.rowid, new.content
  WHERE new.deleted = 0;
END;

CREATE TRIGGER messages_au
AFTER UPDATE ON messages BEGIN
  INSERT INTO messages_fts(messages_fts, rowid, content)
  VALUES('delete', old.rowid, old.content);

  INSERT INTO messages_fts(rowid, content)
  SELECT new.rowid, new.content
  WHERE new.deleted = 0;
END;

CREATE TRIGGER messages_ad
AFTER DELETE ON messages BEGIN
  INSERT INTO messages_fts(messages_fts, rowid, content)
  VALUES('delete', old.rowid, old.content);
END;

-- ───────────────────────────────────────────────────────────────────────────
-- 4. Keywords (with sync metadata + FTS5)
-- ───────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS keywords (
    id          INTEGER  PRIMARY KEY AUTOINCREMENT,
    keyword     TEXT     NOT NULL UNIQUE COLLATE NOCASE,
    created_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted       BOOLEAN  NOT NULL DEFAULT 0,
    client_id     TEXT     NOT NULL DEFAULT 'unknown',
    version       INTEGER  NOT NULL DEFAULT 1
);
CREATE VIRTUAL TABLE IF NOT EXISTS keywords_fts
USING fts5(
    keyword,
    content='keywords',
    content_rowid='id' -- 'id' is INTEGER PK, fine
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
    name        TEXT     NOT NULL UNIQUE COLLATE NOCASE,
    parent_id   INTEGER  REFERENCES keyword_collections(id) ON DELETE SET NULL ON UPDATE CASCADE,
    created_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted       BOOLEAN  NOT NULL DEFAULT 0,
    client_id     TEXT     NOT NULL DEFAULT 'unknown',
    version       INTEGER  NOT NULL DEFAULT 1
);
CREATE VIRTUAL TABLE IF NOT EXISTS keyword_collections_fts
USING fts5(
    name,
    content='keyword_collections',
    content_rowid='id' -- 'id' is INTEGER PK, fine
);
CREATE TRIGGER keyword_collections_ai
AFTER INSERT ON keyword_collections BEGIN
  INSERT INTO keyword_collections_fts(rowid, name)
  SELECT new.id, new.name
  WHERE new.deleted = 0;
END;

CREATE TRIGGER keyword_collections_au
AFTER UPDATE ON keyword_collections BEGIN
  INSERT INTO keyword_collections_fts(keyword_collections_fts, rowid, name)
  VALUES('delete', old.id, old.name);

  INSERT INTO keyword_collections_fts(rowid, name)
  SELECT new.id, new.name
  WHERE new.deleted = 0;
END;

CREATE TRIGGER keyword_collections_ad
AFTER DELETE ON keyword_collections BEGIN
  INSERT INTO keyword_collections_fts(keyword_collections_fts, rowid, name)
  VALUES('delete', old.id, old.name);
END;

-- ───────────────────────────────────────────────────────────────────────────
-- 6. Notes (Independent, with sync metadata + FTS5)
-- ───────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS notes (
    id            TEXT      PRIMARY KEY, 
    title         TEXT      NOT NULL,    
    content       TEXT      NOT NULL,
    created_at    DATETIME  NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_modified DATETIME  NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted       BOOLEAN   NOT NULL DEFAULT 0,
    client_id     TEXT      NOT NULL DEFAULT 'unknown',
    version       INTEGER   NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_notes_last_modified ON notes(last_modified);

CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts
USING fts5(
    title,
    content,
    content='notes',
    content_rowid='rowid' -- FIXED: 'id' is TEXT PK, changed to 'rowid'
);
CREATE TRIGGER notes_ai
AFTER INSERT ON notes BEGIN
  INSERT INTO notes_fts(rowid, title, content)
  SELECT new.rowid, new.title, new.content
  WHERE new.deleted = 0;
END;

CREATE TRIGGER notes_au
AFTER UPDATE ON notes BEGIN
  INSERT INTO notes_fts(notes_fts, rowid, title, content)
  VALUES('delete', old.rowid, old.title, old.content);

  INSERT INTO notes_fts(rowid, title, content)
  SELECT new.rowid, new.title, new.content
  WHERE new.deleted = 0;
END;

CREATE TRIGGER notes_ad
AFTER DELETE ON notes BEGIN
  INSERT INTO notes_fts(notes_fts, rowid, title, content)
  VALUES('delete', old.rowid, old.title, old.content);
END;

-- ───────────────────────────────────────────────────────────────────────────
-- 7. Linking Tables (with created_at for potential sync ordering)
-- ───────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS conversation_keywords (
    conversation_id TEXT    NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    keyword_id      INTEGER NOT NULL REFERENCES keywords(id) ON DELETE CASCADE ON UPDATE CASCADE,
    created_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(conversation_id, keyword_id)
);
CREATE INDEX IF NOT EXISTS idx_convkw_kw ON conversation_keywords(keyword_id);

CREATE TABLE IF NOT EXISTS collection_keywords (
    collection_id INTEGER NOT NULL REFERENCES keyword_collections(id) ON DELETE CASCADE ON UPDATE CASCADE,
    keyword_id    INTEGER NOT NULL REFERENCES keywords(id) ON DELETE CASCADE ON UPDATE CASCADE,
    created_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(collection_id, keyword_id)
);
CREATE INDEX IF NOT EXISTS idx_collkw_kw ON collection_keywords(keyword_id);

CREATE TABLE IF NOT EXISTS note_keywords (
    note_id    TEXT    NOT NULL REFERENCES notes(id) ON DELETE CASCADE ON UPDATE CASCADE, -- Changed from INTEGER to TEXT
    keyword_id INTEGER NOT NULL REFERENCES keywords(id) ON DELETE CASCADE ON UPDATE CASCADE,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(note_id, keyword_id)
);
CREATE INDEX IF NOT EXISTS idx_notekw_kw ON note_keywords(keyword_id);

-- ───────────────────────────────────────────────────────────────────────────
-- 8. Sync Log for bi‑directional, offline‑first synchronization
-- ───────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sync_log (
    change_id    INTEGER  PRIMARY KEY AUTOINCREMENT,
    entity       TEXT     NOT NULL,
    entity_id    TEXT     NOT NULL,
    operation    TEXT     NOT NULL   CHECK(operation IN ('create','update','delete')),
    timestamp    DATETIME NOT NULL,
    client_id    TEXT     NOT NULL,
    version      INTEGER  NOT NULL,
    payload      TEXT     NOT NULL -- JSON blob
);
CREATE INDEX IF NOT EXISTS idx_sync_log_ts     ON sync_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_sync_log_entity ON sync_log(entity, entity_id);

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
WHEN OLD.deleted = NEW.deleted AND (
     OLD.content IS NOT NEW.content OR
     OLD.ranking IS NOT NEW.ranking OR
     OLD.parent_message_id IS NOT NEW.parent_message_id OR
     OLD.last_modified IS NOT NEW.last_modified OR -- Should be NEW.last_modified for timestamp
     OLD.version IS NOT NEW.version -- Should be NEW.version
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
       json_object('id', NEW.id, 'deleted', NEW.deleted, 'last_modified', NEW.last_modified, 'version', NEW.version, 'client_id', NEW.client_id)
    );
END;

CREATE TRIGGER IF NOT EXISTS messages_sync_undelete -- Handles UNDELETE
AFTER UPDATE ON messages
WHEN OLD.deleted = 1 AND NEW.deleted = 0
BEGIN
    INSERT INTO sync_log (entity,entity_id,operation,timestamp,client_id,version,payload)
    VALUES (
      'messages', NEW.id, 'update', -- Treat undelete as an update with full payload
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
         'rating', NEW.rating, 'created_at', NEW.created_at, 'last_modified', NEW.last_modified, 'deleted', NEW.deleted,
         'client_id', NEW.client_id, 'version', NEW.version
       )
    );
END;

CREATE TRIGGER IF NOT EXISTS conversations_sync_update
AFTER UPDATE ON conversations
WHEN OLD.deleted = NEW.deleted AND (
     OLD.title IS NOT NEW.title OR
     OLD.rating IS NOT NEW.rating OR
     OLD.forked_from_message_id IS NOT NEW.forked_from_message_id OR
     OLD.parent_conversation_id IS NOT NEW.parent_conversation_id OR
     OLD.character_id IS NOT NEW.character_id OR
     OLD.last_modified IS NOT NEW.last_modified OR
     OLD.version IS NOT NEW.version
)
BEGIN
  INSERT INTO sync_log (entity,entity_id,operation,timestamp,client_id,version,payload)
    VALUES (
      'conversations', NEW.id, 'update',
       NEW.last_modified, NEW.client_id, NEW.version,
       json_object(
         'id', NEW.id, 'root_id', NEW.root_id, 'forked_from_message_id', NEW.forked_from_message_id,
         'parent_conversation_id', NEW.parent_conversation_id, 'character_id', NEW.character_id, 'title', NEW.title,
         'rating', NEW.rating, 'created_at', NEW.created_at, 'last_modified', NEW.last_modified, 'deleted', NEW.deleted,
         'client_id', NEW.client_id, 'version', NEW.version
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
       json_object('id', NEW.id, 'deleted', NEW.deleted, 'last_modified', NEW.last_modified, 'version', NEW.version, 'client_id', NEW.client_id)
    );
END;

CREATE TRIGGER IF NOT EXISTS conversations_sync_undelete
AFTER UPDATE ON conversations
WHEN OLD.deleted = 1 AND NEW.deleted = 0
BEGIN
    INSERT INTO sync_log (entity,entity_id,operation,timestamp,client_id,version,payload)
    VALUES (
      'conversations', NEW.id, 'update',
       NEW.last_modified, NEW.client_id, NEW.version,
       json_object(
         'id', NEW.id, 'root_id', NEW.root_id, 'forked_from_message_id', NEW.forked_from_message_id,
         'parent_conversation_id', NEW.parent_conversation_id, 'character_id', NEW.character_id, 'title', NEW.title,
         'rating', NEW.rating, 'created_at', NEW.created_at, 'last_modified', NEW.last_modified, 'deleted', NEW.deleted,
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
         'scenario', NEW.scenario, -- 'image' BLOB intentionally excluded from payload here for size
         'post_history_instructions', NEW.post_history_instructions, 'first_message', NEW.first_message,
         'message_example', NEW.message_example, 'creator_notes', NEW.creator_notes, 'system_prompt', NEW.system_prompt,
         'alternate_greetings', NEW.alternate_greetings, 'tags', NEW.tags, 'creator', NEW.creator,
         'character_version', NEW.character_version, 'extensions', NEW.extensions, 'created_at', NEW.created_at,
         'last_modified', NEW.last_modified, 'deleted', NEW.deleted, 'client_id', NEW.client_id, 'version', NEW.version
         -- To include image (as hex or indication): ,'image_present', CASE WHEN NEW.image IS NOT NULL THEN 1 ELSE 0 END
       )
    );
END;

CREATE TRIGGER IF NOT EXISTS character_cards_sync_update
AFTER UPDATE ON character_cards
WHEN OLD.deleted = NEW.deleted AND (
     OLD.name IS NOT NEW.name OR
     OLD.description IS NOT NEW.description OR
     OLD.personality IS NOT NEW.personality OR
     OLD.scenario IS NOT NEW.scenario OR
     OLD.image IS NOT NEW.image OR -- BLOB comparison might be slow/problematic
     OLD.post_history_instructions IS NOT NEW.post_history_instructions OR
     OLD.first_message IS NOT NEW.first_message OR
     OLD.message_example IS NOT NEW.message_example OR
     OLD.creator_notes IS NOT NEW.creator_notes OR
     OLD.system_prompt IS NOT NEW.system_prompt OR
     OLD.alternate_greetings IS NOT NEW.alternate_greetings OR
     OLD.tags IS NOT NEW.tags OR
     OLD.creator IS NOT NEW.creator OR
     OLD.character_version IS NOT NEW.character_version OR
     OLD.extensions IS NOT NEW.extensions OR
     OLD.last_modified IS NOT NEW.last_modified OR
     OLD.version IS NOT NEW.version
)
BEGIN
  INSERT INTO sync_log (entity,entity_id,operation,timestamp,client_id,version,payload)
    VALUES (
      'character_cards', CAST(NEW.id AS TEXT), 'update',
       NEW.last_modified, NEW.client_id, NEW.version,
       json_object(
         'id', NEW.id, 'name', NEW.name, 'description', NEW.description, 'personality', NEW.personality,
         'scenario', NEW.scenario, -- 'image_present', CASE WHEN NEW.image IS NOT NULL THEN 1 ELSE 0 END,
         'post_history_instructions', NEW.post_history_instructions, 'first_message', NEW.first_message,
         'message_example', NEW.message_example, 'creator_notes', NEW.creator_notes, 'system_prompt', NEW.system_prompt,
         'alternate_greetings', NEW.alternate_greetings, 'tags', NEW.tags, 'creator', NEW.creator,
         'character_version', NEW.character_version, 'extensions', NEW.extensions, 'created_at', NEW.created_at,
         'last_modified', NEW.last_modified, 'deleted', NEW.deleted, 'client_id', NEW.client_id, 'version', NEW.version
       )
    );
END;

CREATE TRIGGER IF NOT EXISTS character_cards_sync_delete
AFTER UPDATE ON character_cards
WHEN OLD.deleted = 0 AND NEW.deleted = 1
BEGIN
  INSERT INTO sync_log (entity,entity_id,operation,timestamp,client_id,version,payload)
    VALUES (
      'character_cards', CAST(NEW.id AS TEXT), 'delete',
       NEW.last_modified, NEW.client_id, NEW.version,
       json_object('id', NEW.id, 'deleted', NEW.deleted, 'last_modified', NEW.last_modified, 'version', NEW.version, 'client_id', NEW.client_id)
    );
END;

CREATE TRIGGER IF NOT EXISTS character_cards_sync_undelete
AFTER UPDATE ON character_cards
WHEN OLD.deleted = 1 AND NEW.deleted = 0
BEGIN
    INSERT INTO sync_log (entity,entity_id,operation,timestamp,client_id,version,payload)
    VALUES (
      'character_cards', CAST(NEW.id AS TEXT), 'update',
       NEW.last_modified, NEW.client_id, NEW.version,
       json_object(
         'id', NEW.id, 'name', NEW.name, 'description', NEW.description, 'personality', NEW.personality,
         'scenario', NEW.scenario, -- 'image_present', CASE WHEN NEW.image IS NOT NULL THEN 1 ELSE 0 END,
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
      'notes', NEW.id, 'create', -- No CAST needed, id is TEXT
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
     OLD.title IS NOT NEW.title OR
     OLD.content IS NOT NEW.content OR
     OLD.last_modified IS NOT NEW.last_modified OR
     OLD.version IS NOT NEW.version
)
BEGIN
  INSERT INTO sync_log (entity,entity_id,operation,timestamp,client_id,version,payload)
    VALUES (
      'notes', NEW.id, 'update', -- No CAST needed
       NEW.last_modified, NEW.client_id, NEW.version,
       json_object(
         'id', NEW.id, 'title', NEW.title, 'content', NEW.content,
         'created_at', NEW.created_at, 'last_modified', NEW.last_modified,
         'deleted', NEW.deleted, 'client_id', NEW.client_id, 'version', NEW.version
       )
    );
END;

CREATE TRIGGER IF NOT EXISTS notes_sync_delete
AFTER UPDATE ON notes
WHEN OLD.deleted = 0 AND NEW.deleted = 1
BEGIN
  INSERT INTO sync_log (entity,entity_id,operation,timestamp,client_id,version,payload)
    VALUES (
      'notes', NEW.id, 'delete', -- No CAST needed
       NEW.last_modified, NEW.client_id, NEW.version,
       json_object('id', NEW.id, 'deleted', NEW.deleted, 'last_modified', NEW.last_modified, 'version', NEW.version, 'client_id', NEW.client_id)
    );
END;

CREATE TRIGGER IF NOT EXISTS notes_sync_undelete
AFTER UPDATE ON notes
WHEN OLD.deleted = 1 AND NEW.deleted = 0
BEGIN
    INSERT INTO sync_log (entity,entity_id,operation,timestamp,client_id,version,payload)
    VALUES (
      'notes', NEW.id, 'update', -- No CAST needed
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
     OLD.keyword IS NOT NEW.keyword OR -- Though keyword text itself is unlikely to change
     OLD.last_modified IS NOT NEW.last_modified OR
     OLD.version IS NOT NEW.version
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

CREATE TRIGGER IF NOT EXISTS keywords_sync_delete
AFTER UPDATE ON keywords
WHEN OLD.deleted = 0 AND NEW.deleted = 1
BEGIN
  INSERT INTO sync_log (entity,entity_id,operation,timestamp,client_id,version,payload)
    VALUES (
      'keywords', CAST(NEW.id AS TEXT), 'delete',
       NEW.last_modified, NEW.client_id, NEW.version,
       json_object('id', NEW.id, 'deleted', NEW.deleted, 'last_modified', NEW.last_modified, 'version', NEW.version, 'client_id', NEW.client_id)
    );
END;

CREATE TRIGGER IF NOT EXISTS keywords_sync_undelete
AFTER UPDATE ON keywords
WHEN OLD.deleted = 1 AND NEW.deleted = 0
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
     OLD.name IS NOT NEW.name OR
     OLD.parent_id IS NOT NEW.parent_id OR
     OLD.last_modified IS NOT NEW.last_modified OR
     OLD.version IS NOT NEW.version
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

CREATE TRIGGER IF NOT EXISTS keyword_collections_sync_delete
AFTER UPDATE ON keyword_collections
WHEN OLD.deleted = 0 AND NEW.deleted = 1
BEGIN
  INSERT INTO sync_log (entity,entity_id,operation,timestamp,client_id,version,payload)
    VALUES (
      'keyword_collections', CAST(NEW.id AS TEXT), 'delete',
       NEW.last_modified, NEW.client_id, NEW.version,
       json_object('id', NEW.id, 'deleted', NEW.deleted, 'last_modified', NEW.last_modified, 'version', NEW.version, 'client_id', NEW.client_id)
    );
END;

CREATE TRIGGER IF NOT EXISTS keyword_collections_sync_undelete
AFTER UPDATE ON keyword_collections
WHEN OLD.deleted = 1 AND NEW.deleted = 0
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

-- == Triggers for LINK TABLES ==
-- Sync log entries for link tables are now handled by Python methods.
-- SQL Triggers for link table sync_log entries have been removed.

-- Final step: Update schema version to 2
UPDATE db_schema_version SET version = 2 WHERE schema_name = 'rag_char_chat_schema' AND version = 0;
    """

    def __init__(self, db_path: Union[str, Path], client_id: str):
        if isinstance(db_path, Path):
            self.is_memory_db = False
            self.db_path = db_path.resolve()
        else:
            self.is_memory_db = (db_path == ':memory:')
            self.db_path = Path(db_path).resolve() if not self.is_memory_db else Path(":memory:")
        self.db_path_str = str(self.db_path) if not self.is_memory_db else ':memory:'

        if not client_id:
            raise ValueError("Client ID cannot be empty or None.")
        self.client_id = client_id

        if not self.is_memory_db:
            try:
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                raise CharactersRAGDBError(f"Failed to create database directory {self.db_path.parent}: {e}")

        logger.info(f"Initializing CharactersRAGDB for path: {self.db_path_str} [Client ID: {self.client_id}]")
        self._local = threading.local()
        try:
            self._initialize_schema()
            logger.debug(f"CharactersRAGDB initialization completed successfully for {self.db_path_str}")
        except (CharactersRAGDBError, sqlite3.Error) as e:
            logger.critical(f"FATAL: DB Initialization failed for {self.db_path_str}: {e}", exc_info=True)
            self.close_connection()  # Attempt to clean up
            raise CharactersRAGDBError(f"Database initialization failed: {e}") from e
        except Exception as e:
            logger.critical(f"FATAL: Unexpected error during DB Initialization for {self.db_path_str}: {e}",
                            exc_info=True)
            self.close_connection()
            raise CharactersRAGDBError(f"Unexpected database initialization error: {e}") from e

    # --- Connection Management ---
    def _get_thread_connection(self) -> sqlite3.Connection:
        conn = getattr(self._local, 'conn', None)
        if conn:
            try:
                conn.execute("SELECT 1")  # Check if connection is still alive
            except (sqlite3.ProgrammingError, sqlite3.OperationalError):
                logger.warning(
                    f"Thread-local connection for {self.db_path_str} was closed or became unusable. Reopening.")
                try:
                    conn.close()
                except Exception:
                    pass
                conn = None

        if not conn:
            try:
                conn = sqlite3.connect(
                    self.db_path_str,
                    detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
                    check_same_thread=False, # Required for threading.local approach
                    timeout=15 # Maybe slightly increase timeout?
                )
                conn.row_factory = sqlite3.Row
                if not self.is_memory_db:
                    conn.execute("PRAGMA journal_mode=WAL;")

                conn.execute("PRAGMA foreign_keys = ON;")
                self._local.conn = conn
                logger.debug(
                    f"Opened/Reopened SQLite connection to {self.db_path_str} (Journal: {conn.execute('PRAGMA journal_mode;').fetchone()[0]}) for thread {threading.get_ident()}")
            except sqlite3.Error as e:
                logger.error(f"Failed to connect to database {self.db_path_str}: {e}", exc_info=True)
                self._local.conn = None
                raise CharactersRAGDBError(f"Failed to connect to database '{self.db_path_str}': {e}") from e
        return self._local.conn

    def get_connection(self) -> sqlite3.Connection:
        return self._get_thread_connection()

    def close_connection(self):
        conn = getattr(self._local, 'conn', None)
        if conn is not None:
            try:
                if not self.is_memory_db:
                    # Resolve any pending transaction before checkpointing
                    if conn.in_transaction:
                        try:
                            logger.warning(
                                f"Connection to {self.db_path_str} is in an uncommitted transaction during close. Attempting rollback.")
                            conn.rollback()  # Attempt rollback if transaction is open
                        except sqlite3.Error as rb_err:
                            logger.error(f"Rollback attempt during close for {self.db_path_str} failed: {rb_err}")
                            # Don't proceed to checkpoint if rollback fails and we're still in transaction potentially
                            # However, conn.close() below should still be attempted.

                    # Checkpoint WAL only if not in a failed transaction state that prevents it
                    # and WAL mode is active.
                    # We assume if conn.in_transaction is false now, any transaction was committed/rolled back.
                    if not conn.in_transaction:  # Re-check after potential rollback
                        mode_row = conn.execute("PRAGMA journal_mode;").fetchone()
                        if mode_row and mode_row[0].lower() == 'wal':
                            try:
                                logger.debug(
                                    f"Attempting WAL checkpoint (TRUNCATE) before closing {self.db_path_str} on thread {threading.get_ident()}.")
                                conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                                logger.debug(f"WAL checkpoint TRUNCATE executed for {self.db_path_str}.")
                            except sqlite3.Error as cp_err:
                                logger.warning(f"WAL checkpoint failed for {self.db_path_str}: {cp_err}")
                conn.close()
                logger.debug(f"Closed connection for thread {threading.get_ident()} to {self.db_path_str}.")
            except sqlite3.Error as e:  # Catches errors from execute, checkpoint, or close
                logger.warning(
                    f"Error during SQLite connection close/checkpoint for {self.db_path_str} on thread {threading.get_ident()}: {e}")
            finally:
                # This ensures that the reference is cleared from threading.local
                # even if conn.close() itself raised an exception.
                if hasattr(self._local, 'conn'):
                    self._local.conn = None

    # --- Query Execution ---
    def execute_query(self, query: str, params: Optional[Union[tuple, Dict[str, Any]]] = None, *, commit: bool = False,
                      script: bool = False) -> sqlite3.Cursor:
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            if logger.isEnabledFor(logging.DEBUG):  # Avoid formatting query/params if not debugging
                logger.debug(f"Executing SQL (script={script}): {query[:300]}... Params: {str(params)[:200]}...")

            if script:
                cursor.executescript(query)
            else:
                cursor.execute(query, params or ())

            if commit and not conn.in_transaction:  # Only commit if not already in a transaction handled by the context manager
                conn.commit()
                logger.debug("Committed directly by execute_query.")
            return cursor
        except sqlite3.IntegrityError as e:
            logger.warning(f"Integrity constraint violation: {query[:300]}... Error: {e}")
            # Distinguish unique constraint from other integrity errors if possible
            if "unique constraint failed" in str(e).lower():
                raise ConflictError(message=f"Unique constraint violation: {e}") from e
            raise CharactersRAGDBError(
                f"Database constraint violation: {e}") from e  # Broader for other integrity issues
        except sqlite3.Error as e:
            logger.error(f"Query execution failed: {query[:300]}... Error: {e}", exc_info=True)
            raise CharactersRAGDBError(f"Query execution failed: {e}") from e

    def execute_many(self, query: str, params_list: List[tuple], *, commit: bool = False) -> Optional[sqlite3.Cursor]:
        conn = self.get_connection()
        if not isinstance(params_list, list) or not params_list:
            logger.debug("execute_many called with empty or invalid params_list.")
            return None
        try:
            cursor = conn.cursor()
            logger.debug(f"Executing Many: {query[:150]}... with {len(params_list)} sets.")
            cursor.executemany(query, params_list)
            if commit and not conn.in_transaction:  # Only commit if not already in a transaction
                conn.commit()
                logger.debug("Committed Many directly by execute_many.")
            return cursor
        except sqlite3.IntegrityError as e:
            logger.warning(f"Integrity constraint violation during batch: {query[:150]}... Error: {e}")
            if "unique constraint failed" in str(e).lower():
                raise ConflictError(message=f"Unique constraint violation during batch: {e}") from e
            raise CharactersRAGDBError(f"Database constraint violation during batch: {e}") from e
        except sqlite3.Error as e:
            logger.error(f"Execute Many failed: {query[:150]}... Error: {e}", exc_info=True)
            raise CharactersRAGDBError(f"Execute Many failed: {e}") from e

    # --- Transaction Context ---
    def transaction(self) -> 'TransactionContextManager':
        return TransactionContextManager(self)

    # --- Schema Initialization and Migration ---
    def _get_db_version(self, conn: sqlite3.Connection) -> int:
        try:
            cursor = conn.execute("SELECT version FROM db_schema_version WHERE schema_name = ? LIMIT 1",
                                  (self._SCHEMA_NAME,))
            result = cursor.fetchone()
            return result['version'] if result else 0
        except sqlite3.Error as e:
            if "no such table" in str(e).lower() and "db_schema_version" in str(e).lower():
                return 0
            logger.error(f"Could not determine database schema version for '{self._SCHEMA_NAME}': {e}", exc_info=True)
            raise SchemaError(f"Could not determine schema version for '{self._SCHEMA_NAME}': {e}") from e

    # FIXME - Fixup to use f-strings; ya.
    def _apply_schema_v3(self, conn: sqlite3.Connection): # Renamed from _apply_schema_v1
        logger.info(f"Applying schema Version {self._CURRENT_SCHEMA_VERSION} for '{self._SCHEMA_NAME}' to DB: {self.db_path_str}...")
        try:
            # Using conn.executescript directly as it manages its own transaction
            #conn.executescript(self._FULL_SCHEMA_SQL_V3)
            conn.executescript(self._FULL_SCHEMA_SQL_V3) # DEBUGTEST
            logger.debug(f"[{self._SCHEMA_NAME} V{self._CURRENT_SCHEMA_VERSION}] Full schema script executed.")

            final_version = self._get_db_version(conn)
            if final_version != self._CURRENT_SCHEMA_VERSION:
                raise SchemaError(
                    f"[{self._SCHEMA_NAME} {self._CURRENT_SCHEMA_VERSION} Schema version update check failed. Expected 3, got: {final_version}")
            logger.info(f"[{self._SCHEMA_NAME} {self._CURRENT_SCHEMA_VERSION}] Schema {self._CURRENT_SCHEMA_VERSION} applied and version confirmed for DB: {self.db_path_str}.")
        except sqlite3.Error as e:
            logger.error(f"[{self._SCHEMA_NAME} V3] Schema application failed: {e}", exc_info=True)
            raise SchemaError(f"DB schema V3 setup failed for '{self._SCHEMA_NAME}': {e}") from e
        except SchemaError:
            raise
        except Exception as e:
            logger.error(f"[{self._SCHEMA_NAME} V3] Unexpected error during schema V3 application: {e}", exc_info=True)
            raise SchemaError(f"Unexpected error applying schema V3 for '{self._SCHEMA_NAME}': {e}") from e

    def _apply_schema_v2(self, conn: sqlite3.Connection): # Renamed from _apply_schema_v1
        logger.info(f"Applying schema Version {self._CURRENT_SCHEMA_VERSION} for '{self._SCHEMA_NAME}' to DB: {self.db_path_str}...")
        try:
            # Using conn.executescript directly as it manages its own transaction
            #conn.executescript(self._FULL_SCHEMA_SQL_V2)
            conn.executescript(self._FULL_SCHEMA_SQL_V2) # DEBUGTEST
            logger.debug(f"[{self._SCHEMA_NAME} V{self._CURRENT_SCHEMA_VERSION}] Full schema script executed.")

            final_version = self._get_db_version(conn)
            if final_version != self._CURRENT_SCHEMA_VERSION:
                raise SchemaError(
                    f"[{self._SCHEMA_NAME} {self._CURRENT_SCHEMA_VERSION} Schema version update check failed. Expected 2, got: {final_version}")
            logger.info(f"[{self._SCHEMA_NAME} {self._CURRENT_SCHEMA_VERSION}] Schema {self._CURRENT_SCHEMA_VERSION} applied and version confirmed for DB: {self.db_path_str}.")
        except sqlite3.Error as e:
            logger.error(f"[{self._SCHEMA_NAME} V2] Schema application failed: {e}", exc_info=True)
            raise SchemaError(f"DB schema V2 setup failed for '{self._SCHEMA_NAME}': {e}") from e
        except SchemaError:
            raise
        except Exception as e:
            logger.error(f"[{self._SCHEMA_NAME} V2] Unexpected error during schema V2 application: {e}", exc_info=True)
            raise SchemaError(f"Unexpected error applying schema V2 for '{self._SCHEMA_NAME}': {e}") from e

    def _initialize_schema(self):
        conn = self.get_connection()
        current_initial_version = 0  # To track version before applying schema
        try:
            current_db_version = self._get_db_version(conn)
            current_initial_version = self._get_db_version(conn)
            target_version = self._CURRENT_SCHEMA_VERSION
            logger.info(
                f"Checking DB schema '{self._SCHEMA_NAME}'. Current version: {current_db_version}. Code supports: {target_version}")

            if current_db_version == target_version:
                logger.debug(f"Database schema '{self._SCHEMA_NAME}' is up to date.")
                return
            if current_db_version > target_version:
                raise SchemaError(
                    f"Database schema '{self._SCHEMA_NAME}' version ({current_db_version}) is newer than supported by code ({target_version}). Aborting.")

            if current_db_version == 0:
                self._apply_schema_v3(conn)
            # Add future migrations here:
            # if current_db_version == 1: self._migrate_schema_v1_to_v2(conn); current_db_version = 2
            elif current_initial_version < target_version:
                # This means an older schema exists (e.g., V1) and no direct migration path
                # is defined in this simplified _initialize_schema function to bring it to V2.
                raise SchemaError(
                    f"Migration path undefined for '{self._SCHEMA_NAME}' from version {current_initial_version} to {target_version}. "
                    f"Manual migration or a new database may be required.")
            else: # Should not be reached due to prior checks
                 raise SchemaError(f"Unexpected schema state: current {current_initial_version}, target {target_version}")


            final_version_check = self._get_db_version(conn)
            if final_version_check != target_version:
                raise SchemaError(
                    f"Schema migration process completed, but final DB version is {final_version_check}, expected {target_version}. Manual check required.")
            logger.info(
                f"Database schema '{self._SCHEMA_NAME}' successfully initialized/migrated to version {final_version_check}.")
        except (SchemaError, sqlite3.Error) as e:
            logger.error(f"Schema initialization/migration failed for '{self._SCHEMA_NAME}': {e}", exc_info=True)
            raise SchemaError(f"Schema initialization/migration for '{self._SCHEMA_NAME}' failed: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error during schema initialization for '{self._SCHEMA_NAME}': {e}", exc_info=True)
            raise CharactersRAGDBError(f"Unexpected error applying schema for '{self._SCHEMA_NAME}': {e}") from e

    # --- Internal Helpers ---
    def _get_current_utc_timestamp_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')

    def _generate_uuid(self) -> str:
        return str(uuid.uuid4())

    def _get_current_db_version(self, conn: sqlite3.Connection, table_name: str, pk_col_name: str,
                                pk_value: Any) -> int:
        """
        Fetches the current version of an active (not soft-deleted) record.
        Raises ConflictError if the record is not found or is soft-deleted.
        Returns the version number if found and active.
        """
        cursor = conn.execute(f"SELECT version, deleted FROM {table_name} WHERE {pk_col_name} = ?", (pk_value,))
        row = cursor.fetchone()

        if not row:
            logger.warning(f"Record not found in {table_name} with {pk_col_name} = {pk_value} for version check.")
            raise ConflictError(f"Record not found in {table_name}.", entity=table_name, entity_id=pk_value)

        if row['deleted']:
            logger.warning(f"Record in {table_name} with {pk_col_name} = {pk_value} is soft-deleted.")
            raise ConflictError(f"Record is soft-deleted in {table_name}.", entity=table_name, entity_id=pk_value)

        return row['version']

    def _ensure_json_string(self, data: Optional[Union[List, Dict, Set]]) -> Optional[str]:
        if data is None:
            return None
        if isinstance(data, Set):
            data = list(data)  # Convert set to list before dumping
        return json.dumps(data)

    def _deserialize_row_fields(self, row: sqlite3.Row, json_fields: List[str]) -> Optional[Dict[str, Any]]:
        if not row:
            return None
        item = dict(row)
        for field in json_fields:
            if field in item and isinstance(item[field], str):
                try:
                    item[field] = json.loads(item[field])
                except json.JSONDecodeError:
                    pk_val = item.get('id') or item.get('uuid', 'N/A')  # Try to get an identifier
                    logger.warning(
                        f"Failed to decode JSON for field '{field}' in row (ID: {pk_val}). Value: '{item[field][:100]}...'")
                    item[field] = None  # Or sensible default
        return item

    _CHARACTER_CARD_JSON_FIELDS = ['alternate_greetings', 'tags', 'extensions']

    # --- Character Card Methods ---
    def _ensure_json_string_from_mixed(data: Optional[Union[List, Dict, Set, str]]) -> Optional[str]:
        if data is None:
            return None
        if isinstance(data, str):  # If it's already a string, assume it's valid JSON or pass it through
            try:
                json.loads(data)  # Validate if it's a JSON string
                return data
            except json.JSONDecodeError:
                # If it's a plain string not meant to be JSON, this logic might need adjustment
                # For now, let's assume if it's a string, it's intended as a pre-formatted JSON string
                return data  # Or raise an error if non-JSON strings are not allowed for these fields
        if isinstance(data, Set):
            new_data = list(data)
            return json.dumps(new_data)
        return json.dumps(data)

    def add_character_card(self, card_data: Dict[str, Any]) -> Optional[int]:
        """
        Adds a new character card to the database.
        The `client_id` is taken from the DB instance.
        `version` defaults to 1. `last_modified` is set to current time.
        `alternate_greetings`, `tags`, `extensions` are stored as JSON strings.
        Relies on SQL triggers for sync_log and FTS updates.
        """

        required_fields = ['name']
        for field in required_fields:
            if field not in card_data or not card_data[field]:
                raise InputError(f"Required field '{field}' is missing or empty.")

        now = self._get_current_utc_timestamp_iso()

        # Ensure JSON fields are strings or None
        def get_json_field_as_string(field_value):
            if isinstance(field_value, str):
                # Assume it's already a JSON string if it's a string
                return field_value
            return self._ensure_json_string(field_value)

        alt_greetings_json = get_json_field_as_string(card_data.get('alternate_greetings'))
        tags_json = get_json_field_as_string(card_data.get('tags'))
        extensions_json = get_json_field_as_string(card_data.get('extensions'))

        query = """
                INSERT INTO character_cards (name, description, personality, scenario, image, post_history_instructions, \
                                             first_message, message_example, creator_notes, system_prompt, \
                                             alternate_greetings, tags, creator, character_version, extensions, \
                                             last_modified, client_id, version, deleted) \
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0) \
                """
        params = (
            card_data['name'], card_data.get('description'), card_data.get('personality'),
            card_data.get('scenario'), card_data.get('image'), card_data.get('post_history_instructions'),
            card_data.get('first_message'), card_data.get('message_example'), card_data.get('creator_notes'),
            card_data.get('system_prompt'), alt_greetings_json, tags_json,
            card_data.get('creator'), card_data.get('character_version'), extensions_json,
            now, self.client_id, 1  # version defaults to 1, deleted to 0
        )
        try:
            with self.transaction() as conn:
                cursor = conn.execute(query, params)  # execute_query not needed due to conn from context
                char_id = cursor.lastrowid
                logger.info(f"Added character card '{card_data['name']}' with ID: {char_id}.")
                return char_id
        except ConflictError as e:
            if "character_cards.name" in str(e).lower(): # This check is specific to execute_query's ConflictError re-raising
                # If using conn.execute directly, the original sqlite3.IntegrityError is caught by transaction manager.
                # The transaction manager will re-raise based on its own logic or a generic CharactersRAGDBError.
                # For specific handling of 'character_cards.name' conflict, we might need to catch sqlite3.IntegrityError here
                # and inspect it, or ensure execute_query's logic for raising ConflictError is robust.
                # Assuming ConflictError is raised correctly by transaction manager or execute_query for unique constraints.
                logger.warning(f"Character card with name '{card_data['name']}' already exists.")
                raise ConflictError(f"Character card with name '{card_data['name']}' already exists.",
                                    entity="character_cards", entity_id=card_data['name']) from e
            raise
        except sqlite3.IntegrityError as e: # Catching this if conn.execute is used directly
            if "UNIQUE constraint failed: character_cards.name" in str(e):
                logger.warning(f"Character card with name '{card_data['name']}' already exists.")
                raise ConflictError(f"Character card with name '{card_data['name']}' already exists.",
                                    entity="character_cards", entity_id=card_data['name']) from e
            raise CharactersRAGDBError(f"Database integrity error adding character card: {e}") from e
        except CharactersRAGDBError as e:
            logger.error(f"Database error adding character card '{card_data.get('name')}': {e}")
            raise

    def get_character_card_by_id(self, character_id: int) -> Optional[Dict[str, Any]]:
        query = "SELECT * FROM character_cards WHERE id = ? AND deleted = 0"
        try:
            cursor = self.execute_query(query, (character_id,))
            row = cursor.fetchone()
            return self._deserialize_row_fields(row, self._CHARACTER_CARD_JSON_FIELDS)
        except CharactersRAGDBError as e:
            logger.error(f"Database error fetching character card ID {character_id}: {e}")
            raise

    def get_character_card_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        query = "SELECT * FROM character_cards WHERE name = ? AND deleted = 0"
        try:
            cursor = self.execute_query(query, (name,))
            row = cursor.fetchone()
            return self._deserialize_row_fields(row, self._CHARACTER_CARD_JSON_FIELDS)
        except CharactersRAGDBError as e:
            logger.error(f"Database error fetching character card by name '{name}': {e}")
            raise

    def list_character_cards(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        query = "SELECT * FROM character_cards WHERE deleted = 0 ORDER BY name LIMIT ? OFFSET ?"
        try:
            cursor = self.execute_query(query, (limit, offset))
            rows = cursor.fetchall()
            return [self._deserialize_row_fields(row, self._CHARACTER_CARD_JSON_FIELDS) for row in rows if row]
        except CharactersRAGDBError as e:
            logger.error(f"Database error listing character cards: {e}")
            raise

    def update_character_card(self, character_id: int, card_data: Dict[str, Any], expected_version: int) -> bool:
        logger.debug(
            f"Starting update_character_card for ID {character_id}, expected_version {expected_version} (SINGLE UPDATE STRATEGY)")

        # If card_data is empty, treat as a no-op as per original behavior.
        # No version check, no transaction, no version bump.
        if not card_data:
            logger.info(f"No data provided in card_data for character card update ID {character_id}. No-op.")
            return True

        now = self._get_current_utc_timestamp_iso()

        try:
            with self.transaction() as conn:
                logger.debug(f"Transaction started. Connection object: {id(conn)}")

                # Initial version check. This also confirms the record exists and is not deleted.
                current_db_version_initial_check = self._get_current_db_version(conn, "character_cards", "id",
                                                                                character_id)
                logger.debug(
                    f"Initial DB version: {current_db_version_initial_check}, Client expected: {expected_version}")

                if current_db_version_initial_check != expected_version:
                    raise ConflictError(
                        f"Update failed: version mismatch (db has {current_db_version_initial_check}, client expected {expected_version}) for character_cards ID {character_id}.",
                        entity="character_cards", entity_id=character_id
                    )

                set_clauses_sql = []
                params_for_set_clause = []
                fields_updated_log = []  # For logging which fields from payload were processed

                # Define fields that can be directly updated and JSON fields
                updatable_direct_fields = [
                    "name", "description", "personality", "scenario", "image",
                    "post_history_instructions", "first_message", "message_example",
                    "creator_notes", "system_prompt", "creator", "character_version"
                ]
                # self._CHARACTER_CARD_JSON_FIELDS is already defined in your class

                for key, value in card_data.items():
                    if key in self._CHARACTER_CARD_JSON_FIELDS:
                        set_clauses_sql.append(f"{key} = ?")
                        params_for_set_clause.append(self._ensure_json_string(value))
                        fields_updated_log.append(key)
                    elif key in updatable_direct_fields:
                        set_clauses_sql.append(f"{key} = ?")
                        params_for_set_clause.append(value)
                        fields_updated_log.append(key)
                    elif key not in ['id', 'created_at', 'last_modified', 'version', 'client_id', 'deleted']:
                        # Log if a key in card_data is not recognized as updatable, but don't error.
                        # This matches the original sequential strategy's behavior of skipping unknown fields.
                        logger.warning(
                            f"Skipping unknown or non-updatable field '{key}' in update_character_card payload.")

                # If expected_version check passed, we always update metadata (last_modified, version, client_id),
                # effectively "touching" the record and bumping its version, even if fields_updated_log is empty
                # (meaning card_data might have contained only non-updatable fields like 'id', or only unknown fields).
                next_version_val = expected_version + 1

                # Add metadata fields to be updated
                set_clauses_sql.extend(["last_modified = ?", "version = ?", "client_id = ?"])
                params_for_set_clause.extend([now, next_version_val, self.client_id])

                # Construct the final query
                # The set_clauses_sql will always have at least the metadata updates if this point is reached.
                final_update_query = f"UPDATE character_cards SET {', '.join(set_clauses_sql)} WHERE id = ? AND version = ? AND deleted = 0"

                # WHERE clause parameters
                where_params = [character_id, expected_version]
                final_params = tuple(params_for_set_clause + where_params)

                logger.debug(f"Executing SINGLE character update query: {final_update_query}")
                logger.debug(f"Params: {final_params}")

                cursor = conn.execute(final_update_query, final_params)
                logger.debug(f"Character Update executed, rowcount: {cursor.rowcount}")

                if cursor.rowcount == 0:
                    # This could happen if a concurrent modification occurred between the initial version check and this UPDATE SQL.
                    # Re-check the record's state to provide a more specific error.
                    check_again_cursor = conn.execute("SELECT version, deleted FROM character_cards WHERE id = ?",
                                                      (character_id,))
                    final_state = check_again_cursor.fetchone()

                    if not final_state:
                        msg = f"Character card ID {character_id} disappeared before update completion (expected v{expected_version})."
                    elif final_state['deleted']:
                        msg = f"Character card ID {character_id} was soft-deleted concurrently (expected v{expected_version} for update)."
                    elif final_state[
                        'version'] != expected_version:  # Version changed from what we expected for the WHERE clause
                        msg = f"Character card ID {character_id} version changed to {final_state['version']} concurrently (expected v{expected_version} for update's WHERE clause)."
                    else:  # This case implies the record was found with the correct version and not deleted, yet rowcount was 0. Unlikely.
                        msg = f"Update for character card ID {character_id} (expected v{expected_version}) affected 0 rows for an unknown reason after passing initial checks."
                    raise ConflictError(msg, entity="character_cards", entity_id=character_id)

                log_msg_fields_updated = f"Fields from payload processed: {fields_updated_log if fields_updated_log else 'None'}."
                logger.info(
                    f"Updated character card ID {character_id} (SINGLE UPDATE) from client-expected version {expected_version} to final DB version {next_version_val}. {log_msg_fields_updated}")
                return True

        except sqlite3.DatabaseError as e:  # Catching this specifically for robustness
            logger.critical(f"DATABASE ERROR during update_character_card (SINGLE UPDATE STRATEGY)!")
            logger.critical(f"Error details: {str(e)}")
            raise CharactersRAGDBError(f"Database error during single update: {e}") from e
        except ConflictError:  # Re-raise ConflictErrors from _get_current_db_version or manual checks
            logger.warning(f"ConflictError during update_character_card for ID {character_id}.",
                           exc_info=False)  # exc_info=True if needed
            raise
        except InputError:  # Should not happen if initial `if not card_data:` check is there.
            logger.warning(f"InputError during update_character_card for ID {character_id}.", exc_info=False)
            raise
        except Exception as e:  # Catch any other unexpected Python errors
            logger.error(
                f"Unexpected Python error in update_character_card (SINGLE UPDATE STRATEGY) for ID {character_id}: {e}",
                exc_info=True)
            raise CharactersRAGDBError(f"Unexpected error updating character card: {e}") from e

    def soft_delete_character_card(self, character_id: int, expected_version: int) -> bool:
        now = self._get_current_utc_timestamp_iso()
        next_version_val = expected_version + 1

        query = "UPDATE character_cards SET deleted = 1, last_modified = ?, version = ?, client_id = ? WHERE id = ? AND version = ? AND deleted = 0"
        params = (now, next_version_val, self.client_id, character_id, expected_version)

        try:
            with self.transaction() as conn:
                try:
                    current_db_version = self._get_current_db_version(conn, "character_cards", "id", character_id)
                    # If here, record is active.
                except ConflictError as e:
                    # Check if ConflictError from _get_current_db_version was because it's ALREADY soft-deleted.
                    check_status_cursor = conn.execute("SELECT deleted, version FROM character_cards WHERE id = ?",
                                                       (character_id,))
                    record_status = check_status_cursor.fetchone()
                    if record_status and record_status['deleted']:
                        logger.info(
                            f"Character card ID {character_id} already soft-deleted. Soft delete successful (idempotent).")
                        return True
                    # If not found, or some other conflict, re-raise.
                    raise e

                if current_db_version != expected_version:
                    raise ConflictError(
                        f"Soft delete for Character ID {character_id} failed: version mismatch (db has {current_db_version}, client expected {expected_version}).",
                        entity="character_cards", entity_id=character_id
                    )

                cursor = conn.execute(query, params)

                if cursor.rowcount == 0:
                    # Race condition: Record changed between pre-check and UPDATE.
                    check_again_cursor = conn.execute("SELECT version, deleted FROM character_cards WHERE id = ?",
                                                      (character_id,))
                    final_state = check_again_cursor.fetchone()

                    if not final_state:
                        msg = f"Character card ID {character_id} disappeared before soft delete (expected active version {expected_version})."
                    elif final_state['deleted']:
                        # If it got deleted by another process. Consider this success if the state is 'deleted'.
                        logger.info(
                            f"Character card ID {character_id} was soft-deleted concurrently to version {final_state['version']}. Soft delete successful.")
                        return True
                    elif final_state['version'] != expected_version:  # Still active but version changed
                        msg = f"Soft delete for Character ID {character_id} failed: version changed to {final_state['version']} concurrently (expected {expected_version})."
                    else:
                        msg = f"Soft delete for Character ID {character_id} (expected version {expected_version}) affected 0 rows for an unknown reason after passing initial checks."
                    raise ConflictError(msg, entity="character_cards", entity_id=character_id)

                logger.info(
                    f"Soft-deleted character card ID {character_id} (was version {expected_version}), new version {next_version_val}.")
                return True
        except ConflictError:
            raise
        except CharactersRAGDBError as e:  # Catches sqlite3.Error from conn.execute
            logger.error(
                f"Database error soft-deleting character card ID {character_id} (expected v{expected_version}): {e}",
                exc_info=True)
            raise

    def search_character_cards(self, search_term: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Searches character_cards_fts. Returns full card details for matches."""
        # FTS query matches against name, description, personality, scenario, system_prompt
        # FTS table columns: name, description, personality, scenario, system_prompt
        # Content table: character_cards, content_rowid: id
        query = """
                SELECT cc.*
                FROM character_cards_fts fts
                         JOIN character_cards cc ON fts.rowid = cc.id
                WHERE fts.character_cards_fts MATCH ? \
                  AND cc.deleted = 0
                ORDER BY rank LIMIT ? \
                """
        try:
            cursor = self.execute_query(query, (search_term, limit))
            rows = cursor.fetchall()
            return [self._deserialize_row_fields(row, self._CHARACTER_CARD_JSON_FIELDS) for row in rows if row]
        except CharactersRAGDBError as e:
            logger.error(f"Error searching character cards for '{search_term}': {e}")
            raise

    # --- Conversation Methods ---
    def add_conversation(self, conv_data: Dict[str, Any]) -> Optional[str]:
        """
        Adds a new conversation.
        `id` and `root_id` must be provided (UUIDs).
        If `root_id` is not provided, `id` is used as `root_id`.
        """
        conv_id = conv_data.get('id') or self._generate_uuid()
        root_id = conv_data.get('root_id') or conv_id  # If root_id not given, this is a new root.

        if 'character_id' not in conv_data:
            raise InputError("Required field 'character_id' is missing for conversation.")

        client_id = conv_data.get('client_id') or self.client_id
        if not client_id:
            raise InputError("Client ID is required for conversation.")

        now = self._get_current_utc_timestamp_iso()
        query = """
                INSERT INTO conversations (id, root_id, forked_from_message_id, parent_conversation_id, \
                                           character_id, title, rating, \
                                           last_modified, client_id, version, deleted) \
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 0) \
                """
        params = (
            conv_id, root_id, conv_data.get('forked_from_message_id'),
            conv_data.get('parent_conversation_id'), conv_data['character_id'],
            conv_data.get('title'), conv_data.get('rating'),
            now, client_id
        )
        try:
            with self.transaction() as conn:
                conn.execute(query, params) # Use conn.execute
            logger.info(f"Added conversation ID: {conv_id}.")
            return conv_id
        except sqlite3.IntegrityError as e: # Catch specific error if PK (id) is duplicated
            if "UNIQUE constraint failed: conversations.id" in str(e):
                 raise ConflictError(f"Conversation with ID '{conv_id}' already exists.", entity="conversations", entity_id=conv_id) from e
            raise CharactersRAGDBError(f"Database integrity error adding conversation: {e}") from e
        except CharactersRAGDBError as e:
            logger.error(f"Database error adding conversation: {e}")
            raise

    def get_conversation_by_id(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        query = "SELECT * FROM conversations WHERE id = ? AND deleted = 0"
        try:
            cursor = self.execute_query(query, (conversation_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
        except CharactersRAGDBError as e:
            logger.error(f"Database error fetching conversation ID {conversation_id}: {e}")
            raise

    def get_conversations_for_character(self, character_id: int, limit: int = 50, offset: int = 0) -> List[
        Dict[str, Any]]:
        query = "SELECT * FROM conversations WHERE character_id = ? AND deleted = 0 ORDER BY last_modified DESC LIMIT ? OFFSET ?"
        try:
            cursor = self.execute_query(query, (character_id, limit, offset))
            return [dict(row) for row in cursor.fetchall()]
        except CharactersRAGDBError as e:
            logger.error(f"Database error fetching conversations for character ID {character_id}: {e}")
            raise

    def update_conversation(self, conversation_id: str, update_data: Dict[str, Any], expected_version: int) -> bool:
        logger.debug(
            f"Starting update_conversation for ID {conversation_id}, expected_version {expected_version} (FTS handled by DB triggers)")

        if not update_data:
            # If update_data is empty, it implies only metadata (last_modified, version) might be updated.
            # The logic below will handle this by not finding specific fields and then constructing
            # an update query that only touches last_modified, version, and client_id.
            logger.info(
                f"No specific fields in update_data for conversation {conversation_id}. Will update metadata if version matches.")
            # Allow processing to continue to the transaction block.
            # The update query construction will adapt.

        now = self._get_current_utc_timestamp_iso()

        try:
            with self.transaction() as conn:
                logger.debug(f"Conversation update transaction started. Connection object: {id(conn)}")

                # Fetch current state, including rowid (though not used for manual FTS, it's good practice to fetch if available)
                # and current title for potential non-FTS related "title_changed" logic.
                cursor_check = conn.execute("SELECT rowid, title, version, deleted FROM conversations WHERE id = ?",
                                            (conversation_id,))
                current_state = cursor_check.fetchone()

                if not current_state:
                    raise ConflictError(f"Conversation ID {conversation_id} not found for update.",
                                        entity="conversations", entity_id=conversation_id)
                if current_state['deleted']:
                    raise ConflictError(f"Conversation ID {conversation_id} is deleted, cannot update.",
                                        entity="conversations", entity_id=conversation_id)

                current_db_version = current_state['version']
                current_title = current_state['title']  # For logging or other conditional logic if title changed

                logger.debug(
                    f"Conversation current DB version: {current_db_version}, Expected by client: {expected_version}, Current title: {current_title}")

                if current_db_version != expected_version:
                    raise ConflictError(
                        f"Conversation ID {conversation_id} update failed: version mismatch (db has {current_db_version}, client expected {expected_version}).",
                        entity="conversations", entity_id=conversation_id
                    )

                fields_to_update_sql = []
                params_for_set_clause = []
                title_changed_flag = False  # Flag to indicate if title was among the updated fields and changed value

                # Process 'title' if present in update_data
                if 'title' in update_data:
                    fields_to_update_sql.append("title = ?")
                    params_for_set_clause.append(update_data['title'])
                    if update_data['title'] != current_title:
                        title_changed_flag = True

                # Process 'rating' if present in update_data
                if 'rating' in update_data:
                    fields_to_update_sql.append("rating = ?")
                    params_for_set_clause.append(update_data['rating'])

                # Add other updatable fields from update_data here if needed in the future
                # Example:
                # if 'some_other_field' in update_data:
                #     fields_to_update_sql.append("some_other_field = ?")
                #     params_for_set_clause.append(update_data['some_other_field'])

                next_version_val = expected_version + 1  # Version always increments on successful update

                if not fields_to_update_sql:
                    # This block executes if update_data was empty or contained no recognized updatable fields.
                    # We still need to update last_modified, version, and client_id due to the successful version check.
                    logger.info(
                        f"No specific updatable fields (e.g. title, rating) found for conversation {conversation_id}. Updating metadata only.")
                    main_update_query = "UPDATE conversations SET last_modified = ?, version = ?, client_id = ? WHERE id = ? AND version = ? AND deleted = 0"
                    main_update_params = (now, next_version_val, self.client_id, conversation_id, expected_version)
                else:
                    # If specific fields were found, add metadata fields to the update
                    fields_to_update_sql.extend(["last_modified = ?", "version = ?", "client_id = ?"])

                    final_set_values = params_for_set_clause[:]  # Copy of values for specific fields
                    final_set_values.extend([now, next_version_val, self.client_id])  # Add values for metadata fields

                    main_update_query = f"UPDATE conversations SET {', '.join(fields_to_update_sql)} WHERE id = ? AND version = ? AND deleted = 0"
                    main_update_params = tuple(final_set_values + [conversation_id, expected_version])

                logger.debug(f"Executing MAIN conversation update query: {main_update_query}")
                logger.debug(f"Params: {main_update_params}")

                cursor_main = conn.execute(main_update_query, main_update_params)
                logger.debug(f"Main Conversation Update executed, rowcount: {cursor_main.rowcount}")

                if cursor_main.rowcount == 0:
                    # This could happen if a concurrent modification occurred between the version check and this UPDATE.
                    # Or if the record was deleted concurrently.
                    # Re-check the state to provide a more accurate error.
                    check_again_cursor = conn.execute("SELECT version, deleted FROM conversations WHERE id = ?",
                                                      (conversation_id,))
                    final_state = check_again_cursor.fetchone()
                    if not final_state:
                        msg = f"Conversation ID {conversation_id} disappeared before update completion (expected v{expected_version})."
                    elif final_state['deleted']:
                        msg = f"Conversation ID {conversation_id} was soft-deleted concurrently (expected v{expected_version} for update)."
                    elif final_state['version'] != expected_version:
                        msg = f"Conversation ID {conversation_id} version changed to {final_state['version']} concurrently (expected v{expected_version} for update)."
                    else:  # Should not happen if rowcount is 0 and version check was successful.
                        msg = f"Main update for conversation ID {conversation_id} (expected v{expected_version}) affected 0 rows for an unknown reason after passing initial checks."
                    raise ConflictError(msg, entity="conversations", entity_id=conversation_id)

                # FTS synchronization is handled by database triggers.
                # No manual FTS DML (DELETE/INSERT on conversations_fts) is performed here.

                logger.info(
                    f"Updated conversation ID {conversation_id} from version {expected_version} to version {next_version_val} (FTS handled by DB triggers). Title changed: {title_changed_flag}")
                return True

        except sqlite3.DatabaseError as e:
            # This broad catch is for unexpected SQLite errors, including potential "malformed" if it still occurs.
            logger.critical(f"DATABASE ERROR during update_conversation (FTS handled by DB triggers): {e}")
            logger.critical(f"Error details: {str(e)}")
            # Specific handling for "malformed" can be added if needed, but the goal is to prevent it.
            raise CharactersRAGDBError(f"Database error during update_conversation: {e}") from e
        except ConflictError:  # Re-raise ConflictErrors for tests or callers to handle
            raise
        except CharactersRAGDBError as e:  # Other custom DB errors
            logger.error(f"Application-level database error in update_conversation for ID {conversation_id}: {e}",
                         exc_info=True)
            raise
        except Exception as e:  # Catch-all for any other unexpected Python errors
            logger.error(f"Unexpected Python error in update_conversation for ID {conversation_id}: {e}", exc_info=True)
            raise CharactersRAGDBError(f"Unexpected error during update_conversation: {e}") from e

    def soft_delete_conversation(self, conversation_id: str, expected_version: int) -> bool:
        now = self._get_current_utc_timestamp_iso()
        next_version_val = expected_version + 1

        query = "UPDATE conversations SET deleted = 1, last_modified = ?, version = ?, client_id = ? WHERE id = ? AND version = ? AND deleted = 0"
        params = (now, next_version_val, self.client_id, conversation_id, expected_version)

        try:
            with self.transaction() as conn:
                try:
                    current_db_version = self._get_current_db_version(conn, "conversations", "id", conversation_id)
                except ConflictError as e:
                    check_status_cursor = conn.execute("SELECT deleted, version FROM conversations WHERE id = ?",
                                                       (conversation_id,))
                    record_status = check_status_cursor.fetchone()
                    if record_status and record_status['deleted']:
                        logger.info(f"Conversation ID {conversation_id} already soft-deleted. Success (idempotent).")
                        return True
                    raise e

                if current_db_version != expected_version:
                    raise ConflictError(
                        f"Soft delete for Conversation ID {conversation_id} failed: version mismatch (db has {current_db_version}, client expected {expected_version}).",
                        entity="conversations", entity_id=conversation_id
                    )

                cursor = conn.execute(query, params)

                if cursor.rowcount == 0:
                    check_again_cursor = conn.execute("SELECT version, deleted FROM conversations WHERE id = ?",
                                                      (conversation_id,))
                    final_state = check_again_cursor.fetchone()
                    if not final_state:
                        msg = f"Conversation ID {conversation_id} disappeared."
                    elif final_state['deleted']:
                        logger.info(f"Conversation ID {conversation_id} was soft-deleted concurrently. Success.")
                        return True
                    elif final_state['version'] != expected_version:
                        msg = f"Conversation ID {conversation_id} version changed to {final_state['version']} concurrently."
                    else:
                        msg = f"Soft delete for conversation ID {conversation_id} (expected v{expected_version}) affected 0 rows."
                    raise ConflictError(msg, entity="conversations", entity_id=conversation_id)

                logger.info(
                    f"Soft-deleted conversation ID {conversation_id} (was v{expected_version}), new version {next_version_val}.")
                return True
        except ConflictError:
            raise
        except CharactersRAGDBError as e:
            logger.error(
                f"Database error soft-deleting conversation ID {conversation_id} (expected v{expected_version}): {e}",
                exc_info=True)
            raise

    def search_conversations_by_title(self, title_query: str, character_id: Optional[int] = None, limit: int = 10) -> \
            List[Dict[str, Any]]:
        """Searches conversations_fts (title). Corrected JOIN condition."""
        # FTS table: conversations_fts, content_rowid: rowid (maps to conversations.rowid)
        # Content table: conversations, PK is 'id' (TEXT)
        base_query = """
                     SELECT c.*
                     FROM conversations_fts fts
                              JOIN conversations c ON fts.rowid = c.rowid -- Corrected Join condition
                     WHERE fts.conversations_fts MATCH ? \
                       AND c.deleted = 0 \
                     """
        params_list: List[Any] = [title_query]
        if character_id is not None:
            base_query += " AND c.character_id = ?"
            params_list.append(character_id)

        base_query += " ORDER BY rank LIMIT ?"
        params_list.append(limit)

        try:
            cursor = self.execute_query(base_query, tuple(params_list))
            return [dict(row) for row in cursor.fetchall()]
        except CharactersRAGDBError as e:
            logger.error(f"Error searching conversations for title '{title_query}': {e}")
            raise

    # --- Message Methods ---
    def add_message(self, msg_data: Dict[str, Any]) -> Optional[str]:
        """Adds a new message to a conversation."""
        msg_id = msg_data.get('id') or self._generate_uuid()

        required_fields = ['conversation_id', 'sender', 'content']
        for field in required_fields:
            if field not in msg_data or not msg_data[field]:  # Ensure content is not empty
                raise InputError(f"Required field '{field}' is missing or empty for message.")

        client_id = msg_data.get('client_id') or self.client_id
        if not client_id:
            raise InputError("Client ID is required for message.")

        now = self._get_current_utc_timestamp_iso()
        # Timestamp is the creation time, last_modified tracks actual modifications.
        # For a new message, timestamp and last_modified will be the same initially.
        timestamp = msg_data.get('timestamp') or now

        query = """
                INSERT INTO messages (id, conversation_id, parent_message_id, sender, content, timestamp, \
                                      ranking, last_modified, client_id, version, deleted) \
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 0) \
                """
        params = (
            msg_id, msg_data['conversation_id'], msg_data.get('parent_message_id'),
            msg_data['sender'], msg_data['content'], timestamp,
            msg_data.get('ranking'), now, client_id
        )
        try:
            with self.transaction():
                # Ensure conversation exists and is not deleted
                conv_cursor = self.execute_query("SELECT 1 FROM conversations WHERE id = ? AND deleted = 0",
                                                 (msg_data['conversation_id'],))
                if not conv_cursor.fetchone():
                    raise InputError(
                        f"Cannot add message: Conversation ID '{msg_data['conversation_id']}' not found or deleted.")

                self.execute_query(query, params)
            logger.info(f"Added message ID: {msg_id} to conversation {msg_data['conversation_id']}.")
            return msg_id
        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed: messages.id" in str(e):
                 raise ConflictError(f"Message with ID '{msg_id}' already exists.", entity="messages", entity_id=msg_id) from e
            # Could also be foreign key constraint on conversation_id if that check was somehow bypassed
            # or if ON DELETE CASCADE fired elsewhere unexpectedly (unlikely for add).
            raise CharactersRAGDBError(f"Database integrity error adding message: {e}") from e
        except InputError: # From conversation check
            raise
        except CharactersRAGDBError as e:
            logger.error(f"Database error adding message: {e}")
            raise

    def get_message_by_id(self, message_id: str) -> Optional[Dict[str, Any]]:
        query = "SELECT * FROM messages WHERE id = ? AND deleted = 0"
        try:
            cursor = self.execute_query(query, (message_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
        except CharactersRAGDBError as e:
            logger.error(f"Database error fetching message ID {message_id}: {e}")
            raise

    def get_messages_for_conversation(self, conversation_id: str, limit: int = 100, offset: int = 0,
                                      order_by_timestamp: str = "ASC") -> List[Dict[str, Any]]:
        if order_by_timestamp.upper() not in ["ASC", "DESC"]:
            raise InputError("order_by_timestamp must be 'ASC' or 'DESC'.")
        query = f"SELECT * FROM messages WHERE conversation_id = ? AND deleted = 0 ORDER BY timestamp {order_by_timestamp} LIMIT ? OFFSET ?"
        try:
            cursor = self.execute_query(query, (conversation_id, limit, offset))
            return [dict(row) for row in cursor.fetchall()]
        except CharactersRAGDBError as e:
            logger.error(f"Database error fetching messages for conversation ID {conversation_id}: {e}")
            raise

    def update_message(self, message_id: str, update_data: Dict[str, Any], expected_version: int) -> bool:
        if not update_data:
            raise InputError("No data provided for message update.")

        now = self._get_current_utc_timestamp_iso()
        fields_to_update_sql = []
        params_for_set_clause = []

        allowed_to_update = ['content', 'ranking', 'parent_message_id']
        for key, value in update_data.items():
            if key in allowed_to_update:
                fields_to_update_sql.append(f"{key} = ?")
                params_for_set_clause.append(value)
            elif key not in ['id', 'conversation_id', 'sender', 'timestamp', 'last_modified', 'version', 'client_id',
                             'deleted']:
                logger.warning(
                    f"Attempted to update immutable or unknown field '{key}' in message ID {message_id}, skipping.")

        if not fields_to_update_sql:
            logger.info(f"No updatable fields provided for message ID {message_id}.")
            return True

        next_version_val = expected_version + 1
        fields_to_update_sql.extend(["last_modified = ?", "version = ?", "client_id = ?"])

        all_set_values = params_for_set_clause[:]
        all_set_values.extend([now, next_version_val, self.client_id])

        where_values = [message_id, expected_version]
        final_params_for_execute = tuple(all_set_values + where_values)

        query = f"UPDATE messages SET {', '.join(fields_to_update_sql)} WHERE id = ? AND version = ? AND deleted = 0"

        try:
            with self.transaction() as conn:
                current_db_version = self._get_current_db_version(conn, "messages", "id", message_id)

                if current_db_version != expected_version:
                    raise ConflictError(
                        f"Message ID {message_id} update failed: version mismatch (db has {current_db_version}, client expected {expected_version}).",
                        entity="messages", entity_id=message_id
                    )

                cursor = conn.execute(query, final_params_for_execute)

                if cursor.rowcount == 0:
                    check_again_cursor = conn.execute("SELECT version, deleted FROM messages WHERE id = ?",
                                                      (message_id,))
                    final_state = check_again_cursor.fetchone()
                    if not final_state:
                        msg = f"Message ID {message_id} disappeared."
                    elif final_state['deleted']:
                        msg = f"Message ID {message_id} was soft-deleted concurrently."
                    elif final_state['version'] != expected_version:
                        msg = f"Message ID {message_id} version changed to {final_state['version']} concurrently."
                    else:
                        msg = f"Update for message ID {message_id} (expected v{expected_version}) affected 0 rows."
                    raise ConflictError(msg, entity="messages", entity_id=message_id)

                logger.info(
                    f"Updated message ID {message_id} from version {expected_version} to version {next_version_val}.")
                return True
        except sqlite3.IntegrityError as e:  # e.g. parent_message_id FK violation
            logger.error(f"SQLite integrity error updating message ID {message_id} (expected v{expected_version}): {e}",
                         exc_info=True)
            raise CharactersRAGDBError(f"Database integrity error updating message: {e}") from e
        except ConflictError:
            raise
        except CharactersRAGDBError as e:
            logger.error(f"Database error updating message ID {message_id} (expected v{expected_version}): {e}",
                         exc_info=True)
            raise

    def soft_delete_message(self, message_id: str, expected_version: int) -> bool:
        now = self._get_current_utc_timestamp_iso()
        next_version_val = expected_version + 1

        query = "UPDATE messages SET deleted = 1, last_modified = ?, version = ?, client_id = ? WHERE id = ? AND version = ? AND deleted = 0"
        params = (now, next_version_val, self.client_id, message_id, expected_version)

        try:
            with self.transaction() as conn:
                try:
                    current_db_version = self._get_current_db_version(conn, "messages", "id", message_id)
                except ConflictError as e:
                    check_status_cursor = conn.execute("SELECT deleted, version FROM messages WHERE id = ?",
                                                       (message_id,))
                    record_status = check_status_cursor.fetchone()
                    if record_status and record_status['deleted']:
                        logger.info(f"Message ID {message_id} already soft-deleted. Success (idempotent).")
                        return True
                    raise e

                if current_db_version != expected_version:
                    raise ConflictError(
                        f"Soft delete for Message ID {message_id} failed: version mismatch (db has {current_db_version}, client expected {expected_version}).",
                        entity="messages", entity_id=message_id
                    )

                cursor = conn.execute(query, params)

                if cursor.rowcount == 0:
                    check_again_cursor = conn.execute("SELECT version, deleted FROM messages WHERE id = ?",
                                                      (message_id,))
                    final_state = check_again_cursor.fetchone()
                    if not final_state:
                        msg = f"Message ID {message_id} disappeared."
                    elif final_state['deleted']:
                        logger.info(f"Message ID {message_id} was soft-deleted concurrently. Success.")
                        return True
                    elif final_state['version'] != expected_version:
                        msg = f"Message ID {message_id} version changed to {final_state['version']} concurrently."
                    else:
                        msg = f"Soft delete for message ID {message_id} (expected v{expected_version}) affected 0 rows."
                    raise ConflictError(msg, entity="messages", entity_id=message_id)

                logger.info(
                    f"Soft-deleted message ID {message_id} (was v{expected_version}), new version {next_version_val}.")
                return True
        except ConflictError:
            raise
        except CharactersRAGDBError as e:
            logger.error(f"Database error soft-deleting message ID {message_id} (expected v{expected_version}): {e}",
                         exc_info=True)
            raise

    def search_messages_by_content(self, content_query: str, conversation_id: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        base_query = """
                     SELECT m.*
                     FROM messages_fts fts
                              JOIN messages m ON fts.rowid = m.rowid
                     WHERE fts.messages_fts MATCH ? \
                       AND m.deleted = 0 \
                     """
        params_list: List[Any] = [content_query]
        if conversation_id:
            base_query += " AND m.conversation_id = ?"
            params_list.append(conversation_id)

        base_query += " ORDER BY rank LIMIT ?"
        params_list.append(limit)

        try:
            cursor = self.execute_query(base_query, tuple(params_list))
            return [dict(row) for row in cursor.fetchall()]
        except CharactersRAGDBError as e:
            logger.error(f"Error searching messages for content '{content_query}': {e}")
            raise

    # --- Keyword, KeywordCollection, Note Methods (CRUD + Search) ---
    # These follow similar patterns to CharacterCard, Conversation, Message methods:
    # - add: INSERT with default version 1, set last_modified, client_id.
    # - get_by_id/name: SELECT WHERE deleted = 0.
    # - list: SELECT WHERE deleted = 0.
    # - update: UPDATE SET fields, last_modified, version, client_id WHERE id/name = ? AND version = ? AND deleted = 0.
    # - soft_delete: UPDATE SET deleted = 1, last_modified, version, client_id WHERE id/name = ? AND version = ? AND deleted = 0.
    # - search: Use respective FTS table.

    def _add_generic_item(self, table_name: str, unique_col_name: str, item_data: Dict[str, Any], main_col_value: str,
                          other_fields_map: Dict[str, str]) -> Optional[int]:
        """Helper for adding keywords, collections, notes - tables with autoinc ID and a unique text column."""
        now = self._get_current_utc_timestamp_iso()
        client_id_to_use = item_data.get('client_id', self.client_id)

        other_cols = list(other_fields_map.keys())
        other_placeholders_list = ['?'] * len(other_cols)
        other_values = [item_data.get(other_fields_map[col_db]) for col_db in other_cols]

        cols_str_list = [unique_col_name]
        placeholders_str_list = ["?"]
        if other_cols:
            cols_str_list.extend(other_cols)
            placeholders_str_list.extend(other_placeholders_list)

        cols_str = ", ".join(cols_str_list)
        placeholders_str = ", ".join(placeholders_str_list)

        query = f"""
            INSERT INTO {table_name} (
                {cols_str}, last_modified, client_id, version, deleted
            ) VALUES ({placeholders_str}, ?, ?, 1, 0)
        """
        params_tuple = tuple([main_col_value] + other_values + [now, client_id_to_use])

        try:
            with self.transaction() as conn:
                # Check if a soft-deleted item exists and undelete it
                undelete_cursor = conn.execute(
                    f"SELECT id, version FROM {table_name} WHERE {unique_col_name} = ? AND deleted = 1",
                    (main_col_value,))
                existing_deleted = undelete_cursor.fetchone()
                if existing_deleted:
                    item_id, current_version = existing_deleted['id'], existing_deleted['version']
                    next_version = current_version + 1

                    update_set_parts = [f"{unique_col_name} = ?"]
                    update_params_list = [main_col_value]
                    for i, col_db in enumerate(other_cols):
                        update_set_parts.append(f"{col_db} = ?")
                        update_params_list.append(other_values[i])
                    update_set_parts.extend(["deleted = 0", "last_modified = ?", "version = ?", "client_id = ?"])
                    update_params_list.extend([now, next_version, client_id_to_use, item_id, current_version])

                    undelete_query = f"UPDATE {table_name} SET {', '.join(update_set_parts)} WHERE id = ? AND version = ?"

                    row_count = conn.execute(undelete_query, tuple(update_params_list)).rowcount
                    if row_count == 0:
                        raise ConflictError(
                            f"Failed to undelete {table_name} '{main_col_value}' due to version mismatch or it became active.",
                            entity=table_name, entity_id=main_col_value)
                    logger.info(
                        f"Undeleted and updated {table_name} '{main_col_value}' with ID: {item_id}, new version {next_version}.")
                    return item_id

                cursor = conn.execute(query, params_tuple)
                item_id = cursor.lastrowid
                logger.info(f"Added {table_name} '{main_col_value}' with ID: {item_id}.")
                return item_id
        except sqlite3.IntegrityError as e:
             if f"UNIQUE constraint failed: {table_name}.{unique_col_name}" in str(e):
                logger.warning(f"{table_name} with {unique_col_name} '{main_col_value}' already exists and is active.")
                raise ConflictError(f"{table_name} '{main_col_value}' already exists and is active.", entity=table_name,
                                    entity_id=main_col_value) from e
             raise CharactersRAGDBError(f"Database integrity error adding {table_name}: {e}") from e
        except ConflictError: # From undelete path
            raise
        except CharactersRAGDBError as e:
            logger.error(f"Database error adding {table_name} '{main_col_value}': {e}")
            raise
        return None  # Should not be reached if exceptions are raised properly

    def _get_generic_item_by_id(self, table_name: str, item_id: int) -> Optional[Dict[str, Any]]:
        query = f"SELECT * FROM {table_name} WHERE id = ? AND deleted = 0"
        try:
            cursor = self.execute_query(query, (item_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
        except CharactersRAGDBError as e:
            logger.error(f"Database error fetching {table_name} ID {item_id}: {e}")
            raise

    def _get_generic_item_by_unique_text(self, table_name: str, unique_col_name: str, value: str) -> Optional[Dict[str, Any]]:
        query = f"SELECT * FROM {table_name} WHERE {unique_col_name} = ? AND deleted = 0"
        try:
            cursor = self.execute_query(query, (value,))
            row = cursor.fetchone()
            return dict(row) if row else None
        except CharactersRAGDBError as e:
            logger.error(f"Database error fetching {table_name} by {unique_col_name} '{value}': {e}")
            raise

    def _list_generic_items(self, table_name: str, order_by_col: str, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        query = f"SELECT * FROM {table_name} WHERE deleted = 0 ORDER BY {order_by_col} LIMIT ? OFFSET ?"
        try:
            cursor = self.execute_query(query, (limit, offset))
            return [dict(row) for row in cursor.fetchall()]
        except CharactersRAGDBError as e:
            logger.error(f"Database error listing {table_name}: {e}")
            raise

    def _update_generic_item(self, table_name: str, item_id: Union[int, str],
                             update_data: Dict[str, Any], expected_version: int,
                             allowed_fields: List[str], pk_col_name: str = "id",
                             unique_col_name_in_data: Optional[str] = None) -> bool:
        if not update_data:
            raise InputError(f"No data provided for update of {table_name} ID {item_id}.")

        now = self._get_current_utc_timestamp_iso()
        fields_to_update_sql = []
        params_for_set_clause = []

        for key, value in update_data.items():
            if key in allowed_fields:
                fields_to_update_sql.append(f"{key} = ?")
                # Special handling for specific field types if necessary, e.g., stripping title
                if table_name == "notes" and key == "title" and isinstance(value, str):
                    params_for_set_clause.append(value.strip())
                else:
                    params_for_set_clause.append(value)
            elif key not in [pk_col_name, 'created_at', 'last_modified', 'version', 'client_id', 'deleted']:
                logger.warning(
                    f"Attempted to update immutable or unknown field '{key}' in {table_name} ID {item_id}, skipping.")

        if not fields_to_update_sql:
            logger.info(f"No updatable fields provided for {table_name} ID {item_id}.")
            return True  # Or False, depending on desired behavior for no-op update

        next_version_val = expected_version + 1
        fields_to_update_sql.extend(["last_modified = ?", "version = ?", "client_id = ?"])

        # Values for the SET clause
        final_set_values = params_for_set_clause[:]
        final_set_values.extend([now, next_version_val, self.client_id])

        # Values for the WHERE clause
        where_clause_values = [item_id, expected_version]

        final_query_params = tuple(final_set_values + where_clause_values)

        query = f"UPDATE {table_name} SET {', '.join(fields_to_update_sql)} WHERE {pk_col_name} = ? AND version = ? AND deleted = 0"

        try:
            with self.transaction() as conn:
                # Explicit pre-check. _get_current_db_version raises ConflictError if not found or soft-deleted.
                current_db_version = self._get_current_db_version(conn, table_name, pk_col_name, item_id)

                if current_db_version != expected_version:
                    raise ConflictError(
                        f"{table_name} ID {item_id} was modified by another client (version mismatch: db has {current_db_version}, client expected {expected_version}).",
                        entity=table_name, entity_id=item_id
                    )

                # If current_db_version == expected_version, proceed with the update.
                cursor = conn.execute(query, final_query_params)

                if cursor.rowcount == 0:
                    # This state implies the record was active with expected_version during the _get_current_db_version check,
                    # but was either deleted or its version changed *just before* the UPDATE SQL executed.
                    check_again_cursor = conn.execute(
                        f"SELECT version, deleted FROM {table_name} WHERE {pk_col_name} = ?", (item_id,))
                    final_state = check_again_cursor.fetchone()

                    if not final_state:
                        raise ConflictError(
                            f"{table_name} ID {item_id} disappeared before update completion (was present at version {expected_version}).",
                            entity=table_name, entity_id=item_id)
                    if final_state['deleted']:
                        raise ConflictError(
                            f"{table_name} ID {item_id} was soft-deleted concurrently while attempting update from version {expected_version}.",
                            entity=table_name, entity_id=item_id)
                    if final_state['version'] != expected_version:  # Version changed from expected
                        raise ConflictError(
                            f"{table_name} ID {item_id} version changed to {final_state['version']} concurrently, expected {expected_version} for update.",
                            entity=table_name, entity_id=item_id)

                    raise ConflictError(
                        f"Update for {table_name} ID {item_id} (expected version {expected_version}) affected 0 rows for an unknown reason after passing initial checks.",
                        entity=table_name, entity_id=item_id)

                logger.info(
                    f"Updated {table_name} ID {item_id} from version {expected_version} to version {next_version_val}.")
                return True
        except sqlite3.IntegrityError as e:
            if unique_col_name_in_data and unique_col_name_in_data in update_data and \
                    f"UNIQUE constraint failed: {table_name}.{unique_col_name_in_data}" in str(
                e).lower():  # Use lower() for broader match
                val = update_data[unique_col_name_in_data]
                logger.warning(
                    f"Update failed for {table_name} ID {item_id}: {unique_col_name_in_data} '{val}' already exists.")
                raise ConflictError(
                    f"Cannot update {table_name} ID {item_id}: {unique_col_name_in_data} '{val}' already exists.",
                    entity=table_name, entity_id=val) from e  # Pass val as entity_id for unique conflict
            logger.error(
                f"SQLite integrity error during update of {table_name} ID {item_id} (expected version {expected_version}): {e}",
                exc_info=True)
            raise CharactersRAGDBError(f"Database integrity error updating {table_name} ({item_id}): {e}") from e
        except ConflictError:
            raise
        except CharactersRAGDBError as e:
            logger.error(
                f"Database error updating {table_name} ID {item_id} (expected version {expected_version}): {e}",
                exc_info=True)
            raise
        # No implicit return None, function should return True or raise.

    def _soft_delete_generic_item(self, table_name: str, item_id: Union[int, str],
                                  expected_version: int, pk_col_name: str = "id") -> bool:
        now = self._get_current_utc_timestamp_iso()
        next_version_val = expected_version + 1

        query = f"UPDATE {table_name} SET deleted = 1, last_modified = ?, version = ?, client_id = ? WHERE {pk_col_name} = ? AND version = ? AND deleted = 0"
        params = (now, next_version_val, self.client_id, item_id, expected_version)

        try:
            with self.transaction() as conn:
                try:
                    current_db_version = self._get_current_db_version(conn, table_name, pk_col_name, item_id)
                    # If we are here, record is active and current_db_version is its version.
                except ConflictError as e:
                    # Check if the ConflictError is because it's already soft-deleted.
                    # Query again to be absolutely sure of the 'deleted' status.
                    check_deleted_cursor = conn.execute(
                        f"SELECT deleted, version FROM {table_name} WHERE {pk_col_name} = ?", (item_id,))
                    record_status = check_deleted_cursor.fetchone()

                    if record_status and record_status['deleted']:
                        logger.info(
                            f"{table_name} ID {item_id} already soft-deleted. Operation considered successful (idempotent).")
                        return True
                    # If it was not found, or some other conflict from _get_current_db_version, re-raise.
                    raise e

                if current_db_version != expected_version:
                    raise ConflictError(
                        f"Soft delete failed for {table_name} ID {item_id}: version mismatch (db has {current_db_version}, client expected {expected_version}).",
                        entity=table_name, entity_id=item_id
                    )

                cursor = conn.execute(query, params)

                if cursor.rowcount == 0:
                    # This means the record (which was active with expected_version) changed state
                    # between the _get_current_db_version check and the UPDATE execution.
                    check_again_cursor = conn.execute(
                        f"SELECT deleted, version FROM {table_name} WHERE {pk_col_name} = ?", (item_id,))
                    changed_record = check_again_cursor.fetchone()

                    if not changed_record:
                        raise ConflictError(
                            f"{table_name} ID {item_id} disappeared before soft-delete completion (expected version {expected_version}).",
                            entity=table_name, entity_id=item_id)

                    if changed_record['deleted']:
                        # If it got deleted by another process, and the new version matches what we intended, it's fine.
                        if changed_record['version'] == next_version_val:
                            logger.info(
                                f"{table_name} ID {item_id} was soft-deleted concurrently to version {next_version_val}. Operation successful.")
                            return True
                        else:
                            raise ConflictError(
                                f"{table_name} ID {item_id} was soft-deleted concurrently to an unexpected version {changed_record['version']} (expected to set to {next_version_val}).",
                                entity=table_name, entity_id=item_id)

                    if changed_record['version'] != expected_version:  # Still active, but version changed
                        raise ConflictError(
                            f"Soft delete failed for {table_name} ID {item_id}: version changed to {changed_record['version']} concurrently (expected {expected_version}).",
                            entity=table_name, entity_id=item_id)

                    raise ConflictError(
                        f"Soft delete for {table_name} ID {item_id} (expected version {expected_version}) affected 0 rows for an unknown reason after passing initial checks.",
                        entity=table_name, entity_id=item_id)

                logger.info(
                    f"Soft-deleted {table_name} ID {item_id} (was version {expected_version}), new version {next_version_val}.")
                return True
        except ConflictError:
            raise
        except CharactersRAGDBError as e:  # Catches sqlite3.Error from conn.execute
            logger.error(
                f"Database error soft-deleting {table_name} ID {item_id} (expected version {expected_version}): {e}",
                exc_info=True)
            raise
        # No implicit return None.

    def _search_generic_items_fts(self, fts_table_name: str, main_table_name: str, fts_match_cols_or_table: str,
                                  search_term: str, limit: int = 10) -> List[Dict[str, Any]]:
        """ Helper for FTS search on simple tables like keywords, notes, collections """
        query = f"""
            SELECT main.*
            FROM {fts_table_name} fts
            JOIN {main_table_name} main ON fts.rowid = main.id -- Assumes main.id is PK and matches fts.rowid (content_rowid)
            WHERE fts.{fts_match_cols_or_table} MATCH ? AND main.deleted = 0
            ORDER BY rank
            LIMIT ?
        """
        try:
            cursor = self.execute_query(query, (search_term, limit))
            return [dict(row) for row in cursor.fetchall()]
        except CharactersRAGDBError as e:
            logger.error(f"Error searching {main_table_name} for '{search_term}': {e}")
            raise

    # Keywords
    def add_keyword(self, keyword_text: str) -> Optional[int]:
        if not keyword_text or not keyword_text.strip():
            raise InputError("Keyword text cannot be empty.")
        return self._add_generic_item("keywords", "keyword", {}, keyword_text.strip(), {})  # No other_fields_map

    def get_keyword_by_id(self, keyword_id: int) -> Optional[Dict[str, Any]]:
        return self._get_generic_item_by_id("keywords", keyword_id)

    def get_keyword_by_text(self, keyword_text: str) -> Optional[Dict[str, Any]]:
        return self._get_generic_item_by_unique_text("keywords", "keyword", keyword_text.strip())

    def list_keywords(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        return self._list_generic_items("keywords", "keyword COLLATE NOCASE", limit, offset) # Order with NOCASE for consistency

    # Keyword text is unique and COLLATE NOCASE, so update is usually for undelete or metadata if any.
    # The _add_generic_item handles undelete. If keyword text itself could change (rare), a specific update method would be needed.
    def soft_delete_keyword(self, keyword_id: int, expected_version: int) -> bool:
        """
        Soft-deletes a keyword with optimistic locking.

        Args:
            keyword_id: The ID of the keyword to soft-delete.
            expected_version: The version number the client expects the record to have.

        Returns:
            True if the soft-delete was successful or if the keyword was already soft-deleted.

        Raises:
            ConflictError: If the record is not found, or if (it's active and)
                           the expected_version does not match the current database version.
            CharactersRAGDBError: For other database-related errors.
        """
        # pk_col_name for 'keywords' is 'id' (default in _soft_delete_generic_item)
        # item_id is int.
        return self._soft_delete_generic_item(
            table_name="keywords",
            item_id=keyword_id,
            expected_version=expected_version,
            pk_col_name="id" # Explicitly pass, though "id" is default
        )

    def search_keywords(self, search_term: str, limit: int = 10) -> List[Dict[str, Any]]:
        return self._search_generic_items_fts("keywords_fts", "keywords", "keyword", search_term, limit)

    # Keyword Collections
    def add_keyword_collection(self, name: str, parent_id: Optional[int] = None) -> Optional[int]:
        if not name or not name.strip():
            raise InputError("Collection name cannot be empty.")
        return self._add_generic_item("keyword_collections", "name", {"parent_id": parent_id}, name.strip(),
                                      {"parent_id": "parent_id"})

    def get_keyword_collection_by_id(self, collection_id: int) -> Optional[Dict[str, Any]]:
        return self._get_generic_item_by_id("keyword_collections", collection_id)

    def get_keyword_collection_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        return self._get_generic_item_by_unique_text("keyword_collections", "name", name.strip())

    def list_keyword_collections(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        return self._list_generic_items("keyword_collections", "name COLLATE NOCASE", limit, offset)

    def update_keyword_collection(self, collection_id: int, update_data: Dict[str, Any], expected_version: int) -> bool:
        """
        Updates a keyword collection with optimistic locking.

        Args:
            collection_id: The ID of the keyword collection to update.
            update_data: A dictionary containing the fields to update (e.g., 'name', 'parent_id').
            expected_version: The version number the client expects the record to have.

        Returns:
            True if the update was successful.

        Raises:
            InputError: If no update data is provided.
            ConflictError: If the record is not found, already soft-deleted,
                           or if the expected_version does not match the current database version.
            CharactersRAGDBError: For other database-related errors.
        """
        # pk_col_name for 'keyword_collections' is 'id' (default in _update_generic_item)
        # item_id is int.
        return self._update_generic_item(
            table_name="keyword_collections",
            item_id=collection_id,
            update_data=update_data,
            expected_version=expected_version,
            allowed_fields=['name', 'parent_id'],
            pk_col_name="id", # Explicitly pass, though "id" is default
            unique_col_name_in_data='name' # For handling unique constraint on name if it's updated
        )

    def soft_delete_keyword_collection(self, collection_id: int, expected_version: int) -> bool:
        """
        Soft-deletes a keyword collection with optimistic locking.

        Args:
            collection_id: The ID of the keyword collection to soft-delete.
            expected_version: The version number the client expects the record to have.

        Returns:
            True if the soft-delete was successful or if the collection was already soft-deleted.

        Raises:
            ConflictError: If the record is not found, or if (it's active and)
                           the expected_version does not match the current database version.
            CharactersRAGDBError: For other database-related errors.
        """
        # pk_col_name for 'keyword_collections' is 'id' (default in _soft_delete_generic_item)
        # item_id is int.
        return self._soft_delete_generic_item(
            table_name="keyword_collections",
            item_id=collection_id,
            expected_version=expected_version,
            pk_col_name="id" # Explicitly pass, though "id" is default
        )

    def search_keyword_collections(self, search_term: str, limit: int = 10) -> List[Dict[str, Any]]:
        return self._search_generic_items_fts("keyword_collections_fts", "keyword_collections", "name", search_term,
                                              limit)

    # Notes (Now with UUID and specific methods)
    def add_note(self, title: str, content: str, note_id: Optional[str] = None) -> str | None:
        if not title or not title.strip():
            raise InputError("Note title cannot be empty.")
        if content is None: # Allow empty string for content
            raise InputError("Note content cannot be None.")

        final_note_id = note_id or self._generate_uuid()
        now = self._get_current_utc_timestamp_iso()
        client_id_to_use = self.client_id # Notes use the instance's client_id directly

        query = """
            INSERT INTO notes (id, title, content, last_modified, client_id, version, deleted, created_at)
            VALUES (?, ?, ?, ?, ?, 1, 0, ?)
        """
        params = (final_note_id, title.strip(), content, now, client_id_to_use, now) # created_at is also now

        try:
            with self.transaction() as conn:
                conn.execute(query, params)
                logger.info(f"Added note '{title.strip()}' with ID: {final_note_id}.")
                return final_note_id
        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed: notes.id" in str(e):
                raise ConflictError(f"Note with ID '{final_note_id}' already exists.", entity="notes", entity_id=final_note_id) from e
            raise CharactersRAGDBError(f"Database integrity error adding note: {e}") from e
        except CharactersRAGDBError as e:
            logger.error(f"Database error adding note '{title.strip()}': {e}")
            raise

    def get_note_by_id(self, note_id: str) -> Optional[Dict[str, Any]]:
        query = "SELECT * FROM notes WHERE id = ? AND deleted = 0"
        cursor = self.execute_query(query, (note_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def list_notes(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        # Using _list_generic_items but ensuring table name and order_by_col are correct for notes
        return self._list_generic_items("notes", "last_modified DESC", limit, offset)

    def update_note(self, note_id: str, update_data: Dict[str, Any], expected_version: int) -> bool:
        if not update_data:
            raise InputError("No data provided for note update.")

        now = self._get_current_utc_timestamp_iso()
        fields_to_update_sql = []
        params_for_set_clause = []

        allowed_to_update = ['title', 'content']
        for key, value in update_data.items():
            if key in allowed_to_update:
                fields_to_update_sql.append(f"{key} = ?")
                # Title might need stripping, content is as-is
                params_for_set_clause.append(value.strip() if key == 'title' and isinstance(value, str) else value)
            elif key not in ['id', 'created_at', 'last_modified', 'version', 'client_id', 'deleted']:
                logger.warning(
                    f"Attempted to update immutable or unknown field '{key}' in note ID {note_id}, skipping.")

        if not fields_to_update_sql:
            logger.info(f"No updatable fields provided for note ID {note_id}.")
            return True

        next_version_val = expected_version + 1
        fields_to_update_sql.extend(["last_modified = ?", "version = ?", "client_id = ?"])

        all_set_values = params_for_set_clause[:]
        all_set_values.extend([now, next_version_val, self.client_id])

        where_values = [note_id, expected_version]
        final_params_for_execute = tuple(all_set_values + where_values)

        query = f"UPDATE notes SET {', '.join(fields_to_update_sql)} WHERE id = ? AND version = ? AND deleted = 0"

        try:
            with self.transaction() as conn:
                current_db_version = self._get_current_db_version(conn, "notes", "id", note_id)

                if current_db_version != expected_version:
                    raise ConflictError(
                        f"Note ID {note_id} update failed: version mismatch (db has {current_db_version}, client expected {expected_version}).",
                        entity="notes", entity_id=note_id
                    )

                cursor = conn.execute(query, final_params_for_execute)

                if cursor.rowcount == 0:
                    check_again_cursor = conn.execute("SELECT version, deleted FROM notes WHERE id = ?", (note_id,))
                    final_state = check_again_cursor.fetchone()
                    if not final_state:
                        msg = f"Note ID {note_id} disappeared."
                    elif final_state['deleted']:
                        msg = f"Note ID {note_id} was soft-deleted concurrently."
                    elif final_state['version'] != expected_version:
                        msg = f"Note ID {note_id} version changed to {final_state['version']} concurrently."
                    else:
                        msg = f"Update for note ID {note_id} (expected v{expected_version}) affected 0 rows."
                    raise ConflictError(msg, entity="notes", entity_id=note_id)

                logger.info(f"Updated note ID {note_id} from version {expected_version} to version {next_version_val}.")
                return True
        # No specific UNIQUE constraint on notes.title or notes.content in the schema, so sqlite3.IntegrityError less likely for these fields.
        except ConflictError:
            raise
        except CharactersRAGDBError as e:  # Catches sqlite3.Error
            logger.error(f"Database error updating note ID {note_id} (expected v{expected_version}): {e}",
                         exc_info=True)
            raise

    def soft_delete_note(self, note_id: str, expected_version: int) -> bool:
        now = self._get_current_utc_timestamp_iso()
        next_version_val = expected_version + 1

        query = "UPDATE notes SET deleted = 1, last_modified = ?, version = ?, client_id = ? WHERE id = ? AND version = ? AND deleted = 0"
        params = (now, next_version_val, self.client_id, note_id, expected_version)

        try:
            with self.transaction() as conn:
                try:
                    current_db_version = self._get_current_db_version(conn, "notes", "id", note_id)
                except ConflictError as e:
                    check_status_cursor = conn.execute("SELECT deleted, version FROM notes WHERE id = ?", (note_id,))
                    record_status = check_status_cursor.fetchone()
                    if record_status and record_status['deleted']:
                        logger.info(f"Note ID {note_id} already soft-deleted. Success (idempotent).")
                        return True
                    raise e

                if current_db_version != expected_version:
                    raise ConflictError(
                        f"Soft delete for Note ID {note_id} failed: version mismatch (db has {current_db_version}, client expected {expected_version}).",
                        entity="notes", entity_id=note_id
                    )

                cursor = conn.execute(query, params)

                if cursor.rowcount == 0:
                    check_again_cursor = conn.execute("SELECT version, deleted FROM notes WHERE id = ?", (note_id,))
                    final_state = check_again_cursor.fetchone()
                    if not final_state:
                        msg = f"Note ID {note_id} disappeared."
                    elif final_state['deleted']:
                        logger.info(f"Note ID {note_id} was soft-deleted concurrently. Success.")
                        return True
                    elif final_state['version'] != expected_version:
                        msg = f"Note ID {note_id} version changed to {final_state['version']} concurrently."
                    else:
                        msg = f"Soft delete for note ID {note_id} (expected v{expected_version}) affected 0 rows."
                    raise ConflictError(msg, entity="notes", entity_id=note_id)

                logger.info(
                    f"Soft-deleted note ID {note_id} (was v{expected_version}), new version {next_version_val}.")
                return True
        except ConflictError:
            raise
        except CharactersRAGDBError as e:
            logger.error(f"Database error soft-deleting note ID {note_id} (expected v{expected_version}): {e}",
                         exc_info=True)
            raise

    def search_notes(self, search_term: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Searches notes_fts (title and content). Corrected JOIN condition."""
        # notes_fts matches against title and content
        # FTS table column group: notes_fts
        # Content table: notes, content_rowid: rowid (maps to notes.rowid)
        query = """
                SELECT main.*
                FROM notes_fts fts
                         JOIN notes main ON fts.rowid = main.rowid -- Corrected Join condition
                WHERE fts.notes_fts MATCH ? \
                  AND main.deleted = 0
                ORDER BY rank LIMIT ? \
                """
        try:
            cursor = self.execute_query(query, (search_term, limit))
            return [dict(row) for row in cursor.fetchall()]
        except CharactersRAGDBError as e:
            logger.error(f"Error searching notes for '{search_term}': {e}")
            raise


    # --- Linking Table Methods (with manual sync_log entries) ---
    def _manage_link(self, link_table: str, col1_name: str, col1_val: Any, col2_name: str, col2_val: Any,
                     operation: str) -> bool:
        """Helper to add ('link') or remove ('unlink') entries from a linking table."""
        now_iso = self._get_current_utc_timestamp_iso()
        sync_payload_dict: Dict[str, Any] = {}
        log_sync_entry = False
        rows_affected = 0

        try:
            with self.transaction() as conn:
                if operation == "link":
                    query = f"INSERT OR IGNORE INTO {link_table} ({col1_name}, {col2_name}, created_at) VALUES (?, ?, ?)"
                    params = (col1_val, col2_val, now_iso)
                    cursor = conn.execute(query, params)
                    rows_affected = cursor.rowcount
                    if rows_affected > 0: # Link was actually created
                        log_sync_entry = True
                        sync_payload_dict = {col1_name: col1_val, col2_name: col2_val, 'created_at': now_iso}
                elif operation == "unlink":
                    query = f"DELETE FROM {link_table} WHERE {col1_name} = ? AND {col2_name} = ?"
                    params = (col1_val, col2_val)
                    cursor = conn.execute(query, params)
                    rows_affected = cursor.rowcount
                    if rows_affected > 0: # Link was actually deleted
                        log_sync_entry = True
                        sync_payload_dict = {col1_name: col1_val, col2_name: col2_val}
                else:
                    raise InputError("Invalid operation for link management.")

                if log_sync_entry:
                    sync_entity_id = f"{col1_val}_{col2_val}"
                    sync_op = 'create' if operation == 'link' else 'delete'
                    sync_timestamp = now_iso # Use now_iso for create, and also for delete event time

                    sync_log_query = """
                        INSERT INTO sync_log (entity, entity_id, operation, timestamp, client_id, version, payload)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """
                    sync_log_params = (
                        link_table, sync_entity_id, sync_op, sync_timestamp,
                        self.client_id, 1, # Link table entries don't have their own version, use 1 for sync log
                        json.dumps(sync_payload_dict)
                    )
                    conn.execute(sync_log_query, sync_log_params)
                    logger.debug(f"Logged sync event for {link_table}: {sync_op} on {sync_entity_id}")

            logger.info(
                f"{operation.capitalize()}ed {link_table}: {col1_name}={col1_val}, {col2_name}={col2_val}. Rows affected: {rows_affected}")
            return rows_affected > 0
        except sqlite3.Error as e: # Catch SQLite specific errors from conn.execute
            logger.error(f"SQLite error during {operation} for {link_table} ({col1_name}={col1_val}, {col2_name}={col2_val}): {e}", exc_info=True)
            raise CharactersRAGDBError(f"Database error during {operation} for {link_table}: {e}") from e
        except CharactersRAGDBError as e: # Catch custom errors like InputError
            logger.error(f"Application error during {operation} for {link_table}: {e}", exc_info=True)
            raise


    # Conversation <-> Keyword
    def link_conversation_to_keyword(self, conversation_id: str, keyword_id: int) -> bool:
        return self._manage_link("conversation_keywords", "conversation_id", conversation_id, "keyword_id", keyword_id,
                                 "link")

    def unlink_conversation_from_keyword(self, conversation_id: str, keyword_id: int) -> bool:
        return self._manage_link("conversation_keywords", "conversation_id", conversation_id, "keyword_id", keyword_id,
                                 "unlink")

    def get_keywords_for_conversation(self, conversation_id: str) -> List[Dict[str, Any]]:
        query = """
                SELECT k.* \
                FROM keywords k \
                         JOIN conversation_keywords ck ON k.id = ck.keyword_id
                WHERE ck.conversation_id = ? \
                  AND k.deleted = 0 \
                ORDER BY k.keyword COLLATE NOCASE
                """
        cursor = self.execute_query(query, (conversation_id,))
        return [dict(row) for row in cursor.fetchall()]

    def get_conversations_for_keyword(self, keyword_id: int, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        query = """
                SELECT c.* \
                FROM conversations c \
                         JOIN conversation_keywords ck ON c.id = ck.conversation_id
                WHERE ck.keyword_id = ? \
                  AND c.deleted = 0
                ORDER BY c.last_modified DESC LIMIT ? \
                OFFSET ? \
                """
        cursor = self.execute_query(query, (keyword_id, limit, offset))
        return [dict(row) for row in cursor.fetchall()]

    # Collection <-> Keyword
    def link_collection_to_keyword(self, collection_id: int, keyword_id: int) -> bool:
        return self._manage_link("collection_keywords", "collection_id", collection_id, "keyword_id", keyword_id,
                                 "link")

    def unlink_collection_from_keyword(self, collection_id: int, keyword_id: int) -> bool:
        return self._manage_link("collection_keywords", "collection_id", collection_id, "keyword_id", keyword_id,
                                 "unlink")

    def get_keywords_for_collection(self, collection_id: int) -> List[Dict[str, Any]]:
        query = """
                SELECT k.* \
                FROM keywords k \
                         JOIN collection_keywords ck ON k.id = ck.keyword_id
                WHERE ck.collection_id = ? \
                  AND k.deleted = 0 \
                ORDER BY k.keyword COLLATE NOCASE
                """
        cursor = self.execute_query(query, (collection_id,))
        return [dict(row) for row in cursor.fetchall()]

    def get_collections_for_keyword(self, keyword_id: int, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        query = """
                SELECT kc.* \
                FROM keyword_collections kc \
                         JOIN collection_keywords ck ON kc.id = ck.collection_id
                WHERE ck.keyword_id = ? \
                  AND kc.deleted = 0
                ORDER BY kc.name COLLATE NOCASE LIMIT ? \
                OFFSET ? \
                """
        cursor = self.execute_query(query, (keyword_id, limit, offset))
        return [dict(row) for row in cursor.fetchall()]

    # Note <-> Keyword
    def link_note_to_keyword(self, note_id: str, keyword_id: int) -> bool: # note_id is str
        return self._manage_link("note_keywords", "note_id", note_id, "keyword_id", keyword_id, "link")

    def unlink_note_from_keyword(self, note_id: str, keyword_id: int) -> bool: # note_id is str
        return self._manage_link("note_keywords", "note_id", note_id, "keyword_id", keyword_id, "unlink")

    def get_keywords_for_note(self, note_id: str) -> List[Dict[str, Any]]: # note_id is str
        query = """
                SELECT k.* \
                FROM keywords k \
                         JOIN note_keywords nk ON k.id = nk.keyword_id
                WHERE nk.note_id = ? \
                  AND k.deleted = 0 \
                ORDER BY k.keyword COLLATE NOCASE
                """
        cursor = self.execute_query(query, (note_id,))
        return [dict(row) for row in cursor.fetchall()]

    def get_notes_for_keyword(self, keyword_id: int, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        query = """
                SELECT n.* \
                FROM notes n \
                         JOIN note_keywords nk ON n.id = nk.note_id
                WHERE nk.keyword_id = ? \
                  AND n.deleted = 0
                ORDER BY n.last_modified DESC LIMIT ? \
                OFFSET ? \
                """
        cursor = self.execute_query(query, (keyword_id, limit, offset))
        return [dict(row) for row in cursor.fetchall()]

    # --- Sync Log Methods ---
    def get_sync_log_entries(self, since_change_id: int = 0, limit: Optional[int] = None,
                             entity_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieves sync log entries newer than a given change_id, optionally filtered by entity type."""
        query_parts = ["SELECT * FROM sync_log WHERE change_id > ?"]
        params_list: List[Any] = [since_change_id]

        if entity_type:
            query_parts.append("AND entity = ?")
            params_list.append(entity_type)

        query_parts.append("ORDER BY change_id ASC")
        if limit is not None:
            query_parts.append("LIMIT ?")
            params_list.append(limit)

        query = " ".join(query_parts)

        try:
            cursor = self.execute_query(query, tuple(params_list))
            results = []
            for row in cursor.fetchall():
                entry = dict(row)
                try:
                    entry['payload'] = json.loads(entry['payload'])
                except json.JSONDecodeError:
                    logger.warning(
                        f"Failed to decode JSON payload for sync_log ID {entry['change_id']}. Payload: {entry['payload'][:100]}")
                    entry['payload'] = None  # Or keep as string, depending on consumer needs
                results.append(entry)
            return results
        except CharactersRAGDBError as e:
            logger.error(f"Error fetching sync log entries: {e}")
            raise

    def get_latest_sync_log_change_id(self) -> int:
        """Returns the highest change_id from the sync_log table."""
        query = "SELECT MAX(change_id) as max_id FROM sync_log"
        try:
            cursor = self.execute_query(query)
            row = cursor.fetchone()
            return row['max_id'] if row and row['max_id'] is not None else 0
        except CharactersRAGDBError as e:
            logger.error(f"Error fetching latest sync log change_id: {e}")
            raise


# --- Transaction Context Manager Class (Helper for `with db.transaction():`) ---
class TransactionContextManager:
    def __init__(self, db_instance: CharactersRAGDB):
        self.db = db_instance
        self.conn: Optional[sqlite3.Connection] = None
        self.is_outermost_transaction = False

    def __enter__(self) -> sqlite3.Connection:
        self.conn = self.db.get_connection()
        if not self.conn.in_transaction:
            # Using deferred transaction by default. Could be "IMMEDIATE" or "EXCLUSIVE" if needed.
            self.conn.execute("BEGIN")
            self.is_outermost_transaction = True
            logger.debug(f"Transaction started (outermost) on thread {threading.get_ident()}.")
        else:
            # SQLite handles nested transactions using savepoints automatically with BEGIN/COMMIT.
            # However, true nested transactions are not supported directly. Python's `in_transaction`
            # might not reflect savepoint depth. We only manage the outermost BEGIN/COMMIT/ROLLBACK.
            logger.debug(
                f"Entering possibly nested transaction block on thread {threading.get_ident()}.")
        return self.conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.conn:  # Should not happen if __enter__ succeeded
            logger.error("Transaction context: Connection is None in __exit__.")
            return False # Re-raise exception if any

        if self.is_outermost_transaction:
            if exc_type:
                logger.error(
                    f"Transaction (outermost) failed, rolling back on thread {threading.get_ident()}: {exc_type.__name__} - {exc_val}",
                    exc_info=False) # exc_info=exc_tb if full traceback wanted here
                try:
                    self.conn.rollback()
                    logger.debug(f"Rollback successful on thread {threading.get_ident()}.")
                except sqlite3.Error as rb_err:
                    logger.critical(f"Rollback FAILED on thread {threading.get_ident()}: {rb_err}", exc_info=True)
            else:
                try:
                    self.conn.commit()
                    logger.debug(
                        f"Transaction (outermost) committed successfully on thread {threading.get_ident()}.")
                except sqlite3.Error as commit_err:
                    logger.error(f"Commit FAILED on thread {threading.get_ident()}, attempting rollback: {commit_err}",
                                 exc_info=True)
                    try:
                        self.conn.rollback()
                        logger.debug(f"Rollback after failed commit successful on thread {threading.get_ident()}.")
                    except sqlite3.Error as rb_err_after_commit_fail:
                        logger.critical(
                            f"Rollback after failed commit also FAILED on thread {threading.get_ident()}: {rb_err_after_commit_fail}",
                            exc_info=True)
                    # Re-raise the commit error so the caller knows the transaction failed.
                    # Encapsulate it if it's not already a DB-specific error from our library.
                    if not isinstance(commit_err, CharactersRAGDBError):
                        raise CharactersRAGDBError(f"Commit failed: {commit_err}") from commit_err
                    else:
                        raise commit_err
        elif exc_type:
            # If an exception occurred in a nested block, we don't do anything here.
            # The outermost block will handle the rollback.
            logger.debug(
                f"Exception in nested transaction block on thread {threading.get_ident()}: {exc_type.__name__}. Outermost transaction will handle rollback if this exception propagates.")

        # Return False to re-raise any exceptions that occurred within the `with` block,
        # allowing them to be handled by the caller or to propagate further up.
        # This is standard behavior for context managers.
        return False
#
# End of ChaChaNotes_DB.py
#######################################################################################################################
