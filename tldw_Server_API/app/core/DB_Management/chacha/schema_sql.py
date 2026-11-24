"""Dependency-free constants of CharactersRAGDB: schema and migration SQL, and constants that moved chacha modules share with the class (TASK-13950.22)."""

from __future__ import annotations

DOCUMENT_CHAT_LINK_SCHEMA = (
    """CREATE TABLE document_chat_links (
        conversation_id TEXT PRIMARY KEY NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
        source_incarnation_id TEXT NOT NULL,
        owner_id TEXT NOT NULL,
        media_uuid TEXT NOT NULL
    )""",
    "CREATE INDEX idx_document_chat_links_owner_media ON document_chat_links(owner_id, media_uuid)",
)

# The persona older Sync v2 projections stored for a chat that named no assistant
# (#3182). It never existed. This one definition serves the migration that clears it
# from stored chats and the Sync materializer, which reads it in a payload as no assistant.
RETIRED_SYNC_PLACEHOLDER_ASSISTANT_KIND = "persona"
RETIRED_SYNC_PLACEHOLDER_ASSISTANT_ID = "sync-v2"

_NOTE_TASK_V60_TABLES = (
    "note_tasks",
    "task_note_projections",
    "task_events",
    "task_event_read_state",
    "note_task_reconciliation_state",
    "task_projection_drifts",
)
_NOTE_TASK_SCOPE_AUTHORITY_TABLE = "note_task_scope_authority"
_NOTE_TASK_V60_RELATIONS = (*_NOTE_TASK_V60_TABLES, _NOTE_TASK_SCOPE_AUTHORITY_TABLE)
_NOTES_MOODBOARD_STUDIO_V61_RELATIONS = (
    _NOTE_TASK_SCOPE_AUTHORITY_TABLE,
    "moodboards",
    "moodboard_notes",
    "note_studio_documents",
)

_SQLITE_CORE_SCHEMA_TABLES = frozenset(
    {
        "character_cards",
        "conversations",
        "messages",
        "notes",
        "note_attachments",
    }
)

_SQLITE_MIGRATION_FIXTURE_DROP_STATEMENTS: dict[str, str] = {
    "persona_profiles": "DROP TABLE IF EXISTS persona_profiles",
    "persona_scope_rules": "DROP TABLE IF EXISTS persona_scope_rules",
    "persona_policy_rules": "DROP TABLE IF EXISTS persona_policy_rules",
    "persona_sessions": "DROP TABLE IF EXISTS persona_sessions",
    "persona_memory_entries": "DROP TABLE IF EXISTS persona_memory_entries",
    "persona_visual_packs": "DROP TABLE IF EXISTS persona_visual_packs",
    "persona_visual_assets": "DROP TABLE IF EXISTS persona_visual_assets",
    "persona_visual_candidates": "DROP TABLE IF EXISTS persona_visual_candidates",
}
_SQLITE_SCHEMA_TABLE_INFO_STATEMENTS: dict[str, str] = {
    "persona_profiles": "PRAGMA table_info('persona_profiles')",
    "persona_memory_entries": "PRAGMA table_info('persona_memory_entries')",
    "persona_visual_candidates": "PRAGMA table_info('persona_visual_candidates')",
    "quiz_questions": "PRAGMA table_info('quiz_questions')",
    "keywords": "PRAGMA table_info('keywords')",
    "keyword_collections": "PRAGMA table_info('keyword_collections')",
    "note_folders": "PRAGMA table_info('note_folders')",
}
_SQLITE_SCHEMA_INDEX_LIST_STATEMENTS: dict[str, str] = {
    "persona_profiles": "PRAGMA index_list('persona_profiles')",
    "persona_memory_entries": "PRAGMA index_list('persona_memory_entries')",
    "persona_visual_packs": "PRAGMA index_list('persona_visual_packs')",
    "persona_visual_assets": "PRAGMA index_list('persona_visual_assets')",
    "keywords": "PRAGMA index_list('keywords')",
    "keyword_collections": "PRAGMA index_list('keyword_collections')",
    "note_folders": "PRAGMA index_list('note_folders')",
}

_FTS_CONFIG: list[tuple[str, str, list[str]]] = [
    (
        "character_cards_fts",
        "character_cards",
        ["name", "description", "personality", "scenario", "system_prompt"],
    ),
    (
        "character_exemplars_fts",
        "character_exemplars",
        ["text", "emotion", "scenario"],
    ),
    (
        "conversations_fts",
        "conversations",
        ["title"],
    ),
    (
        "messages_fts",
        "messages",
        ["content"],
    ),
    (
        "keywords_fts",
        "keywords",
        ["keyword"],
    ),
    (
        "keyword_collections_fts",
        "keyword_collections",
        ["name"],
    ),
    (
        "notes_fts",
        "notes",
        ["title", "content"],
    ),
    (
        "flashcards_fts",
        "flashcards",
        ["front", "back", "notes"],
    ),
    (
        "quiz_questions_fts",
        "quiz_questions",
        ["question_text", "explanation"],
    ),
]

_POSTGRES_SEQUENCE_TABLES: tuple[tuple[str, str], ...] = (
    ("character_cards", "id"),
    ("chacha_keywords", "id"),
    ("keyword_collections", "id"),
    ("sync_log", "change_id"),
    ("moodboards", "id"),
    ("decks", "id"),
    ("flashcards", "id"),
    ("flashcard_reviews", "id"),
    ("flashcard_review_sessions", "id"),
    ("quizzes", "id"),
    ("quiz_questions", "id"),
    ("quiz_attempts", "id"),
    ("osce_stations", "id"),
    ("osce_practice_attempts", "id"),
    ("study_packs", "id"),
    ("study_pack_cards", "id"),
    ("flashcard_citations", "id"),
    ("source_review_plans", "id"),
    ("source_review_occurrences", "id"),
    ("suggestion_snapshots", "id"),
    ("suggestion_generation_links", "id"),
    ("study_assistant_threads", "id"),
    ("study_assistant_messages", "id"),
    ("writing_templates", "id"),
    ("writing_themes", "id"),
    ("voice_command_events", "id"),
    ("skill_registry", "id"),
    ("persona_scope_rules", "id"),
    ("persona_policy_rules", "id"),
)

_FULL_SCHEMA_SQL_V4 = """
/*───────────────────────────────────────────────────────────────
  RAG Character-Chat Schema  -  Version 4   (2025-05-14)
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

/* Ensure default character card (ID 1) exists */
INSERT OR IGNORE INTO character_cards
    (id, name, description, personality, scenario, system_prompt, image,
     post_history_instructions, first_message, message_example,
     creator_notes, alternate_greetings, tags, creator, character_version, extensions,
     created_at, last_modified, client_id, version, deleted)
VALUES
    (1, 'Default Assistant', 'A general-purpose assistant.', NULL, NULL, NULL, NULL, NULL,
     'Hello! How can I help you today?', NULL, NULL, '[]', '[]', 'System', '1.0', '{}',
     CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 'system_init', 1, 0);
/* End of insertion of default character card */

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
  SELECT 'delete',old.id,old.name,old.description,old.personality,old.scenario,old.system_prompt
  WHERE old.deleted = 0;

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
  content           TEXT, -- Text content of the message (nullable for image-only)
  image_data        BLOB DEFAULT NULL,
  image_mime_type   TEXT DEFAULT NULL,
  timestamp         DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  ranking           INTEGER,
  last_modified     DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted           BOOLEAN NOT NULL DEFAULT 0,
  client_id         TEXT    NOT NULL,
  version           INTEGER NOT NULL DEFAULT 1
);
/* ... indexes for messages ... */
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
  3b. Message images (multi-image support)
----------------------------------------------------------------*/
CREATE TABLE IF NOT EXISTS message_images(
  message_id      TEXT    NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
  position        INTEGER NOT NULL,
  image_data      BLOB    NOT NULL,
  image_mime_type TEXT    NOT NULL,
  created_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY(message_id, position)
);

CREATE INDEX IF NOT EXISTS idx_message_images_message ON message_images(message_id);

/*----------------------------------------------------------------
  4. Keywords
----------------------------------------------------------------*/
CREATE TABLE IF NOT EXISTS keywords(
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  sync_id       TEXT    NOT NULL,
  keyword       TEXT    UNIQUE NOT NULL COLLATE NOCASE,
  created_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted       BOOLEAN  NOT NULL DEFAULT 0,
  client_id     TEXT     NOT NULL DEFAULT 'unknown',
  version       INTEGER  NOT NULL DEFAULT 1
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_keywords_sync_id_unique ON keywords(sync_id);

CREATE VIRTUAL TABLE IF NOT EXISTS keywords_fts
USING fts5(
  keyword,
  content='keywords',
  content_rowid='id'
);

/* ───── clean slate ─────────────────────────────────────────── */
DROP TRIGGER IF EXISTS keywords_ai;
DROP TRIGGER IF EXISTS keywords_au;
DROP TRIGGER IF EXISTS keywords_bd;

/* ───── AFTER INSERT → add to index if not deleted ─────────── */
CREATE TRIGGER keywords_ai
AFTER INSERT ON keywords BEGIN
  INSERT INTO keywords_fts(rowid, keyword)
  SELECT new.id, new.keyword
  WHERE new.deleted = 0;
END;

/* ───── AFTER UPDATE → conditional delete + add -────────────── */
CREATE TRIGGER keywords_au
AFTER UPDATE ON keywords BEGIN
  /* delete the old doc only if it was indexed */
  INSERT INTO keywords_fts(keywords_fts, rowid, keyword)
  SELECT 'delete', old.id, old.keyword
  WHERE old.deleted = 0;

  /* add the new doc if it should be indexed */
  INSERT INTO keywords_fts(rowid, keyword)
  SELECT new.id, new.keyword
  WHERE new.deleted = 0;
END;

/* ───── BEFORE DELETE → remove from index if present ────────── */
CREATE TRIGGER keywords_bd
BEFORE DELETE ON keywords BEGIN
  INSERT INTO keywords_fts(keywords_fts, rowid, keyword)
  SELECT 'delete', old.id, old.keyword
  WHERE old.deleted = 0;
END;

/*----------------------------------------------------------------
  5. Keyword collections
----------------------------------------------------------------*/
CREATE TABLE IF NOT EXISTS keyword_collections(
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  sync_id       TEXT    NOT NULL,
  name          TEXT    UNIQUE NOT NULL COLLATE NOCASE,
  parent_id     INTEGER REFERENCES keyword_collections(id)
                         ON DELETE SET NULL ON UPDATE CASCADE,
  created_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted       BOOLEAN  NOT NULL DEFAULT 0,
  client_id     TEXT     NOT NULL DEFAULT 'unknown',
  version       INTEGER  NOT NULL DEFAULT 1
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_keyword_collections_sync_id_unique ON keyword_collections(sync_id);

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
  SELECT 'delete',old.id,old.name
  WHERE old.deleted = 0;

  INSERT INTO keyword_collections_fts(rowid,name)
  SELECT new.id,new.name
  WHERE new.deleted = 0;
END;

CREATE TRIGGER keyword_collections_ad
AFTER DELETE ON keyword_collections BEGIN
  INSERT INTO keyword_collections_fts(keyword_collections_fts,rowid,name)
  SELECT 'delete',old.id,old.name
  WHERE old.deleted = 0;
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
  SELECT 'delete',old.rowid,old.title,old.content
  WHERE old.deleted = 0;

  INSERT INTO notes_fts(rowid,title,content)
  SELECT new.rowid,new.title,new.content
  WHERE new.deleted = 0;
END;

CREATE TRIGGER notes_ad
AFTER DELETE ON notes BEGIN
  INSERT INTO notes_fts(notes_fts,rowid,title,content)
  SELECT 'delete',old.rowid,old.title,old.content
  WHERE old.deleted = 0;
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

/*-- drop any pre-existing sync triggers -----------*/
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

/*-- sync triggers: messages ---------------*/
CREATE TRIGGER messages_sync_create
AFTER INSERT ON messages BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('messages',NEW.id,'create',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'conversation_id',NEW.conversation_id,'parent_message_id',NEW.parent_message_id,
                     'sender',NEW.sender,'content',NEW.content,
                     'image_mime_type',NEW.image_mime_type,
                     'timestamp',NEW.timestamp,'ranking',NEW.ranking,
                     'last_modified',NEW.last_modified,'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER messages_sync_update
AFTER UPDATE ON messages
WHEN OLD.deleted = NEW.deleted AND (
     OLD.content IS NOT NEW.content OR
     OLD.image_data IS NOT NEW.image_data OR
     OLD.image_mime_type IS NOT NEW.image_mime_type OR
     OLD.ranking IS NOT NEW.ranking OR
     OLD.parent_message_id IS NOT NEW.parent_message_id OR
     OLD.last_modified IS NOT NEW.last_modified OR
     OLD.version IS NOT NEW.version)
BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('messages',NEW.id,'update',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'conversation_id',NEW.conversation_id,'parent_message_id',NEW.parent_message_id,
                     'sender',NEW.sender,'content',NEW.content,
                     'image_mime_type',NEW.image_mime_type,
                     'timestamp',NEW.timestamp,'ranking',NEW.ranking,
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

/* messages_sync_delete and messages_sync_undelete don't strictly need image_data in payload
   as 'delete' just tombstones, and 'undelete' (as 'update') would repopulate all fields.
   The 'undelete' trigger would need to include the image fields if they are to be restored.
*/
CREATE TRIGGER messages_sync_undelete
AFTER UPDATE ON messages
WHEN OLD.deleted = 1 AND NEW.deleted = 0
BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('messages',NEW.id,'update',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'conversation_id',NEW.conversation_id,'parent_message_id',NEW.parent_message_id,
                     'sender',NEW.sender,'content',NEW.content,
                     'image_mime_type',NEW.image_mime_type,
                     'timestamp',NEW.timestamp,'ranking',NEW.ranking,
                     'last_modified',NEW.last_modified,'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

/*-- sync triggers: conversations -----------*/
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

/*-- sync triggers: character_cards ---------*/
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

/*-- sync triggers: notes ---------------*/
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

/*-- sync triggers: keywords ----*/
CREATE TRIGGER keywords_sync_create
AFTER INSERT ON keywords BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('keywords',CAST(NEW.id AS TEXT),'create',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'keyword',NEW.keyword,'created_at',NEW.created_at,
                     'last_modified',NEW.last_modified,'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER keywords_sync_update
AFTER UPDATE ON keywords
WHEN OLD.deleted = NEW.deleted AND (
     OLD.keyword IS NOT NEW.keyword OR -- Though keyword itself is unlikely to change if it's the unique identifier
     OLD.last_modified IS NOT NEW.last_modified OR
     OLD.version IS NOT NEW.version)
BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('keywords',CAST(NEW.id AS TEXT),'update',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'keyword',NEW.keyword,'created_at',NEW.created_at,
                     'last_modified',NEW.last_modified,'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER keywords_sync_delete
AFTER UPDATE ON keywords
WHEN OLD.deleted = 0 AND NEW.deleted = 1
BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('keywords',CAST(NEW.id AS TEXT),'delete',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'deleted',NEW.deleted,'last_modified',NEW.last_modified,
                     'version',NEW.version,'client_id',NEW.client_id));
END;

CREATE TRIGGER keywords_sync_undelete
AFTER UPDATE ON keywords
WHEN OLD.deleted = 1 AND NEW.deleted = 0
BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('keywords',CAST(NEW.id AS TEXT),'update',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'keyword',NEW.keyword,'created_at',NEW.created_at,
                     'last_modified',NEW.last_modified,'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;


/*-- sync triggers: keyword_collections ----*/
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
   SET version = 4
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 4;
"""

# --- Migration: V4 -> V5 (Flashcards/Decks/Reviews) ---
_MIGRATION_SQL_V4_TO_V5 = """
/*───────────────────────────────────────────────────────────────
  Migration to Version 5 - Flashcards/Decks/SRS (2025-09-21)
───────────────────────────────────────────────────────────────*/
PRAGMA foreign_keys = ON;

/* Decks table */
CREATE TABLE IF NOT EXISTS decks(
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  name          TEXT UNIQUE NOT NULL,
  description   TEXT,
  parent_deck_id INTEGER REFERENCES decks(id) ON DELETE SET NULL,
  workspace_id  TEXT REFERENCES workspaces(id) ON DELETE SET NULL,
  review_prompt_side TEXT NOT NULL DEFAULT 'front',
  created_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted       BOOLEAN  NOT NULL DEFAULT 0,
  client_id     TEXT     NOT NULL DEFAULT 'unknown',
  version       INTEGER  NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_decks_deleted ON decks(deleted);
CREATE INDEX IF NOT EXISTS idx_decks_last_modified ON decks(last_modified);
CREATE INDEX IF NOT EXISTS idx_decks_workspace_id ON decks(workspace_id);
CREATE INDEX IF NOT EXISTS idx_decks_parent_deck_id ON decks(parent_deck_id);

/* Flashcards table - with integer id for FTS external-content */
CREATE TABLE IF NOT EXISTS flashcards(
  id               INTEGER PRIMARY KEY AUTOINCREMENT,
  uuid             TEXT UNIQUE NOT NULL,
  deck_id          INTEGER REFERENCES decks(id) ON DELETE SET NULL,
  front            TEXT NOT NULL,
  back             TEXT NOT NULL,
  notes            TEXT,
  is_cloze         BOOLEAN NOT NULL DEFAULT 0,
  tags_json        TEXT,
  source_ref_type  TEXT CHECK(source_ref_type IN ('media','message','note','manual')) DEFAULT 'manual',
  source_ref_id    TEXT,
  ef               REAL NOT NULL DEFAULT 2.5,
  interval_days    INTEGER NOT NULL DEFAULT 0,
  repetitions      INTEGER NOT NULL DEFAULT 0,
  lapses           INTEGER NOT NULL DEFAULT 0,
  due_at           DATETIME,
  last_reviewed_at DATETIME,
  created_at       DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted          BOOLEAN NOT NULL DEFAULT 0,
  client_id        TEXT NOT NULL DEFAULT 'unknown',
  version          INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_flashcards_deck_id ON flashcards(deck_id);
CREATE INDEX IF NOT EXISTS idx_flashcards_due_at ON flashcards(due_at);
CREATE INDEX IF NOT EXISTS idx_flashcards_deleted ON flashcards(deleted);
CREATE INDEX IF NOT EXISTS idx_flashcards_created_at ON flashcards(created_at);
CREATE UNIQUE INDEX IF NOT EXISTS idx_flashcards_uuid ON flashcards(uuid);

/* FTS for flashcards (front/back/notes) */
CREATE VIRTUAL TABLE IF NOT EXISTS flashcards_fts
USING fts5(
  front, back, notes,
  content='flashcards',
  content_rowid='id'
);

DROP TRIGGER IF EXISTS flashcards_ai;
DROP TRIGGER IF EXISTS flashcards_au;
DROP TRIGGER IF EXISTS flashcards_ad;

CREATE TRIGGER flashcards_ai
AFTER INSERT ON flashcards BEGIN
  INSERT INTO flashcards_fts(rowid,front,back,notes)
  SELECT new.id,new.front,new.back,new.notes
  WHERE new.deleted = 0;
END;

CREATE TRIGGER flashcards_au
AFTER UPDATE ON flashcards BEGIN
  INSERT INTO flashcards_fts(flashcards_fts,rowid,front,back,notes)
  VALUES('delete',old.id,old.front,old.back,old.notes);

  INSERT INTO flashcards_fts(rowid,front,back,notes)
  SELECT new.id,new.front,new.back,new.notes
  WHERE new.deleted = 0;
END;

CREATE TRIGGER flashcards_ad
AFTER DELETE ON flashcards BEGIN
  INSERT INTO flashcards_fts(flashcards_fts,rowid,front,back,notes)
  VALUES('delete',old.id,old.front,old.back,old.notes);
END;

/* Flashcard keyword linking */
CREATE TABLE IF NOT EXISTS flashcard_keywords(
  card_id    INTEGER NOT NULL REFERENCES flashcards(id) ON DELETE CASCADE,
  keyword_id INTEGER NOT NULL REFERENCES keywords(id)  ON DELETE CASCADE,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY(card_id, keyword_id)
);
CREATE INDEX IF NOT EXISTS idx_flashcard_kw_kw ON flashcard_keywords(keyword_id);
CREATE INDEX IF NOT EXISTS idx_flashcard_kw_card ON flashcard_keywords(card_id);

/* Reviews table (history of SRS reviews) */
CREATE TABLE IF NOT EXISTS flashcard_reviews(
  id                     INTEGER PRIMARY KEY AUTOINCREMENT,
  card_id                INTEGER NOT NULL REFERENCES flashcards(id) ON DELETE CASCADE,
  reviewed_at            DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  rating                 INTEGER NOT NULL,
  answer_time_ms         INTEGER,
  scheduled_interval_days INTEGER,
  new_ef                 REAL,
  new_repetitions        INTEGER,
  was_lapse              BOOLEAN NOT NULL DEFAULT 0,
  client_id              TEXT NOT NULL DEFAULT 'unknown'
);
CREATE INDEX IF NOT EXISTS idx_flashcard_reviews_card ON flashcard_reviews(card_id);
CREATE INDEX IF NOT EXISTS idx_flashcard_reviews_time ON flashcard_reviews(reviewed_at);

/* Review session persistence for grouped flashcard study runs */
CREATE TABLE IF NOT EXISTS flashcard_review_sessions(
  id               INTEGER PRIMARY KEY AUTOINCREMENT,
  deck_id          INTEGER REFERENCES decks(id) ON DELETE SET NULL,
  deck_name_snapshot TEXT,
  review_mode      TEXT NOT NULL DEFAULT 'due',
  tag_filter       TEXT,
  scope_key        TEXT NOT NULL,
  status           TEXT NOT NULL DEFAULT 'active',
  started_at       DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_activity_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  completed_at     DATETIME,
  cards_reviewed   INTEGER DEFAULT 0,
  correct_count    INTEGER DEFAULT 0,
  source_bundle_json TEXT,
  study_pack_id    INTEGER,
  client_id        TEXT NOT NULL DEFAULT 'unknown'
);
CREATE INDEX IF NOT EXISTS idx_flashcard_review_sessions_scope ON flashcard_review_sessions(scope_key);
CREATE INDEX IF NOT EXISTS idx_flashcard_review_sessions_status ON flashcard_review_sessions(status);
CREATE INDEX IF NOT EXISTS idx_flashcard_review_sessions_deck ON flashcard_review_sessions(deck_id);
CREATE INDEX IF NOT EXISTS idx_flashcard_review_sessions_activity ON flashcard_review_sessions(last_activity_at);

/* Sync triggers for decks */
DROP TRIGGER IF EXISTS decks_sync_create;
DROP TRIGGER IF EXISTS decks_sync_update;
DROP TRIGGER IF EXISTS decks_sync_delete;
DROP TRIGGER IF EXISTS decks_sync_undelete;

CREATE TRIGGER decks_sync_create
AFTER INSERT ON decks BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('decks',CAST(NEW.id AS TEXT),'create',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'name',NEW.name,'description',NEW.description,
                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER decks_sync_update
AFTER UPDATE ON decks
WHEN OLD.deleted = NEW.deleted AND (
     OLD.name IS NOT NEW.name OR
     OLD.description IS NOT NEW.description OR
     OLD.last_modified IS NOT NEW.last_modified OR
     OLD.version IS NOT NEW.version)
BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('decks',CAST(NEW.id AS TEXT),'update',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'name',NEW.name,'description',NEW.description,
                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER decks_sync_delete
AFTER UPDATE ON decks
WHEN OLD.deleted = 0 AND NEW.deleted = 1
BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('decks',CAST(NEW.id AS TEXT),'delete',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'deleted',NEW.deleted,'last_modified',NEW.last_modified,
                     'version',NEW.version,'client_id',NEW.client_id));
END;

CREATE TRIGGER decks_sync_undelete
AFTER UPDATE ON decks
WHEN OLD.deleted = 1 AND NEW.deleted = 0
BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('decks',CAST(NEW.id AS TEXT),'update',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'name',NEW.name,'description',NEW.description,
                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

/* Sync triggers for flashcards */
DROP TRIGGER IF EXISTS flashcards_sync_create;
DROP TRIGGER IF EXISTS flashcards_sync_update;
DROP TRIGGER IF EXISTS flashcards_sync_delete;
DROP TRIGGER IF EXISTS flashcards_sync_undelete;

CREATE TRIGGER flashcards_sync_create
AFTER INSERT ON flashcards BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('flashcards',NEW.uuid,'create',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('uuid',NEW.uuid,'deck_id',NEW.deck_id,'front',NEW.front,'back',NEW.back,
                     'notes',NEW.notes,'is_cloze',NEW.is_cloze,'tags_json',NEW.tags_json,
                     'ef',NEW.ef,'interval_days',NEW.interval_days,'repetitions',NEW.repetitions,
                     'lapses',NEW.lapses,'due_at',NEW.due_at,'last_reviewed_at',NEW.last_reviewed_at,
                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER flashcards_sync_update
AFTER UPDATE ON flashcards
WHEN OLD.deleted = NEW.deleted AND (
     OLD.deck_id IS NOT NEW.deck_id OR
     OLD.front IS NOT NEW.front OR
     OLD.back IS NOT NEW.back OR
     OLD.notes IS NOT NEW.notes OR
     OLD.is_cloze IS NOT NEW.is_cloze OR
     OLD.tags_json IS NOT NEW.tags_json OR
     OLD.ef IS NOT NEW.ef OR
     OLD.interval_days IS NOT NEW.interval_days OR
     OLD.repetitions IS NOT NEW.repetitions OR
     OLD.lapses IS NOT NEW.lapses OR
     OLD.due_at IS NOT NEW.due_at OR
     OLD.last_reviewed_at IS NOT NEW.last_reviewed_at OR
     OLD.last_modified IS NOT NEW.last_modified OR
     OLD.version IS NOT NEW.version)
BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('flashcards',NEW.uuid,'update',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('uuid',NEW.uuid,'deck_id',NEW.deck_id,'front',NEW.front,'back',NEW.back,
                     'notes',NEW.notes,'is_cloze',NEW.is_cloze,'tags_json',NEW.tags_json,
                     'ef',NEW.ef,'interval_days',NEW.interval_days,'repetitions',NEW.repetitions,
                     'lapses',NEW.lapses,'due_at',NEW.due_at,'last_reviewed_at',NEW.last_reviewed_at,
                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER flashcards_sync_delete
AFTER UPDATE ON flashcards
WHEN OLD.deleted = 0 AND NEW.deleted = 1
BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('flashcards',NEW.uuid,'delete',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('uuid',NEW.uuid,'deleted',NEW.deleted,'last_modified',NEW.last_modified,
                     'version',NEW.version,'client_id',NEW.client_id));
END;

CREATE TRIGGER flashcards_sync_undelete
AFTER UPDATE ON flashcards
WHEN OLD.deleted = 1 AND NEW.deleted = 0
BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('flashcards',NEW.uuid,'update',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('uuid',NEW.uuid,'deck_id',NEW.deck_id,'front',NEW.front,'back',NEW.back,
                     'notes',NEW.notes,'is_cloze',NEW.is_cloze,'tags_json',NEW.tags_json,
                     'ef',NEW.ef,'interval_days',NEW.interval_days,'repetitions',NEW.repetitions,
                     'lapses',NEW.lapses,'due_at',NEW.due_at,'last_reviewed_at',NEW.last_reviewed_at,
                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

/* Finalise version bump to 5 */
UPDATE db_schema_version
   SET version = 5
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 5;
"""

# --- Migration: V5 -> V6 (Flashcard model_type + extra) ---
_MIGRATION_SQL_V5_TO_V6 = """
/*───────────────────────────────────────────────────────────────
  Migration to Version 6 - Flashcard model_type + extra (2025-09-21)
───────────────────────────────────────────────────────────────*/
PRAGMA foreign_keys = ON;

ALTER TABLE flashcards ADD COLUMN model_type TEXT NOT NULL DEFAULT 'basic' CHECK(model_type IN ('basic','basic_reverse','cloze'));
ALTER TABLE flashcards ADD COLUMN extra TEXT;

UPDATE db_schema_version
   SET version = 6
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 6;
"""

# --- Migration: V6 -> V7 (Flashcard reverse flag) ---
_MIGRATION_SQL_V6_TO_V7 = """
/*───────────────────────────────────────────────────────────────
  Migration to Version 7 - Flashcard reverse flag (2025-09-21)
───────────────────────────────────────────────────────────────*/
PRAGMA foreign_keys = ON;

ALTER TABLE flashcards ADD COLUMN reverse BOOLEAN NOT NULL DEFAULT 0;

UPDATE db_schema_version
   SET version = 7
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 7;
"""

# --- Migration: V7 -> V8 (Message images table) ---
_MIGRATION_SQL_V7_TO_V8 = """
/*───────────────────────────────────────────────────────────────
  Migration to Version 8 - Message images table (2025-10-14)
───────────────────────────────────────────────────────────────*/
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS message_images(
  message_id      TEXT    NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
  position        INTEGER NOT NULL,
  image_data      BLOB    NOT NULL,
  image_mime_type TEXT    NOT NULL,
  created_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY(message_id, position)
);

CREATE INDEX IF NOT EXISTS idx_message_images_message ON message_images(message_id);

INSERT OR IGNORE INTO message_images (message_id, position, image_data, image_mime_type, created_at)
SELECT id, 0, image_data, image_mime_type, CURRENT_TIMESTAMP
FROM messages
WHERE image_data IS NOT NULL;

UPDATE db_schema_version
   SET version = 8
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 8;
"""

# --- Migration: V8 -> V9 (Notes Graph: note_edges table) ---
_MIGRATION_SQL_V8_TO_V9 = """
/*───────────────────────────────────────────────────────────────
  Migration to Version 9 - Notes Graph manual edges (2025-11-11)
───────────────────────────────────────────────────────────────*/
PRAGMA foreign_keys = ON;

/* Explicit manual note edges (per user). Derived edges are NOT persisted. */
CREATE TABLE IF NOT EXISTS note_edges(
  edge_id       TEXT PRIMARY KEY,
  user_id       TEXT NOT NULL,
  from_note_id  TEXT NOT NULL,
  to_note_id    TEXT NOT NULL,
  type          TEXT NOT NULL, /* 'manual' (MVP) */
  directed      INTEGER NOT NULL DEFAULT 0,
  weight        REAL DEFAULT 1.0,
  created_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  created_by    TEXT NOT NULL,
  metadata      JSON
);

/* Canonicalization for undirected edges (enforced at application layer).
   Self-loop prevention (from_note_id != to_note_id) is also enforced at the
   application layer for portability. We intentionally do not add a DB-level
   CHECK constraint here because the Postgres migration transformer does not
   currently translate SQLite CHECK clauses. */

/* Helpful indexes */
CREATE INDEX IF NOT EXISTS idx_note_edges_user_from_to ON note_edges(user_id, from_note_id, to_note_id);
CREATE INDEX IF NOT EXISTS idx_note_edges_user_type     ON note_edges(user_id, type);
CREATE INDEX IF NOT EXISTS idx_note_edges_user_to       ON note_edges(user_id, to_note_id);

/* Uniqueness for undirected edges after canonicalization */
CREATE UNIQUE INDEX IF NOT EXISTS uniq_note_edges_undirected ON note_edges(user_id, type, directed, from_note_id, to_note_id);

UPDATE db_schema_version
   SET version = 9
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 9;
"""

# --- Migration: V9 -> V10 (Chat metadata + note backlinks + indexes) ---
_MIGRATION_SQL_V9_TO_V10 = """
/*───────────────────────────────────────────────────────────────
  Migration to Version 10 - Chat metadata + note backlinks (2025-11-20)
───────────────────────────────────────────────────────────────*/
PRAGMA foreign_keys = ON;

/* Conversations: state/topic/cluster/source/external_ref */
ALTER TABLE conversations ADD COLUMN state TEXT NOT NULL DEFAULT 'in-progress' CHECK(state IN ('in-progress','resolved','backlog','non-viable'));
ALTER TABLE conversations ADD COLUMN topic_label TEXT;
ALTER TABLE conversations ADD COLUMN cluster_id TEXT;
ALTER TABLE conversations ADD COLUMN source TEXT;
ALTER TABLE conversations ADD COLUMN external_ref TEXT;

/* Backfill legacy rows to default state */
UPDATE conversations SET state = 'in-progress' WHERE state IS NULL OR TRIM(state) = '';

/* Helpful indexes for new filters/orderings */
CREATE INDEX IF NOT EXISTS idx_conversations_state ON conversations(state);
CREATE INDEX IF NOT EXISTS idx_conversations_cluster ON conversations(cluster_id);
CREATE INDEX IF NOT EXISTS idx_conversations_last_modified ON conversations(last_modified);
CREATE INDEX IF NOT EXISTS idx_conversations_topic_label ON conversations(topic_label);

/* Notes: backlinks to conversations/messages */
ALTER TABLE notes ADD COLUMN conversation_id TEXT REFERENCES conversations(id) ON DELETE SET NULL;
ALTER TABLE notes ADD COLUMN message_id TEXT REFERENCES messages(id) ON DELETE SET NULL;
CREATE INDEX IF NOT EXISTS idx_notes_conversation ON notes(conversation_id);
CREATE INDEX IF NOT EXISTS idx_notes_message ON notes(message_id);

UPDATE db_schema_version
   SET version = 10
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 10;
"""

# --- Migration: V10 -> V11 (Allow NULL message content) ---
_MIGRATION_SQL_V10_TO_V11 = """
/*───────────────────────────────────────────────────────────────
  Migration to Version 11 - Nullable message content (2025-11-25)
───────────────────────────────────────────────────────────────*/
PRAGMA foreign_keys = OFF;

DROP TRIGGER IF EXISTS messages_ai;
DROP TRIGGER IF EXISTS messages_au;
DROP TRIGGER IF EXISTS messages_ad;
DROP TRIGGER IF EXISTS messages_sync_create;
DROP TRIGGER IF EXISTS messages_sync_update;
DROP TRIGGER IF EXISTS messages_sync_delete;
DROP TRIGGER IF EXISTS messages_sync_undelete;

DROP TABLE IF EXISTS messages_fts;
DROP TABLE IF EXISTS messages_new;

CREATE TABLE IF NOT EXISTS messages_new(
  id                TEXT PRIMARY KEY,                 /* UUID */
  conversation_id   TEXT  NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  parent_message_id TEXT  REFERENCES messages_new(id) ON DELETE SET NULL,
  sender            TEXT  NOT NULL,
  content           TEXT, -- Text content of the message (nullable for image-only)
  image_data        BLOB DEFAULT NULL,
  image_mime_type   TEXT DEFAULT NULL,
  timestamp         DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  ranking           INTEGER,
  last_modified     DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted           BOOLEAN NOT NULL DEFAULT 0,
  client_id         TEXT    NOT NULL,
  version           INTEGER NOT NULL DEFAULT 1
);

INSERT INTO messages_new (
  id, conversation_id, parent_message_id, sender, content, image_data, image_mime_type,
  timestamp, ranking, last_modified, deleted, client_id, version
)
SELECT
  id, conversation_id, parent_message_id, sender, content, image_data, image_mime_type,
  timestamp, ranking, last_modified, deleted, client_id, version
FROM messages;

DROP TABLE messages;
ALTER TABLE messages_new RENAME TO messages;

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

CREATE TRIGGER messages_sync_create
AFTER INSERT ON messages BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('messages',NEW.id,'create',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'conversation_id',NEW.conversation_id,'parent_message_id',NEW.parent_message_id,
                     'sender',NEW.sender,'content',NEW.content,
                     'image_mime_type',NEW.image_mime_type,
                     'timestamp',NEW.timestamp,'ranking',NEW.ranking,
                     'last_modified',NEW.last_modified,'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER messages_sync_update
AFTER UPDATE ON messages
WHEN OLD.deleted = NEW.deleted AND (
     OLD.content IS NOT NEW.content OR
     OLD.image_data IS NOT NEW.image_data OR
     OLD.image_mime_type IS NOT NEW.image_mime_type OR
     OLD.ranking IS NOT NEW.ranking OR
     OLD.parent_message_id IS NOT NEW.parent_message_id OR
     OLD.last_modified IS NOT NEW.last_modified OR
     OLD.version IS NOT NEW.version)
BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('messages',NEW.id,'update',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'conversation_id',NEW.conversation_id,'parent_message_id',NEW.parent_message_id,
                     'sender',NEW.sender,'content',NEW.content,
                     'image_mime_type',NEW.image_mime_type,
                     'timestamp',NEW.timestamp,'ranking',NEW.ranking,
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
                     'sender',NEW.sender,'content',NEW.content,
                     'image_mime_type',NEW.image_mime_type,
                     'timestamp',NEW.timestamp,'ranking',NEW.ranking,
                     'last_modified',NEW.last_modified,'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

INSERT INTO messages_fts(messages_fts) VALUES('rebuild');

PRAGMA foreign_keys = ON;

UPDATE db_schema_version
   SET version = 11
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 11;
"""

# --- Migration: V11 -> V12 (Quizzes + attempts) ---
_MIGRATION_SQL_V11_TO_V12 = """
/*───────────────────────────────────────────────────────────────
  Migration to Version 12 - Quizzes (2025-12-01)
───────────────────────────────────────────────────────────────*/
PRAGMA foreign_keys = ON;

/* Quizzes table */
CREATE TABLE IF NOT EXISTS quizzes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  description TEXT,
  workspace_tag TEXT,
  workspace_id TEXT REFERENCES workspaces(id) ON DELETE SET NULL,
  media_id INTEGER,
  total_questions INTEGER DEFAULT 0,
  time_limit_seconds INTEGER,
  passing_score INTEGER,
  deleted BOOLEAN NOT NULL DEFAULT 0,
  client_id TEXT NOT NULL DEFAULT 'unknown',
  version INTEGER NOT NULL DEFAULT 1,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_quizzes_media_id ON quizzes(media_id);
CREATE INDEX IF NOT EXISTS idx_quizzes_deleted ON quizzes(deleted);
CREATE INDEX IF NOT EXISTS idx_quizzes_last_modified ON quizzes(last_modified);
CREATE INDEX IF NOT EXISTS idx_quizzes_workspace_id ON quizzes(workspace_id);

/* Quiz questions */
CREATE TABLE IF NOT EXISTS quiz_questions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  quiz_id INTEGER NOT NULL REFERENCES quizzes(id) ON DELETE CASCADE,
  question_type TEXT NOT NULL CHECK(question_type IN ('multiple_choice', 'true_false', 'fill_blank')),
  question_text TEXT NOT NULL,
  options TEXT,
  correct_answer TEXT NOT NULL,
  explanation TEXT,
  points INTEGER NOT NULL DEFAULT 1,
  order_index INTEGER NOT NULL DEFAULT 0,
  tags_json TEXT,
  deleted BOOLEAN NOT NULL DEFAULT 0,
  client_id TEXT NOT NULL DEFAULT 'unknown',
  version INTEGER NOT NULL DEFAULT 1,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_quiz_questions_quiz_id ON quiz_questions(quiz_id);
CREATE INDEX IF NOT EXISTS idx_quiz_questions_deleted ON quiz_questions(deleted);
CREATE INDEX IF NOT EXISTS idx_quiz_questions_order ON quiz_questions(quiz_id, order_index);

/* FTS for quiz questions */
CREATE VIRTUAL TABLE IF NOT EXISTS quiz_questions_fts
USING fts5(
  question_text, explanation,
  content='quiz_questions',
  content_rowid='id'
);

DROP TRIGGER IF EXISTS quiz_questions_ai;
DROP TRIGGER IF EXISTS quiz_questions_au;
DROP TRIGGER IF EXISTS quiz_questions_ad;

CREATE TRIGGER quiz_questions_ai
AFTER INSERT ON quiz_questions BEGIN
  INSERT INTO quiz_questions_fts(rowid, question_text, explanation)
  SELECT new.id, new.question_text, new.explanation
  WHERE new.deleted = 0;
END;

CREATE TRIGGER quiz_questions_au
AFTER UPDATE ON quiz_questions BEGIN
  INSERT INTO quiz_questions_fts(quiz_questions_fts,rowid,question_text,explanation)
  VALUES('delete',old.id,old.question_text,old.explanation);

  INSERT INTO quiz_questions_fts(rowid, question_text, explanation)
  SELECT new.id, new.question_text, new.explanation
  WHERE new.deleted = 0;
END;

CREATE TRIGGER quiz_questions_ad
AFTER DELETE ON quiz_questions BEGIN
  INSERT INTO quiz_questions_fts(quiz_questions_fts,rowid,question_text,explanation)
  VALUES('delete',old.id,old.question_text,old.explanation);
END;

/* Quiz attempts */
CREATE TABLE IF NOT EXISTS quiz_attempts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  quiz_id INTEGER NOT NULL REFERENCES quizzes(id),
  started_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  completed_at DATETIME,
  score INTEGER,
  total_possible INTEGER NOT NULL,
  time_spent_seconds INTEGER,
  questions_snapshot TEXT,
  answers TEXT,
  client_id TEXT NOT NULL DEFAULT 'unknown'
);
CREATE INDEX IF NOT EXISTS idx_quiz_attempts_quiz_id ON quiz_attempts(quiz_id);
CREATE INDEX IF NOT EXISTS idx_quiz_attempts_completed_at ON quiz_attempts(completed_at);

UPDATE db_schema_version
   SET version = 12
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 12;
"""

# --- Migration: V12 -> V13 (Message metadata) ---
_MIGRATION_SQL_V12_TO_V13 = """
/*───────────────────────────────────────────────────────────────
  Migration to Version 13 - Message metadata (2026-01-07)
───────────────────────────────────────────────────────────────*/
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS message_metadata(
  message_id TEXT PRIMARY KEY REFERENCES messages(id) ON DELETE CASCADE,
  tool_calls_json TEXT,
  extra_json TEXT,
  last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

UPDATE db_schema_version
   SET version = 13
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 13;
"""

# --- Migration: V13 -> V14 (Chat topic metadata + clusters + flashcard backlinks) ---
_MIGRATION_SQL_V13_TO_V14 = """
/*───────────────────────────────────────────────────────────────
  Migration to Version 14 - Chat topic metadata + clusters (2026-01-20)
───────────────────────────────────────────────────────────────*/
PRAGMA foreign_keys = ON;

/* Conversations: topic tagging metadata */
ALTER TABLE conversations ADD COLUMN IF NOT EXISTS topic_label_source TEXT CHECK(topic_label_source IN ('manual','auto'));
ALTER TABLE conversations ADD COLUMN IF NOT EXISTS topic_last_tagged_at DATETIME;
ALTER TABLE conversations ADD COLUMN IF NOT EXISTS topic_last_tagged_message_id TEXT;

/* Cluster metadata */
CREATE TABLE IF NOT EXISTS conversation_clusters(
  cluster_id TEXT PRIMARY KEY,
  title TEXT,
  centroid TEXT,
  size INTEGER NOT NULL DEFAULT 0,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

/* Helpful indexes */
CREATE INDEX IF NOT EXISTS idx_conversations_source_external_ref ON conversations(source, external_ref);

/* Flashcards: backlinks to conversations/messages */
ALTER TABLE flashcards ADD COLUMN IF NOT EXISTS conversation_id TEXT REFERENCES conversations(id) ON DELETE SET NULL;
ALTER TABLE flashcards ADD COLUMN IF NOT EXISTS message_id TEXT REFERENCES messages(id) ON DELETE SET NULL;
CREATE INDEX IF NOT EXISTS idx_flashcards_conversation ON flashcards(conversation_id);
CREATE INDEX IF NOT EXISTS idx_flashcards_message ON flashcards(message_id);

UPDATE db_schema_version
   SET version = 14
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 14;
"""

# --- Migration: V14 -> V15 (Writing Playground persistence) ---
_MIGRATION_SQL_V14_TO_V15 = """
/*───────────────────────────────────────────────────────────────
  Migration to Version 15 - Writing Playground persistence (2026-02-10)
───────────────────────────────────────────────────────────────*/
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS writing_sessions(
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  schema_version INTEGER NOT NULL DEFAULT 1,
  version_parent_id TEXT,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted BOOLEAN NOT NULL DEFAULT 0,
  client_id TEXT NOT NULL DEFAULT 'unknown',
  version INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_writing_sessions_deleted ON writing_sessions(deleted);
CREATE INDEX IF NOT EXISTS idx_writing_sessions_last_modified ON writing_sessions(last_modified);
CREATE INDEX IF NOT EXISTS idx_writing_sessions_name ON writing_sessions(name);

CREATE TABLE IF NOT EXISTS writing_templates(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT UNIQUE NOT NULL,
  payload_json TEXT NOT NULL,
  schema_version INTEGER NOT NULL DEFAULT 1,
  version_parent_id TEXT,
  is_default BOOLEAN NOT NULL DEFAULT 0,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted BOOLEAN NOT NULL DEFAULT 0,
  client_id TEXT NOT NULL DEFAULT 'unknown',
  version INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_writing_templates_deleted ON writing_templates(deleted);
CREATE INDEX IF NOT EXISTS idx_writing_templates_last_modified ON writing_templates(last_modified);

CREATE TABLE IF NOT EXISTS writing_themes(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT UNIQUE NOT NULL,
  class_name TEXT,
  css TEXT,
  schema_version INTEGER NOT NULL DEFAULT 1,
  version_parent_id TEXT,
  is_default BOOLEAN NOT NULL DEFAULT 0,
  order_index INTEGER NOT NULL DEFAULT 0,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted BOOLEAN NOT NULL DEFAULT 0,
  client_id TEXT NOT NULL DEFAULT 'unknown',
  version INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_writing_themes_deleted ON writing_themes(deleted);
CREATE INDEX IF NOT EXISTS idx_writing_themes_last_modified ON writing_themes(last_modified);
CREATE INDEX IF NOT EXISTS idx_writing_themes_order ON writing_themes(order_index);

DROP TRIGGER IF EXISTS writing_sessions_sync_create;
DROP TRIGGER IF EXISTS writing_sessions_sync_update;
DROP TRIGGER IF EXISTS writing_sessions_sync_delete;
DROP TRIGGER IF EXISTS writing_sessions_sync_undelete;

CREATE TRIGGER writing_sessions_sync_create
AFTER INSERT ON writing_sessions BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('writing_sessions',NEW.id,'create',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'name',NEW.name,'payload_json',NEW.payload_json,
                     'schema_version',NEW.schema_version,'version_parent_id',NEW.version_parent_id,
                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER writing_sessions_sync_update
AFTER UPDATE ON writing_sessions
WHEN OLD.deleted = NEW.deleted AND (
     OLD.name IS NOT NEW.name OR
     OLD.payload_json IS NOT NEW.payload_json OR
     OLD.schema_version IS NOT NEW.schema_version OR
     OLD.version_parent_id IS NOT NEW.version_parent_id OR
     OLD.last_modified IS NOT NEW.last_modified OR
     OLD.version IS NOT NEW.version)
BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('writing_sessions',NEW.id,'update',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'name',NEW.name,'payload_json',NEW.payload_json,
                     'schema_version',NEW.schema_version,'version_parent_id',NEW.version_parent_id,
                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER writing_sessions_sync_delete
AFTER UPDATE ON writing_sessions
WHEN OLD.deleted = 0 AND NEW.deleted = 1
BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('writing_sessions',NEW.id,'delete',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'deleted',NEW.deleted,'last_modified',NEW.last_modified,
                     'version',NEW.version,'client_id',NEW.client_id));
END;

CREATE TRIGGER writing_sessions_sync_undelete
AFTER UPDATE ON writing_sessions
WHEN OLD.deleted = 1 AND NEW.deleted = 0
BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('writing_sessions',NEW.id,'update',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'name',NEW.name,'payload_json',NEW.payload_json,
                     'schema_version',NEW.schema_version,'version_parent_id',NEW.version_parent_id,
                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

DROP TRIGGER IF EXISTS writing_templates_sync_create;
DROP TRIGGER IF EXISTS writing_templates_sync_update;
DROP TRIGGER IF EXISTS writing_templates_sync_delete;
DROP TRIGGER IF EXISTS writing_templates_sync_undelete;

CREATE TRIGGER writing_templates_sync_create
AFTER INSERT ON writing_templates BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('writing_templates',CAST(NEW.id AS TEXT),'create',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'name',NEW.name,'payload_json',NEW.payload_json,
                     'schema_version',NEW.schema_version,'version_parent_id',NEW.version_parent_id,
                     'is_default',NEW.is_default,'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER writing_templates_sync_update
AFTER UPDATE ON writing_templates
WHEN OLD.deleted = NEW.deleted AND (
     OLD.name IS NOT NEW.name OR
     OLD.payload_json IS NOT NEW.payload_json OR
     OLD.schema_version IS NOT NEW.schema_version OR
     OLD.version_parent_id IS NOT NEW.version_parent_id OR
     OLD.is_default IS NOT NEW.is_default OR
     OLD.last_modified IS NOT NEW.last_modified OR
     OLD.version IS NOT NEW.version)
BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('writing_templates',CAST(NEW.id AS TEXT),'update',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'name',NEW.name,'payload_json',NEW.payload_json,
                     'schema_version',NEW.schema_version,'version_parent_id',NEW.version_parent_id,
                     'is_default',NEW.is_default,'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER writing_templates_sync_delete
AFTER UPDATE ON writing_templates
WHEN OLD.deleted = 0 AND NEW.deleted = 1
BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('writing_templates',CAST(NEW.id AS TEXT),'delete',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'deleted',NEW.deleted,'last_modified',NEW.last_modified,
                     'version',NEW.version,'client_id',NEW.client_id));
END;

CREATE TRIGGER writing_templates_sync_undelete
AFTER UPDATE ON writing_templates
WHEN OLD.deleted = 1 AND NEW.deleted = 0
BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('writing_templates',CAST(NEW.id AS TEXT),'update',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'name',NEW.name,'payload_json',NEW.payload_json,
                     'schema_version',NEW.schema_version,'version_parent_id',NEW.version_parent_id,
                     'is_default',NEW.is_default,'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

DROP TRIGGER IF EXISTS writing_themes_sync_create;
DROP TRIGGER IF EXISTS writing_themes_sync_update;
DROP TRIGGER IF EXISTS writing_themes_sync_delete;
DROP TRIGGER IF EXISTS writing_themes_sync_undelete;

CREATE TRIGGER writing_themes_sync_create
AFTER INSERT ON writing_themes BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('writing_themes',CAST(NEW.id AS TEXT),'create',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'name',NEW.name,'class_name',NEW.class_name,'css',NEW.css,
                     'schema_version',NEW.schema_version,'version_parent_id',NEW.version_parent_id,
                     'is_default',NEW.is_default,'order_index',NEW.order_index,
                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER writing_themes_sync_update
AFTER UPDATE ON writing_themes
WHEN OLD.deleted = NEW.deleted AND (
     OLD.name IS NOT NEW.name OR
     OLD.class_name IS NOT NEW.class_name OR
     OLD.css IS NOT NEW.css OR
     OLD.schema_version IS NOT NEW.schema_version OR
     OLD.version_parent_id IS NOT NEW.version_parent_id OR
     OLD.is_default IS NOT NEW.is_default OR
     OLD.order_index IS NOT NEW.order_index OR
     OLD.last_modified IS NOT NEW.last_modified OR
     OLD.version IS NOT NEW.version)
BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('writing_themes',CAST(NEW.id AS TEXT),'update',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'name',NEW.name,'class_name',NEW.class_name,'css',NEW.css,
                     'schema_version',NEW.schema_version,'version_parent_id',NEW.version_parent_id,
                     'is_default',NEW.is_default,'order_index',NEW.order_index,
                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER writing_themes_sync_delete
AFTER UPDATE ON writing_themes
WHEN OLD.deleted = 0 AND NEW.deleted = 1
BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('writing_themes',CAST(NEW.id AS TEXT),'delete',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'deleted',NEW.deleted,'last_modified',NEW.last_modified,
                     'version',NEW.version,'client_id',NEW.client_id));
END;

CREATE TRIGGER writing_themes_sync_undelete
AFTER UPDATE ON writing_themes
WHEN OLD.deleted = 1 AND NEW.deleted = 0
BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('writing_themes',CAST(NEW.id AS TEXT),'update',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'name',NEW.name,'class_name',NEW.class_name,'css',NEW.css,
                     'schema_version',NEW.schema_version,'version_parent_id',NEW.version_parent_id,
                     'is_default',NEW.is_default,'order_index',NEW.order_index,
                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

UPDATE db_schema_version
   SET version = 15
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 15;
"""

# --- Migration: V15 -> V16 (Writing Wordclouds) ---
_MIGRATION_SQL_V15_TO_V16 = """
/*───────────────────────────────────────────────────────────────
  Migration to Version 16 - Writing Wordclouds (2026-01-24)
───────────────────────────────────────────────────────────────*/
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS writing_wordclouds(
  id TEXT PRIMARY KEY,
  status TEXT NOT NULL DEFAULT 'queued',
  options_json TEXT NOT NULL,
  words_json TEXT,
  meta_json TEXT,
  error TEXT,
  input_chars INTEGER NOT NULL DEFAULT 0,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  client_id TEXT NOT NULL DEFAULT 'unknown'
);
CREATE INDEX IF NOT EXISTS idx_writing_wordclouds_status ON writing_wordclouds(status);
CREATE INDEX IF NOT EXISTS idx_writing_wordclouds_last_modified ON writing_wordclouds(last_modified);

UPDATE db_schema_version
   SET version = 16
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 16;
"""

# --- Migration: V16 -> V17 (Workspace tag for quizzes) ---
_MIGRATION_SQL_V16_TO_V17 = """
/*───────────────────────────────────────────────────────────────
  Migration to Version 17 - Quizzes workspace_tag (2026-02-18)
───────────────────────────────────────────────────────────────*/
PRAGMA foreign_keys = ON;

ALTER TABLE quizzes ADD COLUMN IF NOT EXISTS workspace_tag TEXT;
CREATE INDEX IF NOT EXISTS idx_quizzes_workspace_tag ON quizzes(workspace_tag);

UPDATE db_schema_version
   SET version = 17
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 17;
"""

_MIGRATION_SQL_V17_TO_V18 = """
/*───────────────────────────────────────────────────────────────
  Migration to Version 18 - Voice Assistant Tables (2026-01-26)
───────────────────────────────────────────────────────────────*/
PRAGMA foreign_keys = ON;

-- Voice Commands: User-defined voice command triggers
CREATE TABLE IF NOT EXISTS voice_commands (
    id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    phrases TEXT NOT NULL,        -- JSON array of trigger phrases
    action_type TEXT NOT NULL,    -- mcp_tool|workflow|custom|llm_chat
    action_config TEXT NOT NULL,  -- JSON configuration
    priority INTEGER DEFAULT 0,
    enabled INTEGER DEFAULT 1,
    requires_confirmation INTEGER DEFAULT 0,
    description TEXT,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_voice_commands_user_id ON voice_commands(user_id);
CREATE INDEX IF NOT EXISTS idx_voice_commands_enabled ON voice_commands(enabled, deleted);

-- Voice Sessions: Track voice assistant session state
CREATE TABLE IF NOT EXISTS voice_sessions (
    session_id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL,
    conversation_id TEXT REFERENCES conversations(id) ON DELETE SET NULL,
    state TEXT NOT NULL DEFAULT 'idle',  -- idle|listening|processing|speaking|awaiting_confirmation|error
    context TEXT,                 -- JSON session context
    conversation_history TEXT,    -- JSON array of turns
    pending_intent TEXT,          -- JSON pending intent awaiting confirmation
    last_action_result TEXT,      -- JSON last action result
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_activity DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_voice_sessions_user_id ON voice_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_voice_sessions_last_activity ON voice_sessions(last_activity);

-- Sync log triggers for voice_commands
DROP TRIGGER IF EXISTS voice_commands_insert_sync;
DROP TRIGGER IF EXISTS voice_commands_update_sync;

CREATE TRIGGER IF NOT EXISTS voice_commands_insert_sync
AFTER INSERT ON voice_commands
BEGIN
    INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
    VALUES(
        'voice_commands',
        NEW.id,
        'create',
        NEW.created_at,
        'system',
        1,
        json_object(
            'id',NEW.id,
            'user_id',NEW.user_id,
            'name',NEW.name,
            'phrases',NEW.phrases,
            'action_type',NEW.action_type,
            'action_config',NEW.action_config,
            'priority',NEW.priority,
            'enabled',NEW.enabled,
            'requires_confirmation',NEW.requires_confirmation,
            'description',NEW.description,
            'created_at',NEW.created_at,
            'updated_at',NEW.updated_at,
            'deleted',NEW.deleted
        )
    );
END;

CREATE TRIGGER IF NOT EXISTS voice_commands_update_sync
AFTER UPDATE ON voice_commands
BEGIN
    INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
    VALUES(
        'voice_commands',
        NEW.id,
        'update',
        NEW.updated_at,
        'system',
        1,
        json_object(
            'id',NEW.id,
            'user_id',NEW.user_id,
            'name',NEW.name,
            'phrases',NEW.phrases,
            'action_type',NEW.action_type,
            'action_config',NEW.action_config,
            'priority',NEW.priority,
            'enabled',NEW.enabled,
            'requires_confirmation',NEW.requires_confirmation,
            'description',NEW.description,
            'created_at',NEW.created_at,
            'updated_at',NEW.updated_at,
            'deleted',NEW.deleted
        )
    );
END;

UPDATE db_schema_version
   SET version = 18
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 18;
"""

_MIGRATION_SQL_V18_TO_V19 = """
/*───────────────────────────────────────────────────────────────
  Migration to Version 19 - Voice Assistant Analytics (2026-01-27)
───────────────────────────────────────────────────────────────*/
PRAGMA foreign_keys = ON;

-- Voice Command Events: Track usage for analytics
CREATE TABLE IF NOT EXISTS voice_command_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    command_id TEXT,
    command_name TEXT,
    user_id INTEGER NOT NULL,
    action_type TEXT NOT NULL,
    success INTEGER DEFAULT 1,
    response_time_ms REAL,
    session_id TEXT,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_voice_command_events_user_time
    ON voice_command_events(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_voice_command_events_command_time
    ON voice_command_events(command_id, created_at);

UPDATE db_schema_version
   SET version = 19
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 19;
"""

_MIGRATION_SQL_V19_TO_V20 = """
/*───────────────────────────────────────────────────────────────
  Migration to Version 20 - Skills registry (2026-02-03)
───────────────────────────────────────────────────────────────*/
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS skill_registry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    argument_hint TEXT,
    disable_model_invocation BOOLEAN DEFAULT 0,
    user_invocable BOOLEAN DEFAULT 1,
    allowed_tools TEXT,
    model TEXT,
    context TEXT DEFAULT 'inline',
    directory_path TEXT NOT NULL,
    last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    file_hash TEXT,
    uuid TEXT UNIQUE NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    deleted BOOLEAN DEFAULT 0,
    version INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_skill_name ON skill_registry(name);
CREATE INDEX IF NOT EXISTS idx_skill_user_invocable ON skill_registry(user_invocable);
CREATE INDEX IF NOT EXISTS idx_skill_deleted ON skill_registry(deleted);

UPDATE db_schema_version
   SET version = 20
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 20;
"""

_MIGRATION_SQL_V20_TO_V21 = """
/*───────────────────────────────────────────────────────────────
  Migration to Version 21 - Conversation settings (2026-02-05)
───────────────────────────────────────────────────────────────*/
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS conversation_settings(
  conversation_id TEXT PRIMARY KEY REFERENCES conversations(id) ON DELETE CASCADE,
  settings_json TEXT NOT NULL,
  last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

UPDATE db_schema_version
   SET version = 21
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 21;
"""

_MIGRATION_SQL_V21_TO_V22 = """
/*───────────────────────────────────────────────────────────────
  Migration to Version 22 - Character exemplars (2026-02-08)
───────────────────────────────────────────────────────────────*/
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS character_exemplars(
  id TEXT PRIMARY KEY,
  character_id INTEGER NOT NULL REFERENCES character_cards(id) ON DELETE CASCADE,
  text TEXT NOT NULL,
  source_type TEXT NOT NULL DEFAULT 'other'
    CHECK(source_type IN ('audio_transcript','video_transcript','article','other')),
  source_url_or_id TEXT,
  source_date TEXT,
  novelty_hint TEXT NOT NULL DEFAULT 'unknown'
    CHECK(novelty_hint IN ('post_cutoff','unknown','pre_cutoff')),
  emotion TEXT NOT NULL DEFAULT 'other'
    CHECK(emotion IN ('angry','neutral','happy','other')),
  scenario TEXT NOT NULL DEFAULT 'other'
    CHECK(scenario IN ('press_challenge','fan_banter','debate','boardroom','small_talk','other')),
  rhetorical TEXT NOT NULL DEFAULT '[]',
  register TEXT,
  safety_allowed TEXT NOT NULL DEFAULT '[]',
  safety_blocked TEXT NOT NULL DEFAULT '[]',
  rights_public_figure BOOLEAN NOT NULL DEFAULT 1,
  rights_notes TEXT,
  length_tokens INTEGER NOT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  is_deleted BOOLEAN NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_character_exemplars_character ON character_exemplars(character_id);
CREATE INDEX IF NOT EXISTS idx_character_exemplars_scenario_emotion ON character_exemplars(scenario, emotion);
CREATE INDEX IF NOT EXISTS idx_character_exemplars_novelty ON character_exemplars(novelty_hint);

CREATE VIRTUAL TABLE IF NOT EXISTS character_exemplars_fts
USING fts5(
  text, emotion, scenario,
  content='character_exemplars',
  content_rowid='rowid'
);

DROP TRIGGER IF EXISTS character_exemplars_ai;
DROP TRIGGER IF EXISTS character_exemplars_au;
DROP TRIGGER IF EXISTS character_exemplars_ad;

CREATE TRIGGER character_exemplars_ai
AFTER INSERT ON character_exemplars BEGIN
  INSERT INTO character_exemplars_fts(rowid, text, emotion, scenario)
  SELECT new.rowid, new.text, new.emotion, new.scenario
  WHERE new.is_deleted = 0;
END;

CREATE TRIGGER character_exemplars_au
AFTER UPDATE ON character_exemplars BEGIN
  INSERT INTO character_exemplars_fts(character_exemplars_fts, rowid, text, emotion, scenario)
  VALUES('delete', old.rowid, old.text, old.emotion, old.scenario);

  INSERT INTO character_exemplars_fts(rowid, text, emotion, scenario)
  SELECT new.rowid, new.text, new.emotion, new.scenario
  WHERE new.is_deleted = 0;
END;

CREATE TRIGGER character_exemplars_ad
AFTER DELETE ON character_exemplars BEGIN
  INSERT INTO character_exemplars_fts(character_exemplars_fts, rowid, text, emotion, scenario)
  VALUES('delete', old.rowid, old.text, old.emotion, old.scenario);
END;

INSERT INTO character_exemplars_fts(character_exemplars_fts) VALUES('rebuild');

UPDATE db_schema_version
   SET version = 22
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 22;
"""

_MIGRATION_SQL_V22_TO_V23 = """
/*───────────────────────────────────────────────────────────────
  Migration to Version 23 - Quiz multi-select question type (2026-02-18)
───────────────────────────────────────────────────────────────*/
PRAGMA foreign_keys = OFF;

DROP TRIGGER IF EXISTS quiz_questions_ai;
DROP TRIGGER IF EXISTS quiz_questions_au;
DROP TRIGGER IF EXISTS quiz_questions_ad;

CREATE TABLE IF NOT EXISTS quiz_questions_new (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  quiz_id INTEGER NOT NULL REFERENCES quizzes(id) ON DELETE CASCADE,
  question_type TEXT NOT NULL CHECK(question_type IN ('multiple_choice', 'multi_select', 'true_false', 'fill_blank')),
  question_text TEXT NOT NULL,
  options TEXT,
  correct_answer TEXT NOT NULL,
  explanation TEXT,
  points INTEGER NOT NULL DEFAULT 1,
  order_index INTEGER NOT NULL DEFAULT 0,
  tags_json TEXT,
  deleted BOOLEAN NOT NULL DEFAULT 0,
  client_id TEXT NOT NULL DEFAULT 'unknown',
  version INTEGER NOT NULL DEFAULT 1,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO quiz_questions_new(
  id, quiz_id, question_type, question_text, options, correct_answer, explanation,
  points, order_index, tags_json, deleted, client_id, version, created_at, last_modified
)
SELECT
  id, quiz_id, question_type, question_text, options, correct_answer, explanation,
  points, order_index, tags_json, deleted, client_id, version, created_at, last_modified
FROM quiz_questions;

DROP TABLE IF EXISTS quiz_questions;
ALTER TABLE quiz_questions_new RENAME TO quiz_questions;

CREATE INDEX IF NOT EXISTS idx_quiz_questions_quiz_id ON quiz_questions(quiz_id);
CREATE INDEX IF NOT EXISTS idx_quiz_questions_deleted ON quiz_questions(deleted);
CREATE INDEX IF NOT EXISTS idx_quiz_questions_order ON quiz_questions(quiz_id, order_index);

DROP TABLE IF EXISTS quiz_questions_fts;
CREATE VIRTUAL TABLE IF NOT EXISTS quiz_questions_fts
USING fts5(
  question_text, explanation,
  content='quiz_questions',
  content_rowid='id'
);

CREATE TRIGGER quiz_questions_ai
AFTER INSERT ON quiz_questions BEGIN
  INSERT INTO quiz_questions_fts(rowid, question_text, explanation)
  SELECT new.id, new.question_text, new.explanation
  WHERE new.deleted = 0;
END;

CREATE TRIGGER quiz_questions_au
AFTER UPDATE ON quiz_questions BEGIN
  INSERT INTO quiz_questions_fts(quiz_questions_fts,rowid,question_text,explanation)
  VALUES('delete',old.id,old.question_text,old.explanation);

  INSERT INTO quiz_questions_fts(rowid, question_text, explanation)
  SELECT new.id, new.question_text, new.explanation
  WHERE new.deleted = 0;
END;

CREATE TRIGGER quiz_questions_ad
AFTER DELETE ON quiz_questions BEGIN
  INSERT INTO quiz_questions_fts(quiz_questions_fts,rowid,question_text,explanation)
  VALUES('delete',old.id,old.question_text,old.explanation);
END;

INSERT INTO quiz_questions_fts(quiz_questions_fts) VALUES('rebuild');

PRAGMA foreign_keys = ON;

UPDATE db_schema_version
   SET version = 23
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 23;
"""

_MIGRATION_SQL_V23_TO_V24 = """
/*───────────────────────────────────────────────────────────────
  Migration to Version 24 - Quiz matching question type (2026-02-18)
───────────────────────────────────────────────────────────────*/
PRAGMA foreign_keys = OFF;

DROP TRIGGER IF EXISTS quiz_questions_ai;
DROP TRIGGER IF EXISTS quiz_questions_au;
DROP TRIGGER IF EXISTS quiz_questions_ad;

CREATE TABLE IF NOT EXISTS quiz_questions_new (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  quiz_id INTEGER NOT NULL REFERENCES quizzes(id) ON DELETE CASCADE,
  question_type TEXT NOT NULL CHECK(question_type IN ('multiple_choice', 'multi_select', 'matching', 'true_false', 'fill_blank')),
  question_text TEXT NOT NULL,
  options TEXT,
  correct_answer TEXT NOT NULL,
  explanation TEXT,
  points INTEGER NOT NULL DEFAULT 1,
  order_index INTEGER NOT NULL DEFAULT 0,
  tags_json TEXT,
  deleted BOOLEAN NOT NULL DEFAULT 0,
  client_id TEXT NOT NULL DEFAULT 'unknown',
  version INTEGER NOT NULL DEFAULT 1,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO quiz_questions_new(
  id, quiz_id, question_type, question_text, options, correct_answer, explanation,
  points, order_index, tags_json, deleted, client_id, version, created_at, last_modified
)
SELECT
  id, quiz_id, question_type, question_text, options, correct_answer, explanation,
  points, order_index, tags_json, deleted, client_id, version, created_at, last_modified
FROM quiz_questions;

DROP TABLE IF EXISTS quiz_questions;
ALTER TABLE quiz_questions_new RENAME TO quiz_questions;

CREATE INDEX IF NOT EXISTS idx_quiz_questions_quiz_id ON quiz_questions(quiz_id);
CREATE INDEX IF NOT EXISTS idx_quiz_questions_deleted ON quiz_questions(deleted);
CREATE INDEX IF NOT EXISTS idx_quiz_questions_order ON quiz_questions(quiz_id, order_index);

DROP TABLE IF EXISTS quiz_questions_fts;
CREATE VIRTUAL TABLE IF NOT EXISTS quiz_questions_fts
USING fts5(
  question_text, explanation,
  content='quiz_questions',
  content_rowid='id'
);

CREATE TRIGGER quiz_questions_ai
AFTER INSERT ON quiz_questions BEGIN
  INSERT INTO quiz_questions_fts(rowid, question_text, explanation)
  SELECT new.id, new.question_text, new.explanation
  WHERE new.deleted = 0;
END;

CREATE TRIGGER quiz_questions_au
AFTER UPDATE ON quiz_questions BEGIN
  INSERT INTO quiz_questions_fts(quiz_questions_fts,rowid,question_text,explanation)
  VALUES('delete',old.id,old.question_text,old.explanation);

  INSERT INTO quiz_questions_fts(rowid, question_text, explanation)
  SELECT new.id, new.question_text, new.explanation
  WHERE new.deleted = 0;
END;

CREATE TRIGGER quiz_questions_ad
AFTER DELETE ON quiz_questions BEGIN
  INSERT INTO quiz_questions_fts(quiz_questions_fts,rowid,question_text,explanation)
  VALUES('delete',old.id,old.question_text,old.explanation);
END;

INSERT INTO quiz_questions_fts(quiz_questions_fts) VALUES('rebuild');

PRAGMA foreign_keys = ON;

UPDATE db_schema_version
   SET version = 24
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 24;
"""

_MIGRATION_SQL_V24_TO_V25 = """
/*───────────────────────────────────────────────────────────────
  Migration to Version 25 - Quiz hint metadata and source citations (2026-02-18)
───────────────────────────────────────────────────────────────*/
ALTER TABLE quiz_questions
  ADD COLUMN IF NOT EXISTS hint TEXT;

ALTER TABLE quiz_questions
  ADD COLUMN IF NOT EXISTS hint_penalty_points INTEGER NOT NULL DEFAULT 0;

ALTER TABLE quiz_questions
  ADD COLUMN IF NOT EXISTS source_citations_json TEXT;

UPDATE db_schema_version
   SET version = 25
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 25;
"""

_MIGRATION_SQL_V25_TO_V26 = """
/*───────────────────────────────────────────────────────────────
  Migration to Version 26 - Persona profile/scoping persistence (2026-02-22)
───────────────────────────────────────────────────────────────*/
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS persona_profiles(
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  name TEXT NOT NULL,
  character_card_id INTEGER REFERENCES character_cards(id) ON DELETE SET NULL ON UPDATE CASCADE,
  mode TEXT NOT NULL DEFAULT 'session_scoped'
    CHECK(mode IN ('session_scoped','persistent_scoped')),
  system_prompt TEXT,
  is_active BOOLEAN NOT NULL DEFAULT 1,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted BOOLEAN NOT NULL DEFAULT 0,
  version INTEGER NOT NULL DEFAULT 1,
  UNIQUE(user_id, name)
);

CREATE INDEX IF NOT EXISTS idx_persona_profiles_user ON persona_profiles(user_id);
CREATE INDEX IF NOT EXISTS idx_persona_profiles_user_active ON persona_profiles(user_id, deleted, is_active);
CREATE INDEX IF NOT EXISTS idx_persona_profiles_character ON persona_profiles(character_card_id);

CREATE TABLE IF NOT EXISTS persona_scope_rules(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  persona_id TEXT NOT NULL REFERENCES persona_profiles(id) ON DELETE CASCADE,
  user_id TEXT NOT NULL,
  rule_type TEXT NOT NULL
    CHECK(rule_type IN ('conversation_id','character_id','media_id','media_tag','note_id')),
  rule_value TEXT NOT NULL,
  include BOOLEAN NOT NULL DEFAULT 1,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted BOOLEAN NOT NULL DEFAULT 0,
  version INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_persona_scope_rules_persona ON persona_scope_rules(persona_id, deleted, include);
CREATE INDEX IF NOT EXISTS idx_persona_scope_rules_user_type ON persona_scope_rules(user_id, rule_type, deleted);
CREATE INDEX IF NOT EXISTS idx_persona_scope_rules_value ON persona_scope_rules(rule_type, rule_value);

CREATE TABLE IF NOT EXISTS persona_policy_rules(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  persona_id TEXT NOT NULL REFERENCES persona_profiles(id) ON DELETE CASCADE,
  user_id TEXT NOT NULL,
  rule_kind TEXT NOT NULL CHECK(rule_kind IN ('mcp_tool','skill')),
  rule_name TEXT NOT NULL,
  allowed BOOLEAN NOT NULL DEFAULT 1,
  require_confirmation BOOLEAN NOT NULL DEFAULT 0,
  max_calls_per_turn INTEGER,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted BOOLEAN NOT NULL DEFAULT 0,
  version INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_persona_policy_rules_persona ON persona_policy_rules(persona_id, deleted, rule_kind);
CREATE INDEX IF NOT EXISTS idx_persona_policy_rules_user ON persona_policy_rules(user_id, deleted);
CREATE INDEX IF NOT EXISTS idx_persona_policy_rules_lookup ON persona_policy_rules(rule_kind, rule_name, deleted);

CREATE TABLE IF NOT EXISTS persona_sessions(
  id TEXT PRIMARY KEY,
  persona_id TEXT NOT NULL REFERENCES persona_profiles(id) ON DELETE CASCADE,
  user_id TEXT NOT NULL,
  conversation_id TEXT REFERENCES conversations(id) ON DELETE SET NULL,
  mode TEXT NOT NULL DEFAULT 'session_scoped'
    CHECK(mode IN ('session_scoped','persistent_scoped')),
  reuse_allowed BOOLEAN NOT NULL DEFAULT 0,
  status TEXT NOT NULL DEFAULT 'active'
    CHECK(status IN ('active','paused','closed','archived')),
  scope_snapshot_json TEXT NOT NULL DEFAULT '{}',
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted BOOLEAN NOT NULL DEFAULT 0,
  version INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_persona_sessions_user ON persona_sessions(user_id, deleted, status);
CREATE INDEX IF NOT EXISTS idx_persona_sessions_persona ON persona_sessions(persona_id, deleted, status);
CREATE INDEX IF NOT EXISTS idx_persona_sessions_conversation ON persona_sessions(conversation_id);
CREATE INDEX IF NOT EXISTS idx_persona_sessions_last_modified ON persona_sessions(last_modified);

CREATE TABLE IF NOT EXISTS persona_memory_entries(
  id TEXT PRIMARY KEY,
  persona_id TEXT NOT NULL REFERENCES persona_profiles(id) ON DELETE CASCADE,
  user_id TEXT NOT NULL,
  memory_type TEXT NOT NULL,
  content TEXT NOT NULL,
  source_conversation_id TEXT REFERENCES conversations(id) ON DELETE SET NULL,
  salience REAL NOT NULL DEFAULT 0,
  archived BOOLEAN NOT NULL DEFAULT 0,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted BOOLEAN NOT NULL DEFAULT 0,
  version INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_persona_memory_persona ON persona_memory_entries(persona_id, archived, deleted);
CREATE INDEX IF NOT EXISTS idx_persona_memory_user ON persona_memory_entries(user_id, archived, deleted);
CREATE INDEX IF NOT EXISTS idx_persona_memory_source ON persona_memory_entries(source_conversation_id);
CREATE INDEX IF NOT EXISTS idx_persona_memory_last_modified ON persona_memory_entries(last_modified);

UPDATE db_schema_version
   SET version = 26
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 26;
"""

_MIGRATION_SQL_V26_TO_V27 = """
/*───────────────────────────────────────────────────────────────
  Migration to Version 27 - Persona memory namespace keying (2026-02-22)
───────────────────────────────────────────────────────────────*/
ALTER TABLE persona_memory_entries
  ADD COLUMN IF NOT EXISTS scope_snapshot_id TEXT;

ALTER TABLE persona_memory_entries
  ADD COLUMN IF NOT EXISTS session_id TEXT;

CREATE INDEX IF NOT EXISTS idx_persona_memory_scope
  ON persona_memory_entries(user_id, persona_id, scope_snapshot_id, archived, deleted);

CREATE INDEX IF NOT EXISTS idx_persona_memory_session
  ON persona_memory_entries(user_id, persona_id, session_id, archived, deleted);

UPDATE db_schema_version
   SET version = 27
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 27;
"""

_MIGRATION_SQL_V27_TO_V28 = """
/*───────────────────────────────────────────────────────────────
  Migration to Version 28 - Persona profile state-context defaults (2026-02-22)
───────────────────────────────────────────────────────────────*/
ALTER TABLE persona_profiles
  ADD COLUMN IF NOT EXISTS use_persona_state_context_default BOOLEAN NOT NULL DEFAULT 1;

UPDATE db_schema_version
   SET version = 28
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 28;
"""

_MIGRATION_SQL_V28_TO_V29 = """
/*───────────────────────────────────────────────────────────────
  Migration to Version 29 - Notes moodboards and board memberships (2026-02-26)
───────────────────────────────────────────────────────────────*/
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS moodboards(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  description TEXT,
  smart_rule_json TEXT,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted BOOLEAN NOT NULL DEFAULT 0,
  client_id TEXT NOT NULL DEFAULT 'unknown',
  version INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_moodboards_last_modified ON moodboards(last_modified);
CREATE INDEX IF NOT EXISTS idx_moodboards_deleted ON moodboards(deleted);

CREATE TABLE IF NOT EXISTS moodboard_notes(
  moodboard_id INTEGER NOT NULL REFERENCES moodboards(id) ON DELETE CASCADE ON UPDATE CASCADE,
  note_id TEXT NOT NULL REFERENCES notes(id) ON DELETE CASCADE ON UPDATE CASCADE,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY(moodboard_id, note_id)
);

CREATE INDEX IF NOT EXISTS idx_moodboard_notes_board ON moodboard_notes(moodboard_id);
CREATE INDEX IF NOT EXISTS idx_moodboard_notes_note ON moodboard_notes(note_id);

UPDATE db_schema_version
   SET version = 29
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 29;
"""

_MIGRATION_SQL_V29_TO_V30 = """
/*───────────────────────────────────────────────────────────────
  Migration to Version 30 - Quiz source bundle metadata (2026-03-03)
───────────────────────────────────────────────────────────────*/
ALTER TABLE quizzes
  ADD COLUMN IF NOT EXISTS source_bundle_json TEXT;

UPDATE db_schema_version
   SET version = 30
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 30;
"""

_MIGRATION_SQL_V30_TO_V31 = """
/*───────────────────────────────────────────────────────────────
  Migration to Version 31 - Persona origin snapshots (2026-03-07)
───────────────────────────────────────────────────────────────*/
ALTER TABLE persona_profiles
  ADD COLUMN IF NOT EXISTS origin_character_id INTEGER;

ALTER TABLE persona_profiles
  ADD COLUMN IF NOT EXISTS origin_character_name TEXT;

ALTER TABLE persona_profiles
  ADD COLUMN IF NOT EXISTS origin_character_snapshot_at TEXT;

UPDATE db_schema_version
   SET version = 31
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 31;
"""

_MIGRATION_SQL_V31_TO_V32 = """
/*───────────────────────────────────────────────────────────────
  Migration to Version 32 - Conversation assistant identity (2026-03-08)
───────────────────────────────────────────────────────────────*/
ALTER TABLE conversations
  ADD COLUMN IF NOT EXISTS assistant_kind TEXT CHECK (assistant_kind IN ('character', 'persona'));

ALTER TABLE conversations
  ADD COLUMN IF NOT EXISTS assistant_id TEXT;

ALTER TABLE conversations
  ADD COLUMN IF NOT EXISTS persona_memory_mode TEXT CHECK (persona_memory_mode IN ('read_only', 'read_write'));

UPDATE conversations
   SET assistant_kind = 'character'
 WHERE character_id IS NOT NULL
   AND COALESCE(TRIM(assistant_kind), '') = '';

UPDATE conversations
   SET assistant_id = CAST(character_id AS TEXT)
 WHERE character_id IS NOT NULL
   AND COALESCE(TRIM(assistant_id), '') = '';

UPDATE db_schema_version
   SET version = 32
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 32;
"""

_MIGRATION_SQL_V32_TO_V33 = """
/*───────────────────────────────────────────────────────────────
  Migration to Version 33 - Persona exemplars (2026-03-08)
───────────────────────────────────────────────────────────────*/
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS persona_exemplars(
  id TEXT PRIMARY KEY,
  persona_id TEXT NOT NULL REFERENCES persona_profiles(id) ON DELETE CASCADE,
  user_id TEXT NOT NULL,
  kind TEXT NOT NULL
    CHECK(kind IN ('style','catchphrase','boundary','scenario_demo','tool_behavior')),
  content TEXT NOT NULL,
  tone TEXT,
  scenario_tags_json TEXT NOT NULL DEFAULT '[]',
  capability_tags_json TEXT NOT NULL DEFAULT '[]',
  priority INTEGER NOT NULL DEFAULT 0,
  enabled BOOLEAN NOT NULL DEFAULT 1,
  source_type TEXT NOT NULL DEFAULT 'manual'
    CHECK(source_type IN ('manual','transcript_import','character_seed','generated_candidate')),
  source_ref TEXT,
  notes TEXT,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted BOOLEAN NOT NULL DEFAULT 0,
  version INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_persona_exemplars_persona
  ON persona_exemplars(persona_id, deleted, enabled);
CREATE INDEX IF NOT EXISTS idx_persona_exemplars_user
  ON persona_exemplars(user_id, deleted, enabled);
CREATE INDEX IF NOT EXISTS idx_persona_exemplars_kind
  ON persona_exemplars(kind, deleted);
CREATE INDEX IF NOT EXISTS idx_persona_exemplars_enabled
  ON persona_exemplars(enabled, deleted);

UPDATE db_schema_version
   SET version = 33
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 33;
"""

_MIGRATION_SQL_V32_TO_V33_POSTGRES = _MIGRATION_SQL_V32_TO_V33.replace(
    "PRAGMA foreign_keys = ON;\n\n",
    "",
)

_MIGRATION_SQL_V33_TO_V34 = """
/*───────────────────────────────────────────────────────────────
  Migration to Version 34 - Persona session activity surfaces (2026-03-10)
───────────────────────────────────────────────────────────────*/
ALTER TABLE persona_sessions
  ADD COLUMN IF NOT EXISTS activity_surface TEXT NOT NULL DEFAULT 'api.persona';

UPDATE db_schema_version
   SET version = 34
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 34;
"""

_MIGRATION_SQL_V34_TO_V35 = """
/*───────────────────────────────────────────────────────────────
  Migration to Version 35 - Persona session preferences (2026-03-10)
───────────────────────────────────────────────────────────────*/
ALTER TABLE persona_sessions
  ADD COLUMN IF NOT EXISTS preferences_json TEXT NOT NULL DEFAULT '{}';

UPDATE db_schema_version
   SET version = 35
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 35;
"""
_MIGRATION_SQL_V35_TO_V36_POSTGRES = """
/*───────────────────────────────────────────────────────────────
  Migration to Version 36 - Workspaces table + conversation scope (2026-03-12)
───────────────────────────────────────────────────────────────*/
CREATE TABLE IF NOT EXISTS workspaces (
    id            TEXT    PRIMARY KEY NOT NULL,
    name          TEXT    NOT NULL,
    description   TEXT,
    metadata_json TEXT    NOT NULL DEFAULT '{}',
    study_materials_policy TEXT NOT NULL DEFAULT 'general',
    workspace_profile TEXT NOT NULL DEFAULT 'research',
    archived      BOOLEAN NOT NULL DEFAULT false,
    created_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_modified TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted       BOOLEAN NOT NULL DEFAULT false,
    client_id     TEXT    NOT NULL DEFAULT 'unknown',
    version       INTEGER NOT NULL DEFAULT 1,
    assistant_defaults_json TEXT
);

ALTER TABLE conversations
  ADD COLUMN IF NOT EXISTS scope_type TEXT NOT NULL DEFAULT 'global';

ALTER TABLE conversations
  ADD COLUMN IF NOT EXISTS workspace_id TEXT REFERENCES workspaces(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_conversations_scope ON conversations(scope_type, workspace_id);

UPDATE db_schema_version
   SET version = 36
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 36;
"""
_MIGRATION_SQL_V36_TO_V37 = """
/*───────────────────────────────────────────────────────────────
  Migration to Version 37 - Workspace sub-resources + flashcard scheduler state (2026-03-13)
───────────────────────────────────────────────────────────────*/
UPDATE db_schema_version
   SET version = 37
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 37;
"""
_MIGRATION_SQL_V36_TO_V37_POSTGRES = """
/*───────────────────────────────────────────────────────────────
  Migration to Version 37 - Workspace sub-resources + flashcard scheduler state (2026-03-13)
───────────────────────────────────────────────────────────────*/
ALTER TABLE workspaces ADD COLUMN IF NOT EXISTS tag TEXT;
ALTER TABLE workspaces ADD COLUMN IF NOT EXISTS banner_title TEXT;
ALTER TABLE workspaces ADD COLUMN IF NOT EXISTS banner_subtitle TEXT;
ALTER TABLE workspaces ADD COLUMN IF NOT EXISTS banner_color TEXT;
ALTER TABLE workspaces ADD COLUMN IF NOT EXISTS audio_provider TEXT;
ALTER TABLE workspaces ADD COLUMN IF NOT EXISTS audio_model TEXT;
ALTER TABLE workspaces ADD COLUMN IF NOT EXISTS audio_voice TEXT;
ALTER TABLE workspaces ADD COLUMN IF NOT EXISTS audio_speed REAL DEFAULT 1.0;

CREATE TABLE IF NOT EXISTS workspace_sources (
    id            TEXT    NOT NULL,
    workspace_id  TEXT    NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    media_id      INTEGER NOT NULL,
    title         TEXT    NOT NULL,
    source_type   TEXT    NOT NULL,
    url           TEXT,
    position      INTEGER NOT NULL DEFAULT 0,
    selected      BOOLEAN NOT NULL DEFAULT true,
    added_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    review_state  TEXT NOT NULL DEFAULT 'unset',
    review_state_updated_at TEXT,
    reviewed_at   TEXT,
    reviewed_by_user_id TEXT,
    version       INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (workspace_id, id)
);
CREATE INDEX IF NOT EXISTS idx_ws_sources_workspace ON workspace_sources(workspace_id);
ALTER TABLE workspace_sources ADD COLUMN IF NOT EXISTS review_state TEXT NOT NULL DEFAULT 'unset';
ALTER TABLE workspace_sources ADD COLUMN IF NOT EXISTS review_state_updated_at TEXT;
ALTER TABLE workspace_sources ADD COLUMN IF NOT EXISTS reviewed_at TEXT;
ALTER TABLE workspace_sources ADD COLUMN IF NOT EXISTS reviewed_by_user_id TEXT;
UPDATE workspace_sources
   SET review_state = 'unset'
 WHERE review_state IS NULL
    OR btrim(review_state) = ''
    OR review_state NOT IN ('unset', 'needs_review', 'reviewed');
UPDATE workspace_sources
   SET review_state_updated_at = COALESCE(
       NULLIF(btrim(review_state_updated_at), ''),
       NULLIF(btrim(added_at::text), ''),
       (CURRENT_TIMESTAMP AT TIME ZONE 'UTC')::text
   )
 WHERE review_state_updated_at IS NULL OR btrim(review_state_updated_at) = '';
UPDATE workspace_sources
   SET reviewed_at = NULL,
       reviewed_by_user_id = NULL
 WHERE review_state <> 'reviewed'
   AND (reviewed_at IS NOT NULL OR reviewed_by_user_id IS NOT NULL);

CREATE TABLE IF NOT EXISTS workspace_artifacts (
    id              TEXT    NOT NULL,
    workspace_id    TEXT    NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    artifact_type   TEXT    NOT NULL,
    title           TEXT    NOT NULL,
    status          TEXT    NOT NULL DEFAULT 'pending',
    content         TEXT,
    total_tokens    INTEGER,
    total_cost_usd  REAL,
    created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at    TIMESTAMP,
    version         INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (workspace_id, id)
);
CREATE INDEX IF NOT EXISTS idx_ws_artifacts_workspace ON workspace_artifacts(workspace_id);

CREATE TABLE IF NOT EXISTS workspace_notes (
    id            SERIAL PRIMARY KEY,
    workspace_id  TEXT    NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    title         TEXT    NOT NULL DEFAULT '',
    content       TEXT    NOT NULL DEFAULT '',
    keywords_json TEXT    NOT NULL DEFAULT '[]',
    created_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_modified TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted       BOOLEAN NOT NULL DEFAULT false,
    version       INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_ws_notes_workspace ON workspace_notes(workspace_id);
UPDATE db_schema_version
   SET version = 37
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 37;
"""
_MIGRATION_SQL_V37_TO_V38 = """
/*───────────────────────────────────────────────────────────────
  Migration to Version 38 - Quiz remediation conversion state (2026-03-13)
───────────────────────────────────────────────────────────────*/
UPDATE db_schema_version
   SET version = 38
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 38;
"""
_MIGRATION_SQL_V37_TO_V38_POSTGRES = """
/*───────────────────────────────────────────────────────────────
  Migration to Version 38 - Quiz remediation conversion state (2026-03-13)
───────────────────────────────────────────────────────────────*/
UPDATE db_schema_version
   SET version = 38
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 38;
"""
_MIGRATION_SQL_V38_TO_V39 = """
/*───────────────────────────────────────────────────────────────
  Migration to Version 39 - Workspace ownership for quizzes/decks + workspace policy (2026-03-27)
───────────────────────────────────────────────────────────────*/
UPDATE db_schema_version
   SET version = 39
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 39;
"""
_MIGRATION_SQL_V38_TO_V39_POSTGRES = """
/*───────────────────────────────────────────────────────────────
  Migration to Version 39 - Workspace ownership for quizzes/decks + workspace policy (2026-03-27)
───────────────────────────────────────────────────────────────*/
UPDATE db_schema_version
   SET version = 39
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 39;
"""
_MIGRATION_SQL_V39_TO_V40 = """
/*───────────────────────────────────────────────────────────────
  Migration to Version 40 - Persona buddy storage contract (2026-03-31)
───────────────────────────────────────────────────────────────*/
CREATE TABLE IF NOT EXISTS persona_buddies (
  persona_id TEXT PRIMARY KEY REFERENCES persona_profiles(id) ON DELETE CASCADE,
  user_id TEXT NOT NULL,
  derivation_version INTEGER NOT NULL DEFAULT 1,
  source_fingerprint TEXT NOT NULL,
  derived_core_json TEXT NOT NULL DEFAULT '{}',
  overlay_preferences_json TEXT NOT NULL DEFAULT '{}',
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  version INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_persona_buddies_user
  ON persona_buddies(user_id, persona_id);
UPDATE db_schema_version
   SET version = 40
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 40;
"""
_MIGRATION_SQL_V39_TO_V40_POSTGRES = """
/*───────────────────────────────────────────────────────────────
  Migration to Version 40 - Persona buddy storage contract (2026-03-31) [Postgres]
───────────────────────────────────────────────────────────────*/
CREATE TABLE IF NOT EXISTS persona_buddies (
  persona_id TEXT PRIMARY KEY REFERENCES persona_profiles(id) ON DELETE CASCADE,
  user_id TEXT NOT NULL,
  derivation_version INTEGER NOT NULL DEFAULT 1,
  source_fingerprint TEXT NOT NULL,
  derived_core_json TEXT NOT NULL DEFAULT '{}',
  overlay_preferences_json TEXT NOT NULL DEFAULT '{}',
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  version INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_persona_buddies_user
  ON persona_buddies(user_id, persona_id);
UPDATE db_schema_version
   SET version = 40
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 40;
"""
_MIGRATION_SQL_V40_TO_V41 = """
/*───────────────────────────────────────────────────────────────
  Migration to Version 41 - Manuscript management tables (2026-04-03)
───────────────────────────────────────────────────────────────*/

/* ── manuscript_projects ───────────────────────────────────── */
CREATE TABLE IF NOT EXISTS manuscript_projects (
  id               TEXT PRIMARY KEY,
  title            TEXT NOT NULL,
  subtitle         TEXT,
  author           TEXT,
  genre            TEXT,
  status           TEXT NOT NULL DEFAULT 'draft'
                     CHECK(status IN ('draft','outlining','writing','revising','complete','archived')),
  synopsis         TEXT,
  target_word_count INTEGER,
  word_count       INTEGER NOT NULL DEFAULT 0,
  settings_json    TEXT NOT NULL DEFAULT '{}',
  created_at       DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted          BOOLEAN  NOT NULL DEFAULT 0,
  client_id        TEXT     NOT NULL,
  version          INTEGER  NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_manuscript_projects_deleted
  ON manuscript_projects(deleted);

/* ── manuscript_parts ──────────────────────────────────────── */
CREATE TABLE IF NOT EXISTS manuscript_parts (
  id               TEXT PRIMARY KEY,
  project_id       TEXT NOT NULL REFERENCES manuscript_projects(id) ON DELETE CASCADE,
  title            TEXT NOT NULL,
  sort_order       REAL NOT NULL DEFAULT 0,
  synopsis         TEXT,
  word_count       INTEGER NOT NULL DEFAULT 0,
  created_at       DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted          BOOLEAN  NOT NULL DEFAULT 0,
  client_id        TEXT     NOT NULL,
  version          INTEGER  NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_manuscript_parts_deleted
  ON manuscript_parts(deleted);
CREATE INDEX IF NOT EXISTS idx_manuscript_parts_project
  ON manuscript_parts(project_id, sort_order);

/* ── manuscript_chapters ───────────────────────────────────── */
CREATE TABLE IF NOT EXISTS manuscript_chapters (
  id               TEXT PRIMARY KEY,
  project_id       TEXT NOT NULL REFERENCES manuscript_projects(id) ON DELETE CASCADE,
  part_id          TEXT REFERENCES manuscript_parts(id) ON DELETE SET NULL,
  title            TEXT NOT NULL,
  sort_order       REAL NOT NULL DEFAULT 0,
  synopsis         TEXT,
  pov_character_id TEXT,
  word_count       INTEGER NOT NULL DEFAULT 0,
  status           TEXT NOT NULL DEFAULT 'outline'
                     CHECK(status IN ('outline','draft','revising','final')),
  created_at       DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted          BOOLEAN  NOT NULL DEFAULT 0,
  client_id        TEXT     NOT NULL,
  version          INTEGER  NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_manuscript_chapters_deleted
  ON manuscript_chapters(deleted);
CREATE INDEX IF NOT EXISTS idx_manuscript_chapters_project
  ON manuscript_chapters(project_id, sort_order);
CREATE INDEX IF NOT EXISTS idx_manuscript_chapters_part
  ON manuscript_chapters(part_id, sort_order);

/* ── manuscript_scenes ─────────────────────────────────────── */
CREATE TABLE IF NOT EXISTS manuscript_scenes (
  id               TEXT PRIMARY KEY,
  chapter_id       TEXT NOT NULL REFERENCES manuscript_chapters(id) ON DELETE CASCADE,
  project_id       TEXT NOT NULL REFERENCES manuscript_projects(id) ON DELETE CASCADE,
  title            TEXT,
  sort_order       REAL NOT NULL DEFAULT 0,
  content_json     TEXT,
  content_plain    TEXT,
  synopsis         TEXT,
  word_count       INTEGER NOT NULL DEFAULT 0,
  pov_character_id TEXT,
  status           TEXT NOT NULL DEFAULT 'outline'
                     CHECK(status IN ('outline','draft','revising','final')),
  created_at       DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted          BOOLEAN  NOT NULL DEFAULT 0,
  client_id        TEXT     NOT NULL,
  version          INTEGER  NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_manuscript_scenes_deleted
  ON manuscript_scenes(deleted);
CREATE INDEX IF NOT EXISTS idx_manuscript_scenes_chapter
  ON manuscript_scenes(chapter_id, sort_order);
CREATE INDEX IF NOT EXISTS idx_manuscript_scenes_project
  ON manuscript_scenes(project_id);

/* ── manuscript_scenes FTS5 ────────────────────────────────── */
CREATE VIRTUAL TABLE IF NOT EXISTS manuscript_scenes_fts
USING fts5(
  title, content_plain, synopsis,
  content='manuscript_scenes',
  content_rowid='rowid'
);

DROP TRIGGER IF EXISTS manuscript_scenes_ai;
DROP TRIGGER IF EXISTS manuscript_scenes_au;
DROP TRIGGER IF EXISTS manuscript_scenes_ad;

CREATE TRIGGER manuscript_scenes_ai
AFTER INSERT ON manuscript_scenes BEGIN
  INSERT INTO manuscript_scenes_fts(rowid, title, content_plain, synopsis)
  SELECT new.rowid, new.title, new.content_plain, new.synopsis
  WHERE new.deleted = 0;
END;

CREATE TRIGGER manuscript_scenes_au
AFTER UPDATE ON manuscript_scenes BEGIN
  INSERT INTO manuscript_scenes_fts(manuscript_scenes_fts, rowid, title, content_plain, synopsis)
  SELECT 'delete', old.rowid, old.title, old.content_plain, old.synopsis
  WHERE old.deleted = 0;

  INSERT INTO manuscript_scenes_fts(rowid, title, content_plain, synopsis)
  SELECT new.rowid, new.title, new.content_plain, new.synopsis
  WHERE new.deleted = 0;
END;

CREATE TRIGGER manuscript_scenes_ad
AFTER DELETE ON manuscript_scenes BEGIN
  INSERT INTO manuscript_scenes_fts(manuscript_scenes_fts, rowid, title, content_plain, synopsis)
  VALUES('delete', old.rowid, old.title, old.content_plain, old.synopsis);
END;

/* ── sync triggers: manuscript_projects ────────────────────── */
DROP TRIGGER IF EXISTS manuscript_projects_sync_create;
DROP TRIGGER IF EXISTS manuscript_projects_sync_update;
DROP TRIGGER IF EXISTS manuscript_projects_sync_delete;
DROP TRIGGER IF EXISTS manuscript_projects_sync_undelete;

CREATE TRIGGER manuscript_projects_sync_create
AFTER INSERT ON manuscript_projects BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('manuscript_projects', NEW.id, 'create', NEW.last_modified, NEW.client_id, NEW.version,
         json_object('id',NEW.id,'title',NEW.title,'subtitle',NEW.subtitle,'author',NEW.author,
                     'genre',NEW.genre,'status',NEW.status,'synopsis',NEW.synopsis,
                     'target_word_count',NEW.target_word_count,'word_count',NEW.word_count,
                     'settings_json',NEW.settings_json,
                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER manuscript_projects_sync_update
AFTER UPDATE ON manuscript_projects
WHEN OLD.deleted = NEW.deleted AND (
     OLD.title IS NOT NEW.title OR
     OLD.subtitle IS NOT NEW.subtitle OR
     OLD.author IS NOT NEW.author OR
     OLD.genre IS NOT NEW.genre OR
     OLD.status IS NOT NEW.status OR
     OLD.synopsis IS NOT NEW.synopsis OR
     OLD.target_word_count IS NOT NEW.target_word_count OR
     OLD.word_count IS NOT NEW.word_count OR
     OLD.settings_json IS NOT NEW.settings_json OR
     OLD.last_modified IS NOT NEW.last_modified OR
     OLD.version IS NOT NEW.version)
BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('manuscript_projects', NEW.id, 'update', NEW.last_modified, NEW.client_id, NEW.version,
         json_object('id',NEW.id,'title',NEW.title,'subtitle',NEW.subtitle,'author',NEW.author,
                     'genre',NEW.genre,'status',NEW.status,'synopsis',NEW.synopsis,
                     'target_word_count',NEW.target_word_count,'word_count',NEW.word_count,
                     'settings_json',NEW.settings_json,
                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER manuscript_projects_sync_delete
AFTER UPDATE ON manuscript_projects
WHEN OLD.deleted = 0 AND NEW.deleted = 1
BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('manuscript_projects', NEW.id, 'delete', NEW.last_modified, NEW.client_id, NEW.version,
         json_object('id',NEW.id,'deleted',NEW.deleted,'last_modified',NEW.last_modified,
                     'version',NEW.version,'client_id',NEW.client_id));
END;

CREATE TRIGGER manuscript_projects_sync_undelete
AFTER UPDATE ON manuscript_projects
WHEN OLD.deleted = 1 AND NEW.deleted = 0
BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('manuscript_projects', NEW.id, 'update', NEW.last_modified, NEW.client_id, NEW.version,
         json_object('id',NEW.id,'title',NEW.title,'subtitle',NEW.subtitle,'author',NEW.author,
                     'genre',NEW.genre,'status',NEW.status,'synopsis',NEW.synopsis,
                     'target_word_count',NEW.target_word_count,'word_count',NEW.word_count,
                     'settings_json',NEW.settings_json,
                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

/* ── sync triggers: manuscript_parts ───────────────────────── */
DROP TRIGGER IF EXISTS manuscript_parts_sync_create;
DROP TRIGGER IF EXISTS manuscript_parts_sync_update;
DROP TRIGGER IF EXISTS manuscript_parts_sync_delete;
DROP TRIGGER IF EXISTS manuscript_parts_sync_undelete;

CREATE TRIGGER manuscript_parts_sync_create
AFTER INSERT ON manuscript_parts BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('manuscript_parts', NEW.id, 'create', NEW.last_modified, NEW.client_id, NEW.version,
         json_object('id',NEW.id,'project_id',NEW.project_id,'title',NEW.title,
                     'sort_order',NEW.sort_order,'synopsis',NEW.synopsis,'word_count',NEW.word_count,
                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER manuscript_parts_sync_update
AFTER UPDATE ON manuscript_parts
WHEN OLD.deleted = NEW.deleted AND (
     OLD.title IS NOT NEW.title OR
     OLD.sort_order IS NOT NEW.sort_order OR
     OLD.synopsis IS NOT NEW.synopsis OR
     OLD.word_count IS NOT NEW.word_count OR
     OLD.last_modified IS NOT NEW.last_modified OR
     OLD.version IS NOT NEW.version)
BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('manuscript_parts', NEW.id, 'update', NEW.last_modified, NEW.client_id, NEW.version,
         json_object('id',NEW.id,'project_id',NEW.project_id,'title',NEW.title,
                     'sort_order',NEW.sort_order,'synopsis',NEW.synopsis,'word_count',NEW.word_count,
                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER manuscript_parts_sync_delete
AFTER UPDATE ON manuscript_parts
WHEN OLD.deleted = 0 AND NEW.deleted = 1
BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('manuscript_parts', NEW.id, 'delete', NEW.last_modified, NEW.client_id, NEW.version,
         json_object('id',NEW.id,'deleted',NEW.deleted,'last_modified',NEW.last_modified,
                     'version',NEW.version,'client_id',NEW.client_id));
END;

CREATE TRIGGER manuscript_parts_sync_undelete
AFTER UPDATE ON manuscript_parts
WHEN OLD.deleted = 1 AND NEW.deleted = 0
BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('manuscript_parts', NEW.id, 'update', NEW.last_modified, NEW.client_id, NEW.version,
         json_object('id',NEW.id,'project_id',NEW.project_id,'title',NEW.title,
                     'sort_order',NEW.sort_order,'synopsis',NEW.synopsis,'word_count',NEW.word_count,
                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

/* ── sync triggers: manuscript_chapters ────────────────────── */
DROP TRIGGER IF EXISTS manuscript_chapters_sync_create;
DROP TRIGGER IF EXISTS manuscript_chapters_sync_update;
DROP TRIGGER IF EXISTS manuscript_chapters_sync_delete;
DROP TRIGGER IF EXISTS manuscript_chapters_sync_undelete;

CREATE TRIGGER manuscript_chapters_sync_create
AFTER INSERT ON manuscript_chapters BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('manuscript_chapters', NEW.id, 'create', NEW.last_modified, NEW.client_id, NEW.version,
         json_object('id',NEW.id,'project_id',NEW.project_id,'part_id',NEW.part_id,
                     'title',NEW.title,'sort_order',NEW.sort_order,'synopsis',NEW.synopsis,
                     'pov_character_id',NEW.pov_character_id,'word_count',NEW.word_count,
                     'status',NEW.status,'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER manuscript_chapters_sync_update
AFTER UPDATE ON manuscript_chapters
WHEN OLD.deleted = NEW.deleted AND (
     OLD.title IS NOT NEW.title OR
     OLD.part_id IS NOT NEW.part_id OR
     OLD.sort_order IS NOT NEW.sort_order OR
     OLD.synopsis IS NOT NEW.synopsis OR
     OLD.pov_character_id IS NOT NEW.pov_character_id OR
     OLD.word_count IS NOT NEW.word_count OR
     OLD.status IS NOT NEW.status OR
     OLD.last_modified IS NOT NEW.last_modified OR
     OLD.version IS NOT NEW.version)
BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('manuscript_chapters', NEW.id, 'update', NEW.last_modified, NEW.client_id, NEW.version,
         json_object('id',NEW.id,'project_id',NEW.project_id,'part_id',NEW.part_id,
                     'title',NEW.title,'sort_order',NEW.sort_order,'synopsis',NEW.synopsis,
                     'pov_character_id',NEW.pov_character_id,'word_count',NEW.word_count,
                     'status',NEW.status,'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER manuscript_chapters_sync_delete
AFTER UPDATE ON manuscript_chapters
WHEN OLD.deleted = 0 AND NEW.deleted = 1
BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('manuscript_chapters', NEW.id, 'delete', NEW.last_modified, NEW.client_id, NEW.version,
         json_object('id',NEW.id,'deleted',NEW.deleted,'last_modified',NEW.last_modified,
                     'version',NEW.version,'client_id',NEW.client_id));
END;

CREATE TRIGGER manuscript_chapters_sync_undelete
AFTER UPDATE ON manuscript_chapters
WHEN OLD.deleted = 1 AND NEW.deleted = 0
BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('manuscript_chapters', NEW.id, 'update', NEW.last_modified, NEW.client_id, NEW.version,
         json_object('id',NEW.id,'project_id',NEW.project_id,'part_id',NEW.part_id,
                     'title',NEW.title,'sort_order',NEW.sort_order,'synopsis',NEW.synopsis,
                     'pov_character_id',NEW.pov_character_id,'word_count',NEW.word_count,
                     'status',NEW.status,'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

/* ── sync triggers: manuscript_scenes ──────────────────────── */
DROP TRIGGER IF EXISTS manuscript_scenes_sync_create;
DROP TRIGGER IF EXISTS manuscript_scenes_sync_update;
DROP TRIGGER IF EXISTS manuscript_scenes_sync_delete;
DROP TRIGGER IF EXISTS manuscript_scenes_sync_undelete;

CREATE TRIGGER manuscript_scenes_sync_create
AFTER INSERT ON manuscript_scenes BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('manuscript_scenes', NEW.id, 'create', NEW.last_modified, NEW.client_id, NEW.version,
         json_object('id',NEW.id,'chapter_id',NEW.chapter_id,'project_id',NEW.project_id,
                     'title',NEW.title,'sort_order',NEW.sort_order,
                     'content_json',NEW.content_json,'content_plain',NEW.content_plain,
                     'synopsis',NEW.synopsis,
                     'word_count',NEW.word_count,'pov_character_id',NEW.pov_character_id,
                     'status',NEW.status,'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER manuscript_scenes_sync_update
AFTER UPDATE ON manuscript_scenes
WHEN OLD.deleted = NEW.deleted AND (
     OLD.title IS NOT NEW.title OR
     OLD.chapter_id IS NOT NEW.chapter_id OR
     OLD.sort_order IS NOT NEW.sort_order OR
     OLD.content_json IS NOT NEW.content_json OR
     OLD.content_plain IS NOT NEW.content_plain OR
     OLD.synopsis IS NOT NEW.synopsis OR
     OLD.word_count IS NOT NEW.word_count OR
     OLD.pov_character_id IS NOT NEW.pov_character_id OR
     OLD.status IS NOT NEW.status OR
     OLD.last_modified IS NOT NEW.last_modified OR
     OLD.version IS NOT NEW.version)
BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('manuscript_scenes', NEW.id, 'update', NEW.last_modified, NEW.client_id, NEW.version,
         json_object('id',NEW.id,'chapter_id',NEW.chapter_id,'project_id',NEW.project_id,
                     'title',NEW.title,'sort_order',NEW.sort_order,
                     'content_json',NEW.content_json,'content_plain',NEW.content_plain,
                     'synopsis',NEW.synopsis,
                     'word_count',NEW.word_count,'pov_character_id',NEW.pov_character_id,
                     'status',NEW.status,'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER manuscript_scenes_sync_delete
AFTER UPDATE ON manuscript_scenes
WHEN OLD.deleted = 0 AND NEW.deleted = 1
BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('manuscript_scenes', NEW.id, 'delete', NEW.last_modified, NEW.client_id, NEW.version,
         json_object('id',NEW.id,'deleted',NEW.deleted,'last_modified',NEW.last_modified,
                     'version',NEW.version,'client_id',NEW.client_id));
END;

CREATE TRIGGER manuscript_scenes_sync_undelete
AFTER UPDATE ON manuscript_scenes
WHEN OLD.deleted = 1 AND NEW.deleted = 0
BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('manuscript_scenes', NEW.id, 'update', NEW.last_modified, NEW.client_id, NEW.version,
         json_object('id',NEW.id,'chapter_id',NEW.chapter_id,'project_id',NEW.project_id,
                     'title',NEW.title,'sort_order',NEW.sort_order,
                     'content_json',NEW.content_json,'content_plain',NEW.content_plain,
                     'synopsis',NEW.synopsis,
                     'word_count',NEW.word_count,'pov_character_id',NEW.pov_character_id,
                     'status',NEW.status,'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

/* ── FK consistency triggers: parent project_id must match ─── */
DROP TRIGGER IF EXISTS manuscript_chapters_check_project;
DROP TRIGGER IF EXISTS manuscript_chapters_check_project_update;
DROP TRIGGER IF EXISTS manuscript_scenes_check_project;
DROP TRIGGER IF EXISTS manuscript_scenes_check_project_update;

CREATE TRIGGER manuscript_chapters_check_project
BEFORE INSERT ON manuscript_chapters
WHEN NEW.part_id IS NOT NULL
BEGIN
  SELECT CASE
    WHEN (SELECT project_id FROM manuscript_parts WHERE id = NEW.part_id) != NEW.project_id
    THEN RAISE(ABORT, 'chapter project_id must match its parent part project_id')
  END;
END;

CREATE TRIGGER manuscript_chapters_check_project_update
BEFORE UPDATE ON manuscript_chapters
WHEN NEW.part_id IS NOT NULL
BEGIN
  SELECT CASE
    WHEN (SELECT project_id FROM manuscript_parts WHERE id = NEW.part_id) != NEW.project_id
    THEN RAISE(ABORT, 'chapter project_id must match its parent part project_id')
  END;
END;

CREATE TRIGGER manuscript_scenes_check_project
BEFORE INSERT ON manuscript_scenes
BEGIN
  SELECT CASE
    WHEN (SELECT project_id FROM manuscript_chapters WHERE id = NEW.chapter_id) != NEW.project_id
    THEN RAISE(ABORT, 'scene project_id must match its parent chapter project_id')
  END;
END;

CREATE TRIGGER manuscript_scenes_check_project_update
BEFORE UPDATE ON manuscript_scenes
BEGIN
  SELECT CASE
    WHEN (SELECT project_id FROM manuscript_chapters WHERE id = NEW.chapter_id) != NEW.project_id
    THEN RAISE(ABORT, 'scene project_id must match its parent chapter project_id')
  END;
END;

UPDATE db_schema_version
   SET version = 41
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 41;
"""

_MIGRATION_SQL_V40_TO_V41_POSTGRES = """
/*───────────────────────────────────────────────────────────────
  Migration to Version 41 - Manuscript management tables (2026-04-03) [Postgres]
───────────────────────────────────────────────────────────────*/

/* ── manuscript_projects ───────────────────────────────────── */
CREATE TABLE IF NOT EXISTS manuscript_projects (
  id               TEXT PRIMARY KEY,
  title            TEXT NOT NULL,
  subtitle         TEXT,
  author           TEXT,
  genre            TEXT,
  status           TEXT NOT NULL DEFAULT 'draft'
                     CHECK(status IN ('draft','outlining','writing','revising','complete','archived')),
  synopsis         TEXT,
  target_word_count INTEGER,
  word_count       INTEGER NOT NULL DEFAULT 0,
  settings_json    TEXT NOT NULL DEFAULT '{}',
  created_at       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted          BOOLEAN  NOT NULL DEFAULT FALSE,
  client_id        TEXT     NOT NULL,
  version          INTEGER  NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_manuscript_projects_deleted
  ON manuscript_projects(deleted);

/* ── manuscript_parts ──────────────────────────────────────── */
CREATE TABLE IF NOT EXISTS manuscript_parts (
  id               TEXT PRIMARY KEY,
  project_id       TEXT NOT NULL REFERENCES manuscript_projects(id) ON DELETE CASCADE,
  title            TEXT NOT NULL,
  sort_order       REAL NOT NULL DEFAULT 0,
  synopsis         TEXT,
  word_count       INTEGER NOT NULL DEFAULT 0,
  created_at       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted          BOOLEAN  NOT NULL DEFAULT FALSE,
  client_id        TEXT     NOT NULL,
  version          INTEGER  NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_manuscript_parts_deleted
  ON manuscript_parts(deleted);
CREATE INDEX IF NOT EXISTS idx_manuscript_parts_project
  ON manuscript_parts(project_id, sort_order);

/* ── manuscript_chapters ───────────────────────────────────── */
CREATE TABLE IF NOT EXISTS manuscript_chapters (
  id               TEXT PRIMARY KEY,
  project_id       TEXT NOT NULL REFERENCES manuscript_projects(id) ON DELETE CASCADE,
  part_id          TEXT REFERENCES manuscript_parts(id) ON DELETE SET NULL,
  title            TEXT NOT NULL,
  sort_order       REAL NOT NULL DEFAULT 0,
  synopsis         TEXT,
  pov_character_id TEXT,
  word_count       INTEGER NOT NULL DEFAULT 0,
  status           TEXT NOT NULL DEFAULT 'outline'
                     CHECK(status IN ('outline','draft','revising','final')),
  created_at       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted          BOOLEAN  NOT NULL DEFAULT FALSE,
  client_id        TEXT     NOT NULL,
  version          INTEGER  NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_manuscript_chapters_deleted
  ON manuscript_chapters(deleted);
CREATE INDEX IF NOT EXISTS idx_manuscript_chapters_project
  ON manuscript_chapters(project_id, sort_order);
CREATE INDEX IF NOT EXISTS idx_manuscript_chapters_part
  ON manuscript_chapters(part_id, sort_order);

/* ── manuscript_scenes ─────────────────────────────────────── */
CREATE TABLE IF NOT EXISTS manuscript_scenes (
  id               TEXT PRIMARY KEY,
  chapter_id       TEXT NOT NULL REFERENCES manuscript_chapters(id) ON DELETE CASCADE,
  project_id       TEXT NOT NULL REFERENCES manuscript_projects(id) ON DELETE CASCADE,
  title            TEXT,
  sort_order       REAL NOT NULL DEFAULT 0,
  content_json     TEXT,
  content_plain    TEXT,
  synopsis         TEXT,
  word_count       INTEGER NOT NULL DEFAULT 0,
  pov_character_id TEXT,
  status           TEXT NOT NULL DEFAULT 'outline'
                     CHECK(status IN ('outline','draft','revising','final')),
  created_at       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted          BOOLEAN  NOT NULL DEFAULT FALSE,
  client_id        TEXT     NOT NULL,
  version          INTEGER  NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_manuscript_scenes_deleted
  ON manuscript_scenes(deleted);
CREATE INDEX IF NOT EXISTS idx_manuscript_scenes_chapter
  ON manuscript_scenes(chapter_id, sort_order);
CREATE INDEX IF NOT EXISTS idx_manuscript_scenes_project
  ON manuscript_scenes(project_id);

/* Note: FTS5 is SQLite-only; Postgres would use tsvector/GIN instead.
   Sync triggers also use SQLite-specific json_object(); Postgres equivalent
   would use jsonb_build_object(). Skipped for now — only tables + indexes. */

-- Postgres v40→v41 adds manuscript tables only (FTS5 and sync triggers are SQLite-specific)
-- Version stays at 40 until Postgres equivalents (tsvector search, NOTIFY-based sync) are added
UPDATE db_schema_version
   SET version = 40
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 40;
"""

# --- Migration: V41 -> V42 (Characters, world info, plot, citations) ---
_MIGRATION_SQL_V41_TO_V42 = """
/*───────────────────────────────────────────────────────────────
  Migration to Version 42 — Characters, world info, plot, citations (2026-04-XX)
───────────────────────────────────────────────────────────────*/

-- CHARACTERS
CREATE TABLE IF NOT EXISTS manuscript_characters (
  id              TEXT PRIMARY KEY,
  project_id      TEXT NOT NULL REFERENCES manuscript_projects(id) ON DELETE CASCADE,
  name            TEXT NOT NULL,
  role            TEXT NOT NULL DEFAULT 'supporting'
                    CHECK(role IN ('protagonist','antagonist','supporting','minor','mentioned')),
  cast_group      TEXT,
  full_name       TEXT,
  age             TEXT,
  gender          TEXT,
  appearance      TEXT,
  personality     TEXT,
  backstory       TEXT,
  motivation      TEXT,
  arc_summary     TEXT,
  notes           TEXT,
  custom_fields_json TEXT NOT NULL DEFAULT '{}',
  sort_order      REAL NOT NULL DEFAULT 0,
  created_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified   DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted         BOOLEAN NOT NULL DEFAULT 0,
  client_id       TEXT NOT NULL DEFAULT 'unknown',
  version         INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_mchr_project ON manuscript_characters(project_id, sort_order);
CREATE INDEX IF NOT EXISTS idx_mchr_role ON manuscript_characters(project_id, role);
CREATE INDEX IF NOT EXISTS idx_mchr_deleted ON manuscript_characters(deleted);

-- CHARACTER RELATIONSHIPS
CREATE TABLE IF NOT EXISTS manuscript_character_relationships (
  id                TEXT PRIMARY KEY,
  project_id        TEXT NOT NULL REFERENCES manuscript_projects(id) ON DELETE CASCADE,
  from_character_id TEXT NOT NULL REFERENCES manuscript_characters(id) ON DELETE CASCADE,
  to_character_id   TEXT NOT NULL REFERENCES manuscript_characters(id) ON DELETE CASCADE,
  relationship_type TEXT NOT NULL,
  description       TEXT,
  bidirectional     BOOLEAN NOT NULL DEFAULT 1,
  created_at        DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified     DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted           BOOLEAN NOT NULL DEFAULT 0,
  client_id         TEXT NOT NULL DEFAULT 'unknown',
  version           INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_mcrel_project ON manuscript_character_relationships(project_id);
CREATE INDEX IF NOT EXISTS idx_mcrel_from ON manuscript_character_relationships(from_character_id);
CREATE INDEX IF NOT EXISTS idx_mcrel_to ON manuscript_character_relationships(to_character_id);
CREATE INDEX IF NOT EXISTS idx_mcrel_deleted ON manuscript_character_relationships(deleted);

-- WORLD INFO
CREATE TABLE IF NOT EXISTS manuscript_world_info (
  id              TEXT PRIMARY KEY,
  project_id      TEXT NOT NULL REFERENCES manuscript_projects(id) ON DELETE CASCADE,
  kind            TEXT NOT NULL CHECK(kind IN ('location','item','faction','concept','event','custom')),
  name            TEXT NOT NULL,
  description     TEXT,
  parent_id       TEXT REFERENCES manuscript_world_info(id) ON DELETE SET NULL,
  properties_json TEXT NOT NULL DEFAULT '{}',
  tags_json       TEXT NOT NULL DEFAULT '[]',
  sort_order      REAL NOT NULL DEFAULT 0,
  created_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified   DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted         BOOLEAN NOT NULL DEFAULT 0,
  client_id       TEXT NOT NULL DEFAULT 'unknown',
  version         INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_mwi_project_kind ON manuscript_world_info(project_id, kind);
CREATE INDEX IF NOT EXISTS idx_mwi_parent ON manuscript_world_info(parent_id);
CREATE INDEX IF NOT EXISTS idx_mwi_deleted ON manuscript_world_info(deleted);

-- PLOT LINES
CREATE TABLE IF NOT EXISTS manuscript_plot_lines (
  id            TEXT PRIMARY KEY,
  project_id    TEXT NOT NULL REFERENCES manuscript_projects(id) ON DELETE CASCADE,
  title         TEXT NOT NULL,
  description   TEXT,
  status        TEXT NOT NULL DEFAULT 'active'
                  CHECK(status IN ('active','resolved','abandoned','dormant')),
  color         TEXT,
  sort_order    REAL NOT NULL DEFAULT 0,
  created_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted       BOOLEAN NOT NULL DEFAULT 0,
  client_id     TEXT NOT NULL DEFAULT 'unknown',
  version       INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_mpl_project ON manuscript_plot_lines(project_id);
CREATE INDEX IF NOT EXISTS idx_mpl_deleted ON manuscript_plot_lines(deleted);

-- PLOT EVENTS
CREATE TABLE IF NOT EXISTS manuscript_plot_events (
  id            TEXT PRIMARY KEY,
  project_id    TEXT NOT NULL REFERENCES manuscript_projects(id) ON DELETE CASCADE,
  plot_line_id  TEXT NOT NULL REFERENCES manuscript_plot_lines(id) ON DELETE CASCADE,
  scene_id      TEXT REFERENCES manuscript_scenes(id) ON DELETE SET NULL,
  chapter_id    TEXT REFERENCES manuscript_chapters(id) ON DELETE SET NULL,
  title         TEXT NOT NULL,
  description   TEXT,
  event_type    TEXT NOT NULL DEFAULT 'plot'
                  CHECK(event_type IN ('setup','conflict','action','emotional','plot','resolution')),
  sort_order    REAL NOT NULL DEFAULT 0,
  created_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted       BOOLEAN NOT NULL DEFAULT 0,
  client_id     TEXT NOT NULL DEFAULT 'unknown',
  version       INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_mpe_plot_line ON manuscript_plot_events(plot_line_id, sort_order);
CREATE INDEX IF NOT EXISTS idx_mpe_scene ON manuscript_plot_events(scene_id);
CREATE INDEX IF NOT EXISTS idx_mpe_deleted ON manuscript_plot_events(deleted);

-- PLOT HOLES
CREATE TABLE IF NOT EXISTS manuscript_plot_holes (
  id            TEXT PRIMARY KEY,
  project_id    TEXT NOT NULL REFERENCES manuscript_projects(id) ON DELETE CASCADE,
  title         TEXT NOT NULL,
  description   TEXT,
  severity      TEXT NOT NULL DEFAULT 'medium'
                  CHECK(severity IN ('low','medium','high','critical')),
  status        TEXT NOT NULL DEFAULT 'open'
                  CHECK(status IN ('open','investigating','resolved','wontfix')),
  scene_id      TEXT REFERENCES manuscript_scenes(id) ON DELETE SET NULL,
  chapter_id    TEXT REFERENCES manuscript_chapters(id) ON DELETE SET NULL,
  plot_line_id  TEXT REFERENCES manuscript_plot_lines(id) ON DELETE SET NULL,
  resolution    TEXT,
  detected_by   TEXT NOT NULL DEFAULT 'manual'
                  CHECK(detected_by IN ('manual','ai')),
  created_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted       BOOLEAN NOT NULL DEFAULT 0,
  client_id     TEXT NOT NULL DEFAULT 'unknown',
  version       INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_mph_project ON manuscript_plot_holes(project_id, status);
CREATE INDEX IF NOT EXISTS idx_mph_deleted ON manuscript_plot_holes(deleted);

-- SCENE-CHARACTER LINKING (no soft delete, no version — simple join table)
CREATE TABLE IF NOT EXISTS manuscript_scene_characters (
  scene_id      TEXT NOT NULL REFERENCES manuscript_scenes(id) ON DELETE CASCADE,
  character_id  TEXT NOT NULL REFERENCES manuscript_characters(id) ON DELETE CASCADE,
  is_pov        BOOLEAN NOT NULL DEFAULT 0,
  last_modified TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f','now')),
  deleted       BOOLEAN NOT NULL DEFAULT 0,
  client_id     TEXT,
  version       INTEGER NOT NULL DEFAULT 1,
  PRIMARY KEY (scene_id, character_id)
);

-- SCENE-WORLD_INFO LINKING
CREATE TABLE IF NOT EXISTS manuscript_scene_world_info (
  scene_id        TEXT NOT NULL REFERENCES manuscript_scenes(id) ON DELETE CASCADE,
  world_info_id   TEXT NOT NULL REFERENCES manuscript_world_info(id) ON DELETE CASCADE,
  last_modified   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f','now')),
  deleted         BOOLEAN NOT NULL DEFAULT 0,
  client_id       TEXT,
  version         INTEGER NOT NULL DEFAULT 1,
  PRIMARY KEY (scene_id, world_info_id)
);

-- CITATIONS (references from RAG into manuscript)
CREATE TABLE IF NOT EXISTS manuscript_citations (
  id            TEXT PRIMARY KEY,
  project_id    TEXT NOT NULL REFERENCES manuscript_projects(id) ON DELETE CASCADE,
  scene_id      TEXT NOT NULL REFERENCES manuscript_scenes(id) ON DELETE CASCADE,
  source_type   TEXT NOT NULL,
  source_id     TEXT,
  source_title  TEXT,
  excerpt       TEXT,
  query_used    TEXT,
  anchor_offset INTEGER,
  created_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted       BOOLEAN NOT NULL DEFAULT 0,
  client_id     TEXT NOT NULL DEFAULT 'unknown',
  version       INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_mcit_scene ON manuscript_citations(scene_id);
CREATE INDEX IF NOT EXISTS idx_mcit_project ON manuscript_citations(project_id);
CREATE INDEX IF NOT EXISTS idx_mcit_deleted ON manuscript_citations(deleted);

/* ── sync triggers: manuscript_characters ─────────────────── */
DROP TRIGGER IF EXISTS manuscript_characters_sync_create;
DROP TRIGGER IF EXISTS manuscript_characters_sync_update;
DROP TRIGGER IF EXISTS manuscript_characters_sync_delete;
DROP TRIGGER IF EXISTS manuscript_characters_sync_undelete;

CREATE TRIGGER manuscript_characters_sync_create
AFTER INSERT ON manuscript_characters BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('manuscript_characters', NEW.id, 'create', NEW.last_modified, NEW.client_id, NEW.version,
         json_object('id',NEW.id,'project_id',NEW.project_id,'name',NEW.name,'role',NEW.role,
                     'cast_group',NEW.cast_group,'full_name',NEW.full_name,
                     'age',NEW.age,'gender',NEW.gender,
                     'appearance',NEW.appearance,'personality',NEW.personality,
                     'backstory',NEW.backstory,'motivation',NEW.motivation,
                     'arc_summary',NEW.arc_summary,'notes',NEW.notes,
                     'custom_fields_json',NEW.custom_fields_json,
                     'sort_order',NEW.sort_order,
                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER manuscript_characters_sync_update
AFTER UPDATE ON manuscript_characters
WHEN OLD.deleted = NEW.deleted AND (
     OLD.name IS NOT NEW.name OR
     OLD.role IS NOT NEW.role OR
     OLD.cast_group IS NOT NEW.cast_group OR
     OLD.full_name IS NOT NEW.full_name OR
     OLD.age IS NOT NEW.age OR
     OLD.gender IS NOT NEW.gender OR
     OLD.appearance IS NOT NEW.appearance OR
     OLD.personality IS NOT NEW.personality OR
     OLD.backstory IS NOT NEW.backstory OR
     OLD.motivation IS NOT NEW.motivation OR
     OLD.arc_summary IS NOT NEW.arc_summary OR
     OLD.notes IS NOT NEW.notes OR
     OLD.custom_fields_json IS NOT NEW.custom_fields_json OR
     OLD.sort_order IS NOT NEW.sort_order OR
     OLD.age IS NOT NEW.age OR
     OLD.gender IS NOT NEW.gender OR
     OLD.notes IS NOT NEW.notes OR
     OLD.last_modified IS NOT NEW.last_modified OR
     OLD.version IS NOT NEW.version)
BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('manuscript_characters', NEW.id, 'update', NEW.last_modified, NEW.client_id, NEW.version,
         json_object('id',NEW.id,'project_id',NEW.project_id,'name',NEW.name,'role',NEW.role,
                     'cast_group',NEW.cast_group,'full_name',NEW.full_name,
                     'age',NEW.age,'gender',NEW.gender,
                     'appearance',NEW.appearance,'personality',NEW.personality,
                     'backstory',NEW.backstory,'motivation',NEW.motivation,
                     'arc_summary',NEW.arc_summary,'notes',NEW.notes,
                     'custom_fields_json',NEW.custom_fields_json,
                     'sort_order',NEW.sort_order,
                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER manuscript_characters_sync_delete
AFTER UPDATE ON manuscript_characters
WHEN OLD.deleted = 0 AND NEW.deleted = 1
BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('manuscript_characters', NEW.id, 'delete', NEW.last_modified, NEW.client_id, NEW.version,
         json_object('id',NEW.id,'deleted',NEW.deleted,'last_modified',NEW.last_modified,
                     'version',NEW.version,'client_id',NEW.client_id));
END;

CREATE TRIGGER manuscript_characters_sync_undelete
AFTER UPDATE ON manuscript_characters
WHEN OLD.deleted = 1 AND NEW.deleted = 0
BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('manuscript_characters', NEW.id, 'update', NEW.last_modified, NEW.client_id, NEW.version,
         json_object('id',NEW.id,'project_id',NEW.project_id,'name',NEW.name,'role',NEW.role,
                     'cast_group',NEW.cast_group,'full_name',NEW.full_name,
                     'age',NEW.age,'gender',NEW.gender,
                     'appearance',NEW.appearance,'personality',NEW.personality,
                     'backstory',NEW.backstory,'motivation',NEW.motivation,
                     'arc_summary',NEW.arc_summary,'notes',NEW.notes,
                     'custom_fields_json',NEW.custom_fields_json,
                     'sort_order',NEW.sort_order,
                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

/* ── sync triggers: manuscript_character_relationships ─────── */
DROP TRIGGER IF EXISTS manuscript_character_relationships_sync_create;
DROP TRIGGER IF EXISTS manuscript_character_relationships_sync_update;
DROP TRIGGER IF EXISTS manuscript_character_relationships_sync_delete;
DROP TRIGGER IF EXISTS manuscript_character_relationships_sync_undelete;

CREATE TRIGGER manuscript_character_relationships_sync_create
AFTER INSERT ON manuscript_character_relationships BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('manuscript_character_relationships', NEW.id, 'create', NEW.last_modified, NEW.client_id, NEW.version,
         json_object('id',NEW.id,'project_id',NEW.project_id,
                     'from_character_id',NEW.from_character_id,'to_character_id',NEW.to_character_id,
                     'relationship_type',NEW.relationship_type,'description',NEW.description,
                     'bidirectional',NEW.bidirectional,
                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER manuscript_character_relationships_sync_update
AFTER UPDATE ON manuscript_character_relationships
WHEN OLD.deleted = NEW.deleted AND (
     OLD.relationship_type IS NOT NEW.relationship_type OR
     OLD.description IS NOT NEW.description OR
     OLD.bidirectional IS NOT NEW.bidirectional OR
     OLD.last_modified IS NOT NEW.last_modified OR
     OLD.version IS NOT NEW.version)
BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('manuscript_character_relationships', NEW.id, 'update', NEW.last_modified, NEW.client_id, NEW.version,
         json_object('id',NEW.id,'project_id',NEW.project_id,
                     'from_character_id',NEW.from_character_id,'to_character_id',NEW.to_character_id,
                     'relationship_type',NEW.relationship_type,'description',NEW.description,
                     'bidirectional',NEW.bidirectional,
                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER manuscript_character_relationships_sync_delete
AFTER UPDATE ON manuscript_character_relationships
WHEN OLD.deleted = 0 AND NEW.deleted = 1
BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('manuscript_character_relationships', NEW.id, 'delete', NEW.last_modified, NEW.client_id, NEW.version,
         json_object('id',NEW.id,'deleted',NEW.deleted,'last_modified',NEW.last_modified,
                     'version',NEW.version,'client_id',NEW.client_id));
END;

CREATE TRIGGER manuscript_character_relationships_sync_undelete
AFTER UPDATE ON manuscript_character_relationships
WHEN OLD.deleted = 1 AND NEW.deleted = 0
BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('manuscript_character_relationships', NEW.id, 'update', NEW.last_modified, NEW.client_id, NEW.version,
         json_object('id',NEW.id,'project_id',NEW.project_id,
                     'from_character_id',NEW.from_character_id,'to_character_id',NEW.to_character_id,
                     'relationship_type',NEW.relationship_type,'description',NEW.description,
                     'bidirectional',NEW.bidirectional,
                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

/* ── sync triggers: manuscript_world_info ─────────────────── */
DROP TRIGGER IF EXISTS manuscript_world_info_sync_create;
DROP TRIGGER IF EXISTS manuscript_world_info_sync_update;
DROP TRIGGER IF EXISTS manuscript_world_info_sync_delete;
DROP TRIGGER IF EXISTS manuscript_world_info_sync_undelete;

CREATE TRIGGER manuscript_world_info_sync_create
AFTER INSERT ON manuscript_world_info BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('manuscript_world_info', NEW.id, 'create', NEW.last_modified, NEW.client_id, NEW.version,
         json_object('id',NEW.id,'project_id',NEW.project_id,'kind',NEW.kind,'name',NEW.name,
                     'description',NEW.description,'parent_id',NEW.parent_id,
                     'properties_json',NEW.properties_json,'tags_json',NEW.tags_json,
                     'sort_order',NEW.sort_order,
                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER manuscript_world_info_sync_update
AFTER UPDATE ON manuscript_world_info
WHEN OLD.deleted = NEW.deleted AND (
     OLD.kind IS NOT NEW.kind OR
     OLD.name IS NOT NEW.name OR
     OLD.description IS NOT NEW.description OR
     OLD.parent_id IS NOT NEW.parent_id OR
     OLD.properties_json IS NOT NEW.properties_json OR
     OLD.tags_json IS NOT NEW.tags_json OR
     OLD.sort_order IS NOT NEW.sort_order OR
     OLD.last_modified IS NOT NEW.last_modified OR
     OLD.version IS NOT NEW.version)
BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('manuscript_world_info', NEW.id, 'update', NEW.last_modified, NEW.client_id, NEW.version,
         json_object('id',NEW.id,'project_id',NEW.project_id,'kind',NEW.kind,'name',NEW.name,
                     'description',NEW.description,'parent_id',NEW.parent_id,
                     'properties_json',NEW.properties_json,'tags_json',NEW.tags_json,
                     'sort_order',NEW.sort_order,
                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER manuscript_world_info_sync_delete
AFTER UPDATE ON manuscript_world_info
WHEN OLD.deleted = 0 AND NEW.deleted = 1
BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('manuscript_world_info', NEW.id, 'delete', NEW.last_modified, NEW.client_id, NEW.version,
         json_object('id',NEW.id,'deleted',NEW.deleted,'last_modified',NEW.last_modified,
                     'version',NEW.version,'client_id',NEW.client_id));
END;

CREATE TRIGGER manuscript_world_info_sync_undelete
AFTER UPDATE ON manuscript_world_info
WHEN OLD.deleted = 1 AND NEW.deleted = 0
BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('manuscript_world_info', NEW.id, 'update', NEW.last_modified, NEW.client_id, NEW.version,
         json_object('id',NEW.id,'project_id',NEW.project_id,'kind',NEW.kind,'name',NEW.name,
                     'description',NEW.description,'parent_id',NEW.parent_id,
                     'properties_json',NEW.properties_json,'tags_json',NEW.tags_json,
                     'sort_order',NEW.sort_order,
                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

/* ── sync triggers: manuscript_plot_lines ─────────────────── */
DROP TRIGGER IF EXISTS manuscript_plot_lines_sync_create;
DROP TRIGGER IF EXISTS manuscript_plot_lines_sync_update;
DROP TRIGGER IF EXISTS manuscript_plot_lines_sync_delete;
DROP TRIGGER IF EXISTS manuscript_plot_lines_sync_undelete;

CREATE TRIGGER manuscript_plot_lines_sync_create
AFTER INSERT ON manuscript_plot_lines BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('manuscript_plot_lines', NEW.id, 'create', NEW.last_modified, NEW.client_id, NEW.version,
         json_object('id',NEW.id,'project_id',NEW.project_id,'title',NEW.title,
                     'description',NEW.description,'status',NEW.status,'color',NEW.color,
                     'sort_order',NEW.sort_order,
                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER manuscript_plot_lines_sync_update
AFTER UPDATE ON manuscript_plot_lines
WHEN OLD.deleted = NEW.deleted AND (
     OLD.title IS NOT NEW.title OR
     OLD.description IS NOT NEW.description OR
     OLD.status IS NOT NEW.status OR
     OLD.color IS NOT NEW.color OR
     OLD.sort_order IS NOT NEW.sort_order OR
     OLD.last_modified IS NOT NEW.last_modified OR
     OLD.version IS NOT NEW.version)
BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('manuscript_plot_lines', NEW.id, 'update', NEW.last_modified, NEW.client_id, NEW.version,
         json_object('id',NEW.id,'project_id',NEW.project_id,'title',NEW.title,
                     'description',NEW.description,'status',NEW.status,'color',NEW.color,
                     'sort_order',NEW.sort_order,
                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER manuscript_plot_lines_sync_delete
AFTER UPDATE ON manuscript_plot_lines
WHEN OLD.deleted = 0 AND NEW.deleted = 1
BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('manuscript_plot_lines', NEW.id, 'delete', NEW.last_modified, NEW.client_id, NEW.version,
         json_object('id',NEW.id,'deleted',NEW.deleted,'last_modified',NEW.last_modified,
                     'version',NEW.version,'client_id',NEW.client_id));
END;

CREATE TRIGGER manuscript_plot_lines_sync_undelete
AFTER UPDATE ON manuscript_plot_lines
WHEN OLD.deleted = 1 AND NEW.deleted = 0
BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('manuscript_plot_lines', NEW.id, 'update', NEW.last_modified, NEW.client_id, NEW.version,
         json_object('id',NEW.id,'project_id',NEW.project_id,'title',NEW.title,
                     'description',NEW.description,'status',NEW.status,'color',NEW.color,
                     'sort_order',NEW.sort_order,
                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

/* ── sync triggers: manuscript_plot_events ─────────────────── */
DROP TRIGGER IF EXISTS manuscript_plot_events_sync_create;
DROP TRIGGER IF EXISTS manuscript_plot_events_sync_update;
DROP TRIGGER IF EXISTS manuscript_plot_events_sync_delete;
DROP TRIGGER IF EXISTS manuscript_plot_events_sync_undelete;

CREATE TRIGGER manuscript_plot_events_sync_create
AFTER INSERT ON manuscript_plot_events BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('manuscript_plot_events', NEW.id, 'create', NEW.last_modified, NEW.client_id, NEW.version,
         json_object('id',NEW.id,'project_id',NEW.project_id,'plot_line_id',NEW.plot_line_id,
                     'scene_id',NEW.scene_id,'chapter_id',NEW.chapter_id,
                     'title',NEW.title,'description',NEW.description,'event_type',NEW.event_type,
                     'sort_order',NEW.sort_order,
                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER manuscript_plot_events_sync_update
AFTER UPDATE ON manuscript_plot_events
WHEN OLD.deleted = NEW.deleted AND (
     OLD.title IS NOT NEW.title OR
     OLD.description IS NOT NEW.description OR
     OLD.event_type IS NOT NEW.event_type OR
     OLD.scene_id IS NOT NEW.scene_id OR
     OLD.chapter_id IS NOT NEW.chapter_id OR
     OLD.plot_line_id IS NOT NEW.plot_line_id OR
     OLD.sort_order IS NOT NEW.sort_order OR
     OLD.last_modified IS NOT NEW.last_modified OR
     OLD.version IS NOT NEW.version)
BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('manuscript_plot_events', NEW.id, 'update', NEW.last_modified, NEW.client_id, NEW.version,
         json_object('id',NEW.id,'project_id',NEW.project_id,'plot_line_id',NEW.plot_line_id,
                     'scene_id',NEW.scene_id,'chapter_id',NEW.chapter_id,
                     'title',NEW.title,'description',NEW.description,'event_type',NEW.event_type,
                     'sort_order',NEW.sort_order,
                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER manuscript_plot_events_sync_delete
AFTER UPDATE ON manuscript_plot_events
WHEN OLD.deleted = 0 AND NEW.deleted = 1
BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('manuscript_plot_events', NEW.id, 'delete', NEW.last_modified, NEW.client_id, NEW.version,
         json_object('id',NEW.id,'deleted',NEW.deleted,'last_modified',NEW.last_modified,
                     'version',NEW.version,'client_id',NEW.client_id));
END;

CREATE TRIGGER manuscript_plot_events_sync_undelete
AFTER UPDATE ON manuscript_plot_events
WHEN OLD.deleted = 1 AND NEW.deleted = 0
BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('manuscript_plot_events', NEW.id, 'update', NEW.last_modified, NEW.client_id, NEW.version,
         json_object('id',NEW.id,'project_id',NEW.project_id,'plot_line_id',NEW.plot_line_id,
                     'scene_id',NEW.scene_id,'chapter_id',NEW.chapter_id,
                     'title',NEW.title,'description',NEW.description,'event_type',NEW.event_type,
                     'sort_order',NEW.sort_order,
                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

/* ── sync triggers: manuscript_plot_holes ──────────────────── */
DROP TRIGGER IF EXISTS manuscript_plot_holes_sync_create;
DROP TRIGGER IF EXISTS manuscript_plot_holes_sync_update;
DROP TRIGGER IF EXISTS manuscript_plot_holes_sync_delete;
DROP TRIGGER IF EXISTS manuscript_plot_holes_sync_undelete;

CREATE TRIGGER manuscript_plot_holes_sync_create
AFTER INSERT ON manuscript_plot_holes BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('manuscript_plot_holes', NEW.id, 'create', NEW.last_modified, NEW.client_id, NEW.version,
         json_object('id',NEW.id,'project_id',NEW.project_id,'title',NEW.title,
                     'description',NEW.description,'severity',NEW.severity,'status',NEW.status,
                     'scene_id',NEW.scene_id,'chapter_id',NEW.chapter_id,'plot_line_id',NEW.plot_line_id,
                     'resolution',NEW.resolution,'detected_by',NEW.detected_by,
                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER manuscript_plot_holes_sync_update
AFTER UPDATE ON manuscript_plot_holes
WHEN OLD.deleted = NEW.deleted AND (
     OLD.title IS NOT NEW.title OR
     OLD.description IS NOT NEW.description OR
     OLD.severity IS NOT NEW.severity OR
     OLD.status IS NOT NEW.status OR
     OLD.resolution IS NOT NEW.resolution OR
     OLD.scene_id IS NOT NEW.scene_id OR
     OLD.chapter_id IS NOT NEW.chapter_id OR
     OLD.plot_line_id IS NOT NEW.plot_line_id OR
     OLD.detected_by IS NOT NEW.detected_by OR
     OLD.last_modified IS NOT NEW.last_modified OR
     OLD.version IS NOT NEW.version)
BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('manuscript_plot_holes', NEW.id, 'update', NEW.last_modified, NEW.client_id, NEW.version,
         json_object('id',NEW.id,'project_id',NEW.project_id,'title',NEW.title,
                     'description',NEW.description,'severity',NEW.severity,'status',NEW.status,
                     'scene_id',NEW.scene_id,'chapter_id',NEW.chapter_id,'plot_line_id',NEW.plot_line_id,
                     'resolution',NEW.resolution,'detected_by',NEW.detected_by,
                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER manuscript_plot_holes_sync_delete
AFTER UPDATE ON manuscript_plot_holes
WHEN OLD.deleted = 0 AND NEW.deleted = 1
BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('manuscript_plot_holes', NEW.id, 'delete', NEW.last_modified, NEW.client_id, NEW.version,
         json_object('id',NEW.id,'deleted',NEW.deleted,'last_modified',NEW.last_modified,
                     'version',NEW.version,'client_id',NEW.client_id));
END;

CREATE TRIGGER manuscript_plot_holes_sync_undelete
AFTER UPDATE ON manuscript_plot_holes
WHEN OLD.deleted = 1 AND NEW.deleted = 0
BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('manuscript_plot_holes', NEW.id, 'update', NEW.last_modified, NEW.client_id, NEW.version,
         json_object('id',NEW.id,'project_id',NEW.project_id,'title',NEW.title,
                     'description',NEW.description,'severity',NEW.severity,'status',NEW.status,
                     'scene_id',NEW.scene_id,'chapter_id',NEW.chapter_id,'plot_line_id',NEW.plot_line_id,
                     'resolution',NEW.resolution,'detected_by',NEW.detected_by,
                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

/* ── sync triggers: manuscript_citations ───────────────────── */
DROP TRIGGER IF EXISTS manuscript_citations_sync_create;
DROP TRIGGER IF EXISTS manuscript_citations_sync_update;
DROP TRIGGER IF EXISTS manuscript_citations_sync_delete;
DROP TRIGGER IF EXISTS manuscript_citations_sync_undelete;

CREATE TRIGGER manuscript_citations_sync_create
AFTER INSERT ON manuscript_citations BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('manuscript_citations', NEW.id, 'create', NEW.last_modified, NEW.client_id, NEW.version,
         json_object('id',NEW.id,'project_id',NEW.project_id,'scene_id',NEW.scene_id,
                     'source_type',NEW.source_type,'source_id',NEW.source_id,
                     'source_title',NEW.source_title,'excerpt',NEW.excerpt,
                     'query_used',NEW.query_used,'anchor_offset',NEW.anchor_offset,
                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER manuscript_citations_sync_update
AFTER UPDATE ON manuscript_citations
WHEN OLD.deleted = NEW.deleted AND (
     OLD.source_type IS NOT NEW.source_type OR
     OLD.source_id IS NOT NEW.source_id OR
     OLD.source_title IS NOT NEW.source_title OR
     OLD.excerpt IS NOT NEW.excerpt OR
     OLD.query_used IS NOT NEW.query_used OR
     OLD.anchor_offset IS NOT NEW.anchor_offset OR
     OLD.last_modified IS NOT NEW.last_modified OR
     OLD.version IS NOT NEW.version)
BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('manuscript_citations', NEW.id, 'update', NEW.last_modified, NEW.client_id, NEW.version,
         json_object('id',NEW.id,'project_id',NEW.project_id,'scene_id',NEW.scene_id,
                     'source_type',NEW.source_type,'source_id',NEW.source_id,
                     'source_title',NEW.source_title,'excerpt',NEW.excerpt,
                     'query_used',NEW.query_used,'anchor_offset',NEW.anchor_offset,
                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER manuscript_citations_sync_delete
AFTER UPDATE ON manuscript_citations
WHEN OLD.deleted = 0 AND NEW.deleted = 1
BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('manuscript_citations', NEW.id, 'delete', NEW.last_modified, NEW.client_id, NEW.version,
         json_object('id',NEW.id,'deleted',NEW.deleted,'last_modified',NEW.last_modified,
                     'version',NEW.version,'client_id',NEW.client_id));
END;

CREATE TRIGGER manuscript_citations_sync_undelete
AFTER UPDATE ON manuscript_citations
WHEN OLD.deleted = 1 AND NEW.deleted = 0
BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('manuscript_citations', NEW.id, 'update', NEW.last_modified, NEW.client_id, NEW.version,
         json_object('id',NEW.id,'project_id',NEW.project_id,'scene_id',NEW.scene_id,
                     'source_type',NEW.source_type,'source_id',NEW.source_id,
                     'source_title',NEW.source_title,'excerpt',NEW.excerpt,
                     'query_used',NEW.query_used,'anchor_offset',NEW.anchor_offset,
                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

UPDATE db_schema_version
   SET version = 42
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 42;
"""

_MIGRATION_SQL_V41_TO_V42_POSTGRES = """
/*───────────────────────────────────────────────────────────────
  Migration to Version 42 — Characters, world info, plot, citations (2026-04-XX) [Postgres]
───────────────────────────────────────────────────────────────*/

CREATE TABLE IF NOT EXISTS manuscript_characters (
  id              TEXT PRIMARY KEY,
  project_id      TEXT NOT NULL REFERENCES manuscript_projects(id) ON DELETE CASCADE,
  name            TEXT NOT NULL,
  role            TEXT NOT NULL DEFAULT 'supporting'
                    CHECK(role IN ('protagonist','antagonist','supporting','minor','mentioned')),
  cast_group      TEXT,
  full_name       TEXT,
  age             TEXT,
  gender          TEXT,
  appearance      TEXT,
  personality     TEXT,
  backstory       TEXT,
  motivation      TEXT,
  arc_summary     TEXT,
  notes           TEXT,
  custom_fields_json TEXT NOT NULL DEFAULT '{}',
  sort_order      REAL NOT NULL DEFAULT 0,
  created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted         BOOLEAN NOT NULL DEFAULT FALSE,
  client_id       TEXT NOT NULL DEFAULT 'unknown',
  version         INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_mchr_project ON manuscript_characters(project_id, sort_order);
CREATE INDEX IF NOT EXISTS idx_mchr_role ON manuscript_characters(project_id, role);
CREATE INDEX IF NOT EXISTS idx_mchr_deleted ON manuscript_characters(deleted);

CREATE TABLE IF NOT EXISTS manuscript_character_relationships (
  id                TEXT PRIMARY KEY,
  project_id        TEXT NOT NULL REFERENCES manuscript_projects(id) ON DELETE CASCADE,
  from_character_id TEXT NOT NULL REFERENCES manuscript_characters(id) ON DELETE CASCADE,
  to_character_id   TEXT NOT NULL REFERENCES manuscript_characters(id) ON DELETE CASCADE,
  relationship_type TEXT NOT NULL,
  description       TEXT,
  bidirectional     BOOLEAN NOT NULL DEFAULT TRUE,
  created_at        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified     TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted           BOOLEAN NOT NULL DEFAULT FALSE,
  client_id         TEXT NOT NULL DEFAULT 'unknown',
  version           INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_mcrel_project ON manuscript_character_relationships(project_id);
CREATE INDEX IF NOT EXISTS idx_mcrel_from ON manuscript_character_relationships(from_character_id);
CREATE INDEX IF NOT EXISTS idx_mcrel_to ON manuscript_character_relationships(to_character_id);
CREATE INDEX IF NOT EXISTS idx_mcrel_deleted ON manuscript_character_relationships(deleted);

CREATE TABLE IF NOT EXISTS manuscript_world_info (
  id              TEXT PRIMARY KEY,
  project_id      TEXT NOT NULL REFERENCES manuscript_projects(id) ON DELETE CASCADE,
  kind            TEXT NOT NULL CHECK(kind IN ('location','item','faction','concept','event','custom')),
  name            TEXT NOT NULL,
  description     TEXT,
  parent_id       TEXT REFERENCES manuscript_world_info(id) ON DELETE SET NULL,
  properties_json TEXT NOT NULL DEFAULT '{}',
  tags_json       TEXT NOT NULL DEFAULT '[]',
  sort_order      REAL NOT NULL DEFAULT 0,
  created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted         BOOLEAN NOT NULL DEFAULT FALSE,
  client_id       TEXT NOT NULL DEFAULT 'unknown',
  version         INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_mwi_project_kind ON manuscript_world_info(project_id, kind);
CREATE INDEX IF NOT EXISTS idx_mwi_parent ON manuscript_world_info(parent_id);
CREATE INDEX IF NOT EXISTS idx_mwi_deleted ON manuscript_world_info(deleted);

CREATE TABLE IF NOT EXISTS manuscript_plot_lines (
  id            TEXT PRIMARY KEY,
  project_id    TEXT NOT NULL REFERENCES manuscript_projects(id) ON DELETE CASCADE,
  title         TEXT NOT NULL,
  description   TEXT,
  status        TEXT NOT NULL DEFAULT 'active'
                  CHECK(status IN ('active','resolved','abandoned','dormant')),
  color         TEXT,
  sort_order    REAL NOT NULL DEFAULT 0,
  created_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted       BOOLEAN NOT NULL DEFAULT FALSE,
  client_id     TEXT NOT NULL DEFAULT 'unknown',
  version       INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_mpl_project ON manuscript_plot_lines(project_id);
CREATE INDEX IF NOT EXISTS idx_mpl_deleted ON manuscript_plot_lines(deleted);

CREATE TABLE IF NOT EXISTS manuscript_plot_events (
  id            TEXT PRIMARY KEY,
  project_id    TEXT NOT NULL REFERENCES manuscript_projects(id) ON DELETE CASCADE,
  plot_line_id  TEXT NOT NULL REFERENCES manuscript_plot_lines(id) ON DELETE CASCADE,
  scene_id      TEXT REFERENCES manuscript_scenes(id) ON DELETE SET NULL,
  chapter_id    TEXT REFERENCES manuscript_chapters(id) ON DELETE SET NULL,
  title         TEXT NOT NULL,
  description   TEXT,
  event_type    TEXT NOT NULL DEFAULT 'plot'
                  CHECK(event_type IN ('setup','conflict','action','emotional','plot','resolution')),
  sort_order    REAL NOT NULL DEFAULT 0,
  created_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted       BOOLEAN NOT NULL DEFAULT FALSE,
  client_id     TEXT NOT NULL DEFAULT 'unknown',
  version       INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_mpe_plot_line ON manuscript_plot_events(plot_line_id, sort_order);
CREATE INDEX IF NOT EXISTS idx_mpe_scene ON manuscript_plot_events(scene_id);
CREATE INDEX IF NOT EXISTS idx_mpe_deleted ON manuscript_plot_events(deleted);

CREATE TABLE IF NOT EXISTS manuscript_plot_holes (
  id            TEXT PRIMARY KEY,
  project_id    TEXT NOT NULL REFERENCES manuscript_projects(id) ON DELETE CASCADE,
  title         TEXT NOT NULL,
  description   TEXT,
  severity      TEXT NOT NULL DEFAULT 'medium'
                  CHECK(severity IN ('low','medium','high','critical')),
  status        TEXT NOT NULL DEFAULT 'open'
                  CHECK(status IN ('open','investigating','resolved','wontfix')),
  scene_id      TEXT REFERENCES manuscript_scenes(id) ON DELETE SET NULL,
  chapter_id    TEXT REFERENCES manuscript_chapters(id) ON DELETE SET NULL,
  plot_line_id  TEXT REFERENCES manuscript_plot_lines(id) ON DELETE SET NULL,
  resolution    TEXT,
  detected_by   TEXT NOT NULL DEFAULT 'manual'
                  CHECK(detected_by IN ('manual','ai')),
  created_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted       BOOLEAN NOT NULL DEFAULT FALSE,
  client_id     TEXT NOT NULL DEFAULT 'unknown',
  version       INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_mph_project ON manuscript_plot_holes(project_id, status);
CREATE INDEX IF NOT EXISTS idx_mph_deleted ON manuscript_plot_holes(deleted);

CREATE TABLE IF NOT EXISTS manuscript_scene_characters (
  scene_id      TEXT NOT NULL REFERENCES manuscript_scenes(id) ON DELETE CASCADE,
  character_id  TEXT NOT NULL REFERENCES manuscript_characters(id) ON DELETE CASCADE,
  is_pov        BOOLEAN NOT NULL DEFAULT FALSE,
  last_modified TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted       BOOLEAN NOT NULL DEFAULT FALSE,
  client_id     TEXT,
  version       INTEGER NOT NULL DEFAULT 1,
  PRIMARY KEY (scene_id, character_id)
);

CREATE TABLE IF NOT EXISTS manuscript_scene_world_info (
  scene_id        TEXT NOT NULL REFERENCES manuscript_scenes(id) ON DELETE CASCADE,
  world_info_id   TEXT NOT NULL REFERENCES manuscript_world_info(id) ON DELETE CASCADE,
  last_modified   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted         BOOLEAN NOT NULL DEFAULT FALSE,
  client_id       TEXT,
  version         INTEGER NOT NULL DEFAULT 1,
  PRIMARY KEY (scene_id, world_info_id)
);

CREATE TABLE IF NOT EXISTS manuscript_citations (
  id            TEXT PRIMARY KEY,
  project_id    TEXT NOT NULL REFERENCES manuscript_projects(id) ON DELETE CASCADE,
  scene_id      TEXT NOT NULL REFERENCES manuscript_scenes(id) ON DELETE CASCADE,
  source_type   TEXT NOT NULL,
  source_id     TEXT,
  source_title  TEXT,
  excerpt       TEXT,
  query_used    TEXT,
  anchor_offset INTEGER,
  created_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted       BOOLEAN NOT NULL DEFAULT FALSE,
  client_id     TEXT NOT NULL DEFAULT 'unknown',
  version       INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_mcit_scene ON manuscript_citations(scene_id);
CREATE INDEX IF NOT EXISTS idx_mcit_project ON manuscript_citations(project_id);
CREATE INDEX IF NOT EXISTS idx_mcit_deleted ON manuscript_citations(deleted);

UPDATE db_schema_version
   SET version = 42
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 42;
"""

# --- Migration: V42 -> V43 (AI Analyses) ---
_MIGRATION_SQL_V42_TO_V43 = """
/*───────────────────────────────────────────────────────────────
  Migration to Version 43 — Manuscript AI Analyses (2026-04-XX)
───────────────────────────────────────────────────────────────*/

CREATE TABLE IF NOT EXISTS manuscript_ai_analyses (
  id            TEXT PRIMARY KEY,
  project_id    TEXT NOT NULL REFERENCES manuscript_projects(id) ON DELETE CASCADE,
  scope_type    TEXT NOT NULL CHECK(scope_type IN ('scene','chapter','part','project')),
  scope_id      TEXT NOT NULL,
  analysis_type TEXT NOT NULL,
  provider      TEXT,
  model         TEXT,
  result_json   TEXT NOT NULL DEFAULT '{}',
  score         REAL,
  stale         BOOLEAN NOT NULL DEFAULT 0,
  created_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted       BOOLEAN NOT NULL DEFAULT 0,
  client_id     TEXT NOT NULL DEFAULT 'unknown',
  version       INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_maa_scope ON manuscript_ai_analyses(scope_type, scope_id);
CREATE INDEX IF NOT EXISTS idx_maa_project_type ON manuscript_ai_analyses(project_id, analysis_type);
CREATE INDEX IF NOT EXISTS idx_maa_stale ON manuscript_ai_analyses(stale);
CREATE INDEX IF NOT EXISTS idx_maa_deleted ON manuscript_ai_analyses(deleted);

/* ── sync triggers: manuscript_ai_analyses ─────────────────── */
DROP TRIGGER IF EXISTS manuscript_ai_analyses_sync_create;
DROP TRIGGER IF EXISTS manuscript_ai_analyses_sync_update;
DROP TRIGGER IF EXISTS manuscript_ai_analyses_sync_delete;
DROP TRIGGER IF EXISTS manuscript_ai_analyses_sync_undelete;

CREATE TRIGGER manuscript_ai_analyses_sync_create
AFTER INSERT ON manuscript_ai_analyses BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('manuscript_ai_analyses', NEW.id, 'create', NEW.last_modified, NEW.client_id, NEW.version,
         json_object('id',NEW.id,'project_id',NEW.project_id,'scope_type',NEW.scope_type,
                     'scope_id',NEW.scope_id,'analysis_type',NEW.analysis_type,
                     'result_json',NEW.result_json,
                     'provider',NEW.provider,'model',NEW.model,'score',NEW.score,
                     'stale',NEW.stale,
                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER manuscript_ai_analyses_sync_update
AFTER UPDATE ON manuscript_ai_analyses
WHEN OLD.deleted = NEW.deleted AND (
     OLD.analysis_type IS NOT NEW.analysis_type OR
     OLD.result_json IS NOT NEW.result_json OR
     OLD.score IS NOT NEW.score OR
     OLD.stale IS NOT NEW.stale OR
     OLD.provider IS NOT NEW.provider OR
     OLD.model IS NOT NEW.model OR
     OLD.last_modified IS NOT NEW.last_modified OR
     OLD.version IS NOT NEW.version)
BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('manuscript_ai_analyses', NEW.id, 'update', NEW.last_modified, NEW.client_id, NEW.version,
         json_object('id',NEW.id,'project_id',NEW.project_id,'scope_type',NEW.scope_type,
                     'scope_id',NEW.scope_id,'analysis_type',NEW.analysis_type,
                     'result_json',NEW.result_json,
                     'provider',NEW.provider,'model',NEW.model,'score',NEW.score,
                     'stale',NEW.stale,
                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER manuscript_ai_analyses_sync_delete
AFTER UPDATE ON manuscript_ai_analyses
WHEN OLD.deleted = 0 AND NEW.deleted = 1
BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('manuscript_ai_analyses', NEW.id, 'delete', NEW.last_modified, NEW.client_id, NEW.version,
         json_object('id',NEW.id,'deleted',NEW.deleted,'last_modified',NEW.last_modified,
                     'version',NEW.version,'client_id',NEW.client_id));
END;

CREATE TRIGGER manuscript_ai_analyses_sync_undelete
AFTER UPDATE ON manuscript_ai_analyses
WHEN OLD.deleted = 1 AND NEW.deleted = 0
BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('manuscript_ai_analyses', NEW.id, 'update', NEW.last_modified, NEW.client_id, NEW.version,
         json_object('id',NEW.id,'project_id',NEW.project_id,'scope_type',NEW.scope_type,
                     'scope_id',NEW.scope_id,'analysis_type',NEW.analysis_type,
                     'provider',NEW.provider,'model',NEW.model,'result_json',NEW.result_json,'score',NEW.score,
                     'stale',NEW.stale,
                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

UPDATE db_schema_version
   SET version = 43
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 43;
"""

_MIGRATION_SQL_V42_TO_V43_POSTGRES = """
/*───────────────────────────────────────────────────────────────
  Migration to Version 43 — Manuscript AI Analyses (2026-04-XX) [Postgres]
───────────────────────────────────────────────────────────────*/

CREATE TABLE IF NOT EXISTS manuscript_ai_analyses (
  id            TEXT PRIMARY KEY,
  project_id    TEXT NOT NULL REFERENCES manuscript_projects(id) ON DELETE CASCADE,
  scope_type    TEXT NOT NULL CHECK(scope_type IN ('scene','chapter','part','project')),
  scope_id      TEXT NOT NULL,
  analysis_type TEXT NOT NULL,
  provider      TEXT,
  model         TEXT,
  result_json   TEXT NOT NULL DEFAULT '{}',
  score         REAL,
  stale         BOOLEAN NOT NULL DEFAULT FALSE,
  created_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted       BOOLEAN NOT NULL DEFAULT FALSE,
  client_id     TEXT NOT NULL DEFAULT 'unknown',
  version       INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_maa_scope ON manuscript_ai_analyses(scope_type, scope_id);
CREATE INDEX IF NOT EXISTS idx_maa_project_type ON manuscript_ai_analyses(project_id, analysis_type);
CREATE INDEX IF NOT EXISTS idx_maa_stale ON manuscript_ai_analyses(stale);
CREATE INDEX IF NOT EXISTS idx_maa_deleted ON manuscript_ai_analyses(deleted);

UPDATE db_schema_version
   SET version = 43
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 43;
"""

_MIGRATION_SQL_V43_TO_V44 = """
/*───────────────────────────────────────────────────────────────
  Migration to Version 44 — Sync metadata on scene-link tables (2026-04-XX)
  NOTE: ALTER TABLE ADD COLUMN is handled in _migrate_from_v43_to_v44()
  to avoid errors on fresh databases that already have the columns.
───────────────────────────────────────────────────────────────*/

-- Backfill last_modified for existing rows (no-op on fresh DBs)
UPDATE manuscript_scene_characters SET last_modified = strftime('%Y-%m-%dT%H:%M:%f','now') WHERE last_modified = '';
UPDATE manuscript_scene_world_info SET last_modified = strftime('%Y-%m-%dT%H:%M:%f','now') WHERE last_modified = '';

/* ── sync triggers: manuscript_scene_characters ────────────── */
DROP TRIGGER IF EXISTS msc_sync_create;
DROP TRIGGER IF EXISTS msc_sync_update;
DROP TRIGGER IF EXISTS msc_sync_delete;

CREATE TRIGGER msc_sync_create
AFTER INSERT ON manuscript_scene_characters BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('manuscript_scene_characters', NEW.scene_id || ':' || NEW.character_id, 'create',
         NEW.last_modified, NEW.client_id, NEW.version,
         json_object('scene_id',NEW.scene_id,'character_id',NEW.character_id,
                     'is_pov',NEW.is_pov,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER msc_sync_update
AFTER UPDATE ON manuscript_scene_characters
WHEN OLD.deleted = NEW.deleted AND (
     OLD.is_pov IS NOT NEW.is_pov OR
     OLD.last_modified IS NOT NEW.last_modified OR
     OLD.version IS NOT NEW.version)
BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('manuscript_scene_characters', NEW.scene_id || ':' || NEW.character_id, 'update',
         NEW.last_modified, NEW.client_id, NEW.version,
         json_object('scene_id',NEW.scene_id,'character_id',NEW.character_id,
                     'is_pov',NEW.is_pov,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER msc_sync_delete
AFTER UPDATE ON manuscript_scene_characters
WHEN OLD.deleted = 0 AND NEW.deleted = 1
BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('manuscript_scene_characters', NEW.scene_id || ':' || NEW.character_id, 'delete',
         NEW.last_modified, NEW.client_id, NEW.version,
         json_object('scene_id',NEW.scene_id,'character_id',NEW.character_id,
                     'deleted',NEW.deleted,'last_modified',NEW.last_modified,
                     'version',NEW.version,'client_id',NEW.client_id));
END;

/* ── sync triggers: manuscript_scene_world_info ────────────── */
DROP TRIGGER IF EXISTS mswi_sync_create;
DROP TRIGGER IF EXISTS mswi_sync_update;
DROP TRIGGER IF EXISTS mswi_sync_delete;

CREATE TRIGGER mswi_sync_create
AFTER INSERT ON manuscript_scene_world_info BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('manuscript_scene_world_info', NEW.scene_id || ':' || NEW.world_info_id, 'create',
         NEW.last_modified, NEW.client_id, NEW.version,
         json_object('scene_id',NEW.scene_id,'world_info_id',NEW.world_info_id,
                     'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER mswi_sync_update
AFTER UPDATE ON manuscript_scene_world_info
WHEN OLD.deleted = NEW.deleted AND (
     OLD.last_modified IS NOT NEW.last_modified OR
     OLD.version IS NOT NEW.version)
BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('manuscript_scene_world_info', NEW.scene_id || ':' || NEW.world_info_id, 'update',
         NEW.last_modified, NEW.client_id, NEW.version,
         json_object('scene_id',NEW.scene_id,'world_info_id',NEW.world_info_id,
                     'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER mswi_sync_delete
AFTER UPDATE ON manuscript_scene_world_info
WHEN OLD.deleted = 0 AND NEW.deleted = 1
BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('manuscript_scene_world_info', NEW.scene_id || ':' || NEW.world_info_id, 'delete',
         NEW.last_modified, NEW.client_id, NEW.version,
         json_object('scene_id',NEW.scene_id,'world_info_id',NEW.world_info_id,
                     'deleted',NEW.deleted,'last_modified',NEW.last_modified,
                     'version',NEW.version,'client_id',NEW.client_id));
END;

UPDATE db_schema_version
   SET version = 44
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 44;
"""

_MIGRATION_SQL_V43_TO_V44_POSTGRES = """
/*───────────────────────────────────────────────────────────────
  Migration to Version 44 — Sync metadata on scene-link tables (2026-04-XX) [Postgres]
───────────────────────────────────────────────────────────────*/

ALTER TABLE manuscript_scene_characters ADD COLUMN IF NOT EXISTS last_modified TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP;
ALTER TABLE manuscript_scene_characters ADD COLUMN IF NOT EXISTS deleted BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE manuscript_scene_characters ADD COLUMN IF NOT EXISTS client_id TEXT;
ALTER TABLE manuscript_scene_characters ADD COLUMN IF NOT EXISTS version INTEGER NOT NULL DEFAULT 1;

ALTER TABLE manuscript_scene_world_info ADD COLUMN IF NOT EXISTS last_modified TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP;
ALTER TABLE manuscript_scene_world_info ADD COLUMN IF NOT EXISTS deleted BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE manuscript_scene_world_info ADD COLUMN IF NOT EXISTS client_id TEXT;
ALTER TABLE manuscript_scene_world_info ADD COLUMN IF NOT EXISTS version INTEGER NOT NULL DEFAULT 1;

UPDATE db_schema_version
   SET version = 44
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 44;
"""

_MIGRATION_SQL_V44_TO_V45 = """
/*───────────────────────────────────────────────────────────────
  Migration to Version 45 — Persona visual packs (2026-05-09)
───────────────────────────────────────────────────────────────*/
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS persona_visual_packs (
  id               TEXT PRIMARY KEY,
  persona_id       TEXT NOT NULL REFERENCES persona_profiles(id) ON DELETE CASCADE,
  user_id          TEXT NOT NULL,
  title            TEXT NOT NULL,
  renderer_type    TEXT NOT NULL DEFAULT 'sprite_frames'
                   CHECK(renderer_type IN ('sprite_frames','sprite_sheet','static_image','live2d')),
  status           TEXT NOT NULL DEFAULT 'draft'
                   CHECK(status IN ('draft','review','active','archived','failed')),
  manifest_version INTEGER NOT NULL DEFAULT 1,
  manifest_json    TEXT NOT NULL DEFAULT '{}',
  parent_pack_id   TEXT REFERENCES persona_visual_packs(id) ON DELETE SET NULL,
  revision_number  INTEGER NOT NULL DEFAULT 1,
  provenance       TEXT NOT NULL DEFAULT 'uploaded'
                   CHECK(provenance IN ('uploaded','generated','imported','mixed')),
  active_at        TEXT,
  created_at       DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted          BOOLEAN NOT NULL DEFAULT 0,
  version          INTEGER NOT NULL DEFAULT 1
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_persona_visual_packs_one_active
  ON persona_visual_packs(user_id, persona_id)
  WHERE status = 'active' AND deleted = 0;
CREATE INDEX IF NOT EXISTS idx_persona_visual_packs_persona
  ON persona_visual_packs(user_id, persona_id, deleted, status);

CREATE TABLE IF NOT EXISTS persona_visual_assets (
  id                 TEXT PRIMARY KEY,
  pack_id            TEXT NOT NULL REFERENCES persona_visual_packs(id) ON DELETE CASCADE,
  persona_id         TEXT NOT NULL REFERENCES persona_profiles(id) ON DELETE CASCADE,
  user_id            TEXT NOT NULL,
  asset_role         TEXT NOT NULL
                     CHECK(asset_role IN ('frame','still_pose','sprite_sheet','preview','generated_candidate')),
  storage_key        TEXT NOT NULL,
  original_filename  TEXT,
  mime_type          TEXT NOT NULL,
  byte_size          INTEGER NOT NULL,
  checksum_sha256    TEXT NOT NULL,
  width              INTEGER,
  height             INTEGER,
  duration_ms        INTEGER,
  provenance         TEXT NOT NULL DEFAULT 'uploaded'
                     CHECK(provenance IN ('uploaded','generated','imported','mixed')),
  created_at         DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted            BOOLEAN NOT NULL DEFAULT 0,
  version            INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_persona_visual_assets_pack
  ON persona_visual_assets(user_id, pack_id, deleted);

CREATE TABLE IF NOT EXISTS persona_visual_candidates (
  id                           TEXT PRIMARY KEY,
  pack_id                      TEXT NOT NULL REFERENCES persona_visual_packs(id) ON DELETE CASCADE,
  persona_id                   TEXT NOT NULL REFERENCES persona_profiles(id) ON DELETE CASCADE,
  user_id                      TEXT NOT NULL,
  job_id                       TEXT,
  status                       TEXT NOT NULL DEFAULT 'review'
                               CHECK(status IN ('review','accepted','rejected','failed')),
  proposed_manifest_patch_json TEXT NOT NULL DEFAULT '{}',
  generated_asset_ids_json     TEXT NOT NULL DEFAULT '[]',
  prompt                       TEXT,
  failure_reason               TEXT,
  created_at                   DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified                DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted                      BOOLEAN NOT NULL DEFAULT 0,
  version                      INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_persona_visual_candidates_pack
  ON persona_visual_candidates(user_id, pack_id, status, deleted);

UPDATE db_schema_version
   SET version = 45
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 45;
"""

_MIGRATION_SQL_V44_TO_V45_POSTGRES = """
/*───────────────────────────────────────────────────────────────
  Migration to Version 45 — Persona visual packs (2026-05-09) [Postgres]
───────────────────────────────────────────────────────────────*/

CREATE TABLE IF NOT EXISTS persona_visual_packs (
  id               TEXT PRIMARY KEY,
  persona_id       TEXT NOT NULL REFERENCES persona_profiles(id) ON DELETE CASCADE,
  user_id          TEXT NOT NULL,
  title            TEXT NOT NULL,
  renderer_type    TEXT NOT NULL DEFAULT 'sprite_frames'
                   CHECK(renderer_type IN ('sprite_frames','sprite_sheet','static_image','live2d')),
  status           TEXT NOT NULL DEFAULT 'draft'
                   CHECK(status IN ('draft','review','active','archived','failed')),
  manifest_version INTEGER NOT NULL DEFAULT 1,
  manifest_json    TEXT NOT NULL DEFAULT '{}',
  parent_pack_id   TEXT REFERENCES persona_visual_packs(id) ON DELETE SET NULL,
  revision_number  INTEGER NOT NULL DEFAULT 1,
  provenance       TEXT NOT NULL DEFAULT 'uploaded'
                   CHECK(provenance IN ('uploaded','generated','imported','mixed')),
  active_at        TEXT,
  created_at       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted          BOOLEAN NOT NULL DEFAULT FALSE,
  version          INTEGER NOT NULL DEFAULT 1
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_persona_visual_packs_one_active
  ON persona_visual_packs(user_id, persona_id)
  WHERE status = 'active' AND deleted = FALSE;
CREATE INDEX IF NOT EXISTS idx_persona_visual_packs_persona
  ON persona_visual_packs(user_id, persona_id, deleted, status);

CREATE TABLE IF NOT EXISTS persona_visual_assets (
  id                 TEXT PRIMARY KEY,
  pack_id            TEXT NOT NULL REFERENCES persona_visual_packs(id) ON DELETE CASCADE,
  persona_id         TEXT NOT NULL REFERENCES persona_profiles(id) ON DELETE CASCADE,
  user_id            TEXT NOT NULL,
  asset_role         TEXT NOT NULL
                     CHECK(asset_role IN ('frame','still_pose','sprite_sheet','preview','generated_candidate')),
  storage_key        TEXT NOT NULL,
  original_filename  TEXT,
  mime_type          TEXT NOT NULL,
  byte_size          INTEGER NOT NULL,
  checksum_sha256    TEXT NOT NULL,
  width              INTEGER,
  height             INTEGER,
  duration_ms        INTEGER,
  provenance         TEXT NOT NULL DEFAULT 'uploaded'
                     CHECK(provenance IN ('uploaded','generated','imported','mixed')),
  created_at         TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted            BOOLEAN NOT NULL DEFAULT FALSE,
  version            INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_persona_visual_assets_pack
  ON persona_visual_assets(user_id, pack_id, deleted);

CREATE TABLE IF NOT EXISTS persona_visual_candidates (
  id                           TEXT PRIMARY KEY,
  pack_id                      TEXT NOT NULL REFERENCES persona_visual_packs(id) ON DELETE CASCADE,
  persona_id                   TEXT NOT NULL REFERENCES persona_profiles(id) ON DELETE CASCADE,
  user_id                      TEXT NOT NULL,
  job_id                       TEXT,
  status                       TEXT NOT NULL DEFAULT 'review'
                               CHECK(status IN ('review','accepted','rejected','failed')),
  proposed_manifest_patch_json TEXT NOT NULL DEFAULT '{}',
  generated_asset_ids_json     TEXT NOT NULL DEFAULT '[]',
  prompt                       TEXT,
  failure_reason               TEXT,
  created_at                   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified                TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted                      BOOLEAN NOT NULL DEFAULT FALSE,
  version                      INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_persona_visual_candidates_pack
  ON persona_visual_candidates(user_id, pack_id, status, deleted);

UPDATE db_schema_version
   SET version = 45
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 45;
"""

_MIGRATION_SQL_V45_TO_V46 = """
/*───────────────────────────────────────────────────────────────
  Migration to Version 46 — Persona visual personal library (2026-05-09)
───────────────────────────────────────────────────────────────*/
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS persona_visual_library_items (
  id                            TEXT PRIMARY KEY,
  user_id                       TEXT NOT NULL,
  source_persona_id             TEXT REFERENCES persona_profiles(id) ON DELETE SET NULL,
  source_pack_id                TEXT REFERENCES persona_visual_packs(id) ON DELETE SET NULL,
  title                         TEXT NOT NULL,
  notes                         TEXT,
  tags_json                     TEXT NOT NULL DEFAULT '[]',
  source_pack_version           INTEGER,
  created_at                    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified                 DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted                       BOOLEAN NOT NULL DEFAULT 0,
  version                       INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_persona_visual_library_items_user_time
  ON persona_visual_library_items(user_id, deleted, last_modified);

CREATE UNIQUE INDEX IF NOT EXISTS idx_persona_visual_library_items_live_source
  ON persona_visual_library_items(user_id, source_persona_id, source_pack_id)
  WHERE deleted = 0 AND source_persona_id IS NOT NULL AND source_pack_id IS NOT NULL;

UPDATE db_schema_version
   SET version = 46
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 46;
"""

_MIGRATION_SQL_V45_TO_V46_POSTGRES = """
/*───────────────────────────────────────────────────────────────
  Migration to Version 46 — Persona visual personal library (2026-05-09) [Postgres]
───────────────────────────────────────────────────────────────*/

CREATE TABLE IF NOT EXISTS persona_visual_library_items (
  id                            TEXT PRIMARY KEY,
  user_id                       TEXT NOT NULL,
  source_persona_id             TEXT REFERENCES persona_profiles(id) ON DELETE SET NULL,
  source_pack_id                TEXT REFERENCES persona_visual_packs(id) ON DELETE SET NULL,
  title                         TEXT NOT NULL,
  notes                         TEXT,
  tags_json                     TEXT NOT NULL DEFAULT '[]',
  source_pack_version           INTEGER,
  created_at                    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified                 TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted                       BOOLEAN NOT NULL DEFAULT FALSE,
  version                       INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_persona_visual_library_items_user_time
  ON persona_visual_library_items(user_id, deleted, last_modified);

CREATE UNIQUE INDEX IF NOT EXISTS idx_persona_visual_library_items_live_source
  ON persona_visual_library_items(user_id, source_persona_id, source_pack_id)
  WHERE deleted = FALSE AND source_persona_id IS NOT NULL AND source_pack_id IS NOT NULL;

UPDATE db_schema_version
   SET version = 46
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 46;
"""

_MIGRATION_SQL_V46_TO_V47 = """
/*───────────────────────────────────────────────────────────────
  Migration to Version 47 — Research Workspace migration protocol state (2026-05-23)
───────────────────────────────────────────────────────────────*/
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS workspace_migration_sessions (
  id                     TEXT PRIMARY KEY NOT NULL,
  target_workspace_id    TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
  target_workspace_name  TEXT NOT NULL,
  source_product         TEXT NOT NULL,
  idempotency_key        TEXT NOT NULL,
  manifest_hash          TEXT NOT NULL,
  manifest_json          TEXT NOT NULL DEFAULT '{}',
  diagnostics_json       TEXT NOT NULL DEFAULT '{}',
  declared_chunks_json   TEXT NOT NULL DEFAULT '[]',
  status                 TEXT NOT NULL DEFAULT 'created',
  finalized_at           DATETIME,
  recovery_manifest_json TEXT,
  client_delete_eligible BOOLEAN NOT NULL DEFAULT 0,
  client_delete_ack_at   DATETIME,
  created_at             DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at             DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  version                INTEGER NOT NULL DEFAULT 1
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_ws_migration_sessions_idempotency
  ON workspace_migration_sessions(idempotency_key);
CREATE INDEX IF NOT EXISTS idx_ws_migration_sessions_workspace
  ON workspace_migration_sessions(target_workspace_id);

CREATE TABLE IF NOT EXISTS workspace_migration_chunks (
  migration_id TEXT NOT NULL REFERENCES workspace_migration_sessions(id) ON DELETE CASCADE,
  id           TEXT NOT NULL,
  sha256       TEXT NOT NULL,
  byte_count   INTEGER NOT NULL,
  chunk_kind   TEXT NOT NULL DEFAULT 'workspace_bundle',
  metadata_json TEXT NOT NULL DEFAULT '{}',
  accepted_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (migration_id, id)
);

CREATE INDEX IF NOT EXISTS idx_ws_migration_chunks_migration
  ON workspace_migration_chunks(migration_id);

UPDATE db_schema_version
   SET version = 47
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 47;
"""

_MIGRATION_SQL_V46_TO_V47_POSTGRES = """
/*───────────────────────────────────────────────────────────────
  Migration to Version 47 — Research Workspace migration protocol state (2026-05-23) [Postgres]
───────────────────────────────────────────────────────────────*/

ALTER TABLE persona_visual_candidates
  ADD COLUMN IF NOT EXISTS generation_provenance_json TEXT NOT NULL DEFAULT '{}';

CREATE TABLE IF NOT EXISTS workspace_migration_sessions (
  id                     TEXT PRIMARY KEY NOT NULL,
  target_workspace_id    TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
  target_workspace_name  TEXT NOT NULL,
  source_product         TEXT NOT NULL,
  idempotency_key        TEXT NOT NULL,
  manifest_hash          TEXT NOT NULL,
  manifest_json          TEXT NOT NULL DEFAULT '{}',
  diagnostics_json       TEXT NOT NULL DEFAULT '{}',
  declared_chunks_json   TEXT NOT NULL DEFAULT '[]',
  status                 TEXT NOT NULL DEFAULT 'created',
  finalized_at           TIMESTAMP,
  recovery_manifest_json TEXT,
  client_delete_eligible BOOLEAN NOT NULL DEFAULT FALSE,
  client_delete_ack_at   TIMESTAMP,
  created_at             TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at             TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  version                INTEGER NOT NULL DEFAULT 1
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_ws_migration_sessions_idempotency
  ON workspace_migration_sessions(idempotency_key);
CREATE INDEX IF NOT EXISTS idx_ws_migration_sessions_workspace
  ON workspace_migration_sessions(target_workspace_id);

CREATE TABLE IF NOT EXISTS workspace_migration_chunks (
  migration_id TEXT NOT NULL REFERENCES workspace_migration_sessions(id) ON DELETE CASCADE,
  id           TEXT NOT NULL,
  sha256       TEXT NOT NULL,
  byte_count   INTEGER NOT NULL,
  chunk_kind   TEXT NOT NULL DEFAULT 'workspace_bundle',
  metadata_json TEXT NOT NULL DEFAULT '{}',
  accepted_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (migration_id, id)
);

CREATE INDEX IF NOT EXISTS idx_ws_migration_chunks_migration
  ON workspace_migration_chunks(migration_id);

UPDATE db_schema_version
   SET version = 47
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 47;
"""

_MIGRATION_SQL_V47_TO_V48 = """
/*───────────────────────────────────────────────────────────────
  Migration to Version 48 — Notes task-backed checklist storage (2026-06-05)
───────────────────────────────────────────────────────────────*/
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS note_tasks (
  id                TEXT PRIMARY KEY,
  note_id           TEXT NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
  text              TEXT NOT NULL,
  status            TEXT NOT NULL CHECK(status IN ('open','done')),
  metadata_json     TEXT NOT NULL DEFAULT '{}',
  projection_status TEXT NOT NULL DEFAULT 'live' CHECK(projection_status IN ('live','unlinked','ambiguous','deleted')),
  deleted           BOOLEAN NOT NULL DEFAULT 0,
  created_at        DATETIME NOT NULL,
  updated_at        DATETIME NOT NULL,
  completed_at      DATETIME,
  client_id         TEXT NOT NULL,
  version           INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_note_tasks_client_scope
  ON note_tasks(client_id, deleted, updated_at);
CREATE INDEX IF NOT EXISTS idx_note_tasks_note
  ON note_tasks(note_id, deleted);
CREATE INDEX IF NOT EXISTS idx_note_tasks_status
  ON note_tasks(status, deleted);
CREATE INDEX IF NOT EXISTS idx_note_tasks_projection_status
  ON note_tasks(projection_status, deleted);

CREATE TABLE IF NOT EXISTS task_note_projections (
  task_id              TEXT PRIMARY KEY REFERENCES note_tasks(id) ON DELETE CASCADE,
  note_id              TEXT NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
  note_version         INTEGER NOT NULL,
  line_number          INTEGER NOT NULL,
  start_offset         INTEGER NOT NULL,
  end_offset           INTEGER NOT NULL,
  normalized_text_hash TEXT NOT NULL,
  occurrence_index     INTEGER NOT NULL,
  block_fingerprint    TEXT NOT NULL,
  raw_line             TEXT NOT NULL,
  has_child_content    BOOLEAN NOT NULL DEFAULT 0,
  projection_status    TEXT NOT NULL DEFAULT 'live' CHECK(projection_status IN ('live','unlinked','ambiguous','deleted')),
  updated_at           DATETIME NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_task_note_projections_note
  ON task_note_projections(note_id, note_version);
CREATE INDEX IF NOT EXISTS idx_task_note_projections_status
  ON task_note_projections(projection_status, updated_at);

CREATE TABLE IF NOT EXISTS task_events (
  id             TEXT PRIMARY KEY,
  task_id        TEXT REFERENCES note_tasks(id) ON DELETE SET NULL,
  note_id        TEXT REFERENCES notes(id) ON DELETE SET NULL,
  event_type     TEXT NOT NULL,
  actor_type     TEXT NOT NULL,
  actor_id       TEXT,
  tool_name      TEXT,
  policy_mode    TEXT,
  approval_id    TEXT,
  old_value_json TEXT,
  new_value_json TEXT,
  created_at     DATETIME NOT NULL,
  client_id      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_task_events_task_created
  ON task_events(task_id, created_at);
CREATE INDEX IF NOT EXISTS idx_task_events_note_created
  ON task_events(note_id, created_at);
CREATE INDEX IF NOT EXISTS idx_task_events_client_created
  ON task_events(client_id, created_at);

CREATE TABLE IF NOT EXISTS task_event_read_state (
  event_id     TEXT NOT NULL REFERENCES task_events(id) ON DELETE CASCADE,
  user_id      TEXT NOT NULL,
  read_at      DATETIME,
  dismissed_at DATETIME,
  PRIMARY KEY(event_id, user_id)
);

CREATE INDEX IF NOT EXISTS idx_task_event_read_state_user
  ON task_event_read_state(user_id, read_at, dismissed_at);
CREATE INDEX IF NOT EXISTS idx_task_event_read_state_dismissed
  ON task_event_read_state(user_id, dismissed_at);

CREATE TABLE IF NOT EXISTS note_task_reconciliation_state (
  note_id       TEXT PRIMARY KEY REFERENCES notes(id) ON DELETE CASCADE,
  note_version  INTEGER NOT NULL,
  status        TEXT NOT NULL,
  reconciled_at DATETIME NOT NULL,
  item_count    INTEGER NOT NULL DEFAULT 0,
  warning_count INTEGER NOT NULL DEFAULT 0,
  cursor        TEXT
);

CREATE INDEX IF NOT EXISTS idx_note_task_reconciliation_status
  ON note_task_reconciliation_state(status, reconciled_at);
CREATE INDEX IF NOT EXISTS idx_note_task_reconciliation_version
  ON note_task_reconciliation_state(note_id, note_version);

UPDATE db_schema_version
   SET version = 48
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 48;
"""

_MIGRATION_SQL_V48_TO_V49 = """
/*───────────────────────────────────────────────────────────────
  Migration to Version 49 — Workspace Assistant Defaults storage (2026-06-07)
───────────────────────────────────────────────────────────────*/
UPDATE db_schema_version
   SET version = 49
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 49;
"""

_MIGRATION_SQL_V49_TO_V50 = """
/*───────────────────────────────────────────────────────────────
  Migration to Version 50 — Workspace activity event storage (2026-06-18)
───────────────────────────────────────────────────────────────*/
CREATE TABLE IF NOT EXISTS workspace_activity_events (
  workspace_id   TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
  event_id       TEXT NOT NULL,
  event_type     TEXT NOT NULL,
  category       TEXT NOT NULL,
  actor_user_id  TEXT,
  resource_type  TEXT,
  resource_id    TEXT,
  summary        TEXT,
  metadata_json  TEXT NOT NULL DEFAULT '{}',
  created_at     DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  client_id      TEXT NOT NULL DEFAULT 'unknown',
  version        INTEGER NOT NULL DEFAULT 1,
  PRIMARY KEY (workspace_id, event_id)
);

CREATE INDEX IF NOT EXISTS idx_ws_activity_events_created
  ON workspace_activity_events(workspace_id, created_at);
CREATE INDEX IF NOT EXISTS idx_ws_activity_events_category
  ON workspace_activity_events(workspace_id, category, created_at);

UPDATE db_schema_version
   SET version = 50
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 50;
"""

_MIGRATION_SQL_V50_TO_V51 = """
/*───────────────────────────────────────────────────────────────
  Migration to Version 51 — Manuscript annotations (2026-06-23)
───────────────────────────────────────────────────────────────*/
CREATE TABLE IF NOT EXISTS manuscript_annotations (
  id                   TEXT PRIMARY KEY,
  project_id           TEXT NOT NULL REFERENCES manuscript_projects(id) ON DELETE CASCADE,
  target_type          TEXT NOT NULL CHECK(target_type IN ('scene','chapter','project')),
  target_id            TEXT NOT NULL,
  status               TEXT NOT NULL DEFAULT 'open' CHECK(status IN ('open','resolved')),
  category             TEXT NOT NULL CHECK(category IN ('style','clarity','pacing','continuity','character','worldbuilding','structure','research','other')),
  tags_json            TEXT NOT NULL DEFAULT '[]',
  source               TEXT NOT NULL CHECK(source IN ('user','ai_selected_text','ai_scene_review')),
  body                 TEXT NOT NULL,
  suggested_fix        TEXT,
  followup_note        TEXT,
  metadata_json        TEXT NOT NULL DEFAULT '{}',
  scene_version        INTEGER,
  anchor_start         INTEGER,
  anchor_end           INTEGER,
  selected_text        TEXT,
  document_fingerprint TEXT,
  anchor_prefix        TEXT,
  anchor_suffix        TEXT,
  anchor_status        TEXT NOT NULL DEFAULT 'scene_level' CHECK(anchor_status IN ('scene_level','attached','reattached','needs_review')),
  created_at           DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified        DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted              BOOLEAN NOT NULL DEFAULT 0,
  client_id            TEXT NOT NULL,
  version              INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_mann_project_target
  ON manuscript_annotations(project_id, target_type, target_id, status, deleted);
CREATE INDEX IF NOT EXISTS idx_mann_project_status
  ON manuscript_annotations(project_id, status, last_modified);
CREATE INDEX IF NOT EXISTS idx_mann_source
  ON manuscript_annotations(project_id, source, deleted);
CREATE INDEX IF NOT EXISTS idx_mann_deleted
  ON manuscript_annotations(deleted);

DROP TRIGGER IF EXISTS manuscript_annotations_sync_create;
DROP TRIGGER IF EXISTS manuscript_annotations_sync_update;
DROP TRIGGER IF EXISTS manuscript_annotations_sync_delete;
DROP TRIGGER IF EXISTS manuscript_annotations_sync_undelete;

CREATE TRIGGER manuscript_annotations_sync_create
AFTER INSERT ON manuscript_annotations BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('manuscript_annotations', NEW.id, 'create', NEW.last_modified, NEW.client_id, NEW.version,
         json_object('id',NEW.id,'project_id',NEW.project_id,'target_type',NEW.target_type,
                     'target_id',NEW.target_id,'status',NEW.status,'category',NEW.category,
                     'tags_json',NEW.tags_json,'source',NEW.source,'body',NEW.body,
                     'suggested_fix',NEW.suggested_fix,'followup_note',NEW.followup_note,
                     'metadata_json',NEW.metadata_json,'scene_version',NEW.scene_version,
                     'anchor_start',NEW.anchor_start,'anchor_end',NEW.anchor_end,
                     'selected_text',NEW.selected_text,'document_fingerprint',NEW.document_fingerprint,
                     'anchor_prefix',NEW.anchor_prefix,'anchor_suffix',NEW.anchor_suffix,
                     'anchor_status',NEW.anchor_status,'created_at',NEW.created_at,
                     'last_modified',NEW.last_modified,'deleted',NEW.deleted,
                     'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER manuscript_annotations_sync_update
AFTER UPDATE ON manuscript_annotations
WHEN OLD.deleted = NEW.deleted AND (
     OLD.status IS NOT NEW.status OR OLD.category IS NOT NEW.category OR
     OLD.tags_json IS NOT NEW.tags_json OR OLD.source IS NOT NEW.source OR
     OLD.body IS NOT NEW.body OR OLD.suggested_fix IS NOT NEW.suggested_fix OR
     OLD.followup_note IS NOT NEW.followup_note OR
     OLD.metadata_json IS NOT NEW.metadata_json OR
     OLD.scene_version IS NOT NEW.scene_version OR OLD.anchor_start IS NOT NEW.anchor_start OR
     OLD.anchor_end IS NOT NEW.anchor_end OR OLD.selected_text IS NOT NEW.selected_text OR
     OLD.document_fingerprint IS NOT NEW.document_fingerprint OR
     OLD.anchor_prefix IS NOT NEW.anchor_prefix OR OLD.anchor_suffix IS NOT NEW.anchor_suffix OR
     OLD.anchor_status IS NOT NEW.anchor_status OR
     OLD.last_modified IS NOT NEW.last_modified OR OLD.version IS NOT NEW.version)
BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('manuscript_annotations', NEW.id, 'update', NEW.last_modified, NEW.client_id, NEW.version,
         json_object('id',NEW.id,'project_id',NEW.project_id,'target_type',NEW.target_type,
                     'target_id',NEW.target_id,'status',NEW.status,'category',NEW.category,
                     'tags_json',NEW.tags_json,'source',NEW.source,'body',NEW.body,
                     'suggested_fix',NEW.suggested_fix,'followup_note',NEW.followup_note,
                     'metadata_json',NEW.metadata_json,'scene_version',NEW.scene_version,
                     'anchor_start',NEW.anchor_start,'anchor_end',NEW.anchor_end,
                     'selected_text',NEW.selected_text,'document_fingerprint',NEW.document_fingerprint,
                     'anchor_prefix',NEW.anchor_prefix,'anchor_suffix',NEW.anchor_suffix,
                     'anchor_status',NEW.anchor_status,'created_at',NEW.created_at,
                     'last_modified',NEW.last_modified,'deleted',NEW.deleted,
                     'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER manuscript_annotations_sync_delete
AFTER UPDATE ON manuscript_annotations
WHEN OLD.deleted = 0 AND NEW.deleted = 1
BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('manuscript_annotations', NEW.id, 'delete', NEW.last_modified, NEW.client_id, NEW.version,
         json_object('id',NEW.id,'project_id',NEW.project_id,'target_type',NEW.target_type,
                     'target_id',NEW.target_id,'scene_version',NEW.scene_version,
                     'anchor_start',NEW.anchor_start,'anchor_end',NEW.anchor_end,
                     'selected_text',NEW.selected_text,'document_fingerprint',NEW.document_fingerprint,
                     'anchor_prefix',NEW.anchor_prefix,'anchor_suffix',NEW.anchor_suffix,
                     'anchor_status',NEW.anchor_status,'deleted',NEW.deleted,
                     'last_modified',NEW.last_modified,'version',NEW.version,
                     'client_id',NEW.client_id));
END;

CREATE TRIGGER manuscript_annotations_sync_undelete
AFTER UPDATE ON manuscript_annotations
WHEN OLD.deleted = 1 AND NEW.deleted = 0
BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('manuscript_annotations', NEW.id, 'update', NEW.last_modified, NEW.client_id, NEW.version,
         json_object('id',NEW.id,'project_id',NEW.project_id,'target_type',NEW.target_type,
                     'target_id',NEW.target_id,'status',NEW.status,'category',NEW.category,
                     'tags_json',NEW.tags_json,'source',NEW.source,'body',NEW.body,
                     'suggested_fix',NEW.suggested_fix,'followup_note',NEW.followup_note,
                     'metadata_json',NEW.metadata_json,'scene_version',NEW.scene_version,
                     'anchor_start',NEW.anchor_start,'anchor_end',NEW.anchor_end,
                     'selected_text',NEW.selected_text,'document_fingerprint',NEW.document_fingerprint,
                     'anchor_prefix',NEW.anchor_prefix,'anchor_suffix',NEW.anchor_suffix,
                     'anchor_status',NEW.anchor_status,'created_at',NEW.created_at,
                     'last_modified',NEW.last_modified,'deleted',NEW.deleted,
                     'client_id',NEW.client_id,'version',NEW.version));
END;

UPDATE db_schema_version
   SET version = 51
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 51;
"""

_CHAT_MACROS_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS chat_macro_registry (
  id                TEXT PRIMARY KEY,
  user_id           TEXT NOT NULL,
  name              TEXT NOT NULL,
  command           TEXT NOT NULL,
  description       TEXT,
  enabled           BOOLEAN NOT NULL DEFAULT 1,
  source            TEXT NOT NULL,
  builtin_version   INTEGER,
  schema_version    INTEGER NOT NULL,
  digest            TEXT NOT NULL,
  validation_status TEXT NOT NULL,
  validation_error  TEXT,
  updated_at        DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted_at        DATETIME
);

CREATE TABLE IF NOT EXISTS chat_macro_settings (
  user_id       TEXT PRIMARY KEY,
  settings_json TEXT NOT NULL DEFAULT '{}',
  updated_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chat_macro_runs (
  run_id               TEXT PRIMARY KEY,
  user_id              TEXT NOT NULL,
  macro_name           TEXT NOT NULL,
  macro_command        TEXT NOT NULL,
  macro_source         TEXT,
  macro_version        INTEGER,
  macro_digest         TEXT,
  status               TEXT NOT NULL DEFAULT 'pending',
  surface              TEXT,
  conversation_id      TEXT,
  workspace_id         TEXT,
  acp_session_id       TEXT,
  normalized_args      TEXT NOT NULL DEFAULT '{}',
  output_profile       TEXT,
  context_snapshot     TEXT,
  model_selection      TEXT,
  status_message_id    TEXT,
  final_message_id     TEXT,
  final_output         TEXT,
  final_output_format  TEXT,
  final_post_status    TEXT,
  post_idempotency_key TEXT,
  cancel_requested_at  DATETIME,
  error_code           TEXT,
  error_message        TEXT,
  created_at           DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  started_at           DATETIME,
  completed_at         DATETIME
);

CREATE TABLE IF NOT EXISTS chat_macro_run_branches (
  branch_id            TEXT PRIMARY KEY,
  run_id               TEXT NOT NULL REFERENCES chat_macro_runs(run_id) ON DELETE CASCADE,
  step_id              TEXT NOT NULL,
  label                TEXT,
  status               TEXT NOT NULL DEFAULT 'pending',
  attempt_count        INTEGER NOT NULL DEFAULT 0,
  prompt_digest        TEXT,
  output_text          TEXT,
  citations            TEXT NOT NULL DEFAULT '[]',
  usage                TEXT NOT NULL DEFAULT '{}',
  acp_child_session_id TEXT,
  retained             BOOLEAN NOT NULL DEFAULT 0,
  error_code           TEXT,
  error_message        TEXT,
  created_at           DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  completed_at         DATETIME
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_chat_macro_registry_user_command
  ON chat_macro_registry(user_id, command);
CREATE INDEX IF NOT EXISTS idx_chat_macro_runs_user_status_created
  ON chat_macro_runs(user_id, status, created_at);
DROP INDEX IF EXISTS idx_chat_macro_runs_post_idempotency_key_unique;
CREATE UNIQUE INDEX IF NOT EXISTS idx_chat_macro_runs_run_post_idempotency_key_unique
  ON chat_macro_runs(run_id, post_idempotency_key)
  WHERE post_idempotency_key IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS idx_chat_macro_run_branches_run_step
  ON chat_macro_run_branches(run_id, step_id);
"""

_MIGRATION_SQL_V51_TO_V52 = """
/*───────────────────────────────────────────────────────────────
  Migration to Version 52 - Source review plan storage (2026-07-09)
───────────────────────────────────────────────────────────────*/
CREATE TABLE IF NOT EXISTS source_review_plans(
  id                 INTEGER PRIMARY KEY AUTOINCREMENT,
  title              TEXT NOT NULL,
  starts_on          TEXT NOT NULL,
  timezone           TEXT NOT NULL,
  source_bundle_json TEXT NOT NULL,
  created_at         DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted            BOOLEAN NOT NULL DEFAULT 0,
  client_id          TEXT NOT NULL DEFAULT 'unknown',
  version            INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS source_review_occurrences(
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  plan_id           INTEGER NOT NULL REFERENCES source_review_plans(id) ON DELETE CASCADE,
  offset_value      INTEGER NOT NULL,
  offset_unit       TEXT NOT NULL CHECK(offset_unit IN ('day','month')),
  activity_type     TEXT NOT NULL CHECK(activity_type IN ('reread','quiz','flashcards','cloze')),
  due_at            DATETIME NOT NULL,
  status            TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','in_progress','completed','skipped')),
  launch_state_json TEXT,
  started_at        DATETIME,
  completed_at      DATETIME,
  completion_source TEXT,
  created_at        DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified     DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted           BOOLEAN NOT NULL DEFAULT 0,
  client_id         TEXT NOT NULL DEFAULT 'unknown',
  version           INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_source_review_plans_deleted
  ON source_review_plans(deleted);
CREATE INDEX IF NOT EXISTS idx_source_review_plans_list
  ON source_review_plans(deleted, created_at DESC, id DESC);
CREATE INDEX IF NOT EXISTS idx_source_review_occurrences_plan_id
  ON source_review_occurrences(plan_id, id);
CREATE INDEX IF NOT EXISTS idx_source_review_occurrences_due_status
  ON source_review_occurrences(status, due_at, id);
CREATE INDEX IF NOT EXISTS idx_source_review_occurrences_deleted
  ON source_review_occurrences(deleted);
DROP INDEX IF EXISTS idx_source_review_occurrences_due_list;
CREATE INDEX IF NOT EXISTS idx_source_review_occurrences_due_list
  ON source_review_occurrences(deleted, due_at, id, status);

DROP TRIGGER IF EXISTS source_review_plans_sync_create;
DROP TRIGGER IF EXISTS source_review_plans_sync_update;
DROP TRIGGER IF EXISTS source_review_plans_sync_delete;

CREATE TRIGGER source_review_plans_sync_create
AFTER INSERT ON source_review_plans BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('source_review_plans',NEW.id,'create',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'title',NEW.title,'starts_on',NEW.starts_on,
                     'timezone',NEW.timezone,'source_bundle_json',NEW.source_bundle_json,
                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER source_review_plans_sync_update
AFTER UPDATE ON source_review_plans
WHEN OLD.deleted = NEW.deleted AND (
     OLD.title IS NOT NEW.title OR OLD.starts_on IS NOT NEW.starts_on OR
     OLD.timezone IS NOT NEW.timezone OR OLD.source_bundle_json IS NOT NEW.source_bundle_json OR
     OLD.last_modified IS NOT NEW.last_modified OR OLD.version IS NOT NEW.version)
BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('source_review_plans',NEW.id,'update',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'title',NEW.title,'starts_on',NEW.starts_on,
                     'timezone',NEW.timezone,'source_bundle_json',NEW.source_bundle_json,
                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER source_review_plans_sync_delete
AFTER UPDATE ON source_review_plans
WHEN OLD.deleted = 0 AND NEW.deleted = 1
BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('source_review_plans',NEW.id,'delete',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'deleted',NEW.deleted,'last_modified',NEW.last_modified,
                     'client_id',NEW.client_id,'version',NEW.version));
END;

DROP TRIGGER IF EXISTS source_review_occurrences_sync_create;
DROP TRIGGER IF EXISTS source_review_occurrences_sync_update;
DROP TRIGGER IF EXISTS source_review_occurrences_sync_delete;

CREATE TRIGGER source_review_occurrences_sync_create
AFTER INSERT ON source_review_occurrences BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('source_review_occurrences',NEW.id,'create',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'plan_id',NEW.plan_id,'offset_value',NEW.offset_value,
                     'offset_unit',NEW.offset_unit,'activity_type',NEW.activity_type,
                     'due_at',NEW.due_at,'status',NEW.status,'launch_state_json',NEW.launch_state_json,
                     'started_at',NEW.started_at,'completed_at',NEW.completed_at,
                     'completion_source',NEW.completion_source,'created_at',NEW.created_at,
                     'last_modified',NEW.last_modified,'deleted',NEW.deleted,
                     'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER source_review_occurrences_sync_update
AFTER UPDATE ON source_review_occurrences
WHEN OLD.deleted = NEW.deleted AND (
     OLD.status IS NOT NEW.status OR OLD.launch_state_json IS NOT NEW.launch_state_json OR
     OLD.started_at IS NOT NEW.started_at OR OLD.completed_at IS NOT NEW.completed_at OR
     OLD.completion_source IS NOT NEW.completion_source OR
     OLD.last_modified IS NOT NEW.last_modified OR OLD.version IS NOT NEW.version)
BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('source_review_occurrences',NEW.id,'update',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'plan_id',NEW.plan_id,'offset_value',NEW.offset_value,
                     'offset_unit',NEW.offset_unit,'activity_type',NEW.activity_type,
                     'due_at',NEW.due_at,'status',NEW.status,'launch_state_json',NEW.launch_state_json,
                     'started_at',NEW.started_at,'completed_at',NEW.completed_at,
                     'completion_source',NEW.completion_source,'created_at',NEW.created_at,
                     'last_modified',NEW.last_modified,'deleted',NEW.deleted,
                     'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER source_review_occurrences_sync_delete
AFTER UPDATE ON source_review_occurrences
WHEN OLD.deleted = 0 AND NEW.deleted = 1
BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('source_review_occurrences',NEW.id,'delete',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'plan_id',NEW.plan_id,'deleted',NEW.deleted,
                     'last_modified',NEW.last_modified,'client_id',NEW.client_id,'version',NEW.version));
END;

UPDATE db_schema_version
   SET version = 52
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 52;
"""

_MIGRATION_SQL_V52_TO_V53 = """
UPDATE db_schema_version
   SET version = 53
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 53;
"""

_MIGRATION_SQL_V47_TO_V48_POSTGRES = """
/*───────────────────────────────────────────────────────────────
  Migration to Version 48 — Notes task-backed checklist storage (2026-06-05) [Postgres]
───────────────────────────────────────────────────────────────*/

CREATE TABLE IF NOT EXISTS note_tasks (
  id                TEXT PRIMARY KEY,
  note_id           TEXT NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
  text              TEXT NOT NULL,
  status            TEXT NOT NULL CHECK(status IN ('open','done')),
  metadata_json     TEXT NOT NULL DEFAULT '{}',
  projection_status TEXT NOT NULL DEFAULT 'live' CHECK(projection_status IN ('live','unlinked','ambiguous','deleted')),
  deleted           BOOLEAN NOT NULL DEFAULT FALSE,
  created_at        TIMESTAMP NOT NULL,
  updated_at        TIMESTAMP NOT NULL,
  completed_at      TIMESTAMP,
  client_id         TEXT NOT NULL,
  version           INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_note_tasks_client_scope
  ON note_tasks(client_id, deleted, updated_at);
CREATE INDEX IF NOT EXISTS idx_note_tasks_note
  ON note_tasks(note_id, deleted);
CREATE INDEX IF NOT EXISTS idx_note_tasks_status
  ON note_tasks(status, deleted);
CREATE INDEX IF NOT EXISTS idx_note_tasks_projection_status
  ON note_tasks(projection_status, deleted);

CREATE TABLE IF NOT EXISTS task_note_projections (
  task_id              TEXT PRIMARY KEY REFERENCES note_tasks(id) ON DELETE CASCADE,
  note_id              TEXT NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
  note_version         INTEGER NOT NULL,
  line_number          INTEGER NOT NULL,
  start_offset         INTEGER NOT NULL,
  end_offset           INTEGER NOT NULL,
  normalized_text_hash TEXT NOT NULL,
  occurrence_index     INTEGER NOT NULL,
  block_fingerprint    TEXT NOT NULL,
  raw_line             TEXT NOT NULL,
  has_child_content    BOOLEAN NOT NULL DEFAULT FALSE,
  projection_status    TEXT NOT NULL DEFAULT 'live' CHECK(projection_status IN ('live','unlinked','ambiguous','deleted')),
  updated_at           TIMESTAMP NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_task_note_projections_note
  ON task_note_projections(note_id, note_version);
CREATE INDEX IF NOT EXISTS idx_task_note_projections_status
  ON task_note_projections(projection_status, updated_at);

CREATE TABLE IF NOT EXISTS task_events (
  id             TEXT PRIMARY KEY,
  task_id        TEXT REFERENCES note_tasks(id) ON DELETE SET NULL,
  note_id        TEXT REFERENCES notes(id) ON DELETE SET NULL,
  event_type     TEXT NOT NULL,
  actor_type     TEXT NOT NULL,
  actor_id       TEXT,
  tool_name      TEXT,
  policy_mode    TEXT,
  approval_id    TEXT,
  old_value_json TEXT,
  new_value_json TEXT,
  created_at     TIMESTAMP NOT NULL,
  client_id      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_task_events_task_created
  ON task_events(task_id, created_at);
CREATE INDEX IF NOT EXISTS idx_task_events_note_created
  ON task_events(note_id, created_at);
CREATE INDEX IF NOT EXISTS idx_task_events_client_created
  ON task_events(client_id, created_at);

CREATE TABLE IF NOT EXISTS task_event_read_state (
  event_id     TEXT NOT NULL REFERENCES task_events(id) ON DELETE CASCADE,
  user_id      TEXT NOT NULL,
  read_at      TIMESTAMP,
  dismissed_at TIMESTAMP,
  PRIMARY KEY(event_id, user_id)
);

CREATE INDEX IF NOT EXISTS idx_task_event_read_state_user
  ON task_event_read_state(user_id, read_at, dismissed_at);
CREATE INDEX IF NOT EXISTS idx_task_event_read_state_dismissed
  ON task_event_read_state(user_id, dismissed_at);

CREATE TABLE IF NOT EXISTS note_task_reconciliation_state (
  note_id       TEXT PRIMARY KEY REFERENCES notes(id) ON DELETE CASCADE,
  note_version  INTEGER NOT NULL,
  status        TEXT NOT NULL,
  reconciled_at TIMESTAMP NOT NULL,
  item_count    INTEGER NOT NULL DEFAULT 0,
  warning_count INTEGER NOT NULL DEFAULT 0,
  cursor        TEXT
);

CREATE INDEX IF NOT EXISTS idx_note_task_reconciliation_status
  ON note_task_reconciliation_state(status, reconciled_at);
CREATE INDEX IF NOT EXISTS idx_note_task_reconciliation_version
  ON note_task_reconciliation_state(note_id, note_version);

UPDATE db_schema_version
   SET version = 48
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 48;
"""

_MIGRATION_SQL_V48_TO_V49_POSTGRES = """
/*───────────────────────────────────────────────────────────────
  Migration to Version 49 — Workspace Assistant Defaults storage (2026-06-07) [Postgres]
───────────────────────────────────────────────────────────────*/
ALTER TABLE workspaces ADD COLUMN IF NOT EXISTS assistant_defaults_json TEXT;

UPDATE db_schema_version
   SET version = 49
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 49;
"""

_MIGRATION_SQL_V49_TO_V50_POSTGRES = """
/*───────────────────────────────────────────────────────────────
  Migration to Version 50 — Workspace activity event storage (2026-06-18) [Postgres]
───────────────────────────────────────────────────────────────*/
CREATE TABLE IF NOT EXISTS workspace_activity_events (
  workspace_id   TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
  event_id       TEXT NOT NULL,
  event_type     TEXT NOT NULL,
  category       TEXT NOT NULL,
  actor_user_id  TEXT,
  resource_type  TEXT,
  resource_id    TEXT,
  summary        TEXT,
  metadata_json  TEXT NOT NULL DEFAULT '{}',
  created_at     TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  client_id      TEXT NOT NULL DEFAULT 'unknown',
  version        INTEGER NOT NULL DEFAULT 1,
  PRIMARY KEY (workspace_id, event_id)
);

CREATE INDEX IF NOT EXISTS idx_ws_activity_events_created
  ON workspace_activity_events(workspace_id, created_at);
CREATE INDEX IF NOT EXISTS idx_ws_activity_events_category
  ON workspace_activity_events(workspace_id, category, created_at);

UPDATE db_schema_version
   SET version = 50
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 50;
"""

_MIGRATION_SQL_V50_TO_V51_POSTGRES = """
/*───────────────────────────────────────────────────────────────
  Migration to Version 51 — Manuscript annotations (2026-06-23) [Postgres]
───────────────────────────────────────────────────────────────*/
CREATE TABLE IF NOT EXISTS manuscript_annotations (
  id                   TEXT PRIMARY KEY,
  project_id           TEXT NOT NULL REFERENCES manuscript_projects(id) ON DELETE CASCADE,
  target_type          TEXT NOT NULL CHECK(target_type IN ('scene','chapter','project')),
  target_id            TEXT NOT NULL,
  status               TEXT NOT NULL DEFAULT 'open' CHECK(status IN ('open','resolved')),
  category             TEXT NOT NULL CHECK(category IN ('style','clarity','pacing','continuity','character','worldbuilding','structure','research','other')),
  tags_json            TEXT NOT NULL DEFAULT '[]',
  source               TEXT NOT NULL CHECK(source IN ('user','ai_selected_text','ai_scene_review')),
  body                 TEXT NOT NULL,
  suggested_fix        TEXT,
  followup_note        TEXT,
  metadata_json        TEXT NOT NULL DEFAULT '{}',
  scene_version        INTEGER,
  anchor_start         INTEGER,
  anchor_end           INTEGER,
  selected_text        TEXT,
  document_fingerprint TEXT,
  anchor_prefix        TEXT,
  anchor_suffix        TEXT,
  anchor_status        TEXT NOT NULL DEFAULT 'scene_level' CHECK(anchor_status IN ('scene_level','attached','reattached','needs_review')),
  created_at           TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted              BOOLEAN NOT NULL DEFAULT FALSE,
  client_id            TEXT NOT NULL,
  version              INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_mann_project_target
  ON manuscript_annotations(project_id, target_type, target_id, status, deleted);
CREATE INDEX IF NOT EXISTS idx_mann_project_status
  ON manuscript_annotations(project_id, status, last_modified);
CREATE INDEX IF NOT EXISTS idx_mann_source
  ON manuscript_annotations(project_id, source, deleted);
CREATE INDEX IF NOT EXISTS idx_mann_deleted
  ON manuscript_annotations(deleted);

CREATE OR REPLACE FUNCTION manuscript_annotations_sync_log_fn()
RETURNS trigger AS $$
BEGIN
  IF TG_OP = 'INSERT' THEN
    INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
    VALUES(
      'manuscript_annotations',
      NEW.id,
      'create',
      NEW.last_modified,
      NEW.client_id,
      NEW.version,
      json_build_object(
        'id', NEW.id,
        'project_id', NEW.project_id,
        'target_type', NEW.target_type,
        'target_id', NEW.target_id,
        'status', NEW.status,
        'category', NEW.category,
        'tags_json', NEW.tags_json,
        'source', NEW.source,
        'body', NEW.body,
        'suggested_fix', NEW.suggested_fix,
        'followup_note', NEW.followup_note,
        'metadata_json', NEW.metadata_json,
        'scene_version', NEW.scene_version,
        'anchor_start', NEW.anchor_start,
        'anchor_end', NEW.anchor_end,
        'selected_text', NEW.selected_text,
        'document_fingerprint', NEW.document_fingerprint,
        'anchor_prefix', NEW.anchor_prefix,
        'anchor_suffix', NEW.anchor_suffix,
        'anchor_status', NEW.anchor_status,
        'created_at', NEW.created_at,
        'last_modified', NEW.last_modified,
        'deleted', NEW.deleted,
        'client_id', NEW.client_id,
        'version', NEW.version
      )::text
    );
  ELSIF OLD.deleted = FALSE AND NEW.deleted = TRUE THEN
    INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
    VALUES(
      'manuscript_annotations',
      NEW.id,
      'delete',
      NEW.last_modified,
      NEW.client_id,
      NEW.version,
      json_build_object(
        'id', NEW.id,
        'project_id', NEW.project_id,
        'target_type', NEW.target_type,
        'target_id', NEW.target_id,
        'scene_version', NEW.scene_version,
        'anchor_start', NEW.anchor_start,
        'anchor_end', NEW.anchor_end,
        'selected_text', NEW.selected_text,
        'document_fingerprint', NEW.document_fingerprint,
        'anchor_prefix', NEW.anchor_prefix,
        'anchor_suffix', NEW.anchor_suffix,
        'anchor_status', NEW.anchor_status,
        'deleted', NEW.deleted,
        'last_modified', NEW.last_modified,
        'version', NEW.version,
        'client_id', NEW.client_id
      )::text
    );
  ELSIF OLD.deleted = TRUE AND NEW.deleted = FALSE THEN
    INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
    VALUES(
      'manuscript_annotations',
      NEW.id,
      'update',
      NEW.last_modified,
      NEW.client_id,
      NEW.version,
      json_build_object(
        'id', NEW.id,
        'project_id', NEW.project_id,
        'target_type', NEW.target_type,
        'target_id', NEW.target_id,
        'status', NEW.status,
        'category', NEW.category,
        'tags_json', NEW.tags_json,
        'source', NEW.source,
        'body', NEW.body,
        'suggested_fix', NEW.suggested_fix,
        'followup_note', NEW.followup_note,
        'metadata_json', NEW.metadata_json,
        'scene_version', NEW.scene_version,
        'anchor_start', NEW.anchor_start,
        'anchor_end', NEW.anchor_end,
        'selected_text', NEW.selected_text,
        'document_fingerprint', NEW.document_fingerprint,
        'anchor_prefix', NEW.anchor_prefix,
        'anchor_suffix', NEW.anchor_suffix,
        'anchor_status', NEW.anchor_status,
        'created_at', NEW.created_at,
        'last_modified', NEW.last_modified,
        'deleted', NEW.deleted,
        'client_id', NEW.client_id,
        'version', NEW.version
      )::text
    );
  ELSIF OLD.deleted IS NOT DISTINCT FROM NEW.deleted AND (
    OLD.status IS DISTINCT FROM NEW.status OR
    OLD.category IS DISTINCT FROM NEW.category OR
    OLD.tags_json IS DISTINCT FROM NEW.tags_json OR
    OLD.source IS DISTINCT FROM NEW.source OR
    OLD.body IS DISTINCT FROM NEW.body OR
    OLD.suggested_fix IS DISTINCT FROM NEW.suggested_fix OR
    OLD.followup_note IS DISTINCT FROM NEW.followup_note OR
    OLD.metadata_json IS DISTINCT FROM NEW.metadata_json OR
    OLD.scene_version IS DISTINCT FROM NEW.scene_version OR
    OLD.anchor_start IS DISTINCT FROM NEW.anchor_start OR
    OLD.anchor_end IS DISTINCT FROM NEW.anchor_end OR
    OLD.selected_text IS DISTINCT FROM NEW.selected_text OR
    OLD.document_fingerprint IS DISTINCT FROM NEW.document_fingerprint OR
    OLD.anchor_prefix IS DISTINCT FROM NEW.anchor_prefix OR
    OLD.anchor_suffix IS DISTINCT FROM NEW.anchor_suffix OR
    OLD.anchor_status IS DISTINCT FROM NEW.anchor_status OR
    OLD.last_modified IS DISTINCT FROM NEW.last_modified OR
    OLD.version IS DISTINCT FROM NEW.version
  ) THEN
    INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
    VALUES(
      'manuscript_annotations',
      NEW.id,
      'update',
      NEW.last_modified,
      NEW.client_id,
      NEW.version,
      json_build_object(
        'id', NEW.id,
        'project_id', NEW.project_id,
        'target_type', NEW.target_type,
        'target_id', NEW.target_id,
        'status', NEW.status,
        'category', NEW.category,
        'tags_json', NEW.tags_json,
        'source', NEW.source,
        'body', NEW.body,
        'suggested_fix', NEW.suggested_fix,
        'followup_note', NEW.followup_note,
        'metadata_json', NEW.metadata_json,
        'scene_version', NEW.scene_version,
        'anchor_start', NEW.anchor_start,
        'anchor_end', NEW.anchor_end,
        'selected_text', NEW.selected_text,
        'document_fingerprint', NEW.document_fingerprint,
        'anchor_prefix', NEW.anchor_prefix,
        'anchor_suffix', NEW.anchor_suffix,
        'anchor_status', NEW.anchor_status,
        'created_at', NEW.created_at,
        'last_modified', NEW.last_modified,
        'deleted', NEW.deleted,
        'client_id', NEW.client_id,
        'version', NEW.version
      )::text
    );
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS manuscript_annotations_sync_log ON manuscript_annotations;
CREATE TRIGGER manuscript_annotations_sync_log
AFTER INSERT OR UPDATE ON manuscript_annotations
FOR EACH ROW EXECUTE FUNCTION manuscript_annotations_sync_log_fn();

UPDATE db_schema_version
   SET version = 51
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 51;
"""

_MIGRATION_SQL_V52_TO_V53_POSTGRES = """
ALTER TABLE quiz_questions ADD COLUMN IF NOT EXISTS group_id TEXT;
ALTER TABLE quiz_questions ADD COLUMN IF NOT EXISTS group_prompt TEXT;

UPDATE db_schema_version
   SET version = 53
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 53;
"""

_MIGRATION_SQL_V60_TO_V61 = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS shared_workspace_chat_threads (
  recipient_user_id TEXT NOT NULL CHECK(length(trim(recipient_user_id)) > 0),
  share_id INTEGER NOT NULL CHECK(share_id > 0),
  conversation_id TEXT NOT NULL UNIQUE REFERENCES conversations(id) ON DELETE CASCADE,
  owner_user_id TEXT NOT NULL CHECK(length(trim(owner_user_id)) > 0),
  workspace_id TEXT NOT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (recipient_user_id, share_id),
  UNIQUE (recipient_user_id, share_id, conversation_id)
);

CREATE TABLE IF NOT EXISTS shared_workspace_chat_requests (
  recipient_user_id TEXT NOT NULL CHECK(length(trim(recipient_user_id)) > 0),
  share_id INTEGER NOT NULL CHECK(share_id > 0),
  request_id TEXT NOT NULL,
  request_fingerprint TEXT NOT NULL,
  conversation_id TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('in_progress','retryable','completed','conflicted')),
  lease_epoch INTEGER NOT NULL DEFAULT 1 CHECK(lease_epoch >= 1),
  lease_token TEXT,
  lease_expires_at DATETIME,
  source_mode TEXT CHECK(source_mode IN ('all','include')),
  source_ids_json TEXT,
  source_snapshot_hash TEXT,
  provider TEXT,
  model TEXT,
  user_message_id TEXT REFERENCES messages(id) ON DELETE SET NULL,
  assistant_message_id TEXT REFERENCES messages(id) ON DELETE SET NULL,
  error_code TEXT,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  completed_at DATETIME,
  PRIMARY KEY (recipient_user_id, share_id, request_id),
  FOREIGN KEY (recipient_user_id, share_id, conversation_id)
    REFERENCES shared_workspace_chat_threads(recipient_user_id, share_id, conversation_id)
    ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_shared_workspace_chat_threads_conversation
  ON shared_workspace_chat_threads(conversation_id);
CREATE INDEX IF NOT EXISTS idx_shared_workspace_chat_requests_status_lease
  ON shared_workspace_chat_requests(status, lease_expires_at);
CREATE INDEX IF NOT EXISTS idx_shared_workspace_chat_requests_status_updated
  ON shared_workspace_chat_requests(status, updated_at);
CREATE INDEX IF NOT EXISTS idx_shared_workspace_chat_requests_share_updated
  ON shared_workspace_chat_requests(share_id, updated_at);

UPDATE db_schema_version
   SET version = 61
 WHERE schema_name = 'rag_char_chat_schema'
   AND version = 60;
"""

_MIGRATION_SQL_V61_TO_V62 = """
ALTER TABLE workspaces ADD COLUMN system_operation_id TEXT;
ALTER TABLE workspaces ADD COLUMN system_operation_kind TEXT
  CHECK(system_operation_kind IS NULL OR system_operation_kind = 'shared_workspace_clone');
ALTER TABLE workspaces ADD COLUMN system_operation_state TEXT
  CHECK(system_operation_state IS NULL OR system_operation_state IN ('staged', 'publication_pending'));
ALTER TABLE workspaces ADD COLUMN system_request_fingerprint TEXT;

CREATE INDEX IF NOT EXISTS idx_workspaces_system_operation
  ON workspaces(system_operation_kind, system_operation_state, system_operation_id);
"""

_MIGRATION_SQL_V60_TO_V61_POSTGRES = """
CREATE TABLE IF NOT EXISTS shared_workspace_chat_threads (
  recipient_user_id TEXT NOT NULL CHECK(char_length(btrim(recipient_user_id)) > 0),
  share_id BIGINT NOT NULL CHECK(share_id > 0),
  conversation_id TEXT NOT NULL UNIQUE REFERENCES conversations(id) ON DELETE CASCADE,
  owner_user_id TEXT NOT NULL CHECK(char_length(btrim(owner_user_id)) > 0),
  workspace_id TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (recipient_user_id, share_id),
  UNIQUE (recipient_user_id, share_id, conversation_id)
);

CREATE TABLE IF NOT EXISTS shared_workspace_chat_requests (
  recipient_user_id TEXT NOT NULL CHECK(char_length(btrim(recipient_user_id)) > 0),
  share_id BIGINT NOT NULL CHECK(share_id > 0),
  request_id TEXT NOT NULL,
  request_fingerprint TEXT NOT NULL,
  conversation_id TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('in_progress','retryable','completed','conflicted')),
  lease_epoch INTEGER NOT NULL DEFAULT 1 CHECK(lease_epoch >= 1),
  lease_token TEXT,
  lease_expires_at TIMESTAMPTZ,
  source_mode TEXT CHECK(source_mode IN ('all','include')),
  source_ids_json TEXT,
  source_snapshot_hash TEXT,
  provider TEXT,
  model TEXT,
  user_message_id TEXT REFERENCES messages(id) ON DELETE SET NULL,
  assistant_message_id TEXT REFERENCES messages(id) ON DELETE SET NULL,
  error_code TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
  completed_at TIMESTAMPTZ,
  PRIMARY KEY (recipient_user_id, share_id, request_id),
  FOREIGN KEY (recipient_user_id, share_id, conversation_id)
    REFERENCES shared_workspace_chat_threads(recipient_user_id, share_id, conversation_id)
    ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_shared_workspace_chat_threads_conversation
  ON shared_workspace_chat_threads(conversation_id);
CREATE INDEX IF NOT EXISTS idx_shared_workspace_chat_requests_status_lease
  ON shared_workspace_chat_requests(status, lease_expires_at);
CREATE INDEX IF NOT EXISTS idx_shared_workspace_chat_requests_status_updated
  ON shared_workspace_chat_requests(status, updated_at);
CREATE INDEX IF NOT EXISTS idx_shared_workspace_chat_requests_share_updated
  ON shared_workspace_chat_requests(share_id, updated_at);
"""

_MIGRATION_SQL_V61_TO_V62_POSTGRES = """
ALTER TABLE workspaces ADD COLUMN IF NOT EXISTS system_operation_id TEXT;
ALTER TABLE workspaces ADD COLUMN IF NOT EXISTS system_operation_kind TEXT
  CHECK (system_operation_kind IS NULL OR system_operation_kind = 'shared_workspace_clone');
ALTER TABLE workspaces ADD COLUMN IF NOT EXISTS system_operation_state TEXT
  CHECK (system_operation_state IS NULL OR system_operation_state IN ('staged', 'publication_pending'));
ALTER TABLE workspaces ADD COLUMN IF NOT EXISTS system_request_fingerprint TEXT;

CREATE INDEX IF NOT EXISTS idx_workspaces_system_operation
  ON workspaces(system_operation_kind, system_operation_state, system_operation_id);
"""

_MIGRATION_SQL_V63_TO_V64 = """
CREATE TABLE note_graph_suggestion_operation_receipts(
  id TEXT PRIMARY KEY CHECK(length(trim(id)) > 0),
  operation_kind TEXT NOT NULL CHECK(operation_kind IN ('run_admit', 'run_cancel', 'suggestion_accept', 'suggestion_reject', 'rejections_reset')),
  owner_user_id TEXT NOT NULL CHECK(length(trim(owner_user_id)) > 0),
  dataset_id TEXT NOT NULL CHECK(length(trim(dataset_id)) > 0),
  source_note_id TEXT NOT NULL,
  resource_identity TEXT NOT NULL CHECK(length(trim(resource_identity)) > 0),
  idempotency_key_digest TEXT NOT NULL CHECK(length(trim(idempotency_key_digest)) > 0),
  request_fingerprint TEXT NOT NULL CHECK(length(trim(request_fingerprint)) > 0),
  state TEXT NOT NULL CHECK(state IN ('in_progress', 'completed', 'failed')),
  http_status INTEGER CHECK(http_status BETWEEN 100 AND 599),
  replay_envelope TEXT,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  completed_at DATETIME,
  expires_at DATETIME NOT NULL,
  UNIQUE(owner_user_id, dataset_id, id),
  UNIQUE(owner_user_id, dataset_id, operation_kind, resource_identity, idempotency_key_digest),
  FOREIGN KEY(owner_user_id, source_note_id) REFERENCES notes(client_id, id)
    ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE TABLE note_graph_suggestion_runs(
  id TEXT PRIMARY KEY CHECK(length(trim(id)) > 0),
  owner_user_id TEXT NOT NULL CHECK(length(trim(owner_user_id)) > 0),
  dataset_id TEXT NOT NULL CHECK(length(trim(dataset_id)) > 0),
  source_note_id TEXT NOT NULL,
  source_fingerprint TEXT NOT NULL CHECK(length(trim(source_fingerprint)) > 0),
  admission_receipt_id TEXT,
  provider TEXT NOT NULL CHECK(length(trim(provider)) > 0),
  model TEXT NOT NULL CHECK(length(trim(model)) > 0),
  capability_revision TEXT NOT NULL CHECK(length(trim(capability_revision)) > 0),
  prompt_contract_version TEXT NOT NULL CHECK(length(trim(prompt_contract_version)) > 0),
  job_id TEXT,
  expected_completion_token TEXT,
  state TEXT NOT NULL CHECK(state IN ('admitting', 'queued', 'running', 'cancelling', 'publishing', 'succeeded', 'failed', 'cancelled', 'stale')),
  revision INTEGER NOT NULL DEFAULT 1 CHECK(revision >= 1),
  maintenance_lease_token TEXT CHECK(maintenance_lease_token IS NULL OR length(trim(maintenance_lease_token)) > 0),
  maintenance_lease_expires_at DATETIME,
  result_digest TEXT,
  suggestion_count INTEGER NOT NULL DEFAULT 0 CHECK(suggestion_count >= 0),
  related_note_count INTEGER NOT NULL DEFAULT 0 CHECK(related_note_count >= 0),
  tag_count INTEGER NOT NULL DEFAULT 0 CHECK(tag_count >= 0),
  invalid_item_count INTEGER NOT NULL DEFAULT 0 CHECK(invalid_item_count >= 0),
  error_code TEXT,
  guidance_key TEXT,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  started_at DATETIME,
  completed_at DATETIME,
  expires_at DATETIME NOT NULL,
  UNIQUE(owner_user_id, dataset_id, id),
  FOREIGN KEY(owner_user_id, source_note_id) REFERENCES notes(client_id, id)
    ON DELETE CASCADE ON UPDATE CASCADE,
  FOREIGN KEY(owner_user_id, dataset_id, admission_receipt_id)
    REFERENCES note_graph_suggestion_operation_receipts(owner_user_id, dataset_id, id)
    ON DELETE NO ACTION ON UPDATE CASCADE,
  CHECK(
    (maintenance_lease_token IS NULL AND maintenance_lease_expires_at IS NULL)
    OR (maintenance_lease_token IS NOT NULL AND maintenance_lease_expires_at IS NOT NULL)
  )
);

CREATE TABLE note_graph_suggestion_rejection_sets(
  owner_user_id TEXT NOT NULL CHECK(length(trim(owner_user_id)) > 0),
  dataset_id TEXT NOT NULL CHECK(length(trim(dataset_id)) > 0),
  source_note_id TEXT NOT NULL,
  source_fingerprint TEXT NOT NULL CHECK(length(trim(source_fingerprint)) > 0),
  revision INTEGER NOT NULL DEFAULT 1 CHECK(revision >= 1),
  rejection_count INTEGER NOT NULL DEFAULT 0 CHECK(rejection_count >= 0),
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY(owner_user_id, dataset_id, source_note_id, source_fingerprint),
  FOREIGN KEY(owner_user_id, source_note_id) REFERENCES notes(client_id, id)
    ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE TABLE note_graph_suggestions(
  id TEXT PRIMARY KEY CHECK(length(trim(id)) > 0),
  run_id TEXT NOT NULL,
  owner_user_id TEXT NOT NULL CHECK(length(trim(owner_user_id)) > 0),
  dataset_id TEXT NOT NULL CHECK(length(trim(dataset_id)) > 0),
  kind TEXT NOT NULL CHECK(kind IN ('related_note', 'tag')),
  source_note_id TEXT NOT NULL,
  source_fingerprint TEXT NOT NULL CHECK(length(trim(source_fingerprint)) > 0),
  target_note_id TEXT,
  target_fingerprint TEXT,
  normalized_tag TEXT,
  display_tag TEXT,
  keyword_sync_id TEXT,
  match_strength TEXT CHECK(match_strength IN ('strong', 'possible')),
  rationale TEXT CHECK(rationale IS NULL OR length(rationale) <= 240),
  state TEXT NOT NULL CHECK(state IN ('staged', 'pending', 'accepting', 'accepted', 'rejected', 'stale')),
  revision INTEGER NOT NULL DEFAULT 1 CHECK(revision >= 1),
  decision_reason TEXT,
  accepted_resource_identity TEXT,
  decision_at DATETIME,
  acceptance_lease_token TEXT,
  acceptance_lease_expires_at DATETIME,
  decision_receipt_id TEXT,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  expires_at DATETIME,
  UNIQUE(owner_user_id, dataset_id, id),
  FOREIGN KEY(owner_user_id, dataset_id, run_id)
    REFERENCES note_graph_suggestion_runs(owner_user_id, dataset_id, id)
    ON DELETE CASCADE ON UPDATE CASCADE,
  FOREIGN KEY(owner_user_id, source_note_id) REFERENCES notes(client_id, id)
    ON DELETE CASCADE ON UPDATE CASCADE,
  FOREIGN KEY(owner_user_id, target_note_id) REFERENCES notes(client_id, id)
    ON DELETE CASCADE ON UPDATE CASCADE,
  FOREIGN KEY(owner_user_id, dataset_id, decision_receipt_id)
    REFERENCES note_graph_suggestion_operation_receipts(owner_user_id, dataset_id, id)
    ON DELETE NO ACTION ON UPDATE CASCADE,
  CHECK(
    (kind = 'related_note' AND target_note_id IS NOT NULL AND target_fingerprint IS NOT NULL AND normalized_tag IS NULL AND display_tag IS NULL)
    OR (kind = 'tag' AND target_note_id IS NULL AND target_fingerprint IS NULL AND normalized_tag IS NOT NULL AND display_tag IS NOT NULL)
  )
);

CREATE TABLE note_graph_suggestion_evidence(
  suggestion_id TEXT NOT NULL,
  owner_user_id TEXT NOT NULL CHECK(length(trim(owner_user_id)) > 0),
  dataset_id TEXT NOT NULL CHECK(length(trim(dataset_id)) > 0),
  side TEXT NOT NULL CHECK(side IN ('source', 'target')),
  ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
  note_id TEXT NOT NULL,
  field TEXT NOT NULL CHECK(field IN ('title', 'content')),
  content_fingerprint TEXT NOT NULL CHECK(length(trim(content_fingerprint)) > 0),
  start_offset INTEGER NOT NULL CHECK(start_offset >= 0),
  end_offset INTEGER NOT NULL CHECK(end_offset > start_offset),
  PRIMARY KEY(suggestion_id, side, ordinal),
  FOREIGN KEY(owner_user_id, dataset_id, suggestion_id)
    REFERENCES note_graph_suggestions(owner_user_id, dataset_id, id)
    ON DELETE CASCADE ON UPDATE CASCADE,
  FOREIGN KEY(owner_user_id, note_id) REFERENCES notes(client_id, id)
    ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE INDEX idx_note_graph_suggestion_runs_owner_dataset_note_state
  ON note_graph_suggestion_runs(owner_user_id, dataset_id, source_note_id, state, created_at DESC);
CREATE UNIQUE INDEX idx_note_graph_suggestion_runs_active_source
  ON note_graph_suggestion_runs(
    owner_user_id, dataset_id, source_note_id, source_fingerprint,
    provider, model, prompt_contract_version
  )
  WHERE state IN ('admitting', 'queued', 'running', 'cancelling', 'publishing');
CREATE INDEX idx_note_graph_suggestion_runs_retention
  ON note_graph_suggestion_runs(state, expires_at);
CREATE INDEX idx_note_graph_suggestion_runs_maintenance
  ON note_graph_suggestion_runs(
    owner_user_id, dataset_id, state, maintenance_lease_expires_at, created_at, id
  )
  WHERE state IN ('admitting', 'queued', 'running', 'cancelling', 'publishing');
CREATE INDEX idx_note_graph_suggestion_operation_receipts_retention
  ON note_graph_suggestion_operation_receipts(owner_user_id, dataset_id, state, expires_at);
CREATE INDEX idx_note_graph_suggestions_owner_dataset_source_state
  ON note_graph_suggestions(owner_user_id, dataset_id, source_note_id, state, updated_at DESC);
CREATE INDEX idx_note_graph_suggestions_target_state
  ON note_graph_suggestions(owner_user_id, dataset_id, target_note_id, state);
CREATE INDEX idx_note_graph_suggestions_suppression_related
  ON note_graph_suggestions(owner_user_id, dataset_id, source_note_id, source_fingerprint, target_note_id, target_fingerprint);
CREATE INDEX idx_note_graph_suggestions_suppression_tag
  ON note_graph_suggestions(owner_user_id, dataset_id, source_note_id, source_fingerprint, normalized_tag);
CREATE UNIQUE INDEX idx_note_graph_suggestions_canonical_related_identity
  ON note_graph_suggestions(
    owner_user_id, dataset_id,
    CASE WHEN source_note_id < target_note_id THEN source_note_id ELSE target_note_id END,
    CASE WHEN source_note_id < target_note_id THEN target_note_id ELSE source_note_id END,
    CASE WHEN source_note_id < target_note_id THEN source_fingerprint ELSE target_fingerprint END,
    CASE WHEN source_note_id < target_note_id THEN target_fingerprint ELSE source_fingerprint END
  )
  WHERE kind = 'related_note' AND state IN ('pending', 'rejected');
CREATE UNIQUE INDEX idx_note_graph_suggestions_canonical_tag_identity
  ON note_graph_suggestions(owner_user_id, dataset_id, source_note_id, source_fingerprint, normalized_tag)
  WHERE kind = 'tag' AND state IN ('pending', 'rejected');
CREATE UNIQUE INDEX idx_note_graph_suggestions_staged_related_identity
  ON note_graph_suggestions(
    owner_user_id, dataset_id, run_id,
    CASE WHEN source_note_id < target_note_id THEN source_note_id ELSE target_note_id END,
    CASE WHEN source_note_id < target_note_id THEN target_note_id ELSE source_note_id END,
    CASE WHEN source_note_id < target_note_id THEN source_fingerprint ELSE target_fingerprint END,
    CASE WHEN source_note_id < target_note_id THEN target_fingerprint ELSE source_fingerprint END
  )
  WHERE kind = 'related_note' AND state = 'staged';
CREATE UNIQUE INDEX idx_note_graph_suggestions_staged_tag_identity
  ON note_graph_suggestions(owner_user_id, dataset_id, run_id, source_note_id, source_fingerprint, normalized_tag)
  WHERE kind = 'tag' AND state = 'staged';
CREATE INDEX idx_note_graph_suggestions_acceptance_lease
  ON note_graph_suggestions(state, acceptance_lease_expires_at);
CREATE INDEX idx_note_graph_suggestions_retention
  ON note_graph_suggestions(state, expires_at);
CREATE INDEX idx_note_graph_suggestion_evidence_note
  ON note_graph_suggestion_evidence(owner_user_id, dataset_id, note_id);
"""

_MIGRATION_SQL_V63_TO_V64_POSTGRES = """
CREATE TABLE IF NOT EXISTS note_graph_suggestion_operation_receipts(
  id TEXT PRIMARY KEY CHECK(char_length(btrim(id)) > 0),
  operation_kind TEXT NOT NULL CHECK(operation_kind IN ('run_admit', 'run_cancel', 'suggestion_accept', 'suggestion_reject', 'rejections_reset')),
  owner_user_id TEXT NOT NULL CHECK(char_length(btrim(owner_user_id)) > 0),
  dataset_id TEXT NOT NULL CHECK(char_length(btrim(dataset_id)) > 0),
  source_note_id TEXT NOT NULL,
  resource_identity TEXT NOT NULL CHECK(char_length(btrim(resource_identity)) > 0),
  idempotency_key_digest TEXT NOT NULL CHECK(char_length(btrim(idempotency_key_digest)) > 0),
  request_fingerprint TEXT NOT NULL CHECK(char_length(btrim(request_fingerprint)) > 0),
  state TEXT NOT NULL CHECK(state IN ('in_progress', 'completed', 'failed')),
  http_status INTEGER CHECK(http_status BETWEEN 100 AND 599),
  replay_envelope TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
  completed_at TIMESTAMPTZ,
  expires_at TIMESTAMPTZ NOT NULL,
  UNIQUE(owner_user_id, dataset_id, id),
  UNIQUE(owner_user_id, dataset_id, operation_kind, resource_identity, idempotency_key_digest),
  FOREIGN KEY(owner_user_id, source_note_id) REFERENCES notes(client_id, id)
    ON DELETE CASCADE ON UPDATE CASCADE
);
CREATE TABLE IF NOT EXISTS note_graph_suggestion_runs(
  id TEXT PRIMARY KEY CHECK(char_length(btrim(id)) > 0),
  owner_user_id TEXT NOT NULL CHECK(char_length(btrim(owner_user_id)) > 0),
  dataset_id TEXT NOT NULL CHECK(char_length(btrim(dataset_id)) > 0),
  source_note_id TEXT NOT NULL,
  source_fingerprint TEXT NOT NULL CHECK(char_length(btrim(source_fingerprint)) > 0),
  admission_receipt_id TEXT,
  provider TEXT NOT NULL CHECK(char_length(btrim(provider)) > 0),
  model TEXT NOT NULL CHECK(char_length(btrim(model)) > 0),
  capability_revision TEXT NOT NULL CHECK(char_length(btrim(capability_revision)) > 0),
  prompt_contract_version TEXT NOT NULL CHECK(char_length(btrim(prompt_contract_version)) > 0),
  job_id TEXT, expected_completion_token TEXT,
  state TEXT NOT NULL CHECK(state IN ('admitting', 'queued', 'running', 'cancelling', 'publishing', 'succeeded', 'failed', 'cancelled', 'stale')),
  revision INTEGER NOT NULL DEFAULT 1 CHECK(revision >= 1),
  maintenance_lease_token TEXT CHECK(maintenance_lease_token IS NULL OR char_length(btrim(maintenance_lease_token)) > 0),
  maintenance_lease_expires_at TIMESTAMPTZ,
  result_digest TEXT,
  suggestion_count INTEGER NOT NULL DEFAULT 0 CHECK(suggestion_count >= 0),
  related_note_count INTEGER NOT NULL DEFAULT 0 CHECK(related_note_count >= 0),
  tag_count INTEGER NOT NULL DEFAULT 0 CHECK(tag_count >= 0),
  invalid_item_count INTEGER NOT NULL DEFAULT 0 CHECK(invalid_item_count >= 0),
  error_code TEXT, guidance_key TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
  started_at TIMESTAMPTZ, completed_at TIMESTAMPTZ, expires_at TIMESTAMPTZ NOT NULL,
  UNIQUE(owner_user_id, dataset_id, id),
  FOREIGN KEY(owner_user_id, source_note_id) REFERENCES notes(client_id, id)
    ON DELETE CASCADE ON UPDATE CASCADE,
  FOREIGN KEY(owner_user_id, dataset_id, admission_receipt_id)
    REFERENCES note_graph_suggestion_operation_receipts(owner_user_id, dataset_id, id)
    ON DELETE NO ACTION ON UPDATE CASCADE,
  CHECK(
    (maintenance_lease_token IS NULL AND maintenance_lease_expires_at IS NULL)
    OR (maintenance_lease_token IS NOT NULL AND maintenance_lease_expires_at IS NOT NULL)
  )
);
CREATE TABLE IF NOT EXISTS note_graph_suggestion_rejection_sets(
  owner_user_id TEXT NOT NULL CHECK(char_length(btrim(owner_user_id)) > 0),
  dataset_id TEXT NOT NULL CHECK(char_length(btrim(dataset_id)) > 0),
  source_note_id TEXT NOT NULL,
  source_fingerprint TEXT NOT NULL CHECK(char_length(btrim(source_fingerprint)) > 0),
  revision INTEGER NOT NULL DEFAULT 1 CHECK(revision >= 1),
  rejection_count INTEGER NOT NULL DEFAULT 0 CHECK(rejection_count >= 0),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY(owner_user_id, dataset_id, source_note_id, source_fingerprint),
  FOREIGN KEY(owner_user_id, source_note_id) REFERENCES notes(client_id, id)
    ON DELETE CASCADE ON UPDATE CASCADE
);
CREATE TABLE IF NOT EXISTS note_graph_suggestions(
  id TEXT PRIMARY KEY CHECK(char_length(btrim(id)) > 0),
  run_id TEXT NOT NULL,
  owner_user_id TEXT NOT NULL CHECK(char_length(btrim(owner_user_id)) > 0),
  dataset_id TEXT NOT NULL CHECK(char_length(btrim(dataset_id)) > 0),
  kind TEXT NOT NULL CHECK(kind IN ('related_note', 'tag')),
  source_note_id TEXT NOT NULL,
  source_fingerprint TEXT NOT NULL CHECK(char_length(btrim(source_fingerprint)) > 0),
  target_note_id TEXT,
  target_fingerprint TEXT, normalized_tag TEXT, display_tag TEXT, keyword_sync_id TEXT,
  match_strength TEXT CHECK(match_strength IN ('strong', 'possible')),
  rationale TEXT CHECK(rationale IS NULL OR char_length(rationale) <= 240),
  state TEXT NOT NULL CHECK(state IN ('staged', 'pending', 'accepting', 'accepted', 'rejected', 'stale')),
  revision INTEGER NOT NULL DEFAULT 1 CHECK(revision >= 1),
  decision_reason TEXT, accepted_resource_identity TEXT, decision_at TIMESTAMPTZ,
  acceptance_lease_token TEXT, acceptance_lease_expires_at TIMESTAMPTZ,
  decision_receipt_id TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
  expires_at TIMESTAMPTZ,
  UNIQUE(owner_user_id, dataset_id, id),
  FOREIGN KEY(owner_user_id, dataset_id, run_id)
    REFERENCES note_graph_suggestion_runs(owner_user_id, dataset_id, id)
    ON DELETE CASCADE ON UPDATE CASCADE,
  FOREIGN KEY(owner_user_id, source_note_id) REFERENCES notes(client_id, id)
    ON DELETE CASCADE ON UPDATE CASCADE,
  FOREIGN KEY(owner_user_id, target_note_id) REFERENCES notes(client_id, id)
    ON DELETE CASCADE ON UPDATE CASCADE,
  FOREIGN KEY(owner_user_id, dataset_id, decision_receipt_id)
    REFERENCES note_graph_suggestion_operation_receipts(owner_user_id, dataset_id, id)
    ON DELETE NO ACTION ON UPDATE CASCADE,
  CHECK((kind = 'related_note' AND target_note_id IS NOT NULL AND target_fingerprint IS NOT NULL AND normalized_tag IS NULL AND display_tag IS NULL) OR (kind = 'tag' AND target_note_id IS NULL AND target_fingerprint IS NULL AND normalized_tag IS NOT NULL AND display_tag IS NOT NULL))
);
CREATE TABLE IF NOT EXISTS note_graph_suggestion_evidence(
  suggestion_id TEXT NOT NULL,
  owner_user_id TEXT NOT NULL CHECK(char_length(btrim(owner_user_id)) > 0),
  dataset_id TEXT NOT NULL CHECK(char_length(btrim(dataset_id)) > 0),
  side TEXT NOT NULL CHECK(side IN ('source', 'target')),
  ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
  note_id TEXT NOT NULL,
  field TEXT NOT NULL CHECK(field IN ('title', 'content')),
  content_fingerprint TEXT NOT NULL CHECK(char_length(btrim(content_fingerprint)) > 0),
  start_offset INTEGER NOT NULL CHECK(start_offset >= 0),
  end_offset INTEGER NOT NULL CHECK(end_offset > start_offset),
  PRIMARY KEY(suggestion_id, side, ordinal),
  FOREIGN KEY(owner_user_id, dataset_id, suggestion_id)
    REFERENCES note_graph_suggestions(owner_user_id, dataset_id, id)
    ON DELETE CASCADE ON UPDATE CASCADE,
  FOREIGN KEY(owner_user_id, note_id) REFERENCES notes(client_id, id)
    ON DELETE CASCADE ON UPDATE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_note_graph_suggestion_runs_owner_dataset_note_state ON note_graph_suggestion_runs(owner_user_id, dataset_id, source_note_id, state, created_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS idx_note_graph_suggestion_runs_active_source
  ON note_graph_suggestion_runs(
    owner_user_id, dataset_id, source_note_id, source_fingerprint,
    provider, model, prompt_contract_version
  )
  WHERE state IN ('admitting', 'queued', 'running', 'cancelling', 'publishing');
CREATE INDEX IF NOT EXISTS idx_note_graph_suggestion_runs_retention ON note_graph_suggestion_runs(state, expires_at);
CREATE INDEX IF NOT EXISTS idx_note_graph_suggestion_runs_maintenance
  ON note_graph_suggestion_runs(
    owner_user_id, dataset_id, state, maintenance_lease_expires_at, created_at, id
  )
  WHERE state IN ('admitting', 'queued', 'running', 'cancelling', 'publishing');
CREATE INDEX IF NOT EXISTS idx_note_graph_suggestion_operation_receipts_retention ON note_graph_suggestion_operation_receipts(owner_user_id, dataset_id, state, expires_at);
CREATE INDEX IF NOT EXISTS idx_note_graph_suggestions_owner_dataset_source_state ON note_graph_suggestions(owner_user_id, dataset_id, source_note_id, state, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_note_graph_suggestions_target_state ON note_graph_suggestions(owner_user_id, dataset_id, target_note_id, state);
CREATE INDEX IF NOT EXISTS idx_note_graph_suggestions_suppression_related ON note_graph_suggestions(owner_user_id, dataset_id, source_note_id, source_fingerprint, target_note_id, target_fingerprint);
CREATE INDEX IF NOT EXISTS idx_note_graph_suggestions_suppression_tag ON note_graph_suggestions(owner_user_id, dataset_id, source_note_id, source_fingerprint, normalized_tag);
CREATE UNIQUE INDEX IF NOT EXISTS idx_note_graph_suggestions_canonical_related_identity
  ON note_graph_suggestions(
    owner_user_id, dataset_id,
    (CASE WHEN source_note_id < target_note_id THEN source_note_id ELSE target_note_id END),
    (CASE WHEN source_note_id < target_note_id THEN target_note_id ELSE source_note_id END),
    (CASE WHEN source_note_id < target_note_id THEN source_fingerprint ELSE target_fingerprint END),
    (CASE WHEN source_note_id < target_note_id THEN target_fingerprint ELSE source_fingerprint END)
  )
  WHERE kind = 'related_note' AND state IN ('pending', 'rejected');
CREATE UNIQUE INDEX IF NOT EXISTS idx_note_graph_suggestions_canonical_tag_identity
  ON note_graph_suggestions(owner_user_id, dataset_id, source_note_id, source_fingerprint, normalized_tag)
  WHERE kind = 'tag' AND state IN ('pending', 'rejected');
CREATE UNIQUE INDEX IF NOT EXISTS idx_note_graph_suggestions_staged_related_identity
  ON note_graph_suggestions(
    owner_user_id, dataset_id, run_id,
    (CASE WHEN source_note_id < target_note_id THEN source_note_id ELSE target_note_id END),
    (CASE WHEN source_note_id < target_note_id THEN target_note_id ELSE source_note_id END),
    (CASE WHEN source_note_id < target_note_id THEN source_fingerprint ELSE target_fingerprint END),
    (CASE WHEN source_note_id < target_note_id THEN target_fingerprint ELSE source_fingerprint END)
  )
  WHERE kind = 'related_note' AND state = 'staged';
CREATE UNIQUE INDEX IF NOT EXISTS idx_note_graph_suggestions_staged_tag_identity
  ON note_graph_suggestions(owner_user_id, dataset_id, run_id, source_note_id, source_fingerprint, normalized_tag)
  WHERE kind = 'tag' AND state = 'staged';
CREATE INDEX IF NOT EXISTS idx_note_graph_suggestions_acceptance_lease ON note_graph_suggestions(state, acceptance_lease_expires_at);
CREATE INDEX IF NOT EXISTS idx_note_graph_suggestions_retention ON note_graph_suggestions(state, expires_at);
CREATE INDEX IF NOT EXISTS idx_note_graph_suggestion_evidence_note ON note_graph_suggestion_evidence(owner_user_id, dataset_id, note_id);

ALTER TABLE note_graph_suggestion_operation_receipts ENABLE ROW LEVEL SECURITY;
ALTER TABLE note_graph_suggestion_operation_receipts FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS note_graph_suggestion_operation_receipts_tenant_isolation ON note_graph_suggestion_operation_receipts;
CREATE POLICY note_graph_suggestion_operation_receipts_tenant_isolation ON note_graph_suggestion_operation_receipts USING (owner_user_id = current_setting('app.current_user_id', true) AND dataset_id = current_setting('app.current_dataset_id', true)) WITH CHECK (owner_user_id = current_setting('app.current_user_id', true) AND dataset_id = current_setting('app.current_dataset_id', true));
ALTER TABLE note_graph_suggestion_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE note_graph_suggestion_runs FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS note_graph_suggestion_runs_tenant_isolation ON note_graph_suggestion_runs;
CREATE POLICY note_graph_suggestion_runs_tenant_isolation ON note_graph_suggestion_runs USING (owner_user_id = current_setting('app.current_user_id', true) AND dataset_id = current_setting('app.current_dataset_id', true)) WITH CHECK (owner_user_id = current_setting('app.current_user_id', true) AND dataset_id = current_setting('app.current_dataset_id', true));
ALTER TABLE note_graph_suggestion_rejection_sets ENABLE ROW LEVEL SECURITY;
ALTER TABLE note_graph_suggestion_rejection_sets FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS note_graph_suggestion_rejection_sets_tenant_isolation ON note_graph_suggestion_rejection_sets;
CREATE POLICY note_graph_suggestion_rejection_sets_tenant_isolation ON note_graph_suggestion_rejection_sets USING (owner_user_id = current_setting('app.current_user_id', true) AND dataset_id = current_setting('app.current_dataset_id', true)) WITH CHECK (owner_user_id = current_setting('app.current_user_id', true) AND dataset_id = current_setting('app.current_dataset_id', true));
ALTER TABLE note_graph_suggestions ENABLE ROW LEVEL SECURITY;
ALTER TABLE note_graph_suggestions FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS note_graph_suggestions_tenant_isolation ON note_graph_suggestions;
CREATE POLICY note_graph_suggestions_tenant_isolation ON note_graph_suggestions USING (owner_user_id = current_setting('app.current_user_id', true) AND dataset_id = current_setting('app.current_dataset_id', true)) WITH CHECK (owner_user_id = current_setting('app.current_user_id', true) AND dataset_id = current_setting('app.current_dataset_id', true));
ALTER TABLE note_graph_suggestion_evidence ENABLE ROW LEVEL SECURITY;
ALTER TABLE note_graph_suggestion_evidence FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS note_graph_suggestion_evidence_tenant_isolation ON note_graph_suggestion_evidence;
CREATE POLICY note_graph_suggestion_evidence_tenant_isolation ON note_graph_suggestion_evidence USING (owner_user_id = current_setting('app.current_user_id', true) AND dataset_id = current_setting('app.current_dataset_id', true)) WITH CHECK (owner_user_id = current_setting('app.current_user_id', true) AND dataset_id = current_setting('app.current_dataset_id', true));
"""

_MIGRATION_SQL_V81_TO_V82 = """
CREATE TABLE note_semantic_index_configs(
  owner_user_id TEXT NOT NULL CHECK(length(trim(owner_user_id)) > 0),
  dataset_id TEXT NOT NULL CHECK(length(trim(dataset_id)) > 0),
  desired_state TEXT NOT NULL CHECK(desired_state IN ('enabled', 'disabled')),
  active_generation_id TEXT,
  configuration_revision INTEGER NOT NULL DEFAULT 1 CHECK(configuration_revision >= 1),
  semantic_index_revision INTEGER NOT NULL DEFAULT 0 CHECK(semantic_index_revision >= 0),
  capability_revision TEXT,
  disclosure_hash TEXT,
  compatibility_hash TEXT,
  provider TEXT,
  model TEXT,
  endpoint_origin_revision TEXT,
  endpoint_origin_display TEXT,
  data_boundary TEXT,
  vector_backend TEXT,
  storage_boundary TEXT,
  storage_label TEXT,
  metric TEXT NOT NULL DEFAULT 'cosine' CHECK(metric = 'cosine'),
  dimension_state TEXT NOT NULL CHECK(dimension_state IN ('pending', 'resolved')),
  dimensions INTEGER CHECK(dimensions IS NULL OR dimensions >= 1),
  normalization_version TEXT NOT NULL CHECK(length(trim(normalization_version)) > 0),
  chunker_version TEXT NOT NULL CHECK(length(trim(chunker_version)) > 0),
  enabled_at DATETIME,
  disabled_at DATETIME,
  consented_at DATETIME,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY(owner_user_id, dataset_id),
  CHECK(
    (dimension_state = 'pending' AND dimensions IS NULL AND compatibility_hash IS NULL)
    OR
    (dimension_state = 'resolved' AND dimensions IS NOT NULL AND compatibility_hash IS NOT NULL
      AND length(compatibility_hash) BETWEEN 1 AND 128
      AND substr(compatibility_hash, 1, 1) GLOB '[A-Za-z0-9]'
      AND compatibility_hash NOT GLOB '*[^A-Za-z0-9._:-]*')
  )
);

CREATE TABLE note_semantic_generations(
  id TEXT PRIMARY KEY CHECK(length(trim(id)) > 0),
  owner_user_id TEXT NOT NULL CHECK(length(trim(owner_user_id)) > 0),
  dataset_id TEXT NOT NULL CHECK(length(trim(dataset_id)) > 0),
  configuration_revision INTEGER NOT NULL CHECK(configuration_revision >= 1),
  state TEXT NOT NULL CHECK(state IN ('staging', 'active', 'retired', 'failed', 'deleting')),
  compatibility_hash TEXT,
  dimension_state TEXT NOT NULL CHECK(dimension_state IN ('pending', 'resolved')),
  dimensions INTEGER CHECK(dimensions IS NULL OR dimensions >= 1),
  root_job_id TEXT,
  expected_note_count INTEGER NOT NULL DEFAULT 0 CHECK(expected_note_count >= 0),
  expected_chunk_count INTEGER NOT NULL DEFAULT 0 CHECK(expected_chunk_count >= 0),
  published_note_count INTEGER NOT NULL DEFAULT 0 CHECK(published_note_count >= 0),
  published_chunk_count INTEGER NOT NULL DEFAULT 0 CHECK(published_chunk_count >= 0),
  manifest_hash TEXT,
  publication_receipt TEXT,
  terminal_error_code TEXT,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  published_at DATETIME,
  retired_at DATETIME,
  deleted_at DATETIME,
  UNIQUE(owner_user_id, dataset_id, id),
  FOREIGN KEY(owner_user_id, dataset_id) REFERENCES note_semantic_index_configs(owner_user_id, dataset_id) ON DELETE CASCADE ON UPDATE CASCADE,
  CHECK(
    (dimension_state = 'pending' AND dimensions IS NULL AND compatibility_hash IS NULL)
    OR
    (dimension_state = 'resolved' AND dimensions IS NOT NULL AND compatibility_hash IS NOT NULL
      AND length(compatibility_hash) BETWEEN 1 AND 128
      AND substr(compatibility_hash, 1, 1) GLOB '[A-Za-z0-9]'
      AND compatibility_hash NOT GLOB '*[^A-Za-z0-9._:-]*')
  )
);

CREATE TABLE note_semantic_note_state(
  owner_user_id TEXT NOT NULL CHECK(length(trim(owner_user_id)) > 0),
  dataset_id TEXT NOT NULL CHECK(length(trim(dataset_id)) > 0),
  generation_id TEXT NOT NULL,
  note_id TEXT NOT NULL,
  content_version INTEGER NOT NULL CHECK(content_version >= 1),
  content_fingerprint TEXT NOT NULL CHECK(
    length(content_fingerprint) = 71
    AND substr(content_fingerprint, 1, 7) = 'sha256:'
    AND substr(content_fingerprint, 8) NOT GLOB '*[^0-9a-f]*'
  ),
  dirty_generation INTEGER NOT NULL CHECK(dirty_generation >= 1),
  state TEXT NOT NULL CHECK(state IN ('pending', 'indexed', 'excluded', 'failed', 'tombstoned')),
  chunk_count INTEGER NOT NULL DEFAULT 0 CHECK(chunk_count >= 0),
  manifest_hash TEXT,
  error_code TEXT,
  published_at DATETIME,
  PRIMARY KEY(owner_user_id, dataset_id, generation_id, note_id),
  FOREIGN KEY(owner_user_id, dataset_id, generation_id) REFERENCES note_semantic_generations(owner_user_id, dataset_id, id) ON DELETE CASCADE ON UPDATE CASCADE,
  FOREIGN KEY(owner_user_id, note_id) REFERENCES notes(client_id, id) ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE TABLE note_semantic_chunks(
  chunk_id TEXT PRIMARY KEY CHECK(length(trim(chunk_id)) > 0),
  owner_user_id TEXT NOT NULL CHECK(length(trim(owner_user_id)) > 0),
  dataset_id TEXT NOT NULL CHECK(length(trim(dataset_id)) > 0),
  generation_id TEXT NOT NULL,
  note_id TEXT NOT NULL,
  content_version INTEGER NOT NULL CHECK(content_version >= 1),
  ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
  field TEXT NOT NULL CHECK(field IN ('title', 'content')),
  start_offset INTEGER NOT NULL CHECK(start_offset >= 0),
  end_offset INTEGER NOT NULL CHECK(end_offset > start_offset),
  chunk_fingerprint TEXT NOT NULL CHECK(
    length(chunk_fingerprint) = 71
    AND substr(chunk_fingerprint, 1, 7) = 'sha256:'
    AND substr(chunk_fingerprint, 8) NOT GLOB '*[^0-9a-f]*'
  ),
  normalization_version TEXT NOT NULL CHECK(length(trim(normalization_version)) > 0),
  chunker_version TEXT NOT NULL CHECK(length(trim(chunker_version)) > 0),
  UNIQUE(owner_user_id, dataset_id, chunk_id),
  UNIQUE(owner_user_id, dataset_id, generation_id, note_id, ordinal),
  FOREIGN KEY(owner_user_id, dataset_id, generation_id, note_id) REFERENCES note_semantic_note_state(owner_user_id, dataset_id, generation_id, note_id) ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE TABLE note_semantic_work(
  id TEXT PRIMARY KEY CHECK(length(trim(id)) > 0),
  owner_user_id TEXT NOT NULL CHECK(length(trim(owner_user_id)) > 0),
  dataset_id TEXT NOT NULL CHECK(length(trim(dataset_id)) > 0),
  kind TEXT NOT NULL CHECK(kind IN ('index_note', 'delete_note_vectors', 'delete_generation')),
  note_id TEXT,
  generation_id TEXT,
  dirty_generation INTEGER CHECK(dirty_generation IS NULL OR dirty_generation >= 1),
  fencing_token TEXT NOT NULL CHECK(length(trim(fencing_token)) > 0),
  claim_state TEXT NOT NULL CHECK(claim_state IN ('pending', 'claimed', 'completed', 'failed')),
  attempt_count INTEGER NOT NULL DEFAULT 0 CHECK(attempt_count >= 0 AND attempt_count <= 5),
  next_eligible_at DATETIME NOT NULL,
  claim_token TEXT,
  claimed_at DATETIME,
  error_code TEXT,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(owner_user_id, dataset_id, id),
  FOREIGN KEY(owner_user_id, dataset_id, generation_id) REFERENCES note_semantic_generations(owner_user_id, dataset_id, id) ON DELETE CASCADE ON UPDATE CASCADE,
  CHECK(generation_id IS NOT NULL AND (
    (kind = 'delete_generation' AND note_id IS NULL AND dirty_generation IS NULL)
    OR
    (kind IN ('index_note', 'delete_note_vectors') AND note_id IS NOT NULL AND dirty_generation IS NOT NULL)
  )),
  CHECK((claim_state = 'claimed' AND claim_token IS NOT NULL AND claimed_at IS NOT NULL) OR (claim_state != 'claimed'))
);

CREATE UNIQUE INDEX idx_note_semantic_generations_one_active ON note_semantic_generations(owner_user_id, dataset_id) WHERE state = 'active';
CREATE UNIQUE INDEX idx_note_semantic_generations_one_staging ON note_semantic_generations(owner_user_id, dataset_id) WHERE state = 'staging';
CREATE INDEX idx_note_semantic_note_state_pending ON note_semantic_note_state(owner_user_id, dataset_id, generation_id, state, dirty_generation);
CREATE INDEX idx_note_semantic_chunks_note ON note_semantic_chunks(owner_user_id, dataset_id, generation_id, note_id, ordinal);
CREATE INDEX idx_note_semantic_work_claimable ON note_semantic_work(owner_user_id, dataset_id, claim_state, next_eligible_at, id);
CREATE UNIQUE INDEX idx_note_semantic_work_note_coalesce ON note_semantic_work(owner_user_id, dataset_id, kind, note_id) WHERE note_id IS NOT NULL;
CREATE UNIQUE INDEX idx_note_semantic_work_generation_coalesce ON note_semantic_work(owner_user_id, dataset_id, kind, generation_id) WHERE generation_id IS NOT NULL AND kind = 'delete_generation';
"""

_MIGRATION_SQL_V85_TO_V86_POSTGRES = """
CREATE TABLE IF NOT EXISTS note_semantic_index_configs(
  owner_user_id TEXT NOT NULL CHECK(char_length(btrim(owner_user_id)) > 0),
  dataset_id TEXT NOT NULL CHECK(char_length(btrim(dataset_id)) > 0),
  desired_state TEXT NOT NULL CHECK(desired_state IN ('enabled', 'disabled')),
  active_generation_id TEXT,
  configuration_revision INTEGER NOT NULL DEFAULT 1 CHECK(configuration_revision >= 1),
  semantic_index_revision INTEGER NOT NULL DEFAULT 0 CHECK(semantic_index_revision >= 0),
  capability_revision TEXT, disclosure_hash TEXT, compatibility_hash TEXT, provider TEXT, model TEXT,
  endpoint_origin_revision TEXT, endpoint_origin_display TEXT, data_boundary TEXT, vector_backend TEXT,
  storage_boundary TEXT, storage_label TEXT,
  metric TEXT NOT NULL DEFAULT 'cosine' CHECK(metric = 'cosine'),
  dimension_state TEXT NOT NULL CHECK(dimension_state IN ('pending', 'resolved')),
  dimensions INTEGER CHECK(dimensions IS NULL OR dimensions >= 1),
  normalization_version TEXT NOT NULL CHECK(char_length(btrim(normalization_version)) > 0),
  chunker_version TEXT NOT NULL CHECK(char_length(btrim(chunker_version)) > 0),
  enabled_at TIMESTAMPTZ, disabled_at TIMESTAMPTZ, consented_at TIMESTAMPTZ,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY(owner_user_id, dataset_id),
  CHECK(
    (dimension_state = 'pending' AND dimensions IS NULL AND compatibility_hash IS NULL)
    OR
    (dimension_state = 'resolved' AND dimensions IS NOT NULL AND compatibility_hash IS NOT NULL
      AND compatibility_hash ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$')
  )
);
CREATE TABLE IF NOT EXISTS note_semantic_generations(
  id TEXT PRIMARY KEY CHECK(char_length(btrim(id)) > 0),
  owner_user_id TEXT NOT NULL CHECK(char_length(btrim(owner_user_id)) > 0),
  dataset_id TEXT NOT NULL CHECK(char_length(btrim(dataset_id)) > 0),
  configuration_revision INTEGER NOT NULL CHECK(configuration_revision >= 1),
  state TEXT NOT NULL CHECK(state IN ('staging', 'active', 'retired', 'failed', 'deleting')),
  compatibility_hash TEXT,
  dimension_state TEXT NOT NULL CHECK(dimension_state IN ('pending', 'resolved')),
  dimensions INTEGER CHECK(dimensions IS NULL OR dimensions >= 1), root_job_id TEXT,
  expected_note_count INTEGER NOT NULL DEFAULT 0 CHECK(expected_note_count >= 0),
  expected_chunk_count INTEGER NOT NULL DEFAULT 0 CHECK(expected_chunk_count >= 0),
  published_note_count INTEGER NOT NULL DEFAULT 0 CHECK(published_note_count >= 0),
  published_chunk_count INTEGER NOT NULL DEFAULT 0 CHECK(published_chunk_count >= 0),
  manifest_hash TEXT, publication_receipt TEXT, terminal_error_code TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP, published_at TIMESTAMPTZ,
  retired_at TIMESTAMPTZ, deleted_at TIMESTAMPTZ,
  UNIQUE(owner_user_id, dataset_id, id),
  FOREIGN KEY(owner_user_id, dataset_id) REFERENCES note_semantic_index_configs(owner_user_id, dataset_id) ON DELETE CASCADE ON UPDATE CASCADE,
  CHECK(
    (dimension_state = 'pending' AND dimensions IS NULL AND compatibility_hash IS NULL)
    OR
    (dimension_state = 'resolved' AND dimensions IS NOT NULL AND compatibility_hash IS NOT NULL
      AND compatibility_hash ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$')
  )
);
CREATE TABLE IF NOT EXISTS note_semantic_note_state(
  owner_user_id TEXT NOT NULL CHECK(char_length(btrim(owner_user_id)) > 0),
  dataset_id TEXT NOT NULL CHECK(char_length(btrim(dataset_id)) > 0), generation_id TEXT NOT NULL, note_id TEXT NOT NULL,
  content_version INTEGER NOT NULL CHECK(content_version >= 1), content_fingerprint TEXT NOT NULL CHECK(content_fingerprint ~ '^sha256:[0-9a-f]{64}$'),
  dirty_generation INTEGER NOT NULL CHECK(dirty_generation >= 1),
  state TEXT NOT NULL CHECK(state IN ('pending', 'indexed', 'excluded', 'failed', 'tombstoned')),
  chunk_count INTEGER NOT NULL DEFAULT 0 CHECK(chunk_count >= 0), manifest_hash TEXT, error_code TEXT, published_at TIMESTAMPTZ,
  PRIMARY KEY(owner_user_id, dataset_id, generation_id, note_id),
  FOREIGN KEY(owner_user_id, dataset_id, generation_id) REFERENCES note_semantic_generations(owner_user_id, dataset_id, id) ON DELETE CASCADE ON UPDATE CASCADE,
  FOREIGN KEY(owner_user_id, note_id) REFERENCES notes(client_id, id) ON DELETE CASCADE ON UPDATE CASCADE
);
CREATE TABLE IF NOT EXISTS note_semantic_chunks(
  chunk_id TEXT PRIMARY KEY CHECK(char_length(btrim(chunk_id)) > 0),
  owner_user_id TEXT NOT NULL CHECK(char_length(btrim(owner_user_id)) > 0), dataset_id TEXT NOT NULL CHECK(char_length(btrim(dataset_id)) > 0),
  generation_id TEXT NOT NULL, note_id TEXT NOT NULL, content_version INTEGER NOT NULL CHECK(content_version >= 1), ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
  field TEXT NOT NULL CHECK(field IN ('title', 'content')), start_offset INTEGER NOT NULL CHECK(start_offset >= 0), end_offset INTEGER NOT NULL CHECK(end_offset > start_offset),
  chunk_fingerprint TEXT NOT NULL CHECK(chunk_fingerprint ~ '^sha256:[0-9a-f]{64}$'), normalization_version TEXT NOT NULL CHECK(char_length(btrim(normalization_version)) > 0),
  chunker_version TEXT NOT NULL CHECK(char_length(btrim(chunker_version)) > 0),
  UNIQUE(owner_user_id, dataset_id, chunk_id), UNIQUE(owner_user_id, dataset_id, generation_id, note_id, ordinal),
  FOREIGN KEY(owner_user_id, dataset_id, generation_id, note_id) REFERENCES note_semantic_note_state(owner_user_id, dataset_id, generation_id, note_id) ON DELETE CASCADE ON UPDATE CASCADE
);
CREATE TABLE IF NOT EXISTS note_semantic_work(
  id TEXT PRIMARY KEY CHECK(char_length(btrim(id)) > 0),
  owner_user_id TEXT NOT NULL CHECK(char_length(btrim(owner_user_id)) > 0), dataset_id TEXT NOT NULL CHECK(char_length(btrim(dataset_id)) > 0),
  kind TEXT NOT NULL CHECK(kind IN ('index_note', 'delete_note_vectors', 'delete_generation')), note_id TEXT, generation_id TEXT,
  dirty_generation INTEGER CHECK(dirty_generation IS NULL OR dirty_generation >= 1), fencing_token TEXT NOT NULL CHECK(char_length(btrim(fencing_token)) > 0),
  claim_state TEXT NOT NULL CHECK(claim_state IN ('pending', 'claimed', 'completed', 'failed')),
  attempt_count INTEGER NOT NULL DEFAULT 0 CHECK(attempt_count >= 0 AND attempt_count <= 5), next_eligible_at TIMESTAMPTZ NOT NULL,
  claim_token TEXT, claimed_at TIMESTAMPTZ, error_code TEXT, created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(owner_user_id, dataset_id, id),
  FOREIGN KEY(owner_user_id, dataset_id, generation_id) REFERENCES note_semantic_generations(owner_user_id, dataset_id, id) ON DELETE CASCADE ON UPDATE CASCADE,
  CHECK(generation_id IS NOT NULL AND (
    (kind = 'delete_generation' AND note_id IS NULL AND dirty_generation IS NULL)
    OR
    (kind IN ('index_note', 'delete_note_vectors') AND note_id IS NOT NULL AND dirty_generation IS NOT NULL)
  )),
  CHECK((claim_state = 'claimed' AND claim_token IS NOT NULL AND claimed_at IS NOT NULL) OR (claim_state != 'claimed'))
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_note_semantic_generations_one_active ON note_semantic_generations(owner_user_id, dataset_id) WHERE state = 'active';
CREATE UNIQUE INDEX IF NOT EXISTS idx_note_semantic_generations_one_staging ON note_semantic_generations(owner_user_id, dataset_id) WHERE state = 'staging';
CREATE INDEX IF NOT EXISTS idx_note_semantic_note_state_pending ON note_semantic_note_state(owner_user_id, dataset_id, generation_id, state, dirty_generation);
CREATE INDEX IF NOT EXISTS idx_note_semantic_chunks_note ON note_semantic_chunks(owner_user_id, dataset_id, generation_id, note_id, ordinal);
CREATE INDEX IF NOT EXISTS idx_note_semantic_work_claimable ON note_semantic_work(owner_user_id, dataset_id, claim_state, next_eligible_at, id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_note_semantic_work_note_coalesce ON note_semantic_work(owner_user_id, dataset_id, kind, note_id) WHERE note_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS idx_note_semantic_work_generation_coalesce ON note_semantic_work(owner_user_id, dataset_id, kind, generation_id) WHERE generation_id IS NOT NULL AND kind = 'delete_generation';
ALTER TABLE note_semantic_index_configs ENABLE ROW LEVEL SECURITY;
ALTER TABLE note_semantic_index_configs FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS note_semantic_index_configs_tenant_isolation ON note_semantic_index_configs;
CREATE POLICY note_semantic_index_configs_tenant_isolation ON note_semantic_index_configs USING (owner_user_id = current_setting('app.current_user_id', true) AND dataset_id = current_setting('app.current_dataset_id', true)) WITH CHECK (owner_user_id = current_setting('app.current_user_id', true) AND dataset_id = current_setting('app.current_dataset_id', true));
ALTER TABLE note_semantic_generations ENABLE ROW LEVEL SECURITY;
ALTER TABLE note_semantic_generations FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS note_semantic_generations_tenant_isolation ON note_semantic_generations;
CREATE POLICY note_semantic_generations_tenant_isolation ON note_semantic_generations USING (owner_user_id = current_setting('app.current_user_id', true) AND dataset_id = current_setting('app.current_dataset_id', true)) WITH CHECK (owner_user_id = current_setting('app.current_user_id', true) AND dataset_id = current_setting('app.current_dataset_id', true));
ALTER TABLE note_semantic_note_state ENABLE ROW LEVEL SECURITY;
ALTER TABLE note_semantic_note_state FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS note_semantic_note_state_tenant_isolation ON note_semantic_note_state;
CREATE POLICY note_semantic_note_state_tenant_isolation ON note_semantic_note_state USING (owner_user_id = current_setting('app.current_user_id', true) AND dataset_id = current_setting('app.current_dataset_id', true)) WITH CHECK (owner_user_id = current_setting('app.current_user_id', true) AND dataset_id = current_setting('app.current_dataset_id', true));
ALTER TABLE note_semantic_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE note_semantic_chunks FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS note_semantic_chunks_tenant_isolation ON note_semantic_chunks;
CREATE POLICY note_semantic_chunks_tenant_isolation ON note_semantic_chunks USING (owner_user_id = current_setting('app.current_user_id', true) AND dataset_id = current_setting('app.current_dataset_id', true)) WITH CHECK (owner_user_id = current_setting('app.current_user_id', true) AND dataset_id = current_setting('app.current_dataset_id', true));
ALTER TABLE note_semantic_work ENABLE ROW LEVEL SECURITY;
ALTER TABLE note_semantic_work FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS note_semantic_work_tenant_isolation ON note_semantic_work;
CREATE POLICY note_semantic_work_tenant_isolation ON note_semantic_work USING (owner_user_id = current_setting('app.current_user_id', true) AND dataset_id = current_setting('app.current_dataset_id', true)) WITH CHECK (owner_user_id = current_setting('app.current_user_id', true) AND dataset_id = current_setting('app.current_dataset_id', true));
"""

_MIGRATION_SQL_V82_TO_V83 = """
CREATE TABLE note_semantic_obsolete_vectors(
  id TEXT PRIMARY KEY CHECK(length(trim(id)) > 0),
  owner_user_id TEXT NOT NULL CHECK(length(trim(owner_user_id)) > 0),
  dataset_id TEXT NOT NULL CHECK(length(trim(dataset_id)) > 0),
  generation_id TEXT NOT NULL CHECK(length(trim(generation_id)) > 0),
  vector_id TEXT NOT NULL CHECK(length(trim(vector_id)) > 0),
  note_id TEXT,
  source_kind TEXT NOT NULL CHECK(source_kind IN (
    'unpublished','manifest_replace','tombstone','hard_delete','note_failure'
  )),
  dirty_generation INTEGER CHECK(dirty_generation IS NULL OR dirty_generation >= 1),
  claim_state TEXT NOT NULL CHECK(claim_state IN ('pending','claimed','failed')),
  attempt_count INTEGER NOT NULL DEFAULT 0 CHECK(attempt_count >= 0 AND attempt_count <= 5),
  next_eligible_at DATETIME NOT NULL,
  claim_token TEXT,
  claimed_at DATETIME,
  error_code TEXT,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(owner_user_id,dataset_id,generation_id,vector_id),
  CHECK(
    (claim_state='claimed' AND claim_token IS NOT NULL AND claimed_at IS NOT NULL)
    OR
    (claim_state<>'claimed' AND claim_token IS NULL AND claimed_at IS NULL)
  )
);
CREATE INDEX idx_note_semantic_obsolete_vectors_claimable
  ON note_semantic_obsolete_vectors(
    owner_user_id,dataset_id,claim_state,next_eligible_at,generation_id,id
  );
CREATE INDEX idx_note_semantic_obsolete_vectors_generation
  ON note_semantic_obsolete_vectors(owner_user_id,dataset_id,generation_id,vector_id);
"""

_MIGRATION_SQL_V86_TO_V87_POSTGRES = """
CREATE TABLE IF NOT EXISTS note_semantic_obsolete_vectors(
  id TEXT PRIMARY KEY CHECK(char_length(btrim(id)) > 0),
  owner_user_id TEXT NOT NULL CHECK(char_length(btrim(owner_user_id)) > 0),
  dataset_id TEXT NOT NULL CHECK(char_length(btrim(dataset_id)) > 0),
  generation_id TEXT NOT NULL CHECK(char_length(btrim(generation_id)) > 0),
  vector_id TEXT NOT NULL CHECK(char_length(btrim(vector_id)) > 0),
  note_id TEXT,
  source_kind TEXT NOT NULL CHECK(source_kind IN (
    'unpublished','manifest_replace','tombstone','hard_delete','note_failure'
  )),
  dirty_generation INTEGER CHECK(dirty_generation IS NULL OR dirty_generation >= 1),
  claim_state TEXT NOT NULL CHECK(claim_state IN ('pending','claimed','failed')),
  attempt_count INTEGER NOT NULL DEFAULT 0 CHECK(attempt_count >= 0 AND attempt_count <= 5),
  next_eligible_at TIMESTAMPTZ NOT NULL,
  claim_token TEXT,
  claimed_at TIMESTAMPTZ,
  error_code TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(owner_user_id,dataset_id,generation_id,vector_id),
  CHECK(
    (claim_state='claimed' AND claim_token IS NOT NULL AND claimed_at IS NOT NULL)
    OR
    (claim_state<>'claimed' AND claim_token IS NULL AND claimed_at IS NULL)
  )
);
CREATE INDEX IF NOT EXISTS idx_note_semantic_obsolete_vectors_claimable
  ON note_semantic_obsolete_vectors(
    owner_user_id,dataset_id,claim_state,next_eligible_at,generation_id,id
  );
CREATE INDEX IF NOT EXISTS idx_note_semantic_obsolete_vectors_generation
  ON note_semantic_obsolete_vectors(owner_user_id,dataset_id,generation_id,vector_id);
ALTER TABLE note_semantic_obsolete_vectors ENABLE ROW LEVEL SECURITY;
ALTER TABLE note_semantic_obsolete_vectors FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS note_semantic_obsolete_vectors_tenant_isolation
  ON note_semantic_obsolete_vectors;
CREATE POLICY note_semantic_obsolete_vectors_tenant_isolation
  ON note_semantic_obsolete_vectors
  USING (
    owner_user_id=current_setting('app.current_user_id', true)
    AND dataset_id=current_setting('app.current_dataset_id', true)
  )
  WITH CHECK (
    owner_user_id=current_setting('app.current_user_id', true)
    AND dataset_id=current_setting('app.current_dataset_id', true)
  );
"""

_MIGRATION_SQL_V83_TO_V84 = """
ALTER TABLE note_semantic_index_configs ADD COLUMN model_revision TEXT;
ALTER TABLE note_semantic_generations ADD COLUMN model_revision TEXT;
CREATE UNIQUE INDEX idx_note_semantic_generations_root_job
  ON note_semantic_generations(owner_user_id,dataset_id,root_job_id)
  WHERE root_job_id IS NOT NULL;
CREATE TABLE note_semantic_operation_receipts(
  owner_user_id TEXT NOT NULL CHECK(length(trim(owner_user_id)) > 0),
  dataset_id TEXT NOT NULL CHECK(length(trim(dataset_id)) > 0),
  key_digest TEXT NOT NULL CHECK(length(key_digest) = 64),
  action TEXT NOT NULL CHECK(action IN ('enable','cancel')),
  request_fingerprint TEXT NOT NULL CHECK(length(request_fingerprint) = 64),
  run_id TEXT,
  expected_revision INTEGER NOT NULL CHECK(expected_revision >= 0),
  state TEXT NOT NULL CHECK(state IN ('pending','completed')),
  response_json TEXT CHECK(response_json IS NULL OR length(response_json) <= 8192),
  expires_at DATETIME NOT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY(owner_user_id,key_digest)
);
CREATE INDEX idx_note_semantic_operation_receipts_scope
  ON note_semantic_operation_receipts(
    owner_user_id,dataset_id,action,state,updated_at,key_digest
  );
CREATE INDEX idx_note_semantic_operation_receipts_expiry
  ON note_semantic_operation_receipts(
    owner_user_id,dataset_id,expires_at,key_digest
  );
"""

_MIGRATION_SQL_V87_TO_V88_POSTGRES = """
ALTER TABLE note_semantic_index_configs
  ADD COLUMN IF NOT EXISTS model_revision TEXT;
ALTER TABLE note_semantic_generations
  ADD COLUMN IF NOT EXISTS model_revision TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS idx_note_semantic_generations_root_job
  ON note_semantic_generations(owner_user_id,dataset_id,root_job_id)
  WHERE root_job_id IS NOT NULL;
CREATE TABLE IF NOT EXISTS note_semantic_operation_receipts(
  owner_user_id TEXT NOT NULL CHECK(char_length(btrim(owner_user_id)) > 0),
  dataset_id TEXT NOT NULL CHECK(char_length(btrim(dataset_id)) > 0),
  key_digest TEXT NOT NULL CHECK(char_length(key_digest) = 64),
  action TEXT NOT NULL CHECK(action IN ('enable','cancel')),
  request_fingerprint TEXT NOT NULL CHECK(char_length(request_fingerprint) = 64),
  run_id TEXT,
  expected_revision INTEGER NOT NULL CHECK(expected_revision >= 0),
  state TEXT NOT NULL CHECK(state IN ('pending','completed')),
  response_json TEXT CHECK(response_json IS NULL OR octet_length(response_json) <= 8192),
  expires_at TIMESTAMPTZ NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY(owner_user_id,key_digest)
);
CREATE INDEX IF NOT EXISTS idx_note_semantic_operation_receipts_scope
  ON note_semantic_operation_receipts(
    owner_user_id,dataset_id,action,state,updated_at,key_digest
  );
CREATE INDEX IF NOT EXISTS idx_note_semantic_operation_receipts_expiry
  ON note_semantic_operation_receipts(
    owner_user_id,dataset_id,expires_at,key_digest
  );
ALTER TABLE note_semantic_operation_receipts ENABLE ROW LEVEL SECURITY;
ALTER TABLE note_semantic_operation_receipts FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS note_semantic_operation_receipts_tenant_isolation
  ON note_semantic_operation_receipts;
CREATE POLICY note_semantic_operation_receipts_tenant_isolation
  ON note_semantic_operation_receipts
  USING (
    owner_user_id=current_setting('app.current_user_id', true)
    AND dataset_id=current_setting('app.current_dataset_id', true)
  )
  WITH CHECK (
    owner_user_id=current_setting('app.current_user_id', true)
    AND dataset_id=current_setting('app.current_dataset_id', true)
  );
"""

_NOTE_GRAPH_SUGGESTION_RECEIPT_DELETE_SQLITE_TRIGGER_SQL = """
CREATE TRIGGER note_graph_suggestion_operation_receipts_clear_references
BEFORE DELETE ON note_graph_suggestion_operation_receipts
BEGIN
  UPDATE note_graph_suggestion_runs
     SET admission_receipt_id = NULL
   WHERE owner_user_id = OLD.owner_user_id
     AND dataset_id = OLD.dataset_id
     AND admission_receipt_id = OLD.id
     AND EXISTS (
       SELECT 1 FROM notes
        WHERE client_id = OLD.owner_user_id AND id = OLD.source_note_id
     );

  UPDATE note_graph_suggestions
     SET decision_receipt_id = NULL
   WHERE owner_user_id = OLD.owner_user_id
     AND dataset_id = OLD.dataset_id
     AND decision_receipt_id = OLD.id
     AND EXISTS (
       SELECT 1 FROM notes
        WHERE client_id = OLD.owner_user_id AND id = OLD.source_note_id
     );
END;
"""

_NOTE_GRAPH_SUGGESTION_RECEIPT_DELETE_POSTGRES_TRIGGER_STATEMENTS = (
    """
        CREATE OR REPLACE FUNCTION note_graph_suggestion_clear_receipt_references()
        RETURNS trigger AS $$
        BEGIN
          UPDATE note_graph_suggestion_runs
             SET admission_receipt_id = NULL
           WHERE owner_user_id = OLD.owner_user_id
             AND dataset_id = OLD.dataset_id
             AND admission_receipt_id = OLD.id
             AND EXISTS (
               SELECT 1 FROM notes
                WHERE client_id = OLD.owner_user_id AND id = OLD.source_note_id
             );

          UPDATE note_graph_suggestions
             SET decision_receipt_id = NULL
           WHERE owner_user_id = OLD.owner_user_id
             AND dataset_id = OLD.dataset_id
             AND decision_receipt_id = OLD.id
             AND EXISTS (
               SELECT 1 FROM notes
                WHERE client_id = OLD.owner_user_id AND id = OLD.source_note_id
             );
          RETURN OLD;
        END;
        $$ LANGUAGE plpgsql
        """,
    "DROP TRIGGER IF EXISTS note_graph_suggestion_operation_receipts_clear_references "
    "ON note_graph_suggestion_operation_receipts",
    "CREATE TRIGGER note_graph_suggestion_operation_receipts_clear_references "
    "BEFORE DELETE ON note_graph_suggestion_operation_receipts "
    "FOR EACH ROW EXECUTE FUNCTION note_graph_suggestion_clear_receipt_references()",
)

_SHARED_WORKSPACE_CHAT_V61_POSTGRES_VERIFY_SQL = """
DO $shared_workspace_chat_v61_verify$
BEGIN
  IF (
    SELECT count(*)
      FROM pg_class AS relation
      JOIN pg_namespace AS namespace ON namespace.oid = relation.relnamespace
     WHERE namespace.nspname = current_schema()
       AND relation.relname IN (
         'shared_workspace_chat_threads',
         'shared_workspace_chat_requests'
       )
       AND relation.relrowsecurity
       AND relation.relforcerowsecurity
  ) <> 2 THEN
    RAISE EXCEPTION 'Shared workspace chat v61 relations require forced RLS';
  END IF;

  IF (
    SELECT count(*)
      -- Presence checks need stored trees, not pg_policies expression deparsing.
      -- Deparsing can lock Media relations during concurrent schema bootstrap.
      FROM pg_policy AS policy
      JOIN pg_class AS relation ON relation.oid = policy.polrelid
      JOIN pg_namespace AS namespace ON namespace.oid = relation.relnamespace
     WHERE namespace.nspname = current_schema()
       AND (relation.relname, policy.polname) IN (
         (
           'shared_workspace_chat_threads',
           'shared_workspace_chat_threads_tenant_isolation'
         ),
         (
           'shared_workspace_chat_requests',
           'shared_workspace_chat_requests_tenant_isolation'
         )
       )
       AND policy.polqual IS NOT NULL
       AND policy.polwithcheck IS NOT NULL
  ) <> 2 THEN
    RAISE EXCEPTION 'Shared workspace chat v61 policy catalog is incomplete';
  END IF;
END;
$shared_workspace_chat_v61_verify$;
"""

_MIGRATION_SQL_V64_TO_V65 = """
CREATE TABLE conversation_behavior_snapshots(
  conversation_id TEXT PRIMARY KEY REFERENCES conversations(id) ON DELETE CASCADE,
  status TEXT NOT NULL CHECK(status IN ('valid','missing','invalid')),
  schema_version INTEGER,
  canonical_json TEXT,
  digest TEXT,
  size_bytes INTEGER,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CHECK(
    (
      status = 'valid'
      AND schema_version IS NOT NULL
      AND schema_version >= 1
      AND canonical_json IS NOT NULL
      AND digest IS NOT NULL
      AND length(digest) = 71
      AND substr(digest, 1, 7) = 'sha256:'
      AND substr(digest, 8) NOT GLOB '*[^0-9a-f]*'
      AND size_bytes IS NOT NULL
      AND size_bytes >= 1
      AND length(CAST(canonical_json AS BLOB)) = size_bytes
    )
    OR
    (
      status IN ('missing','invalid')
      AND schema_version IS NULL
      AND canonical_json IS NULL
      AND digest IS NULL
      AND size_bytes IS NULL
    )
  )
);
CREATE INDEX idx_conversation_behavior_snapshots_status
  ON conversation_behavior_snapshots(status);
CREATE INDEX idx_conversation_behavior_snapshots_digest
  ON conversation_behavior_snapshots(digest);
ALTER TABLE conversations
  ADD COLUMN history_version INTEGER NOT NULL DEFAULT 1 CHECK(history_version >= 1);
ALTER TABLE conversation_settings
  ADD COLUMN settings_version INTEGER NOT NULL DEFAULT 1 CHECK(settings_version >= 1);

UPDATE db_schema_version
   SET version = 65
 WHERE schema_name = 'rag_char_chat_schema'
   AND version = 64;
"""

_MIGRATION_SQL_V64_TO_V65_POSTGRES = """
CREATE TABLE conversation_behavior_snapshots(
  conversation_id TEXT PRIMARY KEY REFERENCES conversations(id) ON DELETE CASCADE,
  status TEXT NOT NULL CHECK(status IN ('valid','missing','invalid')),
  schema_version INTEGER,
  canonical_json TEXT,
  digest TEXT,
  size_bytes INTEGER,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CHECK(
    (
      status = 'valid'
      AND schema_version IS NOT NULL
      AND schema_version >= 1
      AND canonical_json IS NOT NULL
      AND digest IS NOT NULL
      AND digest ~ '^sha256:[0-9a-f]{64}$'
      AND size_bytes IS NOT NULL
      AND size_bytes >= 1
      AND octet_length(canonical_json) = size_bytes
    )
    OR
    (
      status IN ('missing','invalid')
      AND schema_version IS NULL
      AND canonical_json IS NULL
      AND digest IS NULL
      AND size_bytes IS NULL
    )
  )
);
CREATE INDEX idx_conversation_behavior_snapshots_status
  ON conversation_behavior_snapshots(status);
CREATE INDEX idx_conversation_behavior_snapshots_digest
  ON conversation_behavior_snapshots(digest);
ALTER TABLE conversations
  ADD COLUMN history_version INTEGER NOT NULL DEFAULT 1
  CHECK (history_version >= 1);
ALTER TABLE conversation_settings
  ADD COLUMN settings_version INTEGER NOT NULL DEFAULT 1
  CHECK (settings_version >= 1);

UPDATE db_schema_version
   SET version = 65
 WHERE schema_name = 'rag_char_chat_schema'
   AND version = 64;
"""

_MIGRATION_SQL_V66_TO_V67 = """
ALTER TABLE quizzes
  ADD COLUMN activity_type TEXT NOT NULL DEFAULT 'questions'
  CHECK (activity_type IN ('questions', 'osce'));
ALTER TABLE quizzes
  ADD COLUMN generation_profile TEXT;
ALTER TABLE quizzes
  ADD COLUMN total_stations INTEGER NOT NULL DEFAULT 0
  CHECK (total_stations >= 0);

CREATE TABLE osce_stations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  quiz_id INTEGER NOT NULL REFERENCES quizzes(id) ON DELETE CASCADE,
  schema_version TEXT NOT NULL CHECK (schema_version = 'osce.station.v1'),
  content_json TEXT NOT NULL,
  order_index INTEGER NOT NULL CHECK (order_index >= 0),
  version INTEGER NOT NULL DEFAULT 1 CHECK (version >= 1),
  origin TEXT NOT NULL CHECK (origin IN ('generated', 'manual')),
  provenance_json TEXT,
  source_bundle_json TEXT,
  verification_state TEXT NOT NULL
    CHECK (verification_state IN ('source_verified', 'modified_after_verification', 'manually_authored')),
  verification_timestamp TEXT,
  verification_summary TEXT,
  deleted INTEGER NOT NULL DEFAULT 0 CHECK (deleted IN (0, 1)),
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE osce_practice_attempts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  station_id INTEGER NOT NULL REFERENCES osce_stations(id) ON DELETE CASCADE,
  quiz_id INTEGER NOT NULL REFERENCES quizzes(id) ON DELETE CASCADE,
  client_attempt_id TEXT NOT NULL,
  station_snapshot_json TEXT NOT NULL,
  state TEXT NOT NULL CHECK (state IN ('in_progress', 'self_assessment', 'completed')),
  candidate_notes TEXT NOT NULL DEFAULT '' CHECK (length(candidate_notes) <= 10000),
  checklist_selections_json TEXT NOT NULL DEFAULT '{}',
  rubric_selections_json TEXT NOT NULL DEFAULT '{}',
  started_at TEXT NOT NULL,
  self_assessment_started_at TEXT,
  completed_at TEXT,
  frozen_elapsed_seconds INTEGER CHECK (frozen_elapsed_seconds IS NULL OR frozen_elapsed_seconds >= 0),
  version INTEGER NOT NULL DEFAULT 1 CHECK (version >= 1),
  last_modified_at TEXT NOT NULL,
  UNIQUE (station_id, client_attempt_id)
);

CREATE INDEX idx_osce_stations_quiz_active_order
  ON osce_stations(quiz_id, deleted, order_index);
CREATE INDEX idx_osce_attempts_station_state_modified
  ON osce_practice_attempts(station_id, state, last_modified_at DESC);
CREATE INDEX idx_osce_attempts_quiz_state_modified
  ON osce_practice_attempts(quiz_id, state, last_modified_at DESC);

UPDATE db_schema_version
   SET version = 67
 WHERE schema_name = 'rag_char_chat_schema'
   AND version = 66;
"""

_MIGRATION_SQL_V72_TO_V73_PERSONA_COMPANION = """
ALTER TABLE persona_visual_packs ADD COLUMN companion_behavior_json TEXT;

CREATE TABLE IF NOT EXISTS persona_buddy_preferences (
  user_id      TEXT PRIMARY KEY,
  ambient_mode TEXT NOT NULL CHECK(ambient_mode IN ('off', 'expressive', 'roaming')),
  version      INTEGER NOT NULL DEFAULT 1 CHECK(version >= 1),
  created_at   TEXT NOT NULL,
  updated_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS persona_visual_pack_reviews (
  id               TEXT PRIMARY KEY,
  pack_id          TEXT NOT NULL REFERENCES persona_visual_packs(id) ON DELETE CASCADE,
  user_id          TEXT NOT NULL,
  reviewer_user_id TEXT NOT NULL,
  fingerprint      TEXT NOT NULL CHECK(length(fingerprint) = 64),
  pack_version     INTEGER NOT NULL CHECK(pack_version >= 1),
  reviewed_at      TEXT NOT NULL,
  created_at       TEXT NOT NULL,
  UNIQUE(pack_id, fingerprint)
);
CREATE INDEX IF NOT EXISTS idx_persona_visual_pack_reviews_pack
  ON persona_visual_pack_reviews(pack_id, user_id, reviewed_at);

UPDATE db_schema_version SET version = 73
 WHERE schema_name = 'rag_char_chat_schema' AND version = 72;
"""

_MIGRATION_SQL_V76_TO_V77_PERSONA_COMPANION_POSTGRES = """
ALTER TABLE persona_visual_packs ADD COLUMN IF NOT EXISTS companion_behavior_json TEXT;

CREATE TABLE IF NOT EXISTS persona_buddy_preferences (
  user_id      TEXT PRIMARY KEY,
  ambient_mode TEXT NOT NULL CHECK(ambient_mode IN ('off', 'expressive', 'roaming')),
  version      INTEGER NOT NULL DEFAULT 1 CHECK(version >= 1),
  created_at   TIMESTAMP NOT NULL,
  updated_at   TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS persona_visual_pack_reviews (
  id               TEXT PRIMARY KEY,
  pack_id          TEXT NOT NULL REFERENCES persona_visual_packs(id) ON DELETE CASCADE,
  user_id          TEXT NOT NULL,
  reviewer_user_id TEXT NOT NULL,
  fingerprint      TEXT NOT NULL CHECK(length(fingerprint) = 64),
  pack_version     INTEGER NOT NULL CHECK(pack_version >= 1),
  reviewed_at      TIMESTAMP NOT NULL,
  created_at       TIMESTAMP NOT NULL,
  UNIQUE(pack_id, fingerprint)
);
CREATE INDEX IF NOT EXISTS idx_persona_visual_pack_reviews_pack
  ON persona_visual_pack_reviews(pack_id, user_id, reviewed_at);

UPDATE db_schema_version SET version = 77
 WHERE schema_name = 'rag_char_chat_schema' AND version = 76;
"""

_MIGRATION_SQL_V66_TO_V67_POSTGRES = """
ALTER TABLE quizzes
  ADD COLUMN IF NOT EXISTS activity_type TEXT NOT NULL DEFAULT 'questions';
ALTER TABLE quizzes
  ADD COLUMN IF NOT EXISTS generation_profile TEXT;
ALTER TABLE quizzes
  ADD COLUMN IF NOT EXISTS total_stations INTEGER NOT NULL DEFAULT 0;
ALTER TABLE quizzes
  ADD CONSTRAINT quizzes_activity_type_check CHECK (activity_type IN ('questions', 'osce'));
ALTER TABLE quizzes
  ADD CONSTRAINT quizzes_total_stations_check CHECK (total_stations >= 0);

CREATE TABLE osce_stations (
  id BIGSERIAL PRIMARY KEY,
  quiz_id BIGINT NOT NULL REFERENCES quizzes(id) ON DELETE CASCADE,
  schema_version TEXT NOT NULL CHECK (schema_version = 'osce.station.v1'),
  content_json TEXT NOT NULL,
  order_index INTEGER NOT NULL CHECK (order_index >= 0),
  version INTEGER NOT NULL DEFAULT 1 CHECK (version >= 1),
  origin TEXT NOT NULL CHECK (origin IN ('generated', 'manual')),
  provenance_json TEXT,
  source_bundle_json TEXT,
  verification_state TEXT NOT NULL
    CHECK (verification_state IN ('source_verified', 'modified_after_verification', 'manually_authored')),
  verification_timestamp TIMESTAMPTZ,
  verification_summary TEXT,
  deleted BOOLEAN NOT NULL DEFAULT FALSE,
  created_at TIMESTAMPTZ NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE osce_practice_attempts (
  id BIGSERIAL PRIMARY KEY,
  station_id BIGINT NOT NULL REFERENCES osce_stations(id) ON DELETE CASCADE,
  quiz_id BIGINT NOT NULL REFERENCES quizzes(id) ON DELETE CASCADE,
  client_attempt_id TEXT NOT NULL,
  station_snapshot_json TEXT NOT NULL,
  state TEXT NOT NULL CHECK (state IN ('in_progress', 'self_assessment', 'completed')),
  candidate_notes TEXT NOT NULL DEFAULT '' CHECK (char_length(candidate_notes) <= 10000),
  checklist_selections_json TEXT NOT NULL DEFAULT '{}',
  rubric_selections_json TEXT NOT NULL DEFAULT '{}',
  started_at TIMESTAMPTZ NOT NULL,
  self_assessment_started_at TIMESTAMPTZ,
  completed_at TIMESTAMPTZ,
  frozen_elapsed_seconds INTEGER CHECK (frozen_elapsed_seconds IS NULL OR frozen_elapsed_seconds >= 0),
  version INTEGER NOT NULL DEFAULT 1 CHECK (version >= 1),
  last_modified_at TIMESTAMPTZ NOT NULL,
  UNIQUE (station_id, client_attempt_id)
);

CREATE INDEX idx_osce_stations_quiz_active_order
  ON osce_stations(quiz_id, deleted, order_index);
CREATE INDEX idx_osce_attempts_station_state_modified
  ON osce_practice_attempts(station_id, state, last_modified_at DESC);
CREATE INDEX idx_osce_attempts_quiz_state_modified
  ON osce_practice_attempts(quiz_id, state, last_modified_at DESC);

UPDATE db_schema_version
   SET version = 67
 WHERE schema_name = 'rag_char_chat_schema'
   AND version = 66;
"""

_MIGRATION_SQL_V68_TO_V69 = """
ALTER TABLE workspaces ADD COLUMN assistant_defaults_explicit_none INTEGER NOT NULL DEFAULT 0
    CHECK (assistant_defaults_explicit_none IN (0, 1));
UPDATE workspaces SET assistant_defaults_explicit_none = 1 WHERE assistant_defaults_json IS NULL;
UPDATE db_schema_version SET version = 69 WHERE schema_name = 'rag_char_chat_schema' AND version = 68;
"""

_MIGRATION_SQL_V72_TO_V73_POSTGRES = """
ALTER TABLE workspaces ADD COLUMN assistant_defaults_explicit_none BOOLEAN NOT NULL DEFAULT FALSE;
UPDATE workspaces SET assistant_defaults_explicit_none = TRUE WHERE assistant_defaults_json IS NULL;
UPDATE db_schema_version SET version = 73 WHERE schema_name = 'rag_char_chat_schema' AND version = 72;
"""

_MIGRATION_SQL_V69_TO_V70 = """
ALTER TABLE conversations ADD COLUMN assistant_startup_json TEXT
    CHECK (assistant_startup_json IS NULL OR length(CAST(assistant_startup_json AS BLOB)) <= 1024);
UPDATE db_schema_version SET version = 70 WHERE schema_name = 'rag_char_chat_schema' AND version = 69;
"""

_MIGRATION_SQL_V73_TO_V74_POSTGRES = """
ALTER TABLE conversations ADD COLUMN assistant_startup_json TEXT
    CHECK (assistant_startup_json IS NULL OR octet_length(assistant_startup_json) <= 1024);
UPDATE db_schema_version SET version = 74 WHERE schema_name = 'rag_char_chat_schema' AND version = 73;
"""

# D7 P3: fingerprint of a client-id create request, written by the same INSERT.
# Existing chats stay NULL, so they are never mistaken for client-id creates.
_MIGRATION_SQL_V74_TO_V75 = """
ALTER TABLE conversations ADD COLUMN create_request_fingerprint TEXT
    CHECK (create_request_fingerprint IS NULL OR (
        typeof(create_request_fingerprint) = 'text'
        AND length(create_request_fingerprint) = 64
        AND length(CAST(create_request_fingerprint AS BLOB)) = 64
        AND create_request_fingerprint NOT GLOB '*[^0-9a-f]*'
    ));
UPDATE db_schema_version SET version = 75 WHERE schema_name = 'rag_char_chat_schema' AND version = 74;
"""

_MIGRATION_SQL_V78_TO_V79_POSTGRES = """
ALTER TABLE conversations ADD COLUMN create_request_fingerprint TEXT
    CHECK (create_request_fingerprint IS NULL OR (
        length(create_request_fingerprint) = 64 AND create_request_fingerprint !~ '[^0-9a-f]'
    ));
UPDATE db_schema_version SET version = 79 WHERE schema_name = 'rag_char_chat_schema' AND version = 78;
"""

# #3182: until the Sync v2 conversation materializer stopped doing it, a chat whose
# payload named no assistant was stored as persona 'sync-v2'. That persona never
# existed, so persona admission answered 404 for the chat. The repair makes such a
# row a plain chat: no kind, no id, no memory mode, and no startup provenance, which
# belongs to a binding the chat never really had.
#
# Only the exact placeholder is matched, compared byte for byte. A chat whose owner
# has, or had, a persona profile with that id is that owner's own binding and is
# left alone, as is any chat bound to a character. `version` and `last_modified` do
# not move: nothing was edited, and under Sync the row version is the object revision.
_RETIRED_SYNC_PLACEHOLDER_REPAIR_SQL = f"""
UPDATE conversations
   SET assistant_kind = NULL,
       assistant_id = NULL,
       persona_memory_mode = NULL,
       assistant_startup_json = NULL
 WHERE assistant_kind = '{RETIRED_SYNC_PLACEHOLDER_ASSISTANT_KIND}'
   AND assistant_id = '{RETIRED_SYNC_PLACEHOLDER_ASSISTANT_ID}'
   AND character_id IS NULL
   AND NOT EXISTS (
       SELECT 1 FROM persona_profiles AS profile
        WHERE profile.id = '{RETIRED_SYNC_PLACEHOLDER_ASSISTANT_ID}'
          AND profile.user_id = conversations.client_id
   )
"""  # nosec B608 - the interpolated values are the two module constants above.

_MIGRATION_SQL_V10_TO_V11_POSTGRES = """
ALTER TABLE messages ALTER COLUMN content DROP NOT NULL;
"""

_HISTORY_PROJECTIONS_SCHEMA_SQL = """
        CREATE TABLE IF NOT EXISTS conversation_history_projections (
            projection_id TEXT NOT NULL,
            conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            client_id TEXT NOT NULL,
            owner_key TEXT NOT NULL,
            interpretation_version INTEGER NOT NULL CHECK (interpretation_version = 1),
            source_digest TEXT NOT NULL,
            history_fence TEXT NOT NULL,
            source_members_json TEXT NOT NULL,
            ordered_path_ids_json TEXT NOT NULL,
            confirmation_json TEXT NOT NULL,
            projection_digest TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (client_id, owner_key, conversation_id, projection_id)
        )
    """

_LOCAL_UNBOUND_TASK_DATASET_ID = "local-unbound"

_NOTES_MOODBOARD_STUDIO_V61_MIGRATION_PAGE_SIZE = 128

_NOTES_MOODBOARD_STUDIO_V61_MIGRATION_PAGE_SECONDS = 25.0
_NOTES_MOODBOARD_STUDIO_V61_POSTGRES_LOCK_TIMEOUT = "5s"
_NOTES_MOODBOARD_STUDIO_V61_POSTGRES_STATEMENT_TIMEOUT = "30s"
_NOTES_MOODBOARD_STUDIO_V61_MIGRATION_ID = "notes_moodboard_studio_v60_to_v61"
