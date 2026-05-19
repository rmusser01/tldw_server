# Writing Suite Phase 1: Foundation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add manuscript structure (Projects → Parts → Chapters → Scenes) with structured DB tables, REST API, TipTap rich-text editor, manuscript tree panel, and focus mode to the existing WritingPlayground.

**Architecture:** New SQLite tables in ChaChaNotes DB (migration V40→V41) with a separate ManuscriptDB helper class for CRUD. New FastAPI endpoint file for manuscript routes. Frontend evolves the existing WritingPlayground by decomposing the 2358-line index.tsx monolith, adding TipTap as a lazy-loaded editor option, and a manuscript tree panel in the library sidebar.

**Tech Stack:** FastAPI, SQLite FTS5, Pydantic v2, TipTap/ProseMirror, @dnd-kit/react, Zustand, React Query, Ant Design

---

## Task 1: Database Migration V40→V41 — Table Definitions

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`

**Step 1: Add migration SQL class attribute**

Add after `_MIGRATION_SQL_V39_TO_V40_POSTGRES` (after line ~3334). Follow the exact pattern of `_MIGRATION_SQL_V39_TO_V40`:

```python
_MIGRATION_SQL_V40_TO_V41 = """
/*───────────────────────────────────────────────────────────────
  Migration to Version 41 — Manuscript management (2026-04-XX)
───────────────────────────────────────────────────────────────*/
PRAGMA foreign_keys = ON;

-- PROJECTS
CREATE TABLE IF NOT EXISTS manuscript_projects (
  id            TEXT PRIMARY KEY,
  title         TEXT NOT NULL,
  subtitle      TEXT,
  author        TEXT,
  genre         TEXT,
  status        TEXT NOT NULL DEFAULT 'draft'
                  CHECK(status IN ('draft','outlining','writing','revising','complete','archived')),
  synopsis      TEXT,
  target_word_count INTEGER,
  word_count    INTEGER NOT NULL DEFAULT 0,
  settings_json TEXT NOT NULL DEFAULT '{}',
  created_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted       BOOLEAN NOT NULL DEFAULT 0,
  client_id     TEXT NOT NULL DEFAULT 'unknown',
  version       INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_msp_deleted ON manuscript_projects(deleted);
CREATE INDEX IF NOT EXISTS idx_msp_last_modified ON manuscript_projects(last_modified);
CREATE INDEX IF NOT EXISTS idx_msp_status ON manuscript_projects(status);

-- PARTS
CREATE TABLE IF NOT EXISTS manuscript_parts (
  id            TEXT PRIMARY KEY,
  project_id    TEXT NOT NULL REFERENCES manuscript_projects(id) ON DELETE CASCADE,
  title         TEXT NOT NULL,
  sort_order    REAL NOT NULL DEFAULT 0,
  synopsis      TEXT,
  word_count    INTEGER NOT NULL DEFAULT 0,
  created_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted       BOOLEAN NOT NULL DEFAULT 0,
  client_id     TEXT NOT NULL DEFAULT 'unknown',
  version       INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_mpt_project ON manuscript_parts(project_id, sort_order);
CREATE INDEX IF NOT EXISTS idx_mpt_deleted ON manuscript_parts(deleted);

-- CHAPTERS
CREATE TABLE IF NOT EXISTS manuscript_chapters (
  id            TEXT PRIMARY KEY,
  project_id    TEXT NOT NULL REFERENCES manuscript_projects(id) ON DELETE CASCADE,
  part_id       TEXT REFERENCES manuscript_parts(id) ON DELETE SET NULL,
  title         TEXT NOT NULL,
  sort_order    REAL NOT NULL DEFAULT 0,
  synopsis      TEXT,
  pov_character_id TEXT,
  word_count    INTEGER NOT NULL DEFAULT 0,
  status        TEXT NOT NULL DEFAULT 'draft'
                  CHECK(status IN ('outline','draft','revising','final')),
  created_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted       BOOLEAN NOT NULL DEFAULT 0,
  client_id     TEXT NOT NULL DEFAULT 'unknown',
  version       INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_mch_project ON manuscript_chapters(project_id, sort_order);
CREATE INDEX IF NOT EXISTS idx_mch_part ON manuscript_chapters(part_id);
CREATE INDEX IF NOT EXISTS idx_mch_deleted ON manuscript_chapters(deleted);

-- SCENES
CREATE TABLE IF NOT EXISTS manuscript_scenes (
  id            TEXT PRIMARY KEY,
  chapter_id    TEXT NOT NULL REFERENCES manuscript_chapters(id) ON DELETE CASCADE,
  project_id    TEXT NOT NULL REFERENCES manuscript_projects(id) ON DELETE CASCADE,
  title         TEXT NOT NULL DEFAULT 'Untitled Scene',
  sort_order    REAL NOT NULL DEFAULT 0,
  content_json  TEXT NOT NULL DEFAULT '{}',
  content_plain TEXT NOT NULL DEFAULT '',
  synopsis      TEXT,
  word_count    INTEGER NOT NULL DEFAULT 0,
  pov_character_id TEXT,
  status        TEXT NOT NULL DEFAULT 'draft'
                  CHECK(status IN ('outline','draft','revising','final')),
  created_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted       BOOLEAN NOT NULL DEFAULT 0,
  client_id     TEXT NOT NULL DEFAULT 'unknown',
  version       INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_msc_chapter ON manuscript_scenes(chapter_id, sort_order);
CREATE INDEX IF NOT EXISTS idx_msc_project ON manuscript_scenes(project_id);
CREATE INDEX IF NOT EXISTS idx_msc_deleted ON manuscript_scenes(deleted);

-- FTS5 for scene content search
CREATE VIRTUAL TABLE IF NOT EXISTS manuscript_scenes_fts
USING fts5(title, content_plain, synopsis, content='manuscript_scenes', content_rowid='rowid');

-- FTS triggers (ai=insert, au=update, ad=delete)
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

-- Sync triggers for manuscript_projects (4 triggers: create/update/delete/undelete)
CREATE TRIGGER manuscript_projects_sync_create
AFTER INSERT ON manuscript_projects BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('manuscript_projects',NEW.id,'create',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'title',NEW.title,'status',NEW.status,
                     'word_count',NEW.word_count,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER manuscript_projects_sync_update
AFTER UPDATE ON manuscript_projects
WHEN OLD.deleted = NEW.deleted AND (
     OLD.title IS NOT NEW.title OR OLD.subtitle IS NOT NEW.subtitle OR
     OLD.author IS NOT NEW.author OR OLD.genre IS NOT NEW.genre OR
     OLD.status IS NOT NEW.status OR OLD.synopsis IS NOT NEW.synopsis OR
     OLD.target_word_count IS NOT NEW.target_word_count OR
     OLD.word_count IS NOT NEW.word_count OR OLD.settings_json IS NOT NEW.settings_json OR
     OLD.last_modified IS NOT NEW.last_modified OR OLD.version IS NOT NEW.version)
BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('manuscript_projects',NEW.id,'update',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'title',NEW.title,'status',NEW.status,
                     'word_count',NEW.word_count,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER manuscript_projects_sync_delete
AFTER UPDATE ON manuscript_projects
WHEN OLD.deleted = 0 AND NEW.deleted = 1
BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('manuscript_projects',NEW.id,'delete',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'deleted',NEW.deleted,'last_modified',NEW.last_modified,
                     'version',NEW.version,'client_id',NEW.client_id));
END;

CREATE TRIGGER manuscript_projects_sync_undelete
AFTER UPDATE ON manuscript_projects
WHEN OLD.deleted = 1 AND NEW.deleted = 0
BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('manuscript_projects',NEW.id,'update',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'title',NEW.title,'status',NEW.status,
                     'word_count',NEW.word_count,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

-- Sync triggers for manuscript_parts
CREATE TRIGGER manuscript_parts_sync_create
AFTER INSERT ON manuscript_parts BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('manuscript_parts',NEW.id,'create',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'project_id',NEW.project_id,'title',NEW.title,
                     'sort_order',NEW.sort_order,'word_count',NEW.word_count,
                     'last_modified',NEW.last_modified,'deleted',NEW.deleted,
                     'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER manuscript_parts_sync_update
AFTER UPDATE ON manuscript_parts
WHEN OLD.deleted = NEW.deleted AND (
     OLD.title IS NOT NEW.title OR OLD.sort_order IS NOT NEW.sort_order OR
     OLD.synopsis IS NOT NEW.synopsis OR OLD.word_count IS NOT NEW.word_count OR
     OLD.last_modified IS NOT NEW.last_modified OR OLD.version IS NOT NEW.version)
BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('manuscript_parts',NEW.id,'update',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'project_id',NEW.project_id,'title',NEW.title,
                     'sort_order',NEW.sort_order,'word_count',NEW.word_count,
                     'last_modified',NEW.last_modified,'deleted',NEW.deleted,
                     'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER manuscript_parts_sync_delete
AFTER UPDATE ON manuscript_parts
WHEN OLD.deleted = 0 AND NEW.deleted = 1
BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('manuscript_parts',NEW.id,'delete',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'deleted',NEW.deleted,'last_modified',NEW.last_modified,
                     'version',NEW.version,'client_id',NEW.client_id));
END;

CREATE TRIGGER manuscript_parts_sync_undelete
AFTER UPDATE ON manuscript_parts
WHEN OLD.deleted = 1 AND NEW.deleted = 0
BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('manuscript_parts',NEW.id,'update',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'project_id',NEW.project_id,'title',NEW.title,
                     'sort_order',NEW.sort_order,'word_count',NEW.word_count,
                     'last_modified',NEW.last_modified,'deleted',NEW.deleted,
                     'client_id',NEW.client_id,'version',NEW.version));
END;

-- Sync triggers for manuscript_chapters
CREATE TRIGGER manuscript_chapters_sync_create
AFTER INSERT ON manuscript_chapters BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('manuscript_chapters',NEW.id,'create',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'project_id',NEW.project_id,'part_id',NEW.part_id,
                     'title',NEW.title,'sort_order',NEW.sort_order,'word_count',NEW.word_count,
                     'status',NEW.status,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER manuscript_chapters_sync_update
AFTER UPDATE ON manuscript_chapters
WHEN OLD.deleted = NEW.deleted AND (
     OLD.title IS NOT NEW.title OR OLD.part_id IS NOT NEW.part_id OR
     OLD.sort_order IS NOT NEW.sort_order OR OLD.synopsis IS NOT NEW.synopsis OR
     OLD.pov_character_id IS NOT NEW.pov_character_id OR
     OLD.word_count IS NOT NEW.word_count OR OLD.status IS NOT NEW.status OR
     OLD.last_modified IS NOT NEW.last_modified OR OLD.version IS NOT NEW.version)
BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('manuscript_chapters',NEW.id,'update',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'project_id',NEW.project_id,'part_id',NEW.part_id,
                     'title',NEW.title,'sort_order',NEW.sort_order,'word_count',NEW.word_count,
                     'status',NEW.status,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER manuscript_chapters_sync_delete
AFTER UPDATE ON manuscript_chapters
WHEN OLD.deleted = 0 AND NEW.deleted = 1
BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('manuscript_chapters',NEW.id,'delete',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'deleted',NEW.deleted,'last_modified',NEW.last_modified,
                     'version',NEW.version,'client_id',NEW.client_id));
END;

CREATE TRIGGER manuscript_chapters_sync_undelete
AFTER UPDATE ON manuscript_chapters
WHEN OLD.deleted = 1 AND NEW.deleted = 0
BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('manuscript_chapters',NEW.id,'update',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'project_id',NEW.project_id,'part_id',NEW.part_id,
                     'title',NEW.title,'sort_order',NEW.sort_order,'word_count',NEW.word_count,
                     'status',NEW.status,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

-- Sync triggers for manuscript_scenes
CREATE TRIGGER manuscript_scenes_sync_create
AFTER INSERT ON manuscript_scenes BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('manuscript_scenes',NEW.id,'create',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'chapter_id',NEW.chapter_id,'project_id',NEW.project_id,
                     'title',NEW.title,'sort_order',NEW.sort_order,'word_count',NEW.word_count,
                     'status',NEW.status,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER manuscript_scenes_sync_update
AFTER UPDATE ON manuscript_scenes
WHEN OLD.deleted = NEW.deleted AND (
     OLD.title IS NOT NEW.title OR OLD.chapter_id IS NOT NEW.chapter_id OR
     OLD.sort_order IS NOT NEW.sort_order OR OLD.content_json IS NOT NEW.content_json OR
     OLD.content_plain IS NOT NEW.content_plain OR OLD.synopsis IS NOT NEW.synopsis OR
     OLD.pov_character_id IS NOT NEW.pov_character_id OR
     OLD.word_count IS NOT NEW.word_count OR OLD.status IS NOT NEW.status OR
     OLD.last_modified IS NOT NEW.last_modified OR OLD.version IS NOT NEW.version)
BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('manuscript_scenes',NEW.id,'update',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'chapter_id',NEW.chapter_id,'project_id',NEW.project_id,
                     'title',NEW.title,'sort_order',NEW.sort_order,'word_count',NEW.word_count,
                     'status',NEW.status,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

CREATE TRIGGER manuscript_scenes_sync_delete
AFTER UPDATE ON manuscript_scenes
WHEN OLD.deleted = 0 AND NEW.deleted = 1
BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('manuscript_scenes',NEW.id,'delete',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'deleted',NEW.deleted,'last_modified',NEW.last_modified,
                     'version',NEW.version,'client_id',NEW.client_id));
END;

CREATE TRIGGER manuscript_scenes_sync_undelete
AFTER UPDATE ON manuscript_scenes
WHEN OLD.deleted = 1 AND NEW.deleted = 0
BEGIN
  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
  VALUES('manuscript_scenes',NEW.id,'update',NEW.last_modified,NEW.client_id,NEW.version,
         json_object('id',NEW.id,'chapter_id',NEW.chapter_id,'project_id',NEW.project_id,
                     'title',NEW.title,'sort_order',NEW.sort_order,'word_count',NEW.word_count,
                     'status',NEW.status,'last_modified',NEW.last_modified,
                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
END;

UPDATE db_schema_version
   SET version = 41
 WHERE schema_name = 'rag_char_chat_schema'
   AND version < 41;
"""
```

Also add the PostgreSQL variant `_MIGRATION_SQL_V40_TO_V41_POSTGRES` (same SQL but use `TIMESTAMP` instead of `DATETIME`).

**Step 2: Add migration method**

Add after `_migrate_from_v39_to_v40` (after line ~5156). Follow the exact method signature:

```python
def _migrate_from_v40_to_v41(self, conn: sqlite3.Connection) -> None:
    """Migrate schema from V40 to V41 (manuscript management tables)."""
    logger.info(f"Migrating '{self._SCHEMA_NAME}' schema from V40 to V41 for DB: {self.db_path_str}...")
    try:
        conn.executescript(self._MIGRATION_SQL_V40_TO_V41)
        final_version = self._get_db_version(conn)
        if final_version != 41:
            raise SchemaError(
                f"[{self._SCHEMA_NAME}] Migration V40->V41 failed version check. Expected 41, got: {final_version}"
            )
        logger.info(f"[{self._SCHEMA_NAME}] Migration to V41 completed.")
    except sqlite3.Error as e:
        logger.error(f"[{self._SCHEMA_NAME}] Migration V40->V41 failed: {e}", exc_info=True)
        raise SchemaError(f"Migration V40->V41 failed for '{self._SCHEMA_NAME}': {e}") from e
    except SchemaError:
        raise
    except _CHACHA_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"[{self._SCHEMA_NAME}] Unexpected error during migration V40->V41: {e}", exc_info=True)
        raise SchemaError(f"Unexpected error migrating to V41 for '{self._SCHEMA_NAME}': {e}") from e
```

**Step 3: Wire migration into initialization chain**

Add after line ~6111 (the V39→V40 wiring):

```python
if target_version >= 41 and current_db_version == 40:
    self._migrate_from_v40_to_v41(conn)
    current_db_version = self._get_db_version(conn)
```

**Step 4: Update schema version constant**

Change line 534:
```python
_CURRENT_SCHEMA_VERSION = 41  # Schema v41 adds manuscript management tables
```

**Step 5: Run migration test**

Run: `python -c "from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB; db = CharactersRAGDB(':memory:', 'test'); print('Migration OK, version:', db._get_db_version(db._get_connection()))"`

Expected: `Migration OK, version: 41`

**Step 6: Commit**
```bash
git add tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py
git commit -m "feat(db): add manuscript management tables (migration V40→V41)"
```

---

## Task 2: ManuscriptDB Helper — CRUD Operations

**Files:**
- Create: `tldw_Server_API/app/core/DB_Management/ManuscriptDB.py`
- Test: `tldw_Server_API/tests/Writing/test_manuscript_db.py`

**Step 1: Write unit tests for project CRUD**

```python
# tldw_Server_API/tests/Writing/test_manuscript_db.py
"""Unit tests for ManuscriptDB helper."""
import pytest
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.ManuscriptDB import ManuscriptDBHelper


@pytest.fixture()
def mdb(tmp_path):
    db = CharactersRAGDB(str(tmp_path / "test.db"), client_id="test_client")
    return ManuscriptDBHelper(db)


class TestProjectCRUD:
    def test_create_project(self, mdb):
        pid = mdb.create_project(title="My Novel", author="Alice")
        assert pid is not None
        proj = mdb.get_project(pid)
        assert proj["title"] == "My Novel"
        assert proj["author"] == "Alice"
        assert proj["status"] == "draft"
        assert proj["word_count"] == 0
        assert proj["version"] == 1
        assert proj["deleted"] is False

    def test_list_projects(self, mdb):
        mdb.create_project(title="Novel A")
        mdb.create_project(title="Novel B")
        projects, total = mdb.list_projects()
        assert total == 2
        assert {p["title"] for p in projects} == {"Novel A", "Novel B"}

    def test_update_project(self, mdb):
        pid = mdb.create_project(title="Draft")
        mdb.update_project(pid, {"title": "Final Title", "status": "writing"}, expected_version=1)
        proj = mdb.get_project(pid)
        assert proj["title"] == "Final Title"
        assert proj["status"] == "writing"
        assert proj["version"] == 2

    def test_update_project_version_conflict(self, mdb):
        pid = mdb.create_project(title="Draft")
        with pytest.raises(Exception):  # ConflictError
            mdb.update_project(pid, {"title": "X"}, expected_version=99)

    def test_soft_delete_project(self, mdb):
        pid = mdb.create_project(title="To Delete")
        mdb.soft_delete_project(pid, expected_version=1)
        assert mdb.get_project(pid) is None
        projects, total = mdb.list_projects()
        assert total == 0


class TestHierarchyCRUD:
    def test_create_part(self, mdb):
        pid = mdb.create_project(title="Novel")
        part_id = mdb.create_part(project_id=pid, title="Part One")
        part = mdb.get_part(part_id)
        assert part["title"] == "Part One"
        assert part["project_id"] == pid

    def test_create_chapter(self, mdb):
        pid = mdb.create_project(title="Novel")
        part_id = mdb.create_part(project_id=pid, title="Part One")
        ch_id = mdb.create_chapter(project_id=pid, title="Chapter 1", part_id=part_id)
        ch = mdb.get_chapter(ch_id)
        assert ch["title"] == "Chapter 1"
        assert ch["part_id"] == part_id

    def test_create_scene(self, mdb):
        pid = mdb.create_project(title="Novel")
        ch_id = mdb.create_chapter(project_id=pid, title="Chapter 1")
        scene_id = mdb.create_scene(
            chapter_id=ch_id, project_id=pid,
            title="Opening", content_json='{"type":"doc"}',
            content_plain="It was a dark and stormy night."
        )
        scene = mdb.get_scene(scene_id)
        assert scene["title"] == "Opening"
        assert scene["word_count"] == 7
        assert scene["content_plain"] == "It was a dark and stormy night."

    def test_word_count_propagation(self, mdb):
        pid = mdb.create_project(title="Novel")
        part_id = mdb.create_part(project_id=pid, title="Part 1")
        ch_id = mdb.create_chapter(project_id=pid, title="Ch 1", part_id=part_id)
        mdb.create_scene(chapter_id=ch_id, project_id=pid, title="S1",
                         content_json="{}", content_plain="one two three")
        mdb.create_scene(chapter_id=ch_id, project_id=pid, title="S2",
                         content_json="{}", content_plain="four five")
        ch = mdb.get_chapter(ch_id)
        assert ch["word_count"] == 5
        part = mdb.get_part(part_id)
        assert part["word_count"] == 5
        proj = mdb.get_project(pid)
        assert proj["word_count"] == 5

    def test_get_structure(self, mdb):
        pid = mdb.create_project(title="Novel")
        part_id = mdb.create_part(project_id=pid, title="Part 1")
        ch_id = mdb.create_chapter(project_id=pid, title="Ch 1", part_id=part_id)
        mdb.create_scene(chapter_id=ch_id, project_id=pid, title="Scene A",
                         content_json="{}", content_plain="hello world")
        structure = mdb.get_project_structure(pid)
        assert len(structure["parts"]) == 1
        assert len(structure["parts"][0]["chapters"]) == 1
        assert len(structure["parts"][0]["chapters"][0]["scenes"]) == 1

    def test_fts_search(self, mdb):
        pid = mdb.create_project(title="Novel")
        ch_id = mdb.create_chapter(project_id=pid, title="Chapter 1")
        mdb.create_scene(chapter_id=ch_id, project_id=pid, title="Storm",
                         content_json="{}", content_plain="The thunderstorm raged across the valley")
        mdb.create_scene(chapter_id=ch_id, project_id=pid, title="Calm",
                         content_json="{}", content_plain="The morning was peaceful and still")
        results = mdb.search_scenes(pid, "thunderstorm")
        assert len(results) == 1
        assert results[0]["title"] == "Storm"

    def test_reorder(self, mdb):
        pid = mdb.create_project(title="Novel")
        ch_id = mdb.create_chapter(project_id=pid, title="Chapter 1")
        s1 = mdb.create_scene(chapter_id=ch_id, project_id=pid, title="S1",
                              content_json="{}", content_plain="")
        s2 = mdb.create_scene(chapter_id=ch_id, project_id=pid, title="S2",
                              content_json="{}", content_plain="")
        mdb.reorder_items("scenes", [
            {"id": s2, "sort_order": 1.0},
            {"id": s1, "sort_order": 2.0},
        ])
        scenes = mdb.list_scenes(ch_id)
        assert scenes[0]["id"] == s2
        assert scenes[1]["id"] == s1
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tldw_Server_API/tests/Writing/test_manuscript_db.py -v`
Expected: FAIL — `ManuscriptDB` module doesn't exist yet

**Step 3: Implement ManuscriptDBHelper**

Create `tldw_Server_API/app/core/DB_Management/ManuscriptDB.py`:

```python
"""Manuscript management CRUD helper.

Delegates to CharactersRAGDB for connection management, transactions,
and optimistic locking infrastructure.
"""
from __future__ import annotations

import json
import uuid
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    ConflictError,
)


def _word_count(text: str) -> int:
    """Count words in plain text."""
    return len(text.split()) if text and text.strip() else 0


def _new_id() -> str:
    return str(uuid.uuid4())


class ManuscriptDBHelper:
    """CRUD operations for manuscript_* tables.

    Receives a CharactersRAGDB instance and uses its transaction/connection
    infrastructure. All methods are synchronous (matching CharactersRAGDB).
    """

    def __init__(self, db: CharactersRAGDB) -> None:
        self.db = db

    # ── Projects ────────────────────────────────────────────

    def create_project(
        self,
        title: str,
        *,
        subtitle: str | None = None,
        author: str | None = None,
        genre: str | None = None,
        status: str = "draft",
        synopsis: str | None = None,
        target_word_count: int | None = None,
        settings: dict[str, Any] | None = None,
        project_id: str | None = None,
    ) -> str:
        pid = project_id or _new_id()
        now = self.db._get_current_utc_timestamp_iso()
        with self.db.transaction() as conn:
            conn.execute(
                """INSERT INTO manuscript_projects
                   (id, title, subtitle, author, genre, status, synopsis,
                    target_word_count, settings_json, created_at, last_modified, client_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (pid, title, subtitle, author, genre, status, synopsis,
                 target_word_count, json.dumps(settings or {}), now, now, self.db.client_id),
            )
        return pid

    def get_project(self, project_id: str) -> dict[str, Any] | None:
        with self.db.transaction() as conn:
            cur = conn.execute(
                "SELECT * FROM manuscript_projects WHERE id = ? AND deleted = 0",
                (project_id,),
            )
            row = cur.fetchone()
        if not row:
            return None
        d = dict(row)
        d["settings"] = json.loads(d.pop("settings_json", "{}"))
        d["deleted"] = bool(d["deleted"])
        return d

    def list_projects(
        self, *, status_filter: str | None = None, limit: int = 100, offset: int = 0
    ) -> tuple[list[dict[str, Any]], int]:
        where = "deleted = 0"
        params: list[Any] = []
        if status_filter:
            where += " AND status = ?"
            params.append(status_filter)
        with self.db.transaction() as conn:
            count_cur = conn.execute(f"SELECT COUNT(*) FROM manuscript_projects WHERE {where}", params)
            total = count_cur.fetchone()[0]
            cur = conn.execute(
                f"SELECT * FROM manuscript_projects WHERE {where} ORDER BY last_modified DESC LIMIT ? OFFSET ?",
                [*params, limit, offset],
            )
            rows = [dict(r) for r in cur.fetchall()]
        for r in rows:
            r["settings"] = json.loads(r.pop("settings_json", "{}"))
            r["deleted"] = bool(r["deleted"])
        return rows, total

    def update_project(
        self, project_id: str, updates: dict[str, Any], expected_version: int
    ) -> None:
        now = self.db._get_current_utc_timestamp_iso()
        if "settings" in updates:
            updates["settings_json"] = json.dumps(updates.pop("settings"))
        sets = ", ".join(f"{k} = ?" for k in updates)
        vals = list(updates.values())
        with self.db.transaction() as conn:
            cur = conn.execute(
                f"""UPDATE manuscript_projects
                    SET {sets}, last_modified = ?, version = ?, client_id = ?
                    WHERE id = ? AND version = ? AND deleted = 0""",
                [*vals, now, expected_version + 1, self.db.client_id,
                 project_id, expected_version],
            )
            if cur.rowcount == 0:
                raise ConflictError(
                    f"Project {project_id} version conflict or not found",
                    entity="manuscript_projects", entity_id=project_id,
                )

    def soft_delete_project(self, project_id: str, expected_version: int) -> None:
        now = self.db._get_current_utc_timestamp_iso()
        with self.db.transaction() as conn:
            cur = conn.execute(
                """UPDATE manuscript_projects SET deleted = 1, last_modified = ?,
                   version = ?, client_id = ?
                   WHERE id = ? AND version = ? AND deleted = 0""",
                (now, expected_version + 1, self.db.client_id,
                 project_id, expected_version),
            )
            if cur.rowcount == 0:
                raise ConflictError(
                    f"Project {project_id} version conflict or not found",
                    entity="manuscript_projects", entity_id=project_id,
                )

    # ── Parts ───────────────────────────────────────────────

    def create_part(self, project_id: str, title: str, *, sort_order: float = 0,
                    synopsis: str | None = None, part_id: str | None = None) -> str:
        pid = part_id or _new_id()
        now = self.db._get_current_utc_timestamp_iso()
        with self.db.transaction() as conn:
            conn.execute(
                """INSERT INTO manuscript_parts
                   (id, project_id, title, sort_order, synopsis, created_at, last_modified, client_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (pid, project_id, title, sort_order, synopsis, now, now, self.db.client_id),
            )
        return pid

    def get_part(self, part_id: str) -> dict[str, Any] | None:
        with self.db.transaction() as conn:
            cur = conn.execute("SELECT * FROM manuscript_parts WHERE id = ? AND deleted = 0", (part_id,))
            row = cur.fetchone()
        return dict(row) if row else None

    def list_parts(self, project_id: str) -> list[dict[str, Any]]:
        with self.db.transaction() as conn:
            cur = conn.execute(
                "SELECT * FROM manuscript_parts WHERE project_id = ? AND deleted = 0 ORDER BY sort_order",
                (project_id,),
            )
            return [dict(r) for r in cur.fetchall()]

    def update_part(self, part_id: str, updates: dict[str, Any], expected_version: int) -> None:
        now = self.db._get_current_utc_timestamp_iso()
        sets = ", ".join(f"{k} = ?" for k in updates)
        vals = list(updates.values())
        with self.db.transaction() as conn:
            cur = conn.execute(
                f"""UPDATE manuscript_parts SET {sets}, last_modified = ?, version = ?, client_id = ?
                    WHERE id = ? AND version = ? AND deleted = 0""",
                [*vals, now, expected_version + 1, self.db.client_id, part_id, expected_version],
            )
            if cur.rowcount == 0:
                raise ConflictError(f"Part {part_id} conflict", entity="manuscript_parts", entity_id=part_id)

    def soft_delete_part(self, part_id: str, expected_version: int) -> None:
        now = self.db._get_current_utc_timestamp_iso()
        with self.db.transaction() as conn:
            cur = conn.execute(
                """UPDATE manuscript_parts SET deleted = 1, last_modified = ?, version = ?, client_id = ?
                   WHERE id = ? AND version = ? AND deleted = 0""",
                (now, expected_version + 1, self.db.client_id, part_id, expected_version),
            )
            if cur.rowcount == 0:
                raise ConflictError(f"Part {part_id} conflict", entity="manuscript_parts", entity_id=part_id)

    # ── Chapters ────────────────────────────────────────────

    def create_chapter(self, project_id: str, title: str, *, part_id: str | None = None,
                       sort_order: float = 0, synopsis: str | None = None,
                       status: str = "draft", chapter_id: str | None = None) -> str:
        cid = chapter_id or _new_id()
        now = self.db._get_current_utc_timestamp_iso()
        with self.db.transaction() as conn:
            conn.execute(
                """INSERT INTO manuscript_chapters
                   (id, project_id, part_id, title, sort_order, synopsis, status,
                    created_at, last_modified, client_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (cid, project_id, part_id, title, sort_order, synopsis, status,
                 now, now, self.db.client_id),
            )
        return cid

    def get_chapter(self, chapter_id: str) -> dict[str, Any] | None:
        with self.db.transaction() as conn:
            cur = conn.execute("SELECT * FROM manuscript_chapters WHERE id = ? AND deleted = 0", (chapter_id,))
            row = cur.fetchone()
        return dict(row) if row else None

    def list_chapters(self, project_id: str, *, part_id: str | None = None) -> list[dict[str, Any]]:
        where = "project_id = ? AND deleted = 0"
        params: list[Any] = [project_id]
        if part_id is not None:
            where += " AND part_id = ?"
            params.append(part_id)
        with self.db.transaction() as conn:
            cur = conn.execute(
                f"SELECT * FROM manuscript_chapters WHERE {where} ORDER BY sort_order", params,
            )
            return [dict(r) for r in cur.fetchall()]

    def update_chapter(self, chapter_id: str, updates: dict[str, Any], expected_version: int) -> None:
        now = self.db._get_current_utc_timestamp_iso()
        sets = ", ".join(f"{k} = ?" for k in updates)
        vals = list(updates.values())
        with self.db.transaction() as conn:
            cur = conn.execute(
                f"""UPDATE manuscript_chapters SET {sets}, last_modified = ?, version = ?, client_id = ?
                    WHERE id = ? AND version = ? AND deleted = 0""",
                [*vals, now, expected_version + 1, self.db.client_id, chapter_id, expected_version],
            )
            if cur.rowcount == 0:
                raise ConflictError(f"Chapter {chapter_id} conflict", entity="manuscript_chapters", entity_id=chapter_id)

    def soft_delete_chapter(self, chapter_id: str, expected_version: int) -> None:
        now = self.db._get_current_utc_timestamp_iso()
        with self.db.transaction() as conn:
            cur = conn.execute(
                """UPDATE manuscript_chapters SET deleted = 1, last_modified = ?, version = ?, client_id = ?
                   WHERE id = ? AND version = ? AND deleted = 0""",
                (now, expected_version + 1, self.db.client_id, chapter_id, expected_version),
            )
            if cur.rowcount == 0:
                raise ConflictError(f"Chapter {chapter_id} conflict", entity="manuscript_chapters", entity_id=chapter_id)

    # ── Scenes ──────────────────────────────────────────────

    def create_scene(
        self, chapter_id: str, project_id: str, *, title: str = "Untitled Scene",
        content_json: str = "{}", content_plain: str = "", synopsis: str | None = None,
        sort_order: float = 0, status: str = "draft", scene_id: str | None = None,
    ) -> str:
        sid = scene_id or _new_id()
        wc = _word_count(content_plain)
        now = self.db._get_current_utc_timestamp_iso()
        with self.db.transaction() as conn:
            conn.execute(
                """INSERT INTO manuscript_scenes
                   (id, chapter_id, project_id, title, sort_order, content_json,
                    content_plain, synopsis, word_count, status,
                    created_at, last_modified, client_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (sid, chapter_id, project_id, title, sort_order, content_json,
                 content_plain, synopsis, wc, status, now, now, self.db.client_id),
            )
            self._propagate_word_counts(conn, chapter_id, project_id)
        return sid

    def get_scene(self, scene_id: str) -> dict[str, Any] | None:
        with self.db.transaction() as conn:
            cur = conn.execute("SELECT * FROM manuscript_scenes WHERE id = ? AND deleted = 0", (scene_id,))
            row = cur.fetchone()
        if not row:
            return None
        d = dict(row)
        d["deleted"] = bool(d["deleted"])
        return d

    def list_scenes(self, chapter_id: str) -> list[dict[str, Any]]:
        with self.db.transaction() as conn:
            cur = conn.execute(
                "SELECT * FROM manuscript_scenes WHERE chapter_id = ? AND deleted = 0 ORDER BY sort_order",
                (chapter_id,),
            )
            return [dict(r) for r in cur.fetchall()]

    def update_scene(self, scene_id: str, updates: dict[str, Any], expected_version: int) -> None:
        now = self.db._get_current_utc_timestamp_iso()
        if "content_plain" in updates:
            updates["word_count"] = _word_count(updates["content_plain"])
        sets = ", ".join(f"{k} = ?" for k in updates)
        vals = list(updates.values())
        with self.db.transaction() as conn:
            # Get scene's chapter/project for word count propagation
            cur = conn.execute(
                "SELECT chapter_id, project_id FROM manuscript_scenes WHERE id = ? AND deleted = 0",
                (scene_id,),
            )
            row = cur.fetchone()
            if not row:
                raise ConflictError(f"Scene {scene_id} not found", entity="manuscript_scenes", entity_id=scene_id)
            chapter_id, project_id = row["chapter_id"], row["project_id"]

            cur = conn.execute(
                f"""UPDATE manuscript_scenes SET {sets}, last_modified = ?, version = ?, client_id = ?
                    WHERE id = ? AND version = ? AND deleted = 0""",
                [*vals, now, expected_version + 1, self.db.client_id, scene_id, expected_version],
            )
            if cur.rowcount == 0:
                raise ConflictError(f"Scene {scene_id} conflict", entity="manuscript_scenes", entity_id=scene_id)

            if "content_plain" in updates:
                self._propagate_word_counts(conn, chapter_id, project_id)

    def soft_delete_scene(self, scene_id: str, expected_version: int) -> None:
        now = self.db._get_current_utc_timestamp_iso()
        with self.db.transaction() as conn:
            cur = conn.execute(
                "SELECT chapter_id, project_id FROM manuscript_scenes WHERE id = ? AND deleted = 0",
                (scene_id,),
            )
            row = cur.fetchone()
            if not row:
                raise ConflictError(f"Scene {scene_id} not found", entity="manuscript_scenes", entity_id=scene_id)
            chapter_id, project_id = row["chapter_id"], row["project_id"]

            cur = conn.execute(
                """UPDATE manuscript_scenes SET deleted = 1, last_modified = ?, version = ?, client_id = ?
                   WHERE id = ? AND version = ? AND deleted = 0""",
                (now, expected_version + 1, self.db.client_id, scene_id, expected_version),
            )
            if cur.rowcount == 0:
                raise ConflictError(f"Scene {scene_id} conflict", entity="manuscript_scenes", entity_id=scene_id)
            self._propagate_word_counts(conn, chapter_id, project_id)

    # ── Word Count Propagation ──────────────────────────────

    def _propagate_word_counts(self, conn, chapter_id: str, project_id: str) -> None:
        """Cascade word counts from scenes → chapter → part → project."""
        # Chapter word count
        cur = conn.execute(
            "SELECT COALESCE(SUM(word_count), 0) FROM manuscript_scenes WHERE chapter_id = ? AND deleted = 0",
            (chapter_id,),
        )
        ch_wc = cur.fetchone()[0]
        conn.execute("UPDATE manuscript_chapters SET word_count = ? WHERE id = ?", (ch_wc, chapter_id))

        # Part word count (if chapter has a part)
        cur = conn.execute("SELECT part_id FROM manuscript_chapters WHERE id = ?", (chapter_id,))
        row = cur.fetchone()
        if row and row["part_id"]:
            part_id = row["part_id"]
            cur = conn.execute(
                """SELECT COALESCE(SUM(word_count), 0) FROM manuscript_chapters
                   WHERE part_id = ? AND deleted = 0""",
                (part_id,),
            )
            conn.execute("UPDATE manuscript_parts SET word_count = ? WHERE id = ?", (cur.fetchone()[0], part_id))

        # Project word count
        cur = conn.execute(
            "SELECT COALESCE(SUM(word_count), 0) FROM manuscript_chapters WHERE project_id = ? AND deleted = 0",
            (project_id,),
        )
        conn.execute("UPDATE manuscript_projects SET word_count = ? WHERE id = ?", (cur.fetchone()[0], project_id))

    # ── Structure ───────────────────────────────────────────

    def get_project_structure(self, project_id: str) -> dict[str, Any]:
        """Return full tree: parts → chapters → scenes with metadata."""
        with self.db.transaction() as conn:
            parts = conn.execute(
                "SELECT id, title, sort_order, word_count FROM manuscript_parts WHERE project_id = ? AND deleted = 0 ORDER BY sort_order",
                (project_id,),
            ).fetchall()

            chapters = conn.execute(
                "SELECT id, title, sort_order, word_count, status, part_id FROM manuscript_chapters WHERE project_id = ? AND deleted = 0 ORDER BY sort_order",
                (project_id,),
            ).fetchall()

            scenes = conn.execute(
                "SELECT id, title, sort_order, word_count, status, chapter_id FROM manuscript_scenes WHERE project_id = ? AND deleted = 0 ORDER BY sort_order",
                (project_id,),
            ).fetchall()

        # Build tree
        scenes_by_chapter: dict[str, list] = {}
        for s in scenes:
            scenes_by_chapter.setdefault(s["chapter_id"], []).append(dict(s))

        chapters_by_part: dict[str | None, list] = {}
        for c in chapters:
            ch = dict(c)
            ch["scenes"] = scenes_by_chapter.get(c["id"], [])
            chapters_by_part.setdefault(c["part_id"], []).append(ch)

        part_list = []
        for p in parts:
            pd = dict(p)
            pd["chapters"] = chapters_by_part.get(p["id"], [])
            part_list.append(pd)

        return {
            "project_id": project_id,
            "parts": part_list,
            "unassigned_chapters": chapters_by_part.get(None, []),
        }

    # ── Search ──────────────────────────────────────────────

    def search_scenes(self, project_id: str, query: str, *, limit: int = 20) -> list[dict[str, Any]]:
        """FTS5 search across scene titles, content, and synopses."""
        with self.db.transaction() as conn:
            cur = conn.execute(
                """SELECT s.id, s.title, s.chapter_id, s.word_count, s.status,
                          snippet(manuscript_scenes_fts, 1, '<mark>', '</mark>', '...', 32) as snippet
                   FROM manuscript_scenes_fts fts
                   JOIN manuscript_scenes s ON s.rowid = fts.rowid
                   WHERE manuscript_scenes_fts MATCH ? AND s.project_id = ? AND s.deleted = 0
                   LIMIT ?""",
                (query, project_id, limit),
            )
            return [dict(r) for r in cur.fetchall()]

    # ── Reorder ─────────────────────────────────────────────

    def reorder_items(self, entity_type: str, items: list[dict[str, Any]]) -> None:
        """Batch update sort_order for parts, chapters, or scenes."""
        table_map = {
            "parts": "manuscript_parts",
            "chapters": "manuscript_chapters",
            "scenes": "manuscript_scenes",
        }
        table = table_map.get(entity_type)
        if not table:
            raise ValueError(f"Invalid entity_type: {entity_type}")
        with self.db.transaction() as conn:
            for item in items:
                params: list[Any] = [item["sort_order"]]
                set_clause = "sort_order = ?"
                if "new_parent_id" in item and item["new_parent_id"] is not None:
                    if entity_type == "chapters":
                        set_clause += ", part_id = ?"
                        params.append(item["new_parent_id"])
                    elif entity_type == "scenes":
                        set_clause += ", chapter_id = ?"
                        params.append(item["new_parent_id"])
                params.append(item["id"])
                conn.execute(f"UPDATE {table} SET {set_clause} WHERE id = ?", params)
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tldw_Server_API/tests/Writing/test_manuscript_db.py -v`
Expected: All tests PASS

**Step 5: Commit**
```bash
git add tldw_Server_API/app/core/DB_Management/ManuscriptDB.py tldw_Server_API/tests/Writing/test_manuscript_db.py
git commit -m "feat(manuscripts): add ManuscriptDB helper with CRUD, word count propagation, FTS search"
```

---

## Task 3: Pydantic Schemas for Manuscripts

**Files:**
- Create: `tldw_Server_API/app/api/v1/schemas/writing_manuscript_schemas.py`

**Step 1: Create schema file**

Follow the conventions in `writing_schemas.py` (lines 1-42):

```python
"""Pydantic schemas for manuscript management endpoints."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


# ── Project ─────────────────────────────────────────────

class ManuscriptProjectCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=500, description="Project title")
    subtitle: str | None = Field(None, max_length=500)
    author: str | None = Field(None, max_length=255)
    genre: str | None = Field(None, max_length=100)
    status: Literal["draft", "outlining", "writing", "revising", "complete", "archived"] = "draft"
    synopsis: str | None = None
    target_word_count: int | None = Field(None, ge=0)
    settings: dict[str, Any] = Field(default_factory=dict)
    id: str | None = Field(None, description="Optional client-provided UUID")


class ManuscriptProjectUpdate(BaseModel):
    title: str | None = Field(None, min_length=1, max_length=500)
    subtitle: str | None = None
    author: str | None = None
    genre: str | None = None
    status: Literal["draft", "outlining", "writing", "revising", "complete", "archived"] | None = None
    synopsis: str | None = None
    target_word_count: int | None = None
    settings: dict[str, Any] | None = None


class ManuscriptProjectResponse(BaseModel):
    id: str
    title: str
    subtitle: str | None = None
    author: str | None = None
    genre: str | None = None
    status: str
    synopsis: str | None = None
    target_word_count: int | None = None
    word_count: int
    settings: dict[str, Any]
    created_at: datetime
    last_modified: datetime
    deleted: bool
    client_id: str
    version: int


class ManuscriptProjectListResponse(BaseModel):
    projects: list[ManuscriptProjectResponse]
    total: int


# ── Part ────────────────────────────────────────────────

class ManuscriptPartCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    sort_order: float = 0
    synopsis: str | None = None
    id: str | None = None


class ManuscriptPartUpdate(BaseModel):
    title: str | None = Field(None, min_length=1, max_length=500)
    sort_order: float | None = None
    synopsis: str | None = None


class ManuscriptPartResponse(BaseModel):
    id: str
    project_id: str
    title: str
    sort_order: float
    synopsis: str | None = None
    word_count: int
    created_at: datetime
    last_modified: datetime
    deleted: bool
    client_id: str
    version: int


# ── Chapter ─────────────────────────────────────────────

class ManuscriptChapterCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    part_id: str | None = None
    sort_order: float = 0
    synopsis: str | None = None
    status: Literal["outline", "draft", "revising", "final"] = "draft"
    id: str | None = None


class ManuscriptChapterUpdate(BaseModel):
    title: str | None = Field(None, min_length=1, max_length=500)
    part_id: str | None = None
    sort_order: float | None = None
    synopsis: str | None = None
    status: Literal["outline", "draft", "revising", "final"] | None = None


class ManuscriptChapterResponse(BaseModel):
    id: str
    project_id: str
    part_id: str | None = None
    title: str
    sort_order: float
    synopsis: str | None = None
    word_count: int
    status: str
    created_at: datetime
    last_modified: datetime
    deleted: bool
    client_id: str
    version: int


# ── Scene ───────────────────────────────────────────────

class ManuscriptSceneCreate(BaseModel):
    title: str = Field("Untitled Scene", min_length=1, max_length=500)
    content: dict[str, Any] = Field(default_factory=dict, description="TipTap JSON document")
    content_plain: str = Field("", description="Plain text extraction for search/word count")
    synopsis: str | None = None
    sort_order: float = 0
    status: Literal["outline", "draft", "revising", "final"] = "draft"
    id: str | None = None


class ManuscriptSceneUpdate(BaseModel):
    title: str | None = Field(None, min_length=1, max_length=500)
    content: dict[str, Any] | None = None
    content_plain: str | None = None
    synopsis: str | None = None
    sort_order: float | None = None
    status: Literal["outline", "draft", "revising", "final"] | None = None


class ManuscriptSceneResponse(BaseModel):
    id: str
    chapter_id: str
    project_id: str
    title: str
    sort_order: float
    content_json: str
    content_plain: str
    synopsis: str | None = None
    word_count: int
    status: str
    created_at: datetime
    last_modified: datetime
    deleted: bool
    client_id: str
    version: int


# ── Structure (tree view) ──────────────────────────────

class SceneSummary(BaseModel):
    id: str
    title: str
    sort_order: float
    word_count: int
    status: str


class ChapterSummary(BaseModel):
    id: str
    title: str
    sort_order: float
    word_count: int
    status: str
    part_id: str | None = None
    scenes: list[SceneSummary] = Field(default_factory=list)


class PartSummary(BaseModel):
    id: str
    title: str
    sort_order: float
    word_count: int
    chapters: list[ChapterSummary] = Field(default_factory=list)


class ManuscriptStructureResponse(BaseModel):
    project_id: str
    parts: list[PartSummary]
    unassigned_chapters: list[ChapterSummary] = Field(default_factory=list)


# ── Reorder ─────────────────────────────────────────────

class ReorderItem(BaseModel):
    id: str
    sort_order: float
    new_parent_id: str | None = None


class ReorderRequest(BaseModel):
    entity_type: Literal["parts", "chapters", "scenes"]
    items: list[ReorderItem]


# ── Search ──────────────────────────────────────────────

class ManuscriptSearchResult(BaseModel):
    id: str
    title: str
    chapter_id: str
    word_count: int
    status: str
    snippet: str


class ManuscriptSearchResponse(BaseModel):
    query: str
    results: list[ManuscriptSearchResult]
```

**Step 2: Commit**
```bash
git add tldw_Server_API/app/api/v1/schemas/writing_manuscript_schemas.py
git commit -m "feat(manuscripts): add Pydantic schemas for manuscript endpoints"
```

---

## Task 4: Manuscript API Endpoints

**Files:**
- Create: `tldw_Server_API/app/api/v1/endpoints/writing_manuscripts.py`
- Modify: `tldw_Server_API/app/main.py` (router registration)

**Step 1: Create endpoint file**

Follow the exact pattern from `writing.py` (lines 1-92 for setup, lines 1046-1141 for CRUD patterns):

```python
"""Manuscript management API endpoints."""
from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_rate_limiter_dep, get_request_user, rbac_rate_limit
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.schemas.writing_manuscript_schemas import (
    ManuscriptChapterCreate,
    ManuscriptChapterResponse,
    ManuscriptChapterUpdate,
    ManuscriptPartCreate,
    ManuscriptPartResponse,
    ManuscriptPartUpdate,
    ManuscriptProjectCreate,
    ManuscriptProjectListResponse,
    ManuscriptProjectResponse,
    ManuscriptProjectUpdate,
    ManuscriptSceneCreate,
    ManuscriptSceneResponse,
    ManuscriptSceneUpdate,
    ManuscriptSearchResponse,
    ManuscriptStructureResponse,
    ReorderRequest,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, ConflictError
from tldw_Server_API.app.core.DB_Management.ManuscriptDB import ManuscriptDBHelper

router = APIRouter()

_MANUSCRIPT_NONCRITICAL = (
    ConflictError, ValueError, KeyError, TypeError, RuntimeError,
    json.JSONDecodeError, AttributeError,
)


def _get_mdb(db: CharactersRAGDB) -> ManuscriptDBHelper:
    return ManuscriptDBHelper(db)


def _handle_conflict(exc: ConflictError) -> None:
    raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))


def _handle_errors(exc: Exception, entity: str) -> None:
    if isinstance(exc, ConflictError):
        _handle_conflict(exc)
    logger.error(f"Manuscript {entity} error: {exc}", exc_info=True)
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error with {entity}")


# ── Projects ────────────────────────────────────────────

@router.get("/projects", response_model=ManuscriptProjectListResponse, summary="List projects", tags=["manuscripts"])
async def list_projects(
    status_filter: str | None = Query(None, alias="status"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.list")),
) -> ManuscriptProjectListResponse:
    try:
        mdb = _get_mdb(db)
        projects, total = mdb.list_projects(status_filter=status_filter, limit=limit, offset=offset)
        return ManuscriptProjectListResponse(
            projects=[ManuscriptProjectResponse(**p) for p in projects],
            total=total,
        )
    except _MANUSCRIPT_NONCRITICAL as exc:
        _handle_errors(exc, "projects")


@router.post("/projects", response_model=ManuscriptProjectResponse, status_code=status.HTTP_201_CREATED,
             summary="Create project", tags=["manuscripts"])
async def create_project(
    payload: ManuscriptProjectCreate,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.create")),
) -> ManuscriptProjectResponse:
    try:
        mdb = _get_mdb(db)
        pid = mdb.create_project(
            title=payload.title, subtitle=payload.subtitle, author=payload.author,
            genre=payload.genre, status=payload.status, synopsis=payload.synopsis,
            target_word_count=payload.target_word_count, settings=payload.settings,
            project_id=payload.id,
        )
        proj = mdb.get_project(pid)
        if not proj:
            raise HTTPException(status_code=500, detail="Created but not found")
        return ManuscriptProjectResponse(**proj)
    except _MANUSCRIPT_NONCRITICAL as exc:
        _handle_errors(exc, "project")


@router.get("/projects/{project_id}", response_model=ManuscriptProjectResponse,
            summary="Get project", tags=["manuscripts"])
async def get_project(
    project_id: str,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.get")),
) -> ManuscriptProjectResponse:
    try:
        mdb = _get_mdb(db)
        proj = mdb.get_project(project_id)
        if not proj:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
        return ManuscriptProjectResponse(**proj)
    except _MANUSCRIPT_NONCRITICAL as exc:
        _handle_errors(exc, "project")


@router.patch("/projects/{project_id}", response_model=ManuscriptProjectResponse,
              summary="Update project", tags=["manuscripts"])
async def update_project(
    project_id: str,
    payload: ManuscriptProjectUpdate,
    expected_version: int = Header(..., description="Optimistic locking version"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.update")),
) -> ManuscriptProjectResponse:
    try:
        mdb = _get_mdb(db)
        updates = payload.model_dump(exclude_none=True)
        if not updates:
            raise HTTPException(status_code=400, detail="No fields to update")
        mdb.update_project(project_id, updates, expected_version)
        proj = mdb.get_project(project_id)
        if not proj:
            raise HTTPException(status_code=404, detail="Project not found after update")
        return ManuscriptProjectResponse(**proj)
    except _MANUSCRIPT_NONCRITICAL as exc:
        _handle_errors(exc, "project")


@router.delete("/projects/{project_id}", status_code=status.HTTP_204_NO_CONTENT,
               summary="Delete project", tags=["manuscripts"])
async def delete_project(
    project_id: str,
    expected_version: int = Header(...),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.delete")),
) -> None:
    try:
        mdb = _get_mdb(db)
        mdb.soft_delete_project(project_id, expected_version)
    except _MANUSCRIPT_NONCRITICAL as exc:
        _handle_errors(exc, "project")


@router.get("/projects/{project_id}/structure", response_model=ManuscriptStructureResponse,
            summary="Get full project structure tree", tags=["manuscripts"])
async def get_structure(
    project_id: str,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.get")),
) -> ManuscriptStructureResponse:
    try:
        mdb = _get_mdb(db)
        structure = mdb.get_project_structure(project_id)
        return ManuscriptStructureResponse(**structure)
    except _MANUSCRIPT_NONCRITICAL as exc:
        _handle_errors(exc, "project structure")


@router.post("/projects/{project_id}/reorder", status_code=status.HTTP_204_NO_CONTENT,
             summary="Batch reorder items", tags=["manuscripts"])
async def reorder_items(
    project_id: str,
    payload: ReorderRequest,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.update")),
) -> None:
    try:
        mdb = _get_mdb(db)
        mdb.reorder_items(payload.entity_type, [item.model_dump() for item in payload.items])
    except _MANUSCRIPT_NONCRITICAL as exc:
        _handle_errors(exc, "reorder")


@router.get("/projects/{project_id}/search", response_model=ManuscriptSearchResponse,
            summary="FTS search across scenes", tags=["manuscripts"])
async def search_scenes(
    project_id: str,
    q: str = Query(..., min_length=1),
    limit: int = Query(20, ge=1, le=100),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.get")),
) -> ManuscriptSearchResponse:
    try:
        mdb = _get_mdb(db)
        results = mdb.search_scenes(project_id, q, limit=limit)
        return ManuscriptSearchResponse(query=q, results=results)
    except _MANUSCRIPT_NONCRITICAL as exc:
        _handle_errors(exc, "search")


# ── Parts ───────────────────────────────────────────────
# (Follow same pattern as projects — omitted for brevity, implement create/list/get/update/delete)

@router.post("/projects/{project_id}/parts", response_model=ManuscriptPartResponse,
             status_code=status.HTTP_201_CREATED, summary="Create part", tags=["manuscripts"])
async def create_part(
    project_id: str, payload: ManuscriptPartCreate,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.create")),
) -> ManuscriptPartResponse:
    try:
        mdb = _get_mdb(db)
        pid = mdb.create_part(project_id, payload.title, sort_order=payload.sort_order,
                              synopsis=payload.synopsis, part_id=payload.id)
        part = mdb.get_part(pid)
        if not part:
            raise HTTPException(status_code=500, detail="Created but not found")
        return ManuscriptPartResponse(**part)
    except _MANUSCRIPT_NONCRITICAL as exc:
        _handle_errors(exc, "part")


# ── Chapters ────────────────────────────────────────────

@router.post("/projects/{project_id}/chapters", response_model=ManuscriptChapterResponse,
             status_code=status.HTTP_201_CREATED, summary="Create chapter", tags=["manuscripts"])
async def create_chapter(
    project_id: str, payload: ManuscriptChapterCreate,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.create")),
) -> ManuscriptChapterResponse:
    try:
        mdb = _get_mdb(db)
        cid = mdb.create_chapter(project_id, payload.title, part_id=payload.part_id,
                                 sort_order=payload.sort_order, synopsis=payload.synopsis,
                                 status=payload.status, chapter_id=payload.id)
        ch = mdb.get_chapter(cid)
        if not ch:
            raise HTTPException(status_code=500, detail="Created but not found")
        return ManuscriptChapterResponse(**ch)
    except _MANUSCRIPT_NONCRITICAL as exc:
        _handle_errors(exc, "chapter")


# ── Scenes ──────────────────────────────────────────────

@router.post("/chapters/{chapter_id}/scenes", response_model=ManuscriptSceneResponse,
             status_code=status.HTTP_201_CREATED, summary="Create scene", tags=["manuscripts"])
async def create_scene(
    chapter_id: str, payload: ManuscriptSceneCreate,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.create")),
) -> ManuscriptSceneResponse:
    try:
        mdb = _get_mdb(db)
        # Get project_id from chapter
        ch = mdb.get_chapter(chapter_id)
        if not ch:
            raise HTTPException(status_code=404, detail="Chapter not found")
        sid = mdb.create_scene(
            chapter_id=chapter_id, project_id=ch["project_id"],
            title=payload.title, content_json=json.dumps(payload.content),
            content_plain=payload.content_plain, synopsis=payload.synopsis,
            sort_order=payload.sort_order, status=payload.status, scene_id=payload.id,
        )
        scene = mdb.get_scene(sid)
        if not scene:
            raise HTTPException(status_code=500, detail="Created but not found")
        return ManuscriptSceneResponse(**scene)
    except _MANUSCRIPT_NONCRITICAL as exc:
        _handle_errors(exc, "scene")


@router.get("/scenes/{scene_id}", response_model=ManuscriptSceneResponse,
            summary="Get scene", tags=["manuscripts"])
async def get_scene(
    scene_id: str,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.get")),
) -> ManuscriptSceneResponse:
    try:
        mdb = _get_mdb(db)
        scene = mdb.get_scene(scene_id)
        if not scene:
            raise HTTPException(status_code=404, detail="Scene not found")
        return ManuscriptSceneResponse(**scene)
    except _MANUSCRIPT_NONCRITICAL as exc:
        _handle_errors(exc, "scene")


@router.patch("/scenes/{scene_id}", response_model=ManuscriptSceneResponse,
              summary="Update scene", tags=["manuscripts"])
async def update_scene(
    scene_id: str, payload: ManuscriptSceneUpdate,
    expected_version: int = Header(...),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.update")),
) -> ManuscriptSceneResponse:
    try:
        mdb = _get_mdb(db)
        updates = payload.model_dump(exclude_none=True)
        if "content" in updates:
            updates["content_json"] = json.dumps(updates.pop("content"))
        if not updates:
            raise HTTPException(status_code=400, detail="No fields to update")
        mdb.update_scene(scene_id, updates, expected_version)
        scene = mdb.get_scene(scene_id)
        if not scene:
            raise HTTPException(status_code=404, detail="Scene not found after update")
        return ManuscriptSceneResponse(**scene)
    except _MANUSCRIPT_NONCRITICAL as exc:
        _handle_errors(exc, "scene")
```

**Step 2: Register router in main.py**

Find the writing router registration (around line 7129-7131) and add after it:

```python
from tldw_Server_API.app.api.v1.endpoints.writing_manuscripts import router as manuscripts_router

if "manuscripts_router" in locals() and manuscripts_router is not None:
    include_router_idempotent(
        app, manuscripts_router, prefix=f"{API_V1_PREFIX}/writing/manuscripts",
        tags=["manuscripts"], default_stable=True
    )
```

**Step 3: Write integration test**

Create `tldw_Server_API/tests/Writing/test_manuscript_endpoint_integration.py`:

```python
"""Integration tests for manuscript endpoints."""
import importlib
import pytest
from fastapi.testclient import TestClient
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.core.AuthNZ.models import User


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db = CharactersRAGDB(str(tmp_path / "test.db"), client_id="test")
    async def override_user():
        return User(id=1, username="tester", email="t@e.com", is_active=True, is_admin=True)
    def override_db():
        return db
    monkeypatch.setenv("MINIMAL_TEST_APP", "1")
    monkeypatch.setenv("ROUTES_DISABLE", "media,audio")
    monkeypatch.setenv("SKIP_AUDIO_ROUTERS_IN_TESTS", "1")
    from tldw_Server_API.app import main as app_main
    importlib.reload(app_main)
    app = app_main.app
    app.dependency_overrides[get_request_user] = override_user
    app.dependency_overrides[get_chacha_db_for_user] = override_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_full_manuscript_crud(client):
    # Create project
    r = client.post("/api/v1/writing/manuscripts/projects", json={"title": "Test Novel", "author": "Alice"})
    assert r.status_code == 201
    proj = r.json()
    pid = proj["id"]
    assert proj["title"] == "Test Novel"

    # Create part
    r = client.post(f"/api/v1/writing/manuscripts/projects/{pid}/parts", json={"title": "Part One"})
    assert r.status_code == 201
    part_id = r.json()["id"]

    # Create chapter
    r = client.post(f"/api/v1/writing/manuscripts/projects/{pid}/chapters",
                    json={"title": "Chapter 1", "part_id": part_id})
    assert r.status_code == 201
    ch_id = r.json()["id"]

    # Create scene
    r = client.post(f"/api/v1/writing/manuscripts/chapters/{ch_id}/scenes",
                    json={"title": "Opening", "content": {"type": "doc"}, "content_plain": "hello world"})
    assert r.status_code == 201
    scene = r.json()
    assert scene["word_count"] == 2

    # Get structure
    r = client.get(f"/api/v1/writing/manuscripts/projects/{pid}/structure")
    assert r.status_code == 200
    structure = r.json()
    assert len(structure["parts"]) == 1
    assert len(structure["parts"][0]["chapters"]) == 1

    # Update scene
    r = client.patch(f"/api/v1/writing/manuscripts/scenes/{scene['id']}",
                     json={"content_plain": "one two three four five"},
                     headers={"expected-version": str(scene["version"])})
    assert r.status_code == 200
    assert r.json()["word_count"] == 5

    # Verify word count propagation
    r = client.get(f"/api/v1/writing/manuscripts/projects/{pid}")
    assert r.json()["word_count"] == 5

    # Search
    r = client.get(f"/api/v1/writing/manuscripts/projects/{pid}/search", params={"q": "three"})
    assert r.status_code == 200
    assert len(r.json()["results"]) >= 1
```

**Step 4: Run integration tests**

Run: `python -m pytest tldw_Server_API/tests/Writing/test_manuscript_endpoint_integration.py -v`
Expected: All tests PASS

**Step 5: Commit**
```bash
git add tldw_Server_API/app/api/v1/endpoints/writing_manuscripts.py \
       tldw_Server_API/app/api/v1/schemas/writing_manuscript_schemas.py \
       tldw_Server_API/app/main.py \
       tldw_Server_API/tests/Writing/test_manuscript_endpoint_integration.py
git commit -m "feat(manuscripts): add REST API endpoints for manuscript management"
```

---

## Task 5: Frontend — Install TipTap Dependencies

**Files:**
- Modify: `apps/packages/ui/package.json` (peerDependencies)
- Modify: `apps/tldw-frontend/package.json` (dependencies)
- Modify: `apps/extension/package.json` (dependencies)

**Step 1: Install TipTap packages**

```bash
cd /Users/macbook-dev/Documents/GitHub/tldw_server2/apps/tldw-frontend
pnpm add @tiptap/react @tiptap/starter-kit @tiptap/extension-placeholder @tiptap/extension-character-count @tiptap/pm
```

```bash
cd /Users/macbook-dev/Documents/GitHub/tldw_server2/apps/extension
pnpm add @tiptap/react @tiptap/starter-kit @tiptap/extension-placeholder @tiptap/extension-character-count @tiptap/pm
```

Add to `apps/packages/ui/package.json` peerDependencies:
```json
"@tiptap/react": "^2.0.0",
"@tiptap/starter-kit": "^2.0.0",
"@tiptap/extension-placeholder": "^2.0.0",
"@tiptap/extension-character-count": "^2.0.0",
"@tiptap/pm": "^2.0.0"
```

**Step 2: Verify installation**

```bash
cd /Users/macbook-dev/Documents/GitHub/tldw_server2/apps/tldw-frontend
pnpm exec tsc --noEmit 2>&1 | head -20
```
Expected: No TipTap-related type errors

**Step 3: Commit**
```bash
git add apps/packages/ui/package.json apps/tldw-frontend/package.json apps/extension/package.json pnpm-lock.yaml
git commit -m "chore: add TipTap editor dependencies"
```

---

## Task 6: Frontend — Zustand Store Extensions

**Files:**
- Modify: `apps/packages/ui/src/store/writing-playground.tsx`

**Step 1: Read current store file**

Read: `apps/packages/ui/src/store/writing-playground.tsx` (16 lines)

**Step 2: Extend store**

Add new state fields after the existing ones:

```typescript
// Add to the state type
activeProjectId: string | null
setActiveProjectId: (id: string | null) => void
activeNodeId: string | null
setActiveNodeId: (id: string | null) => void
editorMode: "plain" | "tiptap"
setEditorMode: (mode: "plain" | "tiptap") => void
focusMode: boolean
setFocusMode: (enabled: boolean) => void
```

Add to the store creation:

```typescript
activeProjectId: null,
setActiveProjectId: (id) => set({ activeProjectId: id }),
activeNodeId: null,
setActiveNodeId: (id) => set({ activeNodeId: id }),
editorMode: "plain",
setEditorMode: (mode) => set({ editorMode: mode }),
focusMode: false,
setFocusMode: (enabled) => set({ focusMode: enabled }),
```

**Step 3: Run existing tests**

Run: `cd apps && pnpm test -- --run --reporter=verbose 2>&1 | grep -E "PASS|FAIL|writing"`
Expected: Existing writing tests still pass

**Step 4: Commit**
```bash
git add apps/packages/ui/src/store/writing-playground.tsx
git commit -m "feat(writing): extend Zustand store with manuscript navigation state"
```

---

## Task 7: Frontend — Manuscript API Service Layer

**Files:**
- Modify: `apps/packages/ui/src/services/writing-playground.ts`

**Step 1: Add manuscript API client functions**

Add at the end of the existing file, following the `createResourceClient` pattern:

```typescript
// ── Manuscript API ─────────────────────────────────────

const manuscriptsProjectsClient = createResourceClient({
  basePath: "/api/v1/writing/manuscripts/projects" as AllowedPath,
  detailPath: (id) => `/api/v1/writing/manuscripts/projects/${encodeURIComponent(String(id))}` as AllowedPath,
})

export async function listManuscriptProjects(params?: { status?: string; limit?: number; offset?: number }) {
  return manuscriptsProjectsClient.list(params)
}

export async function getManuscriptProject(id: string) {
  return manuscriptsProjectsClient.get(id)
}

export async function createManuscriptProject(data: Record<string, unknown>) {
  return manuscriptsProjectsClient.create(data)
}

export async function updateManuscriptProject(id: string, data: Record<string, unknown>, version: number) {
  return manuscriptsProjectsClient.update(id, data, { headers: { "expected-version": String(version) } })
}

export async function deleteManuscriptProject(id: string, version: number) {
  return manuscriptsProjectsClient.delete(id, { headers: { "expected-version": String(version) } })
}

export async function getManuscriptStructure(projectId: string) {
  const client = createResourceClient({
    basePath: `/api/v1/writing/manuscripts/projects/${encodeURIComponent(projectId)}/structure` as AllowedPath,
  })
  return client.list()
}

export async function searchManuscriptScenes(projectId: string, query: string, limit = 20) {
  const client = createResourceClient({
    basePath: `/api/v1/writing/manuscripts/projects/${encodeURIComponent(projectId)}/search` as AllowedPath,
  })
  return client.list({ q: query, limit })
}
```

**Step 2: Commit**
```bash
git add apps/packages/ui/src/services/writing-playground.ts
git commit -m "feat(writing): add manuscript API service layer"
```

---

## Task 8: Frontend — TipTap Editor Component

**Files:**
- Create: `apps/packages/ui/src/components/Option/WritingPlayground/WritingTipTapEditor.tsx`
- Create: `apps/packages/ui/src/components/Option/WritingPlayground/extensions/SceneBreakExtension.ts`
- Create: `apps/packages/ui/src/components/Option/WritingPlayground/writing-tiptap-utils.ts`

**Step 1: Create SceneBreak extension**

```typescript
// extensions/SceneBreakExtension.ts
import { Node } from "@tiptap/core"

export const SceneBreakExtension = Node.create({
  name: "sceneBreak",
  group: "block",
  parseHTML() {
    return [{ tag: "hr.scene-break" }]
  },
  renderHTML() {
    return ["hr", { class: "scene-break" }]
  },
  addInputRules() {
    return [
      {
        find: /^\*\*\*\s$/,
        handler: ({ state, range }) => {
          state.tr.replaceRangeWith(range.from, range.to, this.type.create())
        },
      },
    ]
  },
})
```

**Step 2: Create TipTap-to-plain conversion utilities**

```typescript
// writing-tiptap-utils.ts
import type { JSONContent } from "@tiptap/react"

export function tipTapJsonToPlainText(json: JSONContent): string {
  if (!json) return ""
  if (json.type === "text") return json.text || ""
  const children = json.content?.map(tipTapJsonToPlainText).join("") || ""
  if (json.type === "paragraph") return children + "\n"
  if (json.type === "sceneBreak") return "\n***\n"
  return children
}

export function plainTextToTipTapJson(text: string): JSONContent {
  const paragraphs = text.split("\n").map((line): JSONContent => {
    if (line.trim() === "***") {
      return { type: "sceneBreak" }
    }
    return {
      type: "paragraph",
      content: line ? [{ type: "text", text: line }] : [],
    }
  })
  return { type: "doc", content: paragraphs }
}
```

**Step 3: Create TipTap editor component**

```typescript
// WritingTipTapEditor.tsx
import { lazy, Suspense, useCallback, useEffect, useMemo } from "react"
import { EditorContent, useEditor, type JSONContent } from "@tiptap/react"
import StarterKit from "@tiptap/starter-kit"
import Placeholder from "@tiptap/extension-placeholder"
import CharacterCount from "@tiptap/extension-character-count"
import { SceneBreakExtension } from "./extensions/SceneBreakExtension"
import { tipTapJsonToPlainText } from "./writing-tiptap-utils"

type WritingTipTapEditorProps = {
  content: JSONContent | null
  onContentChange: (json: JSONContent, plainText: string) => void
  editable?: boolean
  placeholder?: string
  className?: string
  themeCss?: string
}

export function WritingTipTapEditor({
  content,
  onContentChange,
  editable = true,
  placeholder = "Start writing...",
  className,
  themeCss,
}: WritingTipTapEditorProps) {
  const extensions = useMemo(
    () => [
      StarterKit,
      SceneBreakExtension,
      Placeholder.configure({ placeholder }),
      CharacterCount,
    ],
    [placeholder],
  )

  const editor = useEditor({
    extensions,
    content: content || { type: "doc", content: [{ type: "paragraph" }] },
    editable,
    onUpdate: ({ editor }) => {
      const json = editor.getJSON()
      const plain = tipTapJsonToPlainText(json)
      onContentChange(json, plain)
    },
  })

  // Sync content from outside (e.g., when switching scenes)
  useEffect(() => {
    if (editor && content && !editor.isFocused) {
      const currentJson = JSON.stringify(editor.getJSON())
      const newJson = JSON.stringify(content)
      if (currentJson !== newJson) {
        editor.commands.setContent(content)
      }
    }
  }, [editor, content])

  if (!editor) return null

  return (
    <div className={className}>
      {themeCss && <style>{themeCss}</style>}
      <EditorContent editor={editor} className="prose max-w-none min-h-[400px] focus:outline-none" />
    </div>
  )
}
```

**Step 4: Commit**
```bash
git add apps/packages/ui/src/components/Option/WritingPlayground/WritingTipTapEditor.tsx \
       apps/packages/ui/src/components/Option/WritingPlayground/extensions/SceneBreakExtension.ts \
       apps/packages/ui/src/components/Option/WritingPlayground/writing-tiptap-utils.ts
git commit -m "feat(writing): add TipTap editor component with scene break extension"
```

---

## Task 9: Frontend — Manuscript Tree Panel

**Files:**
- Create: `apps/packages/ui/src/components/Option/WritingPlayground/ManuscriptTreePanel.tsx`

**Step 1: Create tree panel component**

Follow the @dnd-kit pattern from `AudiobookStudio/ChapterEditor/ChapterList.tsx` (lines 168-178):

```typescript
// ManuscriptTreePanel.tsx
import { useState } from "react"
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import { DragDropProvider } from "@dnd-kit/react"
import { useSortable } from "@dnd-kit/react/sortable"
import { closestCenter } from "@dnd-kit/collision"
import { Button, Input, Tree, Typography } from "antd"
import {
  BookOpen, ChevronRight, ChevronDown, FileText,
  FolderOpen, GripVertical, Plus, Layers,
} from "lucide-react"
import { useWritingPlaygroundStore } from "@/store/writing-playground"
import { getManuscriptStructure, createManuscriptProject } from "@/services/writing-playground"

type ManuscriptTreePanelProps = {
  isOnline: boolean
}

export function ManuscriptTreePanel({ isOnline }: ManuscriptTreePanelProps) {
  const { activeProjectId, setActiveProjectId, activeNodeId, setActiveNodeId } =
    useWritingPlaygroundStore()
  const queryClient = useQueryClient()

  const { data: structure, isLoading } = useQuery({
    queryKey: ["manuscript-structure", activeProjectId],
    queryFn: () => getManuscriptStructure(activeProjectId!),
    enabled: isOnline && !!activeProjectId,
    staleTime: 30_000,
  })

  if (!activeProjectId) {
    return (
      <div className="flex flex-col items-center justify-center gap-3 p-4 text-center">
        <BookOpen className="h-8 w-8 text-gray-400" />
        <Typography.Text type="secondary">No project selected</Typography.Text>
        <Button
          type="primary"
          icon={<Plus className="h-4 w-4" />}
          onClick={() => {
            /* Open create project modal */
          }}
        >
          New Project
        </Button>
      </div>
    )
  }

  if (isLoading) {
    return <div className="p-4 text-center text-gray-400">Loading...</div>
  }

  // Render tree using Ant Design Tree or custom implementation
  // This is a simplified version — full implementation adds @dnd-kit sortable
  const treeData = buildTreeData(structure)

  return (
    <div className="flex flex-col gap-1 p-2">
      <Tree
        treeData={treeData}
        selectedKeys={activeNodeId ? [activeNodeId] : []}
        onSelect={(keys) => setActiveNodeId(keys[0]?.toString() || null)}
        showIcon
        blockNode
      />
    </div>
  )
}

function buildTreeData(structure: any) {
  if (!structure) return []
  const parts = structure.parts?.map((part: any) => ({
    key: part.id,
    title: `${part.title} (${part.word_count} words)`,
    icon: <Layers className="h-4 w-4" />,
    children: part.chapters?.map((ch: any) => ({
      key: ch.id,
      title: `${ch.title} (${ch.word_count} words)`,
      icon: <FolderOpen className="h-4 w-4" />,
      children: ch.scenes?.map((s: any) => ({
        key: s.id,
        title: `${s.title} (${s.word_count} words)`,
        icon: <FileText className="h-4 w-4" />,
        isLeaf: true,
      })) || [],
    })) || [],
  })) || []

  const unassigned = structure.unassigned_chapters?.map((ch: any) => ({
    key: ch.id,
    title: `${ch.title} (${ch.word_count} words)`,
    icon: <FolderOpen className="h-4 w-4" />,
    children: ch.scenes?.map((s: any) => ({
      key: s.id,
      title: `${s.title} (${s.word_count} words)`,
      icon: <FileText className="h-4 w-4" />,
      isLeaf: true,
    })) || [],
  })) || []

  return [...parts, ...unassigned]
}
```

**Step 2: Commit**
```bash
git add apps/packages/ui/src/components/Option/WritingPlayground/ManuscriptTreePanel.tsx
git commit -m "feat(writing): add manuscript tree panel with project structure view"
```

---

## Task 10: Frontend — Wire TipTap + Manuscript Tree into WritingPlayground

**Files:**
- Modify: `apps/packages/ui/src/components/Option/WritingPlayground/index.tsx`
- Modify: `apps/packages/ui/src/components/Option/WritingPlayground/WritingPlaygroundShell.tsx`

This is the integration task. It wires the new components into the existing WritingPlayground without decomposing index.tsx (that's a separate, larger refactoring task).

**Step 1: Add editor mode toggle to toolbar**

In index.tsx, in the toolbar JSX section (around line 2107), add an editor mode segmented control:

```typescript
// Add import at top
import { useWritingPlaygroundStore } from "@/store/writing-playground"
// Use lazy import for TipTap
const WritingTipTapEditor = lazy(() => import("./WritingTipTapEditor").then(m => ({ default: m.WritingTipTapEditor })))

// In the toolbar area, add:
<Segmented
  value={editorMode}
  onChange={(v) => setEditorMode(v as "plain" | "tiptap")}
  options={[
    { label: "Plain", value: "plain" },
    { label: "Rich Text", value: "tiptap" },
  ]}
  size="small"
/>
```

**Step 2: Conditionally render TipTap or plain textarea**

In the editor panel area (around line 2167-2192), wrap the existing textarea in a condition:

```typescript
{editorMode === "tiptap" ? (
  <Suspense fallback={<div className="p-4">Loading editor...</div>}>
    <WritingTipTapEditor
      content={tipTapContent}
      onContentChange={(json, plain) => {
        setTipTapContent(json)
        setEditorText(plain) // keeps generation pipeline working
      }}
    />
  </Suspense>
) : (
  /* existing textarea JSX */
)}
```

**Step 3: Add library panel segmented control**

In the library drawer content (around line 1932), add a segmented control:

```typescript
const [libraryView, setLibraryView] = useState<"sessions" | "manuscript">("sessions")

// In the library drawer header:
<Segmented
  value={libraryView}
  onChange={(v) => setLibraryView(v as "sessions" | "manuscript")}
  options={[
    { label: "Sessions", value: "sessions" },
    { label: "Manuscript", value: "manuscript" },
  ]}
  block
  size="small"
/>

// Conditionally render:
{libraryView === "manuscript" ? (
  <ManuscriptTreePanel isOnline={isOnline} />
) : (
  /* existing session list JSX */
)}
```

**Step 4: Add focus mode to shell**

In `WritingPlaygroundShell.tsx`, add focus mode support:

```typescript
// Add prop
focusMode?: boolean

// In the layout, hide sidebars when focusMode is true
{!focusMode && /* library sidebar */}
{!focusMode && /* inspector sidebar */}
```

Add keyboard shortcut in index.tsx:

```typescript
useEffect(() => {
  const handler = (e: KeyboardEvent) => {
    if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === "f") {
      e.preventDefault()
      setFocusMode(!focusMode)
    }
    if (e.key === "Escape" && focusMode) {
      setFocusMode(false)
    }
  }
  window.addEventListener("keydown", handler)
  return () => window.removeEventListener("keydown", handler)
}, [focusMode, setFocusMode])
```

**Step 5: Run tests**

```bash
cd apps && pnpm test -- --run --reporter=verbose 2>&1 | grep -E "PASS|FAIL|writing"
```

**Step 6: Commit**
```bash
git add apps/packages/ui/src/components/Option/WritingPlayground/index.tsx \
       apps/packages/ui/src/components/Option/WritingPlayground/WritingPlaygroundShell.tsx
git commit -m "feat(writing): wire TipTap editor, manuscript tree, and focus mode into WritingPlayground"
```

---

## Verification Checklist

After all tasks are complete:

1. **Backend migration**: `python -c "from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB; db = CharactersRAGDB(':memory:', 'test')"` succeeds
2. **Backend unit tests**: `python -m pytest tldw_Server_API/tests/Writing/test_manuscript_db.py -v` all pass
3. **Backend integration tests**: `python -m pytest tldw_Server_API/tests/Writing/test_manuscript_endpoint_integration.py -v` all pass
4. **Frontend build**: `cd apps/tldw-frontend && pnpm build` succeeds
5. **Extension build**: `cd apps/extension && pnpm build` succeeds
6. **Frontend tests**: `cd apps && pnpm test -- --run` all pass
7. **API docs**: Visit `http://127.0.0.1:8000/docs` and verify `/api/v1/writing/manuscripts/` endpoints appear
8. **Manual smoke test**: Create project → add chapter → add scene → type in TipTap → verify word count → switch to manuscript tree → verify structure → toggle focus mode
