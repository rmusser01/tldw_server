# Writing Suite Phase 2: Knowledge — Characters, World Info, Plot, Research

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add character sheets, world info, plot tracking, scene-entity linking, and RAG-powered research with inline citations to the manuscript writing suite.

**Architecture:** Extends the Phase 1 ManuscriptDB helper with CRUD for 8 new tables (migration V41→V42). New REST endpoints follow the same pattern. Frontend adds two new inspector tabs (Characters/World, Research) and a TipTap CitationExtension for inline references. Research integration calls the existing `unified_rag_pipeline` with scene context.

**Tech Stack:** FastAPI, SQLite, Pydantic v2, TipTap Mark extension, React Query, Ant Design

---

## Task 1: DB Migration V41→V42 — Characters, World Info, Plot, Citations

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`

**Step 1: Add migration SQL**

Add `_MIGRATION_SQL_V41_TO_V42` and `_MIGRATION_SQL_V41_TO_V42_POSTGRES` class attributes after the V40→V41 migration SQL. Update `_CURRENT_SCHEMA_VERSION` from 41 to 42.

The SQLite migration SQL must create these tables:

```sql
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
  PRIMARY KEY (scene_id, character_id)
);

-- SCENE-WORLD_INFO LINKING
CREATE TABLE IF NOT EXISTS manuscript_scene_world_info (
  scene_id        TEXT NOT NULL REFERENCES manuscript_scenes(id) ON DELETE CASCADE,
  world_info_id   TEXT NOT NULL REFERENCES manuscript_world_info(id) ON DELETE CASCADE,
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
```

Add sync triggers (4 per table: create/update/delete/undelete) for: `manuscript_characters`, `manuscript_character_relationships`, `manuscript_world_info`, `manuscript_plot_lines`, `manuscript_plot_events`, `manuscript_plot_holes`, `manuscript_citations`. Follow the exact pattern from V41 (e.g., `manuscript_projects_sync_create`). The linking tables (`manuscript_scene_characters`, `manuscript_scene_world_info`) do NOT get sync triggers (simple join tables).

End with: `UPDATE db_schema_version SET version = 42 WHERE schema_name = 'rag_char_chat_schema' AND version < 42;`

Add `_MIGRATION_SQL_V41_TO_V42_POSTGRES` (same SQL but `TIMESTAMP` instead of `DATETIME`, `FALSE` instead of `0`).

**Step 2: Add migration method**

```python
def _migrate_from_v41_to_v42(self, conn: sqlite3.Connection) -> None:
    """Migrate schema from V41 to V42 (characters, world info, plot, citations)."""
    # Same pattern as _migrate_from_v40_to_v41
```

**Step 3: Wire migration**

```python
if target_version >= 42 and current_db_version == 41:
    self._migrate_from_v41_to_v42(conn)
    current_db_version = self._get_db_version(conn)
```

**Step 4: Update version constant**

```python
_CURRENT_SCHEMA_VERSION = 42  # Schema v42 adds characters, world info, plot, citations
```

**Step 5: Test migration**

Run: `python -c "from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB; db = CharactersRAGDB(':memory:', 'test'); conn = db.get_connection(); tables = [r[0] for r in conn.execute(\"SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'manuscript_%'\").fetchall()]; print('Tables:', len(tables)); ver = conn.execute(\"SELECT version FROM db_schema_version WHERE schema_name='rag_char_chat_schema'\").fetchone(); print('Version:', ver[0])"`

Expected: Tables: 13+ (4 from V41 + 9 new), Version: 42

**Step 6: Commit**

```bash
git add tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py
git commit -m "feat(db): add characters, world info, plot, citations tables (migration V41→V42)"
```

---

## Task 2: ManuscriptDB Helper — Characters & Relationships CRUD

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/ManuscriptDB.py`
- Create: `tldw_Server_API/tests/Writing/test_manuscript_characters_db.py`

**Step 1: Write tests**

```python
"""Unit tests for manuscript character CRUD."""
import pytest
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, ConflictError
from tldw_Server_API.app.core.DB_Management.ManuscriptDB import ManuscriptDBHelper


@pytest.fixture()
def mdb(tmp_path):
    db = CharactersRAGDB(str(tmp_path / "test.db"), client_id="test_client")
    return ManuscriptDBHelper(db)


class TestCharacterCRUD:
    def test_create_and_get(self, mdb):
        pid = mdb.create_project("Novel")
        cid = mdb.create_character(pid, "Alice", role="protagonist")
        ch = mdb.get_character(cid)
        assert ch["name"] == "Alice"
        assert ch["role"] == "protagonist"
        assert ch["project_id"] == pid

    def test_list_characters(self, mdb):
        pid = mdb.create_project("Novel")
        mdb.create_character(pid, "Alice", role="protagonist")
        mdb.create_character(pid, "Bob", role="antagonist")
        mdb.create_character(pid, "Charlie", role="supporting")
        chars = mdb.list_characters(pid)
        assert len(chars) == 3

    def test_list_characters_filter_by_role(self, mdb):
        pid = mdb.create_project("Novel")
        mdb.create_character(pid, "Alice", role="protagonist")
        mdb.create_character(pid, "Bob", role="supporting")
        chars = mdb.list_characters(pid, role_filter="protagonist")
        assert len(chars) == 1
        assert chars[0]["name"] == "Alice"

    def test_update_character(self, mdb):
        pid = mdb.create_project("Novel")
        cid = mdb.create_character(pid, "Alice")
        mdb.update_character(cid, {"name": "Alicia", "backstory": "Born in..."}, expected_version=1)
        ch = mdb.get_character(cid)
        assert ch["name"] == "Alicia"
        assert ch["backstory"] == "Born in..."
        assert ch["version"] == 2

    def test_soft_delete_character(self, mdb):
        pid = mdb.create_project("Novel")
        cid = mdb.create_character(pid, "Alice")
        mdb.soft_delete_character(cid, expected_version=1)
        assert mdb.get_character(cid) is None

    def test_custom_fields(self, mdb):
        pid = mdb.create_project("Novel")
        cid = mdb.create_character(pid, "Alice", custom_fields={"hair": "red", "magic": True})
        ch = mdb.get_character(cid)
        assert ch["custom_fields"]["hair"] == "red"


class TestCharacterRelationships:
    def test_create_and_list(self, mdb):
        pid = mdb.create_project("Novel")
        c1 = mdb.create_character(pid, "Alice")
        c2 = mdb.create_character(pid, "Bob")
        rel_id = mdb.create_relationship(pid, c1, c2, "sibling", description="Twins")
        rels = mdb.list_relationships(pid)
        assert len(rels) == 1
        assert rels[0]["relationship_type"] == "sibling"
        assert rels[0]["from_character_id"] == c1
        assert rels[0]["to_character_id"] == c2

    def test_delete_relationship(self, mdb):
        pid = mdb.create_project("Novel")
        c1 = mdb.create_character(pid, "Alice")
        c2 = mdb.create_character(pid, "Bob")
        rel_id = mdb.create_relationship(pid, c1, c2, "rival")
        mdb.soft_delete_relationship(rel_id, expected_version=1)
        rels = mdb.list_relationships(pid)
        assert len(rels) == 0


class TestSceneCharacterLinking:
    def test_link_and_list(self, mdb):
        pid = mdb.create_project("Novel")
        ch_id = mdb.create_chapter(pid, "Ch1")
        scene_id = mdb.create_scene(ch_id, pid, title="Opening", content_json="{}", content_plain="hello")
        char_id = mdb.create_character(pid, "Alice")
        mdb.link_scene_character(scene_id, char_id)
        linked = mdb.list_scene_characters(scene_id)
        assert len(linked) == 1
        assert linked[0]["character_id"] == char_id

    def test_unlink(self, mdb):
        pid = mdb.create_project("Novel")
        ch_id = mdb.create_chapter(pid, "Ch1")
        scene_id = mdb.create_scene(ch_id, pid, title="Opening", content_json="{}", content_plain="hello")
        char_id = mdb.create_character(pid, "Alice")
        mdb.link_scene_character(scene_id, char_id)
        mdb.unlink_scene_character(scene_id, char_id)
        assert len(mdb.list_scene_characters(scene_id)) == 0
```

**Step 2: Implement character CRUD in ManuscriptDB.py**

Add methods to `ManuscriptDBHelper`:

- `create_character(project_id, name, *, role='supporting', cast_group=None, full_name=None, age=None, gender=None, appearance=None, personality=None, backstory=None, motivation=None, arc_summary=None, notes=None, custom_fields=None, sort_order=0, character_id=None) -> str`
- `get_character(character_id) -> dict | None` — deserialize `custom_fields_json` to `custom_fields`
- `list_characters(project_id, *, role_filter=None, cast_group_filter=None) -> list[dict]`
- `update_character(character_id, updates, expected_version)`
- `soft_delete_character(character_id, expected_version)`
- `create_relationship(project_id, from_id, to_id, rel_type, *, description=None, bidirectional=True, rel_id=None) -> str`
- `list_relationships(project_id) -> list[dict]`
- `soft_delete_relationship(rel_id, expected_version)`
- `link_scene_character(scene_id, character_id, *, is_pov=False)` — INSERT OR IGNORE
- `unlink_scene_character(scene_id, character_id)` — DELETE
- `list_scene_characters(scene_id) -> list[dict]`

Follow the exact patterns from Phase 1 CRUD (transactions, timestamps, optimistic locking, `_uuid()`, `_now()`, `_client_id`).

**Step 3: Run tests**

Run: `python -m pytest tldw_Server_API/tests/Writing/test_manuscript_characters_db.py -v`
Expected: All PASS

**Step 4: Commit**

```bash
git add tldw_Server_API/app/core/DB_Management/ManuscriptDB.py tldw_Server_API/tests/Writing/test_manuscript_characters_db.py
git commit -m "feat(manuscripts): add character and relationship CRUD to ManuscriptDB"
```

---

## Task 3: ManuscriptDB Helper — World Info, Plot, Citations CRUD

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/ManuscriptDB.py`
- Create: `tldw_Server_API/tests/Writing/test_manuscript_world_plot_db.py`

**Step 1: Write tests**

```python
"""Unit tests for world info, plot tracking, and citations CRUD."""
import pytest
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, ConflictError
from tldw_Server_API.app.core.DB_Management.ManuscriptDB import ManuscriptDBHelper


@pytest.fixture()
def mdb(tmp_path):
    db = CharactersRAGDB(str(tmp_path / "test.db"), client_id="test_client")
    return ManuscriptDBHelper(db)


class TestWorldInfoCRUD:
    def test_create_and_get(self, mdb):
        pid = mdb.create_project("Novel")
        wid = mdb.create_world_info(pid, kind="location", name="Mordor", description="Dark land")
        wi = mdb.get_world_info(wid)
        assert wi["name"] == "Mordor"
        assert wi["kind"] == "location"

    def test_list_by_kind(self, mdb):
        pid = mdb.create_project("Novel")
        mdb.create_world_info(pid, kind="location", name="Mordor")
        mdb.create_world_info(pid, kind="item", name="Ring")
        mdb.create_world_info(pid, kind="location", name="Shire")
        items = mdb.list_world_info(pid, kind_filter="location")
        assert len(items) == 2

    def test_hierarchical(self, mdb):
        pid = mdb.create_project("Novel")
        parent = mdb.create_world_info(pid, kind="location", name="Middle Earth")
        child = mdb.create_world_info(pid, kind="location", name="Shire", parent_id=parent)
        wi = mdb.get_world_info(child)
        assert wi["parent_id"] == parent

    def test_properties_and_tags(self, mdb):
        pid = mdb.create_project("Novel")
        wid = mdb.create_world_info(pid, kind="item", name="Ring",
                                     properties={"power": "invisibility"}, tags=["artifact", "danger"])
        wi = mdb.get_world_info(wid)
        assert wi["properties"]["power"] == "invisibility"
        assert "artifact" in wi["tags"]

    def test_update_and_delete(self, mdb):
        pid = mdb.create_project("Novel")
        wid = mdb.create_world_info(pid, kind="faction", name="Elves")
        mdb.update_world_info(wid, {"description": "Ancient race"}, expected_version=1)
        assert mdb.get_world_info(wid)["description"] == "Ancient race"
        mdb.soft_delete_world_info(wid, expected_version=2)
        assert mdb.get_world_info(wid) is None

    def test_scene_linking(self, mdb):
        pid = mdb.create_project("Novel")
        ch_id = mdb.create_chapter(pid, "Ch1")
        scene_id = mdb.create_scene(ch_id, pid, title="S1", content_json="{}", content_plain="text")
        wid = mdb.create_world_info(pid, kind="location", name="Shire")
        mdb.link_scene_world_info(scene_id, wid)
        linked = mdb.list_scene_world_info(scene_id)
        assert len(linked) == 1
        mdb.unlink_scene_world_info(scene_id, wid)
        assert len(mdb.list_scene_world_info(scene_id)) == 0


class TestPlotLineCRUD:
    def test_create_and_list(self, mdb):
        pid = mdb.create_project("Novel")
        pl_id = mdb.create_plot_line(pid, "Main Quest", description="Destroy the ring")
        lines = mdb.list_plot_lines(pid)
        assert len(lines) == 1
        assert lines[0]["title"] == "Main Quest"

    def test_update_status(self, mdb):
        pid = mdb.create_project("Novel")
        pl_id = mdb.create_plot_line(pid, "Side Quest")
        mdb.update_plot_line(pl_id, {"status": "resolved"}, expected_version=1)
        pl = mdb.get_plot_line(pl_id)
        assert pl["status"] == "resolved"


class TestPlotEventCRUD:
    def test_create_and_list(self, mdb):
        pid = mdb.create_project("Novel")
        pl_id = mdb.create_plot_line(pid, "Main Quest")
        pe_id = mdb.create_plot_event(pid, pl_id, "Ring found", event_type="setup")
        events = mdb.list_plot_events(pl_id)
        assert len(events) == 1
        assert events[0]["title"] == "Ring found"
        assert events[0]["event_type"] == "setup"


class TestPlotHoleCRUD:
    def test_create_and_list(self, mdb):
        pid = mdb.create_project("Novel")
        ph_id = mdb.create_plot_hole(pid, "Timeline inconsistency", severity="high")
        holes = mdb.list_plot_holes(pid)
        assert len(holes) == 1
        assert holes[0]["severity"] == "high"

    def test_resolve(self, mdb):
        pid = mdb.create_project("Novel")
        ph_id = mdb.create_plot_hole(pid, "Plot gap")
        mdb.update_plot_hole(ph_id, {"status": "resolved", "resolution": "Added scene"}, expected_version=1)
        ph = mdb.get_plot_hole(ph_id)
        assert ph["status"] == "resolved"


class TestCitationCRUD:
    def test_create_and_list(self, mdb):
        pid = mdb.create_project("Novel")
        ch_id = mdb.create_chapter(pid, "Ch1")
        scene_id = mdb.create_scene(ch_id, pid, title="S1", content_json="{}", content_plain="text")
        cit_id = mdb.create_citation(pid, scene_id, source_type="media_db",
                                      source_title="Wikipedia", excerpt="Key fact")
        cits = mdb.list_citations(scene_id)
        assert len(cits) == 1
        assert cits[0]["source_title"] == "Wikipedia"

    def test_delete_citation(self, mdb):
        pid = mdb.create_project("Novel")
        ch_id = mdb.create_chapter(pid, "Ch1")
        scene_id = mdb.create_scene(ch_id, pid, title="S1", content_json="{}", content_plain="text")
        cit_id = mdb.create_citation(pid, scene_id, source_type="note", source_title="My notes")
        mdb.soft_delete_citation(cit_id, expected_version=1)
        assert len(mdb.list_citations(scene_id)) == 0
```

**Step 2: Implement CRUD methods in ManuscriptDB.py**

Add to `ManuscriptDBHelper`:

**World Info:**
- `create_world_info(project_id, kind, name, *, description=None, parent_id=None, properties=None, tags=None, sort_order=0, world_info_id=None) -> str` — serialize `properties` to `properties_json`, `tags` to `tags_json`
- `get_world_info(world_info_id) -> dict | None` — deserialize JSON fields
- `list_world_info(project_id, *, kind_filter=None) -> list[dict]`
- `update_world_info(world_info_id, updates, expected_version)` — handle `properties`→`properties_json`, `tags`→`tags_json` conversion
- `soft_delete_world_info(world_info_id, expected_version)`
- `link_scene_world_info(scene_id, world_info_id)` — INSERT OR IGNORE
- `unlink_scene_world_info(scene_id, world_info_id)` — DELETE
- `list_scene_world_info(scene_id) -> list[dict]`

**Plot Lines:**
- `create_plot_line(project_id, title, *, description=None, status='active', color=None, sort_order=0, plot_line_id=None) -> str`
- `get_plot_line(plot_line_id) -> dict | None`
- `list_plot_lines(project_id) -> list[dict]`
- `update_plot_line(plot_line_id, updates, expected_version)`
- `soft_delete_plot_line(plot_line_id, expected_version)`

**Plot Events:**
- `create_plot_event(project_id, plot_line_id, title, *, description=None, scene_id=None, chapter_id=None, event_type='plot', sort_order=0, event_id=None) -> str`
- `list_plot_events(plot_line_id) -> list[dict]`
- `update_plot_event(event_id, updates, expected_version)`
- `soft_delete_plot_event(event_id, expected_version)`

**Plot Holes:**
- `create_plot_hole(project_id, title, *, description=None, severity='medium', scene_id=None, chapter_id=None, plot_line_id=None, detected_by='manual', plot_hole_id=None) -> str`
- `get_plot_hole(plot_hole_id) -> dict | None`
- `list_plot_holes(project_id, *, status_filter=None) -> list[dict]`
- `update_plot_hole(plot_hole_id, updates, expected_version)`
- `soft_delete_plot_hole(plot_hole_id, expected_version)`

**Citations:**
- `create_citation(project_id, scene_id, source_type, *, source_id=None, source_title=None, excerpt=None, query_used=None, anchor_offset=None, citation_id=None) -> str`
- `list_citations(scene_id) -> list[dict]`
- `soft_delete_citation(citation_id, expected_version)`

**Step 3: Run tests**

Run: `python -m pytest tldw_Server_API/tests/Writing/test_manuscript_world_plot_db.py -v`
Expected: All PASS

**Step 4: Commit**

```bash
git add tldw_Server_API/app/core/DB_Management/ManuscriptDB.py tldw_Server_API/tests/Writing/test_manuscript_world_plot_db.py
git commit -m "feat(manuscripts): add world info, plot tracking, and citations CRUD"
```

---

## Task 4: Pydantic Schemas — Phase 2 Entities

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/writing_manuscript_schemas.py`

**Step 1: Add schemas**

Append to the existing schema file, following the same Create/Update/Response trio pattern:

**Character schemas:** `ManuscriptCharacterCreate` (name required, role Literal, cast_group, appearance, personality, backstory, motivation, arc_summary, notes, custom_fields dict), `ManuscriptCharacterUpdate` (all optional), `ManuscriptCharacterResponse` (includes project_id, custom_fields dict, timestamps, version)

**Relationship schemas:** `ManuscriptRelationshipCreate` (from_character_id, to_character_id, relationship_type required; description, bidirectional optional), `ManuscriptRelationshipResponse`

**World info schemas:** `ManuscriptWorldInfoCreate` (kind Literal, name required; description, parent_id, properties dict, tags list), `ManuscriptWorldInfoUpdate`, `ManuscriptWorldInfoResponse` (properties as dict, tags as list)

**Plot line schemas:** `ManuscriptPlotLineCreate/Update/Response`

**Plot event schemas:** `ManuscriptPlotEventCreate/Update/Response` (includes event_type Literal)

**Plot hole schemas:** `ManuscriptPlotHoleCreate/Update/Response` (includes severity Literal, detected_by Literal)

**Citation schemas:** `ManuscriptCitationCreate` (source_type, source_id, source_title, excerpt, query_used), `ManuscriptCitationResponse`

**Scene linking schemas:** `SceneCharacterLink` (character_id, is_pov), `SceneWorldInfoLink` (world_info_id)

**Research schemas:** `ManuscriptResearchRequest` (query required, top_k optional), `ManuscriptResearchResponse` (query, results list)

**Step 2: Commit**

```bash
git add tldw_Server_API/app/api/v1/schemas/writing_manuscript_schemas.py
git commit -m "feat(manuscripts): add Phase 2 Pydantic schemas (characters, world, plot, citations, research)"
```

---

## Task 5: API Endpoints — Characters, World Info, Plot, Citations, Research

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/writing_manuscripts.py`
- Create: `tldw_Server_API/tests/Writing/test_manuscript_phase2_integration.py`

**Step 1: Add endpoints to writing_manuscripts.py**

Append new endpoint sections following the existing CRUD pattern (router decorator, RBAC dep, `_get_helper(db)`, `model_dump(exclude_none=True)` for updates, `_handle_db_errors`):

**Characters (under `/projects/{project_id}/characters`):**
- `POST` — create (201)
- `GET` — list (optional query params: role, cast_group)
- `GET /characters/{character_id}` — get
- `PATCH /characters/{character_id}` — update (Header expected_version)
- `DELETE /characters/{character_id}` — soft delete (204)

**Relationships (under `/projects/{project_id}/characters/relationships`):**
- `POST` — create
- `GET` — list
- `DELETE /characters/relationships/{rel_id}` — soft delete

**World Info (under `/projects/{project_id}/world-info`):**
- `POST` — create
- `GET` — list (optional query: kind)
- `GET /world-info/{item_id}` — get
- `PATCH /world-info/{item_id}` — update (handle properties→properties_json, tags→tags_json)
- `DELETE /world-info/{item_id}` — soft delete

**Plot Lines (under `/projects/{project_id}/plot-lines`):**
- `POST` — create
- `GET` — list
- `PATCH /plot-lines/{plot_line_id}` — update
- `DELETE /plot-lines/{plot_line_id}` — soft delete

**Plot Events (under `/plot-lines/{plot_line_id}/events`):**
- `POST` — create (resolve project_id from plot_line)
- `GET` — list
- `PATCH /plot-events/{event_id}` — update
- `DELETE /plot-events/{event_id}` — soft delete

**Plot Holes (under `/projects/{project_id}/plot-holes`):**
- `POST` — create
- `GET` — list (optional query: status)
- `PATCH /plot-holes/{hole_id}` — update
- `DELETE /plot-holes/{hole_id}` — soft delete

**Scene Links:**
- `POST /scenes/{scene_id}/characters` — link character (body: `SceneCharacterLink`)
- `DELETE /scenes/{scene_id}/characters/{character_id}` — unlink
- `GET /scenes/{scene_id}/characters` — list linked characters
- `POST /scenes/{scene_id}/world-info` — link world info
- `DELETE /scenes/{scene_id}/world-info/{world_info_id}` — unlink
- `GET /scenes/{scene_id}/world-info` — list linked world info

**Citations:**
- `POST /scenes/{scene_id}/citations` — create
- `GET /scenes/{scene_id}/citations` — list
- `DELETE /citations/{citation_id}` — soft delete

**Research:**
- `POST /scenes/{scene_id}/research` — RAG search contextualized with scene content. Implementation:
  1. Fetch scene content_plain + synopsis from DB
  2. Fetch linked characters and world info for context
  3. Build enhanced query: `f"{scene_context}\n\nResearch query: {request.query}"`
  4. Call `/api/v1/rag/search` internally or import the RAG search function
  5. Return results

For the research endpoint, the simplest approach is to use `httpx` to call the existing RAG search endpoint internally, or import and call the search function directly. Check which approach the codebase prefers. Use internal function call if available, otherwise HTTP.

**Step 2: Write integration tests**

```python
def test_character_crud(client):
    # Create project → create character → list → update → delete
    ...

def test_world_info_crud(client):
    # Create project → create world info → list by kind → update → delete
    ...

def test_plot_tracking(client):
    # Create project → create plot line → create event → create hole → list all
    ...

def test_scene_character_linking(client):
    # Create project → chapter → scene → character → link → list → unlink
    ...

def test_citations(client):
    # Create project → chapter → scene → create citation → list → delete
    ...
```

**Step 3: Run tests**

Run: `python -m pytest tldw_Server_API/tests/Writing/test_manuscript_phase2_integration.py -v`

**Step 4: Commit**

```bash
git add tldw_Server_API/app/api/v1/endpoints/writing_manuscripts.py \
       tldw_Server_API/tests/Writing/test_manuscript_phase2_integration.py
git commit -m "feat(manuscripts): add Phase 2 endpoints (characters, world, plot, citations, research)"
```

---

## Task 6: Frontend — Extend Inspector Tabs & API Service

**Files:**
- Modify: `apps/packages/ui/src/components/Option/WritingPlayground/WritingPlayground.types.ts`
- Modify: `apps/packages/ui/src/services/writing-playground.ts`

**Step 1: Extend InspectorTabKey**

In `WritingPlayground.types.ts`, change:
```typescript
export type InspectorTabKey = "sampling" | "context" | "setup" | "inspect"
```
to:
```typescript
export type InspectorTabKey = "sampling" | "context" | "setup" | "inspect" | "characters" | "research"
```

**Step 2: Add Phase 2 API service functions**

Append to `apps/packages/ui/src/services/writing-playground.ts`:

```typescript
// ── Characters ──
export async function listManuscriptCharacters(projectId: string, params?: { role?: string }) { ... }
export async function createManuscriptCharacter(projectId: string, data: Record<string, unknown>) { ... }
export async function updateManuscriptCharacter(characterId: string, data: Record<string, unknown>, version: number) { ... }
export async function deleteManuscriptCharacter(characterId: string, version: number) { ... }
export async function listManuscriptRelationships(projectId: string) { ... }
export async function createManuscriptRelationship(projectId: string, data: Record<string, unknown>) { ... }

// ── World Info ──
export async function listManuscriptWorldInfo(projectId: string, params?: { kind?: string }) { ... }
export async function createManuscriptWorldInfo(projectId: string, data: Record<string, unknown>) { ... }
export async function updateManuscriptWorldInfo(itemId: string, data: Record<string, unknown>, version: number) { ... }
export async function deleteManuscriptWorldInfo(itemId: string, version: number) { ... }

// ── Plot Lines ──
export async function listManuscriptPlotLines(projectId: string) { ... }
export async function createManuscriptPlotLine(projectId: string, data: Record<string, unknown>) { ... }

// ── Plot Holes ──
export async function listManuscriptPlotHoles(projectId: string) { ... }

// ── Scene Links ──
export async function linkSceneCharacter(sceneId: string, characterId: string) { ... }
export async function unlinkSceneCharacter(sceneId: string, characterId: string) { ... }
export async function listSceneCharacters(sceneId: string) { ... }

// ── Citations ──
export async function listManuscriptCitations(sceneId: string) { ... }
export async function createManuscriptCitation(sceneId: string, data: Record<string, unknown>) { ... }

// ── Research ──
export async function searchManuscriptResearch(sceneId: string, query: string, topK?: number) { ... }
```

Use `bgRequest()` for endpoints that need `expected-version` headers, plain `fetch` wrapper for simple GETs.

**Step 3: Commit**

```bash
git add apps/packages/ui/src/components/Option/WritingPlayground/WritingPlayground.types.ts \
       apps/packages/ui/src/services/writing-playground.ts
git commit -m "feat(writing): extend InspectorTabKey and add Phase 2 API service functions"
```

---

## Task 7: Frontend — CharacterWorldTab Component

**Files:**
- Create: `apps/packages/ui/src/components/Option/WritingPlayground/CharacterWorldTab.tsx`

**Step 1: Create the component**

A compact inspector tab with a `[Characters] [World Info] [Plot]` segmented control:

```typescript
import { useState } from "react"
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import { Button, Empty, Input, List, Segmented, Select, Spin, Tag, Typography } from "antd"
import { Plus, Trash2, Users, Globe, GitBranch } from "lucide-react"
import { useWritingPlaygroundStore } from "@/store/writing-playground"
import {
  listManuscriptCharacters, createManuscriptCharacter, deleteManuscriptCharacter,
  listManuscriptWorldInfo, createManuscriptWorldInfo,
  listManuscriptPlotLines, createManuscriptPlotLine,
  listManuscriptPlotHoles,
} from "@/services/writing-playground"

type CharacterWorldTabProps = { isOnline: boolean }
type SubView = "characters" | "world" | "plot"
```

**Characters sub-view:** List characters with role badges, "Add" button that creates with a prompt for name, click to expand inline details (appearance, personality, backstory — readonly for now, editable in future).

**World Info sub-view:** List world info grouped by kind (locations, items, factions), filterable by kind via Select dropdown.

**Plot sub-view:** List plot lines with status badges + list plot holes with severity badges.

All data fetched via React Query with `["manuscript-characters", activeProjectId]` etc. query keys. Mutations invalidate the relevant query key.

Show `<Empty>` with "Select a project first" when `!activeProjectId`.

**Step 2: Commit**

```bash
git add apps/packages/ui/src/components/Option/WritingPlayground/CharacterWorldTab.tsx
git commit -m "feat(writing): add CharacterWorldTab inspector component"
```

---

## Task 8: Frontend — ResearchTab Component

**Files:**
- Create: `apps/packages/ui/src/components/Option/WritingPlayground/ResearchTab.tsx`

**Step 1: Create the component**

An inspector tab for RAG-powered research:

```typescript
import { useState } from "react"
import { useMutation } from "@tanstack/react-query"
import { Button, Card, Empty, Input, List, Spin, Typography } from "antd"
import { Search, BookOpen, Plus } from "lucide-react"
import { useWritingPlaygroundStore } from "@/store/writing-playground"
import { searchManuscriptResearch, createManuscriptCitation } from "@/services/writing-playground"
```

**Layout:**
1. Search input with "Search" button
2. Results list: compact cards showing `source_title`, snippet, source_type badge
3. Each result has a "Cite" button that creates a `manuscript_citations` record for the active scene

**Behavior:**
- Search calls `POST /manuscripts/scenes/{activeNodeId}/research` with the query
- If `activeNodeId` is null, show "Select a scene to search" message
- Results displayed as `List` with `List.Item` components
- "Cite" button calls `createManuscriptCitation(sceneId, { source_type, source_title, excerpt })`

**Step 2: Commit**

```bash
git add apps/packages/ui/src/components/Option/WritingPlayground/ResearchTab.tsx
git commit -m "feat(writing): add ResearchTab inspector component with RAG search"
```

---

## Task 9: Frontend — CitationExtension for TipTap

**Files:**
- Create: `apps/packages/ui/src/components/Option/WritingPlayground/extensions/CitationExtension.ts`
- Modify: `apps/packages/ui/src/components/Option/WritingPlayground/WritingTipTapEditor.tsx`

**Step 1: Create the extension**

A TipTap Mark (inline, not block) for citation references:

```typescript
import { Mark, mergeAttributes } from "@tiptap/core"

export const CitationExtension = Mark.create({
  name: "citation",

  addAttributes() {
    return {
      sourceId: { default: null },
      sourceTitle: { default: null },
      sourceType: { default: null },
    }
  },

  parseHTML() {
    return [{ tag: "span[data-citation]" }]
  },

  renderHTML({ HTMLAttributes }) {
    return [
      "span",
      mergeAttributes(HTMLAttributes, {
        "data-citation": "",
        class: "citation-mark",
        style: "background: rgba(59, 130, 246, 0.1); border-bottom: 1px dashed #3b82f6; cursor: pointer;",
        title: HTMLAttributes.sourceTitle || "Citation",
      }),
      0, // children go here
    ]
  },
})
```

**Step 2: Register in WritingTipTapEditor**

Add `CitationExtension` to the extensions array in `WritingTipTapEditor.tsx`:

```typescript
import { CitationExtension } from "./extensions/CitationExtension"

// In useMemo extensions:
CitationExtension,
```

**Step 3: Commit**

```bash
git add apps/packages/ui/src/components/Option/WritingPlayground/extensions/CitationExtension.ts \
       apps/packages/ui/src/components/Option/WritingPlayground/WritingTipTapEditor.tsx
git commit -m "feat(writing): add TipTap CitationExtension for inline references"
```

---

## Task 10: Frontend — Wire Phase 2 Tabs into WritingPlayground

**Files:**
- Modify: `apps/packages/ui/src/components/Option/WritingPlayground/WritingPlaygroundInspectorPanel.tsx`
- Modify: `apps/packages/ui/src/components/Option/WritingPlayground/index.tsx`

**Step 1: Add tabs to InspectorPanel**

In `WritingPlaygroundInspectorPanel.tsx`:

Add to `TAB_DEFINITIONS`:
```typescript
{ key: "characters", label: "Characters", testId: "writing-inspector-tab-characters" },
{ key: "research", label: "Research", testId: "writing-inspector-tab-research" },
```

Add new props: `characters?: ReactNode`, `research?: ReactNode`

Add to `panelMap`:
```typescript
characters: characters ?? null,
research: research ?? null,
```

**Step 2: Wire tab content in index.tsx**

Add imports:
```typescript
import { CharacterWorldTab } from "./CharacterWorldTab"
import { ResearchTab } from "./ResearchTab"
```

Create tab content variables (near the other tab content blocks around line 1900):
```typescript
const charactersTabContent = <CharacterWorldTab isOnline={isOnline} />
const researchTabContent = <ResearchTab isOnline={isOnline} />
```

Pass to `WritingPlaygroundInspectorPanel`:
```typescript
characters={charactersTabContent}
research={researchTabContent}
```

Add tab labels:
```typescript
tabLabels={{
  ...existing labels,
  characters: t("option:writingPlayground.sidebarCharacters", "Characters"),
  research: t("option:writingPlayground.sidebarResearch", "Research"),
}}
```

**Step 3: Commit**

```bash
git add apps/packages/ui/src/components/Option/WritingPlayground/WritingPlaygroundInspectorPanel.tsx \
       apps/packages/ui/src/components/Option/WritingPlayground/index.tsx
git commit -m "feat(writing): wire CharacterWorldTab and ResearchTab into inspector panel"
```

---

## Verification Checklist

1. **Backend migration**: `:memory:` DB creates at version 42 with 13+ manuscript tables
2. **Character CRUD**: `python -m pytest tldw_Server_API/tests/Writing/test_manuscript_characters_db.py -v`
3. **World/Plot/Citations CRUD**: `python -m pytest tldw_Server_API/tests/Writing/test_manuscript_world_plot_db.py -v`
4. **All Phase 1 tests still pass**: `python -m pytest tldw_Server_API/tests/Writing/test_manuscript_db.py -v`
5. **Integration tests**: `python -m pytest tldw_Server_API/tests/Writing/test_manuscript_phase2_integration.py -v`
6. **Frontend build**: `cd apps/tldw-frontend && pnpm build` (or `bun build`)
7. **Inspector tabs**: Characters and Research tabs appear in the inspector panel
8. **Character list**: Create project → add characters → see them in Characters tab
9. **Research search**: Select a scene → type query → see results → click Cite
10. **Citation marks**: Citation appears as highlighted inline text in TipTap editor
