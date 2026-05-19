# Writing Backend Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the confirmed Writing backend review findings and land the agreed regression coverage for snapshot atomicity, manuscript ownership and visibility invariants, PATCH null semantics, analysis parity and invalidation, and structured analysis parsing.

**Architecture:** The remediation is boundary-first. Persistence invariants live in `ManuscriptDBHelper`, HTTP contract fixes stay in `writing_manuscripts.py` and `writing.py`, and analysis response parsing stays in `manuscript_analysis.py`. Tests are added first in the existing Writing suites, then the smallest implementation changes needed to satisfy them are applied, followed by targeted verification and a final combined Writing slice plus Bandit.

**Tech Stack:** Python 3, FastAPI, Pydantic v2, pytest, sqlite/Postgres transaction helpers, Bandit, git

---

## File Structure

- `tldw_Server_API/app/api/v1/endpoints/writing.py`
  Snapshot import orchestration, soft-deleted session restore helper, wordcloud route behavior.
- `tldw_Server_API/app/api/v1/endpoints/writing_manuscripts.py`
  Manuscript create and update request shaping, project analysis routes, endpoint-level validation.
- `tldw_Server_API/app/core/DB_Management/ManuscriptDB.py`
  Chapter and world-info ownership checks, project-scoped list visibility, part delete cascade, analysis invalidation.
- `tldw_Server_API/app/core/Writing/manuscript_analysis.py`
  Structured LLM response parsing before JSON decoding.
- `tldw_Server_API/tests/Writing/test_writing_endpoint_integration.py`
  Snapshot replace rollback and wordcloud contract regressions.
- `tldw_Server_API/tests/Writing/test_manuscript_db.py`
  Helper-level chapter, part, reorder, and stale invalidation regressions.
- `tldw_Server_API/tests/Writing/test_manuscript_endpoint_integration.py`
  Core manuscript endpoint regressions, including deleted-project visibility for part and chapter list routes, PATCH null handling, and reorder contract coverage.
- `tldw_Server_API/tests/Writing/test_manuscript_world_plot_db.py`
  World-info parent validation regressions.
- `tldw_Server_API/tests/Writing/test_manuscript_phase2_integration.py`
  Character, relationship, world-info, plot-line, plot-event, and plot-hole regressions.
- `tldw_Server_API/tests/Writing/test_manuscript_analysis_integration.py`
  Project-analysis runtime validation and aggregate stale-analysis regressions.
- `tldw_Server_API/tests/Writing/test_manuscript_analysis_service.py`
  Structured `message.content` parsing regressions.

### Task 1: Make Snapshot Replace Atomic and Lock Wordcloud Error Contracts

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/writing.py:509-550`
- Modify: `tldw_Server_API/app/api/v1/endpoints/writing.py:1600-1740`
- Modify: `tldw_Server_API/app/api/v1/endpoints/writing.py:1934-2016`
- Test: `tldw_Server_API/tests/Writing/test_writing_endpoint_integration.py:180-440`

- [ ] **Step 1: Write the failing regression tests**

```python
def test_writing_snapshot_import_replace_rolls_back_on_restore_failure(
    client_with_writing_db: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    client = client_with_writing_db

    assert client.post(
        "/api/v1/writing/sessions",
        json={"name": "Keep Session", "payload": {"text": "keep"}},
    ).status_code == 201
    assert client.post(
        "/api/v1/writing/templates",
        json={"name": "Keep Template", "payload": {"inst_pre": "[K]"}},
    ).status_code == 201
    assert client.post(
        "/api/v1/writing/themes",
        json={"name": "Keep Theme", "class_name": "keep-theme", "css": ".keep-theme{}", "order": 1},
    ).status_code == 201

    from tldw_Server_API.app.api.v1.endpoints import writing as writing_endpoints

    original_restore = writing_endpoints._restore_soft_deleted_writing_session

    def fail_restore(*args, **kwargs):
        original_restore(*args, **kwargs)
        raise RuntimeError("restore failed after mutation")

    monkeypatch.setattr(writing_endpoints, "_restore_soft_deleted_writing_session", fail_restore)

    resp = client.post(
        "/api/v1/writing/snapshot/import",
        json={
            "mode": "replace",
            "snapshot": {
                "sessions": [{"id": "keep-session-id", "name": "Restored Session", "payload": {"text": "new"}, "schema_version": 1}],
                "templates": [],
                "themes": [],
            },
        },
    )

    assert resp.status_code == 500, resp.text
    sessions = client.get("/api/v1/writing/sessions").json()["sessions"]
    templates = client.get("/api/v1/writing/templates").json()["templates"]
    themes = client.get("/api/v1/writing/themes").json()["themes"]
    assert {item["name"] for item in sessions} == {"Keep Session"}
    assert {item["name"] for item in templates} == {"Keep Template"}
    assert {item["name"] for item in themes} == {"Keep Theme"}


def test_writing_snapshot_import_rejects_blank_session_name(client_with_writing_db: TestClient):
    client = client_with_writing_db

    resp = client.post(
        "/api/v1/writing/snapshot/import",
        json={
            "mode": "merge",
            "snapshot": {
                "sessions": [{"name": "   ", "payload": {"text": "ignored"}, "schema_version": 1}],
                "templates": [],
                "themes": [],
            },
        },
    )

    assert resp.status_code == 400, resp.text
    assert "session name" in resp.json()["detail"].lower()


def test_get_wordcloud_returns_404_for_unknown_id(client_with_writing_db: TestClient):
    client = client_with_writing_db

    resp = client.get("/api/v1/writing/wordclouds/does-not-exist")

    assert resp.status_code == 404, resp.text


def test_get_wordcloud_returns_failed_result(client_with_writing_db: TestClient, monkeypatch: pytest.MonkeyPatch):
    client = client_with_writing_db
    from tldw_Server_API.app.api.v1.endpoints import writing as writing_endpoints

    def boom(*_args, **_kwargs):
        raise RuntimeError("wordcloud failed")

    monkeypatch.setattr(writing_endpoints, "_compute_wordcloud", boom)

    create_resp = client.post("/api/v1/writing/wordclouds", json={"text": "alpha beta"})
    assert create_resp.status_code == 200, create_resp.text
    payload = create_resp.json()
    assert payload["status"] == "failed"

    get_resp = client.get(f"/api/v1/writing/wordclouds/{payload['id']}")
    assert get_resp.status_code == 200, get_resp.text
    assert get_resp.json()["status"] == "failed"
```

- [ ] **Step 2: Run the targeted Writing endpoint slice and verify it fails**

Run:
```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Writing/test_writing_endpoint_integration.py -k "snapshot_import_replace_rolls_back or blank_session_name or unknown_id or failed_result" -v
```

Expected: FAIL because the replace import leaves partially deleted state after the injected restore failure. The blank-session import test may also fail if names are only stripped at some call sites. The wordcloud tests may already pass; if they do, keep them as locked regression coverage.

- [ ] **Step 3: Write the minimal snapshot-atomic implementation**

```python
def _restore_soft_deleted_writing_session(
    conn: Any,
    db: CharactersRAGDB,
    *,
    session_id: str,
    name: str,
    payload: dict[str, Any],
    schema_version: int,
    version_parent_id: str | None,
) -> None:
    existing = db.get_writing_session(session_id, include_deleted=True)
    if not existing:
        raise ConflictError(
            f"Session with ID '{session_id}' already exists.",
            entity="writing_sessions",
            entity_id=session_id,
        )
    payload_json = json.dumps(payload, ensure_ascii=True)
    next_version = int(existing.get("version") or 1) + 1
    conn.execute(
        """
        UPDATE writing_sessions
           SET name = ?,
               payload_json = ?,
               schema_version = ?,
               version_parent_id = ?,
               deleted = 0,
               last_modified = CURRENT_TIMESTAMP,
               version = ?,
               client_id = ?
         WHERE id = ?
        """,
        (
            name,
            payload_json,
            int(schema_version),
            version_parent_id,
            next_version,
            db.client_id,
            session_id,
        ),
    )


# inside import_writing_snapshot()
    await _enforce_rate_limit(rate_limiter, int(current_user.id), "writing.snapshot.import")
    try:
        with db.transaction() as conn:
            # NOTE: Do NOT call db.soft_delete_writing_session() (or similar
            # high-level methods) inside an active transaction block -- those
            # methods may start their own transactions internally, causing
            # SQLite "cannot start a transaction within a transaction" errors.
            # Instead, pass the existing `conn` to perform deletions directly,
            # or refactor the helpers to accept an optional connection argument.
            if payload.mode == "replace":
                for session in _list_all_writing_sessions(db):
                    session_id = str(session.get("id") or "")
                    if session_id:
                        _soft_delete_writing_session_with_conn(conn, session_id, int(session.get("version") or 1))
                for template in _list_all_writing_templates(db):
                    template_name = str(template.get("name") or "")
                    if template_name:
                        _soft_delete_writing_template_with_conn(conn, template_name, int(template.get("version") or 1))
                for theme in _list_all_writing_themes(db):
                    theme_name = str(theme.get("name") or "")
                    if theme_name:
                        _soft_delete_writing_theme_with_conn(conn, theme_name, int(theme.get("version") or 1))

            imported_sessions = 0
            for session_item in payload.snapshot.sessions:
                session_name = session_item.name.strip()
                if not session_name:
                    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Session name cannot be empty")
                session_id = session_item.id.strip() if isinstance(session_item.id, str) and session_item.id.strip() else None
                try:
                    db.add_writing_session(
                        name=session_name,
                        payload=session_item.payload,
                        schema_version=int(session_item.schema_version),
                        session_id=session_id,
                        version_parent_id=session_item.version_parent_id,
                    )
                except ConflictError:
                    if not session_id:
                        raise
                    existing = db.get_writing_session(session_id, include_deleted=True)
                    if not existing:
                        raise
                    if bool(existing.get("deleted")):
                        _restore_soft_deleted_writing_session(
                            conn,
                            db,
                            session_id=session_id,
                            name=session_name,
                            payload=session_item.payload,
                            schema_version=int(session_item.schema_version),
                            version_parent_id=session_item.version_parent_id,
                        )
                    else:
                        db.update_writing_session(
                            session_id,
                            {
                                "name": session_name,
                                "payload": session_item.payload,
                                "schema_version": int(session_item.schema_version),
                                "version_parent_id": session_item.version_parent_id,
                            },
                            int(existing.get("version") or 1),
                        )
                imported_sessions += 1
            # keep the existing template/theme loops inside the same outer transaction
```

- [ ] **Step 4: Re-run the Writing endpoint slice**

Run:
```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Writing/test_writing_endpoint_integration.py -k "snapshot_import_replace_rolls_back or blank_session_name or unknown_id or failed_result" -v
```

Expected: PASS. The replace import rolls back fully on the injected failure, blank session names are rejected before mutation, and the wordcloud tests stay green.

- [ ] **Step 5: Commit the snapshot and wordcloud remediation**

```bash
git add tldw_Server_API/app/api/v1/endpoints/writing.py tldw_Server_API/tests/Writing/test_writing_endpoint_integration.py
git commit -m "fix: make writing snapshot replace atomic"
```

### Task 2: Enforce Helper-Level Ownership, Cascade, and Project-Scoped Visibility Rules

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/ManuscriptDB.py:191-245`
- Modify: `tldw_Server_API/app/core/DB_Management/ManuscriptDB.py:443-600`
- Modify: `tldw_Server_API/app/core/DB_Management/ManuscriptDB.py:493-566`
- Modify: `tldw_Server_API/app/core/DB_Management/ManuscriptDB.py:602-640`
- Modify: `tldw_Server_API/app/core/DB_Management/ManuscriptDB.py:1031-1095`
- Modify: `tldw_Server_API/app/core/DB_Management/ManuscriptDB.py:1171-1200`
- Modify: `tldw_Server_API/app/core/DB_Management/ManuscriptDB.py:1367-1375`
- Modify: `tldw_Server_API/app/core/DB_Management/ManuscriptDB.py:1510-1538`
- Modify: `tldw_Server_API/app/core/DB_Management/ManuscriptDB.py:1694-1702`
- Modify: `tldw_Server_API/app/core/DB_Management/ManuscriptDB.py:1957-1972`
- Test: `tldw_Server_API/tests/Writing/test_manuscript_db.py:232-663`
- Test: `tldw_Server_API/tests/Writing/test_manuscript_endpoint_integration.py:45-340`
- Test: `tldw_Server_API/tests/Writing/test_manuscript_world_plot_db.py:136-189`
- Test: `tldw_Server_API/tests/Writing/test_manuscript_phase2_integration.py:85-407`

- [ ] **Step 1: Write the failing helper and integration regressions**

```python
def test_create_chapter_rejects_cross_project_part(self, mdb):
    left = mdb.create_project("Left")
    right = mdb.create_project("Right")
    foreign_part = mdb.create_part(right, "Foreign")

    with pytest.raises(ValueError, match="different project"):
        mdb.create_chapter(left, "Intruder", part_id=foreign_part)


def test_update_chapter_rejects_cross_project_part(self, mdb):
    left = mdb.create_project("Left")
    right = mdb.create_project("Right")
    local_part = mdb.create_part(left, "Local")
    foreign_part = mdb.create_part(right, "Foreign")
    chapter_id = mdb.create_chapter(left, "Movable", part_id=local_part)

    with pytest.raises(ValueError, match="different project"):
        mdb.update_chapter(chapter_id, {"part_id": foreign_part}, expected_version=1)


def test_reorder_with_reparent_rejects_cross_project_part(self, mdb):
    left = mdb.create_project("Left")
    right = mdb.create_project("Right")
    local_part = mdb.create_part(left, "Local")
    foreign_part = mdb.create_part(right, "Foreign")
    chapter_id = mdb.create_chapter(left, "Movable", part_id=local_part)

    with pytest.raises(ValueError, match="different project"):
        mdb.reorder_items("chapter", [{"id": chapter_id, "sort_order": 0, "part_id": foreign_part}], project_id=left)


def test_soft_delete_part_cascades_scenes(self, mdb):
    project_id = mdb.create_project("Novel")
    part_id = mdb.create_part(project_id, "Part I")
    chapter_id = mdb.create_chapter(project_id, "Chapter 1", part_id=part_id)
    scene_id = mdb.create_scene(chapter_id, project_id, title="Scene 1", content_plain="alpha beta")

    mdb.soft_delete_part(part_id, expected_version=1)

    assert mdb.get_part(part_id) is None
    assert mdb.get_chapter(chapter_id) is None
    assert mdb.get_scene(scene_id) is None


def test_update_world_info_rejects_cross_project_parent(self, mdb):
    left = mdb.create_project("Left")
    right = mdb.create_project("Right")
    local = mdb.create_world_info(left, kind="location", name="Local")
    foreign_parent = mdb.create_world_info(right, kind="location", name="Foreign")

    with pytest.raises(ValueError, match="different project"):
        mdb.update_world_info(local, {"parent_id": foreign_parent}, expected_version=1)


def test_deleted_project_hides_project_scoped_lists(client: TestClient):
    project = _create_project(client, "Ghost Project")
    project_id = project["id"]

    part = client.post(f"{PREFIX}/projects/{project_id}/parts", json={"title": "Part I"}).json()
    client.post(
        f"{PREFIX}/projects/{project_id}/chapters",
        json={"title": "Chapter 1", "part_id": part["id"]},
    )

    char = client.post(f"{PREFIX}/projects/{project_id}/characters", json={"name": "Aldric", "role": "protagonist"}).json()
    other = client.post(f"{PREFIX}/projects/{project_id}/characters", json={"name": "Brin", "role": "supporting"}).json()
    client.post(
        f"{PREFIX}/projects/{project_id}/characters/relationships",
        json={"from_character_id": char["id"], "to_character_id": other["id"], "relationship_type": "ally", "bidirectional": True},
    )
    client.post(f"{PREFIX}/projects/{project_id}/world-info", json={"kind": "location", "name": "Keep"})
    plot_line = client.post(f"{PREFIX}/projects/{project_id}/plot-lines", json={"title": "Main Quest"}).json()
    client.post(f"{PREFIX}/projects/{project_id}/plot-holes", json={"title": "Gap", "description": "Missing", "severity": "high"})

    delete_resp = client.delete(f"{PREFIX}/projects/{project_id}", headers={"expected-version": "1"})
    assert delete_resp.status_code == 204

    assert client.get(f"{PREFIX}/projects/{project_id}/parts").json() == []
    assert client.get(f"{PREFIX}/projects/{project_id}/chapters").json() == []
    assert client.get(f"{PREFIX}/projects/{project_id}/characters").json() == []
    assert client.get(f"{PREFIX}/projects/{project_id}/characters/relationships").json() == []
    assert client.get(f"{PREFIX}/projects/{project_id}/world-info").json() == []
    assert client.get(f"{PREFIX}/projects/{project_id}/plot-lines").json() == []
    assert client.get(f"{PREFIX}/projects/{project_id}/plot-holes").json() == []
```

- [ ] **Step 2: Run the helper and phase-2 regression slices**

Run:
```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Writing/test_manuscript_db.py tldw_Server_API/tests/Writing/test_manuscript_endpoint_integration.py tldw_Server_API/tests/Writing/test_manuscript_world_plot_db.py tldw_Server_API/tests/Writing/test_manuscript_phase2_integration.py -k "cross_project or cascades_scenes or deleted_project_hides_project_scoped_lists" -v
```

Expected: FAIL because cross-project chapter reparenting is currently accepted, part delete leaves scenes behind, and deleted projects still leak descendants through project-scoped list routes, including parts and chapters.

- [ ] **Step 3: Implement the helper invariants**

```python
_PROJECT_CHECK_TABLES: frozenset[str] = frozenset({
    "manuscript_parts",
    "manuscript_characters",
    "manuscript_scenes",
    "manuscript_chapters",
    "manuscript_world_info",
    "manuscript_plot_lines",
})


def _project_is_active(self, conn: Any, project_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM manuscript_projects WHERE id = ? AND deleted = 0",
        (project_id,),
    ).fetchone()
    return row is not None


def list_parts(self, project_id: str) -> list[dict[str, Any]]:
    with self.db.transaction() as conn:
        if not self._project_is_active(conn, project_id):
            return []
        rows = conn.execute(
            "SELECT * FROM manuscript_parts WHERE project_id = ? AND deleted = 0 ORDER BY sort_order",
            (project_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def create_chapter(
    self,
    project_id: str,
    title: str,
    *,
    part_id: str | None = None,
    sort_order: float = 0.0,
    synopsis: str | None = None,
    status: str = "planned",
    chapter_id: str | None = None,
) -> str:
    with self.db.transaction() as conn:
        if part_id:
            self._assert_same_project(conn, "manuscript_parts", part_id, project_id, "part")
        conn.execute(
            """
            INSERT INTO manuscript_chapters
                (id, project_id, part_id, title, sort_order, synopsis,
                 pov_character_id, word_count, status,
                 created_at, last_modified, deleted, client_id, version)
            VALUES (?, ?, ?, ?, ?, ?, NULL, 0, ?, ?, ?, 0, ?, 1)
            """,
            (cid, project_id, part_id, title, sort_order, synopsis, status, now, now, self._client_id),
        )


def update_chapter(self, chapter_id: str, updates: dict[str, Any], expected_version: int) -> None:
    with self.db.transaction() as conn:
        if "part_id" in updates and updates["part_id"] is not None:
            chapter_row = conn.execute(
                "SELECT project_id FROM manuscript_chapters WHERE id = ? AND deleted = 0",
                (chapter_id,),
            ).fetchone()
            if chapter_row is None:
                raise ConflictError(
                    f"Chapter {chapter_id!r} update failed (version conflict or not found).",
                    entity="manuscript_chapters",
                    entity_id=chapter_id,
                )
            self._assert_same_project(
                conn,
                "manuscript_parts",
                updates["part_id"],
                chapter_row["project_id"],
                "part",
            )
        cur = conn.execute(
            f"UPDATE manuscript_chapters SET {', '.join(set_parts)} WHERE id = ? AND version = ? AND deleted = 0",
            params,
        )
        if cur.rowcount == 0:
            raise ConflictError(
                f"Chapter {chapter_id!r} update failed (version conflict or not found).",
                entity="manuscript_chapters",
                entity_id=chapter_id,
            )


def reorder_items(self, entity_type: str, items: list[dict[str, Any]], *, project_id: str | None = None) -> None:
    with self.db.transaction() as conn:
        if project_id is not None:
            if not self._project_is_active(conn, project_id):
                return
            for item in items:
                row = conn.execute(
                    f"SELECT project_id FROM {table} WHERE id = ? AND deleted = 0",
                    (item["id"],),
                ).fetchone()
                if row is None:
                    raise ValueError(f"{entity_type} {item['id']!r} not found")
                if row["project_id"] != project_id:
                    raise ValueError(f"{entity_type} {item['id']!r} does not belong to project {project_id!r}")
                if entity_type == "chapter" and item.get("part_id") is not None:
                    self._assert_same_project(conn, "manuscript_parts", item["part_id"], project_id, "part")
        for item in items:
            assignments = ["sort_order = ?"]
            params = [item["sort_order"]]
            if entity_type == "chapter" and "part_id" in item:
                assignments.append("part_id = ?")
                params.append(item["part_id"])
            if "version" in item and item["version"] is not None:
                params.extend([item["id"], item["version"]])
                where_clause = "WHERE id = ? AND version = ? AND deleted = 0"
            else:
                params.append(item["id"])
                where_clause = "WHERE id = ? AND deleted = 0"
            conn.execute(
                f"UPDATE {table} SET {', '.join(assignments)} {where_clause}",
                tuple(params),
            )


def soft_delete_part(self, part_id: str, expected_version: int) -> None:
    with self.db.transaction() as conn:
        chapter_ids = [
            row["id"]
            for row in conn.execute(
                "SELECT id FROM manuscript_chapters WHERE part_id = ? AND deleted = 0",
                (part_id,),
            ).fetchall()
        ]
        cur = conn.execute(
            "UPDATE manuscript_parts SET deleted = 1, last_modified = ?, version = ?, client_id = ? "
            "WHERE id = ? AND version = ? AND deleted = 0",
            (now, next_version, self._client_id, part_id, expected_version),
        )
        if cur.rowcount == 0:
            raise ConflictError(
                f"Part {part_id!r} delete failed (version conflict or not found).",
                entity="manuscript_parts",
                entity_id=part_id,
            )
        conn.execute(
            "UPDATE manuscript_chapters SET deleted = 1, last_modified = ?, client_id = ? "
            "WHERE part_id = ? AND deleted = 0",
            (now, self._client_id, part_id),
        )
        if chapter_ids:
            placeholders = ", ".join("?" for _ in chapter_ids)
            conn.execute(
                f"UPDATE manuscript_scenes SET deleted = 1, last_modified = ?, client_id = ? "
                f"WHERE chapter_id IN ({placeholders}) AND deleted = 0",
                (now, self._client_id, *chapter_ids),
            )


def update_world_info(self, world_info_id: str, updates: dict[str, Any], expected_version: int) -> None:
    with self.db.transaction() as conn:
        if "parent_id" in updates and updates["parent_id"] is not None:
            world_info_row = conn.execute(
                "SELECT project_id FROM manuscript_world_info WHERE id = ? AND deleted = 0",
                (world_info_id,),
            ).fetchone()
            if world_info_row is None:
                raise ConflictError(
                    f"WorldInfo {world_info_id!r} update failed (version conflict or not found).",
                    entity="manuscript_world_info",
                    entity_id=world_info_id,
                )
            self._assert_same_project(
                conn,
                "manuscript_world_info",
                updates["parent_id"],
                world_info_row["project_id"],
                "parent_world_info",
            )
        cur = conn.execute(
            f"UPDATE manuscript_world_info SET {', '.join(set_parts)} WHERE id = ? AND version = ? AND deleted = 0",
            params,
        )
        if cur.rowcount == 0:
            raise ConflictError(
                f"WorldInfo {world_info_id!r} update failed (version conflict or not found).",
                entity="manuscript_world_info",
                entity_id=world_info_id,
            )
```

- [ ] **Step 4: Re-run the helper and phase-2 slices**

Run:
```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Writing/test_manuscript_db.py tldw_Server_API/tests/Writing/test_manuscript_endpoint_integration.py tldw_Server_API/tests/Writing/test_manuscript_world_plot_db.py tldw_Server_API/tests/Writing/test_manuscript_phase2_integration.py -k "cross_project or cascades_scenes or deleted_project_hides_project_scoped_lists" -v
```

Expected: PASS. Cross-project reparent attempts now fail, part deletes remove scenes, and deleted projects return empty project-scoped descendant lists, including parts and chapters.

- [ ] **Step 5: Commit the helper-boundary remediation**

```bash
git add tldw_Server_API/app/core/DB_Management/ManuscriptDB.py tldw_Server_API/tests/Writing/test_manuscript_db.py tldw_Server_API/tests/Writing/test_manuscript_world_plot_db.py tldw_Server_API/tests/Writing/test_manuscript_phase2_integration.py
git commit -m "fix: enforce manuscript ownership invariants"
```

### Task 3: Honor Explicit Null and Trimmed-Text Contracts for Core Manuscript Endpoints

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/writing_manuscripts.py:203-301`
- Modify: `tldw_Server_API/app/api/v1/endpoints/writing_manuscripts.py:468-572`
- Modify: `tldw_Server_API/app/api/v1/endpoints/writing_manuscripts.py:610-723`
- Modify: `tldw_Server_API/app/api/v1/endpoints/writing_manuscripts.py:757-892`
- Test: `tldw_Server_API/tests/Writing/test_manuscript_endpoint_integration.py:225-355`

- [ ] **Step 1: Write the failing core endpoint contract tests**

```python
def test_project_patch_null_clears_synopsis_and_settings(client: TestClient):
    create_resp = client.post(
        f"{PREFIX}/projects",
        json={"title": "Null Project", "synopsis": "Filled", "settings": {"theme": "dark"}},
    )
    assert create_resp.status_code == 201, create_resp.text
    project = create_resp.json()

    resp = client.patch(
        f"{PREFIX}/projects/{project['id']}",
        json={"synopsis": None, "settings": None},
        headers={"expected-version": str(project["version"])},
    )

    assert resp.status_code == 200, resp.text
    assert resp.json()["synopsis"] is None
    assert resp.json()["settings"] == {}


def test_chapter_patch_null_clears_part_and_synopsis(client: TestClient):
    project = client.post(f"{PREFIX}/projects", json={"title": "Chapter Nulls"}).json()
    part = client.post(f"{PREFIX}/projects/{project['id']}/parts", json={"title": "Part I"}).json()
    chapter = client.post(
        f"{PREFIX}/projects/{project['id']}/chapters",
        json={"title": "Chapter 1", "part_id": part["id"], "synopsis": "Filled"},
    ).json()

    resp = client.patch(
        f"{PREFIX}/chapters/{chapter['id']}",
        json={"part_id": None, "synopsis": None},
        headers={"expected-version": str(chapter["version"])},
    )

    assert resp.status_code == 200, resp.text
    assert resp.json()["part_id"] is None
    assert resp.json()["synopsis"] is None


def test_create_project_rejects_whitespace_title(client: TestClient):
    resp = client.post(f"{PREFIX}/projects", json={"title": "   "})
    assert resp.status_code == 400, resp.text


def test_chapter_patch_rejects_whitespace_title(client: TestClient):
    project = client.post(f"{PREFIX}/projects", json={"title": "Whitespace Patch"}).json()
    chapter = client.post(
        f"{PREFIX}/projects/{project['id']}/chapters",
        json={"title": "Chapter 1"},
    ).json()

    resp = client.patch(
        f"{PREFIX}/chapters/{chapter['id']}",
        json={"title": "   "},
        headers={"expected-version": str(chapter["version"])},
    )

    assert resp.status_code == 400, resp.text


def test_patch_requires_expected_version_header(client: TestClient):
    project = client.post(f"{PREFIX}/projects", json={"title": "Missing Header"}).json()
    resp = client.patch(f"{PREFIX}/projects/{project['id']}", json={"synopsis": None})
    assert resp.status_code == 422, resp.text


def test_project_patch_rejects_empty_payload(client: TestClient):
    project = client.post(f"{PREFIX}/projects", json={"title": "Empty Patch"}).json()
    resp = client.patch(
        f"{PREFIX}/projects/{project['id']}",
        json={},
        headers={"expected-version": str(project["version"])},
    )
    assert resp.status_code == 400, resp.text


def test_reorder_rejects_stale_item_version(client: TestClient):
    project = client.post(f"{PREFIX}/projects", json={"title": "Reorder Conflict"}).json()
    chapter = client.post(
        f"{PREFIX}/projects/{project['id']}/chapters",
        json={"title": "Chapter R"},
    ).json()
    scene = client.post(
        f"{PREFIX}/chapters/{chapter['id']}/scenes",
        json={"title": "Scene 0", "sort_order": 0.0},
    ).json()

    patch_resp = client.patch(
        f"{PREFIX}/scenes/{scene['id']}",
        json={"title": "Scene 0 Updated"},
        headers={"expected-version": str(scene["version"])},
    )
    assert patch_resp.status_code == 200, patch_resp.text

    resp = client.post(
        f"{PREFIX}/projects/{project['id']}/reorder",
        json={
            "entity_type": "scenes",
            "items": [{"id": scene["id"], "sort_order": 1.0, "version": scene["version"]}],
        },
    )
    assert resp.status_code == 409, resp.text
```

- [ ] **Step 2: Run the core endpoint regression slice**

Run:
```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Writing/test_manuscript_endpoint_integration.py -k "null_clears or whitespace_title or expected_version_header or empty_payload or stale_item_version" -v
```

Expected: FAIL because explicit `null` is currently dropped, empty effective PATCH payloads are accepted, stale reorder item versions are not rejected consistently, and the new contract tests expose the mismatch.

- [ ] **Step 3: Add field-presence-aware parsing for project, part, chapter, and scene handlers**

```python
def _field_present(payload: Any, field_name: str) -> bool:
    return field_name in payload.model_fields_set


def _require_non_empty_text(value: str | None, label: str) -> str:
    if value is None:
        raise ValueError(f"{label} cannot be null")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{label} cannot be empty")
    return normalized


def _normalize_mapping_field(value: dict[str, Any] | None) -> dict[str, Any]:
    return {} if value is None else value


# inside create_project endpoint
    project_id = helper.create_project(
        title=_require_non_empty_text(payload.title, "Project title"),
        subtitle=payload.subtitle,
        author=payload.author,
        genre=payload.genre,
        status=payload.status,
        synopsis=payload.synopsis,
        target_word_count=payload.target_word_count,
        settings=_normalize_mapping_field(payload.settings) if payload.settings is not None else payload.settings,
        project_id=payload.id,
    )


# inside update_project endpoint
    update_data: dict[str, Any] = {}
    if _field_present(payload, "title"):
        update_data["title"] = _require_non_empty_text(payload.title, "Project title")
    if _field_present(payload, "subtitle"):
        update_data["subtitle"] = payload.subtitle
    if _field_present(payload, "author"):
        update_data["author"] = payload.author
    if _field_present(payload, "genre"):
        update_data["genre"] = payload.genre
    if _field_present(payload, "status"):
        update_data["status"] = payload.status
    if _field_present(payload, "synopsis"):
        update_data["synopsis"] = payload.synopsis
    if _field_present(payload, "target_word_count"):
        update_data["target_word_count"] = payload.target_word_count
    if _field_present(payload, "settings"):
        update_data["settings"] = _normalize_mapping_field(payload.settings)
    if not update_data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No fields provided for update")


# inside update_chapter endpoint
    update_data: dict[str, Any] = {}
    if _field_present(payload, "title"):
        update_data["title"] = _require_non_empty_text(payload.title, "Chapter title")
    if _field_present(payload, "part_id"):
        update_data["part_id"] = payload.part_id
    if _field_present(payload, "sort_order"):
        update_data["sort_order"] = payload.sort_order
    if _field_present(payload, "synopsis"):
        update_data["synopsis"] = payload.synopsis
    if _field_present(payload, "status"):
        update_data["status"] = payload.status
    if not update_data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No fields provided for update")


# inside update_scene endpoint
    update_data: dict[str, Any] = {}
    if _field_present(payload, "title"):
        update_data["title"] = _require_non_empty_text(payload.title, "Scene title")
    if _field_present(payload, "content"):
        update_data["content_json"] = json.dumps(payload.content) if payload.content is not None else None
    elif _field_present(payload, "content_plain"):
        update_data["content_json"] = None
    if _field_present(payload, "content_plain"):
        update_data["content_plain"] = payload.content_plain
    if _field_present(payload, "synopsis"):
        update_data["synopsis"] = payload.synopsis
    if _field_present(payload, "sort_order"):
        update_data["sort_order"] = payload.sort_order
    if _field_present(payload, "status"):
        update_data["status"] = payload.status
```

- [ ] **Step 4: Re-run the core endpoint regression slice**

Run:
```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Writing/test_manuscript_endpoint_integration.py -k "null_clears or whitespace_title or expected_version_header or empty_payload or stale_item_version" -v
```

Expected: PASS. Explicit `null` now clears genuine nullable fields, container-backed fields normalize correctly, empty effective PATCH payloads return `400`, stale reorder versions return `409`, whitespace-only required titles return `400`, and the missing-header regression remains locked at `422`.

- [ ] **Step 5: Commit the core endpoint contract changes**

```bash
git add tldw_Server_API/app/api/v1/endpoints/writing_manuscripts.py tldw_Server_API/tests/Writing/test_manuscript_endpoint_integration.py
git commit -m "fix: honor manuscript patch null semantics"
```

### Task 4: Extend the Same PATCH and Trimmed-Text Rules to Character, World, and Plot Endpoints

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/writing_manuscripts.py:927-1043`
- Modify: `tldw_Server_API/app/api/v1/endpoints/writing_manuscripts.py:1112-1260`
- Modify: `tldw_Server_API/app/api/v1/endpoints/writing_manuscripts.py:1333-1505`
- Modify: `tldw_Server_API/app/api/v1/endpoints/writing_manuscripts.py:1591-1622`
- Test: `tldw_Server_API/tests/Writing/test_manuscript_phase2_integration.py:85-407`

- [ ] **Step 1: Write the failing phase-2 endpoint regressions**

```python
def test_character_patch_null_custom_fields_resets_to_empty_object(client: TestClient):
    project = _create_project(client, "Character Nulls")
    created = client.post(
        f"{PREFIX}/projects/{project['id']}/characters",
        json={"name": "Aldric", "role": "protagonist", "custom_fields": {"hair_color": "black"}},
    ).json()

    resp = client.patch(
        f"{PREFIX}/characters/{created['id']}",
        json={"custom_fields": None},
        headers={"expected-version": "1"},
    )

    assert resp.status_code == 200, resp.text
    assert resp.json()["custom_fields"] == {}


def test_world_info_patch_null_parent_and_collections_reset(client: TestClient):
    project = _create_project(client, "World Nulls")
    parent = client.post(f"{PREFIX}/projects/{project['id']}/world-info", json={"kind": "location", "name": "Parent"}).json()
    child = client.post(
        f"{PREFIX}/projects/{project['id']}/world-info",
        json={"kind": "location", "name": "Child", "parent_id": parent["id"], "properties": {"population": 3}, "tags": ["keep"]},
    ).json()

    resp = client.patch(
        f"{PREFIX}/world-info/{child['id']}",
        json={"parent_id": None, "properties": None, "tags": None},
        headers={"expected-version": "1"},
    )

    assert resp.status_code == 200, resp.text
    assert resp.json()["parent_id"] is None
    assert resp.json()["properties"] == {}
    assert resp.json()["tags"] == []


def test_plot_line_create_rejects_whitespace_title(client: TestClient):
    project = _create_project(client, "Plot Titles")
    resp = client.post(f"{PREFIX}/projects/{project['id']}/plot-lines", json={"title": "   "})
    assert resp.status_code == 400, resp.text


def test_plot_line_patch_null_description(client: TestClient):
    project = _create_project(client, "Plot Line Nulls")
    plot_line = client.post(
        f"{PREFIX}/projects/{project['id']}/plot-lines",
        json={"title": "Main Quest", "description": "Filled"},
    ).json()

    resp = client.patch(
        f"{PREFIX}/plot-lines/{plot_line['id']}",
        json={"description": None},
        headers={"expected-version": "1"},
    )

    assert resp.status_code == 200, resp.text
    assert resp.json()["description"] is None


def test_plot_event_patch_allows_nullable_scene_links(client: TestClient):
    project = _create_project(client, "Plot Event Nulls")
    plot_line = client.post(f"{PREFIX}/projects/{project['id']}/plot-lines", json={"title": "Main Quest"}).json()
    event = client.post(
        f"{PREFIX}/plot-lines/{plot_line['id']}/events",
        json={"title": "Dragon Appears", "scene_id": None, "chapter_id": None},
    ).json()

    resp = client.patch(
        f"{PREFIX}/plot-events/{event['id']}",
        json={"scene_id": None, "chapter_id": None, "description": None},
        headers={"expected-version": "1"},
    )

    assert resp.status_code == 200, resp.text
    assert resp.json()["scene_id"] is None
    assert resp.json()["chapter_id"] is None
    assert resp.json()["description"] is None


def test_character_patch_rejects_empty_payload(client: TestClient):
    project = _create_project(client, "Character Empty Patch")
    created = client.post(
        f"{PREFIX}/projects/{project['id']}/characters",
        json={"name": "Aldric", "role": "protagonist"},
    ).json()

    resp = client.patch(
        f"{PREFIX}/characters/{created['id']}",
        json={},
        headers={"expected-version": "1"},
    )

    assert resp.status_code == 400, resp.text
```

- [ ] **Step 2: Run the phase-2 endpoint regression slice**

Run:
```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Writing/test_manuscript_phase2_integration.py -k "null_custom_fields or collections_reset or whitespace_title or patch_null_description or nullable_scene_links or empty_payload" -v
```

Expected: FAIL because these handlers still rely on `exclude_none=True` or unguarded `strip()` behavior, and empty effective PATCH payloads are not rejected consistently.

- [ ] **Step 3: Reuse the core PATCH helpers across the phase-2 endpoints**

```python
# inside create_character endpoint
    character_id = helper.create_character(
        project_id=project_id,
        name=_require_non_empty_text(payload.name, "Character name"),
        role=payload.role,
        cast_group=payload.cast_group,
        full_name=payload.full_name,
        age=payload.age,
        gender=payload.gender,
        appearance=payload.appearance,
        personality=payload.personality,
        backstory=payload.backstory,
        motivation=payload.motivation,
        arc_summary=payload.arc_summary,
        notes=payload.notes,
        custom_fields=payload.custom_fields or {},
        sort_order=payload.sort_order,
        character_id=payload.id,
    )


# inside update_character endpoint
    update_data: dict[str, Any] = {}
    if _field_present(payload, "name"):
        update_data["name"] = _require_non_empty_text(payload.name, "Character name")
    if _field_present(payload, "role"):
        update_data["role"] = payload.role
    if _field_present(payload, "motivation"):
        update_data["motivation"] = payload.motivation
    if _field_present(payload, "notes"):
        update_data["notes"] = payload.notes
    if _field_present(payload, "custom_fields"):
        update_data["custom_fields"] = {} if payload.custom_fields is None else payload.custom_fields


# inside create_world_info endpoint
    item_id = helper.create_world_info(
        project_id=project_id,
        kind=payload.kind,
        name=_require_non_empty_text(payload.name, "World info name"),
        description=payload.description,
        parent_id=payload.parent_id,
        properties=payload.properties or {},
        tags=payload.tags or [],
        sort_order=payload.sort_order,
        world_info_id=payload.id,
    )


# inside update_world_info endpoint
    update_data: dict[str, Any] = {}
    if _field_present(payload, "name"):
        update_data["name"] = _require_non_empty_text(payload.name, "World info name")
    if _field_present(payload, "description"):
        update_data["description"] = payload.description
    if _field_present(payload, "parent_id"):
        update_data["parent_id"] = payload.parent_id
    if _field_present(payload, "properties"):
        update_data["properties"] = {} if payload.properties is None else payload.properties
    if _field_present(payload, "tags"):
        update_data["tags"] = [] if payload.tags is None else payload.tags
    if _field_present(payload, "sort_order"):
        update_data["sort_order"] = payload.sort_order


# inside update_plot_line endpoint
    update_data: dict[str, Any] = {}
    if _field_present(payload, "title"):
        update_data["title"] = _require_non_empty_text(payload.title, "Plot line title")
    if _field_present(payload, "description"):
        update_data["description"] = payload.description
    if _field_present(payload, "status"):
        update_data["status"] = payload.status
    if _field_present(payload, "color"):
        update_data["color"] = payload.color
    if _field_present(payload, "sort_order"):
        update_data["sort_order"] = payload.sort_order


# inside update_plot_event endpoint
    update_data: dict[str, Any] = {}
    if _field_present(payload, "title"):
        update_data["title"] = _require_non_empty_text(payload.title, "Plot event title")
    if _field_present(payload, "description"):
        update_data["description"] = payload.description
    if _field_present(payload, "scene_id"):
        update_data["scene_id"] = payload.scene_id
    if _field_present(payload, "chapter_id"):
        update_data["chapter_id"] = payload.chapter_id
    if _field_present(payload, "event_type"):
        update_data["event_type"] = payload.event_type
    if _field_present(payload, "sort_order"):
        update_data["sort_order"] = payload.sort_order


# inside update_plot_hole endpoint
    update_data: dict[str, Any] = {}
    if _field_present(payload, "title"):
        update_data["title"] = _require_non_empty_text(payload.title, "Plot hole title")
    if _field_present(payload, "description"):
        update_data["description"] = payload.description
    if _field_present(payload, "resolution"):
        update_data["resolution"] = payload.resolution
    if _field_present(payload, "scene_id"):
        update_data["scene_id"] = payload.scene_id
    if _field_present(payload, "chapter_id"):
        update_data["chapter_id"] = payload.chapter_id
    if _field_present(payload, "plot_line_id"):
        update_data["plot_line_id"] = payload.plot_line_id
    if _field_present(payload, "status"):
        update_data["status"] = payload.status
    if _field_present(payload, "severity"):
        update_data["severity"] = payload.severity
```

- [ ] **Step 4: Re-run the phase-2 endpoint regression slice**

Run:
```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Writing/test_manuscript_phase2_integration.py -k "null_custom_fields or collections_reset or whitespace_title or patch_null_description or nullable_scene_links or empty_payload" -v
```

Expected: PASS. Character, world, and plot handlers now preserve explicit field presence, normalize collection-backed nulls, reject empty effective PATCH payloads, and reject whitespace-only required titles.

- [ ] **Step 5: Commit the phase-2 endpoint contract changes**

```bash
git add tldw_Server_API/app/api/v1/endpoints/writing_manuscripts.py tldw_Server_API/tests/Writing/test_manuscript_phase2_integration.py
git commit -m "fix: normalize phase2 manuscript patch payloads"
```

### Task 5: Align Analysis Route Validation, Aggregate Staleness, and Structured Parsing

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/ManuscriptDB.py:675-830`
- Modify: `tldw_Server_API/app/core/DB_Management/ManuscriptDB.py:2294-2304`
- Modify: `tldw_Server_API/app/api/v1/endpoints/writing_manuscripts.py:1977-2188`
- Modify: `tldw_Server_API/app/core/Writing/manuscript_analysis.py:116-170`
- Test: `tldw_Server_API/tests/Writing/test_manuscript_analysis_integration.py:211-423`
- Test: `tldw_Server_API/tests/Writing/test_manuscript_analysis_service.py:12-163`

- [ ] **Step 1: Write the failing analysis and parser regressions**

```python
def test_project_analysis_endpoints_enforce_runtime_rate_limit(client: TestClient):
    project_id, _chapter_id, _scene_id = _create_project_chapter_scene(client)

    class _RejectingRateLimiter:
        async def check_user_rate_limit(self, *_args, **_kwargs):
            return False, {"retry_after": 13}

    client.app.dependency_overrides[get_rate_limiter_dep] = lambda: _RejectingRateLimiter()

    resp = client.post(f"{PREFIX}/projects/{project_id}/analyze/plot-holes", json={})
    assert resp.status_code == 429, resp.text
    assert resp.headers["Retry-After"] == "13"


def test_project_analysis_rejects_unknown_provider_override(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    project_id, _chapter_id, _scene_id = _create_project_chapter_scene(client)
    import tldw_Server_API.app.api.v1.endpoints.writing_manuscripts as writing_endpoint

    monkeypatch.setattr(
        writing_endpoint,
        "get_provider_manager",
        lambda: SimpleNamespace(providers=["openai"], primary_provider="openai"),
    )
    monkeypatch.setattr(writing_endpoint, "is_model_known_for_provider", lambda *_args, **_kwargs: True)

    resp = client.post(
        f"{PREFIX}/projects/{project_id}/analyze/consistency",
        json={"provider": "bad-provider", "model": "gpt-4o-mini"},
    )
    assert resp.status_code == 400, resp.text


def test_project_analysis_rejects_unknown_model_override(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    project_id, _chapter_id, _scene_id = _create_project_chapter_scene(client)
    import tldw_Server_API.app.api.v1.endpoints.writing_manuscripts as writing_endpoint

    monkeypatch.setattr(
        writing_endpoint,
        "get_provider_manager",
        lambda: SimpleNamespace(providers=["openai"], primary_provider="openai"),
    )
    monkeypatch.setattr(
        writing_endpoint,
        "is_model_known_for_provider",
        lambda provider, model: False if provider == "openai" and model == "bad-model" else True,
    )

    resp = client.post(
        f"{PREFIX}/projects/{project_id}/analyze/plot-holes",
        json={"provider": "openai", "model": "bad-model"},
    )
    assert resp.status_code == 400, resp.text


def test_stale_after_scene_create_marks_chapter_and_project_analyses(client: TestClient):
    project_id, chapter_id, scene_id = _create_project_chapter_scene(client)

    with patch(f"{_LLM_MODULE}.perform_chat_api_call_async", new_callable=AsyncMock, return_value=_mock_llm_response(_pacing_json())):
        assert client.post(f"{PREFIX}/chapters/{chapter_id}/analyze", json={"analysis_types": ["pacing"]}).status_code == 200
    with patch(f"{_LLM_MODULE}.perform_chat_api_call_async", new_callable=AsyncMock, return_value=_mock_llm_response(_consistency_json())):
        assert client.post(f"{PREFIX}/projects/{project_id}/analyze/consistency", json={}).status_code == 200

    create_resp = client.post(
        f"{PREFIX}/chapters/{chapter_id}/scenes",
        json={"title": "Inserted Scene", "content_plain": "new text"},
    )
    assert create_resp.status_code == 201, create_resp.text

    analyses = client.get(f"{PREFIX}/projects/{project_id}/analyses", params={"include_stale": True}).json()["analyses"]
    assert {item["scope_type"] for item in analyses} == {"chapter", "project"}
    assert all(item["stale"] is True for item in analyses)


def test_stale_after_scene_update_marks_chapter_and_project_analyses(client: TestClient):
    project_id, chapter_id, scene_id = _create_project_chapter_scene(client)

    with patch(f"{_LLM_MODULE}.perform_chat_api_call_async", new_callable=AsyncMock, return_value=_mock_llm_response(_pacing_json())):
        assert client.post(f"{PREFIX}/chapters/{chapter_id}/analyze", json={"analysis_types": ["pacing"]}).status_code == 200
    with patch(f"{_LLM_MODULE}.perform_chat_api_call_async", new_callable=AsyncMock, return_value=_mock_llm_response(_consistency_json())):
        assert client.post(f"{PREFIX}/projects/{project_id}/analyze/consistency", json={}).status_code == 200

    scene = client.get(f"{PREFIX}/scenes/{scene_id}").json()
    patch_resp = client.patch(
        f"{PREFIX}/scenes/{scene_id}",
        json={"content_plain": "changed text for stale aggregate coverage"},
        headers={"expected-version": str(scene["version"])},
    )
    assert patch_resp.status_code == 200, patch_resp.text

    analyses = client.get(f"{PREFIX}/projects/{project_id}/analyses", params={"include_stale": True}).json()["analyses"]
    assert {item["scope_type"] for item in analyses} == {"chapter", "project"}
    assert all(item["stale"] is True for item in analyses)


def test_stale_after_scene_delete_marks_chapter_and_project_analyses(client: TestClient):
    project_id, chapter_id, scene_id = _create_project_chapter_scene(client)

    with patch(f"{_LLM_MODULE}.perform_chat_api_call_async", new_callable=AsyncMock, return_value=_mock_llm_response(_pacing_json())):
        assert client.post(f"{PREFIX}/chapters/{chapter_id}/analyze", json={"analysis_types": ["pacing"]}).status_code == 200
    with patch(f"{_LLM_MODULE}.perform_chat_api_call_async", new_callable=AsyncMock, return_value=_mock_llm_response(_consistency_json())):
        assert client.post(f"{PREFIX}/projects/{project_id}/analyze/consistency", json={}).status_code == 200

    scene = client.get(f"{PREFIX}/scenes/{scene_id}").json()
    delete_resp = client.delete(
        f"{PREFIX}/scenes/{scene_id}",
        headers={"expected-version": str(scene["version"])},
    )
    assert delete_resp.status_code == 204, delete_resp.text

    analyses = client.get(f"{PREFIX}/projects/{project_id}/analyses", params={"include_stale": True}).json()["analyses"]
    assert {item["scope_type"] for item in analyses} == {"chapter", "project"}
    assert all(item["stale"] is True for item in analyses)


@pytest.mark.asyncio
async def test_analyze_pacing_handles_list_message_content():
    from tldw_Server_API.app.core.Writing.manuscript_analysis import analyze_pacing

    structured = {
        "choices": [
            {
                "message": {
                    "content": [
                        {"type": "text", "text": '{"pacing": 0.44, "assessment": "Structured"}'}
                    ]
                }
            }
        ]
    }
    with patch(f"{MODULE}.perform_chat_api_call_async", new_callable=AsyncMock, return_value=structured):
        result = await analyze_pacing("Text")
        assert result["pacing"] == 0.44
        assert result["assessment"] == "Structured"
```

- [ ] **Step 2: Run the analysis and parser regression slice**

Run:
```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Writing/test_manuscript_analysis_integration.py tldw_Server_API/tests/Writing/test_manuscript_analysis_service.py -k "project_analysis or stale_after_scene_create or stale_after_scene_update or stale_after_scene_delete or handles_list_message_content" -v
```

Expected: FAIL because project analysis routes currently bypass runtime rate-limit and override validation, aggregate analyses are not invalidated on scene create, update, and delete, and `_extract_content()` cannot parse list-based `message.content`.

- [ ] **Step 3: Implement the analysis route and parser fixes**

```python
def _mark_scope_family_stale_in_txn(
    self,
    conn: Any,
    *,
    scene_id: str | None = None,
    chapter_id: str | None = None,
    project_id: str | None = None,
) -> None:
    if scene_id is not None:
        self._mark_analyses_stale_in_txn(conn, "scene", scene_id)
    if chapter_id is not None:
        self._mark_analyses_stale_in_txn(conn, "chapter", chapter_id)
    if project_id is not None:
        self._mark_analyses_stale_in_txn(conn, "project", project_id)


# inside ManuscriptDBHelper.create_scene()
    with self.db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO manuscript_scenes
                (id, chapter_id, project_id, title, content_json, content_plain, synopsis,
                 sort_order, word_count, status, created_at, last_modified, deleted, client_id, version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, 1)
            """,
            (sid, chapter_id, project_id, title, content_json, content_plain, synopsis, sort_order, word_count, status, now, now, self._client_id),
        )
        self._propagate_word_counts(conn, chapter_id, project_id)
        self._mark_scope_family_stale_in_txn(conn, chapter_id=chapter_id, project_id=project_id)


# inside ManuscriptDBHelper.update_scene()
    with self.db.transaction() as conn:
        cur = conn.execute(
            f"UPDATE manuscript_scenes SET {', '.join(set_parts)} WHERE id = ? AND version = ? AND deleted = 0",
            params,
        )
        if cur.rowcount == 0:
            raise ConflictError(
                f"Scene {scene_id!r} update failed (version conflict or not found).",
                entity="manuscript_scenes",
                entity_id=scene_id,
            )
        if "content_plain" in updates or "content_json" in updates:
            row = conn.execute(
                "SELECT chapter_id, project_id FROM manuscript_scenes WHERE id = ?",
                (scene_id,),
            ).fetchone()
            if row:
                self._propagate_word_counts(conn, row["chapter_id"], row["project_id"])
                self._mark_scope_family_stale_in_txn(
                    conn,
                    scene_id=scene_id,
                    chapter_id=row["chapter_id"],
                    project_id=row["project_id"],
                )


# inside ManuscriptDBHelper.soft_delete_scene()
    with self.db.transaction() as conn:
        row = conn.execute(
            "SELECT chapter_id, project_id FROM manuscript_scenes WHERE id = ? AND deleted = 0",
            (scene_id,),
        ).fetchone()
        cur = conn.execute(
            "UPDATE manuscript_scenes SET deleted = 1, last_modified = ?, version = ?, client_id = ? "
            "WHERE id = ? AND version = ? AND deleted = 0",
            (now, next_version, self._client_id, scene_id, expected_version),
        )
        if cur.rowcount == 0:
            raise ConflictError(
                f"Scene {scene_id!r} delete failed (version conflict or not found).",
                entity="manuscript_scenes",
                entity_id=scene_id,
            )
        if row:
            self._propagate_word_counts(conn, row["chapter_id"], row["project_id"])
            self._mark_scope_family_stale_in_txn(
                conn,
                chapter_id=row["chapter_id"],
                project_id=row["project_id"],
            )


async def analyze_plot_holes_endpoint(
    project_id: str,
    payload: ManuscriptAnalysisRequest,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
    current_user: User = Depends(get_request_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.analyze")),
) -> list[ManuscriptAnalysisResponse]:
    await _enforce_rate_limit(rate_limiter, int(current_user.id), "writing.manuscripts.analyze")
    provider_override, model_override = _validate_analysis_overrides(
        provider=payload.provider,
        model=payload.model,
    )

    # Build analysis inputs from the project's manuscript content.
    # combined_text: full manuscript text concatenated from all scenes/chapters.
    # char_summary / world_summary: summaries extracted from project metadata.
    # helper: a ManuscriptAnalysisHelper(db) instance scoped to the current user's DB.
    helper = ManuscriptAnalysisHelper(db)
    combined_text = helper.get_combined_manuscript_text(project_id)
    char_summary = helper.get_character_summary(project_id)
    world_summary = helper.get_world_summary(project_id)

    result = await _analyze_plot_holes(
        combined_text,
        char_summary,
        world_summary,
        provider=provider_override,
        model=model_override,
    )
    aid = helper.create_analysis(
        project_id,
        "project",
        project_id,
        "plot_holes",
        result,
        provider=provider_override,
        model=model_override,
    )


def _extract_content_block(block: Any) -> str:
    if isinstance(block, str):
        return block
    if isinstance(block, dict):
        text = block.get("text")
        if isinstance(text, str):
            return text
        content = block.get("content")
        if isinstance(content, str):
            return content
    return ""


def _extract_content(response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        choices = response.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            if isinstance(msg, dict):
                content = msg.get("content", "")
                if isinstance(content, list):
                    return "".join(_extract_content_block(part) for part in content)
                if isinstance(content, str):
                    return content
                return _extract_content_block(content)
        content = response.get("content", "")
        if isinstance(content, list):
            return "".join(_extract_content_block(part) for part in content)
        return content if isinstance(content, str) else ""
    return str(response)
```

- [ ] **Step 4: Run the full touched Writing suite and Bandit**

Run:
```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Writing/test_writing_endpoint_integration.py tldw_Server_API/tests/Writing/test_manuscript_endpoint_integration.py tldw_Server_API/tests/Writing/test_manuscript_phase2_integration.py tldw_Server_API/tests/Writing/test_manuscript_analysis_integration.py tldw_Server_API/tests/Writing/test_manuscript_analysis_service.py tldw_Server_API/tests/Writing/test_manuscript_db.py tldw_Server_API/tests/Writing/test_manuscript_world_plot_db.py -v
```

Expected: PASS for the full touched Writing slice.

Run:
```bash
source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/endpoints/writing.py tldw_Server_API/app/api/v1/endpoints/writing_manuscripts.py tldw_Server_API/app/core/DB_Management/ManuscriptDB.py tldw_Server_API/app/core/Writing/manuscript_analysis.py -f json -o /tmp/bandit_writing_backend_remediation.json
```

Expected: Bandit completes successfully, and any new findings in the touched code are addressed before claiming completion.

- [ ] **Step 5: Commit the analysis and final remediation pass**

```bash
git add tldw_Server_API/app/api/v1/endpoints/writing_manuscripts.py tldw_Server_API/app/core/DB_Management/ManuscriptDB.py tldw_Server_API/app/core/Writing/manuscript_analysis.py tldw_Server_API/tests/Writing/test_manuscript_analysis_integration.py tldw_Server_API/tests/Writing/test_manuscript_analysis_service.py
git commit -m "fix: remediate writing backend review findings"
```
