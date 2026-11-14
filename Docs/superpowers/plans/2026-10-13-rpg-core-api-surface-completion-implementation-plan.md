# RPG Core API Surface Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the 12 Phase-1 REST routes (8 reads, 4 mutations) to the RPG module: list/read endpoints, archive transitions, authority-settings mutation, and snapshot rebuild.

**Architecture:** Purely additive. Repository query/CAS methods in `RPG_DB.py`, thin validated service wrappers in `core/RPG/service.py`, routes in `endpoints/rpg.py` reusing the existing guard builders and error mapper, Pydantic models in `rpg_schemas.py`. No DDL, reducer, event-registry, or permission-catalog changes. Every mutation follows the module's established pattern: idempotency replay check before CAS check, single transaction, `operation_scope` formatted `{entity}:{id}:{action}`.

**Tech Stack:** FastAPI, Pydantic v2 (`ConfigDict(extra="forbid")`), SQLite via `CharactersRAGDB.transaction()`, pytest.

**Spec:** `Docs/superpowers/specs/2026-10-11-rpg-core-api-surface-completion-design.md` — the plan argues from the spec; executors read both.

## Global Constraints

- Activate the project venv before any python/pytest command: `source .venv/bin/activate` (repo root).
- All tests run from the repo root: `python -m pytest tldw_Server_API/tests/RPG/<file> -v`.
- Every new Pydantic request model uses `model_config = ConfigDict(extra="forbid")`.
- Idempotency scopes, verbatim: `campaign:{campaign_id}:update`, `session:{session_id}:update`, `session:{session_id}:authority_settings`, `session:{session_id}:snapshot_rebuild`.
- Status values: `active`, `archived` only. Title 1–200 chars; description ≤ 4000.
- Events listing: `after_sequence` ≥ 0 default 0, `limit` 1–100 default 50, ascending. Entity lists: `limit` 1–100 default 25, `offset` ≥ 0; ordering `updated_at DESC, id DESC` (proposals `created_at DESC, id DESC`).
- Idempotency-Key is a bare header; the service raises `RPGConflictError("idempotency_key_required")` when empty. No length bound.
- Backlog task: TASK-13561 (update notes as work proceeds). ADR: none required (recorded in task).
- Never weaken an existing test. If a test conflicts with the spec, stop and escalate.

---

### Task 1: Repository list/read queries

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/RPG_DB.py` (add methods after `get_proposal`, ~line 665)
- Test: `tldw_Server_API/tests/RPG/test_rpg_db.py` (append)

**Interfaces:**
- Consumes: existing `_campaign_from_row`, `_session_from_row`, `_event_from_row`, `_proposal_from_row` (RPG_DB.py:932-1034), `get_session`, `RPGNotFoundError` import (already present).
- Produces (later tasks call these exactly):
  - `list_campaigns(self, owner_user_id: int, *, status: str | None = None, limit: int = 25, offset: int = 0) -> tuple[list[RPGCampaign], int]`
  - `list_sessions(self, owner_user_id: int, *, campaign_id: int | None = None, status: str | None = None, limit: int = 25, offset: int = 0) -> tuple[list[RPGSession], int]`
  - `count_campaign_sessions(self, owner_user_id: int, campaign_id: int) -> int`
  - `count_pending_proposals(self, owner_user_id: int, session_id: int) -> int`
  - `list_events(self, owner_user_id: int, session_id: int, *, after_sequence: int = 0, limit: int = 50, source_type: str | None = None, event_type: str | None = None) -> list[RPGSessionEvent]`
  - `list_proposals(self, owner_user_id: int, session_id: int, *, status: str | None = None, limit: int = 25, offset: int = 0) -> tuple[list[RPGProposalRecord], int]`
  - `get_snapshot_by_version(self, owner_user_id: int, session_id: int, snapshot_version: int) -> RPGSnapshotRecord` — raises `RPGNotFoundError("snapshot_not_found")`

- [ ] **Step 1: Write the failing tests**

Append to `tests/RPG/test_rpg_db.py` (fixtures `_repo`, `_campaign`, `_session` already exist at the top of that file):

```python
def test_list_campaigns_filters_status_and_paginates():
    repo = _repo()
    first = _campaign(repo)
    second = repo.create_campaign(
        owner_user_id=42, title="Second", description=None,
        default_adapter_key="fate", default_adapter_version="1.0.0",
        settings={}, linked_rules_pack_refs=[],
        idempotency_key="campaign-2", request_payload_hash="hash-2", source_type="user",
    )
    repo.update_campaign_status_direct = None  # guard: no such helper; status set via Task 2
    listed, total = repo.list_campaigns(owner_user_id=42)
    assert total == 2  # nosec B101
    assert [c.id for c in listed] == [second.id, first.id]  # nosec B101  (updated_at DESC, id DESC)
    empty, empty_total = repo.list_campaigns(owner_user_id=42, status="archived")
    assert empty == [] and empty_total == 0  # nosec B101
    page, page_total = repo.list_campaigns(owner_user_id=42, limit=1, offset=1)
    assert page_total == 2 and [c.id for c in page] == [first.id]  # nosec B101
    foreign, foreign_total = repo.list_campaigns(owner_user_id=7)
    assert foreign == [] and foreign_total == 0  # nosec B101


def test_list_sessions_filters_and_counts():
    repo = _repo()
    campaign = _campaign(repo)
    session = _session(repo, campaign.id)
    listed, total = repo.list_sessions(owner_user_id=42)
    assert total == 1 and listed[0].id == session.id  # nosec B101
    assert repo.list_sessions(owner_user_id=42, campaign_id=campaign.id)[1] == 1  # nosec B101
    assert repo.list_sessions(owner_user_id=42, campaign_id=999)[1] == 0  # nosec B101
    assert repo.count_campaign_sessions(owner_user_id=42, campaign_id=campaign.id) == 1  # nosec B101
    assert repo.count_pending_proposals(owner_user_id=42, session_id=session.id) == 0  # nosec B101


def test_list_events_cursors_and_filters():
    repo = _repo()
    campaign = _campaign(repo)
    session = _session(repo, campaign.id)
    for seq, (event_type, source) in enumerate(
        [("note.added", "user"), ("note.added", "mcp"), ("scene.updated", "user")], start=1
    ):
        repo.commit_events_and_snapshot(
            owner_user_id=42, session_id=session.id,
            expected_last_event_sequence=seq - 1, base_snapshot_version=seq - 1,
            events=[{"event_type": event_type, "event_payload": {"note_id": f"n{seq}", "scene_id": f"s{seq}"}, "source_type": source}],
            snapshot={}, diagnostics={}, idempotency_key=f"evt-{seq}", request_payload_hash=f"h{seq}",
            adapter_key=session.adapter_key, adapter_version=session.adapter_version, proposal_id=None,
        )
    events = repo.list_events(owner_user_id=42, session_id=session.id)
    assert [e.sequence_number for e in events] == [1, 2, 3]  # nosec B101
    assert [e.sequence_number for e in repo.list_events(owner_user_id=42, session_id=session.id, after_sequence=1)] == [2, 3]  # nosec B101
    assert len(repo.list_events(owner_user_id=42, session_id=session.id, source_type="mcp")) == 1  # nosec B101
    assert len(repo.list_events(owner_user_id=42, session_id=session.id, event_type="scene.updated")) == 1  # nosec B101


def test_list_proposals_and_get_snapshot_by_version():
    repo = _repo()
    campaign = _campaign(repo)
    session = _session(repo, campaign.id)
    assert repo.list_proposals(owner_user_id=42, session_id=session.id) == ([], 0)  # nosec B101
    record = repo.get_snapshot_by_version(owner_user_id=42, session_id=session.id, snapshot_version=0)
    assert record.snapshot_version == 0  # nosec B101
    with pytest.raises(RPGNotFoundError):
        repo.get_snapshot_by_version(owner_user_id=42, session_id=session.id, snapshot_version=99)
```

Note: `commit_events_and_snapshot` validates envelopes upstream in the service, not in the repo — the raw dicts above pass because the repo only requires `event_type`, `event_payload`, `source_type` keys. If the repo rejects `snapshot={}` (empty dict is valid JSON), keep it.

Remove the guard line `repo.update_campaign_status_direct = None` before running — it exists only to flag that status filtering is tested via a directly-set value; set archived status in a second test after Task 2, or execute `UPDATE rpg_campaigns SET status='archived'` inline via `repo.db.transaction()`:

```python
    with repo.db.transaction() as conn:
        conn.execute("UPDATE rpg_campaigns SET status = 'archived' WHERE id = ?", (second.id,))
    archived, archived_total = repo.list_campaigns(owner_user_id=42, status="archived")
    assert archived_total == 1 and archived[0].id == second.id  # nosec B101
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tldw_Server_API/tests/RPG/test_rpg_db.py -v -k "list_campaigns or list_sessions or list_events or list_proposals"`
Expected: FAIL with `AttributeError: 'RPGRepository' object has no attribute 'list_campaigns'`

- [ ] **Step 3: Implement the repository methods**

Add to `RPG_DB.py` after `get_proposal` (~line 665):

```python
    def list_campaigns(
        self,
        owner_user_id: int,
        *,
        status: str | None = None,
        limit: int = 25,
        offset: int = 0,
    ) -> tuple[list[RPGCampaign], int]:
        clauses = ["owner_user_id = ?"]
        params: list[Any] = [owner_user_id]
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        where = " AND ".join(clauses)
        with self.db.transaction() as conn:
            rows = conn.execute(
                f"SELECT * FROM rpg_campaigns WHERE {where} ORDER BY updated_at DESC, id DESC LIMIT ? OFFSET ?",
                (*params, limit, offset),
            ).fetchall()
            total = int(conn.execute(f"SELECT COUNT(*) FROM rpg_campaigns WHERE {where}", params).fetchone()[0])
        return [self._campaign_from_row(row) for row in rows], total

    def list_sessions(
        self,
        owner_user_id: int,
        *,
        campaign_id: int | None = None,
        status: str | None = None,
        limit: int = 25,
        offset: int = 0,
    ) -> tuple[list[RPGSession], int]:
        clauses = ["owner_user_id = ?"]
        params: list[Any] = [owner_user_id]
        if campaign_id is not None:
            clauses.append("campaign_id = ?")
            params.append(campaign_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        where = " AND ".join(clauses)
        with self.db.transaction() as conn:
            rows = conn.execute(
                f"SELECT * FROM rpg_sessions WHERE {where} ORDER BY updated_at DESC, id DESC LIMIT ? OFFSET ?",
                (*params, limit, offset),
            ).fetchall()
            total = int(conn.execute(f"SELECT COUNT(*) FROM rpg_sessions WHERE {where}", params).fetchone()[0])
        return [self._session_from_row(row) for row in rows], total

    def count_campaign_sessions(self, owner_user_id: int, campaign_id: int) -> int:
        with self.db.transaction() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM rpg_sessions WHERE owner_user_id = ? AND campaign_id = ?",
                (owner_user_id, campaign_id),
            ).fetchone()
        return int(row[0])

    def count_pending_proposals(self, owner_user_id: int, session_id: int) -> int:
        with self.db.transaction() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM rpg_session_proposals WHERE owner_user_id = ? AND session_id = ? AND status = 'pending'",
                (owner_user_id, session_id),
            ).fetchone()
        return int(row[0])

    def list_events(
        self,
        owner_user_id: int,
        session_id: int,
        *,
        after_sequence: int = 0,
        limit: int = 50,
        source_type: str | None = None,
        event_type: str | None = None,
    ) -> list[RPGSessionEvent]:
        clauses = ["owner_user_id = ?", "session_id = ?", "sequence_number > ?"]
        params: list[Any] = [owner_user_id, session_id, after_sequence]
        if source_type is not None:
            clauses.append("source_type = ?")
            params.append(source_type)
        if event_type is not None:
            clauses.append("event_type = ?")
            params.append(event_type)
        with self.db.transaction() as conn:
            rows = conn.execute(
                f"SELECT * FROM rpg_session_events WHERE {' AND '.join(clauses)} ORDER BY sequence_number ASC LIMIT ?",
                (*params, limit),
            ).fetchall()
        return [self._event_from_row(row) for row in rows]

    def list_proposals(
        self,
        owner_user_id: int,
        session_id: int,
        *,
        status: str | None = None,
        limit: int = 25,
        offset: int = 0,
    ) -> tuple[list[RPGProposalRecord], int]:
        clauses = ["owner_user_id = ?", "session_id = ?"]
        params: list[Any] = [owner_user_id, session_id]
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        where = " AND ".join(clauses)
        with self.db.transaction() as conn:
            rows = conn.execute(
                f"SELECT * FROM rpg_session_proposals WHERE {where} ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?",
                (*params, limit, offset),
            ).fetchall()
            total = int(conn.execute(f"SELECT COUNT(*) FROM rpg_session_proposals WHERE {where}", params).fetchone()[0])
        return [self._proposal_from_row(row) for row in rows], total

    def get_snapshot_by_version(
        self,
        owner_user_id: int,
        session_id: int,
        snapshot_version: int,
    ) -> RPGSnapshotRecord:
        with self.db.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM rpg_session_snapshots WHERE owner_user_id = ? AND session_id = ? AND snapshot_version = ?",
                (owner_user_id, session_id, snapshot_version),
            ).fetchone()
        if row is None:
            raise RPGNotFoundError("snapshot_not_found")
        return self._snapshot_from_row(row)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tldw_Server_API/tests/RPG/test_rpg_db.py -v`
Expected: PASS (all existing + new)

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/DB_Management/RPG_DB.py tldw_Server_API/tests/RPG/test_rpg_db.py
git commit -m "feat(rpg): add repository list/read queries (TASK-13561)"
```

---

### Task 2: Repository metadata updates (PATCH CAS)

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/RPG_DB.py` (result dataclasses near `CommitEventsResult`; methods after Task 1's)
- Test: `tldw_Server_API/tests/RPG/test_rpg_db.py` (append)

**Interfaces:**
- Consumes: `_find_idempotency_record`, `_insert_idempotency_record`, `_ensure_replay_hash`, `_get_campaign_with_conn`, `_get_session_with_conn`, `_now` (all existing).
- Produces:
  - `@dataclass(frozen=True, slots=True) class CampaignUpdateResult: campaign: RPGCampaign; replayed: bool`
  - `@dataclass(frozen=True, slots=True) class SessionUpdateResult: session: RPGSession; replayed: bool`
  - `update_campaign(self, owner_user_id, campaign_id, *, title: str | None = None, description: str | None = None, status: str | None = None, expected_version: int, idempotency_key: str, request_payload_hash: str, source_type: str) -> CampaignUpdateResult`
  - `update_session(self, owner_user_id, session_id, *, title: str | None = None, status: str | None = None, expected_version: int, idempotency_key: str, request_payload_hash: str, source_type: str) -> SessionUpdateResult`
  - Both raise `RPGConflictError("stale_version")` on CAS mismatch and replay stored results on retry with the same key + hash.

- [ ] **Step 1: Write the failing tests**

```python
def test_update_campaign_metadata_with_cas_and_replay():
    repo = _repo()
    campaign = _campaign(repo)
    result = repo.update_campaign(
        owner_user_id=42, campaign_id=campaign.id,
        title="Renamed", description="Now with words", status="archived",
        expected_version=campaign.version, idempotency_key="upd-1", request_payload_hash="h-upd-1",
        source_type="user",
    )
    assert result.replayed is False  # nosec B101
    assert result.campaign.title == "Renamed" and result.campaign.status == "archived"  # nosec B101
    assert result.campaign.version == campaign.version + 1  # nosec B101
    replay = repo.update_campaign(
        owner_user_id=42, campaign_id=campaign.id,
        title="Renamed", description="Now with words", status="archived",
        expected_version=campaign.version, idempotency_key="upd-1", request_payload_hash="h-upd-1",
        source_type="user",
    )
    assert replay.replayed is True and replay.campaign.id == campaign.id  # nosec B101
    with pytest.raises(RPGConflictError):
        repo.update_campaign(
            owner_user_id=42, campaign_id=campaign.id, title="X",
            expected_version=campaign.version, idempotency_key="upd-2",
            request_payload_hash="h-upd-2", source_type="user",
        )
    with pytest.raises(RPGConflictError):  # same key, different payload
        repo.update_campaign(
            owner_user_id=42, campaign_id=campaign.id, title="Y",
            expected_version=campaign.version, idempotency_key="upd-1",
            request_payload_hash="h-DIFFERENT", source_type="user",
        )
    with pytest.raises(RPGNotFoundError):
        repo.update_campaign(
            owner_user_id=7, campaign_id=campaign.id, title="Z",
            expected_version=1, idempotency_key="upd-3", request_payload_hash="h-3", source_type="user",
        )


def test_update_session_metadata_partial_and_cas():
    repo = _repo()
    campaign = _campaign(repo)
    session = _session(repo, campaign.id)
    result = repo.update_session(
        owner_user_id=42, session_id=session.id, title="Act II", status="archived",
        expected_version=session.version, idempotency_key="s-upd-1",
        request_payload_hash="h-s-1", source_type="user",
    )
    assert result.session.title == "Act II" and result.session.status == "archived"  # nosec B101
    untouched = repo.update_session(
        owner_user_id=42, session_id=session.id,
        expected_version=result.session.version, idempotency_key="s-upd-2",
        request_payload_hash="h-s-2", source_type="user",
    )
    assert untouched.session.title == "Act II"  # nosec B101  (COALESCE keeps omitted fields)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tldw_Server_API/tests/RPG/test_rpg_db.py -v -k update_`
Expected: FAIL with `AttributeError: ... no attribute 'update_campaign'`

- [ ] **Step 3: Implement**

Dataclasses next to `CommitEventsResult` (top of RPG_DB.py, after the existing result dataclasses):

```python
@dataclass(frozen=True, slots=True)
class CampaignUpdateResult:
    campaign: RPGCampaign
    replayed: bool


@dataclass(frozen=True, slots=True)
class SessionUpdateResult:
    session: RPGSession
    replayed: bool
```

Methods (after Task 1 methods):

```python
    def update_campaign(
        self,
        owner_user_id: int,
        campaign_id: int,
        *,
        title: str | None = None,
        description: str | None = None,
        status: str | None = None,
        expected_version: int,
        idempotency_key: str,
        request_payload_hash: str,
        source_type: str,
    ) -> CampaignUpdateResult:
        self._validate_source_type(source_type)
        operation_scope = f"campaign:{campaign_id}:update"
        now = self._now()
        with self.db.transaction() as conn:
            replay = self._find_idempotency_record(
                conn, owner_user_id=owner_user_id, session_id=None, source_type=source_type,
                operation_scope=operation_scope, idempotency_key=idempotency_key,
            )
            if replay is not None:
                self._ensure_replay_hash(replay, request_payload_hash)
                replayed_id = int(self._from_json(self._row_value(replay, "response_json"))["campaign_id"])
                return CampaignUpdateResult(campaign=self._get_campaign_with_conn(conn, owner_user_id, replayed_id), replayed=True)

            campaign = self._get_campaign_with_conn(conn, owner_user_id, campaign_id)
            if campaign.version != expected_version:
                raise RPGConflictError("stale_version")
            cursor = conn.execute(
                """
                UPDATE rpg_campaigns
                SET title = COALESCE(?, title), description = COALESCE(?, description),
                    status = COALESCE(?, status), version = version + 1, updated_at = ?
                WHERE id = ? AND owner_user_id = ? AND version = ?
                """,
                (title, description, status, now, campaign_id, owner_user_id, expected_version),
            )
            if cursor.rowcount != 1:
                raise RPGConflictError("stale_version")
            self._insert_idempotency_record(
                conn, owner_user_id=owner_user_id, session_id=None, source_type=source_type,
                operation_scope=operation_scope, idempotency_key=idempotency_key,
                request_payload_hash=request_payload_hash, event_ids=[],
                response={"campaign_id": campaign_id}, created_at=now,
            )
            return CampaignUpdateResult(campaign=self._get_campaign_with_conn(conn, owner_user_id, campaign_id), replayed=False)

    def update_session(
        self,
        owner_user_id: int,
        session_id: int,
        *,
        title: str | None = None,
        status: str | None = None,
        expected_version: int,
        idempotency_key: str,
        request_payload_hash: str,
        source_type: str,
    ) -> SessionUpdateResult:
        self._validate_source_type(source_type)
        operation_scope = f"session:{session_id}:update"
        now = self._now()
        with self.db.transaction() as conn:
            replay = self._find_idempotency_record(
                conn, owner_user_id=owner_user_id, session_id=session_id, source_type=source_type,
                operation_scope=operation_scope, idempotency_key=idempotency_key,
            )
            if replay is not None:
                self._ensure_replay_hash(replay, request_payload_hash)
                replayed_id = int(self._from_json(self._row_value(replay, "response_json"))["session_id"])
                return SessionUpdateResult(session=self._get_session_with_conn(conn, owner_user_id, replayed_id), replayed=True)

            session = self._get_session_with_conn(conn, owner_user_id, session_id)
            if session.version != expected_version:
                raise RPGConflictError("stale_version")
            cursor = conn.execute(
                """
                UPDATE rpg_sessions
                SET title = COALESCE(?, title), status = COALESCE(?, status),
                    version = version + 1, updated_at = ?
                WHERE id = ? AND owner_user_id = ? AND version = ?
                """,
                (title, status, now, session_id, owner_user_id, expected_version),
            )
            if cursor.rowcount != 1:
                raise RPGConflictError("stale_version")
            self._insert_idempotency_record(
                conn, owner_user_id=owner_user_id, session_id=session_id, source_type=source_type,
                operation_scope=operation_scope, idempotency_key=idempotency_key,
                request_payload_hash=request_payload_hash, event_ids=[],
                response={"session_id": session_id}, created_at=now,
            )
            return SessionUpdateResult(session=self._get_session_with_conn(conn, owner_user_id, session_id), replayed=False)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tldw_Server_API/tests/RPG/test_rpg_db.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/DB_Management/RPG_DB.py tldw_Server_API/tests/RPG/test_rpg_db.py
git commit -m "feat(rpg): add campaign/session metadata CAS updates (TASK-13561)"
```

---

### Task 3: Repository authority-settings replace

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/RPG_DB.py` (after `update_session`)
- Test: `tldw_Server_API/tests/RPG/test_rpg_db.py` (append)

**Interfaces:**
- Consumes: Task 2's `SessionUpdateResult`; existing helpers.
- Produces: `update_session_authority_settings(self, owner_user_id: int, session_id: int, *, authority_settings: dict[str, Any], expected_version: int, idempotency_key: str, request_payload_hash: str, source_type: str) -> SessionUpdateResult` — scope `session:{session_id}:authority_settings`, same CAS/replay/error contract as Task 2. Validation of the settings dict happens in the service (Task 7); the repo stores what it is given.

- [ ] **Step 1: Write the failing test**

```python
def test_update_session_authority_settings_cas_and_replay():
    repo = _repo()
    campaign = _campaign(repo)
    session = _session(repo, campaign.id)
    settings = {"import_auto_commit": True, "mcp_auto_commit": False, "model_auto_commit": False, "model_auto_commit_event_types": []}
    result = repo.update_session_authority_settings(
        owner_user_id=42, session_id=session.id, authority_settings=settings,
        expected_version=session.version, idempotency_key="auth-1",
        request_payload_hash="h-auth-1", source_type="user",
    )
    assert result.session.authority_settings["import_auto_commit"] is True  # nosec B101
    assert result.session.version == session.version + 1  # nosec B101
    replay = repo.update_session_authority_settings(
        owner_user_id=42, session_id=session.id, authority_settings=settings,
        expected_version=session.version, idempotency_key="auth-1",
        request_payload_hash="h-auth-1", source_type="user",
    )
    assert replay.replayed is True  # nosec B101
    with pytest.raises(RPGConflictError):
        repo.update_session_authority_settings(
            owner_user_id=42, session_id=session.id, authority_settings=settings,
            expected_version=session.version, idempotency_key="auth-2",
            request_payload_hash="h-auth-2", source_type="user",
        )
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tldw_Server_API/tests/RPG/test_rpg_db.py -v -k authority`
Expected: FAIL `AttributeError`

- [ ] **Step 3: Implement**

```python
    def update_session_authority_settings(
        self,
        owner_user_id: int,
        session_id: int,
        *,
        authority_settings: dict[str, Any],
        expected_version: int,
        idempotency_key: str,
        request_payload_hash: str,
        source_type: str,
    ) -> SessionUpdateResult:
        self._validate_source_type(source_type)
        operation_scope = f"session:{session_id}:authority_settings"
        now = self._now()
        with self.db.transaction() as conn:
            replay = self._find_idempotency_record(
                conn, owner_user_id=owner_user_id, session_id=session_id, source_type=source_type,
                operation_scope=operation_scope, idempotency_key=idempotency_key,
            )
            if replay is not None:
                self._ensure_replay_hash(replay, request_payload_hash)
                replayed_id = int(self._from_json(self._row_value(replay, "response_json"))["session_id"])
                return SessionUpdateResult(session=self._get_session_with_conn(conn, owner_user_id, replayed_id), replayed=True)

            session = self._get_session_with_conn(conn, owner_user_id, session_id)
            if session.version != expected_version:
                raise RPGConflictError("stale_version")
            cursor = conn.execute(
                """
                UPDATE rpg_sessions
                SET authority_settings_json = ?, version = version + 1, updated_at = ?
                WHERE id = ? AND owner_user_id = ? AND version = ?
                """,
                (self._to_json(authority_settings), now, session_id, owner_user_id, expected_version),
            )
            if cursor.rowcount != 1:
                raise RPGConflictError("stale_version")
            self._insert_idempotency_record(
                conn, owner_user_id=owner_user_id, session_id=session_id, source_type=source_type,
                operation_scope=operation_scope, idempotency_key=idempotency_key,
                request_payload_hash=request_payload_hash, event_ids=[],
                response={"session_id": session_id}, created_at=now,
            )
            return SessionUpdateResult(session=self._get_session_with_conn(conn, owner_user_id, session_id), replayed=False)
```

- [ ] **Step 4: Run tests** — `python -m pytest tldw_Server_API/tests/RPG/test_rpg_db.py -v` — Expected: PASS

- [ ] **Step 5: Commit** — `git commit -m "feat(rpg): add authority-settings repository replace (TASK-13561)"` (add both files)

---

### Task 4: Repository snapshot rebuild

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/RPG_DB.py` (result dataclass + method; new import)
- Test: `tldw_Server_API/tests/RPG/test_rpg_db.py` (append)

**Interfaces:**
- Consumes: existing helpers; new import `from tldw_Server_API.app.core.RPG.reducer import reduce_events` (safe: reducer imports only models/constants — no cycle); `RPG_REDUCER_VERSION`, `RPG_SNAPSHOT_SCHEMA_VERSION`, `RPGSnapshotState` (already imported); `asdict` (already imported from dataclasses).
- Produces:
  - `@dataclass(frozen=True, slots=True) class SnapshotRebuildResult: snapshot: RPGSnapshotRecord; event_count: int; replayed: bool`
  - `rebuild_session_snapshot(self, owner_user_id: int, session_id: int, *, idempotency_key: str, request_payload_hash: str, source_type: str) -> SnapshotRebuildResult` — scope `session:{session_id}:snapshot_rebuild`; raises `RPGConflictError("snapshot_version_conflict")` on `sqlite3.IntegrityError`; diagnostics `{"rebuilt": True, "event_count": N, "previous_snapshot_version": v}`.

- [ ] **Step 1: Write the failing test**

```python
def test_rebuild_session_snapshot_replays_full_log():
    repo = _repo()
    campaign = _campaign(repo)
    session = _session(repo, campaign.id)
    for seq in (1, 2):
        repo.commit_events_and_snapshot(
            owner_user_id=42, session_id=session.id,
            expected_last_event_sequence=seq - 1, base_snapshot_version=seq - 1,
            events=[{"event_type": "note.added", "event_payload": {"note_id": f"n{seq}"}, "source_type": "user"}],
            snapshot={}, diagnostics={}, idempotency_key=f"rb-evt-{seq}", request_payload_hash=f"rh{seq}",
            adapter_key=session.adapter_key, adapter_version=session.adapter_version, proposal_id=None,
        )
    result = repo.rebuild_session_snapshot(
        owner_user_id=42, session_id=session.id,
        idempotency_key="rebuild-1", request_payload_hash="h-rebuild-1", source_type="user",
    )
    assert result.replayed is False and result.event_count == 2  # nosec B101
    assert result.snapshot.snapshot_version == 3  # nosec B101  (v0 seed + 2 commits -> rebuild writes v3)
    assert result.snapshot.last_event_sequence == 2  # nosec B101
    assert result.snapshot.diagnostics_json["rebuilt"] is True  # nosec B101
    notes = result.snapshot.snapshot_json["notes"]
    assert len(notes) == 2  # nosec B101  (full-log fold reproduces appended notes)
    replay = repo.rebuild_session_snapshot(
        owner_user_id=42, session_id=session.id,
        idempotency_key="rebuild-1", request_payload_hash="h-rebuild-1", source_type="user",
    )
    assert replay.replayed is True and replay.snapshot.snapshot_version == 3  # nosec B101
```

- [ ] **Step 2: Run to verify it fails** — `python -m pytest tldw_Server_API/tests/RPG/test_rpg_db.py -v -k rebuild` — Expected: FAIL `AttributeError`

- [ ] **Step 3: Implement**

```python
    def rebuild_session_snapshot(
        self,
        owner_user_id: int,
        session_id: int,
        *,
        idempotency_key: str,
        request_payload_hash: str,
        source_type: str,
    ) -> SnapshotRebuildResult:
        self._validate_source_type(source_type)
        operation_scope = f"session:{session_id}:snapshot_rebuild"
        now = self._now()
        try:
            with self.db.transaction() as conn:
                replay = self._find_idempotency_record(
                    conn, owner_user_id=owner_user_id, session_id=session_id, source_type=source_type,
                    operation_scope=operation_scope, idempotency_key=idempotency_key,
                )
                if replay is not None:
                    self._ensure_replay_hash(replay, request_payload_hash)
                    response = self._from_json(self._row_value(replay, "response_json"))
                    return SnapshotRebuildResult(
                        snapshot=self.get_snapshot_by_version(owner_user_id, session_id, int(response["snapshot_version"])),
                        event_count=int(response["event_count"]),
                        replayed=True,
                    )

                session = self._get_session_with_conn(conn, owner_user_id, session_id)
                rows = conn.execute(
                    "SELECT * FROM rpg_session_events WHERE owner_user_id = ? AND session_id = ? ORDER BY sequence_number ASC",
                    (owner_user_id, session_id),
                ).fetchall()
                events = [self._event_from_row(row) for row in rows]
                reducer_events = [{"event_type": e.event_type, "event_payload": e.event_payload} for e in events]
                state = reduce_events(RPGSnapshotState(), reducer_events)
                next_version = session.current_snapshot_version + 1
                last_sequence = events[-1].sequence_number if events else session.last_event_sequence
                diagnostics = {
                    "rebuilt": True,
                    "event_count": len(events),
                    "previous_snapshot_version": session.current_snapshot_version,
                }
                conn.execute(
                    """
                    INSERT INTO rpg_session_snapshots (
                        session_id, owner_user_id, snapshot_version, last_event_sequence,
                        reducer_version, snapshot_schema_version, snapshot_json, diagnostics_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id, owner_user_id, next_version, last_sequence,
                        RPG_REDUCER_VERSION, RPG_SNAPSHOT_SCHEMA_VERSION,
                        self._to_json(asdict(state)), self._to_json(diagnostics), now,
                    ),
                )
                cursor = conn.execute(
                    """
                    UPDATE rpg_sessions
                    SET last_event_sequence = MAX(last_event_sequence, ?), current_snapshot_version = ?,
                        version = version + 1, updated_at = ?
                    WHERE id = ? AND owner_user_id = ? AND current_snapshot_version = ?
                    """,
                    (last_sequence, next_version, now, session_id, owner_user_id, session.current_snapshot_version),
                )
                if cursor.rowcount != 1:
                    raise RPGConflictError("stale_session_cursor")
                self._insert_idempotency_record(
                    conn, owner_user_id=owner_user_id, session_id=session_id, source_type=source_type,
                    operation_scope=operation_scope, idempotency_key=idempotency_key,
                    request_payload_hash=request_payload_hash, event_ids=[],
                    response={"snapshot_version": next_version, "event_count": len(events)}, created_at=now,
                )
                return SnapshotRebuildResult(
                    snapshot=self.get_snapshot_by_version(owner_user_id, session_id, next_version),
                    event_count=len(events),
                    replayed=False,
                )
        except sqlite3.IntegrityError as exc:
            raise RPGConflictError("snapshot_version_conflict") from exc
```

Note: `get_snapshot_by_version` opens its own transaction — inside `self.db.transaction()` this nests on the same connection (SQLite nested `with` on one connection is a no-op savepoint-wise here; CharactersRAGDB transactions are per-call). If the nested transaction errors in practice, replace the two `get_snapshot_by_version` calls with a private `_snapshot_by_version_with_conn(conn, owner_user_id, session_id, version)` helper and use it in both methods.

- [ ] **Step 4: Run tests** — `python -m pytest tldw_Server_API/tests/RPG/test_rpg_db.py -v` — Expected: PASS

- [ ] **Step 5: Commit** — `git commit -m "feat(rpg): add transactional snapshot rebuild (TASK-13561)"` (add both files)

---

### Task 5: Pydantic schemas

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/rpg_schemas.py` (append)
- Test: `tldw_Server_API/tests/RPG/test_rpg_schemas.py` (create)

**Interfaces:**
- Consumes: existing `RPGCampaignResponse`, `RPGSessionResponse`, `ConfigDict`, `Field` imports (add `Literal`, `model_validator` to imports).
- Produces (used by Tasks 8-9):
  - `RPGCampaignUpdateRequest` — `title: str | None` (1-200), `description: str | None` (≤4000), `status: Literal["active","archived"] | None`, `expected_version: int (ge=1)`; requires ≥1 of title/description/status via model_validator (`ValueError("no_mutable_fields")`).
  - `RPGSessionUpdateRequest` — `title: str | None` (1-200), `status: Literal[...] | None`, `expected_version: int (ge=1)`; same validator.
  - `RPGAuthoritySettingsReplaceRequest` — `import_auto_commit: bool`, `mcp_auto_commit: bool`, `model_auto_commit: bool`, `model_auto_commit_event_types: list[str]` (each 1-120 chars), `expected_version: int (ge=1)`.
  - `RPGCampaignListItem(RPGCampaignResponse)` + `created_at: datetime`, `updated_at: datetime`.
  - `RPGCampaignDetail(RPGCampaignListItem)` + `session_count: int`.
  - `RPGSessionListItem(RPGSessionResponse)` + `pending_proposal_count: int`, `created_at: datetime`, `updated_at: datetime`.
  - `RPGSessionDetail(RPGSessionListItem)` + `authority_settings: dict[str, Any]`, `linked_chat_id: int | None`.
  - `RPGCampaignListResponse` — `items: list[RPGCampaignListItem]`, `total_count: int`, `limit: int`, `offset: int`.
  - `RPGSessionListResponse` — `items: list[RPGSessionListItem]`, `total_count: int`, `limit: int`, `offset: int`.
  - `RPGEventsListResponse` — `items: list[RPGEventResponse]`, `next_after_sequence: int`, `limit: int`.
  - `RPGEventResponse` — mirrors `RPGSessionEvent`: `id, session_id, sequence_number, event_type, event_payload: dict, source_type, source_actor_id, source_label, operation_id, event_schema_version, adapter_key, adapter_version, proposal_id, created_at: datetime`.
  - `RPGProposalListResponse` — `items: list[dict[str, Any]]`, `total_count: int`, `limit: int`, `offset: int` (items serialized by the endpoint's existing `_proposal_payload` helper, matching `RPGRecordEventsResponse.proposal`).
  - `RPGSnapshotResponse` — `snapshot_version: int`, `last_event_sequence: int`, `snapshot: dict[str, Any]`, `diagnostics: dict[str, Any]`.
  - `RPGSnapshotRebuildResponse` — `snapshot_version: int`, `last_event_sequence: int`, `event_count: int`, `replayed: bool`, `diagnostics: dict[str, Any]`.
  - All request models `extra="forbid"`.

- [ ] **Step 1: Write the failing tests** (new file `tests/RPG/test_rpg_schemas.py`)

```python
import pytest
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.schemas.rpg_schemas import (
    RPGAuthoritySettingsReplaceRequest,
    RPGCampaignUpdateRequest,
    RPGSessionUpdateRequest,
)

pytestmark = pytest.mark.unit


def test_campaign_update_requires_one_mutable_field():
    with pytest.raises(ValidationError):
        RPGCampaignUpdateRequest(expected_version=1)
    ok = RPGCampaignUpdateRequest(title="New", expected_version=1)
    assert ok.title == "New"  # nosec B101


def test_campaign_update_rejects_unknown_status_and_extra_fields():
    with pytest.raises(ValidationError):
        RPGCampaignUpdateRequest(status="deleted", expected_version=1)
    with pytest.raises(ValidationError):
        RPGCampaignUpdateRequest(title="New", expected_version=1, owner_user_id=7)


def test_session_update_bounds():
    with pytest.raises(ValidationError):
        RPGSessionUpdateRequest(title="", expected_version=1)
    with pytest.raises(ValidationError):
        RPGSessionUpdateRequest(title="x" * 201, expected_version=1)
    assert RPGSessionUpdateRequest(status="archived", expected_version=3).status == "archived"  # nosec B101


def test_authority_settings_request_strict_shape():
    base = {"import_auto_commit": False, "mcp_auto_commit": False, "model_auto_commit": True, "model_auto_commit_event_types": ["note.added"]}
    assert RPGAuthoritySettingsReplaceRequest(**base, expected_version=2).model_auto_commit is True  # nosec B101
    with pytest.raises(ValidationError):  # extra field
        RPGAuthoritySettingsReplaceRequest(**base, expected_version=2, surprise=1)
    with pytest.raises(ValidationError):  # missing field
        RPGAuthoritySettingsReplaceRequest(import_auto_commit=False, mcp_auto_commit=False, model_auto_commit=False, expected_version=2)
```

- [ ] **Step 2: Run to verify they fail** — `python -m pytest tldw_Server_API/tests/RPG/test_rpg_schemas.py -v` — Expected: FAIL `ImportError`

- [ ] **Step 3: Implement** — append the models to `rpg_schemas.py` exactly as listed in Interfaces; imports needed: `from datetime import datetime`, `from typing import Any, Literal`, `from pydantic import model_validator`. Example of the validator pattern:

```python
class RPGCampaignUpdateRequest(BaseModel):
    """Request body for PATCH /rpg/campaigns/{campaign_id}."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=4000)
    status: Literal["active", "archived"] | None = None
    expected_version: int = Field(ge=1)

    @model_validator(mode="after")
    def _require_mutable_field(self) -> "RPGCampaignUpdateRequest":
        if self.title is None and self.description is None and self.status is None:
            raise ValueError("no_mutable_fields")
        return self
```

- [ ] **Step 4: Run tests** — `python -m pytest tldw_Server_API/tests/RPG/test_rpg_schemas.py -v` — Expected: PASS

- [ ] **Step 5: Commit** — `git commit -m "feat(rpg): add Phase-1 request/response schemas (TASK-13561)"` (add both files)

---

### Task 6: Service read methods

**Files:**
- Modify: `tldw_Server_API/app/core/RPG/service.py` (after `get_snapshot`, ~line 401)
- Test: `tldw_Server_API/tests/RPG/test_rpg_service.py` (append)

**Interfaces:**
- Consumes: Task 1 repo methods; existing `SnapshotResult`, `RPGNotFoundError`, `self.repo`, `self.owner_user_id`.
- Produces (exact signatures for Task 8):
  - `list_campaigns(self, *, status: str | None = None, limit: int = 25, offset: int = 0) -> tuple[list[RPGCampaign], int]`
  - `get_campaign_detail(self, campaign_id: int) -> tuple[RPGCampaign, int]` — campaign + session_count; 404 via repo getter.
  - `list_sessions(self, *, campaign_id: int | None = None, status: str | None = None, limit: int = 25, offset: int = 0) -> tuple[list[RPGSession], int]` — when `campaign_id` is provided, calls `self.repo.get_campaign(...)` first so foreign/unknown ids raise `RPGNotFoundError`.
  - `get_session_detail(self, session_id: int) -> tuple[RPGSession, int]` — session + pending_proposal_count.
  - `list_events(self, session_id: int, *, after_sequence: int = 0, limit: int = 50, source_type: str | None = None, event_type: str | None = None) -> list[RPGSessionEvent]`
  - `list_proposals(self, session_id: int, *, status: str | None = None, limit: int = 25, offset: int = 0) -> tuple[list[RPGProposalRecord], int]`
  - `get_snapshot(self, session_id: int, snapshot_version: int | None = None) -> SnapshotResult` — extend the existing method (line 391): `record = self.repo.get_snapshot_by_version(...) if snapshot_version is not None else self.repo.get_latest_snapshot(...)`.

- [ ] **Step 1: Write the failing tests** (append; mirror the file's existing repo/service fixture style — build `RPGService(repo=_repo-like fixture, owner_user_id=42)` the way `test_rpg_service.py` already constructs services)

```python
def test_service_lists_campaigns_and_details():
    service = _service()  # use the file's existing service fixture/builder
    campaign = service.create_campaign("Title", None, "fate", idempotency_key="svc-c-1")
    listed, total = service.list_campaigns()
    assert total >= 1 and any(c.id == campaign.id for c in listed)  # nosec B101
    detail, session_count = service.get_campaign_detail(campaign.id)
    assert detail.id == campaign.id and session_count == 0  # nosec B101
    with pytest.raises(RPGNotFoundError):
        service.get_campaign_detail(999)


def test_service_lists_sessions_with_campaign_ownership_check():
    service = _service()
    campaign = service.create_campaign("Title", None, "fate", idempotency_key="svc-c-2")
    session = asyncio.run(service.create_session(campaign_id=campaign.id, title="S1", adapter_key="fate", idempotency_key="svc-s-2"))
    listed, total = service.list_sessions(campaign_id=campaign.id)
    assert total == 1 and listed[0].id == session.id  # nosec B101
    with pytest.raises(RPGNotFoundError):
        service.list_sessions(campaign_id=999)
    detail, pending = service.get_session_detail(session.id)
    assert detail.id == session.id and pending == 0  # nosec B101


def test_service_lists_events_and_proposals_and_snapshot_versions():
    service = _service()
    campaign = service.create_campaign("Title", None, "fate", idempotency_key="svc-c-3")
    session = asyncio.run(service.create_session(campaign_id=campaign.id, title="S1", adapter_key="fate", idempotency_key="svc-s-3"))
    service.record_events(session.id, [{"event_type": "note.added", "event_payload": {"note_id": "n1"}}], "user", 0, "svc-evt-3")
    events = service.list_events(session.id)
    assert len(events) == 1 and events[0].sequence_number == 1  # nosec B101
    proposals, total = service.list_proposals(session.id)
    assert total == 0  # nosec B101
    latest = service.get_snapshot(session.id)
    assert latest.snapshot_version == 1  # nosec B101
    historical = service.get_snapshot(session.id, snapshot_version=0)
    assert historical.snapshot_version == 0  # nosec B101
    with pytest.raises(RPGNotFoundError):
        service.get_snapshot(session.id, snapshot_version=99)
```

Adjust fixture names/`asyncio.run` usage to match how `test_rpg_service.py` already builds services and calls async methods (`create_session` is async — see its existing tests; if the file defines an event-loop fixture, use it).

- [ ] **Step 2: Run to verify they fail** — `python -m pytest tldw_Server_API/tests/RPG/test_rpg_service.py -v -k "lists or snapshot_versions"` — Expected: FAIL `AttributeError`

- [ ] **Step 3: Implement** — add the six methods per Interfaces. `list_sessions`:

```python
    def list_sessions(
        self,
        *,
        campaign_id: int | None = None,
        status: str | None = None,
        limit: int = 25,
        offset: int = 0,
    ) -> tuple[list[RPGSession], int]:
        if campaign_id is not None:
            self.repo.get_campaign(owner_user_id=self.owner_user_id, campaign_id=campaign_id)
        return self.repo.list_sessions(
            owner_user_id=self.owner_user_id,
            campaign_id=campaign_id,
            status=status,
            limit=limit,
            offset=offset,
        )
```

The others are one-line delegations to the Task-1 repo methods with `owner_user_id=self.owner_user_id`, plus the two detail tuples combining `count_campaign_sessions` / `count_pending_proposals`. Modify `get_snapshot` (line 391) as described in Interfaces.

- [ ] **Step 4: Run tests** — `python -m pytest tldw_Server_API/tests/RPG/test_rpg_service.py -v` — Expected: PASS

- [ ] **Step 5: Commit** — `git commit -m "feat(rpg): add service read methods (TASK-13561)"` (add both files)

---

### Task 7: Service mutation methods

**Files:**
- Modify: `tldw_Server_API/app/core/RPG/service.py` (after Task 6 methods)
- Test: `tldw_Server_API/tests/RPG/test_rpg_service.py` (append)

**Interfaces:**
- Consumes: Tasks 2-4 repo methods; `canonical_request_hash`, `SUPPORTED_EVENT_TYPES` (new import from `core/RPG/events.py`), `_require_idempotency_key`, `RPGValidationError`.
- Produces (exact signatures for Task 9):
  - `update_campaign(self, campaign_id: int, *, title: str | None = None, description: str | None = None, status: str | None = None, expected_version: int, idempotency_key: str) -> CampaignUpdateResult`
  - `update_session(self, session_id: int, *, title: str | None = None, status: str | None = None, expected_version: int, idempotency_key: str) -> SessionUpdateResult`
  - `replace_authority_settings(self, session_id: int, *, authority_settings: dict[str, Any], expected_version: int, idempotency_key: str) -> SessionUpdateResult`
  - `rebuild_snapshot(self, session_id: int, *, idempotency_key: str) -> SnapshotRebuildResult`
  - `_validate_authority_settings(settings: dict[str, Any]) -> None` (staticmethod): exact key set `{import_auto_commit, mcp_auto_commit, model_auto_commit, model_auto_commit_event_types}`; strict bools (`isinstance(x, bool)` — note `isinstance(True, int)` is True but `isinstance(1, bool)` is False, so check bool first); event-type list of strings ⊆ `SUPPORTED_EVENT_TYPES`; violations raise `RPGValidationError` with codes `authority_settings_fields`, `<field>_must_be_boolean`, `model_auto_commit_event_types_must_be_strings`, `unknown_event_types`.

- [ ] **Step 1: Write the failing tests**

```python
def test_service_updates_campaign_and_session():
    service = _service()
    campaign = service.create_campaign("Title", None, "fate", idempotency_key="svc-u-1")
    result = service.update_campaign(campaign.id, title="Act II", status="archived", expected_version=campaign.version, idempotency_key="svc-pu-1")
    assert result.campaign.status == "archived"  # nosec B101
    replay = service.update_campaign(campaign.id, title="Act II", status="archived", expected_version=campaign.version, idempotency_key="svc-pu-1")
    assert replay.replayed is True  # nosec B101
    with pytest.raises(RPGConflictError):
        service.update_campaign(campaign.id, title="X", expected_version=999, idempotency_key="svc-pu-2")


def test_service_validates_authority_settings():
    service = _service()
    campaign = service.create_campaign("Title", None, "fate", idempotency_key="svc-a-1")
    session = asyncio.run(service.create_session(campaign_id=campaign.id, title="S", adapter_key="fate", idempotency_key="svc-as-1"))
    good = {"import_auto_commit": False, "mcp_auto_commit": True, "model_auto_commit": False, "model_auto_commit_event_types": []}
    result = service.replace_authority_settings(session.id, authority_settings=good, expected_version=session.version, idempotency_key="svc-au-1")
    assert result.session.authority_settings["mcp_auto_commit"] is True  # nosec B101
    bad_type = {**good, "model_auto_commit_event_types": ["note.added", "not.an.event"]}
    with pytest.raises(RPGValidationError):
        service.replace_authority_settings(session.id, authority_settings=bad_type, expected_version=result.session.version, idempotency_key="svc-au-2")
    with pytest.raises(RPGValidationError):
        service.replace_authority_settings(session.id, authority_settings={"mcp_auto_commit": True}, expected_version=result.session.version, idempotency_key="svc-au-3")


def test_service_rebuild_matches_incremental_snapshot():
    service = _service()
    campaign = service.create_campaign("Title", None, "fate", idempotency_key="svc-r-1")
    session = asyncio.run(service.create_session(campaign_id=campaign.id, title="S", adapter_key="fate", idempotency_key="svc-rs-1"))
    service.record_events(session.id, [{"event_type": "note.added", "event_payload": {"note_id": "n1"}}], "user", 0, "svc-re-1")
    incremental = service.get_snapshot(session.id)
    rebuilt = service.rebuild_snapshot(session.id, idempotency_key="svc-rb-1")
    assert rebuilt.snapshot.snapshot_json == incremental.snapshot_json  # nosec B101
    assert rebuilt.snapshot.snapshot_version == incremental.snapshot_version + 1  # nosec B101
    assert rebuilt.replayed is False  # nosec B101
```

- [ ] **Step 2: Run to verify they fail** — Expected: FAIL `AttributeError`

- [ ] **Step 3: Implement**

```python
    def update_campaign(
        self,
        campaign_id: int,
        *,
        title: str | None = None,
        description: str | None = None,
        status: str | None = None,
        expected_version: int,
        idempotency_key: str,
    ) -> CampaignUpdateResult:
        self._require_idempotency_key(idempotency_key)
        request_hash = canonical_request_hash(
            {
                "target_type": "campaign",
                "campaign_id": campaign_id,
                "title": title,
                "description": description,
                "status": status,
                "expected_version": expected_version,
                "source_type": "user",
            }
        )
        return self.repo.update_campaign(
            owner_user_id=self.owner_user_id,
            campaign_id=campaign_id,
            title=title,
            description=description,
            status=status,
            expected_version=expected_version,
            idempotency_key=idempotency_key,
            request_payload_hash=request_hash,
            source_type="user",
        )
```

`update_session` is the same shape (no description, `session_id`). `replace_authority_settings`:

```python
    def replace_authority_settings(
        self,
        session_id: int,
        *,
        authority_settings: dict[str, Any],
        expected_version: int,
        idempotency_key: str,
    ) -> SessionUpdateResult:
        self._require_idempotency_key(idempotency_key)
        self._validate_authority_settings(authority_settings)
        request_hash = canonical_request_hash(
            {
                "target_type": "session",
                "session_id": session_id,
                "authority_settings": authority_settings,
                "expected_version": expected_version,
                "source_type": "user",
            }
        )
        return self.repo.update_session_authority_settings(
            owner_user_id=self.owner_user_id,
            session_id=session_id,
            authority_settings=authority_settings,
            expected_version=expected_version,
            idempotency_key=idempotency_key,
            request_payload_hash=request_hash,
            source_type="user",
        )

    @staticmethod
    def _validate_authority_settings(settings: dict[str, Any]) -> None:
        expected_keys = {
            "import_auto_commit",
            "mcp_auto_commit",
            "model_auto_commit",
            "model_auto_commit_event_types",
        }
        if set(settings) != expected_keys:
            raise RPGValidationError("authority_settings_fields")
        for field in ("import_auto_commit", "mcp_auto_commit", "model_auto_commit"):
            if not isinstance(settings[field], bool):
                raise RPGValidationError(f"{field}_must_be_boolean")
        event_types = settings["model_auto_commit_event_types"]
        if not isinstance(event_types, list) or not all(isinstance(item, str) for item in event_types):
            raise RPGValidationError("model_auto_commit_event_types_must_be_strings")
        unknown = set(event_types) - set(SUPPORTED_EVENT_TYPES)
        if unknown:
            raise RPGValidationError("unknown_event_types")
```

`rebuild_snapshot` follows the same wrapper shape with hash payload `{"target_type": "session", "session_id": session_id, "operation": "snapshot_rebuild", "source_type": "user"}`, delegating to `self.repo.rebuild_session_snapshot`.

- [ ] **Step 4: Run tests** — `python -m pytest tldw_Server_API/tests/RPG/test_rpg_service.py -v` — Expected: PASS

- [ ] **Step 5: Commit** — `git commit -m "feat(rpg): add service mutation methods with authority validation (TASK-13561)"` (add both files)

---

### Task 8: REST read routes

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/rpg.py` (append routes; extend imports with the Task-5 schemas and Task-6 service results)
- Test: `tldw_Server_API/tests/RPG/test_rpg_api.py` (append)

**Interfaces:**
- Consumes: Task 5 schemas, Task 6 service methods, existing `_read_dependencies`, `_service`, `_map_error`, `_event_payload`, `_proposal_payload`, `RPGCampaignResponse`-style construction via `model_validate(asdict(...))` (import `asdict` if missing).
- Produces: 8 GET routes (paths + permission dependencies exactly as the spec table):

| Route | Dependencies |
|---|---|
| `GET /campaigns` | `_read_dependencies(RPG_CAMPAIGNS_READ)` |
| `GET /campaigns/{campaign_id}` | `_read_dependencies(RPG_CAMPAIGNS_READ)` |
| `GET /campaigns/{campaign_id}/sessions` | `_read_dependencies(RPG_SESSIONS_READ)` |
| `GET /sessions` | `_read_dependencies(RPG_SESSIONS_READ)` |
| `GET /sessions/{session_id}` | `_read_dependencies(RPG_SESSIONS_READ)` |
| `GET /sessions/{session_id}/snapshot` | `_read_dependencies(RPG_SESSIONS_READ)` |
| `GET /sessions/{session_id}/events` | `_read_dependencies(RPG_SESSIONS_READ)` |
| `GET /sessions/{session_id}/proposals` | `_read_dependencies(RPG_PROPOSALS_REVIEW)` |

- [ ] **Step 1: Write the failing tests** (append; reuse `_headers`, `_create_campaign_and_session`, `_route_permissions`)

```python
def test_list_campaigns_route_returns_created_campaign():
    with TestClient(app) as client:
        campaign = client.post(
            "/api/v1/rpg/campaigns", headers=_headers(**{"Idempotency-Key": "route-list-c"}),
            json={"title": "Route Campaign", "default_adapter_key": "fate"},
        )
        listed = client.get("/api/v1/rpg/campaigns", headers=_headers())
        assert listed.status_code == 200  # nosec B101
        body = listed.json()
        assert body["total_count"] >= 1  # nosec B101
        assert any(item["id"] == campaign.json()["id"] for item in body["items"])  # nosec B101


def test_read_routes_scope_matrix():
    expected = {
        ("GET", "/api/v1/rpg/campaigns"): {rpg_endpoint.RPG_CAMPAIGNS_READ},
        ("GET", "/api/v1/rpg/campaigns/{campaign_id}"): {rpg_endpoint.RPG_CAMPAIGNS_READ},
        ("GET", "/api/v1/rpg/campaigns/{campaign_id}/sessions"): {rpg_endpoint.RPG_SESSIONS_READ},
        ("GET", "/api/v1/rpg/sessions"): {rpg_endpoint.RPG_SESSIONS_READ},
        ("GET", "/api/v1/rpg/sessions/{session_id}"): {rpg_endpoint.RPG_SESSIONS_READ},
        ("GET", "/api/v1/rpg/sessions/{session_id}/snapshot"): {rpg_endpoint.RPG_SESSIONS_READ},
        ("GET", "/api/v1/rpg/sessions/{session_id}/events"): {rpg_endpoint.RPG_SESSIONS_READ},
        ("GET", "/api/v1/rpg/sessions/{session_id}/proposals"): {rpg_endpoint.RPG_PROPOSALS_REVIEW},
    }
    for (method, path), scopes in expected.items():
        assert scopes <= _route_permissions(method, path), f"{method} {path} missing {scopes - _route_permissions(method, path)}"  # nosec B101


def test_session_snapshot_events_and_foreign_404():
    with TestClient(app) as client:
        campaign_id, session_id = _create_campaign_and_session(client, "route-read")
        snap = client.get(f"/api/v1/rpg/sessions/{session_id}/snapshot", headers=_headers())
        assert snap.status_code == 200 and snap.json()["snapshot_version"] == 0  # nosec B101
        events = client.get(f"/api/v1/rpg/sessions/{session_id}/events", headers=_headers())
        assert events.status_code == 200 and events.json()["items"] == []  # nosec B101
        assert client.get("/api/v1/rpg/campaigns/999999", headers=_headers()).status_code == 404  # nosec B101
```

- [ ] **Step 2: Run to verify they fail** — Expected: FAIL (404 route-not-found / missing scopes)

- [ ] **Step 3: Implement** — one route per table row. Representative implementations:

```python
@router.get(
    "/campaigns",
    response_model=RPGCampaignListResponse,
    dependencies=_read_dependencies(RPG_CAMPAIGNS_READ),
)
def list_campaigns(
    status_filter: str | None = Query(default=None, alias="status"),
    limit: int = Query(default=25, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    service: RPGService = Depends(_service),
) -> RPGCampaignListResponse:
    """List the authenticated user's RPG campaigns."""

    try:
        campaigns, total = service.list_campaigns(status=status_filter, limit=limit, offset=offset)
        return RPGCampaignListResponse(
            items=[RPGCampaignListItem.model_validate({**asdict(c)}) for c in campaigns],
            total_count=total, limit=limit, offset=offset,
        )
    except Exception as exc:
        raise _map_error(exc) from exc


@router.get(
    "/sessions/{session_id}/events",
    response_model=RPGEventsListResponse,
    dependencies=_read_dependencies(RPG_SESSIONS_READ),
)
def list_session_events(
    session_id: int,
    after_sequence: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=100),
    source_type: str | None = Query(default=None),
    event_type: str | None = Query(default=None),
    service: RPGService = Depends(_service),
) -> RPGEventsListResponse:
    """List a session's event log with cursor pagination."""

    try:
        events = service.list_events(
            session_id, after_sequence=after_sequence, limit=limit,
            source_type=source_type, event_type=event_type,
        )
        next_after = events[-1].sequence_number if len(events) == limit else (
            events[-1].sequence_number if events else after_sequence
        )
        return RPGEventsListResponse(
            items=[RPGEventResponse.model_validate(_event_payload(e)) for e in events],
            next_after_sequence=next_after, limit=limit,
        )
    except Exception as exc:
        raise _map_error(exc) from exc
```

Remaining routes follow identically: campaign detail (`session_count` from `get_campaign_detail`), campaign sessions and flat sessions (`RPGSessionListResponse`; `status`/`campaign_id` query filters; flat `/sessions` also validates foreign `campaign_id` → 404 via the service), session detail (`RPGSessionDetail` with `pending_proposal_count`), snapshot (`RPGSnapshotResponse` from `service.get_snapshot(session_id, snapshot_version)` with `version: int | None = Query(default=None, ge=0)`), proposals (`RPGProposalListResponse` with items `[_proposal_payload(p) for p in proposals]`). Add `Query` to the fastapi imports.

- [ ] **Step 4: Run tests** — `python -m pytest tldw_Server_API/tests/RPG/test_rpg_api.py -v` — Expected: PASS (all existing + new)

- [ ] **Step 5: Commit** — `git commit -m "feat(rpg): expose list/read REST routes (TASK-13561)"` (add both files)

---

### Task 9: REST mutation routes

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/rpg.py` (append)
- Test: `tldw_Server_API/tests/RPG/test_rpg_api.py` (append)

**Interfaces:**
- Consumes: Task 5 schemas, Task 7 service methods, `Header(alias="Idempotency-Key")`, `_write_dependencies`.
- Produces:

| Route | Dependencies |
|---|---|
| `PATCH /campaigns/{campaign_id}` | `_write_dependencies(RPG_CAMPAIGNS_MANAGE)` |
| `PATCH /sessions/{session_id}` | `_write_dependencies(RPG_SESSIONS_MANAGE)` |
| `PUT /sessions/{session_id}/authority-settings` | `_write_dependencies(RPG_SESSIONS_MANAGE)` |
| `POST /sessions/{session_id}/snapshot/rebuild` | `_write_dependencies(RPG_SESSIONS_MANAGE)` |

- [ ] **Step 1: Write the failing tests**

```python
def test_patch_campaign_archive_and_conflict():
    with TestClient(app) as client:
        campaign_id, _session_id = _create_campaign_and_session(client, "route-patch")
        patched = client.patch(
            f"/api/v1/rpg/campaigns/{campaign_id}", headers=_headers(**{"Idempotency-Key": "route-pc-1"}),
            json={"status": "archived", "expected_version": 1},
        )
        assert patched.status_code == 200 and patched.json()["status"] == "archived"  # nosec B101
        assert patched.json()["version"] == 2  # nosec B101
        replay = client.patch(
            f"/api/v1/rpg/campaigns/{campaign_id}", headers=_headers(**{"Idempotency-Key": "route-pc-1"}),
            json={"status": "archived", "expected_version": 1},
        )
        assert replay.status_code == 200 and replay.json()["version"] == 2  # nosec B101  (replay)
        conflict = client.patch(
            f"/api/v1/rpg/campaigns/{campaign_id}", headers=_headers(**{"Idempotency-Key": "route-pc-2"}),
            json={"title": "Stale", "expected_version": 1},
        )
        assert conflict.status_code == 409  # nosec B101
        bad_status = client.patch(
            f"/api/v1/rpg/campaigns/{campaign_id}", headers=_headers(**{"Idempotency-Key": "route-pc-3"}),
            json={"status": "deleted", "expected_version": 2},
        )
        assert bad_status.status_code == 400  # nosec B101


def test_authority_settings_put_roundtrip():
    with TestClient(app) as client:
        _campaign_id, session_id = _create_campaign_and_session(client, "route-auth")
        payload = {
            "import_auto_commit": False, "mcp_auto_commit": True,
            "model_auto_commit": False, "model_auto_commit_event_types": [],
            "expected_version": 1,
        }
        updated = client.put(
            f"/api/v1/rpg/sessions/{session_id}/authority-settings",
            headers=_headers(**{"Idempotency-Key": "route-au-1"}), json=payload,
        )
        assert updated.status_code == 200  # nosec B101
        assert updated.json()["authority_settings"]["mcp_auto_commit"] is True  # nosec B101
        detail = client.get(f"/api/v1/rpg/sessions/{session_id}", headers=_headers())
        assert detail.json()["authority_settings"]["mcp_auto_commit"] is True  # nosec B101
        bad = dict(payload, model_auto_commit_event_types=["nope.nope"], expected_version=2)
        assert client.put(
            f"/api/v1/rpg/sessions/{session_id}/authority-settings",
            headers=_headers(**{"Idempotency-Key": "route-au-2"}), json=bad,
        ).status_code == 400  # nosec B101


def test_snapshot_rebuild_route():
    with TestClient(app) as client:
        _campaign_id, session_id = _create_campaign_and_session(client, "route-rb")
        rebuilt = client.post(
            f"/api/v1/rpg/sessions/{session_id}/snapshot/rebuild", headers=_headers(**{"Idempotency-Key": "route-rb-1"}),
        )
        assert rebuilt.status_code == 200  # nosec B101
        assert rebuilt.json()["event_count"] == 0 and rebuilt.json()["replayed"] is False  # nosec B101
        replay = client.post(
            f"/api/v1/rpg/sessions/{session_id}/snapshot/rebuild", headers=_headers(**{"Idempotency-Key": "route-rb-1"}),
        )
        assert replay.status_code == 200 and replay.json()["replayed"] is True  # nosec B101
```

Plus one scope-matrix test for the four mutation routes mirroring Task 8's `test_read_routes_scope_matrix` (expected scopes: `RPG_CAMPAIGNS_MANAGE` / `RPG_SESSIONS_MANAGE` respectively).

- [ ] **Step 2: Run to verify they fail** — Expected: FAIL

- [ ] **Step 3: Implement**

```python
@router.patch(
    "/campaigns/{campaign_id}",
    response_model=RPGCampaignDetail,
    dependencies=_write_dependencies(RPG_CAMPAIGNS_MANAGE),
)
def update_campaign(
    campaign_id: int,
    request: RPGCampaignUpdateRequest,
    idempotency_key: str = Header(alias="Idempotency-Key"),
    service: RPGService = Depends(_service),
) -> RPGCampaignDetail:
    """Update campaign metadata or archive status."""

    try:
        result = service.update_campaign(
            campaign_id,
            title=request.title,
            description=request.description,
            status=request.status,
            expected_version=request.expected_version,
            idempotency_key=idempotency_key,
        )
        session_count = 0  # filled from service.get_campaign_detail on replay-safe reads
        detail, session_count = service.get_campaign_detail(campaign_id)
        return RPGCampaignDetail.model_validate({**asdict(detail), "session_count": session_count})
    except Exception as exc:
        raise _map_error(exc) from exc
```

(The redundant `session_count = 0` line must be deleted — the tuple assignment overwrites it; the response after update always re-reads the row, which is both replay-correct and current.) The other three routes follow the same pattern: `PATCH /sessions/{session_id}` returns `RPGSessionDetail` via `service.get_session_detail`; `PUT .../authority-settings` returns `RPGSessionDetail`; `POST .../snapshot/rebuild` returns `RPGSnapshotRebuildResponse(snapshot_version=..., last_event_sequence=result.snapshot.last_event_sequence, event_count=result.event_count, replayed=result.replayed, diagnostics=result.snapshot.diagnostics_json)` and takes no body.

- [ ] **Step 4: Run tests** — `python -m pytest tldw_Server_API/tests/RPG/test_rpg_api.py -v` — Expected: PASS

- [ ] **Step 5: Commit** — `git commit -m "feat(rpg): expose PATCH/PUT/rebuild mutation routes (TASK-13561)"` (add both files)

---

### Task 10: Documentation, CI gates, final verification

**Files:**
- Modify: `tldw_Server_API/app/core/RPG/README.md` (route tables + non-goals)
- Modify: `backlog/tasks/task-13561 - *.md` (via backlog-py; status/notes/final summary)

**Interfaces:**
- Consumes: all prior tasks.
- Produces: documented, CI-clean feature.

- [ ] **Step 1: Update README** — add all 12 routes to the REST table (method, path, permission, one-line purpose) and rewrite "Current Non-Goals": remove list/read endpoints, authority-settings mutation, and snapshot rebuild (now implemented); keep no-deletion, no dice endpoint (Phase 2), no MCP parity (Phase 3), no compaction.

- [ ] **Step 2: Run CI lints**

```bash
python Helper_Scripts/ci/rg_route_map_lint.py
python Helper_Scripts/ci/route_auth_ratchet.py
```
Expected: both exit 0. (New routes carry `RequirePermission`/`TokenScopeGuard` and resolve `get_request_user` via `_service`, so the auth ratchet must see them protected; the RG lint only flags dead/shadowed policy entries, which we did not touch. If either fails, fix the finding — never edit baselines to suppress.)

- [ ] **Step 3: Full RPG suite + Bandit**

```bash
python -m pytest tldw_Server_API/tests/RPG/ -v
source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/RPG tldw_Server_API/app/core/DB_Management/RPG_DB.py tldw_Server_API/app/api/v1/endpoints/rpg.py -f json -o /tmp/bandit_task_13561.json
```
Expected: all tests PASS; no new Bandit findings in touched files (f-strings in SQL are parameterized values only — the interpolated fragments are static `WHERE` column lists built from fixed clauses, not user input; if Bandit flags B608, add `# nosec B608` with a comment explaining the interpolation is constant).

- [ ] **Step 4: Update backlog task and commit**

```bash
PYTHONPATH=tools/backlog-py/src python -m backlog_py --cwd <repo> task edit TASK-13561 -s "Done" --final-summary "..."  # only after verification passes
git add -A && git commit -m "docs(rpg): document Phase-1 API surface; verify CI gates (TASK-13561)"
```

---

## Plan self-review record

- **Spec coverage:** 8 read routes → Tasks 1/6/8; PATCH ×2 → 2/7/9; authority PUT → 3/7/9; rebuild → 4/7/9; schemas/bounds → 5; ordering/pagination/termination → 1/5/8; RG + ratchet + README + catalog tests → 10; ADR recorded in task notes (none required). No spec section is unimplemented.
- **Placeholders:** none; every code step carries verbatim code. Two flagged judgment points exist by design (Task 4 nested-transaction note; Task 6 fixture adaptation) with concrete fallback instructions.
- **Type consistency:** result dataclass names (`CampaignUpdateResult`, `SessionUpdateResult`, `SnapshotRebuildResult`) and method signatures are identical across Tasks 2-4 (producers) and 6-9 (consumers); schema names in Task 5 match Task 8/9 usage.
