# Workspace Core Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the first backend Workspace Core contract slice so existing Research Workspaces can expose a persisted profile, project-root capability state, fail-closed runtime context, and additive API fields without building the full Project Workspace UI or Sandbox runtime yet.

**Architecture:** Keep `/api/v1/workspaces` as the canonical API family and add a typed `core/Workspaces` contract layer over existing Workspace storage, source status projection, service capability probes, and future root bindings. Persist `workspace_profile` on the existing Workspace row, persist one primary root in a new Workspace-owned table set, and compute capability/runtime state on read. Preserve existing response fields for compatibility while adding `schema_version: 2` fields that power future MCP, ACP, Sandbox, and harness work.

**Tech Stack:** FastAPI, Pydantic v2, `CharactersRAGDB`, SQLite/PostgreSQL-compatible SQL through existing DB abstractions, pytest, Bandit.

---

## Scope Boundaries

This plan implements the first contract slice only.

In scope:

- Persist `workspace_profile` as `research` or `project`.
- Add Workspace-owned primary root records for `host_local` and `sandbox_volume` backend types.
- Compute `project_root_state`, file inventory defaults, runtime capability states, allowed actions, and resolver status on read.
- Extend existing `/api/v1/workspaces/{workspace_id}`, `/capabilities`, `/context`, and a read-only `/roots` endpoint additively.
- Keep `workspace_kind` as a compatibility/display alias, not a new source of truth.
- Add tests for SQLite behavior and schema-level API contracts.
- Handle SQLite and PostgreSQL uniqueness/backend errors through existing `ConflictError`, `InputError`, and `CharactersRAGDBError` patterns.

Out of scope:

- No UI redesign.
- No root attach wizard.
- No actual Sandbox volume creation.
- No host-local allowlist enforcement beyond persisted contract validation.
- No file tree scanner Jobs worker.
- No file-content indexing worker.
- No public preview/deploy feature.
- No route aliases or redirects.

Compatibility rules:

- Existing fields must keep their current names and meanings.
- New response fields must be optional or defaulted in Pydantic schemas where old rows may lack data.
- `workspace_kind` remains present for clients that already consume it.
- `workspace_profile` is the new source of truth for profile intent.
- Context `schema_version` may advance to `2`, but the old top-level keys remain available.

## File Structure

- Create `tldw_Server_API/app/core/Workspaces/models.py`
  - Owns literals, dataclasses, and small normalization helpers for Workspace Core contracts.
- Create `tldw_Server_API/app/core/Workspaces/context.py`
  - Builds read-only Workspace context envelopes from a workspace row, optional primary root row, source summary, service capabilities, and partial errors.
- Modify `tldw_Server_API/app/core/Workspaces/status_projection.py`
  - Preserve source status behavior and route capability projection through Workspace Core helpers where appropriate.
- Modify `tldw_Server_API/app/core/Workspaces/README.md`
  - Document the canonical Workspace Core module boundaries and first-slice limitations.
- Modify `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
  - Add `workspace_profile` to `workspaces`.
  - Add new Workspace-owned primary-root table and DB methods.
  - Keep all direct SQL inside the existing DB abstraction.
  - Cascade root cleanup on workspace hard delete and hide roots for soft-deleted workspaces.
- Modify `tldw_Server_API/app/api/v1/schemas/workspace_schemas.py`
  - Add additive profile/root/runtime schemas.
  - Extend existing response schemas without removing current fields.
- Modify `tldw_Server_API/app/api/v1/endpoints/workspaces.py`
  - Return new fields from existing workspace, capabilities, and context endpoints.
  - Add `GET /api/v1/workspaces/{workspace_id}/roots` as read-only contract surface.
- Test `tldw_Server_API/tests/Workspaces/test_workspace_core_models.py`
  - Unit tests for normalization, fail-closed defaults, and runtime envelope shape.
- Test `tldw_Server_API/tests/Workspaces/test_workspace_project_roots_db.py`
  - DB tests for profile persistence and one-primary-root behavior.
- Test `tldw_Server_API/tests/Workspaces/test_workspace_core_context.py`
  - Context resolver unit tests.
- Modify `tldw_Server_API/tests/Workspaces/test_workspace_service_capabilities.py`
  - Regression tests for capability projection compatibility and fail-closed actions.
- Modify `tldw_Server_API/tests/Workspaces/test_workspaces_api.py`
  - API tests for additive response fields and read-only root listing.

## Parallelization Map

Sequential blocker:

- Task 1 must land first because later code imports these contracts.

Can run in parallel after Task 1:

- Task 2 DB persistence.
- Task 3 context resolver.
- Task 4 API schema drafting.

Must integrate after Tasks 2-4:

- Task 5 endpoint wiring.
- Task 6 compatibility and verification.

Avoid parallel edits to `workspace_schemas.py`, `workspaces.py`, and `ChaChaNotes_DB.py` unless ownership is explicitly split by file.

---

### Task 1: Workspace Core Model Contracts

**Files:**
- Create: `tldw_Server_API/app/core/Workspaces/models.py`
- Test: `tldw_Server_API/tests/Workspaces/test_workspace_core_models.py`

- [ ] **Step 1: Write failing model contract tests**

Create `test_workspace_core_models.py` with coverage for profile normalization, project-root states, root backend validation, and fail-closed allowed actions.

```python
from tldw_Server_API.app.core.Workspaces.models import (
    WorkspaceProfile,
    normalize_workspace_profile,
    normalize_project_root_state,
    workspace_kind_for_profile,
    fail_closed_action,
)


def test_workspace_profile_defaults_to_research() -> None:
    assert normalize_workspace_profile(None) == "research"
    assert normalize_workspace_profile("") == "research"
    assert normalize_workspace_profile("project") == "project"


def test_workspace_kind_is_compatibility_alias() -> None:
    assert workspace_kind_for_profile("research") == "research_workspace"
    assert workspace_kind_for_profile("project") == "project_workspace"


def test_project_root_state_fails_closed_for_unknown_values() -> None:
    assert normalize_project_root_state("attached") == "attached"
    assert normalize_project_root_state("unexpected") == "failed"


def test_fail_closed_action_uses_reason_code() -> None:
    assert fail_closed_action("root_unresolved") == {
        "allowed": False,
        "reason_code": "root_unresolved",
    }
```

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_core_models.py -q
```

Expected: FAIL because `models.py` does not exist.

- [ ] **Step 2: Implement the minimal model contracts**

Create `models.py` with:

```python
from __future__ import annotations

from typing import Any, Literal, TypedDict

WorkspaceProfile = Literal["research", "project"]
WorkspaceKind = Literal["research_workspace", "project_workspace"]
ProjectRootBackend = Literal["host_local", "sandbox_volume"]
ProjectRootState = Literal[
    "not_configured",
    "attached",
    "missing",
    "detached",
    "failed",
    "archived",
]
ResolutionStatus = Literal["complete", "partial", "failed"]


class AllowedAction(TypedDict):
    allowed: bool
    reason_code: str | None


def normalize_workspace_profile(value: Any) -> WorkspaceProfile:
    return "project" if str(value or "").strip().lower() == "project" else "research"


def workspace_kind_for_profile(profile: WorkspaceProfile) -> WorkspaceKind:
    return "project_workspace" if profile == "project" else "research_workspace"


def normalize_project_root_state(value: Any) -> ProjectRootState:
    normalized = str(value or "").strip().lower()
    if normalized in {"not_configured", "attached", "missing", "detached", "failed", "archived"}:
        return normalized  # type: ignore[return-value]
    return "failed"


def allowed_action() -> AllowedAction:
    return {"allowed": True, "reason_code": None}


def fail_closed_action(reason_code: str) -> AllowedAction:
    return {"allowed": False, "reason_code": reason_code}
```

- [ ] **Step 3: Run model tests**

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_core_models.py -q
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tldw_Server_API/app/core/Workspaces/models.py tldw_Server_API/tests/Workspaces/test_workspace_core_models.py
git commit -m "feat: add workspace core model contracts"
```

---

### Task 2: Workspace Profile And Primary Root Persistence

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Test: `tldw_Server_API/tests/Workspaces/test_workspace_project_roots_db.py`
- Modify: `tldw_Server_API/tests/Workspaces/test_workspaces_api.py`

- [ ] **Step 1: Write failing DB tests for profile persistence**

Create `test_workspace_project_roots_db.py`.

```python
import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    ConflictError,
    InputError,
)


@pytest.fixture
def db(tmp_path):
    return CharactersRAGDB(db_path=str(tmp_path / "chacha.db"), client_id="user-1")


def test_workspace_defaults_to_research_profile(db):
    ws = db.upsert_workspace("ws-1", "Workspace")
    assert ws["workspace_profile"] == "research"


def test_workspace_can_be_created_as_project_profile(db):
    ws = db.upsert_workspace("ws-1", "Workspace", workspace_profile="project")
    assert ws["workspace_profile"] == "project"
```

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_project_roots_db.py -q
```

Expected: FAIL because `workspace_profile` is not accepted or returned.

- [ ] **Step 2: Add `workspace_profile` schema migration**

In `ChaChaNotes_DB.py`:

- Add `workspace_profile TEXT NOT NULL DEFAULT 'research'` to SQLite and PostgreSQL workspace table creation paths.
- Add migration logic that adds the column when missing.
- Update `upsert_workspace(..., workspace_profile: str = "research")`.
- Update `update_workspace` allowlist to include `workspace_profile`.
- Normalize invalid values to `research` or raise `InputError`. Use the stricter option for writes:

```python
if workspace_profile not in {"research", "project"}:
    raise InputError("workspace_profile must be 'research' or 'project'")
```

- [ ] **Step 3: Run profile DB tests**

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_project_roots_db.py -q
```

Expected: PASS for profile tests; root tests not added yet.

- [ ] **Step 4: Write failing primary-root DB tests**

Add:

```python
def test_upsert_primary_host_local_root_upgrades_workspace_profile(db):
    db.upsert_workspace("ws-1", "Workspace")

    root = db.upsert_workspace_primary_root(
        "ws-1",
        {
            "root_id": "root-host",
            "backend": "host_local",
            "display_name": "Local project",
            "absolute_root": "/Users/example/project",
            "root_state": "attached",
        },
    )

    assert root["workspace_id"] == "ws-1"
    assert root["backend"] == "host_local"
    assert root["root_state"] == "attached"
    assert root["is_primary"] in (True, 1)
    assert db.get_workspace("ws-1")["workspace_profile"] == "project"


def test_upsert_primary_sandbox_root_is_first_class(db):
    db.upsert_workspace("ws-1", "Workspace")

    root = db.upsert_workspace_primary_root(
        "ws-1",
        {
            "root_id": "root-sandbox",
            "backend": "sandbox_volume",
            "display_name": "Sandbox project",
            "sandbox_volume_id": "volume-123",
            "root_state": "not_configured",
        },
    )

    assert root["backend"] == "sandbox_volume"
    assert root["sandbox_volume_id"] == "volume-123"


def test_primary_root_upsert_replaces_existing_primary_root(db):
    db.upsert_workspace("ws-1", "Workspace")
    db.upsert_workspace_primary_root("ws-1", {"root_id": "root-1", "backend": "host_local"})
    root = db.upsert_workspace_primary_root("ws-1", {"root_id": "root-2", "backend": "sandbox_volume"})

    roots = db.list_workspace_project_roots("ws-1")
    assert [item["root_id"] for item in roots] == ["root-2"]
    assert root["root_id"] == "root-2"


def test_invalid_root_backend_raises_input_error(db):
    db.upsert_workspace("ws-1", "Workspace")
    with pytest.raises(InputError):
        db.upsert_workspace_primary_root("ws-1", {"root_id": "root-1", "backend": "git_clone"})


def test_soft_deleted_workspace_roots_are_not_listed(db):
    ws = db.upsert_workspace("ws-1", "Workspace")
    db.upsert_workspace_primary_root("ws-1", {"root_id": "root-1", "backend": "host_local"})
    db.delete_workspace("ws-1", expected_version=ws["version"])

    assert db.list_workspace_project_roots("ws-1") == []
```

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_project_roots_db.py -q
```

Expected: FAIL because root table and methods do not exist.

- [ ] **Step 5: Implement Workspace-owned root table and methods**

In `ChaChaNotes_DB.py`:

- Add `workspace_project_roots` table for SQLite and PostgreSQL.
- Columns:
  - `root_id TEXT PRIMARY KEY`
  - `workspace_id TEXT NOT NULL`
  - `is_primary BOOLEAN NOT NULL DEFAULT true`
  - `backend TEXT NOT NULL`
  - `absolute_root TEXT`
  - `sandbox_volume_id TEXT`
  - `display_name TEXT`
  - `root_state TEXT NOT NULL DEFAULT 'not_configured'`
  - `git_state TEXT NOT NULL DEFAULT 'absent'`
  - `file_inventory_state TEXT NOT NULL DEFAULT 'not_started'`
  - `indexing_state TEXT NOT NULL DEFAULT 'disabled'`
  - `sandbox_mount_state TEXT NOT NULL DEFAULT 'not_configured'`
  - `mcp_trust_state TEXT NOT NULL DEFAULT 'not_configured'`
  - `metadata_json TEXT NOT NULL DEFAULT '{}'`
  - `created_at TEXT/TIMESTAMP`
  - `updated_at TEXT/TIMESTAMP`
  - `version INTEGER NOT NULL DEFAULT 1`
- Add a partial unique index where supported, or enforce in transaction:
  - one primary root per `workspace_id`.
- Add a normal index on `workspace_id` for lookup.
- Add schema helpers close to the existing workspace schema helpers rather than scattering root DDL in unrelated migration sections.
- Add methods:
  - `upsert_workspace_primary_root(workspace_id: str, data: dict[str, Any]) -> dict[str, Any]`
  - `get_workspace_primary_root(workspace_id: str) -> dict[str, Any] | None`
  - `list_workspace_project_roots(workspace_id: str) -> list[dict[str, Any]]`
  - `update_workspace_project_root_state(workspace_id: str, root_id: str, updates: dict[str, Any], expected_version: int) -> dict[str, Any]`
- When a primary root is created, update the parent workspace profile to `project`.
- Do not silently downgrade profile when root is removed or marked detached.
- Catch `sqlite3.IntegrityError` and `BackendDatabaseError`; map uniqueness/race conditions to `ConflictError` or idempotent return where content matches, and wrap other backend failures in `CharactersRAGDBError`.
- Ensure workspace soft delete hides roots from list/get methods.
- Ensure workspace hard delete removes root records.

- [ ] **Step 6: Run DB tests**

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_project_roots_db.py -q
```

Expected: PASS.

- [ ] **Step 7: Run existing workspace DB/API regression tests**

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Workspaces/test_workspaces_api.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_workspace_sub_resources_db.py \
  -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/tests/Workspaces/test_workspace_project_roots_db.py tldw_Server_API/tests/Workspaces/test_workspaces_api.py
git commit -m "feat: persist workspace profiles and primary roots"
```

---

### Task 3: Read-Only Workspace Context Resolver

**Files:**
- Create: `tldw_Server_API/app/core/Workspaces/context.py`
- Modify: `tldw_Server_API/app/core/Workspaces/status_projection.py`
- Test: `tldw_Server_API/tests/Workspaces/test_workspace_core_context.py`
- Modify: `tldw_Server_API/tests/Workspaces/test_workspace_service_capabilities.py`

- [x] **Step 1: Write failing context resolver tests**

Create `test_workspace_core_context.py`.

```python
from tldw_Server_API.app.core.Workspaces.context import build_workspace_core_context


def test_research_workspace_context_has_fail_closed_project_capabilities() -> None:
    context = build_workspace_core_context(
        workspace={"id": "ws-1", "workspace_profile": "research"},
        primary_root=None,
        source_summary={"total": 0, "queryable": 0},
        service_capabilities={},
        partial_errors=[],
    )

    assert context["workspace_profile"] == "research"
    assert context["workspace_kind"] == "research_workspace"
    assert context["project_root"]["state"] == "not_configured"
    assert context["resolution"]["status"] == "complete"
    assert context["allowed_actions"]["write_files"] == {
        "allowed": False,
        "reason_code": "project_root_not_configured",
    }


def test_project_workspace_context_represents_sandbox_root() -> None:
    context = build_workspace_core_context(
        workspace={"id": "ws-1", "workspace_profile": "project"},
        primary_root={
            "root_id": "root-1",
            "backend": "sandbox_volume",
            "display_name": "Sandbox project",
            "root_state": "attached",
            "sandbox_volume_id": "volume-1",
            "git_state": "absent",
            "file_inventory_state": "not_started",
            "indexing_state": "disabled",
        },
        source_summary={"total": 3, "queryable": 2},
        service_capabilities={
            "workspace_services": {
                "sandbox": {"state": "available", "reason_code": None},
                "mcp": {"state": "available", "reason_code": None},
                "acp": {"state": "available", "reason_code": None},
            },
            "allowed_actions": {
                "use_sandbox": {"allowed": True, "reason_code": None},
                "run_mcp_tools": {"allowed": True, "reason_code": None},
                "use_acp_agents": {"allowed": True, "reason_code": None},
            },
        },
        partial_errors=[],
    )

    assert context["workspace_profile"] == "project"
    assert context["workspace_kind"] == "project_workspace"
    assert context["project_root"]["backend"] == "sandbox_volume"
    assert context["allowed_actions"]["write_files"]["allowed"] is True
    assert context["allowed_actions"]["run_sandbox"]["allowed"] is True


def test_context_resolution_becomes_partial_for_dependency_failures() -> None:
    context = build_workspace_core_context(
        workspace={"id": "ws-1", "workspace_profile": "project"},
        primary_root={"root_id": "root-1", "backend": "host_local", "root_state": "attached"},
        source_summary={},
        service_capabilities={},
        partial_errors=[
            {"scope": "mcp", "code": "mcp_policy_resolution_failed", "message": "MCP unavailable"}
        ],
    )

    assert context["resolution"]["status"] == "partial"
    assert context["allowed_actions"]["run_mcp_tools"]["allowed"] is False
```

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_core_context.py -q
```

Expected: FAIL because `context.py` does not exist.

- [x] **Step 2: Implement `build_workspace_core_context`**

In `context.py`:

- Normalize profile with `normalize_workspace_profile`.
- Use `workspace_kind_for_profile`.
- Build `resolution`:
  - `complete` when no partial errors.
  - `partial` when any partial error exists and core workspace row/root still resolved.
  - `failed` only when workspace identity cannot resolve; endpoint should normally 404 before this.
- Build `project_root`:
  - state `not_configured` for research/no-root.
  - include `root_id`, `backend`, `display_name`, `git_state`, `file_inventory_state`, `indexing_state`, `sandbox_mount_state`, and `mcp_trust_state`.
  - redact `absolute_root` to `path_hint` unless an explicit privileged admin endpoint is added later.
- Build fail-closed actions:
  - `write_files`
  - `run_sandbox`
  - `use_mcp_tools`
  - `use_acp_agents`
  - `create_preview`
  - `index_file_content`
- Preserve existing research actions from `status_projection` where available.

- [x] **Step 3: Update capability projection to consume Workspace Core context**

In `status_projection.py`, keep `build_source_status_projection` unchanged. Update `build_workspace_capability_projection` to:

- return `workspace_profile`
- return compatibility `workspace_kind`
- include `project_root`
- include `resolution`
- preserve existing `source_summary`, `workspace_services`, and `allowed_actions` keys
- keep `ask_grounded_questions` behavior unchanged
- never allow `write_files`, `run_sandbox`, `use_mcp_tools`, `use_acp_agents`, `create_preview`, or `index_file_content` when resolution is `failed` or the required subsystem state is unknown

- [x] **Step 4: Run context and capability tests**

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Workspaces/test_workspace_core_context.py \
  tldw_Server_API/tests/Workspaces/test_workspace_service_capabilities.py \
  -q
```

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Workspaces/context.py tldw_Server_API/app/core/Workspaces/status_projection.py tldw_Server_API/tests/Workspaces/test_workspace_core_context.py tldw_Server_API/tests/Workspaces/test_workspace_service_capabilities.py
git commit -m "feat: add workspace core context resolver"
```

---

### Task 4: Additive API Schemas

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/workspace_schemas.py`
- Test: `tldw_Server_API/tests/Workspaces/test_workspaces_api.py`

- [x] **Step 1: Write failing API schema assertions**

Extend existing API tests to verify:

```python
def test_workspace_response_includes_profile_contract(workspace_fastapi_app, db):
    # create workspace through API
    # assert response has workspace_profile == "research"
    # assert response keeps existing workspace_kind compatibility only where currently exposed
```

Add context/capability assertions:

```python
assert payload["schema_version"] == 2
assert payload["workspace_profile"] == "research"
assert payload["workspace_kind"] == "research_workspace"
assert payload["project_root"]["state"] == "not_configured"
assert payload["resolution"]["status"] in {"complete", "partial"}
assert payload["capabilities"]["workspace_profile"] == "research"
```

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspaces_api.py -k "profile_contract or workspace_context" -q
```

Expected: FAIL because schemas do not expose the new fields.

- [x] **Step 2: Add schema models**

In `workspace_schemas.py`, add:

- `WorkspaceProfile = Literal["research", "project"]`
- `WorkspaceKind = Literal["research_workspace", "project_workspace"]`
- `WorkspaceResolution`
- `WorkspaceProjectRoot`
- `WorkspaceFileInventory`
- `WorkspaceRuntimeBinding`
- `WorkspaceRootResponse`
- `WorkspaceRootsResponse`

Extend:

- `WorkspaceUpsertRequest.workspace_profile: WorkspaceProfile = "research"`
- `WorkspacePatchRequest.workspace_profile: WorkspaceProfile | None = None`
- `WorkspaceResponse.workspace_profile: WorkspaceProfile = "research"`
- `WorkspaceCapabilitiesResponse.workspace_profile`
- `WorkspaceCapabilitiesResponse.project_root`
- `WorkspaceCapabilitiesResponse.resolution`
- `WorkspaceContextResponse.schema_version = 2`
- `WorkspaceContextResponse.workspace_profile`
- `WorkspaceContextResponse.project_root`
- `WorkspaceContextResponse.resolution`

Keep old fields unless a test proves they are unused and safe to remove. Do not remove `workspace_kind`.

Use defaults for additive fields so old rows remain readable:

```python
workspace_profile: WorkspaceProfile = "research"
workspace_kind: WorkspaceKind = "research_workspace"
project_root: WorkspaceProjectRoot = Field(default_factory=WorkspaceProjectRoot)
resolution: WorkspaceResolution = Field(default_factory=WorkspaceResolution)
```

- [x] **Step 3: Run schema/API tests**

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspaces_api.py -k "profile_contract or workspace_context" -q
```

Expected: PASS after endpoint wiring in Task 5; it may still fail at this step if endpoint mappings are not updated yet.

- [x] **Step 4: Commit only if tests pass independently**

Status note: the additive request/response schema bridge was pulled forward into
the Task 3 review-fix commit because the capability/context endpoints already
validated project workspace payloads through `WorkspaceCapabilitiesResponse`.
The read-only roots endpoint and remaining endpoint wiring stay in Task 5.

If endpoint wiring is needed, do not commit half-wired schemas. Continue to Task 5 and commit both tasks together.

---

### Task 5: Endpoint Wiring And Read-Only Roots API

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/workspaces.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/workspace_schemas.py`
- Test: `tldw_Server_API/tests/Workspaces/test_workspaces_api.py`

- [x] **Step 1: Write failing read-only roots endpoint test**

Add to `test_workspaces_api.py`:

```python
@pytest.mark.integration
def test_workspace_roots_endpoint_returns_primary_root_contract(workspace_fastapi_app, db):
    # set auth/rate-limit overrides using existing helpers/patterns
    db.upsert_workspace("ws-root", "Rooted")
    db.upsert_workspace_primary_root(
        "ws-root",
        {
            "root_id": "root-1",
            "backend": "host_local",
            "display_name": "Local root",
            "absolute_root": "/Users/example/project",
            "root_state": "attached",
        },
    )

    with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
        response = client.get("/api/v1/workspaces/ws-root/roots")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["workspace_id"] == "ws-root"
    assert payload["primary_root"]["root_id"] == "root-1"
    assert payload["primary_root"]["backend"] == "host_local"
    assert "absolute_root" not in payload["primary_root"]
```

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspaces_api.py -k "roots_endpoint or profile_contract or workspace_context" -q
```

Expected: FAIL because endpoint mapping is not implemented.

- [x] **Step 2: Update response mapping helpers**

In `workspaces.py`:

- Update `_ws_to_response` to include `workspace_profile`.
- Add `_root_to_response` that redacts `absolute_root` and emits `path_hint`.
- Add `_workspace_core_context_payload` helper that calls `build_workspace_core_context`.
- In `upsert_workspace`, pass `body.workspace_profile`.
- In `patch_workspace`, allow `workspace_profile`.

- [x] **Step 3: Extend `/capabilities` and `/context`**

- Fetch `primary_root = db.get_workspace_primary_root(workspace_id)`.
- Pass `primary_root` into capability/context projection.
- Return `schema_version=2` for context.
- Keep existing `sources`, `services`, `allowed_actions`, and `active_jobs` keys unchanged.
- Partial failures from Media DB and Jobs should feed `resolution.status: partial`.

- [x] **Step 4: Add read-only roots route**

Add:

```python
@router.get(
    "/{workspace_id}/roots",
    response_model=WorkspaceRootsResponse,
    dependencies=[Depends(WORKSPACES_READ_RATE_LIMIT)],
    summary="List workspace project roots",
)
async def list_workspace_roots(...):
    workspace = _require_workspace(db, workspace_id)
    roots = db.list_workspace_project_roots(workspace_id)
    primary = next((root for root in roots if bool(root.get("is_primary"))), None)
    return WorkspaceRootsResponse(
        workspace_id=workspace_id,
        workspace_profile=workspace.get("workspace_profile") or "research",
        primary_root=_root_to_response(primary) if primary else None,
        roots=[_root_to_response(root) for root in roots],
    )
```

Do not add `POST /roots/primary` in this task. Root mutation requires the next root-attach/Sandbox-wrapper implementation plan.

- [x] **Step 5: Run focused API tests**

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspaces_api.py -k "roots_endpoint or profile_contract or workspace_context" -q
```

Expected: PASS.

- [x] **Step 6: Run workspace API regression tests**

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspaces_api.py -q
```

Expected: PASS.

- [x] **Step 7: Commit**

```bash
git add tldw_Server_API/app/api/v1/endpoints/workspaces.py tldw_Server_API/app/api/v1/schemas/workspace_schemas.py tldw_Server_API/tests/Workspaces/test_workspaces_api.py
git commit -m "feat: expose workspace project roots contract"
```

Status note: Task 5 added the read-only `/api/v1/workspaces/{workspace_id}/roots`
contract, `WorkspaceRootsResponse.workspace_profile`, and `primary_root`.
Code review found mapped-error, response-validation, and path-redaction edge
cases; all were fixed with regressions before commit. Local verification: focused
Workspace subset -> 97 passed, 6 warnings; compile smoke passed; `git diff
--check` passed; Bandit on touched API/schema files -> 0 findings.

---

### Task 6: Documentation, Compatibility, And Verification

**Files:**
- Modify: `tldw_Server_API/app/core/Workspaces/README.md`
- Modify only if needed: `Docs/superpowers/specs/2026-06-03-canonical-workspace-core-project-model-design.md`
- Modify Backlog task for implementation, not this plan task.

- [x] **Step 1: Update Workspace Core README**

Document:

- `workspace_id` is canonical.
- `workspace_profile` is persisted.
- `workspace_kind` is compatibility/display output.
- Root creation is not public API in this slice.
- `GET /roots`, `/capabilities`, and `/context` are read contract surfaces.
- Sandbox volume creation, root attach UX, file inventory Jobs, and indexing policy are follow-up slices.

- [x] **Step 2: Run focused backend tests**

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Workspaces/test_workspace_core_models.py \
  tldw_Server_API/tests/Workspaces/test_workspace_project_roots_db.py \
  tldw_Server_API/tests/Workspaces/test_workspace_core_context.py \
  tldw_Server_API/tests/Workspaces/test_workspace_service_capabilities.py \
  tldw_Server_API/tests/Workspaces/test_workspaces_api.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_workspace_sub_resources_db.py \
  -q
```

Expected: PASS.

- [x] **Step 3: Run import/compile smoke**

```bash
source .venv/bin/activate && python -m compileall \
  tldw_Server_API/app/core/Workspaces \
  tldw_Server_API/app/api/v1/endpoints/workspaces.py \
  tldw_Server_API/app/api/v1/schemas/workspace_schemas.py
```

Expected: exit code 0.

- [x] **Step 4: Run Bandit on touched backend scope**

```bash
source .venv/bin/activate && python -m bandit \
  -r tldw_Server_API/app/core/Workspaces \
     tldw_Server_API/app/api/v1/endpoints/workspaces.py \
     tldw_Server_API/app/api/v1/schemas/workspace_schemas.py \
     tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  -f json -o /tmp/bandit_workspace_core_contract.json
```

Expected: no new findings in touched code. Existing baseline findings in large legacy files must be documented with exact issue IDs and why they are unrelated.

- [x] **Step 5: Run diff hygiene**

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; only intended files changed.

- [x] **Step 6: Commit closeout docs**

```bash
git add tldw_Server_API/app/core/Workspaces/README.md
git commit -m "docs: document workspace core contract slice"
```

Skip this commit if README changes were included in an earlier task commit.

Status note: Task 6 documented the canonical Workspace Core contract, read-only
API surfaces, privacy boundaries for project roots, and follow-up slices. Local
verification: focused backend suite -> 119 passed, 6 warnings; compile smoke
passed; Bandit full touched backend scope -> 0 findings with legacy `nosec`
skips in `ChaChaNotes_DB.py`; `git diff --check` passed.

---

## Follow-Up Work Items After This Plan

These are deliberately separate tasks:

1. Root attach/write API with host-local validation and Sandbox-bound volume wrapper.
2. File inventory Jobs worker with ignore policy, scan state, partial success, and bounded diagnostics.
3. Explicit file-content indexing policy API and indexing Jobs.
4. MCP trusted-root binding migration and terminology cleanup in MCP Hub surfaces.
5. ACP/harness runtime envelope consumption for Codex and future adapters.
6. Project Workspace UI mode with root health, file tree metadata, Git state, and remediation actions.
7. Authenticated Sandbox preview instances.
