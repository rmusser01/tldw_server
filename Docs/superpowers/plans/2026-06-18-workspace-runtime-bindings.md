# Workspace Runtime Bindings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Workspace Phase 2 runtime binding descriptors for issue #1991 with durable persistence, secret-safe metadata handling, and focused API routes.

**Architecture:** Add a Workspace-owned runtime binding descriptor layer separate from project roots and resource memberships. The new core helper normalizes binding vocabulary and redacts/rejects sensitive metadata before persistence; ChaChaNotes stores descriptor rows; the Workspaces API exposes read/write/archive routes using existing workspace lifecycle and error mapping patterns.

**Tech Stack:** Python 3.11, FastAPI, Pydantic v2, SQLite/PostgreSQL-compatible ChaChaNotesDB helpers, pytest.

---

## Scope

This is a backend-only first slice for GitHub issue #1991 and Backlog task `TASK-2381`.

Implement:

- Canonical binding vocabulary:
  - `binding_kind`: `repo`, `git_worktree`, `local_path`, `workspace_project_root`, `acp_execution_workspace`, `acp_session`, `acp_run`, `sandbox_root`, `sandbox_session`, `mcp_workspace_set`, `remote_runtime`
  - `owner_domain`: `workspaces`, `acp`, `sandbox`, `mcp`, `jobs`, `workflows`, `watchlists`, `external`
  - `status`: `ready`, `missing`, `inspect-only`, `blocked`, `provisioning`, `unavailable`, `detached`, `conflict`, `runtime-missing`, `archived`, `unsupported`
  - `portability`: `reference`, `metadata-only`, `local-only`, `copy`
- Secret-safe metadata normalization:
  - Reject sensitive metadata keys by default.
  - Redact sensitive values in `path_hint` and descriptor metadata response paths.
  - Persist `redaction_report` with omitted/rejected field names.
  - Accept underscore status/portability aliases (`inspect_only`, `runtime_missing`, `metadata_only`, `local_only`) and normalize them to the contract's hyphenated wire values.
- Durable descriptor persistence:
  - Upsert one active descriptor by `(workspace_id, binding_id)`.
  - List active descriptors, optionally filtered by `binding_kind` or `owner_domain`.
  - Get one active descriptor.
  - Archive one descriptor idempotently.
- API routes:
  - `POST /api/v1/workspaces/{workspace_id}/runtime-bindings`
  - `GET /api/v1/workspaces/{workspace_id}/runtime-bindings`
  - `GET /api/v1/workspaces/{workspace_id}/runtime-bindings/{binding_id}`
  - `DELETE /api/v1/workspaces/{workspace_id}/runtime-bindings/{binding_id}`
- Docs:
  - Update `tldw_Server_API/app/core/Workspaces/README.md`.
  - Keep the contract untouched unless implementation discovers ambiguity.

Do not implement:

- ACP or Sandbox runtime resume.
- Secret storage.
- Root/path trust admission.
- Frontend UI.
- Membership adapters for ACP/Sandbox. That remains #2378.

## File Map

- Create `tldw_Server_API/app/core/Workspaces/runtime_bindings.py`
  - Runtime binding literals, normalization, path hint redaction, metadata sanitizer, and response payload shaping.
- Modify `tldw_Server_API/app/core/Workspaces/models.py`
  - Re-export or define literal types where core Workspace code already centralizes vocabulary.
- Modify `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
  - Add schema table/indexes for SQLite and PostgreSQL.
  - Add normalize/serialize/deserialize helpers.
  - Add upsert/list/get/archive methods.
- Modify `tldw_Server_API/app/api/v1/schemas/workspace_schemas.py`
  - Add request/response schemas and enum literals.
- Modify `tldw_Server_API/app/api/v1/endpoints/workspaces.py`
  - Add route handlers and map DB/core errors to HTTP responses.
- Create `tldw_Server_API/tests/Workspaces/test_workspace_runtime_bindings.py`
  - Core sanitizer and DB persistence tests.
- Create `tldw_Server_API/tests/Workspaces/test_workspace_runtime_bindings_api.py`
  - FastAPI route tests.
- Modify `tldw_Server_API/app/core/Workspaces/README.md`
  - Document the runtime binding descriptor contract.
- Modify `backlog/tasks/task-2381 - Implement-Workspace-runtime-binding-descriptors-for-issue-1991.md`
  - Track plan path, implementation notes, verification, and closeout.

## Task 1: Core Runtime Binding Normalizer

**Files:**

- Create: `tldw_Server_API/app/core/Workspaces/runtime_bindings.py`
- Modify: `tldw_Server_API/app/core/Workspaces/models.py`
- Test: `tldw_Server_API/tests/Workspaces/test_workspace_runtime_bindings.py`

- [x] **Step 1: Write failing tests for descriptor normalization**

Add tests like:

```python
import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import InputError
from tldw_Server_API.app.core.Workspaces.runtime_bindings import (
    normalize_runtime_binding_payload,
)


def test_runtime_binding_normalizer_redacts_path_and_preserves_safe_metadata():
    payload = normalize_runtime_binding_payload(
        {
            "binding_id": "repo-main",
            "binding_kind": "repo",
            "owner_domain": "workspaces",
            "locator_ref": "repo-123",
            "label": "Main Repo",
            "status": "ready",
            "path_hint": "/Users/example/private/project",
            "portability": "reference",
            "metadata": {"branch": "main", "remote": "origin"},
        }
    )

    assert payload["path_hint"] == "project"
    assert payload["metadata"] == {"branch": "main", "remote": "origin"}
    assert payload["redaction_report"]["redacted_fields"] == ["path_hint"]


def test_runtime_binding_normalizer_normalizes_contract_aliases():
    payload = normalize_runtime_binding_payload(
        {
            "binding_id": "sandbox-root",
            "binding_kind": "sandbox_root",
            "owner_domain": "sandbox",
            "locator_ref": "sandbox-123",
            "status": "inspect_only",
            "portability": "metadata_only",
        }
    )

    assert payload["status"] == "inspect-only"
    assert payload["portability"] == "metadata-only"


def test_runtime_binding_normalizer_rejects_secret_metadata_keys():
    with pytest.raises(InputError):
        normalize_runtime_binding_payload(
            {
                "binding_id": "acp-session",
                "binding_kind": "acp_session",
                "owner_domain": "acp",
                "locator_ref": "session-123",
                "label": "ACP Session",
                "status": "ready",
                "portability": "metadata-only",
                "metadata": {"OPENAI_API_KEY": "sk-secret"},
            }
        )
```

- [x] **Step 2: Run tests to verify RED**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_runtime_bindings.py -q
```

Expected: fail because `runtime_bindings.py` does not exist.

- [x] **Step 3: Implement minimal normalizer**

Create `runtime_bindings.py` with:

- frozensets for allowed literals.
- `_normalized_required_string(value, field_name, max_length)`.
- `redacted_path_hint(value)` matching existing root redaction behavior.
- `normalize_runtime_binding_payload(data)`.
- `runtime_binding_response_payload(row)`.

Use `InputError` for malformed values. Keep the first slice intentionally strict: secret-looking metadata keys cause `InputError`, while path hints are redacted to basename.

- [x] **Step 4: Run tests to verify GREEN**

Run the same pytest command. Expected: pass.

- [x] **Step 5: Refactor**

Remove duplicate path-hint logic only within the new module. Do not refactor existing root helpers in this task.

## Task 2: Durable DB Persistence

**Files:**

- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Test: `tldw_Server_API/tests/Workspaces/test_workspace_runtime_bindings.py`

- [x] **Step 1: Write failing DB CRUD/archive tests**

Add tests:

```python
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


def test_workspace_runtime_binding_upsert_list_get_and_archive(tmp_path):
    db = CharactersRAGDB(db_path=str(tmp_path / "chacha.db"), client_id="user-1")
    db.upsert_workspace("ws-1", "Workspace")

    created = db.upsert_workspace_runtime_binding(
        "ws-1",
        {
            "binding_id": "repo-main",
            "binding_kind": "repo",
            "owner_domain": "workspaces",
            "locator_ref": "repo-123",
            "label": "Main Repo",
            "status": "ready",
            "path_hint": "/Users/example/project",
            "portability": "reference",
            "metadata": {"branch": "main"},
        },
        user_id="user-1",
    )

    assert created["path_hint"] == "project"
    assert created["metadata"] == {"branch": "main"}
    assert db.get_workspace_runtime_binding("ws-1", "repo-main")["binding_id"] == "repo-main"
    assert [item["binding_id"] for item in db.list_workspace_runtime_bindings("ws-1")] == ["repo-main"]

    archived = db.archive_workspace_runtime_binding("ws-1", "repo-main", user_id="user-1")
    assert archived["status"] == "archived"
    assert archived["deleted"] in (True, 1)
    assert db.get_workspace_runtime_binding("ws-1", "repo-main") is None
    assert db.get_workspace_runtime_binding("ws-1", "repo-main", include_deleted=True)["status"] == "archived"
```

- [x] **Step 2: Run tests to verify RED**

Expected: fail because DB methods/table do not exist.

- [x] **Step 3: Add schema**

In `_ensure_workspace_subresource_schema_sqlite`, create:

```sql
CREATE TABLE IF NOT EXISTS workspace_runtime_bindings (
    workspace_id        TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    binding_id          TEXT NOT NULL,
    binding_kind        TEXT NOT NULL,
    owner_domain        TEXT NOT NULL,
    locator_ref         TEXT NOT NULL,
    label               TEXT,
    status              TEXT NOT NULL,
    path_hint           TEXT,
    portability         TEXT NOT NULL,
    metadata_json       TEXT NOT NULL DEFAULT '{}',
    redaction_report_json TEXT NOT NULL DEFAULT '{}',
    created_by_user_id  TEXT,
    updated_by_user_id  TEXT,
    created_at          DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at          DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted             BOOLEAN NOT NULL DEFAULT 0,
    client_id           TEXT NOT NULL DEFAULT 'unknown',
    version             INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (workspace_id, binding_id)
)
```

Add indexes:

- `idx_ws_runtime_bindings_workspace ON workspace_runtime_bindings(workspace_id, deleted, binding_kind, owner_domain)`
- `idx_ws_runtime_bindings_locator ON workspace_runtime_bindings(owner_domain, locator_ref, deleted)`
- `idx_ws_runtime_bindings_updated ON workspace_runtime_bindings(workspace_id, updated_at)`

Add PostgreSQL statements in `_ensure_workspace_subresource_schema_postgres` with `BOOLEAN DEFAULT false` and `TIMESTAMP`.

- [x] **Step 4: Add DB methods**

Add methods near membership methods:

- `_normalize_workspace_runtime_binding_row`
- `_workspace_runtime_binding_write_error`
- `_get_workspace_runtime_binding_with_conn`
- `upsert_workspace_runtime_binding`
- `get_workspace_runtime_binding`
- `list_workspace_runtime_bindings`
- `archive_workspace_runtime_binding`

Use `normalize_runtime_binding_payload()` from the core module. Follow membership methods for idempotency, soft delete, version increment, and BackendDatabaseError wrapping.

- [x] **Step 5: Run tests to verify GREEN**

Run:

```bash
python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_runtime_bindings.py -q
```

Expected: pass.

## Task 3: API Schemas And Routes

**Files:**

- Modify: `tldw_Server_API/app/api/v1/schemas/workspace_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/workspaces.py`
- Test: `tldw_Server_API/tests/Workspaces/test_workspace_runtime_bindings_api.py`

- [x] **Step 1: Write failing API tests**

Create tests using the existing FastAPI app fixture style from `test_workspaces_api.py`:

```python
def test_workspace_runtime_binding_api_upserts_lists_gets_and_archives(workspace_fastapi_app, db):
    db.upsert_workspace("ws-runtime", "Runtime Workspace")
    response = _post_runtime_binding(
        workspace_fastapi_app,
        db,
        "ws-runtime",
        {
            "binding_id": "acp-session-1",
            "binding_kind": "acp_session",
            "owner_domain": "acp",
            "locator_ref": "session-1",
            "label": "ACP Session",
            "status": "runtime-missing",
            "path_hint": "/Users/example/agent-workspace",
            "portability": "metadata-only",
            "metadata": {"agent": "codex"},
        },
    )
    assert response.status_code == 201
    created = response.json()
    assert created["path_hint"] == "agent-workspace"
    assert "absolute_root" not in created

    listed = _list_runtime_bindings(workspace_fastapi_app, db, "ws-runtime")
    assert listed.status_code == 200
    assert listed.json()["items"][0]["binding_id"] == "acp-session-1"

    fetched = _get_runtime_binding(workspace_fastapi_app, db, "ws-runtime", "acp-session-1")
    assert fetched.status_code == 200

    deleted = _delete_runtime_binding(workspace_fastapi_app, db, "ws-runtime", "acp-session-1")
    assert deleted.status_code == 204
    assert _get_runtime_binding(workspace_fastapi_app, db, "ws-runtime", "acp-session-1").status_code == 404
```

Also test:

- Unknown binding kind returns 422.
- Secret-looking metadata key returns 422.
- Missing workspace returns 404.
- Archived workspace rejects upsert/archive with 409 or existing writable-workspace pattern.

- [x] **Step 2: Run tests to verify RED**

Expected: fail because schemas/routes do not exist.

- [x] **Step 3: Add Pydantic schemas**

Add literals and models:

- `WorkspaceRuntimeBindingKind`
- `WorkspaceRuntimeBindingOwnerDomain`
- `WorkspaceRuntimeBindingStatus`
- `WorkspaceRuntimeBindingPortability`
- `WorkspaceRuntimeBindingUpsertRequest`
- `WorkspaceRuntimeBindingRedactionReport`
- `WorkspaceRuntimeBindingResponse`
- `WorkspaceRuntimeBindingListResponse`

Use `extra="forbid"`, field length bounds, bounded metadata validation, and pass values through `normalize_runtime_binding_payload` where possible.

- [x] **Step 4: Add routes**

In `workspaces.py`, import new schemas and add helpers:

- `_runtime_binding_to_response(row)`
- `_runtime_binding_not_found(binding_id)`
- `_runtime_binding_write_forbidden(workspace)`

Add routes after root routes or before membership routes:

- POST upsert: require workspace exists and is not archived/deleted; return 201 for create and 200 for update is acceptable only if implementation can detect update. Prefer 201 for both unless tests enforce version behavior.
- GET list: supports `binding_kind`, `owner_domain`, `include_archived=false`, `limit`.
- GET one: 404 if absent/deleted.
- DELETE/archive: 204, idempotent for missing active rows if existing endpoint patterns prefer no-op; otherwise 404. Choose 204 for idempotent archive if row already archived, 404 if never existed.

- [x] **Step 5: Run tests to verify GREEN**

Run:

```bash
python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_runtime_bindings_api.py -q
```

Expected: pass.

## Task 4: Docs, Backlog, And Local Verification

**Files:**

- Modify: `tldw_Server_API/app/core/Workspaces/README.md`
- Modify: `backlog/tasks/task-2381 - Implement-Workspace-runtime-binding-descriptors-for-issue-1991.md`

- [x] **Step 1: Update README**

Add a `Runtime Bindings` section that states:

- Runtime bindings are descriptor metadata, not trust grants.
- Public responses expose `path_hint` and `redaction_report`, not secrets or raw absolute paths.
- ACP/Sandbox/MCP remain admission owners.
- The API surface is under `/api/v1/workspaces/{workspace_id}/runtime-bindings`.

- [x] **Step 2: Update Backlog task**

Set `TASK-2381` to `In Progress`, add implementation plan path, touched files, and verification notes as commands are run.

- [x] **Step 3: Run focused verification**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Workspaces/test_workspace_runtime_bindings.py \
  tldw_Server_API/tests/Workspaces/test_workspace_runtime_bindings_api.py \
  tldw_Server_API/tests/Workspaces/test_workspace_project_roots_db.py \
  tldw_Server_API/tests/Workspaces/test_workspaces_api.py -q
```

- [x] **Step 4: Run Bandit on touched backend paths**

Run:

```bash
python -m bandit -r \
  tldw_Server_API/app/core/Workspaces/runtime_bindings.py \
  tldw_Server_API/app/core/Workspaces/models.py \
  tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  tldw_Server_API/app/api/v1/schemas/workspace_schemas.py \
  tldw_Server_API/app/api/v1/endpoints/workspaces.py \
  -f json -o /tmp/bandit_workspace_runtime_bindings.json
```

Expected: no new findings in changed code. If Bandit reports pre-existing findings in large touched files, document exact IDs and confirm whether changed lines are implicated.

- [x] **Step 5: Run formatting/diff checks**

Run:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; only intended files changed.

## Task 5: Commit And PR

**Files:**

- All touched files from prior tasks.

- [x] **Step 1: Final Backlog update**

Mark acceptance criteria complete, add final summary, DoD, verification results, and known skips/blockers.

- [x] **Step 2: Review diff**

Run:

```bash
git diff --stat
git diff -- tldw_Server_API/app/core/Workspaces/runtime_bindings.py
git diff -- tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py
git diff -- tldw_Server_API/app/api/v1/endpoints/workspaces.py
```

- [ ] **Step 3: Commit**

Run:

```bash
git add \
  "backlog/tasks/task-2381 - Implement-Workspace-runtime-binding-descriptors-for-issue-1991.md" \
  Docs/superpowers/plans/2026-06-18-workspace-runtime-bindings.md \
  tldw_Server_API/app/core/Workspaces/README.md \
  tldw_Server_API/app/core/Workspaces/models.py \
  tldw_Server_API/app/core/Workspaces/runtime_bindings.py \
  tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  tldw_Server_API/app/api/v1/schemas/workspace_schemas.py \
  tldw_Server_API/app/api/v1/endpoints/workspaces.py \
  tldw_Server_API/tests/Workspaces/test_workspace_runtime_bindings.py \
  tldw_Server_API/tests/Workspaces/test_workspace_runtime_bindings_api.py
git commit -m "feat: add workspace runtime binding descriptors"
```

- [ ] **Step 4: Push and open PR**

Run:

```bash
git push -u origin codex/workspace-runtime-bindings
gh pr create --base dev --head codex/workspace-runtime-bindings --title "Add Workspace runtime binding descriptors" --body-file /tmp/workspace_runtime_bindings_pr.md
```

PR body must include:

- Change summary with what changed and why.
- Test plan with exact commands and outcomes.
- Links to #1991, #1984, and `TASK-2381`.
- Note that runtime bindings are descriptors only, not trust grants or secret stores.
