# Workspace Cross-Resource Membership Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the first server-backed Workspace cross-resource membership slice for `workspace_note`, `media`, `workspace_source`, `workspace_artifact`, and `chat`.

**Architecture:** Add durable membership persistence to ChaChaNotes, then put a Workspace Core service and fail-closed resource adapter registry above it. Expose workspace-scoped membership routes under `/api/v1/workspaces/{workspace_id}/memberships` and a reverse lookup route under `/api/v1/workspace-memberships/resources/{resource_type}/{resource_id}`. Keep Research Workspace source selection, Project Workspace roots, MCP trust/path policy, ACP, and Sandbox runtime semantics separate.

**Tech Stack:** FastAPI, Pydantic, ChaChaNotes SQLite/PostgreSQL backend abstraction, Media DB read API, pytest, Bandit.

---

## Source Documents

- Spec: `Docs/superpowers/specs/2026-06-07-workspace-cross-resource-membership-design.md`
- Backlog task: `backlog/tasks/task-2315 - Design-Workspace-cross-resource-membership-foundation.md`
- Existing Workspace endpoint: `tldw_Server_API/app/api/v1/endpoints/workspaces.py`
- Existing Workspace schemas: `tldw_Server_API/app/api/v1/schemas/workspace_schemas.py`
- Existing Workspace Core helpers: `tldw_Server_API/app/core/Workspaces/`
- Existing Workspace DB methods: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`

## Scope

Implement:

- `workspace_resource_memberships` persistence in ChaChaNotes.
- Fail-closed adapters for `workspace_note`, `media`, `workspace_source`, `workspace_artifact`, and `chat`.
- Membership service with link, unlink, get, list-by-workspace, list-by-resource, summary resolution, archived workspace write rejection, restore-after-soft-delete, deterministic ordering, and cursor pagination.
- API schemas and endpoints.
- Explicit idempotent backfill helper for existing Workspace sub-resource rows.
- Compact membership summary in Workspace context.

Do not implement:

- UI adoption.
- Global `note` adapter.
- Prompt/workflow/watchlist/ACP/Sandbox/project-file adapters.
- Automatic startup backfill.
- MCP trust/path admission changes.

## File Structure

Create:

- `tldw_Server_API/app/core/Workspaces/membership_models.py`
  Typed dataclasses, constants, cursor helpers, and response payload helpers used by the membership service and tests.
- `tldw_Server_API/app/core/Workspaces/membership_adapters.py`
  Adapter protocol, adapter registry, and pilot resource adapters.
- `tldw_Server_API/app/core/Workspaces/membership_service.py`
  Workspace-level orchestration for validate/link/unlink/list/resolve/backfill.
- `tldw_Server_API/app/api/v1/endpoints/workspace_memberships.py`
  Reverse resource lookup endpoint mounted at `/api/v1/workspace-memberships`.
- `tldw_Server_API/tests/ChaChaNotesDB/test_workspace_resource_memberships_db.py`
- `tldw_Server_API/tests/Workspaces/test_workspace_membership_adapters.py`
- `tldw_Server_API/tests/Workspaces/test_workspace_memberships_api.py`
- `tldw_Server_API/tests/Workspaces/test_workspace_context_membership_summary.py`

Modify:

- `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
  Schema ensure path and membership persistence methods.
- `tldw_Server_API/app/api/v1/schemas/workspace_schemas.py`
  Membership request/response/list schemas and compact summary schema.
- `tldw_Server_API/app/api/v1/endpoints/workspaces.py`
  Workspace-scoped membership routes and context summary integration.
- `tldw_Server_API/app/api/v1/router_groups/content.py`
  Register the reverse-lookup router.
- `tldw_Server_API/app/api/v1/router_groups/minimal.py`
  Add route key if the minimal route group gates content endpoints by key.
- `tldw_Server_API/app/core/Workspaces/README.md`
  Document membership boundaries and extension checklist.
- `backlog/tasks/task-2315 - Design-Workspace-cross-resource-membership-foundation.md`
  Track implementation plan path and verification.

## Task 1: Persistence And DB Contract

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Test: `tldw_Server_API/tests/ChaChaNotesDB/test_workspace_resource_memberships_db.py`

- [ ] **Step 1: Write failing DB tests**

Create `tldw_Server_API/tests/ChaChaNotesDB/test_workspace_resource_memberships_db.py`.

Cover:

```python
def test_add_workspace_resource_membership_creates_row(db):
    row = db.add_workspace_resource_membership("ws-1", {
        "resource_type": "media",
        "resource_id": "42",
        "role": "source",
        "label": "Paper",
        "transfer_policy": "link",
        "provenance": {"source_surface": "library"},
    }, user_id="1")
    assert row["workspace_id"] == "ws-1"
    assert row["resource_type"] == "media"
    assert row["resource_id"] == "42"
    assert row["deleted"] in (False, 0)
```

Also cover:

- duplicate create with identical fields returns existing row.
- duplicate active row with different role or transfer policy raises `ConflictError`.
- delete soft-deletes and default list hides it.
- re-add after soft-delete restores the row after updating role/label/provenance.
- list order is `updated_at DESC`, `resource_type ASC`, `resource_id ASC`.
- list-by-resource returns all active workspace memberships for one resource.
- `delete_workspace` soft-deletes the workspace and does not erase membership history.
- `hard_delete_workspace`, if covered, removes membership rows through the FK cascade/schema cleanup path.
- PostgreSQL-like backend errors are wrapped into `CharactersRAGDBError` or `ConflictError` consistently where the existing backend abstraction surfaces `BackendDatabaseError`.

- [ ] **Step 2: Run the DB tests and verify they fail for missing methods**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_workspace_resource_memberships_db.py -q
```

Expected: fail with missing `add_workspace_resource_membership` or missing table.

- [ ] **Step 3: Add the membership table to SQLite and PostgreSQL ensure paths**

In `CharactersRAGDB._ensure_workspace_subresource_schema_sqlite`, create:

```sql
CREATE TABLE IF NOT EXISTS workspace_resource_memberships (
    workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    resource_type TEXT NOT NULL,
    resource_id TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'member',
    label TEXT,
    transfer_policy TEXT NOT NULL DEFAULT 'link',
    provenance_json TEXT NOT NULL DEFAULT '{}',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_by_user_id TEXT,
    updated_by_user_id TEXT,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted BOOLEAN NOT NULL DEFAULT 0,
    client_id TEXT NOT NULL DEFAULT 'unknown',
    version INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (workspace_id, resource_type, resource_id)
)
```

Add indexes:

```sql
CREATE INDEX IF NOT EXISTS idx_ws_resource_memberships_workspace
ON workspace_resource_memberships(workspace_id, deleted, resource_type, role);

CREATE INDEX IF NOT EXISTS idx_ws_resource_memberships_resource
ON workspace_resource_memberships(resource_type, resource_id, deleted);

CREATE INDEX IF NOT EXISTS idx_ws_resource_memberships_updated
ON workspace_resource_memberships(workspace_id, updated_at);
```

Mirror these statements in `_ensure_workspace_subresource_schema_postgres` with PostgreSQL boolean/timestamp defaults.

- [ ] **Step 4: Add normalization helpers and public getters for scoped resources**

Add or expose:

```python
def get_workspace_source(self, workspace_id: str, source_id: str) -> dict[str, Any] | None: ...
def get_workspace_note(self, workspace_id: str, note_id: int) -> dict[str, Any] | None: ...
def get_conversation_for_workspace_membership(self, conversation_id: str) -> dict[str, Any] | None: ...
```

`get_conversation_for_workspace_membership` should return non-deleted conversations and include `id`, `title` if present, `workspace_id`, `scope_type`, `last_modified`, and `version`.

- [ ] **Step 5: Add membership persistence methods**

Add:

```python
def add_workspace_resource_membership(
    self,
    workspace_id: str,
    data: dict[str, Any],
    *,
    user_id: str | None = None,
) -> dict[str, Any]: ...

def get_workspace_resource_membership(
    self,
    workspace_id: str,
    resource_type: str,
    resource_id: str,
    *,
    include_deleted: bool = False,
) -> dict[str, Any] | None: ...

def list_workspace_resource_memberships(
    self,
    workspace_id: str,
    *,
    resource_type: str | None = None,
    role: str | None = None,
    include_deleted: bool = False,
    limit: int = 100,
    cursor: tuple[str, str, str] | None = None,
) -> list[dict[str, Any]]: ...

def list_resource_workspace_memberships(
    self,
    resource_type: str,
    resource_id: str,
    *,
    include_deleted: bool = False,
    limit: int = 100,
    cursor: tuple[str, str] | None = None,
) -> list[dict[str, Any]]: ...

def delete_workspace_resource_membership(
    self,
    workspace_id: str,
    resource_type: str,
    resource_id: str,
    *,
    user_id: str | None = None,
) -> dict[str, Any] | None: ...
```

Implementation rules:

- Normalize `resource_type`, `resource_id`, `role`, and `transfer_policy`.
- Serialize `provenance` to `provenance_json` and `metadata` to `metadata_json`.
- Duplicate active rows with matching meaningful fields return existing.
- Duplicate active rows with conflicts raise `ConflictError(entity="workspace_resource_memberships")`.
- Duplicate deleted rows restore after the service has already validated resource access. DB method should restore when `data["restore_deleted"]` is true.
- For workspace-scoped pagination, fetch `limit + 1` rows and apply the cursor predicate matching `ORDER BY updated_at DESC, resource_type ASC, resource_id ASC`:
  `updated_at < cursor.updated_at OR (updated_at = cursor.updated_at AND resource_type > cursor.resource_type) OR (updated_at = cursor.updated_at AND resource_type = cursor.resource_type AND resource_id > cursor.resource_id)`.
- For reverse resource pagination, use a deterministic order such as `updated_at DESC, workspace_id ASC` and encode a separate cursor shape rather than reusing the workspace-list cursor.
- Catch `sqlite3.IntegrityError` and `BackendDatabaseError`. Treat uniqueness/constraint messages as conflict/duplicate paths; wrap unrelated backend errors in `CharactersRAGDBError`.

- [ ] **Step 6: Run DB tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_workspace_resource_memberships_db.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit Task 1**

```bash
git add tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_workspace_resource_memberships_db.py
git commit -m "feat: add workspace resource membership persistence"
```

## Task 2: Membership Models, Cursors, And API Schemas

**Files:**
- Create: `tldw_Server_API/app/core/Workspaces/membership_models.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/workspace_schemas.py`
- Test: `tldw_Server_API/tests/Workspaces/test_workspace_membership_adapters.py`

- [ ] **Step 1: Write failing model/schema tests**

In `test_workspace_membership_adapters.py`, add small tests for:

- valid cursor round-trip.
- invalid cursor rejected.
- request schema rejects unsupported role/transfer policy.
- bounded provenance/metadata JSON sizes.

- [ ] **Step 2: Run schema tests and verify failure**

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_membership_adapters.py -q
```

Expected: fail because modules/schemas do not exist.

- [ ] **Step 3: Create `membership_models.py`**

Include constants:

```python
WORKSPACE_MEMBERSHIP_RESOURCE_TYPES = frozenset({
    "workspace_note", "media", "workspace_source", "workspace_artifact", "chat",
})
WORKSPACE_MEMBERSHIP_FUTURE_RESOURCE_TYPES = frozenset({
    "note", "prompt", "workflow", "watchlist", "acp_session",
    "sandbox_session", "project_file", "study_deck", "quiz", "study_pack",
})
WORKSPACE_MEMBERSHIP_ROLES = frozenset({
    "member", "source", "artifact", "conversation", "runtime", "reference",
})
WORKSPACE_MEMBERSHIP_TRANSFER_POLICIES = frozenset({"link", "copy", "promote", "import"})
```

Add dataclasses:

```python
@dataclass(frozen=True)
class WorkspaceMembershipCursor:
    updated_at: str
    resource_type: str
    resource_id: str

@dataclass(frozen=True)
class WorkspaceResourceMembershipCursor:
    updated_at: str
    workspace_id: str

@dataclass(frozen=True)
class WorkspaceResourceRef:
    resource_type: str
    resource_id: str
    title: str | None = None
    subtitle: str | None = None
    href: str | None = None
    updated_at: str | None = None
    state: str = "available"
    metadata: Mapping[str, Any] = field(default_factory=dict)
```

Add `encode_membership_cursor` and `decode_membership_cursor` using base64-url JSON. Invalid cursors should raise `ValueError`.
Add `encode_resource_membership_cursor` and `decode_resource_membership_cursor` for reverse resource lookup pagination. Do not reuse the workspace-list cursor shape for the reverse route.

- [ ] **Step 4: Add Pydantic schemas**

In `workspace_schemas.py`, add:

```python
WorkspaceMembershipResourceType = Literal[
    "workspace_note", "media", "workspace_source", "workspace_artifact", "chat",
]
WorkspaceMembershipRole = Literal["member", "source", "artifact", "conversation", "runtime", "reference"]
WorkspaceMembershipTransferPolicy = Literal["link", "copy", "promote", "import"]

class WorkspaceMembershipCreateRequest(BaseModel): ...
class WorkspaceMembershipSummaryResponse(BaseModel): ...
class WorkspaceMembershipResponse(BaseModel): ...
class WorkspaceMembershipListSummary(BaseModel): ...
class WorkspaceMembershipListResponse(BaseModel): ...
class WorkspaceContextMembershipSummary(BaseModel): ...
```

Add JSON size validation using the existing `_validate_json_size` helper pattern. Use conservative limits:

- provenance: 16 KiB.
- metadata: 16 KiB.

- [ ] **Step 5: Run schema/model tests**

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_membership_adapters.py -q
```

Expected: model/schema tests pass. Adapter tests may still be skipped or xfailed until Task 3 if grouped in the same file.

- [ ] **Step 6: Commit Task 2**

```bash
git add tldw_Server_API/app/core/Workspaces/membership_models.py \
  tldw_Server_API/app/api/v1/schemas/workspace_schemas.py \
  tldw_Server_API/tests/Workspaces/test_workspace_membership_adapters.py
git commit -m "feat: add workspace membership schemas"
```

## Task 3: Adapter Registry And Membership Service

**Files:**
- Create: `tldw_Server_API/app/core/Workspaces/membership_adapters.py`
- Create: `tldw_Server_API/app/core/Workspaces/membership_service.py`
- Modify: `tldw_Server_API/app/core/Workspaces/__init__.py` if needed
- Test: `tldw_Server_API/tests/Workspaces/test_workspace_membership_adapters.py`

- [ ] **Step 1: Write failing adapter/service tests**

Cover:

- unsupported resource type fails closed.
- `workspace_note` validates a scoped note in the same workspace.
- `workspace_source` validates a source in the same workspace.
- `workspace_artifact` validates an artifact in the same workspace.
- `media` validates through `media_db_api.get_media_by_id`.
- `chat` allows a global conversation or a conversation already scoped to the same workspace, and rejects a conversation scoped to another workspace.
- archived workspace rejects `link_membership` with code `workspace_archived`.
- `resolve=false` list returns rows without adapter summaries.
- adapter read failure during list returns `summary.state = "unresolved"` without failing the whole list.

- [ ] **Step 2: Run adapter tests and verify failure**

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_membership_adapters.py -q
```

Expected: fail because adapters/service do not exist.

- [ ] **Step 3: Implement adapter protocol and registry**

In `membership_adapters.py`, define:

```python
class WorkspaceMembershipAdapter(Protocol):
    resource_type: str

    def validate_access(self, resource_id: str, context: WorkspaceMembershipContext) -> WorkspaceResourceRef: ...
    def summarize(self, resource_id: str, context: WorkspaceMembershipContext) -> WorkspaceResourceRef: ...
    def on_link(self, membership: Mapping[str, Any], context: WorkspaceMembershipContext) -> None: ...
    def on_unlink(self, membership: Mapping[str, Any], context: WorkspaceMembershipContext) -> None: ...
```

Implement:

- `WorkspaceNoteMembershipAdapter`
- `WorkspaceSourceMembershipAdapter`
- `WorkspaceArtifactMembershipAdapter`
- `MediaMembershipAdapter`
- `ChatMembershipAdapter`

Add:

```python
def default_workspace_membership_adapters() -> dict[str, WorkspaceMembershipAdapter]: ...
def get_workspace_membership_adapter(resource_type: str, adapters: Mapping[str, WorkspaceMembershipAdapter] | None = None) -> WorkspaceMembershipAdapter: ...
```

- [ ] **Step 4: Implement service errors and service**

In `membership_service.py`, define a local service error similar to `WorkspaceRootServiceError`:

```python
class WorkspaceMembershipServiceError(Exception):
    def __init__(self, code: str, message: str, *, status_code: int = 409): ...
```

Implement:

```python
class WorkspaceMembershipService:
    def link_membership(...): ...
    def get_membership(...): ...
    def list_workspace_memberships(...): ...
    def list_resource_memberships(...): ...
    def unlink_membership(...): ...
    def backfill_workspace_memberships(...): ...
    def workspace_membership_summary(...): ...
```

Service rules:

- `_require_workspace` loads `db.get_workspace(workspace_id)`.
- If missing, raise `workspace_not_found` with 404.
- If archived and write operation, raise `workspace_archived` with 409.
- Link validates adapter before DB insert/restore.
- Delete calls adapter `on_unlink` after soft-delete.
- List supports `resolve=false`.
- Resolved list failures are bounded and do not fail the entire list.
- Reverse lookup validates adapter support and canonicalizes resource ID before listing.

- [ ] **Step 5: Run adapter/service tests**

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_membership_adapters.py -q
```

Expected: all adapter/service tests pass.

- [ ] **Step 6: Commit Task 3**

```bash
git add tldw_Server_API/app/core/Workspaces/membership_adapters.py \
  tldw_Server_API/app/core/Workspaces/membership_service.py \
  tldw_Server_API/app/core/Workspaces/__init__.py \
  tldw_Server_API/tests/Workspaces/test_workspace_membership_adapters.py
git commit -m "feat: add workspace membership adapters"
```

## Task 4: Membership API Routes

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/workspaces.py`
- Create: `tldw_Server_API/app/api/v1/endpoints/workspace_memberships.py`
- Modify: `tldw_Server_API/app/api/v1/router_groups/content.py`
- Modify: `tldw_Server_API/app/api/v1/router_groups/minimal.py` if route keys require it
- Test: `tldw_Server_API/tests/Workspaces/test_workspace_memberships_api.py`

- [ ] **Step 1: Write failing API tests**

Create `test_workspace_memberships_api.py` with a local FastAPI app that includes:

```python
app.include_router(workspaces_endpoint.router, prefix="/api/v1/workspaces")
app.include_router(workspace_memberships_endpoint.router, prefix="/api/v1/workspace-memberships")
```

Cover:

- `POST /api/v1/workspaces/{workspace_id}/memberships` creates a membership.
- duplicate same request returns existing row.
- duplicate conflicting request returns 409.
- archived workspace write returns 409 with `workspace_archived`.
- `GET /api/v1/workspaces/{workspace_id}/memberships` filters by type/role.
- `DELETE /api/v1/workspaces/{workspace_id}/memberships/{resource_type}/{resource_id}` soft-deletes.
- re-link after delete restores.
- `GET /api/v1/workspace-memberships/resources/{resource_type}/{resource_id}` returns current user's workspace memberships.
- unsupported type returns stable error.
- missing media DB during `media` link returns service unavailable.

- [ ] **Step 2: Run API tests and verify failure**

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_memberships_api.py -q
```

Expected: fail because routes do not exist.

- [ ] **Step 3: Add workspace-scoped routes to `workspaces.py`**

Add imports for schemas and `WorkspaceMembershipService`.

Add routes:

```python
@router.get("/{workspace_id}/memberships", response_model=WorkspaceMembershipListResponse)
async def list_workspace_memberships(...): ...

@router.post("/{workspace_id}/memberships", response_model=WorkspaceMembershipResponse, status_code=201)
async def add_workspace_membership(...): ...

@router.get("/{workspace_id}/memberships/{resource_type}/{resource_id}", response_model=WorkspaceMembershipResponse)
async def get_workspace_membership(...): ...

@router.delete("/{workspace_id}/memberships/{resource_type}/{resource_id}", status_code=204)
async def delete_workspace_membership(...): ...
```

Use existing `WORKSPACES_READ_RATE_LIMIT`, `WORKSPACES_WRITE_RATE_LIMIT`, and `WORKSPACES_DELETE_RATE_LIMIT`.

- [ ] **Step 4: Add reverse lookup router**

Create `workspace_memberships.py` with:

```python
router = APIRouter()

@router.get("/resources/{resource_type}/{resource_id}", response_model=WorkspaceMembershipListResponse)
async def list_resource_workspace_memberships(...): ...
```

Use `get_chacha_db_for_user`, `try_get_media_db_for_user`, and `get_request_user`.

- [ ] **Step 5: Register reverse lookup router**

In `router_groups/content.py`, add a route entry:

```python
RouterSpec(
    import_path="tldw_Server_API.app.api.v1.endpoints.workspace_memberships",
    log_name="workspace_memberships",
    prefix=f"{API_V1_PREFIX}/workspace-memberships",
    tags=("workspaces",),
    route_key="workspaces",
)
```

Update `minimal.py` only if the current route group requires explicit route-key listings for this endpoint.

- [ ] **Step 6: Run API tests**

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_memberships_api.py -q
```

Expected: all API tests pass.

- [ ] **Step 7: Commit Task 4**

```bash
git add tldw_Server_API/app/api/v1/endpoints/workspaces.py \
  tldw_Server_API/app/api/v1/endpoints/workspace_memberships.py \
  tldw_Server_API/app/api/v1/router_groups/content.py \
  tldw_Server_API/app/api/v1/router_groups/minimal.py \
  tldw_Server_API/tests/Workspaces/test_workspace_memberships_api.py
git commit -m "feat: expose workspace membership api"
```

## Task 5: Backfill Helper And Context Summary

**Files:**
- Modify: `tldw_Server_API/app/core/Workspaces/membership_service.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/workspace_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/workspaces.py`
- Test: `tldw_Server_API/tests/Workspaces/test_workspace_context_membership_summary.py`

- [ ] **Step 1: Write failing backfill/context tests**

Cover:

- backfill creates `workspace_source`, `media`, `workspace_artifact`, `workspace_note`, and `chat` memberships from existing rows.
- backfill is idempotent.
- unresolved rows produce bounded diagnostics but do not delete or rewrite existing sub-resources.
- `GET /api/v1/workspaces/{workspace_id}/context` returns compact membership totals without full list.
- MCP permission/trust fields are not derived from generic membership summary.

- [ ] **Step 2: Run tests and verify failure**

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_context_membership_summary.py -q
```

Expected: fail because helper/summary do not exist.

- [ ] **Step 3: Implement explicit backfill helper**

In `WorkspaceMembershipService.backfill_workspace_memberships`, read:

- `db.list_workspace_sources(workspace_id)`
- `db.list_workspace_artifacts(workspace_id)`
- `db.list_workspace_notes(workspace_id)`
- workspace-scoped conversations through a new or existing DB helper.

Create memberships:

- `workspace_source` role `source`
- optional `media` role `source` when `media_id > 0`
- `workspace_artifact` role `artifact`
- `workspace_note` role `reference`
- `chat` role `conversation`

Do not run this automatically on startup or schema ensure.

- [ ] **Step 4: Add compact context summary**

Add `WorkspaceContextMembershipSummary` to `workspace_schemas.py`.

Add `memberships: WorkspaceContextMembershipSummary = Field(default_factory=WorkspaceContextMembershipSummary)` to `WorkspaceContextResponse`.

In `get_workspace_context`, call the service summary method and include:

```json
{
  "total": 14,
  "by_resource_type": {"media": 6, "workspace_note": 4},
  "by_role": {"source": 6, "reference": 4}
}
```

If summary resolution fails, add a partial error with `scope="memberships"` and return empty summary.

- [ ] **Step 5: Run context summary tests**

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_context_membership_summary.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit Task 5**

```bash
git add tldw_Server_API/app/core/Workspaces/membership_service.py \
  tldw_Server_API/app/api/v1/schemas/workspace_schemas.py \
  tldw_Server_API/app/api/v1/endpoints/workspaces.py \
  tldw_Server_API/tests/Workspaces/test_workspace_context_membership_summary.py
git commit -m "feat: add workspace membership backfill summary"
```

## Task 6: Documentation, Regression Suite, And Security

**Files:**
- Modify: `tldw_Server_API/app/core/Workspaces/README.md`
- Modify: `backlog/tasks/task-2315 - Design-Workspace-cross-resource-membership-foundation.md`

- [ ] **Step 1: Update Workspaces README**

Add a Membership section that states:

- `workspace_resource_memberships` is association, not ownership transfer.
- `workspace_sources` remains Research Workspace source selection/readiness.
- MCP effective permission preview/path admission uses MCP policy/root bindings, not generic membership.
- Future adapters must validate access through domain adapters.

- [ ] **Step 2: Run focused Python tests**

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/ChaChaNotesDB/test_workspace_resource_memberships_db.py \
  tldw_Server_API/tests/Workspaces/test_workspace_membership_adapters.py \
  tldw_Server_API/tests/Workspaces/test_workspace_memberships_api.py \
  tldw_Server_API/tests/Workspaces/test_workspace_context_membership_summary.py \
  tldw_Server_API/tests/Workspaces/test_workspaces_api.py \
  tldw_Server_API/tests/Workspaces/test_workspace_sub_resources_api.py \
  tldw_Server_API/tests/MCP_unified/test_mcp_hub_path_enforcement_service.py \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 3: Run broader Workspace regression if time allows**

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Workspaces -q
```

Expected: pass, or document unrelated failures with exact test names and errors.

- [ ] **Step 4: Run Bandit on touched Python paths**

```bash
source .venv/bin/activate
python -m bandit -r \
  tldw_Server_API/app/core/Workspaces/membership_models.py \
  tldw_Server_API/app/core/Workspaces/membership_adapters.py \
  tldw_Server_API/app/core/Workspaces/membership_service.py \
  tldw_Server_API/app/api/v1/endpoints/workspaces.py \
  tldw_Server_API/app/api/v1/endpoints/workspace_memberships.py \
  tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  -f json -o /tmp/bandit_workspace_memberships.json
```

Expected: no new findings in touched implementation paths. If Bandit reports existing baseline findings outside the changed membership code, document them and do not silently ignore new findings.

- [ ] **Step 5: Run diff checks**

```bash
git diff --check
git status --short --branch
```

Expected: no whitespace errors; only intended files modified.

- [ ] **Step 6: Update Backlog final summary**

Record:

- implemented DB/API/service/adapters.
- tests run and results.
- Bandit result.
- known skips or blockers.

- [ ] **Step 7: Commit Task 6**

```bash
git add tldw_Server_API/app/core/Workspaces/README.md \
  "backlog/tasks/task-2315 - Design-Workspace-cross-resource-membership-foundation.md"
git commit -m "docs: document workspace membership service"
```

## Parallelization Notes

After Task 1 lands, Tasks 2 and adapter test scaffolding can be done in parallel. After Task 2 lands, Task 3 adapters can be split by resource type with disjoint ownership:

- Worker A: `workspace_note`, `workspace_source`, `workspace_artifact`.
- Worker B: `media`, `chat`, adapter registry.
- Worker C: API tests and schema fixtures.

Do not parallelize edits to `ChaChaNotes_DB.py` unless the DB method surface has already landed. It is large and conflict-prone.

## Implementation Guardrails

- Do not use generic membership to filter global Library/Notes/search.
- Do not make membership a permission source for MCP tool execution.
- Do not auto-run backfill during startup.
- Do not implement global `note` until the adapter can prove access cleanly.
- Keep `absolute_root`, sandbox mount paths, prompts, model outputs, and file contents out of membership provenance and summaries.
- Treat unsupported resource types as fail-closed errors.
- Use `apply_patch` for manual edits.
- Use TDD: write the failing test first for each task, then implement the narrowest passing change.
