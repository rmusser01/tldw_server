# Canonical Workspaces Manager Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the canonical `/workspaces` manager and the backend contracts needed to create, edit, archive, unarchive, and upgrade Workspaces into Project Workspaces with host-local or sandbox-managed primary roots.

**Architecture:** Workspace Core remains the canonical product identity. Sandbox owns durable runtime volume mechanics. ACP, MCP Hub, and Prototype Workspaces remain specialized projections or policy surfaces, and the new manager links to them without reusing their response models as canonical Workspace models. Backend work lands first so the WebUI can consume stable normalized read models.

**Tech Stack:** FastAPI, Pydantic, `CharactersRAGDB`, Sandbox store/services, pytest, Next/React WebUI, TypeScript, Vitest, Testing Library, Playwright/CDP.

---

## Scope Check

The approved spec spans backend contracts, WebUI manager UI, local Research Workspace reconciliation, Sandbox volume ownership, and cross-surface validation. These are related but independently reviewable, so implement them as sequential PR-sized tasks with clear parallel seams:

1. Backend canonical read model and operation envelope.
2. WebUI API client parity and normalized manager models.
3. Sandbox durable workspace-volume contract.
4. Workspace-owned sandbox provision-and-attach command.
5. Canonical `/workspaces` manager CRUD.
6. Project upgrade and root panel.
7. Local Research Workspace reconciliation.
8. Cross-surface links and live UAT.

Tasks 1 and 2 can proceed in parallel after this plan is accepted. Task 4 depends on Task 3. Tasks 5 and 6 depend on Task 2; Task 6 also depends on Task 4. Task 7 depends on Task 5. Task 8 validates all completed slices.

## File Structure

Backend files to modify:

- `tldw_Server_API/app/api/v1/schemas/workspace_schemas.py`
  - Add manager-facing fields, operation schemas, sandbox provision request/response schemas, and any missing enum values.
- `tldw_Server_API/app/api/v1/endpoints/workspaces.py`
  - Add operation polling and sandbox root provision-and-attach endpoints.
  - Keep canonical Workspace API ownership here.
- `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
  - Add Workspace operation/idempotency persistence and any required project-root state methods.
- `tldw_Server_API/app/core/Workspaces/models.py`
  - Add shared attention, mount, inventory availability, and operation state normalization helpers.
- `tldw_Server_API/app/core/Workspaces/context.py`
  - Add active operations, attention state, file inventory availability, and shared Sandbox-to-Workspace projection use.
- `tldw_Server_API/app/core/Workspaces/root_binding_service.py`
  - Extend sandbox volume state support from the current conservative set.
- `tldw_Server_API/app/core/Workspaces/operations.py`
  - New module for Workspace operation records, idempotency fingerprinting, redaction, status projection, and retention.
- `tldw_Server_API/app/core/Workspaces/sandbox_root_provisioning.py`
  - New module for the Workspace-owned product command that provisions a sandbox volume and attaches it as the primary root.
- `tldw_Server_API/app/core/Sandbox/models.py`
  - Add durable workspace-volume dataclasses/enums.
- `tldw_Server_API/app/core/Sandbox/store.py`
  - Add durable workspace-volume persistence methods and SQLite schema.
- `tldw_Server_API/app/core/Sandbox/workspace_volumes.py`
  - New module for Sandbox-owned durable workspace-volume service behavior.

Backend tests to add or modify:

- `tldw_Server_API/tests/Workspaces/test_workspace_core_models.py`
- `tldw_Server_API/tests/Workspaces/test_workspace_core_context.py`
- `tldw_Server_API/tests/Workspaces/test_workspace_root_binding_service.py`
- `tldw_Server_API/tests/Workspaces/test_workspaces_api.py`
- `tldw_Server_API/tests/Workspaces/test_workspace_project_roots_db.py`
- `tldw_Server_API/tests/Workspaces/test_workspace_operations.py`
- `tldw_Server_API/tests/Workspaces/test_workspace_sandbox_root_provisioning.py`
- `tldw_Server_API/tests/sandbox/test_workspace_volumes.py`

Frontend files to modify:

- `apps/packages/ui/src/services/tldw/domains/workspace-api.ts`
  - Add missing Workspace Core methods and response types.
- `apps/packages/ui/src/services/__tests__/tldw-api-client.workspace-api.test.ts`
  - Add route/method/body coverage for client parity.
- `apps/packages/ui/src/routes/route-paths.ts`
  - Add `WORKSPACES_PATH` and deep-link helpers.
- `apps/packages/ui/src/routes/route-metadata.ts`
  - Register `/workspaces` without aliases or redirects.
- `apps/packages/ui/src/routes/route-registry.tsx`
  - Register the new route component.
- `apps/packages/ui/src/routes/option-workspaces.tsx`
  - New route shell.
- `apps/packages/ui/src/components/Option/Workspaces/WorkspacesManagerPage.tsx`
  - New canonical manager page.
- `apps/packages/ui/src/components/Option/Workspaces/workspace-manager-models.ts`
  - New canonical normalization helpers.
- `apps/packages/ui/src/components/Option/Workspaces/workspace-manager-copy.ts`
  - New copy/label helpers, including ACP/MCP naming guardrails.
- `apps/packages/ui/src/components/Option/Workspaces/WorkspaceList.tsx`
  - New list/filter/search table or dense list.
- `apps/packages/ui/src/components/Option/Workspaces/WorkspaceCreateDialog.tsx`
  - New create Research/Project shell flow.
- `apps/packages/ui/src/components/Option/Workspaces/WorkspaceMetadataDialog.tsx`
  - New edit/archive/unarchive flow.
- `apps/packages/ui/src/components/Option/Workspaces/WorkspaceProjectRootPanel.tsx`
  - New root/upgrade/status panel.
- `apps/packages/ui/src/components/Option/Workspaces/WorkspaceReconciliationPanel.tsx`
  - New local Research Workspace reconciliation panel.
- `apps/packages/ui/src/components/Option/Workspaces/workspace-local-reconciliation.ts`
  - New local-only detection, marker, and dry-run logic.
- `apps/packages/ui/src/components/Option/ResearchWorkspace/WorkspaceHeader.tsx`
  - Add link back to canonical manager where appropriate.
- `apps/packages/ui/src/components/Option/MCPHub/SharedWorkspacesTab.tsx`
  - Add copy/link guardrail only if cross-surface links need it.
- `apps/packages/ui/src/components/Option/ACPPlayground/ACPWorkspacePanel.tsx`
  - Add copy/link guardrail only if cross-surface links need it.

Frontend tests to add or modify:

- `apps/packages/ui/src/components/Option/Workspaces/__tests__/WorkspacesManagerPage.test.tsx`
- `apps/packages/ui/src/components/Option/Workspaces/__tests__/workspace-manager-models.test.ts`
- `apps/packages/ui/src/components/Option/Workspaces/__tests__/WorkspaceProjectRootPanel.test.tsx`
- `apps/packages/ui/src/components/Option/Workspaces/__tests__/workspace-local-reconciliation.test.ts`
- `apps/packages/ui/src/routes/__tests__/option-workspaces.route.test.tsx`
- `apps/packages/ui/src/routes/__tests__/route-metadata.coverage.test.ts`
- `apps/packages/ui/src/routes/__tests__/route-registry.visibility.test.ts`
- `apps/packages/ui/src/design-system/__tests__/product-state-guard.mcp-acp-workspace-baseline.test.ts`

Validation files:

- `apps/tldw-frontend/e2e/workflows/workspaces-manager.spec.ts`
  - New Playwright/CDP smoke coverage for live WebUI.
- `Docs/Validation/workspaces-manager-uat-matrix.md`
  - New live validation matrix and results template.

## Implementation Tasks

### Task 1: Backend Canonical Read Model And Operation Envelope

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/workspace_schemas.py`
- Modify: `tldw_Server_API/app/core/Workspaces/models.py`
- Modify: `tldw_Server_API/app/core/Workspaces/context.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/workspaces.py`
- Test: `tldw_Server_API/tests/Workspaces/test_workspace_core_models.py`
- Test: `tldw_Server_API/tests/Workspaces/test_workspace_core_context.py`
- Test: `tldw_Server_API/tests/Workspaces/test_workspaces_api.py`

- [ ] **Step 1: Write failing model tests for manager projection states**

Add tests like:

```python
def test_attention_state_projects_project_without_root_to_setup_pending() -> None:
    assert workspace_attention_state(
        workspace_profile="project",
        project_root_state="not_configured",
        inventory_state="not_started",
        service_errors=[],
        archived=False,
    ) == "setup_pending"


def test_sandbox_projection_ready_without_mount_fails_closed() -> None:
    projection = project_sandbox_volume_projection("ready", usable_mount=False)
    assert projection["root_state"] == "attached"
    assert projection["mount_state"] == "not_ready"
    assert projection["file_inventory"]["available"] is False
    assert projection["attention_state"] == "needs_attention"
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_core_models.py -v
```

Expected: FAIL because `workspace_attention_state` and projection helpers do not exist yet.

- [ ] **Step 3: Add shared model helpers**

In `tldw_Server_API/app/core/Workspaces/models.py`, add typed literals and helpers:

```python
WorkspaceAttentionState = Literal[
    "ready",
    "setup_pending",
    "working",
    "needs_attention",
    "blocked",
    "archived",
]


def workspace_attention_state(
    *,
    workspace_profile: Any,
    project_root_state: Any,
    inventory_state: Any,
    service_errors: list[str] | None = None,
    archived: bool = False,
) -> WorkspaceAttentionState:
    ...


def project_sandbox_volume_projection(
    sandbox_state: Any,
    *,
    usable_mount: bool,
) -> dict[str, Any]:
    ...
```

Map Sandbox states exactly as the spec defines. Unknown states must fail closed.

- [ ] **Step 4: Add schema fields for manager context and operation status**

In `workspace_schemas.py`, add:

```python
WorkspaceOperationStatus = Literal[
    "queued",
    "running",
    "succeeded",
    "failed",
    "conflicted",
    "expired",
]


class WorkspaceOperationResponse(BaseModel):
    operation_id: str
    workspace_id: str
    command: str
    status: WorkspaceOperationStatus
    started_at: str
    updated_at: str
    retryable: bool = False
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    poll_href: str
```

Extend `WorkspaceFileInventory` with `available: bool = False`. Extend `WorkspaceContextResponse` with `attention_state` and `active_operations: list[WorkspaceOperationResponse]`.

- [ ] **Step 5: Update `build_workspace_core_context`**

In `context.py`, compute:

- `file_inventory.available`
- `attention_state`
- `active_operations`, defaulting to `[]` until Task 4 persists operations

Use the shared helpers from `models.py`; do not derive attention state inside components later.

- [ ] **Step 6: Write context tests**

Add coverage for:

- Research Workspace with no root returns `ready`
- Project Workspace with no root returns `setup_pending`
- Project Workspace with root unavailable returns `needs_attention` or `blocked`
- Project Workspace with inventory queued/running returns `working`
- Archived Workspace returns `archived`
- `file_inventory.available` is false for sandbox roots without ready mount
- `active_operations` is present and empty

- [ ] **Step 7: Run focused backend tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_core_models.py tldw_Server_API/tests/Workspaces/test_workspace_core_context.py tldw_Server_API/tests/Workspaces/test_workspaces_api.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit Task 1**

```bash
git add tldw_Server_API/app/api/v1/schemas/workspace_schemas.py tldw_Server_API/app/core/Workspaces/models.py tldw_Server_API/app/core/Workspaces/context.py tldw_Server_API/app/api/v1/endpoints/workspaces.py tldw_Server_API/tests/Workspaces/test_workspace_core_models.py tldw_Server_API/tests/Workspaces/test_workspace_core_context.py tldw_Server_API/tests/Workspaces/test_workspaces_api.py
git commit -m "feat: add workspace manager read model"
```

### Task 2: WebUI Workspace API Client Parity And Normalized Models

**Files:**
- Modify: `apps/packages/ui/src/services/tldw/domains/workspace-api.ts`
- Modify: `apps/packages/ui/src/services/__tests__/tldw-api-client.workspace-api.test.ts`
- Create: `apps/packages/ui/src/components/Option/Workspaces/workspace-manager-models.ts`
- Create: `apps/packages/ui/src/components/Option/Workspaces/workspace-manager-copy.ts`
- Test: `apps/packages/ui/src/components/Option/Workspaces/__tests__/workspace-manager-models.test.ts`

- [ ] **Step 1: Write failing client route tests**

Add tests for:

- `patchWorkspace`
- `deleteWorkspace` only as raw API support, not manager UI delete
- `getWorkspaceRoots`
- `attachWorkspacePrimaryRoot`
- `queueWorkspaceFileInventoryScan`
- `getWorkspaceFileInventoryStatus`
- `getWorkspaceFileInventoryItems`
- current Workspace source, artifact, and note sub-resource methods

Do not add callables for `GET /operations/{operation_id}` or
`POST /roots/primary/sandbox-volume` in Task 2. Those backend routes land in
Task 4, and the frontend callables are added with the Project root panel slice
after the backend contract exists.

- [ ] **Step 2: Run client tests and verify they fail**

Run:

```bash
bunx vitest run apps/packages/ui/src/services/__tests__/tldw-api-client.workspace-api.test.ts
```

Expected: FAIL because methods/types are missing.

- [ ] **Step 3: Add canonical Workspace API types and methods**

Update `workspace-api.ts`:

- Add `workspace_profile` to `WorkspaceApiResponse`.
- Add `WorkspaceRootsResponse`, `WorkspaceRootResponse`, and `WorkspaceOperationResponse` for `context.active_operations`.
- Add methods listed in Step 1.
- Encode backend-valid path segment IDs and reject slash-delimited IDs that
  FastAPI segment routes cannot match as a single parameter.

- [ ] **Step 4: Add normalized manager model helpers**

Create `workspace-manager-models.ts`:

```ts
export type WorkspaceManagerAttention =
  | "ready"
  | "setup_pending"
  | "working"
  | "needs_attention"
  | "blocked"
  | "archived"

export const normalizeWorkspaceManagerItem = (
  workspace: WorkspaceApiResponse,
  context?: WorkspaceContextResponse | null
): WorkspaceManagerItem => {
  ...
}
```

Guardrail: canonical manager types must not import ACP/prototype/MCP workspace response types.

- [ ] **Step 5: Add copy helper tests**

Create `workspace-manager-copy.ts` with labels:

- `Workspace`
- `Research Workspace`
- `Project Workspace`
- `Host-local root`
- `Sandbox-managed root`
- `MCP trusted root binding`
- `MCP tool scope`
- `agent execution workspace`

Add tests that reject `Workspace Playground` and prevent using `Shared Workspace` as a canonical label.

- [ ] **Step 6: Run focused frontend tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/services/__tests__/tldw-api-client.workspace-api.test.ts apps/packages/ui/src/components/Option/Workspaces/__tests__/workspace-manager-models.test.ts
```

Expected: PASS.

- [ ] **Step 7: Commit Task 2**

```bash
git add apps/packages/ui/src/services/tldw/domains/workspace-api.ts apps/packages/ui/src/services/__tests__/tldw-api-client.workspace-api.test.ts apps/packages/ui/src/components/Option/Workspaces/workspace-manager-models.ts apps/packages/ui/src/components/Option/Workspaces/workspace-manager-copy.ts apps/packages/ui/src/components/Option/Workspaces/__tests__/workspace-manager-models.test.ts
git commit -m "feat: add canonical workspace api client models"
```

### Task 3: Durable Sandbox Workspace-Volume Contract

**Files:**
- Modify: `tldw_Server_API/app/core/Sandbox/models.py`
- Modify: `tldw_Server_API/app/core/Sandbox/store.py`
- Create: `tldw_Server_API/app/core/Sandbox/workspace_volumes.py`
- Modify: `tldw_Server_API/app/core/Workspaces/root_binding_service.py`
- Test: `tldw_Server_API/tests/sandbox/test_workspace_volumes.py`
- Test: `tldw_Server_API/tests/Workspaces/test_workspace_root_binding_service.py`

- [ ] **Step 1: Write failing Sandbox volume service tests**

Add tests for:

- Create workspace-bound volume returns `provisioning` or `ready`
- Same idempotency key and same request returns same volume
- Same idempotency key and different request raises conflict
- Validate volume rejects wrong `workspace_id` or `user_id`
- Resolve volume maps unavailable runtimes to `not_configured` or `unavailable`
- Diagnostics are bounded and redacted

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/sandbox/test_workspace_volumes.py -v
```

Expected: FAIL because `workspace_volumes.py` does not exist.

- [ ] **Step 3: Add durable volume dataclasses**

In `Sandbox/models.py`, add:

```python
class WorkspaceVolumeState(str, Enum):
    provisioning = "provisioning"
    ready = "ready"
    not_configured = "not_configured"
    unavailable = "unavailable"
    failed = "failed"
    cleanup_pending = "cleanup_pending"


@dataclass
class WorkspaceVolume:
    id: str
    workspace_id: str
    user_id: str
    state: WorkspaceVolumeState
    runtime: RuntimeType | None
    display_name: str | None = None
    mount_path: str | None = None
    diagnostics: dict[str, str] = field(default_factory=dict)
```

- [ ] **Step 4: Add store persistence methods**

Extend `SandboxStore` and its SQLite implementation in `store.py` with:

- `put_workspace_volume`
- `get_workspace_volume`
- `find_workspace_volume_by_idempotency`
- `update_workspace_volume_state`
- `list_workspace_volumes_for_workspace`

Use a `workspace_volumes` table owned by Sandbox store. Persist only redacted mount hints and bounded diagnostics.

- [ ] **Step 5: Add service module**

Create `workspace_volumes.py`:

```python
class SandboxWorkspaceVolumeService:
    def provision_workspace_volume(...): ...
    def validate_workspace_volume(...): ...
    def resolve_workspace_volume_mount(...): ...
```

Default V1 behavior can be conservative:

- If no runtime can support durable volumes, create or report `not_configured`.
- Do not fabricate host-local paths.
- Return `ready` only when a usable mount path exists.

- [ ] **Step 6: Wire resolver into root binding tests**

Update `root_binding_service.py` supported states to include `provisioning` and `cleanup_pending`. Ensure `strict_sandbox_validation=True` rejects `not_configured`, `unavailable`, `failed`, and `cleanup_pending`.

- [ ] **Step 7: Run Sandbox and root binding tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/sandbox/test_workspace_volumes.py tldw_Server_API/tests/Workspaces/test_workspace_root_binding_service.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit Task 3**

```bash
git add tldw_Server_API/app/core/Sandbox/models.py tldw_Server_API/app/core/Sandbox/store.py tldw_Server_API/app/core/Sandbox/workspace_volumes.py tldw_Server_API/app/core/Workspaces/root_binding_service.py tldw_Server_API/tests/sandbox/test_workspace_volumes.py tldw_Server_API/tests/Workspaces/test_workspace_root_binding_service.py
git commit -m "feat: add sandbox workspace volume contract"
```

### Task 4: Workspace-Owned Sandbox Root Provision-And-Attach Command

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Create: `tldw_Server_API/app/core/Workspaces/operations.py`
- Create: `tldw_Server_API/app/core/Workspaces/sandbox_root_provisioning.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/workspace_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/workspaces.py`
- Test: `tldw_Server_API/tests/Workspaces/test_workspace_operations.py`
- Test: `tldw_Server_API/tests/Workspaces/test_workspace_sandbox_root_provisioning.py`
- Test: `tldw_Server_API/tests/Workspaces/test_workspaces_api.py`

- [ ] **Step 1: Write failing operation DB tests**

Cover:

- Insert operation idempotency record
- Same key and same fingerprint returns existing record
- Same key and different fingerprint raises `ConflictError`
- Expired record cleanup does not delete attached root
- Diagnostics are bounded and do not include raw paths/secrets

- [ ] **Step 2: Add `workspace_operations` persistence**

In `ChaChaNotes_DB.py`, add a table with:

- `id`
- `workspace_id`
- `user_id`
- `command`
- `idempotency_key`
- `request_fingerprint`
- `linked_idempotency_key`
- `status`
- `result_ref_json`
- `diagnostics_json`
- `created_at`
- `updated_at`
- `expires_at`

Add DB methods:

- `create_workspace_operation`
- `get_workspace_operation`
- `get_workspace_operation_by_idempotency`
- `update_workspace_operation`
- `cleanup_expired_workspace_operations`
- `list_active_workspace_operations`

- [ ] **Step 3: Add operation helper module**

Create `operations.py`:

```python
def fingerprint_workspace_command(payload: Mapping[str, Any]) -> str:
    ...

def redact_operation_diagnostics(value: Mapping[str, Any]) -> dict[str, Any]:
    ...

def operation_poll_href(workspace_id: str, operation_id: str) -> str:
    ...
```

- [ ] **Step 4: Write failing API tests**

Add tests for:

- Missing `Idempotency-Key` returns 400
- Active provisioning returns 202
- Already attached equivalent root returns 200
- Same idempotency key retry returns same operation
- Different request same key returns 409
- `GET /api/v1/workspaces/{workspace_id}/operations/{operation_id}` returns operation status
- `/context` includes active operations

- [ ] **Step 5: Add provision-and-attach service**

Create `sandbox_root_provisioning.py`:

```python
def provision_and_attach_sandbox_root(
    *,
    db: CharactersRAGDB,
    sandbox_volume_service: SandboxWorkspaceVolumeService,
    workspace_id: str,
    user_id: str,
    request: WorkspaceSandboxRootProvisionRequest,
    idempotency_key: str,
) -> WorkspaceSandboxRootProvisionResult:
    ...
```

Rules:

- Workspace command owns idempotency and operation records.
- Sandbox service owns durable volume creation.
- Attach root only through `attach_primary_workspace_root`.
- Set/preserve `workspace_profile: project`.
- Return 202 while volume is provisioning.
- Return 200 for already-attached equivalent root or synchronous ready completion.

- [ ] **Step 6: Add schemas and endpoints**

In `workspace_schemas.py`, add `WorkspaceSandboxRootProvisionRequest` and `WorkspaceSandboxRootProvisionResponse`.

In `workspaces.py`, add:

- `POST /{workspace_id}/roots/primary/sandbox-volume`
- `GET /{workspace_id}/operations/{operation_id}`

Keep errors inside existing `(ConflictError, InputError, CharactersRAGDBError)` mapping where possible. Map service errors to structured HTTP details.

- [ ] **Step 7: Run backend tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_operations.py tldw_Server_API/tests/Workspaces/test_workspace_sandbox_root_provisioning.py tldw_Server_API/tests/Workspaces/test_workspaces_api.py -v
```

Expected: PASS.

- [ ] **Step 8: Run Bandit on touched backend paths**

Run:

```bash
source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Workspaces tldw_Server_API/app/core/Sandbox tldw_Server_API/app/api/v1/endpoints/workspaces.py tldw_Server_API/app/api/v1/schemas/workspace_schemas.py -f json -o /tmp/bandit_workspaces_project_root.json
```

Expected: no new findings in touched code.

- [ ] **Step 9: Commit Task 4**

```bash
git add tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/app/core/Workspaces/operations.py tldw_Server_API/app/core/Workspaces/sandbox_root_provisioning.py tldw_Server_API/app/api/v1/schemas/workspace_schemas.py tldw_Server_API/app/api/v1/endpoints/workspaces.py tldw_Server_API/tests/Workspaces/test_workspace_operations.py tldw_Server_API/tests/Workspaces/test_workspace_sandbox_root_provisioning.py tldw_Server_API/tests/Workspaces/test_workspaces_api.py
git commit -m "feat: add workspace sandbox root provisioning"
```

Do not commit `/tmp/bandit_workspaces_project_root.json`; record its path/output in the Backlog task instead.

### Task 5: Canonical `/workspaces` Manager CRUD

**Files:**
- Modify: `apps/packages/ui/src/routes/route-paths.ts`
- Modify: `apps/packages/ui/src/routes/route-metadata.ts`
- Modify: `apps/packages/ui/src/routes/route-registry.tsx`
- Create: `apps/packages/ui/src/routes/option-workspaces.tsx`
- Create: `apps/packages/ui/src/components/Option/Workspaces/WorkspacesManagerPage.tsx`
- Create: `apps/packages/ui/src/components/Option/Workspaces/WorkspaceList.tsx`
- Create: `apps/packages/ui/src/components/Option/Workspaces/WorkspaceCreateDialog.tsx`
- Create: `apps/packages/ui/src/components/Option/Workspaces/WorkspaceMetadataDialog.tsx`
- Test: `apps/packages/ui/src/routes/__tests__/option-workspaces.route.test.tsx`
- Test: `apps/packages/ui/src/routes/__tests__/route-metadata.coverage.test.ts`
- Test: `apps/packages/ui/src/components/Option/Workspaces/__tests__/WorkspacesManagerPage.test.tsx`

- [ ] **Step 1: Write failing route registry tests**

Add tests asserting:

- `WORKSPACES_PATH === "/workspaces"`
- route metadata exists for `/workspaces`
- no aliases or redirects
- route group is `workspace`
- route registry renders `WorkspacesManagerPage`
- `AUDITED_ROOT_ROUTE_PATHS` includes `/workspaces`

- [ ] **Step 2: Run route tests and verify failure**

Run:

```bash
bunx vitest run apps/packages/ui/src/routes/__tests__/option-workspaces.route.test.tsx apps/packages/ui/src/routes/__tests__/route-metadata.coverage.test.ts
```

Expected: FAIL because route files do not exist or metadata is missing.

- [ ] **Step 3: Add route constants and metadata**

Add `WORKSPACES_PATH = "/workspaces"` to `route-paths.ts`.

Add route metadata:

```ts
defineRoute({
  path: "/workspaces",
  label: "Workspaces",
  group: "workspace",
  surface: "advanced_self_hosted",
  availability: webAndExtension,
  smoke: "manual",
  nav: "secondary",
  requiresBackend: true,
  rationale: "Canonical Workspace manager for research and project workspaces."
})
```

Do not add aliases or redirects.

- [ ] **Step 4: Add route shell**

Create `option-workspaces.tsx`:

```tsx
import OptionLayout from "~/components/Layouts/Layout"
import { WorkspacesManagerPage } from "@/components/Option/Workspaces/WorkspacesManagerPage"

const OptionWorkspaces = () => (
  <OptionLayout>
    <div className="flex h-full min-h-0 w-full flex-1 overflow-hidden">
      <WorkspacesManagerPage />
    </div>
  </OptionLayout>
)

export default OptionWorkspaces
```

- [ ] **Step 5: Write failing manager component tests**

Cover:

- loading
- backend unavailable
- empty state
- list with Research and Project Workspaces
- search/filter by profile/archive/attention
- create Research Workspace
- create Project shell without root setup
- edit metadata
- archive/unarchive
- no visible hard delete action
- open in Research Workspace

- [ ] **Step 6: Implement manager page and dialogs**

Use existing design system primitives. Keep layout dense and operational:

- page toolbar with search, profile filter, archived toggle
- primary action menu: Research Workspace, Project Workspace
- list/table rows with name, profile, root summary, source summary, updated time, attention
- details side panel or inline expansion for actions

Do not add a trust banner. Do not call MCP Shared Workspaces canonical Workspaces.

- [ ] **Step 7: Run frontend manager tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/routes/__tests__/option-workspaces.route.test.tsx apps/packages/ui/src/routes/__tests__/route-metadata.coverage.test.ts apps/packages/ui/src/components/Option/Workspaces/__tests__/WorkspacesManagerPage.test.tsx
```

Expected: PASS.

- [ ] **Step 8: Commit Task 5**

```bash
git add apps/packages/ui/src/routes/route-paths.ts apps/packages/ui/src/routes/route-metadata.ts apps/packages/ui/src/routes/route-registry.tsx apps/packages/ui/src/routes/option-workspaces.tsx apps/packages/ui/src/components/Option/Workspaces/WorkspacesManagerPage.tsx apps/packages/ui/src/components/Option/Workspaces/WorkspaceList.tsx apps/packages/ui/src/components/Option/Workspaces/WorkspaceCreateDialog.tsx apps/packages/ui/src/components/Option/Workspaces/WorkspaceMetadataDialog.tsx apps/packages/ui/src/routes/__tests__/option-workspaces.route.test.tsx apps/packages/ui/src/routes/__tests__/route-metadata.coverage.test.ts apps/packages/ui/src/components/Option/Workspaces/__tests__/WorkspacesManagerPage.test.tsx
git commit -m "feat: add canonical workspaces manager route"
```

### Task 6: Project Upgrade And Root Panel

**Files:**
- Create: `apps/packages/ui/src/components/Option/Workspaces/WorkspaceProjectRootPanel.tsx`
- Modify: `apps/packages/ui/src/components/Option/Workspaces/WorkspacesManagerPage.tsx`
- Modify: `apps/packages/ui/src/components/Option/Workspaces/workspace-manager-models.ts`
- Modify: `apps/packages/ui/src/services/tldw/domains/workspace-api.ts`
- Test: `apps/packages/ui/src/components/Option/Workspaces/__tests__/WorkspaceProjectRootPanel.test.tsx`

- [ ] **Step 1: Write failing root panel tests**

Cover:

- Research Workspace shows "Upgrade to Project Workspace"
- Project without root shows root type selection
- Host-local attach calls existing `PUT /roots/primary`
- Sandbox-managed root calls `POST /roots/primary/sandbox-volume` with `Idempotency-Key`
- 202 response shows provisioning and starts polling
- refresh can recover active operation from context
- inventory scan action is disabled until `file_inventory.available === true`
- raw host-local path is not shown in passive display

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/Workspaces/__tests__/WorkspaceProjectRootPanel.test.tsx
```

Expected: FAIL because panel does not exist.

- [ ] **Step 3: Implement root panel**

Implement:

- profile upgrade action via `patchWorkspace({ workspace_profile: "project" })`
- host-local attach form with `expected_workspace_version`
- frontend client methods for `GET /operations/{operation_id}` and
  `POST /roots/primary/sandbox-volume`, backed by Task 4 endpoint tests
- sandbox-managed form with runtime/display name and idempotency key generation
- operation polling with exponential backoff capped at a few seconds
- retry/remediation copy for `not_configured`, `unavailable`, `failed`, and `cleanup_pending`

- [ ] **Step 4: Add inventory action gating**

Only enable scan when:

```ts
item.projectRoot?.fileInventory?.available === true
```

For sandbox roots without a ready mount, show:

```text
File inventory is unavailable until the sandbox-managed root is mounted.
```

- [ ] **Step 5: Run root panel and API client tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/Workspaces/__tests__/WorkspaceProjectRootPanel.test.tsx apps/packages/ui/src/services/__tests__/tldw-api-client.workspace-api.test.ts
```

Expected: PASS.

- [ ] **Step 6: Commit Task 6**

```bash
git add apps/packages/ui/src/components/Option/Workspaces/WorkspaceProjectRootPanel.tsx apps/packages/ui/src/components/Option/Workspaces/WorkspacesManagerPage.tsx apps/packages/ui/src/components/Option/Workspaces/workspace-manager-models.ts apps/packages/ui/src/services/tldw/domains/workspace-api.ts apps/packages/ui/src/components/Option/Workspaces/__tests__/WorkspaceProjectRootPanel.test.tsx apps/packages/ui/src/services/__tests__/tldw-api-client.workspace-api.test.ts
git commit -m "feat: add project workspace root panel"
```

### Task 7: Local Research Workspace Reconciliation

**Files:**
- Create: `apps/packages/ui/src/components/Option/Workspaces/workspace-local-reconciliation.ts`
- Create: `apps/packages/ui/src/components/Option/Workspaces/WorkspaceReconciliationPanel.tsx`
- Modify: `apps/packages/ui/src/components/Option/Workspaces/WorkspacesManagerPage.tsx`
- Modify: `apps/packages/ui/src/store/research-workspace-legacy-storage-inventory.ts`
- Test: `apps/packages/ui/src/components/Option/Workspaces/__tests__/workspace-local-reconciliation.test.ts`
- Test: `apps/packages/ui/src/components/Option/Workspaces/__tests__/WorkspacesManagerPage.test.tsx`

- [ ] **Step 1: Write failing reconciliation helper tests**

Cover dry-run states:

- local-only
- server row exists
- name conflict
- possible duplicate
- unsupported local payload
- ready to create metadata

Test marker shape:

```ts
type WorkspaceReconciliationMarkerV1 = {
  schemaVersion: 1
  serverWorkspaceId: string
  serverName: string
  serverProfile: "research" | "project"
  linkedAt: string
  status: "linked" | "metadata_promoted" | "conflict"
  conflictState?: string
}
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/Workspaces/__tests__/workspace-local-reconciliation.test.ts
```

Expected: FAIL because helper does not exist.

- [ ] **Step 3: Implement local detection and marker helpers**

Use existing local Research Workspace storage inventory helpers. Do not rewrite source, note, artifact, chat, or IndexedDB payloads.

- [ ] **Step 4: Implement reconciliation panel**

Panel behavior:

- separate "Local only" entries from server-backed Workspaces
- show dry-run status and conflict reason
- allow "Create server metadata" and "Link to existing Workspace"
- preserve tombstones and undo behavior
- write marker only after confirmed metadata promotion/link

- [ ] **Step 5: Run reconciliation tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/Workspaces/__tests__/workspace-local-reconciliation.test.ts apps/packages/ui/src/components/Option/Workspaces/__tests__/WorkspacesManagerPage.test.tsx apps/packages/ui/src/store/__tests__/research-workspace-legacy-storage-inventory.test.ts
```

Expected: PASS.

- [ ] **Step 6: Commit Task 7**

```bash
git add apps/packages/ui/src/components/Option/Workspaces/workspace-local-reconciliation.ts apps/packages/ui/src/components/Option/Workspaces/WorkspaceReconciliationPanel.tsx apps/packages/ui/src/components/Option/Workspaces/WorkspacesManagerPage.tsx apps/packages/ui/src/store/research-workspace-legacy-storage-inventory.ts apps/packages/ui/src/components/Option/Workspaces/__tests__/workspace-local-reconciliation.test.ts apps/packages/ui/src/components/Option/Workspaces/__tests__/WorkspacesManagerPage.test.tsx
git commit -m "feat: add workspace reconciliation panel"
```

### Task 8: Cross-Surface Links And Live UAT

**Files:**
- Modify: `apps/packages/ui/src/routes/route-paths.ts`
- Modify: `apps/packages/ui/src/components/Option/Workspaces/WorkspacesManagerPage.tsx`
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/WorkspaceHeader.tsx`
- Modify: `apps/packages/ui/src/design-system/__tests__/product-state-guard.mcp-acp-workspace-baseline.test.ts`
- Create: `apps/tldw-frontend/e2e/workflows/workspaces-manager.spec.ts`
- Create: `Docs/Validation/workspaces-manager-uat-matrix.md`

- [ ] **Step 1: Write route/link tests**

Cover:

- manager opens Research Workspace through `RESEARCH_WORKSPACE_PATH`
- MCP link text says `MCP trusted root binding` or `MCP tool scope`
- ACP link text says `agent execution workspace` when referring to ACP rows
- no `Workspace Playground`
- no `/workspace-playground`
- no redirects or aliases for `/workspaces`

- [ ] **Step 2: Implement cross-surface links**

Add deep links from manager rows/panels to:

- Research Workspace
- MCP Hub trusted-root or tool-scope views
- ACP diagnostics
- Sandbox workspace diagnostics

Use route builders. Do not embed raw absolute host-local paths in hrefs or labels.

- [ ] **Step 3: Add live UAT matrix**

Create `Docs/Validation/workspaces-manager-uat-matrix.md` with rows:

- create Research Workspace
- create Project Workspace shell
- attach host-local root
- provision sandbox-managed root when Sandbox is ready
- sandbox unavailable recovery
- archive/unarchive
- local-only reconciliation metadata flow
- Research Workspace deep link
- MCP trusted-root/tool-scope deep link
- ACP diagnostics link
- Sandbox diagnostics link

Each row must include: route, setup, expected backend call, expected UI state, and result.

- [ ] **Step 4: Add Playwright/CDP smoke**

Create `workspaces-manager.spec.ts`. Use the existing e2e auth/test utilities and keep it resilient:

- skip sandbox-managed ready-path assertions when runtime is not configured
- always validate the unavailable recovery state
- never rely on `/workspace-playground`

- [ ] **Step 5: Run focused frontend tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/Workspaces/__tests__/WorkspacesManagerPage.test.tsx apps/packages/ui/src/design-system/__tests__/product-state-guard.mcp-acp-workspace-baseline.test.ts apps/packages/ui/src/routes/__tests__/route-metadata.coverage.test.ts
```

Expected: PASS.

- [ ] **Step 6: Run live backend/WebUI validation**

Start the real backend and WebUI in separate terminals or managed sessions. Then run:

```bash
bunx playwright test apps/tldw-frontend/e2e/workflows/workspaces-manager.spec.ts --reporter=line
```

Expected: PASS or documented sandbox-ready skip. Update `Docs/Validation/workspaces-manager-uat-matrix.md` with results.

- [ ] **Step 7: Run Bandit on touched backend paths if backend changed in this task**

If Task 8 only changes frontend/docs, record Bandit as not applicable. If backend files changed, run:

```bash
source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/endpoints/workspaces.py tldw_Server_API/app/core/Workspaces -f json -o /tmp/bandit_workspaces_links.json
```

- [ ] **Step 8: Commit Task 8**

```bash
git add apps/packages/ui/src/routes/route-paths.ts apps/packages/ui/src/components/Option/Workspaces/WorkspacesManagerPage.tsx apps/packages/ui/src/components/Option/ResearchWorkspace/WorkspaceHeader.tsx apps/packages/ui/src/design-system/__tests__/product-state-guard.mcp-acp-workspace-baseline.test.ts apps/tldw-frontend/e2e/workflows/workspaces-manager.spec.ts Docs/Validation/workspaces-manager-uat-matrix.md
git commit -m "test: validate canonical workspaces manager flow"
```

### Task 9: Final Workstream Verification

**Files:**
- Modify: Backlog tasks for each implementation slice.
- No production file changes unless verification finds issues.

- [ ] **Step 1: Run backend focused test suite**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces tldw_Server_API/tests/sandbox/test_workspace_volumes.py -v
```

Expected: PASS.

- [ ] **Step 2: Run frontend focused test suite**

Run:

```bash
bunx vitest run apps/packages/ui/src/services/__tests__/tldw-api-client.workspace-api.test.ts apps/packages/ui/src/components/Option/Workspaces apps/packages/ui/src/routes/__tests__/option-workspaces.route.test.tsx apps/packages/ui/src/routes/__tests__/route-metadata.coverage.test.ts
```

Expected: PASS.

- [ ] **Step 3: Run route/product guard**

Run:

```bash
bunx vitest run apps/packages/ui/src/design-system/__tests__/product-state-guard.mcp-acp-workspace-baseline.test.ts
```

Expected: PASS and no `/workspace-playground` references.

- [ ] **Step 4: Run live UAT**

Run the Playwright/CDP smoke from Task 8 against a live backend and WebUI. Update `Docs/Validation/workspaces-manager-uat-matrix.md`.

- [ ] **Step 5: Run Bandit for backend touched scope**

Run:

```bash
source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Workspaces tldw_Server_API/app/core/Sandbox tldw_Server_API/app/api/v1/endpoints/workspaces.py tldw_Server_API/app/api/v1/schemas/workspace_schemas.py -f json -o /tmp/bandit_workspaces_manager_final.json
```

Expected: no new findings in touched code. Record result in Backlog.

- [ ] **Step 6: Final commit or PR**

If verification fixes changed files, commit them. Otherwise create the PR against `dev` after rebasing on latest `origin/dev`.

## Parallelization Notes

- Task 1 and Task 2 can be done by separate workers after this plan is accepted.
- Task 3 can run in parallel with Task 2, but Task 4 must wait for Task 3.
- Task 5 can start after Task 2 using mocked backend responses, but merge it after Task 1 so response fields match.
- Task 6 depends on Tasks 2 and 4.
- Task 7 depends on Task 5.
- Task 8 should be last and must use a live backend plus WebUI.

## Risks And Controls

- **Route confusion:** Guard with route metadata tests and product-state guard. No aliases or redirects.
- **ACP/MCP naming drift:** Keep canonical types separate from ACP/prototype/MCP response types. Test copy helpers.
- **Idempotency data leakage:** Store fingerprints and redacted diagnostics only. Test raw path/env var redaction.
- **Sandbox unavailable on dev machines:** Support `not_configured` and unavailable states as first-class paths. Live smoke may skip ready-runtime assertions but must validate recovery.
- **Manager exposing delete too early:** V1 UI only exposes archive/unarchive. API raw delete client support can exist for parity but must not be rendered as a manager action.
- **Large component drift:** Keep manager components split by responsibility. Do not put cards inside cards or add persistent trust banners.

## Handoff Checklist

- Rebase the implementation worktree onto latest `origin/dev` before Task 1 begins.
- Create or update one Backlog task per implementation task before editing code.
- Use TDD for each task: failing test, minimal implementation, passing test.
- Record Bandit for backend tasks.
- Use Playwright/CDP, not computer control, for live WebUI validation.
- Keep commits task-sized and reviewable.
