# Workspace Assistant Defaults Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement #1911 V1 Workspace Assistant Defaults so Workspace APIs store Persona-backed defaults, expose permission-safe effective defaults, and Chat Workspace applies them only to new Workspace-scoped chats with no explicit assistant.

**Architecture:** Store a reference-backed `assistant_defaults_json` payload on Workspace records and normalize it through Pydantic/API helpers. Runtime consumers read `effective_assistant_default`, which resolves Persona visibility for the current user without copying Persona profile content. Chat Workspace receives the effective default as startup metadata and writes concrete conversation `assistant_kind`, `assistant_id`, and `persona_memory_mode` through the existing chat session path.

**Tech Stack:** FastAPI, Pydantic, SQLite/Postgres ChaChaNotes DB migrations, pytest, React/TypeScript, Zustand, Vitest.

---

## Scope

In scope:

- V1 `assistant_defaults` schema/API with `assistant_kind: "persona"`, `assistant_id`, and `persona_memory_mode`.
- Reference-only storage with no Persona snapshots.
- `effective_assistant_default` response shape with available/unavailable degraded states.
- Workspace settings UI for selecting/clearing a Persona default and confirming `read_write`.
- Chat Workspace startup application for new Workspace-scoped chats without explicit assistant selection.

Out of scope:

- Buddy runtime, Buddy animation, Persona visual packs, VN, scheduled work, broad personalization memory, tool marketplace/admin, and non-Workspace global `/chat`.
- Voice/style/tool-policy implementation beyond accepting absent or `null` deferred fields.
- Research Workspace, Prompt Studio, writing, audio overview, or agent/tool adoption beyond preserving the shared contract for future work.

## File Map

Backend:

- Modify `tldw_Server_API/app/api/v1/schemas/workspace_schemas.py`
  - Add `WorkspaceAssistantDefaults`, `WorkspaceEffectiveAssistantDefault`, status/source/degraded literals, and `read_write` acknowledgement field on patch requests.
- Modify `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
  - Add schema v49 migration with `assistant_defaults_json TEXT`.
  - Add JSON normalization helpers and include the column in create/update/get/list paths.
- Modify `tldw_Server_API/app/api/v1/endpoints/workspaces.py`
  - Convert stored JSON into response schemas.
  - Validate Persona references for update.
  - Build permission-filtered `effective_assistant_default`.
- Test `tldw_Server_API/tests/ChaChaNotesDB/test_workspace_assistant_defaults_db.py`
  - Storage/migration and no-snapshot tests.
- Test `tldw_Server_API/tests/Workspaces/test_workspace_assistant_defaults_api.py`
  - API schema, optimistic locking, reference validation, read_write confirmation, and degraded states.
- Update `tldw_Server_API/tests/Workspaces/test_workspaces_api.py`
  - Existing workspace response fixtures need the new nullable fields when asserting full payloads.

Frontend:

- Modify `apps/packages/ui/src/types/workspace.ts`
  - Add `WorkspaceAssistantDefaults`, `EffectiveWorkspaceAssistantDefault`, and extend workspace types.
- Modify `apps/packages/ui/src/services/tldw/TldwApiClient.ts`
  - Normalize Workspace assistant defaults in workspace responses and patch payloads.
- Add or modify `apps/packages/ui/src/services/tldw/domains/workspaces.ts` if the current client has a workspace domain helper in the implementation branch; otherwise keep normalization in `TldwApiClient.ts`.
- Modify `apps/packages/ui/src/store/workspace.ts`
  - Keep active Workspace default metadata available to UI as server state; persist only stored references when necessary, never resolved labels.
- Modify `apps/packages/ui/src/components/Option/ResearchWorkspace/WorkspaceHeader.tsx`
  - Add a Workspace settings menu item/modal for the default Persona.
- Modify `apps/packages/ui/src/components/Option/ChatWorkspace/ChatWorkspacePage.tsx`
  - Read effective default from the active Workspace state/API result and pass it to the panel.
- Modify `apps/packages/ui/src/components/Option/ChatWorkspace/WorkspaceChatPanel.tsx`
  - Apply effective default only before first send when no explicit assistant is selected.
- Modify `apps/packages/ui/src/components/Option/ChatWorkspace/InspectorRail.tsx`
  - Label inherited vs explicit Persona and unavailable defaults.
- Test existing Vitest files under:
  - `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/WorkspaceHeader.test.tsx`
  - `apps/packages/ui/src/components/Option/ChatWorkspace/__tests__/WorkspaceChatPanel.test.tsx`
  - `apps/packages/ui/src/components/Option/ChatWorkspace/__tests__/InspectorRail.test.tsx`
  - `apps/packages/ui/src/services/__tests__/tldw-api-client.assistant-identity.test.ts`

## Implementation Tasks

### Task 1: Backend Schema And Storage Contract

**Files:**

- Modify: `tldw_Server_API/app/api/v1/schemas/workspace_schemas.py`
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Create: `tldw_Server_API/tests/ChaChaNotesDB/test_workspace_assistant_defaults_db.py`
- Update: `tldw_Server_API/tests/ChaChaNotesDB/test_workspace_sub_resources_db.py`

- [ ] **Step 1: Write failing schema tests**

Add tests for:

```python
def test_workspace_assistant_defaults_accepts_persona_read_only() -> None:
    payload = WorkspaceAssistantDefaults(
        assistant_kind="persona",
        assistant_id="persona-1",
        persona_memory_mode="read_only",
    )
    assert payload.model_dump(exclude_none=True) == {
        "assistant_kind": "persona",
        "assistant_id": "persona-1",
        "persona_memory_mode": "read_only",
    }


def test_workspace_assistant_defaults_rejects_deferred_fields() -> None:
    with pytest.raises(ValidationError, match="voice must be null"):
        WorkspaceAssistantDefaults(
            assistant_kind="persona",
            assistant_id="persona-1",
            persona_memory_mode="read_only",
            voice={"provider": "openai"},
        )
```

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_workspace_assistant_defaults_db.py -q`

Expected: FAIL because schemas/storage do not exist yet.

- [ ] **Step 2: Add Pydantic models**

In `workspace_schemas.py`, add:

```python
WorkspaceAssistantKind = Literal["persona"]
WorkspacePersonaMemoryMode = Literal["read_only", "read_write"]
WorkspaceEffectiveAssistantDefaultStatus = Literal["available", "unavailable", "none"]
WorkspaceEffectiveAssistantDefaultSource = Literal["workspace", "none"]
WorkspaceAssistantDefaultDegradedReason = Literal[
    "persona_deleted",
    "persona_unavailable",
    "persona_feature_disabled",
    "permission_denied",
    "invalid_default",
    "unsupported_assistant_kind",
]


class WorkspaceAssistantDefaults(BaseModel):
    assistant_kind: WorkspaceAssistantKind
    assistant_id: str = Field(..., min_length=1, max_length=128)
    persona_memory_mode: WorkspacePersonaMemoryMode = "read_only"
    voice: None = None
    style: None = None
    tool_policy_profile_id: None = None


class WorkspaceEffectiveAssistantDefault(BaseModel):
    status: WorkspaceEffectiveAssistantDefaultStatus
    source: WorkspaceEffectiveAssistantDefaultSource
    assistant_kind: WorkspaceAssistantKind | None = None
    assistant_id: str | None = None
    label: str | None = None
    persona_memory_mode: WorkspacePersonaMemoryMode | None = None
    degraded_reason: WorkspaceAssistantDefaultDegradedReason | None = None
```

Extend:

```python
class WorkspacePatchRequest(BaseModel):
    assistant_defaults: WorkspaceAssistantDefaults | None = None
    confirm_read_write_assistant_default: StrictBool | None = None


class WorkspaceResponse(BaseModel):
    assistant_defaults: WorkspaceAssistantDefaults | None = None
    effective_assistant_default: WorkspaceEffectiveAssistantDefault
```

- [ ] **Step 3: Write failing DB tests**

Add tests proving:

- new DBs have `assistant_defaults_json`.
- updating Workspace stores/retrieves only JSON references.
- changing default increments `version`.
- invalid JSON in old/drifted rows returns `None` or an invalid effective default, not a crash.

Example assertion:

```python
ws = db.update_workspace(
    "ws-1",
    {
        "assistant_defaults_json": {
            "assistant_kind": "persona",
            "assistant_id": "persona-1",
            "persona_memory_mode": "read_only",
        }
    },
    expected_version=1,
)
assert ws["assistant_defaults_json"] == {
    "assistant_kind": "persona",
    "assistant_id": "persona-1",
    "persona_memory_mode": "read_only",
}
assert "persona_name" not in json.dumps(ws["assistant_defaults_json"])
```

- [ ] **Step 4: Add v49 migration and storage helpers**

In `ChaChaNotes_DB.py`:

- bump `_CURRENT_SCHEMA_VERSION` from `48` to `49`.
- add `_MIGRATION_SQL_V48_TO_V49` and `_MIGRATION_SQL_V48_TO_V49_POSTGRES`.
- add `assistant_defaults_json TEXT` / `TEXT NULL` to initial Workspace table creation paths.
- add migration dispatch for SQLite and Postgres.
- include `assistant_defaults_json` in `update_workspace`.

Use helpers like:

```python
@classmethod
def _serialize_workspace_assistant_defaults_json(cls, value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        json.loads(stripped)
        return stripped
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


@classmethod
def _load_workspace_assistant_defaults_json(cls, value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = json.loads(value)
        return dict(parsed) if isinstance(parsed, Mapping) else None
    except (json.JSONDecodeError, ValueError):
        return None
```

Do not store Persona names, prompts, avatars, tool scopes, or policy snapshots.

- [ ] **Step 5: Run backend storage tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/ChaChaNotesDB/test_workspace_assistant_defaults_db.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_workspace_sub_resources_db.py \
  -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/api/v1/schemas/workspace_schemas.py \
  tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_workspace_assistant_defaults_db.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_workspace_sub_resources_db.py
git commit -m "feat: add workspace assistant default storage"
```

### Task 2: Workspace API Validation And Effective Defaults

**Files:**

- Modify: `tldw_Server_API/app/api/v1/endpoints/workspaces.py`
- Create: `tldw_Server_API/tests/Workspaces/test_workspace_assistant_defaults_api.py`
- Update: `tldw_Server_API/tests/Workspaces/test_workspaces_api.py`

- [ ] **Step 1: Write failing API tests**

Cover:

- `PATCH /api/v1/workspaces/{id}` accepts a valid Persona default and returns stored/effective defaults.
- `read_write` requires `confirm_read_write_assistant_default: true`.
- unsupported assistant kind returns 422.
- non-existent/deleted Persona returns 404 or 422 on save.
- inaccessible/deleted stored Persona returns `effective_assistant_default.status == "unavailable"` and does not leak a label.
- listing Workspaces includes the effective shape.

Example:

```python
def test_patch_workspace_rejects_read_write_without_confirmation(client, db):
    persona = db.create_persona_profile({"name": "Researcher", "user_id": "1"})
    db.upsert_workspace("ws-1", "Workspace")

    response = client.patch(
        "/api/v1/workspaces/ws-1",
        json={
            "version": 1,
            "assistant_defaults": {
                "assistant_kind": "persona",
                "assistant_id": persona["id"],
                "persona_memory_mode": "read_write",
            },
        },
    )

    assert response.status_code == 422
    assert "confirm_read_write" in response.text
```

- [ ] **Step 2: Add response conversion helpers**

In `workspaces.py`, add helpers:

```python
def _parse_workspace_assistant_defaults(raw: Any) -> WorkspaceAssistantDefaults | None:
    ...


def _effective_workspace_assistant_default(
    *,
    db: CharactersRAGDB,
    stored: WorkspaceAssistantDefaults | None,
    user_id: str,
) -> WorkspaceEffectiveAssistantDefault:
    if stored is None:
        return WorkspaceEffectiveAssistantDefault(status="none", source="none")
    if stored.assistant_kind != "persona":
        return WorkspaceEffectiveAssistantDefault(
            status="unavailable",
            source="workspace",
            degraded_reason="unsupported_assistant_kind",
        )
    profile = db.get_persona_profile(stored.assistant_id, user_id=user_id, include_deleted=False)
    if profile is None:
        return WorkspaceEffectiveAssistantDefault(
            status="unavailable",
            source="workspace",
            degraded_reason="permission_denied",
        )
    return WorkspaceEffectiveAssistantDefault(
        status="available",
        source="workspace",
        assistant_kind="persona",
        assistant_id=stored.assistant_id,
        label=str(profile.get("name") or stored.assistant_id),
        persona_memory_mode=stored.persona_memory_mode,
    )
```

Keep degraded labels redacted unless the implementation has a clear owner/admin repair permission.

- [ ] **Step 3: Validate update payloads before DB write**

In `patch_workspace`:

- normalize `assistant_defaults` into `assistant_defaults_json`.
- if default is `None`, clear the stored JSON.
- if `persona_memory_mode == "read_write"` and acknowledgement is not true, return 422.
- call `db.get_persona_profile(..., user_id=str(current_user.id), include_deleted=False)` before saving.
- reject voice/style/tool fields unless absent or `null` through Pydantic.

- [ ] **Step 4: Run API tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Workspaces/test_workspace_assistant_defaults_api.py \
  tldw_Server_API/tests/Workspaces/test_workspaces_api.py \
  -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/api/v1/endpoints/workspaces.py \
  tldw_Server_API/tests/Workspaces/test_workspace_assistant_defaults_api.py \
  tldw_Server_API/tests/Workspaces/test_workspaces_api.py
git commit -m "feat: expose workspace effective assistant defaults"
```

### Task 3: Frontend API Types And Store Mapping

**Files:**

- Modify: `apps/packages/ui/src/types/workspace.ts`
- Modify: `apps/packages/ui/src/services/tldw/TldwApiClient.ts`
- Modify: `apps/packages/ui/src/store/workspace.ts`
- Test: `apps/packages/ui/src/services/__tests__/tldw-api-client.assistant-identity.test.ts`

- [ ] **Step 1: Write failing client normalization tests**

Add tests that mock a Workspace response containing:

```ts
{
  id: "ws-1",
  name: "Literature",
  assistant_defaults: {
    assistant_kind: "persona",
    assistant_id: "persona-1",
    persona_memory_mode: "read_only",
    voice: null,
    style: null,
    tool_policy_profile_id: null
  },
  effective_assistant_default: {
    status: "available",
    source: "workspace",
    assistant_kind: "persona",
    assistant_id: "persona-1",
    label: "Literature Review Assistant",
    persona_memory_mode: "read_only",
    degraded_reason: null
  }
}
```

Assert the normalized Workspace preserves `assistantDefaults` and `effectiveAssistantDefault`.

Run:

```bash
bunx vitest run apps/packages/ui/src/services/__tests__/tldw-api-client.assistant-identity.test.ts
```

Expected: FAIL until types/client normalize the new fields.

- [ ] **Step 2: Add TypeScript types**

In `workspace.ts`, add:

```ts
export type WorkspaceAssistantKind = "persona"
export type WorkspacePersonaMemoryMode = "read_only" | "read_write"
export type EffectiveWorkspaceAssistantDefaultStatus =
  | "available"
  | "unavailable"
  | "none"

export interface WorkspaceAssistantDefaults {
  assistantKind: WorkspaceAssistantKind
  assistantId: string
  personaMemoryMode: WorkspacePersonaMemoryMode
  voice: null
  style: null
  toolPolicyProfileId: null
}

export interface EffectiveWorkspaceAssistantDefault {
  status: EffectiveWorkspaceAssistantDefaultStatus
  source: "workspace" | "none"
  assistantKind: WorkspaceAssistantKind | null
  assistantId: string | null
  label: string | null
  personaMemoryMode: WorkspacePersonaMemoryMode | null
  degradedReason:
    | "persona_deleted"
    | "persona_unavailable"
    | "persona_feature_disabled"
    | "permission_denied"
    | "invalid_default"
    | "unsupported_assistant_kind"
    | null
}
```

Extend active Workspace state with stored reference metadata and the latest server-fetched effective metadata. Do not persist `effectiveAssistantDefault.label` into saved Workspace entries or local snapshot bundles.

- [ ] **Step 3: Normalize API fields**

In `TldwApiClient.ts`, add snake_case to camelCase conversion for:

- `assistant_defaults` to `assistantDefaults`
- `effective_assistant_default` to `effectiveAssistantDefault`
- `persona_memory_mode` to `personaMemoryMode`
- `degraded_reason` to `degradedReason`

Also add patch request support that serializes camelCase store values back to snake_case.

- [ ] **Step 4: Store active Workspace default reference**

In `workspace.ts` store:

- add `assistantDefaults` and `effectiveAssistantDefault` to active in-memory Workspace state where server state is represented.
- keep local persisted data reference-only; omit `effectiveAssistantDefault.label` and other resolved Persona display fields from persisted snapshots.
- do not hydrate an effective default into a global assistant selection.

- [ ] **Step 5: Run frontend type/client tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/services/__tests__/tldw-api-client.assistant-identity.test.ts
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add apps/packages/ui/src/types/workspace.ts \
  apps/packages/ui/src/services/tldw/TldwApiClient.ts \
  apps/packages/ui/src/store/workspace.ts \
  apps/packages/ui/src/services/__tests__/tldw-api-client.assistant-identity.test.ts
git commit -m "feat: map workspace assistant defaults in web client"
```

### Task 4: Workspace Settings UI

**Files:**

- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/WorkspaceHeader.tsx`
- Test: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/WorkspaceHeader.test.tsx`

- [ ] **Step 1: Write failing UI tests**

Add tests for:

- settings menu opens a “Default assistant” modal.
- selecting a Persona default saves a patch payload with `assistant_defaults`.
- selecting `read_write` requires a visible confirmation checkbox before save is enabled.
- clearing default sends `assistant_defaults: null`.
- unavailable effective default displays a redacted degraded message.

Expected test selectors:

- `workspace-default-assistant-modal`
- `workspace-default-assistant-select`
- `workspace-default-assistant-memory-mode`
- `workspace-default-assistant-read-write-confirm`
- `workspace-default-assistant-clear`

- [ ] **Step 2: Implement modal using existing settings menu**

In `WorkspaceHeader.tsx`:

- add a `default-assistant` menu item near “Customize banner”.
- reuse existing Persona list API/client path if available; otherwise use the same service method used by `AssistantSelect`.
- render only Persona choices in V1.
- show current effective state:
  - available: label and memory mode.
  - unavailable: redacted status and degraded reason.
  - none: empty state.
- call Workspace patch API with current Workspace `version`.

Keep copy concise and do not introduce design-system backlog changes.

- [ ] **Step 3: Run WorkspaceHeader tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/WorkspaceHeader.test.tsx
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add apps/packages/ui/src/components/Option/ResearchWorkspace/WorkspaceHeader.tsx \
  apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/WorkspaceHeader.test.tsx
git commit -m "feat: add workspace default assistant settings"
```

### Task 5: Chat Workspace Startup Application

**Files:**

- Modify: `apps/packages/ui/src/components/Option/ChatWorkspace/ChatWorkspacePage.tsx`
- Modify: `apps/packages/ui/src/components/Option/ChatWorkspace/WorkspaceChatPanel.tsx`
- Modify: `apps/packages/ui/src/components/Option/ChatWorkspace/types.ts`
- Modify: `apps/packages/ui/src/components/Option/ChatWorkspace/InspectorRail.tsx`
- Test: `apps/packages/ui/src/components/Option/ChatWorkspace/__tests__/WorkspaceChatPanel.test.tsx`
- Test: `apps/packages/ui/src/components/Option/ChatWorkspace/__tests__/InspectorRail.test.tsx`

- [ ] **Step 1: Write failing Chat Workspace tests**

Add tests proving:

- available effective Workspace Persona default is sent on the first submit when no explicit assistant is selected.
- explicit selected assistant overrides the Workspace default.
- unavailable effective default does not auto-apply.
- changing Workspace default after a session exists does not mutate the persisted chat metadata.
- inspector displays “Inherited from workspace” for inherited defaults and “Explicit” for selected assistant.

Example expected submit payload:

```ts
expect(getSubmitPayload()).toMatchObject({
  requestOverrides: expect.objectContaining({
    assistant_kind: "persona",
    assistant_id: "persona-1",
    persona_memory_mode: "read_only"
  })
})
```

If the existing chat hook requires a different option shape, adapt the test to the actual `useMessageOption`/`useChatActions` contract but keep the assertion at the eventual create-chat payload boundary.

- [ ] **Step 2: Thread effective default into panel**

Add a prop:

```ts
effectiveAssistantDefault?: EffectiveWorkspaceAssistantDefault | null
```

Pass it from `ChatWorkspacePage` from active Workspace state. Do not set the global selected assistant.

- [ ] **Step 3: Apply default only at startup**

In `WorkspaceChatPanel.tsx`:

- derive `workspaceDefaultAssistant` only when:
  - Workspace scope is active,
  - effective default status is `available`,
  - assistant kind is `persona`,
  - no explicit assistant is selected,
  - no current server chat/session assistant metadata exists.
- include default metadata in the create/send path before first send.
- keep existing conversation metadata as the authority after creation.

If `useMessageOption` cannot accept inherited assistant metadata today, add an explicit option to the hook or `useChatActions` with tests in the same task. Keep it route-scoped; do not alter ordinary global `/chat` behavior.

- [ ] **Step 4: Update inspector runtime state**

Extend `ChatWorkspaceRuntimeState` with:

```ts
assistantSource: "explicit" | "workspace" | "none" | "unavailable"
workspaceAssistantDegradedReason?: string | null
```

Render source labels without duplicating Persona content.

- [ ] **Step 5: Run Chat Workspace tests**

Run:

```bash
bunx vitest run \
  apps/packages/ui/src/components/Option/ChatWorkspace/__tests__/WorkspaceChatPanel.test.tsx \
  apps/packages/ui/src/components/Option/ChatWorkspace/__tests__/InspectorRail.test.tsx
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add apps/packages/ui/src/components/Option/ChatWorkspace/ChatWorkspacePage.tsx \
  apps/packages/ui/src/components/Option/ChatWorkspace/WorkspaceChatPanel.tsx \
  apps/packages/ui/src/components/Option/ChatWorkspace/types.ts \
  apps/packages/ui/src/components/Option/ChatWorkspace/InspectorRail.tsx \
  apps/packages/ui/src/components/Option/ChatWorkspace/__tests__/WorkspaceChatPanel.test.tsx \
  apps/packages/ui/src/components/Option/ChatWorkspace/__tests__/InspectorRail.test.tsx
git commit -m "feat: apply workspace persona defaults to chat workspace startup"
```

### Task 6: End-To-End Regression And Docs Closeout

**Files:**

- Update: `Docs/Product/Workspace_Persona_Defaults_PRD.md`
- Update: `backlog/tasks/<implementation-task>.md`
- Optional test: `apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts` or a focused Chat Workspace E2E if the harness already supports Persona setup.

- [ ] **Step 1: Add focused integration regression**

Prefer backend/API-level regression first:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Workspaces/test_workspace_assistant_defaults_api.py \
  tldw_Server_API/tests/Chat/unit/test_chat_conversations_api.py \
  -q
```

If the existing E2E harness can cheaply create a Persona and Workspace, add one happy-path browser test proving the first Workspace chat create payload includes the inherited Persona metadata.

- [ ] **Step 2: Run full touched-scope checks**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/ChaChaNotesDB/test_workspace_assistant_defaults_db.py \
  tldw_Server_API/tests/Workspaces/test_workspace_assistant_defaults_api.py \
  tldw_Server_API/tests/Workspaces/test_workspaces_api.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_workspace_sub_resources_db.py \
  -q
```

Run:

```bash
bunx vitest run \
  apps/packages/ui/src/services/__tests__/tldw-api-client.assistant-identity.test.ts \
  apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/WorkspaceHeader.test.tsx \
  apps/packages/ui/src/components/Option/ChatWorkspace/__tests__/WorkspaceChatPanel.test.tsx \
  apps/packages/ui/src/components/Option/ChatWorkspace/__tests__/InspectorRail.test.tsx
```

Run:

```bash
source .venv/bin/activate && python -m bandit -r \
  tldw_Server_API/app/api/v1/schemas/workspace_schemas.py \
  tldw_Server_API/app/api/v1/endpoints/workspaces.py \
  tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  -f json -o /tmp/bandit_workspace_assistant_defaults.json
```

- [ ] **Step 3: Update PRD status note**

In `Docs/Product/Workspace_Persona_Defaults_PRD.md`, change status from `Draft` to an implementation status such as:

```markdown
Status: V1 implemented for Workspace schema/API, Workspace settings, and Chat Workspace startup.
```

Do not mark later surfaces implemented.

- [ ] **Step 4: Commit closeout**

```bash
git add Docs/Product/Workspace_Persona_Defaults_PRD.md backlog/tasks/<implementation-task>.md
git commit -m "docs: record workspace assistant defaults closeout"
```

## Review Notes

- The highest-risk decision is whether Workspace defaults should be a JSON column or companion table. The PRD allows both; this plan chooses a JSON column because existing Workspace settings already use direct fields and V1 stores one small reference object. If reviewers prefer queryable audit/history later, a companion table can be introduced without changing the API contract.
- Treat `effective_assistant_default` as a runtime view. Do not persist labels or degraded labels.
- Use `permission_denied` for redacted missing/inaccessible Persona unless implementation can confidently distinguish deleted from inaccessible without leaking cross-user existence.
- Keep `read_write` confirmation backend-enforced; frontend-only confirmation is not enough.
- Do not auto-select the inherited assistant in global assistant storage. It is a Workspace-scoped startup hint only.

## Final Verification Checklist

- [ ] Backend schema/storage tests pass.
- [ ] Workspace API tests pass.
- [ ] Frontend client/store tests pass.
- [ ] Workspace settings tests pass.
- [ ] Chat Workspace startup tests pass.
- [ ] `git diff --check` passes.
- [ ] Bandit touched backend scope passes or findings are resolved.
- [ ] PR body includes a human-written Change summary explaining what changed and why.
