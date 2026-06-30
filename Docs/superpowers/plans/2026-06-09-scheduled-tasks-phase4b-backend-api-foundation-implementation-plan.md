# Scheduled Tasks Phase 4B Backend API Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Phase 4B Scheduled Tasks API foundation for durable Recurring Question and Agent Task definitions, durable previews, lifecycle management, audit, normalized control-plane projection, and reference-client WebUI wiring without executing scheduled work.

**Architecture:** Add Scheduled Tasks-owned definition resources under the existing `/api/v1/scheduled-tasks` namespace. Keep persistence behind a focused repository backed first by per-user SQLite, layer business rules in a service, expose resource-oriented endpoints, and project definitions into the existing scheduled-tasks control-plane list as a new `automation_definition` primitive. The WebUI should act as an API client that can preview, create, inspect, edit, pause/resume, archive, and duplicate definitions while clearly showing execution is unavailable.

**Tech Stack:** FastAPI, Pydantic, SQLite, pytest, Loguru, Next.js package UI, TypeScript, React, Ant Design, Vitest, Testing Library, existing scheduled-tasks control-plane client and components.

---

## Source Inputs

- Approved spec: `Docs/superpowers/specs/2026-06-09-scheduled-tasks-phase4b-backend-api-foundation-design.md`
- Prior Phase 4 API contract spec: `Docs/superpowers/specs/2026-06-09-scheduled-tasks-phase4-recurring-question-agent-task-api-contract-design.md`
- Prior Phase 4A shell plan: `Docs/superpowers/plans/2026-06-09-scheduled-tasks-phase4a-api-first-planned-shell-implementation-plan.md`
- Scheduler vs Jobs ADR: `Docs/ADR/003-jobs-vs-scheduler-default.md`
- Backlog: `TASK-2350`

## Scope Check

This plan is one backend/API foundation slice with a reference WebUI client. It intentionally does not implement execution.

In scope:

- capability discovery for `recurring_question` and `agent_task`;
- durable preview records for create/update;
- persisted definitions;
- lifecycle mutations: create, update, pause, resume, archive, duplicate;
- per-definition audit;
- optional idempotency support for mutating endpoints;
- normalized `automation_definition` rows in `/api/v1/scheduled-tasks`;
- WebUI preview/create/edit/lifecycle/detail wiring over the API.

Out of scope:

- scheduled execution, manual run, Jobs enqueueing, Scheduler integration;
- RAG query execution;
- ACP/API agent dispatch;
- approval queue mutations;
- notification delivery;
- fake run rows, fake results, fake Home cards;
- migration or replacement of Watchlists or `/acp/schedules`.

## File Structure

| File | Responsibility |
| --- | --- |
| `tldw_Server_API/app/api/v1/schemas/scheduled_tasks_automation_schemas.py` | New Pydantic contract for capabilities, previews, definitions, schedules, visibility, lifecycle, audit, and errors. |
| `tldw_Server_API/app/core/DB_Management/Scheduled_Tasks_DB.py` | Per-user SQLite repository for previews, definitions, audit events, and idempotency records. |
| `tldw_Server_API/app/services/scheduled_task_automation_service.py` | Business rules: capabilities, validation, preview consumption, lifecycle transitions, duplicate semantics, audit, idempotency, and projection helpers. |
| `tldw_Server_API/app/api/v1/endpoints/scheduled_tasks_control_plane.py` | Add static child routes before `/{task_id}` and connect them to the automation service. Preserve existing reminder endpoints. |
| `tldw_Server_API/app/api/v1/schemas/scheduled_tasks_control_plane_schemas.py` | Extend `ScheduledTaskPrimitive` with `automation_definition`. |
| `tldw_Server_API/app/services/scheduled_tasks_control_plane_service.py` | Compose automation definitions into the existing list/detail read model without changing reminders or Watchlists behavior. |
| `tldw_Server_API/tests/Notifications/test_scheduled_task_automation_db.py` | Repository unit tests. |
| `tldw_Server_API/tests/Notifications/test_scheduled_task_automation_service.py` | Service/business-rule unit tests. |
| `tldw_Server_API/tests/Notifications/test_scheduled_task_automation_api.py` | Endpoint integration tests for capabilities, previews, definitions, lifecycle, audit, idempotency, route ordering, and permissions. |
| `tldw_Server_API/tests/Notifications/test_scheduled_tasks_control_plane.py` | Extend existing projection compatibility tests for `automation_definition`. |
| `apps/packages/ui/src/services/scheduled-tasks-control-plane.ts` | Add API client types and methods for capabilities, previews, definitions, lifecycle, and audit. |
| `apps/packages/ui/src/components/Option/ScheduledTasks/scheduled-task-automation-status.ts` | Pure helper for `automation_definition` family/type/status/lifecycle copy. |
| `apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskAutomationDefinitionEditor.tsx` | Preview-backed create/edit form for Recurring Question and Agent Task definitions. |
| `apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskCreatePanel.tsx` | Replace planned-only state with API-driven create affordances when capabilities allow definition management. |
| `apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskTable.tsx` | Add `automation_definition` filter/type/status handling and lifecycle actions. |
| `apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskDetailDrawer.tsx` | Show definition detail, execution-unavailable copy, preview history, audit events, and lifecycle actions. |
| `apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTasksPage.tsx` | Wire API client flows, optimistic refresh, and error messages. |
| `apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/scheduled-task-automation-status.test.ts` | Pure helper tests for status/type copy. |
| `apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTaskAutomationDefinitionEditor.test.tsx` | Editor preview/save tests. |
| `apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx` | Integration tests for create/manage/detail flows and no fake results. |
| `backlog/tasks/task-2350 - Plan-Scheduled-Tasks-Phase-4B-backend-API-foundation-implementation.md` | Planning task notes and verification record. |

## Guardrails

- Do not enqueue Jobs, register Scheduler tasks, or run RAG/agent work.
- Do not create run history, results, notifications, or Home cards for 4B definitions.
- Do not collapse Watchlists into Scheduled Tasks. Watchlists remains an external management workspace.
- Do not store raw Agent Task message text inline in definition, preview, or audit rows.
- Keep all new list endpoints paginated and filterable.
- Register static `/scheduled-tasks/capabilities`, `/scheduled-tasks/previews`, and `/scheduled-tasks/definitions` routes before `/{task_id}`.
- Treat `owner_id` as part of the logical key for definitions, previews, audit reads, and idempotency records.
- Use `TASKS_READ` for read endpoints and `TASKS_CONTROL` for preview/create/update/lifecycle mutations.
- Use `archive`, not hard delete.

## Task 1: Backend Schema Contracts And Route Skeleton

**Files:**
- Create: `tldw_Server_API/app/api/v1/schemas/scheduled_tasks_automation_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/scheduled_tasks_control_plane_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/scheduled_tasks_control_plane.py`
- Create: `tldw_Server_API/tests/Notifications/test_scheduled_task_automation_api.py`

- [x] **Step 1: Write failing route and schema tests**

Add tests that prove static child routes are not captured by `/{task_id}` and that capabilities return explicit unavailable execution actions.

The existing `scheduled_tasks_client` fixture currently lives inside `test_scheduled_tasks_control_plane.py`, so it is not visible to a new test module. In `test_scheduled_task_automation_api.py`, either move the shared fixture helpers into `tldw_Server_API/tests/Notifications/conftest.py` or duplicate a small local fixture block with `_make_principal`, `_override_auth`, and `scheduled_tasks_client`. Do not import fixtures from another test module.

```python
def test_scheduled_task_static_child_routes_do_not_resolve_as_task_ids(scheduled_tasks_client, auth_headers):
    for path in (
        "/api/v1/scheduled-tasks/capabilities",
        "/api/v1/scheduled-tasks/previews",
        "/api/v1/scheduled-tasks/definitions",
    ):
        response = scheduled_tasks_client.get(path, headers=auth_headers)
        assert response.status_code != 404, response.text
        assert response.text != "scheduled_task_not_found"


def test_capabilities_report_definition_actions_but_no_execution(scheduled_tasks_client, auth_headers):
    response = scheduled_tasks_client.get("/api/v1/scheduled-tasks/capabilities", headers=auth_headers)

    assert response.status_code == 200, response.text
    body = response.json()
    families = {item["family"]: item for item in body["items"]}
    assert {"recurring_question", "agent_task"} <= set(families)
    for family in ("recurring_question", "agent_task"):
        actions = families[family]["actions"]
        assert actions["preview"]["status"] == "available"
        assert actions["create_definition"]["status"] == "available"
        assert actions["execute"]["status"] == "unavailable"
        assert actions["execute"]["reason"] == "execution_not_implemented"
```

- [x] **Step 2: Run tests to verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Notifications/test_scheduled_task_automation_api.py -q
```

Expected: FAIL because schemas/routes do not exist or static routes are shadowed.

- [x] **Step 3: Add schema types**

Create `scheduled_tasks_automation_schemas.py` with Pydantic models and literal aliases. Include at minimum:

```python
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ScheduledTaskAutomationFamily = Literal["recurring_question", "agent_task"]
ScheduledTaskAutomationActionStatus = Literal["available", "unavailable", "planned", "disabled"]
ScheduledTaskAutomationFamilyAvailability = Literal["available", "planned", "unavailable", "degraded"]
ScheduledTaskPreviewMode = Literal["create", "update"]
ScheduledTaskPreviewStatus = Literal["valid", "invalid", "expired", "consumed"]
ScheduledTaskDefinitionLifecycle = Literal["configured", "paused", "archived", "disabled"]
ScheduledTaskDefinitionHealth = Literal[
    "ready",
    "execution_unavailable",
    "capability_unavailable",
    "needs_attention",
    "permission_required",
]


class ScheduledTaskActionCapability(BaseModel):
    status: ScheduledTaskAutomationActionStatus
    reason: str | None = None
    required_permissions: list[str] = Field(default_factory=list)


class ScheduledTaskAutomationCapability(BaseModel):
    family: ScheduledTaskAutomationFamily
    family_availability: ScheduledTaskAutomationFamilyAvailability
    actions: dict[str, ScheduledTaskActionCapability]
    missing_dependencies: list[str] = Field(default_factory=list)
    related_capabilities: dict[str, Any] = Field(default_factory=dict)
    reason: str | None = None
    schema_version: str = "2026-06-09"


class ScheduledTaskAutomationCapabilitiesResponse(BaseModel):
    items: list[ScheduledTaskAutomationCapability] = Field(default_factory=list)
```

Also define request/response model placeholders needed by later tasks:

- `ScheduledTaskPreviewCreateRequest`
- `ScheduledTaskPreviewResponse`
- `ScheduledTaskDefinitionCreateRequest`
- `ScheduledTaskDefinitionUpdateRequest`
- `ScheduledTaskDefinitionResponse`
- `ScheduledTaskDefinitionListResponse`
- `ScheduledTaskAuditEventResponse`
- `ScheduledTaskAuditListResponse`
- `ScheduledTaskDuplicateRequest`

Use typed fields from the spec, but keep nested `config`, `input`, `schedule`, `visibility_policy`, and `notification_policy` as structured `dict[str, Any]` initially. Later tasks tighten validation in the service layer.

- [x] **Step 4: Extend normalized primitive type**

In `scheduled_tasks_control_plane_schemas.py`, change:

```python
ScheduledTaskPrimitive = Literal["reminder_task", "watchlist_job"]
```

to:

```python
ScheduledTaskPrimitive = Literal["reminder_task", "watchlist_job", "automation_definition"]
```

- [x] **Step 5: Add route skeleton before `/{task_id}`**

In `scheduled_tasks_control_plane.py`, register these routes before the existing `@router.get("/{task_id}")`:

```python
@router.get(
    "/capabilities",
    response_model=ScheduledTaskAutomationCapabilitiesResponse,
    dependencies=[Depends(rbac_rate_limit("tasks.read"))],
)
async def get_scheduled_task_automation_capabilities(
    _principal=Depends(RequirePermission(TASKS_READ)),  # noqa: B008
    service: ScheduledTaskAutomationService = Depends(get_scheduled_task_automation_service),
) -> ScheduledTaskAutomationCapabilitiesResponse:
    return service.get_capabilities()
```

Add skeleton `GET /previews`, `GET /definitions`, and `GET /definitions/{definition_id}/audit` routes that return empty paginated responses until later tasks implement storage. Add `POST /previews`, `POST /definitions`, `PATCH /definitions/{definition_id}`, and lifecycle route skeletons that call service methods or return explicit `501` not-implemented responses while preserving route shape.

- [x] **Step 6: Implement minimal capability service**

Create `tldw_Server_API/app/services/scheduled_task_automation_service.py` with:

```python
class ScheduledTaskAutomationService:
    """Business service for Scheduled Tasks-owned automation definitions."""

    def get_capabilities(self) -> ScheduledTaskAutomationCapabilitiesResponse:
        actions = {
            "preview": ScheduledTaskActionCapability(status="available", required_permissions=[TASKS_CONTROL]),
            "create_definition": ScheduledTaskActionCapability(status="available", required_permissions=[TASKS_CONTROL]),
            "update_definition": ScheduledTaskActionCapability(status="available", required_permissions=[TASKS_CONTROL]),
            "pause": ScheduledTaskActionCapability(status="available", required_permissions=[TASKS_CONTROL]),
            "resume": ScheduledTaskActionCapability(status="available", required_permissions=[TASKS_CONTROL]),
            "archive": ScheduledTaskActionCapability(status="available", required_permissions=[TASKS_CONTROL]),
            "duplicate": ScheduledTaskActionCapability(status="available", required_permissions=[TASKS_CONTROL]),
            "execute": ScheduledTaskActionCapability(
                status="unavailable",
                reason="execution_not_implemented",
                required_permissions=[TASKS_CONTROL],
            ),
        }
        return ScheduledTaskAutomationCapabilitiesResponse(
            items=[
                ScheduledTaskAutomationCapability(
                    family="recurring_question",
                    family_availability="available",
                    actions=actions,
                    related_capabilities={"rag": {"status": "not_checked"}},
                ),
                ScheduledTaskAutomationCapability(
                    family="agent_task",
                    family_availability="available",
                    actions=actions,
                    related_capabilities={"acp": {"status": "not_checked"}},
                ),
            ]
        )
```

- [x] **Step 7: Run route/schema tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Notifications/test_scheduled_task_automation_api.py -q
python -m pytest tldw_Server_API/tests/Notifications/test_scheduled_tasks_control_plane.py -q
```

Expected: PASS for capability/static route tests and existing control-plane tests.

- [x] **Step 8: Commit schema and route skeleton**

```bash
git add \
  tldw_Server_API/app/api/v1/schemas/scheduled_tasks_automation_schemas.py \
  tldw_Server_API/app/api/v1/schemas/scheduled_tasks_control_plane_schemas.py \
  tldw_Server_API/app/api/v1/endpoints/scheduled_tasks_control_plane.py \
  tldw_Server_API/app/services/scheduled_task_automation_service.py \
  tldw_Server_API/tests/Notifications/test_scheduled_task_automation_api.py
git commit -m "feat: add scheduled task automation API skeleton"
```

## Task 2: Repository And Per-User SQLite Storage

**Files:**
- Create: `tldw_Server_API/app/core/DB_Management/Scheduled_Tasks_DB.py`
- Create: `tldw_Server_API/tests/Notifications/test_scheduled_task_automation_db.py`

- [x] **Step 1: Write failing repository tests**

Test per-user isolation, preview persistence, definition persistence, disabled lock fields, audit events, idempotency scoping, pagination, filtering, and redacted Agent Task storage.

```python
def test_scheduled_tasks_repository_isolates_users(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path))
    repo_a = ScheduledTasksDatabase.for_user(user_id=101)
    repo_b = ScheduledTasksDatabase.for_user(user_id=202)
    repo_a.ensure_schema()
    repo_b.ensure_schema()

    preview = repo_a.create_preview(
        owner_id=101,
        mode="create",
        family="recurring_question",
        definition_id=None,
        definition_version=None,
        status="valid",
        payload_hash="hash-a",
        normalized_config={"name": "Question", "input": {"question": "What changed?"}},
        validation_errors=[],
        warnings=[],
        risk_class=None,
        visibility_policy="findings_only",
        schedule_preview={"summary": "daily"},
        redaction_policy={"mode": "none"},
        expires_at="2026-06-10T00:00:00+00:00",
        created_by="101",
    )

    assert repo_a.get_preview(owner_id=101, preview_id=preview.id).id == preview.id
    assert repo_b.get_preview(owner_id=202, preview_id=preview.id) is None
```

Add separate tests:

- `test_create_definition_and_audit_roundtrip`
- `test_update_preview_consumption_sets_consumed_at`
- `test_idempotency_records_are_owner_and_route_scoped`
- `test_list_definitions_filters_by_family_lifecycle_health_and_query`
- `test_definition_persists_disabled_lock_kind_and_reason`
- `test_agent_task_definition_and_audit_storage_do_not_contain_raw_message_secret`

- [x] **Step 2: Run repository tests to verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Notifications/test_scheduled_task_automation_db.py -q
```

Expected: FAIL because `Scheduled_Tasks_DB.py` does not exist.

- [x] **Step 3: Implement row dataclasses and schema**

Create `Scheduled_Tasks_DB.py` with:

- `PreviewRow`
- `DefinitionRow`
- `AuditEventRow`
- `IdempotencyRecordRow`
- `ScheduledTasksDatabase`

Use SQLite tables:

```sql
CREATE TABLE IF NOT EXISTS scheduled_task_previews (
    id TEXT PRIMARY KEY,
    owner_id INTEGER NOT NULL,
    mode TEXT NOT NULL,
    family TEXT NOT NULL,
    definition_id TEXT,
    definition_version INTEGER,
    status TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    normalized_config_json TEXT NOT NULL,
    validation_errors_json TEXT NOT NULL,
    warnings_json TEXT NOT NULL,
    risk_class TEXT,
    visibility_policy TEXT NOT NULL,
    schedule_preview_json TEXT NOT NULL,
    redaction_policy_json TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    created_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    consumed_at TEXT,
    created_definition_id TEXT
);
```

Also create:

- `scheduled_task_definitions`
- `scheduled_task_audit_events`
- `scheduled_task_idempotency`

`scheduled_task_definitions` must include:

- `disabled_lock_kind TEXT NOT NULL DEFAULT 'none'`
- `disabled_reason TEXT`

Use `disabled_lock_kind` values `none`, `admin`, `security`, and `system`. This field is required by duplicate guardrails: admin/security locked disabled definitions cannot be copied into a future runnable state.

Indexes:

- previews: `(owner_id, created_at)`, `(owner_id, definition_id)`, `(owner_id, status)`
- definitions: `(owner_id, family)`, `(owner_id, lifecycle)`, `(owner_id, health)`, `(owner_id, updated_at)`
- audit: `(definition_id, created_at)`
- idempotency: unique `(owner_id, route, key)`

- [x] **Step 4: Implement repository methods**

Add focused methods:

- `for_user(user_id: int) -> ScheduledTasksDatabase`
- `ensure_schema() -> None`
- `create_preview(...) -> PreviewRow`
- `get_preview(owner_id: int, preview_id: str) -> PreviewRow | None`
- `list_previews(owner_id: int, filters..., limit: int, offset: int) -> tuple[list[PreviewRow], int]`
- `mark_preview_consumed(owner_id: int, preview_id: str, created_definition_id: str | None) -> PreviewRow`
- `create_definition(...) -> DefinitionRow`
- `get_definition(owner_id: int, definition_id: str) -> DefinitionRow | None`
- `list_definitions(owner_id: int, filters..., limit: int, offset: int) -> tuple[list[DefinitionRow], int]`
- `update_definition(owner_id: int, definition_id: str, patch: dict[str, Any], expected_version: int | None) -> DefinitionRow`
- `create_audit_event(...) -> AuditEventRow`
- `list_audit_events(owner_id: int, definition_id: str, filters..., limit: int, offset: int)`
- `get_idempotency_record(owner_id: int, route: str, key: str) -> IdempotencyRecordRow | None`
- `create_idempotency_record(...) -> IdempotencyRecordRow`

Use JSON helpers that always encode dictionaries/lists with sorted keys.

- [x] **Step 5: Run repository tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Notifications/test_scheduled_task_automation_db.py -q
```

Expected: PASS.

- [x] **Step 6: Commit repository**

```bash
git add \
  tldw_Server_API/app/core/DB_Management/Scheduled_Tasks_DB.py \
  tldw_Server_API/tests/Notifications/test_scheduled_task_automation_db.py
git commit -m "feat: add scheduled task automation repository"
```

## Task 3: Preview Validation And Definition Lifecycle Service

**Files:**
- Modify: `tldw_Server_API/app/services/scheduled_task_automation_service.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/scheduled_tasks_automation_schemas.py`
- Create: `tldw_Server_API/tests/Notifications/test_scheduled_task_automation_service.py`

- [x] **Step 1: Write failing service tests for preview validation**

Cover valid, invalid, expired, stale, consumed, cross-user, and redacted Agent Task previews.

```python
def test_create_preview_persists_invalid_semantic_preview(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path))
    service = ScheduledTaskAutomationService()

    preview = service.create_preview(
        owner_id=880,
        actor="880",
        payload=ScheduledTaskPreviewCreateRequest(
            mode="create",
            family="recurring_question",
            definition_id=None,
            definition_version=None,
            config={
                "name": "",
                "input": {"question": ""},
                "schedule": {},
                "visibility_policy": "findings_only",
                "notification_policy": {},
            },
        ),
    )

    assert preview.status == "invalid"
    assert preview.validation_errors
    assert preview.id
```

```python
def test_agent_task_preview_redacts_raw_message(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path))
    service = ScheduledTaskAutomationService()

    preview = service.create_preview(
        owner_id=880,
        actor="880",
        payload=ScheduledTaskPreviewCreateRequest(
            mode="create",
            family="agent_task",
            definition_id=None,
            definition_version=None,
            config={
                "name": "Agent check",
                "input": {
                    "agent_ref": {"kind": "acp_agent", "id": "codex"},
                    "message": "secret api key sk-test-123 should not leak",
                    "allowed_tool_classes": [],
                    "denied_tool_classes": [],
                    "approval_policy": {"mode": "none"},
                },
                "schedule": {"kind": "daily", "timezone": "UTC"},
                "visibility_policy": "failures_and_approvals",
                "notification_policy": {},
            },
        ),
    )

    serialized = preview.model_dump_json()
    assert "sk-test-123" not in serialized
    assert preview.normalized_config["input"]["message_payload"]["storage_mode"] == "redacted_only"
```

Add tests for:

- create consumes valid preview;
- update requires preview version match;
- duplicate creates paused copy and two audit events;
- duplicate of disabled definitions succeeds only for non-admin/non-security lock kinds;
- duplicate of `disabled_lock_kind` `admin` or `security` fails and creates no copy;
- pause/resume/archive transition matrix;
- preview idempotency replay and same-key/different-payload conflict;
- create idempotency replay before preview consumption;
- update idempotency replay after preview consumption;
- duplicate idempotency replay without creating a second copy or second audit pair;
- pause/resume/archive idempotency replay without extra audit events;
- same key different payload conflict for representative preview/create/lifecycle routes;
- no-key consumed preview reuse fails.
- Agent Task raw secret strings are absent from preview response, created definition response, definition list/detail response models, and audit response models.

- [x] **Step 2: Run service tests to verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Notifications/test_scheduled_task_automation_service.py -q
```

Expected: FAIL because service methods are not implemented.

- [x] **Step 3: Implement canonical hashing and validation helpers**

In `scheduled_task_automation_service.py`, add private helpers:

```python
def _canonical_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
```

Validation helpers:

- `_validate_recurring_question_config(config) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]`
- `_validate_agent_task_config(config) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]`
- `_validate_schedule(schedule) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]`
- `_normalize_visibility_policy(family, value) -> str`
- `_redact_agent_message(message: str) -> dict[str, Any]`

Keep validation conservative:

- require non-empty `name`;
- require Recurring Question `input.question`;
- require Agent Task `input.agent_ref` and `input.message`;
- validate schedule `kind` only against `one_time`, `interval`, `daily`, `weekly`, `cron`;
- do not call RAG or ACP live execution APIs in 4B.

- [x] **Step 4: Implement preview creation**

`create_preview(owner_id, actor, payload)` should:

- compute preview payload hash from `mode`, `family`, `definition_id`, `definition_version`, `config`;
- persist `valid` or `invalid` preview;
- set default expiry using a documented constant, for example 24 hours;
- return `201`-safe response model through API later;
- never persist raw Agent Task messages inline in normalized config.

For Agent Task tests, use a unique sentinel such as `RAW_AGENT_SECRET_DO_NOT_LEAK_4B`. Assert this string is absent from:

- preview response serialization;
- repository `normalized_config_json`;
- definition response serialization after create;
- definition list/detail serialization;
- audit `before`/`after` metadata;
- repository audit JSON.

- [x] **Step 5: Implement definition create/update lifecycle**

Rules:

- create/update require valid preview;
- preview owner must match current user;
- preview status must be `valid`;
- preview must be unexpired and unconsumed;
- create preview has no `definition_id`;
- update preview `definition_id` matches route and `definition_version` matches current definition version;
- successful create/update marks preview consumed;
- create/update writes deterministic audit events;
- definitions default health is `execution_unavailable`.

- [x] **Step 6: Implement lifecycle and duplicate**

Implement:

- `pause_definition`
- `resume_definition`
- `archive_definition`
- `duplicate_definition`

Follow the transition matrix:

- pause paused: idempotent 200, no audit;
- resume configured: idempotent 200, no audit;
- archive archived: idempotent 200;
- update/pause/resume archived: conflict;
- duplicate archived: conflict;
- duplicate disabled only if `disabled_lock_kind` is not `admin` or `security`;
- duplicate always creates `paused` copy;
- duplicate audit records both `definition_duplicated` and `definition_duplicate_created`.

- [x] **Step 7: Implement idempotency wrapper**

Add a service-level helper:

```python
def _with_idempotency(
    self,
    *,
    owner_id: int,
    route: str,
    key: str | None,
    payload_hash: str,
    operation: Callable[[], BaseModel],
) -> BaseModel:
    ...
```

Behavior:

- no key: run operation normally;
- same owner + route + key + same hash: return stored response;
- same owner + route + key + different hash: raise `scheduled_task_idempotency_conflict`;
- idempotency lookup happens before preview validation/consumption for mutation replay.

Apply the wrapper to all 4B mutating routes that accept `Idempotency-Key`: preview, create, update, duplicate, pause, resume, and archive.

- [x] **Step 8: Run service tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Notifications/test_scheduled_task_automation_service.py -q
```

Expected: PASS.

- [x] **Step 9: Commit service layer**

```bash
git add \
  tldw_Server_API/app/services/scheduled_task_automation_service.py \
  tldw_Server_API/app/api/v1/schemas/scheduled_tasks_automation_schemas.py \
  tldw_Server_API/tests/Notifications/test_scheduled_task_automation_service.py
git commit -m "feat: implement scheduled task definition service"
```

Task 3 review hardening also added repository write-transaction support so idempotency lookup, domain mutation, audit writes, preview consumption, and response snapshot persistence commit as one durable command.

## Task 4: API Endpoints, Errors, Permissions, And OpenAPI Contract

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/scheduled_tasks_control_plane.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/scheduled_tasks_automation_schemas.py`
- Modify: `tldw_Server_API/tests/Notifications/test_scheduled_task_automation_api.py`

- [x] **Step 1: Write failing endpoint integration tests**

Add endpoint tests for:

- preview valid/invalid responses;
- create from preview;
- update from preview;
- pause/resume/archive/duplicate;
- audit list;
- filters and pagination;
- permission failures;
- cross-user preview/definition denial;
- Agent Task sentinel redaction across preview, create, list, detail, and audit responses;
- idempotent preview replay and conflict;
- idempotent create replay after preview consumed;
- idempotent update replay after preview consumed;
- idempotent duplicate replay without a second copy or duplicate audit pair;
- idempotent pause/resume/archive replay without extra audit events;
- same idempotency key with different payload conflict;
- representative validation error envelopes for schedule invalid, scope invalid, agent ref invalid, permission policy invalid, family unavailable, and execution unavailable.

Example:

```python
def test_create_definition_consumes_preview_and_exposes_audit(scheduled_tasks_client, auth_headers):
    preview_response = scheduled_tasks_client.post(
        "/api/v1/scheduled-tasks/previews",
        headers=auth_headers,
        json={
            "mode": "create",
            "family": "recurring_question",
            "definition_id": None,
            "definition_version": None,
            "config": {
                "name": "Track licensing answer",
                "input": {"question": "Has the licensing answer appeared?"},
                "schedule": {"kind": "daily", "timezone": "UTC"},
                "visibility_policy": "findings_only",
                "notification_policy": {"home_enabled": True},
            },
        },
    )
    assert preview_response.status_code == 201, preview_response.text
    preview_id = preview_response.json()["id"]

    create_response = scheduled_tasks_client.post(
        "/api/v1/scheduled-tasks/definitions",
        headers=auth_headers,
        json={"preview_id": preview_id, "initial_lifecycle": "configured"},
    )

    assert create_response.status_code == 201, create_response.text
    definition = create_response.json()
    assert definition["family"] == "recurring_question"
    assert definition["lifecycle"] == "configured"
    assert definition["health"] == "execution_unavailable"

    audit_response = scheduled_tasks_client.get(
        f"/api/v1/scheduled-tasks/definitions/{definition['id']}/audit",
        headers=auth_headers,
    )
    assert audit_response.status_code == 200, audit_response.text
    assert audit_response.json()["items"][0]["event_type"] == "definition_created"
```

- [x] **Step 2: Run endpoint tests to verify failures**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Notifications/test_scheduled_task_automation_api.py -q
```

Expected: FAIL for unimplemented routes/error mapping.

- [x] **Step 3: Add endpoint error helper**

Use a local helper in the endpoint module to map service exceptions to consistent error bodies:

```python
def _scheduled_task_error(
    *,
    request: Request,
    status_code: int,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
    retryable: bool = False,
) -> HTTPException:
    correlation_id = getattr(getattr(request, "state", None), "request_id", None)
    return HTTPException(
        status_code=status_code,
        detail={
            "code": code,
            "message": message,
            "details": details or {},
            "field_errors": [],
            "retryable": retryable,
            "correlation_id": correlation_id,
        },
    )
```

Map service exceptions:

- `scheduled_task_family_unavailable`: 409;
- `scheduled_task_preview_required`: 400;
- `scheduled_task_definition_not_found`: 404;
- `scheduled_task_preview_mismatch`: 409;
- `scheduled_task_preview_expired`: 409;
- `scheduled_task_schedule_invalid`: 422;
- `scheduled_task_scope_invalid`: 422;
- `scheduled_task_agent_ref_invalid`: 422;
- `scheduled_task_permission_policy_invalid`: 422;
- `scheduled_task_execution_unavailable`: 409;
- `scheduled_task_definition_version_conflict`: 409;
- `scheduled_task_definition_archived`: 409;
- `scheduled_task_lifecycle_transition_invalid`: 409;
- `scheduled_task_idempotency_conflict`: 409.

Every mapped error body should include `code`, `message`, `details`, `field_errors`, `retryable`, and `correlation_id`. Tests should assert the envelope shape for at least one 409 conflict and one 422 validation error.

- [x] **Step 4: Implement all endpoint handlers**

Endpoints:

- `GET /api/v1/scheduled-tasks/capabilities`
- `POST /api/v1/scheduled-tasks/previews`
- `GET /api/v1/scheduled-tasks/previews`
- `GET /api/v1/scheduled-tasks/previews/{preview_id}`
- `POST /api/v1/scheduled-tasks/definitions`
- `GET /api/v1/scheduled-tasks/definitions`
- `GET /api/v1/scheduled-tasks/definitions/{definition_id}`
- `PATCH /api/v1/scheduled-tasks/definitions/{definition_id}`
- `POST /api/v1/scheduled-tasks/definitions/{definition_id}/pause`
- `POST /api/v1/scheduled-tasks/definitions/{definition_id}/resume`
- `POST /api/v1/scheduled-tasks/definitions/{definition_id}/archive`
- `POST /api/v1/scheduled-tasks/definitions/{definition_id}/duplicate`
- `GET /api/v1/scheduled-tasks/definitions/{definition_id}/audit`

Read `Idempotency-Key` from the request headers for mutating routes.

- [x] **Step 5: Add pagination/filter query parameters**

Use explicit query params:

- `limit: int = Query(50, ge=1, le=200)`
- `offset: int = Query(0, ge=0)`
- definitions: `family`, `lifecycle`, `health`, `visibility_policy`, `q`, `created_from`, `created_to`
- previews: `family`, `mode`, `status`, `definition_id`, `expired`
- audit: `event_type`, `actor`, `created_from`, `created_to`, `idempotency_key`, `request_id`

- [x] **Step 6: Run endpoint and existing control-plane tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Notifications/test_scheduled_task_automation_api.py \
  tldw_Server_API/tests/Notifications/test_scheduled_tasks_control_plane.py \
  -q
```

Expected: PASS.

- [x] **Step 7: Commit endpoint implementation**

```bash
git add \
  tldw_Server_API/app/api/v1/endpoints/scheduled_tasks_control_plane.py \
  tldw_Server_API/app/api/v1/schemas/scheduled_tasks_automation_schemas.py \
  tldw_Server_API/tests/Notifications/test_scheduled_task_automation_api.py
git commit -m "feat: expose scheduled task automation endpoints"
```

Task 4 review hardening also added direct public error-code aliases, audit request-id propagation/filtering, and timezone-aware datetime filter validation/normalization.

## Task 5: Unified Control-Plane Projection

**Files:**
- Modify: `tldw_Server_API/app/services/scheduled_tasks_control_plane_service.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/scheduled_tasks_control_plane_schemas.py`
- Modify: `tldw_Server_API/tests/Notifications/test_scheduled_tasks_control_plane.py`

- [x] **Step 1: Write failing projection tests**

Extend `test_scheduled_tasks_endpoint_combines_reminders_and_watchlist_jobs` or add a new test:

```python
def test_scheduled_tasks_endpoint_projects_automation_definitions(scheduled_tasks_client, auth_headers):
    # Create preview and definition through API so the projection reads real storage.
    preview_response = scheduled_tasks_client.post(
        "/api/v1/scheduled-tasks/previews",
        headers=auth_headers,
        json={
            "mode": "create",
            "family": "recurring_question",
            "definition_id": None,
            "definition_version": None,
            "config": {
                "name": "Track API question",
                "input": {"question": "Any new answer?"},
                "schedule": {"kind": "daily", "timezone": "UTC"},
                "visibility_policy": "findings_only",
                "notification_policy": {},
            },
        },
    )
    definition_response = scheduled_tasks_client.post(
        "/api/v1/scheduled-tasks/definitions",
        headers=auth_headers,
        json={"preview_id": preview_response.json()["id"], "initial_lifecycle": "configured"},
    )
    definition_id = definition_response.json()["id"]

    list_response = scheduled_tasks_client.get("/api/v1/scheduled-tasks", headers=auth_headers)

    assert list_response.status_code == 200, list_response.text
    item = next(item for item in list_response.json()["items"] if item["primitive"] == "automation_definition")
    assert item["id"] == f"automation_definition:{definition_id}"
    assert item["status"] == "configured_execution_unavailable"
    assert item["enabled"] is True
    assert item["edit_mode"] == "native"
    assert item["next_run_at"] is None
    assert item["last_run_at"] is None
    assert item["source_ref"]["family"] == "recurring_question"
    assert item["source_ref"]["health"] == "execution_unavailable"
    assert item["source_ref"]["execution_available"] is False
```

Add compatibility checks:

- reminders still use `reminder_task`;
- Watchlists still use `watchlist_job` and `external`;
- get detail works for `automation_definition:{definition_id}`;
- paused/archived definitions do not project as generic disabled unless status token says so.

- [x] **Step 2: Run projection tests to verify failures**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Notifications/test_scheduled_tasks_control_plane.py -q
```

Expected: FAIL until projection is implemented.

- [x] **Step 3: Compose automation repository into control-plane service**

In `ScheduledTasksControlPlaneService`:

- add `_scheduled_tasks_db(user_id: int) -> ScheduledTasksDatabase`;
- list active definitions and append normalized rows;
- catch noncritical automation read errors and append `automation_definitions_unavailable` to partial errors;
- implement detail lookup for `automation_definition:{definition_id}`.

Projection helper:

```python
def _automation_definition_status(row: DefinitionRow) -> tuple[bool, str]:
    if row.lifecycle == "configured":
        if row.health == "execution_unavailable":
            return True, "configured_execution_unavailable"
        if row.health == "capability_unavailable":
            return True, "blocked_capability_unavailable"
        if row.health == "permission_required":
            return True, "blocked_permission_required"
        return True, row.health
    if row.lifecycle == "paused":
        return False, "paused"
    if row.lifecycle == "archived":
        return False, "archived"
    if row.lifecycle == "disabled":
        return False, "disabled"
    return False, "needs_attention"
```

- [x] **Step 4: Run projection tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Notifications/test_scheduled_tasks_control_plane.py -q
```

Expected: PASS.

- [x] **Step 5: Commit projection**

```bash
git add \
  tldw_Server_API/app/services/scheduled_tasks_control_plane_service.py \
  tldw_Server_API/app/api/v1/schemas/scheduled_tasks_control_plane_schemas.py \
  tldw_Server_API/tests/Notifications/test_scheduled_tasks_control_plane.py
git commit -m "feat: project automation definitions into scheduled tasks"
```

## Task 6: Frontend API Client And Status Helpers

**Files:**
- Modify: `apps/packages/ui/src/services/scheduled-tasks-control-plane.ts`
- Create: `apps/packages/ui/src/components/Option/ScheduledTasks/scheduled-task-automation-status.ts`
- Create: `apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/scheduled-task-automation-status.test.ts`
- Modify: `apps/packages/ui/src/components/Option/ScheduledTasks/scheduled-task-status.ts`

- [x] **Step 1: Write failing frontend helper tests**

```ts
import { describe, expect, it } from "vitest"

import {
  getAutomationDefinitionFamilyLabel,
  getAutomationDefinitionProductStatus,
  isAutomationDefinitionTask
} from "../scheduled-task-automation-status"

describe("scheduled task automation status", () => {
  it("labels configured non-executable definitions without waiting-for-run copy", () => {
    const task = {
      id: "automation_definition:def_1",
      primitive: "automation_definition",
      title: "Track answer",
      status: "configured_execution_unavailable",
      enabled: true,
      edit_mode: "native",
      source_ref: {
        family: "recurring_question",
        lifecycle: "configured",
        health: "execution_unavailable",
        execution_available: false
      }
    } as const

    expect(isAutomationDefinitionTask(task)).toBe(true)
    expect(getAutomationDefinitionFamilyLabel(task)).toBe("Recurring question")
    expect(getAutomationDefinitionProductStatus(task).label).toBe("Configured, execution unavailable")
  })

  it("does not collapse paused automation definitions into disabled", () => {
    const status = getAutomationDefinitionProductStatus({
      id: "automation_definition:def_2",
      primitive: "automation_definition",
      title: "Agent task",
      status: "paused",
      enabled: false,
      edit_mode: "native",
      source_ref: { family: "agent_task", lifecycle: "paused", health: "execution_unavailable" }
    } as const)

    expect(status.label).toBe("Paused")
  })
})
```

- [x] **Step 2: Run helper tests to verify failure**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/ScheduledTasks/__tests__/scheduled-task-automation-status.test.ts
```

Expected: FAIL because helper does not exist and client primitive type does not include `automation_definition`.

- [x] **Step 3: Extend frontend client types and methods**

In `scheduled-tasks-control-plane.ts`:

- extend `ScheduledTaskPrimitive` with `"automation_definition"`;
- add family, action, preview, definition, audit, and lifecycle TypeScript interfaces matching backend schemas;
- add API methods:
  - `getScheduledTaskCapabilities`
  - `createScheduledTaskPreview`
  - `listScheduledTaskPreviews`
  - `getScheduledTaskPreview`
  - `createScheduledTaskDefinition`
  - `listScheduledTaskDefinitions`
  - `getScheduledTaskDefinition`
  - `updateScheduledTaskDefinition`
  - `pauseScheduledTaskDefinition`
  - `resumeScheduledTaskDefinition`
  - `archiveScheduledTaskDefinition`
  - `duplicateScheduledTaskDefinition`
  - `listScheduledTaskDefinitionAudit`

- [x] **Step 4: Implement status helper**

`scheduled-task-automation-status.ts` should:

- detect `task.primitive === "automation_definition"`;
- label family `recurring_question` as "Recurring question";
- label family `agent_task` as "Agent task";
- map `configured_execution_unavailable` to "Configured, execution unavailable";
- map lifecycle `paused`, `archived`, `disabled` before generic `enabled === false`;
- return copy that says execution is unavailable, not waiting.

Then update `scheduled-task-status.ts`:

- call automation helper first in `getScheduledTaskProductStatus`;
- call automation family helper first in `getScheduledTaskTypeLabel`;
- keep reminder/watchlist behavior unchanged.

- [x] **Step 5: Run helper tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/ScheduledTasks/__tests__/scheduled-task-automation-status.test.ts src/components/Option/ScheduledTasks/__tests__/scheduled-task-status.test.ts
```

Expected: PASS.

- [x] **Step 6: Commit client and helpers**

```bash
git add \
  apps/packages/ui/src/services/scheduled-tasks-control-plane.ts \
  apps/packages/ui/src/components/Option/ScheduledTasks/scheduled-task-automation-status.ts \
  apps/packages/ui/src/components/Option/ScheduledTasks/scheduled-task-status.ts \
  apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/scheduled-task-automation-status.test.ts
git commit -m "feat: add scheduled task automation frontend client types"
```

## Task 7: WebUI Reference Client Create, Detail, And Lifecycle Flows

**Files:**
- Create: `apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskAutomationDefinitionEditor.tsx`
- Create: `apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTaskAutomationDefinitionEditor.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskCreatePanel.tsx`
- Modify: `apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskTable.tsx`
- Modify: `apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskDetailDrawer.tsx`
- Modify: `apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTasksPage.tsx`
- Modify: `apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTaskCreatePanel.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTaskDetailDrawer.test.tsx`

- [x] **Step 1: Write failing editor tests**

Cover:

- Recurring Question preview before save;
- Agent Task preview redaction copy;
- expired preview error prompts another preview;
- save disabled until preview is valid;
- create sends only `preview_id` and `initial_lifecycle`;
- update sends only `preview_id`.

```ts
it("previews and creates a recurring question definition", async () => {
  const onPreview = vi.fn().mockResolvedValue({
    id: "preview_1",
    status: "valid",
    family: "recurring_question",
    normalized_config: { name: "Track answer" },
    validation_errors: [],
    warnings: [],
    expires_at: "2026-06-10T00:00:00Z"
  })
  const onCreate = vi.fn().mockResolvedValue({ id: "definition_1" })

  render(
    <ScheduledTaskAutomationDefinitionEditor
      family="recurring_question"
      mode="create"
      onPreview={onPreview}
      onCreate={onCreate}
      onCancel={vi.fn()}
    />
  )

  await userEvent.type(screen.getByLabelText("Question"), "Has the answer appeared?")
  await userEvent.click(screen.getByRole("button", { name: "Preview" }))
  await screen.findByText("Preview ready")
  await userEvent.click(screen.getByRole("button", { name: "Save definition" }))

  expect(onCreate).toHaveBeenCalledWith({ preview_id: "preview_1", initial_lifecycle: "configured" })
})
```

- [x] **Step 2: Run editor tests to verify failure**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/ScheduledTasks/__tests__/ScheduledTaskAutomationDefinitionEditor.test.tsx
```

Expected: FAIL because editor does not exist.

- [x] **Step 3: Implement editor component**

Keep the editor focused:

- common fields: name, description, schedule kind, timezone, visibility;
- Recurring Question fields: question, success criteria, scope JSON textarea;
- Agent Task fields: agent ref JSON/text, message, allowed/denied tool classes, approval mode;
- Preview button builds full config and calls `onPreview`;
- Save button consumes preview via `onCreate` or `onUpdate`;
- Show validation errors from preview response;
- Show "Execution is not available yet" after valid preview and save.

Do not implement advanced schedule builders in this task. Use basic fields and JSON textareas where necessary to keep the reference client honest and API-first.

- [x] **Step 4: Wire create panel and page**

In `ScheduledTasksPage.tsx`:

- fetch capabilities on load;
- if `create_definition` for Recurring Question or Agent Task is available, render editor instead of planned-only panel;
- call API client preview/create/update methods;
- refresh list after successful create/update/lifecycle;
- show API error messages from `detail.code`/`detail.message`.

In `ScheduledTaskCreatePanel.tsx`:

- keep existing Reminder and Watch/Ingest behavior unchanged;
- for API-available `recurring_question` and `agent_task`, expose create affordance;
- keep planned copy when capability fetch fails or action is unavailable.

- [x] **Step 5: Wire table and detail drawer**

Table:

- add `automation_definition` type filter option;
- label family using helper;
- show lifecycle actions for automation definitions;
- do not show Results button unless real results exist.

Detail drawer:

- show lifecycle, health, schedule, visibility, notification policy;
- show preview history and audit events;
- show "Execution is not available yet";
- show pause/resume/archive/duplicate actions;
- no fake run rows.

- [x] **Step 6: Run frontend tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run \
  src/components/Option/ScheduledTasks/__tests__/ScheduledTaskAutomationDefinitionEditor.test.tsx \
  src/components/Option/ScheduledTasks/__tests__/ScheduledTaskCreatePanel.test.tsx \
  src/components/Option/ScheduledTasks/__tests__/ScheduledTaskDetailDrawer.test.tsx \
  src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx \
  src/components/Option/ScheduledTasks/__tests__/scheduled-task-results.test.ts
```

Expected: PASS.

- [x] **Step 7: Commit WebUI wiring**

```bash
git add \
  apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskAutomationDefinitionEditor.tsx \
  apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskCreatePanel.tsx \
  apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskTable.tsx \
  apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskDetailDrawer.tsx \
  apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTasksPage.tsx \
  apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTaskAutomationDefinitionEditor.test.tsx \
  apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTaskCreatePanel.test.tsx \
  apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTaskDetailDrawer.test.tsx \
  apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx
git commit -m "feat: wire scheduled task automation reference client"
```

## Task 8: Focused End-To-End Verification, Docs, And Cleanup

**Files:**
- Modify: `Docs/superpowers/plans/2026-06-09-scheduled-tasks-phase4b-backend-api-foundation-implementation-plan.md`
- Modify: `backlog/tasks/task-2351 - Implement-Scheduled-Tasks-Phase-4B-backend-API-foundation.md`
- Modify: `tldw_Server_API/Config_Files/config.txt`
- Optional modify: `Docs/Development/Scheduled_Tasks.md` if an existing scheduled-tasks API doc exists.

- [x] **Step 1: Run backend focused tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Notifications/test_scheduled_task_automation_db.py \
  tldw_Server_API/tests/Notifications/test_scheduled_task_automation_service.py \
  tldw_Server_API/tests/Notifications/test_scheduled_task_automation_api.py \
  tldw_Server_API/tests/Notifications/test_scheduled_tasks_control_plane.py \
  -q
```

Expected: PASS.

Result: PASS, `82 passed, 13 warnings in 65.27s`.

- [x] **Step 2: Run frontend focused tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run \
  src/components/Option/ScheduledTasks/__tests__/scheduled-task-automation-status.test.ts \
  src/components/Option/ScheduledTasks/__tests__/ScheduledTaskAutomationDefinitionEditor.test.tsx \
  src/components/Option/ScheduledTasks/__tests__/ScheduledTaskCreatePanel.test.tsx \
  src/components/Option/ScheduledTasks/__tests__/ScheduledTaskDetailDrawer.test.tsx \
  src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx \
  src/components/Option/ScheduledTasks/__tests__/scheduled-task-results.test.ts
```

Expected: PASS.

Result: PASS, `102 passed`.

- [x] **Step 3: Run Bandit on touched backend scope**

Run:

```bash
source .venv/bin/activate
python -m bandit -r \
  tldw_Server_API/app/api/v1/endpoints/scheduled_tasks_control_plane.py \
  tldw_Server_API/app/api/v1/schemas/scheduled_tasks_automation_schemas.py \
  tldw_Server_API/app/core/DB_Management/Scheduled_Tasks_DB.py \
  tldw_Server_API/app/services/scheduled_task_automation_service.py \
  -f json -o /tmp/bandit_scheduled_tasks_phase4b.json
```

Expected: no new findings in touched code. If Bandit reports findings, inspect `/tmp/bandit_scheduled_tasks_phase4b.json` and fix before continuing.

Result: PASS, `/tmp/bandit_scheduled_tasks_phase4b.json` reported 0 errors and 0 findings.

- [x] **Step 4: Verify no execution paths were added**

Run:

```bash
rg -n "enqueue|Scheduler|APScheduler|run_now|manual run|execute" \
  tldw_Server_API/app/services/scheduled_task_automation_service.py \
  tldw_Server_API/app/api/v1/endpoints/scheduled_tasks_control_plane.py \
  apps/packages/ui/src/components/Option/ScheduledTasks
```

Expected: only capability/status/copy references such as `execution_unavailable` and `execution_not_implemented`; no Jobs enqueue, Scheduler registration, or run execution code.

Result: PASS. Matches were limited to the unsupported `execute` action capability and existing APScheduler reminder utility tests; no enqueue, Scheduler registration, `run_now`, manual-run, RAG execution, ACP dispatch, notification, fake run, fake result, or Home surfacing code was added.

- [x] **Step 5: Verify OpenAPI generation/import health**

Run:

```bash
source .venv/bin/activate
python - <<'PY'
from tldw_Server_API.app.main import app
schema = app.openapi()
paths = schema["paths"]
required = [
    "/api/v1/scheduled-tasks/capabilities",
    "/api/v1/scheduled-tasks/previews",
    "/api/v1/scheduled-tasks/definitions",
]
missing = [path for path in required if path not in paths]
if missing:
    raise SystemExit(f"missing OpenAPI paths: {missing}")
print("scheduled task automation OpenAPI paths present")
PY
```

Expected: prints `scheduled task automation OpenAPI paths present`.

Result: PASS. Verification initially found the full app route policy did not enable the new scheduled-tasks control-plane route from default config, so `scheduled-tasks` was added to `[API-Routes].enable`. Rerunning OpenAPI import with only auth configured printed `scheduled task automation OpenAPI paths present`.

- [x] **Step 6: Update Backlog task and plan statuses**

Update `TASK-2351` notes with:

- commits completed;
- backend test command result;
- frontend test command result;
- Bandit result path;
- any known skips.

If the implementation required a doc update, add it under `modified_files`.

- [x] **Step 7: Final self-review**

Review:

```bash
git status --short
git diff --stat origin/dev...HEAD
git diff --check origin/dev...HEAD
```

Expected:

- only intended files changed;
- no whitespace errors;
- no unrelated dirty files.

Result: PASS. `git status --short` showed only intended final edits plus two unrelated untracked Watchlists template files. `git diff --check origin/dev...HEAD` and `git diff --check` returned no whitespace errors.

- [x] **Step 8: Commit verification/docs cleanup**

```bash
git add Docs/superpowers/plans/2026-06-09-scheduled-tasks-phase4b-backend-api-foundation-implementation-plan.md \
  'backlog/tasks/task-2351 - Implement-Scheduled-Tasks-Phase-4B-backend-API-foundation.md' \
  apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx \
  tldw_Server_API/Config_Files/config.txt
git commit -m "docs: record scheduled tasks phase 4b verification"
```

## Final Acceptance Checklist

- [x] `GET /api/v1/scheduled-tasks/capabilities` exposes Recurring Question and Agent Task with execution unavailable.
- [x] Preview records persist valid and invalid semantic validations.
- [x] Create/update require preview consumption and reject expired, consumed, wrong-user, mismatched, and stale previews.
- [x] Agent Task raw prompt text is not returned from preview, definition, list, or audit responses by default.
- [x] Agent Task raw prompt sentinel strings are absent from persisted preview, definition, and audit JSON by default.
- [x] Definitions can be listed, read, updated, paused, resumed, archived, and duplicated.
- [x] Duplicate creates a paused copy and deterministic source/copy audit events.
- [x] Duplicate cannot bypass `admin` or `security` disabled locks.
- [x] Optional idempotency works for replay and conflict across preview, create, update, duplicate, pause, resume, and archive, scoped by owner and route.
- [x] Error responses use the documented envelope including `correlation_id`.
- [x] `/api/v1/scheduled-tasks` projects `automation_definition` rows without breaking reminders or Watchlists.
- [x] WebUI can preview, create, inspect, edit, pause/resume, archive, and duplicate definitions using the API.
- [x] WebUI shows execution-unavailable states and does not create fake results or fake Home items.
- [x] Focused backend tests pass.
- [x] Focused frontend tests pass.
- [x] Bandit passes for touched backend scope.
- [x] No Jobs, Scheduler, RAG execution, ACP dispatch, notifications, or approval queue mutations were implemented.

## Plan Review

Reviewed with a `plan-document-reviewer`-style subagent on 2026-06-09.

Review pass 1 found four issues:

- Agent Task redaction coverage did not explicitly include definition/list/detail/audit responses or persisted JSON.
- Idempotency coverage did not cover every required 4B mutating route.
- Disabled duplicate guardrails lacked data-model support.
- Error envelope and mappings did not fully match the spec.

The plan was revised to add redaction sentinel tests, full idempotency route coverage, `disabled_lock_kind`/`disabled_reason`, duplicate lock tests, and complete error-envelope requirements including `correlation_id`.

Review pass 2 approved the revised plan and spec with no remaining blocking issues.
