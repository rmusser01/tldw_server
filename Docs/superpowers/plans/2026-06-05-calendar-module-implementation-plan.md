# Calendar Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the approved first-class Calendar module through a staged MVP and v1: local calendars/events/todos, calendar views, scheduled-task projections, practical frontend UI, then read-only CalDAV VEVENT import.

**Architecture:** Add a dedicated Calendar backend domain with a shared Calendar DB, Pydantic schemas, a thin FastAPI router, and focused services for permissions, recurrence expansion, links, annotations, reminders, and external sync. The frontend gets a dedicated `/calendar` route, typed API client, agenda/week views, item drawer, filters, sync settings, and locked provider-owned item treatment. CalDAV import remains read-only, user-scoped, Jobs-backed, capability-aware, and bounded by sync/query windows.

**Tech Stack:** FastAPI, Pydantic, SQLite via existing DB management patterns, AuthNZ/RBAC dependencies, APScheduler, Jobs `JobManager`, pytest/Hypothesis, Next.js/WebUI extension routes, React, Ant Design, TanStack Query, Vitest.

Calendar parsing dependencies already present in `pyproject.toml`: use `icalendar` for VEVENT parsing/serialization boundaries and `python-dateutil` for safe datetime parsing where the existing codebase already does so. Do not hand-roll `.ics` parsing.

---

## Scope Boundary

This plan implements the PRD's MVP plus v1 read-only CalDAV import. It intentionally does not implement outbound push, two-way sync, external VTODO import, Microsoft Graph, Proton, Google Calendar, month/day/free-busy UI, org-managed external provider accounts, resource calendars, or rich policy screens.

Implementation order:

1. Backend local calendar foundation.
2. Backend views, projections, annotations, links, and reminder handoff.
3. Frontend practical calendar workspace.
4. CalDAV account, discovery, sync, and provider-owned context.
5. Hardening, docs, and verification.

Reference spec: `Docs/superpowers/specs/2026-06-05-calendar-module-prd-design.md`

Backlog task: `TASK-516`

---

## Existing Anchors

Backend:

- Router registration: `tldw_Server_API/app/api/v1/router_groups/content.py`, `tldw_Server_API/app/api/v1/router_groups/minimal.py`
- Auth dependencies: `tldw_Server_API/app/api/v1/API_Deps/auth_deps.py`
- Permission constants: `tldw_Server_API/app/core/AuthNZ/permissions.py`
- Scheduled task API: `tldw_Server_API/app/api/v1/endpoints/scheduled_tasks_control_plane.py`
- Scheduled task schemas: `tldw_Server_API/app/api/v1/schemas/scheduled_tasks_control_plane_schemas.py`
- Scheduled task service: `tldw_Server_API/app/services/scheduled_tasks_control_plane_service.py`
- Reminder schemas: `tldw_Server_API/app/api/v1/schemas/reminders_schemas.py`
- Reminder scheduler: `tldw_Server_API/app/services/reminders_scheduler.py`
- Jobs manager: `tldw_Server_API/app/core/Jobs/manager.py`
- Startup poller pattern: `tldw_Server_API/app/services/startup_sidecar_owned_jobs_pollers.py`, `tldw_Server_API/app/services/startup_content_jobs_pollers.py`
- DB examples: `tldw_Server_API/app/core/DB_Management/Watchlists_DB.py`, `tldw_Server_API/app/core/DB_Management/Collections_DB.py`
- Secret helpers: `tldw_Server_API/app/core/AuthNZ/user_provider_secrets.py`, `tldw_Server_API/app/core/Security/crypto.py`, `tldw_Server_API/app/core/External_Sources/connectors_service.py`

Frontend:

- Existing scheduled tasks page: `apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTasksPage.tsx`
- Existing scheduled tasks service: `apps/packages/ui/src/services/scheduled-tasks-control-plane.ts`
- Existing scheduled tasks service tests: `apps/packages/ui/src/services/__tests__/scheduled-tasks-control-plane.test.ts`
- Option route wrapper pattern: `apps/tldw-frontend/extension/routes/option-scheduled-tasks.tsx`
- Next page wrapper pattern: `apps/tldw-frontend/pages/scheduled-tasks.tsx`
- Route registry: `apps/tldw-frontend/extension/routes/route-registry.tsx`
- Locale file: `apps/packages/ui/src/public/_locales/en/option.json`
- State/error component pattern: `apps/packages/ui/src/components/ui/state`

---

## File Structure

Create backend:

- `tldw_Server_API/app/core/DB_Management/Calendar_DB.py`
  - Owns Calendar DB schema, migrations, row dataclasses, CRUD, queries, tombstones, and sync metadata persistence.
- `tldw_Server_API/app/core/Calendar/__init__.py`
  - Package marker and public exports.
- `tldw_Server_API/app/core/Calendar/constants.py`
  - Calendar domain constants, max windows, source-owner values, role names, job domain/type names.
- `tldw_Server_API/app/core/Calendar/errors.py`
  - Domain exceptions mapped by the router.
- `tldw_Server_API/app/core/Calendar/permissions.py`
  - CalendarMembership role evaluation and AuthNZ/org-aware access checks.
- `tldw_Server_API/app/core/Calendar/recurrence.py`
  - Bounded recurrence normalization and expansion for local items.
- `tldw_Server_API/app/core/Calendar/view_service.py`
  - Agenda/week query orchestration and linked projection merging.
- `tldw_Server_API/app/core/Calendar/calendar_service.py`
  - Local calendar, item, annotation, link, copy, and reminder-handoff operations.
- `tldw_Server_API/app/core/Calendar/secret_store.py`
  - Minimal encrypted secret-ref adapter for CalDAV credentials using existing crypto/BYOK patterns.
- `tldw_Server_API/app/core/Calendar/providers/__init__.py`
  - Provider package marker.
- `tldw_Server_API/app/core/Calendar/providers/caldav.py`
  - CalDAV verification, discovery, capability probing, and VEVENT fetch adapter.
- `tldw_Server_API/app/core/Calendar/calendar_sync_worker.py`
  - Jobs-backed sync execution for one binding.
- `tldw_Server_API/app/api/v1/schemas/calendar_schemas.py`
  - Pydantic request/response schemas.
- `tldw_Server_API/app/api/v1/endpoints/calendar.py`
  - Thin API router.
- `tldw_Server_API/app/services/calendar_sync_scheduler.py`
  - APScheduler bridge that queues binding-scoped sync Jobs.
- `tldw_Server_API/app/services/shutdown_calendar_sync_worker.py`
  - Shutdown helper if a long-lived worker is started at app startup.

Create backend tests:

- `tldw_Server_API/tests/Calendar/unit/test_calendar_db.py`
- `tldw_Server_API/tests/Calendar/unit/test_calendar_permissions.py`
- `tldw_Server_API/tests/Calendar/unit/test_calendar_recurrence.py`
- `tldw_Server_API/tests/Calendar/unit/test_calendar_service.py`
- `tldw_Server_API/tests/Calendar/unit/test_calendar_secret_store.py`
- `tldw_Server_API/tests/Calendar/unit/test_calendar_caldav_provider.py`
- `tldw_Server_API/tests/Calendar/unit/test_calendar_sync_worker.py`
- `tldw_Server_API/tests/Calendar/integration/test_calendar_api.py`
- `tldw_Server_API/tests/Calendar/property/test_calendar_recurrence_properties.py`

Modify backend:

- `tldw_Server_API/app/core/AuthNZ/permissions.py`
  - Add `CALENDAR_READ`, `CALENDAR_WRITE`, `CALENDAR_SYNC`, and `CALENDAR_ADMIN` permission constants.
- `tldw_Server_API/app/api/v1/router_groups/content.py`
  - Register `calendar` router under `/api/v1/calendar`.
- `tldw_Server_API/app/api/v1/router_groups/minimal.py`
  - Register `calendar` router for minimal/control-support contexts if consistent with scheduled tasks.
- `tldw_Server_API/app/services/startup_sidecar_owned_jobs_pollers.py` or `tldw_Server_API/app/services/startup_content_jobs_pollers.py`
  - Start calendar sync worker/scheduler only behind explicit env flags.
- `tldw_Server_API/app/services/shutdown_resource_cleanup.py`
  - Include calendar scheduler/worker shutdown if startup adds long-lived handles.

Create frontend:

- `apps/packages/ui/src/services/calendar.ts`
  - Typed Calendar API client.
- `apps/packages/ui/src/services/__tests__/calendar.test.ts`
  - API contract tests for paths, payloads, and unsafe mutation guards.
- `apps/packages/ui/src/components/Option/Calendar/CalendarPage.tsx`
  - Calendar workspace shell.
- `apps/packages/ui/src/components/Option/Calendar/CalendarAgenda.tsx`
  - Agenda list.
- `apps/packages/ui/src/components/Option/Calendar/CalendarWeekView.tsx`
  - Week grid.
- `apps/packages/ui/src/components/Option/Calendar/CalendarFilterRail.tsx`
  - Calendar/source filters.
- `apps/packages/ui/src/components/Option/Calendar/CalendarItemDrawer.tsx`
  - Create/edit/detail drawer.
- `apps/packages/ui/src/components/Option/Calendar/CalendarSyncSettings.tsx`
  - CalDAV account/binding settings and sync status.
- `apps/packages/ui/src/components/Option/Calendar/CalendarOwnershipBadge.tsx`
  - Local/org/provider/linked ownership labels.
- `apps/packages/ui/src/components/Option/Calendar/__tests__/CalendarPage.test.tsx`
- `apps/packages/ui/src/components/Option/Calendar/__tests__/CalendarItemDrawer.test.tsx`
- `apps/packages/ui/src/components/Option/Calendar/__tests__/CalendarSyncSettings.test.tsx`
- `apps/tldw-frontend/extension/routes/option-calendar.tsx`
- `apps/tldw-frontend/pages/calendar.tsx`

Modify frontend:

- `apps/tldw-frontend/extension/routes/route-registry.tsx`
  - Add lazy `OptionCalendar`, route `/calendar`, nav label token, and `CalendarDays` or closest lucide icon.
- `apps/packages/ui/src/public/_locales/en/option.json`
  - Add calendar nav and core UI strings.
- `apps/packages/ui/src/services/tldw/client-ownership.ts`
  - Add Calendar API methods if this guard expects every service client method to be listed.

Do not create:

- A second Scheduled Tasks backend.
- Provider-owned event edit endpoints.
- External VTODO import.
- Shared org CalDAV account support.
- Month/day/free-busy UI.

---

## Cross-Cutting Defaults

- Default local timezone: use request payload timezone when provided, else server-configured default, else `UTC`.
- Query window limit: reject agenda/week requests beyond 370 days.
- Sync lookback/lookahead defaults: 90 days back, 365 days forward.
- Recurrence occurrence cap per query: 2,000 expanded instances.
- Remote tombstone retention default: 90 days unless local annotations/links exist.
- Calendar Jobs domain: `calendar`.
- Calendar sync job type: `calendar_sync`.
- Calendar sync idempotency key: `calendar_sync:{binding_id}:{window_start}:{window_end}:{reason}`.
- Calendar sync queue env var: `CALENDAR_SYNC_JOBS_QUEUE`, default `default`.
- Startup flags: `CALENDAR_SYNC_SCHEDULER_ENABLED=false` and `CALENDAR_SYNC_WORKER_ENABLED=false` by default.

---

### Task 1: Backend Calendar Schema, Repository, And Constants

**Files:**

- Create: `tldw_Server_API/app/core/DB_Management/Calendar_DB.py`
- Create: `tldw_Server_API/app/core/Calendar/__init__.py`
- Create: `tldw_Server_API/app/core/Calendar/constants.py`
- Create: `tldw_Server_API/app/core/Calendar/errors.py`
- Test: `tldw_Server_API/tests/Calendar/unit/test_calendar_db.py`

- [ ] **Step 1: Write failing DB schema tests**

  Add tests that create a temporary Calendar DB, call `ensure_schema()`, and assert these behaviors:

  ```python
  def test_calendar_db_creates_personal_calendar(tmp_path):
      db = CalendarDatabase(db_path=tmp_path / "calendar.db")
      calendar = db.create_calendar(
          tenant_id="default",
          owner_user_id=1,
          org_id=None,
          name="Research",
          timezone="America/Los_Angeles",
          color="#2563eb",
      )
      assert calendar.name == "Research"
      assert calendar.owner_user_id == 1
      assert calendar.archived_at is None
  ```

  Also cover:

  - owner membership row is created automatically;
  - `create_item()` rejects provider-owned local creates;
  - `remote_deleted_at` tombstones hide provider items from normal list queries;
  - account rows do not expose credential payloads, only `secret_ref`;
  - binding rows store `lookback_days`, `lookahead_days`, and `provider_capabilities_json`.

- [ ] **Step 2: Run DB tests and verify failure**

  Run:

  ```bash
  source .venv/bin/activate
  python -m pytest tldw_Server_API/tests/Calendar/unit/test_calendar_db.py -v
  ```

  Expected: FAIL because `CalendarDatabase` does not exist.

- [ ] **Step 3: Add constants and errors**

  Implement:

  ```python
  CALENDAR_DOMAIN = "calendar"
  CALENDAR_SYNC_JOB_TYPE = "calendar_sync"
  DEFAULT_SYNC_LOOKBACK_DAYS = 90
  DEFAULT_SYNC_LOOKAHEAD_DAYS = 365
  MAX_QUERY_WINDOW_DAYS = 370
  MAX_EXPANDED_OCCURRENCES = 2000
  ```

  Add domain exceptions: `CalendarNotFound`, `CalendarPermissionDenied`, `CalendarValidationError`, `CalendarItemNotFound`, `CalendarReadOnlyError`, `CalendarSyncError`.

- [ ] **Step 4: Implement `CalendarDatabase` schema and rows**

  Follow `Watchlists_DB.py` style: dataclasses for rows, explicit SQL, context-managed connections, `ensure_schema()`, SQLite default path `Databases/calendar.db`, and no raw SQL outside the DB module.

  Required tables:

  - `calendars`
  - `calendar_memberships`
  - `calendar_items`
  - `calendar_recurrences`
  - `calendar_annotations`
  - `calendar_links`
  - `external_calendar_accounts`
  - `external_calendar_bindings`
  - `calendar_sync_events`
  - `calendar_external_account_secrets` or an equivalent encrypted secret table managed only through `CalendarSecretStore`

  Add indexes for:

  - `calendars(tenant_id, owner_user_id, org_id, archived_at)`
  - `calendar_memberships(calendar_id, principal_type, principal_id)`
  - `calendar_items(calendar_id, start_at, end_at, due_at, deleted_at, remote_deleted_at)`
  - `calendar_items(external_binding_id, source_uid)`
  - `calendar_links(calendar_item_id, target_type, target_id)`
  - `external_calendar_bindings(account_id, sync_enabled)`

- [ ] **Step 5: Add CRUD/query methods used by later tasks**

  Implement at least:

  - `create_calendar()`
  - `update_calendar()`
  - `get_calendar()`
  - `list_calendars_for_user()`
  - `archive_calendar()`
  - `create_membership()`
  - `list_memberships()`
  - `remove_membership()`
  - `create_item()`
  - `get_item()`
  - `update_item()`
  - `soft_delete_item()`
  - `list_items_window()`
  - `upsert_provider_item()`
  - `mark_provider_item_remote_deleted()`
  - `delete_remote_tombstones_eligible_for_cleanup()`
  - `create_annotation()`
  - `update_annotation()`
  - `delete_annotation()`
  - `list_annotations()`
  - `create_link()`
  - `delete_link()`
  - `list_links()`
  - `create_secret_ref()`
  - `resolve_secret_ref()`
  - `delete_secret_ref()`
  - `create_external_account()`
  - `get_external_account()`
  - `list_external_accounts_for_user()`
  - `delete_external_account()`
  - `revoke_external_account()`
  - `create_external_binding()`
  - `get_external_binding()`
  - `list_external_bindings_for_account()`
  - `list_sync_enabled_bindings_due_for_scan()`
  - `update_external_binding()`
  - `disable_external_binding()`
  - `delete_external_binding()`
  - `update_binding_sync_state()`
  - `record_sync_event()`
  - `list_sync_events()`

  Account delete semantics:

  - revoke/delete secret material first;
  - disable bindings;
  - mark provider-owned imported items remote-deleted or delete them only when the endpoint explicitly requests destructive cleanup;
  - preserve annotations/links unless the user confirms destructive imported-record deletion;
  - never delete copied tldw-owned items.

- [ ] **Step 6: Run DB tests and commit**

  Run:

  ```bash
  source .venv/bin/activate
  python -m pytest tldw_Server_API/tests/Calendar/unit/test_calendar_db.py -v
  ```

  Expected: PASS.

  Commit:

  ```bash
  git add tldw_Server_API/app/core/DB_Management/Calendar_DB.py \
          tldw_Server_API/app/core/Calendar/__init__.py \
          tldw_Server_API/app/core/Calendar/constants.py \
          tldw_Server_API/app/core/Calendar/errors.py \
          tldw_Server_API/tests/Calendar/unit/test_calendar_db.py
  git commit -m "feat(calendar): add calendar database foundation"
  ```

---

### Task 2: Calendar Permissions And Local Domain Service

**Files:**

- Create: `tldw_Server_API/app/core/Calendar/permissions.py`
- Create: `tldw_Server_API/app/core/Calendar/calendar_service.py`
- Modify: `tldw_Server_API/app/core/DB_Management/Calendar_DB.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/permissions.py`
- Test: `tldw_Server_API/tests/Calendar/unit/test_calendar_permissions.py`
- Test: `tldw_Server_API/tests/Calendar/unit/test_calendar_service.py`

- [ ] **Step 1: Write failing permission tests**

  Cover:

  ```python
  def test_viewer_cannot_edit_local_item(calendar_db):
      service = CalendarService(db=calendar_db)
      calendar = service.create_calendar(actor_user_id=1, name="Shared", timezone="UTC")
      service.add_membership(actor_user_id=1, calendar_id=calendar.id, principal_type="user", principal_id="2", role="viewer")
      with pytest.raises(CalendarPermissionDenied):
          service.create_item(actor_user_id=2, calendar_id=calendar.id, kind="event", title="Nope", start_at="2026-06-05T10:00:00Z")
  ```

  Also cover:

  - owner can manage membership;
  - org-role membership grants access only through existing AuthNZ/org role resolution;
  - editor can create/edit local items;
  - commenter can add own annotations but cannot edit provider fields;
  - provider-owned item edits raise `CalendarReadOnlyError`;
  - copied provider item becomes `source_owner="tldw"` and independent.

- [ ] **Step 2: Run permission tests and verify failure**

  Run:

  ```bash
  source .venv/bin/activate
  python -m pytest tldw_Server_API/tests/Calendar/unit/test_calendar_permissions.py \
                   tldw_Server_API/tests/Calendar/unit/test_calendar_service.py -v
  ```

  Expected: FAIL because services do not exist.

- [ ] **Step 3: Add permission constants**

  In `tldw_Server_API/app/core/AuthNZ/permissions.py`, add:

  ```python
  CALENDAR_READ = "calendar.read"
  CALENDAR_WRITE = "calendar.write"
  CALENDAR_SYNC = "calendar.sync"
  CALENDAR_ADMIN = "calendar.admin"
  ```

  Use existing permission naming and export patterns in the file.

- [ ] **Step 4: Implement role checks**

  `permissions.py` should expose:

  - `CalendarRole = Literal["owner", "editor", "commenter", "viewer"]`
  - `can_read_calendar()`
  - `can_write_items()`
  - `can_comment()`
  - `can_manage_calendar()`
  - `assert_calendar_access()`

  Keep linked-object authorization out of this file; it should return link metadata and let the owning domain enforce its own read permissions.

- [ ] **Step 5: Implement local service methods**

  `CalendarService` should wrap DB operations and enforce permissions. Implement:

  - calendar create/list/update/archive;
  - membership add/list/remove;
  - local event/todo create/update/delete;
  - provider-owned read-only guard;
  - annotation create/update/delete;
  - local tag update through annotation overlay;
  - link create/delete/list;
  - copy provider item into local calendar.

- [ ] **Step 6: Run tests and commit**

  Run:

  ```bash
  source .venv/bin/activate
  python -m pytest tldw_Server_API/tests/Calendar/unit/test_calendar_permissions.py \
                   tldw_Server_API/tests/Calendar/unit/test_calendar_service.py -v
  ```

  Expected: PASS.

  Commit:

  ```bash
  git add tldw_Server_API/app/core/AuthNZ/permissions.py \
          tldw_Server_API/app/core/DB_Management/Calendar_DB.py \
          tldw_Server_API/app/core/Calendar/permissions.py \
          tldw_Server_API/app/core/Calendar/calendar_service.py \
          tldw_Server_API/tests/Calendar/unit/test_calendar_permissions.py \
          tldw_Server_API/tests/Calendar/unit/test_calendar_service.py
  git commit -m "feat(calendar): add local calendar service and permissions"
  ```

---

### Task 3: Recurrence, View Expansion, And Scheduled-Task Projections

**Files:**

- Create: `tldw_Server_API/app/core/Calendar/recurrence.py`
- Create: `tldw_Server_API/app/core/Calendar/view_service.py`
- Test: `tldw_Server_API/tests/Calendar/unit/test_calendar_recurrence.py`
- Test: `tldw_Server_API/tests/Calendar/property/test_calendar_recurrence_properties.py`
- Test: `tldw_Server_API/tests/Calendar/unit/test_calendar_service.py`

- [ ] **Step 1: Write failing recurrence tests**

  Cover:

  - daily recurrence respects `count`;
  - weekly recurrence respects weekday list;
  - monthly-by-date skips impossible dates rather than crashing;
  - `until` bounds occurrences;
  - all-day recurrence remains date-stable across DST;
  - expansion rejects query windows over `MAX_QUERY_WINDOW_DAYS`;
  - expansion stops at `MAX_EXPANDED_OCCURRENCES`.

- [ ] **Step 2: Write failing projection tests**

  Stub `ScheduledTasksControlPlaneService.list_tasks()` and assert `CalendarViewService.agenda()` returns read-only linked projections with:

  ```python
  assert item.source_owner == "linked_projection"
  assert item.read_only_reason == "linked_projection"
  assert item.link.target_type == "scheduled_task"
  ```

- [ ] **Step 3: Run tests and verify failure**

  Run:

  ```bash
  source .venv/bin/activate
  python -m pytest tldw_Server_API/tests/Calendar/unit/test_calendar_recurrence.py \
                   tldw_Server_API/tests/Calendar/property/test_calendar_recurrence_properties.py \
                   tldw_Server_API/tests/Calendar/unit/test_calendar_service.py -v
  ```

  Expected: FAIL because recurrence/view services are incomplete.

- [ ] **Step 4: Implement recurrence model and expansion**

  Support local recurrence:

  - daily;
  - weekly by weekday;
  - monthly by date;
  - interval;
  - count;
  - until;
  - timezone-aware all-day handling.

  Preserve raw provider recurrence in DB but do not author complex provider rules.

- [ ] **Step 5: Implement `CalendarViewService`**

  Methods:

  - `agenda(actor_user_id, start_at, end_at, filters)`;
  - `week(actor_user_id, week_start, timezone, filters)`;
  - `expand_items_window()`;
  - `load_scheduled_task_projections()`.

  Enforce:

  - explicit query windows;
  - max range;
  - membership permissions;
  - provider tombstones hidden by default;
  - linked projections read-only.

- [ ] **Step 6: Run tests and commit**

  Run:

  ```bash
  source .venv/bin/activate
  python -m pytest tldw_Server_API/tests/Calendar/unit/test_calendar_recurrence.py \
                   tldw_Server_API/tests/Calendar/property/test_calendar_recurrence_properties.py \
                   tldw_Server_API/tests/Calendar/unit/test_calendar_service.py -v
  ```

  Expected: PASS.

  Commit:

  ```bash
  git add tldw_Server_API/app/core/Calendar/recurrence.py \
          tldw_Server_API/app/core/Calendar/view_service.py \
          tldw_Server_API/tests/Calendar/unit/test_calendar_recurrence.py \
          tldw_Server_API/tests/Calendar/property/test_calendar_recurrence_properties.py \
          tldw_Server_API/tests/Calendar/unit/test_calendar_service.py
  git commit -m "feat(calendar): add bounded calendar views and recurrence"
  ```

---

### Task 4: Calendar API Schemas, Router, And Reminder Handoff

**Files:**

- Create: `tldw_Server_API/app/api/v1/schemas/calendar_schemas.py`
- Create: `tldw_Server_API/app/api/v1/endpoints/calendar.py`
- Modify: `tldw_Server_API/app/api/v1/router_groups/content.py`
- Modify: `tldw_Server_API/app/api/v1/router_groups/minimal.py`
- Test: `tldw_Server_API/tests/Calendar/integration/test_calendar_api.py`

- [ ] **Step 1: Write failing API integration tests**

  Build a FastAPI test app with the calendar router and dependency overrides. Cover:

  - `POST /api/v1/calendar/calendars` creates a local calendar;
  - `GET /api/v1/calendar/calendars` lists only visible calendars;
  - calendar membership endpoints add/list/remove viewer/editor/commenter roles and enforce owner-only management;
  - `POST /api/v1/calendar/items` creates event/todo;
  - `PATCH /api/v1/calendar/items/{item_id}` rejects provider-owned edits;
  - `GET /api/v1/calendar/views/agenda?start_at=...&end_at=...` requires bounded range;
  - `POST /api/v1/calendar/items/{item_id}/annotations`;
  - `POST /api/v1/calendar/items/{item_id}/links`;
  - `POST /api/v1/calendar/items/{item_id}/copy`;
  - `POST /api/v1/calendar/reminders` calls existing reminder primitives and returns a linked projection.
  - personal provider-owned imports are not returned through shared org calendar queries unless copied into a tldw-owned org item.

- [ ] **Step 2: Run API tests and verify failure**

  Run:

  ```bash
  source .venv/bin/activate
  python -m pytest tldw_Server_API/tests/Calendar/integration/test_calendar_api.py -v
  ```

  Expected: FAIL because router/schemas do not exist.

- [ ] **Step 3: Implement Pydantic schemas**

  Include request/response models for:

  - calendars;
  - memberships;
  - items;
  - recurrence;
  - annotations;
  - links;
  - views;
  - reminder create handoff;
  - external account/binding/sync placeholders used by later tasks.

  Validate:

  - `event` requires `start_at`;
  - `todo` requires `due_at` or `start_at`;
  - provider-owned mutation payloads are never accepted on local item endpoints;
  - view range is explicit and bounded.

- [ ] **Step 4: Implement thin router**

  Router prefix inside file should be:

  ```python
  router = APIRouter(prefix="/calendar", tags=["calendar"])
  ```

  Use:

  - `get_request_user`;
  - `RequirePermission(CALENDAR_READ)` for reads;
  - `RequirePermission(CALENDAR_WRITE)` for local writes;
  - `RequirePermission(CALENDAR_SYNC)` for external sync/account operations;
  - `rbac_rate_limit()`.

  Map domain exceptions to stable HTTP errors, for example `CalendarReadOnlyError` -> `409 item_read_only`.

- [ ] **Step 5: Register router**

  Add `ImportedRouterSpec` entries in content and minimal router groups:

  - import path `tldw_Server_API.app.api.v1.endpoints.calendar`;
  - prefix `f"{API_V1_PREFIX}"`;
  - tags `("calendar",)`;
  - route key `"calendar"`;
  - `default_stable=False` while the module is new.

- [ ] **Step 6: Run API tests and commit**

  Run:

  ```bash
  source .venv/bin/activate
  python -m pytest tldw_Server_API/tests/Calendar/integration/test_calendar_api.py -v
  python -m pytest tldw_Server_API/tests/Config/test_openapi_config_jobs.py -v
  ```

  Expected: PASS.

  Commit:

  ```bash
  git add tldw_Server_API/app/api/v1/schemas/calendar_schemas.py \
          tldw_Server_API/app/api/v1/endpoints/calendar.py \
          tldw_Server_API/app/api/v1/router_groups/content.py \
          tldw_Server_API/app/api/v1/router_groups/minimal.py \
          tldw_Server_API/tests/Calendar/integration/test_calendar_api.py
  git commit -m "feat(calendar): expose local calendar API"
  ```

---

### Task 5: Frontend API Client And Route Scaffold

**Files:**

- Create: `apps/packages/ui/src/services/calendar.ts`
- Create: `apps/packages/ui/src/services/__tests__/calendar.test.ts`
- Create: `apps/tldw-frontend/extension/routes/option-calendar.tsx`
- Create: `apps/tldw-frontend/pages/calendar.tsx`
- Modify: `apps/tldw-frontend/extension/routes/route-registry.tsx`
- Modify: `apps/packages/ui/src/public/_locales/en/option.json`
- Modify: `apps/packages/ui/src/services/tldw/client-ownership.ts` if required by ownership guard tests.

- [ ] **Step 1: Write failing frontend service tests**

  Mock `bgRequest` like `scheduled-tasks-control-plane.test.ts`. Cover:

  - `listCalendars()` calls `GET /api/v1/calendar/calendars`;
  - `createCalendarItem()` posts typed payload;
  - `updateCalendarItem()` rejects provider-owned mutation attempts when passed `source_owner: "provider"`;
  - `getCalendarAgenda()` URL-encodes query params and requires `start_at`/`end_at`;
  - external account functions never send secret values except to create/verify endpoints.

- [ ] **Step 2: Run service tests and verify failure**

  Run:

  ```bash
  bunx vitest run apps/packages/ui/src/services/__tests__/calendar.test.ts
  ```

  Expected: FAIL because service file does not exist.

- [ ] **Step 3: Implement typed frontend service**

  `calendar.ts` should export TypeScript types matching backend schema names and functions:

  - `listCalendars()`;
  - `createCalendar()`;
  - `createCalendarItem()`;
  - `updateCalendarItem()`;
  - `deleteCalendarItem()`;
  - `getCalendarAgenda()`;
  - `getCalendarWeek()`;
  - `createCalendarAnnotation()`;
  - `createCalendarLink()`;
  - `copyCalendarItemIntoTldw()`;
  - `listCalDavAccounts()`;
  - `createCalDavAccount()`;
  - `verifyCalDavAccount()`;
  - `revokeCalDavAccount()`;
  - `deleteCalDavAccount()`;
  - `discoverExternalCalendars()`;
  - `createExternalCalendarBinding()`;
  - `listExternalCalendarBindings()`;
  - `triggerCalendarSync()`.

  Use `bgRequest` and `toAllowedPath` consistently.

- [ ] **Step 4: Add route wrappers**

  Create `option-calendar.tsx` matching scheduled tasks:

  ```tsx
  import OptionLayout from "@web/components/layout/WebLayout"
  import { RouteErrorBoundary } from "@/components/Common/RouteErrorBoundary"
  import { CalendarPage } from "@/components/Option/Calendar/CalendarPage"

  const OptionCalendar = () => (
    <RouteErrorBoundary routeId="calendar" routeLabel="Calendar">
      <OptionLayout>
        <CalendarPage />
      </OptionLayout>
    </RouteErrorBoundary>
  )

  export default OptionCalendar
  ```

  Create Next wrapper:

  ```tsx
  import dynamic from "next/dynamic"

  export default dynamic(() => import("@/routes/option-calendar"), { ssr: false })
  ```

- [ ] **Step 5: Register navigation**

  In `route-registry.tsx`:

  - import a calendar icon from `lucide-react`;
  - add `const OptionCalendar = lazy(() => import("./option-calendar"))`;
  - add `/calendar` in the workspace or knowledge group near scheduled tasks;
  - use locale key `option:calendar.nav`.

  Add locale entries in `option.json`, keeping JSON valid.

- [ ] **Step 6: Run frontend tests and commit**

  Run:

  ```bash
  bunx vitest run apps/packages/ui/src/services/__tests__/calendar.test.ts
  ```

  Expected: PASS.

  Commit:

  ```bash
  git add apps/packages/ui/src/services/calendar.ts \
          apps/packages/ui/src/services/__tests__/calendar.test.ts \
          apps/tldw-frontend/extension/routes/option-calendar.tsx \
          apps/tldw-frontend/pages/calendar.tsx \
          apps/tldw-frontend/extension/routes/route-registry.tsx \
          apps/packages/ui/src/public/_locales/en/option.json \
          apps/packages/ui/src/services/tldw/client-ownership.ts
  git commit -m "feat(calendar): add frontend calendar route and API client"
  ```

---

### Task 6: Practical Calendar UI MVP

**Files:**

- Create: `apps/packages/ui/src/components/Option/Calendar/CalendarPage.tsx`
- Create: `apps/packages/ui/src/components/Option/Calendar/CalendarAgenda.tsx`
- Create: `apps/packages/ui/src/components/Option/Calendar/CalendarWeekView.tsx`
- Create: `apps/packages/ui/src/components/Option/Calendar/CalendarFilterRail.tsx`
- Create: `apps/packages/ui/src/components/Option/Calendar/CalendarItemDrawer.tsx`
- Create: `apps/packages/ui/src/components/Option/Calendar/CalendarOwnershipBadge.tsx`
- Create: `apps/packages/ui/src/components/Option/Calendar/__tests__/CalendarPage.test.tsx`
- Create: `apps/packages/ui/src/components/Option/Calendar/__tests__/CalendarItemDrawer.test.tsx`

- [ ] **Step 1: Write failing UI tests**

  Test:

  - page loads calendars and agenda;
  - unsupported backend path shows `RecoveryCallout`;
  - agenda renders local event, local todo, provider-owned event, and linked projection with distinct ownership labels;
  - provider-owned item drawer disables provider field editing and shows copy action;
  - linked projection opens source/manage URL and does not show local edit controls;
  - local event/todo create form submits through service.

- [ ] **Step 2: Run UI tests and verify failure**

  Run:

  ```bash
  bunx vitest run apps/packages/ui/src/components/Option/Calendar/__tests__/CalendarPage.test.tsx \
                   apps/packages/ui/src/components/Option/Calendar/__tests__/CalendarItemDrawer.test.tsx
  ```

  Expected: FAIL because UI components do not exist.

- [ ] **Step 3: Implement `CalendarPage` shell**

  Mirror the scheduled tasks page pattern:

  - use `useCanonicalConnectionConfig()`;
  - probe `/openapi.json` for `/api/v1/calendar/calendars`;
  - use TanStack Query for calendars and agenda/week data;
  - show `RecoveryCallout` for unsupported, load errors, and partial sync states;
  - avoid landing-page copy; show the actual agenda/week workspace immediately.

- [ ] **Step 4: Implement agenda and week views**

  Agenda:

  - compact, scannable list grouped by day;
  - show title, time/due, calendar color, ownership badge, and source/link hint.

  Week:

  - stable seven-column grid;
  - all-day row;
  - time-grid placement;
  - no drag/drop in this slice;
  - provider-owned and linked projections visually distinct but not loud.

- [ ] **Step 5: Implement drawer and filters**

  Drawer:

  - create/edit local `event` and `todo`;
  - local tags and plain-text annotation editor;
  - links list;
  - read-only provider fields;
  - copy-into-tldw action;
  - narrow reminder/deferred task action only for one-time reminders.

  Filter rail:

  - calendar/source toggles;
  - local/org/provider/linked filters;
  - item kind filters.

- [ ] **Step 6: Run UI tests and commit**

  Run:

  ```bash
  bunx vitest run apps/packages/ui/src/components/Option/Calendar/__tests__/CalendarPage.test.tsx \
                   apps/packages/ui/src/components/Option/Calendar/__tests__/CalendarItemDrawer.test.tsx
  ```

  Expected: PASS.

  Commit:

  ```bash
  git add apps/packages/ui/src/components/Option/Calendar
  git commit -m "feat(calendar): add practical agenda and week UI"
  ```

---

### Task 7: CalDAV Account, Secret Store, Discovery, And API

**Files:**

- Create: `tldw_Server_API/app/core/Calendar/secret_store.py`
- Create: `tldw_Server_API/app/core/Calendar/providers/__init__.py`
- Create: `tldw_Server_API/app/core/Calendar/providers/caldav.py`
- Modify: `tldw_Server_API/app/core/Calendar/calendar_service.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/calendar_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/calendar.py`
- Test: `tldw_Server_API/tests/Calendar/unit/test_calendar_secret_store.py`
- Test: `tldw_Server_API/tests/Calendar/unit/test_calendar_caldav_provider.py`
- Test: `tldw_Server_API/tests/Calendar/integration/test_calendar_api.py`

- [ ] **Step 1: Write failing secret-store tests**

  Cover:

  - creating a CalDAV account stores `secret_ref`, not password;
  - resolving a secret returns username/password only to service code;
  - deleting/revoking clears secret material;
  - API response redacts all secret fields;
  - missing encryption key prevents account creation with a clear error.

- [ ] **Step 2: Write failing CalDAV provider tests**

  Use mocked HTTP responses, not a real provider. Cover:

  - `verify_account()` rejects non-http(s) URLs;
  - discovery records capabilities;
  - unsupported sync-token falls back to bounded polling capability;
  - VEVENT import ignores VTODO;
  - VEVENT parsing uses `icalendar` and datetime parsing uses `python-dateutil`/timezone-aware helpers instead of ad hoc string parsing;
  - provider metadata is scrubbed of auth headers.

- [ ] **Step 3: Run tests and verify failure**

  Run:

  ```bash
  source .venv/bin/activate
  python -m pytest tldw_Server_API/tests/Calendar/unit/test_calendar_secret_store.py \
                   tldw_Server_API/tests/Calendar/unit/test_calendar_caldav_provider.py \
                   tldw_Server_API/tests/Calendar/integration/test_calendar_api.py -v
  ```

  Expected: FAIL because provider and secret-store code does not exist.

- [ ] **Step 4: Implement secret store**

  Use existing crypto/BYOK helper patterns. Do not put raw credentials in:

  - `external_calendar_accounts`;
  - API responses;
  - Jobs payloads;
  - logs.

  `CalendarSecretStore` should expose:

  - `create_secret(owner_user_id, provider, payload) -> secret_ref`;
  - `resolve_secret(owner_user_id, secret_ref) -> dict`;
  - `delete_secret(owner_user_id, secret_ref) -> bool`.

- [ ] **Step 5: Implement CalDAV adapter**

  Keep it dependency-light. If adding a CalDAV library is unavoidable, document why and add tests around adapter boundaries. Prefer HTTP-level adapter functions first:

  - URL validation and SSRF guard;
  - credential verification;
  - principal/home-set/calendar discovery;
  - capability extraction;
  - bounded VEVENT fetch;
  - ETag/CTag metadata capture;
  - VTODO ignore behavior.

- [ ] **Step 6: Add external account/binding endpoints**

  Add endpoints under `/api/v1/calendar/external/...`:

  - create/verify/revoke/delete account;
  - discover remote calendars;
  - bind/unbind/enable/disable/list calendar bindings;
  - read sync status.
  - list recent sync events/errors for a binding.

  Require `CALENDAR_SYNC`.

- [ ] **Step 7: Run tests and commit**

  Run:

  ```bash
  source .venv/bin/activate
  python -m pytest tldw_Server_API/tests/Calendar/unit/test_calendar_secret_store.py \
                   tldw_Server_API/tests/Calendar/unit/test_calendar_caldav_provider.py \
                   tldw_Server_API/tests/Calendar/integration/test_calendar_api.py -v
  ```

  Expected: PASS.

  Commit:

  ```bash
  git add tldw_Server_API/app/core/Calendar/secret_store.py \
          tldw_Server_API/app/core/Calendar/providers \
          tldw_Server_API/app/core/Calendar/calendar_service.py \
          tldw_Server_API/app/api/v1/schemas/calendar_schemas.py \
          tldw_Server_API/app/api/v1/endpoints/calendar.py \
          tldw_Server_API/tests/Calendar/unit/test_calendar_secret_store.py \
          tldw_Server_API/tests/Calendar/unit/test_calendar_caldav_provider.py \
          tldw_Server_API/tests/Calendar/integration/test_calendar_api.py
  git commit -m "feat(calendar): add read-only CalDAV account discovery"
  ```

---

### Task 8: Jobs-Backed Calendar Sync Worker And Scheduler

**Files:**

- Create: `tldw_Server_API/app/core/Calendar/calendar_sync_worker.py`
- Create: `tldw_Server_API/app/services/calendar_sync_scheduler.py`
- Create: `tldw_Server_API/app/services/shutdown_calendar_sync_worker.py`
- Modify: `tldw_Server_API/app/core/Calendar/calendar_service.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/calendar.py`
- Modify: `tldw_Server_API/app/services/startup_sidecar_owned_jobs_pollers.py` or `tldw_Server_API/app/services/startup_content_jobs_pollers.py`
- Modify: `tldw_Server_API/app/services/shutdown_resource_cleanup.py`
- Test: `tldw_Server_API/tests/Calendar/unit/test_calendar_sync_worker.py`
- Test: `tldw_Server_API/tests/Calendar/integration/test_calendar_api.py`

- [ ] **Step 1: Write failing sync worker tests**

  Cover:

  - manual sync creates Jobs `domain="calendar"`, `job_type="calendar_sync"`;
  - idempotency key uses binding and window;
  - payload contains binding/window/reason but no credentials;
  - worker resolves credentials at execution time;
  - provider update changes provider fields but preserves annotations/local tags/links;
  - provider delete creates tombstone;
  - provider-owned items synced from a personal external account remain private to that account owner;
  - sync failure updates binding error fields;
  - overlapping active sync for same binding is not queued.

- [ ] **Step 2: Run sync tests and verify failure**

  Run:

  ```bash
  source .venv/bin/activate
  python -m pytest tldw_Server_API/tests/Calendar/unit/test_calendar_sync_worker.py \
                   tldw_Server_API/tests/Calendar/integration/test_calendar_api.py -v
  ```

  Expected: FAIL because sync code is incomplete.

- [ ] **Step 3: Implement queue helper**

  Add service method:

  ```python
  queue_binding_sync(actor_user_id, binding_id, reason, window_start, window_end) -> CalendarSyncJobResponse
  ```

  It should:

  - check account owner;
  - enforce bounded window;
  - load binding/account through explicit DB lookup methods;
  - check for active binding job;
  - call `JobManager.create_job()`;
  - use stable idempotency key;
  - store job id/status in sync state.

- [ ] **Step 4: Implement worker**

  Worker loop should:

  - acquire `calendar` domain jobs;
  - load binding/account;
  - resolve secret from `secret_ref`;
  - fetch bounded VEVENT changes;
  - upsert provider-owned items;
  - mark tombstones;
  - preserve annotation/link rows;
  - update `last_sync_*`, counts, errors, and capability fallback diagnostics;
  - record a sync event row for queued/running/succeeded/failed transitions;
  - support stale binding scans through `list_sync_enabled_bindings_due_for_scan()`;
  - complete/fail Jobs with sanitized result/error.

- [ ] **Step 5: Implement APScheduler bridge**

  `calendar_sync_scheduler.py` should:

  - start only when `CALENDAR_SYNC_SCHEDULER_ENABLED=true`;
  - periodically scan enabled bindings;
  - queue sync jobs for stale bindings;
  - never execute provider work directly inside APScheduler.

- [ ] **Step 6: Wire optional startup/shutdown**

  Add startup handles only behind env flags. Avoid starting this worker by default in tests or development unless explicitly enabled.

- [ ] **Step 7: Run tests and commit**

  Run:

  ```bash
  source .venv/bin/activate
  python -m pytest tldw_Server_API/tests/Calendar/unit/test_calendar_sync_worker.py \
                   tldw_Server_API/tests/Calendar/integration/test_calendar_api.py -v
  ```

  Expected: PASS.

  Commit:

  ```bash
  git add tldw_Server_API/app/core/Calendar/calendar_sync_worker.py \
          tldw_Server_API/app/services/calendar_sync_scheduler.py \
          tldw_Server_API/app/services/shutdown_calendar_sync_worker.py \
          tldw_Server_API/app/core/Calendar/calendar_service.py \
          tldw_Server_API/app/api/v1/endpoints/calendar.py \
          tldw_Server_API/app/services/startup_sidecar_owned_jobs_pollers.py \
          tldw_Server_API/app/services/startup_content_jobs_pollers.py \
          tldw_Server_API/app/services/shutdown_resource_cleanup.py \
          tldw_Server_API/tests/Calendar/unit/test_calendar_sync_worker.py \
          tldw_Server_API/tests/Calendar/integration/test_calendar_api.py
  git commit -m "feat(calendar): add jobs-backed CalDAV sync"
  ```

---

### Task 9: Frontend Sync Settings And Provider-Owned UX

**Files:**

- Create: `apps/packages/ui/src/components/Option/Calendar/CalendarSyncSettings.tsx`
- Create: `apps/packages/ui/src/components/Option/Calendar/__tests__/CalendarSyncSettings.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/Calendar/CalendarPage.tsx`
- Modify: `apps/packages/ui/src/components/Option/Calendar/CalendarItemDrawer.tsx`
- Modify: `apps/packages/ui/src/services/calendar.ts`
- Modify: `apps/packages/ui/src/services/__tests__/calendar.test.ts`

- [ ] **Step 1: Write failing sync settings tests**

  Cover:

  - add CalDAV account form sends password only to create/verify endpoint;
  - discovery renders remote calendars and capability/fallback notes;
  - binding selected calendar stores lookback/lookahead;
  - sync now calls trigger endpoint;
  - stale/error states render recovery actions;
  - delete account asks for confirmation;
  - provider-owned event drawer shows read-only reason and copy action.

- [ ] **Step 2: Run tests and verify failure**

  Run:

  ```bash
  bunx vitest run apps/packages/ui/src/components/Option/Calendar/__tests__/CalendarSyncSettings.test.tsx \
                   apps/packages/ui/src/services/__tests__/calendar.test.ts
  ```

  Expected: FAIL until UI is implemented.

- [ ] **Step 3: Implement sync settings**

  Keep the UI practical:

  - account list;
  - add account drawer/form;
  - remote calendar discovery results;
  - binding toggles;
  - last success/failure;
  - next sync/stale indicator;
  - imported item count;
  - sync now;
  - revoke/delete.

- [ ] **Step 4: Tighten provider-owned item drawer**

  Provider-owned records:

  - cannot edit provider fields;
  - can edit local annotation/local tags/links;
  - can copy into tldw;
  - show provider/source metadata only after scrubbing.

- [ ] **Step 5: Run tests and commit**

  Run:

  ```bash
  bunx vitest run apps/packages/ui/src/components/Option/Calendar/__tests__/CalendarSyncSettings.test.tsx \
                   apps/packages/ui/src/services/__tests__/calendar.test.ts
  ```

  Expected: PASS.

  Commit:

  ```bash
  git add apps/packages/ui/src/components/Option/Calendar \
          apps/packages/ui/src/services/calendar.ts \
          apps/packages/ui/src/services/__tests__/calendar.test.ts
  git commit -m "feat(calendar): add CalDAV sync settings UI"
  ```

---

### Task 10: Documentation, Fastmail Smoke Path, And Final Verification

**Files:**

- Create: `Docs/Design/Calendar_Module.md`
- Create: `Docs/Development/Calendar_CalDAV_Smoke_Test.md`
- Modify: `Docs/superpowers/specs/2026-06-05-calendar-module-prd-design.md` only if implementation discovers a necessary spec correction.
- Modify: `Docs/superpowers/plans/2026-06-05-calendar-module-implementation-plan.md` task statuses during execution.
- Modify: `backlog/tasks/task-516 - Plan-first-class-calendar-module-implementation.md`

- [ ] **Step 1: Write docs**

  `Calendar_Module.md` should cover:

  - module boundary;
  - local vs provider-owned items;
  - Scheduled Tasks boundary;
  - DB location;
  - permission model;
  - sync model;
  - env flags.

  `Calendar_CalDAV_Smoke_Test.md` should cover:

  - Fastmail profile target;
  - required app password/credential setup without storing secrets in docs;
  - add account;
  - discover calendars;
  - bind one calendar;
  - sync now;
  - verify provider event appears read-only;
  - verify VTODO is ignored;
  - verify account revoke.

- [ ] **Step 2: Run backend calendar tests**

  Run:

  ```bash
  source .venv/bin/activate
  python -m pytest tldw_Server_API/tests/Calendar -v
  ```

  Expected: PASS.

- [ ] **Step 3: Run frontend calendar tests**

  Run:

  ```bash
  bunx vitest run apps/packages/ui/src/services/__tests__/calendar.test.ts \
                   apps/packages/ui/src/components/Option/Calendar/__tests__/CalendarPage.test.tsx \
                   apps/packages/ui/src/components/Option/Calendar/__tests__/CalendarItemDrawer.test.tsx \
                   apps/packages/ui/src/components/Option/Calendar/__tests__/CalendarSyncSettings.test.tsx
  ```

  Expected: PASS.

- [ ] **Step 4: Run route/openapi focused checks**

  Run:

  ```bash
  source .venv/bin/activate
  python -m pytest tldw_Server_API/tests/Config/test_openapi_config_jobs.py -v
  bunx vitest run apps/tldw-frontend/extension/__tests__/writing-playground-route-parity.guard.test.ts
  ```

  Expected: PASS or update the route parity test only if it intentionally tracks all option routes.

- [ ] **Step 5: Run Bandit on touched backend scope**

  Run:

  ```bash
  source .venv/bin/activate
  python -m bandit -r tldw_Server_API/app/core/Calendar \
                         tldw_Server_API/app/core/DB_Management/Calendar_DB.py \
                         tldw_Server_API/app/api/v1/endpoints/calendar.py \
                         tldw_Server_API/app/api/v1/schemas/calendar_schemas.py \
                         tldw_Server_API/app/services/calendar_sync_scheduler.py \
                         tldw_Server_API/app/core/Calendar/calendar_sync_worker.py \
                         -f json -o /tmp/bandit_calendar_module.json
  ```

  Expected: no new high/medium findings in touched Calendar code. Fix new findings before finalizing.

- [ ] **Step 6: Optional browser verification**

  If a dev server is already available or can be started:

  ```bash
  bun run dev
  ```

  Open `/calendar` in the Browser plugin and verify:

  - agenda and week views render;
  - create/edit local event works;
  - provider-owned fixture item is visibly read-only;
  - sync settings states do not overlap on mobile/desktop.

- [ ] **Step 7: Update Backlog and commit docs**

  Update `TASK-516` final summary with:

  - implementation scope completed;
  - tests run;
  - Bandit result path;
  - known skips.

  Commit:

  ```bash
  git add Docs/Design/Calendar_Module.md \
          Docs/Development/Calendar_CalDAV_Smoke_Test.md \
          Docs/superpowers/plans/2026-06-05-calendar-module-implementation-plan.md \
          "backlog/tasks/task-516 - Plan-first-class-calendar-module-implementation.md"
  git commit -m "docs(calendar): document calendar module rollout"
  ```

---

## Verification Matrix

Backend:

- `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Calendar -v`
- `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Config/test_openapi_config_jobs.py -v`
- `source .venv/bin/activate && python -m bandit -r <touched calendar backend paths> -f json -o /tmp/bandit_calendar_module.json`

Frontend:

- `bunx vitest run apps/packages/ui/src/services/__tests__/calendar.test.ts`
- `bunx vitest run apps/packages/ui/src/components/Option/Calendar/__tests__/CalendarPage.test.tsx`
- `bunx vitest run apps/packages/ui/src/components/Option/Calendar/__tests__/CalendarItemDrawer.test.tsx`
- `bunx vitest run apps/packages/ui/src/components/Option/Calendar/__tests__/CalendarSyncSettings.test.tsx`

Manual/browser:

- `/calendar` loads with unsupported-backend recovery on older servers.
- `/calendar` shows agenda/week and item drawer on a server with Calendar API.
- Provider-owned items are visibly locked and copyable.
- CalDAV settings show stale/error/revoke states without exposing secrets.

---

## Risk Controls

- Keep external sync disabled by default through env flags until local Calendar MVP is stable.
- Treat provider-owned records as read-only at DB, service, API, and UI layers.
- Never store raw credentials in Calendar account rows or Jobs payloads.
- Require bounded API windows and recurrence expansion caps before exposing agenda/week endpoints.
- Preserve annotations/links/local tags on provider refresh and remote delete.
- Do not add external VTODO import during this plan.
- Do not let personal external provider-owned imports appear in org calendar queries.

---

## Execution Handoff

Plan implementers should use `superpowers:subagent-driven-development` for parallel task execution where file ownership is disjoint, or `superpowers:executing-plans` for inline execution. Start with Task 1 and do not skip the failing-test step. Each task should end with its listed commit before moving to the next task unless the user explicitly asks for a different batching strategy.
