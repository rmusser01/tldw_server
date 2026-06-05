# Calendar Module PRD Design

Date: 2026-06-05
Status: Draft ready for spec review
Owner: Codex brainstorming session
Backlog: TASK-515

## Summary

Create a first-class Calendar module for tldw that stores local calendars, events, and lightweight dated todos/checkpoints while importing external calendars as read-only provider-owned context.

The module should be a calendar product, not only a Scheduled Tasks view. It should still integrate tightly with Scheduled Tasks, reminders, Watchlists, Workflows, Jobs, ACP schedules, notes, media, RAG, meetings, and other tldw objects through typed links and time-based projections.

V1 is intentionally read-only for external providers. Users can create and edit tldw-owned calendar items locally. External provider events are imported as locked records that can be annotated, linked, searched, copied into tldw, and displayed in agenda/week views, but cannot be edited or pushed back to providers. Outbound push, two-way sync, Microsoft, Proton, and richer policy controls are backlog items.

## Product Decision

Use a **Local Calendar Core with Read-Only External Overlays** model.

tldw owns the local writable calendar model:

- personal calendars;
- org calendars using existing AuthNZ/org RBAC;
- local events;
- lightweight dated todos/checkpoints;
- pragmatic recurrence;
- calendar annotations;
- links to other tldw objects;
- agenda/week calendar views.

External providers are read-only inputs in v1. Generic CalDAV is the first provider family, with configuration profiles for providers such as Fastmail when practical. Google Calendar should be the first non-CalDAV provider after the model and CalDAV adapter are stable. Microsoft and Proton remain named roadmap targets because they require provider-specific integration decisions that should not block first value.

This approach gives tldw a durable local calendar module while avoiding the complexity and user-risk of pretending v1 is full two-way calendar sync.

## Current Context

Source review found these implementation facts:

- `/scheduled-tasks` currently lists normalized reminder tasks and Watchlists jobs.
- Reminder tasks are editable from `/scheduled-tasks`; Watchlists jobs are shown as externally managed and deep-link to Watchlists.
- The current scheduled task create path is reminder-only and asks for raw ISO timestamps or cron.
- A recent Scheduled Tasks Automation Workbench PRD already positions `/scheduled-tasks` as the lifecycle, run history, result, and debugging hub for automations.
- Jobs provides durable background work with leasing, retries, quotas, metrics, queue controls, and admin operations.
- Scheduler provides internal task orchestration with idempotency, dependencies, leases, leader election, worker pools, and metadata ownership checks.
- Reminder tasks already use APScheduler to enqueue Jobs-backed notification jobs.
- External Sources connectors already provide OAuth/account/source/sync state patterns for content imports, but are file/content oriented rather than calendar-event oriented.
- Fastmail documentation describes syncing calendars from Google, iCloud, or other CalDAV servers and notes external tasks, reminders, or todos are not synced by default.
- Google Calendar uses RFC 5545-style recurrence fields such as RRULE, RDATE, and EXDATE.
- Microsoft Graph supports delta synchronization for calendar views, but that belongs in a later provider-specific adapter.
- Proton Calendar public support material currently emphasizes read-only external calendar subscriptions and does not provide an obvious generic two-way sync target for this module.

## Problem

tldw has many time-based concepts, but no first-class calendar surface for local planning and external calendar context.

Users need to see and manage:

- local events;
- dated todos/checkpoints;
- reminders;
- scheduled tldw work;
- Watchlist jobs and outputs;
- workflow/agent run timing;
- meetings and media events;
- external calendar commitments.

Today those concepts are split across domain surfaces. Scheduled Tasks should remain the automation lifecycle and debugging hub, but users still need a calendar-native workspace for time, planning, and schedule context.

The hard product challenge is preserving a simple v1 while supporting future expansion into two-way sync, richer recurrence, org policy, and more providers.

## Goals

1. Create a first-class local calendar module with native personal and org calendars.
2. Support local events and lightweight dated todos/checkpoints.
3. Display linked scheduled tasks, reminders, Jobs, Watchlists, Workflows, ACP schedules, and other tldw objects as calendar context without absorbing their domain configuration.
4. Import external calendar events as read-only provider-owned local records.
5. Allow tldw-local annotations, tags, and links on imported provider-owned events.
6. Support pragmatic recurrence for native items and preserve provider recurrence data for imported records.
7. Ship practical agenda and week views before building a full calendar suite.
8. Use existing AuthNZ/org RBAC rather than inventing a separate permission system.
9. Keep the sync architecture Jobs-backed, observable, and safe for local/self-hosted deployments.
10. Explicitly stage MVP, v1, and roadmap scope to prevent calendar sync from becoming unbounded.

## Non-Goals

- No outbound push or two-way sync in v1.
- No provider-owned event editing in v1.
- No full iCalendar recurrence editor in v1.
- No full task-list product replacement.
- No replacement for `/scheduled-tasks` as the automation workbench.
- No replacement for Watchlists, Workflows, ACP, Jobs, notes, meetings, or RAG surfaces.
- No Microsoft Graph calendar adapter in v1.
- No Proton-specific sync adapter in v1.
- No month/day/free-busy UI in the first practical UI slice.
- No rich org policy engine, resource calendars, delegation, or admin transfer controls in v1.

## Personas

### Personal Research User

Wants a single place to see local study plans, research meetings, reminders, content-ingestion schedules, and external calendar commitments.

### Org Planner

Uses org calendars to coordinate shared research, review, ingestion, or automation work. Needs RBAC-controlled visibility and editing without a separate policy model.

### Automation User

Uses Watchlists, Scheduled Tasks, Workflows, Jobs, and ACP schedules. Needs calendar context and deep links, but still expects automation debugging and result review to happen in the owning domain surface.

### External Calendar User

Uses Fastmail, another CalDAV service, Google Calendar, Microsoft, or Proton. Wants tldw to understand their schedule without taking over external calendar ownership.

### Operator / Admin

Needs sync health, stale-calendar warnings, credential revocation, and enough observability to diagnose failed imports without exposing secrets.

## Product Principles

1. **Local first, external read-only.** tldw-owned records are writable; provider-owned records are locked.
2. **Calendar owns time, domains own work.** Calendar shows when things happen; domain surfaces own specialized configuration and debugging.
3. **Annotations survive sync.** Provider refreshes must not overwrite tldw-local notes, tags, links, or copied-item provenance.
4. **Recurrence is pragmatic.** Support common recurrence now, preserve complex provider recurrence, and defer deep exception editing.
5. **Org RBAC is reused.** Calendar access should integrate with existing AuthNZ/org role and permission semantics.
6. **Sync status is product surface, not logs only.** Users need clear import state, counts, stale warnings, and failure recovery.
7. **Roadmap is explicit.** Push, Google, Microsoft, Proton, VTODO, free-busy, month/day, and rich policy should be named without being v1 blockers.

## Scope Phasing

### MVP

MVP should prove the local calendar model and core product value before external sync adds provider complexity.

MVP includes:

- personal calendars;
- org calendars with AuthNZ/org RBAC;
- local events;
- lightweight dated todos/checkpoints;
- basic one-time and common recurring items;
- agenda view;
- week view;
- item create/edit drawer;
- typed links to notes, media, scheduled tasks, Jobs, Watchlists, Workflows, ACP schedules, meetings, and URLs;
- linked scheduled-task projections as read-only calendar context;
- basic reminder/deferred task creation from calendar for simple use cases;
- clear ownership labels for local, org, linked, and provider-owned records.

MVP excludes:

- external sync;
- outbound push;
- provider accounts;
- month/day/free-busy;
- deep recurrence exception editing;
- external VTODO import.

### V1

V1 adds read-only external calendar import and durable local context around imported records.

V1 includes:

- generic CalDAV account configuration;
- provider profiles where useful, starting with Fastmail-style CalDAV configuration guidance;
- remote calendar discovery;
- external calendar binding and visibility settings;
- read-only import of provider-owned VEVENT records;
- provider ID, UID, ETag/CTag, sync cursor/state, deletion/tombstone, and recurrence preservation;
- sync now and periodic sync jobs;
- sync status with last sync, next sync, item counts, stale state, and errors;
- annotations, tags, and links on provider-owned records;
- copy-into-tldw flow for read-only provider events;
- mocked CalDAV integration tests;
- at least one real-provider smoke path documented for maintainers.

V1 may include read-only external VTODO only if provider behavior is predictable and testable. It should not block VEVENT import.

### Roadmap

Roadmap items:

- Google Calendar adapter;
- Microsoft Graph adapter;
- Proton-specific strategy after provider capability review;
- outbound push to dedicated tldw-owned provider calendars;
- full two-way sync;
- external VTODO import if deferred from v1;
- month view;
- day view;
- free-busy and availability;
- richer recurrence exception editing;
- resource calendars;
- delegation;
- admin transfer controls;
- richer org/team calendar policies;
- embeddings/RAG enrichment for annotations if useful after first usage.

## Core Data Model

### Calendar

Represents a local personal or org calendar.

Key fields:

- `id`;
- `tenant_id`;
- `owner_user_id`;
- `org_id`;
- `name`;
- `description`;
- `color`;
- `timezone`;
- `visibility`;
- `default_reminder_policy`;
- `rbac_policy_ref`;
- `archived_at`;
- `created_at`;
- `updated_at`.

### CalendarItem

Shared base for events and dated todos/checkpoints.

Key fields:

- `id`;
- `calendar_id`;
- `kind`: `event` or `todo`;
- `source_owner`: `tldw`, `provider`, or `linked_projection`;
- `provider_owned`: boolean;
- `title`;
- `description`;
- `start_at`;
- `end_at`;
- `due_at`;
- `all_day`;
- `timezone`;
- `status`;
- `priority`;
- `location_text`;
- `location_url`;
- `tags`;
- `recurrence_id`;
- `external_binding_id`;
- `source_uid`;
- `read_only_reason`;
- `deleted_at`;
- `created_at`;
- `updated_at`.

Rules:

- Local `tldw` items are editable when the principal has permission.
- Provider-owned items are locked for provider fields.
- Linked projections are read-only summaries of scheduled tasks, Jobs, Watchlists, Workflows, ACP schedules, or related domain records.
- Lightweight todos/checkpoints are calendar-native dated items, not a full task-list replacement.

### Recurrence

Stores normalized recurrence for local tldw-owned items and preserves raw provider recurrence for provider-owned records.

V1 local recurrence supports:

- daily;
- weekly by weekday;
- monthly by date;
- interval;
- count;
- until;
- all-day recurrence with IANA timezone handling.

Provider recurrence preservation supports:

- raw RRULE/RDATE/EXDATE payloads;
- recurrence instance identity where provided;
- provider exceptions displayed as imported read-only instances when available.

Roadmap:

- editing one instance;
- editing this-and-following;
- complex exception generation;
- full RFC 5545 authoring.

### CalendarAnnotation

Stores tldw-local context that must survive provider refreshes.

Fields:

- `id`;
- `calendar_item_id`;
- `author_user_id`;
- `note`;
- `tags`;
- `summary`;
- `metadata_json`;
- `created_at`;
- `updated_at`.

Annotations can exist on local, provider-owned, and linked projection records, subject to permissions.

### CalendarLink

Typed link between a calendar item and another tldw object or URL.

Supported link targets:

- notes;
- media;
- chunks/transcripts;
- meetings;
- RAG queries;
- scheduled tasks;
- Jobs;
- Workflow runs;
- Watchlist jobs/runs/outputs;
- ACP schedules/sessions/artifacts;
- chat sessions;
- prompts;
- files;
- arbitrary URLs.

### ExternalCalendarAccount

Represents user-owned external calendar credentials and provider metadata.

Fields:

- `id`;
- `user_id`;
- `provider`: `caldav` initially;
- `profile`: `generic`, `fastmail`, or future profiles;
- `display_name`;
- `server_url`;
- `username_hint`;
- `secret_ref`;
- `status`;
- `last_verified_at`;
- `revoked_at`;
- `created_at`;
- `updated_at`.

Credentials must be stored encrypted using existing secret-storage/security patterns where possible. Raw passwords, app passwords, tokens, and auth headers must never be logged or returned.

### ExternalCalendarBinding

Maps a provider account and remote calendar to a local calendar or imported provider-owned item namespace.

Fields:

- `id`;
- `account_id`;
- `local_calendar_id`;
- `remote_calendar_id`;
- `remote_display_name`;
- `remote_color`;
- `sync_enabled`;
- `import_visibility`;
- `sync_cursor`;
- `ctag`;
- `etag_snapshot`;
- `last_sync_started_at`;
- `last_sync_succeeded_at`;
- `last_sync_failed_at`;
- `last_error`;
- `last_item_count`;
- `stale_after`;
- `created_at`;
- `updated_at`.

## Sync Model

V1 sync is read-only import.

Flow:

1. User adds a CalDAV account.
2. Server verifies credentials without exposing secrets.
3. Server discovers remote calendars.
4. User binds selected remote calendars.
5. Calendar sync scheduler periodically enqueues sync Jobs.
6. Sync worker fetches changes and upserts provider-owned records.
7. Provider field changes replace cached provider fields.
8. Local annotations, tags, and links remain untouched.
9. Deleted remote events are marked as remote-deleted/tombstoned locally.
10. Agenda/week queries include local items, provider-owned items, and linked projections according to filters and permissions.

Conflict rules:

- Provider-owned fields are provider-authoritative.
- Local annotations are tldw-authoritative.
- Local copied items are independent tldw-owned records with source provenance.
- If a copied item source changes, the UI may show source-changed context but should not auto-merge.
- No local changes are pushed back to providers in v1.

Scheduler/Jobs behavior:

- Periodic sync scan uses APScheduler as a bridge.
- Actual sync execution uses Jobs.
- Jobs should be source-scoped and idempotent.
- A binding should not have overlapping active sync jobs.
- Retry/backoff should handle transient network/provider failures.
- Manual sync should queue a job and return immediately.
- Provider callbacks/webhooks are not required for v1.

## Calendar And Scheduled Tasks Boundary

Calendar should not duplicate the Scheduled Tasks Automation Workbench.

Calendar owns:

- time-based visualization;
- agenda/week placement;
- local event/todo creation;
- simple reminder/deferred task creation;
- links and annotations;
- calendar-specific filtering.

Scheduled Tasks owns:

- automation creation templates;
- run history;
- result inbox;
- logs;
- retry/cancel/debug operations;
- bulk automation management;
- Watchlists/Workflows/ACP domain-specific handoff.

Calendar projections of scheduled work should be read-only and deep-link to their source. Complex automation edits must happen in the owning surface.

## Backend Architecture

Suggested module layout:

- `tldw_Server_API/app/core/Calendar/`
  - domain services;
  - permissions;
  - recurrence;
  - view expansion;
  - links and annotations;
  - sync orchestration.
- `tldw_Server_API/app/core/Calendar/providers/`
  - CalDAV adapter;
  - provider profiles;
  - future Google/Microsoft/Proton adapters.
- `tldw_Server_API/app/core/DB_Management/Calendar_DB.py`
  - SQLite default persistence;
  - PostgreSQL-compatible patterns where existing repo conventions support them.
- `tldw_Server_API/app/api/v1/endpoints/calendar.py`
  - thin FastAPI router.
- `tldw_Server_API/app/api/v1/schemas/calendar_schemas.py`
  - Pydantic models.
- `tldw_Server_API/app/services/calendar_sync_scheduler.py`
  - periodic sync scan.
- `tldw_Server_API/app/core/Calendar/calendar_sync_worker.py`
  - Jobs-backed sync execution.

API groups:

- calendars: create, list, update, archive, permissions;
- items: create, list, update, delete local events and todos;
- views: agenda/week queries with recurrence expansion, overlays, projections, filters, and permission checks;
- annotations: create, update, delete notes/tags/summaries;
- links: create, delete, list typed links;
- external accounts: create, verify, revoke, delete CalDAV accounts;
- external bindings: discover, bind, unbind, enable/disable, inspect sync status;
- sync jobs: trigger sync, list active/recent jobs, inspect errors.

Permission model:

- Reuse AuthNZ and org roles/permissions.
- Calendar-level overrides can exist as narrow ACL rows.
- Avoid a separate policy engine in v1.
- Prepare the model for future rich policy controls.

## Frontend UX

V1 should ship a practical calendar workspace.

Primary views:

- Agenda.
- Week.
- Item detail drawer.
- Create/edit form.
- Calendar/source filter rail.
- External sync settings.
- Sync health panel.
- Copy-into-tldw action for provider-owned events.

Agenda view:

- Upcoming local events.
- Upcoming dated todos/checkpoints.
- Linked scheduled work.
- Provider-owned imported events.
- Sync warnings and stale calendar states.
- Filters by calendar, org, owner, source type, item kind, and linked domain.

Week view:

- Time-grid placement.
- All-day row.
- Provider-owned styling.
- Local/org calendar styling.
- Linked projection styling.
- Drag/drop can be roadmap unless implementation proves low risk.

Item drawer:

- Title, times, timezone, all-day state.
- Kind: event, todo, linked projection.
- Ownership: local, org, provider-owned, linked.
- Recurrence summary.
- Description.
- Location/link fields.
- Tags.
- Annotations.
- Links to related tldw objects.
- Provider metadata when relevant.
- Read-only reason for locked records.
- Copy-into-tldw for provider-owned events.

External sync settings:

- Add CalDAV account.
- Discover calendars.
- Select calendars to import.
- Show sync enabled/disabled.
- Show last success/failure.
- Show next sync.
- Show imported item count.
- Show stale warning.
- Revoke account.
- Delete account and imported provider-owned records, with confirmation.

Roadmap UI:

- Month view.
- Day view.
- Free-busy/availability.
- Resource calendars.
- Rich admin policy screens.
- Outbound sync configuration.

## Error Handling And Recovery

User-facing states:

- account invalid;
- credential expired/revoked;
- provider unreachable;
- remote calendar deleted;
- remote calendar renamed;
- sync stale;
- partial import;
- recurrence unsupported;
- item locked because provider-owned;
- linked source unavailable;
- org permission denied;
- job queued/running/failed.

Recovery actions:

- verify account;
- reconnect account;
- sync now;
- disable binding;
- delete binding;
- copy read-only event into tldw;
- open source scheduled task/job/workflow/watchlist;
- view diagnostics;
- contact admin when org policy blocks access.

Logging:

- Include account id, binding id, job id, user/org scope, provider, and status.
- Do not log secrets, auth headers, raw credential fields, or full provider payloads unless scrubbed.

## Security And Privacy

Requirements:

- Encrypt external calendar credentials.
- Never expose raw credentials through APIs.
- Never log credentials.
- Enforce owner/org permissions on every calendar, item, annotation, link, and sync status read.
- Scope external accounts to users unless future org-managed service accounts are explicitly designed.
- Make account revoke/delete behavior explicit.
- Provide clear deletion semantics for imported provider-owned records.
- Preserve local/self-hosted privacy expectations.
- Rate limit account verification and sync trigger endpoints.
- Avoid SSRF by validating CalDAV server URLs and applying existing egress controls where appropriate.
- Store only provider metadata needed for sync, display, and diagnostics.

## Testing Strategy

Backend unit tests:

- calendar/item validation;
- event vs todo invariants;
- permission checks;
- recurrence expansion for common rules;
- timezone and DST edge cases;
- provider-owned edit blocking;
- annotation survival across provider refresh;
- link normalization;
- sync state transitions.

Backend integration tests:

- create/list/update/archive calendars;
- org RBAC access checks;
- create/list/update/delete local items;
- agenda/week queries;
- linked scheduled-task projections;
- CalDAV account verification with mock provider;
- remote calendar discovery with mock provider;
- sync job enqueue and completion;
- provider update/delete import behavior;
- sync failure and retry status.

Frontend tests:

- agenda rendering;
- week rendering;
- source filters;
- local event create/edit;
- local dated todo create/edit;
- provider-owned event read-only behavior;
- copy-into-tldw;
- annotation editing;
- sync settings states;
- stale/failure recovery states.

Property-style tests:

- daily recurrence;
- weekly by weekday recurrence;
- monthly recurrence;
- recurrence count/until boundaries;
- all-day timezone handling;
- DST transitions;
- generated occurrences stay within query windows.

Security verification:

- Bandit on touched backend scope during implementation.
- Secret logging regression tests for provider auth paths.
- Permission regression tests for org/private calendar access.

## Provider Notes

These notes are planning constraints, not implementation commitments.

- Fastmail supports syncing from Google, iCloud, or other CalDAV servers and notes that tasks, reminders, or todos are not synced by default from external calendars. This supports treating external VTODO as optional rather than a v1 blocker.
- Google Calendar recurrence uses RFC 5545-style recurrence fields including RRULE, RDATE, and EXDATE. This supports preserving raw recurrence data and avoiding premature full recurrence editing.
- Microsoft Graph supports event delta queries for calendar views. This supports a future Microsoft adapter, but the date-range delta model is provider-specific enough to stay roadmap.
- Proton Calendar support material describes read-only external calendar subscriptions and public-link sharing for non-Proton users. Proton should remain a named roadmap target pending a provider capability review.

Reference links:

- Fastmail calendar sync: https://www.fastmail.help/hc/en-us/articles/360058752754-How-to-synchronize-a-calendar
- Google Calendar events and recurrence: https://developers.google.com/workspace/calendar/api/concepts/events-calendars
- Microsoft Graph calendar view delta: https://learn.microsoft.com/en-us/graph/delta-query-events
- Proton Calendar support: https://proton.me/support/calendar

## Open Questions

1. Should external CalDAV VTODO import be included in v1 when a provider supports it, or should all external todos remain roadmap until VEVENT import proves stable?
2. Should calendar annotations be plain text only in v1, or should they reuse note/link models from an existing domain?
3. Should the first CalDAV real-provider smoke test target Fastmail specifically?
4. Should calendar data live in a new shared calendar DB, per-user calendar DBs, or the existing Collections DB with a new domain namespace?

## Acceptance Criteria

- The PRD distinguishes MVP, v1, and roadmap scope.
- The PRD defines Calendar as a first-class local module, not only a Scheduled Tasks view.
- The PRD includes native local events and lightweight dated todos/checkpoints.
- The PRD keeps external sync read-only/import-only in v1.
- The PRD names CalDAV as the first provider family and Google/Microsoft/Proton as later provider targets.
- The PRD requires provider-owned imported events to be read-only but locally annotatable/linkable.
- The PRD keeps Scheduled Tasks as the automation workbench and Calendar as the time/context surface.
- The PRD includes org RBAC integration without a new rich policy engine in v1.
- The PRD includes backend architecture, API groups, data model, sync model, UX, error handling, security, testing, and rollout risks.

## Risks And Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Scope expands beyond simple calendar value | Delayed delivery | Keep MVP/v1/roadmap gates explicit |
| Org calendars become a policy project | Implementation churn | Reuse AuthNZ/org RBAC and defer rich policy |
| Users expect external edits to sync back | Trust loss | Strong ownership labels and clear no-push messaging |
| Recurrence complexity explodes | Bugs around dates/timezones | Support common local rules, preserve complex provider data |
| Calendar duplicates Scheduled Tasks | Confusing IA | Calendar shows projections; Scheduled Tasks owns automation lifecycle |
| CalDAV provider quirks cause inconsistent imports | Bad sync trust | Mock tests, provider profiles, sync diagnostics, and real-provider smoke docs |
| Credentials leak through logs or errors | Security incident | Encrypted storage, scrubbing, logging tests, no raw payload logging |

## Rollout Plan

1. MVP local calendar foundation:
   - Calendar DB/service/API.
   - Local personal calendars.
   - Org calendars with RBAC checks.
   - Local events and dated todos.
   - Agenda/week query APIs.
2. Practical UI:
   - Agenda and week views.
   - Item drawer.
   - Create/edit forms.
   - Filter rail.
   - Linked scheduled-task projections.
3. CalDAV import:
   - Account verification.
   - Remote discovery.
   - Binding management.
   - Jobs-backed sync worker.
   - Sync health states.
4. Provider-owned context:
   - Read-only imported item UX.
   - Annotations.
   - Links.
   - Copy-into-tldw.
5. Hardening:
   - Real-provider smoke documentation.
   - Stale/error recovery.
   - Permission/security regression tests.
   - Roadmap adapter planning for Google.

## Definition Of Done For Implementation Planning

- Design spec is reviewed and approved.
- Backlog task links to the final spec.
- Implementation plan splits MVP, v1 sync, and roadmap preparation.
- Tests are assigned to each implementation stage.
- Provider capability assumptions are rechecked before adapter implementation begins.
