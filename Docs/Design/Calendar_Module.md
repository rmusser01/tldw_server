# Calendar Module

Last updated: 2026-06-06

## Purpose

The Calendar module is the time and schedule surface for tldw. It stores local calendars, events, lightweight dated todos, annotations, and links while projecting scheduled work and importing external calendars as read-only context.

V1 is intentionally conservative:

- local tldw items are editable;
- linked scheduled-task projections are read-only and deep-link to their owning workflow;
- external CalDAV imports are read-only provider-owned records;
- outbound push, two-way sync, Google Calendar, Microsoft Graph, Proton, and external VTODO import remain roadmap work.

## Module Boundary

Calendar owns:

- calendar containers, membership rows, colors, visibility, and timezone metadata;
- local event and todo records;
- recurrence expansion for local recurring items;
- agenda and week views over local items, provider imports, and scheduled-task projections;
- local annotations, local tags, and links attached to calendar items;
- external account metadata, secret references, bindings, sync state, and sync event history.

Calendar does not own:

- Scheduled Tasks automation definitions, queues, retries, or debugging;
- provider-side event mutation;
- arbitrary notification job orchestration;
- Notes storage for annotations.

Use `CalendarLink` to connect a calendar item to a Note, scheduled task, media item, or other source record.

## Local, Provider-Owned, And Linked Items

Calendar item source ownership is explicit:

- `source_owner="tldw"`: local Calendar item. Users with sufficient Calendar permissions can edit item fields.
- `source_owner="provider"` with `provider_owned=true`: imported external item. Provider-owned fields are locked in service, API, and UI layers. Users can add local context such as tags, annotations, links, and can copy the item into a local tldw-owned record.
- `source_owner="linked_projection"`: read-only projection from another tldw domain, currently Scheduled Tasks/reminders. Edits belong in the owning domain.

Provider refreshes must preserve local annotations, links, and local tags. Remote deletes tombstone provider-owned imports rather than deleting local context silently.

## Scheduled Tasks Boundary

Scheduled Tasks remains the automation workbench. Calendar shows scheduled work as time-based context.

Calendar can create narrow reminder handoffs through `POST /api/v1/calendar/reminders`, which delegates to the existing reminder task path. Complex automation edits, retries, and execution inspection stay in Scheduled Tasks and Jobs.

## Storage

The Calendar DB is a shared SQLite database at:

```text
Databases/calendar.db
```

The repository class is `CalendarDatabase` in `tldw_Server_API/app/core/DB_Management/Calendar_DB.py`.

Core tables include:

- `calendars`
- `calendar_memberships`
- `calendar_items`
- `calendar_recurrences`
- `calendar_annotations`
- `calendar_links`
- `external_calendar_accounts`
- `external_calendar_bindings`
- `calendar_sync_events`
- `calendar_external_account_secrets`

External account rows store provider metadata and a secret reference. Raw app passwords, tokens, auth headers, and provider credential payloads must not be stored in account rows, Jobs payloads, logs, or API responses.

## Permissions

Calendar API endpoints use existing AuthNZ permissions:

- `CALENDAR_READ`
- `CALENDAR_WRITE`
- `CALENDAR_SYNC`

Calendar-level roles are:

- `owner`
- `editor`
- `commenter`
- `viewer`

Calendar memberships support user and org-role principals. External personal accounts can bind only to calendars owned by the same user and tenant; they cannot bind to org calendars.

## API Surface

The router is mounted under `/api/v1/calendar`.

Core endpoints:

- `POST /calendars`
- `GET /calendars`
- `POST /calendars/{calendar_id}/memberships`
- `GET /calendars/{calendar_id}/memberships`
- `DELETE /calendars/{calendar_id}/memberships/{principal_type}/{principal_id}`
- `POST /items`
- `PATCH /items/{item_id}`
- `GET /views/agenda`
- `GET /views/week`
- `POST /items/{item_id}/annotations`
- `PUT /items/{item_id}/local-tags`
- `POST /items/{item_id}/links`
- `POST /items/{item_id}/copy`
- `POST /reminders`

External sync endpoints:

- `GET /external/accounts`
- `POST /external/accounts`
- `POST /external/accounts/{account_id}/verify`
- `POST /external/accounts/{account_id}/discover`
- `POST /external/accounts/{account_id}/revoke`
- `DELETE /external/accounts/{account_id}`
- `POST /external/bindings`
- `GET /external/accounts/{account_id}/bindings`
- `PATCH /external/bindings/{binding_id}`
- `POST /external/bindings/{binding_id}/enable`
- `POST /external/bindings/{binding_id}/disable`
- `DELETE /external/bindings/{binding_id}`
- `GET /external/bindings/{binding_id}/sync-status`
- `GET /external/bindings/{binding_id}/sync-events`
- `POST /external/bindings/{binding_id}/sync`

## Sync Model

V1 external sync supports generic CalDAV VEVENT import. Fastmail is the first documented real-provider smoke target because Fastmail documents calendar access over CalDAV and app-password authentication for non-OAuth personal testing.

The import flow is:

1. Create a CalDAV account with provider metadata and a secret reference.
2. Verify credentials with a provider call.
3. Discover remote calendars through CalDAV `PROPFIND`.
4. Bind a discovered remote calendar to a local personal calendar.
5. Queue bounded sync Jobs by binding and time window.
6. Fetch VEVENTs only, upsert provider-owned Calendar items, tombstone missing remote records, and record sync events.

Sync uses Jobs:

- domain: `calendar`
- queue: `default`
- job type: `calendar_sync`
- idempotency key shape: `calendar:sync:binding:{binding_id}:{window_start}:{window_end}:{reason}`

Manual sync defaults the window from binding lookback/lookahead values. API callers may pass `window_start`, `window_end`, and `reason` to `POST /external/bindings/{binding_id}/sync`.

## Runtime Flags

Calendar API routes are always registered when the backend is running, subject to route-gating and AuthNZ.

The sync worker and scheduler are off by default and must be enabled explicitly:

```bash
CALENDAR_SYNC_JOBS_WORKER_ENABLED=true
CALENDAR_SYNC_SCHEDULER_ENABLED=true
```

Additional worker tuning:

```bash
CALENDAR_SYNC_WORKER_ID=calendar-sync-worker-1
CALENDAR_SYNC_LEASE_SECONDS=120
CALENDAR_SYNC_RENEW_THRESHOLD_SECONDS=10
CALENDAR_SYNC_RENEW_JITTER_SECONDS=0
CALENDAR_SYNC_SCHEDULER_INTERVAL_SECONDS=60
```

`TLDW_WORKERS_SIDECAR_MODE=true` disables in-process workers, including Calendar sync workers. In that mode, run Calendar Jobs workers out of process.

## Frontend

The primary UI route is `/calendar`.

The current frontend includes:

- Agenda view;
- Week view;
- calendar/source/kind filters;
- local item drawer;
- source ownership labels;
- provider-owned read-only handling;
- local tags, annotations, links, and copy action for provider items;
- Sync view for CalDAV account creation, discovery, binding, manual sync, revoke, and delete.

The frontend intentionally does not expose provider field editing for provider-owned imports.

## Known Limits

- CalDAV import is VEVENT-only. VTODOs are intentionally ignored in v1.
- External sync is read-only.
- Provider events are imported into personal calendars only.
- Month/day/free-busy views are not part of this slice.
- Google Calendar, Microsoft, Proton, and outbound push remain roadmap items.
