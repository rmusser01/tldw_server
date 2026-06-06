# Calendar CalDAV Smoke Test

Last updated: 2026-06-06

## Scope

This smoke path verifies the Calendar module's read-only CalDAV import using Fastmail as the first real-provider target. Fastmail documents that calendar data is accessible via CalDAV and that app passwords can be generated for non-OAuth personal testing; do not store real credentials in this repository or in screenshots.

The smoke target is:

```text
https://caldav.fastmail.com
```

The module should discover concrete calendar collection URLs from that server. Do not hard-code a user's discovered collection URL in docs or tests.

## Preconditions

Backend:

```bash
source .venv/bin/activate
CALENDAR_SYNC_JOBS_WORKER_ENABLED=true \
CALENDAR_SYNC_SCHEDULER_ENABLED=true \
python -m uvicorn tldw_Server_API.app.main:app --reload
```

Frontend:

```bash
bun run dev
```

Use a non-production Fastmail account or a disposable test calendar. Create at least:

- one event inside the binding lookback/lookahead window;
- one event outside the window;
- one VTODO/reminder item, if the client used to seed data supports CalDAV VTODO.

Create a Fastmail app password through the Fastmail account security settings. Never paste the app password into docs, commits, issue comments, logs, or screenshots.

## Backend API Smoke

Use the authenticated tldw API client or equivalent HTTP calls. Replace placeholders locally.

1. Create a local calendar.

   ```bash
   curl -X POST "$TLDW_URL/api/v1/calendar/calendars" \
     -H "X-API-KEY: $TLDW_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"name":"Fastmail Smoke","timezone":"UTC","visibility":"private"}'
   ```

2. Create the CalDAV account.

   ```bash
   curl -X POST "$TLDW_URL/api/v1/calendar/external/accounts" \
     -H "X-API-KEY: $TLDW_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{
       "provider":"caldav",
       "display_name":"Fastmail Smoke",
       "server_url":"https://caldav.fastmail.com",
       "username":"REDACTED_FASTMAIL_USERNAME",
       "password":"REDACTED_APP_PASSWORD"
     }'
   ```

   Expected:

   - response includes `provider="caldav"` and `display_name`;
   - response does not include the raw password;
   - `account_metadata` may include non-secret `server_url` and `username`;
   - account row has a `secret_ref`.

3. Verify the account.

   ```bash
   curl -X POST "$TLDW_URL/api/v1/calendar/external/accounts/$ACCOUNT_ID/verify" \
     -H "X-API-KEY: $TLDW_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"password":"REDACTED_APP_PASSWORD"}'
   ```

   Expected: `verified=true`.

4. Discover remote calendars.

   ```bash
   curl -X POST "$TLDW_URL/api/v1/calendar/external/accounts/$ACCOUNT_ID/discover" \
     -H "X-API-KEY: $TLDW_API_KEY"
   ```

   Expected:

   - at least one `remote_calendar_id`;
   - display name when Fastmail exposes it;
   - capability metadata is scrubbed of secrets.

5. Bind one remote calendar to the local calendar.

   ```bash
   curl -X POST "$TLDW_URL/api/v1/calendar/external/bindings" \
     -H "X-API-KEY: $TLDW_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{
       "account_id":ACCOUNT_ID,
       "calendar_id":LOCAL_CALENDAR_ID,
       "remote_calendar_id":"REMOTE_CALENDAR_ID",
       "remote_display_name":"Fastmail Smoke",
       "lookback_days":30,
       "lookahead_days":120,
       "sync_interval_minutes":60,
       "sync_enabled":true
     }'
   ```

   Expected:

   - binding is enabled;
   - binding uses the requested lookback/lookahead window;
   - personal external accounts cannot bind to org calendars.

6. Queue a manual sync.

   ```bash
   curl -X POST "$TLDW_URL/api/v1/calendar/external/bindings/$BINDING_ID/sync" \
     -H "X-API-KEY: $TLDW_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"reason":"manual-smoke"}'
   ```

   Expected:

   - response includes `job_id`;
   - `queued=true` or `status="already_active"` if a matching sync is already queued/processing;
   - Jobs payload stores binding id, window, and reason only, not credentials.

7. Inspect sync status and events.

   ```bash
   curl "$TLDW_URL/api/v1/calendar/external/bindings/$BINDING_ID/sync-status" \
     -H "X-API-KEY: $TLDW_API_KEY"

   curl "$TLDW_URL/api/v1/calendar/external/bindings/$BINDING_ID/sync-events" \
     -H "X-API-KEY: $TLDW_API_KEY"
   ```

   Expected:

   - successful sync event after worker completion;
   - `last_sync_at` is set;
   - `last_error` is empty after success.

8. Query agenda.

   ```bash
   curl "$TLDW_URL/api/v1/calendar/views/agenda?start_at=2026-06-01T00%3A00%3A00%2B00%3A00&end_at=2026-06-30T00%3A00%3A00%2B00%3A00&calendar_ids=$LOCAL_CALENDAR_ID&include_scheduled_tasks=true" \
     -H "X-API-KEY: $TLDW_API_KEY"
   ```

   Expected:

   - Fastmail VEVENT inside the window appears;
   - item has `source_owner="provider"` and `read_only_reason`;
   - seeded VTODO does not appear;
   - event outside the sync/query window does not appear.

9. Verify provider-owned mutation lock.

   Attempt a field edit against the provider-owned item:

   ```bash
   curl -X PATCH "$TLDW_URL/api/v1/calendar/items/$PROVIDER_ITEM_ID" \
     -H "X-API-KEY: $TLDW_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"title":"Should not update"}'
   ```

   Expected: `409 item_read_only`.

10. Verify local context is allowed.

   ```bash
   curl -X PUT "$TLDW_URL/api/v1/calendar/items/$PROVIDER_ITEM_ID/local-tags" \
     -H "X-API-KEY: $TLDW_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"tags":["smoke"]}'
   ```

   Expected: local tags are saved and survive a second sync.

11. Revoke the account.

   ```bash
   curl -X POST "$TLDW_URL/api/v1/calendar/external/accounts/$ACCOUNT_ID/revoke" \
     -H "X-API-KEY: $TLDW_API_KEY"
   ```

   Expected:

   - account is revoked;
   - stored secret material is no longer active;
   - bindings are disabled;
   - copied local tldw-owned items are not deleted.

## Frontend Smoke

1. Open `/calendar`.
2. Confirm Agenda and Week views render without overlapping filters or item rows.
3. Create a local event.
4. Open the Sync view.
5. Add the Fastmail CalDAV account with server URL `https://caldav.fastmail.com`.
6. Discover calendars.
7. Bind one remote calendar to the local calendar with lookback/lookahead values.
8. Click Sync now.
9. Return to Agenda and verify the Fastmail event appears with provider ownership.
10. Open the provider-owned item drawer.

Expected:

- provider-owned fields are visibly locked;
- local tags/annotations/links remain available;
- copy into tldw is available;
- raw passwords are never displayed after the account create/verify flow;
- Sync states render cleanly on desktop and mobile widths.

## Cleanup

After the smoke run:

1. Revoke or delete the external account in tldw.
2. Delete the Fastmail app password from Fastmail account settings.
3. Remove any disposable smoke calendar data from the Fastmail test account.
4. If test data was created in a persistent local tldw instance, archive/delete the local smoke calendar.

## Troubleshooting

- `400 calendar_validation_error` during account create: confirm `server_url`, `username`, and `password` are present for CalDAV credentials.
- `400` or provider discovery failure: confirm the server URL is a public HTTPS CalDAV URL. The provider rejects localhost and private/local address targets.
- `already_active` sync response: wait for the existing queued/processing sync or change the sync window/reason.
- No imported items: confirm the event is a VEVENT inside the sync window; VTODO import is intentionally unsupported in v1.
- Provider item is editable: treat as a blocker. Provider-owned field edits should fail in service/API/UI.
