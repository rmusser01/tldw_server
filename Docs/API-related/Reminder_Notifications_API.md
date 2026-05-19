# Reminder Tasks and Notifications API

## Overview

This module provides:

- User-managed reminder tasks (`/api/v1/tasks`)
- In-app notifications inbox (`/api/v1/notifications`)
- Realtime notifications stream (`/api/v1/notifications/stream`)

Reminder execution flow:

1. Create/update a task
2. Reminders scheduler enqueues due jobs into Jobs
3. Reminder jobs worker creates `reminder_due` notifications
4. Jobs notifications bridge maps terminal job events into `job_completed` / `job_failed` notifications

## Auth and Permissions

Routes require AuthNZ permissions:

- `tasks.read`
- `tasks.control`
- `notifications.read`
- `notifications.control`

Rate-limit keys:

- `tasks.read`, `tasks.control`
- `notifications.read`, `notifications.control`

## Tasks Endpoints

Base: `/api/v1/tasks`

### Create Task

- `POST /api/v1/tasks`
- Permission: `tasks.control`
- Response: `201`

Example request:

```json
{
  "title": "Review export output",
  "body": "Validate citations and rerun if needed",
  "schedule_kind": "one_time",
  "run_at": "2026-03-01T10:00:00+00:00",
  "enabled": true,
  "link_type": "item",
  "link_id": "item-42"
}
```

Recurring example:

```json
{
  "title": "Weekly follow-up",
  "schedule_kind": "recurring",
  "cron": "0 9 * * MON",
  "timezone": "America/New_York",
  "enabled": true
}
```

### List Tasks

- `GET /api/v1/tasks`
- Permission: `tasks.read`

### Get Task

- `GET /api/v1/tasks/{task_id}`
- Permission: `tasks.read`

### Update Task

- `PATCH /api/v1/tasks/{task_id}`
- Permission: `tasks.control`

Example patch:

```json
{
  "enabled": false,
  "title": "Updated title"
}
```

### Delete Task

- `DELETE /api/v1/tasks/{task_id}`
- Permission: `tasks.control`

## Notifications Endpoints

Base: `/api/v1/notifications`

### List Notifications

- `GET /api/v1/notifications?limit=100&offset=0&include_archived=false`
- Permission: `notifications.read`
- Archived notification items include `snooze_until` when the notification has an active snooze reminder backing it.

### Unread Count

- `GET /api/v1/notifications/unread-count`
- Permission: `notifications.read`

### Mark Read

- `POST /api/v1/notifications/mark-read`
- Permission: `notifications.control`

Request:

```json
{
  "ids": [101, 102]
}
```

### Dismiss

- `POST /api/v1/notifications/{notification_id}/dismiss`
- Permission: `notifications.control`

### Snooze

- `POST /api/v1/notifications/{notification_id}/snooze`
- Permission: `notifications.control`
- Creates the reminder task and dismisses the source notification from the active inbox.

Request:

```json
{
  "minutes": 30
}
```

Response:

```json
{
  "task_id": "task_abc123",
  "run_at": "2026-03-01T10:30:00+00:00"
}
```

### Cancel Snooze

- `DELETE /api/v1/notifications/{notification_id}/snooze`
- Permission: `notifications.control`

Response:

```json
{
  "cancelled": true,
  "deleted_tasks": 1
}
```

### Preferences

- `GET /api/v1/notifications/preferences`
- `PATCH /api/v1/notifications/preferences`
- Permissions: `notifications.read` (GET), `notifications.control` (PATCH)

Patch body fields:

- `reminder_enabled`
- `job_completed_enabled`
- `job_failed_enabled`

## Realtime Stream (SSE)

Endpoint:

- `GET /api/v1/notifications/stream`
- Permission: `notifications.read`
- Content type: `text/event-stream`

Cursoring:

- Preferred: `Last-Event-ID` header
- Alternate: `after` query param
- Server emits `id:` per event for replay/resume

Event types:

- `notification`
- `notifications_coalesced`
- `reset_required`
- `heartbeat`

Notification frame example:

```text
id: 1205
event: notification
data: {"event_id":1205,"notification_id":1205,"kind":"job_failed","created_at":"2026-03-01T10:00:00+00:00","title":"Job failed","message":"chatbooks/export failed.","severity":"error"}
```

`reset_required` example:

```text
event: reset_required
data: {"reason":"cursor_too_old","min_event_id":700,"latest_event_id":1205}
```

## Environment Flags and Defaults

### Reminder Scheduling / Workers

- `REMINDERS_SCHEDULER_ENABLED` (default: disabled)
- `REMINDERS_SCHEDULER_TZ` (default: `UTC`)
- `REMINDERS_SCHEDULER_RESCAN_SEC` (default: `300`, minimum `30`)
- Task create/update/delete requests attempt immediate in-process scheduler reconciliation when a scheduler instance is running; periodic rescan remains as safety sync.
- `REMINDER_JOBS_WORKER_ENABLED` (default: disabled)
- `REMINDER_JOBS_QUEUE` (default: `default`)

### Jobs -> Notifications Bridge

- `JOBS_NOTIFICATIONS_BRIDGE_ENABLED` (default: disabled)
- `JOBS_NOTIFICATIONS_CONSUMER_NAME` (default: `jobs_notifications_bridge`)
- `JOBS_NOTIFICATIONS_LEASE_OWNER_ID` (default: auto-generated from pid)
- `JOBS_NOTIFICATIONS_LEASE_SECONDS` (default: `30`, minimum `5`)
- `JOBS_NOTIFICATIONS_BATCH_SIZE` (default: `200`, clamped `1..500`)
- `JOBS_NOTIFICATIONS_POLL_INTERVAL_SEC` (default: `1.0`, minimum `0.01`)
- `JOBS_NOTIFICATIONS_BRIDGE_STATE_USER_ID` (default: single-user ID, fallback `1`)

### Notifications SSE Stream

- `NOTIFICATIONS_STREAM_REPLAY_WINDOW` (default: `500`, clamped `1..5000`)
- `NOTIFICATIONS_STREAM_BATCH_SIZE` (default: `200`, clamped `1..1000`)
- `NOTIFICATIONS_STREAM_BURST_THRESHOLD` (default: `50`, clamped `1..1000`)
- `NOTIFICATIONS_STREAM_POLL_SEC` (default: `1.0`, minimum `0.01`)
- `NOTIFICATIONS_STREAM_HEARTBEAT_SEC` (default: `10.0`, minimum `0.05`)
- `NOTIFICATIONS_STREAM_MAX_DURATION_SEC` (default: disabled when unset/`<=0`)
- `NOTIFICATIONS_STREAM_FLOOR_CHECK_EVERY_POLLS` (default: `15`, clamped `1..3600`)
- `NOTIFICATIONS_STREAM_SEND_TIMEOUT_SEC` (default: `1.0`, minimum `0.05`)

### Retention Prune

- `NOTIFICATIONS_PRUNE_ENABLED` (default: disabled)
- `NOTIFICATIONS_PRUNE_INTERVAL_SEC` (default: `3600`, minimum `60`)
- `NOTIFICATIONS_PRUNE_READ_DISMISSED_DAYS` (default: `30`)
- `NOTIFICATIONS_PRUNE_ARCHIVE_GRACE_DAYS` (default: `7`)
- `NOTIFICATIONS_RETENTION_DAYS_REMINDER_DUE` (default: `90`)
- `NOTIFICATIONS_RETENTION_DAYS_REMINDER_FAILED` (default: `90`)
- `NOTIFICATIONS_RETENTION_DAYS_JOB_COMPLETED` (default: `30`)
- `NOTIFICATIONS_RETENTION_DAYS_JOB_FAILED` (default: `60`)

## Notes

- Notifications are deduped via `dedupe_key` (jobs bridge uses `jobs-event:{event_id}`).
- Current implementation does **not** enforce an active reminder task cap at API level.
