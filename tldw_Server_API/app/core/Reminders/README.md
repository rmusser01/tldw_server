# Reminders

Reminders owns reminder task lifecycle helpers and due-reminder job handling for the in-app notifications system. Reminder tasks are stored in Collections DB, scheduled by an APScheduler service, executed through the core Jobs pipeline, and delivered as user notifications.

## Start Here

- `reminders_service.py` provides task CRUD helpers and notification snooze behavior.
- `reminder_jobs.py` handles `reminder_due` Jobs and creates notifications.
- Related API surface: `tldw_Server_API/app/api/v1/endpoints/reminders.py` and `tldw_Server_API/app/api/v1/endpoints/notifications.py`.
- Related schemas: `tldw_Server_API/app/api/v1/schemas/reminders_schemas.py`.
- Related scheduler and worker: `tldw_Server_API/app/services/reminders_scheduler.py` and `tldw_Server_API/app/services/reminder_jobs_worker.py`.
- Related tests live under `tldw_Server_API/tests/Notifications/`: `test_reminders_service.py`, `test_reminders_api.py`, `test_reminders_scheduler.py`, `test_reminder_jobs_worker.py`, and `test_reminders_schemas.py`.

## Responsibilities

- Create, list, read, update, and delete reminder tasks through Collections DB.
- Validate notification snooze windows and create one-time snooze tasks.
- Match archived or dismissed notifications to active snooze reminder tasks.
- Cancel snooze tasks and clear notification snooze links.
- Handle due reminder Jobs by creating `reminder_due` notifications.
- Record reminder task runs and dedupe by scheduled run slot.
- Disable one-time tasks after successful execution.

## Module Map

- `reminders_service.py`: reminder task service facade plus snooze matching, creation, listing, and cancellation.
- `reminder_jobs.py`: Jobs handler for due reminder execution and notification creation.
- `__init__.py`: package marker.

## How It Connects

- `reminders.py` exposes `/tasks` CRUD routes with AuthNZ task permissions and rate limiting.
- `notifications.py` uses `RemindersService` for `POST /notifications/{notification_id}/snooze`, `DELETE /notifications/{notification_id}/snooze`, and `only_snoozed` listing.
- `app/services/reminders_scheduler.py` loads enabled tasks, schedules one-time or cron triggers, and enqueues Jobs with domain `notifications` and type `reminder_due`.
- `app/services/reminder_jobs_worker.py` acquires and completes reminder Jobs.
- Reminder task mutations record companion activity through `Personalization.companion_activity` when the user has opted in.
- API documentation lives in `Docs/API-related/Reminder_Notifications_API.md`.

## Extension Points

- Add task fields in Collections DB, `reminders_schemas.py`, endpoint mapping, and service tests together.
- Change snooze behavior in `reminders_service.py` and notification endpoint tests.
- Change scheduler behavior in `app/services/reminders_scheduler.py`.
- Change due-notification creation in `reminder_jobs.py`.
- Extend notification preferences in `notifications.py` and `reminders_schemas.py`.

## Testing

- Service, schema, endpoint, scheduler, worker, and companion bridge tests live under `tldw_Server_API/tests/Notifications/`.
- Collections DB reminder persistence is covered by `tldw_Server_API/tests/Collections/test_reminders_notifications_db.py`.
- Scheduled task control-plane coverage also exercises reminders through `tldw_Server_API/tests/Notifications/test_scheduled_tasks_control_plane.py`.

## Gotchas

- Snooze windows are bounded from 1 minute to 10080 minutes.
- One-time reminders are disabled after a successful due job.
- Endpoint scheduler reconciliation is best-effort and logs a sanitized warning on failure.
- Current API documentation notes that active reminder task caps are not enforced at the API level.
