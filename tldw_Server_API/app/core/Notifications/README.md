# Notifications

The Notifications module delivers user-facing events and generated outputs
through email, Chatbook persistence, notification APIs, SSE streams, reminder
bridges, and Jobs-backed notification workers. The small core package contains
delivery services; endpoint and scheduler packages provide the broader control
plane.

## Start Here

- Core services: `service.py` and `email_delivery.py`.
- API endpoint: `app/api/v1/endpoints/notifications.py`.
- Related reminder endpoint/services: `app/api/v1/endpoints/reminders.py`,
  `app/services/reminders_scheduler.py`, and
  `app/services/reminder_jobs_worker.py`.
- Tests: `tests/Notifications/`.

## Responsibilities

- Send email through the AuthNZ email service and report per-recipient status.
- Store generated content as Chatbook documents when requested by Watchlists or
  other workflows.
- Support notification API/service flows for lifecycle, pruning, SSE, reminders,
  and companion/reflection bridges.
- Keep delivery failures structured rather than raising raw provider exceptions
  into callers.

## Module Map

- `service.py` defines `NotificationsService` and `NotificationResult` for email
  and Chatbook delivery.
- `email_delivery.py` contains email delivery helper logic used by notification
  tests and services.

## How It Connects

- Watchlists uses Notifications to deliver report outputs by email or Chatbook.
- Reminders enqueue notification work through scheduler and worker services.
- AuthNZ owns email provider configuration and message sending.
- Chat document generation and ChaChaNotes DB persistence are used for Chatbook
  delivery.

## Extension Points

- Add a new delivery channel by adding a focused method returning
  `NotificationResult`, then wire it from endpoint/service code.
- Keep channel-specific provider credentials outside this module; use the owning
  integration or AuthNZ provider.

## Testing

- Core delivery: `tests/Notifications/test_notifications_service.py` and
  `tests/Notifications/test_email_delivery.py`.
- API/SSE/lifecycle: `tests/Notifications/test_notifications_api.py`,
  `tests/Notifications/test_notifications_sse.py`, and
  `tests/Notifications/test_notifications_service_lifecycle.py`.
- Reminder and bridge flows: `tests/Notifications/test_reminders_api.py`,
  `tests/Notifications/test_reminder_jobs_worker.py`, and
  `tests/Notifications/test_companion_reflection_notifications.py`.

## Gotchas

- Email delivery is often partially successful. Preserve per-recipient details
  when aggregating status.
- Chatbook delivery writes to the per-user ChaChaNotes DB; use temp DB fixtures
  in tests.
