# Notifications

The Notifications core package contains delivery helpers for user-facing
events and generated outputs. It handles bounded email delivery through AuthNZ
and Chatbook persistence. Endpoint, scheduler, reminder, SSE, and Jobs worker
packages own the broader notification control plane.

## Start Here

- Core services: `service.py` and `email_delivery.py`.
- API endpoint: `app/api/v1/endpoints/notifications.py`.
- Related reminder endpoint/services: `app/api/v1/endpoints/reminders.py`,
  `app/services/reminders_scheduler.py`, and
  `app/services/reminder_jobs_worker.py`.
- Tests: `tests/Notifications/`.

## Responsibilities

- Send bounded email through the AuthNZ email service and report masked
  per-recipient status.
- Store generated content as Chatbook documents when requested by Watchlists or
  other workflows.
- Provide formatting helpers for notification email content.
- Keep delivery failures structured rather than raising raw provider exceptions
  into callers.

## Module Map

- `service.py` defines `NotificationsService` and `NotificationResult` for email
  and Chatbook delivery.
- `email_delivery.py` contains email formatting and a compatibility
  `send_notification_email()` helper that delegates to AuthNZ email delivery.

## How It Connects

- Watchlists uses Notifications to deliver report outputs by email or Chatbook.
- Reminders enqueue notification work through scheduler and worker services
  outside this core package.
- AuthNZ owns email provider configuration and message sending.
- Chat document generation and ChaChaNotes DB persistence are used for Chatbook
  delivery.

## Extension Points

- Add a new core delivery channel by adding a focused method returning
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

- Email delivery is often partially successful. Preserve masked per-recipient
  details when aggregating status.
- Keep fanout and attachment limits at this service boundary so new callers
  inherit safe defaults.
- Chatbook delivery writes to the per-user ChaChaNotes DB; use temp DB fixtures
  in tests.
