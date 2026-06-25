---
id: TASK-10004
title: Harden Notifications module review findings
status: Done
assignee: []
created_date: 2026-06-23 21:44
updated_date: 2026-06-24 20:11
labels:
- notifications
- security
- review-fix
dependencies: []
priority: high
modified_files:
- IMPLEMENTATION_PLAN_notifications_review_hardening_10004.md
- backlog/tasks/task-10004 - Harden-Notifications-module-review-findings.md
- tldw_Server_API/app/api/v1/endpoints/watchlists.py
- tldw_Server_API/app/core/Notifications/README.md
- tldw_Server_API/app/core/Notifications/email_delivery.py
- tldw_Server_API/app/core/Notifications/service.py
- tldw_Server_API/tests/Notifications/test_email_delivery.py
- tldw_Server_API/tests/Notifications/test_notifications_service.py
- tldw_Server_API/tests/Watchlists/test_delivery_integrations.py
- tldw_Server_API/tests/Watchlists/test_watchlists_operator_recovery.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify and fix validated Notifications module review findings. Scope: bounded email fanout and attachments, redacted delivery failure reporting, duplicate SMTP helper cleanup/delegation, explicit SMTP timeout handling, and README scope accuracy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Validated review findings are fixed or explicitly documented as not applicable
- [x] #2 Focused regression tests cover each behavior change
- [x] #3 Touched Notifications tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
- Add failing regression tests for recipient fanout limits, attachment limits, and redacted failure results.
- Add failing regression tests for email_delivery delegation/config timeout behavior.
- Implement the smallest compatible fixes in the Notifications module.
- Update the Notifications README to match actual core package ownership.
- Run focused tests plus Bandit on touched production files and record results.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added service-level recipient normalization, dedupe, validation, and fanout limits before email delivery.
- Added attachment count, filename, and total byte checks before delivery.
- Changed email result details and failure logs to use masked recipients and exception type only; raw exception strings, subjects, and full recipient addresses are no longer returned by `NotificationsService`.
- Addressed PR review feedback after rebasing on `origin/dev`: email delivery failures now use bound Loguru context plus a redacted traceback-bearing exception object.
- Rebased again on the latest `origin/dev` on 2026-06-24 (`3a3ed8042`); the final base movements did not touch Notifications/Watchlists paths, so focused tests and static checks were re-run on that base.
- Added explicit attachment `content` presence/type validation before delivery to keep malformed attachments from reaching AuthNZ SMTP delivery.
- Added a shared Watchlists email attachment filename builder and routed both create-output and retry-delivery paths through it so `/`, `\`, control characters, and overlong titles do not produce invalid notification attachments.
- Lazily initialize the AuthNZ email service so non-email notification flows do not require email/AuthNZ configuration at service construction time.
- Preserved the public `send_notification_email()` helper while removing its duplicate raw SMTP sender path; it now delegates to AuthNZ `EmailService`.
- Kept the legacy SMTP config compatibility helper and added explicit positive `SMTP_TIMEOUT` parsing.
- Updated the core Notifications README to describe actual package ownership and safe delivery boundaries.
<!-- SECTION:NOTES:END -->

## Verification

<!-- SECTION:VERIFICATION:BEGIN -->
- Red step confirmed: focused Notifications tests initially failed for too many recipients, oversized attachments, raw failure details, missing SMTP timeout config, and raw SMTP helper behavior.
- `source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && python -m pytest --confcutdir=tldw_Server_API/tests/Notifications tldw_Server_API/tests/Notifications/test_notifications_service.py tldw_Server_API/tests/Notifications/test_email_delivery.py -q` - 12 passed.
- `SINGLE_USER_TEST_API_KEY=test-key-for-notifications-pr /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest --confcutdir=tldw_Server_API/tests/Notifications tldw_Server_API/tests/Notifications/test_notifications_service.py tldw_Server_API/tests/Notifications/test_email_delivery.py tldw_Server_API/tests/Watchlists/test_delivery_integrations.py tldw_Server_API/tests/Watchlists/test_watchlists_operator_recovery.py -q` - 30 passed.
- `SINGLE_USER_TEST_API_KEY=test-key-for-notifications-pr /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest --confcutdir=tldw_Server_API/tests/Notifications tldw_Server_API/tests/Notifications/test_companion_reflection_notifications.py tldw_Server_API/tests/Notifications/test_reminder_jobs_worker.py tldw_Server_API/tests/Notifications/test_notifications_sse.py tldw_Server_API/tests/Notifications/test_jobs_notifications_service.py tldw_Server_API/tests/Notifications/test_email_delivery.py tldw_Server_API/tests/Notifications/test_reminders_service.py tldw_Server_API/tests/Notifications/test_scheduled_task_automation_db.py tldw_Server_API/tests/Notifications/test_bridge_opt_out.py tldw_Server_API/tests/Notifications/test_reminders_api.py tldw_Server_API/tests/Notifications/test_reminders_scheduler.py tldw_Server_API/tests/Notifications/test_scheduled_task_automation_service.py tldw_Server_API/tests/Notifications/test_notifications_service.py tldw_Server_API/tests/Notifications/test_notifications_api.py tldw_Server_API/tests/Notifications/test_reminders_schemas.py tldw_Server_API/tests/Notifications/test_notifications_service_lifecycle.py tldw_Server_API/tests/Notifications/test_notifications_prune_service.py tldw_Server_API/tests/Notifications/test_companion_reminders_activity_bridge.py -q` - 125 passed.
- `SINGLE_USER_TEST_API_KEY=test-key-for-notifications-pr /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/Notifications/test_scheduled_task_automation_api.py tldw_Server_API/tests/Notifications/test_scheduled_tasks_control_plane.py -q` - 46 passed.
- `/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m ruff check tldw_Server_API/app/core/Notifications/service.py tldw_Server_API/app/core/Notifications/email_delivery.py tldw_Server_API/tests/Notifications/test_notifications_service.py tldw_Server_API/tests/Notifications/test_email_delivery.py` - passed.
- `/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m py_compile tldw_Server_API/app/core/Notifications/service.py tldw_Server_API/app/core/Notifications/email_delivery.py tldw_Server_API/app/api/v1/endpoints/watchlists.py` - compiled; Watchlists emitted two pre-existing `return` in `finally` syntax warnings outside this change.
- `/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m bandit -r tldw_Server_API/app/core/Notifications/service.py tldw_Server_API/app/core/Notifications/email_delivery.py tldw_Server_API/app/api/v1/endpoints/watchlists.py -f json -o /tmp/bandit_notifications_pr2493_rebase_20260624_3a3ed8042.json` - 0 findings, 0 errors.
- `git diff --check` - passed.
- A full `ruff check` over the large Watchlists endpoint still reports pre-existing import-order/SIM114/B009 findings unrelated to this PR; the scoped Notifications lint check passed.
- Running all `tldw_Server_API/tests/Notifications` under `--confcutdir=tldw_Server_API/tests/Notifications` produced fixture errors in scheduled-task API/control-plane tests because that boundary hides the shared `client_user_only` fixture. Those two files passed when run without the conftest boundary.
<!-- SECTION:VERIFICATION:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Hardened the Notifications core package by bounding email fanout and attachments, redacting delivery failure details, consolidating legacy notification email delivery through AuthNZ, and correcting package documentation. Rebased on latest `origin/dev` and addressed all current PR review comments, including structured redacted traceback logging, attachment content schema validation, Watchlists attachment filename sanitization, `memoryview.nbytes` attachment sizing, and ASCII control-character filename rejection. Added focused regression tests for the validated review findings and verified the broader Notifications suite using the appropriate fixture setup for scheduled-task API/control-plane tests.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-06-24 13:00: Addressed two additional CodeRabbit review comments after the latest rebase check: `_attachment_content_size_bytes()` now uses `memoryview.nbytes` so non-byte memoryviews are measured by bytes, and attachment filename validation now rejects all ASCII control characters including NUL, unit separator, and DEL. Added focused regression coverage for both cases. Fresh verification before amend: focused Notifications/Watchlists delivery slice passed with 34 tests, broader non-scheduled Notifications slice passed with 129 tests, scoped Ruff passed, `py_compile` passed with the same pre-existing Watchlists `return`-in-`finally` warnings, `git diff --check` passed, and Bandit report `/tmp/bandit_notifications_pr2493_coderabbit_final.json` produced 0 findings.
2026-06-24 13:06 final verification after rebasing onto `origin/dev` at `7ab6ae8c4`: focused delivery slice passed (`34 passed, 86 warnings`), broader non-scheduled Notifications slice passed (`129 passed, 661 warnings`), scheduled API/control-plane slice passed (`46 passed, 1534 warnings`), scoped Ruff passed, `py_compile` passed with the same pre-existing Watchlists `return`-in-`finally` warnings, `git diff --check` passed, and Bandit report `/tmp/bandit_notifications_pr2493_rebased_final.json` produced 0 findings. The branch is 0 behind and 1 ahead of `origin/dev` after rebase.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
