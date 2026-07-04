---
id: TASK-12156
title: Avoid mutating Monitoring generic notification payloads
status: Done
created_date: 2026-07-04 20:51
labels:
- monitoring
- tests
- stability
priority: medium
modified_files:
- tldw_Server_API/app/core/Monitoring/notification_service.py
updated_date: 2026-07-04 22:01
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix NotificationService.notify_generic so it adds timestamps and redacts sensitive data on an internal copy, without mutating the caller-provided payload.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 notify_generic leaves caller payload unchanged when adding a timestamp
- [x] #2 Stored JSONL and queued webhook payloads still include timestamp and redacted sensitive values
- [x] #3 Focused notification service test passes
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Changed NotificationService.notify_generic to copy the caller-provided payload before adding a timestamp and sanitizing it. Focused verification: `python -m pytest -q tldw_Server_API/tests/Monitoring/test_notification_service.py::test_notification_service_records_generic_notifications` -> 1 passed. Order-sensitive neighborhood -> 83 passed. Changed-scope slice passed later: 1838 passed, 54 skipped, 1 xfailed, 2 xpassed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Prevented generic notification recording from mutating caller-owned payload dictionaries while preserving recorded timestamp behavior.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Focused test passes
- [x] #2 Order-sensitive neighborhood rerun passes
- [x] #3 Backlog task updated with verification results
<!-- DOD:END -->
