---
id: TASK-12932
title: Fix Guardian generic notify timestamp regression on main CI
status: In Progress
labels:
- ci
- guardian
- main-followup
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Post-PR #2692 main CI fails gap-verified-7 on Python 3.12 and 3.13 because NotificationService.notify_generic records a copied payload with ts while the existing Guardian test expects the caller payload to receive the default timestamp. Prepare a minimal fix locally and keep it unpushed until the current CI run completes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] NotificationService.notify_generic adds a default ts to the caller payload when severity passes the threshold and no ts is present.
- [x] The existing sanitized copy is still used for persistence and delivery.
- [x] Regression test and touched-file Bandit verification pass locally.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Root cause: notify_generic copied the payload before applying the default ts, so the persisted record included ts but the caller-visible payload did not. The fix applies `payload.setdefault("ts", datetime.now(timezone.utc).isoformat())` immediately after threshold filtering, then continues copying and sanitizing the payload for notification persistence.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Prepared a minimal local fix for the main CI Guardian failure. Verified the exact failing test, the full Guardian comprehensive edge-case file, git diff --check, and Bandit on tldw_Server_API/app/core/Monitoring/notification_service.py. Patch remains unpushed pending completion of the requested CI run.
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
