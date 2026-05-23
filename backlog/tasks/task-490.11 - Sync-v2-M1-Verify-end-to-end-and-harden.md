---
id: TASK-490.11
title: 'Sync v2 M1: Verify end to end and harden'
status: To Do
assignee:
- '@Codex'
labels:
- sync
- sync-v2
- m1
- verification
priority: high
parent_task_id: TASK-490
documentation:
- Docs/superpowers/specs/2026-05-23-chatbook-sync-v2-roadmap-prd-design.md
- Docs/superpowers/plans/2026-05-23-chatbook-sync-v2-m1-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Run final Sync v2 M1 verification and hardening, including two-device scenarios, server-front-end writes, restore previews, conflicts, tombstones, attachment refs, cross-user isolation, targeted tests, broader relevant tests, Bandit, and final documentation/backlog updates.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 E2E scenario matrix covers two devices, server-origin writes, clean restore, non-empty conflicts, stable-message dedupe/conflicts, tombstones, attachment refs, and cross-user isolation.
- [ ] #2 Targeted Sync and ChaChaNotes tests pass or documented pre-existing failures are recorded.
- [ ] #3 Bandit runs on all touched production scope with no new findings.
- [ ] #4 Backlog child tasks record touched files, verification, skips, and final summaries.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-23-chatbook-sync-v2-m1-implementation-plan.md#task-11-end-to-end-verification-and-hardening
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
