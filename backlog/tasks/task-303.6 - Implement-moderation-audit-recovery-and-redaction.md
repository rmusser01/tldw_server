---
id: TASK-303.6
title: Implement moderation audit recovery and redaction
status: Done
assignee: []
created_date: '2026-05-13 00:58'
updated_date: '2026-05-13 01:08'
labels:
  - moderation
  - webui
  - backend
  - frontend
dependencies: []
documentation:
  - >-
    Docs/superpowers/plans/2026-05-12-moderation-review-rules-remediation-implementation-plan.md
parent_task_id: TASK-303
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Stage 6 of the moderation remediation plan. Make moderation review decisions trustworthy, reversible when eligible, auditable, and privacy-preserving. Keep review data sanitized and avoid exposing raw unsafe content.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Review item detail includes sanitized decision history with actor, action, resulting status, reason, timestamps, undo eligibility, and redaction state.
- [x] #2 Undo tokens are hashed, expire, are single-use, and fail when a later decision supersedes the original decision.
- [x] #3 Audit list supports filters for item, decision, actor, action, date range, cursor, and limit.
- [x] #4 Redacted review items preserve metadata and audit history while replacing excerpt, context, and match samples with safe placeholders.
- [x] #5 Backend and frontend tests cover audit timeline, undo edge cases, and redacted content states.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Stage 6 implemented. Focused backend verification: python -m pytest test_moderation_review_store.py test_moderation_review_service.py test_moderation_review_audit.py -q => 12 passed. Focused frontend verification: vitest AuditTimeline.test.tsx ModerationReviewShell.test.tsx => 8 passed. Audit export was not added because no export endpoint exists; filtered audit listing is the documented v1 surface pending Stage 8 docs.

Stage 6 security scan: bandit on moderation review store/service/schemas/endpoint wrote /tmp/bandit_moderation_stage6.json and exited 0 after suppressing a false-positive B105 on the explicit null undo_token response field.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added sanitized decision history, hashed/expiring single-use undo tokens, superseded undo conflict handling, audit filters, content redaction placeholders, and the Review detail audit timeline.
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
