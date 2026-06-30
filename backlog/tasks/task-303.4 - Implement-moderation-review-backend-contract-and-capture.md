---
id: TASK-303.4
title: Implement moderation review backend contract and capture
status: Done
assignee: []
created_date: '2026-05-12 23:07'
updated_date: '2026-05-12 23:26'
labels:
  - moderation
  - backend
  - authnz
dependencies:
  - TASK-303.3
documentation:
  - >-
    Docs/superpowers/plans/2026-05-12-moderation-review-rules-remediation-implementation-plan.md
  - >-
    Docs/superpowers/specs/2026-05-12-moderation-review-rules-remediation-design.md
parent_task_id: TASK-303
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Stage 4 of the moderation review/rules remediation plan. Add backend moderation review permissions, sanitized review schemas, a SQLite-backed review store/service, gated event capture from moderation decisions, review/audit endpoints, and OpenAPI guard coverage without changing the Stage 5 UI scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Review permissions are defined, seeded for SQLite/Postgres RBAC, and included in single-user defaults as specified.
- [x] #2 Rules/config endpoints keep system configuration authorization while review endpoints use review-specific read/decide/bulk/audit permissions.
- [x] #3 Sanitized review item, decision, bulk, and audit schemas support list/detail/decision/undo/bulk/audit endpoint responses without trusting actor fields from request bodies.
- [x] #4 Review store creates schema, supports idempotent insert, filtering, pagination, decision, undo, bulk partial failure, audit ordering, and content redaction.
- [x] #5 Review service maps decision actions to statuses, records authenticated actors, produces audit events, and returns safe payloads only.
- [x] #6 Moderation event capture is gated off by default and can create idempotent sanitized review items for block/redact/warn moderation outcomes when enabled.
- [x] #7 Focused backend tests and OpenAPI guard coverage are added and pass, with known skips/blockers documented.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented Stage 4 backend moderation review contract: review permissions/RBAC seed defaults, sanitized schemas, SQLite review store/service, review/audit endpoints, OpenAPI guard entries, and gated capture hooks for chat moderation outcomes. The shared rbac_seed.py bootstrap covers Postgres and SQLite baseline RBAC; no pg_migrations_extra.py change was needed.

Verification: focused Stage 4 pytest passed 21 tests; existing chat moderation integration passed 15 tests; bun run verify:openapi passed with existing reviewed exception paths; py_compile passed for touched backend modules; Bandit on touched backend code reported no findings; git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stage 4 adds the backend contract needed for the moderation review queue: explicit review permissions, a sanitized durable review store/service with audit trail and undo, review-specific endpoints, gated/idempotent capture from moderation outcomes, and OpenAPI guard coverage. Capture remains off by default via MODERATION_REVIEW_CAPTURE_ENABLED until the Stage 5 UI and retention behavior are wired.
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
