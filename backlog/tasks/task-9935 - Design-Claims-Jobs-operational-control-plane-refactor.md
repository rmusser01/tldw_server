---
id: TASK-9935
title: Design Claims Jobs operational control plane refactor
status: Done
created_date: 2026-06-24 18:27
labels:
- claims
- jobs
- refactor
- design
priority: high
references:
- tldw_Server_API/app/core/Claims_Extraction
- tldw_Server_API/app/core/Jobs
- Docs/ADR/003-jobs-vs-scheduler-default.md
updated_date: 2026-06-24 18:30
modified_files:
- Docs/superpowers/specs/2026-06-24-claims-jobs-operational-control-plane-design.md
- backlog/tasks/task-9935 - Design-Claims-Jobs-operational-control-plane-refactor.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write a staged refactor spec for moving Claims background rebuild and notification work onto the existing Jobs module, then expanding to all admin-visible Claims background work and recurring control-plane orchestration.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Spec documents a staged 1->2->3 Claims operational redesign using the existing Jobs module for queue/lifecycle mechanics.
- [x] #2 Spec keeps Claims ownership limited to domain contracts, enqueue helpers, validation, and handlers.
- [x] #3 Spec defines ID-only payloads, idempotency, retry/skipped semantics, owner-scoped DB resolution, worker startup flags, and dashboard/admin boundaries.
- [x] #4 Spec includes testing, migration, security, and rollout guidance.
- [x] #5 Spec is self-reviewed for placeholders, contradictions, scope drift, and ambiguity.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Started spec for staged Claims Jobs operational control plane refactor. User selected staged rollout 1->2->3 and confirmed Jobs module must own queue/lifecycle mechanics; Claims should only own domain contracts, enqueue helpers, validation, and handlers.
Wrote refactor spec at Docs/superpowers/specs/2026-06-24-claims-jobs-operational-control-plane-design.md. Self-review completed inline: no TODO/TBD placeholders, Jobs/Claims ownership is consistent, Stage 1 is implementation-sized, later stages are follow-ups, and payload/idempotency/retry/toggle/dashboard boundaries are explicit. Verification: docs-only change; git diff --check passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the staged Claims Jobs operational control plane refactor spec. The design uses the existing Jobs module for queue/lifecycle mechanics, keeps Claims ownership to domain contracts/enqueue helpers/handlers, and defines the 1->2->3 migration path, ID-only payloads, idempotency, retry/skipped semantics, owner-scoped DB resolution, worker flags, dashboard/admin boundaries, testing, security, and rollout guidance.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Backlog task created and updated.
- [x] #2 Spec written under Docs/superpowers/specs.
- [x] #3 Spec self-review completed inline.
- [x] #4 Spec committed separately from implementation.
- [x] #5 No unrelated generated files staged.
<!-- DOD:END -->
