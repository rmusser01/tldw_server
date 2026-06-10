---
id: TASK-2351
title: Implement Scheduled Tasks Phase 4B backend API foundation
status: In Progress
labels:
- scheduled-tasks
- api
- backend
- frontend
priority: high
references:
- TASK-2350
- TASK-2349
documentation:
- Docs/superpowers/plans/2026-06-09-scheduled-tasks-phase4b-backend-api-foundation-implementation-plan.md
- Docs/superpowers/specs/2026-06-09-scheduled-tasks-phase4b-backend-api-foundation-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute the reviewed Scheduled Tasks Phase 4B implementation plan: durable automation definitions, previews, lifecycle, audit, idempotency, control-plane projection, and reference WebUI client without execution.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Capability, preview, definition, lifecycle, audit, idempotency, and projection backend APIs are implemented without execution.
- [ ] #2 Agent Task raw message text is redacted from preview, definition, list/detail, audit, and persisted JSON by default.
- [ ] #3 `/api/v1/scheduled-tasks` projects `automation_definition` rows without breaking reminder or Watchlists behavior.
- [ ] #4 WebUI reference client can preview, create, inspect, edit, pause/resume, archive, and duplicate definitions using the API.
- [ ] #5 Focused backend and frontend tests pass for the touched scope.
- [ ] #6 Bandit passes for touched backend scope or any findings are fixed.
- [ ] #7 No Jobs enqueueing, Scheduler integration, RAG execution, ACP dispatch, notifications, fake runs, fake results, or fake Home items are implemented.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implementation will follow `Docs/superpowers/plans/2026-06-09-scheduled-tasks-phase4b-backend-api-foundation-implementation-plan.md` using subagent-driven development with TDD and review checkpoints per task.
Task 1 completed in commit 8e2dd7b1fc: added automation schemas, route skeletons, capability service, primitive extension, and route/capability tests.
Task 1 verification: focused pytest for `test_scheduled_task_automation_api.py` and `test_scheduled_tasks_control_plane.py` passed with 6 tests; touched-scope Bandit reported zero findings; git diff --check was clean.
Task 1 review: spec-compliance reviewer approved; code-quality reviewer approved with one minor suggestion to add extra 501 skeleton assertions if useful before later endpoint wiring.
Task 2 completed across commits 84a7007f37, d6f528ba1a, and cea06ef558: added the per-user SQLite repository, repository tests, and review-driven hardening for optimistic locking, preview ownership, single-use preview consumption, idempotency expiry/reuse, shared SQLite policy, strict JSON encoding, atomic create-from-preview, owner-checked update preview references, and owner-checked audit writes.
Task 2 verification: repository pytest passed with 19 tests; formatting, py_compile, Bandit touched-scope scan, and git diff --check passed in the worker verification.
Task 2 review: spec-compliance reviewer approved. Code-quality review initially found repository lifecycle/ownership issues; worker fixed them in two follow-up commits. Final code-quality re-review approved with no remaining issues.
Task 3 completed across commits 3a10dd3723, 4ca66f690, and 126f93fb38: implemented preview validation, Agent Task redaction, definition create/update lifecycle, pause/resume/archive, duplicate, service idempotency replay/conflict behavior, and review-driven repository write transactions for atomic command units.
Task 3 verification: service pytest passed with 26 tests; repository and API pytest passed with 21 tests; git diff --check and Bandit touched-scope scan passed in worker verification.
Task 3 review: spec-compliance review initially found stale idempotency replay responses; worker fixed this with response snapshots. Code-quality review then found idempotency race and multi-step atomicity risks; worker fixed them with immediate write transactions and rollback/race coverage. Final spec and code-quality re-reviews approved.
Task 4 completed across commits f3050f3c13, 688ce4c5e5, and b6d8caef1c: exposed preview, definition, lifecycle, duplicate, audit, idempotency, filter, and public error-envelope API endpoints with owner scoping and no execution behavior.
Task 4 verification: endpoint/control-plane pytest passed with 34 tests; service/db pytest passed with 45 tests; targeted endpoint pytest passed with 30 tests; git diff --check and Bandit touched-scope scan passed in worker/reviewer verification.
Task 4 review: spec-compliance review found a missing direct public `scheduled_task_definition_not_found` alias and worker fixed it. Code-quality review found audit request-id and datetime filter correctness issues and worker/coordinator fixed them. Final spec and code-quality re-reviews approved.
Task 5 completed in commit 5f7f6cd632: projected automation definitions into the unified `/api/v1/scheduled-tasks` control-plane list/detail responses while preserving reminder and Watchlists behavior.
Task 5 verification: control-plane pytest passed with 7 tests; automation API pytest passed with 30 tests; git diff --check and Bandit touched-scope scan passed in worker/reviewer verification.
Task 5 review: spec-compliance reviewer approved. Code-quality reviewer approved with only residual optional coverage suggestions for disabled lifecycle and cross-user/missing detail behavior.
Task 6 completed across commits e27c3ec169, e269e62004, and a0a2485fec: added frontend automation client types/methods, automation definition status/type helpers, projected-id normalization, idempotency header support, and backend-aligned status handling.
Task 6 verification: targeted Vitest helper/client suite passed with 34 tests; git diff --check passed in worker/reviewer verification.
Task 6 review: spec-compliance reviewer approved. Code-quality review found projected-id normalization, unknown-status, and client-method coverage gaps; worker fixed them and aligned helper statuses with backend projection. Final spec and code-quality re-reviews approved.
Task 7 completed across commits 4f3924e128, 18cd196bb5, and 27afbc9dc2: wired the WebUI reference client for automation definition preview/create/update, lifecycle actions, duplicate, detail/audit display, real-results-only row actions, and API error display without execution behavior.
Task 7 verification: focused Scheduled Tasks Vitest suite passed with 93 tests; git diff --check and touched-scope Bandit scans passed in worker/reviewer verification.
Task 7 review: spec-compliance review initially found missing update reachability and row-level fake Results actions; worker fixed both. Code-quality review found schedule/agent_ref contract, stale preview, JSON validation, and API error parsing issues; worker fixed them. Final spec and code-quality re-reviews approved.
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
