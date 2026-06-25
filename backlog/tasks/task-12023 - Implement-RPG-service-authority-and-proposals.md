---
id: TASK-12023
title: Implement RPG service authority and proposals
status: Done
created_date: 2026-06-25 03:39
labels:
- rpg
- ttrpg
- backend
- implementation
- service
priority: high
references:
- TASK-12018
- TASK-12019
- TASK-12020
- TASK-12021
- TASK-12022
documentation:
- Docs/superpowers/plans/2026-06-25-rpg-campaign-session-runtime-implementation-plan.md
updated_date: 2026-06-25 03:51
modified_files:
- tldw_Server_API/app/core/RPG/authority.py
- tldw_Server_API/app/core/RPG/proposals.py
- tldw_Server_API/app/core/RPG/service.py
- tldw_Server_API/app/core/RPG/__init__.py
- tldw_Server_API/app/core/DB_Management/RPG_DB.py
- tldw_Server_API/tests/RPG/test_rpg_service.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the RPG service authority/proposal slice from the reviewed plan: authority decisions, service orchestration, repository proposal methods, snapshot access, and focused service tests. Scope excludes REST API, MCP, rules context, and frontend/UI work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Model-sourced events create pending proposals by default
- [x] #2 User-sourced events commit directly and update the snapshot
- [x] #3 Applying a proposal atomically commits its events and advances the snapshot once
- [x] #4 Repository proposal methods are owner-scoped and idempotent where applicable
- [x] #5 Focused service tests are written test-first and pass
- [x] #6 Bandit/diff checks are recorded
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Write failing tests for model proposal flow, user direct commit, and proposal apply.
2. Implement authority.py, proposals.py, service.py, and repository proposal helpers using existing RPG events/reducer/repository contracts.
3. Run focused tests plus adjacent RPG tests, compileall, Bandit, and diff checks.
4. Record modified files and final notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented service authority/proposal orchestration and repository proposal persistence. Review added coverage for idempotent proposal rejection in addition to idempotent proposal apply. Verification: service tests passed (5 passed); adjacent focused RPG suite passed (35 passed); compileall passed; Bandit on core RPG/DB touched scope reported 0 results; git diff --check passed. Note: worker-created tests were already green by the time the implementation was reviewed, so RED output for this slice was not captured in the task record.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added RPG authority decisions, proposal record mapping, service orchestration for campaign/session/event workflows, and proposal create/apply/reject/conflict persistence. Model events now create pending proposals by default; user events commit directly through the reducer and repository; proposal apply/reject flows are idempotent and owner-scoped.
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
