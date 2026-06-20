---
id: TASK-2393
title: Address PR 2326 Explainer review feedback
status: In Progress
references:
- https://github.com/rmusser01/tldw_server/pull/2326
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rebase the Explainer workspace PR onto latest dev and address unresolved review threads from CodeRabbit, Gemini, and Qodo. Scope includes valid review fixes in backend routing/persistence, Chatbook export hardening, frontend query/UI behavior, E2E mocks, Backlog task metadata, and targeted verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Rebase PR #2326 branch onto latest `origin/dev` and adapt Explainer route registration to the dev router-groups architecture.
- [x] Address valid backend review comments for Explainer persistence, helper typing, job-status offloading, Chatbook export hardening, and grounding coercion.
- [x] Address valid frontend review comments for Explainer query invalidation, option access, E2E mocks, and detail-panel coverage.
- [x] Clean reviewed Backlog task metadata inconsistencies.
- [x] Run focused backend/frontend/E2E/security verification before push.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Rebasing onto `origin/dev` made the old `main.py` direct `_HAS_EXPLAINER` route-registration comments obsolete; Explainer now registers through `router_groups/content.py`.
- Moved the raw-SQL Explainer repository implementation under `core/DB_Management/Explainer_Repository.py` and kept `core/Explainer/repository.py` as a compatibility shim.
- Replaced Explainer DB `threading.local()` connection storage with context-local connections plus tracked all-connection cleanup for cache eviction/shutdown.
- Moved service exceptions to `core/exceptions.py`, added endpoint helper type hints, offloaded synchronous job lookup with `asyncio.to_thread`, and typed the worker lazy generator.
- Hardened Chatbook sync export persistence with service/request user ownership validation and sanitized `ExportError` endpoint handling.
- Fixed grounding citation coercion after string normalization, rejecting only missing/empty normalized fields.
- Moved Explainer job polling cache invalidation out of the query function, used index access for question option records, added detail-panel regression coverage, and widened the notes-search E2E mock to allow an optional trailing slash.
- Corrected TASK-546/TASK-547 DoD checkboxes and TASK-548 duplicated description sentinels.
- Verification so far: targeted Explainer/Chatbook pytest suite passed 52 tests; router/OpenAPI smoke passed 2 tests; Explainer Vitest suite passed 9 tests; Explainer Playwright E2E passed 3 tests; Bandit on touched backend scope reported 0 findings; `git diff --check` passed.
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
