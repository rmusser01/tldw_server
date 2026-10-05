---
id: TASK-13449
title: Retire USER_DB_BASE legacy alias in db_path_utils
status: To Do
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up from TASK-13444 Step 9 decision gate. Evidence 2026-10-02: USER_DB_BASE referenced by 8+ production modules (config.py, Research/service.py, Embeddings/ChromaDB_Library.py, Embeddings/services/jobs_worker.py, Evaluations/embeddings_abtest_service.py, Setup/setup_manager.py, Workflows/adapters/utility/misc.py) beyond db_path_utils itself; allow_legacy_alias=True at db_path_utils.py:164 with resolver paths at :442/:471/:502. Removal requires migrating all consumers to the canonical base-dir resolution first - out of TASK-13444 scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Renumbered 2026-10-04 from TASK-13411 when PR #3155 was rebased onto dev: the branch was cut from codex/post2970-uat-20260920, and dev had meanwhile given that id to an unrelated task. Branch commit messages and code references use the new id. The stage plan (Docs/Plans/2026-10-01-due-debt-sweep-implementation-plan.md) exists only on that unmerged branch.
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
