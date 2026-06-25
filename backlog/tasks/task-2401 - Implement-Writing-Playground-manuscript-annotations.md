---
id: TASK-2401
title: Implement Writing Playground manuscript annotations
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-06-23 15:03'
labels:
  - implementation
  - webui
  - extension
  - writing-playground
  - manuscripts
dependencies: []
documentation:
  - >-
    Docs/superpowers/plans/2026-06-23-writing-playground-manuscript-annotations-implementation-plan.md
  - >-
    Docs/superpowers/specs/2026-05-24-writing-playground-manuscript-annotations-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute Docs/superpowers/plans/2026-06-23-writing-playground-manuscript-annotations-implementation-plan.md task-by-task using subagent-driven development and TDD. Start with Task 1 pure annotation anchor helpers, then run spec and code-quality reviews before advancing.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Task 1 complete: added pure manuscript annotation anchor constants/helpers and tests.

TDD evidence:
- Initial red run failed as expected before implementation.
- Review-fix red run failed as expected for absent selected-text context recovery and malformed scene_version handling.
- Green run: `source ../../.venv/bin/activate && python -m pytest tldw_Server_API/tests/Writing/test_manuscript_annotations_anchor.py -q` -> 16 passed, 6 warnings.

Review evidence:
- Spec review approved after re-checking the keyword-only API contract in the plan.
- Code-quality review found two anchor hardening issues; both were fixed.
- Code-quality re-review approved.

Security/static checks:
- Bandit on `tldw_Server_API/app/core/Writing/manuscript_annotations.py` wrote `/tmp/bandit_manuscript_annotations_task1_verify_after_fix.json` with no findings.
- `git diff --check HEAD~2..HEAD` passed.

Task 2 complete: added manuscript annotation persistence schema, migrations, DB helper methods, and regression tests.

TDD evidence:
- Initial DB red run failed as expected before the schema/helper implementation.
- Review-fix red run failed as expected for duplicate suppression, anchor_status constraints/filter validation, and structured tags/metadata validation.
- Green run: `../../.venv/bin/python -m pytest tldw_Server_API/tests/Writing/test_manuscript_annotations_db.py tldw_Server_API/tests/Writing/test_manuscript_annotations_anchor.py -q` -> 31 passed, 6 warnings.

Review evidence:
- Spec review initially found anchor_status filtering after SQL LIMIT could return incomplete results.
- Fix commit `68f2c9f412` filters derived anchor status before pagination and enforces the unbounded candidate cap.
- Spec re-review approved.
- Code-quality review found duplicate suppression ignored anchor identity and schema lacked anchor_status CHECK constraints.
- Fix commit `6766200de7` added anchor identity duplicate keys, SQLite/PostgreSQL anchor_status CHECK constraints, anchor_status filter validation, and structured tags/metadata validation.
- Code-quality re-review approved with no remaining Critical or Important issues.

Security/static checks:
- Bandit on `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py` and `tldw_Server_API/app/core/DB_Management/ManuscriptDB.py` wrote `/tmp/bandit_task_2402.json` with no findings.
- `git diff --check origin/dev..HEAD` passed before the Task 2 review-fix commit and will be rerun before the next task starts.

Task 3 complete: added manuscript annotation API schemas and CRUD/list endpoints.

TDD evidence:
- API red run failed as expected before routes existed: `../../.venv/bin/python -m pytest tldw_Server_API/tests/Writing/test_manuscript_annotations_api.py -q` -> 8 failed, 1 passed.
- Review-fix red run failed as expected for stale annotation exposure after target deletion.
- Green run: `../../.venv/bin/python -m pytest tldw_Server_API/tests/Writing/test_manuscript_annotations_api.py tldw_Server_API/tests/Writing/test_manuscript_annotations_db.py tldw_Server_API/tests/Writing/test_manuscript_annotations_anchor.py tldw_Server_API/tests/Writing/test_writing_error_mapping.py -q` -> 59 passed, 6 warnings.

Review evidence:
- Spec review approved the schema literals, create/update/response/list schemas, five endpoints, RBAC scopes, expected-version headers, and required API coverage.
- Code-quality review found existing annotations could still be exposed after their scene/chapter targets were soft-deleted.
- Fix commit `d47dcd70e0` added a shared active-target predicate to `get_annotation()` and `list_annotations()` plus DB/API regressions.
- Code-quality re-review approved with no remaining Critical or Important issues.

Security/static checks:
- Bandit on Task 3 endpoint/schema/helper touched scope wrote `/tmp/bandit_task_2401_task3_review_fix.json` with no findings.
- `git diff --check HEAD~1..HEAD` passed for the Task 3 review-fix commit.

Task 4 complete: added selected-text AI annotation review prompt/parsing helpers, request schema, API endpoint, and regression tests.

TDD evidence:
- API red run failed as expected before the endpoint existed: `../../.venv/bin/python -m pytest tldw_Server_API/tests/Writing/test_manuscript_annotations_api.py -q` -> 7 failed, 10 passed.
- Green run: `../../.venv/bin/python -m pytest tldw_Server_API/tests/Writing/test_manuscript_annotations_api.py tldw_Server_API/tests/Writing/test_manuscript_analysis_integration.py -q` -> 43 passed, 7 warnings.

Review evidence:
- Spec review approved the explicit provider/model contract, scene version/range/selected-text conflict handling, parse-before-persist behavior, `ai_selected_text` source, and no manuscript mutation guarantee.
- Code-quality review approved with no Critical, Important, or Minor issues.
- Reviewer follow-up recommendations were non-blocking: add future coverage for out-of-bounds ranges and multi-annotation model output, and consider prompt/context hardening for selections beyond truncated scene text.

Security/static checks:
- Bandit on Task 4 endpoint/schema/helper touched scope wrote `/tmp/bandit_task_2401_task4_controller.json` with no findings.
- `git diff --check HEAD~1..HEAD` passed for the Task 4 implementation commit.

Task 5 complete: added Jobs-backed full-scene manuscript annotation review helpers, API endpoint, worker processor/service, startup registration, and regression tests.

TDD and review-fix evidence:
- Initial Task 5 target suites passed after implementation: `../../.venv/bin/python -m pytest tldw_Server_API/tests/Writing/test_manuscript_annotation_review_jobs.py tldw_Server_API/tests/Writing/test_manuscript_annotations_api.py -q` -> 28 passed, 5 warnings; `../../.venv/bin/python -m pytest tldw_Server_API/tests/Services/test_writing_annotation_review_jobs_worker.py tldw_Server_API/tests/Services/test_startup_primary_jobs_pollers.py -q` -> 33 passed, 3 warnings.
- Review-fix red tests failed as expected for missing worker job_type filtering, cross-owner idempotency collision, same-batch duplicate suppression, acquire_next_jobs job_type forwarding, and transient provider retryability.
- Final Task 5 green run: `../../.venv/bin/python -m pytest tldw_Server_API/tests/Writing/test_manuscript_annotation_review_jobs.py tldw_Server_API/tests/Writing/test_manuscript_annotations_db.py tldw_Server_API/tests/Jobs/test_jobs_manager_acquire.py tldw_Server_API/tests/Writing/test_manuscript_annotations_api.py tldw_Server_API/tests/Services/test_startup_primary_jobs_pollers.py tldw_Server_API/tests/Jobs/test_worker_sdk.py tldw_Server_API/tests/Services/test_writing_annotation_review_jobs_worker.py -q` -> 94 passed, 5 warnings.

Review evidence:
- Spec review initially found the worker leased all writing-domain jobs before validating job_type; fix commit `030135992e` added job_type filtering through JobManager, WorkerSDK, and the writing review worker.
- Spec re-review approved the full Task 5 range after the acquisition fix.
- Code-quality review found cross-user idempotency collisions, same-batch duplicate persistence, acquire_next_jobs inconsistency, and transient provider retry classification gaps.
- Fix commits `26b223c503` and `d7d10a014e` owner-scoped scene-review idempotency with a digest, dedupe retained candidates in memory, thread job_type through batch acquisition, and mark Chat rate limits/upstream 408/429/5xx failures retryable while keeping auth/config/bad-request terminal.
- Final code-quality re-review approved with no remaining Critical, Important, or Minor issues.

Security/static checks:
- Bandit on Task 5 touched app files wrote `/tmp/bandit_task_2401_task5_after_filter_fix.json`, `/tmp/bandit_task_2401_task5_hardening_uncommitted.json`, and `/tmp/bandit_task_2401_task5_retry_classification.json` with empty results.
- `git diff --check 83293fb52a1bbb6f3464386ac028d848d8070c6e..HEAD` passed after the final Task 5 fix.

Task 6 complete: added frontend manuscript annotation service types and client methods.

TDD and verification evidence:
- Root requested Vitest command could not discover files in this `.worktrees` checkout because the active parent Vitest config excludes `.worktrees/**` and filters to unrelated calendar tests.
- Package-local red run failed as expected before implementation because the new service methods were undefined.
- Review-fix red run failed as expected on the source guard because `ManuscriptAnnotationUpdateInput` still allowed null for backend-rejected `status`, `category`, and `body`.
- Final package-local green run: `cd apps/packages/ui && bunx vitest run src/services/__tests__/writing-playground.annotations.test.ts src/components/Option/WritingPlayground/__tests__/WritingPlayground.manuscript-api-shapes.guard.test.ts` -> 2 files passed, 13 tests passed.

Review evidence:
- Spec review approved the seven service methods, exact endpoint paths, provider/model review payloads, explicit annotation contracts, and expected-version header usage.
- Code-quality review found `ManuscriptAnnotationUpdateInput` allowed null for fields the backend rejects and noted the annotation source guard was broader than needed.
- Fix commit `6520b5194b` made `status`, `category`, and `body` optional non-null fields while preserving nullable clearable fields, and narrowed the guard to individual exported type declarations.
- Code-quality re-review approved with no remaining Critical or Important issues.

Security/static checks:
- Bandit was skipped for Task 6 because the slice changed TypeScript/TS test files only and no Python files.
- `git diff --check HEAD~1..HEAD` passed after the Task 6 review fix.

Task 7 complete: bound the Writing Playground editor to active saved manuscript scenes while preserving session prompt/settings isolation.

TDD and verification evidence:
- Initial package-local hook test run failed as expected before `useActiveManuscriptScene` existed.
- Review-fix red tests failed as expected for session prompt reclaim, scene text leaking into session saves, stale scene transitions, pending generation/revision actions, scene-leave autosave restoration, revision queue mutation locks, and manuscript node switching during in-flight writing requests.
- Final focused green run: `cd apps/packages/ui && bunx vitest run src/components/Option/WritingPlayground/__tests__/useActiveManuscriptScene.test.tsx src/components/Option/WritingPlayground/__tests__/WritingPlayground.phase1-baseline.test.tsx src/components/Option/WritingPlayground/__tests__/WritingRevisionQueue.test.tsx` -> 3 files passed, 44 tests passed.

Review evidence:
- Spec review approved the saved-scene binding hook, hook export, editor wiring, scene save affordance, and dirty scene switch boundary after the implementation matched the Task 7 contract.
- Code-quality review found and fixed multiple scene/session ownership gaps: bound scenes could be reclaimed by session prompt sync, scene content could be saved into session prompts through settings/template/theme/session-payload saves, stale scene responses could stay bound during transitions, pending scene binding still allowed generation/revision entry points, leaving a scene during scene-bound autosave could leave scene text in session mode, revision queue mutations remained active during scene binding, and tree selection could change during in-flight generation/revision.
- Final code-quality re-review approved with no blockers; residual gap is only that the full UI suite was not run.

Security/static checks:
- Bandit was skipped for Task 7 because the slice changed TypeScript/TSX test files only and no Python files.
- `git diff --check 1fc61f26e4296b5c29d0dfca918a65ea9a1b581d..HEAD` passed after the final Task 7 review fix.
<!-- SECTION:NOTES:END -->

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
