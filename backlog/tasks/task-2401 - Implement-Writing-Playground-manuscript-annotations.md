---
id: TASK-2401
title: Implement Writing Playground manuscript annotations
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-06-25 04:18'
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

Task 8 complete: added the Writing Playground annotation inspector tab, compact annotation list/actions, range-anchor utilities, and `useWritingAnnotations` React Query hook using the existing annotation service functions.

TDD and verification evidence:
- Initial red run failed as expected before implementation because `writing-annotation-anchor-utils`, `useWritingAnnotations`, and `WritingAnnotationsTab` did not exist and the inspector lacked the Annotations tab.
- Code-quality review-fix red run failed as expected for scene annotation context falling back to project while a selected scene was binding, invalid selected-text AI review enablement, and selected-model provider resolution not enabling annotation AI affordances.
- Final focused green run from `apps/packages/ui`: `bunx vitest run src/components/Option/WritingPlayground/__tests__/writing-annotation-anchor-utils.test.ts src/components/Option/WritingPlayground/__tests__/useWritingAnnotations.test.tsx src/components/Option/WritingPlayground/__tests__/WritingAnnotationsTab.test.tsx src/components/Option/WritingPlayground/__tests__/WritingPlayground.inspector-tabs.test.tsx` -> 4 files passed, 28 tests passed.
- `bunx tsc --noEmit --pretty false` hit the default Node heap limit; rerun with `NODE_OPTIONS=--max-old-space-size=8192` completed and failed only on existing unrelated package errors in Notes, AudioStudio, ScheduledTasks, background, scheduled-task services, Dexie audiobook migration, and voice-cloning files. No touched Writing Playground files were reported in the final typecheck output.
- `git diff --check` passed.

Review evidence:
- Spec review approved the Task 8 file set, hook export, query/filter behavior, disabled query behavior, exact invalidation, inspector keyboard navigation, saved-scene gating, note creation paths, list actions, `needs_review` display, code-point conversion, context clamp, and disabled AI review state.
- Code-quality review found scene/project context leakage during scene binding, invalid selected-text AI review enablement, unresolved provider routing for annotation AI affordances, and a chapter/project default-target UX mismatch.
- Review-fix commits in this task added `resolveWritingAnnotationTargetContext`, reused range validation for selected-text AI review gating, resolved effective annotation providers through `resolveApiProviderForModel`, and defaulted note targets to the available active context.
- Code-quality re-review approved.

Security/static checks:
- Bandit skipped for Task 8 because this slice changed TypeScript/TSX/frontend test files only and no Python files.

Task 9 complete: added TipTap-backed manuscript annotation range measurement plus the Google Docs-style margin annotation rail for rich edit/split modes.

TDD and verification evidence:
- Initial red adapter/rail tests failed as expected before `measureRange`, margin card, and margin rail behavior existed.
- Spec-review red/fix cycles covered active card expansion/details, removal of layout-affecting transitions, and stable card/body/inspector row ARIA linkages.
- Code-quality review-fix red runs failed as expected for stable annotations not remeasuring on editor layout changes and for missing post-external-sync TipTap apply notification.
- Final focused green run from `apps/packages/ui`: `NODE_OPTIONS=--max-old-space-size=8192 bunx vitest run src/components/Option/WritingPlayground/__tests__/writing-editor-adapter.test.ts src/components/Option/WritingPlayground/__tests__/WritingAnnotationMarginRail.test.tsx src/components/Option/WritingPlayground/__tests__/WritingTipTapEditor.external-sync.test.tsx src/components/Option/WritingPlayground/__tests__/WritingPlayground.phase1-baseline.test.tsx` -> 4 files passed, 49 tests passed.
- Extension parity verification from `apps/extension`: `bunx playwright test tests/e2e/writing-playground-mode-parity.spec.ts --reporter=line` built the Chrome MV3 extension successfully and reported 4 skipped tests in this environment.
- `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false` still fails on existing unrelated package errors in Notes, AudioStudio, ScheduledTasks, Setup, Dexie audiobook migration, background, scheduled-tasks control-plane, and voice-cloning files. No touched Writing Playground Task 9 files were reported.
- `git diff --check` passed.

Review evidence:
- Spec review approved the optional TipTap-only `measureRange`, textarea measurement fallback, rail filtering to open scene range annotations, deterministic ordering/collision avoidance, active card affordances, responsive hiding without measurement/plain/preview, and stable margin-card-to-inspector ARIA IDs.
- Code-quality review found stale card positioning when editor content reflowed without scroll/resize; fix added a `measurementVersion` invalidation path and a regression test for stable annotation remeasurement.
- Code-quality re-review found the first external-content sync invalidation could fire before ProseMirror applied `setContent`; fix moved external-sync invalidation into `WritingTipTapEditor` via `onContentApplied` after `setContent(..., { emitUpdate: false })` and the next animation frame.
- Final code-quality re-review approved.

Security/static checks:
- Bandit skipped for Task 9 because this slice changed TypeScript/TSX/frontend E2E files only and no Python files.

Task 10 complete: wired selected-text review, scene-review job feedback, and suggested-fix handoff into the existing revision queue.

TDD evidence:
- Red run from `apps/packages/ui`: `bunx vitest run src/components/Option/WritingPlayground/__tests__/useWritingAnnotations.test.tsx src/components/Option/WritingPlayground/__tests__/WritingAnnotationsTab.test.tsx src/components/Option/WritingPlayground/__tests__/WritingAnnotationMarginRail.test.tsx` -> failed as expected with 7 failures for the old two-argument review hook API, missing scene-review job status display, and missing suggested-fix create-revision/manual-copy actions.
- Code-quality review-fix red run failed as expected for trimmed suggested-fix text in the annotation-to-revision handoff.
- Scene-review version reset red run failed as expected for stale queued job display after the scene version changed.
- Async scene-review race red run failed as expected for stale job display after an in-flight older-version job resolved.
- Final focused green run from `apps/packages/ui`: `bunx vitest run src/components/Option/WritingPlayground/__tests__/useWritingAnnotations.test.tsx src/components/Option/WritingPlayground/__tests__/WritingAnnotationsTab.test.tsx src/components/Option/WritingPlayground/__tests__/WritingAnnotationMarginRail.test.tsx` -> 3 files passed, 29 tests passed.
- Final baseline green run from `apps/packages/ui`: `NODE_OPTIONS=--max-old-space-size=8192 bunx vitest run src/components/Option/WritingPlayground/__tests__/WritingPlayground.phase1-baseline.test.tsx` -> 1 file passed, 32 tests passed.
- Route parity guard from `apps/tldw-frontend`: `bunx vitest run extension/__tests__/writing-playground-route-parity.guard.test.ts` -> 1 file passed, 1 test passed.

Implementation notes:
- `useWritingAnnotations` now exposes `reviewSelection(input)` and `reviewScene(input)` with `sceneId` inside the input object and forwards provider/model without defaults.
- `WritingAnnotationsTab` disables review actions while the scene is dirty, sends provider/model/scene version/range/selected text, displays queued scene-review job id/status, clears stale job state on scene version changes, and ignores older in-flight job responses after scene/version changes.
- Margin annotation suggested fixes now show `Create revision` for attached/reattached anchors and `Copy fix manually` for `needs_review` anchors.
- `index.tsx` bridges stable suggested fixes into `useWritingRevisions` by creating a pending replacement proposal from saved scene text/current editor text without mutating the editor, preserving suggested-fix leading/trailing whitespace exactly while using trimming only as an empty guard.

Review evidence:
- Spec review approved selected-text payloads, dirty-scene disablement, scene-review job display, suggested-fix revision/manual-copy routing, and extension parity.
- Code-quality review found exact suggested-fix whitespace preservation and missing integration coverage for annotation-to-revision handoff; both were fixed with regression coverage.
- Code-quality re-review approved with no Critical or Important issues.
- Final sanity review approved the async stale-job guard as non-blocking; the implementation was further hardened by updating the current scene/version ref synchronously during render.

Security/static checks:
- Bandit skipped because Task 10 touched TypeScript/TSX/frontend test files only and no Python files.
- `git diff --check` passed.
- `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false` still fails only on existing unrelated package errors in Notes, AudioStudio, ScheduledTasks, Setup, Dexie audiobook migration, background, scheduled-tasks control-plane, and voice-cloning files. No touched Task 10 Writing Playground files were reported.
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
