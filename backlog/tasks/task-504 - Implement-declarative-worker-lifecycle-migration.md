---
id: TASK-504
title: Implement declarative worker lifecycle migration
status: In Progress
labels:
- backend
- implementation
- lifecycle
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute the approved declarative worker lifecycle implementation plan using subagent-driven development. Replace managed-worker task/stop-event field threading with WorkerSpec definitions, lifecycle session/engine ownership, diagnostics compatibility, parity checks, and final verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All tasks in Docs/superpowers/plans/2026-06-02-declarative-worker-lifecycle-implementation-plan.md are completed or blockers documented.
- [ ] #2 Each task receives spec compliance and code quality review before moving to the next task.
- [ ] #3 Focused lifecycle tests, compileall on app/services, and Bandit on app/services run with results recorded.
- [ ] #4 Final summary and touched files are recorded in Backlog.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Task 1 completed at commit 3dfcf77f4e0a561bd399b9d640f90d9fa0be3f1f. Added lifecycle worker spec types, graph validation, post-predicate dependency validation, POST_WORKER_SHUTDOWN phase, and focused spec tests. Verification: focused pytest 36 passed/6 warnings; git diff --check passed; Bandit on lifecycle_worker_specs.py reported 0 findings. Reviews: spec compliance passed; code quality passed with one minor non-blocking note about future string-field validation.

Task 2 completed at commit 5d18e330639689b1880b1031078f43a0f62246ea. Added WorkerLifecycleSession diagnostics and stop-state tracking, preserved inventory compatibility through publish_worker_inventory(), mapped stopped-name app-state fields, and fixed callback-only diagnostics to publish the spec task_name with has_stop_event=false. Verification: focused pytest 19 passed/6 warnings; git diff --check passed; Bandit on lifecycle_worker_session.py reported 0 findings. Reviews: spec compliance passed after the callback-only task_name fix; code quality passed with no blocking issues.

Task 3 completed at commit fb189903fbd2ef2b6cfc1ded46d843a5f63716fa. Added LifecycleWorkerEngine startup/shutdown orchestration, dependency-aware startup/reverse shutdown, failure-policy handling, startup abort cleanup, immediate task failure/cancellation classification, diagnostic-name publication, and focused engine tests. Verification: lifecycle pytest suite 47 passed/6 warnings; Ruff touched-file check passed; git diff --check and git diff --cached --check passed; Bandit on engine/session scope reported 0 findings. Reviews: spec compliance passed; code quality passed after startup cleanup, cancellation/diagnostic-name, and lint fixes.

Task 4 completed at commit dd81dd1bc808af7e857dafc4e517c019331bccb2. Added lifecycle worker catalog helpers, provider collection graph validation, legacy worker spec parity assertion, synthetic parity coverage from ownership matrix plus runtime task fields, and missing-name reporting. Verification: focused pytest 8 passed/6 warnings; git diff --check and git diff --cached --check passed; Bandit on lifecycle_worker_catalog.py reported 0 findings. Reviews: spec compliance passed; code quality passed with one accepted minor note that the runtime _task heuristic should be replaced by real provider parity as Task 5+ lands.

Task 5 completed at commit 191af27747daba552beec454759660e477efd2ff. Added primary, study/privilege, and content job poller WorkerSpec providers, centralized shared stop-event and route-gated predicate helpers, removed duplicated provider boilerplate/globals lookup, and extended provider/catalog parity coverage for the first real worker groups. Verification: focused provider/catalog pytest 67 passed/6 warnings; lifecycle_worker_specs pytest 24 passed/6 warnings in spec review; Ruff touched-file check passed; Bandit on touched service modules reported 0 findings. Reviews: spec compliance passed; code quality passed after helper centralization.

Task 6 completed at amended commit e0883ae87713ad106f2f485194f397aca7995c7f. Added background, sidecar-owned, cleanup, notification/A-B, compactor/WebSub, claims rebuild, usage aggregator, and LLM usage aggregator WorkerSpec providers plus provider/catalog parity tests. Fixed the initial spec-review gap by converting notifications bridge, storage cleanup, and WebSub renewal from callback-only specs to engine-startable stop-event adapters that start the legacy resource and cancel/stop it on lifecycle stop. Verification: red run failed as expected before providers; final focused pytest 83 passed/6 warnings; Ruff touched-file check passed; Bandit on touched service modules reported 0 findings; git diff --check passed. Reviews: initial spec review requested changes for callback-only startup semantics; local re-review approved after amend due subagent usage limit; code quality approved locally with no blocking findings.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Task 7 completed at commit 8f2b58596a. Added service-tail/runtime-monitor/optional/infra/auxiliary/maintenance/recurring scheduler WorkerSpec providers plus shared startup adapters for legacy task-returning services. Included tts_history_cleanup_task and connectors_jobs_task in startup_infra_services.py because that is their actual legacy owner, and included claims_alerts_task/claims_review_metrics_task because they remain service-tail runtime fields. Verification: red run failed as expected before providers with 12 missing-provider failures; final focused pytest passed 80 tests/6 warnings; Ruff touched-file check passed; Bandit on touched service modules wrote /tmp/bandit_task504_task7.json with 0 results/errors; git diff --check and git diff --cached --check passed. Reviews: local spec compliance and code quality review passed; no blocking findings. Subagent re-review was unavailable due the usage-limit error carried over from Task 6.
Task 8 completed at commit 68dbda8894. Startup bootstrap now builds a WorkerLifecycleContext, collects the full declarative provider graph through startup_worker_groups.collect_startup_worker_specs(), starts workers through LifecycleWorkerEngine, returns/stores WorkerLifecycleSession, and runs only non-worker startup tail setup afterward. LifespanWorkerRuntimeState now stores worker_lifecycle_session while preserving a compatibility owned_job_pollers/session-handle bridge for the still-legacy shutdown path until Task 9/10 remove it. Startup integration contract tests were updated from direct worker-group/service-tail delegation to spec collection, lifecycle start, bootstrap delegation, and non-worker tail delegation. Verification: red Task 8 run failed as expected with 5 missing-session/collector failures; focused startup suite passed 16 tests; final Task 8 suite passed 37 tests/6 warnings; updated main lifecycle contract subset passed 4 tests/6 warnings; Ruff touched-file check passed; Bandit on touched service modules wrote /tmp/bandit_task504_task8.json with 0 results/errors; git diff --check and git diff --cached --check passed. Reviews: local spec compliance and code quality review passed; accepted deviation from the written field-removal wording because removing managed-worker fields before Task 9 would break the legacy shutdown midpoint.
Task 9 started. Scope: wire shutdown sequence and job-poller handoff to stop WorkerLifecycleSession phases through LifecycleWorkerEngine while preserving non-worker cleanup gates and shutdown diagnostics.
Task 9 completed. Shutdown now stops lifecycle-managed workers from the stored WorkerLifecycleSession through LifecycleWorkerEngine phases: job poller quiesce during handoff, background worker shutdown before coordinated legacy cleanup, and post-worker shutdown before non-worker post cleanup. Job-poller handoff now receives the lifecycle session/engine instead of raw owned_job_pollers, while preserving lease-wait behavior and quiesced/stopped diagnostics. Added non-worker post-cleanup helper so personalization consolidation remains outside managed-worker stopping. Verification: focused pytest passed 53 tests/10 warnings; scoped Ruff touched-file check passed; git diff --check passed; Bandit on touched production files wrote /tmp/bandit_task504_task9.json with 0 findings. Note: broad Ruff over app/main.py and test_main_shutdown_job_pollers.py remains blocked by pre-existing baseline issues outside this task scope.
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
