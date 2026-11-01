---
id: TASK-15410
title: Isolate Web Scraping Phase 5 bounded job queue
status: In Progress
labels:
- web-scraping
- phase5
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute Task 3 of the approved Phase 5 plan as an independently reviewable unit. Parent TASK-15407. Scope, interfaces, exact defaults, named tests and commits are defined by that plan task; preserve all global constraints.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Task 3 behavior and interfaces match the approved spec and plan.
- [ ] #2 Focused red-green tests, relevant regressions, lint and scoped Bandit evidence are recorded.
- [ ] #3 Fresh task reviewer approves spec compliance and code quality; findings are addressed before the next task.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Parent TASK-15407. Plan Docs/superpowers/plans/2026-11-01-web-scraping-phase-5-crawl-process-local-jobs.md, Task 3. ADR assessment: no new ADR; existing ADR-085 governs. Use backlog-py; no product edits outside this task scope without controller ruling. Implementation and review evidence will be linked from the plan-scoped SDD ledger.
Task 2 independently approved after I1-I5 fixes; start new queue implementation without switching legacy wrapper. Consume runtime.controls ParentControls/WorkHandle and contracts.execution PageData. Define validated QueueLimits in jobs/models.py; create minimal jobs package initializer. Preserve pending retry reservations and live/payload/worker charges through physical drain; keep subscribers private and rejection immediate. Parent TASK-15407; no new ADR beyond ADR-085.
Task 3 implementation started at cc1c4e5180 in codex/web-scraping-phase-5. Read task-3 brief, global constraints, root AGENTS, reviewed runtime controls/execution contracts and legacy queue. ADR required: no; ADR-085 governs isolated process-local ownership; no wrapper/service cutover. TDD and scoped verification; no subagents or installs per controller. Report: .superpowers/sdd/2026-11-01-web-scraping-phase-5-crawl-process-local-jobs/task-3-report.md. Leave In Progress for controller review.
Task 3 implementation evidence (controller review pending): isolated jobs/{__init__,models,admission,queue,projections}.py plus phase5_queue_fakes.py and the three brief-named queue suites. Legacy enhanced wrapper and service unchanged against cc1c4e5180. RED: missing jobs API (3 collection errors); independent cancellation-without-result, failure traceback retention, invalid timestamp and pre-cancelled job tests reproduced failures before fixes. Final focused run: 330 passed / 672 warnings (23.57s), including queue, legacy wrapper retention/context, Phase 5 contracts/controls/profile regressions. New jobs coverage 97% (473 statements, 12 missed). Explicit Ruff --no-force-exclude, Ruff format --check (9 files), py_compile and scoped Bandit all exit 0; final Bandit has no findings. Scoped pre-commit previously passed; final task-evidence recheck follows. No new ADR (ADR-085 applies), wrapper/service cutover, global lint edits, installs, subagents, branches, pushes or merges. Full commands/interfaces/race evidence/concerns: .superpowers/sdd/2026-11-01-web-scraping-phase-5-crawl-process-local-jobs/task-3-report.md. TASK-15410 stays In Progress; controller owns fresh review/acceptance and TASK-15407 metadata. Existing warning counts are reported, not attributed as new or triaged. Whole-project httpx2 baseline is controller-known; controller reports the two PG persistence cases now pass after approved parent-venv install; no duplicate PG rerun here.
Final exact brief-named three-suite rerun: 110 passed / 232 warnings in 1.14s, exit 0 (/tmp/task3-final-queue.log). Final focused 330-test run includes the last validation/pre-cancel parity fixes. No blockers/rulings remain; warning totals untriaged, controller fresh review pending. Final report is local ignored SDD handoff material; committed task evidence records commands/results and links that report.
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
