---
id: TASK-15411
title: Guard Web Scraping lifecycle and public projections
status: In Progress
labels:
- web-scraping
- phase5
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute Task 4 of the approved Phase 5 plan as an independently reviewable unit. Parent TASK-15407. Scope, interfaces, exact defaults, named tests and commits are defined by that plan task; preserve all global constraints.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Task 4 behavior and interfaces match the approved spec and plan.
- [ ] #2 Focused red-green tests, relevant regressions, lint and scoped Bandit evidence are recorded.
- [ ] #3 Fresh task reviewer approves spec compliance and code quality; findings are addressed before the next task.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Parent TASK-15407. Plan Docs/superpowers/plans/2026-11-01-web-scraping-phase-5-crawl-process-local-jobs.md, Task 4. ADR assessment: no new ADR; existing ADR-085 governs. Use backlog-py; no product edits outside this task scope without controller ruling. Implementation and review evidence will be linked from the plan-scoped SDD ledger.
Task3 queue/projection base independently approved. Start lifecycle authorization and safe public projections, keeping legacy queue runtime until Task9. baseline-notes.md records canonical platform roles/permissions and why narrower admin-bypass helper is not interchangeable. Verified remote shared lifecycle commands require admin before side effects; trusted automatic initialization for ordinary scraping remains supported. Existing service admin/persistence baseline now covers both PG owner-isolation cases after approved dependency install.
Task 4 execution started at BASE 34cb0a0676 in the designated Phase 5 worktree. Read task brief/global constraints/boundary notes and ADR-059. Implementing TDD for canonical lifecycle authorization, serialized ownership-aware shutdown/reinitialize, and shared safe job/progress/history projections. Child remains In Progress pending controller independent gate.
Task 4 implementation and self-review complete; independent controller gate pending, keep In Progress. Added canonical remote lifecycle authorization before singleton side effects; shielded serialized startup/shutdown with queue-owned drain checks; shared allowlisted job/progress/history projections and pre-cutover legacy security adapters. TDD evidence: initial focused RED 15 failed/12 passed; added regressions RED for progress, concurrent cleanup, aliases, startup cancellation, and real legacy logging, all GREEN. Final focused 34 passed; existing admin/retention 10 passed; broader Phase5/guards/preflight/persistence 471 passed (955 warnings). Compile and changed-range Ruff formatting pass; Bandit all five touched production files: 0 findings/errors. Ruff only unchanged legacy UP015 confirmed against BASE. Full pytest collection blocked by existing Workspaces httpx2 import (83977 items, 1 error, 12 skipped), unchanged. Full commands, implementation decisions, paths and concerns: .superpowers/sdd/2026-11-01-web-scraping-phase-5-crawl-process-local-jobs/task-4-report.md. No controller plan/ledger edits, subagents, installs or main-checkout changes.
Independent review fix round 1 started from reviewed HEAD cc4db584dc. Verified Important 1 nested metadata prose exemption and Important 2 permanently cached failed cleanup. Adding RED/GREEN regressions; preserve article-boundary prose, incremental byte sizing, scraper ownership and serialized explicit retry. Minor 3 remains deferred to controller final review. Covering tests will expose warning provenance via addopts override; no full-repository collection rerun.
Review fix round 1 complete; keep In Progress for controller gate. Important 1 fixed via explicit trusted article path in incremental encoder: only actual top-level article content/raw fields bypass URL redaction; structured metadata/preflight nested same-named fields sanitized. RED 28 failed -> GREEN 28 passed, including exact-byte/one-less history retention and both history adapters. Important 2 fixed: explicit serialized shutdown may retry only completed failed/canceled cleanup, retaining scraper/singleton/drain guard. RED 2 failed -> GREEN 2 passed (OSError, CancelledError; concurrent retry and reinitialization gating). Final covering 197 passed/406 warnings with visible summary via addopts override; sources/counts compared to Task3 in appended report. Scoped Ruff/format/compile pass; Bandit 0 findings/errors. No full-repo collection rerun or warning suppression. Minor 3 unchanged/deferred as instructed. Report: .superpowers/sdd/2026-11-01-web-scraping-phase-5-crawl-process-local-jobs/task-4-report.md.
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
