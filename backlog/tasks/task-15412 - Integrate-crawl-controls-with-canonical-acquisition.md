---
id: TASK-15412
title: Integrate crawl controls with canonical acquisition
status: In Progress
labels:
- web-scraping
- phase5
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute Task 5 of the approved Phase 5 plan as an independently reviewable unit. Parent TASK-15407. Scope, interfaces, exact defaults, named tests and commits are defined by that plan task; preserve all global constraints.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Task 5 behavior and interfaces match the approved spec and plan.
- [ ] #2 Focused red-green tests, relevant regressions, lint and scoped Bandit evidence are recorded.
- [ ] #3 Fresh task reviewer approves spec compliance and code quality; findings are addressed before the next task.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Parent TASK-15407. Plan Docs/superpowers/plans/2026-11-01-web-scraping-phase-5-crawl-process-local-jobs.md, Task 5. ADR assessment: no new ADR; existing ADR-085 governs. Use backlog-py; no product edits outside this task scope without controller ruling. Implementation and review evidence will be linked from the plan-scoped SDD ledger.
Tasks1-4 independent gates complete. Start Task5 canonical parent-controlled acquisition using Task2 contracts and Task3 queue accounting owner: start_page owns page_attempts including retries; queue only checks. baseline-notes.md records current shared HTTP boundaries and physical HTTP thread plus extraction completion ownership. Preserve standalone analyzer behavior, advice precedence, all public signatures and browser attestation. No wrapper/queue cutover yet. Existing inventory artifact line drift remains Task11, full-repo collection blocked by unrelated httpx2 import.
Task5 execution at BASE c592c0576a, designated worktree/branch unchanged. Initial named-suite RED: 19 failed on missing start_page/parent/observer hooks; first GREEN: 19 passed. Adapter/ownership expansion RED: 2 failed, 24 passed (missing robots parent composition and preflight CDP observer); GREEN: 26 passed. Evidence: .superpowers/sdd/2026-11-01-web-scraping-phase-5-crawl-process-local-jobs/task-5-{red,red-2,green-2,green-3}.log. Physical ownership observes actual HTTP-thread completion, extraction concurrent futures, article-browser leases and preflight cleanup tasks. No wrapper/queue cutover. Independent gate pending; retain In Progress.
Task5 final verification: 39 new cases covered; new suite plus preflight architecture/runner compatibility 135 passed. Full required Web_Scraping + http_client scope completed: 3288 passed, 26 skipped, 5 failures, 7040 warnings (task-5-regression-final.log). Failures retained: two arXiv blocked-result expectations reproduced with BASE article source, untouched Task2 cancellation-tuple AST guard, two lifecycle error tests expecting 500 but receiving Task4 admin 403 (also reproduced with BASE endpoint source). Skip provenance: 17 optional curl-cffi unavailable and 9 external APIs disabled; Chromium smoke 3 passed with no skips. No installs, suppressions, or unrelated fixes. Ruff, Black (shared HTTP changed ranges), compileall pass; touched production Bandit zero findings. Report: .superpowers/sdd/2026-11-01-web-scraping-phase-5-crawl-process-local-jobs/task-5-report.md. Implementation ready for independent gate, task remains In Progress.
Final exact named suite GREEN: 39 passed, 90 warnings (2.22s). Explicit inventory equality test fails as expected on current-dev records/line drift, task-5-inventory.log (1 failed); no regeneration. All five full-scope failures are reproduced or source-confirmed baseline behavior, documented individually in the report. Task5 self-review fixed internal HTTP awaiter cancellation by releasing ownership from the actual physical callable, not its asyncio proxy (RED task-5-red-physical-thread.log, GREEN task-5-compat-final.log and final suite). Commit scope includes implementation, three new tests, this task evidence; ignored SDD report/logs remain available locally. Independent review/AC gate remains pending.
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
