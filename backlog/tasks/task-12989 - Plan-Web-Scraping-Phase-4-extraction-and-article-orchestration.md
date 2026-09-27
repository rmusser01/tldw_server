---
id: TASK-12989
title: Plan Web_Scraping Phase 4 extraction and article orchestration
status: Done
created_date: 2026-07-27 15:44
labels:
- web-scraping
- refactor
- phase-4
- planning
priority: High
references:
- TASK-12988
- Docs/superpowers/specs/2026-07-26-web-scraping-phase-4-extraction-article-orchestration-design.md
- Docs/superpowers/plans/2026-07-15-web-scraping-phase-3-governed-preflight-package.md
- Docs/Design/WebScraping_Refactor_Import_Inventory.md
documentation:
- Docs/superpowers/plans/2026-07-27-web-scraping-phase-4-extraction-article-orchestration.md
modified_files:
- Docs/superpowers/plans/2026-07-27-web-scraping-phase-4-extraction-article-orchestration.md
updated_date: 2026-07-27 16:01
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Convert the user-approved Phase 4 extraction and article-orchestration design into a complete test-driven implementation plan with exact file ownership, commands, expected failures, compatibility gates, security checks, and incremental commits.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The plan maps every approved design requirement to an executable task and preserves the four delivery units 4A-4D.
- [x] #2 Each task names exact files, contains test-first steps, concrete implementation guidance, exact verification commands, expected outcomes, and a commit boundary.
- [x] #3 The plan covers governed preflight, direct-browser egress controls, acquisition limits, executor lifecycle synchronization, sanitized observability, deterministic public failures, compatibility fixtures, and both sync-loop guards.
- [x] #4 The plan includes architecture, differential, integration, full Web_Scraping, Bandit, and import-inventory gates without expanding into Phase 5-7 scope.
- [x] #5 The written plan passes spec-coverage, placeholder, type/signature consistency, and scope self-review and is committed.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Inspect the approved specification and current implementation, map responsibilities and public consumers, write the Phase 4A-4D TDD plan, self-review it against every approved contract, verify documentation gates, and commit the plan.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Created a 23-task, 97-step implementation plan organized as the approved sequential Phase 4A-4D merge train. Self-review corrected premature selector integration, Python 3.10-incompatible StrEnum usage, executor shutdown registration, immutable browser-cookie snapshots, shared observability ownership, cache-state transition sequencing, and Black gates that would otherwise cause unrelated formatting churn in predecessor files. Verification on 2026-07-27: Phase 1 contracts passed 19 tests; task/file-section counts, balanced code fences, required-contract coverage, ASCII, unfinished-marker, git diff, and path checks passed. Bandit is not applicable because this planning task changes documentation and Backlog records only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the test-driven Phase 4 extraction and article-orchestration implementation plan. It maps every approved design contract into 23 dependency-ordered tasks across four independently reviewed merge units, with exact file ownership, RED/GREEN tests, implementation interfaces, verification commands, security gates, and commit boundaries. The plan explicitly preserves governed preflight and enhanced-scraper behavior while covering bounded regex, neutral content/selectors, canonical extraction, guarded HTTP/browser acquisition, response limits, executor generations, sync-loop guards, deterministic failures, sanitized observability, consumer migration, inventory regeneration, and final certification. Focused baseline verification passed 19 tests; documentation checks were clean. No known blockers remain before selecting an execution mode.
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
