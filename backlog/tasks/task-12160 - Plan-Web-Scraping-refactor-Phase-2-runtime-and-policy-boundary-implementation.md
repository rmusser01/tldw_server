---
id: TASK-12160
title: Plan Web_Scraping refactor Phase 2 runtime and policy boundary implementation
status: Done
created_date: 2026-07-05 04:50
labels:
- web-scraping
- plan
- refactor
references:
- Docs/superpowers/specs/2026-07-04-web-scraping-phase-2-runtime-policy-boundary-design.md
- Docs/superpowers/specs/2026-07-03-web-scraping-refactor-design.md
- backlog/tasks/task-12159 - Design-Web-Scraping-refactor-Phase-2-runtime-and-policy-boundary.md
modified_files:
- Docs/superpowers/plans/2026-07-05-web-scraping-phase-2-runtime-policy-boundary.md
updated_date: 2026-07-05 16:13
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the implementation plan for Web_Scraping refactor Phase 2 based on the approved runtime and policy boundary design. Scope is planning only; no Python runtime code changes in this task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Implementation plan is saved under Docs/superpowers/plans with concrete staged tasks.
- [x] #2 Plan covers runtime contracts, policy adapter, fetch adapter, browser/session/timeout/cancellation placeholders, scrape_article wiring, compatibility tests, verification, and Backlog finalization.
- [x] #3 Plan includes explicit guardrails for preflight analyzer preservation, curl backend call mode, curl-to-httpx fallback, import boundaries, and public API compatibility.
- [x] #4 Plan is self-reviewed for spec coverage, placeholders, and type consistency.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read the approved Phase 2 design and current Web_Scraping touchpoints. 2. Write a detailed TDD implementation plan under Docs/superpowers/plans. 3. Self-review the plan against the design spec and fix gaps before handing off for execution choice.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-07-05 finalization: Implementation plan executed via TASK-12161. Rebase onto latest origin/dev completed cleanly before final verification; verification base recorded as HEAD e6bfdabe1fd2d3d67ef85eab59fc7da77f434108 with merge-base 242297a2b8e5defeb5b1d5d74253ea75e787c4b0 and branch initially ahead 16/not behind. Initial final verification evidence: focused Phase 2 pytest group passed 62 tests; compatibility/hardening pytest group passed 49 tests; legacy compatibility subset passed 6 tests; git diff --check exited 0; Bandit production scan wrote /tmp/bandit_web_scraping_phase2_runtime_policy.json with exit 0 and zero results. Whole-branch review then found the default runtime httpx fetch adapter had moved article fetches from response-mode egress/DNS-pin behavior to the simplified no-method path. TASK-12161 fixed that by keeping curl on the simplified backend path while routing httpx/default through response-mode method="GET"; post-fix verification repeated the focused Phase 2 group (62 passed), compatibility/hardening group (49 passed), legacy subset (6 passed), git diff --check (0), and Bandit production scan (/tmp/bandit_web_scraping_phase2_runtime_policy_final_review_fix.json, zero results).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Phase 2 runtime/policy boundary planning is finalized. The implementation plan was executed by TASK-12161, the branch was rebased cleanly onto latest origin/dev before final verification, a final security review finding was fixed, and the required focused, compatibility, whitespace, and Bandit checks all passed after that fix. No remaining blockers are recorded for this planning task.
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
