---
id: TASK-15406
title: Plan Web_Scraping Phase 5 crawl and process-local jobs implementation
status: Done
labels:
- web-scraping
- architecture
- planning
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the implementation plan from the user-approved Phase 5 written specification and ADR-085. Map scoped files, exact cross-task interfaces, concrete finite safety defaults, test-first reviewable tasks, staged compatibility cutovers, and verification/rollback gates. Preserve analyzer and legacy behavior, bounded batch production, outcome/cancellation ownership, and the process-local queue; durable Jobs migration remains deferred. Documentation and tracking only: no product implementation until the user reviews this plan and selects an execution method.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The plan maps every approved spec requirement to 3-5 stages of independently testable tasks with exact files, interfaces, red/green tests, verification commands, and commits.
- [x] #2 The plan specifies numeric safety defaults and trusted override rules, downstream public/legacy outcome compatibility, parent-budget integration, batch/queue/drain ownership, checkpoint migration and recovery limits.
- [x] #3 The complete plan is self-reviewed for coverage, interface consistency, testable steps and five concrete review-focus cases; documentation/task checks pass and the committed plan is presented for user review before execution.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-11-01: TASK-15282 written design and ADR approved by human requester. Plan path: Docs/superpowers/plans/2026-11-01-web-scraping-phase-5-crawl-process-local-jobs.md. Spec path: Docs/superpowers/specs/2026-10-04-web-scraping-phase-5-crawl-process-local-jobs-design.md. Governing ADR: Docs/ADR/085-web-scraping-crawl-and-process-local-job-ownership.md (mark Accepted with this approval). ADR required: no new ADR; this plan implements approved ADR-085 without changing the decision. Read-only current integration baseline origin/dev 54b321dcec8cdbb6358581f42c1548696260570b; design worktree baseline 75ab224081bf140ef52017c1a9b0a04f6878d488 plus approved local doc commits. Do not rebase or edit the dirty main checkout in this planning task. No duplicate Phase 5 plan task found in local Backlog/current dev paths. Task ID reserved above local TASK-15282, dev max 15281, and all paginated open-PR files max 15405. Plan review and execution choice are not implied by written-spec approval.
2026-11-01: Wrote Docs/superpowers/plans/2026-11-01-web-scraping-phase-5-crawl-process-local-jobs.md with four stages/eleven task-level TDD and commit gates, numeric finite defaults/trusted overrides, exact internal interfaces, file map, staged cutovers/rollback, service outcome mappings, and certification. Self-review corrected post-admission subscription races, lower-level concrete-context coupling, async admission composition, article-plan/credential propagation, single accounting ownership and actual conversion/doc paths. Five Review Focus failure classes each have owning named tests. Opaque external-tool invocation/time/output accounting explicitly does not claim internal wire-byte measurement. No product code or future tests were implemented.
Verification: task-format pytest suite 4 passed, 1 warning; backlog-py normalize --check passed for both tasks; all 82 relative file links across four documents resolve; scoped pre-commit hooks, ASCII and git diff --check pass. Bandit/product compile/runtime suites are not applicable to this Markdown-only plan. Main checkout unchanged. Commit and present plan for review, then wait for explicit plan approval and execution-method selection. TASK-15406 remains In Progress through the review gate; no rebase, implementation, PR readiness or merge is claimed.
2026-11-01: Human selected option 1 (subagent-driven execution) after plan presentation. Approved plan carried onto current dev without replaying obsolete colliding draft identifiers. Product execution is now authorized under TASK-15407 and its eleven child tasks; this planning task is complete.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Four-stage eleven-task implementation plan written, self-reviewed, verified and presented. Human selected subagent-driven execution on 2026-11-01, satisfying plan review and execution-method gate. TASK-15407 and TASK-15408 through TASK-15418 track implementation; no product completion is implied.
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
