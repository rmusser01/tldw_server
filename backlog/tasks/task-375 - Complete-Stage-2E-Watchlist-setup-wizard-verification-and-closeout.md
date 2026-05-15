---
id: TASK-375
title: Complete Stage 2E Watchlist setup wizard verification and closeout
status: To Do
assignee: []
created_date: '2026-05-15 04:57'
labels:
  - watchlists
  - stage2
  - verification
dependencies:
  - TASK-374
references:
  - >-
    Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage2-setup-wizard-plan.md
  - Docs/superpowers/specs/2026-05-15-first-class-watchlists-design.md
documentation:
  - >-
    Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage2-setup-wizard-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Close out the Stage 2 Watchlist-first setup wizard implementation. Scope: focused frontend suite, TypeScript/package check if practical, browser/CDP constrained viewport smoke, docs updates if API/user-facing behavior needs clarification, Backlog final summaries, and diff hygiene. No new feature behavior beyond fixes found by verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Focused Stage 2 frontend test suite passes or any failures are fixed/documented.
- [ ] #2 TypeScript/package check is run if practical or exact unrelated baseline failure is recorded.
- [ ] #3 CDP/Playwright smoke covers desktop and 390x844 setup wizard flow.
- [ ] #4 Docs and Backlog records capture final behavior, verification, known skips, and remaining Stage 3+ boundaries.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
