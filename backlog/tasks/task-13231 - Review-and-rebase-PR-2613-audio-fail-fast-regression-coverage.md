---
id: TASK-13231
title: Review and rebase PR 2613 audio fail-fast regression coverage
status: In Progress
assignee: []
created_date: '2026-09-10 00:41'
updated_date: '2026-09-10 00:41'
labels:
  - audit
  - media
  - tests
  - pr-followup
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2613'
  - >-
    backlog/archive/tasks/task-13001 -
    Refresh-audit-oversized-audio-download-regression-PR.md
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Review PR #2613 against current dev, retain useful test-only fail-fast assertions, reconcile task-ID collisions, verify the result, and assess merge readiness.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Rebased PR retains incremental downloader contract coverage on current dev without production changes.
- [x] #2 The audit task has a unique active ID and existing UserProfiles records remain unchanged.
- [x] #3 Focused tests, Ruff, Bandit, diff checks and independent review are recorded.
- [ ] #4 Rebased head is pushed and current CI and human-summary merge gates are reported.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Stage 1: inspect PR, dev, and review threads (complete). Stage 2: rebase and archive the colliding audit task through Backlog CLI from its unambiguous original revision (complete). Stage 3: verify focused behavior and independent review, push with an explicit lease, and assess fresh CI and merge policy (in progress).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Supersedes the archived PR-specific TASK-13001 audit record, which collides with current dev UserProfiles TASK-13001. MCP reads were unresponsive, so official Backlog CLI is used. Rebased onto origin/dev 40345571a2cfc8b3a8893545836097d27e4ee86c. The original no-op regression repair is already in dev; remaining value is correct URL, stream=True and no body iteration for oversized Content-Length in the injected downloader path. Three focused tests pass with four warnings; Ruff passes; Bandit has seven LOW B101 findings, all ordinary pytest assertions, zero errors. Independent review found only the now-archived active task collision. Both GitHub review threads are resolved. Old July backend-required failure was OpenAPI fingerprint drift, unrelated to the test-only diff. A human-written Change summary is still required by repository merge policy.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
