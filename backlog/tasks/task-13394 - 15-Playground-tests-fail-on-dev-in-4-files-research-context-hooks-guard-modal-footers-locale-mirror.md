---
id: TASK-13394
title: >-
  15 Playground tests fail on dev in 4 files (research context, hooks guard,
  modal footers, locale mirror)
status: To Do
assignee: []
created_date: '2026-09-28 19:59'
labels:
  - frontend
  - tests
  - ci
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found 2026-09-28 while burning down TASK-13390. From apps/packages/ui, the way CI runs them, these fail identically on dev with no local changes:
- Playground.research-context.integration.test.tsx: 12 tests (follow-up surfaces, auto-restore of attached research context, pin/unpin, StrictMode readiness)
- PlaygroundHooks.jsx-extension.guard.test.ts: 'stores JSX-bearing hooks in .tsx modules'
- PlaygroundModalFooters.design-system.test.tsx: startup template footer buttons
- playground-locale-mirror.test.ts: English playground strings not mirrored into the extension locale

Like the Media and chat-submit breaks fixed in #3035/#3046, these are hidden by the frontend ratchet, which only runs impacted tests, and will block any PR whose changes reach these files. Two of them (the hooks .tsx guard and the locale mirror) are source-contract checks and likely point at real drift, not stale tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each file triaged: stale test (fix the test) vs real drift (fix the code), recorded here
- [ ] #2 All four files pass from apps/packages/ui
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
