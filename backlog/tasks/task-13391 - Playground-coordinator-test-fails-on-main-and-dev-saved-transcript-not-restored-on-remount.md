---
id: TASK-13391
title: >-
  Playground coordinator test fails on main and dev: saved transcript not
  restored on remount
status: To Do
assignee: []
created_date: '2026-09-28 00:17'
labels:
  - frontend
  - chat
  - ci
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Playground.coordinator.integration.test.tsx fails on BOTH origin/main (v0.1.45, 96bf0996eb: 3 failed) and origin/dev (3c9d97c56b: 4 failed). Reproduced 2026-09-27 from apps/packages/ui with bunx vitest run, the way CI runs UI tests.

- 'promotes accepted entry to saved route and restores its transcript on remount' (both URL forms): after remount, messages are [] instead of ['Question', 'Saved final answer'] (:526).
- 'retires a saved Character route before its accepted retry branch can be reasserted': [] instead of ['Question', 'Recovered answer'] (:590).

It went unnoticed because the frontend ratchet runs only impacted tests. When it is impacted, the ratchet cannot tolerate it even though the base fails too: it rejects the result as 'contains an unfinished assertion', so any PR whose changes reach the Playground graph goes red. This blocks #3035 (v0.1.45 main-to-dev sync) and will block #3033 once it merges current dev.

It may be a real defect (a saved Character chat's transcript not restored after reload), or fixture drift from the H1 ownership work (a14c509e55, 15003d4135). Owner: the Chatbook history/fork workstream.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Root cause identified: product defect or stale fixture
- [ ] #2 Both coordinator tests pass from apps/packages/ui on dev
- [ ] #3 If a product defect: a regression test that fails before the fix
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
