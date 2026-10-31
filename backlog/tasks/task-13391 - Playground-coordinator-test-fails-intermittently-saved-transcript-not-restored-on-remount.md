---
id: TASK-13391
title: >-
  Playground coordinator test fails intermittently: saved transcript not
  restored on remount
status: To Do
assignee: []
created_date: '2026-09-28 00:17'
updated_date: '2026-09-29 21:11'
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

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
CORRECTION 2026-09-28: not a deterministic failure. The identical code that failed yesterday (dev 3c9d97c56b: 4 failed; main 96bf0996eb: 3; the sync branch: 3; and CI on #3035 shard 7/8) passes today: 7 of 7 full-file runs, including 3c9d97c56b itself, current dev 26deeb6525 and runs with every core saturated. All of yesterday's failures, local and CI, happened between about 22:45 and 23:20 UTC. The test's only Date.now() is fixture data, and the loader's only time use is mirror timestamps, so no time dependence was found. It stays open as intermittent: the next failure should capture the time, machine load and full output. When it does flake it is still a hard blocker, because the ratchet rejects the result as an unfinished assertion instead of tolerating a base failure.

2026-09-29: wall-clock dependence ruled out. All failures fell in 22:45-23:20 UTC on 2026-09-27, so the process clock was shifted into that window with a --require preload that replaces Date (the shift was confirmed inside a jsdom Vitest test).

Results:
- The exact commit that failed (3c9d97c56b, apps/packages/ui exported with its extension shims) passes 61/61 on the real clock, at 2026-09-27T22:50Z and at 23:10Z.
- Current dev passes 61/61 at the same two instants.

The window therefore reflects something environmental, not time-of-day or date logic in the test or the loader. Candidates that fit a failure shared by local and CI runs for about 35 minutes are an external service the test path reached unmocked, or a dependency or registry state during that window. Neither is confirmed. Still intermittent, with no reproduction. The next failure should capture the full output, the network activity and the resolved dependency versions (bun.lock hash) alongside the time.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
