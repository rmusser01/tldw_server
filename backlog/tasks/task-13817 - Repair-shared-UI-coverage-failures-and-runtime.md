---
id: TASK-13817
title: Repair shared UI coverage failures and runtime
status: In Progress
labels:
- frontend
- testing
- ci
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reproduce remaining native shared UI coverage failures and the 15-minute timeout on latest dev; fix root causes, review independently, and merge a new PR.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Full shared UI coverage completes with zero failing tests and within the existing CI cap
- [ ] #2 Regression evidence covers Skills, research import, and roleplay failures without disabling cases
- [ ] #3 Independent review and all required PR checks pass before merge
- [ ] #4 Human requester provides a new Change summary before merge
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Baseline origin/dev bfafa07604. Prior PR #3248 report-only coverage timed out after 900s with Skills (3), research import (7), and roleplay (1) failures. ADR assessment pending investigation.
Plan: Docs/Design/IMPLEMENTATION_PLAN_shared_ui_coverage_13817.md. Reproduced 68 import cases with 7 failures: real useStartTutorial requires Router absent in mounted fixture. Independent diagnosis confirms; use existing MemoryRouter render pattern. Native discovery has 2,519 files; prior 900s run completed only 30 files / 3518 tests. Investigating bounded full-suite sharding rather than excluding tests. ADR required: no for fixture-only repair, runner assessment pending.
Root fixes implemented: MemoryRouter render wrapper follows existing workspace fixture; shared package native testTimeout 15000 matches required unit shards (5s server-switch timeout reproduced without coverage, shifting roleplay deadlines observed); full native coverage now runs in eight bounded jobs with all-eight blob validation and Vitest native merge, separate from required gate. Research 68/68 coverage and Skills83/83 coverage pass. Workflow boundary/connected contracts74 pass; missing-shard guard red5/green4 variants. Scoped ruff/pre-commit pass. Bandit test scope skips assertion-only B101 and has zero remaining findings; trusted repo guard subprocess uses resolved Bash and scratch cwd, matching existing CI tests. Independent review found no Critical/Important issues; full hosted coverage and final merge clearance pending. ADR required:no; bounded test execution repair preserves scope/advisory policy, ADR059 governs task tracking. Initial concurrent baseline diagnostic used default coverage/tmp and collided; discarded, subsequent runs use unique report directories.
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
