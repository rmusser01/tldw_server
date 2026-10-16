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
- [x] #2 Regression evidence covers Skills, research import, and roleplay failures without disabling cases
- [ ] #3 Independent review and all required PR checks pass before merge
- [x] #4 Human requester provides a new Change summary before merge
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Baseline origin/dev bfafa07604. Prior PR #3248 report-only coverage timed out after 900s with Skills (3), research import (7), and roleplay (1) failures. ADR assessment pending investigation.
Plan: Docs/Design/IMPLEMENTATION_PLAN_shared_ui_coverage_13817.md. Reproduced 68 import cases with 7 failures: real useStartTutorial requires Router absent in mounted fixture. Independent diagnosis confirms; use existing MemoryRouter render pattern. Native discovery has 2,519 files; prior 900s run completed only 30 files / 3518 tests. Investigating bounded full-suite sharding rather than excluding tests. ADR required: no for fixture-only repair, runner assessment pending.
Root fixes implemented: MemoryRouter render wrapper follows existing workspace fixture; shared package native testTimeout 15000 matches required unit shards (5s server-switch timeout reproduced without coverage, shifting roleplay deadlines observed); full native coverage now runs in eight bounded jobs with all-eight blob validation and Vitest native merge, separate from required gate. Research 68/68 coverage and Skills83/83 coverage pass. Workflow boundary/connected contracts74 pass; missing-shard guard red5/green4 variants. Scoped ruff/pre-commit pass. Bandit test scope skips assertion-only B101 and has zero remaining findings; trusted repo guard subprocess uses resolved Bash and scratch cwd, matching existing CI tests. Independent review found no Critical/Important issues; full hosted coverage and final merge clearance pending. ADR required:no; bounded test execution repair preserves scope/advisory policy, ADR059 governs task tracking. Initial concurrent baseline diagnostic used default coverage/tmp and collided; discarded, subsequent runs use unique report directories.
PR #3250: https://github.com/rmusser01/tldw_server/pull/3250 head d6de1918a2. Final sizing is16 shards/max8 concurrent, not8, because first native eighth completed only150/315 files at9m32s, inadequate hosted15m margin. Native report merge gets8GB heap on public16GB runners; allsource maps preserved. Real2blob57test merge passes; roleplay9previousdeadline variants pass nativecoverage. Final14workflow tests pass; connected74contracts pass. Independent review of final16shape found no Critical/Important findings. New human-written Change summary requested; old PR3248 summary does not apply. Full hosted coverage pending.
Human requester supplied their own Change summary for PR3250; recorded verbatim in PR body, merge human gate satisfied. CPU profile of WorldBook interaction suite: dominant selector RegExp13s/30s; jsdom29.1.1 is Node20.19 compatible (official v29.1.1 package/release docs), pinned consistently in3workspace manifests andlock. Nativecoverage154cases pass withWB3cases4.9s vs76s baseline andSkills33.1s vs213.9s; no interaction/accessibilitychecksdisabled. First complete16-shard run reveals stalecache/router/transport/path/migration/mirrorfixtures beyondoldcutoff; correcting under same task, productionguards retained. Native16-map merger stress passes with1.5GBreal fullsource blobs and8GBheap. Independentfullreview torepeat afternewfixturecommit.
Complete first hosted run exercised all 2,519 shared test files; it exposed stale fixtures previously hidden behind the monolithic timeout. Repaired router/query/browser/storage fixtures, canonical scoped-auth/cache/API contracts and async interaction boundaries; retained production ownership, recovery, credential and submit safeguards. CPU profiling identified jsdom 28 selector recursion: pinned existing test dependency to compatible jsdom 29.1.1 across workspace manifests/lockfile; measured WorldBooks bulk 76s to 5s and Skills 214s to 33s. No new production behavior. Generated English playground/tutorial extension locale mirrors now match canonical sources. Workflow license/admission inventory updated for two exact guarded downstream jobs with immutable event checkout. Focused Python contracts 46 pass; touched Python Ruff passes; Bandit baseline 11/current 11, zero new findings (assertions excluded as test code). Human requester Change summary is recorded verbatim in PR #3250. All 16-shard hosted rerun and final independent review pending. ADR required: no; test fixture and existing runner/dependency repair, no durable architecture change.
Final combined local coverage passes all 548 tests in 40 edited UI suites (125.94s), preserving the full 262,197-statement source denominator. Additional previously slow/unmodified suites separately pass, including 101 character-manager cases and connected Skills/WorldBooks/media/literature/source-view suites. WebUI typecheck exits0, frozen dependency install unchanged, scoped installed ESLint9 exits0 (existing any warnings), precommit and Python Ruff pass. Full current diff independently reviewed without actionable Critical/Important findings; exact committed-head review and hosted all-shard/merged summary verification next.
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
