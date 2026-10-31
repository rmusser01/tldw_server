---
id: TASK-13192
title: Expose synchronous audio summarization in Service Prompts
status: In Progress
assignee: []
created_date: '2026-09-05 19:10'
updated_date: '2026-09-05 20:30'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2901'
  - TASK-12957
documentation:
  - Docs/Design/audio-summary-service-prompt.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continues the approved audio Service Prompts slice from the branch-local TASK-13185. Latest dev assigned TASK-13185 to manual llama.cpp snapshots, so preserve that upstream task and archive the branch-local audio record using the official workflow. PR #2901 exposes an atomic literal system/user pair through shared Settings with owner-scoped request snapshots and existing defaults.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Shared WebUI and extension edit/reset the audio system/user pair
- [x] #2 Explicit-part precedence and owner snapshots survive files and recursive passes
- [x] #3 Inactive analysis bypasses prompt storage and deployment/direct-core defaults remain unchanged
- [ ] #4 Review findings resolved and latest-dev rebase verified before merge
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Preserve upstream snapshot changes and archive only the superseded audio tracking record; resolve the generated OpenAPI conflict by regeneration; verify targeted regressions and fingerprint; push with an exact lease; monitor required checks and merge the reviewed head.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Original implementation3b126b37d9, Qodo fixes02c1b38ffb. Baseline102 passed, full targeted backend236/sharedUI198/WebUI5 passed; Qodo helper corrections125passed. Bandit zero, Ruff/compileall/OpenAPI checks passed. Qodo currently zero bugs/violations; all3threads resolved. Rebase onto dev53d683f0ed now required after snapshot PR2883 merged. Full repository suite/live browser/STT/provider checks not run locally.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Rebased onto dev53d683f0ed. Only generated fingerprint conflicted; regenerated combined API has2073paths/3140schemas, fingerprintaffe135193fca5726d8e378e82620553b55ceafc2f882d3af59b8b28ca5f0788. Range-diff confirms no runtime/test patch changes beyond fingerprint. Post-rebase140tests passed, Bandit zero findings, OpenAPI typegen and fingerprint check passed. PR body updated to TASK-13192. Awaiting fresh remote review/checks before merge.

Second latest-dev rebase on 2026-09-05: dev advanced to a61abea39d (Personal Context activation) before merge. Regenerated the sole conflicting OpenAPI fingerprint (cb2cab1debb70065972c7e42a6895fdb53eba9b0fc21a6b786891bb2c43f5778). Range-diff confirms no runtime/test patch changes. Focused audio, registry, API and request-contract suite passed 125 tests; Bandit zero findings; OpenAPI type generation and fingerprint validation passed. Prior head had all seven required checks green and Qodo zero findings. Fresh rebased-head review/checks must pass before merge.

Dev PR2902 allocated TASK-13192 independently to Resource Governor cookie identity. Archive this audio record before rebasing so the upstream task is preserved. Audio PR2901 tracking will continue under a freshly allocated task after the rebase.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

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
