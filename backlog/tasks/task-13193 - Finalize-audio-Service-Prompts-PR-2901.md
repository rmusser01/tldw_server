---
id: TASK-13193
title: Finalize audio Service Prompts PR 2901
status: In Progress
assignee: []
created_date: '2026-09-05 20:31'
updated_date: '2026-09-05 20:33'
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
Continue the approved synchronous audio Service Prompts slice. Preserve archived audio TASK-13185 and TASK-13192 histories; upstream separately allocated both numeric IDs. This task tracks final verification and merge of PR2901, not a new feature.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Audio prompts remain owner-scoped with explicit-part precedence and existing defaults
- [x] #2 Current dev rebase preserves reviewed runtime changes and passes focused tests, Bandit and OpenAPI validation
- [ ] #3 Qodo findings resolved and required current-head checks pass before merge
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Resolve generated fingerprint against dev84b6928dcf; verify audio and authentication integration; push exact lease; monitor current-head Qodo/checks and merge.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Previous head5647aa39e2 passed all seven required checks; Qodo zero bugs/violations, all threads resolved. Rebase required after Buddy PR2902 merged. Only generated fingerprint conflicted. Historic verification:236 backend,198 shared UI,5 WebUI tests; post-activation rebase125 focused tests and zero Bandit findings. Full suite and live STT/provider calls not run locally.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Rebased onto dev84b6928dcf. Range-diff confirms runtime and test patches unchanged; generated fingerprint now65bab92528ee0dbfb42d33045bbaa85508eb56617591eb5303edef38c40456dd (2073paths/3142schemas). Audio/registry/API/request-contract plus new cookie-owner regressions:140passed,10warnings. Bandit zero findings; official OpenAPI typegen and fingerprint validation pass. Logs: /tmp/audio-buddy-dev-tests.log, /tmp/bandit_audio_buddy_dev.json, /tmp/audio-buddy-fingerprint.log. Awaiting fresh current-head CI and Qodo before merge.
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
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
