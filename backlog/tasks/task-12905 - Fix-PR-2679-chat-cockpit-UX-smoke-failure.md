---
id: TASK-12905
title: Fix PR 2679 chat cockpit UX smoke failure
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-07-07 16:39'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR #2679 current-head CI follow-up after the 0.1.38 release. Current failures found and addressed in this task: Frontend UX Gates / UX Smoke Gate restore-control detachment and mobile tabpanel target drift; Guardian generic notification timestamp mutation regression in full-suite shard gap-verified-7 on Python 3.12/3.13; backend OpenAPI contract drift gate fingerprint mismatch under CI Python 3.12 generation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PR #2679 UX Smoke Gate failure root cause documented.
- [x] #2 Mobile cockpit tab aria-controls targets remain mounted and stable when rails are hidden.
- [x] #3 Focused regression coverage added for hidden mobile restore tab panels.
- [x] #4 Relevant local verification recorded before pushing.
- [x] #5 PR #2679 current-head CI failure root causes documented.
- [x] #6 Mobile cockpit tab aria-controls targets remain mounted and stable while rails are hidden and restore controls do not detach during rail visibility changes.
- [x] #7 Generic notification payloads receive a timestamp in-place before notification persistence.
- [x] #8 Checked-in OpenAPI fingerprint matches the current CI-reported backend contract fingerprint.
- [x] #9 Relevant local verification recorded before pushing.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Current-head PR #2679 failures investigated: UX Smoke Gate still saw the desktop right-rail restore button detach during click retries and mobile tab aria-controls referenced a missing generated id; root cause is shell nodes/ids changing during cockpit state transitions. Fix keeps mobile cockpit ids deterministic and keeps desktop restore controls mounted while toggling hidden state. Guardian full-suite gap-verified-7 failed test_generic_notify_adds_timestamp because notify_generic added ts only to the recorded copy, while the test and prior behavior expect the caller payload to be mutated. Fix restores in-place payload timestamping before copying/sanitizing. OpenAPI contract drift gate reported current CI fingerprint sha256=2276c3777b96d19e6359719464ede2a4f0843e6efa7cbc3f86dbaa0d41d4c5fa, paths=1963, schemas=2828; local Python 3.12 regeneration was blocked by uv dependency solving for optional tts-chatterbox-lang/russian-text-stresser, so the checked fingerprint was updated from the exact CI-reported values. Verification run before staging: cockpit vitest 3 files / 20 tests passed; apps/tldw-frontend bun run typecheck passed; Guardian targeted pytest passed; Bandit on notification_service.py reported zero findings; git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
