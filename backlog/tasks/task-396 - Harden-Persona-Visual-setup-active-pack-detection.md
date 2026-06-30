---
id: TASK-396
title: Harden Persona Visual setup active-pack detection
status: Done
assignee: []
created_date: '2026-05-16 00:56'
updated_date: '2026-05-16 01:15'
labels:
  - persona
  - persona-visual
  - review-fix
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1730'
  - 'https://github.com/rmusser01/tldw_server/pull/1725'
  - 'https://github.com/rmusser01/tldw_server/pull/1735'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address the still-valid Persona Visual setup review finding from closed PR #1730 after PR #1725 merged: first-run setup choices must not appear when the visual-pack list response returns an active_pack separately from packs. Keep the follow-up focused on the setup gate and regression coverage against the merged dev implementation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 VisualPackEditor treats a separately returned active_pack as an active visual for setup-choice gating.
- [x] #2 A focused regression test covers a response with packs excluding active_pack and active_pack populated.
- [x] #3 Review findings from the closed superseded PR are triaged with still-valid items fixed and obsolete items skipped with a reason.
- [x] #4 Focused Persona Visual tests pass and diff whitespace validation is clean.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Triaged closed PR #1730 review comments against current origin/dev: setup error gating and draft-title helper coupling are obsolete in the superseding PR #1725 implementation; active_pack returned separately from packs remained valid and is fixed in this follow-up.

Verification: added red regression for active_pack outside packs; initial focused file run failed on the new test as expected. After the fix, targeted regression passed, full VisualPackEditor suite passed with --testTimeout=20000, git diff --check passed, and bun run lint exited 0 with existing warnings only.

Bandit skipped because the touched implementation is frontend TypeScript plus Backlog task metadata only.

Follow-up draft PR opened: https://github.com/rmusser01/tldw_server/pull/1735

Addressed PR #1735 CodeRabbit test-coverage comment by asserting the separate active_pack response is rendered with active status before asserting the setup card is absent. Validation: targeted regression passed, git diff --check passed, and bun run lint exited 0 with existing warnings only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Merged active_pack into VisualPackEditor local pack state when the backend returns it separately from packs, so first-run setup choices do not appear when an active visual pack exists. Added regression coverage for the separate active_pack response shape and validated the focused Persona Visual component suite plus lint/whitespace checks.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
