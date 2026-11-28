---
id: TASK-15653
title: Stabilize note creation in the Notes to Flashcards journey
status: In Progress
labels:
- ci
- testing
- notes
references:
- https://github.com/rmusser01/tldw_server/pull/3612
- https://github.com/rmusser01/tldw_server/actions/runs/45662221877/job/139401304301
- Docs/ADR/031-notes-capability-sync-domains.md
- apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.canonical-save.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The required Notes to Flashcards journey on PR #3612 timed out before it observed the note creation POST. Trace and fix the cause while retaining real note creation, reload, and flashcard assertions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Reproduce the original timeout or its trigger and verify reliable note creation and reload through the journey fixture.
- [ ] #2 Verify the focused browser journey and the new PR check.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required: no. The work restores an existing journey and preserves the Notes identity and persistence rules in Docs/ADR/031-notes-capability-sync-domains.md.
The hosted Playwright trace accepted a Save click while the first authenticated profile request was still pending. The editor reported a signed-out error and sent no Note creation POST. A deferred-profile test on the unmodified page reproduced the premature enablement. Save, Save & new, Ctrl+S, and autosave scheduling now wait for the verified Note owner while the editor preserves the draft and existing ownership fences. The deferred-profile test and the 32-case canonical-save suite passed locally alongside offline draft, save machine and monitoring tests. WebUI typechecks, targeted ESLint and English locale parity passed. The full browser journey and required checks provide the build-level verification.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Notes accepts draft text while verifying the account and enables saving once the account is verified. The focused regression verifies that the first write uses the confirmed owner and the original draft text.
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
