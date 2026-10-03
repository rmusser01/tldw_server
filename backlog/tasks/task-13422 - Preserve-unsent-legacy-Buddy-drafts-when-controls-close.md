---
id: TASK-13422
title: Preserve unsent legacy Buddy drafts when controls close
status: Done
created_date: 2026-10-02 17:59
assignee:
- '@codex'
labels:
- buddy
- uat
updated_date: 2026-10-02 17:59
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Closing the legacy Persona Buddy controls discards unsent text. Retain the draft for the existing Buddy lifetime so users can close and reopen the controls without losing their work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Closing and reopening controls for the same Persona and session preserves unsent text.
- [x] #2 Closing controls hides the composer from keyboard and accessibility navigation.
- [x] #3 Pending and failed sends retain their existing request identity and draft guards across close and reopen.
- [x] #4 Changing Persona or unmounting the Buddy cannot expose the previous Persona draft.
- [x] #5 Targeted regressions and a disposable real-browser replay verify the fix without provider or microphone requests.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce close/reopen draft loss through the real Dock and Popover, including pending-send and Persona-lifetime controls.
2. Retain the existing Popover component while closed, render no controls, and reset its state on Persona change. Keep existing send/session routing and draft revision checks.
3. Run focused Buddy tests and scoped static checks, replay the original failure in a copied authenticated browser profile, and record source-bound evidence.
ADR required: no. ADR paths: backlog/decisions/005-independent-buddy-bindings-and-work-ownership.md and Docs/ADR/046-persona-live-conversation-and-voice-runtime.md. Reason: local presentation lifetime repair; no storage, service, authentication, or worker boundary changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Closing legacy Buddy controls now keeps the existing Popover state, preserving unsent text, pending-send state, retry identity, and draft revision guards. Closed controls render no DOM; changing Persona resets the component. Session routing is unchanged.

- Changed BuddyShellDock, BuddyShellPopover, added five Dock draft regressions, and documented the draft lifetime in Persona_Buddy_Guide.md.
- Focused verification: 86 tests passed in six Buddy/lifecycle files. The final new tests against unchanged production code failed in four expected cases; the unmount control passed. The repaired files were restored afterward.
- Real-browser replay on a fresh copy of the prior authenticated UAT profile retained the exact unsent draft after both close-button and Escape dismissal. The closed composer was absent from the accessibility tree and DOM. No Send, Start, Connect, or microphone controls were used.
- ESLint and git diff --check passed. The new test passes Prettier; existing production files retain their established no-semicolon style and pre-existing formatter differences. ESLint emitted only the shared-package pages-directory notice. Bandit is inapplicable to this TypeScript/test/documentation change.
- Independent code review found no actionable issues. Private source hashes, test receipts, screenshot, and cleanup evidence: /private/tmp/buddy-draft-fix-uat-20261002/uat-result.json. Temporary listeners and tab were closed; the previous profile database and preserved acceptance checkout were unchanged.
- ADR required: no. Existing ownership boundaries remain governed by backlog/decisions/005-independent-buddy-bindings-and-work-ownership.md and Docs/ADR/046-persona-live-conversation-and-voice-runtime.md.
- Scope limits: no full suite, physical microphone, audibility, or provider request test. Persona change, Buddy unmount, and page reload end the draft lifetime. The historical repeated-reload cause and overall Buddy acceptance remain open.

2026-10-02 publication qualification: current server dev df17c8ac3f introduced an unrelated TASK-13420. Recreated this unpublished Buddy record as TASK-13422 with the canonical Python CLI after scanning all reachable and local task IDs, then removed only this branch's superseded record. Existing acceptance outcomes, plan and evidence are preserved. Rebase retained the repair patch exactly; no incoming apps files changed, so prior tests/browser evidence remain applicable with their original source attribution. The original local receipt and commit remain unchanged.
October 3 follow-up: the published Buddy guide now includes the source closed-draft guidance after the canonical refresh performed under TASK-13423. All 33 targeted docs-refresh parity tests pass; no draft implementation changed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Legacy Buddy close/reopen draft loss is repaired and verified by focused regressions, baseline negative control, independent review, and live browser replay. Draft state remains scoped to the current Persona and Buddy lifetime.
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
