---
id: TASK-13710
title: Fix post-merge Chat UX reassessment findings across both clients
status: In Progress
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement all six findings and remaining improvements from TASK-13709 on latest dev in a new PR. User approved character-first roleplay setup and filtered paginated compact character history, root-cause repairs, independent review and merge after checks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Disconnected and unconfigured Chat uses one accurate connection-first recovery path without weakening ownership or draft guards
- [ ] #2 Both extension starter prompts update controlled draft state and survive interaction
- [ ] #3 Compact tracked sessions use server character filtering and pagination with honest loading stale error and empty states and disambiguating metadata
- [ ] #4 Roleplay setup leads with Character and provides optional details and accessible current/proposed preview labels
- [ ] #5 Task language customization disclosure and active legacy character-picker error recovery address remaining report improvements
- [ ] #6 Regression tests production verification independent review and new PR against latest remote dev qualify the change before merge
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Design approved by user in direct reply to TASK-13709 review: recommended character-first and filtered paginated history. ADR required: no; local shared-component state, recovery and presentation repairs reuse current APIs and trust boundaries. Governing ownership/persistence rules are unchanged. Existing isolated managed remediation worktree reused; new branch codex/chat-ux-followup-20261015 begins origin/dev 2e48dc91848887dcb74ebe39d6ff877fed4d64bd. Plan: Docs/Design/IMPLEMENTATION_PLAN_chat_ux_followup_13710.md. Preserve original nine conversations/28 messages and unrelated primary-checkout work.
Stage1 setup/readiness and controlled starter group47 passes after observed regression failures. History21 passes and scoped independent reviewer clears final-page boundary fix. Customization test reassessment after three verification passes: old group-containment assertions were still checking relocated Tools/Compare in primary groups; inspected complete four-layout test, retained all controls/focus/visibility assertions and explicitly opens Advanced. No test timeouts/retries changed. Stage2/3 integration and final qualification remain ongoing.
Stages 1–3 implemented. Shared starter fix proven red on old source (both suggestions lose text after controls) and green through actual production message effect and submit callback: 34 tests pass, exactly one send each. Independent follow-up reviewer cleared duplicate setup guidance and stronger starter evidence at source/unit level; installed-browser confirmation pending. Fresh frozen-lock Bun production WebUI build, WebUI typecheck, strict shared-boundary types, Chrome production extension build and compile all pass. Touched ESLint31 files has0 errors and163 baseline warnings, unchanged. Shell qualification:86/86 adjacent tests plus corrected64/64 shell tests pass; old labels/navigation expectations updated to intended behavior. Original9 chats/28 rows remain deep-equal to immutable baseline. No Python changed: Bandit not applicable to source scope.
Qualification report: Docs/Reviews/CHAT_UX_FOLLOWUP_13710_QUALIFICATION_2026_10_15.md. WebUI touched16 suites362 tests pass. Installed production extension confirms both real starter/control cycles and all6 authoritative older page2 rows (147 filtered character sessions), Previous navigation and responsive roleplay. Settled WebUI drawer390px wide at viewport390. Source provenance4318 production app files matches;969 emitted JSON manifests reference1130 existing JS/CSS assets. Frozen lock unchanged. Independent follow-up source review has no actionable findings. Physical AT, permission-grant UI and Firefox/Edge runtime not checked. New PR/CI/human Change summary remain before merge.
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
