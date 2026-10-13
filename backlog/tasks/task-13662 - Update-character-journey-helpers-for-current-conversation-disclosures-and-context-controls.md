---
id: TASK-13662
title: Update character journey helpers for current conversation disclosures and context
  controls
status: In Progress
labels:
- chat
- testing
- ux
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critical UX journeys still target collapsed character controls, removed Buddy fallback and a button as tab. Reuse current public Casual Customize/Pro Options, actual setup action/overflow and context button semantics; retain complete character/setup/viewport assertions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Shared ChatPage character selection reveals current controls without changing conversation data or dropping selected-character checks.
- [x] #2 Phase6 setup/context helpers use actual button/disclosure roles across desktop/tablet/mobile and keep geometry/readiness assertions.
- [ ] #3 Behavior/static/source review and current owned-page controls pass, with full critical runner qualification and limits tracked.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
PR3229 critical job115425762846: phase6desktopBuddyPersona absent/tablet+mobileContext tab wrong role;4phase7/createCharacter cases failChatPage character trigger while CasualCustomize collapsed. Two other cited/prompt failures are separate ordinaryserver-save receipt fixtures, reserve separate unit. ADR required:no, E2E navigation contracts only, no production/permission/persistence changes. Governing current UI source ComposerToolbar/CockpitShell and existing13656 disclosure helper.
Checkpoint: oldactualclass fails10s missingtrigger; currentclass selects ownsyntheticchar viarealCasualCustomize. Final actualsourcephase6 helper functions pass sessions/setupdialog geometry ondesktop1440/tablet768/mobile390; declaredlocatorassertadapter,not fullrunner.7DOMboundary/readinessunit checks pass incldelayedCasual; Playwrightlists7cases/3files noautostart. Independentfrontendboundedreviewclear after Minor delayedmountfixed. Original9/28 andseparate synthetic4 preserved. Review Docs/Reviews/CHAT_CHARACTER_JOURNEY_CONTROLS_13662_2026_10_13.md. AC3/Stage3 remainsopen untilfullcriticaljourneysqualify; do not markDone/wholePRready.
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
