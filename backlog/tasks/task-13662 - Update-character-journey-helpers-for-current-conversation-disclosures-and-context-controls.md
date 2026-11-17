---
id: TASK-13662
title: Update character journey helpers for current conversation disclosures and context
  controls
status: Done
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
- [x] #3 Behavior/static/source review and current owned-page controls pass, with full critical runner qualification and limits tracked.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
PR3229 critical job115425762846: phase6desktopBuddyPersona absent/tablet+mobileContext tab wrong role;4phase7/createCharacter cases failChatPage character trigger while CasualCustomize collapsed. Two other cited/prompt failures are separate ordinaryserver-save receipt fixtures, reserve separate unit. ADR required:no, E2E navigation contracts only, no production/permission/persistence changes. Governing current UI source ComposerToolbar/CockpitShell and existing13656 disclosure helper.
Checkpoint: oldactualclass fails10s missingtrigger; currentclass selects ownsyntheticchar viarealCasualCustomize. Final actualsourcephase6 helper functions pass sessions/setupdialog geometry ondesktop1440/tablet768/mobile390; declaredlocatorassertadapter,not fullrunner.7DOMboundary/readinessunit checks pass incldelayedCasual; Playwrightlists7cases/3files noautostart. Independentfrontendboundedreviewclear after Minor delayedmountfixed. Original9/28 andseparate synthetic4 preserved. Review Docs/Reviews/CHAT_CHARACTER_JOURNEY_CONTROLS_13662_2026_10_13.md. AC3/Stage3 remainsopen untilfullcriticaljourneysqualify; do not markDone/wholePRready.
Actual full7case native headless runner started after TASK13663 qualified2journeys. Phase6 desktop/tablet/mobile complete cases pass. Phase7 blockedmodel selection fails because fixture seeds unavailablemodel before trackedcharacterpreflight, which correctly refuses publication before identity write (independentsource audit AssistantSelect619/PlaygroundForm2017). Providerfailure andsuccess cases hit120s completionresponse waits; nativecall transport evidence stillunderinvestigation. No timeoutextension/guardbypass/closure. TASK13662 remainsInProgress; normal runner report pending. PRcurrentlydraft/conflicting, MERGE_QUEUEunset; no merge authorization gate bypass.
Full current native7case character qualification completed under TASK13664:desktop/tablet/mobile controls and firstcreate, realblockedsetup, providerrecovery/originalretry/cold persistence allpass0skip0flake23.95s. Character helper/publicCasual/Pro disclosure remains unchanged; no forcedmode or API response mocking. Rootprovidererror/transientacceptedsource correction and exact original nativeprotocol evidence in Docs/Reviews/CHAT_SAVED_ROLEPLAY_FEEDBACK_GUARD_13664_2026_10_13.md. Bandit N/A fixture/Markdown-only. Broadercockpit/master/H2 and finalPRmerge remainopen.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed current native character control journeys across desktop/tablet/mobile and original roleplay success/recovery contracts. Correct public disclosures and setup refusal retain original drafts and identity. Seven actual cases pass; broader cockpit and final PR gates remain open.
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
