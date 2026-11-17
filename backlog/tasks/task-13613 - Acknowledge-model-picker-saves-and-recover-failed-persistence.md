---
id: TASK-13613
title: Acknowledge model picker saves and recover failed persistence
status: Done
labels:
- chat
- ux
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix the important PR3229 model-picker persistence defect: consumed save completion, visible failed preference save, bounded revision-specific retry, and stale failure retirement in Full/Compact. Keep editor rejection semantics. Put the real WebUI storage persistence test under its owning runner instead of changing all native extension tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Physical preference-write failure is handled and visible in both pickers, with direct bounded retry.
- [x] #2 Retry uses the original current revision; later selection or stale failure cannot overwrite newer state.
- [x] #3 Acknowledged editors still receive rejected completion; native extension storage coverage is retained.
- [x] #4 Behavior tests, both normal types/configs, shared build budget and independent review qualify the change.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required:no, existing acknowledged-modelsave/owner behavior; no storage protocolchange. Userallremaining PR3229 fixes/review/rebase/mergeauthorized. Plan Docs/Plans/IMPLEMENTATION_PLAN_chat_model_save_recovery_13613.md.
Reviewed Compact lifetime defect reproduced: old conditional host lost failure owner, catalog loading/error bypassed feedback (4 failed/2 passed). ModelSelect now stays mounted in connected nonPro inline form with showPicker hiding only trigger; failure feedback remains outside catalog returns. Final four-suite run 55 passed; both normal Node26 WebUI tsconfig.json and extension tsconfig.compile.json passed. Bounded independent reviewer found no Critical/Important/Minor remaining in this unit. Docker-profile local Turbopack is unavailable because pre-existing dependency symlinks point outside its filesystem root; supported production webpack build is running. CI Turbopack budget and whole-PR gates remain open.
Qualified actual Compact JSX plus storage/picker/model-validation adapters:97passed; unchanged native shared configuration:52passed; both normal types passed. Supported production quickstart webpack build/token/budgetpassed (shared596.4KB/chat798.7KB). CI Turbopack and standalone packaging remain unqualified: dependency symlink root restriction/traced copy warnings. No threshold increases. Bounded reviewer found no remaining findings after fixture repairs. Review Docs/Reviews/CHAT_MODEL_SAVE_RECOVERY_13613_2026_10_12.md; completed unique plan retired to /private/tmp. Bandit not applicable (no Python production changes). WholePR/H2/finalreview/rebase/merge remain open.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fix model preference failure visibility and revision-specific retry in Full/Compact, keep Compact recovery mounted across successful selection, preserve editor rejection, and move real WebUI storage tests to their owning runner. Qualified97WebUI+52native-config tests, normal types and supported webpack budget; CI Turbopack/standalone packaging explicitly remain open.
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
