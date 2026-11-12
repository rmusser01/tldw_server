---
id: TASK-13673
title: Expose owned refresh for retained native outcomes with empty history
status: Done
labels:
- chat
- ux
- recovery
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A retained native outcome can coexist with an empty captured manifest while admission is uncertain or settlement arrives later. Review is hidden when there are no nodes. Expose the existing explicit owned Refresh history operation without selecting latest, resending, changing the recovery record or inventing admission certainty.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Ready captured native empty history with a retained server outcome exposes an explicit Refresh history action.
- [x] #2 Refresh preserves current selection and recovery; changed source/view/account and pending state cannot invoke a retained action.
- [x] #3 Ordinary empty history and nonnative/unqualified sources remain unchanged; both profiles and current source/browser checks pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR assessment: existing ADR-049 original-owner selected history; existing controller refresh preserves cursor. No new protocol/storage rule. Current complete cockpit rerun17pass37.5s after one earlier timing failure; no reproducible evidence attributes that failure to13671/72. This bounded root accessibility/recovery seam is independently source-confirmed at HistorySelectionReview976 nodes-only Review gating; no new saved-boundary assumptions or sleeps.
Current combined production build wrOHaLwN_653REHUW-fbU: source hashes51 match,969 emitted manifests1132 references zero missing, token sync and590.4KB shared/869.1KB maximum budgets pass. Actual headless empty-to-late-settled refresh and all12 width/theme/CSS renderer zoom cells pass with zero axe violations. Lint found one inherited controller regression assigning its hook result during render; move only that test publication to layout effect, preserving the retained callback oracle, then rerun controller tests. No physical AT/native toolbar zoom claim.
Causal shared empty-source control,11 UI/guard controls and actual controller preservation pass. Final native3suites273checks and WebUI8suites377checks include exact owner-without-key views, no implicit selection or journal edits, retired callback and send gates. Actual API empty-to-late-settlement browser case passes without automatic turn/provider POST. Existing owned cursor/recovery remain exact. Current production build wrOHaLwN_653REHUW-fbU source51 hashes match;969 manifests1132 static references zero missing; actual token sync and590.4/600 shared869.1/900 max budgets pass. Current production dependency compilers native/WebUI exit0; browser-spec types exit0. Ten touched TS files lint0errors263unchangedwarnings0new0ignored after correcting one inherited test render side effect. Bounded independent source review clear. TS/JSON/Markdown-only scope: Bandit not applicable; whole-PR backend security remains separate. Original9conversations28rows deep-equal, immutable baseline not rewritten; compact fixture4rows unchanged. Physical packaged-extension/AT/native-toolbar zoom and participant evidence remain unrun.
Fresh independent final review found a keyboard continuity gap: refresh publishes loading and unmounts its focused button, then success mounts Review without focus handoff. Reopen this bounded task before source edits; retain mounted owned action during its read or guarded handoff through existing focus helpers, with held actual-controller keyboard and later-user-intent controls.
Owned empty-history Refresh returns keyboard focus only to its current review/refresh/error destination; later user/source/account intent retires it. Four mounted focus cases, native/WebUI recovery/status382 each, and current production real empty-refresh keyboard cell pass; original recovery/boundary retained and no replay. PR #3229; final bounded source review by two independent reviewers. TypeScript/JSON/Markdown-only unit: Bandit N/A locally; fresh full PR67-production-Python scan has0 introduced findings and0 scan errors vs exact merge-base. Installed extension/physical AT/touch/native-toolbar zoom/participants remain unrun. Overall merge and current Scene stability remain separate gates.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Owned empty-history Refresh returns keyboard focus only to its current review/refresh/error destination; later user/source/account intent retires it. Four mounted focus cases, native/WebUI recovery/status382 each, and current production real empty-refresh keyboard cell pass; original recovery/boundary retained and no replay.
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
