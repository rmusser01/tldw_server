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
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Expose explicit owned Refresh history for a qualified empty native source with retained server outcome. Preserve current cursor/recovery and refuse stale owner/view/request callbacks.
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
