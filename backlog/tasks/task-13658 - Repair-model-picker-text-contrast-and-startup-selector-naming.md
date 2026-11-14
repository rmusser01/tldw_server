---
id: TASK-13658
title: Repair model picker text contrast and startup selector naming
status: Done
labels:
- chat
- accessibility
- ux
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Actual axe in the owned current build reproduces Current heading and Help me choose link contrast failures. Startup template Select also lacks an accessible name. Use existing semantic readable text tokens and explicit localized label; keep WCAG thresholds.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Model picker heading and help text meet AA under both default themes.
- [x] #2 Startup template launcher has a localized accessible name.
- [x] #3 Behavior regression suites, both compilers, scoped lint, browser verification and independent review pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Original CI light model-picker baseline improved3to2 but still fails Current/help contrast. Owned actual dark reproduces heading3.67/help4.33. Whole-page scans expose additional existing badge/history contrast issues; preserve those as remaining scope. ADR assessment: no new ADR; existing semantic token and shared UI contracts. PR3229.
Mounted RED1/13pass proves unnamed actual AntD launcher before source change. Final affected native36/3suites and WebUI36/3suites pass, no skips. Both normal source compilers exit0. Independent frontend bounded source review clear; emitted CSS/browser verification still pending. Whole-page dark5 contrast nodes plus unnamed launcher and light28 contrast nodes are diagnostic evidence on saved Pro/history UI, not the fresh Casual CI baseline; keep wider history/status/setup text failures open. No global theme tokens, thresholds or CSP settings changed.
Current emitted wOvlwP6xPkDviaT1_XU24 light scan proves Current/help fixed after finite animation settle. Expanded configured catalog reveals10 residual text contrast nodes: one scope hint4.46 on surface and9 provider labels4.28 on elevated, both existing text-text-subtle. Extend bounded picker fix to actual useModelSelector group text and PlaygroundModelCatalogControls hint using existing text-text-muted. Initial unsettled scan is excluded; repository finite-animation procedure retained. Baseline not changed until exact fresh-state qualification.
Actual all-catalog toggle expands picker and retains selected model but exposes11 contrast nodes; its active text-primary on bg-primary/10 is4.00. Use text-text while retaining accent border/background and aria-pressed. Actual current group/help fix already passes. Expanded model interactions63/4suites pass in both actual runtime profiles. Source reviewer clear on provider/hint additions; this final active-state style needs emitted proof.
Final build93hY3Hu5jwB2p2_20kX5S,154pages; tokens and unchanged600/900KB budgets pass590.2/868.9KB;800manifests1127staticrefs0missing. Actual exact light/dark configured/catalog states each0serious/critical axe violations after finite animation settle, named launcher visible/model unchanged. Original9/28deepEqualtrue, baselineRewrittenfalse. Final source reviewer clear. Corrected/reran first mislabeled proof; do not credit unsettled fade scan. Review Docs/Reviews/CHAT_PICKER_ACCESSIBILITY_13658_2026_10_13.md. Broader contrast, baseline cleanup, whole-PR review/CI/integration remain open.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Repaired actual picker text owners and startup launcher name using existing semantic tokens and localized copy. No selection, focus or theme API changes.63tests in each runtime, both source compilers,0introduced static findings, final independent bounded review, current four-state browser axe and data preservation pass.
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
