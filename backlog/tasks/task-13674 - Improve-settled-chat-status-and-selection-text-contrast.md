---
id: TASK-13674
title: Improve settled chat status and selection text contrast
status: Done
labels:
- chat
- ux
- accessibility
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The complete stable-theme native accessibility matrix identifies three actual text contrast defects: Draft saved (4.46 on white), active small control text (3.17 on dark tinted surface), and success status text (2.69 on light tinted surface). Use existing readable semantic text tokens while preserving meaningful colored borders/icons/backgrounds and status labels. Repair shared/full/compact callers once without global palette changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Settled native WebUI light/dark at320/390/desktop has no AA contrast findings on supported chat controls.
- [x] #2 Full and compact Draft saved feedback and shared selected/status labels retain states and readable text without changing actions or ownership.
- [x] #3 Relevant tests, source types/lint, production gates and independent bounded review qualify the changed source.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
MasterTASK13458 matrix uses actual installed axe and screenshots; finite animations settle before scanning. Earlier transient theme-transition findings excluded; stable rules identify only three target families. Renderer200%stress is explicit CSS layout evidence, not native browser-toolbar zoom or physical AT/participant results. ADR assessment: existing semantic theme/component patterns; no new durable architecture rule.
Stable12-cell actual Axe identified persistent Draft saved4.46, activeCustomize3.17 and successstatus2.69 text failures. Existing semantic readable tokens fix full/compact saved text and shared status/control producers, keeping tint/dots/actions. Native5suites104 and connectedWebUI377 pass. All12actual320/390/1280/light/dark/CSSzoom1/2 cells now have0axeviolations and retain exactdraft/accessiblename/focus. Narrow CSSzoom screenshot is stress evidence, not total physical native zoom/reflow proof. Current production build wrOHaLwN_653REHUW-fbU source51 hashes match;969 manifests1132 static references zero missing; actual token sync and590.4/600 shared869.1/900 max budgets pass. Current production dependency compilers native/WebUI exit0; browser-spec types exit0. Ten touched TS files lint0errors263unchangedwarnings0new0ignored after correcting one inherited test render side effect. Bounded independent source review clear. TS/JSON/Markdown-only scope: Bandit not applicable; whole-PR backend security remains separate. Original9conversations28rows deep-equal, immutable baseline not rewritten; compact fixture4rows unchanged. Physical packaged-extension/AT/native-toolbar zoom and participant evidence remain unrun.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Improve the three measured text contrast producers through existing semantic tokens, retaining status and control behavior.
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
