---
id: TASK-13675
title: Show truthful saved-model guidance after private Quick Chat binding
status: Done
labels:
- chat
- ux
- roleplay
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The private popup model selector changes only local defaults while bound native replies use saved server authority. Keep the picker for prospective setup and replace it after binding with localized saved-settings guidance, retaining the existing full-chat action and original draft/source guards.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Fresh popup exposes its model picker; after binding only truthful saved-settings guidance is shown.
- [x] #2 Changing defaults cannot reintroduce a misleading bound picker; reopening a fresh popup restores supported selection.
- [x] #3 Mounted UI, ownership and wire evidence, both profiles, types, lint, current build and review pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Independent frontend Important finding at CharacterDialogs893-911/Manager267-272. Bound sender168-205 ignores local override; actual native13671wire proves saved authority. ADR049 governs; no new persistence or capability.
Visibility reassessment after two DOM visibility failures: the installed AntD modal remains ant-zoom-appear-active at opacity zero in jsdom, which does not run CSS animations. All guidance ancestors are visible except this animation container. Disable motion only in this mounted DOM fixture through installed ConfigProvider; retain the product motion and visible guidance assertion. Temporary ancestor probe will be removed.
Valid causal RED proves bound prospectivepicker remains. Complete installed Manager renderer native101 and WebUI101 pass, including truthful guidance, no picker after bind/global-model change, and fresh reopen. Fixture motion disabled only in ConfigProvider because jsdom never finishes AntD CSS appear; diagnostic probe removed. Existing13671real private hook/client/native sender/API wire proof remains unchanged production and qualifies frozen authority independently; mounted adapter UI is not that wire proof. Current production build wrOHaLwN_653REHUW-fbU source51 hashes match;969 manifests1132 static references zero missing; actual token sync and590.4/600 shared869.1/900 max budgets pass. Current production dependency compilers native/WebUI exit0; browser-spec types exit0. Ten touched TS files lint0errors263unchangedwarnings0new0ignored after correcting one inherited test render side effect. Bounded independent source review clear. TS/JSON/Markdown-only scope: Bandit not applicable; whole-PR backend security remains separate. Original9conversations28rows deep-equal, immutable baseline not rewritten; compact fixture4rows unchanged. Physical packaged-extension/AT/native-toolbar zoom and participant evidence remain unrun.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Keep model selection for fresh private setup and replace the bound picker with localized saved-model/provider guidance and existing full-chat review action.
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
