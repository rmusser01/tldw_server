---
id: TASK-528
title: Restore chat model scope toggle in selector
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-27 19:43'
labels:
  - chat
  - ux
  - model-selector
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address /chat UX rebaseline F6 by wiring the existing configured/catalog model scope controls into the main chat model selector dropdown. Keep scope limited to the /chat model selector and existing selector tests/real-server contract.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 When the /chat model selector opens, it exposes the existing configured/catalog scope toggle and scope hint before the model list.
- [x] #2 The configured scope shows usable configured models and the catalog scope shows all known models without replacing the main selector button contract.
- [x] #3 The dropdown keeps model search, sort, and help link behavior available.
- [x] #4 Regression coverage proves provided catalog controls render in ChatModelSelectorDropdown and PlaygroundForm wires them into the selector.
- [x] #5 Focused /chat model selector tests pass and verification/known skips are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Starting focused /chat F6 model selector slice. Investigation found PlaygroundModelCatalogControls and useModelSelector scope state already existed, but PlaygroundForm did not pass those controls into ChatModelSelectorDropdown, whose popup still rendered the older inline search/sort header.

Implemented catalogControls support in ChatModelSelectorDropdown and wired PlaygroundForm to pass PlaygroundModelCatalogControls into the model selector popup. Kept the fallback inline search/sort header for selector usages that do not provide catalogControls.

Verification: RED focused run failed as expected because the dropdown ignored supplied catalog controls and PlaygroundForm did not contain catalogControls={modelCatalogControls}. GREEN focused run passed 8 tests across ChatModelSelectorDropdown.character-usability and Playground.cockpit-regression.guard. Broader nearby run passed 47 tests across selector, cockpit guard, cockpit a11y, and cockpit shell. TypeScript compiler gate still fails only on known baseline CharacterListContent.design-system.test.tsx GalleryCardDensity error. git diff --check passed. Bandit not run because touched code is TS/TSX UI only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Restored the configured/catalog model scope affordance in the /chat model selector by mounting the existing PlaygroundModelCatalogControls inside ChatModelSelectorDropdown from PlaygroundForm. The selector button contract stays unchanged, popup search/sort/help behavior remains available through the existing controls, and fallback popup behavior is preserved for other usages.
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
