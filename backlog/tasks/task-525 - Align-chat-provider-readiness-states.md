---
id: TASK-525
title: Align chat provider readiness states
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-27 18:58'
labels:
  - chat
  - ux
  - bug
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix the /chat first-run readiness contradiction where the empty-state setup banner can say no LLM provider is configured while cockpit/status rails show the selected route as ready.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 When provider status reports no configured providers, /chat runtime/status rails do not show Ready for a selected model route.
- [x] #2 The setup banner and cockpit/status rails use a shared blocking model/provider readiness interpretation for the same state.
- [x] #3 A regression covers a selected tldw:gpt-4o route with no configured providers and model catalog entries lacking explicit configured flags.
- [x] #4 Focused readiness/cockpit tests pass.
- [x] #5 Verification and known skips are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
RED: focused cockpit regression failed because the runtime rail rendered Ready when provider status reported OpenAI unconfigured.

Implemented mergeChatProviderStatusIntoModels so chat readiness consumers enrich model catalog rows with provider configured state before evaluating usability.

Updated Playground cockpit readiness to fetch provider status alongside chat models and evaluate rails/status strip from the enriched model catalog.

Updated PlaygroundChat empty-state logic to count usable chat models via shared availability semantics instead of raw catalog length.

Kept warning-only degraded server readiness non-blocking in cockpit/status rails, while explicit blocked server state prefers server recovery copy.

Verification passed: focused readiness suite, 102 tests; status-strip suite, 24 tests; git diff --check.

Type-check note: frontend tsc still fails only on pre-existing unrelated CharacterListContent.design-system.test.tsx GalleryCardDensity mismatch.

Bandit skipped because this slice touched only TypeScript and TSX frontend files.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Aligned /chat provider readiness by merging provider configuration status into chat model catalog rows before usability checks. The no-provider setup banner, cockpit runtime rail, and status strip now agree when a selected model route belongs to an unconfigured provider, and warning-only degraded server health remains non-blocking while blocked server health keeps explicit recovery copy.
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
