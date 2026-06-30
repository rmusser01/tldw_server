---
id: TASK-399
title: Implement main /chat model-provider confidence
status: Done
assignee: []
created_date: '2026-05-16 01:41'
updated_date: '2026-05-16 02:17'
labels:
  - chat
  - cockpit
  - webui
  - ux
dependencies: []
references:
  - >-
    Docs/superpowers/specs/2026-05-15-main-chat-cockpit-maturity-roadmap-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR3 slice for the main WebUI /chat cockpit maturity roadmap. Scope is strictly the main chat page, not the browser-extension sidepanel/sidebar. Make model/provider selection, configured-vs-catalog discovery, recent/frequent model surfacing, and provider:model-scoped settings visibly reliable from the /chat cockpit.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Default model selector scope shows configured usable provider/model choices, not catalog-only or unusable entries.
- [x] #2 Users can explicitly broaden search to all known catalog models without making catalog noise the default.
- [x] #3 Recently/frequently selected configured models are discoverable without masking current provider grouping.
- [x] #4 Provider:model scoped settings persist, restore, and remain isolated for same model ids under different providers.
- [x] #5 Runtime rail and composition preview clearly expose provider route and settings scope for the selected model.
- [x] #6 Focused unit/component coverage and real-server /chat proof cover the model selector and settings-scope flow.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Plan file: Docs/superpowers/plans/2026-05-16-chat-cockpit-model-provider-confidence.md

Slice sequence:
1. Normalize Playground provider-qualified model ids so user-visible/settings scopes use provider:model without leaking internal tldw: prefixes.
2. Make chat availability validation accept provider-qualified selected models from the main /chat selector.
3. Certify configured-default/catalog-explicit/recent-model UX contracts.
4. Add real-server /chat proof for selecting a real configured model and sending a conversation.
5. Run focused Vitest, real-server Playwright, git diff --check, design-system verification if classes change, and record Bandit skip if no Python files are touched.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Initial inspection found the main /chat cockpit already uses PlaygroundForm + useModelSelector, not the older shared ModelSelect. Existing code covers configured/catalog scope and recent usage, but provider-qualified selector keys can currently retain the internal tldw: model prefix, which risks confusing scope labels and blocking submit validation after a dropdown selection.

Implementation complete for the model/provider confidence slice. Normalized tldw transport model ids at the selector boundary, added provider-qualified availability validation, preserved configured-by-default/catalog-explicit model search, added recent/current model promotion coverage, and added a real-server /chat proof that selects a real configured provider:model, verifies composition preview/runtime rail provider scope, sends a live conversation, and saves a screenshot.

Verification recorded:
- bunx vitest run src/hooks/playground/__tests__/modelSelectorUtils.test.ts: PASS, 6 tests.
- bunx vitest run src/utils/__tests__/chat-model-availability.test.ts: PASS, 13 tests.
- bunx vitest run focused cockpit/model/settings suite: PASS, 9 files, 80 tests.
- bunx playwright test e2e/workflows/chat-cockpit.real-server.spec.ts --project=chromium --grep "model provider confidence": PASS, 1 test, real running server at http://127.0.0.1:8000.
- bunx playwright test e2e/workflows/chat-cockpit.real-server.spec.ts --project=chromium: PASS, 9 tests, real running server at http://127.0.0.1:8000.
- git diff --check: PASS.
- bun run verify:design-system-state: PASS with existing allowed legacy product-state exceptions.
- Bandit: skipped because this slice touched no Python files.

Screenshot proof: apps/tldw-frontend/test-results/workflows-chat-cockpit.rea-b5bfe--selection-and-conversation-chromium/chat-cockpit-model-provider-conversation.png

Final verification refresh before commit:
- Focused Vitest cockpit/model/settings suite: PASS, 9 files, 80 tests.
- Full real-server Playwright /chat cockpit spec: PASS, 9 tests in 32.0s against http://127.0.0.1:8000.
- git diff --check: PASS.
- bun run verify:design-system-state: PASS with existing allowed baseline exceptions.
- Bandit remains skipped because no Python files were touched.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the main /chat model-provider confidence slice. The cockpit selector now keeps configured usable models as the default path, allows explicit all-catalog search, exposes provider:model keys without leaking internal tldw: transport ids, accepts provider-qualified selections in chat availability validation, and proves the flow through focused unit/component tests plus a real-server /chat conversation screenshot.
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
