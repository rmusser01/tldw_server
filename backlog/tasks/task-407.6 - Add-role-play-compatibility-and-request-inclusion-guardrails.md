---
id: TASK-407.6
title: Add role-play compatibility and request-inclusion guardrails
status: Done
labels:
- chat
- ux
- roleplay
- stage-6
parent_task_id: TASK-407
documentation:
- Docs/superpowers/plans/2026-05-17-main-chat-role-play-preset-remediation-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Stage 6 implementation for the main /chat role-play preset plan: add compatibility tests and user-visible guardrails for cases where selected character/persona context is excluded from the outgoing request.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Compatibility helper mirrors request inclusion/exclusion axes from usePlaygroundRawPreview.
- [x] #2 UI does not claim character/persona context will be sent when compare, image, docs/search, document context, selected knowledge, or file-retrieval RAG paths exclude it.
- [x] #3 Guardrails cover character-vs-persona behavior and shared-component compatibility.
- [x] #4 Focused Stage 6 tests, frontend checks, and browser verification are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added `role-play-compatibility.ts` as the pure request-inclusion helper used by raw preview and UI context chips.
- Added visible role-play compatibility chips for included, blended-with-sources, excluded, and prompt-override-risk states, with actions for clearing prompt behavior, opening knowledge/context controls, disabling compare mode, or reopening setup where applicable.
- Wired persona awareness into raw preview compatibility so persona context gets persona-specific copy instead of character-only labels.
- Added a web Plasmo storage shim guardrail after CDP showed `selectedAssistantSyncStorage.remove()` was deleting the local `selectedAssistant` value in the Next web app. Local and sync/session storage keys are now scoped separately while preserving existing local raw-key behavior.
- Custom prompt override-risk is covered by focused unit/hook tests; CDP browser verification covered the stable visible `/chat` states for character included, pinned-source blended, compare-mode excluded, and persona included.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stage 6 now makes `/chat` truthful about whether character/persona context is included in the current request path. It adds request-inclusion tests, UI guardrail chips, raw-preview alignment, persona-specific state, and a web storage compatibility fix required for stable role-play identity state in the Next app.
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

## Verification

- `bunx vitest run __tests__/extension/plasmo-storage.test.ts ../packages/ui/src/components/Option/Playground/__tests__/role-play-compatibility.test.ts ../packages/ui/src/components/Option/Playground/__tests__/role-play-state.test.ts ../packages/ui/src/components/Option/Playground/__tests__/usePlaygroundRawPreview.mcp-tools.test.tsx ../packages/ui/src/components/Option/Playground/__tests__/compare-interoperability.test.ts ../packages/ui/src/components/Option/Playground/__tests__/usePlaygroundContextItems.role-play.test.tsx` - passed, 6 files / 38 tests.
- `node /private/tmp/cdp-verify-stage6-role-play-compatibility.mjs` - passed via CDP on dedicated port 9333; verified character included, pinned-source blended, compare excluded, and persona included states in `/chat`.
- `bunx tsc --noEmit --pretty false` - failed on known unrelated baseline TypeScript errors in `EmbeddingsModelSelectionConfig.tsx`, `persona-visuals.ts`, and `lib/api/vnPlay.ts`; no Stage 6 files were reported.
- `git diff --check` - passed.
- Bandit not applicable: touched implementation is TypeScript/React/frontend web-shim code only.

## Known Skips / Blockers

- Browser verification did not use the setup drawer path for the custom-prompt override-risk state because the drawer/template picker path was flaky under CDP in this environment. The override-risk state and cleanup action are covered by focused hook/unit tests.
