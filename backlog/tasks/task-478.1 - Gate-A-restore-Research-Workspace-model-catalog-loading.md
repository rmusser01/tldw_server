---
id: TASK-478.1
title: 'Gate A: restore Research Workspace model catalog loading'
status: Done
labels:
- research-workspace
- uat
- gate-a
- frontend
- models
priority: Critical
milestone: Research Workspace UAT Remediation
parent_task_id: TASK-478
modified_files:
- apps/packages/ui/src/hooks/playground/modelSelectorUtils.ts
- apps/packages/ui/src/hooks/playground/useModelSelector.tsx
- apps/packages/ui/src/hooks/playground/__tests__/modelSelectorUtils.test.ts
- apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/ChatPane.stage2.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
User-visible failure: the model selector shows "No models available" even when the backend is connected. Console error from UAT: `Failed to fetch chat models: TypeError: this.isSelectableChatModel is not a function`, with source hint around `apps/packages/ui/src/services/tldw/TldwModels.ts:193`.

User goal: choose a configured provider/model before asking questions, using local llama.cpp or any configured provider.

Scope:
- Fix the frontend model service binding/selection path so Research Workspace can load selectable chat models.
- Verify provider/model metadata is normalized consistently for Research Workspace, Chat, and any shared selectors.
- Add regression coverage for the broken method-binding/destructuring path.
- Validate against a live backend and WebUI via CDP/Playwright.

Acceptance criteria:
- Research Workspace model selector lists configured selectable models when the backend has providers.
- Empty/no-provider state is explicit and actionable, not a false "No models available" caused by a client exception.
- Console has no `isSelectableChatModel` error.
- User can select a model and the selection persists across normal Research Workspace interactions.

Blocks: all chat, RAG, and Studio validation.
Parallelization: backend ingestion-status investigation may run in parallel, but any UI RAG/Studio behavior depends on this task being resolved.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Bound `TldwModelsService` public model-loading methods so callback/destructuring usage preserves `this` for selectable chat-model filtering.
- Kept provider availability filtering for explicitly unconfigured, disabled, failed, unavailable, and not-configured chat models while preserving legacy metadata-less chat models.
- Avoided caching empty `fetchChatModels` startup results over later configured backend model responses.
- Added a Research Workspace model-selector retry on dropdown open with `forceRefresh: true` when the composer has no models.
- Added focused regression tests for the unbound model service method, empty-startup cache behavior, and selector-open retry.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed current Gate A model selector compatibility cleanup in this worktree. Verification: `bunx vitest run src/components/Option/ResearchWorkspace/__tests__/AddSourceModal.stage2.intake.test.tsx src/components/Option/ResearchWorkspace/__tests__/ChatPane.stage2.test.tsx src/hooks/playground/__tests__/modelSelectorUtils.test.ts --maxWorkers=1 --no-file-parallelism` passed: 3 files, 38 tests. Live CDP UAT selected a configured model (`Ollama / gemma3:1b`) successfully and a fresh `/research-workspace` tab reported 0 console errors after Add Sources -> My Media. Bandit is not applicable to frontend-only model-selector files.
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
