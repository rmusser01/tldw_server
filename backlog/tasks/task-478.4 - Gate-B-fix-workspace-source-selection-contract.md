---
id: TASK-478.4
title: 'Gate B: fix workspace source selection contract'
status: Done
labels:
- research-workspace
- uat
- gate-b
- frontend
- selection
- rag
priority: High
milestone: Research Workspace UAT Remediation
parent_task_id: TASK-478
documentation:
- Docs/superpowers/plans/2026-05-24-research-workspace-source-selection-contract.md
modified_files:
- Docs/superpowers/plans/2026-05-24-research-workspace-source-selection-contract.md
- apps/packages/ui/src/components/Option/ResearchWorkspace/workspace-server-reconcile.ts
- apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/workspace-server-reconcile.test.ts
- apps/packages/ui/src/components/Option/ResearchWorkspace/index.tsx
- apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.stage3.test.tsx
- apps/packages/ui/src/services/tldw/domains/workspace-api.ts
- apps/packages/ui/src/store/workspace-api.ts
- apps/packages/ui/src/store/workspace-slices/sources-slice.ts
- apps/packages/ui/src/store/__tests__/workspace-api-first.test.ts
- apps/packages/ui/src/store/__tests__/workspace.test.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
User-visible failure: individual source checkbox clicks did not reliably persist selection, while `Select all` did. Studio also disagreed with selected-source state by saying no sources were selected while status APIs reported selected sources.

User goal: choose exactly which sources participate in RAG, Studio outputs, export, and later extension/agent handoffs.

Scope:
- Define the canonical selected-source state: local UI only, backend persisted, or both with clear synchronization semantics.
- Fix individual checkbox interaction, bulk select/deselect, folder-filtered selection, and reload persistence according to that contract.
- Ensure selected-source counts, disabled states, RAG mode, and Studio enablement all read the same source-selection contract.
- Add keyboard-accessible selection behavior and tests for single, bulk, filtered, and reload paths.

Acceptance criteria:
- Individual checkbox selection persists and is reflected everywhere selected sources are consumed.
- `Select all` and filtered/bulk selection produce predictable counts and do not silently select hidden/unintended sources unless explicitly designed.
- Source selection survives normal reload/workspace switching if the chosen contract says it should.
- CDP/Playwright validation confirms the selected-source state drives RAG and Studio consistently.

Depends on: TASK-478.3 for final readiness/selection semantics; can start UI-state investigation earlier.
Blocks: TASK-478.5 and TASK-478.6.
Parallelization: can proceed in parallel with source acquisition/layout tasks after the status contract is settled.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Implemented selected-source state as local optimistic `selectedSourceIds` synchronized to the backend batch selection endpoint.
- Reconciliation now creates missing workspace source rows with selected bits from local state and updates existing rows through `/api/v1/workspaces/{workspace_id}/sources/selection`.
- Server hydration now maps selected backend workspace source rows back into local state.
- Live CDP validation found a follow-on issue: source status projection could mark fresh media as queued/processing, clear local selection, and sync an empty backend selection. The store now preserves selected intent for processing sources while `getSelectedMediaIds()` remains ready-only and terminal errors still clear selection.
- Fixed the workspace quota warning test mock to patch the concrete `localStorage.setItem` used in the current Vitest runtime.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
- Fixed the Research Workspace selected-source contract so individual checkbox selection, backend workspace source rows, status APIs, and RAG request construction agree on the selected source set.
- Verified with focused Vitest coverage and a live backend/WebUI CDP pass. The CDP pass seeded two real media documents, selected one source, confirmed `/sources/selection` received the selected source id, confirmed `/sources/status` reported only that row as selected, and confirmed `/api/v1/rag/search` used only the selected media id.
- Known non-blocker: the chat completion after RAG returned `503 no_provider_configured` in the live test environment; the source-selection/RAG request contract under this task passed before that provider-layer failure.
- Bandit skipped because this task touched frontend TypeScript/tests/docs/backlog only, with no Python/backend code changes.
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
