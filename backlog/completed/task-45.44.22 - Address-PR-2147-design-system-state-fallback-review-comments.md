---
id: TASK-45.44.22
title: Address PR 2147 design-system state fallback review comments
status: Done
parent_task_id: TASK-45.44
references:
- https://github.com/rmusser01/tldw_server/pull/2147
- apps/packages/ui/src/components/Notes/NotesEditorPane.tsx
- apps/packages/ui/src/components/Notes/NotesManagerPage.tsx
- apps/packages/ui/src/components/Option/ResearchWorkspace/ChatPane/index.tsx
- apps/packages/ui/src/components/Option/ResearchWorkspace/SourcesPane/index.tsx
- apps/packages/ui/src/components/Option/ResearchWorkspace/WorkspaceCapabilityRemediation.tsx
- apps/packages/ui/src/components/Option/WritingPlayground/WritingActionBar.tsx
modified_files:
- apps/packages/ui/src/components/Notes/NotesEditorPane.tsx
- apps/packages/ui/src/components/Notes/NotesManagerPage.tsx
- apps/packages/ui/src/components/Option/ResearchWorkspace/ChatPane/index.tsx
- apps/packages/ui/src/components/Option/ResearchWorkspace/SourcesPane/index.tsx
- apps/packages/ui/src/components/Option/ResearchWorkspace/WorkspaceCapabilityRemediation.tsx
- apps/packages/ui/src/components/Option/WritingPlayground/WritingActionBar.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address unresolved PR review comments requesting safe fallback labels around design-system state registry reads in the product-state verifier cleanup slice.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] PR #2147 unresolved review comments about direct state-registry `.label` access are addressed.
- [x] Product files do not reintroduce canonical state-label literals that fail the design-system verifier.
- [x] Focused tests, product-state verifier, TypeScript, and diff whitespace checks are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Resolved PR #2147 review comments by adding defensive fallback paths to design-system state label reads without reintroducing product-file canonical state literals. Fallback labels now come from the existing design-system constants, preserving the verifier contract while preventing runtime failures if a registry lookup returns no definition. Verification: focused Vitest set passed (6 files / 51 tests), `bun run verify:design-system-state` passed, `env NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false` passed, and `git diff --check` passed. Bandit skipped because this is TypeScript/UI-only touched code.
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
