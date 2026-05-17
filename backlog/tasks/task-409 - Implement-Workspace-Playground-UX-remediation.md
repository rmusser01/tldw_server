---
id: TASK-409
title: Implement Workspace Playground UX remediation
status: In Progress
references:
- Docs/superpowers/specs/2026-05-17-workspace-playground-ux-remediation-design.md
- Docs/superpowers/plans/2026-05-17-workspace-playground-ux-remediation-implementation-plan.md
- TASK-407
- TASK-408
- apps/packages/ui/src/components/Option/WorkspacePlayground/index.tsx
- apps/packages/ui/src/components/Option/WorkspacePlayground/SourcesPane/index.tsx
- apps/packages/ui/src/components/Option/WorkspacePlayground/SourcesPane/AddSourceModal.tsx
- apps/packages/ui/src/components/Option/WorkspacePlayground/ChatPane/index.tsx
- apps/packages/ui/src/routes/option-workspace-playground.tsx
- apps/tldw-frontend/extension/routes/option-workspace-playground.tsx
- apps/test-utils/workspace-playground/page.ts
- apps/test-utils/workspace-playground/contract.ts
modified_files:
- apps/packages/ui/src/components/Option/WorkspacePlayground/__tests__/WorkspacePlayground.desktop-layout.test.tsx
- apps/tldw-frontend/e2e/utils/page-objects/WorkspacePlaygroundPage.ts
- apps/tldw-frontend/e2e/workflows/workspace-playground.spec.ts
- apps/test-utils/workspace-playground/page.ts
- apps/test-utils/workspace-playground/contract.ts
- apps/packages/ui/src/components/Option/WorkspacePlayground/index.tsx
- apps/packages/ui/src/routes/option-workspace-playground.tsx
- apps/tldw-frontend/extension/routes/option-workspace-playground.tsx
- apps/packages/ui/src/components/Option/WorkspacePlayground/SourcesPane/index.tsx
- apps/packages/ui/src/components/Option/WorkspacePlayground/SourcesPane/AddSourceModal.tsx
- apps/packages/ui/src/components/Option/WorkspacePlayground/SourcesPane/media-library-normalization.ts
- apps/packages/ui/src/components/Option/WorkspacePlayground/__tests__/SourcesPane.stage2.test.tsx
- apps/packages/ui/src/components/Option/WorkspacePlayground/__tests__/AddSourceModal.stage2.intake.test.tsx
- apps/packages/ui/src/components/Option/WorkspacePlayground/__tests__/media-library-normalization.test.ts
- apps/packages/ui/src/components/Option/WorkspacePlayground/ChatPane/index.tsx
- apps/packages/ui/src/components/Option/WorkspacePlayground/__tests__/ChatPane.stage2.test.tsx
- backlog/tasks/task-409 - Implement-Workspace-Playground-UX-remediation.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved Workspace Playground UX remediation plan. Scope covers shared WebUI/extension fixes for body-level collapsed-pane restore controls, bounded route layout so the chat composer is visible without page scroll, Add Sources discoverability, My Media media-library response normalization and selection behavior, chat model picker repair, and focused verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-17-workspace-playground-ux-remediation-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Workspace Playground UX remediation through focused commits. Verified
focused Workspace Playground unit tests, shared UI and extension OpenAPI contract
checks, and browser-rendered /workspace-playground behavior. Visual approval is
still pending before marking this task Done. Targeted full WebUI Playwright
workflow ran with 9/12 passing; remaining failures were outside this remediation
scope: Cmd/Ctrl+K global search shortcut tests and an older Compare Sources
studio-button expectation.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
