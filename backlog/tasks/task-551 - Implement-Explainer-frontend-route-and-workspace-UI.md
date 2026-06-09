---
id: TASK-551
title: Implement Explainer frontend route and workspace UI
status: To Do
labels:
- frontend
- explainer
- implementation
priority: High
references:
- TASK-546
- TASK-547
- Docs/superpowers/specs/2026-06-09-explainer-workspace-design.md
- Docs/superpowers/plans/2026-06-09-explainer-workspace-implementation-plan.md
modified_files:
- apps/tldw-frontend/pages/explainer.tsx
- apps/tldw-frontend/extension/routes/option-explainer.tsx
- apps/tldw-frontend/extension/routes/route-registry.tsx
- apps/packages/ui/src/routes/route-metadata.ts
- apps/packages/ui/src/services/tldw/openapi-guard.ts
- apps/packages/ui/src/services/tldw/TldwApiClient.ts
- apps/packages/ui/src/components/Option/Explainer/types.ts
- apps/packages/ui/src/components/Option/Explainer/explainerApi.ts
- apps/packages/ui/src/components/Option/Explainer/tree.ts
- apps/packages/ui/src/components/Option/Explainer/useExplainerQueries.ts
- apps/packages/ui/src/components/Option/Explainer/ExplainerWorkspace.tsx
- apps/packages/ui/src/components/Option/Explainer/ExplainerModeTabs.tsx
- apps/packages/ui/src/components/Option/Explainer/ExplainerGoalComposer.tsx
- apps/packages/ui/src/components/Option/Explainer/ExplainerSourcePicker.tsx
- apps/packages/ui/src/components/Option/Explainer/ExplainerTree.tsx
- apps/packages/ui/src/components/Option/Explainer/ExplainerDetailPanel.tsx
- apps/packages/ui/src/components/Option/Explainer/ExplainerChatbookExportButton.tsx
- apps/packages/ui/src/public/_locales/en/option.json
- apps/packages/ui/src/services/__tests__/tldw-api-client.explainer.test.ts
- apps/packages/ui/src/components/Option/Explainer/__tests__/ExplainerWorkspace.test.tsx
- apps/packages/ui/src/components/Option/Explainer/__tests__/explainer-tree.test.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implement Task 4 from Docs/superpowers/plans/2026-06-09-explainer-workspace-implementation-plan.md: /explainer route, typed client, explicit Goal/Sources tabs, source picker, tree/detail UI, polling, and Chatbook export button. Follow TDD for client/tree/workspace tests and the existing WebUI design patterns.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

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
