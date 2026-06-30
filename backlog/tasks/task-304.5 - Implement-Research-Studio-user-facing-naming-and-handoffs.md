---
id: TASK-304.5
title: Implement Research Studio user-facing naming and handoffs
status: Done
assignee:
  - Codex
created_date: '2026-05-12 18:23'
updated_date: '2026-05-12 18:39'
labels:
  - implementation
  - research-studio
  - webui
  - extension
  - docs
dependencies:
  - TASK-304.4
documentation:
  - >-
    Docs/superpowers/plans/2026-05-12-research-studio-ux-remediation-implementation-plan.md
parent_task_id: TASK-304
priority: medium
---

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Navigation, helper copy, tutorials, docs, and handoff surfaces use Research Studio as the user-facing name
- [x] #2 User-facing links target /research-studio while legacy aliases remain available
- [x] #3 Knowledge QA, Shared With Me, Quick Chat helper, tutorial, extension, and E2E route inventories are checked before edits
- [x] #4 Internal IDs, storage, telemetry, tutorial IDs, and code names remain stable unless visibly user-facing
- [x] #5 Focused tests cover updated labels, handoff links, tutorials, and route inventory expectations
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inventory current Workspace Playground, workspace-studio, and /workspace-playground references under the scoped WebUI, extension, tests, and source docs.
2. Add or update failing tests for user-facing labels and canonical handoff paths.
3. Replace visible labels and links with Research Studio and /research-studio, preserving internal compatibility names.
4. Run focused UI/frontend tests and scoped search checks to classify remaining references.
5. Update this task with verification and final summary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the Research Studio naming and handoff sweep across scoped WebUI, extension, tests, and source docs. Header shortcuts, Quick Chat workflow guides, docs RAG page context, Knowledge QA handoff, Knowledge empty-state CTA, Shared With Me open action, tutorials, E2E inventories, and route page objects now use /research-studio and Research Studio where user-facing.

Preserved internal compatibility names: WorkspacePlayground component/module names, workspace-playground telemetry/prefill/storage imports, tutorial ID workspace-playground-basics, DOM ids such as workspace-studio-panel, and legacy /workspace-playground plus /workspace-studio aliases.

TDD notes: tests were first updated to expect canonical /research-studio routing and Research Studio labels. Red failures confirmed the old Quick Chat canonicalization, docs RAG route label, tutorial route matching, and Knowledge QA navigation still used /workspace-playground. During implementation, the active Shared With Me route was found to use components/Option/SharedWithMe.tsx rather than the sibling folder component; this exposed that the active UI lacked an Open action, so the canonical Research Studio Open link was added there with an accessible label.

Verification run:
- bunx vitest run src/components/Common/QuickChatHelper/__tests__/workflow-guides.test.ts src/components/Common/QuickChatHelper/__tests__/docs-rag-profile.test.ts src/components/Common/QuickChatHelper/__tests__/rag-response.test.ts src/components/Option/KnowledgeQA/__tests__/AnswerPanel.workspace-handoff.test.tsx src/components/Option/__tests__/SharedWithMe.research-studio-route.test.tsx src/components/Option/Settings/__tests__/ChatSettings.test.tsx src/tutorials/__tests__/registry.test.ts src/components/Layouts/__tests__/header-shortcut-descriptions.test.ts src/components/Layouts/__tests__/HeaderShortcuts.test.tsx src/components/Layouts/__tests__/persona-shortcut-defaults.test.ts -> 10 files passed, 91 tests passed. Known existing stderr warnings: tldw server not configured in ChatSettings tests and antd Space direction deprecation in SharedWithMe.
- bun run test:run __tests__/extension/route-registry.workspace-playground.test.ts __tests__/extension/route-registry.research-studio-alias.test.ts __tests__/navigation/research-studio-route-files.test.ts -> 3 files passed, 8 tests passed.
- rg -n "Workspace Playground|workspace-studio|/workspace-playground" apps/packages/ui/src apps/tldw-frontend Docs/User_Guides Docs/Code_Documentation Docs/Operations --glob !Docs/site/** -> remaining hits are classified as compatibility aliases, internal module/file/telemetry/storage identifiers, DOM ids, tests intentionally exercising legacy aliases, or untranslated locale baselines not regenerated in this slice.
- git diff --check -> clean.

Bandit was not run because this slice touched frontend TypeScript, frontend tests, E2E specs, source markdown docs, and Backlog metadata only; no Python/backend code changed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated user-facing Research Studio naming and canonical /research-studio handoffs while preserving internal WorkspacePlayground compatibility identifiers. Added canonical Shared With Me open behavior, canonicalized Quick Chat and tutorial route handling, updated Knowledge QA handoffs and docs, and refreshed focused tests plus route inventories.
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
