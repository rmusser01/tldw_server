---
id: TASK-45.15
title: Adapt WorkspaceChatEmpty to shared EmptyState
status: Done
assignee: []
created_date: '2026-05-07 02:10'
updated_date: '2026-05-07 15:01'
labels:
  - design-system
  - ui
  - workspace-playground
dependencies: []
references:
  - >-
    apps/packages/ui/src/components/Option/WorkspacePlayground/ChatPane/index.tsx
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
  - 'https://github.com/rmusser01/tldw_server/pull/1351'
documentation:
  - Docs/Design/tldw_web_design_system_contract.md
parent_task_id: TASK-45
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate the Workspace Playground chat empty-state wrapper onto the canonical EmptyState primitive so the design-system product-state guard no longer needs the WorkspaceChatEmpty local-empty-state baseline exception. Preserve the existing chat empty-state copy, source-aware examples, prompt seeding, add-source CTA, template actions, and Knowledge QA link.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 WorkspaceChatEmpty renders the canonical EmptyState primitive directly and preserves existing empty chat UX behavior.
- [x] #2 The WorkspaceChatEmpty local-empty-state baseline entry is removed without introducing new unbaselined product-state guard findings.
- [x] #3 Focused WorkspacePlayground ChatPane tests and the design-system product-state guard verification pass.
- [x] #4 The task records verification results and any frontend-only Bandit skip rationale.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Use the product-state guard baseline removal as the red test for the remaining WorkspaceChatEmpty local-empty-state debt. 2. Migrate WorkspaceChatEmpty from the compatibility FeatureEmptyState adapter to the canonical EmptyState primitive while preserving visible behavior and actions. 3. Remove the matching baseline entry. 4. Run focused ChatPane tests, guard verification, diff checks, and update the task with results.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
TDD red checks: removing the WorkspaceChatEmpty baseline entry made bun run verify:design-system-state fail with the expected blocked local-empty-state finding for src/components/Option/WorkspacePlayground/ChatPane/index.tsx. Adding the focused runtime assertion initially failed because the canonical EmptyState marker was absent. Test harnesses in ChatPane stage 2, stage 3, and stage 5 were wrapped in MemoryRouter because direct EmptyState renders the existing Knowledge QA Link path instead of the previous mocked FeatureEmptyState adapter path.

Implementation: WorkspaceChatEmpty now imports and renders the canonical EmptyState directly with size lg and card variant while preserving the existing title, description, source-aware prompt chips, Knowledge QA link, source guidance, add-source CTA, and template actions. Removed the matching local-empty-state baseline entry.

Verification: bunx vitest run src/components/Option/WorkspacePlayground/__tests__/ChatPane.stage1.test.tsx src/components/Option/WorkspacePlayground/__tests__/ChatPane.stage2.test.tsx src/components/Option/WorkspacePlayground/__tests__/ChatPane.stage3.test.tsx src/components/Option/WorkspacePlayground/__tests__/ChatPane.stage4.lorebook-activity.test.tsx src/components/Option/WorkspacePlayground/__tests__/ChatPane.stage5.folder-context.test.tsx src/design-system/__tests__/product-state-guard.test.ts --maxWorkers=1 --reporter=dot passed with 6 files and 79 tests. bun run verify:design-system-state passed with 520 baseline exceptions and no local-empty-state bucket. git diff --check passed. Bandit skipped because touched code is frontend TypeScript/JSON and a Backlog task file, not Python.

Broader check on PR branch: bunx vitest run src/components/Option/WorkspacePlayground --maxWorkers=1 --reporter=dot was attempted and failed in WorkspacePlayground suites outside this slice: AddSourceModal tab ordering, StudioPane stage 3 output/audio expectations, and WorkspacePlayground tests whose workspace-store mocks lack createWorkspaceStorage. The touched ChatPane and guard tests passed after the scoped fix.

Qodo review follow-up: verified the same broader WorkspacePlayground folder command against origin/dev at afe6990ee before applying PR #1351 changes. The base branch also failed that command, with 8 failed files, 23 failed tests, and 278 passed tests. Base failures included existing ChatPane Link router-context errors that this PR's MemoryRouter test harness update removes, plus WorkspacePlayground failures outside the touched ChatPane slice. This shows the broad folder command is already red on dev and the remaining PR-branch broad-run failures are not introduced by the WorkspaceChatEmpty design-system migration.

Review-fix verification: focused ChatPane plus product-state guard Vitest command passed with 6 files and 79 tests after the Qodo evidence update. bun run verify:design-system-state passed with 520 baseline exceptions. git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Adapted WorkspaceChatEmpty to render the canonical EmptyState primitive directly while preserving its existing research-start copy, source-aware prompt chips, Knowledge QA link, source guidance, add-source CTA, and template actions. Removed the WorkspaceChatEmpty local-empty-state baseline exception so the product-state guard now has no local-empty-state bucket and 520 total baseline exceptions. Updated ChatPane stage 2, stage 3, and stage 5 tests to render with MemoryRouter now that the real empty shell path includes the existing Knowledge QA Link. Opened PR #1351 against dev. Verification passed for the full ChatPane focused suite plus the product-state guard test, the design-system guard CLI, and git diff --check. The broader WorkspacePlayground folder run remains red on current origin/dev before this PR, and the PR task now records that base evidence plus the remaining unrelated PR-branch broad-run failures outside the touched slice.
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
