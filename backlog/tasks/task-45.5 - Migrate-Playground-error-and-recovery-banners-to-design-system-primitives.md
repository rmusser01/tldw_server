---
id: TASK-45.5
title: Migrate Playground error and recovery banners to design-system primitives
status: Done
assignee:
  - '@Codex'
created_date: '2026-05-05 05:21'
labels:
  - design-system
  - frontend
  - playground
dependencies: []
documentation:
  - Docs/Design/tldw_web_design_system_contract.md
  - Docs/Design/tldw_web_design_system_inventory.md
parent_task_id: TASK-45
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the next bounded Chat/Playground migration slice from the design-system inventory after TASK-45.4. Scope is limited to user-facing Playground error/recovery banners and notices: PlaygroundChatErrorBanner, PlaygroundComposerNotices recovery notices, and DocumentGeneratorDrawer availability/conversation recovery alerts. Do not migrate modal footers, product status chips, or broad AntD mechanics in this slice.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PlaygroundChatErrorBanner uses the canonical shared Alert or state primitive for user-facing error recovery while preserving diagnostics and dismiss actions.
- [x] #2 PlaygroundComposerNotices disconnected and degraded recovery notices use canonical design-system state language or primitives without changing composer controls or informational mode notices outside the scope.
- [x] #3 DocumentGeneratorDrawer availability and missing-conversation recovery alerts use canonical shared Alert or state primitive while preserving the drawer workflow and AntD form/modal mechanics.
- [x] #4 Focused tests cover migrated error/recovery states and assert accessible labels, state markers, and actions.
- [x] #5 Verification includes focused Playground Vitest coverage, formatting/lint/diff checks, and Bandit is skipped or documented as not applicable for frontend-only changes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing tests for design-system markers and accessible recovery actions on the target Playground banner surfaces.
2. Migrate only user-facing error/recovery banners to shared Alert or RecoveryCallout primitives.
3. Preserve composer controls, AntD form/drawer/modal mechanics, and informational notices that are not recovery states.
4. Run focused Vitest, lint/format/diff checks, update this task, and commit the slice.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Started after PR #1290 merged into dev at merge commit b0db36f1074e65c2db75a548c2fbabebc5ed66b3. Work is isolated in .worktrees/tldw-playground-banner-design-system on branch codex/tldw-playground-banner-design-system from origin/dev.

Verification:
- Red tests were added first for PlaygroundChatErrorBanner, PlaygroundComposerNotices disconnected/degraded notices, and DocumentGeneratorDrawer recovery alerts; the first run failed on missing shared Alert markers.
- `bunx vitest run ../packages/ui/src/components/Option/Playground/__tests__/PlaygroundChatErrorBanner.test.tsx ../packages/ui/src/components/Option/Playground/__tests__/PlaygroundComposerNotices.first-run.test.tsx ../packages/ui/src/components/Common/Playground/__tests__/DocumentGeneratorDrawer.design-system.test.tsx --reporter=dot` passed with 3 files and 11 tests.
- `bunx vitest run ../packages/ui/src/components/Option/Playground/__tests__/PlaygroundEmpty.test.tsx ../packages/ui/src/components/Option/Playground/__tests__/PlaygroundEmpty.disconnected.test.tsx ../packages/ui/src/components/Option/Playground/__tests__/PlaygroundChatErrorBanner.test.tsx ../packages/ui/src/components/Option/Playground/__tests__/PlaygroundComposerNotices.first-run.test.tsx ../packages/ui/src/components/Common/Playground/__tests__/DocumentGeneratorDrawer.design-system.test.tsx ../packages/ui/src/components/Sidepanel/Chat/__tests__/ConnectionBanner.test.tsx ../packages/ui/src/components/Sidepanel/Chat/__tests__/empty.test.tsx --reporter=dot` passed with 7 files and 25 tests.
- `apps/tldw-frontend/node_modules/.bin/eslint -c apps/tldw-frontend/eslint.config.mjs <touched files>` exited 0 with only the existing Next pages-directory notice.
- `bunx prettier --check <touched files>` was not used as a completion gate because this package does not expose a repo-matching Prettier config for the no-semicolon style; the accidental default-format churn was reversed.
- `bunx tsc --noEmit --pretty false -p tsconfig.json` still exits 2 on the broader frontend baseline; filtering the compiler log for the touched files produced no matches.
- `git diff --check` passed.
- Bandit skipped: this is a frontend-only TypeScript/React migration with no Python touched scope.
- PR review pass: CodeRabbit and Qodo both flagged the same accessibility issue in DocumentGeneratorDrawer. Added `role="status"` and `aria-live="polite"` to the non-urgent capability and missing-conversation Alert instances, plus regression assertions for both attributes.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated the bounded Playground recovery-banner slice to the shared design-system Alert primitive. Chat error recovery now uses the shared Alert while preserving diagnostics and dismiss actions; disconnected/degraded composer recovery notices use shared Alert markers without touching unrelated composer notices; DocumentGeneratorDrawer availability and missing-conversation recovery alerts now use the shared Alert while preserving the AntD drawer/form/modal workflow.
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
