---
id: TASK-45.6
title: Migrate Chat and Playground product status chips to design-system primitives
status: Done
assignee: []
created_date: '2026-05-05 17:14'
updated_date: '2026-05-05 17:33'
labels:
  - design-system
  - frontend
  - playground
  - chat
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1315'
documentation:
  - Docs/Design/tldw_web_design_system_contract.md
  - Docs/Design/tldw_web_design_system_inventory.md
parent_task_id: TASK-45
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the next bounded Chat/Playground migration slice from the design-system inventory after the empty/recovery banner slices. Scope is limited to product status indicators and chips: Sidepanel/Chat/StatusDot, Sidepanel/Chat/SaveStatusIcon, Option/Playground/ResearchRunStatusStack, Option/Playground/VoiceChatIndicator, and Common/Playground/PlaygroundUserMessage. Do not migrate generic visual pills, composer-only controls, modal footers, or broad AntD mechanics in this slice.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Chat status indicators use canonical Badge or state primitives for user-facing ready/saving/error/status semantics while preserving accessible labels and layout.
- [x] #2 Playground research run and voice status indicators map domain statuses to canonical Badge or state variants without changing non-status controls or workflows.
- [x] #3 Playground user-message status chips use canonical design-system status primitives for product state while preserving message actions and metadata behavior.
- [x] #4 Focused tests cover migrated product status indicators and assert design-system markers, accessible labels, and status text/actions.
- [x] #5 Verification includes focused Chat/Playground Vitest coverage, lint/diff checks, and Bandit is skipped or documented as not applicable for frontend-only changes.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Migrated the bounded Chat/Playground product status slice to the shared Badge primitive: StatusDot, SaveStatusIcon, ResearchRunStatusStack, VoiceChatIndicator, PlaygroundUserMessage system/message-type chips, plus a Badge data-ds-component marker for testable design-system ownership.

Added focused design-system tests for Chat status badges, Playground research/voice status badges, and Playground user-message chips. Verified red first before implementation, then green after migration.

Verification: bunx vitest run src/components/Sidepanel/Chat/__tests__/StatusBadges.design-system.test.tsx src/components/Option/Playground/__tests__/PlaygroundStatusBadges.design-system.test.tsx src/components/Common/Playground/__tests__/PlaygroundUserMessage.design-system.test.tsx src/components/Option/Playground/__tests__/PlaygroundChat.research-status.integration.test.tsx src/components/Option/Playground/__tests__/research-run-status.test.ts --reporter=dot passed 37/37 tests.

Verification: tldw-frontend/node_modules/.bin/eslint --config tldw-frontend/eslint.config.mjs on the touched UI files exited 0. It emits the existing Next pages-directory notice when run from apps against packages/ui files, but no lint errors or warnings on the touched files.

Verification: git diff --check exited 0.

Bandit skipped: frontend-only TypeScript/React changes, no Python touched. Full package tsc is not a useful gate in this worktree because it currently fails on unrelated pre-existing package-wide test/type errors outside this slice.

Draft PR opened: https://github.com/rmusser01/tldw_server/pull/1315
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated the next Chat/Playground design-system inventory slice by replacing local status spans/icons with the shared Badge primitive for connection status, chat save status, linked research run status, voice chat status, and Playground user-message system/message-type chips. Added focused Vitest coverage asserting design-system markers, accessible labels, visible status text, and preserved actions. Bandit is not applicable because this slice touches only frontend TypeScript/React files.
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
