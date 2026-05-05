---
id: TASK-45.6.1
title: Address PR 1315 review comments
status: Done
assignee: []
created_date: '2026-05-05 18:07'
updated_date: '2026-05-05 18:10'
labels:
  - design-system
  - frontend
  - review-fix
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1315'
parent_task_id: TASK-45.6
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix actionable review feedback on PR #1315 for the Chat/Playground status Badge migration. Scope: remove redundant srLabel props from icon badge wrappers, make StatusDot icons inherit Badge variant colors, add message-type fallback labels in PlaygroundUserMessage, and make ResearchRunStatusStack badge test IDs unique per run. Keep changes limited to PR review fixes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 StatusDot and SaveStatusIcon Badge wrappers no longer duplicate parent button aria labels with sr-only text.
- [x] #2 StatusDot icons inherit the Badge variant color for checking, demo, connected, config/error, and failed states.
- [x] #3 PlaygroundUserMessage message-type Badge renders a human-friendly fallback when a copilot translation key is missing.
- [x] #4 ResearchRunStatusStack status badge test IDs are unique per run while tests still assert Badge markers for multiple runs.
- [x] #5 Focused Vitest, targeted ESLint, and git diff checks are recorded; Bandit skipped as frontend-only if no Python is touched.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed PR #1315 review comments: removed redundant Badge srLabel props under aria-labeled buttons, changed StatusDot icons to text-current so they inherit Badge variant color, added human-readable message-type fallback labels, and made research-run status badge test IDs unique per run.

Verification: bunx vitest run src/components/Sidepanel/Chat/__tests__/StatusBadges.design-system.test.tsx src/components/Option/Playground/__tests__/PlaygroundStatusBadges.design-system.test.tsx src/components/Common/Playground/__tests__/PlaygroundUserMessage.design-system.test.tsx src/components/Option/Playground/__tests__/PlaygroundChat.research-status.integration.test.tsx src/components/Option/Playground/__tests__/research-run-status.test.ts --reporter=dot passed 39/39 tests.

Verification: tldw-frontend/node_modules/.bin/eslint --config tldw-frontend/eslint.config.mjs on the touched UI files exited 0. It emits the existing Next pages-directory notice when run from apps against packages/ui files, but no lint errors or warnings on touched files.

Verification: git diff --check exited 0. Bandit skipped because this is frontend-only TypeScript/React with no Python changes.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed all actionable PR #1315 review comments by removing redundant hidden labels from icon badge wrappers, aligning StatusDot icon color with Badge variants, adding fallback text for missing message-type translations, and giving research-run status badges unique per-run test IDs. Focused Vitest and targeted ESLint passed; Bandit is not applicable for frontend-only changes.
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
