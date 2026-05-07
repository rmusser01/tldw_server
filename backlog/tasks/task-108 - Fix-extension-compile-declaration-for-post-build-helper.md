---
id: TASK-108
title: Fix extension compile declaration for post-build helper
status: Done
assignee:
  - '@codex'
created_date: '2026-05-07 04:57'
updated_date: '2026-05-07 04:58'
labels:
  - extension
  - typescript
  - build
dependencies:
  - TASK-106
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1357'
  - 'https://github.com/rmusser01/tldw_server/issues/1346'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix the extension TypeScript compile failure that blocks the WebUI dependency cleanup PR. The failure is TS7016 in apps/extension/wxt.config.ts because the config imports ./scripts/post-build-tasks.mjs and the compile tsconfig has strict checking without a declaration for that local ESM helper.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 bun run compile passes from apps/extension without suppressing strict TypeScript globally.
- [x] #2 The post-build helper import remains typed enough for wxt.config.ts to call getWxtTargetName and runPostBuildTasks safely.
- [x] #3 The fix is scoped to the extension compile blocker and does not alter package-cleanup behavior.
- [x] #4 Focused verification and any Bandit skip rationale are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the TS7016 compile failure with bun run compile from apps/extension. 2. Inspect the post-build helper exports and TypeScript declaration patterns in the extension/shared UI code. 3. Add the smallest declaration or typing support for the local .mjs helper import. 4. Re-run extension compile and focused package-cleanup verification. 5. Update the task record, amend/update the PR branch, push, and verify PR state.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: apps/extension/tsconfig.compile.json includes wxt.config.ts under strict checking, and wxt.config.ts imports the local ESM helper ./scripts/post-build-tasks.mjs without a sibling declaration file. TypeScript therefore treated the helper as implicit any and failed with TS7016.

Fix: added apps/extension/scripts/post-build-tasks.d.mts with typed exports for PostBuildTask, RunPostBuildTasksOptions, getPostBuildTasks, runPostBuildTasks, and getWxtTargetName. This keeps strict TypeScript enabled and avoids broad .mjs-any declarations.

Verification: bun run compile passed from apps/extension; bunx vitest run tests/unit/post-build-tasks.test.ts passed from apps/extension with 1 file and 4 tests; bun install --frozen-lockfile passed from apps/ with no changes; pubsub usage search still returned no matches; git diff --check passed.

Bandit skipped: this follow-up changes TypeScript declaration metadata and Backlog task documentation only; no Python code changed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the extension TypeScript compile blocker by adding a typed .d.mts declaration beside the local post-build .mjs helper. The compile check now passes without weakening strictness or changing runtime build behavior.
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
