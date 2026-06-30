---
id: TASK-117
title: Replace WebUI clsx helper for issue 1346
status: Done
assignee:
  - '@codex'
created_date: '2026-05-07 19:34'
updated_date: '2026-05-07 20:16'
labels:
  - webui
  - dependencies
  - cleanup
dependencies:
  - TASK-104
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1346'
  - 'https://github.com/rmusser01/tldw_server/pull/1365'
documentation:
  - Docs/Design/WebUI_Dependency_Audit.md
  - Docs/superpowers/specs/2026-05-07-webui-dependency-trimming-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the clsx cleanup slice from the WebUI dependency audit for issue #1346. Replace the tiny local WebUI class-name helper usage with a compatibility-safe local implementation, remove the direct clsx declaration, and update the Bun lockfile. Do not change unrelated styling helpers or Tailwind merge behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The direct clsx dependency is removed from apps/tldw-frontend/package.json and apps/bun.lock is updated consistently.
- [x] #2 Focused tests cover the local class-name helper behavior and fail before the replacement.
- [x] #3 Focused install, lint, compile/build, and relevant WebUI verification are run or blockers documented.
- [x] #4 Bandit is skipped with rationale if this slice changes only WebUI TypeScript/package metadata and Backlog documentation.
- [x] #5 The WebUI cn helper preserves the input shapes currently delegated to clsx, including strings, nested arrays, object maps, falsey values, and numbers while keeping unsupported standalone bigint and boolean values ignored.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect current clsx usage and package declarations. 2. Add focused tests for the WebUI cn helper compatibility and dependency guard. 3. Replace the clsx import with a local class-name flattener while preserving twMerge behavior. 4. Remove the direct clsx declaration and regenerate apps/bun.lock. 5. Run focused tests and WebUI verification, update task, commit, push, and open a draft PR against dev.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
RED verification: bunx vitest run __tests__/utils-cn.test.ts initially failed before implementation. The dependency guard failed on package.json clsx. The first draft also revealed that current clsx ignores standalone bigint values, so AC/test expectations were corrected to preserve observed behavior.

Changed-test sweep also surfaced a stale researchers-page CTA assertion unrelated to class joining. Current page renders Download Free / Read the Docs, so the test was updated to match the existing page contract before continuing verification.

GREEN verification: bunx vitest run __tests__/utils-cn.test.ts passed after the local class helper replacement. NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 bunx vitest run --changed=origin/dev passed 21 files / 82 tests after the stale researchers CTA assertion was corrected. bun install --frozen-lockfile passed with no lockfile changes after regeneration. bun run lint passed with 0 errors and the existing 127 warning backlog. NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 bun run compile passed and token sync succeeded. bunx tsc --noEmit -p tsconfig.json --pretty false passed. git diff --check passed. Bandit skipped because this slice touches only WebUI TypeScript/package metadata and Backlog documentation, with no Python files.

Post-rebase verification on latest origin/dev surfaced a research-run-console test timing issue: the plan checkpoint editor is populated from selectedSnapshot checkpoint state in an effect, so the test now waits for the Focus areas field before editing it.

Post-rebase final verification: NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 bunx vitest run --changed=origin/dev passed 23 files / 89 tests. bun run lint passed with 0 errors and the existing 127 warning backlog. bunx tsc --noEmit -p tsconfig.json --pretty false passed after adapting ControlRow's translator prop for McpToolSelector. NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 bun run compile passed after rebase and token sync succeeded. git diff --check passed.

PR #1368 review sweep found Gemini feedback to remove lockfile parsing from the cn helper test because it was brittle against package-manager formatting. Keeping package.json and source import guards instead.

PR #1368 review fix removed bun.lock parsing from the cn helper dependency guard while retaining package.json and utils.ts source guards. Verification after the review fix: bunx vitest run __tests__/utils-cn.test.ts passed 1 file / 3 tests; NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 bunx vitest run --changed=origin/dev passed 23 files / 89 tests; bun run lint passed with 0 errors and the existing 127-warning backlog; bunx tsc --noEmit -p tsconfig.json --pretty false passed; NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 bun run compile passed with token sync OK; git diff --check passed. Bandit remains skipped because the review fix touches only WebUI TypeScript tests and Backlog documentation, with no Python files.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Replaced the WebUI cn helper's direct clsx dependency with a small local class-value flattener that preserves existing supported input behavior and keeps tailwind-merge conflict handling. Removed clsx from apps/tldw-frontend/package.json and the tldw-frontend importer section of apps/bun.lock, added focused guard/behavior tests, corrected stale/timing-sensitive tests surfaced by changed-test verification, and adapted the sidepanel MCP tool selector translator prop to satisfy the current TypeScript contract on latest dev.
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
