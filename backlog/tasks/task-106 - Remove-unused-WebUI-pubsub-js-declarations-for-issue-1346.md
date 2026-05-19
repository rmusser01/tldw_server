---
id: TASK-106
title: Remove unused WebUI pubsub-js declarations for issue 1346
status: Done
assignee:
  - codex
created_date: '2026-05-07 04:46'
updated_date: '2026-05-07 04:51'
labels:
  - webui
  - dependencies
  - cleanup
dependencies:
  - TASK-104
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1346'
  - 'https://github.com/rmusser01/tldw_server/pull/1353'
documentation:
  - Docs/Design/WebUI_Dependency_Audit.md
  - Docs/superpowers/specs/2026-05-07-webui-dependency-trimming-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the first quick-cleanup slice from the merged WebUI dependency audit for issue #1346. Remove only direct pubsub-js and @types/pubsub-js declarations from the WebUI/shared UI/extension package manifests and update the Bun lockfile. Do not remove other package candidates or rewrite runtime code in this slice.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Direct pubsub-js and @types/pubsub-js declarations are removed from the audited package manifests only.
- [x] #2 apps/bun.lock is regenerated or updated consistently with the manifest changes.
- [x] #3 A focused search confirms no remaining import/config/package-script usage for pubsub-js in WebUI/shared UI/extension sources.
- [x] #4 Focused install, typecheck/build, and relevant WebUI/shared UI verification are run or any environment blockers are documented.
- [x] #5 Bandit is skipped with rationale because this slice changes only WebUI package metadata and Backlog task documentation.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect current package declarations and lockfile entries for pubsub-js and @types/pubsub-js. 2. Remove only those direct declarations from the relevant package manifests. 3. Run Bun install from apps/ to update apps/bun.lock. 4. Search source/config/scripts to confirm no usage remains. 5. Run focused WebUI/shared UI verification and git diff --check. 6. Update this task with verification, commit, push, and open a PR against dev.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Removed direct pubsub-js and @types/pubsub-js declarations from apps/tldw-frontend/package.json and apps/extension/package.json, then regenerated apps/bun.lock with bun install from apps/.

Focused search returned no pubsub-js, @types/pubsub-js, or PubSub matches in apps/tldw-frontend, apps/packages/ui, apps/extension, or apps/bun.lock after removal.

Verification: bun install --frozen-lockfile passed in apps/; bun run lint passed in apps/tldw-frontend with 127 existing warnings and 0 errors; bunx vitest run --changed=origin/dev exited 0 with no changed test files; package JSON parse check passed; git diff --check passed.

Extension compile check: bun run compile in apps/extension failed on unchanged baseline config, wxt.config.ts import of ./scripts/post-build-tasks.mjs lacks a declaration file under tsconfig.compile.json. This slice did not modify wxt.config.ts, tsconfig.compile.json, or the helper module, so the blocker is documented rather than fixed here.

Bandit skipped: touched source consists only of WebUI package metadata, apps/bun.lock, and Backlog task documentation; no Python code changed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed the unused direct pubsub-js and @types/pubsub-js declarations from the WebUI and extension package manifests and updated the Bun lockfile. Verification confirms no remaining PubSub usage in the audited WebUI/shared UI/extension scope; the only blocked check is the pre-existing extension TypeScript config issue for an undeclared .mjs helper import.
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
