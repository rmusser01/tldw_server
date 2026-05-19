---
id: TASK-115
title: Remove unused WebUI hook form declarations for issue 1346
status: Done
assignee:
  - '@codex'
created_date: '2026-05-07 15:00'
updated_date: '2026-05-07 15:03'
labels:
  - webui
  - dependencies
  - cleanup
dependencies:
  - TASK-104
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1346'
  - 'https://github.com/rmusser01/tldw_server/pull/1359'
documentation:
  - Docs/Design/WebUI_Dependency_Audit.md
  - Docs/superpowers/specs/2026-05-07-webui-dependency-trimming-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the next quick-cleanup slice from the WebUI dependency audit for issue #1346. Remove only unused direct @hookform/resolvers and react-hook-form declarations from the WebUI package manifest and update the Bun lockfile. Do not remove other package candidates or rewrite runtime code in this slice.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Direct @hookform/resolvers and react-hook-form declarations are removed from the WebUI package manifest only when no import/config/package-script usage is present.
- [x] #2 apps/bun.lock is regenerated or updated consistently with the manifest changes.
- [x] #3 A focused search confirms no remaining direct import/config/package-script usage for @hookform/resolvers or react-hook-form in WebUI/shared UI/extension sources.
- [x] #4 Focused install, compile/build, and relevant WebUI verification are run or any environment blockers are documented.
- [x] #5 Bandit is skipped with rationale if this slice changes only WebUI package metadata and Backlog task documentation.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm current declarations and source/config usage for @hookform/resolvers and react-hook-form in apps/tldw-frontend, apps/packages/ui, apps/extension, and apps/bun.lock. 2. Remove only those direct declarations from the WebUI package manifest. 3. Run bun install from apps/ to update apps/bun.lock. 4. Re-run focused usage searches. 5. Run focused install/compile/lint/test checks and git diff --check. 6. Update this task with verification, commit, push, and open a PR against dev.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Removed direct @hookform/resolvers and react-hook-form declarations from apps/tldw-frontend/package.json, then regenerated apps/bun.lock with bun install from apps/.

Focused pre-removal import/config search found no direct package usage for @hookform/resolvers or react-hook-form in apps/tldw-frontend, apps/packages/ui, or apps/extension source/config/script files. Post-removal manifest and lockfile search found no remaining direct declarations or package records for either package.

Verification: bun install --frozen-lockfile passed in apps/ with 1846 installs across 1953 packages; bun run compile passed in apps/extension; NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 bun run compile passed in apps/tldw-frontend and token-sync passed; bun run lint passed in apps/tldw-frontend with 127 existing warnings and 0 errors; bunx vitest run --changed=origin/dev exited 0 with no changed test files; package JSON parse check passed; git diff --check passed.

Bandit skipped: this slice changes only WebUI package metadata, apps/bun.lock, and Backlog task documentation; no Python code changed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed unused direct @hookform/resolvers and react-hook-form declarations from the WebUI package manifest and updated the Bun lockfile. Focused source searches and WebUI/extension compile-oriented verification passed.
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
