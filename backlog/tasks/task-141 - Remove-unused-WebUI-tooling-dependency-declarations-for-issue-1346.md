---
id: TASK-141
title: Remove unused WebUI tooling dependency declarations for issue 1346
status: Done
assignee:
  - Codex
created_date: '2026-05-09 01:34'
updated_date: '2026-05-09 01:40'
labels:
  - webui
  - dependencies
  - cleanup
dependencies:
  - TASK-104
  - TASK-134
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1346'
documentation:
  - Docs/Design/WebUI_Dependency_Audit.md
  - Docs/superpowers/specs/2026-05-07-webui-dependency-trimming-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue GitHub issue #1346 by investigating the tooling-only lockfile candidates @eslint/eslintrc, eslint-config-next, eslint-config-prettier, and fake-indexeddb from the WebUI dependency audit. Remove only direct declarations that have no current source/config/script usage and are not required by the active ESLint/Vitest toolchain; document any retained package with rationale.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current origin/dev usage for @eslint/eslintrc, eslint-config-next, eslint-config-prettier, and fake-indexeddb is checked across WebUI, shared UI, extension source, config, scripts, tests, package manifests, and apps/bun.lock before removal.
- [x] #2 Only confirmed unused direct tooling declarations are removed from package manifests, and apps/bun.lock is regenerated consistently.
- [x] #3 The audit document records removed or retained decisions plus measurable dependency and lockfile deltas where feasible.
- [x] #4 Focused install/lint/test verification is run for the changed tooling scope, or blockers are documented with evidence.
- [x] #5 Bandit is skipped with rationale if no Python files are changed.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Recheck exact source, config, script, manifest, and lockfile references for the tooling candidate set.
2. Remove only direct WebUI dev declarations that have no direct usage and are not needed by the active flat ESLint/Vitest setup.
3. Regenerate apps/bun.lock and measure manifest/lockfile impact against origin/dev.
4. Update the dependency audit table and verification notes with removed/retained decisions.
5. Run focused install, lint, and changed-test verification, then open a PR against dev.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Exact usage search found @eslint/eslintrc, eslint-config-next, eslint-config-prettier, and fake-indexeddb only in apps/tldw-frontend/package.json and apps/bun.lock. The active WebUI flat ESLint config imports @eslint/js, @next/eslint-plugin-next, eslint-plugin-react, eslint-plugin-react-hooks, and TypeScript ESLint directly; package scripts call eslint . and do not invoke eslint-config-prettier. No tests or setup files import fake-indexeddb.

Removed the four direct WebUI devDeclarations and regenerated apps/bun.lock with bun install. eslint-config-next, eslint-config-prettier, and fake-indexeddb dropped out completely. @eslint/eslintrc remains only as an ESLint transitive dependency. Impact against origin/dev: direct declaration entries changed from 260 to 256 (-4); candidate declaration entries changed from 4 to 0 (-4); apps/bun.lock changed from 518,386 bytes to 501,563 bytes (-16,823), from 4,473 lines to 4,347 lines (-126), and from 2,077 package records to 2,016 package records (-61). Bandit skipped because no Python files were modified.

Verification completed in /Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/webui-tooling-deps-cleanup-1346: git diff --check (exit 0), bun install --frozen-lockfile from apps (exit 0), bun run lint from apps/tldw-frontend (exit 0 with 0 errors and 131 warnings), bunx vitest run --changed=origin/dev from apps/tldw-frontend (exit 0 with no changed tests), and NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 bun run compile from apps/tldw-frontend (exit 0).
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed four unused direct WebUI tooling dependency declarations: @eslint/eslintrc, eslint-config-next, eslint-config-prettier, and fake-indexeddb. Regenerated apps/bun.lock and updated the dependency audit with removed/retained decisions plus measured deltas. eslint-config-next, eslint-config-prettier, and fake-indexeddb dropped out of the lockfile completely; @eslint/eslintrc remains only as an ESLint transitive dependency. Bandit was skipped because no Python files changed.
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
