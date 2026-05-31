---
id: TASK-134
title: Remove unused WebUI lockfile-only dependency declarations for issue 1346
status: Done
assignee:
  - Codex
created_date: '2026-05-09 00:26'
updated_date: '2026-05-09 00:42'
labels:
  - webui
  - dependencies
  - cleanup
dependencies:
  - TASK-104
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1346'
documentation:
  - Docs/Design/WebUI_Dependency_Audit.md
  - Docs/superpowers/specs/2026-05-07-webui-dependency-trimming-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue GitHub issue #1346 after the axios replacement by investigating a narrow set of zero-direct-usage WebUI dependency declarations that the audit marked as lockfile investigation candidates. Remove only declarations that are confirmed unused by WebUI/shared UI/extension source, config, scripts, and package manifests; leave any package with active usage or uncertain ownership in place with rationale.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Candidate package declarations from the audit's investigate-lockfile group are rechecked against current origin/dev source, config, scripts, package manifests, and apps/bun.lock before any removal.
- [x] #2 Only confirmed unused direct declarations are removed from WebUI/shared UI/extension package manifests, and apps/bun.lock is regenerated consistently.
- [x] #3 Packages with active source usage, peer ownership requirements, or uncertain domain risk are retained and documented with rationale.
- [x] #4 Focused install/build/lint/test verification is run for the changed package scope, or blockers are documented with evidence.
- [x] #5 Bandit is skipped with rationale if the slice changes only TypeScript/package metadata/docs, or run on touched Python paths if any Python files are changed.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Recheck current origin/dev package declarations and exact package-specifier usage for the lockfile-investigation candidate set.
2. Remove only direct declarations with no source/config/script references and no peer ownership requirement.
3. Regenerate apps/bun.lock and verify which packages remain only transitively.
4. Run install, build, lint, changed-test, and diff hygiene checks.
5. Document retained-package rationale and open a PR against dev.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Removed unused direct package declarations for react-syntax-highlighter, @types/react-syntax-highlighter, react-toastify, rehype-mathjax, unist-util-visit, and zod from the WebUI, extension, and shared UI manifests where they had no exact source/config/script references. Regenerated apps/bun.lock with bun install. Retained @dnd-kit/abstract, @dnd-kit/dom, and @tiptap/pm because the lockfile shows active DnD package graph ownership and Tiptap peer/runtime ownership.

Verification completed: bun install --frozen-lockfile from apps; bun run compile from apps/extension; NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 bun run compile from apps/tldw-frontend; bun run lint from apps/tldw-frontend; bunx vitest run --changed=origin/dev from apps/tldw-frontend; git diff --check. Vitest changed-file probe exited 0 with no matching tests for manifest-only changes. Bandit skipped because no Python files changed.

PR #1385 review follow-up: Qodo reported two actionable issues. Fix needed: record measurable dependency/lockfile deltas, and update stale dependency-audit table rows so packages removed by TASK-134 are no longer listed as active declarations or pending investigate-lockfile candidates.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Final Summary

<!-- SECTION:FINAL:BEGIN -->
Removed unused direct WebUI/extension/shared UI declarations for react-syntax-highlighter, @types/react-syntax-highlighter, react-toastify, rehype-mathjax, unist-util-visit, and zod after exact source/config/script usage checks. Regenerated apps/bun.lock and documented the lockfile investigation results, including why @dnd-kit/abstract, @dnd-kit/dom, and @tiptap/pm remain declared. PR #1385 review follow-up addressed Qodo's findings by recording measurable deltas and updating stale audit-table rows to removed/TASK-134 complete. Impact: direct declaration entries across the three scanned manifests changed from 270 to 260 (-10); removed candidate declaration entries changed from 10 to 0 (-10) across 6 unique package names; apps/bun.lock changed from 536,939 bytes to 518,386 bytes (-18,553), from 4,641 lines to 4,473 lines (-168), and from 2,156 package records to 2,077 package records (-79). Verification passed for frozen install, extension compile, frontend compile with NEXT_PUBLIC_API_URL, frontend lint, changed-file Vitest probe, and git diff --check; Bandit was skipped because no Python files changed.
<!-- SECTION:FINAL:END -->
