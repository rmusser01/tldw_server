---
id: TASK-418.10.1.1
title: Address PR 1953 route governance review comments
status: Done
labels:
- wp12
- review
- webui
priority: High
parent_task_id: TASK-418.10.1
references:
- https://github.com/rmusser01/tldw_server/pull/1953
modified_files:
- apps/packages/ui/src/routes/__tests__/route-registry-ast-helpers.ts
- apps/packages/ui/src/routes/__tests__/route-governance.metadata-coverage.test.ts
- apps/packages/ui/src/routes/__tests__/route-registry.visibility.test.ts
- apps/packages/ui/src/routes/__tests__/route-registry.sidepanel-availability.test.ts
- apps/packages/ui/src/routes/route-metadata.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve PR 1953 review feedback for the WP12 route governance slice: normalized smoke inventory duplicate detection, shared route-registry AST helpers, dynamic-route metadata coverage, robust test path resolution, and /chat/agent canonical path cleanup.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Governance duplicate checks use the same path normalization semantics as route metadata lookup.
- [x] #2 Route-registry AST parsing helpers are shared by the affected route tests instead of duplicated.
- [x] #3 Parameterized shared option routes are covered by route metadata governance.
- [x] #4 /chat/agent canonical path points directly to the main web Agents surface.
- [x] #5 Focused route governance tests and smoke metadata contract pass, or any unrelated baseline failures are documented.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Resolved PR 1953 review feedback by:

- Sharing route-registry AST parsing helpers across the governance, visibility, and sidepanel availability tests.
- Resolving route-registry fixtures relative to the test file instead of using working-directory candidate lists.
- Exporting and reusing `normalizeRoutePath` for smoke inventory duplicate detection.
- Covering dynamic shared option routes with metadata: `/sources/:sourceId`, `/share/:token`, `/knowledge/thread/:threadId`, `/knowledge/shared/:shareToken`, and `/presentation-studio/:projectId`.
- Canonicalizing `/chat/agent` directly to `/agents`.

Verification:

- `bunx vitest run ../packages/ui/src/routes/__tests__/route-governance.metadata-coverage.test.ts ../packages/ui/src/routes/__tests__/route-metadata.coverage.test.ts ../packages/ui/src/routes/__tests__/route-registry.visibility.test.ts ../packages/ui/src/routes/__tests__/route-registry.sidepanel-availability.test.ts`: 21 passed.
- `bunx playwright test e2e/smoke/route-contract-stage2.spec.ts --grep "Route metadata smoke inventory contract" --reporter=line`: 3 passed after escalated local server bind.
- `git diff --check`: passed.
- `bunx tsc --noEmit`: still fails on existing baseline issues outside touched route-governance files (MediaReadAlongPopover, Watchlists quick setup/run drawer, WorkspacePlayground, shortcut config, persona live control, admin llamacpp e2e typing).
- Bandit skipped: touched files are TypeScript tests/metadata and Backlog task records only.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed all current PR 1953 review comments for the WP12 route governance slice: duplicate route inventory checks now normalize paths, AST route extraction is shared, dynamic routes are governed by metadata, path resolution is test-file-relative, and /chat/agent now canonicalizes to /agents.
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
