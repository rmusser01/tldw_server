---
id: TASK-463.5
title: Add Research Workspace source status drilldown
status: Done
labels:
- research-workspace
- workspace
- source-status
- frontend
- phase-d
priority: high
parent_task_id: TASK-463
references:
- Docs/superpowers/specs/2026-05-23-research-workspace-hard-replacement-roadmap-design.md
modified_files:
- apps/packages/ui/src/types/workspace.ts
- apps/packages/ui/src/store/workspace.ts
- apps/packages/ui/src/store/workspace-slices/sources-slice.ts
- apps/packages/ui/src/store/__tests__/workspace.test.ts
- apps/packages/ui/src/components/Option/ResearchWorkspace/index.tsx
- apps/packages/ui/src/components/Option/ResearchWorkspace/SourcesPane/index.tsx
- apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.stage3.test.tsx
- apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/SourcesPane.stage2.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the next Phase D trust/transparency slice by giving each Research Workspace source a focused status drilldown that explains lifecycle/readiness, source-of-truth, retry/stale state, timestamps, and next action without adding /workspace-playground aliases.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 SourcesPane exposes a keyboard-accessible status details action for sources with processing, failed, stale, retryable, or diagnostic status fields.
- [x] #2 Status details show user-facing lifecycle/readiness summary, status code/message, source of truth, last refresh, retry eligibility, stale state, media/source identifiers, and practical next action copy.
- [x] #3 The drilldown remains compact, does not duplicate the preview/annotation workflow, and uses existing Research Workspace visual patterns.
- [x] #4 Focused Vitest coverage proves status details render and no /workspace-playground alias is introduced.
- [x] #5 Implementation is frontend-only unless verification finds a backend/API gap.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Use TDD. Add a focused SourcesPane test that fails because no status-details action/dialog exists. Implement minimal SourcesPane state/action/dialog using existing WorkspaceSource status fields and source-status formatting helpers. Run focused Vitest suite, route guard tests, CDP validation against live backend, and git diff hygiene. Bandit is not applicable unless Python backend files are touched.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented frontend-only source status drilldown for Research Workspace.

- Added WorkspaceSourceStatusDetails types and persisted revival for updatedAt.
- Preserved backend workspace status projection details through ResearchWorkspace status mapping and source store actions.
- Added compact keyboard-accessible SourcesPane status details action/modal for processing, failed, stale, retryable, diagnostic, and incomplete-readiness sources.
- Modal shows lifecycle, status reason/message, source of truth, last refresh, progress, retry eligibility, stale state, readiness, media/source identifiers, and next action copy.
- Kept route language on research-workspace and added negative coverage against workspace-playground labels/aliases.

Verification recorded 2026-05-26:
- Red test confirmed missing status details action before implementation.
- PASS: bunx vitest run src/components/Option/ResearchWorkspace/__tests__/SourcesPane.stage2.test.tsx src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.stage3.test.tsx src/store/__tests__/workspace.test.ts --maxWorkers=1 --no-file-parallelism (99 tests)
- PASS: bunx vitest run src/routes/__tests__/route-metadata.coverage.test.ts src/routes/__tests__/route-paths.viewport.test.ts src/components/Option/ResearchWorkspace/__tests__/WorkspaceCapabilityRemediation.test.tsx --maxWorkers=1 --no-file-parallelism (13 tests)
- PASS: git diff --check
- PASS: rg confirmed only negative guard/test assertions for workspace-playground under active ResearchWorkspace/route/tutorial/layout utility scope.
- PASS: live backend + WebUI + CDP check at /research-workspace opened the status details dialog for a seeded diagnostic source. Backend projection marked the seeded missing media as failed, dialog showed lifecycle/source-of-truth/retry/identifiers/next-action, and the page contained no workspace-playground text.
- SKIP: Bandit not applicable; touched code is frontend TypeScript/React only.

Known baseline issues not fixed in this task:
- tsc still exits 2 on unrelated baseline failures in CharacterListContent.design-system.test.tsx density typing and sidepanel-flashcards tuple/undefined assertions.
- Broad ResearchWorkspace AddSourceModal.stage3.performance.test.tsx still fails because ExistingTab mount bypasses the TTL media cache and calls listMedia twice.
- Broad ResearchWorkspace desktop-layout tests still show duplicate restore rail failures for Show sources/workspace-restore-sources/Show studio.
- Live validation console showed expected 404/request warnings for the intentionally seeded nonexistent media id.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Source rows now expose a compact status details action/modal showing lifecycle, readiness, progress, source of truth, retry/stale state, identifiers, and next action. Backend workspace status projection details are preserved into source state and persistence, with stale details cleared when status changes without new details. Focused tests, route/no-alias checks, git diff hygiene, and live backend + WebUI + CDP validation passed; unrelated baseline failures are documented.
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
