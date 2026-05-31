---
id: TASK-45.44.10
title: 'Migrate design-system product state: Document and Workspace surfaces'
status: Done
assignee: []
created_date: '2026-05-14 03:20'
updated_date: '2026-05-31 18:14'
labels:
  - design-system
  - webui
  - extension
  - product-state
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1667'
  - >-
    Docs/superpowers/specs/2026-05-14-design-system-remaining-work-tracker-design.md
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
parent_task_id: TASK-45.44
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Mirror the linked GitHub product-area migration issue. Closure requires zero current product-state baseline exceptions for the owned path map area and the verification gates recorded in the GitHub issue.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The linked GitHub issue owns current count and public status.
- [x] #2 Implementation PR tasks are created under this child when the area is too broad for one PR.
- [x] #3 Backlog notes record PR links and before/after count evidence.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Created and completed TASK-45.44.10.1 for the first narrow Document/Workspace slice: DocumentPickerModal Alert migration. PR: https://github.com/rmusser01/tldw_server/pull/1950. Before/after product-state verifier evidence in that child task reduced total baseline exceptions from 303 to 300 and Document/Workspace exceptions from 12 to 9.
- Created and completed TASK-45.44.10.2 for the next narrow Document/Workspace slice: DocumentViewer PDF/EPUB Alert migration. PR: https://github.com/rmusser01/tldw_server/pull/1952. Verifier evidence after the slice reduces total baseline exceptions from 300 to 296 and Document/Workspace exceptions from 9 to 5.
- Created and completed TASK-45.44.10.3 for the next narrow Document/Workspace slice: DocumentWorkspacePage loading/health Alert migration. PR: https://github.com/rmusser01/tldw_server/pull/1955. Verifier evidence after the slice reduces total baseline exceptions from 296 to 294 and Document/Workspace exceptions from 5 to 3.
- Created and completed TASK-45.44.10.4 to fix the merged PdfDocument design-system Alert translation binding discovered by the TypeScript check. PR: https://github.com/rmusser01/tldw_server/pull/1955.
- Created and completed TASK-45.44.10.5 for the next narrow Document/Workspace slice: ReferencesTab server-unavailable/error EmptyState migration. PR: https://github.com/rmusser01/tldw_server/pull/1959. Verifier evidence after the slice reduces total baseline exceptions from 294 to 292 and Document/Workspace exceptions from 3 to 1.
- Created and completed TASK-45.44.10.6 for the final current Document/Workspace slice: DocumentWorkspaceErrorBoundary Result migration. PR: https://github.com/rmusser01/tldw_server/pull/1961. Verifier evidence after the slice reduces total baseline exceptions from 292 to 291 and removes the Document/Workspace product-area bucket.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Closed after confirming the current design-system product-state baseline no longer contains Document/Workspace-owned entries. `bun run verify:design-system-state` passes from `apps/packages/ui` and reports 82 allowed legacy exceptions in other product areas. A targeted baseline parse reports `{ "total": 82, "documentWorkspaceHits": 0 }` for DocumentWorkspace, Document/Workspace, ResearchWorkspace, Research Workspace, and Workspace labels.

Updated the linked public GitHub issue #1667 with the verified 2026-05-31 status, current zero owned-exception count, and implementation PR links (#1950, #1952, #1955, #1959, #1961). This closeout changes Backlog metadata only, so Bandit is not applicable. Known remaining work is outside this tracker: the shared-product-state baseline still contains 82 allowed exceptions owned by other queues.
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
