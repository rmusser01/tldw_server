---
id: TASK-12093
title: Design Paperless-inspired document lifecycle improvements
status: Done
labels:
- prd
- webui
- extension
- documents
- storage
priority: medium
documentation:
- Docs/Product/Paperless_Inspired_Document_Lifecycle_PRD.md
modified_files:
- Docs/Product/Paperless_Inspired_Document_Lifecycle_PRD.md
- backlog/tasks/task-12093 - Design-Paperless-inspired-document-lifecycle-improvements.md
- backlog/tasks/task-12093.1 - Implement-persisted-source-review-lifecycle.md
- backlog/tasks/task-12093.2 - Implement-saved-source-filter-presets-and-views.md
- backlog/tasks/task-12093.3 - Implement-duplicate-detection-and-attach-existing-recovery.md
- backlog/tasks/task-12093.4 - Implement-Document-Workspace-provenance-and-storage-metadata-panel.md
- backlog/tasks/task-12093.5 - Implement-unified-ingest-entrypoints-and-storage-policy-visibility.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the umbrella PRD for Paperless-inspired document lifecycle improvements across Quick Ingest, Research Workspace sources, Document Workspace, duplicate recovery, and storage policy visibility. Scope is product planning only; implementation is split into child tasks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Umbrella PRD exists and defines goals, non-goals, current repo anchors, user flows, risks, and phased child-task boundaries.
- [x] #2 PRD explicitly avoids cloning Paperless-ngx wholesale and excludes barcode/ASN, public share links, arbitrary storage templates, and full workflow builder scope.
- [x] #3 Five child Backlog tasks are created for persisted review lifecycle, saved source views, duplicate recovery, metadata panel, and unified ingest/storage policy visibility.
- [x] #4 Each child task is independently reviewable and links back to the umbrella PRD.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Create one umbrella PRD under Docs/Product, then create five child Backlog implementation tasks linked to the PRD. Run spec review before closeout and record any fixes or residual risks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created the umbrella PRD and five child implementation Backlog tasks for Paperless-inspired document lifecycle improvements. Addressed spec-review findings around duplicate privacy, workspace-scoped review ownership, server-backed saved views, storage-policy specificity, terminology, product outcomes, readable workspace membership privacy, user flows, and risks/mitigations. Follow-up review fixes added explicit review-state fields and filters separate from processing status, `unset`/Needs review semantics, review timestamp reset behavior, the first canonical extension path for unified ingest labels, and readiness for implementation planning. Verification: docs/task-only diff review and git diff --check; Bandit not applicable because no Python/code files changed.
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
