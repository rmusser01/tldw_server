---
id: TASK-338
title: Document traceable artifact contract and ACP output mapping
status: Done
assignee: []
created_date: '2026-05-14 06:46'
updated_date: '2026-05-14 12:45'
labels:
  - acp
  - docs
  - artifacts
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1525'
  - 'https://github.com/rmusser01/tldw_server/issues/1538'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Define the traceable work-product artifact contract needed by issue #1525, then document how ACP run outputs map into that contract for issue #1538. This is a docs/design slice intended to unblock later storage/API, UI detail, and verification implementation work without pretending all artifact types are implemented.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Artifact contract docs define minimum schema, source-lineage, review-state, versioning, export, ownership, and migration semantics for generated work products.
- [x] #2 ACP docs define which run outputs can be promoted into generated workspace artifacts and which remain low-level ACP session artifacts.
- [x] #3 ACP review, retry, reject, accept, redaction, and retention behavior is aligned with existing reviewer-loop and support-safe view semantics.
- [x] #4 Follow-up implementation slices are split for storage/API, UI detail, and verification.
- [x] #5 GitHub issues #1525 and #1538 can be updated with concrete closeout evidence or explicit remaining blockers.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Address live PR #1687 review findings. Status: Complete.
2. Clarify producer_id versus workspace task_id in the artifact contract. Status: Complete.
3. Add workspace-history linkage to artifact versioning. Status: Complete.
4. Define CSV/XLSX lineage export convention. Status: Complete.
5. Enumerate ACP artifact-detail visibility categories and redaction behavior. Status: Complete.
6. Normalize docs to the canonical needs_revision state literal. Status: Complete.
7. Keep Backlog task provisional until PR review/merge, then verify and push. Status: Complete.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added Traceable_Work_Product_Artifact_Contract.md for #1525, mapped ACP session outputs and promotable deliverables for #1538, and connected ACP PRD/operator/readiness/workspace docs to the contract. Verification so far: git diff --check passed; targeted rg presence and stale-language absence guards passed. Bandit is skipped because this slice only touches Markdown docs and Backlog metadata.

Opened draft PR #1687 and posted closeout-evidence comments on issues #1525 and #1538. Both issues should remain open until PR #1687 is reviewed and merged.

Reopened for PR #1687 review-fix pass. Live unresolved review threads: Gemini producer_id/task_id distinction, Gemini CSV/XLSX lineage convention, Qodo workspace-history revision linkage, Qodo ACP artifact-detail visibility, Qodo needs_revision state literal consistency, CodeRabbit provisional Backlog status until PR merge.

Addressed review feedback locally: clarified ACP producer_id versus workspace task_id, linked version history to workspace history/audit events, specified CSV/XLSX lineage sidecar/hidden-sheet conventions, enumerated ACP artifact-detail visibility by category, normalized needs_revision literals, and kept task status provisional while PR #1687 remains in review.

Review-fix verification before commit: git diff --check passed; presence guard found producer_id ACP Task ID semantics, workspace-history linkage, sidecar JSON lineage convention, ACP artifact-detail visibility table, needs_revision literal, and provisional Backlog status; stale-language guard found no hyphenated review-state spelling, no premature Done front matter, and no completed final-summary wording.

Post-merge closeout: PR #1687 merged into dev at 2026-05-14T19:39:31Z with merge commit c63e0070c26140e1aa01b0ef9af896306125c0be. Issues #1525 and #1538 were closed with merge evidence. This task is now safe to mark Done.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed after PR #1687 merged.
- Added the traceable work-product artifact contract.
- Mapped ACP run outputs and promotable deliverables into the contract.
- Addressed review feedback on producer/task identity, workspace-history linkage, CSV/XLSX lineage, artifact-detail visibility, and needs_revision state spelling.
- Closed issues #1525 and #1538 with merge evidence.
- Identified follow-up implementation slices for storage/API, UI detail, export adapters, ACP promotion, and verification.
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
