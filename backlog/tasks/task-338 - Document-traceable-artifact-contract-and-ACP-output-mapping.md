---
id: TASK-338
title: Document traceable artifact contract and ACP output mapping
status: Done
assignee: []
created_date: '2026-05-14 06:46'
updated_date: '2026-05-14 07:10'
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
1. Add a product contract doc for traceable generated work-product artifacts (#1525). Status: Complete.
2. Update ACP product/operator/readiness docs to map ACP run outputs into that contract (#1538). Status: Complete.
3. Run docs-only verification, update task evidence, and open a PR. Status: Complete.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added Traceable_Work_Product_Artifact_Contract.md for #1525, mapped ACP session outputs and promotable deliverables for #1538, and connected ACP PRD/operator/readiness/workspace docs to the contract. Verification so far: git diff --check passed; targeted rg presence and stale-language absence guards passed. Bandit is skipped because this slice only touches Markdown docs and Backlog metadata.

Opened draft PR #1687 and posted closeout-evidence comments on issues #1525 and #1538. Both issues should remain open until PR #1687 is reviewed and merged.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed docs/design slice for #1525 and #1538 in draft PR #1687. Added the traceable work-product artifact contract, mapped ACP run outputs and promotable deliverables into it, updated ACP readiness/operator/product/workspace docs, ran docs-only verification, and commented concrete evidence on both GitHub issues. Bandit was skipped because the touched scope is Markdown docs and Backlog metadata only.
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
