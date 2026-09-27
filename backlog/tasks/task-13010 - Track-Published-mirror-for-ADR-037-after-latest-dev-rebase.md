---
id: TASK-13010
title: Track Published mirror for ADR 037 after latest dev rebase
status: Done
created_date: 2026-08-12 00:46
labels:
- docs
- ci
- adr
priority: high
references:
- https://github.com/rmusser01/tldw_server/pull/2774
- TASK-13004
- Docs/ADR/037-canonical-notes-link-sync-and-derived-graph-projections.md
modified_files:
- Docs/Published/ADR/037-canonical-notes-link-sync-and-derived-graph-projections.md
- Docs/Published/ADR/README.md
updated_date: 2026-08-12 00:49
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Repair the exact Docs Published manifest failure exposed after rebasing PR #2774 onto latest dev: ADR 037 exists in Docs/ADR but its generated Docs/Published mirror and index entry are absent.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Docs/Published/ADR/037-canonical-notes-link-sync-and-derived-graph-projections.md matches the canonical source ADR.
- [x] #2 Docs/Published/ADR/README.md includes ADR 037 without removing later upstream entries.
- [x] #3 The exact published-refresh manifest test passes.
- [x] #4 The repair is committed and linked to PR #2774.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce and verify the missing Published ADR 037 manifest entry. 2. Run the repository's published-docs refresh workflow. 3. Inspect the generated diff to keep only the required tracked mirror/index changes. 4. Run the exact manifest test and diff checks, then record verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Reproduced after rebasing PR #2774 onto origin/dev@414e81a12a: the exact Published manifest test failed because ADR 037 was generated but untracked. Ran Helper_Scripts/refresh_docs_published.sh; the only generated changes were the exact ADR 037 mirror and one README index line. Verified source/mirror byte equality with cmp, exact manifest test passed (1 passed), and git diff checks passed. This is a docs-only repair, so no additional Bandit scope applies; the PR's touched Embeddings production files separately passed Bandit with zero findings across 702 LOC.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the missing generated Published mirror for ADR 037 and its ADR index entry after the latest dev rebase exposed an upstream docs-manifest failure. Used the canonical refresh workflow so the mirror remains byte-identical to the source and preserved all newer upstream ADR entries. The exact manifest test and diff checks pass, and the repair is included in PR #2774.
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
