---
id: TASK-2327
title: Revise Scheduled Tasks Phase 2B capability shell plan after review
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-09 04:32'
labels:
  - scheduled-tasks
  - webui
  - ux
  - phase-2b
  - implementation-plan
  - review
dependencies: []
references:
  - TASK-2326
  - >-
    Docs/superpowers/plans/2026-06-09-scheduled-tasks-phase2b-capability-aware-frontend-shell-implementation-plan.md
  - >-
    Docs/superpowers/specs/2026-06-09-scheduled-tasks-phase2b-watch-ingest-product-contract-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Plan prevents Watch/Ingest templates from appearing Available in Phase 2B.2 unless a real creation adapter/frontdoor is explicitly supported.
- [x] #2 Plan requires source-intent detection details and reason copy to be visible and tested in the Create panel.
- [x] #3 Plan expands redaction test coverage for URL fragments, common secret query params, bearer/prose secrets, and provider snippets.
- [x] #4 Plan includes extension-width/narrow-container verification for the Create panel shell.
- [x] #5 Plan fixes the file-structure indentation issue found in review.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed all review findings in the Phase 2B.2 capability-aware frontend shell plan. The plan now requires both all gates and creationAdapterSupported === true before Watch/Ingest can resolve Available, keeps page defaults non-creating, surfaces source-intent copy in UI/tests, expands redaction coverage for URL fragments/query secrets/bearer/prose/provider snippets, adds extension-width component coverage, and fixes the file-list indentation. Verification: git diff --check exited 0; stale wording and placeholder scan exited 1 with no matches. Bandit skipped because this is a docs-only plan revision with no Python touched.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Backlog task tracks the revision and final verification.
- [x] #2 Plan file is updated with focused changes only.
- [x] #3 Docs checks pass with no unresolved placeholders introduced.
- [x] #4 Changes are committed with the Backlog task update.
<!-- DOD:END -->
