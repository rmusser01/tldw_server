---
id: TASK-13357
title: Backfill per-user audio preset ownership ADR
status: Done
assignee: []
created_date: '2026-09-25 16:24'
updated_date: '2026-09-25 19:17'
labels:
  - docs
  - adr
  - audio
dependencies: []
references:
  - Docs/ADR/inventory/2026-06-03-decision-inventory.md
  - Docs/Design/Audio_Presets_Ownership_2026_05.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Backfill the implemented TTS/STT preset ownership decision from INV-022 and the accepted Audio Presets Ownership design. Keep the ADR limited to current Audio API, Media DB, and browser-local behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Add one accepted ADR for the implemented per-user audio preset ownership and storage/API boundary with evidence and alternatives.
- [x] #2 Link the ADR from the index, decision inventory, and relevant source design; keep known validation and schema caveats explicit.
- [x] #3 Sync published documentation and record focused verification and Bandit non-applicability.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm design, endpoint, storage, schemas, and tests. 2. Draft bounded ADR and references. 3. Verify source/published consistency and close the task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
ADR check: ADR required: yes. ADR-047 records the current per-user Audio API/Media DB ownership boundary, backed by TASK-12356 owner decision and TASK-12363 implementation. Live provider-readiness validation and stricter kind/config behavior are explicitly outside this ADR.

Verification 2026-09-25: Inspected audio preset endpoint, schema, Media DB SQLite/PostgreSQL structures, and existing Audio endpoint tests. ADR/source design links resolve, ADR-047 index status matches, source/published ADR copies compare byte-for-byte, and git diff --check passed. Documentation-only backfill; pytest and Bandit not run. No .venv exists in isolated worktree.

PR: https://github.com/rmusser01/tldw_server/pull/3014. Merge gate remains: requester must provide a human-written Change summary.

PR #3014 Qodo review: added a dated subsequent-resolution note to ADR-011 and its published copy, preserving the original accepted rationale while clarifying that ADR-047 now covers INV-022. Qodo task-status finding was already resolved by commit b25cd3d4ef.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Backfilled the implemented per-user Audio API/Media DB audio preset ownership decision as accepted ADR-047. Linked it from the ADR index, June/September inventories, and source design; preserved endpoint validation and schema caveats. Published ADR mirror matches. PR #3014. Docs-only: pytest/Bandit not applicable; no blocker to documentation work.
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
