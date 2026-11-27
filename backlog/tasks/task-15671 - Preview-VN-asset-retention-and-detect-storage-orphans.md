---
id: TASK-15671
title: Preview VN asset retention and detect storage orphans
status: Done
labels:
- vn-assets
- storage
- retention
references:
- https://github.com/rmusser01/tldw_server/issues/2025
- https://github.com/rmusser01/tldw_server/pull/3648
- Docs/ADR/020-db-management-per-user-paths-and-content-backend.md
- Docs/ADR/094-jobs-resource-admission-and-domain-publication.md
- Docs/Design/2026-11-26-vn-asset-retention-and-orphans.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue GitHub #2025 on current remote dev. Add age-aware cleanup previews and a read-only owner-scoped inventory for orphaned bytes, orphaned registrations and broken pack pointers.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A retention preview applies documented age thresholds and omits protected, active, or still-referenced assets. Existing explicit cleanup and stored idempotency receipts keep their behavior.
- [x] #2 The owner can page through orphaned bytes and metadata, including missing registrations and bytes, with no cross-owner, trash, or in-flight false positives for supported paths.
- [x] #3 The inventory and retention preview never delete data. Document thresholds, evidence and scan limits and verify with owner-scoped storage and API tests.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Base: remote dev ef6ebe2ee0. ADR required: no. ADR-020 governs owner-local paths and ADR-094 governs active work. Both new views are read-only and reuse existing storage ownership and cleanup checks.
The retention-only dry run applies 30 days to rejected items and 7 days to hidden items. Existing manual cleanup keeps its response replay and approval controls. Published drafts, approved assets, shared or referenced files, recent items and active pack work stay protected in the preview.

The owner-only storage report pages through item pointers, AuthNZ registrations and recognized VN image files. The scan gives writes 24 hours to settle, checks owner-local portability Jobs, and protects restorable trash and paths in published scripts or retained play data. It reports a superseded attempt after the item has a different complete binding. A report never removes files or registrations.

Verification passed for 88 tests across VN storage, portability and AuthNZ boundaries. The final focused rerun passed 21 cases. The AuthNZ storage path test used SQLite and exercised the PostgreSQL query boundary with a transaction test double. OpenAPI fingerprint and path union, scoped Ruff and Bandit passed. Bandit reported zero findings.

The report covers normal VN image filenames, including date-directory layouts. Preview archives and unrecognized paths are outside the scan. The existing explicit cleanup endpoint is still the operator control for deletion.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added an age-aware cleanup preview and an owner-scoped, read-only report for missing and unreferenced VN storage.
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
