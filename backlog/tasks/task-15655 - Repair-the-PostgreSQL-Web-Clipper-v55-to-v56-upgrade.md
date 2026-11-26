---
id: TASK-15655
title: Repair the PostgreSQL Web Clipper v55-to-v56 upgrade
status: Done
labels:
- bug
- postgres
- migration
references:
- https://github.com/rmusser01/tldw_server/issues/3001
- Docs/ADR/036-web-clipper-external-identity-mapping.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix GitHub issue #3001 by accepting existing owner-scoped Web Clipper constraints during the v55-to-v56 PostgreSQL transition.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The PostgreSQL migration accepts both the legacy and the already owner-scoped schema.
- [x] #2 Owner-scoped identities and workspace mappings remain correct when users share a clip ID.
- [x] #3 The migration installs and validates the canonical constraints and preserves fail-closed data checks.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Existing ADR-036 governs owner-scoped Web Clipper identity and the transactional migration. The fix corrects historical schema recognition and normalization within that decision.
Restricted-role PostgreSQL reproduced the v56 constraint verification error on both empty and populated owner-scoped layouts. The migration verifies key shapes and referential actions, replaces recognized keys with canonical names, matches placements using the clip and source note, and uses the note owner for rekeying. It temporarily releases FORCE RLS for the verified table owner under the migration transaction and restores it before advancing the version.
Python 3.12 verification passed 37 migration and owner-isolation tests with PostgreSQL required and 46 schema structure and function-size checks. Scoped Ruff and Bandit passed. Independent review prompted exact foreign-key action checks and a regression for preserving a restrictive key.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
The PostgreSQL v56 step accepts the legacy and owner-scoped layouts and installs the verified canonical constraints. It preserves public clip IDs, reconciles the owner of each note and placement, and restores forced RLS before publishing the version. Real PostgreSQL tests cover both layouts, shared public IDs, canonical and legacy note IDs, restrictive custom keys, and rollback for a mismatched placement.
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
