---
id: TASK-13600
title: Persist explicit native attachment reference roles
status: Done
labels:
- chat
- ux
- database
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The approved H2 attachment context requires explicit reference roles. Current reference storage omits this authority, and embedded-image representation cannot distinguish message images from reference images. Add a new registered SQLite/PostgreSQL migration without modifying published migrations or inventing legacy roles; preserve existing data and leave canonical context/CAS/public qualification pending.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Fresh and repeated initialization and genuine prior-version upgrades install a nullable constrained reference role on both registered schema paths.
- [x] #2 Existing references, claims, receipts, quota records, bundles and revisions remain unchanged; no legacy role is inferred.
- [x] #3 Causal regression tests, connected checks, scoped static/security checks and two independent reviews are recorded with unavailable PostgreSQL validation explicit.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR assessment: ADR-049/050 and accepted H2 govern this additive storage detail; no new ADR. Own3stage plan completed and retired only to /private/tmp/IMPLEMENTATION_PLAN_chat_reference_roles_13600_completed.md. Record: Docs/Reviews/CHAT_REFERENCE_ROLES_13600_2026_10_12.md. New registered SQLite79-to80/Pg83-to84 nullable constrained role, all5 paths and Pg82/83 short admission; published DDL/readers/RLS unchanged. Causal red3/3PG, migration9/9PG, finalconnected326/208skip/12warnings107.57s. Matched3file Ruff0/0, final manual lint/new-code formatter/AST and supported compile pass; actual1production-file Bandit0/0/errors0. Two independent source CLEAR; version-fixture maintenance addressed without changing pinned prior setup. Same sealed profile/supportedPython API59482 after verified old27579 graceful exit; immutable original9/28 and distinct character3 canonical rows exact, one existing headless tab/current selection/blank composer. First incorrect collection path and network-idle harness diagnostics excluded. Initial exact-owned8-file hooks0; final7file hooks/localcommit follow. Physical PG, canonical context reader/CAS/source fence/retention publication/public readiness and broader acceptance unqualified.
All3AC/6DOD done, initial8/final7 normal exact-owned hooks0 with no bypass; only own completed plan retired. Fresh dev fetch stilla744577 with0 missing commits. Seven owned files only for local commit; no PR/push/remote merge. Broader parent criteria remain open.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Persist explicit native reference roles in new registered migrations, preserving all legacy records with unknown role instead of inferring from representation. Connected326pass/208skip and manual lint/security/compile evidence pass; both source reviewers clear. Same-profile runtime and original records preserved. No canonical context or public H2 readiness is granted by this prerequisite.
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
