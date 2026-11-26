---
id: TASK-15667
title: 'Verify and harden v1 VN asset-pack portability for GitHub #2026'
status: Done
labels:
- vn-assets
- portability
- uat
references:
- https://github.com/rmusser01/tldw_server/issues/2026
- Docs/ADR/003-jobs-vs-scheduler-default.md
- Docs/ADR/094-jobs-resource-admission-and-domain-publication.md
- Docs/API-related/VN_ASSET_PACKS_API.md
- Docs/Reviews/VN_PACK_V1_PORTABILITY_2026_11_25.md
- TASK-15667.1
- TASK-15667.2
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Exercise the shipping tldw.vnpack.v1 workflow on freshly fetched dev with isolated authenticated installations. Verify archive integrity, byte and metadata transfer, ownership and review behavior, conflict handling, interruption and retry behavior, and cancelled or expired preview cleanup. Correct bounded v1 failures with focused regression coverage and document any remaining gaps and richer-format dependencies.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Export an approved v1 pack and import it into independently provisioned storage with verification of bytes, metadata, ownership remapping and both trust modes.
- [x] #2 Verify repeated commit and interrupted delivery behavior, ordinary failure rollback, and cancelled or stale preview cleanup. Correct bounded defects and name remaining recovery gaps.
- [x] #3 Record v1 format compatibility and conflict decisions and prove corrupted, missing-byte and quota-blocked cases against the existing API and UI where applicable.
- [x] #4 Record reproducible evidence, focused regression results and scoped security validation.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required: no. This v1 exercise and any bounded fixes use the existing Jobs and owner-scoped persistence contracts from ADR-003 and ADR-094. Freshly fetched remote dev was 1d8589b7c43e5dbb6fca3247e023f562129c70f3.
Completed the two-install v1 API proof and live Chromium import after repair. Verified byte identity, ownership remapping, trusted and untrusted review states, duplicate character handling, corruption, missing bytes, quota, cancellation and staged archive deletion. Chromium restored into an empty account and discarded the old preview after switching accounts.
Post-completion worker replay returns the original result. The journal transition and readback protect a completed record and the legacy repair matches the known create-new case. The workbench uses the validated included-character action on an empty destination, explains quarantine and fences previews by account revision. Independent review findings were addressed.
Final focused checks passed 89 backend and 12 frontend tests. Changed frontend ESLint, Python compilation and production Bandit passed. Bandit found zero issues or errors. TASK-15667.1 and TASK-15667.2 track recovery before completion and workbench job status and downloads. The GitHub issue remains open.
The expanded frontend run had 371 passes and two workbench storage timeouts that also reproduced with the original components in this worktree. The expanded backend run was stopped after setup errors. The replay I/O boundary external DNS error reproduced on unchanged dev backend files.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Verified v1 backup portability across two authenticated installations and corrected completed-import replay and empty-account workbench restore. Recorded the browser and API evidence and linked the two remaining recovery follow-ups.
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
