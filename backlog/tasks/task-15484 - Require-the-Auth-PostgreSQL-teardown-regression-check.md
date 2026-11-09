---
id: TASK-15484
title: Require the Auth PostgreSQL teardown regression check
status: In Progress
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow up on PR #3394. Make the late PostgreSQL connection at Auth test teardown reproducible and run the focused regression in the required backend workflow.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A real connection opened at the isolated fixture drop boundary fails the old split cleanup and passes the merged cleanup against PostgreSQL.
- [x] #2 The existing backend-required workflow runs the focused regression for backend changes, with missing PostgreSQL or an unexecuted case treated as failure.
- [x] #3 The workflow keeps existing license admission and backend no-op behavior and adds no new protected status name.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plan: 1. Bind the existing fixture and verify a deterministic red/green control. 2. Wire one short test into backend-required and verify the CI contracts. 3. Review and record the result. ADR required: no. This exercises the existing fixture and required check without changing runtime or repository policy. ADR-059 and ADR-084 govern tracking.
Local proof: ran the exact workflow command against existing PostgreSQL 18 and the real isolated Auth fixture. A real session connects immediately before the drop. The historical split terminate/drop raised ObjectInUseError (26.43 s outer); the merged atomic drop passed with one executed case and no skips (20.03 s outer). Historical source restored byte-for-byte. The observer verifies the database is absent and only recovers the same fixture database if the check fails. No runtime Auth code changed.

The existing backend-required job runs this focused test before compile and the other backend tests. Its conditional PostgreSQL service is selected only after successful backend change detection. Missing PostgreSQL, pytest failure, and missing or skipped JUnit execution are blocking. The existing gate name, license admission and backend no-op contract are preserved. Scoped Bandit baseline/candidate 61/61, zero new findings; Ruff clean.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
