---
id: TASK-15634
title: Honor the stored media source hash before legacy metadata
status: Done
labels:
- media
- bug
references:
- https://github.com/rmusser01/tldw_server/pull/3602
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Local audio/video deduplication can skip a file based on a historical document version even when the media row has a different stored hash. Apply the legacy fallback only when the row lacks a source hash or the column is unavailable.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A local re-upload with a matching historical hash and a different current hash proceeds to processing.
- [x] #2 The indexed match remains authoritative, and a NULL hash still allows the legacy fallback.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required: no. This restores the precedence specified by the existing media pre-check design. Verification stages: reproduce with mismatched row and version hashes, verify the fix and indexed path, then review and publish.
Regression tests reproduced stale metadata overriding the stored hash and a failed schema inspection unlocking fallback. The final focused run passed all 56 tests, including a genuine old SQLite schema. Ruff and git diff --check passed. Bandit reported only pytest assertions.
Independent review follow-up: catch the backend error family during schema inspection and verify filename and live PostgreSQL owner precedence.
Independent review follow-up verifies real backend exception handling, preserved local filename identity, and owner filters on a live PostgreSQL instance. Final validation: 62 focused tests plus one live PostgreSQL test passed. Ruff and Bandit production checks passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stored hashes retain precedence when schema inspection raises a backend error. The lookup only checks legacy metadata for NULL hashes or a confirmed old schema, and preserves filename and PostgreSQL owner scoping.
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
