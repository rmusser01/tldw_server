---
id: TASK-15633
title: Parse legacy media source hashes as JSON
status: Done
labels:
- media
- bug
references:
- https://github.com/rmusser01/tldw_server/pull/3602
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Local audio/video deduplication misses legacy source hashes in normally formatted safe_metadata JSON. Match the top-level hash regardless of JSON whitespace and tolerate unreadable legacy metadata.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A local re-upload with a matching legacy top-level hash is skipped for compact and spaced JSON.
- [x] #2 Malformed, nested, or non-object metadata cannot produce a duplicate match.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required: no. This corrects how the existing metadata fallback interprets stored JSON. Verification stages: reproduce in the local audio pre-check, fix and run adjacent tests, then review and publish.
Regression tests reproduced the whitespace and malformed JSON behavior before the fix. The final focused run passed all 56 tests, including an unreadable version followed by a valid match. Ruff and git diff --check passed. Bandit reported only pytest assertions.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
The media lookup repository parses legacy metadata and matches its top-level hash. Tests cover compact, spaced, malformed, nested, and deeply nested documents. The draft PR contains the fix and the updated design note.
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
