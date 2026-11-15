---
id: TASK-15500
title: 'Own-key media analysis: act on PR D''s final-review minors'
status: In Progress
labels:
- byok
- usage-quotas
- tests
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up of TASK-15426 PR D (#3470). Its final review left eight Minor findings (numbered 2-9 there). Three get a fix: a unit test over every entry of media_analysis_provider's other-stage denylist, including claims by the server's ENABLE_INGESTION_CLAIMS default (2); the media ingest job's fire-and-forget embeddings task inheriting the job's admission and the owner's key (4); two test fixtures built at import time (8). Five need no change, each for a stated reason (3, 5, 6, 7, 9).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 media_analysis_provider has a unit test per denylist entry, and one for the server's claims default on, off, and opted out
- [ ] #2 The media ingest job's embeddings task runs with no provider key and no admission mark, with a test that fails without the fix
- [ ] #3 test_media_llm_quota.py builds no fixture bytes at import time
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
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
