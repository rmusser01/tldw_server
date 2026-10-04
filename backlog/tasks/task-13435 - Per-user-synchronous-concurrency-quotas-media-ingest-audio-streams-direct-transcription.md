---
id: TASK-13435
title: Per-user synchronous concurrency quotas (media ingest, audio streams, direct
  transcription)
status: To Do
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deferred from spec 2 (Docs/Design/2026-10-02-usage-quota-posture-design.md, review ruling R1). RGRequest has no per-request max_concurrent, and PR B stopped reserving the media/audio jobs/streams leases, so these paths are unlimited. Approach: an optional RGRequest.max_concurrent honored by the memory and Redis governors, with limits.media_concurrent_jobs / limits.audio_concurrent_streams resolved per user. Parent: TASK-13434.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A per-user concurrency limit on media ingest and audio streams is enforced on both RG backends
- [ ] #2 Unset means unlimited
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
