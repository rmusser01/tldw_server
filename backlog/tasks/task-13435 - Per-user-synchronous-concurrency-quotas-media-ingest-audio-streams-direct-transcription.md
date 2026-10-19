---
id: TASK-13435
title: Per-user synchronous concurrency quotas (media ingest, audio streams, direct
  transcription)
status: Done
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deferred from spec 2 (Docs/Design/2026-10-02-usage-quota-posture-design.md, review ruling R1). RGRequest has no per-request max_concurrent, and PR B stopped reserving the media/audio jobs/streams leases, so these paths are unlimited. Approach: an optional RGRequest.max_concurrent honored by the memory and Redis governors, with limits.media_concurrent_jobs / limits.audio_concurrent_streams resolved per user. Parent: TASK-13434.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A per-user concurrency limit on media ingest and audio streams is enforced on both RG backends
- [x] #2 Unset means unlimited
- [x] #3 A per-user concurrency limit on direct transcription (POST /audio/transcriptions and /audio/translations) is enforced on both RG backends
- [x] #4 A held slot is released on success, error, client disconnect or cancellation, and graceful shutdown; a lease nobody releases lapses within 120 s
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plan: Docs/superpowers/plans/2026-10-18-per-user-concurrency-quotas.md (one PR). ADR check: ADR required: yes, ADR path: Docs/ADR/076-per-user-concurrency-quotas.md. The change makes synchronous concurrency a usage quota, which ADR-064's consequences ruled out, and takes media/audio concurrency out from under RG_ENABLED (ADR-056); a partial amendment by the ADR-056 precedent. ADR-075 is claimed by open PR #3270, so this is ADR-076. Owner decisions (2026-10-18): D1 (RG_ENABLED does not turn them off) = yes, as recommended: the limits follow USAGE_QUOTAS_ENABLED only; D2 (direct transcription key limits.audio_concurrent_transcriptions) = yes, as recommended, added as AC #3; D3 (keep the name limits.media_concurrent_jobs) = yes, keep it. ADR-076 accepted by the owner with the plan.
Implemented on feat/per-user-concurrency-quotas (PR #3271). RED/GREEN: T1 governors 13 failed -> 13 passed (real Redis); T2 slots 1 collection error + catalog 1 failed -> 15 passed + catalog (23); T3 streams 5 failed -> 17 passed; T4 transcription 2 failed -> 2 passed; T5 media 1 failed -> 32 passed (with the email harness). Release, heartbeat-cancel, shield, keep-alive owner check, RedisError, Redis commit() and renew() lines each shown to fail exactly their tests when removed. Sweep after rebase on origin/dev 4d35534699 (Resource_Governance, Usage, Audio, AudioJobs, UserProfile, MediaIngestion_NEW, Docs, TTS endpoints, CI ratchet): 1 failed, 3774 passed, 16 skipped, 3 xfailed, real Redis, 0 rg_rcl_* keys left; only failure test_nemo_streaming_error_sentinel (no pyaudio, fails on origin/dev). Test-quality ratchet: OK. Ruff tallies no worse than origin/dev (2 I001 + 1 SIM114 vs 3 I001 + 1 SIM114); Bandit -ll: only the pre-existing B113 false positive at policy_eval.py:77. ADR-076 written (Accepted). Follow-up: TASK-13859 (RG _RG_NONCRITICAL_EXCEPTIONS misses RedisError).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Per-user concurrency quotas: limits.media_concurrent_jobs (/media/add), limits.audio_concurrent_streams (audio WebSockets, realtime, POST /audio/chat) and limits.audio_concurrent_transcriptions (/audio/transcriptions, /audio/translations). Unset is unlimited, 0 blocks, resolved per user/team/org (ADR-064). Each request holds one RG lease: an RGRequest streams/jobs entry may now carry max_concurrent/ttl_sec, honored on the memory and Redis backends; the Redis backend now releases leases whose policy has no block and renews only the buckets a lease charged. Slots renew every 30 s while their task runs, release shielded as the last statement of finally on success/error/disconnect/shutdown, and lapse within 120 s if never released. Redis errors never fail a request. The lease store follows RG_BACKEND and is not turned off by RG_ENABLED (ADR-076). Also fixed: /stream/transcribe took its slot 312 lines before the try that released it, and create_transcription leaked its job-heartbeat task on early exits. Known skips: test_nemo_streaming_error_sentinel (no pyaudio). Not done: an active-stream count in /audio/stream/limits (active_streams stays null, can_start_stream stays true); RG's own RedisError gap is TASK-13859.
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
