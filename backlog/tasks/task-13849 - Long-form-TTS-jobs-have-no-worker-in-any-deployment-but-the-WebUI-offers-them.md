---
id: TASK-13849
title: Long-form TTS jobs have no worker in any deployment, but the WebUI offers them
status: To Do
labels:
- tts
- jobs
- bug
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
POST /api/v1/audio/speech/jobs queues tts_longform jobs (on the tts queue since TASK-13641), and the WebUI calls it (apps/packages/ui/src/services/tldw/domains/models-audio.ts and TldwApiClient.ts). But nothing starts the TTS jobs worker: there is no lifespan hook in app/services/startup_content_jobs_pollers.py, no sidecar entry, no Dockerfile and no service unit. It runs only when an operator starts python -m tldw_Server_API.app.core.TTS.tts_jobs_worker, so on a default deployment a long-form speech job stays queued with nothing to say why. Found in the plan review for TASK-13641.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A default deployment either runs the TTS jobs worker (for example a lifespan worker behind a flag, like the other Jobs workers) or tells the caller that long-form TTS needs it, instead of leaving jobs queued silently
- [ ] #2 The deployment docs say how to run the TTS jobs worker
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
