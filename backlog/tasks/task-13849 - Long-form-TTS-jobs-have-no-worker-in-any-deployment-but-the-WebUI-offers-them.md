---
id: TASK-13849
title: Long-form TTS jobs have no worker in any deployment, but the WebUI offers them
status: Done
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
- [x] #1 A default deployment either runs the TTS jobs worker (for example a lifespan worker behind a flag, like the other Jobs workers) or tells the caller that long-form TTS needs it, instead of leaving jobs queued silently
- [x] #2 The deployment docs say how to run the TTS jobs worker
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Fixed: app/services/startup_content_jobs_pollers.py gains the tts_jobs_task worker spec (route policy on the audio route via media_ingest_worker_predicate / should_start_inprocess_worker: on by default wherever the audio routes run, off in sidecar and test mode, TTS_JOBS_WORKER_ENABLED=false turns it off), running core/TTS/tts_jobs_worker.run_tts_jobs_worker; the server's startup already refreshes LLM provider overrides. The sidecar manifest gains the tts worker (default) and the tts-only profile selects it; the generator added the worker-tts compose service and tldw-worker-tts systemd/launchd units. Docs/Operations/Env_Vars.md documents TTS_JOBS_WORKER_ENABLED (with the restart and GPU notes) and TTS_JOBS_QUEUE and corrects AUDIO_JOBS_WORKER_ENABLED; Docs/Architecture/audio.md, jobs-background.md and Docs/Deployment/Long_Term_Admin_Guide.md say where the worker runs. Chosen over refusing long-form jobs when no worker runs, because the endpoint cannot see a standalone or sidecar worker elsewhere. Upgrade note: sidecar deployments with hand-installed units or an explicit TLDW_SIDECAR_WORKERS list must add tts. Tests: RED 9 failed, 72 passed on dev (no tts_jobs_task spec, no tts sidecar worker), GREEN 81 passed; Services, sidecar, startup, Docs, TTS_NEW, the TTS Audio tests and lint 1 failed, 3411 passed, 8 skipped, 1 xfailed, 1 xpassed. Ruff clean on the changed files apart from the I001 dev already has; bandit clean.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
The long-form TTS jobs worker runs in-process wherever the audio routes do (on by default by route policy, off in sidecar and test mode, TTS_JOBS_WORKER_ENABLED=false turns it off) and as the tts sidecar worker, so jobs from POST /api/v1/audio/speech/jobs and the WebUI are processed in a default deployment.
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
