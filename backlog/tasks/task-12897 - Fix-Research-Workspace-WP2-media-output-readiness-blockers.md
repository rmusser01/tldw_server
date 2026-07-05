---
id: TASK-12897
title: Fix Research Workspace WP2 media-output readiness blockers
status: Done
labels:
- research-workspace
- notebooklm
- wp2
- backend-jobs
- bug
references:
- TASK-12173
- https://github.com/rmusser01/tldw_server/pull/2669
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Root-cause and fix the two blockers recorded by TASK-12173: image generation capability reports an unhelpful `image_backend_unknown` when the backend is enabled but not configured, and Video Overview jobs can surface a generic `worker_exception` even when the linked artifact records `tts_generation_failed`. Keep scope to backend job readiness/capability behavior and precise public failure codes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Infographic capability readiness fails closed with a precise reason when enabled image backends are not configured.
- [x] #2 Video Overview capability readiness accounts for failed enabled TTS providers without synthesizing audio in the capabilities endpoint.
- [x] #3 Research Workspace output job status prefers the linked artifact's precise error when the Jobs row only has `worker_exception`.
- [x] #4 Focused tests cover the fixed readiness/failure behavior for the touched backend paths.
- [x] #5 No WP3 discovery-loop or WP4 agent-task UI work is included.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Root cause:
- Image readiness already detected enabled image backends, but when no configured image model was cataloged it returned the generic `image_backend_unknown`; this made the WP2 Infographic blocker harder to diagnose.
- TTS readiness only counted configured/enabled providers and did not account for providers already marked failed by the TTS factory status, so Video Overview could appear allowed even when all enabled TTS providers were unusable.
- Output job status trusted the Jobs row error first; when the worker row held generic `worker_exception`, the API hid the linked failed artifact's precise `producer_metadata.error` such as `tts_generation_failed`.

Implementation:
- Preserved precise TTS health reason codes through Research Workspace capability derivation.
- Added lightweight TTS factory status inspection that subtracts failed enabled providers without synthesizing audio.
- Changed image readiness to return `image_backend_not_configured` when backends exist but no configured model exists.
- Changed output job status to prefer the failed artifact error when the job error is absent or generic `worker_exception`.
- Added focused regression coverage for image readiness, TTS failed-provider readiness, and artifact error precedence.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Verification:
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Research_Workspace/test_capability_derivation.py tldw_Server_API/tests/Research_Workspace/test_output_jobs_api.py -q` -> 40 passed, 4 warnings.
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Research_Workspace/test_capability_endpoint.py tldw_Server_API/tests/Research_Workspace/test_output_jobs_worker.py -q` -> 34 passed, 2 warnings.
- `git diff --check` -> exit 0.
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/core/Research_Workspace/capabilities.py tldw_Server_API/app/core/Research_Workspace/output_jobs.py -f json -o /tmp/bandit_research_workspace_wp2_readiness.json` -> 0 findings, 0 errors.

Remaining environment requirement: actual WP2 media generation still requires configured local image/TTS providers; this change makes the readiness/status failures precise and fail-closed when providers are missing or failed.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Root cause investigation notes are recorded in the task.
- [x] #2 Focused tests pass.
- [x] #3 Bandit runs on touched backend Python scope.
- [x] #4 Final summary records verification and any remaining local environment requirements.
<!-- DOD:END -->
