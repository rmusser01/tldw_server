---
id: TASK-13185
title: Expose synchronous audio summarization in Service Prompts
status: Done
assignee: []
created_date: '2026-09-05 18:15'
updated_date: '2026-09-05 18:29'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Approved bounded slice: expose the existing audio analysis system/user instruction pair through shared Service Prompts Settings. Resolve one authenticated-owner pair per synchronous request; preserve explicit prompts, deployment defaults, analysis disablement and provider configuration. Exclude transcription changes, video and queued/persisted ingestion.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Shared WebUI and extension can edit and reset the audio system/user pair
- [x] #2 Synchronous audio requests preserve explicit-field precedence and freeze owner-specific instructions across files and summary passes
- [x] #3 Disabled analysis and absent providers do not read prompt storage; deployment defaults remain effective
- [x] #4 Regression tests, security checks and independent review pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Register the literal audio pair and extend shared Settings metadata. 2. Resolve missing request parts once at the authenticated boundary; preserve explicit empty values without changing direct/background defaults. 3. Exercise real multipart/storage/batch/analyzer behavior and shared/WebUI editor flows. 4. Verify security, OpenAPI, regression suites and independent review.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Approved design: Docs/Design/audio-summary-service-prompt.md. Incremental slice of TASK-12957 (broad media/audio migration). Worktree codex/audio-summary-service-prompt from dev 86eb9e517c. Baseline102 passed; new behavior RED13 failures, registryRED3 failures, UIRED1 failure. Reviewer found existing direct-form test helper must supply Request; verified three failing audio/ebook/email cases and adapted helper without weakening production. Audio deployment fixture now isolates its own file approval policy; long transcript exercises recursive passes. Shared UI198 passed; WebUI5 targeted passed. Verification in progress.

Final verification: 236 backend regressions passed (10 warnings), 198 shared UI tests passed, and 5 targeted WebUI Settings tests passed. Independent reviewer approved with no remaining actionable findings. Compileall and Ruff lint passed for touched runtime/tests; changed endpoint/registry/form/tests formatted. Existing audio_batch formatting drift is present at base and was not swept into this feature. Bandit touched runtime scope: zero findings/errors (/tmp/bandit_audio_service_prompt.json). Official OpenAPI export, TypeScript generation, and fingerprint check passed (2068 paths/3133 schemas). Full repository suite, full frontend typecheck/build, live browser/STT/provider calls were not run locally; tests replace external transcription/model boundaries. No implementation blockers; PR/integration not yet requested.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added media.audio.analysis as a literal system/user pair through the existing shared Service Prompts Settings and API. Synchronous audio processing captures one authenticated-owner pair before work begins, preserves independent explicit values including empty text, and retains deployment defaults plus direct/background caller behavior. Reusing the registry/editor/storage avoids an audio-only configuration system. Updated the direct-form regression helper to supply real Request objects and regenerated the OpenAPI fingerprint.
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
