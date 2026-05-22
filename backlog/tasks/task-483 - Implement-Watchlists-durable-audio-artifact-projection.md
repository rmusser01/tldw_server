---
id: TASK-483
title: Implement Watchlists durable audio artifact projection
status: In Progress
assignee: []
created_date: ''
updated_date: 2026-05-22 21:00
labels:
- watchlists
- audio
- implementation
dependencies: []
documentation:
- Docs/superpowers/specs/2026-05-22-watchlists-durable-audio-artifact-projection-design.md
- Docs/superpowers/plans/2026-05-22-watchlists-durable-audio-artifact-projection-implementation-plan.md
priority: high
modified_files:
- tldw_Server_API/app/core/DB_Management/Workflows_DB.py
- tldw_Server_API/app/core/Scheduler/handlers/workflows.py
- tldw_Server_API/app/core/Watchlists/audio_briefing_workflow.py
- tldw_Server_API/app/core/Watchlists/audio_artifact_projection.py
- tldw_Server_API/app/core/Workflows/adapters/_common.py
- tldw_Server_API/app/core/Workflows/adapters/audio/_config.py
- tldw_Server_API/app/core/Workflows/adapters/audio/multi_voice_tts.py
- tldw_Server_API/app/core/Workflows/adapters/audio/tts.py
- tldw_Server_API/app/core/Workflows/adapters/content/audio_briefing.py
- tldw_Server_API/app/api/v1/endpoints/watchlists.py
- tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py
- tldw_Server_API/tests/Workflows/test_workflows_run_metadata.py
- tldw_Server_API/tests/Workflows/test_adapter_path_security.py
- tldw_Server_API/tests/Workflows/adapters/test_audio_adapters.py
- tldw_Server_API/tests/Workflows/adapters/test_content_adapters.py
- tldw_Server_API/tests/Watchlists/test_audio_briefing_workflow.py
- tldw_Server_API/tests/Watchlists/test_audio_artifact_projection.py
- tldw_Server_API/tests/Watchlists/test_audio_output_delivery.py
- tldw_Server_API/tests/Watchlists/test_watchlists_operator_recovery.py
- apps/packages/ui/src/types/watchlists.ts
- apps/packages/ui/src/components/Option/Watchlists/OutputsTab/outputMetadata.ts
- apps/packages/ui/src/components/Option/Watchlists/OutputsTab/OutputPreviewDrawer.tsx
- apps/packages/ui/src/components/Option/Watchlists/RunsTab/RunDetailDrawer.tsx
- apps/packages/ui/src/assets/locale/en/watchlists.json
- apps/packages/ui/src/public/_locales/en/watchlists.json
- apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/outputMetadata.test.ts
- apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/OutputPreviewDrawer.audio.test.tsx
- apps/packages/ui/src/components/Option/Watchlists/RunsTab/__tests__/RunDetailDrawer.stream-lifecycle.test.tsx
- Docs/superpowers/plans/2026-05-22-watchlists-durable-audio-artifact-projection-implementation-plan.md
- backlog/tasks/task-483 - Implement-Watchlists-durable-audio-artifact-projection.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved Watchlists durable audio artifact projection plan. Work is staged so each task can be tested and committed independently, beginning with durable Workflows correlation metadata and continuing toward request IDs, artifact tagging, projection/read-repair, retry hardening, frontend rendering, and verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Workflows runs persist audio correlation metadata and expose it through `get_run`, `list_runs`, and idempotency lookup paths.
- [x] #2 Scheduler `workflow_run` propagates payload metadata into both run metadata and workflow definition metadata without mutating caller payloads.
- [x] #3 Watchlists audio triggers generate retry-safe `audio_request_id` values and persist them in run/output metadata.
- [x] #4 Workflow audio/script artifacts carry enough correlation and role metadata for projection.
- [x] #5 `/watchlists` run audio endpoint can repair and return durable script/speaker/final audio projections from canonical Workflows artifacts.
- [x] #6 Retry paths do not present stale audio artifacts as the active request.
- [x] #7 Frontend renders durable mirrored audio graph, stale state, and live overrides without raw file paths.
- [x] #8 Focused backend/frontend tests and Bandit/diff hygiene are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute the durable audio projection plan in staged commits. Start with Task 1: persist Workflow run metadata and propagate Scheduler payload metadata into Workflow definition metadata for adapter context. Continue through request IDs, artifact tagging, projection/read-repair, retry stale-state, frontend graph rendering, and verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Task 1 completed:
- Added `workflow_runs.metadata_json` support to the SQLite and backend Workflows schemas with schema version 9 migrations.
- Added `metadata` to `WorkflowsDatabase.create_run(...)`.
- Added metadata persistence coverage for `get_run`, `list_runs`, `get_run_by_idempotency`, and legacy SQLite schema migration.
- Updated Scheduler `workflow_run` to persist payload metadata and merge it into a copied `definition_snapshot["metadata"]` for adapter context without mutating the caller payload.

Task 1 verification:
- Red run: `source ../../.venv/bin/activate && python -m pytest tldw_Server_API/tests/Workflows/test_workflows_run_metadata.py -q` failed for missing `metadata`, missing `metadata_json`, and dropped Scheduler metadata.
- Green run: same command passed, `3 passed`.
- `source ../../.venv/bin/activate && python -m pytest tldw_Server_API/tests/Workflows/test_workflows_db.py -q` passed, `12 passed`.
- `source ../../.venv/bin/activate && python -m pytest tldw_Server_API/tests/Workflows/test_workflows_postgres_migrations.py -q` skipped locally, `1 skipped`.
- `source ../../.venv/bin/activate && python -m pytest tldw_Server_API/tests/Workflows/test_workflows_postgres_indexes.py -q` skipped locally, `2 skipped`.
- `source ../../.venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/DB_Management/Workflows_DB.py tldw_Server_API/app/core/Scheduler/handlers/workflows.py -f json -o /tmp/bandit_watchlists_audio_projection_task1.json` passed with `results 0`.

Task 2 completed:
- Added opaque `wla_...` audio request IDs to Watchlists audio trigger results.
- Propagated `audio_request_id` into Workflow inputs, Workflow payload metadata, Scheduler metadata, and the Scheduler idempotency key.
- Kept `audio_request_id` out of user/job `output_prefs`; stale supplied values are ignored.
- Persisted the active request ID through `apply_audio_briefing_result_metadata(...)`.
- Updated audio retry handling to drop the active `audio` projection, preserve the previous one under `previous_audio`, and mark it stale/superseded by the new request ID.

Task 2 verification:
- Red run: `source ../../.venv/bin/activate && python -m pytest tldw_Server_API/tests/Watchlists/test_audio_briefing_workflow.py::TestBuildWorkflowInputs::test_audio_result_metadata_persists_request_id tldw_Server_API/tests/Watchlists/test_audio_briefing_workflow.py::TestTriggerAudioBriefing::test_trigger_submits_workflow tldw_Server_API/tests/Watchlists/test_watchlists_operator_recovery.py::test_retry_run_audio_reuses_job_audio_config_without_rerunning_ingestion -q` failed for missing `audio_request_id` support.
- Green run: same command passed, `3 passed`.
- `source ../../.venv/bin/activate && python -m pytest tldw_Server_API/tests/Watchlists/test_audio_briefing_workflow.py tldw_Server_API/tests/Watchlists/test_watchlists_operator_recovery.py -q` passed, `40 passed`.
- `source ../../.venv/bin/activate && python -m pytest tldw_Server_API/tests/Watchlists/test_watchlists_pipeline.py tldw_Server_API/tests/Watchlists/test_watchlists_api.py -q` passed, `38 passed, 1 skipped`.
- `source ../../.venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Watchlists/audio_briefing_workflow.py tldw_Server_API/app/api/v1/endpoints/watchlists.py -f json -o /tmp/bandit_watchlists_audio_projection_task2.json` passed with `results 0`.
- `git diff --check` passed.

Task 3 completed:
- Added a shared adapter helper that copies Watchlists correlation metadata only when `workflow_metadata.source == "watchlist_audio_briefing"`.
- Tagged audio script artifacts with `source`, `watchlist_job_id`, `watchlist_run_id`, and `audio_request_id`.
- Tagged multi-voice per-speaker and final TTS artifacts with the same correlation fields.
- Added generic TTS `artifact_metadata` passthrough for fallback/final markers.
- Marked the Watchlists single-voice fallback TTS step as the final fallback artifact.

Task 3 verification:
- Red run: `source ../../.venv/bin/activate && python -m pytest tldw_Server_API/tests/Watchlists/test_audio_briefing_workflow.py::TestAudioBriefingWorkflowDefinition::test_workflow_def_marks_single_voice_fallback_artifact tldw_Server_API/tests/Workflows/adapters/test_content_adapters.py::TestAudioBriefingComposeAdapter::test_compose_registers_script_artifact tldw_Server_API/tests/Workflows/adapters/test_audio_adapters.py::test_tts_adapter_merges_watchlist_and_config_artifact_metadata tldw_Server_API/tests/Workflows/adapters/test_audio_adapters.py::TestMultiVoiceTTSAdapter::test_multi_voice_tts_registers_per_speaker_artifacts -q` failed for missing artifact metadata.
- Green run: same command passed, `4 passed`.
- `source ../../.venv/bin/activate && python -m pytest tldw_Server_API/tests/Workflows/adapters/test_audio_adapters.py tldw_Server_API/tests/Workflows/adapters/test_content_adapters.py -q` passed, `193 passed`.
- `source ../../.venv/bin/activate && python -m pytest tldw_Server_API/tests/Watchlists/test_audio_briefing_workflow.py -q` passed, `29 passed`.
- `source ../../.venv/bin/activate && python -m pytest tldw_Server_API/tests/Workflows/test_adapter_path_security.py::test_watchlist_artifact_metadata_filters_to_watchlists_source tldw_Server_API/tests/Workflows/test_adapter_path_security.py::test_watchlist_artifact_metadata_ignores_non_watchlists_metadata -q` passed, `2 passed`.
- `source ../../.venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Workflows/adapters/_common.py tldw_Server_API/app/core/Workflows/adapters/audio/_config.py tldw_Server_API/app/core/Workflows/adapters/audio/multi_voice_tts.py tldw_Server_API/app/core/Workflows/adapters/audio/tts.py tldw_Server_API/app/core/Workflows/adapters/content/audio_briefing.py tldw_Server_API/app/core/Watchlists/audio_briefing_workflow.py -f json -o /tmp/bandit_watchlists_audio_projection_task3.json` passed with `results 0`.
- `git diff --check` passed.

Task 4 completed:
- Added pure Watchlists audio projection helpers for status normalization, artifact download URLs, workflow metadata extraction, artifact summarization, and full graph construction.
- Added request-aware artifact selection so current `audio_request_id` artifacts win over older same-run artifacts.
- Kept mirrored artifact summaries free of raw `file://` URIs.
- Added metadata merge and stale-state helpers that preserve unrelated delivery/template/Chatbook fields.
- Added synchronous DB-facing helpers for mirrored run/output metadata, canonical output lookup, mirrored fallback reads, and matching Workflow run lookup.

Task 4 verification:
- Red run: `source ../../.venv/bin/activate && python -m pytest tldw_Server_API/tests/Watchlists/test_audio_artifact_projection.py -q` failed because the helper module did not exist.
- Green run: same command passed, `8 passed`.
- `source ../../.venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Watchlists/audio_artifact_projection.py -f json -o /tmp/bandit_watchlists_audio_projection_task4.json` passed with `results 0`.
- `git diff --check` passed.

Task 5 completed:
- Replaced `/runs/{run_id}/audio` inline Workflow artifact scanning/classification with the Watchlists audio projection helper.
- Resolved Workflows DB through `create_workflows_database(get_content_backend_instance())` instead of manually constructing a user SQLite path.
- Added `collections_db` dependency and lazy read-repair mirror writes into Watchlists run stats plus canonical output metadata.
- Preserved live Scheduler fallback when canonical Workflow artifacts are not available.
- Returned mirrored audio projection metadata when canonical lookup fails.
- Extended `WatchlistRunAudioResponse` with durable projection fields: `audio_request_id`, `workflow_run_id`, `schema_version`, `synced_at`, and `stale`.
- Ensured mirrored `metadata.audio` stores download URLs/artifact summaries without raw file URIs while preserving legacy `audio_uri` in endpoint responses.

Task 5 verification:
- Red run: focused Task 5 endpoint tests failed for missing `collections_db` support and mirrored fallback behavior.
- Green run: `source ../../.venv/bin/activate && python -m pytest tldw_Server_API/tests/Watchlists/test_audio_output_delivery.py::TestGetRunAudioEndpoint::test_canonical_workflow_audio_projection_is_mirrored tldw_Server_API/tests/Watchlists/test_audio_output_delivery.py::TestGetRunAudioEndpoint::test_workflows_lookup_failure_returns_mirrored_audio_metadata -q` passed, `2 passed`.
- `source ../../.venv/bin/activate && python -m pytest tldw_Server_API/tests/Watchlists/test_audio_output_delivery.py -q` passed, `23 passed`.
- `source ../../.venv/bin/activate && python -m pytest tldw_Server_API/tests/Watchlists/test_audio_artifact_projection.py tldw_Server_API/tests/Watchlists/test_audio_output_delivery.py tldw_Server_API/tests/Watchlists/test_watchlists_operator_recovery.py -q` passed, `43 passed`.
- `source ../../.venv/bin/activate && python -m pytest tldw_Server_API/tests/Watchlists/test_audio_briefing_workflow.py tldw_Server_API/tests/Watchlists/test_watchlists_pipeline.py tldw_Server_API/tests/Watchlists/test_watchlists_api.py -q` passed, `67 passed, 1 skipped`.
- `source ../../.venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/endpoints/watchlists.py tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py -f json -o /tmp/bandit_watchlists_audio_projection_task5.json` passed with `results 0`.
- `git diff --check` passed.

Task 6 completed:
- Updated audio retry to use the projection stale helper instead of hand-copying stale audio state.
- Retry now leaves a new active queued audio graph for the new request while moving the old completed graph to `previous_audio`.
- Added `collections_db` handling to `retry_run_audio(...)` and best-effort mirrored retry state into the canonical Watchlists output metadata.
- Preserved unrelated output metadata such as delivery status while clearing active old final artifacts.

Task 6 verification:
- Red run: `source ../../.venv/bin/activate && python -m pytest tldw_Server_API/tests/Watchlists/test_watchlists_operator_recovery.py::test_retry_run_audio_reuses_job_audio_config_without_rerunning_ingestion tldw_Server_API/tests/Watchlists/test_watchlists_operator_recovery.py::test_retry_run_audio_marks_output_audio_stale -q` failed for missing active retry graph and missing `collections_db` support.
- Green run: same focused command passed, `2 passed`.
- `source ../../.venv/bin/activate && python -m pytest tldw_Server_API/tests/Watchlists/test_audio_output_delivery.py::TestGetRunAudioEndpoint tldw_Server_API/tests/Watchlists/test_watchlists_operator_recovery.py -q` passed, `36 passed`.
- `source ../../.venv/bin/activate && python -m pytest tldw_Server_API/tests/Watchlists/test_audio_output_delivery.py tldw_Server_API/tests/Watchlists/test_watchlists_operator_recovery.py -q` passed, `36 passed`.
- `source ../../.venv/bin/activate && python -m pytest tldw_Server_API/tests/Watchlists/test_audio_artifact_projection.py tldw_Server_API/tests/Watchlists/test_audio_briefing_workflow.py tldw_Server_API/tests/Watchlists/test_watchlists_pipeline.py tldw_Server_API/tests/Watchlists/test_watchlists_api.py -q` passed, `75 passed, 1 skipped`.
- `source ../../.venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/endpoints/watchlists.py tldw_Server_API/app/core/Watchlists/audio_artifact_projection.py -f json -o /tmp/bandit_watchlists_audio_projection_task6.json` passed with `results 0`.
- `git diff --check` passed.

Task 7 completed:
- Extended frontend audio status summaries with durable request/workflow/schema/sync/stale/superseded fields.
- Added request-aware live status merging so newer queued live requests do not inherit older mirrored final artifacts.
- Rendered stale/superseded state in Output Preview and the full script/speaker/final artifact graph in Run Details.
- Kept artifact display path-safe by using download URLs and filename-only display labels.
- Added WebUI and extension locale keys for audio artifacts, stale/superseded, and artifact links.

Task 7 verification:
- Red run: `cd apps/packages/ui && bunx vitest run src/components/Option/Watchlists/OutputsTab/__tests__/OutputPreviewDrawer.audio.test.tsx src/components/Option/Watchlists/RunsTab/__tests__/RunDetailDrawer.stream-lifecycle.test.tsx` failed for missing stale label and missing Run Details artifact graph.
- Green run: `cd apps/packages/ui && bunx vitest run src/components/Option/Watchlists/OutputsTab/__tests__/outputMetadata.test.ts src/components/Option/Watchlists/OutputsTab/__tests__/OutputPreviewDrawer.audio.test.tsx src/components/Option/Watchlists/RunsTab/__tests__/RunDetailDrawer.stream-lifecycle.test.tsx` passed, `43 passed`.
- `git diff --check` passed.
- `node -e "for (const file of ['apps/packages/ui/src/assets/locale/en/watchlists.json','apps/packages/ui/src/public/_locales/en/watchlists.json']) { JSON.parse(require('fs').readFileSync(file, 'utf8')); console.log(file + ' ok') }"` passed.
- Bandit not applicable to Task 7 because the touched implementation scope is TypeScript/locale/docs only.

Task 8 deferred:
- Verified `watchlist_run` is registered on the Scheduler `watchlists` queue and recurring Watchlist-backed schedules route there.
- Found no startup or enqueue path that guarantees a `watchlists` queue worker; the generic Scheduler starts `default` workers, while audio generation explicitly scales the `workflows` queue.
- Did not add a proactive projection task in this PR because an unserved `watchlists` task would be unreliable, and placing the projection task on `workflows` would depend on retry-as-poll behavior after async Workflow submission.
- Lazy read-repair plus mirrored fallback remains the durable MVP reliability path.

Task 8 verification:
- `rg -n "watchlist.*@task|queue=.*watch|watchlist_run|scale_workers|@task" tldw_Server_API/app/core tldw_Server_API/app/services`
- `rg -n "scale_workers\([^\n]*(watchlists|workflows)|watchlists_queue|queue_name=\"watchlists\"|\"watchlists\"\)" tldw_Server_API/app tldw_Server_API/tests`
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
