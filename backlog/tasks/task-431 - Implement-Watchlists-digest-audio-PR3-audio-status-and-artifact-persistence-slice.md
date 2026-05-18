---
id: TASK-431
title: Implement Watchlists digest audio PR3 audio status and artifact persistence
  slice
status: Done
priority: High
modified_files:
- apps/packages/ui/src/components/Option/Watchlists/RunsTab/RunDetailDrawer.tsx
- apps/packages/ui/src/components/Option/Watchlists/RunsTab/__tests__/RunDetailDrawer.stream-lifecycle.test.tsx
- apps/packages/ui/src/components/Option/Watchlists/OutputsTab/OutputPreviewDrawer.tsx
- apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/OutputPreviewDrawer.audio.test.tsx
- apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/outputMetadata.test.ts
- apps/packages/ui/src/components/Option/Watchlists/OutputsTab/outputMetadata.ts
- tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py
- tldw_Server_API/app/api/v1/endpoints/watchlists.py
- tldw_Server_API/app/core/Watchlists/audio_briefing_workflow.py
- tldw_Server_API/app/core/Workflows/adapters/content/_config.py
- tldw_Server_API/app/core/Workflows/adapters/content/audio_briefing.py
- tldw_Server_API/app/core/Workflows/adapters/audio/multi_voice_tts.py
- tldw_Server_API/tests/Watchlists/test_audio_briefing_workflow.py
- tldw_Server_API/tests/Watchlists/test_audio_output_delivery.py
- tldw_Server_API/tests/Watchlists/test_watchlists_api.py
- tldw_Server_API/tests/Workflows/adapters/test_content_adapters.py
- tldw_Server_API/tests/Workflows/adapters/test_audio_adapters.py
- Docs/superpowers/plans/2026-05-18-watchlists-digest-audio-briefing-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement PR3 from Docs/superpowers/plans/2026-05-18-watchlists-digest-audio-briefing-implementation-plan.md: Task 6 run audio status in activity/reports and Task 7 backend audio artifact persistence. Keep core workflow inside /watchlists, preserve existing watchlists/OSINT/CTI flows, and avoid creating a parallel podcast job system.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Task 6 frontend tests are written red first, then pass after implementation.
- [x] #2 Task 7 backend tests are written red first, then pass after implementation.
- [x] #3 /watchlists run and output surfaces distinguish audio not requested, pending/running, final, failed, fallback, and unknown states.
- [x] #4 GET /api/v1/watchlists/runs/{run_id}/audio returns script artifact, per-speaker artifacts, final artifact, fallback reason, status, and download URL when available.
- [x] #5 Structured audio_cast is accepted while preserving voice_map compatibility.
- [x] #6 Focused frontend/backend tests, git diff --check, and Bandit on touched backend production files pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-18-watchlists-digest-audio-briefing-implementation-plan.md

PR3 implemented Task 6 and Task 7. Verification recorded in final summary before task completion.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Watchlists digest audio PR3 and review fixes. The PR now surfaces run audio status and output artifact graphs, persists script/per-speaker/final audio artifacts, accepts structured audio_cast while preserving voice_map compatibility, and exposes /api/v1/watchlists/runs/{run_id}/audio. Review fixes addressed Qodo findings by adding audio schema docstrings, isolating db.list_items through run_in_threadpool in the async audio trigger, aligning digit-bearing speaker markers with the parser, and preventing raw file:// or server filesystem artifact paths from rendering or becoming links in /watchlists audio UI. CodeRabbit follow-up fixes enforced audio_cast speaker-count/identity consistency, removed the arbitrary 1,000 workflow-run scan cap, changed multi-voice TTS artifact registration from per-section to per-speaker aggregation, checked the completed Backlog criteria, and corrected the async pending-state UI test.

PR: https://github.com/rmusser01/tldw_server/pull/1844

Verification: bunx vitest run src/components/Option/Watchlists/OutputsTab/__tests__/outputMetadata.test.ts src/components/Option/Watchlists/OutputsTab/__tests__/OutputPreviewDrawer.audio.test.tsx src/components/Option/Watchlists/RunsTab/__tests__/RunDetailDrawer.stream-lifecycle.test.tsx --maxWorkers=1 --no-file-parallelism (26 passed); bun run test:watchlists:a11y (82 passed); bun run test:watchlists:typecheck (3 passed); python -m pytest tldw_Server_API/tests/Watchlists/test_audio_briefing_workflow.py tldw_Server_API/tests/Watchlists/test_audio_output_delivery.py tldw_Server_API/tests/Watchlists/test_watchlists_api.py::test_outputs_generate_audio_payload_triggers_workflow_and_updates_run_stats tldw_Server_API/tests/Workflows/adapters/test_content_adapters.py::TestAudioBriefingComposeAdapter tldw_Server_API/tests/Workflows/adapters/test_audio_adapters.py::TestMultiVoiceTTSAdapter -q (56 passed); black --check touched Python files passed; git diff --check passed; Bandit touched backend production files passed with 0 findings. Full UI tsc was not rerun for this review fix because prior PR verification already identified unrelated repo-wide TypeScript debt outside the touched Watchlists slice.
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
