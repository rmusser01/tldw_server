---
id: TASK-12732
title: Implement Audio Studio jobs and providers
status: Done
labels:
- audio
- jobs
priority: high
documentation:
- Docs/superpowers/plans/2026-06-23-audio-studio-mvp-implementation-plan.md
- Docs/superpowers/specs/2026-06-23-audio-studio-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Audio Studio provider adapters, external endpoint allowlisting/secret handling, Jobs enqueue helpers, worker handler/startup registration, and generation endpoints from the accepted MVP plan.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Provider adapter registry supports speech generation and ACE-Step external HTTP generation with fail-closed configuration.
- [x] #2 External endpoint allowlisting, redirect validation, and secret redaction are covered by tests.
- [x] #3 Audio Studio generation jobs are idempotent, revision-pinned, worker-processed, and exposed through API endpoints.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Follow Stage 2 tasks 2.1 through 2.5 in Docs/superpowers/plans/2026-06-23-audio-studio-mvp-implementation-plan.md after TASK-2348 is complete. Use TDD and run the listed pytest commands plus Bandit on touched backend code.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-06-23 recovery: inspected the partial TASK-2349 implementation against the Stage 2 requirements. No additional code fixes were needed after focused verification. Removed generated __pycache__ directories from the Audio Studio app/test scope before staging.
2026-06-23 follow-up: fixed idempotent generation replays so existing Jobs rows with known terminal statuses, including completed, return the existing generation job row instead of raising a client error. Added focused unit coverage for completed replay preserving a single Jobs row and generation index record. Verification: focused jobs/API pytest 7 passed, broader TASK-2349 suite 33 passed, Bandit on jobs.py 0 findings, scoped git diff check clean.
2026-06-23 code-quality follow-up: hardened ACE-Step redirects to reject cross-origin redirects when bearer auth is present, expanded Audio Studio payload sanitization for credential and URL-bearing variants, updated generation rows for skipped stale revisions and provider failures, and scoped handler-created CollectionsDatabase instances with context managers. Verification: affected unit group 15 passed, full TASK-2349 suite 37 passed, Bandit on touched Audio Studio app paths 0 findings, scoped git diff check clean.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Audio Studio Stage 2 provider and Jobs support: secret-free provider registry with TTS speech and fail-closed ACE-Step HTTP music adapter, allowlisted external endpoint and redirect validation, sanitized idempotent generation job enqueue helpers, generation worker artifact persistence, provider/generation/artifact API endpoints, and startup poller registration. Verification: .venv/bin/python -m pytest tldw_Server_API/tests/Audio_Studio/unit/test_audio_studio_provider_registry.py tldw_Server_API/tests/Audio_Studio/unit/test_audio_studio_external_security.py tldw_Server_API/tests/Audio_Studio/unit/test_audio_studio_jobs.py tldw_Server_API/tests/Audio_Studio/unit/test_audio_studio_jobs_worker.py tldw_Server_API/tests/Audio_Studio/integration/test_audio_studio_generation_api.py tldw_Server_API/tests/Services/test_startup_content_jobs_pollers.py -v (32 passed, 7 warnings); .venv/bin/python -m pytest tldw_Server_API/tests/Audio_Studio/unit/test_audio_studio_schemas.py tldw_Server_API/tests/Audio_Studio/unit/test_audio_studio_collections_db.py tldw_Server_API/tests/Audio_Studio/integration/test_audio_studio_projects_api.py -v (65 passed, 7 warnings); .venv/bin/python -m bandit -r touched app paths -f json -o /tmp/bandit_audio_studio_jobs_providers.json (0 findings). Known skips/blockers: Stage 3 render/export/migration services intentionally deferred; current helpers enqueue generic jobs or return deferred handler results.
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
