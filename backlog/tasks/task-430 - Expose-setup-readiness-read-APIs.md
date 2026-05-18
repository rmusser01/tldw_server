---
id: TASK-430
title: Expose setup readiness read APIs
status: Done
assignee: []
created_date: ''
updated_date: 2026-05-18 23:30
labels:
- implementation
- setup
- backend
- api
dependencies: []
documentation:
- Docs/superpowers/specs/2026-05-18-first-time-model-readiness-setup-design.md
- Docs/superpowers/plans/2026-05-18-first-time-readiness-setup-implementation-plan.md
modified_files:
- Docs/superpowers/plans/2026-05-18-first-time-readiness-setup-implementation-plan.md
- tldw_Server_API/app/api/v1/endpoints/setup.py
- tldw_Server_API/tests/Setup/test_setup_readiness_api.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the third backend slice from Docs/superpowers/plans/2026-05-18-first-time-readiness-setup-implementation-plan.md: read-only setup readiness profiles, preview, and status endpoints backed by the existing local first-run setup access guard and the profile/preview service contracts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Local first-run users can GET /api/v1/setup/readiness/profiles and receive canonical chat, embeddings/RAG, and speech lanes backed by the profile builder.
- [x] #2 Local first-run users can GET /api/v1/setup/readiness/status and receive overlay state separately from lane statuses.
- [x] #3 Local first-run users can POST /api/v1/setup/readiness/preview and receive the sanitized read-only preview contract without config writes or secret echoing.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added first-run read-only setup readiness profiles, status, and preview routes under the existing /api/v1/setup router with require_local_setup_access and openapi_extra security clearing. Added a shared endpoint helper that builds profile/status payloads from setup status, config snapshot, and audio recommendations, with an overlays alias for UI consumers. Added API tests for profiles, status overlay separation, and preview secret redaction/no-write behavior. Updated Task 3 in the implementation plan and removed the schema file from this slice because the schema models landed in Task 2.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented first-run read-only setup readiness API routes for profiles, status, and preview. Verification: bounded TDD red run failed with 404s before routes existed; final `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Setup/test_setup_readiness_profiles.py tldw_Server_API/tests/Setup/test_setup_readiness_preview.py tldw_Server_API/tests/Setup/test_setup_readiness_api.py -q --timeout=30` passed with 12 tests; `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/api/v1/endpoints/setup.py tldw_Server_API/app/api/v1/schemas/setup_schemas.py tldw_Server_API/app/core/Setup/readiness_profiles.py tldw_Server_API/app/core/Setup/readiness_service.py tldw_Server_API/app/core/Setup/readiness_models.py -f json -o /tmp/bandit_first_time_readiness_api.json` completed with zero findings. Test harness note: an initial red run using `with TestClient(app)` hit the inherited full-app shutdown hang, so the new tests avoid the lifespan context manager and use `--timeout=30`.
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
