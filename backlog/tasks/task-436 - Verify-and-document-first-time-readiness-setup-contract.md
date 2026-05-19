---
id: TASK-436
title: Verify and document first-time readiness setup contract
status: Done
labels:
- verification
- setup
- docs
- frontend
- backend
documentation:
- Docs/superpowers/specs/2026-05-18-first-time-model-readiness-setup-design.md
- Docs/superpowers/plans/2026-05-18-first-time-readiness-setup-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Run the final verification slice for the first-time readiness setup work, update setup developer documentation with the native WebUI readiness flow, and record known verification caveats.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Setup developer documentation describes the native WebUI readiness screen, first-run/admin endpoint split, backend `/setup` fallback, and explicit `Provision now` behavior.
- [x] Backend setup/readiness tests are run and recorded.
- [x] Focused WebUI setup tests and OpenAPI client path verification are run and recorded.
- [x] Bandit and whitespace checks are run and recorded.
- [x] Known verification caveats are documented.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Update setup developer docs to describe the native WebUI readiness screen and first-run/admin endpoint split.
2. Run focused backend setup readiness tests, focused frontend setup tests, OpenAPI path verification, Bandit on touched backend setup code, and diff whitespace checks.
3. Record inherited or skipped verification caveats in Backlog and plan.
4. Commit final docs/verification bookkeeping.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Updated `Docs/Code_Documentation/Setup_UI_Developer_Guide.md` with the native WebUI readiness flow and endpoint contract.
- Aligned the setup endpoint's local audio pack request models with the schema-level legacy `pack_path` compatibility so `/audio/packs/import` resolves `pack_name` consistently.
- Stabilized the setup audio lifecycle API tests by avoiding the hanging `TestClient` lifespan context in this slice, making the async provision mock awaitable, and aligning stale 400-detail assertions with the endpoint's generic setup error contract.
- Verification:
  - `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Setup/test_setup_readiness_profiles.py tldw_Server_API/tests/Setup/test_setup_readiness_preview.py tldw_Server_API/tests/Setup/test_setup_readiness_store.py tldw_Server_API/tests/Setup/test_setup_readiness_api.py tldw_Server_API/tests/Setup/test_setup_audio_installer_lifecycle_api.py tldw_Server_API/tests/Setup/test_setup_manager_masking.py -q --timeout=30` -> 42 passed.
  - `bunx vitest run src/components/Option/Setup/__tests__/ReadinessSetupScreen.test.tsx src/components/Option/Setup/hooks/__tests__/useSetupReadiness.test.tsx src/components/Option/Setup/__tests__/AudioInstallerPanel.test.tsx src/routes/__tests__/option-setup-readiness.test.tsx` -> 20 passed.
  - `bun run verify:openapi` -> 269 ClientPath entries verified; existing 10 reviewed OSS exception paths allowed.
  - `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/api/v1/endpoints/setup.py tldw_Server_API/app/core/Setup/readiness_models.py tldw_Server_API/app/core/Setup/readiness_profiles.py tldw_Server_API/app/core/Setup/readiness_service.py tldw_Server_API/app/core/Setup/readiness_store.py -f json -o /tmp/bandit_first_time_readiness_setup.json` -> 0 findings.
  - `git diff --check` -> passed.
- Caveat: repo-wide `bunx tsc --noEmit --pretty false` still fails on inherited TypeScript debt outside the setup readiness slice; observed output did not reference the new setup readiness client, hook, screen, or route test files.
- Browser QA was not run in this final pass; route/component behavior is covered by focused Vitest tests and should still get manual end-to-end coverage with a configured first-run backend before release.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Final verification and documentation are recorded for the first-time readiness setup contract. Backend setup/readiness tests, focused WebUI setup tests, OpenAPI verification, Bandit, and whitespace checks passed. The remaining known caveat is inherited repo-wide UI TypeScript debt outside this setup readiness slice.
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
