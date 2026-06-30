---
id: TASK-500
title: Fix setup audio release-gate test failures
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-31 16:56'
labels: []
dependencies: []
references:
  - TASK-499
  - >-
    Docs/superpowers/plans/2026-05-31-first-time-solo-user-onboarding-implementation-plan.md#final-verification-checklist
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve the broader tldw_Server_API/tests/Setup failures observed while closing the unified first-run onboarding release gate. Scope is limited to setup audio health logging, audio pack manifest validation/import path handling, and HuggingFace installer download test compatibility.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Audio health/setup audio release-gate failures are resolved without leaking internal diagnostics.
- [x] #2 Audio pack import/export endpoints support both managed pack names and external pack paths where applicable.
- [x] #3 Focused and broad setup/config tests pass after fixes.
- [x] #4 Bandit and whitespace checks are clean for touched backend code.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed setup audio release-gate failures found while closing TASK-499. Changes included sanitized TTS health exception logging, external-path support for audio pack manifest reads/imports, dict-compatible machine profile projection, sanitized-but-specific setup validation details, installer download fake compatibility with revision kwargs, and TestClient teardown-safe API fixtures.

Verification run: source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Setup tldw_Server_API/tests/integration/test_unified_first_run_setup_api.py tldw_Server_API/tests/Config/test_config_providers_endpoints.py -v -> 324 passed, 4 warnings.
Bandit run: source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/endpoints/setup.py tldw_Server_API/app/api/v1/endpoints/audio/audio_health.py tldw_Server_API/app/core/Setup/audio_pack_service.py -f json -o /tmp/bandit_setup_audio_release_gate.json -> 0 findings.
git diff --check -> clean.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Resolved the setup audio release-gate failures blocking TASK-499 closeout. Broad setup/config verification now passes, Bandit reports no findings on touched backend setup/audio code, and whitespace validation is clean.
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
