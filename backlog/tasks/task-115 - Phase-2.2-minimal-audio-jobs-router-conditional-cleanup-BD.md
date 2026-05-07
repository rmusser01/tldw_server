---
id: TASK-115
title: Phase 2.2 minimal audio jobs router conditional cleanup BD
status: Done
assignee: []
created_date: '2026-05-07 15:02'
updated_date: '2026-05-07 19:30'
labels:
  - phase-2.2
  - router-groups
  - minimal
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1116'
  - 'https://github.com/rmusser01/tldw_server/pull/1363'
  - 'https://github.com/rmusser01/tldw_server/pull/1366'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Move the gated minimal-test audio jobs router factory onto the shared lazy optional-router registration path so it uses the same missing-target skip semantics and diagnostics as other optional minimal router specs while preserving the existing /api/v1/audio metadata and MINIMAL_TEST_INCLUDE_AUDIO_JOBS opt-in behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Minimal audio-jobs route metadata is represented through the shared optional router spec path while preserving prefix /api/v1/audio, audio-jobs tag, route_key audio-jobs, and existing opt-in gate.
- [x] #2 Focused contract coverage proves audio_jobs module import and router attribute lookup stay deferred until router resolution.
- [x] #3 Focused contract coverage verifies missing audio_jobs target skips while runtime import defects propagate.
- [x] #4 Existing router group, main router, and OpenAPI contracts still pass for the touched scope.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add RED contract tests for the gated minimal audio_jobs router using the shared optional-router expectations.
2. Replace the hand-written audio_jobs router factory with ImportedRouterSpec while preserving the existing opt-in gate.
3. Run focused, router-group, main-router, OpenAPI, Bandit, and diff hygiene verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
RED: python -m pytest tldw_Server_API/tests/Services/test_router_groups_contract.py -k "minimal_optional_router_specs and audio_jobs" -q failed with the expected old-factory behavior: 6 failed, 2 passed. Failures showed missing shared spec metadata, bypassed ImportedRouterSpec import hooks, and broad skip diagnostics.

GREEN/validation: focused selector passed 8 passed; router_groups_contract.py passed 159 passed; test_main_router_contract.py passed 6 passed; test_openapi_contracts.py passed 69 passed; Bandit on tldw_Server_API/app/api/v1/router_groups/minimal.py reported 0 results/errors; git diff --check passed.

No documentation update was required for this internal router-registration cleanup. No skips or blockers remain.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Moved the gated minimal audio_jobs router from a hand-written lazy factory to the shared ImportedRouterSpec registration path. This preserves the MINIMAL_TEST_INCLUDE_AUDIO_JOBS opt-in gate, /api/v1/audio prefix, audio-jobs tag, route_key=audio-jobs, and uses the shared missing-module/missing-attribute skip semantics while runtime import defects propagate. Added focused contract coverage for lazy resolution, missing optional target skips, missing router attribute skips, and runtime import failure propagation.
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
