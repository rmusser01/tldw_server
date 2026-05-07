---
id: TASK-117
title: Phase 2.2 minimal audio router conditional cleanup BE
status: Done
assignee: []
created_date: '2026-05-07 19:42'
updated_date: '2026-05-07 19:58'
labels:
  - phase-2.2
  - router-groups
  - minimal
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1116'
  - 'https://github.com/rmusser01/tldw_server/pull/1366'
  - 'https://github.com/rmusser01/tldw_server/pull/1367'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Move the gated minimal-test audio and audio websocket router factories onto the shared lazy optional-router registration path so they use the same missing-target skip semantics and diagnostics as other optional minimal router specs while preserving the existing /api/v1/audio metadata and MINIMAL_TEST_INCLUDE_AUDIO opt-in behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Minimal audio route metadata is represented through the shared optional router spec path while preserving prefix /api/v1/audio, audio tag, route_key audio, and existing opt-in gate.
- [x] #2 Minimal audio websocket route metadata is represented through the shared optional router spec path while preserving prefix /api/v1/audio, audio-ws tag, route_key audio-websocket, and existing opt-in gate.
- [x] #3 Focused contract coverage proves audio module import and router/ws_router attribute lookup stay deferred until router resolution.
- [x] #4 Focused contract coverage verifies missing audio target skips while runtime import defects propagate.
- [x] #5 Existing router group, main router, and OpenAPI contracts still pass for the touched scope.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add RED contract tests for the gated minimal audio and audio websocket routers using shared optional-router expectations.
2. Replace the hand-written audio/router and audio/ws_router factories with ImportedRouterSpec while preserving the existing opt-in gate.
3. Run focused, router-group, main-router, OpenAPI, Bandit, and diff hygiene verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
RED verification: python -m pytest tldw_Server_API/tests/Services/test_router_groups_contract.py -k "minimal_optional_router_specs and audio_router" -q failed before implementation with 6 failed, 159 deselected, and 6 warnings, showing the old hand-written factory behavior bypassed shared ImportedRouterSpec semantics.

GREEN verification: focused selector passed 6 passed, 159 deselected, 6 warnings. Full router group contracts passed 165 passed, 30 warnings. Main router contracts passed 6 passed, 5 warnings. OpenAPI contracts passed 69 passed, 24 warnings. Bandit on tldw_Server_API/app/api/v1/router_groups/minimal.py reported 0 results and 0 errors. git diff --check passed.

No documentation change required for this internal router-registration cleanup. No blockers or known skips beyond existing test-suite warnings.

PR review follow-up: addressed Gemini comments by sharing the audio import path through a local audio_module_path variable and making the lazy attribute tracking test use an explicit ModuleType subclass. Verified the ModuleType instance __getattr__ behavior separately before changing the test, then kept the clearer subclass form. Post-review validation: focused selector passed 6 passed, 159 deselected, 6 warnings; router group contracts passed 165 passed, 30 warnings; main router contracts passed 6 passed, 5 warnings; OpenAPI contracts passed 69 passed, 24 warnings; Bandit reported 0 results and 0 errors; git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Moved the gated minimal audio and audio websocket routers from hand-written lazy factories to the shared ImportedRouterSpec registration path. This preserves the MINIMAL_TEST_INCLUDE_AUDIO opt-in gate, /api/v1/audio prefixes, audio/audio-ws tags, route_key=audio and route_key=audio-websocket, and uses shared missing-module/missing-attribute skip semantics while runtime import defects propagate. Added focused contract coverage for lazy resolution, missing optional target skips, missing router/ws_router attribute skips, and runtime import failure propagation. Verification covered the focused selector, full router group contracts, main router contracts, OpenAPI contracts, Bandit on the touched production file, and git diff --check.
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
