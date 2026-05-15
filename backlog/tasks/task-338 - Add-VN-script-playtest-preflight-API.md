---
id: TASK-338
title: Add VN script playtest preflight API
status: Done
assignee: []
created_date: '2026-05-14 07:15'
updated_date: '2026-05-14 07:15'
labels:
  - vn
  - api
  - scripts
  - playtest
dependencies: []
references:
  - Docs/superpowers/plans/2026-05-14-vn-script-playtest-preflight-implementation-plan.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a backend-owned VN script playtest/preflight API so bundled and custom frontends can dry-run draft or published scripts before starting runtime sessions. The API should analyze deterministic script execution paths with server validation, manifest/profile context, and existing scripted-story interpreter semantics, without creating VN Play sessions, calling models, or mutating runtime state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 API exposes draft and published-version script playtest/preflight endpoints under the existing VN scripts API without creating sessions or calling models.
- [x] #2 Playtest responses report runtime readiness, deterministic traversal summary, visited labels, choice boundaries, generation boundaries, endings, warnings, errors, truncation, and max-step/path limits in a custom-frontend friendly schema.
- [x] #3 Preflight reuses backend-owned script validation, manifest/profile/audio context, and existing scripted-story interpreter semantics instead of duplicating runtime rules in frontend code.
- [x] #4 Playtest detects and reports loops or max-step truncation, invalid/unreachable choice targets, generation boundaries, missing approved visual/audio refs, and publish/runtime blockers with stable diagnostic codes.
- [x] #5 Focused backend tests cover draft playtest, version playtest, branching choices, loop truncation, generation boundary reporting, missing asset diagnostics, permissions/not-found behavior, and no runtime session mutation.
- [x] #6 Relevant VN API documentation describes endpoint contract, non-goals, diagnostic shape, and custom frontend usage.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implementation plan: Docs/superpowers/plans/2026-05-14-vn-script-playtest-preflight-implementation-plan.md.

Implemented backend-owned VN script playtest/preflight API in branch worktree `.worktrees/vn-script-playtest-preflight`. Added shared VN Play script runtime helpers, pure VN Scripts playtest analyzer, draft/version service methods, public API schemas/endpoints, capability flag, docs, and focused tests.

Verification:
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Scripts/test_vn_script_playtest.py tldw_Server_API/tests/VN_Scripts/test_vn_scripts_api.py tldw_Server_API/tests/VN_Play/test_vn_play_scripted_generation_runtime.py tldw_Server_API/tests/VN_Platform/test_vn_capabilities_api.py tldw_Server_API/tests/Services/test_openapi_contracts.py` -> 139 passed, 27 warnings.
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_api.py -q` -> 48 passed, 8 warnings.
- `git diff --check` -> passed.
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/core/VN_Play/script_runtime.py tldw_Server_API/app/core/VN_Play/errors.py tldw_Server_API/app/core/VN_Scripts/playtest.py tldw_Server_API/app/core/VN_Scripts/service.py tldw_Server_API/app/api/v1/endpoints/vn_scripts.py -f json -o /tmp/bandit_vn_script_playtest_preflight.json` -> 0 results.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added backend-owned VN script playtest/preflight support for draft and published script versions. The implementation extracts shared scripted-story runtime helpers so playtest and VN Play use the same deterministic semantics, adds a pure analyzer that reports path traversal, choice/generation boundaries, endings, validation diagnostics, truncation, and readiness without mutating runtime state or calling models, exposes draft/version API endpoints and capability discovery, and documents the API for custom frontends. Focused backend, OpenAPI, capability, runtime-regression, diff, and Bandit checks passed.
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
