---
id: TASK-233
title: Implement VN platform API shell
status: Done
assignee: []
created_date: '2026-05-10 03:07'
updated_date: '2026-05-10 04:18'
labels:
  - vn
  - api
  - backend
  - platform
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1391'
  - 'https://github.com/rmusser01/tldw_server/issues/1486'
documentation:
  - Docs/superpowers/specs/2026-05-10-vn-platform-api-design.md
  - Docs/superpowers/plans/2026-05-10-vn-platform-api-implementation-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the first slice of the VN platform API plan: create the backend-owned `/api/v1/vn` namespace shell, canonical `vn-*` route registration, shared VN error/idempotency primitives, and the `vn-capabilities` endpoint. This task follows Docs/superpowers/specs/2026-05-10-vn-platform-api-design.md and Docs/superpowers/plans/2026-05-10-vn-platform-api-implementation-plan.md, Task 1 only.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Expose GET /api/v1/vn/vn-capabilities with resources for assets, scripts, play, policy, and audio based on registered routes/configured modules.
- [x] #2 Register VN assets and VN play only under canonical `/api/v1/vn/vn-*` paths, with route/OpenAPI tests confirming old top-level `/api/v1/vn-assets` and `/api/v1/vn-play` paths are absent.
- [x] #3 Add shared VN error and idempotency helpers with focused unit coverage.
- [x] #4 Update existing frontend VN API constants/tests only where needed for the route migration.
- [x] #5 Run focused backend tests, relevant frontend API tests, git diff checks, and Bandit on touched production Python paths before completion.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Baseline before implementation: /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Assets tldw_Server_API/tests/VN_Play tldw_Server_API/tests/Services/test_router_groups_contract.py tldw_Server_API/tests/Services/test_openapi_contracts.py -q produced 567 passed, 13 failed, 33 warnings. Failures are existing VN prompt-preview/generation tests that attempted to fetch cl100k_base.tiktoken from openaipublic.blob.core.windows.net under restricted network; no source edits had been made yet.

Implementation complete for Task 1. Added VN_Platform shared error/idempotency/capabilities helpers, registered `vn-capabilities`/`vn-assets`/`vn-play` under `/api/v1/vn`, updated VN asset manifest content URLs, and moved frontend VN API clients/tests to `/vn/vn-*` client paths.
Verification: backend focused pytest command passed with 253 passed and 33 warnings; frontend focused Vitest command passed with 2 files and 7 tests; git diff --check passed; Bandit on touched production Python paths produced 0 results in /tmp/bandit_vn_platform_api_shell.json.
Frontend dependency note: plain bun install hung after dependency extraction twice; bun install --ignore-scripts completed and reported no dependency changes before running focused Vitest.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the first VN platform API shell slice: canonical `/api/v1/vn/vn-*` route registration, `GET /api/v1/vn/vn-capabilities`, shared VN error/idempotency helpers, updated manifest URLs, and frontend client/test route migration. Focused backend tests, focused frontend API tests, whitespace check, and Bandit all passed for the touched scope. Baseline broad VN tests still have pre-existing network-dependent tiktoken fetch failures under restricted network.
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
