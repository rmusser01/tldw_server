---
id: TASK-219
title: Migrate VN assets to platform API and harden safety
status: Done
assignee: []
created_date: '2026-05-10 04:23'
updated_date: '2026-05-10 04:50'
labels:
  - vn
  - api
  - backend
  - assets
  - security
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1391'
  - 'https://github.com/rmusser01/tldw_server/issues/1486'
documentation:
  - Docs/superpowers/specs/2026-05-10-vn-platform-api-design.md
  - Docs/superpowers/plans/2026-05-10-vn-platform-api-implementation-plan.md
  - Docs/API-related/VN_ASSET_PACKS_API.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 2 of the VN platform API implementation plan. Scope: migrate and verify VN asset endpoints under canonical /api/v1/vn/vn-assets, add/reuse durable idempotency behavior for asset mutators, harden item preview/content validation, add cleanup blocker behavior, and update VN asset API documentation. This task follows the reviewed VN platform API spec and should stay API/backend-owned except for test/client path updates required by route migration.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 VN asset API tests use canonical /api/v1/vn/vn-assets paths and assert old /api/v1/vn-assets paths are absent for the target router registration.
- [x] #2 Asset mutators covered in this slice have durable same-key same-payload replay and same-key different-payload conflict behavior with stable VN error details.
- [x] #3 VN item preview and content endpoints deny cross-owner access invalid generated-file provenance disallowed media types and policy-blocked item access.
- [x] #4 Asset cleanup dry-run and execution report blockers and skip blocked generated files through a pluggable blocker provider.
- [x] #5 VN asset API documentation reflects canonical paths idempotency behavior content validation and cleanup blocker semantics.
- [x] #6 Focused VN asset tests git diff checks and Bandit on touched production Python paths are recorded before completion.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented canonical VN asset route tests, durable idempotency records for VN asset work/file mutators, preview/content validation hardening, cleanup blocker reporting, docs updates, and tiktoken fallback hardening for offline test environments. Verification: VN_Assets 232 passed, VN_Platform 8 passed, git diff --check passed, Bandit touched production files results 0 at /tmp/bandit_vn_assets_task2.json.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Task 2 VN asset API migration and safety hardening: canonical /api/v1/vn/vn-assets route coverage, durable VN asset idempotency records, preview/content access validation, cleanup blocker reporting, portability/upload idempotency coverage, and offline tiktoken fallback hardening. Verification: /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Assets tldw_Server_API/tests/VN_Platform -q => 240 passed; git diff --check => passed; Bandit touched production files => results 0 at /tmp/bandit_vn_assets_task2_final.json.
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
