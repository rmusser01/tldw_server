---
id: TASK-235
title: Add VN policy and generation profiles API
status: Done
assignee: []
created_date: '2026-05-10 04:53'
updated_date: '2026-05-10 05:43'
labels:
  - vn
  - api
  - backend
  - policy
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
Implement Task 3 of the VN platform API implementation plan. Scope: add backend-owned VN policy profiles and generation profiles under /api/v1/vn/vn-policy, including policy/generation profile persistence or config-backed definitions, immutable per-user snapshots for user-owned VN resources, safety metadata evaluation, admin-only mutation APIs, route/OpenAPI registration, and focused tests. Keep the feature API-first and avoid WebUI-only policy coupling.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Policy and generation profile service exposes built-in local_default and strict_hosted behavior plus validated admin-created profile definitions.
- [x] #2 Snapshot creation stores immutable effective policy/generation settings for user-owned VN resources and remains stable after profile updates.
- [x] #3 Safety metadata evaluation implements the reviewed missing unknown ambiguous conflicting and imported-untrusted behavior for local_default and strict_hosted profiles.
- [x] #4 VN policy API exposes evaluate list read and admin mutation endpoints under /api/v1/vn/vn-policy with stable VN error details pagination and owner scoping.
- [x] #5 Route registration and OpenAPI tests include /api/v1/vn/vn-policy/evaluate /profiles and /generation-profiles.
- [x] #6 Focused VN_Policy tests git diff checks and Bandit on touched production Python paths are recorded before completion.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented Task 3 from Docs/superpowers/plans/2026-05-10-vn-platform-api-implementation-plan.md. Added global AuthNZ-backed VN policy/generation profile definitions, per-user immutable profile snapshots, safety metadata evaluation, VN policy API schemas/endpoints, router/capability/OpenAPI registration, and focused tests. Addressed review findings by allowing minor metadata for non-mature content, wrapping profile definition/version writes in one transaction, avoiding user ChaCha policy definition table creation for read-only list/read/evaluate paths, failing closed for unresolved custom runtime policy profiles, and honoring acknowledgement_required_for_warnings.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the backend-owned VN policy and generation profiles API for Task 3. The slice adds global AuthNZ-backed policy/generation profile definitions with built-in local_default, strict_hosted, and story_default profiles; immutable per-user profile snapshots; safety metadata evaluation; admin-only profile mutation endpoints; stable VN error details; offset pagination; canonical /api/v1/vn/vn-policy route registration; capability/OpenAPI coverage; and API docs. The implementation keeps profile definitions global for custom frontend/API reuse while keeping resource snapshots in the user's ChaChaNotes database so resource history preserves the exact effective policy/generation settings. Verification recorded: affected VN/API suite passed with 292 passed, compileall passed for touched production modules, git diff --check passed, and Bandit returned zero findings on touched production Python files.
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
