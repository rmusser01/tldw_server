---
id: TASK-222
title: Add VN scripts authoring API
status: Done
assignee: []
created_date: '2026-05-10 05:46'
updated_date: '2026-05-10 06:56'
labels:
  - vn
  - api
  - backend
  - scripts
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
Implement Task 4 of the VN platform API implementation plan. Scope: add backend-owned VN scripts draft, validation, diagnostics, publish, version, manifest snapshot, and version policy-evaluate APIs under /api/v1/vn/vn-scripts. Keep script source as structured JSON in V1, store script metadata/drafts/versions/manifests in the owning user's ChaChaNotes database, snapshot effective asset manifest and policy/generation profiles at publish, enforce optimistic draft revisions and publish idempotency, and add focused DB, validator, service/API, publish snapshot, route, and OpenAPI tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Scripts tables are created in per-user ChaChaNotes DB and draft updates enforce if_revision optimistic conflicts.
- [x] #2 Pure validator covers entry labels, missing targets, typed assignments, structured conditions, manifest/media/profile restrictions, and unreachable-label warnings.
- [x] #3 VN scripts API exposes script CRUD, draft read/write, validation, diagnostics, publish, version, manifest snapshot, and version policy-evaluate endpoints under /api/v1/vn/vn-scripts.
- [x] #4 Publish is idempotent, rejects same-key/different-payload conflicts, repeats authoritative policy evaluation, and snapshots approved asset manifest plus effective policy/generation profiles.
- [x] #5 Route registration and OpenAPI tests include draft, validation, publish, version, manifest snapshot, and version policy-evaluate paths.
- [x] #6 Focused VN_Scripts tests, git diff checks, and Bandit on touched production Python paths are recorded before completion.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Started Task 4 after rebasing codex/vn-platform-api-design onto origin/dev. Task 3 commit is now a9749d044 after rebase. Implementation will follow TDD and keep the API backend-owned under /api/v1/vn/vn-scripts.

Implemented Task 4 VN scripts authoring API, including per-user ChaChaNotes tables, draft revision conflicts, pure validator coverage, backend-owned scripts CRUD/draft/validate/diagnostics/publish/version/manifest/policy endpoints under /api/v1/vn/vn-scripts, atomic publish idempotency, manifest/profile snapshots, profile-store resolution, selected-pack character safety policy evaluation, and raw generation routing rejection at both generation_defaults and generate opcodes. Verification: VN_Scripts 29 passed; affected VN/platform/assets/policy/router/OpenAPI suite 320 passed; compileall exit 0; Bandit touched production scope 0 findings; git diff --check exit 0. Final read-only subagent re-review reported no blocking issues.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the backend-owned VN scripts authoring API for structured JSON scripts. The implementation stores script metadata, drafts, versions, manifest snapshots, publish idempotency records, and profile snapshots in the user ChaChaNotes database; exposes the /api/v1/vn/vn-scripts CRUD, draft, validation, diagnostics, publish, version, manifest snapshot, and version policy-evaluate endpoints; and validates scripts against approved asset manifests, accessible audio refs, selected generation profile limits, and authoritative policy profiles. Publish now snapshots the approved asset manifest and effective policy/generation profiles atomically with the idempotency replay record.
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
