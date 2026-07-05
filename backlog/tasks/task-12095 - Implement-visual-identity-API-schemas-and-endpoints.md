---
id: TASK-12095
title: Implement visual identity API schemas and endpoints
status: Done
labels:
- visual-identities
- expression-packs
- api
priority: High
references:
- Docs/superpowers/specs/2026-07-01-visual-identity-expression-packs-design.md
- Docs/superpowers/plans/2026-07-01-visual-identity-expression-packs-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Stage 6 API schemas and endpoints for visual identity expression packs: authenticated capabilities, expression slots, pack/draft/asset/binding operations, ZIP import start, activation, binding resolution, asset content serving, and router registration.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Stage 6: expose visual identity capabilities, expression slots, pack/draft/asset/binding operations, ZIP import start, draft activation, and binding resolution through authenticated FastAPI schemas/endpoints registered under /api/v1/visual-identities. Keep frontend/chat integration out of scope.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- 2026-07-02: Added TDD API coverage first. RED was observed with `ImportError: cannot import name 'visual_identities'` when running `python -m pytest -q tldw_Server_API/tests/Visual_Identities/test_visual_identities_api.py`.
- Added `visual_identity_schemas.py` with the required Stage 6 schema names for capabilities, expression slots, packs, drafts, assets, bindings, resolution, ZIP import start, and generated-file asset attachment.
- Added `visual_identities.py` endpoints under the router-local root, using `get_request_user`, `get_chacha_db_for_user`, `rbac_rate_limit`, `JobManager()`, and `AuthnzGeneratedFilesRepo` dependency patterns.
- Registered the router in `router_groups/core.py` with `prefix=f"{API_V1_PREFIX}/visual-identities"`, `tags=("visual-identities",)`, and `route_key="visual-identities"`.
- V1 asset upload behavior: `POST /packs/{pack_id}/assets` attaches uploads to a draft. If `draft_id` is omitted, the endpoint creates a `ready_for_review` draft for the pack so active pack versions remain immutable until draft activation.
- Implemented `/packs/{pack_id}/assets/from-generated-file` with the existing generated-file repo lookup and `copy_generated_file_record_to_expression_asset` storage helper.
- Implemented `/imports/zip` as a job-start wrapper: it creates an importing draft, stores the ZIP under the user's visual identity imports directory, and queues `create_visual_identity_import_zip_job`.
- Asset content uses owner-scoped `repo.get_asset`, validates pack ownership and MIME type, resolves the stored relpath through `resolve_visual_identity_asset_path`, and returns `FileResponse` with `Cache-Control: public, max-age=31536000, immutable`.
- Verification: `python -m pytest -q tldw_Server_API/tests/Visual_Identities/test_visual_identities_api.py` passed with 5 tests.
- Verification: `python -m pytest -q tldw_Server_API/tests/Visual_Identities` passed with 100 tests.
- Verification: `python -m bandit -r tldw_Server_API/app/api/v1/endpoints/visual_identities.py tldw_Server_API/app/api/v1/schemas/visual_identity_schemas.py tldw_Server_API/app/core/Visual_Identities tldw_Server_API/app/core/DB_Management/VisualIdentity_DB.py -f json -o /tmp/bandit_visual_identity_stage6.json` completed with zero findings.
- Verification: `git diff --check 7eee48dc66..HEAD` passed with no output.
2026-07-02 post-review hardening: addressed Stage 6 review findings by returning resolvable asset URLs with null public fallback for direct requests, requiring and enforcing API-layer idempotency for generated-file asset attachment and ZIP import starts, persisting ZIP import job IDs on drafts, scoping ZIP job idempotency away from raw client keys, validating generated-file records and image bytes before draft creation, reclaiming stale in-progress idempotency claims, and adding claim-token fencing so older attempts cannot complete or release newer claims. Added regression coverage for resolver asset URLs/fallbacks, ZIP replay/job persistence, generated-file replay/conflict behavior, invalid generated-file preflight rejection, stale claim reclaim, and claim-token stale completion/release no-ops. Follow-up quality review reported no remaining Critical/Important issues. Verification: git diff --check passed; compileall passed for touched backend files; python -m pytest -q tldw_Server_API/tests/Visual_Identities passed with 106 tests; Bandit JSON /tmp/bandit_visual_identity_stage6_final.json reported errors [] and results_count 0.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stage 6 visual identity API schemas and endpoints are implemented and registered. The API exposes capabilities, expression slots, pack CRUD, draft slot updates and activation, asset upload/content serving, generated-file asset attachment, ZIP import job start, binding upsert/delete, and binding resolution while preserving owner scoping and immutable active-version assets.
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
