---
id: TASK-402
title: Implement durable conference media collections
status: Done
labels:
- media-ingest
- collections
- backend
- frontend
priority: High
documentation:
- Docs/superpowers/plans/2026-05-16-bulk-conference-ingest-workflow-implementation-plan.md
- Docs/superpowers/plans/2026-05-16-bulk-conference-contract-inventory.md
modified_files:
- Docs/superpowers/plans/2026-05-16-bulk-conference-ingest-workflow-implementation-plan.md
- Docs/superpowers/plans/2026-05-16-bulk-conference-contract-inventory.md
- tldw_Server_API/app/core/DB_Management/Collections_DB.py
- tldw_Server_API/app/api/v1/schemas/media_collections.py
- tldw_Server_API/app/api/v1/endpoints/media/collections.py
- tldw_Server_API/app/api/v1/endpoints/media/__init__.py
- tldw_Server_API/app/api/v1/endpoints/config_info.py
- tldw_Server_API/tests/Collections/test_conference_media_collections.py
- tldw_Server_API/tests/Config/test_docs_info_capabilities.py
- apps/packages/ui/src/services/tldw/domains/media.ts
- apps/packages/ui/src/services/tldw/conference-collections.ts
- apps/packages/ui/src/services/tldw/__tests__/conference-collections.test.ts
- apps/packages/ui/src/services/__tests__/server-capabilities.test.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute Task 2 from the bulk conference ingest workflow implementation plan. Add durable owner-scoped conference/media collection storage, planned item membership, basic collection CRUD/status APIs, frontend service normalizers/wrappers, and keep localStorage review collections explicitly local-only until a migration feature exists.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented Task 2 from the bulk conference ingest workflow plan.

- Added durable `media_collections`, `media_collection_items`, and `media_collection_runs` schema inside `CollectionsDatabase`.
- Added collection create/list/get/update/delete helpers, ordered planned-item membership, item status updates, and media/content resolution helpers.
- Added `/api/v1/media/collections` CRUD and item endpoints under the existing media subrouter.
- Advertised durable media collection capability through docs-info once the route exists.
- Added shared frontend normalizers and `mediaMethods` wrappers for collection CRUD and item updates.
- Recorded the explicit boundary that `media:collections:v1` remains a local-only manual review collection store until a migration UX slice exists.

Verification:

- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Collections/test_conference_media_collections.py tldw_Server_API/tests/Config/test_docs_info_capabilities.py::test_docs_info_exposes_bulk_conference_ingest_capabilities -q`
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/tldw-frontend/node_modules/.bin/vitest run -c vitest.config.ts ../packages/ui/src/services/tldw/__tests__/conference-collections.test.ts ../packages/ui/src/services/__tests__/server-capabilities.test.ts`
- `git diff --check`
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/core/DB_Management/Collections_DB.py tldw_Server_API/app/api/v1/endpoints/media/collections.py tldw_Server_API/app/api/v1/schemas/media_collections.py tldw_Server_API/app/api/v1/endpoints/config_info.py -f json -o /tmp/bandit_bulk_conference_collections.json`

Known skip: full shared UI TypeScript still fails on repo-wide pre-existing errors outside this slice. After adding the missing local `MediaMethods` export, the rerun did not report touched files from this task.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added durable conference/media collection storage and APIs plus shared frontend service support, preserving planned playlist items separately from resolved content_items and leaving existing localStorage review collections local-only.
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
