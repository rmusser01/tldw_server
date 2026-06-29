---
id: TASK-530.11
title: Implement Skills version-aware bulk delete
status: Done
labels:
- skills
- webui
- safe-operations
- backend
priority: high
parent_task_id: TASK-530
documentation:
- Docs/superpowers/plans/2026-06-28-skills-version-aware-bulk-delete.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue TASK-530 Safe Operations after TASK-530.10 by adding version-aware bulk delete for selected Skills rows. Preserve single-delete compatibility, block stale destructive bulk deletes with recoverable conflict feedback, and keep export feedback and permission metadata panels out of scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Bulk delete sends per-skill versions when known and omits versions only for unknown legacy rows.
- [x] #2 Backend bulk delete validates expected versions atomically enough to avoid partial stale deletes and returns a recoverable conflict when any selected skill is stale.
- [x] #3 The Skills manager exposes a clear selected-row bulk delete action with destructive confirmation and stale-conflict recovery copy.
- [x] #4 Existing single delete behavior and unversioned compatibility remain unchanged.
- [x] #5 Focused frontend and backend tests cover successful bulk delete, unknown-version compatibility, and stale-version conflict handling.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added `POST /api/v1/skills/bulk-delete` with Pydantic request/response schemas and service-level prevalidation before any selected skill is deleted.
- Backend validation rejects duplicate/invalid selection items, preserves legacy unknown-version compatibility, and returns 409 conflicts when a known selected version is stale.
- Added `workspaceApiMethods.bulkDeleteSkills()` and shared frontend bulk-delete request/response types; unsafe versions are omitted from request items.
- Added Skills manager row selection, selected-action bar, destructive confirmation, success clearing, and stale-conflict recovery feedback that keeps selection recoverable.

## Modified Files
- `tldw_Server_API/app/api/v1/endpoints/skills.py`
- `tldw_Server_API/app/api/v1/schemas/skills_schemas.py`
- `tldw_Server_API/app/core/Skills/skills_service.py`
- `tldw_Server_API/tests/Skills/integration/test_skills_api.py`
- `apps/packages/ui/src/types/skill.ts`
- `apps/packages/ui/src/services/tldw/domains/workspace-api.ts`
- `apps/packages/ui/src/services/tldw/domains/__tests__/workspace-api.skills.test.ts`
- `apps/packages/ui/src/components/Option/Skills/Manager.tsx`
- `apps/packages/ui/src/components/Option/Skills/__tests__/Manager.test.tsx`
- `Docs/superpowers/plans/2026-06-28-skills-version-aware-bulk-delete.md`

## Verification
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Skills/integration/test_skills_api.py -k "bulk_delete or delete_skill" -v` - 8 passed, 59 deselected.
- `bunx vitest run src/services/tldw/domains/__tests__/workspace-api.skills.test.ts src/components/Option/Skills/__tests__/Manager.test.tsx --reporter=dot` - 2 files passed, 44 tests passed.
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/api/v1/endpoints/skills.py tldw_Server_API/app/api/v1/schemas/skills_schemas.py tldw_Server_API/app/core/Skills/skills_service.py -f json -o /tmp/bandit_task_530_11.json` - 0 findings.

## Known Skips Or Blockers
- None.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
- Implemented version-aware bulk deletion for Skills across backend, frontend API, and manager UI.
- The backend prevalidates all selected names and optimistic versions before deleting any skill, so a stale selected row returns a recoverable conflict without partial deletion.
- The UI now supports row selection, a selected-action bar, destructive confirmation, version-aware request payloads, success clearing, and reload-before-delete conflict guidance.
- PR: https://github.com/rmusser01/tldw_server/pull/2545

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
