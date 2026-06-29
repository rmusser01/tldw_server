# Skills Version-Aware Bulk Delete Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a selected-row bulk delete flow for Skills that sends optimistic versions and refuses stale destructive deletes before removing any selected skill.

**Architecture:** Reuse the existing `SkillSummary.version` contract from TASK-530.10. Add a POST bulk-delete endpoint with an explicit request body instead of relying on DELETE bodies. The backend prevalidates all selected names and expected versions before deletion; the frontend adds table selection, a small selected-action bar, and the same reload-before-delete recovery pattern used by single delete.

**Tech Stack:** FastAPI, Pydantic, SkillsService, ChaChaNotesDB skill registry, React, Ant Design Table row selection, TanStack Query, Vitest, pytest.

---

## Scope

In scope:
- `POST /api/v1/skills/bulk-delete`
- Frontend API client `bulkDeleteSkills()`
- Skills manager row selection and destructive confirmation
- Stale-version recovery feedback
- Focused backend and frontend tests

Out of scope:
- Delete all filtered results across pages
- Export feedback
- Permission metadata panels
- Undo/restore
- New DB schema

## Files

- Modify: `tldw_Server_API/app/api/v1/schemas/skills_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/skills.py`
- Modify: `tldw_Server_API/app/core/Skills/skills_service.py`
- Modify: `tldw_Server_API/tests/Skills/integration/test_skills_api.py`
- Modify: `apps/packages/ui/src/types/skill.ts`
- Modify: `apps/packages/ui/src/services/tldw/domains/workspace-api.ts`
- Modify: `apps/packages/ui/src/services/tldw/domains/__tests__/workspace-api.skills.test.ts`
- Modify: `apps/packages/ui/src/components/Option/Skills/Manager.tsx`
- Modify: `apps/packages/ui/src/components/Option/Skills/__tests__/Manager.test.tsx`
- Modify: `backlog/tasks/task-530.11 - Implement-Skills-version-aware-bulk-delete.md`

## Task 1: Backend Bulk Delete Contract

- [x] Add Pydantic models `SkillBulkDeleteItem`, `SkillBulkDeleteRequest`, and `SkillBulkDeleteResponse`.
- [x] Add failing integration tests for successful bulk delete, unknown-version compatibility, stale conflict, and no partial delete on stale conflict.
- [x] Add `SkillsService.bulk_delete_skills(items)` that normalizes names, syncs once, validates every selected row/version, then deletes the validated rows.
- [x] Add `POST /api/v1/skills/bulk-delete` endpoint that returns deleted names/count and maps `SkillConflictError` to 409, `SkillNotFoundError` to 404, and generic `SkillsError` to sanitized 500.
- [x] Run: `python -m pytest tldw_Server_API/tests/Skills/integration/test_skills_api.py -k "bulk_delete or delete_skill" -v`

## Task 2: Frontend API Client

- [x] Add TypeScript request/response types for bulk delete.
- [x] Add `workspaceApiMethods.bulkDeleteSkills(items)` using `POST /api/v1/skills/bulk-delete`.
- [x] Send only positive safe integer versions in request items.
- [x] Add focused API-client tests for valid versions and unknown/invalid versions.
- [x] Run: `bunx vitest run src/services/tldw/domains/__tests__/workspace-api.skills.test.ts --reporter=dot`

## Task 3: Skills Manager Bulk UX

- [x] Add selected row state keyed by skill name.
- [x] Add AntD `rowSelection` to the Skills table.
- [x] Add an unframed action bar above the table when rows are selected: `<count> selected`, clear selection, and destructive `Delete selected`.
- [x] Confirm bulk deletion with selected count and a short irreversible warning.
- [x] On success, clear selection, invalidate `["skills"]`, and show a bulk-delete success notification.
- [x] On stale 409 conflict, keep the selection recoverable, invalidate `["skills"]`, and show reload-before-delete guidance.
- [x] Add focused manager tests for selected-row bulk delete, unknown-version compatibility, stale conflict recovery, and clearing selection on success.
- [x] Run: `bunx vitest run src/components/Option/Skills/__tests__/Manager.test.tsx --reporter=dot`

## Task 4: Verification And Closeout

- [x] Run backend focused pytest from Task 1.
- [x] Run frontend focused Vitest from Tasks 2 and 3.
- [x] Run Bandit touched backend scope: `python -m bandit -r tldw_Server_API/app/api/v1/endpoints/skills.py tldw_Server_API/app/api/v1/schemas/skills_schemas.py tldw_Server_API/app/core/Skills/skills_service.py -f json -o /tmp/bandit_task_530_11.json`
- [x] Run `git diff --check`.
- [x] Update TASK-530.11 acceptance criteria, implementation notes, final summary, modified files, verification, and known skips.
- [x] Commit and open PR against `dev`.

PR: https://github.com/rmusser01/tldw_server/pull/2545
