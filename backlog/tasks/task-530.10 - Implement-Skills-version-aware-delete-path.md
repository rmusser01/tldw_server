---
id: TASK-530.10
title: Implement Skills version-aware delete path
status: Done
labels:
- skills
- webui
- safe-operations
- backend
priority: high
ordinal: 530.1
parent_task_id: TASK-530
documentation:
- Docs/superpowers/plans/2026-06-28-skills-version-aware-delete.md
modified_files:
- tldw_Server_API/app/api/v1/endpoints/skills.py
- tldw_Server_API/app/api/v1/schemas/skills_schemas.py
- tldw_Server_API/app/core/Skills/skills_service.py
- tldw_Server_API/tests/Skills/integration/test_skills_api.py
- apps/packages/ui/src/types/skill.ts
- apps/packages/ui/src/services/tldw/domains/workspace-api.ts
- apps/packages/ui/src/services/tldw/domains/__tests__/workspace-api.skills.test.ts
- apps/packages/ui/src/components/Option/Skills/Manager.tsx
- apps/packages/ui/src/components/Option/Skills/__tests__/Manager.test.tsx
references:
- https://github.com/rmusser01/tldw_server/pull/2544
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue TASK-530 Safe Operations after TASK-530.9 by adding version-aware single-skill delete behavior. Extend the frontend/API path so stale destructive deletes can be blocked and recovered before any bulk-delete work. Keep bulk delete, export feedback, and permission metadata panels out of scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Delete requests send an If-Match version when the frontend has a known skill version.
- [x] #2 Backend delete validates If-Match consistently and returns a recoverable conflict for stale versions.
- [x] #3 The Skills manager shows a clear reload-before-delete recovery message on stale delete conflicts.
- [x] #4 Existing delete behavior remains compatible when no version is known.
- [x] #5 Focused frontend and backend tests cover successful delete, no-version compatibility, and stale-version conflict handling.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Spec approved by user and written at `Docs/superpowers/specs/2026-06-28-skills-version-aware-delete-design.md`. Design scope: expose row versions in Skills list summaries, send `If-Match` from frontend deletes when known, preserve unversioned delete compatibility, and show reload-before-delete recovery copy on stale conflicts.

Self-review completed before implementation planning. Found and patched a spec gap: `SkillSummary` is reused by `/skills/context` and async context integration paths, so requiring `version` must also update `_build_context_payload()` and context/MCP fixtures while keeping `context_text` unchanged. Also tightened delete conflict detection guidance to use a helper covering common wrapped error shapes.

Spec review loop completed: reviewer approved with no blocking issues. Advisory for planning: confirm the actual React Query skills query key/invalidation behavior or use an existing shared key helper if one exists.

Final pre-plan review completed. Confirmed React Query invalidation uses the existing `['skills']` prefix. Patched two planning issues: `If-Match` should only be sent for positive safe integer versions, and backend verification must include context/MCP paths affected by the shared `SkillSummary` schema.

Implementation plan written at `Docs/superpowers/plans/2026-06-28-skills-version-aware-delete.md`. Plan splits work into backend contract/API coverage, frontend API-client header handling, Skills manager UX recovery, and final verification/bookkeeping.

Plan review loop completed: reviewer approved with no blocking issues or recommendations.

Implementation completed in staged commits. Task 1 added backend summary/context version exposure and delete API coverage. Task 2 added the frontend API client `If-Match` header guard. Task 3 wired row versions through the Skills manager delete confirmation, added stale-conflict recovery copy, and tightened conflict detection after review to avoid treating arbitrary messages containing `409` as conflicts. Final branch review follow-up made handled stale conflicts resolve the confirmation flow so users are not left in a stale delete modal after recovery feedback, while generic delete failures still reject.

Verification completed:
- Backend focused pytest: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Skills/integration/test_skills_api.py tldw_Server_API/tests/Skills/integration/test_skill_mcp_integration.py -k "list_skills or delete_skill or get_context_payload or async_variant_uses_async_context_payload" -v` -> 21 passed, 55 deselected.
- Frontend focused Vitest: `bunx vitest run src/services/tldw/domains/__tests__/workspace-api.skills.test.ts src/components/Option/Skills/__tests__/Manager.test.tsx --reporter=dot` -> 2 files passed, 39 tests passed.
- Bandit: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/api/v1/endpoints/skills.py tldw_Server_API/app/api/v1/schemas/skills_schemas.py tldw_Server_API/app/core/Skills/skills_service.py -f json -o /tmp/bandit_task_530_10.json` -> 0 results.

Known skip: no full frontend build/typecheck was run; focused backend/API-client/manager suites cover this task slice.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented version-aware single-skill delete across backend and WebUI. Skill summaries now expose `version`, context payload summaries include version metadata without changing prompt text, and delete API integration tests cover matching and stale `If-Match` behavior. The frontend API client now accepts an optional delete version and sends `If-Match` only for positive safe integers. The Skills manager passes row versions through the existing destructive confirmation, preserves unknown-version compatibility, refreshes the skills list on stale 409 conflicts, and shows reload-before-delete recovery copy. Verification: backend focused pytest selected 21 tests passed; frontend focused Vitest selected 39 tests passed; Bandit on touched backend files wrote `/tmp/bandit_task_530_10.json` with 0 findings. Known skip: no full frontend build/typecheck was run; focused suites cover this slice.
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
