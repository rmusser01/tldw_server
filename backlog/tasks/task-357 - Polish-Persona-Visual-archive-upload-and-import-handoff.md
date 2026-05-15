---
id: TASK-357
title: Polish Persona Visual archive upload and import handoff
status: Done
assignee:
  - Codex
created_date: '2026-05-15 02:29'
updated_date: '2026-05-15 02:53'
labels:
  - persona
  - webui
  - visual-packs
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1696'
  - 'https://github.com/rmusser01/tldw_server/issues/1510'
documentation:
  - Docs/Code_Documentation/Persona_Visual_Packs.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement GitHub issue #1696 as the next persona-side setup-ready slice under the Persona/Buddy epic. Tighten the end-user Persona Visual archive upload/import path with clearer validation/failure copy, visible worker/status feedback where the current flow supports it, and an explicit post-commit handoff that selects/shows the new draft while preserving separate explicit activation. Keep this scoped to Persona Visual import handoff behavior; do not add automatic activation, Live2D runtime support, external provider execution, VN/CYOA behavior, or a new default starter catalog path.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Unsupported extensions or malformed archives fail with clear user-facing copy.
- [x] #2 Import worker readiness/status is visible enough for a user to understand pending or unavailable work.
- [x] #3 Successful import commit selects or shows the resulting draft and does not activate it.
- [x] #4 Focused tests cover extension or failure copy and post-commit draft selection behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Stage 1: Plan and baseline
- Keep work isolated in .worktrees/persona-visual-import-handoff-1696 on codex/persona-visual-import-handoff-1696.
- Scope remains #1696 only: Persona Visual archive upload/import handoff polish; no automatic activation, Live2D runtime, external provider execution, VN/CYOA, or starter catalog expansion.

Stage 2: Failing focused tests
- Add VisualPackEditor tests that unsupported archive names show clear copy and do not POST.
- Add/extend import commit tests so completed commit selects the returned draft pack_id and keeps activation separate.
- Add status/failure copy coverage for import preview or commit job responses.

Stage 3: Minimal implementation
- Fix createPersonaVisualImportPreview FormData to send archive, matching FastAPI.
- Add client-side .tldw-persona-vpack guard and clear UI copy near the import picker.
- Add compact import preview/commit status messages from status/stage/error fields.
- Update completed commit refresh to select the returned draft pack_id after loadPacks succeeds.

Stage 4: Validation
- Run focused Vitest for VisualPackEditor.
- Run targeted backend Persona visual API tests if backend import contract or errors change.
- Run git diff --check and Bandit for touched Python if any Python changes are made; otherwise record Bandit as not applicable for TypeScript/Markdown-only changes.

Stage 5: Closeout
- Update TASK-357 acceptance criteria, notes, verification, and final summary.
- Commit the focused branch and open/update the PR for #1696.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-05-15: Started planning in isolated worktree .worktrees/persona-visual-import-handoff-1696 after PR #1701 merged. MCP Backlog is rooted at the dirty main checkout, so this task is being maintained with the Backlog CLI in the branch worktree.

Verification 2026-05-15:
- RED: focused VisualPackEditor Vitest failed on the new import expectations before implementation.
- GREEN: bunx vitest run ../packages/ui/src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx passed with 26 tests.
- Hygiene: git diff --check passed.
- Bandit: not applicable because this slice touches TypeScript and Markdown only; no Python code changed.

Known skips/blockers 2026-05-15:
- Backend/API import contract was not changed in this slice, so backend Persona visual API tests were not run.
- Bandit was skipped as not applicable because no Python files changed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the Persona Visual archive import handoff polish for #1696. The WebUI now sends import preview uploads with the backend archive field, rejects non-.tldw-persona-vpack filenames before upload, surfaces preview/commit job copy from status/stage/error fields, and selects the imported draft returned by a completed commit without activating it.

Verification:
- bunx vitest run ../packages/ui/src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx (26 passed)
- git diff --check (passed)
- Bandit N/A: TypeScript/Markdown-only changes.
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
