---
id: TASK-294
title: Create Persona Buddy renderer capability implementation plan
status: Done
assignee: []
created_date: '2026-05-12 05:01'
updated_date: '2026-05-12 05:08'
labels:
  - persona
  - buddy
  - visual-packs
  - plan
dependencies:
  - TASK-293
documentation:
  - Docs/superpowers/specs/2026-05-12-persona-buddy-renderer-capability-registry-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the implementation plan for the approved Persona Buddy renderer capability registry spec. The plan should be a standalone agentic handoff that covers backend capability registry/API schemas/endpoint, frontend service and Buddy renderer registry changes, focused tests, verification, and review boundaries. It should preserve sprite_frames-only V1 behavior, fail-closed unsupported renderers at activation/import-preview validation, permissive draft manifest saves, and no Persona Chat/VN/CYOA/Live2D runtime work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Implementation plan is added under Docs/superpowers/plans and references the approved renderer capability spec.
- [x] #2 Plan decomposes backend registry/API and frontend Buddy runtime work into reviewable TDD tasks with exact files and commands.
- [x] #3 Plan explicitly preserves permissive draft manifest saves and avoids renderer-level asset-role enforcement.
- [x] #4 Plan includes verification commands for backend tests, frontend tests, diff check, and Bandit on touched backend scope.
- [x] #5 Plan review feedback is addressed or documented before implementation starts.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Write a standalone implementation plan from the approved spec.
2. Run the plan review loop and patch valid findings.
3. Record verification evidence and final summary in this task.
4. Commit the plan and task update on the clean renderer capability branch.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Worktree: /Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/persona-buddy-renderer-capability-spec

Branch: codex/persona-buddy-renderer-capability-spec

Plan review approved with no blocking issues. Incorporated advisory notes to remove SUPPORTED_RENDERER_TYPES from visuals.py exports and to include SpriteFrameRenderer.tsx if the frontend helper split is needed to avoid a registry/diagnostics import cycle.

Verification: git diff --check passed. ASCII scan found no non-ASCII characters. Bandit skipped because this task changed only markdown documentation and Backlog task metadata.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the Persona Buddy renderer capability registry implementation plan at Docs/superpowers/plans/2026-05-12-persona-buddy-renderer-capability-registry-implementation-plan.md. The plan decomposes the work into backend registry validation, API capability endpoint, validation-boundary regressions, frontend service/renderer registry, Buddy diagnostics integration, and final verification. Plan review approved with no blocking issues and advisory improvements were incorporated.
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
