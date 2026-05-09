---
id: TASK-180
title: Improve Persona Visuals generation setup and unavailable-provider states
status: Done
assignee: []
created_date: '2026-05-09 19:16'
updated_date: '2026-05-09 19:42'
labels:
  - WebUI
  - Persona
  - Buddy
  - visual-packs
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1428'
  - 'https://github.com/rmusser01/tldw_server/issues/1431'
  - 'https://github.com/rmusser01/tldw_server/pull/1439'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement GitHub issue #1431 for the Persona/Buddy visual-pack system. Make asset generation readiness obvious before users start background jobs by distinguishing provider/model unavailability from Jobs/backend unavailability, preventing or clearly gating generation actions that cannot succeed, and preserving the generated-asset review workflow.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Persona Visuals editor shows clear setup/unavailable states when generation provider/backend configuration is missing or disabled.
- [x] #2 Jobs/backend unavailability is distinguished from image provider/model unavailability in user-facing copy and gating.
- [x] #3 Generation actions that cannot succeed are disabled or clearly guarded without bypassing the review-step workflow.
- [x] #4 Focused UI/service tests cover disabled and missing-provider generation readiness states.
- [x] #5 Existing persona visual generation job enqueue and review candidate flows continue to work when readiness is available.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Create an isolated branch/worktree from latest origin/dev for issue #1431.
2. Inspect existing persona visual generation service/API/editor code to find the current readiness signal and generation action path.
3. Add failing focused tests for missing provider/backend readiness and action gating in the Persona Visuals editor/service boundary.
4. Implement a minimal frontend readiness classifier and wire it into VisualPackEditor generation controls without changing provider implementations.
5. Preserve the existing generation job enqueue and candidate review flow for ready states.
6. Run focused Vitest/service tests, diff hygiene, and applicable static checks; update Backlog, commit, push, and open/update PR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the Persona Visuals generation readiness slice for issue #1431 in worktree .worktrees/persona-visual-generation-readiness. Added a pack-scoped backend readiness endpoint, WebUI service/type support, a tested readiness classifier, and VisualPackEditor setup-state rendering/gating for disabled Jobs worker and missing image provider states.

Verification: Vitest focused Persona visual suite passed (VisualPackEditor, personaVisualGenerationReadiness, personaVisualDiagnostics, SpriteFrameRenderer): 34 tests passed. Pytest Persona visual API suite passed: 24 tests passed. Bandit on touched backend endpoint/schema files reported zero findings. Package-wide tsc currently fails on existing unrelated baseline errors; grep of tsc output for touched files found no errors in VisualPackEditor/persona-visuals/personaVisualGenerationReadiness.

PR review follow-up: addressed Qodo and Gemini comments by checking adapter instantiation in the readiness endpoint, adding the requested Python docstrings, adding adapter-failure regression coverage, and guarding VisualPackEditor readiness loads against stale async responses after rapid pack switches.

Review verification: Vitest focused Persona visual suite passed (VisualPackEditor, personaVisualGenerationReadiness, personaVisualDiagnostics, SpriteFrameRenderer): 36 tests passed. Pytest Persona visual API suite passed: 25 tests passed. Bandit on touched backend endpoint/schema files reported zero findings. git diff --check passed. The package-wide TypeScript baseline still reports an unrelated PersonaGarden/MCPExternalCatalog.tsx error; touched Persona Visuals files were clean in the grep-filtered output.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented PR #1439 for issue #1431 and addressed follow-up review comments. The slice adds a pack-scoped Persona Visuals generation readiness API, WebUI service/types, a tested readiness classifier, VisualPackEditor setup-state rendering/gating, adapter-instantiation preflight, and stale-readiness request protection. Generation enqueue remains available when readiness is available, and generated candidates still go through the review workflow. Verification recorded in implementation notes; package-wide TypeScript still has unrelated baseline failures, with no errors found in the touched Persona Visuals files.
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
