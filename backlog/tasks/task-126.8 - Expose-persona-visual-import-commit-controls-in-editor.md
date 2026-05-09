---
id: TASK-126.8
title: Expose persona visual import commit controls in editor
status: Done
assignee: []
created_date: '2026-05-09 16:53'
updated_date: '2026-05-09 17:02'
labels:
  - persona
  - visual-packs
  - portability
  - frontend
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1422'
  - 'https://github.com/rmusser01/tldw_server/pull/1135'
documentation:
  - Docs/Product/WebUI/Persona_Live_Visual_Packs_PRD.md
  - >-
    Docs/superpowers/plans/2026-05-08-persona-visual-packs-implementation-plan.md
  - >-
    Docs/superpowers/plans/2026-05-09-persona-visual-import-commit-controls-plan.md
parent_task_id: TASK-126
priority: medium
---

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 VisualPackEditor shows an explicit import commit action only after an import preview is completed.
- [x] #2 Import commit starts the existing create_new import_commit job and keeps activation separate from commit.
- [x] #3 VisualPackEditor shows import commit status/stage/job id and can refresh status.
- [x] #4 Completed import commit refreshes pack state so the new draft can be selected or inspected.
- [x] #5 Focused frontend tests cover commit start/status and completed pack refresh behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Plan: Docs/superpowers/plans/2026-05-09-persona-visual-import-commit-controls-plan.md. Scope: frontend service/types plus VisualPackEditor controls for the existing import_commit backend flow. TDD target: VisualPackEditor focused import-preview/commit coverage first.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created implementation plan at Docs/superpowers/plans/2026-05-09-persona-visual-import-commit-controls-plan.md.

RED: bunx vitest run src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx first hit missing fresh-worktree dependencies; after bun install, the test failed as intended because persona-visual-import-commit-button was absent after a completed import preview.

GREEN: bunx vitest run src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx passed with 7 tests after adding import commit service helpers and editor controls.

RELATED VERIFICATION: bunx vitest run src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx src/components/Common/PersonaBuddy/__tests__/BuddyShellHost.test.tsx src/utils/__tests__/persona-garden-route.test.ts passed with 31 tests.

HYGIENE: git diff --check passed.

BANDIT: not applicable; touched production code is frontend TypeScript plus plan/task metadata only.

No known blockers. Bandit skipped as non-applicable for frontend TypeScript-only production changes.

TSC: bunx tsc --noEmit --pretty false still exits 2 on existing repo-wide diagnostics; filtered /tmp/persona-visual-import-tsc.log shows no diagnostics for the touched VisualPackEditor, VisualPackEditor test, persona-visuals service, or persona-visuals types files after the ArrayBufferView BlobPart copy fix.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added typed Persona Visuals import-commit helpers and exposed commit-as-draft controls in VisualPackEditor after a completed import preview. The editor now starts the existing untrusted create_new import_commit flow, shows status/stage/job id, refreshes commit status, and reloads packs when the committed draft is available without activating it. Also tightened the touched Persona Visuals service export-download BlobPart handling so arbitrary ArrayBufferView responses are copied into a Blob-safe Uint8Array.

Verification: focused/related Persona Garden and Buddy Vitest passed, git diff --check passed, and a filtered tsc log showed no diagnostics for the touched files. Full tsc still exits 2 on existing repo-wide diagnostics outside this slice. Bandit was not applicable because touched production code is frontend TypeScript.
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
