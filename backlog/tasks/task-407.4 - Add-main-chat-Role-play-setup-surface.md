---
id: TASK-407.4
title: Add main chat Role-play setup surface
status: Done
labels:
- chat
- ux
- roleplay
- stage-4
parent_task_id: TASK-407
documentation:
- Docs/superpowers/plans/2026-05-17-main-chat-role-play-preset-remediation-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Stage 4 implementation for the main /chat role-play preset plan: consolidate identity, behavior, scene, generation style, and context preview/apply into a dedicated Role-play setup surface on /chat.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Role-play setup surface shows before/after state for identity, behavior, scene, generation style, and context summary.
- [x] #2 Apply, cancel, clear, and reset are explicit and reversible.
- [x] #3 Scene setup uses existing Actor settings sources without inventing a second scene model.
- [x] #4 Focused Stage 4 tests and browser verification are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Stage 4 started after Stage 3 commit e235cf1cd. Scope remains limited to a main /chat Role-play setup surface that consolidates identity, behavior, scene, generation style, and context preview/apply using existing Actor settings sources.

Implemented a dedicated Role-play setup drawer wired into /chat. The drawer previews before/after state for identity, behavior, scene, generation style, and context; uses the existing AssistantSelect for identity; stages behavior templates and generation preset changes; uses existing Actor settings load/save paths for scene state; and exposes explicit cancel/apply/clear/reset paths.

Verification recorded:
- `bunx vitest run ../packages/ui/src/components/Option/Playground/__tests__/RolePlaySetupDrawer.test.tsx ../packages/ui/src/components/Option/Playground/__tests__/role-play-scene.test.ts ../packages/ui/src/components/Option/Playground/__tests__/ComposerToolbar.role-play-mobile.test.tsx ../packages/ui/src/components/Option/Playground/__tests__/ComposerToolbar.test.tsx --reporter=verbose` passed, 4 files and 37 tests.
- `node -e "JSON.parse(require('fs').readFileSync('../packages/ui/src/assets/locale/en/playground.json','utf8')); JSON.parse(require('fs').readFileSync('../packages/ui/src/public/_locales/en/playground.json','utf8')); console.log('json ok')"` passed.
- `git diff --check` passed.
- `bunx tsc --noEmit --pretty false` still fails only on known unrelated baseline errors in Evaluations recipe config, persona visuals, and vnPlay API typing.
- Browser/CDP verification remains blocked by the existing /chat target-policy conflict; Computer Use was not used.
- Bandit was not run because this stage touched only TypeScript/React, locale JSON, plan docs, and this Backlog task; no Python code was modified.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the main chat Role-play setup surface for Stage 4. The setup drawer consolidates role-play state preview, existing identity selection, behavior template staging, generation preset staging, scene editing through Actor settings, and explicit apply/cancel/clear/reset controls. Focused Stage 4 tests pass; TypeScript remains blocked only by pre-existing unrelated baseline errors; browser verification was recorded as blocked by the current CDP target-policy conflict.
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
