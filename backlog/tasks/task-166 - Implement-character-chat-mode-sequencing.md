---
id: TASK-166
title: Implement character-chat mode sequencing
status: Done
assignee: []
created_date: '2026-05-09 16:18'
updated_date: '2026-05-09 16:25'
labels:
  - character-chat
  - frontend
  - ux-audit
  - mode-sequencing
dependencies:
  - TASK-159
  - TASK-161
documentation:
  - Docs/superpowers/plans/2026-05-09-character-chat-mode-sequencing-plan.md
  - Docs/superpowers/specs/2026-05-09-character-chat-ux-work-packages-design.md
  - Docs/Reviews/CHARACTER_CHAT_WEBUI_UX_AUDIT_2026_05_09.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute the character-chat mode sequencing work package so Chat's character mode follows the task order: choose character, confirm model readiness, optionally configure scene, then send the first message. Preserve model-readiness and selected-character intent contracts already implemented in TASK-159 and TASK-161.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A failing test first captures current character mode opening scene/actor setup before character selection or otherwise failing to present character-first next steps.
- [x] #2 Entering character mode without a selected character presents a character picker or recent-character chooser before optional scene controls.
- [x] #3 Entering character mode with a selected character shows that character as active and keeps Scene/Actor configuration optional.
- [x] #4 Missing model state appears after character selection and names the selected character using the shared readiness contract.
- [x] #5 First send uses the selected character, and switching back to normal chat clears character-only state as intended.
- [x] #6 Focused component/unit tests and the full UI package typecheck are run and recorded.
- [x] #7 Bandit is skipped only if final touched scope remains frontend-only TypeScript/tests/backlog.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace character-mode entry points and identify where Scene/Actor settings open before character selection.
2. Add failing regression coverage for the header character action and Playground character starter.
3. Reuse the existing AssistantSelect character picker via a shared browser event instead of creating a parallel picker.
4. Preserve selected-character behavior and verify normal/temporary chat starts clear character-only state.
5. Run focused Vitest coverage, full UI typecheck, diff hygiene, and record frontend-only Bandit skip.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: two character-mode entry points opened Scene Director before character selection. Header.startCharacterChat dispatched tldw:open-actor-settings when no character was selected, and the Playground empty-state character starter called setOpenActorSettings(true).

Red tests added first: Header.character-mode.test.tsx failed because the header dispatched actor settings; AssistantSelect.behavior.test.tsx failed because AssistantSelect ignored the character-selection event; PlaygroundForm.signals.guard.test.ts failed because the character starter did not use dispatchOpenAssistantSelect and still opened Actor settings.

Implementation: added utils/assistant-select-events.ts with tldw:open-assistant-select and dispatchOpenAssistantSelect. AssistantSelect now listens for that event, activates the requested Characters or Personas tab, clears stale search text, and opens its existing dropdown. Header character chat and Playground character starter dispatch the event with tab=character. Scene Director remains available from the AssistantSelect footer as an optional advanced action.

Regression coverage: Header tests cover no-character character-chat opening character selection instead of Actor settings, selected-character character-chat keeping the active character without opening extra controls, and saved/temporary chat starts clearing selected-character state. AssistantSelect behavior covers opening the character tab from the shared event. PlaygroundForm guard covers the character starter branch.

Verification: bunx vitest run src/components/Common/__tests__/AssistantSelect.behavior.test.tsx src/components/Layouts/__tests__/Header.character-mode.test.tsx src/components/Option/Playground/__tests__/PlaygroundForm.signals.guard.test.ts --testTimeout=20000 exited 0 with 3 files and 11 tests passed.

Typecheck: from apps/packages/ui, ../../tldw-frontend/node_modules/.bin/tsc --noEmit -p tsconfig.json --pretty false > /tmp/tldw_ui_mode_sequencing_tsc.log 2>&1 exited 0; wc -l /tmp/tldw_ui_mode_sequencing_tsc.log reported 0 lines.

Diff hygiene: git diff --check exited 0 with no whitespace errors.

Bandit skipped because this package touched frontend TypeScript/tests, docs, and Backlog tracking only; no Python/backend code changed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the character-chat mode sequencing package by routing character-mode entry points through character selection before optional scene controls. Header Character chat and the Playground character starter now reuse the existing AssistantSelect picker through a shared tldw:open-assistant-select browser event, opening the Characters tab instead of Scene Director when no character is active. Existing selected-character state remains active for fresh character chats, and switching to saved or temporary chat clears character-specific state.

Verification: focused Vitest coverage passes for AssistantSelect behavior, Header character-mode sequencing, and PlaygroundForm starter-branch guard. The full apps/packages/ui TypeScript check exits 0 with an empty log, and git diff --check passes. Bandit was skipped because this is frontend TypeScript/tests/docs/backlog only.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
- [x] #7 Implementation plan updated with executed outcomes and blockers
- [x] #8 Typecheck command and result recorded
<!-- DOD:END -->
