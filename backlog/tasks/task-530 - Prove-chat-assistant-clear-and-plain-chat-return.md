---
id: TASK-530
title: Prove chat assistant clear and plain-chat return
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-27 20:20'
labels:
  - chat
  - ux
  - e2e
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address /chat UX rebaseline F7 by tightening the real-server/runtime-rail contract for character/persona clear state and plain-chat return after assistant selection. Keep scope limited to /chat assistant state continuity and directly relevant tests; do not redesign persona, history, or sidepanel handoff flows.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Clearing an assistant from the /chat runtime rail clears selected assistant state, legacy character mirror state, server-chat assistant metadata, and persisted assistantOverlay chat settings.
- [x] #2 The runtime rail real-server contract expects the region-specific empty assistant label and no longer looks for the stale generic No assistant selected copy.
- [x] #3 The real-server character-clear journey includes a plain-chat return assertion proving the next create-chat payload is webui-chat without character_id, assistant_kind, or assistant_id.
- [x] #4 Focused /chat cockpit/runtime tests pass, Playwright real-server spec lists/parses the touched tests, and known live-server/type-check limitations are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Starting focused /chat F7 assistant-clear/plain-chat continuity slice. Existing unit coverage cleared selected assistant and server metadata, but investigation found the cockpit clear path did not clear persisted assistantOverlay chat settings, which can rehydrate assistant behavior after the UI appears plain.

RED: added Playground.cockpit-controls coverage expecting Clear assistant to call applyChatSettingsPatch with assistantOverlay: null. It failed with zero calls before implementation.

Implementation: clearAssistantFromCockpit now best-effort clears assistantOverlay for the current history/server chat while preserving existing selectedAssistant, selectedCharacter, server metadata, persisted-session, workflow-mode, and focus-reset behavior.

Real-server contract update: chat-cockpit.real-server now uses No runtime assistant selected for cleared runtime state and extends the disposable character clear journey to assert the next create-chat request is plain webui-chat without character_id, assistant_kind, or assistant_id.

Verification: focused RED failed as expected before implementation; GREEN focused run passed 78 tests across Playground.cockpit-controls, PlaygroundRuntimeInspector.first-slice, Playground.cockpit-a11y, PlaygroundCompositionPreview, playground-composition-preview, and playground-cockpit-summaries. git diff --check passed. Playwright --list parsed/listed the touched real-server character/persona clear tests. UI tsc remains blocked by the known unrelated CharacterListContent.design-system.test.tsx GalleryCardDensity baseline. Live real-server execution was not run because no backend was listening on 127.0.0.1:8000 in this turn. Bandit skipped because touched code is TS/TSX/Playwright only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Hardened /chat assistant clear continuity. Runtime clear now removes persisted assistantOverlay settings so a cleared assistant cannot rehydrate on the next plain turn, and the real-server contract now asserts the corrected empty runtime label plus a plain-chat return create payload after character clear.
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
