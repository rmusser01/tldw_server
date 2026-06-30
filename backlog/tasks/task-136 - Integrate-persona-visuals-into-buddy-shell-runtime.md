---
id: TASK-136
title: Integrate persona visuals into buddy shell runtime
status: Done
assignee: []
created_date: '2026-05-09 00:37'
updated_date: '2026-05-09 00:44'
labels:
  - persona
  - webui
  - frontend
  - implementation
dependencies:
  - TASK-135
documentation:
  - Docs/superpowers/specs/2026-05-08-persona-visual-packs-design.md
  - >-
    Docs/superpowers/plans/2026-05-08-persona-visual-packs-implementation-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Load active persona visual packs in the buddy shell, render sprite-frame visuals when available, add bounded runtime visual-state overrides, and expose compact visual-state feedback in live persona UI. Keep pack editor and generation jobs for later slices.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Buddy shell requests active visual packs for the active persona and renders SpriteFrameRenderer when a valid active pack is available
- [x] #2 Buddy shell preserves existing text and dormant behavior when no pack exists or pack loading fails
- [x] #3 Live voice speaking state and active tool status map into resolved visual states for shell rendering
- [x] #4 Persona visual runtime store supports bounded visual_state_override payloads and expiry fallback
- [x] #5 Live session UI exposes compact current visual-state feedback without changing voice controls
- [x] #6 Focused tests cover shell loading/rendering/fallback and runtime override behavior
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Red verification: `bunx vitest run src/components/Common/PersonaBuddy/__tests__/BuddyShellHost.test.tsx src/hooks/__tests__/usePersonaLiveVoiceController.test.tsx src/components/PersonaGarden/__tests__/LiveSessionPanel.test.tsx src/store/__tests__/persona-visual-runtime.test.ts src/routes/hooks/__tests__/usePersonaIncomingPayload.visuals.test.tsx` failed because `persona-visual-runtime` was missing, AssistantVoiceCard had no visual-state feedback row, and BuddyShellHost did not call visual-pack APIs or render SpriteFrameRenderer.

Green verification: the same focused Vitest command passed, 5 files and 81 tests passed.

Static check note: package-wide `tsc --noEmit` still has unrelated existing errors. A filtered rerun found no TypeScript errors referencing the touched persona visual, buddy shell, sidepanel, incoming-payload, runtime-store, or AssistantVoiceCard files after fixing the touched-file issues.

Hygiene/security: `git diff --check` passed. Bandit is not applicable because this slice only touches TypeScript/React frontend files and markdown task tracking.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Integrated active persona visual packs into the buddy shell runtime. The shell now loads active packs for the resolved persona, fetches pack detail when needed, renders SpriteFrameRenderer only for valid sprite-frame packs, and preserves existing derived/dormant fallback behavior. Added a Zustand runtime override store, incoming visual_state_override handling, render-context live state propagation, and compact Assistant Voice visual-state feedback.
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
