---
id: TASK-347.1
title: Implement Persona Visual custom-state runtime triggers
status: Done
assignee: []
created_date: '2026-05-15 03:17'
updated_date: '2026-05-15 05:19'
labels:
  - persona
  - buddy
  - visual-packs
  - frontend
dependencies: []
documentation:
  - >-
    Docs/superpowers/specs/2026-05-14-persona-buddy-default-catalog-state-catalog-extension-design.md
parent_task_id: TASK-347
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the frontend Stage 3 runtime/type slice for Persona Visual state catalogs. The buddy shell should consume active-pack authored exact tool_name triggers from structured live tool context and render declared custom visual states through the existing Persona Visual pack runtime, while preserving current built-in live-state and tool-category fallback behavior. This is scoped to the WebUI runtime/type path, not MCP direct trigger-state handling, generation jobs, or default starter-pack art production.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Frontend Persona Visual types represent built-in and custom state IDs plus tool_name authored trigger sources without dropping the known built-in state set.
- [x] #2 Buddy visual runtime resolves exact tool_name authored triggers from structured active tool context before falling back to tool_running.
- [x] #3 Buddy shell render context carries structured active tool identity separately from human-readable tool status.
- [x] #4 Custom state animations render through the existing SpriteFrameRenderer path for active visual packs.
- [x] #5 Focused tests cover exact tool_name matching, no text-only inference, BuddyShell rendering of a custom tool state, and existing category fallback behavior.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented frontend Persona Visual custom-state runtime support. Added custom visual state typing and state_catalog metadata, exact tool_name authored-trigger matching from structured activeToolName, active_tool_name render-context propagation from live voice, and custom state display/preservation in VisualPackEditor.

Verification: bun run test src/components/Common/PersonaBuddy/__tests__/personaVisualState.test.ts src/components/Common/PersonaBuddy/__tests__/BuddyShellHost.test.tsx src/hooks/__tests__/usePersonaLiveVoiceController.test.tsx passed with 81 tests. VisualPackEditor default timeout run exposed two slow existing tests; rerun with --testTimeout=20000 passed all 24 editor tests. git diff --check passed. Package tsc was attempted and exited nonzero due existing repo-wide test type errors outside this slice, so it is recorded as a known non-clean baseline gate rather than a pass. Bandit is not applicable because this slice touched frontend TypeScript and Backlog task files only.

Review follow-up started for PR #1717. Actionable comments found: branded custom-state ID typing, built-in-only runtime override type contract, stale generationTargetState clamp/submit guard, object-safe getPayloadToolName extraction, and explicit structured activeToolName guard for tool_name trigger matching.

PR #1717 review follow-up addressed: branded PersonaVisualCustomStateId no longer widens PersonaVisualStateId to plain string; runtime visual_state_override is built-in-only in store, resolver, and incoming payload handling; tool_name triggers now require a present structured activeToolName; live voice tool payloads extract object-shaped tool.name/function.name safely; VisualPackEditor clamps stale generation target states after pack switches and guards submit payloads; import conflict commit eligibility now waits for an explicit target-mode choice for the active preview.

Review follow-up verification: bun run test src/hooks/__tests__/usePersonaLiveVoiceController.test.tsx -t "extracts activeToolName" passed; bun run test src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx -t "clamps stale generation" --testTimeout=20000 passed; bun run test src/components/Common/PersonaBuddy/__tests__/personaVisualState.test.ts src/components/Common/PersonaBuddy/__tests__/BuddyShellHost.test.tsx src/hooks/__tests__/usePersonaLiveVoiceController.test.tsx src/routes/hooks/__tests__/usePersonaIncomingPayload.visuals.test.tsx src/store/__tests__/persona-visual-runtime.test.ts passed with 86 tests; bun run test src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx --testTimeout=30000 passed with 25 tests; git diff --check passed. ./node_modules/.bin/tsc --noEmit -p tsconfig.json --pretty false still exits nonzero on existing repo-wide type drift outside this slice; no changed persona visual files were reported in the emitted errors. Bandit remains not applicable for this frontend-only TypeScript/task-file slice.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Persona Visual custom-state runtime triggers for the WebUI. The buddy shell now carries structured active tool identity, resolves exact tool_name authored triggers from active visual packs, and can render custom state animations through the existing sprite-frame path. Frontend manifest types now include state_catalog/custom states and tool_name trigger sources, and the visual pack editor preserves/displays custom state mappings for imported or backend-validated packs.

PR #1717 review follow-up tightened the runtime visual trigger contract: custom state ids are truly branded, incoming runtime overrides are built-in-only for this V1 path, tool_name matching uses structured active tool identity, generation jobs cannot submit stale custom target_state values after pack switches, and import conflict commits require an explicit target-mode choice.
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
