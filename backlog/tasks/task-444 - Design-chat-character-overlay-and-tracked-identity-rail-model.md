---
id: TASK-444
title: Design /chat character overlay and tracked identity rail model
status: Done
labels:
- design
- chat
- webui
- extension
- characters
- personas
priority: high
documentation:
- Docs/superpowers/specs/2026-05-22-chat-character-overlay-and-tracked-identity-design.md
- Docs/superpowers/plans/2026-05-22-chat-character-overlay-and-tracked-identity-implementation-plan.md
modified_files:
- Docs/superpowers/specs/2026-05-22-chat-character-overlay-and-tracked-identity-design.md
- Docs/superpowers/plans/2026-05-22-chat-character-overlay-and-tracked-identity-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reconcile the /chat character/persona overlay design against the current restored cockpit rails. Preserve the tracked identity versus assistant overlay state split, but supersede the original standalone side-rail control surface in favor of the existing runtime/context cockpit rails.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Spec records the 2026-05-30 reconciliation that tracked identity and assistantOverlay remain separate while a standalone CharacterControlRail is superseded.
- [x] #2 Implementation plan is marked historical and explicitly tells future workers not to create CharacterControlRail, CharacterControlRailSheet, or a new character-control coordinator panel.
- [x] #3 Current cockpit runtime/context rails are identified as the canonical assistant control surface, with existing guard/e2e proof references recorded.
- [x] #4 Verification and Bandit applicability are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Reconciled TASK-444 against current `origin/dev`. The durable architecture already exists in dev: `ChatAssistantOverlay` typing/normalization, backend `AssistantOverlaySettings` validation, effective-assistant-state resolver, assistant-overlay snapshot helpers, runtime/context rail assistant controls, and real-server proofs for runtime-rail character/persona select/clear/tracked start/reload.

The stale part was the original standalone `CharacterControlRail` / `CharacterControlRailSheet` direction. Updated the spec and implementation plan to prohibit reintroducing that rail and to treat the existing cockpit runtime/context rails as the canonical assistant control surface.

Verification:
- Inspected current source/tests with `rg` and `sed`.
- Confirmed `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-regression.guard.test.ts` asserts `CharacterControlRail` remains absent.
- Confirmed `apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts` asserts `character-control-rail` count `0` while proving runtime-rail character/persona select, clear, tracked start, and reload.
- Ran `rg` over the reconciled spec/plan to verify remaining `CharacterControlRail` mentions are warnings or historical notes rather than executable instructions.
- Bandit skipped because this reconciliation touched Markdown/Backlog only; no Python source changed.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
TASK-444 is reconciled and closed. The state split and overlay contract remain valid/current, but the standalone character rail work is superseded by the restored /chat cockpit runtime/context rails. Future assistant UX work should extend `PlaygroundRuntimeInspector`, `PlaygroundContextRail`, and the existing mobile cockpit tabs unless a new product decision explicitly reopens a separate panel.

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
