---
id: TASK-233
title: Map Persona Live voice and wake recovery reasons
status: Done
assignee: []
created_date: '2026-05-10 16:11'
updated_date: '2026-05-10 16:29'
labels:
  - persona
  - buddy
  - live-voice
  - wake
  - recovery
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1510'
  - 'https://github.com/rmusser01/tldw_server/issues/1519'
documentation:
  - Docs/Reviews/PERSONA_BUDDY_CURRENT_STATE_AUDIT_2026_05_10.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the next Stage 1 Persona/Buddy reliability slice. Normalize existing Persona Live voice and wake degraded/recovering states into stable reason codes and user-facing recovery copy without adding new runtime capabilities.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Existing live voice and wake states map to stable reason codes and concise recovery copy.
- [x] #2 Unsupported browser, permission, detector unavailable, wake rejected, manual mode, TTS fallback, reconnect, and teardown states avoid misleading broken-state copy when manual controls remain available.
- [x] #3 Persona Live or Buddy diagnostics surfaces the mapped recovery copy through existing UI paths.
- [x] #4 Focused tests cover mapper behavior and at least one route or controller state.
- [x] #5 Implementation links back to GitHub issue #1519 and the Stage 0 audit.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Plan: Docs/superpowers/plans/2026-05-10-persona-live-wake-recovery-reasons.md

Implementation stages:
1. Add stable live voice and wake warning reason codes to the Persona Live controller.
2. Map those codes to recovery-oriented Buddy diagnostics copy.
3. Wire route diagnostics input and verify hook/card/diagnostics behavior.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented stable hook metadata fields warningReasonCode and wakeWarningReasonCode for existing Persona Live voice/wake warning paths.

Mapped reason-coded live voice and wake states in Buddy diagnostics so no-trigger/no-transcript/no-config cases do not read as broken when manual controls remain available.

Verification: ./node_modules/.bin/vitest run src/hooks/__tests__/usePersonaLiveVoiceController.test.tsx src/components/PersonaGarden/__tests__/personaBuddyDiagnostics.test.ts src/components/PersonaGarden/__tests__/LiveSessionPanel.test.tsx --maxWorkers=1 from apps/packages/ui passed: 3 files, 75 tests.

Verification: git diff --check passed.

Bandit: not applicable; this slice touches frontend TypeScript and Backlog/plan docs only, with no backend Python changes.

Known skips/blockers: none for this slice; Bandit remains not applicable because no backend Python changed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added stable Persona Live voice and wake recovery reason codes to the existing controller and surfaced them through Buddy diagnostics copy. The slice keeps runtime behavior unchanged while making unsupported browser, permission, rejected wake activation, manual mode, TTS fallback, disconnected, and non-broken fallback states explicit and test-covered.
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
