---
id: TASK-228
title: Add Persona/Buddy diagnostics surface
status: In Progress
assignee:
  - Codex
created_date: '2026-05-10 07:18'
updated_date: '2026-05-10 15:20'
labels:
  - persona
  - buddy
  - diagnostics
  - stage-1
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1510'
  - 'https://github.com/rmusser01/tldw_server/issues/1511'
documentation:
  - Docs/Reviews/PERSONA_BUDDY_CURRENT_STATE_AUDIT_2026_05_10.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the first Stage 1 Persona/Buddy reliability slice from the merged current-state audit. Add a narrow read-only diagnostics surface for the existing Persona Garden / Persona Live / Buddy shell runtime so degraded, unavailable, and recovery states are visible from one place. Keep the work centered on existing contracts and do not add new Persona capabilities, MCP tools, renderer behavior, native/background wake support, Persona Chat quality changes, or VN/CYOA runtime changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Diagnostics projection covers selected persona identity, profile load state, Buddy summary/dormant state, active visual pack load/render diagnostic, Persona Live websocket/session state, live voice/recovery state, wake armed/state/rejection reason, and available persona_visuals MCP readiness from existing client/runtime state.
- [ ] #2 Persona Garden or Live displays a compact diagnostics summary without blocking existing controls or changing normal healthy flows.
- [ ] #3 Diagnostics distinguish healthy, unavailable, degraded, and recovering states with actionable copy derived from existing reason codes and state inputs.
- [ ] #4 Broken or missing visual packs continue to fail open and expose the relevant visual diagnostic in the summary.
- [ ] #5 Focused tests cover diagnostic derivation and at least one route-level degraded state.
- [ ] #6 Docs or developer notes link back to the Stage 0 audit and GitHub issue #1511.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implementation plan saved at Docs/superpowers/plans/2026-05-10-persona-buddy-diagnostics-implementation-plan.md.

Stages:
1. Add pure diagnostics projector and focused unit coverage.
2. Add compact diagnostics panel using existing design-system state components.
3. Wire diagnostics into Persona Live without changing healthy controls or backend contracts.
4. Add Stage 1 audit note, run focused Vitest/diff checks, and update task closeout.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Planning started in isolated worktree .worktrees/persona-buddy-diagnostics on branch codex/persona-buddy-diagnostics. GitHub tracking: epic #1510, Stage 1 issue #1511.

Task 1 complete: added pure Persona/Buddy diagnostics projector and unit tests. Verification: bunx vitest run apps/packages/ui/src/components/PersonaGarden/__tests__/personaBuddyDiagnostics.test.ts passed (5 tests).

Task 2 complete: added PersonaBuddyDiagnosticsPanel using the existing StatePanel design-system component. Verification: bun run test src/components/PersonaGarden/__tests__/PersonaBuddyDiagnosticsPanel.test.tsx passed (2 tests) from apps/packages/ui.

Task 3 complete: wired diagnostics into Persona Live, added visual runtime diagnostics to the existing persona visual runtime store, and published BuddyShellHost visual diagnostics for route consumption. Verification: bun run test src/routes/__tests__/sidepanel-persona.test.tsx -t diagnostics passed; bun run test src/store/__tests__/persona-visual-runtime.test.ts passed; bun run test src/components/PersonaGarden/__tests__/LiveSessionPanel.test.tsx passed; bun run test src/components/Common/PersonaBuddy/__tests__/BuddyShellHost.test.tsx -t visual passed.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
