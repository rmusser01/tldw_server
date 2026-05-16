---
id: TASK-232
title: 'Address PR #1518 Persona/Buddy diagnostics review comments'
status: Done
assignee: []
created_date: '2026-05-10 15:49'
labels:
  - persona
  - buddy
  - diagnostics
  - review-fix
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1518'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve actionable PR #1518 review feedback for the Persona/Buddy diagnostics surface. Scope is limited to projector output correctness and review-thread cleanup for Stage 1 diagnostics.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Dormant Buddy summary is represented as non-degraded when the selected persona intentionally lacks a Buddy summary.
- [x] #2 Unknown persona_visuals MCP readiness is represented as degraded or unconfirmed instead of healthy.
- [x] #3 Core Persona server capability reports ready only when hasPersona is explicitly true.
- [x] #4 Live session diagnostics include the last relevant event or reason code when present.
- [x] #5 Visual pack diagnostics surface the active pack id even when a pack title is available.
- [x] #6 Focused tests and diff checks pass.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Resolved PR #1518 review feedback by keeping intentionally dormant Buddy summaries healthy, treating unknown persona_visuals MCP readiness as degraded, requiring hasPersona to be explicitly true before reporting Persona capability ready, surfacing live session lastEvent in the session row, and preserving active visual pack ids when titles are present.

Verification passed from apps/packages/ui:
- bun run test src/components/PersonaGarden/__tests__/personaBuddyDiagnostics.test.ts
- bun run test src/routes/__tests__/sidepanel-persona.test.tsx -t diagnostics
- bun run test src/components/PersonaGarden/__tests__/personaBuddyDiagnostics.test.ts src/components/PersonaGarden/__tests__/PersonaBuddyDiagnosticsPanel.test.tsx src/components/Common/PersonaBuddy/__tests__/personaVisualDiagnostics.test.ts src/store/__tests__/persona-visual-runtime.test.ts src/components/PersonaGarden/__tests__/LiveSessionPanel.test.tsx

Bandit skipped because the touched scope is TypeScript and Backlog Markdown only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed all actionable PR #1518 review comments by tightening Persona/Buddy diagnostics state derivation and adding regression coverage for the reviewed cases. The patch keeps the diagnostics read-only and frontend-only.
<!-- SECTION:FINAL_SUMMARY:END -->
