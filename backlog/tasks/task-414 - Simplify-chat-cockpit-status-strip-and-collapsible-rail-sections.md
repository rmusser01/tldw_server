---
id: TASK-414
title: Simplify chat cockpit status strip and collapsible rail sections
status: Done
assignee: []
created_date: ''
updated_date: 2026-05-16 21:02
labels:
- chat
- webui
- cockpit
- ux
- frontend
dependencies: []
priority: high
references:
- https://github.com/rmusser01/tldw_server/pull/1801
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tighten the main /chat cockpit UI by reducing the existing status strip to critical state/action information and making each main chat sidechannel rail section collapsible in place. Scope is limited to the WebUI /chat cockpit rails and status strip; no bottom rail replacement, no mocked data, no extension sidepanel/sidebar work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Ready-state status strip avoids duplicating routine session, persistence, and context summary chips already visible in cockpit rails.
- [x] #2 Critical states remain visible in the status strip, including degraded chat-available warnings, streaming stop, missing model recovery, server-blocked state, loading-context state, and session failures.
- [x] #3 Each existing main /chat sidechannel rail section can collapse and expand its body content without removing the rail or moving controls to the bottom of the page.
- [x] #4 Rail section collapse controls are keyboard-accessible and expose aria-expanded/aria-controls semantics.
- [x] #5 Focused Vitest coverage proves the status strip and collapsible rail behavior.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented a shared PlaygroundRailSection primitive for in-place collapsible cockpit rail sections. Wired it into the main /chat context rail and runtime rail, and added a top-level collapse control to the composition preview box. Simplified PlaygroundStatusStrip so normal Ready/Degraded state no longer repeats routine mode, session, persistence, and context-summary chips already visible in the rails, while critical session/runtime recovery state remains visible.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the main /chat cockpit status/sidechannel polish slice and opened draft PR #1801. Focused Vitest coverage proves compact critical status behavior, rail-section collapse semantics, and adjacent cockpit accessibility/composition behavior. Verification run: bunx vitest run PlaygroundStatusStrip.first-slice, PlaygroundContextRail.first-slice, PlaygroundRuntimeInspector.first-slice, Playground.cockpit-maturity, Playground.cockpit-a11y, PlaygroundCompositionPreview; bun run verify:design-system-state; git diff --check. Bandit skipped because this touched frontend TypeScript/TSX and Backlog markdown only. Live browser verification was not run because no local /chat WebUI/API listener was discoverable at the common checked ports and /api/v1/health on 127.0.0.1:18001 was unavailable.
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
