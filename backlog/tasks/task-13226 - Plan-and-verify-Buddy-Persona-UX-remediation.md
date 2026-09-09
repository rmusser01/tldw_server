---
id: TASK-13226
title: Plan and verify Buddy Persona UX remediation
status: Done
assignee:
  - '@codex'
created_date: '2026-09-08 22:15'
updated_date: '2026-09-09 00:10'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address the approved Buddy and Persona usability findings in tldw_server and its shared WebUI/extension, preserving independent artwork, explicit attachment scope, workspace defaults, and work continuity.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every approved finding maps to an implemented change or verified existing behavior on latest dev.
- [x] #2 Focused automated tests, rendered desktop and compact walkthroughs, security checks, and a final review verify the combined behavior.
- [x] #3 Design, ownership decision, user documentation, and implementation evidence describe the final behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Read the existing server workspace-default and Persona runtime contracts, record the approved design and ADR, track independently testable implementation tasks, integrate and review the changes, then verify the full Buddy/Persona journey.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Completed Buddy/Persona remediation for tldw_server and the shared WebUI/extension on isolated origin/dev base6cd2745f69. No Flashcards changes. All reviewed findings map to implemented behavior or verified latest-dev equivalents in Docs/superpowers/plans/2026-09-08-buddy-persona-ux-remediation.md; existing workspace defaults and Ant Design status contrast were reused. Final shared UI437 tests/26 files and WebUI app-layout/networking40 tests passed, plus backend foundation34 tests including real PostgreSQL18 and targeted authenticated Chat/queue/ledger regressions. Rendered management and drawer mobile/theme matrices, exact target reply, real Next navigation preserving draft/selection/speech controls, Cancel/Escape/X focus and discard, one visible Buddy, and38 status contrast checks passed. Python security/static checks and final diff whitespace checks passed. Frontend typecheck retains81 unchanged errors outside modified files; no changed-file errors. ADR-005, design/plan, user guide/navigation, API/Operations and evidence lessons are updated. Limitations: fixture browser APIs and controlled backend provider responses; no real audio/model provider, packaged extension or full suite. Accepted Buddy work is process-owned with documented restart/no-replay and principal-affinity limits; legacy Persona Live remains connection-owned. No commit, PR, merge or publication performed.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
