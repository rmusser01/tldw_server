---
id: TASK-478.11
title: 'Gate D: repair first-run tour, onboarding copy, and state-specific guidance'
status: To Do
labels:
- research-workspace
- uat
- gate-d
- onboarding
- copy
- tour
priority: Medium
milestone: Research Workspace UAT Remediation
parent_task_id: TASK-478
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
User-visible failure/gap: the top tour button and Settings > Replay tour did not show a visible walkthrough overlay during UAT. Earlier design feedback also rejected a separate workspace trust bar, so guidance should be contextual and not add non-core banner clutter.

User goal: as a first-time NotebookLM migrant, understand what a workspace contains, what to do next, where data lives, and how processing failures recover without losing the core work surface.

Scope:
- Fix tour launch/replay behavior or remove dead controls if the tour is not ready.
- Add concise empty/loading/error/partial-success copy in context: add sources, processing/indexing, missing model, selected sources, Studio disabled, failed ingestion, retry.
- Preserve a dense research-oriented layout without extra persistent banners that compete with the core source/chat/Studio panes.
- Ensure local-first/privacy/data ownership messaging is present where users make relevant decisions, not as global clutter.
- Add tests or UI assertions for tour open/replay and key empty/error states.

Acceptance criteria:
- Tour/replay controls either open a visible, navigable tour or are not exposed.
- First-run and error-state copy tells the user the next action and system state without generic marketing language.
- No reintroduction of the rejected workspace trust bar or similar persistent banner clutter.
- CDP/Playwright validation covers first-run empty state, tour/replay, missing model, processing, and failed-source copy where available.

Depends on: should align final terms with TASK-478.3 and TASK-478.7.
Parallelization: can run in parallel with layout/source acquisition after terminology is agreed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
