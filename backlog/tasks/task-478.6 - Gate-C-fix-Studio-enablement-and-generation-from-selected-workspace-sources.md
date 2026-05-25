---
id: TASK-478.6
title: 'Gate C: fix Studio enablement and generation from selected workspace sources'
status: To Do
labels:
- research-workspace
- uat
- gate-c
- studio
- frontend
priority: High
milestone: Research Workspace UAT Remediation
parent_task_id: TASK-478
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
User-visible failure: Studio remained disabled with copy saying "Select sources from the Sources pane to enable generation" even after sources were selected and status APIs reported selected sources.

User goal: generate workspace outputs such as summaries, briefs, comparisons, or reports from the selected research sources without guessing why the controls are disabled.

Scope:
- Connect Studio enablement to the canonical selected-source and readiness contracts.
- Clarify which Studio actions require FTS-ready, vector-ready, citation-ready, or summary-ready sources.
- Show precise disabled reasons: no selected sources, selected sources still indexing, no model selected, provider unavailable, unsupported source state, etc.
- Validate output generation with a configured provider and source evidence where applicable.
- Add tests for enabled, disabled, partially queryable, failed source, and missing model states.

Acceptance criteria:
- Studio enables when selected sources meet the action's readiness requirements.
- Disabled controls explain the exact missing prerequisite.
- Generated outputs are saved/rendered in the expected workspace location and survive reload if that is the product contract.
- Live CDP/Playwright validation covers at least one successful generation path and one disabled prerequisite path.

Depends on: TASK-478.1, TASK-478.3, TASK-478.4.
Blocks: final acceptance matrix.
Parallelization: can run in parallel with grounded RAG Q&A once Gate A/B blockers are resolved.
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
