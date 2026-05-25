---
id: TASK-478.4
title: 'Gate B: fix workspace source selection contract'
status: To Do
labels:
- research-workspace
- uat
- gate-b
- frontend
- selection
- rag
priority: High
milestone: Research Workspace UAT Remediation
parent_task_id: TASK-478
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
User-visible failure: individual source checkbox clicks did not reliably persist selection, while `Select all` did. Studio also disagreed with selected-source state by saying no sources were selected while status APIs reported selected sources.

User goal: choose exactly which sources participate in RAG, Studio outputs, export, and later extension/agent handoffs.

Scope:
- Define the canonical selected-source state: local UI only, backend persisted, or both with clear synchronization semantics.
- Fix individual checkbox interaction, bulk select/deselect, folder-filtered selection, and reload persistence according to that contract.
- Ensure selected-source counts, disabled states, RAG mode, and Studio enablement all read the same source-selection contract.
- Add keyboard-accessible selection behavior and tests for single, bulk, filtered, and reload paths.

Acceptance criteria:
- Individual checkbox selection persists and is reflected everywhere selected sources are consumed.
- `Select all` and filtered/bulk selection produce predictable counts and do not silently select hidden/unintended sources unless explicitly designed.
- Source selection survives normal reload/workspace switching if the chosen contract says it should.
- CDP/Playwright validation confirms the selected-source state drives RAG and Studio consistently.

Depends on: TASK-478.3 for final readiness/selection semantics; can start UI-state investigation earlier.
Blocks: TASK-478.5 and TASK-478.6.
Parallelization: can proceed in parallel with source acquisition/layout tasks after the status contract is settled.
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
