---
id: TASK-15639
title: Admit replay-safe Research Workspace Data Tables
status: Done
labels:
- research
- workspace
- api
dependencies:
- TASK-15638
references:
- Docs/ADR/101-research-workspace-data-tables.md
- Docs/Design/research-workspace-data-tables.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let Studio use the exact supported Server processing route and recover its canonical native table after an interrupted launch.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Capability checks the exact Server route and workspace authority before generation.
- [x] #2 Only selected, versioned Media is accepted, and retries recover one canonical table and generation job.
- [x] #3 The published table opens and exports through Data Tables and existing callers retain their behavior.
- [x] #4 Focused API, worker, persistence and OpenAPI checks cover the new contract.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implements Docs/ADR/101-research-workspace-data-tables.md. The Workspace API qualifies the selected Server-managed provider and model, checks selected association and Media revisions, and admits one native table and job for each operation. Native publication and export remain in Data Tables. The focused Server suite passed 175 tests with PostgreSQL required. Ruff, Bandit and both OpenAPI drift gates passed. A localhost Chatbook test with a model adapter fixture recovered a dropped successful response, published all three output kinds and exported their native CSVs. Workspace generation is gated when usage quotas are enabled.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Studio can recover an authenticated native Data Tables generation and its exact published revision after a lost response.
<!-- SECTION:FINAL_SUMMARY:END -->

## Implementation Plan

ADR required: yes
ADR path: Docs/ADR/101-research-workspace-data-tables.md
Reason: The new Workspace admission and provider route are durable cross-owner contracts.

Follow the implementation plan in Docs/Design/research-workspace-data-tables.md.

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
