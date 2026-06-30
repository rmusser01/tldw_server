---
id: TASK-528.1
title: Add /knowledge QA deterministic state fixtures and baseline coverage
status: Done
labels:
- webui
- extension
- knowledge
- testing
- ux
priority: high
parent_task_id: TASK-528
documentation:
- Docs/superpowers/plans/2026-06-07-knowledge-qa-state-fixtures-and-baseline-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Build deterministic test coverage and state fixtures for the /knowledge Knowledge QA audit states before changing the UI. Cover backend offline, setup required, no indexed sources, no selected sources, ready search, results with citations, no results, settings drawer, export, and WebUI versus extension differences. Do not add flashcard behavior to /knowledge.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Deterministic fixtures or mocks exist for every audited /knowledge state.
- [ ] #2 Baseline coverage captures the current WebUI readiness/blank-state failure before remediation.
- [ ] #3 WebUI and extension routes can be tested without depending on a live backend.
- [ ] #4 Test naming and fixture data preserve the Knowledge QA-only scope and do not use flashcard terminology.
- [ ] #5 Verification commands are recorded in the task notes or plan.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-06-07-knowledge-qa-state-fixtures-and-baseline-plan.md.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed deterministic /knowledge Knowledge QA baseline coverage for TASK-528.1. Added a shared state fixture builder for audited Knowledge QA states, fixture-backed unit coverage for connection/layout states, a WebUI readiness timeout baseline test documenting the current no-recovery behavior, WebUI Playwright route-state coverage for ready search and cited results without a live backend, and extension Playwright route-state coverage for setup-required and connected-ready states. Added TLDW_E2E_SKIP_EXTENSION_BUILD to extension global setup so existing valid builds can be used for targeted e2e verification. Verification: Knowledge QA unit fixtures passed (18 tests), ServerReadinessGate passed (5 tests), WebUI route-state Playwright passed (2 tests), extension route-state Playwright passed (2 tests) when run outside sandbox due Chromium extension launch permissions. Bandit not applicable: no Python files touched.
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
