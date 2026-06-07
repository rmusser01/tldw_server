---
id: TASK-528.2
title: Fix /knowledge WebUI readiness timeout and route-level recovery
status: Done
labels:
- webui
- knowledge
- ux
- accessibility
priority: high
parent_task_id: TASK-528
documentation:
- Docs/superpowers/plans/2026-06-07-knowledge-webui-readiness-recovery-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix the WebUI /knowledge blocker where backend readiness failure leads to a blank or visually missing main state. Ensure users get actionable Knowledge QA recovery instead of a dead end. Do not add flashcard behavior to /knowledge.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 WebUI /knowledge never shows a blank main area after backend health timeout.
- [ ] #2 The recovery state identifies backend readiness, setup, auth, or connectivity failure where possible.
- [ ] #3 Users can retry, open diagnostics, or navigate to setup/settings from the recovery state.
- [ ] #4 KnowledgeQA route-level recovery is reachable after global readiness timeout.
- [ ] #5 Automated coverage verifies stalled health and failed health states.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-06-07-knowledge-webui-readiness-recovery-plan.md.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed WebUI /knowledge readiness recovery. ServerReadinessGate now uses a per-attempt health timeout, preserves healthy-backend behavior, and renders an actionable backend readiness recovery panel after timeout while keeping route children mounted so Knowledge QA route-level recovery remains reachable. Recovery includes the health endpoint, waited duration, Retry, Health & diagnostics, and Server settings actions. Added failed-health and stalled-health unit coverage plus /knowledge Playwright recovery coverage. Verification: ServerReadinessGate Vitest passed (6 tests), KnowledgeQA.connection Vitest passed (7 tests), knowledge-readiness-recovery Playwright passed (2 tests), and the prior knowledge-qa-states Playwright baseline passed (2 tests). A stale existing Next dev server had to be terminated once so Playwright could load the updated bundle. Bandit not applicable: no Python files touched.
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
