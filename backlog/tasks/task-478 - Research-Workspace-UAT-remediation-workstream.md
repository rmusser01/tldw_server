---
id: TASK-478
title: Research Workspace UAT remediation workstream
status: To Do
labels:
- research-workspace
- uat
- workstream
priority: High
milestone: Research Workspace UAT Remediation
references:
- /private/tmp/tldw-uat-01-empty-workspace.png
- /private/tmp/tldw-uat-02-model-selector-broken.png
- /private/tmp/tldw-uat-03-chat-null-assistant.png
- /private/tmp/tldw-uat-04-source-ready-status-mismatch.png
- /private/tmp/tldw-uat-05-mobile-overflow.png
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Parent tracking task for the live backend/WebUI UAT remediation pass on /research-workspace. Scope covers the validated failures from the acceptance walkthrough: model catalog loading, send-time guardrails, ingestion/indexing readiness, RAG source selection, grounded Q&A, Studio enablement, source acquisition/search, source inspection, responsive layout, onboarding/tour behavior, and final live UAT regression coverage.

Execution rule: resolve child tasks in dependency order by gate. Do not move to a dependent gate until the blocking tasks are verified against a live backend and WebUI using CDP/Playwright, not Computer Control.

Dependency gates:
A. Model/provider selection and send integrity.
B. Workspace source readiness and selection contract.
C. Grounded RAG and Studio end-to-end behavior.
D. Source-management UX, inspection, layout, and onboarding polish.
E. Final acceptance matrix across backend, WebUI, and extension handoff when extension build is available.

Parallelization model: tasks inside a later gate may be investigated in parallel, but implementation should not land behavior that depends on an unresolved earlier contract.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Sequential workstream:
1. TASK-478.1 - Gate A: restore Research Workspace model catalog loading.
2. TASK-478.2 - Gate A: harden chat send and failed-response states.
3. TASK-478.7 - Cross-cutting: align Research Workspace with Shared Workspaces, MCP, ACP, and sandbox model. Start early; resolve before final API/UI contracts are frozen.
4. TASK-478.3 - Gate B: implement first-class workspace ingestion and indexing status.
5. TASK-478.4 - Gate B: fix workspace source selection contract.
6. TASK-478.5 - Gate C: validate grounded RAG Q&A end to end.
7. TASK-478.6 - Gate C: fix Studio enablement and generation from selected workspace sources.
8. TASK-478.8 - Gate D: harden source acquisition, URL validation, and My Media search.
9. TASK-478.9 - Gate D: improve source preview, annotations, and evidence inspection.
10. TASK-478.10 - Gate D: fix layout, responsive behavior, and keyboard accessibility.
11. TASK-478.11 - Gate D: repair first-run tour, onboarding copy, and state-specific guidance.
12. TASK-478.12 - Gate E: validate browser extension handoff into canonical workspaces.
13. TASK-478.13 - Gate E: maintain live UAT matrix and regression coverage.

Parallelization lanes:
- Lane 1 critical frontend: TASK-478.1 -> TASK-478.2.
- Lane 2 backend/model contract: TASK-478.7 and TASK-478.3 can begin while Lane 1 runs.
- Lane 3 source-selection/UI contract: TASK-478.4 starts once TASK-478.3 semantics are stable.
- Lane 4 end-to-end features: TASK-478.5 and TASK-478.6 run in parallel after Gate A/B blockers are resolved.
- Lane 5 polish and source management: TASK-478.8, TASK-478.9, TASK-478.10, and TASK-478.11 can run in parallel after terminology/readiness contracts are settled.
- Lane 6 validation: TASK-478.13 starts immediately as the matrix scaffold, then closes last; TASK-478.12 waits for extension build availability plus canonical workspace/status contracts.

Operating rule: do not close or move past a gate without live backend + WebUI validation through CDP/Playwright and recorded verification notes in the child task.
<!-- SECTION:PLAN:END -->

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
