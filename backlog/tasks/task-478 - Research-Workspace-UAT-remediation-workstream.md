---
id: TASK-478
title: Research Workspace UAT remediation workstream
status: Done
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
- [x] All Research Workspace UAT child tasks through TASK-478.31 are closed or explicitly split out.
- [x] Post-rebase frontend Research Workspace test and TypeScript gates are verified.
- [x] Post-rebase backend workspace/status/migration/capability/sandbox/embeddings regression gates are verified.
- [x] Remaining fixture-backed risks were validated, resolved, or tracked in the child task records without overclaiming live coverage.
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
- Post-original-plan follow-ups are tracked as TASK-478.14 through TASK-478.18. These cover backend startup restore, parity workflow naming, smoke metadata governance, model metadata allowlist ownership, and the migration true-move matrix refresh.
- TASK-515 and TASK-516 moved migration true-move deletion from server-ineligible/retained to live-verified delete eligibility plus durable tombstone suppression. TASK-478.18 refreshed RW-UAT-025 with that evidence, and TASK-478.25 later closed the guided migration import/export recovery walkthrough with live WebUI evidence.
- TASK-478.19, TASK-478.21, TASK-478.22, TASK-478.23, TASK-478.24, and TASK-478.25 closed the keyboard, MCP handoff, ACP bridge, sandbox diagnostics, sandbox execution-contract, and migration import/export recovery slices as far as live evidence supports.
- TASK-478.26 reconciled the remaining risks into fixture-backed follow-ups rather than overclaiming the matrix. TASK-478.27 through TASK-478.31 then closed the MCP workspace-set policy/tool execution, ACP workspace-scoped run diagnostics, sandbox enabled-runtime workspace runs, long-running vector completion with real embeddings, and frontend TypeScript baseline verification blockers.
- Parent closeout was rebased onto origin/dev and verified against the Research Workspace UI/store/API gate, the shared UI TypeScript gate, and the focused backend workspace/status/migration/capability/sandbox/embeddings regression gate. Workspace endpoint tests that only need overridden endpoint dependencies now use minimal router fixtures instead of the production FastAPI lifespan, avoiding unrelated startup worker shutdown hangs during focused endpoint verification.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Closed the Research Workspace UAT remediation workstream. All known TASK-478 child slices through TASK-478.31 are Done, including the Shared Workspaces/MCP/ACP/sandbox model alignment, ingestion/indexing status, extension handoff, live UAT matrix maintenance, fixture-backed MCP/ACP/sandbox/vector validation, and the frontend TypeScript blocker.

Post-rebase verification recorded:
- `bunx vitest run src/components/Option/ResearchWorkspace/__tests__ src/store/__tests__/workspace.test.ts src/store/__tests__/workspace-api-first.test.ts src/store/__tests__/workspace-migration.test.ts src/store/__tests__/research-workspace-legacy-storage-inventory.test.ts src/services/tldw/domains/__tests__/workspace-api.status-capabilities.test.ts --reporter=dot` -> 56 files, 523 tests passed.
- `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false` -> passed.
- `python -m pytest tldw_Server_API/tests/Workspaces/test_workspaces_api.py tldw_Server_API/tests/Workspaces/test_workspace_source_status_api.py tldw_Server_API/tests/Workspaces/test_workspace_source_preview_context_api.py tldw_Server_API/tests/Workspaces/test_workspace_migration_api.py tldw_Server_API/tests/Research_Workspace/test_capability_endpoint.py tldw_Server_API/tests/Research_Workspace/test_capability_derivation.py tldw_Server_API/tests/sandbox/test_workspace_diagnostics.py tldw_Server_API/tests/Embeddings/test_backpressure_and_quotas.py tldw_Server_API/tests/Embeddings/test_embeddings_redis_worker.py tldw_Server_API/tests/Embeddings/test_embeddings_jobs_worker.py -q` -> 111 tests passed.
- `python -m bandit -r <touched workspace API test files> -s B101 -f json` -> 0 results, 0 errors. The initial unskipped Bandit run only reported pytest `assert` usage.
- `git diff --check` -> passed.
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
