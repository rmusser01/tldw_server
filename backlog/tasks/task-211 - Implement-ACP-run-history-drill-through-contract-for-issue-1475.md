---
id: TASK-211
title: Implement ACP run history drill-through contract for issue 1475
status: Done
assignee: []
created_date: '2026-05-10 02:01'
labels:
  - ACP
  - run-history
  - backend
  - frontend-contract
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1475'
  - 'https://github.com/rmusser01/tldw_server/issues/1471'
documentation:
  - Docs/Development/ACP_Production_Readiness.md
  - Docs/Development/Agent_Client_Protocol.md
  - Docs/Plans/IMPLEMENTATION_PLAN_acp_run_history_1475.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the #1475 run-history and session drill-through slice in the ACP productionization worktree. The current backend already exposes ACP session events, artifacts, diagnostics, audit, and aggregate session run history, but Agent Orchestration task detail only returns raw run rows. Add an additive structured contract so frontend Agent Tasks can trace project/task -> run -> ACP session -> events/audit/artifacts/diagnostics without hardcoding URL construction or parsing raw session messages.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Task detail run history exposes structured ACP session drill-through links and availability flags.
- [x] #2 Run history includes prompt/result/stop-reason/tool-call/artifact/failure diagnostic metadata when available from the linked ACP session.
- [x] #3 Failed orchestration runs include normalized diagnostic reason and session/audit pointers where available.
- [x] #4 Focused tests cover successful and failed run drill-through without requiring a live ACP runner.
- [x] #5 GitHub issue #1475 is updated with implementation status and verification evidence.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
## Stage 1: Contract Shape
**Goal**: Define the additive task-detail run-history fields and document the backend/frontend contract.
**Success Criteria**: Plan and docs identify links, availability flags, previews, counts, and diagnostics without changing existing run storage.
**Tests**: Contract-focused API tests.
**Status**: Complete

## Stage 2: Red Tests
**Goal**: Add focused tests for enriched task run detail on successful and failed linked ACP sessions.
**Success Criteria**: Tests fail because the enrichment fields are missing.
**Tests**: Targeted Agent Orchestration API tests.
**Status**: Complete

## Stage 3: Backend Enrichment
**Goal**: Reuse ACP session store/message helpers to enrich task runs with drill-through links, event/artifact/diagnostic counts, stop reason, tool-call count, and failure context.
**Success Criteria**: Existing raw run fields remain stable and new fields are additive.
**Tests**: Targeted tests pass.
**Status**: Complete

## Stage 4: Verification and Issue Evidence
**Goal**: Run focused tests, relevant ACP/orchestration suites, Bandit for touched backend Python, diff check, then update #1475.
**Success Criteria**: Backlog and GitHub issue include evidence.
**Tests**: Focused pytest, relevant suites, Bandit, git diff check.
**Status**: Complete
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added red/green tests for `GET /api/v1/agent-orchestration/tasks/{task_id}` run history enrichment covering a successful linked ACP session and a failed linked ACP session with normalized diagnostics.
- Added an additive run-entry contract with `session` drill-through links/availability, `history` event/audit/artifact/diagnostic/tool-call/stop-reason metadata, prompt/result previews, `failure_context`, and reviewer decision summaries where durable review rows can be matched.
- Reused existing ACP session store messages, session diagnostic normalization, and in-memory audit lookup; no new persistence table was added for this slice.
- Documented the frontend-facing contract in `Docs/Development/Agent_Client_Protocol.md` and updated the #1475 readiness row in `Docs/Development/ACP_Production_Readiness.md`.
- Verification refreshed on 2026-05-10: red run failed on missing `session`; targeted green run `2 passed, 5 warnings`; focused ACP/session/orchestration set `46 passed, 5 warnings`; full `Agent_Orchestration` suite `150 passed, 5 warnings`; Bandit touched backend scope `0` findings; `git diff --check` clean.
- GitHub issue #1475 updated with implementation and verification evidence: https://github.com/rmusser01/tldw_server/issues/1475#issuecomment-4414212170
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the #1475 backend/frontend-contract slice by enriching orchestration task run history with ACP session drill-through links, availability flags, prompt/result previews, stop reason, tool-call/artifact/diagnostic/audit counts, normalized failure context, and reviewer decision summaries where available. The implementation reuses existing ACP session, audit, artifact, and diagnostic surfaces instead of adding new storage, and documents the contract for the separate #1473 Agent Tasks/ACP Playground UX workstream. Verification passed with red/green focused tests, the focused ACP/session/orchestration set, full Agent Orchestration suite, Bandit, and `git diff --check`.
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
