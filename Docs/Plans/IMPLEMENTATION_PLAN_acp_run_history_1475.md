# ACP Run History Drill-Through Implementation Plan

## Stage 1: Contract Shape
**Goal**: Define an additive task-detail contract for run history drill-through.
**Success Criteria**: `GET /api/v1/agent-orchestration/tasks/{task_id}` keeps the existing run fields and adds structured `session`, `history`, `failure_context`, and `review_decision` blocks that frontend Agent Tasks can consume without building raw URLs.
**Tests**: Focused Agent Orchestration API tests assert the response shape for successful and failed linked ACP sessions.
**Status**: Complete

## Stage 2: Red Tests
**Goal**: Add failing tests for successful and failed run history enrichment.
**Success Criteria**: Tests fail because task run entries do not yet expose session links, prompt/result metadata, artifact/diagnostic counts, or normalized failure reason.
**Tests**: `python -m pytest tldw_Server_API/tests/Agent_Orchestration/test_orchestration_api.py::<test_name> -q`
**Status**: Complete

## Stage 3: Backend Enrichment
**Goal**: Reuse existing ACP session store, audit, artifact, and diagnostic helpers to enrich orchestration run entries.
**Success Criteria**: Linked ACP sessions expose detail/events/artifacts/diagnostics/audit URLs, availability flags, prompt/result previews, stop reason, tool-call count, artifact count, diagnostic count, audit count, and normalized failure context where available. Missing ACP sessions fail open with `available=false`.
**Tests**: Targeted Agent Orchestration API tests pass.
**Status**: Complete

## Stage 4: Documentation and Frontend Contract
**Goal**: Document the backend contract for the upcoming Agent Tasks and ACP Playground UI drill-through work.
**Success Criteria**: ACP development docs and readiness matrix describe how frontend code should trace task -> run -> ACP session -> session detail endpoints.
**Tests**: Documentation review plus focused backend tests.
**Status**: Complete

## Stage 5: Verification and Issue Evidence
**Goal**: Run focused tests, relevant ACP/orchestration suites, Bandit for touched backend code, and update issue #1475.
**Success Criteria**: Backlog task and GitHub issue contain implementation notes, verification results, and remaining caveats.
**Tests**: Focused pytest, Agent Orchestration suite, relevant ACP session tests, Bandit, `git diff --check`.
**Status**: Complete
