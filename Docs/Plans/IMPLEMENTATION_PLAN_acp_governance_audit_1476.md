# ACP Governance And Audit Implementation Plan (#1476)

## Stage 1: Governance/RBAC Baseline
**Goal**: Capture current ACP route authorization and policy-authority assumptions.
**Success Criteria**: #1476 documentation lists REST `TokenScopeGuard` endpoints, WebSocket write-scope behavior, and MCP Hub/runtime-policy snapshot authority.
**Tests**: Documentation review plus existing authorization tests.
**Status**: In Progress

## Stage 2: Session And Agent Audit Events
**Goal**: Add sanitized audit records for agent registration/management and ACP session creation.
**Success Criteria**: Events avoid cwd, env, command args, prompt text, and configured secrets while preserving actor/action/session/agent identifiers.
**Tests**: Focused ACP endpoint tests.
**Status**: Not Started

## Stage 3: Orchestration Audit Events
**Goal**: Add sanitized audit records for dispatch, completion signal, reviewer decision, retry, complete, and triage transitions.
**Success Criteria**: Task/run/reviewer IDs and status/reason metadata reconstruct the control-plane path without raw task descriptions or reviewer feedback.
**Tests**: Focused orchestration dispatch/review tests.
**Status**: Not Started

## Stage 4: Verification And Issue Evidence
**Goal**: Run focused tests, full relevant suites, Bandit, and update #1476.
**Success Criteria**: Verification is recorded in Backlog and GitHub issue comments.
**Tests**: ACP focused tests, Agent Orchestration suite, Bandit, `git diff --check`.
**Status**: Not Started
