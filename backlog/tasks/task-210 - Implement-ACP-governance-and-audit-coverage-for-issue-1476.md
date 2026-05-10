---
id: TASK-210
title: Implement ACP governance and audit coverage for issue 1476
status: Done
assignee: []
created_date: '2026-05-10 01:43'
updated_date: '2026-05-10 02:00'
labels:
  - ACP
  - governance
  - audit
  - backend
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1476'
  - 'https://github.com/rmusser01/tldw_server/issues/1471'
documentation:
  - Docs/Development/ACP_Production_Readiness.md
  - Docs/Plans/2026-03-14-acp-runtime-policy-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the #1476 governance/RBAC/audit slice in the ACP productionization worktree. The first review found route-level TokenScopeGuard coverage already present on REST surfaces and write-scoped WebSocket auth, but durable audit coverage is incomplete for agent registration, session creation, and orchestration dispatch/review/triage. Add focused tests, sanitized audit metadata, and documentation that identifies policy authority and remaining caveats.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ACP permission authority and route authorization coverage are documented for #1476.
- [x] #2 Agent registration and session creation produce sanitized audit evidence.
- [x] #3 Orchestration dispatch, review, retry, and triage decisions produce sanitized audit evidence.
- [x] #4 Focused tests cover audit records without leaking prompt text, cwd, env, or reviewer feedback secrets.
- [x] #5 GitHub issue #1476 is updated with implementation status and verification evidence.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
## Stage 1: Governance/RBAC Baseline
**Goal**: Capture current ACP route authorization and policy-authority assumptions.
**Success Criteria**: #1476 doc lists REST TokenScopeGuard endpoints, WebSocket write-scope behavior, and MCP Hub/runtime-policy snapshot authority.
**Tests**: Documentation review plus existing authorization tests.
**Status**: Complete

## Stage 2: Session and Agent Audit Events
**Goal**: Add sanitized audit records for agent registration/management and ACP session creation.
**Success Criteria**: Events avoid cwd, env, command args, prompt text, and configured secrets while preserving actor/action/session/agent identifiers.
**Tests**: Focused ACP endpoint tests.
**Status**: Complete

## Stage 3: Orchestration Audit Events
**Goal**: Add sanitized audit records for dispatch, completion signal, reviewer decision, retry, complete, and triage transitions.
**Success Criteria**: Task/run/reviewer IDs and status/reason metadata reconstruct the control-plane path without raw task descriptions or reviewer feedback.
**Tests**: Focused orchestration dispatch/review tests.
**Status**: Complete

## Stage 4: Verification and Issue Evidence
**Goal**: Run focused tests, full relevant suites, Bandit, and update #1476.
**Success Criteria**: Verification is recorded in Backlog and GitHub issue comments.
**Tests**: ACP focused tests, Agent Orchestration suite, Bandit, git diff check.
**Status**: Complete
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added ACP governance/audit documentation that identifies MCP Hub/runtime-policy authority, TokenScopeGuard route coverage, WebSocket write/access checks, audit event inventory, and remaining caveats.
- Added sanitized ACP audit metadata handling for agent registration/updates/deregistration and session creation without storing cwd, env, command args, prompts, tokens, or MCP server payloads.
- Added orchestration audit events for dispatch start, task completion signal, reviewer start/decision, finalization, retry, and triage using identifiers, statuses, counts, reason codes, and presence booleans instead of task descriptions, completion summaries, or reviewer feedback text.
- Verification refreshed on 2026-05-10: focused #1476 gate `62 passed, 5 warnings`; `Agent_Orchestration` suite `148 passed, 5 warnings`; ACP WebSocket suite `33 passed, 5 warnings`; Bandit touched backend scope `0` findings; `git diff --check` clean.
- GitHub issue #1476 updated with implementation and verification evidence: https://github.com/rmusser01/tldw_server/issues/1476#issuecomment-4414191965
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented #1476 by documenting ACP governance authority and route authorization coverage, adding sanitized ACP endpoint and orchestration audit records, tightening session SSH WebSocket access checks, and adding regression coverage that confirms sensitive prompt/cwd/env/reviewer data is not persisted in audit metadata. Verification passed with the focused #1476 gate, full Agent Orchestration suite, ACP WebSocket suite, Bandit on touched backend code, and `git diff --check`; the only remaining caveat is future operational policy detail for audit retention/export.
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
