# ACP Workspace And Sandbox Readiness Implementation Plan

## Stage 1: Scope And Contract
**Goal**: Land a narrow #1477 hardening slice for workspace validation and per-session environment propagation.
**Success Criteria**: The slice covers actionable workspace validation failures, workspace MCP/env dispatch injection, standard runner env forwarding, sandbox runner env merge behavior, and operator caveats.
**Tests**: Focused workspace helper, orchestration dispatch, and sandbox runner tests.
**Status**: Complete

## Stage 2: Red Tests
**Goal**: Add failing tests before implementation.
**Success Criteria**: Tests fail because workspace validation errors are plain strings, workspace env vars are not forwarded to ACP session creation, and sandbox sessions do not merge per-session env with configured agent env.
**Tests**: `python -m pytest tldw_Server_API/tests/Agent_Orchestration/test_workspace_api_helpers.py tldw_Server_API/tests/Agent_Orchestration/test_orchestration_api.py tldw_Server_API/tests/Agent_Client_Protocol/test_acp_sandbox_runner_client.py -q`
**Status**: Complete

## Stage 3: Backend Hardening
**Goal**: Implement structured error details and per-session environment propagation.
**Success Criteria**: `_validate_workspace_root()` returns stable actionable error payloads; orchestration dispatch passes workspace `env_vars` to ACP session creation; standard and sandbox ACP runners accept optional session env; sandbox env JSON preserves configured env and applies workspace env as additive per-session values.
**Tests**: Targeted red tests pass.
**Status**: Complete

## Stage 4: Documentation And Verification
**Goal**: Document the #1477 operator guidance and run relevant verification.
**Success Criteria**: ACP docs/readiness matrix describe workspace allowlist setup, env propagation, MCP injection, sandbox runtime caveats, and stable failure codes.
**Tests**: Focused pytest, relevant broader suites, Bandit, `git diff --check`.
**Status**: Complete
