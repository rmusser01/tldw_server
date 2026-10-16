# Workflow WebSocket rejection — TASK-13818

Investigate the pre-accept rejection in `test_ws_events_stream` on CI run 38867120384. The original log omitted the close code; a hosted rerun and prior isolated Python 3.11 probes passed. Do not infer a token, ownership, or timing defect without reproduction.

Scope: workflow WebSocket handler and its auth/database dependencies, MCP JWT construction/verification if implicated, workflow fixtures and connected security/event tests. Preserve explicit credentials, cookie protections, run ownership, and streaming behavior. Do not add retries or relax authentication to hide the failure.

ADR required: no for an anticipated local correctness or fixture repair preserving current security policy; reassess if investigation changes a durable security rule. ADR-059 governs task editing.

## Stage 1: Reproduce and trace
**Goal**: Identify the rejection boundary and demonstrate its cause against latest dev.
**Success Criteria**: Close code, auth-manager identity, isolated database and owner evidence distinguish auth, missing-run, and cross-owner rejection without exposing secrets.
**Tests**: Original workflow shard on Python 3.12; compare isolated and full collection/order; controlled boundary reproduction after evidence narrows the cause.
**Status**: Complete

## Stage 2: Minimal repair
**Goal**: Correct the confirmed source and add a deterministic behavior regression.
**Success Criteria**: Regression fails before repair and passes afterward; token validation and cross-owner refusal remain effective.
**Tests**: Focused regression, workflow event/auth tests, connected affected callers, lint and scoped Bandit.
**Status**: Complete

## Stage 3: Review and integrate
**Goal**: Independently review a new PR against remote dev and merge after checks pass.
**Success Criteria**: Exact-head review clear, human Change summary recorded, latest-dev integration and required checks pass.
**Tests**: Hosted affected shard and required gates; verify merged PR and remote dev ancestry.
**Status**: In Progress

Root evidence: concurrent run reads return another run/owner on CPython 3.12, and the actual WS endpoint falsely refuses its owner with 4403. Disable cached statements on the two reusable SQLite connections; retain independent reads, PostgreSQL, and all authorization rules. Upstream characterization: https://github.com/python/cpython/issues/118172.

Verification: regressions fail before repair and pass 5/5 afterward; connected suites 116/116 and final lint-clean rerun 64/64 pass. Linux actual database reproduction returns 72 incorrect identities in 1600 reads before repair and zero after. Ruff and scoped Bandit pass. Independent working-tree review is clear; exact-commit review and hosted integration remain pending.
