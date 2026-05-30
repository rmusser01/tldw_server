## Stage 1: CLI Contract
**Goal**: Define the `show-preset` CLI behavior for deterministic full preset inspection.
**Success Criteria**: Tests cover success output and unknown preset JSON errors.
**Tests**: Focused CLI tests in `test_gateway_cli_package.py`.
**Status**: Complete

## Stage 2: Minimal Implementation
**Goal**: Add the smallest CLI handler and preset lookup needed to satisfy the contract.
**Success Criteria**: `mcp-unified-gateway show-preset <id>` emits stable JSON containing preset id, version, and full profile data.
**Tests**: Focused CLI tests pass.
**Status**: Complete

## Stage 3: Verification
**Goal**: Validate the package boundary and touched scope.
**Success Criteria**: Focused MCP tests, lint, diff check, and Bandit touched-scope scan complete with results recorded in TASK-567.
**Tests**: `pytest`, `ruff`, `bandit`, and `git diff --check`.
**Status**: Complete
