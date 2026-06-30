## Stage 1: Contract Boundary Tests
**Goal**: Prove the host external transport base reuses package-owned neutral contracts.
**Success Criteria**: Failing tests show `BrokeredExternalCredential` is missing from `mcp_unified.federation` and host/base dataclasses are not package identities.
**Tests**: Focused `test_runtime_package_boundary.py` contract tests.
**Status**: Complete

## Stage 2: Package Contract Extraction
**Goal**: Add the missing brokered credential contract to `mcp_unified.federation` and make the host base re-export package contracts.
**Success Criteria**: Host imports remain compatible while neutral dataclasses are package-owned.
**Tests**: Focused MCP Unified package-boundary and external transport tests.
**Status**: Complete

## Stage 3: Verification And PR
**Goal**: Record focused test, lint, security, and whitespace verification before opening the PR.
**Success Criteria**: Pytest, Ruff, Bandit, and `git diff --check` results are captured in Backlog and the PR.
**Tests**: Focused pytest suite plus Ruff and Bandit on touched files.
**Status**: Complete
