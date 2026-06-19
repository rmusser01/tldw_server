# ACP Live Backend Browser E2E Closeout Plan

## Stage 1: Establish Live Backend Baseline
**Goal**: Start the API with a reproducible single-user test configuration and confirm ACP health endpoints respond with the seeded API key.
**Success Criteria**: `/api/v1/health` and `/api/v1/acp/sessions` return successful responses against the live server.
**Tests**: Local backend startup plus API smoke checks.
**Status**: Complete

## Stage 2: Reproduce Browser E2E Results
**Goal**: Run the Tier 3 ACP browser specs against the live backend and record exact pass, fail, and skip behavior.
**Success Criteria**: ACP Playground, Agent Registry, and Agent Tasks results are attributable to concrete server or runtime behavior.
**Tests**: Playwright ACP Tier 3 specs with `TLDW_E2E_SERVER_URL` and `TLDW_E2E_API_KEY`.
**Status**: Complete

## Stage 3: Resolve E2E Setup Gap
**Goal**: Use the product's bundled ACP runner home so the server and runner share the expected downstream agent registry.
**Success Criteria**: The Research Workspace ACP history test creates a diagnostics-linked live ACP run instead of skipping on `unknown agent type`.
**Tests**: Focused Playwright rerun for `binds a Research Workspace to a real ACP run history and diagnostics path`.
**Status**: Complete

## Stage 4: Verify and Publish Evidence
**Goal**: Clean local artifacts, run relevant verification, and update the Backlog and GitHub tracker with command/output evidence.
**Success Criteria**: `TASK-2388`, `#2404`, and parent `#2398` show the final state and exact verification command.
**Tests**: Final Playwright result, non-code Bandit rationale, and git status review.
**Status**: Complete
